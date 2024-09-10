#include <iostream>
#include <string>
#include <fstream>
#include <cstring>
#include <thread>
#include <chrono>
#include <queue>
#include <mutex>
#include <unordered_set>
#include <condition_variable>
#include <unordered_map>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
extern "C" {
    #include <libavutil/hwcontext.h>
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/opt.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
}
#include "config.h"

struct PacketHeader {
    uint64_t timestamp;
    uint32_t sequence_number;
};

struct PacketInfo {
    PacketHeader header;
    std::vector<uint8_t> data;
    size_t size;
};

class PacketQueue {
public:
    void push(const PacketInfo& packet) {
        std::lock_guard<std::mutex> lock(mtx);
        packet_queue.push(packet);
        cv.notify_one();
    }

    bool wait_and_pop(PacketInfo& packet) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return !packet_queue.empty(); });
        packet = packet_queue.front();
        packet_queue.pop();
        return true;
    }

private:
    std::queue<PacketInfo> packet_queue;
    std::mutex mtx;
    std::condition_variable cv;
};

void log_packet_info(const std::string& source_ip, uint32_t sequence_number, double latency, const std::string& mode, 
                     uint64_t receive_time, size_t packet_size, std::ofstream& log_file) {
    if (log_file.is_open()) {
        log_file << source_ip << "," << sequence_number << "," << latency << "," << mode << "," << receive_time << "," << packet_size << "\n";
    }
}

// GPU 디코딩 초기화 함수
AVBufferRef* init_hw_device(AVHWDeviceType hw_type) {
    AVBufferRef* hw_device_ctx = nullptr;
    if (av_hwdevice_ctx_create(&hw_device_ctx, hw_type, nullptr, nullptr, 0) < 0) {
        std::cerr << "Failed to create hardware device context." << std::endl;
        return nullptr;
    }
    return hw_device_ctx;
}

// 디코더 초기화 함수 (CUDA 기반 또는 소프트웨어 디코더)
AVCodecContext* init_hw_decoder(AVHWDeviceType hw_type) {
    const AVCodec* codec;
    codec = avcodec_find_decoder_by_name("h264_cuvid");  // CUDA 기반 H.264 디코더

    if (!codec) {
        std::cerr << "Decoder not found!" << std::endl;
        return nullptr;
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        std::cerr << "Failed to allocate codec context!" << std::endl;
        return nullptr;
    }
    codec_ctx->hw_device_ctx = init_hw_device(hw_type);
    if (!codec_ctx->hw_device_ctx) {
        avcodec_free_context(&codec_ctx);
        return nullptr;
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Failed to open codec!" << std::endl;
        avcodec_free_context(&codec_ctx);
        return nullptr;
    }

    return codec_ctx;
}

void decode_and_save_frame(AVCodecContext* codec_ctx, const uint8_t* data, size_t size, const std::string& frame_dir, uint64_t timestamp) {
    AVPacket* av_packet = av_packet_alloc();
    if (!av_packet) {
        std::cerr << "Error allocating AVPacket." << std::endl;
        return;
    }
    av_packet->data = const_cast<uint8_t*>(data);
    av_packet->size = size;

    av_packet->pts = timestamp;
    av_packet->dts = timestamp;

    if (avcodec_send_packet(codec_ctx, av_packet) < 0) {
        std::cerr << "Error sending packet to decoder" << std::endl;
        av_packet_free(&av_packet);
        return;
    }

    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        std::cerr << "Failed to allocate AVFrame" << std::endl;
        av_packet_free(&av_packet);
        return;
    }

    while (avcodec_receive_frame(codec_ctx, frame) == 0) {
        if (frame->format == AV_PIX_FMT_CUDA) {
            std::cerr << "Unexpected GPU frame, should transfer to CPU." << std::endl;
            av_frame_free(&frame);
            av_packet_free(&av_packet);
            return;
        }

        SwsContext* sws_ctx = sws_getContext(
            frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
            frame->width, frame->height, AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr, nullptr);

        if (!sws_ctx) {
            std::cerr << "Failed to initialize sws context" << std::endl;
            av_frame_free(&frame);
            av_packet_free(&av_packet);
            return;
        }

        uint8_t* dst_data[4];
        int dst_linesize[4];
        av_image_alloc(dst_data, dst_linesize, frame->width, frame->height, AV_PIX_FMT_BGR24, 1);

        sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, dst_data, dst_linesize);

        cv::Mat img(frame->height, frame->width, CV_8UC3, dst_data[0]);

        std::ostringstream filename;
        filename << frame_dir << "/frame_" << timestamp << ".png";

        bool result = cv::imwrite(filename.str(), img);
        if (!result) {
            std::cerr << "Failed to save image to " << filename.str() << std::endl;
        }

        av_freep(&dst_data[0]);
        sws_freeContext(sws_ctx);
    }

    av_frame_free(&frame);
    av_packet_free(&av_packet);
}

class PacketProcessor {
public:
    PacketProcessor(uint64_t play_delay_ms, const std::string& frame_dir)
        : play_delay_ms(play_delay_ms), frame_dir(frame_dir) {
        codec_ctx = init_hw_decoder(AV_HWDEVICE_TYPE_CUDA);
        if (!codec_ctx) {
            throw std::runtime_error("Decoder initialization failed");
        }
    }

    ~PacketProcessor() {
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
        }
    }

    void process_packet(const PacketInfo& packet) {
        uint64_t current_time_us = get_current_time_us();  // 현재 시간 (마이크로초 단위)
        uint64_t packet_send_time_us = packet.header.timestamp;  // 패킷에 기록된 송신 타임스탬프
        uint64_t playback_time_us = packet_send_time_us + (play_delay_ms - 1) * 1000;  // 송신 타임스탬프에서 50ms 지연 후 재생해야 할 시점

        // 수신된 시간과 50ms 지연된 재생 시점을 비교하여 너무 늦게 도착한 경우 무시
        if (current_time_us - packet_send_time_us > (play_delay_ms + 1) * 1000) {
            // 패킷이 송신된 후 60ms 이상 지나서 도착한 경우, 재생하지 않고 무시
            std::cerr << "Skipping packet: Sequence Number = " << packet.header.sequence_number
                    << ", Latency = " << (current_time_us - packet_send_time_us) / 1000.0 << " ms" << std::endl;
            return;
        }

        // 아직 50ms 재생 시점이 지나지 않았다면, 재생 시점까지 대기
        if (playback_time_us > current_time_us) {
            std::this_thread::sleep_for(std::chrono::microseconds(playback_time_us - current_time_us));
        }

        // 재생할 준비가 되었으면 패킷을 디코딩하고 프레임을 저장
        decode_and_save_frame(codec_ctx, packet.data.data(), packet.size, frame_dir, packet.header.timestamp);
    }


    void process(PacketQueue& packet_queue) {
        while (true) {
            PacketInfo packet;
            if (packet_queue.wait_and_pop(packet)) {
                process_packet(packet);
            }
        }
    }

private:
    AVCodecContext* codec_ctx;
    uint64_t play_delay_ms;
    std::string frame_dir;

    uint64_t get_current_time_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};

class PacketReceiver {
public:
    PacketReceiver(int port, const std::string& mode, PacketQueue& packet_queue)
        : port(port), mode(mode), packet_queue(packet_queue) {
        std::ostringstream log_filename;
        log_filename << FILEPATH_LOG << "/" << mode << "_log.csv";
        log_file.open(log_filename.str(), std::ios::out);
        if (!log_file.is_open()) {
            throw std::runtime_error("Failed to open log file");
        }
    }

    ~PacketReceiver() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }

    void receive_packets() {
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) {
            std::cerr << "Socket creation failed" << std::endl;
            return;
        }

        sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
        server_addr.sin_port = htons(port);

        if (bind(sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Socket bind failed" << std::endl;
            close(sock);
            return;
        }

        char buffer[BUFFER_SIZE];
        sockaddr_in sender_addr;
        socklen_t addr_len = sizeof(sender_addr);
        ssize_t recv_len;

        while ((recv_len = recvfrom(sock, buffer, BUFFER_SIZE, 0, (sockaddr*)&sender_addr, &addr_len)) > 0) {
            PacketHeader* header = reinterpret_cast<PacketHeader*>(buffer);
            auto receive_time = std::chrono::system_clock::now();
            uint64_t receive_time_us = std::chrono::duration_cast<std::chrono::microseconds>(receive_time.time_since_epoch()).count();
            double latency = (receive_time_us - header->timestamp) / 1000.0;
            std::string sender_ip = inet_ntoa(sender_addr.sin_addr);

            log_packet_info(sender_ip, header->sequence_number, latency, mode, receive_time_us, recv_len, log_file);

            PacketInfo packet_info;
            packet_info.header = *header;
            packet_info.data.assign(buffer + sizeof(PacketHeader), buffer + recv_len);
            packet_info.size = recv_len - sizeof(PacketHeader);

            packet_queue.push(packet_info);
        }

        close(sock);
    }

private:
    int port;
    std::string mode;
    PacketQueue& packet_queue;
    std::ofstream log_file;
};

class CombineMode {
public:
    CombineMode(uint64_t play_delay_ms)
        : processor(play_delay_ms, std::string(FILEPATH_FRAME) + "/combine") {
        combine_log_file.open(std::string(FILEPATH_LOG) + "/combine_log.csv", std::ios::out);
        if (!combine_log_file.is_open()) {
            throw std::runtime_error("Failed to open combine log file");
        }
        combine_log_file << "Source_IP,Sequence_Number,Latency,Mode,Receive_Time,Packet_Size\n";
    }

    ~CombineMode() {
        if (combine_log_file.is_open()) {
            combine_log_file.close();
        }
    }

    void start() {
        int sockfd1 = create_socket(SERVER_PORT1);
        int sockfd2 = create_socket(SERVER_PORT2);

        std::thread receiver_thread([&]() { process_combined_packets(sockfd1, sockfd2); });
        std::thread processor_thread([&]() { packet_processor_thread(); });

        receiver_thread.join();
        processor_thread.join();
    }

private:
    PacketProcessor processor;
    std::ofstream combine_log_file;
    PacketQueue packet_queue;
    std::mutex packet_mutex;
    std::unordered_map<uint32_t, uint64_t> packet_arrival_time;
    std::unordered_set<uint32_t> processed_sequences;

    int create_socket(int port) {
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) {
            std::cerr << "Socket creation failed" << std::endl;
            throw std::runtime_error("Socket creation failed");
        }

        sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
        server_addr.sin_port = htons(port);

        if (bind(sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Socket bind failed" << std::endl;
            close(sock);
            throw std::runtime_error("Socket bind failed");
        }

        return sock;
    }

    void process_combined_packets(int sockfd1, int sockfd2) {
        fd_set read_fds;
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);

        std::vector<uint8_t> buffer1(BUFFER_SIZE), buffer2(BUFFER_SIZE);

        while (true) {
            FD_ZERO(&read_fds);
            FD_SET(sockfd1, &read_fds);
            FD_SET(sockfd2, &read_fds);

            int max_sd = std::max(sockfd1, sockfd2);
            int activity = select(max_sd + 1, &read_fds, nullptr, nullptr, nullptr);

            if (activity > 0) {
                bool is_first_arrival = false;
                PacketHeader header;
                std::string source_ip;
                uint64_t receive_time;
                double latency;

                if (FD_ISSET(sockfd1, &read_fds)) {
                    int len1 = recvfrom(sockfd1, buffer1.data(), BUFFER_SIZE, 0, (struct sockaddr*)&client_addr, &addr_len);
                    if (len1 > 0) {
                        memcpy(&header, buffer1.data(), sizeof(PacketHeader));
                        source_ip = inet_ntoa(client_addr.sin_addr);
                        receive_time = get_current_time_us();
                        latency = (receive_time - header.timestamp) / 1000.0;

                        {
                            std::lock_guard<std::mutex> lock(packet_mutex);
                            if (processed_sequences.count(header.sequence_number) == 0) {
                                processed_sequences.insert(header.sequence_number);
                                packet_arrival_time[header.sequence_number] = receive_time;
                                is_first_arrival = true;
                            } else if (receive_time < packet_arrival_time[header.sequence_number]) {
                                packet_arrival_time[header.sequence_number] = receive_time;
                                is_first_arrival = true;
                            }
                        }

                        if (is_first_arrival) {
                            log_packet_info(source_ip, header.sequence_number, latency, "Combined_Stream", receive_time, len1, combine_log_file);
                            packet_queue.push({header, std::vector<uint8_t>(buffer1.begin() + sizeof(PacketHeader), buffer1.end()), static_cast<size_t>(len1 - sizeof(PacketHeader))});
                        }
                    }
                }

                if (FD_ISSET(sockfd2, &read_fds)) {
                    int len2 = recvfrom(sockfd2, buffer2.data(), BUFFER_SIZE, 0, (struct sockaddr*)&client_addr, &addr_len);
                    if (len2 > 0) {
                        memcpy(&header, buffer2.data(), sizeof(PacketHeader));
                        source_ip = inet_ntoa(client_addr.sin_addr);
                        receive_time = get_current_time_us();
                        latency = (receive_time - header.timestamp) / 1000.0;

                        {
                            std::lock_guard<std::mutex> lock(packet_mutex);
                            if (processed_sequences.count(header.sequence_number) == 0) {
                                processed_sequences.insert(header.sequence_number);
                                packet_arrival_time[header.sequence_number] = receive_time;
                                is_first_arrival = true;
                            } else if (receive_time < packet_arrival_time[header.sequence_number]) {
                                packet_arrival_time[header.sequence_number] = receive_time;
                                is_first_arrival = true;
                            }
                        }

                        if (is_first_arrival) {
                            log_packet_info(source_ip, header.sequence_number, latency, "Combined_Stream", receive_time, len2, combine_log_file);
                            packet_queue.push({header, std::vector<uint8_t>(buffer2.begin() + sizeof(PacketHeader), buffer2.end()), static_cast<size_t>(len2 - sizeof(PacketHeader))});
                        }
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void packet_processor_thread() {
        processor.process(packet_queue);
    }

    uint64_t get_current_time_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mode: lg | kt | combine>" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::string frame_dir;

    if (mode == "lg") {
        frame_dir = std::string(FILEPATH_FRAME) + "/lg";
        PacketQueue packet_queue;
        PacketProcessor processor(PLAY_DELAY_MS, frame_dir);
        PacketReceiver receiver(SERVER_PORT1, "lg", packet_queue);
        std::thread receiver_thread(&PacketReceiver::receive_packets, &receiver);
        std::thread processor_thread(&PacketProcessor::process, &processor, std::ref(packet_queue));
        receiver_thread.join();
        processor_thread.join();
    } else if (mode == "kt") {
        frame_dir = std::string(FILEPATH_FRAME) + "/kt";
        PacketQueue packet_queue;
        PacketProcessor processor(PLAY_DELAY_MS, frame_dir);
        PacketReceiver receiver(SERVER_PORT2, "kt", packet_queue);
        std::thread receiver_thread(&PacketReceiver::receive_packets, &receiver);
        std::thread processor_thread(&PacketProcessor::process, &processor, std::ref(packet_queue));
        receiver_thread.join();
        processor_thread.join();
    } else if (mode == "combine") {
        CombineMode combine_mode(PLAY_DELAY_MS);
        combine_mode.start();
    } else {
        std::cerr << "Invalid mode. Use lg, kt, or combine." << std::endl;
        return 1;
    }

    return 0;
}
