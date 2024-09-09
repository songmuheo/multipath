#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <thread>
#include <mutex>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <sys/select.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <algorithm>

#include "config.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
}

// Packet header structure
struct PacketHeader {
    uint64_t timestamp;
    uint32_t sequence_number;
};

// Global variables for state management
std::unordered_map<uint32_t, uint64_t> packet_arrival_time;
std::unordered_set<uint32_t> processed_sequences;
std::mutex packet_mutex;

// 글로벌 변수로 파일 버퍼 및 쓰기 상태 관리
std::vector<std::string> write_buffer;
std::mutex write_mutex;

const size_t WRITE_BUFFER_THRESHOLD = 1000; // 버퍼에 저장될 수 있는 임계값

// Function to log packet information
void log_packet_info(const std::string& source_ip, uint32_t sequence_number, double latency, const std::string& video_label) {
    std::lock_guard<std::mutex> lock(write_mutex);
    write_buffer.push_back(source_ip + "," + std::to_string(sequence_number) + "," + std::to_string(latency) + "\n");

    // 버퍼가 일정 크기 이상일 때 파일에 기록
    if (write_buffer.size() >= WRITE_BUFFER_THRESHOLD) {
        std::ofstream log_file(video_label + "_packet_log.csv", std::ios_base::app);
        for (const auto& line : write_buffer) {
            log_file << line;
        }
        write_buffer.clear();
    }
}

// Function to decode and save video frame
void decode_and_save_frame(AVCodecContext* codec_ctx, SwsContext*& sws_ctx, std::vector<uint8_t>& buffer, const PacketHeader& header, const std::string& save_dir) {
    AVPacket* pkt = av_packet_alloc();
    av_new_packet(pkt, buffer.size() - sizeof(PacketHeader));
    memcpy(pkt->data, buffer.data() + sizeof(PacketHeader), buffer.size() - sizeof(PacketHeader));

    if (avcodec_send_packet(codec_ctx, pkt) == 0) {
        AVFrame* frame = av_frame_alloc();
        if (avcodec_receive_frame(codec_ctx, frame) == 0) {
            // Check if sws_ctx is already initialized
            if (!sws_ctx) {
                sws_ctx = sws_getContext(frame->width, frame->height, (AVPixelFormat)frame->format,
                                         frame->width, frame->height, AV_PIX_FMT_BGR24,
                                         SWS_BILINEAR, nullptr, nullptr, nullptr);
                if (!sws_ctx) {
                    std::cerr << "Could not initialize the conversion context" << std::endl;
                    av_frame_free(&frame);
                    av_packet_free(&pkt);
                    return;
                }
            }

            cv::Mat mat(frame->height, frame->width, CV_8UC3);
            AVFrame* bgr_frame = av_frame_alloc();
            int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, frame->width, frame->height, 1);
            std::vector<uint8_t> bgr_buffer(num_bytes);
            av_image_fill_arrays(bgr_frame->data, bgr_frame->linesize, bgr_buffer.data(), AV_PIX_FMT_BGR24, frame->width, frame->height, 1);

            sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, bgr_frame->data, bgr_frame->linesize);
            memcpy(mat.data, bgr_frame->data[0], num_bytes);

            // Save frame to file using sequence number and timestamp
            std::ostringstream filename;
            filename << save_dir << "/frame_" << header.sequence_number << "_" << header.timestamp << ".png";
            cv::imwrite(filename.str(), mat);

            av_frame_free(&bgr_frame);
        }
        av_frame_free(&frame);
    }
    av_packet_free(&pkt);
}

// Function to process packets from each socket
void process_packet(int sockfd, struct sockaddr_in& client_addr, socklen_t addr_len, AVCodecContext* codec_ctx, SwsContext*& sws_ctx, const std::string& save_dir) {
    std::vector<uint8_t> buffer(BUFFER_SIZE);
    int len = recvfrom(sockfd, buffer.data(), BUFFER_SIZE, 0, (struct sockaddr*)&client_addr, &addr_len);
    if (len > 0) {
        PacketHeader header;
        memcpy(&header, buffer.data(), sizeof(PacketHeader));

        std::string source_ip = inet_ntoa(client_addr.sin_addr);
        uint64_t receive_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        double latency = (receive_time - header.timestamp) / 1000.0;  // Convert to milliseconds

        log_packet_info(source_ip, header.sequence_number, latency, "Combined_Stream");
        decode_and_save_frame(codec_ctx, sws_ctx, buffer, header, save_dir);
    }
}

// Function to process packets using select() for non-blocking I/O
void process_combined_packets(int sockfd1, int sockfd2, AVCodecContext* codec_ctx, SwsContext*& sws_ctx, const std::string& save_dir) {
    fd_set read_fds;
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);

    std::vector<uint8_t> buffer1(BUFFER_SIZE), buffer2(BUFFER_SIZE);

    while (true) {
        // Initialize the set of active sockets
        FD_ZERO(&read_fds);
        FD_SET(sockfd1, &read_fds);
        FD_SET(sockfd2, &read_fds);

        int max_sd = std::max(sockfd1, sockfd2);

        // Select system call to monitor multiple file descriptors
        int activity = select(max_sd + 1, &read_fds, nullptr, nullptr, nullptr);

        if (activity > 0) {
            if (FD_ISSET(sockfd1, &read_fds)) {
                int len1 = recvfrom(sockfd1, buffer1.data(), BUFFER_SIZE, 0, (struct sockaddr*)&client_addr, &addr_len);
                if (len1 > 0) {
                    PacketHeader header;
                    memcpy(&header, buffer1.data(), sizeof(PacketHeader));

                    std::string source_ip = inet_ntoa(client_addr.sin_addr);
                    uint64_t receive_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                    double latency = (receive_time - header.timestamp) / 1000.0;

                    bool is_first_arrival = false;

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
                        log_packet_info(source_ip, header.sequence_number, latency, "Combined_Stream");
                        decode_and_save_frame(codec_ctx, sws_ctx, buffer1, header, save_dir);
                    }
                }
            }

            if (FD_ISSET(sockfd2, &read_fds)) {
                int len2 = recvfrom(sockfd2, buffer2.data(), BUFFER_SIZE, 0, (struct sockaddr*)&client_addr, &addr_len);
                if (len2 > 0) {
                    PacketHeader header;
                    memcpy(&header, buffer2.data(), sizeof(PacketHeader));

                    std::string source_ip = inet_ntoa(client_addr.sin_addr);
                    uint64_t receive_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                    double latency = (receive_time - header.timestamp) / 1000.0;

                    bool is_first_arrival = false;

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
                        log_packet_info(source_ip, header.sequence_number, latency, "Combined_Stream");
                        decode_and_save_frame(codec_ctx, sws_ctx, buffer2, header, save_dir);
                    }
                }
            }
        }

        // Add a small sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void run_server() {
    // Initialize FFmpeg and other settings
    SwsContext* sws_ctx = nullptr; // Only declared once here

    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) {
        std::cerr << "Codec not found" << std::endl;
        return;
    }

    AVCodecContext* codec_ctx1 = avcodec_alloc_context3(codec);
    AVCodecContext* codec_ctx2 = avcodec_alloc_context3(codec);
    AVCodecContext* codec_ctx_combined = avcodec_alloc_context3(codec);

    // Enable multithreading in FFmpeg codec contexts
    codec_ctx1->thread_count = 4;
    codec_ctx2->thread_count = 4;
    codec_ctx_combined->thread_count = 4;

    if (!codec_ctx1 || !codec_ctx2 || !codec_ctx_combined) {
        std::cerr << "Could not allocate video codec context" << std::endl;
        return;
    }

    if (avcodec_open2(codec_ctx1, codec, nullptr) < 0 ||
        avcodec_open2(codec_ctx2, codec, nullptr) < 0 ||
        avcodec_open2(codec_ctx_combined, codec, nullptr) < 0) {
        std::cerr << "Could not open codec" << std::endl;
        return;
    }

    // Setup sockets
    int sockfd1 = socket(AF_INET, SOCK_DGRAM, 0);
    int sockfd2 = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd1 < 0 || sockfd2 < 0) {
        std::cerr << "Socket creation failed" << std::endl;
        return;
    }

    struct sockaddr_in servaddr1 = {};
    servaddr1.sin_family = AF_INET;
    servaddr1.sin_addr.s_addr = INADDR_ANY;
    servaddr1.sin_port = htons(SERVER_PORT1);

    struct sockaddr_in servaddr2 = {};
    servaddr2.sin_family = AF_INET;
    servaddr2.sin_addr.s_addr = INADDR_ANY;
    servaddr2.sin_port = htons(SERVER_PORT2);

    if (bind(sockfd1, (struct sockaddr*)&servaddr1, sizeof(servaddr1)) < 0) {
        std::cerr << "Bind failed on port " << SERVER_PORT1 << std::endl;
        return;
    }
    if (bind(sockfd2, (struct sockaddr*)&servaddr2, sizeof(servaddr2)) < 0) {
        std::cerr << "Bind failed on port " << SERVER_PORT2 << std::endl;
        return;
    }

    std::string save_dir = "./saved_frames"; // 디코딩된 프레임을 저장할 경로 설정
    // 디렉토리가 없으면 생성
    if (access(save_dir.c_str(), F_OK) == -1) {
        if (mkdir(save_dir.c_str(), 0777) == -1) {
            std::cerr << "Could not create save directory: " << save_dir << std::endl;
            return;
        }
    }

    // Start threads for handling each socket
    std::thread socket1_thread([&]() {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        while (true) {
            process_packet(sockfd1, client_addr, addr_len, codec_ctx1, sws_ctx, save_dir);
        }
    });

    std::thread socket2_thread([&]() {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        while (true) {
            process_packet(sockfd2, client_addr, addr_len, codec_ctx2, sws_ctx, save_dir);
        }
    });

    std::thread combined_thread([&]() {
        process_combined_packets(sockfd1, sockfd2, codec_ctx_combined, sws_ctx, save_dir);
    });

    socket1_thread.join();
    socket2_thread.join();
    combined_thread.join();

    // Cleanup
    sws_freeContext(sws_ctx);
    avcodec_free_context(&codec_ctx1);
    avcodec_free_context(&codec_ctx2);
    avcodec_free_context(&codec_ctx_combined);
    close(sockfd1);
    close(sockfd2);

    // 프로그램 종료 시 남은 버퍼 내용을 파일에 기록
    if (!write_buffer.empty()) {
        std::ofstream log_file("Combined_Stream_packet_log.csv", std::ios_base::app);
        for (const auto& line : write_buffer) {
            log_file << line;
        }
        write_buffer.clear();
    }
}

int main() {
    run_server();
    return 0;
}
