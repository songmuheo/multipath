// 실시간 영상 재생 코드

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <thread>
#include <mutex>
#include <cstring>
#include <atomic>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <opencv2/opencv.hpp>
#include <sys/select.h>
#include <unistd.h> // close 함수를 위해 추가된 헤더 파일
#include <vector>   // vector 사용을 위한 헤더 파일
#include <string>   // string 사용을 위한 헤더 파일
#include <algorithm> // max 사용을 위한 헤더 파일

#include "config.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
}

// Packet header structure
// struct PacketHeader {
//     uint64_t timestamp;
//     uint32_t sequence_number;
// };
struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};

// Global variables for state management
std::unordered_map<uint32_t, uint64_t> packet_arrival_time;
std::unordered_set<uint32_t> processed_sequences;
std::mutex packet_mutex;

// Function to decode and display video
void decode_and_display(AVCodecContext* codec_ctx, SwsContext*& sws_ctx, std::vector<uint8_t>& buffer, const PacketHeader& header, const std::string& window_name) {
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

            cv::imshow(window_name, mat);
            cv::waitKey(1);

            // Save frame to file
            // cv::imwrite(window_name + "_frame_" + std::to_string(header.sequence_number) + ".jpg", mat);

            av_frame_free(&bgr_frame);
        }
        av_frame_free(&frame);
    }
    av_packet_free(&pkt);
}

// Function to process packets from each socket
void process_packet(int sockfd, struct sockaddr_in& client_addr, socklen_t addr_len, AVCodecContext* codec_ctx, SwsContext*& sws_ctx, const std::string& window_name) {
    std::vector<uint8_t> buffer(BUFFER_SIZE);
    int len = recvfrom(sockfd, buffer.data(), BUFFER_SIZE, 0, (struct sockaddr*)&client_addr, &addr_len);
    if (len > 0) {
        PacketHeader header;
        memcpy(&header, buffer.data(), sizeof(PacketHeader));

        std::string source_ip = inet_ntoa(client_addr.sin_addr);
        uint64_t receive_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        double latency = (receive_time - header.timestamp_sending) / 1000.0;  // Convert to milliseconds

        // Log packet information for the specific socket
        // log_packet_info(source_ip, header.sequence_number, latency, window_name);

        // Decode and display the video
        decode_and_display(codec_ctx, sws_ctx, buffer, header, window_name);
    }
}

// Function to process packets using select() for non-blocking I/O
void process_combined_packets(int sockfd1, int sockfd2, AVCodecContext* codec_ctx, SwsContext*& sws_ctx) {
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
                    double latency = (receive_time - header.timestamp_sending) / 1000.0;

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
                        // log_packet_info(source_ip, header.sequence_number, latency, "Combined_Stream");
                        decode_and_display(codec_ctx, sws_ctx, buffer1, header, "Combined_Stream");
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
                    double latency = (receive_time - header.timestamp_sending) / 1000.0;

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
                        // log_packet_info(source_ip, header.sequence_number, latency, "Combined_Stream");
                        decode_and_display(codec_ctx, sws_ctx, buffer2, header, "Combined_Stream");
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

    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
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

    // Prepare for display using OpenCV
    cv::namedWindow("LGU+", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("KT", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Combined_Stream", cv::WINDOW_AUTOSIZE);

    // Start threads for handling each socket
    std::thread socket1_thread([&]() {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        while (true) {
            process_packet(sockfd1, client_addr, addr_len, codec_ctx1, sws_ctx, "LGU+");
        }
    });

    std::thread socket2_thread([&]() {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        while (true) {
            process_packet(sockfd2, client_addr, addr_len, codec_ctx2, sws_ctx, "KT");
        }
    });

    std::thread combined_thread([&]() {
        process_combined_packets(sockfd1, sockfd2, codec_ctx_combined, sws_ctx);
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
}

int main() {
    run_server();
    return 0;
}
