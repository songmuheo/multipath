#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <algorithm>
#include <future>
#include <condition_variable>
#include <opencv2/opencv.hpp>
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
std::shared_mutex packet_mutex;  // Use shared_mutex for read-write separation

std::atomic<bool> stop_server(false);  // Global stop flag

// Function to log packet information
void log_packet_info(const std::string& source_ip, uint32_t sequence_number, double latency, const std::string& video_label, const std::string& log_filename) {
    std::ofstream log_file(log_filename, std::ios_base::app);
    log_file << source_ip << "," << sequence_number << "," << latency << "," << video_label << "\n";
}

// Function to save the video packets directly to a file asynchronously
void save_video_packet(const std::vector<uint8_t>& buffer, std::ofstream& output_file) {
    if (output_file.is_open()) {
        output_file.write(reinterpret_cast<const char*>(buffer.data() + sizeof(PacketHeader)), buffer.size() - sizeof(PacketHeader));
    }
}

// Function to process packets from each socket
void process_packet(int sockfd, struct sockaddr_in& client_addr, socklen_t addr_len, std::ofstream& video_file, const std::string& log_filename, const std::string& video_label) {
    std::vector<uint8_t> buffer(BUFFER_SIZE);
    while (!stop_server) {
        int len = recvfrom(sockfd, buffer.data(), BUFFER_SIZE, 0, (struct sockaddr*)&client_addr, &addr_len);
        if (len > 0) {
            uint64_t receive_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            PacketHeader header;
            memcpy(&header, buffer.data(), sizeof(PacketHeader));

            std::string source_ip = inet_ntoa(client_addr.sin_addr);
            double latency = (receive_time - header.timestamp) / 1000.0;

            log_packet_info(source_ip, header.sequence_number, latency, video_label, log_filename);

            save_video_packet(buffer, video_file);
        }
    }
}

// Function to process packets using select() for non-blocking I/O
void process_combined_packets(int sockfd1, int sockfd2, std::ofstream& video_file, const std::string& log_filename) {
    const int MAX_EVENTS = 2;
    struct epoll_event ev, events[MAX_EVENTS];
    int epoll_fd = epoll_create1(0);
    if (epoll_fd == -1) {
        std::cerr << "Failed to create epoll file descriptor" << std::endl;
        return;
    }

    ev.events = EPOLLIN;
    ev.data.fd = sockfd1;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sockfd1, &ev) == -1) {
        std::cerr << "Failed to add sockfd1 to epoll" << std::endl;
        close(epoll_fd);
        return;
    }

    ev.data.fd = sockfd2;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sockfd2, &ev) == -1) {
        std::cerr << "Failed to add sockfd2 to epoll" << std::endl;
        close(epoll_fd);
        return;
    }

    std::vector<uint8_t> buffer(BUFFER_SIZE);
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);

    while (!stop_server) {
        int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        if (nfds == -1) {
            std::cerr << "epoll_wait failed" << std::endl;
            break;
        }

        for (int n = 0; n < nfds; ++n) {
            int sockfd = events[n].data.fd;
            int len = recvfrom(sockfd, buffer.data(), BUFFER_SIZE, 0, (struct sockaddr*)&client_addr, &addr_len);
            if (len > 0) {
                PacketHeader header;
                memcpy(&header, buffer.data(), sizeof(PacketHeader));

                std::string source_ip = inet_ntoa(client_addr.sin_addr);
                uint64_t receive_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                double latency = (receive_time - header.timestamp) / 1000.0;

                // Check for duplicate packets
                std::shared_lock<std::shared_mutex> read_lock(packet_mutex);
                if (processed_sequences.count(header.sequence_number) == 0) {
                    read_lock.unlock();
                    std::unique_lock<std::shared_mutex> write_lock(packet_mutex);
                    processed_sequences.insert(header.sequence_number);
                    packet_arrival_time[header.sequence_number] = receive_time;

                    // Log and save only if not already processed
                    log_packet_info(source_ip, header.sequence_number, latency, "Combined_Stream", log_filename);
                    save_video_packet(buffer, video_file);
                }
            }
        }
    }

    close(epoll_fd);
}

void run_server() {
    // Initialize FFmpeg and other settings
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) {
        std::cerr << "Codec not found" << std::endl;
        return;
    }

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
        close(sockfd1);
        close(sockfd2);
        return;
    }
    if (bind(sockfd2, (struct sockaddr*)&servaddr2, sizeof(servaddr2)) < 0) {
        std::cerr << "Bind failed on port " << SERVER_PORT2 << std::endl;
        close(sockfd1);
        close(sockfd2);
        return;
    }

    std::string video_filename1 = "LGU_plus_video.h264";
    std::string video_filename2 = "KT_video.h264";
    std::string combined_video_filename = "Combined_Stream_video.h264";
    std::string log_filename1 = "LGU_plus_packet_log.csv";
    std::string log_filename2 = "KT_packet_log.csv";
    std::string combined_log_filename = "Combined_Stream_packet_log.csv";

    std::ofstream video_file1(video_filename1, std::ios_base::binary);
    std::ofstream video_file2(video_filename2, std::ios_base::binary);
    std::ofstream combined_video_file(combined_video_filename, std::ios_base::binary);

    std::thread socket1_thread([&]() {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        process_packet(sockfd1, client_addr, addr_len, video_file1, log_filename1, "LGU+_Stream");
    });

    std::thread socket2_thread([&]() {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        process_packet(sockfd2, client_addr, addr_len, video_file2, log_filename2, "KT_Stream");
    });

    std::thread combined_thread([&]() {
        process_combined_packets(sockfd1, sockfd2, combined_video_file, combined_log_filename);
    });

    // Wait for user input to stop the server (or implement your own stop mechanism)
    std::cin.get();
    stop_server = true;

    socket1_thread.join();
    socket2_thread.join();
    combined_thread.join();

    video_file1.close();
    video_file2.close();
    combined_video_file.close();

    close(sockfd1);
    close(sockfd2);
}

int main() {
    run_server();
    return 0;
}
