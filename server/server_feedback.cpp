#include <iostream>
#include <fstream>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <ctime>
#include <thread>
#include <filesystem>
#include <mutex>
#include "config.h"

using namespace std;
namespace fs = std::filesystem;

struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};

std::string client_relay_address = "";
std::mutex relay_mutex;

string create_timestamped_directory(const string& base_dir) {
    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    tm* local_time = localtime(&now_time);
    char folder_name[100];
    strftime(folder_name, sizeof(folder_name), "%Y_%m_%d_%H_%M", local_time);
    string full_path = base_dir + "/" + folder_name;
    fs::create_directories(full_path);
    return full_path;
}

void create_log_file(const char* logpath) {
    ofstream logfile(logpath, ios::app);
    logfile << "source_ip,sequence_number,timestamp_frame,timestamp_sending,received_time,network_latency_ms,message_size,frame_type\n";
    logfile.close();
}

string get_h264_frame_type(const uint8_t* data, size_t size) {
    if (size < 5)
        return "UNKNOWN";
    for (size_t i = 0; i < size - 3; i++) {
        if (data[i] == 0x00 && data[i+1] == 0x00 && data[i+2] == 0x01) {
            if ((data[i+3] & 0x1F) == 5) return "I";
        }
        if (i < size - 4 && data[i] == 0x00 && data[i+1] == 0x00 &&
            data[i+2] == 0x00 && data[i+3] == 0x01) {
            if ((data[i+4] & 0x1F) == 1) return "P";
        }
    }
    return "OTHER";
}

void log_packet_info(const char* logpath,
                     const string& source_ip,
                     uint32_t sequence_number,
                     uint64_t timestamp_frame,
                     uint64_t timestamp_sending,
                     uint64_t received_time_us,
                     uint64_t network_latency_us,
                     size_t message_size,
                     const string& frame_type)
{
    ofstream logfile(logpath, ios::app);
    logfile << source_ip << ","
            << sequence_number << ","
            << timestamp_frame << ","
            << timestamp_sending << ","
            << received_time_us << ","
            << (network_latency_us / 1000.0) << ","
            << message_size << ","
            << frame_type << "\n";
    logfile.close();
}

void send_ack(int sockfd, uint32_t sequence_number, uint64_t latency_us) {
    std::lock_guard<std::mutex> lock(relay_mutex);
    if (!client_relay_address.empty()) {
        size_t colon_pos = client_relay_address.find(':');
        if (colon_pos != std::string::npos) {
            std::string ip = client_relay_address.substr(0, colon_pos);
            std::string port_str = client_relay_address.substr(colon_pos + 1);
            int port = std::stoi(port_str);
            struct sockaddr_in relay_addr;
            inet_pton(AF_INET, ip.c_str(), &relay_addr.sin_addr);
            relay_addr.sin_family = AF_INET;
            relay_addr.sin_port = htons(port);
            std::string ack_msg = "ACK:" + std::to_string(sequence_number) + "," + std::to_string(latency_us / 1000.0);
            if (sendto(sockfd, ack_msg.c_str(), ack_msg.size(), 0,
                       (const struct sockaddr*)&relay_addr, sizeof(relay_addr)) < 0)
            {
                std::cerr << "ACK sending failed: " << strerror(errno) << std::endl;
            }
        } else {
            std::cerr << "Invalid relay address format: " << client_relay_address << std::endl;
        }
    } else {
        std::cerr << "Relay address not set, cannot send ACK" << std::endl;
    }
}

void receive_packets(int port, const char* log_filepath) {
    int sockfd;
    char buffer[BUFFER_SIZE];
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);

    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    memset(&server_addr, 0, sizeof(server_addr));
    memset(&client_addr, 0, sizeof(client_addr));

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    std::cout << "Listening on port " << port << std::endl;
    create_log_file(log_filepath);

    while (true) {
        ssize_t len = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, (struct sockaddr *)&client_addr, &addr_len);
        if (len < 0) {
            perror("recvfrom failed");
            continue;
        }

        std::string data(buffer, len);
        if (data.find("RELAY") == 0) {
            // It's a control message
            std::lock_guard<std::mutex> lock(relay_mutex);
            size_t space_pos = data.find(' ');
            if (space_pos != std::string::npos) {
                client_relay_address = data.substr(space_pos + 1);
                std::cout << "Received relay address: " << client_relay_address << std::endl;
            } else {
                std::cerr << "Invalid RELAY message: " << data << std::endl;
            }
        } else {
            // It's a video packet
            uint64_t received_time_us = chrono::duration_cast<chrono::microseconds>(
                                        chrono::system_clock::now().time_since_epoch()).count();

            char client_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);

            PacketHeader* header = reinterpret_cast<PacketHeader*>(buffer);
            uint64_t network_latency_us = received_time_us - header->timestamp_sending;

            size_t header_size = sizeof(PacketHeader);
            if (len > header_size) {
                const uint8_t* h264_data = reinterpret_cast<const uint8_t*>(buffer + header_size);
                size_t h264_size = len - header_size;
                string frame_type = get_h264_frame_type(h264_data, h264_size);

                log_packet_info(log_filepath,
                                client_ip,
                                header->sequence_number,
                                header->timestamp_frame,
                                header->timestamp_sending,
                                received_time_us,
                                network_latency_us,
                                len,
                                frame_type);

                // Send ACK using relay address
                send_ack(sockfd, header->sequence_number, network_latency_us);
            } else {
                std::cerr << "Error: Packet size (" << len << ") < Header size (" << header_size << ")\n";
            }
        }
    }
    close(sockfd);
}

int main() {
    string base_dir = FILEPATH_LOG;
    string folder_path = create_timestamped_directory(base_dir);
    string port1_log = folder_path + "/lg_log.csv";
    string port2_log = folder_path + "/kt_log.csv";

    thread port1_thread(receive_packets, SERVER_PORT1, port1_log.c_str());
    thread port2_thread(receive_packets, SERVER_PORT2, port2_log.c_str());

    port1_thread.join();
    port2_thread.join();

    return 0;
}