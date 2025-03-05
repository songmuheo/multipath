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
#include "config.h"

using namespace std;
namespace fs = std::filesystem;

struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};

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
        if (i < size - 3 && data[i] == 0x00 && data[i+1] == 0x00 && data[i+2] == 0x01) {
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

// ACK 송신 함수: 서버가 수신한 패킷에 대해 ACK 메시지를 클라이언트 TURN 릴레이 주소로 전송
void send_ack(int sockfd, const sockaddr_in &client_turn_addr,
              uint32_t sequence_number, uint64_t latency_us)
{
    string ack_msg = "ACK:" + to_string(sequence_number) + "," + to_string(latency_us / 1000.0);
    if (sendto(sockfd, ack_msg.c_str(), ack_msg.size(), 0,
               (const struct sockaddr*)&client_turn_addr, sizeof(client_turn_addr)) < 0)
    {
        cerr << "ACK 송신 실패: " << strerror(errno) << endl;
    }
}

void receive_packets(int port, const char* log_filepath) {
    int sockfd;
    char buffer[BUFFER_SIZE];
    sockaddr_in server_addr, client_addr;
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

    cout << "Listening on port " << port << endl;
    create_log_file(log_filepath);

    // 클라이언트 TURN 릴레이 주소 (config.h에 정의된 값)
    sockaddr_in client_turn_addr = {};
    client_turn_addr.sin_family = AF_INET;
    client_turn_addr.sin_port = htons(CLIENT_TURN_PORT);
    client_turn_addr.sin_addr.s_addr = inet_addr(CLIENT_TURN_IP);

    while (true) {
        ssize_t len = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, (struct sockaddr *)&client_addr, &addr_len);
        if (len < 0) {
            perror("recvfrom failed");
            continue;
        }

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

            // ACK 송신: TURN 릴레이 주소로 전송
            send_ack(sockfd, client_turn_addr, header->sequence_number, network_latency_us);
        }
        else {
            cerr << "Error: Packet size (" << len << ") < Header size (" << header_size << ")\n";
        }
    }
    close(sockfd);
}

int main() {
    string base_dir = FILEPATH_LOG;
    string folder_path = create_timestamped_directory(base_dir);
    string port1_log = folder_path + "/lg_log.csv";
    string port2_log = folder_path + "/kt_log.csv";

    // 포트 1과 포트 2에서 패킷 수신을 별도 스레드로 처리
    thread port1_thread(receive_packets, SERVER_PORT1, port1_log.c_str());
    thread port2_thread(receive_packets, SERVER_PORT2, port2_log.c_str());

    port1_thread.join();
    port2_thread.join();

    return 0;
}
