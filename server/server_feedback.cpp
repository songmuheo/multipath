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
#include <mutex>
#include <filesystem>
#include "config.h"

using namespace std;
namespace fs = std::filesystem;

class BufferedLogger {
public:
    BufferedLogger(const string& filepath) {
        log_stream.open(filepath, ios::out | ios::app);
        if (!log_stream.is_open()) {
            throw runtime_error("Failed to open log file: " + filepath);
        }
    }
    ~BufferedLogger() {
        flush();
        log_stream.close();
    }
    void log(const string& message) {
        lock_guard<mutex> lock(log_mutex);
        log_stream << message << "\n";
    }
    void flush() {
        lock_guard<mutex> lock(log_mutex);
        log_stream.flush();
    }
private:
    ofstream log_stream;
    mutex log_mutex;
};

#pragma pack(push, 1)
struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};
#pragma pack(pop)

// ---------- 타임스탬프 폴더 ----------
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

// ---------- 간단한 H.264 프레임 타입 추출 ----------
string get_h264_frame_type(const uint8_t* data, size_t size) {
    if (size < 5)
        return "UNKNOWN";
    for (size_t i = 0; i < size - 3; i++) {
        if (data[i] == 0x00 && data[i+1] == 0x00 && data[i+2] == 0x01) {
            uint8_t nal = data[i+3] & 0x1F;
            if (nal == 5) return "I";
            else if (nal == 1) return "P";
        }
        // 4바이트 start code
        if (i < size - 4 &&
            data[i] == 0x00 && data[i+1] == 0x00 &&
            data[i+2] == 0x00 && data[i+3] == 0x01) {
            uint8_t nal = data[i+4] & 0x1F;
            if (nal == 5) return "I";
            else if (nal == 1) return "P";
        }
    }
    return "OTHER";
}

// ---------- 로그 출력 ----------
void log_packet_info(BufferedLogger& logger,
                     const string& source_ip,
                     uint32_t sequence_number,
                     uint64_t timestamp_frame,
                     uint64_t timestamp_sending,
                     uint64_t received_time_us,
                     uint64_t network_latency_us,
                     size_t message_size,
                     const string& frame_type)
{
    // CSV
    ostringstream oss;
    oss << source_ip << ","
        << sequence_number << ","
        << timestamp_frame << ","
        << timestamp_sending << ","
        << received_time_us << ","
        << (network_latency_us / 1000.0) << ","
        << message_size << ","
        << frame_type;
    logger.log(oss.str());
}

// ---------- 서버에서 ACK 전송 ----------
void send_ack(int sockfd, const sockaddr_in &dest_addr,
              uint32_t sequence_number, uint64_t latency_us)
{
    // 단순 문자열
    string ack_msg = "ACK:" + to_string(sequence_number) + "," + to_string(latency_us / 1000.0);
    if (sendto(sockfd, ack_msg.c_str(), ack_msg.size(), 0,
               (const struct sockaddr*)&dest_addr, sizeof(dest_addr)) < 0)
    {
        cerr << "ACK send error: " << strerror(errno) << endl;
    } else {
        cout << "[Server] ACK sent to " << inet_ntoa(dest_addr.sin_addr)
             << ":" << ntohs(dest_addr.sin_port) << " seq=" << sequence_number << endl;
    }
}

// ------------- TURN 등록 (클라이언트 Relay 주소) 관련 -------------
mutex reg_mutex;
bool  turn_registered = false;
sockaddr_in client_turn_addr; // 클라이언트의 Relay 주소

// ------------- 패킷 수신 -------------
void receive_packets(int port, BufferedLogger& logger) {
    int sockfd;
    char buffer[BUFFER_SIZE];
    sockaddr_in server_addr, sender_addr;
    socklen_t addr_len = sizeof(sender_addr);

    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    memset(&server_addr, 0, sizeof(server_addr));
    memset(&sender_addr, 0, sizeof(sender_addr));

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    cout << "[Server] Listening on port " << port << endl;

    while (true) {
        ssize_t len = recvfrom(sockfd, buffer, BUFFER_SIZE, 0,
                               (struct sockaddr *)&sender_addr, &addr_len);
        if (len < 0) {
            perror("recvfrom failed");
            continue;
        }
        uint64_t received_time_us = chrono::duration_cast<chrono::microseconds>(
                                        chrono::system_clock::now().time_since_epoch()
                                    ).count();

        char sender_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &sender_addr.sin_addr, sender_ip, INET_ADDRSTRLEN);

        // TURN 등록 패킷: "TURN_REG:1.2.3.4:9999"
        if (len >= 9 && strncmp(buffer, "TURN_REG:", 9) == 0) {
            string reg_msg(buffer, len);
            size_t pos1 = reg_msg.find(":");
            size_t pos2 = reg_msg.find(":", pos1+1);
            if (pos1 != string::npos && pos2 != string::npos) {
                string ip_str   = reg_msg.substr(pos1+1, pos2 - pos1 - 1);
                string port_str = reg_msg.substr(pos2+1);
                memset(&client_turn_addr, 0, sizeof(client_turn_addr));
                client_turn_addr.sin_family = AF_INET;
                client_turn_addr.sin_addr.s_addr = inet_addr(ip_str.c_str());
                client_turn_addr.sin_port = htons(stoi(port_str));
                {
                    lock_guard<mutex> lock(reg_mutex);
                    turn_registered = true;
                }
                cout << "[Server] Registered client TURN address: "
                     << ip_str << ":" << port_str << endl;
            }
            continue;
        }

        // 일반 패킷 (H.264 + header)
        if (len < (ssize_t)sizeof(PacketHeader)) {
            cerr << "[Server] Packet too small: " << len << endl;
            continue;
        }
        PacketHeader* hdr = reinterpret_cast<PacketHeader*>(buffer);
        uint64_t network_latency_us = received_time_us - hdr->timestamp_sending;

        // H.264 본문
        size_t header_size = sizeof(PacketHeader);
        if (len > (ssize_t)header_size) {
            const uint8_t* h264_data = (const uint8_t*)(buffer + header_size);
            size_t h264_size = len - header_size;
            string frame_type = get_h264_frame_type(h264_data, h264_size);

            // 로깅
            log_packet_info(logger, sender_ip,
                            hdr->sequence_number,
                            hdr->timestamp_frame,
                            hdr->timestamp_sending,
                            received_time_us,
                            network_latency_us,
                            len,
                            frame_type);

            // ACK 전송할 대상
            sockaddr_in dest_addr;
            {
                lock_guard<mutex> lock(reg_mutex);
                if (turn_registered) {
                    // 클라이언트가 등록한 TURN relay 주소가 있으면 그쪽으로 ACK
                    dest_addr = client_turn_addr;
                } else {
                    // 아직 등록 전이면 그냥 sender_addr로
                    dest_addr = sender_addr;
                }
            }
            send_ack(sockfd, dest_addr, hdr->sequence_number, network_latency_us);
        }
    }
    close(sockfd);
}

int main() {
    try {
        // 로그 폴더
        string base_dir = FILEPATH_LOG;  // 예: "/home/user/logs"
        string folder_path = create_timestamped_directory(base_dir);

        string port1_log_path = folder_path + "/lg_log.csv";
        string port2_log_path = folder_path + "/kt_log.csv";

        BufferedLogger logger1(port1_log_path);
        BufferedLogger logger2(port2_log_path);

        // 두 개 포트 수신(예: SERVER_REG_PORT, SERVER_PORT2 등 config.h에 지정)
        thread port1_thread(receive_packets, SERVER_REG_PORT, ref(logger1));
        thread port2_thread(receive_packets, SERVER_PORT2,    ref(logger2));

        port1_thread.join();
        port2_thread.join();

    } catch (const exception &e) {
        cerr << "[Server] error: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    return 0;
}
