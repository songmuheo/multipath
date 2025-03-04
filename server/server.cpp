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

struct PacketHeader {
    uint64_t timestamp_frame;   // 8 bytes, microsecond 단위, timestamp 생성시의 시점 (인코딩과는 다름)
    uint64_t timestamp_sending; // 8 bytes, microsecond 단위, 해당 패킷을 위한 데이터 인코딩 이후 시점(보냈을 때의 시점)
    uint32_t sequence_number;   // 4 bytes
};

// 현재 날짜와 시간을 기반으로 폴더 경로 생성
std::string create_timestamped_directory(const std::string& base_dir) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);

    std::ostringstream folder_name;
    folder_name << std::put_time(local_time, "%Y_%m_%d_%H_%M");

    std::string full_path = base_dir + "/" + folder_name.str();
    std::filesystem::create_directories(full_path);

    return full_path;
}

void create_log_file(const char* logpath) {
    std::ofstream logfile(logpath, std::ios::app);
    // 헤더에 frame_type 컬럼 추가
    logfile << "source_ip,sequence_number,timestamp_frame,timestamp_sending,received_time,network_latency_ms,message_size,frame_type\n";
    logfile.close();
}

// H.264 프레임 타입을 단순히 판별하는 함수
// - nal_unit_type == 5 : IDR -> I-frame
// - nal_unit_type == 1 : Non-IDR -> P-frame (실제로는 B, P 등이 있지만 여기서는 단순화)
// - 그 외 -> OTHER
std::string get_h264_frame_type(const uint8_t* data, size_t size) {
    if (size < 5) {
        return "UNKNOWN";
    }

    for (size_t i = 0; i < size - 4; i++) {
        if (data[i] == 0x00 && data[i + 1] == 0x00) {
            if (data[i + 2] == 0x01) {  // 3-byte Start Code (I-frame)
                if ((data[i + 3] & 0x1F) == 5) return "I";
            } 
            else if (data[i + 2] == 0x00 && data[i + 3] == 0x01) {  // 4-byte Start Code (P-frame)
                if ((data[i + 4] & 0x1F) == 1) return "P";
            }
        }
    }

    return "OTHER";
}

// 패킷 정보를 CSV 파일에 기록하는 함수
// frame_type을 새로 추가
void log_packet_info(const char* logpath,
                     const std::string& source_ip,
                     uint32_t sequence_number,
                     uint64_t timestamp_frame,
                     uint64_t timestamp_sending,
                     uint64_t received_time_us,
                     uint64_t network_latency_us,
                     size_t message_size,
                     const std::string& frame_type)
{
    std::ofstream logfile(logpath, std::ios::app);
    logfile << source_ip << ","
            << sequence_number << ","
            << timestamp_frame << ","
            << timestamp_sending << ","
            << received_time_us << ","
            << (network_latency_us / 1000.0) << ","
            << message_size << ","
            << frame_type
            << "\n";
    logfile.close();
}

// 소켓 설정 및 데이터 수신
void receive_packets(int port, const char* log_filepath) {
    int sockfd;
    char buffer[BUFFER_SIZE];
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);

    // 소켓 생성
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    memset(&client_addr, 0, sizeof(client_addr));

    // 서버 주소 설정
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    // 소켓에 주소 바인딩
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

        // 패킷 수신 시간 기록 (마이크로초 단위)
        uint64_t received_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::system_clock::now().time_since_epoch()).count();

        // 클라이언트 IP 주소 가져오기
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);

        // 사용자 정의 헤더 파싱 (PacketHeader의 크기는 20바이트)
        PacketHeader* header = reinterpret_cast<PacketHeader*>(buffer);

        // 패킷 도착 지연 시간 계산 (마이크로초 단위)
        uint64_t network_latency_us = received_time_us - header->timestamp_sending;

        // PacketHeader 바로 뒤부터 H.264 데이터가 시작된다고 가정
        // (주의: 실제로는 파편화 등 다양한 변수가 있을 수 있음)
        size_t header_size = sizeof(PacketHeader); // 20 bytes
        if (len > header_size) {
            const uint8_t* h264_data = reinterpret_cast<const uint8_t*>(buffer + header_size);
            size_t h264_size = len - header_size;

            // H.264 프레임 타입 판별
            std::string frame_type = get_h264_frame_type(h264_data, h264_size);

            // 패킷 정보 로그 파일에 기록
            log_packet_info(log_filepath,
                            client_ip,
                            header->sequence_number,
                            header->timestamp_frame,
                            header->timestamp_sending,
                            received_time_us,
                            network_latency_us,
                            len,
                            frame_type);
        }
        else {
            // 헤더보다 패킷 전체 길이가 작으면 에러 로그만 찍고 넘어감
            std::cerr << "Error: Packet size (" << len << ") < Header size (" << header_size << ")\n";
        }
    }

    close(sockfd);
}

int main() {
    // 새로운 디렉터리 생성
    std::string base_dir = FILEPATH_LOG;
    std::string folder_path = create_timestamped_directory(base_dir);

    std::string port1_log = folder_path + "/lg_log.csv";
    std::string port2_log = folder_path + "/kt_log.csv";

    // 포트 1과 포트 2에서 패킷 수신을 처리하는 쓰레드 생성 (멀티스레드로 처리할 경우)
    std::thread port1_thread(receive_packets, SERVER_PORT1, port1_log.c_str());
    std::thread port2_thread(receive_packets, SERVER_PORT2, port2_log.c_str());

    // 쓰레드가 종료될 때까지 대기
    port1_thread.join();
    port2_thread.join();

    return 0;
}