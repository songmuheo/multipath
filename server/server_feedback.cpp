#include <iostream>
#include <fstream>
#include <string>
#include <atomic>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

// OpenSSL
#include <openssl/hmac.h>
#include <openssl/evp.h>

// pjnath
extern "C" {
#include <pjlib.h>
#include <pjlib-util.h>
#include <pjnath.h>
}

// config.h
#include "config.h"

using namespace std;
namespace fs = std::filesystem;

// ----------------------------
//   패킷 헤더
// ----------------------------
struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};

// ----------------------------
//   H.264 프레임 타입 판별
// ----------------------------
string get_h264_frame_type(const uint8_t* data, size_t size) {
    if (size < 5) return "UNKNOWN";
    // 단순 검색
    for (size_t i = 0; i < size - 4; i++) {
        // 00 00 01 (3바이트) 또는 00 00 00 01 (4바이트) NAL 헤더
        if ((data[i] == 0x00 && data[i+1] == 0x00 && data[i+2] == 0x01)) {
            // i+3 위치가 NAL Type
            uint8_t nal_type = data[i+3] & 0x1F;
            if (nal_type == 5) return "I";
            if (nal_type == 1) return "P";
        }
        if (i+4 < size && data[i] == 0x00 && data[i+1] == 0x00 &&
            data[i+2] == 0x00 && data[i+3] == 0x01) {
            uint8_t nal_type = data[i+4] & 0x1F;
            if (nal_type == 5) return "I";
            if (nal_type == 1) return "P";
        }
    }
    return "OTHER";
}

// ----------------------------
//   로그 함수
// ----------------------------
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

// ----------------------------
//   TURN 인증 유틸
// ----------------------------
std::string generate_turn_username(const std::string& identifier, uint32_t validSeconds) {
    uint64_t expiration = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::system_clock::now().time_since_epoch()
                          ).count() + validSeconds;
    return std::to_string(expiration) + ":" + identifier;
}

std::string compute_turn_password(const std::string& data, const std::string& secret) {
    unsigned char hmac_result[EVP_MAX_MD_SIZE] = {0};
    unsigned int hmac_length = 0;
    HMAC(EVP_sha1(),
         secret.c_str(), secret.size(),
         reinterpret_cast<const unsigned char*>(data.c_str()), data.size(),
         hmac_result, &hmac_length);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < hmac_length; i++)
        oss << std::setw(2) << (int)hmac_result[i];
    return oss.str();
}

// ----------------------------
//   서버: on_rx_data (영상 수신 + ACK 전송)
// ----------------------------
static string g_port_log_path; // 로그파일 경로 (하나만 사용 예)

static void on_rx_data_server(pj_turn_sock *sock,
                              void *pkt,
                              pj_size_t size,
                              const pj_sockaddr_t *src_addr,
                              unsigned int addr_len)
{
    if (!pkt || size < sizeof(PacketHeader)) {
        cerr << "[SERVER] Invalid packet size=" << size << endl;
        return;
    }

    // 1) 현재 시각
    uint64_t received_time_us = chrono::duration_cast<chrono::microseconds>(
                                    chrono::system_clock::now().time_since_epoch()
                                ).count();

    // 2) PacketHeader 파싱
    PacketHeader *header = reinterpret_cast<PacketHeader*>(pkt);
    uint64_t network_latency_us = received_time_us - header->timestamp_sending;

    // 3) (header 뒤) H.264 데이터
    size_t header_size = sizeof(PacketHeader);
    size_t h264_size = (size_t)size - header_size;
    uint8_t* h264_data = (uint8_t*)((char*)pkt + header_size);

    // 4) NAL 타입 추출
    string frame_type = get_h264_frame_type(h264_data, h264_size);

    // 5) src_addr → 문자열(IP) 변환
    char ip_str[64];
    pj_sockaddr_print(src_addr, ip_str, sizeof(ip_str), 0);

    // 6) 로그 기록
    log_packet_info(g_port_log_path.c_str(),
                    ip_str,
                    header->sequence_number,
                    header->timestamp_frame,
                    header->timestamp_sending,
                    received_time_us,
                    network_latency_us,
                    size,
                    frame_type);

    // 7) ACK 전송
    //    예: "ACK:seqnum,latency_ms"
    double latency_ms = network_latency_us / 1000.0;
    ostringstream oss;
    oss << "ACK:" << header->sequence_number << "," << latency_ms;
    string ack_msg = oss.str();

    // TURN 소켓을 통해, 수신한 src_addr로 그대로 전송
    pj_status_t st = pj_turn_sock_sendto(sock,
                                         ack_msg.data(),
                                         (pj_size_t)ack_msg.size(),
                                         0,
                                         src_addr,
                                         addr_len);
    if (st != PJ_SUCCESS) {
        cerr << "[SERVER] Failed to send ACK via TURN. status=" << st << endl;
    }
}

// ----------------------------
//   서버 main
// ----------------------------
int main() {
    // (1) 로그 폴더 생성
    string base_dir = FILEPATH_LOG; // config.h에서 가져오는 베이스 경로
    string folder_path = create_timestamped_directory(base_dir);

    // 단일 로그파일만 사용 (기존 코드에서는 port1/port2 따로 쓰지만 여기선 단일)
    g_port_log_path = folder_path + "/turn_log.csv";
    create_log_file(g_port_log_path.c_str());

    // (2) pjnath 초기화
    pj_status_t status;
    status = pj_init();
    if (status != PJ_SUCCESS) {
        cerr << "[SERVER] pj_init() failed" << endl;
        return 1;
    }
    status = pjlib_util_init();
    if (status != PJ_SUCCESS) {
        cerr << "[SERVER] pjlib_util_init() failed" << endl;
        return 1;
    }

    pj_caching_pool cp;
    pj_caching_pool_init(&cp, nullptr, 0);

    pj_pool_t *pool = pj_pool_create(&cp.factory, "srv_pool", 4000, 4000, nullptr);
    if (!pool) {
        cerr << "[SERVER] pj_pool_create() failed" << endl;
        return 1;
    }

    pj_ioqueue_t *ioqueue = nullptr;
    status = pj_ioqueue_create(pool, 64, &ioqueue);
    if (status != PJ_SUCCESS) {
        cerr << "[SERVER] pj_ioqueue_create() error" << endl;
        return 1;
    }

    pj_timer_heap_t *timer_heap = nullptr;
    status = pj_timer_heap_create(pool, 100, &timer_heap);
    if (status != PJ_SUCCESS) {
        cerr << "[SERVER] pj_timer_heap_create() error" << endl;
        return 1;
    }

    pj_stun_config stun_cfg;
    pj_stun_config_init(&stun_cfg, &cp.factory, PJ_AF_INET, ioqueue, timer_heap);

    // (3) TURN 소켓 생성
    static pj_turn_sock_cb turn_cb_server;
    memset(&turn_cb_server, 0, sizeof(turn_cb_server));
    turn_cb_server.on_rx_data = &on_rx_data_server;

    pj_turn_sock *turn_sock_server = nullptr;
    status = pj_turn_sock_create(&stun_cfg,
                                 PJ_AF_INET,
                                 PJ_TURN_TP_UDP,
                                 &turn_cb_server,
                                 nullptr, // user_data
                                 nullptr, // existing socket
                                 &turn_sock_server);
    if (status != PJ_SUCCESS || !turn_sock_server) {
        cerr << "[SERVER] pj_turn_sock_create() error" << endl;
        return 1;
    }

    // (4) TURN Allocate
    //     서버도 ephemeral username/password
    {
        std::string ephemeral_username = generate_turn_username(TURN_IDENTIFIER, TURN_VALID_SECONDS);
        std::string ephemeral_password = compute_turn_password(ephemeral_username + std::string(":") + TURN_REALM,
                                                               TURN_SECRET);

        pj_stun_auth_cred auth_cred;
        memset(&auth_cred, 0, sizeof(auth_cred));
        auth_cred.type = PJ_STUN_AUTH_CRED_STATIC;
        auth_cred.data.static_cred.username = pj_str(const_cast<char*>(ephemeral_username.c_str()));
        auth_cred.data.static_cred.data     = pj_str(const_cast<char*>(ephemeral_password.c_str()));
        auth_cred.data.static_cred.data_type = PJ_STUN_PASSWD_PLAIN;

        pj_str_t turn_srv_ip = pj_str(const_cast<char*>(TURN_SERVER_IP));
        status = pj_turn_sock_alloc(turn_sock_server,
                                    &turn_srv_ip,
                                    TURN_SERVER_PORT,
                                    nullptr,
                                    &auth_cred,
                                    nullptr);
        if (status != PJ_SUCCESS) {
            cerr << "[SERVER] pj_turn_sock_alloc() fail. status=" << status << endl;
            return 1;
        }
    }

    // (5) 할당 완료 대기
    bool allocated = false;
    for(int i=0; i<200; i++){
        pj_time_val delay = {0, 10}; // 10ms
        pj_ioqueue_poll(ioqueue, &delay);
        pj_timer_heap_poll(timer_heap, nullptr);

        pj_turn_sock_info info;
        memset(&info, 0, sizeof(info));
        pj_turn_sock_get_info(turn_sock_server, &info);

        if (info.state == PJ_TURN_STATE_READY) {
            allocated = true;
            break;
        }
        pj_thread_sleep(10);
    }
    if (!allocated) {
        cerr << "[SERVER] TURN allocate not completed" << endl;
        return 1;
    }

    // (6) 서버 Relay 주소를 콘솔에 표시
    {
        pj_turn_sock_info info;
        memset(&info, 0, sizeof(info));
        pj_turn_sock_get_info(turn_sock_server, &info);

        char relay_ipstr[128];
        pj_sockaddr_print(&info.relay_addr, relay_ipstr, sizeof(relay_ipstr), 0);
        cout << "[SERVER] TURN relay allocated = " << relay_ipstr << endl
             << "         (클라이언트가 이 주소로 sendto 하도록 설정해야 함)" << endl;
    }

    // (7) 메인 루프
    cout << "[SERVER] Press Enter to stop...\n";
    while (true) {
        if (cin.peek() != EOF) {
            // 엔터 입력하면 종료
            break;
        }
        pj_time_val delay = {0, 10};
        pj_ioqueue_poll(ioqueue, &delay);
        pj_timer_heap_poll(timer_heap, nullptr);
        pj_thread_sleep(10);
    }

    // (8) 자원정리
    if (turn_sock_server) {
        pj_turn_sock_destroy(turn_sock_server);
        turn_sock_server = nullptr;
    }
    pj_pool_release(pool);
    pj_caching_pool_destroy(&cp);
    pj_shutdown();

    return 0;
}
