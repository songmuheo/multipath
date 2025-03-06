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

// OpenSSL 헤더 추가
#include <openssl/evp.h>
#include <openssl/hmac.h>

// libnice / GLib (TURN ACK 송신용)
#include <nice/agent.h>
#include <glib.h>

using namespace std;
namespace fs = std::filesystem;

// 만약 NICE_RELAY_TURN가 정의되어 있지 않으면 NICE_RELAY_TYPE_TURN_TLS로 정의
#ifndef NICE_RELAY_TURN
#define NICE_RELAY_TURN NICE_RELAY_TYPE_TURN_TLS
#endif

// 전역 변수 (TURN 관련)
static NiceAgent *g_server_agent = nullptr;
static guint g_server_stream_id = 0;
static GMainLoop *g_server_loop = nullptr;
static atomic<bool> server_turn_ready(false);

// 전역 TURN 인증 정보 (한 번만 생성해서 재사용)
static string g_turn_username;
static string g_turn_password;

// TURN credential 관련 함수
std::string base64_encode(const std::string& input) {
    std::ostringstream out;
    int val = 0, valb = -6;
    for (unsigned char c : input) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out << "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(val >> valb) & 0x3F];
            valb -= 6;
        }
    }
    if (valb > -6)
        out << "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[((val << 8) >> (valb + 8)) & 0x3F];
    while (out.tellp() % 4)
        out << '=';
    return out.str();
}

std::string compute_turn_password(const std::string& user_with_timestamp, const std::string& realm, const std::string& secret) {
    std::string key = user_with_timestamp + ":" + realm;
    unsigned char hmac_result[EVP_MAX_MD_SIZE];
    unsigned int hmac_length = 0;
    memset(hmac_result, 0, sizeof(hmac_result));
    HMAC(EVP_sha1(), secret.c_str(), secret.size(),
         reinterpret_cast<const unsigned char*>(key.c_str()), key.size(),
         hmac_result, &hmac_length);
    return base64_encode(std::string(reinterpret_cast<const char*>(hmac_result), hmac_length));
}

std::string generate_turn_username(const std::string& identifier, uint32_t validSeconds) {
    uint64_t expiration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() + validSeconds;
    return std::to_string(expiration) + ":" + identifier;
}

// TURN 인증 정보를 초기화 (한 번만 호출)
// void initialize_turn_credentials() {
//     // TURN_IDENTIFIER, TURN_VALID_SECONDS, TURN_REALM, TURN_SECRET는 config.h에서 정의됨
//     g_turn_username = generate_turn_username(TURN_IDENTIFIER, TURN_VALID_SECONDS);
//     g_turn_password = compute_turn_password(g_turn_username, TURN_REALM, TURN_SECRET);
//     cout << "Initialized TURN credentials: username=" << g_turn_username << endl;
// }

// // server_setup_turn_info() 함수: libnice의 nice_agent_set_relay_info() 사용
// void server_setup_turn_info() {
//     if (g_turn_username.empty()) {
//         initialize_turn_credentials();
//     }
//     if (!nice_agent_set_relay_info(g_server_agent, g_server_stream_id, 1,
//                                    TURN_SERVER_IP, TURN_SERVER_PORT,
//                                    g_turn_username.c_str(), g_turn_password.c_str(),
//                                    NICE_RELAY_TURN)) {
//         cerr << "Server: Failed to set relay info\n";
//     } else {
//         g_print("Server: Relay info set successfully\n");
//     }
// }

// static string g_turn_username;
// static string g_turn_password;
void initialize_turn_credentials() {
    // TURN_IDENTIFIER, TURN_VALID_SECONDS, TURN_REALM, TURN_SECRET는 config.h에 정의됨
    g_turn_username = generate_turn_username(TURN_IDENTIFIER, TURN_VALID_SECONDS);
    g_turn_password = compute_turn_password(g_turn_username, TURN_REALM, TURN_SECRET);
    cout << "Initialized TURN credentials: username=" << g_turn_username << endl;
}

void server_setup_turn_info() {
    if (g_turn_username.empty()) {
        initialize_turn_credentials();
    }
    if (!nice_agent_set_relay_info(g_server_agent, g_server_stream_id, 1,
                                   TURN_SERVER_IP, TURN_SERVER_PORT,
                                   g_turn_username.c_str(), g_turn_password.c_str(),
                                   NICE_RELAY_TURN)) {
        cerr << "Server: Failed to set relay info\n";
    } else {
        g_print("Server: Relay info set successfully\n");
    }
}

// --- 로깅 및 패킷 관련 ---
class BufferedLogger {
public:
    BufferedLogger(const string& filepath) {
        log_stream.open(filepath, ios::out | ios::app);
        if (!log_stream.is_open())
            throw runtime_error("Failed to open log file: " + filepath);
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

struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};

string create_timestamped_directory(const string& base_dir) {
    auto now = std::chrono::system_clock::now();
    time_t now_time = std::chrono::system_clock::to_time_t(now);
    tm* local_time = localtime(&now_time);
    char folder_name[100];
    strftime(folder_name, sizeof(folder_name), "%Y_%m_%d_%H_%M", local_time);
    string full_path = base_dir + "/" + folder_name;
    fs::create_directories(full_path);
    return full_path;
}

string get_h264_frame_type(const uint8_t* data, size_t size) {
    if (size < 5)
        return "UNKNOWN";
    for (size_t i = 0; i < size - 3; i++) {
        if (data[i] == 0x00 && data[i+1] == 0x00 && data[i+2] == 0x01) {
            if ((data[i+3] & 0x1F) == 5)
                return "I";
            else if ((data[i+3] & 0x1F) == 1)
                return "P";
        }
        if (i < size - 4 && data[i] == 0x00 && data[i+1] == 0x00 &&
            data[i+2] == 0x00 && data[i+3] == 0x01) {
            if ((data[i+4] & 0x1F) == 5)
                return "I";
            else if ((data[i+4] & 0x1F) == 1)
                return "P";
        }
    }
    return "OTHER";
}

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

void send_ack(int sockfd, const sockaddr_in &dest_addr,
              uint32_t sequence_number, uint64_t latency_us)
{
    string ack_msg = "ACK:" + to_string(sequence_number) + "," + to_string(latency_us / 1000.0);
    if (sendto(sockfd, ack_msg.c_str(), ack_msg.size(), 0,
               (const struct sockaddr*)&dest_addr, sizeof(dest_addr)) < 0)
    {
        cerr << "ACK send error: " << strerror(errno) << endl;
    } else {
        cout << "ACK sent to " << inet_ntoa(dest_addr.sin_addr)
             << ":" << ntohs(dest_addr.sin_port) << endl;
    }
}

// --- libnice TURN ACK 송신 (서버용) ---

// 클라이언트 TURN 등록 정보 저장
struct RemoteCandidate {
    string ip;
    int port;
};
static RemoteCandidate client_remote_candidate;
static atomic<bool> client_turn_registered(false);
mutex candidate_mutex;

static void server_cb_candidate_gathering_done(NiceAgent *agent, guint stream_id, gpointer user_data) {
    g_print("Server: Candidate gathering done for stream %u\n", stream_id);
}

static void server_cb_component_state_changed(NiceAgent *agent, guint stream_id, guint component_id, guint state, gpointer user_data) {
    g_print("Server: Component %u state changed to %u\n", component_id, state);
    if (state == NICE_COMPONENT_STATE_READY) {
        server_turn_ready = true;
        g_print("Server: Component %u is ready (TURN 서버 연결 성공)\n", component_id);
    }
}

// 데이터 수신 콜백 (실제로 사용되지는 않음)
static gboolean server_cb_data_received(NiceAgent *agent, guint stream_id, guint component_id,
                                          guint len, gchar *buf, gpointer user_data) {
    g_print("Server: Data received via TURN (should not happen on sender): %u bytes\n", len);
    return TRUE;
}

static void server_turn_sender_thread() {
    g_server_loop = g_main_loop_new(NULL, FALSE);
    GMainContext *context = g_main_loop_get_context(g_server_loop);
    g_server_agent = nice_agent_new(context, NICE_COMPATIBILITY_RFC5245);
    if (!g_server_agent) {
        cerr << "Server: Failed to create NiceAgent\n";
        return;
    }
    g_server_stream_id = nice_agent_add_stream(g_server_agent, 1);
    // TURN 설정: component_id는 1, 그리고 NICE_RELAY_TURN 사용
    server_setup_turn_info();
    g_signal_connect(G_OBJECT(g_server_agent), "candidate-gathering-done", G_CALLBACK(server_cb_candidate_gathering_done), NULL);
    g_signal_connect(G_OBJECT(g_server_agent), "component-state-changed", G_CALLBACK(server_cb_component_state_changed), NULL);
    // 데이터 수신 콜백은 등록하지 않습니다.
    if (!nice_agent_gather_candidates(g_server_agent, g_server_stream_id)) {
        cerr << "Server: Failed to start candidate gathering\n";
        return;
    }
    g_print("Server: Running Nice TURN sender...\n");
    g_main_loop_run(g_server_loop);
    g_object_unref(g_server_agent);
    g_main_loop_unref(g_server_loop);
}

// 클라이언트의 TURN 등록 메시지 ("TURN_REG:ip:port")를 처리하여 원격 후보로 설정
static void set_remote_candidate(const string &ip, int port) {
    lock_guard<mutex> lock(candidate_mutex);
    client_remote_candidate.ip = ip;
    client_remote_candidate.port = port;
    client_turn_registered = true;
    
    // 원격 후보 생성
    NiceCandidate *rcand = nice_candidate_new(NICE_CANDIDATE_TYPE_RELAYED);
    rcand->component_id = 1;
    nice_address_set_from_string(&rcand->addr, ip.c_str());
    nice_address_set_port(&rcand->addr, port);
    GSList *rcand_list = NULL;
    rcand_list = g_slist_append(rcand_list, rcand);
    if (!nice_agent_set_remote_candidates(g_server_agent, g_server_stream_id, 1, rcand_list))
        g_print("Server: Failed to set remote candidate\n");
    else
        g_print("Server: Remote candidate set: %s:%d\n", ip.c_str(), port);
    g_slist_free_full(rcand_list, (GDestroyNotify)&nice_candidate_free);
}

// TURN을 통해 ACK 전송 (서버)
void turn_send_ack_libnice(uint32_t sequence_number, uint64_t latency_us) {
    if (!server_turn_ready || !client_turn_registered.load()) {
        g_print("Server: TURN not ready or client not registered; cannot send ACK via TURN\n");
        return;
    }
    string ack_msg = "ACK:" + to_string(sequence_number) + "," + to_string(latency_us / 1000.0);
    int ret = nice_agent_send(g_server_agent, g_server_stream_id, 1, ack_msg.size(), ack_msg.c_str());
    if (ret < 0)
         g_print("Server: Failed to send ACK via TURN\n");
    else
         g_print("Server: Sent ACK via TURN for seq=%u\n", sequence_number);
}

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
    cout << "Server listening on port " << port << endl;
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
        // TURN_REG 메시지 처리
        if (len >= 9 && strncmp(buffer, "TURN_REG:", 9) == 0) {
            string reg_msg(buffer, len);
            size_t pos1 = reg_msg.find(":");
            size_t pos2 = reg_msg.find(":", pos1 + 1);
            if (pos1 != string::npos && pos2 != string::npos) {
                string ip_str   = reg_msg.substr(pos1 + 1, pos2 - pos1 - 1);
                string port_str = reg_msg.substr(pos2 + 1);
                int reg_port = atoi(port_str.c_str());
                set_remote_candidate(ip_str, reg_port);
                cout << "Server registered client TURN candidate: " << ip_str
                     << ":" << reg_port << endl;
            }
            continue;
        }
        if (len < (ssize_t)sizeof(PacketHeader)) {
            cerr << "Error: Packet too small (" << len << ")" << endl;
            continue;
        }
        PacketHeader* header = reinterpret_cast<PacketHeader*>(buffer);
        uint64_t network_latency_us = received_time_us - header->timestamp_sending;
        size_t header_size = sizeof(PacketHeader);
        if (len > (ssize_t)header_size) {
            const uint8_t* h264_data = reinterpret_cast<const uint8_t*>(buffer + header_size);
            size_t h264_size = len - header_size;
            string frame_type = get_h264_frame_type(h264_data, h264_size);
            log_packet_info(logger, sender_ip,
                            header->sequence_number,
                            header->timestamp_frame,
                            header->timestamp_sending,
                            received_time_us,
                            network_latency_us,
                            len,
                            frame_type);
            if (client_turn_registered.load())
                turn_send_ack_libnice(header->sequence_number, network_latency_us);
            else
                send_ack(sockfd, sender_addr, header->sequence_number, network_latency_us);
        }
    }
    close(sockfd);
}

int main() {
    try {
        string base_dir = FILEPATH_LOG;
        string folder_path = create_timestamped_directory(base_dir);
        string port1_log_path = folder_path + "/lg_log.csv";
        string port2_log_path = folder_path + "/kt_log.csv";
        BufferedLogger logger1(port1_log_path);
        BufferedLogger logger2(port2_log_path);
        
        // 서버 TURN 송신용 libnice 에이전트 스레드 실행
        thread turn_sender_thread(server_turn_sender_thread);
        // UDP 패킷 수신 스레드 (두 포트)
        thread port1_thread(receive_packets, SERVER_REG_PORT, ref(logger1));
        thread port2_thread(receive_packets, SERVER_PORT2, ref(logger2));
        
        port1_thread.join();
        port2_thread.join();
        turn_sender_thread.join();
    } catch (const exception& e) {
        cerr << "Server error: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
