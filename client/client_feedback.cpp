// client_feedback.cpp
#include <iostream>
#include <fstream>
#include <atomic>
#include <thread>
#include <vector>
#include <future>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <arpa/inet.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <unistd.h>
#include <filesystem>
#include <cerrno>
#include <cstring>
#include <netdb.h> // NI_MAXHOST 정의를 위해
#include "config.h"

// OpenSSL (HMAC, EVP)
#include <openssl/hmac.h>
#include <openssl/evp.h>

// FFmpeg
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

// libnice / GLib (TURN ACK 수신용)
#include <nice/agent.h>
#include <glib.h>

using namespace std;
namespace fs = std::filesystem;

// 만약 NICE_RELAY_TURN가 정의되어 있지 않으면 NICE_RELAY_TYPE_TURN_TLS로 정의
#ifndef NICE_RELAY_TURN
#define NICE_RELAY_TURN NICE_RELAY_TYPE_TURN_TLS
#endif

//─────────────────────────────────────────────────────────────────────────────
// Base64 인코더 및 HMAC-SHA1, TURN username/password 생성 함수들
static const char* base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const std::string& input)
{
    std::ostringstream out;
    int val = 0, valb = -6;
    for (unsigned char c : input) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out << base64_chars[(val >> valb) & 0x3F];
            valb -= 6;
        }
    }
    if (valb > -6)
        out << base64_chars[((val << 8) >> (valb + 8)) & 0x3F];
    while (out.tellp() % 4)
        out << '=';
    return out.str();
}

std::string compute_turn_password_bin(const std::string& data, const std::string& secret) {
    unsigned char hmac_result[EVP_MAX_MD_SIZE];
    unsigned int hmac_length = 0;
    memset(hmac_result, 0, sizeof(hmac_result));
    HMAC(EVP_sha1(),
         secret.c_str(), secret.size(),
         reinterpret_cast<const unsigned char*>(data.c_str()), data.size(),
         hmac_result, &hmac_length);
    return std::string(reinterpret_cast<const char*>(hmac_result), hmac_length);
}

std::string compute_turn_password(const std::string& user_with_timestamp_and_colon,
                                  const std::string& realm,
                                  const std::string& secret)
{
    std::string key = user_with_timestamp_and_colon + ":" + realm;
    std::string raw_hmac = compute_turn_password_bin(key, secret);
    return base64_encode(raw_hmac);
}

std::string generate_turn_username(const std::string& identifier, uint32_t validSeconds) {
    uint64_t expiration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() + validSeconds;
    return std::to_string(expiration) + ":" + identifier;
}

//─────────────────────────────────────────────────────────────────────────────
// 로깅 클래스
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
    void log(const string& msg) {
        lock_guard<mutex> lock(m);
        log_stream << msg << "\n";
    }
    void flush() {
        lock_guard<mutex> lock(m);
        log_stream.flush();
    }
private:
    ofstream log_stream;
    mutex m;
};

//─────────────────────────────────────────────────────────────────────────────
// 패킷 헤더
struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};

//─────────────────────────────────────────────────────────────────────────────
// VideoStreamer: FFmpeg 인코딩 + UDP 전송 (영상 전송은 그대로)
class VideoStreamer {
public:
    VideoStreamer() : frame_counter(0), sequence_number(0) {
        create_and_set_output_folders();
        string logfile = logs_folder + "/packet_log.csv";
        logger = make_unique<BufferedLogger>(logfile);
        logger->log("sequence_number,pts,size,timestamp_frame,timestamp_sending,encoding_latency,frame_type");

        codec = avcodec_find_encoder_by_name("libx264");
        if (!codec) throw runtime_error("Codec 'libx264' not found");
        codec_ctx.reset(avcodec_alloc_context3(codec));
        if (!codec_ctx) throw runtime_error("Could not allocate codec context");
        codec_ctx->width = WIDTH;
        codec_ctx->height = HEIGHT;
        codec_ctx->time_base = {1, FPS};
        codec_ctx->framerate = {FPS, 1};
        codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        codec_ctx->max_b_frames = 0;
        codec_ctx->gop_size = 10;

        AVDictionary* opt = nullptr;
        av_dict_set(&opt, "preset", "veryfast", 0);
        av_dict_set(&opt, "tune", "zerolatency", 0);
        av_dict_set(&opt, "crf", "26", 0);
        string xparam = "keyint=10:min-keyint=10:scenecut=0:bframes=0:"
                        "force-cfr=1:rc-lookahead=0:ref=1:sliced-threads=0:"
                        "aq-mode=1:trellis=0:psy-rd=1.0:1.0";
        av_dict_set(&opt, "x264-params", xparam.c_str(), 0);
        if (avcodec_open2(codec_ctx.get(), codec, &opt) < 0)
            throw runtime_error("Could not open codec");
        av_dict_free(&opt);

        frame.reset(av_frame_alloc());
        if (!frame) throw runtime_error("Could not allocate frame");
        frame->format = codec_ctx->pix_fmt;
        frame->width = codec_ctx->width;
        frame->height = codec_ctx->height;
        if (av_frame_get_buffer(frame.get(), 32) < 0)
            throw runtime_error("Could not allocate frame data");

        pkt.reset(av_packet_alloc());
        if (!pkt) throw runtime_error("Could not allocate AVPacket");

        sws_ctx = sws_getContext(WIDTH, HEIGHT, AV_PIX_FMT_YUYV422,
                                 WIDTH, HEIGHT, AV_PIX_FMT_YUV420P,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx) throw runtime_error("Could not initialize sws context");

        sockfd1 = create_socket_and_bind(INTERFACE1_IP, INTERFACE1_NAME);
        sockfd2 = create_socket_and_bind(INTERFACE2_IP, INTERFACE2_NAME);
        servaddr1 = create_sockaddr(SERVER_IP, SERVER_PORT);
        servaddr2 = create_sockaddr(SERVER_IP, SERVER_PORT + 1);
    }
    ~VideoStreamer() {
        if (sockfd1 >= 0) close(sockfd1);
        if (sockfd2 >= 0) close(sockfd2);
        if (sws_ctx) sws_freeContext(sws_ctx);
    }
    void stream(rs2::video_frame &color_frame, uint64_t ts) {
        frame->pts = frame_counter++;
        uint8_t* yuyv = (uint8_t*)color_frame.get_data();
        const uint8_t* src_slices[1] = { yuyv };
        int src_stride[1] = { 2 * WIDTH };
        if (sws_scale(sws_ctx, src_slices, src_stride,
                      0, HEIGHT, frame->data, frame->linesize) < 0) {
            cerr << "Error in sws_scale\n";
            return;
        }
        encode_and_send_frame(ts);
    }
private:
    void create_and_set_output_folders() {
        auto now = chrono::system_clock::now();
        time_t tnow = chrono::system_clock::to_time_t(now);
        tm local_t = *localtime(&tnow);
        char folder_name[100];
        strftime(folder_name, sizeof(folder_name), "%Y_%m_%d_%H_%M", &local_t);
        string base = SAVE_FILEPATH + string(folder_name);
        fs::create_directories(base);
        frames_folder = base + "/frames";
        logs_folder = base + "/logs";
        fs::create_directories(frames_folder);
        fs::create_directories(logs_folder);
    }
    sockaddr_in create_sockaddr(const char* ip, int port) {
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = inet_addr(ip);
        return addr;
    }
    int create_socket_and_bind(const char* ip, const char* if_name) {
        int s = socket(AF_INET, SOCK_DGRAM, 0);
        if (s < 0) throw runtime_error("socket fail");
        if (setsockopt(s, SOL_SOCKET, SO_BINDTODEVICE, if_name, strlen(if_name)) < 0) {
            close(s);
            throw runtime_error("SO_BINDTODEVICE fail: " + string(if_name));
        }
        sockaddr_in bindaddr = create_sockaddr(ip, 0);
        if (bind(s, (struct sockaddr*)&bindaddr, sizeof(bindaddr)) < 0) {
            close(s);
            throw runtime_error("Bind fail on " + string(ip));
        }
        return s;
    }
    void encode_and_send_frame(uint64_t ts) {
        if (avcodec_send_frame(codec_ctx.get(), frame.get()) < 0) {
            cerr << "Error sending frame\n";
            return;
        }
        int ret;
        while ((ret = avcodec_receive_packet(codec_ctx.get(), pkt.get())) == 0) {
            PacketHeader hdr;
            hdr.timestamp_frame = ts;
            hdr.timestamp_sending = chrono::duration_cast<chrono::microseconds>(
                chrono::system_clock::now().time_since_epoch()).count();
            hdr.sequence_number = sequence_number++;
            double enc_lat = (hdr.timestamp_sending - hdr.timestamp_frame) / 1000.0;
            string frame_type = (pkt->flags & AV_PKT_FLAG_KEY) ? "I-frame" : "P-frame";
            vector<uint8_t> data(sizeof(PacketHeader) + pkt->size);
            memcpy(data.data(), &hdr, sizeof(PacketHeader));
            memcpy(data.data() + sizeof(PacketHeader), pkt->data, pkt->size);
            ostringstream logMsg;
            logMsg << hdr.sequence_number << "," << frame->pts << "," << pkt->size << ","
                   << hdr.timestamp_frame << "," << hdr.timestamp_sending << ","
                   << enc_lat << "," << frame_type;
            logger->log(logMsg.str());
            // 영상 데이터는 UDP로 전송 (변경 없음)
            auto t1 = async(launch::async, [this, &data]() {
                if (sendto(sockfd1, data.data(), data.size(), 0,
                           (struct sockaddr*)&servaddr1, sizeof(servaddr1)) < 0)
                    cerr << "Send error(if1): " << strerror(errno) << "\n";
            });
            auto t2 = async(launch::async, [this, &data]() {
                if (sendto(sockfd2, data.data(), data.size(), 0,
                           (struct sockaddr*)&servaddr2, sizeof(servaddr2)) < 0)
                    cerr << "Send error(if2): " << strerror(errno) << "\n";
            });
            t1.get();
            t2.get();
            av_packet_unref(pkt.get());
        }
        if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF)
            cerr << "Error receiving encoded packet\n";
    }
    const AVCodec* codec;
    unique_ptr<AVCodecContext, void(*)(AVCodecContext*)> codec_ctx{
        nullptr, [](AVCodecContext* p) { avcodec_free_context(&p); }
    };
    unique_ptr<AVFrame, void(*)(AVFrame*)> frame{
        nullptr, [](AVFrame* p) { av_frame_free(&p); }
    };
    unique_ptr<AVPacket, void(*)(AVPacket*)> pkt{
        nullptr, [](AVPacket* p) { av_packet_free(&p); }
    };
    SwsContext* sws_ctx = nullptr;
    int sockfd1 = -1, sockfd2 = -1;
    sockaddr_in servaddr1, servaddr2;
    unique_ptr<BufferedLogger> logger;
    string frames_folder, logs_folder;
    atomic<int> frame_counter;
    atomic<int> sequence_number;
};

//─────────────────────────────────────────────────────────────────────────────
// libnice TURN ACK 수신 (클라이언트용)
static NiceAgent *g_nice_agent = nullptr;
static GMainLoop *g_main_loop = nullptr;
static guint g_stream_id = 0;
static atomic<bool> nice_turn_ready(false);

static void cb_candidate_gathering_done(NiceAgent *agent, guint stream_id, gpointer user_data) {
    g_print("Candidate gathering done for stream %u\n", stream_id);
    // 모든 로컬 후보 목록 출력
    GSList *candidates = nice_agent_get_local_candidates(agent, stream_id, 1);
    for (GSList *iter = candidates; iter; iter = iter->next) {
        NiceCandidate *cand = (NiceCandidate*)iter->data;
        char ip[NI_MAXHOST];
        nice_address_to_string(&cand->addr, ip);
        int port = nice_address_get_port(&cand->addr);
        g_print("Local candidate: %s:%d, type=%d\n", ip, port, cand->type);
    }
    g_slist_free_full(candidates, (GDestroyNotify)&nice_candidate_free);
}

static void cb_component_state_changed(NiceAgent *agent, guint stream_id, guint component_id, guint state, gpointer user_data) {
    g_print("Component %u state changed to %u\n", component_id, state);
    if (state == NICE_COMPONENT_STATE_READY) {
        nice_turn_ready = true;
        // 로컬 TURN candidate 정보를 추출하여 서버에 등록
        GSList *lcandidates = nice_agent_get_local_candidates(agent, stream_id, component_id);
        if (lcandidates) {
            NiceCandidate *candidate = (NiceCandidate*)lcandidates->data;
            char ip[NI_MAXHOST];
            nice_address_to_string(&candidate->addr, ip);
            int port = nice_address_get_port(&candidate->addr);
            g_print("Local TURN candidate: %s:%d\n", ip, port);
            // UDP로 TURN_REG 메시지 전송
            int sock = socket(AF_INET, SOCK_DGRAM, 0);
            if (sock < 0) {
                cerr << "Failed to create UDP socket for TURN_REG\n";
                return;
            }
            sockaddr_in server_addr = {};
            server_addr.sin_family = AF_INET;
            server_addr.sin_port = htons(SERVER_REG_PORT);
            server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
            ostringstream oss;
            oss << "TURN_REG:" << ip << ":" << port;
            string reg_msg = oss.str();
            if (sendto(sock, reg_msg.c_str(), reg_msg.size(), 0, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0)
                cerr << "Failed to send TURN_REG: " << strerror(errno) << "\n";
            else
                cout << "Sent TURN_REG message: " << reg_msg << "\n";
            close(sock);
            g_slist_free_full(lcandidates, (GDestroyNotify)&nice_candidate_free);
        }
    }
}

// 기존의 cb_data_received는 더 이상 g_signal_connect()로 등록하지 않습니다.
static gboolean cb_data_received(NiceAgent *agent, guint stream_id, guint component_id,
                                 guint len, gchar *buf, gpointer user_data) {
    string ack(buf, len);
    cout << "[TURN ACK Received] " << ack << endl;
    return TRUE;
}

// TURN 데이터를 수신하기 위해 nice_agent_recv()를 폴링하는 스레드 함수
static void poll_turn_data() {
    const int buf_size = 2048;
    char buffer[buf_size];
    while (g_main_loop_is_running(g_main_loop)) {
        // nice_agent_recv의 시그니처: (NiceAgent*, guint, guint, guint8*, gsize, GCancellable*, GError**)
        gssize ret = nice_agent_recv(g_nice_agent, g_stream_id, 1,
                                     reinterpret_cast<guint8*>(buffer),
                                     static_cast<gsize>(buf_size),
                                     nullptr, nullptr);
        if (ret > 0) {
            string ack(buffer, ret);
            cout << "[TURN ACK Received] " << ack << endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

static void nice_turn_receiver_thread() {
    g_main_loop = g_main_loop_new(NULL, FALSE);
    GMainContext *context = g_main_loop_get_context(g_main_loop);
    g_nice_agent = nice_agent_new(context, NICE_COMPATIBILITY_RFC5245);
    if (!g_nice_agent) {
        cerr << "Failed to create NiceAgent\n";
        return;
    }
    g_stream_id = nice_agent_add_stream(g_nice_agent, 1);
    string ephemeral_username = generate_turn_username(TURN_IDENTIFIER, TURN_VALID_SECONDS);
    string ephemeral_password = compute_turn_password(ephemeral_username, TURN_REALM, TURN_SECRET);
    if (!nice_agent_set_relay_info(g_nice_agent, g_stream_id, 1,
                                   TURN_SERVER_IP, TURN_SERVER_PORT,
                                   ephemeral_username.c_str(), ephemeral_password.c_str(),
                                   NICE_RELAY_TURN)) {
        cerr << "Failed to set TURN/relay info\n";
        return;
    }
    g_signal_connect(G_OBJECT(g_nice_agent), "candidate-gathering-done", G_CALLBACK(cb_candidate_gathering_done), NULL);
    g_signal_connect(G_OBJECT(g_nice_agent), "component-state-changed", G_CALLBACK(cb_component_state_changed), NULL);
    
    // TURN 데이터 수신은 별도의 폴링 스레드를 통해 진행합니다.
    if (!nice_agent_gather_candidates(g_nice_agent, g_stream_id)) {
        cerr << "Failed to start candidate gathering\n";
        return;
    }
    g_print("Running Nice TURN ack receiver...\n");
    
    thread poll_thread(poll_turn_data);
    
    g_main_loop_run(g_main_loop);
    poll_thread.join();
    g_object_unref(g_nice_agent);
    g_main_loop_unref(g_main_loop);
}

void client_stream(VideoStreamer &streamer, rs2::pipeline &pipe, atomic<bool> &running) {
    while (running.load()) {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) continue;
        uint64_t ts = chrono::duration_cast<chrono::microseconds>(
            chrono::system_clock::now().time_since_epoch()).count();
        streamer.stream(color_frame, ts);
    }
}

int main() {
    try {
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_YUYV, FPS);
        pipe.start(cfg);
        VideoStreamer streamer;
        atomic<bool> running(true);
        thread client_thread(client_stream, ref(streamer), ref(pipe), ref(running));
        // libnice TURN ACK 수신 스레드 실행
        thread turn_thread(nice_turn_receiver_thread);
        cout << "Press Enter to stop streaming..." << endl;
        cin.get();
        running.store(false);
        g_main_loop_quit(g_main_loop);
        client_thread.join();
        turn_thread.join();
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
