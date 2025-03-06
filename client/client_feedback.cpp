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
#include "config.h"   // WIDTH, HEIGHT, FPS, SERVER_IP, SERVER_PORT, etc.
#include "turn_rest_util.h"  // Ephemeral username/password

// OpenSSL
#include <openssl/hmac.h>
#include <openssl/evp.h>

// FFmpeg
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

// pjnath, pjlib
extern "C" {
#include <pjlib.h>
#include <pjlib-util.h>
#include <pjnath.h>
}

using namespace std;
namespace fs = std::filesystem;

/* 
   config.h 에 들어있는 매크로 예시:
   #define WIDTH 640
   #define HEIGHT 480
   #define FPS 30
   #define SERVER_IP "1.2.3.4"
   #define SERVER_PORT 5000
   #define INTERFACE1_IP "192.168.0.10"
   #define INTERFACE1_NAME "eth0"
   #define INTERFACE2_IP "192.168.10.20"
   #define INTERFACE2_NAME "eth1"

   // TURN 관련
   #define TURN_SERVER_IP "121.128.220.205"
   #define TURN_SERVER_PORT 3478
   #define TURN_REALM "v2n2v"
   #define TURN_SECRET "v2n2v123"
   #define TURN_IDENTIFIER "client_id"
   #define TURN_VALID_SECONDS 3600

   // 저장용
   #define SAVE_FILEPATH "/home/user/recorded/"
*/

// --------------------- 전역: TURN Ephemeral username/password ---------------------
static std::string g_ephemeral_username;
static std::string g_ephemeral_password_bin; 

// --------------------- PacketHeader ---------------------
#pragma pack(push, 1)
struct PacketHeader {
    uint64_t timestamp_frame;   // 캡처 시각(μs)
    uint64_t timestamp_sending; // 송신 시각(μs)
    uint32_t sequence_number;
};
#pragma pack(pop)

// --------------------- 로거 ---------------------
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

// --------------------- VideoStreamer ---------------------
class VideoStreamer {
public:
    VideoStreamer() : frame_counter(0), sequence_number(0) {
        create_and_set_output_folders();

        string logfile = logs_folder + "/packet_log.csv";
        logger = make_unique<BufferedLogger>(logfile);
        logger->log("sequence_number,pts,size,timestamp_frame,timestamp_sending,encoding_latency,frame_type");

        // FFmpeg H.264(libx264) 초기화
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

        // libx264 옵션
        AVDictionary* opt = nullptr;
        av_dict_set(&opt, "preset", "veryfast", 0);
        av_dict_set(&opt, "tune", "zerolatency", 0);
        av_dict_set(&opt, "crf", "26", 0);
        // 키프레임 주기
        string xparam = "keyint=10:min-keyint=10:scenecut=0:bframes=0:"
                        "force-cfr=1:rc-lookahead=0:ref=1:sliced-threads=0:"
                        "aq-mode=1:trellis=0:psy-rd=1.0:1.0";
        av_dict_set(&opt, "x264-params", xparam.c_str(), 0);

        if (avcodec_open2(codec_ctx.get(), codec, &opt) < 0) {
            throw runtime_error("Could not open codec");
        }
        av_dict_free(&opt);

        // 프레임 alloc
        frame.reset(av_frame_alloc());
        if (!frame) throw runtime_error("Could not allocate frame");
        frame->format = codec_ctx->pix_fmt;
        frame->width  = codec_ctx->width;
        frame->height = codec_ctx->height;
        if (av_frame_get_buffer(frame.get(), 32) < 0)
            throw runtime_error("Could not allocate frame data");

        pkt.reset(av_packet_alloc());
        if (!pkt) throw runtime_error("Could not allocate AVPacket");

        sws_ctx = sws_getContext(WIDTH, HEIGHT, AV_PIX_FMT_YUYV422,
                                 WIDTH, HEIGHT, AV_PIX_FMT_YUV420P,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx) throw runtime_error("Could not initialize sws context");

        // UDP 소켓 2개 (두 인터페이스에 바인딩)
        sockfd1 = create_socket_and_bind(INTERFACE1_IP, INTERFACE1_NAME);
        sockfd2 = create_socket_and_bind(INTERFACE2_IP, INTERFACE2_NAME);

        // 서버 주소
        servaddr1 = create_sockaddr(SERVER_IP, SERVER_PORT);
        servaddr2 = create_sockaddr(SERVER_IP, SERVER_PORT + 1); // (예시로 +1)
    }

    ~VideoStreamer() {
        if (sockfd1 >= 0) close(sockfd1);
        if (sockfd2 >= 0) close(sockfd2);
        if (sws_ctx) sws_freeContext(sws_ctx);
    }

    void stream(rs2::video_frame &color_frame, uint64_t ts) {
        frame->pts = frame_counter++;

        // YUYV -> YUV420
        uint8_t* yuyv = (uint8_t*)color_frame.get_data();
        const uint8_t* src_slices[1] = { yuyv };
        int src_stride[1] = { 2 * WIDTH };

        if (sws_scale(sws_ctx, src_slices, src_stride,
                      0, HEIGHT, frame->data, frame->linesize) < 0) {
            cerr << "Error in sws_scale\n";
            return;
        }
        // 인코딩 후 UDP 전송
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
        logs_folder   = base + "/logs";
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

        // 특정 인터페이스에 바인딩
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
            // 헤더
            PacketHeader hdr;
            hdr.timestamp_frame   = ts;
            hdr.timestamp_sending = chrono::duration_cast<chrono::microseconds>(
                chrono::system_clock::now().time_since_epoch()).count();
            hdr.sequence_number   = sequence_number++;

            double enc_lat = (hdr.timestamp_sending - hdr.timestamp_frame) / 1000.0;
            string frame_type = (pkt->flags & AV_PKT_FLAG_KEY) ? "I-frame" : "P-frame";

            // 전송용 버퍼: [PacketHeader + H.264 ES]
            vector<uint8_t> data(sizeof(PacketHeader) + pkt->size);
            memcpy(data.data(), &hdr, sizeof(PacketHeader));
            memcpy(data.data() + sizeof(PacketHeader), pkt->data, pkt->size);

            // 로깅
            ostringstream logMsg;
            logMsg << hdr.sequence_number << "," << frame->pts << "," << pkt->size << ","
                   << hdr.timestamp_frame << "," << hdr.timestamp_sending << ","
                   << enc_lat << "," << frame_type;
            logger->log(logMsg.str());

            // 두 인터페이스에 비동기로 sendto()
            auto t1 = async(launch::async, [this, &data]() {
                if (sendto(sockfd1, data.data(), data.size(), 0,
                           (struct sockaddr*)&servaddr1, sizeof(servaddr1)) < 0) {
                    cerr << "Send error(if1): " << strerror(errno) << "\n";
                }
            });
            auto t2 = async(launch::async, [this, &data]() {
                if (sendto(sockfd2, data.data(), data.size(), 0,
                           (struct sockaddr*)&servaddr2, sizeof(servaddr2)) < 0) {
                    cerr << "Send error(if2): " << strerror(errno) << "\n";
                }
            });
            t1.get();
            t2.get();

            av_packet_unref(pkt.get());
        }
        if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
            cerr << "Error receiving encoded packet\n";
        }
    }

private:
    const AVCodec* codec = nullptr;
    unique_ptr<AVCodecContext, void(*)(AVCodecContext*)> codec_ctx{
        nullptr, [](AVCodecContext* p){ avcodec_free_context(&p); }
    };
    unique_ptr<AVFrame, void(*)(AVFrame*)> frame{
        nullptr, [](AVFrame* p){ av_frame_free(&p); }
    };
    unique_ptr<AVPacket, void(*)(AVPacket*)> pkt{
        nullptr, [](AVPacket* p){ av_packet_free(&p); }
    };
    SwsContext* sws_ctx = nullptr;

    int sockfd1 = -1, sockfd2 = -1;
    sockaddr_in servaddr1, servaddr2;

    unique_ptr<BufferedLogger> logger;
    string frames_folder, logs_folder;

    atomic<int> frame_counter;
    atomic<int> sequence_number;
};

// --------------------- pjnath TURN 콜백 ---------------------
static pj_bool_t on_data_sent_cb(pj_turn_sock *sock, pj_ssize_t sent) {
    if (sent < 0) {
        pj_status_t err = (pj_status_t)(-sent);
        PJ_LOG(1, ("TURN", "Data send err: %d", err));
    } else {
        PJ_LOG(4, ("TURN", "Data sent: %zd bytes", (ssize_t)sent));
    }
    return PJ_TRUE;
}
static void on_rx_data_cb(pj_turn_sock *turn_sock,
                          void *pkt, unsigned pkt_len,
                          const pj_sockaddr_t *peer_addr,
                          unsigned addr_len)
{
    // 서버로부터 온 ACK 등을 수신
    // 단순히 콘솔에 출력
    if (pkt_len > 0) {
        std::string msg((char*)pkt, (char*)pkt + pkt_len);
        PJ_LOG(3, ("TURN", "TURN on_rx_data: %u bytes --> %s", pkt_len, msg.c_str()));
        // 필요하다면 여기서 ACK 파싱 가능
    }
}
static pj_status_t on_connection_attempt_cb(pj_turn_sock *turn_sock,
                                            pj_uint32_t conn_id,
                                            const pj_sockaddr_t *peer_addr,
                                            unsigned addr_len)
{
    PJ_LOG(3, ("TURN", "TCP connection attempt (RFC6062)"));
    return PJ_SUCCESS;
}
static void on_connection_status_cb(pj_turn_sock *turn_sock,
                                    pj_status_t status,
                                    pj_uint32_t conn_id,
                                    const pj_sockaddr_t *peer_addr,
                                    unsigned addr_len)
{
    PJ_LOG(3, ("TURN", "Connection status: %d", status));
}
static void on_state_cb(pj_turn_sock *turn_sock,
                        pj_turn_state_t old_state,
                        pj_turn_state_t new_state)
{
    PJ_LOG(3, ("TURN", "state changed: %d->%d", old_state, new_state));
    if (new_state == PJ_TURN_STATE_READY) {
        pj_turn_session_info info;
        if (pj_turn_sock_get_info(turn_sock, &info) == PJ_SUCCESS) {
            if (info.relay_addr.ipv4.sin_family == PJ_AF_INET) {
                char relay_ip[64];
                pj_inet_ntop(PJ_AF_INET, &info.relay_addr.ipv4.sin_addr, relay_ip, sizeof(relay_ip));
                pj_uint16_t rport = pj_ntohs(info.relay_addr.ipv4.sin_port);
                PJ_LOG(3, ("TURN", "Relay(IPv4): %s:%d", relay_ip, rport));

                // 클라이언트는 이 Relay 주소를 서버에게 전달(등록)해야,
                // 서버가 ACK를 해당 주소로 sendto() 가능.
                // 간단히 UDP 메시지로 "TURN_REG:<relay_ip>:<rport>" 전송
                // => 기존 server.cpp에선 이 메시지를 받아서 'client_turn_addr' 등록
                pj_sockaddr_in dest;
                pj_bzero(&dest, sizeof(dest));
                dest.sin_family = PJ_AF_INET;
                dest.sin_addr.s_addr = pj_inet_addr(pj_cstr(&dest.sin_addr, SERVER_IP));
                dest.sin_port = pj_htons(SERVER_PORT);

                // 예: "TURN_REG:1.2.3.4:12345"
                char reg_msg[100];
                snprintf(reg_msg, sizeof(reg_msg), "TURN_REG:%s:%d", relay_ip, (int)rport);

                pj_ssize_t send_len = (pj_ssize_t)strlen(reg_msg);
                pj_status_t st = pj_turn_sock_sendto(turn_sock,
                                                     reg_msg, send_len,
                                                     0,
                                                     (pj_sockaddr_t*)&dest,
                                                     sizeof(dest));
                PJ_LOG(3, ("TURN", "Send TURN_REG to server, status=%d", st));
            }
        }
    }
}

static pj_turn_sock_cb turn_callbacks = {
    &on_rx_data_cb,
    &on_data_sent_cb,
    &on_state_cb,
    &on_connection_attempt_cb,
    &on_connection_status_cb
};

// --------------------- TURN 전역 ---------------------
static pj_caching_pool g_cp;
static pj_pool_t*      g_pool       = nullptr;
static pj_ioqueue_t*   g_ioqueue    = nullptr;
static pj_timer_heap_t* g_timer_heap= nullptr;
static pj_turn_sock*   turn_sock    = nullptr;

static atomic<bool> turn_running(true);

// --------------------- TURN 세션 스레드 ---------------------
static void turn_ack_receiver_thread()
{
    pj_status_t status;

    // pjlib 초기화
    status = pj_init(); 
    if (status != PJ_SUCCESS) {
        PJ_LOG(1, ("TURN", "pj_init err=%d", status));
        return;
    }
    status = pjlib_util_init();
    if (status != PJ_SUCCESS) {
        PJ_LOG(1, ("TURN", "pjlib_util_init err=%d", status));
        pj_shutdown();
        return;
    }

    pj_caching_pool_init(&g_cp, NULL, 0);
    g_pool = pj_pool_create(&g_cp.factory, "turn_pool", 4000, 4000, NULL);

    // ioqueue, timer_heap
    status = pj_ioqueue_create(g_pool, 32, &g_ioqueue);
    if (status != PJ_SUCCESS) {
        PJ_LOG(1, ("TURN", "pj_ioqueue_create err=%d", status));
        goto on_error;
    }
    status = pj_timer_heap_create(g_pool, 32, &g_timer_heap);
    if (status != PJ_SUCCESS) {
        PJ_LOG(1, ("TURN", "pj_timer_heap_create err=%d", status));
        goto on_error;
    }

    // STUN config
    pj_stun_config stun_cfg;
    pj_stun_config_init(&stun_cfg, &g_cp.factory, 0, g_ioqueue, g_timer_heap);

    // TURN 소켓 cfg
    pj_turn_sock_cfg tcfg;
    pj_turn_sock_cfg_default(&tcfg);

    // turn_sock create
    status = pj_turn_sock_create(&stun_cfg,
                                 PJ_AF_INET,
                                 PJ_TURN_TP_UDP,
                                 &turn_callbacks,
                                 &tcfg,
                                 NULL,
                                 &turn_sock);
    if (status != PJ_SUCCESS) {
        PJ_LOG(1, ("TURN", "pj_turn_sock_create err=%d", status));
        goto on_error;
    }

    // Ephemeral username/password
    if (g_ephemeral_username.empty()) {
        // username = "<expiry>:<identifier>"
        g_ephemeral_username = generate_ephemeral_username(TURN_IDENTIFIER, TURN_VALID_SECONDS);
        // password(raw binary) = HMAC-SHA1("<username>:<realm>", secret)
        g_ephemeral_password_bin = hmac_sha1_raw(
            g_ephemeral_username + string(":") + TURN_REALM,
            TURN_SECRET
        );
    }

    // (pjnath에서) credential
    pj_stun_auth_cred auth_cred;
    pj_bzero(&auth_cred, sizeof(auth_cred));
    auth_cred.type = PJ_STUN_AUTH_CRED_STATIC;
    auth_cred.data.static_cred.realm = pj_str(const_cast<char*>(TURN_REALM));
    auth_cred.data.static_cred.username = pj_str(const_cast<char*>(g_ephemeral_username.c_str()));
    // password는 base64가 아닌 "plaintext"로 준다고 선언
    // => coturn이 use-auth-secret인 경우, 실제로는 ephemeral base64 password를 넣어야 할 수도 있으나
    //    pjnath 의 "plain" 설정시, TURN 서버가 "일반적인 long-term" 인증 과정을 할 수 있음.
    //    *단*, coturn --use-auth-secret + static-auth-secret인 경우엔, 
    //     "username:realm" 의 HMAC-SHA1 결과를 base64로 한 것이 "plain password"로 사용됨.
    {
        // ephemeral_pass_ascii = base64(HMAC(...)) 
        std::string ephemeral_pass_ascii = generate_ephemeral_password(g_ephemeral_username, TURN_REALM, TURN_SECRET);
        auth_cred.data.static_cred.data_type = PJ_STUN_PASSWD_PLAIN;
        auth_cred.data.static_cred.data = pj_str(const_cast<char*>(ephemeral_pass_ascii.c_str()));
    }

    // TURN Allocate
    {
        pj_str_t server = pj_str(const_cast<char*>(TURN_SERVER_IP));
        status = pj_turn_sock_alloc(turn_sock,
                                    &server,
                                    TURN_SERVER_PORT,
                                    NULL, 
                                    &auth_cred,
                                    NULL);
        if (status != PJ_SUCCESS) {
            PJ_LOG(1, ("TURN", "pj_turn_sock_alloc err=%d", status));
            goto on_error;
        }
    }

    PJ_LOG(3, ("TURN", "TURN ack receiver thread started."));

    // 메인 루프
    while (turn_running.load()) {
        pj_time_val delay = {0, 10};
        pj_ioqueue_poll(g_ioqueue, &delay);
        pj_timer_heap_poll(g_timer_heap, NULL);
        pj_thread_sleep(10);
    }

on_error:;
    if (turn_sock) {
        pj_turn_sock_destroy(turn_sock);
        turn_sock = nullptr;
    }
    if (g_ioqueue) {
        pj_ioqueue_destroy(g_ioqueue);
        g_ioqueue = nullptr;
    }
    if (g_timer_heap) {
        pj_timer_heap_destroy(g_timer_heap);
        g_timer_heap = nullptr;
    }
    if (g_pool) {
        pj_pool_release(g_pool);
        g_pool = nullptr;
    }
    pj_caching_pool_destroy(&g_cp);
    pj_shutdown();
}

// --------------------- main ---------------------
int main()
{
    try {
        // RealSense 파이프라인
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_YUYV, FPS);
        pipe.start(cfg);

        VideoStreamer streamer;

        atomic<bool> running(true);
        thread client_thread([&](){
            while(running.load()) {
                rs2::frameset frames = pipe.wait_for_frames();
                rs2::video_frame color_frame = frames.get_color_frame();
                if (!color_frame) continue;

                uint64_t ts = chrono::duration_cast<chrono::microseconds>(
                    chrono::system_clock::now().time_since_epoch()).count();

                // 전송
                streamer.stream(color_frame, ts);
            }
        });

        // TURN 세션 시작 (ACK 수신용)
        thread turn_thread(turn_ack_receiver_thread);

        cout << "Press Enter to stop streaming..." << endl;
        cin.get();

        running.store(false);
        turn_running.store(false);

        client_thread.join();
        turn_thread.join();

    } catch(const exception &e) {
        cerr << "[Client] Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
