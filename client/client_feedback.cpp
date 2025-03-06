// client.cpp
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
#include "config.h"  // 여기에 WIDTH, HEIGHT, FPS, SERVER_IP, SERVER_PORT 등 매크로 정의

// OpenSSL (HMAC 계산)
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

//
// BufferedLogger
//
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

//
// PacketHeader (영상 전송 시 헤더)
//
struct PacketHeader {
    uint64_t timestamp_frame;    // 프레임 캡처 시각(µs)
    uint64_t timestamp_sending;  // 전송 시각(µs)
    uint32_t sequence_number;
};

//
// TURN 관련 헬퍼 함수
//
std::string generate_turn_username(const std::string& identifier, uint32_t validSeconds) {
    uint64_t expiration = chrono::duration_cast<chrono::seconds>(
                              chrono::system_clock::now().time_since_epoch()).count() + validSeconds;
    return to_string(expiration) + ":" + identifier;
}

std::string compute_turn_password(const std::string& data, const std::string& secret) {
    unsigned char hmac_result[EVP_MAX_MD_SIZE] = {0};
    unsigned int hmac_length = 0;
    HMAC(EVP_sha1(),
         secret.c_str(), secret.size(),
         reinterpret_cast<const unsigned char*>(data.c_str()), data.size(),
         hmac_result, &hmac_length);
    ostringstream oss;
    for (unsigned int i = 0; i < hmac_length; i++) {
        oss << hex << setw(2) << setfill('0') << (int)hmac_result[i];
    }
    return oss.str();
}

//
// TURN 콜백 정의
//
static pj_bool_t on_data_sent_cb(pj_turn_sock *sock, pj_ssize_t sent)
{
    if (sent < 0) {
        pj_status_t err = (pj_status_t)(-sent);  // 음수 → 오류 코드
        PJ_LOG(1,("TURN", "Data send error: %d", err));
    } else {
        PJ_LOG(4,("TURN", "Data sent: %zd bytes", (ssize_t)sent));
    }
    return PJ_TRUE; 
}

static void on_rx_data_cb(pj_turn_sock *turn_sock,
                          void *pkt,
                          unsigned pkt_len,
                          const pj_sockaddr_t *peer_addr,
                          unsigned addr_len)
{
    PJ_LOG(4,("TURN", "TURN on_rx_data: got %u bytes from peer", pkt_len));
    // 필요 시 여기서 처리
}

static pj_status_t on_connection_attempt_cb(pj_turn_sock *turn_sock,
                                            pj_uint32_t conn_id,
                                            const pj_sockaddr_t *peer_addr,
                                            unsigned addr_len)
{
    PJ_LOG(3,("TURN", "TCP connection attempt from peer! (RFC6062)"));
    return PJ_SUCCESS;
}

static void on_connection_status_cb(pj_turn_sock *turn_sock,
                                    pj_status_t status,
                                    pj_uint32_t conn_id,
                                    const pj_sockaddr_t *peer_addr,
                                    unsigned addr_len)
{
    PJ_LOG(3,("TURN", "Connection status to peer: %d", status));
}

// state 변화 콜백
static void on_state_cb(pj_turn_sock *turn_sock,
                        pj_turn_state_t old_state,
                        pj_turn_state_t new_state)
{
    PJ_LOG(3,("TURN", "TURN state changed: %d --> %d", old_state, new_state));

    if (new_state == PJ_TURN_STATE_READY) {
        // Relay 주소 얻기
        pj_turn_session_info info;
        if (pj_turn_sock_get_info(turn_sock, &info) == PJ_SUCCESS) {
            char relay_ip[48];
            int family = pj_sockaddr_get_af(&info.relay_addr);

            pj_inet_ntop(family, 
                         &info.relay_addr.ipv4.sin_addr,
                         relay_ip, sizeof(relay_ip));
            pj_uint16_t relay_port = pj_ntohs(info.relay_addr.ipv4.sin_port);

            PJ_LOG(3,("TURN", "Relay allocated: %s:%d", relay_ip, relay_port));
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

//
// VideoStreamer
//
class VideoStreamer {
public:
    VideoStreamer() : frame_counter(0), sequence_number(0) {
        create_and_set_output_folders();

        // 로그 파일 준비
        string logFilePath = logs_folder + "/packet_log.csv";
        logger = make_unique<BufferedLogger>(logFilePath);
        logger->log("sequence_number,pts,size,timestamp_frame,timestamp_sending,encoding_latency,frame_type");

        // FFmpeg 코덱 초기화
        codec = avcodec_find_encoder_by_name("libx264");
        if (!codec) throw runtime_error("Codec not found (libx264)");

        codec_ctx.reset(avcodec_alloc_context3(codec));
        if (!codec_ctx) throw runtime_error("Could not allocate video codec context");

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
        string x264_params = "keyint=10:min-keyint=10:scenecut=0:"
                             "bframes=0:force-cfr=1:rc-lookahead=0:ref=1:"
                             "sliced-threads=0:aq-mode=1:trellis=0:psy-rd=1.0:1.0";
        av_dict_set(&opt, "x264-params", x264_params.c_str(), 0);

        if (avcodec_open2(codec_ctx.get(), codec, &opt) < 0) {
            throw runtime_error("Could not open codec");
        }
        av_dict_free(&opt);

        frame.reset(av_frame_alloc());
        if (!frame) throw runtime_error("Could not allocate video frame");
        frame->format = codec_ctx->pix_fmt;
        frame->width  = codec_ctx->width;
        frame->height = codec_ctx->height;
        if (av_frame_get_buffer(frame.get(), 32) < 0) {
            throw runtime_error("Could not allocate the video frame data");
        }

        pkt.reset(av_packet_alloc());
        if (!pkt) throw runtime_error("Could not allocate AVPacket");

        sws_ctx = sws_getContext(WIDTH, HEIGHT, AV_PIX_FMT_YUYV422,
                                 WIDTH, HEIGHT, AV_PIX_FMT_YUV420P,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx) throw runtime_error("Could not initialize the conversion context");

        // UDP 소켓 2개 생성 (두 인터페이스)
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

    void stream(rs2::video_frame& color_frame, uint64_t timestamp_frame) {
        frame->pts = frame_counter++;

        uint8_t* yuyv_data = (uint8_t*)color_frame.get_data();
        const uint8_t* src_slices[1] = { yuyv_data };
        int src_stride[1] = { 2 * WIDTH };

        if (sws_scale(sws_ctx, src_slices, src_stride,
                      0, HEIGHT, frame->data, frame->linesize) < 0) {
            cerr << "Error in sws_scale" << endl;
            return;
        }

        // (선택) PNG 저장
        save_frame_image(color_frame, timestamp_frame);

        // H.264 인코딩 + sendto()
        encode_and_send_frame(timestamp_frame);
    }

private:
    void create_and_set_output_folders() {
        auto now = chrono::system_clock::now();
        time_t time_now = chrono::system_clock::to_time_t(now);
        tm local_time = *localtime(&time_now);

        char folder_name[100];
        strftime(folder_name, sizeof(folder_name), "%Y_%m_%d_%H_%M", &local_time);

        string base_folder = SAVE_FILEPATH + string(folder_name);
        fs::create_directories(base_folder);

        frames_folder = base_folder + "/frames";
        logs_folder   = base_folder + "/logs";
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

    int create_socket_and_bind(const char* interface_ip, const char* interface_name) {
        int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) throw runtime_error("Socket creation failed");

        if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE,
                       interface_name, strlen(interface_name)) < 0) {
            close(sockfd);
            throw runtime_error("SO_BINDTODEVICE failed for " + string(interface_name));
        }
        sockaddr_in bindaddr = create_sockaddr(interface_ip, 0);
        if (bind(sockfd, (struct sockaddr*)&bindaddr, sizeof(bindaddr)) < 0) {
            close(sockfd);
            throw runtime_error("Bind failed on " + string(interface_ip));
        }
        return sockfd;
    }

    void save_frame_image(const rs2::video_frame& frame_data, uint64_t timestamp_frame) {
        // 디버깅 목적 PNG 저장
        string filename = frames_folder + "/" + to_string(timestamp_frame) + ".png";
        cv::Mat yuyv_image(HEIGHT, WIDTH, CV_8UC2, (void*)frame_data.get_data());
        cv::Mat bgr_image;
        cv::cvtColor(yuyv_image, bgr_image, cv::COLOR_YUV2BGR_YUYV);
        cv::imwrite(filename, bgr_image);
    }

    void encode_and_send_frame(uint64_t timestamp_frame) {
        if (avcodec_send_frame(codec_ctx.get(), frame.get()) < 0) {
            cerr << "Error sending frame for encoding" << endl;
            return;
        }

        int ret;
        while ((ret = avcodec_receive_packet(codec_ctx.get(), pkt.get())) == 0) {
            PacketHeader header;
            header.timestamp_frame   = timestamp_frame;
            header.timestamp_sending = chrono::duration_cast<chrono::microseconds>(
                chrono::system_clock::now().time_since_epoch()
            ).count();
            header.sequence_number   = sequence_number++;

            double encoding_latency_ms = (header.timestamp_sending - header.timestamp_frame)/1000.0;
            string frame_type = (pkt->flags & AV_PKT_FLAG_KEY) ? "I-frame" : "P-frame";

            vector<uint8_t> packet_data(sizeof(PacketHeader) + pkt->size);
            memcpy(packet_data.data(), &header, sizeof(PacketHeader));
            memcpy(packet_data.data() + sizeof(PacketHeader), pkt->data, pkt->size);

            // 로그
            ostringstream logMsg;
            logMsg << header.sequence_number << ","
                   << frame->pts << ","
                   << pkt->size << ","
                   << header.timestamp_frame << ","
                   << header.timestamp_sending << ","
                   << encoding_latency_ms << ","
                   << frame_type;
            logger->log(logMsg.str());

            // 두 인터페이스로 비동기 전송
            auto send_task1 = async(launch::async, [this, &packet_data]() {
                if (sendto(sockfd1, packet_data.data(), packet_data.size(), 0,
                           (const struct sockaddr*)&servaddr1, sizeof(servaddr1)) < 0) {
                    cerr << "Error sending packet on interface 1: " << strerror(errno) << endl;
                }
            });
            auto send_task2 = async(launch::async, [this, &packet_data]() {
                if (sendto(sockfd2, packet_data.data(), packet_data.size(), 0,
                           (const struct sockaddr*)&servaddr2, sizeof(servaddr2)) < 0) {
                    cerr << "Error sending packet on interface 2: " << strerror(errno) << endl;
                }
            });
            send_task1.get();
            send_task2.get();

            av_packet_unref(pkt.get());
        }

        if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
            cerr << "Error receiving encoded packet" << endl;
        }
    }

private:
    const AVCodec* codec;
    unique_ptr<AVCodecContext, void(*)(AVCodecContext*)> codec_ctx {
        nullptr, [](AVCodecContext* p){ avcodec_free_context(&p); }
    };
    unique_ptr<AVFrame, void(*)(AVFrame*)> frame {
        nullptr, [](AVFrame* p){ av_frame_free(&p); }
    };
    unique_ptr<AVPacket, void(*)(AVPacket*)> pkt {
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

//
// TURN 관련
//
pj_caching_pool cp;
pj_pool_t *pool = nullptr;
pj_turn_sock *turn_sock = nullptr;

atomic<bool> turn_running{true};

static void turn_ack_receiver_thread()
{
    pj_status_t status;

    // 1) pjlib 초기화
    status = pj_init();
    if (status != PJ_SUCCESS) {
        PJ_LOG(1,("TAG","pj_init() error"));
        return;
    }
    status = pjlib_util_init();
    if (status != PJ_SUCCESS) {
        PJ_LOG(1,("TAG","pjlib_util_init error"));
        pj_shutdown();
        return;
    }

    // 2) caching pool
    pj_caching_pool_init(&cp, nullptr, 0);
    pool = pj_pool_create(&cp.factory, "turn_pool", 4000, 4000, nullptr);

    pj_stun_config stun_cfg;
    pj_stun_config_init(&stun_cfg, &cp.factory, 0, nullptr, nullptr);
    // 실제 사용 시 pj_ioqueue_create(), pj_timer_heap_create() 등을 통해
    // stun_cfg.ioqueue, stun_cfg.timer_heap 할당 가능

    // TURN 소켓 세팅
    pj_turn_sock_cfg tcfg;
    pj_turn_sock_cfg_default(&tcfg);

    // 소켓 생성
    status = pj_turn_sock_create(
        &stun_cfg,
        PJ_AF_INET,
        PJ_TURN_TP_UDP,
        &turn_callbacks,
        &tcfg,
        nullptr,
        &turn_sock
    );
    if (status != PJ_SUCCESS) {
        PJ_LOG(1,("TAG","pj_turn_sock_create() error %d", status));
        pj_pool_release(pool);
        pj_caching_pool_destroy(&cp);
        pj_shutdown();
        return;
    }

    // 인증정보
    pj_str_t turnServer = pj_str(const_cast<char*>(TURN_SERVER_IP));
    pj_stun_auth_cred auth_cred;
    pj_bzero(&auth_cred, sizeof(auth_cred));
    auth_cred.type = PJ_STUN_AUTH_CRED_STATIC;

    std::string ephemeral_username = generate_turn_username(TURN_IDENTIFIER, TURN_VALID_SECONDS);
    std::string ephemeral_password = compute_turn_password(
        ephemeral_username + ":" + TURN_REALM,
        TURN_SECRET
    );
    auth_cred.data.static_cred.username = pj_str(const_cast<char*>(ephemeral_username.c_str()));
    auth_cred.data.static_cred.data     = pj_str(const_cast<char*>(ephemeral_password.c_str()));
    auth_cred.data.static_cred.data_type= PJ_STUN_PASSWD_PLAIN;

    // TURN allocate
    status = pj_turn_sock_alloc(
        turn_sock,
        &turnServer,
        TURN_SERVER_PORT,
        nullptr,  // DNS resolver
        &auth_cred,
        nullptr   // alloc param
    );
    if (status != PJ_SUCCESS) {
        PJ_LOG(1,("TAG","pj_turn_sock_alloc() error %d", status));
        pj_turn_sock_destroy(turn_sock);
        pj_pool_release(pool);
        pj_caching_pool_destroy(&cp);
        pj_shutdown();
        return;
    }

    PJ_LOG(3,("TURN", "TURN ack receiver started. polling loop..."));

    while (turn_running.load()) {
        pj_thread_sleep(10);
        // 여기에 pj_ioqueue_poll(), pj_timer_heap_poll() 수행 가능
    }

    // 종료 시 처리
    if (turn_sock) {
        pj_turn_sock_destroy(turn_sock);
        turn_sock = nullptr;
    }
    if (pool) {
        pj_pool_release(pool);
        pool = nullptr;
    }
    pj_caching_pool_destroy(&cp);
    pj_shutdown();
}

void client_stream(VideoStreamer& streamer, rs2::pipeline& pipe, atomic<bool>& running) {
    while (running.load()) {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) continue;

        uint64_t timestamp_frame = chrono::duration_cast<chrono::microseconds>(
            chrono::system_clock::now().time_since_epoch()
        ).count();

        streamer.stream(color_frame, timestamp_frame);
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

        thread turn_thread(turn_ack_receiver_thread);

        cout << "Press Enter to stop streaming..." << endl;
        cin.get();

        running.store(false);
        turn_running.store(false);
        client_thread.join();
        turn_thread.join();
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
