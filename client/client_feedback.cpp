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
#include "config.h"  // client/config.h

// OpenSSL 헤더 (HMAC 계산용)
#include <openssl/hmac.h>
#include <openssl/evp.h>

// FFmpeg 헤더
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

// TURN 관련 헤더 (pjnath, pjlib)
extern "C" {
#include <pjlib.h>
#include <pjlib-util.h>
#include <pjnath.h>
}

using namespace std;
namespace fs = std::filesystem;

//
// BufferedLogger: 로그를 내부 버퍼에 기록하는 클래스
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
// 영상 전송 관련 구조체
//
struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};

//
// TURN 관련 헬퍼 함수들
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
// VideoStreamer 클래스
//
class VideoStreamer {
public:
    VideoStreamer() : frame_counter(0), sequence_number(0) {
        create_and_set_output_folders();

        // BufferedLogger를 이용하여 로그 파일 버퍼링 (CSV 헤더 기록)
        string logFilePath = logs_folder + "/packet_log.csv";
        logger = make_unique<BufferedLogger>(logFilePath);
        logger->log("sequence_number,pts,size,timestamp_frame,timestamp_sending,encoding_latency,frame_type");

        // FFmpeg 코덱 초기화
        codec = avcodec_find_encoder_by_name("libx264");
        if (!codec) throw runtime_error("Codec not found");

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
        string x264_params = "keyint=10:min-keyint=10:scenecut=0:bframes=0:force-cfr=1:rc-lookahead=0:ref=1:sliced-threads=0:aq-mode=1:trellis=0:psy-rd=1.0:1.0";
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
        if (av_frame_get_buffer(frame.get(), 32) < 0)
            throw runtime_error("Could not allocate the video frame data");

        pkt.reset(av_packet_alloc());
        if (!pkt) throw runtime_error("Could not allocate AVPacket");

        // 영상 전송용 UDP 소켓 생성 및 바인딩 (두 인터페이스)
        sockfd1 = create_socket_and_bind(INTERFACE1_IP, INTERFACE1_NAME);
        sockfd2 = create_socket_and_bind(INTERFACE2_IP, INTERFACE2_NAME);

        servaddr1 = create_sockaddr(SERVER_IP, SERVER_PORT);
        servaddr2 = create_sockaddr(SERVER_IP, SERVER_PORT + 1);

        // 색상 변환 컨텍스트 (YUYV422 → YUV420P)
        sws_ctx = sws_getContext(WIDTH, HEIGHT, AV_PIX_FMT_YUYV422,
                                 WIDTH, HEIGHT, AV_PIX_FMT_YUV420P,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx) throw runtime_error("Could not initialize the conversion context");
    }

    ~VideoStreamer() {
        if (sockfd1 >= 0) close(sockfd1);
        if (sockfd2 >= 0) close(sockfd2);
        if (sws_ctx) sws_freeContext(sws_ctx);
    }

    atomic<int> sequence_number;

    // 영상 프레임 전송 (UDP 전송)
    void stream(rs2::video_frame& color_frame, uint64_t timestamp_frame) {
        frame->pts = frame_counter++;

        uint8_t* yuyv_data = (uint8_t*)color_frame.get_data();
        const uint8_t* src_slices[1] = { yuyv_data };
        int src_stride[1] = { 2 * WIDTH };
        if (sws_scale(sws_ctx, src_slices, src_stride, 0, HEIGHT,
                      frame->data, frame->linesize) < 0) {
            cerr << "Error in sws_scale" << endl;
            return;
        }

        // (선택사항) 프레임을 이미지로 저장
        save_frame(color_frame, timestamp_frame);
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
        if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, interface_name, strlen(interface_name)) < 0) {
            close(sockfd);
            throw runtime_error("SO_BINDTODEVICE failed");
        }
        sockaddr_in bindaddr = create_sockaddr(interface_ip, 0);
        if (bind(sockfd, (struct sockaddr*)&bindaddr, sizeof(bindaddr)) < 0) {
            close(sockfd);
            throw runtime_error("Bind failed");
        }
        return sockfd;
    }

    void save_frame(const rs2::video_frame& frame_data, uint64_t timestamp_frame) {
        string filename = frames_folder + "/" + to_string(timestamp_frame) + ".png";
        cv::Mat yuyv_image(HEIGHT, WIDTH, CV_8UC2, (void*)frame_data.get_data());
        cv::Mat bgr_image;
        cv::cvtColor(yuyv_image, bgr_image, cv::COLOR_YUV2BGR_YUYV);
        cv::imwrite(filename, bgr_image);
    }

    void encode_and_send_frame(uint64_t timestamp_frame) {
        if (avcodec_send_frame(codec_ctx.get(), frame.get()) < 0) {
            cerr << "Error sending a frame for encoding" << endl;
            return;
        }
        int ret;
        while ((ret = avcodec_receive_packet(codec_ctx.get(), pkt.get())) == 0) {
            PacketHeader header;
            header.timestamp_frame = timestamp_frame;
            header.timestamp_sending = chrono::duration_cast<chrono::microseconds>(
                chrono::system_clock::now().time_since_epoch()).count();
            header.sequence_number = sequence_number++;
            double encoding_latency = (header.timestamp_sending - header.timestamp_frame) / 1000.0;
            string frame_type = (pkt->flags & AV_PKT_FLAG_KEY) ? "I-frame" : "P-frame";
            vector<uint8_t> packet_data(sizeof(PacketHeader) + pkt->size);
            memcpy(packet_data.data(), &header, sizeof(PacketHeader));
            memcpy(packet_data.data() + sizeof(PacketHeader), pkt->data, pkt->size);

            // 로그 기록 (버퍼된 로그)
            ostringstream logMsg;
            logMsg << (sequence_number - 1) << "," << frame->pts << "," << pkt->size << ","
                   << header.timestamp_frame << "," << header.timestamp_sending << ","
                   << encoding_latency << "," << frame_type;
            logger->log(logMsg.str());

            // 두 인터페이스로 비동기 전송
            auto send_task1 = async(launch::async, [this, &packet_data]() {
                if (sendto(sockfd1, packet_data.data(), packet_data.size(), 0,
                           (const struct sockaddr*)&servaddr1, sizeof(servaddr1)) < 0)
                    cerr << "Error sending packet on interface 1: " << strerror(errno) << endl;
            });
            auto send_task2 = async(launch::async, [this, &packet_data]() {
                if (sendto(sockfd2, packet_data.data(), packet_data.size(), 0,
                           (const struct sockaddr*)&servaddr2, sizeof(servaddr2)) < 0)
                    cerr << "Error sending packet on interface 2: " << strerror(errno) << endl;
            });
            send_task1.get();
            send_task2.get();
            av_packet_unref(pkt.get());
        }
        if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
            cerr << "Error receiving encoded packet" << endl;
        }
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
    unique_ptr<BufferedLogger> logger;
    int sockfd1, sockfd2;
    sockaddr_in servaddr1, servaddr2;
    atomic<int> frame_counter;
    string frames_folder;
    string logs_folder;
};

//
// TURN ACK 수신 및 등록 스레드 (TURN 소켓 이용)
//
atomic<bool> turn_running{true};

void turn_ack_receiver_thread() {
    pj_status_t status;
    pj_caching_pool cp;
    pj_pool_t* pool = nullptr;
    pj_ioqueue_t* ioqueue = nullptr;
    pj_timer_heap_t* timer_heap = nullptr;
    pj_turn_sock* turn_sock = nullptr;
    pj_str_t turn_server;
    pj_stun_auth_cred auth_cred;

    // 로컬 TURN 할당용 주소를 사용하지 않고, resolver가 필요 없으므로 NULL 전달
    // (만약 로컬 인터페이스 지정이 필요하다면 별도의 API 호출을 확인)
    turn_server = pj_str(const_cast<char*>(TURN_SERVER_IP));

    // 동적 자격증명 생성
    std::string ephemeral_username = generate_turn_username(TURN_IDENTIFIER, TURN_VALID_SECONDS);
    std::string ephemeral_password = compute_turn_password(ephemeral_username + ":" + TURN_REALM, TURN_SECRET);
    auth_cred.type = PJ_STUN_AUTH_CRED_STATIC;
    auth_cred.data.static_cred.username = pj_str(const_cast<char*>(ephemeral_username.c_str()));
    auth_cred.data.static_cred.data = pj_str(const_cast<char*>(ephemeral_password.c_str()));
    auth_cred.data.static_cred.data_type = PJ_STUN_PASSWD_PLAIN;

    status = pj_init();
    if (status != PJ_SUCCESS) {
        cerr << "pj_init() error" << endl;
        return;
    }
    status = pjlib_util_init();
    if (status != PJ_SUCCESS) {
        cerr << "pjlib_util_init() error" << endl;
        pj_shutdown();
        return;
    }
    pj_caching_pool_init(&cp, nullptr, 0);
    pool = pj_pool_create(&cp.factory, "turn_ack_pool", 4000, 4000, nullptr);
    if (!pool) {
        cerr << "Failed to create pool" << endl;
        pj_shutdown();
        return;
    }
    status = pj_ioqueue_create(pool, 64, &ioqueue);
    if (status != PJ_SUCCESS) {
        cerr << "pj_ioqueue_create() error" << endl;
        pj_pool_release(pool);
        pj_shutdown();
        return;
    }
    status = pj_timer_heap_create(pool, 128, &timer_heap);
    if (status != PJ_SUCCESS) {
        cerr << "pj_timer_heap_create() error" << endl;
        pj_ioqueue_destroy(ioqueue);
        pj_pool_release(pool);
        pj_shutdown();
        return;
    }

    // TURN 소켓 콜백 등록 (데이터 수신 시 간단히 로그 출력)
    static pj_turn_sock_cb turn_callbacks;
    pj_bzero(&turn_callbacks, sizeof(turn_callbacks));
    turn_callbacks.on_rx_data = [](pj_turn_sock* sock, void* user_data, unsigned int size,
                                     const pj_sockaddr_t* src_addr, unsigned int addr_len) {
        if (size > 0) {
            cout << "[TURN] Received data of size: " << size << endl;
        }
    };

    status = pj_turn_sock_create(nullptr, PJ_AF_INET, PJ_TURN_TP_UDP, &turn_callbacks, nullptr, nullptr, &turn_sock);
    if (status != PJ_SUCCESS) {
        cerr << "pj_turn_sock_create() error" << endl;
        pj_timer_heap_destroy(timer_heap);
        pj_ioqueue_destroy(ioqueue);
        pj_pool_release(pool);
        pj_shutdown();
        return;
    }

    // 네 번째 인자: resolver가 필요하므로 NULL 전달 (로컬 주소 지정은 별도 API로 처리)
    status = pj_turn_sock_alloc(turn_sock, &turn_server, TURN_SERVER_PORT, NULL, &auth_cred, nullptr);
    if (status != PJ_SUCCESS) {
        cerr << "pj_turn_sock_alloc() error" << endl;
        pj_turn_sock_destroy(turn_sock);
        pj_timer_heap_destroy(timer_heap);
        pj_ioqueue_destroy(ioqueue);
        pj_pool_release(pool);
        pj_shutdown();
        return;
    }

    // TURN 할당된 relay 주소 가져오기 (함수명 수정: get_relay_addr)
    pj_sockaddr relay_addr;
    unsigned int relay_addr_len = sizeof(relay_addr);
    status = pj_turn_sock_get_relay_addr(turn_sock, &relay_addr, &relay_addr_len);
    if (status != PJ_SUCCESS) {
        cerr << "pj_turn_sock_get_relay_addr() error" << endl;
        pj_turn_sock_destroy(turn_sock);
        pj_timer_heap_destroy(timer_heap);
        pj_ioqueue_destroy(ioqueue);
        pj_pool_release(pool);
        pj_shutdown();
        return;
    }
    char relay_ip[32];
    pj_inet_ntop(PJ_AF_INET, &((pj_sockaddr_in*)&relay_addr)->sin_addr, relay_ip, sizeof(relay_ip));
    uint16_t relay_port = ntohs(((pj_sockaddr_in*)&relay_addr)->sin_port);
    cout << "TURN allocated relay address: " << relay_ip << ":" << relay_port << endl;

    // 클라이언트의 TURN relay 주소를 서버에 등록하는 메시지 전송
    string reg_msg = "TURN_REG:" + string(relay_ip) + ":" + to_string(relay_port);
    pj_str_t reg_str = pj_str(const_cast<char*>(reg_msg.c_str()));
    pj_sockaddr server_reg_addr;
    pj_str_t server_ip_str = pj_str(const_cast<char*>(SERVER_IP));

    status = pj_sockaddr_parse(PJ_AF_INET, SERVER_REG_PORT, pj_str(const_cast<char*>(SERVER_IP)), &server_reg_addr);
    if (status != PJ_SUCCESS) {
        cerr << "pj_sockaddr_parse() for registration failed" << endl;
    } else {
        status = pj_turn_sock_sendto(turn_sock, reinterpret_cast<const pj_uint8_t*>(reg_str.ptr),
                                     reg_str.slen, &server_reg_addr, sizeof(server_reg_addr));
        if (status != PJ_SUCCESS) {
            cerr << "Failed to send TURN registration message" << endl;
        } else {
            cout << "TURN registration message sent to server." << endl;
        }
    }

    cout << "TURN ACK receiver started. Entering polling loop." << endl;
    while (turn_running.load()) {
        pj_time_val timeout;
        timeout.sec = 0;
        timeout.msec = 10;
        pj_ioqueue_poll(ioqueue, &timeout);
        pj_thread_sleep(10);
    }

    // 종료 처리: 자원 해제
    pj_turn_sock_destroy(turn_sock);
    pj_timer_heap_destroy(timer_heap);
    pj_ioqueue_destroy(ioqueue);
    pj_pool_release(pool);
    pj_caching_pool_destroy(&cp);
    pj_shutdown();
}

//
// 클라이언트 영상 스트림 처리 스레드
//
void client_stream(VideoStreamer& streamer, rs2::pipeline& pipe, atomic<bool>& running) {
    while (running.load()) {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) continue;
        uint64_t timestamp_frame = chrono::duration_cast<chrono::microseconds>(
            chrono::system_clock::now().time_since_epoch()).count();
        streamer.stream(color_frame, timestamp_frame);
    }
}

//
// main()
//
int main() {
    try {
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_YUYV, FPS);
        pipe.start(cfg);

        VideoStreamer streamer;

        atomic<bool> running(true);
        thread client_thread(client_stream, ref(streamer), ref(pipe), ref(running));

        // TURN ACK 수신 스레드 시작
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
