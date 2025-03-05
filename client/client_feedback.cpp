#include <iostream>
#include <fstream>
#include <atomic>
#include <thread>
#include <vector>
#include <future>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <arpa/inet.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <unistd.h>
#include <filesystem>
#include <cerrno>
#include <cstring>
#include "config.h"

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

// ----------------------------
//     영상 관련 구조체/클래스
// ----------------------------
struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};

class VideoStreamer {
public:
    VideoStreamer() : frame_counter(0), sequence_number(0) {
        // 출력 폴더 생성 및 로그 파일 오픈
        create_and_set_output_folders();
        open_log_file();

        // 1) 코덱 찾기
        codec = avcodec_find_encoder_by_name("libx264");
        if (!codec) throw runtime_error("Codec not found");

        // 2) 코덱 컨텍스트 할당
        codec_ctx.reset(avcodec_alloc_context3(codec));
        if (!codec_ctx) throw runtime_error("Could not allocate video codec context");

        // 3) 기본 설정
        codec_ctx->width = WIDTH;
        codec_ctx->height = HEIGHT;
        codec_ctx->time_base = {1, FPS};
        codec_ctx->framerate = {FPS, 1};
        codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        codec_ctx->max_b_frames = 0;
        codec_ctx->gop_size = 10;

        // 4) libx264 옵션 설정
        AVDictionary* opt = nullptr;
        av_dict_set(&opt, "preset", "veryfast", 0);
        av_dict_set(&opt, "tune", "zerolatency", 0);
        av_dict_set(&opt, "crf", "26", 0);

        string x264_params =
            "keyint=10:"
            "min-keyint=10:"
            "scenecut=0:"
            "bframes=0:"
            "force-cfr=1:"
            "rc-lookahead=0:"
            "ref=1:"
            "sliced-threads=0:"
            "aq-mode=1:"
            "trellis=0:"
            "psy-rd=1.0:1.0";
        av_dict_set(&opt, "x264-params", x264_params.c_str(), 0);

        // 5) 코덱 오픈
        if (avcodec_open2(codec_ctx.get(), codec, &opt) < 0) {
            throw runtime_error("Could not open codec");
        }
        av_dict_free(&opt);

        // 6) 프레임/패킷 구조체 할당
        frame.reset(av_frame_alloc());
        if (!frame) throw runtime_error("Could not allocate video frame");
        frame->format = codec_ctx->pix_fmt;
        frame->width  = codec_ctx->width;
        frame->height = codec_ctx->height;
        if (av_frame_get_buffer(frame.get(), 32) < 0)
            throw runtime_error("Could not allocate the video frame data");

        pkt.reset(av_packet_alloc());
        if (!pkt) throw runtime_error("Could not allocate AVPacket");

        // 7) 소켓 생성 및 바인딩
        sockfd1 = create_socket_and_bind(INTERFACE1_IP, INTERFACE1_NAME);
        sockfd2 = create_socket_and_bind(INTERFACE2_IP, INTERFACE2_NAME);

        servaddr1 = create_sockaddr(SERVER_IP, SERVER_PORT);
        servaddr2 = create_sockaddr(SERVER_IP, SERVER_PORT + 1);

        // 8) 색상 변환 컨텍스트 (YUYV422 → YUV420P)
        sws_ctx = sws_getContext(WIDTH, HEIGHT, AV_PIX_FMT_YUYV422,
                                 WIDTH, HEIGHT, AV_PIX_FMT_YUV420P,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx) throw runtime_error("Could not initialize the conversion context");
    }

    ~VideoStreamer() {
        close(sockfd1);
        close(sockfd2);
        sws_freeContext(sws_ctx);
        log_file.close();
    }

    // 영상 스트리밍 함수
    void stream(rs2::video_frame& color_frame, uint64_t timestamp_frame) {
        frame->pts = frame_counter++;

        uint8_t* yuyv_data = (uint8_t*)color_frame.get_data();
        const uint8_t* src_slices[1] = { yuyv_data };
        int src_stride[1] = { 2 * WIDTH };
        sws_scale(sws_ctx, src_slices, src_stride, 0, HEIGHT,
                  frame->data, frame->linesize);

        save_frame(color_frame, timestamp_frame);
        encode_and_send_frame(timestamp_frame);
    }

    atomic<int> sequence_number;

private:
    // 출력 폴더 생성
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

    // sockaddr_in 생성
    struct sockaddr_in create_sockaddr(const char* ip, int port) {
        struct sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = inet_addr(ip);
        return addr;
    }

    // 소켓 생성 및 인터페이스 바인딩
    int create_socket_and_bind(const char* interface_ip, const char* interface_name) {
        int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) throw runtime_error("Socket creation failed");

        if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, interface_name, strlen(interface_name)) < 0) {
            close(sockfd);
            throw runtime_error("SO_BINDTODEVICE failed");
        }

        struct sockaddr_in bindaddr = create_sockaddr(interface_ip, 0);
        if (bind(sockfd, (struct sockaddr*)&bindaddr, sizeof(bindaddr)) < 0) {
            close(sockfd);
            throw runtime_error("Bind failed");
        }
        return sockfd;
    }

    // PNG 파일로 프레임 저장
    void save_frame(const rs2::video_frame& frame_data, uint64_t timestamp_frame) {
        string filename = frames_folder + "/" + to_string(timestamp_frame) + ".png";
        cv::Mat yuyv_image(HEIGHT, WIDTH, CV_8UC2, (void*)frame_data.get_data());
        cv::Mat bgr_image;
        cv::cvtColor(yuyv_image, bgr_image, cv::COLOR_YUV2BGR_YUYV);
        cv::imwrite(filename, bgr_image);
    }

    // 인코딩 및 송신
    void encode_and_send_frame(uint64_t timestamp_frame) {
        if (avcodec_send_frame(codec_ctx.get(), frame.get()) < 0) {
            cerr << "Error sending a frame for encoding" << endl;
            return;
        }

        int ret;
        while ((ret = avcodec_receive_packet(codec_ctx.get(), pkt.get())) == 0) {
            PacketHeader header;
            header.timestamp_frame = timestamp_frame;
            header.timestamp_sending =
                chrono::duration_cast<chrono::microseconds>(
                    chrono::system_clock::now().time_since_epoch()).count();
            header.sequence_number = sequence_number++;

            double encoding_latency =
                (header.timestamp_sending - header.timestamp_frame) / 1000.0;

            string frame_type = (pkt->flags & AV_PKT_FLAG_KEY) ? "I-frame" : "P-frame";

            vector<uint8_t> packet_data(sizeof(PacketHeader) + pkt->size);
            memcpy(packet_data.data(), &header, sizeof(PacketHeader));
            memcpy(packet_data.data() + sizeof(PacketHeader), pkt->data, pkt->size);

            log_packet_to_csv(sequence_number - 1, pkt->size,
                              header.timestamp_frame,
                              header.timestamp_sending,
                              frame->pts,
                              encoding_latency,
                              frame_type);

            // 두 인터페이스로 전송
            auto send_task1 = async(launch::async, [this, &packet_data] {
                if (sendto(sockfd1, packet_data.data(), packet_data.size(), 0,
                           (const struct sockaddr*)&servaddr1, sizeof(servaddr1)) < 0)
                    cerr << "Error sending packet on interface 1: " << strerror(errno) << endl;
            });
            auto send_task2 = async(launch::async, [this, &packet_data] {
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

    void open_log_file() {
        log_file.open(logs_folder + "/packet_log.csv");
        if (!log_file.is_open())
            throw runtime_error("Failed to open CSV log file");
        log_file << "sequence_number,pts,size,timestamp_frame,timestamp_sending,encoding_latency,frame_type\n";
    }

    void log_packet_to_csv(int sequence_number, int size,
                           uint64_t timestamp, uint64_t sendtime,
                           int64_t pts, double encoding_latency,
                           const string& frame_type) {
        log_file << sequence_number << ","
                 << pts << ","
                 << size << ","
                 << timestamp << ","
                 << sendtime << ","
                 << encoding_latency << ","
                 << frame_type << "\n";
    }

    // 멤버 변수들
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
    ofstream log_file;
    int sockfd1, sockfd2;
    struct sockaddr_in servaddr1, servaddr2;
    atomic<int> frame_counter;
    string frames_folder;
    string logs_folder;
};

// ----------------------------
// client_stream 함수 추가!!
// ----------------------------
void client_stream(VideoStreamer& streamer, rs2::pipeline& pipe, atomic<bool>& running)
{
    while (running.load()) {
        // RealSense에서 프레임 가져오기
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) continue;

        // 현재 시간(마이크로초) 기준 타임스탬프
        uint64_t timestamp_frame = chrono::duration_cast<chrono::microseconds>(
            chrono::system_clock::now().time_since_epoch()).count();

        // VideoStreamer로 전달
        streamer.stream(color_frame, timestamp_frame);
    }
}

// 예시) 콜백 함수
static void on_rx_data(pj_turn_sock *sock,
                       void *user_data,
                       unsigned int size,
                       const pj_sockaddr_t *src_addr,
                       unsigned int addr_len)
{
    if (size > 0) {
        std::cout << "[TURN] Received data of size: " << size << std::endl;
    }
}


// 예시) 콜백 함수들을 모아놓은 구조체
static pj_turn_sock_cb g_turn_callbacks;

// TURN 스레드 예시
void turn_ack_receiver_thread()
{
    pj_status_t    status;
    pj_caching_pool cp;
    pj_pool_t      *pool       = nullptr;
    pj_ioqueue_t   *ioqueue    = nullptr;
    pj_stun_config  stun_cfg;
    pj_turn_sock   *turn_sock  = nullptr;

    pj_str_t        turn_server;
    pj_stun_auth_cred auth_cred;

    // 1) PJ 초기화
    status = pj_init();
    if (status != PJ_SUCCESS) {
        std::cerr << "pj_init() error" << std::endl;
        return;
    }
    pj_caching_pool_init(&cp, nullptr, 0);

    // 2) 메모리 풀 생성
    pool = pj_pool_create(&cp.factory, "turn_ack_pool", 4000, 4000, nullptr);
    if (!pool) {
        std::cerr << "Failed to create pool" << std::endl;
        goto on_return;
    }

    // 3) ioqueue 생성
    status = pj_ioqueue_create(pool, 64, &ioqueue);
    if (status != PJ_SUCCESS) {
        std::cerr << "pj_ioqueue_create() error" << std::endl;
        goto on_return;
    }

    // 4) STUN/TURN 설정 초기화
    pj_stun_config_init(&stun_cfg, &cp.factory, PJ_AF_INET, ioqueue, nullptr);

    // 5) 콜백 구조체 준비
    pj_bzero(&g_turn_callbacks, sizeof(g_turn_callbacks));
    g_turn_callbacks.on_rx_data = &on_rx_data;
    // 필요하다면 on_state_changed, on_alloc_success 등 다른 콜백도 설정 가능

    // 6) TURN 소켓 생성 시 콜백 등록
    status = pj_turn_sock_create(&stun_cfg,
                                 0,                 // flags
                                 PJ_TURN_TP_UDP,    // UDP
                                 &g_turn_callbacks, // 콜백 구조체
                                 nullptr,           // user_data(필요하면 사용)
                                 nullptr,           // 소켓 설정
                                 &turn_sock);
    if (status != PJ_SUCCESS) {
        std::cerr << "pj_turn_sock_create() error" << std::endl;
        goto on_return;
    }

    // 7) TURN 할당
    turn_server = pj_str(const_cast<char*>(TURN_SERVER_IP));
    auth_cred.type = PJ_STUN_AUTH_CRED_STATIC;
    auth_cred.data.static_cred.username = pj_str(const_cast<char*>(TURN_USERNAME));
    auth_cred.data.static_cred.data_type = PJ_STUN_PASSWD_PLAIN;
    auth_cred.data.static_cred.data = pj_str(const_cast<char*>(TURN_PASSWORD));

    status = pj_turn_sock_alloc(
        turn_sock,
        &turn_server,
        TURN_SERVER_PORT,
        nullptr,     // DNS Resolver
        &auth_cred,
        nullptr      // 추가 파라미터
    );
    if (status != PJ_SUCCESS) {
        std::cerr << "pj_turn_sock_alloc() error" << std::endl;
        goto on_return;
    }

    std::cout << "TURN ACK receiver started. (callback-based)" << std::endl;

    // 8) 여기서부터는 콜백으로만 데이터가 들어옴
    //    이 스레드를 단순 대기 상태로 둡니다.
    while (true) {
        pj_thread_sleep(10);  // 10ms
    }

on_return:
    if (turn_sock) {
        pj_turn_sock_destroy(turn_sock);
        turn_sock = nullptr;
    }
    pj_caching_pool_destroy(&cp);
    pj_shutdown();
}


// ----------------------------
//              main()
// ----------------------------
int main() {
    try {
        // RealSense 파이프라인 설정
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_YUYV, FPS);
        pipe.start(cfg);

        VideoStreamer streamer;

        // TURN ACK 수신용 스레드
        thread turn_thread(turn_ack_receiver_thread);

        atomic<bool> running(true);
        // 여기서 client_stream() 함수를 실행
        thread client_thread(client_stream, ref(streamer), ref(pipe), ref(running));

        cout << "Press Enter to stop streaming..." << endl;
        cin.get();

        // 종료 처리
        running.store(false);
        client_thread.join();

        // TURN 스레드 종료 (임시로 pthread_cancel 사용)
        pthread_cancel(turn_thread.native_handle());
        turn_thread.join();
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
