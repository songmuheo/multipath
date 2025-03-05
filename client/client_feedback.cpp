#include <iostream>
#include <fstream>
#include <atomic>
#include <thread>
#include <vector>
#include <future>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <arpa/inet.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <filesystem>

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

// pjnath (TURN)
extern "C" {
#include <pjlib.h>
#include <pjlib-util.h>
#include <pjnath.h>
}

// config.h (사용자 정의 상수 등)
#include "config.h"

using namespace std;
namespace fs = std::filesystem;

// ----------------------------
//     영상 관련 구조체/클래스
// ----------------------------
struct PacketHeader {
    uint64_t timestamp_frame;   // 프레임 캡처 시점 (us)
    uint64_t timestamp_sending; // 전송 시점 (us)
    uint32_t sequence_number;
};

// ----------------------------
//     TURN 인증 유틸
// ----------------------------
std::string generate_turn_username(const std::string& identifier, uint32_t validSeconds) {
    uint64_t expiration = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::system_clock::now().time_since_epoch()).count()
                          + validSeconds;
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
    for (unsigned int i = 0; i < hmac_length; i++) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)hmac_result[i];
    }
    return oss.str();
}

// ----------------------------
//     VideoStreamer 클래스
// ----------------------------
class VideoStreamer {
public:
    VideoStreamer(pj_turn_sock *turn_sock, const pj_sockaddr_in &server_relay_addr)
        : turn_sock_(turn_sock),
          server_addr_(server_relay_addr),
          frame_counter(0),
          sequence_number(0)
    {
        // 출력 폴더 생성
        create_and_set_output_folders();
        open_log_file();

        // (1) 코덱 찾기
        codec = avcodec_find_encoder_by_name("libx264");
        if (!codec) throw runtime_error("Codec not found");

        // (2) 코덱 컨텍스트
        codec_ctx.reset(avcodec_alloc_context3(codec));
        if (!codec_ctx) throw runtime_error("Could not allocate video codec context");

        // (3) 인코더 설정
        codec_ctx->width = WIDTH;
        codec_ctx->height = HEIGHT;
        codec_ctx->time_base = {1, FPS};
        codec_ctx->framerate = {FPS, 1};
        codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        codec_ctx->max_b_frames = 0;
        codec_ctx->gop_size = 10;

        // (4) libx264 옵션
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

        // (5) 코덱 오픈
        if (avcodec_open2(codec_ctx.get(), codec, &opt) < 0) {
            throw runtime_error("Could not open codec");
        }
        av_dict_free(&opt);

        // (6) 프레임/패킷 구조체
        frame.reset(av_frame_alloc());
        if (!frame) throw runtime_error("Could not allocate video frame");
        frame->format = codec_ctx->pix_fmt;
        frame->width  = codec_ctx->width;
        frame->height = codec_ctx->height;
        if (av_frame_get_buffer(frame.get(), 32) < 0)
            throw runtime_error("Could not allocate the video frame data");

        pkt.reset(av_packet_alloc());
        if (!pkt) throw runtime_error("Could not allocate AVPacket");

        // (7) sws (YUYV422 → YUV420P 변환)
        sws_ctx = sws_getContext(WIDTH, HEIGHT, AV_PIX_FMT_YUYV422,
                                 WIDTH, HEIGHT, AV_PIX_FMT_YUV420P,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx) throw runtime_error("Could not initialize sws context");
    }

    ~VideoStreamer() {
        sws_freeContext(sws_ctx);
        log_file.close();
    }

    void stream(rs2::video_frame& color_frame, uint64_t timestamp_frame) {
        // frame->pts = 프레임 카운터
        frame->pts = frame_counter++;

        // YUYV422 -> YUV420P
        uint8_t* yuyv_data = (uint8_t*)color_frame.get_data();
        const uint8_t* src_slices[1] = { yuyv_data };
        int src_stride[1] = { 2 * WIDTH };
        if (sws_scale(sws_ctx, src_slices, src_stride, 0, HEIGHT,
                      frame->data, frame->linesize) < 0) {
            cerr << "Error in sws_scale" << endl;
            return;
        }

        // 디스크에 PNG 저장 (Debug용)
        save_frame(color_frame, timestamp_frame);

        // libx264 인코딩 & 전송
        encode_and_send_frame(timestamp_frame);
    }

    atomic<int> sequence_number;

private:
    void create_and_set_output_folders() {
        auto now = chrono::system_clock::now();
        time_t time_now = chrono::system_clock::to_time_t(now);
        tm local_tm = *localtime(&time_now);
        char folder_name[100];
        strftime(folder_name, sizeof(folder_name), "%Y_%m_%d_%H_%M", &local_tm);
        string base_folder = SAVE_FILEPATH + string(folder_name);
        fs::create_directories(base_folder);
        frames_folder = base_folder + "/frames";
        logs_folder   = base_folder + "/logs";
        fs::create_directories(frames_folder);
        fs::create_directories(logs_folder);
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
            header.timestamp_sending =
                chrono::duration_cast<chrono::microseconds>(
                    chrono::system_clock::now().time_since_epoch()).count();
            header.sequence_number = sequence_number++;

            // 인코딩 딜레이 (디버그용)
            double encoding_latency =
                (header.timestamp_sending - header.timestamp_frame) / 1000.0;
            string frame_type = (pkt->flags & AV_PKT_FLAG_KEY) ? "I-frame" : "P-frame";

            // 패킷 구성 (Header + H264 바이트)
            vector<uint8_t> packet_data(sizeof(PacketHeader) + pkt->size);
            memcpy(packet_data.data(), &header, sizeof(PacketHeader));
            memcpy(packet_data.data() + sizeof(PacketHeader), pkt->data, pkt->size);

            // CSV 로그에 기록
            log_packet_to_csv(sequence_number - 1, pkt->size,
                              header.timestamp_frame,
                              header.timestamp_sending,
                              frame->pts,
                              encoding_latency,
                              frame_type);

            // TURN 소켓으로 전송
            pj_status_t status = pj_turn_sock_sendto(
                turn_sock_,
                packet_data.data(),
                (pj_size_t)packet_data.size(),
                0, // flags
                (pj_sockaddr*)&server_addr_,
                sizeof(server_addr_)
            );
            if (status != PJ_SUCCESS) {
                cerr << "[CLIENT] Error sending packet via TURN: pj_status=" << status << endl;
            }

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

    void log_packet_to_csv(int seq, int size,
                           uint64_t timestamp, uint64_t sendtime,
                           int64_t pts, double encoding_latency,
                           const string& frame_type)
    {
        log_file << seq << ","
                 << pts << ","
                 << size << ","
                 << timestamp << ","
                 << sendtime << ","
                 << encoding_latency << ","
                 << frame_type << "\n";
    }

private:
    // AV
    const AVCodec* codec = nullptr;
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

    // TURN
    pj_turn_sock *turn_sock_ = nullptr;
    pj_sockaddr_in server_addr_;

    // 기타
    ofstream log_file;
    atomic<int> frame_counter;
    string frames_folder;
    string logs_folder;
};

// ----------------------------
//   클라이언트: Realsense + 인코딩
// ----------------------------
void client_stream(VideoStreamer& streamer, rs2::pipeline& pipe, atomic<bool>& running) {
    while (running.load()) {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) continue;

        uint64_t timestamp_frame = chrono::duration_cast<chrono::microseconds>(
            chrono::system_clock::now().time_since_epoch()).count();

        // 스트리머에 전송
        streamer.stream(color_frame, timestamp_frame);
    }
}

// ----------------------------
//   TURN 수신 콜백: 서버 ACK 수신
// ----------------------------
static void on_rx_data_client(pj_turn_sock *sock,
                              void *pkt,
                              pj_size_t size,
                              const pj_sockaddr_t *src_addr,
                              unsigned int addr_len)
{
    if (size > 0 && pkt) {
        // 예: 수신한 ACK 문자열을 그대로 출력
        std::string ack_str((char*)pkt, (size_t)size);
        std::cout << "[CLIENT] Received ACK: " << ack_str << std::endl;
    }
}

// ----------------------------
//   클라이언트 메인
// ----------------------------
int main() {
    try {
        // ffmpeg 초기화
        av_log_set_level(AV_LOG_QUIET);
        avcodec_register_all();
        av_register_all();

        // (1) Realsense 파이프라인 시작
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_YUYV, FPS);
        pipe.start(cfg);

        // (2) pjnath/PJLIB 초기화
        pj_status_t status;
        status = pj_init();
        if (status != PJ_SUCCESS) {
            cerr << "[CLIENT] pj_init() error" << endl;
            return 1;
        }
        status = pjlib_util_init();
        if (status != PJ_SUCCESS) {
            cerr << "[CLIENT] pjlib_util_init() error" << endl;
            return 1;
        }

        pj_caching_pool cp;
        pj_caching_pool_init(&cp, nullptr, 0);

        pj_pool_t *pool = pj_pool_create(&cp.factory, "client_pool", 4000, 4000, nullptr);
        if (!pool) {
            cerr << "[CLIENT] pj_pool_create() fail" << endl;
            return 1;
        }

        pj_ioqueue_t *ioqueue = nullptr;
        status = pj_ioqueue_create(pool, 64, &ioqueue);
        if (status != PJ_SUCCESS) {
            cerr << "[CLIENT] pj_ioqueue_create() error" << endl;
            return 1;
        }

        pj_timer_heap_t *timer_heap = nullptr;
        status = pj_timer_heap_create(pool, 100, &timer_heap);
        if (status != PJ_SUCCESS) {
            cerr << "[CLIENT] pj_timer_heap_create() error" << endl;
            return 1;
        }

        pj_stun_config stun_cfg;
        pj_stun_config_init(&stun_cfg, &cp.factory, PJ_AF_INET, ioqueue, timer_heap);

        // (3) TURN 소켓 생성
        static pj_turn_sock_cb turn_cb_client;
        memset(&turn_cb_client, 0, sizeof(turn_cb_client));
        turn_cb_client.on_rx_data = &on_rx_data_client; // ACK 수신 처리

        pj_turn_sock *turn_sock_client = nullptr;
        status = pj_turn_sock_create(&stun_cfg,
                                     PJ_AF_INET,
                                     PJ_TURN_TP_UDP,
                                     &turn_cb_client,
                                     nullptr, // user_data
                                     nullptr, // 개별 소켓
                                     &turn_sock_client);
        if (status != PJ_SUCCESS || !turn_sock_client) {
            cerr << "[CLIENT] pj_turn_sock_create() error" << endl;
            return 1;
        }

        // (4) TURN Allocate (ephemeral username/password)
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
        status = pj_turn_sock_alloc(turn_sock_client,
                                    &turn_srv_ip,
                                    TURN_SERVER_PORT,
                                    nullptr,   // 로컬 바인드 X
                                    &auth_cred,
                                    nullptr);
        if (status != PJ_SUCCESS) {
            cerr << "[CLIENT] pj_turn_sock_alloc() fail. status=" << status << endl;
            return 1;
        }

        // (5) TURN 할당 완료 대기(비동기이므로 폴링)
        bool allocated = false;
        for(int i=0; i<200; i++){
            pj_time_val delay = {0, 10}; // 10ms
            pj_ioqueue_poll(ioqueue, &delay);
            pj_timer_heap_poll(timer_heap, nullptr);

            pj_turn_sock_info info;
            memset(&info, 0, sizeof(info));
            pj_turn_sock_get_info(turn_sock_client, &info);

            if (info.state == PJ_TURN_STATE_READY) {
                allocated = true;
                break;
            }
            pj_thread_sleep(10);
        }
        if (!allocated) {
            cerr << "[CLIENT] TURN allocate not completed" << endl;
            return 1;
        }

        // (6) 서버의 TURN Relay 주소 지정
        // 실제 운영 시에는 "서버가 콘솔에 찍어준 relay 주소"를 여기서 세팅해야 함.
        // 일단 질문 예시에 따르면 SERVER_IP / SERVER_PORT를 그대로 'relay 주소'로 볼 수도 있지만,
        // coturn 설정 "min-port=6000, max-port=6000"이면 보통 ip:6000 형태일 것임.
        // 여기서는 config.h에 SERVER_IP, SERVER_PORT에 '서버 relay IP:Port'가 들어있다고 가정.
        pj_sockaddr_in server_relay_addr;
        memset(&server_relay_addr, 0, sizeof(server_relay_addr));
        server_relay_addr.sin_family = PJ_AF_INET;
        server_relay_addr.sin_port   = pj_htons(SERVER_PORT); // config.h
        pj_inet_pton(PJ_AF_INET, SERVER_IP, &server_relay_addr.sin_addr);

        // (7) Permission 설정(서버 relay 주소)
        status = pj_turn_sock_set_perm(turn_sock_client,
                                       1,
                                       (pj_sockaddr*)&server_relay_addr,
                                       sizeof(server_relay_addr));
        if (status != PJ_SUCCESS) {
            cerr << "[CLIENT] pj_turn_sock_set_perm() error: " << status << endl;
            // 에러여도 일단 진행
        }

        // (8) VideoStreamer 생성
        VideoStreamer streamer(turn_sock_client, server_relay_addr);

        // (9) 영상 송신용 스레드
        atomic<bool> running(true);
        thread client_thread(client_stream, ref(streamer), ref(pipe), ref(running));

        // (10) 메인 루프: TURN 이벤트 폴링
        cout << "[CLIENT] Press Enter to stop streaming..." << endl;
        while (true) {
            if (cin.peek() != EOF) {
                // 엔터 입력 시 종료
                break;
            }
            pj_time_val delay = {0, 10};
            pj_ioqueue_poll(ioqueue, &delay);
            pj_timer_heap_poll(timer_heap, nullptr);
            pj_thread_sleep(10);
        }

        // (11) 종료 처리
        running.store(false);
        client_thread.join();

        if (turn_sock_client) {
            pj_turn_sock_destroy(turn_sock_client);
            turn_sock_client = nullptr;
        }
        pj_pool_release(pool);
        pj_caching_pool_destroy(&cp);
        pj_shutdown();
    }
    catch (const exception& e) {
        cerr << "[CLIENT] Exception: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
