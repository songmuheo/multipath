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

using namespace std;
namespace fs = std::filesystem;

struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};

class VideoStreamer {
public:
    VideoStreamer() : frame_counter(0), sequence_number(0) {
        // Create the output folders for frames and logs
        create_and_set_output_folders();

        // Open the CSV log file
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
        codec_ctx->max_b_frames = 0;    // B-프레임 사용 안 함
        codec_ctx->gop_size = 30;       // (FFmpeg 레벨) I-frame 간격

        // 4) libx264 옵션 설정
        AVDictionary* opt = nullptr;

        // (a) Preset, Tune, CRF 등
        av_dict_set(&opt, "preset", "veryfast", 0);
        av_dict_set(&opt, "tune", "zerolatency", 0);
        av_dict_set(&opt, "crf", "26", 0);

        // (b) x264-params: 필요한 모든 x264 옵션을 하나의 문자열로 몰아서 전달
        //    keyint=30, min-keyint=30, scenecut=0 등으로 정확히 30프레임 간격으로 I-frame을 강제
        //    (B-프레임 0, lookahead=0, refs=1 등)
        std::string x264_params =
            "keyint=30:"
            "min-keyint=30:"
            "scenecut=0:"
            "bframes=0:" 
            "force-cfr=1:"
            "rc-lookahead=0:"
            "refs=1:"
            "no-sliced-threads=1:"
            "aq-mode=1:"
            "trellis=0:"
            "psy-rd=1.0:"
            "psy-rdoq=1.0";

        av_dict_set(&opt, "x264-params", x264_params.c_str(), 0);

        // 5) 코덱 오픈
        if (avcodec_open2(codec_ctx.get(), codec, &opt) < 0) {
            throw runtime_error("Could not open codec");
        }
        // (사용 끝난 뒤 딕셔너리 해제)
        av_dict_free(&opt);

        // 6) 프레임/패킷 구조체 할당
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

    void stream(rs2::video_frame& color_frame, uint64_t timestamp_frame) {
        // 프레임 pts 설정
        frame->pts = frame_counter++;

        // YUYV 포맷 데이터를 sws_scale로 YUV420P에 매핑
        uint8_t* yuyv_data = (uint8_t*)color_frame.get_data();
        const uint8_t* src_slices[1] = { yuyv_data };
        int src_stride[1] = { 2 * WIDTH };

        sws_scale(sws_ctx, src_slices, src_stride, 0, HEIGHT,
                  frame->data, frame->linesize);

        // 프레임 저장(디스크)
        save_frame(color_frame, timestamp_frame);

        // 프레임 인코딩 + 전송
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

        // Root directory based on current time
        string base_folder = SAVE_FILEPATH + string(folder_name);
        fs::create_directories(base_folder);

        // Create frames and logs subdirectories
        frames_folder = base_folder + "/frames";
        logs_folder   = base_folder + "/logs";
        fs::create_directories(frames_folder);
        fs::create_directories(logs_folder);
    }

    // sockaddr_in 생성 함수
    struct sockaddr_in create_sockaddr(const char* ip, int port) {
        struct sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = inet_addr(ip);
        return addr;
    }

    // 소켓 생성 + 인터페이스 바인딩
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

    // 인코딩 + 전송
    void encode_and_send_frame(uint64_t timestamp_frame) {
        // 1) 인코딩 요청
        if (avcodec_send_frame(codec_ctx.get(), frame.get()) < 0) {
            cerr << "Error sending a frame for encoding" << endl;
            return;
        }

        // 2) 인코딩 완료된 패킷 수신
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

            // I/P 프레임 구분(간단히 AV_PKT_FLAG_KEY 플래그로 구분)
            string frame_type = (pkt->flags & AV_PKT_FLAG_KEY) ? "I-frame" : "P-frame";

            // 패킷 데이터(헤더 + 영상)
            vector<uint8_t> packet_data(sizeof(PacketHeader) + pkt->size);
            memcpy(packet_data.data(), &header, sizeof(PacketHeader));
            memcpy(packet_data.data() + sizeof(PacketHeader), pkt->data, pkt->size);

            // 로그 기록
            log_packet_to_csv(
                sequence_number - 1, pkt->size,
                header.timestamp_frame,
                header.timestamp_sending,
                frame->pts,
                encoding_latency,
                frame_type
            );

            // 비동기로 sendto
            auto send_task2 = async(launch::async, [this, &packet_data] {
                if (sendto(sockfd2, packet_data.data(), packet_data.size(), 0,
                           (const struct sockaddr*)&servaddr2, sizeof(servaddr2)) < 0) {
                    cerr << "Error sending packet on interface 2: " << strerror(errno) << endl;
                }
            });

            auto send_task1 = async(launch::async, [this, &packet_data] {
                if (sendto(sockfd1, packet_data.data(), packet_data.size(), 0,
                           (const struct sockaddr*)&servaddr1, sizeof(servaddr1)) < 0) {
                    cerr << "Error sending packet on interface 1: " << strerror(errno) << endl;
                }
            });

            // 전송 완료 대기
            send_task1.get();
            send_task2.get();

            // 패킷 메모리 해제
            av_packet_unref(pkt.get());
        }

        // 버퍼에 더 이상 패킷이 없는 경우(EAGAIN) 외 에러처리
        if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
            cerr << "Error receiving encoded packet" << endl;
        }
    }

    // CSV 로그 파일 오픈
    void open_log_file() {
        log_file.open(logs_folder + "/packet_log.csv");
        if (!log_file.is_open()) {
            throw runtime_error("Failed to open CSV log file");
        }
        log_file << "sequence_number,pts,size,timestamp_frame,timestamp_sending,encoding_latency,frame_type\n";
    }

    // CSV 로깅 함수
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

    string frames_folder;  // Path to store frames
    string logs_folder;    // Path to store logs
};

// ▼▼▼ poll_for_frames() 대신 wait_for_frames() 사용 ▼▼▼
void client(VideoStreamer& streamer, rs2::pipeline& pipe, atomic<bool>& running) {
    while (running.load()) {
        // wait_for_frames() → 내부적으로 30FPS에 맞춰 프레임을 블로킹 수신
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) continue;

        // 프레임 생성 시점 타임스탬프
        uint64_t timestamp_frame = chrono::duration_cast<chrono::microseconds>(
            chrono::system_clock::now().time_since_epoch()).count();

        streamer.stream(color_frame, timestamp_frame);
    }
}

int main() {
    try {
        // RealSense 파이프라인 및 설정
        rs2::pipeline pipe;
        rs2::config cfg;

        // YUYV, WIDTH x HEIGHT, FPS = 30
        cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_YUYV, FPS);

        // 스트리밍 시작
        pipe.start(cfg);

        // 비디오 스트리머 객체 생성
        VideoStreamer streamer;

        // 스레드 구동
        atomic<bool> running(true);
        thread client_thread(client, ref(streamer), ref(pipe), ref(running));

        cout << "Press Enter to stop streaming..." << endl;
        cin.get();  // 엔터 입력 대기

        // 종료 처리
        running.store(false);
        client_thread.join();
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
