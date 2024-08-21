#include <librealsense2/rs.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstring>
#include <arpa/inet.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <unistd.h>
#include <memory>
#include <future>
#include "config.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

using namespace std;

class VideoStreamer {
public:
    VideoStreamer() : frame_counter(0) {
        codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        if (!codec) throw runtime_error("Codec not found");

        codec_ctx.reset(avcodec_alloc_context3(codec));
        if (!codec_ctx) throw runtime_error("Could not allocate video codec context");

        codec_ctx->bit_rate = BITRATE;
        codec_ctx->width = WIDTH;
        codec_ctx->height = HEIGHT;
        codec_ctx->time_base = { 1, FPS };
        codec_ctx->framerate = { FPS, 1 };
        codec_ctx->gop_size = 10;
        codec_ctx->max_b_frames = 1;
        codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

        codec_ctx->thread_count = 4;  // 멀티스레드 활성화
        av_opt_set(codec_ctx->priv_data, "preset", "ultrafast", 0);  // 인코딩 속도 최적화

        if (avcodec_open2(codec_ctx.get(), codec, nullptr) < 0) {
            throw runtime_error("Could not open codec");
        }

        frame.reset(av_frame_alloc());
        if (!frame) throw runtime_error("Could not allocate video frame");

        frame->format = codec_ctx->pix_fmt;
        frame->width = codec_ctx->width;
        frame->height = codec_ctx->height;

        if (av_frame_get_buffer(frame.get(), 32) < 0) {
            throw runtime_error("Could not allocate the video frame data");
        }

        pkt.reset(av_packet_alloc());
        if (!pkt) throw runtime_error("Could not allocate AVPacket");

        sockfd1 = create_socket_and_bind(INTERFACE1_IP, INTERFACE1_NAME);
        sockfd2 = create_socket_and_bind(INTERFACE2_IP, INTERFACE2_NAME);

        servaddr1 = create_sockaddr(SERVER_IP, SERVER_PORT);
        servaddr2 = create_sockaddr(SERVER_IP, SERVER_PORT + 1);

        sws_ctx = sws_getContext(WIDTH, HEIGHT, AV_PIX_FMT_YUYV422, WIDTH, HEIGHT, AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx) throw runtime_error("Could not initialize the conversion context");
    }

    ~VideoStreamer() {
        close(sockfd1);
        close(sockfd2);
        sws_freeContext(sws_ctx);
    }

    void stream(rs2::video_frame& color_frame) {
        frame->pts = frame_counter++;

        uint8_t* yuyv_data = (uint8_t*)color_frame.get_data();

        // YUYV 데이터를 YUV420P 형식으로 변환하여 AVFrame에 복사
        const uint8_t* src_slices[1] = { yuyv_data };
        int src_stride[1] = { 2 * WIDTH };
        sws_scale(sws_ctx, src_slices, src_stride, 0, HEIGHT, frame->data, frame->linesize);

        encode_and_send_frame();
    }

private:
    struct sockaddr_in create_sockaddr(const char* ip, int port) {
        struct sockaddr_in addr = {};
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

        struct sockaddr_in bindaddr = create_sockaddr(interface_ip, 0);

        if (bind(sockfd, (struct sockaddr*)&bindaddr, sizeof(bindaddr)) < 0) {
            close(sockfd);
            throw runtime_error("Bind failed");
        }

        return sockfd;
    }

    void encode_and_send_frame() {
        if (avcodec_send_frame(codec_ctx.get(), frame.get()) < 0) {
            cerr << "Error sending a frame for encoding" << endl;
            return;
        }

        int ret;
        while ((ret = avcodec_receive_packet(codec_ctx.get(), pkt.get())) == 0) {
            // 두 개의 소켓을 비동기적으로 사용하여 패킷을 전송
            auto send_task1 = async(launch::async, [this] {
                if (sendto(sockfd1, pkt->data, pkt->size, 0, (const struct sockaddr*)&servaddr1, sizeof(servaddr1)) < 0) {
                    cerr << "Error sending packet on interface 1" << endl;
                }
            });

            auto send_task2 = async(launch::async, [this] {
                if (sendto(sockfd2, pkt->data, pkt->size, 0, (const struct sockaddr*)&servaddr2, sizeof(servaddr2)) < 0) {
                    cerr << "Error sending packet on interface 2" << endl;
                }
            });

            // 전송이 완료될 때까지 대기
            send_task1.get();
            send_task2.get();

            av_packet_unref(pkt.get());
        }

        if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
            cerr << "Error receiving encoded packet" << endl;
        }
    }

    const AVCodec* codec;
    unique_ptr<AVCodecContext, void(*)(AVCodecContext*)> codec_ctx{nullptr, [](AVCodecContext* p) { avcodec_free_context(&p); }};
    unique_ptr<AVFrame, void(*)(AVFrame*)> frame{nullptr, [](AVFrame* p) { av_frame_free(&p); }};
    unique_ptr<AVPacket, void(*)(AVPacket*)> pkt{nullptr, [](AVPacket* p) { av_packet_free(&p); }};

    SwsContext* sws_ctx = nullptr;

    int sockfd1, sockfd2;
    struct sockaddr_in servaddr1, servaddr2;
    atomic<int> frame_counter;
};

void frame_capture_thread(VideoStreamer& streamer, rs2::pipeline& pipe, atomic<bool>& running) {
    while (running.load()) {
        rs2::frameset frames;
        if (pipe.poll_for_frames(&frames)) {
            rs2::video_frame color_frame = frames.get_color_frame();
            if (!color_frame) continue;

            streamer.stream(color_frame);
        }
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
        thread capture_thread(frame_capture_thread, ref(streamer), ref(pipe), ref(running));

        cout << "Press Enter to stop streaming..." << endl;
        cin.get();

        running.store(false);
        capture_thread.join();
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
