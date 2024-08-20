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
#include "config.h"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

using namespace std;

class VideoStreamer {
public:
    VideoStreamer() : frame_counter(0) {
        codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        if (!codec) throw runtime_error("Codec not found");

        codec_ctx.reset(avcodec_alloc_context3(codec));
        if (!codec_ctx) throw runtime_error("Could not allocate video codec context");

        codec_ctx->bit_rate = 400000;
        codec_ctx->width = WIDTH;
        codec_ctx->height = HEIGHT;
        codec_ctx->time_base = { 1, FPS };
        codec_ctx->framerate = { FPS, 1 };
        codec_ctx->gop_size = 10;
        codec_ctx->max_b_frames = 1;
        codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

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

        sws_ctx.reset(sws_getContext(
            codec_ctx->width, codec_ctx->height, AV_PIX_FMT_RGB24,
            codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
            SWS_BILINEAR, nullptr, nullptr, nullptr));

        if (!sws_ctx) throw runtime_error("Could not initialize the conversion context");

        sockfd1 = create_socket_and_bind(INTERFACE1_IP, INTERFACE1_NAME);
        sockfd2 = create_socket_and_bind(INTERFACE2_IP, INTERFACE2_NAME);

        servaddr1 = create_sockaddr(SERVER_IP, SERVER_PORT);
        servaddr2 = create_sockaddr(SERVER_IP, SERVER_PORT + 1);
    }

    ~VideoStreamer() {
        close(sockfd1);
        close(sockfd2);
    }

    void stream(rs2::video_frame& color_frame) {
        const int w = color_frame.get_width();
        const int h = color_frame.get_height();

        uint8_t* rgb_data = (uint8_t*)color_frame.get_data();

        const uint8_t* inData[1] = { rgb_data };
        int inLinesize[1] = { 3 * w };

        sws_scale(sws_ctx.get(), inData, inLinesize, 0, h, frame->data, frame->linesize);

        frame->pts = frame_counter++;

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
            sendto(sockfd1, pkt->data, pkt->size, 0, (const struct sockaddr*)&servaddr1, sizeof(servaddr1));
            sendto(sockfd2, pkt->data, pkt->size, 0, (const struct sockaddr*)&servaddr2, sizeof(servaddr2));
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
    unique_ptr<SwsContext, void(*)(SwsContext*)> sws_ctx{nullptr, &sws_freeContext};

    int sockfd1, sockfd2;
    struct sockaddr_in servaddr1, servaddr2;
    atomic<int> frame_counter;
};

void frame_capture_thread(VideoStreamer& streamer, rs2::pipeline& pipe, atomic<bool>& running) {
    while (running.load()) {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) continue;

        streamer.stream(color_frame);
    }
}

int main() {
    try {
        rs2::pipeline pipe;
        rs2::config cfg;

        cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_RGB8, FPS);
        pipe.start(cfg);

        VideoStreamer streamer;

        atomic<bool> running(true);
        thread capture_thread(frame_capture_thread, ref(streamer), ref(pipe), ref(running));

        // 메인 스레드가 다른 작업을 수행하거나 사용자 입력을 대기
        // ...

        running.store(false);  // 스레드 종료 신호 전송
        capture_thread.join();  // 스레드 종료 대기
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
