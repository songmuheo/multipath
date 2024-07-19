// client/client.cpp

#include <librealsense2/rs.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <arpa/inet.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <unistd.h>
#include "config.h"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

using namespace std;

void send_packets(const char* interface_ip, int interface_id, rs2::pipeline& pipe, int port) {
    int sockfd;
    struct sockaddr_in servaddr, bindaddr;

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // 소켓을 인터페이스 IP에 바인딩
    memset(&bindaddr, 0, sizeof(bindaddr));
    bindaddr.sin_family = AF_INET;
    bindaddr.sin_port = htons(0);  // 포트를 0으로 설정하여 시스템에서 사용 가능한 포트를 자동 할당
    bindaddr.sin_addr.s_addr = inet_addr(interface_ip);

    if (bind(sockfd, (struct sockaddr*)&bindaddr, sizeof(bindaddr)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(port);
    servaddr.sin_addr.s_addr = inet_addr(SERVER_IP);

    AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        cerr << "Codec not found" << endl;
        exit(EXIT_FAILURE);
    }

    AVCodecContext* c = avcodec_alloc_context3(codec);
    if (!c) {
        cerr << "Could not allocate video codec context" << endl;
        exit(EXIT_FAILURE);
    }

    c->bit_rate = 400000;
    c->width = 640;
    c->height = 480;
    c->time_base = {1, 30};
    c->framerate = {30, 1};
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;

    if (avcodec_open2(c, codec, NULL) < 0) {
        cerr << "Could not open codec" << endl;
        exit(EXIT_FAILURE);
    }

    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        cerr << "Could not allocate video frame" << endl;
        exit(EXIT_FAILURE);
    }
    frame->format = c->pix_fmt;
    frame->width = c->width;
    frame->height = c->height;

    if (av_frame_get_buffer(frame, 32) < 0) {
        cerr << "Could not allocate the video frame data" << endl;
        exit(EXIT_FAILURE);
    }

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        cerr << "Could not allocate AVPacket" << endl;
        exit(EXIT_FAILURE);
    }

    struct SwsContext* sws_ctx = sws_getContext(
        c->width, c->height, AV_PIX_FMT_RGB24,
        c->width, c->height, c->pix_fmt,
        SWS_BILINEAR, NULL, NULL, NULL
    );

    rs2::frameset frames;
    int frame_counter = 0;

    while (true) {
        try {
            frames = pipe.wait_for_frames(15000);  // 15초로 설정 (15000ms)
            rs2::video_frame color_frame = frames.get_color_frame().as<rs2::video_frame>();

            if (!color_frame) continue;

            const int w = color_frame.get_width();
            const int h = color_frame.get_height();

            uint8_t* rgb_data = (uint8_t*)color_frame.get_data();

            const uint8_t* inData[1] = { rgb_data };
            int inLinesize[1] = { 3 * w };

            sws_scale(sws_ctx, inData, inLinesize, 0, h, frame->data, frame->linesize);

            frame->pts = frame_counter++;

            if (avcodec_send_frame(c, frame) < 0) {
                cerr << "Error sending a frame for encoding" << endl;
                exit(EXIT_FAILURE);
            }

            while (avcodec_receive_packet(c, pkt) == 0) {
                sendto(sockfd, pkt->data, pkt->size, 0, (const struct sockaddr*)&servaddr, sizeof(servaddr));
            }
        } catch (const rs2::error& e) {
            cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "): " << e.what() << endl;
            // 예외가 발생하면 프레임을 다시 시도
            this_thread::sleep_for(chrono::seconds(1));
            continue;
        }
    }

    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&c);
    sws_freeContext(sws_ctx);
}

int main() {
    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
    pipe.start(cfg);

    thread interface1_thread(send_packets, INTERFACE1_IP, 1, ref(pipe), SERVER_PORT);
    thread interface2_thread(send_packets, INTERFACE2_IP, 2, ref(pipe), SERVER_PORT + 1);

    interface1_thread.join();
    interface2_thread.join();

    return 0;
}
