#include <librealsense2/rs.hpp>
#include <iostream>
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

int create_socket_and_bind(const char* interface_ip, const char* interface_name) {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, interface_name, strlen(interface_name)) < 0) {
        perror("SO_BINDTODEVICE failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in bindaddr;
    memset(&bindaddr, 0, sizeof(bindaddr));
    bindaddr.sin_family = AF_INET;
    bindaddr.sin_port = htons(0);
    bindaddr.sin_addr.s_addr = inet_addr(interface_ip);

    if (bind(sockfd, (struct sockaddr*)&bindaddr, sizeof(bindaddr)) < 0) {
        perror("Bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    return sockfd;
}

int main() {
    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_RGB8, FPS);
    pipe.start(cfg);

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
    c->width = WIDTH;
    c->height = HEIGHT;
    c->time_base = {1, FPS};
    c->framerate = {FPS, 1};
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;

    if (avcodec_open2(c, codec, NULL) < 0) {
        cerr << "Could not open codec" << endl;
        avcodec_free_context(&c);
        exit(EXIT_FAILURE);
    }

    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        cerr << "Could not allocate video frame" << endl;
        avcodec_free_context(&c);
        exit(EXIT_FAILURE);
    }

    frame->format = c->pix_fmt;
    frame->width = c->width;
    frame->height = c->height;

    if (av_frame_get_buffer(frame, 32) < 0) {
        cerr << "Could not allocate the video frame data" << endl;
        av_frame_free(&frame);
        avcodec_free_context(&c);
        exit(EXIT_FAILURE);
    }

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        cerr << "Could not allocate AVPacket" << endl;
        av_frame_free(&frame);
        avcodec_free_context(&c);
        exit(EXIT_FAILURE);
    }

    struct SwsContext* sws_ctx = sws_getContext(
        c->width, c->height, AV_PIX_FMT_RGB24,
        c->width, c->height, c->pix_fmt,
        SWS_BILINEAR, NULL, NULL, NULL
    );

    if (!sws_ctx) {
        cerr << "Could not initialize the conversion context" << endl;
        av_packet_free(&pkt);
        av_frame_free(&frame);
        avcodec_free_context(&c);
        exit(EXIT_FAILURE);
    }

    rs2::frameset frames;
    int frame_counter = 0;

    int sockfd1 = create_socket_and_bind(INTERFACE1_IP, INTERFACE1_NAME);
    int sockfd2 = create_socket_and_bind(INTERFACE2_IP, INTERFACE2_NAME);

    struct sockaddr_in servaddr1, servaddr2;
    memset(&servaddr1, 0, sizeof(servaddr1));
    servaddr1.sin_family = AF_INET;
    servaddr1.sin_port = htons(SERVER_PORT);
    servaddr1.sin_addr.s_addr = inet_addr(SERVER_IP);
    memset(&servaddr2, 0, sizeof(servaddr2));
    servaddr2.sin_family = AF_INET;
    servaddr2.sin_port = htons(SERVER_PORT+1);
    servaddr2.sin_addr.s_addr = inet_addr(SERVER_IP);


    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame().as<rs2::video_frame>();

        if (!color_frame) continue;

        const int w = color_frame.get_width();
        const int h = color_frame.get_height();

        uint8_t* rgb_data = (uint8_t*)color_frame.get_data();

        const uint8_t* inData[1] = { rgb_data };
        int inLinesize[1] = { 3 * w };

        sws_scale(sws_ctx, inData, inLinesize, 0, h, frame->data, frame->linesize);

        // RealSense 프레임의 타임스탬프를 사용하여 PTS 설정
        frame->pts = frame_counter ++;

        if (avcodec_send_frame(c, frame) < 0) {
            cerr << "Error sending a frame for encoding" << endl;
            break;
        }

        size_t packet_size = 0;
        uint8_t* packet_data = nullptr;

        if (avcodec_receive_packet(c, pkt) == 0) {
            sendto(sockfd1, pkt->data, pkt->size, 0, (const struct sockaddr*)&servaddr1, sizeof(servaddr1));
            sendto(sockfd1, pkt->data, pkt->size, 0, (const struct sockaddr*)&servaddr2, sizeof(servaddr2));
        } else {
            cerr << "Error receiving encoded packet" << endl;
            break;
        }
    }

    close(sockfd1);
    close(sockfd2);

    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&c);
    sws_freeContext(sws_ctx);

    return 0;
}
