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

    // 인터페이스에 소켓 바인딩
    if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, interface_name, strlen(interface_name)) < 0) {
        perror("SO_BINDTODEVICE failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    // 소켓을 인터페이스 IP에 바인딩
    struct sockaddr_in bindaddr;
    memset(&bindaddr, 0, sizeof(bindaddr));
    bindaddr.sin_family = AF_INET;
    bindaddr.sin_port = htons(0);  // 포트를 0으로 설정하여 시스템에서 사용 가능한 포트를 자동 할당
    bindaddr.sin_addr.s_addr = inet_addr(interface_ip);

    if (bind(sockfd, (struct sockaddr*)&bindaddr, sizeof(bindaddr)) < 0) {
        perror("Bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    return sockfd;
}

void send_packet(int sockfd, const struct sockaddr_in& servaddr, const uint8_t* packet_data, size_t packet_size) {
    // 패킷 전송
    if (sendto(sockfd, packet_data, packet_size, 0, (const struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
        perror("sendto failed");
    }
}

int main() {
    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_RGB8, FPS);
    pipe.start(cfg);

    // 비디오 프레임 인코딩 설정
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

    // 소켓을 미리 생성하고 바인딩
    int sockfd1 = create_socket_and_bind(INTERFACE1_IP, INTERFACE1_NAME);
    int sockfd2 = create_socket_and_bind(INTERFACE2_IP, INTERFACE2_NAME);

    // 서버 주소를 미리 정의
    struct sockaddr_in servaddr1, servaddr2;
    memset(&servaddr1, 0, sizeof(servaddr1));
    servaddr1.sin_family = AF_INET;
    servaddr1.sin_port = htons(SERVER_PORT);
    servaddr1.sin_addr.s_addr = inet_addr(SERVER_IP);

    memset(&servaddr2, 0, sizeof(servaddr2));
    servaddr2.sin_family = AF_INET;
    servaddr2.sin_port = htons(SERVER_PORT + 1);
    servaddr2.sin_addr.s_addr = inet_addr(SERVER_IP);

    while (true) {
        // 프레임 캡처 및 인코딩
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame().as<rs2::video_frame>();

        const int w = color_frame.get_width();
        const int h = color_frame.get_height();

        uint8_t* rgb_data = (uint8_t*)color_frame.get_data();

        const uint8_t* inData[1] = { rgb_data };
        int inLinesize[1] = { 3 * w };

        sws_scale(sws_ctx, inData, inLinesize, 0, h, frame->data, frame->linesize);

        frame->pts += 1;

        if (avcodec_send_frame(c, frame) < 0) {
            cerr << "Error sending a frame for encoding" << endl;
            break;
        }

        size_t packet_size = 0;
        uint8_t* packet_data = nullptr;

        if (avcodec_receive_packet(c, pkt) == 0) {
            packet_size = pkt->size;
            packet_data = new uint8_t[packet_size];
            memcpy(packet_data, pkt->data, packet_size);
        } else {
            cerr << "Error receiving encoded packet" << endl;
            break;
        }

        // 두 개의 인터페이스로 동일한 패킷 데이터 전송
        send_packet(sockfd1, servaddr1, packet_data, packet_size);
        send_packet(sockfd2, servaddr2, packet_data, packet_size);

        // 메모리 정리
        delete[] packet_data;
    }

    // 소켓 종료
    close(sockfd1);
    close(sockfd2);

    // 메모리 정리
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&c);
    sws_freeContext(sws_ctx);

    return 0;
}
