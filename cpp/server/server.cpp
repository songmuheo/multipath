#include <iostream>
#include <fstream>
#include <map>
#include <chrono>
#include <thread>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include "config.h"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

using namespace std;

void server(int port) {
    int sockfd;
    struct sockaddr_in servaddr, cliaddr;
    char buffer[PACKET_SIZE];
    socklen_t len;

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));

    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(port);
    servaddr.sin_addr.s_addr = inet_addr(SERVER_IP);

    if (bind(sockfd, (const struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) {
        cerr << "Codec not found" << endl;
        exit(EXIT_FAILURE);
    }

    AVCodecContext* c = avcodec_alloc_context3(codec);
    if (!c) {
        cerr << "Could not allocate video codec context" << endl;
        exit(EXIT_FAILURE);
    }

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

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        cerr << "Could not allocate AVPacket" << endl;
        av_frame_free(&frame);
        avcodec_free_context(&c);
        exit(EXIT_FAILURE);
    }

    cv::Mat img;
    struct SwsContext* sws_ctx = nullptr;

    while (true) {
        len = sizeof(cliaddr);
        int n = recvfrom(sockfd, buffer, PACKET_SIZE, 0, (struct sockaddr*)&cliaddr, &len);
        if (n < 0) {
            perror("recvfrom error");
            continue;
        }

        // n이 버퍼 크기를 초과하는 경우 처리
        if (n > PACKET_SIZE) {
            cerr << "Received packet size exceeds buffer limit: " << n << " > " << PACKET_SIZE << endl;
            continue;
        }

        pkt->data = (uint8_t*)buffer;
        pkt->size = n;

        if (avcodec_send_packet(c, pkt) < 0) {
            cerr << "Error sending a packet for decoding" << endl;
            continue;
        }

        while (avcodec_receive_frame(c, frame) == 0) {
            if (!sws_ctx) {
                sws_ctx = sws_getContext(c->width, c->height, c->pix_fmt,
                                         c->width, c->height, AV_PIX_FMT_BGR24,
                                         SWS_BILINEAR, nullptr, nullptr, nullptr);

                if (!sws_ctx) {
                    cerr << "Could not initialize the conversion context" << endl;
                    break;
                }

                img = cv::Mat(c->height, c->width, CV_8UC3);
            }

            uint8_t* data[1] = { img.data };
            int linesize[1] = { static_cast<int>(img.step1()) };

            sws_scale(sws_ctx, frame->data, frame->linesize, 0, c->height, data, linesize);

            // string window_name = "Interface " + interface_ip + ":" + to_string(port);
            cv::imshow(to_string(port), img);
            if (cv::waitKey(1) == 27) {  // ESC 키를 누르면 종료
                break;
            }
        }
    }

    sws_freeContext(sws_ctx);
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&c);
    close(sockfd);
}

int main() {
    thread server_thread1(server, SERVER_PORT);
    thread server_thread2(server, SERVER_PORT + 1);

    server_thread1.join();
    server_thread2.join();

    return 0;
}
