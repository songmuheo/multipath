// server/server.cpp

#include <iostream>
#include <fstream>
#include <map>
#include <chrono>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#include "config.h"
// FFmpeg의 일부 -> 비디오 코덱, 포맷, 스케일링 기능 제공
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

using namespace std;

// 패킷 정보를 LOG_FILE_PATH에 기록함
void log_packet(const char* interface_ip, int interface_id, int sequence, double latency) {
    ofstream log_file(LOG_FILE_PATH, ios::app);
    log_file << interface_ip << "," << interface_id << "," << sequence << "," << latency << endl;
}

void server() {
    int sockfd;
    struct sockaddr_in servaddr, cliaddr;
    char buffer[PACKET_SIZE];  // 패킷 크기를 위한 버퍼 선언
    socklen_t len;

    // IPv4 사용, UDP 소켓 타입, '0': 기본 프로토콜(UDP), 실패시 -1을 return
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // 메모리를 초기화 함
    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));

    // sing_family - 주소 체계(IPv4), sin_port - 포트 번호(네트워크 바이트 순서로 변환), sin_addr.s_addr - IP주소(문자열 -> 네트워크 바이트 순서로 변환)
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(SERVER_PORT);
    servaddr.sin_addr.s_addr = inet_addr(SERVER_IP);

    // 소켓을 특정 IP 주소와 포트 번호에 바인딩, 실패시 예외 처리
    if (bind(sockfd, (const struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    // 모든 코덱과 포맷을 등록 -> (Deprecated 하다고 함?)
    avcodec_register_all();  // deprecated 경고 무시하고 사용

    // H.264 디코더를 찾음 -> 실패시 NULL을 리턴
    AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) {
        cerr << "Codec not found" << endl;
        exit(EXIT_FAILURE);
    }

    // 코덱 컨텍스트를 할당
    AVCodecContext* c = avcodec_alloc_context3(codec);
    if (!c) {
        cerr << "Could not allocate video codec context" << endl;
        exit(EXIT_FAILURE);
    }

    // 코덱 컨텍스트를 초기화 -> 실패시 음수, 성공시 0
    if (avcodec_open2(c, codec, NULL) < 0) {
        cerr << "Could not open codec" << endl;
        exit(EXIT_FAILURE);
    }

    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        cerr << "Could not allocate video frame" << endl;
        exit(EXIT_FAILURE);
    }

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        cerr << "Could not allocate AVPacket" << endl;
        exit(EXIT_FAILURE);
    }

    // 인터페이스별로 이미지를 저장할 맵
    map<string, cv::Mat> frames;

    while (true) {
        len = sizeof(cliaddr);
        int n = recvfrom(sockfd, buffer, PACKET_SIZE, 0, (struct sockaddr*)&cliaddr, &len);
        if (n < 0) {
            perror("recvfrom error");
            continue;
        }

        double arrival_time = static_cast<double>(chrono::system_clock::now().time_since_epoch().count() / 1000000.0);
        string interface_ip = inet_ntoa(cliaddr.sin_addr);

        pkt->data = (uint8_t*)buffer;
        pkt->size = n;

        // 패킷을 디코더로 전송
        if (avcodec_send_packet(c, pkt) < 0) {
            cerr << "Error sending a packet for decoding" << endl;
            continue;
        }

        // 디코더로부터 프레임 수신
        while (avcodec_receive_frame(c, frame) == 0) {
            struct SwsContext* sws_ctx = sws_getContext(
                c->width, c->height, c->pix_fmt,
                c->width, c->height, AV_PIX_FMT_BGR24,
                SWS_BILINEAR, NULL, NULL, NULL
            );

            cv::Mat img(c->height, c->width, CV_8UC3);
            uint8_t* data[1] = { img.data };
            int linesize[1] = { static_cast<int>(img.step1()) };

            sws_scale(sws_ctx, frame->data, frame->linesize, 0, c->height, data, linesize);
            sws_freeContext(sws_ctx);

            frames[interface_ip] = img.clone();

            cv::imshow("Interface " + interface_ip, frames[interface_ip]);
            cv::waitKey(1);
        }

        log_packet(interface_ip.c_str(), 0, 0, 0);  // logging is simplified for this example
    }

    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&c);
}

int main() {
    ofstream log_file(LOG_FILE_PATH);
    log_file << "Interface IP,Interface ID,Sequence Number,Latency" << endl;

    server();

    return 0;
}
