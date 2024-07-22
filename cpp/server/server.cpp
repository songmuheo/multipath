// server/server.cpp

#include <iostream>
#include <fstream>
#include <map>
#include <chrono>
#include <thread>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#include "config.h"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

using namespace std;

// 패킷 정보 구조체 정의
struct PacketInfo {
    double arrival_time;
    AVPacket* pkt;
};

// 시퀀스 번호를 키로 하는 패킷 맵을 인터페이스별로 구분
map<int, PacketInfo> interface1_packets;
map<int, PacketInfo> interface2_packets;
mutex mtx; // 쓰레드 안전성을 위한 뮤텍스

void log_packet(const char* interface_ip, int interface_id, int sequence, double latency) {
    ofstream log_file(LOG_FILE_PATH, ios::app);
    log_file << interface_ip << "," << interface_id << "," << sequence << "," << latency << endl;
}

void process_frame(AVCodecContext* c, AVFrame* frame, cv::Mat& img) {
    struct SwsContext* sws_ctx = sws_getContext(
        c->width, c->height, c->pix_fmt,
        c->width, c->height, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, NULL, NULL, NULL
    );

    img = cv::Mat(c->height, c->width, CV_8UC3);
    uint8_t* data[1] = { img.data };
    int linesize[1] = { static_cast<int>(img.step1()) };

    sws_scale(sws_ctx, frame->data, frame->linesize, 0, c->height, data, linesize);
    sws_freeContext(sws_ctx);

    // 단일 윈도우에 표시
    cv::imshow("Multipath Video Stream", img);
    cv::waitKey(1);
}

void server(int port, map<int, PacketInfo>& packet_map) {
    int sockfd;
    struct sockaddr_in servaddr, cliaddr;
    char buffer[PACKET_SIZE + 8];  // 패킷 크기를 위한 버퍼 선언
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

    avcodec_register_all();  // deprecated 경고 무시하고 사용

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
        exit(EXIT_FAILURE);
    }

    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        cerr << "Could not allocate video frame" << endl;
        exit(EXIT_FAILURE);
    }

    cv::Mat img;

    while (true) {
        len = sizeof(cliaddr);
        int n = recvfrom(sockfd, buffer, PACKET_SIZE + 8, 0, (struct sockaddr*)&cliaddr, &len);
        if (n < 0) {
            perror("recvfrom error");
            continue;
        }

        double arrival_time = static_cast<double>(chrono::duration_cast<chrono::milliseconds>(
                chrono::system_clock::now().time_since_epoch()).count()) / 1000.0;
        string interface_ip = inet_ntoa(cliaddr.sin_addr);

        // 헤더에서 인터페이스 ID와 시퀀스 번호를 추출
        int interface_id, sequence;
        memcpy(&interface_id, buffer, 4);
        memcpy(&sequence, buffer + 4, 4);

        // 뮤텍스를 사용하여 패킷 처리
        {
            lock_guard<mutex> lock(mtx);
            auto& packets = (port == SERVER_PORT) ? interface1_packets : interface2_packets;
            auto it = packets.find(sequence);
            if (it == packets.end() || arrival_time < it->second.arrival_time) {
                AVPacket* pkt = av_packet_alloc();
                pkt->data = (uint8_t*)(buffer + 8);
                pkt->size = n - 8;

                packets[sequence] = {arrival_time, pkt};
            }

            // 같은 시퀀스 번호의 패킷이 두 인터페이스 모두에서 도착했는지 확인
            if (interface1_packets.find(sequence) != interface1_packets.end() &&
                interface2_packets.find(sequence) != interface2_packets.end()) {
                // 더 빨리 도착한 패킷을 선택
                PacketInfo& selected_packet = (interface1_packets[sequence].arrival_time < interface2_packets[sequence].arrival_time) ?
                                              interface1_packets[sequence] : interface2_packets[sequence];

                if (avcodec_send_packet(c, selected_packet.pkt) < 0) {
                    cerr << "Error sending a packet for decoding" << endl;
                    continue;
                }

                while (avcodec_receive_frame(c, frame) == 0) {
                    process_frame(c, frame, img);
                }

                // 로그 기록
                log_packet(interface_ip.c_str(), interface_id, sequence, selected_packet.arrival_time);

                // 사용한 패킷 삭제
                av_packet_free(&interface1_packets[sequence].pkt);
                av_packet_free(&interface2_packets[sequence].pkt);
                interface1_packets.erase(sequence);
                interface2_packets.erase(sequence);
            }
        }
    }

    av_frame_free(&frame);
    avcodec_free_context(&c);
}

int main() {
    ofstream log_file(LOG_FILE_PATH);
    log_file << "Interface IP,Interface ID,Sequence Number,Latency" << endl;

    thread server_thread1(server, SERVER_PORT, ref(interface1_packets));
    thread server_thread2(server, SERVER_PORT + 1, ref(interface2_packets));

    server_thread1.join();
    server_thread2.join();

    return 0;
}
