#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <thread>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <set>
#include <mutex>
#include <functional>
#include "config.h"


extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

using namespace std;
using namespace cv;

struct PacketInfo {
    size_t hash;
    vector<uint8_t> data;
};

size_t generate_hash(const vector<uint8_t>& packet_data) {
    // 간단한 해시 생성 함수 (여기서는 std::hash 사용)
    return hash<string>{}(string(packet_data.begin(), packet_data.end()));
}

void receive_and_process_packets(int port, unordered_map<size_t, PacketInfo>& packet_buffer, set<size_t>& processed_hashes, mutex& buffer_mutex) {
    int sockfd;
    struct sockaddr_in servaddr;

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(port);
    servaddr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sockfd, (const struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
        perror("Bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    uint8_t buffer[65536];
    socklen_t len = sizeof(servaddr);

    while (true) {
        int n = recvfrom(sockfd, buffer, sizeof(buffer), MSG_WAITALL, (struct sockaddr*)&servaddr, &len);
        if (n > 0) {
            vector<uint8_t> packet_data(buffer, buffer + n);
            size_t packet_hash = generate_hash(packet_data);

            lock_guard<mutex> lock(buffer_mutex);

            if (processed_hashes.find(packet_hash) == processed_hashes.end()) {
                packet_buffer[packet_hash] = {packet_hash, packet_data};
                processed_hashes.insert(packet_hash);
            }
        }
    }
}

int main() {
    unordered_map<size_t, PacketInfo> packet_buffer;
    set<size_t> processed_hashes; // 이미 처리된 패킷의 해시를 추적
    mutex buffer_mutex;

    thread interface1_thread(receive_and_process_packets, SERVER_PORT, ref(packet_buffer), ref(processed_hashes), ref(buffer_mutex));
    thread interface2_thread(receive_and_process_packets, SERVER_PORT + 1, ref(packet_buffer), ref(processed_hashes), ref(buffer_mutex));

    interface1_thread.detach();
    interface2_thread.detach();

    // Initialize FFmpeg decoder
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

    SwsContext* sws_ctx = nullptr;

    while (true) {
        size_t min_hash = SIZE_MAX;
        vector<uint8_t> selected_packet;

        {
            lock_guard<mutex> lock(buffer_mutex);

            for (auto& [hash, packet_info] : packet_buffer) {
                if (hash < min_hash) {
                    min_hash = hash;
                    selected_packet = packet_info.data;
                }
            }

            if (min_hash != SIZE_MAX) {
                packet_buffer.erase(min_hash);
            }
        }

        if (!selected_packet.empty()) {
            pkt->data = selected_packet.data();
            pkt->size = selected_packet.size();

            if (avcodec_send_packet(c, pkt) < 0) {
                cerr << "Error sending packet for decoding" << endl;
                continue;
            }

            while (avcodec_receive_frame(c, frame) == 0) {
                if (!sws_ctx) {
                    sws_ctx = sws_getContext(frame->width, frame->height, c->pix_fmt,
                                             frame->width, frame->height, AV_PIX_FMT_BGR24,
                                             SWS_BILINEAR, NULL, NULL, NULL);
                }

                uint8_t* data[1] = { new uint8_t[frame->width * frame->height * 3] };
                int linesize[1] = { 3 * frame->width };

                sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, data, linesize);

                Mat img(frame->height, frame->width, CV_8UC3, data[0]);

                imshow("Received Video", img);
                delete[] data[0];

                if (waitKey(30) >= 0) {
                    break;
                }
            }
        }

        this_thread::sleep_for(chrono::milliseconds(10));
    }

    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&c);
    if (sws_ctx) sws_freeContext(sws_ctx);

    return 0;
}
