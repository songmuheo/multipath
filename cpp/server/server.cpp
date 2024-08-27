#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unordered_set>
#include <vector>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <sys/select.h>
#include "config.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

using namespace std;

struct PacketHeader {
    uint64_t timestamp;  // 8 bytes
    uint32_t sequence_number;  // 4 bytes
};

class VideoReceiver {
public:
    VideoReceiver(int port, const string& output_filename, const string& window_name, bool is_merged = false)
        : port(port), output_filename(output_filename), window_name(window_name), running(true), is_merged(is_merged) {
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) {
            perror("Socket creation failed");
            exit(EXIT_FAILURE);
        }

        memset(&servaddr, 0, sizeof(servaddr));
        servaddr.sin_family = AF_INET;
        servaddr.sin_port = htons(port);
        servaddr.sin_addr.s_addr = inet_addr(SERVER_IP);

        if (bind(sockfd, (const struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
            perror("Bind failed");
            close(sockfd);
            exit(EXIT_FAILURE);
        }

        if (!is_merged) {
            output_file.open(output_filename, ios::out | ios::binary);
            if (!output_file.is_open()) {
                cerr << "Could not open output file: " << output_filename << endl;
                close(sockfd);
                exit(EXIT_FAILURE);
            }
        }

        codec = avcodec_find_decoder(AV_CODEC_ID_H264);
        if (!codec) {
            cerr << "Codec not found" << endl;
            exit(EXIT_FAILURE);
        }

        codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            cerr << "Could not allocate video codec context" << endl;
            exit(EXIT_FAILURE);
        }

        if (avcodec_open2(codec_ctx, codec, NULL) < 0) {
            cerr << "Could not open codec" << endl;
            exit(EXIT_FAILURE);
        }

        frame = av_frame_alloc();
        if (!frame) {
            cerr << "Could not allocate video frame" << endl;
            exit(EXIT_FAILURE);
        }

        pkt = av_packet_alloc();
        if (!pkt) {
            cerr << "Could not allocate AVPacket" << endl;
            exit(EXIT_FAILURE);
        }

        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    }

    ~VideoReceiver() {
        running = false;
        if (recv_thread.joinable()) {
            recv_thread.join();
        }
        if (decode_thread.joinable()) {
            decode_thread.join();
        }
        if (!is_merged) {
            output_file.close();
        }
        close(sockfd);
        av_frame_free(&frame);
        av_packet_free(&pkt);
        avcodec_free_context(&codec_ctx);
    }

    void start() {
        if (is_merged) {
            recv_thread = thread(&VideoReceiver::receive_packets_merged, this);
        } else {
            recv_thread = thread(&VideoReceiver::receive_packets, this);
        }
        decode_thread = thread(&VideoReceiver::decode_and_display, this);
    }

private:
    void receive_packets() {
        char buffer[PACKET_SIZE];
        socklen_t len;
        struct sockaddr_in cliaddr;

        while (running) {
            len = sizeof(cliaddr);
            int n = recvfrom(sockfd, buffer, PACKET_SIZE, 0, (struct sockaddr*)&cliaddr, &len);
            if (n > sizeof(PacketHeader)) {
                PacketHeader header;
                memcpy(&header, buffer, sizeof(PacketHeader));

                vector<uint8_t> packet_data(buffer + sizeof(PacketHeader), buffer + n);

                {
                    unique_lock<mutex> lock(queue_mutex);
                    packet_queue.push(make_pair(header, packet_data));
                    queue_cond.notify_one();
                }

                uint64_t receive_time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
                uint64_t latency = receive_time - header.timestamp;
                cout << "Sequence Number: " << header.sequence_number << ", Latency: " << latency << " ms" << endl;

                if (!is_merged) {
                    output_file.write(buffer, n);
                }
            }
        }
    }

    void receive_packets_merged() {
        int sockfd1, sockfd2;
        struct sockaddr_in servaddr1, servaddr2;

        // 첫 번째 포트에 대한 소켓 설정
        sockfd1 = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd1 < 0) {
            perror("Socket creation failed");
            exit(EXIT_FAILURE);
        }

        memset(&servaddr1, 0, sizeof(servaddr1));
        servaddr1.sin_family = AF_INET;
        servaddr1.sin_port = htons(port);  // 기본 포트 사용
        servaddr1.sin_addr.s_addr = inet_addr(SERVER_IP);

        if (bind(sockfd1, (const struct sockaddr*)&servaddr1, sizeof(servaddr1)) < 0) {
            perror("Bind failed on port 1");
            close(sockfd1);
            exit(EXIT_FAILURE);
        }

        // 두 번째 포트에 대한 소켓 설정
        sockfd2 = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd2 < 0) {
            perror("Socket creation failed");
            exit(EXIT_FAILURE);
        }

        memset(&servaddr2, 0, sizeof(servaddr2));
        servaddr2.sin_family = AF_INET;
        servaddr2.sin_port = htons(port + 1);  // 두 번째 포트 사용
        servaddr2.sin_addr.s_addr = inet_addr(SERVER_IP);

        if (bind(sockfd2, (const struct sockaddr*)&servaddr2, sizeof(servaddr2)) < 0) {
            perror("Bind failed on port 2");
            close(sockfd2);
            exit(EXIT_FAILURE);
        }

        fd_set readfds;
        char buffer[PACKET_SIZE];
        socklen_t len;
        struct sockaddr_in cliaddr;

        while (running) {
            FD_ZERO(&readfds);
            FD_SET(sockfd1, &readfds);
            FD_SET(sockfd2, &readfds);

            int max_fd = max(sockfd1, sockfd2) + 1;
            struct timeval timeout = {0, 10000};  // 10ms timeout to reduce CPU usage
            int activity = select(max_fd, &readfds, NULL, NULL, &timeout);

            if (activity < 0 && errno != EINTR) {
                perror("Select error");
                break;
            }

            int n;
            if (FD_ISSET(sockfd1, &readfds)) {
                len = sizeof(cliaddr);
                n = recvfrom(sockfd1, buffer, PACKET_SIZE, 0, (struct sockaddr*)&cliaddr, &len);
            } else if (FD_ISSET(sockfd2, &readfds)) {
                len = sizeof(cliaddr);
                n = recvfrom(sockfd2, buffer, PACKET_SIZE, 0, (struct sockaddr*)&cliaddr, &len);
            } else {
                continue;  // Timeout, check again
            }

            if (n > sizeof(PacketHeader)) {
                PacketHeader header;
                memcpy(&header, buffer, sizeof(PacketHeader));

                vector<uint8_t> packet_data(buffer + sizeof(PacketHeader), buffer + n);

                {
                    unique_lock<mutex> lock(seq_mutex);
                    if (received_sequences.count(header.sequence_number)) {
                        continue;  // 중복된 패킷 무시
                    }
                    received_sequences.insert(header.sequence_number);
                }

                {
                    unique_lock<mutex> lock(queue_mutex);
                    packet_queue.push(make_pair(header, packet_data));
                    queue_cond.notify_one();
                }

                uint64_t receive_time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
                uint64_t latency = receive_time - header.timestamp;
                cout << "Sequence Number: " << header.sequence_number << ", Latency: " << latency << " ms" << endl;

                output_file.write(buffer, n);
            }
        }

        close(sockfd1);
        close(sockfd2);
    }

    void decode_and_display() {
        struct SwsContext* sws_ctx = nullptr;
        cv::Mat img;

        while (running) {
            pair<PacketHeader, vector<uint8_t>> packet_info;

            {
                unique_lock<mutex> lock(queue_mutex);
                queue_cond.wait(lock, [this] { return !packet_queue.empty() || !running; });
                if (!running && packet_queue.empty()) break;
                packet_info = packet_queue.front();
                packet_queue.pop();
            }

            pkt->data = packet_info.second.data();
            pkt->size = packet_info.second.size();

            if (avcodec_send_packet(codec_ctx, pkt) >= 0) {
                while (avcodec_receive_frame(codec_ctx, frame) >= 0) {
                    if (!sws_ctx) {
                        sws_ctx = sws_getContext(codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
                                                 codec_ctx->width, codec_ctx->height, AV_PIX_FMT_BGR24,
                                                 SWS_BILINEAR, nullptr, nullptr, nullptr);

                        img = cv::Mat(codec_ctx->height, codec_ctx->width, CV_8UC3);
                    }

                    uint8_t* data[1] = { img.data };
                    int linesize[1] = { static_cast<int>(img.step1()) };
                    sws_scale(sws_ctx, frame->data, frame->linesize, 0, codec_ctx->height, data, linesize);

                    cv::imshow(window_name, img);
                    if (cv::waitKey(1) == 27) {
                        running = false;
                        break;
                    }
                }
            }
        }

        if (sws_ctx) {
            sws_freeContext(sws_ctx);
        }
    }

    int sockfd;
    int port;
    string output_filename;
    string window_name;
    ofstream output_file;
    atomic<bool> running;
    thread recv_thread;
    thread decode_thread;
    queue<pair<PacketHeader, vector<uint8_t>>> packet_queue;
    mutex queue_mutex;
    condition_variable queue_cond;
    struct sockaddr_in servaddr;

    AVCodec* codec;
    AVCodecContext* codec_ctx;
    AVFrame* frame;
    AVPacket* pkt;

    bool is_merged;
    mutex seq_mutex;
    unordered_set<uint32_t> received_sequences;
};

int main() {
    // 개별 인터페이스에서의 비디오 수신기
    VideoReceiver receiver1(SERVER_PORT, "LGUp.h264", "LGU+");
    VideoReceiver receiver2(SERVER_PORT + 1, "KT.h264", "KT");

    // 중복 제거된 통합 비디오 수신기
    VideoReceiver mergedReceiver(SERVER_PORT, "MergedOutput.h264", "Merged Stream", true);

    receiver1.start();
    receiver2.start();
    mergedReceiver.start();

    cout << "Press Enter to stop..." << endl;
    cin.get();

    return 0;
}
