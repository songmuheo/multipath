#include <iostream>
#include <fstream>
#include <atomic>
#include <thread>
#include <vector>
#include <future>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <arpa/inet.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <unistd.h>
#include <filesystem>
#include <cerrno>
#include <cstring>
#include "config.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

using namespace std;
namespace fs = std::filesystem;

struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};

class VideoStreamer {
public:
    VideoStreamer() : frame_counter(0), sequence_number(0) {
        // Create the output folders for frames and logs
        create_and_set_output_folders();

        // Open the CSV log file
        open_log_file();

        codec = avcodec_find_encoder(AV_CODEC_ID_HEVC);
        if (!codec) throw runtime_error("Codec not found");

        codec_ctx.reset(avcodec_alloc_context3(codec));
        if (!codec_ctx) throw runtime_error("Could not allocate video codec context");

        codec_ctx->bit_rate = BITRATE;
        codec_ctx->width = WIDTH;
        codec_ctx->height = HEIGHT;
        codec_ctx->time_base = { 1, FPS };
        codec_ctx->framerate = { FPS, 1 };
        codec_ctx->gop_size = 10;
        codec_ctx->max_b_frames = 0;
        codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        codec_ctx->thread_count = 4;

        av_opt_set(codec_ctx->priv_data, "scenecut", "0", 0);   // Scene Change Detection 비활성화
        av_opt_set(codec_ctx->priv_data, "b-adapt", "0", 0); // B-프레임 적응 비활성화
        av_opt_set(codec_ctx->priv_data, "bframes", "0", 0); // B-프레임 수를 0으로 설정
        av_opt_set(codec_ctx->priv_data, "keyint", "10", 0);    // GoP 크기를 10으로 고정
        av_opt_set(codec_ctx->priv_data, "min-keyint", "10", 0);    // 최소 GoP 크기를 10으로 설정
        av_opt_set(codec_ctx->priv_data, "max-keyint", "10", 0);    // 최대 GoP 크기를 10으로 설정
        av_opt_set(codec_ctx->priv_data, "no-open-gop", "1", 0); // Open-GOP 비활성화
        av_opt_set(codec_ctx->priv_data, "vbv-bufsize", "0", 0); // VBV 버퍼 크기 제거
        av_opt_set(codec_ctx->priv_data, "vbv-maxrate", "0", 0); // VBV 최대 비트레이트 제거
        av_opt_set(codec_ctx->priv_data, "preset", "ultrafast", 0);

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
        log_file.close();
    }

    void stream(rs2::video_frame& color_frame, uint64_t timestamp_frame) {
        frame->pts = frame_counter++;

        uint8_t* yuyv_data = (uint8_t*)color_frame.get_data();

        const uint8_t* src_slices[1] = { yuyv_data };
        int src_stride[1] = { 2 * WIDTH };
        sws_scale(sws_ctx, src_slices, src_stride, 0, HEIGHT, frame->data, frame->linesize);

        // 프레임 저장
        save_frame(color_frame, timestamp_frame);

        // 프레임 인코딩 및 전송
        encode_and_send_frame(timestamp_frame);
    }

    atomic<int> sequence_number;

private:
    void create_and_set_output_folders() {
        auto now = chrono::system_clock::now();
        time_t time_now = chrono::system_clock::to_time_t(now);
        tm local_time = *localtime(&time_now);

        char folder_name[100];
        strftime(folder_name, sizeof(folder_name), "%Y_%m_%d_%H_%M", &local_time);

        // Root directory based on current time
        string base_folder = SAVE_FILEPATH + string(folder_name);
        fs::create_directories(base_folder);

        // Create frames and logs subdirectories
        frames_folder = base_folder + "/frames";
        logs_folder = base_folder + "/logs";
        fs::create_directories(frames_folder);
        fs::create_directories(logs_folder);
    }

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

    void save_frame(const rs2::video_frame& frame_data, uint64_t timestamp_frame) {
        // Save frames to the "frames" folder
        string filename = frames_folder + "/" + to_string(timestamp_frame) + ".png";

        cv::Mat yuyv_image(HEIGHT, WIDTH, CV_8UC2, (void*)frame_data.get_data());
        cv::Mat bgr_image;
        cv::cvtColor(yuyv_image, bgr_image, cv::COLOR_YUV2BGR_YUYV);

        cv::imwrite(filename, bgr_image);
    }

    void encode_and_send_frame(uint64_t timestamp_frame) {
        if (avcodec_send_frame(codec_ctx.get(), frame.get()) < 0) {
            cerr << "Error sending a frame for encoding" << endl;
            return;
        }

        int ret;
        int packet_count = 0;
        while ((ret = avcodec_receive_packet(codec_ctx.get(), pkt.get())) == 0) {
            PacketHeader header;
            // Frame이 생성 됐을 때의 time -> 현재 Encoded data의 sequence_number와 같은 frame은 아니다(GoP 등의 설정에 따라 다름)
            header.timestamp_frame = timestamp_frame;
            // Encoded data가 생성됐을 때의 time (Sending하기 직전의 time)
            header.timestamp_sending = chrono::duration_cast<chrono::microseconds>(
                chrono::system_clock::now().time_since_epoch()).count();
            header.sequence_number = sequence_number++;

            vector<uint8_t> packet_data(sizeof(PacketHeader) + pkt->size);
            memcpy(packet_data.data(), &header, sizeof(PacketHeader));
            memcpy(packet_data.data() + sizeof(PacketHeader), pkt->data, pkt->size);

            log_packet_to_csv(sequence_number - 1, pkt->size, header.timestamp_frame, header.timestamp_sending, frame->pts);

            auto send_task2 = async(launch::async, [this, &packet_data] {
                if (sendto(sockfd2, packet_data.data(), packet_data.size(), 0, (const struct sockaddr*)&servaddr2, sizeof(servaddr2)) < 0) {
                    cerr << "Error sending packet on interface 2: " << strerror(errno) << endl;
                }
            });
            
            auto send_task1 = async(launch::async, [this, &packet_data] {
                if (sendto(sockfd1, packet_data.data(), packet_data.size(), 0, (const struct sockaddr*)&servaddr1, sizeof(servaddr1)) < 0) {
                    cerr << "Error sending packet on interface 1: " << strerror(errno) << endl;
                }
            });

            send_task1.get();
            send_task2.get();

            av_packet_unref(pkt.get());
            packet_count++;
        }

        if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
            cerr << "Error receiving encoded packet" << endl;
        }
    }

    void open_log_file() {
        // Save logs to the "logs" folder
        log_file.open(logs_folder + "/packet_log.csv");
        if (!log_file.is_open()) {
            throw runtime_error("Failed to open CSV log file");
        }
        log_file << "SequenceNumber,Size,Timestamp_frame,Timstamp_sending,PTS\n";
    }

    void log_packet_to_csv(int sequence_number, int size, uint64_t timestamp, uint64_t sendtime, int64_t pts) {
        log_file << sequence_number << "," 
                 << size << "," 
                 << timestamp << "," 
                 << sendtime << ","
                 << pts << "," 
                 << "\n";
    }

    const AVCodec* codec;
    unique_ptr<AVCodecContext, void(*)(AVCodecContext*)> codec_ctx{nullptr, [](AVCodecContext* p) { avcodec_free_context(&p); }};
    unique_ptr<AVFrame, void(*)(AVFrame*)> frame{nullptr, [](AVFrame* p) { av_frame_free(&p); }};
    unique_ptr<AVPacket, void(*)(AVPacket*)> pkt{nullptr, [](AVPacket* p) { av_packet_free(&p); }};

    SwsContext* sws_ctx = nullptr;
    ofstream log_file;

    int sockfd1, sockfd2;
    struct sockaddr_in servaddr1, servaddr2;
    atomic<int> frame_counter;

    string frames_folder;  // Path to store frames
    string logs_folder;    // Path to store logs
};

void client(VideoStreamer& streamer, rs2::pipeline& pipe, atomic<bool>& running) {
    while (running.load()) {
        rs2::frameset frames;
        if (pipe.poll_for_frames(&frames)) {
            rs2::video_frame color_frame = frames.get_color_frame();
            if (!color_frame) continue;
            // frame 생성시의 timestamp 생성
            uint64_t timestamp_frame = chrono::duration_cast<chrono::microseconds>(
                chrono::system_clock::now().time_since_epoch()).count();

            streamer.stream(color_frame, timestamp_frame);
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
        thread client_thread(client, ref(streamer), ref(pipe), ref(running));

        cout << "Press Enter to stop streaming..." << endl;
        cin.get();

        running.store(false);
        client_thread.join();
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
