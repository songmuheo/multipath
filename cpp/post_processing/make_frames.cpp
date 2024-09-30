// make_frames.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include "config.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// 에러 메시지 헬퍼 함수
static std::string get_av_error(int errnum) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, errbuf, sizeof(errbuf));
    return std::string(errbuf);
}

// bin files을 가져오고, sequence 번호대로 정렬한다
std::vector<std::string> get_bin_files_sorted(const std::string& folder_path) {
    std::vector<std::string> bin_files;

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".bin") {
            bin_files.push_back(entry.path().string());
        }
    }

    // 시퀀스 번호로 정렬
    std::sort(bin_files.begin(), bin_files.end(), [](const std::string& a, const std::string& b) {
        std::string a_filename = fs::path(a).filename().string();
        std::string b_filename = fs::path(b).filename().string();

        int a_seq = std::stoi(a_filename.substr(0, a_filename.find('_')));
        int b_seq = std::stoi(b_filename.substr(0, b_filename.find('_')));

        return a_seq < b_seq;
    });

    return bin_files;
}

void process_decoded_frame(AVFrame* frame, int sequence_number, uint64_t timestamp_sending, uint64_t received_time, uint64_t timestamp_frame, const std::string& output_dir) {
    int width = frame->width;
    int height = frame->height;

    static SwsContext* sws_ctx = nullptr;
    if (!sws_ctx) {
        sws_ctx = sws_getContext(
            width, height, (AVPixelFormat)frame->format,
            width, height, AV_PIX_FMT_BGR24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
    }

    AVFrame* bgr_frame = av_frame_alloc();
    int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, width, height, 1);
    std::vector<uint8_t> bgr_buffer(num_bytes);
    av_image_fill_arrays(bgr_frame->data, bgr_frame->linesize, bgr_buffer.data(), AV_PIX_FMT_BGR24, width, height, 1);

    sws_scale(sws_ctx, frame->data, frame->linesize, 0, height, bgr_frame->data, bgr_frame->linesize);

    cv::Mat img(height, width, CV_8UC3, bgr_buffer.data(), bgr_frame->linesize[0]);

    std::ostringstream oss;
    oss << output_dir << "/"
        << sequence_number << "_"
        << timestamp_sending << "_"
        << received_time << "_"
        << timestamp_frame << ".png";
    std::string frame_filepath = oss.str();

    cv::imwrite(frame_filepath, img);
    std::cout << "save complete";
    av_frame_free(&bgr_frame);
}

// void flush_decoder(AVCodecContext* codec_ctx, const std::string& output_dir_base, const std::string& stream_name, const std::string& delay_label) {
//     AVFrame* frame = av_frame_alloc();
//     int ret;
//     while ((ret = avcodec_receive_frame(codec_ctx, frame)) == 0) {
//         std::string output_dir = output_dir_base + "_delay_" + delay_label;
//         fs::create_directories(output_dir);
//         process_decoded_frame(frame, -1, 0, 0, frame->pts, output_dir);  // 임의의 sequence_number와 timestamp들
//         std::cout << "[" << stream_name << " | delay_" << delay_label << "] 플러쉬된 프레임 pts: " << frame->pts << "이(가) 저장되었습니다." << std::endl;
//         av_frame_free(&frame);
//         frame = av_frame_alloc();
//     }

//     if (ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
//         std::cerr << "[" << stream_name << "] Error receiving frame from decoder during flush: " << get_av_error(ret) << std::endl;
//     }

//     av_frame_free(&frame);
// }

void process_stream(const std::string& stream_name) {
    std::string bin_folder = BINS_FILEPATH + stream_name;
    std::string output_dir_base = FRAMES_OUT_FILEPATH + stream_name;

    std::vector<std::string> bin_files = get_bin_files_sorted(bin_folder);

    std::string csv_file_path = CSV_FILEPATH + stream_name + "_log.csv";

    std::ifstream csv_file(csv_file_path);
    std::string header_line;
    std::getline(csv_file, header_line);

    std::istringstream header_stream(header_line);
    std::vector<std::string> headers;
    std::string header;
    while (std::getline(header_stream, header, ',')) {
        headers.push_back(header);
    }

    auto it = std::find(headers.begin(), headers.end(), "sequence number");
    if (it == headers.end()) {
        std::cerr << "sequence number column not found in CSV file." << std::endl;
        return;
    }
    size_t sequence_col_index = std::distance(headers.begin(), it);

    std::map<int, std::vector<std::string>> csv_data;
    std::string line;
    while (std::getline(csv_file, line)) {
        std::istringstream line_stream(line);
        std::vector<std::string> columns;
        std::string column;
        while (std::getline(line_stream, column, ',')) {
            columns.push_back(column);
        }
        int sequence_number = std::stoi(columns[sequence_col_index]);
        csv_data[sequence_number] = columns;
    }
    csv_file.close();

    // 각 DELAY에 대해 별도의 디코더 인스턴스를 생성하고 사용
    for (const auto& delay : DELAYS) {
        std::string delay_label = DELAY_LABELS[&delay - DELAYS];

        const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
        AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);

        // 디코더 설정 수정
        // 버퍼링을 없애기 위한 ? -> 디코더 안의 버퍼링
        codec_ctx->thread_type = FF_THREAD_SLICE;
        codec_ctx->thread_count = 1;
        codec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;

        if (!codec || !codec_ctx) {
            std::cerr << "Codec not found or could not allocate context for delay " << delay_label << std::endl;
            continue;
        }

        if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            std::cerr << "Could not open codec for delay " << delay_label << std::endl;
            avcodec_free_context(&codec_ctx);
            continue;
        }

        size_t total_latency_index = std::find(headers.begin(), headers.end(), "total_latency_" + delay_label) - headers.begin();
        size_t is_use_index = std::find(headers.begin(), headers.end(), "is_use_" + delay_label) - headers.begin();

        bool total_latency_exists = (total_latency_index < headers.size());
        bool is_use_exists = (is_use_index < headers.size());

        if (!total_latency_exists) {
            headers.push_back("total_latency_" + delay_label);
            total_latency_index = headers.size() - 1;
        }
        if (!is_use_exists) {
            headers.push_back("is_use_" + delay_label);
            is_use_index = headers.size() - 1;
        }

        for (const auto& bin_file : bin_files) {
            std::string filename = fs::path(bin_file).filename().string();
            std::istringstream iss(filename);
            std::vector<std::string> tokens;
            std::string token;
            while (std::getline(iss, token, '_')) {
                tokens.push_back(token);
            }
            if (tokens.size() < 4) {
                std::cerr << "Invalid filename format: " << filename << std::endl;
                continue;
            }

            int sequence_number = std::stoi(tokens[0]);
            uint64_t timestamp_sending = std::stoull(tokens[1]);
            uint64_t received_time = std::stoull(tokens[2]);
            uint64_t timestamp_frame = std::stoull(tokens[3].substr(0, tokens[3].find('.')));

            int64_t network_latency = received_time - timestamp_sending;
            int64_t total_latency = network_latency;
            bool is_use = (total_latency <= delay);

            if (csv_data.find(sequence_number) != csv_data.end()) {
                auto& columns = csv_data[sequence_number];

                if (columns.size() <= total_latency_index) {
                    columns.resize(total_latency_index + 1, "");
                }
                columns[total_latency_index] = std::to_string(total_latency / 1000.0);

                if (columns.size() <= is_use_index) {
                    columns.resize(is_use_index + 1, "");
                }
                columns[is_use_index] = is_use ? "true" : "false";
            }

            if (is_use) {
                std::ifstream infile(bin_file, std::ios::binary);
                if (!infile) {
                    std::cerr << "Could not open file: " << bin_file << std::endl;
                    avcodec_free_context(&codec_ctx);
                    return;
                }

                std::vector<uint8_t> encoded_data((std::istreambuf_iterator<char>(infile)),
                                                  std::istreambuf_iterator<char>());
                infile.close();

                AVPacket* packet = av_packet_alloc();
                packet->data = encoded_data.data();
                packet->size = encoded_data.size();

                int ret = avcodec_send_packet(codec_ctx, packet);
                if (ret < 0) {
                    std::cerr << "Error sending packet to decoder: " << get_av_error(ret) << std::endl;
                    av_packet_free(&packet);
                    continue;
                }

                AVFrame* frame = av_frame_alloc();
                ret = avcodec_receive_frame(codec_ctx, frame);

                if (ret == 0) {
                    std::string output_dir = output_dir_base + "_delay_" + delay_label;
                    fs::create_directories(output_dir);
                    process_decoded_frame(frame, sequence_number, timestamp_sending, timestamp_frame, frame->pts, output_dir);
                    std::cout << "[" << stream_name << " | delay_" << delay_label << "] 프레임 seq_num: " << sequence_number << ", pts: " << frame->pts << "이(가) 사용되었습니다." << std::endl;
                    av_frame_free(&frame);
                } else if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    av_frame_free(&frame);
                } else {
                    std::cerr << "[" << stream_name << "] Error receiving frame from decoder (playback): " << get_av_error(ret) << std::endl;
                    av_frame_free(&frame);
                }

                av_packet_free(&packet);
            }
        }

        avcodec_free_context(&codec_ctx); // 각 딜레이에 대한 디코더 인스턴스 해제
    }

    std::ofstream csv_out(csv_file_path);
    if (!csv_out.is_open()) {
        std::cerr << "Failed to open CSV file for writing: " << csv_file_path << std::endl;
        return;
    }

    for (size_t i = 0; i < headers.size(); ++i) {
        csv_out << headers[i];
        if (i < headers.size() - 1) {
            csv_out << ",";
        }
    }
    csv_out << "\n";

    for (const auto& [seq_num, data] : csv_data) {
        for (size_t i = 0; i < data.size(); ++i) {
            csv_out << data[i];
            if (i < data.size() - 1) {
                csv_out << ",";
            }
        }
        csv_out << "\n";
    }
    csv_out.close();
}

int main() {
    std::vector<std::string> streams = {"kt", "lg", "combine"};

    for (const auto& stream_name : streams) {
        std::cout << "==============================\n";
        std::cout << "Stream [" << stream_name << "] processing start\n";
        process_stream(stream_name);
        std::cout << "Stream [" << stream_name << "] processing finish\n";
    }

    return 0;
}
