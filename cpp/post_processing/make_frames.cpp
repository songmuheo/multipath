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

// Packet header structure
// #pragma pack(push, 1)
struct PacketHeader {
    uint64_t timestamp_frame;
    uint64_t timestamp_sending;
    uint32_t sequence_number;
};
// #pragma pack(pop)

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


// (frame, sequence_number, timestamp_sending, received_time, frame_pts, output_dir);
void process_decoded_frame(AVFrame* frame, int sequence_number, uint64_t timestamp_sending, uint64_t received_time, int64_t frame_pts, const std::string& output_dir) {
    // AVFrame을 OpenCV Mat로 변환
    int width = frame->width;
    int height = frame->height;

    // 스케일러 초기화 (정적 변수로 유지)
    static SwsContext* sws_ctx = nullptr;
    if (!sws_ctx) {
        sws_ctx = sws_getContext(
            width, height, (AVPixelFormat)frame->format,
            width, height, AV_PIX_FMT_BGR24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
    }

    // 변환된 데이터 저장을 위한 버퍼 생성
    AVFrame* bgr_frame = av_frame_alloc();
    int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, width, height, 1);
    std::vector<uint8_t> bgr_buffer(num_bytes);
    av_image_fill_arrays(bgr_frame->data, bgr_frame->linesize, bgr_buffer.data(), AV_PIX_FMT_BGR24, width, height, 1);

    // 색상 공간 변환
    sws_scale(sws_ctx, frame->data, frame->linesize, 0, height, bgr_frame->data, bgr_frame->linesize);

    // OpenCV Mat 생성
    cv::Mat img(height, width, CV_8UC3, bgr_buffer.data(), bgr_frame->linesize[0]);

    // 이미지 저장 (파일명에 sequence_number, timestamp, received_time, play_time 포함)
    std::ostringstream oss;
    oss << output_dir << "/"
        << sequence_number << "_"
        << timestamp_sending << "_"
        << received_time << "_"
        << frame_pts << ".png";
    std::string frame_filepath = oss.str();

    cv::imwrite(frame_filepath, img);
    std::cout << "save complete";
    av_frame_free(&bgr_frame);
}

void process_stream(const std::string& stream_name) {
    // 디코더 초기화
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec); // 실제 재생용 디코더

    if (!codec || !codec_ctx) {
        std::cerr << "Codec not found or could not allocate context" << std::endl;
        return;
    }

    // 디코더 열기
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Could not open codec" << std::endl;
        return;
    }

    // 입력 폴더 및 출력 폴더 설정
    std::string bin_folder = BINS_FILEPATH + stream_name;
    std::string output_dir = FRAMES_OUT_FILEPATH + stream_name;
    fs::create_directories(output_dir);

    // .bin 파일 목록 가져오기
    std::vector<std::string> bin_files = get_bin_files_sorted(bin_folder);

    for (const auto& bin_file : bin_files) {
        // 파일명에서 정보 추출
        std::string filename = fs::path(bin_file).filename().string();
        std::istringstream iss(filename);
        std::string token;
        std::vector<std::string> tokens;
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
        uint64_t timestamp_frame = std::stoull(tokens[3].substr(0, tokens[3].find('.')));   // 확장자 제거

        // 네트워크 지연 계산 (microseconds 단위)
        int64_t network_latency = received_time - timestamp_sending;

        // Open the binary file
        std::ifstream infile(bin_file, std::ios::binary);
        if (!infile) {
            std::cerr << "Could not open file: " << bin_file << std::endl;
            // Handle the error or exit the function/program
            return;
        }

        //  Read the header
        // PacketHeader header;
        // infile.read(reinterpret_cast<char*>(&header), sizeof(PacketHeader));
        // if (infile.gcount() != sizeof(PacketHeader)) {
        //     std::cerr << "Error reading header" << std::endl;
        //     infile.close(); // Close the file before returning
        //     return;
        // } 
        
        // // 나머지 데이터를 읽어서 인코딩된 데이터로 간주
        // std::vector<uint8_t> encoded_data((std::istreambuf_iterator<char>(infile)),
        //                                 std::istreambuf_iterator<char>());

        std::vector<uint8_t> encoded_data((std::istreambuf_iterator<char>(infile)),
                                         std::istreambuf_iterator<char>());

        infile.close();

        bool is_use = false;
        if (network_latency <= DELAY) {  // 50ms = 50000us
            is_use = true;

            // 패킷 생성 및 데이터 설정
            AVPacket* packet = av_packet_alloc();
            packet->data = encoded_data.data();
            packet->size = encoded_data.size();

            // 디코더에 패킷 전송 (재생용 디코더)
            int ret = avcodec_send_packet(codec_ctx, packet);
            if (ret < 0) {
                std::cerr << "Error sending packet to decoder: " << get_av_error(ret) << std::endl;
                av_packet_free(&packet);
                continue;
            }

            // 디코딩된 프레임 수신 (재생용 디코더)
            AVFrame* frame = av_frame_alloc();
            ret = avcodec_receive_frame(codec_ctx, frame);

            if (ret == 0) { // 디코딩 성공  
                // frame_pts = ;

                // 디코딩된 프레임 처리 및 저장
                process_decoded_frame(frame, sequence_number, timestamp_sending, received_time, frame->pts, output_dir);

                std::cout << "[" << stream_name << "] 프레임 seq_num: " << sequence_number << ", pts: " << frame->pts << "이(가) 사용되었습니다." << std::endl;

                av_frame_free(&frame);
            } else if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                // 추가 프레임이 없음
                av_frame_free(&frame);
            } else {
                // 디코딩 실패
                std::cerr << "[" << stream_name << "] Error receiving frame from decoder (playback): " << get_av_error(ret) << std::endl;
                av_frame_free(&frame);
            }

            av_packet_free(&packet);
        } else {
            // 조건 불만족으로 프레임 스킵 (재생용 디코더에 패킷 전달하지 않음)
            std::cout << "[" << stream_name << "] 프레임 seq_num: " << sequence_number << "이(가) 조건 불만족으로 스킵되었습니다." << std::endl;
        }
    }

    // 디코더 해제
    avcodec_free_context(&codec_ctx);
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
