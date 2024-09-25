// main.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <opencv2/opencv.hpp>

#include "config.h"

namespace fs = std::filesystem;

// 에러 문자열 변환 매크로 제거 (FFmpeg에서 제공하는 av_err2str 함수 사용)
//#define av_err2str(errnum) av_make_error_string((char[AV_ERROR_MAX_STRING_SIZE]){0}, AV_ERROR_MAX_STRING_SIZE, errnum)

// 에러 메시지 헬퍼 함수
static std::string get_av_error(int errnum) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, errbuf, sizeof(errbuf));
    return std::string(errbuf);
}


struct PacketInfo {
    int sequence_number;
    uint64_t timestamp;
    uint64_t received_time;
    int64_t network_latency;
    double decoding_time;
    bool is_use;
};

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

void process_decoded_frame(AVFrame* frame, int sequence_number, uint64_t timestamp, uint64_t received_time, double play_time, const std::string& output_dir) {
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
        << timestamp << "_"
        << received_time << "_"
        << std::fixed << std::setprecision(3) << play_time << ".png";
    std::string frame_filepath = oss.str();

    cv::imwrite(frame_filepath, img);

    av_frame_free(&bgr_frame);
}


void process_stream(const std::string& stream_name) {
    // 디코더 초기화
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
    AVCodecContext* codec_ctx_measure = avcodec_alloc_context3(codec);  // 디코딩 시간 측정용 디코더
    AVCodecContext* codec_ctx_playback = avcodec_alloc_context3(codec); // 실제 재생용 디코더

    if (!codec || !codec_ctx_measure || !codec_ctx_playback) {
        std::cerr << "Codec not found or could not allocate context" << std::endl;
        return;
    }

    // 디코더 열기
    if (avcodec_open2(codec_ctx_measure, codec, nullptr) < 0 || avcodec_open2(codec_ctx_playback, codec, nullptr) < 0) {
        std::cerr << "Could not open codec" << std::endl;
        return;
    }

    // 입력 폴더 및 출력 폴더 설정
    std::string bin_folder = "/home/songmu/Downloads/server/bins/" + stream_name;
    std::string output_dir = "/home/songmu/Downloads/frames/" + stream_name;
    fs::create_directories(output_dir);

    // CSV 파일 열기
    std::ofstream csv_file("packet_info_" + stream_name + ".csv");
    csv_file << "sequence_number,timestamp,received_time,network_latency,decoding_time,is_use\n";

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
        if (tokens.size() < 3) {
            std::cerr << "Invalid filename format: " << filename << std::endl;
            continue;
        }
        int sequence_number = std::stoi(tokens[0]);
        uint64_t timestamp = std::stoull(tokens[1]);
        uint64_t received_time = std::stoull(tokens[2].substr(0, tokens[2].find('.')));  // 확장자 제거

        // 네트워크 지연 계산 (microseconds 단위)
        int64_t network_latency = received_time - timestamp;

        // 프레임 데이터 읽기
        std::ifstream infile(bin_file, std::ios::binary);
        if (!infile) {
            std::cerr << "Could not open file: " << bin_file << std::endl;
            continue;
        }

        std::vector<uint8_t> frame_data((std::istreambuf_iterator<char>(infile)),
                                         std::istreambuf_iterator<char>());
        infile.close();

        // NAL 유닛 시작 코드 추가 (필요한 경우)
        // const uint8_t start_code[] = {0x00, 0x00, 0x00, 0x01};
        // frame_data.insert(frame_data.begin(), std::begin(start_code), std::end(start_code));

        // 디코딩 시간 측정 시작
        auto decode_start = std::chrono::high_resolution_clock::now();

        // 패킷 생성 및 데이터 설정 (디코딩 시간 측정용)
        AVPacket* packet_measure = av_packet_alloc();
        // av_init_packet(packet_measure); // av_packet_alloc()으로 대체됨
        packet_measure->data = frame_data.data();
        packet_measure->size = frame_data.size();

        // 디코더에 패킷 전송 (디코딩 시간 측정용)
        int ret_measure = avcodec_send_packet(codec_ctx_measure, packet_measure);
        if (ret_measure < 0) {
            std::cerr << "Error sending packet to decoder (measure): " << get_av_error(ret_measure) << std::endl;
            av_packet_free(&packet_measure);
            continue;
        }

        // 디코딩된 프레임 수신 (디코딩 시간 측정용)
        AVFrame* frame_measure = av_frame_alloc();
        ret_measure = avcodec_receive_frame(codec_ctx_measure, frame_measure);

        // 디코딩 시간 측정 종료
        auto decode_end = std::chrono::high_resolution_clock::now();
        double decoding_time = std::chrono::duration<double, std::micro>(decode_end - decode_start).count();  // 마이크로초 단위

        // 측정용 디코더에서 사용한 리소스 해제
        av_frame_free(&frame_measure);
        av_packet_free(&packet_measure);

        bool is_use = false;
        int64_t total_delay = network_latency + static_cast<int64_t>(decoding_time);  // 마이크로초 단위

        if (total_delay <= DELAY) {  // 50ms = 50000us
            is_use = true;

            // 패킷 생성 및 데이터 설정 (재생용 디코더)
            AVPacket* packet_playback = av_packet_alloc();
            // av_init_packet(packet_playback); // av_packet_alloc()으로 대체됨
            packet_playback->data = frame_data.data();
            packet_playback->size = frame_data.size();

            // 디코더에 패킷 전송 (재생용 디코더)
            int ret_playback = avcodec_send_packet(codec_ctx_playback, packet_playback);
            if (ret_playback < 0) {
                std::cerr << "Error sending packet to decoder (playback): " << get_av_error(ret_playback) << std::endl;
                av_packet_free(&packet_playback);
                continue;
            }

            // 디코딩된 프레임 수신 (재생용 디코더)
            AVFrame* frame_playback = av_frame_alloc();
            ret_playback = avcodec_receive_frame(codec_ctx_playback, frame_playback);

            if (ret_playback == 0) {
                // 디코딩 성공
                // 재생 시간 계산 (초 단위)
                double play_time = (timestamp + DELAY) / 1e6;  // 마이크로초 -> 초 단위

                // 디코딩된 프레임 처리 및 저장
                process_decoded_frame(frame_playback, sequence_number, timestamp, received_time, play_time, output_dir);


                std::cout << "[" << stream_name << "] 프레임 " << sequence_number << "이(가) 사용되었습니다." << std::endl;

                av_frame_free(&frame_playback);
            } else if (ret_playback == AVERROR(EAGAIN) || ret_playback == AVERROR_EOF) {
                // 추가 프레임이 없음
                av_frame_free(&frame_playback);
            } else {
                // 디코딩 실패
                std::cerr << "[" << stream_name << "] Error receiving frame from decoder (playback): " << get_av_error(ret_playback) << std::endl;
                av_frame_free(&frame_playback);
            }

            av_packet_free(&packet_playback);
        } else {
            // 조건 불만족으로 프레임 스킵 (재생용 디코더에 패킷 전달하지 않음)
            std::cout << "[" << stream_name << "] 프레임 " << sequence_number << "이(가) 조건 불만족으로 스킵되었습니다." << std::endl;
        }

        // 패킷 정보 CSV 파일에 저장
        csv_file << sequence_number << ","
                 << timestamp << ","
                 << received_time << ","
                 << network_latency << ","
                 << decoding_time << ","
                 << (is_use ? "True" : "False") << "\n";
    }

    csv_file.close();

    // 디코더 해제
    avcodec_free_context(&codec_ctx_measure);
    avcodec_free_context(&codec_ctx_playback);

    // 프레임 목록 파일 생성
    std::ofstream list_file("frames_list_" + stream_name + ".txt");
    for (const auto& entry : fs::directory_iterator(output_dir)) {
        if (entry.path().extension() == ".png") {
            std::string filepath = entry.path().string();
            std::string filename = entry.path().filename().string();
            // 파일명에서 재생 시간 추출
            std::size_t pos = filename.find_last_of('_');
            std::size_t dot_pos = filename.find_last_of('.');
            std::string time_str = filename.substr(pos + 1, dot_pos - pos - 1);
            double duration = FPS; // 기본 프레임 지속 시간 (30fps 기준)
            list_file << "file '" << filepath << "'\n";
            list_file << "duration " << duration << "\n";
        }
    }
    list_file.close();

    // FFmpeg를 사용하여 동영상 생성
    // std::string ffmpeg_command = "ffmpeg -y -f concat -safe 0 -i frames_list_" + stream_name + ".txt -vsync vfr -pix_fmt yuv420p output_" + stream_name + ".mp4";
    // std::system(ffmpeg_command.c_str());
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
