// decoder.cpp

#include "decoder.h"
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h> // py::bytes 사용을 위한 pybind11 헤더 포함
#include <pybind11/stl.h>      // Python <-> C++ STL 변환을 위한 헤더 포함
namespace py = pybind11;       // pybind11을 py로 간략화
extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

Decoder::Decoder() {
    init_decoder();
}

void Decoder::init_decoder() {
    // avcodec_register_all(); // FFmpeg 초기화 (필요에 따라 추가)

    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) throw std::runtime_error("Codec not found");

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) throw std::runtime_error("Could not allocate video codec context");

    codec_ctx->thread_type = FF_THREAD_SLICE;
    codec_ctx->thread_count = 1;
    codec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        throw std::runtime_error("Could not open codec");
    }

    frame = av_frame_alloc();
    if (!frame) throw std::runtime_error("Could not allocate video frame");

    pkt = av_packet_alloc();
    if (!pkt) throw std::runtime_error("Could not allocate AVPacket");

    sws_ctx = nullptr;
}

py::array_t<uint8_t> Decoder::decode_frame(const py::bytes& encoded_data_py, int size, int width, int height) {
    // 입력 데이터를 uint8_t 포인터로 변환
    std::string encoded_data = static_cast<std::string>(encoded_data_py);
    if (encoded_data.empty()) {
        throw std::runtime_error("Encoded data is empty");
    }

    pkt->data = reinterpret_cast<uint8_t*>(const_cast<char*>(encoded_data.data()));
    pkt->size = size;

    // 패킷을 디코더로 전송
    int ret = avcodec_send_packet(codec_ctx, pkt);
    if (ret < 0) {
        throw std::runtime_error("Error sending packet for decoding");
    }

    cv::Mat decoded_frame;
    // 프레임 수신 및 처리
    while (ret >= 0) {
        ret = avcodec_receive_frame(codec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break; // 더 이상 프레임이 없음
        } else if (ret < 0) {
            throw std::runtime_error("Error during decoding");
        }

        // 첫 번째 프레임에서 sws_ctx 초기화
        if (!sws_ctx) {
            sws_ctx = sws_getContext(
                frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
                width, height, AV_PIX_FMT_BGR24,
                SWS_BILINEAR, nullptr, nullptr, nullptr
            );
            if (!sws_ctx) {
                throw std::runtime_error("Could not initialize the conversion context");
            }
        }

        // YUV420P에서 BGR로 변환하여 OpenCV Mat에 저장
        decoded_frame = cv::Mat(height, width, CV_8UC3);
        uint8_t* dest_data[1] = { decoded_frame.data };
        int dest_linesize[1] = { static_cast<int>(decoded_frame.step[0]) };
        sws_scale(
            sws_ctx,
            frame->data, frame->linesize,
            0, frame->height,
            dest_data, dest_linesize
        );

        // 한 프레임만 처리하기 위해 break
        break;
    }

    // 패킷 초기화
    av_packet_unref(pkt);

    if (!decoded_frame.empty()) {
        // cv::Mat을 numpy 배열로 변환하여 반환
        // 메모리 관리가 필요하므로 capsule을 사용
        cv::Mat* mat = new cv::Mat(decoded_frame);
        py::capsule free_when_done(mat, [](void* f) {
            delete reinterpret_cast<cv::Mat*>(f);
        });

        return py::array_t<uint8_t>(
            { mat->rows, mat->cols, mat->channels() },
            { mat->step[0], mat->step[1], sizeof(uint8_t) },
            mat->data,
            free_when_done
        );
    } else {
        // 빈 배열 반환
        return py::array_t<uint8_t>();
    }
}

void Decoder::reset() {
    avcodec_flush_buffers(codec_ctx);
}

void Decoder::close() {
    if (codec_ctx) {
        avcodec_free_context(&codec_ctx);
        codec_ctx = nullptr;
    }
    if (frame) {
        av_frame_free(&frame);
        frame = nullptr;
    }
    if (pkt) {
        av_packet_free(&pkt);
        pkt = nullptr;
    }
    if (sws_ctx) {
        sws_freeContext(sws_ctx);
        sws_ctx = nullptr;
    }
}

Decoder::~Decoder() {
    close();
}
