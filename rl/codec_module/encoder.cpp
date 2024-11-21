// encoder.cpp

#include "encoder.h"
#include <stdexcept>
#include <opencv2/opencv.hpp>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

Encoder::Encoder(int w, int h) : width(w), height(h), frame_index(0) {
    init_encoder();
}

void Encoder::init_encoder() {
    // avcodec_register_all(); // FFmpeg 초기화 (필요에 따라 추가)

    const AVCodec* codec = avcodec_find_encoder_by_name("libx264");
    if (!codec) throw std::runtime_error("Codec not found");

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) throw std::runtime_error("Could not allocate video codec context");

    /*
    바꾼 인코더 셋팅
    */
    codec_ctx->width = width;
    codec_ctx->height = height;
    codec_ctx->time_base = {1, 30};
    codec_ctx->framerate = {30, 1};
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx->max_b_frames = 0;
    codec_ctx->thread_type = FF_THREAD_SLICE;
    codec_ctx->thread_count = 1;
    // codec_ctx->gop_size = 12; // I-프레임 간격 설정

    AVDictionary* opt = nullptr;
    
    // Preset and tune settings for low latency
    av_dict_set(&opt, "preset", "veryfast", 0);    // Use "superfast" if you want better compression
    av_dict_set(&opt, "tune", "zerolatency", 0);

    // Disable B-frames explicitly
    av_dict_set(&opt, "bframes", "0", 0);

    // Rate control settings
    av_dict_set(&opt, "crf", "21", 0);                 // Adjust CRF value as needed (lower = better quality)
    av_dict_set(&opt, "rc-lookahead", "0", 0);         // Disable look-ahead
    av_dict_set(&opt, "scenecut", "0", 0);             // Disable scene cut detection

    // Keyframe interval settings
    av_dict_set(&opt, "keyint", "30", 0);              // Set to frame rate
    av_dict_set(&opt, "min-keyint", "30", 0);          // Force constant keyframe interval

    // Additional settings for latency and quality
    av_dict_set(&opt, "refs", "1", 0);                 // Use 3 reference frame
    av_dict_set(&opt, "no-sliced-threads", "1", 0);    // Disable sliced threads for better latency
    av_dict_set(&opt, "aq-mode", "1", 0);              // Disable adaptive quantization
    av_dict_set(&opt, "trellis", "0", 0);              // Disable trellis optimization

    av_dict_set(&opt, "psy-rd", "1.0", 0);
    av_dict_set(&opt, "psy-rdoq", "1.0", 0);

    if (avcodec_open2(codec_ctx, codec, &opt) < 0) {
        av_dict_free(&opt);
        throw std::runtime_error("Could not open codec");
    }
    av_dict_free(&opt);

    frame = av_frame_alloc();
    if (!frame) throw std::runtime_error("Could not allocate video frame");

    frame->format = codec_ctx->pix_fmt;
    frame->width = codec_ctx->width;
    frame->height = codec_ctx->height;

    if (av_frame_get_buffer(frame, 32) < 0) {
        throw std::runtime_error("Could not allocate frame data");
    }

    pkt = av_packet_alloc();
    if (!pkt) throw std::runtime_error("Could not allocate AVPacket");

    sws_ctx = sws_getContext(
        width, height, AV_PIX_FMT_BGR24,
        width, height, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    if (!sws_ctx) throw std::runtime_error("Could not initialize the conversion context");
}

py::bytes Encoder::encode_frame(const std::string& frame_path, bool is_i_frame) {
    cv::Mat img = cv::imread(frame_path);
    if (img.empty()) throw std::runtime_error("Could not load image");

    cv::resize(img, img, cv::Size(width, height));

    // BGR24에서 YUV420P로 변환
    uint8_t* in_data[1] = { img.data };
    int in_linesize[1] = { static_cast<int>(img.step[0]) };
    int height_scaled = sws_scale(
        sws_ctx, in_data, in_linesize,
        0, height, frame->data, frame->linesize
    );
    if (height_scaled <= 0) {
        throw std::runtime_error("sws_scale failed");
    }

    frame->pts = frame_index++;
    frame->pict_type = is_i_frame ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;

    // 프레임을 인코더로 전송
    int ret = avcodec_send_frame(codec_ctx, frame);
    if (ret < 0) {
        throw std::runtime_error("Error sending frame for encoding");
    }

    // 인코딩된 패킷 수신 및 처리
    std::string encoded_data;
    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break; // 더 이상 패킷이 없음
        } else if (ret < 0) {
            throw std::runtime_error("Error during encoding");
        }

        // 인코딩된 데이터 추가
        encoded_data.append(reinterpret_cast<char*>(pkt->data), pkt->size);
        av_packet_unref(pkt);
    }

    return py::bytes(encoded_data);
}

void Encoder::reset() {
    frame_index = 0;
    avcodec_flush_buffers(codec_ctx);
}

void Encoder::close() {
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

Encoder::~Encoder() {
    close();
}
