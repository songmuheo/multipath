// decoder.h

#ifndef DECODER_H
#define DECODER_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // numpy 지원을 위한 헤더 포함
#include <opencv2/opencv.hpp>
namespace py = pybind11;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

class Decoder {
public:
    Decoder();
    ~Decoder();

    py::array_t<uint8_t> decode_frame(const py::bytes& encoded_data, int size, int width, int height);
    void reset();
    void close();

private:
    void init_decoder();

    AVCodecContext* codec_ctx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* pkt = nullptr;
    SwsContext* sws_ctx = nullptr;
};

#endif // DECODER_H
