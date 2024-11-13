// encoder.h

#ifndef ENCODER_H
#define ENCODER_H

#include <string>
#include <pybind11/pybind11.h>
namespace py = pybind11;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

class Encoder {
public:
    Encoder(int width, int height);
    ~Encoder();

    py::bytes encode_frame(const std::string& frame_path, bool is_i_frame);
    void reset();
    void close();

private:
    void init_encoder();

    int width;
    int height;
    int frame_index;

    AVCodecContext* codec_ctx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* pkt = nullptr;
    SwsContext* sws_ctx = nullptr;
};

#endif // ENCODER_H
