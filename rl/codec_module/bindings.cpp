// codec_module.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "encoder.h"
#include "decoder.h"

namespace py = pybind11;

PYBIND11_MODULE(codec_module, m) {
    py::class_<Encoder>(m, "Encoder")
        .def(py::init<int, int>())
        .def("encode_frame", &Encoder::encode_frame)
        .def("reset", &Encoder::reset)
        .def("close", &Encoder::close);

    py::class_<Decoder>(m, "Decoder")
        .def(py::init<>())
        .def("decode_frame", &Decoder::decode_frame)
        .def("reset", &Decoder::reset)
        .def("close", &Decoder::close);
}
