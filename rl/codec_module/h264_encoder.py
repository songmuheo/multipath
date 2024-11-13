# codec/h264_encoder.py
import ctypes
import os

class H264Encoder:
    def __init__(self):
        lib_path = os.path.join(os.path.dirname(__file__), 'libh264_encoder.so')
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.obj = self.lib.H264Encoder_new()

    def reset(self):
        self.lib.H264Encoder_reset(self.obj)

    def encode(self, frame_path, frame_type):
        frame_path_c = ctypes.c_char_p(frame_path.encode('utf-8'))
        frame_type_c = ctypes.c_char_p(frame_type.encode('utf-8'))
        self.lib.H264Encoder_encode.restype = ctypes.c_char_p
        output_path = self.lib.H264Encoder_encode(self.obj, frame_path_c, frame_type_c)
        return output_path.decode('utf-8')

    def get_encoded_size(self, frame_path, frame_type, temp):
        frame_path_c = ctypes.c_char_p(frame_path.encode('utf-8'))
        frame_type_c = ctypes.c_char_p(frame_type.encode('utf-8'))
        temp_c = ctypes.c_bool(temp)
        size = self.lib.H264Encoder_get_encoded_size(self.obj, frame_path_c, frame_type_c, temp_c)
        return size
