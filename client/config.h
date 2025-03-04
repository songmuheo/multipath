// client/config.h

#include <string>

#ifndef CONFIG_H
#define CONFIG_H

const char* SERVER_IP = "121.128.220.205";
const int SERVER_PORT = 12345;

const int PACKET_SIZE = 1500;
const float PACKET_INTERVAL = 0.033;
const int FPS = 30;
const int BITRATE = 2000000;

// const std::string BASE_FILEPATH = "/home/widen/";
const std::string BASE_FILEPATH = "/home/songmu/";

// File path to save png files
const std::string SAVE_FILEPATH = BASE_FILEPATH + "multipath/cpp/client/logs";

// const char* INTERFACE1_IP = "10.121.134.99";
// const char* INTERFACE1_IP = "192.168.0.19";
// const char* INTERFACE1_NAME = "wlp1s0";
// const char* INTERFACE2_IP = "192.168.0.24";
// const char* INTERFACE2_NAME = "wlx588694fd23d6";

// LGU+
const char* INTERFACE1_IP = "192.168.10.101";
const char* INTERFACE1_NAME = "enx588694fda665";
// KT
const char* INTERFACE2_IP = "192.168.1.16";
const char* INTERFACE2_NAME = "enx588694f64878";

const int HEIGHT = 480;
const int WIDTH = 640;

#endif // CONFIG_H