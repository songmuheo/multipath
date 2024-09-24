// client/config.h

#include <string>

#ifndef CONFIG_H
#define CONFIG_H

const char* SERVER_IP = "203.229.155.232";
const int SERVER_PORT = 12345;

const int PACKET_SIZE = 1500;
const float PACKET_INTERVAL = 0.033;
const int FPS = 30;
const int BITRATE = 1500000;

// const std::string BASE_FILEPATH = "/home/widen/"
const std::string BASE_FILEPATH = "/home/songmu/";

// File path to save png files
const std::string SAVE_FILEPATH = BASE_FILEPATH + "Multipath/cpp/results/client/";

const char* INTERFACE1_IP = "10.16.130.87";
const char* INTERFACE1_NAME = "wlp1s0";
const char* INTERFACE2_IP = "192.168.0.24";
const char* INTERFACE2_NAME = "wlx588694fd23d6";

// // LGU+
// const char* INTERFACE1_IP = "192.168.10.100";
// const char* INTERFACE1_NAME = "enx588694f65060";
// // KT
// const char* INTERFACE2_IP = "192.168.1.17";
// const char* INTERFACE2_NAME = "enx588694f747d7";

const int HEIGHT = 480;
const int WIDTH = 640;

#endif // CONFIG_H