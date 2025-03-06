// client/config.h

#include <string>

#ifndef CONFIG_H
#define CONFIG_H

const char* SERVER_IP = "121.128.220.205";
const int SERVER_PORT = 12345;
const int SERVER_REG_PORT = 12345;


const int PACKET_SIZE = 1500;
const float PACKET_INTERVAL = 0.033;
const int FPS = 30;
const int BITRATE = 2000000;

// const std::string BASE_FILEPATH = "/home/widen/";
const std::string BASE_FILEPATH = "/home/songmu/";

// File path to save png files
const std::string SAVE_FILEPATH = BASE_FILEPATH + "multipath/client/logs/";

// LGU+
const char* INTERFACE1_IP = "192.168.10.101";
const char* INTERFACE1_NAME = "enx588694fda665";
// KT
const char* INTERFACE2_IP = "192.168.1.16";
const char* INTERFACE2_NAME = "enx588694f64878";

// TURN 서버 설정
// TURN 할당 시 사용할 로컬(모바일) 인터페이스 IP
const char* MOBILE_INTERFACE_IP = "192.168.10.101";

// TURN 서버 설정 (coturn 서버의 설정과 일치해야 함)
#define TURN_SERVER_IP "121.128.220.205"
#define TURN_SERVER_PORT 3478
#define TURN_IDENTIFIER "client_id"         // 클라이언트를 식별할 문자열
#define TURN_VALID_SECONDS 600              // 유효 시간 (초)
#define TURN_REALM "v2n2v"
#define TURN_SECRET "v2n2v123"

const int HEIGHT = 480;
const int WIDTH = 640;

#endif // CONFIG_H