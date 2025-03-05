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
const std::string SAVE_FILEPATH = BASE_FILEPATH + "multipath/client/logs/";

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

// TURN 서버 설정
#define TURN_SERVER_IP "121.128.220.205"    // coturn 서버 IP (환경에 맞게 수정)
#define TURN_SERVER_PORT 3478                     // coturn 기본 포트
#define TURN_USERNAME "user"                  // coturn 사용자 이름
#define TURN_PASSWORD "v2n2v123"                  // coturn 비밀번호


// 클라이언트 TURN 릴레이 주소 (TURN 할당 후 클라이언트가 ACK 수신용으로 사용할 주소)
// 실제 운영에서는 TURN 할당 시 동적으로 얻어야 함
#define CLIENT_TURN_IP "121.128.220.205"  // 예시 값, TURN 할당 시 결정됨
#define CLIENT_TURN_PORT 6000                     //

const int HEIGHT = 480;
const int WIDTH = 640;

#endif // CONFIG_H