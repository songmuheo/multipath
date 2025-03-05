// server/config.h

#ifndef CONFIG_H
#define CONFIG_H

#include <string>

const char* SERVER_IP = "121.128.220.205";

const int SERVER_PORT1 = 12345; //lg
const int SERVER_PORT2 = 12346; //kt

const int BUFFER_SIZE = 65536;

const std::string BASE_FILEPATH = "/home/songmu/";

// Path to save bin files(received packet)
const std::string FILEPATH_LOG = BASE_FILEPATH + "multipath/server/logs/";

const uint64_t PLAY_DELAY_MS = 50;

// TURN 서버 설정
// config.h (일부)
#define TURN_SERVER_IP      "121.128.220.205"
#define TURN_SERVER_PORT    3478
#define TURN_SECRET         "v2n2v123"   // static-auth-secret
#define TURN_REALM          "v2n2v"
#define TURN_IDENTIFIER     "client"     // 클라이언트 식별자 (원하는 값)
#define TURN_VALID_SECONDS  3600         // 유효기간 (초)

// 클라이언트 TURN 릴레이 주소 (TURN 할당 후 클라이언트가 ACK 수신용으로 사용할 주소)
// 실제 운영에서는 TURN 할당 시 동적으로 얻어야 함
#define CLIENT_TURN_IP "121.128.220.205"  // 예시 값, TURN 할당 시 결정됨
#define CLIENT_TURN_PORT 6000                     //

#endif // CONFIG_H