// server/config.h

#ifndef CONFIG_H
#define CONFIG_H

#include <string>

const char* SERVER_IP = "121.128.220.205";

const int SERVER_PORT1 = 12345; //lg
const int SERVER_PORT2 = 12346; //kt
const int SERVER_REG_PORT = 12345; // TURN 등록 메시지 수신용 (SERVER_PORT1와 동일)

const int BUFFER_SIZE = 65536;

const std::string BASE_FILEPATH = "/home/songmu/";

// Path to save bin files(received packet)
const std::string FILEPATH_LOG = BASE_FILEPATH + "multipath/server/logs/";

const uint64_t PLAY_DELAY_MS = 50;

// TURN 서버 설정
// TURN 관련 설정 (coturn 서버와 일치시켜야 함)
#define TURN_SERVER_IP "121.128.220.205"
#define TURN_SERVER_PORT 3478
#define TURN_IDENTIFIER "client_id"          // 클라이언트를 식별할 문자열
#define TURN_VALID_SECONDS 3600                 // 유효 시간 (초)
#define TURN_REALM "v2n2v"
#define TURN_SECRET "v2n2v123"

#endif // CONFIG_H