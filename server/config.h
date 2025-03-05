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
#define TURN_SERVER_IP "121.128.220.205"    // coturn 서버 IP (환경에 맞게 수정)
#define TURN_SERVER_PORT 3478                     // coturn 기본 포트
#define TURN_USERNAME "user"                  // coturn 사용자 이름
#define TURN_PASSWORD "v2n2v123"                  // coturn 비밀번호

// 클라이언트 TURN 릴레이 주소 (TURN 할당 후 클라이언트가 ACK 수신용으로 사용할 주소)
// 실제 운영에서는 TURN 할당 시 동적으로 얻어야 함
#define CLIENT_TURN_IP "121.128.220.205"  // 예시 값, TURN 할당 시 결정됨
#define CLIENT_TURN_PORT 6000                     //


#endif // CONFIG_H