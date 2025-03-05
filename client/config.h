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
// common_turn_utils.h
#pragma once

#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <openssl/hmac.h>
#include <openssl/evp.h>

// TURN 서버 정보
// (실환경에 맞게 수정: TURN_SERVER_IP, TURN_SERVER_PORT, TURN_REALM, TURN_SECRET 등)
static const char* TURN_SERVER_IP   = "121.128.220.205"; // coturn IP
static const int   TURN_SERVER_PORT = 3478;
static const char* TURN_REALM       = "v2n2v";     // turnserver.conf의 realm
static const char* TURN_SECRET      = "v2n2v123";  // turnserver.conf의 static-auth-secret

// TURN username 생성 (예: "만료시각:식별자")
inline std::string generate_turn_username(const std::string& identifier, uint32_t validSeconds)
{
    uint64_t expiration = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::system_clock::now().time_since_epoch()
                          ).count() + validSeconds;
    return std::to_string(expiration) + ":" + identifier;
}

// HMAC-SHA1(password) 계산
inline std::string compute_turn_password(const std::string& data, const std::string& secret)
{
    unsigned char hmac_result[EVP_MAX_MD_SIZE] = {0};
    unsigned int hmac_length = 0;
    HMAC(EVP_sha1(),
         secret.c_str(), secret.size(),
         reinterpret_cast<const unsigned char*>(data.c_str()), data.size(),
         hmac_result, &hmac_length);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < hmac_length; i++)
        oss << std::setw(2) << (int)hmac_result[i];
    return oss.str();
}



// 클라이언트 TURN 릴레이 주소 (TURN 할당 후 클라이언트가 ACK 수신용으로 사용할 주소)
// 실제 운영에서는 TURN 할당 시 동적으로 얻어야 함
#define CLIENT_TURN_IP "121.128.220.205"  // 예시 값, TURN 할당 시 결정됨
#define CLIENT_TURN_PORT 6000                     //

const int HEIGHT = 480;
const int WIDTH = 640;

#endif // CONFIG_H