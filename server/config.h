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


#endif // CONFIG_H