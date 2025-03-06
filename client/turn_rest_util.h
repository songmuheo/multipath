#ifndef TURN_REST_UTIL_H
#define TURN_REST_UTIL_H

#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <openssl/hmac.h>
#include <openssl/evp.h>

// ========== Base64 인코딩 ==========
static std::string base64_encode(const std::string &input)
{
    static const char* base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int val=0, valb=-6;
    for (unsigned char c : input) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(base64_chars[(val>>valb)&0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) {
        out.push_back(base64_chars[((val << 8)>>(valb+8))&0x3F]);
    }
    while (out.size()%4) {
        out.push_back('=');
    }
    return out;
}

// ========== raw HMAC-SHA1 (binary) ==========
static std::string hmac_sha1_raw(const std::string &data, const std::string &key)
{
    unsigned char hmac_res[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    HMAC(EVP_sha1(),
         key.data(), (int)key.size(),
         reinterpret_cast<const unsigned char*>(data.data()), data.size(),
         hmac_res, &len);
    return std::string(reinterpret_cast<char*>(hmac_res), len);
}

// ========== username: "<expire_epoch>:<identifier>" ==========
static std::string generate_ephemeral_username(const std::string &identifier, uint32_t valid_secs)
{
    uint64_t expiry = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count() + valid_secs;
    // "만료시각:식별자"
    return std::to_string(expiry) + ":" + identifier;
}

// ========== password = base64( HMAC-SHA1("<username>:<realm>", <secret>) ) ==========
static std::string generate_ephemeral_password(const std::string &username,
                                               const std::string &realm,
                                               const std::string &secret)
{
    std::string key = username + ":" + realm; 
    std::string raw = hmac_sha1_raw(key, secret);
    return base64_encode(raw);
}

#endif // TURN_REST_UTIL_H
