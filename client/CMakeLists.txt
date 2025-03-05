cmake_minimum_required(VERSION 3.10)
project(Client)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# realsense2 및 OpenCV 라이브러리 설정
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)

# FFmpeg 라이브러리 설정 (직접 경로로 설정)
find_library(AVFORMAT avformat REQUIRED)
find_library(AVCODEC avcodec REQUIRED)
find_library(AVUTIL avutil REQUIRED)
find_library(SWSCALE swscale REQUIRED)

# PJSIP 관련 라이브러리 수동 경로 설정
set(PJSIP_INCLUDE_DIR "/usr/local/include")
set(PJSIP_LIB_DIR "/usr/local/lib")

include_directories(${PJSIP_INCLUDE_DIR})
link_directories(${PJSIP_LIB_DIR})

set(PJSIP_LIBS
    pj
    pjnath
    pjlib-util
    pjsip
    pjsip-ua
    pjsip-simple
    pjsua
)

# 실행 파일 추가
add_executable(client client.cpp)
target_link_libraries(client
    ${realsense2_LIBRARY}
    ${OpenCV_LIBS}
    ${AVFORMAT} ${AVCODEC} ${AVUTIL} ${SWSCALE}
    pthread
)

add_executable(client_feedback client_feedback.cpp)
target_link_libraries(client_feedback
    ${realsense2_LIBRARY}
    ${OpenCV_LIBS}
    ${AVFORMAT} ${AVCODEC} ${AVUTIL} ${SWSCALE}
    ${PJSIP_LIBS}
    OpenSSL::SSL OPenSSL::Crypto
    pthread
)
