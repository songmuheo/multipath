cmake_minimum_required(VERSION 3.5)
project(server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV 설정
find_package(OpenCV REQUIRED)

# Threads 설정
find_package(Threads REQUIRED)

# FFmpeg 설정
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libavdevice libavfilter)

# CUDA 설정
find_package(CUDA REQUIRED)

message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Library Path: ${CUDA_LIBRARY_DIRS}")
message(STATUS "CUDA Include Path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# PJSIP 직접 경로 설정 (pkg-config 사용하지 않고 수동 추가)
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

# 실행 파일 설정
add_executable(server server.cpp)
add_executable(server_feedback server_feedback.cpp)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# 서버 실행 파일에 필요한 라이브러리 추가
target_link_libraries(server
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}
    ${CUDA_LIBRARIES}
    Threads::Threads
)

# 서버 피드백 실행 파일에 PJSIP 라이브러리 추가
target_link_libraries(server_feedback
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${PJSIP_LIBS}
    Threads::Threads
)
