cmake_minimum_required(VERSION 3.5)
project(server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV 설정
find_package(OpenCV REQUIRED)

# Threads 설정
find_package(Threads REQUIRED)

# FFmpeg 설정 (pkg-config 사용)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libavdevice libavfilter)

# CUDA 설정
find_package(CUDA REQUIRED)
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Library Path: ${CUDA_LIBRARY_DIRS}")
message(STATUS "CUDA Include Path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# PJSIP 직접 경로 설정 (수동 추가)
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

# GLib (libnice에 필요)
pkg_check_modules(GLIB REQUIRED glib-2.0 gobject-2.0)
include_directories(${GLIB_INCLUDE_DIRS})
link_directories(${GLIB_LIBRARY_DIRS})

# libnice 설정
pkg_check_modules(NICE REQUIRED nice)
include_directories(/usr/include/nice)  # 필요 시 /usr/local/include/nice 로 변경
link_directories(/usr/lib/x86_64-linux-gnu)  # 필요 시 /usr/local/lib 로 변경


# 실행 파일 설정
# add_executable(server server.cpp)
add_executable(server_feedback server_feedback.cpp)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${NICE_INCLUDE_DIRS}
)

# server 실행 파일 연결 라이브러리
# target_link_libraries(server
#     ${OpenCV_LIBS}
#     ${FFMPEG_LIBRARIES}
#     ${CUDA_LIBRARIES}
#     Threads::Threads
# )

# server_feedback 실행 파일에 필요한 라이브러리 연결 (PJSIP, libnice, OpenSSL 등)
target_link_libraries(server_feedback
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${PJSIP_LIBS}
    ${GLIB_LIBRARIES}
    ${NICE_LIBRARIES}
    -lnice  # libnice 직접 링크
    -lgio-2.0 -lgobject-2.0 -lglib-2.0  # 필요할 수 있음
    -lssl -lcrypto
    Threads::Threads
)
