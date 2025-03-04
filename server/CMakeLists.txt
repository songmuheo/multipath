cmake_minimum_required(VERSION 3.5)
project(server)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV 라이브러리 설정
find_package(OpenCV REQUIRED)

# Threads 라이브러리 설정
find_package(Threads REQUIRED)

# FFmpeg 라이브러리 설정
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libavdevice libavfilter)

# CUDA 설정
find_package(CUDA REQUIRED)

# CUDA 라이브러리 경로 수동 설정
# set(CUDA_LIBRARY_DIRS "/usr/local/cuda/lib64")

message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Library Path: ${CUDA_LIBRARY_DIRS}")
message(STATUS "CUDA Include Path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})


# 실행 파일 설정
add_executable(server server.cpp)
# add_executable(server2 server2.cpp)


# 라이브러리 포함 디렉토리 설정
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# receive에 대한 FFmpeg 및 pthread 라이브러리 링크 추가
target_link_libraries(server
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}  # FFmpeg 관련 라이브러리 일괄 링크
    ${CUDA_LIBRARIES}
    Threads::Threads     # pthread 라이브러리 링크 추가
)

# receive에 대한 FFmpeg 및 pthread 라이브러리 링크 추가
# target_link_libraries(server2
#     ${OpenCV_LIBS}
#     ${FFMPEG_LIBRARIES}  # FFmpeg 관련 라이브러리 일괄 링크
#     ${CUDA_LIBRARIES}
#     Threads::Threads     # pthread 라이브러리 링크 추가
# )
