cmake_minimum_required(VERSION 3.5)
project(server)

# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV 라이브러리 설정
find_package(OpenCV REQUIRED)

# Threads 라이브러리 설정
find_package(Threads REQUIRED)

# FFmpeg 라이브러리 설정
find_package(PkgConfig REQUIRED)

# FFmpeg 라이브러리 경로 및 설정을 가져옴
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)

# 실행 파일 설정
add_executable(server server.cpp)
add_executable(play server_play.cpp)
# add_executable(play2 server_play2.cpp)

# 라이브러리 포함 디렉토리 설정
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
)

# 라이브러리 링크 디렉토리 설정
link_directories(
    ${FFMPEG_LIBRARY_DIRS}
)

# FFmpeg 및 기타 필요한 라이브러리를 링크
target_link_libraries(server
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}
    avcodec
    avformat
    avutil
    swscale
    ${CMAKE_THREAD_LIBS_INIT}
)

# FFmpeg 및 기타 필요한 라이브러리를 링크 (play)
target_link_libraries(play
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

# FFmpeg 및 기타 필요한 라이브러리를 링크 (play2)
# target_link_libraries(play2
#     ${OpenCV_LIBS}
#     ${FFMPEG_LIBRARIES}
#     ${CMAKE_THREAD_LIBS_INIT}
# )