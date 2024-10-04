# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(StreamProcessing)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(SWSCALE REQUIRED libswscale)
pkg_check_modules(OPENCV REQUIRED opencv4)

include_directories(
    ${AVCODEC_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${SWSCALE_INCLUDE_DIRS}
    ${OPENCV_INCLUDE_DIRS}
)

add_executable(MakeFramesProcessing make_frames_with_decoding_latency.cpp)

target_link_libraries(MakeFramesProcessing
    ${AVCODEC_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    ${OPENCV_LIBRARIES}
)

