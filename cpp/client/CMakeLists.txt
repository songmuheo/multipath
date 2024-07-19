# client/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(Client)

set(CMAKE_CXX_STANDARD 11)

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

find_library(AVFORMAT avformat REQUIRED)
find_library(AVCODEC avcodec REQUIRED)
find_library(AVUTIL avutil REQUIRED)
find_library(SWSCALE swscale REQUIRED)

add_executable(client client.cpp)
target_link_libraries(client ${realsense2_LIBRARY} ${OpenCV_LIBS} ${AVFORMAT} ${AVCODEC} ${AVUTIL} ${SWSCALE} pthread)
