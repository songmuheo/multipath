# server/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(Server)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET avformat avcodec avutil swscale)

add_executable(server server.cpp)
target_link_libraries(server ${OpenCV_LIBS} PkgConfig::FFMPEG)
