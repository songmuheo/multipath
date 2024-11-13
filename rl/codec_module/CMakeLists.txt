# codec_module/CMakeLists.txt

cmake_minimum_required(VERSION 3.4...3.18)
project(codec_module)

set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/codec_module/build/codec_module)


find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavutil libavformat)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${FFMPEG_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

pybind11_add_module(codec_module bindings.cpp encoder.cpp decoder.cpp)

target_link_libraries(codec_module PRIVATE ${FFMPEG_LIBRARIES} ${OpenCV_LIBS} swscale)
