#!/bin/bash

# 빌드 디렉토리 생성
mkdir -p build
cd build

# CMake를 이용한 빌드
cmake ..
make

# 실행
./MakeFramesProcessing