cmake_minimum_required(VERSION 3.8)
project(roipool)

set(CMAKE_CXX_STANDARD 11)

# You should change it to your build pytorch lib
set(ATEN_DIR "/your_pytorch_path/pytorch/build/lib.linux-x86_64-3.6/torch/lib")
include_directories(${ATEN_DIR}/include)

set(SOURCE_FILES main.cpp roi_pool_cpu.cpp temp.h)

add_executable(roipool ${SOURCE_FILES})

target_link_libraries(roipool ${ATEN_DIR}/libATen.so)
