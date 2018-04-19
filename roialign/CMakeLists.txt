cmake_minimum_required(VERSION 3.8)
project(roialign)

set(CMAKE_CXX_STANDARD 11)

# change to your pytorch path
set(ATEN_DIR "/your_pytorch_path/pytorch/build/lib.linux-x86_64-3.6/torch/lib")
include_directories(${ATEN_DIR}/include)

set(SOURCE_FILES main.cpp roi_align_cpu.cpp temp.h)

add_executable(roialign ${SOURCE_FILES})

target_link_libraries(roialign ${ATEN_DIR}/libATen.so)
