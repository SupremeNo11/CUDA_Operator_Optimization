cmake_minimum_required(VERSION 3.10)

project(GEMM_Optimization
    VERSION 1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "GEMM Optimization"
)

set(CMAKE_CUDA_ARCHITECTURES 61)

file(GLOB UTILS_SRC
    ../utils/src/*.cu
    ../utils/src/*.cpp
)

set(SOURCES gemm_optimization.cu gemm_cpu.cpp gemm_gpu.cu ${UTILS_SRC})

add_executable(gemm_optimization ${SOURCES})
