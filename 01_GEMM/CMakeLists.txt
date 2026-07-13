cmake_minimum_required(VERSION 3.10)

project(GEMM_Optimization
    VERSION 1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "GEMM Optimization"
)

set(CMAKE_CUDA_ARCHITECTURES 61)

set(SOURCES gemm_optimization.cu gemm_cpu.cpp ../utils/src/cuda_common.cu)

add_executable(gemm_optimization ${SOURCES})
