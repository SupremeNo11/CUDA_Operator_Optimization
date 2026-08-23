cmake_minimum_required(VERSION 3.10)

project(CUDA_Operator_Optimization
    VERSION 1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "CUDA Operator Optimization"
)

# GPU architecture: GTX 1050 Ti (Pascal, SM 6.1)
# Change this when targeting a different GPU
set(CMAKE_CUDA_ARCHITECTURES 61)

# C++ standard: gnu++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)

# Auto-discover operator directories (NN_Name pattern) that have CMakeLists.txt
file(GLOB OPERATOR_DIRS RELATIVE ${CMAKE_SOURCE_DIR} "[0-9][0-9]_*")

foreach(DIR ${OPERATOR_DIRS})
    if(EXISTS "${CMAKE_SOURCE_DIR}/${DIR}/CMakeLists.txt")
        add_subdirectory(${DIR})
    endif()
endforeach()
