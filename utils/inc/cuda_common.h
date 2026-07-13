#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>


void initialize_matrix(std::vector<std::vector<float>>& matrix);


#endif // CUDA_COMMON_H