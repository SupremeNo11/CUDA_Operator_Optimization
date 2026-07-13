#ifndef GEMM_COMMON_H
#define GEMM_COMMON_H

#include "../utils/inc/cuda_common.h"


void gemm_cpu(std::vector<std::vector<float>>& A, 
            std::vector<std::vector<float>>& B, 
            std::vector<std::vector<float>>& C, 
            const int m, 
            const int n, 
            const int k);


#endif // GEMM_COMMON_H