#ifndef GEMM_COMMON_H
#define GEMM_COMMON_H

#include "../utils/inc/cuda_common.h"


void gemm_cpu(float* A, float* B, float* C, const int m, const int n, const int k);

__global__ void gemm_kernel_01(float *A, float *B, float *C, int M, int N, int K);
int gemm_gpu(float *A, float *B, float *C, int M, int N, int K);

#endif // GEMM_COMMON_H