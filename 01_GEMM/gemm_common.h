#ifndef GEMM_COMMON_H
#define GEMM_COMMON_H

#include "../utils/inc/cuda_common.h"


void gemm_cpu(float* A, float* B, float* C, const int m, const int n, const int k);

__global__ void gemm_kernel_comm(float *A, float *B, float *C, int M, int N, int K);
template <int BLOCK_SIZE> __global__ void gemm_kernel_smem(float *A, float *B, float *C, int wA, int wB);
int gemm_gpu(float *A, float *B, float *C, int M, int N, int K);

#endif // GEMM_COMMON_H