#include "gemm_common.h"

/**
 * GEMM Kernel
 */
__global__ void gemm_kernel_01(float *A, float *B, float *C, int M, int N, int K)
{
    /**
     * A = M × K
     * B = K × N
     * C = M * N
     */
    // Single thread calculate one element of C
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // Check boundary
    if (row < M && col < N)
    {
        // Temp var
        float sum = 0.0f;

        // calculate dot product
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }

        // Store result
        C[row * N + col] = sum;
    }
}

int gemm_gpu(float *A, float *B, float *C, int M, int N, int K)
{
    // Using CUDA events to measure the execution time of the kernel
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(start));

    // 0. Create and Malloc device memmory
    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 1. Copy data to device form host
    CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 2. Calculate
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_kernel_01<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    // 3. Copy data to host from device
    CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 4. Free device memmory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    std::cout << "GEMM kernel execution time: " << static_cast<long long>(milliseconds * 1000) << " us" << std::endl;

    // 5. Completed
    // Check the result of the GEMM operation
    std::cout << "Result matrix C (first 3 elements):" << std::endl;
    for (int i = 0; i < std::min(M, 3); i++) {
        for (int j = 0; j < std::min(N, 3); j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return SYS_STATUS_SUCCESS;
}

