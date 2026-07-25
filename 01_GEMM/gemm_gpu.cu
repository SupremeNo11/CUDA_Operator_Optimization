#include "gemm_common.h"

/**
 * GEMM Kernel
 */
__global__ void gemm_kernel_comm(float *A, float *B, float *C, int M, int N, int K)
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

/**
 * 静态分配共享内存优化GEMM，采用模板方法，最大程度优化kernel
 */
template <int BLOCK_SIZE> __global__ void gemm_kernel_smem(float *A, float *B, float *C, int wA, int wB)
{
    // wA：Matrix A width
    // wB：Matrix B width

    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];
    
    // Block Index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread Index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // A矩阵子矩阵的索引起始位置，深度绑定Block的行索引
    int aBegin = wA * BLOCK_SIZE * by;
    
    // A矩阵子矩阵的索引结束位置
    int aEnd = aBegin + wA - 1;

    // A矩阵子矩阵访问步幅（间隔）应当为Block的高
    int aStep = BLOCK_SIZE;

    // B矩阵子矩阵的起始索引
    int bBegin = BLOCK_SIZE * bx;

    // B矩阵子矩阵的迭代步幅
    int bStep = BLOCK_SIZE * wB;

    // 每个线程都会提前申请一个变量用于存储电积运算结果
    float Csub = 0;

    // 开始遍历A和B矩阵，分好几轮计算
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        // 全局内存到共享内存
        s_A[ty][tx] = A[wA * ty + tx + a];
        s_B[ty][tx] = B[wB * ty + tx + b];

        // 线程间同步屏障
        __syncthreads();

#pragma unroll      // 取消循环控制，没有循环控制开销，for循环直接展开

        // 展开计算
        for (int k = 0; k < BLOCK_SIZE; k++) {
            Csub += s_A[ty][k] * s_B[k][tx];
        }

        // 线程间同步屏障
        __syncthreads();
    }

    // 将计算结果写回设备内存(全局内存)
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;   // Bolck初始位置，横向跨度为BLOCK_SIZE，纵向跨度为wB * BLOCK_SIZE
    int cIdx = c + wB * ty + tx;
    
    C[cIdx] = Csub;
}

int gemm_gpu(float *A, float *B, float *C, int M, int N, int K)
{
    // Using CUDA events to measure the execution time of the kernel
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    // 0. Create and Malloc device memmory
    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy data to device form host
    CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    // 向上整除，<cuda/cmath> 提供向上整除的方法，CUDA docs 13.3
    // dim3 grid(cuda::ceil_div(N, block.x), cuda::ceil_div(M, block.y));

    /**
     * gemm_kernel_comm
     */
    CUDA_CHECK(cudaEventRecord(start));
    gemm_kernel_comm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    // Copy data to host from device
    CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    std::cout << "gemm_kernel_comm：GEMM kernel execution time: [" << static_cast<long long>(milliseconds * 1000) << "] us" << std::endl;

    // Completed
    // Check the result of the GEMM operation
    std::cout << "Result matrix C (first 3 elements):" << std::endl;
    for (int i = 0; i < std::min(M, 3); i++) {
        for (int j = 0; j < std::min(N, 3); j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    /**
     * gemm_kernel_smem
     */
    CUDA_CHECK(cudaEventRecord(start));
    gemm_kernel_smem<16><<<grid, block>>>(d_A, d_B, d_C, K, N);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    std::cout << "gemm_kernel_smem: kernel execution time: [" << static_cast<long long>(milliseconds * 1000) << "] us" << std::endl;

    // Compare results
    float* smem_C = new float[M * N];
    CUDA_CHECK(cudaMemcpy(smem_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fabs(C[i * M + j] - smem_C[i * M + j]) > 0.0001)
            {
                std::cout << "Result Error!" << std::endl;
                break;
            }
        }
    }

    // Free device memmory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free Host memmory
    delete[]    smem_C;

    return SYS_STATUS_SUCCESS;
}

