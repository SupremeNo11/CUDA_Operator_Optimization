#include "gemm_common.h"

// Function to perform GEMM operation on CPU
void gemm_cpu(float* A, float* B, float* C, const int m, const int n, const int k)
{
    // Start of the GEMM operation
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0f; // Initialize the result element
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j]; // Perform the multiplication and accumulation
            }
        }
    }
}