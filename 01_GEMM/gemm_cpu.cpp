#include "gemm_common.h"

// Function to perform GEMM operation on CPU
void gemm_cpu(std::vector<std::vector<float>>& A, 
            std::vector<std::vector<float>>& B, 
            std::vector<std::vector<float>>& C, 
            const int m, 
            const int n, 
            const int k)
{
    // Start of the GEMM operation
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0f; // Initialize the result element
            for (int l = 0; l < k; l++) {
                C[i][j] += A[i][l] * B[l][j]; // Perform the multiplication and accumulation
            }
        }
    }
}