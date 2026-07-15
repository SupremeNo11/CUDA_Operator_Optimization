#include "../inc/cuda_common.h"

// Function to initialize a matrix with random values
// Using 2D vector to represent the matrix
void init_matrix(float* Matrix, int H, int W) 
{
    int i, j;
    for (i = 0; i < H; ++i) {
        for (j = 0; j < W; ++j) {
            Matrix[i * W + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}