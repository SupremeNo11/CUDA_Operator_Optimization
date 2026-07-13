#include "../inc/cuda_common.h"

// Function to initialize a matrix with random values
// Using 2D vector to represent the matrix
void initialize_matrix(std::vector<std::vector<float>>& matrix) {
    for (auto& row : matrix) {
        for (auto& element : row) {
            element = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
        }
    }
}