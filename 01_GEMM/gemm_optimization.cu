#include <iostream>
#include <vector>
#include <chrono>



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

// Function to initialize a matrix with random values
// Using 2D vector to represent the matrix
void initialize_matrix(std::vector<std::vector<float>>& matrix) {
    for (auto& row : matrix) {
        for (auto& element : row) {
            element = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
        }
    }
}

int main() {
    std::cout << "GEMM Optimization Example" << std::endl;
    std::cout << "Initializing matrices..." << std::endl;
    const int m = 1024; // Number of rows in A and C
    const int n = 1024; // Number of columns in B and C
    const int k = 512; // Number of columns in A and rows in B

    // Initialize matrices A, B, and C
    std::vector<std::vector<float>> A(m, std::vector<float>(k));
    std::vector<std::vector<float>> B(k, std::vector<float>(n));
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0.0f)); // Initialize C with zeros

    // A = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
    // B = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    initialize_matrix(A);
    initialize_matrix(B);
    initialize_matrix(C);

    // Perform GEMM operation
    // Time the GEMM operation
    std::cout << "Performing GEMM operation..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    gemm_cpu(A, B, C, m, n, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "GEMM operation completed in " << duration.count() << " microseconds." << std::endl;

    // Check the result of the GEMM operation
    std::cout << "Result matrix C (first 3 elements):" << std::endl;
    for (int i = 0; i < std::min(m, 3); i++) {
        for (int j = 0; j < std::min(n, 3); j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }


    std::cout << "Program completed." << std::endl;
    return 0;
}