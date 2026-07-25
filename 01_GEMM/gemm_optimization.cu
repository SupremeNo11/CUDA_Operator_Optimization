#include "gemm_common.h"



int main() 
{
    /*Show all devices information*/
    SYS_STATUS stas = cuPrintDevicesInfo();

    if (SYS_STATUS_SUCCESS != stas) {
        exit(EXIT_FAILURE);
    }

    std::cout << "===========================================================\n" << std::endl;

    std::cout << "GEMM Optimization Example" << std::endl;
    std::cout << "Initializing matrices..." << std::endl;
    const int m = 2048; // Number of rows in A and C
    const int n = 2048; // Number of columns in B and C
    const int k = 1024; // Number of columns in A and rows in B

    float* h_A = new float[m * k];
    float* h_B = new float[k * n];
    float* h_C = new float[m * n];

    // float temp_A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    // float temp_B[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // memcpy(h_A, temp_A, m * k * sizeof(float));
    // memcpy(h_B, temp_B, k * n * sizeof(float));

    init_matrix(h_A, m, k);
    init_matrix(h_B, k, n);
    init_matrix(h_C, m, n);

    // Perform GEMM operation
    // Time the GEMM operation
    std::cout << "Performing GEMM operation..." << std::endl;

#if _ENABLE_CPU
    std::cout << "============================= CPU ==============================\n" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    gemm_cpu(h_A, h_B, h_C, m, n, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "GEMM operation completed in " << duration.count() << " microseconds." << std::endl;

    // Check the result of the GEMM operation
    std::cout << "Result matrix C (first 3 elements):" << std::endl;
    for (int i = 0; i < std::min(m, 3); i++) {
        for (int j = 0; j < std::min(n, 3); j++) {
            std::cout << h_C[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
#endif

    std::cout << "============================= GPU ==============================\n" << std::endl;
    // GPU calculate logic
    gemm_gpu(h_A, h_B, h_C, m, n, k);

    delete[]    h_A;
    delete[]    h_B;
    delete[]    h_C;

    std::cout << "Program completed." << std::endl;
    return 0;
}