#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void initializeMatrices(std::vector<std::vector<double>>& A,
    std::vector<std::vector<double>>& B,
    int N) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = (i == j) ? 2.0 * N : N;
            B[i][j] = (i == j) ? N : 1.0;
        }
    }
}

void blockMatrixMultiplication(const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B,
    std::vector<std::vector<double>>& C,
    int N, int blockSize) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += blockSize) {
        for (int j = 0; j < N; j += blockSize) {
            for (int k = 0; k < N; k += blockSize) {
                // Process block
                for (int ii = i; ii < std::min(i + blockSize, N); ++ii) {
                    for (int jj = j; jj < std::min(j + blockSize, N); ++jj) {
                        double sum = 0.0;
                        for (int kk = k; kk < std::min(k + blockSize, N); ++kk) {
                            sum += A[ii][kk] * B[kk][jj];
                        }
#pragma omp atomic
                        C[ii][jj] += sum;
                    }
                }
            }
        }
    }
}

void runMultiplication(int N, int threads) {
    omp_set_num_threads(threads);

    // Initialize matrices
    std::vector<std::vector<double>> A(N, std::vector<double>(N));
    std::vector<std::vector<double>> B(N, std::vector<double>(N));
    std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));

    initializeMatrices(A, B, N);

    // Choose block size - common heuristic is to use sqrt of cache size, 
    // but for simplicity we'll use a fixed value
    int blockSize = 64; // Can be adjusted based on cache size

    auto start = std::chrono::high_resolution_clock::now();

    blockMatrixMultiplication(A, B, C, N, blockSize);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Matrix size: " << N << "x" << N
        << ", Threads: " << threads
        << ", Time: " << duration.count() << " seconds" << std::endl;
}

int main() {
    // Test with different matrix sizes and thread counts
    int N = 1000; // Matrix size (can be changed)

    std::vector<int> threads_counts = { 2, 4, 8, 16 };

    for (int threads : threads_counts) {
        runMultiplication(N, threads);
    }

    return 0;
}