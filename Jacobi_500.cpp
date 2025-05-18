#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

vector<double> jacobiMethod(const vector<vector<double>>& A, const vector<double>& b, int max_iter = 1000, double tol = 1e-10) {
    int n = A.size();
    vector<double> x(n, 0.0);
    vector<double> x_new(n, 0.0);

    for (int iter = 0; iter < max_iter; ++iter) {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
        }

        double error = 0.0;
        for (int i = 0; i < n; ++i) {
            error += abs(x_new[i] - x[i]);
        }

        x = x_new;

        if (error < tol) {
            break;
        }
    }

    return x;
}

vector<vector<double>> generateMatrix(int N) {
    vector<vector<double>> A(N, vector<double>(N, 1.0));
    for (int i = 0; i < N; ++i) {
        A[i][i] = N;
    }
    return A;
}

vector<double> generateVector(int N) {
    return vector<double>(N, 1.0);
}

void solveAndMeasure(int N, int num_threads) {
    omp_set_num_threads(num_threads);

    auto A = generateMatrix(N);
    auto b = generateVector(N);

    double start = omp_get_wtime();
    auto x = jacobiMethod(A, b);
    double end = omp_get_wtime();

    cout << "N = " << N << ", Threads = " << num_threads << ", Time = " << (end - start) << " seconds" << endl;
}

int main() {
    int N = 500; // Размер матрицы

    vector<int> threads_list = { 2, 4, 8, 16 };

    for (int threads : threads_list) {
        solveAndMeasure(N, threads);
    }

    return 0;
}