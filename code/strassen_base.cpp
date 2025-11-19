#include <algorithm>
#include <iostream>
#include <random>
#include <iomanip>
#include <chrono>

constexpr size_t size = 2880;
constexpr size_t threshold = 128;
constexpr size_t block_sz = 64;  // Change this to suit your device
float A_in[size][size], B_in[size][size], C_stras[size][size]{};

inline size_t idx(const size_t row, const size_t col, const size_t n) {
    return row * n + col;
}

void decompose(const size_t n, const float *M, 
               float *M11, float *M12, float *M21, float *M22) {
    const size_t half_n = n / 2;

    for (size_t i = 0; i < half_n; ++i) {
        for (size_t j = 0; j < half_n; ++j) {
            M11[idx(i, j, half_n)] = M[idx(i, j, n)];
            M12[idx(i, j, half_n)] = M[idx(i, j + half_n, n)];
            M21[idx(i, j, half_n)] = M[idx(i + half_n, j, n)];
            M22[idx(i, j, half_n)] = M[idx(i + half_n, j + half_n, n)];
        }
    }
}

// Gemini-refactored Matmul on 1D arrays 
void improved_matmul(const size_t n, const float* A, const float* B, float* C) {
    // Local accumulator on the stack.
    // alignas(64) helps with AVX-512, alignas(32) for AVX2.
    alignas(64) float C_loc[block_sz][block_sz];

    for (size_t chunk_i = 0; chunk_i < n; chunk_i += block_sz) {
        size_t max_i = std::min(chunk_i + block_sz, n);

        for (size_t chunk_j = 0; chunk_j < n; chunk_j += block_sz) {
            size_t max_j = std::min(chunk_j + block_sz, n);

            for (size_t i = 0; i < block_sz; ++i) {
                 std::fill(std::begin(C_loc[i]), std::end(C_loc[i]), 0.0f);
            }

            for (size_t chunk_k = 0; chunk_k < n; chunk_k += block_sz) {
                size_t max_k = std::min(chunk_k + block_sz, n);

                for (size_t i = chunk_i; i < max_i; ++i) {
                    size_t loc_i = i - chunk_i;

                    for (size_t k = chunk_k; k < max_k; ++k) {
                        float val_A = A[idx(i, k, n)];

                        #pragma omp simd
                        for (size_t j = chunk_j; j < max_j; ++j) {
                            C_loc[loc_i][j - chunk_j] += val_A * B[idx(k, j, n)];
                        }
                    }
                }
            }

            for (size_t i = chunk_i; i < max_i; ++i) {
                size_t loc_i = i - chunk_i;
                
                #pragma omp simd
                for (size_t j = chunk_j; j < max_j; ++j) {
                    C[idx(i, j, n)] += C_loc[loc_i][j - chunk_j];
                }
            }
        }
    }
}

void strassen_matmul(const size_t n, const float *A, const float *B, float *C) {
    if (n <= 0) return;

    // small matrix, use improved matmul
    if (n <= threshold) {
        improved_matmul(n, A, B, C);
        return;
    }

    if (n % 2) {
        // n odd, strip the last row and column
        const size_t new_n = n - 1;

        float *A_row_last = new float[new_n];
        float *B_col_last = new float[new_n];
        
        for (size_t i = 0; i < new_n; ++i) {
            A_row_last[i] = A[idx(new_n, i, n)];
            B_col_last[i] = B[idx(i, new_n, n)];
        }

        // manually handle the last row and column
        // row
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                C[idx(new_n, j, n)] += A_row_last[k] * B[idx(k, j, n)];
            }
        }

        // column
        for (size_t i = 0; i < new_n; ++i) {
            for (size_t k = 0; k < n; ++k) {
                C[idx(i, new_n, n)] += A[idx(i, k, n)] * B_col_last[k];
            }
        }
    }

    // decompose matrices
    const size_t half_n = n / 2;
    float *A11 = new float[half_n * half_n];
    float *A12 = new float[half_n * half_n];
    float *A21 = new float[half_n * half_n];
    float *A22 = new float[half_n * half_n];
    float *B11 = new float[half_n * half_n];
    float *B12 = new float[half_n * half_n];
    float *B21 = new float[half_n * half_n];
    float *B22 = new float[half_n * half_n];
    
    decompose(n, A, A11, A12, A21, A22);
    decompose(n, B, B11, B12, B21, B22);

    
}

int main() {
    std::cout << "Matrix size: " << size << "x" << size << "\n";

    // Allocate and fill matrices (inline; uniform [-1,1] random)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            A_in[i][j] = dis(gen);
            B_in[i][j] = dis(gen);
        }
    }
}