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

inline void add(const size_t n, const float *A, const float *B, float *C) {
    for (size_t i = 0; i < n * n; ++i) {
        C[i] = A[i] + B[i];
    }
}

inline void subtract(const size_t n, const float *A, const float *B, float *C) {
    for (size_t i = 0; i < n * n; ++i) {
        C[i] = A[i] - B[i];
    }
}

void decompose(const size_t n, const float *M, 
               float *M11, float *M12, float *M21, float *M22) {
    const size_t half_n = n / 2;

    for (size_t i = 0; i < half_n; ++i) {
        for (size_t j = 0; j < half_n; ++j) {
            size_t i_offset = i + half_n;
            size_t j_offset = j + half_n;

            M11[idx(i, j, half_n)] = M[idx(i, j, n)];
            M12[idx(i, j, half_n)] = M[idx(i, j_offset, n)];
            M21[idx(i, j, half_n)] = M[idx(i_offset, j, n)];
            M22[idx(i, j, half_n)] = M[idx(i_offset, j_offset, n)];
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

            std::fill_n(&C_loc[0][0], block_sz * block_sz, 0.0f);

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
        const size_t n_minus_1 = n - 1;

        // manually handle the last row and column
        // row
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                C[idx(n_minus_1, j, n)] += A[idx(n_minus_1, k, n)] * B[idx(k, j, n)];
            }
        }

        // column
        for (size_t i = 0; i < n_minus_1; ++i) {
            for (size_t k = 0; k < n; ++k) {
                C[idx(i, n_minus_1, n)] += A[idx(i, k, n)] * B[idx(k, n_minus_1, n)];
            }
        }
    }

    // decompose matrices
    const size_t half_n = n / 2;
    const size_t half_n_sq = half_n * half_n;
    float *A11 = new float[half_n_sq];
    float *A12 = new float[half_n_sq];
    float *A21 = new float[half_n_sq];
    float *A22 = new float[half_n_sq];
    float *B11 = new float[half_n_sq];
    float *B12 = new float[half_n_sq];
    float *B21 = new float[half_n_sq];
    float *B22 = new float[half_n_sq];
    
    decompose(n, A, A11, A12, A21, A22);
    decompose(n, B, B11, B12, B21, B22);

    // procedure as described in https://arxiv.org/pdf/0707.2347

    float *S1 = new float[half_n_sq];
    float *S2 = new float[half_n_sq];
    float *S3 = new float[half_n_sq];
    float *S4 = new float[half_n_sq];
    float *T1 = new float[half_n_sq];
    float *T2 = new float[half_n_sq];
    float *T3 = new float[half_n_sq];
    float *T4 = new float[half_n_sq];

    // 8 additions/subtractions
    add(        half_n,     A21,    A22,    S1);        // S1 = A21 + A22
    subtract(   half_n,     S1,     A11,    S2);        // S2 = S1 - A11
    subtract(   half_n,     A11,    A21,    S3);        // S3 = A11 - A21
    subtract(   half_n,     B12,    B11,    T1);        // T1 = B12 - B11
    subtract(   half_n,     B22,    T1,     T2);        // T2 = B22 - T1
    subtract(   half_n,     B22,    B12,    T3);        // T3 = B22 - B12
    subtract(   half_n,     A12,    S2,     S4);        // S4 = A12 - S2
    subtract(   half_n,     T2,     B21,    T4);        // T4 = T2 - B21

    // 7 multiplications
    float *P1 = new float[half_n_sq];
    float *P2 = new float[half_n_sq];
    float *P3 = new float[half_n_sq];
    float *P4 = new float[half_n_sq];
    float *P5 = new float[half_n_sq];
    float *P6 = new float[half_n_sq];
    float *P7 = new float[half_n_sq];

    strassen_matmul(   half_n,     A11,    B11,     P1);         // P1 = A11 * B11
    strassen_matmul(   half_n,     A12,    B21,     P2);         // P2 = A12 * B21
    strassen_matmul(   half_n,     S4,     B22,     P3);         // P3 = S4 * B22
    strassen_matmul(   half_n,     A22,    T4,      P4);         // P4 = A22 * T4
    strassen_matmul(   half_n,     S1,     T1,      P5);         // P5 = S1 * T1
    strassen_matmul(   half_n,     S2,     T2,      P6);         // P6 = S2 * T2
    strassen_matmul(   half_n,     S3,     T3,      P7);         // P7 = S3 * T3

    // 7 final additions/subtractions
    // we reuse S and T matrices as temporaries, namely
    // S1 to S4 will be U1 to U4 and T1 to T3 will be U5 to U7
    float *U1 = S1, *U2 = S2, *U3 = S3, *U4 = S4;
    float *U5 = T1, *U6 = T2, *U7 = T3;

    add(        half_n,     P1,     P2,     U1);        // U1 = P1 + P2
    add(        half_n,     P1,     P6,     U2);        // U2 = P1 + P6
    add(        half_n,     U2,     P7,     U3);        // U3 = U2 + P7
    add(        half_n,     U2,     P5,     U4);        // U4 = U2 + P5
    add(        half_n,     U4,     P3,     U5);        // U5 = U4 + P3
    subtract(   half_n,     U3,     P4,     U6);        // U6 = U3 - P4
    add(        half_n,     U3,     P5,     U7);        // U7 = U3 + P5

    // finally, compose C
    // C = | U1  U5 |
    //     | U6  U7 |

    for (size_t i = 0; i < half_n; ++i) {
        for (size_t j = 0; j < half_n; ++j) {
            size_t i_offset = i + half_n;
            size_t j_offset = j + half_n;

            C[idx(i, j, n)] = U1[idx(i, j, half_n)];
            C[idx(i, j_offset, n)] = U5[idx(i, j, half_n)];
            C[idx(i_offset, j, n)] = U6[idx(i, j, half_n)];
            C[idx(i_offset, j_offset, n)] = U7[idx(i, j, half_n)];
        }
    }
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