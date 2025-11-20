// Compile with:
// g++ strassen.cpp -Ofast -march=native -o main -lstdc++exp -std=c++23 -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion
// -lstdc++exp -std=c++23: link the experimental standard library for std::println
// -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion: enable more warnings to help write better code (optional)

#include <algorithm>

constexpr size_t size = 2000;
constexpr size_t threshold = 601;
constexpr size_t block_sz = 280;

struct MatrixView {
    float* where;
    size_t dim;

    const float& operator()(size_t i, size_t j) const {
        return where[i * dim + j];
    }
    float& operator()(size_t i, size_t j) {
        return where[i * dim + j];
    }
    MatrixView subview(size_t row_offset, size_t col_offset) const {
        return MatrixView{&where[row_offset * dim + col_offset], dim};
    }

    void add(const MatrixView other, size_t n) {
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                (*this)(i, j) += other(i, j);
    }
};

// C += A @ B. C must not overlap with A or B
void naive_matmul(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k)
                C(i, j) += A(i, k) * B(k, j);
}

// C = A + B. C may overlap with A or B
void add(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            C(i, j) = A(i, j) + B(i, j);
}

// C = A - B. C may overlap with A or B
void sub(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            C(i, j) = A(i, j) - B(i, j);
}

// C = A @ B. C must not overlap with A or B
void improved_matmul(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    alignas(64) static float C_loc[block_sz][block_sz];

    for (size_t chunk_iA = 0; chunk_iA < n; chunk_iA += block_sz) {
        size_t max_iA = std::min(chunk_iA + block_sz, n);
        for (size_t chunk_iB = 0; chunk_iB < n; chunk_iB += block_sz) {
            size_t max_iB = std::min(chunk_iB + block_sz, n);
            std::fill_n(&C_loc[0][0], block_sz * block_sz, 0.0f);
            for (size_t chunk_jA = 0; chunk_jA < n; chunk_jA += block_sz) {
                size_t max_jA = std::min(chunk_jA + block_sz, n);
                for (size_t iA = chunk_iA; iA < max_iA; iA++)
                    for (size_t jA = chunk_jA; jA < max_jA; jA++) {
                        float a = A(iA, jA);
                        for (size_t jB = chunk_iB; jB < max_iB; jB++)
                            C_loc[iA - chunk_iA][jB - chunk_iB] += a * B(jA, jB);
                    }
            }

            for (size_t iA = chunk_iA; iA < max_iA; iA++)
                for (size_t jB = chunk_iB; jB < max_iB; jB++)
                    C(iA, jB) += C_loc[iA - chunk_iA][jB - chunk_iB];
        }
    }
}

// C += A @ B. C must not overlap with A or B
void strassen_matmul(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    if (n <= threshold) {
        improved_matmul(A, B, C, n);
        return;
    }

    if (n % 2 != 0) {
        size_t n1 = n + 1;
        float* memory_block = new float[3 * n1 * n1]{};
        MatrixView A_pad{memory_block, n1};
        MatrixView B_pad{memory_block + n1 * n1, n1};
        MatrixView C_pad{memory_block + 2 * n1 * n1, n1};

        for (size_t i = 0; i < n; ++i) {
            std::copy(&A(i, 0), &A(i, n), &A_pad(i, 0));
            std::copy(&B(i, 0), &B(i, n), &B_pad(i, 0));
        }

        strassen_matmul(A_pad, B_pad, C_pad, n1);
        for (size_t i = 0; i < n; ++i) {
            std::copy(&C_pad(i, 0), &C_pad(i, n), &C(i, 0));
        }

        delete[] memory_block;
        return;
    }

    size_t half_n = n / 2;
    auto A11 = A.subview(0, 0);
    auto A12 = A.subview(0, half_n);
    auto A21 = A.subview(half_n, 0);
    auto A22 = A.subview(half_n, half_n);
    auto B11 = B.subview(0, 0);
    auto B12 = B.subview(0, half_n);
    auto B21 = B.subview(half_n, 0);
    auto B22 = B.subview(half_n, half_n);

    size_t half_n_sq = half_n * half_n;
    float* memory_block = new float[18 * half_n_sq];

    MatrixView S1{memory_block, half_n};
    MatrixView S2{memory_block + half_n_sq, half_n};
    MatrixView S3{memory_block + 2 * half_n_sq, half_n};
    MatrixView S4{memory_block + 3 * half_n_sq, half_n};
    add(A21, A22, S1, half_n);
    sub(S1, A11, S2, half_n);
    sub(A11, A21, S3, half_n);
    sub(A12, S2, S4, half_n);

    MatrixView T1{memory_block + 4 * half_n_sq, half_n};
    MatrixView T2{memory_block + 5 * half_n_sq, half_n};
    MatrixView T3{memory_block + 6 * half_n_sq, half_n};
    MatrixView T4{memory_block + 7 * half_n_sq, half_n};
    sub(B12, B11, T1, half_n);
    sub(B22, T1, T2, half_n);
    sub(B22, B12, T3, half_n);
    sub(T2, B21, T4, half_n);
    
    MatrixView P1{memory_block + 8 * half_n_sq, half_n};
    MatrixView P2{memory_block + 9 * half_n_sq, half_n};
    MatrixView P3{memory_block + 10 * half_n_sq, half_n};
    MatrixView P4{memory_block + 11 * half_n_sq, half_n};
    MatrixView P5{memory_block + 12 * half_n_sq, half_n};
    MatrixView P6{memory_block + 13 * half_n_sq, half_n};
    MatrixView P7{memory_block + 14 * half_n_sq, half_n};
    strassen_matmul(A11, B11, P1, half_n);
    strassen_matmul(A12, B21, P2, half_n);
    strassen_matmul(S4, B22, P3, half_n);
    strassen_matmul(A22, T4, P4, half_n);
    strassen_matmul(S1, T1, P5, half_n);
    strassen_matmul(S2, T2, P6, half_n);
    strassen_matmul(S3, T3, P7, half_n);

    auto U1 = C.subview(0, 0);
    auto U5 = C.subview(0, half_n);
    auto U6 = C.subview(half_n, 0);
    auto U7 = C.subview(half_n, half_n);
    MatrixView U2{memory_block + 15 * half_n_sq, half_n};
    MatrixView U3{memory_block + 16 * half_n_sq, half_n};
    MatrixView U4{memory_block + 17 * half_n_sq, half_n};
    add(P1, P2, U1, half_n);
    add(P1, P6, U2, half_n);
    add(U2, P7, U3, half_n);
    add(U2, P5, U4, half_n);
    add(U4, P3, U5, half_n);
    sub(U3, P4, U6, half_n);
    add(U3, P5, U7, half_n);

    delete[] memory_block;
}

float maxdiff(const MatrixView C1, const MatrixView C2, size_t n) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            max_diff = std::max(max_diff, std::abs(C1(i, j) - C2(i, j)));
        }
    }
    return max_diff;
}

#include <print>
#include <random>
#include <chrono>

float A_in[size * size];
float B_in[size * size];
float C1[size * size];
float C2[size * size];

// Compile with:
// g++ strassen.cpp -Ofast -march=native -o main -lstdc++exp -std=c++23 -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion
// -lstdc++exp -std=c++23: link the experimental standard library for std::println
// -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion: enable more warnings to help write better code (optional)

int main() {
    std::println("Matrix size: {}x{}", size, size);
    std::println("Threshold: {}", threshold);
    std::println("Block size: {}", block_sz);

    MatrixView A{A_in, size};
    MatrixView B{B_in, size};

    // Allocate and fill matrices (inline; uniform [-1,1] random)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            A(i, j) = dis(gen);
            B(i, j) = dis(gen);
        }
    }
    
    MatrixView C_naive{C1, size};
    auto start = std::chrono::high_resolution_clock::now();
    naive_matmul(A, B, C_naive, size);
    auto end = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::duration<double, std::milli>(end - start);
    std::println("Naive: {} ms", t1);

    MatrixView C_improved{C2, size};
    start = std::chrono::high_resolution_clock::now();
    improved_matmul(A, B, C_improved, size);
    end = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::duration<double, std::milli>(end - start);
    std::println("Improved: {} ms", t2);
    std::println("Vs Naive: max diff: {:.5f}. Speedup: {:.2f}x", maxdiff(C_naive, C_improved, size), t1.count() / t2.count());

    std::fill_n(C_improved.where, size * size, 0.0f);
    start = std::chrono::high_resolution_clock::now();
    strassen_matmul(A, B, C_improved, size);
    end = std::chrono::high_resolution_clock::now();
    t2 = std::chrono::duration<double, std::milli>(end - start);
    std::println("Strassen: {} ms", t2);
    std::println("Vs Naive: max diff: {:.5f}. Speedup: {:.2f}x", maxdiff(C_naive, C_improved, size), t1.count() / t2.count());

    return 0;
}