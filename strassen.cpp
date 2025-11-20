// Compile with:
// g++ strassen.cpp -Ofast -march=native -o main -lstdc++exp -std=c++23 -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion
// -lstdc++exp -std=c++23: link the experimental standard library for std::println
// -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion: enable more warnings to help write better code (optional)

#include <algorithm>

constexpr size_t threshold = 300;
constexpr size_t block_sz = 280;


// utility struct to represent a matrix view
// @param where: pointer to the first element of the view
// @param dim: leading dimension (number of columns in the original matrix)
struct MatrixView {
    float* where;
    size_t dim;

    // access element (i,j) of the view
    const float& operator()(size_t i, size_t j) const {
        return where[i * dim + j];
    }

    // access element (i,j) of the view
    float& operator()(size_t i, size_t j) {
        return where[i * dim + j];
    }

    // create a subview starting at (row_offset, col_offset)
    MatrixView subview(size_t row_offset, size_t col_offset) const {
        return MatrixView{where + row_offset * dim + col_offset, dim};
    }

    // kinda like `MatrixView += other`
    void add(const MatrixView other, size_t n) {
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                (*this)(i, j) += other(i, j);
    }

    // kinda like `MatrixView -= other`
    void sub(const MatrixView other, size_t n) {
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                (*this)(i, j) -= other(i, j);
    }

    // set all elements (of the view only) to zero
    void clear(size_t n) {
        for (size_t i = 0; i < n; ++i)
            std::fill_n(where + i * dim, n, 0.0f);
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

// C += A @ B. C must not overlap with A or B
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

// C = A @ B. C must not overlap with A or B
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
    auto C11 = C.subview(0, 0);
    auto C12 = C.subview(0, half_n);
    auto C21 = C.subview(half_n, 0);
    auto C22 = C.subview(half_n, half_n);

    size_t half_n_sq = half_n * half_n;
    float* memory_block = new float[2 * half_n_sq]{};
    MatrixView X{memory_block, half_n};
    MatrixView Y{memory_block + half_n_sq, half_n};

    sub(A11, A21, X, half_n);
    sub(B22, B12, Y, half_n);
    strassen_matmul(X, Y, C21, half_n);
    add(A21, A22, X, half_n);
    sub(B12, B11, Y, half_n);
    strassen_matmul(X, Y, C22, half_n);
    X.sub(A11, half_n);
    sub(B22, Y, Y, half_n);
    strassen_matmul(X, Y, C12, half_n);
    sub(A12, X, X, half_n);
    strassen_matmul(X, B22, C11, half_n);
    std::fill_n(&X.where[0], half_n_sq, 0.0f);
    strassen_matmul(A11, B11, X, half_n);
    C12.add(X, half_n);
    C21.add(C12, half_n);
    C12.add(C22, half_n);
    C22.add(C21, half_n);
    C12.add(C11, half_n);
    Y.sub(B21, half_n);
    C11.clear(half_n);
    strassen_matmul(A22, Y, C11, half_n);
    C21.sub(C11, half_n);
    C11.clear(half_n);
    strassen_matmul(A12, B21, C11, half_n);
    C11.add(X, half_n);

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
#include <string>

// Compile with:
// g++ strassen.cpp -Ofast -march=native -o main -lstdc++exp -std=c++23 -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion
// -lstdc++exp -std=c++23: link the experimental standard library for std::println
// -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion: enable more warnings to help write better code (optional)

int main(int argc, char** argv) {
    if (argc != 2) {
        std::println("Usage: {} <matrix_size>", argv[0]);
        return 1;
    }

    size_t size = std::stoul(argv[1]);
    // std::println("Matrix size: {}x{}", size, size);
    // std::println("Threshold: {}", threshold);
    // std::println("Block size: {}", block_sz);

    float *A_in = new float[size * size];
    float *B_in = new float[size * size];
    float *C1 = new float[size * size]{};
    // float *C2 = new float[size * size]{};

    MatrixView A{A_in, size};
    MatrixView B{B_in, size};
    MatrixView C_naive{C1, size};
    // MatrixView C_improved{C2, size};

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
    
    auto start = std::chrono::high_resolution_clock::now();
    improved_matmul(A, B, C_naive, size);
    auto end = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::duration<double, std::milli>(end - start);
    std::println("{}x{}: {}", size, size, t1);

    // start = std::chrono::high_resolution_clock::now();
    // strassen_matmul(A, B, C_improved, size);
    // end = std::chrono::high_resolution_clock::now();
    // auto t2 = std::chrono::duration<double, std::milli>(end - start);
    // std::println("Strassen: {}", t2);
    // std::println("Speedup: {:.2f}x", t1.count() / t2.count());
    // std::println("Max difference: {:.5f}", maxdiff(C_naive, C_improved, size));

    return 0;
}