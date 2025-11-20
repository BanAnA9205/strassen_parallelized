// Compile with:
// g++ strassen.cpp -Ofast -march=native -o main -lstdc++exp -std=c++23 -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion
// -lstdc++exp -std=c++23: link the experimental standard library for std::println
// -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion: enable more warnings to help write better code (optional)

#include <algorithm>
#include <vector>

constexpr size_t threshold = 300;
constexpr size_t block_sz = 280;

// utility struct to represent a matrix view
// @param where: pointer to the first element of the view
// @param dim: leading dimension (number of columns in the original matrix)
struct MatrixView {
    float* where;
    size_t dim, rangeI, rangeJ;

    // access element (i,j) of the view
    float operator()(size_t i, size_t j) const {
        if (i >= rangeI || j >= rangeJ) return 0.0f;
        return where[i * dim + j];
    }

    // access element (i,j) of the view
    float& operator()(size_t i, size_t j) {
        return where[i * dim + j];
    }

    // create a subview starting at (i, j)
    MatrixView subview(size_t i, size_t j, size_t range) const {
        return MatrixView{where + i * dim + j, dim, std::min(rangeI - i, range), std::min(rangeJ - j, range)};
    }

    static MatrixView makeview(float* where, size_t dim) {
        return MatrixView{where, dim, dim, dim};
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

// add() with bounds checking for C
void addSafe(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    size_t rangeI = std::min(C.rangeI, n);
    size_t rangeJ = std::min(C.rangeJ, n);
    for (size_t i = 0; i < rangeI; ++i)
        for (size_t j = 0; j < rangeJ; ++j)
            C(i, j) = A(i, j) + B(i, j);
}

// sub() with bounds checking for C
void subSafe(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    size_t rangeI = std::min(C.rangeI, n);
    size_t rangeJ = std::min(C.rangeJ, n);
    for (size_t i = 0; i < rangeI; ++i)
        for (size_t j = 0; j < rangeJ; ++j)
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

    size_t half_n = (n + 1) / 2;
    auto A11 = A.subview(0, 0, half_n);
    auto A12 = A.subview(0, half_n, half_n);
    auto A21 = A.subview(half_n, 0, half_n);
    auto A22 = A.subview(half_n, half_n, half_n);
    auto B11 = B.subview(0, 0, half_n);
    auto B12 = B.subview(0, half_n, half_n);
    auto B21 = B.subview(half_n, 0, half_n);
    auto B22 = B.subview(half_n, half_n, half_n);
    auto C11 = C.subview(0, 0, half_n);
    auto C12 = C.subview(0, half_n, half_n);
    auto C21 = C.subview(half_n, 0, half_n);
    auto C22 = C.subview(half_n, half_n, half_n);

    size_t half_n_sq = half_n * half_n;
    std::vector<float> memory_block(5 * half_n_sq, 0.0f);
    auto X = MatrixView::makeview(&memory_block[0], half_n);
    auto Y = MatrixView::makeview(&memory_block[half_n_sq], half_n);
    auto U1 = MatrixView::makeview(&memory_block[2 * half_n_sq], half_n);
    auto U2 = MatrixView::makeview(&memory_block[3 * half_n_sq], half_n);
    auto U3 = MatrixView::makeview(&memory_block[4 * half_n_sq], half_n);

    sub(A11, A21, X, half_n);
    sub(B22, B12, Y, half_n);
    strassen_matmul(X, Y, U1, half_n);
    add(A21, A22, X, half_n);
    sub(B12, B11, Y, half_n);
    strassen_matmul(X, Y, U2, half_n);
    X.sub(A11, half_n);
    sub(B22, Y, Y, half_n);
    strassen_matmul(X, Y, U3, half_n);
    sub(A12, X, X, half_n);
    strassen_matmul(X, B22, C11, half_n);
    std::fill_n(&X.where[0], half_n_sq, 0.0f);
    strassen_matmul(A11, B11, X, half_n);
    U3.add(X, half_n);
    U1.add(U3, half_n);
    U3.add(U2, half_n);
    addSafe(U2, U1, C22, half_n);
    addSafe(U3, C11, C12, half_n);
    Y.sub(B21, half_n);
    C11.clear(half_n);
    strassen_matmul(A22, Y, C11, half_n);
    subSafe(U1, C11, C21, half_n);
    C11.clear(half_n);
    strassen_matmul(A12, B21, C11, half_n);
    C11.add(X, half_n);
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
    std::println("Matrix size: {}x{}", size, size);
    std::println("Threshold: {}", threshold);
    std::println("Block size: {}", block_sz);

    std::vector<float> memory_block(4 * size * size, 0.0f);
    MatrixView A = MatrixView::makeview(&memory_block[0], size);
    MatrixView B = MatrixView::makeview(&memory_block[size * size], size);
    MatrixView C_naive = MatrixView::makeview(&memory_block[2 * size * size], size);
    MatrixView C_improved = MatrixView::makeview(&memory_block[3 * size * size], size);

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
    std::println("Improved: {}", t1);

    start = std::chrono::high_resolution_clock::now();
    strassen_matmul(A, B, C_improved, size);
    end = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::duration<double, std::milli>(end - start);
    std::println("Strassen: {}", t2);
    std::println("Speedup: {:.2f}x", t1.count() / t2.count());
    std::println("Max difference: {:.5f}", maxdiff(C_naive, C_improved, size));

    return 0;
}