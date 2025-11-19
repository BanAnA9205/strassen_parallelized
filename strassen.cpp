// Compile with:
// g++ strassen.cpp -Ofast -march=native -o main -lstdc++exp -std=c++23 -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion
// -lstdc++exp -std=c++23: link the experimental standard library for std::println
// -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion: enable more warnings to help write better code (optional)

#include <algorithm>

constexpr size_t size = 10000;
constexpr size_t threshold = 1000;
constexpr size_t block_sz = 280;

/**
 * @brief Non-owning view into a 2D row-major float matrix.
 *
 * This lightweight structure provides indexed access to a contiguous float buffer
 * interpreted as a 2D matrix with a fixed row stride. It does not allocate or
 * manage memory; it only stores a pointer into an existing buffer and the
 * leading dimension (stride) used to compute element offsets.
 *
 *  - where: Pointer to the first element of this view.
 *  - dim:   Leading dimension (row stride). Used to compute element offsets as
 *           index = i * dim + j. This value should be at least the number of
 *           columns in any matrix or subview accessed through this view.
 *
 *  - operator()(size_t i, size_t j) const
 *      Returns the element at row i, column j by value. The caller must ensure
 *      indices are within bounds.
 *
 *  - operator()(size_t i, size_t j)
 *      Returns a reference to the element at row i, column j allowing mutation.
 *      The caller must ensure indices are within bounds.
 *
 *  - subview(size_t row_offset, size_t col_offset) const
 *      Creates and returns another MatrixView that begins at the element
 *      (row_offset, col_offset) within the same underlying buffer. The returned
 *      view preserves the same leading dimension (dim). No copying occurs; the
 *      returned view is non-owning and its safety depends on the original buffer.
 */
struct MatrixView {
    float* where;
    size_t dim;

    float operator()(size_t i, size_t j) const {
        return where[i * dim + j];
    }
    float& operator()(size_t i, size_t j) {
        return where[i * dim + j];
    }
    MatrixView subview(size_t row_offset, size_t col_offset) const {
        return MatrixView{&where[row_offset * dim + col_offset], dim};
    }
};

// C may overlap with A or B
void add(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            C(i, j) = A(i, j) + B(i, j);
}

// C may overlap with A or B
void sub(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            C(i, j) = A(i, j) - B(i, j);
}

// C must not overlap with A or B
void improved_matmul(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    alignas(64) float C_loc[block_sz][block_sz];

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

void strassen_matmul(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    if (n <= threshold) {
        improved_matmul(A, B, C, n);
        return;
    }

    if (n % 2 != 0) {
        // n odd, strip the last row and column
        size_t n_minus_1 = n - 1;

        // manually handle the last row and column
        // row
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                C(n_minus_1, j) += A(n_minus_1, k) * B(k, j);
            }
        }

        // column
        for (size_t i = 0; i < n_minus_1; ++i) {
            for (size_t k = 0; k < n; ++k) {
                C(i, n_minus_1) += A(i, k) * B(k, n_minus_1);
            }
        }
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
    auto U1 = C.subview(0, 0);
    auto U5 = C.subview(0, half_n);
    auto U6 = C.subview(half_n, 0);
    auto U7 = C.subview(half_n, half_n);

    float* arr = new float[8 * half_n * half_n];
    MatrixView S1{arr + 0 * half_n * half_n, half_n};
    MatrixView S2{arr + 1 * half_n * half_n, half_n};
    MatrixView T1{arr + 2 * half_n * half_n, half_n};
    MatrixView T2{arr + 3 * half_n * half_n, half_n};
    MatrixView P1{arr + 4 * half_n * half_n, half_n};
    MatrixView P5{arr + 5 * half_n * half_n, half_n};
    MatrixView U2{arr + 6 * half_n * half_n, half_n};
    MatrixView U3{arr + 7 * half_n * half_n, half_n};

    add(A21, A22, S1, half_n);  // S1 = A21 + A22
    sub(S1, A11, S2, half_n);   // S2 = S1 - A11
    sub(B12, B11, T1, half_n);  // T1 = B12 - B11
    sub(B22, T1, T2, half_n);   // T2 = B22 - T1
    strassen_matmul(A11, B11, P1, half_n);  // P1 = A11 * B11
    strassen_matmul(S1, T1, P5, half_n);    // P5 = S1 * T1

    // U1 = P1 + (A12 * B21)
    strassen_matmul(A12, B21, U1, half_n);
    add(U1, P1, U1, half_n);

    // U2 = P1 + (S2 * T2)
    strassen_matmul(S2, T2, U2, half_n);
    add(U2, P1, U2, half_n);

    // U3 = U2 + (A11 - A21) * (B22 - B12)
    sub(A11, A21, S1, half_n); // reuse S1
    sub(B22, B12, T1, half_n); // reuse T1
    strassen_matmul(S1, T1, U3, half_n);
    add(U3, U2, U3, half_n);

    // U5 = U2 + P5 + (A12 - S2) * B22
    sub(A12, S2, S1, half_n); // reuse S1
    strassen_matmul(S1, B22, U5, half_n);
    add(U5, U2, U5, half_n);
    add(U5, P5, U5, half_n);

    // U6 = U3 - A22 * (T2 - B21)
    sub(T2, B21, T1, half_n); // reuse T1
    strassen_matmul(A22, T1, U6, half_n);
    sub(U3, U6, U6, half_n);

    // U7 = U3 + P5
    add(U3, P5, U7, half_n);

    delete[] arr;
}

#include <print>
#include <random>
#include <chrono>

float A_in[size * size];
float B_in[size * size];
float C_out[size * size];

// Compile with:
// g++ strassen.cpp -Ofast -march=native -o main -lstdc++exp -std=c++23 -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion
// -lstdc++exp -std=c++23: link the experimental standard library for std::println
// -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion: enable more warnings to help write better code (optional)

int main() {
    std::println("Matrix size: {}x{}", size, size);

    MatrixView A{A_in, size};
    MatrixView B{B_in, size};
    MatrixView C{C_out, size};

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
    strassen_matmul(A, B, C, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::println("Strassen: {} ms", elapsed.count());
}