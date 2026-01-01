#include <algorithm>
#include <bit>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>

constexpr size_t blockSize = 480;
constexpr size_t threshold = 480;

struct MatrixView {
    float* where;
    size_t dim;

    MatrixView subview(size_t i, size_t j) const { return MatrixView{where + i * dim + j, dim}; }

    void add(const MatrixView other, size_t n) {
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                (*this)(i, j) += other(i, j);
    }

    void sub(const MatrixView other, size_t n) {
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                (*this)(i, j) -= other(i, j);
    }

    void clear(size_t n) {
        for (size_t i = 0; i < n; ++i)
            std::fill_n(where + i * dim, n, 0.0f);
    }

    float operator()(size_t i, size_t j) const { return where[i * dim + j]; }
    float& operator()(size_t i, size_t j) { return where[i * dim + j]; }
};

void add(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            C(i, j) = A(i, j) + B(i, j);
}

void sub(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            C(i, j) = A(i, j) - B(i, j);
}

void mul(const MatrixView a, const MatrixView b, MatrixView c, size_t rangeI, size_t rangeJ, size_t rangeK) {
    for (size_t ii = 0; ii < rangeI; ii += blockSize) {
        for (size_t kk = 0; kk < rangeK; kk += blockSize) {
            for (size_t jj = 0; jj < rangeJ; jj += blockSize) {
                float localC[blockSize][blockSize]{};

                for (size_t i = ii; i < std::min(ii + blockSize, rangeI); ++i) {
                    for (size_t k = kk; k < std::min(kk + blockSize, rangeK); ++k) {
                        float a_ik = a(i, k);
                        for (size_t j = jj; j < std::min(jj + blockSize, rangeJ); ++j) {
                            localC[i - ii][j - jj] += a_ik * b(k, j);
                        }
                    }
                }

                for (size_t i = ii; i < std::min(ii + blockSize, rangeI); ++i) {
                    for (size_t j = jj; j < std::min(jj + blockSize, rangeJ); ++j) {
                        c(i, j) += localC[i - ii][j - jj];
                    }
                }
            }
        }
    }
}

void _strassen(const MatrixView A, const MatrixView B, MatrixView C, size_t n, float* memory) {
    if (n < threshold) { mul(A, B, C, n, n, n); return; }

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
    MatrixView X{memory, half_n};
    memory += half_n_sq;
    MatrixView Y{memory, half_n};
    memory += half_n_sq;

    sub(A11, A21, X, half_n);
    sub(B22, B12, Y, half_n);
    _strassen(X, Y, C21, half_n, memory);
    add(A21, A22, X, half_n);
    sub(B12, B11, Y, half_n);
    _strassen(X, Y, C22, half_n, memory);
    X.sub(A11, half_n);
    sub(B22, Y, Y, half_n);
    _strassen(X, Y, C12, half_n, memory);
    sub(A12, X, X, half_n);
    _strassen(X, B22, C11, half_n, memory);
    std::fill_n(&X.where[0], half_n_sq, 0.0f);
    _strassen(A11, B11, X, half_n, memory);
    C12.add(X, half_n);
    C21.add(C12, half_n);
    C12.add(C22, half_n);
    C22.add(C21, half_n);
    C12.add(C11, half_n);
    Y.sub(B21, half_n);
    C11.clear(half_n);
    _strassen(A22, Y, C11, half_n, memory);
    C21.sub(C11, half_n);
    C11.clear(half_n);
    _strassen(A12, B21, C11, half_n, memory);
    C11.add(X, half_n);
}

// Find k, r  s.t.  n - k = l * 2^r,  where  l < blockSize
auto getPeelSize(size_t n) {
    int r = static_cast<int>(std::bit_width(n) - std::bit_width(threshold)) + 1;
    size_t mask = (1ull << r) - 1ull;
    size_t k = n & mask;
    return std::make_pair(k , r);
}

// memory = (size/2)^2*2 + (size/4)^2*2 + ... + (size / 2^(numLevels-1))^2*2 = (size / 2^numLevels)^2 * (4^numLevels - 1) * 2 / 3
size_t getMemorySize(size_t size, int numLevels) {
    size_t m = size >> numLevels;
    return m * m * ((1ull << (numLevels * 2)) - 1ull) * 2 / 3;
}

void strassen(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    auto [k, r] = getPeelSize(n);
    size_t m = n - k;
    auto memBuf{new float[getMemorySize(n, r)]};
    _strassen(A, B, C, m, memBuf);
    delete[] memBuf;

    if (k == 0) return;
    mul(A.subview(0, m), B.subview(m, 0), C, m, m, k); // C11 += A12 * B21
    mul(A, B.subview(0, m), C.subview(0, m), m, k, n); // C12 = (A11 A12) * (B12 B22)^T
    mul(A.subview(m, 0), B, C.subview(m, 0), k, n, n); // (C21 C22) = (A21 A22) * B
}


int main(int argc, char** argv) {
    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "Threshold: " << threshold << std::endl;
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
        return 1;
    }
    size_t size = std::stoul(argv[1]);

    std::vector<float> aMem(size * size);
    std::vector<float> bMem(size * size);
    std::vector<float> cMem(size * size, 0.0f);

    std::mt19937 rng(123);
    std::uniform_real_distribution dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size * size; ++i) {
        aMem[i] = dist(rng);
        bMem[i] = dist(rng);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    strassen({aMem.data(), size}, {bMem.data(), size}, {cMem.data(), size}, size);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    std::cout << "Strassen: " << elapsed.count() << " seconds." << std::endl;
}