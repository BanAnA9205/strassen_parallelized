#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>

constexpr size_t blockSize = 480;

struct MatrixView {
    float* where;
    size_t dim;

    MatrixView subview(size_t i, size_t j) const { return MatrixView{where + i * dim + j, dim}; }

    float operator()(size_t i, size_t j) const { return where[i * dim + j]; }
    float& operator()(size_t i, size_t j) { return where[i * dim + j]; }
};

void mul(const MatrixView a, const MatrixView b, MatrixView c, size_t N) {
    for (size_t ii = 0; ii < N; ii += blockSize) {
        for (size_t kk = 0; kk < N; kk += blockSize) {
            for (size_t jj = 0; jj < N; jj += blockSize) {
                float localC[blockSize][blockSize]{};

                for (size_t i = ii; i < std::min(ii + blockSize, N); ++i) {
                    for (size_t k = kk; k < std::min(kk + blockSize, N); ++k) {
                        float a_ik = a(i, k);
                        for (size_t j = jj; j < std::min(jj + blockSize, N); ++j) {
                            localC[i - ii][j - jj] += a_ik * b(k, j);
                        }
                    }
                }

                for (size_t i = ii; i < std::min(ii + blockSize, N); ++i) {
                    for (size_t j = jj; j < std::min(jj + blockSize, N); ++j) {
                        c(i, j) += localC[i - ii][j - jj];
                    }
                }
            }
        }
    }
}


int main(int argc, char** argv) {
    std::cout << "Block Size: " << blockSize << std::endl;
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
    mul({aMem.data(), size}, {bMem.data(), size}, {cMem.data(), size}, size);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    std::cout << "ImprovedMul: " << elapsed.count() << " seconds." << std::endl;
}