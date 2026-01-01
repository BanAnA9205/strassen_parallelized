#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>

struct MatrixView {
    float* where;
    size_t dim;

    MatrixView subview(size_t i, size_t j) const { return MatrixView{where + i * dim + j, dim}; }

    float operator()(size_t i, size_t j) const { return where[i * dim + j]; }
    float& operator()(size_t i, size_t j) { return where[i * dim + j]; }
};

void mul(const MatrixView a, const MatrixView b, MatrixView c, size_t N) {
    for (size_t ii = 0; ii < N; ii++) {
        for (size_t jj = 0; jj < N; jj++) {
            for (size_t kk = 0; kk < N; kk++) {
                c(ii, jj) += a(ii, kk) * b(kk, jj);
            }
        }
    }
}


int main(int argc, char** argv) {
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
    std::cout << "NaiveMul: " << elapsed.count() << " seconds." << std::endl;
}