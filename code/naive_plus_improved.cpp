#include "utils.h"

Matrix_t A_in{}, B_in{}, C_naive{}, C_improved{};

// Compile with:
// g++ code/naive_plus_improved.cpp -Ofast -march=native -o main -lstdc++exp -std=c++23 -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion
// -lstdc++exp -std=c++23: link the experimental standard library for std::println
// -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion: enable more warnings to help write better code (optional)

// compile with this long ass command
// g++ -O3 -fopenmp -march=native -ffast-math -funroll-loops -std=c++23 naive_plus_improved.cpp -o npi_matmul
// OR, consider this (doesnt use openmp)
// g++ -Ofast -march=native -std=c++23 naive_plus_improved.cpp -o npi_matmul

// the most basic matrix multiplication. why did it even exist?
void naive_matmul(const Matrix_t& A, const Matrix_t& B, Matrix_t& C) {
    // size_t size = std::size(A);
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            for (size_t k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Implement matrix multiplication with:
//  1. chunking
//  2. cache-friendly
//  3. vectorization (simd)
//  4. local accumulator
// Note: remember to compile with --fopenmp flag
void improved_matmul(const Matrix_t& A, const Matrix_t& B, Matrix_t& C) {
    // local accumulator
    alignas(64) float C_loc[block_sz][block_sz];

    // super advanced typa shit (it's just blocked matmul)
    for (size_t chunk_iA = 0; chunk_iA < size; chunk_iA += block_sz) {
        size_t max_iA = std::min(chunk_iA + block_sz, size);

        // I HATE INDEXING I HATE INDEXING I HATE INDEXING I HATE INDEXING I HATE INDEXING I HATE INDEXING I HATE INDEXING I HATE INDEXING 
        for (size_t chunk_iB = 0; chunk_iB < size; chunk_iB += block_sz) {
            size_t max_iB = std::min(chunk_iB + block_sz, size);

            std::fill_n(&C_loc[0][0], block_sz * block_sz, 0.0f);

            for (size_t chunk_jA = 0; chunk_jA < size; chunk_jA += block_sz) {
                size_t max_jA = std::min(chunk_jA + block_sz, size);

                for (size_t iA = chunk_iA; iA < max_iA; iA++)
                    for (size_t jA = chunk_jA; jA < max_jA; jA++) {
                        float a = A[iA][jA];

#pragma omp simd
                        for (size_t jB = chunk_iB; jB < max_iB; jB++)
                            C_loc[iA - chunk_iA][jB - chunk_iB] += a * B[jA][jB];
                    }
            }

            for (size_t iA = chunk_iA; iA < max_iA; iA++)
#pragma omp simd
                for (size_t jB = chunk_iB; jB < max_iB; jB++)
                    C[iA][jB] += C_loc[iA - chunk_iA][jB - chunk_iB];
        }
    }
}



//////////////////////////////////////////////////////////////
///// Grok-chan DOES NOT sponsored this testing code UwU /////
//////////////////////////////////////////////////////////////

int main() {
    std::println("Matrix size: {}x{}", size, size);

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

    // Time naive
	auto start_naive = std::chrono::high_resolution_clock::now();
	naive_matmul(A_in, B_in, C_naive);
	auto end_naive = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration_naive = end_naive - start_naive;
    std::println("Naive: {}", duration_naive);

    // Time improved
	auto start_improved = std::chrono::high_resolution_clock::now();
	improved_matmul(A_in, B_in, C_improved);
	auto end_improved = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration_improved = end_improved - start_improved;
    std::println("Improved (bs={}): {}", block_sz, duration_improved);
    std::println("Speedup (bs={}): {:.2f}x", block_sz, duration_naive.count() / duration_improved.count());

    // Check correctness
    if (!matrices_equal(C_naive, C_improved)) {
        std::println("WARNING: Improved (bs={}) does not match naive!", block_sz);
    }

    return 0;
}