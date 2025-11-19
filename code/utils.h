#include <algorithm>
#include <iostream>
#include <random>
#include <iomanip>
#include <chrono>

// Adjustable parameters
constexpr size_t size = 2880;
constexpr size_t block_sz = 192;  // Change this to suit your device

using Matrix_t = float[size][size];

// Check if two matrices match (absolute tolerance; add relative if needed)
bool matrices_equal(const Matrix_t& C1, const Matrix_t& C2, float tol = 1e-4f) {
	// size_t size = std::size(C1);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            if (std::abs(C1[i][j] - C2[i][j]) > tol) {
                return false;
            }
        }
    }
    return true;
}