#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <omp.h> 
#include <functional>  
#include <numeric>

// compile with this long ass command
// g++ -O3 -fopenmp -march=native -ffast-math -funroll-loops -std=c++23 naive_plus_improved.cpp -o npi_matmul

// the most basic matrix multiplication. why did it even exist?
void naive_matmul(int size, const float *A, const float *B, float *C){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            for (int k = 0; k < size; k++){
                C[i * size + j] += A[i * size + k] * B[k * size + j];
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
void improved_matmul(const int size, 
                     const float *A, const float *B, float *C, 
                     const int block_sz = 160){

    if (size <= 0 || !A || !B || !C || block_sz <= 0) throw std::invalid_argument("bad args");   
    
    // local accumulator
    std::vector<float> C_loc(block_sz * block_sz);
    
    // super advanced typa shit (it's just blocked matmul)
    for (int chunk_iA = 0; chunk_iA < size; chunk_iA += block_sz){
        int max_iA = std::min(chunk_iA + block_sz, size);
        int sz_iA = max_iA - chunk_iA;

        // I HATE INDEXING I HATE INDEXING I HATE INDEXING I HATE INDEXING I HATE INDEXING I HATE INDEXING I HATE INDEXING I HATE INDEXING 
        for (int chunk_iB = 0; chunk_iB < size; chunk_iB += block_sz){
            int max_iB = std::min(chunk_iB + block_sz, size);   
            int sz_iB = max_iB - chunk_iB;

            for (int Ci_loc = 0; Ci_loc < sz_iA; Ci_loc++)
                #pragma omp simd
                for (int Cj_loc = 0; Cj_loc < sz_iB; Cj_loc++)
                    C_loc[Ci_loc * block_sz + Cj_loc] = 0.0;

            for (int chunk_jA = 0; chunk_jA < size; chunk_jA += block_sz){
                int max_jA = std::min(chunk_jA + block_sz, size);

                for (int iA = 0; iA < sz_iA; iA++)
                    for (int jA = chunk_jA; jA < max_jA; jA++){
                        float a = A[(iA + chunk_iA) * size + jA];
                        
                        #pragma omp simd
                        for (int jB = chunk_iB; jB < max_iB; jB++)
                            C_loc[iA * block_sz + jB - chunk_iB] += a * B[jA * size + jB];
                    }
            }

            for (int iA = chunk_iA; iA < max_iA; iA++)
                #pragma omp simd
                for (int jB = chunk_iB; jB < max_iB; jB++)
                    C[iA * size + jB] += C_loc[(iA - chunk_iA) * block_sz + jB - chunk_iB];
        }
    }
}



/////////////////////////////////////////////////////
///// Grok-chan sponsored this testing code UwU /////
/////////////////////////////////////////////////////

// Check if two matrices match (absolute tolerance; add relative if needed)
bool matrices_equal(const std::vector<float>& C1, const std::vector<float>& C2, int size, float tol = 1e-4f) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float diff = std::abs(C1[i * size + j] - C2[i * size + j]);
            if (diff > tol) {
                return false;
            }
        }
    }
    return true;
}

// Time a function multiple times and compute mean/std dev (in ms); uses C++23 reductions for conciseness
struct TimingResult {
    double mean_ms;
    double std_ms;
};

TimingResult time_function(std::function<void()> func, int runs) {
    std::vector<double> times_ms(runs);
    for (int r = 0; r < runs; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        times_ms[r] = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Mean: reduce with + (binary_op implicit)
    double mean = std::reduce(times_ms.begin(), times_ms.end()) / runs;
    
    // Variance: transform to squared diffs (unary_op), reduce with + (binary_op std::plus)
    double sum_sq_diff = std::transform_reduce(times_ms.begin(), times_ms.end(), 0.0,
                                               std::plus<>{},  // binary_op: sum transformed values
                                               [mean](double t) { return (t - mean) * (t - mean); });  // unary_op: square diff
    double variance = sum_sq_diff / (runs - 1);
    double std = std::sqrt(variance);
    
    return {mean, std};
}

int main() {
    // Adjustable parameters
    const int size = 1440;
    const int naive_runs = 3;
    const int improved_runs = 20;
    const int block_sz = 160;  // Change this to suit your device

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Matrix size: " << size << "x" << size 
              << " (bs=" << block_sz << ", naive_runs=" << naive_runs 
              << ", improved_runs=" << improved_runs << ")" << std::endl << std::endl;
    
    // Allocate and fill matrices (inline; uniform [-1,1] random)
    std::vector<float> A(size * size, 0.0f);
    std::vector<float> B(size * size, 0.0f);
    std::vector<float> C_naive(size * size, 0.0f);
    std::vector<float> C_improved(size * size, 0.0f);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            A[i * size + j] = dis(gen);
            B[i * size + j] = dis(gen);
        }
    }
    
    const float* A_ptr = A.data();
    const float* B_ptr = B.data();
    float* C_n = C_naive.data();
    float* C_i = C_improved.data();
    
    // Time naive (zero C inside lambda)
    auto naive_func = [&]() {
        std::fill(C_n, C_n + size * size, 0.0f);
        naive_matmul(size, A_ptr, B_ptr, C_n);
    };
    TimingResult naive_res = time_function(naive_func, naive_runs);
    std::cout << "Naive: " << naive_res.mean_ms << " ± " << naive_res.std_ms << " ms" << std::endl;
    
    // Update C_naive for reference
    naive_func();
    
    // Time improved (zero C inside lambda)
    auto improved_func = [&]() {
        std::fill(C_i, C_i + size * size, 0.0f);
        improved_matmul(size, A_ptr, B_ptr, C_i, block_sz);
    };
    TimingResult improved_res = time_function(improved_func, improved_runs);
    
    // Check correctness
    bool correct = matrices_equal(C_naive, C_improved, size);
    if (!correct) {
        std::cout << "WARNING: Improved (bs=" << block_sz << ") does not match naive!" << std::endl;
    } else {
        std::cout << "Improved (bs=" << block_sz << "): " << improved_res.mean_ms 
                  << " ± " << improved_res.std_ms << " ms" << std::endl;
        double speedup = naive_res.mean_ms / improved_res.mean_ms;
        std::cout << "Speedup: " << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "\nAll improved versions match naive (correct)." << std::endl;
    }
    
    return 0;
}
