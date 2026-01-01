#include <algorithm>
#include <bit>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <mpi.h>


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

void localStrassen(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
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

void pack(const MatrixView M, float* buffer, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            buffer[i * n + j] = M(i, j);
}

void strassenMPI(const MatrixView A, const MatrixView B, MatrixView C, size_t n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 8) {
        if (rank == 0) {
            std::cout << "Not enough MPI processes for parallel Strassen, running localStrassen on rank 0." << std::endl;
            localStrassen(A, B, C, n);
        }
        return;
    }

    if (n < threshold) {
        if (rank == 0) {
            std::cout << "Matrix size below threshold for Strassen, running mul on rank 0." << std::endl;
            mul(A, B, C, n, n, n);
        }
        return;
    }
    
    if (size > 8) {
        if (rank == 0) std::cout << "More than 8 MPI processes detected. Only 8 will be used for Strassen." << std::endl;
    }

    if (rank == 0) {
        size_t half_n = n / 2;
        int half_n_sq = static_cast<int>(half_n * half_n);
        auto mem{new float[half_n_sq * 14]};
        MatrixView A11{mem, half_n};
        MatrixView B11{mem + half_n_sq, half_n};
        MatrixView A12{mem + half_n_sq * 2, half_n};
        MatrixView B21{mem + half_n_sq * 3, half_n};
        MatrixView S4{mem + half_n_sq * 4, half_n};
        MatrixView B22{mem + half_n_sq * 5, half_n};
        MatrixView A22{mem + half_n_sq * 6, half_n};
        MatrixView T4{mem + half_n_sq * 7, half_n};
        MatrixView S1{mem + half_n_sq * 8, half_n};
        MatrixView T1{mem + half_n_sq * 9, half_n};
        MatrixView S2{mem + half_n_sq * 10, half_n};
        MatrixView T2{mem + half_n_sq * 11, half_n};
        MatrixView S3{mem + half_n_sq * 12, half_n};
        MatrixView T3{mem + half_n_sq * 13, half_n};

        pack(A.subview(0, 0), A11.where, half_n);
        pack(A.subview(0, half_n), A12.where, half_n);
        pack(A.subview(half_n, half_n), A22.where, half_n);
        pack(B.subview(0, 0), B11.where, half_n);
        pack(B.subview(half_n, 0), B21.where, half_n);
        pack(B.subview(half_n, half_n), B22.where, half_n);

        auto A21 = A.subview(half_n, 0);
        auto B12 = B.subview(0, half_n);
        add(A21, A22, S1, half_n);
        sub(S1, A11, S2, half_n);
        sub(A11, A21, S3, half_n);
        sub(A12, S2, S4, half_n);
        sub(B12, B11, T1, half_n);
        sub(B22, T1, T2, half_n);
        sub(B22, B12, T3, half_n);
        sub(T2, B21, T4, half_n);

        MPI_Request requests[7];
        for (int i = 1; i <= 7; ++i) {
            MPI_Isend(mem + half_n_sq * (i - 1) * 2, half_n_sq * 2, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        }
        MPI_Waitall(7, requests, MPI_STATUSES_IGNORE);
        for (int i = 1; i <= 7; ++i) {
            MPI_Irecv(mem + half_n_sq * (i - 1), half_n_sq, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &requests[i - 1]);
        }
        MPI_Waitall(7, requests, MPI_STATUSES_IGNORE);

        MatrixView P1{mem, half_n};
        MatrixView C11{mem + half_n_sq, half_n};
        MatrixView C12{mem + half_n_sq * 2, half_n};
        MatrixView C21{mem + half_n_sq * 3, half_n};
        MatrixView C22{mem + half_n_sq * 4, half_n};
        MatrixView U2{mem + half_n_sq * 5, half_n};
        MatrixView U3{mem + half_n_sq * 6, half_n};

        add(C11, P1, C.subview(0, 0), half_n);
        U2.add(P1, half_n);
        U3.add(U2, half_n);
        C12.add(U2, half_n);
        add(C12, C22, C.subview(0, half_n), half_n);
        sub(U3, C21, C.subview(half_n, 0), half_n);
        add(C22, U3, C.subview(half_n, half_n), half_n);
        delete[] mem;

        if (n % 2 != 0) {
            for (size_t i = 0; i < n - 1; ++i) { // C11 += A12 * B21
                float a_in = A(i, n - 1);
                for (size_t j = 0; j < n - 1; ++j) {
                    C(i, j) += a_in * B(n - 1, j);
                }
            }
            for (size_t i = 0; i < n - 1; ++i) { // C12 = (A11 A12) * (B12 B22)^T
                float sum = 0.0f;
                for (size_t j = 0; j < n; ++j) {
                    sum += A(i, j) * B(j, n - 1);
                }
                C(i, n - 1) = sum;
            }
            for (size_t i = 0; i < n; ++i) { // (C21 C22) = (A21 A22) * B
                float a_in = A(n - 1, i);
                for (size_t j = 0; j < n; ++j) {
                    C(n - 1, j) += a_in * B(i, j);
                }
            }
        }
    } else if (rank <= 7) {
        if (size < 8 || n < threshold) return;
        size_t half_n = n / 2;
        int half_n_sq = static_cast<int>(half_n * half_n);
        auto mem{new float[half_n_sq * 3]};
        MatrixView AA{mem, half_n};
        MatrixView BB{mem + half_n_sq, half_n};
        MatrixView CC{mem + half_n_sq * 2, half_n};
        std::fill_n(CC.where, half_n_sq, 0.0f);
        MPI_Recv(AA.where, half_n_sq * 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localStrassen(AA, BB, CC, half_n);
        MPI_Send(CC.where, half_n_sq, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        delete[] mem;
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "Block Size: " << blockSize << std::endl;
        std::cout << "Threshold: " << threshold << std::endl;
        if (argc < 2) {
            std::cout << "Usage: mpiexec -n <num_processes> ./strassen_mpi <matrix_size>" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    size_t n = std::stoul(argv[1]);
    std::vector<float> aMem;
    std::vector<float> bMem;
    std::vector<float> cMem;

    if (rank == 0) {
        aMem.resize(n * n);
        bMem.resize(n * n);
        cMem.resize(n * n);

        std::mt19937 rng(123);
        std::uniform_real_distribution dist(-1.0f, 1.0f);
        for (size_t i = 0; i < n * n; ++i) {
            aMem[i] = dist(rng);
            bMem[i] = dist(rng);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    strassenMPI({aMem.data(), n}, {bMem.data(), n}, {cMem.data(), n}, n);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "StrassenMPI: " << (t1 - t0) << " seconds." << std::endl;
    }

    MPI_Finalize();
}