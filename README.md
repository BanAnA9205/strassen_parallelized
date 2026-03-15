# Parallelized Strassen's Algorithm for Matrix Multiplication

This repository contains high-performance C++ implementations of **Strassen’s Algorithm** for matrix multiplication, optimized for both shared-memory (OpenMP) and distributed-memory (MPI) systems.

The project explores the transition from a naive $O(n^3)$ approach to a sub-cubic $O(n^{2.81})$ Strassen-Winograd implementation, culminating in a Hybrid MPI+OpenMP solution capable of running on computer clusters.

**Authors:** Nguyen Van An, Le Huu Trieu

**Course:** CO3067 - Parallel Computing (HCMUT)

## 🚀 Features

* **Sequential Optimizations:**
    * **Naive:** Standard triple-loop multiplication.
    * **Improved:** Cache-friendly Loop Tiling (Blocking) and Auto-vectorization.
    * **Strassen Serial:** Implementation of the **Strassen-Winograd** variant using **Boyer’s Memory-Efficient Schedule** (requires only 2 temporary matrices instead of the standard massive allocation).
* **Parallel Implementations:**
    * **OpenMP:** Task-based parallelism (`#pragma omp task`) for recursive matrix operations.
    * **MPI:** Distributed memory implementation using a Master-Worker model (parallelizes the top-level recursion).
    * **Hybrid (MPI + OpenMP):** Combines inter-node distribution (MPI) with intra-node multi-threading (OpenMP) for maximum cluster utilization.

## 📂 Project Structure

```text
├── code/
│   ├── naive.cpp          # Standard O(n^3) implementation
│   ├── improved.cpp       # Tiling, Local Accumulation, Auto-vectorization
│   ├── strass_serial.cpp  # Memory-efficient Strassen-Winograd
│   ├── omp.cpp            # Shared-memory parallelization (OpenMP)
│   ├── mpi.cpp            # Distributed-memory parallelization (MPI)
│   └── ompi.cpp           # Hybrid MPI + OpenMP implementation
├── texts/                 # Project report and references
└── README.md
```

## 🛠️ Prerequisites

* **C++ Compiler:** GCC recommended (requires C++20 or later).
* **OpenMP:** Usually included with GCC (`libgomp`).
* **MPI:** OpenMPI or MPICH implementation.

## 🔨 Build Instructions

Use the following flags to enable aggressive compiler optimizations (`-Ofast`, `-march=native`) which are critical for vectorization performance.

### Sequential & OpenMP
```bash
# Naive
g++ -std=c++20 -Ofast -march=native code/naive.cpp -o naive

# Improved (Tiled)
g++ -std=c++20 -Ofast -march=native code/improved.cpp -o improved

# Strassen Serial
g++ -std=c++20 -Ofast -march=native code/strass_serial.cpp -o strass_serial

# Strassen OpenMP
g++ -std=c++20 -Ofast -march=native -fopenmp code/omp.cpp -o omp
```

### MPI & Hybrid
```bash
# Strassen MPI
mpic++ -std=c++20 -Ofast -march=native code/mpi.cpp -o mpi

# Strassen Hybrid (MPI + OpenMP)
mpic++ -std=c++20 -Ofast -march=native -fopenmp code/ompi.cpp -o ompi
```

## ⚡ Usage

All executables take the matrix size $N$ as a command-line argument. Matrices are initialized with random values in $[-1, 1]$.

**Sequential / OpenMP:**
```bash
./improved 4096
./omp 4096
```

**MPI / Hybrid:**
Run with `mpiexec` or `mpirun`. The implementation requires **at least 8 processes** to distribute the 7 Strassen sub-tasks effectively.
```bash
mpiexec -n 8 ./mpi 4096
mpiexec -n 8 ./ompi 4096
```

## 📊 Technical Details

### 1. Memory Management
Standard Strassen implementations suffer from memory explosion due to temporary matrices at each recursion level. We implemented the **Boyer et al. schedule**, which reduces the auxiliary memory requirement to just **2 temporary blocks** by reusing memory and overwriting intermediates as soon as they are consumed.

### 2. Parallel Strategy
* **OpenMP:** Uses `task` directives to parallelize the 7 recursive calls. A `threshold` constant (default: 480) switches to the standard blocked multiplication at the leaves of the recursion tree to reduce overhead.
* **MPI:** The Master process pre-computes the initial additions ($S_1 \dots T_4$) and scatters data to 7 worker processes. Due to high communication costs, only the **first level** of recursion is distributed across nodes; deeper levels are handled locally (or via OpenMP in the hybrid version).

## 📈 Performance

### Shared-Memory Performance

| Size (N) | Naive (ms) | Improved (ms) | Strassen Serial (ms) | Strassen OMP (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **2,000** | 6744.61 | 342.97 | 354.29 | 108.52 |
| **4,000** | 136691.00 | 2880.57 | 2492.40 | 831.70 |
| **6,000** | 552019.00 | 10092.24 | 7720.80 | 2301.73 |
| **8,000** | TLE | 21924.07 | 16717.17 | 4976.20 |
| **10,000** | TLE | 43913.83 | 32277.63 | 9914.28 |

### Distributed-Memory Performance

| Size (N) | Strassen MPI (ms) | Strassen Hybrid (ms) |
| :--- | :--- | :--- |
| **2,000** | 1956.90 | 911.14 |
| **4,000** | 11193.70 | 3878.87 |
| **6,000** | 35789.20 | 10380.90 |
| **8,000** | 77665.80 | 21872.30 |
| **10,000** | 52110.20 | 19652.00 |

*Note: While significantly faster than naive approaches, this implementation is generally slower than highly-tuned libraries like OpenBLAS, which utilize assembly-level optimizations.*

## 📚 References

1.  **Volker Strassen**, "Gaussian elimination is not optimal", *Numerische Mathematik*, 1969.
2.  **Boyer et al.**, "Memory efficient scheduling of Strassen-Winograd's matrix multiplication algorithm", *arXiv:0707.2347*, 2009.
