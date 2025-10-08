#ifndef MATRIX_IO_H
#define MATRIX_IO_H
#include <string_view>
#include <fstream>
#include <cstdint>
#include <random>
#include <iostream>
#include <string>
#include <array>
#include <tuple>
#define NDEBUG

template<typename T, size_t N>
using Matrix = std::array<std::array<T, N>, N>;

// Write dim * dim random floats in the range [lower, upper) to a binary file
void writeRandomMatrixToBinaryFile(std::string_view filename, uint32_t dim, float lower = -10.0f, float upper = 10.0f) {
	std::ofstream outFile(filename.data(), std::ios::binary);
	if (!outFile) throw std::ios_base::failure("Failed to open file for writing.");

	outFile.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

	static std::mt19937 gen(std::random_device{}());
	static std::uniform_real_distribution<float> dist(lower, upper);
	for (uint32_t i = 0; i < dim; ++i) {
		for (uint32_t j = 0; j < dim; ++j) {
			float value = dist(gen);
#ifndef NDEBUG
			std::cout << value << ' ';
#endif
			outFile.write(reinterpret_cast<const char*>(&value), sizeof(value));
		}
#ifndef NDEBUG
		std::cout << std::endl;
#endif
	}

	std::cout << "Matrix written to " << filename << std::endl;
}

// Read a matrix from a binary file and return it
template <uint32_t dim>
auto readMatrixFromBinaryFile(std::string_view filename) {
	std::ifstream inFile(filename.data(), std::ios::binary);
	if (!inFile) throw std::ios_base::failure("Failed to open file for reading.");

	uint32_t in_dim;
	inFile.read(reinterpret_cast<char*>(&in_dim), sizeof(in_dim));
	if (dim != in_dim) {
		throw std::runtime_error("Dimension mismatch: expected " + std::to_string(dim) + ", got " + std::to_string(in_dim));
	}

	Matrix<float, dim> matrix;
	inFile.read(reinterpret_cast<char*>(&matrix[0][0]), static_cast<std::streamsize>(dim * dim * sizeof(float)));

	return matrix;
}



//// Example usage
//#include <print>
//
//int main() {
//	const char* filename = "matrix.bin";
//	constexpr uint32_t dim = 4; // dim must be known at compile time for readMatrixFromBinaryFile
//	writeRandomMatrixToBinaryFile(filename, dim);
//	Matrix<float, dim> matrix{ readMatrixFromBinaryFile<dim>(filename) };
//	for (const auto& row : matrix) {
//		std::println("{}", row);
//	}
//}

#endif // MATRIX_IO_H

