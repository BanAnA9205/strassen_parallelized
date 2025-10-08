#include <string_view>
#include <fstream>
#include <cstdint>
#include <random>
#include <iostream>
#include <vector>
#include <tuple>

// Write n_row * n_col random floats in the range [lower, upper) to a binary file
void writeRandomMatrixToBinaryFile(std::string_view filename, uint32_t n_row, uint32_t n_col, float lower, float upper) {
	std::ofstream outFile(filename.data(), std::ios::binary);
	if (!outFile) throw std::ios_base::failure("Failed to open file for writing.");

	outFile.write(reinterpret_cast<const char*>(&n_row), sizeof(n_row));
	outFile.write(reinterpret_cast<const char*>(&n_col), sizeof(n_col));

	static std::mt19937 gen(std::random_device{}());
	static std::uniform_real_distribution<float> dist(lower, upper);
	for (uint32_t i = 0; i < n_row * n_col; ++i) {
		float value = dist(gen);
		outFile.write(reinterpret_cast<const char*>(&value), sizeof(value));
	}

	std::cout << "Matrix written to " << filename << std::endl;
}

// Write dim * dim random floats in the range [-10.0f, 10.0f) to a binary file
void writeMatrixToBinaryFile(std::string_view filename, uint32_t dim) {
	writeRandomMatrixToBinaryFile(filename, dim, dim, -10.0f, 10.0f);
}

// Read a matrix from a binary file and return it as a vector along with its dimensions
// The returned tuple contains (matrix, n_row, n_col)
auto readMatrixFromBinaryFile(std::string_view filename) {
	std::ifstream inFile(filename.data(), std::ios::binary);
	if (!inFile) throw std::ios_base::failure("Failed to open file for reading.");

	uint32_t n_row, n_col;
	inFile.read(reinterpret_cast<char*>(&n_row), sizeof(n_row));
	inFile.read(reinterpret_cast<char*>(&n_col), sizeof(n_col));

	std::vector<float> matrix(n_row * n_col);
	inFile.read(reinterpret_cast<char*>(matrix.data()), n_row * n_col * sizeof(float));

	return std::make_tuple(matrix, n_row, n_col);
}



//// Example usage
//
//#include <string>
//
//int main() {
//	for (uint32_t dim = 100; dim < 1000; dim += 100) {
//		std::string filename = "matrix_" + std::to_string(dim) + "x" + std::to_string(dim) + ".bin";
//		writeMatrixToBinaryFile(filename, dim);
//		auto [matrix, n_row, n_col] = readMatrixFromBinaryFile(filename);
//		std::cout << "Read matrix of size " << n_row << "x" << n_col << " from " << filename << std::endl;
//	}
//}