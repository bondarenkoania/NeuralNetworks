#include "FileReader.h"
#include <fstream>

namespace NeuralNetworks {

FileReader::FileReader(std::filesystem::path path) : path_(std::move(path)) {
}

Data FileReader::read(Index lines) const {
    std::ifstream file(path_);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file " + path_.string());
    }
    Matrix images(lines, k_num_pixels_);
    Matrix labels = Matrix::Zero(lines, k_labels_size_);

    std::string line;
    for (Index i = 0; i < lines && std::getline(file, line); ++i) {
        std::stringstream ss(std::move(line));
        std::string pix;

        std::getline(ss, pix, ',');
        int label = std::stoi(pix);
        labels(i, label) = 1.0;

        for (Index j = 0; j < k_num_pixels_; ++j) {
            std::getline(ss, pix, ',');
            images(i, j) = std::stod(pix) / 255.0;
        }
    }
    return {images, labels};
}

}  // namespace NeuralNetworks
