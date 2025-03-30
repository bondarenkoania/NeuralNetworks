#include "FileReader.h"

namespace NeuralNetworks {

FileReader::FileReader(std::filesystem::path path) : file_(path) {
}

std::optional<Data> FileReader::read(Index lines) noexcept {
    try {
        return read_helper(lines);
    } catch (...) {
        return std::nullopt;
    }
}

bool FileReader::isOpen() const {
    return file_.is_open();
}

Data FileReader::read_helper(Index lines) {
    Matrix images(lines, k_num_pixels_);
    Matrix labels = Matrix::Zero(lines, k_labels_size_);

    std::string line;
    for (Index i = 0; i < lines && std::getline(file_, line); ++i) {
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
