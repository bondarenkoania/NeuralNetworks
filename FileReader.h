#pragma once
#include <filesystem>
#include "LinearAlgebra.h"

namespace NeuralNetworks {

struct Data {
    Matrix images;
    Matrix labels;
};

class FileReader {
public:
    explicit FileReader(std::filesystem::path path);
    Data read(Index lines) const;

private:
    std::filesystem::path path_;
    const Index k_num_pixels_ = 784;
    const Index k_labels_size_ = 10;
};

}  // namespace NeuralNetworks
