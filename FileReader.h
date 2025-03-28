#pragma once
#include <filesystem>
#include <fstream>
#include "LinearAlgebra.h"

namespace NeuralNetworks {

struct Data {
    Matrix images;
    Matrix labels;
};

class FileReader {
public:
    explicit FileReader(std::filesystem::path path);
    std::optional<Data> read(Index lines) noexcept;
    bool isOpen() const;

private:
    Data read_helper(Index lines);

    static constexpr Index k_num_pixels_ = 784;
    static constexpr Index k_labels_size_ = 10;

    std::ifstream file_;
};

}  // namespace NeuralNetworks
