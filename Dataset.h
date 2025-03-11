#pragma once

#include <iostream>
#include "NeuralNetworksTypes.h"

namespace NeuralNetworks {

class Dataset {
public:
    Dataset(const std::string& images_path, const std::string& labels_path);

    std::pair<Matrix, Matrix> GetBatch(size_t batch_size);
    void Shuffle();
    size_t Size();

private:
    void LoadImages(const std::string& path);
    void LoadLabels(const std::string& path);

    Matrix images_;
    Matrix labels_;

    size_t current_index_ = 0;
};

}
