#pragma once

#include "../Network.h"

namespace NeuralNetworks {

Network default_train_network(std::filesystem::path train_data, std::filesystem::path path_to_save,
                              int epochs, BatchSize batch_size, int n_samples);

double test_network(std::filesystem::path parameters, std::filesystem::path test_data,
                    int n_samples);

}  // namespace NeuralNetworks
