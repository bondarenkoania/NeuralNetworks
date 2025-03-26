#pragma once

#include "Layer.h"
#include "LossFunction.h"
#include "Dataset.h"
#include "Optimizer.h"

#include <vector>

namespace NeuralNetworks {

class Network {
public:
    void train(int epochs, BatchSize batch_size, Optimizer optimizer, const LossFunction& loss_func,
               Dataset& dataset);
    Matrix predict(Matrix&& data) const;

private:
    friend class NetworkBuilder;
    Network();
    Matrix forward(Matrix&& data);
    void backward(Matrix&& grad, Optimizer optimizer);

    std::vector<Layer> layers_;

    void initCache(Optimizer opt);
    void resetCache();
};

}  // namespace NeuralNetworks
