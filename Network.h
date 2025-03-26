#pragma once

#include "Layer.h"
#include "LossFunction.h"
#include "Scheduler.h"
#include "Dataset.h"

#include <vector>

namespace NeuralNetworks {

class Network {
public:
    void train(int epochs, BatchSize batch_size, AMSGradOptimizer optimizer,
               const LossFunction& loss_func, Dataset& dataset);
    Matrix predict(Matrix&& data) const;

private:
    friend class NetworkBuilder;
    Network();
    Matrix forward(Matrix&& data);
    void backward(Matrix&& grad, AMSGradOptimizer optimizer);

    std::vector<Layer> layers_;

    void initCache();
    void resetCache();
};

}  // namespace NeuralNetworks
