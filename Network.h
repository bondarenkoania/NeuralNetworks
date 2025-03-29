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
    Matrix predict(const Matrix& data) const;

private:
    class SwitchGuard {
    public:
        SwitchGuard(Network* network, Optimizer optimizer);
        ~SwitchGuard();

    private:
        Network* network_;
    };

    friend class NetworkBuilder;
    friend class Saver;

    Network() = default;
    Matrix forward(Matrix&& data);
    void backward(Matrix&& grad, Optimizer optimizer);

    void initCache(Optimizer opt);
    void resetCache();

    std::vector<Layer> layers_;
};

}  // namespace NeuralNetworks
