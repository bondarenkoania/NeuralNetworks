#include "Dataset.h"
#include "Network.h"
#include <ranges>

#include "LinearAlgebra.h"

namespace NeuralNetworks {

void Network::train(int epochs, BatchSize batch_size, Optimizer optimizer,
                    const LossFunction& loss_func, Dataset& dataset) {
    SwitchGuard to_train_mode(this, optimizer);
    for (int e = 0; e < epochs; ++e) {
        dataset.shuffle();
        for (Dataset::BatchRange range = dataset.batches(batch_size);
             Dataset::Batch batch : range) {
            Matrix prediction = forward(std::move(batch.images));
            Matrix loss_gradient = loss_func.derivative(std::move(prediction), batch.labels);
            backward(std::move(loss_gradient), optimizer);
        }
    }
}

Matrix Network::predict(Matrix&& data) const {
    for (const Layer& layer : layers_) {
        data = layer.predict(std::move(data));
    }
    return data;
}

Matrix Network::predict(const Matrix& data) const {
    return predict(std::move(Matrix(data)));
}

Network::SwitchGuard::SwitchGuard(Network* network, Optimizer optimizer) : network_(network) {
    network_->initCache(optimizer);
}

Network::SwitchGuard::~SwitchGuard() {
    network_->resetCache();
}

Matrix Network::forward(Matrix&& data) {
    for (Layer& layer : layers_) {
        data = layer.forward(std::move(data));
    }
    return data;
}

void Network::backward(Matrix&& grad, Optimizer optimizer) {
    for (Layer& layer : std::ranges::reverse_view(layers_)) {
        grad = layer.backward(std::move(grad), optimizer);
    }
}

void Network::initCache(Optimizer opt) {
    for (Layer& layer : layers_) {
        layer.initCache(opt);
    }
}

void Network::resetCache() {
    for (Layer& layer : layers_) {
        layer.resetCache();
    }
}

}  // namespace NeuralNetworks
