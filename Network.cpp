#include "Dataset.h"
#include "Network.h"
#include <ranges>

#include "LinearAlgebra.h"

namespace NeuralNetworks {

Network::Network() = default;

Matrix Network::forward(Matrix&& data) {
    for (Layer& layer : layers_) {
        data = layer.forward(std::move(data));
    }
    return data;
}

void Network::backward(Matrix&& grad, Scheduler scheduler) {
    for (Layer& layer : std::ranges::reverse_view(layers_)) {
        grad = layer.backward(std::move(grad), scheduler.getLearningRate());
    }
}

void Network::train(int epochs, BatchSize batch_size, Scheduler scheduler,
                    const LossFunction& loss_func, Dataset& dataset) {
    initCache();
    for (int e = 0; e < epochs; ++e) {
        dataset.shuffle();
        for (Dataset::BatchRange range = dataset.getBatches(batch_size); Batch batch : range) {
            Matrix prediction = forward(std::move(batch.images));
            Matrix loss_gradient = loss_func.backward(std::move(prediction), batch.labels);
            backward(std::move(loss_gradient), scheduler);
        }
    }
    resetCache();
}

Matrix Network::predict(Matrix&& data) const {
    for (const Layer& layer : layers_) {
        data = layer.predict(std::move(data));
    }
    return data;
}

void Network::initCache() {
    for (Layer& layer : layers_) {
        layer.initCache();
    }
}

void Network::resetCache() {
    for (Layer& layer : layers_) {
        layer.resetCache();
    }
}

}  // namespace NeuralNetworks
