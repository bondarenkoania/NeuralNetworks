#include "Dataset.h"
#include "Model.h"
#include <ranges>

#include "NeuralNetworksTypes.h"

namespace NeuralNetworks {

    Network::Network() = default;
    Network::Network(
        const std::vector<std::pair<InputSize, OutputSize>>& layers_sizes, ActivationFunction&& activation_func)
            : activation_func_(std::move(activation_func)) {
        for (auto [input_size, output_size] : layers_sizes) {
            layers.emplace_back(input_size, output_size, urng, &activation_func_);
        }
    }

    Matrix Network::forward(Matrix data) {
        for (Layer& layer : layers) {
            data = layer.forward(data);
        }
        return data;
    }

    void Network::backward(Matrix grad) {
        for (Layer& layer : std::ranges::reverse_view(layers)) {
            grad = layer.backward(grad);
        }
    }

    void Network::train(int epochs, int batch_size, Dataset* dataset, Scheduler* scheduler, LossFunction* loss_func) {
        size_t iterations = (dataset->Size() + batch_size - 1) / batch_size;
        double learning_rate;

        for (int e = 0; e < epochs; ++e) {
            dataset->Shuffle();
            for (int i = 0; i < iterations; ++i) {
                auto [batch, batch_labels] = dataset->GetBatch(batch_size);
                Matrix prediction = this->forward(batch);
                // double loss = loss_func->forward(prediction, batch_labels);
                Matrix loss_gradient = loss_func->backward(prediction, batch_labels);
                this->backward(loss_gradient);

                for (Layer& layer : layers) {
                    learning_rate = scheduler->GetLearningRate(this);
                    layer.update(learning_rate, learning_rate);
                }
            }
        }

    }

    void Network::add_layer(InputSize input_size, OutputSize output_size) {
        layers.emplace_back(input_size, output_size, urng, &activation_func_);
    }

    void Network::reset() {
        layers.clear();
    }

    void Network::set_activation_function(ActivationFunction&& activation_function) {
        activation_func_ = std::move(activation_function);
    }

}
