#pragma once

#include "Layer.h"
#include "LossFunction.h"
#include "Scheduler.h"
#include "Dataset.h"

#include <vector>

namespace NeuralNetworks {

class Network {
public:
    Network();
    explicit Network(
        const std::vector<std::pair<InputSize, OutputSize>>& layers_sizes,
        ActivationFunction&& activation_func = ActivationFunction(ReLU, ReLU_Der));

    void train(int epochs, int batch_size, Dataset* dataset, Scheduler* scheduler, LossFunction* loss_func);
    void add_layer(InputSize input_size, OutputSize output_size);
    void set_activation_function(ActivationFunction&& activation_function);
    void reset();

private:
    Matrix forward(Matrix data);
    void backward(Matrix grad);

    ActivationFunction activation_func_;

    std::vector<Layer> layers;
    Eigen::Rand::P8_mt19937_64 urng{42};
};

}
