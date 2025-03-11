#pragma once

#include "ActivationFunction.h"
#include <EigenRand/EigenRand>

namespace NeuralNetworks {

class Layer {
public:
    Layer(InputSize input_size, OutputSize output_size,
        Eigen::Rand::P8_mt19937_64& urng, ActivationFunction* activation_func);

    Matrix forward(const Matrix& X);
    Matrix backward(Matrix U);
    void update(double learning_rate_A, double learning_rate_b);

private:
    Matrix A;
    Vector b;
    ActivationFunction* activation_func_;

    Matrix input_batch_;
    Matrix modified_input_batch_;

    Matrix gradA;
    Vector gradb;
};

}
