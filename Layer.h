#pragma once

#include "ActivationFunction.h"
#include "Random.h"

namespace NeuralNetworks {

enum In : Index;
enum Out : Index;

class Layer {
public:
    Layer(In input_size, Out output_size, ActivationFunction func, Random& rnd);

    Matrix forward(Matrix&& X);
    Matrix backward(Matrix&& U, double learning_rate);
    Matrix predict(Matrix&& X) const;

    struct Cache {
        Matrix input_batch;
        Matrix modified_input_batch;
        Matrix gradA;
        Vector gradb;
    };
    void initCache();
    void resetCache();

private:
    Matrix A_;
    Vector b_;
    ActivationFunction activation_func_;
    std::unique_ptr<Cache> cache_;
};

}  // namespace NeuralNetworks
