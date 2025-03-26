#pragma once

#include "ActivationFunction.h"
#include "Random.h"
#include "Optimizer.h"

namespace NeuralNetworks {

enum In : Index;
enum Out : Index;

class Layer {
public:
    Layer(In input_size, Out output_size, ActivationFunction func, Random& rnd);

    Matrix forward(Matrix&& X);
    Matrix backward(Matrix&& U, Optimizer opt);
    Matrix predict(Matrix&& X) const;

    struct LayerCache {
        Matrix input_batch;
        Matrix modified_input_batch;
    };

    void initCache(Optimizer opt);
    void resetCache();

private:
    Matrix A_;
    Vector b_;
    ActivationFunction activation_func_;
    std::unique_ptr<LayerCache> lcache_;
    std::any opt_cache_A_;
    std::any opt_cache_b_;
};

}  // namespace NeuralNetworks
