#pragma once

#include "LinearAlgebra.h"
#include <any>

namespace NeuralNetworks {

class SimpleOptimizer {
public:
    SimpleOptimizer() = default;
    SimpleOptimizer(double lr);

    void update(Matrix& w, Matrix&& gradA, std::any& any_cache) const;
    void update(Vector& w, Vector&& gradb, std::any& any_cache) const;
    std::any initCache(const Vector& w) const;
    std::any initCache(const Matrix& w) const;

private:
    double learning_rate_ = 0.001;
};

}  // namespace NeuralNetworks
