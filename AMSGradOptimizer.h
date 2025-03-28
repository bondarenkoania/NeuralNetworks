#pragma once

#include "LinearAlgebra.h"
#include <any>

namespace NeuralNetworks {

class AMSGradOptimizer {
public:
    AMSGradOptimizer() = default;
    AMSGradOptimizer(double a, double beta1, double beta2, double eps);

    struct Cache {
        Matrix m;
        Matrix v;
        Matrix v_hat;
        int t = 0;
    };

    void update(Matrix& w, Matrix&& gradA, std::any& any_cache) const;
    void update(Vector& w, Vector&& gradb, std::any& any_cache) const;
    std::any initCache(const Vector& w) const;
    std::any initCache(const Matrix& w) const;

private:
    double a_ = 0.001;
    double beta1_ = 0.9;
    double beta2_ = 0.999;
    double eps_ = 1e-8;
};

}  // namespace NeuralNetworks
