#pragma once

#include "LinearAlgebra.h"
#include <any>

namespace NeuralNetworks {

class AMSGradOptimizer {
public:
    AMSGradOptimizer();
    AMSGradOptimizer(double a, double beta1, double beta2, double eps);

    struct AMSGradCache {
        Matrix m;
        Matrix v;
        Matrix v_hat;
        int t = 0;
    };

    void update(Matrix& w, Matrix&& grad, std::any& any_cache) const;
    void update(Vector& w, Vector&& grad, std::any& any_cache) const;
    void initCache(std::any& any_cache, const Vector& w) const;
    void initCache(std::any& any_cache, const Matrix& w) const;

private:
    double a_ = 0.001;
    double beta1_ = 0.9;
    double beta2_ = 0.999;
    double eps_ = 1e-8;
};

}  // namespace NeuralNetworks
