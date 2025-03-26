#pragma once

#include "LinearAlgebra.h"

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

    void update(Matrix& w, Matrix&& grad, AMSGradCache& cache) const;
    void update(Vector& w, Vector&& grad, AMSGradCache& cache) const;

private:
    double a_ = 0.001;
    double beta1_ = 0.9;
    double beta2_ = 0.999;
    double eps_ = 1e-8;
};

}  // namespace NeuralNetworks
