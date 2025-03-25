#include "Optimizer.h"

namespace NeuralNetworks {

Optimizer::Optimizer() = default;

Optimizer::Optimizer(double a, double beta1, double beta2, double eps)
    : a_(a), beta1_(beta1), beta2_(beta2), eps_(eps) {
}

void Optimizer::update(Matrix& w, Matrix&& grad, Optimizer::AMSGradCache& cache) const {
    if (cache.t == 0) {
        cache.m = Matrix::Zero(w.rows(), w.cols());
        cache.v = Matrix::Zero(w.rows(), w.cols());
        cache.v_hat = Matrix::Zero(w.rows(), w.cols());
    }
    ++cache.t;

    cache.m = beta1_ * cache.m + (1.0 - beta1_) * grad;
    cache.v = beta2_ * cache.v + (1.0 - beta2_) * grad.cwiseAbs2();
    cache.v_hat = cache.v_hat.cwiseMax(cache.v);

    Matrix m_corr = cache.m / (1.0 - pow(beta1_, cache.t));
    Matrix v_hat_corr = cache.v_hat / (1.0 - pow(beta2_, cache.t));

    w -= (a_ * m_corr.array() / (v_hat_corr.array().sqrt() + eps_).array()).matrix();
}

void Optimizer::update(Vector& w, Vector&& grad, Optimizer::AMSGradCache& cache) const {
    if (cache.t == 0) {
        cache.m = Vector::Zero(w.rows());
        cache.v = Vector::Zero(w.rows());
        cache.v_hat = Vector::Zero(w.rows());
    }
    ++cache.t;

    cache.m = beta1_ * cache.m + (1.0 - beta1_) * grad;
    cache.v = beta2_ * cache.v + (1.0 - beta2_) * grad.cwiseAbs2();
    cache.v_hat = cache.v_hat.cwiseMax(cache.v);

    Vector m_corr = cache.m / (1.0 - pow(beta1_, cache.t));
    Vector v_hat_corr = cache.v_hat / (1.0 - pow(beta2_, cache.t));

    w -= (a_ * m_corr.array() / (v_hat_corr.array().sqrt() + eps_).array()).matrix();
}

}  // namespace NeuralNetworks
