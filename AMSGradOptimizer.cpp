#include "AMSGradOptimizer.h"

namespace NeuralNetworks {

AMSGradOptimizer::AMSGradOptimizer(double a, double beta1, double beta2, double eps)
    : a_(a), beta1_(beta1), beta2_(beta2), eps_(eps) {
}

void AMSGradOptimizer::update(Matrix& w, Matrix&& gradA, std::any& any_cache) const {
    auto& cache = std::any_cast<Cache&>(any_cache);
    assert(cache.m.rows() != 0 && "Uninitialized AMSGrad cache during training.");

    ++cache.t;
    cache.m = beta1_ * cache.m + (1.0 - beta1_) * gradA;
    cache.v = beta2_ * cache.v + (1.0 - beta2_) * gradA.cwiseAbs2();
    cache.v_hat = cache.v_hat.cwiseMax(cache.v);

    Matrix m_corr = cache.m / (1.0 - pow(beta1_, cache.t));
    Matrix v_hat_corr = cache.v_hat / (1.0 - pow(beta2_, cache.t));

    w -= (a_ * m_corr.array() / (v_hat_corr.array().sqrt() + eps_).array()).matrix();
}

void AMSGradOptimizer::update(Vector& w, Vector&& gradb, std::any& any_cache) const {
    auto& cache = std::any_cast<Cache&>(any_cache);
    assert(cache.m.rows() != 0 && "Uninitialized AMSGrad cache during training.");

    ++cache.t;
    cache.m = beta1_ * cache.m + (1.0 - beta1_) * gradb;
    cache.v = beta2_ * cache.v + (1.0 - beta2_) * gradb.cwiseAbs2();
    cache.v_hat = cache.v_hat.cwiseMax(cache.v);

    Vector m_corr = cache.m / (1.0 - pow(beta1_, cache.t));
    Vector v_hat_corr = cache.v_hat / (1.0 - pow(beta2_, cache.t));

    w -= (a_ * m_corr.array() / (v_hat_corr.array().sqrt() + eps_).array()).matrix();
}

std::any AMSGradOptimizer::initCache(const Vector& w) const {
    return Cache{
        .m = Vector::Zero(w.rows()), .v = Vector::Zero(w.rows()), .v_hat = Vector::Zero(w.rows())};
}

std::any AMSGradOptimizer::initCache(const Matrix& w) const {
    return Cache{.m = Matrix::Zero(w.rows(), w.cols()),
                 .v = Matrix::Zero(w.rows(), w.cols()),
                 .v_hat = Matrix::Zero(w.rows(), w.cols())};
}

}  // namespace NeuralNetworks
