#include "Layer.h"
#include <cassert>

namespace NeuralNetworks {

Layer::Layer(In input_size, Out output_size, ActivationFunction func, Random& rnd)
    : A_(rnd.normalMatrix(output_size, input_size)),
      b_(rnd.normalVector(output_size)),
      activation_func_(std::move(func)) {
}

Matrix Layer::forward(Matrix&& X) {
    assert(X.rows() == A_.cols() && "Incorrect size of input vectors in forward.");
    assert(lcache_ != nullptr && "Uninitialized layer cache during training in forward.");

    lcache_->input_batch = std::move(X);
    lcache_->modified_input_batch = A_ * lcache_->input_batch;
    lcache_->modified_input_batch.colwise() += b_;

    Matrix result = lcache_->modified_input_batch;
    for (Index i = 0; i < result.cols(); ++i) {
        result.col(i) = activation_func_.apply(result.col(i));
    }
    return result;
}

Matrix Layer::backward(Matrix&& U, Optimizer opt) {
    assert((U.cols() == A_.rows()) && "Incorrect size of input rows in backward.");
    assert((U.rows() == lcache_->input_batch.cols()) && "Incorrect batch size in backward.");
    assert(lcache_ != nullptr && "Uninitialized layer cache during training in backward.");

    Index batch_size = U.rows();
    for (Index i = 0; i < batch_size; ++i) {
        U.row(i) *= activation_func_.derivative(lcache_->modified_input_batch.col(i));
    }
    Vector gradb = U.transpose().rowwise().mean();
    Matrix gradA = U.transpose() * lcache_->input_batch.transpose() / batch_size;

    Matrix result = U * A_;

    opt->update(A_, std::move(gradA), opt_cache_A_);
    opt->update(b_, std::move(gradb), opt_cache_b_);

    return result;
}

Matrix Layer::predict(Matrix&& X) const {
    assert((X.rows() == A_.cols()) && "Incorrect size of input vectors in predict.");

    X = A_ * X;
    for (Index i = 0; i < X.cols(); ++i) {
        X.col(i) = activation_func_.apply(X.col(i));
    }
    return X;
}

void Layer::initCache(Optimizer opt) {
    lcache_ = std::make_unique<LayerCache>();
    opt_cache_b_ = opt->initCache(b_);
    opt_cache_A_ = opt->initCache(A_);
}

void Layer::resetCache() {
    lcache_.reset();
    opt_cache_b_.reset();
    opt_cache_A_.reset();
}

}  // namespace NeuralNetworks
