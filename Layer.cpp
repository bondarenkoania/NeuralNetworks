#include "Layer.h"

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>

namespace NeuralNetworks {

Layer::Layer(In input_size, Out output_size, ActivationFunction func, Random& rnd)
    : A_(rnd.normalMatrix(output_size, input_size)),
      b_(rnd.normalVector(output_size)),
      activation_func_(std::move(func)) {
}

Matrix Layer::forward(Matrix&& X) {
    assert(X.rows() == A_.cols() && "Incorrect size of input vectors in forward.");
    assert(cache_ != nullptr && "Uninitialized cache during training in forward.");

    cache_->input_batch = std::move(X);
    cache_->modified_input_batch = A_ * cache_->input_batch;
    cache_->modified_input_batch.colwise() += b_;

    Matrix result = cache_->modified_input_batch;
    for (Index i = 0; i < result.cols(); ++i) {
        result.col(i) = activation_func_.apply(result.col(i));
    }
    return result;
}

Matrix Layer::backward(Matrix&& U, double learning_rate) {
    assert((U.cols() == A_.rows()) && "Incorrect size of input rows in backward.");
    assert((U.rows() == cache_->input_batch.cols()) && "Incorrect batch size in backward.");
    assert(cache_ != nullptr && "Uninitialized cache during training in backward.");

    Index batch_size = U.rows();
    for (Index i = 0; i < batch_size; ++i) {
        U.row(i) *= activation_func_.derivative(cache_->modified_input_batch.col(i));
    }
    cache_->gradb = U.transpose().rowwise().mean();
    cache_->gradA = U.transpose() * cache_->input_batch.transpose() / batch_size;

    A_ -= cache_->gradA * learning_rate;
    b_ -= cache_->gradb * learning_rate;

    return U * A_;
}

Matrix Layer::predict(Matrix&& X) const {
    assert((X.rows() == A_.cols()) && "Incorrect size of input vectors in predict.");

    X = A_ * X;
    for (Index i = 0; i < X.cols(); ++i) {
        X.col(i) = activation_func_.apply(X.col(i));
    }
    return X;
}

void Layer::initCache() {
    cache_ = std::make_unique<Cache>();
}

void Layer::resetCache() {
    cache_.reset();
}

}  // namespace NeuralNetworks
