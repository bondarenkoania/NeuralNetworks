#include "Layer.h"

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>

namespace NeuralNetworks {

    Layer::Layer(
        InputSize input_size, OutputSize output_size,
        Eigen::Rand::P8_mt19937_64& urng, ActivationFunction* activation_func)
            : A(Eigen::Rand::normal<Matrix>(output_size, input_size, urng))
            , b(Eigen::Rand::normal<Matrix>(output_size, 1, urng))
            , activation_func_(activation_func) {
    }

    Matrix Layer::forward(const Matrix& X) {
        assert((X.rows() == A.cols()) && "Incorrect size of input vectors in forward.");
        auto batch_size = X.cols();
        input_batch_ = X;
        modified_input_batch_ = A * X + b.replicate(1, batch_size);
        Matrix result(A.rows(), batch_size);
        for (int i = 0; i < batch_size; ++i) {
            result.col(i) = activation_func_->apply(modified_input_batch_.col(i));
        }
        return result;
    }

    Matrix Layer::backward(Matrix U) {
        assert((U.cols() == A.rows()) && "Incorrect size of input rows in backward.");
        assert((U.rows() == input_batch_.cols()) && "Incorrect batch size in backward.");
        size_t batch_size = U.rows();
        for (int i = 0; i < batch_size; ++i) {
            U.row(i) *= activation_func_->derivative(modified_input_batch_.col(i));
        }
        gradb = U.transpose().rowwise().mean();
        gradA = U.transpose() * input_batch_.transpose() / batch_size;
        return U * A;
    }

    void Layer::update(double learning_rate_A, double learning_rate_b) {
        A -= gradA * learning_rate_A;
        b -= gradb * learning_rate_b;
    }

}
