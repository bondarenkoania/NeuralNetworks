#pragma once

#include "DefaultFunctions.h"
#include "NeuralNetworksTypes.h"
#include <functional>

namespace NeuralNetworks {
    class LossFunction {
    public:
        LossFunction() : loss_func_(SquaredEuclidDist), loss_func_der_(SquaredEuclidDistDer) {
        }

        LossFunction(std::function<double(const Vector&, const Vector&)> loss_func,
            std::function<Row(const Vector&, const Vector&)> loss_func_der)
                : loss_func_(std::move(loss_func)), loss_func_der_(std::move(loss_func_der)) {
        }

        double forward(const Matrix& x, const Matrix& y) const {
            double loss = 0;
            for (int i = 0; i < x.cols(); ++i) {
                loss += loss_func_(x.col(i), y.col(i));
            }
            return loss;
        }

        Row backward(const Matrix& x, const Matrix& y) const {
            size_t batch_size = x.cols();
            size_t grad_size = x.rows();
            Matrix grad_batch(batch_size, grad_size);
            for (int i = 0; i < batch_size; ++i) {
                grad_batch.row(i) = loss_func_der_(x.col(i), y.col(i));
            }
            return grad_batch;
        }

    private:
        std::function<double(const Vector&, const Vector&)> loss_func_;
        std::function<Row(const Vector&, const Vector&)> loss_func_der_;
    };

}
