#include "LossFunction.h"

namespace NeuralNetworks {

namespace LFunc {
double SquaredEuclidDist(const Vector& x, const Vector& y) {
    assert(x.size() == y.size() && "Mismatched vector sizes in EucludDist");
    return (x - y).squaredNorm();
}

Row SquaredEuclidDistDer(const Vector& x, const Vector& y) {
    assert(x.size() == y.size() && "Mismatched vector sizes in EucludDistDer");
    return 2 * (x - y).transpose();
}
}  // namespace LFunc

LossFunction::LossFunction() = default;

LossFunction::LossFunction(LossFunc loss_func, LossDer loss_func_der)
    : loss_func_(std::move(loss_func)), loss_func_der_(std::move(loss_func_der)) {
}

double LossFunction::forward(const Matrix& X, const Matrix& Y) const {
    assert(X.size() == Y.size() && "Mismatched matrix sizes in LossFunction forward.");

    double loss = 0;
    for (Index i = 0; i < X.cols(); ++i) {
        loss += loss_func_(X.col(i), Y.col(i));
    }
    return loss;
}

Matrix LossFunction::backward(Matrix&& X, const Matrix& Y) const {
    assert(X.size() == Y.size() && "Mismatched matrix sizes in LossFunction backward.");

    X.transposeInPlace();
    for (Index i = 0; i < X.rows(); ++i) {
        X.row(i) = loss_func_der_(X.row(i), Y.col(i));
    }
    return X;
}

}  // namespace NeuralNetworks
