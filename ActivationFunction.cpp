#include "ActivationFunction.h"

namespace NeuralNetworks {

namespace AFunc {
Vector ReLU(const Vector& x) {
    return x.unaryExpr([](double a) { return std::max(a, 0.0); });
}

Matrix ReLU_Der(const Vector& x) {
    Vector der = x.unaryExpr([](double a) { return (a >= 0.0) ? 1.0 : 0.0; });
    return der.asDiagonal();
}
}  // namespace AFunc

ActivationFunction::ActivationFunction() = default;

ActivationFunction::ActivationFunction(ApplyFunc func, DerFunc func_der)
    : sigma_(std::move(func)), sigma_derivative_(std::move(func_der)) {
}

Vector ActivationFunction::apply(Vector x) const {
    return sigma_(std::move(x));
}

Matrix ActivationFunction::derivative(Vector x) const {
    return sigma_derivative_(std::move(x));
}

}  // namespace NeuralNetworks
