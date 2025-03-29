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

Vector Sigmoid(const Vector& x) {
    return x.unaryExpr([](double a) { return 1.0 / (1.0 + std::exp(-a)); });
}

Matrix Sigmoid_Der(const Vector& x) {
    Vector der = x.unaryExpr([](double a) { return 1.0 / (std::exp(a) + std::exp(-a) + 2); });
    return der.asDiagonal();
}
}  // namespace AFunc

ActivationFunction::ActivationFunction(ActivationType type) : type_(type) {
    switch (type_) {
        case ActivationType::Sigmoid:
            break;
        case ActivationType::ReLU:
            sigma_ = AFunc::ReLU;
            sigma_derivative_ = AFunc::ReLU_Der;
            break;
        default:
            throw std::invalid_argument("Unknown activation type.");
    }
}

ActivationFunction::ActivationFunction(ApplyFunc func, DerFunc func_der)
    : sigma_(std::move(func)),
      sigma_derivative_(std::move(func_der)),
      type_(ActivationType::Custom) {
}

Vector ActivationFunction::apply(const Vector& x) const {
    return sigma_(x);
}

Matrix ActivationFunction::derivative(const Vector& x) const {
    return sigma_derivative_(x);
}

ActivationType ActivationFunction::getType() const {
    return type_;
}

}  // namespace NeuralNetworks
