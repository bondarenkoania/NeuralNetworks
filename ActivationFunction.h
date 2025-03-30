#pragma once

#include "LinearAlgebra.h"
#include <functional>

namespace NeuralNetworks {

namespace AFunc {

Vector ReLU(const Vector& x);
Matrix ReLU_Der(const Vector& x);

Vector Sigmoid(const Vector& x);
Matrix Sigmoid_Der(const Vector& x);

}  // namespace AFunc

enum class ActivationType { Custom = -1, ReLU = 0, Sigmoid = 1 };

class ActivationFunction {
    using ApplyFunc = std::function<Vector(const Vector&)>;
    using DerFunc = std::function<Matrix(const Vector&)>;

public:
    ActivationFunction() = default;
    ActivationFunction(ActivationType type);
    ActivationFunction(ApplyFunc func, DerFunc func_der);

    Vector apply(const Vector& x) const;
    Matrix derivative(const Vector& x) const;
    ActivationType getType() const;

private:
    ApplyFunc sigma_ = AFunc::Sigmoid;
    DerFunc sigma_derivative_ = AFunc::Sigmoid_Der;
    ActivationType type_ = ActivationType::Sigmoid;
};

}  // namespace NeuralNetworks
