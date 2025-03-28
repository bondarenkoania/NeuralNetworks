#pragma once

#include "LinearAlgebra.h"
#include <functional>

namespace NeuralNetworks {

namespace AFunc {

Vector ReLU(const Vector& x);
Matrix ReLU_Der(const Vector& x);

}  // namespace AFunc

class ActivationFunction {
    using ApplyFunc = std::function<Vector(const Vector&)>;
    using DerFunc = std::function<Matrix(const Vector&)>;

public:
    ActivationFunction() = default;
    ActivationFunction(ApplyFunc func, DerFunc func_der);
    Vector apply(const Vector& x) const;
    Matrix derivative(const Vector& x) const;

private:
    ApplyFunc sigma_ = AFunc::ReLU;
    DerFunc sigma_derivative_ = AFunc::ReLU_Der;
};

}  // namespace NeuralNetworks
