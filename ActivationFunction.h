#pragma once

#include "LinearAlgebra.h"
#include <functional>

namespace NeuralNetworks {

namespace AFunc {

Vector ReLU(const Vector& x);
Matrix ReLU_Der(const Vector& x);

}  // namespace AFunc

class ActivationFunction {
    using ApplyFunc = std::function<Vector(Vector)>;
    using DerFunc = std::function<Matrix(Vector)>;

public:
    ActivationFunction();
    ActivationFunction(ApplyFunc func, DerFunc func_der);
    Vector apply(Vector x) const;
    Matrix derivative(Vector x) const;

private:
    ApplyFunc sigma_ = AFunc::ReLU;
    DerFunc sigma_derivative_ = AFunc::ReLU_Der;
};

}  // namespace NeuralNetworks
