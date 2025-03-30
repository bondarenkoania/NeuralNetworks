#pragma once

#include "LinearAlgebra.h"
#include <functional>

namespace NeuralNetworks {

namespace LFunc {

double SquaredEuclidDist(const Vector& x, const Vector& y);
Row SquaredEuclidDistDer(const Vector& x, const Vector& y);

}  // namespace LFunc

class LossFunction {
    using LossFunc = std::function<double(const Vector&, const Vector&)>;
    using LossDer = std::function<Row(const Vector&, const Vector&)>;

public:
    LossFunction() = default;
    LossFunction(LossFunc loss_func, LossDer loss_func_der);
    double calculate(const Matrix& X, const Matrix& Y) const;
    Matrix derivative(Matrix&& X, const Matrix& Y) const;

private:
    LossFunc loss_func_ = LFunc::SquaredEuclidDist;
    LossDer loss_func_der_ = LFunc::SquaredEuclidDistDer;
};

}  // namespace NeuralNetworks
