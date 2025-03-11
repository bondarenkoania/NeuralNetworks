#pragma once

#include "DefaultFunctions.h"
#include "NeuralNetworksTypes.h"
#include <functional>

namespace NeuralNetworks {

    class ActivationFunction {
    public:
        ActivationFunction() : sigma_(ReLU), sigma_derivative_(ReLU_Der) {
        }

        ActivationFunction(std::function<Vector(Vector)> func, std::function<Matrix(Vector)> func_der)
            : sigma_(std::move(func)), sigma_derivative_(std::move(func_der)) {}

        Vector apply(Vector x) const {
            return sigma_(std::move(x));
        }

        Matrix derivative(Vector x) const {
            return sigma_derivative_(std::move(x));
        }

    private:
        std::function<Vector(Vector)> sigma_;
        std::function<Matrix(Vector)> sigma_derivative_;
    };

}
