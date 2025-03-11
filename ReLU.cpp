#include "NeuralNetworksTypes.h"

namespace NeuralNetworks {
    Vector ReLU(Vector x) {
        for (int i = 0; i < x.size(); ++i) {
            x(i) = std::max(x(i), 0.0);
        }
        return x;
    }

    Matrix ReLU_Der(Vector x) {
        Matrix derivative = Matrix::Zero(x.size(), x.size());
        for (int i = 0; i < x.size(); ++i) {
            derivative(i, i) = (x(i) < 0.0) ? 0 : 1;
        }
        return derivative;
    }
}
