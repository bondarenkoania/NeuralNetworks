#pragma once

#include "NeuralNetworksTypes.h"

namespace NeuralNetworks {

    Vector ReLU(Vector x);
    Matrix ReLU_Der(Vector x);

    double SquaredEuclidDist(const Vector& x, const Vector& y);
    Row SquaredEuclidDistDer(const Vector& x, const Vector& y);

}
