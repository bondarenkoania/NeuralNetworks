#pragma once

#include "LinearAlgebra.h"
#include <EigenRand/EigenRand>

namespace NeuralNetworks {

class Random {
    using Generator = Eigen::Rand::P8_mt19937_64;

public:
    Random() = default;
    explicit Random(int seed);
    Matrix normalMatrix(Index rows, Index cols);
    Matrix normalMatrix(Index rows, Index cols, double mean, double stdev);
    Vector normalVector(Index rows);
    Vector normalVector(Index rows, double mean, double stdev);
    Matrix permMatrix(Index size);

    static Random& globalRandom();

private:
    static constexpr int k_default_seed_ = 42;
    Generator generator_{k_default_seed_};
};

}  // namespace NeuralNetworks
