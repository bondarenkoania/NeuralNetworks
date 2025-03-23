#include "Random.h"
#include <algorithm>

namespace NeuralNetworks {

Random::Random() = default;

Random::Random(int seed) : generator_(seed) {
}

Matrix Random::normalMatrix(Index rows, Index cols) {
    return Eigen::Rand::normal<Matrix>(rows, cols, generator_);
}

Matrix Random::normalMatrix(Index rows, Index cols, double mean, double stdev) {
    return Eigen::Rand::normal<Matrix>(rows, cols, generator_, mean, stdev);
}

Vector Random::normalVector(Index rows) {
    return Eigen::Rand::normal<Matrix>(rows, 1, generator_);
}

Vector Random::normalVector(Index rows, double mean, double stdev) {
    return Eigen::Rand::normal<Matrix>(rows, 1, generator_, mean, stdev);
}

void Random::shuffleData(Matrix& images, Matrix& labels) {
    Permutation perm(images.rows());
    perm.setIdentity();
    std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), generator_);

    images = perm * images;
    labels = perm * labels;
}

Random& Random::globalRandom() {
    static Random rnd;
    return rnd;
}

}  // namespace NeuralNetworks
