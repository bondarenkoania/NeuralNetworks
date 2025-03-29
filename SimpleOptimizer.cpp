#include "SimpleOptimizer.h"

namespace NeuralNetworks {

SimpleOptimizer::SimpleOptimizer(double lr) : learning_rate_(lr) {
}

void SimpleOptimizer::update(Matrix& w, Matrix&& gradA, std::any& any_cache) const {
    w -= learning_rate_ * gradA;
}

void SimpleOptimizer::update(Vector& w, Vector&& gradb, std::any& any_cache) const {
    w -= learning_rate_ * gradb;
}

std::any SimpleOptimizer::initCache(const Matrix& w) const {
    return {};
}

std::any SimpleOptimizer::initCache(const Vector& w) const {
    return {};
}

}  // namespace NeuralNetworks
