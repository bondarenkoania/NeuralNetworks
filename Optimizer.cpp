#include "Optimizer.h"

namespace NeuralNetworks {

Optimizer::Optimizer(const Optimizer& other)
    : model_(other.isDefined() ? other->make_copy_() : nullptr) {
}

Optimizer& Optimizer::operator=(const Optimizer& other) {
    return *this = Optimizer(other);
}

const Optimizer::Concept* Optimizer::operator->() const {
    return model_.get();
}

Optimizer::Concept* Optimizer::operator->() {
    return model_.get();
}

bool Optimizer::isDefined() const {
    return model_ != nullptr;
}

void Optimizer::clear() {
    model_.reset();
}

}  // namespace NeuralNetworks
