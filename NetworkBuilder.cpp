#include "NetworkBuilder.h"

namespace NeuralNetworks {

NetworkBuilder::NetworkBuilder(In input_size) : last_layer_size_(input_size) {
}

NetworkBuilder& NetworkBuilder::add_layer(Out output_size, ActivationFunction func, Random& rnd) {
    net_.layers_.emplace_back(In{last_layer_size_}, output_size, std::move(func), rnd);
    last_layer_size_ = output_size;
    return *this;
}

void NetworkBuilder::add_layers(std::initializer_list<Out> outputs,
                                std::initializer_list<ActivationFunction> funcs, Random& rnd) {
    assert(outputs.size() == funcs.size() &&
           "Mismatched number of layers and number of activation functions.");
    auto it1 = outputs.begin();
    auto it2 = funcs.begin();
    for (; it1 != outputs.end(); ++it1, ++it2) {
        add_layer(*it1, *it2, rnd);
    }
}

void NetworkBuilder::reset(In input_size) {
    net_.layers_.clear();
    last_layer_size_ = input_size;
}

Network NetworkBuilder::extract() {
    return std::move(net_);
}

}  // namespace NeuralNetworks
