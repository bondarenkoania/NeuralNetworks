#pragma once

#include "Network.h"
#include <initializer_list>

namespace NeuralNetworks {

class NetworkBuilder {
public:
    explicit NetworkBuilder(In input_size);
    NetworkBuilder& add_layer(Out output_size, ActivationFunction func,
                              Random& rnd = Random::globalRandom());
    void add_layers(std::initializer_list<Out> outputs,
                    std::initializer_list<ActivationFunction> funcs,
                    Random& rnd = Random::globalRandom());
    void reset(In input_size);
    Network extract();

private:
    Index last_layer_size_;
    Network net_;
};

}  // namespace NeuralNetworks
