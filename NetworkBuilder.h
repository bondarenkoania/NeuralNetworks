#pragma once

#include "Network.h"
#include <initializer_list>

namespace NeuralNetworks {

class NetworkBuilder {
public:
    explicit NetworkBuilder(In input_size);
    NetworkBuilder& add_layer(Out output_size, ActivationFunction func,
                              Random& rnd = Random::globalRandom());
    NetworkBuilder& add_layers(std::initializer_list<Out> outputs,
                               std::initializer_list<ActivationFunction> funcs,
                               Random& rnd = Random::globalRandom());
    Network load_layers(std::filesystem::path path);

    void reset(In input_size);
    Network extract();

private:
    Index last_layer_size_;
    Network net_;
};

}  // namespace NeuralNetworks
