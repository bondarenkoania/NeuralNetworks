#pragma once

namespace NeuralNetworks {

class Network;

class Scheduler {
public:
    double GetLearningRate(Network* model) {
        return 0.01;
    }
};

}