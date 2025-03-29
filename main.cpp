#include "tests/test_network.h"

#include <iostream>

namespace nn = NeuralNetworks;

constexpr int mnist_test_size = 10000;
constexpr int mnist_train_size = 60000;

int main() {
    try {
        auto cur_dir = std::string(SOURCE_DIR);
        nn::test_network(cur_dir + "/saved/sigmoid_10epochs_20_full.txt",
                         cur_dir + "/MNIST_CSV/mnist_test.csv", mnist_test_size);

    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}
