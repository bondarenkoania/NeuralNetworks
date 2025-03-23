#include <iostream>
#include <Eigen/Dense>
#include "Dataset.h"

using namespace NeuralNetworks;

int main() {
    try {
        FileReader fl{"/Users/annabondarenko/NeuralNetworks/MNIST_CSV/mnist_train.csv"};
        Data data = fl.read(60000);
        Dataset dataset{std::move(data)};
    } catch (...) {
        std::cout << ":(";
    }

    Matrix im(5, 4);
    int val = 1;
    for (int i = 0; i < im.rows(); ++i) {
        for (int j = 0; j < im.cols(); ++j) {
            im(i, j) = val++;
        }
    }

    Dataset dataset({im, im});
    std::cout << "all data: " << std::endl << im << std::endl;
    dataset.shuffle();

    for (auto batch : dataset.getBatches(BatchSize{2})) {
        std::cout << std::endl << "batch! " << std::endl;
        std::cout << batch.images << std::endl;
    }
}
