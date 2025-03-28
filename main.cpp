#include <iostream>
#include <Eigen/Dense>
#include "Dataset.h"
#include "Optimizer.h"
#include "AMSGradOptimizer.h"
#include "Dataset.h"

using namespace NeuralNetworks;

int main() {

    FileReader fl{std::string(SOURCE_DIR) + "/MNIST_CSV/mnist_train.csv"};
    std::optional<Data> data = fl.read(10);
    if (data.has_value()) {
        Dataset dataset{std::move(data.value())};
        for (auto b : dataset.batches(BatchSize{3})) {
            std::cout << std::endl << "baaatch! " << std::endl;
            std::cout << b.images.middleRows(0, 20) << std::endl;
        }
    }

    std::cout << "Working dir: " << std::filesystem::current_path() << std::endl;
    Optimizer opt = AMSGradOptimizer();

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

    for (auto batch : dataset.batches(BatchSize{2})) {
        std::cout << std::endl << "batch! " << std::endl;
        std::cout << batch.images << std::endl;
    }
}
