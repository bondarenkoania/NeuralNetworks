#include "test_network.h"

#include "../Dataset.h"
#include "../Saver.h"
#include "../NetworkBuilder.h"
#include "../AMSGradOptimizer.h"

#include <iostream>

namespace NeuralNetworks {

Matrix softmax(Matrix pred) {
    for (Index i = 0; i < pred.cols(); ++i) {
        pred.col(i) = (pred.col(i).array() - pred.col(i).maxCoeff()).exp();
        pred.col(i) = pred.col(i).array() / pred.col(i).sum();
    }
    return pred;
}

bool is_same(const Vector& p, const Vector& l) {
    int pred_digit;
    p.maxCoeff(&pred_digit);
    int real_digit;
    l.maxCoeff(&real_digit);
    return pred_digit == real_digit;
}

int num_hits(Matrix&& pred, const Matrix& labels) {
    pred = softmax(std::move(pred));
    int h = 0;
    for (Index i = 0; i < pred.cols(); ++i) {
        if (is_same(pred.col(i), labels.col(i)))
            ++h;
    }
    return h;
}

Network default_train_network(std::filesystem::path train_data, std::filesystem::path path_to_save,
                              int epochs, BatchSize batch_size, int n_samples) {
    n_samples = std::min(n_samples, 60000);
    FileReader fl{train_data};
    std::optional<Data> data = fl.read(n_samples);
    if (!data.has_value()) {
        throw std::runtime_error("Error reading " + train_data.string());
    }
    Dataset dataset{std::move(data.value())};
    NetworkBuilder builder(In{dataset.inputSize()});

    auto sigmoid = ActivationFunction();
    auto eucl_loss = LossFunction();

    builder.add_layer(Out{128}, sigmoid).add_layer(Out{64}, sigmoid).add_layer(Out{10}, sigmoid);
    Network network = builder.extract();

    network.train(epochs, batch_size, AMSGradOptimizer(), eucl_loss, dataset);

    if (!Saver::saveParameters(network, path_to_save)) {
        throw std::runtime_error("Error writing parameters to " + path_to_save.string());
    }
    return network;
}

double test_network(std::filesystem::path parameters, std::filesystem::path test_data,
                    int n_samples) {
    n_samples = std::min(n_samples, 10000);
    NetworkBuilder builder(In{784});
    Network net = builder.load_layers(parameters);

    FileReader fl{test_data};
    if (!fl.isOpen()) {
        throw std::invalid_argument("Failed open test data " + test_data.string());
    }
    auto t_data = fl.read(n_samples);
    if (!t_data.has_value()) {
        throw std::invalid_argument("Failed read test data " + test_data.string());
    }
    Dataset test_dataset{std::move(t_data.value())};

    double hits = 0;
    for (auto batch : test_dataset.batches(BatchSize{10})) {
        Matrix pred = softmax(net.predict(batch.images));
        hits += num_hits(std::move(pred), batch.labels);
    }

    std::cout << "hits: " << hits << std::endl;
    double accuracy = hits / test_dataset.size();
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
    return accuracy;
}

}  // namespace NeuralNetworks
