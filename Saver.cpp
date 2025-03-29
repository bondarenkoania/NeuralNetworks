#include "Saver.h"

namespace NeuralNetworks {

bool Saver::saveParameters(const Network& network, const std::filesystem::path& path) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << std::fixed;
    out << network.layers_.size() << "\n";

    for (const auto& layer : network.layers_) {
        const Matrix& A = layer.getA();
        const Vector& b = layer.getb();
        ActivationType type = layer.getActivationType();

        out << static_cast<int>(type) << "\n";
        out << A.rows() << " " << A.cols() << "\n";
        for (Index i = 0; i < A.rows(); ++i) {
            for (Index j = 0; j < A.cols(); ++j)
                out << A(i, j) << " ";
            out << "\n";
        }

        out << b.size() << "\n";
        for (Index i = 0; i < b.size(); ++i)
            out << b(i) << " ";
        out << "\n";
    }
    return true;
}

bool Saver::loadParameters(Network& network, const std::filesystem::path& path) {
    assert(network.layers_.empty() && "Not empty network while parameters loading.");
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Error opening file: " + path.string());
    }

    size_t numLayers;
    input >> numLayers;

    for (size_t l = 0; l < numLayers; ++l) {
        Index rows, cols;
        int act_type;
        input >> act_type >> rows >> cols;
        Matrix A(rows, cols);
        for (Index i = 0; i < rows; ++i) {
            for (Index j = 0; j < cols; ++j)
                input >> A(i, j);
        }

        Index bsize;
        input >> bsize;
        Vector b(bsize);
        for (Index i = 0; i < bsize; ++i)
            input >> b(i);

        network.layers_.emplace_back(std::move(A), std::move(b),
                                     ActivationFunction(static_cast<ActivationType>(act_type)));
    }

    return true;
}

}  // namespace NeuralNetworks
