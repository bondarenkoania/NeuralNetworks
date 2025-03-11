#pragma once

#include <Eigen/Core>

namespace NeuralNetworks {
    enum InputSize : int;
    enum OutputSize : int;
    enum BatchSize : int;

    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    using Row = Eigen::RowVectorXd;
}
