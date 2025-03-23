#pragma once

#include <Eigen/Core>

namespace NeuralNetworks {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using Row = Eigen::RowVectorXd;
using Index = Eigen::Index;
using Permutation = Eigen::PermutationMatrix<Eigen::Dynamic>;

}  // namespace NeuralNetworks
