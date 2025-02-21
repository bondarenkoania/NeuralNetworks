#include <iostream>
#include <Eigen/Core>
 #include <EigenRand/EigenRand>

using namespace Eigen;
int main() {
    std::cout << EIGEN_WORLD_VERSION << EIGEN_MAJOR_VERSION << EIGEN_MINOR_VERSION << '\n';
    // Initialize random number generator with seed=42 for following codes.
    // Or you can use C++11 RNG such as std::mt19937 or std::ranlux48.
    Rand::P8_mt19937_64 urng{ 42 };

    // this will generate 4x4 real matrix with range [-1, 1]
    MatrixXf mat = Rand::balanced<MatrixXf>(4, 4, urng);
    std::cout << mat << std::endl;

    // this will generate 10x10 real 2d array on the normal distribution
    ArrayXXf arr = Rand::normal<ArrayXXf>(10, 10, urng);
    std::cout << arr << std::endl;

    return 0;
}