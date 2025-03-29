#include <catch2/catch_test_macros.hpp>

#include "../Layer.h"
#include "../SimpleOptimizer.h"

using namespace NeuralNetworks;

TEST_CASE("Layer: predict and forward correctness with ReLU") {
    Random rnd = Random::globalRandom();
    Layer layer(In{2}, Out{1}, ActivationFunction(ActivationType::ReLU), rnd);

    Matrix input(2, 3);
    input << 1.0, -2.0, 1.0, -1.0, 0.0, 0.5;

    Matrix output;
    SECTION("Layer predict") {
        output = layer.predict(std::move(input));
    }

    SECTION("Layer forward with SimpleOptimizer") {
        layer.initCache(SimpleOptimizer());
        output = layer.forward(std::move(input));
    }

    REQUIRE(output.rows() == 1);
    REQUIRE(output.cols() == 3);

    for (Index i = 0; i < output.cols(); ++i) {
        REQUIRE(output(0, i) >= 0.0);
    }
}
