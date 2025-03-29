#include "../Dataset.h"

#include <catch2/catch_test_macros.hpp>

#include "../Dataset.h"

using namespace NeuralNetworks;

Matrix makeMatrix(Index rows, Index cols) {
    Matrix m(rows, cols);
    int val = 1;
    for (Index i = 0; i < rows; ++i) {
        for (Index j = 0; j < cols; ++j) {
            m(i, j) = val++;
        }
    }
    return m;
}

TEST_CASE("Dataset") {
    Index data_size = 7;
    Index vec_size = 4;
    Matrix im = makeMatrix(data_size, vec_size);

    Dataset dataset({im, im});
    dataset.shuffle();

    int b = 0;
    int batch_size = 3;
    for (auto batch : dataset.batches(BatchSize{batch_size})) {
        ++b;
        REQUIRE(batch.images.rows() == vec_size);
        if (b != data_size / batch_size + 1) {
            REQUIRE(batch.images.cols() == batch_size);
        } else {
            REQUIRE(batch.images.cols() == data_size % batch_size);
        }
    }
}
