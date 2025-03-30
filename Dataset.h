#pragma once

#include "FileReader.h"
#include "Random.h"

namespace NeuralNetworks {

enum BatchSize : Index;

class Dataset {
public:
    using Batch = Data;

    explicit Dataset(Data&& data);
    void shuffle(Random& rnd = Random::globalRandom());
    Index size() const;
    Index inputSize() const;

    class BatchIterator {
    public:
        BatchIterator(const Dataset& dataset, BatchSize batch_size, Index ind);
        Batch operator*() const;
        BatchIterator& operator++();
        bool operator==(const BatchIterator& other) const;
        bool operator!=(const BatchIterator& other) const;

    private:
        std::reference_wrapper<const Dataset> dataset_;
        BatchSize batch_size_;
        Index ind_;
    };

    class BatchRange {
    public:
        BatchRange(const Dataset& dataset, BatchSize batch_size);
        BatchIterator begin() const;
        BatchIterator end() const;

    private:
        std::reference_wrapper<const Dataset> dataset_;
        BatchSize batch_size_;
    };

    BatchRange batches(BatchSize batch_size) const;

private:
    Data data_;
};

}  // namespace NeuralNetworks
