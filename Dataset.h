#pragma once

#include "FileReader.h"

namespace NeuralNetworks {

enum BatchSize : Index;

using Batch = Data;

class Dataset {
public:
    explicit Dataset(Data&& data);
    void shuffle();
    Index size() const;

    class BatchIterator {
    public:
        BatchIterator(const Dataset& dataset, BatchSize batch_size, Index ind);
        Batch operator*() const;
        BatchIterator& operator++();
        bool operator==(const BatchIterator& other) const;
        bool operator!=(const BatchIterator& other) const;

    private:
        const Dataset& dataset_;
        BatchSize batch_size_;
        Index ind_;
    };

    class BatchRange {
    public:
        BatchRange(const Dataset& dataset, BatchSize batch_size);
        BatchIterator begin() const;
        BatchIterator end() const;

    private:
        const Dataset& dataset_;
        BatchSize batch_size_;
    };

    BatchRange getBatches(BatchSize batch_size) const;

private:
    Data data_;
};

}  // namespace NeuralNetworks
