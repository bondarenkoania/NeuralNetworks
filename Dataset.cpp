#include "Dataset.h"
#include "Random.h"

namespace NeuralNetworks {

Dataset::Dataset(Data&& data) {
    assert(data.images.rows() == data.labels.rows() && "Mismatched numbers of images and labels.");
    data_ = std::move(data);
}

Index Dataset::size() const {
    return data_.images.rows();
}

Dataset::BatchIterator::BatchIterator(const Dataset& dataset, BatchSize batch_size, Index ind)
    : dataset_(dataset), batch_size_(batch_size), ind_(ind) {
}

Batch Dataset::BatchIterator::operator*() const {
    Index start = ind_ * batch_size_;
    Index len = (dataset_.size() - start >= batch_size_) ? batch_size_ : dataset_.size() - start;
    return {dataset_.data_.images.middleRows(start, len).transpose(),
            dataset_.data_.labels.middleRows(start, len).transpose()};
}

Dataset::BatchIterator& Dataset::BatchIterator::operator++() {
    ind_++;
    return *this;
}

bool Dataset::BatchIterator::operator==(const BatchIterator& other) const {
    return ind_ == other.ind_;
}

bool Dataset::BatchIterator::operator!=(const BatchIterator& other) const {
    return ind_ != other.ind_;
}

Dataset::BatchRange::BatchRange(const Dataset& dataset, BatchSize batch_size)
    : dataset_(dataset), batch_size_(batch_size) {
}

Dataset::BatchIterator Dataset::BatchRange::begin() const {
    return {dataset_, batch_size_, 0};
}

Dataset::BatchIterator Dataset::BatchRange::end() const {
    return {dataset_, batch_size_, (dataset_.size() + batch_size_ - 1) / batch_size_};
}

Dataset::BatchRange Dataset::getBatches(BatchSize batch_size) const {
    return {*this, batch_size};
}

void Dataset::shuffle() {
    Random::globalRandom().shuffleData(data_.images, data_.labels);
}

}  // namespace NeuralNetworks
