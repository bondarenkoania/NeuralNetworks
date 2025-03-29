#include "Dataset.h"

namespace NeuralNetworks {

Dataset::Dataset(Data&& data) : data_(std::move(data)) {
    assert(data_.images.rows() == data_.labels.rows() &&
           "Mismatched numbers of images and labels.");
}

void Dataset::shuffle(Random& rnd) {
    auto perm = rnd.permMatrix(data_.images.rows());
    data_.images = perm * data_.images;
    data_.labels = perm * data_.labels;
}

Index Dataset::size() const {
    return data_.images.rows();
}

Index Dataset::inputSize() const {
    return data_.images.cols();
}

Dataset::BatchIterator::BatchIterator(const Dataset& dataset, BatchSize batch_size, Index ind)
    : dataset_(dataset), batch_size_(batch_size), ind_(ind) {
}

Dataset::Batch Dataset::BatchIterator::operator*() const {
    Index start = ind_ * batch_size_;
    Index len = (dataset_.get().size() - start >= batch_size_) ? batch_size_
                                                               : dataset_.get().size() - start;
    return {dataset_.get().data_.images.middleRows(start, len).transpose(),
            dataset_.get().data_.labels.middleRows(start, len).transpose()};
}

Dataset::BatchIterator& Dataset::BatchIterator::operator++() {
    ind_++;
    return *this;
}

bool Dataset::BatchIterator::operator==(const BatchIterator& other) const {
    return ind_ == other.ind_;
}

bool Dataset::BatchIterator::operator!=(const BatchIterator& other) const {
    return !(*this == other);
}

Dataset::BatchRange::BatchRange(const Dataset& dataset, BatchSize batch_size)
    : dataset_(dataset), batch_size_(batch_size) {
}

Dataset::BatchIterator Dataset::BatchRange::begin() const {
    return {dataset_, batch_size_, 0};
}

Dataset::BatchIterator Dataset::BatchRange::end() const {
    return {dataset_, batch_size_, (dataset_.get().size() + batch_size_ - 1) / batch_size_};
}

Dataset::BatchRange Dataset::batches(BatchSize batch_size) const {
    return {*this, batch_size};
}

}  // namespace NeuralNetworks
