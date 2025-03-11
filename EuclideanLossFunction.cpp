#include "NeuralNetworksTypes.h"

namespace NeuralNetworks {
    double SquaredEuclidDist(const Vector& x, const Vector& y) {
        assert(x.size() == y.size() && "Mismatched vector sizes in EucludDist");
        double dist = 0.0;
        for (auto i = 0; i < x.size(); ++i) {
            dist += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return dist;
    }
    Row SquaredEuclidDistDer(const Vector& x, const Vector& y) {
        Row grad(x.size());
        for (auto i = 0; i < x.size(); ++i) {
            grad[i] = 2 * (x[i] - y[i]);
        }
        return grad;
    }
}
