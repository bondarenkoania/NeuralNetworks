#pragma once
#include "LinearAlgebra.h"
#include <any>
#include <memory>

namespace NeuralNetworks {

class Optimizer {
private:
    class Concept {
    public:
        virtual void update(Matrix& w, Matrix&& grad, std::any& cache) const = 0;
        virtual void update(Vector& w, Vector&& grad, std::any& cache) const = 0;
        virtual void initCache(std::any& cache, const Vector& w) const = 0;
        virtual void initCache(std::any& cache, const Matrix& w) const = 0;
        virtual ~Concept() = default;

    private:
        friend class Optimizer;
        virtual std::unique_ptr<Concept> make_copy_() const = 0;
    };

    template <typename Opt>
    class Model : public Concept {
    public:
        using DecayedOpt = std::decay_t<Opt>;
        Model(const DecayedOpt& optimizer) : obj_(optimizer) {
        }
        Model(DecayedOpt&& optimizer) : obj_(std::move(optimizer)) {
        }

        void update(Matrix& w, Matrix&& grad, std::any& cache) const final {
            obj_.update(w, std::move(grad), cache);
        }
        void update(Vector& w, Vector&& grad, std::any& cache) const final {
            obj_.update(w, std::move(grad), cache);
        }
        void initCache(std::any& cache, const Vector& w) const final {
            obj_.initCache(cache, w);
        }
        void initCache(std::any& cache, const Matrix& w) const final {
            obj_.initCache(cache, w);
        }

    private:
        std::unique_ptr<Concept> make_copy_() const final {
            return std::make_unique<Model<DecayedOpt>>(obj_);
        }
        DecayedOpt obj_;
    };

public:
    Optimizer() = default;

    template <typename Opt,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<Opt>, Optimizer>>>
    Optimizer(Opt&& object) : model_(std::make_unique<Model<Opt>>(std::forward<Opt>(object))) {
    }

    Optimizer(const Optimizer& other);
    Optimizer& operator=(const Optimizer& other);
    Optimizer(Optimizer&& other) noexcept = default;
    Optimizer& operator=(Optimizer&& other) noexcept = default;

    const Concept* operator->() const;
    Concept* operator->();

    bool isDefined() const;
    void clear();

private:
    std::unique_ptr<Concept> model_;
};

}  // namespace NeuralNetworks
