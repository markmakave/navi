#pragma once

#include "lumina.hpp"

#include <functional>

namespace lm {
namespace neural {

auto sigmoid = [](float x){ return 1.f / (1.f + std::exp(-x)); };
auto sigmoid_derivative = [](float x){ float s = sigmoid(x); return s * (1.f - s); };

auto tanh = [](float x){ return std::tanh(x); };
auto tanh_derivative = [](float x){ float t = tanh(x); return 1.f - t*t; };

auto relu = [](float x){ return x > 0 ? x : 0; };
auto relu_derivative = [](float x){ return x > 0 ? 1 : 0; };

class layer
{
    friend class network;

public:

    using size_type = typename matrix<float>::size_type;

public:

layer()
{}

layer(
    size_type in_size, 
    size_type out_size
)
:   _neurons(out_size),
    _weights(in_size, out_size),
    _biases(out_size)
{}

const array<float>&
forward(const array<float>& in)
{
    assert(in.size() == in_size());

    blas::mv(_weights, in, _neurons);
    blas::add(_neurons, _biases, _neurons);
    blas::map(_neurons, activation, _neurons);

    return _neurons;
}

void
backward(const array<float>& input, array<float>& error, float learning_rate)
{
    static array<float> gradient;

    blas::map(_neurons, activation_derivative, gradient);
    blas::mul(gradient, error, gradient);

    blas::axpy(-learning_rate, gradient, _biases);
    blas::ger(gradient, input, -learning_rate, _weights);

    blas::mv(_weights, gradient, error, true);
}

size_type
in_size() const
{
    return _weights.shape()[0];
}

size_type
out_size() const
{
    return _weights.shape()[1];
}

public:

    std::function<float(float)> activation = sigmoid, activation_derivative = sigmoid_derivative;

protected:

    array<float> _neurons;
    matrix<float> _weights;
    array<float> _biases;
};

}
}
