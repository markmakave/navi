#pragma once

#include <base/matrix.hpp>

namespace lm {
namespace neural {

class layer
{
public:

    layer(size_t in_size, size_t out_size)
    :   _neurons(out_size, 1),
        _weights(out_size, in_size),
        _bias(out_size, 1)
    {
        randomize();
    }

    void
    randomize()
    {
        _weights.randomize();
        _bias.randomize();
    }

    void
    activate()
    {
        // positive sigmoid
        _neurons = _neurons.map([](double x) { return std::tanh(x); });
    }

    void
    forward(const matrix<double>& input)
    {
        _neurons = _weights * input + _bias;
        activate();
    }

    void
    backpropagate(const matrix<double>& input, matrix<double>& error, double learning_rate)
    {
        if (input.height() != in_size() || error.height() != out_size())
        {
            throw std::runtime_error("Invalid input or error size");
        }

        // calculate error
        matrix<double> delta = _weights.transpose() * error;

        // calculate gradient
        matrix<double> gradient = _neurons;
        gradient = gradient.map([](double x) {
            double t = std::tanh(x);
            return (1.0 - t * t);
        });
        elemul(gradient, error, gradient);

        // update weights
        _weights += gradient * input.transpose() * learning_rate;
        _bias += gradient * learning_rate;
        
        // update error
        error = delta;
    }

    // getters

    const matrix<double>&
    neurons() const
    {
        return _neurons;
    }

    unsigned
    in_size() const
    {
        return _weights.width();
    }

    unsigned
    out_size() const
    {
        return _weights.height();
    }

private:

    matrix<double> _neurons;
    matrix<double> _weights;
    matrix<double> _bias;
};


} // namespace neural
} // namespace lm