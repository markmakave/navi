#pragma once

#include <base/array.hpp>
#include <base/matrix.hpp>

#include <neural/blas.hpp>
#include <neural/layer.hpp>

#include <array>

namespace lm {
namespace neural {

class network
{
public:

    /// @brief Construct a new network object
    /// @tparam ...Args Variadic template parameter pack
    /// @param ...args List of layer sizes
    template<typename... Args>
    network(Args... args)
    {
        array<int, sizeof...(Args)> sizes = { args... };

        for (unsigned i = 0; i < sizes.size() - 1; ++i)
        {
            _layers.push(layer(sizes[i], sizes[i + 1]));
        }
    }

    /// @brief Forward pass of the network
    /// @param input Matrix-column of input values
    /// @return Matrix-column of output values
    lm::matrix<double>
    forward(lm::matrix<double> input)
    {   
        for (auto& layer : _layers)
        {
            layer.forward(input);
            input = layer.neurons();
        }

        return input;
    }

    /// @brief Train the network
    /// @param input Matrix-column of input values
    /// @param target Matrix-column of target values
    /// @return Total error
    double
    train(const matrix<double>& input, const matrix<double>& target, double learning_rate)
    {
        if (input.height() != _layers.front().in_size() || target.height() != _layers.back().out_size())
        {
            throw std::runtime_error("Invalid input or target size");
        }
        
        matrix<double> output = forward(input);

        matrix<double> error = output - target;
        error = error.map([](double x) { return x * x; });

        for (int i = _layers.size() - 1; i > 0; --i)
        {
            _layers[i].backpropagate(_layers[i - 1].neurons(), error, learning_rate);
        }
        _layers.front().backpropagate(input, error, learning_rate);

        return error.sum();
    }

private:
    
    array<layer> _layers;

};
    
} // namespace neural
} // namespace lm
