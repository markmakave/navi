#pragma once

#include <base/array.hpp>
#include <base/matrix.hpp>

#include <cmath>

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
        _weights = lm::array<lm::matrix<double>>(sizeof...(Args) - 1);
        _biases = lm::array<lm::matrix<double>>(sizeof...(Args) - 1);

        lm::array<int> sizes = {args...};

        for (unsigned i = 0; i < _weights.size(); ++i)
        {
            _weights[i] = lm::matrix<double>(sizes[i + 1], sizes[i]);
            _biases[i] = lm::matrix<double>(sizes[i + 1], 1);
        }

        for (unsigned i = 0; i < _weights.size(); ++i)
        {
            _weights[i].randomize();
            _biases[i].randomize();
        }
    }

    /// @brief Forward pass of the network
    /// @param input Matrix-column of input values
    /// @return Matrix-column of output values
    lm::matrix<double>
    forward(lm::matrix<double> input)
    {  
        if (input.height() != _weights[0].width())
        {
            throw std::runtime_error("input size mismatch");
        }

        lm::matrix<double> output = input;

        for (unsigned i = 0; i < _weights.size(); ++i)
        {
            output = _weights[i] * output + _biases[i];
            output = output.map([](double x) { return std::tanh(x); });
        }

        return output;
    }

    /// @brief Train the network
    /// @param input Matrix-column of input values
    /// @param target Matrix-column of target values
    /// @return Total error
    double
    train(const matrix<double>& input, const matrix<double>& target, double learning_rate)
    {
        if (input.height() != _weights[0].width())
        {
            throw std::runtime_error("input size mismatch");
        }

        if (target.height() != _weights[_weights.size() - 1].height())
        {
            throw std::runtime_error("target size mismatch");
        }

        matrix<double> output = forward(input);

        matrix<double> error = target - output;
        error = error.map([](double x) { return std::pow(x, 2); });
        double total_error = error.sum();

        matrix<double> gradient = output.map([](double x) { return 1 - std::pow(std::tanh(x), 2); });
        gradient = gradient * error;
        gradient = gradient * learning_rate;

        matrix<double> delta = gradient * input.transpose();

        _weights[_weights.size() - 1] = _weights.back() + delta;
        _biases[_biases.size() - 1] = _biases.back() + gradient;


        // for (unsigned i = _weights.size() - 2; i > 0; --i)
        // {
        //     gradient = _weights[i + 1].transpose() * gradient;
        //     gradient = gradient.map([](double x) { return 1 - std::pow(std::tanh(x), 2); });
        //     gradient = gradient * error;
        //     gradient = gradient * learning_rate;

        //     delta = gradient * input.transpose();

        //     _weights[i] = _weights[i] + delta;
        //     _biases[i] = _biases[i] + gradient;
        // }

        return total_error;
    }

private:
    
    lm::array<lm::matrix<double>> _weights;
    lm::array<lm::matrix<double>> _biases;

};
    
} // namespace neural
} // namespace lm
