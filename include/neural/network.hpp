/*

    Copyright (c) 2023 Mokhov Mark

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.

*/

#pragma once

#include "base/matrix.hpp"
#include "base/array.hpp"
#include "base/blas.hpp"

#include <cassert>
#include <random>

namespace lm {
namespace neural {

inline float
sigmoid(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

inline float
sigmoid_derivative(float x)
{
    float s = sigmoid(x);
    return s * (1.f - s);
}

inline float
tanh(float x)
{
    return std::tanh(x);
}

inline float
tanh_derivative(float x)
{
    float t = tanh(x);
    return 1.f - t * t;
}

inline float
relu(float x)
{
    return x > 0 ? x : 0;
};

inline float
relu_derivative(float x)
{
    return x > 0 ? 1 : 0;
}

static float
random(float)
{
    static std::random_device              rd;
    static std::mt19937                    engine(rd());
    static std::normal_distribution<float> dis(-1, 1);
    return dis(engine);
};

class network;

class layer
{
    friend class network;

public:

    using size_type = typename matrix<float>::size_type;

public:

    layer()
    {}

    layer(size_type in_size, size_type out_size)
      : _neurons(out_size),
        _weights(in_size, out_size),
        _biases(out_size)
    {
        blas::map(_weights, random, _weights);
        blas::map(_biases, random, _biases);
    }

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
    backward(const array<float>& input,
             array<float>&       error,
             float               learning_rate)
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

    void
    read(std::ifstream& file)
    {
        _weights.read(file);
        _biases.read(file);
    }

    void
    write(std::ofstream& file) const
    {
        _weights.write(file);
        _biases.write(file);
    }

public:

    std::function<float(float)> activation            = relu,
                                activation_derivative = relu_derivative;

protected:

    array<float>  _neurons;
    matrix<float> _weights;
    array<float>  _biases;
};

class network
{
public:

    using type      = float;
    using size_type = matrix<float>::size_type;

public:

    template <typename... Args>
    network(Args... args)
    {
        size_type sizes[sizeof...(Args)] = {args...};

        _layers.reshape(sizeof...(Args) - 1);

        for (unsigned i = 0; i < sizeof...(Args) - 1; ++i)
            _layers(i) = layer(sizes[i], sizes[i + 1]);
    }

    network(const char* filename)
    {
        read(filename);
    }

    const array<type>&
    infer(const array<type>& in)
    {
        _layers.front().forward(in);
        for (size_type i = 1; i < _layers.size(); ++i)
            _layers(i).forward(_layers(i - 1)._neurons);

        return _layers.back()._neurons;
    }

    void
    train(const array<type>& input,
          const array<type>& target,
          type               learning_rate)
    {
        assert(input.size() == _layers.front().in_size());
        assert(target.size() == _layers.back().out_size());

        array<type> error;
        array<type> gradient;

        blas::sub(infer(input), target, error);

        for (int i = _layers.size() - 1; i >= 0; i--) {
            _layers(i).backward(
                i == 0 ? input : _layers(i - 1)._neurons, error, learning_rate);
        }
    }

    size_type
    in_size() const
    {
        return _layers.front().in_size();
    }

    size_type
    out_size() const
    {
        return _layers.back().out_size();
    }

    void
    read(const char* filename)
    {
        std::ifstream file(filename);

        size_type model_size;
        file.read((char*)&model_size, sizeof(model_size));
        _layers.reshape(model_size);

        for (auto& layer : _layers)
            layer.read(file);
    }

    void
    write(const char* filename) const
    {
        std::ofstream file(filename);

        size_type model_size = _layers.size();
        file.write((char*)&model_size, sizeof(model_size));

        for (const auto& layer : _layers)
            layer.write(file);
    }

protected:

    array<layer> _layers;
};

} // namespace neural
} // namespace lm
