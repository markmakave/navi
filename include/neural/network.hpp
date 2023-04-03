/* 

    Copyright (c) 2023 Mark Mokhov

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

*/

#pragma once

#include "base/matrix.hpp"
#include "base/array.hpp"

// #include "cuda/matrix.cuh"
// #include "cuda/array.cuh"

#include "base/blas.hpp"

namespace lm {
namespace neural {

auto activation = [](float x) {
    return std::tanh(x);
};

auto activation_derivative(float x) {
    float tanh2 = activation(2);
    tanh2 *= tanh2;
    return 1 - tanh2;
};

auto random_float(float) {
    static std::random_device rd;
    static std::mt19937 engine(rd());
    static std::uniform_real_distribution<float> dis;
    return dis(engine);
}

template <typename T>
class network
{
public:

    template <typename... Args>
    network(Args... args)
    {
        unsigned sizes[sizeof...(Args)] = {args...};

        _layers.resize(sizeof...(Args) - 1);
        _weights.resize(_layers.size());
        _biases.resize(_layers.size());

        for (unsigned i = 0; i < sizeof...(Args) - 1; ++i)
        {
            _layers[i].resize(sizes[i+1]);
            
            _weights[i].resize(sizes[i+1], sizes[i]);
            blas::map(_weights[i], random_float, _weights[i]);

            _biases[i].resize(_layers[i].size());
            blas::map(_biases[i], random_float, _biases[i]);
        }
    }

    const array<float>&
    forward(const array<float>& in)
    {
        blas::mv(_weights[0], in, _layers[0]);

        for (int i = 1; i < _weights.size(); ++i)
        {
            blas::mv(_weights[i], _layers[i-1], _layers[i]);
            blas::add(_biases[i], _layers[i], _layers[i]);
            blas::map(_layers[i], activation, _layers[i]);
        }

        return _layers[_layers.size() - 1];
    }

    T
    train(const array<T>& in, const array<T>& target, T learning_rate)
    {
        array<T> error = forward(in);
        blas::sub(error, target, error);
        log::info("Norm:", blas::nrm2(error));

        // обратное распространение ошибки
        for (int i = _layers.size() - 1; i >= 0; i--)
        {
            const array<T>&  output = _layers[i];
            const array<T>&  input  = i == 0 ? in : _layers[i - 1];
                  matrix<T>& weights = _weights[i];
                  array<T>&  biases = _biases[i];

            // вычисляем градиент функции потерь по выходу
            array<T> output_gradient;
            blas::map(output, activation_derivative, output_gradient);
            blas::mul(output_gradient, error, output_gradient);

            // вычисляем градиент функции потерь по входу
            //array<T> input_gradient = matrix<T>::dot(weights.transpose(), output_gradient);
            array<T> input_gradient;
            blas::mv(weights, output_gradient, input_gradient, true);

            // обновляем веса и смещения
            blas::ger(input, output_gradient, -learning_rate, weights);
            blas::sub(biases, output_gradient, biases);

            // переходим к следующему слою
            error = input_gradient;
        }

        return blas::nrm2(error);
    } 

protected:

    array<array<T>>  _layers;
    array<matrix<T>> _weights;
    array<array<T>>  _biases;
};

}
} // namespace lm
