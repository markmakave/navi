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
#include "cuda/blas.cuh"

float
relu(float x)
{
    return x > 0 ? x : 0;
}

float
relu_derivative(float x)
{
    return x > 0 ? 1 : 0;
}

float
sigmoid(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

float
sigmoid_derivative(float x)
{
    float s = sigmoid(x);
    return s * (1.f - s);
}

namespace lm {
namespace neural {

template <typename T>
T random(T) {
    static std::random_device rd;
    static std::mt19937 engine(rd());
    static std::normal_distribution<T> dis(0, 0.1);
    return dis(engine);
}

template <typename T>
class network
{
public:

    template <typename... Args>
    network(Args... args)
    {
        int sizes[sizeof...(Args)] = {args...};

        _layers.reshape(sizeof...(Args) - 1);
        _weights.reshape(_layers.size());
        _biases.reshape(_layers.size());

        for (unsigned i = 0; i < sizeof...(Args) - 1; ++i)
        {
            _layers(i).reshape(sizes[i+1]);
            
            _weights(i).reshape(sizes[i], sizes[i+1]);
            blas::map(_weights(i), random<T>, _weights(i));

            _biases(i).reshape(_layers(i).size());
            blas::map(_biases(i), random<T>, _biases(i));
        }
    }

    const array<T>&
    forward(const array<T>& in)
    {
        for (int i = 0; i < _weights.size(); ++i)
        {
            blas::mv(_weights(i), i == 0 ? in : _layers(i-1), _layers(i));
            blas::add(_biases(i), _layers(i), _layers(i));
            blas::map(_layers(i), sigmoid, _layers(i));
        }

        return _layers.back();
    }

    void
    train(
        const array<T>& in, 
        const array<T>& target,
              int epochs,
              T learning_rate
    ) {
        assert(in.size() / _weights.front().shape()[0] == target.size() / _layers.back().size());
        int dataset_size = target.size() / _layers.back().size();

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            array<T> error;
            array<T> gradient;
            array<T> batch_in(_weights.front().shape()[0]), batch_target(_layers.back().size());
            
            for (int batch = 0; batch < dataset_size; ++batch)
            {
                int stride = batch * batch_in.size();
                for (int i = 0; i < batch_in.size(); ++i)
                    batch_in(i) = in(stride + i);

                stride = batch * batch_target.size();
                for (int i = 0; i < batch_target.size(); ++i)
                    batch_target(i) = target(stride + i);

                blas::sub(forward(batch_in), batch_target, error);
                
                // back propogation
                for (int i = _layers.size() - 1; i >= 0; i--)
                {
                    const array<T>&  input   = i == 0 ? batch_in : _layers(i - 1);
                    const array<T>&  output  = _layers(i);
                          matrix<T>& weights = _weights(i);
                          array<T>&  biases  = _biases(i);

                    // calculate gradients
                    blas::map(output, sigmoid_derivative, gradient);
                    blas::mul(error, gradient, gradient);

                    // update weights and biases
                    blas::axpy(-learning_rate, gradient, biases);
                    blas::ger(gradient, input, -learning_rate, weights);

                    // calculate error for previous layer
                    blas::mv(weights, gradient, error, true);
                }

                printf("\rEpoch: %d, Batch: %d", epoch + 1, batch + 1);
            }

            printf("\n");
        }
    }

protected:

    array<array<T>>  _layers;
    array<matrix<T>> _weights;
    array<array<T>>  _biases;
};

}
} // namespace lm
