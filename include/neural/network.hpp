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

#include <random>

#include "base/matrix.hpp"
#include "base/array.hpp"

// #include "cuda/matrix.cuh"
// #include "cuda/array.cuh"

#include "base/blas.hpp"
#include "cuda/blas.cuh"

#include "neural/layer.hpp"

namespace lm {
namespace neural {

static
float random(float) {
    static std::random_device rd;
    static std::mt19937 engine(rd());
    static std::normal_distribution<float> dis(0, 0.1);
    return dis(engine);
}

class network
{
public:

    using type = float;
    using size_type = i64;

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
            blas::map(_weights(i), random, _weights(i));

            _biases(i).reshape(_layers(i).size());
            blas::map(_biases(i), random, _biases(i));
        }
    }

    network(const char* filename)
    {
        read(filename);
    }

    const array<type>&
    forward(const array<type>& in)
    {
        for (int i = 0; i < _weights.size(); ++i)
        {
            blas::mv(_weights(i), i == 0 ? in : _layers(i-1), _layers(i));
            blas::add(_biases(i), _layers(i), _layers(i));
            blas::map(_layers(i), activation, _layers(i));
        }

        return _layers.back();
    }

    void
    train(
        const array<type>& in, 
        const array<type>& target,
              int           epochs,
              type         learning_rate
    ) {
        assert(in.size() / _weights.front().shape()[0] == target.size() / _layers.back().size());
        int dataset_size = target.size() / _layers.back().size();

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            array<type> error;
            array<type> gradient;
            array<type> batch_in(_weights.front().shape()[0]), batch_target(_layers.back().size());
            
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
                    const array<type>&  input   = i == 0 ? batch_in : _layers(i - 1);
                    const array<type>&  output  = _layers(i);
                          matrix<type>& weights = _weights(i);
                          array<type>&  biases  = _biases(i);

                    // calculate gradients
                    blas::map(output, activation_derivative, gradient);
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

    size_type
    in_size() const
    {
        return _weights.front().shape()[0];
    }

    size_type
    out_size() const
    {
        return _weights.back().shape()[1];
    }

    void
    read(const char* filename)
    {
        std::ifstream file(filename);
        if (!file)
            log::error("No such model file:", filename);

        size_type model_size;
        file.read((char*)&model_size, sizeof(model_size));
        _layers.reshape(model_size);
        _weights.reshape(model_size);
        _biases.reshape(model_size);

        for (size_type l = 0; l < model_size; ++l)
        {
            auto& weights = _weights(l);
            auto& biases = _biases(l);

            size_type in_size, out_size;
            file.read((char*)&in_size, sizeof(in_size));
            file.read((char*)&out_size, sizeof(out_size));

            weights.reshape(in_size, out_size);
            biases.reshape(out_size);

            file.read((char*)weights.data(), weights.size() * sizeof(float));
            file.read((char*)biases.data(), biases.size() * sizeof(float));
        }
    }

    void
    write(const char* filename) const
    {
        std::ofstream file(filename);

        size_type model_size = _layers.size();
        file.write((char*)&model_size, sizeof(model_size));

        for (size_type l = 0; l < model_size; ++l)
        {
            const auto& weights = _weights(l);
            const auto& biases = _biases(l);

            size_type in_size = weights.shape()[0], out_size = weights.shape()[1];
            file.write((char*)&in_size, sizeof(in_size));
            file.write((char*)&out_size, sizeof(out_size));
            file.write((char*)weights.data(), weights.size() * sizeof(float));
            file.write((char*)biases.data(), biases.size() * sizeof(float));
        }
    }

protected:

    array<array<type>>  _layers;
    array<matrix<type>> _weights;
    array<array<type>>  _biases;

    float (*activation)(float) = tanh;
    float (*activation_derivative)(float) = tanh_derivative;
};

}
} // namespace lm
