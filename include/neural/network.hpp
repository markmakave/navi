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

#include "cuda/matrix.cuh"
#include "cuda/array.cuh"

#include "base/blas.hpp"

namespace lm {
namespace neural {

class network
{
public:

    template <typename... Args>
    network(Args... args)
    {
        unsigned sizes[sizeof...(Args)] = {args...};

        _layers.resize(sizeof...(Args));
        _weights.resize(sizeof...(Args));

        for (unsigned i = 0; i < sizeof...(Args) - 1; ++i)
        {
            _layers[i].resize(sizes[i+1]);
            _weights[i].resize(sizes[i+1], sizes[i]);
        }
    }

    const cuda::array<float>&
    forward(const cuda::array<float>& in)
    {
        blas::mv(_weights[0], in, _layers[0]);

        for (int i = 1; i < _weights.size(); ++i)
        {
            blas::mv(_weights[i], _layers[i-1], _layers[i]);
        }

        return _layers.back();
    }

    float
    train(const cuda::array<float>& in, const cuda::array<float>& out)
    {
        forward(in);

        blas::axpy(-1.f, out, _layers.back());

        float error;
        blas::nrm2(_layers.back(), error);
    
        return error;
    }

protected:

    array<cuda::array<float>> _layers;
    array<cuda::matrix<float>> _weights;
};

}
} // namespace lm
