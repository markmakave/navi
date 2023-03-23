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

#include "lumina.hpp"

namespace lm {
namespace neural {

class network
{
public:

    template <typename... Args>
    network(Args... args)
    {
        unsigned sizes[sizeof...(Args)] = {args...};
        _weights.resize(sizeof...(Args))

        for (unsigned i = 0; i < sizeof...(Args) - 1; ++i)
        {
            _weights[i].resize(sizes[i+1], sizes[i]);
        }
    }

    const array<float>&
    forward(const array<float>& in) const
    {
        cuda::array<float> din, dout;
        din << in;

        for (int i = 0; i < _weights.size(); ++i)
        {
            blas::mv(_weights[i], din, dout);
            din.swap(dout);
        }

        array<float> out;
        dout >> out;

        return out;
    }


protected:

    array<cuda::matrix<float>> _weights;
};

}
} // namespace lm
