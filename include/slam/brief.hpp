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

#include <array>
#include <bitset>
#include <random>

#include "base/matrix.hpp"
#include "base/color.hpp"

namespace lm {
namespace slam {

struct feature;

struct dim2
{
    float x, y;
};

template <unsigned N>
class brief
{

    unsigned radius;
    std::array<std::pair<dim2, dim2>, N> net;

public:

    typedef std::bitset<N> descriptor;

    brief(unsigned radius)
    :   radius(radius)
    {

        std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count());
        std::binomial_distribution<int> distribution(radius * 2, 0.5);

        for (auto& pair : net)
        {
            pair.first.x() = distribution(engine) - radius;
            pair.first.y() = distribution(engine) - radius;

            pair.second.x() = distribution(engine) - radius;
            pair.second.y() = distribution(engine) - radius;
        }
    }

    void
    compute(const matrix<gray> &frame, feature& f) const;

    static
    unsigned
    distance(const descriptor& d1, const descriptor& d2)
    {
        return (d1 ^ d2).count();
    }
};

struct feature
{
    unsigned x, y;
    brief<256>::descriptor descriptor = 0;
};

template <unsigned N>
void
brief<N>::compute(const matrix<gray> &frame, feature& f) const
{
    if (f.x < radius || f.x >= frame.shape()[0] - radius || f.y < radius || f.y >= frame.shape()[1] - radius)
        return;

    for (unsigned i = 0; i < N; ++i)
    {
        auto& pair = net[i];
        auto& p1 = pair.first;
        auto& p2 = pair.second;

        auto& v1 = frame[f.y + p1.y()][f.x + p1.x()];
        auto& v2 = frame[f.y + p2.y()][f.x + p2.x()];

        f.descriptor[i] = v1 < v2;
    }
}

} // namespace slam
} // namespace lm
