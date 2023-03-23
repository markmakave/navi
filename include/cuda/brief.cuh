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

#include <iostream>
#include <random>
#include <cstdint>

#include "cuda/array.cuh"
#include "cuda/matrix.cuh"
#include "base/color.hpp"

namespace lm {
namespace cuda {

struct point_pair
{
    int x1, x2, y1, y2;
};

template <unsigned N>
class brief
{
public:

    struct descriptor
    {
        uint64_t value[4] = {};
    };

    point_pair* net;

public:

    __host__
    brief()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::binomial_distribution<int> dist(50, 0.5);

        cuda::managed_allocator<point_pair>::allocate(N);

        for (int i = 0; i < 256; ++i)
        {
            auto& pair = net[i];

            do {
                pair.x1 = dist(gen) - 25;
            } while (pair.x1 >= 25 || pair.x1 <= -25);

            do {
                pair.y1 = dist(gen) - 25;
            } while (pair.y1 >= 25 || pair.y1 <= -25);

            do {
                pair.x2 = dist(gen) - 25;
            } while (pair.x2 >= 25 || pair.x2 <= -25);

            do {
                pair.y2 = dist(gen) - 25;
            } while (pair.y2 >= 25 || pair.y2 <= -25);
        }
    }

    __device__
    descriptor
    descript(int x, int y, const matrix<gray>& image) const
    {
        descriptor desc;

        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 64; ++j)
            {
                int index = i * 64 + j; 

                int val1 = image[y + net[index].y1][x + net[index].x1];
                int val2 = image[y + net[index].y2][x + net[index].x2];

                desc.value[i] = desc.value[i] & ((val1 > val2) << j);
            }

        return desc;
    }

    __host__
    ~brief()
    {
        cuda::managed_allocator<point_pair>::deallocate(net);
    }

    // __host__
    // image<rgb>
    // draw() const
    // {
    //     image<rgb> image(500, 500, 0);

    //     for (int i = 0; i < 256; ++i)
    //     {
    //         const auto& pair = net[i];

    //         image.line(pair.x1 * 10 + 250, pair.y1 * 10 + 250, pair.x2 * 10 + 250, pair.y2 * 10 + 250, rgb::random());
    //     }

    //     return image;
    // }

};

}
}
