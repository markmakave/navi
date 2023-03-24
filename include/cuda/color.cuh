#pragma once

#include "base/types.hpp"

namespace lm {
namespace cuda {

template <int lower, int upper>
__host__ __device__
static inline
int
clamp(int value)
{
    if (value > upper) return upper;
    if (value < lower) return lower;
    return value;
}

struct rgb
{
    byte r, g, b;

    __host__ __device__
    rgb()
    :   r(0),
        g(0),
        b(0)
    {}

    __host__ __device__
    rgb(gray g)
    :   r(g),
        g(g),
        b(g)
    {}

    __host__ __device__
    rgb(byte r, byte g, byte b)
    :   r(r),
        g(g),
        b(b)
    {}

    __host__
    rgb(lm::rgb color)
    :   r(color.r),
        g(color.g),
        b(color.b)
    {}

    __host__ __device__
    operator gray() const
    {
        return clamp<0, 255>(0.2161 * r + 0.7152 * g + 0.0722 * b);
    }

    __host__
    operator lm::rgb() const
    {
        return {r, g ,b};
    }
};

struct rgba : rgb
{
    byte a;

    __host__ __device__
    rgba()
    :   rgb(),
        a(255)
    {}

    __host__ __device__
    rgba(gray g)
    :   rgb(g),
        a(255)
    {}

    __host__ __device__
    rgba(byte r, byte g, byte b, byte a = 255)
    :   rgb(r, g, b),
        a(a)
    {}

    __host__ __device__
    operator gray () const
    {
        return clamp<0, 255>(0.2161 * r + 0.7152 * g + 0.0722 * b);
    }
};

}
}
