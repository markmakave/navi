#pragma once

#include "utility/types.hpp"

namespace lumina
{

template <u64 N>
requires (N > 0)
struct shape
{
public:

    using size_type = u64;

public:

    shape()
    :   _data {}
    {}

    template <typename... Dims>
    requires(sizeof...(Dims) == N)
    shape(Dims... dims)
    :   _data {static_cast<size_type>(dims)...}
    {}

    shape(const shape& s)
    {
        for (size_type n = 0; n < N; ++n)
            _data[n] = s._data[n];
    }

    shape& operator=(const shape& s)
    {
        for (size_type n = 0; n < N; ++n)
            _data[n] = s._data[n];
        return *this;
    }

    bool operator==(const shape& s) const
    {
        for (size_type n = 0; n < N; ++n)
            if (_data[n] != s[n])
                return false;
        return true;
    }

    size_type& operator[](size_type dim)
    {
        return _data[dim];
    }

    const size_type& operator[](size_type dim) const
    {
        return _data[dim];
    }

    size_type volume() const
    {
        size_type s = 1;
        for (size_type n = 0; n < N; ++n)
            s *= _data[n];
        return s;
    }

protected:

    size_type _data[N];
};


// deduction guide for shape
template <typename... Dims>
shape(Dims...) -> shape<sizeof...(Dims)>;


// shape broadcasting
template <u64 N1, u64 N2>
inline auto broadcast (const shape<N1>& lhs, const shape<N2>& rhs)
{
    shape<std::max(N1, N2)> bcast;

    for (u64 n = 0; n < std::min(N1, N2); ++n)
    {
        if (lhs[n] == rhs[n])
            bcast[n] = lhs[n];

        if (lhs[n] == 1)
            bcast[n] = rhs[n];

        if (rhs[n] == 1)
            bcast[n] = lhs[n];
        
        throw std::runtime_error("Shapes are not broadcastable");
    }

    return bcast;
}

}
