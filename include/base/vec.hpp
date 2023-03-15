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
#include <cmath>

#include "base/array.hpp"

namespace lm {

template <unsigned N, typename T = float>
struct vec : public array<T, N>
{
public:

    typedef T                   value_type;
    typedef value_type*         pointer;
    typedef const value_type*   const_pointer;
    typedef value_type&         reference;
    typedef const value_type&   const_reference;
    typedef value_type*         iterator;
    typedef const value_type*   const_iterator;
    typedef unsigned            size_type;
    typedef int                 difference_type;

public:

    vec()
    :   array<T, N>()
    {}

    vec(const vec<N, T>& v)
    :   array<T, N>(v)
    {}

    vec(const T& v)
    :   array<T, N>(v)
    {}

    vec(const T* v)
    :   array<T, N>(v)
    {}

    vec(const array<T, N>& v)
    :   array<T, N>(v)
    {}

    vec(const T& x, const T& y)
    {
        this->_data[0] = x;
        this->_data[1] = y;

        for (unsigned i = 2; i < N; i++)
            this->_data[i] = 0;
    }

    vec(const T& x, const T& y, const T& z)
    {
        this->_data[0] = x;
        this->_data[1] = y;
        this->_data[2] = z;

        for (unsigned i = 3; i < N; i++)
            this->_data[i] = 0;
    }

    vec(const T& x, const T& y, const T& z, const T& w)
    {
        this->_data[0] = x;
        this->_data[1] = y;
        this->_data[2] = z;
        this->_data[3] = w;

        for (unsigned i = 4; i < N; i++)
            this->_data[i] = 0;
    }

    reference
    x()
    {
        return this->_data[0];
    }

    const_reference
    x() const
    {
        return this->_data[0];
    }

    reference
    y()
    {
        return this->_data[1];
    }

    const_reference
    y() const
    {
        return this->_data[1];
    }

    reference
    z()
    {
        return this->_data[2];
    }

    const_reference
    z() const
    {
        return this->_data[2];
    }

    reference
    w()
    {
        return this->_data[3];
    }

    const_reference
    w() const
    {
        return this->_data[3];
    }

    value_type
    length() const
    {
        value_type l = 0;

        for (unsigned i = 0; i < N; i++)
            l += this->_data[i] * this->_data[i];

        return std::sqrt(l);
    }

    friend
    value_type
    dot(const vec& v1, const vec& v2)
    {
        value_type d = 0;

        for (unsigned i = 0; i < N; i++)
            d += v1[i] * v2[i];

        return d;
    }

    friend
    std::ostream&
    operator << (std::ostream& os, const vec& v)
    {
        os << "(";

        for (unsigned i = 0; i < N; i++)
        {
            os << v[i];

            if (i < N - 1)
                os << ", ";
        }

        os << ")";

        return os;
    }

    vec&
    normalize()
    {
        return *this /= this->length();
    }

    friend
    vec
    normalize(const vec& v)
    {
        return v / v.length();
    }

    vec&
    operator = (const vec& v)
    {
        for (unsigned i = 0; i < N; i++)
            this->_data[i] = v[i];
        
        return *this;
    }

    vec&
    operator = (const array<T, N>& v)
    {
        for (unsigned i = 0; i < N; i++)
            this->_data[i] = v[i];
        
        return *this;
    }

    vec&
    operator += (const vec& v)
    {
        for (unsigned i = 0; i < N; i++)
            this->_data[i] += v[i];

        return *this;
    }

    vec&
    operator -= (const vec& v)
    {
        for (unsigned i = 0; i < N; i++)
            this->_data[i] -= v[i];

        return *this;
    }

    vec&
    operator *= (const vec& v)
    {
        for (unsigned i = 0; i < N; i++)
            this->_data[i] *= v[i];

        return *this;
    }

    vec&
    operator /= (const vec& v)
    {
        for (unsigned i = 0; i < N; i++)
            this->_data[i] /= v[i];

        return *this;
    }

    vec
    operator + (const vec& v) const
    {
        vec<N, T> r;

        for (unsigned i = 0; i < N; i++)
            r[i] = this->_data[i] + v[i];

        return r;
    }

    vec
    operator - (const vec& v) const
    {
        vec<N, T> r;

        for (unsigned i = 0; i < N; i++)
            r[i] = this->_data[i] - v[i];

        return r;
    }

    vec
    operator * (const vec& v) const
    {
        vec<N, T> r;

        for (unsigned i = 0; i < N; i++)
            r[i] = this->_data[i] * v[i];

        return r;
    }

    vec
    operator / (const vec& v) const
    {
        vec<N, T> r;

        for (unsigned i = 0; i < N; i++)
            r[i] = this->_data[i] / v[i];

        return r;
    }

    vec
    operator - () const
    {
        vec<N, T> r;

        for (unsigned i = 0; i < N; i++)
            r[i] = -this->_data[i];

        return r;
    }

    vec
    operator + () const
    {
        return *this;
    }

    vec
    operator * (const T& s) const
    {
        vec<N, T> r;

        for (unsigned i = 0; i < N; i++)
            r[i] = this->_data[i] * s;

        return r;
    }

    vec
    operator / (const T& s) const
    {
        vec<N, T> r;

        for (unsigned i = 0; i < N; i++)
            r[i] = this->_data[i] / s;

        return r;
    }

    vec&
    operator *= (const T& s)
    {
        for (unsigned i = 0; i < N; i++)
            this->_data[i] *= s;

        return *this;
    }

    vec&
    operator /= (const T& s)
    {
        for (unsigned i = 0; i < N; i++)
            this->_data[i] /= s;

        return *this;
    }

};

template <typename T>
inline
vec<2, T>
cross(const vec<2, T>& v1, const vec<2, T>& v2)
{
    return vec<2, T>(v1.x() * v2.y() - v1.y() * v2.x());
}

template <typename T>
inline
vec<3, T>
cross(const vec<3, T>& v1, const vec<3, T>& v2)
{
    return vec<3, T>(v1.y() * v2.z() - v1.z() * v2.y(),
                     v1.z() * v2.x() - v1.x() * v2.z(),
                     v1.x() * v2.y() - v1.y() * v2.x());
}

typedef vec<2> vec2;
typedef vec<3> vec3;
typedef vec<4> vec4;

typedef vec<2, int> dim2;
typedef vec<3, int> dim3;
typedef vec<4, int> dim4;

}
