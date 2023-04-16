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

#include "base/types.hpp"
#include "util/log.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <functional>
#include <cmath>

namespace lm {

template <i64 N>
struct shape_t
{
public:

    using size_type = i64;

public:

    shape_t()
    :   _data{}
    {}

    template <typename... Size>
    shape_t(Size ...sizes)
    :   _data{static_cast<size_type>(sizes)...}
    {
        static_assert(sizeof...(Size) == N);
    }

    shape_t(const shape_t& s)
    {
        for (size_type n = 0; n < N; ++n)
            _data[n] = s._data[n];
    }

    shape_t&
    operator = (const shape_t& s)
    {
        for (size_type n = 0; n < N; ++n)
            _data[n] = s._data[n];
        return *this;
    }

    bool
    operator == (const shape_t& s) const
    {
        for (size_type n = 0; n < N; ++n)
            if (_data[n] != s[n])
                return false;
        return true;
    }

    bool
    operator != (const shape_t& s) const
    {
        return !((*this) == s);
    }

    size_type&
    operator [] (size_type dim)
    {
        return _data[dim];
    }

    size_type
    operator [] (size_type dim) const
    {
        return _data[dim];
    }

private:

    size_type _data[N];
};

template <i64 N, typename T, typename _alloc>
class tensor
{
public:

    using value_type        = T;
    using pointer           = value_type*;
    using const_pointer     = const value_type*;
    using reference         = value_type&;
    using const_reference   = const value_type&;
    using iterator          = pointer;
    using const_iterator    = const_pointer;
    using shape_type        = shape_t<N>;
    using size_type         = typename shape_type::size_type;

public:

    tensor()
    :   _data(nullptr),
        _shape()
    {}
    
    template <typename... Size>
    tensor(Size ...sizes)
    :   _data(nullptr),
        _shape()
    {
        reshape(sizes...);
    }

    tensor(const shape_type& shape)
    :   _data(nullptr),
        _shape()
    {
        reshape(shape);
    }

    tensor(const tensor& t)
    :   _data(nullptr),
        _shape()
    {
        reshape(t.shape());
        _alloc::copy(t._data, _data, size()); 
    }

    tensor(tensor&& t)
    :   _data(t._data),
        _shape(t._shape)
    {
        t._data = nullptr;
        t._shape = shape_type();
    }

    ~tensor()
    {
        _deallocate();
    }

    tensor&
    operator = (const tensor& t)
    {
        if (&t != this)
        {
            reshape(t._shape);
            _alloc::copy(t._data, _data, t.size() * sizeof(value_type));
        }

        return *this;
    }

    tensor&
    operator = (tensor&& t)
    {
        if (&t != this)
        {
            _data = t._data;
            _shape = t._shape;

            t._data = nullptr;
            t._shape = shape_type();
        }

        return *this;
    }

    template <typename U, typename alloc>
    tensor&
    operator = (const tensor<N, U, alloc>& t)
    {
        reshape(t.shape());
        for (size_type i = 0; i < t.size(); ++i)
            _alloc::access(_data + i) = static_cast<value_type>(t.data()[i]);

        return *this;
    }

    void
    fill(const_reference fillament)
    {
        for (size_type i = 0; i < size(); ++i)
            _alloc::access(_data + i) = fillament;

    }

    const shape_type&
    shape() const
    {
        return _shape;
    }

    pointer
    data()
    {
        return _data;
    }

    const_pointer
    data() const
    {
        return _data;
    }

    size_type
    size() const
    {
        size_type s = 1;
        for (size_type n = 0; n < N; ++n)
            s *= _shape[n];
        return s;
    }

    template <typename... Size>
    void
    reshape(Size ...sizes)
    {
        static_assert(sizeof...(Size) == N);

        shape_type new_shape(sizes...);
        reshape(new_shape);
    }

    void
    reshape(const shape_type& shape)
    {
        if (_shape == shape)
            return;

        _deallocate();
        _shape = shape;
        _allocate();
    }

    reference
    operator () (size_type indices[N])
    {
        size_type offset = 0;
        size_type dim = 1;
        for (size_type n = 0; n < N; ++n)
        {
            offset += dim * indices[n];
            dim *= _shape[n];
        }

        return _alloc::access(_data + offset);
    }

    reference
    operator () (size_type indices[N]) const
    {
        size_type offset = 0;
        size_type dim = 1;
        for (size_type n = 0; n < N; ++n)
        {
            offset += dim * indices[n];
            dim *= _shape[n];
        }

        return _alloc::access(_data + offset);
    }

    template <typename... Index>
    reference
    operator () (Index ...index)
    {
        static_assert(sizeof...(Index) == N);

        size_type indices[N] = {index...};

        return (*this)(indices);
    }

    template <typename... Index>
    const_reference
    operator () (Index ...index) const
    {
        static_assert(sizeof...(Index) == N);

        size_type indices[N] = {index...};

        return (*this)(indices);
    }

    #if __cplusplus >= 202002L

    template <typename... Index>
    reference
    operator [] (Index ...index)
    {
        return operator()(index...);
    }

    template <typename... Index>
    const_reference
    operator [] (Index ...index) const
    {
        return operator()(index...);
    }

    #endif

    reference
    front()
    {
        return _data[0];
    }

    const_reference
    front() const
    {
        return _data[0];
    }

    reference
    back()
    {
        return _data[size() - 1];
    }

    const_reference
    back() const
    {
        return _data[size() - 1];
    }

    friend
    std::ostream&
    operator << (std::ostream& out, const tensor& t)
    {
        out << "Tensor<" << typeid(value_type).name() << "> " << t._shape[0];
        for (size_type n = 1; n < N; ++n)
            out << "×" << t._shape[n];
        out << '\n';

        value_type max_index = 0;
        for (size_type i = 0; i < t.size(); ++i)
            if (t._data[i] > t._data[max_index])
                max_index = i;

        int number_length = std::log10(t._data[max_index]) + 1;

        size_type index[N] = {};

        std::function<void(size_type)> print;
        print = [&](size_type dim){
            if (dim == 0)
            {
                for (size_type n = 0; n < t.shape()[0]; ++n)
                {
                    index[0] = n;
                    out << std::setw(number_length + 1) << t(index) << ' ';
                }
            } else {
                for (size_type n = 0; n < t.shape()[dim]; ++n)
                {
                    index[dim] = n;

                    if (n != 0)
                        out << std::setw(N - dim) << ' ';
                    out << "[";

                    print(dim - 1);

                    out << "]";
                    if (n != t.shape()[dim] - 1)
                        out << '\n';
                }
            }
        };

        out << "[";
        print(N - 1);
        out << "]";

        return out;
    }

    void
    write(const char* filename)
    {
        std::ofstream file(filename);
        size_type order = N;
        file.write((char*)&order, sizeof(order));
        for (size_type n = 0; n < N; ++n)
            file.write((char*)&_shape[n], sizeof(_shape[n]));

        file.write((char*)_data, size() * sizeof(value_type));
    }

    void
    read(const char* filename)
    {
        std::ifstream file(filename);
        size_type order;
        file.read((char*)&order, sizeof(order));

        assert(order == N);

        shape_type shape;
        for (size_type n = 0; n < N; ++n)
            file.read((char*)&shape[n], sizeof(shape[n]));

        reshape(shape);

        file.read((char*)_data, size() * sizeof(value_type));
    }

protected:

    pointer   _data;
    shape_type _shape;

protected:

    void
    _allocate()
    {
        _data = _alloc::allocate(size());
    }

    void
    _deallocate()
    {
        _alloc::deallocate(_data);
    }
};

}
