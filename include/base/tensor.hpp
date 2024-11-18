/*

    Copyright (c) 2023 Mokhov Mark

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

*/

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <functional>

#include <cassert>
#include <cmath>

#include "base/types.hpp"
#include "base/memory.hpp"
#include "util/utility.hpp"

namespace lumina {

template <i64 N>
struct shape
{
public:

    using size_type = i64;

public:

    shape()
      : _data {}
    {}

    template <typename... Size>
    shape(Size... sizes)
      : _data {static_cast<size_type>(sizes)...}
    {
        static_assert(sizeof...(Size) == N);
    }

    shape(const shape& s)
    {
        for (size_type n = 0; n < N; ++n)
            _data[n] = s._data[n];
    }

    shape&
    operator=(const shape& s)
    {
        for (size_type n = 0; n < N; ++n)
            _data[n] = s._data[n];
        return *this;
    }

    bool
    operator==(const shape& s) const
    {
        for (size_type n = 0; n < N; ++n)
            if (_data[n] != s[n])
                return false;
        return true;
    }

    bool
    operator!=(const shape& s) const
    {
        return !((*this) == s);
    }

    size_type&
    operator[](size_type dim)
    {
        return _data[dim];
    }

    const size_type&
    operator[](size_type dim) const
    {
        return _data[dim];
    }

    size_type
    volume() const
    {
        size_type s = 1;
        for (size_type n = 0; n < N; ++n)
            s *= _data[n];
        return s;
    }

protected:

    size_type _data[N];
};

template <i64 N, typename T, typename _alloc = heap_allocator<T>>
class tensor
{
public:

    using value_type      = T;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using iterator        = pointer;
    using const_iterator  = const_pointer;
    using size_type       = decltype(N);
    using shape_type      = lumina::shape<N>;

public:

    tensor()
      : _data(nullptr),
        _shape()
    {}

    template <typename... Size>
    tensor(Size... sizes)
      : _data(nullptr),
        _shape()
    {
        reshape(sizes...);
    }

    tensor(const shape_type& shape)
      : _data(nullptr),
        _shape()
    {
        reshape(shape);
    }

    tensor(const tensor& t)
      : _data(nullptr),
        _shape()
    {
        reshape(t.shape());
        _alloc::copy(t._data, this->_data, this->size());
    }

    tensor(tensor&& t)
      : _data(t._data),
        _shape(t._shape)
    {
        t._data  = nullptr;
        t._shape = shape_type();
    }

    template <typename U, typename alloc>
    tensor(const tensor<N, U, alloc>& t)
      : _data(nullptr),
        _shape()
    {
        reshape(t.shape());
        for (size_type i = 0; i < t.size(); ++i)
            _alloc::access(this->_data + i) = static_cast<value_type>(t[i]);
    }

    ~tensor()
    {
        _deallocate();
    }

    tensor&
    operator=(const tensor& t)
    {
        if (&t != this) {
            reshape(t._shape);
            _alloc::copy(t._data, this->_data, t.size() * sizeof(value_type));
        }

        return *this;
    }

    tensor&
    operator=(tensor&& t)
    {
        if (&t != this) {
            this->_data  = t._data;
            this->_shape = t._shape;

            t._data  = nullptr;
            t._shape = shape_type();
        }

        return *this;
    }

    template <typename U, typename alloc>
    tensor&
    operator=(const tensor<N, U, alloc>& t)
    {
        reshape(t.shape());
        for (size_type i = 0; i < t.size(); ++i)
            _alloc::access(this->_data + i) =
                static_cast<value_type>(t.data()[i]);

        return *this;
    }

    void
    reshape(const shape_type& shape)
    {
        if (this->_shape == shape)
            return;

        _deallocate();
        this->_shape = shape;
        _allocate();
    }

    template <typename... Size>
    void
    reshape(Size... sizes)
    {
        static_assert(sizeof...(Size) == N);

        shape_type new_shape(sizes...);
        reshape(new_shape);
    }

    void
    fill(const_reference fillament)
    {
        for (size_type i = 0; i < size(); ++i)
            _data[i] = fillament;
    }

    const shape_type&
    shape() const
    {
        return _shape;
    }

    size_type
    shape(size_type dimension) const
    {
        return _shape[dimension];
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
        return _shape.volume();
    }

    reference
    operator()(size_type indices[N])
    {
        size_type offset = 0;
        size_type dim    = 1;
        for (size_type n = 0; n < N; ++n) {
            offset += dim * indices[n];
            dim *= _shape[n];
        }

        return _data[offset];
    }

    const_reference
    operator()(size_type indices[N]) const
    {
        size_type offset = 0;
        size_type dim    = 1;
        for (size_type n = 0; n < N; ++n) {
            offset += dim * indices[n];
            dim *= _shape[n];
        }

        return _data[offset];
    }

    template <typename... Index>
    requires (sizeof...(Index) == N)
    reference
    operator()(Index... index)
    {
        size_type indices[N] = {static_cast<size_type>(index)...};
        return (*this)(indices);
    }

    template <typename... Index>
    requires (sizeof...(Index) == N)
    const_reference
    operator()(Index... index) const
    {
        size_type indices[N] = {static_cast<size_type>(index)...};
        return (*this)(indices);
    }

#if __cplusplus >= 202002L

    template <typename... Index>
    reference
    operator[](Index... index)
    {
        return operator()(index...);
    }

    template <typename... Index>
    const_reference
    operator[](Index... index) const
    {
        return operator()(index...);
    }

#endif

    reference
    operator[](size_type index)
    {
        return _data[index];
    }

    const_reference
    operator[](size_type index) const
    {
        return _data[index];
    }

    template <typename... Index>
    requires (sizeof...(Index) == N)
    reference
    at(Index... index)
    {
        size_type indices[N] = {index...};

        for (size_type n = 0; n < N; ++n)
            if (indices[n] < 0 or indices[n] >= _shape[n]) {
                static value_type trash;
                trash = {};
                return trash;
            }

        return operator()(indices);
    }

    template <typename... Index>
    requires (sizeof...(Index) == N)
    const_reference
    at(Index... index) const
    {
        size_type indices[N] = {index...};

        for (size_type n = 0; n < N; ++n)
            if (indices[n] < 0 or indices[n] >= _shape[n]) {
                static value_type trash;
                trash = {};
                return trash;
            }

        return operator()(indices);
    }

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

    iterator
    begin()
    {
        return _data;
    }

    const_iterator
    begin() const
    {
        return _data;
    }

    iterator
    end()
    {
        return _data + size();
    }

    const_iterator
    end() const
    {
        return _data + size();
    }

    friend std::ostream&
    operator<<(std::ostream& out, const tensor& t)
    {
        out << "Tensor<" << typeid(value_type).name() << "> " << t._shape[0];
        for (size_type n = 1; n < N; ++n)
            out << "×" << t._shape[n];
        out << '\n';

        size_type max_index = 0;
        for (size_type i = 0; i < t.size(); ++i)
            if (t._data[i] > t._data[max_index])
                max_index = i;

        int number_length = std::log10(t._data[max_index]) + 1;

        size_type index[N] = {};

        std::function<void(size_type)> print;
        print = [&](size_type dim) {
            if (dim == 0) {
                for (size_type n = 0; n < t.shape()[0]; ++n) {
                    index[0] = n;
                    out << std::setw(number_length + 1) << t(index) << ' ';
                }
            } else {
                for (size_type n = 0; n < t.shape()[dim]; ++n) {
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
    write(std::ofstream& file) const
    {
        size_type order = N;
        file.write((char*)&order, sizeof(order));
        for (size_type n = 0; n < N; ++n)
            file.write((char*)&_shape[n], sizeof(_shape[n]));

        file.write((char*)_data, size() * sizeof(value_type));
    }

    void
    write(const char* filename) const
    {
        std::ofstream file(filename);
        write(file);
    }

    void
    read(std::ifstream& file)
    {
        size_type order;
        file.read((char*)&order, sizeof(order));

        assert(order == N);

        shape_type shape;
        for (size_type n = 0; n < N; ++n)
            file.read((char*)&shape[n], sizeof(shape[n]));

        reshape(shape);

        file.read((char*)_data, size() * sizeof(value_type));
    }

    void
    read(const char* filename)
    {
        std::ifstream file(filename);
        read(file);
    }


    tensor<N, bool> operator== (const tensor& other) const
    {
        if (_shape != other._shape)
            throw std::runtime_error("shape mismatch");

        tensor<N, bool> mask(_shape);
        const size_type __size = size();
        for (size_type i = 0; i < __size; ++i)
            mask._data[i] = (this->_data[i] == other._data[i]);

        return mask;
    }


    tensor<N, bool> operator!= (const tensor& other) const
    {
        if (_shape != other._shape)
            throw std::runtime_error("shape mismatch");

        tensor<N, bool> mask(_shape);
        const size_type __size = size();
        for (size_type i = 0; i < __size; ++i)
            mask._data[i] = (this->_data[i] != other._data[i]);

        return mask;
    }

    
    size_type count() const
    requires (std::is_same_v<value_type, bool>)
    {
        size_type result = 0;
        for (const auto& x : *this)
            if (x)
                ++result;

        return result;
    }

protected:

    pointer    _data;
    shape_type _shape;

protected:

    void
    _allocate()
    {
        this->_data = _alloc::allocate(this->size());
    }

    void
    _deallocate()
    {
        _alloc::deallocate(this->_data);
    }
};

} // namespace lumina
