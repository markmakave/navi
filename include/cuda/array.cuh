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

#include "base/array.hpp"
#include "cuda/memory.cuh"

namespace lm {
namespace cuda {

template <typename T, typename _alloc = device_allocator<T>>
struct array
{
public:

    typedef T                   value_type;
    typedef value_type*         pointer;
    typedef const value_type*   const_pointer;
    typedef value_type&         reference;
    typedef const value_type&   const_reference;
    typedef value_type*         iterator;
    typedef const value_type*   const_iterator;
    typedef int64_t             size_type;

public:

    __host__ __device__
    array()
    :   _data(nullptr),
        _size(0)
    {}

    __host__ __device__
    array(size_type size)
    :   _size(size)
    {
        _allocate();
    }

    __host__ __device__
    array(const array& a)
    :   _size(a._size)
    {
        _allocate();
        _alloc::copy(_data, a._data, a._size);
    }

    __host__ __device__
    array(array&& a)
    :   _data(a._data),
        _size(a._size)
    {}

    __host__ __device__
    array&
    operator = (const array& a)
    {
        if (&a != this)
        {
            resize(a._size);
            _alloc::copy(_data, a._data, a._size);
        }

        return *this;
    }

    __host__ __device__
    array&
    operator = (array&& a)
    {
        if (&a != this)
        {
            _deallocate();

            _data = a._data;
            _size = a._size;

            a._data = nullptr;
            a._size = 0;
        }

        return *this;
    }

    __host__ __device__
    void
    resize(size_type size)
    {
        if (size != _size)
        {
            _deallocate();
            _size = size;
            _allocate();
        }
    }

    __host__ __device__
    size_type
    size() const
    {
        return _size;
    }

    __host__ __device__
    const_pointer
    data() const
    {
        return _data;
    }

    __host__ __device__
    pointer
    data()
    {
        return _data;
    }

    __host__ __device__
    decltype(auto)
    operator [] (size_type index) const
    {
        return _alloc::access(_data + index);
    }

    __host__ __device__
    decltype(auto)
    operator [] (size_type index)
    {
        return _alloc::access(_data + index);
    }

    __host__
    void
    operator << (const lm::array<value_type>& a)
    {
        if (_size != a.size())
            resize(a.size());

        memcpy(_data, a.data(), a.size() * sizeof(value_type), D2H);
    }

    __host__
    void
    operator >> (lm::array<float>& a) const
    {
        if (a.size() != _size)
            a.resize(_size);

        memcpy(a.data(), _data, _size * sizeof(value_type), H2D);
    }

private:

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

protected:

    pointer _data;
    size_type _size;
};

}
}
