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
#include <functional>

#include "base/memory.hpp"

namespace lm {

template <typename T, typename _alloc = heap_allocator<T>>
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

    array()
    :   _data(nullptr),
        _size(0)
    {}

    array(size_type size)
    :   _size(size)
    {
        _allocate();
    }

    array(const array& a)
    :   _size(a._size)
    {
        _allocate();

        for (size_type i = 0; i < _size; ++i)
            _data[i] = a._data[i];
    }

    template <typename U>
    array(const array<U>& a)
    :   _size(a.size())
    {
        _allocate();

        for (size_type i = 0; i < _size; ++i)
            _data[i] = a[i];
    }

    array(pointer data, size_type size)
    :   _data(data),
        _size(size)
    {}

    array(array&& other)
    :   _data(other._data),
        _size(other._size)
    {
        other._data = nullptr;
        other._size = 0;
    }

    ~array()
    {
        _deallocate();
    }

    array&
    operator = (const array& a)
    {
        if (&a != this)
        {
            resize(a._size);

            _alloc::copy(a._data, _data, a._size);
        }

        return *this;
    }

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

    reference
    operator [] (size_type index)
    {
        return _data[index];
    }

    const_reference
    operator [] (size_type index) const
    {
        return _data[index];
    }

    void
    fill(const_reference fillament)
    {
        for (size_type i = 0; i < _size; ++i)
            _data[i] = fillament;
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
        return _size;
    }

    friend
    std::ostream&
    operator << (std::ostream& os, const array& a)
    {
        os << "[";
        for (size_type i = 0; i < a._size; ++i)
        {
            os << a._data[i];
            if (i < a._size - 1) os << ", ";
        }
        os << "]";

        return os;
    }

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

private:

    void
    _allocate()
    {
        _data = _alloc::allocate(_size);
    }

    void
    _deallocate()
    {
        _alloc::deallocate(_data);
    }

protected:

    pointer     _data;
    size_type   _size;

};

}
