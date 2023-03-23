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

#include "base/memory.hpp"

#include <cstdint>

namespace lm {

/// @brief Matrix class with basic operations and STL-like interface
/// @tparam T Type of the elements in the matrix (int, float, double, etc.)
template <typename T, typename _alloc = heap_allocator<T>>
struct matrix
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

    matrix()
    :   _data(nullptr),
        _height(0),
        _width(0)
    {}

    matrix(size_type height, size_type width) 
    :   _height(height),
        _width(width)
    {
        _allocate();
    }

    matrix(size_type height, size_type width, const_reference value)
    :   _height(height),
        _width(width)
    {
        _allocate();
        fill(value);
    }

    matrix(const matrix& m) 
    :   _height(m._height),
        _width(m._width)
    {
        _allocate();

        size_type size = this->size();
        for (size_type i = 0; i < size; ++i)
            _data[i] = m._data[i];
    }

    template <typename U>
    matrix(const matrix<U>& m)
    :   _height(m.height()),
        _width(m.width())
    {
        _allocate();

        size_type size = this->size();
        for (size_type i = 0; i < size; ++i)
            _data[i] = static_cast<value_type>(m.data()[i]);
    }

    matrix(matrix&& m) 
    :   _data(m._data),
        _height(m._height),
        _width(m._width)
    {
        m._data = nullptr;
        m._height = 0;
        m._width = 0;
    }
    
    ~matrix()
    {
        _deallocate();
    }

    matrix& 
    operator = (const matrix& m)
    {
        if (&m != this)
        {
            resize(m._height, m._width);
            
            size_type size = this->size();
            for (size_type i = 0; i < size; ++i)
                _data[i] = m._data[i];
        }
        return *this;
    }

    template <typename U>
    matrix& 
    operator = (const matrix<U>& m)
    {
        if (&m != this)
        {
            resize(m._height, m._width);
            
            size_type size = this->size();
            for (size_type i = 0; i < size; ++i)
                _data[i] = static_cast<value_type>(m._data[i]);
        }
        return *this;
    }

    matrix& 
    operator = (matrix&& m)
    {
        if (&m != this)
        {
            _deallocate();
            
            _data = m._data;
            _height = m._height;
            _width = m._width;

            m._data = nullptr;
            m._height = 0;
            m._width = 0;
        }
        return *this;
    }

    void
    resize(size_type height, size_type width)
    {
        if (size() == height * width)
        {
            _height = height;
            _width = width;
        }
        else
        {
            _deallocate();

            _height = height;
            _width = width;

            _allocate();
        }
    }

    void
    fill(const_reference fillament)
    {
        size_type size = this->size();
        for (size_type i = 0; i < size; ++i)
            _data[i] = fillament;
    }

    pointer
    row(size_type index) 
    {
        return _data + index * _width;
    }

    const_pointer
    row(size_type index) const
    {
        return _data + index * _width;
    }

    pointer 
    operator [] (size_type index)
    {
        return row(index);
    }

    const_pointer 
    operator [] (size_type index) const 
    {
        return row(index);
    }

    reference
    operator () (size_type y, size_type x)
    {
        return row(y)[x];
    }

    const_reference
    operator () (size_type y, size_type x) const
    {
        return row(y)[x];
    }

    reference
    at(size_type y, size_type x)
    {
        static value_type value;
        if (y >= _height || x >= _width || y < 0 || x < 0)
        {
            value = 0;
            return value;
        }

        return _data[y * _width + x];
    }

    const_reference
    at(size_type y, size_type x) const
    {
        static value_type value;
        if (y >= _height || x >= _width || y < 0 || x < 0)
        {
            value = 0;
            return value;
        }

        return _data[y * _width + x];
    }

    size_type 
    size() const
    {
        return _width * _height;
    }

    /// @brief Matrix width getter
    /// @return Matrix width
    size_type 
    width() const
    {
        return _width;
    }

    /// @brief Matrix height getter
    /// @return Matrix height
    size_type 
    height() const
    {
        return _height;
    }

    /// @brief Matrix data pointer getter
    /// @return Matrix data pointer
    pointer
    data()
    {
        return _data;
    }

    /// @brief Matrix data pointer getter
    /// @return Matrix data pointer
    const_pointer
    data() const
    {
        return _data;
    }

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

protected:

    pointer _data;
    size_type _height, _width;

};

} // namespace lm
