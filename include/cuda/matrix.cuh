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

#include <cuda_runtime.h>

#include "base/matrix.hpp"
#include "cuda/memory.cuh"

namespace lm {
namespace cuda {

template <typename T, typename _alloc = device_allocator<T>>
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

    __host__ __device__
    matrix()
    :   _data(nullptr),
        _height(0),
        _width(0)
    {}

    __host__ __device__
    matrix(size_type height, size_type width) 
    :   _height(height),
        _width(width)
    {
        _allocate();
    }

    __host__ __device__
    matrix(const matrix& m) 
    :   _height(m._height),
        _width(m._width)
    {
        _allocate();

        size_type size = this->size();

        #ifdef __CUDA_ARCH__

        for (size_type i = 0; i < size; ++i)
            _data[i] = m._data[i];

        #else

        cudaMemcpy(_data, m._data, size * sizeof(value_type), cudaMemcpyDeviceToDevice);

        #endif
    }

    __host__ __device__
    matrix(matrix&& m) 
    :   _data(m._data),
        _height(m._height),
        _width(m._width)
    {
        m._data = nullptr;
        m._height = 0;
        m._width = 0;
    }
    
    __host__ __device__
    ~matrix()
    {
        _deallocate();
    }

    __host__ __device__
    matrix& 
    operator = (const matrix& m)
    {
        if (&m != this)
        {
            resize(m._height, m._width);

            size_type size = this->size();

            #ifdef __CUDA_ARCH__

            for (size_type i = 0; i < size; ++i)
                _data[i] = m._data[i];

            #else

            cudaMemcpy(_data, m._data, size * sizeof(value_type), cudaMemcpyDeviceToDevice);

            #endif
        }

        return *this;
    }

    __host__ __device__
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

    __host__ __device__
    void
    resize(size_type height, size_type width)
    {
        if (size() == height * width)
        {
            _height = height;
            _width = width;
        } else {
            _deallocate();

            _height = height;
            _width = width;

            _allocate();
        }
    }

    __host__ __device__
    void
    fill(const_reference fillament)
    {
        size_type size = this->size();
        for (size_type i = 0; i < size; ++i)
            _data[i] = fillament;
    }

    __host__ __device__
    pointer
    row(size_type index) 
    {
        return _data + index * _width;
    }

    __host__ __device__
    const_pointer
    row(size_type index) const
    {
        return _data + index * _width;
    }

    __host__ __device__
    auto 
    operator [] (size_type index)
    {
        #ifdef __CUDA_ARCH__
        return _data + index * _width;
        #else
        return _alloc::access(_data + index * _width);
        #endif
    }

    __host__ __device__
    auto 
    operator [] (size_type index) const 
    {
        #ifdef __CUDA_ARCH__
        return _data + index * _width;
        #else
        return _alloc::access(_data + index * _width);
        #endif
    }

    __host__ __device__
    auto
    operator () (size_type y, size_type x)
    {
        return _alloc::access(_data + y * _width + x);
    }

    __host__ __device__
    auto
    operator () (size_type y, size_type x) const
    {
        return _alloc::access(_data + y * _width + x);
    }

    __host__ __device__
    const_reference
    at(size_type y, size_type x) const
    {
        if (y >= _height or x >= _width)
        {
            static value_type value;
            value = value_type();
            return value;
        }

        return _alloc::access(_data + y * _width + x);
    }

    __host__ __device__
    size_type 
    size() const
    {
        return _width * _height;
    }

    __host__ __device__
    size_type 
    width() const
    {
        return _width;
    }

    __host__ __device__
    size_type 
    height() const
    {
        return _height;
    }

    __host__ __device__
    pointer
    data()
    {
        return _data;
    }

    __host__ __device__
    const_pointer
    data() const
    {
        return _data;
    }

    __host__
    void
    copy_to(lm::matrix<value_type>& m) const
    {
        m.resize(_height, _width);
        memcpy(m.data(), _data, size() * sizeof(value_type), cuda::D2H);
    }

    __host__
    void
    copy_from(const lm::matrix<value_type>& m)
    {
        resize(m.height(), m.width());
        memcpy(_data, m.data(), size() * sizeof(value_type), cuda::H2D);
    }

    __host__
    void
    copy_to_async(lm::matrix<value_type>& m, const stream& stream) const
    {
        m.resize(_height, _width);
        memcpy_async(m.data(), _data, size() * sizeof(value_type), cuda::D2H, stream);
    }

    __host__
    void
    copy_from_async(const lm::matrix<value_type>& m, const stream& stream)
    {
        resize(m.height(), m.width());
        memcpy_async(_data, m.data(), size() * sizeof(value_type), cuda::H2D, stream);
    }

    __host__
    void
    operator << (const lm::matrix<value_type>& m)
    {
        copy_from(m);
    }

    __host__
    void
    operator >> (lm::matrix<value_type>& m)
    {
        copy_to(m);
    }

private:

    __host__ __device__
    void
    _allocate()
    {
        _data = _alloc::allocate(size());
    }

    __host__ __device__
    void
    _deallocate()
    {
        _alloc::deallocate(_data);
    }

protected:

    pointer _data;
    size_type _height, _width;
};

} // namespace cuda
} // namespace lm
