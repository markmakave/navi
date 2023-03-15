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
// #include "util/log.hpp"

#define DEBUG_LOG(message, ...) { /*lm::log::debug(message, ##__VA_ARGS__);*/ }
#define ERROR_LOG(message, ...) { /*lm::log::error(message, ##__VA_ARGS__);*/ }

namespace lm {
namespace cuda {

/// @brief Matrix class with basic operations and STL-like interface
/// @tparam T Type of the elements in the matrix (int, float, double, etc.)
template <typename T>
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
    typedef unsigned            size_type;
    typedef int                 difference_type;

public:

    /// @brief Default constructor
    __host__ __device__
    matrix()
    :   _data(nullptr),
        _height(0),
        _width(0)
    {}

    /// @brief Constructor with height and width
    /// @param height Height of the matrix
    /// @param width Width of the matrix
    __host__ __device__
    matrix(size_type height, size_type width) 
    :   _height(height),
        _width(width)
    {
        _allocate();
    }

    /// @brief Copy constructor
    /// @param m matrix to copy from
    __host__ __device__
    matrix(const matrix<value_type>& m) 
    :   _height(m._height),
        _width(m._width)
    {
        _allocate();

        size_type size = this->size();

        #ifdef __CUDA_ARCH__

        for (size_type i = 0; i < size; ++i)
            _data[i] = m._data[i];

        #else

        cudaError_t status = cudaMemcpy(_data, m._data, size * sizeof(value_type), cudaMemcpyDeviceToDevice);
        if (status != cudaSuccess)
            ERROR_LOG("cudaMemcpy error occured:", cudaGetErrorString(status));

        #endif
    }

    /// @brief Move constructor
    /// @param m matrix to move from
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
    
    /// @brief Destructor
    __host__ __device__
    ~matrix()
    {
        _deallocate();
    }

    /// @brief Copy assignment operator
    /// @param m matrix to copy from
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

            cudaError_t status = cudaMemcpy(_data, m._data, size * sizeof(value_type), cudaMemcpyDeviceToDevice);
            if (status != cudaSuccess)
                ERROR_LOG("cudaMemcpy error occured:", cudaGetErrorString(status));

            #endif
        }

        return *this;
    }

    /// @brief Move assignment operator
    /// @param m matrix to move from
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

    /// @brief Resizes the matrix to the given dimensions
    /// @param height New height of the matrix
    /// @param width New width of the matrix
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
    /// @brief Fill the matrix with a given value
    /// @param fillament Value to fill the matrix with
    void
    fill(const_reference fillament)
    {
        size_type size = this->size();
        for (size_type i = 0; i < size; ++i)
            _data[i] = fillament;
    }

    /// @brief Returns matrix row pointer
    /// @param index Row index
    /// @return Pointer to the row
    __host__ __device__
    pointer
    row(size_type index) 
    {
        return _data + index * _width;
    }

    /// @brief Returns matrix row pointer
    /// @param index Row index
    /// @return Pointer to the row
    __host__ __device__
    const_pointer
    row(size_type index) const
    {
        return _data + index * _width;
    }

    /// @brief Returns matrix row
    /// @param index Row index
    /// @return Row
    __host__ __device__
    pointer 
    operator [] (size_type index)
    {
        return row(index);
    }

    /// @brief Returns matrix row
    /// @param index Row index
    /// @return Row
    __host__ __device__
    const_pointer 
    operator [] (size_type index) const 
    {
        return row(index);
    }

    #ifdef __CUDA_ARCH__

    /// @brief Element access operator
    /// @param y 
    /// @param x
    /// @return Reference to the element
    __device__
    reference
    operator () (size_type y, size_type x)
    {
        return row(y)[x];
    }

    #else

    class bridge
    {
    public:

        __host__
        bridge&
        operator = (const_reference value)
        {
            cudaError_t status = cudaMemcpy(_parent[_y] + _x, &value, sizeof(value), cudaMemcpyHostToDevice);
            if (status != cudaSuccess)
                ERROR_LOG("cudaMemcpy error occured:", cudaGetErrorString(status));

            return *this;
        }

        __host__
        operator value_type() const
        {
            value_type value;
            cudaError_t status = cudaMemcpy(&value, _parent[_y] + _x, sizeof(value), cudaMemcpyDeviceToHost);
            if (status != cudaSuccess)
                ERROR_LOG("cudaMemcpy error occured:", cudaGetErrorString(status));

            return value;
        }

    private:

        friend class matrix;

        __host__
        bridge() {}

        __host__
        bridge(matrix& parent, size_type y, size_type x)
        :   _parent(parent),
            _y(y),
            _x(x)
        {}

    private:

        matrix& _parent;
        size_type _y, _x;

    };

    __host__
    bridge
    operator () (size_type y, size_type x)
    {
        return bridge(*this, y, x);
    }

    #endif

    /// @brief Element access operator
    /// @param y 
    /// @param x 
    /// @return Const reference to the element
    __host__ __device__
    const_reference
    operator () (size_type y, size_type x) const
    {
        #ifdef __CUDA_ARCH__

        return row(y)[x];

        #else

        static value_type value;
        
        cudaError_t status = cudaMemcpy(&value, row(y) + x, sizeof(value), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess)
            ERROR_LOG("cudaMemcpy error occured:", cudaGetErrorString(status));

        return value;

        #endif
    }

    #ifdef __CUDA_ARCH__

    /// @brief Element access operator
    /// @param y Row index
    /// @param x Column index
    /// @return Reference to the element
    __device__
    reference
    at(size_type y, size_type x)
    {
        if (y >= _height || x >= _width)
            throw std::out_of_range("matrix::at");

        return _data[y * _width + x];
    }

    #endif

    /// @brief Element access operator
    /// @param y Row index
    /// @param x Column index
    /// @return Reference to the element
    __host__ __device__
    const_reference
    at(size_type y, size_type x) const
    {
        static value_type value;

        #ifdef __CUDA_ARCH__
        
        if (y >= _height || x >= _width)
        {
            value = value_type();
            return value;
        }

        return _data[y * _width + x];

        #else

        if (y >= _height || x >= _width)
        {
            value = value_type();
            return value;
        }

        cudaError_t status = cudaMemcpy(&value, row(y) + x, sizeof(value), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess)
            ERROR_LOG("cudaMemcpy error occured:", cudaGetErrorString(status));

        return value;

        #endif
    }

    /// @brief Matrix size getter
    /// @return Matrix size in elements
    __host__ __device__
    size_type 
    size() const
    {
        return _width * _height;
    }

    /// @brief Matrix width getter
    /// @return Matrix width
    __host__ __device__
    size_type 
    width() const
    {
        return _width;
    }

    /// @brief Matrix height getter
    /// @return Matrix height
    __host__ __device__
    size_type 
    height() const
    {
        return _height;
    }

    /// @brief Matrix data pointer getter
    /// @return Matrix data pointer
    __host__ __device__
    pointer
    data()
    {
        return _data;
    }

    /// @brief Matrix data pointer getter
    /// @return Matrix data pointer
    __host__ __device__
    const_pointer
    data() const
    {
        return _data;
    }

    __host__
    void
    operator << (const lm::matrix<value_type>& m)
    {
        resize(m.height(), m.width());

        cudaError_t status = cudaMemcpy(_data, m.data(), size(), cudaMemcpyHostToDevice);
        if (status != cudaSuccess)
            ERROR_LOG("cudaMemcpy error occured:", cudaGetErrorString(status));
    }

    __host__
    void
    operator >> (lm::matrix<value_type>& m)
    {
        m.resize(_height, _width);

        cudaError_t status = cudaMemcpy(m.data(), _data, size() * sizeof(value_type), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess)
            ERROR_LOG("cudaMemcpy error occured:", cudaGetErrorString(status));
    }

private:

    __host__ __device__
    void
    _allocate()
    {
        #ifdef __CUDA_ARCH__

        if (size() == 0)
            _data = nullptr;
        else
            _data = reinterpret_cast<pointer>(operator new[](size() * sizeof(value_type)));

        #else

        DEBUG_LOG("Allocating", size() * sizeof(value_type), "bytes on GPU");

        cudaError_t status = cudaMalloc((void**)&_data, size() * sizeof(value_type));
        if (status != cudaSuccess)
            ERROR_LOG("cudaMalloc error occured:", cudaGetErrorString(status));

        #endif
    }

    __host__ __device__
    void
    _deallocate()
    {
        #ifdef __CUDA_ARCH__

        operator delete[](_data);

        #else

        if (_data != nullptr)
        {
            DEBUG_LOG("Deallocating", (void*)_data, "from GPU");

            cudaError_t status = cudaFree(_data);
            if (status != cudaSuccess)
                ERROR_LOG("cudaFree error occured:", cudaGetErrorString(status));
        }
        
        #endif
    }

protected:

    pointer _data;
    size_type _height, _width;

};

} // namespace cuda
} // namespace lm
