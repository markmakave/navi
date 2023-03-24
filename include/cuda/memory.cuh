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

#include "cuda/cuda.hpp"

namespace lm {
namespace cuda {

template <typename T>
class proxy
{
public:

    typedef T                 value_type;
    typedef value_type*       pointer;
    typedef const value_type& const_reference;
    typedef unsigned          size_type;

public:

    __host__
    proxy(pointer ptr)
    :   _ptr(ptr)
    {}

    __host__
    proxy&
    operator = (const_reference value)
    {
        cuda::memcpy(_ptr, &value, sizeof(value), cuda::H2D);
        return *this;
    }

    __host__
    operator value_type() const
    {
        value_type value;
        cuda::memcpy(&value, _ptr, sizeof(value), cuda::D2H);
        return value;
    }

    __host__
    proxy
    operator [] (size_type index)
    {
        return proxy(_ptr + index);
    }

private:

    pointer _ptr;
};

template <typename T>
class device_allocator
{
public:

    typedef T                   value_type;
    typedef value_type*         pointer;
    typedef const value_type*   const_pointer;
    typedef value_type&         reference;
    typedef const value_type&   const_reference;
    typedef unsigned            size_type;

public:

    __host__ __device__
    static
    pointer
    allocate(size_type size)
    {
        void* ptr;

        #ifdef __CUDA_ARCH__
        ptr = operator new[](size * sizeof(T));
        #else
        ptr = cuda::malloc(size * sizeof(T));
        #endif

        return reinterpret_cast<pointer>(ptr);
    }

    __host__ __device__
    static
    void
    deallocate(pointer ptr)
    {
        #ifdef __CUDA_ARCH__
        operator delete[](ptr);
        #else
        cuda::free(ptr);
        #endif
    }

    __host__ __device__
    static
    const auto
    access(const_pointer p)
    {
        #ifdef __CUDA_ARCH__
        return *p;
        #else
        return proxy<value_type>(p);
        #endif
    }

    __host__ __device__
    static
    auto
    access(pointer p)
    {
        #ifdef __CUDA_ARCH__
        return *p;
        #else
        return proxy<value_type>(p);
        #endif
    }

    __host__ __device__
    static
    void
    copy(const_pointer src, pointer dst, size_type size)
    {
        #ifdef __CUDA_ARCH__
        for (size_type i = 0; i < size; ++i)
            dst[i] = src[i];
        #else
        cuda::memcpy(dst, src, size * sizeof(value_type), D2D);
        #endif
    }

};

template <typename T>
class pinned_allocator
{
public:

    typedef T                   value_type;
    typedef value_type*         pointer;
    typedef const value_type*   const_pointer;
    typedef value_type&         reference;
    typedef const value_type&   const_reference;
    typedef unsigned            size_type;

public:

    __host__
    static
    pointer
    allocate(size_t size)
    {
        void* ptr;
        
        cudaMallocHost((void**)&ptr, size * sizeof(T));

        return reinterpret_cast<pointer>(ptr);
    }

    __host__
    static
    void
    deallocate(pointer ptr)
    {
        cudaFree(ptr);
    }

    __host__
    static
    const_reference
    access(const_pointer p)
    {
        return *p;
    }

    __host__
    static
    reference
    access(pointer p)
    {
        return *p;
    }

};

template <typename T>
class managed_allocator
{
public:

    typedef T                   value_type;
    typedef value_type*         pointer;
    typedef const value_type*   const_pointer;
    typedef value_type&         reference;
    typedef const value_type&   const_reference;
    typedef unsigned            size_type;

public:

    __host__
    static
    pointer
    allocate(size_t size)
    {
        void* ptr;
        
        cudaMallocManaged((void**)&ptr, size * sizeof(T));

        return reinterpret_cast<pointer>(ptr);
    }

    __host__
    static
    void
    deallocate(pointer ptr)
    {
        cudaFree(ptr);
    }

    __host__ __device__
    static
    const_reference
    access(const_pointer p)
    {
        return *p;
    }

    __host__ __device__
    static
    reference
    access(pointer p)
    {
        return *p;
    }

};

}
}
