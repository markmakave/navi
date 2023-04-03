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

#include <alloca.h>
#include "util/log.hpp"

namespace lm {

template <typename T>
class heap_allocator
{
public:

    typedef T                 value_type;
    typedef value_type*       pointer;
    typedef const value_type* const_pointer;
    typedef int64_t           size_type;


public:

    static
    pointer
    allocate(size_t size)
    {
        void* ptr = operator new[](size * sizeof(T));
        if (ptr == nullptr)
            log::error("memory allocation error");

        return reinterpret_cast<pointer>(ptr);
    }

    static
    void
    deallocate(pointer ptr)
    {
        operator delete[](ptr);
    }

    static
    void
    copy(const_pointer src, pointer dst, size_type size)
    {
        for (size_type i = 0; i < size; ++i)
            dst[i] = src[i];
    }

};

template <typename T>
class stack_allocator
{
public:

    static
    T*
    allocate(size_t size)
    {
        void* ptr = alloca(size * sizeof(T));

        return reinterpret_cast<T*>(ptr);
    }

    static
    void
    deallocate(T* ptr)
    {
        // nop
    }

};

}
