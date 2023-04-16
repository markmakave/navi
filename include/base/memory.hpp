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
#include "base/types.hpp"

namespace lm {

template <typename T>
class heap_allocator
{
public:

    using value_type        = T;
    using pointer           = value_type*;
    using const_pointer     = const value_type*;
    using reference         = value_type&;
    using const_reference   = const value_type&;
    using size_type         = i64;

public:

    static
    pointer
    allocate(size_type size)
    {
        void* ptr = new T[size];
        if (ptr == nullptr)
            log::error("memory allocation error");

        return reinterpret_cast<pointer>(ptr);
    }

    static
    void
    deallocate(pointer ptr)
    {
        delete[] ptr;
    }

    static
    reference
    access(pointer ptr)
    {
        return *ptr;
    }

    static
    const_reference
    access(const_pointer ptr)
    {
        return *ptr;
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

    using value_type        = T;
    using pointer           = value_type*;
    using const_pointer     = const value_type*;
    using reference         = value_type&;
    using const_reference   = const value_type&;
    using size_type         = i64;

public:

    static
    T*
    allocate(size_type size)
    {
        void* ptr = alloca(size * sizeof(T));

        return reinterpret_cast<pointer>(ptr);
    }

    static
    void
    deallocate([[maybe_unused]] pointer ptr)
    {
        // nop
    }

    static
    reference
    access(pointer ptr)
    {
        return *ptr;
    }

    static
    const_reference
    access(const_pointer ptr)
    {
        return *ptr;
    }

    static
    void
    copy(const_pointer src, pointer dst, size_type size)
    {
        for (size_type i = 0; i < size; ++i)
            dst[i] = src[i];
    }
};

}
