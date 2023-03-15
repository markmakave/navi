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

namespace lm {

template <typename T>
class heap_allocator {

    static
    void*
    allocate(size_t size)
    {
        void* ptr = operator new[](size * sizeof(T));

        if (ptr == nullptr)
            lm::log::error("memory allocation failed");

        return ptr;
    }

    static
    void
    deallocate(void* ptr)
    {
        operator delete[](ptr);
    }

};

template <typename T>
class stack_allocator {

    static
    void*
    allocate(size_t size)
    {
        void* ptr = alloca(size * sizeof(T));

        if (ptr == nullptr)
            lm::log::error("memory allocation failed");

        return ptr;
    }

    static
    void
    deallocate(void* ptr)
    {
        // nop
    }

}

}
