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
#include <iostream>
#include <cstddef>

#include "util/log.hpp"

namespace lm {
namespace cuda {

class error
{
public:

    error(cudaError_t error);

    const char*
    describe() const;

    operator bool() const;
    operator cudaError_t() const;

private:

    cudaError_t _handle;
};

class stream
{
public:

    static const stream main;

public:

    stream();
    stream(const stream&)      = delete;
    stream(stream&&)           = delete;
    ~stream();

    void
    operator = (const stream&) = delete;
    void
    operator = (stream&&)      = delete;

    bool
    operator == (const stream& s);
    bool
    operator != (const stream& s);

    operator cudaStream_t() const;
    
    void
    synchronize() const;

private:

    stream(cudaStream_t handle);

private:

    cudaStream_t _handle;
};

template <typename... Args>
class kernel
{
public:

    __host__
    kernel(void(*f)(Args...))
    :   _ptr(f)
    {}

    __host__
    void
    operator () (dim3 blocks, dim3 threads, const stream& stream, const Args&... args) const
    {
        const void* arg_ptrs[sizeof...(Args)] = {&args...};
        cudaError_t status = cudaLaunchKernel((void*)_ptr, blocks, threads, (void**)arg_ptrs, 0, stream);
        if (status != cudaSuccess)
            lm::log::error("cudaLaunchKernel failed:", cudaGetErrorString(status));
    }

private:

    void(*_ptr)(Args...);
};

void*
malloc(size_t size);

void
free(void* ptr);

enum memcpy_kind {
    H2H = 0,
    H2D,
    D2H,
    D2D
};

void
memcpy(void* dst, const void* src, size_t size, memcpy_kind kind);

void
memcpy_async(void* dst, const void* src, size_t size, memcpy_kind kind, const stream& stream);

}
}
