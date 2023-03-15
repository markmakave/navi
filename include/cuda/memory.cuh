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

namespace lm {
namespace cuda {

template <typename T>
class device_allocator {

    __host__ __device__
    static
    void*
    allocate(size_t size)
    {
        void* ptr;

        #ifdef __CUDA_ARCH__

        ptr = operator new[](size * sizeof(T));
        if (ptr == nullptr)
            printf("[ DEVICE ] memory allocation failed\n");

        #else

        cudaError_t status = cudaMalloc((void**)&ptr, size * sizeof(T));
        if (status != cudaSuccess)
            lm::log::error("cudaMalloc failed:", cudaGetErrorString(status));

        #endif

        return ptr;
    }

    __host__ __device__
    static
    void
    deallocate(void* ptr)
    {
        #ifdef __CUDA_ARCH__

        operator delete[](ptr);

        #else

        cudaError_t status = cudaFree(ptr);
        if (status != cudaSuccess)
            lm::log::error("cudaFree failed:", cudaGetErrorString(status));

        #endif
    }

};

template <typename T>
class host_allocator {

    __host__
    static
    void*
    allocate(size_t size)
    {
        void* ptr;
        
        cudaError_t status = cudaMallocHost((void**)&ptr, size * sizeof(T));
        if (status != cudaSuccess)
            lm::log::error("cudaMallocHost failed:", cudaGetErrorString(status));

        return ptr;
    }

    __host__
    static
    void
    deallocate(void* ptr)
    {
        cudaError_t status = cudaFree(ptr);
        if (status != cudaSuccess)
            lm::log::error("cudaFree failed:", cudaGetErrorString(status));
    }

}

template <typename T>
class managed_allocator {

    __host__
    static
    void*
    allocate(size_t size)
    {
        void* ptr;
        
        cudaError_t status = cudaMallocManaged((void**)&ptr, size * sizeof(T));
        if (status != cudaSuccess)
            lm::log::error("cudaMallocHost failed:", cudaGetErrorString(status));

        return ptr;
    }

    __host__
    static
    void
    deallocate(void* ptr)
    {
        cudaError_t status = cudaFree(ptr);
        if (status != cudaSuccess)
            lm::log::error("cudaFree failed:", cudaGetErrorString(status));
    }

}

}
}
