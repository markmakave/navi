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
