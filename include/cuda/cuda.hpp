#pragma once

#include <cuda_runtime.h>
#include <iostream>

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
    friend class kernel;

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

class kernel
{
public:

    template <typename T>
    __host__
    kernel(T* ptr)
    :   _ptr(reinterpret_cast<void*>(ptr))
    {}

    template <typename... Args>
    __host__
    void
    operator () (dim3 blocks, dim3 threads, const stream& stream, Args&&... args) const
    {
        void* arg_ptrs[sizeof...(Args)] = {&args...};
        cudaError_t status = cudaLaunchKernel(_ptr, blocks, threads, arg_ptrs, 0, stream._handle);
        if (status != cudaSuccess)
            lm::log::error("cudaLaunchKernel failed:", cudaGetErrorString(status));
    }

private:

    void* _ptr;
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
memcpy(void* dst, void* src, size_t size, memcpy_kind kind);

void
memcpy_async(void* dst, void* src, size_t size, memcpy_kind kind, const stream& stream);

}
}
