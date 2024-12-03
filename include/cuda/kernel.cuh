#pragma once

#include <cuda_runtime.h>

#include "util/log.hpp"

#include "cuda/error.cuh"
#include "cuda/stream.cuh"

namespace lumina::cuda {

template <typename... Args>
class kernel
{
public:

    __host__
    kernel(void (*f)(Args...))
    :   _ptr(f)
    {}

    __host__ auto
    bind(dim3          blocks,
         dim3          threads,
         const stream& stream,
         size_t        shared = 0) const
    {
        return [this, blocks, threads, &stream, shared](const Args&... args) {
            return operator()(blocks, threads, stream, shared, args...);
        };
    }

    __host__ void
    operator()(dim3          blocks,
               dim3          threads,
               const stream& stream,
               size_t        shared,
               const Args&... args
    ) const {
        const void* arg_ptrs[sizeof...(Args)] = {&args...};

        cuda::error error = cudaLaunchKernel((void*)_ptr, blocks, threads, (void**)arg_ptrs, shared, stream);

        if (error)
            lumina::log::error("kernel launch failed: ", error.message());
    }

private:

    void (*_ptr)(Args...);
};

} // namespace lumina::cuda
