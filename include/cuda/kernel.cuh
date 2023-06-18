#pragma once

#include "util/log.hpp"

namespace lumina {
namespace cuda {

template <typename... Args>
class kernel
{
public:

    __host__
    kernel(void (*f)(Args...))
      : _ptr(f)
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
               const Args&... args) const
    {
        const void* arg_ptrs[sizeof...(Args)] = {&args...};
        cudaError_t status                    = cudaLaunchKernel(
            (void*)_ptr, blocks, threads, (void**)arg_ptrs, shared, stream);
        if (status != cudaSuccess)
            lumina::log::error("cudaLaunchKernel failed:", cudaGetErrorString(status));
    }

private:

    void (*_ptr)(Args...);
};

} // namespace cuda
} // namespace lumina
