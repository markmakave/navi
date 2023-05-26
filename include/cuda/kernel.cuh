#pragma once

#include "util/log.hpp"

namespace lm {
namespace cuda {

template <typename... Args>
class kernel
{   
public:

    __host__
    kernel(void(*f)(Args...))
    :   _ptr(f)
    {}

    __host__
    auto
    bind(dim3 blocks, dim3 threads, const stream& stream) const
    {
        return [this, blocks, threads, &stream](const Args&... args) {
            return operator()(blocks, threads, stream, args...);
        };
    }

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

}
}
