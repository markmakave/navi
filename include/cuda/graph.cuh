#pragma once

#include <cuda_runtime.h>

#include "cuda/cuda.cuh"

namespace lumina::cuda {

class graph
{
public:

    class node
    {
    public:

        enum type { host, kernel, malloc, memcpy, memset };

    public:

        node(cudaGraphNode_t handle)
        :   _handle(handle)
        {}

        operator cudaGraphNode_t()
        {
            return _handle;
        }

    protected:

        cudaGraphNode_t _handle;
    };

public:

    graph()
    {
        if (error e = cudaGraphCreate(&_handle, 0); e)
            log::error("cudaGraphCreate failed:", e.message());
    }

    graph(const graph& other)
    {
        if (error e = cudaGraphClone(&_handle, other._handle); e)
            log::error("cudaGraphClone failed:", e.message());
    }

    ~graph()
    {
        if (error e = cudaGraphDestroy(_handle); e)
            log::error("cudaGraphDestroy failed:", e.message());
    }

    template <node::type T>
    node add_node();

protected:
    
    cudaGraph_t _handle;
};

template <>
graph::node add_node<graph::node::type::host>()
{}

}
