#include "cuda/cuda.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

lm::cuda::error::error(cudaError_t error)
:   _handle(error)
{}

const char*
lm::cuda::error::describe() const
{
    return cudaGetErrorString(_handle);
}

lm::cuda::error::operator bool() const
{
    return _handle != cudaSuccess;
}

lm::cuda::error::operator cudaError_t() const
{
    return _handle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const lm::cuda::stream lm::cuda::stream::main{0};

lm::cuda::stream::stream()
{
    cudaError_t status = cudaStreamCreate(&_handle);
    if (status != cudaSuccess)
        lm::log::error("cudaStreamCreate failed:", cudaGetErrorString(status));
}

lm::cuda::stream::~stream()
{
    if (*this == main)
        return;

    cudaError_t status = cudaStreamDestroy(_handle);
    if (status != cudaSuccess)
        lm::log::error("cudaStreamDestroy failed:", cudaGetErrorString(status));
}

bool
lm::cuda::stream::operator == (const stream& s)
{
    return _handle == s._handle;
}

bool
lm::cuda::stream::operator != (const stream& s)
{
    return !(*this == s);
}

lm::cuda::stream::operator cudaStream_t() const
{
    return _handle;
}

void
lm::cuda::stream::synchronize() const
{
    cudaError_t status = cudaStreamSynchronize(_handle);
    if (status != cudaSuccess)
        lm::log::error("cudaStreamSynchronize failed:", cudaGetErrorString(status));
}

lm::cuda::stream::stream(cudaStream_t handle)
:   _handle(handle)
{}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Kernel is fully-templated class

////////////////////////////////////////////////////////////////////////////////////////////////////

void*
lm::cuda::malloc(size_t size)
{
    void* ptr = nullptr;

    error e = cudaMalloc((void**)&ptr, size);
    if (e)
        log::error("cudaMemcpy failed:", e.describe());

    return ptr;
}

void
lm::cuda::free(void* ptr)
{
    error e = cudaFree(ptr);
    if (e)
        log::error("cudaMemcpy failed:", e.describe());
}

void
lm::cuda::memcpy(void* dst, void* src, size_t size, memcpy_kind kind)
{
    error e = cudaMemcpy(dst, src, size, cudaMemcpyKind(kind + cudaMemcpyHostToHost));
    if (e)
        log::error("cudaMemcpy failed:", e.describe());
}

void
lm::cuda::memcpy_async(void* dst, void* src, size_t size, memcpy_kind kind, const stream& stream)
{
    error e = cudaMemcpyAsync(dst, src, size, cudaMemcpyKind(kind + cudaMemcpyHostToHost), stream);
    if (e)
        log::error("cudaMemcpy failed:", e.describe());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
