/* 

    Copyright (c) 2023 Mokhov Mark

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

#include "cuda/cuda.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////

lumina::cuda::error::error(cudaError_t error)
:   _handle(error)
{}

const char*
lumina::cuda::error::message() const
{
    return cudaGetErrorString(_handle);
}

lumina::cuda::error::operator bool() const
{
    return _handle != cudaSuccess;
}

lumina::cuda::error::operator cudaError_t() const
{
    return _handle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const lumina::cuda::stream lumina::cuda::stream::main{0};

lumina::cuda::stream::stream()
{
    error status = cudaStreamCreate(&_handle);
    if (status)
        log::error("cudaStreamCreate failed:", status.message());
}

lumina::cuda::stream::~stream()
{
    if (*this == main)
        return;

    error status = cudaStreamDestroy(_handle);
    if (status)
        log::error("cudaStreamDestroy failed:", status.message());
}

bool
lumina::cuda::stream::operator == (const stream& s)
{
    return _handle == s._handle;
}

bool
lumina::cuda::stream::operator != (const stream& s)
{
    return !(*this == s);
}

lumina::cuda::stream::operator cudaStream_t() const
{
    return _handle;
}

void
lumina::cuda::stream::synchronize() const
{
    error status = cudaStreamSynchronize(_handle);
    if (status)
        log::error("cudaStreamSynchronize failed:", status.message());
}

lumina::cuda::stream::stream(cudaStream_t handle)
:   _handle(handle)
{}

////////////////////////////////////////////////////////////////////////////////////////////////////

void*
lumina::cuda::malloc(size_t size)
{
    void* ptr = nullptr;

    error status = cudaMalloc((void**)&ptr, size);
    if (status)
        log::error("cudaMalloc failed:", status.message());

    return ptr;
}

void
lumina::cuda::free(void* ptr)
{
    error status = cudaFree(ptr);
    if (status)
        log::error("cudaFree failed:", status.message());
}

void
lumina::cuda::memcpy(void* dst, const void* src, size_t size, memcpy_kind kind)
{
    error status = cudaMemcpy(dst, src, size, cudaMemcpyKind(kind + cudaMemcpyHostToHost));
    if (status)
        log::error("cudaMemcpy failed:", status.message());
}

void
lumina::cuda::memcpy_async(void* dst, const void* src, size_t size, memcpy_kind kind, const stream& stream)
{
    error status = cudaMemcpyAsync(dst, src, size, cudaMemcpyKind(kind + cudaMemcpyHostToHost), stream);
    if (status)
        log::error("cudaMemcpyAsync failed:", status.message());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
