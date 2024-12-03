#pragma once

#include <tuple>

#include <cuda.h>

#include "cuda/array.cuh"

#include "utility/types.hpp"
#include "utility/log.hpp"

#define CU_CHECK(x) if (auto err = (x); err != CUDA_SUCCESS) { const char* s; cuGetErrorString(err, &s); log::error(#x " error:", s); }

namespace lumina::cuda
{

enum Filter   { NEAREST, LINEAR };
enum Overflow { WRAP, CLAMP, MIRROR, BORDER };
enum Indexing { UNNORMALIZED, NORMALIZED };

template <u64 D, typename T>
class texture : public array<D, T>
{
    using base = array<D, T>;

public:

    texture()
    :   _handle(0)
    {}

    template <typename... Dims>
    requires(sizeof...(Dims) == D)
    texture(Dims... dims)
    :   base(dims...)
    {
        CUDA_RESOURCE_DESC res_desc = {
            .resType = CU_RESOURCE_TYPE_ARRAY,
            .res = {
                .array = {
                    .hArray = base::_handle
                }
            },
        };

        CUDA_TEXTURE_DESC tex_desc = {
            .addressMode = { CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_WRAP },
            .filterMode = CU_TR_FILTER_MODE_LINEAR,

            .flags = CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION,
            .borderColor = { 0.f }
        };

        CU_CHECK(cuTexObjectCreate(&_handle, &res_desc, &tex_desc, nullptr));
    }

    ~texture()
    {
        cuTexObjectDestroy(_handle);
    }

    texture(const texture&) = delete;
    texture(texture&& tex)
    :   _handle(tex._handle)
    {
        tex._handle = 0;
    }

    texture& operator= (const texture&) = delete;
    texture& operator= (texture&& tex)
    {
        if (&tex != this)
        {
            _handle = tex._handle;
            tex._handle = 0;
        }

        return *this;
    }

    template <typename... Dims>
    __device__ __forceinline__
    T operator() (Dims... dims) const
    requires (sizeof...(Dims) == D)
    {
        if constexpr (sizeof...(Dims) == 1) return tex1D<T>(_handle, static_cast<float>(dims)...);
        if constexpr (sizeof...(Dims) == 2) return tex2D<T>(_handle, static_cast<float>(dims)...);
        if constexpr (sizeof...(Dims) == 3) return tex3D<T>(_handle, static_cast<float>(dims)...);
    }

    operator CUtexObject() const
    {
        return _handle;
    }

protected:

    void _configure()
    {}

protected:

    CUtexObject _handle;
};

}
