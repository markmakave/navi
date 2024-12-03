#pragma once

#include <cuda.h>

#include "util/types.hpp"
#include "util/log.hpp"

#define CU_CHECK(x) if (auto err = (x); err != CUDA_SUCCESS) { const char* s; cuGetErrorString(err, &s); log::error(#x " error:", s); }

namespace lumina::cuda
{

template <size_t Dim, typename T>
class texture
{
public:

    texture()
    {
        float data[] = { -1.f, 0.5f, -0.5f, 1.f };
        
        CUarray array;
        CUDA_ARRAY_DESCRIPTOR array_desc = {
            .Width = sizeof(data) / sizeof(*data),
            .Height = 1,
            .Format = get_type_enum(),
            .NumChannels = 1
        };
        CU_CHECK(cuArrayCreate(&array, &array_desc));

        CUDA_MEMCPY2D params = {
            .srcXInBytes = 0,
            .srcY = 0,

            .srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST,
            .srcHost = data,
            .srcPitch = sizeof(data),

            .dstXInBytes = 0,
            .dstY = 0,

            .dstMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY,
            .dstArray = array,

            .WidthInBytes = sizeof(data),
            .Height = 1
        };
        CU_CHECK(cuMemcpy2D(&params));

        CUDA_RESOURCE_DESC res_desc = {
            .resType = CU_RESOURCE_TYPE_ARRAY,
            .res = {
                .array = {
                    .hArray = array
                }
            },
        };

        CUDA_TEXTURE_DESC tex_desc = {
            // .addressMode = { CU_TR_ADDRESS_MODE_CLAMP },
            .addressMode = { CU_TR_ADDRESS_MODE_WRAP },
            // .addressMode = { CU_TR_ADDRESS_MODE_MIRROR },
            // .addressMode = { CU_TR_ADDRESS_MODE_BORDER },

            .filterMode = CU_TR_FILTER_MODE_LINEAR,
            // .filterMode = CU_TR_FILTER_MODE_POINT,

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
    texture(texture&&) = delete;

    template <typename... Dims>
    requires(sizeof...(Dims) == Dim)
    __device__ __forceinline__
    T operator[] (Dims... dims) const
    {
        cudaTextureObject_t handle = _handle;
        if constexpr (Dim == 1)
            return tex1D<T>(handle, dims...);
        if constexpr (Dim == 2)
            return tex2D<T>(handle, dims...);
        if constexpr (Dim == 3)
            return tex3D<T>(handle, dims...);
    }

    operator CUtexObject() const
    {
        return _handle;
    }

protected:

    static CUarray_format get_type_enum()
    {
        // Integers
        if constexpr (std::is_same_v<T, u8>)  return CU_AD_FORMAT_UNSIGNED_INT8;
        if constexpr (std::is_same_v<T, u16>) return CU_AD_FORMAT_UNSIGNED_INT16;
        if constexpr (std::is_same_v<T, u32>) return CU_AD_FORMAT_UNSIGNED_INT32;
        if constexpr (std::is_same_v<T, i8>)  return CU_AD_FORMAT_SIGNED_INT8;
        if constexpr (std::is_same_v<T, i16>) return CU_AD_FORMAT_SIGNED_INT16;
        if constexpr (std::is_same_v<T, i32>) return CU_AD_FORMAT_SIGNED_INT32;

        // Floating point
        if constexpr (std::is_same_v<T, f32>) return CU_AD_FORMAT_FLOAT;

        throw std::invalid_argument("Unsupported texture value type");
    }

    void _init()
    {

    }

protected:

    CUtexObject _handle;
};

}
