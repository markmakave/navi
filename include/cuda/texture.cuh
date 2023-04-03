#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

namespace lm {
namespace cuda {

template <typename T>
struct texture
{
public:

    __host__
    texture(const matrix<T>& ref)
    {
        cudaResourceDesc res_desc;
        cudaMallocArray

        cudaCreateTextureObject(&_handle,
    }

    __host__
    ~texture()
    {
        cudaDestroyTextureObject(_handle);
    }

private:

    static

protected:

    CUtexObject* _handle;
};

}
}
