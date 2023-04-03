#include "cuda/matrix.cuh"
#include "cuda/cuda.hpp"
#include "base/color.hpp"

#include <cuda_runtime.h>

__global__ void kernel(cudaTextureObject_t texture) {
    for (int i = -3; i < 6; ++i)
    {
        for (int j = -3; j < 6; ++j)
            printf("%d ", tex2D<lm::gray>(texture, j, i));
        printf("\n");
    }  
}

int main()
{
    lm::cuda::matrix<lm::gray> m(3, 3);
    for (int i = 0; i < 9; ++i)
        m(i / 3, i % 3) = i;

    cudaArray_t array;
    auto channed_desc = cudaCreateChannelDesc<lm::gray>();
    cudaMallocArray(&array, &channed_desc, m.width(), m.height());

    cudaMemcpyToArray(array, 0, 0, m.data(), m.size() * sizeof(lm::gray), cudaMemcpyDeviceToDevice);

    cudaTextureDesc tex_desc = {};
    tex_desc.normalizedCoords = false;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.addressMode[0] = cudaAddressModeBorder;
    tex_desc.addressMode[1] = cudaAddressModeBorder;
    tex_desc.readMode = cudaReadModeElementType;

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    cudaTextureObject_t texture;
    cudaCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr);

    kernel <<<1, 1>>> (texture);
    cudaDeviceSynchronize();

    return 0;
}
