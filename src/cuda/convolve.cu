#include "cuda/kernels.cuh"

namespace lumina {
namespace cuda {

template <i64 N, typename T, typename K, typename U>
__global__ void
convolve(const tensor<N, T> in, const tensor<N, K> kernel, tensor<N, U> out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // kernel caching
    extern __shared__ K kernel_cache[];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < kernel.size(); ++i)
            kernel_cache[i] = kernel[i];
    }
    __syncthreads();

    const int kx_half = kernel.shape()[0] / 2;
    const int ky_half = kernel.shape()[1] / 2;
    const int kz_half = kernel.shape()[2] / 2;

    if (x < kx_half || y < ky_half || x >= in.shape()[0] - kx_half ||
        y >= in.shape()[1] - ky_half)
        return;

    auto op = [&](int x, int y, int z) {
        using accum_t = decltype(in(0, 0, 0) * kernel(0, 0, 0));
        accum_t accum = 0;

        for (int kz = -kz_half; kz <= kz_half; ++kz) {
            int iz_index = (z + kz) * in.shape()[0] * in.shape()[1];
            int kz_index =
                (kz + kz_half) * kernel.shape()[0] * kernel.shape()[1];

            for (int ky = -ky_half; ky <= ky_half; ++ky) {
                int iyz_index = (y + ky) * in.shape()[0] + iz_index;
                int kyz_index = (ky + ky_half) * kernel.shape()[1] + kz_index;

                for (int kx = -kx_half; kx <= kx_half; ++kx) {
                    int ix_index = (x + kx);
                    int kx_index = (kx + kx_half);

                    auto in_value     = in[iyz_index + ix_index];
                    auto kernel_value = kernel_cache[kyz_index + kx_index];

                    accum += in_value * kernel_value;
                }
            }
        }

        auto clamp = [](auto x, auto min, auto max) {
            return x < min ? min : x > max ? max
                                           : x;
        };

        out(x, y, z) = clamp(accum, 0, 255);
    };

    for (int z = kz_half; z < in.shape()[2] - kz_half; ++z)
        op(x, y, z);
}

template __global__ void
convolve<3, byte, float, byte>(const tensor<3, byte>  in,
                               const tensor<3, float> kernel,
                               tensor<3, byte>        out);

} // namespace cuda
} // namespace lumina
