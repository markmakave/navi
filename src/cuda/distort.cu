#include "cuda/kernels.cuh"

template <>
__global__ void
lumina::cuda::distort(const matrix<rgb> in,
                      const __half      k1,
                      const __half      k2,
                      const __half      k3,
                      matrix<rgb>       out)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y >= in.shape()[1] || x >= in.shape()[0])
        return;

    __half one(1);

    __half2 dcord(x, y);
    __half2 dim((long long)in.shape()[0], (long long)in.shape()[1]);
    __half2 ones(1, 1);
    __half2 twos(2, 2);

    __half2 ncord = dcord / dim * twos - ones;

    __half r     = ncord.x * ncord.x + ncord.y * ncord.y;
    __half rsqrd = r * r;
    __half index =
        one +
        (k1 * rsqrd); // + (k2 * rsqrd * rsqrd) + (k3 * rsqrd * rsqrd * rsqrd);
    __half2 indices(index, index);

    __half2 scord = (ncord * indices + ones) * dim / twos;

    int sx = scord.x;
    int sy = scord.y;

    if (sx < 0 || sy < 0 || sx >= in.shape()[0] || sy >= in.shape()[1])
        out(x, y) = {};
    else
        out(x, y) = in(sx, sy);
}

template <>
__global__ void
lumina::cuda::distort(const matrix<rgba> in,
                      const __half       k1,
                      const __half       k2,
                      const __half       k3,
                      matrix<rgba>       out)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y >= in.shape()[1] || x >= in.shape()[0])
        return;

    __half  one(1);
    __half2 ones(one, one), twos(2, 2);

    __half2 v(x, y);
    __half2 dim((int)in.shape()[0], (int)in.shape()[1]);
    __half2 cv = v / twos;
    __half2 nv = v / cv - ones;

    __half  r     = nv.x * nv.x + nv.y * nv.y;
    __half  index = one + (k1 * r); // + (k2 * r * r) + (k3 * r * r * r);
    __half2 indices(index, index);

    __half2 dv = v * indices * dim + cv;

    int sx = dv.x;
    int sy = dv.y;

    if (sx < 0 || sy < 0 || sx >= in.shape()[0] || sy >= in.shape()[1])
        out(x, y) = {};
    else
        out(x, y) = in(sx, sy);
}
