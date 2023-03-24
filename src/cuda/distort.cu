#include "cuda/kernel.cuh"

__global__
void
lm::cuda::distort(
    const matrix<rgb> in,
    const double       k1,
    const double       k2, 
    const double       k3,
          matrix<rgb> out
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (y >= in.height() || x >= in.width())
        return;

    auto index = [&](double r) -> double {
        return 1 + k1*r*r + k2*r*r*r*r + k3*r*r*r*r*r*r;
    };

    auto normalize = [](int x, int dimention) -> double {
        return double(x) / dimention * 2 - 1;
    };

    auto unnormalize = [](double x, int dimention) -> double {
        return (x + 1) * dimention / 2;
    };

    double x_normalized = normalize(x, in.width());
    double y_normalized = normalize(y, in.height());

    double r = x_normalized*x_normalized + y_normalized*y_normalized;

    int x_source = unnormalize(x_normalized * index(r), in.width());
    int y_source = unnormalize(y_normalized * index(r), in.height());

    if (x_source < 0 || y_source < 0 || x_source >= in.width() || y_source >= in.height())
        out[y][x] = 0;
    else
        out[y][x] = in[y_source][x_source];
}
