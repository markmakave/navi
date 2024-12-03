#include <functional>
#include <iostream>
#include <random>
#include <fstream>

#include "cuda/texture.cuh"
#include "cuda/kernel.cuh"

#include "base/shape.hpp"

__managed__ float x;

namespace lm = lumina;

template <typename... Dims>
__global__
void kernel(lm::cuda::texture<sizeof...(Dims), float> tex, Dims... dims)
{
    x = tex(dims...);
}

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    lm::cuda::texture<2, float> tex(1, 2);

    lm::cuda::kernel k(kernel<float, float>);

    std::ofstream csv("interpolation.csv");
    csv << "u,v,tex\n";
    for (float u = -0.5f; u <= 1.5f; u += 0.01)
        for (float v = -0.5f; v <= 1.5f; v += 0.01)
        {
            k.bind(dim3(1, 1, 1), dim3(1, 1, 1), lm::cuda::stream::main)(tex, u, v);
            lm::cuda::stream::main.synchronize();

            csv << u << ',' << v << ',' << x << '\n';
        }

    return 0;
}
