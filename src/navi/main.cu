#include <functional>
#include <iostream>
#include <random>
#include <fstream>

#include "cuda/texture.cuh"
#include "cuda/kernel.cuh"

__managed__ float x;

__global__
void kernel(lumina::cuda::texture<1, float> tex, float u)
{
    x = tex[u];
}

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    lumina::cuda::texture<1, float> tex;

    lumina::cuda::kernel k(kernel);

    std::ofstream csv("interpolation.csv");
    csv << "x,y\n";
    for (float u = -1.f; u <= 2.f; u += 0.001)
    {
        k.bind(dim3(1, 1, 1), dim3(1, 1, 1), lumina::cuda::stream::main, 0)(tex, u);
        lumina::cuda::stream::main.synchronize();

        csv << u << ',' << x << '\n';
    }

    return 0;
}
