#include <functional>
#include <iostream>
#include <random>

#define NDEBUG

#include "lumina.hpp"

#include "cuda/kernels.cuh"

using namespace lumina;

#define N 7
#define ITERATIONS 100

int
main(int argc, char** argv)
{
    image<rgb>      raw(argv[1]);
    tensor<3, byte> image(raw.shape()[0], raw.shape()[1], 3);

    LM_PROFILE("Channel transform")
    for (int i = 0; i < raw.shape()[0]; ++i)
        for (int j = 0; j < raw.shape()[1]; ++j) {
            image(i, j, 0) = raw(i, j).r;
            image(i, j, 1) = raw(i, j).g;
            image(i, j, 2) = raw(i, j).b;
        }

    cuda::tensor<3, byte> d_image;

    LM_PROFILE("Transfer H2D")
    d_image << image;

    cuda::tensor<3, float> d_kernel(3, 3, 1);
    d_kernel(0, 0, 0) = -1;
    d_kernel(1, 0, 0) = -1;
    d_kernel(2, 0, 0) = -1;

    d_kernel(0, 1, 0) = -1;
    d_kernel(1, 1, 0) = 8;
    d_kernel(2, 1, 0) = -1;

    d_kernel(0, 2, 0) = -1;
    d_kernel(1, 2, 0) = -1;
    d_kernel(2, 2, 0) = -1;

    cuda::tensor<3, byte> d_result(
        image.shape()[0], image.shape()[1], image.shape()[2]);

    cuda::kernel convolve(cuda::convolve<3, byte, float, byte>);

    auto convolve_launchable = convolve.bind({image.shape()[0] / 8 + 1, image.shape()[1] / 8 + 1},
                                             {8, 8},
                                             cuda::stream::main,
                                             d_kernel.size() * sizeof(float));

    convolve_launchable(d_image, d_kernel, d_result);
    cuda::stream::main.synchronize();

    {
        LM_PROFILE("Workload")
        for (int i = 0; i < ITERATIONS; ++i) {
            LM_PROFILE("Convolve")
            {
                convolve_launchable(d_image, d_kernel, d_result);
                cuda::stream::main.synchronize();
            }
        }
    }

    LM_PROFILE("Transfer D2H")
    d_result >> image;

    LM_PROFILE("Channel transform")
    #pragma omp parallel for
    for (int i = 0; i < image.shape()[0]; ++i)
        for (int j = 0; j < image.shape()[1]; ++j) {
            raw(i, j).r = image(i, j, 0);
            raw(i, j).g = image(i, j, 1);
            raw(i, j).b = image(i, j, 2);
        }

    raw.write("convolve.png");

    profiler::stage("trace.json");
}
