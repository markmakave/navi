#include <iostream>
#include <cmath>

#include <base/matrix.hpp>
#include <cuda/matrix.cuh>
#include <base/image.hpp>
#include <cuda/kernel.cuh>
#include <util/timer.hpp>

int main()
{

    lm::image<lm::rgb> rgb("../dataset/test.png");

    lm::image<lm::gray> gray(rgb.width(), rgb.height());
    for (unsigned y = 0; y < rgb.height(); ++y)
        for (unsigned x = 0; x < rgb.width(); ++x)
            gray[y][x] = (lm::gray)rgb[y][x];

    lm::cuda::matrix<lm::gray> din;
    lm::cuda::matrix<bool> dout;
    din << gray;
    dout.resize(din.height(), din.width());

    void* args[] = {&din, &dout};
    cudaLaunchKernel((void*)lm::cuda::detect, dim3(din.width() / 8 + 1, din.height() / 8 + 1), dim3(8, 8), args, 0, 0);
    cudaDeviceSynchronize();

    {
        lm::timer _("compute");

        cudaLaunchKernel((void*)lm::cuda::detect, dim3(din.width() / 8 + 1, din.height() / 8 + 1), dim3(8, 8), args, 0, 0);
        cudaDeviceSynchronize();
    }

    lm::matrix<bool> out;
    dout >> out;

    for (unsigned y = 0; y < rgb.height(); ++y)
        for (unsigned x = 0; x < rgb.width(); ++x)
            if (out[y][x])
                gray.circle(x, y, 2, 255);

    gray.write("out.png");
}
