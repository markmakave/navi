#include <iostream>
#include <vector>
#include <fstream>

#include "util/timer.hpp"

#include "base/image.hpp"
#include "slam/kernels.hpp"

#include "cuda/blas.cuh"
#include "cuda/tensor.cuh"
#include "cuda/kernel.cuh"
#include "cuda/kernels.cuh"
#include "cuda/graph.cuh"

namespace lm = lumina;

int main(int argc, char** argv)
{
    lm::image<lm::gray> image;
    image.read("../resource/stereo/photo.png");
    lm::matrix<bool> mask(image.shape());

    lm::cuda::matrix<lm::gray> d_image;
    lm::cuda::matrix<bool> d_mask(image.shape());

    constexpr size_t N = 100000;

    lm::cuda::kernel detect_k(lm::cuda::detect);
    auto detect = detect_k.bind({d_image.shape()[0] / 8 + 1, d_image.shape()[1] / 8 + 1}, {8, 8}, lumina::cuda::stream::main, 0);

    d_image << image;
    detect(d_image, 8, d_mask);
    lm::cuda::stream::main.synchronize();
    d_mask >> mask;

    std::ofstream of("time.csv");
    of << "h2d,kernel,d2h\n";

    double h2d, kernel, d2h;
    
    for (int i = 0; i < N; ++i)
    {
        lm::timer t;
        d_image << image;
        h2d = t.elapsed();

        detect(d_image, 8, d_mask);
        lm::cuda::stream::main.synchronize();
        kernel = t.elapsed() - h2d;

        d_mask >> mask;
        d2h = t.elapsed() - h2d - kernel;

        //

        of << h2d << ',' << kernel << ',' << d2h << '\n';
    }

    d_mask >> mask;

    lm::matrix<bool> reference;


    lm::slam::detect(image, 8, reference);

    lm::log::info("Number of mismatches:", (reference != mask).count());

    return 0;
}
