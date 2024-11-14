#include <iostream>
#include <vector>

#include "util/utility.hpp"
#include "util/timer.hpp"

#include "slam/kernels.hpp"

#include "base/image.hpp"

#include "cuda/tensor.cuh"

int main(int argc, char** argv)
{
    lumina::image<lumina::gray> image;
    image.read("../resource/stereo/Staircase/im0.png");

    lumina::matrix<bool> mask;

    constexpr size_t N = 100;
    
    lumina::timer t;
    for (int i = 0; i < N; ++i)
        lumina::slam::detect(image, mask);
    lumina::log::info(t.elapsed() / N, "s");

    t = lumina::timer();
    image.circle(mask, 2, 255);
    lumina::log::info(t.elapsed(), "s");

    image.write("out.png");

    return 0;
}
