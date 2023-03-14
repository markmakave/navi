#include <iostream>

#include <cuda_runtime.h>

#include <slam/detect.hpp>

#include <base/image.hpp>
#include <base/matrix.hpp>
#include <base/color.hpp>

#include <cuda/cuda.hpp>

#include <cuda/matrix.cuh>
#include <cuda/kernel.cuh>

#include <util/timer.hpp>
#include <util/profiler.hpp>

#define ITERATIONS 10000
#define RADIUS 2

#define LM_FRAMEWORK_VERSION(v) using namespace lm

LM_FRAMEWORK_VERSION(1.0);

int main()
{
    profiler::begin("trace.json");

    // CPU
    LM_PROFILE("CPU")
    {
        image<rgb> l_rgb("../dataset/car.png"),
                   r_rgb("../dataset/car.png");

        image<gray> l_gray(l_rgb), 
                    r_gray(r_rgb);

        matrix<bool> l_features(l_gray.height(), l_gray.width(), false),
                     r_features(r_gray.height(), r_gray.width(), false);
        
        {
            timer timer("cpu", ITERATIONS);

            for (int i = 0; i < ITERATIONS; ++i)
            {
                slam::detect(l_gray, l_features);
                slam::detect(r_gray, r_features);
            }
        }

        for (unsigned y = RADIUS; y < l_features.height() - RADIUS; ++y)
            for (unsigned x = RADIUS; x < l_features.width() - RADIUS; ++x)
            {
                if (l_features[y][x])
                    l_gray.circle(x, y, RADIUS, 255);

                if (r_features[y][x])
                    r_gray.circle(x, y, RADIUS, 255);
            }

        l_gray.write("cpu_l_out.png");
        r_gray.write("cpu_r_out.png");
    }

    // GPU
    LM_PROFILE("GPU")
    {
        image<rgb> l_rgb("../dataset/car.png"),
                   r_rgb("../dataset/car.png");

        image<gray> l_gray(l_rgb), 
                    r_gray(r_rgb);

        cuda::matrix<gray> dl_gray(l_gray.height(), l_gray.width()),
                           dr_gray(r_gray.height(), r_gray.width());

        matrix<bool> l_features(l_gray.height(), l_gray.width()),
                     r_features(r_gray.height(), r_gray.width());

        cuda::matrix<bool> dl_features(l_features.height(), l_features.width()),
                           dr_features(r_features.height(), r_features.width());

        cuda::stream l_stream, r_stream;
        cuda::kernel detect(cuda::detect);

        {
            timer _("gpu", ITERATIONS);

            for (int i = 0; i < ITERATIONS; ++i)
            {
                cuda::memcpy_async(dl_gray.data(), l_gray.data(), l_gray.size() * sizeof(gray), cuda::H2D, l_stream);
                cuda::memcpy_async(dr_gray.data(), r_gray.data(), r_gray.size() * sizeof(gray), cuda::H2D, r_stream);

                detect({dl_gray.width() / 8 + 1, dl_gray.height() / 8 + 1}, {8, 8}, l_stream, dl_gray, dl_features);
                detect({dr_gray.width() / 8 + 1, dr_gray.height() / 8 + 1}, {8, 8}, r_stream, dr_gray, dr_features);

                cuda::memcpy_async(l_features.data(), dl_features.data(), dl_features.size() * sizeof(bool), cuda::D2H, l_stream);
                cuda::memcpy_async(r_features.data(), dr_features.data(), dr_features.size() * sizeof(bool), cuda::D2H, r_stream);

                l_stream.synchronize();
                r_stream.synchronize();
            }
        }

        for (unsigned y = RADIUS; y < l_features.height() - RADIUS; ++y)
            for (unsigned x = RADIUS; x < l_features.width() - RADIUS; ++x)
            {
                if (l_features[y][x])
                    l_gray.circle(x, y, RADIUS, 255);

                if (r_features[y][x])
                    r_gray.circle(x, y, RADIUS, 255);
            }

        l_gray.write("gpu_l_out.png");
        r_gray.write("gpu_r_out.png");
    }

    profiler::end();
}
