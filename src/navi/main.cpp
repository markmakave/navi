#include <iostream>

#include <cuda_runtime.h>

#include <slam/detect.hpp>

#include <base/image.hpp>
#include <base/matrix.hpp>
#include <base/color.hpp>

#include <cuda/matrix.cuh>
#include <cuda/kernel.cuh>

#include <util/timer.hpp>

#define ITERATIONS 10000

int main()
{

    {
        lm::image<lm::rgb> l_rgb("../dataset/car.png"),
                           r_rgb("../dataset/car.png");

        lm::image<lm::gray> l_gray(l_rgb.width(), l_rgb.height()),
                            r_gray(r_rgb.width(), r_rgb.height());
                            
        for (unsigned y = 0; y < l_rgb.height(); ++y)
            for (unsigned x = 0; x < l_rgb.width(); ++x)
            {
                l_gray[y][x] = (lm::gray)l_rgb[y][x];
                r_gray[y][x] = (lm::gray)r_rgb[y][x];
            }

        lm::matrix<bool> l_features(l_gray.height(), l_gray.width(), false),
                         r_features(r_gray.height(), r_gray.width(), false);
        
        {
            lm::timer _("cpu", ITERATIONS);

            for (int i = 0; i < ITERATIONS; ++i)
            {
                lm::slam::detect(l_gray, l_features);
                lm::slam::detect(r_gray, r_features);
            }
        }

        for (unsigned y = 2; y < l_features.height() - 2; ++y)
            for (unsigned x = 2; x < l_features.width() - 2; ++x)
            {
                if (l_features[y][x])
                    l_gray.circle(x, y, 2, 255);

                if (r_features[y][x])
                    r_gray.circle(x, y, 2, 255);
            }

        l_gray.write("cpu_l_out.png");
        r_gray.write("cpu_r_out.png");
    }

    {
        lm::image<lm::rgb> l_rgb("../dataset/car.png"), r_rgb("../dataset/car.png");

        lm::image<lm::gray> l_gray(l_rgb.width(), l_rgb.height()),
                            r_gray(r_rgb.width(), r_rgb.height());

        lm::cuda::matrix<lm::gray> dl_gray(l_gray.height(), l_gray.width()),
                                   dr_gray(r_gray.height(), r_gray.width());

        for (unsigned y = 0; y < l_rgb.height(); ++y)
            for (unsigned x = 0; x < l_rgb.width(); ++x)
            {
                l_gray[y][x] = (lm::gray)l_rgb[y][x];
                r_gray[y][x] = (lm::gray)r_rgb[y][x];
            }

        lm::matrix<bool> l_features(l_gray.height(), l_gray.width()),
                         r_features(r_gray.height(), r_gray.width());

        lm::cuda::matrix<bool> dl_features(l_features.height(), l_features.width()),
                               dr_features(r_features.height(), r_features.width());

        void* l_args[] = { &dl_gray, &dl_features };
        void* r_args[] = { &dr_gray, &dr_features };

        cudaStream_t l_stream, r_stream;
        cudaStreamCreate(&l_stream);
        cudaStreamCreate(&r_stream);

        {
            lm::timer _("gpu", ITERATIONS);

            for (int i = 0; i < ITERATIONS; ++i)
            {
                // dl_gray << l_gray;
                // dr_gray << r_gray;

                cudaMemcpyAsync(dl_gray.data(), l_gray.data(), l_gray.size() * sizeof(lm::gray), cudaMemcpyHostToDevice, l_stream);
                cudaMemcpyAsync(dr_gray.data(), r_gray.data(), r_gray.size() * sizeof(lm::gray), cudaMemcpyHostToDevice, r_stream);

                cudaLaunchKernel((void*)lm::cuda::detect, dim3(dl_gray.width() / 8 + 1, dl_gray.height() / 8 + 1), dim3(8, 8), l_args, 0, l_stream);
                cudaLaunchKernel((void*)lm::cuda::detect, dim3(dr_gray.width() / 8 + 1, dr_gray.height() / 8 + 1), dim3(8, 8), r_args, 0, r_stream);

                // dl_features >> l_features;
                // dr_features >> r_features;

                cudaMemcpyAsync(l_features.data(), dl_features.data(), dl_features.size() * sizeof(bool), cudaMemcpyDeviceToHost, l_stream);
                cudaMemcpyAsync(r_features.data(), dr_features.data(), dr_features.size() * sizeof(bool), cudaMemcpyDeviceToHost, r_stream);  
            
                cudaStreamSynchronize(l_stream);
                cudaStreamSynchronize(r_stream);
            }
        }

        for (unsigned y = 2; y < l_features.height() - 2; ++y)
            for (unsigned x = 2; x < l_features.width() - 2; ++x)
            {
                if (l_features[y][x])
                    l_gray.circle(x, y, 2, 255);

                if (r_features[y][x])
                    r_gray.circle(x, y, 2, 255);
            }

        l_gray.write("gpu_l_out.png");
        r_gray.write("gpu_r_out.png");
    }

}
