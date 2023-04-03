#include <iostream>
#include <random>

#include "lumina.hpp"
#include "util/timer.hpp"
#include "util/profiler.hpp"

using namespace lm;

#define DATASET "Staircase"

int main()
{
    image<rgb>
        l_frame("../dataset/stereo/" DATASET "/im0.png"),
        r_frame("../dataset/stereo/" DATASET "/im1.png");

    cuda::matrix<gray>
        dl_frame,
        dr_frame;

    dl_frame << l_frame;
    dr_frame << r_frame;

    cuda::stream
        l_stream,
        r_stream;

    // DETECT
    cuda::kernel detect(cuda::detect);

    unsigned
        *l_nfeatures = cuda::managed_allocator<unsigned>::allocate(1),
        *r_nfeatures = cuda::managed_allocator<unsigned>::allocate(1);
    
    *l_nfeatures = 0;
    *r_nfeatures = 0;

    cuda::matrix<bool>
        dl_features(dl_frame.height(), dl_frame.width()),
        dr_features(dr_frame.height(), dr_frame.width());
    
    // DESCRIPT
    cuda::kernel descript(cuda::descript);
    cuda::brief<256> engine;
    cuda::matrix<cuda::brief<256>::descriptor>
        dl_descriptors(dl_frame.height(), dl_frame.width()),
        dr_descriptors(dr_frame.height(), dr_frame.width());

    // MATCH

    {
        detect({dl_frame.width() / 8 + 1, dl_frame.height() / 8 + 1}, {8, 8}, l_stream, dl_frame, 30, l_nfeatures, dl_features);
        // detect({dr_frame.width() / 8 + 1, dr_frame.height() / 8 + 1}, {8, 8}, r_stream, dr_frame, 30, dr_features);

        l_stream.synchronize();
        // r_stream.synchronize();
    }

    std::cout << *l_nfeatures << std::endl;

    matrix<bool> l_features;
    dl_features >> l_features;

    for (int y = 0; y < l_features.height(); ++y)
    {
        for (int x = 0; x < l_features.width(); ++x)
        {
            if (l_features[y][x])
                l_frame.circle(x, y, 2, rgb::random());
        }
    }

    l_frame.write("out.qoi");

    // {
    //     descript({dl_frame.width() / 8 + 1, dl_frame.height() / 8 + 1}, {8, 8}, l_stream, dl_frame, dl_features, engine, dl_descriptors);
    //     descript({dr_frame.width() / 8 + 1, dr_frame.height() / 8 + 1}, {8, 8}, r_stream, dr_frame, dr_features, engine, dr_descriptors);

    //     l_stream.synchronize();
    //     r_stream.synchronize();
    // }



}
