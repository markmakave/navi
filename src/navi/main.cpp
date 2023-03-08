#include <iostream>
#include <cmath>

#include <base/matrix.hpp>
#include <cuda/matrix.cuh>
#include <base/image.hpp>

#include "cuda/kernel.cuh"

int main()
{
    lm::cuda::matrix<float> dm(10, 10);
    
    for (unsigned y = 0; y < dm.height(); ++y)
    {
        for (unsigned x = 0; x < dm.width(); ++x)
        {
            dm(y, x) = y * dm.width() + x;
        }
    }

}
