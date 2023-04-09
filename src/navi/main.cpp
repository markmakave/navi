#include <iostream>
#include <random>

#define NDEBUG

#include "util/timer.hpp"
#include "util/profiler.hpp"

#include "base/tensor.hpp"
#include "base/memory.hpp"

#include "cuda/tensor.cuh"
#include "cuda/memory.cuh"

using namespace lm;

int main()
{
    
    tensor<3, float, stack_allocator<float>> t(10, 10, 3);

    for (int z = 0; z < t.shape()[2]; ++z)
    {
        for (int y = 0; y < t.shape()[1]; ++y)
        {
            for (int x = 0; x < t.shape()[0]; ++x)
                std::cout << t(x, y ,z) << ' ';
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

}
