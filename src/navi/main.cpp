#include <iostream>
#include <random>

#define NDEBUG

#include "util/timer.hpp"
#include "util/profiler.hpp"

#include "base/tensor.hpp"
#include "base/memory.hpp"

#include "cuda/tensor.cuh"
#include "cuda/cuda.cuh"
#include "cuda/matrix.cuh"

using namespace lm;

int main()
{
    tensor<3, int, heap_allocator<int>> t(5, 5, 5);

    for (int i = 0; i < t.size(); ++i)
        t.data()[i] = i;

    std::cout << t;
}
