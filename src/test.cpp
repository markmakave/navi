#include "cuda/cuda.hpp"

#include "cuda/kernel.cuh"

using namespace lm;

int main()
{
    cuda::kernel kernel(lm::cuda::test);
    kernel({2, 2, 2}, {2, 2, 2}, cuda::stream::main, 1, 2, 3);
    cuda::stream::main.synchronize();
}
