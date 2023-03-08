#include <cuda.h>

#include "cuda/matrix.cuh"

namespace lm {
namespace cuda {

__global__
void
detect(const lm::cuda::matrix<float> input, lm::cuda::matrix<bool> output)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= input.width() || y >= input.height() || x < 2 || y < 2)
        return;

    output[y][x] = fast(input, y, x);
}

}
}
