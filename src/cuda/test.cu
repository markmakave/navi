#include <cuda_runtime.h>

#include <stdio.h>

namespace lm {
namespace cuda {

__global__
void
test(int x, int y, int z)
{
    x = threadIdx.x + blockIdx.x * blockDim.x;
    y = threadIdx.y + blockIdx.y * blockDim.y;
    z = threadIdx.z + blockIdx.z * blockDim.z;

    float len = sqrtf(x*x + y*y + z*z);

    printf("Length of [%d, %d, %d] is %f\n", x, y, z, len);
}

}
}
