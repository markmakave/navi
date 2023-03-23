#include <cuda_runtime.h>

#include "cuda/matrix.cuh"
#include "cuda/brief.cuh"
#include "base/color.hpp"

namespace lm {
namespace cuda {

__global__
void
descript(
    const matrix<gray>                   image, 
    const matrix<bool>                   features,
    const brief<256>                     engine,
          matrix<brief<256>::descriptor> descriptors
) {
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= image.width() - 3 || y >= image.height() - 3 || x < 3 || y < 3)
        return;

    if (features[y][x])
        descriptors[y][x] = engine.descript(x, y, image);
}

}
}
