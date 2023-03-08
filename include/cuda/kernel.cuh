#pragma once

#include <cuda_runtime.h>
#include "cuda/matrix.cuh"

namespace lm {
namespace cuda {

__global__
void
detect(matrix<float> input, matrix<bool> output);

}
}
