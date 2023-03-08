#pragma once

#include <cuda_runtime.h>
#include "cuda/matrix.cuh"

namespace lm {
namespace cuda {

__global__
void
detect(matrix<lm::gray> input, matrix<bool> output);

}
}
