/* 

    Copyright (c) 2023 Mark Mokhov

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

*/

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>

#include "cuda/matrix.cuh"
#include "cuda/brief.cuh"
#include "cuda/color.cuh"

namespace lm {
namespace cuda {

__global__
void
detect(
    const matrix<lm::gray> image, 
    const int              threshold,
          unsigned*        nfeatures,
          matrix<bool>     features
);

__global__
void
descript(
    const matrix<gray>                   image, 
    const matrix<bool>                   features,
    const brief<256>                     engine,
          matrix<brief<256>::descriptor> descriptors
);

template <typename T>
__global__
void
distort(
    const matrix<T> in,
    const __half    k1,
    const __half    k2, 
    const __half    k3,
          matrix<T> out
);

}
}
