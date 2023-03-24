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

#include <cublas_v2.h>

#include "cuda/array.cuh"
#include "cuda/matrix.cuh"

namespace lm {
namespace blas {

// L1

template <typename T>
void
amax(
    const cuda::array<T>& x,
          T&              max,
    const cuda::stream&   stream = cuda::stream::main
);

template <typename T>
void
amin(
    const cuda::array<T>& x, 
          T&              min,
    const cuda::stream&   stream = cuda::stream::main
);

template <typename T>
void
asum(
    const cuda::array<T>& x,
          T&              sum,
    const cuda::stream&   stream = cuda::stream::main
);

template <typename T>
void
axpy(
    const T               alpha,
    const cuda::array<T>& x, 
          cuda::array<T>& y, 
    const cuda::stream&   stream = cuda::stream::main
);

template <typename T>
void
copy(
    const cuda::array<T>& x,
          cuda::array<T>& y,
    const cuda::stream&   stream = cuda::stream::main
);

template <typename T>
void
dot(
    const cuda::array<T>& x,
    const cuda::array<T>& y,
          T&              res,
    const cuda::stream&   stream = cuda::stream::main
);

template <typename T>
void
nrm2(
    const cuda::array<T>& x,
          T&              nrm,
    const cuda::stream&   stream = cuda::stream::main
);

// rot
// rotg
// rotm
// rotmg
// scal

template <typename T>
void
swap(
    cuda::array<T>& x,
    cuda::array<T>& y
);

// L2

template <typename T>
void
mv(
    const cuda::matrix<T>& A,
    const cuda::array<T>&  x,
          cuda::array<T>&  y,
    const cuda::stream&    stream = cuda::stream::main
);

// L3

template <typename T>
void
mm(
    const cuda::matrix<T>& A,
    const cuda::matrix<T>& B,
          cuda::matrix<T>& C,
    const cuda::stream&    stream = cuda::stream::main
);

}
}
