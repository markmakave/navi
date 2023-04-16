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
namespace cuda {
namespace blas {

// L1

template <typename T>
int
amax(
    const array<T>& x,
    
    const stream&   stream = stream::main
);

template <typename T>
int
amin(
    const array<T>& x, 

    const stream&   stream = stream::main
);

template <typename T>
T
asum(
    const array<T>& x,

    const stream&   stream = stream::main
);

template <typename T>
void
axpy(
    const T         alpha,
    const array<T>& x, 
          array<T>& y, 

    const stream&   stream = stream::main
);

template <typename T>
void
copy(
    const array<T>& x,
          array<T>& y,

    const stream&   stream = stream::main
);

template <typename T>
T
dot(
    const array<T>& x,
    const array<T>& y,

    const stream&   stream = stream::main
);

template <typename T>
T
nrm2(
    const array<T>& x,

    const stream&   stream = stream::main
);

// rot
// rotg
// rotm
// rotmg
// scal

template <typename T>
void
swap(
    array<T>& x,
    array<T>& y
);

// L2

template <typename T>
void
mv(
    const matrix<T>& A,
    const array<T>&  x,
          array<T>&  y,

    const bool       A_transpose = false,
    const stream&    stream = stream::main
);

template <typename T>
void
ger(
    const array<T>&  x,
    const array<T>&  y,
          T          alpha,
          matrix<T>& A,

    const stream&    stream = stream::main
);

// L3

template <typename T>
void
mm(
    const matrix<T>& A,
    const matrix<T>& B,
          matrix<T>& C,

    const stream&    stream = stream::main
);

// Universal

template <typename T>
__global__
void
add_kernel(
    const lm::cuda::array<T> x,
    const lm::cuda::array<T> y,
          lm::cuda::array<T> z
);

template <typename T>
void
add(
    const array<T>& x,
    const array<T>& y,
          array<T>& z,

    const stream&   stream = stream::main
);

template <typename T>
__global__
void
sub_kernel(
    const lm::cuda::array<T> x,
    const lm::cuda::array<T> y,
          lm::cuda::array<T> z
);

template <typename T>
void
sub(
    const array<T>& x,
    const array<T>& y,
          array<T>& z,

    const stream&   stream = stream::main
);

template <typename T>
__global__
void
mul_kernel(
    const lm::cuda::array<T> x,
    const lm::cuda::array<T> y,
          lm::cuda::array<T> z
);

template <typename T>
void
mul(
    const array<T>& x,
    const array<T>& y,
          array<T>& z,

    const stream&   stream = stream::main
);

__device__
float
sigmoid(float x);

__device__
float
sigmoid_derivative(float x);

template <typename T>
__global__
void
sigmoid_kernel(
    const array<T> x,
          array<T> y
);

template <typename T>
__global__
void
sigmoid_derivative_kernel(
    const array<T> x,
          array<T> y
);

template <typename T>
void
sigmoid(
    const array<T>& x,
          array<T>& y,

    const stream&   stream = stream::main
);

template <typename T>
void
sigmoid_derivative(
    const array<T>& x,
          array<T>& y,

    const stream&   stream = stream::main
);

}
}
}
