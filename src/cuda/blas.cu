#include "cuda/blas.cuh"

template <>
__global__
void
lm::cuda::blas::add_kernel(
    const lm::cuda::array<float> x,
    const lm::cuda::array<float> y,
          lm::cuda::array<float> z
) {
    i64 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= x.size()) return;
    z(i) = x(i) + y(i);
}

template <>
__global__
void
lm::cuda::blas::sub_kernel(
    const lm::cuda::array<float> x,
    const lm::cuda::array<float> y,
          lm::cuda::array<float> z
) {
    i64 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= x.size()) return;
    z(i) = x(i) - y(i);
}

template <>
__global__
void
lm::cuda::blas::mul_kernel(
    const lm::cuda::array<float> x,
    const lm::cuda::array<float> y,
          lm::cuda::array<float> z
) {
    i64 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= x.size()) return;
    z(i) = x(i) * y(i);
}

__device__
float
lm::cuda::blas::sigmoid(float x)
{
    return x > 0 ? x : 0;
}

__device__
float
lm::cuda::blas::sigmoid_derivative(float x)
{
    return x > 0 ? 1 : 0;
}

template <>
__global__
void
lm::cuda::blas::sigmoid_kernel(
    const array<float> x,
          array<float> y
) {
    i64 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= x.size()) return;
    y(i) = sigmoid(x(i));
}

template <>
__global__
void
lm::cuda::blas::sigmoid_derivative_kernel(
    const array<float> x,
          array<float> y
) {
    i64 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= x.size()) return;
    y(i) = sigmoid_derivative(x(i));
}
