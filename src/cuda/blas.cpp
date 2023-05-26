#include <cuda.h>
#include <cublas_v2.h>
#include "cuda/blas.cuh"

#include <cassert>

static
cublasHandle_t
static_init()
{
    cuInit(0); // preorder cuda context initialization

    cublasHandle_t handle;
    cublasCreate(&handle);
    return handle;
}

static cublasHandle_t handle = static_init();

static
const char*
cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        default: return "unknown error";
    }
}

template <>
void
lm::cuda::blas::axpy(
    const float               alpha,
    const cuda::array<float>& x,
          cuda::array<float>& y,

    const cuda::stream&       stream
) {
    cublasSetStream(handle, stream);

    cublasStatus_t status = cublasSaxpy(
        handle,
        x.size(),
        &alpha,
        x.data(),
        1,
        y.data(),
        1
    );

    if (status != CUBLAS_STATUS_SUCCESS)
        lm::log::error("CUBLAS axpy error:", cublasGetErrorString(status));

    stream.synchronize();

    cublasSetStream(handle, cuda::stream::main);
}

template <>
int
lm::cuda::blas::amax(
    const array<float>& x,

    const stream&   stream
) {
    cublasSetStream(handle, stream);

    int max;
    cublasStatus_t status = cublasIsamax(
        handle,
        x.size(),
        x.data(),
        1,
        &max
    );

    if (status != CUBLAS_STATUS_SUCCESS)
        lm::log::error("CUBLAS axpy error:", cublasGetErrorString(status));

    stream.synchronize();

    cublasSetStream(handle, cuda::stream::main);

    return max;
}

template <>
float
lm::cuda::blas::nrm2(
    const array<float>& x,

    const stream&   stream
) {
    cublasSetStream(handle, stream);

    float nrm;
    cublasStatus_t status = cublasSnrm2(
        handle, x.size(), x.data(), 1, &nrm
    );

    if (status != CUBLAS_STATUS_SUCCESS)
        lm::log::error("CUBLAS axpy error:", cublasGetErrorString(status));

    stream.synchronize();

    cublasSetStream(handle, cuda::stream::main);

    return nrm;
}

template <>
void
lm::cuda::blas::mv(
    const matrix<float>& A,
    const array<float>&  x,
          array<float>&  y,

    const bool           A_transpose,
    const stream&        stream
) {
    
    float alpha = 1, beta = 0;
    cublasSetStream(handle, stream);
    cublasStatus_t status;

    if (A_transpose)
    {
        assert(x.size() == A.shape()[1]);
        y.reshape(A.shape()[0]);

        status = cublasSgemv(
            handle,
            CUBLAS_OP_N, // NOT A BUG
            A.shape()[0],
            A.shape()[1],
            &alpha,
            A.data(),
            A.shape()[0],
            x.data(),
            1,
            &beta,
            y.data(),
            1
        );
    } else {
        assert(x.size() == A.shape()[0]);
        y.reshape(A.shape()[1]);

        status = cublasSgemv(
            handle,
            CUBLAS_OP_T, // NOT A BUG
            A.shape()[0],
            A.shape()[1],
            &alpha,
            A.data(),
            A.shape()[0],
            x.data(),
            1,
            &beta,
            y.data(),
            1
        );
    }

    if (status != CUBLAS_STATUS_SUCCESS)
        lm::log::error("CUBLAS axpy error:", cublasGetErrorString(status));

    stream.synchronize();

    cublasSetStream(handle, stream::main);
}

template <>
void
lm::cuda::blas::ger(
    const array<float>&  x,
    const array<float>&  y,
          float          alpha,
          matrix<float>& A,

    const stream&        stream
) {
    A.reshape(y.size(), x.size());
    
    cublasSetStream(handle, stream);

    cublasStatus_t status = cublasSger(
        handle,
        y.size(),
        x.size(),
        &alpha,
        y.data(),
        1,
        x.data(),
        1,
        A.data(),
        A.shape()[0]
    );

    if (status != CUBLAS_STATUS_SUCCESS)
        lm::log::error("CUBLAS axpy error:", cublasGetErrorString(status));

    stream.synchronize();

    cublasSetStream(handle, stream::main);
}

template <>
void
lm::cuda::blas::mm(
    const matrix<float>& A,
    const matrix<float>& B,
          matrix<float>& C,

    const stream&        stream
) {
    float alpha = 1, beta = 0;

    cublasSetStream(handle, stream);

    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        A.shape()[1],
        A.shape()[0],
        B.shape()[0],
        &alpha,
        A.data(),
        A.shape()[0],
        B.data(),
        B.shape()[0],
        &beta,
        C.data(),
        C.shape()[0]
    );

    if (status != CUBLAS_STATUS_SUCCESS)
        lm::log::error("CUBLAS axpy error:", cublasGetErrorString(status));

    stream.synchronize();

    cublasSetStream(handle, stream::main);
}

template <>
void
lm::cuda::blas::add(
    const array<float>& x,
    const array<float>& y,
          array<float>& z,

    const stream&       stream
) {
    assert(x.shape() == y.shape());
    z.reshape(x.shape());

    kernel add_k(add_kernel<float>);
    add_k(x.size() / 8 + 1, 8, stream, x, y, z);
    stream.synchronize();
}

template <>
void
lm::cuda::blas::sub(
    const array<float>& x,
    const array<float>& y,
          array<float>& z,

    const stream&       stream
) {
    assert(x.shape() == y.shape());
    z.reshape(x.shape());

    kernel sub_k(sub_kernel<float>);
    sub_k(x.size() / 8 + 1, 8, stream, x, y, z);
    stream.synchronize();
}

template <>
void
lm::cuda::blas::mul(
    const array<float>& x,
    const array<float>& y,
          array<float>& z,

    const stream&       stream
) {
    assert(x.shape() == y.shape());
    z.reshape(x.shape());

    kernel mul_k(mul_kernel<float>);
    mul_k(x.size() / 8 + 1, 8, stream, x, y, z);
    stream.synchronize();
}

template <>
void
lm::cuda::blas::sigmoid(
    const array<float>& x,
          array<float>& y,

    const stream&   stream
) {
    y.reshape(x.shape());
    kernel sigmoid_k(sigmoid_kernel<float>);
    sigmoid_k(x.size() / 8 + 1, 8, stream, x, y);
    stream.synchronize();
}

template <>
void
lm::cuda::blas::sigmoid_derivative(
    const array<float>& x,
          array<float>& y,

    const stream&   stream
) {
    y.reshape(x.shape());
    kernel sigmoid_derivative_k(sigmoid_derivative_kernel<float>);
    sigmoid_derivative_k(x.size() / 8 + 1, 8, stream, x, y);
    stream.synchronize();
}
