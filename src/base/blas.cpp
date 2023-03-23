#include "lumina.hpp"
#include <cuda.h>

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

template <>
void
lm::blas::axpy(
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
        lm::log::error("CUBLAS axpy error");

    stream.synchronize();

    cublasSetStream(handle, cuda::stream::main);
}

template <>
void
lm::blas::axpy(
    const double               alpha,
    const cuda::array<double>& x,
          cuda::array<double>& y,
    const cuda::stream&        stream
) {
    cublasSetStream(handle, stream);

    cublasStatus_t status = cublasDaxpy(
        handle,
        x.size(),
        &alpha,
        x.data(),
        1,
        y.data(),
        1
    );

    if (status != CUBLAS_STATUS_SUCCESS)
        lm::log::error("CUBLAS axpy error");

    stream.synchronize();

    cublasSetStream(handle, cuda::stream::main);
}

template <>
void
lm::blas::mv(
    const cuda::matrix<float>& A,
    const cuda::array<float>&  x,
          cuda::array<float>&  y,
    const cuda::stream&        stream
) {
    float alpha = 1, beta = 0;

    cublasSetStream(handle, stream);

    cublasStatus_t status = cublasSgemv(
        handle,
        CUBLAS_OP_N,
        A.height(),
        A.width(),
        &alpha,
        A.data(),
        A.width(),
        x.data(),
        1,
        &beta,
        y.data(),
        1
    );

    if (status != CUBLAS_STATUS_SUCCESS)
        lm::log::error("CUBLAS axpy error");

    stream.synchronize();

    cublasSetStream(handle, cuda::stream::main);
}

template <>
void
lm::blas::mv(
    const cuda::matrix<double>& A,
    const cuda::array<double>&  x,
          cuda::array<double>&  y,
    const cuda::stream&         stream
) {
    double alpha = 1, beta = 0;

    cublasSetStream(handle, stream);

    cublasStatus_t status = cublasDgemv(
        handle,
        CUBLAS_OP_N,
        A.height(),
        A.width(),
        &alpha,
        A.data(),
        A.width(),
        x.data(),
        1,
        &beta,
        y.data(),
        1
    );

    if (status != CUBLAS_STATUS_SUCCESS)
        lm::log::error("CUBLAS axpy error");

    stream.synchronize();

    cublasSetStream(handle, cuda::stream::main);
}

template <>
void
lm::blas::mm(
    const cuda::matrix<float>& A,
    const cuda::matrix<float>& B,
          cuda::matrix<float>& C,
    const cuda::stream&        stream
) {
    float alpha = 1, beta = 0;

    cublasSetStream(handle, stream);

    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        A.height(),
        A.width(),
        B.width(),
        &alpha,
        A.data(),
        A.width(),
        B.data(),
        B.width(),
        &beta,
        C.data(),
        C.width()
    );

    if (status != CUBLAS_STATUS_SUCCESS)
        lm::log::error("CUBLAS axpy error");

    stream.synchronize();

    cublasSetStream(handle, cuda::stream::main);
}

template <>
void
lm::blas::mm(
    const cuda::matrix<double>& A,
    const cuda::matrix<double>& B,
          cuda::matrix<double>& C,
    const cuda::stream&         stream
) {
    double alpha = 1, beta = 0;

    cublasSetStream(handle, stream);

    cublasStatus_t status = cublasDgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        A.height(),
        A.width(),
        B.width(),
        &alpha,
        A.data(),
        A.width(),
        B.data(),
        B.width(),
        &beta,
        C.data(),
        C.width()
    );

    if (status != CUBLAS_STATUS_SUCCESS)
        lm::log::error("CUBLAS axpy error");

    stream.synchronize();

    cublasSetStream(handle, cuda::stream::main);
}
