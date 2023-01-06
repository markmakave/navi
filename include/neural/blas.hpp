#pragma once

#include <functional>
#include <base/matrix.hpp>

namespace lm {
namespace neural {

template <typename T>
matrix<T>&
matmul(const matrix<T>& a, const matrix<T>& b, matrix<T>& c)
{
    if (a.width() != b.height())
    {
        throw std::runtime_error("matrix size mismatch");
    }

    c.resize(a.height(), b.width());

    #pragma omp parallel for collapse(2) schedule(static)
    for (unsigned i = 0; i < a.height(); ++i)
    {
        auto a_row = a[i];
        auto c_row = c[i];

        for (unsigned j = 0; j < b.width(); ++j)
        {
            T sum = 0;

            #pragma omp simd reduction(+:sum)
            for (unsigned k = 0; k < a.width(); ++k)
            {
                sum += a_row[k] * b[k][j];
            }

            c_row[j] = sum;
        }
    }

    return c;
}

template <typename T>
matrix<T>&
matadd(const matrix<T>& a, const matrix<T>& b, matrix<T>& c)
{
    if (a.width() != b.width() || a.height() != b.height())
    {
        throw std::runtime_error("matrix size mismatch");
    }

    c.resize(a.height(), a.width());

    #pragma omp parallel for
    for (unsigned i = 0; i < a.height(); ++i)
    {
        auto a_row = a[i];
        auto b_row = b[i];
        auto c_row = c[i];

        #pragma omp simd
        for (unsigned j = 0; j < a.width(); ++j)
        {
            c_row[j] = a_row[j] + b_row[j];
        }
    }

    return c;
}

template <typename T>
matrix<T>&
matsub(const matrix<T>& a, const matrix<T>& b, matrix<T>& c)
{
    if (a.width() != b.width() || a.height() != b.height())
    {
        throw std::runtime_error("matrix size mismatch");
    }

    c.resize(a.height(), a.width());

    #pragma omp parallel for
    for (unsigned i = 0; i < a.height(); ++i)
    {
        auto a_row = a[i];
        auto b_row = b[i];
        auto c_row = c[i];

        #pragma omp simd
        for (unsigned j = 0; j < a.width(); ++j)
        {
            c_row[j] = a_row[j] - b_row[j];
        }
    }

    return c;
}


template <typename T>
matrix<T>&
matmap(const matrix<T>& a, std::function<T(T)> f, matrix<T>& c)
{
    c.resize(a.height(), a.width());

    #pragma omp parallel for
    for (unsigned i = 0; i < a.height(); ++i)
    {
        auto a_row = a[i];
        auto c_row = c[i];

        #pragma omp simd
        for (unsigned j = 0; j < a.width(); ++j)
        {
            c_row[j] = f(a_row[j]);
        }
    }

    return c;
}

// element-wise multiplication
template <typename T>
matrix<T>&
elemul(const matrix<T>& a, const matrix<T>& b, matrix<T>& c)
{
    if (a.width() != b.width() || a.height() != b.height())
    {
        throw std::runtime_error("matrix size mismatch");
    }

    c.resize(a.height(), a.width());

    #pragma omp parallel for
    for (unsigned i = 0; i < a.height(); ++i)
    {
        auto a_row = a[i];
        auto b_row = b[i];
        auto c_row = c[i];

        #pragma omp simd
        for (unsigned j = 0; j < a.width(); ++j)
        {
            c_row[j] = a_row[j] * b_row[j];
        }
    }

    return c;
}

} // namespace neural
} // namespace lm
