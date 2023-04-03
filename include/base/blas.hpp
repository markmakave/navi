#pragma once

#include "base/array.hpp"
#include "base/matrix.hpp"

#include <cmath>
#include <functional>

namespace lm {
namespace blas {

using size_type = int64_t;

// L1
// - amax
// - amin
// - asum
// - axpy
// - copy
// - dot
// - nrm2

template <typename T>
int
amax(
    const array<T>& x
) {
    int max_index = 0;
    
    for (size_type i = 1; i < x.size(); ++i)
        if (x[i] > x[max_index])
            max_index = i;

    return max_index;
}

template <typename T>
int
amin(
    const array<T>& x
) {
    int min_index = 0;
    
    for (size_type i = 1; i < x.size(); ++i)
        if (x[i] < x[min_index])
            min_index = i;

    return min_index;
}

template <typename T>
T
asum(
    const array<T>& x
) {
    T sum = {};

    for (size_type i = 0; i < x.size(); ++i)
        sum += x[i];

    return sum;
}

template <typename T>
void
axpy(
    const T         alpha,
    const array<T>& x, 
          array<T>& y
) {
    if (y.size() != x.size())
        y.resize(x.size());

    for (size_type i = 0; i < x.size(); ++i)
        y[i] = alpha * x[i] + y[i];
}

template <typename T>
void
copy(
    const array<T>& x,
          array<T>& y
) {
    if (y.size() != x.size())
        y.resize(x.size());

    for (size_type i = 0; i < x.size(); ++i)
        y[i] = x[i];
}

template <typename T>
T
dot(
    const array<T>& x,
    const array<T>& y
) {
    assert(x.size() == y.size());

    T res = {};

    for (size_type i = 0; i < x.size(); ++i)
        res += x[i] * y[i];

    return res;
}

template <typename T>
T
nrm2(
    const array<T>& x
) {
    T norm(0);

    for (size_type i = 0; i < x.size(); ++i)
        norm += x[i] * x[i];
    
    return std::sqrt(norm);
}

// L2

template <typename T>
void
mv(
    const matrix<T>& A, 
    const array<T>&  x, 
          array<T>&  y,

    const bool       A_transpose = false
) {
    if (A_transpose)
    {
        assert(x.size() == A.height());
        if (y.size() != A.width())
            y.resize(A.width());

        for (int j = 0; j < A.width(); j++) {
            T sum = 0;
            for (int i = 0; i < A.height(); i++) {
                sum += A[i][j] * x[i];
            }
            y[j] = sum;
        }
    } 
    else
    {
        assert(x.size() == A.width());
        if (y.size() != A.height())
            y.resize(A.height());
        
        for (int i = 0; i < A.height(); i++) {
            T sum = 0;
            for (int j = 0; j < A.width(); j++) {
                sum += A[i][j] * x[j];
            }
            y[i] = sum;
        }
    }
}

template <typename T>
void
ger(
    const array<T>&  x,
    const array<T>&  y,
          T          alpha,
          matrix<T>& A
) {
    assert(x.size() == A.height() && y.size() == A.width());

    for (size_type i = 0; i < x.size(); ++i) {
        for (size_type j = 0; j < y.size(); ++j) {
            A[i][j] += alpha * x[i] * y[j];
        }
    }
}

// L3

template <typename T>
void
mm(
    const matrix<T>& A, 
    const matrix<T>& B, 
          matrix<T>& C
) {
    assert(A.height() == B.height() && A.width() == B.width());
    if (C.height() != A.height() || C.width() != A.width())
        C.resize(A.height(), A.width());

}

// Universal

template <typename T, typename F>
void
binary_op(
    const T*        x,
    const T*        y,
          size_type size,
          F         functor,
          T*        z
) {
    for (size_type i = 0; i < size; ++i)
        z[i] = functor(x[i], y[i]);
}

// Array

template <typename T, typename F>
void
map(
    const array<T>& x,
          F         functor,
          array<T>& y
) {
    if (y.size() != x.size())
        y.resize(x.size());

    for (size_type i = 0; i < x.size(); ++i)
        y[i] = functor(x[i]);
}

template <typename T>
void
add(
    const array<T>& x,
    const array<T>& y,
          array<T>& z
) {
    assert(x.size() == y.size());
    if (z.size() != x.size())
        z.resize(x.size());

    binary_op(x.data(), y.data(), x.size(), [](T a, T b){ return a + b; }, z.data());
}

template <typename T>
void
sub(
    const array<T>& x,
    const array<T>& y,
          array<T>& z
) {
    assert(x.size() == y.size());
    if (z.size() != x.size())
        z.resize(x.size());

    binary_op(x.data(), y.data(), x.size(), [](T a, T b){ return a - b; }, z.data());
}

template <typename T>
void
mul(
    const array<T>& x,
    const array<T>& y,
          array<T>& z
) {
    assert(x.size() == y.size());
    if (z.size() != x.size())
        z.resize(x.size());

    binary_op(x.data(), y.data(), x.size(), [](T a, T b){ return a * b; }, z.data());
}

template <typename T>
void
div(
    const array<T>& x,
    const array<T>& y,
          array<T>& z
) {
    assert(x.size() == y.size());
    if (z.size() != x.size())
        z.resize(x.size());

    binary_op(x.data(), y.data(), x.size(), [](T a, T b){ return a / b; }, z.data());
}

// Matrix

template <typename T, typename F>
void
map(
    const matrix<T>& A,
          F          functor,
          matrix<T>& B
) {
    if (B.height() != A.height() || B.width() != A.width())
        B.resize(A.height(), A.width());

    for (size_type i = 0; i < A.size(); ++i)
        B.data()[i]= functor(A.data()[i]);
}

template <typename T>
void
add(
    const matrix<T>& A,
    const matrix<T>& B,
          matrix<T>& C
) {
    assert(A.height() == B.height() and A.width() == B.width());
    if (C.height() != A.height() or C.width() != A.width())
        C.resize(A.height(), A.width());

    binary_op(A.data(), B.data(), A.size(), [](T a, T b){ return a + b; }, C.data());
}

template <typename T>
void
sub(
    const matrix<T>& A,
    const matrix<T>& B,
          matrix<T>& C
) {
    assert(A.size() == B.size());
    if (C.height() != A.height() or C.width() != A.width())
        C.resize(A.height(), A.width());

    binary_op(A.data(), B.data(), A.size(), [](T a, T b){ return a - b; }, C.data());
}

template <typename T>
void
mul(
    const matrix<T>& A,
    const matrix<T>& B,
          matrix<T>& C
) {
    assert(A.size() == B.size());
    if (C.height() != A.height() or C.width() != A.width())
        C.resize(A.height(), A.width());

    binary_op(A.data(), B.data(), A.size(), [](T a, T b){ return a * b; }, C.data());
}

template <typename T>
void
div(
    const matrix<T>& A,
    const matrix<T>& B,
          matrix<T>& C
) {
    assert(A.size() == B.size());
    if (C.height() != A.height() or C.width() != A.width())
        C.resize(A.height(), A.width());

    binary_op(A.data(), B.data(), A.size(), [](T a, T b){ return a / b; }, C.data());
}

}
}
