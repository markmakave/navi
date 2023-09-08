/*

    Copyright (c) 2023 Mokhov Mark

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

*/

#pragma once

#include "base/types.hpp"
#include "base/memory.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <functional>

#include <cassert>
#include <cmath>

namespace lumina {
namespace compile_time {

template <i64... Dims>
class shape {};

template <typename T, typename Shape>
class tensor;

template <typename T, i64 N, i64... Rest>
class tensor<T, shape<N, Rest...>>
{
    using base = tensor<T, shape<Rest...>>;

public:

    using value_type      = base::value_type;
    using pointer         = base::pointer;
    using const_pointer   = base::const_pointer;
    using reference       = base::reference;
    using const_reference = base::const_reference;
    using iterator        = base::iterator;
    using const_iterator  = base::const_iterator;
    using size_type       = base::size_type;

public:

    template <typename... Size>
    reference
    operator ()(size_type size, Size... sizes)
    {
        static_assert(sizeof...(Size) == N);
        return _data[size](sizes...);
    }    

// protected:

    base _data[N];
};

template <typename T>
class tensor<T, shape<>>
{
public:

    using value_type      = T;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using iterator        = pointer;
    using const_iterator  = const_pointer;
    using size_type       = i64;

public:

// protected:

    value_type _data;  
};

}
} // namespace lumina
