/*

	Copyright (c) 2023 Mokhov Mark

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

#include <cmath>

#include "base/types.hpp"

namespace lumina {

template <i64 N, typename T>
struct vec
{
public:

	using value_type	  = T;
	using pointer		  = value_type*;
	using const_pointer	  = const value_type*;
	using reference		  = value_type&;
	using const_reference = const value_type&;
	using iterator		  = pointer;
	using const_iterator  = const_pointer;
	using size_type		  = decltype(N);

public:

	vec()
	  : _data {}
	{}

	template <typename... Args>
	vec(Args... args)
	  : _data {args...}
	{
		static_assert(sizeof...(Args) == N);
	}

	reference
	operator[] (size_type index)
	{
		return _data[index];
	}

	const_reference
	operator[] (size_type index) const
	{
		return _data[index];
	}

	pointer
	data()
	{
		return _data;
	}

	const_pointer
	data() const
	{
		return _data;
	}

	value_type
	length() const
	{
		value_type length = 0;
		for (size_type n = 0; n < N; ++n)
			length += _data[n] * _data[n];

		return std::sqrt(length);
	}

protected:

	value_type _data[N];
};

using vec3 = vec<3, float>;

} // namespace lumina
