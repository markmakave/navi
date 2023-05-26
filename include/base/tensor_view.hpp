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

#include "base/types.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <functional>

#include <cassert>
#include <cmath>

namespace lm {

template <i64 N>
struct shape_t
{
public:

	using size_type = i64;

public:

	shape_t()
	  : _data {}
	{}

	template <typename... Size>
	shape_t(Size... sizes)
	  : _data {static_cast<size_type>(sizes)...}
	{
		static_assert(sizeof...(Size) == N);
	}

	shape_t(const shape_t& s)
	{
		for (size_type n = 0; n < N; ++n)
			_data[n] = s._data[n];
	}

	shape_t&
	operator= (const shape_t& s)
	{
		for (size_type n = 0; n < N; ++n)
			_data[n] = s._data[n];
		return *this;
	}

	bool
	operator== (const shape_t& s) const
	{
		for (size_type n = 0; n < N; ++n)
			if (_data[n] != s[n])
				return false;
		return true;
	}

	bool
	operator!= (const shape_t& s) const
	{
		return !((*this) == s);
	}

	size_type&
	operator[] (size_type dim)
	{
		return _data[dim];
	}

	const size_type&
	operator[] (size_type dim) const
	{
		return _data[dim];
	}

	size_type
	size() const
	{
		size_type s = 1;
		for (size_type n = 0; n < N; ++n)
			s *= _data[n];
		return s;
	}

protected:

	size_type _data[N];
};

template <i64 N, typename T>
class tensor_view
{
public:

	using value_type	  = T;
	using pointer		  = value_type*;
	using const_pointer	  = const value_type*;
	using reference		  = value_type&;
	using const_reference = const value_type&;
	using iterator		  = pointer;
	using const_iterator  = const_pointer;
	using shape_type	  = shape_t<N>;
	using size_type		  = typename shape_type::size_type;

public:

	tensor_view()
	  : _data(nullptr),
		_shape()
	{}

	tensor_view(pointer data, const shape_type& shape)
	  : _data(data),
		_shape(shape)
	{}

	template <typename... Size>
	tensor_view(pointer data, Size... sizes)
	  : _data(nullptr),
		_shape(sizes...)
	{}

	~tensor_view() = default;

	tensor_view&
	operator= (const tensor_view& t) = default;

	tensor_view&
	operator= (tensor_view&& t) = default;

	void
	fill(const_reference fillament)
	{
		for (size_type i = 0; i < size(); ++i)
			_data[i] = fillament;
	}

	const shape_type&
	shape() const
	{
		return _shape;
	}

	size_type
	shape(size_type dimension) const
	{
		return _shape[dimension];
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

	size_type
	size() const
	{
		return _shape.size();
	}

	template <typename... Size>
	void
	reshape(Size... sizes)
	{
		static_assert(sizeof...(Size) == N);

		shape_type new_shape(sizes...);
		reshape(new_shape);
	}

	void
	reshape(const shape_type& shape)
	{
		assert(shape.size() == size());
		_shape = shape;
	}

	reference
	operator() (size_type indices[N])
	{
		size_type offset = 0;
		size_type dim	 = 1;
		for (size_type n = 0; n < N; ++n) {
			offset += dim * indices[n];
			dim *= _shape[n];
		}

		return _data[offset];
	}

	const_reference
	operator() (size_type indices[N]) const
	{
		size_type offset = 0;
		size_type dim	 = 1;
		for (size_type n = 0; n < N; ++n) {
			offset += dim * indices[n];
			dim *= _shape[n];
		}

		return _data[offset];
	}

	template <typename... Index>
	reference
	operator() (Index... index)
	{
		static_assert(sizeof...(Index) == N);

		size_type indices[N] = {index...};

		return (*this)(indices);
	}

	template <typename... Index>
	const_reference
	operator() (Index... index) const
	{
		static_assert(sizeof...(Index) == N);

		size_type indices[N] = {index...};

		return (*this)(indices);
	}

#if __cplusplus >= 202002L

	template <typename... Index>
	reference
	operator[] (Index... index)
	{
		return operator() (index...);
	}

	template <typename... Index>
	const_reference
	operator[] (Index... index) const
	{
		return operator() (index...);
	}

#endif

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

	template <typename... Index>
	reference
	at(Index... index)
	{
		static_assert(sizeof...(Index) == N);

		size_type indices[N] = {index...};

		for (size_type n = 0; n < N; ++n)
			if (indices[n] < 0 or indices[n] >= _shape[n]) {
				static value_type trash;
				trash = {};
				return trash;
			}

		return operator() (indices);
	}

	template <typename... Index>
	const_reference
	at(Index... index) const
	{
		static_assert(sizeof...(Index) == N);

		size_type indices[N] = {index...};

		for (size_type n = 0; n < N; ++n)
			if (indices[n] < 0 or indices[n] >= _shape[n]) {
				static value_type trash;
				trash = {};
				return trash;
			}

		return operator() (indices);
	}

	reference
	front()
	{
		return _data[0];
	}

	const_reference
	front() const
	{
		return _data[0];
	}

	reference
	back()
	{
		return _data[size() - 1];
	}

	const_reference
	back() const
	{
		return _data[size() - 1];
	}

	iterator
	begin()
	{
		return _data;
	}

	const_iterator
	begin() const
	{
		return _data;
	}

	iterator
	end()
	{
		return _data + size();
	}

	const_iterator
	end() const
	{
		return _data + size();
	}

	friend std::ostream&
	operator<< (std::ostream& out, const tensor_view& t)
	{
		out << "Tensor<" << typeid(value_type).name() << "> " << t._shape[0];
		for (size_type n = 1; n < N; ++n)
			out << "×" << t._shape[n];
		out << '\n';

		size_type max_index = 0;
		for (size_type i = 0; i < t.size(); ++i)
			if (t._data[i] > t._data[max_index])
				max_index = i;

		int number_length = std::log10(t._data[max_index]) + 1;

		size_type index[N] = {};

		std::function<void(size_type)> print;
		print = [&](size_type dim) {
			if (dim == 0) {
				for (size_type n = 0; n < t.shape()[0]; ++n) {
					index[0] = n;
					out << std::setw(number_length + 1) << t(index) << ' ';
				}
			} else {
				for (size_type n = 0; n < t.shape()[dim]; ++n) {
					index[dim] = n;

					if (n != 0)
						out << std::setw(N - dim) << ' ';
					out << "[";

					print(dim - 1);

					out << "]";
					if (n != t.shape()[dim] - 1)
						out << '\n';
				}
			}
		};

		out << "[";
		print(N - 1);
		out << "]";

		return out;
	}

	void
	write(std::ofstream& file) const
	{
		size_type order = N;
		file.write((char*)&order, sizeof(order));
		for (size_type n = 0; n < N; ++n)
			file.write((char*)&_shape[n], sizeof(_shape[n]));

		file.write((char*)_data, size() * sizeof(value_type));
	}

	void
	write(const char* filename) const
	{
		std::ofstream file(filename);
		write(file);
	}

	void
	read(std::ifstream& file)
	{
		size_type order;
		file.read((char*)&order, sizeof(order));

		assert(order == N);

		shape_type shape;
		for (size_type n = 0; n < N; ++n)
			file.read((char*)&shape[n], sizeof(shape[n]));

		reshape(shape);

		file.read((char*)_data, size() * sizeof(value_type));
	}

	void
	read(const char* filename)
	{
		std::ifstream file(filename);
		read(file);
	}

protected:

	pointer	   _data;
	shape_type _shape;
};

} // namespace lm
