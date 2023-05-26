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

#include "base/tensor_view.hpp"

namespace lm {

template <i64 N, typename T, typename _alloc>
class tensor : public tensor_view<N, T>
{
	using base = tensor_view<N, T>;

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

	tensor()
	  : base()
	{}

	template <typename... Size>
	tensor(Size... sizes)
	  : base()
	{
		reshape(sizes...);
	}

	tensor(const shape_type& shape)
	  : base()
	{
		reshape(shape);
	}

	tensor(const tensor& t)
	  : base()
	{
		reshape(t.shape());
		_alloc::copy(t._data, this->_data, this->size());
	}

	tensor(tensor&& t)
	  : base()
	{
		t._data	 = nullptr;
		t._shape = shape_type();
	}

	template <typename U, typename alloc>
	tensor(const tensor<N, U, alloc>& t)
	  : base()
	{
		reshape(t.shape());
		for (size_type i = 0; i < t.size(); ++i)
			_alloc::access(this->_data + i) = static_cast<value_type>(t[i]);
	}

	~tensor()
	{
		_deallocate();
	}

	tensor&
	operator= (const tensor& t)
	{
		if (&t != this) {
			reshape(t._shape);
			_alloc::copy(t._data, this->_data, t.size() * sizeof(value_type));
		}

		return *this;
	}

	tensor&
	operator= (tensor&& t)
	{
		if (&t != this) {
			this->_data	 = t._data;
			this->_shape = t._shape;

			t._data	 = nullptr;
			t._shape = shape_type();
		}

		return *this;
	}

	template <typename U, typename alloc>
	tensor&
	operator= (const tensor<N, U, alloc>& t)
	{
		reshape(t.shape());
		for (size_type i = 0; i < t.size(); ++i)
			_alloc::access(this->_data + i) = static_cast<value_type>(t.data()[i]);

		return *this;
	}

	void
	reshape(const shape_type& shape)
	{
		if (this->_shape == shape)
			return;

		_deallocate();
		this->_shape = shape;
		_allocate();
	}

	template <typename... Size>
	void
	reshape(Size... sizes)
	{
		static_assert(sizeof...(Size) == N);

		shape_type new_shape(sizes...);
		reshape(new_shape);
	}

protected:

	void
	_allocate()
	{
		this->_data = _alloc::allocate(this->size());
	}

	void
	_deallocate()
	{
		_alloc::deallocate(this->_data);
	}
};

} // namespace lm
