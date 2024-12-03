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

#include <cuda.h>

#include "base/shape.hpp"
#include "utility/types.hpp"

namespace lumina::cuda {

template <u64 D, typename T>
requires (D <= 3)
class array
{
public:

	array()
	:	_handle(0)
	{}

	template <typename... Dims>
	requires(sizeof...(Dims) == D)
	array(Dims... dims)
	{
		u32 dim[3] = { static_cast<u32>(dims)... };

		CUDA_ARRAY3D_DESCRIPTOR array_desc = {
			.Width = dim[0],
			.Height = dim[1],
			.Depth = dim[2],
			.Format = _get_type_enum(),
			.NumChannels = 1
		};

		cuArray3DCreate(&_handle, &array_desc);
	}

	~array()
	{
		cuArrayDestroy(_handle);
	}


	u64 size()   const					 { return _shape.volume(); }
	u32 width()  const					 { return _shape[0]; }
	u32 height() const requires (D >= 2) { return _shape[1]; }
	u32 depth()  const requires (D >= 3) { return _shape[2]; }

protected:

	static constexpr CUarray_format _get_type_enum()
    {
        // Integers
        if constexpr (std::is_same_v<T, u8>)  return CU_AD_FORMAT_UNSIGNED_INT8;
        if constexpr (std::is_same_v<T, u16>) return CU_AD_FORMAT_UNSIGNED_INT16;
        if constexpr (std::is_same_v<T, u32>) return CU_AD_FORMAT_UNSIGNED_INT32;
        if constexpr (std::is_same_v<T, i8>)  return CU_AD_FORMAT_SIGNED_INT8;
        if constexpr (std::is_same_v<T, i16>) return CU_AD_FORMAT_SIGNED_INT16;
        if constexpr (std::is_same_v<T, i32>) return CU_AD_FORMAT_SIGNED_INT32;

        // Floating point
        if constexpr (std::is_same_v<T, f32>) return CU_AD_FORMAT_FLOAT;

        throw std::invalid_argument("Unsupported array value type");
    }

protected:

	CUarray _handle;
	shape<D> _shape;
};

}
