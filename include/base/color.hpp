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

#include <cstdint>
#include <random>
#include <chrono>

#include "base/types.hpp"

namespace lumina {

template <int lower, int higher>
static int
clamp(int value)
{
	if (value > higher)
		return higher;
	if (value < lower)
		return lower;
	return value;
}

inline u8
random_channel()
{
	static std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count());
	static std::uniform_int_distribution<u8> distribution(0, 255);
	return distribution(engine);
}

struct yuv;
struct rgb;
struct rgba;
using gray = u8;

struct yuv
{
	u8 y, u, v;

	yuv(u8 y = 0, u8 u = 0, u8 v = 0)
	  : y(y),
		u(u),
		v(v)
	{}
};

struct rgb
{
	u8 r, g, b;

	rgb()
	  : r(0),
		g(0),
		b(0)
	{}

	rgb(u8 gray)
	  : r(gray),
		g(gray),
		b(gray)
	{}

	rgb(u8 r, u8 g, u8 b)
	  : r(r),
		g(g),
		b(b)
	{}

	rgb(const yuv& color)
	{
		b = clamp<0, 255>((int)color.y + (1.732446 * ((int)color.u - 128)));
		g = clamp<0, 255>((int)color.y - (0.698001 * ((int)color.v - 128)) -
						  (0.337633 * ((int)color.u - 128)));
		r = clamp<0, 255>((int)color.y + (1.370705 * ((int)color.v - 128)));
	}

	operator u8 () const
	{
		return clamp<0, 255>(0.2161 * r + 0.7152 * g + 0.0722 * b);
	}

	static rgb
	random()
	{
		rgb color = {random_channel(), random_channel(), random_channel()};
		return color;
	}
};

struct rgba : rgb
{
	u8 a;

	rgba()
	  : rgb(),
		a(0)
	{}

	rgba(u8 gray)
	  : rgb(gray),
		a(255)
	{}

	rgba(u8 r, u8 g, u8 b, u8 a = 255)
	  : rgb(r, g, b),
		a(a)
	{}

	rgba(const rgb& color)
	  : rgb(color),
		a(255)
	{}

	rgba(const yuv& color)
	  : rgb(color),
		a(255)
	{}

	static rgba
	random()
	{
		rgba color = {random_channel(), random_channel(), random_channel(), random_channel()};
		return color;
	}
};

} // namespace lumina
