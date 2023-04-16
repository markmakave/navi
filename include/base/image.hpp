/* 

    Copyright (c) 2023 Mark Mokhov

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

#include "base/matrix.hpp"
#include "base/array.hpp"
#include "base/color.hpp"
#include "base/types.hpp"

#include <png++/png.hpp>
#include <cstring>
#include <sstream>

namespace lm
{

template <typename T>
class image : public matrix<T>
{
public:

    typedef T                   value_type;
    typedef value_type*         pointer;
    typedef const value_type*   const_pointer;
    typedef value_type&         reference;
    typedef const value_type&   const_reference;
    typedef value_type*         iterator;
    typedef const value_type*   const_iterator;
    typedef int64_t             size_type;
    
public:

    image() 
    :   matrix<T>()
    {}

    image(size_type width, size_type height)
    :   matrix<T>(height, width)
    {}

    image(size_type width, size_type height, const_reference value)
    :   matrix<T>(height, width, value)
    {}

    image(const char* filename)
    :   matrix<T>()
    {
        read(filename);
    }

    image(const matrix<T>& m) 
    :   matrix<T>(m)
    {}

    template <typename U>
    image(const matrix<U>& m) 
    :   matrix<T>(m)
    {}

    image(matrix<T>&& m)
    :   matrix<T>(std::move(m))
    {}

    image&
    operator = (const matrix<T>& m)
    {
        matrix<T>::operator=(m);
        return *this;
    }

    image&
    operator = (matrix<T>&& m)
    {
        matrix<T>::operator=(std::move(m));
        return *this;
    }

    void
    resize(size_type width, size_type height)
    {
        matrix<T>::reshape(width, height);
    }

    void
    circle(size_type x, size_type y, size_type radius, const_reference color)
    {
        int f = 1 - radius;
        int ddF_x = 1;
        int ddF_y = -2 * radius;
        int x0 = 0;
        int y0 = radius;

        (*this)[y + radius][x] = color;
        (*this)[y - radius][x] = color;
        (*this)[y][x + radius] = color;
        (*this)[y][x - radius] = color;

        while (x0 < y0)
        {
            if (f >= 0)
            {
                y0--;
                ddF_y += 2;
                f += ddF_y;
            }

            x0++;
            ddF_x += 2;
            f += ddF_x;

            (*this)[y + y0][x + x0] = color;
            (*this)[y - y0][x + x0] = color;
            (*this)[y + y0][x - x0] = color;
            (*this)[y - y0][x - x0] = color;
            (*this)[y + x0][x + y0] = color;
            (*this)[y - x0][x + y0] = color;
            (*this)[y + x0][x - y0] = color;
            (*this)[y - x0][x - y0] = color;
        }
    }

    void
    line(int x0, int y0, int x1, int y1, const_reference color)
    {
        bool steep = std::abs(y1 - y0) > std::abs(x1 - x0);

        if (steep)
        {
            std::swap(x0, y0);
            std::swap(x1, y1);
        }

        if (x0 > x1)
        {
            std::swap(x0, x1);
            std::swap(y0, y1);
        }

        int dx = x1 - x0;
        int dy = std::abs(y1 - y0);
        int error = dx / 2;
        int ystep = (y0 < y1) ? 1 : -1;
        int y = y0;

        for (int x = x0; x < x1; ++x)
        {
            if (steep)
            {
                (*this)[x][y] = color;
            }
            else
            {
                (*this)[y][x] = color;
            }

            error -= dy;
            if (error < 0)
            {
                y += ystep;
                error += dx;
            }
        }
    }

    void
    read(const char* filename)
    {
        std::string ext = filename;
        ext = ext.substr(ext.find_last_of(".") + 1);

        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open())
            throw std::runtime_error("could not open file");
        size_type size = file.tellg();
        file.seekg(0, std::ios::beg);

        lm::array<uint8_t> buffer(size);
        file.read((char*)buffer.data(), size);

        if (ext == "png")
            decode<format::png>(buffer);
        else if (ext == "qoi")
            decode<format::qoi>(buffer);
    }
    
    void
    write(const char* filename) const
    {
        std::string ext = filename;
        ext = ext.substr(ext.find_last_of(".") + 1);

        array<byte> buffer;

        if (ext == "png")
            buffer = encode<format::png>();
        else if (ext == "qoi")
            buffer = encode<format::qoi>();

        std::ofstream file(filename, std::ios::binary);
        file.write((char*)buffer.data(), buffer.size());
    }

    enum class format
    {
        png,
        qoi
    };

    template <format fmt>
    array<byte>
    encode() const;

    template <enum format fmt>
    void
    decode(const array<byte>& buffer);
};

} // namespace lm
