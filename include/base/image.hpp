#pragma once

#include "base/matrix.hpp"
#include "base/array.hpp"
#include "base/color.hpp"
#include "base/types.hpp"

#include <png++/png.hpp>
#include <jpeglib.h>
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
    typedef unsigned            size_type;
    typedef int                 difference_type;
    
public:

    image() 
    :   matrix<T>()
    {}

    image(size_type width, size_type height)
    :   matrix<T>(height, width)
    {}

    image(size_type width, size_type height, const_reference fillament)
    :   matrix<T>(height, width, fillament)
    {}

    image(const char* filename)
    :   matrix<T>()
    {
        read(filename);
    }

    image(const image& m) 
    :   matrix<T>(m)
    {}

    image(image&& m)
    :   matrix<T>(std::move(m))
    {}

    image(const matrix<T>& m) 
    :   matrix<T>(m)
    {}

    image(matrix<T>&& m)
    :   matrix<T>(std::move(m))
    {}

    image&
    operator=(const image& m)
    {
        matrix<T>::operator=(m);
        return *this;
    }

    image&
    operator = (image&& m)
    {
        matrix<T>::operator=(std::move(m));
        return *this;
    }

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
        matrix<T>::resize(height, width);
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
        {
            decode<format::png>(buffer);
        }
        else if (ext == "jpg" || ext == "jpeg")
        {
            decode<format::jpeg>(buffer);
        }
    }

    void
    write(const char* filename, int quality = 100) const
    {
        std::string ext = filename;
        ext = ext.substr(ext.find_last_of(".") + 1);

        lm::array<byte> buffer;

        if (ext == "png")
        {
            buffer = encode<format::png>();
        } 
        else if (ext == "jpg" || ext == "jpeg")
        {
            buffer = encode<format::jpeg>(quality);
        }

        std::ofstream file(filename, std::ios::binary);
        file.write((char*)buffer.data(), buffer.size());
    }

    enum class format
    {
        png,
        jpeg
    };

    template <enum format fmt>
    array<byte>
    encode() const;

    template <enum format fmt>
    array<byte>
    encode(int quality) const;

    template <enum format fmt>
    void
    decode(const array<byte>& buffer);
};

template <>
template <>
inline
array<byte>
image<gray>::encode<image<gray>::format::png>() const
{
    array<byte> buffer;

    png::image<png::gray_pixel> img(this->width(), this->height());
    for (unsigned y = 0; y < this->height(); ++y)
    {
        auto this_row = this->row(y);
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < this->width(); ++x)
        {
            img_row[x] = png::gray_pixel(this_row[x]);
        }
    }

    std::stringstream ss;
    img.write_stream(ss);
    ss.seekg(0, std::ios::end);
    size_type size = ss.tellg();
    ss.seekg(0, std::ios::beg);

    buffer.resize(size);
    ss.read((char*)buffer.data(), size);

    return buffer;
}

template <>
template <>
inline
array<byte>
image<rgb>::encode<image<rgb>::format::png>() const
{
    

    return lm::array<lm::byte>();
}

template <>
template <>
inline
array<byte>
image<rgba>::encode<image<rgba>::format::png>() const
{
    

    return lm::array<lm::byte>();
}

template <>
template <>
inline
array<byte>
image<gray>::encode<image<gray>::format::jpeg>(int quality) const
{
    

    return lm::array<lm::byte>();
}

template <>
template <>
inline
array<byte>
image<rgb>::encode<image<rgb>::format::jpeg>(int quality) const
{
    

    return lm::array<lm::byte>();
}

template <>
template <>
inline
array<byte>
image<rgba>::encode<image<rgba>::format::jpeg>(int quality) const
{
    

    return lm::array<lm::byte>();
}

template <>
template <>
void
image<gray>::decode<image<gray>::format::png>(const array<byte>& buffer) {


}

template <>
template <>
void
image<rgb>::decode<image<rgb>::format::png>(const array<byte>& buffer) {
    
    std::stringstream ss;
    ss.write((char*)buffer.data(), buffer.size());

    png::image<png::rgb_pixel> img(ss);

    resize(img.get_width(), img.get_height());

    for (unsigned y = 0; y < img.get_height(); ++y)
    {
        auto this_row = row(y);
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < img.get_width(); ++x)
        {
            this_row[x] = reinterpret_cast<rgb&>(img_row[x]);
        }
    }
}

template <>
template <>
void
image<rgba>::decode<image<rgba>::format::png>(const array<byte>& buffer) {


}

template <>
template <>
void
image<gray>::decode<image<gray>::format::jpeg>(const array<byte>& buffer) {


}

template <>
template <>
void
image<rgb>::decode<image<rgb>::format::jpeg>(const array<byte>& buffer) {


}

template <>
template <>
void
image<rgba>::decode<image<rgba>::format::jpeg>(const array<byte>& buffer) {


}

} // namespace lm
