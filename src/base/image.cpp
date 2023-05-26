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

#include "base/image.hpp"
#include "util/log.hpp"

// PNG GRAY

template <>
template <>
lm::array<lm::byte>
lm::image<lm::gray>::encode<lm::image<lm::gray>::format::png>() const
{
    array<byte> buffer;

    png::image<png::gray_pixel> img(_shape[0], _shape[1]);
    for (unsigned y = 0; y < _shape[1]; ++y)
    {
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < _shape[0]; ++x)
            img_row[x] = png::gray_pixel((*this)(x, y));
    }

    std::stringstream ss;
    img.write_stream(ss);
    ss.seekg(0, std::ios::end);
    size_type size = ss.tellg();
    ss.seekg(0, std::ios::beg);

    buffer.reshape(size);
    ss.read((char*)buffer.data(), size);

    return buffer;
}

template <>
template <>
lm::array<lm::byte>
lm::image<bool>::encode<lm::image<bool>::format::png>() const
{
    array<byte> buffer;

    png::image<png::gray_pixel_1> img(_shape[0], _shape[1]);
    for (unsigned y = 0; y < _shape[1]; ++y)
    {
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < _shape[0]; ++x)
            img_row[x] = png::gray_pixel_1((*this)(x, y));
    }

    std::stringstream ss;
    img.write_stream(ss);
    ss.seekg(0, std::ios::end);
    size_type size = ss.tellg();
    ss.seekg(0, std::ios::beg);

    buffer.reshape(size);
    ss.read((char*)buffer.data(), size);

    return buffer;
}

template <>
template <>
void
lm::image<lm::gray>::decode<lm::image<lm::gray>::format::png>(const lm::array<lm::byte>& buffer)
{
    std::stringstream ss;
    ss.write((char*)buffer.data(), buffer.size());

    png::image<png::gray_pixel> img(ss);

    resize(img.get_width(), img.get_height());

    for (unsigned y = 0; y < img.get_height(); ++y)
    {
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < img.get_width(); ++x)
            (*this)(x, y) = reinterpret_cast<lm::byte&>(img_row[x]);
    }
}

template <>
template <>
void
lm::image<bool>::decode<lm::image<bool>::format::png>([[maybe_unused]] const lm::array<lm::byte>& buffer)
{}

// PNG RGB

template <>
template <>
lm::array<lm::byte>
lm::image<lm::rgb>::encode<lm::image<lm::rgb>::format::png>() const
{
    array<byte> buffer;

    png::image<png::rgb_pixel> img(_shape[0], _shape[1]);
    for (unsigned y = 0; y < _shape[1]; ++y)
    {
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < _shape[0]; ++x)
            img_row[x] = reinterpret_cast<const png::rgb_pixel&>((*this)(x, y));
    }

    std::stringstream ss;
    img.write_stream(ss);
    ss.seekg(0, std::ios::end);
    size_type size = ss.tellg();
    ss.seekg(0, std::ios::beg);

    buffer.reshape(size);
    ss.read((char*)buffer.data(), size);

    return buffer;
}

template <>
template <>
void
lm::image<lm::rgb>::decode<lm::image<lm::rgb>::format::png>(const lm::array<lm::byte>& buffer)
{
    std::stringstream ss;
    ss.write((char*)buffer.data(), buffer.size());

    png::image<png::rgb_pixel> img(ss);

    resize(img.get_width(), img.get_height());

    for (unsigned y = 0; y < img.get_height(); ++y)
    {
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < img.get_width(); ++x)
            (*this)(x, y) = reinterpret_cast<rgb&>(img_row[x]);
    }
}

// PNG RGBA

template <>
template <>
lm::array<lm::byte>
lm::image<lm::rgba>::encode<lm::image<lm::rgba>::format::png>() const
{
    array<byte> buffer;

    png::image<png::rgba_pixel> img(_shape[0], _shape[1]);
    for (unsigned y = 0; y < _shape[1]; ++y)
    {
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < _shape[0]; ++x)
            img_row[x] = reinterpret_cast<const png::rgba_pixel&>((*this)(x, y));
    }

    std::stringstream ss;
    img.write_stream(ss);
    ss.seekg(0, std::ios::end);
    size_type size = ss.tellg();
    ss.seekg(0, std::ios::beg);

    buffer.reshape(size);
    ss.read((char*)buffer.data(), size);

    return buffer;
}

template <>
template <>
void
lm::image<lm::rgba>::decode<lm::image<lm::rgba>::format::png>(const lm::array<lm::byte>& buffer)
{
    std::stringstream ss;
    ss.write((char*)buffer.data(), buffer.size());

    png::image<png::rgba_pixel> img(ss);

    resize(img.get_width(), img.get_height());

    for (unsigned y = 0; y < img.get_height(); ++y)
    {
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < img.get_width(); ++x)
            (*this)(x, y) = reinterpret_cast<rgba&>(img_row[x]);
    }
}

// QOI GRAY

#include "vendor/qoi/qoi.hpp"

// NOT AVAILABLE AT THE MOMENT
// waiting for custom qoi implementation

template <>
template <>
lm::array<lm::byte>
lm::image<lm::gray>::encode<lm::image<lm::gray>::format::qoi>() const
{
    lm::log::error("image<gray>::encode<qoi> not implemented");
    return {};
}


template <>
template <>
lm::array<lm::byte>
lm::image<bool>::encode<lm::image<bool>::format::qoi>() const
{
    lm::log::error("image<gray>::encode<qoi> not implemented");
    return {};
}

template <>
template <>
void
lm::image<lm::gray>::decode<lm::image<lm::gray>::format::qoi>([[maybe_unused]] const lm::array<lm::byte>& buffer)
{
    lm::log::error("image<gray>::decode<qoi> not implemented");
}

// QOI RGB

template <>
template <>
lm::array<lm::byte>
lm::image<lm::rgb>::encode<lm::image<lm::rgb>::format::qoi>() const
{
    qoi_desc desc;
    desc.width = _shape[0];
    desc.height = _shape[1];
    desc.channels = 3;
    desc.colorspace = QOI_LINEAR;

    int buffer_length;
    byte* buffer_data = reinterpret_cast<byte*>(qoi_encode(_data, &desc, &buffer_length));

    array<byte> buffer(buffer_length);
    for (int i = 0; i < buffer_length; ++i)
        buffer(i) = buffer_data[i];

    return buffer;
}

template <>
template <>
void
lm::image<lm::rgb>::decode<lm::image<lm::rgb>::format::qoi>(const lm::array<lm::byte>& buffer)
{
    qoi_desc desc = {};
    void* ptr = qoi_decode(buffer.data(), buffer.size(), &desc, 0);

    if (desc.channels == sizeof(rgb))
    {
        rgb* data = reinterpret_cast<rgb*>(ptr);

        _deallocate();
        _data = data;
        _shape[0] = desc.width;
        _shape[1] = desc.height;
    }
    else
    {
        rgb* data = reinterpret_cast<rgba*>(ptr);

        resize(desc.width, desc.height);
        size_type size = this->size();
        for (size_type i = 0; i < size; ++i)
            _data[i] = data[i];
    }
}

// QOI RGBA

template <>
template <>
lm::array<lm::byte>
lm::image<lm::rgba>::encode<lm::image<lm::rgba>::format::qoi>() const
{
    qoi_desc desc;
    desc.width = _shape[0];
    desc.height = _shape[1];
    desc.channels = 4;
    desc.colorspace = QOI_LINEAR;

    int buffer_length;
    byte* buffer_data = reinterpret_cast<byte*>(qoi_encode(_data, &desc, &buffer_length));

    array<byte> buffer(buffer_length);
    for (int i = 0; i < buffer_length; ++i)
        buffer(i) = buffer_data[i];

    return buffer;
}

template <>
template <>
void
lm::image<lm::rgba>::decode<lm::image<lm::rgba>::format::qoi>([[maybe_unused]] const lm::array<lm::byte>& buffer)
{
    
}

//
