#include "base/image.hpp"
#include "util/log.hpp"

// PNG GRAY

template <>
template <>
lm::array<lm::byte>
lm::image<lm::gray>::encode<lm::image<lm::gray>::format::png>() const
{
    array<byte> buffer;

    png::image<png::gray_pixel> img(_width, _height);
    for (unsigned y = 0; y < _height; ++y)
    {
        auto this_row = row(y);
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < _width; ++x)
            img_row[x] = png::gray_pixel(this_row[x]);
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
void
lm::image<lm::gray>::decode<lm::image<lm::gray>::format::png>(const lm::array<lm::byte>& buffer)
{
    std::stringstream ss;
    ss.write((char*)buffer.data(), buffer.size());

    png::image<png::gray_pixel> img(ss);

    resize(img.get_width(), img.get_height());

    for (unsigned y = 0; y < img.get_height(); ++y)
    {
        auto this_row = row(y);
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < img.get_width(); ++x)
            this_row[x] = reinterpret_cast<lm::byte&>(img_row[x]);
    }
}

// PNG RGB

template <>
template <>
lm::array<lm::byte>
lm::image<lm::rgb>::encode<lm::image<lm::rgb>::format::png>() const
{
    array<byte> buffer;

    png::image<png::rgb_pixel> img(_width, _height);
    for (unsigned y = 0; y < _height; ++y)
    {
        auto this_row = row(y);
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < _width; ++x)
            img_row[x] = reinterpret_cast<const png::rgb_pixel&>(this_row[x]);
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
void
lm::image<lm::rgb>::decode<lm::image<lm::rgb>::format::png>(const lm::array<lm::byte>& buffer)
{
    std::stringstream ss;
    ss.write((char*)buffer.data(), buffer.size());

    png::image<png::rgb_pixel> img(ss);

    resize(img.get_width(), img.get_height());

    for (unsigned y = 0; y < img.get_height(); ++y)
    {
        auto this_row = row(y);
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < img.get_width(); ++x)
            this_row[x] = reinterpret_cast<rgb&>(img_row[x]);
    }
}

// PNG RGBA

template <>
template <>
lm::array<lm::byte>
lm::image<lm::rgba>::encode<lm::image<lm::rgba>::format::png>() const
{
    array<byte> buffer;

    png::image<png::rgba_pixel> img(_width, _height);
    for (unsigned y = 0; y < _height; ++y)
    {
        auto this_row = row(y);
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < _width; ++x)
            img_row[x] = reinterpret_cast<const png::rgba_pixel&>(this_row[x]);
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
void
lm::image<lm::rgba>::decode<lm::image<lm::rgba>::format::png>(const lm::array<lm::byte>& buffer)
{
    std::stringstream ss;
    ss.write((char*)buffer.data(), buffer.size());

    png::image<png::rgba_pixel> img(ss);

    resize(img.get_width(), img.get_height());

    for (unsigned y = 0; y < img.get_height(); ++y)
    {
        auto this_row = row(y);
        auto& img_row = img.get_row(y);

        for (unsigned x = 0; x < img.get_width(); ++x)
            this_row[x] = reinterpret_cast<rgba&>(img_row[x]);
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
void
lm::image<lm::gray>::decode<lm::image<lm::gray>::format::qoi>(const lm::array<lm::byte>& buffer)
{
    qoi_desc desc = {};
    void* ptr = qoi_decode(buffer.data(), buffer.size(), &desc, 0);

    resize(desc.width, desc.height);

    if (desc.channels == sizeof(rgb))
    {
        rgb* data = reinterpret_cast<rgb*>(ptr);

        size_type size = this->size();
        for (size_type i = 0; i < size; ++i)
            _data[i] = data[i];
    }
    else // desc.channels == sizeof(rgba)
    {
        rgb* data = reinterpret_cast<rgba*>(ptr);

        size_type size = this->size();
        for (size_type i = 0; i < size; ++i)
            _data[i] = data[i];
    }
}

// QOI RGB

template <>
template <>
lm::array<lm::byte>
lm::image<lm::rgb>::encode<lm::image<lm::rgb>::format::qoi>() const
{
    qoi_desc desc;
    desc.width = _width;
    desc.height = _height;
    desc.channels = 3;
    desc.colorspace = QOI_LINEAR;

    int buffer_length;
    byte* buffer_data = reinterpret_cast<byte*>(qoi_encode(_data, &desc, &buffer_length));

    return array<byte>(buffer_data, buffer_length);
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
        _width = desc.width;
        _height = desc.height;
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
    desc.width = _width;
    desc.height = _height;
    desc.channels = 4;
    desc.colorspace = QOI_LINEAR;

    int buffer_length;
    byte* buffer_data = reinterpret_cast<byte*>(qoi_encode(_data, &desc, &buffer_length));

    return array<byte>(buffer_data, buffer_length);
}

template <>
template <>
void
lm::image<lm::rgba>::decode<lm::image<lm::rgba>::format::qoi>(const lm::array<lm::byte>& buffer)
{
    
}

//
