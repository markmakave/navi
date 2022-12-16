#pragma once

#include "base/matrix.hpp"
#include "base/array.hpp"
#include "graphics/color.hpp"

#include <png.h>
#include <jpeglib.h>

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

    image(const matrix<T>& m) 
    :   matrix<T>(m)
    {}

    image(image&& m)
    :   matrix<T>(std::move(m))
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
    operator=(const matrix<T>& m)
    {
        matrix<T>::operator=(m);
        return *this;
    }

    image&
    operator=(image&& m)
    {
        matrix<T>::operator=(std::move(m));
        return *this;
    }

    image&
    operator=(matrix<T>&& m)
    {
        matrix<T>::operator=(std::move(m));
        return *this;
    }

    void
    read(const char* filename)
    {
        FILE* fp = fopen(filename, "rb");
        if (!fp) return;

        png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (!png_ptr) return;

        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr) return;

        if (setjmp(png_jmpbuf(png_ptr))) return;

        png_init_io(png_ptr, fp);

        png_read_info(png_ptr, info_ptr);

        png_uint_32 width, height;
        int bit_depth, color_type, interlace_type, compression_type, filter_method;

        png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
                     &interlace_type, &compression_type, &filter_method);

        if (color_type != PNG_COLOR_TYPE_RGB) return;

        this->resize(height, width);

        png_bytep row = (png_bytep) malloc(3 * this->width() * sizeof(png_byte));

        for (size_type i = 0; i < this->height(); ++i)
        {
            png_read_row(png_ptr, row, NULL);
            for (size_type j = 0; j < this->width(); ++j)
            {
                this->at(i, j).r = row[j*3 + 0];
                this->at(i, j).g = row[j*3 + 1];
                this->at(i, j).b = row[j*3 + 2];
            }
        }

        free(row);
        fclose(fp);
    }

    void
    write(const char* filename) const
    {
        std::string ext = filename;
        ext = ext.substr(ext.find_last_of(".") + 1);

        lm::array<uint8_t> buffer;

        if (ext == "png")
        {
            buffer = encode_png();
        }
        else if (ext == "jpg" || ext == "jpeg")
        {
            buffer = encode_jpeg(100);
        }

        FILE* fp = fopen(filename, "wb");
        if (!fp) return;

        fwrite(buffer.data(), 1, buffer.size(), fp);

        fclose(fp);
    }

    lm::array<uint8_t>
    encode_png() const;

    lm::array<uint8_t>
    encode_jpeg(int quality) const;

};

template<>
inline
lm::array<uint8_t>
lm::image<lm::gray>::encode_png() const
{
    lm::array<uint8_t> buffer;

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return buffer;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) return buffer;

    if (setjmp(png_jmpbuf(png_ptr))) return buffer;

    png_set_IHDR(png_ptr, info_ptr, this->width(), this->height(),
                 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    for (size_type i = 0; i < this->height(); ++i)
    {
        png_write_row(png_ptr, (png_bytep) this->row(i));
    }

    png_write_end(png_ptr, NULL);

    png_destroy_write_struct(&png_ptr, &info_ptr);

    return buffer;
}

template<>
inline
lm::array<uint8_t>
lm::image<lm::rgb>::encode_png() const
{
    lm::array<uint8_t> buffer;

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return buffer;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) return buffer;

    if (setjmp(png_jmpbuf(png_ptr))) return buffer;

    png_set_IHDR(png_ptr, info_ptr, this->width(), this->height(),
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);


    for (size_type i = 0; i < this->height(); ++i)
    {
        png_write_row(png_ptr, (png_bytep) this->row(i));
    }

    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);

    return buffer;
}

template<>
inline
lm::array<uint8_t>
lm::image<lm::rgba>::encode_png() const
{
    lm::array<uint8_t> buffer;

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return buffer;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) return buffer;

    if (setjmp(png_jmpbuf(png_ptr))) return buffer;

    png_set_IHDR(png_ptr, info_ptr, this->width(), this->height(),
                 8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);


    for (size_type i = 0; i < this->height(); ++i)
    {   
        png_write_row(png_ptr, (png_bytep) this->row(i));
    }   

    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);

    return buffer;
}

template<>
inline
lm::array<uint8_t>
lm::image<lm::rgb>::encode_jpeg(int quality) const
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    uint8_t* data = NULL;
    unsigned long size = 0;
    jpeg_mem_dest(&cinfo, &data, &size);

    cinfo.image_width = this->width();
    cinfo.image_height = this->height();
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = (JSAMPROW) &this->at(cinfo.next_scanline, 0);
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    lm::array<uint8_t> buffer(data, size);

    return buffer;
}

template<>
inline
lm::array<uint8_t>
lm::image<lm::rgba>::encode_jpeg(int quality) const
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    uint8_t* data = NULL;
    unsigned long size = 0;
    jpeg_mem_dest(&cinfo, &data, &size);

    cinfo.image_width = width();
    cinfo.image_height = height();
    cinfo.input_components = 4;
    cinfo.in_color_space = JCS_EXT_RGBA;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, FALSE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = (JSAMPROW) row(cinfo.next_scanline);
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    lm::array<uint8_t> buffer(data, size);

    return buffer;
}

} // namespace lm
