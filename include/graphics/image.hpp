#pragma once

#include "base/matrix.hpp"

#include <png.h>

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
        FILE* fp = fopen(filename, "wb");
        if (!fp) return;

        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (!png_ptr) return;

        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr) return;

        if (setjmp(png_jmpbuf(png_ptr))) return;

        png_init_io(png_ptr, fp);

        png_set_IHDR(png_ptr, info_ptr, this->width(), this->height(),
                     8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

        png_write_info(png_ptr, info_ptr);

        png_bytep row = (png_bytep) malloc(3 * this->width() * sizeof(png_byte));

        for (size_type i = 0; i < this->height(); ++i)
        {
            for (size_type j = 0; j < this->width(); ++j)
            {
                row[j*3 + 0] = this->at(i, j).r;
                row[j*3 + 1] = this->at(i, j).g;
                row[j*3 + 2] = this->at(i, j).b;
            }
            png_write_row(png_ptr, row);
        }

        png_write_end(png_ptr, NULL);

        free(row);
        fclose(fp);

        return;
    }

};

} // namespace lm
