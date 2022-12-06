#pragma once

#include <functional>
#include <stdexcept>
#include <assert.h>
#include <fstream>

namespace lm {

template <typename T>
struct matrix
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

protected:

    pointer _data;
    size_type _height, _width;

public:

    /*
        @brief Default constructor
    */
    matrix()
    :   _data(nullptr),
        _height(0),
        _width(0)
    {}

    /*
        @brief Constructor with height and width
        @param height Height of the matrix
        @param width Width of the matrix
    */
    matrix(size_type height, size_type width) 
    :   _height(height),
        _width(width)
    {
        _allocate(size());
    }

    /*
        @brief Copy constructor
        @param m The matrix to copy
    */
    matrix(const matrix<T>& m) 
    :   _height(m._height),
        _width(m._width)
    {
        size_type size = this->size();
        _allocate(size);

        #pragma omp parallel for
        for (size_type i = 0; i < size; ++i)
            _data[i] = m._data[i];
    }

    /*
        @brief Move constructor
        @param m matrix to move from
    */
    matrix(matrix&& m) 
    :   _data(m._data),
        _height(m._height),
        _width(m._width)
    {
        m._data = nullptr;
        m._height = 0;
        m._width = 0;
    }

    /*
        @brief Constructor from .mtx file
        @param filename The name of the file
    */
    matrix(const std::string& filename)
    {
        read_mtx(filename);
    }
    
    /*
        @brief Constructs a matrix from matrix of different type
        @param m matrix of different type
    */
    template <typename _T>
    matrix(const matrix<_T>& m, const std::function<T(_T)>& converter = [](const _T& a) { return static_cast<T>(a); })
    :   _height(m.height()),
        _width(m.width())
    {
        size_type size = this->size();
        _allocate(size);

        #pragma omp parallel for
        for (size_type i = 0; i < size; ++i)
            _data[i] = converter(m._data[i]);
    }
    
    /*
        @brief Destructor for matrix
    */
    ~matrix()
    {
        _deallocate();
    }

    /*
        @brief Assignment operator
        @param m matrix to copy
    */
    matrix& 
    operator = (const matrix& m)
    {
        if (&m != this)
        {
            resize(m._width, m._height);
            size_type size = this->size();

            #pragma omp parallel for
            for (size_type i = 0; i < size; ++i)
                _data[i] = m._data[i];
        }
        return *this;
    }

    /*
        @brief Move assignment operator
        @param m matrix to move from
    */
    matrix& 
    operator = (matrix&& m)
    {
        if (&m != this)
        {
            _deallocate();
            _width = m._width;
            _height = m._height;
            _data = m._data;
            m._data = nullptr;
        }
        return *this;
    }

    /*
        @brief Resizes the matrix
        @param width new width
        @param height new height
    */
    void
    resize(size_type height, size_type width)
    {
        if (height != _height || width != _width)
        {
            _deallocate();
            _height = height;
            _width = width;
            _allocate(size());
        }
    }

    /*
        @brief Fills the matrix with a value
        @param fillament value to fill the matrix with
    */
    void
    fill(const_reference fillament)
    {
        size_type size = this->size();
        #pragma omp parallel for
        for (size_type i = 0; i < size; ++i)
            _data[i] = fillament;
    }

    /*
        @brief Fills the matrix with a computed value
        @param fillament function to fill the matrix with
        @note The function is called with the row and column index as arguments
    */
    void
    fill(const std::function<T(size_type, size_type)>& fillament)
    {
        size_type size = this->size();
        #pragma omp parallel for
        for (size_type y = 0; y < _height; ++y)
        {
            auto row = this->row(y);
            #pragma omp parallel for
            for (size_type x = 0; x < _width; ++x)
            {
                row[x] = fillament(y, x);
            }
        }
    }

    /*
        @brief Returns matrix row pointer
        @param row row index
        @return pointer to the row
        @note This is a non-const version of the function
    */
    pointer
    row(size_type index) 
    {
        return _data + index * _width;
    }

    /*
        @brief Returns matrix row pointer
        @param row row index
        @return pointer to the row
        @note This is a const version of the function
    */
    const_pointer
    row(size_type index) const
    {
        return _data + index * _width;
    }

    /*
        @brief Returns matrix row pointer by subscript operator
        @param row row index
        @return pointer to the row
        @note This is a non-const version of the subscript operator
    */
    pointer 
    operator [] (size_type index)
    {
        return row(index);
    }

    /*
        @brief Returns matrix row pointer by subscript operator
        @param row row index
        @return pointer to the row
        @note This is a const version of the subscript operator
    */
    const_pointer 
    operator [] (size_type index) const 
    {
        return row(index);
    }

    /*
        @brief Safe access to the matrix element
        @param y row index
        @param x column index
        @return reference to the element
        @note This is a non-const version of the function
    */
    reference
    at(size_type y, size_type x)
    {
        if (y >= _height || x >= _width)
            throw std::out_of_range("matrix::at");
        return _data[y * _width + x];
    }

    /*
        @brief Safe access to the matrix element
        @param y row index
        @param x column index
        @return reference to the element
        @note This is a const version of the function
    */
    const_reference
    at(size_type y, size_type x) const
    {
        if (y >= _height || x >= _width)
            throw std::out_of_range("matrix::at");
        return _data[y * _width + x];
    }

    /*
        @brief Returns matrix size (width * height)
        @return matrix size
    */
    size_type 
    size() const
    {
        return _width * _height;
    }

    /*
        @brief Returns matrix width
        @return matrix width
    */
    size_type 
    width() const
    {
        return _width;
    }

    /*
        @brief Returns matrix height
        @return matrix height
    */
    size_type 
    height() const
    {
        return _height;
    }

    /*
        @brief Returns matrix data pointer
        @return matrix data pointer
        @note This is a non-const version of the function
    */
    pointer
    data()
    {
        return _data;
    }

    /*
        @brief Returns matrix data pointer
        @return matrix data pointer
        @note This is a const version of the function
    */
    const_pointer
    data() const
    {
        return _data;
    }

    /*
        @brief Returns iterator to the first element
        @return non-const iterator to the first element
    */
    iterator
    begin()
    {
        return _data;
    }

    /*
        @brief Returns iterator to the first element
        @return const iterator to the first element
    */
    const_iterator
    begin() const
    {
        return _data;
    }

    /*
        @brief Returns iterator to the element after the last element
        @return non-const iterator to the element after the last element
    */
    iterator
    end()
    {
        return _data + size();
    }

    /*
        @brief Returns iterator to the element after the last element
        @return const iterator to the element after the last element
    */
    const_iterator
    end() const
    {
        return _data + size();
    }

    /*
        @brief Matrix multiplication operator
        @param m matrix to multiply by
        @return result of the multiplication
    */
    matrix
    operator * (const matrix& m) const
    {
        if (_width != m._height) throw;
        matrix result(m._width, _height);

        #pragma omp parallel for
        for (size_type y = 0; y < result._height; ++y)
        {
            auto this_row = row(y);
            auto result_row = result.row(y);

            #pragma omp parallel for
            for (size_type x = 0; x < result._width; ++x)
            {
                T sum = 0;

                auto m_row = m.row(x);

                #pragma omp parallel for reduction(+:sum)
                for (size_type i = 0; i < _width; ++i)
                    sum += this_row[i] * m_row[i];

                result_row[x] = sum;
            }
        }

        return result;
    }

    /*
        @brief Matrix transposition function
        @return transposed matrix
    */
    void
    transpose()
    {
        #pragma omp parallel for
        for (size_type y = 0; y < _height; ++y)
        {
            #pragma omp parallel for
            for (size_type x = 0; x < y; ++x)
            {
                size_type idx1 = _width * y + x,
                          idx2 = _width * x + y;
                    
                size_type temp = _data[idx1];
                _data[idx1] = _data[idx2];
                _data[idx2] = temp;
            }
        }
    }

    /*
        @brief Matrix determinant function
        @return matrix determinant
    */
    value_type
    determinant() const
    {
        assert(_width == _height);

        if (_width == 1)
            return _data[0];

        value_type det = 0;

        #pragma omp parallel for reduction(+:det)
        for (size_type i = 0; i < _width; ++i)
        {
            matrix minor(_width - 1, _height - 1);

            for (size_type y = 1; y < _height; ++y)
            {
                size_type minor_y = y - 1;

                for (size_type x = 0; x < _width; ++x)
                {
                    if (x == i) continue;

                    size_type minor_x = x < i ? x : x - 1;

                    minor[minor_y][minor_x] = _data[y * _width + x];
                }
            }

            det += _data[i] * minor.determinant() * (i % 2 == 0 ? 1 : -1);
        }

        return det;
    }

    /*
        @brief Compares two matrices
        @param m matrix to compare with
        @return true if matrices are equal, false otherwise
    */
    bool
    operator == (const matrix& m) const
    {
        if (_width != m._width or _height != m._height) return false;

        for (size_type i = 0; i < size(); ++i)
            if (_data[i] != m._data[i]) return false;

        return true;
    }

    /*
        @brief Compares two matrices
        @param m matrix to compare with
        @return true if matrices are not equal, false otherwise
    */
    bool
    operator != (const matrix& m) const
    {
        return !((*this) == m);
    }

    /*
        @brief Read matrix from .mtx file
    */
    void
    read_mtx(const std::string& filename)
    {
        std::ifstream file(filename);

        if (!file.is_open())
            throw std::runtime_error("matrix::read_mtx: cannot open file");

        file.read(reinterpret_cast<char*>(&_width), sizeof(size_type));
        file.read(reinterpret_cast<char*>(&_height), sizeof(size_type));

        size_type size = this->size();
        _allocate(size);

        file.read(reinterpret_cast<char*>(_data), sizeof(value_type) * size);
    }

    /*
        @brief Write matrix to .mtx file
    */
    void
    write_mtx(const std::string& filename) const
    {
        std::ofstream file(filename);

        if (!file.is_open())
            throw std::runtime_error("matrix::write_mtx: cannot open file");

        file.write(reinterpret_cast<const char*>(&_width), sizeof(size_type));
        file.write(reinterpret_cast<const char*>(&_height), sizeof(size_type));

        size_type size = this->size();
        file.write(reinterpret_cast<const char*>(_data), sizeof(value_type) * size);
    }

    /*
        @brief Convolves matrix with kernel
        @param kernel kernel to convolve with
        @return convolved matrix
    */
    template <typename _T>
    matrix
    convolve(const matrix<_T> kernel) const
    {
        assert(kernel.width() % 2 == 1 and kernel.height() % 2 == 1);

        matrix result(_height, _width);

        #pragma omp parallel for
        for (size_type y = 0; y < height(); ++y)
        {
            auto this_row = row(y);
            auto result_row = result.row(y);

            #pragma omp parallel for
            for (size_type x = 0; x < width(); ++x)
            {
                value_type sum = 0;

                #pragma omp parallel for reduction(+:sum)
                for (size_type ky = 0; ky < kernel.height(); ++ky)
                {
                    size_type this_y = y + ky - kernel.height() / 2;

                    if (this_y < 0 or this_y >= _height) continue;

                    auto this_row = row(this_y);
                    
                    #pragma omp parallel for reduction(+:sum)
                    for (size_type kx = 0; kx < kernel.width(); ++kx)
                    {
                        size_type this_x = x + kx - kernel.width() / 2;

                        if (this_x < 0 or this_x >= _width) continue;

                        sum += this_row[this_x] * kernel[ky][kx];
                    }
                }

                result_row[x] = sum;
            }
        }

        return result;
    }

private:

    void
    _allocate(size_type size)
    {
        if (size == 0) return;

        void* ptr = operator new(size * sizeof(value_type));

        if (ptr == nullptr)
            throw std::runtime_error("matrix::_allocate: cannot allocate memory");

        _data = reinterpret_cast<value_type*>(ptr);
    }

    void
    _deallocate()
    {
        if (_data == nullptr) return;

        operator delete(_data);
    }

};

} // namespace lm
