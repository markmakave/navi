#pragma once

#include <functional>
#include <stdexcept>
#include <assert.h>
#include <fstream>
#include <random>

#include <base/array.hpp>

namespace lm {

/// @brief Matrix class with basic operations and STL-like interface
/// @tparam T Type of the elements in the matrix (int, float, double, etc.)
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

public:

    /// @brief Default constructor
    matrix()
    :   _data(nullptr),
        _height(0),
        _width(0)
    {}

    /// @brief Constructor with height and width
    /// @param height Height of the matrix
    /// @param width Width of the matrix
    matrix(size_type height, size_type width) 
    :   _height(height),
        _width(width)
    {
        _allocate(size());
    }

    /// @brief Constructor with height, width and fillament
    /// @param height Height of the matrix
    /// @param width Width of the matrix
    /// @param fillament Element to fill the matrix with
    matrix(size_type height, size_type width, const_reference fillament)
    :   _height(height),
        _width(width)
    {
        _allocate(size());
        fill(fillament);
    }

    /// @brief Copy constructor
    /// @param m matrix to copy from
    matrix(const matrix<T>& m) 
    :   _height(m._height),
        _width(m._width)
    {
        size_type size = this->size();
        _allocate(size);

        for (size_type i = 0; i < size; ++i)
            _data[i] = m._data[i];
    }

    /// @brief Move constructor
    /// @param m matrix to move from
    matrix(matrix&& m) 
    :   _data(m._data),
        _height(m._height),
        _width(m._width)
    {
        m._data = nullptr;
        m._height = 0;
        m._width = 0;
    }

    /// @brief Constructs a matrix from an matrix-like initializer list
    /// @param list Matrix-like initializer list
    matrix(const std::initializer_list<std::initializer_list<T>>& list)
    :   _height(list.size()),
        _width(list.begin()->size())
    {
        _allocate(size());

        size_type i = 0;
        for (const auto& row : list)
        {
            assert(row.size() == _width);
            for (const auto& element : row)
                _data[i++] = element;
        }
    }

    /// @brief Constructs a matrix from an '.mtx' file
    /// @param filename Path to the '.mtx' file
    matrix(const std::string& filename)
    {
        read_mtx(filename);
    }
    
    /// @brief Constructs a matrix from matrix of different type
    /// @param m matrix of different type
    /// @param converter function to convert elements of different type to elements of this type
    template <typename U>
    matrix(const matrix<U>& m, const std::function<T(U)>& converter = [](const U& a) { return static_cast<T>(a); })
    :   _height(m.height()),
        _width(m.width())
    {
        size_type size = this->size();
        _allocate(size);

        for (size_type i = 0; i < size; ++i)
            _data[i] = converter(m.data()[i]);
    }

    matrix(const lm::array<value_type>& a, size_type height, size_type width)
    :   _height(height),
        _width(width)
    {
        size_type size = this->size();
        _allocate(size);

        for (size_type i = 0; i < size; ++i)
            _data[i] = a[i];
    }
    
    /// @brief Destructor
    ~matrix()
    {
        _deallocate();
    }

    /// @brief Copy assignment operator
    /// @param m matrix to copy from
    matrix& 
    operator = (const matrix& m)
    {
        if (&m != this)
        {
            resize(m._height, m._width);
            
            size_type size = this->size();
            for (size_type i = 0; i < size; ++i)
                _data[i] = m._data[i];
        }
        return *this;
    }

    /// @brief Move assignment operator
    /// @param m matrix to move from
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

    /// @brief Resizes the matrix to the given dimensions
    /// @param height New height of the matrix
    /// @param width New width of the matrix
    void
    resize(size_type height, size_type width)
    {
        if (size() == height * width)
        {
            _height = height;
            _width = width;
        } else {
            _deallocate();
            _height = height;
            _width = width;
            _allocate(size());
        }
    }

    /// @brief Fill the matrix with a given value
    /// @param fillament Value to fill the matrix with
    void
    fill(const_reference fillament)
    {
        size_type size = this->size();
        for (size_type i = 0; i < size; ++i)
            _data[i] = fillament;
    }

    /// @brief Fill the matrix with a given function
    /// @param fillament Function to fill the matrix with
    void
    fill(const std::function<T(size_type, size_type)>& fillament)
    {
        for (size_type y = 0; y < _height; ++y)
        {
            auto row = this->row(y);
            for (size_type x = 0; x < _width; ++x)
            {
                row[x] = fillament(y, x);
            }
        }
    }

    void
    randomize(const_reference min = 0, const_reference max = 1)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(min, max);

        size_type size = this->size();
        for (size_type i = 0; i < size; ++i)
            _data[i] = dis(gen);
    }

    /// @brief Maps the matrix with a given function
    /// @param f Function to map the matrix with
    matrix
    map(const std::function<T(const_reference)>& f)
    {
        matrix result(_height, _width);
        size_type size = this->size();
        for (size_type i = 0; i < size; ++i)
            result._data[i] = f(_data[i]);
        return result;
    }

    /// @brief Returns matrix row pointer
    /// @param index Row index
    /// @return Pointer to the row
    pointer
    row(size_type index) 
    {
        return _data + index * _width;
    }

    /// @brief Returns matrix row pointer
    /// @param index Row index
    /// @return Pointer to the row
    const_pointer
    row(size_type index) const
    {
        return _data + index * _width;
    }

    /// @brief Returns matrix row
    /// @param index Row index
    /// @return Row
    pointer 
    operator [] (size_type index)
    {
        return row(index);
    }

    /// @brief Returns matrix row
    /// @param index Row index
    /// @return Row
    const_pointer 
    operator [] (size_type index) const 
    {
        return row(index);
    }

    /// @brief Element access operator
    /// @param y 
    /// @param x
    /// @return Reference to the element
    reference
    operator () (size_type y, size_type x)
    {
        return row(y)[x];
    }

    /// @brief Element access operator
    /// @param y 
    /// @param x 
    /// @return Const reference to the element
    const_reference
    operator () (size_type y, size_type x) const
    {
        return row(y)[x];
    }

    /// @brief Element access operator
    /// @param y Row index
    /// @param x Column index
    /// @return Reference to the element
    reference
    at(size_type y, size_type x)
    {
        if (y >= _height || x >= _width)
            throw std::out_of_range("matrix::at");
        return _data[y * _width + x];
    }

    /// @brief Element access operator
    /// @param y Row index
    /// @param x Column index
    /// @return Reference to the element
    const_reference
    at(size_type y, size_type x) const
    {
        if (y >= _height || x >= _width)
            throw std::out_of_range("matrix::at");
        return _data[y * _width + x];
    }

    /// @brief Matrix size getter
    /// @return Matrix size in elements
    size_type 
    size() const
    {
        return _width * _height;
    }

    /// @brief Matrix width getter
    /// @return Matrix width
    size_type 
    width() const
    {
        return _width;
    }

    /// @brief Matrix height getter
    /// @return Matrix height
    size_type 
    height() const
    {
        return _height;
    }

    /// @brief Matrix data pointer getter
    /// @return Matrix data pointer
    pointer
    data()
    {
        return _data;
    }

    /// @brief Matrix data pointer getter
    /// @return Matrix data pointer
    const_pointer
    data() const
    {
        return _data;
    }
    
    iterator
    begin()
    {
        return _data;
    }

    
    const_iterator
    begin() const
    {
        return _data;
    }

    iterator
    end()
    {
        return _data + size();
    }

    const_iterator
    end() const
    {
        return _data + size();
    }

    /// @brief Matrix multiplication operator
    /// @param m Matrix to multiply with
    /// @return Resulting matrix
    matrix
    operator * (const matrix& m) const
    {
        if (_width != m._height)
        {
            throw std::invalid_argument("matrix::operator *");
        }

        matrix result(_height, m._width);

        for (size_type y = 0; y < result._height; ++y)
        {
            auto this_row = row(y);
            auto result_row = result.row(y);

            for (size_type x = 0; x < result._width; ++x)
            {
                result_row[x] = 0;
                for (size_type i = 0; i < _width; ++i)
                {
                    result_row[x] += this_row[i] * m[i][x];
                }
            }
        }

        return result;
    }

    /// @brief Matrix addition operator
    /// @param m Matrix to add
    /// @return Resulting matrix
    matrix
    operator + (const matrix& m) const
    {
        if (_width != m._width || _height != m._height)
        {
            throw std::invalid_argument("matrix::operator +");
        }

        matrix result(_height, _width);

        for (size_type y = 0; y < result._height; ++y)
        {
            auto this_row = row(y);
            auto m_row = m.row(y);
            auto result_row = result.row(y);

            for (size_type x = 0; x < result._width; ++x)
            {
                result_row[x] = this_row[x] + m_row[x];
            }
        }

        return result;
    }

    /// @brief Matrix subtraction operator
    /// @param m Matrix to subtract
    /// @return Resulting matrix
    matrix
    operator - (const matrix& m) const
    {
        if (_width != m._width || _height != m._height)
        {
            throw std::invalid_argument("matrix::operator -");
        }

        matrix result(_height, _width);

        for (size_type y = 0; y < result._height; ++y)
        {
            auto this_row = row(y);
            auto m_row = m.row(y);
            auto result_row = result.row(y);

            for (size_type x = 0; x < result._width; ++x)
            {
                result_row[x] = this_row[x] - m_row[x];
            }
        }

        return result;
    }

    matrix
    operator * (const_reference v) const
    {
        matrix result(_height, _width);

        for (size_type y = 0; y < result._height; ++y)
        {
            auto this_row = row(y);
            auto result_row = result.row(y);

            for (size_type x = 0; x < result._width; ++x)
            {
                result_row[x] = this_row[x] * v;
            }
        }

        return result;
    }

    /// @brief Mutating matrix multiplication operator
    /// @param m Matrix to multiply by
    /// @return Reference to parent matrix
    matrix&
    operator *= (const matrix& m)
    {
        *this = *this * m;
        return *this;
    }

    /// @brief Mutating matrix addition operator
    /// @param m Matrix to add
    /// @return Reference to parent matrix
    matrix&
    operator += (const matrix& m)
    {
        *this = *this + m;
        return *this;
    }

    /// @brief Mutating matrix subtraction operator
    /// @param m Matrix to subtract
    /// @return Reference to parent matrix
    matrix&
    operator -= (const matrix& m)
    {
        *this = *this - m;
        return *this;
    }

    /// @brief Matrix transposition function
    /// @return Transposed matrix
    matrix
    transpose() const
    {
        matrix result(_width, _height);

        for (size_type y = 0; y < _height; ++y)
        {
            auto this_row = row(y);

            for (size_type x = 0; x < _width; ++x)
            {
                result[x][y] = this_row[x];
            }
        }

        return result;
    }

    /// @brief Matrix determinant calculation function
    /// @return Matrix determinant
    value_type
    determinant() const
    {
        assert(_width == _height);

        if (_width == 1)
            return _data[0];

        value_type det = 0;

        for (size_type i = 0; i < _width; ++i)
        {
            matrix minor_submatrix(_width - 1, _height - 1);

            for (size_type y = 1; y < _height; ++y)
            {
                size_type minor_y = y - 1;

                for (size_type x = 0; x < _width; ++x)
                {
                    if (x == i) continue;

                    size_type minor_x = x < i ? x : x - 1;

                    minor_submatrix[minor_y][minor_x] = _data[y * _width + x];
                }
            }

            det += _data[i] * minor_submatrix.determinant() * (i % 2 == 0 ? 1 : -1);
        }

        return det;
    }

    template <typename U>
    matrix
    convolve(const matrix<U>& kernel) const
    {
        assert((kernel.width() & 1) == 1);
        assert((kernel.height() & 1) == 1);

        typedef decltype(T() * U()) accumulator_type;

        matrix result(_height, _width);

        for (size_type y = kernel.height() / 2; y < _height - kernel.height() / 2; ++y)
        {
            auto this_row = row(y);
            auto result_row = result.row(y);

            for (size_type x = kernel.width() / 2; x < _width - kernel.width() / 2; ++x)
            {
                accumulator_type sum = 0;

                for (size_type ky = 0; ky < kernel.height(); ++ky)
                {
                    auto kernel_row = kernel.row(ky);

                    for (size_type kx = 0; kx < kernel.width(); ++kx)
                    {
                        sum += this_row[x - kernel.width() / 2 + kx] * kernel_row[kx];
                    }
                }

                // clamp based on type traits
                if (sum > std::numeric_limits<value_type>::max())
                    result_row[x] = std::numeric_limits<value_type>::max();
                else if (sum < std::numeric_limits<value_type>::min())
                    result_row[x] = std::numeric_limits<value_type>::min();
                else
                    result_row[x] = sum;
            }
        }

        return result;
    }

    /// @brief Matrix trace calculation function
    /// @return Matrix trace
    value_type
    trace() const
    {
        assert(_width == _height);

        value_type trace = 0;

        for (size_type i = 0; i < _width; ++i)
            trace += _data[i * _width + i];

        return trace;
    }

    /// @brief Matrix element sum calculation function
    /// @return Matrix element sum
    value_type
    sum() const
    {
        value_type sum = 0;

        for (const auto& element : *this)
            sum += element;

        return sum;
    }

    /// @brief Matrix element mean calculation function
    /// @return Matrix element mean
    value_type
    mean() const
    {
        return sum() / size();
    }

    /// @brief Matrix element variance calculation function
    /// @return Matrix element variance
    value_type
    variance() const
    {
        value_type mean = this->mean();
        value_type variance = 0;

        for (size_type i = 0; i < size(); ++i)
            variance += (_data[i] - mean) * (_data[i] - mean);

        return variance / size();
    }

    /// @brief Matrix element standard deviation calculation function
    /// @return Matrix element standard deviation
    value_type
    standard_deviation() const
    {
        return std::sqrt(variance());
    }

    /// @brief Inverse matrix calculation function
    /// @return Inverse matrix
    matrix
    inverse() const
    {
        assert(_width == _height);

        matrix result(_width, _height);

        value_type det = determinant();

        for (size_type y = 0; y < _height; ++y)
        {
            for (size_type x = 0; x < _width; ++x)
            {
                matrix minor_submatrix(_width - 1, _height - 1);

                for (size_type minor_y = 0; minor_y < _height - 1; ++minor_y)
                {
                    for (size_type minor_x = 0; minor_x < _width - 1; ++minor_x)
                    {
                        size_type this_y = minor_y < y ? minor_y : minor_y + 1;
                        size_type this_x = minor_x < x ? minor_x : minor_x + 1;

                        minor_submatrix[minor_y][minor_x] = _data[this_y * _width + this_x];
                    }
                }

                result[y][x] = minor_submatrix.determinant() * ((x + y) % 2 == 0 ? 1 : -1) / det;
            }
        }

        return result.transpose();
    }

    /// @brief Matrix element-wise comparison operator
    /// @param m Matrix to compare with
    /// @return True if matrices are equal, false otherwise
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

    friend
    std::ostream&
    operator << (std::ostream& os, const matrix& m)
    {
        std::cout << '[';

        for (size_type y = 0; y < m._height; ++y)
        {
            std::cout << '[';

            for (size_type x = 0; x < m._width; ++x)
            {
                std::cout << m._data[y * m._width + x];

                if (x != m._width - 1)
                    std::cout << ", ";
            }

            std::cout << ']';

            if (y != m._height - 1)
                std::cout << ", ";
        }

        std::cout << ']';

        return os;
    }

private:

    void
    _allocate(size_type size)
    {
        if (size == 0)
        {
            _data = nullptr;
            return;
        }

        pointer ptr = new value_type[size];

        if (ptr == nullptr)
            throw std::bad_alloc();

        _data = ptr;
    }

    void
    _deallocate()
    {
        if (_data == nullptr) return;

        delete[] _data;
    }

protected:

    pointer _data;
    size_type _height, _width;

};

} // namespace lm
