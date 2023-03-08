#pragma once

#include "util/log.hpp"

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
        _allocate();
    }

    /// @brief Copy constructor
    /// @param m matrix to copy from
    matrix(const matrix<T>& m) 
    :   _height(m._height),
        _width(m._width)
    {
        _allocate();

        size_type size = this->size();
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
            
            _data = m._data;
            _height = m._height;
            _width = m._width;

            m._data = nullptr;
            m._height = 0;
            m._width = 0;
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
        }
        else
        {
            _deallocate();

            _height = height;
            _width = width;

            _allocate();
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

private:

    void
    _allocate()
    {
        lm::log::debug("Allocating", size() * sizeof(T), "bytes on HEAP");

        if (size() == 0)
        {
            _data = nullptr;
        }
        else
        {
            _data = reinterpret_cast<pointer>(operator new[](size() * sizeof(T)));
            if (_data == nullptr)
                lm::log::error("Memory allocation failed");
        }
    }

    void
    _deallocate()
    {
        lm::log::debug("Deallocating", (void*)_data, "from HEAP");

        operator delete[](_data);
    }

protected:

    pointer _data;
    size_type _height, _width;

};

} // namespace lm
