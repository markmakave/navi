#pragma once

#include <iostream>
#include <functional>

namespace lm {

template <typename, unsigned ...dummy>
struct array;

/*
    @brief Static array.
*/
template <typename T, unsigned N>
struct array<T, N>
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

    array()
    {
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
            _data[i] = value_type();
    }

    array(const value_type& value)
    {
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
            _data[i] = value;
    }

    array(const array& other)
    {
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
            _data[i] = other._data[i];
    }

    array(array&& other)
    {
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
            _data[i] = other._data[i];
    }

    ~array()
    {}

    array&
    operator = (const array& other)
    {
        if (this == &other) return *this;

        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
            _data[i] = other._data[i];

        return *this;
    }

    array&
    operator = (array&& other)
    {
        if (this == &other) return *this;

        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i)
            _data[i] = other._data[i];

        return *this;
    }

    reference
    operator [] (size_type index)
    {
        return _data[index];
    }

    const_reference
    operator [] (size_type index) const
    {
        return _data[index];
    }

    reference
    at(size_type index)
    {
        if (index >= N)
            throw std::out_of_range("array::at");

        return _data[index];
    }

    const_reference
    at(size_type index) const
    {
        if (index >= N)
            throw std::out_of_range("array::at");

        return _data[index];
    }

    reference
    front()
    {
        return _data[0];
    }

    const_reference
    front() const
    {
        return _data[0];
    }

    reference
    back()
    {
        return _data[N - 1];
    }

    const_reference
    back() const
    {
        return _data[N - 1];
    }

    pointer
    data()
    {
        return _data;
    }

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
        return _data + N;
    }

    const_iterator
    end() const
    {
        return _data + N;
    }

    friend
    std::ostream&
    operator << (std::ostream& os, const array& a)
    {
        os << "[";
        for (size_t i = 0; i < N; ++i)
        {
            os << a._data[i];
            if (i < N - 1) os << ", ";
        }
        os << "]";
        return os;
    }

protected:

    value_type _data[N];

};

/*
    @brief Dynamic array.    
*/
template <typename T>
struct array<T>
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

    array()
    :   _data(nullptr),
        _size(0),
        _capacity(0)
    {}

    array(size_t size)
    :   _size(size),
        _capacity(size)
    {
        _data = _allocate(_capacity);
    }

    array(const array& other)
    :   _size(other._size),
        _capacity(other._size)
    {
        _data = _allocate(_capacity);

        #pragma omp parallel for
        for (size_t i = 0; i < _size; ++i)
            _data[i] = other._data[i];
    }

    array(array&& other)
    :   _data(other._data),
        _size(other._size),
        _capacity(other._capacity)
    {
        other._data = nullptr;
        other._size = 0;
        other._capacity = 0;
    }

    ~array()
    {
        _deallocate(_data);
    }

    array&
    operator = (const array& other)
    {
        if (this == &other) return *this;

        _deallocate(_data);

        _size = other._size;
        _capacity = other._size;
        _data = _allocate(_capacity);

        #pragma omp parallel for
        for (size_t i = 0; i < _size; ++i)
            _data[i] = other._data[i];

        return *this;
    }

    array&
    operator = (array&& other)
    {
        if (this == &other) return *this;

        _deallocate(_data);

        _data = other._data;
        _size = other._size;
        _capacity = other._capacity;

        other._data = nullptr;
        other._size = 0;
        other._capacity = 0;

        return *this;
    }

    reference
    operator [] (size_type index)
    {
        return _data[index];
    }

    const_reference
    operator [] (size_type index) const
    {
        return _data[index];
    }

    reference
    at(size_type index)
    {
        if (index >= _size)
            throw std::out_of_range("array::at");

        return _data[index];
    }

    const_reference
    at(size_type index) const
    {
        if (index >= _size)
            throw std::out_of_range("array::at");

        return _data[index];
    }

    reference
    front()
    {
        return _data[0];
    }

    const_reference
    front() const
    {
        return _data[0];
    }

    reference
    back()
    {
        return _data[_size - 1];
    }

    const_reference
    back() const
    {
        return _data[_size - 1];
    }

    pointer
    data()
    {
        return _data;
    }

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

    const_iterator
    cbegin() const
    {
        return _data;
    }

    iterator
    end()
    {
        return _data + _size;
    }

    const_iterator
    end() const
    {
        return _data + _size;
    }

    const_iterator
    cend() const
    {
        return _data + _size;
    }

    bool
    empty() const
    {
        return _size == 0;
    }

    size_type
    size() const
    {
        return _size;
    }

    size_type
    capacity() const
    {
        return _capacity;
    }

    void
    reserve(size_type size)
    {
        if (size <= _capacity) return;

        pointer new_data = _allocate(size);

        #pragma omp parallel for
        for (size_t i = 0; i < _size; ++i)
            new_data[i] = _data[i];

        _deallocate(_data);

        _data = new_data;
        _capacity = size;
    }

    friend
    std::ostream&
    operator << (std::ostream& os, const array& a)
    {
        os << "[";
        for (size_t i = 0; i < a._size; ++i)
        {
            os << a._data[i];
            if (i < a._size - 1) os << ", ";
        }
        os << "]";

        return os;
    }

    void
    push(const_reference value)
    {
        if (_size == _capacity)
        {
            size_type new_capacity = _capacity == 0 ? 1 : _capacity * 2;
            reserve(new_capacity);
        }

        _data[_size++] = value;
    }

    void
    pop()
    {
        if (_size == 0) return;

        --_size;
    }

    void
    clear()
    {
        _size = 0;
    }

    void
    resize(size_type size)
    {
        if (size > _capacity)
        {
            reserve(size);
        }

        _size = size;
    }

    void
    resize(size_type size, const_reference value)
    {
        if (size > _capacity)
        {
            reserve(size);
        }

        #pragma omp parallel for
        for (size_t i = _size; i < size; ++i)
            _data[i] = value;

        _size = size;
    }

    value_type
    sum() const
    {
        value_type sum = 0;

        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < _size; ++i)
            sum += _data[i];

        return sum;
    }

    value_type
    mean() const
    {
        return sum() / _size;
    }

    value_type
    min() const
    {
        value_type min = _data[0];

        #pragma omp parallel for reduction(min:min)
        for (size_t i = 1; i < _size; ++i)
            min = std::min(min, _data[i]);

        return min;
    }

    value_type
    max() const
    {
        value_type max = _data[0];

        #pragma omp parallel for reduction(max:max)
        for (size_t i = 1; i < _size; ++i)
            max = std::max(max, _data[i]);

        return max;
    }

    void
    affect(std::function<value_type(value_type)> f)
    {
        #pragma omp parallel for
        for (size_t i = 0; i < _size; ++i)
            _data[i] = f(_data[i]);
    }

private:

    static
    pointer
    _allocate(size_type size)
    {
        if (size == 0)
        {
            return nullptr;
        }

        void* ptr = operator new(size * sizeof(value_type));

        if (ptr == nullptr)
            throw std::bad_alloc();

        return reinterpret_cast<value_type*>(ptr);
    }

    static
    void
    _deallocate(pointer ptr)
    {
        if (ptr == nullptr) return;

        operator delete(ptr);
    }

protected:

    pointer     _data;
    size_type   _size;
    size_type   _capacity;

};

}
