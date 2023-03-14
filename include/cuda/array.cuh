#pragma once

namespace lm {
namespace cuda {

template <typename, unsigned ...dummy>
struct array;

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

    __host__ __device__
    array()
    {}

    __host__ __device__
    array(const_reference value)
    {
        fill(value);
    }

    __host__ __device__
    reference
    operator [] (size_type index)
    {
        return _data[index];
    }

private:

    value_type _data[N];

}

}
}
