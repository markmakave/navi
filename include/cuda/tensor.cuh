#pragma once

#include <cuda_runtime.h>

namespace lm {
namespace cuda {

template <int N, typename T, typename _alloc>
class tensor
{
public:

    using value_type        = T;
    using pointer           = value_type*;
    using const_pointer     = const value_type*;
    using reference         = value_type&;
    using const_reference   = const value_type&;
    using iterator          = pointer;
    using const_iterator    = const_pointer;
    using size_type         = int64_t;

public:

    __host__ __device__
    tensor()
    :   _data(nullptr),
        _shape{}
    {}
    
    __host__ __device__
    template <typename... Size>
    tensor(Size ...sizes)
    :   _shape{sizes...}
    {
        static_assert(sizeof...(Size) == N);
        _allocate();
    }

    __host__ __device__
    tensor(const tensor& c)
    {
        for (size_type n = 0; n < N; ++n)
            _shape[n] = c._shape[n];

        _allocate();

        _alloc::copy(c._data, _data, size()); 
    }

    __host__ __device__
    tensor(tensor&& c)
    :   _data(c._data)
    {
        for (size_type n = 0; n < N; ++n)
            _shape[n] = c._shape[n];

        for (size_type n = 0; n < N; ++n)
            c._shape[n] = 0;
            
        c._data = nullptr;
    }

    __host__ __device__
    ~tensor()
    {
        _deallocate();
    }

    __host__ __device__
    const size_type*
    shape() const
    {
        return _shape;
    }

    __host__ __device__
    pointer
    data()
    {
        return _data;
    }

    __host__ __device__
    const_pointer
    data() const
    {
        return _data;
    }

    __host__ __device__
    size_type
    size() const
    {
        size_type s = 1;
        for (size_type n = 0; n < N; ++n)
            s *= _shape[n];
        return s;
    }

    __host__ __device__
    template <typename... Size>
    void
    reshape(Size ...sizes)
    {
        static_assert(sizeof...(Size) == N);

        _deallocate();

        size_type shape[N] = {sizes...};
        for (size_type n = 0; n < N; ++n)
            _shape[n] = shape[n];

        _allocate();
    }

    __host__ __device__
    template <typename... Index>
    decltype(auto)
    operator () (Index ...index)
    {
        static_assert(sizeof...(Index) == N);

        size_type indices[N] = {index...};

        size_type offset = 0;
        size_type dim = 1;
        for (size_type n = 0; n < N; ++n)
        {
            offset += dim * indices[n];
            dim *= _shape[n];
        }

        return _alloc::access(_data + offset);
    }

    __host__ __device__
    template <typename... Index>
    decltype(auto)
    operator () (Index ...index) const
    {
        static_assert(sizeof...(Index) == N);

        size_type indices[N] = {index...};

        size_type offset = 0;
        size_type dim = 1;
        for (size_type n = 0; n < N; ++n)
        {
            offset += dim * indices[n];
            dim *= _shape[n];
        }

        return _alloc::access(_data + offset);
    }

    __host__
    friend
    std::ostream&
    operator << (std::ostream& out, const tensor& t)
    {
        // TODO

        return out;
    }

protected:

    pointer   _data;
    size_type _shape[N];

private:

    __host__ __device__
    void
    _allocate()
    {
        _data = _alloc::allocate(size());
    }

    __host__ __device__
    void
    _deallocate()
    {
        _alloc::deallocate(_data);
    }
};

}
}
