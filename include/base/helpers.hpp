#pragma once

#include "base/tuple.hpp"

namespace lumina {

template <typename... Args>
struct pack_iterator : public tuple<Args&& ...>
{
    pack_iterator(Args&& ...args)
        : tuple<Args&& ...>(args...)
    {}

    auto
    operator [](int i) const
    {
        return get<i>()::begin();
    }

    pack_iterator&
    operator ++()
    {
        return *this;
    }

    pack_iterator
    operator ++(int)
    {
        return *this;
    }

    pack_iterator&
    operator *()
    {
        return *this;
    }


};

template <typename... Args>
pack_iterator<Args...> zip(Args&&... args)
{
    return {args...};
}


}
