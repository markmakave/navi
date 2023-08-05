#pragma once

namespace lumina {

template <typename T, typename... Args>
class tuple : public tuple<Args...>
{
public:

    using base = tuple<Args...>;

public:

    tuple(T&& data, Args&&... args)
        : base(args...),
        _data(data)
    {}

    template <int N>
    decltype(auto) get() const
    {
        return base::get<N>();
    }

    template <>
    decltype(auto) get<0>() const
    {
        return _data;
    }

    template <int N>
    decltype(auto) get()
    {
        return base::get<N>();
    }

    template <>
    decltype(auto) get<0>()
    {
        return _data;
    }

protected:

    T _data;
};

}
