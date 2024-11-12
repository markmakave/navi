#pragma once

namespace lumina {

template <typename... Types>
class tuple;

template <typename T>
class tuple<T>
{
public:

    static constexpr size_t degree = 1;

public:

    tuple(T&& data)
    :   _data(std::forward<T>(data))
    {}

    template <size_t N>
    requires (N == 0)
    T& get()
    {
        return _data;
    }

    template <size_t N>
    requires (N == 0)
    const T& get() const
    {
        return _data;
    }

protected:

    T _data;
};

template <typename T, typename... Types>
requires (sizeof...(Types) > 0)
class tuple<T, Types...> : public tuple<Types...>
{
public:

    using base = tuple<Types...>;

    static constexpr size_t degree = base::degree + 1;

public:

    tuple(T&& data, Types&&... args)
    :   base(std::forward<Types>(args)...),
        _data(std::forward<T>(data))
    {}

    template <size_t N>
    auto& get()
    {
        if constexpr (N == 0)
            return _data;
        else  
            return base::template get<N - 1>();
    }

    template <size_t N>
    const auto& get() const
    {
        if constexpr (N == 0)
            return _data;
        else  
            return base::template get<N - 1>();
    }

protected:

    T _data;
};

template <typename... Types>
tuple(Types&&...) -> tuple<Types...>;

}
