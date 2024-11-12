#pragma once

namespace lumina
{

/**
 * Zip class for itrator zipping
 * 
 * Requires:
 * - T::iterator
 * - T::iterator::reference
 */

template <typename... Args>
class zip;

template <typename T>
class zip<T>
{
public:

    using iterator = typename std::decay_t<T>::iterator;

    struct proxy
    {
        typename iterator::reference ref;

        friend std::ostream& operator<< (std::ostream& os, const proxy& p)
        {
            return os << p.ref;
        }
    };

public:

    zip(T&& container)
    :   _container(std::forward<T>(container))
    {}

    iterator begin()
    {
        return _container.begin();
    }

    iterator end()
    {
        return _container.end();
    }

protected:

    T _container;
};

template <typename T, typename... Args>
requires (sizeof...(Args) > 0)
class zip<T, Args...> : public zip<Args...>
{
    using base = zip<Args...>;

public:

    struct proxy;

    struct iterator : public base::iterator
    {
        typename std::decay_t<T>::iterator it;

        bool operator!= (const iterator& other) const
        {
            return (it != other.it) and
                (static_cast<const typename base::iterator&>(*this) !=
                static_cast<const typename base::iterator&>(other));
        }

        iterator& operator++ ()
        {
            ++it;
            base::iterator::operator++();
            return *this;
        }

        proxy operator* ()
        {
            return { *static_cast<typename base::iterator&>(*this), *it };
        }
    };

    struct proxy : public base::proxy
    {
        typename std::decay_t<T>::iterator::reference ref;

        friend std::ostream& operator<< (std::ostream& os, const proxy& p)
        {
            return os << p.ref << " " << static_cast<const typename base::proxy&>(p);
        }
    };

public:

    zip(T&& container, Args&&... next)
    :   base(std::forward<Args>(next)...),
        _container(std::forward<T>(container))
    {}

    iterator begin()
    {
        return { base::begin(), _container.begin() };
    }

    iterator end()
    {
        return { base::end(), _container.end() };
    }

protected:

    T _container;
};

template <typename... Args>
zip(Args&&...) -> zip<Args...>;


template <typename size_type>
class range
{
public:

    struct iterator
    {
        using reference = size_type&;

        size_type value;
        size_type step;

        bool operator!= (const iterator& other) const
        {
            return step > 0 ? (value < other.value) : (value > other.value);
        }

        iterator& operator++ ()
        {
            value += step;
            return *this;
        }

        size_type& operator* ()
        {
            return value;
        }
    };

public:

    range(size_type end)
    :   range(0, end)
    {}

    range(size_type begin, size_type end, size_type step = 1)
    :   _begin { begin, step },
        _end { end }
    {}

    iterator begin() const { return _begin; }
    iterator end() const { return _end; }

protected:

    iterator _begin, _end;
};


/**
 * Enumerate class for iterator enumeration
 * 
 * Requires:
 * - T::iterator
 * - T::iterator::reference
 * - T::iterator::size_type
 */

template <typename T>
class enumerate
{
public:

    enumerate(T&& container)
    :   _container({container.size()}, std::forward<T>(container))
    {}

    auto begin()
    {
        return _container.begin();
    }

    auto end()
    {
        return _container.end();
    }

protected:

    zip<range<typename std::decay_t<T>::size_type>, T> _container;
};

template <typename T>
enumerate(T&&) -> enumerate<T>;

}
