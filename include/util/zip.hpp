#pragma once

namespace lumina
{

template <typename... Args>
class zip;

template <typename T>
class zip<T>
{
public:

    using iterator = typename std::decay_t<T>::iterator;

    struct proxy
    {
        iterator::reference ref;

        friend std::ostream& operator<< (std::ostream& os, const proxy& p)
        {
            return os << p.ref;
        }
    };

public:

    zip(T&& container)
    :   _container(container)
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

    T&& _container;
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
                (static_cast<const base::iterator&>(*this) !=
                static_cast<const base::iterator&>(other));
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
        std::decay_t<T>::iterator::reference ref;

        friend std::ostream& operator<< (std::ostream& os, const proxy& p)
        {
            return os << p.ref << " " << static_cast<const base::proxy&>(p);
        }
    };

public:

    zip(T&& container, Args&&... next)
    :   base(next...),
        _container(container)
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

    T&& _container;
};

template <typename... Args>
zip(Args&&...) -> zip<Args...>;

}
