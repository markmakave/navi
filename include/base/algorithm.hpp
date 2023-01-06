#pragma once>

namespace lm {

template <typename T, typename U>
inline bool
equal(const T& a_begin, const T& a_end, const U& b_begin, const U& b_end)
{
    if (std::distance(a_begin, a_end) != std::distance(b_begin, b_end))
    {
        return false;
    }

    for (auto a = a_begin, b = b_begin; a != a_end; ++a, ++b)
    {
        if (*a != *b)
        {
            return false;
        }
    }

    return true;
}

}