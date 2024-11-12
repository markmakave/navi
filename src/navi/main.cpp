#include <iostream>
#include <vector>

#include "util/utility.hpp"

#include "base/tensor.hpp"
#include "base/tuple.hpp"


struct A
{
    A() { std::cout << "A()\n"; }
    A(const A&) { std::cout << "A(const A&)\n"; }
    A(A&&) { std::cout << "A(A&&)\n"; }
    ~A() { std::cout << "~A()\n"; }

    A& operator= (const A&) { std::cout << "A::operator=(const A&)\n"; return *this; }
    A& operator= (A&&) { std::cout << "A::operator=(A&&)\n"; return *this; }
};

template <typename T>
void print(const T& t)
{
    std::cout << t.template get<0>() << '\n';

    if constexpr (T::degree > 1)
        print(static_cast<const typename T::base&>(t));
}

int main(int argc, char** argv)
{
    auto t = lumina::tuple(1, 2.f, 3.0, '4', "5");

    print(t);

    return 0;
}
