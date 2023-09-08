#include <functional>
#include <iostream>
#include <random>

#define NDEBUG

// #include "lumina.hpp"

#include "base/static_tensor.hpp"

#define FILENAME "../resource/include.txt"

const char text[] = 
#include FILENAME
;

int
main(int argc, char** argv)
{
    // lumina::compile_time::tensor<float, lumina::compile_time::shape<2, 2, 2>> tensor = {
    //     {
    //         {{1.f}, {2.f}}, 
    //         {{3.f}, {4.f}}
    //     },
    //     {
    //         {{5.f}, {6.f}}, 
    //         {{7.f}, {8.f}}
    //     }
    // };

    lumina::compile_time::tensor<float, lumina::compile_time::shape<>> tensor = {1.f};

    return 0;
}
