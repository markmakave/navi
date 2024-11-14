#include "slam/kernels.hpp"
#include "slam/fast.hpp"

namespace lumina::slam {

void detect(
    const matrix<lumina::gray>& input, 
          matrix<bool>&         output
) {
    output.reshape(input.shape());

    #pragma omp parallel for
    for (size_t y = 3; y < input.shape(1) - 3; ++y)
        for (size_t x = 3; x < input.shape(0) - 3; ++x)
        {
            lumina::gray center = input(x, y);
            int circle[16] = {
                input(x, y - 3), input(x + 1, y - 3), input(x + 2, y - 2), input(x + 3, y - 1),
                input(x + 3, y), input(x + 3, y + 1), input(x + 2, y + 2), input(x + 1, y + 3),
                input(x, y + 3), input(x - 1, y + 3), input(x - 2, y + 2), input(x - 3, y + 1),
                input(x - 3, y), input(x - 3, y - 1), input(x - 2, y - 2), input(x - 1, y - 3)
            };

            output(x, y) = fast<9>(center, circle, 8);
        } 
}

}
