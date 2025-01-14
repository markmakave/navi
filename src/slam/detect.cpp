
#include "slam/kernels.hpp"
#include "slam/fast.hpp"

void lumina::slam::detect(
    const matrix<gray>& input,
    const int                   threshold,
          matrix<bool>&         output
) {
    output.reshape(input.shape());

    #pragma omp parallel for
    for (size_t y = 3; y < input.shape(1) - 3; ++y)
        for (size_t x = 3; x < input.shape(0) - 3; ++x)
        {
            gray center = input(x, y);
            int circle[16] = {
                input(x, y - 3), input(x + 1, y - 3), input(x + 2, y - 2), input(x + 3, y - 1),
                input(x + 3, y), input(x + 3, y + 1), input(x + 2, y + 2), input(x + 1, y + 3),
                input(x, y + 3), input(x - 1, y + 3), input(x - 2, y + 2), input(x - 3, y + 1),
                input(x - 3, y), input(x - 3, y - 1), input(x - 2, y - 2), input(x - 1, y - 3)
            };

            output(x, y) = fast<11>(center, circle, threshold);
        }
}
