#pragma once

#include "base/matrix.hpp"
#include <cmath>

namespace lm {
namespace kernel {

matrix<float>
gaussian(int size, float sigma)
{
    matrix<float> kernel(size, size);

    float sum = 0.0f;
    int center = size / 2;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel[j][i] = exp(-0.5f * (pow((i - center) / sigma, 2.0f) + pow((j - center) / sigma, 2.0f)))
                         / (2.0f * M_PI * sigma * sigma);
            sum += kernel[j][i];
        }
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel[j][i] /= sum;
        }
    }

    return kernel;
}

}
}