/* 

    Copyright (c) 2023 Mokhov Mark

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

*/

#include <cuda_runtime.h>

#include "cuda/matrix.cuh"
#include "base/color.hpp"

namespace lumina {
namespace cuda {

static
__device__
bool
fast11(const gray *p, int origin, int t);

__global__
void
detect(
    const cuda::matrix<gray> image,
    const int                threshold,
          cuda::matrix<bool> features
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= image.shape()[0] - 3 || y >= image.shape()[1] - 3 || x < 3 || y < 3)
        return;

    gray circle[16] = {
        image(x, y - 3), image(x + 1, y - 3), image(x + 2, y - 2), image(x + 3, y - 1),
        image(x + 3, y), image(x + 3, y + 1), image(x + 2, y + 2), image(x + 1, y + 3),
        image(x, y + 3), image(x - 1, y + 3), image(x - 2, y + 2), image(x - 3, y + 1),
        image(x - 3, y), image(x - 3, y - 1), image(x - 2, y - 2), image(x - 1, y - 3)
    };

    if (fast11(circle, image(x, y), threshold))
        features(x, y) = true;
}

static
__device__
bool
fast11(const gray *p, int origin, int t)
{
    int bright = origin + t;
    int dark = origin - t;

    if (p[0] > bright)
        if (p[1] > bright)
            if (p[2] > bright)
                if (p[3] > bright)
                    if (p[4] > bright)
                        if (p[5] > bright)
                            if (p[6] > bright)
                                if (p[7] > bright)
                                    if (p[8] > bright)
                                        if (p[9] > bright)
                                            if (p[10] > bright)
                                                return true;
                                            else if (p[15] > bright)
                                                return true;
                                            else
                                                return false;
                                        else if (p[14] > bright)
                                            if (p[15] > bright)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else if (p[13] > bright)
                                        if (p[14] > bright)
                                            if (p[15] > bright)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else if (p[12] > bright)
                                    if (p[13] > bright)
                                        if (p[14] > bright)
                                            if (p[15] > bright)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else if (p[11] > bright)
                                if (p[12] > bright)
                                    if (p[13] > bright)
                                        if (p[14] > bright)
                                            if (p[15] > bright)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else if (p[5] < dark)
                            if (p[10] > bright)
                                if (p[11] > bright)
                                    if (p[12] > bright)
                                        if (p[13] > bright)
                                            if (p[14] > bright)
                                                if (p[15] > bright)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else if (p[10] < dark)
                                if (p[6] < dark)
                                    if (p[7] < dark)
                                        if (p[8] < dark)
                                            if (p[9] < dark)
                                                if (p[11] < dark)
                                                    if (p[12] < dark)
                                                        if (p[13] < dark)
                                                            if (p[14] < dark)
                                                                if (p[15] < dark)
                                                                    return true;
                                                                else
                                                                    return false;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else if (p[10] > bright)
                            if (p[11] > bright)
                                if (p[12] > bright)
                                    if (p[13] > bright)
                                        if (p[14] > bright)
                                            if (p[15] > bright)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else if (p[4] < dark)
                        if (p[15] > bright)
                            if (p[9] > bright)
                                if (p[10] > bright)
                                    if (p[11] > bright)
                                        if (p[12] > bright)
                                            if (p[13] > bright)
                                                if (p[14] > bright)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else if (p[9] < dark)
                                if (p[5] < dark)
                                    if (p[6] < dark)
                                        if (p[7] < dark)
                                            if (p[8] < dark)
                                                if (p[10] < dark)
                                                    if (p[11] < dark)
                                                        if (p[12] < dark)
                                                            if (p[13] < dark)
                                                                if (p[14] < dark)
                                                                    return true;
                                                                else
                                                                    return false;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else if (p[5] < dark)
                            if (p[6] < dark)
                                if (p[7] < dark)
                                    if (p[8] < dark)
                                        if (p[9] < dark)
                                            if (p[10] < dark)
                                                if (p[11] < dark)
                                                    if (p[12] < dark)
                                                        if (p[13] < dark)
                                                            if (p[14] < dark)
                                                                return true;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else if (p[9] > bright)
                        if (p[10] > bright)
                            if (p[11] > bright)
                                if (p[12] > bright)
                                    if (p[13] > bright)
                                        if (p[14] > bright)
                                            if (p[15] > bright)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else if (p[9] < dark)
                        if (p[5] < dark)
                            if (p[6] < dark)
                                if (p[7] < dark)
                                    if (p[8] < dark)
                                        if (p[10] < dark)
                                            if (p[11] < dark)
                                                if (p[12] < dark)
                                                    if (p[13] < dark)
                                                        if (p[14] < dark)
                                                            if (p[15] < dark)
                                                                return true;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else if (p[3] < dark)
                    if (p[14] > bright)
                        if (p[8] > bright)
                            if (p[9] > bright)
                                if (p[10] > bright)
                                    if (p[11] > bright)
                                        if (p[12] > bright)
                                            if (p[13] > bright)
                                                if (p[15] > bright)
                                                    return true;
                                                else if (p[4] > bright)
                                                    if (p[5] > bright)
                                                        if (p[6] > bright)
                                                            if (p[7] > bright)
                                                                return true;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else if (p[8] < dark)
                            if (p[4] < dark)
                                if (p[5] < dark)
                                    if (p[6] < dark)
                                        if (p[7] < dark)
                                            if (p[9] < dark)
                                                if (p[10] < dark)
                                                    if (p[11] < dark)
                                                        if (p[12] < dark)
                                                            if (p[13] < dark)
                                                                return true;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else if (p[14] < dark)
                        if (p[5] < dark)
                            if (p[6] < dark)
                                if (p[7] < dark)
                                    if (p[8] < dark)
                                        if (p[9] < dark)
                                            if (p[10] < dark)
                                                if (p[11] < dark)
                                                    if (p[12] < dark)
                                                        if (p[13] < dark)
                                                            if (p[4] < dark)
                                                                return true;
                                                            else if (p[15] < dark)
                                                                return true;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else if (p[4] < dark)
                        if (p[5] < dark)
                            if (p[6] < dark)
                                if (p[7] < dark)
                                    if (p[8] < dark)
                                        if (p[9] < dark)
                                            if (p[10] < dark)
                                                if (p[11] < dark)
                                                    if (p[12] < dark)
                                                        if (p[13] < dark)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else if (p[8] > bright)
                    if (p[9] > bright)
                        if (p[10] > bright)
                            if (p[11] > bright)
                                if (p[12] > bright)
                                    if (p[13] > bright)
                                        if (p[14] > bright)
                                            if (p[15] > bright)
                                                return true;
                                            else if (p[4] > bright)
                                                if (p[5] > bright)
                                                    if (p[6] > bright)
                                                        if (p[7] > bright)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else if (p[8] < dark)
                    if (p[5] < dark)
                        if (p[6] < dark)
                            if (p[7] < dark)
                                if (p[9] < dark)
                                    if (p[10] < dark)
                                        if (p[11] < dark)
                                            if (p[12] < dark)
                                                if (p[13] < dark)
                                                    if (p[14] < dark)
                                                        if (p[4] < dark)
                                                            return true;
                                                        else if (p[15] < dark)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else if (p[2] < dark)
                if (p[7] > bright)
                    if (p[8] > bright)
                        if (p[9] > bright)
                            if (p[10] > bright)
                                if (p[11] > bright)
                                    if (p[12] > bright)
                                        if (p[13] > bright)
                                            if (p[14] > bright)
                                                if (p[15] > bright)
                                                    return true;
                                                else if (p[4] > bright)
                                                    if (p[5] > bright)
                                                        if (p[6] > bright)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else if (p[3] > bright)
                                                if (p[4] > bright)
                                                    if (p[5] > bright)
                                                        if (p[6] > bright)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else if (p[7] < dark)
                    if (p[5] < dark)
                        if (p[6] < dark)
                            if (p[8] < dark)
                                if (p[9] < dark)
                                    if (p[10] < dark)
                                        if (p[11] < dark)
                                            if (p[12] < dark)
                                                if (p[4] < dark)
                                                    if (p[3] < dark)
                                                        return true;
                                                    else if (p[13] < dark)
                                                        if (p[14] < dark)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else if (p[13] < dark)
                                                    if (p[14] < dark)
                                                        if (p[15] < dark)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else if (p[7] > bright)
                if (p[8] > bright)
                    if (p[9] > bright)
                        if (p[10] > bright)
                            if (p[11] > bright)
                                if (p[12] > bright)
                                    if (p[13] > bright)
                                        if (p[14] > bright)
                                            if (p[15] > bright)
                                                return true;
                                            else if (p[4] > bright)
                                                if (p[5] > bright)
                                                    if (p[6] > bright)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (p[3] > bright)
                                            if (p[4] > bright)
                                                if (p[5] > bright)
                                                    if (p[6] > bright)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else if (p[7] < dark)
                if (p[5] < dark)
                    if (p[6] < dark)
                        if (p[8] < dark)
                            if (p[9] < dark)
                                if (p[10] < dark)
                                    if (p[11] < dark)
                                        if (p[12] < dark)
                                            if (p[13] < dark)
                                                if (p[4] < dark)
                                                    if (p[3] < dark)
                                                        return true;
                                                    else if (p[14] < dark)
                                                        return true;
                                                    else
                                                        return false;
                                                else if (p[14] < dark)
                                                    if (p[15] < dark)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else
                return false;
        else if (p[1] < dark)
            if (p[6] > bright)
                if (p[7] > bright)
                    if (p[8] > bright)
                        if (p[9] > bright)
                            if (p[10] > bright)
                                if (p[11] > bright)
                                    if (p[12] > bright)
                                        if (p[13] > bright)
                                            if (p[14] > bright)
                                                if (p[15] > bright)
                                                    return true;
                                                else if (p[4] > bright)
                                                    if (p[5] > bright)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else if (p[3] > bright)
                                                if (p[4] > bright)
                                                    if (p[5] > bright)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (p[2] > bright)
                                            if (p[3] > bright)
                                                if (p[4] > bright)
                                                    if (p[5] > bright)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else if (p[6] < dark)
                if (p[5] < dark)
                    if (p[7] < dark)
                        if (p[8] < dark)
                            if (p[9] < dark)
                                if (p[10] < dark)
                                    if (p[11] < dark)
                                        if (p[4] < dark)
                                            if (p[3] < dark)
                                                if (p[2] < dark)
                                                    return true;
                                                else if (p[12] < dark)
                                                    if (p[13] < dark)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else if (p[12] < dark)
                                                if (p[13] < dark)
                                                    if (p[14] < dark)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (p[12] < dark)
                                            if (p[13] < dark)
                                                if (p[14] < dark)
                                                    if (p[15] < dark)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else
                return false;
        else if (p[6] > bright)
            if (p[7] > bright)
                if (p[8] > bright)
                    if (p[9] > bright)
                        if (p[10] > bright)
                            if (p[11] > bright)
                                if (p[12] > bright)
                                    if (p[13] > bright)
                                        if (p[14] > bright)
                                            if (p[15] > bright)
                                                return true;
                                            else if (p[4] > bright)
                                                if (p[5] > bright)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (p[3] > bright)
                                            if (p[4] > bright)
                                                if (p[5] > bright)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else if (p[2] > bright)
                                        if (p[3] > bright)
                                            if (p[4] > bright)
                                                if (p[5] > bright)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else
                return false;
        else if (p[6] < dark)
            if (p[5] < dark)
                if (p[7] < dark)
                    if (p[8] < dark)
                        if (p[9] < dark)
                            if (p[10] < dark)
                                if (p[11] < dark)
                                    if (p[12] < dark)
                                        if (p[4] < dark)
                                            if (p[3] < dark)
                                                if (p[2] < dark)
                                                    return true;
                                                else if (p[13] < dark)
                                                    return true;
                                                else
                                                    return false;
                                            else if (p[13] < dark)
                                                if (p[14] < dark)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (p[13] < dark)
                                            if (p[14] < dark)
                                                if (p[15] < dark)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else
                return false;
        else
            return false;
    else if (p[0] < dark)
        if (p[1] > bright)
            if (p[6] > bright)
                if (p[5] > bright)
                    if (p[7] > bright)
                        if (p[8] > bright)
                            if (p[9] > bright)
                                if (p[10] > bright)
                                    if (p[11] > bright)
                                        if (p[4] > bright)
                                            if (p[3] > bright)
                                                if (p[2] > bright)
                                                    return true;
                                                else if (p[12] > bright)
                                                    if (p[13] > bright)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else if (p[12] > bright)
                                                if (p[13] > bright)
                                                    if (p[14] > bright)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (p[12] > bright)
                                            if (p[13] > bright)
                                                if (p[14] > bright)
                                                    if (p[15] > bright)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else if (p[6] < dark)
                if (p[7] < dark)
                    if (p[8] < dark)
                        if (p[9] < dark)
                            if (p[10] < dark)
                                if (p[11] < dark)
                                    if (p[12] < dark)
                                        if (p[13] < dark)
                                            if (p[14] < dark)
                                                if (p[15] < dark)
                                                    return true;
                                                else if (p[4] < dark)
                                                    if (p[5] < dark)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else if (p[3] < dark)
                                                if (p[4] < dark)
                                                    if (p[5] < dark)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (p[2] < dark)
                                            if (p[3] < dark)
                                                if (p[4] < dark)
                                                    if (p[5] < dark)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else
                return false;
        else if (p[1] < dark)
            if (p[2] > bright)
                if (p[7] > bright)
                    if (p[5] > bright)
                        if (p[6] > bright)
                            if (p[8] > bright)
                                if (p[9] > bright)
                                    if (p[10] > bright)
                                        if (p[11] > bright)
                                            if (p[12] > bright)
                                                if (p[4] > bright)
                                                    if (p[3] > bright)
                                                        return true;
                                                    else if (p[13] > bright)
                                                        if (p[14] > bright)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else if (p[13] > bright)
                                                    if (p[14] > bright)
                                                        if (p[15] > bright)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else if (p[7] < dark)
                    if (p[8] < dark)
                        if (p[9] < dark)
                            if (p[10] < dark)
                                if (p[11] < dark)
                                    if (p[12] < dark)
                                        if (p[13] < dark)
                                            if (p[14] < dark)
                                                if (p[15] < dark)
                                                    return true;
                                                else if (p[4] < dark)
                                                    if (p[5] < dark)
                                                        if (p[6] < dark)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else if (p[3] < dark)
                                                if (p[4] < dark)
                                                    if (p[5] < dark)
                                                        if (p[6] < dark)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else if (p[2] < dark)
                if (p[3] > bright)
                    if (p[14] > bright)
                        if (p[5] > bright)
                            if (p[6] > bright)
                                if (p[7] > bright)
                                    if (p[8] > bright)
                                        if (p[9] > bright)
                                            if (p[10] > bright)
                                                if (p[11] > bright)
                                                    if (p[12] > bright)
                                                        if (p[13] > bright)
                                                            if (p[4] > bright)
                                                                return true;
                                                            else if (p[15] > bright)
                                                                return true;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else if (p[14] < dark)
                        if (p[8] > bright)
                            if (p[4] > bright)
                                if (p[5] > bright)
                                    if (p[6] > bright)
                                        if (p[7] > bright)
                                            if (p[9] > bright)
                                                if (p[10] > bright)
                                                    if (p[11] > bright)
                                                        if (p[12] > bright)
                                                            if (p[13] > bright)
                                                                return true;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else if (p[8] < dark)
                            if (p[9] < dark)
                                if (p[10] < dark)
                                    if (p[11] < dark)
                                        if (p[12] < dark)
                                            if (p[13] < dark)
                                                if (p[15] < dark)
                                                    return true;
                                                else if (p[4] < dark)
                                                    if (p[5] < dark)
                                                        if (p[6] < dark)
                                                            if (p[7] < dark)
                                                                return true;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else if (p[4] > bright)
                        if (p[5] > bright)
                            if (p[6] > bright)
                                if (p[7] > bright)
                                    if (p[8] > bright)
                                        if (p[9] > bright)
                                            if (p[10] > bright)
                                                if (p[11] > bright)
                                                    if (p[12] > bright)
                                                        if (p[13] > bright)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else if (p[3] < dark)
                    if (p[4] > bright)
                        if (p[15] < dark)
                            if (p[9] > bright)
                                if (p[5] > bright)
                                    if (p[6] > bright)
                                        if (p[7] > bright)
                                            if (p[8] > bright)
                                                if (p[10] > bright)
                                                    if (p[11] > bright)
                                                        if (p[12] > bright)
                                                            if (p[13] > bright)
                                                                if (p[14] > bright)
                                                                    return true;
                                                                else
                                                                    return false;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else if (p[9] < dark)
                                if (p[10] < dark)
                                    if (p[11] < dark)
                                        if (p[12] < dark)
                                            if (p[13] < dark)
                                                if (p[14] < dark)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else if (p[5] > bright)
                            if (p[6] > bright)
                                if (p[7] > bright)
                                    if (p[8] > bright)
                                        if (p[9] > bright)
                                            if (p[10] > bright)
                                                if (p[11] > bright)
                                                    if (p[12] > bright)
                                                        if (p[13] > bright)
                                                            if (p[14] > bright)
                                                                return true;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else if (p[4] < dark)
                        if (p[5] > bright)
                            if (p[10] > bright)
                                if (p[6] > bright)
                                    if (p[7] > bright)
                                        if (p[8] > bright)
                                            if (p[9] > bright)
                                                if (p[11] > bright)
                                                    if (p[12] > bright)
                                                        if (p[13] > bright)
                                                            if (p[14] > bright)
                                                                if (p[15] > bright)
                                                                    return true;
                                                                else
                                                                    return false;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else if (p[10] < dark)
                                if (p[11] < dark)
                                    if (p[12] < dark)
                                        if (p[13] < dark)
                                            if (p[14] < dark)
                                                if (p[15] < dark)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else if (p[5] < dark)
                            if (p[6] < dark)
                                if (p[7] < dark)
                                    if (p[8] < dark)
                                        if (p[9] < dark)
                                            if (p[10] < dark)
                                                return true;
                                            else if (p[15] < dark)
                                                return true;
                                            else
                                                return false;
                                        else if (p[14] < dark)
                                            if (p[15] < dark)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else if (p[13] < dark)
                                        if (p[14] < dark)
                                            if (p[15] < dark)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else if (p[12] < dark)
                                    if (p[13] < dark)
                                        if (p[14] < dark)
                                            if (p[15] < dark)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else if (p[11] < dark)
                                if (p[12] < dark)
                                    if (p[13] < dark)
                                        if (p[14] < dark)
                                            if (p[15] < dark)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else if (p[10] < dark)
                            if (p[11] < dark)
                                if (p[12] < dark)
                                    if (p[13] < dark)
                                        if (p[14] < dark)
                                            if (p[15] < dark)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else if (p[9] > bright)
                        if (p[5] > bright)
                            if (p[6] > bright)
                                if (p[7] > bright)
                                    if (p[8] > bright)
                                        if (p[10] > bright)
                                            if (p[11] > bright)
                                                if (p[12] > bright)
                                                    if (p[13] > bright)
                                                        if (p[14] > bright)
                                                            if (p[15] > bright)
                                                                return true;
                                                            else
                                                                return false;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else if (p[9] < dark)
                        if (p[10] < dark)
                            if (p[11] < dark)
                                if (p[12] < dark)
                                    if (p[13] < dark)
                                        if (p[14] < dark)
                                            if (p[15] < dark)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else if (p[8] > bright)
                    if (p[5] > bright)
                        if (p[6] > bright)
                            if (p[7] > bright)
                                if (p[9] > bright)
                                    if (p[10] > bright)
                                        if (p[11] > bright)
                                            if (p[12] > bright)
                                                if (p[13] > bright)
                                                    if (p[14] > bright)
                                                        if (p[4] > bright)
                                                            return true;
                                                        else if (p[15] > bright)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else if (p[8] < dark)
                    if (p[9] < dark)
                        if (p[10] < dark)
                            if (p[11] < dark)
                                if (p[12] < dark)
                                    if (p[13] < dark)
                                        if (p[14] < dark)
                                            if (p[15] < dark)
                                                return true;
                                            else if (p[4] < dark)
                                                if (p[5] < dark)
                                                    if (p[6] < dark)
                                                        if (p[7] < dark)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else if (p[7] > bright)
                if (p[5] > bright)
                    if (p[6] > bright)
                        if (p[8] > bright)
                            if (p[9] > bright)
                                if (p[10] > bright)
                                    if (p[11] > bright)
                                        if (p[12] > bright)
                                            if (p[13] > bright)
                                                if (p[4] > bright)
                                                    if (p[3] > bright)
                                                        return true;
                                                    else if (p[14] > bright)
                                                        return true;
                                                    else
                                                        return false;
                                                else if (p[14] > bright)
                                                    if (p[15] > bright)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else if (p[7] < dark)
                if (p[8] < dark)
                    if (p[9] < dark)
                        if (p[10] < dark)
                            if (p[11] < dark)
                                if (p[12] < dark)
                                    if (p[13] < dark)
                                        if (p[14] < dark)
                                            if (p[15] < dark)
                                                return true;
                                            else if (p[4] < dark)
                                                if (p[5] < dark)
                                                    if (p[6] < dark)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (p[3] < dark)
                                            if (p[4] < dark)
                                                if (p[5] < dark)
                                                    if (p[6] < dark)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else
                return false;
        else if (p[6] > bright)
            if (p[5] > bright)
                if (p[7] > bright)
                    if (p[8] > bright)
                        if (p[9] > bright)
                            if (p[10] > bright)
                                if (p[11] > bright)
                                    if (p[12] > bright)
                                        if (p[4] > bright)
                                            if (p[3] > bright)
                                                if (p[2] > bright)
                                                    return true;
                                                else if (p[13] > bright)
                                                    return true;
                                                else
                                                    return false;
                                            else if (p[13] > bright)
                                                if (p[14] > bright)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (p[13] > bright)
                                            if (p[14] > bright)
                                                if (p[15] > bright)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else
                return false;
        else if (p[6] < dark)
            if (p[7] < dark)
                if (p[8] < dark)
                    if (p[9] < dark)
                        if (p[10] < dark)
                            if (p[11] < dark)
                                if (p[12] < dark)
                                    if (p[13] < dark)
                                        if (p[14] < dark)
                                            if (p[15] < dark)
                                                return true;
                                            else if (p[4] < dark)
                                                if (p[5] < dark)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (p[3] < dark)
                                            if (p[4] < dark)
                                                if (p[5] < dark)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else if (p[2] < dark)
                                        if (p[3] < dark)
                                            if (p[4] < dark)
                                                if (p[5] < dark)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else
                return false;
        else
            return false;
    else if (p[5] > bright)
        if (p[6] > bright)
            if (p[7] > bright)
                if (p[8] > bright)
                    if (p[9] > bright)
                        if (p[10] > bright)
                            if (p[11] > bright)
                                if (p[4] > bright)
                                    if (p[3] > bright)
                                        if (p[2] > bright)
                                            if (p[1] > bright)
                                                return true;
                                            else if (p[12] > bright)
                                                return true;
                                            else
                                                return false;
                                        else if (p[12] > bright)
                                            if (p[13] > bright)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else if (p[12] > bright)
                                        if (p[13] > bright)
                                            if (p[14] > bright)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else if (p[12] > bright)
                                    if (p[13] > bright)
                                        if (p[14] > bright)
                                            if (p[15] > bright)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else
                return false;
        else
            return false;
    else if (p[5] < dark)
        if (p[6] < dark)
            if (p[7] < dark)
                if (p[8] < dark)
                    if (p[9] < dark)
                        if (p[10] < dark)
                            if (p[11] < dark)
                                if (p[4] < dark)
                                    if (p[3] < dark)
                                        if (p[2] < dark)
                                            if (p[1] < dark)
                                                return true;
                                            else if (p[12] < dark)
                                                return true;
                                            else
                                                return false;
                                        else if (p[12] < dark)
                                            if (p[13] < dark)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else if (p[12] < dark)
                                        if (p[13] < dark)
                                            if (p[14] < dark)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else if (p[12] < dark)
                                    if (p[13] < dark)
                                        if (p[14] < dark)
                                            if (p[15] < dark)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else
                                return false;
                        else
                            return false;
                    else
                        return false;
                else
                    return false;
            else
                return false;
        else
            return false;
    else
        return false;

    return false;
}

}
}
