#include "base/matrix.hpp"
#include "base/color.hpp"

namespace lm {

static
bool
fast11(const lm::gray *p, int origin, int t);

void
detect(const matrix<lm::gray>& input, matrix<bool>& output)
{
    #pragma omp parallel for
    for (unsigned y = 3; y < input.height() - 3; ++y)
    {
        for (unsigned x = 3; x < input.width() - 3; ++x)
        {
            lm::gray circle[16] = {
                input[y - 3][x], input[y - 3][x + 1], input[y - 2][x + 2], input[y - 1][x + 3],
                input[y][x + 3], input[y + 1][x + 3], input[y + 2][x + 2], input[y + 3][x + 1],
                input[y + 3][x], input[y + 3][x - 1], input[y + 2][x - 2], input[y + 1][x - 3],
                input[y][x - 3], input[y - 1][x - 3], input[y - 2][x - 2], input[y - 3][x - 1]
            };

            output[y][x] = fast11(circle, input[y][x], 1);
        }
    }
}

static
bool
fast11(const lm::gray *p, int origin, int t)
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
