#include "slam/fast.hpp"

namespace lumina::slam {

template <>
bool fast<11>(lumina::gray center, const int circle[16], int bias)
{
    int high = center + bias;
    int low = center - bias;

    if (circle[0] > high)
        if (circle[1] > high)
            if (circle[2] > high)
                if (circle[3] > high)
                    if (circle[4] > high)
                        if (circle[5] > high)
                            if (circle[6] > high)
                                if (circle[7] > high)
                                    if (circle[8] > high)
                                        if (circle[9] > high)
                                            if (circle[10] > high)
                                                return true;
                                            else if (circle[15] > high)
                                                return true;
                                            else
                                                return false;
                                        else if (circle[14] > high)
                                            if (circle[15] > high)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else if (circle[13] > high)
                                        if (circle[14] > high)
                                            if (circle[15] > high)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else if (circle[12] > high)
                                    if (circle[13] > high)
                                        if (circle[14] > high)
                                            if (circle[15] > high)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else if (circle[11] > high)
                                if (circle[12] > high)
                                    if (circle[13] > high)
                                        if (circle[14] > high)
                                            if (circle[15] > high)
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
                        else if (circle[5] < low)
                            if (circle[10] > high)
                                if (circle[11] > high)
                                    if (circle[12] > high)
                                        if (circle[13] > high)
                                            if (circle[14] > high)
                                                if (circle[15] > high)
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
                            else if (circle[10] < low)
                                if (circle[6] < low)
                                    if (circle[7] < low)
                                        if (circle[8] < low)
                                            if (circle[9] < low)
                                                if (circle[11] < low)
                                                    if (circle[12] < low)
                                                        if (circle[13] < low)
                                                            if (circle[14] < low)
                                                                if (circle[15] < low)
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
                        else if (circle[10] > high)
                            if (circle[11] > high)
                                if (circle[12] > high)
                                    if (circle[13] > high)
                                        if (circle[14] > high)
                                            if (circle[15] > high)
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
                    else if (circle[4] < low)
                        if (circle[15] > high)
                            if (circle[9] > high)
                                if (circle[10] > high)
                                    if (circle[11] > high)
                                        if (circle[12] > high)
                                            if (circle[13] > high)
                                                if (circle[14] > high)
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
                            else if (circle[9] < low)
                                if (circle[5] < low)
                                    if (circle[6] < low)
                                        if (circle[7] < low)
                                            if (circle[8] < low)
                                                if (circle[10] < low)
                                                    if (circle[11] < low)
                                                        if (circle[12] < low)
                                                            if (circle[13] < low)
                                                                if (circle[14] < low)
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
                        else if (circle[5] < low)
                            if (circle[6] < low)
                                if (circle[7] < low)
                                    if (circle[8] < low)
                                        if (circle[9] < low)
                                            if (circle[10] < low)
                                                if (circle[11] < low)
                                                    if (circle[12] < low)
                                                        if (circle[13] < low)
                                                            if (circle[14] < low)
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
                    else if (circle[9] > high)
                        if (circle[10] > high)
                            if (circle[11] > high)
                                if (circle[12] > high)
                                    if (circle[13] > high)
                                        if (circle[14] > high)
                                            if (circle[15] > high)
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
                    else if (circle[9] < low)
                        if (circle[5] < low)
                            if (circle[6] < low)
                                if (circle[7] < low)
                                    if (circle[8] < low)
                                        if (circle[10] < low)
                                            if (circle[11] < low)
                                                if (circle[12] < low)
                                                    if (circle[13] < low)
                                                        if (circle[14] < low)
                                                            if (circle[15] < low)
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
                else if (circle[3] < low)
                    if (circle[14] > high)
                        if (circle[8] > high)
                            if (circle[9] > high)
                                if (circle[10] > high)
                                    if (circle[11] > high)
                                        if (circle[12] > high)
                                            if (circle[13] > high)
                                                if (circle[15] > high)
                                                    return true;
                                                else if (circle[4] > high)
                                                    if (circle[5] > high)
                                                        if (circle[6] > high)
                                                            if (circle[7] > high)
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
                        else if (circle[8] < low)
                            if (circle[4] < low)
                                if (circle[5] < low)
                                    if (circle[6] < low)
                                        if (circle[7] < low)
                                            if (circle[9] < low)
                                                if (circle[10] < low)
                                                    if (circle[11] < low)
                                                        if (circle[12] < low)
                                                            if (circle[13] < low)
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
                    else if (circle[14] < low)
                        if (circle[5] < low)
                            if (circle[6] < low)
                                if (circle[7] < low)
                                    if (circle[8] < low)
                                        if (circle[9] < low)
                                            if (circle[10] < low)
                                                if (circle[11] < low)
                                                    if (circle[12] < low)
                                                        if (circle[13] < low)
                                                            if (circle[4] < low)
                                                                return true;
                                                            else if (circle[15] < low)
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
                    else if (circle[4] < low)
                        if (circle[5] < low)
                            if (circle[6] < low)
                                if (circle[7] < low)
                                    if (circle[8] < low)
                                        if (circle[9] < low)
                                            if (circle[10] < low)
                                                if (circle[11] < low)
                                                    if (circle[12] < low)
                                                        if (circle[13] < low)
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
                else if (circle[8] > high)
                    if (circle[9] > high)
                        if (circle[10] > high)
                            if (circle[11] > high)
                                if (circle[12] > high)
                                    if (circle[13] > high)
                                        if (circle[14] > high)
                                            if (circle[15] > high)
                                                return true;
                                            else if (circle[4] > high)
                                                if (circle[5] > high)
                                                    if (circle[6] > high)
                                                        if (circle[7] > high)
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
                else if (circle[8] < low)
                    if (circle[5] < low)
                        if (circle[6] < low)
                            if (circle[7] < low)
                                if (circle[9] < low)
                                    if (circle[10] < low)
                                        if (circle[11] < low)
                                            if (circle[12] < low)
                                                if (circle[13] < low)
                                                    if (circle[14] < low)
                                                        if (circle[4] < low)
                                                            return true;
                                                        else if (circle[15] < low)
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
            else if (circle[2] < low)
                if (circle[7] > high)
                    if (circle[8] > high)
                        if (circle[9] > high)
                            if (circle[10] > high)
                                if (circle[11] > high)
                                    if (circle[12] > high)
                                        if (circle[13] > high)
                                            if (circle[14] > high)
                                                if (circle[15] > high)
                                                    return true;
                                                else if (circle[4] > high)
                                                    if (circle[5] > high)
                                                        if (circle[6] > high)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else if (circle[3] > high)
                                                if (circle[4] > high)
                                                    if (circle[5] > high)
                                                        if (circle[6] > high)
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
                else if (circle[7] < low)
                    if (circle[5] < low)
                        if (circle[6] < low)
                            if (circle[8] < low)
                                if (circle[9] < low)
                                    if (circle[10] < low)
                                        if (circle[11] < low)
                                            if (circle[12] < low)
                                                if (circle[4] < low)
                                                    if (circle[3] < low)
                                                        return true;
                                                    else if (circle[13] < low)
                                                        if (circle[14] < low)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else if (circle[13] < low)
                                                    if (circle[14] < low)
                                                        if (circle[15] < low)
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
            else if (circle[7] > high)
                if (circle[8] > high)
                    if (circle[9] > high)
                        if (circle[10] > high)
                            if (circle[11] > high)
                                if (circle[12] > high)
                                    if (circle[13] > high)
                                        if (circle[14] > high)
                                            if (circle[15] > high)
                                                return true;
                                            else if (circle[4] > high)
                                                if (circle[5] > high)
                                                    if (circle[6] > high)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (circle[3] > high)
                                            if (circle[4] > high)
                                                if (circle[5] > high)
                                                    if (circle[6] > high)
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
            else if (circle[7] < low)
                if (circle[5] < low)
                    if (circle[6] < low)
                        if (circle[8] < low)
                            if (circle[9] < low)
                                if (circle[10] < low)
                                    if (circle[11] < low)
                                        if (circle[12] < low)
                                            if (circle[13] < low)
                                                if (circle[4] < low)
                                                    if (circle[3] < low)
                                                        return true;
                                                    else if (circle[14] < low)
                                                        return true;
                                                    else
                                                        return false;
                                                else if (circle[14] < low)
                                                    if (circle[15] < low)
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
        else if (circle[1] < low)
            if (circle[6] > high)
                if (circle[7] > high)
                    if (circle[8] > high)
                        if (circle[9] > high)
                            if (circle[10] > high)
                                if (circle[11] > high)
                                    if (circle[12] > high)
                                        if (circle[13] > high)
                                            if (circle[14] > high)
                                                if (circle[15] > high)
                                                    return true;
                                                else if (circle[4] > high)
                                                    if (circle[5] > high)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else if (circle[3] > high)
                                                if (circle[4] > high)
                                                    if (circle[5] > high)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (circle[2] > high)
                                            if (circle[3] > high)
                                                if (circle[4] > high)
                                                    if (circle[5] > high)
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
            else if (circle[6] < low)
                if (circle[5] < low)
                    if (circle[7] < low)
                        if (circle[8] < low)
                            if (circle[9] < low)
                                if (circle[10] < low)
                                    if (circle[11] < low)
                                        if (circle[4] < low)
                                            if (circle[3] < low)
                                                if (circle[2] < low)
                                                    return true;
                                                else if (circle[12] < low)
                                                    if (circle[13] < low)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else if (circle[12] < low)
                                                if (circle[13] < low)
                                                    if (circle[14] < low)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (circle[12] < low)
                                            if (circle[13] < low)
                                                if (circle[14] < low)
                                                    if (circle[15] < low)
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
        else if (circle[6] > high)
            if (circle[7] > high)
                if (circle[8] > high)
                    if (circle[9] > high)
                        if (circle[10] > high)
                            if (circle[11] > high)
                                if (circle[12] > high)
                                    if (circle[13] > high)
                                        if (circle[14] > high)
                                            if (circle[15] > high)
                                                return true;
                                            else if (circle[4] > high)
                                                if (circle[5] > high)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (circle[3] > high)
                                            if (circle[4] > high)
                                                if (circle[5] > high)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else if (circle[2] > high)
                                        if (circle[3] > high)
                                            if (circle[4] > high)
                                                if (circle[5] > high)
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
        else if (circle[6] < low)
            if (circle[5] < low)
                if (circle[7] < low)
                    if (circle[8] < low)
                        if (circle[9] < low)
                            if (circle[10] < low)
                                if (circle[11] < low)
                                    if (circle[12] < low)
                                        if (circle[4] < low)
                                            if (circle[3] < low)
                                                if (circle[2] < low)
                                                    return true;
                                                else if (circle[13] < low)
                                                    return true;
                                                else
                                                    return false;
                                            else if (circle[13] < low)
                                                if (circle[14] < low)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (circle[13] < low)
                                            if (circle[14] < low)
                                                if (circle[15] < low)
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
    else if (circle[0] < low)
        if (circle[1] > high)
            if (circle[6] > high)
                if (circle[5] > high)
                    if (circle[7] > high)
                        if (circle[8] > high)
                            if (circle[9] > high)
                                if (circle[10] > high)
                                    if (circle[11] > high)
                                        if (circle[4] > high)
                                            if (circle[3] > high)
                                                if (circle[2] > high)
                                                    return true;
                                                else if (circle[12] > high)
                                                    if (circle[13] > high)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else if (circle[12] > high)
                                                if (circle[13] > high)
                                                    if (circle[14] > high)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (circle[12] > high)
                                            if (circle[13] > high)
                                                if (circle[14] > high)
                                                    if (circle[15] > high)
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
            else if (circle[6] < low)
                if (circle[7] < low)
                    if (circle[8] < low)
                        if (circle[9] < low)
                            if (circle[10] < low)
                                if (circle[11] < low)
                                    if (circle[12] < low)
                                        if (circle[13] < low)
                                            if (circle[14] < low)
                                                if (circle[15] < low)
                                                    return true;
                                                else if (circle[4] < low)
                                                    if (circle[5] < low)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else if (circle[3] < low)
                                                if (circle[4] < low)
                                                    if (circle[5] < low)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (circle[2] < low)
                                            if (circle[3] < low)
                                                if (circle[4] < low)
                                                    if (circle[5] < low)
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
        else if (circle[1] < low)
            if (circle[2] > high)
                if (circle[7] > high)
                    if (circle[5] > high)
                        if (circle[6] > high)
                            if (circle[8] > high)
                                if (circle[9] > high)
                                    if (circle[10] > high)
                                        if (circle[11] > high)
                                            if (circle[12] > high)
                                                if (circle[4] > high)
                                                    if (circle[3] > high)
                                                        return true;
                                                    else if (circle[13] > high)
                                                        if (circle[14] > high)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else if (circle[13] > high)
                                                    if (circle[14] > high)
                                                        if (circle[15] > high)
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
                else if (circle[7] < low)
                    if (circle[8] < low)
                        if (circle[9] < low)
                            if (circle[10] < low)
                                if (circle[11] < low)
                                    if (circle[12] < low)
                                        if (circle[13] < low)
                                            if (circle[14] < low)
                                                if (circle[15] < low)
                                                    return true;
                                                else if (circle[4] < low)
                                                    if (circle[5] < low)
                                                        if (circle[6] < low)
                                                            return true;
                                                        else
                                                            return false;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else if (circle[3] < low)
                                                if (circle[4] < low)
                                                    if (circle[5] < low)
                                                        if (circle[6] < low)
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
            else if (circle[2] < low)
                if (circle[3] > high)
                    if (circle[14] > high)
                        if (circle[5] > high)
                            if (circle[6] > high)
                                if (circle[7] > high)
                                    if (circle[8] > high)
                                        if (circle[9] > high)
                                            if (circle[10] > high)
                                                if (circle[11] > high)
                                                    if (circle[12] > high)
                                                        if (circle[13] > high)
                                                            if (circle[4] > high)
                                                                return true;
                                                            else if (circle[15] > high)
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
                    else if (circle[14] < low)
                        if (circle[8] > high)
                            if (circle[4] > high)
                                if (circle[5] > high)
                                    if (circle[6] > high)
                                        if (circle[7] > high)
                                            if (circle[9] > high)
                                                if (circle[10] > high)
                                                    if (circle[11] > high)
                                                        if (circle[12] > high)
                                                            if (circle[13] > high)
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
                        else if (circle[8] < low)
                            if (circle[9] < low)
                                if (circle[10] < low)
                                    if (circle[11] < low)
                                        if (circle[12] < low)
                                            if (circle[13] < low)
                                                if (circle[15] < low)
                                                    return true;
                                                else if (circle[4] < low)
                                                    if (circle[5] < low)
                                                        if (circle[6] < low)
                                                            if (circle[7] < low)
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
                    else if (circle[4] > high)
                        if (circle[5] > high)
                            if (circle[6] > high)
                                if (circle[7] > high)
                                    if (circle[8] > high)
                                        if (circle[9] > high)
                                            if (circle[10] > high)
                                                if (circle[11] > high)
                                                    if (circle[12] > high)
                                                        if (circle[13] > high)
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
                else if (circle[3] < low)
                    if (circle[4] > high)
                        if (circle[15] < low)
                            if (circle[9] > high)
                                if (circle[5] > high)
                                    if (circle[6] > high)
                                        if (circle[7] > high)
                                            if (circle[8] > high)
                                                if (circle[10] > high)
                                                    if (circle[11] > high)
                                                        if (circle[12] > high)
                                                            if (circle[13] > high)
                                                                if (circle[14] > high)
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
                            else if (circle[9] < low)
                                if (circle[10] < low)
                                    if (circle[11] < low)
                                        if (circle[12] < low)
                                            if (circle[13] < low)
                                                if (circle[14] < low)
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
                        else if (circle[5] > high)
                            if (circle[6] > high)
                                if (circle[7] > high)
                                    if (circle[8] > high)
                                        if (circle[9] > high)
                                            if (circle[10] > high)
                                                if (circle[11] > high)
                                                    if (circle[12] > high)
                                                        if (circle[13] > high)
                                                            if (circle[14] > high)
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
                    else if (circle[4] < low)
                        if (circle[5] > high)
                            if (circle[10] > high)
                                if (circle[6] > high)
                                    if (circle[7] > high)
                                        if (circle[8] > high)
                                            if (circle[9] > high)
                                                if (circle[11] > high)
                                                    if (circle[12] > high)
                                                        if (circle[13] > high)
                                                            if (circle[14] > high)
                                                                if (circle[15] > high)
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
                            else if (circle[10] < low)
                                if (circle[11] < low)
                                    if (circle[12] < low)
                                        if (circle[13] < low)
                                            if (circle[14] < low)
                                                if (circle[15] < low)
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
                        else if (circle[5] < low)
                            if (circle[6] < low)
                                if (circle[7] < low)
                                    if (circle[8] < low)
                                        if (circle[9] < low)
                                            if (circle[10] < low)
                                                return true;
                                            else if (circle[15] < low)
                                                return true;
                                            else
                                                return false;
                                        else if (circle[14] < low)
                                            if (circle[15] < low)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else if (circle[13] < low)
                                        if (circle[14] < low)
                                            if (circle[15] < low)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else if (circle[12] < low)
                                    if (circle[13] < low)
                                        if (circle[14] < low)
                                            if (circle[15] < low)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else
                                    return false;
                            else if (circle[11] < low)
                                if (circle[12] < low)
                                    if (circle[13] < low)
                                        if (circle[14] < low)
                                            if (circle[15] < low)
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
                        else if (circle[10] < low)
                            if (circle[11] < low)
                                if (circle[12] < low)
                                    if (circle[13] < low)
                                        if (circle[14] < low)
                                            if (circle[15] < low)
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
                    else if (circle[9] > high)
                        if (circle[5] > high)
                            if (circle[6] > high)
                                if (circle[7] > high)
                                    if (circle[8] > high)
                                        if (circle[10] > high)
                                            if (circle[11] > high)
                                                if (circle[12] > high)
                                                    if (circle[13] > high)
                                                        if (circle[14] > high)
                                                            if (circle[15] > high)
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
                    else if (circle[9] < low)
                        if (circle[10] < low)
                            if (circle[11] < low)
                                if (circle[12] < low)
                                    if (circle[13] < low)
                                        if (circle[14] < low)
                                            if (circle[15] < low)
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
                else if (circle[8] > high)
                    if (circle[5] > high)
                        if (circle[6] > high)
                            if (circle[7] > high)
                                if (circle[9] > high)
                                    if (circle[10] > high)
                                        if (circle[11] > high)
                                            if (circle[12] > high)
                                                if (circle[13] > high)
                                                    if (circle[14] > high)
                                                        if (circle[4] > high)
                                                            return true;
                                                        else if (circle[15] > high)
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
                else if (circle[8] < low)
                    if (circle[9] < low)
                        if (circle[10] < low)
                            if (circle[11] < low)
                                if (circle[12] < low)
                                    if (circle[13] < low)
                                        if (circle[14] < low)
                                            if (circle[15] < low)
                                                return true;
                                            else if (circle[4] < low)
                                                if (circle[5] < low)
                                                    if (circle[6] < low)
                                                        if (circle[7] < low)
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
            else if (circle[7] > high)
                if (circle[5] > high)
                    if (circle[6] > high)
                        if (circle[8] > high)
                            if (circle[9] > high)
                                if (circle[10] > high)
                                    if (circle[11] > high)
                                        if (circle[12] > high)
                                            if (circle[13] > high)
                                                if (circle[4] > high)
                                                    if (circle[3] > high)
                                                        return true;
                                                    else if (circle[14] > high)
                                                        return true;
                                                    else
                                                        return false;
                                                else if (circle[14] > high)
                                                    if (circle[15] > high)
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
            else if (circle[7] < low)
                if (circle[8] < low)
                    if (circle[9] < low)
                        if (circle[10] < low)
                            if (circle[11] < low)
                                if (circle[12] < low)
                                    if (circle[13] < low)
                                        if (circle[14] < low)
                                            if (circle[15] < low)
                                                return true;
                                            else if (circle[4] < low)
                                                if (circle[5] < low)
                                                    if (circle[6] < low)
                                                        return true;
                                                    else
                                                        return false;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (circle[3] < low)
                                            if (circle[4] < low)
                                                if (circle[5] < low)
                                                    if (circle[6] < low)
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
        else if (circle[6] > high)
            if (circle[5] > high)
                if (circle[7] > high)
                    if (circle[8] > high)
                        if (circle[9] > high)
                            if (circle[10] > high)
                                if (circle[11] > high)
                                    if (circle[12] > high)
                                        if (circle[4] > high)
                                            if (circle[3] > high)
                                                if (circle[2] > high)
                                                    return true;
                                                else if (circle[13] > high)
                                                    return true;
                                                else
                                                    return false;
                                            else if (circle[13] > high)
                                                if (circle[14] > high)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (circle[13] > high)
                                            if (circle[14] > high)
                                                if (circle[15] > high)
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
        else if (circle[6] < low)
            if (circle[7] < low)
                if (circle[8] < low)
                    if (circle[9] < low)
                        if (circle[10] < low)
                            if (circle[11] < low)
                                if (circle[12] < low)
                                    if (circle[13] < low)
                                        if (circle[14] < low)
                                            if (circle[15] < low)
                                                return true;
                                            else if (circle[4] < low)
                                                if (circle[5] < low)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else if (circle[3] < low)
                                            if (circle[4] < low)
                                                if (circle[5] < low)
                                                    return true;
                                                else
                                                    return false;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else if (circle[2] < low)
                                        if (circle[3] < low)
                                            if (circle[4] < low)
                                                if (circle[5] < low)
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
    else if (circle[5] > high)
        if (circle[6] > high)
            if (circle[7] > high)
                if (circle[8] > high)
                    if (circle[9] > high)
                        if (circle[10] > high)
                            if (circle[11] > high)
                                if (circle[4] > high)
                                    if (circle[3] > high)
                                        if (circle[2] > high)
                                            if (circle[1] > high)
                                                return true;
                                            else if (circle[12] > high)
                                                return true;
                                            else
                                                return false;
                                        else if (circle[12] > high)
                                            if (circle[13] > high)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else if (circle[12] > high)
                                        if (circle[13] > high)
                                            if (circle[14] > high)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else if (circle[12] > high)
                                    if (circle[13] > high)
                                        if (circle[14] > high)
                                            if (circle[15] > high)
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
    else if (circle[5] < low)
        if (circle[6] < low)
            if (circle[7] < low)
                if (circle[8] < low)
                    if (circle[9] < low)
                        if (circle[10] < low)
                            if (circle[11] < low)
                                if (circle[4] < low)
                                    if (circle[3] < low)
                                        if (circle[2] < low)
                                            if (circle[1] < low)
                                                return true;
                                            else if (circle[12] < low)
                                                return true;
                                            else
                                                return false;
                                        else if (circle[12] < low)
                                            if (circle[13] < low)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else if (circle[12] < low)
                                        if (circle[13] < low)
                                            if (circle[14] < low)
                                                return true;
                                            else
                                                return false;
                                        else
                                            return false;
                                    else
                                        return false;
                                else if (circle[12] < low)
                                    if (circle[13] < low)
                                        if (circle[14] < low)
                                            if (circle[15] < low)
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
}

}
