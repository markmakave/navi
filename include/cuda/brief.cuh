#pragma once

namespace lm {
namespace cuda {

struct point_pair
{
    int x1, x2, y1, y2;
};

template <unsigned N>
class brief
{
public:



private:

    array<point_pair, N> net;

};

}
}
