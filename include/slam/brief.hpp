#pragma once

#include <array>
#include <bitset>
#include <random>

#include "base/matrix.hpp"
#include "graphics/color.hpp"
#include "base/vec.hpp"

namespace lm {

template <unsigned N>
class brief
{

    unsigned radius;
    std::array<std::pair<dim2, dim2>, N> net;

public:

    struct descriptor : public std::bitset<N>
    {};

    brief(unsigned radius)
    :   radius(radius)
    {

        std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count());
        std::binomial_distribution<int> distribution(radius * 2, 0.5);

        for (auto& pair : net)
        {
            pair.first.x() = distribution(engine) - radius;
            pair.first.y() = distribution(engine) - radius;

            pair.second.x() = distribution(engine) - radius;
            pair.second.y() = distribution(engine) - radius;
        }
    }

    descriptor
    operator()(const matrix<gray> &frame, unsigned x, unsigned y) const
    {
        if (x < radius or y < radius or x >= frame.width() - radius or y >= frame.height() - radius) return {};

        descriptor desc;

        for (unsigned i = 0; i < net.size(); ++i)
        {
            const auto& pair = net[i];
            desc[i] = frame[y + pair.first.y()][x + pair.first.x()] < frame[y + pair.second.y()][x + pair.second.x()];
        }

        return desc;   
    }

    static
    unsigned
    distance(const descriptor& d1, const descriptor& d2)
    {
        return (d1 ^ d2).count();
    }
};

}
