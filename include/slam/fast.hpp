#pragma once

#include "fast11.hpp"

#include "util/log.hpp"

#include "base/array.hpp"
#include "base/image.hpp"

#include "slam/brief.hpp"
#include "slam/fast11.hpp"

namespace lm {
namespace slam {

class fast
{
public:

    fast(int threshold)
    :   _threshold(threshold)
    {
        lm::log::info("FAST initialized");
    }

    bool
    is_corner(const lm::image<lm::gray>& image, unsigned x, unsigned y)
    {
        lm::gray center = image[y][x];
        lm::gray circle[16] = {
            image[y - 3][x], image[y - 3][x + 1], image[y - 2][x + 2], image[y - 1][x + 3],
            image[y][x + 3], image[y + 1][x + 3], image[y + 2][x + 2], image[y + 3][x + 1],
            image[y + 3][x], image[y + 3][x - 1], image[y + 2][x - 2], image[y + 1][x - 3],
            image[y][x - 3], image[y - 1][x - 3], image[y - 2][x - 2], image[y - 3][x - 1]
        };

        return fast11(circle, center, _threshold);
    }

    void
    detect(const lm::image<lm::gray>& image, lm::array<feature>& features)
    {
        features.clear();

        #pragma omp parallel for
        for (unsigned y = 3; y < image.height() - 3; ++y)
        {
            for (unsigned x = 3; x < image.width() - 3; ++x)
            {
                if (is_corner(image, x, y))
                {
                    #pragma omp critical
                    {
                        features.push(feature{x, y});
                    }
                }
            }
        }
    }

    ~fast()
    {
        lm::log::info("FAST deinitalized");
    }

private:

    int _threshold;
};

} // namespace slam
} // namespace lm
