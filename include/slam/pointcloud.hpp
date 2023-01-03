#pragma once

#include <random>

#include "base/array.hpp"
#include "base/vec.hpp"
#include "base/color.hpp"

#include "util/log.hpp"

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

namespace lm {
namespace slam {

struct point
{
    lm::vec3 position;
    lm::rgba color;

    point()
    {
        // use merseene twister to generate random numbers
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(-1.f, 1.f);
        static std::uniform_int_distribution<int> dis_color(0, 255);

        position = lm::vec3(dis(gen), dis(gen), dis(gen));
        color = lm::rgba(dis_color(gen), dis_color(gen), dis_color(gen), 255);
    }
};

class pointcloud
{
public:

    pointcloud()
    {
        lm::log::info("Pointcloud initialized");
    }

    ~pointcloud()
    {
        lm::log::info("Pointcloud deinitalized");
    }

    void
    push(const point& p)
    {
        _points.push(p);
        lm::log::info("Point added");
    }

private:

    lm::array<point> _points;
};

} // namespace slam
} // namespace lm
