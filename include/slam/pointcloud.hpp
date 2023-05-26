#pragma once

#include "base/vec.hpp"
#include "base/color.hpp"
#include "base/array.hpp"
#include "base/memory.hpp"

namespace lm {
namespace slam {

class pointcloud
{
public:

    struct point
    {
        vec3 pos;
        rgba color;
    };

public:

    pointcloud()
    {}

private:

    array<point> _data;
};

}
}
