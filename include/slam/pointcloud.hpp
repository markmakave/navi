#pragma once

#include "base/vec.hpp"
#include "base/color.hpp"
#include "base/array.hpp"

namespace lm {

class pointcloud
{
public:

    struct point
    {
        vec3 pos;
        rgba color;
    };

public:

    // Sort points by distance from origin
    void
    sort()
    {

    }

private:

    array<point> _data;
}

}
