#pragma once

#include "base/vec.hpp"
#include "base/color.hpp"
#include "base/array.hpp"
#include "base/memory.hpp"

namespace lm {

class octree
{
public:

    struct node
    {
        node(poincloud::point value)
        :   point(value)
        {
            for (int i = 0; i < 8; ++i)
                branch[i] = nullptr;
        }

        pointcloud::point point;
        node* branch[8];
    };

public:

    octree()
    {
        
    }

    node&
    insert(const pointcloud::point& p)
    {

    }

private:

    node* _origin;
};

class pointcloud
{
public:

    struct point
    {
        vec3 pos;
        rgba color;
    };

public:


private:

    octree _data;
};

}
