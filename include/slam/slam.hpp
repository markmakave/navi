#pragma once

#include <graphics/image.hpp>
#include <slam/camera.hpp>
#include <slam/pointcloud.hpp>
#include "util/log.hpp"

#include <future>
#include <thread>
#include <atomic>

namespace lm {

class slam
{
public:

    slam()
    // :   _l_camera(0),
    //     _r_camera(1),
    :   _terminate(false)
    {
        lm::log::info("SLAM initialized");


    }

    ~slam()
    {
        lm::log::info("SLAM deinitalized");
    }

    void
    run()
    {
        lm::log::info("SLAM running");

        while (!_terminate)
        {
            _capture();
        }

        lm::log::info("SLAM terminated");
    }

    void
    terminate()
    {
        _terminate = true;
    }

private:

    void
    _capture()
    {
        // _l_camera >> _l_image;
        // _r_camera >> _r_image;
    }    

private:

    // lm::camera _l_camera, _r_camera;
    lm::image<lm::rgb> _l_image, _r_image;

    lm::pointcloud _pointcloud;

    std::atomic<bool> _terminate;

};

}
