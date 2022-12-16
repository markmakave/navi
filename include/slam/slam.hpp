#pragma once

#include <graphics/image.hpp>

#include <slam/camera.hpp>
#include <slam/pointcloud.hpp>
#include <slam/brief.hpp>

#include "util/log.hpp"
#include "util/timer.hpp"

#include <future>
#include <thread>
#include <atomic>

#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>

namespace lm {

struct feature
{
    lm::dim2 position;
    lm::brief<256> descriptor;
};

class slam
{
public:

    slam()
    // :   _l_camera(0),
    //     _r_camera(1),
    :   _brief(16),
        _terminate(false)
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
        _task = std::async(std::launch::async, &slam::task, this);
        lm::log::info("SLAM running");
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

    void
    _extract()
    {
        // _brief.extract(_l_image, _l_features);
        // _brief.extract(_r_image, _r_features);
    }

    void
    _match()
    {

    }

    void
    task()
    {
        lm::image<lm::rgb> image(640, 480);

        int sock = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr;
        addr.sun_family = AF_UNIX;
        strcpy(addr.sun_path, "/tmp/hertz_video.sock");
        if (connect(sock, (sockaddr*)&addr, sizeof(addr)) == -1)
        {
            lm::log::warning("Failed to connect to socket. No data will be sent");
        }
        
        int fd = open("/dev/urandom", O_RDONLY);

        while (!_terminate)
        {
            // randomize image using 
            read(fd, image.data(), image.size() * sizeof(lm::rgb));

            lm::array<uint8_t> encoded = image.encode_jpeg(10);

            send(sock, encoded.data(), encoded.size(), MSG_DONTWAIT);
        }

        lm::log::info("SLAM terminated");
    }

private:

    // lm::camera _l_camera, _r_camera;
    lm::image<lm::rgb> _l_image, _r_image;

    lm::brief<256> _brief;
    lm::array<lm::feature> _l_features, _r_features;

    lm::pointcloud _pointcloud;

    std::future<void> _task;
    std::atomic<bool> _terminate;
};

}
