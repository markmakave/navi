#pragma once

#include <random>

#include "base/array.hpp"
#include "base/vec.hpp"
#include "graphics/color.hpp"

#include "util/log.hpp"

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

namespace lm {

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

class data_socket
{
public:

    data_socket(const char* socker_file)
    {
        _socket = socket(AF_UNIX, SOCK_STREAM, 0);
        if (_socket < 0)
        {
            lm::log::warning("Failed to create socket. No data will be sent");
        }
        else
        {
            struct sockaddr_un addr;
            memset(&addr, 0, sizeof(addr));
            addr.sun_family = AF_UNIX;
            strncpy(addr.sun_path, socker_file, sizeof(addr.sun_path)-1);
            if (connect(_socket, (struct sockaddr*)&addr, sizeof(addr)) == -1)
            {
                lm::log::warning("Failed to connect to socket. No data will be sent");
            }
        }

        lm::log::info("Data socket initialized");
    }

    ~data_socket()
    {
        if (_socket >= 0)
        {
            close(_socket);
        }
        lm::log::info("Data socket deinitalized");
    }
    
    operator bool() const
    {
        return _socket >= 0;
    }

    void
    send(const point& p) const
    {
        if (_socket >= 0)
        {
            ::send(_socket, &p, sizeof(p), 0);
        }
    }

private:

    int _socket;

};

class pointcloud
{
public:

    pointcloud()
    :   _socket("/tmp/hertz_points.sock")
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
        _socket.send(p);
        lm::log::info("Point added");
    }

private:

    lm::array<point> _points;
    lm::data_socket _socket;
};

}
