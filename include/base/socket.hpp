#pragma once

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

#include "util/log.hpp"
#include "base/array.hpp"

namespace lm {

class socket
{
public:

    socket(const char* path)
    :   _socket(-1)
    {
        _socket = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (_socket < 0)
        {
            lm::log::error("Socket creation failed");
            return;
        }

        struct sockaddr_un address;
        memset(&address, 0, sizeof(address));
        address.sun_family = AF_UNIX;
        strncpy(address.sun_path, path, sizeof(address.sun_path) - 1);

        if (::connect(_socket, (struct sockaddr*) &address, sizeof(address)) < 0)
        {
            lm::log::error("Socket connection failed");
            close(_socket);
            _socket = -1;
            return;
        }

        lm::log::info("Socket initialized");
    }

    ~socket()
    {
        if (_socket >= 0)
        {
            close(_socket);
        }
        lm::log::info("Socket deinitalized");
    }

    operator bool() const
    {
        return _socket >= 0;
    }

    void
    send(const void* data, size_t size) const
    {
        if (_socket >= 0)
        {
            ::send(_socket, data, size, 0);
        }
    }

    template <typename T>
    void
    send(const lm::array<T>& data) const
    {
        send(data.data(), data.size() * sizeof(T));
    }

private:

    int _socket;
};

}
