/* 

    Copyright (c) 2023 Mark Mokhov

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

*/

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
