/* 

    Copyright (c) 2023 Mokhov Mark

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

#include <iostream>
#include <stdexcept>

#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>

namespace lm {
namespace slam {

class device
{
public:

    device(const char* filepath)
    {
        fd = open(filepath, O_RDWR);
        if (fd < 0)
            throw std::runtime_error("Failed to open device");
    }

    virtual
    ~device()
    {
        close(fd);
    }

    virtual 
    void 
    info()
    {
        std::cout << "No device info provided" << std::endl;
    }

    int
    ioctl(int request, void* arg)
    {
        int r;
        
        do {
            r = ::ioctl(fd, request, arg);
        } while (-1 == r && EINTR == errno);

        return r;
    }

protected:

    int fd;

};

} // namespace slam
} // namespace lm
