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
