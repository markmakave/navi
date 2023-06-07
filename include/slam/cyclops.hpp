#pragma once

#include "slam/device.hpp"

namespace lumina {
namespace slam {

class cyclops
{
public:

    cyclops(const char* usb, const char* tof)
      : _usb(usb),
        _tof(tof)
    {}

protected:

    device _usb, _tof;
};

} // namespace slam
} // namespace lumina
