#pragma once

#include "slam/device.hpp"

#include <linux/i2c.h>
#include <linux/i2c-dev.h>

namespace lumina {
namespace slam {

class mpu : device {
public:

    mpu(const char* filename)
    :   device(filename)
    {
        struct i2c_rdwr_ioctl_data data;
        struct i2c_msg messages[2];
        unsigned char write_buf[1] = {0xD0}, read_buf[1] = {0x00};
        unsigned char write[200];

        messages[0].addr = 0x50;
        messages[0].flags = 0;
        messages[0].len = 1;
        messages[0].buf = write_buf;

        messages[1].addr = 0x50;
        messages[1].flags = 1;
        messages[1].len = 1;
        messages[1].buf = read_buf;
 
        data.msgs = messages;
        data.nmsgs = 2;
 
        assert(ioctl(I2C_RDWR, &data) == 0);

        printf("ID = 0x%x\n", read_buf[0]);
    }

    ~mpu() {}
};

}
}
