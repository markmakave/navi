#include <iostream>

#include "slam/slam.hpp"
#include "util/timer.hpp"
#include "base/socket.hpp"

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <omp.h>

#include <neural/network.hpp>

int main()
{
    lm::neural::network net(1, 5, 1);

    

    return 0;
}
