#include <iostream>
#include <future>
#include <thread>

#include "slam/slam.hpp"

int main()
{
    lm::slam slam;

    slam.run();

    std::this_thread::sleep_for(std::chrono::seconds(30));

    slam.terminate();

    return 0;
}
