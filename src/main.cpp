#include <iostream>
#include <future>
#include <thread>

#include "slam/slam.hpp"

int main()
{
    lm::slam slam;

    std::future<void> slam_future = std::async(std::launch::async, &lm::slam::run, &slam);

    std::this_thread::sleep_for(std::chrono::seconds(5));
    slam.terminate();

    slam_future.wait();

    return 0;
}
