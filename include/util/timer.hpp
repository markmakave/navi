#pragma once
#include <iostream>
#include <chrono>

#include <util/log.hpp>

namespace lm {

class timer {
public:

    timer(const std::string& name, int iterations = 1)
    :   _begin(std::chrono::high_resolution_clock::now()),
        _name(name),
        _iterations(iterations)
    {}

    ~timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto difference = std::chrono::duration_cast<std::chrono::nanoseconds>(end - _begin).count();
        double seconds = difference /  1000000000.0;

        if (_iterations == 1)
            lm::log::info(_name, "done in", seconds, "seconds");
        else 
            lm::log::info(_name, "iteration average time is", seconds / _iterations, "seconds");
    }

private:

    std::chrono::high_resolution_clock::time_point _begin;
    std::string _name;
    int _iterations;

};

}
