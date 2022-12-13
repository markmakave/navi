#pragma once
#include <iostream>
#include <chrono>

namespace lm {

class timer {

    std::string name;
    std::chrono::high_resolution_clock::time_point begin;

public:

    timer(const std::string& name) : name(name) {
        begin = std::chrono::high_resolution_clock::now();
    }

    ~timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto difference = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        std::cout << name << " done in " << difference / 1000000.0  << " seconds\n";
    }

};

}
