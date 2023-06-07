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
#include <chrono>

#include <util/log.hpp>

namespace lumina {

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
            lumina::log::info(_name, "done in", seconds, "seconds");
        else 
            lumina::log::info(_name, "iteration average time is", seconds / _iterations, "seconds");
    }

private:

    std::chrono::high_resolution_clock::time_point _begin;
    std::string _name;
    int _iterations;

};

}
