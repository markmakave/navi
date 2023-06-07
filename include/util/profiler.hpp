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

#include <fstream>
#include <chrono>
#include <sstream>
#include <thread>

namespace lumina {

#define LM_PROFILE(name) if (auto _ = lumina::profiler::record(name))

class profiler
{

    static inline std::ofstream file;
    static inline std::chrono::high_resolution_clock::time_point start;

    class timer
    {
        friend class profiler;

    public:

        operator bool() const
        {
            return true;
        }

        ~timer() {
            end = std::chrono::high_resolution_clock::now();
            profiler::note(*this);
        }

    private:

        timer(const std::string& scope_name) : scope_name(scope_name) {
            begin = std::chrono::high_resolution_clock::now();
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        }

    private:

        std::chrono::high_resolution_clock::time_point begin, end;
        const std::string scope_name;
    };

    static
    void
    note(const timer& t) {
        auto ts = std::chrono::duration_cast<std::chrono::microseconds>(t.begin - start).count();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t.end - t.begin).count();

        std::stringstream ss;
        ss << "{ \"pid\":1, \"tid\":1, ";
        ss << "\"ts\":" << ts << ", ";
        ss << "\"dur\":" << dur << ", ";
        ss << "\"ph\":\"X\", ";
        ss << "\"name\":\"" << t.scope_name << "\" }, ";

        auto str = ss.str();
        file.write(str.c_str(), str.size());
    }

    profiler() {};

public:

    static 
    void
    begin(const std::string& filename) {
        file = std::ofstream(filename);
        start = std::chrono::high_resolution_clock::now();

        file.write("{ \"traceEvents\": [ ", 19);
    }

    static
    timer
    record(const std::string& scope_name) {
        return { scope_name };
    }

    static
    void
    end() {
        file.seekp(-2, std::ios::cur);
        file.write(" ] }", 4);
        file.close();
    }

};

}
