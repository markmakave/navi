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
#include <vector>
#include <thread>
#include <mutex>
#include <cstring>

namespace lumina {

#define LM_PROFILE(name) if (auto _ = lumina::profiler::record(name))

class profiler
{
    struct event
    {
        std::thread::id tid;
        int64_t         begin, duration;
        char            name[128];
    };

    static inline std::vector<event> _events;
    static inline std::mutex         _mutex;

    class timer
    {
        friend class profiler;

    public:

        operator bool() const
        {
            return true;
        }

        ~timer()
        {
            end = std::chrono::high_resolution_clock::now();
            profiler::note(*this);
        }

    private:

        timer(const char* name)
          : name(name)
        {
            begin = std::chrono::high_resolution_clock::now();
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        }

        auto
        elapsed() const
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        }

    private:

        std::chrono::high_resolution_clock::time_point begin, end;
        const char*                                    name;
    };

    static void
    note(const timer& t)
    {
        std::lock_guard<std::mutex> lock(_mutex);

        event e;
        e.tid      = std::this_thread::get_id();
        e.begin    = t.begin.time_since_epoch().count();
        e.duration = t.elapsed();
        std::strncpy(e.name, t.name, sizeof(e.name) - 1);

        _events.push_back(e);
    }

    profiler() {};

public:

    static void
    begin()
    {}

    static timer
    record(const char* scope_name)
    {
        return {scope_name};
    }

    static void
    stage(const char* filename)
    {
        std::ofstream     file(filename);
        std::stringstream ss;

        ss << "{ \"traceEvents\": [ ";
        for (auto& e : _events) {
            ss << "{ \"tid\": " << e.tid << ", ";
            ss << "\"name\":\"" << e.name << "\", ";
            ss << "\"ph\":\"X\", ";
            ss << "\"ts\":" << e.begin/1000 << ", ";
            ss << "\"dur\":" << e.duration/1000 << " }, ";
        }
        ss.seekp(-2, std::ios::cur);
        ss << "]}";

        file << ss.str();
    }
};

} // namespace lumina
