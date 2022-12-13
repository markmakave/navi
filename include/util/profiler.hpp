#pragma once

#include <fstream>
#include <chrono>
#include <sstream>
#include <thread>

namespace lm {

#define LM_PROFILE(name) if (auto _ = lm::profiler::record(name))

class profiler {

    static std::ofstream file;
    static std::chrono::high_resolution_clock::time_point start;

    class timer {

        friend class profiler;

        std::chrono::high_resolution_clock::time_point begin, end;
        const std::string scope_name;

        timer(const std::string& scope_name) : scope_name(scope_name) {
            begin = std::chrono::high_resolution_clock::now();
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        }

    public:

        constexpr
        operator bool() const {
            return true;
        }

        ~timer() {
            end = std::chrono::high_resolution_clock::now();
            profiler::note(*this);
        }

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
