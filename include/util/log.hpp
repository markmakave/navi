/* 

    Copyright (c) 2023 Mark Mokhov

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
#include <ctime>

namespace lm {

class log
{
public:

    #define cyan    "\033[36m"
    #define magenta "\033[35m"
    #define yellow  "\033[33m"
    #define red     "\033[31m"
    #define clear   "\033[0m"

    struct config
    {
        std::ostream& stream;
        const char* label;
        const char* color;
        const char* separator;
        const char* terminator;
    };

    #define info_config    { std::cout, "[ INFO ]",    cyan,    " ", "\n" }
    #define debug_config   { std::cout, "[ DEBUG ]",   magenta, " ", "\n" }
    #define warning_config { std::cerr, "[ WARNING ]", yellow,  " ", "\n" }
    #define error_config   { std::cerr, "[ ERROR ]",   red,     " ", "\n" }

    template <typename ...Args>
    static void
    print(const config& cfg, const Args&... args)
    {
        cfg.stream << cfg.color;
        console(cfg, cfg.label, date(), args...);
        cfg.stream << clear;
        std::flush(cfg.stream);
    }

    template <typename ...Args>
    static void
    info(const Args&... args)
    {
        print(info_config, args...);
    }

    template <typename ...Args>
    static void
    debug(const Args&... args)
    {
        print(debug_config, args...);
    }

    template <typename ...Args>
    static void
    warning(const Args&... args)
    {
        print(warning_config, args...);
    }

    template <typename ...Args>
    static void
    error(const Args&... args)
    {
        print(error_config, args...);
    }

private:

    log()                      = delete;
    log(const log&)            = delete;
    log(log&&)                 = delete;
    log& operator=(const log&) = delete;
    log& operator=(log&&)      = delete;
    ~log()                     = delete;

    template <typename Arg>
    static void
    console(const config& conf, const Arg& arg)
    {
        conf.stream << arg << conf.terminator << std::flush;
    }

    template <typename Arg, typename ...Args>
    static void
    console(const config& conf, const Arg& arg, const Args& ...args)
    {
        conf.stream << arg << conf.separator;
        console(conf, args...);
    }

    static
    const char*
    date()
    {
        static char buf[80];
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::strftime(buf, sizeof(buf), "[%d.%m.%Y %X]", std::localtime(&in_time_t));
        return buf;
    }

    #undef cyan
    #undef magenta
    #undef yellow
    #undef red
    #undef clear

};

#define INFO(...)    lm::log::info(__VA_ARGS__)
#define WARNING(...) lm::log::warning(__VA_ARGS__)
#define ERROR(...)   lm::log::error(__VA_ARGS__)

#ifndef NDEBUG
#define DEBUG(...)   lm::log::debug(__VA_ARGS__)
#else
#define DEBUG(...) {}
#endif

}
