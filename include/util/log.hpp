#pragma once

#include <iostream>
#include <chrono>

namespace lm {

#define LOG_PROPERTY static inline const char*

class log
{
public:

    LOG_PROPERTY red     = "\033[31m";
    LOG_PROPERTY green   = "\033[32m";
    LOG_PROPERTY yellow  = "\033[33m";
    LOG_PROPERTY blue    = "\033[34m";
    LOG_PROPERTY magenta = "\033[35m";
    LOG_PROPERTY cyan    = "\033[36m";
    LOG_PROPERTY clear   = "\033[0m";

    LOG_PROPERTY info_label    = "[ INFO ]";
    LOG_PROPERTY info_color    = cyan;

    LOG_PROPERTY debug_label   = "[ DEBUG ]";
    LOG_PROPERTY debug_color   = magenta;

    LOG_PROPERTY warning_label = "[ WARNING ]";
    LOG_PROPERTY warning_color = yellow;

    LOG_PROPERTY error_label   = "[ ERROR ]";
    LOG_PROPERTY error_color   = red;

    LOG_PROPERTY separator     = " ";
    LOG_PROPERTY terminator    = "\n";

    template <typename ...Args>
    static void
    info(const Args&... args)
    {
        std::cout << info_color;
        console(std::cout, info_label, date(), args...);
        std::cout << clear;
    }

    template <typename ...Args>
    static void
    debug(const Args&... args)
    {
        std::cout << debug_color;
        console(std::cout, debug_label, date(), args...);
        std::cout << clear;
    }

    template <typename ...Args>
    static void
    warning(const Args&... args)
    {
        std::cout << warning_color;
        console(std::cout, warning_label, date(), args...);
        std::cout << clear;
    }

    template <typename ...Args>
    static void
    error(const Args&... args)
    {
        std::cerr << error_color;
        console(std::cerr, error_label, date(), args...);
        std::cerr << clear;
    }

private:

    log()                       = delete;
    log(const log&)             = delete;
    log(log&&)                  = delete;
    log& operator=(const log&)  = delete;
    log& operator=(log&&)       = delete;
    ~log()                      = delete;

    template <typename Arg>
    static void
    console(std::ostream& stream, const Arg& arg)
    {
        stream << arg << terminator << std::flush;
    }

    template <typename Arg, typename ...Args>
    static void
    console(std::ostream& stream, const Arg& arg, const Args& ...args)
    {
        stream << arg << separator;
        console(stream, args...);
    }

    static
    const char*
    date()
    {
        static char buf[80];
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::strftime(buf, sizeof(buf), "[%d-%m-%Y %X]", std::localtime(&in_time_t));
        return buf;
    }

};

}
