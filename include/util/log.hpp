#pragma once

#include <iostream>
#include <chrono>

namespace lm {

#define LOG_PROPERTY static inline const char* const

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

    struct config {
        std::ostream& stream;
        const char* label;
        const char* color;
        const char* separator;
        const char* terminator;
    };

    static inline const config info_config = {
        std::cout,
        "[ INFO ]",
        cyan,
        " ",
        "\n"
    };

    static inline const config debug_config = {
        std::cout,
        "[ DEBUG ]",
        magenta,
        " ",
        "\n"
    };

    static inline const config warning_config = {
        std::cout,
        "[ WARNING ]",
        yellow,
        " ",
        "\n"
    };

    static inline const config error_config = {
        std::cerr,
        "[ ERROR ]",
        red,
        " ",
        "\n"
    };

    template <typename ...Args>
    static void
    print(const config& cfg, const Args&... args)
    {
        cfg.stream << cfg.color;
        console(cfg, cfg.label, date(), args...);
        cfg.stream << clear;
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

    log()                       = delete;
    log(const log&)             = delete;
    log(log&&)                  = delete;
    log& operator=(const log&)  = delete;
    log& operator=(log&&)       = delete;
    ~log()                      = delete;

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
        std::strftime(buf, sizeof(buf), "[%d-%m-%Y %X]", std::localtime(&in_time_t));
        return buf;
    }

    #undef LOG_PROPERTY
};

}
