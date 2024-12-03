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

#define ERROR   (1 << 0)
#define WARNING (1 << 1)
#define INFO    (1 << 2)
#define DEBUG   (1 << 3)
#define TRACE   (1 << 4)

#define ALL         (ERROR | WARNING | INFO | DEBUG | TRACE)
#define PRODUCTION  (ERROR | WARNING | INFO)

#define LOG_LEVEL (ALL)

namespace lumina::log
{

#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define GREEN   "\033[32m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define CLEAR   "\033[0m"

struct nosep_t {};
static inline constexpr nosep_t nosep;

template <typename... Args>
void log(std::ostream& os, Args&&... args)
{
    ((os << std::forward<Args>(args)), ...) << '\n';
}

template <typename... Args>
void error(Args&&... args)
{
    log(std::cerr, RED, "[E]", args..., CLEAR);
}

template <typename... Args>
void warning(Args&&... args)
{
    log(std::cerr, YELLOW, "[W]", args..., CLEAR);
}

template <typename... Args>
void info(Args&&... args)
{
    log(std::cout, GREEN, "[I]", args..., CLEAR);
}

template <typename... Args>
void debug(Args&&... args)
{
    log(std::cout, MAGENTA, "[D]", args..., CLEAR);
}

template <typename... Args>
void trace(Args&&... args)
{
    log(std::cout, CYAN, "[T]", args..., CLEAR);
}

}
