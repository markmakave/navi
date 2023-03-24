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

#include <cstdint>
#include <random>
#include <chrono>

namespace lm {

template<int lower, int higher>
static int clamp(int value)
{
    if (value > higher) return higher;
    if (value < lower) return lower;
    return value;
}

inline
uint8_t
random_channel()
{
    static std::mt19937 engine(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_int_distribution<uint8_t> distribution(0, 255);
    return distribution(engine);
}

struct yuv;
struct rgb;
struct rgba;
typedef uint8_t gray;

// YUYV ///////////////////////////////////////////////////////////////////

struct yuv
{

    //  Base yuv color scheme components
    uint8_t y, u, v;

    //  Default constructor sets the color to black
    yuv(uint8_t y = 0, uint8_t u = 0, uint8_t v = 0) : y(y), u(u), v(v) {}

    //  Conversion constructor from rgb
    // yuv(const rgb& color) {
             
    // }

};

// RGB 24bit color structure //////////////////////////////////////////////

struct rgb
{
    // Base rgb color scheme components
    uint8_t r, g, b;

    // Default constructor sets the color to black
    rgb() : r(0), g(0), b(0) {}

    // From greyscale constructor
    rgb(uint8_t gray) : r(gray), g(gray), b(gray) {}

    // Full assignment constructor
    rgb(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}

    // Conversion constructor from yuv
    rgb(const yuv& color) {
        b = clamp<0, 255>((int)color.y + (1.732446 * ((int)color.u - 128)));
        g = clamp<0, 255>((int)color.y - (0.698001 * ((int)color.v - 128)) - (0.337633 * ((int)color.u - 128)));
        r = clamp<0, 255>((int)color.y + (1.370705 * ((int)color.v - 128)));
    }

    operator uint8_t() const {
        return clamp<0, 255>(0.2161 * r + 0.7152 * g + 0.0722 * b);
    }

    static
    rgb
    random() {
        rgb color = {
            random_channel(),
            random_channel(),
            random_channel()
        };
        return color;
    }
    
};

const rgb white = { 255, 255, 255 };
const rgb black = { 0, 0, 0 };
const rgb red = { 255, 0, 0 };
const rgb green = { 0, 255, 0 };
const rgb blue = { 0, 0, 255 };
const rgb yellow = { 255, 255, 0 };
const rgb cyan = { 0, 255, 255 };
const rgb magenta = { 255, 0, 255 };

// RGBA 32bit color structure /////////////////////////////////////////////

struct rgba : rgb
{

    // Adding alpha component to rgb
    uint8_t a;

    // Default constructor
    rgba() : rgb(), a(0) {}

    rgba(uint8_t gray) : rgb(gray), a(255) {}

    // Color components based constructor
    rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) : rgb(r, g, b), a(a) {}

    // Conversion from rgb
    rgba(const rgb& color) : rgb(color), a(255) {}

    // Conversion from yuv
    rgba(const yuv& color) : rgb(color), a(255) {}

    static
    rgba
    random() {
        rgba color = {
            random_channel(),
            random_channel(),
            random_channel(),
            random_channel()
        };
        return color;
    }

};

}
