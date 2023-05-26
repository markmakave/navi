#include "lumina.hpp"

using namespace lm;

#define max_iteration 100

/*
For each pixel (x, y) on the screen, do:
{
    x = scaled x coordinate of pixel (scaled to lie in the Mandelbrot X scale (-2.5, 1))
    y = scaled y coordinate of pixel (scaled to lie in the Mandelbrot Y scale (-1, 1))
       
    
    zx = x; // zx represents the real part of z
    zy = y; // zy represents the imaginary part of z

  
    iteration = 0
    max_iteration = 1000
  
    while (zx*zx + zy*zy < 4  AND  iteration < max_iteration) 
    {
        xtemp = zx*zx - zy*zy + x
        zy = -2*zx*zy + y
        zx = xtemp

        iteration = iteration + 1
    }

    if (iteration == max_iteration) //Belongs to the set
        return insideColor;

    return iteration * color;
}*/

rgb
palette(int iteration)
{
    if (iteration == max_iteration)
        return 0;

    int color = 255 * std::sqrt(float(iteration) / max_iteration);
    return color;

    // static rgb p[max_iteration];
    // static bool initialized = false;

    // if (!initialized)
    // {
    //     initialized = true;
    //     for (int i = 0; i < max_iteration; ++i)
    //         p[i] = rgb::random();
    // }

    // return p[iteration];
}

int main()
{
    image<rgb> frame(4000, 4000);

    #pragma omp parallel for
    for (int px = 0; px < frame.shape(0); ++px)
    for (int py = 0; py < frame.shape(1); ++py)
    {
        #define TRICON

        #ifdef TRICON

        float x = px / float(frame.shape(0)) * 5.0 - 2.5,
              y = py / float(frame.shape(1)) * 5.0 - 2.5;

        float zx = x,
              zy = y;

        int iteration = 0;
        while (((zx*zx + zy*zy) < 2*2) && (iteration < max_iteration))
        {
            auto xtemp = zx*zx - zy*zy + x;
            zy = -2*zx*zy + y;
            zx = xtemp;

            iteration++;
        }

        frame(px, py) = palette(iteration);

        #else

        float x0 = px / float(frame.shape(0)) * 2.47 - 2.0,
              y0 = py / float(frame.shape(1)) * 2.24 - 1.12;

        float x = 0, y = 0;
        int iteration = 0;
        while (((x*x + y*y) < 2*2) && (iteration < max_iteration))
        {
            auto xtemp = x*x - y*y + x0;
            y = 2*x*y + y0;
            x = xtemp;

            iteration++;
        }

        frame(px, py) = palette(iteration);

        #endif
    }

    frame.write("fractal.qoi");
}
