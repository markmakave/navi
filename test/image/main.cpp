#include "base/image.hpp"

int main()
{
    lm::image<lm::gray> gray("../dataset/car.png");
    gray.write("gray.png");

    lm::image<lm::rgb> rgb("../dataset/car.png");
    rgb.write("rgb.qoi");

    lm::image<lm::rgba> rgba("../dataset/car.png");
    rgba.write("rgba.qoi");
}
