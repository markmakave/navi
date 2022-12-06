#include <iostream>

#include "graphics/image.hpp"
#include "graphics/color.hpp"
#include "graphics/kernel.hpp"

int main()
{

    lm::image<lm::rgb> img("image.png");

    std::cout << img.width() << "x" << img.height() << std::endl;

    img = img.convolve(lm::kernel::gaussian(5, 25));

    img.write("blur.png");

    return 0;
}
