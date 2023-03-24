#include <iostream>
#include <random>

#include "lumina.hpp"
#include "util/timer.hpp"

#define ITERATIONS 1
#define RADIUS 2

#define LM_FRAMEWORK_VERSION(v) using namespace lm

LM_FRAMEWORK_VERSION(1.0);

int main()
{
    image<rgb> m1("../dataset/photo.qoi"), m2;

    cuda::matrix<rgb> dm1, dm2;
    dm2.resize(m1.height(), m1.width());

    dm1 << m1;

    cuda::kernel distort(cuda::distort);

    {
        timer _("compute", ITERATIONS);

        for (int i = 0; i < ITERATIONS; ++i)
        {
            distort({unsigned(m1.width()) / 8 + 1, unsigned(m1.height()) / 8 + 1}, {8, 8}, cuda::stream::main, dm1, -0.5, 0.0, 0.0, dm2);
            cuda::stream::main.synchronize();
        }
    }

    dm2 >> m2;

    m2.write("distortion.qoi");
}
