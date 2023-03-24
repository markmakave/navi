#include <iostream>
#include <random>

#include "lumina.hpp"
#include "util/timer.hpp"

#include "neural/network.hpp"

#define ITERATIONS 300
#define RADIUS 2
#define GPU

#define LM_FRAMEWORK_VERSION(v) using namespace lm

LM_FRAMEWORK_VERSION(1.0);

int main()
{
    image<rgb> m1("../dataset/photo.png");
    image<cuda::rgb> m2;

    cuda::matrix<cuda::rgb> dm1, dm2;
    dm2.resize(m1.height(), m1.width());

    dm1 << m1;

    cuda::kernel distort(cuda::distort);

    {
        timer _("compute", ITERATIONS);

        for (int i = 0; i < ITERATIONS; ++i)
        {
            double k = (double(i) / ITERATIONS * 2 - 1) * 10;
            distort(dim3{unsigned(m1.width()) / 8 + 1, unsigned(m1.height()) / 8 + 1}, dim3{8, 8}, cuda::stream::main, dm1, k, 0.0, 0.0, dm2);
            cuda::stream::main.synchronize();
            dm2 >> m2;
            image<rgb>(m2).write(("result/distortion" + std::to_string(i) + ".png").c_str());
        }
    }


}
