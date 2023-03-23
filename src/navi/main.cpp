#include <iostream>
#include <random>

#include "lumina.hpp"
#include "util/timer.hpp"

#define ITERATIONS 1
#define RADIUS 2
#define GPU

#define LM_FRAMEWORK_VERSION(v) using namespace lm

LM_FRAMEWORK_VERSION(1.0);

void
distort(const matrix<rgb>& in, matrix<rgb>& out)
{
    out.resize(in.height(), in.width());

    double k1 = 0.2, k2 = 0, k3 = 0;

    auto index = [&](double r) -> double {
        return 1 + k1*r*r + k2*r*r*r*r + k3*r*r*r*r*r*r;
    };

    auto normalize = [](int x, int dimention) -> double {
        return double(x) / dimention * 2 - 1;
    };

    auto unnormalize = [](double x, int dimention) -> double {
        return (x + 1) * dimention / 2;
    };

    for (int y = 0; y < in.height(); ++y)
        for (int x = 0; x < in.width(); ++x)
        {
            double x_normalized = normalize(x, in.width());
            double y_normalized = normalize(y, in.height());

            double r = x_normalized*x_normalized + y_normalized*y_normalized;

            out[y][x] = in.at(unnormalize(y_normalized * index(r), in.height()), unnormalize(x_normalized * index(r), in.width()));
        }
}

int main()
{
    image<rgb> m1("../dataset/photo.png"), m2;

    distort(m1, m2);

    m2.write("distortion.png");
}
