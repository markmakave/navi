#include "lumina.hpp"

#include "neural/network.hpp"

#define ITERATIONS 0

using namespace lm;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./infer [input image]" << std::endl;
        return 1;
    }

    neural::network nn(argc > 2 ? argv[1] : "model.lmm");

    image<gray> image(argc > 2 ? argv[2] : argv[1]);
    array<float> model_input(nn.in_size());

    for (i64 i = 0; i < image.size(); ++i)
        model_input(i) = image.data()[i] / 255.0;


    for (int i = 0; i < ITERATIONS; ++i)
        nn.forward(model_input);

    const array<float>& model_output = nn.forward(model_input);

    std::cout << blas::amax(model_output) << std::endl;
}
