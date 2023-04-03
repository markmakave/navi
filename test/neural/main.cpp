#include "lumina.hpp"

#include "neural/network.hpp"

using namespace lm;

int main()
{
    std::ifstream mnist_images("../dataset/neural/mnist/t10k-images-idx3-ubyte");
    std::ifstream mnist_labels("../dataset/neural/mnist/t10k-labels-idx3-ubyte");
    mnist_images.seekg(16);
    mnist_labels.seekg(8);

    neural::network<float> nn(28*28, 10);

    // TRAIN

    array<u8> raw_in(28*28);
    array<float> in, target(10);

    for (int i = 0; i < 10000; ++i)
    {
        mnist_images.read((char*)raw_in.data(), raw_in.size());
        in = array<float>(raw_in);

        u8 label;
        mnist_labels.read((char*)&label, 1);
        target.fill(0.f);
        target[label] = 1.f;

        nn.train(in, target, 0.1f);
        exit(1);
    }

    // TEST

    mnist_images.seekg(16);
    mnist_labels.seekg(8);

    int ncorrect = 0;

    for (int i = 0; i < 10000; ++i)
    {
        mnist_images.read((char*)in.data(), in.size());

        lm::u8 label;
        mnist_labels.read((char*)&label, 1);
        target.fill(0.f);
        target[label] = 1.f;

        const array<float>& prediction = nn.forward(in);
        int max_index = blas::amax(prediction);

        if (max_index == label)
            ncorrect++;

        printf("\rTesting %d/10000", i);
    }
    printf("\n");

    std::cout << "Correct " << ncorrect << "/ 10000" << std::endl;
}
