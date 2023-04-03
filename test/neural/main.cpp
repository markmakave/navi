#include "lumina.hpp"

#include "neural/network.hpp"

using namespace lm;

int main()
{
    std::ifstream mnist_images("../dataset/neural/mnist/train-images-idx3-ubyte");
    std::ifstream mnist_labels("../dataset/neural/mnist/train-labels-idx3-ubyte");
    

    neural::network<float> nn(28*28, 10);

    // TRAIN

    array<u8> raw_in(28*28);
    array<float> in, target(10);

    mnist_images.seekg(16);
    mnist_labels.seekg(8);

    for (int i = 0; i < 60000; ++i)
    {
        mnist_images.read((char*)raw_in.data(), raw_in.size());
        in = array<float>(raw_in);

        u8 label;
        mnist_labels.read((char*)&label, 1);
        target.fill(0.f);
        target[label] = 1.f;

        log::info("Error:", nn.train(in, target, 0.1f));
    }

    // TEST

    mnist_images.seekg(16);
    mnist_labels.seekg(8);

    int ncorrect = 0;

    for (int i = 0; i < 60000; ++i)
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

        printf("\rTesting %d/60000", i);
    }
    printf("\n");

    std::cout << "Correct " << ncorrect << "/ 60000" << std::endl;
}
