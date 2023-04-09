#include "lumina.hpp"

#include "neural/network.hpp"

using namespace lm;

#define EPOCHS 1

int main()
{
    std::ifstream train_images("../dataset/neural/mnist/train-images-idx3-ubyte");
    std::ifstream train_labels("../dataset/neural/mnist/train-labels-idx1-ubyte");

    if (!train_images or !train_labels)
    {
        log::error("No such file");
        throw;
    }

    neural::network<float> nn(28*28, 10);

    // TRAIN

    array<u8> raw_in(28*28);
    array<float> in, target(10);

    train_images.seekg(16);
    train_labels.seekg(8);

    for (int epoch = 1; epoch <= EPOCHS; ++epoch)
    {
        for (int i = 1; i <= 60000; ++i)
        {
            train_images.read((char*)raw_in.data(), raw_in.size());
            in = array<float>(raw_in);

            u8 label;
            train_labels.read((char*)&label, 1);
            target.fill(0.f);
            target[label] = 1.f;

            float error = nn.train(in, target, 0.1f);

            printf("\rEpoch: %d. Training %05d/60000. Error: %f", epoch, i, error);
        }
    }
    printf("\n");

    // TEST

    std::ifstream test_images("../dataset/neural/mnist/t10k-images-idx3-ubyte");
    std::ifstream test_labels("../dataset/neural/mnist/t10k-labels-idx1-ubyte");

    if (!test_images or !test_labels)
    {
        log::error("No such file");
        throw;
    }

    test_images.seekg(16);
    test_labels.seekg(8);

    int ncorrect = 0;

    for (int i = 1; i <= 10000; ++i)
    {
        test_images.read((char*)in.data(), in.size());

        lm::u8 label;
        test_labels.read((char*)&label, 1);

        const array<float>& prediction = nn.forward(in);
        int max_index = blas::amax(prediction);

        if (max_index == label)
            ncorrect++;

        printf("\rTesting %d/10000", i);
    }
    printf("\n");

    std::cout << "Correct " << ncorrect / 100.0 << '%' << std::endl;
}
