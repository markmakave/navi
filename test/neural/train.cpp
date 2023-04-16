#include "lumina.hpp"

#include "neural/network.hpp"

using namespace lm;

int main()
{
    std::ifstream train_images("../dataset/neural/mnist/train-images-idx3-ubyte");
    std::ifstream train_labels("../dataset/neural/mnist/train-labels-idx1-ubyte");

    train_images.seekg(16);
    train_labels.seekg(8);

    if (!train_images or !train_labels)
    {
        log::error("No such file");
        throw;
    }

    neural::network nn(28*28, 10);

    // TRAIN

    int dataset_size = 60000;

    array<u8> raw_in(28*28 * dataset_size), raw_labels(1 * dataset_size);
    train_images.read((char*)raw_in.data(), raw_in.size() * sizeof(u8));
    train_labels.read((char*)raw_labels.data(), raw_labels.size() * sizeof(u8));

    array<float> in(raw_in.size()), target(10 * raw_labels.size());
    for (i64 i = 0; i < raw_in.size(); ++i)
        in(i) = raw_in(i) / 255.0;

    target.fill(0.f);
    for (i64 i = 0; i < raw_labels.size(); ++i)
        target(10 * i + raw_labels(i)) = 1.f;

    nn.train(in, target, 5, 0.01);

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

    raw_in.reshape(28*28);
    in.reshape(raw_in.size());

    for (int i = 1; i <= 10000; ++i)
    {
        test_images.read((char*)raw_in.data(), raw_in.size());

        for (i64 i = 0; i < raw_in.size(); ++i)
            in(i) = raw_in(i) / 255.f;

        lm::u8 label;
        test_labels.read((char*)&label, 1);

        const auto& prediction = nn.forward(in);
        int max_index = blas::amax(prediction);

        if (max_index == label)
            ncorrect++;

        printf("\rTesting %d/10000", i);
    }
    printf("\n");

    std::cout << "Correct " << ncorrect / 100.0 << '%' << std::endl;

    nn.write("experimental.lmm");
}
