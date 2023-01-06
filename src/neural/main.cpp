#include <iostream>

#include <neural/network.hpp>

int main()
{
    lm::neural::network net(2, 3, 1);

    // train the network to output 1 for x > y and 0 otherwise

    lm::array<lm::matrix<double>> inputs(10000);
    lm::array<lm::matrix<double>> outputs(10000);

    for (unsigned i = 0; i < inputs.size(); ++i)
    {
        inputs[i] = lm::matrix<double>(2, 1);
        inputs[i].randomize();

        outputs[i] = lm::matrix<double>(1, 1);
        outputs[i][0][0] = inputs[i][0][0] > inputs[i][1][0] ? 1.0 : 0.0;
    }

    for (unsigned i = 0; i < 1000; ++i)
    {
        double error = 0.0;
        for (unsigned j = 0; j < inputs.size(); ++j)
        {
            error += net.train(inputs[j], outputs[j], 0.1);
        }

        std::cout << "Error: " << error / inputs.size() << std::endl;
    }

    double accuracy = 0.0;
    for (unsigned i = 0; i < inputs.size(); ++i)
    {
        accuracy += std::abs(net.forward(inputs[i])[0][0] - outputs[i][0][0]);
    }

    std::cout << "Accuracy: " << accuracy / inputs.size() << std::endl;
}
