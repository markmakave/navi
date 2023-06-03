#include <vendor/json/single_include/nlohmann/json.hpp>
using json = nlohmann::json;

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "neural/network.hpp"

#define INPUT_SIZE 100
#define HIDDEN_SIZE 2000
#define OUTPUT_SIZE 1

int
main(int argc, char** argv)
{
    lm::neural::network net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    std::ifstream chartFile(argv[1]);
    json          chart;
    chartFile >> chart;

    lm::array<float> input(INPUT_SIZE);
    lm::array<float> output(OUTPUT_SIZE);

    std::map<int64_t, std::pair<float, float>> chartValues;

    // init read values
    for (auto& entry : chart) {
        chartValues[entry["x"].get<int64_t>()].first  = entry["y"].get<float>();
        chartValues[entry["x"].get<int64_t>()].second = entry["y"].get<float>();
    }

    for (int i = 0; i < 3000; ++i) {

        // fill input
        for (int j = 0; j < INPUT_SIZE; ++j) {
            int64_t time  = chart[i + j]["x"].get<int64_t>();
            float   value = chart[i + j]["y"].get<float>();

            input[j] = value;
        }

        // fill output
        output[0] = chart[i + INPUT_SIZE]["y"].get<float>();
        lm::log::info("Output:", output[0]);

        net.train(input, output, 0.001f);

        lm::log::info("Training entry:", i);
    }

    // infer

    for (int i = 3000; i < 3500; ++i) {

        for (int j = 0; j < INPUT_SIZE; ++j) {
            int64_t time  = chart[i + j]["x"].get<int64_t>();
            float   value = chart[i + j]["y"].get<float>();

            input[j] = value;
        }

        const auto& out = net.infer(input);

        int64_t nextTime = chart[i + INPUT_SIZE]["x"].get<int64_t>();
        chartValues[nextTime].second = out[0];

        lm::log::info("Inference entry:", i);
    }

    // write to csv
    std::ofstream csvFile(argv[2]);
    for (auto& entry : chartValues) {
        csvFile << entry.first << "," << entry.second.first << ","
                << entry.second.second << std::endl;
    }

    return 0;
}
