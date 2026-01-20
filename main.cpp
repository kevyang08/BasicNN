#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <algorithm>
#include <string.h>
#include "basic_nn.hpp"

#define BUF_SIZE 5000
#define NUM_LAYERS 3
#define RATE 0.34
#define MOMENTUM 0.9
#define EPOCHS 10
#define DEBUG 0

void read_data(const std::string& filename, std::vector<std::pair<int, std::vector<float>>>& data) {
    std::ifstream input(filename);
    data.clear();
    char buf[BUF_SIZE];
    char *tmp;

    while (!input.eof()) {
        input.getline(buf, BUF_SIZE - 1);

        if (strlen(buf) == 0) break;

        int label = std::stoi(strtok(buf, ","));
        std::vector<float> values;
        
        while ((tmp = strtok(NULL, ",")) != NULL) {
            values.push_back(std::stoi(tmp)/255.0);
        }

        data.push_back({label, values});
    }
}
int main() {
    int num_layers = NUM_LAYERS;
    std::vector<int> layer_sizes = {784, 500, 10};
    float learning_rate = RATE;
    float momentum = MOMENTUM;
    neural_network nn(num_layers, layer_sizes, learning_rate, momentum, DEBUG);

    std::vector<std::pair<int, std::vector<float>>> data;

    read_data("mnist_train.csv", data);

    // time how long training takes
    double duration;
    std::clock_t start = std::clock();

    nn.train(data, EPOCHS);

    duration = (std::clock() - start)/(double)CLOCKS_PER_SEC;
    std::cout<<"Training completed in "<< duration << " seconds" << std::endl;

    read_data("mnist_test.csv", data);

    int num_inputs = data.size(), num_correct = 0;

    for (auto &[label, values] : data) {
        int result = nn.query(values);
        num_correct += (result == label);
    }

    std::cout << "Accuracy: " << (double)num_correct/num_inputs << std::endl;

    return 0;
}