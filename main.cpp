#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <string.h>
#include <algorithm>
#include <random>
#include "basic_nn.hpp"

#define BUF_SIZE 5000
#define NUM_LAYERS 3
#define RATE 0.3
#define MOMENTUM 0.9
#define EPOCHS 10

inline void print_progress(int progress) {
    std::cout << "\r[";
    for (int i = 0; i < 50; i++) std::cout << (progress > i * 2 + 1 ? '=' : ' ');
    std::cout << "] " << progress << "%";
}
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
    neural_network nn(num_layers, layer_sizes, learning_rate, momentum);

    std::vector<std::pair<int, std::vector<float>>> data;
    std::vector<float> expected(10);

    std::cout << "Beginning training with " << num_layers << " layers and a learning rate of " << learning_rate << std::endl;

    read_data("mnist_train.csv", data);

    // time how long training takes
    double duration;
    std::clock_t start = std::clock();

    // multiple epochs
    int epochs = EPOCHS;
    for (int e = 1; e <= epochs; e++) {

        std::cout << "Epoch " << e << "/" << epochs << std::endl;

        // randomly shuffle training data
        std::shuffle(data.begin(), data.end(), std::random_device());

        // to reduce redundant updates
        int prev_progress = 0;
        print_progress(0);

        for (int i = 0; i < data.size(); i++) {
            auto &[label, values] = data[i];
            std::fill(expected.begin(), expected.end(), 0);
            expected[label] = 1;
            nn.train(values, expected);

            // progress bar
            int progress = (int)((i + 1.0)/data.size() * 100);
            if (progress == prev_progress) continue;
            print_progress(progress);
            prev_progress = progress;
        }

        std::cout << std::endl;

        nn.adjust_lr();

    }

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