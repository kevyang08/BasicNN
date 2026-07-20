#include "basic_nn.hpp"
#include "progress_bar.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <string.h>

#define BUF_SIZE 4096
#define NUM_LAYERS 3
#define RATE 0.34
#define MOMENTUM 0.9

constexpr int EPOCHS = 1;
constexpr bool VERBOSE = 0;

// TODO: next steps - profiling (including p50-p99 latencies for forward/backward propagate), GPU offloading

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
    // read and load data
    std::vector<std::pair<int, std::vector<float>>> data;
    read_data("mnist_train.csv", data);

    int num_layers = NUM_LAYERS;
    std::vector<int> layer_sizes = {784, 500, 10};
    float learning_rate = RATE;
    float momentum = MOMENTUM;
    neural_network nn(num_layers, layer_sizes, learning_rate, momentum, data);

    // ============================================== START OF TRAINING =====================================================

    std::cout << "Beginning training with " << num_layers << " layers and a learning rate of " << learning_rate << std::endl;

    // progress bar, unique_ptr for runtime polymorphism
    std::unique_ptr<ProgressBar> progress_bar = std::make_unique<NullProgressBar>();
    std::function<void(int)> progress_callback = nullptr;
    if constexpr (VERBOSE) {
        progress_bar = std::make_unique<VerboseProgressBar>(EPOCHS, data.size());
        progress_callback = [&](int cur_progress) {
            progress_bar->update_progress(cur_progress);
        };
    }

    // time how long training takes
    auto start = std::chrono::steady_clock::now();

    for (int e = 1; e <= EPOCHS; ++e) {
        // print epoch
        progress_bar->update_epoch(e);

        // print initial progress bar
        progress_bar->update_progress(0);
        progress_bar->start_print_progress();

        nn.train(progress_callback);

        progress_bar->end_print_progress();

        nn.adjust_lr();
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Training completed in " << duration.count()/1000.0 << " seconds" << std::endl;

    // ================================================ END OF TRAINING ======================================================

    read_data("mnist_test.csv", data);

    int num_inputs = data.size(), num_correct = 0;

    // validation pass
    for (auto &[label, values] : data) {
        int result = nn.query(values);
        num_correct += (result == label);
    }

    std::cout << "Accuracy: " << (double)num_correct/num_inputs << std::endl;

    return 0;
}