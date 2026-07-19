#pragma once

#include "utils.hpp"
#include <vector>

class neural_network {
private:
    const int num_layers;
    const bool verbose;
    std::vector<float, AlignedAllocator<float, 64>> layer;
    std::vector<float, AlignedAllocator<float, 64>> error;
    std::vector<float, AlignedAllocator<float, 64>> weights;
    std::vector<int> layer_bounds;
    std::vector<int> weights_bounds;
    float learning_rate;
    float momentum;
    void forward_propagate(std::vector<float>& inputs);
    void backward_propagate(std::vector<float>& expected);

public:
    neural_network(int num_layers, std::vector<int>& layer_sizes, float learning_rate, float momentum, bool verbose);
    // void train(std::vector<float>& inputs, std::vector<float>& expected);
    void train(std::vector<std::pair<int, std::vector<float>>>& data, int epochs);
    int query(std::vector<float>& inputs);
    void adjust_lr();
};