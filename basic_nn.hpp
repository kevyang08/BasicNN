#pragma once

#include "utils.hpp"
#include <functional>
#include <vector>

class neural_network {
private:
    std::vector<float> expected;
    std::vector<float, AlignedAllocator<float, 64>> layer;
    std::vector<float, AlignedAllocator<float, 64>> error;
    std::vector<float, AlignedAllocator<float, 64>> weights;
    std::vector<int> layer_bounds;
    std::vector<int> weights_bounds;
    const int num_layers;
    float learning_rate;
    float momentum;
    std::vector<std::pair<int, std::vector<float>>>& training_data;
    void forward_propagate(std::vector<float>& inputs);
    void backward_propagate();

public:
    neural_network(const int num_layers, std::vector<int>& layer_sizes, float learning_rate, float momentum, std::vector<std::pair<int, std::vector<float>>>& training_data);
    inline void load_training_data(std::vector<std::pair<int, std::vector<float>>>& training_data) {
        this->training_data = training_data;
    }
    inline void load_training_data_copy(std::vector<std::pair<int, std::vector<float>>> training_data) {
        this->training_data = training_data;
    }
    void train(std::function<void(int)> progress_callback);
    int query(std::vector<float>& inputs);
    inline int get_num_layers() {
        return num_layers;
    }
    inline float get_learning_rate() {
        return learning_rate;
    }
    inline void set_learning_rate(const float learning_rate) {
        this->learning_rate = learning_rate;
    }
    inline float get_momentum() {
        return momentum;
    }
    inline void set_momentum(const float momentum) {
        this->momentum = momentum;
    }
    void adjust_lr();
};