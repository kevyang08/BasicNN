#ifndef BASIC_NN_HPP
#define BASIC_NN_HPP

#include <vector>
#include "utils.hpp"

class neural_network {
private:
    float ***layer;
    float ***error;
    float ***weights;
    float ***delta_w;
    std::vector<int> layer_sizes;
    float learning_rate;
    int num_layers;
    int max_size;
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> expected;

public:
    neural_network(int num_layers, std::vector<int>& layer_sizes, float learning_rate, int max_size);
    void forward_propagate();
    void backward_propagate();
    int query();
    void calc_gradient();
    void clear_batch();
    void load_inputs(std::vector<float>& input);
    void load_expected(std::vector<float> exp);
};

#endif