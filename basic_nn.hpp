#ifndef BASIC_NN_HPP
#define BASIC_NN_HPP

#include <vector>
#include "utils.hpp"

class neural_network {
private:
    float **layer;
    float **error;
    float ***weights;
    std::vector<int> layer_sizes;
    float learning_rate;
    float momentum;
    int num_layers;
    void forward_propagate(std::vector<float>& inputs);
    void backward_propagate(std::vector<float>& expected);

public:
    neural_network(int num_layers, std::vector<int>& layer_sizes, float learning_rate, float momentum);
    void train(std::vector<float>& inputs, std::vector<float>& expected);
    int query(std::vector<float>& inputs);
    void adjust_lr();
};

#endif