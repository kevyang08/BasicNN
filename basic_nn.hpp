#ifndef BASIC_NN_HPP
#define BASIC_NN_HPP

#include <vector>
#include "utils.hpp"

class neural_network {
private:
    double **layer;
    double **error;
    double ***weights;
    std::vector<int> layer_sizes;
    double learning_rate;
    int num_layers;
    void forward_propagate(std::vector<double>& inputs);
    void backward_propagate(std::vector<double>& expected);

public:
    neural_network(int num_layers, std::vector<int>& layer_sizes, double learning_rate);
    void train(std::vector<double>& inputs, std::vector<double>& expected);
    int query(std::vector<double>& inputs);
};

#endif