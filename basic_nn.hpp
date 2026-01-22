#ifndef BASIC_NN_HPP
#define BASIC_NN_HPP

#include "utils.hpp"
#include "threadpool.hpp"
#include <vector>

#define NUM_THREADS 8

class neural_network {
private:
    bool verbose;
    float **layer;
    float **error;
    float ***weights;
    std::vector<int> layer_sizes;
    float learning_rate;
    float momentum;
    int num_layers;
    void forward_propagate(std::vector<float>& inputs);
    void backward_propagate(std::vector<float>& expected);
    threadpool tp{NUM_THREADS};

public:
    neural_network(int num_layers, std::vector<int>& layer_sizes, float learning_rate, float momentum, bool verbose);
    // void train(std::vector<float>& inputs, std::vector<float>& expected);
    void train(std::vector<std::pair<int, std::vector<float>>>& data, int epochs);
    int query(std::vector<float>& inputs);
    void adjust_lr();
};

#endif