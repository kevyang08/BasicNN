#include <cmath>
#include <cassert>
#include <algorithm>
#include "basic_nn.hpp"

neural_network::neural_network(int num_layers, std::vector<int>& layer_sizes, float learning_rate) {
    assert(num_layers == layer_sizes.size());
    this -> num_layers = num_layers;
    layer = (float **)malloc(sizeof(float *) * num_layers);
    error = (float **)malloc(sizeof(float *) * num_layers);
    for (int i = 0; i < num_layers; i++) {
        layer[i] = (float *)calloc(layer_sizes[i], sizeof(float));
        error[i] = (float *)calloc(layer_sizes[i], sizeof(float));
    }
    // Xavier initialization
    weights = (float ***)malloc(sizeof(float **) * (num_layers - 1));
    for (int i = 1; i < num_layers; i++) {
        weights[i - 1] = (float **)malloc(sizeof(float *) * layer_sizes[i - 1]);
        float bounds = sqrt(6)/sqrt(layer_sizes[i - 1] + layer_sizes[i]);
        for (int j = 0; j < layer_sizes[i - 1]; j++) {
            weights[i - 1][j] = (float *)malloc(sizeof(float) * layer_sizes[i]);
            for (int k = 0; k < layer_sizes[i]; k++) {
                weights[i - 1][j][k] = randd(-bounds, bounds);
            }
        }
    }
    this -> learning_rate = learning_rate;
    this -> layer_sizes = layer_sizes;
}

void neural_network::forward_propagate(std::vector<float>& inputs) {
    assert(inputs.size() == layer_sizes[0]);
    for (int i = 0; i < inputs.size(); i++) {
        layer[0][i] = inputs[i];
    }
    for (int k = 1; k < num_layers; k++) {
        for (int i = 0; i < layer_sizes[k - 1]; i++) {
            for (int j = 0; j < layer_sizes[k]; j++) {
                layer[k][j] += layer[k - 1][i] * weights[k - 1][i][j];
            }
        }
        for (int j = 0; j < layer_sizes[k]; j++) {
            layer[k][j] = sigmoid(layer[k][j]);
        }
    }
}

void neural_network::backward_propagate(std::vector<float>& expected) {
    assert(expected.size() == layer_sizes[num_layers - 1]);
    for (int i = 0; i < expected.size(); i++) {
        error[num_layers - 1][i] = expected[i] - layer[num_layers - 1][i];
    }
    for (int k = num_layers - 1; k > 0; k--) {
        for (int i = 0; i < layer_sizes[k - 1]; i++) {
            error[k - 1][i] = 0;
            for (int j = 0; j < layer_sizes[k]; j++) {
                error[k - 1][i] += weights[k - 1][i][j] * error[k][j];
            }
            for (int j = 0; j < layer_sizes[k]; j++) {
                weights[k - 1][i][j] += learning_rate * error[k][j] * layer[k][j] * (1 - layer[k][j]) * layer[k - 1][i];
            }
        }
    }
}

void neural_network::train(std::vector<float>& inputs, std::vector<float>& expected) {
    forward_propagate(inputs);
    backward_propagate(expected);
}

int neural_network::query(std::vector<float>& inputs) {
    forward_propagate(inputs);
    return std::max_element(layer[num_layers - 1], layer[num_layers - 1] + layer_sizes[num_layers - 1]) - layer[num_layers - 1];
}