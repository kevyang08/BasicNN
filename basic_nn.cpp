#include <cmath>
#include <cassert>
#include <algorithm>
#include "basic_nn.hpp"
#include "threadpool.hpp"

neural_network::neural_network(int num_layers, std::vector<int>& layer_sizes, float learning_rate, int max_size) {
    assert(num_layers == layer_sizes.size());
    this -> num_layers = num_layers;
    layer = (float ***)malloc(sizeof(float **) * max_size);
    error = (float ***)malloc(sizeof(float **) * max_size);
    for (int k = 0; k < max_size; k++) {
        layer[k] = (float **)malloc(sizeof(float *) * num_layers);
        error[k] = (float **)malloc(sizeof(float *) * num_layers);
        for (int i = 0; i < num_layers; i++) {
            layer[k][i] = (float *)calloc(layer_sizes[i], sizeof(float));
            error[k][i] = (float *)calloc(layer_sizes[i], sizeof(float));
        }
    }
    // Xavier initialization
    delta_w = (float ***)malloc(sizeof(float **) * (num_layers - 1));
    weights = (float ***)malloc(sizeof(float **) * (num_layers - 1));
    for (int i = 1; i < num_layers; i++) {
        delta_w[i - 1] = (float **)malloc(sizeof(float *) * layer_sizes[i - 1]);
        weights[i - 1] = (float **)malloc(sizeof(float *) * layer_sizes[i - 1]);
        float bounds = sqrt(6)/sqrt(layer_sizes[i - 1] + layer_sizes[i]);
        for (int j = 0; j < layer_sizes[i - 1]; j++) {
            delta_w[i - 1][j] = (float *)calloc(layer_sizes[i], sizeof(float));
            weights[i - 1][j] = (float *)malloc(sizeof(float) * layer_sizes[i]);
            for (int k = 0; k < layer_sizes[i]; k++) {
                weights[i - 1][j][k] = randd(-bounds, bounds);
            }
        }
    }
    this -> learning_rate = learning_rate;
    this -> layer_sizes = layer_sizes;
    this -> max_size = max_size;
}

void neural_network::forward_propagate() {
    for (int b = 0; b < inputs.size(); b++) {
        assert(inputs[b].size() == layer_sizes[0]);
        for (int i = 0; i < inputs[b].size(); i++) {
            layer[b][0][i] = inputs[b][i];
        }
    }
    threadpool tp(8);
    for (int b = 0; b < inputs.size(); b++) {
        tp.enqueue([b, this]() {
            for (int k = 1; k < num_layers; k++) {
                for (int i = 0; i < layer_sizes[k - 1]; i++) {
                    for (int j = 0; j < layer_sizes[k]; j++) {
                        layer[b][k][j] += layer[b][k - 1][i] * weights[k - 1][i][j];
                    }
                }
                for (int j = 0; j < layer_sizes[k]; j++) {
                    layer[b][k][j] = sigmoid(layer[b][k][j]);
                }
            }
        });
    }
}

void neural_network::backward_propagate() {
    for (int k = num_layers - 1; k > 0; k--) {
        for (int i = 0; i < layer_sizes[k - 1]; i++) {
            for (int j = 0; j < layer_sizes[k]; j++) {
                weights[k - 1][i][j] += delta_w[k - 1][i][j]/inputs.size();
            }
            for (int j = 0; j < layer_sizes[k]; j++) {
                delta_w[k - 1][i][j] = 0;
            }
        }
    }
}

int neural_network::query() {
    forward_propagate();
    return std::max_element(layer[0][num_layers - 1], layer[0][num_layers - 1] + layer_sizes[num_layers - 1]) - layer[0][num_layers - 1];
}

void neural_network::calc_gradient() {
    for (int b = 0; b < expected.size(); b++) {
        assert(expected[b].size() == layer_sizes[num_layers - 1]);
        for (int i = 0; i < expected[b].size(); i++) {
            error[b][num_layers - 1][i] = expected[b][i] - layer[b][num_layers - 1][i];
        }
    }
    threadpool tp(8);
    for (int b = 0; b < expected.size(); b++) {
        tp.enqueue([b, this]() {
            for (int k = num_layers - 1; k > 0; k--) {
                for (int i = 0; i < layer_sizes[k - 1]; i++) {
                    error[b][k - 1][i] = 0;
                    for (int j = 0; j < layer_sizes[k]; j++) {
                        error[b][k - 1][i] += weights[k - 1][i][j] * error[b][k][j];
                    }
                    for (int j = 0; j < layer_sizes[k]; j++) {
                        delta_w[k - 1][i][j] += learning_rate * error[b][k][j] * layer[b][k][j] * (1 - layer[b][k][j]) * layer[b][k - 1][i];
                    }
                }
            }
        });
    }
}

void neural_network::clear_batch() {
    inputs.clear();
    expected.clear();
}

void neural_network::load_inputs(std::vector<float>& input) {
    inputs.push_back(input);
}

void neural_network::load_expected(std::vector<float> exp) {
    expected.push_back(exp);
}