#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <omp.h>
#include "basic_nn.hpp"

neural_network::neural_network(int num_layers, std::vector<int>& layer_sizes, float learning_rate, float momentum) {
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
        // weights is transposed; #rows is size of next layer
        weights[i - 1] = (float **)malloc(sizeof(float *) * layer_sizes[i]);
        float bounds = sqrt(6)/sqrt(layer_sizes[i - 1] + layer_sizes[i]);
        for (int j = 0; j < layer_sizes[i]; j++) {
            weights[i - 1][j] = (float *)malloc(sizeof(float) * layer_sizes[i - 1]);
            for (int k = 0; k < layer_sizes[i - 1]; k++) {
                weights[i - 1][j][k] = randd(-bounds, bounds);
            }
        }
    }
    this -> learning_rate = learning_rate;
    this -> momentum = momentum;
    this -> layer_sizes = layer_sizes;
}

void neural_network::forward_propagate(std::vector<float>& inputs) {
    assert(inputs.size() == layer_sizes[0]);
    for (int i = 0; i < inputs.size(); i++) {
        layer[0][i] = inputs[i];
    }
    for (int k = 1; k < num_layers; k++) {
        const float *prev = layer[k - 1];
        for (int j = 0; j < layer_sizes[k]; j++) {
            const float *w = weights[k - 1][j];
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum) simdlen(8)
            for (int i = 0; i < layer_sizes[k - 1]; i++) {
                sum += prev[i] * w[i];
            }
            layer[k][j] = sigmoid(sum);
        }
    }
}

void neural_network::backward_propagate(std::vector<float>& expected) {
    assert(expected.size() == layer_sizes[num_layers - 1]);
    for (int i = 0; i < expected.size(); i++) {
        error[num_layers - 1][i] = expected[i] - layer[num_layers - 1][i];
    }
    for (int k = num_layers - 1; k > 0; k--) {
        std::fill(error[k - 1], error[k - 1] + layer_sizes[k - 1], 0);

        for (int j = 0; j < layer_sizes[k]; j++) {
            for (int i = 0; i < layer_sizes[k - 1]; i++) {
                error[k - 1][i] += weights[k - 1][j][i] * error[k][j];
                weights[k - 1][j][i] += learning_rate * error[k][j] * layer[k][j] * (1 - layer[k][j]) * layer[k - 1][i];
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

void neural_network::adjust_lr() {
    learning_rate *= momentum;
}