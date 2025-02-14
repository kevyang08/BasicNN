#include <cmath>
#include <cassert>
#include <algorithm>
#include <omp.h>
#include "basic_nn.hpp"
#include "threadpool.hpp"

neural_network::neural_network(int num_layers, std::vector<int>& layer_sizes, double learning_rate) {
    assert(num_layers == layer_sizes.size());
    this -> num_layers = num_layers;
    for (int i = 0; i < num_layers; i++) {
        layer.push_back(std::vector<double>(layer_sizes[i]));
        error.push_back(std::vector<double>(layer_sizes[i]));
    }
    // Xavier initialization
    weights.resize(num_layers - 1);
    for (int i = 1; i < num_layers; i++) {
        weights[i - 1].resize(layer_sizes[i - 1]);
        for (int j = 0; j < layer_sizes[i - 1]; j++) {
            for (int k = 0; k < layer_sizes[i]; k++) {
                double bounds = sqrt(6)/sqrt(layer_sizes[i - 1] + layer_sizes[i]);
                weights[i - 1][j].push_back(randd(-bounds, bounds));
            }
        }
    }
    this -> learning_rate = learning_rate;
}

void neural_network::forward_propagate(std::vector<double>& inputs) {
    assert(inputs.size() == layer[0].size());
    for (int i = 0; i < inputs.size(); i++) {
        layer[0][i] = inputs[i];
    }
    for (int k = 1; k < num_layers; k++) {
        threadpool tp(8);
        for (int j = 0; j < layer[k].size(); j++) {
            layer[k][j] = 0;
            tp.enqueue([k, j, this]() {
                # pragma omp parallel for
                for (int i = 0; i < layer[k - 1].size(); i++) {
                    layer[k][j] += layer[k - 1][i] * weights[k - 1][i][j];
                }
                layer[k][j] = sigmoid(layer[k][j]);
            });
        }
    }
}

void neural_network::backward_propagate(std::vector<double>& expected) {
    assert(expected.size() == layer[num_layers - 1].size());
    for (int i = 0; i < expected.size(); i++) {
        error[num_layers - 1][i] = expected[i] - layer[num_layers - 1][i];
    }
    for (int k = num_layers - 1; k > 0; k--) {
        threadpool tp(8);
        for (int i = 0; i < layer[k - 1].size(); i++) {
            error[k - 1][i] = 0;
            tp.enqueue([k, i, this]() {
                # pragma omp parallel for
                for (int j = 0; j < layer[k].size(); j++) {
                    error[k - 1][i] += weights[k - 1][i][j] * error[k][j];
                    weights[k - 1][i][j] += learning_rate * error[k][j] * layer[k][j] * (1 - layer[k][j]) * layer[k - 1][i];
                }
            });
        }
    }
}

void neural_network::train(std::vector<double>& inputs, std::vector<double>& expected) {
    forward_propagate(inputs);
    backward_propagate(expected);
}

int neural_network::query(std::vector<double>& inputs) {
    forward_propagate(inputs);
    return std::max_element(layer[num_layers - 1].begin(), layer[num_layers - 1].end()) - layer[num_layers - 1].begin();
}