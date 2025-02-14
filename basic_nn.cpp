#include <cmath>
#include <cassert>
#include <algorithm>
#include <omp.h>
#include "basic_nn.hpp"
#include "threadpool.hpp"

neural_network::neural_network(int num_layers, std::vector<int>& layer_sizes, double learning_rate) {
    assert(num_layers == layer_sizes.size());
    this -> num_layers = num_layers;
    layer = (double **)malloc(sizeof(double *) * num_layers);
    error = (double **)malloc(sizeof(double *) * num_layers);
    for (int i = 0; i < num_layers; i++) {
        layer[i] = (double *)calloc(layer_sizes[i], sizeof(double));
        error[i] = (double *)calloc(layer_sizes[i], sizeof(double));
    }
    // Xavier initialization
    weights = (double ***)malloc(sizeof(double **) * (num_layers - 1));
    for (int i = 1; i < num_layers; i++) {
        weights[i - 1] = (double **)malloc(sizeof(double *) * layer_sizes[i - 1]);
        for (int j = 0; j < layer_sizes[i - 1]; j++) {
            weights[i - 1][j] = (double *)malloc(sizeof(double) * layer_sizes[i]);
            double bounds = sqrt(6)/sqrt(layer_sizes[i - 1] + layer_sizes[i]);
            # pragma omp parallel for
            for (int k = 0; k < layer_sizes[i]; k++) {
                weights[i - 1][j][k] = randd(-bounds, bounds);
            }
        }
    }
    this -> learning_rate = learning_rate;
    this -> layer_sizes = layer_sizes;
}

void neural_network::forward_propagate(std::vector<double>& inputs) {
    assert(inputs.size() == layer_sizes[0]);
    for (int i = 0; i < inputs.size(); i++) {
        layer[0][i] = inputs[i];
    }
    for (int k = 1; k < num_layers; k++) {
        threadpool tp(8);
        for (int j = 0; j < layer_sizes[k]; j++) {
            layer[k][j] = 0;
            tp.enqueue([k, j, this]() {
                # pragma omp parallel for
                for (int i = 0; i < layer_sizes[k - 1]; i++) {
                    layer[k][j] += layer[k - 1][i] * weights[k - 1][i][j];
                }
                layer[k][j] = sigmoid(layer[k][j]);
            });
        }
    }
}

void neural_network::backward_propagate(std::vector<double>& expected) {
    assert(expected.size() == layer_sizes[num_layers - 1]);
    for (int i = 0; i < expected.size(); i++) {
        error[num_layers - 1][i] = expected[i] - layer[num_layers - 1][i];
    }
    for (int k = num_layers - 1; k > 0; k--) {
        threadpool tp(8);
        for (int i = 0; i < layer_sizes[k - 1]; i++) {
            error[k - 1][i] = 0;
            tp.enqueue([k, i, this]() {
                # pragma omp parallel for
                for (int j = 0; j < layer_sizes[k]; j++) {
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
    return std::max_element(layer[num_layers - 1], layer[num_layers - 1] + layer_sizes[num_layers - 1]) - layer[num_layers - 1];
}