#include "basic_nn.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>

neural_network::neural_network(int num_layers, std::vector<int>& layer_sizes, float learning_rate, float momentum, bool verbose=false) {
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
    this -> verbose = verbose;
}

void neural_network::forward_propagate(std::vector<float>& inputs) {
    assert(inputs.size() == layer_sizes[0]);
    for (int i = 0; i < inputs.size(); i++) {
        layer[0][i] = inputs[i];
    }
    for (int k = 1; k < num_layers; k++) {
        const float *prev = layer[k - 1];
        const int n = layer_sizes[k];
        const int m = layer_sizes[k - 1];

        for (int j = 0; j < n; j++) {
            const float *w = weights[k - 1][j];
            
#ifdef __AVX__
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            int i = 0;
            // SIMD FMA intrinsics
            for (; i + 15 < m; i += 16) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(prev + i), _mm256_loadu_ps(w + i), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(prev + i + 8), _mm256_loadu_ps(w + i + 8), acc1);
            }

            acc0 = _mm256_add_ps(acc0, acc1);

            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, acc0);

            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

            for (; i < m; i++) {
                sum += prev[i] * w[i];
            }
#else
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum) simdlen(8)
            for (int i = 0; i < m; i++) {
                sum += prev[i] * w[i];
            }
#endif

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
        const int n = layer_sizes[k];
        const int m = layer_sizes[k - 1];

        std::fill(error[k - 1], error[k - 1] + m, 0);

        // for (int j = 0; j < n; j++) {
        //     const float e_kj = error[k][j];
        //     const float dt = learning_rate * e_kj * layer[k][j] * (1 - layer[k][j]);

        //     for (int i = 0; i < m; i++) {
        //         error[k - 1][i] += weights[k - 1][j][i] * e_kj;
        //         weights[k - 1][j][i] += dt * layer[k - 1][i];
        //     }
        // }


        //------------------------------TESTING THREADPOOL VS OPENMP-------------------------------------

        for (int j = 0; j < layer_sizes[k]; j++) { // make separate to prevent need for sync
            for (int i = 0; i < layer_sizes[k - 1]; i++) {
                error[k - 1][i] += weights[k - 1][j][i] * error[k][j];
            }
        }

        const int BLK_SZ = (n + NUM_THREADS - 1)/NUM_THREADS;
        for (int idx = 0; idx < n; idx += BLK_SZ) {
            const int start = idx;
            const int end = std::min(n, start + BLK_SZ);
            tp.enqueue([k, n, m, start, end, this]() {
                for (int j = start; j < end; j++) {
                    const float e_kj = error[k][j];
                    const float dt = learning_rate * e_kj * layer[k][j] * (1 - layer[k][j]);

                    for (int i = 0; i < m; i++) {
                        weights[k - 1][j][i] += dt * layer[k - 1][i];
                    }
                }
            });
        }
        tp.wait_all();

        // const int BLK_SZ = (n + NUM_THREADS - 1)/NUM_THREADS;
        // #pragma omp parallel for num_threads(NUM_THREADS)
        // for (int idx = 0; idx < n; idx += BLK_SZ) {
        //     const int start = idx;
        //     const int end = std::min(n, start + BLK_SZ);
        //     for (int j = start; j < end; j++) {
        //         const float e_kj = error[k][j];
        //         const float dt = learning_rate * e_kj * layer[k][j] * (1 - layer[k][j]);

        //         for (int i = 0; i < m; i++) {
        //             weights[k - 1][j][i] += dt * layer[k - 1][i];
        //         }
        //     }
        // }
    }
}

// void neural_network::train(std::vector<float>& inputs, std::vector<float>& expected) {
//     forward_propagate(inputs);
//     backward_propagate(expected);
// }

void neural_network::train(std::vector<std::pair<int, std::vector<float>>>& data, int epochs) {
    std::cout << "Beginning training with " << num_layers << " layers and a learning rate of " << learning_rate << std::endl;

    auto print_progress = [](int progress) {
        std::cout << "\r[";
        for (int i = 0; i < 50; i++) std::cout << (progress > i * 2 + 1 ? '=' : ' ');
        std::cout << "] " << progress << "%";
    };
    
    std::vector<float> expected(10);

    for (int e = 1; e <= epochs; e++) {

        std::cout << "Epoch " << e << "/" << epochs << std::endl;

        // randomly shuffle training data
        std::shuffle(data.begin(), data.end(), std::random_device());

        // to reduce redundant updates
        int prev_progress = 0;
        if (verbose) {
            print_progress(0);
        }

        for (int i = 0; i < data.size(); i++) {
            auto &[label, values] = data[i];
            std::fill(expected.begin(), expected.end(), 0);
            expected[label] = 1;

            // nn.train(values, expected);
            forward_propagate(values);
            backward_propagate(expected);

            // progress bar
            if (verbose) {
                int progress = (int)((i + 1.0)/data.size() * 100);
                if (progress == prev_progress) continue;
                print_progress(progress);
                prev_progress = progress;
            }
        }

        std::cout << std::endl;

        adjust_lr();
    }
}

int neural_network::query(std::vector<float>& inputs) {
    forward_propagate(inputs);
    return std::max_element(layer[num_layers - 1], layer[num_layers - 1] + layer_sizes[num_layers - 1]) - layer[num_layers - 1];
}

void neural_network::adjust_lr() {
    learning_rate *= momentum;
}