#include "basic_nn.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <immintrin.h>
#include <omp.h>

neural_network::neural_network(int num_layers, std::vector<int>& layer_sizes, float learning_rate, float momentum, bool verbose=false) {
    assert(layer_sizes.size() > 0);
    assert(num_layers == layer_sizes.size());
    this -> num_layers = num_layers;
    layer.resize(std::accumulate(layer_sizes.begin(), layer_sizes.end(), 0));
    error.resize(std::accumulate(layer_sizes.begin(), layer_sizes.end(), 0));
    layer_bounds.resize(num_layers + 1);
    std::partial_sum(layer_sizes.begin(), layer_sizes.end(), layer_bounds.begin() + 1);

    // alloc weights
    int w_size = 0;
    weights_bounds.resize(num_layers);
    for (int i = 1; i < num_layers; i++) {
        w_size += layer_sizes[i] * layer_sizes[i - 1];
        weights_bounds[i] = w_size;
    }
    weights.resize(w_size);

    // Xavier initialization
    for (int k = 1; k < num_layers; k++) {
        // weights is transposed; #rows is size of next layer
        float bounds = sqrt(6)/sqrt(layer_sizes[k - 1] + layer_sizes[k]);
        for (int i = weights_bounds[k - 1]; i < weights_bounds[k]; i++) {
            weights[i] = randd(-bounds, bounds);
        }
    }
    this -> learning_rate = learning_rate;
    this -> momentum = momentum;
    this -> verbose = verbose;
}

void neural_network::forward_propagate(std::vector<float>& inputs) {
    assert(inputs.size() == layer_bounds[1]);
    for (int i = 0; i < inputs.size(); i++) {
        layer[i] = inputs[i];
    }
    for (int k = 1; k < num_layers; k++) {
        const int curr_offset = layer_bounds[k];
        const float *prev = &layer[layer_bounds[k - 1]];
        const int n = layer_bounds[k + 1] - layer_bounds[k];
        const int m = layer_bounds[k] - layer_bounds[k - 1];
        const int w_offset = weights_bounds[k - 1];

        for (int j = 0; j < n; j++) {
            const float *w = &weights[w_offset + j * m];
            
#ifdef __AVX512F__
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            int i = 0;
            // SIMD FMA intrinsics
            for (; i + 31 < m; i += 32) {
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(prev + i), _mm512_loadu_ps(w + i), acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(prev + i + 16), _mm512_loadu_ps(w + i + 16), acc1);
            }

            acc0 = _mm512_add_ps(acc0, acc1);
            float sum = _mm512_reduce_add_ps(acc0);

            for (; i < m; i++) {
                sum += prev[i] * w[i];
            }
#elif defined(__AVX__)
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            int i = 0;
            // SIMD FMA intrinsics
            for (; i + 15 < m; i += 16) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(prev + i), _mm256_loadu_ps(w + i), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(prev + i + 8), _mm256_loadu_ps(w + i + 8), acc1);
            }

            acc0 = _mm256_add_ps(acc0, acc1);

            // 1. Fold 8 floats into 4 floats
            __m128 sum_128  = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1)); // [A, B, C, D]

            // 2. Fold 4 floats into 2 floats
            sum_128 = _mm_add_ps(sum_128, _mm_movehdup_ps(sum_128)); // [A + B, B + B, C + D, D + D]

            // 3. Fold 2 floats into the final scalar sum
            sum_128 = _mm_add_ss(sum_128, _mm_movehl_ps(sum_128, sum_128)); // [A + B + C + D, B + B, C + D, D + D]

            // 4. Extract the single final float
            float sum = _mm_cvtss_f32(sum_128);

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

            layer[j + curr_offset] = sigmoid(sum);
        }
    }
}

void neural_network::backward_propagate(std::vector<float>& expected) {
    assert(expected.size() == layer_bounds[num_layers] - layer_bounds[num_layers - 1]);
    const int offset = layer_bounds[num_layers - 1];
    for (int i = 0; i < expected.size(); i++) {
        error[i + offset] = expected[i] - layer[i + offset];
    }
    for (int k = num_layers - 1; k > 0; k--) {
        const int curr_offset = layer_bounds[k];
        const int prev_offset = layer_bounds[k - 1];
        const int w_offset = weights_bounds[k - 1];

        const int n = layer_bounds[k + 1] - layer_bounds[k];
        const int m = layer_bounds[k] - layer_bounds[k - 1];

        std::fill(error.begin() + prev_offset, error.begin() + prev_offset + m, 0);

        for (int j = 0; j < n; j++) {
            const float e_kj = error[j + curr_offset];
            const float dt = learning_rate * e_kj * layer[j + curr_offset] * (1 - layer[j + curr_offset]);

            for (int i = 0; i < m; i++) {
                error[i + prev_offset] += weights[w_offset + j * m + i] * e_kj;
                weights[w_offset + j * m + i] += dt * layer[i + prev_offset];
            }
        }
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
    return std::max_element(layer.begin() + layer_bounds[num_layers - 1], layer.end()) - (layer.begin() + layer_bounds[num_layers - 1]);
}

void neural_network::adjust_lr() {
    learning_rate *= momentum;
}