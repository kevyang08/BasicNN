#include "basic_nn.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <immintrin.h>
#include <omp.h>

neural_network::neural_network(const int num_layers,
                               std::vector<int>& layer_sizes,
                               const float learning_rate,
                               const float momentum,
                               std::vector<std::pair<int, std::vector<float>>>& training_data) : num_layers{num_layers}, training_data{training_data} {
    assert(layer_sizes.size() > 0);
    assert(std::all_of(layer_sizes.begin(), layer_sizes.end(), [](int x) { return x > 0; }));
    assert(num_layers == layer_sizes.size());
    expected.resize(layer_sizes.back());
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
    this->learning_rate = learning_rate;
    this->momentum = momentum;
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

            float sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));

            for (; i < m; i++) {
                sum += prev[i] * w[i];
            }
#elif defined(__AVX2__)
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

void neural_network::backward_propagate() {
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

            float *prev_error = &error[prev_offset];
            float *w = &weights[w_offset + j * m];
            const float *prev = &layer[prev_offset];
            
#ifdef __AVX512F__
            __m512 e_kj_avx = _mm512_set1_ps(e_kj);
            __m512 dt_avx = _mm512_set1_ps(dt);

            int i = 0;
            // SIMD FMA intrinsics
            for (; i + 31 < m; i += 32) {
                _mm512_storeu_ps(prev_error + i, _mm512_fmadd_ps(e_kj_avx, _mm512_loadu_ps(w + i), _mm512_loadu_ps(prev_error + i)));
                _mm512_storeu_ps(prev_error + i + 16, _mm512_fmadd_ps(e_kj_avx, _mm512_loadu_ps(w + i + 16), _mm512_loadu_ps(prev_error + i + 16)));
                _mm512_storeu_ps(w + i, _mm512_fmadd_ps(dt_avx, _mm512_loadu_ps(prev + i), _mm512_loadu_ps(w + i)));
                _mm512_storeu_ps(w + i + 16, _mm512_fmadd_ps(dt_avx, _mm512_loadu_ps(prev + i + 16), _mm512_loadu_ps(w + i + 16)));
            }

            for (; i < m; i++) {
                prev_error[i] += w[i] * e_kj;
                w[i] += dt * prev[i];
            }
#elif defined(__AVX2__)
            __m256 e_kj_avx = _mm256_set1_ps(e_kj);
            __m256 dt_avx = _mm256_set1_ps(dt);

            int i = 0;
            // SIMD FMA intrinsics
            for (; i + 15 < m; i += 16) {
                _mm256_storeu_ps(prev_error + i, _mm256_fmadd_ps(e_kj_avx, _mm256_loadu_ps(w + i), _mm256_loadu_ps(prev_error + i)));
                _mm256_storeu_ps(prev_error + i + 8, _mm256_fmadd_ps(e_kj_avx, _mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(prev_error + i + 8)));
                _mm256_storeu_ps(w + i, _mm256_fmadd_ps(dt_avx, _mm256_loadu_ps(prev + i), _mm256_loadu_ps(w + i)));
                _mm256_storeu_ps(w + i + 8, _mm256_fmadd_ps(dt_avx, _mm256_loadu_ps(prev + i + 8), _mm256_loadu_ps(w + i + 8)));
            }

            for (; i < m; i++) {
                prev_error[i] += w[i] * e_kj;
                w[i] += dt * prev[i];
            }
#else
            #pragma omp simd simdlen(8)
            for (int i = 0; i < m; i++) {
                prev_error[i] += w[i] * e_kj;
                w[i] += dt * prev[i];
            }
#endif
        }
    }
}

void neural_network::train(std::function<void(int)> progress_callback = nullptr) {
    // randomly shuffle training data
    std::shuffle(training_data.begin(), training_data.end(), std::random_device());

    for (int i = 0; i < training_data.size(); i++) {
        auto &[label, values] = training_data[i];
        std::fill(expected.begin(), expected.end(), 0);
        expected[label] = 1;

        forward_propagate(values);
        backward_propagate();

        // call progress_callback() if not nullptr
        if (progress_callback) [[unlikely]] {
            progress_callback(i);
        }
    }
}

int neural_network::query(std::vector<float>& inputs) {
    forward_propagate(inputs);
    return std::max_element(layer.begin() + layer_bounds[num_layers - 1], layer.end()) - (layer.begin() + layer_bounds[num_layers - 1]);
}

void neural_network::adjust_lr() {
    learning_rate *= momentum;
}