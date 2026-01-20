#include "utils.hpp"
#include <random>

std::random_device rd;
std::mt19937_64 gen(rd());

float randd(float l, float r) {
    return std::uniform_real_distribution<float>(l, r)(gen);
}

float sigmoid(float x) {
    return 1/(1 + exp(-x));
}