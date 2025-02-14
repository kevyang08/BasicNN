#include <random>
#include "utils.hpp"

std::random_device rd;
std::mt19937_64 gen(rd());

double randd(double l, double r) {
    return std::uniform_real_distribution<double>(l, r)(gen);
}

double sigmoid(double x) {
    return 1/(1 + exp(-x));
}