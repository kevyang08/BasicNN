#ifndef UTILS_HPP
#define UTILS_HPP

#define CACHE_LINE_SIZE 64

const int BLK_SIZE = 64/sizeof(float);

float randd(float l, float r);

float sigmoid(float x);

#endif