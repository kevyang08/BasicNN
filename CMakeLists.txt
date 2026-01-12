cmake_minimum_required(VERSION 3.12)

project(nn_mnist_test)

find_package(OpenMP REQUIRED)

add_compile_options(-O3 -fopenmp -march=native -std=c++20 -g)

add_executable(main ./basic_nn.cpp ./basic_nn.hpp ./main.cpp ./utils.cpp ./utils.hpp ./threadpool.hpp)

target_link_libraries(main OpenMP::OpenMP_CXX)