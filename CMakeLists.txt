cmake_minimum_required(VERSION 3.12)

project(nn_mnist_test)

add_compile_options(-O3 -march=native -Wall -Wextra -Werror -g)

add_executable(main ./basic_nn.cpp ./basic_nn.hpp ./main.cpp ./utils.cpp ./utils.hpp ./threadpool.hpp)