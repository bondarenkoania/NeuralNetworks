cmake_minimum_required(VERSION 3.30)
project(NeuralNetworks)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external/eigen)
#add_subdirectory(external/eigenrand)

include_directories(external/eigen)
include_directories(external/eigenrand)

add_executable(NeuralNetworks main.cpp)
