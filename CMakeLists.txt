cmake_minimum_required(VERSION 3.30)
project(NeuralNetworks)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external/eigen)

include_directories(external/eigen)
include_directories(external/eigenrand)

add_executable(NeuralNetworks main.cpp
        Layer.cpp
        Network.cpp
        Dataset.cpp
        ActivationFunction.cpp
        LossFunction.cpp
        Random.cpp
        NetworkBuilder.cpp
        FileReader.cpp
        Scheduler.cpp
        Optimizer.cpp)

target_compile_features(NeuralNetworks PRIVATE cxx_std_20)
