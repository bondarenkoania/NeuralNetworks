cmake_minimum_required(VERSION 3.30)
project(NeuralNetworks)

add_compile_definitions(SOURCE_DIR="${CMAKE_SOURCE_DIR}")

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
        AMSGradOptimizer.cpp
        Optimizer.cpp
        SimpleOptimizer.cpp
        Saver.cpp
        tests/test_network.cpp
)

target_compile_features(NeuralNetworks PRIVATE cxx_std_20)

option(BUILD_TESTS "Build tests" ON)
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(external/Catch2)

    add_executable(NeuralNetworksTests
            tests/test_main.cpp
            tests/test_layer.cpp
            tests/test_dataset.cpp
            Layer.cpp
            Network.cpp
            Dataset.cpp
            ActivationFunction.cpp
            LossFunction.cpp
            Random.cpp
            NetworkBuilder.cpp
            FileReader.cpp
            AMSGradOptimizer.cpp
            Optimizer.cpp
            SimpleOptimizer.cpp
            Saver.cpp
    )

    target_compile_features(NeuralNetworksTests PRIVATE cxx_std_20)
    target_include_directories(NeuralNetworksTests PRIVATE external/eigen external/eigenrand)

    target_link_libraries(NeuralNetworksTests PRIVATE Catch2::Catch2WithMain)

    add_test(NAME NeuralNetworksTests COMMAND NeuralNetworksTests)
endif ()
