#pragma once

#include <filesystem>
#include "Network.h"

namespace NeuralNetworks {

class Saver {
public:
    static bool saveParameters(const Network& network, const std::filesystem::path& path);
    static bool loadParameters(Network& network, const std::filesystem::path& path);
};

}  // namespace NeuralNetworks
