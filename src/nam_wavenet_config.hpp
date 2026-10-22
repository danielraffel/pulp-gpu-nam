#pragma once

// Shared NAM WaveNet (A1) configuration types: the per-layer-array shape parsed
// from a .nam file and the activation set. Used by the scalar oracle
// (nam_model.hpp) and the block engine (nam_wavenet_block.hpp).

#include <cmath>
#include <string>
#include <vector>

namespace pulp::examples::nam {

enum class Activation { Tanh, Hardtanh, Fasttanh, ReLU, Sigmoid, Identity };

inline Activation parse_activation(const std::string& name) {
    if (name == "Tanh") return Activation::Tanh;
    if (name == "Hardtanh") return Activation::Hardtanh;
    if (name == "Fasttanh") return Activation::Fasttanh;
    if (name == "ReLU") return Activation::ReLU;
    if (name == "Sigmoid") return Activation::Sigmoid;
    return Activation::Tanh;   // NAM's default activation
}

inline float apply_activation(Activation a, float x) {
    switch (a) {
        case Activation::Tanh: return std::tanh(x);
        case Activation::Hardtanh: return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
        case Activation::Fasttanh: return std::tanh(x);   // exact path; the approximation is an opt-in optimization

        case Activation::ReLU: return x > 0.0f ? x : 0.0f;
        case Activation::Sigmoid: return 1.0f / (1.0f + std::exp(-x));
        case Activation::Identity: return x;
    }
    return std::tanh(x);
}

// Static per-layer-array configuration parsed from the .nam JSON.
struct LayerArrayConfig {
    int input_size = 1;
    int condition_size = 1;
    int channels = 1;
    int kernel_size = 1;
    std::vector<int> dilations;
    int head_size = 1;
    std::string activation = "Tanh";
    bool gated = false;
    bool head_bias = false;
};

}  // namespace pulp::examples::nam
