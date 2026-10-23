// Renders a fixed set of captures through the CPU engine so their output can be
// compared byte for byte against stored hashes (tools/cpu-engine/render_hashes.py).
// A speed change that alters any output sample fails that comparison.
//
// Usage: gpu-nam-golden-render <models dir> <output dir>
// Writes one raw float32 file per capture, tanh mode and block size:
//   <capture>_<mode>_b<block>.f32
//
// Every input is deterministic and platform-independent: the stimulus and the
// synthetic weights come from an integer LCG with single float operations (no
// libm, no multiply-add that a compiler could contract differently).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "nam_lstm.hpp"
#include "nam_model.hpp"
#include "nam_runtime.hpp"

using namespace pulp::examples::nam;

namespace {

constexpr double kRate = 48000.0;
constexpr std::size_t kSamples = 48000 * 2;

float lcg_unit(std::uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<std::int32_t>(s >> 8) % 2001 - 1000) / 1000.0f;
}

// Noise bursts with a linear decay at four levels, then a stretch of silence.
std::vector<float> stimulus() {
    std::vector<float> x(kSamples, 0.0f);
    std::uint32_t s = 0x5eedu;
    constexpr std::size_t kBurst = 4800;
    const float levels[] = {0.05f, 0.25f, 0.7f, 1.0f};
    for (std::size_t i = 0; i < kSamples * 7 / 8; ++i) {
        const float env = static_cast<float>(kBurst - i % kBurst) / static_cast<float>(kBurst);
        const float level = levels[(i / kBurst) % 4];
        x[i] = level * (lcg_unit(s) * env);
    }
    return x;
}

std::vector<float> seeded_weights(std::size_t n, std::uint32_t seed, float scale) {
    std::vector<float> w(n);
    for (auto& v : w) v = lcg_unit(seed) * scale;
    return w;
}

LayerArrayConfig arr(int input, int channels, int head, std::vector<int> dil, bool bias) {
    LayerArrayConfig a;
    a.input_size = input;
    a.condition_size = 1;
    a.channels = channels;
    a.kernel_size = 3;
    a.dilations = std::move(dil);
    a.head_size = head;
    a.head_bias = bias;
    a.gated = false;
    a.activation = "Tanh";
    return a;
}

bool build_wavenet(NamModel& m, const std::vector<LayerArrayConfig>& arrays, std::uint32_t seed) {
    NamModel probe;
    probe.build(arrays, 1.0f, std::vector<float>(1, 0.0f), 1, kRate);
    std::vector<float> w = seeded_weights(probe.expected_weight_count(), seed, 0.35f);
    w.back() = 0.5f;   // head_scale
    return m.build(arrays, 0.5f, std::move(w), 1, kRate);
}

bool build_lstm(NamLstmModel& m, int hidden, int layers, std::uint32_t seed) {
    std::size_t count = 0;
    for (int l = 0; l < layers; ++l) {
        const std::size_t in = l == 0 ? 1u : static_cast<std::size_t>(hidden);
        const std::size_t h = static_cast<std::size_t>(hidden);
        count += 4u * h * (in + h) + 4u * h + 2u * h;
    }
    count += static_cast<std::size_t>(hidden) + 1u;
    return m.build(1, hidden, layers, 1, seeded_weights(count, seed, 0.4f), kRate);
}

template <class Model>
bool render(const Model& prototype, const std::string& name, const std::vector<float>& x, const std::string& out_dir) {
    const std::pair<const char*, TanhMode> modes[] = {{"default", TanhMode::NamRational}, {"exact", TanhMode::Exact}};
    for (const auto& [mode_name, mode] : modes) {
        for (std::uint32_t block : {37u, 64u, 128u, 512u}) {
            Model m = prototype;
            m.set_tanh_mode(mode);
            m.prewarm();
            std::vector<float> y(x.size());
            for (std::size_t pos = 0; pos < x.size(); pos += block) {
                const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(block, x.size() - pos));
                m.process(x.data() + pos, y.data() + pos, n);
            }
            const std::string path = out_dir + "/" + name + "_" + mode_name + "_b" + std::to_string(block) + ".f32";
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(y.data()), static_cast<std::streamsize>(y.size() * sizeof(float)));
            if (!f) {
                std::fprintf(stderr, "cannot write %s\n", path.c_str());
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <models dir> <output dir>\n", argv[0]);
        return 64;
    }
    const std::string models = argv[1], out = argv[2];
    const auto x = stimulus();
    bool ok = true;

    // Trainer preset shapes with seeded weights (compute cost, not a real amp).
    const std::vector<int> d7 = {1, 2, 4, 8, 16, 32, 64};
    const std::vector<int> d13 = {128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    const std::vector<int> d10 = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    struct Preset { const char* name; std::vector<LayerArrayConfig> arrays; std::uint32_t seed; };
    const Preset presets[] = {
        {"synth_nano", {arr(1, 4, 2, d7, false), arr(4, 2, 1, d13, true)}, 101u},
        {"synth_feather", {arr(1, 8, 4, d7, false), arr(8, 4, 1, d13, true)}, 102u},
        {"synth_lite", {arr(1, 12, 6, d7, false), arr(12, 6, 1, d13, true)}, 103u},
        {"synth_standard", {arr(1, 16, 8, d10, false), arr(16, 8, 1, d10, true)}, 104u},
    };
    for (const Preset& p : presets) {
        NamModel m;
        if (!build_wavenet(m, p.arrays, p.seed)) {
            std::fprintf(stderr, "%s: build failed\n", p.name);
            return 1;
        }
        ok = render(m, p.name, x, out) && ok;
    }
    {
        NamLstmModel m;
        if (!build_lstm(m, 24, 3, 105u)) {
            std::fprintf(stderr, "synth_lstm_3x24: build failed\n");
            return 1;
        }
        ok = render(m, "synth_lstm_3x24", x, out) && ok;
    }
    // Bundled real captures, through the same loader the plugin uses.
    for (const char* file : {"wavenet_a1_standard", "lstm", "example"}) {
        NamRuntime m;
        std::string err;
        if (!load_nam_runtime(models + "/" + file + ".nam", m, &err)) {
            std::fprintf(stderr, "%s: %s\n", file, err.c_str());
            return 1;
        }
        ok = render(m, std::string("capture_") + file, x, out) && ok;
    }
    return ok ? 0 : 1;
}
