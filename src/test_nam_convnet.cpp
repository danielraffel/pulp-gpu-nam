// Tests for the NAM ConvNet CPU inference, its .nam loader, and NamRuntime
// dispatch. A ConvNet is a stack of blocks (dilated conv kernel 2 -> optional
// batch-norm -> activation) then a linear 1x1 head. The goldens are computed by
// hand from that definition on single-block models where the arithmetic is
// tractable; the causal conv tap order (weight k=0 -> oldest sample) matches the
// A1-validated Conv the engine reuses.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "nam_convnet.hpp"
#include "nam_runtime.hpp"

using namespace pulp::examples::nam;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace {

std::string write_temp(const std::string& name, const std::string& content) {
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary);
    f << content;
    f.close();
    return p.string();
}

std::string floats_json(const std::vector<float>& w) {
    std::string s = "[";
    for (std::size_t i = 0; i < w.size(); ++i) {
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%.8g", w[i]);
        s += buf;
        if (i + 1 < w.size()) s += ",";
    }
    return s + "]";
}

std::string convnet_json(int channels, const std::vector<int>& dilations, bool batchnorm,
                         const std::string& activation, const std::vector<float>& weights) {
    std::string dl = "[";
    for (std::size_t i = 0; i < dilations.size(); ++i)
        dl += std::to_string(dilations[i]) + (i + 1 < dilations.size() ? "," : "");
    dl += "]";
    return std::string("{\"architecture\":\"ConvNet\",\"sample_rate\":48000,\"config\":{")
           + "\"channels\":" + std::to_string(channels) + ",\"dilations\":" + dl
           + ",\"batchnorm\":" + (batchnorm ? "true" : "false")
           + ",\"activation\":\"" + activation + "\",\"in_channels\":1,\"out_channels\":1},\"weights\":"
           + floats_json(weights) + "}";
}

}  // namespace

TEST_CASE("ConvNet forward matches a hand-computed block (no batchnorm)", "[nam][convnet]") {
    // channels=1, one block dilation=1 kernel=2 bias, ReLU, head 1x1.
    // Weights: conv[c0, c1, bc], head[w_h, b_h].
    // conv[n] = c0*x[n-1] + c1*x[n] + bc ; out[n] = w_h*ReLU(conv[n]) + b_h.
    // c0=1,c1=2,bc=0.5,w_h=1,b_h=0.
    NamConvNet m;
    std::string err;
    REQUIRE(m.build(/*in=*/1, /*out=*/1, /*channels=*/1, {1}, /*batchnorm=*/false,
                    Activation::ReLU, {1.0f, 2.0f, 0.5f, 1.0f, 0.0f}, 48000.0, &err));
    m.reset();
    const std::vector<float> in = {1.0f, 0.0f, -1.0f, 3.0f};
    // n0: relu(0+2+0.5)=2.5 ; n1: relu(1+0+0.5)=1.5 ; n2: relu(0-2+0.5)=0 ; n3: relu(-1+6+0.5)=5.5.
    const std::vector<float> golden = {2.5f, 1.5f, 0.0f, 5.5f};
    for (std::size_t i = 0; i < in.size(); ++i)
        CHECK_THAT(m.process_sample(in[i]), WithinAbs(golden[i], 1e-6));
}

TEST_CASE("ConvNet batch-norm applies the affine map", "[nam][convnet]") {
    // channels=1, one block dilation=1, batchnorm, ReLU. Conv has NO bias.
    // BatchNorm weights [mean, var, weight, bias, eps] with mean=0,var=0,eps=1 give
    // scale=weight, loc=bias -> y = weight*conv + bias. conv[n]=x[n] (c0=0,c1=1).
    // out[n] = ReLU(2*x[n] + 1).
    // Weights: conv[0,1], bn[0,0,2,1,1], head[1,0].
    NamConvNet m;
    std::string err;
    REQUIRE(m.build(1, 1, 1, {1}, /*batchnorm=*/true, Activation::ReLU,
                    {0.0f, 1.0f, /*bn*/ 0.0f, 0.0f, 2.0f, 1.0f, 1.0f, /*head*/ 1.0f, 0.0f},
                    48000.0, &err));
    m.reset();
    const std::vector<float> in = {1.0f, -1.0f, 0.0f};
    const std::vector<float> golden = {3.0f, 0.0f, 1.0f};  // relu(2x+1)
    for (std::size_t i = 0; i < in.size(); ++i)
        CHECK_THAT(m.process_sample(in[i]), WithinAbs(golden[i], 1e-6));
}

TEST_CASE("ConvNet loads through NamRuntime dispatch", "[nam][convnet][runtime]") {
    const std::string json = convnet_json(1, {1}, false, "ReLU", {1.0f, 2.0f, 0.5f, 1.0f, 0.0f});
    const std::string path = write_temp("gpu_nam_convnet.nam", json);
    NamRuntime rt;
    std::string err;
    REQUIRE(load_nam_runtime(path, rt, &err));
    REQUIRE(rt.ok());
    CHECK(rt.arch() == NamRuntime::Arch::ConvNet);
    CHECK(std::string(rt.arch_name()) == "ConvNet");
    CHECK_FALSE(rt.gpu_eligible());   // ConvNet GPU forward not wired yet -> CPU
    CHECK(rt.sample_rate() == 48000.0);
    rt.reset();
    CHECK_THAT(rt.process_sample(1.0f), WithinAbs(2.5f, 1e-6));
    CHECK_THAT(rt.process_sample(0.0f), WithinAbs(1.5f, 1e-6));
    std::filesystem::remove(path);
}

TEST_CASE("ConvNet rejects unsupported shapes", "[nam][convnet]") {
    // Grouped conv (via the JSON loader).
    {
        std::string j = convnet_json(1, {1}, false, "ReLU", {1, 2, 0.5f, 1, 0});
        j.insert(j.find("\"in_channels\""), "\"groups\":2,");
        const std::string path = write_temp("gpu_nam_convnet_grp.nam", j);
        NamConvNet m;
        std::string e;
        CHECK_FALSE(load_nam_convnet(path, m, &e));
        CHECK_THAT(e, ContainsSubstring("grouped conv"));
        std::filesystem::remove(path);
    }
    // Weight-count mismatch (too few).
    {
        NamConvNet m;
        std::string e;
        CHECK_FALSE(m.build(1, 1, 1, {1}, false, Activation::ReLU, {1.0f, 2.0f}, 48000.0, &e));
        CHECK_THAT(e, ContainsSubstring("weight count mismatch"));
    }
    // Non-mono in/out.
    {
        NamConvNet m;
        std::string e;
        CHECK_FALSE(m.build(2, 1, 1, {1}, false, Activation::ReLU, {1, 2, 0.5f, 1, 0}, 48000.0, &e));
        CHECK_FALSE(m.build(1, 2, 1, {1}, false, Activation::ReLU, {1, 2, 0.5f, 1, 0}, 48000.0, &e));
    }
}

TEST_CASE("ConvNet loader rejects hostile metadata without allocating", "[nam][convnet][robustness]") {
    NamConvNet m;
    std::string e;

    // A 200-byte file with a colossal `channels` must be rejected at parse — not
    // turned into an 80 GB block allocation. parse_int caps it well before build().
    {
        std::string j = convnet_json(1, {1}, false, "ReLU", {1, 2, 0.5f, 1, 0});
        j.replace(j.find("\"channels\":1"), std::string("\"channels\":1").size(), "\"channels\":100000");
        const std::string path = write_temp("gpu_nam_convnet_bigch.nam", j);
        CHECK_FALSE(load_nam_convnet(path, m, &e));
        CHECK_THAT(e, ContainsSubstring("out-of-range or malformed"));
        std::filesystem::remove(path);
    }
    // 1e100 in an integer field is undefined to cast to int — parse_int rejects it.
    {
        std::string j = convnet_json(4, {1}, false, "ReLU", {1, 2, 0.5f, 1, 0});
        j.replace(j.find("\"channels\":4"), std::string("\"channels\":4").size(), "\"channels\":1e100");
        const std::string path = write_temp("gpu_nam_convnet_ubint.nam", j);
        CHECK_FALSE(load_nam_convnet(path, m, &e));
        std::filesystem::remove(path);
    }
    // A non-string architecture must fail the load, not throw out of the loader.
    {
        std::string j = convnet_json(1, {1}, false, "ReLU", {1, 2, 0.5f, 1, 0});
        j.replace(j.find("\"architecture\":\"ConvNet\""),
                  std::string("\"architecture\":\"ConvNet\"").size(), "\"architecture\":123");
        const std::string path = write_temp("gpu_nam_convnet_archnum.nam", j);
        CHECK_FALSE(load_nam_convnet(path, m, &e));
        CHECK_THAT(e, ContainsSubstring("architecture"));
        std::filesystem::remove(path);
    }
    // An enormous dilations array would fan out into millions of blocks — cap it.
    {
        std::vector<int> many(200, 1);
        std::string j = convnet_json(1, many, false, "ReLU", {1, 2, 0.5f, 1, 0});
        const std::string path = write_temp("gpu_nam_convnet_manylayers.nam", j);
        CHECK_FALSE(load_nam_convnet(path, m, &e));
        CHECK_THAT(e, ContainsSubstring("too many layers"));
        std::filesystem::remove(path);
    }
    // A weight that is a finite double but overflows float (1e300) must be
    // rejected: casting it to float yields ±inf, which would poison the network.
    // Guarding on isfinite(double) alone would miss this — the loader bounds by
    // FLT_MAX before the narrowing cast.
    {
        std::string j =
            "{\"architecture\":\"ConvNet\",\"sample_rate\":48000,\"config\":{"
            "\"channels\":1,\"dilations\":[1],\"batchnorm\":false,\"activation\":\"ReLU\","
            "\"in_channels\":1,\"out_channels\":1},\"weights\":[1.0,2.0,1e300,1.0,0.0]}";
        const std::string path = write_temp("gpu_nam_convnet_ovfw.nam", j);
        CHECK_FALSE(load_nam_convnet(path, m, &e));
        CHECK_THAT(e, ContainsSubstring("non-finite weight"));
        std::filesystem::remove(path);
    }
}
