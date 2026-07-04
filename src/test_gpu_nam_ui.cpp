// GPU NAM editor tests: the faithful editor renders a non-blank, asset-composited
// frame through the Skia backend, and pointer input drives real parameters (knob
// drag, Noise-Gate / EQ toggles). Headless — no window, no audio device.

#include <catch2/catch_test_macros.hpp>

#include <pulp/state/store.hpp>
#include <pulp/view/screenshot.hpp>

#include "gpu_nam_processor.hpp"
#include "gpu_nam_ui.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

using namespace pulp;
using namespace pulp::examples;

namespace {

std::unique_ptr<GpuNamProcessor> make_prepared(state::StateStore& store) {
    auto proc = std::make_unique<GpuNamProcessor>();
    proc->set_state_store(&store);
    proc->define_parameters(store);
    format::PrepareContext ctx;
    ctx.sample_rate = 48000.0;
    ctx.max_buffer_size = 512;
    ctx.input_channels = 2;
    ctx.output_channels = 2;
    proc->prepare(ctx);
    return proc;
}

GpuNamUi& as_ui(view::View& v) { return dynamic_cast<GpuNamUi&>(v); }

}  // namespace

TEST_CASE("GPU NAM editor renders a non-blank, asset-composited frame", "[nam][ui]") {
    state::StateStore store;
    auto proc = make_prepared(store);
    auto v = proc->create_view();
    REQUIRE(v);

    // Skia backend composites the reused bitmap/SVG chrome (CoreGraphics can't).
    auto res = view::capture_view(*v, 600, 400, 2.0f, view::ScreenshotBackend::skia);
    INFO("capture reason: " << res.reason);
    REQUIRE(res.ok);                 // passes the content floor — not a blank fill
    REQUIRE(res.png.size() > 5000);  // textured, composited chrome (not a flat rect)
    proc->release();
}

TEST_CASE("GPU NAM editor toggles drive Noise-Gate / EQ parameters", "[nam][ui]") {
    state::StateStore store;
    auto proc = make_prepared(store);
    auto v = proc->create_view();
    auto& ui = as_ui(*v);
    ui.set_bounds({0.0f, 0.0f, 600.0f, 400.0f});

    auto click = [&](view::Point p) { ui.on_mouse_down(p); ui.on_mouse_up(p); };

    const bool ng0 = store.get_value(kNoiseGateActive) >= 0.5f;
    click(ui.toggle_center_for_test(0));
    REQUIRE((store.get_value(kNoiseGateActive) >= 0.5f) != ng0);

    const bool eq0 = store.get_value(kEQActive) >= 0.5f;
    click(ui.toggle_center_for_test(1));
    REQUIRE((store.get_value(kEQActive) >= 0.5f) != eq0);
    proc->release();
}

TEST_CASE("GPU NAM editor knob drag changes the parameter", "[nam][ui]") {
    state::StateStore store;
    auto proc = make_prepared(store);
    auto v = proc->create_view();
    auto& ui = as_ui(*v);
    ui.set_bounds({0.0f, 0.0f, 600.0f, 400.0f});

    // Drag the Input knob (index 0) upward → value increases (up = louder).
    const view::Point c = ui.knob_center_for_test(0);
    const float v0 = store.get_value(kInputGain);
    ui.on_mouse_down(c);
    ui.on_mouse_drag({c.x, c.y - 120.0f});
    ui.on_mouse_up({c.x, c.y - 120.0f});
    REQUIRE(store.get_value(kInputGain) > v0);
    proc->release();
}

// A packed SlimmableContainer with two memoryless variants — enough for the UI to
// show the Slim row and for a variant click to move the size param.
static std::string ui_slim_container() {
    auto v = [](float g) {
        char b[32]; std::snprintf(b, sizeof(b), "%.8g", g);
        return std::string(
            "{\"architecture\":\"WaveNet\",\"sample_rate\":48000,\"config\":{\"layers\":[{"
            "\"input_size\":1,\"condition_size\":1,\"channels\":1,\"bottleneck\":1,"
            "\"kernel_sizes\":[1],\"dilations\":[1],"
            "\"activation\":[{\"type\":\"LeakyReLU\",\"negative_slope\":1.0}],"
            "\"head\":{\"out_channels\":1,\"kernel_size\":1,\"bias\":true}}],\"head_scale\":1.0},"
            "\"weights\":[1,") + b + ",0,0,1,0,1,0,1]}";
    };
    return std::string("{\"architecture\":\"SlimmableContainer\",\"sample_rate\":48000,\"config\":{"
        "\"submodels\":[{\"max_value\":0.5,\"model\":") + v(2.0f) +
        "},{\"max_value\":1.0,\"model\":" + v(3.0f) + "}]},\"weights\":[]}";
}

TEST_CASE("GPU NAM settings Output Mode selector picks each mode", "[nam][ui]") {
    state::StateStore store;
    auto proc = make_prepared(store);
    auto v = proc->create_view();
    auto& ui = as_ui(*v);
    ui.set_bounds({0.0f, 0.0f, 600.0f, 400.0f});
    ui.show_settings(true);
    // Paint once so the settings-page hit rects are computed.
    auto res = view::capture_view(*v, 600, 400, 2.0f, view::ScreenshotBackend::skia);
    if (!res.ok) { WARN("settings render unavailable: " << res.reason); proc->release(); return; }
    auto click = [&](view::Point p) { ui.on_mouse_down(p); ui.on_mouse_up(p); };

    // Default is Normalized (parity with NAM).
    CHECK(store.get_value(kOutputMode) == 1.0f);
    click(ui.output_mode_center_for_test(0));  // Raw
    CHECK(store.get_value(kOutputMode) == 0.0f);
    click(ui.output_mode_center_for_test(2));  // Calibrated
    CHECK(store.get_value(kOutputMode) == 2.0f);
    click(ui.output_mode_center_for_test(1));  // Normalized
    CHECK(store.get_value(kOutputMode) == 1.0f);
    proc->release();
}

TEST_CASE("GPU NAM settings shows a Slim selector only for a multi-variant model", "[nam][ui][a2]") {
    const auto path = (std::filesystem::temp_directory_path() / "gpu_nam_ui_slim.nam").string();
    std::ofstream(path, std::ios::binary) << ui_slim_container();

    state::StateStore store;
    auto proc = std::make_unique<GpuNamProcessor>();
    proc->set_state_store(&store);
    proc->define_parameters(store);
    proc->load_model(path);            // before prepare -> synchronous load
    format::PrepareContext ctx;
    ctx.sample_rate = 48000.0; ctx.max_buffer_size = 512;
    ctx.input_channels = 2; ctx.output_channels = 2;
    proc->prepare(ctx);
    REQUIRE(proc->model_variant_count() == 2);

    auto v = proc->create_view();
    auto& ui = as_ui(*v);
    ui.set_bounds({0.0f, 0.0f, 600.0f, 400.0f});
    ui.show_settings(true);
    auto res = view::capture_view(*v, 600, 400, 2.0f, view::ScreenshotBackend::skia);
    if (!res.ok) { WARN("settings render unavailable: " << res.reason); proc->release();
                   std::filesystem::remove(path); return; }
    REQUIRE(ui.slim_count_for_test() == 2);

    // Default selects Full (size 1.0); clicking Lite drops the size below the 0.5
    // boundary so set_size will pick the smaller variant.
    ui.on_mouse_down(ui.slim_center_for_test(0));  // Lite
    ui.on_mouse_up(ui.slim_center_for_test(0));
    CHECK(store.get_value(kSize) < 0.5f);
    ui.on_mouse_down(ui.slim_center_for_test(1));  // Full
    ui.on_mouse_up(ui.slim_center_for_test(1));
    CHECK(store.get_value(kSize) == 1.0f);

    // The ⓘ next to SLIM SIZE toggles a purely informational popover; it must not
    // touch the size parameter, and closing settings must dismiss it.
    const float size_before = store.get_value(kSize);
    CHECK_FALSE(ui.slim_info_open_for_test());
    ui.on_mouse_down(ui.slim_info_center_for_test());
    ui.on_mouse_up(ui.slim_info_center_for_test());
    CHECK(ui.slim_info_open_for_test());
    CHECK(store.get_value(kSize) == size_before);
    ui.on_mouse_down(ui.slim_info_center_for_test());  // toggle closed
    ui.on_mouse_up(ui.slim_info_center_for_test());
    CHECK_FALSE(ui.slim_info_open_for_test());
    ui.show_slim_info(true);
    ui.show_settings(false);                            // closing settings dismisses it
    CHECK_FALSE(ui.slim_info_open_for_test());
    proc->release();
    std::filesystem::remove(path);
}
