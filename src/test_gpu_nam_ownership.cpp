#include <catch2/catch_test_macros.hpp>

#include "gpu_nam.hpp"
#include "nam_model.hpp"

#include <pulp/render/gpu_compute.hpp>
#include <pulp/render/gpu_surface.hpp>

#include <filesystem>
#include <string>

using namespace pulp;

namespace {

bool make_surface(std::unique_ptr<render::GpuSurface>& surface) {
    surface = render::GpuSurface::create_dawn();
    if (!surface) return false;
    render::GpuSurface::Config config;
    config.width = 1;
    config.height = 1;
    config.vsync = false;
    return surface->initialize(config);
}

bool load_model(examples::nam::NamModel& model) {
    std::string error;
    const auto path = std::filesystem::path(GPU_NAM_MODELS_DIR) / "example.nam";
    return examples::nam::load_nam(path.string(), model, &error);
}

}  // namespace

TEST_CASE("GPU NAM surface preparation is non-owning", "[nam][gpu][ownership]") {
    std::unique_ptr<render::GpuSurface> surface;
    if (!make_surface(surface)) {
        WARN("no GPU surface available; skipping injected-device lifetime test");
        return;
    }
    auto probe = render::GpuCompute::create();
    if (!probe || !probe->initialize_from_surface(*surface)) {
        WARN("surface has no usable compute device in this sandbox; skipping injected-device lifetime test");
        return;
    }
    examples::nam::NamModel model;
    REQUIRE(load_model(model));

    {
        examples::nam::GpuNam nam;
        REQUIRE(nam.prepare_from_surface(*surface, model, 32));
        REQUIRE(nam.gpu() != nullptr);
        const auto sharing = nam.gpu()->verify_device_sharing(*surface);
        REQUIRE(sharing.device_obtained);
        REQUIRE(sharing.second_consumer_works);
    }

    // Destroying the compute owner must not destroy the injected surface.
    REQUIRE(surface->is_initialized());
}

TEST_CASE("GPU NAM stereo channels retain independent plan slots", "[nam][gpu][stereo]") {
    std::unique_ptr<render::GpuSurface> surface;
    if (!make_surface(surface)) {
        WARN("no GPU surface available; skipping stereo-slot test");
        return;
    }
    examples::nam::NamModel model;
    REQUIRE(load_model(model));

    auto compute = render::GpuCompute::create();
    if (!compute || !compute->initialize_from_surface(*surface)) {
        WARN("surface has no usable compute device in this sandbox; skipping stereo-slot test");
        return;
    }
    examples::nam::GpuNam left;
    examples::nam::GpuNam right;
    REQUIRE(left.prepare_with(*compute, model, 32, 0));
    REQUIRE(right.prepare_with(*compute, model, 32, 1));
    REQUIRE(left.gpu() == right.gpu());
    REQUIRE(left.instance() != right.instance());
}
