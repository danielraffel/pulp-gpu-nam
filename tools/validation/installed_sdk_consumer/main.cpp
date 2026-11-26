#include <gpu_runtime/gpu_compute.hpp>
#include <gpu_runtime/gpu_surface.hpp>
int main() {
    static_assert(sizeof(gpu_runtime::GpuCompute) > 0);
    static_assert(sizeof(gpu_runtime::GpuSurface) > 0);
    return 0;
}
