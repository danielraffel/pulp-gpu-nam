#include "gpu_nam_paced_options.hpp"
#include <iostream>
#include <limits>
int main() {
    using namespace pulp::examples;
    unsigned count=0;
    auto check=[&](bool good){++count;return good;};
    GpuNamPacedOptions o;
    if (!check(o.valid(32,2))) return 1;
    o.enabled=true;
    if (!check(!o.valid(32,2))) return 2;
    o.sidecar="receipt.csv";
    if (!check(o.valid(32,2)&&o.blocks(32)==15000)) return 3;
    for (const auto arg : {"--blocks=0","--blocks=-1","--blocks=4x","--duration-seconds=0","--sidecar="}) {
        GpuNamPacedOptions invalid;
        if (!check(!invalid.parse(arg))) return 4;
    }
    if (!check(o.parse("--blocks=1000000")&&o.valid(32,2)&&!o.valid(128,2))) return 5;
    if (!check(o.parse("--duration-seconds=10")&&!o.valid(32,2))) return 6;
    for (auto frames : {32u,64u,128u,512u}) {
        if (!check(paced_offset_ns(48000,frames)==std::uint64_t(frames)*1'000'000'000)) return 7;
        if (!check(paced_offset_ns(3,frames)==std::uint64_t(frames)*62500)) return 8;
    }
    if (!check(paced_offset_ns(1,32)==666666&&paced_offset_ns(3,32)==2000000)) return 9;
    if (!check(paced_sample_matches(.2f,.2f)&&!paced_sample_matches(.45f,.2f)&&
               !paced_sample_matches(std::numeric_limits<float>::quiet_NaN(),.2f)&&
               !paced_sample_matches(.2f,std::numeric_limits<float>::infinity()))) return 10;
    std::cout<<count<<" paced option/schedule/oracle controls passed\n";
}
