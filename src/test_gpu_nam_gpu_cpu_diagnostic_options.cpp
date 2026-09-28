#include "gpu_nam_gpu_cpu_diagnostic_options.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using pulp::examples::nam::kDiagnosticDefaultBlocks;
using pulp::examples::nam::kDiagnosticMaxBlocks;
using pulp::examples::nam::parse_diagnostic_blocks;

TEST_CASE("GPU NAM diagnostic defaults to 96 blocks") {
    constexpr std::uint32_t blocks = kDiagnosticDefaultBlocks;
    REQUIRE(blocks == 96);
}

TEST_CASE("GPU NAM diagnostic accepts bounded block counts") {
    std::uint32_t blocks = kDiagnosticDefaultBlocks;
    REQUIRE(parse_diagnostic_blocks("1", blocks));
    REQUIRE(blocks == 1);
    REQUIRE(parse_diagnostic_blocks("1000000", blocks));
    REQUIRE(blocks == kDiagnosticMaxBlocks);
}

TEST_CASE("GPU NAM diagnostic rejects invalid block counts") {
    for (const auto value : {"", "0", "-1", "+1", "1x", "1000001"}) {
        std::uint32_t blocks = kDiagnosticDefaultBlocks;
        REQUIRE_FALSE(parse_diagnostic_blocks(value, blocks));
        REQUIRE(blocks == kDiagnosticDefaultBlocks);
    }
}
