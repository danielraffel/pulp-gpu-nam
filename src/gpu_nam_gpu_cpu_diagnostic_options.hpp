#pragma once

#include <charconv>
#include <cstdint>
#include <string_view>

namespace pulp::examples::nam {

inline constexpr std::uint32_t kDiagnosticDefaultBlocks = 96;
inline constexpr std::uint32_t kDiagnosticMaxBlocks = 1'000'000;

inline bool parse_diagnostic_blocks(std::string_view text,
                                    std::uint32_t& blocks) noexcept {
    if (text.empty()) return false;

    std::uint32_t parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed, 10);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()
        || parsed == 0 || parsed > kDiagnosticMaxBlocks) {
        return false;
    }
    blocks = parsed;
    return true;
}

} // namespace pulp::examples::nam
