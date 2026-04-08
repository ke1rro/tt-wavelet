#include <bits>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

namespace ttwv::utils {

    [[nodiscard]] uint32_t float_to_u32(const float value) noexcept {
        return std::bit_cast<uint32_t>(value);
    }

    [[nodiscard]] std::filesystem:path resolve (const std::filesystem::path& root, const char* relative) {
        return root / relative;
    }
}