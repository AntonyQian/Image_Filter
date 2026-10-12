#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// 8-bit pixels stored row by row, channels interleaved.
struct Image {
    int width = 0;
    int height = 0;
    int channels = 1;                 // always 1 for grayscale, kept for later color support
    std::vector<std::uint8_t> data;   // size should be width * height * channels

    // Cast to size_t first so the multiply can't overflow int on huge images.
    std::size_t index(int x, int y, int c = 0) const {
        return (static_cast<std::size_t>(y) * width + x) * channels + c;
    }
};
