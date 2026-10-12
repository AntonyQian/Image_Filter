#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// A rectangular grid of 8-bit pixels, stored row by row.
// Each pixel holds `channels` bytes next to each other (interleaved).
struct Image {
    int width = 0;
    int height = 0;
    int channels = 1;                 // always 1 for grayscale, kept for later color support
    std::vector<std::uint8_t> data;   // size should be width * height * channels

    // The one place the index formula lives: where channel `c` of pixel (x, y)
    // sits in `data`. For grayscale (channels == 1) this is just y * width + x.
    // The cast to std::size_t comes first so the multiplication is done in a
    // wide type and cannot overflow int on very large images.
    std::size_t index(int x, int y, int c = 0) const {
        return (static_cast<std::size_t>(y) * width + x) * channels + c;
    }
};
