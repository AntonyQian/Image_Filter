#pragma once

#include <cstdint>
#include <vector>

// A grayscale image: one byte per pixel, stored row by row.
// Pixel (x, y) lives at index  y * width + x.
struct Image {
    int width = 0;
    int height = 0;
    int channels = 1;                 // always 1 for grayscale, kept for later color support
    std::vector<std::uint8_t> data;   // size should be width * height
};

// TODO: a small helper to get or set a pixel by (x, y) so do not repeat the y * width + x formula everywhere. 