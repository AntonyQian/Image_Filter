#pragma once

#include <cstddef>
#include <vector>

// The normalised Gaussian coefficients a blur multiplies neighbouring pixels
// by. They cover a square window of (2 * radius + 1) by (2 * radius + 1)
// pixels centred on the pixel being blurred, and they sum to 1.
struct Weights {
    int radius = 0;
    std::vector<float> values;   // size is (2 * radius + 1) * (2 * radius + 1)

    // Width (and height) of the square window.
    int side() const { return 2 * radius + 1; }

    // Where the weight for the neighbour at offset (dx, dy) from the centre
    // sits in `values`. dx and dy each run from -radius to +radius; adding
    // radius shifts them to 0 .. 2 * radius so they can be used as a column
    // and a row, and then it is the same row-by-row formula as Image::index.
    std::size_t index(int dx, int dy) const {
        return static_cast<std::size_t>(dy + radius) * side() + (dx + radius);
    }
};

// How far the blur reaches from the centre pixel: ceil(3 * sigma).
// A Gaussian is never exactly zero, but beyond 3 sigma it holds less than
// 0.3% of its area, so the window is cut off there.
// Expects a valid sigma (finite and greater than zero).
// Throws std::invalid_argument if sigma is so large that the radius, or the
// window width 2 * radius + 1, would not fit in an int.
int radius_for_sigma(double sigma);

// Builds the Weights for one blur.
// Expects a valid sigma (finite and greater than zero).
// Throws std::invalid_argument if sigma is too large (see radius_for_sigma).
Weights make_weights(double sigma);
