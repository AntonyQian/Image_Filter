#pragma once

#include <cstddef>
#include <vector>

// Normalised Gaussian coefficients for a square window centred on the pixel
// being blurred. They sum to 1.
struct Weights {
    int radius = 0;
    std::vector<float> values;   // size is (2 * radius + 1) * (2 * radius + 1)

    // Width (and height) of the square window.
    int side() const { return 2 * radius + 1; }

    // Weight for the neighbour at offset (dx, dy), each in -radius..radius.
    std::size_t index(int dx, int dy) const {
        return static_cast<std::size_t>(dy + radius) * side() + (dx + radius);
    }
};

// ceil(3 * sigma). Beyond 3 sigma a Gaussian holds under 0.3% of its area.
// Expects a finite sigma > 0. Throws std::invalid_argument if the radius or
// the window side would not fit in an int.
int radius_for_sigma(double sigma);

// Expects a finite sigma > 0.
// Throws std::invalid_argument if sigma is too large (see radius_for_sigma).
Weights make_weights(double sigma);
