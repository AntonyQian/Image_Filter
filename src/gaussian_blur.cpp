#include "gaussian_blur.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "weights.h"

// Everything in this unnamed namespace is private to this file.
namespace {

// Computes one row (row `y`) of the Destination from the Source.
//
// This is the only place the blur arithmetic lives. Every Backend calls it
// once per row, so every Backend does exactly the same sums in exactly the
// same order and the results match bit for bit. It only reads `source` and
// only writes row `y` of `destination`, so different rows can safely be
// computed at the same time.
void blur_row(const Image& source, const Weights& weights, int y, Image& destination) {
    const int radius = weights.radius;

    for (int x = 0; x < source.width; ++x) {
        // Accumulate in float: a weight times a pixel is a fraction, and
        // rounding to a whole number at each step would throw the result off.
        float sum = 0.0f;

        // Visit the window around (x, y) in a fixed order: top row first,
        // left to right. Float addition gives slightly different answers in a
        // different order, so the order must never change between Backends.
        for (int dy = -radius; dy <= radius; ++dy) {
            // Clamp-to-edge: a neighbour above the top row is read as the top
            // row, one below the bottom row as the bottom row. Nothing is
            // ever read outside the image, and the edges do not go dark.
            const int source_y = std::clamp(y + dy, 0, source.height - 1);

            for (int dx = -radius; dx <= radius; ++dx) {
                const int source_x = std::clamp(x + dx, 0, source.width - 1);

                sum += weights.values[weights.index(dx, dy)] *
                       static_cast<float>(source.data[source.index(source_x, source_y)]);
            }
        }

        // Round to the nearest whole number. Converting float to int just
        // drops the fraction, so adding 0.5 first turns that into rounding
        // (40.6 + 0.5 = 41.1 -> 41). This works because sum is never negative.
        const int rounded = static_cast<int>(sum + 0.5f);

        // The Weights sum to 1 only as closely as float allows, so a white
        // area could come out a hair above 255. Clamp before narrowing to
        // 8 bits, otherwise 256 would wrap around to 0 (black).
        destination.data[destination.index(x, y)] =
            static_cast<std::uint8_t>(std::clamp(rounded, 0, 255));
    }
}

// The serial Backend (the Reference): one thread, rows from top to bottom.
void blur_serial(const Image& source, const Weights& weights, Image& destination) {
    for (int y = 0; y < source.height; ++y) {
        blur_row(source, weights, y, destination);
    }
}

}  // namespace

Image gaussian_blur(const Image& source, double sigma, Backend backend, int /*threads*/) {
    // Written as "not (finite and positive)" so that NaN is rejected too:
    // every comparison with NaN is false, so a plain `sigma <= 0` would let
    // it through.
    if (!(std::isfinite(sigma) && sigma > 0.0)) {
        throw std::invalid_argument("sigma must be a finite number greater than 0");
    }

    // Built once per blur and then only read.
    const Weights weights = make_weights(sigma);

    // The Destination is a separate Image of the same shape. Blurring in
    // place would be wrong: later pixels would read neighbours that had
    // already been overwritten with blurred values.
    Image destination;
    destination.width = source.width;
    destination.height = source.height;
    destination.channels = source.channels;
    destination.data.resize(source.data.size());

    switch (backend) {
        case Backend::Serial:
            blur_serial(source, weights, destination);
            break;
    }

    return destination;
}
