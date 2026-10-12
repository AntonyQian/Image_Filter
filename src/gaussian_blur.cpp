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

// The OpenMP Backend: the serial loop again, with one added line that shares
// the rows out between `threads` threads. Nothing else may differ from
// blur_serial, because the two must produce the same Destination bit for bit.
//
// Why there is no data race: every thread only reads `source` and `weights`,
// and the thread that owns row y is the only one that writes row y of
// `destination`. No byte is written by two threads, and no thread reads a
// byte another thread writes, so no lock is needed.
//
// Why the result is bit-identical to serial: each pixel is computed entirely
// inside one blur_row call, by one thread, with the same additions in the
// same order. Threads decide only WHICH rows a thread computes and WHEN,
// never HOW a pixel is computed, and nothing is summed across threads.
void blur_openmp(const Image& source, const Weights& weights, int threads,
                 Image& destination) {
    // schedule(static): the rows are cut into `threads` equal consecutive
    // blocks, decided once before the loop starts. Every row costs the same
    // amount of work, so equal blocks are already balanced, there is no
    // run-time bookkeeping, and each thread walks one contiguous stretch of
    // memory (good for the cache).
    //
    // The loop variable is a signed int: the original OpenMP rules required
    // it, it matches Image's int width and height, and the compiler can
    // assume a signed counter never wraps around. It is declared inside the
    // `for`, which makes it private: each thread has its own y.
    //
    // More threads than rows is fine: the extra threads get no rows.
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int y = 0; y < source.height; ++y) {
        blur_row(source, weights, y, destination);
    }
}

}  // namespace

Image gaussian_blur(const Image& source, double sigma, Backend backend, int threads) {
    // Written as "not (finite and positive)" so that NaN is rejected too:
    // every comparison with NaN is false, so a plain `sigma <= 0` would let
    // it through.
    if (!(std::isfinite(sigma) && sigma > 0.0)) {
        throw std::invalid_argument("sigma must be a finite number greater than 0");
    }

    // A team of zero or fewer threads means nothing, and OpenMP does not
    // define what num_threads does with such a value.
    if (backend == Backend::OpenMP && threads < 1) {
        throw std::invalid_argument("threads must be 1 or more");
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
        case Backend::OpenMP:
            blur_openmp(source, weights, threads, destination);
            break;
    }

    return destination;
}
