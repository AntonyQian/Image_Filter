#include "gaussian_blur.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "weights.h"

namespace {

// Computes row y of the Destination. Every Backend goes through here, so the
// sums happen in the same order and the outputs match bit for bit.
void blur_row(const Image& source, const Weights& weights, int y, Image& destination) {
    const int radius = weights.radius;

    for (int x = 0; x < source.width; ++x) {
        float sum = 0.0f;

        for (int dy = -radius; dy <= radius; ++dy) {
            // clamp-to-edge
            const int source_y = std::clamp(y + dy, 0, source.height - 1);

            for (int dx = -radius; dx <= radius; ++dx) {
                const int source_x = std::clamp(x + dx, 0, source.width - 1);

                sum += weights.values[weights.index(dx, dy)] *
                       static_cast<float>(source.data[source.index(source_x, source_y)]);
            }
        }

        // Round to nearest. sum is never negative, so +0.5 then truncate works.
        const int rounded = static_cast<int>(sum + 0.5f);

        // Float error can push a white area just past 255, which would wrap to 0.
        destination.data[destination.index(x, y)] =
            static_cast<std::uint8_t>(std::clamp(rounded, 0, 255));
    }
}

// The Reference.
void blur_serial(const Image& source, const Weights& weights, Image& destination) {
    for (int y = 0; y < source.height; ++y) {
        blur_row(source, weights, y, destination);
    }
}

// Same loop as blur_serial. Each thread writes only its own rows, and source
// and weights are read-only, so there is no race and no lock.
void blur_openmp(const Image& source, const Weights& weights, int threads,
                 Image& destination) {
    // Every row costs the same, so static scheduling is already balanced and
    // keeps each thread on one contiguous block of rows.
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int y = 0; y < source.height; ++y) {
        blur_row(source, weights, y, destination);
    }
}

}  // namespace

Image gaussian_blur(const Image& source, double sigma, Backend backend, int threads) {
    // Written this way so NaN is rejected too.
    if (!(std::isfinite(sigma) && sigma > 0.0)) {
        throw std::invalid_argument("sigma must be a finite number greater than 0");
    }

    if (backend == Backend::OpenMP && threads < 1) {
        throw std::invalid_argument("threads must be 1 or more");
    }

    const Weights weights = make_weights(sigma);

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

const char* backend_name(Backend backend) {
    switch (backend) {
        case Backend::Serial: return "serial";
        case Backend::OpenMP: return "openmp";
    }
    return "unknown";  // not reachable; keeps -Wreturn-type quiet
}
