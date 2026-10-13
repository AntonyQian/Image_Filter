#pragma once

#include "image.h"

enum class Backend {
    Serial,   // the Reference
    OpenMP,   // rows split between threads
};

// The name the --backend flag and the benchmark table use: "serial" or "openmp".
const char* backend_name(Backend backend);

// Blurs `source` and returns the Destination, a new Image of the same size.
// `threads` is only used by the OpenMP Backend, where it must be 1 or more.
// Borders are clamp-to-edge.
//
// Throws std::invalid_argument if sigma is not finite and positive, if it is
// so large the radius doesn't fit in an int, or if OpenMP gets threads < 1.
Image gaussian_blur(const Image& source, double sigma, Backend backend, int threads);
