#pragma once

#include "image.h"

// One implementation of the blur. Every Backend takes the same inputs and
// must produce the same Destination.
enum class Backend {
    Serial,   // one thread, a plain loop over rows: the Reference
};

// Applies a Gaussian blur to `source` and returns the result as a new Image
// (the Destination) of the same size. The Source is never modified.
//
// sigma:   the Gaussian standard deviation in pixels; larger blurs more.
// backend: which implementation does the work.
// threads: how many threads a parallel Backend uses. The serial Backend
//          ignores it.
//
// Pixels past the border are read as the nearest edge pixel (clamp-to-edge).
// Works on any image of at least 1x1, even one smaller than the blur window.
//
// Throws std::invalid_argument if sigma is zero, negative, NaN or infinite.
Image gaussian_blur(const Image& source, double sigma, Backend backend, int threads);
