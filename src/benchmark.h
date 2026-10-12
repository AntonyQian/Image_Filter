#pragma once

#include <iosfwd>
#include <vector>

#include "gaussian_blur.h"
#include "image.h"

// What to benchmark. Every size is combined with every sigma.
// The values must already be valid: sizes 1 or more, sigmas finite and positive.
struct BenchmarkConfig {
    std::vector<int> sizes = {4096};      // side of the square synthetic Image
    std::vector<double> sigmas = {3.0};
};

// One line of the results table: one Backend at one thread count, on one size
// and sigma.
struct BenchmarkRow {
    int width = 0;
    int height = 0;
    double sigma = 0.0;
    int radius = 0;
    Backend backend = Backend::Serial;
    int threads = 1;
};

// A size x size grayscale Image of pseudo-random Pixels, made in memory.
// The same size always gives the same Pixels, on any compiler.
Image make_synthetic_image(int size);

// Blurs a synthetic Image for every size (outer loop) and sigma (inner loop).
// Writes the column header to `progress`, then each row as soon as it is done.
// Returns the same rows.
//
// Throws what gaussian_blur throws, and std::bad_alloc if an Image does not
// fit in memory.
std::vector<BenchmarkRow> run_benchmark(const BenchmarkConfig& config, std::ostream& progress);
