#pragma once

#include <functional>
#include <iosfwd>
#include <optional>
#include <vector>

#include "gaussian_blur.h"
#include "image.h"

// Every thread count from 1 to the OpenMP runtime's maximum
// (omp_get_max_threads), in order.
std::vector<int> default_thread_counts();

// The Backends measured after the Reference when --backends is not given:
// openmp, then cuda-naive and cuda-tiled in a CUDA build.
std::vector<Backend> default_bench_backends();

// What to benchmark. Every size is combined with every sigma. Each
// combination gets a Reference row, then for each of `backends` in order:
// a thread Sweep over `thread_counts` for OpenMP, one row for a CUDA Backend.
// The values must already be valid: sizes 1 or more, sigmas finite and
// positive, thread counts 1 or more, runs 1 or more, and every Backend
// accepted by check_backend_supported for every sigma.
struct BenchmarkConfig {
    std::vector<int> sizes = {4096};      // side of the square synthetic Image
    std::vector<double> sigmas = {3.0};
    std::vector<int> thread_counts = default_thread_counts();
    std::vector<Backend> backends = default_bench_backends();   // never Serial
    int runs = 15;                        // timed Runs per configuration
};

// One line of the results table: one Backend at one thread count (or, for a
// CUDA Backend, its one configuration), on one size
// and sigma. Empty optionals are blank in the CSV and "-" in the table.
struct BenchmarkRow {
    int width = 0;
    int height = 0;
    double sigma = 0.0;
    int radius = 0;
    Backend backend = Backend::Serial;
    std::optional<int> threads = 1;           // empty for CUDA: the GPU has no thread count
    int runs = 0;             // timed Runs behind the figures below
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double speedup = 0.0;     // Reference median / this row's median
    std::optional<double> efficiency = 1.0;   // speedup / threads; empty for CUDA
    // CUDA rows only: the median of each of the three parts over the timed Runs.
    std::optional<GpuTimings> gpu_timings;
};

// A size x size grayscale Image of pseudo-random Pixels, made in memory.
// The same size always gives the same Pixels, on any compiler.
Image make_synthetic_image(int size);

// Measures the blur of a synthetic Image for every size (outer loop) and sigma
// (next loop). Each size and sigma gets one Reference row (serial Backend,
// 1 thread), then, for each of config.backends in order, one OpenMP row per
// thread count or one row for a CUDA Backend. Thread counts above the number
// of processors or Image rows are fine.
// Each configuration is one untimed Warm-up, then config.runs timed Runs. A
// Run is the whole gaussian_blur call, so it includes building the Weights
// and allocating the Destination, and nothing else.
// Speedup is the Reference median of the same size and sigma divided by the
// row's median; efficiency is speedup divided by the thread count. A CUDA
// Warm-up also absorbs creating the CUDA context and loading the Kernels.
// The table has three more columns (h2d, kernel, d2h ms) when config.backends
// has a CUDA Backend.
// Before a row is printed, the Destination of its Warm-up is compared byte
// for byte with the Reference Destination of the same size and sigma.
// Writes the column header to `progress`, then each row as soon as it is done,
// flushing after each. Returns the same rows.
// `on_row`, if given, is called with each finished row just before that row
// is written to `progress`, so the caller can record rows as the session goes.
//
// Throws std::runtime_error naming the size, sigma, Backend and thread count
// if a Destination differs from the Reference; the rows before it have
// already been written and passed to `on_row`. Also throws what gaussian_blur
// throws, and std::bad_alloc if an Image does not fit in memory.
using RowCallback = std::function<void(const BenchmarkRow&)>;
std::vector<BenchmarkRow> run_benchmark(const BenchmarkConfig& config, std::ostream& progress,
                                        const RowCallback& on_row = {});
