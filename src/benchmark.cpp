#include "benchmark.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <random>

#include "weights.h"

namespace {

// Column widths of the table, shared by the header and the rows.
const int size_width = 6;
const int sigma_width = 8;
const int radius_width = 8;
const int backend_width = 9;
const int threads_width = 9;
const int time_width = 12;

void print_column_header(std::ostream& progress) {
    progress << std::setw(size_width) << "size"
             << std::setw(sigma_width) << "sigma"
             << std::setw(radius_width) << "radius"
             << std::setw(backend_width) << "backend"
             << std::setw(threads_width) << "threads"
             << std::setw(time_width) << "median ms"
             << std::setw(time_width) << "min ms"
             << std::setw(time_width) << "max ms"
             << "\n";
}

// Ends with a flush so a row shows up as soon as its configuration finishes.
void print_row(std::ostream& progress, const BenchmarkRow& row) {
    progress << std::setw(size_width) << row.width
             << std::setw(sigma_width) << row.sigma
             << std::setw(radius_width) << row.radius
             << std::setw(backend_width) << backend_name(row.backend)
             << std::setw(threads_width) << row.threads
             << std::setw(time_width) << row.median_ms
             << std::setw(time_width) << row.min_ms
             << std::setw(time_width) << row.max_ms
             << "\n"
             << std::flush;
}

// The middle value of `sorted_times`, which must be sorted and not empty.
// An even count has no single middle value, so it is the mean of the two
// middle ones.
double median_of(const std::vector<double>& sorted_times) {
    const std::size_t upper = sorted_times.size() / 2;
    if (sorted_times.size() % 2 == 1) {
        return sorted_times[upper];
    }
    return (sorted_times[upper - 1] + sorted_times[upper]) / 2.0;
}

// Measures one configuration: one Backend and thread count on `source` with
// `sigma`. Does one untimed Warm-up, then `runs` timed Runs.
BenchmarkRow measure(const Image& source, double sigma, Backend backend, int threads, int runs) {
    BenchmarkRow row;
    row.sigma = sigma;
    row.radius = radius_for_sigma(sigma);
    row.backend = backend;
    row.threads = threads;
    row.runs = runs;

    {
        // Warm-up: loads the caches and starts the OpenMP thread pool, so the
        // first timed Run does not pay for them.
        const Image warm_up = gaussian_blur(source, sigma, backend, threads);
        row.width = warm_up.width;
        row.height = warm_up.height;
    }

    std::vector<double> times_ms;
    times_ms.reserve(static_cast<std::size_t>(runs));
    for (int run = 0; run < runs; ++run) {
        // steady_clock is monotonic: it never jumps when the system clock is
        // adjusted. Only the blur call is between the two readings.
        const auto start = std::chrono::steady_clock::now();
        const Image destination = gaussian_blur(source, sigma, backend, threads);
        const auto stop = std::chrono::steady_clock::now();

        const std::chrono::duration<double, std::milli> elapsed = stop - start;
        times_ms.push_back(elapsed.count());
        // `destination` is freed here, after the clock was stopped.
    }

    std::sort(times_ms.begin(), times_ms.end());
    row.min_ms = times_ms.front();
    row.max_ms = times_ms.back();
    row.median_ms = median_of(times_ms);
    return row;
}

}  // namespace

Image make_synthetic_image(int size) {
    Image image;
    image.width = size;
    image.height = size;
    image.channels = 1;
    image.data.resize(static_cast<std::size_t>(size) * size);

    // The C++ standard fixes the numbers mt19937 produces for a given seed, so
    // every compiler gives the same Image. A distribution class (such as
    // uniform_int_distribution) has no such guarantee, so the top 8 bits of
    // each 32-bit number are used directly instead.
    std::mt19937 generator(20261012u);   // fixed seed
    for (std::uint8_t& pixel : image.data) {
        pixel = static_cast<std::uint8_t>(generator() >> 24);
    }
    return image;
}

std::vector<BenchmarkRow> run_benchmark(const BenchmarkConfig& config, std::ostream& progress) {
    std::vector<BenchmarkRow> rows;
    print_column_header(progress);

    for (const int size : config.sizes) {
        const Image source = make_synthetic_image(size);

        for (const double sigma : config.sigmas) {
            // The Reference: the serial Backend on one thread.
            rows.push_back(measure(source, sigma, Backend::Serial, 1, config.runs));
            print_row(progress, rows.back());
        }
    }
    return rows;
}
