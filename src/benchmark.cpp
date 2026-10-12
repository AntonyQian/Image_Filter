#include "benchmark.h"

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

void print_column_header(std::ostream& progress) {
    progress << std::setw(size_width) << "size"
             << std::setw(sigma_width) << "sigma"
             << std::setw(radius_width) << "radius"
             << std::setw(backend_width) << "backend"
             << std::setw(threads_width) << "threads"
             << "\n";
}

// Ends with a flush so a row shows up as soon as its configuration finishes.
void print_row(std::ostream& progress, const BenchmarkRow& row) {
    progress << std::setw(size_width) << row.width
             << std::setw(sigma_width) << row.sigma
             << std::setw(radius_width) << row.radius
             << std::setw(backend_width) << backend_name(row.backend)
             << std::setw(threads_width) << row.threads
             << "\n"
             << std::flush;
}

// Blurs `source` once with one Backend and thread count and describes the
// configuration as a row.
BenchmarkRow measure(const Image& source, double sigma, Backend backend, int threads) {
    const Image destination = gaussian_blur(source, sigma, backend, threads);

    BenchmarkRow row;
    row.width = destination.width;
    row.height = destination.height;
    row.sigma = sigma;
    row.radius = radius_for_sigma(sigma);
    row.backend = backend;
    row.threads = threads;
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
            rows.push_back(measure(source, sigma, Backend::Serial, 1));
            print_row(progress, rows.back());
        }
    }
    return rows;
}
