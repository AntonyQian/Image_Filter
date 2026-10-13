#include "benchmark.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <random>

#include <omp.h>

#include "weights.h"

namespace {

// Column widths of the table, shared by the header and the rows.
const int size_width = 6;
const int sigma_width = 8;
const int radius_width = 8;
const int backend_width = 9;
const int threads_width = 9;
const int time_width = 12;
// 12 keeps a gap before the widest 6-digit number, such as 1.23457e-05.
const int scaling_width = 12;

void print_column_header(std::ostream& progress) {
    progress << std::setw(size_width) << "size"
             << std::setw(sigma_width) << "sigma"
             << std::setw(radius_width) << "radius"
             << std::setw(backend_width) << "backend"
             << std::setw(threads_width) << "threads"
             << std::setw(time_width) << "median ms"
             << std::setw(time_width) << "min ms"
             << std::setw(time_width) << "max ms"
             << std::setw(scaling_width) << "speedup"
             << std::setw(scaling_width) << "efficiency"
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
             << std::setw(scaling_width) << row.speedup
             << std::setw(scaling_width) << row.efficiency
             << "\n"
             << std::flush;
}

// The Backends that get a thread Sweep after the Reference row. A new Backend
// is one more entry here.
const Backend swept_backends[] = {Backend::OpenMP};

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

// What measuring one configuration gives.
struct Measurement {
    BenchmarkRow row;   // speedup and efficiency are left for the caller
    Image warm_up;      // the Warm-up's Destination
};

// Measures one configuration: one Backend and thread count on `source` with
// `sigma`. Does one untimed Warm-up, then `runs` timed Runs.
Measurement measure(const Image& source, double sigma, Backend backend, int threads, int runs) {
    Measurement measurement;
    BenchmarkRow& row = measurement.row;
    row.sigma = sigma;
    row.radius = radius_for_sigma(sigma);
    row.backend = backend;
    row.threads = threads;
    row.runs = runs;

    // Warm-up: loads the caches and starts the OpenMP thread pool, so the
    // first timed Run does not pay for them.
    measurement.warm_up = gaussian_blur(source, sigma, backend, threads);
    row.width = measurement.warm_up.width;
    row.height = measurement.warm_up.height;

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
    return measurement;
}

// Fills in speedup and efficiency from the Reference median of the same size
// and sigma. The Reference row itself gets exactly 1 and 1.
void set_scaling(BenchmarkRow& row, double reference_median_ms) {
    row.speedup = reference_median_ms / row.median_ms;
    row.efficiency = row.speedup / row.threads;
}

}  // namespace

std::vector<int> default_thread_counts() {
    std::vector<int> thread_counts;
    for (int threads = 1; threads <= omp_get_max_threads(); ++threads) {
        thread_counts.push_back(threads);
    }
    return thread_counts;
}

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
            Measurement reference = measure(source, sigma, Backend::Serial, 1, config.runs);
            const double reference_median_ms = reference.row.median_ms;
            set_scaling(reference.row, reference_median_ms);
            rows.push_back(reference.row);
            print_row(progress, rows.back());

            // The thread Sweep: every other Backend at every thread count.
            for (const Backend backend : swept_backends) {
                for (const int threads : config.threads) {
                    Measurement measurement = measure(source, sigma, backend, threads, config.runs);
                    set_scaling(measurement.row, reference_median_ms);
                    rows.push_back(measurement.row);
                    print_row(progress, rows.back());
                }
            }
        }
    }
    return rows;
}
