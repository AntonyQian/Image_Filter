#include "benchmark.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>

#include <omp.h>

#include "weights.h"

namespace {

// Column widths of the table, shared by the header and the rows.
const int size_width = 6;
const int sigma_width = 8;
const int radius_width = 8;
// 9 fits "openmp" with a gap. The 10-letter CUDA names need 12, used only
// when they can appear, so a CPU-only table looks as it always did.
int backend_width(bool show_gpu) {
    return show_gpu ? 12 : 9;
}
const int threads_width = 9;
const int time_width = 12;
// 12 keeps a gap before the widest 6-digit number, such as 1.23457e-05.
const int scaling_width = 12;

// Ends with a flush so piped output shows it before the first row is done.
// `show_gpu` adds the three GPU timing columns.
void print_column_header(std::ostream& progress, bool show_gpu) {
    progress << std::setw(size_width) << "size"
             << std::setw(sigma_width) << "sigma"
             << std::setw(radius_width) << "radius"
             << std::setw(backend_width(show_gpu)) << "backend"
             << std::setw(threads_width) << "threads"
             << std::setw(time_width) << "median ms"
             << std::setw(time_width) << "min ms"
             << std::setw(time_width) << "max ms"
             << std::setw(scaling_width) << "speedup"
             << std::setw(scaling_width) << "efficiency";
    if (show_gpu) {
        progress << std::setw(time_width) << "h2d ms"
                 << std::setw(time_width) << "kernel ms"
                 << std::setw(time_width) << "d2h ms";
    }
    progress << "\n" << std::flush;
}

// One cell: the value, or "-" if there is none.
template <typename T>
void print_cell(std::ostream& progress, int width, const std::optional<T>& value) {
    progress << std::setw(width);
    if (value) {
        progress << *value;
    } else {
        progress << "-";
    }
}

// Ends with a flush so a row shows up as soon as its configuration finishes.
void print_row(std::ostream& progress, const BenchmarkRow& row, bool show_gpu) {
    progress << std::setw(size_width) << row.width
             << std::setw(sigma_width) << row.sigma
             << std::setw(radius_width) << row.radius
             << std::setw(backend_width(show_gpu)) << backend_name(row.backend);
    print_cell(progress, threads_width, row.threads);
    progress << std::setw(time_width) << row.median_ms
             << std::setw(time_width) << row.min_ms
             << std::setw(time_width) << row.max_ms
             << std::setw(scaling_width) << row.speedup;
    print_cell(progress, scaling_width, row.efficiency);
    if (show_gpu) {
        if (row.gpu_timings) {
            progress << std::setw(time_width) << row.gpu_timings->h2d_ms
                     << std::setw(time_width) << row.gpu_timings->kernel_ms
                     << std::setw(time_width) << row.gpu_timings->d2h_ms;
        } else {
            progress << std::setw(time_width) << "-"
                     << std::setw(time_width) << "-"
                     << std::setw(time_width) << "-";
        }
    }
    progress << "\n" << std::flush;
}

// The middle value of `sorted_times`, which must be sorted and not empty.
// An even count has no single middle value, so it is the mean of the two
// middle ones.
double median_of_sorted(const std::vector<double>& sorted_times) {
    const std::size_t upper = sorted_times.size() / 2;
    if (sorted_times.size() % 2 == 1) {
        return sorted_times[upper];
    }
    return (sorted_times[upper - 1] + sorted_times[upper]) / 2.0;
}

// The same for times not sorted yet (taken by value, so it can sort a copy).
double median_of(std::vector<double> times) {
    std::sort(times.begin(), times.end());
    return median_of_sorted(times);
}

// What measuring one configuration gives.
struct Measurement {
    BenchmarkRow row;            // speedup and efficiency are left for the caller
    Image warm_up_destination;   // what the Warm-up wrote
};

// Measures one configuration: one Backend and thread count on `source` with
// `sigma`. Does one untimed Warm-up, then `runs` timed Runs. A CUDA Backend
// ignores `threads`; its row gets no thread count but gets the GPU timings.
Measurement measure(const Image& source, double sigma, Backend backend, int threads, int runs) {
    const bool gpu = is_cuda(backend);

    Measurement measurement;
    BenchmarkRow& row = measurement.row;
    row.sigma = sigma;
    row.radius = radius_for_sigma(sigma);
    row.backend = backend;
    row.threads = gpu ? std::nullopt : std::optional<int>(threads);
    row.runs = runs;

    // Warm-up: loads the caches and starts the OpenMP thread pool, or, for a
    // CUDA Backend, creates the CUDA context and loads the Kernels onto the
    // GPU, so the first timed Run does not pay for them.
    measurement.warm_up_destination = gaussian_blur(source, sigma, backend, threads);
    row.width = measurement.warm_up_destination.width;
    row.height = measurement.warm_up_destination.height;

    std::vector<double> times_ms;
    std::vector<double> h2d_ms;      // the three GPU parts, CUDA only
    std::vector<double> kernel_ms;
    std::vector<double> d2h_ms;
    times_ms.reserve(static_cast<std::size_t>(runs));
    for (int run = 0; run < runs; ++run) {
        GpuTimings timings;
        // steady_clock is monotonic: it never jumps when the system clock is
        // adjusted. Only the blur call is between the two readings.
        const auto start = std::chrono::steady_clock::now();
        const Image destination =
            gaussian_blur(source, sigma, backend, threads, gpu ? &timings : nullptr);
        const auto stop = std::chrono::steady_clock::now();

        const std::chrono::duration<double, std::milli> elapsed = stop - start;
        times_ms.push_back(elapsed.count());
        if (gpu) {
            h2d_ms.push_back(timings.h2d_ms);
            kernel_ms.push_back(timings.kernel_ms);
            d2h_ms.push_back(timings.d2h_ms);
        }
        // `destination` is freed here, after the clock was stopped.
    }

    std::sort(times_ms.begin(), times_ms.end());
    row.min_ms = times_ms.front();
    row.max_ms = times_ms.back();
    row.median_ms = median_of_sorted(times_ms);
    if (gpu) {
        // Each part's own median. The three need not come from the same Run,
        // so they need not add up to exactly the median total.
        GpuTimings medians;
        medians.h2d_ms = median_of(h2d_ms);
        medians.kernel_ms = median_of(kernel_ms);
        medians.d2h_ms = median_of(d2h_ms);
        row.gpu_timings = medians;
    }
    return measurement;
}

// Fills in speedup and efficiency from the Reference median of the same size
// and sigma. The Reference row itself gets exactly 1 and 1. A CUDA row has no
// thread count, so no efficiency: speedup per thread means nothing on a GPU.
void set_scaling(BenchmarkRow& row, double reference_median_ms) {
    row.speedup = reference_median_ms / row.median_ms;
    if (row.threads) {
        row.efficiency = row.speedup / *row.threads;
    } else {
        row.efficiency = std::nullopt;
    }
}

// Checks that a configuration's Warm-up Destination is byte for byte the
// Reference Destination of the same size and sigma. Throws std::runtime_error
// naming the configuration if it is not, so no timing is reported for a
// Backend that gives the wrong answer. Called outside every timed region.
void verify_against_reference(const Measurement& measurement, const Image& reference_destination) {
    const Image& destination = measurement.warm_up_destination;
    if (destination.width == reference_destination.width &&
        destination.height == reference_destination.height &&
        destination.channels == reference_destination.channels &&
        destination.data == reference_destination.data) {
        return;
    }

    const BenchmarkRow& row = measurement.row;
    std::ostringstream message;
    message << "verification failed: size " << reference_destination.width
            << ", sigma " << row.sigma
            << ", backend " << backend_name(row.backend);
    if (row.threads) {
        message << ", threads " << *row.threads;
    }
    message << ": the Destination differs from the Reference";
    throw std::runtime_error(message.str());
}

}  // namespace

std::vector<int> default_thread_counts() {
    std::vector<int> thread_counts;
    for (int threads = 1; threads <= omp_get_max_threads(); ++threads) {
        thread_counts.push_back(threads);
    }
    return thread_counts;
}

std::vector<Backend> default_bench_backends() {
    if (cuda_built()) {
        return {Backend::OpenMP, Backend::CudaNaive, Backend::CudaTiled};
    }
    return {Backend::OpenMP};
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

std::vector<BenchmarkRow> run_benchmark(const BenchmarkConfig& config, std::ostream& progress,
                                        const RowCallback& on_row) {
    std::vector<BenchmarkRow> rows;
    const bool show_gpu = std::any_of(config.backends.begin(), config.backends.end(), is_cuda);
    print_column_header(progress, show_gpu);

    // A finished row: kept, handed to the caller, then shown.
    const auto report = [&](const BenchmarkRow& row) {
        rows.push_back(row);
        if (on_row) {
            on_row(row);
        }
        print_row(progress, row, show_gpu);
    };

    for (const int size : config.sizes) {
        const Image source = make_synthetic_image(size);

        for (const double sigma : config.sigmas) {
            // The Reference: the serial Backend on one thread. Its Warm-up
            // Destination is kept to verify the rest of this size and sigma.
            Measurement reference = measure(source, sigma, Backend::Serial, 1, config.runs);
            const double reference_median_ms = reference.row.median_ms;
            set_scaling(reference.row, reference_median_ms);
            report(reference.row);

            // One configuration after the Reference: measured, verified, then
            // reported. A wrong Destination gets no timings.
            const auto measure_one = [&](Backend backend, int threads) {
                Measurement measurement = measure(source, sigma, backend, threads, config.runs);
                verify_against_reference(measurement, reference.warm_up_destination);
                set_scaling(measurement.row, reference_median_ms);
                report(measurement.row);
            };

            // Then each requested Backend, in the order given. OpenMP gets a
            // thread Sweep; a CUDA Backend gets one row, since its parallelism
            // comes from the fixed block size and grid, not a thread count.
            for (const Backend backend : config.backends) {
                if (is_cuda(backend)) {
                    measure_one(backend, 1);   // the thread count is ignored
                    continue;
                }
                for (const int threads : config.thread_counts) {
                    measure_one(backend, threads);
                }
            }
        }
    }
    return rows;
}
