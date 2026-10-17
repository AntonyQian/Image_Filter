#include "cli.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <new>
#include <optional>
#include <ostream>
#include <stdexcept>

#include <omp.h>

#include "benchmark.h"
#include "gaussian_blur.h"
#include "image.h"
#include "io.h"
#include "weights.h"

namespace {

// Only checks that the text is a number; gaussian_blur checks the range.
double parse_sigma(const std::string& text) {
    try {
        // stod stops at the first bad character, so "3abc" would pass as 3
        // without the consumed check.
        std::size_t consumed = 0;
        const double sigma = std::stod(text, &consumed);
        if (consumed == text.size()) {
            return sigma;
        }
    } catch (const std::exception&) {
        // fall through to the error below
    }
    throw std::invalid_argument("sigma must be a number, got '" + text + "'");
}

const char* const usage =
    "Usage: imgfilter <input image> <output.png> <sigma> "
    "[--backend serial|openmp|cuda-naive|cuda-tiled] [--threads N]\n"
    "       imgfilter bench [--sizes N,N,...] [--sigmas S,S,...] [--threads N,N,...]\n"
    "                       [--backends B,B,...] [--runs N] [--csv <results.csv>]\n"
    "\n"
    "The first form blurs one image. --threads is only used by openmp. The\n"
    "cuda-naive and cuda-tiled backends run on an NVIDIA GPU and need a build\n"
    "configured with -DIMGFILTER_CUDA=ON; cuda-tiled allows sigma up to 21.\n"
    "\n"
    "bench times the blur of a synthetic square image of each size with each\n"
    "sigma: the serial backend first, then each of --backends in the order\n"
    "given (from openmp, cuda-naive, cuda-tiled), openmp at each thread count\n"
    "and a cuda backend once. Each is one untimed warm-up, then --runs timed\n"
    "runs. Defaults: --sizes 4096 --sigmas 3 --runs 15, --threads every count\n"
    "from 1 to the maximum, --backends openmp (openmp,cuda-naive,cuda-tiled in\n"
    "a CUDA build). --csv also writes the rows of the table to a file,\n"
    "replacing it if it exists.";

struct Options {
    Backend backend = Backend::OpenMP;
    int threads = omp_get_max_threads();
};

// Every Backend, in the order the usage text lists them.
const Backend all_backends[] = {Backend::Serial, Backend::OpenMP, Backend::CudaNaive,
                                Backend::CudaTiled};

// The Backend whose name is `text`, or nothing.
std::optional<Backend> find_backend(const std::string& text) {
    for (const Backend backend : all_backends) {
        if (text == backend_name(backend)) {
            return backend;
        }
    }
    return std::nullopt;
}

Backend parse_backend(const std::string& text) {
    if (const std::optional<Backend> backend = find_backend(text)) {
        return *backend;
    }
    throw std::invalid_argument("unknown backend '" + text +
                                "', valid values are: serial, openmp, cuda-naive, cuda-tiled");
}

// True, with the number in `value`, if `text` is a whole number, 1 or more,
// that fits in an int.
bool to_positive_int(const std::string& text, int& value) {
    try {
        // consumed rejects "4abc" and "2.5"
        std::size_t consumed = 0;
        value = std::stoi(text, &consumed);
        return consumed == text.size() && value >= 1;
    } catch (const std::exception&) {
        return false;   // not a number, or too big for an int
    }
}

// The value of a flag such as --threads 4 or --runs 15.
int parse_positive_int(const std::string& flag, const std::string& text) {
    int value = 0;
    if (!to_positive_int(text, value)) {
        throw std::invalid_argument(
            flag + " must be a whole number, 1 or more, got '" + text + "'");
    }
    return value;
}

// Flags come after the three positional arguments, each followed by its value.
Options parse_options(const std::vector<std::string>& args) {
    Options options;

    for (std::size_t i = 3; i < args.size(); i += 2) {
        const std::string& flag = args[i];
        if (flag != "--backend" && flag != "--threads") {
            throw std::invalid_argument("unknown argument '" + flag + "'\n" + usage);
        }

        if (i + 1 >= args.size()) {
            throw std::invalid_argument(flag + " needs a value\n" + usage);
        }
        const std::string& value = args[i + 1];

        if (flag == "--backend") {
            options.backend = parse_backend(value);
        } else {
            options.threads = parse_positive_int(flag, value);
        }
    }

    // Serial ignores --threads, so report the count actually used. The CUDA
    // Backends ignore it too, and the summary shows no thread count for them.
    if (options.backend == Backend::Serial) {
        options.threads = 1;
    }
    return options;
}

// ---- bench subcommand ----

// Splits "16,37" into {"16", "37"}. Keeps empty items ("16,,37" has one), so
// the caller rejects them as not a number. An empty text gives no items.
std::vector<std::string> split_list(const std::string& text) {
    std::vector<std::string> items;
    if (text.empty()) {
        return items;
    }
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = text.find(',', start);
        if (comma == std::string::npos) {
            items.push_back(text.substr(start));
            return items;
        }
        items.push_back(text.substr(start, comma - start));
        start = comma + 1;
    }
}

// The value of a flag such as --sizes 1024,2048. `example` is a valid value,
// shown in the error message.
std::vector<int> parse_positive_int_list(const std::string& flag, const std::string& example,
                                         const std::string& text) {
    std::vector<int> values;
    for (const std::string& item : split_list(text)) {
        int value = 0;
        if (!to_positive_int(item, value)) {
            values.clear();
            break;
        }
        values.push_back(value);
    }
    if (values.empty()) {
        throw std::invalid_argument(
            flag + " must be a comma-separated list of whole numbers, each 1 or more "
            "(for example " + example + "), got '" + text + "'");
    }
    return values;
}

std::vector<double> parse_sigmas(const std::string& text) {
    std::vector<double> sigmas;
    try {
        for (const std::string& item : split_list(text)) {
            std::size_t consumed = 0;
            const double sigma = std::stod(item, &consumed);
            // The same rule gaussian_blur uses: finite and positive. stod
            // accepts "nan" and "inf", so they are caught here.
            if (consumed != item.size() || !std::isfinite(sigma) || sigma <= 0.0) {
                sigmas.clear();
                break;
            }
            sigmas.push_back(sigma);
        }
    } catch (const std::exception&) {
        sigmas.clear();   // not a number, or out of range for a double
    }
    if (sigmas.empty()) {
        throw std::invalid_argument(
            "--sigmas must be a comma-separated list of numbers greater than 0 "
            "(for example 1,3,5), got '" + text + "'");
    }
    // Throws now, before anything is measured, if a sigma is too large.
    for (const double sigma : sigmas) {
        radius_for_sigma(sigma);
    }
    return sigmas;
}

// The value of --backends: the Backends measured after the Reference. The
// Reference is always measured first, so serial is not accepted here.
std::vector<Backend> parse_bench_backends(const std::string& text) {
    std::vector<Backend> backends;
    for (const std::string& item : split_list(text)) {
        const std::optional<Backend> backend = find_backend(item);
        if (!backend || *backend == Backend::Serial) {
            backends.clear();
            break;
        }
        backends.push_back(*backend);
    }
    if (backends.empty()) {
        throw std::invalid_argument(
            "--backends must be a comma-separated list of openmp, cuda-naive and "
            "cuda-tiled (for example openmp,cuda-tiled; serial is always measured "
            "first), got '" + text + "'");
    }
    return backends;
}

// What the bench flags ask for.
struct BenchOptions {
    BenchmarkConfig config;
    std::string csv_path;   // empty: no CSV file
};

// Flags come after "bench", in any order, each followed by its value.
BenchOptions parse_bench_flags(const std::vector<std::string>& args) {
    BenchOptions options;
    BenchmarkConfig& config = options.config;

    for (std::size_t i = 1; i < args.size(); i += 2) {
        const std::string& flag = args[i];
        if (flag != "--sizes" && flag != "--sigmas" && flag != "--threads" &&
            flag != "--backends" && flag != "--runs" && flag != "--csv") {
            throw std::invalid_argument("unknown bench argument '" + flag + "'\n" + usage);
        }

        if (i + 1 >= args.size()) {
            throw std::invalid_argument(flag + " needs a value\n" + usage);
        }
        const std::string& value = args[i + 1];

        if (flag == "--sizes") {
            config.sizes = parse_positive_int_list(flag, "1024,2048", value);
        } else if (flag == "--sigmas") {
            config.sigmas = parse_sigmas(value);
        } else if (flag == "--threads") {
            config.thread_counts = parse_positive_int_list(flag, "1,2,4,8", value);
        } else if (flag == "--backends") {
            config.backends = parse_bench_backends(value);
        } else if (flag == "--runs") {
            config.runs = parse_positive_int(flag, value);
        } else {
            // An empty path would otherwise mean "no CSV" and silently start
            // a full-size session.
            if (value.empty()) {
                throw std::invalid_argument("--csv needs a file path, got ''");
            }
            options.csv_path = value;
        }
    }
    return options;
}

// The CSV file is one header line, then one line per row. The column order
// is a contract (later tools read this file), so do not reorder it.
// Both functions end with a flush, so a session that stops early still leaves
// a valid file holding every row measured so far.
// New columns only ever go at the end. An empty field means "does not apply":
// threads and efficiency in a CUDA row, the three GPU times in a CPU row.
void write_csv_header(std::ostream& csv) {
    csv << "width,height,sigma,radius,backend,threads,runs,"
           "median_ms,min_ms,max_ms,speedup,efficiency,"
           "h2d_ms,kernel_ms,d2h_ms\n"
        << std::flush;
}

// The value, or nothing at all for an empty field.
template <typename T>
void write_csv_field(std::ostream& csv, const std::optional<T>& value) {
    if (value) {
        csv << *value;
    }
}

void write_csv_row(std::ostream& csv, const BenchmarkRow& row) {
    csv << row.width << ',' << row.height << ',' << row.sigma << ','
        << row.radius << ',' << backend_name(row.backend) << ',';
    write_csv_field(csv, row.threads);
    csv << ',' << row.runs << ','
        << row.median_ms << ',' << row.min_ms << ',' << row.max_ms << ','
        << row.speedup << ',';
    write_csv_field(csv, row.efficiency);
    csv << ',';
    if (row.gpu_timings) {
        csv << row.gpu_timings->h2d_ms << ',' << row.gpu_timings->kernel_ms << ','
            << row.gpu_timings->d2h_ms;
    } else {
        csv << ",,";
    }
    csv << "\n" << std::flush;
}

// The compiler's name and version, from macros the compiler defines itself.
// Clang also defines __GNUC__, so it is checked first.
#if defined(__clang__)
const char* const compiler_version = "Clang " __clang_version__;
#elif defined(__GNUC__)
const char* const compiler_version = "GCC " __VERSION__;
#else
const char* const compiler_version = "unknown";
#endif

// IMGFILTER_BUILD_TYPE and IMGFILTER_COMPILE_FLAGS come from CMakeLists.txt,
// so they are what this build was configured with.
std::string build_type_name() {
    const std::string build_type = IMGFILTER_BUILD_TYPE;
    return build_type.empty() ? "(none)" : build_type;
}

#ifdef IMGFILTER_HAS_CUDA
// CUDA gives a version as one number, 1000 * major + 10 * minor: 12040 is 12.4.
std::string cuda_version_text(int version) {
    return std::to_string(version / 1000) + "." + std::to_string(version % 1000 / 10);
}
#endif

// The CUDA lines of the bench header, in a CUDA build only. The macros come
// from CMakeLists.txt, like IMGFILTER_COMPILE_FLAGS. `gpu` is empty when no
// usable GPU was found (and none was needed).
void print_gpu_environment(std::ostream& out, const std::optional<CudaDeviceInfo>& gpu) {
#ifdef IMGFILTER_HAS_CUDA
    out << "cuda:        " << IMGFILTER_CUDA_COMPILER << "\n"
        << "cuda flags:  " << IMGFILTER_CUDA_FLAGS << "\n"
        << "cuda archs:  " << IMGFILTER_CUDA_ARCHITECTURES << "\n";
    if (!gpu) {
        out << "gpu:         none usable\n";
        return;
    }
    out << "gpu:         " << gpu->name << "\n"
        << "compute:     " << gpu->compute_major << "." << gpu->compute_minor << "\n"
        << "SMs:         " << gpu->multiprocessors << "\n"
        << "gpu memory:  " << gpu->total_memory_bytes / (1024 * 1024) << " MiB\n"
        << "driver:      CUDA " << cuda_version_text(gpu->driver_version) << "\n"
        << "runtime:     CUDA " << cuda_version_text(gpu->runtime_version) << "\n";
#else
    (void)out;
    (void)gpu;
#endif
}

// No line starts with a number, so none can be mistaken for a table row.
// Ends with a flush so piped output shows it before anything is measured.
void print_environment(std::ostream& out, int runs, const std::optional<CudaDeviceInfo>& gpu) {
    out << "compiler:    " << compiler_version << "\n"
        << "build type:  " << build_type_name() << "\n"
        << "flags:       " << IMGFILTER_COMPILE_FLAGS << "\n"
        << "max threads: " << omp_get_max_threads() << "\n";
    print_gpu_environment(out, gpu);
    out << "runs:        " << runs << "\n"
        << "\n"
        << std::flush;
}

// Checks, before anything is measured, that every requested Backend can run
// every requested sigma, and finds the GPU if a CUDA Backend is requested
// (throwing, with the CUDA error, if there is none). Returns the GPU's
// description for the header. A CUDA build looks for one even when no CUDA
// Backend is requested, but then not finding one is fine.
std::optional<CudaDeviceInfo> check_bench_backends(const BenchmarkConfig& config) {
    for (const Backend backend : config.backends) {
        for (const double sigma : config.sigmas) {
            check_backend_supported(backend, sigma);   // no CUDA, or sigma too large
        }
    }
    if (std::any_of(config.backends.begin(), config.backends.end(), is_cuda)) {
        return cuda_device_info();
    }
    if (cuda_built()) {
        try {
            return cuda_device_info();
        } catch (const std::exception&) {
            // no usable GPU, and none is needed
        }
    }
    return std::nullopt;
}

// GCC and Clang define __OPTIMIZE__ at -O1 and above, so this follows the
// flags the code was really compiled with, not the name of the build type.
// The benchmark still runs: the timings are just not worth quoting.
void warn_if_unoptimised(std::ostream& err) {
#ifndef __OPTIMIZE__
    err << "Warning: this build is not optimised (build type: " << build_type_name()
        << "), so these timings say nothing about real speed. "
           "Rebuild with -DCMAKE_BUILD_TYPE=Release before quoting any numbers.\n";
#else
    (void)err;
#endif
}

// args[0] is "bench". Every flag is checked before anything is measured.
int bench_command(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    const BenchOptions options = parse_bench_flags(args);
    const std::optional<CudaDeviceInfo> gpu = check_bench_backends(options.config);

    // Opened before anything is printed or measured, so a bad path fails now
    // and not after a long session.
    std::ofstream csv;
    RowCallback on_row;   // empty: no CSV file
    if (!options.csv_path.empty()) {
        csv.open(options.csv_path);
        if (!csv) {
            throw std::runtime_error("cannot open '" + options.csv_path + "' for writing");
        }
        write_csv_header(csv);
        on_row = [&csv](const BenchmarkRow& row) { write_csv_row(csv, row); };
    }

    warn_if_unoptimised(err);
    print_environment(out, options.config.runs, gpu);
    run_benchmark(options.config, out, on_row);

    // A failed write (a full disk, say) only sets the stream's error state.
    if (csv.is_open() && !csv) {
        throw std::runtime_error("could not write the results to '" + options.csv_path + "'");
    }
    return 0;
}

// ---- blur command ----

int blur_command(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    if (args.size() < 3) {
        err << usage << "\n";
        return 1;
    }

    const std::string& input_path = args[0];
    const std::string& output_path = args[1];

    const double sigma = parse_sigma(args[2]);
    const Options options = parse_options(args);
    // Before loading anything: a build without CUDA, or a sigma too large for
    // cuda-tiled, fails here and writes no output file. A bad sigma is left
    // for gaussian_blur to reject with its own message.
    if (std::isfinite(sigma) && sigma > 0.0) {
        check_backend_supported(options.backend, sigma);
    }
    const Image source = load_image(input_path);
    const bool gpu = is_cuda(options.backend);

    // Time only the blur, not the disk I/O.
    GpuTimings timings;
    const auto start = std::chrono::steady_clock::now();
    const Image destination = gaussian_blur(source, sigma, options.backend, options.threads,
                                            gpu ? &timings : nullptr);
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = stop - start;

    save_image(output_path, destination);

    out << "Blurred " << source.width << "x" << source.height
        << " image, sigma " << sigma
        << ", radius " << radius_for_sigma(sigma)
        << ", backend " << backend_name(options.backend);
    if (gpu) {
        // The total is the Host clock around the whole call, so it also holds
        // the Device allocation and freeing that the three parts leave out.
        out << ", GPU " << cuda_device_info().name
            << ", " << elapsed.count() << " ms"
            << " (host-to-device " << timings.h2d_ms << " ms"
            << ", kernel " << timings.kernel_ms << " ms"
            << ", device-to-host " << timings.d2h_ms << " ms)";
    } else {
        out << ", threads " << options.threads
            << ", " << elapsed.count() << " ms";
    }
    out << ", saved to " << output_path << "\n";
    return 0;
}

const char* const out_of_memory_message =
    "Error: out of memory (a very large sigma or image needs more memory "
    "than this machine has)\n";

}  // namespace

int run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    try {
        // A Source file literally named "bench" needs a path prefix: ./bench
        if (!args.empty() && args[0] == "bench") {
            return bench_command(args, out, err);
        }
        return blur_command(args, out, err);
    } catch (const std::bad_alloc&) {
        // A very large sigma makes the Weights too big to allocate. Must come
        // before the std::exception handler, which would also match.
        err << out_of_memory_message;
        return 1;
    } catch (const std::length_error&) {
        // Larger still and std::vector refuses before even trying to allocate.
        err << out_of_memory_message;
        return 1;
    } catch (const std::exception& e) {
        err << "Error: " << e.what() << "\n";
        return 1;
    }
}
