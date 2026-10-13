#include "cli.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <new>
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
    "[--backend serial|openmp] [--threads N]\n"
    "       imgfilter bench [--sizes N,N,...] [--sigmas S,S,...] [--threads N,N,...]\n"
    "                       [--runs N] [--csv <results.csv>]\n"
    "\n"
    "The first form blurs one image. bench times the blur of a synthetic square\n"
    "image of each size with each sigma: the serial backend first, then the\n"
    "openmp backend at each thread count. Each is one untimed warm-up, then\n"
    "--runs timed runs (defaults: --sizes 4096 --sigmas 3 --runs 15, and\n"
    "--threads every count from 1 to the maximum). --csv also writes the rows\n"
    "of the table to a file, replacing it if it exists.";

struct Options {
    Backend backend = Backend::OpenMP;
    int threads = omp_get_max_threads();
};

Backend parse_backend(const std::string& text) {
    if (text == "serial") {
        return Backend::Serial;
    }
    if (text == "openmp") {
        return Backend::OpenMP;
    }
    throw std::invalid_argument("unknown backend '" + text +
                                "', valid values are: serial, openmp");
}

int parse_threads(const std::string& text) {
    try {
        // consumed rejects "4abc" and "2.5"
        std::size_t consumed = 0;
        const int threads = std::stoi(text, &consumed);
        if (consumed == text.size() && threads >= 1) {
            return threads;
        }
    } catch (const std::exception&) {
        // fall through to the error below
    }
    throw std::invalid_argument(
        "--threads must be a whole number, 1 or more, got '" + text + "'");
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
            options.threads = parse_threads(value);
        }
    }

    // Serial ignores --threads, so report the count actually used.
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

std::vector<int> parse_sizes(const std::string& text) {
    std::vector<int> sizes;
    try {
        for (const std::string& item : split_list(text)) {
            // consumed rejects "16abc" and "2.5"
            std::size_t consumed = 0;
            const int size = std::stoi(item, &consumed);
            if (consumed != item.size() || size < 1) {
                sizes.clear();
                break;
            }
            sizes.push_back(size);
        }
    } catch (const std::exception&) {
        sizes.clear();   // not a number, or too big for an int
    }
    if (sizes.empty()) {
        throw std::invalid_argument(
            "--sizes must be a comma-separated list of whole numbers, each 1 or more "
            "(for example 1024,2048), got '" + text + "'");
    }
    return sizes;
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

std::vector<int> parse_thread_counts(const std::string& text) {
    std::vector<int> thread_counts;
    try {
        for (const std::string& item : split_list(text)) {
            // consumed rejects "4abc" and "2.5"
            std::size_t consumed = 0;
            const int threads = std::stoi(item, &consumed);
            if (consumed != item.size() || threads < 1) {
                thread_counts.clear();
                break;
            }
            thread_counts.push_back(threads);
        }
    } catch (const std::exception&) {
        thread_counts.clear();   // not a number, or too big for an int
    }
    if (thread_counts.empty()) {
        throw std::invalid_argument(
            "--threads must be a comma-separated list of whole numbers, each 1 or more "
            "(for example 1,2,4,8), got '" + text + "'");
    }
    return thread_counts;
}

int parse_runs(const std::string& text) {
    try {
        // consumed rejects "3abc" and "2.5"
        std::size_t consumed = 0;
        const int runs = std::stoi(text, &consumed);
        if (consumed == text.size() && runs >= 1) {
            return runs;
        }
    } catch (const std::exception&) {
        // fall through to the error below
    }
    throw std::invalid_argument(
        "--runs must be a whole number, 1 or more, got '" + text + "'");
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
        if (flag != "--sizes" && flag != "--sigmas" && flag != "--threads" && flag != "--runs" &&
            flag != "--csv") {
            throw std::invalid_argument("unknown bench argument '" + flag + "'\n" + usage);
        }

        if (i + 1 >= args.size()) {
            throw std::invalid_argument(flag + " needs a value\n" + usage);
        }
        const std::string& value = args[i + 1];

        if (flag == "--sizes") {
            config.sizes = parse_sizes(value);
        } else if (flag == "--sigmas") {
            config.sigmas = parse_sigmas(value);
        } else if (flag == "--threads") {
            config.threads = parse_thread_counts(value);
        } else if (flag == "--runs") {
            config.runs = parse_runs(value);
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

// One header line, then one line per row. The column order is a contract
// (later tools read this file), so do not reorder it.
void write_csv(std::ostream& csv, const std::vector<BenchmarkRow>& rows) {
    csv << "width,height,sigma,radius,backend,threads,runs,"
           "median_ms,min_ms,max_ms,speedup,efficiency\n";
    for (const BenchmarkRow& row : rows) {
        csv << row.width << ',' << row.height << ',' << row.sigma << ','
            << row.radius << ',' << backend_name(row.backend) << ','
            << row.threads << ',' << row.runs << ','
            << row.median_ms << ',' << row.min_ms << ',' << row.max_ms << ','
            << row.speedup << ',' << row.efficiency << "\n";
    }
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
// No line starts with a number, so none can be mistaken for a table row.
void print_environment(std::ostream& out, int runs) {
    const std::string build_type = IMGFILTER_BUILD_TYPE;
    out << "compiler:    " << compiler_version << "\n"
        << "build type:  " << (build_type.empty() ? "(none)" : build_type) << "\n"
        << "flags:       " << IMGFILTER_COMPILE_FLAGS << "\n"
        << "max threads: " << omp_get_max_threads() << "\n"
        << "runs:        " << runs << "\n"
        << "\n";
}

// GCC and Clang define __OPTIMIZE__ at -O1 and above, so this follows the
// flags the code was really compiled with, not the name of the build type.
// The benchmark still runs: the timings are just not worth quoting.
void warn_if_unoptimised(std::ostream& err) {
#ifndef __OPTIMIZE__
    const std::string build_type = IMGFILTER_BUILD_TYPE;
    err << "Warning: this build is not optimised (build type: "
        << (build_type.empty() ? "(none)" : build_type)
        << "), so these timings say nothing about real speed. "
           "Rebuild with -DCMAKE_BUILD_TYPE=Release before quoting any numbers.\n";
#else
    (void)err;
#endif
}

// args[0] is "bench". Every flag is checked before anything is measured.
int bench_command(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    const BenchOptions options = parse_bench_flags(args);

    // Opened before anything is printed or measured, so a bad path fails now
    // and not after a long session.
    std::ofstream csv;
    if (!options.csv_path.empty()) {
        csv.open(options.csv_path);
        if (!csv) {
            throw std::runtime_error("cannot open '" + options.csv_path + "' for writing");
        }
    }

    warn_if_unoptimised(err);
    print_environment(out, options.config.runs);
    const std::vector<BenchmarkRow> rows = run_benchmark(options.config, out);

    if (csv.is_open()) {
        write_csv(csv, rows);
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
    const Image source = load_image(input_path);

    // Time only the blur, not the disk I/O.
    const auto start = std::chrono::steady_clock::now();
    const Image destination = gaussian_blur(source, sigma, options.backend, options.threads);
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = stop - start;

    save_image(output_path, destination);

    out << "Blurred " << source.width << "x" << source.height
        << " image, sigma " << sigma
        << ", radius " << radius_for_sigma(sigma)
        << ", backend " << backend_name(options.backend)
        << ", threads " << options.threads
        << ", " << elapsed.count() << " ms"
        << ", saved to " << output_path << "\n";
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
