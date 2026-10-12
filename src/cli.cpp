#include "cli.h"

#include <chrono>
#include <cstddef>
#include <exception>
#include <new>
#include <ostream>
#include <stdexcept>

#include <omp.h>

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
    "[--backend serial|openmp] [--threads N]";

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

const char* backend_name(Backend backend) {
    switch (backend) {
        case Backend::Serial: return "serial";
        case Backend::OpenMP: return "openmp";
    }
    return "unknown";  // not reachable; keeps -Wreturn-type quiet
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

const char* const out_of_memory_message =
    "Error: out of memory (a very large sigma or image needs more memory "
    "than this machine has)\n";

}  // namespace

int run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    if (args.size() < 3) {
        err << usage << "\n";
        return 1;
    }

    const std::string& input_path = args[0];
    const std::string& output_path = args[1];

    try {
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

    return 0;
}
