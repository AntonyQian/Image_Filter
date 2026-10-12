#include "cli.h"

#include <chrono>
#include <cstddef>
#include <exception>
#include <ostream>
#include <stdexcept>

#include <omp.h>

#include "gaussian_blur.h"
#include "image.h"
#include "io.h"
#include "weights.h"

namespace {

// Turns the sigma argument into a number.
// Throws std::invalid_argument if the text is not a number. Whether the
// number is a usable sigma (finite, greater than zero) is checked by
// gaussian_blur itself, so that rule lives in one place.
double parse_sigma(const std::string& text) {
    try {
        // std::stod stops at the first character it cannot use and reports
        // how many it consumed, so "3abc" would otherwise be accepted as 3.
        std::size_t consumed = 0;
        const double sigma = std::stod(text, &consumed);
        if (consumed == text.size()) {
            return sigma;
        }
    } catch (const std::exception&) {
        // Not a number at all, or too large for a double: fall through to
        // the one error message below.
    }
    throw std::invalid_argument("sigma must be a number, got '" + text + "'");
}

const char* const usage =
    "Usage: imgfilter <input image> <output.png> <sigma> "
    "[--backend serial|openmp] [--threads N]";

// What the optional flags asked for.
struct Options {
    Backend backend = Backend::OpenMP;
    // The most threads the OpenMP runtime will use by default: normally one
    // per logical core, or the OMP_NUM_THREADS environment variable if set.
    int threads = omp_get_max_threads();
};

// Turns the value of --backend into a Backend.
// Throws std::invalid_argument for a name that is not a Backend. A mistyped
// name must not quietly fall back to the default: the user would be timing a
// different Backend from the one they asked for.
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

// The name a Backend is selected by and reported as.
const char* backend_name(Backend backend) {
    switch (backend) {
        case Backend::Serial: return "serial";
        case Backend::OpenMP: return "openmp";
    }
    return "unknown";  // not reachable; keeps -Wreturn-type quiet
}

// Turns the value of --threads into a thread count of 1 or more.
// Throws std::invalid_argument for anything else (0, negative, not a number).
int parse_threads(const std::string& text) {
    try {
        // Same trick as parse_sigma: `consumed` rejects "4abc" and "2.5".
        std::size_t consumed = 0;
        const int threads = std::stoi(text, &consumed);
        if (consumed == text.size() && threads >= 1) {
            return threads;
        }
    } catch (const std::exception&) {
        // Not a number at all, or too large for an int: same message below.
    }
    throw std::invalid_argument(
        "--threads must be a whole number, 1 or more, got '" + text + "'");
}

// Reads the optional flags, which come after the three positional arguments.
// Each flag is followed by its value: --backend openmp --threads 4
// Throws std::invalid_argument for an unknown flag, a flag with no value
// after it, or a value that is not valid.
Options parse_options(const std::vector<std::string>& args) {
    Options options;

    for (std::size_t i = 3; i < args.size(); i += 2) {
        const std::string& flag = args[i];
        if (flag != "--backend" && flag != "--threads") {
            throw std::invalid_argument("unknown argument '" + flag + "'\n" + usage);
        }

        // Check before reading args[i + 1]: the flag may be the last word.
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

    // The serial Backend runs on the calling thread alone whatever --threads
    // says, so record (and later print) the count that is really used.
    if (options.backend == Backend::Serial) {
        options.threads = 1;
    }
    return options;
}

}  // namespace

int run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    // Check the argument count before touching args[0..2]:
    // reading past the end of a vector is undefined behaviour.
    if (args.size() < 3) {
        err << usage << "\n";
        return 1;
    }

    const std::string& input_path = args[0];
    const std::string& output_path = args[1];

    // Everything below reports failure by throwing. Catching here, in one
    // place, turns any failure into a message plus a non-zero exit code, so
    // an exception never escapes to main and aborts the program.
    try {
        const double sigma = parse_sigma(args[2]);
        const Options options = parse_options(args);
        const Image source = load_image(input_path);

        // The timer wraps only the blur call: loading and saving are disk
        // work and would hide the number we care about. steady_clock is the
        // clock for measuring intervals because it never jumps backwards or
        // forwards (system_clock can, when Windows adjusts the time).
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
    } catch (const std::exception& e) {
        err << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
