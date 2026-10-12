#include "cli.h"

#include <chrono>
#include <cstddef>
#include <exception>
#include <ostream>
#include <stdexcept>

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

}  // namespace

int run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    // Check the argument count before touching args[0..2]:
    // reading past the end of a vector is undefined behaviour.
    if (args.size() != 3) {
        err << "Usage: imgfilter <input image> <output.png> <sigma>\n";
        return 1;
    }

    const std::string& input_path = args[0];
    const std::string& output_path = args[1];

    // Everything below reports failure by throwing. Catching here, in one
    // place, turns any failure into a message plus a non-zero exit code, so
    // an exception never escapes to main and aborts the program.
    try {
        const double sigma = parse_sigma(args[2]);
        const Image source = load_image(input_path);

        // The timer wraps only the blur call: loading and saving are disk
        // work and would hide the number we care about. steady_clock is the
        // clock for measuring intervals because it never jumps backwards or
        // forwards (system_clock can, when Windows adjusts the time).
        const auto start = std::chrono::steady_clock::now();
        const Image destination = gaussian_blur(source, sigma, Backend::Serial, 1);
        const auto stop = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;

        save_image(output_path, destination);

        out << "Blurred " << source.width << "x" << source.height
            << " image, sigma " << sigma
            << ", radius " << radius_for_sigma(sigma)
            << ", " << elapsed.count() << " ms"
            << ", saved to " << output_path << "\n";
    } catch (const std::exception& e) {
        err << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
