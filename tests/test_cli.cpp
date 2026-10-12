// Tests for the command-line tool. No test framework: each test is a plain
// function, CHECK records failures, and main returns non-zero if any failed.
//
// Every test goes through run(), the same function main() calls, and looks
// only at what a user could see: the exit code, the text printed, and the
// output file.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cli.h"
#include "image.h"
#include "io.h"

namespace fs = std::filesystem;

static int failures = 0;

// Prints the failed condition with its file and line, then carries on, so one
// run reports every failure instead of stopping at the first.
#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << __FILE__ << ":" << __LINE__                         \
                      << ": CHECK failed: " #condition "\n";                 \
            ++failures;                                                      \
        }                                                                    \
    } while (false)

// What one call to run() produced.
struct RunResult {
    int exit_code;
    std::string out;
    std::string err;
};

static RunResult run_cli(const std::vector<std::string>& args) {
    std::ostringstream out;
    std::ostringstream err;
    const int exit_code = run(args, out, err);
    return {exit_code, out.str(), err.str()};
}

// A fresh, empty directory for one test's files.
static fs::path temp_dir(const std::string& test_name) {
    const fs::path dir = fs::temp_directory_path() / "imgfilter_tests" / test_name;
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

// A small image where every pixel has a different value, so a swapped x/y or
// a shifted row cannot go unnoticed. Width != height for the same reason.
static Image make_gradient(int width, int height) {
    Image img;
    img.width = width;
    img.height = height;
    img.channels = 1;
    img.data.resize(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            img.data[img.index(x, y)] = static_cast<std::uint8_t>(y * width + x);
        }
    }
    return img;
}

static bool same_image(const Image& a, const Image& b) {
    return a.width == b.width && a.height == b.height &&
           a.channels == b.channels && a.data == b.data;
}

static void test_png_round_trip() {
    const fs::path dir = temp_dir("png_round_trip");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    const Image original = make_gradient(5, 3);
    save_image(input, original);

    const RunResult result = run_cli({input, output});

    CHECK(result.exit_code == 0);
    CHECK(result.err.empty());
    CHECK(fs::exists(output));
    if (fs::exists(output)) {
        const Image saved = load_image(output);
        CHECK(saved.width == 5);
        CHECK(saved.height == 3);
        CHECK(saved.channels == 1);
        CHECK(same_image(saved, original));
    }
}

int main() {
    // A test that throws is a failure too, not a crash with no explanation.
    try {
        test_png_round_trip();
    } catch (const std::exception& e) {
        std::cerr << "unexpected exception: " << e.what() << "\n";
        ++failures;
    }

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all checks passed\n";
    return 0;
}
