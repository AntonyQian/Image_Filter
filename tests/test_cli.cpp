// Tests for the command-line tool. No test framework: each test is a plain
// function, CHECK records failures, and main returns non-zero if any failed.
//
// Every test goes through run(), the same function main() calls, and looks
// only at what a user could see: the exit code, the text printed, and the
// output file.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Declarations only (no STB_IMAGE_WRITE_IMPLEMENTATION here): the code itself
// is compiled once, in src/io.cpp. Needed for the one JPG test input.
#include <stb_image_write.h>

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

static bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

static void test_missing_input_names_the_path() {
    const fs::path dir = temp_dir("missing_input");
    const std::string input = (dir / "does_not_exist.png").string();
    const std::string output = (dir / "out.png").string();

    const RunResult result = run_cli({input, output});

    CHECK(result.exit_code != 0);
    CHECK(contains(result.err, input));
    CHECK(!fs::exists(output));
}

static void test_colour_jpg_becomes_grayscale_png() {
    // The project can only save PNGs, so this one input is written with stb
    // directly: a 3-channel (RGB) JPG, to exercise both JPG decoding and the
    // conversion to grayscale.
    const fs::path dir = temp_dir("colour_jpg");
    const std::string input = (dir / "in.jpg").string();
    const std::string output = (dir / "out.png").string();
    const int width = 16;
    const int height = 8;
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3);
    for (std::size_t i = 0; i < rgb.size(); ++i) {
        rgb[i] = static_cast<std::uint8_t>(i * 7);
    }
    CHECK(stbi_write_jpg(input.c_str(), width, height, 3, rgb.data(), 90) != 0);

    const RunResult result = run_cli({input, output});

    CHECK(result.exit_code == 0);
    CHECK(fs::exists(output));
    if (fs::exists(output)) {
        // JPG is lossy, so the expected pixels are what the JPG decodes to as
        // grayscale, not the bytes written above.
        const Image expected = load_image(input);
        const Image saved = load_image(output);
        CHECK(saved.width == width);
        CHECK(saved.height == height);
        CHECK(saved.channels == 1);
        CHECK(same_image(saved, expected));
    }
}

static void test_unreadable_input_names_the_path() {
    // The file exists but is not an image, so decoding it must fail.
    const fs::path dir = temp_dir("unreadable_input");
    const std::string input = (dir / "not_an_image.png").string();
    const std::string output = (dir / "out.png").string();
    std::ofstream(input) << "this is text, not a PNG";

    const RunResult result = run_cli({input, output});

    CHECK(result.exit_code != 0);
    CHECK(contains(result.err, input));
    CHECK(!fs::exists(output));
}

static void test_unwritable_output_names_the_path() {
    // The output folder does not exist, so the PNG cannot be created.
    const fs::path dir = temp_dir("unwritable_output");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "no_such_folder" / "out.png").string();
    save_image(input, make_gradient(5, 3));

    const RunResult result = run_cli({input, output});

    CHECK(result.exit_code != 0);
    CHECK(contains(result.err, output));
    CHECK(!fs::exists(output));
}

static void test_success_reports_the_image_size() {
    const fs::path dir = temp_dir("reports_size");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_gradient(5, 3));

    const RunResult result = run_cli({input, output});

    CHECK(result.exit_code == 0);
    CHECK(contains(result.out, "5x3"));
}

static void test_wrong_argument_count_prints_usage() {
    const RunResult none = run_cli({});
    CHECK(none.exit_code != 0);
    CHECK(contains(none.err, "Usage"));

    const RunResult one = run_cli({"only_input.png"});
    CHECK(one.exit_code != 0);
    CHECK(contains(one.err, "Usage"));
}

int main() {
    // A test that throws is a failure too, not a crash with no explanation.
    try {
        test_png_round_trip();
        test_colour_jpg_becomes_grayscale_png();
        test_missing_input_names_the_path();
        test_unreadable_input_names_the_path();
        test_unwritable_output_names_the_path();
        test_success_reports_the_image_size();
        test_wrong_argument_count_prints_usage();
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
