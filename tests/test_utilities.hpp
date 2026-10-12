// Shared by every test file. No test framework: each test is a plain
// function, CHECK records failures, and run_tests returns non-zero if any
// failed.
//
// Every test goes through run(), the same function main() calls, and looks
// only at what a user could see: the exit code, the text printed, and the
// output file.
//
// Everything here is `inline` so each test file (its own executable) can
// include this header without "unused function" warnings.

#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cli.h"
#include "image.h"
#include "io.h"

namespace fs = std::filesystem;

inline int failures = 0;

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

// Runs every test and returns the exit code for main().
inline int run_tests(std::initializer_list<void (*)()> tests) {
    // A test that throws is a failure too, not a crash with no explanation.
    try {
        for (const auto test : tests) {
            test();
        }
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

// What one call to run() produced.
struct RunResult {
    int exit_code;
    std::string out;
    std::string err;
};

inline RunResult run_cli(const std::vector<std::string>& args) {
    std::ostringstream out;
    std::ostringstream err;
    const int exit_code = run(args, out, err);
    return {exit_code, out.str(), err.str()};
}

// A fresh, empty directory for one test's files. Names must be unique across
// all test files, because ctest may run the executables at the same time.
inline fs::path temp_dir(const std::string& test_name) {
    const fs::path dir = fs::temp_directory_path() / "imgfilter_tests" / test_name;
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

inline bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

// The raw bytes of a file, to prove a file was not changed.
inline std::string read_file_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream bytes;
    bytes << file.rdbuf();
    return bytes.str();
}

inline bool same_image(const Image& a, const Image& b) {
    return a.width == b.width && a.height == b.height &&
           a.channels == b.channels && a.data == b.data;
}

// An image where every pixel has the same grey level.
inline Image make_flat(int width, int height, std::uint8_t level) {
    Image img;
    img.width = width;
    img.height = height;
    img.channels = 1;
    img.data.assign(static_cast<std::size_t>(width) * height, level);
    return img;
}

// A small image where every pixel has a different value, so a swapped x/y or
// a shifted row cannot go unnoticed. Use width != height for the same reason.
inline Image make_gradient(int width, int height) {
    Image img = make_flat(width, height, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            img.data[img.index(x, y)] = static_cast<std::uint8_t>(y * width + x);
        }
    }
    return img;
}

// Black everywhere except one white pixel at (x, y).
inline Image make_impulse(int width, int height, int x, int y) {
    Image img = make_flat(width, height, 0);
    img.data[img.index(x, y)] = 255;
    return img;
}

// Random grey levels, the same ones on every run: std::mt19937 is specified
// exactly by the C++ standard, so a fixed seed gives the same numbers on any
// compiler. Noise has no pattern for a bug to hide behind: a row computed
// twice, skipped, or written to the wrong place changes the output.
inline Image make_noise(int width, int height) {
    Image img = make_flat(width, height, 0);
    std::mt19937 generator(20261011u);   // constant seed
    for (std::uint8_t& value : img.data) {
        // mt19937 produces 32-bit numbers; keep the top 8 bits.
        value = static_cast<std::uint8_t>(generator() >> 24);
    }
    return img;
}
