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
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Only for omp_get_max_threads(), the thread count the tool defaults to.
#include <omp.h>

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

// An image where every pixel has the same grey level.
static Image make_flat(int width, int height, std::uint8_t level) {
    Image img;
    img.width = width;
    img.height = height;
    img.channels = 1;
    img.data.assign(static_cast<std::size_t>(width) * height, level);
    return img;
}

// Black everywhere except one white pixel at (x, y).
static Image make_impulse(int width, int height, int x, int y) {
    Image img = make_flat(width, height, 0);
    img.data[img.index(x, y)] = 255;
    return img;
}

// The raw bytes of a file, to prove a file was not changed.
static std::string read_file_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream bytes;
    bytes << file.rdbuf();
    return bytes.str();
}

static std::size_t count_non_zero(const Image& img) {
    std::size_t count = 0;
    for (const std::uint8_t value : img.data) {
        if (value != 0) {
            ++count;
        }
    }
    return count;
}

static bool same_image(const Image& a, const Image& b) {
    return a.width == b.width && a.height == b.height &&
           a.channels == b.channels && a.data == b.data;
}

static void test_destination_keeps_the_source_size() {
    const fs::path dir = temp_dir("keeps_size");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_gradient(5, 3));

    const RunResult result = run_cli({input, output, "1"});

    CHECK(result.exit_code == 0);
    CHECK(result.err.empty());
    CHECK(fs::exists(output));
    if (fs::exists(output)) {
        const Image saved = load_image(output);
        CHECK(saved.width == 5);
        CHECK(saved.height == 3);
        CHECK(saved.channels == 1);
    }
}

static bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

static void test_missing_input_names_the_path() {
    const fs::path dir = temp_dir("missing_input");
    const std::string input = (dir / "does_not_exist.png").string();
    const std::string output = (dir / "out.png").string();

    const RunResult result = run_cli({input, output, "1"});

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

    const RunResult result = run_cli({input, output, "1"});

    CHECK(result.exit_code == 0);
    CHECK(fs::exists(output));
    if (fs::exists(output)) {
        const Image saved = load_image(output);
        CHECK(saved.width == width);
        CHECK(saved.height == height);
        CHECK(saved.channels == 1);
    }
}

static void test_unreadable_input_names_the_path() {
    // The file exists but is not an image, so decoding it must fail.
    const fs::path dir = temp_dir("unreadable_input");
    const std::string input = (dir / "not_an_image.png").string();
    const std::string output = (dir / "out.png").string();
    std::ofstream(input) << "this is text, not a PNG";

    const RunResult result = run_cli({input, output, "1"});

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

    const RunResult result = run_cli({input, output, "1"});

    CHECK(result.exit_code != 0);
    CHECK(contains(result.err, output));
    CHECK(!fs::exists(output));
}

static void test_success_reports_size_sigma_radius_and_time() {
    const fs::path dir = temp_dir("reports_run");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_gradient(5, 3));

    const RunResult result = run_cli({input, output, "1"});

    CHECK(result.exit_code == 0);
    CHECK(contains(result.out, "5x3"));
    CHECK(contains(result.out, "sigma 1"));
    CHECK(contains(result.out, "radius 3"));
    CHECK(contains(result.out, " ms"));
}

static void test_radius_is_ceil_of_three_sigma() {
    const fs::path dir = temp_dir("radius");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_gradient(5, 3));

    // Expected radii worked out by hand: ceil(3 * sigma).
    // Sigma 1 is the trap: 3 * 1 is exactly 3, so the radius must be 3, not 4.
    CHECK(contains(run_cli({input, output, "0.5"}).out, "radius 2"));   // 1.5 -> 2
    CHECK(contains(run_cli({input, output, "1"}).out, "radius 3"));     // 3   -> 3
    CHECK(contains(run_cli({input, output, "1.1"}).out, "radius 4"));   // 3.3 -> 4
    CHECK(contains(run_cli({input, output, "3"}).out, "radius 9"));     // 9   -> 9
    CHECK(contains(run_cli({input, output, "5"}).out, "radius 15"));    // 15  -> 15
}

static void test_wrong_argument_count_prints_usage() {
    const RunResult none = run_cli({});
    CHECK(none.exit_code != 0);
    CHECK(contains(none.err, "Usage"));

    const RunResult one = run_cli({"only_input.png"});
    CHECK(one.exit_code != 0);
    CHECK(contains(one.err, "Usage"));

    // Sigma is required: input and output alone are not enough.
    const RunResult no_sigma = run_cli({"in.png", "out.png"});
    CHECK(no_sigma.exit_code != 0);
    CHECK(contains(no_sigma.err, "Usage"));

    const RunResult too_many = run_cli({"in.png", "out.png", "1", "extra"});
    CHECK(too_many.exit_code != 0);
    CHECK(contains(too_many.err, "Usage"));
}

static void test_flat_image_stays_exactly_flat() {
    // If the Weights sum to exactly 1 and the border reuses the nearest edge
    // pixel, blurring a constant image changes nothing: not in the middle and
    // not at the edges. 20x12 is smaller than the sigma 5 window (31 wide),
    // so every pixel reads past the border at least once.
    const fs::path dir = temp_dir("flat");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();

    for (const char* sigma : {"1", "3", "5"}) {
        for (const int level : {0, 1, 64, 127, 128, 200, 254, 255}) {
            const Image flat = make_flat(20, 12, static_cast<std::uint8_t>(level));
            save_image(input, flat);

            const RunResult result = run_cli({input, output, sigma});

            CHECK(result.exit_code == 0);
            CHECK(same_image(load_image(output), flat));
        }
    }
}

// Blurs one white pixel on black and checks the shape of the blob it becomes.
static void check_impulse_blob(const std::string& sigma, int radius) {
    // Not square, and the bright pixel is not on the diagonal, so swapping x
    // and y anywhere would show. Large enough that the blur window around the
    // bright pixel never touches the border.
    const int width = 45;
    const int height = 41;
    const int cx = 22;
    const int cy = 20;
    const fs::path dir = temp_dir("impulse_sigma_" + sigma);
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_impulse(width, height, cx, cy));

    const RunResult result = run_cli({input, output, sigma});
    CHECK(result.exit_code == 0);
    if (result.exit_code != 0) {
        return;
    }
    const Image blurred = load_image(output);
    CHECK(blurred.width == width);
    CHECK(blurred.height == height);
    if (blurred.width != width || blurred.height != height) {
        return;
    }

    const int centre = blurred.data[blurred.index(cx, cy)];
    CHECK(centre > 0);

    bool outside_is_black = true;
    bool centre_is_brightest = true;
    bool mirrored_left_right = true;
    bool mirrored_top_bottom = true;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int dx = x - cx;
            const int dy = y - cy;
            const int value = blurred.data[blurred.index(x, y)];

            // The bright pixel can only reach pixels within `radius` of it,
            // so everything outside the (2r+1) x (2r+1) window stays black.
            const bool inside = dx >= -radius && dx <= radius &&
                                dy >= -radius && dy <= radius;
            if (!inside) {
                if (value != 0) outside_is_black = false;
                continue;
            }

            if (value > centre) centre_is_brightest = false;
            if (value != blurred.data[blurred.index(cx - dx, y)]) mirrored_left_right = false;
            if (value != blurred.data[blurred.index(x, cy - dy)]) mirrored_top_bottom = false;
        }
    }
    CHECK(outside_is_black);
    CHECK(centre_is_brightest);
    CHECK(mirrored_left_right);
    CHECK(mirrored_top_bottom);

    // It really is a blob, not just the one pixel dimmed.
    CHECK(count_non_zero(blurred) > 1);
    CHECK(blurred.data[blurred.index(cx + 1, cy)] > 0);
    CHECK(blurred.data[blurred.index(cx, cy + 1)] > 0);
}

static void test_bright_pixel_becomes_a_symmetric_blob() {
    check_impulse_blob("1", 3);
    check_impulse_blob("3", 9);
    check_impulse_blob("5", 15);
}

static void test_sigma_1_blob_has_the_expected_values() {
    // Worked out by hand, independently of the code. For sigma 1 (radius 3):
    //   1D Gaussian exp(-d*d/2) for d = 0..3:  1, 0.60653, 0.13534, 0.01111
    //   1D sum = 1 + 2 * (0.60653 + 0.13534 + 0.01111) = 2.50596
    //   normalised 1D: 0.39905 (d = 0), 0.24203 (d = 1)
    // A 2D weight is the product of two 1D ones, and a white pixel (255)
    // spreads out as 255 * weight:
    //   centre:        0.39905 * 0.39905 * 255 = 40.6  -> 41
    //   next to it:    0.39905 * 0.24203 * 255 = 24.6  -> 25
    //   diagonal:      0.24203 * 0.24203 * 255 = 14.9  -> 15
    const fs::path dir = temp_dir("impulse_values");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_impulse(21, 17, 10, 8));

    const RunResult result = run_cli({input, output, "1"});
    CHECK(result.exit_code == 0);
    if (result.exit_code != 0) {
        return;
    }
    const Image blurred = load_image(output);

    CHECK(blurred.data[blurred.index(10, 8)] == 41);
    CHECK(blurred.data[blurred.index(11, 8)] == 25);
    CHECK(blurred.data[blurred.index(10, 9)] == 25);
    CHECK(blurred.data[blurred.index(11, 9)] == 15);
}

static void test_larger_sigma_blurs_more() {
    // A stronger blur spreads the same light over more pixels, so the blob
    // is wider and its centre is dimmer.
    const fs::path dir = temp_dir("larger_sigma");
    const std::string input = (dir / "in.png").string();
    const std::string light = (dir / "sigma1.png").string();
    const std::string heavy = (dir / "sigma3.png").string();
    save_image(input, make_impulse(45, 41, 22, 20));

    CHECK(run_cli({input, light, "1"}).exit_code == 0);
    CHECK(run_cli({input, heavy, "3"}).exit_code == 0);
    if (!fs::exists(light) || !fs::exists(heavy)) {
        return;
    }
    const Image blurred_light = load_image(light);
    const Image blurred_heavy = load_image(heavy);

    CHECK(count_non_zero(blurred_heavy) > count_non_zero(blurred_light));
    CHECK(blurred_heavy.data[blurred_heavy.index(22, 20)] <
          blurred_light.data[blurred_light.index(22, 20)]);
}

static void test_source_file_is_untouched() {
    const fs::path dir = temp_dir("source_untouched");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_gradient(16, 9));
    const std::string before = read_file_bytes(input);
    CHECK(!before.empty());

    const RunResult result = run_cli({input, output, "3"});

    CHECK(result.exit_code == 0);
    CHECK(read_file_bytes(input) == before);
    // And the blur really ran: the Destination differs from the Source.
    if (fs::exists(output)) {
        CHECK(!same_image(load_image(output), load_image(input)));
    }
}

// ---- OpenMP Backend and the --backend / --threads flags ----

static void test_default_backend_and_threads_are_printed() {
    const fs::path dir = temp_dir("defaults_printed");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_gradient(5, 3));

    const RunResult result = run_cli({input, output, "1"});

    // With no flags the tool uses the OpenMP Backend on every thread the
    // OpenMP runtime offers, and says so.
    CHECK(result.exit_code == 0);
    CHECK(contains(result.out, "backend openmp"));
    CHECK(contains(result.out,
                   "threads " + std::to_string(omp_get_max_threads()) + ","));
}

static void test_chosen_backend_and_threads_are_printed() {
    const fs::path dir = temp_dir("flags_printed");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_gradient(5, 3));

    const RunResult openmp = run_cli({input, output, "1", "--backend", "openmp", "--threads", "3"});
    CHECK(openmp.exit_code == 0);
    CHECK(contains(openmp.out, "backend openmp"));
    CHECK(contains(openmp.out, "threads 3,"));

    // The flags may come in either order.
    const RunResult swapped = run_cli({input, output, "1", "--threads", "2", "--backend", "openmp"});
    CHECK(swapped.exit_code == 0);
    CHECK(contains(swapped.out, "backend openmp"));
    CHECK(contains(swapped.out, "threads 2,"));

    // The serial Backend always runs on one thread, and reports that even
    // when it is not what --threads asked for.
    const RunResult serial = run_cli({input, output, "1", "--backend", "serial"});
    CHECK(serial.exit_code == 0);
    CHECK(contains(serial.out, "backend serial"));
    CHECK(contains(serial.out, "threads 1,"));

    const RunResult serial_8 = run_cli({input, output, "1", "--backend", "serial", "--threads", "8"});
    CHECK(serial_8.exit_code == 0);
    CHECK(contains(serial_8.out, "threads 1,"));
}

// Random grey levels, the same ones on every run: std::mt19937 is specified
// exactly by the C++ standard, so a fixed seed gives the same numbers on any
// compiler. Noise has no pattern for a bug to hide behind: a row computed
// twice, skipped, or written to the wrong place changes the output.
static Image make_noise(int width, int height) {
    Image img;
    img.width = width;
    img.height = height;
    img.channels = 1;
    img.data.resize(static_cast<std::size_t>(width) * height);
    std::mt19937 generator(20261011u);   // constant seed
    for (std::uint8_t& value : img.data) {
        // mt19937 produces 32-bit numbers; keep the top 8 bits.
        value = static_cast<std::uint8_t>(generator() >> 24);
    }
    return img;
}

static void test_openmp_output_is_byte_identical_to_serial() {
    // 53 rows is a prime number, so no thread count here divides it evenly
    // (53 = 2*26+1 = 3*17+2 = 4*13+1 = 8*6+5): the static schedule has
    // left-over rows to hand out, and blocks of different sizes.
    const fs::path dir = temp_dir("serial_equals_openmp");
    const std::string input = (dir / "noise.png").string();
    const std::string reference = (dir / "serial.png").string();
    const std::string parallel = (dir / "openmp.png").string();
    save_image(input, make_noise(37, 53));

    for (const char* sigma : {"1", "3", "5"}) {
        CHECK(run_cli({input, reference, sigma, "--backend", "serial"}).exit_code == 0);
        const std::string expected = read_file_bytes(reference);
        CHECK(!expected.empty());
        // The Reference really blurred: its file is not a copy of the Source.
        CHECK(expected != read_file_bytes(input));

        for (const char* threads : {"1", "2", "3", "4", "8"}) {
            fs::remove(parallel);   // never compare against a stale file

            const RunResult result =
                run_cli({input, parallel, sigma, "--backend", "openmp", "--threads", threads});

            CHECK(result.exit_code == 0);
            // The whole output file, byte for byte: not "close", identical.
            CHECK(read_file_bytes(parallel) == expected);
        }
    }
}

static void test_more_threads_than_rows_still_works() {
    // 3 rows shared between 8 and then 64 threads: most threads get no row
    // at all, and the result must still be the Reference's.
    const fs::path dir = temp_dir("more_threads_than_rows");
    const std::string input = (dir / "in.png").string();
    const std::string reference = (dir / "serial.png").string();
    const std::string parallel = (dir / "openmp.png").string();
    save_image(input, make_noise(5, 3));

    CHECK(run_cli({input, reference, "1", "--backend", "serial"}).exit_code == 0);
    const std::string expected = read_file_bytes(reference);
    CHECK(!expected.empty());

    for (const char* threads : {"8", "64"}) {
        fs::remove(parallel);

        const RunResult result =
            run_cli({input, parallel, "1", "--backend", "openmp", "--threads", threads});

        CHECK(result.exit_code == 0);
        CHECK(result.err.empty());
        CHECK(contains(result.out, std::string("threads ") + threads + ","));
        CHECK(read_file_bytes(parallel) == expected);
    }
}

// Runs the tool with a valid input plus `flags`, which must be rejected with
// an error that mentions every text in `expected_in_error`, and without
// writing an output file.
static void check_flags_rejected(const std::vector<std::string>& flags,
                                 const std::vector<std::string>& expected_in_error) {
    const fs::path dir = temp_dir("rejected_flags");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_gradient(5, 3));

    std::vector<std::string> args = {input, output, "1"};
    args.insert(args.end(), flags.begin(), flags.end());
    const RunResult result = run_cli(args);

    CHECK(result.exit_code != 0);
    CHECK(result.out.empty());
    CHECK(!fs::exists(output));
    for (const std::string& expected : expected_in_error) {
        if (!contains(result.err, expected)) {
            std::cerr << "error message lacks '" << expected << "': " << result.err;
        }
        CHECK(contains(result.err, expected));
    }
}

static void test_unknown_backend_lists_the_valid_ones() {
    check_flags_rejected({"--backend", "cuda"}, {"cuda", "serial", "openmp"});
    // Names are matched exactly: no guessing, no silent fallback.
    check_flags_rejected({"--backend", "OpenMP"}, {"serial", "openmp"});
    check_flags_rejected({"--backend", ""}, {"serial", "openmp"});
}

static void test_bad_thread_count_says_what_is_valid() {
    for (const char* bad : {"0", "-1", "-8", "four", "", "2.5", "4x", "99999999999999999999"}) {
        check_flags_rejected({"--threads", bad}, {"--threads", "1 or more"});
        // Also rejected for the serial Backend, which would ignore the count:
        // a mistyped flag is never silently accepted.
        check_flags_rejected({"--backend", "serial", "--threads", bad},
                             {"--threads", "1 or more"});
    }
}

static void test_flag_without_a_value_is_rejected() {
    check_flags_rejected({"--backend"}, {"--backend", "serial", "openmp"});
    check_flags_rejected({"--threads"}, {"--threads", "Usage"});
    check_flags_rejected({"--backend", "serial", "--threads"}, {"--threads", "Usage"});
}

static void test_unknown_flag_lists_the_valid_ones() {
    check_flags_rejected({"--thread", "4"}, {"--thread", "--backend", "--threads"});
    check_flags_rejected({"--fast"}, {"--fast", "--backend", "--threads"});
    check_flags_rejected({"-t", "4"}, {"-t", "--backend", "--threads"});
}

int main() {
    // A test that throws is a failure too, not a crash with no explanation.
    try {
        test_destination_keeps_the_source_size();
        test_colour_jpg_becomes_grayscale_png();
        test_missing_input_names_the_path();
        test_unreadable_input_names_the_path();
        test_unwritable_output_names_the_path();
        test_success_reports_size_sigma_radius_and_time();
        test_radius_is_ceil_of_three_sigma();
        test_wrong_argument_count_prints_usage();
        test_flat_image_stays_exactly_flat();
        test_bright_pixel_becomes_a_symmetric_blob();
        test_sigma_1_blob_has_the_expected_values();
        test_larger_sigma_blurs_more();
        test_source_file_is_untouched();

        // OpenMP Backend and the --backend / --threads flags
        test_default_backend_and_threads_are_printed();
        test_chosen_backend_and_threads_are_printed();
        test_openmp_output_is_byte_identical_to_serial();
        test_more_threads_than_rows_still_works();
        test_unknown_backend_lists_the_valid_ones();
        test_bad_thread_count_says_what_is_valid();
        test_flag_without_a_value_is_rejected();
        test_unknown_flag_lists_the_valid_ones();
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
