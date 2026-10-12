// Loading the Source and saving the Destination: colour inputs, files that
// cannot be read or written, and the Source file being left alone.

// Declarations only (no STB_IMAGE_WRITE_IMPLEMENTATION here): the code itself
// is compiled once, in src/io.cpp. Needed for the colour (RGB) test inputs.
#include <stb_image_write.h>

#include "test_utilities.hpp"

static void test_colour_jpg_becomes_grayscale_png() {
    // The project can only save PNGs, so this input is written with stb
    // directly: a 3-channel (RGB) JPG, to exercise both JPG decoding and the
    // conversion to grayscale. JPG is lossy, so only the shape is checked.
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

static void test_colour_input_gets_the_expected_grey_level() {
    // PNG is lossless, so here the grey level can be checked exactly. Every
    // pixel is the same colour, so the grayscale Source is flat, and a flat
    // image stays flat under the blur. Every Destination pixel must therefore
    // be exactly the grey level stb gives this colour.
    const fs::path dir = temp_dir("colour_grey_level");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    const int width = 16;
    const int height = 8;
    const int r = 200;
    const int g = 100;
    const int b = 50;
    std::vector<std::uint8_t> rgb;
    for (int i = 0; i < width * height; ++i) {
        rgb.push_back(static_cast<std::uint8_t>(r));
        rgb.push_back(static_cast<std::uint8_t>(g));
        rgb.push_back(static_cast<std::uint8_t>(b));
    }
    // The last argument is the stride: bytes from one row to the next.
    CHECK(stbi_write_png(input.c_str(), width, height, 3, rgb.data(), width * 3) != 0);

    // stb's own formula (stbi__compute_y in third_party/stb_image.h):
    //   grey = (r * 77 + g * 150 + b * 29) >> 8
    // For this colour: (200 * 77 + 100 * 150 + 50 * 29) >> 8 = 31850 >> 8 = 124
    const int grey = (r * 77 + g * 150 + b * 29) >> 8;
    CHECK(grey == 124);

    const RunResult result = run_cli({input, output, "1"});

    CHECK(result.exit_code == 0);
    CHECK(fs::exists(output));
    if (fs::exists(output)) {
        const Image expected = make_flat(width, height, static_cast<std::uint8_t>(grey));
        CHECK(same_image(load_image(output), expected));
    }
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

int main() {
    return run_tests({
        test_colour_jpg_becomes_grayscale_png,
        test_colour_input_gets_the_expected_grey_level,
        test_missing_input_names_the_path,
        test_unreadable_input_names_the_path,
        test_unwritable_output_names_the_path,
        test_source_file_is_untouched,
    });
}
