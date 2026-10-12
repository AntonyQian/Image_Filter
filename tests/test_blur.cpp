// What the blur does to the pixels: Weights that sum to 1, clamp-to-edge
// borders, the shape and values of a blurred bright pixel, and edge cases
// (images smaller than the window, a sigma too small to blur).

#include "test_utilities.hpp"

static void test_flat_image_stays_exactly_flat() {
    // If the Weights sum to exactly 1 and the border reuses the nearest edge
    // pixel, blurring a constant image changes nothing: not in the middle and
    // not at the edges. 20x12 is smaller than the sigma 5 window (31 wide),
    // so every pixel reads past the border at least once.
    const fs::path dir = temp_dir("flat");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();

    for (const char* sigma : {"1", "3", "5"}) {
        for (const int level : {0, 1, 127, 128, 255}) {
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

static void test_image_smaller_than_the_window_stays_flat() {
    // At sigma 5 the window is 31 pixels wide, far wider than these images,
    // so almost every neighbour is past the border and is read as the nearest
    // edge pixel instead. Every neighbour has the same grey level, and
    // Weights that sum to 1 must give that level straight back.
    const fs::path dir = temp_dir("tiny_image");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();

    // Both Backends: with 4 threads and only 1 or 2 rows, the OpenMP Backend
    // also has threads that are given no row at all.
    for (const char* backend : {"serial", "openmp"}) {
        for (const int side : {1, 2}) {
            for (const int level : {0, 127, 255}) {
                const Image flat = make_flat(side, side, static_cast<std::uint8_t>(level));
                save_image(input, flat);
                fs::remove(output);   // so a stale file cannot pass the check

                const RunResult result =
                    run_cli({input, output, "5", "--backend", backend, "--threads", "4"});

                CHECK(result.exit_code == 0);
                CHECK(result.err.empty());
                CHECK(fs::exists(output));
                if (fs::exists(output)) {
                    CHECK(same_image(load_image(output), flat));
                }
            }
        }
    }
}

static void test_tiny_sigma_leaves_the_image_unchanged() {
    // 1e-200 is a valid sigma (finite and greater than zero), just absurdly
    // small. Its radius is ceil(3e-200) = 1, a 3x3 window, but the Gaussian
    // is so narrow that the centre gets all the weight and the 8 neighbours
    // get none. So the Destination must equal the Source.
    //
    // The trap: sigma * sigma is 1e-400, too small for a double, so it
    // becomes 0. Dividing by it would give 0 / 0 = NaN at the centre and NaN
    // Weights, which means a garbage image with exit code 0.
    const fs::path dir = temp_dir("tiny_sigma");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    const Image source = make_gradient(16, 9);   // not flat: a real blur would show

    for (const char* backend : {"serial", "openmp"}) {
        save_image(input, source);
        fs::remove(output);   // so a stale file cannot pass the check

        const RunResult result = run_cli({input, output, "1e-200", "--backend", backend});

        CHECK(result.exit_code == 0);
        CHECK(result.err.empty());
        CHECK(contains(result.out, "radius 1"));
        CHECK(fs::exists(output));
        if (fs::exists(output)) {
            CHECK(same_image(load_image(output), source));
        }
    }
}

int main() {
    return run_tests({
        test_flat_image_stays_exactly_flat,
        test_bright_pixel_becomes_a_symmetric_blob,
        test_sigma_1_blob_has_the_expected_values,
        test_image_smaller_than_the_window_stays_flat,
        test_tiny_sigma_leaves_the_image_unchanged,
    });
}
