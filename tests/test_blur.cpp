// What the blur does to the pixels: a flat image stays flat (Weights sum to 1,
// clamp-to-edge borders) and a blurred bright pixel has the values worked out
// by hand.

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

int main() {
    return run_tests({
        test_flat_image_stays_exactly_flat,
        test_sigma_1_blob_has_the_expected_values,
    });
}
