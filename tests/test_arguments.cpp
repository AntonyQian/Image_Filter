// Arguments the tool must refuse: wrong argument count, bad --backend and
// --threads flags, and sigmas that cannot produce a blur. Every refusal means
// a non-zero exit code, a message on the error stream, and no output file.

#include "test_utilities.hpp"

static void test_wrong_argument_count_prints_usage() {
    const RunResult none = run_cli({});
    CHECK(none.exit_code != 0);
    CHECK(contains(none.err, "Usage"));

    // Sigma is required: input and output alone are not enough.
    const RunResult no_sigma = run_cli({"in.png", "out.png"});
    CHECK(no_sigma.exit_code != 0);
    CHECK(contains(no_sigma.err, "Usage"));

    const RunResult too_many = run_cli({"in.png", "out.png", "1", "extra"});
    CHECK(too_many.exit_code != 0);
    CHECK(contains(too_many.err, "Usage"));
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
}

static void test_bad_thread_count_says_what_is_valid() {
    for (const char* bad : {"0", "-1", "four", "", "2.5", "4x", "99999999999999999999"}) {
        check_flags_rejected({"--threads", bad}, {"--threads", "1 or more"});
    }
    // Also rejected for the serial Backend, which would ignore the count:
    // a mistyped flag is never silently accepted.
    check_flags_rejected({"--backend", "serial", "--threads", "0"},
                         {"--threads", "1 or more"});
}

static void test_flag_without_a_value_is_rejected() {
    check_flags_rejected({"--backend"}, {"--backend", "serial", "openmp"});
    check_flags_rejected({"--threads"}, {"--threads", "Usage"});
}

static void test_unknown_flag_lists_the_valid_ones() {
    check_flags_rejected({"--thread", "4"}, {"--thread", "--backend", "--threads"});
    check_flags_rejected({"-t", "4"}, {"-t", "--backend", "--threads"});
}

static void check_sigma_is_rejected(const std::string& sigma) {
    // The same folder is reused for every sigma; temp_dir empties it first,
    // so an output file left by an earlier case cannot be mistaken for ours.
    const fs::path dir = temp_dir("rejected_sigma");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_gradient(5, 3));

    const RunResult result = run_cli({input, output, sigma});

    CHECK(result.exit_code != 0);
    CHECK(contains(result.err, "sigma"));
    CHECK(!fs::exists(output));
    if (result.exit_code == 0 || !contains(result.err, "sigma") || fs::exists(output)) {
        std::cerr << "  (while checking sigma '" << sigma << "')\n";
    }
}

static void test_sigma_that_cannot_blur_is_rejected() {
    check_sigma_is_rejected("0");
    check_sigma_is_rejected("-1");
    // std::stod accepts these spellings and returns a real NaN or infinity,
    // so they get past "is it a number?" and must be caught by the blur.
    check_sigma_is_rejected("nan");
    check_sigma_is_rejected("inf");
}

static void test_sigma_that_is_not_a_number_is_rejected() {
    check_sigma_is_rejected("abc");
    check_sigma_is_rejected("");
    // Starts like a number but has junk after it: not silently read as 3.
    check_sigma_is_rejected("3abc");
}

static void test_sigma_too_large_is_rejected() {
    // 1e300 is a finite double, but its radius (3e300) is far beyond what an
    // int can hold. Converting it would be undefined behaviour.
    check_sigma_is_rejected("1e300");
    // Sigma 1e6 has a radius that fits in an int (3,000,000), but its window
    // needs about 3.6e13 Weights, or 144 TB of floats: the allocation fails
    // (std::bad_alloc) and must be reported, not crash.
    check_sigma_is_rejected("1e6");
    // Sigma 3e8: the window has more Weights than a std::vector can hold at
    // all, which fails differently (std::length_error).
    check_sigma_is_rejected("3e8");
}

int main() {
    return run_tests({
        test_wrong_argument_count_prints_usage,
        test_unknown_backend_lists_the_valid_ones,
        test_bad_thread_count_says_what_is_valid,
        test_flag_without_a_value_is_rejected,
        test_unknown_flag_lists_the_valid_ones,
        test_sigma_that_cannot_blur_is_rejected,
        test_sigma_that_is_not_a_number_is_rejected,
        test_sigma_too_large_is_rejected,
    });
}
