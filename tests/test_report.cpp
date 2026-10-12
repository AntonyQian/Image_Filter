// The line a successful run prints: image size, sigma, radius, Backend,
// thread count and time.

// Only for omp_get_max_threads(), the thread count the tool defaults to.
#include <omp.h>

#include "test_utilities.hpp"

static void test_success_reports_size_sigma_radius_and_time() {
    const fs::path dir = temp_dir("reports_run");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_gradient(5, 3));

    const RunResult result = run_cli({input, output, "1"});

    CHECK(result.exit_code == 0);
    CHECK(result.err.empty());
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
    CHECK(contains(run_cli({input, output, "5"}).out, "radius 15"));    // 15  -> 15
}

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
    const RunResult serial = run_cli({input, output, "1", "--backend", "serial", "--threads", "8"});
    CHECK(serial.exit_code == 0);
    CHECK(contains(serial.out, "backend serial"));
    CHECK(contains(serial.out, "threads 1,"));
}

int main() {
    return run_tests({
        test_success_reports_size_sigma_radius_and_time,
        test_radius_is_ceil_of_three_sigma,
        test_default_backend_and_threads_are_printed,
        test_chosen_backend_and_threads_are_printed,
    });
}
