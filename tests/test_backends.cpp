// The OpenMP Backend must produce the serial Reference's Destination bit for
// bit, whatever the thread count.

#include "test_utilities.hpp"

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

int main() {
    return run_tests({
        test_openmp_output_is_byte_identical_to_serial,
        test_more_threads_than_rows_still_works,
    });
}
