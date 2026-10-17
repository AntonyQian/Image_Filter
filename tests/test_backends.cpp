// The OpenMP Backend must produce the serial Reference's Destination bit for
// bit, whatever the thread count, and so must the CUDA Backends in a CUDA
// build. A build without CUDA must reject them cleanly.

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

#ifdef IMGFILTER_HAS_CUDA

static void test_cuda_output_is_byte_identical_to_serial() {
    // 37 x 53 is not a multiple of the 16 x 16 block in either direction, so
    // the last blocks stick out past the Image. 5 x 3 at sigma 5 (a 31-wide
    // window) is smaller than one Tile, and every read crosses the border.
    struct Case {
        int width;
        int height;
        const char* sigma;
    };
    const Case cases[] = {{37, 53, "1"}, {37, 53, "3"}, {37, 53, "5"}, {5, 3, "5"}};

    const fs::path dir = temp_dir("serial_equals_cuda");
    const std::string input = (dir / "noise.png").string();
    const std::string reference = (dir / "serial.png").string();
    const std::string gpu = (dir / "cuda.png").string();

    for (const Case& c : cases) {
        save_image(input, make_noise(c.width, c.height));
        CHECK(run_cli({input, reference, c.sigma, "--backend", "serial"}).exit_code == 0);
        const std::string expected = read_file_bytes(reference);
        CHECK(!expected.empty());

        for (const char* backend : {"cuda-naive", "cuda-tiled"}) {
            fs::remove(gpu);   // never compare against a stale file

            const RunResult result = run_cli({input, gpu, c.sigma, "--backend", backend});

            CHECK(result.exit_code == 0);
            if (result.exit_code != 0) {
                std::cerr << backend << ": " << result.err;
            }
            CHECK(read_file_bytes(gpu) == expected);
        }
    }
}

#else

static void test_build_without_cuda_rejects_cuda_backends() {
    const fs::path dir = temp_dir("no_cuda_rejects");
    const std::string input = (dir / "in.png").string();
    const std::string output = (dir / "out.png").string();
    save_image(input, make_noise(5, 3));

    const RunResult result = run_cli({input, output, "1", "--backend", "cuda-naive"});

    CHECK(result.exit_code != 0);
    CHECK(contains(result.err, "no CUDA support"));
    CHECK(contains(result.err, "-DIMGFILTER_CUDA=ON"));
    CHECK(!fs::exists(output));
}

#endif

int main() {
    return run_tests({
        test_openmp_output_is_byte_identical_to_serial,
        test_more_threads_than_rows_still_works,
#ifdef IMGFILTER_HAS_CUDA
        test_cuda_output_is_byte_identical_to_serial,
#else
        test_build_without_cuda_rejects_cuda_backends,
#endif
    });
}
