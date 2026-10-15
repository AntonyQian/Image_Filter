# Image_Filter

A C++17 command-line tool that applies a Gaussian blur to a grayscale image, with a serial and an OpenMP implementation.

## Prerequisites

* CMake 3.20+
* A C++17 compiler with OpenMP (developed with GCC 15.2.0 on MSYS2 UCRT64)
* Git

## Build

`cmake -S . -B build`

`cmake --build build`

The build type defaults to `Release`.

## Usage

`build/imgfilter <input image> <output.png> <sigma> [--backend serial|openmp] [--threads N]`

* `sigma`: blur strength in pixels
* `--backend`: `serial` or `openmp` (default: `openmp`)
* `--threads`: thread count for the OpenMP backend (default: all available)

Example:

`build/imgfilter data/input.png data/output.png 3 --backend openmp --threads 4`

## Benchmark Testing

- **CPU:** Intel Core Ultra 7 256V, 8 cores (4 P + 4 E), 8 threads
- **Cache:** L1 832 KB, L2 14.0 MB, L3 12.0 MB (totals across all cores)
- **Memory:** 16.0 GB LPDDR5
- **OS:** Windows 11 Pro 25H2 (build 26200.9457)
- **Compiler:** GCC 15.2.0 (MSYS2 UCRT64, Rev8)
- **Flags:** `-O3 -DNDEBUG -std=gnu++17 -fopenmp -Wall -Wextra`
- **Build Type:** Release
- **Conditions:** plugged in, Windows power mode "Best performance", no other apps running


**Command used:**

```bash
build/imgfilter bench --sizes 1024,2048,4096 --sigmas 1,3,5 --csv results/bench.csv
```