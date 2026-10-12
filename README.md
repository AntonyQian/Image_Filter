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


