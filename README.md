# Image_Filter

A C++17 command-line tool that applies a Gaussian blur to a grayscale image, with a serial, an OpenMP and two CUDA implementations.

## Prerequisites

* CMake 3.20+
* A C++17 compiler with OpenMP (developed with GCC 15.2.0 on MSYS2 UCRT64)
* Git

## Build

`cmake -S . -B build`

`cmake --build build`

The build type defaults to `Release`.

## Usage

`build/imgfilter <input image> <output.png> <sigma> [--backend serial|openmp|cuda-naive|cuda-tiled] [--threads N]`

* `sigma`: blur strength in pixels
* `--backend`: `serial`, `openmp`, `cuda-naive` or `cuda-tiled` (default: `openmp`; the CUDA ones need a CUDA build, see below)
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
The laptop CSV above was produced before `-ffp-contract=off` was added to the flags. On x86-64 without `-march=native` GCC cannot emit FMA instructions anyway, so the Reference's output is the same with or without it.

## GPU (CUDA) Backends

Two more Backends run the same blur on an NVIDIA GPU:

* `cuda-naive`: one GPU thread per Destination Pixel, reading the Source and the Weights from global memory.
* `cuda-tiled`: each 16 x 16 block loads its Tile of the Source plus a Halo of Radius Pixels into shared memory once, and reads the Weights from constant memory. Sigma is limited to 21 (the Weights must fit in 64 KB of constant memory).

Both are built to match the serial Reference bit for bit, and the test suite checks that in a CUDA build: they add the same products in the same order, and multiply-add fusion is off on both sides (`--fmad=false` for nvcc, `-ffp-contract=off` for GCC).

CUDA is a build option, off by default. Without it, `--backend cuda-naive` and `--backend cuda-tiled` fail with a message saying the build has no CUDA support. CUDA builds are Linux only (nvcc needs MSVC as its host compiler on Windows, and this project uses GCC). They are meant to be built and run on Google Colab with `notebooks/colab_gpu.ipynb`.

### Build and run on the cloud machine

```bash
cmake -S . -B build-cuda -DIMGFILTER_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
build-cuda/imgfilter data/input.png data/output-gpu.png 3 --backend cuda-tiled
build-cuda/imgfilter bench --sizes 1024,2048,4096 --sigmas 1,3,5 --csv results/bench-gpu-colab-t4.csv
```

`75` is the compute capability of a T4. Use the value `nvidia-smi --query-gpu=compute_cap --format=csv` prints for the assigned GPU.

For a CUDA Backend the blur command also prints the Host-to-Device Transfer, Kernel and Device-to-Host Transfer times. `bench` measures the Reference, the OpenMP thread Sweep, then each CUDA Backend once per size and sigma. `--backends` picks which non-Reference Backends run, for example `--backends cuda-naive,cuda-tiled`. CUDA rows leave threads and efficiency empty, and fill the `h2d_ms`, `kernel_ms` and `d2h_ms` columns.

### GPU results

TODO (ticket #34): run the command above on an otherwise idle Colab runtime, commit the CSV, and fill in:

- **GPU:**
- **Host CPU:**
- **NVIDIA driver / CUDA toolkit:**
- **CUDA architecture:**
- **C++ flags:** (from the bench header)
- **CUDA flags:** (from the bench header)

Table: `python scripts/csv_to_table.py results/bench-gpu-colab-t4.csv --threads 1,2`

Every speedup in this table is against the serial row measured on the same Colab machine, never against the laptop numbers above.
