#pragma once

#include "cuda_blur.h"
#include "image.h"

enum class Backend {
    Serial,      // the Reference
    OpenMP,      // rows split between threads
    CudaNaive,   // GPU, one thread per Pixel, global memory only
    CudaTiled,   // GPU, shared-memory Tile and Halo, constant-memory Weights
};

// The name the --backend flag and the benchmark table use: "serial",
// "openmp", "cuda-naive" or "cuda-tiled".
const char* backend_name(Backend backend);

// True for the two Backends that run on the GPU.
bool is_cuda(Backend backend);

// Checks, without running anything, that `backend` can blur with `sigma` in
// this build: a CUDA Backend needs a CUDA build, and cuda-tiled needs a sigma
// of at most cuda_tiled_max_sigma. Expects a finite sigma > 0.
// Throws std::invalid_argument saying what is wrong.
void check_backend_supported(Backend backend, double sigma);

// Blurs `source` and returns the Destination, a new Image of the same size.
// `threads` is only used by the OpenMP Backend, where it must be 1 or more.
// Borders are clamp-to-edge.
// `timings`, if not null, is filled by the CUDA Backends with where the time
// went; the CPU Backends leave it untouched.
//
// Throws std::invalid_argument if sigma is not finite and positive, if it is
// so large the radius doesn't fit in an int, if OpenMP gets threads < 1, or if
// check_backend_supported rejects the Backend. A CUDA Backend also throws
// std::runtime_error if the GPU fails.
Image gaussian_blur(const Image& source, double sigma, Backend backend, int threads,
                    GpuTimings* timings = nullptr);
