#pragma once

// The GPU module's interface. Plain C++ only: no CUDA types or headers, so
// everything that includes this still compiles with GCC alone.
//
// Two files implement it, and CMake compiles exactly one of them:
//   cuda_blur.cu       the real GPU code, compiled by nvcc (IMGFILTER_CUDA=ON)
//   cuda_blur_stub.cpp a stand-in that rejects every call (the default build)

#include <cstddef>
#include <string>

#include "image.h"
#include "weights.h"

// The two Kernels a CUDA Backend can launch.
enum class CudaKernel {
    Naive,   // one thread per Destination Pixel, everything read from global memory
    Tiled,   // a shared-memory Tile plus Halo per block, Weights in constant memory
};

// Where the time of one CUDA blur goes, in milliseconds, measured with GPU
// events. Device allocation and freeing are in none of the three: they are
// only in the total the caller measures around the whole call.
struct GpuTimings {
    double h2d_ms = 0.0;      // Host-to-Device Transfer of the Source and the Weights
    double kernel_ms = 0.0;   // the Kernel itself
    double d2h_ms = 0.0;      // Device-to-Host Transfer of the Destination
};

// The tiled Kernel keeps all (2 * radius + 1)^2 Weights in constant memory,
// which holds 64 KB = 16384 floats. 127 * 127 = 16129 fits, 129 * 129 does
// not, so the Radius is at most 63, which is ceil(3 * sigma) <= 63: sigma <= 21.
constexpr int cuda_tiled_max_radius = 63;
constexpr double cuda_tiled_max_sigma = cuda_tiled_max_radius / 3.0;   // 21

// True if this build was configured with IMGFILTER_CUDA=ON.
bool cuda_built();

// The error for asking a build without CUDA to use the GPU.
inline constexpr const char* no_cuda_message =
    "this build has no CUDA support; reconfigure with -DIMGFILTER_CUDA=ON "
    "on a machine with an NVIDIA GPU and the CUDA toolkit";

// What the bench header and the blur summary print about the GPU.
struct CudaDeviceInfo {
    std::string name;
    int compute_major = 0;          // compute capability, e.g. 7.5 for a T4
    int compute_minor = 0;
    int multiprocessors = 0;        // number of SMs
    std::size_t total_memory_bytes = 0;
    int driver_version = 0;         // e.g. 12040 for CUDA 12.4
    int runtime_version = 0;
};

// Describes Device 0, the one every Kernel runs on.
// Throws std::invalid_argument in a build without CUDA, and std::runtime_error
// naming the CUDA error if there is no usable GPU.
CudaDeviceInfo cuda_device_info();

// Blurs `source` into `destination` (already sized like `source`) on the GPU
// with the given Kernel. Fills `timings` if it is not null.
// Expects a tiled Radius no larger than cuda_tiled_max_radius; gaussian_blur
// checks that before calling.
// Throws std::invalid_argument in a build without CUDA, and std::runtime_error
// naming the failing CUDA call and its error otherwise.
void cuda_blur(const Image& source, const Weights& weights, CudaKernel kernel,
               Image& destination, GpuTimings* timings);
