// The GPU module, compiled by nvcc when IMGFILTER_CUDA is ON. Everything CUDA
// stays in this file; the rest of the project only sees cuda_blur.h.
//
// Bit-for-bit rule: both Kernels compute a Destination Pixel exactly as
// blur_row in gaussian_blur.cpp does (see the summation contract there). CMake
// compiles this file with --fmad=false and no fast-math, so `sum += w * p` is
// a separately rounded multiply and add, the same IEEE operations the CPU does.

#include "cuda_blur.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace {

// ---- Launch shape ----

// Both Kernels use 16 x 16 = 256 threads per block, one thread per
// Destination Pixel. 256 is a multiple of the 32-thread warp, so no warp is
// part empty inside the Image; it is small enough that several blocks fit on
// one SM at once (good occupancy) and large enough that the tiled Kernel's
// Halo is a modest share of its Tile. 16 wide means a warp covers two rows of
// 16 neighbouring Pixels, so its global reads are close together.
constexpr int block_side = 16;

// ---- Error handling ----

// Throws std::runtime_error naming the call and the CUDA error, so a GPU
// failure is never a silently wrong Image.
void check(cudaError_t status, const char* call) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + call + ": " +
                                 cudaGetErrorString(status));
    }
}

#define CUDA_CHECK(call) check((call), #call)

// ---- RAII owners ----
// Each frees its resource in its destructor, so an exception on any path
// (a failed Transfer, a failed launch) never leaks Device memory. Destructors
// must not throw, so their own errors are ignored.

// A Device allocation of `bytes` bytes.
class DeviceBuffer {
public:
    DeviceBuffer(std::size_t bytes, const char* what, const Image& image) {
        const cudaError_t status = cudaMalloc(&pointer_, bytes);
        if (status == cudaErrorMemoryAllocation) {
            std::ostringstream message;
            message << "cannot allocate " << bytes << " bytes of GPU memory for the " << what
                    << " of a " << image.width << "x" << image.height << " image";
            throw std::runtime_error(message.str());
        }
        check(status, "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(pointer_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    template <typename T>
    T* as() const { return static_cast<T*>(pointer_); }

private:
    void* pointer_ = nullptr;
};

// A GPU event: a timestamp the GPU records when it reaches that point in its
// queue of work. The difference between two is time spent on the Device, not
// the time a Host call took to return.
class Event {
public:
    Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event() { cudaEventDestroy(event_); }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record() { CUDA_CHECK(cudaEventRecord(event_)); }
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

double elapsed_ms(const Event& start, const Event& stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
    return ms;
}

// ---- Device helpers ----

__device__ int clamp_int(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

// Exactly the last step of blur_row: +0.5f, truncate (a float to int cast
// rounds toward zero on both sides), clamp to 0..255.
__device__ std::uint8_t to_pixel(float sum) {
    const int rounded = static_cast<int>(sum + 0.5f);
    return static_cast<std::uint8_t>(clamp_int(rounded, 0, 255));
}

// ---- Naive Kernel ----

// One thread per Destination Pixel. Every thread reads its whole window of
// Source Pixels and all the Weights from global memory: (2r + 1)^2 of each.
// Neighbouring threads read almost the same Source Pixels, and only the
// caches stop that from being (2r + 1)^2 trips to DRAM per Pixel.
__global__ void blur_naive(const std::uint8_t* source, const float* weights, int width,
                           int height, int radius, std::uint8_t* destination) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    // The grid is rounded up to whole blocks, so the last ones stick out
    // past the Image. Those threads write nothing.
    if (x >= width || y >= height) {
        return;
    }

    const int side = 2 * radius + 1;
    float sum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int source_y = clamp_int(y + dy, 0, height - 1);   // clamp-to-edge
        for (int dx = -radius; dx <= radius; ++dx) {
            const int source_x = clamp_int(x + dx, 0, width - 1);
            // 64-bit index, as Image::index does, so huge Images can't overflow.
            const std::size_t index = static_cast<std::size_t>(source_y) * width + source_x;
            sum += weights[(dy + radius) * side + (dx + radius)] *
                   static_cast<float>(source[index]);
        }
    }
    destination[static_cast<std::size_t>(y) * width + x] = to_pixel(sum);
}

// ---- Tiled Kernel ----

// The Weights for the tiled Kernel, in constant memory. Constant memory has
// its own cache built for broadcast: when every thread of a warp reads the
// same address, which they do here (same dx, dy at the same moment), one
// read serves all 32. Sized for the largest Radius allowed.
constexpr int max_side = 2 * cuda_tiled_max_radius + 1;
__constant__ float constant_weights[max_side * max_side];

// Each block owns a block_side x block_side Tile of the Destination. Its
// threads first copy the Source Pixels the whole Tile needs, the Tile plus a
// Halo `radius` wide on every side, into shared memory, each Pixel once. Then
// every thread computes its Pixel from shared memory only.
//
// Why that cuts global-memory traffic: the naive Kernel reads (2r + 1)^2
// Source bytes per Pixel. Here a block reads (16 + 2r)^2 bytes for 16 * 16
// Pixels, about ((16 + 2r) / 16)^2 per Pixel: at r = 15 (sigma 5) that is
// 46^2 / 256 = 8.3 bytes per Pixel instead of 961.
__global__ void blur_tiled(const std::uint8_t* source, int width, int height, int radius,
                           std::uint8_t* destination) {
    // Sized at launch (the third <<< >>> argument), because the Halo depends
    // on the Radius. Kept as bytes, as in the Source, and converted to float
    // when used, as the Reference does.
    extern __shared__ std::uint8_t region[];
    const int region_side = block_side + 2 * radius;
    // threadIdx and blockIdx are unsigned; int keeps the arithmetic below
    // signed, where origin can be negative.
    const int local_x = static_cast<int>(threadIdx.x);
    const int local_y = static_cast<int>(threadIdx.y);

    // Source coordinates of the top-left corner of the Tile plus Halo. Can be
    // negative or past the Image: clamping below handles that.
    const int origin_x = static_cast<int>(blockIdx.x) * block_side - radius;
    const int origin_y = static_cast<int>(blockIdx.y) * block_side - radius;

    // The region is larger than the block, so each thread loads several Pixels,
    // stepping by the block size. Clamping here makes the Halo the place
    // where Border handling happens: region[ty][tx] is the Source Pixel at
    // clamp(origin + t), the same Pixel the Reference reads.
    for (int ty = local_y; ty < region_side; ty += block_side) {
        const int source_y = clamp_int(origin_y + ty, 0, height - 1);
        for (int tx = local_x; tx < region_side; tx += block_side) {
            const int source_x = clamp_int(origin_x + tx, 0, width - 1);
            region[ty * region_side + tx] =
                source[static_cast<std::size_t>(source_y) * width + source_x];
        }
    }

    // No thread may read the region before every thread has finished writing
    // its part. Every thread reaches this line, including those whose Pixel is
    // outside the Image, which is why they return only after it.
    __syncthreads();

    const int x = origin_x + radius + local_x;
    const int y = origin_y + radius + local_y;
    if (x >= width || y >= height) {
        return;
    }

    // Pixel (x + dx, y + dy) sits at region row local_y + radius + dy, column
    // local_x + radius + dx, because origin = block corner - radius.
    const int side = 2 * radius + 1;
    float sum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int region_row = (local_y + radius + dy) * region_side;
        for (int dx = -radius; dx <= radius; ++dx) {
            sum += constant_weights[(dy + radius) * side + (dx + radius)] *
                   static_cast<float>(region[region_row + local_x + radius + dx]);
        }
    }
    destination[static_cast<std::size_t>(y) * width + x] = to_pixel(sum);
}

}  // namespace

bool cuda_built() {
    return true;
}

CudaDeviceInfo cuda_device_info() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));   // fails with no driver or no GPU

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    CudaDeviceInfo info;
    info.name = properties.name;
    info.compute_major = properties.major;
    info.compute_minor = properties.minor;
    info.multiprocessors = properties.multiProcessorCount;
    info.total_memory_bytes = properties.totalGlobalMem;
    CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
    return info;
}

void cuda_blur(const Image& source, const Weights& weights, CudaKernel kernel,
               Image& destination, GpuTimings* timings) {
    const std::size_t image_bytes = source.data.size();
    const std::size_t weight_bytes = weights.values.size() * sizeof(float);
    const bool tiled = kernel == CudaKernel::Tiled;

    // Allocation is outside the three timed steps, but inside the caller's
    // total: a Run pays for it, as a CPU Run pays for its Destination.
    DeviceBuffer device_source(image_bytes, "Source", source);
    DeviceBuffer device_destination(image_bytes, "Destination", source);
    // The tiled Kernel reads constant memory, so only the naive one needs
    // a global Weights buffer. One byte keeps the class simple when unused.
    DeviceBuffer device_weights(tiled ? 1 : weight_bytes, "Weights", source);

    Event start;
    Event after_h2d;
    Event after_kernel;
    Event after_d2h;

    // Host-to-Device Transfer: the Source and the Weights built on the Host.
    start.record();
    CUDA_CHECK(cudaMemcpy(device_source.as<void>(), source.data.data(), image_bytes,
                          cudaMemcpyHostToDevice));
    if (tiled) {
        CUDA_CHECK(cudaMemcpyToSymbol(constant_weights, weights.values.data(), weight_bytes));
    } else {
        CUDA_CHECK(cudaMemcpy(device_weights.as<void>(), weights.values.data(), weight_bytes,
                              cudaMemcpyHostToDevice));
    }
    after_h2d.record();

    // Enough blocks to cover the Image, rounded up.
    const dim3 block(block_side, block_side);
    const dim3 grid((source.width + block_side - 1) / block_side,
                    (source.height + block_side - 1) / block_side);

    if (tiled) {
        const int region_side = block_side + 2 * weights.radius;
        const std::size_t shared_bytes = static_cast<std::size_t>(region_side) * region_side;
        blur_tiled<<<grid, block, shared_bytes>>>(device_source.as<const std::uint8_t>(),
                                                  source.width, source.height, weights.radius,
                                                  device_destination.as<std::uint8_t>());
    } else {
        blur_naive<<<grid, block>>>(device_source.as<const std::uint8_t>(),
                                    device_weights.as<const float>(), source.width,
                                    source.height, weights.radius,
                                    device_destination.as<std::uint8_t>());
    }
    // A launch returns at once. cudaGetLastError catches a launch that never
    // started (bad shape, too much shared memory); waiting on the event after
    // it catches an error while the Kernel ran.
    CUDA_CHECK(cudaGetLastError());
    after_kernel.record();
    CUDA_CHECK(cudaEventSynchronize(after_kernel.get()));

    // Device-to-Host Transfer: the Destination.
    CUDA_CHECK(cudaMemcpy(destination.data.data(), device_destination.as<void>(), image_bytes,
                          cudaMemcpyDeviceToHost));
    after_d2h.record();
    CUDA_CHECK(cudaEventSynchronize(after_d2h.get()));

    if (timings != nullptr) {
        timings->h2d_ms = elapsed_ms(start, after_h2d);
        timings->kernel_ms = elapsed_ms(after_h2d, after_kernel);
        timings->d2h_ms = elapsed_ms(after_kernel, after_d2h);
    }
    // The buffers and events are freed here, in reverse order of creation.
}
