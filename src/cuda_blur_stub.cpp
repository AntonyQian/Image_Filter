// The stand-in for cuda_blur.cu, compiled when IMGFILTER_CUDA is OFF (the
// default). It lets the rest of the project name the CUDA Backends without
// any CUDA code, and turns asking for one into a clear error.

#include "cuda_blur.h"

#include <stdexcept>

bool cuda_built() {
    return false;
}

CudaDeviceInfo cuda_device_info() {
    throw std::invalid_argument(no_cuda_message);
}

void cuda_blur(const Image&, const Weights&, CudaKernel, Image&, GpuTimings*) {
    throw std::invalid_argument(no_cuda_message);
}
