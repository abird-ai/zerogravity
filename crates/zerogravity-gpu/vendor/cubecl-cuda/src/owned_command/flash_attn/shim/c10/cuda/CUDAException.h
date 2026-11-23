#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define C10_CUDA_CHECK(EXPR) do { \
  const cudaError_t zg_cuda_status = (EXPR); \
  if (zg_cuda_status != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(zg_cuda_status)); \
  } \
} while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
