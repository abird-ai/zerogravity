// Source-derived from pinned Torch cf30153c4: Reduce.cuh and MeanOps.
// Adapted Torch reduction; see LICENSE.pytorch in this directory.
// Keep each original ATen op's global F32 output; caller owns all buffers.
extern "C" __global__ void zg_vllm_input_gamma_widen(
    const unsigned short* source, float* output) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ZG_COLUMNS) output[i] = __uint_as_float((unsigned int)source[i] << 16);
}

extern "C" __global__ void zg_vllm_input_gamma_shift(
    const float* source, float* output) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ZG_COLUMNS) output[i] = __fadd_rn(source[i], 1.0f);
}

extern "C" __global__ void zg_vllm_input_widen(
    const unsigned short* source, float* output) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ZG_ROWS * ZG_COLUMNS) output[i] = __uint_as_float((unsigned int)source[i] << 16);
}

extern "C" __global__ void zg_vllm_input_square(
    const float* source, float* output) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ZG_ROWS * ZG_COLUMNS) output[i] = __fmul_rn(source[i], source[i]);
}

extern "C" __global__ void zg_vllm_input_mean(
    const float* source, float* output) {
  // Flatten the original logical block.x/block.y without changing warp lanes.
  const unsigned int x = threadIdx.x % ZG_MEAN_WIDTH;
  const unsigned int y = threadIdx.x / ZG_MEAN_WIDTH;
  const unsigned int row = blockIdx.x * ZG_MEAN_HEIGHT + y;
  float accum0 = 0.0f, accum1 = 0.0f, accum2 = 0.0f, accum3 = 0.0f;
  // Reduce.cuh input_vectorized_thread_reduce_impl<float,4>: each ordered
  // accumulator consumes one lane of each aligned four-word vector.
  for (unsigned int index = x; index < ZG_COLUMNS / 4; index += ZG_MEAN_WIDTH) {
    const unsigned int offset = row * ZG_COLUMNS + index * 4;
    accum0 = __fadd_rn(accum0, source[offset]);
    accum1 = __fadd_rn(accum1, source[offset + 1]);
    accum2 = __fadd_rn(accum2, source[offset + 2]);
    accum3 = __fadd_rn(accum3, source[offset + 3]);
  }
  float value = __fadd_rn(__fadd_rn(__fadd_rn(accum0, accum1), accum2), accum3);
  __shared__ float shared[ZG_MEAN_WIDTH * ZG_MEAN_HEIGHT];
  if (ZG_MEAN_WIDTH > 32) {
    shared[threadIdx.x] = value;
    for (unsigned int offset = ZG_MEAN_WIDTH / 2; offset >= 32; offset >>= 1) {
      __syncthreads();
      if (x < offset) {
        value = __fadd_rn(value, shared[threadIdx.x + offset]);
        shared[threadIdx.x] = value;
      }
    }
  }
  __syncthreads();
  // Source CUDA block_x_reduce uses decreasing shuffle offsets.
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  if (x == 0) output[row] = __fmul_rn(value, ZG_MEAN_FACTOR);
}

extern "C" __global__ void zg_vllm_input_epsilon(
    const float* source, float* output) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ZG_ROWS) output[i] = __fadd_rn(source[i], __uint_as_float(0x358637bdu));
}

extern "C" __global__ void zg_vllm_input_inverse(
    const float* source, float* output) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ZG_ROWS) {
    // Pinned Torch SM86 rsqrt(float) normal path is MUFU.RSQ. The preceding
    // squared mean + epsilon is >=1e-6, excluding its denormal rescale branch.
    float value;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(value) : "f"(source[i]));
    output[i] = value;
  }
}

extern "C" __global__ void zg_vllm_input_normalize(
    const float* source, const float* inverse, float* output) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ZG_ROWS * ZG_COLUMNS) output[i] = __fmul_rn(source[i], inverse[i / ZG_COLUMNS]);
}

extern "C" __global__ void zg_vllm_input_weight(
    const float* source, const float* gamma, float* output) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ZG_ROWS * ZG_COLUMNS) output[i] = __fmul_rn(source[i], gamma[i % ZG_COLUMNS]);
}

extern "C" __global__ void zg_vllm_input_bf16_store(
    const float* source, unsigned short* output) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ZG_ROWS * ZG_COLUMNS) {
    unsigned short value;
    asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(value) : "f"(source[i]));
    output[i] = value;
  }
}
