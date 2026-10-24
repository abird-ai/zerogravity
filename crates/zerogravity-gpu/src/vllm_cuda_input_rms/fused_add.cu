// Original BF16 operands widen exactly; keep the FP32 sum independently of
// the BF16 residual store. Existing RMS square/normalize consume that sum.
extern "C" __global__ void zg_vllm_post_rms_add(
    const unsigned short* attention, const unsigned short* prior_residual,
    float* sum_f32, unsigned short* residual_bf16) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ZG_ROWS * ZG_COLUMNS) {
    const float left = __uint_as_float((unsigned int)attention[i] << 16);
    const float right = __uint_as_float((unsigned int)prior_residual[i] << 16);
    const float sum = __fadd_rn(left, right);
    sum_f32[i] = sum;
    unsigned short rounded;
    asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(rounded) : "f"(sum));
    residual_bf16[i] = rounded;
  }
}
