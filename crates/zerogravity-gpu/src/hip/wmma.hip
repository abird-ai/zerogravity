// RDNA3: A/B fragments replicated in both 16-lane halves of a wave.
// C is distributed by column (lane % 16) and alternating rows (lane / 16).
#include <hip/hip_runtime.h>
typedef short bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

extern "C" __global__ __launch_bounds__(32)
void zg_bf16_wmma(const short* weights, const short* input,
                 float* output, unsigned* status, const unsigned long long* info) {
    const unsigned m = info[0], n = info[1], k = info[2], stride = info[3];
    const unsigned lane = threadIdx.x;
    const unsigned i = blockIdx.y * 16 + (lane & 15);
    const unsigned j = blockIdx.x * 16 + (lane & 15);
    f32x8 acc = {};
    unsigned invalid = 0;
    for (unsigned inner = 0; inner < k; inner += 16) {
        bf16x16 a = {}, b = {};
        if (i < m) a = *reinterpret_cast<const bf16x16*>(input + i * k + inner);
        if (j < n) b = *reinterpret_cast<const bf16x16*>(weights + j * k + inner);
        // Inspect packed source exponent bits, not native FTZ arithmetic.
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            invalid |= ((a[t] & 0x7f80) == 0x7f80) | ((b[t] & 0x7f80) == 0x7f80);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    }
    #pragma unroll
    for (unsigned t = 0; t < 8; ++t) {
        const unsigned row = blockIdx.y * 16 + 2 * t + lane / 16;
        if (row < m && j < n) {
            const float value = acc[t];
            unsigned bits = __float_as_uint(value);
            invalid |= (bits & 0x7f800000u) == 0x7f800000u;
            bits = (bits + (0x7fffu + ((bits >> 16) & 1u))) & 0xffff0000u;
            invalid |= (bits & 0x7f800000u) == 0x7f800000u;
            output[row * stride + j] = __uint_as_float(bits);
        }
    }
    if (invalid) atomicOr(status, 2u);
}
