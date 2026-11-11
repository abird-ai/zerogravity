// Causal BF16 attention, 16 queries x 64 keys, head width 256.
// CUDA online-softmax profile: FTZ FP32, BF16 unnormalized P, FP32 MMA PV.
// Original stream/allocations only; all working storage is block-local.
struct __align__(8) AttentionShape {
    unsigned long long rows, base, capacity, query_heads, kv_heads;
    float scale_log2;
    unsigned padding;
};
struct __align__(8) AttentionBatch4Shape {
    unsigned long long capacity, query_heads, kv_heads;
    unsigned long long base0, base1, base2, base3;
    float scale_log2;
    unsigned padding;
};
__device__ __forceinline__ unsigned pair(const unsigned short* p) {
    return static_cast<unsigned>(p[0]) | (static_cast<unsigned>(p[1]) << 16);
}
__device__ __forceinline__ unsigned short bf16(float x) {
    unsigned short y;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float add(float a, float b) {
    float y; asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y;
}
__device__ __forceinline__ float mul(float a, float b) {
    float y; asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y;
}
__device__ __forceinline__ float fma(float a, float b, float c) {
    float y; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(a), "f"(b), "f"(c)); return y;
}
__device__ __forceinline__ float maximum(float a, float b) {
    float y; asm("max.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y;
}
__device__ __forceinline__ float ex2(float a) {
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a)); return y;
}
__device__ __forceinline__ float reciprocal(float a) {
    float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a)); return y;
}
__device__ __forceinline__ void mma(float (&d)[4], unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned short checked(unsigned short x, unsigned* status) {
    if ((x & 0x7f80) == 0x7f80) atomicOr(status, 2u);
    return x;
}
extern "C" __global__ __launch_bounds__(128) void bf16_causal_attention(
    const unsigned short* query, const unsigned short* keys, const unsigned short* values,
    unsigned short* output, unsigned* status, AttentionShape s) {
    __shared__ unsigned short q[16 * 256], kv[64 * 256], p[16 * 64];
    __shared__ float scores[16 * 64], maxima[16], sums[64], rescale[16], inverse[16];
    const unsigned tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const unsigned r = lane / 4, c = (lane % 4) * 2;
    const unsigned long long first = static_cast<unsigned long long>(blockIdx.x) * 16;
    const unsigned long long head = blockIdx.y, kh = head / (s.query_heads / s.kv_heads);
    const unsigned long long end = s.base + s.rows;
    const float neg_inf = __int_as_float(0xff800000);
    for (unsigned i = tid; i < 16 * 256; i += 128) {
        const auto row = first + i / 256;
        q[i] = row < s.rows ? checked(query[(row * s.query_heads + head) * 256 + i % 256], status) : 0;
    }
    if (tid < 16) maxima[tid] = neg_inf;
    if (tid < 64) sums[tid] = 0;
    float acc[8][4] = {};
    const auto last_query = first + 16 < s.rows ? first + 16 : s.rows;
    // Descending key tiles match the CUDA causal online-softmax schedule.
    for (long long start = ((s.base + last_query - 1) / 64) * 64; start >= 0; start -= 64) {
        for (unsigned i = tid; i < 64 * 256; i += 128) {
            const auto row = static_cast<unsigned long long>(start) + i / 256;
            kv[i] = row < end ? checked(keys[(row * s.kv_heads + kh) * 256 + i % 256], status) : 0;
        }
        __syncthreads();
        float dot[2][4] = {};
        #pragma unroll
        for (unsigned k = 0; k < 256; k += 16) {
            const auto* a = q + r * 256 + k + c;
            #pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
                const auto* b = kv + (warp * 16 + j * 8 + r) * 256 + k + c;
                mma(dot[j], pair(a), pair(a + 8 * 256), pair(a + 8), pair(a + 8 * 256 + 8), pair(b), pair(b + 8));
            }
        }
        #pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned row = r + (i / 2) * 8, col = warp * 16 + j * 8 + c + i % 2;
                const bool valid = first + row < s.rows && start + col <= s.base + first + row;
                scores[row * 64 + col] = valid ? dot[j][i] : neg_inf;
                if (valid && !isfinite(dot[j][i])) atomicOr(status, 2u);
            }
        }
        __syncthreads();
        if (tid < 64) {
            const unsigned row = tid / 4, group = tid % 4;
            float x[16];
            float m = maxima[row];
            #pragma unroll
            for (unsigned j = 0; j < 16; ++j) {
                x[j] = scores[row * 64 + (j / 2) * 8 + group * 2 + j % 2];
                m = maximum(m, x[j]);
            }
            m = maximum(m, __shfl_xor_sync(0xffffffff, m, 2));
            m = maximum(m, __shfl_xor_sync(0xffffffff, m, 1));
            const float decay = m == neg_inf ? 1.0f : ex2(mul(add(maxima[row], -m), s.scale_log2));
            const float scaled_max = m == neg_inf ? 0.0f : mul(m, s.scale_log2);
            float sum = mul(sums[tid], decay);
            #pragma unroll
            for (unsigned j = 0; j < 16; ++j) {
                const float e = ex2(fma(x[j], s.scale_log2, -scaled_max));
                sum = add(sum, e);
                p[row * 64 + (j / 2) * 8 + group * 2 + j % 2] = bf16(e);
            }
            sums[tid] = sum;
            if (group == 0) { maxima[row] = m; rescale[row] = decay; }
        }
        __syncthreads();
        for (unsigned i = tid; i < 64 * 256; i += 128) {
            const auto row = static_cast<unsigned long long>(start) + i / 256;
            kv[i] = row < end ? checked(values[(row * s.kv_heads + kh) * 256 + i % 256], status) : 0;
        }
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) acc[j][i] = mul(acc[j][i], rescale[r + (i / 2) * 8]);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 64; k += 16) {
            const auto* a = p + r * 64 + k + c;
            #pragma unroll
            for (unsigned j = 0; j < 8; ++j) {
                const unsigned col = warp * 64 + j * 8 + r;
                const auto* b = kv + (k + c) * 256 + col;
                const unsigned b0 = static_cast<unsigned>(b[0]) | (static_cast<unsigned>(b[256]) << 16);
                const unsigned b1 = static_cast<unsigned>(b[8 * 256]) | (static_cast<unsigned>(b[9 * 256]) << 16);
                mma(acc[j], pair(a), pair(a + 8 * 64), pair(a + 8), pair(a + 8 * 64 + 8), b0, b1);
            }
        }
        __syncthreads();
    }
    if (tid < 64) {
        float sum = sums[tid];
        sum = add(sum, __shfl_xor_sync(0xffffffff, sum, 2));
        sum = add(sum, __shfl_xor_sync(0xffffffff, sum, 1));
        if (tid % 4 == 0) inverse[tid / 4] = sum == 0.0f ? 1.0f : reciprocal(sum);
    }
    __syncthreads();
    #pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned row = r + (i / 2) * 8, col = warp * 64 + j * 8 + c + i % 2;
            if (first + row < s.rows) {
                const float result = mul(acc[j][i], inverse[row]);
                const auto rounded = bf16(result);
                if (!isfinite(result) || (rounded & 0x7f80) == 0x7f80) atomicOr(status, 2u);
                output[((first + row) * s.query_heads + head) * 256 + col] = rounded;
            }
        }
    }
}

// Four independent aligned cached-decode rows. Arithmetic is the same one-row
// schedule above; blockIdx.z only selects the row's private K/V owner.
extern "C" __global__ __launch_bounds__(128) void bf16_causal_attention_batch4(
    const unsigned short* query,
    const unsigned short* keys0, const unsigned short* keys1,
    const unsigned short* keys2, const unsigned short* keys3,
    const unsigned short* values0, const unsigned short* values1,
    const unsigned short* values2, const unsigned short* values3,
    unsigned short* output, unsigned* status, AttentionBatch4Shape s) {
    __shared__ unsigned short q[16 * 256], kv[64 * 256], p[16 * 64];
    __shared__ float scores[16 * 64], maxima[16], sums[64], rescale[16], inverse[16];
    const unsigned tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const unsigned r = lane / 4, c = (lane % 4) * 2;
    const unsigned long long first = static_cast<unsigned long long>(blockIdx.x) * 16;
    const unsigned long long head = blockIdx.y, kh = head / (s.query_heads / s.kv_heads);
    const unsigned long long batch = blockIdx.z;
    const unsigned long long base = batch == 0 ? s.base0 : batch == 1 ? s.base1 : batch == 2 ? s.base2 : s.base3;
    const unsigned short* keys = batch == 0 ? keys0 : batch == 1 ? keys1 : batch == 2 ? keys2 : keys3;
    const unsigned short* values = batch == 0 ? values0 : batch == 1 ? values1 : batch == 2 ? values2 : values3;
    const unsigned long long end = base + 1;
    const float neg_inf = __int_as_float(0xff800000);
    for (unsigned i = tid; i < 16 * 256; i += 128) {
        const auto row = first + i / 256;
        q[i] = row < 1 ? checked(query[(batch * s.query_heads + head) * 256 + i % 256], status) : 0;
    }
    if (tid < 16) maxima[tid] = neg_inf;
    if (tid < 64) sums[tid] = 0;
    float acc[8][4] = {};
    const auto last_query = first + 16 < 1 ? first + 16 : 1;
    for (long long start = ((base + last_query - 1) / 64) * 64; start >= 0; start -= 64) {
        for (unsigned i = tid; i < 64 * 256; i += 128) {
            const auto row = static_cast<unsigned long long>(start) + i / 256;
            kv[i] = row < end ? checked(keys[(row * s.kv_heads + kh) * 256 + i % 256], status) : 0;
        }
        __syncthreads();
        float dot[2][4] = {};
        #pragma unroll
        for (unsigned k = 0; k < 256; k += 16) {
            const auto* a = q + r * 256 + k + c;
            #pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
                const auto* b = kv + (warp * 16 + j * 8 + r) * 256 + k + c;
                mma(dot[j], pair(a), pair(a + 8 * 256), pair(a + 8), pair(a + 8 * 256 + 8), pair(b), pair(b + 8));
            }
        }
        #pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned row = r + (i / 2) * 8, col = warp * 16 + j * 8 + c + i % 2;
                const bool valid = first + row < 1 && start + col <= base + first + row;
                scores[row * 64 + col] = valid ? dot[j][i] : neg_inf;
                if (valid && !isfinite(dot[j][i])) atomicOr(status, 2u);
            }
        }
        __syncthreads();
        if (tid < 64) {
            const unsigned row = tid / 4, group = tid % 4;
            float x[16];
            float m = maxima[row];
            #pragma unroll
            for (unsigned j = 0; j < 16; ++j) {
                x[j] = scores[row * 64 + (j / 2) * 8 + group * 2 + j % 2];
                m = maximum(m, x[j]);
            }
            m = maximum(m, __shfl_xor_sync(0xffffffff, m, 2));
            m = maximum(m, __shfl_xor_sync(0xffffffff, m, 1));
            const float decay = m == neg_inf ? 1.0f : ex2(mul(add(maxima[row], -m), s.scale_log2));
            const float scaled_max = m == neg_inf ? 0.0f : mul(m, s.scale_log2);
            float sum = mul(sums[tid], decay);
            #pragma unroll
            for (unsigned j = 0; j < 16; ++j) {
                const float e = ex2(fma(x[j], s.scale_log2, -scaled_max));
                sum = add(sum, e);
                p[row * 64 + (j / 2) * 8 + group * 2 + j % 2] = bf16(e);
            }
            sums[tid] = sum;
            if (group == 0) { maxima[row] = m; rescale[row] = decay; }
        }
        __syncthreads();
        for (unsigned i = tid; i < 64 * 256; i += 128) {
            const auto row = static_cast<unsigned long long>(start) + i / 256;
            kv[i] = row < end ? checked(values[(row * s.kv_heads + kh) * 256 + i % 256], status) : 0;
        }
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) acc[j][i] = mul(acc[j][i], rescale[r + (i / 2) * 8]);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 64; k += 16) {
            const auto* a = p + r * 64 + k + c;
            #pragma unroll
            for (unsigned j = 0; j < 8; ++j) {
                const unsigned col = warp * 64 + j * 8 + r;
                const auto* b = kv + (k + c) * 256 + col;
                const unsigned b0 = static_cast<unsigned>(b[0]) | (static_cast<unsigned>(b[256]) << 16);
                const unsigned b1 = static_cast<unsigned>(b[8 * 256]) | (static_cast<unsigned>(b[9 * 256]) << 16);
                mma(acc[j], pair(a), pair(a + 8 * 64), pair(a + 8), pair(a + 8 * 64 + 8), b0, b1);
            }
        }
        __syncthreads();
    }
    if (tid < 64) {
        float sum = sums[tid];
        sum = add(sum, __shfl_xor_sync(0xffffffff, sum, 2));
        sum = add(sum, __shfl_xor_sync(0xffffffff, sum, 1));
        if (tid % 4 == 0) inverse[tid / 4] = sum == 0.0f ? 1.0f : reciprocal(sum);
    }
    __syncthreads();
    #pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned row = r + (i / 2) * 8, col = warp * 64 + j * 8 + c + i % 2;
            if (first + row < 1) {
                const float result = mul(acc[j][i], inverse[row]);
                const auto rounded = bf16(result);
                if (!isfinite(result) || (rounded & 0x7f80) == 0x7f80) atomicOr(status, 2u);
                output[(batch * s.query_heads + head) * 256 + col] = rounded;
            }
        }
    }
}
