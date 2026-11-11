// BF16 W[columns,inner] * X[rows,inner]^T -> Y[rows,columns].
// Explicit reduction arithmetic; no global scratch or hidden allocations.
struct __align__(8) GemmShape {
    unsigned long long columns, rows, inner, reduction_terms, lane_shift, output_stride;
};

__device__ __forceinline__ unsigned pack(const unsigned short* p) {
    unsigned value;
    unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
    return value;
}
__device__ __forceinline__ unsigned short round_bf16(float x) {
    unsigned short out;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(out) : "f"(x));
    return out;
}
__device__ __forceinline__ void store_bf16(
    unsigned short* output, unsigned long long index, float value, unsigned* status) {
    const unsigned short rounded = round_bf16(value);
    if (status != nullptr && (rounded & 0x7f80u) == 0x7f80u) atomicOr(status, 2u);
    output[index] = rounded;
}
__device__ __forceinline__ void copy16(unsigned short* dst, const unsigned short* src, unsigned bytes) {
    unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(address), "l"(src), "r"(bytes));
}

template<int BN>
__device__ __forceinline__ void stage(
    unsigned short* sw, unsigned short* sx,
    const unsigned short* w, const unsigned short* x,
    GemmShape s, unsigned long long base_m, unsigned long long base_n,
    unsigned long long base_k, bool aligned) {
    constexpr int BK = 32, STRIDE = BK + 8;
    for (unsigned vector = threadIdx.x; vector < (64 + BN) * BK / 8; vector += 128) {
        const bool weight = vector < 64 * BK / 8;
        const unsigned local = weight ? vector : vector - 64 * BK / 8;
        const unsigned row = local / (BK / 8), col = (local % (BK / 8)) * 8;
        const auto global_row = (weight ? base_m : base_n) + row;
        const auto global_col = base_k + col;
        const auto limit = weight ? s.columns : s.rows;
        auto* target = (weight ? sw : sx) + row * STRIDE + col;
        const auto* source = weight ? w : x;
        const unsigned valid = global_row < limit && global_col < s.inner
            ? static_cast<unsigned>((s.inner - global_col) < 8 ? (s.inner - global_col) : 8) : 0;
        if (valid) source += global_row * s.inner + global_col;
        if (aligned) {
            copy16(target, source, valid * 2);
        } else {
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) target[i] = i < valid ? source[i] : 0;
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int BN>
__device__ __forceinline__ void stage64(
    unsigned short* sw, unsigned short* sx,
    const unsigned short* w, const unsigned short* x,
    GemmShape s, unsigned long long base_m, unsigned long long base_n,
    unsigned long long base_k, bool aligned) {
    constexpr int BK = 64, STRIDE = BK + 8;
    for (unsigned vector = threadIdx.x; vector < (64 + BN) * BK / 8; vector += 128) {
        const bool weight = vector < 64 * BK / 8;
        const unsigned local = weight ? vector : vector - 64 * BK / 8;
        const unsigned row = local / (BK / 8), col = (local % (BK / 8)) * 8;
        const auto global_row = (weight ? base_m : base_n) + row;
        const auto global_col = base_k + col;
        const auto limit = weight ? s.columns : s.rows;
        auto* target = (weight ? sw : sx) + row * STRIDE + col;
        const auto* source = weight ? w : x;
        const unsigned valid = global_row < limit && global_col < s.inner
            ? static_cast<unsigned>((s.inner - global_col) < 8 ? (s.inner - global_col) : 8) : 0;
        if (valid) source += global_row * s.inner + global_col;
        if (aligned) {
            copy16(target, source, valid * 2);
        } else {
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) target[i] = i < valid ? source[i] : 0;
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ float widen_bf16(unsigned short value) {
    float result;
    unsigned bits = static_cast<unsigned>(value) << 16;
    asm("mov.b32 %0, %1;" : "=f"(result) : "r"(bits));
    return result;
}
__device__ __forceinline__ void mma_fragment(
    float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// MODE: 0 = ordered FP32, 1 = two interleaved FP32 sums,
// 2 = BF16-rounded segment sums accumulated in FP32.
template<int BN, int MODE>
__device__ __forceinline__ void gemm_body(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s, unsigned short* shared) {
    constexpr int STRIDE = 40, STAGE = (64 + BN) * STRIDE, FRAGMENTS = BN / 8;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned row = lane / 4, pair = (lane % 4) * 2;
    const unsigned long long base_m = static_cast<unsigned long long>(blockIdx.x) * 64;
    const unsigned long long base_n = static_cast<unsigned long long>(blockIdx.y) * BN;
    const bool aligned = ((reinterpret_cast<unsigned long long>(w) |
                           reinterpret_cast<unsigned long long>(x)) & 15) == 0 && s.inner % 8 == 0;
    float d[FRAGMENTS][4] = {}, second[FRAGMENTS][4] = {};
    stage<BN>(shared, shared + 64 * STRIDE, w, x, s, base_m, base_n, 0, aligned);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    unsigned current = 0;
    unsigned long long remaining = s.reduction_terms;
    bool second_group = false;
    for (unsigned long long k = 0; k < s.inner; k += 32) {
        unsigned short* sw = shared + current * STAGE;
        unsigned short* sx = sw + 64 * STRIDE;
        if (k + 32 < s.inner) {
            unsigned short* next = shared + (current ^ 1) * STAGE;
            stage<BN>(next, next + 64 * STRIDE, w, x, s, base_m, base_n, k + 32, aligned);
        }
        #pragma unroll
        for (unsigned kk = 0; kk < 32; kk += 16) {
            const auto* a = sw + (warp * 16 + row) * STRIDE + kk + pair;
            const unsigned a0 = pack(a), a1 = pack(a + 8 * STRIDE);
            const unsigned a2 = pack(a + 8), a3 = pack(a + 8 * STRIDE + 8);
            #pragma unroll
            for (unsigned j = 0; j < FRAGMENTS; ++j) {
                const auto* b = sx + (j * 8 + row) * STRIDE + kk + pair;
                const unsigned b0 = pack(b), b1 = pack(b + 8);
                if (MODE == 1 && second_group) {
                    mma_fragment(second[j], a0, a1, a2, a3, b0, b1);
                } else {
                    mma_fragment(d[j], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        if (MODE != 0) remaining -= 32;
        if (MODE == 2 && (remaining == 0 || k + 32 >= s.inner)) {
            #pragma unroll
            for (unsigned j = 0; j < FRAGMENTS; ++j) {
                #pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    second[j][i] += widen_bf16(round_bf16(d[j][i]));
                    d[j][i] = 0;
                }
            }
        }
        if (MODE != 0 && remaining == 0) {
            remaining = s.reduction_terms;
            second_group = !second_group;
        }
        if (k + 32 < s.inner) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
        current ^= 1;
    }
    #pragma unroll
    for (unsigned j = 0; j < FRAGMENTS; ++j) {
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const auto m = base_m + warp * 16 + row + (i / 2) * 8;
            const auto n = base_n + j * 8 + pair + i % 2;
            const float result = MODE == 2 ? second[j][i]
                : MODE == 1 ? d[j][i] + second[j][i] : d[j][i];
            if (m < s.columns && n < s.rows) store_bf16(y, n * s.output_stride + m, result, status);
        }
    }
}

template<int BN>
__device__ __forceinline__ void gemm_ordered_k64_body(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s, unsigned short* shared) {
    constexpr int STRIDE = 72, STAGE = (64 + BN) * STRIDE, FRAGMENTS = BN / 8;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned row = lane / 4, pair = (lane % 4) * 2;
    const unsigned long long base_m = static_cast<unsigned long long>(blockIdx.x) * 64;
    const unsigned long long base_n = static_cast<unsigned long long>(blockIdx.y) * BN;
    const bool aligned = ((reinterpret_cast<unsigned long long>(w) |
                           reinterpret_cast<unsigned long long>(x)) & 15) == 0 && s.inner % 8 == 0;
    float d[FRAGMENTS][4] = {};
    stage64<BN>(shared, shared + 64 * STRIDE, w, x, s, base_m, base_n, 0, aligned);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    unsigned current = 0;
    for (unsigned long long k = 0; k < s.inner; k += 64) {
        unsigned short* sw = shared + current * STAGE;
        unsigned short* sx = sw + 64 * STRIDE;
        if (k + 64 < s.inner) {
            unsigned short* next = shared + (current ^ 1) * STAGE;
            stage64<BN>(next, next + 64 * STRIDE, w, x, s, base_m, base_n, k + 64, aligned);
        }
        #pragma unroll
        for (unsigned kk = 0; kk < 64; kk += 16) {
            const auto* a = sw + (warp * 16 + row) * STRIDE + kk + pair;
            const unsigned a0 = pack(a), a1 = pack(a + 8 * STRIDE);
            const unsigned a2 = pack(a + 8), a3 = pack(a + 8 * STRIDE + 8);
            #pragma unroll
            for (unsigned j = 0; j < FRAGMENTS; ++j) {
                const auto* b = sx + (j * 8 + row) * STRIDE + kk + pair;
                const unsigned b0 = pack(b), b1 = pack(b + 8);
                mma_fragment(d[j], a0, a1, a2, a3, b0, b1);
            }
        }
        if (k + 64 < s.inner) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
        current ^= 1;
    }
    #pragma unroll
    for (unsigned j = 0; j < FRAGMENTS; ++j) {
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const auto m = base_m + warp * 16 + row + (i / 2) * 8;
            const auto n = base_n + j * 8 + pair + i % 2;
            if (m < s.columns && n < s.rows) {
                store_bf16(y, n * s.output_stride + m, d[j][i], status);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128) void bf16_gemm_ordered_narrow_k64(
    const unsigned short* w, const unsigned short* x, unsigned short* y, GemmShape s) {
    __shared__ __align__(16) unsigned short shared[2 * (64 + 8) * 72];
    gemm_ordered_k64_body<8>(w, x, y, nullptr, s, shared);
}
extern "C" __global__ __launch_bounds__(128) void bf16_gemm_ordered_narrow_k64_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y, unsigned* status, GemmShape s) {
    __shared__ __align__(16) unsigned short shared[2 * (64 + 8) * 72];
    gemm_ordered_k64_body<8>(w, x, y, status, s, shared);
}

#define GEMM_ENTRY(NAME, BN, MODE) \
extern "C" __global__ __launch_bounds__(128) void NAME( \
    const unsigned short* w, const unsigned short* x, unsigned short* y, GemmShape s) { \
    __shared__ __align__(16) unsigned short shared[2 * (64 + BN) * 40]; \
    gemm_body<BN, MODE>(w, x, y, nullptr, s, shared); \
} \
extern "C" __global__ __launch_bounds__(128) void NAME##_certified( \
    const unsigned short* w, const unsigned short* x, unsigned short* y, unsigned* status, GemmShape s) { \
    __shared__ __align__(16) unsigned short shared[2 * (64 + BN) * 40]; \
    gemm_body<BN, MODE>(w, x, y, status, s, shared); \
}
GEMM_ENTRY(bf16_gemm_narrow, 8, 0)
GEMM_ENTRY(bf16_gemm_wide, 32, 0)
GEMM_ENTRY(bf16_gemm_ordered_rows64, 64, 0)
GEMM_ENTRY(bf16_gemm_interleaved_narrow, 8, 1)
GEMM_ENTRY(bf16_gemm_interleaved_wide, 32, 1)
GEMM_ENTRY(bf16_gemm_segmented_narrow, 8, 2)
GEMM_ENTRY(bf16_gemm_segmented_wide, 32, 2)

__device__ __forceinline__ float add_product(float sum, float a, float b) {
    asm("fma.rn.f32 %0, %1, %2, %0;" : "+f"(sum) : "f"(a), "f"(b));
    return sum;
}
__device__ __forceinline__ float lane_value(float value, unsigned lane, unsigned control) {
    float result;
    asm("shfl.sync.idx.b32 %0, %1, %2, %3, 0xffffffff;"
        : "=f"(result) : "f"(value), "r"(lane), "r"(control));
    return result;
}
__device__ __forceinline__ float lane_down(float value, unsigned delta, unsigned control) {
    float result;
    asm("shfl.sync.down.b32 %0, %1, %2, %3, 0xffffffff;"
        : "=f"(result) : "f"(value), "r"(delta), "r"(control));
    return result;
}

// SIMT alternatives retain explicit FMA grouping rather than substituting MMA.
// Each power-of-two lane group owns one output; all lanes reach every shuffle.
template<unsigned LANES, unsigned TERMS>
__device__ __forceinline__ void lane_strided_fixed(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    constexpr unsigned CONTROL = ((32 - LANES) << 8) | 31;
    constexpr unsigned ITEMS = TERMS / LANES;
    const unsigned lane = threadIdx.x & (LANES - 1);
    const unsigned long long col = (unsigned long long)blockIdx.x * (128 / LANES)
        + (threadIdx.x / LANES);
    const unsigned long long row = blockIdx.y;
    float total = 0;
    for (unsigned long long start = 0; start < s.inner; start += TERMS) {
        float partial = 0;
        #pragma unroll
        for (unsigned item = 0; item < ITEMS; ++item) {
            const unsigned long long k = start + lane + item * LANES;
            if (col < s.columns && k < s.inner) partial = add_product(partial,
                widen_bf16(w[col * s.inner + k]), widen_bf16(x[row * s.inner + k]));
        }
        total += partial;
    }
    float result = lane_value(total, 0, CONTROL);
    #pragma unroll
    for (unsigned i = 1; i < LANES; ++i) result += lane_value(total, i, CONTROL);
    if (lane == 0 && col < s.columns) store_bf16(y, row * s.output_stride + col, result, status);
}

__device__ __forceinline__ void lane_strided_batch4_fixed(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    constexpr unsigned LANES = 16, TERMS = 512, ITEMS = TERMS / LANES;
    constexpr unsigned CONTROL = ((32 - LANES) << 8) | 31;
    const unsigned lane = threadIdx.x & (LANES - 1);
    const unsigned long long col = (unsigned long long)blockIdx.x * (128 / LANES)
        + (threadIdx.x / LANES);
    float total0 = 0, total1 = 0, total2 = 0, total3 = 0;
    for (unsigned long long start = 0; start < s.inner; start += TERMS) {
        float partial0 = 0, partial1 = 0, partial2 = 0, partial3 = 0;
        #pragma unroll
        for (unsigned item = 0; item < ITEMS; ++item) {
            const unsigned long long k = start + lane + item * LANES;
            if (col < s.columns && k < s.inner) {
                const float weight = widen_bf16(w[col * s.inner + k]);
                partial0 = add_product(partial0, weight, widen_bf16(x[k]));
                partial1 = add_product(partial1, weight, widen_bf16(x[s.inner + k]));
                partial2 = add_product(partial2, weight, widen_bf16(x[2 * s.inner + k]));
                partial3 = add_product(partial3, weight, widen_bf16(x[3 * s.inner + k]));
            }
        }
        total0 += partial0;
        total1 += partial1;
        total2 += partial2;
        total3 += partial3;
    }
    float result0 = lane_value(total0, 0, CONTROL);
    float result1 = lane_value(total1, 0, CONTROL);
    float result2 = lane_value(total2, 0, CONTROL);
    float result3 = lane_value(total3, 0, CONTROL);
    #pragma unroll
    for (unsigned i = 1; i < LANES; ++i) {
        result0 += lane_value(total0, i, CONTROL);
        result1 += lane_value(total1, i, CONTROL);
        result2 += lane_value(total2, i, CONTROL);
        result3 += lane_value(total3, i, CONTROL);
    }
    if (lane == 0 && col < s.columns) {
        store_bf16(y, col, result0, status);
        store_bf16(y, s.output_stride + col, result1, status);
        store_bf16(y, 2 * s.output_stride + col, result2, status);
        store_bf16(y, 3 * s.output_stride + col, result3, status);
    }
}

__device__ __forceinline__ void lane_strided_batch8_fixed(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    constexpr unsigned LANES = 16, TERMS = 512, ITEMS = TERMS / LANES;
    constexpr unsigned CONTROL = ((32 - LANES) << 8) | 31;
    const unsigned lane = threadIdx.x & (LANES - 1);
    const unsigned long long col = (unsigned long long)blockIdx.x * (128 / LANES)
        + (threadIdx.x / LANES);
    float total[8] = {};
    for (unsigned long long start = 0; start < s.inner; start += TERMS) {
        float partial[8] = {};
        #pragma unroll
        for (unsigned item = 0; item < ITEMS; ++item) {
            const unsigned long long k = start + lane + item * LANES;
            if (col < s.columns && k < s.inner) {
                const float weight = widen_bf16(w[col * s.inner + k]);
                #pragma unroll
                for (unsigned row = 0; row < 8; ++row) {
                    partial[row] = add_product(
                        partial[row], weight, widen_bf16(x[row * s.inner + k]));
                }
            }
        }
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) total[row] += partial[row];
    }
    float result[8];
    #pragma unroll
    for (unsigned row = 0; row < 8; ++row) result[row] = lane_value(total[row], 0, CONTROL);
    #pragma unroll
    for (unsigned i = 1; i < LANES; ++i) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) result[row] += lane_value(total[row], i, CONTROL);
    }
    if (lane == 0 && col < s.columns) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            store_bf16(y, row * s.output_stride + col, result[row], status);
        }
    }
}

__device__ __forceinline__ void lane_strided_body(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    if (s.lane_shift == 4 && s.reduction_terms == 512) {
        lane_strided_fixed<16, 512>(w, x, y, status, s);
        return;
    }
    if (s.lane_shift == 4 && s.reduction_terms == 256) {
        lane_strided_fixed<16, 256>(w, x, y, status, s);
        return;
    }
    const unsigned lanes = 1u << s.lane_shift;
    const unsigned lane = threadIdx.x & (lanes - 1);
    const unsigned control = ((32 - lanes) << 8) | 31;
    const unsigned long long col = (unsigned long long)blockIdx.x * (128 >> s.lane_shift)
        + (threadIdx.x >> s.lane_shift);
    const unsigned long long row = blockIdx.y;
    float total = 0;
    for (unsigned long long start = 0; start < s.inner; start += s.reduction_terms) {
        float partial = 0;
        const unsigned long long end = s.inner < start + s.reduction_terms
            ? s.inner : start + s.reduction_terms;
        for (unsigned long long k = start + lane; k < end; k += lanes) {
            if (col < s.columns) partial = add_product(partial,
                widen_bf16(w[col * s.inner + k]), widen_bf16(x[row * s.inner + k]));
        }
        total += partial;
    }
    float result = lane_value(total, 0, control);
    for (unsigned i = 1; i < lanes; ++i) result += lane_value(total, i, control);
    if (lane == 0 && col < s.columns) store_bf16(y, row * s.output_stride + col, result, status);
}
extern "C" __global__ __launch_bounds__(128) void bf16_gemm_lane_strided(
    const unsigned short* w, const unsigned short* x, unsigned short* y, GemmShape s) {
    lane_strided_body(w, x, y, nullptr, s);
}
extern "C" __global__ __launch_bounds__(128) void bf16_gemm_lane_strided_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y, unsigned* status, GemmShape s) {
    lane_strided_body(w, x, y, status, s);
}
extern "C" __global__ __launch_bounds__(128) void bf16_gemm_lane_strided_batch4(
    const unsigned short* w, const unsigned short* x, unsigned short* y, GemmShape s) {
    lane_strided_batch4_fixed(w, x, y, nullptr, s);
}
extern "C" __global__ __launch_bounds__(128) void bf16_gemm_lane_strided_batch4_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y, unsigned* status, GemmShape s) {
    lane_strided_batch4_fixed(w, x, y, status, s);
}
extern "C" __global__ __launch_bounds__(128) void bf16_gemm_lane_strided_batch8(
    const unsigned short* w, const unsigned short* x, unsigned short* y, GemmShape s) {
    lane_strided_batch8_fixed(w, x, y, nullptr, s);
}
extern "C" __global__ __launch_bounds__(128) void bf16_gemm_lane_strided_batch8_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y, unsigned* status, GemmShape s) {
    lane_strided_batch8_fixed(w, x, y, status, s);
}

__device__ __forceinline__ void lane_blocked_body(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    const unsigned lanes = 1u << s.lane_shift;
    const unsigned lane = threadIdx.x & (lanes - 1);
    const unsigned control = ((32 - lanes) << 8) | 31;
    const unsigned long long col = (unsigned long long)blockIdx.x * (128 >> s.lane_shift)
        + (threadIdx.x >> s.lane_shift);
    const unsigned long long row = blockIdx.y;
    float total = 0;
    const unsigned long long step = s.reduction_terms * lanes;
    for (unsigned long long start = 0; start < s.inner; start += step) {
        const unsigned long long first = start + lane * s.reduction_terms;
        const unsigned long long end = s.inner < first + s.reduction_terms
            ? s.inner : first + s.reduction_terms;
        for (unsigned long long k = first; k < end; ++k) {
            if (col < s.columns) total = add_product(total,
                widen_bf16(w[col * s.inner + k]), widen_bf16(x[row * s.inner + k]));
        }
    }
    for (unsigned delta = lanes >> 1; delta; delta >>= 1) {
        const float other = lane_down(total, delta, control);
        if (lane < delta) total += other;
    }
    if (lane == 0 && col < s.columns) store_bf16(y, row * s.output_stride + col, total, status);
}
extern "C" __global__ __launch_bounds__(128) void bf16_gemm_lane_blocked(
    const unsigned short* w, const unsigned short* x, unsigned short* y, GemmShape s) {
    lane_blocked_body(w, x, y, nullptr, s);
}
extern "C" __global__ __launch_bounds__(128) void bf16_gemm_lane_blocked_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y, unsigned* status, GemmShape s) {
    lane_blocked_body(w, x, y, status, s);
}
