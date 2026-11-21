

template<unsigned ROWS> struct ZgCandidateInnerMajorLoad;

template<> struct ZgCandidateInnerMajorLoad<2> {
    __device__ __forceinline__ static void load(
        const unsigned short* x, unsigned long long k, unsigned short (&values)[2]) {
        const unsigned packed = *reinterpret_cast<const unsigned*>(x + k * 2);
        values[0] = static_cast<unsigned short>(packed & 0xffffu);
        values[1] = static_cast<unsigned short>(packed >> 16);
    }
};
template<> struct ZgCandidateInnerMajorLoad<4> {
    __device__ __forceinline__ static void load(
        const unsigned short* x, unsigned long long k, unsigned short (&values)[4]) {
        const uint2 packed = *reinterpret_cast<const uint2*>(x + k * 4);
        values[0] = static_cast<unsigned short>(packed.x & 0xffffu);
        values[1] = static_cast<unsigned short>(packed.x >> 16);
        values[2] = static_cast<unsigned short>(packed.y & 0xffffu);
        values[3] = static_cast<unsigned short>(packed.y >> 16);
    }
};
template<> struct ZgCandidateInnerMajorLoad<8> {
    __device__ __forceinline__ static void load(
        const unsigned short* x, unsigned long long k, unsigned short (&values)[8]) {
        const uint4 packed = *reinterpret_cast<const uint4*>(x + k * 8);
        const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            const unsigned word = words[row >> 1];
            values[row] = static_cast<unsigned short>(
                row & 1 ? word >> 16 : word & 0xffffu);
        }
    }
};

template<unsigned ROWS>
__device__ __forceinline__ void zg_candidate_lane_blocked_inner_major(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    constexpr unsigned LANES = 32, TERMS = 32, CONTROL = 31;
    const unsigned lane = threadIdx.x & (LANES - 1);
    const unsigned long long col =
        (unsigned long long)blockIdx.x * (128 / LANES) + (threadIdx.x / LANES);
    float total[ROWS] = {};
    constexpr unsigned long long STEP = LANES * TERMS;
    for (unsigned long long start = 0; start < s.inner; start += STEP) {
        const unsigned long long first = start + lane * TERMS;
        const unsigned long long end =
            s.inner < first + TERMS ? s.inner : first + TERMS;
        for (unsigned long long k = first; k < end; ++k) {
            if (col < s.columns) {
                const float weight = widen_bf16(w[col * s.inner + k]);
                unsigned short values[ROWS];
                ZgCandidateInnerMajorLoad<ROWS>::load(x, k, values);
                #pragma unroll
                for (unsigned row = 0; row < ROWS; ++row) {
                    total[row] =
                        add_product(total[row], weight, widen_bf16(values[row]));
                }
            }
        }
    }
    for (unsigned delta = LANES >> 1; delta; delta >>= 1) {
        #pragma unroll
        for (unsigned row = 0; row < ROWS; ++row) {
            const float other = lane_down(total[row], delta, CONTROL);
            if (lane < delta) total[row] += other;
        }
    }
    if (lane == 0 && col < s.columns) {
        #pragma unroll
        for (unsigned row = 0; row < ROWS; ++row) {
            store_bf16(y, row * s.output_stride + col, total[row], status);
        }
    }
}

#define ZG_CANDIDATE_LANE_BLOCKED_INNER_ENTRY(ROWS) \
extern "C" __global__ __launch_bounds__(128) \
void zg_candidate_lane_blocked_inner_major_batch##ROWS##_certified( \
    const unsigned short* w, const unsigned short* x, unsigned short* y, \
    unsigned* status, GemmShape s) { \
    zg_candidate_lane_blocked_inner_major<ROWS>(w, x, y, status, s); \
}
ZG_CANDIDATE_LANE_BLOCKED_INNER_ENTRY(2)
ZG_CANDIDATE_LANE_BLOCKED_INNER_ENTRY(4)
ZG_CANDIDATE_LANE_BLOCKED_INNER_ENTRY(8)
#undef ZG_CANDIDATE_LANE_BLOCKED_INNER_ENTRY

// Historical shared-weight LaneBlocked experiment generalized the accepted
// terms32 row-sharing arithmetic to terms64. It remains dormant: exact rows
// share each weight load, while every row preserves its own contiguous
// 64-term accumulation and the same 32-lane halving reduction tree.
template<unsigned ROWS, unsigned TERMS>
__device__ __forceinline__ void zg_candidate_lane_blocked_shared_rows(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    constexpr unsigned LANES = 32, CONTROL = 31;
    static_assert(TERMS == 64, "retained LaneBlocked terms64 candidate");
    const unsigned lane = threadIdx.x & (LANES - 1);
    const unsigned long long col =
        (unsigned long long)blockIdx.x * (128 / LANES) + (threadIdx.x / LANES);
    float total[ROWS] = {};
    constexpr unsigned long long STEP = LANES * TERMS;
    for (unsigned long long start = 0; start < s.inner; start += STEP) {
        const unsigned long long first = start + lane * TERMS;
        const unsigned long long end =
            s.inner < first + TERMS ? s.inner : first + TERMS;
        for (unsigned long long k = first; k < end; ++k) {
            if (col < s.columns) {
                const float weight = widen_bf16(w[col * s.inner + k]);
                #pragma unroll
                for (unsigned row = 0; row < ROWS; ++row) {
                    total[row] = add_product(
                        total[row], weight, widen_bf16(x[row * s.inner + k]));
                }
            }
        }
    }
    for (unsigned delta = LANES >> 1; delta; delta >>= 1) {
        #pragma unroll
        for (unsigned row = 0; row < ROWS; ++row) {
            const float other = lane_down(total[row], delta, CONTROL);
            if (lane < delta) total[row] += other;
        }
    }
    if (lane == 0 && col < s.columns) {
        #pragma unroll
        for (unsigned row = 0; row < ROWS; ++row) {
            store_bf16(y, row * s.output_stride + col, total[row], status);
        }
    }
}

#define ZG_CANDIDATE_LANE_BLOCKED_T64_ENTRY(ROWS) \
extern "C" __global__ __launch_bounds__(128) \
void zg_candidate_lane_blocked_batch##ROWS##_terms64_certified( \
    const unsigned short* w, const unsigned short* x, unsigned short* y, \
    unsigned* status, GemmShape s) { \
    zg_candidate_lane_blocked_shared_rows<ROWS, 64>(w, x, y, status, s); \
}
ZG_CANDIDATE_LANE_BLOCKED_T64_ENTRY(2)
ZG_CANDIDATE_LANE_BLOCKED_T64_ENTRY(4)
ZG_CANDIDATE_LANE_BLOCKED_T64_ENTRY(8)
#undef ZG_CANDIDATE_LANE_BLOCKED_T64_ENTRY

template<unsigned THREADS>
__device__ __forceinline__ void zg_candidate_lane_strided_b8_inner_major(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    constexpr unsigned LANES = 16, TERMS = 512, ITEMS = TERMS / LANES;
    constexpr unsigned CONTROL = ((32 - LANES) << 8) | 31;
    static_assert(THREADS == 64 || THREADS == 256, "candidate CTA width");
    const unsigned lane = threadIdx.x & (LANES - 1);
    const unsigned long long col =
        (unsigned long long)blockIdx.x * (THREADS / LANES) + (threadIdx.x / LANES);
    float total[8] = {};
    for (unsigned long long start = 0; start < s.inner; start += TERMS) {
        float partial[8] = {};
        #pragma unroll
        for (unsigned item = 0; item < ITEMS; ++item) {
            const unsigned long long k = start + lane + item * LANES;
            if (col < s.columns && k < s.inner) {
                const float weight = widen_bf16(w[col * s.inner + k]);
                const uint4 packed = *reinterpret_cast<const uint4*>(x + k * 8);
                const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
                #pragma unroll
                for (unsigned row = 0; row < 8; ++row) {
                    const unsigned word = words[row >> 1];
                    const unsigned short value = static_cast<unsigned short>(
                        row & 1 ? word >> 16 : word & 0xffffu);
                    partial[row] =
                        add_product(partial[row], weight, widen_bf16(value));
                }
            }
        }
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) total[row] += partial[row];
    }
    float result[8];
    #pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        result[row] = lane_value(total[row], 0, CONTROL);
    }
    #pragma unroll
    for (unsigned i = 1; i < LANES; ++i) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            result[row] += lane_value(total[row], i, CONTROL);
        }
    }
    if (lane == 0 && col < s.columns) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            store_bf16(y, row * s.output_stride + col, result[row], status);
        }
    }
}

extern "C" __global__ __launch_bounds__(64)
void zg_candidate_lane_strided_b8_inner_major_t64_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    zg_candidate_lane_strided_b8_inner_major<64>(w, x, y, status, s);
}
extern "C" __global__ __launch_bounds__(256)
void zg_candidate_lane_strided_b8_inner_major_t256_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    zg_candidate_lane_strided_b8_inner_major<256>(w, x, y, status, s);
}

template<int BN>
__device__ __forceinline__ void zg_candidate_stage32_k64(
    unsigned short* sw, unsigned short* sx,
    const unsigned short* w, const unsigned short* x,
    GemmShape s, unsigned long long base_m, unsigned long long base_n,
    unsigned long long base_k, bool aligned) {
    constexpr int BM = 32, BK = 64, STRIDE = BK + 8;
    for (unsigned vector = threadIdx.x;
         vector < (BM + BN) * BK / 8;
         vector += 64) {
        const bool weight = vector < BM * BK / 8;
        const unsigned local = weight ? vector : vector - BM * BK / 8;
        const unsigned row = local / (BK / 8), col = (local % (BK / 8)) * 8;
        const auto global_row = (weight ? base_m : base_n) + row;
        const auto global_col = base_k + col;
        const auto limit = weight ? s.columns : s.rows;
        auto* target = (weight ? sw : sx) + row * STRIDE + col;
        const auto* source = weight ? w : x;
        const unsigned valid = global_row < limit && global_col < s.inner
            ? static_cast<unsigned>((s.inner - global_col) < 8
                ? (s.inner - global_col) : 8)
            : 0;
        if (valid) source += global_row * s.inner + global_col;
        if (aligned) {
            copy16(target, source, valid * 2);
        } else {
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                target[i] = i < valid ? source[i] : 0;
            }
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void zg_candidate_segmented_b8_m32_k64(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s, unsigned short* shared) {
    constexpr int BM = 32, BN = 8, STRIDE = 72, STAGE = (BM + BN) * STRIDE;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned row = lane / 4, pair = (lane % 4) * 2;
    const unsigned long long base_m =
        static_cast<unsigned long long>(blockIdx.x) * BM;
    const unsigned long long base_n =
        static_cast<unsigned long long>(blockIdx.y) * BN;
    const bool aligned = ((reinterpret_cast<unsigned long long>(w) |
                           reinterpret_cast<unsigned long long>(x)) & 15) == 0
        && s.inner % 8 == 0;
    float d[4] = {}, rounded[4] = {};
    zg_candidate_stage32_k64<BN>(
        shared, shared + BM * STRIDE, w, x, s, base_m, base_n, 0, aligned);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    unsigned current = 0;
    unsigned long long remaining = s.reduction_terms;
    for (unsigned long long k = 0; k < s.inner; k += 64) {
        unsigned short* sw = shared + current * STAGE;
        unsigned short* sx = sw + BM * STRIDE;
        if (k + 64 < s.inner) {
            unsigned short* next = shared + (current ^ 1) * STAGE;
            zg_candidate_stage32_k64<BN>(
                next, next + BM * STRIDE, w, x, s,
                base_m, base_n, k + 64, aligned);
        }
        #pragma unroll
        for (unsigned kk = 0; kk < 64; kk += 16) {
            const auto* a = sw + (warp * 16 + row) * STRIDE + kk + pair;
            const unsigned a0 = pack(a), a1 = pack(a + 8 * STRIDE);
            const unsigned a2 = pack(a + 8), a3 = pack(a + 8 * STRIDE + 8);
            const auto* b = sx + row * STRIDE + kk + pair;
            const unsigned b0 = pack(b), b1 = pack(b + 8);
            mma_fragment(d, a0, a1, a2, a3, b0, b1);
        }
        remaining -= 64;
        if (remaining == 0 || k + 64 >= s.inner) {
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                rounded[i] += widen_bf16(round_bf16(d[i]));
                d[i] = 0;
            }
            remaining = s.reduction_terms;
        }
        if (k + 64 < s.inner) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
        current ^= 1;
    }
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const auto m = base_m + warp * 16 + row + (i / 2) * 8;
        const auto n = base_n + pair + i % 2;
        if (m < s.columns && n < s.rows) {
            store_bf16(y, n * s.output_stride + m, rounded[i], status);
        }
    }
}

extern "C" __global__ __launch_bounds__(64)
void zg_candidate_segmented_b8_m32_k64_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    __shared__ __align__(16) unsigned short shared[2 * (32 + 8) * 72];
    zg_candidate_segmented_b8_m32_k64(w, x, y, status, s, shared);
}

__device__ __forceinline__ void zg_candidate_widen_pair(
    unsigned word, float& low, float& high) {
    const unsigned low_bits = word << 16;
    const unsigned high_bits = word & 0xffff0000u;
    asm("mov.b32 %0, %1;" : "=f"(low) : "r"(low_bits));
    asm("mov.b32 %0, %1;" : "=f"(high) : "r"(high_bits));
}

__device__ __forceinline__ void zg_candidate_lane_strided_b8_paired_unpack(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    constexpr unsigned LANES = 16, TERMS = 512, ITEMS = TERMS / LANES;
    constexpr unsigned CONTROL = ((32 - LANES) << 8) | 31;
    const unsigned lane = threadIdx.x & (LANES - 1);
    const unsigned long long col =
        (unsigned long long)blockIdx.x * (128 / LANES) + (threadIdx.x / LANES);
    float total[8] = {};
    for (unsigned long long start = 0; start < s.inner; start += TERMS) {
        float partial[8] = {};
        #pragma unroll
        for (unsigned item = 0; item < ITEMS; ++item) {
            const unsigned long long k = start + lane + item * LANES;
            if (col < s.columns && k < s.inner) {
                const float weight = widen_bf16(w[col * s.inner + k]);
                const uint4 packed = *reinterpret_cast<const uint4*>(x + k * 8);
                const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
                float values[8];
                #pragma unroll
                for (unsigned pair = 0; pair < 4; ++pair) {
                    zg_candidate_widen_pair(
                        words[pair], values[2 * pair], values[2 * pair + 1]);
                }
                #pragma unroll
                for (unsigned row = 0; row < 8; ++row) {
                    partial[row] = add_product(partial[row], weight, values[row]);
                }
            }
        }
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) total[row] += partial[row];
    }
    float result[8];
    #pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        result[row] = lane_value(total[row], 0, CONTROL);
    }
    #pragma unroll
    for (unsigned i = 1; i < LANES; ++i) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            result[row] += lane_value(total[row], i, CONTROL);
        }
    }
    if (lane == 0 && col < s.columns) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            store_bf16(y, row * s.output_stride + col, result[row], status);
        }
    }
}

extern "C" __global__ __launch_bounds__(128)
void zg_candidate_lane_strided_b8_paired_unpack_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    zg_candidate_lane_strided_b8_paired_unpack(w, x, y, status, s);
}



__device__ __forceinline__ void zg_candidate_lane_strided_b8_row_major_shared_f32(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s, float* staged) {
    constexpr unsigned ROWS = 8, LANES = 16, TERMS = 512, ITEMS = TERMS / LANES;
    constexpr unsigned CONTROL = ((32 - LANES) << 8) | 31;
    const unsigned lane = threadIdx.x & (LANES - 1);
    const unsigned group = threadIdx.x / LANES;
    const unsigned long long col =
        (unsigned long long)blockIdx.x * (128 / LANES) + group;
    float total[ROWS] = {};
    for (unsigned long long start = 0; start < s.inner; start += TERMS) {
        for (unsigned index = threadIdx.x; index < ROWS * TERMS; index += 128) {
            const unsigned row = index / TERMS;
            const unsigned local = index - row * TERMS;
            const unsigned long long k = start + local;
            staged[index] = k < s.inner ? widen_bf16(x[row * s.inner + k]) : 0.0f;
        }
        __syncthreads();
        float partial[ROWS] = {};
        #pragma unroll
        for (unsigned item = 0; item < ITEMS; ++item) {
            const unsigned local = lane + item * LANES;
            const unsigned long long k = start + local;
            if (col < s.columns && k < s.inner) {
                const float weight = widen_bf16(w[col * s.inner + k]);
                #pragma unroll
                for (unsigned row = 0; row < ROWS; ++row) {
                    partial[row] =
                        add_product(partial[row], weight, staged[row * TERMS + local]);
                }
            }
        }
        #pragma unroll
        for (unsigned row = 0; row < ROWS; ++row) total[row] += partial[row];
        __syncthreads();
    }
    float result[ROWS];
    #pragma unroll
    for (unsigned row = 0; row < ROWS; ++row) {
        result[row] = lane_value(total[row], 0, CONTROL);
    }
    #pragma unroll
    for (unsigned i = 1; i < LANES; ++i) {
        #pragma unroll
        for (unsigned row = 0; row < ROWS; ++row) {
            result[row] += lane_value(total[row], i, CONTROL);
        }
    }
    if (lane == 0 && col < s.columns) {
        #pragma unroll
        for (unsigned row = 0; row < ROWS; ++row) {
            store_bf16(y, row * s.output_stride + col, result[row], status);
        }
    }
}

extern "C" __global__ __launch_bounds__(128)
void zg_candidate_lane_strided_b8_row_major_shared_f32_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    __shared__ float staged[8 * 512];
    zg_candidate_lane_strided_b8_row_major_shared_f32(w, x, y, status, s, staged);
}

__device__ __forceinline__ void zg_candidate_lane_strided_b8_row_major_paired_columns(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    constexpr unsigned LANES = 16, TERMS = 512, ITEMS = TERMS / LANES;
    constexpr unsigned CONTROL = ((32 - LANES) << 8) | 31;
    const unsigned lane = threadIdx.x & (LANES - 1);
    const unsigned group = threadIdx.x / LANES;
    const unsigned long long col0 =
        (unsigned long long)blockIdx.x * 16u + group * 2u;
    const unsigned long long col1 = col0 + 1u;
    float total0[8] = {}, total1[8] = {};
    for (unsigned long long start = 0; start < s.inner; start += TERMS) {
        float partial0[8] = {}, partial1[8] = {};
        #pragma unroll
        for (unsigned item = 0; item < ITEMS; ++item) {
            const unsigned long long k = start + lane + item * LANES;
            if (k < s.inner) {
                float values[8];
                #pragma unroll
                for (unsigned row = 0; row < 8; ++row) {
                    values[row] = widen_bf16(x[row * s.inner + k]);
                }
                if (col0 < s.columns) {
                    const float weight = widen_bf16(w[col0 * s.inner + k]);
                    #pragma unroll
                    for (unsigned row = 0; row < 8; ++row) {
                        partial0[row] = add_product(partial0[row], weight, values[row]);
                    }
                }
                if (col1 < s.columns) {
                    const float weight = widen_bf16(w[col1 * s.inner + k]);
                    #pragma unroll
                    for (unsigned row = 0; row < 8; ++row) {
                        partial1[row] = add_product(partial1[row], weight, values[row]);
                    }
                }
            }
        }
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            total0[row] += partial0[row];
            total1[row] += partial1[row];
        }
    }
    float result0[8], result1[8];
    #pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        result0[row] = lane_value(total0[row], 0, CONTROL);
        result1[row] = lane_value(total1[row], 0, CONTROL);
    }
    #pragma unroll
    for (unsigned i = 1; i < LANES; ++i) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            result0[row] += lane_value(total0[row], i, CONTROL);
            result1[row] += lane_value(total1[row], i, CONTROL);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            if (col0 < s.columns) {
                store_bf16(y, row * s.output_stride + col0, result0[row], status);
            }
            if (col1 < s.columns) {
                store_bf16(y, row * s.output_stride + col1, result1[row], status);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128)
void zg_candidate_lane_strided_b8_row_major_paired_columns_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    zg_candidate_lane_strided_b8_row_major_paired_columns(w, x, y, status, s);
}

__device__ __forceinline__ void zg_candidate_lane_strided_b8_row_major_split4(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    constexpr unsigned LANES = 16, TERMS = 512, ITEMS = TERMS / LANES;
    constexpr unsigned CONTROL = ((32 - LANES) << 8) | 31;
    const unsigned lane = threadIdx.x & 15u;
    const unsigned half_lane = threadIdx.x & 63u;
    const unsigned row_base = (threadIdx.x >> 6u) * 4u;
    const unsigned group = half_lane >> 4u;
    const unsigned long long col =
        (unsigned long long)blockIdx.x * 4u + group;
    float total[4] = {};
    for (unsigned long long start = 0; start < s.inner; start += TERMS) {
        float partial[4] = {};
        #pragma unroll
        for (unsigned item = 0; item < ITEMS; ++item) {
            const unsigned long long k = start + lane + item * LANES;
            if (col < s.columns && k < s.inner) {
                const float weight = widen_bf16(w[col * s.inner + k]);
                #pragma unroll
                for (unsigned local = 0; local < 4; ++local) {
                    const unsigned row = row_base + local;
                    partial[local] = add_product(
                        partial[local], weight, widen_bf16(x[row * s.inner + k]));
                }
            }
        }
        #pragma unroll
        for (unsigned local = 0; local < 4; ++local) total[local] += partial[local];
    }
    float result[4];
    #pragma unroll
    for (unsigned local = 0; local < 4; ++local) {
        result[local] = lane_value(total[local], 0, CONTROL);
    }
    #pragma unroll
    for (unsigned i = 1; i < LANES; ++i) {
        #pragma unroll
        for (unsigned local = 0; local < 4; ++local) {
            result[local] += lane_value(total[local], i, CONTROL);
        }
    }
    if (lane == 0 && col < s.columns) {
        #pragma unroll
        for (unsigned local = 0; local < 4; ++local) {
            const unsigned row = row_base + local;
            store_bf16(y, row * s.output_stride + col, result[local], status);
        }
    }
}

extern "C" __global__ __launch_bounds__(128)
void zg_candidate_lane_strided_b8_row_major_split4_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    zg_candidate_lane_strided_b8_row_major_split4(w, x, y, status, s);
}

__device__ __forceinline__ void zg_candidate_lane_strided_b8_inner_major_warp_shared(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    constexpr unsigned LANES = 16, TERMS = 512, ITEMS = TERMS / LANES;
    constexpr unsigned CONTROL = ((32 - LANES) << 8) | 31;
    const unsigned lane = threadIdx.x & 15u;
    const unsigned warp_lane = threadIdx.x & 31u;
    const unsigned group = threadIdx.x >> 4u;
    const unsigned long long col =
        (unsigned long long)blockIdx.x * 8u + group;
    float total[8] = {};
    for (unsigned long long start = 0; start < s.inner; start += TERMS) {
        float partial[8] = {};
        #pragma unroll
        for (unsigned item = 0; item < ITEMS; ++item) {
            const unsigned long long k = start + lane + item * LANES;
            if (col < s.columns && k < s.inner) {
                const float weight = widen_bf16(w[col * s.inner + k]);
                uint4 packed = {};
                if (warp_lane < 16u) {
                    packed = *reinterpret_cast<const uint4*>(x + k * 8);
                }
                packed.x = __shfl_sync(0xffffffffu, packed.x, lane);
                packed.y = __shfl_sync(0xffffffffu, packed.y, lane);
                packed.z = __shfl_sync(0xffffffffu, packed.z, lane);
                packed.w = __shfl_sync(0xffffffffu, packed.w, lane);
                const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
                #pragma unroll
                for (unsigned row = 0; row < 8; ++row) {
                    const unsigned word = words[row >> 1];
                    const unsigned short value = static_cast<unsigned short>(
                        row & 1 ? word >> 16 : word & 0xffffu);
                    partial[row] =
                        add_product(partial[row], weight, widen_bf16(value));
                }
            }
        }
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) total[row] += partial[row];
    }
    float result[8];
    #pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        result[row] = lane_value(total[row], 0, CONTROL);
    }
    #pragma unroll
    for (unsigned i = 1; i < LANES; ++i) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            result[row] += lane_value(total[row], i, CONTROL);
        }
    }
    if (lane == 0 && col < s.columns) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            store_bf16(y, row * s.output_stride + col, result[row], status);
        }
    }
}

extern "C" __global__ __launch_bounds__(128)
void zg_candidate_lane_strided_b8_inner_major_warp_shared_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    zg_candidate_lane_strided_b8_inner_major_warp_shared(w, x, y, status, s);
}

__device__ __forceinline__ void zg_candidate_lane_strided_b8_inner_major_paired_columns(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    constexpr unsigned LANES = 16, TERMS = 512, ITEMS = TERMS / LANES;
    constexpr unsigned CONTROL = ((32 - LANES) << 8) | 31;
    const unsigned lane = threadIdx.x & 15u;
    const unsigned group = threadIdx.x >> 4u;
    const unsigned long long col0 =
        (unsigned long long)blockIdx.x * 16u + group * 2u;
    const unsigned long long col1 = col0 + 1u;
    float total0[8] = {}, total1[8] = {};
    for (unsigned long long start = 0; start < s.inner; start += TERMS) {
        float partial0[8] = {}, partial1[8] = {};
        #pragma unroll
        for (unsigned item = 0; item < ITEMS; ++item) {
            const unsigned long long k = start + lane + item * LANES;
            if (k < s.inner) {
                const uint4 packed = *reinterpret_cast<const uint4*>(x + k * 8);
                const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
                float values[8];
                #pragma unroll
                for (unsigned row = 0; row < 8; ++row) {
                    const unsigned word = words[row >> 1];
                    const unsigned short value = static_cast<unsigned short>(
                        row & 1 ? word >> 16 : word & 0xffffu);
                    values[row] = widen_bf16(value);
                }
                if (col0 < s.columns) {
                    const float weight = widen_bf16(w[col0 * s.inner + k]);
                    #pragma unroll
                    for (unsigned row = 0; row < 8; ++row) {
                        partial0[row] = add_product(partial0[row], weight, values[row]);
                    }
                }
                if (col1 < s.columns) {
                    const float weight = widen_bf16(w[col1 * s.inner + k]);
                    #pragma unroll
                    for (unsigned row = 0; row < 8; ++row) {
                        partial1[row] = add_product(partial1[row], weight, values[row]);
                    }
                }
            }
        }
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            total0[row] += partial0[row];
            total1[row] += partial1[row];
        }
    }
    float result0[8], result1[8];
    #pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        result0[row] = lane_value(total0[row], 0, CONTROL);
        result1[row] = lane_value(total1[row], 0, CONTROL);
    }
    #pragma unroll
    for (unsigned i = 1; i < LANES; ++i) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            result0[row] += lane_value(total0[row], i, CONTROL);
            result1[row] += lane_value(total1[row], i, CONTROL);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            if (col0 < s.columns) {
                store_bf16(y, row * s.output_stride + col0, result0[row], status);
            }
            if (col1 < s.columns) {
                store_bf16(y, row * s.output_stride + col1, result1[row], status);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128)
void zg_candidate_lane_strided_b8_inner_major_paired_columns_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    zg_candidate_lane_strided_b8_inner_major_paired_columns(w, x, y, status, s);
}

__device__ __forceinline__ void zg_candidate_lane_strided_b8_inner_major_unroll8(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    constexpr unsigned LANES = 16, TERMS = 512, ITEMS = TERMS / LANES;
    constexpr unsigned CONTROL = ((32 - LANES) << 8) | 31;
    const unsigned lane = threadIdx.x & 15u;
    const unsigned long long col =
        (unsigned long long)blockIdx.x * 8u + (threadIdx.x >> 4u);
    float total[8] = {};
    for (unsigned long long start = 0; start < s.inner; start += TERMS) {
        float partial[8] = {};
        #pragma unroll 8
        for (unsigned item = 0; item < ITEMS; ++item) {
            const unsigned long long k = start + lane + item * LANES;
            if (col < s.columns && k < s.inner) {
                const float weight = widen_bf16(w[col * s.inner + k]);
                const uint4 packed = *reinterpret_cast<const uint4*>(x + k * 8);
                const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
                #pragma unroll
                for (unsigned row = 0; row < 8; ++row) {
                    const unsigned word = words[row >> 1];
                    const unsigned short value = static_cast<unsigned short>(
                        row & 1 ? word >> 16 : word & 0xffffu);
                    partial[row] =
                        add_product(partial[row], weight, widen_bf16(value));
                }
            }
        }
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) total[row] += partial[row];
    }
    float result[8];
    #pragma unroll
    for (unsigned row = 0; row < 8; ++row) {
        result[row] = lane_value(total[row], 0, CONTROL);
    }
    #pragma unroll
    for (unsigned i = 1; i < LANES; ++i) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            result[row] += lane_value(total[row], i, CONTROL);
        }
    }
    if (lane == 0 && col < s.columns) {
        #pragma unroll
        for (unsigned row = 0; row < 8; ++row) {
            store_bf16(y, row * s.output_stride + col, result[row], status);
        }
    }
}

extern "C" __global__ __launch_bounds__(128)
void zg_candidate_lane_strided_b8_inner_major_unroll8_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    zg_candidate_lane_strided_b8_inner_major_unroll8(w, x, y, status, s);
}

template<int BN>
__device__ __forceinline__ void zg_candidate_stage128_k64(
    unsigned short* sw, unsigned short* sx,
    const unsigned short* w, const unsigned short* x,
    GemmShape s, unsigned long long base_m, unsigned long long base_n,
    unsigned long long base_k, bool aligned) {
    constexpr int BM = 128, BK = 64, STRIDE = BK + 8;
    for (unsigned vector = threadIdx.x;
         vector < (BM + BN) * BK / 8;
         vector += 256) {
        const bool weight = vector < BM * BK / 8;
        const unsigned local = weight ? vector : vector - BM * BK / 8;
        const unsigned row = local / (BK / 8), col = (local % (BK / 8)) * 8;
        const auto global_row = (weight ? base_m : base_n) + row;
        const auto global_col = base_k + col;
        const auto limit = weight ? s.columns : s.rows;
        auto* target = (weight ? sw : sx) + row * STRIDE + col;
        const auto* source = weight ? w : x;
        const unsigned valid = global_row < limit && global_col < s.inner
            ? static_cast<unsigned>((s.inner - global_col) < 8
                ? (s.inner - global_col) : 8)
            : 0;
        if (valid) source += global_row * s.inner + global_col;
        if (aligned) {
            copy16(target, source, valid * 2);
        } else {
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                target[i] = i < valid ? source[i] : 0;
            }
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void zg_candidate_ordered_b8_m128_k64(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s, unsigned short* shared) {
    constexpr int BM = 128, BN = 8, STRIDE = 72, STAGE = (BM + BN) * STRIDE;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned row = lane / 4, pair = (lane % 4) * 2;
    const unsigned long long base_m =
        static_cast<unsigned long long>(blockIdx.x) * BM;
    const unsigned long long base_n =
        static_cast<unsigned long long>(blockIdx.y) * BN;
    const bool aligned = ((reinterpret_cast<unsigned long long>(w) |
                           reinterpret_cast<unsigned long long>(x)) & 15) == 0
        && s.inner % 8 == 0;
    float d[4] = {};
    zg_candidate_stage128_k64<BN>(
        shared, shared + BM * STRIDE, w, x, s, base_m, base_n, 0, aligned);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    unsigned current = 0;
    for (unsigned long long k = 0; k < s.inner; k += 64) {
        unsigned short* sw = shared + current * STAGE;
        unsigned short* sx = sw + BM * STRIDE;
        if (k + 64 < s.inner) {
            unsigned short* next = shared + (current ^ 1) * STAGE;
            zg_candidate_stage128_k64<BN>(
                next, next + BM * STRIDE, w, x, s,
                base_m, base_n, k + 64, aligned);
        }
        #pragma unroll
        for (unsigned kk = 0; kk < 64; kk += 16) {
            const auto* a = sw + (warp * 16 + row) * STRIDE + kk + pair;
            const unsigned a0 = pack(a), a1 = pack(a + 8 * STRIDE);
            const unsigned a2 = pack(a + 8), a3 = pack(a + 8 * STRIDE + 8);
            const auto* b = sx + row * STRIDE + kk + pair;
            const unsigned b0 = pack(b), b1 = pack(b + 8);
            mma_fragment(d, a0, a1, a2, a3, b0, b1);
        }
        if (k + 64 < s.inner) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
        current ^= 1;
    }
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const auto m = base_m + warp * 16 + row + (i / 2) * 8;
        const auto n = base_n + pair + i % 2;
        if (m < s.columns && n < s.rows) {
            store_bf16(y, n * s.output_stride + m, d[i], status);
        }
    }
}

extern "C" __global__ __launch_bounds__(256)
void zg_candidate_ordered_b8_m128_k64_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    __shared__ __align__(16) unsigned short shared[2 * (128 + 8) * 72];
    zg_candidate_ordered_b8_m128_k64(w, x, y, status, s, shared);
}

extern "C" __global__ __launch_bounds__(128)
void zg_candidate_ordered_n16_k64_certified(
    const unsigned short* w, const unsigned short* x, unsigned short* y,
    unsigned* status, GemmShape s) {
    __shared__ __align__(16) unsigned short shared[2 * (64 + 16) * 72];
    gemm_ordered_k64_body<16>(w, x, y, status, s, shared);
}
