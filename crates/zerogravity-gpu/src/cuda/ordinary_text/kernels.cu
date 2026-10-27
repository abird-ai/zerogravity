// Source-derived ordinary BF16 text primitives. No captured launch metadata.
#if __CUDA_ARCH__ != 860
#error Ordinary text CUDA primitives require SM86
#endif
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define ZG_THREADS 256
#define ZG_BF16_BYTES 2

struct alignas(8) ZgWords1 { unsigned long long words[1]; };
struct alignas(8) ZgWords2 { unsigned long long words[2]; };
struct alignas(8) ZgWords3 { unsigned long long words[3]; };
struct alignas(8) ZgWords4 { unsigned long long words[4]; };
struct alignas(8) ZgWords5 { unsigned long long words[5]; };
static_assert(sizeof(ZgWords1) == 8 && alignof(ZgWords1) == 8, "words1 ABI");
static_assert(sizeof(ZgWords2) == 16 && alignof(ZgWords2) == 8, "words2 ABI");
static_assert(sizeof(ZgWords3) == 24 && alignof(ZgWords3) == 8, "words3 ABI");
static_assert(sizeof(ZgWords4) == 32 && alignof(ZgWords4) == 8, "words4 ABI");
static_assert(sizeof(ZgWords5) == 40 && alignof(ZgWords5) == 8, "words5 ABI");

static __device__ __forceinline__ unsigned zg_u32(unsigned long long word) {
    return (unsigned)word;
}
static __device__ __forceinline__ float zg_f32(unsigned long long word) {
    return __uint_as_float((unsigned)word);
}
static __device__ __forceinline__ float zg_negative_infinity() {
    return __uint_as_float(0xff800000u);
}
static __device__ __forceinline__ float zg_load(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}
static __device__ __forceinline__ __nv_bfloat16 zg_store(float x) {
    return __float2bfloat16_rn(x);
}
static __device__ __forceinline__ void zg_bad(unsigned* status, unsigned flag) {
    atomicOr(status, flag);
}

// words = [rows, base, capacity]
extern "C" __global__ __launch_bounds__(ZG_THREADS)
void zg_ot_positions_v1(void* positions_raw, void* status_raw, ZgWords3 scalars) {
    unsigned* positions = (unsigned*)positions_raw;
    unsigned* status = (unsigned*)status_raw;
    const unsigned rows = zg_u32(scalars.words[0]);
    const unsigned base = zg_u32(scalars.words[1]);
    const unsigned capacity = zg_u32(scalars.words[2]);
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        const unsigned position = base + i;
        if (position < capacity) positions[i] = position;
        else zg_bad(status, 1u);
    }
}

// words = [rows, capacity]. Query/key are canonical [row, head, HEAD_DIM].
extern "C" __global__ __launch_bounds__(ZG_THREADS)
void zg_ot_rope_v1(void* query_raw, void* key_raw, void* positions_raw,
                   void* inverse_raw, void* status_raw, ZgWords2 scalars) {
    __nv_bfloat16* query = (__nv_bfloat16*)query_raw;
    __nv_bfloat16* key = (__nv_bfloat16*)key_raw;
    const unsigned* positions = (const unsigned*)positions_raw;
    const float* inverse = (const float*)inverse_raw;
    unsigned* status = (unsigned*)status_raw;
    const unsigned rows = zg_u32(scalars.words[0]);
    const unsigned capacity = zg_u32(scalars.words[1]);
    const unsigned half = ZG_ROTARY_WIDTH / 2;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned q_pairs = rows * ZG_QUERY_HEADS * half;
    const unsigned k_pairs = rows * ZG_KV_HEADS * half;
    if (i < q_pairs) {
        const unsigned pair = i % half;
        const unsigned h = (i / half) % ZG_QUERY_HEADS;
        const unsigned row = i / (half * ZG_QUERY_HEADS);
        const unsigned position = positions[row];
        if (position >= capacity) { zg_bad(status, 1u); return; }
        const unsigned first = (row * ZG_QUERY_HEADS + h) * ZG_HEAD_DIM + pair;
        const unsigned second = first + half;
        const float a = zg_load(query + first);
        const float b = zg_load(query + second);
        const float frequency = inverse[pair];
        if (!isfinite(a) || !isfinite(b) || !isfinite(frequency)) zg_bad(status, 2u);
        const float angle = (float)position * frequency;
        float sine, cosine;
        sincosf(angle, &sine, &cosine);
        const float x = a * cosine - b * sine;
        const float y = b * cosine + a * sine;
        query[first] = zg_store(x); query[second] = zg_store(y);
        if (!isfinite(x) || !isfinite(y)) zg_bad(status, 2u);
    }
    if (i < k_pairs) {
        const unsigned pair = i % half;
        const unsigned h = (i / half) % ZG_KV_HEADS;
        const unsigned row = i / (half * ZG_KV_HEADS);
        const unsigned position = positions[row];
        if (position >= capacity) { zg_bad(status, 1u); return; }
        const unsigned first = (row * ZG_KV_HEADS + h) * ZG_HEAD_DIM + pair;
        const unsigned second = first + half;
        const float a = zg_load(key + first);
        const float b = zg_load(key + second);
        const float frequency = inverse[pair];
        if (!isfinite(a) || !isfinite(b) || !isfinite(frequency)) zg_bad(status, 2u);
        const float angle = (float)position * frequency;
        float sine, cosine;
        sincosf(angle, &sine, &cosine);
        const float x = a * cosine - b * sine;
        const float y = b * cosine + a * sine;
        key[first] = zg_store(x); key[second] = zg_store(y);
        if (!isfinite(x) || !isfinite(y)) zg_bad(status, 2u);
    }
}

// words = [rows]. Table is canonical [vocabulary, hidden].
extern "C" __global__ __launch_bounds__(ZG_THREADS)
void zg_ot_embedding_v1(void* tokens_raw, void* table_raw, void* output_raw,
                        void* status_raw, ZgWords1 scalars) {
    const unsigned* tokens = (const unsigned*)tokens_raw;
    const __nv_bfloat16* table = (const __nv_bfloat16*)table_raw;
    __nv_bfloat16* output = (__nv_bfloat16*)output_raw;
    unsigned* status = (unsigned*)status_raw;
    const unsigned rows = zg_u32(scalars.words[0]);
    const size_t count = (size_t)rows * ZG_HIDDEN_WIDTH;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const unsigned row = (unsigned)(i / ZG_HIDDEN_WIDTH);
        const unsigned column = (unsigned)(i % ZG_HIDDEN_WIDTH);
        const unsigned token = tokens[row];
        if (token >= ZG_VOCABULARY) {
            output[i] = zg_store(0.0f); zg_bad(status, 1u);
        } else {
            const float weight = zg_load(table + (size_t)token * ZG_HIDDEN_WIDTH + column);
            output[i] = zg_store(weight);
            if (!isfinite(weight)) zg_bad(status, 2u);
        }
    }
}

// words = [rows, epsilon_f32_bits].
extern "C" __global__ __launch_bounds__(ZG_THREADS)
void zg_ot_rms_v1(void* input_raw, void* raw_gamma_raw, void* output_raw,
                  void* status_raw, ZgWords2 scalars) {
    const __nv_bfloat16* input = (const __nv_bfloat16*)input_raw;
    const __nv_bfloat16* raw_gamma = (const __nv_bfloat16*)raw_gamma_raw;
    __nv_bfloat16* output = (__nv_bfloat16*)output_raw;
    unsigned* status = (unsigned*)status_raw;
    __shared__ float reduction[ZG_THREADS];
    const unsigned row = blockIdx.x;
    const unsigned rows = zg_u32(scalars.words[0]);
    if (row >= rows) return;
    float sum = 0.0f;
    for (unsigned c = threadIdx.x; c < ZG_HIDDEN_WIDTH; c += blockDim.x) {
        const float x = zg_load(input + (size_t)row * ZG_HIDDEN_WIDTH + c);
        sum += x * x;
        if (!isfinite(x)) zg_bad(status, 2u);
    }
    reduction[threadIdx.x] = sum; __syncthreads();
    for (unsigned stride = ZG_THREADS / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0 && !isfinite(reduction[0])) zg_bad(status, 2u);
    const float inverse = rsqrtf(reduction[0] / (float)ZG_HIDDEN_WIDTH + zg_f32(scalars.words[1]));
    if (threadIdx.x == 0 && (!isfinite(inverse) || inverse == 0.0f)) zg_bad(status, 2u);
    for (unsigned c = threadIdx.x; c < ZG_HIDDEN_WIDTH; c += blockDim.x) {
        const size_t i = (size_t)row * ZG_HIDDEN_WIDTH + c;
        const float x = zg_load(input + i);
        const float gamma = zg_load(raw_gamma + c);
        const float y = x * inverse * (1.0f + gamma);
        output[i] = zg_store(y);
        if (!isfinite(x) || !isfinite(gamma) || !isfinite(inverse) || !isfinite(y)) zg_bad(status, 2u);
    }
}

// words = [rows, epsilon_f32_bits]. Residual is updated before normalization.
extern "C" __global__ __launch_bounds__(ZG_THREADS)
void zg_ot_residual_rms_v1(void* input_raw, void* residual_raw, void* raw_gamma_raw,
                           void* output_raw, void* status_raw, ZgWords2 scalars) {
    const __nv_bfloat16* input = (const __nv_bfloat16*)input_raw;
    __nv_bfloat16* residual = (__nv_bfloat16*)residual_raw;
    const __nv_bfloat16* raw_gamma = (const __nv_bfloat16*)raw_gamma_raw;
    __nv_bfloat16* output = (__nv_bfloat16*)output_raw;
    unsigned* status = (unsigned*)status_raw;
    __shared__ float reduction[ZG_THREADS];
    const unsigned row = blockIdx.x;
    const unsigned rows = zg_u32(scalars.words[0]);
    if (row >= rows) return;
    float sum = 0.0f;
    for (unsigned c = threadIdx.x; c < ZG_HIDDEN_WIDTH; c += blockDim.x) {
        const size_t i = (size_t)row * ZG_HIDDEN_WIDTH + c;
        const float input_value = zg_load(input + i);
        const float residual_value = zg_load(residual + i);
        const float x = input_value + residual_value;
        const __nv_bfloat16 rounded = zg_store(x);
        residual[i] = rounded;
        const float r = __bfloat162float(rounded); sum += r * r;
        if (!isfinite(input_value) || !isfinite(residual_value) || !isfinite(r)) zg_bad(status, 2u);
    }
    reduction[threadIdx.x] = sum; __syncthreads();
    for (unsigned stride = ZG_THREADS / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0 && !isfinite(reduction[0])) zg_bad(status, 2u);
    const float inverse = rsqrtf(reduction[0] / (float)ZG_HIDDEN_WIDTH + zg_f32(scalars.words[1]));
    if (threadIdx.x == 0 && (!isfinite(inverse) || inverse == 0.0f)) zg_bad(status, 2u);
    for (unsigned c = threadIdx.x; c < ZG_HIDDEN_WIDTH; c += blockDim.x) {
        const size_t i = (size_t)row * ZG_HIDDEN_WIDTH + c;
        const float residual_value = zg_load(residual + i);
        const float gamma = zg_load(raw_gamma + c);
        const float y = residual_value * inverse * (1.0f + gamma);
        output[i] = zg_store(y);
        if (!isfinite(residual_value) || !isfinite(gamma) || !isfinite(inverse) || !isfinite(y)) zg_bad(status, 2u);
    }
}

// Q/K zero-centered RMS over packed Q/G/K/V projection rows: one block/head.
// words = [rows, input_offset, input_row_stride, input_head_stride, epsilon_f32_bits].
#define ZG_DEFINE_HEAD_RMS(NAME, HEADS) \
extern "C" __global__ __launch_bounds__(ZG_THREADS) \
void NAME(void* input_raw, void* raw_gamma_raw, void* output_raw, \
          void* status_raw, ZgWords5 scalars) { \
    const __nv_bfloat16* input = (const __nv_bfloat16*)input_raw; \
    const __nv_bfloat16* raw_gamma = (const __nv_bfloat16*)raw_gamma_raw; \
    __nv_bfloat16* output = (__nv_bfloat16*)output_raw; \
    unsigned* status = (unsigned*)status_raw; \
    __shared__ float reduction[ZG_THREADS]; \
    const unsigned group = blockIdx.x; \
    const unsigned rows = zg_u32(scalars.words[0]); \
    const unsigned input_offset = zg_u32(scalars.words[1]); \
    const unsigned input_stride = zg_u32(scalars.words[2]); \
    const unsigned input_head_stride = zg_u32(scalars.words[3]); \
    if (group >= rows * (HEADS)) return; \
    const unsigned lane = threadIdx.x; \
    const unsigned row = group / (HEADS); \
    const unsigned head = group % (HEADS); \
    const size_t input_i = (size_t)row * input_stride + input_offset + (size_t)head * input_head_stride + lane; \
    const size_t output_i = (size_t)group * ZG_HEAD_DIM + lane; \
    const float x = zg_load(input + input_i); \
    reduction[lane] = x * x; \
    if (!isfinite(x)) zg_bad(status, 2u); \
    __syncthreads(); \
    for (unsigned stride = ZG_THREADS / 2; stride; stride >>= 1) { \
        if (lane < stride) reduction[lane] += reduction[lane + stride]; \
        __syncthreads(); \
    } \
    if (lane == 0 && !isfinite(reduction[0])) zg_bad(status, 2u); \
    const float inverse = rsqrtf(reduction[0] / (float)ZG_HEAD_DIM + zg_f32(scalars.words[4])); \
    if (lane == 0 && (!isfinite(inverse) || inverse == 0.0f)) zg_bad(status, 2u); \
    const float gamma = zg_load(raw_gamma + lane); \
    const float y = x * inverse * (1.0f + gamma); \
    output[output_i] = zg_store(y); \
    if (!isfinite(gamma) || !isfinite(inverse) || !isfinite(y)) zg_bad(status, 2u); \
}

ZG_DEFINE_HEAD_RMS(zg_ot_query_head_rms_v1, ZG_QUERY_HEADS)
ZG_DEFINE_HEAD_RMS(zg_ot_key_head_rms_v1, ZG_KV_HEADS)
#undef ZG_DEFINE_HEAD_RMS

// Packed GateUp words = [rows, gate_offset, up_offset, row_stride] in elements.
extern "C" __global__ __launch_bounds__(ZG_THREADS)
void zg_ot_silu_up_v1(void* packed_raw, void* output_raw,
                      void* status_raw, ZgWords4 scalars) {
    const __nv_bfloat16* packed = (const __nv_bfloat16*)packed_raw;
    __nv_bfloat16* output = (__nv_bfloat16*)output_raw;
    unsigned* status = (unsigned*)status_raw;
    const unsigned rows = zg_u32(scalars.words[0]);
    const unsigned gate_offset = zg_u32(scalars.words[1]);
    const unsigned up_offset = zg_u32(scalars.words[2]);
    const unsigned row_stride = zg_u32(scalars.words[3]);
    const size_t count = (size_t)rows * ZG_INTERMEDIATE_WIDTH;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const size_t row = i / ZG_INTERMEDIATE_WIDTH;
        const size_t column = i % ZG_INTERMEDIATE_WIDTH;
        const float gate = zg_load(packed + row * row_stride + gate_offset + column);
        const float up = zg_load(packed + row * row_stride + up_offset + column);
        const __nv_bfloat16 activated = zg_store(gate / (1.0f + expf(-gate)));
        const float y = zg_load(&activated) * up;
        output[i] = zg_store(y);
        if (!isfinite(gate) || !isfinite(up) || !isfinite(y)) zg_bad(status, 2u);
    }
}

// Interleaved Q/G words = [rows, gate_offset, head_stride, row_stride] in elements.
extern "C" __global__ __launch_bounds__(ZG_THREADS)
void zg_ot_sigmoid_v1(void* packed_raw, void* output_raw,
                      void* status_raw, ZgWords4 scalars) {
    const __nv_bfloat16* packed = (const __nv_bfloat16*)packed_raw;
    __nv_bfloat16* output = (__nv_bfloat16*)output_raw;
    unsigned* status = (unsigned*)status_raw;
    const size_t width = (size_t)ZG_QUERY_HEADS * ZG_HEAD_DIM;
    const unsigned rows = zg_u32(scalars.words[0]);
    const unsigned gate_offset = zg_u32(scalars.words[1]);
    const unsigned head_stride = zg_u32(scalars.words[2]);
    const unsigned row_stride = zg_u32(scalars.words[3]);
    const size_t count = (size_t)rows * width;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const size_t row = i / width;
        const size_t column = i % width;
        const size_t head = column / ZG_HEAD_DIM;
        const size_t lane = column % ZG_HEAD_DIM;
        const float gate = zg_load(packed + row * row_stride + head * head_stride + gate_offset + lane);
        const float sigmoid = 1.0f / (1.0f + expf(-gate));
        output[i] = zg_store(sigmoid);
        if (!isfinite(gate) || !isfinite(sigmoid)) zg_bad(status, 2u);
    }
}

// Multiply consumes the separately BF16-rounded sigmoid output.
extern "C" __global__ __launch_bounds__(ZG_THREADS)
void zg_ot_multiply_v1(void* input_raw, void* gate_raw, void* output_raw,
                       void* status_raw, ZgWords1 scalars) {
    const __nv_bfloat16* input = (const __nv_bfloat16*)input_raw;
    const __nv_bfloat16* gate = (const __nv_bfloat16*)gate_raw;
    __nv_bfloat16* output = (__nv_bfloat16*)output_raw;
    unsigned* status = (unsigned*)status_raw;
    const size_t width = (size_t)ZG_QUERY_HEADS * ZG_HEAD_DIM;
    const size_t count = (size_t)zg_u32(scalars.words[0]) * width;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float input_value = zg_load(input + i);
        const float rounded_sigmoid = zg_load(gate + i);
        const float y = input_value * rounded_sigmoid;
        output[i] = zg_store(y);
        if (!isfinite(input_value) || !isfinite(rounded_sigmoid) || !isfinite(y)) zg_bad(status, 2u);
    }
}

extern "C" __global__ __launch_bounds__(ZG_THREADS)
void zg_ot_residual_v1(void* input_raw, void* residual_raw, void* output_raw,
                       void* status_raw, ZgWords1 scalars) {
    const __nv_bfloat16* input = (const __nv_bfloat16*)input_raw;
    const __nv_bfloat16* residual = (const __nv_bfloat16*)residual_raw;
    __nv_bfloat16* output = (__nv_bfloat16*)output_raw;
    unsigned* status = (unsigned*)status_raw;
    const size_t count = (size_t)zg_u32(scalars.words[0]) * ZG_HIDDEN_WIDTH;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float input_value = zg_load(input + i);
        const float residual_value = zg_load(residual + i);
        const float y = input_value + residual_value;
        output[i] = zg_store(y);
        if (!isfinite(input_value) || !isfinite(residual_value) || !isfinite(y)) zg_bad(status, 2u);
    }
}

// words = [rows, base, capacity, value_offset, value_row_stride] in elements.
// Cache layout is [token, kv_head, HEAD_DIM]; V is read from packed Q/G/K/V.
extern "C" __global__ __launch_bounds__(ZG_THREADS)
void zg_ot_kv_append_v1(void* key_raw, void* packed_raw, void* key_cache_raw,
                        void* value_cache_raw, void* status_raw, ZgWords5 scalars) {
    const __nv_bfloat16* key = (const __nv_bfloat16*)key_raw;
    const __nv_bfloat16* packed = (const __nv_bfloat16*)packed_raw;
    __nv_bfloat16* key_cache = (__nv_bfloat16*)key_cache_raw;
    __nv_bfloat16* value_cache = (__nv_bfloat16*)value_cache_raw;
    unsigned* status = (unsigned*)status_raw;
    const unsigned rows = zg_u32(scalars.words[0]);
    const unsigned base = zg_u32(scalars.words[1]);
    const unsigned capacity = zg_u32(scalars.words[2]);
    const unsigned value_offset = zg_u32(scalars.words[3]);
    const unsigned value_stride = zg_u32(scalars.words[4]);
    const size_t row_width = (size_t)ZG_KV_HEADS * ZG_HEAD_DIM;
    const size_t count = (size_t)rows * row_width;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const unsigned row = (unsigned)(i / row_width);
        const size_t destination = ((size_t)base + row) * row_width + i % row_width;
        if (base + row >= capacity) { zg_bad(status, 1u); return; }
        const size_t value_i = (size_t)row * value_stride + value_offset + i % row_width;
        const float key_value = zg_load(key + i);
        const float value_value = zg_load(packed + value_i);
        key_cache[destination] = zg_store(key_value);
        value_cache[destination] = zg_store(value_value);
        if (!isfinite(key_value) || !isfinite(value_value)) zg_bad(status, 2u);
    }
}

// UNQUALIFIED causal online-FP32 softmax diagnostic. It does not reproduce
// Qwen's declared BF16 attention cast/reduction schedule and is never selected.
// words = [rows, base, capacity].
extern "C" __global__ __launch_bounds__(ZG_THREADS)
void zg_ot_unqualified_online_fp32_attention_v1(void* query_raw, void* key_cache_raw, void* value_cache_raw,
                        void* output_raw, void* status_raw, ZgWords3 scalars) {
    const __nv_bfloat16* query = (const __nv_bfloat16*)query_raw;
    const __nv_bfloat16* key_cache = (const __nv_bfloat16*)key_cache_raw;
    const __nv_bfloat16* value_cache = (const __nv_bfloat16*)value_cache_raw;
    __nv_bfloat16* output = (__nv_bfloat16*)output_raw;
    unsigned* status = (unsigned*)status_raw;
    __shared__ float dot[ZG_THREADS];
    const unsigned rows = zg_u32(scalars.words[0]);
    const unsigned base = zg_u32(scalars.words[1]);
    const unsigned capacity = zg_u32(scalars.words[2]);
    const unsigned row = blockIdx.x / ZG_QUERY_HEADS;
    const unsigned head = blockIdx.x % ZG_QUERY_HEADS;
    if (row >= rows || threadIdx.x >= ZG_HEAD_DIM || base + row >= capacity) return;
    const unsigned d = threadIdx.x;
    const unsigned kv_head = head / (ZG_QUERY_HEADS / ZG_KV_HEADS);
    const float q = zg_load(query + ((size_t)row * ZG_QUERY_HEADS + head) * ZG_HEAD_DIM + d);
    float maximum = zg_negative_infinity(), normalizer = 0.0f, accumulator = 0.0f;
    const unsigned valid = base + row + 1;
    for (unsigned token = 0; token < valid; ++token) {
        const size_t cache_i = ((size_t)token * ZG_KV_HEADS + kv_head) * ZG_HEAD_DIM + d;
        const float key_value = zg_load(key_cache + cache_i);
        const float value_value = zg_load(value_cache + cache_i);
        if (!isfinite(q) || !isfinite(key_value) || !isfinite(value_value)) zg_bad(status, 2u);
        dot[d] = q * key_value; __syncthreads();
        for (unsigned stride = ZG_HEAD_DIM / 2; stride; stride >>= 1) {
            if (d < stride) dot[d] += dot[d + stride];
            __syncthreads();
        }
        const float score = dot[0] * ZG_ATTENTION_SCALE;
        const float next_maximum = fmaxf(maximum, score);
        const float previous = maximum == zg_negative_infinity() ? 0.0f : expf(maximum - next_maximum);
        const float current = expf(score - next_maximum);
        accumulator = accumulator * previous + value_value * current;
        normalizer = normalizer * previous + current;
        maximum = next_maximum;
        __syncthreads();
    }
    const float y = accumulator / normalizer;
    const size_t out_i = ((size_t)row * ZG_QUERY_HEADS + head) * ZG_HEAD_DIM + d;
    output[out_i] = zg_store(y);
    if (!isfinite(q) || !isfinite(y) || !(normalizer > 0.0f)) zg_bad(status, 2u);
}

// One block per row performs a deterministic full-vocabulary reduction.
extern "C" __global__ __launch_bounds__(ZG_THREADS)
void zg_ot_greedy_v1(void* logits_raw, void* tokens_raw, void* maxima_raw,
                     void* status_raw, ZgWords1 scalars) {
    const float* logits = (const float*)logits_raw;
    unsigned* tokens = (unsigned*)tokens_raw;
    float* maxima = (float*)maxima_raw;
    unsigned* status = (unsigned*)status_raw;
    __shared__ float values[ZG_THREADS];
    __shared__ unsigned indices[ZG_THREADS];
    const unsigned row = blockIdx.x;
    const unsigned rows = zg_u32(scalars.words[0]);
    if (row >= rows) return;
    float best = zg_negative_infinity(); unsigned best_index = 0;
    for (unsigned i = threadIdx.x; i < ZG_VOCABULARY; i += blockDim.x) {
        const float value = logits[(size_t)row * ZG_VOCABULARY + i];
        if (!isfinite(value)) {
            zg_bad(status, 2u | (isnan(value) ? 4u : 0u));
            continue;
        }
        if (value > best || (value == best && i < best_index)) { best = value; best_index = i; }
    }
    values[threadIdx.x] = best; indices[threadIdx.x] = best_index; __syncthreads();
    for (unsigned stride = ZG_THREADS / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other = values[threadIdx.x + stride];
            const unsigned other_index = indices[threadIdx.x + stride];
            if (other > values[threadIdx.x] ||
                (other == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
                values[threadIdx.x] = other; indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { tokens[row] = indices[0]; maxima[row] = values[0]; }
}
