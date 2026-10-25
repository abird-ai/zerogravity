// Exact per-row K/V transport, dispatched by blockIdx.y. ONE definition per
// equation family: each grouped lane runs the byte-identical per-row equations
// of the original qualification/model_kv_copy.cu (paired K/V lanes, page stride
// block*557056 + head*512 + position*1024 + lane, base/end/block guard OR 1,
// nonfinite OR 2, future padding copied unchanged from that row's own prior).
// Controls are per-row 8/4/4-byte views of the same original leases, so the
// [0] control reads stay byte-identical to the per-row kernel.
#if __CUDA_ARCH__ != 860
#error Closed SM86 original KV transport
#endif
extern "C" __global__ __launch_bounds__(128)
void ZG_ENTRY(const unsigned int* pk0, const unsigned int* pk1,
              const unsigned int* pk2, const unsigned int* pk3,
              const unsigned int* pv0, const unsigned int* pv1,
              const unsigned int* pv2, const unsigned int* pv3,
              unsigned short* native_cache
#if !ZG_IN
              ,
              unsigned int* ck0, unsigned int* ck1, unsigned int* ck2,
              unsigned int* ck3, unsigned int* cv0, unsigned int* cv1,
              unsigned int* cv2, unsigned int* cv3
#endif
              ,
              const long long* base0, const int* end0, const int* block0,
              const long long* base1, const int* end1, const int* block1,
              const long long* base2, const int* end2, const int* block2,
              const long long* base3, const int* end3, const int* block3,
              unsigned int* status) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    const unsigned int* prior_k;
    const unsigned int* prior_v;
    const long long* base_control;
    const int* end_control;
    const int* block_control;
#if !ZG_IN
    unsigned int* candidate_k;
    unsigned int* candidate_v;
#endif
    if (row == 0u) {
        prior_k = pk0; prior_v = pv0; base_control = base0; end_control = end0;
        block_control = block0;
#if !ZG_IN
        candidate_k = ck0; candidate_v = cv0;
#endif
    } else if (row == 1u) {
        prior_k = pk1; prior_v = pv1; base_control = base1; end_control = end1;
        block_control = block1;
#if !ZG_IN
        candidate_k = ck1; candidate_v = cv1;
#endif
    } else if (row == 2u) {
        prior_k = pk2; prior_v = pv2; base_control = base2; end_control = end2;
        block_control = block2;
#if !ZG_IN
        candidate_k = ck2; candidate_v = cv2;
#endif
    } else {
        prior_k = pk3; prior_v = pv3; base_control = base3; end_control = end3;
        block_control = block3;
#if !ZG_IN
        candidate_k = ck3; candidate_v = cv3;
#endif
    }
    const long long base_signed = base_control[0];
    const int end_signed = end_control[0], block_signed = block_control[0];
    if (base_signed < 0 || base_signed >= end_signed || end_signed > 512
        || block_signed < 4 || block_signed > 16 || block_signed % 4 != 0) {
        atomicOr(status, 1u);
        return;
    }
    const unsigned int base = (unsigned int)base_signed;
    const unsigned int end = (unsigned int)end_signed;
    const unsigned int block = (unsigned int)block_signed;
    const unsigned long long highest = (unsigned long long)block * 557056ull
        + 512ull + ((unsigned long long)end - 1ull) * 1024ull + 256ull + 255ull;
    if ((highest + 1ull) * 2ull > 17ull * 2ull * 544ull * 512ull * 2ull) {
        atomicOr(status, 1u);
        return;
    }
#if ZG_IN
    if (index >= base * 512u) return;
#else
    if (index >= 512u * 512u) return;
#endif
    unsigned int position = index / 512u;
    unsigned int head = (index % 512u) / 256u;
    unsigned int lane = index % 256u;
    // Actual shape[block,2,544,512], stride[557056,512,1024,1].
    unsigned long long native = (unsigned long long)block * 557056ull
        + (unsigned long long)head * 512ull + (unsigned long long)position * 1024ull + lane;
#if ZG_IN
    unsigned int k = prior_k[index], v = prior_v[index];
    if ((k & 0xffffu) != 0u || (v & 0xffffu) != 0u
        || (k & 0x7f800000u) == 0x7f800000u
        || (v & 0x7f800000u) == 0x7f800000u) atomicOr(status, 1u);
    native_cache[native] = (unsigned short)(k >> 16);
    native_cache[native + 256u] = (unsigned short)(v >> 16);
#else
    unsigned int k, v;
    if (position < end) {
        k = ((unsigned int)native_cache[native]) << 16;
        v = ((unsigned int)native_cache[native + 256u]) << 16;
        if ((k & 0x7f800000u) == 0x7f800000u
            || (v & 0x7f800000u) == 0x7f800000u) atomicOr(status, 2u);
    } else {
        // Future padding is defined canonical storage, never native input.
        k = prior_k[index]; v = prior_v[index];
    }
    candidate_k[index] = k; candidate_v[index] = v;
#endif
}