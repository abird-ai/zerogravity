// U32/BF16 transport only. K/V are halves of each native per-head512 lane.
#if __CUDA_ARCH__ != 860
#error Closed SM86 original KV transport
#endif
extern "C" __global__ __launch_bounds__(128)
void ZG_ENTRY(const unsigned int* prior_k, const unsigned int* prior_v,
              unsigned short* native_cache,
#if !ZG_IN
              unsigned int* candidate_k, unsigned int* candidate_v,
#endif
#if defined(ZG_DYNAMIC_CONTROLS)
              const long long* base_control, const int* end_control,
              const int* block_control,
#endif
              unsigned int* status) {
#if defined(ZG_DYNAMIC_CONTROLS)
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
#else
    const unsigned int base = ZG_BASE, end = ZG_END, block = ZG_BLOCK;
#endif
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
#if ZG_IN
    if (index >= base * 512u) return;
#else
    if (index >= 512u * 512u) return;
#endif
    unsigned int position = index / 512u;
    unsigned int head = (index % 512u) / 256u;
    unsigned int lane = index % 256u;
    // Actual shape[block,2,544,512], stride[557056,512,1024,1].
#if defined(ZG_DYNAMIC_CONTROLS)
    unsigned long long native = (unsigned long long)block * 557056ull
        + (unsigned long long)head * 512ull + (unsigned long long)position * 1024ull + lane;
#else
    unsigned int native = block * 557056u
        + head * 512u + position * 1024u + lane;
#endif
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
