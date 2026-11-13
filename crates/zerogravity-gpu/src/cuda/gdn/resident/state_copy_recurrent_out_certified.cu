// Exact recurrent H,V,K -> canonical H,K,V F32 transport plus finiteness
// certification of one independent final BF16 payload. The recurrent mapping
// and status bit are identical to state_copy.cu RecurrentOut; certification
// only ORs the shared nonfinite-final-output bit 2.
#if __CUDA_ARCH__ != 860
#error Closed SM86 recurrent certified transport
#endif
extern "C" __global__ __launch_bounds__(128)
void zg_vllm_root_h_out_certified(
    const unsigned int* source,
    unsigned int* destination,
    const unsigned short* certificate,
    unsigned int* status) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ZG_ELEMENTS) return;

    const unsigned int head = index / 16384u;
    const unsigned int major = (index / 128u) % 128u;
    const unsigned int minor = index % 128u;
    const unsigned int bits =
        source[head * 16384u + minor * 128u + major];
    if ((bits & 0x7f800000u) == 0x7f800000u)
        atomicOr(status, 1u << ZG_COPY);
    destination[index] = bits;

    if (index < ZG_CERTIFY_ELEMENTS) {
        const unsigned int bf16 = (unsigned int)certificate[index];
        if ((bf16 & 0x7f80u) == 0x7f80u)
            atomicOr(status, 2u);
    }
}
