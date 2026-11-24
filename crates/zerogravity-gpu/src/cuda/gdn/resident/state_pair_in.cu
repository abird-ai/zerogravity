// Fused canonical GDN state import. No model/root/publication authority.
#if __CUDA_ARCH__ != 860
#error Closed SM86 GDN state import
#endif
extern "C" __global__ __launch_bounds__(128)
void zg_gdn_state_in(
    const unsigned int* conv_source,
    const unsigned int* recurrent_source,
    unsigned short* conv_destination,
    unsigned int* recurrent_destination,
    unsigned int* status) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < ZG_RECURRENT_ELEMENTS) {
        const unsigned int head = index / 16384u;
        const unsigned int major = (index / 128u) % 128u;
        const unsigned int minor = index % 128u;
        const unsigned int bits =
            recurrent_source[head * 16384u + minor * 128u + major];
        if ((bits & 0x7f800000u) == 0x7f800000u)
            atomicOr(status, 4u);
        recurrent_destination[index] = bits;
    }

    if (index < ZG_CONV_ELEMENTS) {
        const unsigned int channel = index % 6144u;
        const unsigned int frame = index / 6144u;
        const unsigned int bits = conv_source[channel * 3u + frame];
        if ((bits & 0xffffu) != 0u || (bits & 0x7f800000u) == 0x7f800000u)
            atomicOr(status, 1u);
        conv_destination[index] = (unsigned short)(bits >> 16);
    }
}
