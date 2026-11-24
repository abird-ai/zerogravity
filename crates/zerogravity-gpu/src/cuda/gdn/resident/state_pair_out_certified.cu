// Fused native GDN state export plus BF16 output certification.
// No model/root/publication authority.
#if __CUDA_ARCH__ != 860
#error Closed SM86 GDN state export
#endif
extern "C" __global__ __launch_bounds__(128)
void zg_gdn_state_out_certified(
    const unsigned short* conv_source,
    const unsigned int* recurrent_source,
    unsigned int* conv_destination,
    unsigned int* recurrent_destination,
    const unsigned short* certificate0,
    const unsigned short* certificate1,
    const unsigned short* certificate2,
    unsigned int* status) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < ZG_RECURRENT_ELEMENTS) {
        const unsigned int head = index / 16384u;
        const unsigned int major = (index / 128u) % 128u;
        const unsigned int minor = index % 128u;
        const unsigned int bits =
            recurrent_source[head * 16384u + minor * 128u + major];
        if ((bits & 0x7f800000u) == 0x7f800000u)
            atomicOr(status, 8u);
        recurrent_destination[index] = bits;
    }

    if (index < ZG_CONV_ELEMENTS) {
        const unsigned int channel = index / 3u;
        const unsigned int frame = index % 3u;
        const unsigned int bits =
            ((unsigned int)conv_source[frame * 6144u + channel]) << 16;
        if ((bits & 0x7f800000u) == 0x7f800000u)
            atomicOr(status, 2u);
        conv_destination[index] = bits;
    }

    if (index < ZG_CERTIFICATE0_ELEMENTS &&
        (((unsigned int)certificate0[index] & 0x7f80u) == 0x7f80u))
        atomicOr(status, 2u);
    if (index < ZG_CERTIFICATE1_ELEMENTS &&
        (((unsigned int)certificate1[index] & 0x7f80u) == 0x7f80u))
        atomicOr(status, 2u);
    if (index < ZG_CERTIFICATE2_ELEMENTS &&
        (((unsigned int)certificate2[index] & 0x7f80u) == 0x7f80u))
        atomicOr(status, 2u);
}
