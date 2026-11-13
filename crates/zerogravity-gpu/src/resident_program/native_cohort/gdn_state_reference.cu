// Frozen independent per-row GDN state-transport oracle.
//
// Keep this deliberately simple and separate from the optimized production
// StateTransport implementation. Grouped transport qualification compares its
// physical bytes/status against these original scalar-index equations.
#if __CUDA_ARCH__ != 860
#error Closed SM86 state transport reference
#endif
extern "C" __global__ __launch_bounds__(128)
void ZG_ENTRY(const ZG_INPUT* source, ZG_OUTPUT* destination, unsigned int* status) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ZG_ELEMENTS) return;
#if ZG_COPY == 0
    unsigned int channel = index % 6144u;
    unsigned int frame = index / 6144u;
    unsigned int bits = source[channel * 3u + frame];
    if ((bits & 0xffffu) != 0u || (bits & 0x7f800000u) == 0x7f800000u)
        atomicOr(status, 1u);
    destination[index] = (unsigned short)(bits >> 16);
#elif ZG_COPY == 1
    unsigned int channel = index / 3u;
    unsigned int frame = index % 3u;
    unsigned int bits = ((unsigned int)source[frame * 6144u + channel]) << 16;
    if ((bits & 0x7f800000u) == 0x7f800000u) atomicOr(status, 2u);
    destination[index] = bits;
#else
    unsigned int head = index / 16384u;
    unsigned int major = (index / 128u) % 128u;
    unsigned int minor = index % 128u;
    unsigned int bits = source[head * 16384u + minor * 128u + major];
    if ((bits & 0x7f800000u) == 0x7f800000u) atomicOr(status, 1u << ZG_COPY);
    destination[index] = bits;
#endif
}
