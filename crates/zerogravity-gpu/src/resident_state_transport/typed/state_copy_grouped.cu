// Exact per-row state transport, dispatched by blockIdx.y. ONE definition per
// equation family: each grouped lane runs the byte-identical per-row equations
// of the original per-row resident/state_copy.cu (conv widened-BF16 high16,
// low16==0 check, status bits 1/2; exact U32 H transpose with bits 4/8). Only
// the row selection is added; no scalar equation, status bit or footprint
// mapping changed.
#if __CUDA_ARCH__ != 860
#error Closed SM86 state transport
#endif
extern "C" __global__ __launch_bounds__(128)
void ZG_ENTRY(const ZG_INPUT* s0, const ZG_INPUT* s1, const ZG_INPUT* s2,
              const ZG_INPUT* s3, ZG_OUTPUT* d0, ZG_OUTPUT* d1, ZG_OUTPUT* d2,
              ZG_OUTPUT* d3, unsigned int* status) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ZG_ELEMENTS) return;
    unsigned int row = blockIdx.y;
    const ZG_INPUT* source;
    ZG_OUTPUT* destination;
    if (row == 0u) { source = s0; destination = d0; }
    else if (row == 1u) { source = s1; destination = d1; }
    else if (row == 2u) { source = s2; destination = d2; }
    else { source = s3; destination = d3; }
#if ZG_COPY == 0
    // Native [frame,channel] BF16 <- canonical [channel,frame] exact-widen F32.
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
    // H,K,V and H,V,K share dimensions, never their element ordering.
    unsigned int head = index / 16384u;
    unsigned int major = (index / 128u) % 128u;
    unsigned int minor = index % 128u;
    unsigned int bits = source[head * 16384u + minor * 128u + major];
    if ((bits & 0x7f800000u) == 0x7f800000u) atomicOr(status, 1u << ZG_COPY);
    destination[index] = bits;
#endif
}