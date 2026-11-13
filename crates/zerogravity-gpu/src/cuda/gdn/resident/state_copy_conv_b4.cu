// Exact convolution state transport across four independent canonical roots.
// Same canonical [channel,frame] F32 <-> native [frame,channel] BF16 mapping
// and status bits as state_copy.cu. Row selection is physical batching only.
#if __CUDA_ARCH__ != 860
#error Closed SM86 convolution B4 transport
#endif
extern "C" __global__ __launch_bounds__(128)
void ZG_ENTRY(
    const ZG_INPUT* s0, const ZG_INPUT* s1,
    const ZG_INPUT* s2, const ZG_INPUT* s3,
    ZG_OUTPUT* d0, ZG_OUTPUT* d1,
    ZG_OUTPUT* d2, ZG_OUTPUT* d3,
    unsigned int* status) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ZG_ELEMENTS) return;
    const unsigned int row = blockIdx.y;
    const ZG_INPUT* source =
        row == 0u ? s0 : row == 1u ? s1 : row == 2u ? s2 : s3;
    ZG_OUTPUT* destination =
        row == 0u ? d0 : row == 1u ? d1 : row == 2u ? d2 : d3;

#if ZG_COPY == 0
    const unsigned int channel = index % 6144u;
    const unsigned int frame = index / 6144u;
    const unsigned int bits = source[channel * 3u + frame];
    if ((bits & 0xffffu) != 0u || (bits & 0x7f800000u) == 0x7f800000u)
        atomicOr(status, 1u);
    destination[index] = (unsigned short)(bits >> 16);
#elif ZG_COPY == 1
    const unsigned int channel = index / 3u;
    const unsigned int frame = index % 3u;
    const unsigned int bits =
        ((unsigned int)source[frame * 6144u + channel]) << 16;
    if ((bits & 0x7f800000u) == 0x7f800000u) atomicOr(status, 2u);
    destination[index] = bits;
#else
#error Conv B4 transport supports ConvIn/ConvOut only
#endif
}
