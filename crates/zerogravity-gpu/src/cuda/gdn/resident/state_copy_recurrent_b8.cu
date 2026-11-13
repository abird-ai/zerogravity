// Exact recurrent F32 state transport across eight independent canonical roots.
// Same H,K,V <-> H,V,K U32 transpose and status bits as state_copy.cu.
// Row selection is physical batching only; no model/session/root policy enters here.
#if __CUDA_ARCH__ != 860
#error Closed SM86 recurrent B8 transport
#endif
extern "C" __global__ __launch_bounds__(128)
void ZG_ENTRY(
    const unsigned int* s0, const unsigned int* s1,
    const unsigned int* s2, const unsigned int* s3,
    const unsigned int* s4, const unsigned int* s5,
    const unsigned int* s6, const unsigned int* s7,
    unsigned int* d0, unsigned int* d1,
    unsigned int* d2, unsigned int* d3,
    unsigned int* d4, unsigned int* d5,
    unsigned int* d6, unsigned int* d7,
    unsigned int* status) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ZG_ELEMENTS) return;
    unsigned int row = blockIdx.y;
    const unsigned int* source;
    unsigned int* destination;
    if (row == 0u) { source = s0; destination = d0; }
    else if (row == 1u) { source = s1; destination = d1; }
    else if (row == 2u) { source = s2; destination = d2; }
    else if (row == 3u) { source = s3; destination = d3; }
    else if (row == 4u) { source = s4; destination = d4; }
    else if (row == 5u) { source = s5; destination = d5; }
    else if (row == 6u) { source = s6; destination = d6; }
    else { source = s7; destination = d7; }

    unsigned int head = index / 16384u;
    unsigned int major = (index / 128u) % 128u;
    unsigned int minor = index % 128u;
    unsigned int bits = source[head * 16384u + minor * 128u + major];
    if ((bits & 0x7f800000u) == 0x7f800000u) atomicOr(status, 1u << ZG_COPY);
    destination[index] = bits;
}
