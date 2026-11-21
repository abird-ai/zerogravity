// cubecl.cuda.dynamic_shared_bytes=0
#if __CUDA_ARCH__ != 860
#error Closed SM86 retained recurrent transpose candidate
#endif
struct alignas(8) ZgRecurrentTransposeScalars {
    unsigned long long words[1];
};
extern "C" __global__ __launch_bounds__(256)
void zg_candidate_recurrent_transpose_u32(
    const unsigned int* source,
    unsigned int* destination,
    unsigned int* status,
    ZgRecurrentTransposeScalars scalars) {
    __shared__ unsigned int tile[32][33];
    const unsigned int tx = threadIdx.x & 31u;
    const unsigned int ty = threadIdx.x >> 5u;
    const unsigned int head = blockIdx.y >> 2u;
    const unsigned int tile_row = blockIdx.y & 3u;
    const unsigned int tile_col = blockIdx.x;
    const unsigned int head_base = head * 16384u;
    const unsigned int status_mask = (unsigned int)scalars.words[0];

    #pragma unroll
    for (unsigned int step = 0; step < 4; ++step) {
        const unsigned int local_row = ty + step * 8u;
        const unsigned int row = tile_row * 32u + local_row;
        const unsigned int column = tile_col * 32u + tx;
        const unsigned int bits = source[head_base + row * 128u + column];
        tile[local_row][tx] = bits;
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            atomicOr(status, status_mask);
        }
    }
    __syncthreads();

    #pragma unroll
    for (unsigned int step = 0; step < 4; ++step) {
        const unsigned int local_row = ty + step * 8u;
        const unsigned int row = tile_col * 32u + local_row;
        const unsigned int column = tile_row * 32u + tx;
        destination[head_base + row * 128u + column] = tile[tx][local_row];
    }
}
