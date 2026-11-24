#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/variants.cuh>

using namespace flashinfer;

struct ZgFlashInferDecodeParams {
  using DTypeQ = nv_bfloat16;
  using DTypeKV = nv_bfloat16;
  using DTypeO = nv_bfloat16;
  using IdType = int32_t;

  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  DTypeO* o;
  float* lse;
  float* maybe_alibi_slopes;
  double logits_soft_cap;
  double sm_scale;
  double rope_rcp_scale;
  double rope_rcp_theta;

  uint32_t kv_len;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t kv_stride_n;
  uint32_t kv_stride_h;
  int32_t window_left;
  uint32_t kv_chunk_size;

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t) const { return 1; }
  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t) const { return kv_len; }
};

extern "C" int zg_flashinfer_bf16_hdim256_decode(
    void* q, void* k, void* v, void* out, void* tmp,
    int kv_len, int num_qo_heads, int num_kv_heads,
    float sm_scale, cudaStream_t stream) {
  if (!q || !k || !v || !out || !tmp || !stream ||
      kv_len <= 0 || num_qo_heads <= 0 || num_kv_heads <= 0 ||
      num_qo_heads % num_kv_heads != 0 || !(sm_scale > 0.0f)) {
    return (int)cudaErrorInvalidValue;
  }

  ZgFlashInferDecodeParams p{};
  p.q = static_cast<nv_bfloat16*>(q);
  p.k = static_cast<nv_bfloat16*>(k);
  p.v = static_cast<nv_bfloat16*>(v);
  p.o = static_cast<nv_bfloat16*>(out);
  p.lse = nullptr;
  p.maybe_alibi_slopes = nullptr;
  p.logits_soft_cap = 0.0;
  p.sm_scale = sm_scale;
  p.rope_rcp_scale = 1.0;
  p.rope_rcp_theta = 1.0 / 10000.0;
  p.kv_len = static_cast<uint32_t>(kv_len);
  p.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
  p.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
  p.q_stride_n = static_cast<uint32_t>(num_qo_heads * 256);
  p.q_stride_h = 256;
  p.kv_stride_n = static_cast<uint32_t>(num_kv_heads * 256);
  p.kv_stride_h = 256;
  p.window_left = -1;
  p.kv_chunk_size = 0;

  using Variant = DefaultAttention<false, false, false, false>;
  try {
    auto status = SingleDecodeWithKVCacheDispatched<
        256, PosEncodingMode::kNone, Variant>(
        p, static_cast<nv_bfloat16*>(tmp), stream);
    return (int)status;
  } catch (...) {
    return (int)cudaErrorUnknown;
  }
}
