#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include "flash.h"

extern "C" int zg_fa2_bf16_hdim256_causal(
    void* q, void* k, void* v, void* out, void* lse,
    int rows, int base, int capacity, int q_heads, int kv_heads,
    float scale, cudaStream_t stream) {
  if (!q || !k || !v || !out || !lse ||
      rows <= 0 || base < 0 || capacity <= 0 ||
      base + rows > capacity || q_heads <= 0 || kv_heads <= 0 ||
      q_heads % kv_heads != 0 || !(scale > 0.0f)) {
    return (int)cudaErrorInvalidValue;
  }

  flash::Flash_fwd_params p{};
  constexpr int d = 256;
  const int seqlen_k = base + rows;
  const auto round128 = [](int x) { return (x + 127) / 128 * 128; };

  p.is_bf16 = true;
  p.q_ptr = q;
  p.k_ptr = k;
  p.v_ptr = v;
  p.o_ptr = out;

  p.q_row_stride = (int64_t)q_heads * d;
  p.k_row_stride = (int64_t)kv_heads * d;
  p.v_row_stride = (int64_t)kv_heads * d;
  p.o_row_stride = (int64_t)q_heads * d;
  p.q_head_stride = d;
  p.k_head_stride = d;
  p.v_head_stride = d;
  p.o_head_stride = d;

  p.q_batch_stride = (int64_t)rows * q_heads * d;
  p.k_batch_stride = (int64_t)capacity * kv_heads * d;
  p.v_batch_stride = (int64_t)capacity * kv_heads * d;
  p.o_batch_stride = (int64_t)rows * q_heads * d;

  p.softmax_lse_ptr = lse;
  p.b = 1;
  p.h = q_heads;
  p.h_k = kv_heads;
  p.h_h_k_ratio = q_heads / kv_heads;
  p.seqlen_q = rows;
  p.seqlen_k = seqlen_k;
  p.seqlen_q_rounded = round128(rows);
  p.seqlen_k_rounded = round128(seqlen_k);
  p.d = d;
  p.d_rounded = d;
  p.total_q = rows;

  p.softcap = 0.0f;
  p.scale_softmax = scale;
  p.scale_softmax_log2 = scale * 1.4426950408889634f;
  p.p_dropout = 1.0f;
  p.p_dropout_in_uint8_t = 255;
  p.rp_dropout = 1.0f;
  p.scale_softmax_rp_dropout = scale;

  p.is_causal = true;
  p.window_size_left = -1;
  p.window_size_right = 0;
  p.is_seqlens_k_cumulative = true;
  p.unpadded_lse = false;
  p.seqlenq_ngroups_swapped = false;
  p.num_splits = 0;

  try {
    flash::run_mha_fwd_<cutlass::bfloat16_t, 256, true>(p, stream);
    return (int)cudaPeekAtLastError();
  } catch (...) {
    return (int)cudaErrorUnknown;
  }
}
