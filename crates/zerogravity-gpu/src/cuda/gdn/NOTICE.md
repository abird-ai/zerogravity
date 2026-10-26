# Original instruction assets

Private qualification assets; no model/backend eligibility.

- vLLM0.31.0, commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`,
  Triton3.7.1, SM86. Original HF Qwen3.5-.8B source revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Original PTX and actual five phases captured unchanged by ROOT's validated
  `root-layer-oracle-2`; CUDA inline wrappers emitted by ROOT's offline
  `render_ptx.py`, output `ptx-port-validated-5` (balanced entry scope;
  trailing debug sections excluded; declared registers renamed into a private
  namespace to preserve compiler-selected pointer operands).
- Scalar constants are fixed in each instruction asset. `pins.json` records
  original PTX/source hashes, scalar bits and launch geometry; runtime Rust
  uses closed phase/stage choices, never recorded addresses or expected data.
- Apache-2.0: copyright contributors to the vLLM project. Causal convolution
  additionally credits Copyright (c)2024, Tri Dao; adapted from
  https://github.com/Dao-AILab/causal-conv1d.
- Bundled flash-linear-attention kernels preserve Copyright (c)2023–2025,
  Songlin Yang, Yu Zhang and the original MIT notice. Full copied license
  texts: `LICENSE-APACHE-2.0`, `LICENSE-FLA-MIT`.
- Original source paths: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`
  and `vllm/third_party/flash_linear_attention/ops/{cumsum,chunk_scaled_dot_kkt,
  solve_tril,wy_fast,chunk_delta_h,chunk_o,layernorm_guard,fused_recurrent,
  fused_gdn_prefill_post_conv}.py`. Generated debug records preserve the
  original source paths.
