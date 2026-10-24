# Durable GDN source-investigation copy

Documentation-only preservation, 20 October 2026.
Author: `/root/spine_review`.

The two bodies below are verbatim copies of the owned source-investigation
report and its observed small-source hash appendix. `REPORT.md` and
`SOURCE_HASHES.md` in their prose refer to those original filenames; both full
bodies are included here. Their original live paths are
`.agents/runs/model-execution-gdn-vllm-resume-20261020/`.
No implementation, operator/model execution, compiled-kernel proof or runtime
qualification is established by this copy. The existing plan TODOs and current
coordinator-authored correctness-first plan remain authoritative.

---

# Pinned vLLM GDN arithmetic audit

Author: `/root/spine_review`; 2026-10-20. Parent: `numerics-and-context` and
`model-execution`. Investigation/planning only. No implementation, compilation,
tests, imports, model/operator/provider execution or weight/binary/DSO scans.
Only this owned run is written; paused SASS/SiLU work is untouched.

## Decision

Use the pinned vLLM CUDA GDN path as the target. Port its operation graph,
storage boundaries and head mapping into the existing mathematical engine.
Keep the accepted GGUF/GGML CPU schedule and the independently qualified Torch
CPU helpers explicit; their different arithmetic is intentional source behavior.

The first **source-proven postprojection mismatch** against vLLM is convolution
finalization in `crates/zerogravity/src/original_qwen_math.rs:806–813`: the old
diagnostic stores BF16 raw convolution, then applies SiLU and stores BF16 again.
vLLM's CUDA convolution applies SiLU to its FP32 accumulator and stores BF16
once. This statement assumes identical projection results and prior convolution
history. It is not evidence that convolution is the first actual trained-model
divergence: input norm/projection arithmetic can already differ.

Next concrete differences: vLLM beta stays FP32; prefill stores normalized Q/K
as BF16; normal decode keeps normalized Q/K in FP32 and scales Q before readout;
its fused gated RMS keeps normalization, gamma and SiLU multiplication FP32
until one BF16 output. The old diagnostic does not implement those boundaries.

No algebraic state-orientation/head-repeat bug was found in the inspected Rust
AR recurrence. There is no source basis for globally replacing its accepted
GGUF schedule or for treating all existing CPU numerical differences as bugs.

## Exact sources

- vLLM `0.31.0+cu129`, commit
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- Installed source root `V`:
  `/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/3d48be3a9656809e23715095eb422187fd959f80ff8ecdeab724e98dfcba213b/diff/usr/local/lib/python3.12/dist-packages`.
- Official plugin commit `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`;
  source root `P`:
  `.agents/runs/serving-and-speculation-vllm-gguf-baseline-20261020/source`.
- Original HF Python source `H`:
  `.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/modeling_qwen3_5.py`.
- Source hashes observed during this audit are in `SOURCE_HASHES.md`.

Primary revision links:

- [Qwen3.5 model/packed-role loader](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/models/qwen3_5.py).
- [GDN dispatch/cache/projections](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py).
- [Packed recurrent decode](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py).
- [Prefill post-convolution preparation](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py).
- [Chunk graph](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/third_party/flash_linear_attention/ops/chunk.py).
- [Official plugin Qwen mapping](https://github.com/vllm-project/vllm-gguf-plugin/blob/e2b8ad532b8b5ea175100202c30430c1d2b5e6a8/vllm_gguf_plugin/weights_adapter/qwen3_5.py).

## Select the actual path before comparing bits

`V/vllm/envs.py:1224–1235` defaults packed recurrent decode ON and the GDN decode
selector to `cuda`. The selector is not proof that every decode runs the C++
MTP kernel: `qwen_gdn_linear_attn.py:1849–1891` only takes that route when the
MTP metadata guard permits it, otherwise it calls `_forward_core` and the fused
norm separately. Normal decode-only, no speculation, with packed decode enabled
takes `_forward_core_decode_non_spec` (`1267–1288`, `1632–1693`): CUDA causal
convolution followed by the Triton packed recurrent kernel.

On SM86, automatic prefill selection resolves to Triton, not FlashInfer;
`qwen_gdn_linear_attn.py:93–151` restricts its automatic FlashInfer routes to
other device capability/runtime combinations. Explicit overrides, mixed
prefill/decode and speculation must be recorded separately.

## Arithmetic/storage table

Notation: `HK` key/query heads; `HV` value heads; `K` key width; `Vd` value width;
`r = HV/HK`; `R_B` BF16 round-to-nearest-even storage; `S[h,v,k]` recurrent state.
BF16 table entries assume the accepted BF16 model/cache configuration; verify
actual cache dtype separately. Norm epsilon is `config.rms_norm_eps` via
`mamba/gdn/base.py:38`, while Q/K L2 epsilon is independently fixed `1e-6`.

| Stage | Pinned vLLM CUDA | Existing Rust / original HF CPU distinction |
| --- | --- | --- |
| Source roles | Flat Qwen3.5 `Q,K,V,Z`; BA is **B,A**. `qwen3_5.py:224–227`; `qwen_gdn_linear_attn.py:632–640,703–713`. | Old diagnostic's separate QKV/Z/A/B names are correct. Do not copy Qwen3-Next interleaved packing. |
| GQA mapping | Value head `h` uses Q/K head `floor(h/r)`; `fused_recurrent.py:291`. | `repeat_qk_to_value_heads` repeats each Q/K head contiguously; this agrees. No head-order bug proved there. |
| Conv history | K4 convolution uses previous three raw projected samples plus current; cache layout can be dim-first or transposed. Source weight is `[C,1,4]`. | Canonical three-history representation is suitable; CPU original cache's K4 physical buffer needs its explicit codec. History stores raw QKV, never activated QKV. |
| Conv + SiLU | FP32 accumulator; SiLU is `acc/(1+tl.exp(-acc))`; BF16 store only after activation. `causal_conv1d.py:425–478,1000–1093`. | Old diagnostic rounds raw conv before SiLU. Authentic `H:250–289` CPU fallback also produces BF16 `F.conv1d` before separate SiLU; that CPU boundary must remain valid under its own profile. |
| Prefill Q/K | Conv BF16 → FP32 square/sum → reciprocal sqrt of `sum+1e-6` → multiply → **BF16 Q/K store**. `fused_gdn_prefill_post_conv.py:74–115,204–208,238`. | Old diagnostic's GGML grouped RMS produces F32. CPU slow HF promotes Q/K to F32 and uses its own `rsqrt` sequence; no BF16 Q/K store there. |
| Decode Q/K | Conv BF16 → FP32; `q / sqrt(sum(q*q)+1e-6)` and same for K; **remain FP32**. Packed kernel `317–320`. | GGUF uses RMS(`eps/K`) then scale `1/sqrt(K)`, with separate rounded F32 operations and a different reduction. Algebraic equivalence does not imply equal bits. |
| Alpha/decay | `a,b,dt_bias,A_log` loaded/widened to FP32. `x=a+dt_bias`; `g=-exp(A_log)*softplus(x)`; state gain `exp(g)`. | Original HF raw `A_log` and GGUF negative-A storage are different representations. Do not add a second negation or log transform. |
| Prefill softplus | If `x<=20`: positive branch `x+log(1+exp(-x))`, nonpositive branch `log(1+exp(x))`; otherwise `x`. Post-conv kernel `136–140`. | CPU GGUF helper uses `x>20→x`, `x<-20→exp(x)`, otherwise `ln(1+exp(x))`. Pinned Torch CPU softplus is another provider schedule. |
| Decode softplus | `where(x<=20, log(1+tl.exp(x)), x)`; packed kernel `327`. | Do not replace with the stable prefill expression if exact decode bits are the target. Host libm does not establish Triton exp/log parity. |
| Beta | FP32 `tl.sigmoid(b.float())`; prefill buffers explicitly F32. Packed kernel `329`, prep `145,208`. | Old diagnostic rounds beta BF16 at `original_qwen_math.rs:831–835`. That matches original `H:617` BF16 sigmoid output but not vLLM fused CUDA beta. |
| State | Canonical value-major `[sequence,HV,Vd,K]`, key contiguous; FP32 for accepted recurrent cache. Packed kernel `310–313,340–344`. | Rust accepted canonical value-major state agrees. Original HF slow state is `[sequence,HV,K,Vd]`; transpose codec is required, not relabeling. |
| Read scale | Packed decode scales Q FP32 by `K**-0.5` **before** state read reduction (`320,335`). | Accepted GGML AR helper scales completed output dot **after** reduction (`lib.rs:1503–1520`). Preserve it under its qualified source profile. |
| Core return | FP32 recurrent computation; output stored in BF16; state stored in configured recurrent dtype. | Existing diagnostic BF16 core return is appropriate as a boundary, not proof of its internal arithmetic. |
| Gated RMS | FP32 core → `rsqrt(sum(core²)/Vd+eps)` → FP32 norm × widened raw gamma × FP32 `z*sigmoid(z)` → one BF16 store. `layernorm_guard.py:119–176`. | Original CPU `H:229–234` casts normalized core BF16 **before** gamma; BF16 gamma yields BF16 weighted result, F32 gamma promotes it. Closed CPU helpers retain those distinctions. The old diagnostic rounds weighted gamma even when F32 gamma would promote, so it is not a complete CPU-profile implementation either. |
| Initial/prefill core | Chunk64 cumulative log-gates, lower triangular system, solve/WY, BF16 intermediate matrices and FP32 final state; details below. | Existing GGUF AR scalar schedule and original CPU slow F32 triangular-solve schedule are neither a bitwise substitute nor evidence of incorrect vLLM. |

### Exact normal packed-decode recurrence

The following is an operation-level transcription of
`fused_recurrent.py:291–344`, not a portable host implementation or a claim
about the final compiled instructions. Every arithmetic temporary is FP32.

```text
hq = floor(hv / (HV/HK))
q,k,v = widen_BF16(conv_silu_qkv slices)
q = q / sqrt(reduce_sum(q*q) + 1e-6)
k = k / sqrt(reduce_sum(k*k) + 1e-6)
q = q * (K**-0.5)
x = widen(a[hv]) + widen(dt_bias[hv])
sp = select(x <= 20, log(1 + exp(x)), x)
g = -exp(widen(A_log[hv])) * sp
beta = sigmoid(widen(b[hv]))
S[v,k] = S[v,k] * exp(g)
residual[v] = v[v] - reduce_sum_k(S[v,k] * k[k])
delta[v] = residual[v] * beta
S[v,k] = S[v,k] + delta[v] * k[k]
o[v] = reduce_sum_k(S[v,k] * q[k])
store_BF16(o); store_configured_cache_dtype(S)
```

`exp(g)` in this kernel uses `ops/op.py`'s alias: `tl.exp` by default,
`tldevice.fast_expf` if `FLA_USE_FAST_OPS=1`. Its coefficient `exp(A_log)`,
softplus `exp/log` and sigmoid are direct Triton operations in the kernel.
Do not silently equate these with SLEEF, glibc, MKL VML or host Rust `exp`.

**Contraction/reduction uncertainty:** Triton expressions allow compiler fusion
and determine a device reduction tree; they do not independently prove the
exact FMA count, multiply rounding, instruction selection or exp approximation
of a CubeCL rewrite. The accepted Rust GGML helper explicitly uses `mul_add`
and its SIMD/FMA dot tree. Reusing that helper would import a different schedule.
An actual pinned vLLM operator replay plus compiled-kernel inspection is needed
before asserting instruction-exact parity. No private ABI or host-expression
fit is proposed.

### Exact prefill/initial chunk boundaries

Source: `ops/chunk.py:35–99`, `chunk_scaled_dot_kkt.py:90–122`,
`wy_fast.py:75–137`, `chunk_delta_h.py:93–313,367–371`,
`chunk_o.py:101–162`. Chunk size is 64 (`ops/utils.py:31`).

1. Prepare normalized Q/K/V in BF16, beta/log-gate in FP32 as above.
2. FP32 chunk-local cumulative log-gate `G = cumsum(g)`.
3. Strict-lower system `A_ij = beta_i * dot(k_i,k_j) * exp(G_i-G_j)`;
   CUDA source retains promoted FP32 beta-scaled dot operands, outputs FP32.
4. `solve_tril` returns inverse/solved representation **in K dtype BF16**.
   Its FP32 block-merge dot precision is separately selected by
   `FLA_TRIL_PRECISION` (`solve_tril.py:22–26`): default `ieee`, CUDA also
   permits `tf32`/`tf32x3`. Record this setting; it is not implicit in BF16
   output storage. The solved representation is the inverse of `I + A_lower`.
5. Recompute `u` and `w` with BF16 operands/results: beta-scaled V and
   beta/exp-scaled K explicitly cast BF16 before dot; outputs BF16.
6. Chunk state accumulator remains FP32 `[HV,Vd,K]`. Initial state is either
   exact prior or zero according to metadata. Save each chunk's entering state
   as **BF16 h**. Compute `v_new = u - w @ h`; store BF16 v_new. Apply chunk
   decay to FP32 state; cast decay-adjusted delta BF16 before its K/delta dot
   update. Final recurrent state buffer is FP32.
7. Output uses BF16 Q/K/h/v_new dots, FP32 accumulators and causal mask;
   gated attention coefficients cast BF16 before the V dot. Scale both output
   terms, add, then BF16 output store.

This is not merely an AR loop batched in time. A clean port must preserve the
actual mixed-precision WY graph, including the triangular-solve output dtype,
or explicitly stop short of claiming prefill parity. The original CPU slow
fallback instead promotes the whole graph to F32 and uses public Torch
triangular solves (`H:337–430`); its accepted captures remain a separate target.

## Immutable GGUF source transformation is part of the profile

Plugin `weights_adapter/qwen3_5.py:43–65,188–203,282–349`:

- GGUF `ssm_a` contains negative exponentiated decay: stock loader reconstructs
  vLLM `A_log` using `torch.log(-weight)`. Re-exp later need not reproduce the
  original stored GGUF bits. Record the loaded tensor, not only raw source kind.
- Ordinary norm weights subtract one for vLLM's zero-centered loader convention;
  GDN norm gamma is excluded from that transform.
- If `HV/HK>1`, restore tiled V/Z/A/B/conv-V/dt/A_log head rows. Q/K rows stay
  in their existing order. Dense out-projection columns can be restored;
  packed quantized columns cannot be permuted as ordinary scalar columns.
  The latter use `GGUFHeadTilingLayout.input_to_gguf` at runtime.
- `quantization/layout.py:49–91` maps stored repeat-major/group-major head tiles
  to contiguous per-key-head groups and inversely maps the output activation.
  For the approved small models `HV==HK`, this particular head tiling is absent;
  for 27B grouped heads it must be explicit.
- Identical GGUF weights do not imply identical projection arithmetic:
  Zerogravity activation preparation uses Q8_0/Q8_1/Q8_K by codec, while the
  plugin's quantized path uses its own Q8_1/native kernels and cast policy.
  A first GDN comparison should feed both cores the **same captured actual
  projection operands/results**, then qualify the projection route separately.

Actual accepted vLLM original-HF loaded-cast profile previously audited by ROOT
is also distinct: A_log F32; GDN gamma BF16. Neither the CPU official-all-BF16
profile nor the restored-36-F32 profile is that loaded GPU profile. Full-model
claims must bind the chosen loaded source/cast behavior, not infer it from
checkpoint metadata.

## Minimal directly portable work and fastest gates

No implementation is authorized now. Recommended sequence inside the existing
engine once resumed:

1. **One layer0 GDN-core golden from the pinned actual vLLM route.** Capture
   postprojection QKV/Z/A/B, original raw conv history/weights, loaded A_log/dt/
   gamma, conv+SiLU, normalized Q/K, beta/g/gain, entering/proposed state, core
   and gated output. First initial prefill and then genuine cached decode,
   teachers 11→353. Compare plain versus observed full logits/state first.
   This localizes real divergence and avoids changing exponentials first when
   an earlier cast boundary is already wrong.
2. **Port the narrow conv+SiLU and post-conv preparation operation graph.**
   Same PlanBuilder/backend/source leases; FP32 activation until BF16 output;
   prefill/decode Q/K boundaries explicit. Preserve CPU helper semantics under
   `TorchCpuBf16Avx2MixedSourceV1` instead of widening that profile to CUDA.
   A distinct execution profile in the existing enum is enough; no new engine.
3. **Port normal cached recurrence and fused norm.** Canonical `[HV,Vd,K]`
   state, contiguous Q/K head replication, FP32 beta/decay, pre-scaled Q and
   final casts from the table. Original SessionAuthority/Ready/leases publish
   exactly as before. Qualify zero/nonzero prior independently; verify every
   state word and output, not just generated text.
4. **Port the actual chunk64 WY graph** with its BF16 intermediates, then qualify
   lengths 1/2/63/64/65 and continued nonzero prior. Equal mathematics alone
   cannot justify replacing it with the accepted CPU AR/slow chunk schedule.
5. Full layer/full model only after the narrow core plus projection/norm route
   agree with actual vLLM; then full vocabulary and all independently owned
   state domains, cancellation/Ready discard/unknown completion tests.

Meaningful narrow tests:

- Convolution inputs crossing a BF16 rounding boundary before SiLU: genuine
  fused vLLM output must distinguish one final store from the old two-store
  diagnostic. Keep raw state history bit-exact.
- Nonzero beta showing F32 coefficient versus prematurely BF16-rounded beta;
  negative/positive softplus and threshold20 cases in each actual route.
- Q/K zero, very small values, nontrivial sums; enforce fixed `1e-6` and the
  prefill BF16/decode FP32 distinction. Do not substitute model RMS epsilon.
- `HK=2,HV=6` head sentinels prove `[0,0,0,1,1,1]` Q/K grouping; roundtrip only
  the proper GGUF V/head rows and quantized-output activation layout.
- Signed/nonuniform state and Q expose scale-before-reduction and state-axis
  errors; checking only all-zero prior would hide both.
- Gated RMS example where BF16 gamma multiplication differs from an FP32
  product before gating, paired with the existing CPU counterexample. Expected
  vLLM bits come from the actual public kernel, not the proposed port.

License for any later source port: retain vLLM Apache-2.0 notices. Vendored FLA
files additionally identify the original MIT code and copyright
`Songlin Yang, Yu Zhang (2023–2025)`; retain that original notice/license too.
The plugin's source has the vLLM Apache-2.0 contributor notice. This audit copies
no implementation into production.

## Proof boundary

- Proven here: source paths, explicit storage dtypes, head/state axes, branch
  conditions and operation order described above.
- Not proven here: actual selected runtime branch/environment, compiled
  contraction/reduction topology, per-operator vLLM→Rust numerical parity, or
  first causal trained-model difference. No model/operator was executed.
- Prior CPU/GGUF exact receipts stay valid within their own declared schedules;
  they are not vLLM CUDA evidence. Existing owning TODOs remain authoritative.



---

<!-- Original SOURCE_HASHES.md follows verbatim. -->

# Small source files read for GDN investigation

SHA256; prefixes `V`, `P`, `H` are defined in REPORT.md. No weight/DSO/binary
payload hashed. Observed source bytes only; no new runtime-source attestation.

```text
d261726aae3ce5d52c37b868c317d9377d5e1d7c344961707acc07785a4ec4ab V/vllm/model_executor/models/qwen3_5.py
e326cba09aceca3e247d0b51e6073240d04591ca433dbb20359c720167869fa0 V/vllm/model_executor/layers/mamba/gdn/base.py
3bd2f805f0f4b821e6902ef18da8ba66ad42a7e5a633b76abe96892532293d9c V/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
cb16cc9250c4195c09d6d83b43bba607c2d749b621d8d135a920443d1577265f V/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
590717b2107abc90d72548de06b5d315229eb2775c94614f3f545df11cc2b5df V/vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py
fe6f1311014809040497aa0a623e7fa97f4b3457cdc7ee1f1f8aed21f326f755 V/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f V/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py
c105ead259e86aedc248c53e7c4cef07034d0ff12da608d80265d1dd0990803f V/vllm/third_party/flash_linear_attention/ops/chunk.py
7ae41df3755acc7e925a134d848f78bb5fc00b90069039d86403c5853323ea42 V/vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py
d1a4ff27623a938825052500afa7e2aa24fadd2fa4c9be6376c9128824f86f82 V/vllm/third_party/flash_linear_attention/ops/solve_tril.py
c9fc8259db0a8746dc04f1d799cb73333457118cf89a0bdce8dde8a80a28cd96 V/vllm/third_party/flash_linear_attention/ops/wy_fast.py
1b3ad391f939d9443c6b7adb19e57fe381bd5dccea064e8417a4f85b0e713b26 V/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
3b63b836a8cc561723c23db5a528a0b474d9cd78508d610f0b348c585a889a19 V/vllm/third_party/flash_linear_attention/ops/chunk_o.py
b4f696187a1326f9aac44f23d0bb5db814dc28ebe4fc31da000edc24365e09c6 V/vllm/third_party/flash_linear_attention/ops/layernorm_guard.py
456da84fd12411cf53c96a90dd4f78f49afb454c3d51d2c524ea8e7106fe8ae5 V/vllm/third_party/flash_linear_attention/ops/op.py
1d7101800f67b803a700fbf7489292cf1f0022b570085166ad407e6b1513dc15 V/vllm/envs.py
ed19007102c518a50e89a92fa2084c4034980f59aabd0881ce13df78df1fa2d5 P/vllm_gguf_plugin/weights_adapter/qwen3_5.py
6ffb4c25b6e862a3953ff8d00fba3c9a59febbafd791cb15ae0c44f0404ef799 P/vllm_gguf_plugin/quantization/layout.py
762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b H
cb7f6a80d1d602e26045aec3584ad53f49b9386808eae1db58bb4493137b1266 crates/zerogravity/src/original_qwen_math.rs
848a0bd39e3b98b7209499e86abae8fa96af61b7eea48208bae72290a7e65d5b crates/zerogravity/src/original_qwen_precision.rs
9a26849b304a8e9dfdd57478414c7845c6069f41a8d49c48742ec6debae22c4b crates/zerogravity-model-qwen35/src/lib.rs
a96a993f73df31ff56ec20b3704480f48c4dc03c8e62d016fa83956d88aa5820 crates/zerogravity-model-qwen35/src/forward.rs
```

