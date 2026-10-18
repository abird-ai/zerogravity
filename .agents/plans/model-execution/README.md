Author: Project owner
Co-authors: GPT-6

# Zerogravity: source-real model execution (CPU-first)

## Active order

1. **Qwen3.5 2B**, with 0.8B as the exact trained regression canary:
   preserve existing resident GGUF CPU/GPU StateRoot and complete logits,
   then optimize measured end-to-end prefill/decode on qualified hardware.
2. **Qwen3.8 27B:** original pinned source, immutable large shards, actual
   model tensor/head/axis binding and tokenizer, complete trained CPU
   logits/StateRoot parity, then GPU specialization and SSD streaming.
3. **Gemma4 E2B → 26B-A4B → 31B:** original learned PLE/MoE/dense
   geometry; source-accurate CPU/quant math first, then GPU physical gates.
4. **Qwen3.8 Flash-Next:** original GDN/QSA/MoE/Engram/MTP as actually
   source-qualified, real Q2/~3.5-bpw/Q4 quants; speculative suffix/n-gram
   decoding is a distinct algorithm from learned n-gram embeddings.

YAML single-process/multi-model pipeline scaffolds are already complete.
Live routing, audio/image/video, context providers, trained decisions
and non-critical cloud provider parity are deferred to their owners.

## LS-B01–B05: verified owned sharded-source transaction

Implementation: `crates/zerogravity-artifacts/src/verified_shards.rs`
and the original `SourceWeights` allocation/issuer in
`source_weights.rs`.

**Input:** one original index JSON plus the independently expected
whole-index SHA256, an exact set of already-opened Safetensors Artifacts
with caller-declared basename, whole-file length and independently
expected SHA256 per shard; `SourceOpenLimits` and the existing
caller-registered `TransferAdmission` for the actual host RAM domain.

**Prepare:** same bounded index/header parsers, exact indexed name→shard
bijection, dtype/shape/offset/size and payload-byte total agreement,
checked whole-file and metadata budgets. Reserve the entire original
file bytes (headers included) plus conservative metadata under ONE
physical ResourceArbiter lease *before source content allocation*.

**Materialize:** copy from the exact originally opened descriptors,
verify their whole-file SHA256 and exact lengths, reparse the copied
immutable bytes, re-check header geometry used during preflight,
and independently re-check owned payload sum against the index.
The already-existing `SourceIdentity::for_new_allocation` is called
only after every check. One `SourceWeights` owns all shard Vecs
plus source lease; its `TensorSource` borrows are tied to its
actual source lifetime. Caller-supplied foreign/altered TensorInfo
cannot select a different shard via offset/name/shape substitution.
Partial failures release admission without publishing any source,
cache entry or model StateRoot.

**Compatibility:** the original GGUF `SourceWeights::open` keeps
its **4GiB default ceiling**, CPU/GPU model code and source identity
semantics. Multiple model instances/Arc owners must share the
existing process-wide ResourceArbiter, not invent independent memory
domains. This source format constructor does **not** implement
original BF16/F16 mathematical schedules or convert Hugging Face
tensor axes to GGUF axes.

**Tested only with tiny owned shards and sparse metadata:** F32/F16/
BF16 reference vectors, exact headers and digests, wrong/incomplete
index, missing/duplicate/extra/misassigned shards, concurrent
admitted allocations, mutation/rename of original descriptors,
same-size different geometry and foreign tensor metadata refusal,
and sparse >5GiB header preflight refusal without reading the data.
A deliberate larger source limit still needs real physical RAM
admission. Neither synthetic 5GiB refusal nor two tiny shard loads
qualifies *real* 27B trained inference or SSD performance.

## Remaining execution gates

- Actual authenticated model/index/tokenizer sources on an admitted
  capable host, original Qwen3.8 head/permutation/numerical binding,
  GQA 16→48, all vocabulary and StateRoot values for trained weights.
- Source-real large-shard allocation/lifetime and actual >4GiB host
  memory admission before claiming large-resident execution. This first
  owned copy is **not** SSD streaming.
- An owned immutable SSD extent and streaming lease abstraction with
  bounded I/O, readiness, errors, cancellation, and **stable** source
  buffer borrows. Evictable tensor caches cannot implement the original
  borrowed `TensorSource` interface by returning movable slices.
- Measured Qwen3.5 2B CPU/GPU speed, accuracy, prefill and decode,
  source/StateRoot, per-session rollback and physical readiness,
  with no unauthorized GPU on a held device.
- Gemma/Flash original trained correctness and quant/SSD/GPU gates.
  No new model SDK/download or runtime process/serve path is implied.

See the [model execution TODO](TODO.md),
[immutable shard specification](../IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md),
[SSD and memory](../memory-and-streaming/TODO.md),
[CPU/GPU optimization](../cpu-gpu-optimization/TODO.md), and
[model acquisition](../model-acquisition/TODO.md).

## ME-P0 continuation — original Qwen3.8 text binding and CPU value path

The original Qwen3.8 **ordinary text** tensor contract is now bound
directly to the immutable `SourceWeights` from the accepted original
index/shard transaction. This is a source/value preparation stage, NOT
a complete original-model token decoder.

- `crates/zerogravity-artifacts/src/model/qwen_owned.rs`: borrow the
  original source with an independently expected config JSON SHA256
  and exact verified original indexed-source SHA256. Reuse the existing
  `QwenTextConfig` source-order tensor definitions. Required name,
  output-first shape and F32/F16/BF16 encoding must all agree with
  the immutable verified bytes. Missing/extra ordinary text tensors,
  untied output head mismatches and wrong source hashes refuse.
  Vision and MTP tensors remain separately unqualified.
- `crates/zerogravity-cpu/src/original_hf.rs`: one
  **[output,input]** source matrix adapter to the existing Rust
  source-typed CPU `MatVecRegion` and optional completed accelerator
  provider. Do not transpose/copy learned matrices or create a
  second mathematical backend. Bound the original rows, finite
  inputs and trusted physical completed outputs before dispatch.
- `crates/zerogravity/src/original_qwen_cpu.rs`:
  `QwenOriginalCpu::bind` exposes source-bound `embed`, `vector` and
  `project` cold value primitives using that one owner and CPU math.
  No new model SourceIdentity, session, token loop or resource issuer.

Small synthetic **complete original hybrid** source tests and original
F32/F16/BF16 projection/embedding integration tests prove the path;
wrong tensor shapes, missing output heads, added learned-text roles,
config/index digest disagreement and incompatible providers refuse.
These tests DO NOT qualify trained 64-layer Qwen3.8-27B inference.

**Remaining:** real learned weights/official tokenizer and reference
framework, original Q/K→V grouped replications, complete GDN/attention/
MLP forward schedule and BF16 rounding, full 248320-vocabulary logits,
session/StateRoot, real >4GiB resident loading and SSD streaming. The
existing 0.8B/2B trained GGUF oracles must remain exact.

## ME-P0 analytic original Qwen3.8 one-layer proposals (CPU)

**Scope:** source-bound, *tiny analytical FP32* proposals of GDN,
full attention and SwiGLU FFN, using the original verified HF tensor
names and the existing Qwen3.5 mathematical operators. This is NOT
a second model executor, trained Qwen3.8 output, production StateRoot,
model-loading CLI, or an HF BF16-equivalent source reference yet.

Location: `crates/zerogravity/src/original_qwen_math.rs`, opt-in
`cpu+cache+model-qwen35` only; default GGUF source/session path
is unchanged. `OriginalQwenLayerMath` borrows the previously
verified `QwenOriginalCpu` source and exercises:

- **Zero-centered RMSNorm:** original ordinary residual and Q/K
  RMSNorm weights are `1 + checkpoint_weight`. The GDN output's
  separately learned gated norm uses its raw weight. The same
  qualified `source_rms_normalize` and
  `source_grouped_rms_normalize` CPU math is reused.
- **Full attention:** HF `q_proj` interleaves queries and sigmoid
  gates **per head**. Split that layout before independent per-head
  Q/K normalization. Use source-derived head count, partial RoPE
  angle and only the configured rotary width. The existing
  `source_rope_kv`, `source_attention_qk`,
  `source_attention_softmax`, `source_attention_value`,
  `source_binary` and one original matrix projection produce an
  unpublished candidate. Original 24Q/4KV mapping repeats each
  KV head for six queries. An `OriginalFullPrefix` holds immutable
  Arc KV slabs, original SourceIdentity and exact layer ID; its
  constructor is available only from the authenticated source-bound
  numerics adapter. A candidate cannot borrow a different model's
  or layer's KV history; it never mutates a previously held prefix.
- **GDN:** use the original fused QKV, z, a and b projections,
  source channel-major four-tap convolution, SiLU, then split
  `[Q_16, K_16, V_48]` in **original** head ordering.
  Repeat each Q/K head three times to align with the 48 V heads.
  Perform independent head L2 normalization; transform stored
  `A_log` to `-exp(A_log)`, add source `dt_bias` in the
  existing `source_gdn_coefficients` and keep beta's sigmoid.
  Reuse the existing `source_gdn_recurrent_step` for value-major
  private-state updates, original GDN gated RMSNorm and the
  owned source out projection. `OriginalGdnPrefix` holds the
  private candidate convolution/recurrent state, source identity,
  layer and proposed position; only a source-bound adapter may mint
  or advance it. This stage explicitly supports the official
  **equal K/V widths and 4-tap** Qwen3.8 contract; unsupported
  geometries refuse instead of changing the kernel.
- **FFN:** source ordinary post-attention RMSNorm, projected
  gate/up, SiLU×up, projected down and candidate residual with
  finite output validation; no alternate token/layer loop.

All of these are *analytical FP32 source-operator diagnostics*.
They reuse the already qualified GGUF-style FP32 numerical
reduction order; **HF's explicit BF16 intermediate casts, FLA
kernel accumulation rules, variant-specific RoPE accuracy and
trained state/logits are not yet qualified**. No output may be
published through SessionAuthority or an API until those
independent gates pass. No GPU kernel is launched.

Bounded independent source-free tests cover 16→48 Q/K repeat with
signed-zero bit preservation, 24Q/4KV six-query-per-key mapping,
per-head interleaved gate split, ordinary `1+gamma`,
`A_log=0`/`alpha+dt=0` yielding gain 0.5 and beta 0.5,
one-token analytical full attention, two deterministic GDN
steps with unchanged original state, source/layer foreign-prefix
refusal, partial-RoPE tail invariance, SwiGLU residual and
malformed/nonfinite/capacity negatives. The original source
owner/leased tensor memory is not duplicated.

**Next:** wire the same operator plan into the *existing*
TaskDriver/SessionAuthority and a pinned HF BF16 numerical reference
(including round/cast positions, GDN/attention state, output heads).
Then validate original trained full logits/StateRoot, source-real
27B host capacity and SSD streaming. Do not infer those capabilities
from tiny analytical tests.

Upstream specification used for original tensor semantics:
[Hugging Face Qwen3.5 modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
