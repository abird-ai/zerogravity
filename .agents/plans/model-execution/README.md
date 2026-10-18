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

## ME-P0 — original Qwen analytical layers and the existing TaskCoordinator

The original `zerogravity-core::state::TaskCoordinator` now has an
**opt-in immutable SourceIdentity binding** and a unique per-coordinator
process-local ID. Its prior generic `new`/`publish` API is preserved.
Source-bound coordinators require `publish_source_bound` and reject
unbound or foreign source identities: this extends the *existing*
publication authority without introducing a second scheduler/session.

The stateless `crates/zerogravity/src/original_qwen_state.rs` maps the
existing `ForwardPlan` state-slot vocabulary (`ConvDomain`,
`RecurrentDomain`, `KvDomain`) to fixed, versioned diagnostic state
domains for **one** original Qwen analytical layer at a time:

- Validate source, coordinator ID, layer, epoch, causal root/frontier
  and exact required domain schemas before reading a proposed step.
- Reconstruct finite, complete CPU values from the existing immutable
  `StateRoot` bytes, or an authentic zero-initialized first state.
  Run the original source-bound GDN/full-attention proposal using
  the already-qualified Qwen3.5 CPU mathematical operators.
- Stage an entire GDN convolution plus recurrence, or complete causal
  attention KV, as private `PreparedDomain` bytes. Reject
  nonfinite/malformed data, and bound diagnostic operation to 8K
  tokens and 64 MiB per-domain; check the *next* full KV budget
  before allocating another history-sized buffer.
- Commit exactly once using the original `TaskCoordinator` atomic
  publisher, after all validation. Abandonment, competing root
  attempts, stale epochs and foreign session/source identities cannot
  advance the state. Prior COW immutable roots remain unchanged.

Prepared proposals borrow the original verified Qwen source owner
until commit/abandonment, so their model weight ownership and retained
host ResourceArbiter lease cannot expire underneath publication.

These are **one-layer FP32 analytical diagnostics**, not a full
multi-layer token loop, compiled original 64-layer forward program,
durable portable StateRoot, BF16 reference, device-ready completion
or trained Qwen3.8 inference. Physical mutable state admission and
real cancellation are distinct production execution gates.
The **next** execution step is to integrate original HF operators into
the sole `forward_token`/`MathematicalBackend` and
`SessionAuthority` state owner, and prove trained BF16/FLA GDN
numerics and original full-vocabulary/StateRoot parity before real
Qwen3.8 27B/Gemma/SSD serving.

## ME-P0 — one full original Qwen FP32 diagnostic token, shared schedule

The existing Qwen3.5 `forward_token` and the original indexed-HF
source call the **same** `qwen35::run_layer_sequence` traversal
in `crates/zerogravity-model-qwen35/src/forward.rs`. That single
bounded, monomorphized iterator handles layer order and early
failure: GGUF `MathematicalBackend` owns its accepted operator
schedule and original HF delegates to source-bound GDN/full-
attention/FFN CPU operators for its profile-specific mathematics.
There is no separate model-load manager or public inference loop.

`OriginalQwenStateBridge::coordinator_for_model` registers all
typed `ConvDomain`, `RecurrentDomain` and `KvDomain` slots in
one existing source-bound `TaskCoordinator`. The four-layer
analytical source fixture has three GDN layers, one full-attention
layer and seven distinct immutable state domains. The explicitly
named `prepare_diagnostic_token_f32`:

1. Validates every domain schema and model source identity before
   numerical work; reads only one immutable root and one source-
   qualified embedding.
2. Uses the common Qwen layer traversal to stage the original
   analytical GDN, attention and FFN values and complete causal
   state for each layer, with the accepted FP32 residual operator.
3. Enforces 8K tokens, 128 layers and 128MiB of total diagnostic
   state, in addition to 64MiB per state domain. A missing or
   malformed late-layer state aborts the entire proposal.
4. Applies the source-bound final ordinary RMS and untied
   `lm_head.weight` projection; requires complete finite logits
   before returning an `OriginalPreparedToken`.
5. Commits all state domains and logits through the existing
   TaskCoordinator only after validation. Discarded, competing,
   stale or foreign proposed tokens cannot advance the root.

Tiny original Safetensors integration tests use a four-layer
hybrid source with all-one F32/F16/BF16 tensor values. The final
synthetic analytical logit is approximately **8**: four positive
uniform hidden lanes normalize to ~2 under original zero-centered
`(1+checkpoint_weight)` RMSNorm (checkpoint weight 1), and each
untied output-head row sums them. This independently derived
oracle checks more than simply uniform vocabulary results. The
complete forward's first GDN state domain matches the separate
one-layer bridge exactly. The shared GGUF loop test also proves
layer-order preservation and fail-fast behavior.

**Qualification boundary:** This is an FP32 **synthetic
whole-model diagnostic**, not source-real trained 27B execution,
not HF BF16/FLA numerical equivalence, not full per-op
`MathematicalBackend` integration for original sources and not
production `SessionAuthority`/GPU completion. We preserve the
GGUF trained source regression independently.

### Remaining production ownership gates

- Qualify original HF/FLA intermediate BF16 casting, accumulation
  and exact reference GDN/attention/FFN operators.
- Compose source profiles in the existing full per-op
  `MathematicalBackend` forward program and original
  `SessionAuthority` without a second token/state runtime.
- Verify trained checkpoint/official tokenizer, complete logits
  and StateRoot, physical memory and device completion.
- Admit original 27B host capacity and independently tested
  SSD streaming before user-facing model execution.

## ME-P0 — original BF16 source-dtype diagnostic (not trained parity)

The original Qwen math now offers two explicit source-bound modes:
`OriginalQwenLayerMath::new` (unchanged FP32 diagnostics), and
`bf16_diagnostic` (finite IEEE BF16 round-to-nearest-even after
source-identified dtype boundaries). The existing
`MathematicalBackend` layer traversal, `SourceIdentity`,
`TensorSource`, CPU kernels and `TaskCoordinator` remain shared.
There is **no BF16-specific token loop, model loader or state issuer**.

Source boundary operator: `crates/zerogravity/src/original_qwen_precision.rs`.

- IEEE BF16 quantization: finite f32 → nearest BF16 with ties-to-even,
  preserving signed zero. Nonfinite inputs/rounded overflow refuse.
- Ordinary zero-centered RMSNorm: promote BF16 input to FP32,
  normalize and multiply `1 + gamma` in FP32, then round once
  to input dtype. Independent Q/K head rows share this same operator.
- Source projection/output and SwiGLU: BF16 result per projection,
  SiLU, product, down-projection and residual. Untied vocabulary
  head also rounds after projection.
- Full attention: per-head interleaved Q/gate and K/V projections,
  headwise norms, source RoPE, FP32 dot with BF16 matmul-output cast,
  separate BF16 scale cast, FP32 softmax with BF16 return, value/GQA,
  sigmoid gate and source output projection. Exact actual GPU
  GEMM accumulation/RoPE cast scheduling remains unqualified.
- GDN: original BF16 QKV/z/a/b projections, four-tap conv and
  separate SiLU return casts, exact grouped Q/K16→V48 mapping.
  `A_log.float().exp` and `softplus(a.float()+dt)` remain FP32;
  beta is cast after sigmoid. The qualified existing recurrent
  CPU operator updates private **FP32** state; its output alone
  returns in BF16. Gated RMSNorm casts FP32 normalized core
  before learned gamma and again at BF16 gated output.
- Distinct immutable `StateRoot` schema versions for FP32 vs BF16
  diagnostic contexts. Neither may prepare a token using the
  other's root, even with the same original model allocation.

Source references: the published
[Qwen3.5 generated model code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
and [modular GDN/FLA fallback](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py).
No upstream SDK or weights are imported.

Independent source-free Python NumPy scalar mathematical fixtures
supply immutable BF16 bit patterns for zero-centered RMS and gated
RMS, signed-zero/tie-to-even cases and the two-step GDN recurrent
operator (BF16 output exact; unrounded FP32 state tolerance 2e-7).
Rust integration runs two full synthetic four-layer BF16 diagnostic
tokens on the existing TaskCoordinator; checks per-operation BF16
representation, 7 domains, two causal frontiers, rollback and
cross-dtype schema rejection. The GGUF 0.8B/2B trained full-logit/
StateRoot regressions remain separate unchanged acceptance gates.

**Still NOT IMPLEMENTED/QUALIFIED:** independent trained Qwen3.8-27B
BF16/FLA reference logits, per-token exact Qwen CUDA/FLA rounding,
torch GEMM accumulation/attention backend parity, performance,
real trained 27B payloads, official tokenizer and production
SessionAuthority/GPU/SSD execution. BF16 stage agreement is not a
claim of original-model learned checkpoint numerical parity.


## ME-P0 — pinned original-Qwen numerical reference observations

**Source identity:** `verified_shards.rs` now calculates a canonical
`source_bundle_sha256` over the independently verified index SHA256 and
every owned shard's basename, file length and whole-file SHA256.
Domain-separated SHA256 input: ASCII
`zerogravity.original.source-bundle.v1` followed by NUL,
the lowercase index hash, then basename-sorted
[LE32 basename length, basename bytes, LE64 shard length,
lowercase whole-shard hash]. Unlike index-only provenance,
changing learned bytes changes the bundle digest. No new source
issuer, loader or inference backend.

**Reference contract:** `original_qwen_trace.rs` reuses the shared
`PlanObservation` format. Explicit, source-bound diagnostics capture
embedding, GDN projections/conv/recurrent, QK/softmax/value,
FFN/residual and final logits with source, config/index/bundle SHA256,
coordinator/root/epoch, token/frontier and FP32/BF16 dtype.
Caps: 32 MiB, 2048 stage records, 1,048,576 values/stage.
Only ONE caller-selected GDN FP32 recurrent-state layer (or none)
is observed per token; capturing all 48 is too expensive.

`OriginalReferenceTrace::from_pinned_json` accepts only a
caller-SHA256-pinned, strictly typed versioned reference document.
Unknown/duplicate fields, incomplete/reordered stages, forged
source/config/bundle, dtype/token/frontier mismatch, nonfinite or
non-BF16 output words, invalid tolerance and over-budget content
refuse. First bad layer/stage/element and both IEEE words are
reported. BF16 outputs require exact bits; FP32 recurrence tolerances
are opt-in and bounded. `ReferenceComparison::exact()` means
bit-for-bit equal; `accepted()` includes explicitly tolerated
nonidentical FP32 values, counted by `tolerated_values`.
Comparing a candidate never publishes session StateRoot.

The exact typed JSON reference format is:

~~~json
{
  "schema": "zerogravity.qwen.original-reference.v1",
  "provenance": {
    "kind": "trained-framework",
    "framework_revision": "pinned-reference-revision",
    "reference_code_sha256": "<64 lowercase hex chars>",
    "config_sha256": "<64 lowercase hex chars>",
    "index_sha256": "<64 lowercase hex chars>",
    "bundle_sha256": "<64 lowercase hex chars>",
    "token": 0,
    "position": 0,
    "gdn_state_layer": 0,
    "dtype": "bf16"
  },
  "observations": [
    {"layer": null, "stage": "embedding", "bits": ["3f800000"]},
    {"layer": 0, "stage": "gdn_recurrent_fp32", "bits": ["3e4ccccd"]}
  ]
}
~~~

This is an **illustrative fragment**, not a complete valid matching
capture. A real reference needs every ordered observation and full
element array (including logits); each value is an exact 8-digit
lowercase IEEE-f32 bit word. The entire reference JSON is separately
pinned by its expected SHA256 at load time. The caller's
`trained-framework` claim does NOT independently prove provenance;
the actual external framework code/revision/source artifacts must be
validated separately. Model traces and weight contents are sensitive:
do not log/export them without source-owner authorization.

**Qualified here:** pinned manifest/parser/comparison, source-bundle
reordering invariant and same-index/different-weights difference;
small source-free four-layer BF16 diagnostics and analytical final
logit eight, first divergent stage/element, missing/reordered
records, bad source/position/dtype, malformed/duplicate JSON,
selected GDN state, bit-exact versus FP32-tolerated comparison,
and BF16 stage bit assertions. The full forward now uses its
existing source-aware BF16 residual cast, closing one previously
unrounded attention residual.

**Still open:** independently captured actual trained Qwen3.8 27B
checkpoint/reference trace, official tokenizer, original HF/FLA
operator and GEMM accumulation equivalence, complete trained
full-vocabulary logits/StateRoot, production physical GPU and SSD
streaming. No Qwen3.8 trained checkpoint, Torch/Transformers SDK or
reference traces were available locally for this milestone.
