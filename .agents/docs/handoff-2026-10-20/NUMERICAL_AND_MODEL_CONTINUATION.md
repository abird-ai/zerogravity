# Numerical and model continuation — 20 October 2026

Author: baseline_review. Coordinator: ROOT. Documentation-only handoff.

Implementation is **paused at the owner's request**. Writing this handoff does
not resume compilation, tests, model/provider/container/GPU execution or downloads.
The latest accepted checkpoint named by the correctness plan is
`fb68d8fa165de16e21d1bd37afef3790b80da971`; later candidate source/builds must not
be mistaken for accepted source. Revalidate actual HEAD, overlays, worktrees,
processes and artifact provenance when implementation resumes.

This module covers numerical/model context and the post-19-October continuation.
It complements the aggregate handoff's serving/resource/Git sections. Live source,
executed tests and owning TODOs outrank historical descriptions. No numerical
result was rerun and no large payload/binary/DSO was read or rehashed while making
this document; values below come from retained actual result metadata and plans.

## 1. Read order and inherited architecture

1. `AGENTS.md`, `.agents/plans/{README,TODO}.md`.
2. `.agents/plans/numerics-and-context/TODO.md` and
   `CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md` in that directory.
3. `.agents/plans/model-execution/TODO.md`,
   `.agents/plans/memory-and-streaming/TODO.md`, then serving TODO/FUSED_GPU_BATCHING.
4. `.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md`,
   `.agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md`,
   `tests/reference/llama_cpp/README.md`.
5. For original history, not current acceptance:
   `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md`,
   `.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`,
   `.agents/plans/IMPLEMENTATION_ROADMAP.md` and linked original V1–V5.1 specs.

Preserve one Rust mathematical engine: shared family layer traversal,
`MathematicalBackend`/`ForwardPlan`, same immutable source owner and identities,
`SessionAuthority`/`StateRoot`, physical `ResourceArbiter`, original GPU queue,
Ready/completion/quarantine. A native per-operator provider may fit beneath that
engine; a foreign Python/model/session engine is only an independent oracle.
Do not build a second model loop, invent a partial `TensorSource`, replace raw
source axes by transposed aliases, or infer model support from metadata.

The original user statement “complete B>1 full-model fusion is absent” is now
superseded by accepted native full-model B2/4/8 work. That acceptance is for the
declared GGUF arithmetic/source/context. It does not establish vLLM's numerical
profile or temporal-token prefill, and genuine GPU context remains bounded512.

Keep root README untouched; coordinator alone reconciles shared TODOs. One MAIN
Cargo/test/GPU coordinator; disjoint agents review/implement. Copy merged source
with fresh modification times: old preserved mtimes have caused stale-library
risk. Signed commits only. Existing acquisition authorization is not authorization
to resume during this pause; push/tag/deploy/destructive Git/device privilege
changes retain their own explicit requirements. Renoir/gfx90c AMD launch hold stays.

## 2. Four numerical targets and the latest decision

| Target | What it means | What it must not silently become |
|---|---|---|
| Accepted GGUF AR1/reference | Existing source-specific quantization/reductions; native CPU8/F32 KV/FA off reference; independently exact fresh8K regressions | Pinned native default bulk, BF16 HF or vLLM CUDA |
| Pinned llama.cpp default bulk | d812350 actual temporal shapes, tinyBLAS/BLAS selection, quantizer, threads, padding and output-row selection | AR1 loop relabeled bulk, or use_ref/CPU_Mapped control relabeled default |
| Pinned Torch CPU original HF | Exact loader dtype policy, AVX2 reductions, casts, scalar/vector tails, initial64 vs cached schedule and native math provider | GPU tensor-core/FLA arithmetic or all-BF16 loader when original F32 parameters were restored |
| Pinned vLLM CUDA | **Primary new target** for correctness/performance; actual selected loaded profile/kernel/phase/shape | CPU64 kernels merely running on CUDA, or another backend selected only by an environment variable's name |

The latest planning decision treats pinned vLLM as correct unless an independent
counterexample establishes otherwise. Original .8B HF with vLLM's actual loaded
GPU dtypes is the recommended first new lane; GGUF-plugin benchmarks remain a
separate loader/quantizer lane. Keep CPU/default-bulk work as bounded parallel
work, not a prerequisite to completing a direct vLLM CUDA port.

No evidence proves all causes are found. “First observed in GDN” never proves
GDN caused the first rounding error. Compare identical actual inputs/weights and
phase/partition before changing an operator. Exact leaf/internal tests preserve
all bits including signed zero. The old0.45 shared-top20-logprob/0.05 top-token
probability limits apply only to their historical functional tests, not arbitrary
state error. No universal BF16 tolerance is qualified. Do not widen bounds after
a failure or normalize zero signs to force success.

## 3. Numerical investigation chronology: historical versus fresh GGUF

### 3.1 Inherited issue and source recovery

Historical final-token evidence reported .8B layer4 GDN and2B layer1 GDN as the
first compared differing boundaries. At native ubatch512 the shared-top20
logprob discrepancies were0.20372924/0.19678696; matching greedy7993 did not
establish numerical equivalence. Old incomplete ubatch1 capture did not finish.
This was independent of the historical AMD FMA problem; no CPU cause/fix followed
from that separate device failure.

At recovery, named old `target/qualification` raw traces/header directories were
empty. The actual pinned Nix runtime survived:

- `/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/{lib,bin}`.
- Separate original glibc:
  `/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25`.
- Exact source recovered to
  `/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0`.
- Official archive37,874,349B SHA
  `09b36dba235fcac180efff18514da7038e65d8e5b9e4aefa59238468abfdec12`;
  independently matched the preexisting recursive Nix source identity
  `sha256-l6l6JIlIVTaVC6xh5M4fRHFtXsweQuugtkNTWHcZZF4=`.
  Receipt: `.agents/runs/numerics-and-context-root-20261020/source-acquisition.json`.

Fresh deterministic mixed/Unicode text was tokenized by the accepted tokenizer;
raw teacher-ID cases512/1024/2048/4096/8192 are under
`.agents/runs/numerics-and-context-root-20261020/`:
`fresh-prefix-cases.json`, `fresh-mixed-v1-*.u32le` and matching `*.json` ID arrays.
The8192 raw-ID SHA is
`44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d`.
Use raw IDs, not HTTP UTF8 roundtrips that can lose incomplete bytes.

The historical artifact digest was recovered from older source:
`24054920794c549afc0c678f58f6f44a1ecc3228bcfa793aeacc9c8516fb86a2`.
Its serialization is unknown. Fourteen bounded existing-artifact/serialization
checks had zero matches. This does **not** prove different historical token IDs.
The original raw corpus remains unrecovered. See
`.agents/runs/numerics-and-context-bulk-f32-primitive-20261020/HISTORICAL-RECOVERY.json`.
Do not make an unbounded recovery campaign a prerequisite to current diagnosis.

### 3.2 Fresh strict AR1 results

The new bounded helper captures actual selected operators, GDN operands/prior/
proposed state, broad stage boundaries and complete vocabulary. Plain→restore
same prior→observed must preserve complete logits and opaque serialized state.
Older `llama_reference_layers.cpp` reconstructed unnamed residuals; the new
capture observes actual tensors and is not evidence derived from that reconstruction.
Native gain/scale standalone probes remain labeled separately from live trace.

Actual both-model AR1 checkpoints512/1024/2048/4096 match all730 selected records,
full248320 logits, and separately decoded ALL42 causal state domains. Each4K
state comparison covers30,216,192 words. Rust6K self-captures were not independent
6K comparisons and must not be relabeled as such.

Actual fresh8K then passed for BOTH models:

| Model | Selected stages/logits | Independently decoded state | Scope |
|---|---|---|---|
| .8B |730 records/full248320, zero bit differences |42 domains/55,382,016 words, zero differences |CPU8, one actual prefix token/call, requested ubatch1, context rounded8448, F32 KV, FA off |
|2B |same |same |same explicit schedule/source policy |

Actual .8B Rust8K diagnostic took2103.161s,2B2345.422s; these are diagnostic
elapsed times, not serving benchmarks. Actual exported post-state is221,528,064B.
Snapshots cover active causal words, not all future physical padding. Source
images remain original GGUF:

- `target/models/qwen35-0.8b/model.gguf`, SHA
  `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`.
- `target/models/qwen35-2b/model.gguf`, SHA
  `91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6`.

Actual comparisons:

- `.agents/runs/numerics-and-context-long-context-audit-20261020/root-compare-{0.8b,2b}-8192-ar1.json`.
- `.agents/runs/numerics-and-context-state-codec-20261020/root-compare-{0.8b,2b}-8192.json`.
- Native/Rust capture siblings in those directories; invocation receipts in
  `.agents/runs/numerics-and-context-root-20261020/`.

State decoding independently closes six KV plus18 convolution plus18 recurrent
domains. Full730/logit comparison alone did not establish those42 domains; keep
the two results together. Neither fresh result replays the lost historical corpus,
qualifies GPU8K, or makes native bulk arithmetic bit-identical.

### 3.3 Default bulk first-gap localization

**.8B unchanged first64:** exact common prior, both modes' plain/observed whole
post-state and next AR1 fullV/state guards passed. First OBSERVED result gap is
layer3 attention AV (`kqv`) at position2,530/2048 words, before the old GDN4
boundary. All131072 probability operands and1,064,960 causal V comparisons
match;114432 future probability words are positive zero. Earlier captured layers
0–2/GDN state match. This is much stronger than attributing a differing incoming
GDN state to the recurrence itself.

**2B unchanged first64:** first OBSERVED result gap is layer0 F32 alpha
projection position0,13/16 words. All960 preceding compared rows and131072 actual
alpha input words match. The complete F32[2048,16] weight equals the original
131072 GGUF bytes. Beta also differs before GDN. All28 physical learned operands
remain unchanged; eight CPU_REPACK operands are opaque transport, not a qualified
decode back to GGUF. No alpha→BLAS-cause inference is justified.

Public original ggml replays reproduce authentic captures BEFORE controls:

- AV65 defaults/262144 words exact; use_ref bulk differs57823 words from
  default bulk and equals all64 default AR1 outputs. Transparent observation
  keeps all130 default/control outputs,524288 words/run. Public entry trace576
  calls: bulk64 true returns, AR1512 false, reference control none.
- Alpha65 defaults/2048 words exact; use_ref bulk equals all64 default AR1
  outputs, reference AR1 unchanged. All130 outputs/4096 words/run preserved.
  Public entry trace520 calls:8 bulk true,512 AR1 false, reference zero.
- Subsequent genuine debugger observation preserves outputs/operands/maps and
  records132 first named entries across392 stops per dataset. Default bulk takes
  pinned tinyBLAS outer/full/tail families; the other129 scopes enter
  `ggml_vec_dot_f32`. Static source identifies one Vec8 partial/output versus
  four Vec8 partials plus different horizontal reduction. These are first-entry
  observations, not an all-instruction/internal-argument execution trace.

Source: pinned `ggml/src/ggml-cpu/ggml-cpu.c:1308–1333`, `sgemm.cpp` tile/reducer
branches, and current Rust `zerogravity-cpu/src/weights.rs:592` plus
`zerogravity-model-qwen35/src/lib.rs:1305/1423`. The current CPU prefill transaction
at lib.rs1175–1215 still calls `forward_token` for every teacher; it has no actual
CPU temporal-N projection/attention contract to switch today.

### 3.4 Whole-runtime reference controls and their failures

The use_ref observer is a separate transparent control, never a production
arithmetic fix. It keeps original providers, arguments and token loop, observes
actual public CPU compute and inner plan/TLS ownership, and refuses non-CPU math.

Failure sequence retained under
`.agents/runs/numerics-and-context-use-ref-bulk-control-20261020/`:

1. V1 provider lexical `dladdr` pathname guard refused during loading.
2. V2 canonical file/inode/length/export attestation compiled, then refused a
   legitimate non-CPU backend constructor. V3 metadata identified BLAS ACCEL,
   input device equals returned device. This was constructor policy, not GDN cause.
3. V4 allowed only exact BLAS construction/retirement while rejecting its math;
   it accidentally rejected legal `ggml_backend_free(nullptr)` during loading.
   V5 proved exact null/tracked=false/count0 before delegation. V6 restores only
   null-free forwarding; all nonnull route/use_ref guards remain.
4. V6 .8B8K use_ref64 succeeds:630 public CPU calls/630 inner plans true,
   original BLAS lifecycle contexts do no math. All730/fullV/all42 zero bits.
5. V6/V7 2B correctly refuse actual BLAS math before delegation. V7 public graph
   metadata identifies `linear_attn_out-0`: Q8_0[2048,2048] source weight times
   F32[2048,64], then reshape. This is **not alpha**.

Pinned BLAS source59–145 dequantizes weights to F32 and calls SGEMM; support
predicate396–432 admits qualifying contiguous quantized matrices. Existing
accepted AR1 packed dots dynamically quantize activations. A tinyBLAS F32 alpha
fix cannot also fix this separate default Q8/dequant/BLAS schedule.

The distinct CPU_Mapped placement control uses real aligned owned public mapped
buffers for intermediates, retaining original [BLAS,CPU] providers/order and CPU
last; source/model buffers remain original. It observes constructor/split/free,
actual per-node mappings, reset/reuse and compute coverage. No BLAS math is
delegated. Plain/observed SAME-CONTROL transparency and actual2B8K all730/fullV/
all42/55,382,016 words pass. It is not default placement transparency.

Control result locations:

- `.../numerics-and-context-use-ref-bulk-control-20261020/root-0.8b-8192-ref64-6/`;
  `root-compare-0.8b-8192-ref64-v6.json` and state comparison sibling.
- `.../numerics-and-context-cpu-mapped-placement-control-20261020/root-2b8192-observed-1/`,
  `root-2b8192-pair-1.json`, `root-compare-2b-8192-cpu-mapped-v1.json` and state sibling.

The leading `.../` here means `.agents/runs/`, not a different cache root.

### 3.5 Default-bulk implementation remains a paused candidate

The closed generic Haswell candidate preserves exact single-Vec8 FMA/horizontal
order, full K, actual strides/padding, finite/overflow/alias bounds and original
source identity. Four-Vec8 production paths are unchanged. The minimal typed
slice is SAME PlanBuilder/MathematicalBackend/HostMath input/input/matrix, with
sealed temporal-span/layout provenance, source-borrowed F32 alpha weights and
admitted output/scratch/retained backing. AV imports authenticated observed inputs;
it is not a live KV producer or model state transaction. No B-cohort→temporal-N
substitution is allowed. Mozilla's exact2024 MIT notice was restored additively.

- `.agents/runs/numerics-and-context-bulk-f32-primitive-20261020/`: original
  freeze `689221fe`, attribution follow-up `a7e8cc3b`.
- `.agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/`: freeze
  `0d11d476`, TERM/INT-safe harness follow-up `fce99`.
- Integration design:
  `.agents/runs/numerics-and-context-bulk-f32-integration-20261020/`.
- ROOT detached worktree:
  `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-default-bulk-20261020`.
  ROOT checklist: `.agents/runs/numerics-and-context-root-default-20261020/TODO.md`.

Independent source review found no remaining blocker. The ROOT worktree has
four exact source patches only and is **uncompiled/unqualified** at this pause.
Prepared genuine AV+alpha132096-word replay must pass before any same-engine
temporal producer/scheduler implementation or default wholemodel claim. Then
close actual2B Q8→F32 BLAS separately and compare unchanged first64 before the
costly512→1K→2K→4K→8K ladder. Full-K output tiling may bound work; splitting K
into separately rounded partial reductions changes the reference.

## 4. Original HF sources, runtime and first diagnostic gaps

### 4.1 Source identities and capacity

| Source | Revision / location | Accepted scope |
|---|---|---|
|Qwen/Qwen3.5-0.8B |`2fc06364715b967f1860aea9cf38778875588b17`; source receipts/captures below retain actual mount/file identities |Original HF metadata/weights and trained operator/reference captures; no complete HF execution |
|Qwen/Qwen3.5-2B |`15852e8c16360a2fea060d615a32b45270f8a8fc`; snapshot `88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52` |12 original files/full hashes;632 headers/320 text roles,596BF16+36F32;4.548GB shard exceeds complete-source4GiB default |
|Qwen/Qwen3.8-27B |`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; snapshot `98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e` |Actual31-file/18-shard acquisition/header/hash closure and bounded operators, not model execution |

XDG cache is `/home/pvl/.cache/abird/zerogravity/models`; snapshots use
`snapshots/<snapshot>/files`.27B selected artifacts55,586,113,293B include metadata;
weight files55,563,006,776B; tensor payload55,562,855,904B.1199 total tensors,
851 ordinary-text roles and348 others. Ordinary-text tensors are genuine BF16;
do not inherit .8B's restored36F32 policy. Exact source pin7058B is retained in
`.agents/runs/model-execution-qwen38-owned-window-20261020/source-pin.json` and
the later replay directories, SHA
`4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a`.
Index SHA `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`;
bundle SHA `7e6a80bc4fc06b6237914007bde59a48a6e4fb0ea570bd5078b3101eb169faf2`.
Keep per-shard hashes in that pin instead of reconstructing them from prose.

27B was acquired through the existing Rust bounded/index-first cache path with
finite64GiB source budget,128MiB transfer budget/per-file4GiB bounds and streaming
buffers. Debug SHA throughput was slow; ROOT built a stable release acquisition
binary, the agent cooperatively interrupted its own debug downloader and resumed
the SAME cache/reference/source pin using durable strong-ETag checkpoints. No
partial cleanup, source replacement or whole55GB RAM materialization was used.

27B geometry:64 layers =48 GDN+16 full attention;112 domains (48×conv/recurrent
+16KV); hidden5120, MLP17408, vocabulary248320, untied output head. Layer0 QKV
raw[10240,5120],Z[6144,5120],a/b[48,5120],out[5120,6144],conv[10240,1,4],
A_log/dt[48],gamma[128]. Raw Q/K heads16 repeat_interleave3→48 value heads;
key/value widths128. State math key-major `[head,key,value]` differs from
canonical/vLLM contiguous-key `[head,value,key]`. Raw source K4 history differs
from canonical newest3 oldest→newest. Equal byte counts do not justify a view.

### 4.2 Pinned native environment and independent captures

Genuine acquired runtime is vLLM0.31.0+cu129, Torch2.13.0+cu129
(`cf30153c4c131c8164ee7798e5022d810682e2cb`), Transformers5.17.0. Original
official image digest used by references:
`docker.io/vllm/vllm-openai@sha256:b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f`.
The later exact derived image identity is
`sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe`.
Different receipts retain their actual image/closure; do not rewrite old evidence
as if all executions used one image.

Stock CLI import hit bundled TorchCodec needing libnvrtc.so.13 in a CUDA12.9
image. A reproducible isolated derived layer replaced only the pinned codec with
the exact CPU wheel. No host driver/package/global config or numerical model
kernel was patched. Original model-class import for the CPU oracle already
worked without this optional codec route. Initial `--entrypoint python` failed
because only python3 was on PATH. Actual original imports and separate CUDA
context probe established RTX3060 sm86/~5.92GB free at that checkpoint, not model
execution or presently free memory.

Independent .8B CPU capture:
`.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/captured-reference/`.
Teachers11/353, complete248320 logits/all42 domains and layer0 intermediates:

- `official-bf16-loader`:473 BF16 loaded parameters.
- `source-36-f32-restored`:437 BF16+36 original F32. First argmax198 versus
  official321; this is an intentionally distinct effective loader profile.
- Full source/raw archives, original manifest and source-receipt distinguish
  source versus loaded parameters. The tied output head fix is source-real,
  selecting original embedding when config says tied, never a fabricated lm_head.

Independent2B reference:
`.agents/runs/model-execution-hf2b-cpu-reference-20261020/root-probe-2/captured-reference/`.
Official617BF16 versus restored581BF16+36F32;47/49 raw tensors differ each step
despite identical argmax. Four full vocabularies/168 canonical domains/60 genuine
intermediates pass archive/owned-lifetime validation. First capture's cleanup-name
bug remains in root-probe-1. Paired attention observation adds216 genuine BF16
tensors over six attention layers/nine stages/two teachers/two profiles while
preserving all196 prior raw tensors and full outputs/state;688128B. Location:
`.agents/runs/model-execution-hf2b-attention-paired-20261020/root-pair-1/`.

### 4.3 Earliest .8B CPU diagnostic diagnosis

The explicit whole-model diagnostic was not trained-qualified. First actual
projection gap was1/6144 at QKV index3553 after layer0 normalization. Missing
live norm trace was not filled by a synthetic claim: ROOT emitted a separate
direct existing-operator norm/raw-projection probe, verified it independently,
then added actual norm observation. The shared explicit CPU64 BF16 row profile
reproduced pinned AVX2 eight-by-eight FMA/reduction order; QKV becomes exact for
both tokens. Existing default/GGUF/F16/F32 arithmetic did not change.

After that fix the old whole diagnostic still reports41/42 differing domains;
full logits differ214245/214285 words, maxabs .22119140625/.1875. First token
exposes one conv/SiLU signed-zero difference; second exposes recurrent state.
Embeddings/direct norm/QKV and convolution history can match while downstream
state does not. This is an old diagnostic result, NOT a rerun of all newly
qualified helpers composed together.

Current unqualified façade `crates/zerogravity/src/original_qwen_math.rs` still
contains legacy grouped-RMS L2, host `.exp()` coefficients and GGUF fused
recurrence; precision.rs140–143 unconditionally rounds the gamma product even
when original raw F32 gamma should promote it. Those are real target-specific
implementation gaps, not justification to rewrite accepted GGUF operators.

## 5. Accepted shared-engine model/math milestones

The following sequence is from the live plan's committed groups; older synthetic
milestones explain architecture but do not confer trained numerical acceptance.

| Commit | Concrete result | Limit retained |
|---|---|---|
|`b796a2e`, `b11e58c` |Same-source original HF axes/roles, raw matrices and analytical one-layer operators; 16→48/GQA/ordinary versus gated norm semantics |Small original-format synthetic sources; no trained27B |
|`6329cff`, `e43ba4c` |Original TaskCoordinator one-layer then shared all-layer analytical schedule, private state proposals and one publication owner |Diagnostic F32, not a second engine or trained model |
|`aaf9ba6`, `1c0bb41` |Explicit BF16 diagnostic casts/schema, original bundle/trace and numerical-reference seam; residual result cast fixed |Early synthetic state tolerance2e-7 is not today's trained exact-state acceptance |
|`f26232e` |Authentic tied/untied original output head selection and immutable source tests |Diagnostic head fix, no full HF promotion |
|`9815504` |MatrixAxes/source roles/raw SourceParameter/state metadata, explicit CPU64 profile, closed FP32 L2/key reducers |HF session/forward/checkpoint guard stays closed |
|`cadcbd9` |Pinned original conv4, cached128 GDN, F32-gamma RMS prefix, SLEEF SiLU; actual original operands |No initial64/coefficient exp/model or GPU permit |
|`2a32aff` |SAME typed PlanBuilder/HostMath embedding/projection; true original BF16/tied fullhead probes; vector softplus |State-free slice; no hidden frontiers or fallback |
|`c5eb1f8` |Closed BF16 sigmoid and ordinary RMS widths256/1024/2048/5120 |Scalar/vector call geometry and raw dtype are part of the contract |
|`5350a4d` |Typed ordinary RMS embedding→InputNorm0→QKV0; sealed rank1 windows, source full-column row wrapper, closed softmax |No partial source alias or full HF attention |
|`bfe62df` |Authentic27B same-engine three-node source-window execution |Pure operators/frontier delta0, not112-domain inference |
|`fe2a342` |Both-source original CPU attention fragment; authentic vLLM public GEMM/operand evidence |Attention fragment excludes RoPE/KV/BMM/softmax/full state |
|`fff1842` |CPU-target BF16 CUDA projection +both-source conv/SiLU fragment/full gates |CPU-target arithmetic on CUDA is not vLLM CUDA arithmetic |

Representative actual quantitative gates:

- Closed128 reducers:14 checks/12416 values, exact. Cached16-head replay:
  262144 state words plus5×2048 normalized/scaled/core/output words exact.
- Original conv four independent FMA partials/direct combines then BF16,
  accepted SiLU source arithmetic. Actual four causal positions6144 channels:
  49152 words; later composed BOTH sources initial/cached routes122880 BF16
  raw/activated/newest3 words. Signed zero was fixed by source schedule, not
  canonicalized. ForwardPlan/HostMath, raw BF16[C,1,4], original lease owner.
- SiLU all65280 finite BF16 inputs +4096 actual gates exact; F32-gamma prefix
  8256 words. F32 gamma product remains F32 after normalized BF16 cast.
- Softplus actual32 biased alphas +28464 adversarial F32 words exact; beta1,
  threshold20, spans divisible16. No arbitrary scalar-tail or Tensor.exp permit.
- Sigmoid326753 output words exact across all finite BF16 values and32-vector/
  scalar/mixed shapes, including actual beta16. Head48 call must retain its
  vector32+scalar16 schedule, not combine teachers into a different span.
- Ordinary RMS154 cases/1694 fields/1680488 words; original122 .8B norm calls
  retain98 prior full-logit/cache checks. Raw BF16/F32 weights, runtime1+gamma,
  correct F32 reduction/division/product and only then final BF16 cast.
- Softmax3599 public cases/1838736 words; width1..512 and explicit refusals.
- Typed embedding→independent QKV/gate/tied head518144 original words; typed
  embedding→norm→QKV .8B16384 words. These are not complete model logits.
- Both-source attention fragment47104 raw BF16 words from seven projection/
  view/QK-normalization outputs, teachers11/353. External actual hidden input
  is not proof of preceding producer/current state.
- CUDA CPU-profile BF16:163840 raw F32 accumulations +163840 final BF16 words
  exact, original .8B/2B layer3 q/k/v teachers11/353, B1/B2/4/8. Synthetic328
  case/band pairs cover widths64..8192; eight actual overflow flights withhold
  Ready/retain quarantine. Exact same queue/leases/source, not cuBLAS profile.
- Composed MAIN BF16/conv phase full isolated1251 Rust/0fail/70ignored/248
  suites +18Python; strict CPU/CUDA/Vulkan/CPU-free cold refusals and both
  approved original cache-direct/HF-import RELEASE routes pass full248320/all42.

Individual earlier phase full gates were1164→1202→1213→1218→1241→1243→1251
Rust passes with zero failures, plus their explicitly ignored tests and18Python.
Do not substitute one old gate for later changed source. Relevant physical
source/state/cancel tests and complete mandated isolated check remain necessary.

## 6. Bounded27B source transport and real operator results

### 6.1 Source owner/windows and three-node path

`SourceWeights::prepare_verified_shards` captures actual opened FD/header/index
closure and aggregate original bytes/metadata before issuing one SourceIdentity.
Its complete-source4GiB guard remains. `prepare_verified_windows` adds bounded
same-owner extents/staging without materializing55GB or fabricating a complete
partial TensorSource. Sealed matrix/rank1 descriptors retain original full shape,
dtype, axis and issuer; partial ranges are explicitly partial raw transport.

Independent review found real capacity-accounting gaps in early window proposals:
retained vector capacity and chunk offsets must be charged, not just lengths.
Final guards count source metadata/shard capacity, staged data+offsets, bounded
labels and retained clones; generic control-plane overhead remains outside the
staged-payload figure. Same-FD full hashes/headers/index bind all geometry.

Actual27B source-window harness:
`.agents/runs/model-execution-qwen38-owned-window-20261020/actual-report.json`.
Scans55,563,006,776B/18 shards in82.847s;12 actual full-width first/last/split
row samples over four real matrices, four extent crossings, exact raw bytes.
Peak stage8,388,656B, configured128MiB host pool/64MiB metadata, final0, observed
RSS17328KiB under512MiB AS. This is PASS_SOURCE_WINDOWS_ONLY, not physical
free-RAM admission or streamed inference.

Authentic original27B public embedding11/353→ordinary input norm5120→QKV10240
reference: `.agents/runs/model-execution-qwen38-typed-oracle-20261020/root-reference-1/captured/reference.json`.
Selected genuine mmap/raw matrices, no model constructor,40,960 words. Actual
same PlanBuilder/HostMath replay at `bfe62df` matches all40,960 words through80
real128-row/full5120-column QKV windows. All851 text roles close to same source;
no fake LoadedQwenModel/TensorSource, copied transformed weights or second loop.
Peak real stage8,388,656B; raw norm staging4,194,328B. Each result retains actual
64KiB Output+960KiB Scratch and source through clone/lastdrop. Stage-triggered
mid-cancel/pre-cancel/pressure/foreign config/index/source/token negatives and
final0 pass. Actual92.74s includes90.79s source scan; RSS18736KiB is observation,
not physical-RAM admission. .8B three-node regression16384 words/final0 also passes.

Historical resolver proposal and actual-replay text live under WORKTREE
`/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-qwen-window-resolver-20261020/.agents/runs/model-execution-qwen-window-resolver-20261020/`.
That run directory does **not** exist under MAIN; production source is now committed.
Preserve the historical actual-replay text there if ignored-run storage is retired.

### 6.2 Genuine27B initial/cached public reference

`.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/`
contains actual original layer0 initial chunk64 and true cached continuation,
teachers11/353, H48 and raw BF16 parameters. Public plain/observed output and
conv/recurrent state/readsets, full source scans and owned retirement validate.
Manifest identity `62e82637c7d24a58b733f2358b3061efd767725901314df4ff7378707d946ab8`;
source plan `658280444d4e3d1b6a3a8f2ce4b97e4be7cbff2c8a6ea7c66b3c733c78f31638`.
Actual raw archive payload80,360,448B. Validation status is
PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY, not native wholemodel parity.

V2 transport failed after saving observed data because a logically contiguous
singleton BF16 view retained last stride10240; `.contiguous().view(uint8)` was
not a safe byte transport. V3 uses same-size integer reinterpretation and NumPy
C-order bytes, no numeric casts. Twelve real layouts and192 literal BF16/F32
signed-zero/subnormal/NaN/Inf bit cases distinguish transport from arithmetic.
Original failure and raw archive remain. Do not regenerate expected values from
the Rust candidate or infer a failed transport was a failed model operation.

### 6.3 Isolated521-source composition and actual H48/wide gates

ROOT owns detached worktree
`/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020`,
base `fff1842` plus reviewed pure wide/coefficients/cached H16/H48/BF16-gamma
overlays and narrow ROOT repairs. MAIN remains separate. Final237 focused
tests (100core/66CPU/71model),0failed/2ignored and strict checks pass.
Exact521 Rust/manifest freeze:

`.agents/runs/model-execution-root-bf16-composition-20261020/root-composed-source-freeze-7.json`
SHA `036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615`.
Source composition journals, all failed attempts1–6, source freezes1–7 and actual
final root-cpu-focused-7.json are retained. These operators are **not merged into
MAIN** and the237 tests do not replace a full isolated gate for a future merge.

Actual replays after new isolated-path/Cargo/lock/TERM+INT adapters:

| Gate | Actual result | Exact continuation location |
|---|---|---|
|H48 cached core |804960 imported +798720 output words, including786432 recurrent words, exact before/after caller drop; pressure/cancel/source/final0 |`.agents/runs/model-execution-hf-cached-gdn-contract-20261020/followup-h48-v3/replay/isolated-root-v2/root-native-1/validation.json` |
|BF16-gamma gated RMS |61632 primitive-stage,12288 typed-final,24576 retained-input words exact; retained source/private backing/final0 |`.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/root-native-1/validation.json` |
|Pure wide roles |32 cases/16 roles/881856 complete BF16 input/output words, including248320-row head and17408-column down; pressure/pre-cancel/retained head/final0 |`.agents/runs/model-execution-qwen38-pure-wide-isolated-20261020/root-native-1/report.json`; ROOT wide build/source receipts in root-bf16-composition run |

Cached result imports genuine observed exp(g) and already repeated Q/K, not
live coefficient/head-repeat producers. Zero previous state does not select
initial chunk arithmetic. Gamma BF16 product has its proper intermediate BF16
cast; the raw-F32-gamma CPU path instead promotes/retains F32. Pure wide inputs
are authenticated external BF16, not trained outputs of preceding model layers.

Standalone replay binary identities recorded by ROOT, not rehashed here:

- Cached `0ac56a44c9d09504c2afea5ff68fe5d33ec1e13d427e525f5e5df15afb869e7e`.
- Gamma `a381ac4cc998e4c16dfcfab4bd1be0be8c77bcd0a7c39745e9bf415f0b9fc477`.
- Wide `461af3178e1b185766f9345a455a166e3208569c90effceb301f51e717b8ca21`.
- Actual root-replay-build-2.json/root-wide-build-1.json and per-replay source
  attestations bind these to036ce0 and generated exact parent-lock subsets.

Original pure-role freeze `0f9f2a8f` has explicit follow-ups `65cd4364` for exact
hidden norm recording and `7759599f` for deterministic completed-output cancel.
Qualification original `16809776089e1c3063629a0e0df556512a66b9ccd04d6812d56e47b744f4cb40`
omitted7058B source-pin; metadata-v2
`e9581cf982e3c83fa85413a4eb707afef001290c6137563a34a9c679a54514e6`
adds it and binds actual compiler inputs. Later isolated relocation manifest
`44de7b7a56cf7511fdac3de9c039395542d58db0c34202cc7c09b964b7d2ac44`
preserves original arithmetic/expected values, not a MAIN retag.

Composition fixes are not numerical-tolerance changes: old prepare_output callers
gained correct H16/H48 sizes under old floors; tests used real<=32MiB pressure
chunks, valid hybrid schedules/minted identities, role-distinct tiedEmbedding
rejection and drop of a genuinely retained diagnostic parameter before final0.
Unnecessary Vec/private-test/import strict failures were corrected narrowly.

## 7. Native CPU provider: actual arithmetic versus admission

Pinned public native probe:
`.agents/runs/model-execution-pinned-native-provider-probe-20261020/root-actual-3/`.
The SAME Rust executable ran inside the already acquired pinned GNU-loader/
libtorch CPU closure, no Python/model/token/session/RPC loop. Exact public ABI:

- `vmsExp`: LP64 count int32 by value, FP32 pointers, explicit int64 mode0x140102.
- `cblas_sgemm_batch`: public grouped C API, column-major102, trans flags111/112,
  group1/head16; no guessed Fortran string lengths.
- `strsm_`: exact pinned Torch declaration,11 pointer arguments, no hidden string
  lengths. Public L/U/T/U shapes64×128, alpha1, ld64 match actual source.

Actual3 matches1,048,640 independently captured words across4 EXP16 calls,6
batch-GEMM families and32 STRSM calls, preserving full source input readsets,
private/redzone outputs, selected public export identities, mapped closure and
MXCSR/thread controls. Internal microkernel remains UNOBSERVED, native internal
memory admission and portable Rust math remain UNQUALIFIED. Expected bytes were
opened after native completion, not passed as operator inputs.

The original OCI cpuset route failed because cgroup controller was unavailable;
another inherited-affinity run had correct arithmetic but remained UNQUALIFIED
at0–15 CPUs. Actual3 sets owned-process four-CPU affinity BEFORE native DSO load
and checks actual proc status. Failed attempts are not relabeled passes.

Startup-envelope experiment:
`.agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/`.
Original freeze `961d803b`; new ROOT native-v3
`5dc8e3b3fd806e9ccc13b77e6f19679da0ccc6fb5d7ebd590eebc685f0616b95`
fixes only unnecessary_unwrap and Cargo relocation. Execution adapter
`34563b94e5e80413a41ed246a9aa57d9000703ef00798ce2e677cd7d9b518957`
preserves original owned helper/guard. Actual binary
`92cd48dc635aba728294633937846bec50ba18629796358f4878d00ed98938b9`.

All6 actual cases and independent validator pass:
normal,pre-cancel,pre-spawn admission refusal,cancel-after-hello,AS pressure,bad-AS.
Parent admits original4GiB ProviderEnvelope before child; owned ticket remains
until known full retirement/EOF, unknown lifetime fences/retains. Child has2GiB
hard AS,4 CPUs, bounded FDs/output/stack/CPU time, NNP/caps0, and denies clone/
new threads/exec/namespaces/limit mutation. SIGTERM/INT is record-only until safe
handoff checkpoints; original WNOWAIT SID/PGID ownership cleanup is reused.
Actual validation status is PASS_STARTUP_OWNERSHIP_ONLY_PROVIDER_UNLOADED.

**No provider was loaded and no math operator was called in these6 cases.**
MKL/OpenMP4-thread initialization is incompatible with the current clone-denying
startup policy. DSO constructors, internal caches/pools, stacks, VM/kernel costs
and opaque allocator lifetime are not admitted by merely charging caller vectors
or choosing AS/cgroup caps. First actual admitted exp16 then genuine H48 exp is a
separate future gate. This CPU-native blocker must not hold the direct CUDA port
hostage; reuse correct CPU oracles as separately named targets.

## 8. Paused coefficient/initial-prefix candidates and exact525 trap

### 8.1 Coefficient preparation

Original preparation at
`.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/`
freeze `d744cb9ddab32ab94e144758ae954548c90fec0ee14ec4eb0f7b1c5c44915660`
is a same-engine two-node beta/alpha preparation, no exp/g/decay:
betaBF16 sigmoid; a.float()+rawBF16/F32 dt→F32 softplus. Core/model/helper
ownership and lease guards remain. Actual public4 cases/all11 .8B archived stages
pass. Original validator's tuple/list JSON metadata equality failed; ROOT V3
normalizes only metadata through JSON roundtrip, preserving the original failure.
Do not call public target availability native full-GDN qualification.

H48 original27B accommodation:
`.agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/`, freeze
`05095281de89dfc2ed136e869b5ca11458ca3664293b94f79a2fae900fc07e2e`.
Only unused A_log descriptor expands to genuine BF16[48]/96B under exact role/
issuer/profile; dt already accepts BF16/F32. A_log remains UNEVALUATED.
RawBF16 a/b, original dt and authentic V3 teachers11/353 supply288 output and192
retained input words. The positive H48 native replay has not run.

Original negative called layer1, but actual27B layers0/1/2 are GDN and3 is FA.
Additive `followup-wrong-layer-v2/` freeze
`e4be9998e85ad13b53a11640a5b56afca6899d7045ca3b859a45b61f06071b8a`
binds realconfig FA3 and changes that negative1→3 only. Keep0509 unchanged.
Synthetic2-layer tests with FA1 had hidden this harness bug; no source/model math
fix follows from correcting the negative. Independent review found no further
source blocker; build/replay remain pending.

### 8.2 Exact CPU initial query/prefix leaves

`.agents/runs/model-execution-hf-initial-prefix-20261020/`, freeze
`3cda106a5aee61dedb96b4255295acbfc556a29a9b80d59493f3f86f56d5cb39`.
Three new CPU sources plus module declaration, no model/GPU/state enablement:
`initial_prelude.rs`, `initial_prelude/{admitted,tests}.rs`, upstream BSD notice.

Exact Torch source cf30153 `ReduceOpsKernel.cpp79–95` uses CPU float→double
carry from+0 and sequential scalar_t stores; `AccumulateType.h64–66/155` confirms
the type. Complete primary source/LICENSE copies and URLs are retained. This
supersedes the older audit's unverified cumsum links.

Closed operations: ONE already-normalized contiguous `[H,1,128]` query multiplied
by F32 bits0x3db504f3, or ONE already-padded `[H,1,64]` decay prefix with F64 carry
and EACH F32 store. H16/H48 only; explicit InitialChunk64; Cached/DIV refuses.
Passive x86_64/AVX2/MXCSR RNE/FTZ/DAZ guard; no environment mutation or provider.
Admitted Linux/cache wrapper retains exact source/backing/WindowWorkspace,
pre-admits input/output+256B, no bare Vec escape. H48 charges49408B/24832B,
total74240B; general control-plane/caller external operands are outside that figure.

Prepared authentic V3 query6144+prefix3072 output and9216 retained-input replay,
real pressure/pre-cancel/foreign/Cached refusal/lastdrop tests are **unexecuted**.
Author's fresh self-audit is labeled self-audit; Batch's separate source review
was requested and reported no concrete blocker in progress, not an actual run.
Initial chunk solver/exp/BMM/TRSM are not implemented by these leaves.

### 8.3 Combined525 attestation derivative — do not reuse old attestors

Both original candidate replay manifests hardcode the same ROOT036ce0 tree.
Initial attestor accepts exactly524 Rust/manifest inputs; H48 accepts522. Applying
their seven disjoint path deltas creates525, so BOTH original attestors correctly
refuse the joint tree. Do not remove other changes to force either old count.

New source-only derivative:
`.agents/runs/model-execution-hf-combined525-attestation-20261020/`, manifest
`e97eabda99bc8a940baa886180f07e30988bfded8062f9d779550f1e6506f841`.
55 frozen files, original manifests/files unchanged. Exact ROOT521 +seven path
deltas =525, plus retained BSD notice. `COMPOSITION.json` binds all before/after
rows/suppliers; `SOURCE_DELTA.patch` identity
`8ed8dfb4bbc1b35af74cf85740dcf05ee954e8e43d2aaeb95e47a9dadfeaddeb`.
`REVIEW.diff` identity
`294d01aaafceaf6b43f4c6f1dd72c9939e4091bf187f48175d5c769123adf781`.

Fresh initial/replay and h48/replay entrypoints require exact525 source closure,
license,six actual helper compile inputs and generated lock =533 attestation rows.
Both main.rs change only an always-emitted lane-specific compiled build identity;
runner and independent validator require it and a distinct actual ROOT compiler/
binary receipt. An old successful binary cannot gain new provenance by relabeling
a sidecar. Complete instructions are in that derivative's README; no new framework
is required elsewhere. AST24/rustfmt2 only by author;6 new metadata fixtures
prepared, not run. No isolated-tree application/build/raw replay happened.

## 9. Actual vLLM evidence and why matching is still open

Pinned vLLM commit `db9527a46873454610df6dbedf79a36d6bf1a7f6` and original
model source are in the retained image/closure. Official GGUF plugin commit
`e2b8ad532b8b5ea175100202c30430c1d2b5e6a8` was separately acquired/installed
in an isolated, RECORD-verified site; old statement “stock registries have no
GGUF entry” describes the no-plugin baseline, not the later official plugin run.

Accepted eager HF .8B public observation reproduces both engines' own original
full496640 raw BF16 logits and320 loaded source roles before/after. Four genuine
QKVZ/BA calls for teachers11/353 use public cublasGemmEx:BF16 A/B/C,FP32 compute,
DEFAULT_TENSOR_OP request, default math, host pointers, atomics disallowed,
alpha1/beta0 and actual8519680B workspace. Exact already-loaded provider scopes
fixed generic resolver failure without fallback/new loads. Internal selected
cuBLAS kernel/reduction and opaque library memory admission remain unqualified.

Six authentic BF16 input/result files36992B were extracted read-only with raw
pointer/extent/call/packing provenance. Location:
`.agents/runs/model-execution-vllm-gemm-operands-20261020/`; observation source
under WORKTREE `p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-vllm-gemm-observation-20261020/`.
Pure BF16 operands are not GGUF-plugin operands. Original allowed-ID oracle
controls were not benchmark generation constraints; serving quality/timing uses
unconstrained actual greedy/EOS policy.

Latest target differences, source-proven but not all captured as causal failures:

- GGUF accepted codecs prepare Q8_0/Q8_1/Q8_K activations by source codec;
  stock plugin uses Q8_1 plus its own rounding/scales/reducer/cast path.
- Plugin reconstructs A_log=log(-ssm_a); exp may not recover exact stored bits.
  Ordinary norm subtracts1; GDN raw gamma is excluded. Loaded-parameter adapters
  and grouped permutations belong in the declared target profile.
- vLLM fused conv applies SiLU to F32 accumulation before its BF16 store;
  CPU original conv stores BF16 before separate SiLU. Neither is an arbitrary bug.
- SM86 automatic prefill selects Triton; ordinary packed decode uses
  `_forward_core_decode_non_spec`, not the guarded C++ MTP path merely because
  GDN_DECODE_KERNEL says cuda. Actual branch/environment must be observed.
- Prefill Q/K normalization stores BF16; beta/log gates stay F32. Cached normal
  decode works in F32 and scales Q before readout. Prefill and decode softplus
  expressions differ. FLA_USE_FAST_OPS and FLA_TRIL_PRECISION matter.
- CUDA chunk64 is mixed-precision WY/solve_tril: source16×16 block solve/merge,
  BF16 solved-A/u/w/saved entering h/v_new boundaries, F32 running state, explicit
  causal masks and dot policies. Do not substitute CPU six-BMM/STRSM schedule.
- vLLM gated norm retains F32 through gamma/SiLU to finalBF16; CPU raw-F32/BF16
  gamma contracts have their own qualified boundaries.

The correctness plan lists exact pinned source modules and phase formulas. It
does not assert a newly executed Rust-vLLM complete GDN comparison. Compare
projections FIRST; later cast differences cannot explain an earlier input gap.
Same source/argmax/text/observer transparency does not establish fullV/allstate.

Real bounded eager C1 comparisons already show native slower: median .8B TTFT
13.400s vs1.561s,2B14.540s vs1.562s.2B client individual ITL p95
0.186–0.199s vs0.026–0.030s. Profiles/quality remain unqualified; source-correct
goodput remains null/false. .8B native hardware window236270 kernels/direct APIs
matches source counter with0drops; stock9318 recorded activities/8810 launches
does NOT have complete losslessness/all-process proof. No exact count ratio or
timed/profiler equivalence is allowed. Full serving/benchmark details belong to
the aggregate's serving module and live plan.

## 10. Concrete remaining integration seams and implementation order

**Primary route:** C0 declare originalHF vLLM-loaded .8B profile and actual
initial/cached phases; reuse true GPU QKVZ/BA operands to find first difference.
Port the smallest selected kernel/cast, compose its real producer/consumer, close
one complete GDN layer then full .8B→2B→27B. Preserve CPU/AR1 regressions. Keep
native-default CPU reproduction parallel. Never require completing MKL CPU before
CUDA. Correctness-safe chunk execution can precede its fastest optimization.

Exact SAME-engine source map:

| Area | Existing seam | Still needed |
|---|---|---|
|Source/roles |model `original_binding.rs`, `original_binding/`, `operator_slice.rs`; core SourceTensor/SourceParameter/MatrixAxes |Actual effective loader/profile in descriptors/compiled keys; no source retag or transformed unadmitted cache |
|Typed operators |core `accelerator/program/typed.rs` and typed children, model `forward.rs`, `host_math/`, CPU `original_hf/` |New vLLM numerical contracts/phase dtype; default/unaware providers refuse before callbacks; no Err fallback |
|Cached CPU |`original_hf/recurrent.rs`, cached typed slice, accepted H48/gamma |Live source QKV/conv/repeat/coefficients/decay, then core→norm→out; no imported target in composed gate |
|Initial CPU |new initial-prefix leaf, existing L2/reductions |True pad64/F64 prefix/masked-exp6BMM/2solve groups/materialized layouts, native exp/BLAS admission; H48 means96STRSM |
|CUDA target |resident_program/batch/operator modules and original queue |Selected vLLM conv/normalization/coefficients/chunk+cached/gated norm/GEMM arithmetic, not CPU64 substitution |
|State |original SessionAuthority/StateRoot/layout descriptors |Private layer proposals first, then complete42/112-domain effects and compatible source/profile/schema/checkpoint bindings |
|Capacity |verified windows/ExtentSource/StagedSlice |Complete model source-window schedule and actual RAM/SSD/GPU admission; source-only scan is not streamed inference |

CPU cached source order: betaBF16; A_log rawBF16/F32→F32 exp/neg;
a.float()+rawdt→softplus; separateF32 g and exp(g); head repeat and exact L2;
query DIVsqrt128; key-major FP32 decay/products/key sums/separate update multiply
and add; BF16 core; dtype-correct gamma/SiLU/out. Accepted SLEEF exp inside
SiLU/sigmoid/softplus is NOT standalone MKL Tensor.exp. Initial uses MULinverse
sqrt and padded64 solver, not cached on zero state. Strict-upper -Inf must be
validated only in a closed mask domain; never weaken general finite guards.

Full forward/session/checkpoint guard remains at
`LoadedQwenModel::require_execution_profile` in original_binding.rs307: only
GgufSourceV1 executes. New raw metadata/private operator views grant no current
state, session, checkpoint-v1 reinterpretation or full originalHF capability.
The old original_qwen façade is a diagnostic; new helpers belong under the same
shared math/plan instead of creating a second diagnostic-to-production engine.

Qualification sequence when the owner actually resumes:

1. Revalidate code/artifacts/processes and paused overlays; no generic target/
   release binary can be presumed accepted by filename. Review source before merge.
2. Genuine fixed-input leaf fails-before/passes-after; actual compiler/provider/
   source identity/finite/canary/readonly/lease/cancel/unknown semantics preserved.
3. Compose REAL layer0 initial then true cached353 with nonzero prior; all source
   intermediates, complete nextconv/recurrent state and output. No injected
   expected decay/QKV in the final composed test.
4. Full B1 .8B then2B: each layer residual, full248320 logits/all42 domains;
   initial/continued/cached partitions compared to the SAME reference partition.
   Then B2/4/8, row order/refill/ragged roots, B1↔native transitions/cancel/final0.
5.27B pure-role/H48 results expose geometry but full trained producer/model/
   source-state112 gates remain. BF1655.56GB does not fit RTX3060 6GiB; use a
   capable matched device or fully admitted real streaming. Report unavailable
   baseline instead of substituting a quantized/new source.
6.512→1K→2K→4K→8K GPU only after context capacity and actual predecessor gates.
   Larger16K/32K remain distinct. Prompt+consumed generation must fit capacity;
   don't run512 prompt+32 output against a512-token device state.
7. Relevant strict/physical checks, both approved original RELEASE parity routes
   and complete `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`.
   Plan reconciliation and signed commit only after actual verified source/gates.

After correctness, temporal prefill is the structural performance priority:
sequentialT2/T4 amortizes transaction/transport but is not stateless temporal
matrix fusion; real B×T grouping and causal chunk attention/GDN are separate
cuts. Preserve every independent session's root/leases/Ready/quarantine. Kernel
count is not runtime percentage; production CUDA graphs remain separately closed.
Benchmark pure model and OpenAI serving separately on qualified source/profile,
then actual optimized/default vLLM with matched caps/phase/policy/hardware.

## 11. Paused text artifacts that must survive ignored-run loss

ROOT is archiving textual drafts separately. This handoff does not copy code or
large model/tensor/cubin/state/reference payloads. The following roots contain
unique code/harness/source metadata, not just checklists; preserve their text
and full parent/additive relationships, excluding target/ and actual payloads:

- `.agents/runs/model-execution-hf-combined525-attestation-20261020/`.
- `.agents/runs/model-execution-hf-initial-prefix-20261020/` including pinned
  primary cumsum/AccumulateType/LICENSE source.
- `.agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/`,
  including followup-wrong-layer-v2.
- `.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/`.
- `.agents/runs/model-execution-hf-cached-gdn-contract-20261020/`, including
  all stride/H48/isolated replay derivatives.
- `.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/`.
- `.agents/runs/model-execution-qwen38-window-roles-20261020/`, including
  qualification/metadata-v2, plus `model-execution-qwen38-pure-wide-isolated-20261020/`.
- `.agents/runs/model-execution-native-cpu-provider-envelope-20261020/`,
  startup-v1, root-native-v3 and SIGTERM derivative; native-provider-probe run.
- `.agents/runs/numerics-and-context-{bulk-f32-primitive,bulk-f32-typed-slice,
  cpu-mapped-placement-control,use-ref-bulk-control}-20261020/`.
- Numerical reproduction tool roots
  `numerics-and-context-{long-context-audit,state-codec,bulk-causality,
  2b-bulk-causality,av-replay,2b-alpha-replay,compiled-f32-routes}-20261020/`.
- Fresh small TEXT raw-ID arrays/input manifest/generator/text under
  numerics-and-context-root-20261020. These are irreplaceable provenance if
  ignored storage disappears; they are not model weight/reference result payloads.
- Full ROOT isolated worktree diffs/untracked source text for
  `root-bf16-composition-20261020` and `root-default-bulk-20261020`.
- Historical resolver run in p12-qwen-window-resolver worktree (section6.1).

Other old isolated source trees still exist and must not be reset: p12-hf-axes,
bf16-dot,hf-binding,torch-gdn-reduce,hf-cached-gdn,hf-conv,hf-norm,hf-silu,
hf-softplus,hf-sigmoid,hf-typed,hf-raw-parameters,hf-ordinary-rms,
hf-ordinary-contract,tensor-window,parameter-window, plus serving/graph trees.
Their accepted changes are in MAIN where the committed plan says so; their
existence never proves an unmerged candidate was accepted. ROOT's source archive
and aggregate inventory decide exact preservation, not deletion/rebase here.

Adjacent paused witness work: primary loaded-SASS ABI b4b7f125 is source-reviewed
only; callback prototype5dc4e9b5 is frozen UNCOMPILED/UNREVIEWED/no runner/validator/
fixtures. It proves no numerical/instruction execution. Ordered-SiLU candidate's
204 operator cases pass, but60-case model/serving integration matrix was not
launched at pause. Do not consume its build as an accepted numerical baseline.

## 12. What a successor must not claim

- Fresh strict8K AR1/control success is not default bulk or historical replay.
- TinyBLAS first-entry evidence is not discovery of every cause/instruction.
- Acquired55GB source, metadata851roles, bounded windows or three correct
  operators are not27B inference/physical streaming/112-domain state parity.
- Native cached/gamma/wide operator passes in an isolated tree are not MAIN
  integration; imported source intermediates are not native producers.
- CPU64 arithmetic running on CUDA is not vLLM cuBLAS/FLA equivalence.
- Plain/observed original-framework equality proves instrumentation transparency,
  not native cross-engine equality; matched argmax/text cannot replace full tensors.
- Public native ABI arithmetic+provider-unloaded startup tests are not native
  internal allocation or physical resource admission.
- Old harness/serializer/free-null/cleanup mistakes do not establish model math
  bugs. Preserve each failed attempt and its additive fix without retagging it.
- No new source acceptance, execution, benchmark, fixed-bug claim or runtime
  authorization follows from this documentation-only handoff.
