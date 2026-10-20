# Model execution

Provenance commit: `6d74f76`.
Scope: P02/P03/P08/P11–P14/P17; CPU truth before GPU specialization.
Contracts: [large immutable sources](../IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md),
[Qwen basics](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md),
[Gemma family](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md).

## Uncommitted

- [x] **Original CPU attention fragment:** both authenticated sources at
  layer3, teachers11/353, source-36-F32-restored profile: all47104 raw BF16
  words across seven projection/view/QK-norm outputs match authentic original
  captures, including signed zero. Same HostMath/ForwardPlan and sealed source
  windows; real source-pool pressure, pre/mid-cancel, foreign input/provider,
  retained output after callers drop and final reservations0 pass. Explicit
  external hidden inputs establish no preceding producer or StateRoot.
  Focused16 tests, strict CPU/CUDA/Vulkan, both approved original release
  routes and complete isolated1243-test phase pass. Fresh helper strict build,
  full source scans and owned process/group/EOF cleanup also pass.
  Earlier metadata-list and omitted second4MiB binding-accounting failures
  remain preserved; corrections change only the replay harness. RoPE, KV,
  attention BMM/softmax, GDN/FLA, full-HF logits/StateRoot and27B stay open.

- [x] **Authentic vLLM projection operands:** read-only extraction from the
  accepted four-call eager-B1 capture preserves all original full496640-logit,
  320-source-audit, provider and cleanup guards. Six raw BF16 files,36992 bytes,
  bind actual teacher11/353 inputs/results, public GEMM pointers/settings and
  learned QKVZ(qkv,z)/BA(b,a) row hashes. Ten transport tests and independent
  rederivation pass; result f0981e62. No new model/GPU execution or dtype cast.
  Native BF16 arithmetic, opaque provider allocation admission and full-HF
  execution remain unqualified.

- [x] **Original2B paired attention observation:** fresh plain/observed CPU
  captures preserve all196 raw tensors/four full248320 logits/168 canonical
  domains/60 prior intermediates against the accepted two-profile reference.
  Observed capture adds216 genuine BF16 tensors across six attention layers,
  nine stages and teachers11/353 in both original loader profiles,688128 bytes.
  Exact signed-bit/shape/stride/source checks, unchanged archive validator,
  all632 source/617 loaded parameter audits and owned group/container/EOF
  cleanup pass. Ten parser fixtures pass. Diagnostic elapsed44.56s/42.01s;
  no benchmark or physical RAM admission claim. Result e45e1903 qualifies
  transparent reference-stage availability, not Rust attention output,
  full HF execution/StateRoot, GPU/FLA, long context or trained27B.

- [x] **Genuine pinned vLLM V1 eager B1 public GEMM observation:** four
  unchanged original0.8B layer0 QKVZ/BA projections at teachers11/353 use
  cublasGemmEx, BF16 A/B/C, FP32 compute, requested DEFAULT_TENSOR_OP,
  default math/host pointers/atomics disallowed, alpha1/beta0 and explicit
  8519680-byte workspace. Actual full tensor/workspace allocation extents,
  original provider versions/mapped SHA and loaded320 source roles before/
  after pass; plain/observed all496640 logit words are exact. Strict observer
  build and all three owned process/container/group/EOF cleanups pass.
  Original header-build and generic resolver failures remain preserved;
  additive exact already-loaded provider binding has no alternative lookup
  or math fallback. Prior resolver failure cause remains unproven.
  This records public requests, not hidden selected kernels, all42 state,
  wider/default graphs, Zerogravity GPU math/profile or matched performance.
  Qualifier result55d68c7a; explicit workspace alone does not qualify opaque
  library allocation admission. Pinned default vLLM loader/quant registries
  have no GGUF entry; original BF16 remains the matched-source route.

- [ ] **Independent trained BF16/FLA checkpoint gate:** original
  Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 acquired
  through the existing bounded Rust cache pipeline; independent31-file/
  18-shard full hashes and1199-tensor header/index checks passed.
  Source preparation only: no full27B RAM load or inference. Original
  Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc is now acquired
  by the same bounded Rust pipeline; independent official Git/LFS full12-file
  hashes,632 headers/320 text shapes,596BF16+36F32 tensors pass. Its4.548GB
  shard remains above the default4GiB complete-source guard. Genuine pinned
  official CPU two-profile reference capture now passes archive and owned
  cleanup validation: tokens11/353, four full248320 logits,168 canonical
  domains and60 actual intermediates. Official BF16 loading has617 BF16
  parameters; restoring only36 original F32 tensors gives581 BF16+36 F32.
  The profiles differ in47/49 raw tensors per step despite identical argmax;
  neither establishes Rust2B HF numerical qualification. First capture's
  cleanup-name bug is preserved; the repaired fresh run proves actual nonce
  container absence and process reap. Original Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17 pinned
  Transformers5.17/Torch2.13 CPU captures passed archive validation:
  tokens11,353, full248320 logits/all42 domains, actual layer0 stages.
  Keep official473-BF16 loader and source437-BF16+36-F32 profiles distinct;
  first-token argmax321 vs198. Explicit new CPU64 diagnostic vs source-mixed
  reference remains UNQUALIFIED: embeddings/direct layer0 norms and both
  QKV projections exact; layer0 convolution state exact,41/42 domains still
  differ. Full248320 logits differ214245/214285 words, maxabs .22119140625/
  .1875. First token exposes conv/SiLU signedzero; second exposes GDN state.
  Complete pinned convolution, chunk64/cached recurrent, coefficients and
  gated RMS arithmetic, then original GPU/FLA/state/streaming gates before enabling
  trained HF execution or27B serving. User authorizes required runtime/model
  acquisition/testing; dated source-free slice restrictions below are history.
- [ ] **Next trained reference gate:** qualify exact original HF BF16
  rounding/FLA order and independent source-trained logits/StateRoot;
  then bind the original profile to the existing per-op
  MathematicalBackend/SessionAuthority, with full physical
  device/SSD resource admission. No 27B weights, tokenizer or
  GPU launches were authorized by this source-free slice.

- [ ] **Following ME-P0 real trained integration:** fuse the
  accepted analytical original source operators into the ONE
  existing forward_token/MathematicalBackend and
  SessionAuthority execution schedule, qualify original HF BF16
  intermediate casts and FLA GDN accuracy, source-trained exact
  full logits and StateRoot, tokenizer, physical mutable-state
  memory accounting, cancellation and device completion. No
  SSD/physical inference claim follows from the diagnostic bridge.

- [ ] **ME-P0 next after diagnostics:** wire the analytically verified
  Qwen original operators into the ONE existing forward-plan and
  TaskDriver/SessionAuthority (no second token loop or state issuer),
  with exact source BF16 intermediate cast and original FLA GDN kernels,
  real trusted tokenizer and full 248320 logits/42-or-more StateRoot
  domains against the pinned checkpoint. Only then begin
  SSD-resident streaming and measured Qwen3.8-27B physical throughput.

- [ ] **ME-P0 next numerical stage:** source-exact original Qwen
  full attention, GDN/MLP, grouped Q/K→V replication, A_log/gate
  transformation and BF16 intermediate numerical schedule using the
  shared original TaskDriver/StateRoot. Qualify tiny independent
  analytic operator/state gates before real approved trained Qwen3.8
  27B original full-vocabulary reference. Measured original Qwen3.5
  2B CPU/GPU end-to-end performance also remains open.

- [ ] **ME-P0 next:** bind source-approved original Qwen3.8 27B
  tensor/index/config and tokenizer to the existing mathematical engine
  with authentic head/axis geometry, strict trained CPU full logits/
  StateRoot, then source-verifiable immutable SSD extents with stable
  leased memory and measured decode/prefill. First finish measured
  Qwen3.5 2B CPU/approved GPU end-to-end performance and
  cancellation/physical readiness. Preserve no model/SDK downloads
  and held device launch policies without explicit authorization.

- [ ] **ACTIVE NEXT: ME-P0 Qwen3.5 2B then Qwen3.8 27B.**
  Begin with original qualified Qwen3.5 .8B/2B CPU and resident GPU
  source/state oracles, inspect release source ownership and exact
  model-2B profiling on the permitted CPU/GPU devices. Keep tests
  bounded to existing 8K context now; 16K/32K extensions remain
  separate numerical gates. Qualify a stable single-model fast path,
  original tokenizer/math/KV/recurrent/StateRoot and actual device
  completion first. Then implement checked immutable source shard leases
  above the present 4GiB executable-source gate and Qwen3.8 27B
  complete trained CPU logits, including SSD streaming and quant-aware
  physical placement. Do not fetch weights or launch held Renoir/gfx90c
  hardware without the corresponding permission and source integrity.
- [ ] **NEXT after Qwen: ME-P1 Gemma4 E2B→26B-A4B→31B**, independently
  source-bind tensors/reference semantics, PLE/MoE/dense quantization
  and trained CPU→GPU correctness before measured throughput
  optimization. **ME-P2 Qwen3.8 Flash-Next** follows with separately
  verified learned n-gram embedding/4B MTP heads, QSA/GDN/MoE and
  explicit Q2/~3.5bpw/Q4 source quant choices (NOT guessed files),
  speculative n-gram as a distinct runtime optimization. Other OSS,
  ASR/TTS/image/video/decision execution and multi-model router remain
  on their own deferred source-real gates.
- [ ] Preserve direct one-model-per-process execution and one
  shared-instance-many-pipelines as **one Rust fabric**; the accepted
  YAML DAG/compiler and extension schemas add no inference backend.
  Do not resume context snapshot/provider/multimodal/router design
  work while ME-P0/P1 are open unless they genuinely block those models.

- [ ] Enforce new owner priority: Qwen3.5 .8B regression, Qwen3.5
  2B highest-perf trained CPU/GPU, Qwen3.8 27B with SSD
  streaming/owned shard tensors, Gemma4 E2B, 26B-A4B, dense 31B,
  then Qwen3.8 Flash-Next QSA/GDN/MoE/gated residual/MTP
  and 51B learned n-gram embeddings. Every additional DeepSeek,
  GLM/Muse/Mistral/Llama family has a typed NOT IMPLEMENTED
  source contract until distinct trained CPU and GPU gates pass.
  No subfamily is generalized from metadata-only source support.
- [ ] Treat single-model local GGUF execution and multi-model
  pipeline-qualified execution as two frontends of ONE TaskDriver/
  SourceIdentity/StateRoot fabric. A reusable model instance
  serves multiple routes without another model process or implicit
  source promotion; named replicas/adapters require separate
  physical memory admission.

- [ ] After the accepted source-acquisition v1 cutover, use the existing
  pinned HF index-first path to inspect explicit original Qwen3.8-27B and
  Gemma4 26B-A4B index/config/tokenizer/chat/processor identities
  (see [acquisition v1 handoff](../model-acquisition/README.md)).
  Select an authorized exact revision and admitted local host. Do not treat
  synthetic >51GB transfer-plan tests, public model pages or cached bytes
  as authorization, execution-ready source, or trained CPU parity.
  2026-10-18 tokenless live metadata-only HF source observation resolved
  Qwen3.8-27B to `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  (18 shards, 55,586,113,293 selected bytes) and Gemma4-26B-A4B-it
  to `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
  (2 shards, 51,644,338,194 selected bytes); Gemma4 E2B-it commit
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
  (1 unsharded safetensors, 10,278,848,001 selected bytes).
  No trained weights downloaded, locally owned, header-audited or
  model-execution-qualified yet.

- [ ] Acquire/bind authorized original config/index/tokenizer/checkpoint identities and choose a capable admitted host. Large BF16 sources exceed current local RAM; current executable source limit stays 4 GiB.
- [ ] Qualify real large (>4GiB) original trained shard payload on an
  authorized capable host using the already verified owned-shard
  implementation from 93a5622; independently measure resident
  allocation/physical RAM pressure and immutable lease lifetime. Next,
  develop the separately admitted SSD streaming extent architecture.
  The bounded source-only synthetic tests do not establish real large
  learned weight loading.
- [ ] Qwen3.8 27B: original HF grouped-head ordering and Q/K replication 16→48, explicit converted-profile permutations, untied head and source-derived norm/RoPE geometry.
- [ ] Gemma E2B/26B: shared GELU/multiply/scale/soft-cap, proportional RoPE, value norm/sliding/shared KV, E2B PLE/doubled FFN and 26B K=V/MoE branches; choose and qualify the exact reference-framework numerical schedule.
- [ ] Qualify actual trained CPU operators/full vocabulary/state/nonzero replay, official tokenization/templates/reasoning/stop behavior, sessions, rollback/checkpoints and resource refusal.
- [ ] Add the same source-bound forward plans and full physical GPU qualification per eligible device, preserving all state/actual-completion/cancellation guarantees.
- [ ] Keep distinct future heavy/hybrid recipes (Qwen Flash-Next, GLM, MiMo and source-defined QSA/MLA/KDA/indexers/hyperconnections), and broaden dense/encoder/pooling/reranking/LoRA support only through independent model/codec/backend gates.
- [ ] Preserve CPU-first family priorities; already-authorized parallel basics do not authorize model/SDK downloads or unsupported whole-model claims.

- [ ] Bind explicit original HF versus converted Qwen source profiles. Conversion permutes grouped/tiled V projections/constants/convolution/output, uses -exp(A_log) and adds1 to non-GDN norms; GDN swish is SiLU while full-attention gating remains sigmoid.
- [ ] Choose and independently qualify Gemma26B reference-framework routing/top-k/reduction/scale placement before production execution. Five analytic test-local routing cases are not trained source parity.

## Committed

### bfe62df — leased source-window execution

- [x] **Shared source-window HostMath fragment:** original27B851-role
  namespace and same PlanBuilder/HostMath embedding→ordinary RMS→full-column
  projection now execute genuine trained bytes without source materialization.
  Both tokens11/353 match all40,960 independent BF16-widened reference words,
  including full5120-wide embedding/norm and10240-row QKV. All18 shards/
 55,563,006,776B are verified through retained FDs. All80 authentic128-row
  windows preserve5120 input columns; maximum real QKV staging charge8,388,656B
  and raw norm staging4,194,328B are measured, not inferred from requested bytes.
  Each output retains admitted64KiB Output+960KiB Scratch and source issuer
  through last drop. Real stage-triggered mid-cancel, pre-cancel, pool pressure,
  foreign-source/config/index/token refusal and retained output comparisons pass;
  final reservations0/unfenced. Actual92.74s includes90.79s source scan;
 18,736KiB observed RSS is not physical free-RAM admission evidence.
  Fresh MAIN-linked original0.8B three-operator regression also matches all
 16,384 words/final source0. Focused17 artifact/1CPU/5model, strict CPU/CUDA/
  Vulkan/qualifier Clippy and full isolated1243 Rust/0failed/70ignored/248suites
  plus18Python/release/installer/Nix pass. Both approved original cache-direct
 17.22s/HF-import22.74s release routes pass fullV/all42/reclamation for both
  models. Only four pointless borrowed-value test drops needed repair. This is three pure state-free operators,
  frontier delta0: full112-domain27B logits/StateRoot and streamed inference
  remain unqualified.


### 5350a4d — typed ordinary math and source windows

- [x] **Typed ordinary RMS in the shared schedule:** the same PlanBuilder/
  HostMath embedding→raw InputNorm(0)→GdnQkv(0) fragment preserves source
  parameter identity, BF16 precision, epsilon1e-6 and explicit packing.
  Fresh MAIN replay matches all16384 genuine original0.8B reference words,
  two tokens11/353, with final source reservation0. Unaware providers and
  GPU plans refuse before callbacks/source admission; full HF session,
  forward and checkpoint guards remain closed.
- [x] **Sealed rank1 parameter windows:** BF16/F16/F32 raw ranges retain
  original full descriptors, issuer, opened source and actual staging charge;
  source/cancel/corruption/capacity/drop tests pass. Partial ranges never
  become complete TensorSource views. This is transport, not27B inference.
- [x] **Closed original BF16 row-window projection:** authentic original
  full-matrix descriptors and complete input columns reuse the accepted
  CPU64 dot without changing its arithmetic. Output-row partition, width5120,
  foreign source/range/dtype/nonfinite/cancel/provider and lease tests pass.
  No partial matrix/provider substitution or final BF16 cast; caller must
  admit private output/scratch. Same-HostMath27B integration remains open.
- [x] **Closed pinned F32 softmax:** source Vec8 maximum/sum order, unchanged
  SLEEF exponential and F32 reciprocal/multiply, width1..512. Genuine public
  full3599-case replay matches1838736 output words and3 refusals; authenticated
  default72-case golden preserves11280 expected words/3 refusals. NaN/+Inf/
  all-masked rows refuse. No legacy GGUF or full HF attention enablement.
- [x] **Composed acceptance:** focused core9/artifact16/CPU47/model6 pass,
  compact default5 pass; full isolated gate1241 Rust/0failed/70ignored/248
  suite runs plus18Python/release/installer/Nix passes. Strict composed CUDA+
  HF+CPUcache, Vulkan serving and CUDA qualifier pass; CPU-free HF binding8
  and GPU cold refusal2 pass. Both original cache-direct/HF-import RELEASE
  routes pass(18.88s/24.75s), full248320/all42 and final source reclamation.
  Full27B Rust reference comparison, streamed model state and CUDA/FLA remain
  open. Compilation/type and Clippy-only first failures were fixed; no
  mathematical tolerance or full-profile guard changed.


### c5eb1f8 — closed pinned sigmoid and ordinary RMS

- [x] **Closed pinned BF16 sigmoid:** explicit contiguous iterator spans,
  vector32 prefix plus the original scalar tail, private SLEEF and licensed
  Arm/glibc-FMA arithmetic. Genuine public Torch and Rust replay match all
  326753 output words, including all65280 finite BF16 values in vector/scalar/
  mixed call shapes and both actual original beta16 calls. No host exp,
  fitted output table, numerical-profile enablement or GGUF math change.
- [x] **Closed pinned ordinary RMS:** raw immutable rank1 BF16/F32 learned
  weights, exact BF16 input, source four-Vec8 reduction/carry order at widths
  256/1024/2048/5120, F32 division and runtime(1+raw weight), then final BF16
  cast after the F32 product. Actual original .8B122 norm calls preserve98
  prior full-logit/cache checks;32 public width/dtype fixtures cover the
  larger geometries. Rust matches all154 cases/1694 full stage fields/
  1680488 words exactly. Width fixtures do not qualify2B/27B inference.
- [x] **Composed source-free gate:** complete isolated check1218 Rust passed,
  0failed/69ignored/248 suite runs plus18Python, release/installer/Nix;
  ordinary RMS7 focused tests and strict CPU Clippy pass. Independent source
  review authenticates every6792 compact golden word from genuine public
  archives. Strict final composed CUDA+HF, Vulkan serving and CUDA qualifier
  Clippy pass; both approved cache-direct/HF-import release routes pass
  (20.09s/24.88s), both original models, full248320/all42, independent sessions,
  source leases and reclamation. Original-HF full session/forward/checkpoint
  guards remain closed; coefficient exp, initial chunk64, attention and
  physical model/state integration are still required.

### 2a32aff — typed source composition and vector softplus

- [x] **Typed original-source operator integration:** explicit program
  profile and activation precision, BF16 OutputInput embedding/projection
  contracts and raw SourceParameter metadata use the existing PlanBuilder
  and HostMath. Private operator slices carry no domains, logits receipt
  or frontier advance. Unknown physical providers and resident GPU plans
  refuse before source reads/admission. Raw learned parameters stay borrowed
  from the same owner; existing RootBinding descriptors distinguish initial/
  cached phase, BF16 conv/KV and FP32 recurrent axis layouts. These metadata
  views grant no freshness, state transport or publication capability.
  Complete original-HF session/forward/checkpoint guards remain closed.
- [x] **Genuine typed operator qualification:** pinned public Torch
  F.embedding for original tokens11/353 followed directly by independent
  GdnQkv(0), FfnGate(0) and tied Output projections matches every Rust bit,
  including all248320 words of both output projections. Eight reference
  payloads preserve518144 values; same loaded source, raw matrix pointers,
  two-node recorder, zero initialized state and final source reservation0
  are checked. No input normalization, complete layer/model logits or
  GPU/27B inference is claimed by this pure operator composition.
- [x] **Closed pinned FP32 vector softplus:** source beta1/threshold20,
  finite general-F32 contiguous spans divisible16, explicit SLEEF3.8
  compensated log1p/exp/FMA order and BSL notice. Actual32 original biased
  alphas and28464 adversarial F32 values, reordered values and16/32/48/64
  spans match public Torch bit for bit. No scalar-tail, MKL coefficient
  exp, general profile or default GGUF arithmetic change.
- [x] **Typed/softplus composed gate:** full isolated check1213 Rust
  passed/0failed/69ignored/248 suite runs plus18Python, installer/release/Nix;
  focused core/CPU/model158 passed/2ignored, CPU-free binding10 passed,
  resident cold refusal2 passed. Strict composed CUDA+HF, CPU-free HF,
  Vulkan serving and CUDA qualification Clippy pass. Both approved original
  cache-direct/HF-import release routes pass (17.15s/23.12s), both original
  models, full248320/all42, independent sessions/source lifetime/reclaim.

### cadcbd9 — closed pinned original-HF CPU operators

- [x] **Closed pinned CPU original-HF operators:** source-bound BF16
  four-tap convolution, FP32 width128 cached GDN, raw-F32-gamma gated RMS
  prefix and contiguous vector SiLU. No model/session/initial-chunk permit;
  default GGUF/scalar profiles stay unchanged. Convolution preserves four
  independent FMA partials followed by direct adds and BF16 output. Cached
  GDN uses direct query division by sqrt128, actual key-major FP32 state,
  materialized key reductions and separate update multiply/add; observed
  decay is an explicit input. RMS casts normalized core before raw original
  F32 gamma, retaining the FP32 product. SiLU ports pinned SLEEF3.8 AVX2
  exp/FMA/scaling and FP32 division with source attribution/license, only
  finite exactly BF16-widened contiguous inputs with length divisible16.
  Twenty-eight focused Rust tests pass including old projection/reducers.
- [x] **Actual closed-operator Rust qualification:** untouched pinned
  Torch2.13 public SiLU with observed SLEEF AVX2 provider supplies exact
  outputs for ALL65280 finite BF16 inputs plus4096 actual source gates;
  Rust matches every F32 bit. Full16-head cached replay matches262144
  state words plus5*2048 normalized/scaled/core/output words. Gated RMS
  prefix matches8256 words across both actual tokens/all16heads.
  Convolution and composed vector-SiLU/BF16 match all6144 channels at
  four actual causal positions (49152 output words). These are pure
  operator replays of genuine source operands, not full-model/profile,
  GPU/FLA or27B numerical acceptance. Initial chunk64, coefficient exp/
  sigmoid/softplus, ordinary RMS and eager attention remain open.
- [x] **Composed verification:** complete isolated check1202 Rust passed/
  0failed/69ignored/248 suite runs plus18Python/release/installer/Nix.
  Strict CUDA+HF-binding+portable, Vulkan serving and CUDA-qualification
  Clippy passed; CPU-free original binding6 and CPU serving22 passed.
  Both approved original0.8B/2B cache-direct/HF-import RELEASE routes
  retain all248320 logits/all42 domains, source lifetime and reclamation.
  Independent long-context GGUF GDN divergence remains a separate required
  gate in [numerics-and-context](../numerics-and-context/TODO.md).

### 9815504 — original HF roles and closed CPU projection/reducers

- [x] **Original HF source roles and closed CPU profiles:** authentic rank/axes,
  raw source dtype and tied/untied heads bind to the same loaded owner;
  Safetensors output/input rows remain untransposed. Typed original-HF
  metadata carries the actual config/index/bundle and independent state
  geometry. Its execution/session/checkpoint-v1 entry points refuse before
  allocation/backend/I/O; metadata is not an executable-model promotion.
  Explicit Torch CPU AVX2 BF16 projection diagnostics retain the pinned
  64-lane FMA/reduction order and separate state/trace schema; the original
  scalar/GGUF profiles remain distinct. Actual original .8B layer0 QKV
  is exact6144/6144 for each of tokens11,353. Closed FP32 width128 L2 and
  key-axis reducers pass authentic compact bitwise goldens and adversarial
  head/order/finite/overflow cases. Real attention-input norm observations
  now cover both full and GDN layers. Focused composed suites185/0/2ignored,
  CPU-free binding8/0, strict CPU/CUDA/Vulkan/qualifier Clippy and full
  isolated check1164/0/69ignored/245suites+18Python passed. Both approved
  original .8B/2B cache-direct/HF-import release routes remain exact full
  logits/all42. Full original HF mathematics remains UNQUALIFIED.


### f26232e — authentic tied original output heads

- [x] **Original HF tied output head:** verified `tie_word_embeddings`
  selects the original embedding or independent `lm_head` projection;
  no alias tensor or fallback source. Source-free authenticated tied/untied
  fixtures test opposite-sign independent heads, exact F32/BF16 diagnostic
  logits, immutable source identity and final lease0. Full isolated gate:
  1148 Rust passed/0 failed/69 ignored/245 suites,18 Python. Both approved
  original0.8B/2B cache-direct/HF-import release routes retain exact full
  logits/all42. This fixes the diagnostic head; no trained HF promotion.

### 1c0bb41 — pinned numerical reference seam

- [x] **ME-P0 pinned original numerical reference gate:**
  extend one immutable source owner with canonical source-bundle
  SHA256 over pinned index and all authenticated shard bytes.
  Capture a bounded subset of original Qwen GDN, attention, FFN
  and logits through shared PlanObservation, with opt-in 32MiB
  private trace and a single selected GDN recurrent state layer.
  Strict typed independently SHA256-pinned JSON comparisons bind
  config/index/bundle, exact stage order/geometry/position/dtype,
  detect first mismatched element, and distinguish truly bit-exact
  BF16 output from bounded, tolerated FP32 state deviations.
  Source-free adversarial tests verify same-index/different-weight
  digest changes, duplicate/missing/reordered reference content,
  bad token/source/dtype, selected GDN state and immutable roots.
  Fix full-model BF16 residual result-cast omission by reusing
  its source-typed existing residual binary operator.
  This stage DOES NOT prove independence merely because a
  reference JSON asserts trained-framework origin.
  Complete isolated scripts/check.sh passed 1,135 Rust tests, 0 failures,
  69 intentionally ignored across 238 suites plus 18 Python fixtures,
  release CLI/installer and CPU numeric comparators. Strict default and
  composed CPU/cache/Qwen/acquisition Clippy -D warnings pass.
  Independent approved Qwen3.5 .8B/2B release cache/direct 17.45s and
  HF-import 21.97s preserve exact full 248320 logits, all 42 StateRoot
  domains, two-token independent sessions and source/lease correctness.
  Offline Nix source-integrity passed with the checked trace files.
  No trained Qwen3.8 reference or Torch/FLA runtime was accessed.

### 6d74f76

- [x] Qwen3.8 27B and Gemma4 E2B/26B configuration, tensor and source-index contracts; no new trained execution, tokenizer or checkpoint loading.

### Source contracts — 023703f

- [x] Qwen3.8 metadata/shard roles and CPU-free Gemma4 family contracts,14 Gemma tests and official metadata/config cases passed within metadata scope. No tokenizer, new learned weights, physical shard-header binding or trained inference follows.
- [x] Preserve Qwen/Gemma source axes, metadata precision and routing/KV facts in the [model semantics reference](../../docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md) and [Gemma technical contract](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md). The executable-source4GiB cap remains.

### 93a5622 — original verified immutable source shard owner

- [x] **ME-P0 LS-B01–B05, verified owned sharded-source allocation:**
  implement `SourceWeights::prepare_verified_shards` in the existing
  artifacts owner with exactly opened original Safetensors file
  descriptors, caller-expected index SHA256 and whole-shard SHA256/length,
  one original SafetensorsIndex parser and checked header name→shard,
  dtype/shape/offset/payload binding. Preflight limits and reserve whole
  captured file allocation plus a conservative metadata envelope via the
  original caller-attested host `ResourceArbiter` before copying. Reparse
  exact owned bytes, compare authenticated headers against preflight
  (including same-length mutation) before issuing the single original
  SourceIdentity. Arc owner lifetime pins all original slices and RAM
  charge; default GGUF `SourceWeights::open` keeps the 4GiB cap and
  original Qwen execution path.
  10 source-only integration tests pass: positive F32/F16/BF16, complete
  index closure, foreign tensor metadata rejection, wrong SHA/length,
  missing/extra/duplicate/wrong shard, same-descriptor pathname
  replacement, original inode mutation, verified copied immutability,
  multiple admitted/shared original leases and exact resource rollback.
  Sparse >5GiB header tests refuse before reads under both 4GiB
  default and physically insufficient explicit >4GiB budgets.
  **No actual >4GiB weight load, SSD streaming, trained original
  Qwen27B/Gemma math, tokenizer or physical GPU support is claimed.**

### b796a2e — original Qwen text source geometry and CPU value binding

- [x] **ME-P0 original Qwen3.8 source/value binding (small original
  format, not trained 27B):** bind caller-authenticated original
  QwenTextConfig SHA256 and pinned indexed-source SHA256 on the *same*
  immutable SourceWeights allocation. Check every ordinary-text
  name and original [output,input] tensor shape/encoding, including
  the untied output head; distinguish source extra vision/MTP from
  unregistered text roles. Reuse the existing CPU source MatVecRegion
  for HF packed F32/F16/BF16 output-first matrices with no transpose,
  plus the existing embedding/parameter-vector decoder through a
  source-bound QwenOriginalCpu façade. Small complete 4-layer hybrid
  original-format integration tests and negative hash/shape/missing-
  head/source-provider tests pass. Oversized HF source matrices
  refuse before the optional accelerator is invoked (observed no-dispatch).
  The complete frozen isolated
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  passes 1,103 Rust tests, 0 failures, 69 ignored across 238 suites,
  plus 18 Python namespace fixtures, strict lint, installer/Nix
  formatting and CPU numeric selftests.
  Both independent Qwen3.5 0.8B/2B original trained
  source-parity release routes pass AFTER the final safety guard:
  cache/direct 16.70s and HF-import 22.88s, each full 248320-vocabulary
  logits, all 42 StateRoot domains, two tokens, independent sessions
  and lease/mutation/reclamation exact. Default and
  acquisition/cache-CPU strict `cargo clippy --locked --offline
  --all-targets -- -D warnings` checks also pass. The original
  GPT-family GGUF CPU and session path is unchanged.
  Does NOT run trained 27B, create a Qwen session, implement
  grouped GDN permutations, or stream SSD.

### b11e58c — original source-bound Qwen analytical one-layer proposals

- [x] **ME-P0 original Qwen3.8 analytic one-layer operators:**
  source-bound ordinary zero-centered RMS gamma (1+weight),
  HF head-interleaved full Q/gate projection, per-head Q/K norm,
  source-configured partial RoPE, 24Q/4KV grouped lookup,
  QK/softmax/AV/sigmoid/out projection and candidate KV slabs under a
  source-and-layer-bound immutable `OriginalFullPrefix`. Source GDN
  four-tap causal conv and SiLU, exact 16→48 repeat_interleave of
  original Q/K, per-head L2, stored A_log→-exp(A_log),
  dt_bias/softplus/sigmoid beta through the existing coefficients and
  recurrent step, output GDN gated norm/out projection. An immutable
  `OriginalGdnPrefix` holds exact source/layer and private tentative
  conv/recurrent state, never replacing SessionAuthority. Source FFN
  post-attention RMSNorm, SwiGLU and residual reuse old math.
  Small independently derived analytical checks include zero-centered
  norms, 16→48 head signed-zero mapping, 24Q/4KV GQA six-query
  grouping, ln2 decay beta0.5/gain0.5, full attention gates/partial
  RoPE, GDN two-token private repeat/branch, source identity mismatch,
  finite and dimensional errors. The original 8K diagnostic bound and
  original default GGUF behavior remain. **No trained HF BF16 intermediate
  rounding, full layer/model StateRoot, tokenization, actual 27B learned
  weights or GPU/SSD support claimed.** Strict source-config-specific
  numerical reference and official checkpoint tests remain open.
  Final isolated `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4
  python3 scripts/native-gpu.py -- bash scripts/check.sh`
  passed 1,111 Rust test executions, 0 failures, 69 ignored across
  238 suites, plus 18 Python namespace fixtures, release CLI/installer
  and CPU numeric-comparator scripts; strict default and composed
  CPU/cache/Qwen/acquisition Clippy `-D warnings` passed.
  Actual offline Nix source-integrity passed. Two independently run
  Qwen3.5 .8B and 2B trained-source release routes remain exact:
  cache/direct 17.36s and HF-import 22.49s, each original complete
  248320 logits/42 StateRoot domains/two tokens/source leases.
  This diagnostic-only milestone is recorded in signed commit b11e58c.

### 6329cff — source-bound Qwen one-layer TaskCoordinator

- [x] **ME-P0 source-bound original Qwen StateRoot admission
  (one-layer diagnostic only):** the sole existing TaskCoordinator
  now supports an opt-in original immutable source identity plus
  a unique coordinator/session ID while retaining its generic
  publication API. A stateless source-bound one-layer bridge maps
  canonical ForwardPlan Conv/Recurrent/Kv StateSlots to versioned
  GDN or full-attention StateRoot domain bytes, with exact model,
  epoch/session, layer, root, schema and frontier verification.
  Immutable source-pinned numerical proposals publish via the
  existing TaskCoordinator only after complete finite-domain
  preparation, 8K token/64MiB diagnostic caps and early next-KV
  admission. Independent tests cover GDN two-domain and causal
  attention KV commit, abandoned and stale competing steps,
  previous immutable roots, corrupted/nonfinite state refusal,
  source/session/schema mismatch and generic coordinator parity.
  There is NO separate multi-layer token loop, model scheduler or
  new state issuer; this is not yet trained BF16 reference parity.
  Final isolated full scripts/check.sh passed 1,116 Rust tests,
  0 failed, 69 intentionally ignored across 238 suites plus
  18 Python namespace fixtures, release CLI, installer, shell/Nix
  formatting and CPU comparator selftests. Strict default and
  composed CPU/cache/Qwen/acquisition Clippy -D warnings passed;
  independently repeated Qwen3.5 .8B/2B release cache/direct
  17.11s and HF-import 22.27s retained exact 248320 logits,
  42 StateRoot domains, two tokens and source lease behavior.
  Offline Nix source-integrity passed before signing.

### e43ba4c — original Qwen full-model FP32 analytical forward

- [x] **ME-P0 full original-HF model analytical forward:** extract
  the one Qwen layer traversal from the existing GGUF forward_token,
  preserving its complete MathematicalBackend operator order.
  Use it also for original source-bound GDN/full-attention/FFN,
  each layer's private Conv/Recurrent/Kv state and residual output,
  and final RMSNorm/untied head. One opt-in TaskCoordinator
  owns **all** model state domains, not one root per layer.
  Validate all source identities/schema/finite bytes, 8K/128-layer/
  128MiB diagnostic bounds and complete logits before atomic
  publication. Tests: complete synthetic four-layer original
  checkpoint with seven state domains/two tokens, matching
  independent one-layer GDN, abandoned/competing/foreign session
  and late-corrupted state refusal, derived final logit ~8, and
  shared GGUF traversal ordering/fail-fast. The method is explicitly
  named `prepare_diagnostic_token_f32`: this is **not** trained BF16/
  FLA inference, production TaskDriver or an alternate runtime.
  Complete isolated scripts/check.sh passed 1,119 Rust test
  executions, 0 failed, 69 intentionally ignored across 238
  suites, plus 18 Python namespace fixtures, release installer/CLI,
  Nix/shell formatting and numeric comparator selftests.
  Strict default and composed CPU/cache/Qwen/acquisition Clippy
  -D warnings passed. Both independent original Qwen3.5 .8B/2B
  trained-source release routes preserve exact 248320 full logits,
  42 StateRoot domains, two tokens and source/session leases:
  cache/direct 17.88s and HF-import 22.59s, both exit0.
  Offline Nix source-integrity passed before the signed commit.

### aaf9ba6 — original Qwen BF16 source dtype diagnostics

- [x] **ME-P0 explicit original BF16 dtype diagnostic:**
  implement finite nearest-even BF16 rounding with signed-zero and
  overflow refusal; introduce optional BF16 source stages into the
  existing original Qwen source binder/CPU operators, without
  mutating the default FP32 path. Cast after ordinary FP32
  (1+gamma) RMSNorm, per-head Q/K norm, HF output-first
  projection, SwiGLU, residual, RoPE, BF16 QK matmul *and*
  scaling, post-FP32 softmax, attention value and gates.
  GDN follows BF16 conv/SILU, FP32 A_log/softplus, BF16 beta,
  FP32 recurrent state and BF16 core output, source gated RMS
  casts before gamma and at gated output. StateRoot schema
  variants disallow cross-dtype replay even for one allocation.
  Two-token four-layer synthetic BF16 whole-model integration
  plus source-independent NumPy BF16 bit-golden RMSNorm and
  gated RMSNorm, FLA two-step recurrent output/FP32 state
  oracle, sign/tie/overflows and F32 parity are required.
  This is explicit BF16 **diagnostics** only: NOT trained
  reference equivalence, a new model-run route or GPU/SSD.
  Source-independent NumPy BF16 goldens match exact bit patterns
  for ordinary/gated RMS and two-step GDN returned cores, with
  FP32 recurrent state tested within 2e-7. The full isolated
  scripts/check.sh gate passed 1,128 Rust test executions with
  0 failures and 69 intentional ignores across 238 suite runs,
  plus 18 Python fixtures, release installer/CLI and CPU numeric
  comparators. Rerunning on the final implementation still
  passes 1,128/0 across 238 suites. Both independent Qwen3.5
  .8B and 2B trained release routes retain complete 248320
  logits, all 42 StateRoot domains and source lease/mutation:
  direct16.26s and HF-import21.78s on the final source.
  Strict default/composed Clippy and Nix source-integrity pass;
  production model run is not enabled.

## Deferred — after Qwen/Gemma and Flash execution

- [ ] Qualify independently selected NanoJev-like calibrated decision
  model sources under the same model/source/cache/residency authority.
  Register finite choice/boolean/score or include/omit heads without
  assuming a specific research checkpoint. Validate calibrated
  probabilities and per-question source-bound state, batching, thresholds,
  accepted/rejected context records and decision-fallback semantics
  before enabling generic context-filter execution; pipeline check alone
  provides only typed static intent.
