# Serving and speculation

**24 Nov delivery order:** the [full implementation plan](../FULL_IMPLEMENTATION_PLAN.md)
sets phase priority. Shared foundations (including optional compiled-provider
scaffolding), CLI/TUI/Web and serving, ROCm, CUDA, Vulkan, then concrete
compiled providers. Native Hugging Face/Safetensors and GGUF are primary.
Older ordering below is context; valid owner constraints remain authoritative.

## Current priority — correctness and hardware-native throughput

24 Nov: explicit rocBLAS native resident execution now runs real source
models through the existing shared scheduler/root owner. Projection-only
C8 throughput improves to 30.978/23.737 tok/s for 0.8B/2B; 27B C1 1.979 tok/s.
The retained vLLM comparison still leaves a substantial gap; no new paired
vLLM result. Native GDN dense products now also use the owned rocBLAS provider, reaching
39.778/29.088 tok/s at C8 in V1. Shared compact dense GEMM/solve
geometry and borrowed HIP command adapter now support native V2 GDN
triangular solve, reaching43.638/30.949tok/s at C8. Next: actual native
row-wise/attention regions and large dispatch amortization, guided by
the current whole-model profile. Prefer mature tuned standard operators wherever their
contract fits; specialize model-specific hot paths. See
[model execution priority reset](../model-execution/TODO.md#priority-reset--24-nov-native-matrix-execution)
and CURRENT_BASELINE. Preserve exact-reference serving and reusable kernels.
CUDA fused lane correctness and production cuBLAS integration are outstanding;
Vulkan cooperative-matrix provider reuse is a separate measured work item.

## Prior context — correctness, HIP batching, measured hot paths

[Fabric transition](../fabric-vnext-transition/TODO.md) S0–S3/F0–F4 and scoped
acceptance are complete on source`8c43844`; ordinary multi-row HIP landed
`bb400c4`. Preserve those accepted owners/numerics/lifetimes; do not repeat the
migration. Populated0.8B/2B HIP correctness and its final gate pass. Finish
remaining populated CUDA/general27B, then configurable native continuous batching:
independent queue/active/member/token limits, including all eligible admitted
work when it fits. Current offered-C64/active≤8 singleton coverage is not native
batching. The matched vLLM provider is now container-free and Nix-native on L5; paired
performance still follows provider correctness and the retained serving contract.
Recurrence`17aafa71` stays cancelled; no new acquisition or production permit.


Provenance commit: `6d74f76`.
Scope: P09/P12/P15/P16/P20/P22.
Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md),
[36 serving gates](../../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md),
[Qwen parity and decoding](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md).

## Uncommitted

### OA-1 — async serving transport (26 Nov; after UX-P)

- [x] OA-1A: mature bounded text HTTP/1 substrate over original service,
  admission and event owners. Focused serving 25/0/0 and engine 110/0/0,
  strict lint; actual cached CPU CLI/native/Firefox listener lifecycle with
  model list and both alias completions on one persistent TCP connection
  passed (55.52s). Real SDK listener/owner lifecycle passed (8.83s).
  Assembled offline private-network gate passed (819.3s), 2,538/0/109 across
  318 Rust suites, release TUI/Web, installer and scripts.
  No numerical/GPU performance comparison. See the
  [OA-1 design](NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md#oa-1--async-transport-substrate).
- [x] OA-1B primitives: shared decoded-body reader with charged fixed/unknown
  backing, absolute deadline, cap and trailer refusal; current fixed-length
  text path migrated. Actual private TCP chunked fixture verifies growth,
  empty/exact/overflow/deadline and full charge retirement. Cached
  tokio-tungstenite 0.29.0 owned upgrade path over the original connection
  and wire budget, bounded frames/reassembly/direct message IO, auth and
  adapter admission before switching. Actual internal test-adapter sockets
  verify authorization, invalid handshakes, frame/message and wire budgets,
  slot retention, ping/pong, close, disconnect, stop and ledger retirement.
  No public Realtime capability or modality route. Focused strict lint and
  serving 31/0/0 passed. Actual cached CPU CLI/native/Firefox lifecycle
  1/0/0 (60.72s) and SDK listener/model lifecycle 1/0/0 (10.33s).
  Full isolated workspace gate passed (466.3s): 2,544/0/109 across
  318 Rust suites, release TUI/Web, installer and scripts.
- [x] OA-1B public chunked text: pinned cached Hyper 1.13.0 server-parser
  patch refuses mixed TE/CL and repeated CL before normalization; exact
  upstream bytes except role.rs and local patch notice. Same charged
  reader, authorization, JSON/SSE, persistent connection and listener.
  Actual public socket tests verify exact cap, UTF-8 split, complete-body
  admission, pipeline, bad framing, unauthorized/oversize/trailer refusal
  and partial disconnect charge retirement. Focused serving 33/0/0 and
  strict lint pass; real cached CLI/TUI/Web (60.52s) and SDK (9.11s)
  regressions pass 1/0/0 each. Assembled HIP/Vulkan/CUDA compile
  boundaries, full-default/CPU strict lint, GPU boundary tests 26/0/2,
  source-only regressions and dependency isolation passed. Initial gate
  timed out at 1100s compiling CPU workspace tests. Full isolated retry
  on 08a9d30e exited 0: 2,546/0/109 across 318 Rust suites, release TUI/Web,
  installer and scripts. Nix source filter/check updated and syntax passes;
  exact locked source-integrity derivation blocked by four unavailable
  flake source hashes offline. No new physical GPU qualification.
- [x] OA-1B multipart: acquired only pinned multer 3.1.0 with index
  checksum verification; other dependencies cached. Vendor strict initial/
  final delimiter, decoded EOF and duplicate part header policy in its
  existing state machine. Reuse original bounded reader, shared charged
  backing, aggregate/per-part/count and metadata bounds, absolute deadline
  and cancellation ownership. Private real socket tests verify complete
  forms and rollback including real host-budget refusal; ordinary server
  refuses media and non-JSON text input. Focused 38/0/0, strict lint,
  architecture passed. Real cached CLI/TUI/Web and SDK passed (76.57s
  and 14.07s). Full isolated gate exited 0 (902.1s): 2,551/0/109 across
  318 Rust suites, release TUI/Web, installer and scripts. Exact Nix
  source-integrity remains separately unverified. No second parser, media
  capability or new scheduler. OA-1 accepted.
- [x] OA-2: core typed text intent and engine SemanticRequest retained
  by original queue/worker; legacy single text and SDK/CLI converge.
  HTTP no longer frames Qwen prompts. Model-owned validation, exact
  rendered byte and prompt/scratch bounds, shared Qwen renderer, original
  typed ResponseEvent/receipt. Preserve configured rendered chat cap;
  typed single-user chat also rejects role delimiters; raw unchanged.
  Legacy-only service/tokenizer adapters refuse structured chat rather
  than flatten. Focused engine/serving 163/0/0 and strict lint, tokenizer
  suite 7/0/0 and actual cached approved-tokenizer test 1/0/0 passed.
  Actual cached CLI/TUI/Web and SDK regressions passed (67.77s and
  15.85s); full isolated assembled gate 2,553/0/109 across 318 Rust
  suites (878.6s), release TUI/Web, installer and scripts passed.
  No Responses, arbitrary Jinja, new modality, model owner or scheduler.
- [x] OA-3: Models list/retrieve accepting aliases, stateless Responses
  text create/stream directly on SemanticRequest and original receipt.
  Typed text items/instructions, bounded metadata/greedy controls,
  charged bounded projection and exact restored prefix usage. Canonical
  ordered SSE completed/incomplete/failed, server request/response/item
  IDs, named parameter refusals; preserve legacy metrics stream ID.
  Focused 167/0/0 and strict lint passed. Cached official OpenAI SDK
  2.33.0 strict actual Qwen3.5 Models/Responses JSON/SSE and Chat parity
  passed; real native/browser/official SDK (80.88s), Rust SDK (15.02s)
  and full isolated gate 2,557/0/109 (318 suites, 617.5s), release
  TUI/Web, installer and scripts passed. Store defaults false locally;
  explicit persistence/continuation, tools, non-text and unsupported
  controls refuse. OA-3 accepted.
- [x] OA-4A: foreground store/retrieve/delete/input-items pagination and
  previous_response_id. Listener-local memory-only charged quota 64
  pending/retained slots, 32 MiB conservative reservation, 30min expiry
  and final-reader credit retirement. Normalized text history excludes
  prior instructions; actual assistant output replay, no parent chain
  or model-state retention. Foreground cancellation explicitly refused;
  disconnect cancels original receipt. Focused 170/0/0 and strict lint
  passed. Real installed SDK 2.33.0/CLI/TUI/Web (134.83s) and Rust SDK
  (15.18s) passed, including exact Chat parity and deletion. Full
  isolated gate 2,560/0/109 (318 suites, 690.5s), release TUI/Web,
  installer and scripts passed. OA-4A accepted.
- [x] OA-4B: bounded joined nonstream background response execution,
  independent of create connection, polling and explicit original receipt
  cancellation, terminal race/idempotence, safe failure fallback and
  joined shutdown. Same quota and final reader ownership, hard 64 task
  handles. store=true 30min continuation, background store=false 10min
  temporary polling only; foreground unchanged. Default/soonest window.
  Background stream/replay refused. Focused 172/0/0, strict lint passed;
  real installed SDK/CLI/TUI/Web (130.88s, actual cancelled) and Rust
  SDK (14.36s) passed. Full isolated gate 2,562/0/109 (318 suites,
  377.3s), release TUI/Web, installer and scripts passed. OA-4B accepted.
- [x] OA-4C: conversation/item resources and atomic context append.
  Same protocol quota and original semantic admission; exact item IDs,
  bounded pages, exclusive mutation lease, terminal append/cancel race
  and independent stored response snapshots. Focused 177/0/0 and strict
  lint passed. Real official SDK 2.33.0 and cached CLI/TUI/Web lifecycle
  passed (124.91s) after serial retry; Rust SDK passed (15.56s). Full
  private-network assembled gate 2,567/0/109 (318 suites, 342s), release
  TUI/Web, installer and scripts passed. OA-4C accepted. See design and
  current handoff for local bounds, retention and retry details.
- [x] OA-4D: exact input-token counting through the original charged
  selected-model tokenizer and shared semantic/rendered input bounds.
  Read-only committed conversation snapshots, previous-response replay,
  original registry reader, no generation queue ticket/session/output/
  prefix or metrics. Native Runtime and official Responses count endpoint.
  Focused 179/0/0 and strict lint passed; real cached Qwen3.5 official
  SDK 2.33.0/CLI/TUI/Firefox (162.61s), Rust SDK (18.79s) passed.
  Full private-network gate 2,569/0/109 across 318 suites (783s),
  release TUI/Web, installer and scripts passed. OA-4D accepted.
- [x] OA-4E implementation: bounded background streaming/replay and
  resumption over the existing joined job/receipt, shared typed SSE
  projection and charged immutable replay, exclusive sequence cursor,
  atomic terminal/context publication. Focused 182/0/0, strict lint,
  actual installed OpenAI SDK 2.33.0 stream/resume/cancel and Rust SDK
  lifecycle (19.96s) passed. No duplicate scheduler/model state.
  Full assembled private-network gate 2,572/0/109 (318 suites, 548s)
  passed, including release TUI/Web, installer and scripts.
- [x] OA-4E final acceptance: serial real resident Firefox/CLI/TUI
  lifecycle revalidation passed (83.29s) on unchanged 37723346.
  Earlier Firefox model-load timeout did not recur; no weakened
  admission or timeout. OA-4 accepted.
- [x] OA-5 text compatibility implementation: typed Chat text parts,
  neutral controls, route-specific validation and named refusals;
  preserve original raw legacy request and token metrics. Shared
  checked usage and finish reason, exact cached/total JSON/SSE counts.
  Focused 184/0/0 and strict lint passed.
- [x] OA-5 acceptance: actual cached official SDK 2.33.0/CLI/TUI/
  Firefox (184.78s) and native Rust SDK (13.99s) passed. Full isolated
  gate 2,574/0/109 (318 suites, 306s), release TUI/Web, installer and
  scripts passed. Earlier admission and fixture-only SDK construction
  failures resolved without production changes; see live handoff.
  Capability-bound tools/structured output/media/non-greedy sampling
  remain explicit unsupported until real implementations are qualified.
- [x] OA-6A: native BERT sentence encoder candidate and shared CPU
  noncausal math; verified single-file Safetensors owner, I64 index
  storage (never a floating projection), original WordPiece source
  semantics, bounded admission and result lifetime. Independent
  synthetic source/vector and ownership tests pass (489/0/16 focused);
  default CLI compile passes. Earlier interrupted assembled gate
  1,844/0/95 (255 suites) retained as history.
- [x] Full isolated gate on unchanged OA-6A 6a5e3518: 2,581/0/109
  (319 suites, 580s), acquisition/release/installer checks passed.
- [x] OA-6B implementation: native embedding request/ordered charged
  vector batch and generic existing receipt, one registry/FIFO/worker
  pool with shared active slots and original retirement. Actual verified
  synthetic BERT SDK plus mixed scheduling, cancellation, accounting
  and loopback refusal pass. Focused 246/0/1 (26 suites), strict lint.
  Explicit candidate feature; no trained-model or public eligibility.
- [x] OA-6B full workspace acceptance: 2,584/0/109 (321 Rust suites, 834s), complete isolated gate passed.
- [x] OA-6C wire implementation: original registry public-capability
  gate (default false), original typed receipt, strict four input forms,
  fixed-width controls, float/LE-F32 base64 and exact usage; bounded
  projection, cancellation, catalog and changed-width admission refusal.
  Actual source-bound synthetic BERT/installed OpenAI SDK 2.33.0 pass,
  focused 246/0/1 (26 suites), strict lint. Candidate remains native-only.
- [x] OA-6C workspace gate: 2,585/0/109 (322 Rust suites, 575s), complete isolated gate passed.
- [ ] OA-6 final trained-model acceptance: authorize/acquire only pinned
  MiniLM-L6-v2 original Safetensors (~90.9 MB) and exact config/tokenizer/
  vocab/module closure, qualify actual learned vectors and tokenizer IDs
  against independent reference and real memory/cancellation. Bind
  qualified built-in loader/public eligibility and rerun actual SDK.
  No fabricated embedding or separate execution owner. OA-6 remains open.
- [x] OA-7A: cached original Gemma4 E2B native audio frontend, complete
  12-layer learned USM encoder and language projection (F32 reference,
  BF16 coefficients), reusable CPU FFT/conv and original source/credit.
  Actual independent Transformers/PyTorch parity: 4/16 x1536 rows,
  max abs 2.5749e-5/4.3988e-5, relL2 1.039e-6/1.594e-6; log-mel
  max abs 4.77e-7, cancellation and ownership pass (28.20s). Focused
  125/0/13, strict lint. No download, ASR or public audio eligibility.
- [x] OA-7A assembled workspace gate: 2,587/0/110 (322 Rust suites, 570s), complete isolated gate passed.
- [x] OA-7B: one original authenticated window source for audio/language,
  bounded original scalar/conv parameters, same family traversal/StateRoot
  audio prefill and cached decode. Exact PAD lookup + audio PLE projection.
  Actual independent 524288 logits max abs0.000320435/0.000189781,
  relL2 0.000004470/0.000009288, exact argmax. Combined retained
  74,392,844 bytes, existing two learned audio cases and cancellation/
  result retirement passed. Focused222/0/14 strict lint; no ASR claim.
- [x] OA-7B assembled workspace gate: 2,591/0/111 (322 Rust suites, 679s), complete isolated gate passed.
- [x] OA-7C implementation: original source-bound audio prompt/tokenizer,
  native typed transcript/usage/finish and cooperative finite speech
  steps in original Runtime/registry/FIFO/worker/receipt; mixed task
  admission, cancellation and joined retirement. Actual local synthesized
  speech independent reference "Hello world"; native prompt exact match.
  Focused247/0/2, strict lint. No new download or public eligibility.
- [x] OA-7C native learned consumer: actual native SDK "Hello world", input=41, output=3 including stop, finish=Stop; 263.484s completion, full real test passed 292.62s (including cancellation/retirement).
- [x] OA-7C assembled workspace gate: 2,593/0/112 (324 Rust suites, 617s), complete isolated gate passed.
- [ ] OA-7D bounded multipart public transcription through original
  receipt with explicit actual adapter eligibility; natural speech/general
  quality qualification before built-in production model promotion.
  Translation/synthesis/streaming only actual supported capability.
  No second source/runtime or fake text.

### Current continuation — 22 Nov 2026

- [x] Fresh baseline capture is complete and canonicalized in
  ../../docs/CURRENT_BASELINE.md. It covers L5 optimized vLLM CUDA + current
  ZeroGravity CUDA for0.8B/2B, X2 configured-envelope matched vLLM ROCm +
  ZeroGravity HIP for0.8B/2B/27B, and exact L5 27B SSD-streamed CUDA. Do not
  revive historical eager-vLLM, Podman/FUSE, old provider matrices or superseded
  benchmark artifacts for implementation decisions.
- [x] The reusable Nix vLLM provider supports both CUDA and ROCm. ROCm real serving
  includes the pinned Triton host-compiler seam; fresh physical runs retire cleanly.
- [x] Current HIP decode cohort: aligned cached-decode rows reuse the existing
  scheduler/publication authority and share projections, row-wise activation/QK/rotary
  work, plus the 18 GDN layers' Conv4 update. Conv4 uses one bounded 2/4/8-row CubeCL
  launch over the original independent prior/candidate state buffers—no staging and no
  new root authority. Physical0.8B C4/C8 stays exact/replay-stable/lane-invariant with
  zero refusals at13.297/13.351tok/s (+10.37%/+9.98% versus clean batch-1), using
  32,086/61,278 forward dispatches per wave.
- [x] Recurrent GDN direct-private-state cohort experiment: signed `11f692b` added
  reusable row-aware recurrent kernels and a Qwen HIP selector without dense state
  gather/scatter. On A1 gfx1150, the unchanged existing population benchmark (temporary
  PCI identity adaptation only) stayed exact/replay-stable/lane-invariant with zero
  refusals. Warm C4 baseline14,080.01ms vs candidate14,046.02ms (~0.24% better);
  C8 baseline23,916.51ms vs candidate24,354.49ms (~1.83% worse). Dispatches fell
  47,758→45,562 per C4 wave (~4.6%) and87,398→82,610 per C8 wave (~5.5%), proving
  launch reduction alone was not an end-to-end win. Signed `7fcc9c4` removes only the
  selector/load-time preparation and keeps the correct source-level cohort kernels plus
  row-generalized B1 primitives dormant. Final public A1 B1 `run-rocm` still passes and
  retires to the exact KFD baseline.
- [x] HIP singleton C1 localization/first projection cut. Temporary execution-only
  instrumentation on A1 finds cached decode4,474 dispatches/token: projection3,360
  (~75%), GDN360, residual-RMS480 and all other groups smaller; prefill is9,080 with
  projection3,360. Recurrent B1 cuts `69630ec`/`0d201ff` reduced dispatches
  18,028→17,992 but did not improve warm wall time; `879bee7` rejects those consumers
  while retaining their reusable private-state/fused-coefficient kernels. `4070610`
  writes active=1 projection matrices directly into their final combined-output subview
  when logical leases are distinct, eliminating one scatter launch per matrix. It stays
  exact/replay-stable and improves A147+3 warm mean ~6.401→6.359s (~0.65%) with
  dispatches18,028→17,653; the decode-heavy16-output cell improves
  ~32.070→31.350s (~2.25%) with76,190→73,384 dispatches. Broader post-pack alias
  permission (`fdc5a8e`) added no structural reduction and `baf4c9f` restores the narrow
  contract. `ec6fb57` coalesced singleton projection status per matrix and cut47+3 to
  14,557 dispatches / output16 to53,310, but the stronger decode-heavy wall time was
  ~31.518s, slower than `4070610`; `d15b02f` keeps that correct model-neutral alternate
  as an explicit dormant method while restoring per-tile status selection.
- [x] Further singleton projection alternatives are retained but not selected.
  `e1701e7` reused one packed activation across matrices in a Projection node and saved
  144 dispatches per47+3 wave, but warm mean regressed~6.359→6.613s (~4.0%);
  `1d49ec7` restores per-matrix packing and keeps `enqueue_rows_prepacked` dormant.
  `5d51896` factors one exact shared CubeCL projection reduction and adds a certified
  sibling that ORs validity directly into flight status. It remains exact and cuts
  47+3 dispatches17,653→14,182 and output16 73,384→50,504, but warm means are
  ~6.419s/~31.886s versus selected~6.359s/~31.350s; `90537b6` removes the Qwen
  selector/load-time preparation while retaining the shared core, certified kernel and
  certified enqueue primitive dormant. Fewer launches are not the governing cost here.
- [x] Temporary cold-accounting probe on A1 explains its streamed0.8B profile under
  the explicit14GiB benchmark envelope: weights=1,503,789,056B, workspace=38,384,640B,
  root=45,809,664B ×18, while native-device admission alone is9,663,676,416B
  (=8GiB `MODULE_OPAQUE` +1GiB `MAX_CODE`). HIP reports15.232GiB total /15.221GiB
  free; resident0.8B under the current charge would leave only~17MiB total headroom,
  so do not force it. Production `run-rocm` derives an explicit total from HIP current
  free memory minus configured headroom; X2's canonical24GiB envelope may therefore
  have different residency and must not inherit A1-streaming conclusions blindly.
- [x] Qualified gfx1150 native-module accounting and small-model residency (`c5b3fad`).
  Reused the existing physical native lifecycle instead of adding a fixture corpus.
  Temporary A1 adaptation of that qualifier passed four cold compile/load/launch/unload
  cycles on exact `hip-kernel_gfx1150`, ROCm+driver70253211, RTC9.0, including second
  live-set refusal and clean final retirement. Four probe modules total81,396B ELF load
  segments; observed first cold compile/load delta is ~149MB and first launch adds~8.4MB.
  The complete real Qwen0.8B catalogue then prepared92 modules with7,843,872B retained
  host bytes and2,398,636B source-derived load segments; observed native prepare delta
  is150,908,928B. The retained policy keeps a conservative1GiB opaque allowance (>6×
  measured first-use delta) only for integrated gfx1150 nominated by cold device facts;
  the actual actor must then exactly match fingerprint/compiler options/runtime/driver/
  RTC before any compile/load. Every mismatch and every other HIP source device keeps
  the existing8GiB fallback; public device-free preflight also stays conservative.
  Under the exact same14GiB A1 envelope, weights become resident.47+3 dispatches fall
  17,653→9,817 and stable elapsed values are ~0.53–0.56s (one~0.895s outlier) versus
  selected streamed~6.359s. Decode-heavy output16 falls73,384→31,592 dispatches and
  warm mean~31.350s→~0.962s (~32.6×), exact/replay/lane-invariant with clean retirement.
- [x] Re-localized the resident gfx1150 path with the existing population benchmark.
  Exact `fd4611e` C1 decode is1,675 dispatches/token: projection561 (~33.5%),
  RMS/residual-RMS489 (~29.2%), GDN360 (~21.5%), QK/rotary150 (~9.0%). Prefill is
  6,467 dispatches. Warm47+3 C1 is~0.541–0.551s; C4~1.978–2.010s; C8
  ~4.044–4.584s. C4 modestly improves per-request goodput; C8 is not clearly better.
  A1 has no matched vLLM baseline; the maintained vLLM matrix is X2/gfx1151, so do not
  compare raw A1 latency against it.
- [x] `ffd5ff8` adds a model-neutral `residual_squares` RMS first pass and selects it
  for `ResidualRms`: materialize the exact `bf16(add(x, prior))` residual and square
  that same rounded value in one launch, then reuse the unchanged RMS sum/scalar/
  normalize/weight/store cascade. The standalone residual kernel remains reusable.
  Dispatches: C1 47+3 9,817→9,673; C4 32,082→31,794; output16 31,592→30,824.
  C4 warm mean~1.994→1.973s and output16~0.962→0.951s (~1.1% each), exact/replay
  stable. C1 timing was noisy, so it was not used as the acceptance signal.
- [x] Rejected two broader RMS launch fusions without discarding their reusable pieces.
  `b44a55c` shares one exact `scalar_value` implementation between the retained standalone
  scalar kernel and a four-stage `scalar_cascade`. It cuts47+3 dispatches9,673→8,260 and
  output16 30,824→26,330, but output16 warm mean regresses~0.951→1.001s (~5.2%).
  `11e61f8` restores the selected separate-stage consumer and keeps the cascade dormant.
  `9040492` adds an exact `normalize_weight_store` tail and alternate fused-tail launchers;
  output16 dispatches fall30,824→28,314 but warm mean regresses~0.951→0.963s (~1.3%).
  `654adbb` restores the selected separate vector tail while retaining normalize/weight
  helpers, fused-tail kernel and alternate launchers dormant. HIP+Qwen and all-features
  compile pass on the final disposition.
- [x] Localized actual resident gfx1150 GPU time with CubeCL/HIP's existing device-event
  windows; the probe was temporary and serialized, so use it only for relative family
  ranking, not serving latency. On current `a5c57c` steady cached decode, projection is
  ~19.8–21.5ms/token (~60–62%), GDN~6.6–7.5ms (~20–22%), RMS~3.0–3.4ms
  (~9–10%); prefill projection~296ms vs GDN~156ms. Inside projection, output16 totals
  pack~22.2ms, exact GEMM~561.1ms, status~21.0ms. Top GEMM shapes:3584x1024
  ~171.8ms,6144x1024~103.1ms, head248320x1024~81.7ms,1024x3584~75.1ms,
  1024x2048~54.9ms. Clean current output16 timed mean is~0.955s (1.002/0.908s).
- [x] `072b791` retains an exact wave32 projection sibling: one thread computes the
  original lane and lane+32 partials independently, then performs the same CPU64
  +32/+16/+8/final reduction. Selection was tested only in a disposable A1 archive.
  Reject the gfx1150 consumer: serialized GEMM time561.1→577.7ms (~3.0% slower) and
  output16 timed mean~0.955→1.101s (~15.3% slower), exact/replay-stable. Primitive stays.
- [x] `253a565` retains an exact model-neutral rows2 projection primitive: one64-thread
  workgroup computes two adjacent rows, sharing only the packed activation load while
  each row keeps its independent64-partial FMA/reduction/rounding/status contract.
  The gfx1150 consumer is rejected before whole-model A/B because direct device time
  does not give a stable useful selector.3584x1024 flips from~22.9% faster in one run
  to~17.6% slower in the paired repeat;1024x3584 is~24% slower; head248320x1024
  is~4.4% slower. Expansion shapes6144x1024,2048x1024,4096x1024 gain only
  ~2.6–2.9%. Rows1 stays selected; rows2/rows4/wave32 remain reusable dormant.
- [x] Probed two exact inner-K64 alternatives and retained both primitives without
  selecting them on gfx1150. `55477b4` adds paired packed-word delivery via
  `plane_shuffle_xor`; arithmetic/reduction are unchanged, but direct device time
  regresses3584x1024 65.20→153.73ms (~2.36x) and6144x1024 42.82→87.91ms
  (~2.05x). `ba8c5ea` adds a projection-specific packer that publishes the same final
  input status bits once plus an input-certified GEMM sibling that skips only the
  repeated activation exponent check. Paired same-commit timing is3584x1024
  61.786→61.422ms (~0.6% faster) and6144x1024 38.045→45.533ms (~19.7% slower).
  No whole-model A/B; canonical rows1 remains selected.
- [x] Weight-finiteness investigation: VerifiedMatrix/source windows and completed
  resident upload authenticate bytes/geometry, not finite BF16 values. The CUDA
  qualification finite-window check and tied-table check do not cover HIP resident
  projections. No new verifier; keep both operand checks. `d717c7b` retains exact
  full-K64 packed-index hoisting; no FMA/reduction/status changes. A1 direct synthetic
  A/B passed bit-exact accumulator/rounding/status including nonfinite operands and
  strided multi-request output. Clean repeat (six paired96-launch HIP device-event
  windows):3584x1024 8.313→7.481ms (-10.0%),6144x1024 13.377→12.193ms (-8.9%);
  first repeat was noisy. Same-binary real0.8B resident C1 47+16 A/B `224085b1`
  (14GiB, ABBA, two fresh loads and eight timed waves per variant) warm means
  924.5→830.2ms (-10.2%); all output IDs identical, replay exact, dispatches30,824
  unchanged, zero failed quanta, clean retirement and original KFD owner set.
  Select hoisted traversal only for the qualified gfx1150 resident HIP policy,
  immutable in the prepared operand and shared by native catalogue/launch.
  Canonical and prior alternatives remain reusable; temporary probes/switches removed.
  Final production CLI at `3be8a812` generated all16 tokens from the same raw prompt,
  clean KFD retirement. Formatting/architecture passed.
- [x] `c89d0ee7` restores combined backend feature boundaries: distinct
  Bf16TextObserver producer/KV callbacks vs TextObserver invocation frames; shared
  HIP keeps the original8GiB opaque allowance and exact X2 environment validation,
  with optional source native policy rather than a single-source loader dependency.
  Retained CUDA-native primitives use the existing exact-length check under
  cuda-native. CPU-free CLI routing test expects early unsupported refusal; the
  renamed eight-session lifecycle filter now runs its two original tests.
  Combined HIP/Vulkan/CUDA-native compilation, affected CLI/refusal tests,
  architecture and GPU feature boundary gate pass; corrected lifecycle filter passes2/2.
- [ ] Full isolated check.sh proceeds through CPU Clippy but stops at
  cache_prepare::role_and_staging_admission_fail_without_model_source_allocation
  (expected Error::Limit assertion). Investigate this separate CPU cache boundary
  before claiming the full gate. CUDA physical provider/correctness unchanged.
- [x] Re-localized the selected resident gfx1150 path at `fd02260` using the
  existing CubeCL resolved device-event observer and population harness; temp
  probe removed. Two physical runs complete exact 47+1/47+16 greedy prefixes,
  replay, zero failed quanta, and original KFD owner set. Repeat ABBA subtraction
  (mean16-output minus mean1-output divided by15) gives ~37.27ms/token summed
  serialized device kernels (first run~37.15ms). Projection19.48ms (~52.3%),
  RMS7.35ms (~19.7%), GDN6.73ms (~18.1%), other2.15ms, attention/KV1.28ms,
  rotary0.27ms. GEMM17.07ms; next individual non-projection hotspot is GDN
  key_reduce3.10ms (36 launches), then RMS scalar2.73ms. Prefill+first output
  sums~386ms (projection200ms, GDN122ms). Not an end-to-end latency measurement.
- [x] `7d0b395b` adds exact model-neutral GDN key_reduce_subtotals: one256-thread
  workgroup computes eight original16-term subtotals for32 adjacent output
  values, one LDS barrier, then the original sequential carry and shared
  rounding/status helper. No global scratch or state-owner changes; scalar
  primitive remains. Direct gfx1150 values2048 A/B against scalar and CPU
  matches finite bits for both output modes (signed zero/subnormal included),
  nonfinite status and NaN classification. Six paired96-launch device windows:
  FP32 5.814→4.332ms (-25.5%), BF16 6.011→4.340ms (-27.8%).
  Actual resident0.8B 47+16 A/B `7d3a130b` (C1,14GiB,ABBA,12 timed
  waves/variant): scalar836.7 vs parallel830.4ms mean (~0.75% lower);
  medians841.4/823.9ms, ranges809.9–864.0/789.9–911.4ms. Both paired
  process means favor parallel by <1%; not a sufficiently robust request-level
  benefit for extra selector wiring. Keep scalar selected and parallel sibling
  reusable; remove temporary selector/catalogue/probe and benchmark adaptation.
  All16 greedy IDs/replay exact, dispatches30,824/request, zero failed quanta,
  clean retirement and original KFD owners.
- [x] RMS scalar review: four tiny passes (<=64 groups) and prior cascade/tail
  fusion losses; no compelling arithmetic change selected. `6f2ededf` adds
  the exact two-barrier CPU64 projection subtree sibling over the selected
  hoisted traversal. Eight disjoint subtrees preserve every +32/+16/+8 add,
  final carry, rounding and status; no new scratch or owner. Physical A1
  direct A/B matches all accumulator/rounded/status bits, including signed
  zero/subnormal inputs, strided multi-request output and nonfinite operands.
  Six paired96-launch device windows (current→candidate ms):3584x1024
  7.453→7.367,6144x1024 12.343→12.345,1024x3584 8.378→8.367,
  2048x1024 4.372→4.310. Gains <=1.5%, mixed individual pairs; no
  whole-model A/B justified. Retain both exact siblings; production selector
  unchanged and temporary probe removed. Original KFD owner set restored.
- [x] Recover actual gfx1151/X2 execution and qualify native accounting.
  The installed CLR7.2.4 works with its glibc2.44 loader and matching libraries;
  the locked Nix dev-shell input remains unavailable locally. Reuse the original
  four-cycle lifecycle: `42d4703d` checks second-live-set refusal and unchanged
  ownership instead of obsolete error wording. All four cycles pass; observed
  peak incremental physical VRAM~194MB, clean original retirement.
  Temporary catalogue observation (`11a44d60`, removed) measures complete
  0.8B/2B/27B native sets:93/90/95 modules,~2.4MB load segments each,
  incremental physical VRAM168.116/168.059/168.153MB. Actual C1 at the original
  24/24/75GiB envelopes:12.508/6.977/0.539 output tok/s, exact IDs/replay,
  no failed GPU quanta, clean retirement. `38f20ca3` qualifies 1GiB opaque
  for integrated gfx1151 on the same runtime/driver70253211,RTC9.0 tuple.
  The 1GiB source-derived code bound is unchanged. Cold version observation
  preserves 8GiB fallback for unknown runtimes/architectures; the actor
  revalidates its exact fingerprint/compiler options and versions before load.
  Distinct policy variants initially preserve gfx1150-only hoisted projection selection.
  Device-free and original finite/shared qualification keep conservative accounting.
  Final `38f20ca3` real2B at14GiB and real27B at68GiB pass exact output/replay,
  9,673/36,741 dispatches per request, zero failed quanta and clean original
  retirement; warm mean434.714/5,556.005ms. Actual native charges
  1,076,111,858/1,076,230,179B; original-envelope throughput essentially
  unchanged. GPU feature/CPU Clippy gates pass; full check.sh stops on the
  pre-existing CPU cache admission assertion.
- [x] Final-policy X2 C4/C8 recheck, both models at24GiB/configured width8:
  0.8B13.743/13.482 tok/s,2B7.842/7.641 tok/s. Exact identical-lane output,
  replay, actual fused decode cohorts (two per wave), zero failed quanta;
  31,794/60,790 dispatches per wave; clean original retirement.
  Against retained matched vLLM, C8 gaps remain~16.3x/~18.4x.
- [x] X2 actual device-event localization (`533c72c`, temporary observer removed):
  0.8B C1 prefill+first output199.62ms (GEMM85.44, GDN triangular32.83),
  cached23.65ms/step (GEMM10.21); C8 prefill1,589.56ms (GEMM680.35,
  triangular262.65), cached87.68ms/cohort step (GEMM31.04).
  2B C1 prefill332.48ms (GEMM214.79, triangular33.01), cached41.14ms
  (GEMM25.98). C8 prefill remains independent; cached cohorts share only
  two additional decode invocations for47+3. Exact prefix/replay/retirement.
- [x] `f7ebf647` retains model-neutral `project_cpu_bf16_requests`:
  two/four independent input rows share packed weight loads, keep each
  original K64 FMA/reduction/status tree; partial request tiles and strided
  output supported, no additional global scratch or owner. X2 direct A/B
  bit-exact accumulator, rounded output, flags including nonfinite operands
  and strided tails; all8/47-row tile2/4 cells lose device time. Keep sibling
  dormant. The same direct A/B finds the existing hoisted sibling wins all
  twelve (four shapes x1/8/47 input rows) by6.4–48.9%, every4-pair mean
  positive. Select the existing primitive instead of new consumer plumbing.
- [x] `c1cd04f3` real0.8B/2B same-binary ABBA,24GiB,active8,configured
  width8,47+3, four timed waves per variant/cell. Mean request latency
  C1/C4/C8 0.8B238.514/872.431/1787.195→211.895/765.040/1540.048ms;
  2B420.655/1533.845/3153.334→388.041/1389.398/2848.838ms.
  Every candidate beats fastest same-cell baseline; exact IDs, replay,
  lane invariance, unchanged dispatches9,673/31,794/60,790, no failed quanta,
  clean original retirement. `99eb7990` selects hoisted traversal for
  both qualified integrated gfx1150/gfx1151 resident policies, no new
  owner/selector machinery. All temporary observers/runners/env switches
  removed. Final unchanged benchmark0.8B24GiB/27B68GiB passes at
  mean211.004/5,376.029ms with exact outputs, replay, unchanged dispatches,
  zero failed quanta and clean retirement. GPU feature boundaries/CPU Clippy
  pass; full isolated check.sh stops at the same CPU cache admission assertion.
  See CURRENT_BASELINE.
- [x] `393d0f56` retains reusable exact `mul_then_add_normal` (separate
  FP32 product and add with guarded normal path, exact zero-product sign,
  widened fallback) and shared-body guarded triangular solver. No change
  to sequential64-row dependence, four-accumulator tree, final subtraction,
  status or scratch. Direct gfx1151 A/B16 heads, six paired16-launch windows
  47 active/padded64 26.365→9.737ms (-63.1%), full64 17.217→9.162ms
  (-46.8%). Both complete matrices finite bits exact, faults/status and
  NaN classification match, original retirement. Existing arithmetic
  scalar-body test extended by one comparison; both denormal modes pass.
- [x] `cd86a965` actual same-binary ABBA,0.8B/2B C1/C4/C8,24GiB,
  active8/configured width8,47+3, four timed waves per variant/cell.
  Means0.8B211.801/765.845/1545.560→191.113/681.951/1378.619ms;
  2B389.243/1386.046/2841.161→370.777/1302.572/2677.231ms.
  All six means improve4.74–10.95%, exact IDs/replay/lane invariance,
  unchanged dispatches9,673/31,794/60,790, zero failed quanta, clean
  retirement. `4a5961da` selects guarded triangular only on qualified
  gfx1151, cold immutable choice shared by catalogue and launch. Canonical
  and original independent qualification paths retained. Temporary direct
  probe/benchmark adaptation/env selector removed. Final unchanged C1
  0.8B24GiB/27B68GiB canaries pass at190.322/5,301.120ms warm means,
  exact IDs/replay, unchanged dispatches, zero failed quanta, clean retirement.
  GPU feature boundaries/CPU Clippy pass; full isolated check.sh stops at
  existing CPU cache admission assertion. See CURRENT_BASELINE.
- [x] Shared HIP prefill composition: `2ed9c10f` generalizes the existing
  cohort traversal/admission/completion owners to aligned equal-length
  2..64-token prefill chunks, same output policy, private state roots;
  one-token tails stay singleton. Flatten projections, gather terminal
  rows, extend only required arena/scratch slots, preserve capacity fallback.
  `282a3437` / `2eee25fa` share stateless operators and direct single-matrix
  output. Untiled C8 locality loss motivates `fd914f63` resident128-token
  projection tiles (streamed operands remain whole-cohort).
  Real0.8B/2B C4/C8 exact outputs, three actual fused cohorts/wave, dispatches
  29,564/55,695 versus31,794/60,790, warm means657.981/1283.333ms and
  1249.184/2531.391ms,3.5–6.9% lower versus retained previous checkpoint.
  Distinct-prompt singleton differential (`47b9f6f6`) passes22,922,240
  exact FP32 state/logit words including state-only continuation, terminal
  discard/replay; `74619a0b` real SDK multi-chunk regression passes with
  observed state-only fusion. Clean retirement. See CURRENT_BASELINE.
- [ ] Next HIP priority: localize shared prefill device costs after tiling.
  Keep actual shared composition; target remaining private GDN/conv/attention
  or projection throughput where measured. No speculative selector machinery
  or fixture proliferation. Restore unavailable pinned vLLM closure separately
  for fresh paired runs; no acquisition/repin merely to benchmark.
- [ ] Parallel CUDA correctness unit: localize and fix the deterministic C4/B8
  identical-lane divergence before using wider CUDA throughput as accepted evidence.
  The existing population benchmark is the reproducer; do not add a fixture corpus.
- [ ] After each coherent fix: rerun only the smallest real-model cells needed to
  prove correctness and compare against CURRENT_BASELINE. Prefer measured hot-path
  work over additional evidence administration.

- [x] `c282f20` restores HIP-only source-profile compilation after a shared CUDA
  optimization accidentally gated the model-neutral packed-BF16 numeric-zero
  greedy kernel behind `cuda-cublas`; stale HIP test/support profiles now specify
  `native_rows: 0`. X2 compile/link validation passed for `zerogravity-gpu`
  `hip-source`, `qwen_hip_sdk` `qwen35-hip,serve`, `fabric_sdk`
  `qwen35-hip,serve`, and the temporary release 27B SDK/HTTP diagnostic.
- [x] Re-established tiny X2 physical HIP health before another large-model
  campaign. X2 was first synced to exact l5 `2caf304` without push transport. A
  temporary self-validating HIP C++ smoke was built with the installed CLR 7.2.4
  toolchain plus the ROCm device-library store referenced by that CLR, then run
  only through `python3 -B scripts/native-gpu.py -- ...`. It exited 0 on the
  AMD Radeon 8060S (`gfx1151`): one device, 256 exact `i*17+5` output words,
  checksum `556160`. Post-run VRAM returned to
  `272351232/103079215104` bytes, GPU busy was `0`, no KFD client remained,
  and the dotlink cgroup still reported `oom=0`/`oom_kill=0`.
- [x] Localized the earlier tiny-smoke failure away from HIP runtime itself.
  Focused Rust test builds were externally terminated without an exit code both
  inside and outside `native-gpu.py`; a plain `hip` unit-test build did complete
  and link in about 2m15s. No cgroup, kernel-log or systemd-oomd evidence of OOM
  was observed. A temporary source-gate experiment was discarded, and all smoke
  sources/binaries/sync bundles were removed; tracked source remains `2caf304`.
- [x] `134a571` repairs a second HIP-only cross-feature compile leak: the broad
  batch-KV qualification surface no longer invokes the CUDA-only private-BF16 KV
  physical qualifier without `cuda-cublas`. Exact l5 HIP qualification check and
  X2 release context-diagnostic build pass.
- [x] Localized the current 27B blocker before serving. Exact X2 canonical
  27B source/cache/plan `cache-preflight` passes host-only with source unchanged,
  but the canonical runtime source/tokenizer test is externally terminated twice
  with no GPU/error/OOM evidence. A fresh explicit75GiB current-main HTTP
  diagnostic likewise dies during model load before first HIP allocation; no HTTP
  status is produced. Historical R9/R10 HTTP504 predates the 8 Nov
  `d16f096` resident-budget plumbing and exercised the ~240s streamed path.
- [x] Isolated the runtime host path further. Exact canonical27B tokenizer
  construction/serving inventory is green; a smaller host-only exact-source probe
  authenticates all18 shards and completes Qwen binding/seal/1K plan. Signed
  `b0cf0b2` removes a redundant initial55.6GB cache snapshot payload hash by using
  an explicitly non-authoritative descriptor lease; all consumed metadata/shards
  keep their downstream SHA authentication and normal `SnapshotLease` stays fully
  verified.
- [x] The full runtime host path is stable when the X2 execution wrapper remains
  alive: an optimized full-runtime unit-test binary completed the exact canonical
  source/tokenizer/final-reader test in75.20s with source plan true and no GPU.
  Earlier silent deaths were execution-envelope/build artifacts, not evidence of a
  Qwen source bug; no source weakening followed.
- [x] Retried the next authorized reference-free liveness gate: exact snapshot +
  explicit75GiB resident budget + public HTTP route + exact47-token synthesized raw
  prompt, under the unchanged180s server deadline and private native-GPU namespace.
  The run reached HIP allocation (~31.376GB VRAM, busy6%) but the entire isolated
  process disappeared after ~10..12s despite an external heartbeat, before any HTTP
  status. This establishes allocation progress only, not HTTP/numerical acceptance.
- [x] Host-global KFD sysfs exposes one existing process entry `2406838` with
  31,098,232,832 bytes VRAM and six queues (`pasid=0`, zero occupancy/faults).
  **Do not attribute that owner to the failed ZeroGravity run.** X2 is a shared
  GPU host and also runs Ollama/model workloads outside the Bubblewrap PID
  namespace; sandboxed `/proc` cannot map the KFD owner back to its host process.
  The previous “stranded ZeroGravity allocation / dirty GPU” interpretation is
  retired.
- [x] Generic execution plumbing is not the10..12s boundary: an otherwise matched
  durable Job/Bubblewrap/`nix develop`/`native-gpu.py`/nested-unshare control
  stayed alive35s and exited0. Dotlink's device pins are `O_PATH` mount
  capabilities, not DRM/KFD clients. The failed75GiB liveness run did reach
  ~31.376GB total device usage before the isolated process disappeared, but shared
  baseline overlap means that total cannot be assigned to ZeroGravity.
- [ ] Treat X2 as a **shared physical resource**, not an exclusive test card.
  Record VRAM/KFD state before each run, execute only when the requested workload
  fits the currently available capacity, and verify only ZeroGravity's incremental
  process/VRAM/queue delta retires afterward. Existing KFD owners/VRAM remain
  other-workload baseline. Never reset/reclaim/restart/kill those owners merely to
  create an empty GPU. Missing27B/populated1K/2K references remain a separate
  numerical-acceptance blocker. **20 Nov update:** a near-empty274,448,384B/0% X2
  baseline was available and exact-current source plus the accepted private child-net
  wrapper were prepared successfully, but both installed clr7.2.4 closures fail
  `dlopen` because their LLVM22 requires GLIBC_2.44 while the current locked shell is
  glibc2.42. No HIP allocation occurred. Treat X2 ROCm model execution as environment-
  gated until that closure is repaired; do not weaken isolation or substitute SDKs.
- [ ] L5 remains the source/repository authority. X2 is execution/testing only and
  must be synced to the exact current l5 `main` before physical work; do not push
  merely to transfer local commits.

### Expanded owner mission — 30 Oct

- [ ] Finish ordinary correctness on `pvl-l5` CUDA `.8B`/`2B` through genuine
  active-context fit, and `pvl-x2` ROCm `.8B`/`2B`/`27B`. Current finite
  references/capacity checks do not fill missing routes or populated contexts.
- [ ] Qualify real C1/C4/C8/C16/C32/C64 scheduling, independent state, faults
  and shutdown; place higher concurrency on capable hardware, especially x2.
- [ ] Then optimize and execute every feasible matched vLLM cell under the
  [expanded campaign](BENCHMARK_CAMPAIGN_2026-10-27.md). Per-run wins are the
  goal; preserve losses, active/offered widths and resource asymmetries.
- [ ] Use x2 resources as needed; continue without routine permission prompts.
  Only consequential external dependencies/decisions justify stopping to ask.
  This supersedes the earlier blanket benchmark deferral, not correctness gates.

### Reviewed continuation — 29 Oct 2026

- [x] Current `main` ordinary Qwen CUDA is integrated but qualification-only: source-derived projections/attention/GDN, one resident queue/flight/completion/quarantine path, one `StateRoot`, explicit provider shutdown and source-free cancellation/fault ownership.
- [x] Approved `.8B` source is accessible and exact at revision `2fc06364715b967f1860aea9cf38778875588b17`; the 1,746,942,600-byte shard hashes to `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
- [x] Recover `.8B` N1 native/reference inputs offline: retained canonical images plus exact whole SM80 FlashAttention modules extracted from the pinned local reference image, original cuBLAS providers, 28 resources matching current Rust identities. All 11 approved model members verify; maintained bundle validation and isolated structural preflight pass. No deleted run trees restored.
- [x] Fresh phase-1 cached47→50 capture passes with the existing explicit 8 GiB reference fixture host profile; 1,008 source windows retire. Preserve the original 4 GiB admission failure. Ordinary host/device allowances remain unchanged.
- [ ] Owner resumed. Owned GEMM/RMS/rotary/attention now pass exact ordinary `.8B` cached47→50: full logits/all42/natural feedback and normal shutdown. Normal/fault lifecycle and final gate pass; long-active-context/2B numerical scopes need independent reference inputs; see [numerics](../numerics-and-context/TODO.md). Original cuBLAS private interface remains correctly refused; no alternate allocator or relaxed reference.
- [ ] Then exact ordinary cached47→50 comparison, real after-launch cancellation/fault/retirement/shutdown, admitted context correctness and assembled gate. Reference capture alone closes none of those gates.
- [ ] Owner30Oct: continue through Qwen ROCm (HIP) on `pvl-x2` now; do not wait on wider CUDA reference gaps. Needed approved2B/9B/27B downloads may use canonical XDG cache. No subagents or benchmark detour; serving remains disabled.

### Matched campaign checkpoint — 27 Oct

Design and acceptance: [Qwen3.5 matched serving campaign](BENCHMARK_CAMPAIGN_2026-10-27.md).

- [x] `1921db1`/`1d278ed` extend bounded comparison through C64 and add one
  serial matrix driver. Every complete per-cell report is retained atomically;
  aggregate/cell output aliases and existing outputs refuse unless explicitly
  overwritten. Self-tests are part of `scripts/check.sh`.
- [x] Current-source production CPU Router test passed genuine offered
  populations16/32/64 through two workers×four active slots, real Qwen
  execution/output, queue/active health, host charges, cap refusal/recovery and
  full cleanup. This closes source-free pending/admission scheduling, not the
  missing ordinary CUDA model route. Synthetic scheduler commit `68cbef5` was
  rejected.
- [ ] Extend retained usage/timing for exact context prompt tokens and explicit
  provider-side prefill/first-token timing. Prepare strict canonical47, ragged
  47/57/62/67/72 and 512..32768 workload/lifecycle inputs under XDG state.
- [ ] After ordinary `.8B`/`2B` correctness, cached/fault/shutdown, long-context
  fit and final gate: execute one serial C1/C4/C8/C16, then admitted C32/C64 and
  context campaign with two balanced ABBA/BAAB repetitions; cold and warm stay
  separate. No 27 Oct timing has run.

### Shutdown qualification — 26 Oct

- [ ] Replay applicable original-model/backend shutdown and fault scopes under
  their existing source/device gates before a benchmark qualification claim.

### Current parity and post-parity optimization order — 23 Oct

- Current MAIN/ISO410: continuous34+dynamic2-key KV and compiler-lifetime7
  assembled; source/strict/ownership checks and optimized release5 pass. Actual
  finite B4/lifecycle/retirement/3fault and both-model N1 matrices/faults pass.
  Source6 shared input geometry fixes the old equal-frontier refusal; grouped
  strict/2401frontier checks and optimized release6 pass. Affected finite B4/
  lifecycle/zero-retirement/3fault cases pass. Genuine mixed N1→B4 reaches math
  but differs from same-prior N1 at SSA9/layer0 post-attention residual; earliest
  divergence was unresolved there. Fixture-only diagnostic7 now finds first
  difference at QKVZ2, with identical embedding/InputRMS and all48prior planes
  perrow. Subsequent original same-prior B4 comparison passed below; the old
  cross-width N1 diagnostic remains unchanged. All original owner/group/EOF
  closures verified. Native mixed/refill/fullstate and stock API evidence below
  are distinct from the separate genuine HTTP mixed/refill result below; fullgate
  remains open.
- Fixture-only source8 exported all15 genuine transitions/35events/1,648,529,280B
  including cancel/refill/permutation/N1 bridge. Healthy source/native retirement
  and final host/device0 verified; this is explicitly unqualified numerical data.
  Original same-prior reference3 PASS: all15transitions/30plain+observed calls,
  full42prior+candidate/fullV/24sums/natural bits/control exact, including first
  mixed B4/refill/N1bridge/permutation. All320loadedroles unchanged; final source/
  engine/process/container/group/EOF closure healthy. Explicit state-initialized
  comparison, not an independently generated original public trajectory. Earlier
  adapter failures and N1 cross-width diagnostic remain preserved. Fresh API8
  PASS24offers192tokens/provider/source/owner closure. Wave uses B4 prefill and
  cached B4; ragged remains N1. Separate actual event-driven HTTPv2 CLOSED0:
  5genuine N1prefills, A→cachedB4[50,48,48,48], completedA then replacementE in
  B4[51,51,51,48] through[54,54,54,51]; all5offers36natural tokens match actual
  same-cap original stock.5B4publications/20rows,21GPUcalls/36roots/peak4;
  original source/process/container/group/one-reap/fullEOF closes. Explicit
  A4+BCD/E8 correctness case; stock serial reference reused after v1's missing
  metadata run_id failure, no retag/performance/general cancellation grant.
  Quiet C1/C4 CLOSED0: ZG66.7151/17.5611 vsstock37.3407/49.0326tok/s; all32offers/
  256tokens and four original owners exact. Singlepass preliminary; C4 regresses,
  C8 held. Finite genuine HTTP proof now passed. Separate direct diagnostic4
  establishes79first-use trips consuming3.10167–3.10218s of its3.72236s instrumented
  wave;185total trips/1480spans, all8/64/clock/source/owners exact, later/second
  wave0compiler spans. No quiet-run fraction or new speed claim. Shared admitted
  cold readiness/context accounting is next, preserving original caps and real
  model/root owners; no fake warmup or unpriced cache. Fullgate
  remains last; latest accepted gate16 covers preceding316 only.
- Latest measured finiteB4 source400v4: all8actual calls/fullV/all42states/
  all24boundaries/24dual sums/32outputs and lifecycle PASS; submitted conv/lastFA/
  head unknown retention PASS. Matched API24offers192tokens/provider exact;
  wave8 uses16realB4 flights, ragged remains N1. Same-flight B4 record retirement
  correction now passes fresh actual healthy zero-retirement, lifecycle and all3
  unknown cuts. Fresh API24offers192tokens/provider/source/owner closures PASS.
  Quiet C1 ZG67.2592/stock37.1156tok/s; **C4 ZG11.7555/stock49.4752**, regression.
  All32offers256tokens exact; C8 held until corrected C4 superiority. First-use
  compile specialization is source-confirmed; no measured compiler fraction.
  Continuous33 canonical47 N1→mixedcached47..53/refill/cancel and dynamic2-key KV
  successor advance in isolated checks; actual continuous qualification pending.
  Wider prompts N1. Final fullgate pending, no accepted B4/performance promotion.
- Residual shared binding Sol8 integrated MAIN/ISO316selected paths; bounded40
  analyzed-site cache with real per-invocation provenance and original late
  physical/state authority. Retained witness/site metadata is charged; GPU551/
  provider+default strict PASS. Original release and both models' complete
  source/state/fault/API plus forced-streamed2B faults PASS. Fresh .8B C1/C4/C8
  ZG67.2018/63.8830/60.9713 vsstock37.6230/50.4311/61.6415tok/s
  (+78.62%/+26.67%/−1.09%); all48offers384tokens exact, six original source/
  owner/group/container/full-EOF closures healthy. Singlepass; C8 superiority
  remains unmet. Final isolated gate16 PASS1752Rust/0failed/89ignored/269suites/
  allscriptchecks;316sourcepaths/rootREADME unchanged. Native remains N1; genuine
  BF16 batching precedes further N1 microtuning.
- Previous prepared-recipe release1:314selected source paths; shared source/
  topology semantics with genuine per-invocation provenance, retained host
  charges and warm binding outside cacheMutex. Both models' complete actual
  source/state/fault/API and forced-streamed2B faults PASS. Quiet .8B C1/C4/C8
  ZG64.8742/63.5759/60.7789 vsstock37.9990/49.3562/62.3861tok/s
  (+70.73%/+28.81%/−2.58%);48offers384tokens exact, all6owner/source/group/
  container/EOF closures healthy. Singlepass; C8 superiority remains unmet.
  Full assembled gate15 LAST PASS1752Rust/0failed/89ignored/269suites/allchecks
  after four qualification-only27B example corrections;314sourcepaths/rootREADME
  unchanged. Native remains N1; true B4/B8 and wider parity remain open.
- Previous accepted source checkpoint: shared native/context ownership+Conv4,
  301selected paths; required release both models' complete physical/state/fault/
  API scopes and forced-streamed2B faults PASS. Full isolated gate14 LAST PASS
  1747Rust/0failed/88ignored/269suites/allscriptchecks; source/rootREADME unchanged.
  Its timings remain storage/rings release2. Actual x2 native3+transport12
  pass only their four-kernel/BF16-transport scopes. Canonical47B4 source discovery
  and immutable fullargument capture2 now accepted:7872complete native launches,
  4344records/1262868896B plain-observed exact/all24/all42/fullV/natural. Exact
  finite54tuple catalogue uses two independently healthy original FAimage
  variants; failedcapture1 stays unaccepted, no normalization/receipt retag.
  All original source/owner closures pass; this is expected/native metadata only.
  Genuine188/376-token prefill and same-root
  decode must be qualified before wider batching or performance claims.
- Previous frozen storage/rings release2,287 selected source paths: both actual
  301finite+40natural/fullV/all42states/all5fault matrices and matchedAPI24offers/
  192tokens perprovider exact. Forced-streamed2B headunknown/headcancel verifies
  actual ring lifetime/quarantine. Fresh .8B C1/C4/C8 ZG62.3484/60.7822/57.4249
  vsstock38.0862/49.9539/61.8017tok/s (+63.70%/+21.68%/−7.08%).2BC1
  ZG43.3738/stock37.6126tok/s (+15.32%); full resident head uses explicit native
  4608MiB vsstockrequest4096+KV512/no harddevicecap, host4096/device5120 unchanged.
  All original timing/source/group/EOF closures closed; singlepass/no confidence
  claim. C8 still needs genuine BF16 batching. Strict/core/GPU/provider checks
  pass; full isolated gate12 PASS1734Rust/0failed/88ignored/266suites/allscriptchecks,
  287selected sourcepaths/rootREADME unchanged. Astra handles architecture only;
  shared integration, model/native ownership and batching implementation use
  Sol HIGH in independent lanes. [Current measurements](../cpu-gpu-optimization/SHORT_HF_RESULTS.md).
- Previous frozenpair9: shared prepared keys/cold copy definitions/source metadata
  reuse,260MAIN/ISO sourcepaths, strict focused checks/release PASS. Original
 301edge+40natural/all5fault/matchedAPI checks exact, unchanged device counters/
  fault scalars, cold metadata +1400B within old bound. Quiet C1ZG61.5227/
  stock38.5419(+59.63%),C4ZG62.4174/stock49.6312(+25.76%),C8ZG62.5204/
  stock61.2769(+2.03%)tok/s. All48offers384tokens exact/all6 originalowners/
  source/EOFclosures reconciled. Singlepass only, small C8 margin unreplicated;
  no complete parity/stable-win claim. Full isolated gate11 PASS1724Rust/
  0failed/88ignored/266suites/allscriptchecks,260sourcepaths/rootREADME unchanged.
 2B actual301edges+40natural/all5faults/matchedAPI24offers192tokens each PASS.
  Quiet2BC1 ZG0.5298/stock37.0680tok/s (stock69.97x),16offers128tokens exact/
  both originalowner/source/EOFclosures reconciled. Further2B bands held for
  shared storage/streaming repair: source shows1.017GB head/970uploads+actual
  streamwaits perforward. No full attribution, fit or general parity claim.
- Latest owner sequence: shared-engine reuse first; superior matched .8B C1
  AND C4, thenC8; after allthree superior,2B. Qwen27B/Gemma integration stays
  parallel; their GPU qualification/benchmarks run on pvl-x2 ROCm/HIP. Vulkan
  parity with CUDA is also required. Backend availability grants no HF permit.
- Previous common196path optimization checkpoint: full isolated gate6b
  PASS1713Rust/0failed/88ignored/266suites and scripted checks; new frozenpair6
  real matrix40/all5faults and matchedAPI24offers192tokens perprovider exact,
  independent roots/active8/source/ownedclosure PASS. Quiet pair6 singlepass
  C1 ZG52.0304 vsstock37.1838tok/s(+39.93%), C4 51.6654 vs47.0443(+9.82%)
  opened C8: ZG52.6333 vsstock61.0984(-13.85%). All48offers384tokens exact,
  allsix original owners/full-source closures healthy. Actual peaks1/4/8,
  batchrowsmax1/zero fused rows; no repeated confidence/stable-tail claim.
  No2B promotion; remaining C8 gap is the next optimization unit. The frozen
  independent Gemma matrix resumed after C8 final closure under its original
  deadline. Backend admission changes are outside this frozen binary; latest
  source checkpoint follows below.
- Previous MAIN202source checkpoint adds the tested backend admission/CLI value-
  routing fixes; isolated gate7 PASS1715Rust/0failed/88ignored/266suites plus
  scriptchecks, exact source/rootREADME unchanged. No new timed binary or HIP/
  Vulkan physical support follows. C8 finite-check cohort successor and portable
  backend source/transport work continue in isolated checkouts.
- Latest accepted MAIN/ISO210selected changed-source unit: frozenpair7 actual
  301deviceedge+40natural calls/allfivefaults/matchedAPI24offers192tokens per
  provider PASS, fullsource/originalgroup/container/EOFclosure. All40 calls
  observe83finitecertificate enqueues vs327previous; exact math unchanged.
  Full isolated gate8b PASS1715Rust/0failed/88ignored/266suites/allscriptchecks;
  one test-module declaration formatting-only gate8failure preserved/fixed.
  Quiet matched .8B C1ZG55.0931/stock38.5359(+42.97%),C4ZG53.1955/stock49.3781
  (+7.73%),C8ZG53.4094/stock63.9834(-16.53%)tok/s. All48offers384tokens exact,
  all6 originalsource/ownerclosures healthy. Singlepass/no confidence claim;
  N1math/zero fusedrows and C8superiority unmet. No2B timing. Next source unit
  investigates shared submission/replay vs genuineHFmultirow execution.
  Newer reviewed portable matrix11 plus exact allocator-range correction are
  integrated MAIN/ISO after timed source closure. All12 actual HIP/Vulkan
  transport modes pass with independent ROOT owner/source/binary/EOF checks;
  no arithmetic/model/serving grant. Actor6 merged; frozen pair8 release and
  actual301+40/all5fault/API pass, every dispatch/finite/fault scalar equals
  pair7. Corrected assembled gate10 PASS1715Rust/0failed/88ignored/266suites/
  allscriptchecks, source/rootREADME unchanged. Quiet pair8 C1ZG60.8262/
  stock38.2050(+59.21%),C4ZG60.9697/stock50.6970(+20.26%),C8ZG60.4256/
  stock63.2563(−4.47%)tok/s, all48offers384tokens exact/all6 original owners/
  finalsource closure. ROOT reconciled actual records. Preliminary singlepass,
  C8superiority unmet/no2B promotion; current actor causal diagnostic next.
  The tiny portable6mode foundation passed separately; neither foundation nor
  transport compilation grants HF/backend/model/serving/performance parity.
- Expanded short-HF runtime converged into MAIN, preserving orderedSiLU/PTX
  work. Both actual native matrices PASS: .8B/2B ×47/57/62/67/72 ×8 natural
  calls, exact24 boundaries/42 state domains/fullV/feedback and independent
  roots, refusal/stale/cancel/survivor/final physical0 checks. Genuine67/72
  two-chunk prefill is covered; all327 independent observations remain open.
  Matched .8B HTTP preflight PASS: active8, both populations, all24 responses'
  eight IDs/text/usage/finish exact against stock. All five .8B cached fault
  scopes PASS for each model. Prior MAIN isolated checkpoint check.sh PASS:
  1531 Rust passes/0failed/81ignored/260suites plus18 namespace fixtures and
  complete scripted release/installer/lint/qualification checks. Matched2B HTTP
  now PASS all24 responses/full8 IDs/text/usage/finish/active8/owned closure.
  Owner23Oct reprioritized after quick .8B canonical C1/C4 baseline: optimize
  .8B to beat vLLM first; no2B timing, C8 canceled and full128arms deferred.
  Quick C1 output tok/s ZG0.9557 vsstock38.5021; C4 ZG0.9511 vsstock49.6408.
  All32 offers completed correctly, one warmed-process pass/configuration.
  No confidence claim. Timed owners retired; separate untimed profiling and
  admitted head-residency optimization now active, Qwen/Gemma parallel.
  Test durations are not serving measurements.
- Both new stock-only seq8 quality preflights pass. Stock2B sampled peak5443MiB
  exceeds native's5120MiB physical cap; original stock protocol does not enforce
  that hard device cap. Preserve resource asymmetry in results; no stock5GiB fit
  or equal-resource performance claim. No2B timed arm has executed.
- Previous finite checkpoint22Oct: original .8B canonical47+7 matches all24 boundaries,
  all42 state domains/fullV/actual feedback in the qualified native scope.
  Existing HTTP observes real active2, cancellation and healthy survivor.
  That accepted checkpoint converged MAIN/ROOT; fresh MAIN six SiLU/two original GGUF B2
  workers/canonical HTTP, strictClippy/fullgate and vendor104 guards PASS.
  Full all327 oracle/original GDN tuple, broader two-model/ragged active8 and
  matched performance remain open. [Current numerics](../numerics-and-context/TODO.md)
  supersedes dated partial-layer milestones below.
- Owner reconfirms binding/publication continuation, then matched vLLM timing
  after parity. Parallel recursive primitive/pipeline/profile design is authorized;
  no new framework blocks the current original-engine correctness units.
- [ ] Profile and fix the .8B gap first, then beat the same matched vLLM C1/C4
  baseline without changing source/quality/caps. Broader2B/C8/ragged/repeated
  timing deferred by explicit owner priority; no unnecessary baseline sweep.
  Resident-head candidate passes all40 actual numerical calls/unchanged limits
  and all five fault scopes after narrow verifier recognizer repair. Quiet
  host phase diagnostic8requests/64forwards exact: binding3.629112s/enqueue
  3.021396s, nestedHead0.005158s/zeroH2D/zero checkpoints; original owned closure
  clean. Follow-on host unit passes fresh40+all5faults/14metadata guards/API24
  per-provider exact; strict GPU Clippy PASS. Fresh unprofiled canonical wave8:
  C1 native9.7748 vsstock38.0061 tok/s (6.54748s vs1.68394s); C4 native9.7972
  vsstock49.1130 (6.53247s vs1.30312s). All32 offers/256tokens exact, actual
  peaks1/4, batchmax1, all original owners/source/EOF close. Preliminary one
  pass only; still3.89x/5.01x slower, no certified win. Combined CPU+phase
  diagnostic finds memcmp70.66% of532 user-cycle samples/0lost; actual process
  6.72user+0.66system over6.77396wall. Nested bind3.074109s/enqueue2.945825s,
  complete/readback0.149683s; Head64launches/zeroH2D/checkpoints. CPU fractions
  are not GPU or wall fractions. Investigate deep GDN static-source identity
  comparisons at their shared metadata owner; do not weaken equality/authority.
  Exact static-source equality now preserves all field/content semantics while
  bypassing same-pointer+length text scans. Metadata10/strictGPU Clippy/joint
  release PASS; frozen pair4 passes fresh actual matrix40/all5faults and matched
  API24/24 per provider, exact output/state/owner closure. Fresh unprofiled pair4:
  C1 ZG40.4089 vsstock37.0951 tok/s (1.58381s vs1.72530s), native+8.93%;
  C4 ZG39.2740 vsstock48.2272 (1.62958s vs1.32705s), native18.57% lower.
  All32/32 offers/256tokens exact, original owners/source/EOF closed. One
  preliminary pass, no confidence/tail/fullparity claim. Actual HF batchmax1;
  source-safe B4 integration is under separate investigation, not yet a causal
  percentage attribution. Subsequent pair5 residentembedding reuse passes
  fresh40calls/allfivefaults/matchedAPI24offers192tokens perprovider. Quiet
  C1 ZG48.9151 vsstock38.8841tok/s (+25.80%); C4 ZG49.9998 vsstock50.4721
  (-0.94%), all32offers/256tokens exact/allfour owners/sourceclosed. Native
 64roots=forwards=outputs/432steps perarm/peak1|4/batchmax1. Singlepass only,
  no C4win/confidence claim. Periodic146path MAIN↔ROOT convergence preserves
  parents/freshmtimes; assembled gate5 PASS1645Rust/0failed/86ignored/265suites
  plus complete scripted checks. Later common preparation/cache work requires
  its own assembled gate and fresh physical/API checks. Models parallel.
- [ ] Remaining .8B C4 optimization: source-safe HF physical coalescing/batching.
  Current CLI rejects native_rows, HFloaded batchNone/native_batchfalse and
  worker excludes originalHF; existing B2/B4/B8 BatchProgram requires GGUF F32.
  Reuse root-cohort validation/publication, not incompatible math admission.
  Unchanged N1 completioncohort needs genuine private lanes or registered
  cohort epochs/status taps; shared RMS/value/readback reuse must await actual
  completion. True BF16 B4 needs its own typed arithmetic/source/state lowering.
  Existing genuine47/57/62/67 B1 prefills then four distinct cached frontiers
  provide strong mixed-row/current-root oracles without another capture; test
  row permutations/all24boundaries/all42state/fullV/F32/feedback/fault closure.
  This is source planning; no B4 implementation or causal gap percentage claim.
- [ ] After source/state/natural-serving parity, execute established short
  C1/C2/C4/C8 comparisons with unchanged source/profile/EOS/caps and original
  lifecycle controls. Then apply measured substantial generic improvements and
  rerun those profiles before long contexts and dense27B. Research alone adds
  no throughput/TTFT/goodput result.
- [ ] After the already authorized matched benchmarks complete, implement
  graceful cancellation plus explicit forced/immediate cancellation with bounded
  escalation deadlines. Add hooks for detecting stalled progress, repeated model
  loops and runaway CPU work; allow the supervisor to break them cleanly. Define
  request/session/model scope, cooperative drain versus forced termination,
  timeout policy and observability. Release GPU/native/source resources only
  after confirmed completion/termination; keep unknown work fenced/quarantined.
  Cover disconnect, stuck execution, timeout/escalation and healthy-survivor
  recovery with real physical tests. Owner directive22Oct: defer this extension
  until after benchmarks; it must not delay the active parity/benchmark path.
- [ ] Compose stable leased rows/diff metadata, compact readout, qualified
  production graph regions, exact fusion and true temporal prefill inside the
  existing worker. Price extra graph pools/padding, output backpressure and
  prefill/decode fairness; unknown capture/launch completion cannot eager-fallback
  through a fenced service.
- [ ] Extend full-state branch verification before any proposer: prefix0..K,
  target correction, all recurrent/conv/KV/indexer/draft/RNG/grammar domains,
  EOS/stop and canceled/stale epochs. Compare n-gram/Recall/suffix, compatible
  trained MTP/block drafters and K=0 using useful complete-cycle work, not assumed
  acceptance or source kernel counts. Keep explicit sampler/distribution profiles.
- [ ] Retain profile/shape/layout/source/context-grant compatibility at every
  batch/prefix/graph/proposal boundary; dynamic policy chooses only already
  qualified compatible variants. The user-facing profile combines pipelines,
  while original roots, physical admission and actual completion authorize work.

Research/priorities: [mechanism comparison](../fabric-vnext/PRIOR_ART_AND_REVALIDATION.md),
[optimization tasks](../cpu-gpu-optimization/TODO.md),
[recursive composition](../pipeline-composition/TODO.md).

### Batching and paged-state follow-up — 21 Oct design only

Implementation and acceptance detail: [batching/paged state](BATCHING_AND_PAGED_STATE.md).
Existing GGUF CUDA B2/B4/B8, refill and one-teacher-token prefix waves are
accepted only in their recorded scope; the inspected HTTP path remains
context512/active<=8. The following work is pending, not new qualification
or a change to the active correctness/benchmark/27B order.

- [ ] Integrate both accepted GGUF/reference and newly qualified vLLM numerical
  profiles through the same session/worker/HTTP path. Bind profile/phase/schema
  to compiled plans and cache compatibility; reject incompatible state reuse.
- [ ] Implement original-plan paged KV append and block-table QK/softmax/AV,
  preserving each profile's masks, positions, dtype/reductions and ownership.
  Qualify independent B1/B2/B4/B8, page boundaries and full logits/state;
  allocator/hybrid prefix/COW dependencies live in the
  [memory plan](../memory-and-streaming/TODO.md).
- [~] Extend bounded refill/regroup into fair aged chunked prefill/decode
  scheduling; qualify actual multi-token temporal prefill separately from
  request batching. **15 Nov fair-interleaving sub-slice accepted:** the shared
  engine owns a bounded fair-round policy above existing source/phase wave
  formation. Every eligible decode row is selected first; young prefill is
  deferred without dropping its original state/source owners; after two complete
  deferred rounds the oldest prefill wave must make progress. Round admission
  requires enough logical-token credit for one maximum admitted prefill wave plus
  every active decode row and refuses invalid/unbounded age/budget controls before
  worker/session creation. Focused model-neutral coverage passes 18 wave tests and
  14 worker tests, including owner retention/age rotation; strict engine clippy
  and production `qwen35-cuda`/CLI checks pass. Physical unequal-arrival HTTP
  qualification passes both approved 0.8B/2B CUDA models: B is admitted only
  after A's first actual ID, the first shared B2 cohort is no earlier than A
  output index 4, both sockets remain drained, CPU IDs/text/usage are exact, zero
  failed/refused/cancelled/discarded work, and final device use returns 41MiB.
  The complete `scripts/check.sh` command set is terminal-green on this exact
  source under isolated `native-gpu`: formatting/architecture, full GPU feature
  boundary, CPU clippy/workspace tests, minimal/core isolation, cache/source
  reference, serving/lifecycle, acquisition, release/installer/shell/Nix and
  qualification self-tests all pass. The monolithic connector wrapper itself
  disappeared during ordinary cached builds on repeated attempts without a
  compiler/test error, surviving child, cgroup OOM event or GPU residue; do not
  relabel those wrapper attempts as passes. This fair scheduling sub-slice is
  accepted from the complete constituent gate results. The broader item remains
  partial. The reusable native B2/B4/B8 equal-chunk candidate chain executes
  2..32 host teacher tokens through private candidate domains, keeps original
  session roots unpublished until one final all-row Ready, aggregates
  state/logit/readback/dispatch receipts and publishes one logical root advance
  for the whole chunk. **15 Nov 0.8B B2 functional physical acceptance:** the
  retained small `zerogravity-gpu` qualifier is now `temporal_prefill`, shared
  across B2/B4/B8. B2 proves abandoned 4-token rollback, one-root 4-token state
  publication, one-root 2-token terminal greedy, CPU/B1/full-all42 exactness,
  exact producer and final reserved0. The actual production CLI HTTP path passes
  prompt10 as chunks4+4+2 with CPU ID264/text ` a`, usage10/1, exactly3 B2
  publications, zero failures/refusals and final41MiB.
  **15 Nov temporal production checkpoint:** original2B independently
  passes full B2/B4/B8 resident qualification with CPU/B1/all42,
  rollback/publication/greedy/producer exact and final reserved0; retained bytes
  are 1,589,873,428 / 1,833,045,908 / 2,319,390,868. Two unsupervised B4 runs
  disappeared before temporal work; a heartbeat-supervised run completed exactly.
  Signed `eaec62d` promoted B4 and signed `8e6f14d` promotes **B8**. Production
  HTTP passes original0.8B and2B B4 as three exact4+4+2 B4 cohorts/publications with mixed
  JSON/SSE CPU text/usage exact, zero failed/refused work and clean retirement;
  B8 passes the same exact production contract on both sources. Retained2B B2 HTTP
  also passes. B2+B4+B8 now all select temporal4 with equal-frontier prefill.
  One earlier supervised0.8B B8 attempt remains incomplete. Canonical
  release-profile B2 remains open after another heartbeat-supervised thin-LTO
  build disappeared during CLI compile/link before the test ran, without terminal
  child/Rust/OOM/Xid/residue evidence; do not loop it.
  **Paged-state phase:** `0027c28`/`54340a6`/`1eb6e0f` establish the logical
  table, finite resident page ownership and actual CUDA COW materialization.
  Signed `3ac2059` adds a typed `StateLayout::Kv` page representation plus
  bounded gather-to-contiguous adapter without selecting it in sessions. K/V
  tables have distinct compatibility identities and paired frontiers. Final-byte
  RTX3060 page-boundary proof uses BF16 row512/page4 and a 4-page cap:
  frontiers3/4/5 gather byte-exact to contiguous (dispatches2/2/4), 3->4 performs
  two COW copies, 4->5 shares full first pages/reuses backing, prior state is
  immutable, inactive tails zero, capacity refusal and final retirement pass.
  Lower materialization and submitted-unknown quarantine cells remain green;
  production `qwen35-cuda` composition and strict clippy pass. The implementation
  also separates pre-submit fresh-target validation from post-Ready candidate
  validation, a lifecycle bug found by the first physical boundary attempt.
  At `dc6a188`, canonical session selection was still disabled; that commit adds
  the missing GPU-only bridge needed for the later wiring:
  transient `ResidentPagedKvStaging` exposes the exact old
  `StateBuffers::Kv` ABI, and private page appends can scatter only new rows
  from contiguous candidate staging under the same original flight while COWing
  the partial prior tail from immutable pages. Physical prior3->candidate4 passes
  with gather2, prefix-copy2, scatter-COW2, scatter-append2, no host readback
  after input, exact parent/child bytes and full retirement. The retained boundary,
  materialization and unknown-completion cells plus production composition/clippy
  remain green.
  Signed `53336b3` now lands the canonical `Domain`/root substrate itself:
  `StateBuffers::PagedKv` is a marker, `Domain` retains the actual paged owner
  and exact page lease/generation cohort, validation checks the original
  `StateRequirement::Kv`/source/service/readiness contract, legacy/native
  consumers fail closed, and single-session commit uses the factored original
  publication sequence. Physical roots0->3->4 publish successfully with2 page
  owners per committed frontier, exact displaced parent bytes and final release;
  all prior paged cells plus production composition/clippy remain green.
  **Production selection remains open.** Admission analysis rejects the earlier
  two-staging sketch: paged committed state + full prior staging + full candidate
  staging becomes three state envelopes at full context. Signed `ef9e4d4` now
  lands the one-staging execution contract: `StateBindingMode::KvInPlace`
  requires exact same fresh K/V lease identities for prior/candidate, and the
  actual ordinary `KvAppend` emits zero preparation launches instead of
  zeroing/copying the already-gathered prefix. Physical base3/capacity8 pass
  preserves the prefix, writes the new row exactly, leaves the tail zero and
  retires staging; all retained paged cells and production composition/clippy
  remain green. Next replace ordinary root allocation with hybrid admitted
  vector/page/staging pools, then select the paged route.
  Signed `93468b1` closes the Ready/snapshot semantics first: paged snapshot
  reads are typed, ordered and valid-tail bounded; physical partial/full snapshots
  plus4->3->4 restore pass. Ready classification permits only exact initialized
  full same-plane/index current-root pages as inherited, so3->4 has2 Writes and
  4->5 has2 inherited+2 Writes. The new pure hybrid inventory plan proves
  vector-root + staging + page maxima fit the old roots*state_bytes envelope and
  refuses insufficient active-fork headroom or heterogeneous page geometry.
  At `93468b1`, production selection remained open with physical pool
  construction and explicit page policy still pending.
  Signed `cb57e1d` now constructs the actual vector/staging/page pool owner and
  source-owned installer. Physical vector4+staging1+page24 residency equals the
  old67584-byte root envelope exactly and retires to0. Initial hybrid roots are
  zeroed vectors + empty paged KV; candidate allocation intentionally refuses
  until a current-root-aware path exists. Stable compatibility IDs use explicit
  numerical tags and checked physical/slab/capacity packing.
  Signed `b9b04aa` now lands the one-flight current-root candidate path itself.
  One `HybridCandidateBundle` owns Vector candidate + complete staging + all KV
  append plans; exact access deduplication merges current Reads and submitted
  Writes; gather precedes the unchanged ordinary program, scatter follows it, and
  only Vector + fresh/COW pages are promoted. Paged Domains materialize only
  after Ready. Physical one-Vector/two-KV base3->4 passes with gather4/scatter8,
  five Ready outputs, exact canonical Ready order/frontier4 continuation/zero
  tails and final resident0. The exact-current eight-cell suite and production
  strict checks pass.
  Signed `1413f72` adds the explicit low-level CUDA source-state policy while
  leaving public/default serving Contiguous. Paged source loading now requires
  `Paged { page_tokens, max_append_tokens }`, B1, and bounded positive policy;
  inventory/runtime append bounds are enforced. Invalid paged/native/bounds
  policy refuses before source IO. The original-HF differential compares
  Contiguous vs Paged full-vocab/all42 at page-1/page/page+1/2page+1.
  The approved managed0.8B/2B cache inputs are now visible on L5; original0.8B
  page4/append4 passes full-vocab/all42 exact at frontiers3,4,5,9.
  Signed `398e446` adds the matching source-real lifecycle target: prelaunch
  cancel, abandoned Ready, cancel-after-Ready discard, page-boundary commits and
  exact all42 restore to pre-boundary/boundary roots, final shutdown.
  Signed `3041457` corrects restore qualification to fresh root/epoch
  publication at the saved frontier. Cold-paged0.8B lifecycle is accepted by
  retained dotlink job `9feb1394...`: exit0,168.08s,
  `contiguous_warmup=false`, exact cancel/abandon/restore/all42/shutdown.
  Signed `e6aac53` removes the warmup/stage scaffolding and leaves the qualifier
  cold-only. Other intermittent null-exit jobs are signalled-Bubblewrap
  infrastructure events, not ZeroGravity failures. Correction: deployed dotlink
  v0.17.3 does not have main's `SharedProcessExit::Signal`; it records
  `ExitStatus::code()==None` for a signalled child, and its sandbox child is
  Bubblewrap. Timeout/cancel/failure use distinct statuses and Completed disarms
  containment before recording. Controlled inner/sandbox-shell SIGKILL paths
  report numeric137, while failure-time cgroup OOM/pids and restart/poll-failure
  evidence are0. A direct no-inner-supervisor cold run still reproduced null exit.
  **Original2B page4/append4 source-real acceptance is now green.** The repaired
  local native-GPU harness executes inside the existing Bubblewrap namespace.
  Dotlink v0.17.4 forensics correct the earlier interpretation: it logs every
  stop it initiates, and the relevant window shows only the explicit
  `job_cancel`; a silent long sleep survived until cancellation. Thus earlier
  SIGKILL9s are outside dotlink stop paths and do not establish a quiet-job
  watchdog or heartbeat requirement. With the model foreground and a heartbeat
  sidecar used only as harness plumbing,
  differential `38c1f321...` exits0 in703.17s with full-vocab/all42 exact at
  3/4/5/9, and cold lifecycle `744f36fd...` exits0 in438.18s with
  `cold_load=true`, exact cancel/abandon/restore/all42/shutdown.
  **19 Nov true in-flight cancellation gate accepted on original0.8B paged B1.**
  The qualification-only launch cut pauses only after the ordinary CUDA flight has
  actually enqueued work and before Ready. Known cancellation then waits for a
  physical status readback and retires the existing event without initializing
  submitted Writes or creating Ready: root/all42 remain exact, Vector/staging/page
  backing stays held until completion, returns to the exact preflight occupancy,
  replay commits successfully and final shutdown is clean. Dropping the same
  submitted future instead preserves the accepted fail-safe path: root binding is
  unchanged, service fenced, one event pending,396 submitted handles plus source
  completion owners quarantined and no pool reuse. Two externally SIGKILLed runs
  before terminal semantics remain incomplete; retained direct known-completion
  and heartbeat-assisted unknown-completion runs both exit0. Heartbeat is optional
  transport plumbing only. **19 Nov explicit high-level selector accepted without
  changing defaults:** public `QwenCudaStatePolicy::{Contiguous,Paged}` is exposed
  only through `ModelLoadRequest::QwenCudaWithStatePolicy`; that historical fixed
  `QwenCuda` Contiguous default is superseded by the21 Nov generic smart-config layer.
  Paged now admits native rows4/8
  through the same Hybrid owner; zero/out-of-context bounds still fail before
  source IO. Native B2/B4/B8 cohort state transport is implemented by reusing
  `OrdinaryCandidateState` gather/body/scatter per lane under one cohort flight.
  Contiguous B8 keeps its private-KV fast path; paged `KvInPlace` uses existing
  per-lane KV append. Focused policy tests, strict CUDA clippy and CUDA-serving
  composition check pass. The compact original0.8B Contiguous-vs-Paged B2/B4/B8
  prefill/decode/all42 cell is terminal-green: exit0 in306.84s, exact state/output
  and clean shutdown; two earlier external-SIGKILL attempts remain incomplete only.
  Public SDK paged serving is also terminal-green:8 concurrent page-boundary
  requests,2 outputs, native B8 observed for both Contiguous/Paged, exact token IDs,
  clean shutdown in340.78s. **Paged HTTP is now terminal-green** under the explicit
  child-private-net wrapper accepted by `4688cba`: the same8 concurrent page-boundary
  workload observes native B8 for Contiguous/Paged, exact token IDs and clean
  shutdown in358.07s. Strict test-target Clippy passes. Direct shared-net L5 still
  refuses `--require-private-network`; no isolation was weakened. Fresh matched-vLLM
  timing remains open because no local vLLM runtime is present and retained C8 data
  is not this workload. Full-gate composition found and fixed one generic
  `cuda-native` import boundary for shared hybrid inventory; targeted production
  `cuda-native` now checks cleanly. The warmed monolithic rerun was externally
  SIGKILLed later during ordinary-Qwen feature compilation with no diagnostic and
  remains incomplete. Matched-vLLM rerun is blocked by the retained pinned layer
  store no longer exposing the exact image ID to Podman from this connector; exact
  HF source remains intact and no reconstruction was attempted. **Prefix fork/COW
  substrate now passes physically:** fresh session, independent root, no replay,
  paged partial-tail3->4 COW, exact full logits/all42, immutable cached snapshot,
  clean shutdown in179.13s. **Bounded semantic prefix lookup/eviction is now
  implemented** in the shared engine with exact grouped compatibility, exact units,
  greatest-prefix/newest-tie lookup, duplicate refresh, deterministic FIFO eviction
  and explicit entry/weight caps; immutable payloads are `Arc`-shared and lookup
  allocates nothing. Two focused tests, strict engine Clippy and CUDA composition
  pass. **Scoped worker hit/miss is now accepted.** A validated nonzero
  `PrefixSharingScope` travels beside the queued request; ordinary `request_text`
  remains isolated and explicit `request_text_scoped` opts in. Generic worker restore
  verifies its cursor against actual executor frontier and caches only already-
  committed state-only prompt roots. Qwen owns a paged-only model-local bounded cache;
  contiguous state is unaffected. Physical0.8B page4/max_append64 prompt65 produces
  scheduler model-token deltas65 warm /1 same-scope /65 different-scope with exact
  output ID and clean shutdown in172.98s. The first run exposed and fixed retirement
  ordering by clearing cache snapshots before resident shutdown. **Trusted HTTP
  scoped hit/miss is now accepted:** `ServerBuilder::prefix_sharing_scope` supplies
  one fixed optional scope from the embedding application; no request field/header
  can set it, and unaware services reject scoped admission. Physical0.8B
  page4/max_append64 prompt65 over HTTP records scheduler model-token deltas65 warm /
  1 same scope7 /65 second-server scope8, exact output ID and clean shutdown in
 171.97s.7/7 server tests, strict serving Clippy and CUDA-serving composition pass.
  **Bounded live-snapshot eviction/cache invalidation is now accepted:** trusted
  `invalidate_prefix_scope` removes only currently retained entries for one scope and
  returns exact entry/weight totals; it does not cancel already-admitted work. Safe
  stale-scope handling is stop/drain admission, then invalidate; server lifecycle/
  scope switching alone does not purge.
  Qwen cache count=`2*active`, weight=`active*context`. Physical0.8B active1 proves
  FIFO prefix64 oldest miss/newer hit, purge2/128 + invalidated-scope miss, and512-token
  retained-weight pressure where prefix320 hits but prefix256 was evicted; final
  purge2/448, repeated purge0, clean shutdown in179.12s. Actual resident page-pool
  pressure is now qualified too: original0.8B page4/max_append64 cached prefix64 plus
 5952 externally held real pages fills6144/6144; the next tail returns prelaunch
  Capacity with unchanged root and no vector/staging leak, then exact full-logit
  recovery after release with all pages idle at shutdown in168.51s. One startup
  SIGTERM attempt remains incomplete only. Minimal Qwen diagnostics expose current
  prefix-cache occupancy/bounds and paged-pool allocated/idle/occupied/max/page bytes;
  no cumulative hot-path counters. Host/device OOM is not claimed. Physical overlap
  remains behind event/lease and independent-root gates.
  **Fresh overlap review checkpoint:** do not widen worker scheduling yet.
  `PhysicalQueue` is still single-active per physical card, `ResidentSubmission`
  completion is tied to trusted readback/physical retirement, and Qwen CUDA is pinned
  to one logical stream; more runnable work would therefore not establish physical
  overlap. The lower CUDA substrate now has a same-service fresh secondary stream and
  non-clone actor-owned event. Physical L5 proof records primary+secondary events
  before either wait on disjoint8MiB buffers; secondary was still not Ready on first
  query and both readbacks were exact (`P12_CUDA_SECONDARY_STREAM_EVENT_PASS`). This
  is explicitly a prerequisite, not a throughput/TTFT/ITL or prefill/decode overlap
  claim. **Resident event/lease in-flight ownership is now accepted:** one armed
  `ResidentSubmission` and one exact CUDA event stay coupled until explicit resolution.
  Known completion still requires original readback/validation before Ready;
  cancellation retires after event completion with writes uninitialized; unresolved
  drop fences/quarantines and retains the attached completion owner. Physical marker:
  `P12_CUDA_RESIDENT_INFLIGHT_EVENT_PASS ... queue_width_unchanged=true`.
  **Bounded multi-flight substrate is now accepted:** configurable active-ID set in
  `PhysicalQueue`, production registry still width1, and shared first-Submitted/
  last-Ready accounting over one prepaid model envelope. Existing event-ledger access
  admission remains authoritative. Physical bounded2 dual resident flights on distinct
  CUDA streams/outputs retire in reverse order with exact8MiB readbacks and reusable
  service (`P12_CUDA_RESIDENT_DUAL_FLIGHT_PASS`). Real Qwen/source feedback is now in:
  existing public B8 SDK,8 concurrent original0.8B,48+3, one warm then measured wave.
  Exact native B8 passes, but steady contiguous is4888.231ms/4.910 output tok/s and
  paged5080.588ms/4.724 tok/s. Mean first-token commits are4523ms/4567ms respectively,
  so prefill dominates; decode is ~44 tok/s contiguous and ~31-32 tok/s paged. The
  immediate bottleneck is not queue concurrency **or weight streaming**. Revalidation
  proves active8/context512 remains fully resident:18-root total2,076,877,813B plus B8
  runtime reserve still fits the4.5GiB provider envelope. The generic prefill cohort
  shares projections/stateless RMS but serializes the remaining stateful operators by
  lane; decode already has cached GDN batching and one-token B4/B8 attention. The
  stateless QK-RMS/gate/multiply/SiLU prefill lowering is retained: exact warm48+3 B8
  improved4888.231→4696.591ms contiguous (~3.9%) and5080.588→4907.715ms paged (~3.4%)
  versus the retained no-extension baseline. The historical captured engine
  is fixed B4/47 and qualification-only, so it is not a clean serving fallback. Next
  multi-token attention is now a reusable shared primitive and is wired into equal-row
  B4/B8 prefill groups; the warm48+3 wave stayed exact but TTFT was effectively flat
  (4696.591→4701.433ms contiguous;4907.715→5022.809ms paged), so attention is not the
  dominant remaining cost. Shared multi-token GDN prefill is now accepted: generic
  sequence-aware GDN geometry/workspace plus typed private-root import/export drives one
  B4/B8 cohort runtime over packed activations. Exact warm48+3 B8 moved4701.433→1007.301ms
  contiguous and5022.809→1160.106ms paged in the default unoptimized test profile.
  That profile is not a serving-performance baseline. The same commit/existing cell
  rebuilt non-LTO/opt1/codegen64 is183.932ms/130.483 output tok/s contiguous and
  242.488ms/98.974 paged, with exact tokens/native B8. This already clears the retained
  historical eager-vLLM B8 72.20 tok/s short-wave baseline on contiguous. Future perf
  feedback must use a matched optimized host profile. Continuous refill/regroup now has
  real public-SDK feedback too: active8/native8 with4 short +4 long survivors +4 queued
  rows exposed and fixed a reusable band-layout defect. Width8 cohort storage now retains
  exact B4 and B8 GDN ABI descriptors while sharing one max-width physical scratch arena;
  legal B4 fallback no longer binds B8 extents. Warm refill passed with `{4,8}` on every
  survivor/queued row, demonstrating B8→B4→refill→B8 regrouping. Opt1 goodput for68
  generated tokens:178.769 tok/s contiguous,136.488 paged. The first matched paged-gap
  optimization now removes host-zero allocation/H2D from paged KV staging and fresh-page
  initialization by reusing the bounded pinned device-side word-zero primitive. The same
  optimized8x48+3 SDK cell remains exact/native B8 and measures184.409ms/130.145 output
  tok/s contiguous versus205.724ms/116.661 paged, improving paged from the retained
 242.488ms/98.974 without moving contiguous materially; first-token commits are
 133.988/133.531ms. Physical zero work is now counted as dispatches, including Hybrid
  scatter accounting. The next shared transport slice now collapses per-page KV
  gather copies with a fixed15-page-pair/2-destination pinned ABI under the existing
 32-pointer ceiling; larger prefixes chunk through the same runtime-scalar kernel. The
 48-token/page4 workload therefore gathers12 key/value pages together instead of issuing
 24 page-copy launches per KV domain. The same matched cell passes exact/native B8 at
 176.800ms/135.746 tok/s contiguous and182.505ms/131.504 paged, leaving ~3.2%
  matched wall-time overhead. The final paired gather now also zero-fills the inactive
  staging tail when the bounded span permits it, eliminating the remaining two staging
  zero launches for this cell while preserving the explicit fallback for oversized or
  empty-prefix cases. The same matched cell passes at177.467ms/135.236 tok/s contiguous
  and180.520ms/132.949 paged, only ~1.7% overhead. The production refill shape was
  then rerun on unchanged `e72030c`:4 one-token +4 eight-token survivors +4 queued
  eight-token requests,68 generated tokens, matched opt1/codegen64/non-LTO. It passes
  at400.383ms/169.837 output tok/s contiguous and422.572ms/160.919 paged (~5.5%
  wall-time overhead), materially improving the retained old paged136.488 tok/s result.
  A device-scatter append-copy+tail-zero fusion was measured at421.159ms/161.459 paged,
  only ~0.3% faster than accepted source, and was removed rather than retaining another
  kernel path. The temporary refill probe was removed too. Fresh vLLM remains
  unavailable and is now explicitly owner-deferred. The admitted CUDA context ladder has
  now advanced through **4K,8K and16K on both0.8B and2B public SDK serving**:
  context-2 prompt +2 output, exact replay on a new session, exact GPU producer frontiers
  and clean shutdown. The retained qualifier is current-path capability evidence and
  does not substitute for the missing independent all42 populated archive. The first
  0.8B16K attempt was externally SIGKILLed during Rust compilation before test entry;
  clean residue/GPU/kernel checks plus a cached retry pass make that an infrastructure
  event, not a model result.2B16K passes too. The32K slice then invalidated the old
  admission assumption rather than producing a serving pass: the retained qualifier now
  accepts32768;0.8B32K was externally SIGKILLed twice soon after test entry, while2B32K
  public model load succeeded and the real qualifier ran~230s before external SIGKILL.
  Clean residue/GPU/kernel checks followed each termination. The old2B32K refusal came
  from the pre-streaming resident-weight calculation; shared source admission has had
  streamed-weight fallback since `d16f096`. Current32K storage is now closed from
  the loader's exact shared preflight:0.8B Resident total=3,427,684,813B;2B Streamed
  total=1,925,110,221B. The external SIGKILL also reproduces twice during compile-only
  current-source `cargo test --no-run`, before test entry, with no dotlink-cgroup
  OOM/CPU/pids-limit event and idle GPU. Do not force a refusal, add another context
  fixture, or retry-loop32K. Two later direct trace runs on the exact already-built
  optimized0.8B32K public-SDK binary produced one full pass and one external kill:
  run1 completed in185.18s with exact replay, independent sessions, exact producer
  frontiers and clean shutdown; run2 received `SIGKILL from outside` at~80.7s after
  >60s execution. So0.8B32K serving capability is now positively established once, but
  the old external-run reliability issue no longer obscures2B32K. Fresh matched optimized
  contiguous reruns stay alive normally and reproducibly hit the retained SDK's600s
  first-response timeout at~616–617s total, with no SIGKILL. Simple source-placement
  experiments (partial residency, state-path-first pack ordering, larger serial page
  through16MiB) did not materially improve that boundary. The actual fix is explicit
  paged-state admission with physically reachable COW-page pricing: it preserves the same
  four logical lifecycle roots while keeping all2B32K weights resident. Same32K capacity,
  1024-token public A/B is283.22s contiguous-streamed vs28.56s paged-resident (~9.9x),
  and the retained full32766-token paged run passes in224.80s execution/260.34s wall with
  exact replay/frontiers and clean shutdown. Canonical `QwenCuda` now runs Enabled smart
  config and jointly resolves state representation + native cohort width from exact current
  admission: resident weights first, then widest qualified B8/B4 cohort, then Contiguous
  as a tie-break. User values still win per knob under Enabled. The existing active8 SDK
  cell with native width unspecified Auto-selects8 and physically publishes `{8}` for both
  Contiguous and user-Paged halves at135.979/133.004 tok/s. No benchmark result
  from the earlier killed
  short2B B8 attempt is retained; do not return to residual paged micro-tuning.
- [x] A clean dynamic-prefill grouping candidate was measured and rejected from the
  current selector rather than discarded globally. For the already-compatible mixed
  native-B8 row pattern
  [49,48,49,48,49,48,49,48], the candidate clustered private physical lanes by row
  count while preserving scheduler/publication order, then reused the existing B4
  attention kernel for equal-row quartets with per-lane leftovers. It added no allocation
  or new kernel and stayed singleton-exact/native-B8, but the matched measured wave moved
  from299.985ms on current source to324.231ms and313.585ms on two candidate runs
  (~8.1%/~4.5% slower). The Qwen-specific wiring/probe was removed, while the reusable
  fixed-width regroup implementation is retained dormant for cross-model A/B. Do not
  select it for this .8B cell merely to reduce attention launches; measure it separately
  on Qwen2B/27B, Gemma4 or other eligible shapes. The later21 Nov restoration sweep
  generalizes that rule to historical CUDA experiments: reusable physical mechanics are
  dormant candidate components, while old serving selectors/runners remain removed.
  Cross-model A/B can now reuse the retained lane regroup, transport, GEMM,
  recurrent-transpose and GDN-occupancy implementations without changing production
  routing.
- [ ] Run profile-matched batching/paging/prefix-hit/overlap ablations and genuine
  pinned-vLLM comparisons: established short C1/2/4/8 first, then qualified long
  contexts. Record quality/full-state acceptance, true ITL/TTFT tails, committed
  SLO-goodput, offered/rejected/canceled work, fairness, actual dispatch/copy
  counts and physical memory/fragmentation. Preserve old-profile regressions;
  code presence or fewer launches cannot close serving/performance parity.

### Resumed correctness-first continuation — 21 Oct

- Latest owner order: .8B/2B genuine vLLM whole-model/source/state parity,
  then benchmarks, then qwen:27b. Actual timing is held until parity closes.
- All4 original short-profile policy/quality preflights complete, including2B
  ragged. Original owners/cgroups/cleanup validate; IDs/text/usage still differ.
  Numerical equivalence and functional-quality parity remain unqualified.
- ROOT executed12 runner fault tests and5 aggregation fault tests: PASS.
  Timed C1/2/4/8 matrix remains unexecuted; no new concurrent speed,
  SLO-goodput or hardware-count claim. Existing87e1 GGUF profile inputs do not
  establish equivalence with the new HF CUDA correctness profile.
- Native MLP and activation leaves pass565248/247296BF16 words. Complete
  connectedlayer0 fivephases now PASS4071072BF16+2171024F32 exact with actual
  computedcache and separatecarriedresidual/sourceWunchanged/host0. Leaf
  cancel/pressure/unknown and4whole-blockfaults pass. Whole-model/
  all42state/serving parity remain open; timings held and27B not started.
- ROOT checklist.
  Rebind any future benchmark to its actual accepted source/profile/binary;
  preserve paused SiLU work. No new unit merged or committed.

- [ ] **P12 ordered SiLU integration:** private CUDA B1/B2/B4/B8 operator
  passes204 physical old-GPU/CPU differential cases,171562710 exact bit
  comparisons and five fresh failure/abandonment processes. All17 intermediate
  checkpoints, status, canaries and original lease/quarantine ownership pass.
  Independent source review and candidate CUDA/Vulkan/default strict checks
  plus two row-partition tests pass. Candidate library, qualifier, server,
  worker and HTTP release artifacts compile; the exact-source diagnostic
  file-transport unit passes. Paused at the owner's request before starting
  the prepared60-case physical matrix. Full-model integration remains unaccepted:
  complete PTX/loaded-code witnesses, model/StateRoot/serving regressions, both
  original release parity routes, isolated full gate and fresh timing/count
  comparisons remain pending. Existing truncated PTX dump filenames overwrite
  specializations; retain that failed evidence and qualify a collision-safe
  diagnostic before claiming complete instruction coverage.

- [ ] **Stock2B loader and matched benchmark:** genuine pinned stock2B
  V4 plain/observed loader and independent comparison now pass: all335
  stored tensors inventoried, exact15 MTP exclusions and320 backbone roles;
  both complete vocabularies,496640 native BF16 logit words, remain exact.
  Source/runtime and owned process/group/container/EOF cleanup pass; retain
  the original V3 observer refusal. Both model loaders are prerequisites,
  not cross-engine/full42/default-graph or throughput qualification.
  Actual0.8B eager HTTP policy passes for both providers: identical47 prompt
  IDs, original Rust tokenizer, effective greedy/EOS settings, observed
  cgroup12GiB/swap0/CPU4 caps and authenticated GPU process membership.
  Original owned cleanup closes. Podman unresolved HPIDs establish no
  ownership; every sampled GPU PID must match the numeric owned subset.
  Generated IDs differ under the declared activation/cast profiles. Actual
  0.8B full-output quality covers seven prompts; quality equivalence remains
  unqualified. First unprofiled eager/cold-process/solo comparison completes
  all eight ABBA/BAAB samples: four requests/32 raw tokens per provider.
  Median TTFT13.400s native versus1.561s stock; stock is faster here.
  Cold-process means first request after readiness, not storage-cache cold
  or complete startup latency. Both providers are internally stable, but
  cross-engine IDs/text/usage differ. One stock sample coalesces delivery;
  individual-token ITL and source-correct SLO-goodput remain unqualified.
  The2B eager/cold-process/solo run also completes all eight samples:
  four requests/32 tokens per provider, identical raw IDs/output hashes/
  usage/finish on that one prompt. Median TTFT14.540s native versus1.562s
  stock; all samples have individual token delivery, native ITL p95
  0.186–0.199s versus stock0.026–0.030s. Seven-prompt2B quality remains
  unqualified despite five matching full texts; this prompt establishes no
  numerical equivalence. Finish default-route, warm/concurrent workloads
  and physical counts;
  this bounded observation does not establish numerical or serving parity.

- [ ] **Production CUDA graph admission:** signed `5bf440d` first made the
  graph identity/pin/replay/owner substrate available to production `cuda-native`
  without selecting it. Signed `db7be60` now adds explicit opaque graph-resource
  admission through the **same** resident ledger/provider envelope used by managed
  buffers and other foreign driver resources; no second ResourceArbiter domain was
  added. The policy reserve is deliberately labeled policy, not measured CUDA
  internals:64KiB control + one4KiB page per prepaid source-dispatch slot + Rust
  graph/receipt/lease-access-vector metadata. It is reserved before `graph_prepare`,
  can grow only before capture starts, must cover the captured dispatch count,
  travels with CaptureWindow/CapturedResidentGraph/quarantine owners, and releases
  only after the existing trusted retirement path. Capture and retirement queue
  work declare that already-prepaid footprint as `AllocationClass::Other`; the
  physical queue remains scheduling-only and never charges the global envelope
  twice. Actual CUDA driver graph bytes/node enumeration remain **unqualified**.
- [x] Graph resource host/compile gates for `db7be60`: plain production `cuda`
  remains graph-free; historical `qualification,cuda` and production
  `cuda-native` share the graph substrate. Production `cuda-cublas` and
  qualification+`cuda-cublas` strict Clippy pass, real
  `zerogravity --features qwen35-cuda` and qualifier feature-union checks pass,
  graph policy1 + topology/identity3 host tests pass, and existing opaque-resource
  ownership tests pass7 foreign-charge +3 detached-budget cases including
  shared-cap concurrency and poison/fence retention. External
  `ResidentModel::capture_*` / `prepare_graph_*` remain qualification-only and
  ordinary serving still constructs only `FusedExecution::Native`.
- [x] Signed `6a5c610` closes the physical CUDA compiler blocker and accepts
  the prepaid graph charge on the RTX3060. The original generic poison was
  localized to `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`: non-persistent C++ NVRTC
  emitted CUDA13.4 PTX newer than the installed driver accepted. Non-persistent
  compilation now emits exact-SM cubin/SASS; persistent compilation still emits
  and caches PTX. The finite compiler owner no longer treats an unrelated CubeCL
  environment-generation bump as module uncertainty; every new compilation still
  revalidates its captured SDK/include configuration, finite cumulative accounting
  remains intact, and sticky poison now preserves the first concrete cause.
- [x] Exact original0.8B B2 graph qualification passes physically:
  warm launches `[3725,3725]`, captured body3632 launch sites, outside93 per
  replay, six graph host enqueues,14,900 committed kernels, full248320 logits and
  all42 state domains exact. Signed `f1ce532` extends retained graph metadata
  from bare leases to exact `(ResidentLease, DeviceAccess)` authority, making the
  current conservative policy charge **15,356,224 bytes** with32,768 bytes of
  owned lease/access-vector metadata. The resident foreign-byte ledger must equal
  that charge while the graph is live, live charge owners must remain retained,
  and foreign bytes must be0 immediately after trusted `graph.retire()`.
  Graph-band release then reclaims additional batch storage, final ResourceArbiter
  reserved bytes are0 and the RTX3060 returns to41MiB.
- [~] One later repeat observed a transient post-capture total resident-byte
  mismatch of exactly15,339,840 bytes with pending_events=0 and an empty physical
  queue. The run failed closed before replay. The condition did **not** reproduce
  in four consecutive subsequent physical runs; all four returned41MiB and the
  final strengthened run explicitly proved charge-retained-while-live=true and
  charge-released-on-retire=true. Do not weaken the idle invariant. If it recurs,
  the `81b7c71` witness reports warm/steady foreign bytes, graph resource bytes,
  charge count and Arc strong counts so the actual ownership boundary can be
  localized. CUDA driver-internal graph bytes/node enumeration remain unqualified.
  This is correctness/lifetime qualification only, not graph performance or
  production serving selection.
- [x] Signed `f1ce532` closes the same-card external storage/event-authority
  seam without selecting graph serving. Capture retains the exact internal
  `(ResidentLease, DeviceAccess)` set; before every replay the graph is attached
  as a completion owner and the armed resident flight must prove those exact
  allocations with compatible access authority. Thus a sibling event or replaced
  storage cannot authorize an otherwise-valid graph merely because geometry and
  graph generation still match. Production + qualification strict Clippy, real
  `qwen35-cuda` checks, graph policy/topology/resource ownership tests and the
  physical original0.8B B2 graph qualifier pass.
- [x] Signed `9d6613c` integrates the private production selector without enabling
  public graph serving. `FusedExecution::Production` resolves only under the existing
  allocation+workspace ownership boundary: an exact compatible live workspace cache
  becomes `Graph`; cold/spent/no-live-cache, device-token and incompatible typed
  profiles remain pre-enqueue Native misses; runtime identity/lifecycle drift is an
  error; an unresolved selector or graph replay failure never retries Native. The
  existing graph/program/storage/access/resource validation remains authoritative.
  Physical original0.8B B2 qualification proves one compatible selector replay,
  incompatible Logits->StateOnly Native fallback with zero graph replay/retirement,
  unchanged roots/state, retained15,356,224-byte graph charge, explicit retirement,
  fresh-workspace recapture, trusted B2->B4 replacement and final reserved0/41MiB.
- [x] Signed `599cb28` closes matched production-selector timing through B4/B8
  without rerunning the already accepted wide-graph arithmetic/state suite. Same
  process/model/workspace/full-logit readback, 4 AB/BA warm pairs +12 samples/path:
  B2 Native mean273,933us / median267,847us vs Graph264,881/263,238
  (~3.30%/~1.72%); B4 273,801/273,910 vs270,766/270,805
  (~1.11%/~1.13%); B8 362,559/360,951 vs359,915/359,649
  (~0.73%/~0.36%). Exact graph replay deltas are12; B2/B4/B8 retained policy charges
  are15,356,224/15,405,376/15,458,624 bytes; trusted retirement and final
  reserved0/41MiB pass. Earlier direct B4/B8 invocations that disappeared without a
  Rust result remain incomplete; supervised in-namespace reruns passed. The gain
  shrinks with useful width and does **not** justify enabling public graph serving,
  so ordinary serving stays `FusedExecution::Native`. Keep the private graph
  cache/selector as a reusable qualified capability for materially different future
  graph regions. CUDA driver-internal graph bytes/node enumeration remain unqualified.

- [x] **P12 prefix scheduling performance:** warm native-prefix evidence remains
  accepted for both models and the missing reconstructed cold-process cells now
  close for both original sources. Exact historical providers are `d0a6084`
  (max32-prefix) and `f788929` (native teacher-prefix); both are rebuilt under the
  current toolchain with the same benchmark-only exact-SM NVRTC cubin/SASS
  compatibility behavior matching later accepted `6a5c610`. Compiler persistence
  is preconditioned symmetrically; every measured sample is still a fresh process
  with no request warmup. Reconstructed source-real 8-request ABBA completes
  32/32 requests/model with exact token IDs/output hashes/reported completion
  usage/finish and zero timeout/refusal/failure/cancel/discard. 0.8B mean
  wave 17.4307s->15.4529s (-11.35%), TTFT p95 15.0368s->14.9527s and
  delivery-ITL p95 0.8188s->0.7804s. 2B mean wave 29.5280s->25.0937s
  (-15.02%), reported output rate 2.1693->2.5512tok/s (+17.60%) and
  delivery-ITL p95 1.5598s->1.3234s (-15.15%), while TTFT p95 **regresses**
  23.1625s->23.8931s (+3.15%). 2B old samples consistently execute
  B2/B4/B8=3/1/6 batches; new samples14/11/17. Earlier rebuilt-old 2B scopes that
  disappeared during first GPU work remain incomplete cold compiler-preconditioning
  attempts, not runtime failures: staged C1, B2, exact B4 and exact B8 physical
  diagnostics subsequently pass, including two B8 batches/16 rows, <=3261MiB
  observed device use, zero cgroup OOM events, and final 41MiB; the exact ragged
  old workload then passes before measurement. The original frozen binaries and
  workload file are gone, so these are reconstructed diagnostics from retained
  `f788929` HTTP prompts, not byte-identical reruns and not a vLLM/source-correct
  goodput claim. Genuine equivalent-source vLLM comparison remains open.

- [ ] **P12 fused continuous GPU serving:** extend the accepted CUDA B2/B4/B8
  HTTP path with paged hybrid KV/recurrent cache, physically safe
  prefill/decode overlap and fairness. Unequal decode frontiers and
  compatible refill are accepted within the B7 source/context scope.
  Extend full-logit/StateRoot qualification to other advertised trained
  profiles/backends and full GPU prefill. Measure actual p50/p95 TTFT,
  true token ITL, committed SLO-goodput, fairness, memory and physical
  dispatch counts against genuine pinned vLLM V1 with equivalent
  source/quant/tokenizer/hardware, cold/warm conditions and caps.
  Local release smoke and fewer launches do not qualify performance.
  Separate native0.8B B2 CUPTI diagnostic records64328 actual kernels and
  64328 successful direct launch calls, zero dropped records and exact source
  counter delta; plain/observed complete outputs and owned retirement match.
  A fresh accepted-baseline0.8B canonical C1 diagnostic also validates all
  four native/stock plain/observed processes:236270 actual native kernels
  and direct launches equal the source delta, zero dropped records/errors;
  exact provider-internal outputs and original owned retirement pass.
  Stock public profiler records9318 kernel activities/8810 host launches,
  but its losslessness/all-process coverage remains unverified. Independent
  validators and ROOT/peer metadata reviews pass. These observations qualify
  the native windows only; complete matched hardware counts, profiler device
  allocation admission, source-correct goodput and profiled timing remain
  unqualified. Equivalent default-route/warm/concurrent vLLM timing stays open.

- [ ] Connect accepted core `ContextOwner`/`ContextReader` snapshot
  capability to the real router's authenticated session/turn frontier:
  source-scoped application/shared grants, per-application and per-tenant
  ownership, immutable complete context records, explicit ON/OFF context
  selection, short-lock optimistic source epochs and cancel/revoke
  invalidation. Recheck `require_fresh` at TaskCoordinator delivery and
  bind source/graph generations to model/media StateRoot publication.
  DecisionAdapter is a typed callback boundary, not trained NanoJev
  logits or actual ASR context support. Cross-session/host source sync,
  bounded CPU staging and QoS, media quality A/B and revocation/stream
  cancellation remain open.

- [ ] One router must snapshot each enabled, source-granted context
  stream at the declared turn/graph epoch before live ASR. Default only
  committed user/assistant session content; explicitly selected tool
  call/result, application-owned system/developer prompts and authorized
  shared/cross-user records are eligible. Strict source ownership and
  requested grants, not a blanket category blacklist, govern access.
  No same-turn LLM feedback loop or unauthorized sources/tenants, raw
  unowned KV pointers or tool-secret leaks. The ON/OFF toggle leaves
  typed graph topology stable; OFF performs no retrieval. Bound
  configurable 4KiB..1MiB plus physical admission, ordered filters,
  typed decision models, credit/backpressure, stale ACKs and source-real
  accuracy/consent tests before live mode.

- [ ] Make YAML/repeatable-flag compiled pipeline configs a first-class
  source-bound router input: many model IDs, many typed component DAGs,
  many explicit protocol/route bindings within ONE process; one model
  can serve standalone chat and ASR→diarization→LLM→TTS. Preserve
  direct one-model CLI execution with no config requirement. Current
  pipeline check is pure scaffolding; implement model residency,
  admission, TaskCoordinator execution, generation/hot reload and
  authenticated OpenAI/SSE/WebSocket/WebRTC transport, then enable
  pipeline run/serve only after source/physical tests.
- [ ] Distinguish model-trained MTP weights and Qwen Flash-Next
  learned n-gram embedding bank from runtime n-gram/suffix
  speculative decoding (no extra source required). Verify
  accept/reject and complete rollback/grammar RNG across
  QSA/GDN/MoE before any speed claim.

- [ ] Implement canonical TaskRequest/coordinator workers and CLI/SDK/API integration, model routing/residency lifecycle, explicit supported OpenAI/Anthropic interfaces, streaming/tools/reasoning/schema/sampling and truthful capabilities.
- [ ] Enforce separate API/tenant payload, queued-prefill/residency and physical active-step admission; bounded overload, cancellation, disconnect, stale epochs and output backpressure.
- [ ] Add real compatible continuous/grouped batches, aged chunked prefill, decode/media fairness and finite joint batch/draft/expert/kernel/placement plans; no per-token catalog/JSON/global-lock work.
- [ ] Finish Qwen basics before optimizations; enumerate the complete pinned llama.cpp/vLLM union, public and internal surfaces. Preserve 105 provisional groups/145 document seeds and CLI/API frontiers as unreviewed inputs, not exhaustive parity.
- [ ] Implement every applicable sampler/logits-processor/RNG/logprob/grammar/tool/stop/reasoning composition and its independently verified outputs.
- [ ] Qualify target-authoritative MTP, draft model, n-gram/suffix, EAGLE, DFlash/DFlash2, DSpark and related registered strategies; DSpark2 exact primary source/version remains unresolved.
- [ ] Test all accepted-prefix lengths, target-distribution correction, target/draft RNG/grammar ownership, full KV/recurrent/index/media rollback, physical branch admission and authentic cancellation/quarantine.
- [ ] Jointly benchmark material combinations, adaptive depth/backoff and all offered/rejected/canceled work: committed SLO-goodput, TTFT/ITL tails, fairness, memory and energy versus equivalent pinned vLLM/V1/Omni.
- [ ] Preserve zero untriaged frontier/disposition requirements before any full-parity claim; synthetic acceptance, fewer launches and competitor measurements do not establish local parity or speed.

## Committed

### Phase 2.1 UX-6 — stable serve CLI (25 Nov)

- [x] Canonical direct CPU GGUF and single approved original GPU snapshot
  aliases share run's exact source/device/config policy and acquisition
  consent. Shared document serve settings; legacy qualified diagnostic/
  GPU GGUF profiles remain explicit. CLI uses ControlPlane load/start/
  exact-generation stop and original runtime retirement. Actual startup
  facts and coarse sampled scheduler/listener counters; no protocol changes.
- [x] Focused real cached Qwen 0.8B CPU release test: two alias completions
  identical, one owner, joined SIGTERM shutdown and closed socket. Cold
  incompatible route refusal, strict combined acquisition/serve lint and
  CUDA/HIP/acquisition/serve compile pass. Full isolated gate 2466 pass/
  0 fail/109 skips; release/installer/scripts pass. No new HTTP/backend
  qualification. Public serving contracts and control transport/auth
  remain later work.

### Phase 2.1 UX-3 — shared control-owned listener (25 Nov)

- [x] Optional ControlPlane slot owns an original ServerHandle; observes
  actual address/metrics, refuses duplicate start and stale generation stop,
  joins exact listener without retiring models/runtime. Real approved Qwen
  0.8B GGUF private-loopback /v1/models two-alias lifecycle and restart pass;
  original task stream/registration, HTTP and inference authentication
  remain authoritative. Independent SDK servers stay independent.
  UX-6 stable serve, later control transport/auth and public serving
  qualification remain open; no new protocol or GPU qualification.

### Phase 2.1 UX-1 — CLI presentation (25 Nov)

- [x] CLI serving startup/shutdown and failure rendering share the terminal
  reporter with run/cache/acquisition. Original serving status/code/detail,
  CLI failure exit status, signal/stop owners, HTTP protocol and authentication
  are unchanged. Full isolated workspace gate (2431 pass/0 fail/106 skips)
  includes existing serving shutdown and feature boundaries.
  UX-3 shared ControlPlane is accepted above; UX-6 stable serve surface
  and broader serving contracts remain in canonical §46A.28 order;
  no new serving capability from presentation alone.


### `63ee14e` — private production CUDA graph lifecycle/cache

- [x] Private production graph lifecycle/cache is now workspace-owned and
  fail-closed without selecting graph serving. Each live native batch workspace is
  `Cold -> Capturing -> Live -> Spent`: one capture attempt only; a failed or
  explicitly retired capture cannot recapture until that workspace is released.
  The cache key binds exact source identity, resident service generation, CUDA
  service/stream and the complete `BatchGraphProfile`; runtime identity drift is
  an error, while a different typed graph profile is a cache miss. The graph's
  existing batch pin and exact retained `(ResidentLease, DeviceAccess)` set remain
  authoritative; no second storage/resource owner was added. Explicit cached-graph
  retirement and incompatible-band replacement both run the existing two trusted
  terminal barriers before graph destruction and release the opaque graph charge
  before workspace reclamation. Unknown retirement still fences/quarantines through
  the existing graph Drop path. The accepted graph policy formula is unchanged:
  original0.8B B2 physical qualification still charges exactly **15,356,224 bytes**.
  The final supervised RTX3060 run preserves full248320/all42 exactness, warm
  `[3725,3725]`, captured body3632, six replay enqueues and14,900 committed
  kernels; it additionally proves exact compatible lookup, incompatible-profile
  miss, second-capture refusal, explicit retirement -> `Spent`/foreign0, fresh
  workspace recapture, and trusted B2->B4 retirement before replacement ->
  `Cold`/foreign0/final reserved0/41MiB. One preceding fresh lifecycle invocation
  disappeared before producing a result; no process/VRAM delta remained and a
  native-gpu control passed, so it remains incomplete rather than relabeled.

### `fix(hip): qualify populated Qwen serving`

- [x] Resolve16K HIP serving deadlines without extending HTTP180s/SDK1200s or
  changing references, caps or owners. R28 accepts all4native scopes; R29 audit2417
  accepts26/26 SDK/HTTP/capacity cases, including both16K HTTP completions and
  actual SDK cancellation/model, exact roots/replay and final readers. Earlier
  failures remain failures. Original R29 monitor missed terminal capture;
  same-invocation exit/physical cleanup is an explicit later observation, with
  the gap retained. Final isolated gate passes2350Rust/0failed;17Python suites.
  [Owning implementation](../model-execution/TODO.md).


### c84b987 — Owned provider stop evidence

- [x] Persist stop intent/results with exact CID/name, redacted argv, UTC and
  monotonic clocks, PID/group, separate private stdout/stderr, return/timeout,
  signals, one reap and EOF. Lost/truncated required evidence blocks the next
  sample after owned cleanup. Recovery cannot erase earlier errors or escalation.
- [x] Distinguish process cleanup, graceful terminal outcome and acceptance.
  Bound same-prefix State observations; malformed/unknown state and positive137
  never establish OOM. Preserve original owner through group signals and one
  reap. No sealed historical capture helper or source21 package is modified.
- [x] Mandatory gate now includes serving clippy/tests,12 new evidence tests and
  existing comparator/lifecycle controls. Full gate0:2049Rust executions,
  91explicit ignores; actual model-free Podman stop0/no escalation, all stream/
  owner/CID closure checks pass. Keep the first failed control and interrupted
  gate separate. Coverage.
- Task checkouts reconciled to the accepted source and removed normally; exact
  candidates/logs and both branch refs retained, no stashes or unique pending
  source. Reconciliation.

### 50abaf7 — Signal-driven serving shutdown

- [x] Install TERM/INT handling before source preparation; close admission,
  interrupt partial/backpressured sockets and original request receipts, join
  all HTTP handlers, then close/join the original CPU/GPU workers. Report handler/
  worker panic, new final GPU retirement failure or retained source readers as
  failure. Original source close/quarantine and physical publication are unchanged.
- [x] Actual model-free process/PID1 TERM+INT controls: exit0, no cleanup signals,
  no surviving members and serving ledger0. Actual approved cached Podman PID1
  stop also exit0/CLOSED/ledger0/no escalation; exact CID independently absent.
  Assembled full `scripts/check.sh` passes; CUDA/HIP/Vulkan feature compilation
  and model-free GPU retirement-failure control pass. No model/device execution
  or performance grant. Coverage.

### `9999adb1a66d` — source-routed continuous BF16 cohort serving

Related source commits: `45570bdb3d7f` (serve-wave-planning), `e6b07efdbb2d` (serve-source-routing), `bc66207ce9c5` (facade-short-hf-loader), `08fc5935e124` (facade-short-hf-cohort-suite).
Aggregate source21 qualification applies to the assembled tree; no per-commit
build or general model/backend grant. Historical failures keep their source scope.

- [x] **MAIN convergence regression checkpoint,22Oct:** preserve orderedSiLU
  and collision-safe PTX dump while integrating the finite HF worker route.
  Fresh MAIN six SiLU modes/two original GGUF B2 workers/canonical HF47 HTTP,
  strictClippy/complete isolated checkgate and vendor104 guards PASS. Actual
  HTTP active2/cancel/survivor/400 and owned retirement are verified; wider
  HF two-model/ragged active8 and matched performance remain open.

#### Current parity and post-parity optimization order — 23 Oct

- [x] Execute quick .8B C1/C4 canonical overview on both CUDA engines; preliminary
  results above. C8 naturally retired before its cancellation lookup, retained
  separately; no2B timing started. All owners/groups/EOF/source closure complete.

#### Resumed correctness-first continuation — 21 Oct

- [x] **Finite original-HF cached worker route:** existing fair worker and CLI
  serve canonical .8B47+7 using real current-root/greedy feedback; generic bounded
  atomic-prefill scheduling charges47 within unchanged64-token waves. Actual
  active2 chat/rawSSE returns three complete8-ID chains, disconnect cancellation
  increments1 and fresh survivor succeeds; max9/wrongprompt400 preserve GPU
  counters and quiescent active0/queued0. Native source/state/normal retirement,
  submitted uncertainty/quarantine, independent reviews and complete isolated
  checkgate pass. HTTP foreground server uses owned SIGTERM/group/EOF cleanup,
  not observed in-process finalzero. Fixed source/prompt/active2 scope now in
  MAIN and ROOT; general two-model/C4/C8 and performance remain open. Broader
  short-HF parity precedes matched benchmarks; cancellation extension stays after.

### fb68d8f — exact state transport and certification

- [x] **Row-native state transport and scatter certification:** exact U32
  vector/KV transport and persistent finite/status certification preserve the
  original all42 preflight, leases, flight, Ready and atomic publication.
  All46 physical model/owner/private-graph/failure cases,10 operator/failure
  cases, complete isolated1251-test gate, strict checks and both approved
  original release parity routes pass. Fresh14 worker/12 HTTP regressions
  pass with exact frozen source/build and original owner/EOF retirement.
  Native-only unprofiled before/after runs complete32 fresh ABBA/BAAB
  processes,144/144 measured offers,1152 output tokens and32 ordinary B1
  warmups. Full IDs/text/usage/finish match exactly; all144 requests have
  individual-ID delivery/1008 actual intervals. C8 uses54 actual B8 cohorts
  per sample,432 independent steps/roots. Median raw token rates improve
  2.2506→2.28945/s(0.8B,+1.73%) and1.6268→1.6543/s(2B,+1.69%);
  C1 is nearly unchanged. Four samples/label, finite bursts, ordinary B1
  warmup and point memory samples establish no confidence-certified win,
  fully warm B8 scope, source-correct goodput or vLLM parity. Source dispatch
  counts292116→239952/284340→232176 are not hardware enumeration. All owned
  process/group/container/EOF retirement and independent ROOT/peer reviews pass.

### fff1842 — stock baseline prerequisites

- [x] **Genuine projector and stock0.8B loader prerequisite:** unchanged
  pinned llama converter produces source-audited BF16 visual projectors for
  both original models; independent complete tensor-byte checks pass.
  Stock pinned vLLM V1/official GGUF plugin loads original0.8B language-only
  weights; all320 backbone roles validate. Plain/observed B1 teacher runs
  preserve both full vocabularies,496640 native BF16 logit words, exactly.
  Native dtype and separate F32 observer representation are explicit; no
  sampler-cast, all42, cross-engine arithmetic or performance qualification.
  Owned process/group/container/EOF cleanup passes. Both projectors are
  genuine source conversions, not dummy parameters or a modified engine.

### 9a19901 — batched graph and baseline qualification

- [x] **Private full-model CUDA graph qualification:** both original models
  B2/4/8, StateOnly teacher prefixes (including genuine EOS input), future full
  logits and outside-body greedy sampling match independent CPU/B1/native
  full248320/all42 state. All six normal cases and18 fresh unknown-completion
  cases pass actual replay, stale/Ready/pressure refusal and reclamation or
  original quarantine. RoPE launch reduction additionally passes both models'
  170 ordered-angle cases and full native B2/4/8 normal/greedy comparisons.
  Physical32-case result87b85eff. Composed strict CUDA/Vulkan, complete
  isolated1243-test phase, both original release routes,14 worker and12 HTTP
  CUDA regressions pass; composed result e76248e8.

- [x] **Genuine GGUF baseline build prerequisite:** official out-of-tree
  vLLM GGUF plugin e2b8ad532b8b builds unchanged against pinned vLLM0.31/
  Torch2.13/CUDA12.9; wheel af9de736,129 RECORD entries/111 packaged Python
  files and full151-source pre/post pins validate. Owned container/group/EOF
  cleanup passes; no GPU/model ran. Stock text loading with a genuine visual
  projector and language-model-only mode, activation/cast/reduction profiles,
  numerical quality and matched benchmarks remain unqualified. Lack of an
  in-tree GGUF registry does not exclude this official plugin route.

- [x] **Bounded native-prefix warm comparison:** 0.8B strict B8 ABBA and 2B
  natural concurrency8 BAAB, each plus B1,16 serialized samples/72offers/
  72completed/576 output tokens. Exact full IDs/text/usage/finish and cleanup.
  Mean concurrent waves32.301205→18.33249s(0.8B,-43.2%) and
  42.19858→28.457115s(2B,-32.6%); actual forward dispatches
  1722640→302160(-82.5%) and1668496→294384(-82.4%). B1 means
  4.43954→4.42998s/5.60352→5.652635s, unchanged kernel counts.
  Both concurrent workloads still miss10s TTFT; no consistent ITL-tail gain,
  source-correct goodput unset. Original strict2B B8 attempt failed its
  declared scheduling guard and remains unqualified; fresh naturalC8 protocol
  permits actual2/4/8+B1, and both successful new samples happened to use54B8.
  Conditions/limits and observed memory in FUSED_GPU_BATCHING; no vLLM claim.

### f788929 — native teacher-prefix serving

- [x] **Bounded same-engine cold/warm comparison complete:** both original
  models, warm ABBA(0.8B)/BAAB(2B) and cold-process ABBA,16 serialized samples,
  128offers/128completed/1024 output tokens. Every sample has8 genuineB8
  cohorts, zero refusals/errors/discards, and identical complete IDs/text/usage/
  finish reason across providers. Provider4GiB/device5GiB on BOTH builds.
  Mean warm waves improve15.3%/6.1%; cold13.8%/6.0%. Actual dispatches fall
  16.9%/14.4%. No consistent ITL-tail improvement or10s TTFT-SLO success;
  observed goodput0/source-correct goodput unset. Exact values/conditions in
  FUSED_GPU_BATCHING. Initial equal4GiB attempt remains rejected evidence.
- [x] **P12 native teacher-prefix serving:** same full-model BatchProgram
  consumes one teacher token/request/wave with StateOnly readout, independent
  causal positions and original all-root publisher. Both modelsB2/4/8 pass
  per-teacher CPU+B1/all42/future full248320, trained EOS input, abandoned Ready,
  stale/invalid rows and final0. Two physical pressure and two fresh unknown
  completion cases pass; unknown retains1817/1712 handles and4GiB envelope,
  unchanged roots/all42 and fence, without recovery. All14 worker and12 HTTP
  scopes pass, including ragged prefix/tail partition, no prefix-generated IDs,
  Ready cancellation/regroup, output-credit fairness and final active/queue0.
  Three readout-phase health counters close against actual native bands and
  four sampler stages/greedy cohort. Complete isolated check1241 Rust/0failed/
 70ignored/248suites+18Python/release/installer/Nix; strict CUDA/Vulkan/qualifier
  and both approved original cache-direct/HF-import release parity routes pass.
  Two test-fixture/assertion failures and one missing-CUDA_PATH prelaunch
  refusal are preserved; no production arithmetic or acceptance limit changed.


### d0a6084 — source activation preparation and physical acceptance

- [x] **P12 block-local CUDA activation preparation:** one native32-lane
  workgroup preserves source ordered finite/max/scale/rounding/code/sum
  arithmetic for Q4_0/Q4_1/Q8_0/Q6_K projections. Original seven buffers,
  composite/status stages, leases, queue and completion remain authoritative.
  Cold checked bindings remove repeated warm handle/capability/alias setup;
  only explicit non-CUDA selection retains the old lowering. CUDA errors
  propagate. Projections use5 launches versus10 (Q6_K:8), before certification.
- [x] **Current physical acceptance:**272 preparation cases/all7 buffers/
  1,483,608 words exact; both original0.8B/2B complete B2/4/8 unequal-frontier
  graphs pass all248320 logits/all42 domains versus independent CPU AND B1,
  feedback/refill/abandoned Ready/stale packets/final reservation0. Two real
  pressure cases pass prelaunch refusal, B4 progress and B8 after release.
  Eight serving-worker and six HTTP cases pass both models B2/4/8, EOS,
  credit/cancel/disconnect/replay and final0. Four fresh graph/sampler unknown
  completions retain1824/1719 and1828/1723 handles, unchanged roots/all42,
 4GiB envelope, fence and no recovery. Complete isolated gate1218 Rust/0fail/
 69ignored/248 suites plus18Python/release/installer/Nix; strict CUDA/Vulkan/
  qualifier and both original release parity routes pass.
- [x] **Bounded same-engine0.8B warmed ABBA:**32 offers/32 completed/256
  output tokens across four fresh serialized samples; all eight requests/sample
  have individual token events and identical IDs/text/usage/finish reason.
  Each sample executes eight B8 cohorts with zero refusal/failure/cancel.
  Mean wave39.14206→33.163665s (-15.3%); dispatches2,072,360→1,722,640
  (-16.9%). Mean of per-sample client ITL p95:0.3911→0.3933s, no tail
  improvement established. Both miss the declared10s TTFT SLO; goodput0.
  Source-correct SLO goodput remains unset. This is ZG versus ZG only.
### 1b599df — serialized owned benchmark provider lifecycle

- [x] **P22 serialized owned provider lifecycle:** optional strict
  `--provider-lifecycle` specs launch one fresh foreground process per
  actual ABBA/BAAB sample, with literal loopback vacancy checks, bounded
  startup/shutdown/logs, exact model readiness, finite optional warmup and
  isolated receipts. `cold_process` means a fresh process without warmup;
  it does not mean evicted OS/storage/device caches. `warmed_process`
  adds one ordinary unmeasured request. Direct argv only; Podman binds
  the first fresh CID and stops only that owned container via the same
  local runtime prefix. Cleanup precedes evidence writes; receipt/log/
  reader/process/endpoint failures block the next sample. Environment
  names and redacted argv are recorded; private raw logs are capped/drained.
  Existing already-running-endpoint comparison remains available.
- [x] **Lifecycle acceptance:** source-free actual localhost ABBA/BAAB,
  exclusive ownership, warm/cold, startup/model/hang/warmup/measurement
  failures, TERM-to-KILL, CID overwrite/missing/stop failure and five
  persistent receipt-I/O failures passed; all owned readers/processes/
  endpoints retired before any next sample. Root independently passed
  comparator/benchmark self-tests and pycompile. Two real GPU providers
  also passed SERIAL startup, one warmup request and complete shutdown:
  accepted B7 ZG mixed-GGUF CUDA, then immutable pinned vLLM0.31.0-cu129
  original-HF BF16. Each returned GPU telemetry to baseline41MiB; live
  samples1059/4203MiB are advisory samples, not peaks or reservations.
  Those sources/quant/math are explicitly unmatched: this is lifecycle
  qualification ONLY, not a benchmark or numerical/performance parity.
  Complete isolated gate1202 Rust/0failed/69ignored/248 suites plus18Python,
  strict CUDA/Vulkan/qualifier Clippy and both approved release parity
  routes passed. Matched source-correct SLO goodput remains unset.


### dbb5f4f — unequal native decode frontiers and bounded token observations

- [x] **P12-B7 unequal-frontier native CUDA decode:** per-row positions,
  RoPE/KV append/masking and packed attention softmax lower
  the same full-model ForwardPlan. B2/B4/B8 retain independent original
  authorities, roots, packets, leases and all-row Ready publication.
  Native decode accepts unequal frontiers; prompt-tail grouping stays
  strict and other prefill uses B1. Actual original0.8B/2B engine tests
  passed all248320 logits/all42 domains versus independent CPU AND B1
  at mixed positions255/256/0/31/32/1/510/511, host/device feedback,
  B1 handoffs/refill, abandoned Ready, stale/foreign packets, capacity512,
  inactive AV poison/signed-zero/subnormal cases and final reservation0.
  This engine run preceded metadata/observability-only composition;
  fresh composed worker/HTTP/pressure/abort suites passed separately.
- [x] **P22 bounded publication/delivery observations:** opt-in
  `--enable-per-request-metrics` plus `return_token_ids:true`; bounded
  ASCII request IDs, at most512 non-EOS token records, exact-once IDs
  independent of UTF-8 emission. Observe commit immediately after original
  root publication, queue receive and socket-write completion separately;
  retain the original observation through output-credit parking. GPU
  producer frontiers and existing successful native publication sequence/
  width bind each row without creating a new authority. Standard vLLM
  scheduler-origin TTFT/queue fields stay unset; parsed-request timing is
  named separately. Tools validate IDs, clocks, usage, frontiers/cohorts;
  true client ITL requires one actual token per arrival. Grouped arrivals
  remain unresolved; source-correct SLO goodput remains unset.
- [x] **B7 physical composed serving acceptance:** eight release worker
  cases and six release HTTP cases passed both models at B2/4/8, exact
  CPU/B1 packets/all42 internally, CPU IDs/text/usage on wire, trained EOS,
  limits, Ready cancellation/regroup, capacity1 parking, disconnect,
  survivors/replay and final reservation/active/queued/failed0. New
  unequal-arrival cases offer the short request AFTER the long request's
  first actual ID and observe a shared native B2 publication at distinct
  frontiers. Each model completed5 native cohorts/10rows/935 native
  projection-dot launches/20 native sampler stages in that wire case.
  Two fresh real-pressure cases passed prelaunch B8 refusal without
  mutation, B4 progress under pressure and B8 after release. Four fresh
  full-graph/sampler unknown-completion probes retained all original
  handles in quarantine (1824/1719;1828/1723), unchanged prior roots/all42,
  refused old Ready/further work/other model, retained4GiB and no recovery.
  First abort invocation used a nonexistent package-named binary and
  launched no GPU work; only the four probes were rerun successfully.
- [x] **B7 verification:** complete isolated check1164 Rust passed/
  0failed/69ignored over245 suite runs plus18Python/release/installer/Nix;
  strict CPU/CUDA/Vulkan/qualification Clippy passed. Fresh source-free
  release CLI35/GPU58 passed; focused composed CPU/Core/Model/facade/
  serving185 passed/2ignored. Both approved original cache-direct and
  HF-import release routes passed full248320/all42 for both models.
  Independent launch audit reconciles the unequal0.8B HTTP counter exactly:
  26 B1 state-only graphs*5489 +6 B1 greedy*5510 +5 nativeB2*5728
  =204414 actual completed launches. This includes private certification
  and state/packet copies, excludes cold initialization/maintenance, and
  counts a native cohort once. No throughput claim follows from it.


### d9fa829 — native CUDA B4/B8 HTTP bands

- [x] **P12-B6 native CUDA B4/B8 serving:** finite unpadded8/4/2/1
  bands, live EOS/length/cancel/credit survivor regrouping and actual
  per-band Ready counters. Explicit `--batch-rows 2|4|8` (default2),
  CUDA/active>=rows checks before model/device; default interleaved.
  Checked physical capacity refusal is typed None BEFORE allocations or
  submission; only that result permits smaller bands. Errors/unknown
  completion retain original fence/quarantine. Real scratch leases prove
  cold/warm B8 refusal leaves queue/uploads/allocations/roots/all42/idle
  workspace untouched; B4 progresses under pressure, B8 after release,
  full248320 CPU/B1 logits/packets/all42 exact, final reservation0.
  Six same-process release worker cases (.8B/2B × B2/4/8) passed exact
  CPU/B1 state, unequal output limits, handoffs, actual after-Ready cancel,
  bounded capacity1 parking/resume and trained EOS. Initial combined run
  failed CUDA OOM after old-stream cached pools accumulated; cold-load
  cleanup now runs under original initialization/queue guards. Repetition
  physically released completed idle pools (2.51GB/2.13GB→0), all6 pass.
  Four release HTTP tests prove both models B2/B4/B8 JSON/SSE CPU text/
  usage, trained EOS, one terminal, active disconnect, survivors and replay;
  final active/queued/failed0. Source-free CLI26 and GPU53 passed;
  strict CPU/CUDA/Vulkan/qualifier Clippy and full isolated check1148/0/
  69ignored/245suites+18Python passed, both approved release cache-direct/
  HF-import routes exact full logits/all42. Initial direct HTTP invocation
  found a non-serve binary left by the full check; rebuilt explicit CUDA
  server and repeated all4 successfully. Same-frontier serving only;
  unequal-frontier/paged state, full-prefix native prefill and matched
  vLLM source-correct performance remain open.

### 6bfcdbe — bounded offered-work benchmark accounting

- [x] **P22 bounded benchmark accounting:** `serve-bench.py` records
  offered arrivals, client backlog/overload, HTTP acceptance/rejections,
  completion/timeout/cancel outcomes, tenant tails and the full offered
  denominator. SSE chunk intervals remain explicit proxies. Optional
  provider token traces require exact run/wire/provider IDs, complete
  ordered token IDs/timestamps and terminal usage equality; malformed or
  incomplete traces cannot qualify true ITL/SLO goodput. Raw loopback JSON
  snapshots remain provider claims. `serve-compare.py` executes actual
  alternating ABBA/BAAB orders, compares per-request text/usage/token IDs
  and keeps caller source declarations separate from independent parity.
  Both isolated self-tests/pycompile and full check.sh passed. Actual pinned
  vLLM0.31.0-cu129 V1 CUDA startup and four-request warm smoke passed with
  original HF0.8B; this is NOT a matched comparison. Common trained source
  arithmetic, tokenizer/generation policy, caps and real token observations
  remain open; qualified source-correct SLO goodput stays unset.

### 2677d8b — native CUDA B2 HTTP waves

- [x] **P12-B5 native CUDA B2 HTTP:** existing worker dispatches the
  original full-model native plan for compatible prompt tails/decode;
  unmatched rows and other prefill remain B1. Independent session/root/
  packet/physical leases and the all-row publisher remain authoritative.
  Present each committed packet once, retire EOS/length/canceled rows,
  park bounded output before feedback and retry discarded Ready cohorts
  without repeated text/usage/terminal. Actual original0.8B/2B RELEASE
  worker tests passed CPU/B1 compact ID/max/all42 parity, B1↔native
  handoffs, output limits2/5, real trained EOS, pre-feedback/after-Ready
  cancellation, capacity1 credit parking/recovery and final reservation0.
  Actual owned CUDA HTTP alias JSON/SSE tests passed exact CPU text/usage,
  trained EOS stop/EOS-excluded usage, one terminal/usage/Done, active
  disconnect, survivor/replay and final active/queued/failed0. Native
  completed cohorts9/8, projection launches1683/1496, sampler stages36/32;
  one Ready cohort discarded per model. Opt-in fused is CUDA B2 only,
  active≥2; default interleaved and non-CUDA refusals are preserved.
  Strict CPU/CUDA/Vulkan serve+cache and CUDA-qualifier Clippy passed;
  CPU serve14 and GPU53 tests passed (GPU3 intentionally ignored).
  Full isolated check.sh exit0:1147 Rust passed/0 failed/69 ignored over
  245 suite runs,18 Python, release/installer/Nix formatting. Both approved
  cache-direct/HF-import RELEASE routes passed full logits/all42 for both
  models. Real release serve-bench smoke4/4 requests16 usage tokens each,
  eight actual native cohorts each; comparator self-test passed. No true
  ITL/SLO/fairness or matched vLLM performance acceptance; vLLM absent.

### 54fd580 — native CUDA greedy/device feedback

- [x] **P12-B4 native CUDA greedy/device feedback:** same full-model
  batch ForwardPlan plus four native vocabulary-reduction stages,
  independent12-byte packets and one native device-input gather.
  Original packet producer is stamped only at all-row checked commit;
  exact source/session/epoch/root/frontier and physical lease bind
  consumption. Mutable public host ID/max fields are non-authoritative.
  Approved trained0.8B/2B mixed GGUF B2/B4/B8 RELEASE passed six
  source steps/four GPU-feedback steps per row: CPU/B1/native
  full-logit shadow all248320 logits and all42 domains exact;
  compact greedy ID/max bits/all42 domains exact,28/52/100 bytes
  per cohort, no warm value uploads and final reservation0.
  Swapped/duplicate/identical-ID foreign packets, isolated same-epoch
  old roots, restored epochs, invalid IDs, abandoned/reordered/stale
  proposals refuse without publication. Full-vocabulary synthetic
  ties/zero/subnormal/dense-bit/EOS winners and nonfinite/sticky-status
  refusals passed; trained EOS input is exact, HTTP EOS stop unqualified.
  Actual feedback launch/drop retained1827/1722 original handles,
  including input/output packets, all42 CPU prior domains and future
  KV zero tails exact; old Ready/further work/new service refused,
  original4GiB envelope fenced/charged after owners drop, no recovery.
  Both models repeated B2 imported-prefix valid1/256/257/512 boundary,
  exhausted-capacity and reclamation proof; no GPU full-prefix claim.
  GPU53 passed/0 failed/3 ignored; strict default/CUDA qualification
  and CUDA/Vulkan serve+cache Clippy passed. Final isolated check.sh
  exit0:1147 Rust passed/0 failed/69 ignored over241 suites,18 Python,
  releaseCLI/installer/Nixfmt; both approved cache-direct/HF-import
  release routes and benchmark/comparator self-tests passed.
  HTTP remains interleaved; no fused-serving/vLLM throughput acceptance.

### 35ebc5b — native full-model CUDA ForwardPlan

- [x] **P12-B3 native full-model CUDA ForwardPlan:** one traversal
  over B2/B4/B8, original shared source handles, admitted packed
  scratch and private per-session prior/candidate leases. One actual
  completion yields exact source/session/epoch/root-bound Ready
  receipts; existing all-row checked commit remains the publisher.
  Original trained0.8B/2B mixed GGUF profiles passed all248320
  logits/all42 domains per row versus independent CPU AND B1 over
  two decode steps, distinct histories, invalid/duplicate/foreign/
  reordered rows, abandoned proposals, stale epoch/root rollback,
  sibling continuation, no warm source/value uploads and reclamation.
  Release B2 for BOTH models passed CPU-hydrated prefix0/255/511
  decode at valid1/256/257/512, exact canonical KV tails, attention
  padding dispatch delta and capacity refusal. Actual launch/drop
  tests retained all1820/1715 original handles and the full4GiB
  envelope behind the existing quarantine; prior roots/all42 domains
  stayed exact, old Ready/next step/other model refused, no recovery.
  Source-free GPU51 passed/0 failed/3 ignored; strict default GPU,
  CUDA qualification and CUDA/Vulkan serve+cache Clippy passed.
  Frozen isolated check.sh exit0:1147 Rust passed/0 failed/69 ignored
  over241 suites,18 Python fixtures, release CLI/installer/Nixfmt.
  Both approved original0.8B/2B direct-cache and HF-import RELEASE
  full-logit/all42-domain parity routes passed; benchmark and
  comparator self-tests passed after private loopback setup.
  Same-frontier full-logit CUDA diagnostic only: no native row
  greedy/EOS, unequal-prefix/paged state, GPU full-prefix prefill,
  fused HTTP or vLLM performance acceptance. HTTP stays interleaved.

### 3ec431a — native quantized batch projections

- [x] **P12-B2 source-quantized native batch projections:**
  Q4_0, Q4_1, Q8_0 and Q6_K source-exact B2/B4/B8
  kernels call the SAME original B1 integer/F32 dot.
  Each physical RTX3060 CUDA test used one native B>1
  launch, bit-identical 128/256/512 F32 outputs versus
  original per-row B1 GPU reference. Original row-specific
  dynamic Q8 scales/offsets/codes, strict typed layouts,
  source-free bad-geometry tests and no CPU dot fallback.
  THESE are synthetic-source operator fixtures only.
  Strict source-free and CUDA Clippy pass; frozen gate
  1,147 Rust tests/0 failed over 241 suites, and trained
  Qwen3.5 .8B/2B direct-cache/HF-import source parity
  remains full logits/all42 StateRoot domains. Next:
  batch-native resident ForwardPlan integration and
  physical B2/4/8 trained full-model root parity.

### 08be9c6 — resident cohort publication

- [x] **P12-A1 source-bound resident GPU cohort:** ONE
  `ResidentModel` original-source method stages B2/B4/B8
  independently completed `ResidentReady` per-session proposals,
  checks all session RootAttempts and complete domain leases
  before one `ResidentService` fence-guarded publication.
  Invalid/foreign/abandoned/stale rows refuse all sibling
  root changes. Physical RTX3060 CUDA Qwen3.5 B2/B4/B8 matched
  full 248320 logits and all42 state domains per row against
  independent CPU; B4/B8 memory returned to baseline.
  Live OpenAI CUDA B2 short-prompt wave published two roots
  in one checked cohort; four separate GPU full-model forwards,
  six completed quanta, zero errors and `gpu_fused_batches=0`.
  Real batch-native ForwardPlan kernels remain pending.
  Acceptance: physical CUDA resident B2/B4/B8 exact all248320
  logits and all42 StateRoot domains per independent session,
  source/session state snapshots, abandoned/invalid/stale rollback,
  one source upload and lease reclamation; HTTP B2 cohort
  publication and active disconnect/survivor/replay pass.
  Strict CPU/CUDA/Vulkan feature Clippy -D warnings; full
  scripts/check.sh exit0: 1,147 Rust tests/0 failures/69 ignored
  across 241 suites plus release CLI/installer, 18 Python fixtures
  and Nix formatting; trained Qwen3.5 .8B/2B direct-cache and
  HF-import release parity both pass exact full logits/all42 domains.
  Separate offline Nix source-integrity cannot evaluate because
  pinned nixpkgs is absent from the local store and connector SSL
  CA path is unavailable. NOT a source-test failure.

### 9790fa2 — resident GPU OpenAI serving

- [x] **P09 GPU OpenAI serving / finite batch-wave scheduler:**
  opt-in CUDA/Vulkan/HIP resident model with independent physical
  StateRoots and device-side greedy feedback; no CPU inference
  fallback. Shared HTTP/SSE/greedy output, cancellation and
  original completion/quarantine. Bounded one-GPU worker,
  active/queue limits, source/phase/frontier wave grouping.
  0 fused GPU batches (no batch-native kernels yet).
  Typed GPU profile, CUDA/CCCL fail-fast preflight and reusable
  Nix split-SDK include view. Metrics separate GPU forward API calls (not kernel launches),
  successful quanta, attempted quanta, active/queued/canceled/error.
  Physical RTX3060 Qwen0.8B CUDA: 2 concurrent aliases under one
  model source, six completed quanta/four GPU forward API calls/zero errors,
  canceled active session retired safely and new session succeeded.
  SSE benchmark: 2/2 requests, 4 tokens, 20.02s debug wall,
  0.1998 token/s (NOT optimized/vLLM parity).

### 35b9093 — initial OpenAI CPU router and verified SSD extents

- [x] **First real OpenAI HTTP/CPU router milestone:** opt-in
  `zerogravity serve` with original Qwen3.5 GGUF sources, alias reuse
  (one resident immutable model/lease per canonical path), model list,
  health/admission, text and chat greedy completions and SSE usage.
  Bounded N CPU workers, independent source-specific StateRoots,
  round-robin 32-token prefill / single decode quanta, finite
  queue+active+connection admission, shared host ResourceArbiter
  weight and per-session state envelopes. Loopback-only with
  optional Bearer token; strict duplicate-key JSON and request
  framing, unsupported sampling/roles/media refusal, cancellation,
  SSE keepalives/deadlines and bounded backpressure. Real live 0.8B
  multi-alias + two concurrent text requests, authenticated health,
  host weight/state reservation release, unsupported parameter
  refusal and streaming passed. Full frozen gate:
  1,147 Rust passed/0 failed/69 ignored across 241 suites,
  18 Python fixtures and release/installer/CPU comparator checks.
  Both approved Qwen3.5 0.8B/2B original direct and HF-import
  release parity tests passed exact full 248320 logits/all 42
  StateRoot domains/two tokens/source lease and mutation.
  Strict default and composed serve+SSD/cache Clippy pass.
  The separate Nix offline source-integrity derivation is blocked:
  pinned nixpkgs 7a0f122 is not cached and the connector cannot
  fetch it with available SSL trust anchors; Nix formatting DID pass.
  Not fused GPU continuous batching or full OpenAI/vLLM API parity.

### 6d74f76

- [x] Shared immutable source, independent StateRoots, bounded fair physical workers and actual completion exist as foundations; production serving/parity/speculation is not complete.
