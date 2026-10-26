# Halogen: composable execution-fabric research

Author: /root/halogen_optimization_research
Date: 2026-10-21
Parent: cpu-gpu-optimization; research only, no implementation promotion.

## Result and limits

Halogen is useful primary prior art for shape specialization, hybrid-state reuse,
sparse selection, draft policy, memory pressure and fair prefill scheduling. Its
core is still closed source at the inspected current revision. The public
repository cannot establish its kernel implementation, graph capture, allocator,
complete rollback or cancellation lifetime correctness. Use its documented
mechanisms as hypotheses for capability stages in Zerogravity's existing engine.
Do not import another inference runtime or duplicate mathematical/state authority.

The independent [65-mechanism inventory](halogen-inventory.md) maps every substantive
public mechanism found to concrete local code/tests/plans and identifies
uninspectable or absent implementations. All speed/quality numbers below are
publisher observations. This research did not execute Halogen, compile/build,
launch GPUs, inspect binaries, fetch model/SDK payloads, change privileges,
recover hardware, edit shared plans, or edit the root README.

## Current source identity, access and license

- Repository main and latest observed tag `v0.18.13` resolve to
  `3f2adcfca6be623c8ec4f69331cbf97beff55251`, committer timestamp
  `2026-10-21T02:51:01Z`, title `0.18.13`. The GitHub releases API returned
  an empty list: this is a tag/image version, not a GitHub release object.
  The image named by current docs is
  `ghcr.io/peonist-ai/halogen-flash-server:0.18.13`; no image digest or binary
  source correspondence was independently established. [Commit](https://github.com/peonist-ai/halogen-flash-server/commit/3f2adcfca6be623c8ec4f69331cbf97beff55251),
  [tags](https://api.github.com/repos/peonist-ai/halogen-flash-server/tags),
  [releases](https://api.github.com/repos/peonist-ai/halogen-flash-server/releases).
- Complete tree enumeration found documentation, deploy/host scripts and public
  benchmark scripts, with no mathematical engine, NPU engine or serving frontend
  source. The historical first-release note states the frontend ships readably
  inside the image; no container was fetched to inspect it. The current model
  card explicitly still labels the engine closed source.
  [Pinned tree](https://github.com/peonist-ai/halogen-flash-server/tree/3f2adcfca6be623c8ec4f69331cbf97beff55251), [Pinned model card](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next/blob/ff0830694ec8461edd827d7300134e346bf03eb1/README.md).
- EULA v0.1, effective 2026-08-25: royalty-free commercial/production use,
  unmodified redistribution with notices, and benchmark publication permitted;
  modified-image redistribution restricted. This is a custom EULA, not an OSS
  license grant. No reverse engineering was performed. Weights and third-party
  components have separate licenses. [EULA](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/LICENSE.md), [notices](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/THIRD-PARTY-NOTICES.md).
- Published weights repository is pinned separately at
  `ff0830694ec8461edd827d7300134e346bf03eb1`, API lastModified
  `2026-10-21T03:10:08Z`. Its card declares Apache-2.0 and derivation from
  `Qwen/Qwen3.8-Flash-Next`; the engine EULA does not license those weights.
  `.hgn` is native to Halogen, not a portable runtime format. Only metadata and
  README text were fetched; no payload hashes or model bytes were verified.
  [Pinned model card](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next/blob/ff0830694ec8461edd827d7300134e346bf03eb1/README.md).

GitHub HTML initially served README text mentioning0.18.12; the API/tree/pinned
raw text established0.18.13. The exact pin above governs this report. No claim
that this remains the newest version after inspection.

## Architecture and semantic boundaries

**Published architecture:** specialized ROCm/HIP gfx1151 Flash-Next GPU engine,
HTTP frontend, optional separately compiled NPU engine/programs behind the same
port. A shared HTTP port and shared memory bus do not establish one mathematical
engine or one cross-device allocation authority. The NPU scope now includes
small decision, embedding, rerank, moderation, text generation, image and ASR
models; Flash GPU specialization still covers one model family. [README](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/README.md), [NPU docs](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/NPU.md),
[notices](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/THIRD-PARTY-NOTICES.md). The public script invokes an engine binary and Python frontend;
it contains useful observable startup/host checks, not their internal math.
[public entrypoint](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/deploy/entrypoint.sh).

**Source semantics to preserve:** trained GDN recurrence/convolution; sparse
attention indexer, selection ties, block order and budget; MoE route/order;
residuals; native learned lookup bank and trained MTP head. Runtime prompt
lookup is a separate proposal algorithm, not learned lookup weights. Native v2
uses a documented mixed-precision checkpoint; its quantized values are a
particular model source and are not equivalent to original BF16 semantics.
[changelog](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/CHANGELOG.md), [quantization](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/QUANT.md), [Pinned model card](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next/blob/ff0830694ec8461edd827d7300134e346bf03eb1/README.md).

**Separate numerical profiles:** cache2 resume seams; KV8; alternate GEMM
reduction; different prefill chunking/kernels; optional unpacked trunk; larger
attention budget; YaRN; approximate composable-context. These need distinct
source/profile/cache compatibility and quality gates. Explicitly retain
cache1's cache-hit-versus-cold token identity boundary. A fixed seed promises
less than cross-batching/cache/release token equality. [flags](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/FLAGS.md), [configuration](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/configuration.md), [sampling](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/sampling.md).

**Name collision:** Halogen's `HALOGEN_COMPOSABLE_CONTEXT` is an opt-in preview
that reuses work for long messages moved to different offsets after compaction.
It explicitly admits numerical differences. The user's ONE shared execution
fabric means shared capability compilation, immutable source residency,
physical admission, context grants and independent complete state roots across
pipelines/models/optimizations. Those are different concepts. No approximate
message reuse is implied by the user's requirement. [configuration](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/configuration.md).

## Benchmark evidence and exact conditions

All external values are reported observations, not our physical verification.
The 0.18.13 measurements are A/B within one session per machine, one request
at a time, greedy/thinking off; new concurrency policy is excluded. v2's
10-shape `bench` mean is66.04tok/s; cache-off synthetic `sweep` pp8192/32768
is1879/1821tok/s; cold cache-on is1784/1850tok/s. v2 machine is Max+395/128GB,
approximately85W; ht43 comparisons use another machine around120W, so avoid
cross-checkpoint causal speed claims. [measurements](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/performance.md).

The older0.18.0 headline94tok/s is repetitive-output best case, not general
text. Its agent-turn median4.2s/TTFT1.6s covers115 turns from three sessions,
30k–96k context, ~200 answer tokens, growing shared prefixes and cache2;
first34,174-token turn is21.2s. It is neither storage-cold startup nor exact
cache1 output. Full client-wall timing and cached-prefix state matter. [README](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/README.md).

For0.18.11 joint speculation, the public comparison switches `SPEC_GROUP=0`
versus default within that image, cache off, short independent prompts,
384tokens/request, mean of two server starts on two further machines per
checkpoint. v2 totals move74.6→91.6tok/s(C3) and85.3→103.0(C4). This isolates
policy more closely than comparing release headlines; it still supplies no
SLO-goodput, offered-load fairness proof or general-vLLM superiority. [measurements](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/performance.md).

Public benchmark code is inspectable: `bench-serving.py` records client wall,
request-text output hashes and engine-log timing/commit rates; it warns about
foreign traffic and wall-only rates. Greedy identity hashes reasoning+content,
not full logits/StateRoot. Sampled identity is skipped; referenced internal
sampler gates are not public in this tree. `halogen-bench.py` calibrates a
repeated synthetic filler through HTTP and records cache-hit caveats; pp is
prompt_tokens/client_wall and tg uses response timing when available. These
instruments are not interchangeable with raw kernel time, per-token ITL,
whole-process startup or independently qualified quality. [bench-serving.py](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/tools/bench-serving.py), [halogen-bench.py](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/tools/halogen-bench.py).

**Quality bounds:** six32-step teacher-forced BF16-golden prompts yield182/192
matching token choices; the publisher separately compares engine math on the
same dequantized weights. Retrieval148/150 covers1k–32k, not general reasoning.
The complete original BF16 model was not run at scale on the128GB host.
Perplexity uses paired arms; checkpoint tooling's top128-support KL is a lower
bound, not full-vocabulary KL. These do not match Zerogravity's raw full-logit
and complete-state gates. [quality bounds](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/quality.md), [checkpoint tools](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/checkpoint-tools.md).

**Cold start:** GGUF repack observations are18s cold/9s page-cache warm; optional
repack cache16s cold/1.4s warm, at ~70GiB extra artifact cost. Sizes/mtimes
validate that external cache, not cryptographic immutable-source equivalence.
The native0.14/w4b~30s/118GiB startup is historical, not current v2 startup.
[GGUF docs](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/gguf.md). Native current startup to ready, image pull, model download and first
request must remain separate benchmark columns.

## Negative experiments, caveats and unresolved proof

- Lower-bit ht43 saves roughly8GiB but can lose6–10% prefill and a few percent
  speculative decode. w4b's speed sidecar gains~2% serial decode while losing
  measured calibration quality. No automatic adoption. [precision options](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/precision.md), [quantization](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/QUANT.md).
- More KV residency can starve file-cache lookup rows and induce host
  compaction. Bigger prefill scratch reduces usable residency; automatic pool
  fit does not establish safe physical headroom. [configuration](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/configuration.md), [host setup](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/host-setup.md).
- Unpinning w4b drops reported decode37→6tok/s and adds expert copying;
  v2/ht43 reject it. It is a memory trade, not evidence for an efficient general
  expert pool or SSD scheduler. [host setup](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/host-setup.md).
- Native/GGUF quality and throughput trade differently. Some GGUF K-quant dense
  tensors require a BF16 expansion. The pinned docs retain a known concurrent
  `UD-Q4_K_XL` empty-reply issue, despite general concurrency identity claims;
  do not close that shape by slogan. [GGUF docs](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/gguf.md).
- GEMM autotuning without a persisted plan can change answers between starts.
  Increasing draft depth can lose time; low acceptance needs measured backoff.
  Gather concurrency saturates:128threads was no better than64; repack16workers
  was no faster than8 on the reference NVMe. [flags](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/FLAGS.md).
- KV8 trades memory for a slight decode slowdown and numerical drift; attention
  budget changes retrieval/perplexity; YaRN changes all positions and draft
  acceptance. No exact-profile or universal speed claim follows. [configuration](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/configuration.md), [flags](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/FLAGS.md).
-0.18.13 smooths streaming pauses through splice/nested admission while moving
  long-prompt TTFT later; task throughput and interactive tails must be optimized
  jointly. Default chunk changes can create load-dependent arithmetic.
  [changelog](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/CHANGELOG.md), [configuration](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/configuration.md).
- Cache-on sweep previously reused calibration prefix; corrected0.18.10 rows
  must be compared within their table. Cache2 may answer differently from a cold
  run. Disk byte-exact restore means exact saved-state restore, not cache1 cold
  identity. [measurements](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/performance.md), [configuration](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/configuration.md).
- NPU padded batches can lose on small groups; alternating large models can
  pay reload cost. GPU/NPU share power/bandwidth and require IOMMU/fabric-clock
  qualification. Their combined correctness/performance is not additive.
  [NPU docs](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/NPU.md).
- Public docs conflict at broad summaries: FLAGS' introductory BITWISE
  exceptions omit later numeric KV8/QSA-force/unpacked-trunk cases; API's broad
  cache determinism sentence and composable-context's generic prompt-cache
  wording conflict with explicit cache1/cache2 caveats. Use specific contracts
  and oracle proof, never infer exactness from blanket prose. [flags](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/FLAGS.md), [API docs](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/api.md),
  [configuration](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/configuration.md), [quality bounds](https://github.com/peonist-ai/halogen-flash-server/blob/3f2adcfca6be623c8ec4f69331cbf97beff55251/docs/quality.md).

**Unknown:** complete kernel source/ISA/layout, fusion boundaries, graph use,
full-state snapshot/restore internals, expert pool LRU/generations, actual
allocation/pointer lifetime, asynchronous IO implementation, exact speculative
rejection correction and rollback kernels, release test datasets/results,
all-source profiler accounting, full trained BF16 quality, model-wide
cross-engine matched benchmarks and multi-model allocator authority. Absence
of public evidence is not evidence the implementation lacks a feature.

## Current Zerogravity mapping (Z1–Z10)

Read current MAIN plans before accepting local scope. ISO HEAD is
`fb68d8fa165de16e21d1bd37afef3790b80da971` plus unrelated/concurrent overlays.
No other owner's changes were modified. The coordinator's latest supplied
checkpoint advances the .8B isolated diagnostic beyond the live shared-plan
summary: source-only five phases1/1/64/2/1, all24/fullV/all42 exact, fault and
regression gates/fullgate17 pass, forced353, frontier0/unpublished. Natural
feedback/production serving/full vLLM parity stay open. The actual2B stock-vLLM
natural reference is accepted; native ZG2B remains open. These are retained
owner results, not tests executed by this research. [Coordinator run](/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-upstream-cuda-root-20261020/TODO.md).

### Z1 — Prepared/fused/batched/private graph

B: original GGUF CUDA B2/B4/B8, source-ordered dynamic activation fusion and private graph qualifiers; production graph admission remains open. Qualification files: resident_inference/graph/qualification.rs, ragged_qualification.rs, prefix_qualification.rs; live serving TODO governs scope.

- [crates/zerogravity-gpu/src/resident_program.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_program.rs)
- [crates/zerogravity-gpu/src/resident_program/batch.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_program/batch.rs)
- [crates/zerogravity-gpu/src/resident_program/batch/graph.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_program/batch/graph.rs)
- [crates/zerogravity-gpu/src/resident_inference/fused.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_inference/fused.rs)
- [crates/zerogravity-gpu/src/activation_quant_fused.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/activation_quant_fused.rs)
- [Live serving plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/serving-and-speculation/TODO.md)

### Z2 — Trained forward and isolated original-source numerical work

B old GGUF .8B/2B bounded context; I new .8B source-only full-model diagnostic. Current model source/state source order preserved; native2B, production vLLM profile and27B runtime open. No Flash-Next QSA/MoE/learned-bank runtime found.

- [crates/zerogravity-model-qwen35/src/forward.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-model-qwen35/src/forward.rs)
- [crates/zerogravity-model-qwen35/src/original_binding/window_model/vllm_model.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-model-qwen35/src/original_binding/window_model/vllm_model.rs)
- [crates/zerogravity-gpu/src/resident_inference/native_fragment/full_model_qualify.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_inference/native_fragment/full_model_qualify.rs)
- [crates/zerogravity-gpu/src/resident_inference/native_fragment/full_model_qualify/faults.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_inference/native_fragment/full_model_qualify/faults.rs)
- [crates/zerogravity-core/src/accelerator/program/vllm_model.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/accelerator/program/vllm_model.rs)
- [Live model plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/model-execution/TODO.md)
- [Live numerics plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/numerics-and-context/TODO.md)
- [Live optimization plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/cpu-gpu-optimization/TODO.md)

### Z3 — Immutable source and supported quant codecs

B F32/F16/BF16 plus Q4_0/Q4_1/Q8_0/Q6_K enum/decoder scope; bounded authenticated source windows. No .hgn, ht43 or Halogen quant importer; supported GGUF source cannot imply every role/shape/kernel qualified.

- [crates/zerogravity-core/src/quant.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/quant.rs)
- [crates/zerogravity-artifacts/src/quant.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-artifacts/src/quant.rs)
- [crates/zerogravity-artifacts/src/source_weights.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-artifacts/src/source_weights.rs)
- [crates/zerogravity-artifacts/src/source_weights/tensor_window.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-artifacts/src/source_weights/tensor_window.rs)
- [crates/zerogravity-artifacts/tests/tensor_window.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-artifacts/tests/tensor_window.rs)
- [Live memory plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/memory-and-streaming/TODO.md)

### Z4 — Physical authority, completion/quarantine and source-only SSD

B configured physical pools/owned leases/uncertain-work quarantine and immutable bounded extent/window reads. P complete OS/UMA admission, async model streaming/expert residency/eviction/ABA, all-domain paging. Source-only SSD tests do not establish physical streamed inference.

- [crates/zerogravity-core/src/resource_arbiter.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/resource_arbiter.rs)
- [crates/zerogravity-gpu/src/resident_leases.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_leases.rs)
- [crates/zerogravity-gpu/src/physical_quarantine.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/physical_quarantine.rs)
- [crates/zerogravity-artifacts/src/ssd_extent.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-artifacts/src/ssd_extent.rs)
- [crates/zerogravity-artifacts/tests/ssd_extent.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-artifacts/tests/ssd_extent.rs)
- [Live memory plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/memory-and-streaming/TODO.md)

### Z5 — Fair workers and bounded HTTP

B model/session fairness and real provider worker foundations; inspected HTTP GPU_CAPACITY512, PREFILL_QUANTUM32, one worker/active<=8, opt-in original CUDA cohorts. P genuine multi-token temporal prefill, fair aged mixed scheduling, physical stream overlap and all-profile production integration.

- [crates/zerogravity-core/src/device_queue.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/device_queue.rs)
- [crates/zerogravity-gpu/src/model_workers.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/model_workers.rs)
- [crates/zerogravity-cli/src/serve/gpu_engine.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-cli/src/serve/gpu_engine.rs)
- [crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-cli/src/serve/gpu_engine/cohort.rs)
- [crates/zerogravity-cli/tests/serve_gpu/prefix.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-cli/tests/serve_gpu/prefix.rs)
- [Live serving plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/serving-and-speculation/TODO.md)

### Z6 — Complete roots and checkpoints

B immutable logical roots, sole SessionAuthority and bounded portable-checkpoint foundations; prefix qualification consumes one teacher token/request per wave. P general source/profile/tenant-safe paged complete-prefix cache/COW and product disk persistence.

- [crates/zerogravity-core/src/state.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/state.rs)
- [crates/zerogravity-core/src/session_root.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/session_root.rs)
- [crates/zerogravity-model-qwen35/src/checkpoint.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-model-qwen35/src/checkpoint.rs)
- [crates/zerogravity-gpu/src/resident_checkpoint.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_checkpoint.rs)
- [crates/zerogravity-gpu/src/resident_inference/prefix_qualification.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-gpu/src/resident_inference/prefix_qualification.rs)
- [Live serving plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/serving-and-speculation/TODO.md)
- [Live memory plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/memory-and-streaming/TODO.md)

### Z7 — ONE engine pipeline/context structure

B pure source-neutral DAG checker, source-granted immutable context owner/reader/filter and Gemma family structural contracts. These own comments explicitly refuse interpretation as loaded models/trained media/NPU/router execution. P actual multi-model integrated fabric and Gemma runtime.

- [crates/zerogravity-core/src/pipeline.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/pipeline.rs)
- [crates/zerogravity-core/src/context_snapshot.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/context_snapshot.rs)
- [crates/zerogravity-core/src/context_filter.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/context_filter.rs)
- [crates/zerogravity-cli/tests/pipeline_config.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-cli/tests/pipeline_config.rs)
- [crates/zerogravity-model-gemma4/tests/family_contract.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-model-gemma4/tests/family_contract.rs)
- [Live model plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/model-execution/TODO.md)
- [Live numerics plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/numerics-and-context/TODO.md)
- [Live optimization plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/cpu-gpu-optimization/TODO.md)

### Z8 — Speculation/sampling/grammar

A actual trained MTP, runtime prompt/suffix lookup, speculative correction/rollback and GPU grammar execution in inspected crates. Metadata/target entries are contracts, not execution. Existing StateRoot/branch foundations are prerequisites; live serving plan holds implementation queue.

- [crates/zerogravity-core/src/targets.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-core/src/targets.rs)
- [crates/zerogravity-artifacts/src/model/qwen_owned.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-artifacts/src/model/qwen_owned.rs)
- [Live serving plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/serving-and-speculation/TODO.md)

### Z9 — Tokenizer/output and benchmark accountability

B official Rust tokenizer and bounded native/provider-internal diagnostics. P equivalent natural full-model serving benchmark matrix, source-correct SLO-goodput, true ITL/fairness and complete pinned-vLLM comparison. Existing profiler windows do not establish broad comparative counts.

- [crates/zerogravity-model-qwen35/src/tokenize.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-model-qwen35/src/tokenize.rs)
- [crates/zerogravity-cli/src/serve/token_metrics.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-cli/src/serve/token_metrics.rs)
- [crates/zerogravity-cli/src/serve/gpu_engine.rs](/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/crates/zerogravity-cli/src/serve/gpu_engine.rs)
- [Live serving plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/serving-and-speculation/TODO.md)

### Z10 — Optimization, numerics and acceptance plans

P source-bound capability kernels, tuned shape plans, production graph, explicit derived quantization and broad quality/performance promotion. No new physical acceptance follows from this report.

- [Live model plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/model-execution/TODO.md)
- [Live numerics plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/numerics-and-context/TODO.md)
- [Live optimization plan](/home/pvl/spaces/abird/src/zerogravity/.agents/plans/cpu-gpu-optimization/TODO.md)



The plan index keeps optimization in Icebox; this report proposes sequencing
for coordinator reconciliation and creates no promotion. Existing pending
same-source vLLM parity/bench order dominates all external prior-art priorities.

## Ranked integration after vLLM parity

1. **G1: .8B/2B source-profile hot path and short matched benchmark.** Qualify
   natural feedback/publication and HTTP route first. Profile actual full
   request kernel launches, copies, host synchronization, output/tokenizer cost,
   resident scratch and target arithmetic. Then specialize existing projections,
   GDN/residual/norm/attention and source-exact fusions by decode, temporal
   prefill and verify shapes. Reuse prepared buffers/plans; production graph
   admission is a dependency, not shorthand for already having captured graphs.
   This is the largest generic immediately relevant Halogen lesson: remove
   repeated orchestration/weight work at an exact source boundary.
2. **G2: true temporal prefill plus aged fair batches and hybrid page/prefix
   ownership.** Preserve per-profile chunk arithmetic. Reuse immutable complete
   authorized prefixes, paying COW for mutable tails/recurrent checkpoints.
   Charge peak pages, branch scratch, disk staging and metadata under the same
   arbiter. Integrate short C1/2/4/8 tests before long contexts. Use Halogen's
   long-prefill stall and cache-credit lessons to choose scheduling experiments;
   performance comes from our matched ablations.
3. **G2 next: long-context gate and actual Qwen3.8-27B execution.** Bind the
   accepted original-index tensor windows to original TensorSource/BufferLease
   readiness. Coalesce/pipeline bounded source reads without mandatory host work
   on resident hits. Full27B numerical/root/physical qualification precedes
   throughput. This dense27B model does not acquire Flash-Next's sparse experts
   or learned lookup bank merely because both are Qwen.
4. **G3: target-authoritative generic runtime proposal stages.** Once basics
   and matched measurements close, try prompt/suffix lookup and grammar drafts
   as bounded optional stages, then authentic trained MTP where the selected
   checkpoint supports it. Cost model chooses draft depth, joint batch and
   backoff. Strong rejection/rollback/RNG/grammar and fair throughput gates are
   required; Halogen's acceptance advantages cannot be transferred as numbers.
5. **G4: Gemma4 E2B→26B→31B.** Reuse the same admitted plan/workspace/page/cache
   facilities with trained Gemma contracts. PLE/MoE/dense layouts require their
   own source oracles; structural config tests do not qualify inference.
6. **Later specialized targets:** Flash-Next QSA/indexer/grouped experts/learned
   lookup and gfx1151 matrix kernels, explicit calibrated derived quant packs,
   optional KV8/YaRN/approximate reuse, and NPU co-execution. Defer DeepSeek and
   other hardware until the user's Qwen→Gemma order is met. The local gfx90c
   hold remains intact; SM86 RTX3060Laptop6GB/relative5GiB is the present bounded
   performance target, not an APU128GB substitute.

## Mandatory gates and useful compositions

- **G0 source/exactness:** pin checkpoint/revision/tensor roles, immutable
  logical source+owned descriptor identity, codec/pack/calibration, tokenizer,
  numerical schedule, dtype/reductions, compiler/device/phase and state schema.
  Compare all required full logits/state/intermediates before timing; keep
  source bytes and original qualified profile unchanged. Different accepted
  profiles have explicit compatibility guards.
- **G1 benchmark:** accepted same-source natural requests; short solo then
  C1/2/4/8, cold-process/storage/prefix/pack cache states separate; real TTFT,
  individual ITL tails, committed SLO-goodput, offered/refused/canceled work,
  dispatch/copy/IO counts, physical peak/fragmentation and power. A failed or
  slower candidate stays recorded. Repeated complete-request improvement and
  unchanged accepted semantics are the promotion criterion.
- **G2 state:** independent complete StateRoots; greatest common restorable
  frontier across KV, recurrent, conv, indexer, draft, RNG/grammar and media;
  immutable authorized shared prefix, source/profile/tenant checks and COW.
  Captured/encoded/queued/in-flight readers hold generations until actual
  completion. Persistence includes corruption/atomicity/compatibility gates.
- **G3 cancellation/speculation:** every accepted-prefix length, rejection,
  stale epoch, output backpressure, pre-submit/late cancel and unknown physical
  completion. No partial publish, invented Ready or timed release; preserve
  quarantine/fence and actual driver ownership through final drop.
- **G4 physical:** one ResourceArbiter for source weights, prepared variants,
  mutable state, scratch, page metadata, branch peaks, IO staging, graph/native
  driver resources and heterogeneous UMA/GPU/RAM/SSD backing. Charge physical
  bytes once across shared views; validate OS/cgroup headroom separately.
- **G5 phase acceptance:** meaningful affected regression and full
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  after any implementation; GPU/model changes add real source/state/cancel/
  physical tests. No such implementation gate was invoked for this research.

**Composition hypotheses, not promised outranking:** shape-tuned projections +
source-exact fusion + admitted production replay can remove repeated launch work;
true temporal prefill + exact prefix/COW + fair batches can reduce both repeated
compute and head-of-line delay; immutable source windows + bounded prefetch +
residency can hide some storage latency without creating a second allocator;
prompt/grammar/MTP proposals + joint verification can amortize target weight
reads when acceptance pays. Each is one compiled capability plan sharing source,
worker, physical budget and root authority. Benefits can interfere through
memory bandwidth, scratch, power, reduced batching opportunities and copy cost;
qualify singles then material pairs/combinations and disable losing plans.

## Inspection coverage and artifacts

- Enumerated the complete current GitHub tree and fetched all public UTF-8
  blobs except the image. `source-inventory.json` pins tree/tag/API metadata;
  `hf-source-inventory.json` pins separate model-card identity. No binary/image/
  checkpoint content was fetched or examined.
- Detailed reads: README, LICENSE, notices, limits, precision/QUANT, quality,
  performance, configuration, FLAGS, GGUF, checkpoint tools, sessions, sampling,
  NPU and host setup. Changelog inspected by all version headings plus focused
  optimization/quality/failure searches and relevant sections, not every line.
  Deploy entrypoint and fabric-clock script inspected at host-admission,
  startup, process/watchdog, NPU, and command-dispatch sections; benchmark files
  inspected directly. Other fetched client/API pages were targeted reads or
  inventory only, not a complete API parity audit. Internal gates named by
  scripts but absent from the public tree remain uninspected.
- Local inspection: plans layout/queue/optimization/serving/memory/model/
  numerics TODOs; batching/paged design from MAIN; historical Strix dossier and
  source atlas only as discovery references; implementation files in Z1–Z10.
  Fresh source and live plans outrank their dated prose.
- Memory registry quick search returned no relevant hit; no memory-derived
  technical claims were used. External snapshots under `primary-text/` are
  research inputs, not imported implementation or a requirement to archive them.
- Only owned files under this run directory were written. No commits, staging,
  push, shared-plan edit or runtime acceptance was performed.
