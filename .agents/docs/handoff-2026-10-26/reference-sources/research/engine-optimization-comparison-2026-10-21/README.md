# Execution fabric: optimization comparison

Author: Project owner
Co-authors: GPT-6 and independent research agents
Coordinator: `/root`
Date: 2026-10-21
Status: source research and proposed integration; no new performance acceptance.

## Outcome

The projects optimize different workloads, numerical contracts and memory
hierarchies. Reuse their mechanisms through the existing Rust engine. A source
or design inventory is not runtime support, and composing individually useful
optimizations does not establish a combined speedup. The objective is the
measured Pareto frontier of correct delivered goodput, latency, memory, energy
and extensibility, rather than an unsupported universal ranking.

All eight unique investigations were completed independently before this
reconciliation. The duplicate NVIDIA URL denotes one deployment recipe, not
two engines. Each report records read depth, primary sources, uninspected
areas, licenses and numerical/performance limits. No foreign engine ran and no
models, SDKs or executables were acquired for this research.

## Independent sources and why they differ

| Project / inspected pin | Design emphasis | Transfer and boundary |
|---|---|---|
| [vLLM](vllm.md), main `8e2beaa3bff595577a069d49d96bde9074294ec2`; stable0.32.0 separately pinned | Broad GPU serving: token-budget scheduling, paged/hybrid state, stable rows, specialized kernels, graphs, speculation and distributed placement | Strong serving/completion/metadata ideas. Our actual oracle stays pinned0.31.0; latest research does not replace that baseline. Some current-main features are experimental and selected PRs remain open. |
| [llama.cpp](llamacpp.md), main `339f3242c761f16c27e5ffcde1d35dcc31933a55`; stable0.7.0 separately pinned | Portable CPU/GPU execution, GGUF codecs, heterogeneous placement, hybrid sessions and desktop serving | Source-profiled SIMD, packing, dispatch, effective fused lifetimes and decode-protected expert residency. Its batch-dependent math is not automatically our exact profile. |
| [TensorFold](tensorfold.md), `d1018b318fb51981c4a18f12e40269a4d3ee9713`,1.1.1 | Native Zig Metal/CUDA lanes, predictable row arithmetic, prepared rounds, complete cache/state and proposal policies | [72-mechanism inventory](tensorfold-inventory.md), plus3 model/input-changing features. Historical Python/MLX and SmartTensor lineage are separate. Own serial/resume equivalence is not cross-engine equality. |
| [Halogen](halogen.md), `3f2adcfca6be623c8ec4f69331cbf97beff55251`,0.18.13 | Highly specialized gfx1151 Flash-Next, UMA/disk learned lookup, hybrid prompt cache, joint speculation and NPU side models | [65 public mechanisms](halogen-inventory.md). Core is closed under a custom EULA; graph/ISA/allocator internals stay unknown. Cache2, KV8 and moved-message reuse have distinct numerical contracts. |
| [DwarfStar](dwarfstar.md), `5379c1c9ac315f1540371b06a3db88491d4059ff`; no release/tag | Explicit C model graphs, desktop/UMA memory hierarchies, specialized CUDA/HIP/Metal, SSD and limited multi-device layouts |60 mechanisms plus6 separate proposal groups. Exact sampling is opt-in for some modes; verifier reduction can differ from serial. Source/issue/version contradictions are retained. |
| [Maya](maya.md), `3384b5e9366e696e5f23f51772fb94a8a7948e41`; VERSION1.1.19/tag differs | Tiered GLM/DeepSeek, CPU/GPU expert overlap, phase lending, layer-major prefill, MTP/DFlash and local multi-GPU | [42 mechanisms](maya-inventory.md). Placement/grouping/prefill can change arithmetic. Separate upstream model runners are not a proposed Rust design. |
| [DeepInfer](deepinfer.md), `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`,0.1Beta | DeepSeek V4.1 reference execution on SM100/Ascend950: overlap, Mega-MoE, confidence-budgeted verification and wide EP | Stable buffers, causal adaptive scheduling and overlap are portable ideas. Model registration is narrow; full production global services are not all released. Bounded replay and residual sampler penalty are explicit approximation boundaries. |
| [NVIDIA2×GB10 recipe](nvidia-guide.md), accessible recipe `16dace6ce9afd9ce6384a838e9b08d2bf2291f69` | Hardware/build/launch composition of an older vLLM branch, TP2, FP8 KV, MTP2, graphs and prefix cache | Deployment evidence, not a new engine. Configured1M capacity and short-prompt throughput do not prove1M prefill performance; moving build refs and revised UCX diagnosis remain explicit. |

The independent numbered inventories contain362 source-scoped rows across six
projects. Coordinator extractions add [38 DeepInfer entries](deepinfer-inventory.md)
and [16 deployment entries](nvidia-guide-inventory.md), for416 entries altogether.
Many describe the same underlying mechanism; these counts do not mean416 distinct
wins, all branches inspected, or knowledge of Halogen's nonpublic implementation.

## Current local evidence

Current source, executed results and live plans outrank all report snapshots.

- Existing original GGUF .8B/2B CUDA B2/B4/B8, unequal-frontier/refill,
  teacher-prefix waves, compact greedy, private captured graphs and46 row-state/
  owner/failure cases are real bounded qualifications. Production graph
  admission and true multi-token temporal prefill remain open.
- ISO original-HF .8B five forced phases close full24 dual boundaries, all42
  active domains, full vocabulary and5 greedy packets exactly; actual faults,
  regressions and complete `scripts/check.sh` gate17 pass. Frontier0/forced353
  diagnostics do not establish natural publication or serving.
- Independent original .8B and2B natural vLLM references pass all42/24-boundary/
  full-vocabulary pairing. New2B reference3 checks actual24views of one physical
 1,062,862,848B cache backing, unchanged relative5GiB target and source/cleanup.
  Native2B, production serving and matched timing remain open.
- Subsequent ISO .8B natural binding/publication passes finalhelper43's seven
  physical modes: all24/all42/fullV and actual feedback match, original Sessions
  publish chains[198,285]/[72,18], prefix64emits0. StrictClippy/fullgate18 pass;
  normal/safe cuts retire, uncertain cuts quarantine without publication. This
  remains five fixed1/1/64/2/1 calls, rotary0..67 and stale-restore rejection only.
  Subsequent finalhelper45 qualifies two fresh request pairs on one loaded model,
  same scratch/fresh candidates/immutable packets, actual cancellation retry and
  fresh restore/competing commit; seven new+nine regression physical modes,
  strictClippy2/fullgate19 pass. Canonical/ragged shapes, production serving and
  native2B still require qualification; repeated five coordinates grant no others.
- Authenticated original27B windows/source scans exist. Dense~55.56GB source
  under a5GiB working target needs real bounded execution/IO planning; MoE
  cache hit rates cannot skip its dense matrices.
- Gemma4 source/config/shared-KV/MoE contracts are not trained runtime support.
  Different models may share granted text and identical preprocessing, not
  arbitrary embeddings/KV/GDN state.
- Maya's report found a missing batching-design link in the ISO checkpoint.
  The current shared [MAIN batching/prefix design](../../plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md)
  exists and governs; this is not a missing current plan.

## Cross-engine mechanism comparison

Numbers below identify mechanism families for reconciliation only. Detailed
per-project rows retain primary citations, local code/test anchors and exact
dispositions. A mechanism already described in historical design is **accounted
for in design**, not implemented. Alternatives in a family are candidates, not
a requirement to run every connector/cache/kernel simultaneously.

| Family | Public evidence | Existing local coverage → remaining work |
|---|---|---|
|01 Source identity, prepared layout, packed ordered members | TF M2/M7/K1; V20–25/28; LC02/05/10/11/14; H15–20; D04/08/28/31; MAYA01/04/07/36; DI loading/layout | Verified source owners/windows, some prepared packs and current source-selected projections → durable exact pack cache, full async source-to-device consumers and broader codecs. A new pack is not a new source issuer. |
|02 Numerical/quality profiles | TF M1/M6/Q1–6; V29–41/45/60; LC03/06/18/22/38/41; H07/11–14/17/40–43; D04/05/07/09/55; MAYA14/16/22/35/39; DI norm/sampler/replay | Bounded source arithmetic and isolated BF16 closure → explicit new profiles only after independent acceptance. No hidden casts, route omission, state compression or residual probability bias. |
|03 Dense and quantized kernel regimes | TF M3–5; V22/39–41; LC04–07/16–19/27/28/30/31; H01/08/09/15; D03/06–09/16–18; MAYA11/14/15/18–20/25 | Current source dots/native B2/4/8 and isolated actual projections → finite decode/prefill/verify shape bands, SIMD/packing/tile tuning, measured cross-row weight reuse. Sequence rows and temporal rows are distinct axes; HIP/Vulkan/Metal capabilities require their own physical gates. |
|04 Source-exact fusion | TF F5/M6–8/A3/4; V36/42/43; LC21/24; H02/06–10; D10/14; MAYA01/19/40; DI fusion/invariants | Existing bounded activation/state transport and dual residual/RMS paths → close legal regions without deleting required BF16/F32 rounding or state edges; account longer effective read lifetimes. |
|05 Attention, recurrence, positions | TF A1–8; V29–38; LC20–23/35; H02–05; D11–13/20–22; MAYA13/21–23/38; DI cache/sharing | Source-specific Qwen state/attention and isolated FLA/FA/rotary → natural temporal/long-context qualification. Stable sparse top-k, MLA, CSA/HCA/KDA and shared-KV are source-dependent; never add them to an unrelated dense model. |
|06 Captured execution and closed regions | TF F6/7; V46–50; LC14/25; H10 unknown; D19; MAYA24; DI stable graphs/PDL | Actual private graph qualifier → production constructors, driver/metadata/pool admission, complete reader/drop participation and source-bound fallback. SM≥9 PDL is not SM86 support; graph submit is not one physical kernel. |
|07 Persistent metadata and async feedback | TF F3/4; V04–08; LC14/26/32; D15/33; MAYA01/09/40; DI worker/remap/deferred release | Original actor/leases/compact packet exist → natural call binding/publication, leased stable rows, bounded diff upload, compact output and overlap. Prepared future work cannot publish uncompleted roots. |
|08 Fair native/temporal/ragged batching | TF C1/7; V01–04; LC33–35/52; H26–32; D35–37; MAYA11/25/26/41; DI prefill/decode scheduler | Finite fair workers and native independent rows → true temporal dense/GDN prefill, aged mixed-work budgets, explicit yields/output credits and real wakeups. Wider batches can alter numerical schedules and tails. |
|09 Physical backing, scratch/liveness, phase lending | TF C2/R1–3/10; V03/19/23/27/49; LC12/15/29; H21–25/35; D23/24/26/34/35; MAYA03/04/06/10; DI shared backing/profiled capacity | Original physical arbiter, actual completion/quarantine and fixed workspaces → OS/cgroup/UMA headroom, safe liveness reuse/lending and broader accounting. Equal bytes/hash do not prove one allocation; aliased views do not imply several backings. |
|10 Complete hybrid pages, prefix intersection and COW | TF A5–7/C6/R4/5/9; V09–14/18; LC36/37/39; H33–39; D38–42/48; MAYA23/33/34; DI block/sharing/replay | Complete roots/checkpoint foundations → paged production ownership and greatest jointly restorable frontier across every domain. Both COW endpoints and captured/queued readers retain charge through completion. |
|11 Persistence/restore economics | TF R6–8; V15–17; LC40; H36–39; D39–41; MAYA34; DI deployment cache description | Bounded portable diagnostic checkpoints → product persistence/corruption/atomicity/source/profile/tenant checks, restore-versus-recompute and fair tier promotion. Restoring saved bytes is not proof of cache-disabled equivalence. |
|12 Bounded source/state IO and prefetch | TF C4/R11/H1–4; V20/24/25; LC10/11; H52–54; D27–34; MAYA02/07–09/11/37; DI Engram/cache descriptions | Verified extents/windows → real async consumers, dense layer-window reuse and genuine copy readiness. Known lookup IDs and predicted expert misses need different queues; resident path incurs no mandatory IO. |
|13 MoE routing/grouping/cache/replicas | TF M8/9/H2–4; V43–45; LC49; H03–07/52; D07/14/23–30/51; MAYA02–06/12/18/28; DI expert balance/shuffle | Source/size-class/readership design only for trained sparse execution → source-exact stable route/group/combine, protected decode hotsets and admitted replicas. Deferred for models that actually contain experts; dense27B is separate. |
|14 Target verifier and full branch rollback | TF S1/A5; V52; LC37/42/48; H44/47/48/51; D45/46/48; MAYA29/32/33; DI sampling/state | Ordinary greedy/root primitives → exact target authority, accepted-prefix0..K, all state/RNG/grammar domains, wrong-draft rejection, EOS and physical failure gates. Distribution correctness and same-seed/bit trace are separate. |
|15 Proposal sources and joint adaptive budgets | TF S2–7/11; V53–59; LC43–48; H44–51; D43/44/47/49; MAYA29–31/42; DI DSpark/causal confidence | Plans/source metadata only → n-gram first without new weights, compatible trained heads later. Choose depth/cohort/graph/residency by total committed benefit and fairness; K=0 is valid. Confidence-dependent scheduling must not bias sampling. |
|16 Sampling, grammar, readout and delivery | TF S8–10; V07/08/51/60/68; LC32/53; H45–50/55/56; D40/57; MAYA32/40/41; DI sampler/grammar | Bounded greedy/status/readout and HTTP/SSE → declared stochastic/filter/RNG/grammar contracts and exact incremental detokenization. Skip unused teacher heads; retain full vocabulary for qualification. Compact readout does not eliminate target head compute. |
|17 CPU pools/topology and heterogeneous placement | V26; LC04–13; D03/50/60; MAYA15–18/26–28; DI NUMA | Existing CPU/fair-worker foundation → measured ISA/core/NUMA policies, bounded CPU/IO overlap and source-compatible partitioning. More threads, wider offload or addressability can lose. |
|18 Distribution/collectives/transport | TF P1–4; V61–66; LC50/51; D50–56; MAYA26–28; DI DP/CP/EP/EPLB; recipeTP2 | Contracts/fleet profiles → actual authenticated topology/partition algebra, transfers/epochs, collective failure and complete state handoff. UMA/RDMA and discrete staged copies differ. Defer until authorized model/hardware work. |
|19 Shared context, models, media and profiles | TF F1/2/S12/C3; V67/68; LC01/39/43/53; H39/59–64; D01/02/57; MAYA01/34/37; DI source/worker separation | Pure typed pipeline/ContextOwner scaffolds and immutable model reuse → recursive compile-time regions, profiles, real grants/freshness and trained multi-model stage binding. Shared HTTP or two model loops do not establish one engine. |
|20 Model-dependent semantics/approximations | TF A8/Q6 and3 explicit features; V30/31/45/58/60; LC23/41/46; H02–05/40–43/52; D20–22/31/32; MAYA21/37–39; DI mHC/YOCO/DSpark | Preserve source/operator contracts; deferred DeepSeek/GLM/Flash-Next/Gemma prerequisites and explicit changed-quality profiles. No context truncation, changed expert count or generic model surgery in exact mode. |
|21 Measurement, tuning, power and negative cases | TF C5/8/K2/3/R8; V49/59/69; LC31/54; H57/58/65; D05/58–60; MAYA20/42; DI cost cache/plots; recipeprobe | Bounded tools and actual counts → matched unprofiled complete-request experiments, cold/startup/pack costs, true token timing, all offered work, memory/energy/fairness, uncertainty and combined ablations. No multiplication/addition of external gains. |

## Priority and model order

The [profile crosswalk](profile-coverage.md) organizes all21 families into
reusable nested pipeline components and eight proposed recipes. Every recipe
retains source/arithmetic/physical prerequisites and explicit qualification gaps.

1. **Current correctness gate:** natural .8B call binding, original Session
   publication/actual feedback, native2B full source/all-state, default serving
   integration and negative completion ownership. Keep failed runs and the
   original numerical policy. No newly researched speed path bypasses this.
2. **Matched short baseline immediately after parity:** established profiles,
   C1/C2/C4/C8, original ABBA+BAAB controls and offered-load tails/goodput.
   Attribute CPU preparation, transfers, launch gaps, kernels, sampling and
   delivery in separate diagnostic runs. Preserve source/EOS/caps/clocks.
3. **First substantial generic improvements, selected by that profile:**
   reusable row/buffer/binding metadata; remove measured redundant transfers;
   qualified production replay and exact adjacent fusion; useful source-window
   sharing; true temporal dense/GDN prefill with fair mixed batches. Existing
   native batching/private graphs are foundations, not work to reimplement.
   Re-run short concurrent comparisons after singles and material combinations.
4. **Long contexts and shared state:** independent natural source/rotary/attention
   limits, genuine prefixes and all-state restore first; full-domain pages,
   exact common-prefix/COW and durable tiers as justified by capacity/workload.
5. **Dense Qwen27B:** actual source-window CPU/GPU execution before throughput;
   bounded coalesced layer/row reuse and completion-backed prefetch if needed.
   Measure bandwidth floors/activation spill and total live resources.
6. **Gemma4 E2B→26B-A4B→31B:** trained source semantics/oracles first, then reuse
   compatible kernels/scheduler/residency. PLE, SWA/K=V/shared-KV and MoE require
   exact variant contracts. Compatible assistant MTP comes after target runtime.
7. **Proposal experiments:** first target-verified n-gram/Recall/suffix; compatible
   trained Qwen/Gemma MTP/EAGLE/DFlash/DSpark later if actual artifacts support
   the exact target. Raise priority for a measured low-QPS opportunity; disable
   under saturation or poor total benefit. No training/acquisition authorized.
8. **Deferred:** DeepSeek/GLM/sparse Flash-Next model semantics, new ISA/device
   paths, EP/EPLB/DBO/remote P-D/CP/RDMA, media/NPU and explicitly derived state/
   weight/approximation profiles. Retain all current payload/device holds.

## Composition opportunities beyond individual ports

These are falsifiable hypotheses, not accepted optimizations or novel-algorithm
claims. The [recursive pipeline/profile proposal](../../plans/pipeline-composition/REGIONS_AND_PROFILES.md)
develops their interfaces through the existing execution spine; its
[live queue](../../plans/pipeline-composition/TODO.md) retains implementation gaps.

- **Reuse before scheduling:** find the complete authorized prefix frontier,
  subtract avoided work, then choose compatible batch/graph/verify shape. Test
  fewer repeated weight/state reads against padding, delayed cold requests and
  changed hit/continuation mix.
- **One immutable source backing, several consumers:** share genuinely common
  verified chunks/prepared packs and tied embedding/head storage; gather actual
  committed-token rows into private activations. Count real retained allocations,
  not equal hashes. Test source mutation, forked callers, capture and cancellation.
- **Joint optional-work controller:** choose finite approved combinations of
  prefill quantum, row band, speculative depth, graph and prefetch horizon from
  measured complete-cycle cost and available physical headroom. Begin with
  simple guarded tables/hysteresis; no online unbounded compiler/search.
- **Lifetime-aware fusion and residency:** use effective fused read intervals
  plus queued/captured readers to share scratch safely, while protecting useful
  decode assets from transient prefill. Test overlap, slow copies, stale slots
  and pressure; a larger kernel or shared arena can reduce occupancy/progress.
- **Dependency-aware IO:** deterministic input lookup rows, dense next-layer
  windows, produced-state restore and uncertain expert predictions share fair
  physical IO admission but have separate identities/priority/readiness.
- **Authorized cross-model preprocessing:** freeze granted context once, reuse
  identical rendering/tokenization/media work only where contracts match, and
  keep model-specific state private. Test revoke/freshness at actual delivery.
- **Closed-region specialization:** represent hierarchy for reasoning, then
  inline/fuse/outline at preparation. Replace a proven region with one kernel,
  an admitted graph or another qualified schedule with the same exports/effects.
  Benchmark actual device kernels and complete requests, not fewer source nodes.

## Promotion and provenance

Every selected change needs independent complete-logit/all-state/natural-output
checks, boundary/rollback/source/profile refusal, actual pressure/cancel/unknown
completion and fair serving evidence. Run the relevant physical tests and full
approved isolated `scripts/check.sh` implementation gate. A capture failure
with uncertain work cannot silently fall back on a fenced service.

Performance acceptance uses equal immutable source, tokenizer/template, numerical
profile, request/EOS/limits, hardware/driver and declared physical caps. Separate
cold process, storage cache, prepared pack and warm prefix. Count correct useful
committed/delivered outputs and requests meeting SLOs over **all offered elapsed
time**, with failures/late/canceled/refused work retained. Report true-token
TTFT/ITL/E2E tails, fairness, memory/retention/IO/energy and startup/tuning cost.
External headline speedups and assumed draft acceptance never enter our results.

Permissive repository licenses do not license every kernel, model or dependency;
Halogen is nonpublic/custom-EULA. No source import was performed. Future work
retains the [provenance policy](../../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md),
Rust-owned math/kernel requirement, immutable source identity, independent roots,
one physical arbiter/fair workers and actual completion/quarantine.

Execution tasks live in the existing optimization, numerical, serving, memory,
model, pipeline and distributed TODOs; this research is not a second work queue.
