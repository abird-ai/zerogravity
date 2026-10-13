# Zerogravity — SSD Streaming as First-Class Model Residency: Comprehensive Next-Agent Handoff

**2026-10-15 • DOCUMENTATION / ARCHIVE TOOLING ONLY. NOT A NEW INFERENCE IMPLEMENTATION.**
**Source repository:** /home/pvl/spaces/abird/src/zerogravity, dotlink-l5 root /home/pvl/spaces/abird.
**User instruction:** user explicitly granted permission to resolve the nested comprehensive-archive generator and strengthen architectural docs/plans, and required SSD streaming to feel natural with the **same tensor/materialization structures** backing GPU/RAM or SSD. No source model, SSD benchmark, huge weight download, ROCm/CUDA driver change, running deployment or model inference code was requested.
**Baseline:** main was clean at bb93449 at start of the work; verify live HEAD/worktree before any future code. Existing native CPU/GPU Qwen3.5 Phase1/physical partial-region evidence remains unaffected. Prior user model order and no-foreign-inference-engine policy remain.

## 1. Required reading before implementation

1. Read existing v4.3 authority: .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md; MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md; COMPOSITION_AND_EXTENSION_CONTRACTS.md; TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md.
2. Read **new governing storage contract**: .agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md in full.
3. Read independent original-source research: .agents/prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md (DS4 expert SSD, file-only n-gram lookup, disk KV, source issue #384/503/629, prior offload projects).
4. Read implementation and all 32 planned tests: .agents/research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md.
5. Read existing .agents/docs/MEMORY_RESIDENCY_AND_IO.md, ASYNC_RESOURCE_COMMAND_IR.md, STATE_CACHE_AND_TRANSACTIONS.md, FORMATS_QUANTIZATION_AND_LOADING.md, COST_MODEL_AND_AUTOTUNING.md, INVARIANTS_AND_FAILURE_SEMANTICS.md and MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md. These now explicitly point at new contract.
6. Read v5.1 verified ModelAssembly and RealtimeSession/QoS specs, EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE plan, DECISION_REGISTER and original complete historical archive **progressively in full**. On true takeover, revalidate most recent Git, artifact hashes, driver and actual device.
7. Read the current modified doc generator and offline regression suite before regenerating archival docs; source archives are snapshots, not editable primary source.

## 2. Architecture diagram — one source tensor, several legal physical homes

~~~text
Verified ModelAssembly / SourceTensorId / TensorView / exact quant + shape
                         |
          Prepared compiled region / KernelCandidate
                         |
                ReadyTensorLease binding
                     /          \
             Direct READY       Miss/provisional
             device slot         source extent file
                  |                   |
        same compiled kernel       bounded async IO
             dispatch               + source verify
                  |                   |
       true completion event     pinned RAM/mapped bytes
                  |                   |
            lease retire          H2D+true upload event
                  |                   |
            output-ready         new generation atomically published
                  |                   |
          TaskCoordinator <---- same region kernel and physical event
                  |
             ONE StateRoot
~~~

**Resident path:** all required source pointers already physically Ready; no per-token filesystem/IO worker wakeup, no global cache lookup, no extra model-specific scheduler, no hidden blocking OS read. The abstraction is intended to add negligible overhead to direct GPU computation when all weights fit.

**SSD path:** same source tensor, quant, model semantics and final mathematical output; bounded, measured SSD IO/mapping and true upload completion. Actual miss latency cannot be zero; prefetch and staged overlap are performance candidates under source and physical effect guards.

## 3. One source/four policies

- **Routed experts:** dynamic per-expert/projection slabs, source quant/size class, valid routes from model router, hotness/IO cost, pinned from GPU command encoding through actual completion (DS4 #384).
- **Huge lookup rows:** n-gram/Engram table remains immutable disk primary with tiny coalesced selected-row cache, source hash/address and row codec. Different from expert bulk slabs.
- **Frozen session StateRoot:** committed private KV/GDN/SWA/codec snapshots to disk with exact model+session+format identity, immutable frozen state, encryption/TTL policy and compatible replay; no mutable active state sharing or false zero-copy restore.
- **Unpinned dense trunk:** optional, explicitly QoS-downgraded capacity mode if dense mandatory weights cannot fit; can be dramatically slower (file-backed lookup source). Not an automatic equal-speed source kernel mode.

All share the original SourceExtent physical IO provider and ResourceArbiter, but a unified LRU across mutable state, experts, lookup, media and trunk is NOT accepted.

## 4. The two highest-priority external regression fixtures

### Expert slot corrupting a queued GPU graph (DS4 issue #384)
- Create A/B layers and cache capacity one. Encode GPU reader for expert X from slot S but do not execute yet.
- Request Y from later layer. Cannot recycle S while earlier GPU encoding/capture/queued reader exists, even if CPU cache directory thinks X is idle.
- Flush/wait at legal boundary, use another verified tier or backpressure; overwrite forbidden.
- Compare original full logits/StateRoot under extreme cache pressure, not only cache table invariants.

### Mixed Q2/Q4 cache thrash and budget overcommit (DS4 issue #503)
- Alternate unequal byte-size expert matrices across layers, then send a short prefill following a warmed decode.
- Preserve active size classes, enforce one global byte cap across all, avoid unconditional cache-clear on format switch/prefill, reserve H2D/staging before GPU submit.
- Cache hit ideally permits direct resident pointer use without copying multiple GiB/token; measure actual DMA bytes and latency even at 97% hit rate.
- Smaller cache/cold SSD may slow tremendously, but must not corrupt source math or hang.

## 5. Phased implementation and honest gates

| Phase | Future owner | Output and test gates |
|---|---|---|
| SSD-P0 / P04 | semantic/CIR | fake GPU/RAM/SSD logical tensor, real generation/lease simulation, SSD-01..04/08/12 |
| SSD-P1 / P03 | ArtifactCatalog+IO | source extent/open descriptor, byte integrity/format, portable async IO, SSD-05..07/17..20 |
| SSD-P2 / P05/P07 | DeviceWorker | real GPU slot, H2D event, encoded/queued reader lease, graph-safe indirection, SSD-09..16 |
| SSD-P3 / P10/P12 | cache/resource | mixed expert size and total bytes, direct pointer hits, protected prefill/cache, SSD-21..24/26/29 |
| SSD-P4 / P11/P10 | model state/lookup | giant sparse lookup and private committed disk KV/codec snapshot, SSD-25/27/28/30 |
| SSD-P5 / P16/P20/P22 | admission/benchmark | full physical original source GPU/SSD parity and memory, cold/warm latency and QoS, SSD-31/32 |

All 32 SSD-xx tests **PLANNED**. No exact model/backend promised until a trained full source checkpoint, actual disk/driver, numerical and StateRoot gate passes. Existing Qwen3.5 original CPU through 8K 16/16, remaining 16K/32K deferred; no more native GPU full-model or audio/video execution is asserted.

## 6. Documentation archive generator repaired in this phase

The former full archive builder failed on nested prior v5.1 complete handoff with a fixed nine-tilde Markdown fence. The minimal permitted tooling fix chooses deterministic per-source delimiter longer than any literal tilde run; source bytes, including UTF-8 and trailing newlines, remain recoverable by indexed original byte length; output is stable and hash-indexed. It adds offline regression tests for nested archives, source bytes, determinism and previous committed handoff. The existing scripts/check.sh source-only Python gate runs these tests. Regenerate **after** all documentation is committed and source list is stable; manifest last; use --check without guessing success.

Do not remove the previous self-contained multimodal handoff, discard user decisions or copy its entire content manually. The deterministic archive must include prior archive as a nested **source document**, with dynamic safe fences; the repo-wide SHA manifest excludes itself and records exact current source/archived bytes. This is a documentation-tooling bug fix only, not an inference engine change.

## 7. Original external sources

- DS4 SSD streaming https://github.com/antirez/ds4/blob/main/docs/SSD_STREAMING.md
- DS4 expert slot queued command race https://github.com/antirez/ds4/issues/384
- DS4 mixed Q2/Q4 thrash https://github.com/antirez/ds4/issues/503
- DS4 host RAM cache proposal https://github.com/antirez/ds4/issues/629
- DS4 server disk KV https://github.com/antirez/ds4/blob/main/docs/SERVER.md
- File-backed disk lookup/trunk and KV prior art (reference removed).
- File-backed lookup flags prior art (reference removed).
- Prior offload background paper https://proceedings.mlr.press/v202/sheng23a.html
- Activation-aware expert-offload prior art (reference removed).

Upstream measured numbers are source author reports, not new Zerogravity benchmarks. Prior-art source licenses and versions require re-verification before implementing algorithms.

## 8. Decision provenance and scope

**VST-USER-001** in decision register records user approval for docs/plans/documentation-tool repair and requirement for first-class SSD residency. Do not confuse this with authorization to run huge model loads or a live remote provider. The implementation must preserve user design goals: simple intuitive one Rust engine, no duplicate resource/state writer, composable virtual tensor backing, zero-extra software work on resident path, and truthful slow/storage-bound modes.

A future implementation agent must follow the smallest source-free acceptance first, then real original source and physical GPU, and only then publish a status. Keep other agents' work staged/unstaged untouched and commit only owned paths; no whole-tree reset/stash.

**This handoff is implementation-ready design and documentation validation, not running native SSD inference.**
