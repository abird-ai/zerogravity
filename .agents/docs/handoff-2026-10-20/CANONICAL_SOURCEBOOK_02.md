# Canonical Sourcebook — volume 2

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md

Original UTF-8 bytes: 602982; SHA256 `8510266441c84e1588373d42ab2d43a6831c8e7cac001d8911ca76ffde305492`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: GPT-6
Co-authors: Human project owner and prior agents (as recorded in source/commit history)
Snapshot: 19 October 2026, Asia/Singapore
Source checkpoint: 3ec431a2df62bbcae5091a048f285801f151da7e

# Zerogravity — full comprehensive implementation and architecture handoff, 19 October 2026

> **CURRENT-STATE FIRST.** This document is a repository-grounded continuation and self-contained historical archive. The sections preceding the appendices describe the verified October 19 baseline. The embedded October 17 handoff and copied plan TODO snapshots are **DATED REFERENCE MATERIAL**, never additional live trackers. The latest user instruction, AGENTS.md, current source, actual tests, and the live .agents/plans/TODO.md and per-plan TODOs outrank historical status claims.

## 0. Executive takeover — current precise checkpoint

**Canonical repository:** /home/pvl/spaces/abird/src/zerogravity, exposed through the connected dotlink-l5 local-machine bridge. Run every repository command with that directory as the working directory.

**Inspected HEAD:** 3ec431a2df62bbcae5091a048f285801f151da7e — feat(gpu): qualify native batched GGUF quantized projections. Branch main was clean, local main was 9 commits ahead of local origin/main, no additional worktrees, no Cargo/rustc/serving process, and the current RTX 3060 Laptop GPU was idle. **No push or remote fetch occurred** during handoff inspection. Revalidate before implementing: source may have advanced further.

**Priority order explicitly selected by the owner:**
1. OpenAI-compatible native Rust serving and **genuine vLLM-level fused GPU continuous batching**, including correct parallel request/state/cancellation semantics and a comparable measured vLLM benchmark. CPU serving and resident CUDA HTTP are already implemented, but full-model fused GPU serving is NOT.
2. Complete a source-safe, physically admitted GPU/RAM/SSD streaming execution path on the SAME model/math/state fabric. An immutable, verified SSD byte/extent reader exists; actual trained SSD-streamed model inference does not.
3. Qwen3.8 original 27B, using authorized real checkpoint source, independent trained BF16/FLA numeric evidence, full logits and StateRoot, GPU/CPU and streaming.
4. Gemma4 E2B, then 26B-A4B and dense 31B, independently verified CPU then GPU and resource/codec gates. Qwen3.8 Flash-Next, multimodal, distributed and speculation follow their separate plans. The earlier conversational “Qwen 3.7 27B” spelling was superseded by the actually observed **Qwen3.8-27B** source family; do not silently bind a nonexistent or different revision.

**Latest physically accepted P12 facts:**
- Real NVIDIA RTX 3060 CUDA Qwen3.5 0.8B/2B resident model execution and bounded GPU HTTP streaming, private sessions and source sharing.
- One native GPU launch for each **isolated** multi-row B=2/4/8 projection, F32 and Q4_0/Q4_1/Q8_0/Q6_K source-codec kernels; exact bit comparison to accepted B=1 operator.
- Physical B=2/4/8 isolated elementwise/RMSNorm/residual/SiLU, GDN recurrent candidate, KV append and grouped QK/softmax/AV operators.
- Physical Qwen3.5 0.8B **resident cohort** B=2/4/8: exact complete 248,320-vocabulary logits and all 42 state domains **per independent session**, checked per-row GPU Ready and one fence-protected logical cohort publication, rollback/refusal, state lease reclamation. **The individual full-model GPU programs were still serialized.** B>1 logits parity of serial model programs is NOT fused full-model GPU-kernel parity.
- Real OpenAI CUDA HTTP B=2 one-token final prefill tail co-published 2 StateRoots with one checked cohort transaction; four model forward API steps, six successful quanta, no retained active sessions, **gpu_fused_batches=0**. Active one-of-two cancellation left the other request correct, retired state, and allowed replay.
- Last P12-B2 GGUF-codec integration accepted by signed 3ec431a; on real RTX 3060 Q4_0/Q4_1/Q8_0/Q6_K × B2/B4/B8 small deterministic source fixtures, bit-identical 128/256/512 projection elements and one physical multi-row projection launch per case. **Synthetic operator fixtures are not original trained 27B or fused full-model runtime.**

**Still NOT implemented:** a complete B>1 batch-shaped ResidentProgram/ForwardPlan execution graph; multi-request physically fused Qwen model inference with paged hybrid KV/GDN state, all decoder operations and device-token readout in one batched pipeline; enabling batch_mode=fused; an independent source-matched vLLM V1 benchmark; real SSD-streamed trained model; executable trained Qwen3.8 27B/Gemma4; finished live multimodal DAGs, speculative MTP or distributed execution. Keep these open until physical and numerical acceptance. A numerical/architecture scaffold never licenses a runtime capability.

**Current frozen check evidence:** the accepted 3ec431a P12-B2 and 08be9c6 cohort run logs report scripts/check.sh **1,147 Rust passed, 0 failed, 69 deliberately ignored over 241 suite runs**, 18 Python fixtures, release CLI/installer, benchmark self-test, shell/Nix formatting and CPU comparator gates. Separate approved trained Qwen3.5 0.8B/2B cache-direct and HF-import release routes each passed exact full logits/all 42 StateRoot domains with source/lease/mutation tests. CUDA B2/B4/B8 operator/cohort qualifications were separately run physically. These results are **at their respective commit/source**, not a new run performed by reading this handoff. See the exact test source, receipts and caveats below.

**Known environmental blocker:** the separately requested offline Nix **filtered source-integrity derivation** may not evaluate because the locked nixpkgs flake input is absent from the local store and connector SSL trust bundle cannot fetch it. Nix formatting passed; do not falsely call source-integrity a pass or a source-code failure. Do not fetch dependencies just to “fix” the handoff.

## 1. Authority, preservation and read order

1. User instructions, AGENTS.md, and the live tree determine current permission and scope. No automatic push/force-push/tag/release/deploy/destructive reset. Signed commits; stage only files this agent owns. Root README.md is user-authored and must remain byte-identical unless the user explicitly requests an edit. The completed/signed Git graph and actual physical test output outrank dated prose.
2. Read .agents/plans/README.md (plan workflow) then .agents/plans/TODO.md (one source of current Active/Icebox/Closed prioritization). Open every active plan's TODO.md and linked README/spec; use .agents/runs/<plan>-<agent>/TODO.md for in-progress agent-local checklists. This handoff is NOT a replacement tracker. Completed material grouped under signed commit in the plan, unchecked work remains live; avoid adding duplicate status ledgers.
3. Read this **current-state front section** and then the embedded **full Oct 17 historical handoff** in Appendix A for original design, decisions, all recovered 94 early nested handoff sources, previously lost/refactored records, historical failures and all proposal IDs. Read Appendix B for the as-of-HEAD full plan TODO snapshot if the live files have moved. Do not use the historical October 17 pause, old f197c9c HEAD, old draft worktrees, or old “unimplemented serve” assertions as current truth.
4. Read original master design: .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md, IMPLEMENTATION_ROADMAP.md, INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md, plus docs/WHITEPAPER.md and docs/ENGINEERING_SPECIFICATION.md. Source ownership, exact math, state and resource lifecycle must be reconciled against present Rust code and tests.
5. For current work read all of .agents/plans/serving-and-speculation/FUSED_GPU_BATCHING.md, README.md and TODO.md, and .agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md. Then inspect crates/zerogravity-gpu/src/{resident_program,resident_inference,resident_batch,batch_*}.rs; crates/zerogravity-cli/src/serve/; tools/serve-bench.py and tools/serve-compare.py. For SSD and 27B work, read their live plans and source/metadata contracts.
6. Inspect live HEAD/index/untracked/worktrees/processes/device state. Use narrowly qualified commands; a passing mock or schema probe never becomes physical inference truth. Preserve unrelated agents' work and the historical evidence boundaries.

**Important historical supersessions:** the Oct 17 handoff explicitly froze implementation at a user-requested PAUSE, but the owner resumed and changed priorities after Oct 17; that pause is not the current execution directive. The Oct 17 takeover referenced two abandoned SIDE worktrees and C3/C4 unfinished drafts; they were resolved by subsequent signed model-acquisition commits, and this inspection found only one clean main worktree. Do not resurrect obsolete branches or archive-state chores. The Oct 17 handoff linked a deleted TODO-SPEC.md and other now-retired files; .agents/plans/README.md plus AGENTS.md now own the live rules. Some old approved labels and early chronology are from a rewritten Git history; see Appendix A's exact recovery limits.

## 2. Original user vision and architecture chronology — beginning to present

### 2.1 What Zerogravity is

A single Rust-native, composable AI execution fabric and eventually one deployable binary to run **text, image, video, audio, real-time and combined multimodal** inference locally and across trusted heterogeneous devices/peers. A simple Ollama-like single-model CLI should coexist with explicit power-user YAML configuration, many model registrations, a router, OpenAI-like APIs and advanced performance scheduling. Model mathematics, hardware-native kernels, memory placement and physical resource ownership specialize independently without forking the semantic/runtime engine. Strong source correctness, cancellation/state safety and real benchmarks take precedence over marketing claims.

Core inspirations/prior art to study (not automatically copy code): llama.cpp/GGUF and portable CPU source math; vLLM continuous batching/paged attention; SGLang cache/scheduled programs; Ollama UX; CubeCL (actually vendored upstream license); gufo, strata/maya, dwarfstar4/ds4, and extreme SSD/sparse execution; heterogeneous TPU/NPU/AMD/NVIDIA research. Project-authored code is MIT; model licenses and vendored upstream licenses are independent. Attribution, exact upstream versions and source-file provenance are required.

### 2.2 Evolution, including accepted and superseded reasoning

| When / design version | Binding direction, retained decisions and limits |
| --- | --- |
| Oct 10 V1 | New Rust single binary and end-to-end architecture; specialization at kernels/math/scheduling/memory; dense, MoE, GDN/recurrent, diffusion, media and extreme models. Original ambitious targets Qwen and Gemma; source-specific model recipes and legal portable fallback. Design/research precedes implementation. |
| V2/V2.1 | Unify compiler/TaskProgram, dynamic source formats/quantization, host/GPU/NPU cooperative prefill and state; separate source semantics from derived compute encoding. GPU and NPU placement cannot silently alter source recurrence. |
| Oct 12 V3 | Neutral portable model/semantic/task/state/region/device/resource interfaces; design review rejected proliferation of permanent IRs, overly generic workflows and multiple state committers. |
| V4 | **Minimal five-owner spine**: immutable source semantics; immutable compiled execution plan; TaskCoordinator/session/root publication; global physical resource authority; physical execution/IO completion owner. RIR/KIR/CIR are transient lowering/command views, not independent engines. |
| V4.1 | Universal region backend: native kernel, captured command graph, AOT device graph, static dataflow or explicitly qualified managed device session; each must declare complete inputs/outputs/effects and actual completion. |
| V4.2 | Compute, immutable parameters, mutable state, and cache placement are separate axes; topology/peer/tunnels costed explicitly. Remote nodes produce private state candidates, not alternative root issuers. |
| V4.3 | ModelCatalog/source vs versioned Alias/Profile vs ResidentReplica vs TaskSession; one-command CLI and multi-model router/serving above the SAME Rust engine; no silent model promotion or per-token router. |
| Oct 14 V5 | Typed task values and assemblies for text, ASR/TTS, image, video, direct decision models, rational clocks, model-part provenance and whole-output state/barrier semantics. |
| Oct 15 V5.1 | Full realtime/live/duplex/AV controls, latency/QoS and media playback frontiers; verified assemblies, finite decisions, specialist codecs; distinct protocol/media/physical epochs. Most remains design-only. |
| Oct 12–17 accepted CPU/GPU | Qwen3.5 source-verified 0.8B/2B CPU and GPU qualified within distinct functional 8K vs numerically accurate 512-consumed-token scopes; physical lease/quarantine/cancel and independent sessions. Numerical long-context GDN drift remains unresolved; do not broaden acceptance. |
| Oct 16–18 model acquisition | Immutable pinned/cache-derived model artifacts, original HF companion closure, LFS/Xet and authenticated cloud/OCI/ModelScope providers, explicit filename/quant selection and source-bounded removals. Actual original 27B weights not downloaded. |
| Oct 17–18 pipelines | YAML-first typed DAG/compiler/extension/context-filter **scaffolds**, in-memory authorized snapshot owner, explicit opt-in model decision adapter, single-model + multimodel UX designs; actual TaskCoordinator routing/media execution deferred. |
| Oct 18 original Qwen | Owned safetensors shards, original Qwen3.8 CPU FP32 analytic proposals/full forward, source BF16 diagnostic stages, pinned numerical reference trace comparator and source-bundle identity; **no trained 27B logits, no original HF/FLA trained parity**. |
| Oct 19 OpenAI CPU serve | 35b9093: localhost bounded HTTP chat/completions/SSE + model list, model aliases sharing one immutable Qwen CPU source, request queues, strict parsing, cancellation, host admission. Verified two concurrent requests; no vLLM CPU goodput claim. |
| Oct 19 resident GPU serve | 9790fa2: original resident CUDA GPU model behind HTTP, independent sessions, fairness wave grouping/credits, shared greedy/text/SSE; real CUDA HTTP & cancellation. **Waves still submit serial full model graphs.** |
| Oct 19 native P12 kernels | 114bba5 F32 B2/4/8 projection; 2869f87 batched RMS/elementwise/GDN/KV/attention; 3ec431a quantized Q4_0/Q4_1/Q8_0/Q6_K B2/4/8 source projection. All isolated CUDA operator acceptance, not model fusion. |
| Oct 19 physical resident cohort | 08be9c6: B2/B4/B8 independent resident sessions, checked device Ready, one state/root publication barrier; source-real 248,320-logit and 42-domain parity and active HTTP B2. **Per-row forward still serialized.** |
| Present P0 frontier | Integrate physically accepted multi-row operator kernels into the ONE model ForwardPlan/ResidentProgram with row-major device buffers and paged hybrid state; prove source-real fully batched model GPU parity, then benchmark against an actually running independent vLLM. |

For complete original U-001…U-070, P-001…P-096, O-001…O-074 and early MM decisions, including nuanced user/proposal/assumption distinctions and supersessions, read the intact **October 17 handoff embedded in Appendix A**, especially “Complete retained decision register,” “Original vision,” and “Recovered original 94-source history.” Do not invent approvals where that historical material records only a proposal or open choice.

### 2.3 Latest owner decisions carried forward verbatim in substance

- Human-led architecture and priorities; AI-assisted/agent implementation; publishable engineering-first, succinct, honest docs.
- One Rust math/StateRoot/physical executor for all model modalities; composability rather than per-model subprocess servers.
- Use many independent subagents when useful but one main coordinator, serialized MAIN Cargo/test ownership and exact file scopes. Never overwrite other worktrees. No new subagents are implied to be available in every tool session.
- YAML, **not TOML**, for user pipeline and serving configs; simple direct CLI remains zero-config by default.
- Concurrent OpenAI-compatible serving/real GPU **fusion and vLLM comparison** now above new-model priority; source-safe SSD residency next; Qwen3.8 27B and Gemma4 follow. No arbitrary 16K/32K proof required before finishing earlier model gates; functional 8K, numerical 8K and resident GPU 512 are not interchangeable.
- Models/quant/license/codecs/commits must be pinned and verified before acquisition or physical use; no silent network download, model swap, tokenizer guess, CPU fallback after explicit GPU selection, or speculative untrained parity.
- XDG owned config/data/cache under $XDG_*_HOME/abird/zerogravity; root README protected. Git commits signed; no push/tag/release without explicit action approval.
- Never assume a local AMD Renoir/gfx90c GPU is launch-safe; it remains held after an observed hang/reset. A qualified gfx1150/gfx1151 on another host does not lift that hold. No remote hardware takeover, drivers, mounts or privilege changes without authorization.

## 3. System ownership and critical invariants (do not fork these)

**Semantic truth:** immutable verified TensorSource/model definition and source-specific geometry/arithmetic/codec/source license; per-model exact tokenizer and runtime parameters. SourceIdentity issued once; a path or readonly mmap is not ownership of immutable bytes. Source bundle identity includes verified index plus each learned shard's SHA256, length and basename. Source-fingerprint mismatches refuse numerical comparison.

**Compilation:** one SemanticProgram/ForwardPlan and guarded typed operation/state/parameter/value schedule. ExecutableBundle binding, preflight, codegen, math/codec specialization and transient lowering views are not new inference engines. Source mathematical operations are authoritative; helper numerical scans must not become independent token loops. Qwen3.8 original analytic diagnostics are separate from trained and production paths until accepted.

**Logical state:** SessionAuthority alone issues RootBinding and root epochs. TaskCoordinator prepares private candidate state (KV/GDN/recurrent, token/grammar/RNG/index/media frontiers) and promotes it only after domain compatibility, complete device execution/readout and causal identity checks. A draft/token event/Ready proposal is not already committed. A canceled or stale proposal MUST NOT publish any root. Cross-request B>1 transaction first validates all rows and physical Ready, then obtains all original RootTransitions before any infallible swap.

**Physical state and safety:** one ResourceArbiter/process-shared physical admission authority, physical queue, BufferLease and device generation. Count actual overlapping VRAM/UMA/system memory once, preserve OS/cgroup headroom and driver allocator reality. Asynchronous Future abandonment is not hardware cancellation: do not reuse buffers, state or staging while completion is unknown. Retain/fence/quarantine and require real terminal events/readback. Model load, token work, transfer staging and output have distinct budget classes.

**Serving:** one route registry/admission and one shared resident model per canonical original source. Per-request queues, deadline/backpressure/cancel/state ownership and safe token/finish/usage responses. Router alias/profile selection is cold control work, not per-token content rewriting. Sessions are sticky to compatible source/provider; no unsafe hot migration, cross-session mutable KV or false OpenAI API claims. CPU and GPU share text output/greedy contract; GPU-only requested backend must fail closed.

**Numerical parity:** exact B1 original source F32/GGUF integer/FMA/reduction, BF16 source casts, GDN decay/softplus/sigmoid/beta, IMRoPE, canonical signed zero/future padded state, QK/softmax/AV, full vocab and every state domain. Operator bit parity, whole-model internal parity, independently trained framework parity, functional context and performance SLO are different gates. No unqualified tolerance relaxation or “training quality” inference from top-20 logits.

**Distributed and media:** remote peers and media task services prepare private changes under one coordinator; independent transport/ACK/playback semantics are not GPU root commits. Hosted service adapter has separate authorization, privacy, cost/idempotency and Unknown states, never counterfeit native kernel status.

## 4. Actual crate/tool/source ownership map

| Owner/location | Implemented purpose / next boundaries |
| --- | --- |
| crates/zerogravity-core | Semantic acceleration contracts, ForwardPlan/PlanOperation, TaskCoordinator, SessionAuthority and StateRoot, SourceIdentity, physical ResourceArbiter/device events, media typed contracts, authorized ContextOwner/ContextReader/ContextSnapshot. |
| crates/zerogravity-artifacts | GGUF and safetensors source containers, immutable owned shards/whole-file digests, explicit pinned pulls/LFS/Xet/cloud acquisition, local HF cache import and source-only SSD extent byte/window reader. |
| crates/zerogravity-cpu | CPU quant/GGUF source arithmetic, independent oracle utilities, physical CPU source activation and projection. |
| crates/zerogravity-model-qwen35 | Original trained Qwen3.5 CPU text forward, tokenizer, GDN/hybrid/full-attention state math and session semantics, source exact low-bit operators. |
| crates/zerogravity-model-gemma4 | Gemma4 family/metadata/variant/MoE/PLE source contracts; **not** trained CPU/GPU inference. |
| crates/zerogravity | Application/runtime facade, direct CPU/cache and original Qwen3.8 analytic/BF16/trace diagnostics; not separate Python model server. |
| crates/zerogravity-gpu | Vendored-CubeCL physical providers, native CUDA/HIP/Vulkan operators, ResidentService/queue/leases/quarantine, ResidentProgram/ResidentModel/ResidentSession, batch geometry/operators and typed resident batch root barrier. |
| crates/zerogravity-cli | Native inspect/model catalog/run, opt-in explicit GPU direct run, YAML pipeline check, optional local HTTP OpenAI text/chat serve (CPU or accepted resident GPU), typed cancellation and metrics. |
| tools/gpu-qualify | Explicit physical GPU/profile/source/codec/abort/operator/session qualifications; never turn synthetic fixture pass into trained-model production claim. |
| scripts/native-gpu.py; scripts/check.sh | Mandatory isolated/offline execution and comprehensive frozen workspace quality gates. |
| scripts/cuda-nix-include-view.sh | Symlinks already-installed split CUDA runtime/CCCL headers into a temporary CUDA_PATH view; does not acquire SDK/change host driver. |
| tools/serve-bench.py; tools/serve-compare.py | Reproducible bounded SSE load and pairwise service comparison/test self-check; no vLLM runtime provision and no equivalence proof by printing a number. |
| .agents/plans/serving-and-speculation/{FUSED_GPU_BATCHING.md,TODO.md} | Current P0 exact P12 batch execution and P22 vLLM acceptance tasks; preserve incomplete P12. |
| .agents/plans/{model-acquisition,model-execution,pipeline-composition,memory-and-streaming} | Historical and current exact owner plans; see sections below and Appendix B. |
| docs/{WHITEPAPER,ENGINEERING_SPECIFICATION}.md | Public publishable argument and engineering requirements, not complete executable capability. |
| AGENTS.md; .agents/plans/{README,TODO}.md | Current ownership, safety, process, priority and tracker source. |

**Important code entrypoints for next agent:** crates/zerogravity-gpu/src/resident_program.rs (currently B1 launch_impl), resident_inference.rs (source/model physical service, private GPU sessions and cohort prepare/commit), resident_batch.rs (validated row identity/stride/frontier), batch_projection.rs, batch_quant_projection.rs, batch_elementwise.rs, batch_recurrent.rs, batch_kv.rs, batch_attention.rs (B>1 CUDA operator fragments); crates/zerogravity-cli/src/serve/{gpu_batch,gpu_engine,http,engine,mod}.rs; crates/zerogravity-core/src/accelerator/program.rs and session_root.rs.

## 5. Implementation status: exact claims versus non-claims

| Capability | Current established evidence | Still unqualified |
| --- | --- | --- |
| Qwen3.5 0.8B/2B CPU GGUF | Real local source model run, tokenizer, basic Qwen decode, functional up-to-8K accepted tests and source/reference state checks. | Independent original long-context 8K GDN drift; full 16/32K; optimized throughput and all sampling/API options. |
| Qwen3.5 resident CUDA | Physically qualified specific RTX3060 profiles, actual full-model GPU, bounded 512 consumed tokens, private physical StateRoot and device-token greedy continuation; prior 4060/AMD qualified profiles are documented separately. | Different GPUs, longer contexts, quant/backend combinations without source/physical proof; no broad performance claim. |
| GPU operator B>1 | Physical isolated B2/B4/B8 fused F32 and GGUF Q4_0/Q4_1/Q8_0/Q6_K projections; RMS/residual/Silu; GDN; causal KV; attention. | Whole-model batch-shaped parameters/activations/all-state GPU ForwardPlan, batch-native last-layer logits/device-greedy, paged KV and measured vLLM performance. |
| Resident multi-session cohort | Source-real Qwen3.5 .8B B2/B4/B8 exact full logits and 42 state domains per row, one checked cross-root publication, rollback and leased-state cleanup. | Full model graph still serial **per row**. Do not count checked logical root transaction as kernel fusion. |
| Native OpenAI HTTP | Localhost GET /health, GET /v1/models, POST /v1/completions, POST /v1/chat/completions; greedy only, typed JSON/SSE, model alias sharing, queues/limits/cancel, optional bearer. CPU and selected resident CUDA physical integration. | All other OpenAI/Responses/Anthropic APIs, tools/grammar/logprobs/embeddings, real auth/TLS outside loopback, multi-node serving, fused vLLM goodput. |
| SSD source bytes | Independently pinned SHA256 strict extent index, source descriptor, ≤4 MiB chunks / ≤32 MiB read / ≤32 chunks, resource-admitted coalesced reads/finite StagedSlice, sparse >5GiB metadata tests. | Tensor-to-extent model bindings, async GPU transfer/BufferLease/effect completion, cold experts, eviction/ABA, trained resident-versus-SSD parity. |
| Original Qwen3.8 27B | Exact public metadata-only source revision, audited config/shard index geometry, owned-shard mechanics, analytical CPU/BF16 original ops, pinned reference-comparison contract. | Authorized real 27B learned shard bytes on sufficient host, original HF/FLA independently trained stage/logit parity, actual CPU/GPU/SSD inference and performance. |
| Gemma4 E2B/26B-A4B/31B | Typed source family/variant contracts, source metadata and candidate revisions, analytical MoE/PLE expectations. | Trained CPU/GPU executable kernels and checkpoints, full logits/StateRoot and throughput. |
| YAML model DAGs and context | YAML compiler/graph validation, typed stage/extension schemas; immutable source-grant ContextOwner/Reader/DecisionAdapter test boundaries and examples. | Actual pipeline.run/serve real model/media TaskCoordinator, authenticated external context providers, model-backed decision/ASR/TTS and live streaming. |
| Other models/modalities/distribution | Normative contracts, model/provider/source catalog, research/spec plans and selected synthetic fixtures. | Native trained models, first-class realtime audio/video/image inference, true remote placement and distributed root reconciliation. |
| Speculative decoding and optimization | Plans/research for runtime n-gram/suffix, target authoritative MTP/drafters, Qwen Flash-Next learned extra heads; fair wave scheduler, backend source/performance baseline. | Real accepted speculative correction across all state domains, paged KV, joint optimization, source-real SLO-goodput comparison. |

## 6. Current P0 — OpenAI, real GPU batching and vLLM acceptance

### 6.1 Working server

From root (local, approved GGUF + config.json + tokenizer.json already present):

    CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
      cargo run --release --locked --offline \
      -p zerogravity-cli --features serve -- \
      serve --config examples/serving/qwen-cpu.yaml

Resident NVIDIA CUDA (on a permitted, qualified host with compatible CUDA_PATH, NVRTC and NVIDIA driver libraries):

    CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
      cargo run --release --locked --offline \
      -p zerogravity-cli --features serve,cuda -- \
      serve --config examples/serving/qwen-cuda.yaml

For Nix split CUDA headers use scripts/cuda-nix-include-view.sh with already-installed CUDART_INCLUDE and CCCL_INCLUDE paths, set CUDA_PATH to its printed path, add compatible NVRTC + driver to LD_LIBRARY_PATH. Local smoke used existing CUDA 12.9 runtime/CCCL/NVRTC; without an include root CubeCL compiler previously panicked/retried and initial physical proof was invalid. Retry only after validating the toolkit and hardware. NVIDIA Vulkan library/device enumeration failed for this local CubeCL process despite vulkaninfo recognizing NVIDIA; do not assert this connector accepted Vulkan physical serving. AMD Renoir/gfx90c is explicitly held.

    curl http://127.0.0.1:8000/v1/models
    curl http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"Hello!"}],"temperature":0,"max_tokens":16}'

Accept greedy-only temperature=0, top_p=1, n=1, zero penalties, bounded max_tokens/max_completion_tokens, text messages only. Unsupported parameters, JSON duplicates, unsupported roles/tool content, synthetic model IDs and role delimiter injection refuse. On non-loopback bind fail closed: place an explicitly authorized TLS/auth reverse proxy in front of localhost; bearer alone is not TLS.

### 6.2 GPU status and numerics

Use [P12 native fused GPU acceptance] in .agents/plans/serving-and-speculation/FUSED_GPU_BATCHING.md. Physical runtime currently uses source-compatible **scheduling waves** (B2/4/8) but submits each full ResidentProgram separately. The checked multi-root publication barrier is functional and preserves independent per-session RootBinding state, but is not a single native B>1 model graph. Health correctly reports **gpu_fused_batches: 0**; explicit batch_mode=fused must refuse until tested.

RTX3060 0.8B debug HTTP source-real observations (19 Oct): two requests with aliases of same source produced identical two-token completions; an accepted test used one B2 cohort publication and four GPU forward API steps, six completed GPU quanta, zero failed quanta and zero retained sessions. Approx ~16s wall for a two-request debug test. The standard-library SSE load tool measured two offered/two complete and four usage-reported tokens, 20.02s debug wall / ~0.20 usage tokens/s and p95 first text ~19.78s. **Diagnostic debug-build numbers, not release performance, not fused-model performance, not a vLLM comparison, not true per-token ITL.**

The standalone native B>1 operator kernels do physically fuse at their own microkernel granularity; the full model graph has not been transformed to call them. No sampling/API performance superiority may be inferred.

### 6.3 Concrete unimplemented integration acceptance steps — do these in order

1. Read ResidentProgram::launch_impl and all PlanOperation variants. Build one typed row-major state/value/parameter ABI for a compatible source/codec/plan/profile, ensuring each lane's original RootBinding, epoch/frontier, source and physical candidate handles remain protected. No shadow state issuer or duplicate Qwen mathematical loop. Use the existing source/RowBinding/Ready primitives.
2. Plan full-model batched memory: immutable source once; private per-row KV/recurrence ping-pong candidates; valid token masks and lengths; status/output scratch; finite provider/physical admission. Reject overlapping writes and source/epoch/codec/profile mismatches before any physical submission. Maintain direct B1 no-copy fast path.
3. Reuse the qualified B>1 projection/quantized, RMS/elementwise, RoPE/GDN, KV and QK/softmax/AV kernels through the SAME ForwardPlan schedule. Add missing batched embed/norm/FFN/head/greedy and remaining operations with exact original rounding and GPU completion. Count **actual device kernel launches**, not scheduling waves or forward API calls.
4. Validate source-real trained Qwen3.5 B1→B2→B4→B8 complete full-vocabulary logits + **every** state domain and independent root/Ready receipts, uneven input histories/positions, EOS, padded tails/signed zeros, mid-batch bad row/rollback, canceled Future and incomplete physical event, unknown device completion quarantine and no cross-session leakage. Only thereafter mark batch-mode=fused as executable; per-GPU backend separately qualifies.
5. Add physically page-allocated hybrid KV/GDN COW, source-exact prefix compatibility and real fair continuous prefill/decode. Source-real vLLM tests require comparable quant/checkpoint/tokenizer/sampling and equal RAM/GPU budgets.
6. Run independent vLLM V1 on the **same** device with pinned versions and source-compatible model/quant profiles; compare offered/admitted/committed/canceled work, p50/p95 TTFT, true token-level ITL, SLO goodput, throughput, page footprints, GPU dispatches, fairness and cold/warm repeated ABBA/BAAB workloads. If vLLM cannot load the identical source representation, explicitly label incomparable shapes rather than invent a speedup.

The host currently lacked a vllm Python package / vllm executable when inspected. tools/serve-compare.py passed source-free self-tests; it does not install or deploy a reference server. The owner requested a benchmark, but a reference installation/endpoint and model-source equivalence are independent prerequisites; do not claim completed comparison without them.

## 7. Current P1 — GPU/RAM/SSD source-residency foundation

Design authority: .agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md, .agents/research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md, .agents/plans/memory-and-streaming/{README,TODO}.md.

Implemented: crates/zerogravity-artifacts/src/ssd_extent.rs supplies separately SHA256-pinned strict contiguous chunk manifest over an opened descriptor; per-chunk whole-byte hashing after positional read, coalesced ≤32MiB requests/≤4MiB chunk/≤32 segments, per-read ResourceArbiter staging lease owned by StagedExtents/StagedSlice; true file mutation/truncation and sparse multi-GiB metadata tests. The manifest's declared whole-file checksum is **not** automatically whole-file verification. This is not a trained TensorSource/SSD inference engine.

Next: original tensor/shard offset directory → stable verified virtual TensorSource and source/quant/layout views; physically admitted host staging + async IO + H2D transfer tied to actual BufferLease/Ready before release; bounded multi-size caches, COW/prefix KV, cache generation ABA/queued readers, IO fairness under GPU critical-path deadlines, optimized no-SSD-hit resident fast path. Run SSD-01…SSD-32 and independent trained resident-vs-streamed full logits/StateRoot including short/corrupt IO, cancellation/fence, memory pressure, cold/warm TTFT/ITL and actual IO amplification. No SSD speed claim follows from source-byte IO.

## 8. Current P2 — Qwen3.8, Gemma4 and independent numerical evidence

**Qwen source order:** accepted Qwen3.5 .8B and 2B source-model CPU/GPU tests; measure full real 2B CPU/GPU performance and numerical boundaries; use already-implemented original source-owned safetensors bridge for Qwen3.8 27B. Exact public **metadata-only** HF observation as of 18 Oct: Qwen3.8-27B commit 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, 18 learned shards, 55,586,113,293 selected bytes. These are selected source bytes, not resident bytes, admissible runtime memory or model downloads. Current locally approved 0.8B/2B GGUF files are present under target/models/qwen35-{0.8b,2b}; do not conflate them with HF 27B.

Original Qwen3.8 source-specific math must preserve actual Q/K/V head axes and interleaving, source RoPE and gate transforms, non-GDN +1 RMSNorm, GDN SiLU/FP32 FLA recurrence, BF16 cast points and softmax/value rounding, QSA/GDN hybrid session state. Current FP32/BF16 original-Qwen analytical and source-trace comparator modules are diagnostic; the strict pinned trace JSON can assert “trained-framework” but that tag does not establish independent capture. Acquire independently generated original HF/FLA reference numerics and exact source fingerprints on authorized hardware before claiming trained logits/StateRoot. Model executable-source admission currently has a 4GiB whole-source limit; synthetic >51GiB transfer planning does not remove the execution cap.

**Gemma4:** source metadata-only E2B-it revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 (1 unsharded safetensors, 10,278,848,001 selected bytes); 26B-A4B-it revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 (2 shards, 51,644,338,194 selected bytes). E2B differs via PLE/doubled FFN, 26B via MoE/raw K=V and distinct source expert routing, dense 31B separately. Family contracts, vocab/tokenizer/processor and signed model source must be independently matched; no Gemma trained inference is currently enabled.

**Flash-Next/advanced:** Qwen3.8 Flash-Next/QSA/GDN/MoE, separately learned n-gram embedding banks (~51B) and trained 4B MTP heads, sparse expert placement, Q2/~3.5bpw/Q4 exact source forms, draft/MTP/suffix/n-gram speculation must be distinctly source-verified. Runtime n-gram speculative lookup is not a learned n-gram bank; speculative candidates cannot publish independent roots or alter target RNG distribution. Gemma/Qwen Flash chronological order and other models remain per live model-execution TODO.

**CPU accuracy warning:** Qwen3.5 real CPU functional 8K gates passed, but independent native llama.cpp numerical 8K GDN divergence was observed (first layer 4 for 0.8B; layer 1 for 2B in the retained investigation), and known 2B signed-zero/ubatch differences remain open. Native top-20 logprob errors around 0.20372924 / 0.19678696 at 8K do not establish full-vocab agreement. Independent 16K/32K reference work deferred. Respect existing pinned 0.45 shared top-20/0.05 top-token functional thresholds without calling them strict source equality.

## 9. Acquisition and immutable source provenance

Current .agents/plans/model-acquisition/{README,TODO}.md describes completed explicit source operations: pinned local HF repo/revision/filename/quant-family candidate discovery; metadata-only HEAD/candidate observations; required index-first original companion closure; exact hashed immutably owned GGUF/source files; dedupe/cache, protected removal and explicit local HF import, cross-filesystem; bounded auxiliary inputs; native HF LFS/Xet with required verified indexes; authenticated SHA/length cloud HTTPS/S3/GCS/Azure/OCI/ModelScope providers; finite capped reconstruction and cooperative cache admission. XDG root under abird/zerogravity. Execution from cache must preserve direct-source exact logits/all state identity. C3 Unicode full-path filters and C4 required Xet reconstruction, previously PAUSED in Oct 17 handoff, were later implemented/verified in their commit groups. Retained paused designs at .agents/docs/handoff-2026-10-17 remain historical reference, not active detached work.

Future providers/credentials, source upgrades, portable snapshot import, automatic advisory metadata, and real large-model learned bytes remain explicit live TODOs. Neither public source metadata discovery nor a synthetic large transfer fixture authorizes downloading or executing large checkpoint payloads. Read the lengthy acquisition README before changing pull/cache ownership.

## 10. YAML multimodal pipeline and authorized context scaffolding

Current source compiles YAML-first typed pipeline graphs, model/extension registry and configurable source filters. Representative inputs:
- examples/pipelines/text-model.yaml
- examples/pipelines/voice-assistant.yaml
- examples/pipelines/contextual-voice-assistant.yaml
- examples/pipelines/contextual-voice-filtered.yaml
- examples/pipelines/transcribe-diarize.yaml
- examples/pipelines/audio-codec.yaml
- examples/pipelines/decision.yaml
- examples/pipelines/image-generation.yaml
- examples/pipelines/image-latent-assembly.yaml
- examples/pipelines/video-stream.yaml

Current pipeline check is source-free structural validation; pipeline run is deliberately **not implemented**. Intended architecture: a model can serve direct chat and be a component of an ASR→speaker-tagged transcript→LLM→TTS graph with one ResidentModel source owner, TaskDriver, StateRoot and physical arbiter; typed input/output ports, clock/timebase, backpressure, media output commit, runtime routing and per-stage quality gates apply.

Authorized previous-chat context: explicit optional ON/OFF source selection, trusted app-owned ContextOwner, authenticated ContextReader bound to exact tenant/app/session and source grants, immutable prior committed complete-turn records with bounded 4KiB…1MiB history window, optimistic source epochs and final freshness recheck. System/developer prompts owned by the application, prior user/assistant, optional tool results and explicitly granted shared/cross-user records are eligible **when authorized**; never invent access to hidden ungranted prompts or private records. Include/exclude and typed decision filters only see already-authorized records; NanoJev “decision model” adapter is a callback contract, not trained model inference. No actual external conversation provider, ASR context A/B, consent or source-real media execution has passed.

Multimodal DMM and open OMM decisions and model family source dossiers (Whisper/Canary/Parakeet, Qwen TTS/Kokoro, codec RVQ/duplex, Qwen Image/FLUX, Wan/LTX/other video, finite-choice schema, ternary codecs) remain in .agents/research/multimodal and the MM V5/V5.1 source contracts. Do not start new modality runtimes while P0 fused GPU serving and P1 SSD/27B remain higher owner priorities unless a bounded prerequisite explicitly blocks them.

## 11. Hardware, physical safety, failures and lessons

Target fleet is the current Ryzen 7 5800H + RTX 3060 Laptop 6GiB + held Renoir gfx90c, remote pvl-a1 Ryzen AI 9 HX370 (RTX 4060 + 890M gfx1150) and pvl-x2 Ryzen AI MAX+395 (8060S gfx1151). Historical catalog calls this three CPUs / five physical GPUs. Recheck each host; nominal shared memory/GTT is not free physical RAM. Remote machine permissions, models and SDKs are separate per host. GPU CUDA acceptance here is the RTX3060; inherited HIP/ROCm tests elsewhere are distinct. CUDA uses existing driver/NVRTC/CCCL; local Vulkan CubeCL failed to find usable NVIDIA discrete despite vulkaninfo seeing it. No AMD held-device launch or recovery.

Remember historical numerical mistakes and observed regressions: signed zero; inaccurate FMA simulation, reciprocal double-rounding, GDN beta/A_log order, source Q4/Q6 scale grouping, QK padding/attention reduction, RoPE iterative phase, runtime activation conversion and state publication. A logit rank match, synthetic oracle, or successful kernel compilation can hide full numerical drift. Source-specific CPU and physical GPU exact-check suites established the accepted kernels.

Remember historical build/process mistakes: a connector RPC can be interrupted after tests succeed; always inspect actual log, stored exit receipt and active compiler/driver processes. Verify no other MAIN Cargo owner. Rust source copied preserving old mtimes can fool incremental build; use fresh mtimes when merging branch copies. Source-integrity test may fail *environmentally* due missing pinned Nix input/SSL CA trust; do not call it a passing check. Old dated commits from rewritten history can have artificial future timestamps and missing objects; treat retained read catalog and actual reachable HEAD as authority. Do not reset to a historical handoff checkpoint, resurrect worktrees or infer commit ancestry from an archive.

## 12. Practical verified command and test runbook

### Safe revalidation

    cd /home/pvl/spaces/abird/src/zerogravity
    git status --short --branch
    git log -12 --oneline
    git worktree list
    git diff --check
    ps -eo pid,comm,args | grep -E 'cargo|rustc|zerogravity' | grep -v grep
    sed -n '1,180p' AGENTS.md
    sed -n '1,140p' .agents/plans/README.md
    cat .agents/plans/TODO.md

### Full isolated frozen gate

    CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
      python3 scripts/native-gpu.py -- bash scripts/check.sh

Use its actual exit=0 and exact Rust suite/failure/ignored totals; don't infer total from prior source after a new change.

### Strict CUDA/CPU/Vulkan lint

    CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
      python3 scripts/native-gpu.py -- \
      cargo clippy --locked --offline \
        -p zerogravity-core -p zerogravity-gpu -p zerogravity-cli \
        --no-default-features --features zerogravity-cli/serve,zerogravity-cli/cuda \
        --all-targets -- -D warnings

Repeat source-appropriate default/CPU, CUDA/qualification and Vulkan variants as necessary. CPU/GPU tests have opt-in feature guards; default frozen tests do **not** prove new GPU hardware behavior.

### Approved independent trained Qwen3.5 regression routes

    CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
      python3 scripts/native-gpu.py -- \
      cargo test --release --locked --offline -p zerogravity \
        --features cache --test cache_prepare \
        approved_cached_models_match_direct_source_exactly \
        -- --ignored --exact --nocapture

    CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
      python3 scripts/native-gpu.py -- \
      cargo test --release --locked --offline -p zerogravity \
        --features cache --test cache_prepare \
        approved_hf_import_models_match_direct_source_exactly \
        -- --ignored --exact --nocapture

Each route covers both approved 0.8B/2B original models; keep these separately verified, not abbreviated into one invocation.

### Physical CUDA P12 acceptance (only on permitted, healthy card)

Prerequisites: approved unchanged local GGUF/config/tokenizer, CUDA_PATH including installed runtime and CCCL headers, compatible NVRTC/driver libraries, correct provider/device limits, actual available memory headroom. Use explicit device physical tests, no AMD launch.

    python3 scripts/native-gpu.py -- \
      cargo run --locked --offline \
      -p zerogravity-gpu-qualify --features qualification,cuda -- \
      batch-projection-cuda

    python3 scripts/native-gpu.py -- \
      cargo run --locked --offline \
      -p zerogravity-gpu-qualify --features qualification,cuda -- \
      batch-elementwise-cuda

    python3 scripts/native-gpu.py -- \
      cargo run --locked --offline \
      -p zerogravity-gpu-qualify --features qualification,cuda -- \
      batch-recurrent-cuda

    python3 scripts/native-gpu.py -- \
      cargo run --locked --offline \
      -p zerogravity-gpu-qualify --features qualification,cuda -- \
      batch-kv-cuda

    python3 scripts/native-gpu.py -- \
      cargo run --locked --offline \
      -p zerogravity-gpu-qualify --features qualification,cuda -- \
      batch-attention-cuda

    python3 scripts/native-gpu.py -- \
      cargo run --locked --offline \
      -p zerogravity-gpu-qualify --features qualification,cuda -- \
      batch-q8-projection-cuda

The final command's retained name covers Q8_0/Q4_1/Q6_K/Q4_0 B2/4/8 quantized projections. All use isolated operator fixtures; they do not establish end-to-end fused model execution.

    python3 scripts/native-gpu.py -- \
      cargo run --locked --offline \
      -p zerogravity-gpu-qualify --features qualification,cuda -- \
      resident-cohort-cuda target/models/qwen35-0.8b/model.gguf \
      4096 5120 8

Final arguments: provider MiB, physical device cap MiB, cohort rows (2/4/8). This runs actual original Qwen3.5 source/GPU session proposals but individual full-model forward graphs remain serialized. Smaller memory profiles can refuse legitimate B8; don't defeat physical admission. The user/Nix test isolation wrapper is mandatory.

### Serving/vLLM benchmark contracts

    python3 tools/serve-bench.py --self-test
    python3 tools/serve-compare.py --self-test

Run an actual comparison ONLY with two live, source-compatible OpenAI endpoints and an accepted fused Zerogravity batch-native graph. Record output and reference provenance. Do not infer vLLM performance from synthetic self-test, debug SSE chunk intervals, or a different model codec. If the reference service is unavailable, mark comparison blocked rather than install it without approval.

### Nix source integrity (separate from formatting)

    nix --extra-experimental-features 'nix-command flakes' \
      build --offline --no-link .#checks.x86_64-linux.source-integrity

This has been blocked by a missing locked nixpkgs store input plus inaccessible TLS trust in the connected environment. Never claim it passed at current head without a real successful derivation. The frozen scripts/check.sh Alejandra formatting check is independent.

## 13. Test evidence and milestone provenance after the Oct 17 handoff

| Signed commit | Delivered change and scope, with source artifact |
| --- | --- |
| 04363c3, 8195c18, f56b574, d5aa5cb, fee0bb2 | Full-path Unicode filters, required native HF Xet index verification, AWS FULL_OBJECT SHA256, original v1 model companion closure and Qwen/Gemma source metadata. |
| 4bf624e, ac44e83, f28a62c, f0746d1 | YAML-first typed multi-model/pipeline scaffolds, opt-in authorized historical context, ordered source-checked filters, immutable context snapshots and typed decisions. No media runtime. |
| 93a5622, b796a2e, b11e58c, 6329cff, e43ba4c | Original safetensors source owner, Qwen3.8 original source-bound CPU mathematical proposals, existing-state bridge and full-model FP32 analytic token. No trained 27B weights. |
| aaf9ba6, 1c0bb41 | Original BF16 stage diagnostics, independent NumPy bit-vectors, exact source-bundle SHA256 and bounded pinned external trace comparator with typed numerical mismatch and provenance claim caution. |
| 35b9093 | Functional native localhost CPU OpenAI chat/completions/SSE, bounded parallel worker queues, model source aliases, physical host budget, cancellation; verified source-only SSD extent reader. |
| 9790fa2 | Real CUDA resident Qwen OpenAI GPU HTTP, independently charged sessions, physical greedy token feedback, wave scheduler, live cancellation, truthful zero GPU model batch fusion. |
| 114bba5 | F32 source multi-row B2/4/8 isolated CUDA projection kernel, bit-identical to accepted GPU B1. |
| 2869f87 | Isolated multi-row RMS/elementwise/GDN/KV/QK/softmax/AV CUDA operators, source identity/shape/refusal and separate benchmark harness. |
| 08be9c6 | B2/B4/B8 resident source-real multi-session Ready + checked collective logical root transaction, GPU HTTP eligible final one-token prefill cohort publication and cancellation. |
| 3ec431a | Native source-quantized Q4_0/Q4_1/Q8_0/Q6_K multi-row GPU projections with exact B1 source arithmetic. **Current inspected HEAD.** |

All details exist in current Git diffs, owning plan TODOs and the legacy historical archive in Appendix A. Physical tests must be rerun when changes affect kernels/owner lifetimes. “x passed at a previous commit” is not automatic new-HEAD acceptance.

### Current local acceptance receipt index

The following are local, normally gitignored .agents/runs checkpoints. They are helpful to verify actual outcomes, but are not a newly mandatory evidence-file workflow:

- .agents/runs/p12-q8-fusion-2026-10-19/{all-four-codecs-final.log,full-gate.log,qwen-direct.log,qwen-hf-import.log,TODO.md}; each approved test exit was 0. CUDA B2/B4/B8 × 4 codecs passed, separate full frozen gate 1,147/0/69.
- .agents/runs/p12-resident-cohort-2026-10-19/{physical-cohort-b2.log,physical-cohort-b4.log,physical-cohort-b8.log,server-cohort-result.log,server-cohort-cancel-result.log,full-gate.log,qwen-direct.log,qwen-hf-import.log,clippy-cpu.log,clippy-cuda.log,clippy-vulkan.log,TODO.md}; accepted source-real B2/4/8 and root rollback, HTTP one cohort publication B2, survivor/cancellation, 1,147/0/69.
- .agents/runs/gpu-serving-batching-2026-10-19/ contains earlier operator, physical CUDA HTTP/debug SSE, wave, bench and build investigations; check exact .exit before drawing conclusions.
- .agents/runs/serving-ssd-2026-10-19/ contains initial native CPU router/SSD check and benchmark results; earlier interrupted runs should not be promoted to complete.
- .agents/runs/model-execution-qwen38-reference-2026-10-18/ and sibling execution dirs hold original source-bound numeric diagnostic development. See owning model plan.
- The source of truth for user-facing status is current Git code/tests/plan TODOs; nontracked evidence may disappear between environments.

## 14. Priority execution program for next agent

**P0 NEXT — native full-model fused CUDA execution, not more synthetic kernel-only proofs:** start from 3ec431a. Inspect B>1 microkernels and their exact source B1 code, current resident-program PlanNode/value memory mapping and state binding. Design a bounded B2 end-to-end lowering that uses **one physical multi-row kernel per compatible stage** under a single ResidentProgram, while keeping a distinct StateRoot/Ready for each lane. Handle mixed source GGUF quant rows and GDN/full attention state; keep progress gates separate for F32 vs quantized models. Prove original Qwen3.5 0.8B B2 complete trained logits and all42 state domains vs B1/CPU, bad lane/rollback, true GPU dispatch reduction and buffer finalization. Then B4/B8 and 2B; do not activate HTTP fused option earlier. Document measurable occupancy and realistic limitations.

**P0 AFTER FULL FUSION — measured vLLM:** obtain a legitimate independent compatible vLLM instance/endpoint and matching model/source layout, pin reference version and sampling, run repeated offered-load/concurrency with true token ITL/TTFT/SLO throughput p95, fairness, cache/VRAM/RAM/energy/cold/warm. Record actual wins and losses without using incomparable codec claims. Benchmark harness alone is not measurement.

**P1 — SSD source model integration:** bind immutable original HF shards/tensors and GPU source buffers to already verified SSD extent reader with async readiness and physically valid eviction; prove trained resident-versus-streamed exact original logits/StateRoot and IO/resource safety. Large >4GiB source plan remains blocked on authorized checkpoint/payload and actual host memory.

**P2 — 27B and Gemma4:** source-authorized original Qwen3.8 27B trained checkpoint and independent HF/FLA trace, then Gemma4 E2B/26B-A4B/31B; same mathematical/TaskDriver/StateRoot/physical owners, no parallel second inference engine. Other models and multimodal after owner promotes their plans.

**Independent nonblocking streams, if separate agents authorized:** test/benchmark review, source numerical investigation, doc/design reconciliation and P12 GPU graph implementation with disjoint file ownership. One MAIN Cargo owner, isolated staged commits and no blind simultaneous edits to shared architecture plans.

## 15. What the new agent must NOT assume

- The root README implementation table is older than signed October 19 GPU and serving work. Do not change its bytes just for status; use live code and plan TODOs.
- The Oct 17 handoff says implementation PAUSED and references unfinished SIDE trees. Those were facts of Oct 17 only; execution was explicitly resumed and later merges changed the repository.
- The current main is 9 commits ahead of the local origin/main; not remotely published by this handoff. A prior signed commit on main is not authorization to push.
- An original 27B or Gemma metadata-only commit/hash proves discovery and file sizes, not trained checkpoint acquisition, ownership or execution.
- SSD source staging can safely materialize bytes, not inference-ready GPU weights or paged KV state.
- A GPU scheduling wave, a full-model forward API call, an atomic B8 StateRoot publication, and a single physically fused B8 CUDA kernel are **four different observations**. Never name one as another.
- Exactly passed internal CPU/GPU logits and StateRoot against a previously accepted local reference is not a true independent HF-trained source/model release gate for an otherwise unqualified model.
- There is no locally installed running vLLM service; do not fabricate comparative throughput.
- Nix offline source-integrity unavailable input is not proof of a failing code derivation; format/test checks are independent.
- Source/driver/device privilege, model downloads, GPU resets, destructive Git rewrites and remote operations remain approval-sensitive.
- This document is NOT an excuse to resume resolved C3/C4 paused drafts, older generated evidence archives or a second state router.

## 16. Comprehensive reading and provenance catalogue

The complete historical October 10–17 design/decision/source recovery is reproduced verbatim in **Appendix A**. The current active plan TODOs are reproduced verbatim, as a dated checkpoint, in **Appendix B**. The generated complete path inventory in **Appendix C** enumerates every currently surviving .agents/docs, .agents/research, .agents/prior-art, .agents/plans and docs Markdown source with its title/length, plus all individual active plan URLs. Therefore an implementer can fetch the authoritative complete designs rather than rely on a compressed summary or accidentally skip an original research decision. Appendix D includes the copy-ready next-agent prompt. Every appendix is a historical snapshot at the named HEAD; the **live source and TODOs** remain the implementation truth.

**Canonical high-value reading paths, in order:**

- AGENTS.md; .agents/plans/{README,TODO}.md; .agents/plans/serving-and-speculation/{README,TODO,FUSED_GPU_BATCHING}.md
- .agents/plans/{ZEROGRAVITY_MASTER_PLAN_2026-10-10,IMPLEMENTATION_ROADMAP,INTEGRATED_PHASE_0_AND_DELIVERY_GATES,EXPERIMENT_PROGRAM}.md
- docs/{WHITEPAPER,ENGINEERING_SPECIFICATION,DEVELOPMENT,PUBLICATION_AND_REVIEW_CHECKLIST,REFERENCES}.md
- .agents/docs/{MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4,TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4,SEMANTIC_IR_AND_COMPILER,INVARIANTS_AND_FAILURE_SEMANTICS,STATE_CACHE_AND_TRANSACTIONS}.md
- .agents/docs/{ENGINE_CORE_EXECUTION_AND_PROTOCOL,ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT,DEVICE_KERNEL_AND_CODEGEN,SCHEDULER_BATCHING_AND_SPECULATION,JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15}.md
- .agents/docs/{MEMORY_RESIDENCY_AND_IO,UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15,MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT,ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT}.md
- .agents/docs/{MODEL_SEMANTICS_AND_COMPATIBILITY,FORMATS_QUANTIZATION_AND_LOADING,NUMERICS_QUALITY_AND_BENCHMARKS,VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15}.md
- .agents/plans/{model-execution,model-acquisition,memory-and-streaming,pipeline-composition}/README.md plus each TODO.md and source-specific acceptance.
- .agents/research/{QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16,GEMMA4_FAMILY_CONTRACTS_2026-10-16,VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15,SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15}.md
- .agents/docs/{UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15,REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15,DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT,REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL}.md

**Appendix ordering:** A = exact old complete historical handoff; B = current plan TODO snapshots (all plan families, not only P0); C = document and run/path inventory; D = exact starting prompt. The next agent should **read** canonical sources and not modify this handoff as a rolling runtime tracker.


---

# Appendix A — Complete October 17 architecture, decisions, historical recovery and prior handoff

**Verbatim preserved historical snapshot; do NOT treat its old HEAD, PAUSED status, missing worktrees, outdated requirements links or past test counts as current.** The front section above supersedes dated current-state claims. The original 3,430-line historical record is retained in full to preserve all original design evolution, U/P/O decisions, 94-source nested history, acquisition lessons and earlier agent instructions.

---

Author: GPT-6

# Zerogravity — comprehensive agent handoff, 17 October 2026

This is the requested context handoff for the live Rust repository at
`/home/pvl/spaces/abird/src/zerogravity`. It records the original vision, design
evolution, retained decisions, implementation history, accepted verification,
known failures, every currently open plan item and the two interrupted drafts.
Read it progressively; retain a short working TODO while moving through the
technical source references. The copyable entry prompt is
[NEXT_AGENT_PROMPT_2026-10-17.md](../plans/NEXT_AGENT_PROMPT_2026-10-17.md).

**Runtime implementation is PAUSED at the owner's request.** The current request
creates documentation only. Reading this handoff does not resume implementation,
launch a GPU, authorize recovery, acquire models/SDKs or publish anything. A new
explicit continuation instruction resumes the appropriate work within its scope.

<a id="contents"></a>

## Contents

1. [Checkpoint, authority, pause and reading order](#checkpoint)
2. [Original vision and architecture V1–V5.1](#architecture)
3. [Implementation chronology, actual tests and historical recovery](#history)
4. [Current acquisition contracts and failed experiments](#acquisition)
5. [Continuation, paused drafts and every open plan item](#continuation)
6. [Portable reconstruction, workspace and operating commands](#operations)
7. [Complete retained decision register, including later owner instructions](#decisions)
8. [Complete technical document and plan reading catalog](#catalog)
9. [Handoff validation and known limits](#handoff-validation)

<a id="checkpoint"></a>

## 1. Read this first: authority and checkpoint

### 1.1 Current source and working state

Recorded from live inspection on17 October2026:

| Item | Recorded fact |
| --- | --- |
| Canonical repository | `/home/pvl/spaces/abird/src/zerogravity` |
| Main branch | `main` |
| Verified implementation HEAD | `f197c9cf4a44e06a3c1cf3c532855545bed945fa` — explicit GGUF filename quant selection |
| Previous implementation | `f22f75972e63ea70771c59fbce065e371d5f29d4` — pinned metadata-only candidate discovery |
| Local tracking ref | `origin/main` at `fa6affa425089fdf9a15bfa10b330e03f738fec8`; MAIN was22 commits ahead,0 behind. This is a local ref observation; no new remote fetch/push was performed for this handoff. |
| Reachable main history | 77 commits, beginning with rewritten root `be95424`; it is not the original chronological implementation graph. |
| Last implementation acceptance | 975 focused executions, four strict artifact/CLI configurations, full isolated gate934 Rust passes/0 failures/69 intentional ignores/205 suite executions plus18 namespace tests, Nix source-integrity and both original trained-model cache routes. |
| Original trained routes | Separate exact calls: cache/direct17.60s; HF import23.70s. Both approved models, full248320-vocabulary logits, all42 state domains, two tokens/nonzero state, sessions/source/lease/mutation/reclamation exact. These are individual test durations, not a fabricated combined invocation or new8K/GPU qualification. |
| Signature | `git verify-commit f197c9c` succeeded during wrap-up. Signed commits remain required. |
| Dirty state before this documentation task | Two routine post-commit rolling TODO updates in `.agents/plans/TODO.md` and `.agents/plans/model-acquisition/TODO.md`, plus the interrupted filters run. Runtime source was committed; no staged changes. This task adds documentation and portable draft context, so do not expect the working tree to be clean. |
| Agent/process state | All implementation agents stopped; no MAIN Cargo/rustc process or process using either draft worktree remained at pause. Recheck rather than assuming this remains true. |
| Protected document | Root `README.md` is user-authored and must remain byte-identical unless explicitly requested otherwise. |

The earlier connector launches at `/home/pvl/spaces/abird`; through that connector
use `./src/zerogravity` as the working directory. In the current local shell the
working directory is already the canonical repository. Resolve it explicitly
before any mutation.

### 1.2 Authority order

1. The latest user instruction, including the current implementation pause and
   the immutable root README boundary.
2. [AGENTS.md](../../AGENTS.md) and [TODO-SPEC.md](../plans/TODO-SPEC.md).
3. [The plan index](../plans/TODO.md), the relevant live plan `TODO.md` and its
   current design/technical owners. These contain actual completed and open work.
4. The current source, tests, configuration and actual qualified source/device
   profile. A source identity, model hash, numerical limit or physical ownership
   boundary is not replaced by a handoff description.
5. This dated context snapshot, retained historical decision rows and old design
   documents. They explain reasoning and prior work; their historical status
   prose never overrides a later verified checklist or owner decision.

The handoff is a documentation snapshot, not a second live task ledger. Work
tracking remains one index, one TODO per plan and one own TODO per active agent.
Do not revive mandatory generated handoffs, evidence archives, staged blob/tree
receipts, per-file SHA inventories or archive-sealing duties. Actual content
hashes and numerical verification remain implementation requirements.

### 1.3 What changed since the original H4 takeover request

The original request began at historical handoff2125612, implementation0cad60a:
H4 was still hybrid, with566 physical completions across two tokens. H5 GDN
recurrence, attention/KV/RoPE/softmax/value, FFN/residual/embedding math, dynamic
activation quantization, resident state, device token feedback and native vendor
profiles were subsequently implemented and accepted within their documented
scope. Starting H5 again would discard substantial completed work.

The accepted baseline now includes original-model resident GPU execution through
512 consumed tokens and native CPP NVRTC/HIPRTC profiles. The old LLVM23 blocker
applies to a separate unqualified LLVM code-generation path; it is not a reason
to claim native CUDA/HIP remain universally unavailable. Local Renoir/gfx90c is
held after a later initialization/reset incident; historical Vulkan success does
not authorize using it now. CPU functional8K success, independent8K numerical
drift, resident512 qualification and deferred16K/32K are distinct scopes.

The latest implemented work is model acquisition, not another GPU phase. The
layout and thoughtful concurrency review are complete; no left-right migration
was justified. Finish acquisition convenience/provider gaps in bounded verified
steps when resumed, then the remaining CPU-first new-model and GPU optimization
plans. The full project is not complete.

### 1.4 Reading sequence

- Read this opening, the architecture and implementation chronology below, and
  the exact continuation/paused-draft section before editing.
- Read the live plan index, current acquisition TODO and
  [acquisition design](../plans/model-acquisition/README.md), including both plain
  completed-TODO archives. The handoff includes their source locations and the
  current exact open-item snapshot.
- Read the decision register snapshot below, especially its new U-056…U-070
  consolidation and the supersession notes. Historical U/P/O rows are retained
  to preserve reasoning, not to reopen obsolete evidence tasks.
- For core work, read the minimal execution spine, semantic/task/compiler,
  state/transactions, resource/accelerator and invariants contracts catalogued
  below; then inspect their actual crate owners.
- For a selected later plan, read that entire TODO and its referenced technical
  specifications/research before implementation. The complete document catalog
  ensures the remaining vision is not reduced to the latest acquisition task.
- Revalidate live HEAD/worktree/source/input/process/device state and relevant
  existing tests. Use one MAIN Cargo owner. Documentation checks do not qualify
  Rust runtime changes, new models, real devices or performance.

### 1.5 Historical recovery and its exact limits

The owner explicitly approved retiring completed raw evidence and replacing it
with concise tested completion lists and Git provenance. The repository was also
rewritten. Labels such as6d74f76 and bfe672c still name recorded approvals in plan
text, but their Git objects do not resolve in the current repository. The
original33-commit Git graph and later129/191-source aggregate versions were not
recovered as current standalone objects. Do not invent their ancestry or tell a
new agent to run `git show` on those unavailable labels.

Substantial earlier text DOES survive: reachable7e61cb6 contains the retired
multimodal aggregate, which embeds a complete older94-source implementation
handoff through U-040/P07-G3b. Its executive brief, early CPU/numerical/P07 phase
reports, older agent prompts and original design bodies can be read by the exact
source ranges catalogued below. The fourteen later deleted files plus one
replaced pre-product README body are also readable at7e61cb6/be95424. The
standalone early paths are absent; their absence does not imply all their text
is lost. Independent review caught and corrected that initial overly broad
conclusion before acceptance.

The retained canonical specifications, decision register, baseline/checklists,
current source and this recovered nested history together preserve the available
context. The history part separates positive recovery, unavailable Git objects
and unrecovered later versions; it includes all94 nested source reading ranges.
This handoff does not claim byte-complete recovery of the later129/191 snapshots
or the original33-node graph, and does not restore obsolete evidence/archive/SHA
maintenance as a duty.



<a id="architecture"></a>

## Original vision, design evolution and architecture

Documentation-only architecture context, integrated for this handoff. Inspected baseline: signed
`f197c9cf4a44e06a3c1cf3c532855545bed945fa` (2026-10-17). Implementation remains
**PAUSED**. This document records retained context; it does not authorize resuming
source work, running tests, downloading inputs, changing devices or publishing.
No implementation tests were run for this audit.

### 1. Authority and limits of the surviving record

Read [AGENTS.md](../../AGENTS.md), the [TODO spec](../plans/TODO-SPEC.md)
and [plan index](../plans/TODO.md) first. Latest user instructions govern;
current source plus applicable actually executed tests establish implemented
behavior; the plan TODOs distinguish accepted checkpoints and unfinished work.
The [engineering specification](../../docs/ENGINEERING_SPECIFICATION.md)
(ES-00–ES-15 and INV-01–INV-25) is the integrated design contract. The
[whitepaper](../../docs/WHITEPAPER.md) explains the argument, hypotheses and
limits. Neither makes every designed feature executable. Old phase statuses,
archive duties and illustrative Rust ABIs are historical context, not additional
tracking requirements or permission to act.

**Recovery distinguishes text from Git objects.** The owner-approved baseline
label6d74f76 and cleanup labelbfe672c survive in current TODOs but their Git
objects do not resolve in the rewritten graph. Original33-node ancestry and the
later129/191-source aggregate versions were not recovered. Standalone early
phase paths were already absent in rewritten rootbe95424; however the reachable
multimodal aggregate DOES embed an older complete94-source implementation
handoff throughU040/G3b, including early CPU/numerics/P07 reports and prompts.
The separate recovered-source catalog gives exact reading ranges. Fifteen later
retired bodies also have verified readable paths. Absence of a standalone file
must not be described as loss of every embedded copy. Old commands/results are
historical records, not newly executed tests. Initial blanket missing70-report
wording was corrected during independent review before handoff acceptance.

The [decision register](../plans/DECISION_REGISTER.md) preserves U-001–U-055,
P-001–P-096, O-001–O-074 and A-001–A-006; its dated owner consolidation adds
U-056–U-070. User requirements, assistant proposals, assumptions and open choices
are different categories. A proposal does not become a user decision through
repetition. Supersession changes the specified dimension, not every earlier
requirement. The multimodal-local ledger has its own historical U-040/U-041
namespace; cite that ledger explicitly rather than conflating them with the
canonical register's different U-040/U-041.

### 2. Beginning-to-current architectural evolution

| Stage | Original concern and retained result | Refinement that governs subsequent work |
| --- | --- | --- |
| V1, 2026-10-10 | One Rust-native semantic engine for ordinary models and extreme large/hybrid models; portable fallback plus hardware-specific kernels, exact state and source formats, easy local use and efficient serving. Research/design only at this date. | Semantic model truth must survive optimized lowering. Performance ambition is not a measured universal win, percentage target or deadline. See U-001–U-012 and recovered initial handoff in §11. |
| V2, 2026-10-10 onward | Combine previously separate compiler/research directions into one semantic/compiler/execution spine. Model recipes specify source math, not hardware kernel plans; include source formats, tokenizer/templates, batching, speculation and whole-state transactions. | SIR/RIR/KIR/CIR were useful compiler views, not four independent runtimes. See P-018–P-025 and [semantic IR addenda](SEMANTIC_IR_AND_COMPILER.md). |
| V2.1, 2026-10-11/12 | Heterogeneous GPU/NPU prefill and distinct ASR/TTS/image/video task loops become explicit. Partial output-row projection splitting is a placement optimization; it must not alter routing or recurrence. | Distinguish source encoding from temporary compute encoding and CPU/GPU-twin parity from source-model quality. Hardware fabric/power/hang risks require separate physical qualification. U-016, P-026–P-030. |
| V3, 2026-10-12 | Neutral, composable ModelDefinition/SemanticOp/StateDomain/TaskProgram/region/device/resource contracts replace project-named ownership and cloned model-specific engines. Preserve attributed research and file-specific licenses. | Broad workflow privileges, ambiguous domain commit ownership and an overgrown permanent IR stack were critically reviewed and narrowed in V4. U-017/U-018, P-031–P-035; recovered V3 snapshots are historical comparisons. |
| V4, 2026-10-12 | Minimal execution spine: two enduring representations, trusted minimal task control, prepare-only state domains, one root publisher, hierarchical physical admission and real completion before release. | The [minimal spine](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinement](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) supersede ambiguous older ABIs. P-036–P-045. |
| V4.1 | Accelerators participate at their natural closed execution region: kernels, captured graphs, AOT graphs, static dataflow or qualified managed sessions. | One model semantic program; no device-specific semantic IR or opaque vendor API masquerading as a native backend. U-019, P-046–P-055. |
| V4.2 | First-class multi-machine execution with independent compute, parameter, mutable-state and immutable-cache placement. Transport/topology and directional costs become first-class. | One semantic/state/resource authority across peers; remote work prepares private state, not independent roots. No automatic peer enrollment or transparent HA. U-020, P-056–P-070. |
| V4.3 | Native model routing, profiles, replicas and residency above the same executor; simple one-command local run/serve/library experience plus explicit advanced control. | Exact source/quant/profile resolution once per request, compatible sticky session ownership, original physical arbiter, no per-token router/proxy or silent model substitution. U-021, P-071–P-086. |
| V5, 2026-10-14 | Unified typed multimodal values, model assemblies, task drivers, state/media frontiers, quantization and finite-choice outputs. | MM-01–MM-11 extend V4.3; synthetic MM0 contract checks do not qualify trained audio/image/video execution. |
| V5.1, 2026-10-15 | Universal task-family and verified-assembly design, synchronized media, realtime sessions/playback, soft QoS and specialized source/placement policies. | Preserve original owners; expose unknown playback/provider outcomes and open policy choices. Do not claim every proposed interface or model is implemented. See §7. |
| Implementation and later acquisitions, through 2026-10-17 | Packaging → original CPU truth → source-exact resident GPU → shared-model/session/layout review → contention-first rejection → explicit acquisition/cache/source bridge → native providers/Xet and explicit filename quant selection. | Actual implementation is narrower than the architecture. Current accepted and unfinished scopes are in §§8–10 and the live TODOs. User pause remains binding. |

The current [master plan](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md)
§§1–15 is the V4.3 synthesis, not a frozen V1 snapshot despite its filename.
Its six concerns—model math, task control, compilation, execution, physical
resources, and service policy—remain separable. The
[reference architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md)
and [composition contracts](COMPOSITION_AND_EXTENSION_CONTRACTS.md)
explain extension points without authorizing parallel engines.

### 3. Exact ownership, representation and state publication

A verified artifact/config/tensor/tokenizer recipe builds one exact
**SemanticProgram**. A compiler produces **ExecutableBundle** objects containing
prepared code/regions, guarded shapes, bindings, local DAG readiness and resource
requirements. RIR/KIR are transient lowering views; CIR is executable dependency
organization, not a semantic commit owner. There is no per-token interpreter of
a universal workflow graph and no family-specific server required for a new
model. See [task/compiler refinement](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md)
§§1–5 and [semantic IR](SEMANTIC_IR_AND_COMPILER.md), V4 refinement.

The [minimal spine](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md)
§§1–10 defines exactly five owners: immutable semantics, immutable compiled plans,
logical session/TaskCoordinator (including TaskDriver), global physical resource
authority, and device/I/O execution authority. TaskDrivers express model-relative control:
token steps, denoising, audio segments or other bounded loops. They do not acquire
arbitrary workflow/tool/network privileges. Actual source currently implements
bounded pieces of this spine; the generic architecture is not fully delivered.

A step begins against an expected root/epoch; kernels and domains build private
proposals, COW versions and prepared outputs. Every participating domain must
validate compatibility/frontiers and prepare successfully. The **sole
TaskCoordinator publishes one root** only after genuine required device events.
A domain does not independently commit a partial model step. Output becomes
committed before external delivery; delivery attempt/ack/playback are separate.
Timeout, aborted Future, canceled I/O or drained socket is not proof DMA stopped.
Unknown completion retains actual handles and charge, fences the affected
service/card authority and refuses stale publication. See
[state/cache transactions](STATE_CACHE_AND_TRANSACTIONS.md)
§§3–8/V4 addendum and the minimal spine §§2–3, 7–10.

Compatibility is the meet of all relevant domains, not just a KV prefix: exact
source/quant/adapter generation, tokenizer/template/history, positions, attention
and recurrent/indexer state, lookup/ngram state, sampler/RNG/grammar and media
frontiers where applicable. Replay must reproduce the same declared numeric and
state profile; a portable trusted diagnostic checkpoint is not automatically
secure durable product persistence. Cross-source restore refuses.

### 4. Source, arithmetic and quality are distinct contracts

Recognized container → known semantic recipe → runnable selected backend →
numerically qualified → performance-qualified are separate statuses. Parsing a
GGUF/safetensors or recognizing metadata never licenses arbitrary custom model
code, unseen operators or new tokenizer behavior. A missing semantic op fails
closed; fallback requires a real legal reference implementation and sufficient
resources. New recipes should reuse exact existing ops rather than new engines.
See [model semantics](MODEL_SEMANTICS_AND_COMPATIBILITY.md) §§6–8,
[product pipeline](PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md), and
[immutable sharded-source specification](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md).

Immutable source identity has one issuer. A pathname, readonly descriptor,
readonly mmap or private-COW mapping does not freeze another writer's file.
The cache bridge hashes the **actual owned bytes** read through retained managed
descriptors, checks exact length and source metadata, reparses those same bytes
and retains admitted immutable-source ownership. Normal and cache preparation
use the same mathematical engine. Current executable source cap remains **4 GiB**;
large admitted BF16/F16 shards, complete index/header/name/dtype/offset/shape/hash
binding and safe host staging remain future work. Remote/provider acquisition
is data acquisition, not ModelReady or inference support.

[Numerical quality](NUMERICS_QUALITY_AND_BENCHMARKS.md) separates
formula/operator agreement, declared compute arithmetic, checkpoint/task quality
and distribution/state exactness. Scalar/FP64 references, pinned independent
source, Rust CPU, optimized devices and serving are different oracle layers.
Comparing an optimized implementation only to itself cannot establish original
source truth. Lossless repack, lossy derived quantization and temporary compute
encoding are different claims. A changed sparse mask/expert top-k/solver/guidance
schedule is changed semantics, not a free kernel optimization.

Retained original Qwen3.5 arithmetic constraints include source block/reduction
orders for Q4_0/Q4_1/Q8_0/Q6_K/F32 projections, explicit FP32 rounding stages,
GDN decay→prediction→beta→update→readout, ordered RMS/L2 and nonlinear policies.
Generic F64 multiply/add followed by F32 is not universal FMA emulation: midpoint
residual corrections and stored boundaries remain required where qualified.
Softmax's F64 reciprocal then F32 conversion is a two-round operation. Attention
value accumulation includes exactly the source's ceil(valid/256) padded FMA
prefix; canonical +0 future KV and signed zero are mathematical state. Preserve
partial IMRoPE's iterative source phase, polynomial versus scalar nonlinear
profiles, activation-quant rounding and finite-output checks. Greedy finite
f32 total ordering distinguishes +0/-0 and resolves equal-bit ties to lowest ID.
The authoritative detail is retained in the final section of the numerical
reference and executable source/tests; do not substitute a more convenient
formula because it is algebraically similar.

For new models, original HF grouped V axes versus converted tiled axes, converted
`-exp(A_log)`, non-GDN norm +1 and GDN SiLU versus attention sigmoid are explicit
Qwen3.8 source profiles. Gemma4 includes proportional RoPE, value norms/sliding
and shared-KV rules, E2B PLE/doubled FFN, and 26B raw K=V/MoE routing. Different
framework top-k/reduction/scale schedules need an independent choice and trained
qualification. Metadata exact-integer parsing and five analytic router cases
are not that proof. See model semantics' “Current metadata-only source profiles”
and the attributed [Gemma family contract](../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md).

Benchmark equal source/quant/input history/shape/context/sampler, power/driver,
physical placement and cold/warm conditions. Report useful committed whole-task
latency, TTFT/P95, memory and energy, not just launch count or a best microkernel.
Full-logit/all-state internal bit equality is stronger than sampled internal
checks but different from independent source full-vocabulary distribution proof;
partial top20 agreement cannot establish KL/TV. No universal faster-than claim
or imposed percentage/deadline was accepted (U-012).

### 5. Resources, accelerators and virtual residency

The single physical arbiter accounts models, private state, code/scratch,
activation, staging and output on actual physical domains. UMA aliases count
once; virtual address/GTT limits do not establish physical free RAM. Coarse
admission and fair workers own global policy; bounded local event-backed pools
serve hot operations without a global per-op mutex. Logical reservations must
state their actual coverage: a raw source lease is not all process RAM and a
cooperative cache quota is not filesystem-wide exclusive capacity. See
[memory/residency](MEMORY_RESIDENCY_AND_IO.md),
[engine protocol](ENGINE_CORE_EXECUTION_AND_PROTOCOL.md) and
[spine resource hierarchy](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md).

The [accelerator contract](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md)
§§1–8 defines NativeKernel, CapturedCommandGraph, AheadOfTimeDeviceGraph,
StaticallyScheduledDataflow and narrowly qualified ManagedDeviceSession. A
closed region exposes every external input/output/effect, guarded shape,
precision, lifecycle and state bridge. Opaque text APIs belong behind a
TaskServiceAdapter, not a purported device backend. Compilation/load/warmup,
code/scratch budgets, actual cancellation and checkpoint/replay are part of
qualification. TPU/wafer/coherent CPU-GPU/NPU descriptions are architectural
coverage, not executed hardware support.

Heterogeneous output-row prefill must preserve exact joins, source semantics,
state and source-versus-derived quality. CPU↔GPU twin↔NPU agreement on a derived
encoding does not prove quality against original weights. Retained replay needs
a compatible state frontier and actual completion; no automatic unqualified
power/fabric tuning. See [heterogeneous prefill/replay](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md)
and [GPU/UMA safety](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md).

The [virtual tensor/SSD contract](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md)
§§1–11 distinguishes stable source extents from generation/layout-specific
materialized views and Ready events. Expert blocks, lookup rows, frozen state
and media have distinct policies over the same I/O/event/arbiter owners.
Encoded/captured/queued consumers must pin before submission through actual
completion, not only once a kernel starts. Mixed-size/codec classes share a
global byte cap, preserve hot entries and charge staging peaks. Cancel does not
free in-flight DMA. A resident hit needs no mandatory host rematerialization;
SSD enables capacity, not a promise dense spilled inference remains interactive.
The SSD-01–SSD-32 gates, including queued readers and mixed cohorts, remain
planned, not an implementation completion inferred from cache acquisition.

### 6. Distributed placement, routing, serving and speculation

[Distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md)
§§1–16 separates compute, immutable parameters, mutable state and immutable
cache maps, with derived physical transfers in a bundle. Legal modes include
layer/state colocation, remote FFN/expert compute, explicit attention/state
separation, hybrid prefill/decode handoff and only operator-proven partition
algebra. Merging attention summaries requires valid masks and source numerical
semantics; it does not license generic GDN/KDA recurrence sharding. Directional
latency/bandwidth, relay/encryption/queue/copy costs and capacity versus speed
must be measured, not guessed from aggregate link bandwidth.

The [remote protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md)
binds source/schema/tenant/session/root/placement epoch/lease and actual events.
Peers prepare state; one coordinator publishes. Migration occurs at a safe
committed frontier with destination reservation, transfer validation and old
owner fencing. A trusted tunnel does not confer authentication, RDMA, model
redistribution rights or permission to use a peer. Local runs have zero automatic
peer overhead; fail-stop/replay is not transparent HA or a consensus cluster.
SSH fleet checks do not qualify distributed inference.

[Router/residency](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md)
§§1–13 separates RegisteredModel/source, revisioned Alias/Profile, physical
ResidentModel/replica and TaskSession/StateRoot. Resolve exact source/quant and
profile once at admission; route only among compatible replicas and keep live
session state sticky. An explicitly configured automatic model selector is a
changed target policy, not equivalent execution. Catalog count is not simultaneous
physical residency. The catalog remains cold control work: the attempted
left-right publication migration was rejected for lack of representative
contention; no accepted speedup or remaining dependency on that experiment.

[CLI/API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md) and
[runtime UX](RUNTIME_SERVING_AND_UX.md) specify one library engine for
run/serve/SDK, progressive configuration and explicit security/admin boundaries.
No mandatory daemon/download/config file; a miss never silently acquires.
Chat/completions/Responses/embeddings/SSE/tools must be explicitly tested subsets;
tool-call data is not permission to execute tools, and API resemblance is not
blanket compatibility. A reserved/disabled service is not a working production
server. Root README is user-owned and remains unchanged.

[Scheduling/speculation](SCHEDULER_BATCHING_AND_SPECULATION.md) and
[joint goodput](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md)
retain one scheduler and target authority: request/tenant admission, queued
prefill/residency and physical active steps are distinct valves. Chunking,
grouped experts, fair age/decode/media scheduling and speculation are costed
against useful committed goodput/P95. Draft work stays provisional; rejection
rolls back every relevant domain and RNG/history law. MTP/ngram/suffix and newer
drafting strategies require exact primary identity and independent numerical/
state qualification; an unresolved name or provisional capability count is not
full parity. No second model scheduler or proxy per token is permitted.

### 7. V5/V5.1 multimodal and realtime vision

The [V5 contract](UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md)
MM-01–MM-11 and [V5.1 universal architecture](UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md)
§§1–14 add typed tokens, PCM/features/RVQ, pixels/latents/frame PTS, packets,
transcripts, finite decisions and structured outputs to the same owners. Rational
rate clocks use checked wide arithmetic and preserve resampler remainder; codec
depth is not elapsed time. Media state is not disguised as text KV.

A [verified assembly](VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md)
binds exact component weights/config/tokenizer/processor/codec/transforms/license,
typed ports and source identities. It differs from a physical replica and from
a workflow joining different source models. Only identity-compatible immutable
parts can be shared. Specialization has independent semantic, physical,
task-control and placement/QoS axes; a codec or source quantizer cannot silently
change the solver/RNG/guidance/task law.

Task families include causal generation, encoding/pooling/ranking, Whisper-style
AR ASR, CTC collapse, RNNT/TDT, non-AR TTS, nested-RVQ/dual-AR generation,
acoustic flow, duplex, image flow/AR, temporal video, joint AV and calibrated
finite-choice outputs. Hosted opaque work is an explicitly authorized
TaskServiceAdapter with privacy/cost/idempotency/Unknown reconciliation; it is
not native model support or permission to invent proprietary internals.
Ternary/PQ-like source encodings include scales/transforms and are not merely a
plain low-bit matrix. Source-specific operators and actual checkpoints remain
necessary for every family.

[Realtime/QoS](REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md)
§§1–18 distinguishes capture/perception/response/output lanes and session/turn/
transport/physical epochs. Multitrack frontiers and AV CommitGroups prevent
partial RVQ, latent frames or unsynchronized outputs from being presented as
playable media. Model commit, server delivery, ACK, playback and playback
uncertainty are distinct. Barge-in changes the relevant turn/output epoch, not
necessarily the physical card fence; a slow client gets bounded queues, not
premature GPU release. Sync, progressive stream and bidirectional live regimes
have different delivery/reconnect contracts. Soft QoS is not a hard realtime
guarantee; measure first-valid playable output and useful media deadlines.

The [multimodal decision ledger](../research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md)
retains DMM-001–DMM-032 as proposals and OMM-001–OMM-022 as open choices.
Checkpoint choices (ASR/TTS/image/AR image/video/decision), media/device memory,
recovery, SLO/playback/consent/hosted cost, transport/midturn/persistence/licenses,
unverified model identity, ABI, session-vs-card cancellation, egress and compile
latency must not be silently decided by a generic illustration. The
[multimodal TODO](../plans/multimodal/TODO.md) and retained family dossiers are
the implementation/test frontier. Existing mock contracts do not establish
trained ASR/TTS/image/video quality or realtime operation.

### 8. User decisions, proposals and supersessions: complete ID crosswalk

Canonical source: [DECISION_REGISTER.md](../plans/DECISION_REGISTER.md).
These are summaries to navigate its exact statements, not replacements.

| IDs | Binding request or historical change |
| --- | --- |
| U-001–U-009 | Name, Rust-owned engine, broad and extreme models, portable legal fallback, attributed prior-art learning and semantic IR without hot-path abstraction tax. |
| U-010–U-015 | Initial research-only scope, complete decisions/pending context, no invented target deadline, corrected canonical path, unified designs and maintainable future research. U-013 replaces only U-002's destination. |
| U-016–U-022 | Heterogeneous prefill design, neutral/source-license policy, minimal critical review, natural accelerator granularity, first-class distribution, native router/simple UX and publication-quality design. None is blanket physical proof. |
| U-023–U-025 | Packaging-only implementation first; U-024 later authorizes reference implementation; U-025 fixes original .8B/2B CPU→CUDA/HIP/Vulkan before new Gemma/Qwen CPU→GPU order. |
| U-026–U-029 | Rigorous numerical/context gates; U-027 permits GPU work after genuine functional8K and defers16K/32K, without relaxing independent numerics; U-028 retains8K drift investigation; independent parallel streams are required without invented external-agent execution. |
| U-030–U-035 | Mixed source codecs, persistent residency, decode-once/static-vs-step split, typed async readback, uncertain-Future keepalive/quarantine and global physical/fair admission. Host proof at an early stage is not later hardware proof. |
| U-036–U-040 | Attested physical binding, real Vulkan workers, AMD native-F32 failure versus qualified WideF64, narrow adversarial fixes and immutable shared-model/independent-session catalog design. |
| U-041–U-049 | Absorb additive multimodal vision, repository ownership/user README preservation, H3/H4/H5 source operators, comprehensive context, heterogeneous retained replay and neutral core terminology. Retain source attribution. |
| U-050–U-055 | H7 source SiLU/output gates, H8 coefficients, H9 RoPE/current KV, H10 QK, H11 padded softmax and full resident GPU continuation/H12 source-padded value accumulation. Stage-local incomplete statuses are superseded by later accepted scope, not by weakening semantics. |
| U-056–U-060 | Finish qualified original GPU/native paths, first optimization fleet, parallel/disposable-source ownership, layout and contention-first pass; relocate reusable probes and preserve the sole product CLI. |
| U-061–U-064 | Explicit XDG HF/GitHub pull and finite pinned parity frontier; new model CPU basics first; full applicable Qwen/advanced decoding frontier; reject speculative cold-catalog concurrency migration. |
| U-065–U-068 | Publication authorization stays action-specific; checkpoint acceptance and simple TODOs supersede generated handoffs/evidence archives/SHA inventories. Production content hashes and numerical tests remain mandatory. |
| U-069–U-070 | Wrap/revalidate then PAUSE; new comprehensive context/start prompt is documentation-only and does not resume implementation or revive obsolete evidence duties. |

Proposal continuity covers **every P ID**, while preserving proposal status:
P-001–P-017 initial portable/tuned/extreme/compiler/state/format/UX proposals;
P-018–P-025 integrated compiler/execution spine; P-026–P-030 heterogeneous/task
research; P-031–P-035 neutral V3; P-036–P-045 minimized V4;
P-046–P-055 accelerator V4.1; P-056–P-070 distribution V4.2;
P-071–P-086 router/UX V4.3; P-087–P-091 public document/agent delivery;
P-092–P-096 packaging/build/release/module foundation. P-036 minimizes the old
permanent IR stack; prepare-only root publication replaces vague domain commit;
U-067/U-068 replace P-091's former hash/archive workflow. The six target build
definitions and installer checks do not themselves prove physical builds on six
platforms, signed releases or a working daemon.

All **O IDs** remain traceable, with narrow later decisions noted rather than
silently removing them: O-001–O-022 initial licensing/platform/FFI/toolchain/model/
quality/API/config/trust questions; O-023–O-026 NPU/source/compiler/hang/task
choices; O-027–O-034 reuse/training/release/delivery/allocator/plugin/control;
O-035–O-041 vendor graph/SDK/state/hardware/privacy/ambiguous accelerator name;
O-042–O-053 remote topology/enrollment/privacy/transport/recovery/placement/
partition/rights/SLA; O-054–O-066 CLI/config/catalog/API/service/replica/profile/
admin/residency/multi-role policy; O-067–O-074 publication/legal/channel/platform/
service qualification. Later U-023/U-025 and actual code settle specific bootstrap
and first-model choices without closing every general question. Historical
O-071/A-003 statements about no measurements belong to their research date;
they do not erase later bounded physical tests or establish universal superiority.
A-001 is superseded by U-013; A-002 requires refreshed source versions; A-004
requires real fallback semantics/resources; A-005/A-006 grant no ambient
publication/account/secret authority. Future decision records must identify
source/options/tradeoffs/tests/conclusion/superseded IDs and affected files.

### 9. Accepted implementation versus historical stage facts

[Accepted baseline](../plans/accepted-baseline/TODO.md) records owner-approved
original Qwen3.5 .8B/2B functional CPU Phase1 **16/16 through genuine8K** and
complete original-model resident GPU execution through **512 consumed tokens**.
Its accepted five backend profiles are CUDA local RTX3060/a1 RTX4060, HIP a1
gfx1150/x2 gfx1151 and SDK-free NVIDIA Vulkan: **not five completely qualified
physical cards**. The record lists135 profile checks plus12 host fingerprints.
It includes compressed embeddings, activation quantization, GDN/attention/FFN,
all42 private domains and compact token feedback. This paragraph relays retained
owner approval; the unavailable old graph prevents independent reconstruction of
its full original report sequence in this audit.

State-only prefill, exact Contiguous aliases, prepared range bindings and fused
View certification are not driver capture, GPU tokenization or a generic device
scheduler. Trusted diagnostic replay is not authenticated product persistence.
Independent source512 comparisons cover146 stages/model with zero nonzero-FP32
differences and56 recorded2B signed-zero distinctions; internal full-bit tests
remain strict. Native CPP scalar/libm/loader profiles are bounded; separate LLVM
device code generation is unqualified. Old statements “LLVM unavailable” or
“full GPU pending” describe earlier stages and must not overwrite later approval.

The failed cold-catalog left-right experiment was functionally reviewed but
rejected as an adoption target because no actual contention was established.
Production retained the simpler prior catalog; future adoption needs realistic
measured contention, lifetime/replay correctness and maintenance benefit.
Functional test success does not imply accepted migration or speedup.

[Model execution](../plans/model-execution/TODO.md) records Qwen3.8 27B and
Gemma4 E2B/26B **metadata/config/tensor/source-index contracts only**. No actual
trained new-model execution, tokenizer qualification or complete checkpoint
loading follows. This remains true despite extensive original-model acquisition
and loader tests. Official source links and subtle layout/math contracts are
retained in model semantics/Gemma references, not disposable historical prose.

The [numerics/context TODO](../plans/numerics-and-context/TODO.md) retains the
independent8K GDN problem: first final-token divergence at .8B layer4/2B layer1,
shared top20 logprob differences0.20372924/0.19678696 at native ubatch512 and
matching greedy7993. These do not prove full-vocabulary agreement. Historical
functional limits0.45/0.05 are not permission to change source-bit operators.
16K/32K whole-model tests and resident GPU beyond512 remain unqualified. The
separate historical AMD FMA failure is not an established cause of CPU GDN drift.

### 10. Current acquisition and remaining execution boundary

The [acquisition TODO](../plans/model-acquisition/TODO.md), its
[design](../plans/model-acquisition/README.md) and
[implementation specification](../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md)
record accepted source-preserving cache lifecycle, bounded explicit HTTP,
HF/GitHub files/releases/components, direct HTTPS contracts, OCI/GCS/S3/Azure,
ModelScope and native Xet, then metadata-only candidates and explicit filename
quant selection. These use the sole HTTP/cache/executor/publisher, original
arbiter and no implicit run/inspect fetch. Default CPU/cache-only closures stay
transport-free. Native provider/profile qualification is synthetic protocol
coverage unless an actual live source test is explicitly recorded; no claim of
complete upstream parity follows from a typed Unsupported frontier.

Exact and capped unknown lengths are separate. AtMost requires explicit opt-in
for auxiliary data only, real terminal EOF and actual length/hash; no percentage
against an unknown total or durable unknown-prefix resume. Required weights,
config/tokenizer/index/license closure stays exact. SHA trust distinguishes
caller/pinned expected digest from observed FirstSeen. Git OIDs, validator labels,
mutable tags and source filenames are not trusted raw content identities.

Local HF import is offline data-only copying from retained opened source FDs,
actual-byte verification, source/target disjointness and conservative qualified
filesystem policies. It never proves remote repository membership. Cache-backed
run preparation holds an admitted raw ImmutableWeights lease through retained
SourceWeights/Arc owners; sidecar copies have bounded staging through parse.
Managed descriptor protections survive dup/inherited handles until final OS
close. V2 exact-snapshot reclamation is cooperative; v1 migration, orphan/eviction
sweeps and arbitrary raw readers are not qualified. Named unique-inode charge is
not physical freed space. Raw run/inspect/File readers need external quiescence.

Native Xet authenticates complete file roots and exact raw length/SHA independently
through original primitives and bounded reconstruction. It preserves signed
interval sets/auth isolation/term order/replay/EOF and counts physical attempts,
observed bytes, chunks, leaves and refreshes across files/reissues. It does not
claim per-term partial authentication, performance gain, persistent native resume
or general required native-index support. Native default256MiB is a scoped
concurrent scratch+cache admission envelope; ordinary default128MiB and explicit
caller limits remain unchanged, not a global process RAM guarantee.

Current signed **f197c9c** accepts C2 explicit filename GGUF quant: exact terminal
basename tag/canonical complete split family, same pinned inventory/closure and
one fetch runtime; no auto quant ranking, whitelist, numeric eligibility or
fallback. C1 candidates atf22f759 list regular choices only; stored-Git/tree/LFS
versus verified-content length bases are explicit. No HEAD/index/model payload or
publication is implied by discovery. Quant offline verifies a requested exact
snapshot subset/current bounds/pointers/roles/closure; original Files offline
remains initial-selection/role-agnostic. Explicit components apply their stricter
Weights closure; generic cache verification still validates literal bytes only.

Recorded latest actual checks:975 focused passes/four strict configurations;
full frozen isolated gate934 Rust passes/0fail/69ignored/205suites plus18namespace
tests, Nix source integrity and26 local links. Original release cache_prepare
routes pass separately17.60s and23.70s. Each uses approved .8B/2B originals,
>=2 tokens, all248320 logits/all42 domain values bit-exact, independent sessions
and source IDs, retained raw lease, mutation/ref retarget and private reclamation.
These are separate test durations, not one aggregate invocation, and add no new
model/context/GPU qualification. Earlier per-phase counts are not additive totals.

**Paused unfinished drafts:** C3 filters in `/home/pvl/spaces/abird/src/zerogravity-filters-work`
has a matcher/oracle scaffold, is unwired/not compile-ready, with all Rust tests
unrun. C4 in `/home/pvl/spaces/abird/src/zerogravity-xet-index-work` changes only
HF/xet mod/budget draft paths; private callback/defaultfalse opt-in/all-counter
Result continuation and two unit drafts are uncompiled/unrun; CLI integration
and actual index tests are absent. Required future policy is same pin and bounded
metadata reader, trusted LFS rawSHA plus complete Xet root/actualEOF before index
parse, same admitted scratch and every planning/execution counter carried forward.
Default/dry-run must remain no-CAS; apply must explicitly opt in. Neither SIDE
is an implemented feature or a task to resume during this documentation request.

Other pending plans retain their own truth: [CPU/GPU optimization](../plans/cpu-gpu-optimization/TODO.md)
(measured hotspots, held Renoir/gfx90c, native numeric modes/recovery/fairness),
[memory/streaming](../plans/memory-and-streaming/TODO.md) (global host/UMA and
SSD/state/prefix policies), [serving/speculation](../plans/serving-and-speculation/TODO.md)
(production API/batching/advanced decoding), [multimodal](../plans/multimodal/TODO.md),
[distributed](../plans/distributed-execution/TODO.md) and
[platform/release](../plans/platform-and-release/TODO.md).
Full applicable pinned Qwen/provider parity needs finite enumerated source/API/
CLI/internal inventories with honest open rows; provisional counts and successful
small examples are not exhaustive closure. New models still need authorized
checkpoints/host capacity, immutable shards, trained CPU truth, then physical GPU.

### 11. Exact surviving historical recovery and unrecoverable material

Read-only recovery ref verified for the following bodies:
`7e61cb63efd0c1429aa96044f2f7d28fda1e59aa`, parent of archive deletion
`9239719c25128028fd78a6a60f817ce6b626517f`. These bodies also belong to the later
retired standalone design/archive set in the rewritten graph. Its multimodal aggregate also embeds the older94-source implementation archive, giving partial early-report recovery.
To inspect without restoring a file, use `git show REF:PATH`; do not recreate
archive directories, generated aggregates or obsolete source inventories.

| Historical body/path at that ref | Unique recoverable context and current semantic owner |
| --- | --- |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md` | 474lines; V2.1/NPU update then V2 synthesis; lines171–474 embed the original V1 initial handoff. Current master, vision, IR, model/state/resource/UX docs retain its live requirements. |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md` | V2 master snapshot; current master and minimal spine supersede old permanent-IR/kernel-plan ownership. |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_ROADMAP_V2.md` | Early implementation sequencing; current roadmap and plan TODOs distinguish delivered phases. |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_STAGE0_V2.md` | Original pre-framework feasibility gates; current experiment/acceptance/roadmap source keeps falsifiable ownership tests. |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_EXPERIMENTS_V2.md` | Early research measurements-to-do, not execution results; current numerical/experimental design owns them. |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_ACCEPTANCE_GATES_V2.md` | Historical quality/memory/source/failure gates; current engineering invariants and plan verification refine them. |
| `.agents/prior-art/DESIGN_EVOLUTION_V3_SNAPSHOT.md` | Three embedded V3 architecture/model-task references; current V4 composition/TaskDriver/prepare-only contracts resolve their overbroad sketches. |
| `.agents/prior-art/DESIGN_EVOLUTION_V3_HANDOFF_SNAPSHOT.md` | V3 decision/minimality/research transition; canonical register retains authority and supersessions. |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_DISTRIBUTED_PARALLELISM_V4_1.md` | Pre-V4.2 topology snapshot; current distributed placement/protocol separates four maps and real transport costs. |
| `.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md` | 2,796,909byte/23,979line aggregate. Nested current contracts/dossiers and historical wrappers were consolidated; current multimodal references, ledger, realtime QoS, SSD and technical source references are semantic owners. Counts129/191 are historical aggregate inventories, not current physical file counts. |
| `.agents/research/MULTIMODAL_PARALLEL_RESEARCH_WORKSTREAMS_2026-10-14.md` | Early parallel agent briefs, negative fixtures and modality gate proposals; retained integrated acceptance/roadmap and multimodal TODO own pending tests. |
| `.agents/research/MULTIMODAL_V5_1_RESEARCH_WORKSTREAMS_AND_AGENT_BRIEFS_2026-10-15.md` | Expanded family/QoS/source-license briefs; current multimodal ledger/dossiers/contracts own decisions and open gates. |
| `.agents/research/SSD_STREAMING_FIRST_CLASS_ARCHITECTURE_AGENT_HANDOFF_2026-10-15.md` | SSD wrapper/context; retained virtual tensor/SSD contract and SSD acceptance roadmap own the technical plan. |
| `.agents/research/EXTREME_MODEL_SPECIALIZATION_DESIGN_AGENT_HANDOFF_2026-10-15.md` | Hybrid/replay/low-bit/GPU specialization context; retained extreme model, heterogeneous replay, GPU/UMA and quantization references own source math/gates. |
| `.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md` | Former README body. Current retained path is a deliberate research pointer required by the immutable root README; preserve it. Historical prose is not current product support. |

**Recovery limits:** current standalone early handoff/phase paths and the original
33-node Git graph were not recovered. The later129/191-source handoff versions
were not found in the inspected current/ref/reflog/unreachable and retained
archive bodies. The older94-source U040/G3b archive IS recoverable inside the
multimodal aggregate; its executive brief, early CPU/P07/report/prompt bodies are
not lost simply because their former paths are absent. Read its complete94-source
catalog in the following history-recovery supplement. Later H4+ implementation
context survives in current decisions, numerical/model technical references,
accepted-baseline/source and TODOs, without a claim of restoring every old verbatim
report/command or original commit ancestry. Do not infer unavailable SHA objects
from labels printed in recovered text. The initial coarse negative audit was
superseded by this positive nested-source discovery and independent review.

### 12. Phase and falsification map for preserving future work

The [P00–P23 roadmap](../plans/IMPLEMENTATION_ROADMAP.md) is a capability graph,
not a claim that each phase is done. Its complete phase vocabulary is:

| Phase | Intended scope |
| --- | --- |
| P00/P01 | Lean build/source foundation; actual architecture-specific Rust/hardware feasibility. |
| P02/P03 | Independent math/state oracle; trusted artifacts, codecs and rendering. |
| P04/P05/P06 | Typed semantic/task control, legal optimizer/async execution and useful portable backend. |
| P07/P08 | Accelerator streams/leases/providers; first complete real model/task vertical slice. |
| P09/P10/P11 | Coordinator/workers/serving; physical full-state memory/prefix management; hybrid/stateful operators. |
| P12/P13/P14 | Routed sparse/tiered parameters; first and independently different extreme model specializations. |
| P15/P16 | Target-authoritative speculation/transactions; continuous batching and joint resource cost. |
| P17/P18 | Broader families/adapters/embedding/rerank; actual audio/image/video/iterative tasks. |
| P19/P20 | Heterogeneous/remote execution; bounded compiler/autotuner and negative experiments. |
| P21/P22/P23 | Security/license/release; full fair quality/performance qualification; extension/change-radius drills. |

The [experiment program](../plans/EXPERIMENT_PROGRAM.md) preserves H01–H86:
H01–H14 kernel/cache/speculation/serving/memory/topology/task hypotheses;
H15–H25 compiler/effects/async/capture/search/frontier/modularity/malformed-input/
benchmark/hardware/runtime-cost tests; H26–H31 heterogeneous projection/NPU/
machine safety/source quality/non-token tasks; H32–H43 V4 root publication,
device progression/hierarchical resources/shapes/task minimality/regions/delivery/
epochs/future parameter version/cache migration/fairness/change-radius;
H44–H55 graph/dataflow/device-source composability; H56–H70 physical distributed
placement/transport/authority/failure/capacity; H71–H86 local run/catalog/residency/
replicas/profiles/API/security/roles/hot reload/minimality. Stage0 and
[release gates](../plans/RELEASE_ACCEPTANCE_GATES.md) retain the matching
negative tests and eligibility rules. Earlier “NOT RUN” section titles are
historical design status; no one should turn a similarly named later test into
blanket completion of those matrices.

**Identifier namespaces matter:** these design H01–H86 hypotheses are not the
same as implementation H3/H7/H12 operator slices or the later H25 workspace
layout milestone. Likewise design P07 contains many later bounded implementation
subphases. Preserve the exact source/namespace when reporting a pass, rather than
inferring coverage from a matching number.

The [multimodal acceptance/roadmap](../research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md)
and family dossiers retain MM-01–MM-24; ASR-01–12, TTS-01–12, IMG-01–12,
VID-01–15, DEC-01–06, BON-01–08 and SRV-01–12. Distinct negatives include bad
source/conditioner/codec masks, rational-clock overflow/discontinuity, incomplete
RVQ depth, incorrect solver/seed, temporal VAE/frame/GOP mismatch, uncalibrated
finite outputs, missing required Hadamard transform, stale ACK/reconnect cursor,
late physical events, mixed AV premature presentation, slow-client growth and
unauthorized egress/cross-tenant replay. These should stay attached to their
source-qualified task, not collapse into one generic “multimodal supported” tick.

The [implementation traceability](../plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md)
map connects user/engineering/invariant/phase IDs; the original source dossiers
provide license and equation context. They are complementary references, not
fresh evidence or an obligation to restore deleted archive bodies. Acceptance
requires each real input/backend/context/task cell and its relevant failure
cases, not merely counting retained Markdown IDs.

### 13. Coordinator integration and next-agent guardrails

Integrate this context under the coordinator-owned handoff plus current plan
links; keep original research/prior-art references in place. Preserve source and
license attribution through [the provenance policy](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md),
not project-branded core types or old engine clones. Do not manufacture a new
archive/generated-handoff maintenance system, extra evidence log hashes or
mandatory stage receipts. Content hashes remain where production integrity,
source identity and authorized original checkpoint tests require them.

A next agent must first read the current plan index/TODO rules, live acquisition
TODO and the two paused draft run TODOs, then verify live source/ownership if the
user actually asks to resume. Read source behavior and failed/interrupted tests;
merge only scoped reviewed bytes with fresh modification times (the documented
stale Cargo experiment is real). Keep one shared Cargo owner, preserve unrelated
work and protected README, and distinguish qualification inputs/profiles from
future capabilities. The immediate next action **now** is completing the requested
documentation and remaining paused, not implementing C3/C4 or running the gate.


<a id="history"></a>

## Implementation history, verification and recovery

Author: Codex history-audit agent (coordinator: root).
Provenance: signed `f197c9cf4a44e06a3c1cf3c532855545bed945fa`.
Status: documentation-only audit; runtime implementation remains **PAUSED**.
No implementation,
Cargo, model, SDK, provider, hardware, Git mutation or network action was run.
All historical test results below are transcribed from retained accepted plans;
this audit did not rerun them. Current source was read only to identify owners
and retained phase labels. A source assertion or a commit subject alone is not
an executed test result.

### Principal findings for the coordinator

1. The current local reachable history contains **77 commits**, not the original
   pre-rewrite 33-commit graph. Its first **46 commits** form an owner/test/vendor/
   tooling decomposition of already developed implementation. They must not be
   relabelled as the original 33 chronological implementation steps. All77 exact
   current IDs/titles are listed below. The original33 ledger cannot be recovered
   from the local objects inspected in this audit.
2. Accepted pre-rewrite results remain authoritative as owner-approved historical
   completion. Their labels, including `6d74f76`, are not current reachable hashes.
   Do not silently replace them with a convenient rewritten commit or imply a
   fresh gate was run on that rewritten commit.
3. Some retired design/archive bodies are recoverable from **reachable** commits
   `7e61cb6` and `be95424`. Standalone early handoff/phase paths were already absent from `be95424`, but its multimodal aggregate embeds an older complete94-source handoff throughU040/G3b. The positive nested recovery and all94 reading ranges supplement this audit below. Original33-node ancestry and later129/191-source versions were not recovered; do not confuse missing Git objects/standalone paths with missing embedded text.
4. CPU original-model functional acceptance is16/16 through genuine8K. Resident
   GPU original-model acceptance is512 consumed tokens on five backend profiles,
   not five physical cards. Independent8K GDN numerical drift and16K/32K whole
   model contexts remain open. Acquisition acceptance does not broaden these
   model/numerical/GPU limits.
5. Current quant selection is accepted at `f197c9c`; full gate934 Rust executions,
   69 ignored,205 suite executions plus18 namespace tests. Both original model
   preparation routes passed separately17.60s and23.70s. Two preserved filters/
   native-index SIDE drafts remain uncompiled/unqualified and runtime is paused.
6. The fixed root-README research link must remain valid. Do not blanket-move
   `.agents/research/`; the pre-product snapshot path is intentionally retained.
   Do not reintroduce the third-party project name the user requested removed,
   even in a historical-title quotation. No current77 subject contains that name.

### Current history, rewritten dates and available references

Read-only commands actually run: `git log --all --reverse`, `git show`,
`git ls-tree`, `git for-each-ref`, `git reflog show --all`,
`git fsck --no-reflogs --unreachable --no-progress`, and individual
`git cat-file -e OLD^{commit}` checks. No refs were created or fetched.

Observed local refs:

| Ref | Exact object |
| --- | --- |
| refs/heads/main | f197c9cf4a44e06a3c1cf3c532855545bed945fa |
| refs/remotes/origin/HEAD | fa6affa425089fdf9a15bfa10b330e03f738fec8 |
| refs/remotes/origin/main | fa6affa425089fdf9a15bfa10b330e03f738fec8 |

These are local remote-tracking refs only; no network remote state was checked.

Topological order is meaningful; rewritten calendar dates are not a trustworthy
execution timeline. Example: root `be95424` is dated2026-10-13, `57a9ad7` is
2026-11-05, and its child `1af2ac4` is2026-10-16. These are observed author AND
committer dates. Do not place the Nov5 owner decomposition after Oct17 provider
work because of those synthetic dates.

The first46 current commits package the accepted baseline by source owner. They
include the final GPU/model/layout/cache implementation, so a current commit
introducing a file is not proof that all its H-phases happened together. The old
approval labels and retained plans supply the historical development sequence.
The later31 commits are the reachable continuation, including documentation,
XDG namespace change and23 acquisition-phase commits through quant selection.

#### Entire reachable commit ledger

Exact subjects below are current Git metadata, not claims of independent tests.
The first46 rows are the rewritten source decomposition; rows47–77 continue it.

| # | Exact current commit | Exact current subject |
| --- | --- | --- |
| 1 | `be95424837deb7919adf34503db71d27d3ad3adb` | docs(design): product, architecture, prior-art and research plans |
| 2 | `1e10b5cc1d5cf34832477bc83ef39117a765c6a0` | feat(core): immutable source identity, state roots and physical resource ownership |
| 3 | `af9a5d3969c9a87951990223b6d289aba69c63fb` | feat(artifacts): sharded source artifacts and model identity |
| 4 | `b276fd01e421f1b7e416ae773610b5823b9da36a` | test(artifacts): cache transactions |
| 5 | `586d440b144727bdd9acea7455afb27f6c38ae25` | test(artifacts): qwen text contract |
| 6 | `782e6fc48f0924cc055232d56338da62b68b1a79` | feat(cpu): portable CPU math and weights |
| 7 | `1fd12da7840151ba47ab43368b4e4f435aac3d3b` | feat(model-qwen35): source-verified Qwen3.5 forward and tokenizer |
| 8 | `901ba9ae2f2ed531d992a6d657a37dc362518bd0` | test(model-qwen35): source layout contract |
| 9 | `e6ecf02bb7d951e01e66b430a24553c690b4ac5b` | feat(model-gemma4): Gemma contract, MoE and 26B variant |
| 10 | `81642a9af4d88d928327a1012af1594168704956` | test(model-gemma4): e2b contract |
| 11 | `7aa19c268026069ad916f6a69440dab602aaa94a` | test(model-gemma4): family contract |
| 12 | `65b026f902501e85afad5fd31f3ca53968a3e836` | test(model-gemma4): variant 26b contract |
| 13 | `b03c16451d65c33a05b3c495c1205240db4ccac0` | feat(data): vendored CubeCL license and provenance |
| 14 | `f491e34fe55e67eed434ac577c32d44eca51430e` | feat(data): vendor CubeCL cubecl-ir 0.11.0 |
| 15 | `627c58ec80ff7b33fa1ce5086a729946c5b1cd44` | feat(data): vendor CubeCL cubecl-cpp 0.11.0 |
| 16 | `3664f975e79242d8b1a12bc4eefe886f81a33c2f` | feat(data): vendor CubeCL cubecl-cuda 0.11.0 |
| 17 | `f77ae37ec0317f39a2e1ce4466d82c645cd344da` | feat(data): vendor CubeCL cubecl-hip 0.11.0 |
| 18 | `7d768a080a84b6132595f7cadbc84b75b69106e4` | feat(gpu): source-exact resident GPU execution and physical ownership |
| 19 | `a9723bceb8ad4aa4df639dbe660efcae6c22711b` | feat(runtime): facade, semantic program, task drivers and state publication |
| 20 | `c3d0d0b06fb1548c8bf3db10f2965e5af52be9bc` | test(runtime): gemma contract |
| 21 | `4add50a023d4fb5dd989989cab75ac4298d46d29` | test(runtime): layout api |
| 22 | `b4a36b829031fba498e45bebb71a6cd8f01e7ef9` | test(runtime): multimodal mm0 |
| 23 | `a691e21c040d2dcf46a52f2cbcf0e0eb934c8369` | test(runtime): qwen real model |
| 24 | `25ec3e27e3c208158e9763944026003005456461` | test(runtime): qwen shared model |
| 25 | `047e3b70bbcbd7acbb5176712208cc4bcc02dd29` | feat(cli): single-binary inspect and run |
| 26 | `52e37313bff57c2a7e831f50e9236fa983e13d02` | test(cli): CLI contract |
| 27 | `22e0cc5abb4a067a95324206c2881de9e3327427` | feat(tooling): workspace GPU qualification crate |
| 28 | `4d82c6704dad47875fa42bdec65f9a65de79722e` | test(tooling): independent reference oracle llama_reference_layers |
| 29 | `a2d6a02f1577554ea8dba669f77ba7b903c1bb29` | test(tooling): independent reference oracle llama_reference_logits |
| 30 | `a8ea8926b53d7b4a9c6d00f3d9be04906f8d64bb` | test(tooling): independent reference oracle llama_reference_qk |
| 31 | `6a0079eb167d1f1593fe2a9372039138cd949770` | test(tooling): independent reference oracle llama_reference_recurrent |
| 32 | `be88e30733f9459eda5f150bdbd70d9d0f8c6684` | test(tooling): independent reference oracle llama_reference_rope |
| 33 | `77778a2c258751c64c56c4ffd3dba2ff70f7296c` | test(tooling): independent reference oracle llama_reference_softmax |
| 34 | `6015241a5f5c2bb7ee65715a998988a026ff800c` | test(tooling): independent reference oracle llama_reference_value |
| 35 | `e190aff1fdd392696e8086010a7bbc1f12d3df4f` | test(tooling): independent reference oracle llama_reference_value_padding |
| 36 | `0aea8de66e3eb3b704e7d26687731a3e5b07d3bb` | test(tooling): build-release-artifacts.sh |
| 37 | `057987203d5de45ad268ba02cfc5084f76f48229` | test(tooling): check.sh |
| 38 | `c74450a27b1e3080e9576b0f95922f17c7cdca8b` | test(tooling): compare-attention-value.py |
| 39 | `9f54d2435fe6af15b3eaf28980bc1039ad02b559` | test(tooling): compare-qwen-layers.py |
| 40 | `7458ffc286f7e72c70ac18acfd3f776186aa204a` | test(tooling): cpu-qualification-gate.py |
| 41 | `73cb6e55e1795e2c13533f60621dc29619cf5e2c` | test(tooling): native-gpu.py |
| 42 | `80145a7f91724fb34a6f67e9972850a279982bf3` | test(tooling): test-install.sh |
| 43 | `ca1138b8cf866c81bbfb276358d03a62576df16a` | test(tooling): test-native-gpu.py |
| 44 | `5a0636c3f9c744c2f979ded40943a269ba5a1176` | test(tooling): test-numeric-resume.py |
| 45 | `9a6c504ebef037e09bd187d65e2bec7e2e456eef` | chore(workspace): restore the locked full workspace manifest |
| 46 | `57a9ad7fe78bf7c5e64d3870a6ba176ceba4cc07` | test(tooling): independent reference probe notes |
| 47 | `1af2ac434279cb5f4b635a51884c936f19b54aa3` | fix(nix): give module check a complete base system |
| 48 | `0452c594d99df6d3b4c972ceaae8df6ea56925e3` | docs(plan): record module eval check fix |
| 49 | `2afa32aa74aff7f35ec1b52e4caa7079a5f68980` | docs(readme): reorder engineering and FAQ sections |
| 50 | `a054d396e1ec12fc5e9b8e2ca60e8060954870c5` | docs(agents): document plan layout and TODO spec |
| 51 | `cc3cab27a2713fdb64bb3f1a42c8138c060c5dce` | docs(contributing): align guide and AI disclosure |
| 52 | `7e61cb63efd0c1429aa96044f2f7d28fda1e59aa` | docs(agents): gate pushes and destructive actions |
| 53 | `9239719c25128028fd78a6a60f817ce6b626517f` | docs: retire historical archive bodies |
| 54 | `6deebf22228315863f563723b5e6761ac6b51c5c` | refactor(cache): namespace model cache under abird/zerogravity |
| 55 | `fa6affa425089fdf9a15bfa10b330e03f738fec8` | feat(artifacts): add pinned model pulls |
| 56 | `3c97c52075563f65928c564818ce73502168778f` | feat(artifacts): resume interrupted pulls |
| 57 | `baa9d8818061958a63eabeb61a7239a103a738f0` | feat(artifacts): load pinned cache components |
| 58 | `20f449d20320b6f2912698c7b97cfdfd82472ba2` | feat(artifacts): recover missing pull sizes |
| 59 | `f44264c017856a015584a094566a077638939e55` | feat(cache): add protected snapshot removal |
| 60 | `ea700f67fca7ff7871613044a989e03b8ad6f9d7` | feat(pull)!: add capped auxiliary streaming |
| 61 | `03b69482a5fb10b8c1cc7032e7cbdf39e2e5fead` | feat(pull): pin multi-asset release components |
| 62 | `e44f4ee8020d5661dd8167cb07dfcb085906fcb7` | feat(cache): import local HF snapshots |
| 63 | `fce1353e649df2eb73fe6a39437043fc8551aef5` | feat(cache): qualify cross-filesystem imports |
| 64 | `e0073a5593e5641494c0f48e8d2dcc9e9e61a119` | feat(pull): add pinned HTTPS objects |
| 65 | `e82d76b8f8e91f7d4b49c211e26baa081ff523e5` | feat(pull): add pinned OCI GGUF layers |
| 66 | `f74a525366e6280a9806ecaabd7e1f4df46a6a97` | feat(pull): add version-pinned GCS objects |
| 67 | `833d68f716783c255b24691552a081a1d2b88e76` | feat(pull): add versioned S3 and SigV4 |
| 68 | `e62222276e0dac62ed554a3fec2f2ae86edad497` | feat(pull): add pinned Azure blobs |
| 69 | `87f6454fb7c4832262d36104a806baad9a9734a7` | feat(pull): add pinned ModelScope files |
| 70 | `223ca794c9d45a0e246ad1cd9ba8613f2f203cd4` | feat(pull): add Xet hash and chunk primitives |
| 71 | `a774ded154ed066c8f4b134ce7f5cba32ccdd09c` | feat(pull): add cloud content contracts |
| 72 | `837e77f5080c0c2c07fee90b2ef9fb791068a7d3` | fix(pull): enforce offline acquisition rules |
| 73 | `52e42fcdf0e7bead1ebefe55d8103501c7bd4f64` | feat(pull): add native Xet reconstruction |
| 74 | `46cf5a5830e7e8d8096234341ef51fafdf63cc16` | feat(pull): add conditional cloud reads |
| 75 | `4db5efbcce973d21bf1c8ca750471e6b2ab956ab` | feat(pull): recover pinned ModelScope sizes |
| 76 | `f22f75972e63ea70771c59fbce065e371d5f29d4` | feat(pull): list pinned file candidates |
| 77 | `f197c9cf4a44e06a3c1cf3c532855545bed945fa` | feat(pull): select explicit GGUF quant families |

#### How the rewritten baseline is split

- `be95424`: technical/design/research and retained acceptance/TODO records.
- `1e10b5c`: std-oriented core source identity, accelerator/program contracts,
  state/session roots, physical ResourceArbiter, events and fair device queue.
- `af9a5d3`: artifact parsing, original source allocation/identity, sharded source
  contracts and original local cache; `b276fd0`/`586d440` split artifact tests.
  The A1 accepted194-test result is retained under historical `3787519`; it is
  not an additional test result invented from these rewritten commit subjects.
- `782e6fc`: CPU math/weights. `1fd12da`/`901ba9a`: Qwen forward, checkpoint,
  tokenizer, model catalog and source-layout tests.
- `e6ecf02` and its three following test commits: Gemma config/tensor/MoE/26B
  contracts and official synthetic metadata, without trained new-family inference.
- `b03c164` through `f77ae37`: licensing/provenance plus four patched CubeCL0.11
  packages, not independent physical backend acceptance by themselves.
- `7d768a0`: all GPU implementation owners, including H1–H23 mechanisms,
  resident sessions, physical resources/events/quarantine and native backends.
- `a9723bc` through `25ec3e2`: facade plus its independent integration tests.
- `047e3b7`/`52e3731`: the sole product CLI and its contract tests.
- `22e0cc5`: GPU qualification tool; following reference/tool test commits split
  eight independent native-source probes and build/check/comparison/namespace/
  installer tools. `9a6c504` restores the locked full workspace manifest;
  `57a9ad7` retains independent reference-probe notes.
- `1af2ac4`: Nix module evaluation check needs a complete base system;
  `0452c59` records that bounded fix, not new model/GPU support.
- `2afa32a`: user-authored README engineering/FAQ ordering in its own historical
  commit. This does not authorize modifying those bytes during current work.
- `a054d39`/`cc3cab2`/`7e61cb6`: current TODO layout/contribution instructions and
  explicit push/destructive-action consent. Current runtime remains unchanged.
- `9239719`: archive retirement, described below. `6deebf2`: user cache namespace
  under `abird/zerogravity/models`, retained by subsequent A2 implementation.

### Recovery: what is really available and what is not

The following commands were tested successfully and are read-only. Use their
output for historical reading, without restoring archive files into the worktree:

```sh
git show be95424:.agents/plans/DECISION_REGISTER.md
git show be95424:.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md
git ls-tree -r --name-only 7e61cb6 -- .agents
git show 9239719 --name-status --format=
```

`git show be95424:PATH` or `git show 7e61cb6:PATH` recovers the following14
files deleted by `9239719` (their paths were verified by ls-tree):

- `.agents/prior-art/DESIGN_EVOLUTION_V3_HANDOFF_SNAPSHOT.md`
- `.agents/prior-art/DESIGN_EVOLUTION_V3_SNAPSHOT.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_ACCEPTANCE_GATES_V2.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_DISTRIBUTED_PARALLELISM_V4_1.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_EXPERIMENTS_V2.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_ROADMAP_V2.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_STAGE0_V2.md`
- `.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md`
- `.agents/research/EXTREME_MODEL_SPECIALIZATION_DESIGN_AGENT_HANDOFF_2026-10-15.md`
- `.agents/research/MULTIMODAL_PARALLEL_RESEARCH_WORKSTREAMS_2026-10-14.md`
- `.agents/research/MULTIMODAL_V5_1_RESEARCH_WORKSTREAMS_AND_AGENT_BRIEFS_2026-10-15.md`
- `.agents/research/SSD_STREAMING_FIRST_CLASS_ARCHITECTURE_AGENT_HANDOFF_2026-10-15.md`

A fifteenth retired body is the pre-product README snapshot, whose fixed path
remains present with pointer content. Thus the precise current diff is14 file
removals plus one replaced historical body, not15 deleted paths. Current build/
release/Nix guidance moved to `docs/DEVELOPMENT.md`. Its original body remains
readable at the tested `be95424` reference above.

**Corrected recovery boundary:** the former standalone full-implementation
handoff, next-agent prompt and early phase paths are absent from the current
root tree. An older complete94-source full handoff throughU040/G3b survives
inside the reachable multimodal aggregate, including several of those reports
and prompts. The exact94-source catalog and old-label chronology follow in the
recovery supplement. Later129/191-source versions and original33-node Git
ancestry were not found. The live TODO redirections are navigation, not former
body copies, and printed historical commit labels do not recreate Git objects.
The earlier draft's blanket loss claim was rejected and corrected before final
handoff acceptance.

All14 individually tested historical labels below return128/unresolvable:

`6d74f76`, `bfe672c`, `3787519`, `aeff567`, `1fca893`, `023703f`, `8c39aa8`,
`0cad60a`, `2125612`, `4ab15a4`, `602c246`, `f28dc10`, `4d1b82f`, `53b360f`.

The retained pre-product body additionally names old `b283fb8` and the decision
register names `67dca72`; these were read as historical text, not individually
validated object lookups in this audit. Do not present them as tested recovery
references. The exact original33 commit list/titles is unavailable locally;
reconstructing a synthetic list would be incorrect.

Read-only fsck exit0 finds only two unreachable **recent workflow** commits,
`e2873a799ebc06ae9b97cf601e3266f68ab06357` and
`67b16e84da9ccb566f64aceb4beffb0ccf6716bf`, plus small recent document blobs.
The reflog goes back to rewritten-era `1af2ac4`; it does not contain the original
33-commit GPU/CPU development graph. Targeted inspection of unreachable blobs
finds no old full implementation aggregate or earlier phase reports. The two
unreachable workflow commits are not an alternative source of original model
history and could be garbage-collected; no recovery ref was created.

### Historical development sequence retained by accepted plans

Primary accepted/current records:

- [Accepted baseline](../plans/accepted-baseline/TODO.md).
- [Decision register](../plans/DECISION_REGISTER.md), especially U026–055 and
  consolidated U056 onward. Older rows deliberately preserve earlier status;
  do not mistake an old NOT VERIFIED paragraph for the final accepted boundary.
- [Numerics and context](../plans/numerics-and-context/TODO.md).
- [CPU/GPU optimization](../plans/cpu-gpu-optimization/TODO.md).
- [Model execution](../plans/model-execution/TODO.md).
- [Hardware targets](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md).
- Reachable historical pre-product README body from the recovery command above.

#### Design/bootstrap and CPU qualification

The original product direction was one Rust mathematical engine with immutable
source identity, private independently owned StateRoots, one authoritative
resource/event/publish path and explicit capability-qualified specialization.
P00–P23 design, compiler/placement/serving/multimodal/SSD hypotheses and fake
contract fixtures are not completion of the later real systems they propose.
Current surviving contracts supersede obsolete archive-generator and multi-writer
publication procedures.

The earlier native CPU milestone used exact original Qwen3.5 .8B/2B GGUF bytes,
co-located official config/tokenizer and a version-pinned independent native
reference. Early5 greedy/short-output checks and selected16-row GPU projections
were explicitly partial. Two12/32-token top20 failures and incomplete long-context
gates motivated independent raw-token/per-layer work rather than looser limits.

Recorded short CPU completion:8 prompts×32 tokens per model,256/256 raw greedy
choices, zero near-ties/hard mismatches and top20 logprob errors at most0.000001;
146-vector trained comparisons per model had zero nonzero-FP32 differences and
an earlier48 signed-zero distinctions for2B. The later accepted512-source scope
records56 signed-zero distinctions; these are different scopes, not an excuse to
merge result counts or relax internal equality.

Real mixed contexts advanced from .8B1024/2B512 to4096 both, then8192 both.
The recorded .8B8192 next token7993,20/20 shared top20 and0.203729244 maximum
shared logprob error passed the functional0.45 bound. Four actual16K/32K cases
were deferred by explicit user decision: accepted Phase1 is16/16 through8K;
extended gate remains16/20, not20/20. The independent8K GDN mismatch remains
open even though this functional gate passes.

#### Partial GPU owners and first physical acceptance

| Historical step | Recorded implementation/verification | Explicit limit at that step |
| --- | --- | --- |
| Initial source Q4 probe | Original first-layer16-row Q4_0 projection on CUDA4060, HIP890M and both vendor Vulkan profiles, roughly1e-8 maximum error against independent scalar oracle | Selected kernels only; no model GPU backend |
| P07-A | Source Q4_0×dynamicQ8_0 provider; host trained tensor and completed-region whole-model mock parity; fail-closed malformed/incomplete outputs and Unsupported-only fallback | No physical full-model inference |
| P07-B | Q4_1/Q8_1, Q8_0/Q8_0, Q6_K/Q8_K and F32; four original2B tensor projections and modeled248320-logit/state equality; MM0 fake contracts5/5 | Physical GPU not yet qualified; new modalities only mocks |
| P07-C | Process-unique source/tensor/codec/geometry keyed bounded weight residency, deterministic LRU, source cache and fencing; each model three host-modeled tokens gives one upload/two hits with full logits/state equality | Activation transfers, host preparation and synchronous coordination remain |
| P07-D | Decode immutable source descriptors once; six trained source tensors over four inputs;6 event tests plus MM06 barge-in mock (6/12 mock track) | Event protocol not yet actual async GPU worker |
| P07-E | Real CubeCL read_async bridge, prelaunch lease/event admission, finite/exact readback before Ready;5 source-free bridge tests and original-model host mocks | Physical readback/cancellation not yet verified |
| P07-F (old602c246) | Original source/dynamic/output GPU handles retained before launch and process-quarantined on uncertainty; combined cached-weight/transient cap;10 source-free and8 source-real host tests | Device execution then unavailable; no global worker integration yet |
| P07-G1 (old4d1b82f/f28dc10 labels) | ResourceArbiter physical-domain admission and bounded FairDeviceQueue;4 resource+4 queue tests | Eight control-plane tests are not physical GPU proof |
| P07-G2a | Attested PCI/LUID/registry domain and provider caps bound to same process-global arbiter; uncertain readback fences that device | Physical backend qualification still separate |
| P07-G2b | Real two-model NVIDIA Vulkan queue:4 physical completions,2 reused weights, exact selected source regions and one-token whole-model hybrid logits/state; AMD physically runs but strict FP32 differs and refuses StateRoot | AMD initially numerically unqualified, no complete GPU model |
| P07-G3a (old67dca72 text) | Explicit WideF64 makes original Q4_0 regions bit-exact on NVIDIA3060 and AMD Renoir Vulkan; four-token hybrid AMD parity; fair queue4/4; forced Future abandonment retains5 handles/256MiB, fences device and leaves root unchanged on both | NativeF32 diagnostic fails AMD; no driver recovery or full graph |
| U039 adversarial review | Pre-submit invalid input refusal avoids false terminal fence; scratch-aware idle LRU; one-descriptor GGUF parse/copy; CPU intermediate prefill elides discarded logits; selected/completed counters fixed; huge geometry and allocation hot-path guards | Actual trained-source/abort checks, not a scheduler/performance rewrite |
| P07-G3b | Arc<LoadedQwenModel>/ModelCatalog explicit alias/dedup/revoke; two independent sessions share one upload and two actual source-exact readbacks; advisory driver telemetry separated from logical charge | No full GPU GDN/attention/KV then; no global physical-free assertion |

Current owners: GPU `q4_q8.rs`, `mixed_ops.rs`, `host_descriptor.rs`,
`residency.rs`, `async_bridge.rs`, `physical_quarantine.rs`,
`physical_service.rs`, `physical_queue.rs`, `model_workers.rs`,
`model_sessions.rs`; core `resource_arbiter.rs`, `device_events.rs`,
`device_queue.rs`; Qwen `model_catalog.rs`. These sources are present in the
rewritten owner commits, not separate recoverable historical P07 commits.

#### H1–H25 progression into resident and native execution

This table distinguishes recorded earlier results from source-label evidence.
Original H-phase reports/commits are not locally recoverable; do not invent their
exact titles, IDs, total test counts or per-phase performance measurements.

| Step | Surviving mechanism/result | Evidence and qualification limit |
| --- | --- | --- |
| H1 | All187 original GGUF GEMV regions source-qualified on both original models and both Vulkan vendors, full logits/root | Decision U041; not full remaining operators |
| H2 | Source RMS rounding (f32 square, f64 sum, f32 mean/sqrt/reciprocal),187 matrices+49 norms=236 regions each model/card | Decision U041; independent GDN/attention still pending then |
| H3 (old4ab15a4) | Grouped per-head norms;530 actual completions over2tokens=266 selected/token minus2 final-only ops; complete248320 logits/state exact and zero outstanding logical reservation | U044; initial counter double-count fixed, no arithmetic relaxation; separate forced abandonment5 handles/256MiB fenced |
| H4 (old0cad60a) | GDN causal convolution raw pre-SiLU plus private3-frame history; source rounding fixed by stored FP32 products then ordered reduction | U045: one kernel failed1840/6144 second-token raw values with history exact; two-kernel fix yields566 actual completions/2tokens,36 raw/history checks and exact full logits/root both models/cards; forced abort6 handles/256MiB |
| H5 | GDN recurrent output AND proposed state on nonzero causal tokens; physical rounding boundaries | U047 and current `gdn_recurrent.rs`; source/current accepted completion, original standalone counts unavailable |
| H6 | GDN Q/K L2 normalization, rollback/admission/status/quarantine | U047 continuation/current `model_layer.rs`; no fabricated original hash/count |
| H7 | Polynomial source SiLU/output gate | U050/current `silu.rs` and model-layer source-bit arithmetic cases |
| H8 | Source GDN nonlinear coefficients exp/ln with exact rounding | U051/current `gdn_coefficients.rs`; stress source bits, no blanket libm portability claim |
| H9 | Attention RoPE and private current KV, nonzero positions/FMA rounding | U052/current `rope_kv.rs`; position32768 operator stress is not a32K model context |
| H10 | Bounded source QK scoring over private prefixes | U053/current `attention_qk.rs`; constructed prefix4096 operator scope is not whole-model context |
| H11 | Padded source softmax, exact exp/reductions/two-round normalization | U054/current `attention_softmax.rs`; constructed rows not long-model evidence |
| H12 | Padded value accumulation and independently proven signed-zero corner in prior CPU valid-only schedule | U055/current `attention_value.rs`; no tolerance relaxation |
| H13 | Remaining binary/residual/embedding operators | Current `operator_stress.rs` retains H13 marker; accepted baseline includes complete embedding/FFN math, original phase-specific gate count unavailable |
| H14 | Device dynamic activation quantization, nine source stress cohorts/codecs, no intermediate model readback | `activation_quant.rs`, `device_activation.rs`, `activation_stress.rs`; forced finite-input/nonzero status probe retains12 handles and fences root publication |
| H15 | Exact original phase title/ID not preserved locally | Compressed embeddings are accepted and implemented, but this audit does not assign that mechanism to H15 without a surviving label |
| H16 | One ForwardPlan binds resident values/leases, complete private state, GPU token feedback, exact full logits/state/ordered trace, restore/reset/stale-ready and independent sessions | `resident_program.rs`, `resident_inference/qualification.rs`; compact16-byte greedy feedback, no repeated source upload/intermediate activation transfer; accepted phase scope not general serving |
| H17 | State-only prefix execution and atomic/segmented prefill, skip output-only allocations/work, one final publication | `resident_prefill_qualification.rs`; original math schedule reused |
| H18 | Genuine continuous512 consumed tokens, milestones and254→257 replay across attention boundary; all42 state domains and full248320 logits; cap refuses beyond512 without root/admission change | `resident_context_qualification.rs`; snapshots survive provider with charge, final release zero; this is actual context acceptance distinct from synthetic long positions |
| H19 | Packed exact whole-range Contiguous aliases eliminate redundant physical values with one lease per original | `resident_alias_qualification.rs`; physical allocation audit, not general strided provenance |
| H20 | Cold prepared ranges/guard equivalence, malformed ranges refuse without root/admission mutation | `resident_range_qualification.rs`; timing threshold none, no model throughput claim |
| H21 | Portable v1 CPU/file/GPU state replay, exact42 domains/full logits/source/capacity/digest/Ready guards; actual upload-abort quarantine | `resident_checkpoint*.rs`; trusted diagnostic wire, not authenticated production persistence |
| H22 | Fused dense View bit-copy/certificate equivalence over original handles, guards/gaps/status/raw nonfinite bits | `resident_view_qualification.rs`; no root publication in fixture, general strided View remains future |
| H23 | Prepared projection bindings compare all10 handle coordinates to frozen H22/typed nodes, bounded host metadata | `resident_projection_qualification.rs`; host binding timing only, no GPU allocations/upload or throughput claim |
| H24 | Native CUDA/HIP CPP NVRTC/HIPRTC enablement and hardware profiles | Accepted baseline/hardware/native README; standalone LLVM code generation remains unqualified |
| H25 (old8c39aa8) | Promote reusable implementation into nine-package workspace/one lock/version/sole CLI, qualification under tools; frozen release-profile regression on five backend profiles | `RUST_WORKSPACE_AND_ENGINEERING.md`; no independent probes product; new-family libraries are source contracts |

Final accepted native/resident boundary: local RTX3060 CUDA, remote a1RTX4060
CUDA, a1gfx1150 HIP, x2gfx1151 HIP, and SDK-free NVIDIA Vulkan: **five profiles
on four healthy physical GPUs**. The fifth inventory GPU (local Renoir/gfx90c)
is held. Accepted135 profile checks+12 host fingerprints must not be summed with
earlier step checks as if they were disjoint. Separate main model merge recorded
135 ordinary Rust executions,23 original-model executions across22 distinct names,
two exact8-token CLI outputs and one official metadata case.

Native providers use the existing CubeCL C++ path and existing runtime/compiler
libraries/headers. Separate LLVM23 device codegen remains unqualified. The a1
HIP profile retains its measured glibc2.44/GCC16.2 loader/scalar/libm closure.
Current packaging release profile differs from the old independent probe;
actual newly packaged binaries were separately qualified, not inferred from old
binary hashes or compilation.

#### Hardware discovery and local AMD hold

Historical target inventory: three CPUs/five physical GPUs, not five accepted
GPU cards. Discovery is a capability inventory, not measured available memory
for current admission or live machine state.

| Host/profile | Retained discovery and boundary |
| --- | --- |
| Current CPU | Ryzen7 5800H,8C/16T,NUMA1,AVX2/FMA/F16C, noAVX512; L1d32KiB/L2512KiB/L316MiB |
| a1 CPU | RyzenAI9 HX370,12C/24T,4fast+8slow,NUMA1; AVX512/VNNI/BF16 capabilities; cache hierarchy48KiB/1MiB/16+8MiB |
| x2 CPU | RyzenAI MAX+395,16C/32T,NUMA1,48KiB/1MiB/2×32MiB; capability-specific optimization future |
| Current NVIDIA | RTX3060 Laptop sm86,6144MiB, PCI01:00.0; accepted CUDA/NVIDIA Vulkan profiles |
| Current AMD | Renoir gfx90c,8CU/wave64/shared RAM; later2B initialization SIGABRT and automated resets wedged; **no launches/resets/recovery experiments authorized now** |
| a1 NVIDIA | RTX4060 Laptop sm89,8188MiB,PCI64:00.0; accepted CUDA |
| a1 AMD | Radeon890M gfx1150,16CU/wave32,PCI66:00.0;512MiB aperture+15.23GiB GTT are NOT additive independent memory; accepted measured HIP closure |
| x2 AMD | Radeon8060S gfx1151,40CU/wave32,PCIc6:00.0; nominal64GiB shared pool/observed67.04GB decimal host; accepted HIP |

Remote AMD Mesa26.1.8/Vulkan1.4.354 and NVIDIA595.91.07/Vulkan1.4.329 advertised
features are discovery facts, not qualifications for every precision/shape.
Local AMD was physically used in earlier Vulkan phases but its subsequent hold
supersedes those historical examples. No current documentation task authorizes
using it or changing device/driver privileges. SSH discovery/temporary historical
qualification also does not authorize future downloads or persistent config.

#### Layout, new-family contracts and concurrency disposition

Nine-package layout: core, artifacts, CPU, Qwen3.5, Gemma4, GPU, facade, sole CLI,
and GPU qualification tooling. Patched CubeCL packages remain excluded vendor
members; family/text/CPU dependencies are absent from production GPU. Curated
`all-models` describes both family contracts, not executable support for every
checkpoint. One workspace version/lock; downstream patch declarations are not
transitive. Source/feature changes need their relevant fresh gates.

Qwen3.8/Gemma4 implementation is source metadata/geometry/tensor/routing contracts
only. Recorded Gemma family14 tests and official metadata do not establish trained
weights/tokenizers/CPU or GPU inference. Current owned GGUF executor retains4GiB
cap; larger BF16 sources exceed the local accepted source budget. Shard/index/
header/name/dtype/offset/shape/fullhash/admitted-owned-copy path remains separate
unfinished work; mutable mmap is not an accepted shortcut.

Thoughtful left-right review followed layout. The cold model-catalog proposal had
no demonstrated contention and was restored, with **no accepted migration or
speedup**. Future contention/publication/replay/reader-retirement evidence is
required. Immutable Arc sharing and existing narrow locks remain accepted.
No new dependency or unnecessary double-buffer publication was added to fill a
plan checkbox.

### Tracking/archive/checkpoint chronology

Historical checkpoint labels below are accepted record labels, not currently
resolvable commits. Reachable replacements/current bodies were checked separately.

1. Older generated full handoff evolved from89 embedded source records to an
   aggregate duplicating191 individual sources. User later explicitly replaced
   that process with plain TODOs; do not resurrect its generators/manifests/SHA
   inventories or impose obsolete byte-receipt duties on future implementation.
2. Historical `bfe672c` simple-TODO migration: single index/per-plan/agent-run
   hierarchy;5 archive/generator/SHA tools and4 check.sh invocations removed while
   every real implementation check was preserved. Retained final full gate194
   pass/27ignored, strict lint/release/installer/shell/Nix/reference plus18 namespace
   fixtures. Initial7 stale pre-layout links fixed;44 live links in13 workflow
   documents resolved. No implementation/dependency/root-README delta.
3. Historical `aeff567`: retire70 obsolete Markdown records:42 GPU phases/notes,
   20 CPU/model/concurrency/checkpoint reports,6 entrypoint aliases and2 takeover/
   layout records. Preserve source rounding/quantization/padding/new-family facts
   in technical docs and current TODOs; keep8K/schedules/deadlines/recovery/actual
   memory/View provenance limits open. Rejected-LR metadata JSON removed after
   ownership review; no model/SDK/build cache cleanup inferred. Full gate194/
   27ignored+18 namespace;114 current first-party document links and GPU144links/
   23documents pass.15 then-retained archive bodies matched the old checkpoint.
4. Current graph begins after those reports were removed; `be95424` already holds
   simplified current documents plus the remaining15 old design/brief bodies.
5. Reachable `a054d39`/`cc3cab2`/`7e61cb6` strengthen TODO/contribution/consent
   workflow. Reflog superseded two earlier workflow objects only.
6. Reachable `9239719` retires those remaining14 files plus replaced pre-product
   body after distinct rationale/open tests are integrated. Keep SSD01..32,
   QTIP/HYB/GFX, MM01..24 and77 family fixture requirements **planned**. One gate
   failure from a copied CLI test's embedded retired checkout was resolved by
   test-source mtime refresh with bytes unchanged. Final full gate passes; no
   additional result count invented where this section records none.
7. Current acquisition completed sections are plain Markdown archives, not a new
   evidence system: through-fce1353 holds10 completed groups; through-a774ded
   holds8 more; all unfinished tasks stay live. Rolling prior-commit TODO updates
   are carried with the next implementation commit, not separate self-reference
   commits. Runtime/user pause at current quant acceptance remains effective.

### Acquisition chronology and accepted gates

Sources read in full (bounded chunks):

- [First completed TODO archive](../plans/model-acquisition/archive/through-fce1353/TODO.md).
- [Second completed TODO archive](../plans/model-acquisition/archive/through-a774ded/TODO.md).
- [Live acquisition TODO](../plans/model-acquisition/TODO.md).
- [Acquisition design/frontier](../plans/model-acquisition/README.md).

Each row below is an implementation phase. Full-gate counts are actual Rust
**executions across feature configurations**, not distinct test names, and are
not additive across rows. Optional/child/helper ignores are intentional; ignore
counts do not mean those cases passed. Separate relevant physical/model selections
were run explicitly where stated. All final gates listed have0 failures.

| Commit/phase | Resulting behavior and mandatory boundary | Recorded complete frozen gate |
| --- | --- | --- |
| old3787519 / rewritten cache owners | A1 local Linux exact schema/path/admission/staging/fullhash/EOF/atomic manifest/ref publication. Trusted source identity/inference remains separate.18 operation types/38 callbacks exercised76 injected I/O+76 actual child exits; retained ref backing/private snapshot peak/hardlink ctime fixed. Cooperative quota is not global host/UMA budget; sameUID mutation not prevented. |194 pass/27ignored |
| fa6affa | A2 optional synchronous HTTPS owner with TLS/origin credentials/limits/retries/deadlines; pinned HF/GH commit/tree or release/asset; required shards/index/splits/config/tokenizer/sidecars; unknown raw digests first-seen fullhash. Offline-before-client; no default CPU HTTP/TLS. ActualTLS/range/redaction fixtures,40-boundary discovery publication matrix. |312/30ignored |
| 3c97c52 | Durable **proper** exact prefixes bound to immutable source/length/digest/strongETag/inode/prefixSHA; rehash+copy using original attempt/publisher.8MiB/1s checkpoints; file-before-atomic-record order. Exact-total prefix cannot bypass terminal EOF; zero-progress retry does not duplicate retained copies. Original dead stages remain charged. Full old+new+24KiB/file peak and one digest lock. Progress separates cached/resumed/downloaded; signals optional. |353/34ignored |
| baa9d88 | Explicit independently pinned component snapshots/role suppliers/per-label transports and cumulative budgets, no atomic composite source. Descriptor-based actual-owned GGUF copy hashes/reparses before sole issuer;4GiB cap and exact raw ImmutableWeights admission/lifetime. Positioned bounded config/tokenizer copies1MiB/64MiB and same Qwen math. HTTP-free run-cache/offline pure opening. |419/54ignored/75suites |
| 20f449d | Missing selected exact sizes from pinned LFS or body-free exact HEAD; missing≠zero, strongHEADvalidator retained. No transport-error GET fallback. Concurrent fill typed verified-hit under same locks avoids redundant payload. Cancellation reaches transaction hash/control-copy and online final open; legacy wrappers remain. |449/54ignored/81suites |
| f44264c | Fresh-root protected policyv2 actual OFD shared pins, whole-transaction activity barrier, safe exact unreferenced snapshot retirement/restart. Existingv1 byte/charge compatibility, no migration/removal. Typed Busy/Referenced/Durable/Incomplete and named-inode charge, no global physical-free claim.34 I/O+34 actual exits; independent charge/lazy/dup/inheritedFD tests. |518/56ignored/94suites |
| ea700f6 | Explicit positive AtMost for Auxiliary only, fullcap admission and max per trusted-digest cohort; exact observed manifest/receipt bytes at terminalEOF. No cap durable resume. HEAD fallback only validated200missingCL/405/501; cap+1 custom/HTTP read guard. Required/model closure stays exact. Intentional exact getters→Option API change. |553/57ignored/103suites |
| 03b6948 | Repeated release component aliases reuse typed first releaseID per provider/repo/tag; per-label asset listing/credentials and all cumulative budgets persist. Pure offline allows distinct assetIDs only same canonical releaseID. ReleaseID does not freeze mutable asset set or atomic cross-asset membership.147 scoped+14 independent executions. |569/57ignored/106suites |
| e44f4ee | Explicit local HF root/repo40commit/path/role/import. Readonly originalFDs, stable pre/postcopy checks; metadata-only source topology before any same-root destination initialization; no source locks/writes/hardlinks/refs. Persist local-hf-cache provenance, FirstSeen unlesscallerSHA. Knownhit still hashes source with copied0. Existing absolute0700 target only. |625/58ignored/116suites |
| fce1353 | Crossfs only distinct-dev+mount Btrfs↔tmpfs and distincttmpfs↔tmpfs, each tree/selectedFD on own dev+mount, ancestry/root/intersection checks. Complete source metadata inventory/hardlinks; sameUID races remain operation-local. Real6diagnostic+5production physical selections, not hiddenbind/subvolume/overlay proof. tmpfs fsync not reboot persistence. |633/68ignored/119suites |
| e0073a5 | URL-free HttpsObjectSpec: normalized bareorigin+logicalpath/role/exactbytes/callerSHA contract, secretURL/query onlymemory.1MiB admitted prepare before parse; noHEAD. Pure offline managed one-entry fullhash/trust/source/pointer check before URL/env/client. CLI bounded nofollow URL-file, no ref firstslice.160 API+35 independent focused. |653/68ignored/126suites |
| e82d76b | Native OCI/Docker schema2 unique/explicit rawGGUF descriptor, rawmanifestSHA/media/digest/exactsize checks. Explicit anonymous/bearer/fixedDockerHub pull broker; no arbitrarychallenge/login/refresh/container/index/compressedpaths. One originalHTTP/executor/publisher. Offline no independent retained-manifest membership/freshness claim. |672/68ignored/133suites |
| f74a525 | GCS JSON object exactgeneration+metageneration/size, locally derived querypin+precondition; optional-present generationheaders must agree. Media nofollow prevents query-loss. Identityencoding/keys/softdelete refusal; callerSHA orFirstSeen. BoundedAux prefix independent plans/noatomicbucket. Typedpins/receipts and pureoffline. |693/68ignored/140suites |
| 833d68f | Nonnullversioned S3 native/explicit compatible endpoints, boundedV2/versionlist, static/session SigV4. Opaque exactURI typedtarget+defaultUnsupported customtransport capabilities, zero redirects. Mandatoryexactversionheader/no-delete beforeeverybody, independentETag/range/hashEOF. Officialliteralvectors+actualwire signatures/rawpaths. |718/68ignored/147suites |
| e622222 | Native Azure BlockBlob explicitVersion/Snapshot, exactHEAD+transientETag, typed source/offlinepin. Mandatorytype/encoding/structured/key/copyguards; optionalpresentversionecho. Public/bearer/accountSharedKey/serviceSAS boundedprofile/API2023-11-03; rawSAScredentialowner only. Strict XML currentversionAux prefix and aggregate control tally. |749/68ignored/154suites |
| 87f6454 | ModelScope explicit40commit direct native boundedshallowtree/files and officialtoken Bearer+cookie in one originalHTTPowner. Nofollow/credentialprofile isolation; exactrawSHAifprovided elseFirstSeen. Source membership/revision serversemantics trusted, no aliasauthority. Digestofflinepointer refusal fixed RED→GREEN. |768/68ignored/161suites |
| 223ca79 | Pure then-unwired Xet32byte/text/keyedMerkle/hash/codec primitives and official/scalar/framegoldens. NONE/LZ4/BG4 boundeddecode/cancel; native transfer notyet enabled.13 actualprimitive tests before/after mechanicallint. |781/68ignored/164suites |
| a774ded | Explicit S3/Azure current-object **callerSHA+exactbytes** contentcontracts; nofakeversion, noHEAD during admittedprepare/dryrun. Originaltypedtargets/auth/executor/resume/fullhashEOF/publisher; distinctsource/offlineTrustedcontract. AzureSAS b/c currentcontent whileoldbv/bs scopesretained.488 focused/all4strict, nofix. |802/68ignored/171suites |
| 837e77f | Legacyoffline pointer/closure/indexbudget eligibility:5 intendedRED+3positives beforecode. Shared128B managedprefix; legacy ordinary Files role-agnostic semantics retained. ExplicitcomponentWeights enforce complete split/index/sidecars, one cumulative index-control budget.215focused/all4strict. |810/68ignored/174suites |
| 52e42fc | Explicit nativeHF Xet through same pin/tree/closure/HTTP/executor. SealedHub/CAS/signedinterval grants, scopedexpiry/nofollow/credential suppression; completeordered/repeatedterms/multipart/codec/EOF/filekeyedroot+rawSHA. Invocation-wide work/wire/control/decode/leaf/refresh caps+admittedscratch. Nativeephemeral fullrestart; ordinarysidecars retainresume.551artifact+175CLI/all4strict. |867/69ignored/181suites |
| 46cf5a5 | Explicit no-callerSHA conditionalcloud: admittedHEADexactbytes+strongETag, typedIf-Match signed on everyinitial/range/retry and exactresponseETag beforebody;412neverrefresh/fallback. FirstSeenactualrawhash, noimmutableversionclaim; pureofflinepin+snapshotDIGEST beforesecrets.872focused/all4strict. |886/69ignored/188suites |
| 4db5efb | ModelScope selected/index absentSize uses exactsamecommit nofollowHEADonce under originalcumulative metadataBudget. MalformedPRESENT/null stillrefuses,unselectedneverHEAD; strongETag checks,weak/absent noresume.891focused/all4strict; stalebuild resolutionbelow. |896/69ignored/191suites |
| f22f759 | Metadata-only HF/GH candidates through originalpinnedtree, noHEAD/index/payload/publication. Fullinventorycollision beforeprojection; observedsize provenance and cachedsubset scopehonest.943focused/all4strict; staticcollision fixbeforefirstexecution. |915/69ignored/198suites |
| f197c9c | ExplicitGGUF filenamequant withinsameoriginaltree/pin/closure/nativebudget; boundedASCII1..32tag, terminalbasename boundaries, canonical5digitcomplete splits beforeuniqueness; nofallback/eligibility. Strictnewquantoffline roles/source/fullhash/pointers/DIGEST beforesecrets.975focused/all4strict; nativequantfixture planonly. |934/69ignored/205suites |

All full phases use the isolated frozen command:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh
```

This includes feature/dependency isolation, formatting/strict lint/build/release/
installer,18 namespace-wrapper tests and comparator/resume self-tests as recorded
per phase. Actual Nix filtered-source integrity and local-link/rootREADME checks
were separately run; new retained fixture paths were explicitly curated. Fixture
HTTP/TLS/cloud traffic is owned local synthetic traffic, not live provider/account
or externally downloaded model qualification.

#### Original-model regression route ledger

Both approved model hashes remain:

- .8B `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`.
- 2B `91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6`.

A3 and later source tests use both originals,2tokens/model/nonzero state,
all248320 logits and all42 full state-domain vectors **bit-for-bit**, independent
sessions and immutable source allocation IDs, foreign-source restore refusal,
rawsource lease lifetime until last owner, ref/private-cache mutation isolation.
Protected-removal acceptance additionally removes the private cached snapshot/
weight while the owned model and its reservation survive and remain numerically
exact. HF-import route additionally copies into a synthetic explicit HF layout,
imports exact known hashes, removes the synthetic source before preparation and
reuses the same numerical/state/ownership tail. This does not requalify8K or a
new family/GPU filesystem execution path.

The original selected release test is
`approved_cached_models_match_direct_source_exactly`. The second is
`approved_hf_import_models_match_direct_source_exactly`, both in
`crates/zerogravity/tests/cache_prepare.rs`.

| Phase | Actual recorded selected release result/duration |
| --- | --- |
| baa9d88 | Both original models accepted; separate final selected test18.49s recorded in archive |
| 20f449d | Original cache route1test21.61s |
| f44264c | Enhanced actualremoval route1test19.81s |
| ea700f6 | Original route1test20.02s;43.49s prior compilation separate |
| 03b6948 | Original route1test20.03s;cachedcompile0.14s |
| e44f4ee | Bothroutes2tests45.15s;releasecompile45.18s separate |
| fce1353 | Bothroutes2tests42.20s;releasecompile45.24s separate |
| e0073a5 | Bothroutes2tests40.55s;preparation0.14s separate |
| e82d76b | Bothroutes2tests41.86s;compile0.15s separate |
| f74a525 | Bothroutes2tests41.83s |
| 833d68f | Bothroutes2tests41.06s |
| e622222 | Bothroutes2tests42.85s |
| 87f6454 | Bothroutes2tests42.91s |
| 223ca79 | Bothroutes2tests41.70s |
| a774ded | Bothroutes2tests40.55s |
| 837e77f | Bothroutes2tests42.34s |
| 52e42fc | Bothroutes2tests43.26s;44.87s rebuild separate |
| 46cf5a5 | Separateexactcalls17.40s and23.69s;not onecombined invocation |
| 4db5efb | Separateexactcalls17.64s and23.60s |
| f22f759 | Separateexactcalls17.47s and23.65s |
| f197c9c | Separateexactcalls17.60s and23.70s |

Recent exact commands (one invocation per NAME, accurately report each timing):

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py --   cargo test --release --locked --offline -p zerogravity --features cache   --test cache_prepare approved_cached_models_match_direct_source_exactly   -- --ignored --exact --nocapture
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py --   cargo test --release --locked --offline -p zerogravity --features cache   --test cache_prepare approved_hf_import_models_match_direct_source_exactly   -- --ignored --exact --nocapture
```

Earlier combined routes were selected with
`--test cache_prepare -- --ignored --test-threads=1 --nocapture` under the same
wrapper, not a fabricated single timing for the later filtered commands.
These are historical commands; **do not run while implementation is paused**.

#### Substantive failures/corrections that explain acceptance

- A2 transport initially misinterpreted whole-body timeouts and allowed TRACE
  URL/Location exposure. Header/total deadlines were separated; acquisition caps
  logTRACE and documents downstreamfeature constraint. A first gate stopped130
  for reviewed redaction; a next127 came from editing its active Bash file.
  Final script frozen and full312 passed;40-boundary matrix superseded an early
  incomplete four-boundary experiment.
- Durable-prefix review found terminalEOF bypass at savedbytes==total, repeated
  unchanged checkpoint copies, canceled returned chunk losing prior accepted
  prefix, and public known-digest private-copy quota escape. Fixes preserve
  proper-prefix-only checkpoints, original old+new peak, explicit checkpoints,
  accepted-prefix retention and stable digest authority.13cachepartial tests,
  16 I/O+16 real exits across8 initial/replaced-checkpoint boundaries pass.
- A3 review rejected publicly mutable geometry as authorization; normal/cache
  family constructors reuse validated config and reject ineligible family before
  rawallocation. Exact bounded owned-copy actualSHA/EOF/reparse precedes issuer.
- Protected-removal first fullgate intermittent absent-target retry was Busy at
  callback19 because unrelated test-thread childspawn inherited allprocess OFDs
  beforeexec. A test-only first-acquired/last-dropped mutex serializes the unrelated
  process fixtures;10 parallel reruns pass. No productionretry/timing weakening.
  Actualnamedcharge is not globalphysicalfree-space attestation.
- Cappedstream tests exposed custombody overread needing cap+1 clamp, and a final
  same-byte progress event was required onlyafterEOF/filefinish. Fixtures using
  `.bin` correctly triggeredWeights classification and were madeAux`.txt`; no
  rolepolicy weakened. No independent fullHTTPtrailers validation was invented.
- LocalHF initial2source failures were42-character supposed40hex fixture ID;
  constant/chain fixture corrected. Independent socket fixture exceededSUN_LEN;
  bind shortprivatepath thenrename retainedsame specialinode case. Two test-only
  default-field assignments fixedlint; no sourceacceptance relaxation.
- OCI CLI syntheticseed omitted TrustedSha256; one genuinefailure/two poisoned
  mutex cascades fixedprovenancefixtureonly. Signing attempt withoutSSHagent
  exit128 had nocommit; existingagentsocket succeeded, signing neverdisabled.
- S3 XML review caught outside-root references/repeateddeclaration/XML1.0invalid
  chars beforefirstcompile. Diagnostic originparser wronglyrejecting HTTPS was
  repaired beforeexecution. OneMaxKeys collapsible_if lint and onehelpcharacter
  failure+three mutexpoison cascades received mechanicalrepairs only.
- Azure firstcompile101 beforetests: wrongexistingcheck_cancel path and equivalent
  quickXML deprecatedwrapper. Subsequent shortbodyfixture omitted two legitimate
  originalrange retries; corrected scriptedrows, notruntime retry policy. Wrapped
  helpassertion+three mutexcascades fixed whitespace-normalizedtest; oneinclusive
  svwindow lint mechanical. Wirecanonical/HMAC expectations unchanged.
- ModelScope initialindex positive lacked mandatorytotal_size; actual-loopback
  tests neededdiagnostic-only cfg. Correct-hash manuallyseededLFS pointer first
  failed atintendednegative, thenmanagedpositionedprefix guard passed. No general
  CacheStoreliteral-byte validity was redefined as loadereligibility.
- PureXet independentgoldengenerator copied65/66hex XOFcharacters into32byte
  expectations beforeRust; fixedofficialprefixes first, verifiedofficialvectors,
  frozecustomgoldens. Official0..31textexample omitted0x10; correctsequence plus
  verbatimsavedsource vector retained. No expectedresult fittedtoRustoutput.
- NativeXet firstE0624 testaccessfixedtestonly; teeouterexit0 hadhiddenCargo101,
  henceinspectactualstatus. A payload.binauxfixture invokedrealmodelclosure:
 19failures correctedvalidAuxpath pluscompleteWeightsclosurepositive. Remaining
  StatevsLimit sanitizedexpectation and CLIgrammar correctedtestsonly. Ordinary
  sidecar retry restriction foundbyroot narrowedtoallowOrdinaryPayload only.
- ModelScope sizing firstdiagnostic run156librarypass/new1pass9fail contained an
  oldmissing-size literal absentfromlivesource butpresentin retainedrlib. Root's
  SIDEcopy2 preservedmtime olderthanpreviousconditionalbuild. Afterbyte-equality,
  only5mergedfile mtimesrefreshed; identical453pass rerun. No sourcepolicy/fixture
  expectationchanged; futuremergesusefreshmtimecopyfile, notworkspacecargo clean.
- Candidates fullinventoryregular-ancestor collision hiddenbyprojection caught
  beforeexecution; orderingfixedwithHF/GHnegatives/real-directorypositive.
  Quantdirectory`-of-`/tag counterexample repairedbybasename-onlyinspection before
  execution. Both finalphases report **no executedRED/sourcefix** fromthese static
  findings. Quantinterruption foundall11frozenMAINpathsunchanged.

### Still-open limits, not erased by accepted chronology

- CPU independent8KGDN numerical root cause: first .8Blayer4/2Blayer1 mismatch;
  nativeubatch512 sharedtop20 logprob errors0.20372924/0.19678696 remain. Greedy
  token7993 matching does not close it.2B native schedule changes; .8B schedule
  control differs. Compare pinned native0.6.0/build11429/d812350 CPU/F32KV/nonflash,
  rawteacherprefix512/1024/2048/4096/8192 andubatch1/64/256/512. Fullubatch1historical
  capture unfinished. Keep0.45sharedtop20/0.05toptokenprob functionalcriteria and
  strict sourcebitoperatorcriteria separate, never loosen to claim parity.
- ResidentGPU beyond512, genuine16K/32K, additionalmodels and five-device fleet
  optimization are unfinished. Synthetic32768positions/4096paddedrows are operator
  tests. Localgfx90c hold persists. NativeF32FMA/AVX2/AVX512/VNNI/specializedpaths
  and realthroughput gains remain separatelymeasured proposals; WideF64 accepted.
- Actualdriver cancellation/recovery/physicalfinalfree and multistream/deadline
  scheduling remainopen. Logicalfence/quarantine successes do not imply recovery;
  queueddeadline wakeups requireactualscheduling/poll opportunities.
- Trustedportablev1 replay is notauthcheckpoint persistence; tokenizer/sampler/
  grammar/media serialization/security UX remainsopen. ModelcatalogLRcontention
  notdemonstrated; noacceptedLR migration/performanceclaim.
- CloudcurrentcallerSHA, conditionalFirstSeen and versionedprofiles have distinct
  contracts. Broaderendpoints/partitions/ambientcredentials/checksumauthority/
  Azurelatest/SASprofiles/prefixdigestmaps stillopen; noETag/MD5/CRC/GitOID upgrade
  to trustedrawSHA. S3FULL_OBJECTchecksum qualification separate.
- ModelScope aliases/redirectCDN/mirror/refresh/nativehubparity remainopen; no
  immutablecommitauthority from callerrevisionecho/timestamp/filelastmodified.
- NativeXet requiredLFS/Xetindex302stillUnsupported; ordinaryindexworks. Native
  partial-xorb/pertermproof notinvented; nofulltrailersection/wholeoperationdeadline
  orcoalescing/throughputclaim. Signed-setoverfetch/refetch countsactualwork.
  Nativeprocessrestart retransfersfullnativepayload, ordinarydurableresume separate.
- v1cachemigration, deadstage/orphan/generalGC, furtherlazy/raw/mmapconsumerpins,
  unmanagedreaderremoval and additionalfilesystempairs remainUnsupported/unqualified.
  ManagedFDpin and observedunique-inodenamespacecharge do not provekernel/process-
  globalphysicalfree. Actual hiddenbind/subvolume/overlay fixtures neverrun;
  authorizing ordinaryexistingtmpfsfixtures did not authorize mounting.
- LocalHFimport is explicitreadonlyexternalcache selectedcopies, noimplicit
  discovery/closure/modelreadiness. SameUID stability checks areoperation-local,
  notanatomicexternal-tree snapshot. Additionallayouts/newdestinationcreation/
  missingmountidentities remainopen. Localoriginassertion isnotremoteHFmembership.
- Finiteprovider/loaderparity remainsopen despite many accepted slices. Newsource
  hashes/licenseauthorization do not authorize opaquepluginruntime or trained
  newfamily inference; noentirellama.cpp/vLLMunioncompletionclaim.

### Paused source drafts that must survive handoff

1. `/home/pvl/spaces/abird/src/zerogravity-filters-work`: C3 base4db5efb plus
   acceptedC1/C2overlays, onlynew`acquisition/filters.rs` differs. UNWIRED and NOT
   compile-ready; RusttestsUNRUN;8integrationpaths unchangedC2,2testfilesabsent.
   Preserve73primaryoraclecases+33345pairs. GrantedgrammarFullpathFnmatchCaseV1,
   includeunion→exclude→ambiguity, requiredveto/optionalskip; adjacentindex checked
   beforeGET.512KiBscratch/16Miwork andextra16MiBsameowneradmission stillneedactual
   integration/qualification. Formatting/whitespace success isnotRustacceptance.
2. `/home/pvl/spaces/abird/src/zerogravity-xet-index-work`: C4 basef22f759+frozenC2;
   onlyHF and`xet/{mod,budget}.rs` drafted. Explicitdefaultfalsebuilder/private
   callback/all-counterResultseed+2units UNCOMPILED/UNRUN;CLI optin/help/integration
   absent. RequiretrustedLFSrawSHA+completekeyedXetroot+EOF beforeJSON, carryall
   countersthroughresolution→execution, reconstructagainforpublication counted
   cumulatively. Default/dryrun noCAS; explicitapply optsoptionsclonein. Reconcile
   C2/eventualC3 overlaps beforeacceptance; currentnativeindexsupportnotadvertised.

CompletedquantSIDE+4integratedruns were retired after15exactcurrentbaseline/
ownedmatches, nochildprocessuse andpreservedusefulcolddesign. Interruptedfilters
MAIN/SIDEruns andbothunfinishedSIDEs remain. Do notreplace theseactualfrontiers
with an obsoletehandoffentrypoint orrestart alreadyacceptedquantwork.

### Audit limits and coordinator use

- This audit onlywritesitsown2runfiles. Parent-owned newhandoff/docs/TODO edits
  were observed in status and leftuntouched. NoGitstaging/commit/ref mutation.
- Oldacceptedlabels remainhistoricalauthority, and currentexact77IDs remain
  Gitprovenance. Testsareonlyclaimed fromcheckedretainedplans; currentimplementation
  audit hasnotfreshlyexecutedthem.
- Original33fullledger/perphaseH5–H23exactcommitIDs/titles and later129/191 aggregate versions
  cannotbe suppliedfrominspectedlocalhistory. A future externalbackup/explicitly
  authorizedfetch mayyieldthem; no suchsourcewasqueriedand norecoverypromised.
- Parentcanintegrate thiscontextintoitscomprehensivehandoff andthenretirethisrun
  afterallusefulresults/unknownsarepreserved. Do notreinstatearchive/SHAautomation.


<a id="recovered-history"></a>

## Recovered original94-source history: exact reading catalog

Author: Codex history-audit agent. Provenance: current implementation `f197c9c`;
historical reading reference `7e61cb63efd0c1429aa96044f2f7d28fda1e59aa`.
Documentation-only; runtime remains PAUSED. No source/Git mutation, Cargo, network,
model, SDK, device or SIDE operations. This file supersedes **only the overly
broad historical-recovery negatives** in the initial history-audit draft. The corrected recovery scope is integrated here.

### Correction and precise recovery boundary

The original audit checked standalone trees, refs, reflog and unreachable objects,
but failed to inspect implementation text nested inside the recoverable multimodal
aggregate. Its statement that the old implementation aggregate and all70 phase
reports were unrecoverable was too broad. Several earlier bodies survive inside
a reachable historical file despite missing standalone paths and original Git
objects. The coordinator was notified immediately during cold review.

The corrected distinction is:

- The current77-commit graph remains verified. The original33 exact commit ledger
  and pre-rewrite Git graph are not recovered. Historical labels in report text
  are not working Git object references.
- The recoverable outer15October multimodal aggregate embeds an **older full
  implementation handoff through P07-G3b**, comprising an executive brief and
  **94 numbered source sections**. It also has13 v5.1 appendices and3 then-live
  shared snapshots. These are historical text, not fresh runtime qualification.
- The later H4-era aggregate, including its original33 ledger and expanded
  inventories (129/191 in historical discussions), is not recovered. The70-report
  set removed in later cleanup is not recovered **as a complete set**. Several
  earlier members ARE recoverable below. Do not claim all70 bodies are missing.
- No H4/H5–H25 implementation report was found inside this94-source version.
  Later retained decision/accepted-plan summaries still supply those phase facts.
  H15 exact title/ID remains unknown. No original object was restored.

### Tested read-only commands and wrapper positions

All following line numbers refer to this specific historical Git object's output,
not the newly assembled handoff. Prefer numbered headings as stable reading aids.

```sh
# Older full implementation executive/index (read progressively).
git show 7e61cb6:.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md | sed -n '3931,4234p'
# Source1 example; every source has its own exact range below.
git show 7e61cb6:.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md | sed -n '4235,4555p'
# Concrete recovered G1, G3b and P00 reports.
git show 7e61cb6:.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md | sed -n '11875,11960p'
git show 7e61cb6:.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md | sed -n '12311,12367p'
git show 7e61cb6:.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md | sed -n '20563,20647p'
```

The nested implementation-history wrapper begins3931 and title3939. Its94-body
sequence begins4235. All94 start/END markers were checked. Inclusive ranges below
include original metadata/fences/end markers, without requiring any hash/byte
inventory. This is a reading catalog, not a revived archival workflow; do not
restore the old tree or execute obsolete command/generator instructions.

### Complete94 nested-source title and inclusive-line catalog

| No. | Exact historical section title | Start line | End line |
| --- | --- | ---: | ---: |
| 1 | README.md | 4235 | 4553 |
| 2 | .agents/plans/DECISION_REGISTER.md | 4556 | 4825 |
| 3 | .agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md | 4828 | 4999 |
| 4 | .agents/plans/IMPLEMENTATION_ROADMAP.md | 5002 | 5421 |
| 5 | .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md | 5424 | 5843 |
| 6 | docs/WHITEPAPER.md | 5846 | 6296 |
| 7 | docs/ENGINEERING_SPECIFICATION.md | 6299 | 7259 |
| 8 | .agents/docs/VISION_SCOPE_AND_PRINCIPLES.md | 7262 | 7397 |
| 9 | .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md | 7400 | 7590 |
| 10 | .agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md | 7593 | 7917 |
| 11 | .agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md | 7920 | 8173 |
| 12 | .agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md | 8176 | 8396 |
| 13 | .agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md | 8399 | 8551 |
| 14 | .agents/docs/SEMANTIC_IR_AND_COMPILER.md | 8554 | 8823 |
| 15 | .agents/docs/ASYNC_RESOURCE_COMMAND_IR.md | 8826 | 9043 |
| 16 | .agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md | 9046 | 9205 |
| 17 | .agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md | 9208 | 9532 |
| 18 | .agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md | 9535 | 9836 |
| 19 | .agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md | 9839 | 10109 |
| 20 | .agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md | 10112 | 10371 |
| 21 | the master handoff | 10374 | 10862 |
| 22 | .agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md | 10865 | 11238 |
| 23 | .agents/plans/the agent entrypoint | 11241 | 11398 |
| 24 | .agents/plans/CURRENT_STATUS_AND_TRACKING.md | 11401 | 11749 |
| 25 | .agents/plans/the next-agent prompt | 11752 | 11872 |
| 26 | the retired P07-G global-resource-arbiter handoff | 11875 | 11958 |
| 27 | the retired P07-G2 physical GPU-service handoff | 11961 | 12007 |
| 28 | the retired P07-G2b physical fair-GPU-workers handoff | 12010 | 12104 |
| 29 | the retired P07-G3a strict AMD Q4 and abort handoff | 12107 | 12196 |
| 30 | .agents/plans/FRESH_CODE_CORRECTNESS_ARCHITECTURE_REVIEW_2026-10-15.md | 12199 | 12308 |
| 31 | the retired P07-G3b shared-models handoff | 12311 | 12365 |
| 32 | .agents/prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md | 12368 | 12473 |
| 33 | .agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md | 12476 | 12711 |
| 34 | .agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md | 12714 | 12861 |
| 35 | .agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md | 12864 | 13198 |
| 36 | .agents/docs/COST_MODEL_AND_AUTOTUNING.md | 13201 | 13418 |
| 37 | .agents/prior-art/DESIGN_EVOLUTION_V3_HANDOFF_SNAPSHOT.md | 13421 | 13640 |
| 38 | .agents/prior-art/DESIGN_EVOLUTION_V3_SNAPSHOT.md | 13643 | 14541 |
| 39 | .agents/docs/DEVICE_KERNEL_AND_CODEGEN.md | 14544 | 14724 |
| 40 | .agents/docs/DISTRIBUTED_PARALLELISM.md | 14727 | 14791 |
| 41 | .agents/prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md | 14794 | 14863 |
| 42 | .agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md | 14866 | 15082 |
| 43 | .agents/docs/EXAMPLE_EXECUTION_TRACES.md | 15085 | 15251 |
| 44 | .agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md | 15254 | 15382 |
| 45 | .agents/prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md | 15385 | 15543 |
| 46 | .agents/docs/MEMORY_RESIDENCY_AND_IO.md | 15546 | 15808 |
| 47 | .agents/prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md | 15811 | 15896 |
| 48 | .agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md | 15899 | 16131 |
| 49 | .agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md | 16134 | 16368 |
| 50 | .agents/docs/OPERATOR_AND_FUSION_CATALOG.md | 16371 | 16565 |
| 51 | .agents/docs/PERFORMANCE_HYPOTHESES_AND_RISKS.md | 16568 | 16722 |
| 52 | .agents/prior-art/PRIOR_ART_HISTORICAL_ACCEPTANCE_GATES_V2.md | 16725 | 16841 |
| 53 | .agents/prior-art/PRIOR_ART_HISTORICAL_DISTRIBUTED_PARALLELISM_V4_1.md | 16844 | 16987 |
| 54 | .agents/prior-art/PRIOR_ART_HISTORICAL_EXPERIMENTS_V2.md | 16990 | 17153 |
| 55 | ../prior-art/PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md | 17156 | 17638 |
| 56 | .agents/prior-art/PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md | 17641 | 17930 |
| 57 | .agents/prior-art/PRIOR_ART_HISTORICAL_ROADMAP_V2.md | 17933 | 18341 |
| 58 | .agents/prior-art/PRIOR_ART_HISTORICAL_STAGE0_V2.md | 18344 | 18532 |
| 59 | .agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md | 18535 | 18681 |
| 60 | .agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md | 18684 | 18842 |
| 61 | .agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md | 18845 | 18991 |
| 62 | .agents/prior-art/RESEARCH_SOURCE_ATLAS.md | 18994 | 19126 |
| 63 | .agents/docs/RUNTIME_SERVING_AND_UX.md | 19129 | 19276 |
| 64 | .agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md | 19279 | 19484 |
| 65 | .agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md | 19487 | 19729 |
| 66 | .agents/docs/SECURITY_AND_RELIABILITY.md | 19732 | 19913 |
| 67 | .agents/docs/STATE_CACHE_AND_TRANSACTIONS.md | 19916 | 20152 |
| 68 | .agents/research/STRIX_HALO_OPTIMIZATION_DOSSIER.md | 20155 | 20317 |
| 69 | .agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md | 20320 | 20560 |
| 70 | the retired bootstrap handoff | 20563 | 20645 |
| 71 | .agents/plans/EXPERIMENT_PROGRAM.md | 20648 | 21024 |
| 72 | .agents/prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md | 21027 | 21138 |
| 73 | .agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md | 21141 | 21330 |
| 74 | .agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md | 21333 | 21496 |
| 75 | .agents/research/MULTIMODAL_PARALLEL_RESEARCH_WORKSTREAMS_2026-10-14.md | 21499 | 21547 |
| 76 | the retired P07-B mixed-quant handoff | 21550 | 21653 |
| 77 | the retired P07-C resident-GPU-weights handoff | 21656 | 21743 |
| 78 | the retired P07-D decode-once handoff | 21746 | 21841 |
| 79 | the retired P07-E async-readback handoff | 21844 | 21903 |
| 80 | the retired P07-F GPU-quarantine handoff | 21906 | 21969 |
| 81 | a retired phase-1 handoff | 21972 | 22065 |
| 82 | .agents/plans/QWEN35_8K_NUMERICAL_DRIFT_INVESTIGATION_2026-10-14.md | 22068 | 22147 |
| 83 | a retired Qwen CPU/GPU handoff | 22150 | 22261 |
| 84 | .agents/plans/QWEN35_CPU_LONG_CONTEXT_RESUME_2026-10-13.md | 22264 | 22370 |
| 85 | a retired Qwen numerics/context handoff | 22373 | 22535 |
| 86 | .agents/plans/QWEN35_EXACT_CPU_NUMERICS_LONG_CONTEXT_PHASE_2026-10-13.md | 22538 | 22779 |
| 87 | .agents/plans/QWEN35_PHASE1_8K_GATE_AND_P07_ACCELERATOR_HANDOFF_2026-10-14.md | 22782 | 22864 |
| 88 | .agents/plans/RELEASE_ACCEPTANCE_GATES.md | 22867 | 23075 |
| 89 | .agents/plans/THREE_STREAM_PROGRESS_2026-10-14.md | 23078 | 23187 |
| 90 | docs/PUBLICATION_AND_REVIEW_CHECKLIST.md | 23190 | 23284 |
| 91 | docs/README.md | 23287 | 23416 |
| 92 | docs/REFERENCES.md | 23419 | 23521 |
| 93 | the CPU reference probe README | 23524 | 23596 |
| 94 | the GPU probe crate README | 23599 | 23953 |

### Outer13 appendices and3 shared snapshots

These wrap the older94-source handoff; they are not another missing later
implementation archive. The end boundary is immediately before the next peer
section, so it may include separators.

| Section | Exact historical source title | Start line | End before next section |
| --- | --- | ---: | ---: |
| V5.1 source 1/13 | .agents/research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md | 73 | 239 |
| V5.1 source 2/13 | .agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md | 240 | 552 |
| V5.1 source 3/13 | .agents/docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md | 553 | 762 |
| V5.1 source 4/13 | .agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md | 763 | 1102 |
| V5.1 source 5/13 | .agents/research/MULTIMODAL_V5_1_RESEARCH_WORKSTREAMS_AND_AGENT_BRIEFS_2026-10-15.md | 1103 | 1245 |
| V5.1 source 6/13 | .agents/research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md | 1246 | 1413 |
| V5.1 source 7/13 | .agents/research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md | 1414 | 1627 |
| V5.1 source 8/13 | .agents/research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md | 1628 | 1833 |
| V5.1 source 9/13 | .agents/research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md | 1834 | 2050 |
| V5.1 source 10/13 | .agents/research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md | 2051 | 2250 |
| V5.1 source 11/13 | .agents/research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md | 2251 | 2457 |
| V5.1 source 12/13 | .agents/research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md | 2458 | 2623 |
| V5.1 source 13/13 | .agents/research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md | 2624 | 2890 |
| Live shared snapshot 1/3 | .agents/plans/DECISION_REGISTER.md | 2891 | 3160 |
| Live shared snapshot 2/3 | .agents/plans/IMPLEMENTATION_ROADMAP.md | 3161 | 3581 |
| Live shared snapshot 3/3 | .agents/plans/CURRENT_STATUS_AND_TRACKING.md | 3582 | 3930 |

### Additional nesting inspected for a second recovery oversight

- Source21 (10374–10864) preserves the earlier master handoff: P00, first source
  foundation, and V4.3→V4.2→V4.1→V4 design transitions. Older incomplete CPU
  statuses are superseded by later accepted results. It is recoverable text.
- Source23 is the then-agent entrypoint and25 the then-next-agent prompt. They
  survive as text; they do not authorize restarting G3/H5 or old evidence scripts.
- Sources37/38 are V3 snapshot versions. Source55 embeds V2/V2.1 plus the original
  V1 initial handoff, whose subsection begins at outerline17331. These design
  layers are not later trained-model/GPU tests.
- Source64 is the old workspace guide;70 P00 packaging;76–87 early implementation
  and numerics;93/94 old CPU/GPU-probe runbooks. Historical separate probe crates,
  locks and pre-layout paths do not override the accepted nine-package layout.
- All14 deleted archive/brief bodies were searched read-only for later H4,
  0cad60a/2125612/6d74f76 and explicit33-commit/129-source/191-source markers.
  No later H4-era implementation aggregate or exact original33 ledger was found.
  The pre-product README also retains earlier G3b-era status. The current decision
  register's later H4 summaries are different from full report-body recovery.

### Recovered historical phase labels and narrower test counts

The following labels are transcribed from executive chronology4063 onward. They
are **historical recorded labels**, not recovered objects or invented exact old
commit subjects. All nine new labels checked below returned128 to individual
`git cat-file -e LABEL^{commit}`; old602c246/4d1b82f/f28dc10 were already checked.

| Phase or baseline | Historical recorded label | Retained narrow meaning and checks |
| --- | --- | --- |
| P07-A |81a3e3c|Source Q4_0×dynamicQ8_0 seam; Unsupported-only fallback, fail-closed output; isolated physical projections are not full-model GPU|
| P07-B |2967edc|Mixed source quantization, host tensor/full-model completed-region simulation; no physical qualification yet|
| P07-C |836948b|Immutable allocation identity/residency; original .8B/2B3-token host tests one1/2MiB upload+2hits, complete logits/state equal|
| P07-D |e589be1|Decode fixed source blocks only on miss; DeviceEventLedger, host/event tests|
| P07-E |b283fb8|Real read_async completion bridge; actual hardware not yet verified in its own step|
| P07-F |602c246|Original physical handles retained on uncertainty, one static+transient cap;10 source-free+8 trained-GGUF host tests|
| P07-G1 |4d1b82f|4arbiter+4queue control-plane tests; no physical workers then|
| G2a starting baseline |f28dc10|Attested envelope;15GPU source-free+8GGUF host tests, not physical-model proof|
| G2b starting baseline |7e6e4f4|20GPU source-free+8GGUF host tests;4physicalNVIDIA completions/readbacks+2weight reuses; AMD F32 refuses root|
| G3a starting baseline |c3e5fca|WideF64 exact on both cards,4-token AMD hybrid; forced uncertainty5handles/256MiB fence, not driver recovery|
| Later review baseline |67dca72|Bounded adversarial source/geometry/queue/pre-submit/ownership fixes|
| G3b starting baseline |e9312ad|Shared Arc source/catalog and private session roots; measured source regions, not full resident/native CPP yet|

New individual lookup failures:81a3e3c,2967edc,836948b,e589be1,b283fb8,7e6e4f4,
c3e5fca,67dca72,e9312ad. These improve textual chronology without recovering the
old33 Git graph.

### Additional substantive early results and limitations

- P00 source70:4unit+3executable integration tests (7/7), native Linux release,
  strict clippy/fmt,4shell entrypoints,9Nix files parse/format and offline installer
  fresh/repeat/corruptSHA/wrongversion fixture. Early Nix evaluation was blocked by
  a read-only store-lock path; no six-target binaries/hosted CI/Windows/macOS
  physical validation/functional serving. Later reachable1af2ac4 fixes a different
  module-evaluation base-system issue; do not conflate the historical failures.
- Reference foundation source81:23passes (13library+5binary+5integration), source
  metadata/quant/config and failed three-domain transaction guards. No trained
  target weights/tokenizer/logits. Later original QwenCPU8K acceptance supersedes
  it; these first contracts do not establish27B/Gemma trained inference.
- CPU executive:8×32-token/256greedy short suites and146-vector trained comparisons
  permodel pass;2B48signed-zero distinctions in that earlier scope. Actual8192
  contexts bothgreedy7993/shared20top20 with logprob maxima.20372924/.19678696 and
  top-token probability errors.00215508/.0000536415. Functional limits are not
  exact source parity; GDN first .8Blayer4≈2.2538e-5/2Blayer1≈.01456 remained open.
  Later accepted512-source56zero signs is a different scope.
- G3b source31: shared vs old .8B3-token logits/state/root exact; two physical
  sessions share oneupload/onehit/two readbacks (AMD .8B/NVIDIA2B), independent
  roots/zero logical teardown charge. This is not later complete resident graph.
- G3b Q4 2048×1024 region measurements:2warmups+16readbacks, unoptimized debug build,
  H2D+kernel+readback included. NVIDIA NativeF32 median/p95 7115/13688µs vs WideF64
  7931/8897µs. AMD NativeF32 8499/10247µs vs WideF64 8473/11583µs. AMD F32 fails
  22896FP32 comparisons over18outputs/max2.3841858e-7, WideF64 exact. These are NOT
  modeltokens/sec, optimizedreleaseP95 or accepted NativeF32 policy. Device-wide
  VRAM/GTT counters remain elevated after zero logical charge: advisory pools/
  unrelated use, not exact model leak/free-space telemetry.
- G3b LLVM23 native blockers are historical. Later accepted CPP NVRTC/HIPRTC
  avoids the unused LLVM path without claiming independent LLVM device codegen.
  Historical AMD use does not lift the subsequent local hardware hold.

### Other assembled-handoff cross-checks

- Exact77 fullIDs+subjects match local `git log --all --reverse` in order. First46
  remain source-owner decomposition;23new acquisition commits+historicalA1 are
  not an invented original33 ledger.
- Acquisition full-gate counts, final934/69/205 and separate originalmodel17.60s/
  23.70s match retained canonical TODOs; no additive totals or combined timing.
- Five accepted backend profiles vs five target physical GPUs remain clear:
  four healthy cards, held local fifth. GPU512/functionalCPU8K/open independent
  8KGDN/16K/32K boundaries are distinct.
- All15 U056–U070 rows exactly match canonical register. Starting prompt keeps
  runtime PAUSED and requires explicit resume; old labels not current references.
- All assembled local Markdown file targets resolve. No other concrete chronology,
  count/timing/profile/pause blocker found in the scoped review.

Independent review required correction of the broad missing-aggregate/all70 claims,
including every inherited context paragraph. That correction and the recovery catalog are integrated in this handoff.


<a id="acquisition"></a>

## Model acquisition: complete current contracts and recent reasoning

The current detailed technical owner is
[model-acquisition/README.md](../plans/model-acquisition/README.md). Its live TODO
and two plain Markdown completed-checklist archives retain each implementation,
failed experiment, qualification command/result and pending extension. Acquisition
lives in `zerogravity-artifacts`; that crate also owns artifact parsing, immutable
source materialization, model/source metadata, quantization and the managed cache.
It is not an evidence-management crate.

### One source, transport, cache and preparation path

The optional Linux `cache` feature provides private managed content-addressed
blobs/manifests/snapshots, atomic refs, bounded locking/quota/admission, full-byte
verification and retained leases. The separate optional `acquisition` feature
adds one synchronous verified HTTPS owner, provider resolvers, exact plans,
streaming/progress/resume and explicit CLI. Default CPU/core/inspection builds
exclude HTTP/TLS/GPU dependencies. No Tokio service, Python loader/SDK, foreign
inference process, second cache or second model identity issuer is introduced.

The cache root is `${XDG_CACHE_HOME:-$HOME/.cache}/abird/zerogravity/models`;
relative XDG values are ignored and explicit absolute override wins. The initial
legacy root namespace was corrected before subsequent acquisition. Credentials
are separate from cache metadata. Paths and manifest/control bytes are bounded;
no absolute, traversal, duplicate, file/directory collision or symlink escape is
accepted. Root-relative retained descriptors protect traversal; pathname checks
alone are insufficient.

One original `ResourceArbiter` owns this process's transfer staging. Disk quota
serializes cooperating cache writers and includes true temporary/final/partial
peaks. Neither is a global host-RAM/free-space reservation. Stable lock inodes
must not be unlinked; a new inode would split authority. Immutable names/read-only
mode/open files do not freeze same-UID mutable backing. The actual source loader
hashes the bytes of an admitted private copy before the existing `SourceWeights`
issuer creates the process-local source identity. Manifest digest is durable
content identity, not that process identity or a runnable model.

Publication orders actual streaming/hash/EOF, file/directory synchronization,
verified dedup, immutable snapshot installation and atomic ref replacement.
Readers see the old complete snapshot or the new complete snapshot. Errors before
publication preserve the old ref; post-rename sync uncertainty has a distinct
`DurabilityUnknown` outcome. A file-complete progress callback is not publication.
A model source, device-ready object and session StateRoot are separate readiness
boundaries. Fault injection includes real process death, not only mocked errors.

Acquisition is explicit: run, inspect, catalog misses and unsupported model loads
do not download. Offline selects the local terminal branch before credentials,
client/DNS, admission or signals. It fully verifies actual retained bytes and
checks source/selection/current limits/pointers as required by the named mode.
Offline does not prove present remote authorization, license acceptance, alias
freshness, full remote membership or native Xet execution.

### Exact selection, closure and independent components

HF pins a branch/tag once to an exact commit, then uses one bounded complete
inventory for selected files, indices, HEAD recovery and sidecars. GitHub pins an
exact commit/tree or release/asset identity and refuses truncated trees. Git blob
object IDs are not raw SHA256. Ordinary `--file` retains its existing bounded
component-local byte `*`/`?` grammar, no slash crossing; `[]` is literal and `?`
consumes one byte. The newer Unicode/full-path filters remain an unwired draft.

Selected weights require complete model closure: config/tokenizer, complete
canonical ordered GGUF splits and safe safetensors index/shards. Indexed tensor
`total_size` is tensor payload, not shard file lengths including headers. Duplicate
keys, unsafe paths, mixed/ambiguous families and incomplete closures refuse.
Explicit components may supply separately authorized pinned config/tokenizer and
auxiliary snapshots. Each retains its own provider/repository/revision/manifest;
there is no invented atomic cross-source transaction, merged source identity or
composite ref. Same declared aliases reuse one pin; repeated release components
reuse one typed release pin without relisting each asset from mutable tags.

Git LFS pointer bytes are rejected as acquisition results, including correctly
hashed cached pointers. Generic cache inspect/verify still exposes literal stored
bytes; source eligibility is the stronger boundary. The legacy ordinary offline
API remains initial-selection/role-agnostic for compatibility. Component Weights
reuse strict closure with one cumulative metadata-byte/request budget across the
whole set. New strict quant/filter modes have their own explicit contracts.

### Transport, resume and bounded auxiliary bodies

Production HTTPS verifies TLS, identity encoding, framing, status and normalized
origin policy. No plaintext HTTP except controlled diagnostic literal-loopback
fixtures. Origin-scoped named credentials never appear in argv, plans, receipts,
errors or expiring signed locators. TRACE is compile-capped at DEBUG for the Rust
`log` facade because upstream ureq TRACE can expose query/Location secrets.
Foreign redirect origins lose credentials. Provider-specific opaque targets
preserve exact source/version/signature/condition authority in the same owner.
No arbitrary header/client/credential getter bypass is exposed.

Durable exact transfers retain proper-prefix checkpoints bound to immutable
source/path/length/strong validator, rehash the saved prefix, request/validate
Range and If-Range, and verify complete final hash/length/EOF. Changed validator,
status200 concatenation, invalid Content-Range and mismatched saved source refuse
or follow only the qualified owned restart path. Interrupted tails/older records
remain charged until a qualified owner reclaims them. Unknown-digest FirstSeen
transfers do not claim network single-flight. Typed cache hits verify actual data
before avoiding network work.

Exact lengths remain the default. Missing selected lengths use trusted pinned
LFS metadata or bounded body-free HEAD with exact representation rules. Chunked
bodies can transfer under an already authorized exact size. Explicit `AtMost`
streaming is Auxiliary-only: reserve the entire cap, distinguish unknown exact
size from the maximum, consume one terminal sentinel and publish actual observed
bytes. No required/model-closure cap, capped resume or independent full HTTP
trailer validation is claimed. Parsing/filesystem sync/blocked reads do not gain
real-time cancellation; original finite per-request deadlines remain binding.
SIGINT/SIGTERM registration is optional online CLI work, unregistered on return;
help/offline/cache/default CPU do not install it.

### Protected removal and explicit local HF import

Protected exact-snapshot removal excludes live readers through original managed
file/dup/inherited-child/lazy Artifact ownership. Graph ownership, locks,
ref/snapshot/blob sharing, actual retained handles, charge rebuild and process
faults are tested. Current removal does not imply automatic eviction, general
orphan/staging cleanup, legacy-root migration, arbitrary unmanaged-reader safety,
physical kernel/driver allocator release or global freed-space attestation.

`cache import-hf` is an explicit read-only external source capability: canonical
`models--owner--repo`, exact lowercase commit, explicit paths/roles, existing
absolute private destination and optional authorized raw hashes. Dry-run is local
metadata; apply creates independent verified owned copies through the original
transaction/publisher. It never adopts/deletes/chmods/renames/locks the external
cache or infers branch refs/config URLs. Open retained descriptors and rechecks
bind device/inode/size/mtime/ctime and exact EOF while copying; a separate owned
copy/load digest protects actual model execution from later mutation/removal.

Physical cross-filesystem qualification covers original same-mount plus distinct
Btrfs↔tmpfs and tmpfs↔tmpfs pairs, both directions, using existing filesystems.
Different mounts must have distinct devices and permitted retained-FD filesystem
types; selected files remain on their root mount/device. Same-device bind aliases,
Btrfs↔Btrfs, overlay/FUSE/network, hidden bind/subvolume fixtures and renamed layouts
remain unqualified. No mount or privilege changes were performed. tmpfs publication
acknowledgment is not persistence after unmount/reboot/power loss.

### Finite native provider profiles

| Surface | Qualified contract and explicit boundary |
| --- | --- |
| HF / GitHub | Exact pinned repository/release source, bounded inventory/closure/size recovery, named origin auth, original transfer/resume/progress/offline/cache. Git LFS object indirection, broader endpoints/env and complete new sidecar policy remain open. |
| HTTPS objects | Explicit origin/path/role/exact bytes/caller rawSHA, original one-entry managed verification; no implicit missing-digest or repository discovery. |
| OCI/Docker | Exact raw manifest digest and selected raw GGUF layer, tag pinning, public/preissued bearer and fixed Docker Hub anonymous broker. No image execution, compressed extraction, general index/platform/signature/private-login profile. |
| GCS | Native JSON generation/metageneration metadata/media, bounded auxiliary prefix/pageToken selections, public/preissued bearer. Independent object pins do not invent an atomic bucket snapshot. ADC/minting/requester-pays/custom endpoints remain open. |
| S3 | Explicit non-null versions, regional path/virtual-host addressing, exact initial/ranged representation, bounded V2/latest-version auxiliary prefixes, static/session SigV4. Additional current objects require caller SHA/exact-size contract or explicit strong-conditional observation. Broader partitions/endpoints/credential chains/FULL_OBJECT checksum authority remain open. |
| Azure | Explicit BlockBlob versions/snapshots and bounded current-version auxiliary listing, anonymous/bearer/SharedKey/ad-hoc serviceSAS, exact ranges. Current caller-SHA and conditional profiles are distinct. Latest/SAS/endpoint/default-managed-chain extensions remain open. |
| ModelScope | Explicit canonical40hex commit, bounded complete shallow walk and exact model closure; anonymous/official-token Bearer+session-cookie profile, digest-only offline. A direct directory hitting the native3000-row truncation boundary refuses. Alias echoes/timestamps/file-last-commit are not an alias→commit proof. |
| Native HF Xet | Exact HF pin/closure, header-only native announcement, scoped ephemeral CAS grant, exact signed range sets/multipart, ordered reconstruction/codecs/full root plus independent rawSHA/EOF through the original cache. At least one native file required; no compatibility substitution on failure. Native index reconstruction remains draft. |

Cloud caller-SHA contracts bind normalized locator/profile/path/role/size plus raw
content digest without inventing remote versioning. Conditional S3/Azure profiles
instead HEAD an exact strong ETag/length and send typed If-Match on every initial,
range and retry request. They remain FirstSeen rather than TrustedSha256. Range
validator contradictions refuse before signing/wire; HTTP412 never refreshes HEAD
or downgrades. SharedKey/SigV4 exact conditional signatures have independent literal
HMAC vectors. ETag/checksum labels are not automatically full raw content hashes.

Selected missing ModelScope file/index sizes use one same-full-commit body-free
exact200 no-follow HEAD through the existing cumulative metadata budget. Exact
identity Content-Length is required; no redirect/405/501/cap/GET fallback. Cache
index length once before its GET; unselected unknown lengths do not probe.
Malformed present/null blob sizes refuse, directory-null legacy behavior stays.
Strong ETag binds index/payload before bytes; weak/absent validators do not grant
resume authority. A later file-only CDN profile needs finite explicit origins,
exact bytes and authorized rawSHA, credential stripping on every delegated hop
including bounce-back, ephemeral redacted locators and same-pin/hash/size/range
refresh. Neither CDN nor alias/mirror authority has been qualified.

### Native Xet details that must survive continuation

The accepted protocol1.1 primitives use keyed BLAKE3 chunk/Merkle/file namespaces,
textual limb order and empty-root rules; rawSHA256 is separate content authority.
The allocation-free decoder bounds encoded chunks256KiB and decoded128KiB, checks
8-byte framing/exact consumption and supports NONE, framed LZ4 and BG4/LZ4 with
all tail remainders. Dictionary/legacy/skippable/concatenated frames refuse. No
absent per-term hash or partial-xorb root is invented as protocol authentication.

The native reader preserves ordered repeated/overlapping terms, exact file
coverage and complete signed interval sets, including arbitrary multipart order.
Only accepted logical chunks enter the file root/progress; edge/refetched work is
still decoded/charged. Replayed leaves must match prior accepted length/hash.
CAS v2 falls back to v1 only on404/501; scoped401/signed403 refresh is finite and
cannot change file identity or ordered semantic terms. Hub credentials never
reach CAS/signed payloads, even at the same origin, and grants never become
persisted model/source identity. Native targets refuse redirects.

One invocation bounds actual HTTP attempts, observed body/control wire bytes,
metadata, decoded chunks, logical leaves, refreshes and cumulative accepted
terms/groups/ranges across files/reissues. An observed overflow sentinel poisons
further work; exact-limit zero-byte EOF probes remain possible. Finite counters
and per-request deadlines are not physical TLS/kernel telemetry or one global
elapsed deadline. Native scratch is admitted before allocation and retained
alongside the cache transaction; native CLI defaults256MiB versus ordinary128MiB,
with explicit smaller limits still binding. Native object staging is ephemeral:
no HTTP-prefix checkpoint/resume; process exit starts a new full native transfer.
Ordinary sidecars retain their original durable resume/credential-redirect behavior.

Current dry-run is Hub metadata/announcement only, no CAS grant/signed body.
Offline verifies retained HF bytes but cannot attest native execution. The new
required-index draft needs complete counter continuation from pre-plan decoding
into execution: the current accepted seed carries only attempts/wire/metadata,
so simply enabling index decode without extending it would reset work limits.
The paused draft addresses this but is uncompiled/unqualified. Future measured
coalescing/chunk-cache/native persistent prefixes remain open; the current reader
may refetch complete signed sets and has no throughput superiority claim.

### Latest candidate and quant features

Candidate discovery is metadata-only through the same one pinned regular-file
inventory, with no family choice, HEAD, index/payload read or cache publication.
Complete inventory validation precedes subfolder projection, including regular
ancestor collisions. Returned scope is pinned-remote-tree or verified-cached-subset;
unknown lengths stay unknown and sizes are observations, not payload reservations.
JSON distinguishes stored Git blob sizes, HF tree/LFS metadata and verified retained
content. Git object IDs and cached FirstSeen digests never become expected rawSHA.
Cached discovery checks original source/current bounds/full hashes/pointers before
credentials/client/admission/signals. It cannot prove remote-global model choices.

Quant selection is an additive private-field `GgufQuantRequest` over that source,
not a change to public `Selection` or `PullRequest`. Tag grammar is1..32ASCII bytes,
letter-led then letters/digits/underscore, normalized uppercase. Exact lowercase
`.gguf`, optional strict five-digit split trailer and terminal basename `TAG`,
`-TAG` or `.TAG` match; underscores remain inside tags. Full path case/family
identity stays exact. Directory tag text never supplies a basename tag or split.
Every matching cohort must be complete before exactly-one-family selection;
malformed/missing/conflicting totals, case variants, singles+splits and multiple
model prefixes refuse. No default quant, numeric quality rank or first-file fallback.
Existing explicit files remain available for other filename conventions.

Selection happens inside the original provider tree before existing closure,
sizing and source construction. Shared logical-call bounds cover pin/tree/HEAD/
index/native discovery; original HTTP independently counts physical work. The CLI
uses one shared fetch/progress/cancellation/publisher path. `--file`, release assets
and `--quant` conflict explicitly. New quant offline requires snapshot digest and
source/current limits/pointers/selected closure plus Weights/Config/Tokenizer roles;
other retained data remains visible as the existing snapshot. Ordinary Files
retains its legacy offline semantics. The new native quant focused fixture proves
inventory/announcement/plan selection, not quant-specific CAS execution/inference.

### Failed experiments and lessons

The completed TODOs retain each failure and exact repair. Particularly important:

- Cache hits must read/hash actual bytes, not trust hash filenames, validators or
  a dedup label; cancellation and callback publication boundaries are independent.
- Legacy offline five intended failing assertions proved pointers/incomplete
  closure/reset budgets before the policy repair; other precompile static findings
  are not retroactively called executed RED tests.
- A native test pipeline with `tee` initially hid Cargo101 behind exit0; capture
  the actual command status. Synthetic `payload.bin` correctly triggered model
  closure, so auxiliary fixtures were renamed and a separate complete weights
  case added. Original sanitized State errors were not weakened to satisfy a
  raw Limit expectation; help do/does grammar was an assertion-only repair.
- Independent long-Xet oracle generation first copied65/66hex XOF prefixes;
  correct official32-byte prefixes and validate six official vectors before
  freezing custom128KiB/tree goldens. Never fit expected Rust results.
- Candidate review found a regular parent hidden by subfolder projection;
  full-inventory validation fixed it before first compile. Quant review found
  `-of-` in a directory falsely interpreted as a filename trailer; basename-only
  checks preserve full family identity. Both are static findings, not executed RED.
- ModelScope's new suite initially ran against an old normal rlib:1pass/9fail,
  with the removed error still inside the executable but absent from live source.
  `copy2` had preserved source mtimes older than Cargo outputs. Byte-equal narrow
  timestamp refresh on exactly five merged paths made the identical command pass.
  Future merges use fresh mtimes; do not relax math/protocol or clear all caches.
- The paused filters reference generator initially failed on Python3.14 strict-end
  regex spelling; only synonymous `\\z`→`\\Z` adaptation on installed3.13 was fixed
  before expected values were frozen. New Rust has never executed. The portable
  oracle copy is inert text because the original generator fetches source and
  overwrites an absolute SIDE path; do not run it as a continuation shortcut.



<a id="continuation"></a>

## Exact continuation and complete open-plan snapshot

Author: GPT-6 Codex (continuation audit).

This is the documentation-only continuation context, inspected on 2026-10-17. It is not authorization to resume and does not replace the live [plan index](../plans/TODO.md), [TODO rules](../plans/TODO-SPEC.md) or [AGENTS.md](../../AGENTS.md). The user explicitly requested wrapping the current phase and **PAUSING**. C2 is accepted; C3 and C4 remain unfinished source drafts. This later handoff request authorizes documentation only. Resume implementation only when the user asks.

### Starting point and verified scope

MAIN is `/home/pvl/spaces/abird/src/zerogravity`, branch `main`, inspected HEAD `f197c9cf4a44e06a3c1cf3c532855545bed945fa` (`feat(pull): select explicit GGUF quant families`). Subsequent coordinator documentation-only commits may advance HEAD: reread current files and status instead of resetting to this reference. At inspection the only MAIN tracked changes were the coordinator's rolling `.agents/plans/TODO.md` and model-acquisition TODO; own/other documentation runs were untracked. Preserve them. Do not treat an unclean status as permission to discard another writer's work.

Latest relevant accepted commits:

| Reference | Qualified slice |
| --- | --- |
| `6d74f76` (historical approval label; object unavailable) | Accepted original engine/source/state/resource baseline; remaining full-platform goals stayed open. |
| `837e77f` | Legacy offline managed-pointer checks and strict component model closure. |
| `52e42fc` | Native HF Xet reconstruction under the original HTTP/executor/cache owners. |
| `46cf5a5` | Strong conditional S3/Azure observations, FirstSeen raw hashing, original signing/range/publisher. |
| `4db5efb` | Selected ModelScope missing exact lengths via pinned HEAD. |
| `f22f759` | Metadata-only pinned candidate inventories, honest cache-subset coverage. |
| `f197c9c` | Explicit GGUF filename quant family selection, no silent fallback or tensor eligibility. |

These entries identify accepted records;6d74f76 is a retained approval label, not a currently resolvable object. The later implementation IDs are reachable. There is no new requirement to maintain SHA inventories, generated receipts or an evidence archive. Actual checkpoint/raw-content hashes remain implementation inputs where required.

The [acquisition TODO](../plans/model-acquisition/TODO.md) records C2 final975 focused executions/0 failures/16 intentional helper ignores and all4 strict artifact/CLI production/diagnostic configurations. Its frozen isolated full gate passed934 Rust executions/0 failures/69 ignored/205 suites plus18 namespace fixtures and the normal formatting/dependency/lint/build/release/installer/comparator/resume checks. Two separate exact release cache_prepare tests passed17.60s and23.70s, preserving both approved original Qwen3.5 models/all248320 logits/all42 state domains/two-token/session/source-lease/mutation/reclamation criteria. These are individual test durations, not a combined invocation. Actual Nix source-integrity,26 local links, protected root README and unchanged Cargo manifests/lock passed. This documentation audit ran none of those commands again.

Candidate coverage and quant/native scope remain deliberately narrow: candidate listing transfers no payload, HEAD or index bodies; quant chooses filenames, not tensor codecs or inference; new native-quant tests qualify shared Hub inventory/announcement/plan generation, not a new quant-specific CAS or inference gate. The original trained executable source boundary remains approved Qwen3.5 .8B/2B GGUF, current4GiB executable-source cap and the numerical/device scopes in their plans. Acquired safetensors/new model configs do not create runnable models.

### Non-negotiable working rules and existing authorization

- Read AGENTS, TODO-SPEC, index, relevant plan TODO/README and linked specifications before changes. Historical handoffs/roadmaps/status documents are design/history, never a second queue or proof of current support.
- One Rust mathematical engine; immutable original source identity with one checked issuer; independent StateRoots; one physical ResourceArbiter, fair workers and actual completion/quarantine. Do not add a downloader/runtime/cache/allocator/model engine merely to implement a provider or selection option.
- Keep the root `README.md` byte-exact: user-authored and outside agent edits. Acquisition design belongs in its plan README; root coordinates canonical docs. Preserve unrelated source/index/worktrees and each agent's own run file.
- Agents use only their assigned paths. Agree cold shared seams first; independent source/review/test work uses subagents, with one coordinator deciding/integrating. Serialize shared Cargo target ownership explicitly; no simultaneous Cargo/rustc/clippy or surprise SIDE builds. At this audit exact process-name inspection observed no cargo/rustc/clippy-driver process; recheck before a future grant. A prior grant is not a permanent target lock.
- Source/provider work authorized during the continuous task did not authorize arbitrary models, SDK downloads, private/gated license acceptance, external payload/account use, host mounts, driver/privilege changes, device reset/recovery or live cache mutations. Synthetic owned fixtures and already-approved original inputs were the qualification scope. Additional physical/network inputs need their applicable explicit authority; do not invent extra permission gates for ordinary reversible implementation already authorized after resume.
- The local Renoir/gfx90c AMD hold remains. Its aborted initialization/wedged recovery is not Drop-retirement/free-memory proof. No launches/resets/recovery on it without separate applicable authorization.
- No selected-GPU error→CPU fallback, numeric tolerance relaxation, implicit lower quant or hidden source change. Internal bit parity, independent oracle comparisons, model quality and throughput are separate acceptance dimensions.
- Offline must branch before credential env/file/FIFO/stdin/client/DNS/provider/admission/signal construction as each new profile specifies. `run`, `inspect`, cache/catalog misses never implicitly fetch. Credentials/tokens/signed URLs remain ephemeral, redacted and origin/profile-scoped; presets/config/model cards grant no transitive source or execution authority.
- Signed commits only; never disable signing. Root is current Git/dependency/lock/canonical/gate integrator. No push/force-push/tag/publish/deploy/destructive history/branch/worktree/cache operation without explicit authorization for that action. A source grant or request to commit is not permission to push.
- Main plan updates are one serialized coordinator write, integrating completed and unfinished results. Commit verified code/tests/plan together with explicit paths, record the actual commit in the next rolling update, retain unfinished work. No commit merely to record its own ID. Keep interrupted runs until recovered; remove finished runs only after useful contents are integrated.

#### Current acquisition environment and ownership boundaries

The explicit cache override is an absolute --cache path. Otherwise managed artifacts use `${XDG_CACHE_HOME:-$HOME/.cache}/abird/zerogravity/models`, ignoring relative/empty XDG overrides as implemented. This is the managed destination, never authority to inspect/delete an ambient external HF cache. HF import requires explicit repository-cache root/repository/canonical commit/selected files and an already-existing private destination; it copies verified bytes without modifying/locking/chmodding/hardlinking external data.

Legacy HF/GH credentials come only through HF_TOKEN/GITHUB_TOKEN and original normalized origins; GitHub explicitly scopes the accepted raw-content origin. Other provider handlers use caller-named environment variables or explicitly supplied bounded credential files/input only for their declared profiles. No ambient ADC/AWS/Azure/SDK metadata chain or automatic broker/login/refresh is implied. Never print values, dump the environment or paste signed URLs. Offline branches happen before those inputs; absence of online credentials does not invalidate a verified retained snapshot or attest current access. Native Xet cannot override Hub/CAS origins in CLI; transient CAS grants and signed range URLs never enter plans/manifests.

Ordinary CLI transfer memory default is128MiB; explicit native Xet default256MiB only when the caller did not specify --memory-limit-bytes. Library callers retain their own registered TransferAdmission. The process ResourceArbiter covers declared transfer/control/source allocations, not all host RAM/kernel/TLS memory or other processes. Blocking HTTP cancellation follows existing per-request deadlines/finite attempts, with no single whole-Xet-operation elapsed-time promise. Library cancellation stays an AtomicBool; CLI scoped SIGINT/SIGTERM flags feed that same owner. No new socket/async runtime is needed.

Cache completion/publication authority remains the original transaction/CommitOutcome. Exact sizes/trusted SHA and FirstSeen observations stay distinct; optional exact-size getters must not return cap maxima as known totals. Capped unknown streams are Auxiliary-only under explicit cap, ephemeral/no resume, true terminal-body/cap+1 checked; required model/index closure remains exact. Strong ETag conditions are observations, not immutable cloud versions or trusted rawSHA. HTTP decoder zero-chunk-line semantics and lack of independent full trailer validation remain documented, not silently strengthened by tests.

Components retain per-source immutable snapshots and verified leases; final component-set readiness is distinct from individual publication and is not a cross-snapshot atomic mutable reference. Protected v2 exact-snapshot reclamation only measures rebuilt managed namespace charge; it does not prove OS final-close/physical free bytes. Current refs/other manifests/shared blob reachability, actual managed descriptors+clones, activity/object/publication locks and retained stages remain authoritative. Legacy v1 migration/general orphan sweep/automatic eviction/unmanaged concurrent raw readers are NOT covered.

### Preserve the two unfinished worktrees

Only these two SIDE worktrees were listed at inspection. Completed quant/native/provider worktrees were already retired by the coordinator after source-preservation checks; do not reconstruct them as onboarding locations.

| Worktree | Actual detached base / overlay | Draft status |
| --- | --- | --- |
| `/home/pvl/spaces/abird/src/zerogravity-filters-work` | `4db5efbcce973d21bf1c8ca750471e6b2ab956ab` plus accepted C1/C2 source overlays | C3: only new `acquisition/filters.rs` differs from MAIN; unwired, not compile-ready. |
| `/home/pvl/spaces/abird/src/zerogravity-xet-index-work` | `f22f75972e63ea70771c59fbce065e371d5f29d4` plus accepted frozen C2 overlay | C4: only HF and xet/{mod,budget}.rs differ from MAIN; uncompiled, no integration/CLI tests. |

SIDE plan/rule copies may be older than MAIN: use current MAIN AGENTS/TODO-SPEC/canonical plans, then the SIDE own pause run and actual delta. Do not let a stale SIDE TODO or old source-grant wording override the current user pause.

Read-only continuation audit compared actual bytes: all15 C3 baseline Rust paths equal current MAIN; only new filters.rs has no MAIN counterpart. C4 has9 dirty/untracked overlay Rust paths equal MAIN and exactly3 declared production differences. Dirty/untracked status relative to an older detached base therefore includes accepted overlays, not just new work. Never copy the whole SIDE over MAIN or stage every dirty path. Inspect actual deltas against the accepted MAIN source and preserve newer fixes.

The15 C3 accepted overlay paths are artifacts acquisition/{components,github,huggingface,mod,selection,xet/mod,candidates,quant}.rs; artifact tests/acquisition_{candidates,quant}.rs; CLI src/{pull,pull_xet,pull_candidates}.rs; CLI tests/pull_{candidates,quant}.rs. C4's9 exact overlay paths are candidates/github/mod/selection/quant, artifact acquisition_quant test and CLI pull/pull_xet/pull_quant test. Its three new deltas are listed below.

Historical agent names: `transfer_resume` owned C3 matcher/integration proposal and independent reviews; `review_h5` owned C1/C2/CLI and C4 draft; `durable_cache` owned cache/provider seams and acted as current exclusive validator; root alone integrated/gated/committed. Names are context, not durable running processes. Reassign explicit ownership on a fresh session.

#### C3 filters — exact recoverable state

Authoritative paused run: [SIDE filters TODO](handoff-2026-10-17/PAUSED_FILTERS_CONTEXT.md). Main copy/design: [filters run](handoff-2026-10-17/PAUSED_FILTERS_CONTEXT.md). The SIDE pause section is newer/more complete than early cold-design checkboxes: source is already granted, but execution is paused. Do not ask again for those settled design choices after an actual user resume.

Only Rust delta: [unwired filters.rs](handoff-2026-10-17/paused-filters.patch). `acquisition/mod.rs` does NOT declare it. It contains the bounded matcher/API scaffold and inline unit expectations, but references NOT-YET-ADDED `quant::{validate_inventory,select_excluding,metadata_admission_extra}`, filtered provider entrypoints and `selection::close_model_selection_filtered`. It cannot be accepted/merged as a working feature. No integration test files were created. Scoped rustfmt/whitespace passed only; no Cargo/rustc/Rust test/lint or product matcher was executed. No child/session was retained when paused.

Frozen API/source grant:

```rust
FilterRequest::new(CandidateRequest, Vec<String> includes, Vec<String> excludes) -> Result<Self>
FilterRequest::for_quant(GgufQuantRequest, Vec<String> excludes) -> Result<Self>
// private fields; source/includes/excludes/quant_request read-only getters
filters::resolve(transport, &FilterRequest, &PullLimits, &TransferAdmission, &AtomicBool)
    -> Result<ResolvedPullPlan>
filters::validate_cached_selection(&SnapshotLease, &FilterRequest, &PullLimits, &AtomicBool)
    -> Result<()>
xet::resolve_filtered(transport, &FilterRequest, &PullLimits, &XetOptions,
                     &TransferAdmission, &AtomicBool) -> Result<ResolvedPullPlan>
```

Approved11 paths, all SIDE-only until integration: artifacts acquisition/{filters.rs NEW,quant.rs,selection.rs,huggingface.rs,github.rs,mod.rs,xet/mod.rs}; new artifacts/tests/acquisition_filters.rs; CLI src/{pull.rs,pull_xet.rs}; new CLI/tests/pull_filters.rs. `candidates.rs` needs a concrete agreed necessity before edits; none was planned. Existing private C2 names are InitialChoice and FetchRequest. Preserve public Selection/PullRequest/PullLimits shape and old byte/component `--file` grammar.

Settled semantics:

- Additive full-path Unicode-scalar case-exact FnmatchCaseV1: `*` spans slash; `?` consumes one scalar including slash; `**` has no extra grammar. No locale/case/Unicode normalization. Brackets/ranges/leading !/initial ]/literal ^/hyphens/descending ranges/literal malformed '[' follow fixed primary CPython grammar; backslash is literal. Inputs exclude control/NUL and lone-surrogate/bytes modes. Subfolder is a literal component-boundary scope; patterns match the relative suffix, so brackets in the folder never become pattern syntax.
- Include union → exclude union → family ambiguity → original required/optional model closure. Unmatched includes need not each match; empty final result refuses. Required config/tokenizer/shard/index exclusion refuses; optional README/license/tokenizer_config exclusions remain omitted. Removing an entire competing family may disambiguate; removing only a still-required split member fails. No silent numeric/source fallback.
- Explicit quant+excludes is supported composition; quant+includes and old file/release/asset+new filters refuse initially. Excludes alone is invalid except with quant base. Do not give unrelated providers/components/cloud modes implicit new grammar.
- Excluded adjacent index refuses BEFORE GET only when original closure would attempt to read that index. Do not blanket-refuse an unrelated index or GGUF selection. Retain separate HF and shared GH/ModelScope closure owners rather than unifying their historical .bin/source policies as a side effect.
- One pin/tree/remaining body/request/probe/admission owner. No discover_candidates→resolve repin/relist. Native thin route preserves original grants/raw SHA/root/EOF/actual budgets, >=1 announced native file and current required-native-index refusal until C4 qualifies.
- New offline mode is digest-only, current source/bounds/full managed bytes/pointer/roles/complete closure and EXACT retained closed snapshot subset: extras/excluded retained data must not be printed as if removed. It does not delete bytes/create a filtered snapshot or prove remote uniqueness/freshness/access/native execution.
- Pattern count<=4096; each nonempty<=4096 UTF8 bytes; aggregate pattern+inventory path bytes<=8MiB. Fixed cumulative16Mi compile/state/scalar/range work, no reset per pattern/file/closure; cancellation at most4096 transitions apart. One pattern compiled at a time, two NFA rows, no recursion/regex/global cache. <=512KiB scratch and extra16MiB SAME resolve-only admission were approved SUBJECT TO proving the actual retained representation. Do not reduce existing buffers or claim this is all host RAM. Tiny caller memory must refuse before HTTP; ordinary128/native256MiB defaults stay unchanged.

Independent expected values are frozen in the new file BEFORE the Rust matcher:73 literals plus585 patterns×57 names=33345 cases; literal SHA256 `c9f3b0c631f703ae44a4f2dafb5c8c71cb2715bfaa3249751cd299c73a40445f`. Generator is own MAIN [oracle.py](handoff-2026-10-17/paused-filters-oracle.py.txt), a disposable research helper. It executed pinned CPython v3.14.0 fnmatch source on installed3.13 with ONLY strict-end spelling `\z`→synonymous `\Z`. First attempt mistakenly searched two backslashes and failed before freezing outputs; corrected oracle then exited0. Never regenerate expectations from or fit them to Rust. No runtime Python dependency. Keep class edge behavior independently reviewed; all new Rust expectations remain UNRUN.

Resume tasks: inspect this draft first; implement missing private seams/default allow-all policy; prove scratch/collection/work/cancel bounds; wire both providers/closure and native path; add CLI grammar/shared FetchRequest path; author API/CLI/default-denial/offline/old-ref/native/legacy regression fixtures; independent review; only then exclusive compile/test/lint/gates. Adversarial classes/missing brackets/many patterns and an actual default admission test are still absent. No new public runtime or allocator is needed.

#### C4 required native Xet indexes — exact recoverable state

Read [SIDE implementation TODO](handoff-2026-10-17/PAUSED_XET_INDEX_CONTEXT.md) and [SIDE design TODO](handoff-2026-10-17/XET_INDEX_DESIGN.md). Design's old statement that it lives in retired quant SIDE and its recommendation for another resolver name are historical: the implementation pause section and coordinator-approved additive builder supersede them.

Three actual unfinished production paths only:

- `crates/zerogravity-artifacts/src/acquisition/huggingface.rs`: private pinned-index context/callback, after original exact length and before original parser.
- `crates/zerogravity-artifacts/src/acquisition/xet/mod.rs`: private index resolver reusing original discovery/Access/Invocation/XetReader/scratch; private defaultfalse XetOptions policy plus public additive `with_index_reconstruction(self)->Self`, same resolve/resolve_quant signatures.
- `crates/zerogravity-artifacts/src/acquisition/xet/budget.rs`: ALL-counter Result seed, poisoned/current-limit refusal and2 private unit cases.

All three are UNCOMPILED/UNRUN. Source-only syntax formatting passed. No new `acquisition_xet_indexes.rs`, no new CLI apply/helper/help edits and no actual index transfer/fault/rollback tests exist. Existing CLI still has no newly enabled index feature. Do not advertise native-index support or merge these three onto C3 blindly.

Approved behavior: default library options and metadata-only dry-run perform no CAS grant/signed body; required native index returns Unsupported before that activity. Only explicit builder opts bounded required-index reconstruction in, and CLI APPLY will use an opted-in options clone with no new flag/function proliferation. Ordinary non-LFS indexes keep existing metadata route. LFS index requires pinned trusted raw SHA independently of the complete Xet file root. Reconstruct full exact index plus terminal sentinel/root/EOF/raw SHA/pointer checks BEFORE parsing; exact `take(n)` without terminal read is insufficient. One pinned tree/selected-or-adjacent policy remains. Index length<=remaining metadata/maxfile/parser/addressable cap before allocations. No arbitrary URL callback or guessed HTTP response/source authority.

Use one existing Scratch sequentially for planning indexes and retained outer metadata control; no new allocator/body cache/preplan transaction. Keep path→XetSpec only to avoid duplicate discovery. Publication reconstructs the small index again through the original reader/installer and charges it again; executor's verified staged index reparse stays authoritative. Default simultaneous planning envelope is outer80MiB +scratch90.5625MiB =170.5625MiB before any later C3 extra control; native256 fits and128 must deny before signed body. Recompute checked formulas after composition, never infer all-RAM telemetry.

ALL cumulative counters must survive resolution→execution: actual attempts/wire/control, decoded chunks, logical leaves, refreshes and accepted terms/groups/ranges. Snapshot seed returns Result; poison/overflow/current tighter bounds cannot silently zero/reset. Final plan contains no grants/tokens/signed URLs. C4's unit cases alone cannot qualify this: add multiple individually-fit indexes+later bodies exhausting aggregate limits, actual owner index/raw SHA/root/EOF/budget/refusal/cancel/token isolation/default admission/dryrun/oldref fixtures and original regression cohorts. Native reader/hash/codec/multipart algorithms remain the existing owners.

C3 and C4 overlap HF/xet/mod and shared CLI. Root must choose serialized reconciliation order (natural continuation is qualify C3, then reconcile/qualify C4) and freeze exact seams before either shared merge. Preserve C3 excluded-index veto before invoking C4's reconstruction callback. Existing C2 source is accepted at f197c9c; never overwrite it with the older detached baseline.

### Commands, environment and failure lessons

These are future commands after user resume and coordinator target grant, not commands run by this documentation task. Use current checked-in scripts as the command owner.

Canonical complete gate:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh
```

`native-gpu.py` always creates private user+network namespaces (`unshare --user --map-root-user --net`) and sets Cargo offline. Failure to establish isolation aborts. It does not install/download/copy/patch SDKs. LLVM verification is opt-in `--require-llvm` for an already available pinned bundle, never an installation authorization. Do not repurpose HOME/CODEX_HOME or invent a target resource key.

For focused commands use the SAME jobs2/Rayon4/native wrapper and `--locked --offline`; explicit packages/features, never workspace `--all-features` (that selects GPU diagnostic closures). Examples to adapt only after source/test names exist:

```sh
cargo test --locked --offline -p zerogravity-artifacts --no-default-features \
  --features acquisition,diagnostic-acquisition,diagnostic-cache \
  --lib --test acquisition_filters
cargo clippy --locked --offline -p zerogravity-artifacts --no-default-features \
  --features acquisition --all-targets -- -D warnings
cargo clippy --locked --offline -p zerogravity-artifacts --no-default-features \
  --features acquisition,diagnostic-acquisition,diagnostic-cache --all-targets -- -D warnings
cargo test --locked --offline -p zerogravity-cli --no-default-features \
  --features acquisition --test pull_filters
```

Commands above show inner Cargo selections; run inside the isolated shell. Actual socket fixtures additionally require enabling ONLY that private namespace's loopback via the exact `PYLOOPBACK` block in [scripts/check.sh](../../scripts/check.sh). It verifies single non-host UID mapping and network namespace owner before touching lo. Use that block inside the same wrapper/session before focused socket commands. Never enable host interfaces/routes, run a privileged bypass or drop isolation to make a fixture pass. Production URL policy remains HTTPS; literal HTTP loopback tests using ResolvedFile require diagnostic-acquisition, not merely cfg(test).

Both original model routes, separately under the same wrapper/default jobs2/Rayon4:

```sh
cargo test --release --locked --offline -p zerogravity --features cache \
  --test cache_prepare approved_cached_models_match_direct_source_exactly \
  -- --ignored --exact --nocapture
cargo test --release --locked --offline -p zerogravity --features cache \
  --test cache_prepare approved_hf_import_models_match_direct_source_exactly \
  -- --ignored --exact --nocapture
```

They read already-approved `target/models/qwen35-0.8b` and `qwen35-2b`, create private synthetic destination/source copies and retain existing exact identities. Do not fetch replacements if missing. Approved raw GGUF SHA256 values are in the [accepted baseline](../plans/accepted-baseline/TODO.md); the tests also bind config/tokenizer hashes. Model/SDK directories are NEVER disposable Cargo cleanup targets. Additional GPU/source/model qualification requires its distinct current scope and authorization.

After root includes new reviewed paths in Git's actual Nix source, run the actual offline source check (typical current target `nix build --offline --no-link .#checks.x86_64-linux.source-integrity`). Check current flake/system interface before use; no broad Nix fetch or cross-platform SDK acquisition is implicit. [nix/source.nix](../../nix/source.nix) and [source-integrity](../../nix/checks.nix) compare original versus actual filtered vendor/model/source/TLS/Azure fixture bytes, one root Cargo.lock and unchanged root README. An untracked new Rust file omitted from the flake is NOT tested by a passing old derivation. Root owns explicit add-N/staging and source-integrity integration; test agents do not mutate the index.

**Copy bytes with fresh modification times.** Do not use cp -p/cp -a/rsync -a to merge old SIDE timestamps over a freshly built MAIN. ModelScope sizing initially produced a stale-library Cargo101 despite byte-correct copied source; narrow affected timestamp refresh resolved it without source/policy/expectation repair. After verifying exact intended bytes and ownership, refresh only affected source mtimes or use a correctly scoped fresh build target, then repeat the same affected checks. Do not hide a genuine compiler failure with unconditional cleans or change test expectations. Moving a checkout can separately leave absolute CARGO_BIN_EXE paths embedded in tests; [development guide](../../docs/DEVELOPMENT.md) explains rebuilding affected outputs. No blanket deletion of target/models, SDKs or another owner's target.

Use `bash` with `set -o pipefail` when teeing logs. Native Xet's initial compiler error was hidden by outer tee exit0; the correct record is compilation failed before tests (E0624), not a captured Cargo101 unless the pipeline actually captured it. Record actual commands/results, failures and repairs; no RED/GREEN claim for static review, formatting or authored tests. Keep literal independent expected vectors fixed. Once appropriate checks pass, do not repeat/broaden them without new changes/failures/concerns; proceed to coordinator acceptance.

### Broader backlog, execution order and concrete external gaps

All9 active plans remain unchecked. The full exact unchecked inventory is appended below so no deferred item is dropped; the live originals remain authoritative. The three completed plans (accepted baseline, handoff cleanup, historical records) have no open checkbox and must not be reopened as new implementation duties.

After an actual user resume, finish acquisition convenience C3/C4 with current provider-frontier rows, then continue finite native provider/auth/layout/preset/sidecar gaps as explicitly qualified slices. Planning/disposition or Unsupported does not close parity. Do not invent57 missing row IDs: historical captures were retired and the live README table is a finite class frontier, not an exhaustive provider proof. Registered tensor/serialization/IPC/plugin/random/dummy loaders belong to loader/runtime work unless an inspected missing byte protocol is identified. No generic plugin/Python engine/SDK delegation is required to claim a concrete native protocol.

CPU-first model addition is next independent major priority: original/config/index/tokenizer/checkpoint binding, admitted immutable complete owned shards and real Qwen3.8 27B plus Gemma4 E2B/26B numerical/state/tokenizer execution. Current large BF16 checkpoints exceed available local RAM/current4GiB executable-source API. A capable admitted host and explicit authorized inputs are real gaps; do not download arbitrary sources, relax source length or call metadata contracts inference. Preserve original Qwen3.5 gates. New GPU execution follows real CPU truth and healthy qualified hardware. Renoir remains held.

Independent8K GDN drift is still unresolved; changing tolerances, matching greedy tokens or internal bit parity does not close the independent numerical oracle. Genuine resident GPU scope remains512 consumed tokens;16K/32K and broad corpus remain open. Physical memory/quarantine/unknown completion cannot be overridden by catalog status, counter observations or a successful teardown assumption.

Serving remains unimplemented (`serve` refuses; NixOS service definition stays disabled). API route/CLI option frontiers are unreviewed parity seeds, not behavior. Full pinned llama.cpp/vLLM union, source kernels, sampler/RNG/grammar/tools/stops/reasoning/logprobs, batching/speculation and true goodput gates all remain. DSpark2 exact primary source/version is unresolved. Runtime/platform/license/security/distributed/multimodal and physical stream/codec/backing qualifications are separate, not inferred from acquisition or mocks.

Real remaining external/authorization gaps include selected large original checkpoints/license/access and capable hosts; other target hardware/OS/driver/compiler availability; independently authorized recovery for held devices; explicit hosted/network/privacy/cost and peer deployment choices; hidden-bind/subvolume/overlay mount fixtures (not authorized by the existing no-mount tests); and any new SDK/download/publishing actions. Record the concrete gap and continue independent authorized work; do not manufacture generic approval loops or convert lack of test access into support.

### Research and prior-art placement for coordinator integration

The user's context-preservation request should be satisfied under current docs/plans through normal coordinator integration; this audit moves nothing. Keep normative contracts separate from attribution/research, retain unique source/decision details, repair references atomically if anything is later relocated, and preserve implementation/reference fixtures. Do not resurrect retired raw archives or require their old evidence/SHA machinery.

- Entry points: [public docs index](../../docs/README.md), [engineering spec](../../docs/ENGINEERING_SPECIFICATION.md), live plans above and `.agents/docs/` subsystem contracts. The architecture audit is integrated earlier in this handoff; its scope is architecture chronology, while this part preserves current resumable state/backlog/commands.
- Acquisition: [current design/frontier](../plans/model-acquisition/README.md), [A2/A3 historical design specification](../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md), [immutable shards specification](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md), [original acquisition research](../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md), [A1 contract](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md). Old specification headers still describe planned A2/A3; live TODO/README/current code decide implemented scope. The acquisition research's57-row reference is historical, NOT an extant exhaustive ledger.
- Model/numerical/parity: [Qwen basics](../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md), [Gemma contracts](../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md), [model semantics](MODEL_SEMANTICS_AND_COMPATIBILITY.md), [numerics](NUMERICS_QUALITY_AND_BENCHMARKS.md), [independent native probes](../../tests/reference/llama_cpp/README.md). Parity source pins llama.cpp2851b3bcbf8d993a2243ae751c0a7dfa8e944a40 and vLLMa8be1500bad93d3c459e435bf71194aa8062e49d differ deliberately from numerical oracle llama.cpp0.6.0/build11429/d812350. DeepSpec005e03b81cec38b7da6399833d609ee89a2587f2 and DFlash07ebd93db9f472af339b644bb70221ad8428328a are specific research pins. Current CLI/API CSV frontiers,105 groups/145 document seeds are unreviewed, not exhaustive proof. Regenerate selected frontier from pinned source only when work requires, no fresh payload/SDK dependency.
- Native protocol primary pins remain linked in acquisition README References: Hub docs39aa2452668744552cc6690c75cd239771148e58 /Xet corec3c726bed5cf54ded92a63fab892cfb7857c751a/LZ4ebb370ca83af193212df4dcbadcc5d87bc0de2f0; ModelScope client65255a78b451c3785f2dfbfaff97172552fea3af; official cloud/OCI/GitHub/HF contracts; cached ureq3.4.2/ureq-proto decoder boundary. No new research fetch occurred in this handoff.
- Fleet/optimization/streaming/multimodal/serving: each plan's Contracts links point to the substantive research and acceptance registers retained in `.agents/research/` and `.agents/docs/`. All24 research Markdown paths and12 prior-art paths are enumerated below for root's no-loss placement review. They are technical/provenance documents, not completed support or new tracking duties.
- `.agents/prior-art/RESEARCH_SOURCE_ATLAS.md`, comparative compiler/runtime, Gufo fork/XDNA2, accelerator, distributed/tunnel, router/API, SSD and vLLM concurrency studies retain unique attributed mechanisms/limitations/licenses. Their historical external corpus/local path references are starting points, not permission to inspect live external caches or assume those paths still exist. External author benchmark figures are not Zerogravity measurements. Proprietary core engine code is unavailable/not authorized to copy; NPU shared-fabric hang notes forbid automatic clock/privilege work. Root may relocate/distill while retaining source attribution, not silently discard those details.

### Exact open-item snapshot by plan

This section transcribes current unchecked/in-progress/blocked item text as ordinary bullets, grouped by its live source section; the authoritative checkboxes remain only in the live plan/run owners. It is a finite resumption snapshot, NOT a second plan to maintain or independently tick. Read the live target before acting. Completed rows and archives are not duplicated; checked future/rolling documentation changes belong to the coordinator.

#### accepted-baseline — 0 open items

Live source: [.agents/plans/accepted-baseline/TODO.md](../plans/accepted-baseline/TODO.md).

Completed plan; no open checkbox. Remaining work is assigned to the active plans.

#### cpu-gpu-optimization — 17 open items

Live source: [.agents/plans/cpu-gpu-optimization/TODO.md](../plans/cpu-gpu-optimization/TODO.md).

Uncommitted:

- Keep the local Renoir/gfx90c hold: 2B initialization aborted and automatic resets wedged it; no further launch/reset/recovery or Drop-retirement proof is implied.
- Establish controlled cold/warm request baselines across all five GPU/three CPU optimization targets; accepted five backend profiles do not mean five cards or fleet-wide maximal capability use.
- Qualify capability-selected AVX2/AVX512/VNNI quantized kernels, cache/thread/topology policies and source-derived repacking while preserving reduction/rounding/zero/subnormal rules.
- Measure native/Vulkan tiling, subgroup/vector loads, packing, source-exact fusion/prepared dispatch; gate tensor/MMA/WMMA/integer-dot use by exact arithmetic, not advertised support.
- Evaluate explicit NativeF32 projection policy only after physical FMA edge tests, all-codec trained vectors and whole-model/cancellation/replay gates; WideF64 stays default meanwhile.
- Profile then qualify captured graphs/replay, bounded device loops/persistent scheduling, CPU/GPU overlap and exact optional GPU tokenization/detokenization; prepared plans alone are not graph capture.
- Keep separate LLVM code-generation paths and broader accelerator backends unqualified until independently built and physically tested; accepted native paths use CPP NVRTC/HIPRTC.
- Qualify optional EXL3/QTIP/trellis/codebook/low-bit derived packs and calibration identity/quality separately from original source semantics. No hidden precision substitution or default steering.
- Promote only repeated complete-request gains with cold/warm cost, tails, memory, power, source correctness, negative results and qualified capability/version guards.
- Gate future compiler/driver/libm and fast-arithmetic profiles with scalar fingerprints, physical edge/stress and trained full-logit/state tests. WideF64 is qualified on measured inputs, not a universal IEEE FMA emulation.
- Qualify real deadline wakeups, concurrent/multi-stream scheduling and actual driver failure/recovery before broader worker claims; queue deadlines currently need scheduling/poll opportunities.
- Keep the local gfx90c hold. If a healthy/recovered profile is separately authorized, localize retained allocation/H2D/final-sync initialization before changing kernels; natural SIGABRT establishes neither Rust Drop quarantine nor reclaimed storage.
- Consider read-mostly publication only after representative contention and ownership evidence. If left-right fits, test deterministic dual replay without IO/RNG/admission/state/device effects, one writer, persistent non-Sync readers, short guards without await/load/device work and explicit publish/drain/shutdown. Account for both indexes/logs while sharing one payload; do not reopen the rejected cold-catalog migration.
- Resolve applicable SYK-011..016 source/derived-quality/LLVM/SLO/pressure/steering choices from the [specialization reference](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) before that slice; proposed SYK-002..010 are not frozen user approvals. Preserve negative candidates and the default disabled steering policy.

Tests and verification:

- QTIP-01..14: derived source codec/pack identity, transforms, held-out quality and physical kernel correctness pass independently of original model semantics.
- HYB-01..18: exact source routing/draft/tap/mask/reduction and target-authoritative verification pass, including cases where a wrong draft is rejected or deeper drafting slows work.
- GFX-01..18: WMMA/LDS caller/shape coverage, driver/compiler numeric gates and coherent-memory physical headroom pass; require matched complete-request gains. See the [50-test register](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md#4-acceptance-register-50-planned-tests-not-implemented).

#### distributed-execution — 7 open items

Live source: [.agents/plans/distributed-execution/TODO.md](../plans/distributed-execution/TODO.md).

Uncommitted:

- Preserve zero-network standalone execution; add capability/transport/placement seams through the same model/task/coordinator/resource authorities.
- Qualify optional local CPU/GPU/NPU split projection, expert ownership and prefill/decode placement with actual event/memory/power-fabric safety; Stage N is independent of CPU/GPU release.
- Begin authorized two-node layer-group plus local-state execution and immutable full-state hydration; then evaluate P/D and attention+state versus FFN/expert disaggregation.
- Extend tensor/head/pipeline/data/context/expert/sequence parallelism and collectives only where exact state partition algebra, source numerics and transport cost support them.
- Authenticate/version source/state/tenant/session epochs and complete transfers; loss/timeout/stale peers cannot publish partial roots or release unknown in-flight storage.
- Qualify topology changes only at committed frontiers and compare collocation against fast/slow/tunneled links, including real transfer/copy/crypto cost and fairness.
- Add a peer/transport/placement without rewriting model, task, coordinator or API; fleet SSH qualification alone is not distributed inference.

#### handoff-cleanup — 0 open items

Live source: [.agents/plans/handoff-cleanup/TODO.md](../plans/handoff-cleanup/TODO.md).

Completed plan; no open checkbox. Remaining work is assigned to the active plans.

#### historical-records — 0 open items

Live source: [.agents/plans/historical-records/TODO.md](../plans/historical-records/TODO.md).

Completed plan; no open checkbox. Remaining work is assigned to the active plans.

#### memory-and-streaming — 14 open items

Live source: [.agents/plans/memory-and-streaming/TODO.md](../plans/memory-and-streaming/TODO.md).

Uncommitted:

- Route legacy/new model allocations, CPU state/scratch/staging and APU/GPU shared backing into the same configured physical pools with OS/cgroup headroom; never sum overlapping UMA/VRAM/GTT counters.
- Implement paged/contiguous state policy and full-domain prefix identity/intersection/COW across KV/SWA/recurrent/indexer/draft/grammar/media state, source/template/adapter and tenant boundaries.
- Qualify committed snapshot persistence/restore, corruption/security/compatibility UX and restore-versus-recompute; diagnostic portable checkpoints are not product persistence.
- Add immutable source extents and bounded coalesced IO/materialization with stable logical tensor identity, genuine copy readiness and original IO/device reader leases.
- Protect encoded-but-unsubmitted/captured expert reads and generation ABA; support mixed-codec expert size classes under one byte cap, source-exact routing and hot-cache preservation.
- Add bounded PLE/Engram lookup rows and separately governed disk state/media tiers; fully resident execution must incur no mandatory IO or host cache dispatch.
- Test source-exact resident/streamed parity, short/corrupt IO, cancellation, safe victim refusal, SSD QoS and actual physical lifetime before eviction/free-space claims.
- Optional direct/async/platform IO needs equivalent semantics and explicit support; large memory-pressure/SSD experiments require approved hardware/source scope.
- Distinguish logical envelope retirement from driver allocator pages/pools, fragmentation, cross-process use and actual OS release. Driver VRAM/GTT usage/max counters are not reservations or immediately free bytes.
- Prove trusted teardown/recovery before reclaiming unknown-work handles or clearing a fence. Diagnostic service sync/readback permits inspection/exit only and cannot synthesize Ready, retire unknown events or authorize reuse.

Tests and verification:

- SSD-01..32: run the [full source/residency suite](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) for each claimed backing/backend; mock tiers cannot qualify physical IO or trained streaming.
- SSD-14/15: encoded-but-unsubmitted, queued and captured readers prevent generation-ABA slot reuse through actual GPU completion/cancellation.
- SSD-21/22/24: mixed Q2/Q4 expert sizes share one byte cap with safe victims and fair progress; hot-cache retention cannot bypass admission.
- SSD-32: an authorized original trained streamed model matches complete resident logits/StateRoot; measure cold/warm IO, staging, OS headroom and genuine resource lifetime.

#### model-acquisition — 20 open items

Live source: [.agents/plans/model-acquisition/TODO.md](../plans/model-acquisition/TODO.md).

Uncommitted:

- PAUSED at user request2026-10-17 after wrapping current quant acceptance. Resume only when asked; preserve the following unfinished source worktrees and run TODOs. Parent interruption revalidation found all11MAINquantpaths exact frozen draft and no duplicate Cargo; dedicated validator finishes only currentphase.
- Resume C3 filters from `/home/pvl/spaces/abird/src/zerogravity-filters-work/.agents/runs/model-acquisition-filters/TODO.md`: base4db5efb+exactC1/C2overlays, only new acquisition/filters.rs differs from supplied baseline. Bounded matcher/oracle/API scaffold is UNWIRED and NOT compile-ready; all Rust tests unrun. Eight existing integration paths match C2, two integration tests absent; no CLI/module/provider/closure/native wiring. Preserve fixed primary oracle73cases+33345pairs, source-granted fullpath FnmatchCaseV1, includeunion→exclude→ambiguity, required-veto/optional-skip and conservative adjacent-index-before-GET policy. Validate actual512KiB scratch/16Miwork and extra16MiB sameowner admission before acceptance; preserve old Files semantics. Scoped formatting/whitespace only passed; owner stopped with no child process.
- Resume C4 native required indexes from `/home/pvl/spaces/abird/src/zerogravity-xet-index-work/.agents/runs/model-acquisition-xet-index/TODO.md` and sibling model-acquisition-xet-index-design/TODO.md: basef22f759+frozenC2, actualdraftpaths only HF provider and xet/{mod,budget}.rs. Private callback/explicit defaultfalse builder/all-counter Result seed and2unitcases are UNCOMPILED/UNRUN; CLI opt-in/help and integration tests absent. Reuse original reader/scratch, require trusted LFSrawSHA+completeXetroot+actualEOF before parser, carry every counter through resolution→execution, reconstruct again for publication with cumulative charging. Default/dryrun noCAS; native apply explicitly opts options clone in. Reconcile accepted C2 then eventual C3 overlap before qualification; syntax/format only passed, owner stopped/no child. No native-index support advertised.
- Extend cloud profiles through separately qualified S3 FULL_OBJECT checksum authority, broader endpoints/partitions/credentials, Azure latest/SAS/managed chains and per-object digest-map prefixes. Keep explicit version, caller-SHA and conditional FirstSeen contracts distinct; never infer immutable versions or trusted raw hashes from validator/checksum labels.
- Extend ModelScope only through separately qualified alias authority and redirect/CDN/mirror/credential profiles. Selected missing sizes now use exact pinned HEAD; no alias-to-commit authority was established from revision echoes, timestamps or file last-modified IDs. The current profile does not close broader parity or independently prove server revision/membership behavior.
- Extend native Xet to required LFS/Xet index JSON reconstruction under the same source/metadata/admission authority. Ordinary non-LFS index closure is supported; native index302 currently refuses. Define dry-run authorization semantics explicitly before changing the current no-CAS-grant discovery path.
- Measure and qualify native Xet fetch coalescing/chunk caching and any persistent native-prefix protocol. The current bounded per-intersection complete-signed-set reader has counted overfetch/refetch and no performance claim; native process interruption restarts a full transfer. Keep per-term/partial-xorb proof limits, fixed CAS scope and existing elapsed-time/HTTP-trailer boundaries explicit.
- Close finite convenience/compatibility gaps: filter/exclude grammar disposition, required sidecar/adapter/sparse-quant policy, bounded auxiliary/presets, mirror/endpoint/env precedence and gate/license diagnostics. Preserve explicit source authority/no silent numeric fallback and separate loader/runtime plans.
- Qualify any future v1 migration only after separate authorization and proven reader/process quiescence; current removal supports originally protected roots only.
- Extend cleanup only through separately qualified exact ownership/recovery protocols for dead retained staging/general orphans and any later eviction policy. Current exact-snapshot operation deliberately retains them and every stable lock inode.
- Any additional lazy/raw-path/mmap consumer must adopt managed descriptor/open ownership or remain externally quiescent during removal. Observed named cache charge is not a process-global final-close/physical-free-space attestation.
- Qualify additional filesystem pairs before extending the explicit allowlist. Different st_dev values alone do not prove independent backing objects for Btrfs subvolumes, overlay/FUSE or virtualized views; retain branch-specific root device/mount checks and complete inode/ancestry exclusion. No unsupported pair gains acceptance from generic mocks.
- Separately authorize any needed namespace-only hidden-bind/subvolume/overlay filesystem fixtures before mount operations. Real target-root/unselected inode aliases and hidden bind aliases in both directions remain NOT RUN; no host mounts, block devices, new privileges or live cache actions are implied. Current tests use already existing distinct filesystems only.
- Independently qualify any broader local layouts, renamed/unnamespaced roots, missing mount identities or new destination-root creation before exposing support. Keep explicit selection, retained descriptors, bounded topology work and existing cancellation/rollback limits; real import still needs user-selected source/repository/commit/files and copying authorization.
- Finish the finite provider/loader parity ledger beyond qualified HF/GitHub/HTTPS/OCI/GCS/versioned-S3/Azure/ModelScope and explicit local-HF import slices: remaining external-cache layouts/filesystems, additional model hubs, unversioned/broader S3/Azure/compatible profiles, native Xet and broader OCI profiles. Opaque serialization/plugin loaders remain separate runtime work.
- Continue explicit source/license authorization and trusted-digest versus first-seen provenance through additional providers/loaders; acquisition grants no license or loading support.

Tests and verification:

- ACQ-A12: complete the finite provider/loader parity inventory and independently qualify remaining supported paths.
- Each remaining provider phase requires focused API/CLI/protocol/auth/version/range/offline/fault gates, full frozen isolated check.sh, relevant source validation, actual source-integrity and documentation/dependency/root README checks before signing.

Deferred:

- Unsupported plans fail closed until qualified: unknown-length required/model-closure files, capped resumable transfers, unqualified external-cache layouts/filesystems and remaining cloud/OCI profiles, legacy-root removal and unmanaged-reader concurrent reclamation remain explicit above. Independent full HTTP trailer validation is unqualified under the documented decoder boundary, rather than a reason these payloads universally refuse. Independent component data grants no additional loader eligibility.
- Linux local-filesystem semantics are qualified first. HTTP cancellation obeys configured whole-request bounds; old retained attempts consume real quota and arbitrary same-UID mutation remains outside descriptor isolation.

#### model-execution — 10 open items

Live source: [.agents/plans/model-execution/TODO.md](../plans/model-execution/TODO.md).

Uncommitted:

- Acquire/bind authorized original config/index/tokenizer/checkpoint identities and choose a capable admitted host. Large BF16 sources exceed current local RAM; current executable source limit stays 4 GiB.
- Implement explicitly admitted verified owned shards, exact complete index/header/name/dtype/offset/shape/hash binding and retained source leases; no unsafe mutable-file mapping shortcut or second issuer.
- Qwen3.8 27B: original HF grouped-head ordering and Q/K replication 16→48, explicit converted-profile permutations, untied head and source-derived norm/RoPE geometry.
- Gemma E2B/26B: shared GELU/multiply/scale/soft-cap, proportional RoPE, value norm/sliding/shared KV, E2B PLE/doubled FFN and 26B K=V/MoE branches; choose and qualify the exact reference-framework numerical schedule.
- Qualify actual trained CPU operators/full vocabulary/state/nonzero replay, official tokenization/templates/reasoning/stop behavior, sessions, rollback/checkpoints and resource refusal.
- Add the same source-bound forward plans and full physical GPU qualification per eligible device, preserving all state/actual-completion/cancellation guarantees.
- Keep distinct future heavy/hybrid recipes (Qwen Flash-Next, GLM, MiMo and source-defined QSA/MLA/KDA/indexers/hyperconnections), and broaden dense/encoder/pooling/reranking/LoRA support only through independent model/codec/backend gates.
- Preserve CPU-first family priorities; already-authorized parallel basics do not authorize model/SDK downloads or unsupported whole-model claims.
- Bind explicit original HF versus converted Qwen source profiles. Conversion permutes grouped/tiled V projections/constants/convolution/output, uses -exp(A_log) and adds1 to non-GDN norms; GDN swish is SiLU while full-attention gating remains sigmoid.
- Choose and independently qualify Gemma26B reference-framework routing/top-k/reduction/scale placement before production execution. Five analytic test-local routing cases are not trained source parity.

#### multimodal — 24 open items

Live source: [.agents/plans/multimodal/TODO.md](../plans/multimodal/TODO.md).

Uncommitted:

- Add typed sample/frame/codebook/latent/decision values, rational clocks, checked conversions and complete source-bound component/codec/processor assemblies.
- Prove a small real non-token encoder/decision task and bounded iterative task through the existing engine before broad media claims.
- Qualify CPU-first original ASR and non-AR TTS (Whisper/Parakeet and Kokoro candidates); then actual backend execution and measured streaming/audio quality.
- Add depth-AR/RVQ speech, dual-AR codebook TTS and genuine codec-based duplex neural audio (including Qwen-Omni) where source/license/resources permit; mock codecs or text cascades do not establish native support.
- Qualify seeded source-exact image flow/editing conditioner+VAE pipelines, then video temporal/3D-VAE/expert switching and synchronized AV commit groups; candidate families remain separately gated.
- Keep ternary PTQ/Hadamard codecs, schema/calibration decision heads and specialist encoders as independent codec/model/task gates, not aliases of Qwen.
- Implement RealtimeSession lanes/epochs, bounded media buffers, VAD/barge-in, presentation/ack/playback frontiers and transactional rollback without speculative/partial media leakage.
- Add per-model transport capabilities and authorized hosted TaskService adapters with privacy/cost/idempotency/Unknown reconciliation; no duplicate local runtime or guessed realtime guarantees.
- Complete MM-01..24 plus model-specific ASR/TTS/image/video/decision quality/clock/resource suites; no real media-model support follows from source-free mocks.

Tests and verification:

- MM-D1/MM-01/MM-15: checked rational sample/frame/codec clocks, 44.1/48 kHz to 16/24 kHz and 24000/1001 or 30000/1001 FPS, long-duration remainder, discontinuity, overflow and bounded parser/buffer refusal.
- MM-D2/MM-02: complete source/processor/codec/license assemblies; wrong conditioner/shape/mask, missing source transform and cross-session mutable leakage refuse before use.
- MM-D3/MM-03..09: seeded solver/RNG and direct finite decision state, CTC/RNNT/TDT distinction, complete RVQ depth and joint AV commit; interrupted/stale device work never publishes or exposes media.
- MM-ASR-01..12: actual frontend/window/blank/predictor/duration/language/timestamp/diarization source fixtures; offline-only checkpoint refuses streaming, malicious audio bounded before allocation, sessions isolated.
- MM-TTS-01..12: non-AR phoneme/duration/vocoder and exact PCM count; incomplete 9/10 RVQ frame unplayable, codec-rate swap refused, overlap/voice generation and duplex/barge-in/state/lease boundaries correct.
- MM-IMG-01..12: source conditioner/solver/CFG/seed/mask/VAE/color and generation-entrypoint traces; invalid source stage/crop/processor fails, cancel retains old root and unknown physical handles, source/license/hosted status truthful.
- MM-VID-01..15: temporal VAE/latent crop/frame count, Wan timestep experts, source tile masks/rectangular attention, AV cross-attention/joint clocks, invalid GOP/backpressure, hot reload, cancellation and whole-assembly admission.
- MM-DEC-01..06: finite and joint schema head options/order/null/refusal/calibration against original source; generated text is not source-equivalent, external egress refused without authorization.
- MM-BON-01..08: PTQ1_0/PQ2_0 packed source scales/signs/padding plus mandatory 1024 Hadamard activation transform; wrong codec or plain-Q2 path refused, original scalar/CPU/device source parity separately qualified.
- MM-13..17/MM-19: real per-session/turn/transport epochs; stale ACK ignored, duplicate reconnect cursor idempotent/authenticated/bounded, playback progress remains uncertain when missing, slow clients do not release device leases or grow queues indefinitely.
- MM-18/MM-22/MM-23: ten speech/five video mixed jobs, fair shared physical admission and legal solver/frame yield; no per-packet catalog/router/global snapshot or silent precision/step downgrade, coupled output stays unplayable until both tracks valid.
- MM-SRV-01..12/MM-20/21/24: exact versioned SDK/protocol/field/task conformance, hosted accepted-timeout Unknown/idempotent reconciliation, callback/signature/TTL/MIME bounds, malformed media/unauthorized egress/cross-tenant replay refused before side effects.
- MM-I0 change-radius drill: small encoder/finite task and iterative mock use existing source/state/resource owners; no central router/worker rewrite or duplicate runtime; original Qwen gates stay unchanged.
- Eligible source-real CPU→physical full-model→playable-stream→soft-QoS gates: exact licensed checkpoints/processors/codecs and original quality/state oracles, real negative cancellation/admission/replay tests and measured cold/warm tails/memory. Mock/reference/region/hosted results never promote another tier.
- Resolve realtime section18 policy choices explicitly (media numeric policy, first sources, SLO/clock uncertainty, mid-turn changes, persistence, biometric/asset consent, hosted cost/egress, transport packaging and safe physical recovery). User examples do not establish released model IDs or grant execution/provider/download authority.

#### numerics-and-context — 7 open items

Live source: [.agents/plans/numerics-and-context/TODO.md](../plans/numerics-and-context/TODO.md).

Uncommitted:

- Resolve independent original-source CPU 8K GDN numerical divergence without loosening tolerances or changing the declared source arithmetic.
- Extend genuine resident GPU context/corpus qualification beyond the currently accepted 512 consumed tokens, including capacity/window/padding/replay boundaries.
- Run genuine whole-model 16K/32K CPU/GPU gates only after prerequisites; functional 8K success is not independent long-context numerical closure.
- Keep internal full-logit/all-state/trace bit parity distinct from independent native-source stage/top20 comparisons, including the known 2B 56 signed-zero distinctions.
- Broaden prompts/tokenizers/precision profiles and sampling quality only with independent oracles, exact history and explicit qualified ranges.
- Diagnose identical teacher-forced prefixes512/1024/2048/4096/8192 against pinned llama.cpp0.6.0/build11429 `d812350`, CPU/F32 KV and Flash Attention off. Fix the physical microbatch/chunk/arithmetic schedule and compare per-head GDN state, gain/beta, quantization and AR-versus-bulk recurrence; distinguish replay correctness from arithmetic.
- Compare native ubatch1/64/256/512 only within bounded feasible runs; the historical full ubatch1 capture did not finish. Use raw token IDs to avoid incomplete UTF-8 HTTP rows. Recheck short corpus, layers, state and actual8K after any fix; retain original functional0.45 shared-top20-logprob/0.05 top-token-probability limits and strict source-bit operator gates.

#### platform-and-release — 13 open items

Live source: [.agents/plans/platform-and-release/TODO.md](../plans/platform-and-release/TODO.md).

Uncommitted:

- Generalize the working typed forward plan into verified effectful semantic regions/task control, closed provider substitution, bounded shape/kernel caches and explicit command/resource legality; no new persistent IR/runtime without a real need.
- Prove artifact→model→compiled task→coordinator→output and explain-plan for supported tasks, including source-qualified preselected fallback and no selected-GPU-error CPU substitution.
- Run extension drills for one stateful semantic op, device/provider, non-token TaskProgram, quant/storage/API adapter and optional ParameterVersion/TrainStep seam; future training is not thereby implemented.
- Finish per-file source/model/license/NOTICE/SBOM policy, malformed-input fuzzing and ownership/security review, authenticated APIs/private caches and qualified driver/device safety.
- Verify Nix/module/native/cross packages and installer/release behavior on actual Linux/Windows/macOS target access; packaging definitions alone do not prove platform support.
- Maintain minimal/default/composed CPU-free feature isolation, one lock/version authority and truthful unsupported capability behavior.
- Gate bounded compiler/autotune/AOT caches by source/device/driver/shape and retain negative candidates; no unbounded first-request search.
- Complete reproducible/signing/release policy and fair model/task/quality/performance matrices with cold/warm, p50/p95/p99, memory/energy and uncertainty. Push/tag/release remain separately authorized actions.
- Demonstrate compatibility/invalidation of operator/task/state/plan caches and bounded change radius before broader platform claims.
- Before a true zero-copy strided View is enabled, validate contiguity against stored value provenance and test public PlanValue retargeting. Current View physically materializes dense output; retain cached-element and dynamic range guards.
- Preserve one GPU-library service/queue/quarantine owner and cold generic-plan admission through packaging changes; verify actual final downstream links, not only compiler/type checks.

Tests and verification:

- Qualify all six [CPU release definitions](../../docs/DEVELOPMENT.md#cpu-cross-release-definitions) with actual target execution, installer checksum/version/refusal cases and platform signing/authenticity policy. Build definitions and same-host checks do not establish other-target support.
- Test enabled NixOS service lifecycle and runtime-secret handling after serving is implemented; the currently disabled service definition and module evaluation are not a running-server claim.

#### serving-and-speculation — 9 open items

Live source: [.agents/plans/serving-and-speculation/TODO.md](../plans/serving-and-speculation/TODO.md).

Uncommitted:

- Implement canonical TaskRequest/coordinator workers and CLI/SDK/API integration, model routing/residency lifecycle, explicit supported OpenAI/Anthropic interfaces, streaming/tools/reasoning/schema/sampling and truthful capabilities.
- Enforce separate API/tenant payload, queued-prefill/residency and physical active-step admission; bounded overload, cancellation, disconnect, stale epochs and output backpressure.
- Add real compatible continuous/grouped batches, aged chunked prefill, decode/media fairness and finite joint batch/draft/expert/kernel/placement plans; no per-token catalog/JSON/global-lock work.
- Finish Qwen basics before optimizations; enumerate the complete pinned llama.cpp/vLLM union, public and internal surfaces. Preserve 105 provisional groups/145 document seeds and CLI/API frontiers as unreviewed inputs, not exhaustive parity.
- Implement every applicable sampler/logits-processor/RNG/logprob/grammar/tool/stop/reasoning composition and its independently verified outputs.
- Qualify target-authoritative MTP, draft model, n-gram/suffix, EAGLE, DFlash/DFlash2, DSpark and related registered strategies; DSpark2 exact primary source/version remains unresolved.
- Test all accepted-prefix lengths, target-distribution correction, target/draft RNG/grammar ownership, full KV/recurrent/index/media rollback, physical branch admission and authentic cancellation/quarantine.
- Jointly benchmark material combinations, adaptive depth/backoff and all offered/rejected/canceled work: committed SLO-goodput, TTFT/ITL tails, fairness, memory and energy versus equivalent pinned vLLM/V1/Omni.
- Preserve zero untriaged frontier/disposition requirements before any full-parity claim; synthetic acceptance, fewer launches and competitor measurements do not establish local parity or speed.

### Retained research and prior-art source path index

For coordinator placement/context preservation only. No files were moved or copied, and no new mandatory archive exists. Source diagrams/code/fixtures remain in their actual owners.

#### .agents/research — 24 files

- [EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md)
- [GEMMA4_FAMILY_CONTRACTS_2026-10-16.md](../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md)
- [GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md](../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md)
- [HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md](../research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md)
- [INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md)
- [MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md](../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md)
- [MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md](../research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md)
- [MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md](../research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md)
- [MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md](../research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md)
- [NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md](../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md)
- [PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md)
- [QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md](../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md)
- [README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md](../research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md)
- [SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md](../research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md)
- [SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md)
- [STRIX_HALO_OPTIMIZATION_DOSSIER.md](../research/STRIX_HALO_OPTIMIZATION_DOSSIER.md)
- [VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md)
- [ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md](../research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md)
- [CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md](../research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md)
- [DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md](../research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md)
- [IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md](../research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md)
- [SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md](../research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md)
- [TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md](../research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md)
- [VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md](../research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md)

#### .agents/prior-art — 12 files

- [ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md)
- [ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md](../prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md)
- [ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md)
- [COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md](../prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md)
- [DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md)
- [FINAL_ARCHITECTURE_REVIEW_2026-10-12.md](../prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md)
- [GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md)
- [MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md)
- [PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md)
- [RESEARCH_SOURCE_ATLAS.md](../prior-art/RESEARCH_SOURCE_ATLAS.md)
- [SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md)
- [VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md](../prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md)

### First resumption checklist for the next coordinator

1. Obtain/recognize an actual user resume; this documentation-only request leaves execution paused. Read the current rules/index/relevant live TODOs and inspect current MAIN/status/worktree processes. Preserve rolling docs and other owners. Do not recreate retired worktrees or treat this snapshot as an authority to reset source.
2. Review the accepted f197c9c code/current delta and the two paused runs. Reassign named paths and one Cargo owner. C3 is a single unwired file; C4 has exactly3 unfinished deltas. Establish integration order for HF/xet/CLI overlap; preserve current C2 basename fix, old Files, native/default budgets and the sole publisher.
3. Resume the chosen smallest source slice in its preserved SIDE, keeping a current own run. Review independent goldens/design before edits; keep absent/unrun fixtures explicit. No source copy/merge/compile merely because a file formatted.
4. At a frozen reviewed boundary, coordinator copies only agreed owned changes with fresh mtimes, checks intended byte preservation and retains original source/dependency/README authority. Never copy an older complete SIDE baseline over newer MAIN or run two Cargo owners.
5. Run appropriate focused actual API/protocol/CLI/auth/offline/cancel/fault/process/admission/legacy tests and both strict configurations under canonical isolation; report actual failures before changing policy/expectations. Full frozen check.sh, actual Nix source-integrity/new-file inclusion/current links, protected README/dependency checks and applicable original source/model gates precede acceptance. Additional device/input/recovery/download authority remains separate.
6. Coordinator atomically integrates useful run results and every unfinished item into current canonical docs/plans, signs verified code/tests/plan with explicit paths and carries rolling commit provenance to the next update. Full plan stays unchecked until every item/gate is actually complete. Future continuation should not stop after a convenience milestone when the user has resumed the broader authorized task, but it must obey any new pause.

Documentation-only validation of this handoff cannot establish implementation correctness, feature support or successful recovery of either paused draft.


<a id="operations"></a>

## Practical reconstruction and handoff boundaries

### Portable unfinished source under the requested documentation roots

All new handoff material is under `.agents/docs/` or `.agents/plans/`. The
supporting directory [handoff-2026-10-17](handoff-2026-10-17/) contains:

- [PAUSED_FILTERS_CONTEXT.md](handoff-2026-10-17/PAUSED_FILTERS_CONTEXT.md): exact
  paused C3 design/grant/state/pending-test record.
- [paused-filters.patch](handoff-2026-10-17/paused-filters.patch): the one new,
  unwired `acquisition/filters.rs` relative to implementationf197c9c.
- [PAUSED_XET_INDEX_CONTEXT.md](handoff-2026-10-17/PAUSED_XET_INDEX_CONTEXT.md): exact
  C4 incomplete implementation and resumption record.
- [XET_INDEX_DESIGN.md](handoff-2026-10-17/XET_INDEX_DESIGN.md): complete cold
  protocol/admission/counter/verification design; the current builder decision in
  the implementation context supersedes its older named-resolver suggestion.
- [paused-xet-index.patch](handoff-2026-10-17/paused-xet-index.patch): exact three
  draft production-file deltas againstf197c9c.
- [paused-filters-oracle.py.txt](handoff-2026-10-17/paused-filters-oracle.py.txt):
  inert source of the independent expected-vector generator. Its original
  MAIN run copy fetches primary code and overwrites an absolute SIDE source path;
  do not execute it as a handoff step or replace fixed expectations with Rust.

These are unfinished source/context, not historical proof archives. Actual
`git apply --check`, application and byte comparison in an owned disposable
reconstruction tree reproduced all four draft files exactly fromf197c9c. That
proves recovery bytes, not type correctness, tests, support or merge readiness.
An initial check from a subdirectory accidentally discovered the parent Git root
and skipped the patch; the missing-file comparison caught it. A separate owned
temporary Git root fixed the check. MAIN/SIDE source was never modified by it.

Prefer inspecting the preserved actual SIDEs when they still exist. If they are
missing and continuation has been authorized, create separate disposable
worktrees atf197c9c and apply only the corresponding patch in each. Inspect
before applying; never apply drafts blindly onto a newer MAIN or combine them
without reconciling shared HF/quant/Xet/CLI seams. A mechanical reconstruction
example, not an instruction to resume during this documentation task:

```sh
git worktree add --detach ../zerogravity-filters-resume f197c9c
git -C ../zerogravity-filters-resume apply --check \
  /home/pvl/spaces/abird/src/zerogravity/.agents/docs/handoff-2026-10-17/paused-filters.patch
git -C ../zerogravity-filters-resume apply \
  /home/pvl/spaces/abird/src/zerogravity/.agents/docs/handoff-2026-10-17/paused-filters.patch
```

Use a different explicit worktree for the native-index patch. Detached bases are
context references, not permission to reset or overwrite current work. The
existing filters SIDE started earlier with accepted overlays; the portable patch
has already normalized its actual one-file delta ontof197c9c. The index patch
likewise excludes accepted C2 overlay changes. Both features are incomplete.

### Workspace owner map

| Owner | Responsibility and first source locations |
| --- | --- |
| `zerogravity-core` | Immutable source IDs, state/session roots, physical ResourceArbiter, device queues/events and typed accelerator programs; `crates/zerogravity-core/src/{source,state,session_root,resource_arbiter,device_queue,device_events,accelerator}.rs`. |
| `zerogravity-artifacts` | GGUF/config/safetensors/model contracts, quant/source weights, admitted immutable source materialization, managed cache/acquisition; `src/{artifact,model,sharded,source_weights,quant}.rs`, `src/{cache,acquisition}/`. It is substantive runtime/data ownership, not evidence tooling. |
| `zerogravity-cpu` | Portable source math/quantized weight execution; `src/{math,weights}.rs`. |
| `zerogravity-model-qwen35` | One Qwen model semantic schedule, immutable shared model/session/checkpoint/tokenizer/source-profile contracts; `src/{lib,forward,host_math,checkpoint,tokenize,model_catalog}.rs`. |
| `zerogravity-model-gemma4` | Config/tensor/variant/MoE/E2B/26B source contracts and their tests, not trained execution. |
| `zerogravity-gpu` | Reusable resident-plan execution, source buffers, prepared bindings, one service/queue/quarantine and Vulkan/CPP CUDA/HIP providers; production features default off, no duplicate Qwen loop. |
| `zerogravity` | Library facade/preparation/semantic task drivers/session coordination; `src/{lib,prepare}.rs` and original-model integration tests. |
| `zerogravity-cli` | Sole product executable; inspect/run/backend frontends/pull/cache; defaultCPU+Qwen+text. `serve` remains unimplemented. |
| `tools/gpu-qualify` | Workspace qualification tool linking the same GPU library. Legacy executable name `zerogravity-gpu-probe` may remain; there is no independent `probes/` workspace or lockfile. |

One root `Cargo.lock`, one workspace version authority, production defaults separated
from diagnostics, default-members facade+CLI. Vendored CubeCL0.11.0 IR/CPP/CUDA/HIP
packages are patched path dependencies under the GPU crate and excluded as
workspace members. Preserve license/NOTICE/provenance and downstream patches.

### Concrete build and qualification commands

The pinned toolchain is Rust1.95.0 with rustfmt/Clippy. See
[development guide](../../docs/DEVELOPMENT.md),
[GPU library runbook](../../crates/zerogravity-gpu/README.md),
[qualification tool](../../tools/gpu-qualify/README.md) and
[reference probes](../../tests/reference/llama_cpp/README.md).

```sh
# Revalidation: read current state, do not reset to a historical reference.
pwd
git status --short --branch
git log -5 --oneline
git worktree list --porcelain

# Locked/offline CPU build, using only existing authorized dependencies.
cargo build --release --locked --offline -p zerogravity-cli

# Complete implementation-phase gate, one MAIN target owner.
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
  python3 scripts/native-gpu.py -- bash scripts/check.sh

# Actual source-filter check; new source must be included in the Git flake first.
nix build --offline --no-link .#checks.x86_64-linux.source-integrity

# Original trained cache/direct route (both approved models in this one test).
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo test --release --locked --offline -p zerogravity --features cache \
  --test cache_prepare approved_cached_models_match_direct_source_exactly \
  -- --ignored --exact --nocapture

# Original trained external-HF-import route, separately and sequentially.
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo test --release --locked --offline -p zerogravity --features cache \
  --test cache_prepare approved_hf_import_models_match_direct_source_exactly \
  -- --ignored --exact --nocapture
```

Do not execute new implementations/tests/hardware work until resumed. The above
commands document the accepted workflow; this handoff documentation task does
not rerun model/build qualification unnecessarily. A new source phase must pass
its focused meaningful API/CLI/protocol/fault checks, all appropriate feature and
strict lint configurations, full frozen gate, actual Nix/source/document checks
and relevant trained/physical gates before acceptance. Source-free mocks and
compilation do not qualify a model or device.

Actual approved GGUF source hashes, also enforced by the existing tests:

| Original input | Raw SHA256 |
| --- | --- |
| Qwen3.5 0.8B | `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf` |
| Qwen3.5 2B | `91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6` |

Existing source/config/tokenizer fixtures live beneath `target/models/qwen35-0.8b`
and `target/models/qwen35-2b`; tests bind their own exact config/tokenizer bytes.
If absent or changed, record the missing approved input; do not download a
replacement or silently select another checkpoint. Source hashes are real
model integrity inputs, not a revived evidence-log inventory.

Use actual native user/network isolation, not Cargo's offline flag alone: build
scripts may otherwise access the network. The default native CPP guard uses
existing NVRTC/HIPRTC without LLVM. Explicit `--require-llvm`/`--verify-only` checks
existing pinned LLVM inputs; it never authorizes acquisition. GPU commands select
an explicit provider and must use a healthy qualified physical profile. Never
run held local AMD or destructive cancellation probes in a shared long-lived
process; unknown work stays quarantined/fenced until safe physical termination.

### Packaging and product boundaries

The six CPU cross-release definitions cover Linux x86_64/aarch64 musl, Windows
x86_64 GNU/aarch64 gnullvm and macOS x86_64/aarch64. Definitions, checksums and same
host installer fixtures do not establish physical per-OS model execution,
authenticity, Windows/macOS signing or permission to fetch Apple SDKs. GPU
binaries are explicit builds, not silently included in the six CPU assets.

Installers require an explicit repository/baseURL, verify checksum and version
and install no service. A checksum from the same host detects corruption but is
not independent authentication against compromise. Nix exports CPU packages,
source/module checks and a disabled service definition. Enabled `serve` currently
refuses and `Restart=no` avoids a restart loop; a service definition is not a
working OpenAI server. No automatic firewall exposure, persistent service
installation, secret ingestion, tag/release/publication or remote peer setup is
part of onboarding.

The owner requested comprehensive capabilities and superior measured performance,
without approving arbitrary percent targets/deadlines or hidden fallback. Native
CPU/GPU source semantics, model quality, request-level throughput/tails, fairness,
energy and memory must be qualified separately. “Full GPU pipeline” currently
means the accepted original-model resident scope; it does not imply GPU
tokenization, graph capture, all-context/model support, all hardware optimized or
complete serving/distributed/multimodal products.



<a id="decisions"></a>

## Complete retained decision register — dated context snapshot

The following reproduces the current canonical decision record so every user, proposal, open-question and assumption ID remains available in this handoff. Its canonical source is [DECISION_REGISTER.md](../plans/DECISION_REGISTER.md); update that owner for future decisions. Historical status and process statements retain their dates and are superseded where the current TODO rules, later owner decisions or verified implementation say so. In particular, older A2-planned/full-GPU-pending/LLVM-blocked/archive-SHA/all-history-in-Git text is not current support or workflow. The sole binary name and MIT project license were subsequently settled; third-party/model licenses, distribution/signing and broader platform questions retain their independent gates. A complete record of an OPEN row is not a claim that every part of it is still unresolved.


**TODO-USER-001 · USER ·16 October2026:** Replace the handoff/evidence process
with `.agents/plans/TODO.md`, one `.agents/plans/<plan>/TODO.md` and independent
`.agents/runs/<plan>-<session-or-agent-id>/TODO.md` files. Keep a provenance
commit, implementation/test checkboxes, completed commit sections and a rolling
Uncommitted section. The coordinator atomically integrates agent results into
main plans; finish a plan by checking its one index line. Archive only completed
TODO sections as needed. No mandatory generated handoffs, SHA inventories,
source snapshots or run receipts. [Workflow](../../AGENTS.md). This supersedes
older process/preservation/update duties below; substantive architecture and
actual model/source/numerical/state/resource requirements stay in force.


**PULL-A1-001 · VERIFIED ·16 October2026:** Keep local cache publication in the
existing artifact owner behind optional Linux cache/diagnostic features. Reuse
caller-owned physical admission, stable process locks and exact retained file
handles; report post-ref-rename sync failure as explicit DurabilityUnknown.
No executable readiness or source identity follows from data publication. A01..09
pass within [the phase's stated limits](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md);
A2 HTTPS/providers/CLI and A3 owned-copy loader integration remain next.


**EVIDENCE-USER-002 · USER ·16 October2026:** Remove SHA inventories for ordinary
evidence logs and simplify the evidence specification. Git commit, actual command,
result and tested scope establish the record. No global evidence catalog,
per-log hash or source-file hash inventory is required. Keep hashes at real
model/artifact/content or explicit byte-identity qualification boundaries.
[plan TODO index](../plans/TODO.md). This supersedes older log
inventory/sealing instructions; design-document SHA tooling remains separate.


**CONC-USER-002 · USER ·16 October2026:** Use left-right thoughtfully at
real contention sites. The cold model catalog is not a justified migration.
Reject its uncommitted implementation and restore the accepted production
source/dependencies; preserve the experiment as rejected evidence. Require
representative access/contended-wait evidence and ownership fit before adding
a concurrency primitive. Existing direct immutable sharing and authoritative
transactions retain their appropriate design. [Correction and current review](../plans/cpu-gpu-optimization/TODO.md).


**H25 promoted-product gate ·16 October2026:** Both original Qwen3.5 models
now pass27 checks on each of local CUDA/NVIDIA Vulkan, pvl-x2 HIP and pvl-a1
CUDA/HIP:135 actual checks, including cold prelaunch negatives, plus12 actual
scalar/libm tests. Genuine512 full logits/state/trace, replay and physical
cancellation/quarantine pass; the local gfx90c hold remains. Canonical archives
and exact retirement now preserve old records inside `.agents/` with original
phase/path/hash mappings. Final repository/catalog/documentation closure remains
in progress. New Qwen/Gemma source work is isolated and does not inherit trained
execution from these results. [Current phase](../plans/accepted-baseline/TODO.md).


**QWEN-PARITY-USER-001 · USER ·16 October2026:** For Qwen, finish the
source-correct basics first, then match every capability and optimization of
llama.cpp and vLLM, then add further appropriate MTP, speculative decoding,
n-gram, DSpark2/DFlash2 and related optimizations. Maintain a version-pinned,
exhaustively enumerated capability ledger; each applicable mechanism needs
functional/source/ownership qualification and measured performance. Methods
already present upstream remain parity gaps until their own gates pass.
No guessed model/drafter pairing or synthetic acceptance counts as inference.
[Ordered implementation, complete coverage and acceptance](../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md).


**MODEL-PARALLEL-USER-001 · USER ·16 October2026:** Begin Qwen3.8 27B,
Gemma4 E2B and Gemma4 26B source/CPU implementation in parallel while current
qualification runs. Use subagents and disposable isolated worktrees where
changes overlap; main coordinates, reviews, reconciles and merges verified
pieces. The two Gemma tracks share one family implementation. Preserve the
existing engine, source identity, state/resource authority and new-model
CPU-first acceptance. No new checkpoint/SDK downloads or GPU capability claims
are authorized. This adds parallel model work while shared layout → left-right
→ XDG acquisition → further GPU optimization sequencing remains binding.
[Ownership, source gates and integration plan](../plans/model-execution/TODO.md).


**LAYOUT-RETIRE-USER-001 · USER ·16 October2026:** Remove `probes/` entirely,
moving implementation, reference tests and helpers to `crates/`, `tests/` and
`tools/` according to their real responsibility. Audit sidetree worktrees and
clean up all that are merged or superseded while preserving unique work and
active users. Resnapshot and archive historical evidence in the existing
`.agents/` documentation model; all retained historical evidence must live
inside `.agents/`, never in external build directories. Store underlying
records and original-path/hash/disposition indexes under
[`.agents/plans/evidence/`](../plans/TODO.md); obsolete build caches and
compiled executable bodies are retired after their provenance is recorded.
This extends the final H25 layout gate and does not change the layout →
left-right → XDG acquisition → GPU optimization order.

**Next: [workspace/model/feature layout redesign](../plans/accepted-baseline/TODO.md), immediately followed by the [left-right concurrency redesign](../plans/cpu-gpu-optimization/TODO.md), then individually qualified CPU/GPU performance variants.** Preserve the [five-GPU/three-CPU target inventory](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md), existing model/serving priorities and [GPU text/device orchestration roadmap](../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md). Fleet-wide capability optimization is unfinished; [native FP32 FMA](../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md) remains a proposal, not an enabled policy. Capacity remains 512; broader GPU contexts/corpora, independent CPU8K GDN divergence, genuine16K/32K qualification and product checkpoint persistence/security UX remain open. The default CPU CLI and user-authored root README are unchanged. Earlier status entries below are preserved as historical evidence.

**CRATES-USER-001 · USER ·16 October2026:** After the native backend qualification boundary, promote cohesive implemented packages under `crates/`, keep a small public core/facade, and make backends, model families and product capabilities independently selectable with real minimal/composed/`all` build gates. The bootstrap crate count is not a permanent cap; extraction preserves the same engine, state authority, physical ownership and source bits. [Workspace/model/feature redesign roadmap](../plans/accepted-baseline/TODO.md) records proposed package seams and sequencing; none is implemented by this documentation.

**CONC-USER-001 · USER ·16 October2026:** Apply [jonhoo/left-right](https://github.com/jonhoo/left-right) broadly wherever it fits read-heavy published immutable data/indexes, replacing application read locks or mutexes where semantics and measured performance justify it. Conduct this concurrency redesign pass **immediately after the workspace/layout/structure redesign**. Reinventory every owner; preserve one writer, coherent publication, deterministic replay, existing session/resource/actual-completion authority and physical retirement. [Complete candidate inventory, migration and acceptance](../plans/cpu-gpu-optimization/TODO.md) includes hidden second-copy slot retention and unregister/drain/re-register source identity, bounded guards and writer delay. This request adds the roadmap/design direction now; dependency and runtime changes belong to its later verified implementation units.


**HW-FLEET-USER-001 · USER ·16 October2026:** Initial optimization targets are pvl-l5 (current), pvl-a1 and pvl-x2: five physical GPUs and all three CPUs. Explicit SSH work is authorized on the two remote machines. Optimize capability-selected execution paths, source-exact kernels, memory placement and complete-request performance for these targets first; advertised instruction support alone is not qualification. The user also explicitly requests subagents for each implementation going forward, using independent ownership and parallel work to keep the main thread focused and efficient. [Observed fleet/ordered optimization gates](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md) and [active native CUDA/HIP phase](../plans/accepted-baseline/TODO.md) retain scope/evidence. Temporary test artifacts and reuse of already-owned approved checkpoint bytes support that implementation; new external model/SDK acquisition, persistent device/service/privilege changes and publication retain existing restrictions. Root README stays unchanged.


**OPEN-001 · USER · 15 October 2026 (MIT, AI disclosure and open contributions):** The user selected **MIT** for the Zerogravity project source and documentation, requested the standard root `LICENSE`, and approved clearly recording `license = "MIT"` for the root and optional GPU package. README must include a succinct AI disclosure influenced by [ds4](https://github.com/antirez/ds4): **human-centric design with AI-centric implementation**, substantial AI assistance without unchecked autonomous control and without claiming human line-by-line inspection. The founder **Prasanna Loganathar** explicitly welcomes everyone to build the project openly and kindly; preferred contributions start as design/architectural issues or small, contained implementation changes with a documented plan under `.agents/plans/` before nontrivial coding. [Public contribution guide](../../CONTRIBUTING.md) and [full policy](../plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md) capture the workflow. MIT applies to this project's own material; model weights, dependencies, codecs and imported assets retain their independent licensing. User approval covers documentation/contributor/licensing metadata, not new inference implementations or external service access.


**SERVOPT-USER-001 · USER · 15 Oct 2026:** After a read-only vLLM concurrency review, the user explicitly instructed strengthening every recommended concurrent serving/design/implementation plan, then a complete product-focused README rewrite with Intro, Quick start, Architecture, FAQ, Implementation status/roadmap, concise engineering bullets and high-level diagrams including joint optimization. The user requested exactly: **The blazingly fast AI execution fabric - built for composable multi-modal inference specialization at every layer, from individual kernels to distributed systems**, and early explicit text/image/video/audio/realtime scope. Examples [dotlink](https://github.com/abird-ai/dotlink) and [agentc](https://github.com/abird-ai/agentc) are structural references, not required implementation copies. [Joint serving contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) adds accurate source-verified vLLM-class bounded concurrency and joint legal batch/spec/cache/kernel/placement/QoS; [benchmark/acceptance](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md) specifies 36 PLANNED, not-run tests and truthful comparison conditions. This is authorization for docs/README and minimal doc tooling only, not new GPU serving engines, benchmarks, downloads or deployment. Existing one-Rust-engine and no-hidden-quant policies remain unchanged.


**VST-USER-001 · USER · 15 October 2026:** The user explicitly authorized clean changes to architectural docs/plans **and the minimal documentation tooling needed to repair the nested full handoff archive**, overriding the earlier design-only restriction for these documentation artifacts, while also requiring **SSD streaming to become a first-class, nearly zero-abstraction-cost storage policy** under the same source tensor model as fully resident GPU memory. DS4/DwarfStar and related SSD-streaming strategies are inspirations; one physical ResourceArbiter/TaskCoordinator and Rust inference engine remain mandatory. Direct fully resident fast path should avoid extra software overhead, but actual SSD misses necessarily pay IO; source tensor identity, verification, GPU command reader pins and real completion must be preserved across GPU/UMA/RAM/SSD tiers. Detailed specs: [first-class virtual tensor residency](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md), [source research](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md), [SSD-P0..P5/SSD-01..32](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md). **Current user decision authorizes docs/tooling, not actual SSD inference implementation or large model downloads**. Proposed policies/tests are specified but not yet verified in actual hardware. This VST-prefixed local decision avoids earlier U-040/U-041 historical numbering collisions.


**15 October 2026 design-only source-specialization clarification:** After the explicit read-only specialist-source review, user requested **all its precision/specialization gaps resolved in documentation and implementation plans, cleanly and elegantly**. The [specialization implementation plan](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) preserves the one-Rust-engine and original model priority user commitments. New identifiers **SYK-001..010 are PROPOSED design obligations; SYK-011..016 are OPEN policy/source/permission questions**, NOT additional user-frozen ABI approvals. Do not conflate with earlier v5.1 addendum's historically reused U-040/U-041 or overwrite the shared U-040 native model implementation entry. No new code/model/driver changes were requested by this instruction.


**Read this before interpreting any spec.** Project date 2026-10-10; provenance is the originating conversation. Identifiers are stable for future design reviews. Status: `USER` = explicitly requested or endorsed; `PROPOSAL` = assistant-origin recommendation, not separately approved; `OPEN` = requires a measured or policy decision; `ASSUMPTION` = working interpretation, reversible.

### Explicit user decisions

| ID | Status | Decision, in the user's intent |
|---|---|---|
| U-001 | USER | Project name **zerogravity** (lowercase). |
| U-002 | USER, SUPERSEDED PATH | Originally requested comprehensive agent-readable plans, specs, decision history and takeover documentation under the relative `./zerogravity/.agents/{docs,plans}` path, via **dotlink l5**. Its location requirement is superseded by U-013; the documentation scope still applies. |
| U-003 | USER | The inference engine's own implementation is to be **entirely Rust**; record a strict desire for Rust-authored kernels, not a C++ inference engine wrapper. What counts as unavoidable vendor runtime bindings is clarified in O-003. |
| U-004 | USER | Support standard model families, expressly **Llama**, **Gemma 4**, **Qwen 3.5+** (dense/MoE/hybrid) and additional newer models, rather than specialize for one fixed checkpoint. |
| U-005 | USER | Support unusually large / specialized models including **Qwen3.8-Flash-Next (125B backbone + auxiliary table/MTP)** and **GLM-5.3-Flash (~321B total)**, especially on **Ryzen AI Max+ 395 / Radeon 8060S / AMD Strix Halo (gfx1151)**. |
| U-006 | USER | Target vLLM/SGLang-level peak server efficiency while retaining llama.cpp-like accessibility/portability and Maya/Gufo/DwarfStar-like hardware-specific extreme optimizations. |
| U-007 | USER | On suitable hardware activate maximal specializations; **otherwise fall back to general model execution**, without degrading unrelated optimized operations. |
| U-008 | USER | Combine architectural learning from vLLM, SGLang, llama.cpp, Maya, Gufo, Strata, DwarfStar (antirez/ds4), plus complementary runtimes and Rust projects. |
| U-009 | USER | Adopt the **Semantic IR-based** architecture as a way to express model computation once and allow specialized execution without a hot-path abstraction penalty; ambition is to *outclass* existing engines where demonstrably possible. |
| U-010 | USER | Prior stage was **research only**, then explicit request to refine architecture, now explicit request to persist comprehensive plans and handoffs. No code implementation was asked in this turn. |
| U-011 | USER | The record must preserve **all user decisions, detailed design decisions and pending questions**, so external LLM agents can implement from the files without relying on chat history. |
| U-012 | USER | Performance ambition is across both model breadth and specialized hardware. There is **no user-approved target performance percentage or deadline**; faster-than-everything is a goal, not a result. |
| U-013 | USER | **Canonical project location is `/home/pvl/spaces/abird/src/zerogravity/`**, explicitly correcting the original relative placement. Move existing documentation there, preserve files, update references and revalidate. This supersedes only the destination in U-002. |
| U-014 | USER | **Incorporate the best of the research/inference architecture and src/zerogravity Semantic IR architecture, adapting all plans to a unified stronger design.** The user approved the combination goal; this does not automatically approve each old council-specific Rust ABI, license or performance threshold. |
| U-015 | USER | Conduct **comprehensive new research** to design zerogravity for maximum potential to surpass existing inference projects while preserving long-term maintainability, clean adaptability to future technologies and robust extensibility at **every stage of the pipeline**. State measured evidence versus conjecture and update the authoritative plans. |
| U-016 | USER | Explicitly include the architectures and pipelines demonstrated by [mikealanni/gufo](https://github.com/mikealanni/gufo) and the October 11 Strix Halo Reddit report on **Qwen3.8 Flash-Next 1,800+ tok/s cooperative GPU+XDNA2 NPU prefill**, making these capabilities fit elegantly into zerogravity rather than becoming model-specific parallel runtimes. This is a **design/support requirement**, not proof that XDNA2 code is implemented or automatically safe to enable. |
| U-017 | USER | **Generalize all project plans, specs and architectural interfaces away from external project names**, with an exception permitting concrete naming/reference to verified MIT-licensed sources. Retain all upstream work as **identified, attributed prior art and research inspiration**, rather than as architectural base classes or cloned engines. Build zerogravity's **own original, modular and composable AI execution architecture** so new model/accelerator/quantization/task/pipeline specializations do not require rewriting new inference engines. Preserve model-family and hardware/protocol identifiers when materially necessary, and do not lose source provenance or license restrictions. |
| U-018 | USER | **Perform another comprehensive critical review/refinement of the entire architecture and detailed specifications**, aiming for the highest-performing and most composable runtime across AI execution tasks, while remaining **minimal whenever complexity has no proven value** and modular enough to incorporate novel models, devices, kernels and AI task pipelines without engine rewrites. This requests architecture/design refinement, not immediate implementation or proof of benchmark superiority. |
| U-019 | USER | **Verify and generalize accelerator/runtime architecture coverage** beyond local Strix Halo: explicitly include NVIDIA DGX Spark/GB10 coherent Grace–Blackwell, discrete and datacenter/rack-scale Blackwell, Google TPU, possibly Cerebras wafer-scale (user phrasing “cerberus” also corresponds to a separate parallel-decoding technique), and other programmable AI accelerators. Maintain one composable source-independent runtime that can absorb **native-kernel, AOT graph, deterministic/static dataflow and other specializations at their natural execution granularity** instead of cloning another engine. This is an architecture/research requirement, not proof of actual hardware support. |
| U-020 | USER | **Distributed and networked execution is an additional first-class zerogravity goal**, not just fastest inference on one machine. Support heterogeneous GPUs/CPU/NPU and programmable accelerators **across multiple machines, including trusted remote tunnels**, workload sharing/sharding and a range of topologies: layers and their cache colocated, different model layer groups on different machines, layer compute and its KV/recurrent/cache on different machines, and other contemporary/near-future combinations. Preserve performance, composability, auto-or-pin topology policy and mathematical/state correctness without writing a different runtime for each placement strategy. This requests architectural design/spec updates, not code/cluster deployment or permission to connect remote peers. |
| U-021 | USER | **Build a native multi-model model router into zerogravity**: resolve and load balance requests for as many models as needed, support separately configured model parameters/profiles and simultaneous serving where physical memory allows, with one OpenAI-compatible HTTP API interface. Preserve an **extremely simple one-command local run/serve experience** comparable to easy local model runtimes, while keeping every aspect of model/quant/placement/replicas/context/sampling/routing/API/device/network resource behavior configurable on demand. This is a **design requirement**, not implemented CLI/API or permission to download models/open network services. |
| U-022 | USER | Perform a **final comprehensive editorial and technical refinement** of zerogravity's full architecture/design specifications, organize them for clear authority and maintainability, prepare a **publication-quality technical whitepaper**, and deliver a **complete implementation handoff for LLMs and autonomous agents** to build and validate the system without reconstructing the chat. Retain all historical decisions/research/phase IDs and be honest about design-only status, prior art and unmeasured performance. No actual runtime, PDF, Git, model download or machine changes were expressly requested. |
| U-023 | USER | **Implement only zerogravity's project/bootstrap packaging foundation**, adapting the existing `/home/pvl/spaces/abird/src/dotlink` Rust/crane release/quality conventions as a **read-only reference**: a real minimal binary and project-defining `README.md`, Rust build/format/lint/test tooling, pinned Nix/crane flake and NixOS module, cross-build definitions for major Linux/Windows/macOS architectures, GitHub Actions build/test/release workflow, checksum-protected install scripts and reproducible release artifacts. **No model inference, tokenizer, serving/router/API, device kernels or distributed execution implementation in this phase.** Actual GitHub remote/release and physical cross-platform builds must be separately verified. |
| U-024 | USER | **Begin Phase 1 implementation now**, transitioning from packaging-only to the first real Rust model-source/reference foundation and ultimately running both real Qwen3.8-27B and Gemma 4 26B-A4B-it through one runtime. The user has **not** separately authorized a multi-gigabyte model download, device firmware or privilege changes, Git publish/release or remote peer enrollment. This supersedes U-023's *temporary implementation scope*, not its validated packaging infrastructure. Phase 1 partial implementation is recorded in the historical Phase 1 source/reference handoff. |

| U-025 | USER | **Reordered practical acceptance milestones**: first Qwen3.5 **0.8B** and **2B**, each CPU text inference first, then complete native **NVIDIA CUDA**, native **AMD ROCm/HIP**, and **Vulkan** on both physical GPU families; only after those, **Gemma 4 E2B**, **Gemma 4 26B-A4B**, and **Qwen3.8 27B** should receive source-correct **CPU execution first**, with their GPU work deferred. Preserve a shared Rust-native semantic execution engine and reference/fallback semantics; a physical specialized kernel proof does **not** qualify full-model GPU inference. See the Qwen3.5 CPU/GPU handoff for exact delivered partial implementation and open gates. |
| U-026 | USER | **Complete rigorous CPU numerical, prefill/state and genuine long-context qualification for Qwen3.5 0.8B and 2B before moving to full-model NVIDIA CUDA / AMD ROCm / Vulkan integration.** The new verification must distinguish actual model CPU token processing from synthetic 32K operator/position tests and independently trace model/reference mismatches, failing closed rather than widening thresholds to hide differences. Current status: substantial added tests and guarded chunked prefill, but the CPU gate is **PARTIAL/BLOCKED** on 2B strict distribution drift, missing per-layer reference, and 4K..32K real-model evidence. See the Qwen3.5 CPU numerics/context handoff. |
| U-027 | USER | **Move on to native GPU development once Qwen3.5 0.8B and 2B have verified real CPU semantics through 8K.** Defer 16K and 32K whole-model source/reference qualification to later repeatable local tests, rather than treating them as prerequisites for CUDA, ROCm and Vulkan integration. Keep all independent short numerical, layer, state, error and 8K context acceptance checks and thresholds intact; the deferred longer tests remain UNQUALIFIED. This supersedes only the 16K/32K-before-GPU ordering implied by U-026. Two explicit CPU scopes now exist: Phase 1 GPU-development admission (16/16 PASS) and deferred extended-context (16/20; four missing). The first native accelerator source-matvec integration seam has passed CPU fallback/failure transaction tests; physical complete GPU model execution is NOT yet qualified. See [Phase 1/P07 handoff](../plans/numerics-and-context/TODO.md). |
| U-028 | USER | Commit Qwen3.5 CPU Phase 1 and accelerate P07 GPU implementation while independently examining high 8K shared top-20 logprob errors (~0.2037/0.1968). Seek source-grounded improvements without widening numeric thresholds. Baseline initial phase commit is a965737; GPU physically qualified inference remains pending. See [8K numerical drift report](../plans/numerics-and-context/TODO.md). |
| U-029 | USER | Organize three parallel workstreams: GPU implementation, accuracy investigation, and multimodal architecture design with independent substreams/subagents where tools permit. Keep model design composable, native Rust and accurately source-grounded. Standalone Codex/Claude agents were discovered on host but authentication failed (Codex HTTP 401, Claude not logged in), so NO independent autonomous agent was claimed to have run. See [multimodal implementation checklist](../plans/multimodal/TODO.md). |
| U-030 | USER | Continue into the next implementation phase, preserving three parallel workstreams. Deliver actual native Rust/CubeCL source-quantized GPU regions beyond Q4_0, improve independent numerical understanding without relaxing parity thresholds, and advance additive multimodal architecture via non-disruptive mock contract tests. P07-B source-mixed checkpoint host parity and five MM0 checks were implemented; physical GPU devices and LLVM23 CUDA/HIP toolchain remain prerequisites for real backend validation. See [P07-B/MM0/numerics handoff](../plans/accepted-baseline/TODO.md). |
| U-031 | USER | Continue into P07-C GPU development: source-identity-safe persistent device weight residency, bounded physical buffer leases, strict source-only fallback and completion/event fencing, reducing repeated per-token uploads without turning CPU-only Zerogravity into a GPU-dependent binary. Preserve the 16/16 CPU Phase1 gate, defer 16K/32K and do not label unverified physical GPU execution as passed. See [P07-C exact implementation handoff](../plans/accepted-baseline/TODO.md). |
| U-032 | USER | Continue P07-D GPU development from P07-C: eliminate repeated host GGUF source-weight decoding on GPU cache hits, establish typed asynchronous device lease/event lifecycle and fail-closed completion/cancellation/epoch rules without perturbing CPU source math, and preserve clear separation between host tests and actual GPU hardware qualification. A decode-once StaticSource / DynamicStep split was implemented; device-independent DeviceEventLedger tests and MM-06 duplex/mock cancellation validation pass. Physical asynchronous CubeCL GPU integration remains later work. See [P07-D handoff](../plans/accepted-baseline/TODO.md). |
| U-033 | USER | Continue to P07-E from P07-D: integrate a real async CubeCL readback with the typed DeviceEventLedger, keep original synchronous CPU model semantics and sole TaskCoordinator StateRoot writer, and preserve explicit hardware qualification evidence. A source-typed async completion bridge and fail-closed future cancellation were implemented and host-tested; multi-device GPU concurrency/full-state execution remain later work. See [P07-E implementation handoff](../plans/accepted-baseline/TODO.md). |
| U-034 | USER | Recover interrupted P07-E state, validate the live committed implementation and continue the next GPU safety phase. P07-F retains the actual CubeCL source, dynamic and output handles on an uncertain/dropped Future in a process-wide service-quarantined keepalive registry, and accounts physical resident GPU weights plus activation/scratch/output under one per-provider budget. All CPU trained source and Phase1 8K acceptance remain unchanged; actual native device cancellation and global multi-model ResourceArbiter remain unqualified. See [P07-F handoff](../plans/accepted-baseline/TODO.md). |
| U-035 | USER | Revalidate and continue all Zerogravity work after the interrupted P07-E continuation, then author a complete comprehensive handoff markdown retaining the original idea/design, every phase decision/specification/implementation and pending item going back to project start, with a ready-to-run takeover prompt. Live baseline was already P07-F 602c246, not the old interrupted P07-E draft. P07-G1 adds source-neutral global device admission and bounded fair queue tests (CPU control plane only); physical GPU worker integration remains pending. Next agent must read the latest comprehensive handoff progressively in full before implementing. See [P07-G source scope](../plans/accepted-baseline/TODO.md). |

| U-036 | USER | Take over the live source-controlled Zerogravity repository at documented P07-G1 handoff commit `f28dc10`; revalidate the complete historical source/quality provenance and continue **P07-G2** binding attested physical CubeCL service/adapter identities to the single shared `ResourceArbiter` and P07-F/E residency, events, buffer keepalives and fail-closed cancellation. Preserve independently owned GDN accuracy and multimodal streams, canonical Qwen-first model order, 8K Phase1 acceptance and 16K/32K deferral. Do not claim hardware inference from compilation/mocks; keep concurrent agents' files, avoid bulk staging/stashing, prohibit unauthorized downloads/driver changes/push/releases. This implementation slice delivers conservative P07-G2a global physical envelopes and source-readback guards; actual cross-model fair GPU workers and full physical CUDA/ROCm/Vulkan model validation are still pending. See [P07-G2 phase handoff](../plans/accepted-baseline/TODO.md). |

| U-037 | USER | The user confirmed the `dotlink-l5` connector should now expose GPU hardware and requested immediate continuation to the next Zerogravity implementation phase, then explicitly requested revalidation/resumption after interruption. Revalidated clean `7e6e4f4` G2a baseline, real `/dev/nvidia*`, DRM render and KFD devices; physically exercised native CubeCL Vulkan on NVIDIA RTX 3060 and AMD Renoir, implementing P07-G2b shared per-physical-card bounded fair workers and strict original Qwen GGUF numerical/StateRoot gates. NVIDIA Vulkan source-region workers and partial-model Qwen0.8B/2B are bit-exact; AMD RADV physical readback is real but source FP32 arithmetic differs and therefore remains **numerically unqualified**, with failed Qwen StateRoot publication refused. CUDA/HIP LLVM23 development toolchain, full GPU GDN/KV/attention and physical cancellation remain pending; 16K/32K CPU contexts deferred. The verified phase evidence/limitations are in [P07-G2b](../plans/accepted-baseline/TODO.md). |

| U-038 | USER | After successful clean P07-G2b GPU worker commit `c3e5fca`, user requested **next implementation phase**. P07-G3a continued with REAL NVIDIA and AMD GPU access, resolving AMD Renoir RADV's observed original Qwen3.5 Q4_0×Q8_0 native-F32 fused-accumulation mismatch by a separate conservative WideF64 GPU compile-time specialization. BOTH physical cards now pass exact original GGUF source-bit four-dispatch 0.8B/2B GPU worker checks; AMD four-token real CPU+one-GPU Qwen hybrid complete-vocab logits/StateRoot identical to pure CPU. Original fast NativeF32 remains separately diagnosable and still FAILS strict source equality on AMD (PASS NVIDIA). Explicit opt-in feature-only physical submitted-Future-abort probe shows both GPUs retain five original buffers in process-lifetime quarantine, conservatively fence/retain 256MiB logical GPU provider charge, reject second model, and publish no stale StateRoot. Do **not** call this proof of driver cancellation/recovery, measured allocator VRAM, fast GPU inference or complete Qwen GPU graph. Native CUDA/HIP compilation needs unavailable LLVM23 and vendor SDK; CPU Phase1 16/16, extended 16/20 deferral, GDN long-context accuracy and MM0 v5 independent streams unchanged. See [P07-G3a physical evidence](../plans/accepted-baseline/TODO.md). |

| U-039 | USER | After `67dca72` G3a commit, requested a **fresh adversarial review** of bugs, correctness, regressions and a robust, simple, intuitive, future-proof, low-maintenance and composable software architecture. Review verified current live `main` and intentionally implemented bounded fixes instead of a wholesale rewrite: pre-submit GPU validation before physical terminal fencing; scratch-aware LRU of idle source weights; same-file-descriptor GGUF source loading/reparse (prevent equal-length pathname replacement source mix); normal CPU intermediate chunk prefill elides discarded logits with real 0.8B/2B parity; correct GPU fair-queue `selected` vs physically `completed` counters and renamed globally destructive card-epoch API; checked/limited untrusted model head geometry; and non-allocating per-card global fence hot path. Tests covered actual NVIDIA/AMD Vulkan positive dual-model work, actual invalid activation, invalid original FP16 scale, underbudget rejection and recovery, real Future-abort quarantine; preserve independent CPU GDN 8K/16K/32K and MM0 streams. The detailed prioritized findings, changes, test evidence, remaining production blockers and smallest safe next steps are in [fresh quality review](../plans/accepted-baseline/TODO.md). No unapproved drivers, model downloads, remote release or Git push. |

| U-040 | USER | After the U-039 fresh correctness and architecture review, requested **resolving remaining issues** with clean, intuitive, scalable, future-proof, maintenance-light, composable Rust design. Continued live repo work on clean `e9312ad` baseline: introduced one immutable `Arc<LoadedQwenModel>` for original GGUF+predecoded layer constants shared across independent Qwen session StateRoots, backward-compatible direct load, minimal explicit `ModelCatalog` alias/lazy per-source singleflight load/retirement; real concurrent source-pinned Qwen0.8B/2B sessions and original GPU shared-resident source reuse verified. Added source-audited physical F32/WideF64 Q4 region latency probes and device-wide GPU VRAM/GTT measurements keyed by attested PCI BDF, which revealed driver memory pools remain reported allocated after logical provider release. Offline native CUDA and HIP compiles still block at LLVM23 `LLVM_SYS_231_PREFIX`, independent CPU GDN 8K arithmetic cause remains open, full GPU model and native HTTP/multi-model router **not claimed complete**. Preserve original models, source/tolerances, MM0 and explicit 16K/32K deferral. See [U-040 implementation evidence and remaining honest gates](../plans/accepted-baseline/TODO.md). |

| U-041 | USER | Absorb the newly committed 15 Oct v5.1 universal multimodal, source specialization, joint serving and virtual SSD-tensor-residency design proposals, then continue real GPU execution in verified phases; remain one Rust source-verified engine, one TaskCoordinator/StateRoot publisher and one physical resource authority. P07-H1 physically source-qualified all 187 Qwen3.5 0.8B/2B GGUF GEMV tensor regions and full-vocab/StateRoot on AMD Renoir and NVIDIA RTX3060 Vulkan; P07-H2 added actual GPU source-compatible RMSNorm with exact original f32-square, f64 sum, f32 mean/sqrt/reciprocal rounding, yielding 187 GPU matrix + 49 GPU norm regions = 236 source-bit-equivalent operations on EACH checkpoint/device combination. The rest of GDN/attention/KV/recurrent and complete device graph are not yet implemented/qualified. [H2 source/hardware evidence](../plans/accepted-baseline/TODO.md). |
| U-042 | USER | Explicitly granted FULL Zerogravity repository ownership, removing the prior constraint to preserve or partition any other agent changes. Revalidate concurrent Git before editing, reconcile all v5.1 docs and code into one coherent implementation, clean up as appropriate, update current handoffs and roadmap, progress in safe validated GPU stages, and do not fabricate full-GPU/native CUDA/ROCm/server or source-accuracy claims. Preserve original source model checkpoints/quality gates and do not push or publish releases without authorization. |

| U-043 | USER | The user explicitly superseded any prior assistant-authored README copy edits: the live README is USER-OWNED and MUST be preserved exactly as the user updated it. Do not revise, replace, clean, stage or bundle README.md into a GPU phase commit. Documentation snapshots may reference its current bytes, but must not mutate README. Continue independently verified P07-H phases. |


| U-044 | USER | Requested validation and continuation after P07-H3 grouped per-head RMS normalization and interrupted two-token device-source acceptance. Revalidated Git `4ab15a4`, original Qwen3.5 0.8B/2B GGUF SHA hashes and AMD Renoir/NVIDIA RTX3060 Vulkan access. The initially failing two-token acceptance MAP was counting final-only vocabulary and output-RMS operations twice; fixed count without relaxing source math, GPU correctness or state acceptance. On BOTH checkpoints and BOTH physical Vulkan GPUs, the original Rust CPU Qwen model exactly matches **530 REAL physical GPU completed regions across two tokens** (266 selected: 187 source GEMV, 49 residual/final RMSNorm, 30 independently grouped GDN/attention head norms; subtract 2 final-only operations), all output bits, complete 248,320-vocabulary logits, committed root and state fingerprint, with ZERO outstanding global logical reservation after real readback. Physically forced post-launch grouped-RMS Future abandonment separately passed on BOTH cards: 5 original GPU handles quarantined, 256MiB retained/fenced and root unchanged; not driver cancellation/recovery. User-owned README MUST remain exactly as supplied SHA256 `736c794cf68d82cf47b20fc8409e4876ab107888a38f98805bdedbde10d2aaf8`. The actual GDN convolution/recurrent, attention/softmax/RoPE/KV, FFN SiLU and device-resident inference graph are NOT complete; do not invent entire-GPU qualification. See [H3 real hardware evidence and H4 continuation](../plans/accepted-baseline/TODO.md). |

| U-045 | USER | Continued physically validated GPU execution after P07-H3. Implemented P07-H4 first original-source **GDN causal convolution island** through an additive typed kernel returning BOTH the raw pre-SiLU activation and private proposed 3-frame history; the one CPU QwenState still owns the commit. One-kernel GPU output failed source-bit comparison on the SECOND original Qwen token (1,840/6,144 raw discrepancies, history exact); split source FP32 multiplication rounding into actual stored device products and separately ordered reduction (two real kernels). Both original Qwen3.5 0.8B/2B checkpoints on BOTH AMD Renoir RADV and NVIDIA RTX3060 Vulkan then passed **566 genuine physical GPU region completions over two tokens**, including 36 conv raw/next-history source bit checks, 187 matrices/49 residual norms/30 grouped norms per token, exact full vocab logits and StateRoot, and zero remaining provider logical charge. Real conv Future abandonment on both GPUs retained 6 original handles, fenced 256MiB charge, rejected another model and preserved Qwen root. This does **NOT** implement device-resident recurrent GDN matrix, attention/KV/FFN or full-GPU Qwen; native CUDA/HIP LLVM23 still blocked; independent CPU 8K GDN drift remains open; original 16K/32K deferred. User-owned README preserved bit-for-bit. [Detailed H4 physical results and H5 direction](../plans/accepted-baseline/TODO.md). |

| U-046 | USER | Requested a **FULL comprehensive agent takeover handoff Markdown** covering the original Zerogravity vision and whitepaper, every specification/design iteration, all decisions/proposals/negative experiments, every implementation/test/commit phase since beginning, incomplete work and exact instructions plus ready-to-paste next-agent prompt. Keep everything in `.agents/docs/` or `.agents/plans/` and do not skip historical sources; retain the user-owned README verbatim. Delivered a new dated, practical complete takeover guide [plan TODO index](../plans/TODO.md) and regenerated the canonical **byte-for-byte lossless historical archive** [FULL_COMPREHENSIVE_IMPLEMENTATION_HANDOFF_2026-10-14.md](../plans/TODO.md), which embeds the dated guide and every earlier versioned specification/decision/phase source with SHA/byte counts. H4 `0cad60a` remains last *implementation* phase; upcoming H5 GPU GDN recurrent state is implementation work, not completed by this documentation task. No unapproved downloads, code alterations, pushes, tags/releases or README changes. |

| U-047 | USER | Requested immediate continuation of the live repository from handoff2125612/H4 through **P07-H5 source-exact GPU GDN recurrence**, after complete takeover-guide reading and live repository/model/hardware/test revalidation. Preserve one Rust engine, immutable source identity, independent StateRoots, physical arbiter/fair workers and actual completion; compare output AND proposed state bits on trained nonzero causal tokens across both checkpoints and both Vulkan vendors, with rollback, admission and post-launch handle quarantine. Continue later attention/KV/FFN/resident/vendor/model milestones in small independently verified phases. Root README remains user-owned; no model/SDK downloads, privilege changes, push/tag/release. [H5 source and acceptance ledger](../plans/accepted-baseline/TODO.md); continued under the same authorization with [H6 Q/K L2 source/rollback/physical acceptance](../plans/accepted-baseline/TODO.md). |

| U-048 | USER | Requested architecture/design validation of native kernels, speculative execution, retained replay and heterogeneous prefill, including relevant task/product capabilities and measured performance objectives. Preserve the one-Rust-engine/source/StateRoot/resource boundaries. This is design qualification, not authorization for downloads, machine tuning or an unmeasured speed claim. [Architecture and planned gates](../research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md); [pipeline contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md). |

| U-049 | USER | Requested removing external implementation/release references from this design update while retaining its technical ideas and design constructs. Generalize document names, text, links and acceptance identifiers; remove imported comparison/provenance/benchmark narrative; regenerate the archive and SHA manifest. Runtime and root README remain unchanged. |

### Strongly recommended architectural choices (not independently approved yet)

| ID | Status | Proposed choice and why |
|---|---|---|
| P-001 | PROPOSAL | Three capability levels: portable semantic reference; tuned generic kernels; extreme model-family/hardware specialization. Graceful per-operation fallback with state-layout compatibility. |
| P-002 | PROPOSAL | Three compiler layers: Semantic IR (exact/effectful model math), Optimized Execution IR (placement/fusion/layout/lifetimes), Device Plan (kernels/events/capture/scratch). |
| P-003 | PROPOSAL | Semantic IR represents recurrent and sparse state and its mutation/rollback as **effects**, not just pure tensor DAG. |
| P-004 | PROPOSAL | Rust-owned kernel registry and compiler selection with CubeCL as candidate, but keep independent CPU/Rust reference and allow target-specific Rust GPU intrinsic/backend work if CubeCL codegen is inadequate. |
| P-005 | PROPOSAL | Rust-authoritative scheduling with mixed prefill/decode/verify batches, bounded admission, latency/throughput modes, no CPU routing round trips in resident MoE hot path. |
| P-006 | PROPOSAL | One global budget arbiter, many specialized caches: dense/experts/KV/recurrent/indexer/lookup/scratch; zero assumption that GPU-visible GTT is truly available RAM. |
| P-007 | PROPOSAL | GPU route-aware MoE grouping across sessions and speculative rows, overlap-safe expert slot management, heterogeneous placement, no universal cache policy. |
| P-008 | PROPOSAL | Persistent, versioned prefix-state snapshots including KV, GDN/KDA, indexer, n-gram history, multimodal positions, speculative frontier; COW where beneficial. |
| P-009 | PROPOSAL | Speculation algorithms behind an exact verifier: native MTP, DSpark where model-compatible, EAGLE/draft model, prompt-lookup, n-gram and disabled; adaptive policy based on accepted **committed** tokens per full cost. |
| P-010 | PROPOSAL | Strict default sampling distribution preservation when available; explicit opt-in for approximate/opportunistic sampling. Quantization changes never implicit. |
| P-011 | PROPOSAL | GGUF and Hugging Face safetensors readers, semantic model adapters, tokenizer/chat-template correctness; recognized metadata != runnable unknown architecture. |
| P-012 | PROPOSAL | Multiple device execution strategies: single-device, hetero CPU/GPU, pipeline, tensor parallel, expert parallel, prefill/decode disaggregation where economics justify. |
| P-013 | PROPOSAL | Rust core and kernel source; use platform GPU drivers, APIs and vendor math facilities via safe Rust wrappers when needed; avoid linking another *inference runtime* as core. |
| P-014 | PROPOSAL | Test-driven optimization with CPU/FP64 oracles, reference logits, checkpoint parity, actual device benchmarks, quality regression, controlled ABBA/BAAB tests; publish failed experiments. |
| P-015 | PROPOSAL | Keep front-end as single binary/SDK with OpenAI/Anthropic-style APIs, streaming, tools, reasoning semantics, observability and simple local CLI. |
| P-016 | PROPOSAL | Prefer clear modular Cargo workspace, no performance-sensitive per-token dynamic-dispatch overhead, few crates, minimal operational deps. |
| P-017 | PROPOSAL | A plan cache keyed by IR/kernel/driver/device/quant/shape/numeric policy; invalidated on relevant changes; no invasive speculative adaptation until measurement proves value. |
| P-018 | WORKING TECHNICAL BASELINE | Reconcile the two design programs with **SIR** state/math, **RIR** legal fusion/layout, independent **KIR** Rust kernel provider branch, **CIR** asynchronous resource/stream/IO command graph, validated immutable ExecutableBundle; HardwarePolicy and StepPlan distinct from KernelPlan. This is a design decision from comparison/research, not a new user-specific API approval. |
| P-019 | WORKING TECHNICAL BASELINE | Adopt research's EngineCore single logical writer, bounded EngineMsg/EngineEvent, persistent InputBatch and StepPlan/WorkerDelta, CPU/IO worker isolation and typed mode receipt, validated against effectful model-state transactions. |
| P-020 | WORKING TECHNICAL BASELINE | Implement generic paged KvPool + chained hashes with optional radix index over same pool; extreme contiguous reserve-all pool only when measured, both under one ResourceArbiter; generalize hybrid prefix sharing to 3+ state-domain groups. |
| P-021 | WORKING TECHNICAL BASELINE | Keep GGUF as core input, optional derived acceleration pack, content-addressed ModelStore, precise tokenizer/pretypes/chat/grammar FSM, encoder/embedding/rerank and LoRA adapter capabilities; scope each release independently. |
| P-022 | WORKING TECHNICAL BASELINE | Rust kernel provider ABI independent of CubeCL; evaluate 2026 independent Rust CUDA tracks, AMD Rust gfx1151 ISA path and CPU/Vulkan/Metal fallback. Vendor driver FFI/non-Rust kernel source policy remains OPEN. |
| P-023 | WORKING TECHNICAL BASELINE | Use first-party independent ML compiler streams, schedule search, the MLIR effect model, attention-kernel serving autotuning and equality-saturation/optimization engines as *architectural research sources*, not required large runtime dependencies. Optimize pure regions offline with bounded compile time; no hot-path search. |
| P-024 | WORKING TECHNICAL BASELINE | Combine research frontier ring/CommitPlan with project full-state causal transaction; semantic, cache schema and async lease/timepoint effects must all remain valid across speculation/fallback. |
| P-025 | WORKING TECHNICAL BASELINE | Deliver a Stage-0 architecture/kernel/format/CPU vertical slice before freezing dozens of crates and advanced API; detailed crosswalk in INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md; no phase done without actual tests. |
| P-026 | WORKING TECHNICAL BASELINE | Treat XDNA2 and future NPUs as capability-probed **compute devices** with external runtime/driver and memory topology, not separate language model runtimes. RIR may partition semantically independent GEMM output rows; KIR selects an NPU GEMM provider; CIR schedules prepack, NPU/GPU concurrent work, event/fence join and full state commit. Missing NPU or unsupported shapes retain GPU-only fallback. |
| P-027 | WORKING TECHNICAL BASELINE | Treat upstream Gufo's distinct ASR/TTS/image/video work as **TaskProgram stage-graph research**: model/task-specific control loops share SIR/KIR/CIR/device memory but encoder, vocoder, denoiser and media outputs cannot all be forced into the decoder-only next-token loop. Maintain separately qualified capabilities. |
| P-028 | WORKING TECHNICAL BASELINE | Because the supplied Gufo fork reports hard GPU/NPU **machine hangs during fabric clock transitions**, its NPU path is experimental/opt-in and GPU-only is the safe default until a **specific hardware/firmware/XRT/ROCm/power configuration** has demonstrated stable concurrency, numerical quality and end-to-end win. Do not automatically modify privileged fabric clocks, IOMMU, NPU suspend or systemd configuration. |
| P-029 | WORKING TECHNICAL BASELINE | Keep **compute owner vs weight residency owner vs state owner** independent: an independent dGPU↔iGPU expert execution design, Gufo-style same-SoC iGPU↔NPU GEMM slicing, and optional CPU/SSD are different CIR placement candidates under one physical ResourceArbiter. |
| P-030 | WORKING TECHNICAL BASELINE | For NPU block-float compute, store source checkpoint precision separately from **temporary device compute encoding**; the source research's nine-byte/eight-value block-float example has quantization error and MUST be validated by per-operator/full-logit and model-quality tests, not merely 30 matching greedy outputs. |
| P-031 | **AUTHORITATIVE V3 DESIGN** | Replace source-project-specific architecture claims with **original zerogravity contracts**: ModelDefinition, SemanticOp/StateDomain, TaskProgram, legal RIR, independent KIR provider, CIR async resource graph, ExecutableBundle, DeviceCapabilities, ResourceArbiter and TaskCoordinator/StepPlan. Prior art documented separately and source-named examples optional only when root/file MIT permits. |
| P-032 | **AUTHORITATIVE V3 DESIGN** | Model semantics and task control are orthogonal; a TaskProgram can compose text, audio ASR, speech synthesis/vocoder, image/video diffusion, encoder/pool/rerank, agent workflows and future compute without a second device runtime or fake decoder-only loop. Each has typed iteration/frontier/output and independent model/task qualification. |
| P-033 | **AUTHORITATIVE V3 DESIGN** | New model/operator/quant/device/kernel/IO/source/task/frontend extensions register narrow versioned capability interfaces with independent reference/fallback/evidence and cannot require changing unrelated API/session/scheduler/memory ownership. Adopt minimal modules and a mandatory architecture extension drill. |
| P-034 | **AUTHORITATIVE V3 DESIGN** | License-aware prior-art catalog owns external project/author links, repository- and per-file SPDX verification, source revision and benchmark provenance. Normative plans use mechanism names; a MIT-licensed reference is a possible source candidate, not a mandatory dependency or blanket permission to copy bundled differently licensed code/weights. |
| P-035 | HISTORICAL V3 BASELINE (REFINED BY V4) | Typed SIR math/effects → legal RIR regions → KIR candidates → CIR async resources → ExecutableBundle runs **directly**, while task control coordinates stages/loops. v4 retains conceptual compiler stages but minimizes persistent interfaces/formats; StateRoot owner/physical budget ownership refined in P-036…P-045. |
| P-036 | **V4 ARCHITECTURAL BASELINE** | Exactly two enduring compilation representations initially: **SemanticProgram** and **ExecutableBundle** (containing async CIR commands). RIR and KIR are transient optimization/provider views; no requirement to implement four serialized IR dialects or hot interpreters. |
| P-037 | **V4 ARCHITECTURAL BASELINE** | StateDomains prepare immutable candidate versions; only **TaskCoordinator per session/shard** performs one logical `PublishRoot` after all state domains and physical events validate. Device/CIR commands report ready/failed, never publish semantic session state. |
| P-038 | **V4 ARCHITECTURAL BASELINE** | Maintain **coarse global physical memory admission/reservation**, with fast device-local slab/page suballocators under generation/event-backed leases; no global per-token allocation lock or impossible promise that the OS cannot change available memory. |
| P-039 | **V4 ARCHITECTURAL BASELINE** | A small trusted built-in `TaskDriver` controls AI stage loops and typed outputs; no general workflow DSL or `ExternalAction` with filesystem/network privilege inside core. External RAG/agent/tool orchestration belongs above the TaskRequest boundary. |
| P-040 | **V4 ARCHITECTURAL BASELINE** | **Async-first device-native StepProgram** with persistent parameter/row buffers, routing/sampler/draft verification and output event transfer where supported, avoiding mandatory CPU↔GPU synchronization for every model step; host-loop reference/fallback remains correct. |
| P-041 | **V4 ARCHITECTURAL BASELINE** | Typed fixed-rank bounded dynamic shapes, ragged row layouts, runtime kernel shape/quant/ISA/precision guards and bounded shape-band/compiled-bundle cache; use portable fallback on unsupported shape rather than unlimited JIT. |
| P-042 | **V4 ARCHITECTURAL BASELINE** | Output after committed root uses sequence/epoch identity, ordered at-most-once **delivery attempt** semantics; optional resumable cursor/ACK/durable store separately scoped. Distributed peers require epoch fencing and prepared cohort; no unqualified exactly-once network or instant multi-GPU physical atomicity claims. |
| P-043 | **V4 ARCHITECTURAL BASELINE** | Preserve a **future optional** `GradientProgram/TrainStep` and mutable `ParameterVersion` semantic/task seam without importing a compulsory autograd/tape/trainer into inference-first core; inference sessions pin immutable checkpoint/adapter versions. |
| P-044 | **V4 ARCHITECTURAL BASELINE** | Start with **one Rust library and one binary** (or 2–4 crates if GPU backend build/separation proven necessary); no mandatory nine-crate design, dynamic plugin framework or generalized external workflow engine. Require measured benefit to add persistent abstractions. |
| P-045 | **V4 ARCHITECTURAL BASELINE** | Specialized entire model/layer closed regions can replace generic math through one provider contract and compatible state/numerical/resource boundaries; disable specialized path and still run portable model/task with no new server/scheduler/allocator. |
| P-046 | **V4.1 BACKEND CONTRACT** | Generalize native `KernelProvider` into one `RegionExecutionProvider` seam supporting NativeKernel, CapturedCommandGraph, AheadOfTimeDeviceGraph and StaticallyScheduledDataflow. The existing KernelProvider is the native implementation subset, **not** another model runtime or an extra IR. |
| P-047 | **V4.1 BACKEND CONTRACT** | Let the provider declare **minimum efficient closed-region granularity**: operator, fused region, whole stage/model or partitioned fabric program. Device compiler owns micro-scheduling while Rust TaskCoordinator retains session task/admission/output/StateRoot, with no host instruction-by-instruction emulation of TPU/wafer/static fabric. |
| P-048 | **V4.1 BACKEND CONTRACT** | StateBridge qualification explicitly requires privately prepared state, checked snapshot/restore, bounded replay or stateless region. An opaque in-place stateful device program without these is **not** an interchangeable general spec/rollback backend; no device-side PublishRoot authority. |
| P-049 | **V4.1 BACKEND CONTRACT** | ProgramLifecycle records compile/build, upload/load, warmup, resident code/scratch, source/driver/firmware, statically constrained shape, cache invalidation, code/weight stream and program teardown; graph-only devices need shape-band fallback and no per-token recompile. |
| P-050 | **V4.1 BACKEND CONTRACT** | Model hardware **compute domains, memory tiers, links, compiler domains and micro-schedule ownership** independently: coherent LPDDR CPU+GPU, discrete HBM/GDDR GPU, multi-GPU rack peer fabric, graph-compiled TPU/NPU, on-chip SRAM/static wafer fabric and authenticated remote transport. Aggregate chip memory never implies common pointer addressability. |
| P-051 | **V4.1 BACKEND CONTRACT** | Separate FP4 variants with 16-value FP8 block scale and two-level tensor scaling vs 32-value shared power-of-two microscale, native FP8/MX and specialized low-bit layouts; source stored quant vs physical repack vs lossy compute/changed model quality remain distinct with full-task tests. |
| P-052 | **V4.1 BACKEND CONTRACT** | A managed remote inference API without exact compiler lowering and StateBridge is **TaskServiceAdapter interoperability**, not a native RegionExecutionProvider; explicit auth/privacy/egress/session limitation and no claim of native device execution. |
| P-053 | **V4.1 BACKEND CONTRACT** | Report qualification ladder per model/task/shape/driver: architecture compatible → SDK feasible → compiles → functional → state-safe → performance-qualified → production-qualified. Missing SDK/physical hardware/source-policy yields BLOCKED/INCOMPATIBLE, not a phantom supported TPU/wafer product. |
| P-054 | **V4.1 RESEARCH** | Distinguish Cerebras wafer-scale accelerator from an independently published Cerberus parallel-decoding algorithm; the latter is an optional DraftSource/TargetVerifier and full-state rollback candidate, not a new hardware backend. User referent remains unconfirmed, both are researched. |
| P-055 | **V4.1 MINIMALITY** | Add no permanent new semantic IR, task runtime or model-specific server. Use one RegionExecutionProvider and one ExecutableBundle physical program enum; optional closed region graph compiler and statically scheduled fabric tests cannot block real CPU/GPU vertical slice. |
| P-056 | **V4.2 DISTRIBUTED BASELINE** | One **PlacementPlan** independently binds ComputePlacement, ParameterPlacement, mutable StatePlacement, immutable CachePlacement, and derived TransferPlan across topology, TaskDriver phases and committed roots; placement is a *view inside ExecutableBundle*, not another enduring IR or model engine. |
| P-057 | **V4.2 DISTRIBUTED BASELINE** | Support colocated baseline, contiguous/noncontiguous layer groups with per-layer cache colocated, remote layer compute with state elsewhere via hydrate/remote-attention/direct-read variants, tensor/head/expert/sharded-sequence, P/D, FFN/attention disaggregation, remote prefix and multi-task stage placements; every mode per-op/source/math legal and costed. |
| P-058 | **V4.2 DISTRIBUTED BASELINE** | Node/LinkTopology capability includes compute/memory/storage/driver, network **effective directional bandwidth, RTT/P95 jitter, direct vs relay, MTU/loss, auth/crypto/egress**, plus physical budget and failure domain; runtime placement epoch frozen for active compiled region/collective. |
| P-059 | **V4.2 DISTRIBUTED BASELINE** | **TransportProvider** supports local IPC/peer/DMA, physically qualified RDMA, authenticated TCP/QUIC or an existing private encrypted tunnel; control and bulk data planes may choose separate paths. VPN/tunnel is not RDMA, a relay is not direct, and private network routing is not peer authorization. |
| P-060 | **V4.2 DISTRIBUTED BASELINE** | Minimal opt-in authenticated **PeerAgent** reuses the same binary/RegionExecutionProvider/DeviceBackend, advertising allowed model/work/resources and executing closed region/state transfers. No second server, arbitrary shell, network public exposure or automatic export of private prompts, activations, KV or model weights. |
| P-061 | **V4.2 DISTRIBUTED BASELINE** | Each mutable remote per-session/layer state shard has owner/lease/generation/schema/placement epoch references in the **single TaskCoordinator-owned StateRoot**; peers only prepare/ready/fail and are fenced on lost cohort/epoch. Immutable prefix cache locations may be eventually discoverable hints and never substitute for live state authority. |
| P-062 | **V4.2 DISTRIBUTED BASELINE** | Replan/migrate live computation/state only at legal committed TaskFrontier or op-defined checkpoint, with destination reservation, compatible verified snapshot transfer, epoch fence/lease move and old-lease retirement; failure preserves old root or replays safe checkpoint. No automatic transparent failover or exactly-once WAN output without explicit durable protocol. |
| P-063 | **V4.2 DISTRIBUTED BASELINE** | **Remote attention next to its KV/state** and independent FFN/experts are distinct from remote raw KV byte reads per token; choose based on measured network critical path and StateBridge. Slow/relayed tunnels normally veto chatty A/FFN or full remote KV reads, favor contiguous layer or one-time cache hydration. |
| P-064 | **V4.2 DISTRIBUTED BASELINE** | Optional **token-range attention shard** computes local softmax summaries (max, normalization, weighted values) on state-owning nodes and composes stable result; requires operator-specific PartitionAlgebra proving positions/masks/global routing/numerics. Ordinary recurrence/sparse global indexer not presumed partitionable. |
| P-065 | **V4.2 DISTRIBUTED BASELINE** | Layer pipeline colocates its layer-local mutable caches by default, reducing network to activation handoffs. Decode C1 serial hops can be slower; multi-chunk prefill/throughput and model capacity may benefit. Distinguish capacity benefit from single-request t/s gain. |
| P-066 | **V4.2 DISTRIBUTED BASELINE** | Separate control/metadata, bulk data and telemetry; scoped node identity, per-tenant/model/weight authorization, checked stream/transfer length, registered memory lease and content integrity, transport priority/backpressure, no arbitrary remote executable shell or untrusted device program. |
| P-067 | **V4.2 DISTRIBUTED BASELINE** | Dynamic cluster membership/topology and state-owner **lease epochs are not the same**; no split brain and old coordinator writes after failed election; start fail-stop/manual peer recovery, defer durable quorum/consensus/autoscaler until product requirement. |
| P-068 | **V4.2 DISTRIBUTED BASELINE** | Joint placement cost based on real compute/shape/phase, model+state bytes, activation/collective/query payload, link P95/copy/crypto/transfer overlap, cold program/cache, failover, security and objective. Do **not** activate a remote machine merely because it has spare memory. |
| P-069 | **V4.2 MINIMALITY** | Standalone has **zero peer/cluster overhead** and no required network permission. Gradual proof from synthetic remote layer, 2 private peers with reliable transport, remote full-state cache, A/FFN/token shards on fast links; cluster DB, DHT and public super-node services not mandatory. |
| P-070 | **V4.2 VALIDATION** | Add optional S0-DIST-A…L and H56…H70 distributed fault/cost/network/topology experiments, preserving P00–P23; source implementation, model/quant, physical network direct/relay, causal state, quality, P95 and privacy required before functional/performance claims. |
| P-071 | **V4.3 ROUTER BASELINE** | **One built-in ModelRouter** sharing ModelCatalog/ArtifactStore, PhysicalResourceArbiter, TaskCoordinator, RegionExecutionProvider/DeviceWorkers with CLI/SDK/HTTP; no mandatory gateway proxy, second inference scheduler or independent model server for each checkpoint. Exact model choice happens on request admission, not on every token. |
| P-072 | **V4.3 MODEL IDENTITY** | Separate `RegisteredModel` (immutable artifact), versioned `ModelAlias/ModelProfile`, `ResidentModel` deployment/physical replica (may span nodes), and `TaskSession` full mutable StateRoot. No conflation of many catalog models, simultaneously resident physical models, same-model replicas or sharded one model. |
| P-073 | **V4.3 EXACT ROUTING** | Default model alias resolves an exact checkpoint/stored quant/tokenizer/adapter/profile revision and compatible TaskDriver/numeric contract. **No silent substitution of another family, quant or model** when overloaded; explicit opt-in policy aliases may select different models and must reveal actual source/model and fidelity. |
| P-074 | **V4.3 REPLICA ROUTING** | Same-model replica load balancing uses verified state/quant compatibility, sticky session/prefix locality, physical headroom, queue/P95 and tenant/peer data permissions. Stateful migration requires safe StateRoot snapshot/PlacementEpoch transfer at committed frontier, not random per-request model switch. |
| P-075 | **V4.3 RESIDENCY** | On-demand model load/verify/compile/warm and safe idle unload, keepalive/pin/preload/scale-to-zero as optional policies. Deduplicate concurrent loads and bound queue. Unlimited registered models are allowed; simultaneous resident models/replicas/contexts bounded by physical RAM/VRAM/OS reserve and distributed capacity, no automatic quant degradation. |
| P-076 | **V4.3 SHARED BUDGET** | Model-level coarse admission and resident eviction reuse PhysicalResourceArbiter/local pools; within-model native continuous batching stays in existing TaskCoordinator and DeviceWorkers. Router must not become per-token bottleneck or starve loaded interactive models during cold loading. |
| P-077 | **V4.3 PER-MODEL PROFILES** | Versioned typed configuration at global, model, alias, deployment, session and request scope, with deterministic effective-source precedence; immutable source/model semantics, hard security and resource limits always constrain overrides. Hot alias revision affects only future requests; active TaskSession remains pinned. |
| P-078 | **V4.3 SIMPLE UX** | Proposed one executable `zerogravity run <model>` / `zerogravity serve` with zero mandatory daemon/config/cloud account: local in-process run, loopback serving, safe auto-device/model plan, concise progress; `models list/ps/pull/inspect` and `config show --effective` provide progressive disclosure. Exact binary name/port subject to open policy. |
| P-079 | **V4.3 SOURCE POLICY** | Unknown/unavailable named source must return diagnostics/offer explicit import/pull, **never silently download a guessed model**. Explicit source/license/size consent and content-addressed atomic verification, no executable model repository code, no public remote peer use without authorization. |
| P-080 | **V4.3 OPENAI API** | One loopback-native API adapter supports a pinned, tested subset of `GET /v1/models`, `POST /v1/chat/completions`, embeddings once encoder qualified, optional completions and versioned Responses. Streaming/SSE/event/tool-call data, usage, stop and error semantics validated with actual SDKs; no blanket OpenAI feature parity promise. |
| P-081 | **V4.3 RESPONSES/PRIVACY** | Stateless Responses first; stateful `previous_response_id`, local `store`, tool execution or cloud built-ins only when separately implemented/authorized. Tool-call *generation* is output data; arbitrary tool execution is outside inference core. Conversation storage privacy-first and opt-in. |
| P-082 | **V4.3 API SECURITY** | Loopback-only default and native admin/control API segregated from public inference API. External bind requires explicit auth/TLS and tenant/model visibility, remote source/peer policy; no API field can bypass hard physical/numeric/tenant limits or fetch arbitrary media URL/local files. |
| P-083 | **V4.3 MULTI-MODEL COMPOSITION** | A TaskProgram may reference **multiple distinct model roles** (target/draft, encoder/projector, embedding/rerank, TTS/vocoder) with separate artifact identities but one coordinated resource/task budget; these aren't replicas or alternate target routing. |
| P-084 | **V4.3 CONFIG CONTROL** | Expose every valid override through discoverable typed model/global/request/profile controls and `inspect --explain-plan/config show --effective` with value/source/guard/fallback. Config revisions/driver/source changes trigger compatible drain/replan rather than unsafe mutation of live KV or model weights. |
| P-085 | **V4.3 VALIDATION** | Add optional S0-SERVE-A…J and H71…H86 tests for one-command local UX, multi-model load/eviction, equivalent-replica routing, exact alias/session consistency, memory pressure, config precedence, OpenAI Python/JS SDK and SSE, auth/privacy and distributed resource placement. None run. |
| P-086 | **V4.3 MINIMALITY** | `run`, `serve` and Rust SDK use **same library engine**. Router owns model request resolution/admission and does not re-tokenize, JIT, marshal or run a second token scheduler per output step. A separate process may later be a security fault-domain option, not the mandatory architecture. |
| P-087 | **PUBLICATION/DOC AUTHORITY** | Provide a public `docs/WHITEPAPER.md` (problem, proposed original architecture, related work, research limitations, falsifiable hypotheses) separate from an integrated normative `docs/ENGINEERING_SPECIFICATION.md` (ES-00…ES-15 exact owners, protocols, tests). The external research/engine names remain properly attributed only as prior art. |
| P-088 | **PUBLICATION/DOC AUTHORITY** | `docs/README.md` is the organized audience-specific document index; `.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md` is the concise agent entrypoint; latest user decisions and live verified code/tests supersede historical addenda, whose full records must be retained. Do not silently rewrite older source history to remove evidence. |
| P-089 | **AGENT DELIVERY** | Maintain `.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md` with bounded P00–P23 tickets, exact Phase0/negative-test and hardware/source/privacy ownership, handoff template, and `REQUIREMENTS_AND_TEST_TRACEABILITY.md` mapping user U requirements to ES/INV/phase/proof without invented test successes. |
| P-090 | **PUBLICATION SAFETY** | Whitepaper is a **proposed design**, not a benchmark-results paper or proof of universal hardware/model support. Provide numbered primary source citations and publication checklist, with unresolved byline, repository/document license, DOI and independent peer review required before final public distribution. No PDF/export or results artifacts promised without separate request. |
| P-091 | **FINAL REVIEW PROCESS** | Run structural Markdown/link/anchor, core source-name neutrality, stable U/P/O IDs, P00–P23 and H01–H86 plus Stage0 coverage, and exact SHA-256 manifest **regenerated last**. Any failure must be fixed rather than marked passing; no source/model/device/remote operations occur as part of design-only review. |
| P-092 | **PACKAGING BOOTSTRAP** | Use **one dependency-free Rust 2024 binary** pinned to available Rust 1.95.0, with `--version`, `--help`, and honest nonzero refusal for reserved `run/serve/models/inspect/config` until implementation. Never supply a fake HTTP service or model output merely to make packaging pass. |
| P-093 | **PINNED CRANE BUILD GRAPH** | Reuse the trusted *pattern* of pinned nixpkgs/crane/rust-overlay flake, separate dependency/build/test/clippy/fmt outputs, NixOS module and six Linux MUSL/Windows GNU+LLVM-MinGW/macOS Apple SDK cross target definitions. An unexecuted Nix cross derivation is not a verified platform build. |
| P-094 | **RELEASE AND INSTALLER BOUNDARY** | GitHub Actions checks/builds six assets, validates SHA-256 and exact Cargo version, creates a release only on matching v* tag; installers select supported OS/arch, verify sidecar/version and atomically replace the user binary. No default GitHub organization is invented or network/download performed by bootstrap. SHA sidecar is not a signature. |
| P-095 | **NIXOS MODULE HONESTY** | Expose `services.zerogravity.enable` disabled by default, safe journal/DynamicUser and future service options; since `serve` is not implemented, an explicitly enabled unit exits clearly and does not restart-loop, without claiming a functional daemon. |
| P-096 | **BOOTSTRAP SCOPE AND STATUS** | Record P00 tooling/binary subset as PARTIAL/IN_PROGRESS, no completed reference model or inference engine. Preserve source/design docs and agent handoff, do not change read-only dotlink/research, leave Git init/public release/license decisions separate. |

### Open decisions (do not fill silently)

| ID | Status | Question / recommended validation |
|---|---|---|
| O-001 | OPEN | What license for zerogravity? MIT/Apache-2.0 dual licensing is a *candidate*, not selected. Verify copied code license, attribution and weight redistribution separately. |
| O-002 | OPEN | Which accelerators must be production-grade at v1? Candidate first targets: Linux x86-64 CPU and AMD Strix Halo, then CUDA, Apple Metal, Vulkan; final order is not approved. |
| O-003 | OPEN | Exact interpretation of entirely Rust: are system CUDA/ROCm/Metal APIs, HIP/BLAS libraries and compiler-emitted non-Rust machine artifacts acceptable? Proposal: Rust-maintained source with driver/runtime FFI allowed; prove performance before forbidding vendor libraries. |
| O-004 | OPEN | Select primary GPU kernel toolchain after comparative proof (CubeCL, Rust GPU/AMDGPU, compiler extensions, vendor GPU APIs). No framework chosen definitively. |
| O-005 | OPEN | First complete model acceptance target: basic Llama-style dense vs Qwen3.5 hybrid vs Qwen Flash-Next vs GLM; phase gates recommend dense then a hybrid and giant MoE. |
| O-006 | OPEN | Which external format/quant combinations are promised at initial release? Capability matrix must be evidence-based, not optimistic. |
| O-007 | OPEN | How much sampling reproducibility across backends/batch schedules is required? Strong guarantee should at least cover same-plan, same-seed, same-schedule. Cross-plan bit identity often impossible without speed cost. |
| O-008 | OPEN | Choose stable versioned API/ABI and whether to ship C ABI, WASM/browser, FFI from other languages. |
| O-009 | OPEN | Distributed authentication/transport defaults and remote execution scope; do not expose unauthenticated RDMA/TCP by default. |
| O-010 | OPEN | Supported deployment platforms and packaging (NixOS, Linux distros, macOS, Windows, ARM) and GPU availability for CI. |
| O-011 | OPEN | Quality policy for low-bit checkpoint selection and acceptable evaluation delta. Do not make default quality regressions invisible. |
| O-012 | OPEN | Service SLOs and max concurrency/contexts for each hardware tier; derive from measured P50/P95 and memory admission. |
| O-013 | OPEN | Should external model definitions be declarative DSL, Rust code, or both? Prefer explicit versioned semantic registry and safe loading. |
| O-014 | OPEN | Whether to allow third-party Rust-authored kernel/model plugins; assess untrusted code and reproducibility before accepting dynamic execution. |
| O-015 | OPEN | Whether/when to initiate a Git repository, CI, automated releases and model benchmark hardware matrix. |
| O-016 | OPEN | Exact application/project branding, executable name, CLI flags and public version semantics. Working examples are illustrative only. |
| O-017 | OPEN | Formal Rust GPU source policy: are driver/vendor math FFI and separately licensed non-Rust kernel source allowed? User asked Rust-authored owned engine/kernels; research vendor-kernel escape is **not** automatically approved. |
| O-018 | OPEN | Final target kernel toolchain per device after physical benchmark: CubeCL gfx1151, target Rust AMDGPU lowering; independent Rust CUDA options are new as of Sep 2026 and require separate maturity/security/ISA assessment. |
| O-019 | OPEN | Should exact binary name zg, dual MIT/Apache-2.0 license, Rust MSRV and package name from old research be adopted? They remain council recommendations, not explicit user decisions. |
| O-020 | OPEN | Versioned acceleration-pack public filename and cache format (avoid overloading APK), plus correct disk-KV header/schema replacing the historical invalid 48/54-byte spec. |
| O-021 | OPEN | Thresholds for eager/capture kernel tactics, strict replay vs cross-plan numeric variance, online resource-cost policy hysteresis, cache tenant privacy and model-quality pass bands: establish by tests. |
| O-022 | OPEN | Prioritize first accepted dense, hybrid, MoE, embeddings/LoRA and multimodal release cells; user specified eventual breadth but not v1 feature commitments or CI hardware purchases. |
| O-023 | OPEN | Can a **Rust-authored XDNA2/AIE kernel** and program be built with a supported compiler/toolchain and XRT FFI, without bundling Gufo's Python MLIR generator/C++ AIE kernel as owned source? This is separate from the already open driver-math FFI policy. |
| O-024 | OPEN | How to qualify **same-SoC GPU+NPU concurrency safety** across firmware/AGESA, power states, GPU fabric clock and ROCm/XRT versions without privileged automatic system tuning? Unqualified configurations must remain GPU-only; physical hard-hang experiments require isolated test hardware. |
| O-025 | OPEN | First supported NPU workload and precision policy: fixed 4096×2560×1024 `v8bfp16ebs8` Qwen Flash-Next prefill versus generic GEMM shapes or encoder work; benchmark actual critical-path gain and quality under controlled clock/power. |
| O-026 | OPEN | TaskProgram production scope: text generation, ASR, TTS, image diffusion/edit and video/audio must have individual model/driver/license/quality/release cells. Prior-art examples do not imply zerogravity already supports these. |
| O-027 | OPEN | Which exact MIT-licensed source files, if any, to reuse as code rather than independently implementing the mechanism? Confirm file/embedded code/model weight licenses, required notices, Rust-source policy and user review before each import. Non-MIT source remains design research unless explicitly authorized. |
| O-028 | OPEN | How to scope external source attribution in public product documentation and distribute an independent legal source/SBOM table without introducing runtime dependencies on upstream project APIs? Current v4 keeps provenance in research docs, not normative interfaces. |
| O-029 | OPEN | Does “all AI execution” require **training/fine-tuning** as a first-release capability or only an extensible future GradientProgram/TrainStep/ParameterVersion seam? v4 chooses inference-first with future seam, user has not approved a training release scope. |
| O-030 | OPEN | Which task kinds beyond generation/embeddings are release-qualified first (ASR, TTS, image/video, agent-compute)? Supported TaskDriver *type* does not imply actual model/source/data/quality readiness. |
| O-031 | OPEN | Exact delivery guarantees for disconnected/remote clients: default ordered at-most-once live **attempt**; optional durable cursor/idempotent replay requires separate retention/privacy/ACK policy. |
| O-032 | OPEN | Physical memory/sharding strategy under device allocation pressure: choose per-device pool and coarse global lease/reservation/compaction, tuning based on real hardware and task workload rather than freezing one allocator. |
| O-033 | OPEN | When to introduce public plugin/IR ABI or external stable model import schemas? v4 keeps built-in Rust providers and minimal versioned SIR/StateRoot until real extension/use and security needs justify them. |
| O-034 | OPEN | How much fully GPU-native sampled control/grammar/device-loop execution is feasible on each target without sacrificing exactness, cancellation and request output streaming? Determine per model/shape/driver from physical proof, not universal promise. |
| O-035 | OPEN | Which external graph/device compiler/runtime backends and foreign device programs can be used while honoring Rust-owned semantics/kernel-source user policy? Graph lowerer through a vendor compiler is **not** a Rust-authored native TPU/wafer compiler; must decide SDK/license boundaries before implementation. |
| O-036 | OPEN | Which TPU and other graph accelerators offer actual documented source-op/state/shape/graph program APIs sufficient for Rust host lowering, snapshot/provisional state and task-loop execution? Architectural support is not real qualified compiler integration. |
| O-037 | OPEN | Can wafer-scale/static dataflow or deterministic LPU/RDU hardware SDKs be accessed and trusted for direct compiled-region execution with state lifecycle, or is only a remote opaque model API available? Treat latter as TaskService boundary, not native region implementation. |
| O-038 | OPEN | Which physical accelerator to qualify after baseline CPU/Strix/CUDA: available GB10 coherent ARM/Blackwell, workstation/discrete Blackwell, B200/B300 rack HBM, TPU7x/6e, cloud AOT, wafer/static fabric, AMD CDNA etc? Do not assume user owns expensive hardware or cloud capacity. |
| O-039 | OPEN | Cloud accelerator remote compilation/model artifact/privacy/egress/tenant authentication and prohibited third-party code import policy; TaskService integrations need separate user authorization and are not a low-level vendor device backend. |
| O-040 | OPEN | Compiler region granularity and device-resident state bridge: which programs may keep static mutable KV/recurrence/optimizer across calls while still supporting prepare/abort, rollback and common logical StateRoot? Must prove per backend, not infer from vendor API name. |
| O-041 | OPEN | Was user’s “cerberus” intended as **Cerebras** wafer-scale hardware, the **Cerberus** adaptive parallel-decoding paper, or both? Preserve both research paths without making an unsupported hardware claim or demanding clarification before design work. |
| O-042 | OPEN | Exact multi-node first release topology: two trusted heterogeneous LAN/tunnel machines with contiguous layer+state colocation, remote prefix, P/D, A/FFN or expert-first? Design supports all; prioritize physical hardware and measurable application workloads, not arbitrary feature count. |
| O-043 | OPEN | Preferred private peer connectivity and transport: direct authenticated TCP/QUIC vs existing WireGuard/Tailscale overlay tunnels, local RDMA/Thunderbolt when physical, NAT relay behavior and trust/double-encryption cost. **No approval to connect machines or install a tunnel is implied**. |
| O-044 | OPEN | Node identity/provisioning/authorization and tenant/model/checkpoint sharing permissions, weight license redistribution and remote prompt/hidden-state privacy policy; default require explicit enrollment. |
| O-045 | OPEN | Topology discovery: manual peer inventory first vs mDNS/optional overlay discovery, dynamic link measurement frequency and peer capability attestation. No automatic use of unknown peers. |
| O-046 | OPEN | Remote state shard transfer schema for KV+GDN/KDA+sparse indexer/ngram/grammar/RNG/modalities and versioned state snapshots: choose portable wire codec, delta/COW and layout conversion only after actual models/quant reference. |
| O-047 | OPEN | Coordinator crash/re-election and durability tier: fail-stop with replay vs durable checkpoints/lease service/quorum, output cursor and exactly-once requirements. v4.2 promises no automatic transparent HA until implemented. |
| O-048 | OPEN | Placement solver sophistication: static simple heuristic vs bounded search and online hysteresis; performance varies dramatically by direct/relay RTT, context, shape, per-layer operator type, task/power and physical cache residence. |
| O-049 | OPEN | Network transfer implementation for GPU direct, RDMA/UCX-style native libraries and fast collectives under Rust-owned code/vendor FFI policy, with fallback buffered TCP/QUIC. A VPN cannot guarantee GPU RDMA. |
| O-050 | OPEN | Which attention operators support exact token-range PartitionAlgebra and stable softmax summary combine? Need source-equation and numerical proof; compressed/selected sparse and GDN/KDA recurrence are NOT automatically sequence partitionable. |
| O-051 | OPEN | When to disaggregate mutable attention KV from FFN versus migrate cache once, keep whole layer+cache colocated or read remote state bytes: need measured per-link cost and safe migration/readiness/replay. |
| O-052 | OPEN | Model/task weights/adapter/quant cache distribution and encrypted storage, remote source permission and content-addressed trusted reference, cache directory presence hints versus authoritative remote mutable state generation. |
| O-053 | OPEN | Private WAN-tunneled machine capacity mode vs home LAN vs rack/cluster behavior and expected SLAs—must qualify separately, rather than one universal default topology. |
| O-054 | OPEN | Final binary name (`zerogravity` vs shorthand), exact command aliases and localhost default port; examples are proposed, not a stable CLI approval. |
| O-055 | OPEN | Per-model default context, keepalive/idle unload, max concurrent resident models and parallel sessions: adaptive resource safety vs predictable user controls, not fixed arbitrary source-project limits. |
| O-056 | OPEN | Config on-disk format/location/schema migration, global/model/alias/request/CLI/env precedence and secrets handling; v4.3 proposes versioned typed TOML but final user setting policy remains open. |
| O-057 | OPEN | Model catalog trusted source registry list, automatic local alias discovery, explicit `pull` network consent, license/size/checkpoint pin and offline semantics; avoid silent remote download. |
| O-058 | OPEN | Which OpenAI API subset to release first beyond chat/models, including completions, embeddings, stateless Responses, stateful previous_response_id and streaming schema/SDK version; no blanket compatibility promise. |
| O-059 | OPEN | `run` process lifecycle: always in-process, optionally connect trusted same-user resident daemon/socket, explicit auto-service mode, NixOS/system service setup, startup and history retention; no silent system service installation. |
| O-060 | OPEN | Exact replica load balancer cost and stickiness: session StateRoot affinity, cache hints, predicted P95/TTFT, per-tenant fairness, model load churn; when to prewarm/create replicas vs move a session. |
| O-061 | OPEN | Policy-defined different-target `auto` alias: whether/how to expose optional automatic model selection by quality/privacy/cost while disallowing unrequested model/quant substitutions for exact requests. |
| O-062 | OPEN | Model profile editable parameters and default modes, including runtime/reload safety boundaries; which sampler/grammar/context overrides should be possible per request versus require new TaskSession. |
| O-063 | OPEN | Native administrative model-management API (local socket vs authenticated HTTP, administrative scoping) and public exposed `/v1/models` catalog/alias privacy; model source paths/hardware/peer topology must not leak. |
| O-064 | OPEN | Many-model physical residency and eviction policy under unified RAM/GTT, VRAM, per-model context state, adapter/shared weights and remote replication; exact pin/keepalive/scale-to-zero guarantees and cold-start SLA. |
| O-065 | OPEN | Advanced cross-model TaskProgram roles and router SLOs: target/draft/encoder/reranker/vocoder on independently placed qualified devices with one tenant admission budget; actual model combos need explicit task quality. |
| O-066 | OPEN | OpenAI Responses-compatible state storage and session retention, privacy and stream cursor contract vs stateless-only initial endpoint; server-side tool execution is outside zerogravity inference core. |
| O-067 | OPEN | Public author/byline, affiliations and citation contact for whitepaper publication; do not attribute the design to unapproved individual or organization names. |
| O-068 | OPEN | Public repository code license and whitepaper/documentation copyright/distribution license; current publication draft must not assume MIT/Apache or CC-BY for original zerogravity source. |
| O-069 | OPEN | Publication channel/URL, DOI or stable version identifier, visual template/PDF/HTML export and accessibility standard; markdown manuscript exists but no publication artifact was requested/produced yet. |
| O-070 | OPEN | Independent technical peer review, legal/trademark review and final owner signoff before public whitepaper release; design arguments are not verified runtime implementation benchmarks. |
| O-071 | OPEN | Future results edition requires exact model/quant/source/driver/hardware/network/task baseline, quality/sampling/p95/energy, raw reproducibility and negative results; no current benchmark evidence exists. |
| O-072 | OPEN | Final GitHub repository OWNER/zerogravity, initial source license and signed-release/provenance policy before public release. Installers deliberately require explicit repository or release base URL rather than guessing a host. |
| O-073 | OPEN | Final tested support matrix for all six cross-built targets; Apple SDK license/source access, macOS signing/notarization, Windows ARM64 toolchain and real per-OS binary smoke tests. Rust native Linux build alone is not proof. |
| O-074 | OPEN | When the real serving runtime exists, update `services.zerogravity` execution arguments, restart behavior, listen/credentials and mutable model directories; currently the module's `serve` command intentionally fails and the unit is disabled by default. |

### Assumptions / boundaries

- A-001 (SUPERSEDED): initial relative-path interpretation placed `./zerogravity` under dotlink l5's launch directory `/home/pvl/spaces/abird`. User decision U-013 explicitly overrides it: canonical root is `/home/pvl/spaces/abird/src/zerogravity`.
- A-002: Research is a 2026-10-10 **snapshot**; named model geometries, release support and benchmark numbers require source pinning and re-verification.
- A-003: Reference benchmark reports are *external author measurements*. zerogravity has not executed tests or proved superiority.
- A-004: Full general fallback requires a known semantic implementation for every operator, sufficient resources and an available execution backend; unknown Python custom model code is not magically executable.
- A-005: The user has not requested using dotlink to modify existing projects or publish a GitHub repo.
- A-006: No specific secret, account, network endpoint or user credential is part of this project design.

### v3 architectural supersession and provenance rule — 2026-10-12

**U-017 remains the user-mandated project-neutral/source-license policy; U-018 is the latest comprehensive review/minimality instruction.** Source-specific implementation patterns mentioned in older U-00x user motivations, P-001…P-030 assistant proposals and old source-council D-xx records are **retained as historic evidence**, not live project-named architectural ownership. They are governed by this v4 distinction:

1. **Original normative design**: `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`, `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`, `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`, v4 `ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`, current roadmap, experiment and release plans. These use typed model/task/region/device/resource/service contracts; P-002's old three-IR abstraction was refined by P-018/P-035 and **minimized by P-036**. Prepared-root publication and hierarchical allocator supersede ambiguous v3 StateDomain/ResourceArbiter sketches.
2. **Named external sources**: `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` verifies root/file license evidence and URLs; Retained source investigations preserve attribution, source findings, decisions and reported benchmarks as non-normative research. Retired snapshot bodies remain in Git history. Exact model checkpoint and hardware identifiers can appear in validation matrices; they are not architecture type names.
3. **MIT exception**: named references and source-reuse candidates from verified MIT-licensed projects are allowed but are not required dependencies. Even an MIT repository can contain differently licensed subfiles/device kernels/model weights. Reuse remains subject to specific file license/NOTICE, source-language and user review. Non-MIT works provide attributed research and engineering ideas, not copied or named project-specific runtime interfaces by default.
4. **No composability exception for performance**: even a much faster optimized model device graph must implement a closed semantic region or TaskProgram stage through the same executor/state/resource contracts, never a cloned model-specific server.
5. **Future decisions**: source attribution/compatibility is a research record; user product/architecture acceptance needs explicit U-/D-style decision and independent model/hardware proof. No prior project's "resolved" option can overrule U-017.

**Policy goal:** external projects could disappear or change license tomorrow without invalidating any core zerogravity architectural interface. v4 adds the further requirement that an optional model/kernel/task specialization can be removed without cloning or corrupting the shared runtime.

### Conflict-resolution protocol

Write future decisions as `D-YYYYMMDD-NN` with source (user instruction/review), options, trade-offs, tests, conclusion, superseded IDs, and affected files. Never upgrade a PROPOSAL to USER on an assistant's own authority. Whenever a model/hardware benchmark contradicts a claim, record the measured evidence and change the claim.

| U-050 | USER | “Let's go, next phase”: resume implementation after the pipeline design/reference cleanup, preserving the single Rust engine and source/StateRoot/physical ownership boundaries. Next bounded H7 moves original GDN polynomial SiLU and output gating; source host exp/ln coefficients are the next separate numerical gate. Full build/lint/tests, trained source checks, AMD/NVIDIA physical qualification, rollback/quarantine and archived handoff/SHA updates precede an atomic local commit. README stays unchanged; no downloads, device changes or publication. |

| U-051 | USER | Requested the next implementation phase after H7. Continue H8 original-source GDN nonlinear coefficients through the existing Rust engine/provider and immutable source/session/resource ownership, with exact FP32 arithmetic and real trained AMD/NVIDIA physical qualification. Preserve the user README, record failures and acceptance, regenerate handoff/SHA, and commit verified work locally; acquisitions, system changes and publication remain unauthorized. |


| U-052 | USER | Requested the next implementation phase after H8. Continue bounded H9 source-exact attention RoPE and private current-token KV proposals through the existing Rust engine/provider, immutable source and independent session/resource ownership. Qualify actual trained AMD/NVIDIA operations, exact rounding and nonzero positions, rollback, admission and post-launch quarantine; retain failed experiments and independent source evidence, regenerate handoff/SHA and commit verified work locally. User README, download/device/publication restrictions remain in force. |


| U-053 | USER | Requested the next implementation phase after H9. Continue source-exact GPU attention through the existing Rust engine/provider and immutable source, independent session StateRoot and physical resource ownership. H10 qualifies bounded QK scoring before separately verified softmax/value stages, preserving exact source rounding and nonzero private prefixes; verify both trained checkpoints/cards, rollback/admission/quarantine and independent source evidence, then update archive/SHA and commit locally. README and acquisition/device/publication restrictions remain in force. |


| U-054 | USER | Requested the next implementation phase after H10. Continue original-source GPU attention through the existing Rust engine/provider, immutable source identity, private StateRoots and physical resource ownership. H11 qualifies padded attention softmax with exact exponential/reduction and two-round normalization, both trained checkpoints/cards, independent native captures, rollback/admission/quarantine, then updates the handoff/SHA and commits verified work locally. README and acquisition/device/publication restrictions remain in force. |


| U-055 | USER | Explicitly requested “let's go, next, let's finish up full gpu inference”, extending the preceding individually requested operator phases. Continue verified implementation through remaining attention/FFN/residual/embedding math, GPU dynamic activation quantization and a complete resident pipeline using the same Rust mathematical schedule and independent transactional StateRoots. Preserve original source bits/identity, fair physical workers/arbiter, actual completion, memory admission and quarantine; no selected-error CPU substitution or unsupported whole-GPU/performance claims. H12 adds source-padded value-cache accumulation and corrects a native-proven signed-zero corner in the former valid-only CPU schedule. Root README remains byte-identical; no models/SDKs, device privileges, push/tag/release authorized. [Full completion tracking](../plans/accepted-baseline/TODO.md), [H12 evidence](../plans/accepted-baseline/TODO.md). |


### Owner decisions after U-055 — consolidated 2026-10-17

These rows preserve the subsequent user instructions from the implementation
conversation. Earlier rows describe the state and authority at their own point
in time. Current implementation is tracked in the [plan index](../plans/TODO.md),
[accepted baseline](../plans/accepted-baseline/TODO.md) and individual plan TODOs.
U-067/U-068 supersede older archive/SHA/generated-handoff process duties;
U-069/U-070 leave runtime implementation paused during this documentation task.
Historical checkpoint labels may predate the rewritten Git graph: a recorded
approval is distinct from a currently resolvable Git object.

| ID | Type | Decision and current meaning |
| --- | --- | --- |
| U-056 | USER | Repeatedly continue verified GPU phases through full GPU inference, then enable native NVIDIA CUDA and AMD ROCm/HIP. Preserve one mathematical engine, original source identity, independent transactional StateRoots, physical admission, fair workers, actual completion and quarantine. A passed operator is not a full-model claim. The accepted original-model resident512-token scope and native CPP profiles are recorded in accepted-baseline; independent8K drift, larger contexts and held hardware remain open. |
| U-057 | USER | Include the current machine and SSH targets pvl-a1/pvl-x2 as the first optimization fleet: five physical GPU devices and the best paths for all three CPUs. Maximize demonstrated capability on these targets using measured, source-correct optimizations. Target discovery/SSH qualification does not qualify distributed inference or authorize model/SDK downloads, persistent configuration, driver/device privilege changes or held-device recovery. |
| U-058 | USER | Use subagents for implementation and independent work, parallelizing wherever efficient; the main agent primarily coordinates, gates, reconciles, decides and merges. Use disposable side worktrees for overlapping source work, preserve ownership, then merge only verified pieces and retire superseded work. One MAIN Cargo/target owner avoids build contention and stale cross-worktree results. |
| U-059 | USER | After the native-enablement checkpoint, do the workspace layout/structure redesign and then a thoughtful left-right concurrency pass before returning to GPU optimization. Add left-right where contention and immutable publication ownership justify it. Its later cold model-catalog proposal was challenged by the user because that area was not contended; the accepted result is no speculative migration. Future candidates need representative contention evidence and deterministic dual-replay/reader-lifetime/resource tests. |
| U-060 | USER | Remove the independent probes/ directory by relocating reusable implementation to crates/, qualification tools to tools/ and appropriate tests to their owners. Consolidate the workspace/lock/version and preserve the sole product CLI. Clean only merged/superseded side worktrees. The old29.7GB build cache and historical evidence binaries need not be preserved; prior resnapshot/archive directions were subsequently superseded by U-066/U-067. Root README remains unchanged. |
| U-061 | USER | Immediately after the concurrency pass, implement explicit model acquisition from GitHub and Hugging Face into the XDG cache, then cover the finite applicable pull capabilities of pinned llama.cpp/vLLM. Preserve source authority, complete required closure, offline-before-credentials, ownership/admission/cancellation and the single native transport/cache/loader boundaries. Additional provider/profile support requires its own tests; SDK/plugin delegation does not imply native equivalence. |
| U-062 | USER | In parallel with independent checks, begin Qwen3.8 27B, Gemma4 E2B and Gemma4 26B model support. First bind source/config/index/tokenizer identities and basic source-correct CPU execution, then GPU. Existing delivered contracts are not trained-model inference; no new model or SDK downloads are authorized by the model request. |
| U-063 | USER | For Qwen, finish basics first, then match every applicable capability and optimization of the pinned llama.cpp/vLLM union, including public and internal surfaces. Afterwards add appropriate MTP, speculative decoding, n-gram/suffix, DSpark2/DFlash2 and related strategies under target-authoritative state/distribution correctness. Exact primary identity/version for DSpark2 remains unresolved; provisional frontier counts are not an exhaustive parity claim. |
| U-064 | USER | Use concurrency primitives thoughtfully, not to fill cheap places arbitrarily. Specifically the model catalog is not demonstrated contention. Preserve a simple existing lock where it fits; performance claims require measured benefit, correct publication/lifetime ownership and minimal maintenance cost. |
| U-065 | USER | Requested commit/push at earlier checkpoints. Those particular actions do not authorize later publication. The archive split proposal initially selected verified ordinary-Git parts rather than LFS, but was superseded by the owner-approved checkpoint/evidence deletion decision. Current AGENTS requires explicit authorization for each push, force-push, tag, release, deployment or destructive Git action; local verified atomic signed commits remain the normal implementation workflow. |
| U-066 | USER | Completed phases may use a singular owner-approved checkpoint as accepted truth; historical evidence archives and binaries are unnecessary and can be removed after preserving completed/tested lists and remaining work. The repository was subsequently rewritten. Preserve actual source/checkpoint/content identities and implementation tests, while distinguishing old approval labels from reachable history. The earlier preference to retain a future evidence model was superseded by the later simpler TODO instructions. |
| U-067 | USER | Replace the entire handoff/evidence process with simple TODO.md files: plans/TODO.md indexes plans; plans/<plan>/TODO.md records provenance, committed implementation/tests and current unfinished work; runs/<plan>-<session-or-agent-id>/TODO.md holds each agent's own work. On completion the coordinator atomically integrates useful results and unfinished items into the main plan; completed plans get one checked index line. Carry unfinished items forward after every commit; archive only long completed Markdown TODO sections. Delete obsolete evidence records and historical archives after useful context is retained. |
| U-068 | USER | Remove SHA inventories and staged tree/blob receipt requirements from evidence logs. Actual implementation tests prove behavior; provenance alone does not prove commands or outputs. A test item must state its intended inputs/behavior and be checked only after execution. Git commit provenance and ordinary resumable TODOs replace mandatory per-file receipts. Actual model/content hashes, integrity checks and numerical acceptance contracts remain required where they are part of implementation. |
| U-069 | USER | After the latest interruption, wrap the current phase, revalidate live source/process/worktree state, then pause. Quant selection finished at signedf197c9c after975focused/fourstrict/full934Rust+18namespace/originalmodel passes; two source-only filters/Xet-index drafts remain unfinished and unmerged. All implementation agents stopped. Only routine rolling TODO updates remained dirty before this handoff documentation request; nothing was pushed. |
| U-070 | USER | Create a comprehensive Markdown handoff and ready-to-copy next-agent starting prompt covering context from the beginning: initial design, specifications, user decisions, implementation steps, real verification scope, all pending work and exact resumption instructions. Record the handoff and its retained context under .agents/docs or .agents/plans, preserve root README, and identify any irrecoverable originals honestly. This is a documentation request; it does not resume the paused implementation or reinstate obsolete evidence/SHA/archive automation. |


<a id="catalog"></a>

## Complete reading catalog and source ownership

All current normative architecture/specification and plan sources are listed here, with retained research/prior-art references for full equations, alternatives, attribution, negative cases and acceptance matrices. New handoff records and portable unfinished work are under `.agents/docs/` or `.agents/plans/`. Existing research paths remain in place to preserve live canonical links, including the immutable root README’s research-pointer target; their relevant context is integrated above. Do not relocate or regenerate them as an onboarding prerequisite.

Read the relevant original in full before implementing that subsystem; a brief catalog entry is navigation, not an additional support claim.


### Public engineering and publication context


- [Development, builds and release definitions](../../docs/DEVELOPMENT.md) — `docs/DEVELOPMENT.md`.

- [Zerogravity — Integrated Engineering Specification](../../docs/ENGINEERING_SPECIFICATION.md) — `docs/ENGINEERING_SPECIFICATION.md`.

- [Whitepaper publication and final architecture review checklist](../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md) — `docs/PUBLICATION_AND_REVIEW_CHECKLIST.md`.

- [Zerogravity documentation](../../docs/README.md) — `docs/README.md`.

- [References and source provenance](../../docs/REFERENCES.md) — `docs/REFERENCES.md`.

- [Zerogravity](../../docs/WHITEPAPER.md) — `docs/WHITEPAPER.md`.


### Normative subsystem architecture


- [zerogravity — Universal region-execution backend contract (v4.1)](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) — `.agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md`.

- [zerogravity — Async Command/Resource IR (CIR), ownership and compiled execution](ASYNC_RESOURCE_COMMAND_IR.md) — `.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md`.

- [zerogravity — v4.1 typed composition and region execution contracts](COMPOSITION_AND_EXTENSION_CONTRACTS.md) — `.agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md`.

- [Joint cost model, online adaptation and compiler autotuning](COST_MODEL_AND_AUTOTUNING.md) — `.agents/docs/COST_MODEL_AND_AUTOTUNING.md`.

- [Zerogravity — EXL3/QTIP and Source-Qualified Derived Quantization Contract](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md) — `.agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md`.

- [Device abstraction, Rust GPU kernels, code generation and backend specialization](DEVICE_KERNEL_AND_CODEGEN.md) — `.agents/docs/DEVICE_KERNEL_AND_CODEGEN.md`.

- [Distributed and heterogeneous execution — v4.2 authoritative index](DISTRIBUTED_PARALLELISM.md) — `.agents/docs/DISTRIBUTED_PARALLELISM.md`.

- [zerogravity — Distributed placement, network and state execution contract (v4.2)](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) — `.agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md`.

- [zerogravity — Native Engine Core, Worker Protocol, KV policies and request lifecycles](ENGINE_CORE_EXECUTION_AND_PROTOCOL.md) — `.agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`.

- [Design walkthroughs — concrete execution and fallback traces](EXAMPLE_EXECUTION_TRACES.md) — `.agents/docs/EXAMPLE_EXECUTION_TRACES.md`.

- [Zerogravity — Extreme Hybrid Model Recipes, Native MTP and DFlash Source Contracts](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) — `.agents/docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md`.

- [Model artifacts, quantization, tokenizer/template parity and safe loading](FORMATS_QUANTIZATION_AND_LOADING.md) — `.agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md`.

- [Zerogravity — gfx1151 ROCm Kernel Specialization, MoE Scheduling and Unified-Memory Safety Contract](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) — `.agents/docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md`.

- [Zerogravity — Heterogeneous prefill and retained replay contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) — `.agents/docs/HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md`.

- [Cross-cutting invariants, state lifetimes and failure semantics](INVARIANTS_AND_FAILURE_SEMANTICS.md) — `.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md`.

- [Zerogravity — Joint Concurrent Serving, Goodput and Combinatory Specialization Contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) — `.agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md`.

- [Global memory, expert residency, storage I/O and resource admission](MEMORY_RESIDENCY_AND_IO.md) — `.agents/docs/MEMORY_RESIDENCY_AND_IO.md`.

- [Minimal execution spine v4 — state, asynchronous workers, resource ownership and hot-path performance](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) — `.agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`.

- [zerogravity — Built-in model router, multi-model residency and serving contract (v4.3)](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) — `.agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md`.

- [Model semantics, model adapters and compatibility qualification](MODEL_SEMANTICS_AND_COMPATIBILITY.md) — `.agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md`.

- [zerogravity — modular AI execution reference architecture v4.1](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md) — `.agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`.

- [zerogravity — NPU/GPU cooperative execution and multimodal TaskProgram design](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) — `.agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md`.

- [Numerical correctness, model-quality evaluation and benchmark methodology](NUMERICS_QUALITY_AND_BENCHMARKS.md) — `.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md`.

- [Operator/region contracts, fusion patterns and model execution catalog](OPERATOR_AND_FUSION_CATALOG.md) — `.agents/docs/OPERATOR_AND_FUSION_CATALOG.md`.

- [Architectural risks, opportunities, trade-offs and falsification criteria](PERFORMANCE_HYPOTHESES_AND_RISKS.md) — `.agents/docs/PERFORMANCE_HYPOTHESES_AND_RISKS.md`.

- [zerogravity — Product, model loading and extension contract across the full pipeline](PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md) — `.agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`.

- [Zerogravity Real-Time Multimodal Serving, Sessions, QoS and API Contract — v5.1](REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md) — `.agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md`.

- [zerogravity — Remote peer transport, state handoff and failure protocol (v4.2)](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md) — `.agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md`.

- [Runtime, APIs, developer experience and observability](RUNTIME_SERVING_AND_UX.md) — `.agents/docs/RUNTIME_SERVING_AND_UX.md`.

- [Rust workspace, interfaces, implementation practices and contributor workflow](RUST_WORKSPACE_AND_ENGINEERING.md) — `.agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md`.

- [Scheduler, continuous batching, prefix reuse and speculative execution](SCHEDULER_BATCHING_AND_SPECULATION.md) — `.agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md`.

- [Security, reliability, resilience and safe defaults](SECURITY_AND_RELIABILITY.md) — `.agents/docs/SECURITY_AND_RELIABILITY.md`.

- [Semantic IR, state-effect system and ahead-of-time/JIT execution planning](SEMANTIC_IR_AND_COMPILER.md) — `.agents/docs/SEMANTIC_IR_AND_COMPILER.md`.

- [Session state, prefix caching, snapshots and transactional semantics](STATE_CACHE_AND_TRANSACTIONS.md) — `.agents/docs/STATE_CACHE_AND_TRANSACTIONS.md`.

- [v4 TaskProgram and compiler contracts — high throughput with a small semantic core](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) — `.agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`.

- [Zerogravity — Unified multimodal execution contracts (v5 design proposal)](UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md) — `.agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md`.

- [Zerogravity — First-Class Virtual Tensor Residency and High-Performance SSD Streaming](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) — `.agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md`.

- [Zerogravity Universal Multimodal Execution Architecture — v5.1 Design Specification](UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md) — `.agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md`.

- [Zerogravity Verified Model Assemblies, Source Codecs and Legal Specialization — v5.1](VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md) — `.agents/docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md`.

- [Vision, product scope, constraints and engineering philosophy](VISION_SCOPE_AND_PRINCIPLES.md) — `.agents/docs/VISION_SCOPE_AND_PRINCIPLES.md`.

- [zerogravity — One-command CLI, progressive configuration and OpenAI-compatible APIs (v4.3)](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md) — `.agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md`.


### Cross-cutting specifications and decision/workflow owners


- [Zerogravity — implementation design reference](../plans/AGENT_IMPLEMENTATION_PLAYBOOK.md) — `.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md`.

- [Current work](../plans/CURRENT_STATUS_AND_TRACKING.md) — `.agents/plans/CURRENT_STATUS_AND_TRACKING.md`.

- [Decision register — confirmed user intent versus recommended design](../plans/DECISION_REGISTER.md) — `.agents/plans/DECISION_REGISTER.md`.

- [zerogravity — Project-neutral experimental program, H01–H43 (v4 reviewed)](../plans/EXPERIMENT_PROGRAM.md) — `.agents/plans/EXPERIMENT_PROGRAM.md`.

- [Large immutable source loading: bounded basic-unit proposal](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md) — `.agents/plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md`.

- [Zerogravity — architecture roadmap P00–P23](../plans/IMPLEMENTATION_ROADMAP.md) — `.agents/plans/IMPLEMENTATION_ROADMAP.md`.

- [zerogravity — Independent Stage-0 feasibility and delivery gates (v4 reviewed)](../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md) — `.agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`.

- [Acquisition next unit — bounded cache first, then HF/GitHub vertical slice](../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md) — `.agents/plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md`.

- [Starting prompt — Zerogravity takeover, 17 October 2026](../plans/NEXT_AGENT_PROMPT_2026-10-17.md) — `.agents/plans/NEXT_AGENT_PROMPT_2026-10-17.md`.

- [Zerogravity — Human-Centric Design, AI-Centric Implementation, and Open Contributions](../plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md) — `.agents/plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md`.

- [zerogravity — Model-neutral release, quality and composability gates (v4 reviewed)](../plans/RELEASE_ACCEPTANCE_GATES.md) — `.agents/plans/RELEASE_ACCEPTANCE_GATES.md`.

- [Zerogravity — requirements and test design](../plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md) — `.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md`.

- [TODO spec — plans, checklists and evidence](../plans/TODO-SPEC.md) — `.agents/plans/TODO-SPEC.md`.

- [Zerogravity plans](../plans/TODO.md) — `.agents/plans/TODO.md`.

- [zerogravity — master architecture v4.3: composable AI execution, model routing and simple user experience](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) — `.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`.


### Live plan TODO owners and completed plain-checklist archives


- [Accepted implementation baseline](../plans/accepted-baseline/TODO.md) — `.agents/plans/accepted-baseline/TODO.md`.

- [Cpu gpu optimization](../plans/cpu-gpu-optimization/TODO.md) — `.agents/plans/cpu-gpu-optimization/TODO.md`.

- [Distributed execution](../plans/distributed-execution/TODO.md) — `.agents/plans/distributed-execution/TODO.md`.

- [Simplify agent handoffs](../plans/handoff-cleanup/TODO.md) — `.agents/plans/handoff-cleanup/TODO.md`.

- [TODO — historical records](../plans/historical-records/TODO.md) — `.agents/plans/historical-records/TODO.md`.

- [Memory and streaming](../plans/memory-and-streaming/TODO.md) — `.agents/plans/memory-and-streaming/TODO.md`.

- [TODO — model acquisition](../plans/model-acquisition/TODO.md) — `.agents/plans/model-acquisition/TODO.md`.

- [Model execution](../plans/model-execution/TODO.md) — `.agents/plans/model-execution/TODO.md`.

- [Multimodal](../plans/multimodal/TODO.md) — `.agents/plans/multimodal/TODO.md`.

- [Numerics and context](../plans/numerics-and-context/TODO.md) — `.agents/plans/numerics-and-context/TODO.md`.

- [Platform and release](../plans/platform-and-release/TODO.md) — `.agents/plans/platform-and-release/TODO.md`.

- [Serving and speculation](../plans/serving-and-speculation/TODO.md) — `.agents/plans/serving-and-speculation/TODO.md`.

- [Model acquisition](../plans/model-acquisition/README.md) — `.agents/plans/model-acquisition/README.md`.

- [TODO — completed model acquisition phases through a774ded](../plans/model-acquisition/archive/through-a774ded/TODO.md) — `.agents/plans/model-acquisition/archive/through-a774ded/TODO.md`.

- [TODO — completed model acquisition through fce1353](../plans/model-acquisition/archive/through-fce1353/TODO.md) — `.agents/plans/model-acquisition/archive/through-fce1353/TODO.md`.


### Retained research and acceptance registers


- [Zerogravity — Extreme Model Specialization Integration Roadmap and Executable Acceptance Design](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) — `.agents/research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md`.

- [Gemma 4 family source contracts — 2026-10-16](../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md) — `.agents/research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md`.

- [GPU tokenization, execution control and text decoding roadmap](../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md) — `.agents/research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md`.

- [Zerogravity — Heterogeneous prefill and retained replay acceptance](../research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md) — `.agents/research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md`.

- [Initial hardware targets and capability optimization](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md) — `.agents/research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md`.

- [Model pull, XDG artifact cache and pinned acquisition parity](../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md) — `.agents/research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md`.

- [Zerogravity — Model-family research atlas and multimodal implementation roadmap](../research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md) — `.agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md`.

- [Zerogravity Multimodal v5.1 — Complete Implementation Roadmap, Dependency Gates and Acceptance Matrix](../research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md) — `.agents/research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md`.

- [Zerogravity v5.1 Multimodal Decisions, User Requirements, Design Proposals and Open Questions](../research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md) — `.agents/research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md`.

- [Proposed native FP32 GEMV FMA qualification](../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md) — `.agents/research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md`.

- [A1 local cache — protocol and API reference](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md) — `.agents/research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md`.

- [Qwen basics, complete reference parity and advanced decoding](../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md) — `.agents/research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md`.

- [Development, builds and releases](../research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md) — `.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md`.

- [Zerogravity Specialization Prior Art — Specialist Source Atlas](../research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md) — `.agents/research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md`.

- [Zerogravity — First-Class SSD Streaming Implementation Roadmap and Acceptance Tests](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) — `.agents/research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md`.

- [AMD Strix Halo dossier — gfx1151 extreme inference targets](../research/STRIX_HALO_OPTIMIZATION_DOSSIER.md) — `.agents/research/STRIX_HALO_OPTIMIZATION_DOSSIER.md`.

- [Zerogravity — vLLM Parity and Combinatory Serving Acceptance Program](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md) — `.agents/research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md`.

- [Research Track B — ASR, Speech Translation and Streaming Speech Understanding](../research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md) — `.agents/research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md`.

- [Research Track H — Cross-Stream Adversarial Reconciliation and Minimal Architecture Delta](../research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md) — `.agents/research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md`.

- [Research Track F — Typed Decision Models, Ternary Formats and Specialist Text Architectures](../research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md) — `.agents/research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md`.

- [Research Track D — Image Generation, Editing, Flow/Diffusion and Autoregressive Image Models](../research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md) — `.agents/research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md`.

- [Research Track G — Real-Time and Multimodal Serving / API Prior Art and Design Reconciliation](../research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md) — `.agents/research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md`.

- [Research Track C — Text-to-Speech, Neural Audio Codecs, Voice and Full-Duplex Conversation](../research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md) — `.agents/research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md`.

- [Research Track E — Video, Temporal Diffusion, Synchronized Audio-Video and Frontier Providers](../research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md) — `.agents/research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md`.


### Prior art, critical review and license/source attribution


- [Accelerator execution prior art — Blackwell, TPU, wafer-scale, static dataflow and specialized runtimes](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md) — `.agents/prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md`.

- [zerogravity — Reconciled architecture and authority matrix](../prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md) — `.agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md`.

- [zerogravity — v4 critical architectural review, challenges and resolution record](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md) — `.agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md`.

- [zerogravity — Comprehensive cross-stack research and design selection](../prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md) — `.agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md`.

- [Distributed execution, state disaggregation and tunnel transport: prior-art research](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md) — `.agents/prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md`.

- [Zerogravity — Final Architecture, Editorial and Agent-Handoff Review](../prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md) — `.agents/prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md`.

- [Gufo XDNA2 cooperative-prefill fork and October 2026 Strix Halo report — primary-source research](../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md) — `.agents/prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md`.

- [Multi-model local runtime, router, CLI and API prior-art research](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md) — `.agents/prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md`.

- [Prior art, provenance and architecture neutrality](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md) — `.agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`.

- [Source atlas — architectural research preserved for implementers](../prior-art/RESEARCH_SOURCE_ATLAS.md) — `.agents/prior-art/RESEARCH_SOURCE_ATLAS.md`.

- [SSD Streaming Research Dossier — DS4/DwarfStar and related offload prior art](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md) — `.agents/prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md`.

- [vLLM Concurrency Prior Art and Zerogravity Architecture Reconciliation](../prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md) — `.agents/prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md`.


### Current executable/qualification/reference runbooks


- [Working in Zerogravity](../../AGENTS.md) — `AGENTS.md`.

- [Contributing to zerogravity](../../CONTRIBUTING.md) — `CONTRIBUTING.md`.

- [Zerogravity CLI](../../crates/zerogravity-cli/README.md) — `crates/zerogravity-cli/README.md`.

- [Zerogravity resident GPU library](../../crates/zerogravity-gpu/README.md) — `crates/zerogravity-gpu/README.md`.

- [CubeCL native target and compiler dependency support](../../crates/zerogravity-gpu/vendor/README.md) — `crates/zerogravity-gpu/vendor/README.md`.

- [Explicit GPU qualification frontend](../../tools/gpu-qualify/README.md) — `tools/gpu-qualify/README.md`.

- [Independent CPU reference probes](../../tests/reference/llama_cpp/README.md) — `tests/reference/llama_cpp/README.md`.


The two [CLI-option frontier](../plans/qwen-parity/CLI_OPTION_FRONTIER.csv) and [API-route frontier](../plans/qwen-parity/API_ROUTE_FRONTIER.csv) tables are unreviewed pinned parity seeds, not certified exhaustive compatibility. The current acquisition README References pin source protocols independently from the numerical oracle; use the exact version appropriate to the question.


<a id="handoff-validation"></a>

## Handoff validation and remaining limitations

This documentation request did not resume implementation, execute model/GPU work, download inputs, modify devices/services/caches or publish. The last implementation test results above are recorded accepted results atf197c9c, not a claim of fresh runtime testing during handoff writing. New-document integrity, exact open-item/decision/catalog coverage, reachable-history recovery, draft patch reconstruction and preservation checks are separately recorded after review. All unrecoverable early history is explicitly identified; no document count, Git label or generated summary repairs that gap.

### Completed documentation validation — 17 October2026

- Independent architecture/decision, complete continuation and history/recovery
  audits reviewed the assembled document. The initial overly broad historical
  loss claim was corrected after finding the embedded older94-source handoff.
  Final history approval compared every94 source title/range, all13+3 outer
  sections and all77 current commit IDs/subjects directly with primary Git text.
- Mechanical coverage verified every121 live open item, all twelve plan owners,
  all current canonical source catalog paths, and complete U-001…070,
  P-001…096, O-001…074 and A-001…006 ranges. The assumptions are prose bullets,
  so the first table-only coverage assertion was corrected; no source decision
  or expected implementation result was changed to satisfy it.
- All current local-file links and handoff navigation anchors resolve, with
  no dependency on completed documentation run files. The missing acquisition
  README catalog entry and transient history-run link found in review are fixed.
- Two portable patches independently reconstruct exactly four paused draft
  source files fromf197c9c. Both actual disposable application/byte comparison
  and an independent in-memory hunk application passed. Three context files and
  the inert generator source preserve interrupted work beneath `.agents/docs/`.
- Root README is byte-equal to HEAD; the existing decision register prefix is
  unchanged, with only the dated owner-decision consolidation appended. Runtime,
  tests, models, dependency manifests/lock, vendor/Nix/CI and SIDE source remain
  unchanged. No MAIN index entries were staged and no commit/push was made for
  this documentation request.
- Completed documentation-agent scratch runs are retired after integration.
  The original interrupted filters run and both unfinished worktrees remain.
  Runtime implementation is PAUSED; this handoff's read-first prompt preserves
  that pause until the owner explicitly resumes it.

This requested handoff is a one-time context document. Future work continues
through the existing simple TODO rules; no handoff generator, archival proof
system or per-file SHA inventory is introduced.


---

# Appendix B — Full as-of-HEAD plan and work-item snapshots

These are literal copies of all current plan TODOs and the plan-workflow rules. They are an **audit snapshot**, not a replacement live queue. The next agent must reopen the originals; do not edit the copies when implementation changes.


## Snapshot: .agents/plans/README.md (105 lines)

~~~~~~markdown
# Plans

This folder is the agent planning entry point. Start here, then work from
`TODO.md`. Git is the authoritative history (author, time, diff, provenance sha);
this folder keeps the live queue and per-plan checklists.

## Layout

```text
.agents/plans/
  README.md                 overview + spec
  TODO.md                   queue: Active / Icebox / Closed
  <plan>/README.md          plan details + author/co-authors
  <plan>/TODO.md            Base + Committed + Uncommitted + Tests
  <plan>/HISTORY.md         optional chronology (heavy plans)
  archive/<year>-Q<n>.md    quarterly rollup of closed plans
```

Plans are directories, flat under this folder; do not nest plans. One plan per
effort; split into sibling plans only when a design is too large for one TODO.

## Queue — `TODO.md`

One line per plan:

```markdown
## Active
- [~] P0 <plan> — @owner — <one-line> — [details](<plan>/README.md)

## Icebox
- [ ] <plan> — [details](<plan>/README.md)

## Closed
- [x] <plan> — <closing-sha> — <date>
```

- Active is sorted P0 → P2; moving a plan from Icebox to Active is the act of
  prioritising it.
- `[ ]` open · `[~]` active · `[x]` done · `[!]` blocked.
- P0 requires human promotion and a named incident coordinator.

## Plan — `README.md`

Open with the plan's authorship and ownership:

```markdown
Author: <user>
Co-authors: <comma-separated AI models used>
Owner: @<user|agent>
```

Then what the plan is and why it exists, plus **Design**, **Decisions** and
**References** sections as needed. Split a section into `DESIGN.md`,
`DECISIONS.md` or `REFERENCES.md` only when it grows large, and link back.

## Plan — `TODO.md`

```markdown
# <plan>

Base: `<sha>`

## Committed
### `<sha>` — <what this set landed>
- [x] <item>

## Uncommitted
- [ ] <item>
- [ ] <item> — <blocker/reason>

## Tests and verification
- [x] <test or gate> — `<sha>`
- [ ] <test or gate>
```

- `Base:` is the commit the plan started from.
- Completed work is grouped under the commit that landed it; append new groups.
- Tests are a flat list; tick each with the sha when the commit's provenance
  group is added.
- A plan is done when `## Uncommitted` is empty and every test is `[x]`.

## History — optional `HISTORY.md`

For heavy or multi-agent plans, keep an append-only chronology:

```markdown
- 2026-10-18 09:14 — @owner — opened; claimed P2
- 2026-10-18 09:40 — abc1234 — implemented parser; tests pass
- 2026-10-18 10:02 — @reviewer — reviewed; requested X
```

Light plans can skip it; the commits are the history.

## Parallel work

The owner on the queue line owns the plan directory; no one else edits it.
Parallel agents keep scratch under `.agents/runs/<plan>-<agent>/TODO.md` and
append back only when finished. The coordinator reconciles the queue line and
the plan's committed groups.

## Closeout

All tests `[x]` and `## Uncommitted` empty → move the queue line to `## Closed`
with the closing sha and date. Quarterly, roll older `## Closed` entries into
`archive/<year>-Q<n>.md` and prune their plan directories; git retains them.
~~~~~~

## Snapshot: .agents/plans/TODO.md (23 lines)

~~~~~~markdown
# Plan queue

## Active

- [~] P0 serving-and-speculation — @pvl — [details](serving-and-speculation/TODO.md)
- [~] P1 memory-and-streaming — @pvl — [details](memory-and-streaming/TODO.md)
- [~] P2 model-execution — @pvl — [details](model-execution/TODO.md)
- [~] P2 model-acquisition — @pvl — [details](model-acquisition/TODO.md)

## Icebox

- [ ] numerics-and-context — [details](numerics-and-context/TODO.md)
- [ ] cpu-gpu-optimization — [details](cpu-gpu-optimization/TODO.md)
- [ ] multimodal — [details](multimodal/TODO.md)
- [ ] distributed-execution — [details](distributed-execution/TODO.md)
- [ ] platform-and-release — [details](platform-and-release/TODO.md)

## Closed

- [x] accepted-baseline — `6d74f76` — 2026-10-18
- [x] handoff-cleanup — `bfe672c` — 2026-10-18
- [x] historical-records — `9239719` — 2026-10-18
- [x] pipeline-composition — 2026-10-18
~~~~~~

## Snapshot: .agents/plans/cpu-gpu-optimization/TODO.md (51 lines)

~~~~~~markdown
# Cpu gpu optimization

Provenance commit: `6d74f76`.
Scope: P01/P06/P07/P20/P22; follow acquisition and measure before selecting complexity.
Contracts: [fleet optimization](../../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md),
[GPU text/control](../../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md),
[FP32 FMA proposal](../../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md),
[derived quant contract](../../docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md).

## Uncommitted

- [ ] Bench and hyper-optimize in strict user order:
  Qwen3.5 2B, Qwen3.8 27B including SSD/coalesced expert streaming;
  Gemma4 E2B/26B/31B; Qwen3.8 Flash-Next Q2/~3.5-bpw/Q4
  and MTP/drafter/n-gram pipeline. Optimize complete
  request goodput with the same model semantics, source quant
  calibration and StateRoot precision. Do not change the
  Renoir/gfx90c launch hold or assume measured superiority.

- [ ] Keep the local Renoir/gfx90c hold: 2B initialization aborted and automatic resets wedged it; no further launch/reset/recovery or Drop-retirement proof is implied.
- [ ] Establish controlled cold/warm request baselines across all five GPU/three CPU optimization targets; accepted five backend profiles do not mean five cards or fleet-wide maximal capability use.
- [ ] Qualify capability-selected AVX2/AVX512/VNNI quantized kernels, cache/thread/topology policies and source-derived repacking while preserving reduction/rounding/zero/subnormal rules.
- [ ] Measure native/Vulkan tiling, subgroup/vector loads, packing, source-exact fusion/prepared dispatch; gate tensor/MMA/WMMA/integer-dot use by exact arithmetic, not advertised support.
- [ ] Evaluate explicit NativeF32 projection policy only after physical FMA edge tests, all-codec trained vectors and whole-model/cancellation/replay gates; WideF64 stays default meanwhile.
- [ ] Profile then qualify captured graphs/replay, bounded device loops/persistent scheduling, CPU/GPU overlap and exact optional GPU tokenization/detokenization; prepared plans alone are not graph capture.
- [ ] Keep separate LLVM code-generation paths and broader accelerator backends unqualified until independently built and physically tested; accepted native paths use CPP NVRTC/HIPRTC.
- [ ] Qualify optional EXL3/QTIP/trellis/codebook/low-bit derived packs and calibration identity/quality separately from original source semantics. No hidden precision substitution or default steering.
- [ ] Promote only repeated complete-request gains with cold/warm cost, tails, memory, power, source correctness, negative results and qualified capability/version guards.

- [ ] Gate future compiler/driver/libm and fast-arithmetic profiles with scalar fingerprints, physical edge/stress and trained full-logit/state tests. WideF64 is qualified on measured inputs, not a universal IEEE FMA emulation.
- [ ] Qualify real deadline wakeups, concurrent/multi-stream scheduling and actual driver failure/recovery before broader worker claims; queue deadlines currently need scheduling/poll opportunities.
- [ ] Keep the local gfx90c hold. If a healthy/recovered profile is separately authorized, localize retained allocation/H2D/final-sync initialization before changing kernels; natural SIGABRT establishes neither Rust Drop quarantine nor reclaimed storage.
- [ ] Consider read-mostly publication only after representative contention and ownership evidence. If left-right fits, test deterministic dual replay without IO/RNG/admission/state/device effects, one writer, persistent non-Sync readers, short guards without await/load/device work and explicit publish/drain/shutdown. Account for both indexes/logs while sharing one payload; do not reopen the rejected cold-catalog migration.

- [ ] Resolve applicable SYK-011..016 source/derived-quality/LLVM/SLO/pressure/steering choices from the [specialization reference](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) before that slice; proposed SYK-002..010 are not frozen user approvals. Preserve negative candidates and the default disabled steering policy.

## Committed

### 6d74f76

- [x] Complete resident original-model pipeline and five bounded promoted profiles accepted; actual rollback/admission/cancellation/quarantine remain baseline requirements.

### Concurrency review — 1fca893

- [x] No demonstrated cold-catalog contention justified left-right. Its uncommitted candidate was restored and no speedup is accepted. Immutable Arc/borrows stay direct; a delayed status view cannot authorize admission, launch, release, StateRoot publication or fence clearing.

## Tests and verification

- [ ] QTIP-01..14: derived source codec/pack identity, transforms, held-out quality and physical kernel correctness pass independently of original model semantics.
- [ ] HYB-01..18: exact source routing/draft/tap/mask/reduction and target-authoritative verification pass, including cases where a wrong draft is rejected or deeper drafting slows work.
- [ ] GFX-01..18: WMMA/LDS caller/shape coverage, driver/compiler numeric gates and coherent-memory physical headroom pass; require matched complete-request gains. See the [50-test register](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md#4-acceptance-register-50-planned-tests-not-implemented).
~~~~~~

## Snapshot: .agents/plans/distributed-execution/TODO.md (23 lines)

~~~~~~markdown
# Distributed execution

Provenance commit: `6d74f76`.
Scope: P19 plus P04/P05/P09/P10/P16/P23 extensions.
Contracts: [placement](../../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md),
[peer/state protocol](../../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md),
[accelerator families](../../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## Uncommitted

- [ ] Preserve zero-network standalone execution; add capability/transport/placement seams through the same model/task/coordinator/resource authorities.
- [ ] Qualify optional local CPU/GPU/NPU split projection, expert ownership and prefill/decode placement with actual event/memory/power-fabric safety; Stage N is independent of CPU/GPU release.
- [ ] Begin authorized two-node layer-group plus local-state execution and immutable full-state hydration; then evaluate P/D and attention+state versus FFN/expert disaggregation.
- [ ] Extend tensor/head/pipeline/data/context/expert/sequence parallelism and collectives only where exact state partition algebra, source numerics and transport cost support them.
- [ ] Authenticate/version source/state/tenant/session epochs and complete transfers; loss/timeout/stale peers cannot publish partial roots or release unknown in-flight storage.
- [ ] Qualify topology changes only at committed frontiers and compare collocation against fast/slow/tunneled links, including real transfer/copy/crypto cost and fairness.
- [ ] Add a peer/transport/placement without rewriting model, task, coordinator or API; fleet SSH qualification alone is not distributed inference.

## Committed

### 6d74f76

- [x] Hardware fleet discovery and independent per-host profiles accepted; distributed/heterogeneous model execution is not established by those results.
~~~~~~

## Snapshot: .agents/plans/memory-and-streaming/TODO.md (68 lines)

~~~~~~markdown
# Memory and streaming

Provenance commit: `6d74f76`.
Scope: P03/P05/P07/P10/P12/P16.
Contracts: [unified residency/SSD](../../docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md),
[SSD gates](../../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md),
[large-source admission](../IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md).

## Uncommitted

- [x] **Verified source-only SSD extent foundation:** original opened
  descriptor, SHA256-pinned complete chunk manifest, checked source
  offsets and per-range hashes, 4MiB chunk/32MiB read bounds,
  coalesced positional file IO and original ResourceArbiter
  transfer-staging admission before allocating/reading. Exact
  byte-window materialization returns immutable leased slices, never
  unsafe borrowed mutable-file views. Tests cover exact reads,
  truncation/hash mutation, path rename, rollback, insufficient
  host memory and sparse >5GiB metadata-only source.
  Six SSD-focused tests pass, and the entire isolated workspace
  gate passes 1,147/0 across 241 Rust suites. The Nix filtered
  source-integrity derivation cannot run offline because the
  pinned nixpkgs flake input is missing (not a source test failure).
  No trained resident-versus-SSD model, device copy or eviction yet.
- [ ] **Next SSD source execution:** source-verified per-tensor
  extent mapping and async I/O/readiness model integrated with
  original CPU/GPU TensorSource and BufferLease, cancellation,
  cache/eviction/ABA and independently trained full-logit/StateRoot
  parity on real large weights.

- [ ] Shared one-process pipeline residency: source-exact immutable
  model/component buffers, per-session private StateRoots, one
  ResourceArbiter (UMA/GPU/RAM/SSD and device queue), cross-model
  lease/zero-copy only when physical placement, codec and lifetime
  permit it. Do not replicate weights per route or release a
  GPU buffer before actual completion/quarantine retirement.
- [ ] Prioritize Qwen3.5 2B resident then Qwen3.8 27B SSD
  streaming and Gemma4 E2B/26B/31B, then sparse Qwen Flash-Next
  learned 51B n-gram bank and 4B MTP head placements. Compare
  source-real logits, resident/streamed StateRoot, cold/warm
  TTFT/ITL, I/O/read amplification and throughput with full
  physical/cancel/error gates; live audio's soft deadline may
  demand resident codecs and refuse cold SSD layouts.

- [ ] Route legacy/new model allocations, CPU state/scratch/staging and APU/GPU shared backing into the same configured physical pools with OS/cgroup headroom; never sum overlapping UMA/VRAM/GTT counters.
- [ ] Implement paged/contiguous state policy and full-domain prefix identity/intersection/COW across KV/SWA/recurrent/indexer/draft/grammar/media state, source/template/adapter and tenant boundaries.
- [ ] Qualify committed snapshot persistence/restore, corruption/security/compatibility UX and restore-versus-recompute; diagnostic portable checkpoints are not product persistence.
- [ ] Add immutable source extents and bounded coalesced IO/materialization with stable logical tensor identity, genuine copy readiness and original IO/device reader leases.
- [ ] Protect encoded-but-unsubmitted/captured expert reads and generation ABA; support mixed-codec expert size classes under one byte cap, source-exact routing and hot-cache preservation.
- [ ] Add bounded PLE/Engram lookup rows and separately governed disk state/media tiers; fully resident execution must incur no mandatory IO or host cache dispatch.
- [ ] Test source-exact resident/streamed parity, short/corrupt IO, cancellation, safe victim refusal, SSD QoS and actual physical lifetime before eviction/free-space claims.
- [ ] Optional direct/async/platform IO needs equivalent semantics and explicit support; large memory-pressure/SSD experiments require approved hardware/source scope.

- [ ] Distinguish logical envelope retirement from driver allocator pages/pools, fragmentation, cross-process use and actual OS release. Driver VRAM/GTT usage/max counters are not reservations or immediately free bytes.
- [ ] Prove trusted teardown/recovery before reclaiming unknown-work handles or clearing a fence. Diagnostic service sync/readback permits inspection/exit only and cannot synthesize Ready, retire unknown events or authorize reuse.

## Committed

### 6d74f76

- [x] Physical arbiter, original GPU leases/quarantine and committed session/checkpoint foundations accepted; global host/UMA, full-state prefix manager and SSD model streaming remain open.

## Tests and verification

- [ ] SSD-01..32: run the [full source/residency suite](../../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) for each claimed backing/backend; mock tiers cannot qualify physical IO or trained streaming.
- [ ] SSD-14/15: encoded-but-unsubmitted, queued and captured readers prevent generation-ABA slot reuse through actual GPU completion/cancellation.
- [ ] SSD-21/22/24: mixed Q2/Q4 expert sizes share one byte cap with safe victims and fair progress; hot-cache retention cannot bypass admission.
- [ ] SSD-32: an authorized original trained streamed model matches complete resident logits/StateRoot; measure cold/warm IO, staging, OS headroom and genuine resource lifetime.
~~~~~~

## Snapshot: .agents/plans/model-acquisition/TODO.md (334 lines)

~~~~~~markdown
# TODO — model acquisition

Status: active
Provenance: 9239719
Scope: P03; prerequisite to new acquisition/loader claims.
Design: [README.md](README.md).
Contracts: [A2/A3 specification](../MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md),
[pull/provider parity](../../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md).

## Committed

Completed phases through signed `fce1353` and their passed checks are in
[the completed checklist](archive/through-fce1353/TODO.md).

Completed phases from `e0073a5` through signed `a774ded` are in
[the next completed checklist](archive/through-a774ded/TODO.md).

### 837e77f — legacy offline acquisition eligibility

- [x] Reproduce five legacy offline policy failures before implementation using eight public-API tests: correct-hash LFS pointers passed repository/component eligibility; incomplete GGUF/index shards and reset index budgets passed component opening. All five failed at intended assertions, with three existing-semantics positives passing; generic CacheStore full verification legitimately accepted literal bytes.
- [x] Add one private managed-descriptor128B prefix helper using original pointer policy, cancellation checks and premature-EOF refusal. Legacy HF/GitHub offline API remains initial-selection/role-agnostic and keeps its public signature; generic cache inspect/verify remains literal-byte access. No second hash/cache/source issuer, new transport or realtime cancellation promise.
- [x] Reuse original pure split/index/sidecar closure only for explicit component Weights snapshots. Existing independent config/tokenizer suppliers, source/release coherence, roles and totals stay authoritative. Actual retained index reads share cumulative max_metadata_bytes/max_pages bounds across the whole component set; complete nested/split/release positives pass.
- [x] Focused production/diagnostic legacy8+8 and original API/lib/component/release/length/progress/transaction183 plus original CLI8+8 pass:215 actual executions/0 failures/2 intentional helpers ignored. All four strict artifact/CLI production/diagnostic configurations and scoped formatting pass. No unexpected compile/test/lint failure or post-GREEN source correction.
- [x] Full frozen isolated check.sh passes810 Rust executions/0 failures/68 ignored/174 suites plus18 namespace tests and complete lint/build/release/installer/comparator/resume gates. Original release cache_prepare passes both original routes in42.34s with both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation criteria exact. Actual Nix source-integrity includes the new test; protected README, lock and dependency manifests remain unchanged. No new model/context/GPU qualification.
- [x] Integrate all findings/results and retain native Xet, conditional/cloud/provider/convenience and separately authorized filesystem work open. Move eight completed groups and ten checked gate records through a774ded verbatim into a linked Markdown TODO archive, keeping every unfinished item live. Retire the completed content-contract worktree after14 exact source comparisons and its two integrated runs plus two primitive runs; preserve active worktrees/runs. No evidence archive, download, live account/cache/device/privilege/mount action or publication.

### 52e42fc — native HF Xet reconstruction

- [x] Wire the same exact Xet primitives into explicit native HF discovery, scoped CAS and signed-range reconstruction. Preserve original immutable HF source/pin/inventory/selection/closure and the sole HTTP/executor/cache/publisher. At least one announced native LFS object is required; discovery is body-free/no-follow, ordinary Git sidecars retain their original route and announced native failures never substitute compatibility HTTP.
- [x] Add sealed native Hub/CAS/signed-range capabilities in the original HTTP owner. Keep transient64K scoped grants separate from original bearer limits; fixed CAS origin/repository/commit and expiry apply. Native metadata and signed payloads never follow redirects; exact raw signed URI and complete interval sets are retained. CAS/signed calls suppress original credentials, including same-origin cases. Ordinary sidecar redirects use the current wire URL and strip cross-origin auth.
- [x] Validate ordered/repeated/overlapping terms and exact full-file lengths/coverage, unique bounded maps, arbitrary-order complete multipart sets, codecs, terminal source EOF and complete keyed Xet file root before publication. Fetched edge/refetched chunks count decode/work but not logical progress/root; accepted-leaf replay compares hash/length. Original raw SHA/size/LFS/cache-hit/publisher remain independent, with no invented per-term or partial-xorb authentication.
- [x] Enforce invocation-wide actual attempts/observed body+control bytes/decoded chunks/logical leaves/refreshes and cumulative accepted term/group/range counts across files/reissues. Overflow charges at most one sentinel then forbids later reads/attempts; exact-limit EOF probes work. Scratch is admitted before allocation and held alongside the original transaction. Native CLI default256MiB covers both envelopes; explicit limits and ordinary128MiB remain binding/unchanged. No global RAM/kernel/TLS traffic or whole-operation deadline claim.
- [x] Add exact ephemeral object streams through the existing cache locks/admission/private stage/installer: no native HTTP-prefix recovery, validators or checkpoints. Actual child exit73 preserves the old ref and a fresh process starts a full native transfer; ordinary sidecars keep qualified durable resume. Root review caught a draft blanket-range restriction affecting ordinary sidecars; narrow allow-only-OrdinaryPayload fix and actual retry/range/budget/ref tests pass.
- [x] Add native CLI through the original shared fetch path, metadata/discovery-only dry-run and existing-cache offline branch before credentials/client/admission/signals. Offline source/selection/current bounds/pointer/full bytes pass without claiming native execution/freshness/access. Private-source/limit/miss/pointer/corruption and feature-disabled subprocess checks pass. Existing provider routes/defaults and dependency/lock bytes are unchanged.
- [x] Final artifact551pass/0finalfail/7intentional helper ignores across diagnostic188, original compatibility179 and production184; both artifact strict all-target configurations pass. Final CLI175pass/0fail/2helpersignored/56suites across production/diagnostic/default/three disabled profiles; both CLI strict configurations pass. Actual native HTTP, default128refusal/256success, complete weights+ordinary sidecars, codecs, reordered multipart, replay, expiry/reissue, cancellation, process exit and old-ref gates pass. Independent long128KiB keyed hash/tree vectors follow six official scalar-oracle checks; no expected result was fitted to Rust.
- [x] Record all repairs: initial E0624 test access failure (outer tee exit0 hid Cargo status) fixed with test-only factory; exactly7 unused pure helpers become test-only, without arithmetic changes. Initial executable154lib/1externalpass/19fail arose from payload.bin correctly triggering model closure; change synthetic auxiliary path and add a complete weights closure case. Next154lib/20externalpass/1fail was original sanitized State versus raw Limit in an ordinary-route assertion. CLI4pass/1fail was grammatical do/does help mismatch. All fixes are narrow, final identical cohorts pass, and no runtime tolerance/redaction/closure was relaxed. Independent generator copied65/66hex XOF prefixes before Rust; correct official32byte prefixes then verify all six official vectors before freezing custom goldens.
- [x] Complete final frozen isolated check.sh867 Rust executions/0fail/69ignored/181suites plus18 namespace tests and all lint/build/release/installer/comparator/resume gates. Original release cache_prepare2routes43.26s (44.87s release rebuild) pass both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation exact. Actual offline Nix includes all9new Rust files;28 current local links, protected README and unchanged dependency/lock manifests pass. This adds no model/context/GPU qualification or live-provider throughput claim.
- [x] Integrate implementation, reviews, numerical/protocol findings, failures/results and all pending work atomically for signed local acceptance. Keep conditional cloud, selected ModelScope sizing, metadata candidate discovery and the wider provider/loader/runtime frontier open. Independent cold reviews find no conditional/sizing blocker but their execution remains unrun; do not promote alias echoes to immutable commits. No model/SDK/provider payload/account/cache/device/privilege/mount action, push, duplicate runtime or evidence archive.

### 46cf5a5 — strong conditional cloud observations

- [x] Add explicit object-only S3/Azure conditional requests and pure pins with separate s3-conditional/azure-conditional source contracts. Bind normalized endpoint/profile/locator/path/role/exact bytes/strong ETag; final original raw SHA stays FirstSeen. HEAD is admitted before transport, body-free/exact200 with complete identity metadata and original provider guards. No caller SHA, fabricated remote version, implicit fallback, prefix or cap profile.
- [x] Add one private typed If-Match condition to the existing opaque HttpTarget, None in all six original/native constructors. Every initial/range/retry emits one exact condition; original validator checks ETag before body. A mismatched resume validator refuses before signing/wire. SigV4 and Azure standard SharedKey slot match independent HMACs; HTTP412 refuses without refreshed HEAD or downgrade. Preserve all native Xet isolation/raw-target/ordinary-resume/counter tests and old cloud defaults.
- [x] Reuse the original one-entry managed verifier with private ExpectedObject/ExpectedDigest::TrustedSha256|FirstSeen, preserving HTTPS ref behavior and native digest-only pins. Exact source/path/role/bytes/provenance/current limits/full hash/pointer/cancel checks remain. No dummy expected SHA, second cache/hash/runtime/publisher or public source issuer.
- [x] Add --conditional-read to existing S3/Azure object handlers and exact offline --bytes/--etag/snapshot inputs. Online dry-run performs HEAD metadata only; offline finishes before env/credential file/FIFO/stdin/client/admission/signals. Escaped JSON ETag receipts preserve declared observation without asserting remote freshness/access. Explicit version/caller-SHA conflicts refuse; original auth/SAS scope and shared fetch/publisher remain unchanged.
- [x] Root and independent review find no protocol blocker; three literal HMAC vectors independently match Python stdlib before Rust gates. Reconcile conditional delta onto signed native52e42fc, preserving all original/native target code/tests; eighteen unowned baseline paths stay exact and only eleven approved source paths merge. A scoped SIDE formatting import-order failure was corrected before freeze; no MAIN compile/test/lint/source repair occurs.
- [x] Final focused872passes/0fail/16intentional ignores:714artifact (443diagnostic/271production),151CLI (75production/76diagnostic),4default-diagnostic and3disabled-profile tests. New conditional API10diagnostic/8production, all wire/signature/HEAD/resume/ETag-before-body/412/old-ref/offline tests pass. All four strict artifact/CLI configurations pass; original provider/native/legacy/cache/process/fault regressions stay green.
- [x] Complete final frozen isolated check.sh886 Rust executions/0fail/69ignored/188suites plus18 namespace and full lint/build/release/installer/comparator/resume gates. Actual Nix includes both new tests and exact Azure fixture;28 local links/protected README/unchanged lock/dependencies pass. Separate exact ignored release cache_prepare routes pass1test17.40s and1test23.69s (sum of test durations41.09s, not one invocation/wall time), both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation exact. No new model/context/GPU or live-cloud qualification.
- [x] Integrate all implementation/review/results and remaining provider work atomically for signed local acceptance. Retire two completed native worktrees after26 exact native source matches and CLI9current+2exact superseded-baseline matches; remove five integrated native runs. Preserve active conditional/ModelScope/candidate/quant work. No new dependency, download, account/cache/device/privilege/mount action, publication or evidence archive.

### 4db5efb — ModelScope selected exact sizing

- [x] Recover absent selected ModelScope file/index Size with one admitted body-free exact200 HEAD at the same fullcommit Direct target. Cache the exact length once before index GET/final planning, retain native raw SHA/provenance and source identity. Unselected missing sizes do not probe; malformed present/null blob sizes still refuse, with existing directory-null behavior preserved. Update the old malformed fixture to explicit null and clarify CLI sizing help.
- [x] Extend only the original provider metadata Budget for HEAD requests and returned header controls, shared with inventory/index bytes and logical call limits. Exact identity Content-Length, file/snapshot limits and no-follow status/framing rules apply. Strong ETag is checked before index/payload bytes; weak/absent validators grant no resume authority. Shared HTTP, native Xet, cloud profiles, cache, runtime, public APIs and dependencies remain unchanged.
- [x] Root proposal/source review and independent five-path review find no policy blocker. Native official-token GETtree/HEAD/GET and no-follow wire tests pass; selected/index size caching, unknown/unselected/zero/oversize/malformed fields, cancellation/budgets/ETags/rawSHA/old-ref/closure/source identity cases pass. Alias/CDN authority stays explicitly open; synthetic metadata grants no real provider or loader qualification.
- [x] Initial diagnostic all-target command exits101:156library tests and earlier providers pass, new sizing1pass/9fail. The old missing-size error is absent from live source but present in retained non-test rlib/executable; preserved SIDE mtimes predate previous builds. Refresh ONLY5merged-file timestamps after exact byte equality, then identical rerun453pass/0fail/13helpersignored confirms causal stale build. No Rust, fixture expectation, protocol or tolerance repair. Future copies use fresh modification times; record this Cargo constraint in AGENTS.md.
- [x] Final focused891passes/0fail/16intentional ignores:733artifact (453diagnostic/280production),151CLI,4default-diagnostic and3disabled. New sizing10diagnostic/9production pass and original provider/native/conditional/legacy/cache/process/resume/CLI tests stay green. All four strict artifact/CLI configurations pass with no lint/source correction.
- [x] Complete final frozen isolated check.sh896 Rust executions/0fail/69ignored/191suites plus18 namespace and complete lint/build/release/installer/comparator/resume gates. Actual Nix includes the new sizing test;28 current local links/protected README/unchanged lock/dependencies pass. Separate exact ignored original release routes pass17.64s and23.60s, both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation exact. No additional model/context/GPU qualification.
- [x] Integrate findings/failure/results and every unfinished task atomically for signed acceptance. Retire completed conditionalSIDE after29exact source/baseline matches to46cf5a5 and its three integrated runs; preserve active sizing/candidate/quant/filter work. No model/SDK/provider payload/account/cache/device/privilege/mount action, push, new dependency/runtime or evidence archive.

### f22f759 — metadata-only pinned file candidates

- [x] Add bounded metadata-only HF/GitHub candidates API/CLI through the original pinned-tree owner. List all regular choices without selecting a family, probing lengths, reading indexes/payloads or publishing. Full inventory collision/control validation precedes subfolder projection; metadata admission and logical-call/body budgets remain cumulative.
- [x] Expose honest pinned-tree versus verified cached-subset coverage, optional sizes and filename-role hints. JSON labels stored-Git versus HF tree/LFS versus verified-content lengths; Git OIDs and cached FirstSeen hashes never become trusted expected SHA. Offline source/current bounds/full hashes/pointers precede credentials/client/admission/signals; original Files/Release semantics remain.
- [x] Independent candidate review catches a regular-ancestor collision hidden by subfolder projection; fix complete-inventory ordering and add HF/GH negatives plus real-directory positive before execution. No executed RED claim. Final943focused passes/0fail/16helpers and four strict configurations pass without a compile/assertion/lint/source correction.
- [x] Frozen candidate full isolated `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passes915Rust/0fail/69ignored/198suites plus18namespace/all normal quality gates. Two separate exact release cache_prepare calls pass: approved_cached_models_match_direct_source_exactly17.47s and approved_hf_import_models_match_direct_source_exactly23.65s (`--release --locked --offline -p zerogravity --features cache --test cache_prepare NAME -- --ignored --exact --nocapture`, same wrapper). Both approved.8B/2B/all248320logits/all42domains/two tokens/session/rawlease/mutation/reclamation remain exact. These are individual test durations, not one combined invocation.
- [x] Integrate implementation/reviews/results and every unfinished task for signed candidate acceptance. Keep explicit quant, full filters/excludes, sidecar/preset/auth/provider/runtime frontier open. Preserve root README and dependencies; no model/SDK/provider payload/account/livecache/device/privilege/mount action, push or new runtime. Native source and cache bytes grant no additional model execution qualification.
- [x] Actual candidate source-integrity Nix passes with four new files included; all9source paths match frozen SIDE,26local documentation links/root README/dependencies pass. Retire completed ModelScope sizing SIDE after5exact signed4db5efb/MAIN matches and no process use; integrate future finite CDN authority/credential/refresh boundaries before deleting four completed sizing/review/frontier runs.

### f197c9c — explicit GGUF filename quant families

- [x] Add private-field GgufQuantRequest and explicit HF/GitHub/native filename-quant selection within the original retained provider tree and closure. Keep public Selection/PullRequest shape, original source identity, mandatory sidecars, same admission/logical counter and sole shared fetch/publisher. No default/rank/fallback or tensor eligibility follows.
- [x] Reuse strict canonical split decomposition, ASCII1..32letter-led tag grammar and terminal basename boundaries. Every matching family must be complete before uniqueness; malformed/incomplete/conflicting-total/case/single+split ambiguities refuse. Independent review's directory -of-/tag counterexample is repaired with basename-only inspection, preserving full path family identity; regression cases drafted before execution, no RED claim.
- [x] New quant offline requires explicit snapshot digest and verifies original source/current bounds/managed hashes/pointers/complete selected closure plus Weights/Config/Tokenizer roles before credentials/client/admission/signals. Report retained-subset scope, keep other retained data and original Files offline semantics. New native quant fixture is announcement/plan qualification only, not a CAS execution or inference test.
- [x] Quant focused975pass/0fail/16intentionalhelpers and all4strictartifact/CLI production/diagnostic configurations pass without compile/assertion/lint/source repair. New11API/5enabledCLI/disabled profiles and all original provider/component/native/legacy/cache/resume cohorts pass. After interruption, all11MAINsource paths still equal frozen draft; the sole existing Cargo owner continued.
- [x] Frozen full isolated `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passes934Rust/0fail/69ignored/205suites plus18namespace, all formatting/workspace/dependency-isolation/lint/build/release/installer/comparator/resume gates. Source remains unchanged. No new provider/model/SDK download; installer fetches are original owned local fixtures.
- [x] Both original exact release cache_prepare routes pass separately under jobs2/Rayon4/native isolation: approved_cached_models_match_direct_source_exactly17.60s; approved_hf_import_models_match_direct_source_exactly23.70s (`cargo test --release --locked --offline -p zerogravity --features cache --test cache_prepare NAME -- --ignored --exact --nocapture`). Each verifies both approved.8B/2B/full248320logits/all42domains/two tokens/sessions/rawsource lease/mutation/reclamation exact; individual reported test durations, not one aggregate call.
- [x] Wrap currentphase after explicit user pause: revalidate HEAD/worktrees/all11MAINfrozen sourcepaths and all15C3baselinepaths/C4exact3draftdeltas, originalrootREADME/deps and26local links; actualNix passes after3newpaths included and again after interruption. Both draft owners stopped/nochildren, full design/pending tests preserved in namedSIDEs. Quant validator returned exclusive Cargo after allcommands/children exit0. Integrate acceptance and all unfinishedwork for signed local commit, then pause; no push/download/model/GPU support change. Earlier C1 rolling-doc script expected7checkeditems but correctly stopped before mutation on actual6; inspected/corrected count and rolled exactly6, no implementation or test change.
- [x] Actual quant source-integrity passes after3newfiles add-N; all11source paths match frozen draft,26local links/root README/allCargo manifests+lock unchanged. Root approves C3 isolated source draft with fixed bounded fnmatchcase/exclusion/closure semantics and C4 native-index isolated source draft with explicit apply opt-in/default no-CAS and all-counter continuation; neither draft is execution evidence.

### 1a22597 — completed comprehensive handoff and source preservation

- [x] At the owner's documentation-only request, create the [comprehensive17Octoberhandoff](../../docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md) and [copyabletakeoverprompt](../NEXT_AGENT_PROMPT_2026-10-17.md), preserving runtimepause. Consolidate all U056–070 decisions, V1–V5.1/currentcontracts,77reachable commits, all121openplanitems, actualprofile/test/failure limits and complete document/research catalogs. Independentreview recovers an older94-source U040/G3b archive nested at7e61cb6; compare all94ranges and13+3 outersections exactly, retain explicit original33graph/later129/191 gaps. Preserve two byte-exact portable sourcepatches/fourunfinishedfiles plus pausedcontexts/inertoracle underdocs; actualapply and independentin-memory reconstruction pass. Links/coverage/README/appendonlydecisions/runtime/deps/index/pausedSIDEs remain intact. No runtime/model/build/GPU execution or publication; retire only completed documentationruns after integration.

- [x] After signedf197c9c acceptance, retire completed quantSIDE after15exact source/baseline matches toHEAD/no process use and exact cold-design copy in preserved indexSIDE; remove four integrated completed quant runs. Leave interrupted filters MAIN/SIDE runs and both unfinished source worktrees. This rolling TODO update belongs to the next commit; no new implementation or testing remains active.

### 04363c3 — bounded C3 Unicode filters and exclusion closure

- [x] C3 full-path filters resumed and source-qualified after the pause:
  exactly 11 intended Rust paths (3 new, 8 changed), fresh copy mtimes,
  one-pin HF/GitHub/native-Xet filtering, required-index-before-GET/shard/
  config/tokenizer veto, optional omission, exact retained offline closure,
  old Files/quant policy and shared original cache/admission/publisher.
  Independent CPython3.14 literal73 +33345-pair frozen digest, 4 filter
  matcher/adversarial units, 11 API integration tests (including actual
  private cache/ref publication on both providers), 4 enabled CLI tests
  and disabled-feature refusal pass. The original C2 candidate/quant suites
  and real private-loopback HTTP/TLS fixtures continue to pass.

- [x] Frozen C3 complete isolated check.sh: 956 Rust executions/0fail/
  69ignored/212 suites, plus 18 namespace fixtures, strict artifact/CLI
  production+diagnostic lint, workspace/dependency isolation, CPU build,
  installer, shell/Nix formatting and reference/comparator/resume self-tests.
  The first long run was interrupted during release compilation by a tool
  runtime failure (not an implementation test failure); the next frozen
  full run exited0 after the source-equal release build completed.

- [x] Actual Nix filtered-source integrity succeeds with all 3 new Rust
  paths included; checked current Markdown links, unchanged root README,
  Cargo.lock/manifests and no unrelated source differences. Two fresh
  separately invoked isolated original release cache_prepare tests pass:
  direct/cache 16.17s and HF-import 21.64s; each validates both approved
  Qwen3.5 checkpoints/all248320 logits/all42 state domains/2 tokens/sessions/
  original rawlease, mutation and removal behavior bit-for-bit. No new
  model, GPU, context or performance qualification is inferred.

- [x] Record the October17 user pause as a historical handoff only:
  C2 was signed at f197c9c, two preserved source drafts survived, and the
  explicit October18 resume moved C3 onto MAIN without losing C4.

- [x] Recover and complete the C3 paused source from
  /home/pvl/spaces/abird/src/zerogravity-filters-work. At takeover only
  filters.rs existed and was unwired/uncompiled; its frozen primary vectors
  and source/ownership policy are now implemented with API/CLI/native/closure
  regressions. Preserve the original paused source/run for traceability until
  reviewed retirement; no new SDK/model/provider/account payload fetched.

### 8195c18 — verified required native Xet indexes

- [x] C4 required native Xet indexes recovered from the preserved
  /home/pvl/spaces/abird/src/zerogravity-xet-index-work source-only draft.
  One isolated C3-base worktree performed a strict three-way merge of the
  three C4 production source deltas; exactly two HF/Xet dispatch conflicts
  were resolved without losing C3 selection/required-index veto. A cold
  typed IndexReader replaces the draft's uncompileable higher-ranked FnMut
  closure; original HF metadata parser, native Access/reader and publisher
  stay authoritative. Only seven reviewed Rust paths differ from C3 MAIN
  (five modified and two new integration tests). The original paused SIDE
  and its portable patch/design context remain untouched.
- [x] Preserve default library resolution and Xet CLI dry-run as body-free:
  required native LFS indexes refuse before CAS grants or signed body.
  Only CLI APPLY clones native options with explicit
  with_index_reconstruction; no new public flag, network client, cache,
  source issuer, device/runtime or parser. Original same-commit pinned
  index gets exact length+trusted LFS rawSHA, full keyed Xet root/EOF,
  raw SHA and LFS-pointer verification before JSON parse/shard closure.
  Planning uses one original admitted scratch; publication reconstructs
  and verifies the same index again in the existing cache transaction.
- [x] Carry ALL Xet consumed attempts, observed wire/control bytes, decoded
  chunks, logical leaves, refreshes and accepted terms/groups/ranges as a
  validated Result seed across resolution, publication and later weights;
  poisoned/tightened owners never silently zero. Native default256MiB
  admits index planning; explicit128MiB denies before signed index data.
  Ten new real original-owner API tests include independent abc Xet golden,
  exact index→shard→cache/ref publication, C3 native filtered positive
  and excluded-index-before-GET negative, wrong SHA/root/JSON/truncated
  signed body/cancel/old-ref, missing shard and aggregate budget refusal.
  Two private budget tests, two enabled CLI and one default-disabled CLI
  test also pass. Existing native Xet, C1 candidates, C2 quant, C3 filters,
  HTTP/TLS, offline and cache regressions remain green.
- [x] Final frozen isolated C4 check.sh exits0: 973 Rust executions,
  0 failures, 69 ignored, 219 suite runs, plus18 native namespace fixtures
  and full CPU/dependency/strict-lint/build/release/installer/Nix-format/
  comparator/resume gates. All four strict artifact/CLI production/diagnostic
  configurations pass after one test-only initializer style correction.
  An early 4/5 fixture failure selected an absent shard before reaching
  the required index; corrected fixture to select existing index. A
  separate truncated-body fixture initially ran out of scripted retries;
  bound its original request budget so refusal occurs before unscripted
  transport. Both corrected focused suites and final gate pass; no
  source arithmetic/protocol/tolerance relaxation or claimed RED test.
- [x] Actual final offline Nix filtered-source check includes both new Rust
  tests. Current local document links, original README, Cargo.lock and
  all dependency manifests are unchanged. Two separate current accepted
  Qwen3.5 release cache_prepare routes pass from the real cargo command:
  cache/direct16.08s and HF-import21.67s; each covers both original checkpoints,
  full248320 logits, all42 state domains, two tokens, independent sessions,
  source leases, mutation and reclamation exact. No new model/GPU/context,
  live provider/account/SDK/data download, privilege/mount/device operation,
  push or publication is implied.

### f56b574 — trusted AWS-standard FULL_OBJECT SHA256

- [x] C5 AWS-standard FULL_OBJECT SHA256 checksum admission:
  one six-file Rust slice with explicit --aws-full-object-sha256 and
  library builder. Only canonical AWS commercial-region immutable-version
  object HEAD requests x-amz-checksum-mode: ENABLED; original SigV4 signer
  includes the exact header, anonymous public profile sends it literally.
  HEAD redirects and absent/COMPOSITE/malformed/untyped checksum data refuse;
  only canonical base64 raw32 SHA256 from FULL_OBJECT plus matching pinned
  version/length becomes TrustedSha256. Independently supplied caller SHA
  must agree. No compatible endpoint, conditional/content mode, prefix or
  other algorithm gains remote SHA authority or fallback.
- [x] Preserve source/byte/provenance and original single publisher;
  dry-run remains metadata-only, offline requires the observed --sha256,
  exact version/length/snapshot before credentials. Seven dedicated
  original executor/cache API fixtures (exact AWS checksum, trusted vs
  FirstSeen, incompatible scope, malformed/composite/missing SHA, offline
  source, wrong GET/ref rollback, admission/cancellation) and one S3 CLI
  extended enabled test pass. One independent frozen Python-stdlib SigV4
  literal golden and two actual signed/unsigned private-loopback HTTP HEAD
  /redirect tests pass; real S3/C3/C4 regression suites remain green.
- [x] Final frozen C5 source/acceptance checks: 984 Rust executions /
  0 failures / 69 intentionally ignored / 222 suite executions in
  isolated check.sh, plus18 Python namespace fixtures, complete CPU,
  strict artifact/CLI production+diagnostic Clippy, release/installer,
  shell/Nix formatting and reference self-tests. Source-integrity passed
  using git add -N for new S3 API suite; local Markdown links, root README,
  Cargo.lock/manifests unchanged. Two independently invoked original
  release cache_prepare model routes pass: cache/direct16.03s and HF
  import21.45s, each both approved Qwen3.5 checkpoints/full248320 logits/
  all42domains/two tokens/independent sessions/source leases/mutation/
  reclaim exact. No inference/multimodal/GPU performance qualification
  is implied. Acceptance requires a signed local exact-path commit;
  no push or external provider payload testing is implied.

### d5aa5cb — original-model v1 source closure and execution cutover

- [x] Model-pull v1 execution cutover: close the finite companion gap
  for original index-first Qwen3.8 and Gemma4-like source layouts,
  using one shared bounded optional same-folder companion list in original
  HF and shared GH/cache closure. Known chat template, processor/image/audio/
  video metadata and alternate tokenizer files remain opaque source bytes;
  no arbitrary directory recursion, inferred model variant or auto-loader.
  Source-only two-provider synthetic >51GB indexed shard tests confirm
  no weight GET or cache publication when disk admission is insufficient,
  plus explicit GGUF/companion filtering/source closure and CLI index-first
  large-disk-budget argument validation. Keep 4GiB executable-source gate,
  original model numerical/CPU/GPU and authorization tasks in model-execution.
- [x] Final frozen model-pull v1 source/transfer gate: 988 Rust executions,
  0 failed, 69 intentionally ignored, 222 suite executions under exact
  original scripts/check.sh with private user/network namespace and guarded
  loopback. Strict artifact and CLI production+diagnostic lint, CPU workspace,
  release build/installer, Nix formatting and 18 Python namespace fixtures
  pass. Actual offline Nix filtered-source-integrity passes; root README,
  Cargo.lock/manifests unchanged, current local plan links checked.
  Independently reran original approved release cache_prepare sources:
  cache/direct16.38s and HF-import21.88s (both 0.8B+2B,
  all248320 logits/all42domains/2 tokens/source lease/mutation/reclaim exact).
  New source-closure tests cover original HF/GH GGUF+12 known metadata
  companions, filtered optional omissions, strict exact offline subsets
  and synthetic two-shard >51.6GB index plan refusing insufficient disk
  before any weight GET; no actual weight/model/SDK download and no license,
  trained-execution, GPU or general-provider compatibility qualification.
  Four bounded Rust files and four plan files only; remaining separately
  qualified provider/mirror/gating/filesystem/SSD throughput work stays
  explicitly unchecked without obstructing the model-execution CPU-first plan.

## Uncommitted

- [ ] Canonical source-pull execution order (18 Oct decision): Qwen3.5 .8B/
  2B, Qwen3.8 27B; Gemma4 E2B/26B-A4B/31B, each exact chosen GGUF
  quants or indexed original safetensors; QwenImage2.1 original
  Diffusers components; Qwen3 TTS/ASR; Whisper; NVIDIA Canary and
  NeMo Parakeet; Qwen3.8 Flash-Next original plus separately sourced
  Q2/~3.5-bpw/Q4 quants and independently verified MTP/n-gram
  embedding banks. Do not guess literal Q3.5 quant tags or conflate
  runtime speculative n-gram with trained n-gram checkpoint tensors.
  Implement exact typed presets/component assembly and run real
  metadata-only then authorized actual payload/sidecar/checksum
  publication gates. DeepSeek V4.1 Flash and GLM5.3 Flash have an
  explicit NO PAYLOAD download hold; prepare metadata pipeline only.
- [ ] Extend the new bounded HF/GitHub Auxiliary-only --bundle to
  source-verified Diffusers model_index nested shard sets, ASR/TTS
  processors/codecs, NeMo archive streaming/extraction and
  release-specific recursive dependencies under one pin/manifest.
  Current --bundle is a user-selected subset and cannot claim a
  complete model/codec assembly, support auto-loading, or run without
  newly qualified source parser and license checks.

- [x] Complete the observed public Qwen3.8+Gemma4 source metadata
  filenames in original pinned same-folder closure. 2026-10-18 actual
  tokenless HF --dry-run returned Qwen3.8-27B commit
  1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0,
  18 shards/55,586,113,293 selected bytes; Gemma4 26B-A4B-it commit
  4d7ae4984b7db7de8f8457170b3f1a419ee76d52,
  2 shards/51,644,338,194 selected bytes. A third tokenless
  HF --dry-run resolves Gemma4 E2B-it commit
  3e22461f65e89153144f8adb70e3b8c2cc9845a7,
  1 unsharded safetensors/10,278,848,001 selected bytes. The actual
  observed Qwen video_preprocessor_config.json and crc32.txt (opaque
  non-trust data) required two additive optional companion filenames.
  Both present in real result; no credentials, tensor weights, cache
  refs, SDK, model execution or GPU touched. Synthetic HF/GH model/quant/filter and
  offline exact closure tests cover both additional filenames.
  Final exact source cutover gates pass: isolated check.sh 988 Rust
  executions/0fail/69ignored/222 suites plus18 namespace fixtures and full
  strict production/diagnostic lint, CPU/release/installer/Nix-format/
  reference self-tests. Actual offline Nix source-integrity and local
  docs links pass; unchanged root README, Cargo.lock/dependency manifests.
  Separate original release source/cached model regressions both pass:
  cache/direct16.88s and HF-import21.72s, each both approved Qwen3.5
  checkpoints/all248320 logits/all42 state domains/two nonzero tokens/
  independent sessions/lease/mutation/reclaim bit-exact. This remains
  tokenless live metadata-only Hub resolution; there is no trained-source
  byte acquisition, inference, 8K drift closure, model-format or physical
  GPU qualification. Sign only the two additive Rust paths and relevant
  plan documentation; no push, tag or user-cache/device mutation.

- [ ] Extend cloud profiles beyond the narrow qualified AWS FULL_OBJECT SHA256 object route: broader S3 partitions/compatible vendors/credentials/checksum algorithms, Azure latest/SAS/managed chains and per-object digest-map prefixes. Keep explicit version, caller-SHA and conditional FirstSeen contracts distinct; never infer immutable versions or trusted raw hashes from unqualified validators or multipart COMPOSITE checksums.
- [ ] Extend ModelScope only through separately qualified alias authority and redirect/CDN/mirror/credential profiles. Selected missing sizes now use exact pinned HEAD; no alias-to-commit authority was established from revision echoes, timestamps or file last-modified IDs. The current profile does not close broader parity or independently prove server revision/membership behavior.
- [ ] Measure and qualify native Xet fetch coalescing/chunk caching and any persistent native-prefix protocol. The current bounded per-intersection complete-signed-set reader has counted overfetch/refetch and no performance claim; native process interruption restarts a full transfer. Keep per-term/partial-xorb proof limits, fixed CAS scope and existing elapsed-time/HTTP-trailer boundaries explicit.
- [ ] Close remaining finite convenience/compatibility gaps: required sidecar/adapter/sparse-quant policy, bounded auxiliary/presets, mirror/endpoint/env precedence and gate/license diagnostics. Preserve explicit source authority/no silent numeric fallback and separate loader/runtime plans.

- [ ] Qualify any future v1 migration only after separate authorization and proven reader/process quiescence; current removal supports originally protected roots only.
- [ ] Extend cleanup only through separately qualified exact ownership/recovery protocols for dead retained staging/general orphans and any later eviction policy. Current exact-snapshot operation deliberately retains them and every stable lock inode.
- [ ] Any additional lazy/raw-path/mmap consumer must adopt managed descriptor/open ownership or remain externally quiescent during removal. Observed named cache charge is not a process-global final-close/physical-free-space attestation.
- [ ] Qualify additional filesystem pairs before extending the explicit allowlist. Different st_dev values alone do not prove independent backing objects for Btrfs subvolumes, overlay/FUSE or virtualized views; retain branch-specific root device/mount checks and complete inode/ancestry exclusion. No unsupported pair gains acceptance from generic mocks.
- [ ] Separately authorize any needed namespace-only hidden-bind/subvolume/overlay filesystem fixtures before mount operations. Real target-root/unselected inode aliases and hidden bind aliases in both directions remain NOT RUN; no host mounts, block devices, new privileges or live cache actions are implied. Current tests use already existing distinct filesystems only.
- [ ] Independently qualify any broader local layouts, renamed/unnamespaced roots, missing mount identities or new destination-root creation before exposing support. Keep explicit selection, retained descriptors, bounded topology work and existing cancellation/rollback limits; real import still needs user-selected source/repository/commit/files and copying authorization.
- [ ] Finish the finite provider/loader parity ledger beyond qualified HF/GitHub/HTTPS/OCI/GCS/versioned-S3/Azure/ModelScope and explicit local-HF import slices: remaining external-cache layouts/filesystems, additional model hubs, unversioned/broader S3/Azure/compatible profiles, native Xet and broader OCI profiles. Opaque serialization/plugin loaders remain separate runtime work.
- [ ] Continue explicit source/license authorization and trusted-digest versus first-seen provenance through additional providers/loaders; acquisition grants no license or loading support.

## Tests and verification

- [ ] ACQ-A12: complete the finite provider/loader parity inventory and independently qualify remaining supported paths.

- [x] Legacy offline tests-first5RED/3positive, then215 focused passes/0 failures/2 helpers ignored and all4 strict configurations pass. Full frozen isolated check.sh810pass/0fail/68ignored/174suites and original cache_prepare2routes42.34s pass exact original criteria; actual source-integrity/current links/protected README/unchanged dependencies pass. No unexpected repair.
- [x] Native Xet final artifact551+CLI175 passes and all4 strict configs, documented test-access/fixture/grammar fixes only; full frozen isolated check.sh867pass/0fail/69ignored/181suites, actual source-integrity/28links/README/unchanged dependencies and original cache_prepare2routes43.26s pass exact original criteria. Native index closure/coalescing/persistence remain open.
- [x] Conditional cloud final872focused passes/all4strict/no MAIN repairs; full isolated886pass/0fail/69ignored/188suites, actual source-integrity/28links/README/unchanged dependencies and two original exact filtered routes17.40s+23.69s pass. Broader cloud authority/auth/prefix profiles remain open.
- [x] Selected ModelScope sizing891focused/all4strict and full896pass/0fail/69ignored/191suites pass; actual Nix/28links/README/unchanged dependencies and original exact routes17.64s+23.60s pass. Initial stale-library101 is resolved by byte-equal timestamp refresh, with no source/policy repair.
- [x] Candidate final943focused/all4strict and full915pass/0fail/69ignored/198suites+18namespace pass without repairs; actualNix/26local links/README/dependencies and original exact cache/direct17.47s+HF-import23.65s pass. Coverage remains metadata discovery/verified retained subset.
- [x] Quant final975focused/all4strict and full934pass/0fail/69ignored/205suites+18namespace pass without repairs; actualNix/26local links/rootREADME/dependencies and original exact cache/direct17.60s+HF-import23.70s pass. Interruption revalidation found no source drift; user requested pause and unfinished C3/C4 drafts remain preserved.
- [x] C4 exact native LFS-index all-counter/source/authorization validation,
  original HF/GH/Xet/C1/C2/C3 plus CLI/default-offline regressions,
  all four strict feature/lint configurations and final frozen source-integrity
  are qualified above. Full final check.sh 973/0fail/69ignored/219suites and
  two original release cache_prepare routes16.08s/21.67s pass separately.
- [ ] Each remaining provider phase requires focused API/CLI/protocol/auth/version/range/offline/fault gates, full frozen isolated check.sh, relevant source validation, actual source-integrity and documentation/dependency/root README checks before signing.

## Deferred

- [ ] Unsupported plans fail closed until qualified: unknown-length required/model-closure files, capped resumable transfers, unqualified external-cache layouts/filesystems and remaining cloud/OCI profiles, legacy-root removal and unmanaged-reader concurrent reclamation remain explicit above. Independent full HTTP trailer validation is unqualified under the documented decoder boundary, rather than a reason these payloads universally refuse. Independent component data grants no additional loader eligibility.
- [ ] Linux local-filesystem semantics are qualified first. HTTP cancellation obeys configured whole-request bounds; old retained attempts consume real quota and arbitrary same-UID mutation remains outside descriptor isolation.
~~~~~~

## Snapshot: .agents/plans/model-execution/TODO.md (365 lines)

~~~~~~markdown
# Model execution

Provenance commit: `6d74f76`.
Scope: P02/P03/P08/P11–P14/P17; CPU truth before GPU specialization.
Contracts: [large immutable sources](../IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md),
[Qwen basics](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md),
[Gemma family](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md).

## Uncommitted

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
- [ ] **Independent trained BF16/FLA checkpoint gate:** obtain
  source-authorized original Qwen3.8 checkpoint and tokenizer
  on a physically admitted host; capture true independent
  Transformers/FLA layer-reference values and code revision with
  pinned SHA256, and compare against this gate before enabling
  real 27B full logits/StateRoot or serving.
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
~~~~~~

## Snapshot: .agents/plans/multimodal/TODO.md (73 lines)

~~~~~~markdown
# Multimodal

Provenance commit: `6d74f76`.
Scope: P17/P18 and the existing v5/v5.1 extensions.
Contracts: [unified execution](../../docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md),
[model assemblies](../../docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md),
[realtime QoS](../../docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md),
[implementation gates](../../research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md).

## Uncommitted

- [ ] Opt-in, source-granted context-filter outputs as bounded,
  immutable, typed hints to ASR, acoustic-first diarization,
  transcript-text and separately source-qualified speaker-preserving
  translation stages. Default only prior committed session user/assistant
  records; opt-in tool call/result, app system/developer prompts and
  explicitly authorized shared data permitted. Generic include/exclude
  and typed decision-model choices operate after source ownership checks.
  Unsupported original ASR prompt dialect refuses or uses a separately
  verified correction stage. Actual audio A/B WER/CER/DER/translation
  adequacy, timestamp/partial-output correctness, consent/authorization
  and canceled-turn isolation are required before any runtime claim.

- [ ] Model each ASR (Qwen ASR/Whisper/Canary/Parakeet),
  diarization (separately sourced speaker recognition), LM, Qwen TTS,
  Qwen Image2.1 and future video as registered typed components,
  not separate per-node subprocesses. The ASR→speaker-tagged
  transcript→LLM→TTS single coherent pipeline must include rational
  clocks, codec validity, VAD, per-turn epochs, barge-in,
  bounded fanout/backpressure and correct playback ACK frontiers.
  Current TOML/audio/image graph skeleton has NO real media runtime.
- [ ] Enable native progressive image output only at complete
  source-valid VAE/tile/frame boundaries; partial latents or
  unverified solver steps are never playable. Actual media clocks,
  atomic commit barriers, ROS/transport and tested streaming
  quality/latency require source-real gates.

- [ ] Add typed sample/frame/codebook/latent/decision values, rational clocks, checked conversions and complete source-bound component/codec/processor assemblies.
- [ ] Prove a small real non-token encoder/decision task and bounded iterative task through the existing engine before broad media claims.
- [ ] Qualify CPU-first original ASR and non-AR TTS (Whisper/Parakeet and Kokoro candidates); then actual backend execution and measured streaming/audio quality.
- [ ] Add depth-AR/RVQ speech, dual-AR codebook TTS and genuine codec-based duplex neural audio (including Qwen-Omni) where source/license/resources permit; mock codecs or text cascades do not establish native support.
- [ ] Qualify seeded source-exact image flow/editing conditioner+VAE pipelines, then video temporal/3D-VAE/expert switching and synchronized AV commit groups; candidate families remain separately gated.
- [ ] Keep ternary PTQ/Hadamard codecs, schema/calibration decision heads and specialist encoders as independent codec/model/task gates, not aliases of Qwen.
- [ ] Implement RealtimeSession lanes/epochs, bounded media buffers, VAD/barge-in, presentation/ack/playback frontiers and transactional rollback without speculative/partial media leakage.
- [ ] Add per-model transport capabilities and authorized hosted TaskService adapters with privacy/cost/idempotency/Unknown reconciliation; no duplicate local runtime or guessed realtime guarantees.
- [ ] Complete MM-01..24 plus model-specific ASR/TTS/image/video/decision quality/clock/resource suites; no real media-model support follows from source-free mocks.

## Committed

### 6d74f76

- [x] Architecture/specifications and bounded synthetic foundations exist; actual new audio/image/video/decision model execution remains unqualified.

## Tests and verification

Detailed numbered source fixtures remain in the linked acceptance/dossier owners.
Each applicable suite must pass at its actual source/device/transport tier.

- [ ] MM-D1/MM-01/MM-15: checked rational sample/frame/codec clocks, 44.1/48 kHz to 16/24 kHz and 24000/1001 or 30000/1001 FPS, long-duration remainder, discontinuity, overflow and bounded parser/buffer refusal.
- [ ] MM-D2/MM-02: complete source/processor/codec/license assemblies; wrong conditioner/shape/mask, missing source transform and cross-session mutable leakage refuse before use.
- [ ] MM-D3/MM-03..09: seeded solver/RNG and direct finite decision state, CTC/RNNT/TDT distinction, complete RVQ depth and joint AV commit; interrupted/stale device work never publishes or exposes media.
- [ ] MM-ASR-01..12: actual frontend/window/blank/predictor/duration/language/timestamp/diarization source fixtures; offline-only checkpoint refuses streaming, malicious audio bounded before allocation, sessions isolated.
- [ ] MM-TTS-01..12: non-AR phoneme/duration/vocoder and exact PCM count; incomplete 9/10 RVQ frame unplayable, codec-rate swap refused, overlap/voice generation and duplex/barge-in/state/lease boundaries correct.
- [ ] MM-IMG-01..12: source conditioner/solver/CFG/seed/mask/VAE/color and generation-entrypoint traces; invalid source stage/crop/processor fails, cancel retains old root and unknown physical handles, source/license/hosted status truthful.
- [ ] MM-VID-01..15: temporal VAE/latent crop/frame count, Wan timestep experts, source tile masks/rectangular attention, AV cross-attention/joint clocks, invalid GOP/backpressure, hot reload, cancellation and whole-assembly admission.
- [ ] MM-DEC-01..06: finite and joint schema head options/order/null/refusal/calibration against original source; generated text is not source-equivalent, external egress refused without authorization.
- [ ] MM-BON-01..08: PTQ1_0/PQ2_0 packed source scales/signs/padding plus mandatory 1024 Hadamard activation transform; wrong codec or plain-Q2 path refused, original scalar/CPU/device source parity separately qualified.
- [ ] MM-13..17/MM-19: real per-session/turn/transport epochs; stale ACK ignored, duplicate reconnect cursor idempotent/authenticated/bounded, playback progress remains uncertain when missing, slow clients do not release device leases or grow queues indefinitely.
- [ ] MM-18/MM-22/MM-23: ten speech/five video mixed jobs, fair shared physical admission and legal solver/frame yield; no per-packet catalog/router/global snapshot or silent precision/step downgrade, coupled output stays unplayable until both tracks valid.
- [ ] MM-SRV-01..12/MM-20/21/24: exact versioned SDK/protocol/field/task conformance, hosted accepted-timeout Unknown/idempotent reconciliation, callback/signature/TTL/MIME bounds, malformed media/unauthorized egress/cross-tenant replay refused before side effects.
- [ ] MM-I0 change-radius drill: small encoder/finite task and iterative mock use existing source/state/resource owners; no central router/worker rewrite or duplicate runtime; original Qwen gates stay unchanged.
- [ ] Eligible source-real CPU→physical full-model→playable-stream→soft-QoS gates: exact licensed checkpoints/processors/codecs and original quality/state oracles, real negative cancellation/admission/replay tests and measured cold/warm tails/memory. Mock/reference/region/hosted results never promote another tier.
- [ ] Resolve realtime section18 policy choices explicitly (media numeric policy, first sources, SLO/clock uncertainty, mid-turn changes, persistence, biometric/asset consent, hosted cost/egress, transport packaging and safe physical recovery). User examples do not establish released model IDs or grant execution/provider/download authority.
~~~~~~

## Snapshot: .agents/plans/numerics-and-context/TODO.md (30 lines)

~~~~~~markdown
# Numerics and context

Provenance commit: `6d74f76`.
Scope: P02/P11/P22, independently of optimization.
Contracts: [numerical/source profile](../../docs/NUMERICS_QUALITY_AND_BENCHMARKS.md),
[independent reference runbook](../../../tests/reference/llama_cpp/README.md).

## Uncommitted

- [ ] Resolve independent original-source CPU 8K GDN numerical divergence without loosening tolerances or changing the declared source arithmetic.
- [ ] Extend genuine resident GPU context/corpus qualification beyond the currently accepted 512 consumed tokens, including capacity/window/padding/replay boundaries.
- [ ] Run genuine whole-model 16K/32K CPU/GPU gates only after prerequisites; functional 8K success is not independent long-context numerical closure.
- [ ] Keep internal full-logit/all-state/trace bit parity distinct from independent native-source stage/top20 comparisons, including the known 2B 56 signed-zero distinctions.
- [ ] Broaden prompts/tokenizers/precision profiles and sampling quality only with independent oracles, exact history and explicit qualified ranges.

- [ ] Diagnose identical teacher-forced prefixes512/1024/2048/4096/8192 against pinned llama.cpp0.6.0/build11429 `d812350`, CPU/F32 KV and Flash Attention off. Fix the physical microbatch/chunk/arithmetic schedule and compare per-head GDN state, gain/beta, quantization and AR-versus-bulk recurrence; distinguish replay correctness from arithmetic.
- [ ] Compare native ubatch1/64/256/512 only within bounded feasible runs; the historical full ubatch1 capture did not finish. Use raw token IDs to avoid incomplete UTF-8 HTTP rows. Recheck short corpus, layers, state and actual8K after any fix; retain original functional0.45 shared-top20-logprob/0.05 top-token-probability limits and strict source-bit operator gates.

## Committed

### 6d74f76

- [x] Original CPU Phase 1 functional 16/16 through genuine 8K; bounded GPU genuine 512-token internal comparisons accepted.
- [x] Neither result closes the open independent numerical or larger-context gates.

### Accepted investigation findings — 6d74f76

- [x] Independent final-token divergence first appears in0.8B layer4 GDN and2B layer1 GDN, after preceding layers match. The separate historical AMD FMA failure is not a demonstrated CPU cause/fix.
- [x] At native ubatch512, shared-top20 logprob differences were0.20372924/0.19678696 (0.8B/2B); matching greedy7993 did not close numerical agreement.2B changes with native scheduling;0.8B did not. These partial-top20/activation observations are not full-vocabulary/KL/total-variation proof.
- [x] A large-position RoPE or padded-state fixture does not qualify a genuine long-context whole model. Canonical future-KV padding and zero signs remain source semantics.
~~~~~~

## Snapshot: .agents/plans/pipeline-composition/TODO.md (299 lines)

~~~~~~markdown
# TODO — Single-process model and pipeline composition

Status: pipeline architecture/scaffolds complete; execution deferred to owner plans
Provenance: 4bf624e (original scaffolds); ac44e83/f28a62c (context source/filter)
Design: [README.md](README.md)
Owners: model-acquisition, model-execution, serving-and-speculation, multimodal,
memory-and-streaming. All model/source/media execution remains subject to
their existing numeric/source/physical acceptance gates.

## Uncommitted

- [x] Canonical YAML schema `zerogravity.pipeline.v1` replaces TOML
  for the single-process/multi-model CLI config. Repeatable model/step/link/
  route/context flags lower into the same typed validator. Strict YAML
  256KiB source/expanded scalar, 16,384 nodes/32 depth, single document,
  deny-unknown-fields/custom-tag/recursive-alias protections; bounded
  anchors/aliases/shallow merges allowed. Ten sample YAML configurations
  cover text, audio, ASR/speaker diarization, chat+context filtering,
  NanoJev-like decisions, latent/complete image stages and video frames.
  `pipeline check` only validates graph structure; `pipeline run` still
  refuses. A maintained YAML parser replacement is a separate dependency
  review, since the already-cached `serde_yaml 0.9.34` is deprecated.
- [x] Add finite native `ComponentRegistry` with independently registered
  immutable typed port contracts, pure `Config::validate_with_registry`,
  duplicate/invalid/forged-context port refusal and source-neutral future
  stage kinds for audio feature/codebook conversion, image solve/decode,
  video frames and calibrated decisions. Every stage and source remains
  NOT IMPLEMENTED for runtime use until separately qualified.
  No per-model binary/process, runtime scheduler, GPU, transport, extra
  licensed source or user code loaded.
- [x] Frozen source-neutral architecture/YAML acceptance passed:
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  exit0, 1,073 Rust executions/0 failures/69 ignored/230 suites
  plus 18 namespace Python fixtures, strict production/diagnostic
  Clippy, CPU comparator self-tests, release CLI, installer, shell
  and Nix formatting. Both original approved Qwen3.5 checkpoints
  retain full 248320-vocabulary logits, 42 StateRoot domains,
  nonzero tokens, independent sessions and source/reclamation leases:
  cache/direct18.43s and HF-import23.39s independent release routes
  pass exit0. Nix offline source-integrity includes the ten
  YAML examples, typed parser and extension registry; root README,
  workspace Cargo.toml and owned model math/source paths unchanged.
  Acceptance is **pure configuration/registry scaffolding only**;
  trained multimodel execution, audio/video/image streams, NanoJev
  inference, source-real decoded shards and production APIs remain
  in the deferred owner plans. Sign exactly reviewed source/examples/docs;
  no push, no device launch, no model download.

## Committed

- [x] Prior accepted source and family contracts remain in
  model-acquisition, model-execution, multimodal, serving, memory and
  earlier signed history; none of those prior unqualified model/runtime
  implementations are retroactively promoted.

### 4bf624e — accepted foundation

- [x] Source-neutral ordered model target matrix for Qwen3.5+/3.5/3.8,
  Gemma4, DeepSeek V4.1 Flash, GLM5.3 Flash/full, Muse Spark, Mistral/Llama,
  additional OSS, QwenImage2.1/QwenTTS/QwenASR/Whisper/Canary/Parakeet.
  Source name/layout/release status distinguished from executable support.
  Exact local 0.8B/2B Qwen GGUF baseline not promoted to arbitrary HF repo.
  DeepSeek V4.1 Flash/GLM5.3 Flash payload hold represented explicitly.

- [x] Pure typed ComponentContract, ModelInstance, Pipeline, Edge and ApiRoute
  types in existing neutral core. Compile bounded DAG with matching semantic
  ports, required producers, topology, source-to-output productivity,
  unambiguous exposed routes and shared model IDs. Output is NOT a server,
  loaded model or qualified media graph.

- [x] Default CLI `models [list|inspect]` static truthful registry and
  fail-closed `models run|stream`. initial TOML schema (now YAML) `zerogravity.pipeline.v1`
  and repeated `--model`/`--step`/`--link`/`--route` flags lower into
  exactly the same graph validator; config file size ≤256KiB. Both single-
  model and multi-model diagrams/example configs exist. `pipeline check`
  is pure and `pipeline run` is explicitly NOT IMPLEMENTED.

- [x] Bounded pinned HF/GitHub `--bundle --include` opaque Auxiliary-only
  data selection for Diffusers/multi-stage speech assets, reusing one Hub
  pin/tree, existing filters/admission/cache publisher and exact offline
  bundle-only source identity. No invented model/codec completeness, native
  Xet bundle, unknown-size cap, arbitrary subdir recursion or executable
  checkpoint interpretation.

- [x] Implement source-neutral explicit optional chat-context policy:
  per-pipeline enabled toggle (default absent/off), shared text LLM model
  instance, originally max1..16384 UTF-8 bytes and prior committed
  same-session visible turns (historical 4bf624e semantics, now superseded
  by the configurable source selection/filtering policy above).
  Typed optional context.value inputs reach ASR/diarization/
  transcript-text/speakers-text/translation. The graph still prevents
  same-turn cyclic feedback and unowned raw KV injection. Disabled policy leaves edges but delivers None.
  Source-real contextual speech/diarization/translation, privacy-safe
  snapshot capture and context A/B quality remain NOT IMPLEMENTED.
  initial TOML (now YAML) and repeated flags compile identical shape; include full
  contextual-voice-assistant example and adversarial negative tests.

- [x] Frozen source-only acceptance: exact MAIN
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  passed 1,032 Rust executions, zero failures, 69 intentional ignores
  across 230 suite runs, plus 18 namespace Python tests, release CLI,
  Nix/shell formatting, installer, CPU numeric self-tests and strict
  default/production/diagnostic Clippy. Actual Nix filtered source-integrity
  includes the new core/CLI modules and five initial TOML examples (migrated to YAML).
  Original Qwen3.5 release cache_prepare source/model parity independently
  passes for both models, all 248320 logits, all42 StateRoot domains,
  2 tokens, session/source leases, mutation/reclaim: direct 21.05s and
  HF-import 24.10s. Both successful after an initial aborted connector
  tool invocation and explicitly corrected two test-local shadowing errors
  plus a bad nested help quote; no numeric/source/runtime tolerances relaxed.
  Acceptance uses an exact-path signed local source commit. No device
  launch, data download, live server, license approval or real inference extension.

### ac44e83 — accepted foundation

- [x] Replace fixed prior-visible-only history with a reusable typed
  ContextFilterPlan for independently configurable user/assistant messages,
  tool calls, tool results, application-owned system/developer prompts and
  explicitly named shared/granted records. Default remains session
  user+assistant; filters include/exclude by record kind, exact tool name,
  source origin or text substring in deterministic order. Typed calibrated
  NanoJev-style decision stage references a separately declared Decision
  model, accepts threshold in basis points and configurable unavailable
  policy (omit/reject/keep_authorized). The latter never reintroduces
  excluded, missing or ungranted records. Supported initial TOML (now YAML) and repeatable CLI
  flags; bound source selectors/controls and configurable 4KiB default
  through 1MiB context budget. Configuration *requests* source rights;
  runtime source owner must verify actual authorization, not a hard-coded
  block against application prompts or authorized sharing. No model
  inference/automatic context history access is claimed.

- [x] Audited all 6 old detached side worktrees against signed C3/C4/C5,
  v1 and pipeline acceptance commits; preserved earlier unfinished portable
  source patches under .agents/docs and copied nine side-only run artifacts
  into existing MAIN .agents/runs before removing all six detached
  worktrees. One new active, clean worktree
  zerogravity-worktrees/agents-conventions at unmerged e0410c7 is
  intentionally retained. No branch or unrelated user work deleted.

### f28a62c — accepted foundation

- [x] Source-grant checked context owner/snapshot service: the Rust core
  now holds committed immutable records and exact revocable session,
  app-prompt and shared/cross-tenant source grants behind one trusted
  `ContextOwner`; separately minted read-only `ContextReader` capabilities
  cannot mutate grants or rebind to another principal. Per-session prior
  turn check, bounded latest-whole window, provenance IDs/source labels,
  source epoch retry/freshness validation, and filter-only exposure to
  verified candidates are implemented. Decision adapter receives record
  text only after source grant + previous filters; it returns a typed
  include/omit verdict or explicit unavailable/error. Snapshot records
  original-source attributes and decision audit (not raw text in audit).
  This is an in-process source owner, not an external chat connector
  or pretrained model executor. External source grants must be issued by
  the trusted application before minting a read-only reader. Separate
  tests prove that a policy selector alone cannot grant access.

## Tests and verification

- [x] Source-authoritative snapshot tests pass: exact tenant/application/
  session isolation, current-turn exclusion, cross-tenant shared grants,
  same-application prompts, independent tool-result filtering,
  authorized-only decision inputs, optimistic revoke during callbacks
  without deadlock, stale/foreign reader refusal, erasure/record provenance,
  bounded latest-whole 4MiB candidate and 1MiB output windows,
  full-or-empty source behavior, decision error propagation and
  secret-free decision audit. Full original scripts/check.sh frozen
  isolated gate passed 1,058 Rust executions/0 failures/69 ignored/230
  suite runs plus 18 Python namespace fixtures, strict lint, release
  build/installer, Nix-format and CPU comparator selftests.
  Independent release cache_prepare exact approved Qwen3.5 0.8B+2B
  parity passes full vocab/42 StateRoot domains/2 token sessions/leases:
  cache-direct17.36s and HF-import21.91s. Actual Nix offline
  source-integrity passed with new Rust module, root README and
  dependencies/lock unchanged; no live model/GPU/provider access.

- [x] Frozen isolated updated-context-filter source-only check.sh
  passed 1,047 Rust executions, 0 failed, 69 intentionally ignored
  across 230 suites, plus 18 Python namespace fixtures, four strict
  production/diagnostic feature lint gates, Nix formatting, release
  installer and CPU comparator gates. Two independently invoked original
  Qwen3.5 release cache_prepare routes pass both approved models,
  full248320 logits/all42 state domains/two tokens/source-lease/
  mutation/reclamation bit-exact: direct18.66s, HF-import23.24s.
  Real offline Nix filtered-source integrity includes two new paths;
  root README, workspace deps/Cargo.lock unchanged.
  This qualifies only config shape plus pure filtering over explicitly
  *provided* already-granted record views. Live source authorization,
  history retrieval, calibrated NanoJev model inference and real media
  execution are not implemented.

- [x] Filter pipeline: independent tool-call and tool-result opt-in/
  exclusion, context source selection for app prompts/shared grants,
  no silent permission acquisition, ordered predicates, user preference
  filters and decision confidence/fallback. Include/omit verdicts must
  not resurrect previously filtered entries or invent inaccessible text.
- [x] initial TOML (now YAML) + equivalent CLI filters and NanoJev-style typed decision
  references. No inferred upstream checkpoint or executable model. Reject
  duplicate/unknown scopes, malformed origin names, unsourced decision
  models, non-decision heads, invalid thresholds, invalid return variants,
  excessive source/filter/budget and unsupported executable pipelines.
- [x] Opt-in previous-turn context *shape* passes initial TOML (now YAML) and repeated flags,
  with toggling ON/OFF on the same graph, absent-by-default channel, and
  static bounds/unknown model/non-language source/no consumer/spoofed
  current-turn feedback refusal. Configuration checking has no
  data/transport access; a later runtime may include explicitly
  requested authorized app prompts or shared user records only under
  real source grants. Runtime content/biometric/clock tests remain open.
- [x] Core shape validation passes: one-stage Qwen, model sharing across
  two routes, ASR+diarization+optional contextual translation+LLM+TTS
  and image flow. Stages are semantic placeholders only.
- [x] Source-free negative tests pass: cycles, orphan stages, wrong
  typed ports, missing speaker source, duplicate model ID/API/path/input,
  unknown schema fields, unsupported model/media routes, invalid quant/role hold,
  context spoof/size/provider/current-turn linkage and disabled output.
- [x] Default CPU & acquisition CLI tests: offline `models list/inspect`,
  `pipeline check --config` and equivalent flags/ON-OFF context toggle;
  `run/stream` refuse without source/physical/API effects.
- [x] Acquisition and CLI gates: original HF/GH closed weights selection
  unchanged; `--bundle` separate Auxiliary role/source, original cache
  publisher and exact offline cache. Native Xet bundle refuses before
  Hub/CAS. No invented FirstSeen trust or guessed release payload;
  conflicting options fail before credentials.
- [x] Full frozen isolated `scripts/check.sh` 1,032/0 failures,
  four strict artifact/CLI profile lint and core CI, Nix filtered-source
  integrity and original two Qwen reference release regressions pass.

## Deferred

- [ ] OSS release/distribution naming and "latest" pointers remain
  observational until an exact immutable source and codec version is bound.
- [ ] Remote routed pipelines/peers, encrypted hosted gateways and
  implicit zero-copy cross-device are not enabled by source-free shapes.


### Runtime work deferred until after core model priorities

All entries below are still UNIMPLEMENTED. Their active owners are
[model-execution](../model-execution/TODO.md),
[serving-and-speculation](../serving-and-speculation/TODO.md),
[multimodal](../multimodal/TODO.md),
[memory-and-streaming](../memory-and-streaming/TODO.md) and
[model-acquisition](../model-acquisition/TODO.md).
The completed pipeline-scaffold plan is not a claim of serving/realtime
availability and does not keep pipeline development on the critical path.

- [ ] Connect authenticated application/conversation history providers to
  ContextOwner, with durable and revocable grants, owner-authenticated
  tenant+application+session identity, event/graph epoch binding to the
  existing TaskCoordinator and physical memory/residency admission. Wire
  source-real calibrated NanoJev/other decision heads and streaming media
  A/B tests before enabling an actual context-fed ASR/translation router.
  Do not infer inference support from the in-memory source fixture.

- [ ] Source execution: bind actual learned model, tokenizer, processor
  assembly and port formats in original ModelCatalog/TaskDriver before
  enabling any pipeline run or route listening.

- [ ] Single-process multi-model scheduler: immutable source residency,
  exact shared instance reuse/replicas, typed cross-model buffer leases,
  physical ready fencing, bounded ingress/admission/cancellation and
  model-specific batch/fairness. One TaskCoordinator publishes StateRoots.

- [ ] Pipeline runtime/hot reload: per-session model identity, graph generation,
  buffered media/source epoch, legal stage yields, failure rollback and
  atomic route transition. No stale audio, image or cross-tenant leakage.

- [ ] API serving: explicit auth/bind/route config and tested OpenAI
  chat, ASR, TTS, images; SSE/WS/WebRTC soft-realtime with content/size
  bounds and accurate NOT IMPLEMENTED capability reports until qualified.

- [ ] Context-aware source-real voice pipeline: stage-specific ASR prompt/
  vocab support or separate contextual transcript reranker, acoustic-first
  diarization and source-preserving contextual translation. Implement
  prior-committed same-session snapshot freezing, cancellation epochs,
  explicit consent, redaction, bounded history and no unauthorized
  cross-user disclosures.
  A/B compare WER/CER/DER/translation adequacy/latency for context
  ON/OFF; never override contradicted acoustic evidence. This does not
  follow from shape validation.

- [ ] Actual staged ASR + diarization (speaker-tagged timestamps), ASR
  -> Qwen -> TTS one coherent command; audio sample clocks/codebook
  playback/VAD/barge-in and image latent/complete preview frames with
  backpressure. Validate codec/latency/source on real media hardware.

- [ ] Optimize model execution: Qwen3.5 2B, Qwen3.8 27B
  resident+SSD, Gemma E2B/26B/31B, then Qwen Flash-Next QSA/GDN/MoE/
  learned n-gram embeddings/MTP and separate speculative n-gram/draft,
  with measured model-real speed/quality/physical safety gates.

- [ ] Complete remaining exact Q2/Q3.x/Q4 quant source pipelines,
  full native image/audio/NeMo assemblies, token/codec quality and
  range/resume/cancellation before declaring "all models supported".
  Keep DeepSeek V4.1 Flash and GLM5.3 Flash strictly pipeline-only now.

- [ ] Trained and streaming multi-model parity, real media correctness,
  device completion, SLO/fairness and zero-copy lease gates are owned by
  the model-execution/serving/multimodal/memory TODOs above.
~~~~~~

## Snapshot: .agents/plans/platform-and-release/TODO.md (39 lines)

~~~~~~markdown
# Platform and release

Provenance commit: `6d74f76`.
Scope: remaining P00/P04/P05/P08/P20/P21/P22/P23.
Contracts: [P00–P23 roadmap](../IMPLEMENTATION_ROADMAP.md),
[minimal execution spine](../../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md),
[compiler/task refinement](../../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md),
[release gates](../RELEASE_ACCEPTANCE_GATES.md).

## Uncommitted

- [ ] Generalize the working typed forward plan into verified effectful semantic regions/task control, closed provider substitution, bounded shape/kernel caches and explicit command/resource legality; no new persistent IR/runtime without a real need.
- [ ] Prove artifact→model→compiled task→coordinator→output and explain-plan for supported tasks, including source-qualified preselected fallback and no selected-GPU-error CPU substitution.
- [ ] Run extension drills for one stateful semantic op, device/provider, non-token TaskProgram, quant/storage/API adapter and optional ParameterVersion/TrainStep seam; future training is not thereby implemented.
- [ ] Finish per-file source/model/license/NOTICE/SBOM policy, malformed-input fuzzing and ownership/security review, authenticated APIs/private caches and qualified driver/device safety.
- [ ] Verify Nix/module/native/cross packages and installer/release behavior on actual Linux/Windows/macOS target access; packaging definitions alone do not prove platform support.
- [ ] Maintain minimal/default/composed CPU-free feature isolation, one lock/version authority and truthful unsupported capability behavior.
- [ ] Gate bounded compiler/autotune/AOT caches by source/device/driver/shape and retain negative candidates; no unbounded first-request search.
- [ ] Complete reproducible/signing/release policy and fair model/task/quality/performance matrices with cold/warm, p50/p95/p99, memory/energy and uncertainty. Push/tag/release remain separately authorized actions.
- [ ] Demonstrate compatibility/invalidation of operator/task/state/plan caches and bounded change radius before broader platform claims.

- [ ] Before a true zero-copy strided View is enabled, validate contiguity against stored value provenance and test public PlanValue retargeting. Current View physically materializes dense output; retain cached-element and dynamic range guards.
- [ ] Preserve one GPU-library service/queue/quarantine owner and cold generic-plan admission through packaging changes; verify actual final downstream links, not only compiler/type checks.

## Committed

### 1af2ac4

- [x] NixOS module eval check evaluates a complete base system (root filesystem and boot loader) so the whole-system assertions hold on current nixpkgs-unstable; the CI `quality` job was failing at this evaluation.

### 6d74f76

- [x] Cohesive workspace, ordinary CPU/product packaging, one shared GPU library and accepted narrow original-model vertical slice.
- [x] These completed slices do not mark every original P00–P23 goal complete.

## Tests and verification

- [ ] Qualify all six [CPU release definitions](../../../docs/DEVELOPMENT.md#cpu-cross-release-definitions) with actual target execution, installer checksum/version/refusal cases and platform signing/authenticity policy. Build definitions and same-host checks do not establish other-target support.
- [ ] Test enabled NixOS service lifecycle and runtime-secret handling after serving is implemented; the currently disabled service definition and module evaluation are not a running-server claim.
~~~~~~

## Snapshot: .agents/plans/serving-and-speculation/TODO.md (159 lines)

~~~~~~markdown
# Serving and speculation

Provenance commit: `6d74f76`.
Scope: P09/P15/P16/P20/P22.
Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md),
[36 serving gates](../../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md),
[Qwen parity and decoding](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md).

## Uncommitted

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
- [ ] **P12 actual fused continuous GPU batching:** batch-shaped
  resident value/parameter/state bindings and per-row source/epoch/
  root identities, multi-row native GEMM/attention/GDN operations,
  paged hybrid KV/recurrent cache, physically safe stage overlap,
  backpressure and fair scheduling. Prove exact full logits and
  StateRoot for B1/2/4/8 and measured comparable p95 TTFT/ITL/
  SLO-goodput versus pinned vLLM V1 on same source/hardware.
  No fusion claim from wave grouping or total launch count.

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
~~~~~~


---

# Appendix C — Complete surviving document, research, prior-art, plan and recovery catalogue

Auto-enumerated from the checked 19 October repository. These relative Markdown links resolve to actual sources; titles and line counts help bounded progressive reading. A link is a source reference, not a statement that its old assertions are current or implemented.


## Public engineering/publication docs (6 Markdown sources)

- [docs/DEVELOPMENT.md](../../docs/DEVELOPMENT.md) — 160 lines; Development, builds and release definitions
- [docs/ENGINEERING_SPECIFICATION.md](../../docs/ENGINEERING_SPECIFICATION.md) — 954 lines; Zerogravity — Integrated Engineering Specification
- [docs/PUBLICATION_AND_REVIEW_CHECKLIST.md](../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md) — 89 lines; Whitepaper publication and final architecture review checklist
- [docs/README.md](../../docs/README.md) — 82 lines; Zerogravity documentation
- [docs/REFERENCES.md](../../docs/REFERENCES.md) — 95 lines; References and source provenance
- [docs/WHITEPAPER.md](../../docs/WHITEPAPER.md) — 443 lines; Zerogravity

## Normative system/architecture designs (45 Markdown sources)

- [.agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) — 274 lines; zerogravity — Universal region-execution backend contract (v4.1)
- [.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md](ASYNC_RESOURCE_COMMAND_IR.md) — 226 lines; zerogravity — Async Command/Resource IR (CIR), ownership and compiled execution
- [.agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md](COMPOSITION_AND_EXTENSION_CONTRACTS.md) — 323 lines; zerogravity — v4.1 typed composition and region execution contracts
- [.agents/docs/COST_MODEL_AND_AUTOTUNING.md](COST_MODEL_AND_AUTOTUNING.md) — 219 lines; Joint cost model, online adaptation and compiler autotuning
- [.agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md) — 131 lines; Zerogravity — EXL3/QTIP and Source-Qualified Derived Quantization Contract
- [.agents/docs/DEVICE_KERNEL_AND_CODEGEN.md](DEVICE_KERNEL_AND_CODEGEN.md) — 176 lines; Device abstraction, Rust GPU kernels, code generation and backend specialization
- [.agents/docs/DISTRIBUTED_PARALLELISM.md](DISTRIBUTED_PARALLELISM.md) — 57 lines; Distributed and heterogeneous execution — v4.2 authoritative index
- [.agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) — 297 lines; zerogravity — Distributed placement, network and state execution contract (v4.2)
- [.agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md](ENGINE_CORE_EXECUTION_AND_PROTOCOL.md) — 215 lines; zerogravity — Native Engine Core, Worker Protocol, KV policies and request lifecycles
- [.agents/docs/EXAMPLE_EXECUTION_TRACES.md](EXAMPLE_EXECUTION_TRACES.md) — 159 lines; Design walkthroughs — concrete execution and fallback traces
- [.agents/docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) — 190 lines; Zerogravity — Extreme Hybrid Model Recipes, Native MTP and DFlash Source Contracts
- [.agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md](FORMATS_QUANTIZATION_AND_LOADING.md) — 138 lines; Model artifacts, quantization, tokenizer/template parity and safe loading
- [.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md](FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md) — 3430 lines; Zerogravity — comprehensive agent handoff, 17 October 2026
- [.agents/docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) — 192 lines; Zerogravity — gfx1151 ROCm Kernel Specialization, MoE Scheduling and Unified-Memory Safety Contract
- [.agents/docs/HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) — 274 lines; Zerogravity — Heterogeneous prefill and retained replay contract
- [.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md](INVARIANTS_AND_FAILURE_SEMANTICS.md) — 158 lines; Cross-cutting invariants, state lifetimes and failure semantics
- [.agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) — 233 lines; Zerogravity — Joint Concurrent Serving, Goodput and Combinatory Specialization Contract
- [.agents/docs/MEMORY_RESIDENCY_AND_IO.md](MEMORY_RESIDENCY_AND_IO.md) — 260 lines; Global memory, expert residency, storage I/O and resource admission
- [.agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) — 246 lines; Minimal execution spine v4 — state, asynchronous workers, resource ownership and hot-path performance
- [.agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) — 324 lines; zerogravity — Built-in model router, multi-model residency and serving contract (v4.3)
- [.agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md](MODEL_SEMANTICS_AND_COMPATIBILITY.md) — 192 lines; Model semantics, model adapters and compatibility qualification
- [.agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md) — 183 lines; zerogravity — modular AI execution reference architecture v4.1
- [.agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) — 239 lines; zerogravity — NPU/GPU cooperative execution and multimodal TaskProgram design
- [.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md](NUMERICS_QUALITY_AND_BENCHMARKS.md) — 301 lines; Numerical correctness, model-quality evaluation and benchmark methodology
- [.agents/docs/OPERATOR_AND_FUSION_CATALOG.md](OPERATOR_AND_FUSION_CATALOG.md) — 193 lines; Operator/region contracts, fusion patterns and model execution catalog
- [.agents/docs/PERFORMANCE_HYPOTHESES_AND_RISKS.md](PERFORMANCE_HYPOTHESES_AND_RISKS.md) — 147 lines; Architectural risks, opportunities, trade-offs and falsification criteria
- [.agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md](PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md) — 151 lines; zerogravity — Product, model loading and extension contract across the full pipeline
- [.agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md](REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md) — 366 lines; Zerogravity Real-Time Multimodal Serving, Sessions, QoS and API Contract — v5.1
- [.agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md) — 139 lines; zerogravity — Remote peer transport, state handoff and failure protocol (v4.2)
- [.agents/docs/RUNTIME_SERVING_AND_UX.md](RUNTIME_SERVING_AND_UX.md) — 149 lines; Runtime, APIs, developer experience and observability
- [.agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md](RUST_WORKSPACE_AND_ENGINEERING.md) — 233 lines; Rust workspace, interfaces, implementation practices and contributor workflow
- [.agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md](SCHEDULER_BATCHING_AND_SPECULATION.md) — 241 lines; Scheduler, continuous batching, prefix reuse and speculative execution
- [.agents/docs/SECURITY_AND_RELIABILITY.md](SECURITY_AND_RELIABILITY.md) — 174 lines; Security, reliability, resilience and safe defaults
- [.agents/docs/SEMANTIC_IR_AND_COMPILER.md](SEMANTIC_IR_AND_COMPILER.md) — 262 lines; Semantic IR, state-effect system and ahead-of-time/JIT execution planning
- [.agents/docs/STATE_CACHE_AND_TRANSACTIONS.md](STATE_CACHE_AND_TRANSACTIONS.md) — 238 lines; Session state, prefix caching, snapshots and transactional semantics
- [.agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) — 213 lines; v4 TaskProgram and compiler contracts — high throughput with a small semantic core
- [.agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md](UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md) — 252 lines; Zerogravity — Unified multimodal execution contracts (v5 design proposal)
- [.agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) — 250 lines; Zerogravity — First-Class Virtual Tensor Residency and High-Performance SSD Streaming
- [.agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md](UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md) — 305 lines; Zerogravity Universal Multimodal Execution Architecture — v5.1 Design Specification
- [.agents/docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md](VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md) — 219 lines; Zerogravity Verified Model Assemblies, Source Codecs and Legal Specialization — v5.1
- [.agents/docs/VISION_SCOPE_AND_PRINCIPLES.md](VISION_SCOPE_AND_PRINCIPLES.md) — 128 lines; Vision, product scope, constraints and engineering philosophy
- [.agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md) — 244 lines; zerogravity — One-command CLI, progressive configuration and OpenAI-compatible APIs (v4.3)
- [.agents/docs/handoff-2026-10-17/PAUSED_FILTERS_CONTEXT.md](handoff-2026-10-17/PAUSED_FILTERS_CONTEXT.md) — 106 lines; TODO — additive filters and exclusions cold design
- [.agents/docs/handoff-2026-10-17/PAUSED_XET_INDEX_CONTEXT.md](handoff-2026-10-17/PAUSED_XET_INDEX_CONTEXT.md) — 38 lines; TODO — native Xet required index reconstruction
- [.agents/docs/handoff-2026-10-17/XET_INDEX_DESIGN.md](handoff-2026-10-17/XET_INDEX_DESIGN.md) — 77 lines; TODO — native Xet required-index reconstruction design

## Implementation plans and runbooks (37 Markdown sources)

- [.agents/plans/EXPERIMENT_PROGRAM.md](../plans/EXPERIMENT_PROGRAM.md) — 379 lines; zerogravity — Project-neutral experimental program, H01–H43 (v4 reviewed)
- [.agents/plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md) — 178 lines; Large immutable source loading: bounded basic-unit proposal
- [.agents/plans/IMPLEMENTATION_ROADMAP.md](../plans/IMPLEMENTATION_ROADMAP.md) — 383 lines; Zerogravity — architecture roadmap P00–P23
- [.agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md](../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md) — 182 lines; zerogravity — Independent Stage-0 feasibility and delivery gates (v4 reviewed)
- [.agents/plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md](../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md) — 219 lines; Acquisition next unit — bounded cache first, then HF/GitHub vertical slice
- [.agents/plans/NEXT_AGENT_PROMPT_2026-10-17.md](../plans/NEXT_AGENT_PROMPT_2026-10-17.md) — 74 lines; Starting prompt — Zerogravity takeover, 17 October 2026
- [.agents/plans/NEXT_AGENT_PROMPT_2026-10-19.md](../plans/NEXT_AGENT_PROMPT_2026-10-19.md) — 93 lines; Starting prompt — Zerogravity comprehensive takeover, 19 October 2026
- [.agents/plans/README.md](../plans/README.md) — 105 lines; Plans
- [.agents/plans/RELEASE_ACCEPTANCE_GATES.md](../plans/RELEASE_ACCEPTANCE_GATES.md) — 201 lines; zerogravity — Model-neutral release, quality and composability gates (v4 reviewed)
- [.agents/plans/TODO.md](../plans/TODO.md) — 23 lines; Plan queue
- [.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) — 402 lines; zerogravity — master architecture v4.3: composable AI execution, model routing and simple user experience
- [.agents/plans/archive/history/AGENT_IMPLEMENTATION_PLAYBOOK.md](../plans/archive/history/AGENT_IMPLEMENTATION_PLAYBOOK.md) — 249 lines; Zerogravity — implementation design reference
- [.agents/plans/archive/history/CURRENT_STATUS_AND_TRACKING.md](../plans/archive/history/CURRENT_STATUS_AND_TRACKING.md) — 4 lines; Current work
- [.agents/plans/archive/history/DECISION_REGISTER.md](../plans/archive/history/DECISION_REGISTER.md) — 431 lines; Decision register — confirmed user intent versus recommended design
- [.agents/plans/archive/history/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md](../plans/archive/history/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md) — 48 lines; Zerogravity — Human-Centric Design, AI-Centric Implementation, and Open Contributions
- [.agents/plans/archive/history/REQUIREMENTS_AND_TEST_TRACEABILITY.md](../plans/archive/history/REQUIREMENTS_AND_TEST_TRACEABILITY.md) — 116 lines; Zerogravity — requirements and test design
- [.agents/plans/archive/history/accepted-baseline/TODO.md](../plans/archive/history/accepted-baseline/TODO.md) — 38 lines; Accepted implementation baseline
- [.agents/plans/archive/history/handoff-cleanup/TODO.md](../plans/archive/history/handoff-cleanup/TODO.md) — 57 lines; Simplify agent handoffs
- [.agents/plans/archive/history/historical-records/TODO.md](../plans/archive/history/historical-records/TODO.md) — 95 lines; TODO — historical records
- [.agents/plans/cpu-gpu-optimization/TODO.md](../plans/cpu-gpu-optimization/TODO.md) — 51 lines; Cpu gpu optimization
- [.agents/plans/distributed-execution/TODO.md](../plans/distributed-execution/TODO.md) — 23 lines; Distributed execution
- [.agents/plans/memory-and-streaming/README.md](../plans/memory-and-streaming/README.md) — 23 lines; Source-preserving SSD residency
- [.agents/plans/memory-and-streaming/TODO.md](../plans/memory-and-streaming/TODO.md) — 68 lines; Memory and streaming
- [.agents/plans/model-acquisition/README.md](../plans/model-acquisition/README.md) — 1416 lines; Model acquisition
- [.agents/plans/model-acquisition/TODO.md](../plans/model-acquisition/TODO.md) — 334 lines; TODO — model acquisition
- [.agents/plans/model-acquisition/archive/through-a774ded/TODO.md](../plans/model-acquisition/archive/through-a774ded/TODO.md) — 120 lines; TODO — completed model acquisition phases through a774ded
- [.agents/plans/model-acquisition/archive/through-fce1353/TODO.md](../plans/model-acquisition/archive/through-fce1353/TODO.md) — 218 lines; TODO — completed model acquisition through fce1353
- [.agents/plans/model-execution/README.md](../plans/model-execution/README.md) — 463 lines; Zerogravity: source-real model execution (CPU-first)
- [.agents/plans/model-execution/TODO.md](../plans/model-execution/TODO.md) — 365 lines; Model execution
- [.agents/plans/multimodal/TODO.md](../plans/multimodal/TODO.md) — 73 lines; Multimodal
- [.agents/plans/numerics-and-context/TODO.md](../plans/numerics-and-context/TODO.md) — 30 lines; Numerics and context
- [.agents/plans/pipeline-composition/README.md](../plans/pipeline-composition/README.md) — 694 lines; Zerogravity — one fabric, single-model and multi-model pipelines
- [.agents/plans/pipeline-composition/TODO.md](../plans/pipeline-composition/TODO.md) — 299 lines; TODO — Single-process model and pipeline composition
- [.agents/plans/platform-and-release/TODO.md](../plans/platform-and-release/TODO.md) — 39 lines; Platform and release
- [.agents/plans/serving-and-speculation/FUSED_GPU_BATCHING.md](../plans/serving-and-speculation/FUSED_GPU_BATCHING.md) — 236 lines; P12 — native fused GPU batch acceptance
- [.agents/plans/serving-and-speculation/README.md](../plans/serving-and-speculation/README.md) — 185 lines; Native serving and request scheduling
- [.agents/plans/serving-and-speculation/TODO.md](../plans/serving-and-speculation/TODO.md) — 159 lines; Serving and speculation

## Source-model/performance research (24 Markdown sources)

- [.agents/research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) — 176 lines; Zerogravity — Extreme Model Specialization Integration Roadmap and Executable Acceptance Design
- [.agents/research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md](../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md) — 39 lines; Gemma 4 family source contracts — 2026-10-16
- [.agents/research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md](../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md) — 237 lines; GPU tokenization, execution control and text decoding roadmap
- [.agents/research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md](../research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md) — 176 lines; Zerogravity — Heterogeneous prefill and retained replay acceptance
- [.agents/research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md) — 135 lines; Initial hardware targets and capability optimization
- [.agents/research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md](../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md) — 116 lines; Model pull, XDG artifact cache and pinned acquisition parity
- [.agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md](../research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md) — 156 lines; Zerogravity — Model-family research atlas and multimodal implementation roadmap
- [.agents/research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md](../research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md) — 277 lines; Zerogravity Multimodal v5.1 — Complete Implementation Roadmap, Dependency Gates and Acceptance Matrix
- [.agents/research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md](../research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md) — 167 lines; Zerogravity v5.1 Multimodal Decisions, User Requirements, Design Proposals and Open Questions
- [.agents/research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md](../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md) — 40 lines; Proposed native FP32 GEMV FMA qualification
- [.agents/research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md) — 83 lines; A1 local cache — protocol and API reference
- [.agents/research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md](../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md) — 211 lines; Qwen basics, complete reference parity and advanced decoding
- [.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md](../research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md) — 10 lines; Development, builds and releases
- [.agents/research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md](../research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md) — 98 lines; Zerogravity Specialization Prior Art — Specialist Source Atlas
- [.agents/research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) — 174 lines; Zerogravity — First-Class SSD Streaming Implementation Roadmap and Acceptance Tests
- [.agents/research/STRIX_HALO_OPTIMIZATION_DOSSIER.md](../research/STRIX_HALO_OPTIMIZATION_DOSSIER.md) — 158 lines; AMD Strix Halo dossier — gfx1151 extreme inference targets
- [.agents/research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md) — 154 lines; Zerogravity — vLLM Parity and Combinatory Serving Acceptance Program
- [.agents/research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md](../research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md) — 160 lines; Research Track B — ASR, Speech Translation and Streaming Speech Understanding
- [.agents/research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md](../research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md) — 157 lines; Research Track H — Cross-Stream Adversarial Reconciliation and Minimal Architecture Delta
- [.agents/research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md](../research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md) — 192 lines; Research Track F — Typed Decision Models, Ternary Formats and Specialist Text Architectures
- [.agents/research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md](../research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md) — 198 lines; Research Track D — Image Generation, Editing, Flow/Diffusion and Autoregressive Image Models
- [.agents/research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md](../research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md) — 192 lines; Research Track G — Real-Time and Multimodal Serving / API Prior Art and Design Reconciliation
- [.agents/research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md](../research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md) — 206 lines; Research Track C — Text-to-Speech, Neural Audio Codecs, Voice and Full-Duplex Conversation
- [.agents/research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md](../research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md) — 209 lines; Research Track E — Video, Temporal Diffusion, Synchronized Audio-Video and Frontier Providers

## Prior-art, critical review and licensing (12 Markdown sources)

- [.agents/prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md) — 98 lines; Accelerator execution prior art — Blackwell, TPU, wafer-scale, static dataflow and specialized runtimes
- [.agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md](../prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md) — 228 lines; zerogravity — Reconciled architecture and authority matrix
- [.agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md) — 142 lines; zerogravity — v4 critical architectural review, challenges and resolution record
- [.agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md](../prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md) — 311 lines; zerogravity — Comprehensive cross-stack research and design selection
- [.agents/prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md) — 62 lines; Distributed execution, state disaggregation and tunnel transport: prior-art research
- [.agents/prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md](../prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md) — 104 lines; Zerogravity — Final Architecture, Editorial and Agent-Handoff Review
- [.agents/prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md) — 155 lines; Gufo XDNA2 cooperative-prefill fork and October 2026 Strix Halo report — primary-source research
- [.agents/prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md) — 78 lines; Multi-model local runtime, router, CLI and API prior-art research
- [.agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md) — 122 lines; Prior art, provenance and architecture neutrality
- [.agents/prior-art/RESEARCH_SOURCE_ATLAS.md](../prior-art/RESEARCH_SOURCE_ATLAS.md) — 125 lines; Source atlas — architectural research preserved for implementers
- [.agents/prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md) — 151 lines; SSD Streaming Research Dossier — DS4/DwarfStar and related offload prior art
- [.agents/prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md](../prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md) — 135 lines; vLLM Concurrency Prior Art and Zerogravity Architecture Reconciliation

## Agent-run recovery notes (16 Markdown sources)

- [.agents/runs/gpu-serving-batching-2026-10-19/TODO.md](../runs/gpu-serving-batching-2026-10-19/TODO.md) — 29 lines; GPU OpenAI serving and batch-wave foundation
- [.agents/runs/model-acquisition-c3-resume/TODO.md](../runs/model-acquisition-c3-resume/TODO.md) — 47 lines; TODO — resumed C3 filters integration
- [.agents/runs/model-acquisition-filters/TODO.md](../runs/model-acquisition-filters/TODO.md) — 88 lines; TODO — additive filters and exclusions cold design
- [.agents/runs/model-acquisition-v1-completion/TODO.md](../runs/model-acquisition-v1-completion/TODO.md) — 18 lines; Model pull v1 source closure cutover
- [.agents/runs/model-acquisition-xet-index/TODO.md](../runs/model-acquisition-xet-index/TODO.md) — 38 lines; TODO — native Xet required index reconstruction
- [.agents/runs/model-acquisition-xet-index-design/TODO.md](../runs/model-acquisition-xet-index-design/TODO.md) — 77 lines; TODO — native Xet required-index reconstruction design
- [.agents/runs/model-execution-full-forward-2026-10-18/TODO.md](../runs/model-execution-full-forward-2026-10-18/TODO.md) — 22 lines; ME-P0 — original Qwen full-forward FP32 diagnostic
- [.agents/runs/model-execution-qwen38-arithmetic-2026-10-18/TODO.md](../runs/model-execution-qwen38-arithmetic-2026-10-18/TODO.md) — 32 lines; Qwen original analytic numerical layer — current agent
- [.agents/runs/model-execution-qwen38-bf16-2026-10-18/TODO.md](../runs/model-execution-qwen38-bf16-2026-10-18/TODO.md) — 20 lines; ME-P0 original Qwen BF16 source dtype diagnostics
- [.agents/runs/model-execution-qwen38-reference-2026-10-18/TODO.md](../runs/model-execution-qwen38-reference-2026-10-18/TODO.md) — 22 lines; Qwen3.8 reference gate — current implementation
- [.agents/runs/model-execution-qwen38-statebridge-2026-10-18/TODO.md](../runs/model-execution-qwen38-statebridge-2026-10-18/TODO.md) — 26 lines; Qwen original source-bound StateRoot diagnostic — agent run
- [.agents/runs/p12-q8-fusion-2026-10-19/TODO.md](../runs/p12-q8-fusion-2026-10-19/TODO.md) — 21 lines; P12-B2 source GGUF quantized batch projection
- [.agents/runs/p12-resident-cohort-2026-10-19/TODO.md](../runs/p12-resident-cohort-2026-10-19/TODO.md) — 25 lines; P12-A1 resident GPU state cohort — current run
- [.agents/runs/qwen-original-bind-2026-10-18/TODO.md](../runs/qwen-original-bind-2026-10-18/TODO.md) — 26 lines; Original Qwen source binding — run checklist
- [.agents/runs/retired-side-worktrees-2026-10-18/zerogravity-filters-work/.agents/runs/model-acquisition-filters/TODO.md](../runs/retired-side-worktrees-2026-10-18/zerogravity-filters-work/.agents/runs/model-acquisition-filters/TODO.md) — 106 lines; TODO — additive filters and exclusions cold design
- [.agents/runs/serving-ssd-2026-10-19/TODO.md](../runs/serving-ssd-2026-10-19/TODO.md) — 33 lines; Native serving/SSD foundations — current run

### Cross-document notes

- The full original October 17 handoff appears above as Appendix A and remains at its canonical standalone path.
- Current source/tests/physical receipts establish implementation; design documents specify future work. Avoid interpreting the original master-roadmap P12 (sparse expert work) as identical to the *current serving-plan P12* fused GPU acceptance label; use the specific plan path and gate.
- Some older historical documents link to retired evidence/status files. The complete retained decision/recovery body survives in Appendix A; a retired path is not authorization to reconstruct or modify it.
- Dated run TODOs are local resume metadata, not a mandatory persistent evidence workflow. Raw device logs are normally ignored and may not travel with a Git checkout.


---

# Appendix D — Complete copy-ready 19 October next-agent starting prompt

Canonical standalone prompt: [.agents/plans/NEXT_AGENT_PROMPT_2026-10-19.md](../plans/NEXT_AGENT_PROMPT_2026-10-19.md).

# Starting prompt — Zerogravity comprehensive takeover, 19 October 2026

Copy the text below into the next agent. The active task is the P12 batch-native resident GPU forward implementation; the handoff also preserves the complete original architecture and all later model/SSD/serving priorities. This is an implementation handoff, NOT the earlier 17 October documentation-only pause.

~~~text
You are taking over the EXISTING live Zerogravity Rust repository at:

  /home/pvl/spaces/abird/src/zerogravity

Use the connected @dotlink-l5 local machine connector; specify that exact directory in every tool call. This is an established, source-verified engine, not a greenfield project and not an invitation to rewrite it.

FIRST: Read the NEW complete handoff progressively, without truncating its context:

  .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md

Read its up-to-date first 16 sections, Appendix A containing the complete previous 3,430-line October 17 historical handoff and decision/recovery register, Appendix B containing live-plan TODO snapshots, and Appendix C containing the exhaustive docs/plans/research/prior-art reading catalog. The embedded October 17 pause and draft statuses are HISTORICAL and were superseded by the user's explicit implementation continuation. Do not resume older unmerged worktrees or the C3/C4 acquisition drafts.

Then read current:
  AGENTS.md
  .agents/plans/README.md
  .agents/plans/TODO.md
  .agents/plans/serving-and-speculation/{README,TODO,FUSED_GPU_BATCHING}.md
  .agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md
  .agents/research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md
  .agents/plans/IMPLEMENTATION_ROADMAP.md
  .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md

Consult other canonical specs listed in the handoff when their ownership is relevant. These source files and active TODOs outrank a dated handoff.

REVALIDATE BEFORE ANY CHANGE:
  git status --short --branch
  git log -12 --oneline
  git worktree list
  git diff --check
  ps for Cargo/rustc/active GPU-serving processes
  relevant source ownership and configured device/model/capacity
  active agent-owned changes/branches/logs; do not overwrite them.

At handoff creation, the authoritative signed HEAD was
  3ec431a2df62bbcae5091a048f285801f151da7e
  feat(gpu): qualify native batched GGUF quantized projections

MAIN was clean and nine commits ahead of local origin/main, with no other worktree. It may have moved; current Git wins. The root README is user-authored: DO NOT EDIT it.

OWNER'S CURRENT PRIORITIES, IN ORDER:
1. Genuine vLLM-equivalent fused continuous GPU batching and physically verified OpenAI-compatible serving, then actual source-matched vLLM goodput benchmark.
2. Source-preserving SSD/GPU/RAM tensor residency integrated into the SAME engine, with real GPU readiness/lease and trained resident-vs-SSD StateRoot/logit proof.
3. Original Qwen3.8 27B CPU/BF16/FLA model source, trained numerical and large-source/SSD GPU execution.
4. Gemma4 E2B, then 26B-A4B and dense 31B, each separately source/numerically qualified; Flash-Next and all other multimodal/distributed/speculative work remain on their own plans.

WHAT HAS ALREADY PASSED (DO NOT REIMPLEMENT):
- Original real Qwen3.5 0.8B/2B CPU GGUF and 512-consumed-token native resident GPU bounded profiles, source/tokenizer/greedy/state/cancel/quarantine and full-vocab/all42-domain parity.
- Real OpenAI local CPU HTTP and resident CUDA GPU serving, multiple parallel sessions/alias-sharing, strict greedy-only JSON/SSE, queues, cancellation and resource admission. HTTP state-cohort publication is now verified for compatible one-token final prefill; the GPU model graphs still execute serially per request.
- Native B=2/4/8 GPU CUDA fused isolated F32 and GGUF Q4_0/Q4_1/Q8_0/Q6_K projection operators, batched elementwise/norm/GDN/KV/attention, source-exact versus B1.
- Real CUDA Qwen3.5 0.8B B2/B4/B8 ResidentBatchRows and PreparedResidentCohort: exact 248,320 logits and all42 state domains PER independent session, source/Ready/epoch/root/state arbitration, aborted/stale/foreign rollback and reclaimed leases. Full model GPU kernel fusion is NOT yet implemented; HTTP must keep gpu_fused_batches=0, and batch_mode=fused must refuse.
- Bounded checksum-verified immutable SSD source chunk/window IO and leased staging; no trained model streamed from SSD.
- YAML model/pipeline compiler/context filters/grants/decision adapter scaffolds, original Qwen3.8 BF16 mathematical diagnostics and pinned reference-trace comparator. No trained 27B or Gemma4 inference.

THE IMMEDIATE TASK: Implement the next physically qualified P12 slice, not another documentation or mock-only phase.

1. Inspect exact current source:
   crates/zerogravity-core/src/accelerator/program.rs
   crates/zerogravity-core/src/session_root.rs
   crates/zerogravity-gpu/src/resident_program.rs
   crates/zerogravity-gpu/src/resident_inference.rs
   crates/zerogravity-gpu/src/resident_batch.rs
   crates/zerogravity-gpu/src/batch_{projection,quant_projection,elementwise,recurrent,kv,attention}.rs
   crates/zerogravity-cli/src/serve/{gpu_engine,gpu_batch,http,engine,mod}.rs
   tools/serve-bench.py, tools/serve-compare.py

2. Introduce the smallest source-qualified B>1 batch-native ForwardPlan/ResidentProgram execution slice that actually CALLS the accepted multi-row GPU kernels for compatible same-source model lanes. Preserve the immutable weight source once, independent per-row prior/candidate KV/GDN, masks and causal frontiers, original SessionAuthority, ResourceArbiter, exact BufferLease/Ready/fence/quarantine, and distinct per-request output/greedy. DO NOT fork the Qwen token loop or invent another StateRoot issuer.

3. Qualify original trained Qwen3.5 0.8B physical CUDA B2 full-model complete logits/all42 domains against the accepted CPU/B1 references, one real fused multi-row kernel per appropriate graph stage, unequal prefixes, EOS, cancellation/stale epoch/foreign row and physical completion. Expand to B4/B8, Qwen3.5 2B and qualified quant codecs only after their smaller source/physical gates pass. Report actual kernel-launch counts, not grouped scheduler waves or model forward API calls.

4. Keep HTTP batch_mode=fused disabled until the complete B>1 resident model graph has passed all advertised-backend source/state/cancellation gates. Later integrate source-safe paged/hybrid KV and finite fair continuous batching; compare to true vLLM V1 on same weights/quant/tokenizer/hardware under matched cold/warm offered and committed load. tools/serve-compare.py is only a harness; vLLM was NOT installed or running on this host at handoff. Do not create fake comparison results.

5. Run focused tests, then the complete isolated gate:
   CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh

   Then source-appropriate strict all-target Clippy -D warnings and ACTUAL physical GPU original-model tests if needed. Run the separate approved Qwen3.5 .8B/2B original cache/direct AND HF-import release parity routes from handoff section 12. Verify full-vocabulary logits/all42 domains and status/exit receipts; a compiler pass is not a device pass.

6. Observe approval boundaries: no model/SDK downloads, held Renoir/gfx90c AMD launches or resets, external host privilege changes, destructive Git operations, root README edits, push/tag/release/deploy unless separately authorized. CUDA runtime/CCCL/NVRTC are present on the qualified machine via already-installed Nix packages; set CUDA_PATH correctly before physical CUDA, not by acquiring new SDKs. Avoid unqualified CUDA/Vulkan fallback. The separate offline Nix source-integrity derivation has been blocked by an absent pinned nixpkgs cache/SSL trust, while Nix formatting passes; report this accurately.

7. Keep the plan workflow simple:
   .agents/plans/TODO.md = one live queue;
   .agents/plans/serving-and-speculation/TODO.md = current P0 owner;
   .agents/runs/<phase>-<agent>/TODO.md = work in progress.
   Commit only reviewed implemented source/tests/docs with signed commits and verify the exact staged paths. Do not push. Record genuinely remaining work as unchecked items, leaving already-completed items under their signed commits.

Use independent subagents for focused review, implementation and testing where available, with disjoint file scopes and one MAIN Cargo owner; otherwise do the work with the available tools. Make progress autonomously. On an interruption, inspect actual live tree and test logs and continue; never rewrite already-qualified work or claim a phase finished from partial results.

Report succinctly: current verified HEAD, physically passing acceptance, exact unqualified fusion/vLLM/SSD/model gates, test exits, changed paths and next execution step. Do not confuse multi-root atomic publication with native multi-row model fusion.
~~~

````

## Source: .agents/docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md

Original UTF-8 bytes: 20317; SHA256 `609d9da39358304ea5d9a501ea39ba0b780d15a58ad2baad8cffdedc5c8e5268`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — gfx1151 ROCm Kernel Specialization, MoE Scheduling and Unified-Memory Safety Contract

**2026-10-15 • Hardware-specific DESIGN REFINEMENT under v4.3 generic DeviceBackend/KIR, not a separate AMD inference engine.**
**Source:** published gfx1151 port notes, Qwen serving recipes, kernel notes, first-image autotune observations, and MiMo coherent-memory experience.

## 0. Governing invariants

Device ISA or APUs never change ModelRecipe/source semantics, accepted StateRoot or source tensor identity. Specialized kernels are guarded candidates in the shared ExecutableBundle, with CPU/source oracle, numerical policy and physical event/lease proof. Missing ISA/shape/numeric capability must select a **qualified** generic candidate or refuse honestly. All flags below from the prior-art study are study references, NOT Zerogravity flags or universal defaults.

## 1. Target hardware and capabilities

AMD Strix Halo Ryzen AI Max+ 395 / Radeon 8060S iGPU is gfx1151 RDNA3.5, wave32 kernel specialization, **40 compute units** and unified host-device LPDDR memory. The prior 20-CU statement was incorrect; [AMD’s hardware specification](https://rocm.docs.amd.com/en/docs-10.0.0/reference/gpu-specs.html) lists 40 CUs. Hardware also supports wave64; each kernel must declare its actual wave mode. The prior-art physical source tested gfx1151 with ROCm7 and specialized WMMA. Nvidia CUDA, gfx12 RDNA4, CDNA matrix cores and non-AMD NPU have different ISA/warp/register and memory contracts.

A fully qualified device receipt must identify actual PCI/driver/firmware/compiler, wave size, memory topology, supported WMMA/MFMA/DOT/SAD instructions, layout and numerical reduction capabilities, LDS bank geometry, VGPR allocation/occupancy, synchronization and physical event behavior. A CUDA-like mnemonic compiled to HIP is not evidence of equivalent instruction support or source output.

### Kernel candidate dimensions
- task regime: decode M1, verify M2–16/32, prefill M64..8K, short vision encoder and long AV;
- source tensor codebook/bit width, scale and value dtype, fused activation quant;
- logical dimensions M/N/K vs physical padded shapes, alignment and block pack;
- wave/warp/occupancy, LDS tile, subgroup precision/reduction and scratch;
- device ISA+driver+compiler and runtime graph capture/eager mode;
- co-resident expert rows, output format, source state snapshots and physical buffer lease;
- verified source equivalence/quality and measured whole-workload performance.

## 2. Wave32 WMMA fragment layout and LDS bank conflict

The prior-art source ported an RDNA3.5 WMMA GEMV for EXL3 decode, with source-oracle fragment layout, rather than using the fallback reconstruct+FP16 GEMM. A classic CUDA cp.async sequence is not available as-is on this target, so their INT8 activation GEMV uses register-staged B loads and AMD v_dot4_i32_iu8 path.

**Important measured optimization:** its half2/shared LDS staging array had physical innermost extent 4, causing repeated 32-bank LDS collisions. Padding that physical stride to 5 (coprime with 32) reduced conflicts, with reported ~10% overall mean decode gain at a particular pipeline stage. This is **a layout-tuning hypothesis scoped to hardware and kernel**, not a new semantic operator or unconditional global padding rule.

Correct specialization:
1. emit source logical tensor layout independent of LDS physical swizzle/padding;
2. choose an ISA-target-specific fragment map with verified lane→matrix element and grouped row indexing;
3. treat packed shared-memory stride as a compiler physical parameter, candidate 4/5/safe alternatives;
4. validate barriers, wave count and no out-of-bounds for ALL callers using the shared scratch, not only MoE (the prior-art source warns a 4-warp scratch used by 16-warp GEMV silently produced NaN logits);
5. run source tensor output golden vectors including ragged M/N, smallest and largest codebook K, zero/nonfinite values, padded rows and multiple model layers;
6. measure LDS bank conflicts, VGPR/spills, occupancy/byte throughput and **whole model decode with MTP**, not only GPU instruction counts.

A broken scratch layout may keep aggregate PPL apparently stable while corrupting individual logits. Must include full tensor/source validation and state/greedy prefix, not solely perplexity.

## 3. Dense/GDN/MLA/vision kernels: source-first closed regions

Hardware candidate list, promoted only after legal semantic guard:
- decode quant GEMV directly from packed low-bit codebook with WMMA wave32;
- prefill dense grouped GEMM and matrix tiles;
- **skinny GEMM** for GDN input projection b/a or other unusually narrow matrices where hipBLASLt split-K overhead dominates;
- fused GDN/KDA short convolution→gate/recurrence update, preserving causal state, numerical order and rollback;
- MLA/DSA compressed latent Q/K selection and sparse attention masked/position-correct;
- mHC MixNorm/residual chain fused only when stream stack/result numeric equivalence demonstrated;
- grouped MoE router, expert compact assignment, packed GEMV/GEMM, post-expert reduction and shared expert;
- source E8M0 scale reader and expert FP16 intermediate range guard (MiMo-specific);
- optional physical INT8 activation GEMV, with v_dot4 variant only where legal and measured;
- architecture-specific bitcodebook SAD trellis decode, no output bit reordering;
- image/video vision patch merge convolution lowered to logically identical GEMM if supported.

**Do not** automatically enable all fused kernels. The prior-art source reports CPU offloaded routed experts on one gfx1151 UMA test **slowed down** decode relative to GPU; A-ring prefetched register/GEMV variants and graph replay also provided null/negative improvements in given cases. Keep one simple source-exact baseline and only promote repeated whole-task wins.

## 4. MoE row-packing, verification and multi-session scheduling

A row count is a *physical kernel dimension* independent of logical token count, because each draft block generates multiple verifier rows. For B concurrent sequences with per-session proposed depths d_i:

R_total = sum_i (1 + d_i).

Actual MoE expert work depends on:
- unique expert IDs across all R rows and models, routes per row and stable compact worklist;
- expert-row histogram, uneven hot/cold placement, shared-expert gates and reduction order;
- logical rows vs padded per-expert tiles, source quant/codebook and kernel specific physical group size G;
- expert weight read bytes, warm/OS/SSD residency and host staging, physical memory bandwidth;
- verification acceptance by prompt/content class, target state branch COW and draft cost;
- source-compatible numerator of **committed** accepted tokens rather than proposed drafts;
- p95 tail and wait fairness, priority and deadline, guard memory and compiled variant cost.

Where a physical kernel rounds work to G-row chunks, padded cost resembles G*ceil(R/G) only as a rough bound: real grouped expert work can have many more per-expert partial tiles. G is recorded with a **KernelCandidate** and device profile, not a constant in TaskDriver/SIR.

### Prior-art measured counterexample (source, not Zerogravity result)

On a particular gfx1151 Qwen EXL3 pack the authors measured a single-session verify forward T_ms approximately 32.4 + 9.04R. At speculative depth 2, five concurrent requests produce 15 verifier rows, near one 16-row chunk. Six produce 18 rows, spilling into a second mostly-empty chunk and reported a strong total-throughput decline. Larger verifier width also reads *more distinct routed experts* and can reduce acceptance; a 16-row speculative tree was counterproductive under measured marginal row costs.

The source notes ~47 tok/s single session on a favorable prompt, versus ~74–75 tok/s aggregate for a realistic deeper-queued B5 batch in one configuration. **These are different objectives**. User-facing interactive latency and aggregate throughput need separate settings and reported metrics. The general scheduler should select one of a small bounded set of source-qualified batch/depth/shape candidates under active state and P95 fairness, not run graph search or retune every token.

### Shape-aware batch cost (conceptual)
- observed GPU launch + per-step fixed cost;
- expert unique-row bytes + packed codebook work;
- quant GEMV bandwidth and LDS spills;
- GPU state update/verification per row and selected attention indexer;
- queue delay and input/output physical DMA, speculative correction and held resources;
- valid accepted tokens forecast and fallback hysteresis.

An unexpected batch trough should cause candidate reselection after measured evidence and cooldown, not a silent change of quant/attention or unlimited queue depth.

## 5. Unified-memory admission and Linux page cache

On coherent-memory systems, GPU allocation, CPU heap, OS page cache, host pinned staging and GPU buffers compete for a finite physical pool. **Memory mapped/addressable GPU-visible bytes, driver free GTT, MemFree and MemAvailable are different quantities.** A large model can fit logically yet exhaust actual RAM/pinned pages or stall the driver during loading.

Prior-art MiMo notes report a near-fatal case on GB10 where buffered pread of about 86GiB model shards left an approximately equal-size clean file page cache while the GPU also held weights. MemFree-like driver reported free memory did not capture reclaimable cache properly, and the system entered severe reclaim thrash. Their opt-in external guard checked MemAvailable, reclaimed only known shard page cache via fadvise(DONTNEED), and killed the isolated serving process if a configured physical headroom floor was crossed. **Do not copy that kill policy as a default Zerogravity runtime behavior**; it is a valuable adversarial failure mode and user-supervised deployment consideration.

### Authoritative resource design adjustments
1. Keep **one PhysicalResourceArbiter** for physical card and model/session envelopes, but admission also checks appropriate **host/OS-safety** envelopes on UMA. Do not make OS file cache a second GPU owner.
2. Distinguish mandatory/evictable resources: packed target weights, draft, vision, recurrent/KV state, page-cache mapped shards, clean file cache, pinned DMA, local scratch, OS health reserve and unrelated system workloads.
3. At startup check **platform-aware** headroom: Linux MemAvailable and cgroup memory.current/memory.max/memory.stat where available, driver committed/resident and known pinned pages, memory pressure indicators/PSI. Driver-reported GTT maximum is NOT guaranteed free physical bytes. Other OSs use native APIs.
4. Estimate and reserve source+codec+model+draft+state+transfer+read-cache footprint before load. During multi-shard model loading, budget peak staging and likely reclaimable clean page-cache duplicates.
5. Use bounded streaming read or mmap with source-file-descriptor lifetime, version/hash validation and controlled prefetch; do not eagerly buffer 100GiB source *again* on CPU by accident. A host page-cache hint is an optimization, not permission to discard dirty files or another process's cache.
6. Optional, **user-authorized** clean-shard DONTNEED on only the manifest-verified model files after their bytes have been safely copied/retired; never blindly evict global page cache or require root.
7. On predicted headroom miss, reject/defers **before** model GPU allocations; on observed pressure suspend model loads, release **safe idle** caches and return typed admission error. No hidden SIGKILL or OOM-based throttling inside EngineCore.
8. A separate supervised daemon policy might terminate a disposable isolated model worker on impending system hang **only if explicitly configured** and after it preserves no false StateRoot receipt. This is a deployment choice, not a library inference default.
9. Keep physical GPU device buffer quarantine on unknown submitted work, and do not treat OS page cache freeing as proof GPU command quiescence.
10. Telemetry must report physical OS headroom, driver commit, page cache bytes, old memory pressure, true live StateRoot leases and per-session rings; never conflate logical source budget release with driver pool eviction.

### Atomic multi-model loading and cache safety
- Model aliases and exact source identity can share immutable weights; per-session StateRoot and mutable drafts/KV are independent.
- Concurrent loads use bounded per-source singleflight to avoid two simultaneous copies and cache thundering herd.
- A ModelCatalog can contain dozens of registered large models while only a few fit resident; TTL/pin/revoke and warm/cold resource receipts are distinct.
- No auto-up/downquantize to fit in 128GB; source/derived artifact choice is an explicit model binding.
- Optional systemd deployment controls ResourceMaxMemory/MemoryHigh and capabilities, but can cause driver/process pressure; verify expected behavior with actual Linux GPU.

## 6. Cold-start and first-image specialization

The prior-art source found GLM5.3 first vision image/resolution sometimes spent **many seconds/minutes in MIOpen convolution solver and dense GEMM autotune search**, triggered anew by subtly different input shape, while steady-state math was fast. Their source has a verified patch-merge convolution with kernel=stride=2 that can become a flattened GEMM with identical contraction and no costly first-use convolution tuning.

Recommended cost/tuning lifecycle:
- **install/compile:** emit known-good reference kernels and bounded guarded variants for shape bands/registered encoders;
- **model load:** compile/prime only finite representative shapes if budget permits, allow eager direct GEMM patch merge for legally verified shapes;
- **request:** choose guarded prequalified kernel cheaply; unfamiliar shapes execute generic known-good path or bounded safe limited compile, **never indefinite live solver search**;
- **offline/autotune:** evaluate alternatives asynchronously in a low-priority, resource-capped window, persist evidence cache keyed by model+shape+device/compiler/numeric mode;
- **future request:** reuse persisted validated cache; event/current driver version and shape must match.

Vision-source equivalence requires exact convolution stride, padding, dilation, groups, channel and image patch token order. A shape-only transformation is invalid if its source convolution has overlapping windows or nonlinear postprocessing. The generic slow path remains available and truthful.

No startup ETA or prefill speed should be invented without measured load history. Progress events may report deterministic counts of uploaded tensor bytes, backend compile stages and applied cache state.

## 7. Qualifying negative results and overhead ceilings

Take seriously the documented **non-wins**:
- CPU MoE expert offload on this UMA target decreased decode speed because shared LPDDR contention/transfer overhead dominated any CPU compute gain.
- Some INT8 activation GEMV paths performed worse than FP16 WMMA; activation encode and quant scale overhead matters.
- Graph replay was neutral at tested MTP workload; saved host launches were a small fraction of total wall time.
- Some candidate prefetch/register rings increased VGPR pressure and did not help.
- Uniform lower bitrate (2.05) performed much worse than 3.05 due to trellis K2 kernel tile throughput.
- Static draft depth 2 beat dynamic on one favorable prompt, but multi-prompt averages preferred default dynamic policy; never promote one cherry-picked prompt.

Experiments therefore require:
- repeated ABBA/BAAB paired runs on warmed and cold model, temperature/power clock stabilized, narrow confidence intervals and a control without change;
- source model/derived quant exactly pinned and same prompt/task/sampling/accepted token policy;
- kernel and end-to-end instrumentation separated (not every request measured under expensive profiler);
- no memory pressure/driver hang tests on an interactive system without explicit isolation and operator approval;
- measure target-only, MTP, DFlash, 1–8 sequences, expert entropy, long context and image shape changes;
- negative/neutral candidates remain in evidence database so later agents don't rediscover them.

## 8. Acceptance gates for P01/P07/P10/P12/P16/P20/P22

| ID | Scenario | Mandatory proof |
|---|---|---|
| GFX-01 | gfx1151 wave32 WMMA fragment/lane layout | scalar source/CPU oracle, full tile bounds and original model output |
| GFX-02 | LDS stride 4 vs 5, full shared-warp users | no bank-related regression, valid 16-warp caller, no NaN logits |
| GFX-03 | direct EXL3 K2/K3/K4 packed GEMV at M1,3,4 and 8/16/32 | measured kernel/full-model speed and exact derived-quant oracle |
| GFX-04 | int8 DOT4 activation against FP16 default | codebook/activation scale, source numerics, full wall benefit or reject |
| GFX-05 | GDN skinny GEMM + short conv fuse, QSA/DSA source attention | recurrence/sparse indexer/StateRoot source invariants |
| GFX-06 | mHC/gated residual fused region and source intermediate | source full logits, overflow/range and aligned stream state |
| GFX-07 | MoE batch five→six under 16-row candidate | measure shape cliff, no hard-coded G=16 in SIR |
| GFX-08 | B and draft width with different expert uniqueness | dynamic candidate selects measured best, no P95 starvation |
| GFX-09 | model + draft + 256K KV/SWA ring + 100GiB input on UMA | OS MemAvailable/cgroup/cache/GPU pool headroom, fails before hang |
| GFX-10 | load 80–110GiB shard files, intentionally warm page cache | safe source-backed fadvise policy opt-in; no global cache eviction |
| GFX-11 | two models concurrent loads and one session disconnect | correct singleflight model/StateRoot/quarantine, no memory double charge |
| GFX-12 | source-equivalent patch-conv lowering for first new image shape | no multi-minute tuner search; numerical/shape/stride parity |
| GFX-13 | cold/warm GEMM tune/shape cache | bounded startup, drivers invalidation, no JIT per audio/media frame |
| GFX-14 | graph capture on/off, CPU MoE offload, int8, warp layout variants | full-model gain or negative receipt retained, source quality |
| GFX-15 | physical GPU driver loss or canceled submitted Future | retain actual buffer handles/envelopes and refuse false recovery |
| GFX-16 | power/thermal and shared LPDDR interference | comparable held-clock ABBA and per-model P95/energy |
| GFX-17 | unsupported wave/ISA or component/source codec | guarded slow reference or typed Unsupported; no false GPU |
| GFX-18 | multiple active live speech/video plus extreme text MoE | bounded QoS, coherent-memory pressure and no hidden numerical degradation |

## 9. Existing roadmap mapping

- P01 and P07: ISA/fragment/WMMA/DOT4/LDS/skinny GEMM candidates with real hardware source proof.
- P05 and P20: compiler guarded shape buckets, no first-use unbounded search; empirical auto-tune database.
- P10: physical UMA, page-cache/allocator distinction and guarded large-model admission.
- P11–P14: source sparse attention, GDN/KDA/mHC, grouped MoE and variants.
- P15/P16: dynamic target verifier row count and expert-aware batch/route fairness.
- P22: cost-quality-matched original checkpoints, the source as external prior art, true reproducible target-specific performance.

**Architecture remains one Rust-owned semantic engine and physical resource authority.** No new AMD-specific executor, no required Python inference runner, no uncontrolled system watchdog and no default change to driver or OS configuration.

## U-048 — combined kernel, replay and NPU qualification

The [heterogeneous pipeline contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) requires shape/ISA fusions, replay transport
and cooperative banks to qualify separately and as the intended composition.
LDS slot reuse must retire all prior reads, including asynchronous loads, under
independent GPU co-tenants; no single-process benchmark can establish that. Native
Rust ISA/AIE code paths may avoid a full SDK at runtime but still need source,
program, ABI and physical device qualification. Charge mapped/pinned arenas,
resident panels, live prepared plans and shared LPDDR once; performance calibration
cannot substitute for SoC safety, exact numerical mode or kernel barrier proof.

````

