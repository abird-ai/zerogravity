# Zerogravity — requirements and test design

This is a reference mapping of requirements and intended tests. Implementation
status and actual results live in [the plan TODO index](TODO.md) and its plans.

## 1. Explicit user intent U-001…U-026

| User ID | Requirement interpreted without overclaim | Authoritative design | Primary acceptance/work |
|---|---|---|---|
| **U-001** | Project name and independent Rust-owned identity | [Whitepaper](../../docs/WHITEPAPER.md), [Engineering Spec ES-00](../../docs/ENGINEERING_SPECIFICATION.md) | P00 project metadata; do not rename silently |
| **U-002** | Maintain complete agent-readable plans and decisions | [Agent Playbook](AGENT_IMPLEMENTATION_PLAYBOOK.md), this matrix, [Index](../../docs/README.md) | Docs integrity/manifest, complete phase handoff; path superseded by U-013 |
| **U-003** | Rust runtime + owned kernel source | [ES-01/04/05](../../docs/ENGINEERING_SPECIFICATION.md), [Device codegen](../docs/DEVICE_KERNEL_AND_CODEGEN.md) | P00–P01 CPU build, GPU kernel source/provenance; vendor FFI O-003 remains open |
| **U-004** | Broad known model-family support | [ES-02](../../docs/ENGINEERING_SPECIFICATION.md), [Model Semantics](../docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md) | P02–P03/P17 exact registered ops and model source vectors |
| **U-005** | Extreme sparse/hybrid large-model support | [ES-02/04/08](../../docs/ENGINEERING_SPECIFICATION.md), [Memory/IO](../docs/MEMORY_RESIDENCY_AND_IO.md) | P11–P14 model/reference/sparse/expert qualification on real feasible machines |
| **U-006** | High performance alongside simple usability | [Whitepaper §3/9/11](../../docs/WHITEPAPER.md), [ES-09](../../docs/ENGINEERING_SPECIFICATION.md) | P07–P09/P16/P22; benchmark direct run/serve + C1/8/32 P95 |
| **U-007** | Fallback from extreme to general execution | [ES-04](../../docs/ENGINEERING_SPECIFICATION.md), [Composition](../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md) | P05/P08 forced specialized kernel disable; no new server |
| **U-008** | Learn comprehensively from external engines | [Prior Art Atlas](../prior-art/RESEARCH_SOURCE_ATLAS.md), [References](../../docs/REFERENCES.md) | Reproducible source attribution, file license review; not copied engine |
| **U-009** | Semantic IR with high-performance specialization | [ES-02/04](../../docs/ENGINEERING_SPECIFICATION.md), [Compiler](../docs/SEMANTIC_IR_AND_COMPILER.md) | P04–P05/P08/P20 two enduring representations, no hot SIR walker |
| **U-010** | Prior research-only, later authorized design/docs | [Status](CURRENT_STATUS_AND_TRACKING.md), [Playbook](AGENT_IMPLEMENTATION_PLAYBOOK.md) | No code/Git/driver/peer work without user authorization |
| **U-011** | Preserve decisions/pending items/handoffs | [Decision Register](DECISION_REGISTER.md), [plan TODO index](TODO.md) | All ID coverage and historical archive preserved |
| **U-012** | Benchmark ambition not fabricated numeric deadline | [Whitepaper §11](../../docs/WHITEPAPER.md), [Numeric quality](../docs/NUMERICS_QUALITY_AND_BENCHMARKS.md) | P22 matched hardware/model/quant/task and raw evidence |
| **U-013** | Canonical path corrected to source tree | [plan TODO index](TODO.md), [README](../../README.md) | Existing files at canonical path, no obsolete active root links |
| **U-014** | Combine research corpus with semantic architecture | [ES-00 authority](../../docs/ENGINEERING_SPECIFICATION.md), [Prior-art policy](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md) | No duplicated model runtime, research tree read-only |
| **U-015** | Adaptable at all pipeline layers and across near-future technology | [ES-14 extension matrix](../../docs/ENGINEERING_SPECIFICATION.md) | P23 change-radius drill and H43/H55/H70/H86 |
| **U-016** | Explicit GPU+NPU partial prefill and multi-task paths | [NPU/Task Pipelines](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) | Optional S0-NPU A–F, source/numeric/host-hang safety |
| **U-017** | Independent project-neutral architecture and license-aware prior art | [Prior Art License Policy](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md), [ES-10](../../docs/ENGINEERING_SPECIFICATION.md) | No non-MIT project-branded core interfaces; file-level source review |
| **U-018** | Minimal, high-performance composable architecture critical review | [v4 critical review](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md), [ES-01/14](../../docs/ENGINEERING_SPECIFICATION.md) | S0-V4 A–J, two enduring representations, one lib+bin until justified |
| **U-019** | GPU/Blackwell, graph TPU and wafer/ASIC execution family coverage | [Accelerator Provider](../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) | S0-FABRIC A–H; mock architecture vs real SDK/state-safe distinction |
| **U-020** | Multi-machine topology, separate compute/parameter/state/cache and tunnels | [ES-08](../../docs/ENGINEERING_SPECIFICATION.md), [Distributed Placement](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) | S0-DIST A–L; P19 and H56–H70 authorized real two-node validation |
| **U-021** | One-command local, built-in router, many models/profiles and OpenAI APIs | [ES-07/09](../../docs/ENGINEERING_SPECIFICATION.md), [Router](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [CLI/API](../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md) | S0-SERVE A–J, P08–P10/P16, H71–H86 tested CLI+SDK |
| **U-022** | Final coherent design review and **publishable whitepaper + full LLM-agent implementation handoff** | [Whitepaper](../../docs/WHITEPAPER.md), [Engineering Spec ES-00…15](../../docs/ENGINEERING_SPECIFICATION.md), [Public Index](../../docs/README.md), [Publication Checklist](../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md) | Design/editorial review; original handoff/SHA process superseded by [TODO workflow](../../AGENTS.md); owner public release approvals OPEN |
| **U-023** | **Packaging-only Rust binary foundation**: README, Cargo checks, Nix/crane cross build, NixOS module, GitHub Actions releases, checksum installers; no AI runtime logic | [Bootstrap Handoff](platform-and-release/TODO.md), [Root README](../../README.md), current P00 | Rust **7/7** + release CLI + shellcheck/nixfmt/Nix parse and offline installer fixture PASS; **Nix full build, Windows/macOS, module evaluation and GitHub release BLOCKED/NOT RUN** |
| **U-024** | Begin Phase 1 real inference development, targeting Qwen3.8-27B and Gemma 4 26B-A4B-it as two source-correct families | [Engineering Spec ES-02/06](../../docs/ENGINEERING_SPECIFICATION.md), [Phase 1 handoff](accepted-baseline/TODO.md) | **PARTIAL:** Rust artifact/config/quant/math/state reference + 23 tests; **BLOCKED/NOT_STARTED:** real model tensors/tokenizer/logits/generation, GPU and Nix cross builds |

| **U-025** | **Prioritize Qwen3.5 0.8B and Qwen3.5 2B: first source-correct CPU, then CUDA, ROCm and Vulkan on both GPUs; Gemma 4 E2B, Gemma 4 26B-A4B and Qwen3.8 27B CPU first, GPU after** | [Current Qwen handoff](accepted-baseline/TODO.md), [Physical GPU probe](../../crates/zerogravity-gpu/README.md) | **PARTIAL:** actual CPU text inference; independent greedy 5/5 12-token (0.8B), 5/5 8-token (2B); 28 core tests+1 opt-in checkpoint oracle; real Q4_0 source projection matches CUDA/HIP/NVIDIA Vulkan/AMD Vulkan on physical GPUs. **NOT_IMPLEMENTED:** all-device full-model GPU decoding, StateRoot/device leases and remaining CPU model families |

| **U-026** | Comprehensive independent CPU numerical/per-layer/long-context qualification of both Qwen3.5 0.8B and 2B before full CUDA/ROCm/Vulkan inference | [CPU numerical and context handoff](numerics-and-context/TODO.md) | **PARTIAL/BLOCKED:** real 1K 0.8B and 512 2B prefill reference parity; guarded private state/rollback, synthetic 32K operator tests; **2B strict numerical failure**, no full-model 4K..32K, no per-layer independent source parity, no GPU model inference |
| **U-027** | GPU-development admission after both Qwen3.5 CPU models independently qualify through 8K; defer but do not pass/drop 16K/32K local tests | [Phase 1 GPU gate and P07 integration handoff](numerics-and-context/TODO.md), [CPU gate](../../scripts/cpu-qualification-gate.py), [Accelerator region](../../crates/zerogravity-core/src/accelerator.rs), [Real model parity tests](../../crates/zerogravity/tests/qwen_real_model.rs) | **CPU Phase 1 PASSED 16/16** (both source-model 8K, short numerics, independent layer captures and state). **Extended context DEFERRED 16/20** (four 16K/32K still unqualified). **P07 STARTED**: shared Rust source-typed matvec dispatcher, explicit unsupported-only fallback and fail-closed GPU errors; real-checkpoint fallback/logit/state test PASSED on both models. Complete physical GPU inference NOT YET IMPLEMENTED. |
| **U-028** | GPU provider development and independent 8K numerical accuracy investigation, no threshold relaxation | [8K drift report](numerics-and-context/TODO.md), [source GPU Q4×Q8 provider](../../crates/zerogravity-gpu/src/q4_q8.rs) | Source Q4×Q8 host numerical parity and 3-token completed-region real-model tests pass on both models. 8K source/GDN layer divergence localized. Physical CUDA/ROCm/Vulkan provider inference NOT VERIFIED. |
| **U-029** | Three parallelizable implementation/research streams and multimodal architecture substreams, with independent agents only when authenticated | [v5 source design](../docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md), [family atlas](../research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md), [substream ownership](../research/MULTIMODAL_PARALLEL_RESEARCH_WORKSTREAMS_2026-10-14.md) | Core architecture v4.3 preserved; proposed media clocks, compound models, non-token TaskDrivers, AV StateRoots, codec transforms, decisions; actual media inference not yet implemented. Standalone Codex/Claude tools unavailable due authentication; no false subagent work claimed. |
| **U-030** | Next phase: source-correct GPU P07-B Q4_1/Q8_1, Q8_0/Q8_0, Q6_K/Q8_K and F32 matrices with host parity, multimodal MM0 fake clocks/decision/flow/AV tests, and 8K GDN reference microbatch investigation | [P07-B handoff](accepted-baseline/TODO.md), [GPU mixed providers](../../crates/zerogravity-gpu/src/mixed_ops.rs), [MM0 tests](../../crates/zerogravity/tests/multimodal_mm0.rs) | Real checkpoint host arithmetic matched bitwise for all four 2B tensors and each source projection produced identical model logits/StateRoot when simulated as Completed. MM0 5/5 fake tests pass. Independent source 8K tensors: 0.8B ubatch256 vs 512 identical 158/158, 2B 149/158 differ (first layer1 GDN ~4.9e-7); causal drift is still under investigation. Actual GPU hardware execution NOT VERIFIED. |
| **U-031** | Bounded process-unique source-matched GPU weight buffers, failed-inflight fences, no repeated immutable H2D upload and complete GPU readback before StateRoot; no new numerical relaxation | [P07-C handoff](accepted-baseline/TODO.md), [residency policy](../../crates/zerogravity-gpu/src/residency.rs), [actual source provider](../../crates/zerogravity-gpu/src/resident_provider.rs), [core source ID](../../crates/zerogravity-core/src/accelerator.rs) | Host-tested local 0.8B and 2B GGUF sources each produce a single source weight upload and two cache hits in 3-token real forward, identical model logits and StateRoot. Bound/eviction/failed-upload/fence tests pass. CubeCL Vulkan builds; PHYSICAL GPU, asynchronous TaskCoordinator and GPU KV/GDN state are still unqualified. |
| **U-032** | P07-D source-weight decode-once, typed physical device-event admissions, cancellation/epoch fencing and fail-closed completion | [P07-D handoff](accepted-baseline/TODO.md), [StaticSource](../../crates/zerogravity-gpu/src/host_descriptor.rs), [event ledger](../../crates/zerogravity-core/src/device_events.rs) | All five GPU source tensor codecs host-verified and six device-event tests pass; physical async CubeCL execution not yet qualified. |
| **U-033** | P07-E actual CubeCL async readback completion bridge tied to pre-launch event/scratch admission; cancellation Drop fencing and source-model parity | [P07-E handoff](accepted-baseline/TODO.md), [bridge guard](../../crates/zerogravity-gpu/src/async_bridge.rs), [CubeCL async provider](../../crates/zerogravity-gpu/src/resident_provider.rs) | Async readback compiles, host Future-drop/lease tests pass, real pinned 0.8B/2B model event-mock parity and fail-closed error tests pass. No physically tested GPU kernel, no concurrent multi-stream task engine. |
| **U-034** | Recover P07-E, retain source/dynamic/output GPU CubeCL handles through uncertain cancellation, fence the entire service after dropped Future, and enforce static+transient per-provider GPU capacity without false physical qualifications | [P07-F handoff](accepted-baseline/TODO.md), [physical buffer quarantine](../../crates/zerogravity-gpu/src/physical_quarantine.rs), [event/handle guard](../../crates/zerogravity-gpu/src/async_bridge.rs), [budget](../../crates/zerogravity-gpu/src/residency.rs) | Optional GPU Vulkan source builds and strict Clippy pass, source-free GPU tests 10/10 and real Qwen3.5 opt-in GGUF host tests 8/8 pass. Native CPU Phase1 16/16 and deferred 16K/32K unchanged. Physical CUDA/HIP/Vulkan cancellation and global device scheduler remain NOT VERIFIED. |
| **U-035** | Validate live P07-F committed state, continue with P07-G control-plane, create full comprehensive phase-0-through-current handoff with original designs, all decisions, pending work and next agent prompt | [P07-G control plane](accepted-baseline/TODO.md), [ResourceArbiter](../../crates/zerogravity-core/src/resource_arbiter.rs), [fair device queue](../../crates/zerogravity-core/src/device_queue.rs), [full takeover document](TODO.md) | Live 602c246 P07-F baseline quality gate passed; P07-G1 4 resource tests + 4 queue tests pass. Comprehensive full-history handoff snapshot being generated. Physical CubeCL GPU model inference and source GDN accuracy changes remain unqualified. |

**New user requests after U-026 require a new U identifier and exact scope; do not silently reinterpret older decisions.** Root U-002 relative path was superseded by U-013 and must not be revived.

## 2. Operational core requirements and integrated invariant mapping

| Core contract | Engineering sections | Primary pass test (proposed; **not a file that exists yet**) | Related phases |
|---|---|---|---|
| Semantic source validation and exact op | ES-02; INV-01/02 | `test_artifact_source_and_unknown_ops`; reference tensors/logits | P02, P03, P06, P11 |
| No silent numeric/model change | ES-02/04; INV-02/23 | `test_quant_storage_compute_vs_changed_model`; route top-k | P03, P05, P12, P22 |
| Portable direct compiled fallback | ES-04; INV-02/09/10 | `test_kernel_guard_and_forced_fallback` | P05, P07, P08 |
| Prepared one-root logical commit | ES-06; INV-03/04/05 | `test_all_domains_prepare_then_single_publish` | P02, P04, P06, P15 |
| Token/grammar/draft exactness | ES-06; INV-05/06/23 | `test_target_distribution_and_rejected_branch_state` | P08, P15 |
| Resource leases and memory safety | ES-05; INV-07/08 | `test_delayed_dma_generation_and_pool_reservation` | P05, P07, P10 |
| Minimal TaskDriver and non-token output | ES-03; INV-11/20 | `test_text_and_encoder_task_reuse_one_engine` | P04, P17, P18 |
| Distinct model/alias/replica/session | ES-01/07; INV-12/13/14 | `test_alias_revision_replica_sticky_state` | P09, P10, P16 |
| Many-model concurrent admission | ES-07; INV-08/13 | `test_cold_load_dedup_and_idle_eviction` | P09, P10, P16 |
| Trusted API adapters and SSE | ES-09; INV-06/15/22 | `test_chat_streaming_python_js_qualified_subset` | P09, P17, P18, P21 |
| Exact peer/data/trust boundaries | ES-08/10; INV-15/16/18 | `test_peer_enrollment_and_wrong_model_reject` | P19, P21 |
| Remote full-state compatibility | ES-06/08; INV-04/16/17 | `test_distributed_prepare_fence_and_migrate` | P10, P19 |
| Token-sharded attention only when valid | ES-08; INV-19 | `test_partition_softmax_mask_numerics` | P11, P19 |
| AOT/static dataflow StateBridge | ES-04/08; INV-21 | `test_graph_only_stateful_bridge_or_reject` | P01, P05, P19 |
| Future learning seam without training bloat | ES-02/03; INV-24 | `test_parameter_version_inference_isolation` | P23 |
| Zero network/service tax for local run | ES-09; INV-13/25 | `test_run_local_no_daemon_http_or_tunnel` | P06, P08, P09, P23 |
| Full physical/measured performance evidence | ES-10/13 | `bench_exact_model_task_shape_driver_quality_matrix` | P20, P22 |

These are **suggested descriptive test names**, not an instruction to create test stubs without executable subject code. Each test must record source version, software/hardware, expected data, what fails, and whether simulated or real.

## 3. Phase-to-requirement closure map

| Stable phase | Primary ES sections | Minimum closure evidence | Notes |
|---|---|---|---|
| ZG-P00 | ES-00/01 | CPU-only clean Cargo executable/test and source policy | Small workspace first |
| ZG-P01 | ES-04/05 | real owned GPU kernel + CPU oracle or BLOCKED with reason | physical compiler optional for CPU |
| ZG-P02 | ES-02/06 | exact source op and provisional state reference | no hidden effect |
| ZG-P03 | ES-02 | bounded artifact/tokenizer/quant test vectors | no unsafe file parsers |
| ZG-P04 | ES-02/03/06 | effect-aware SemanticProgram + minimal TaskDriver | no general workflow |
| ZG-P05 | ES-04/05 | legal prepared region and CIR simulated lease/event | no SIR interpreter hot |
| ZG-P06 | ES-02/03/06 | real model CPU output and sampler/state | source-correct |
| ZG-P07 | ES-04/05 | real GPU event/buffer worker and model parity | physical safety |
| ZG-P08 | ES-03/04/06/09 | **one-command local real model output** with fallback | top viability gate |
| ZG-P09 | ES-07/09 | same-engine API catalog/chat/SSE + SDK conformance | no router engine |
| ZG-P10 | ES-05/06/07 | full-state cache + physical memory/residency | no active eviction |
| ZG-P11 | ES-02/06/08 | hybrid source recurrent/attention model quality | no fake partition |
| ZG-P12 | ES-02/05 | correct expert router/tier/IO | exact top-k |
| ZG-P13 | ES-04/05 | full complex model extreme region + fallback | baseline and quality |
| ZG-P14 | ES-02/04/14 | second complex model extension without server fork | change radius |
| ZG-P15 | ES-02/06 | target-authoritative speculation/rollback | target distribution |
| ZG-P16 | ES-05/07/08 | multi-model native batch/replicas/fairness | P95 and pressure |
| ZG-P17 | ES-02/03/07/09 | adapters, embeddings/rerank source math/API | real encoder |
| ZG-P18 | ES-03/09 | non-text media task source/quality | codecs, frontiers |
| ZG-P19 | ES-04/08 | trusted remote layer/state/transport + optional fabric | actual peer auth for physical |
| ZG-P20 | ES-04/05/10 | bounded offline autotuner/cost model and negative tests | model/task-level benefit |
| ZG-P21 | ES-09/10 | parser/auth/license/privacy build/release tests | no privilege surprise |
| ZG-P22 | ES-02/09/10/13 | matched model/quant/device/task quality+performance/API | raw reports |
| ZG-P23 | ES-14 | extension drill new op/backend/task/model/transport | no engine duplication |

## 4. Suggested milestone-level review gates

**M0 — Engine correctness feasibility:** source parser/CPU reference SIR, state root protocol, fake CIR and one native kernel source proof, no large framework scaffolding. **M1 — Useful local model:** an actual checkpoint produces a correct CLI response with safe local resources and general fallback. **M2 — Shared serving:** same engine behind model router/API, full-state cache, multi-model resident and safe users. **M3 — Advanced semantics:** hybrid/recurrent/MoE, speculation and specialized whole-layer kernels. **M4 — Task/modality breadth:** embedding/rerank, ASR/TTS/image/video only with real source quality. **M5 — Heterogeneous/distributed:** actual authorized multi-device/peer planning and proven state migration, AOT where compiler accessible. **M6 — Qualification and extension:** security, matched p95/energy and repeatable new model/device/task without engine rewrite.

The full Stage0 IDs, experiments and detailed model/hardware matrices remain in their [original files](INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md); don't flatten or renumber them to match this editorial organization.

## 5. Recording actual results

For each implemented item, record the relevant test command, actual result and
remaining failures in its plan TODO. Keep model/backend/context and numerical
limits precise; a mock or build is not physical inference. The phase checkbox
is complete only when its implementation and required tests pass. Git commits
provide source provenance. See [AGENTS.md](../../AGENTS.md) for agent integration
and the rolling Uncommitted section; no separate evidence ledger is required.
