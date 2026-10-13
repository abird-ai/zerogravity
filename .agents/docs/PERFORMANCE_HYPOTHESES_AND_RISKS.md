# Architectural risks, opportunities, trade-offs and falsification criteria

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


Status: initial design research. Risks below are not implemented bugs. We must distinguish **observed external project evidence**, **deduced system requirement**, and **unproven zerogravity hypothesis**.

## 1. Risk register (severity × uncertainty)

| ID | Severity | Risk | Mitigation/proof | Kill or pivot trigger |
|---|---|---|---|---|
| R-01 | Critical | Rust GPU compiler cannot produce competitive `gfx1151` kernels | P01 ISA/occupancy/real kernels bakeoff, add target intrinsics | consistently poor unavoidable codegen in key ops: implement custom lowering or revise toolchain |
| R-02 | Critical | Semantic IR omits state effects, producing wrong hybrid/speculative outputs | typed effect tokens, CPU oracle, mandatory rollback tests | can't express snapshot/commit without hidden side effects: redesign IR before model work |
| R-03 | Critical | Physical APU memory/GTT overcommit hangs system | global arbiter, watchdog, conservative admission | unsafe default loads: freeze release and fail-closed |
| R-04 | Critical | Global planner complexity adds hot-path overhead | ahead-of-time region plans, no semantic interpretation during decode | measured launch/CPU scheduling latency material: simplify/compile direct plans |
| R-05 | High | Numerous specialized kernels create unmaintainable model forks | operator family registry, versioned model recipes, legal region interfaces | adding next model requires rewriting serving/session or many duplicate layers |
| R-06 | High | Quantized speed improvement degrades reasoning/tool accuracy | source/reference logits, KL/NLL, official tasks, explicit quant artifact | significant quality regression hidden by next-token smoke |
| R-07 | High | Dynamic expert-cache miss synchronization stalls GPU | device slot hits, async bounded ring, fixed slabs and lease safety | host round-trip dominates hit path or GPU hangs on IO failure |
| R-08 | High | Cross-request expert grouping costs more than reuse saves | route-overlap sweep and end-to-end P95/throughput | no repeatable net gain on representative workload -> disable policy |
| R-09 | High | Prefix COW snapshots consume too much memory | state-domain cost model, deferred snapshots, recompute alternative | branch creation/eviction costs exceed prefill saved |
| R-10 | High | Adaptively switching speculative policies harms latency/quality | exact target sampler, hysteresis, complete accepted-prefix rollback | low-acceptance workload worse than no-spec beyond tolerances |
| R-11 | High | Graph capture makes cache pointer/memory lifetimes invalid | stable device control table, event-backed lease, eager fallback | captured plan fails exact replay, graph disabled for region |
| R-12 | High | Distributed partial failure corrupts state frontier | handshake+epochs, collective commit and fail-closed | rank mismatch or stale partial reused -> stop deployment |
| R-13 | Medium | Every new format/architecture must be handcoded | semantic operator composition, explicit adapter and CPU oracle | parser accepts but cannot execute new op: report unsupported honestly |
| R-14 | Medium | Autotuner overfits to one shape/thermal condition | ABBA and end-to-end, cache version keys, regression suite | no real-model win or tail latency regression |
| R-15 | Medium | Broad API compatibility masks tool-tokenization mismatches | exact template/role/tool parsing, model-specific vectors | hidden reasoning/tool state leaks or incorrect prefix |
| R-16 | Medium | External source licensing and data rights | pinned attribution/license ledger, independent implementations | unavailable or incompatible license -> study behavior only |
| R-17 | Medium | Vendor driver versions break compiled plans | version fingerprint and cache invalidation, current physical tests | stale binary unsafe/wrong outputs -> invalidate/recompile |
| R-18 | Medium | Complex multi-crate setup slows contributors | few coherent crates, direct examples, decision/handoff docs | extending operator requires pervasive changes -> consolidate interfaces |

## 2. Central falsifiable hypotheses

**H-A Joint scheduling/residency:** dynamic batch/grouped experts will create extra weight reuse beyond separately tuned batcher+cache; predicts benefit at moderate concurrency with routing overlap, no benefit on one row/uniform routes. Falsify with H04/H05.

**H-B Semantic state-aware prefix sharing:** shared recurrent/indexer state prevents expensive hybrid prefill for repeated agent prefixes; predicts TTFT improvements on long common prompts if snapshot size and transfer less than recompute. Falsify with H07.

**H-C Adaptive speculation including memory:** choosing depth/drafter based on full verification+cache cost beats acceptance-only MTP policy across coding and reasoning workloads; expect little benefit on random text. Falsify with H08.

**H-D Rust-generated extreme kernels:** target-specific Rust kernels can equal or improve licensed reference kernels on `gfx1151` with enough backend lowering control; not established by Rust syntax. Falsify with H01.

**H-E No generality tax in hot path:** compiler/plan produces direct launches equivalent to hardcoded model runner, with no dynamic per-op SSA interpretation; falsify with H02.

**H-F Safe scratch borrowing:** flexible prefill scratch reusing cold expert-cache capacity improves real prefill without extra permanent GiBs or decode thrash; falsify with H05/H09.

**H-G Kernel variants per regime:** prefill/decode/verify tailored algorithms beat one generic kernel under same model quant and memory; strongly supported in a quant-aware inference project, Gufo and DwarfStar externally, but must re-prove implementation.

## 3. Key trade-off choices

### A. Semantic IR richness vs maintainability

Too little IR produces per-model manual code and unsafe state transformations; too much creates graph optimizer complexity/compile latency. Choose minimal typed SSA + explicit state effects needed by actual initial dense/GDN/KDA models; expand only with concrete proven operator. Preserve readable IR dump and per-node reference function.

### B. CubeCL vs direct hardware kernels vs vendor libraries

CubeCL: Rust syntax portability/cross-backend, fast development; possible codegen gaps. Direct AMDGPU/CUDA intrinsics: best control, more complexity; must preserve Rust maintenance. Vendor BLAS: peak tuned matmul and lower risk, but strict Rust-only interpretation and ABI/version questions; keep as optional until user resolves O-003. Compare actual shape/latency and source policies instead of dogma.

### C. One cache vs many caches

One budget arbiter required for host safety; one eviction policy across dense weights, experts, KV, recurrent and lookup rows is wrong. Each resource class needs distinct demand/reconstruction/cost and one global arbitration. Use typed leases and inventory to avoid independent hidden overcommit.

### D. Whole-layer fusion vs operation modularity

Big fused kernel reduces launches/global traffic but may spill registers, limit occupancy, obscure numerical testing and dynamic misses. Preserve decompose reference and choose candidate with full-model A/B; no mandatory maximal fusion.

### E. Full CPU reference vs portable but slow GPU backend

CPU reference essential to verify model math and unknown operator; a generic backend may be portable but lack needed FP8/atomics/context and thus unable to execute giant model practically. Support statuses separate; model works where device resources and all ops available, otherwise honest error.

### F. Automatic adaptation vs predictability

Runtime can adapt prefill quantum, batching, speculation and caches at request/step boundaries. But frequent adaptation can thrash plans/scratch and make reproducibility difficult. Use measured cost model, bounded variant count, hysteresis, deterministic quality mode and logging of decisions.

## 4. Common failure of reasoning to avoid

- "IR compiles model" therefore faster: false without kernels, correct layouts and memory.
- "125B parameter" means stream 125B every token: false for sparse MoE and disk-only lookup tables; active parameter/quant/traffic important.
- "All experts fit GPU-addressable GTT" means RAM safely available: false on UMA.
- "Continuous HTTP clients" means true native batched matrix: false when ordered per-session fallback.
- "60 tokens/s speculative" means 60 tokens/s serial reasoning: false; acceptance/workload-specific.
- "General GGUF reader" means any architecture supported: false; model semantics/position templates separate.
- "Bitwise same tokens" means same model quality: false, and cross-plan floating order can change close ties.
- "One high-quality kernel" means full model faster: false; real end-to-end critical path.
- "SSD memory backing" means all huge models interactive: false; random expert misses may dominate.
- "Linux driver supports graph capture" means our ROCm graph capture implementation works: test physical backend.

## 5. Decision process for research conflicts

When a new paper/project/repo claim contradicts this design, record exact commit/config/metric, identify if semantic contract actually differs, reproduce small test, compare required complexity and update decision register and risk. Benchmark evidence overrides attractive architecture diagrams. If can't reproduce, mark uncertainty. **Never silently rewrite history**: add decision superseding old ID, date and rationale.

## 6. Prioritize and prune complexity

If an optimization consumes substantial design complexity without a repeatable full-model win on representative workload, keep the simpler path and note the rejected experiment. Reserve long-term extensibility but build only proven interfaces and at least one complete model end-to-end before distributed/autotuning breadth. User's performance ambition is ambitious; over-engineering jeopardizes it.

## 7. Immediate actionable risks

Before implementation, resolve:
- exact vendor-driver/FFI interpretation of "entirely Rust";
- whether `gfx1151` available for repeatable kernel perf testing;
- first complete dense and hybrid model artifact/checkpoint choices;
- model quantization/checkpoint licensing for continuous integration;
- stable source pins of CubeCL, Gufo, Maya, DwarfStar, GGML;
- quality threshold policy and physical memory watchdog requirements.

These are open decisions or empirical feasibility gates, **not** instructions to ask the user to repeat already answered requirements.

## 2026-10-10 integrated v2 addendum — new research risks and falsifiable differentiation

**Reconciliation finding:** old research already identifies joint expert-affinity scheduling, MTP-as-group, scratch/slot solver, state frontiers and capture-stable expert table; those are not wholly novel zerogravity inventions. The *new potential* comes from embedding them in SIR/RIR/CIR's unified effect/resource-aware compiler, so one policy can legally reuse results across model families and hardware. No outperformance is guaranteed.

**Additional high-priority risks:**
- R-19: **CIR overbuilding** — attempt to recreate a general-purpose stream/IR runtime before a real model. Mitigation: minimum four commands+events+lease sim first; pivot if no legal overlap or too much compile/hot overhead.
- R-20: **GPU codegen lock-in** — selecting one Rust GPU DSL/compiler version before physical `gfx1151` feasibility or considering alternative Rust CUDA tile/SIMT compilers. Mitigation: capability interface, independent bakeoffs and late binding.
- R-21: **graph effect legality drift** — replacing closed region while GDN/DSA/mHC and grammar effects mismatched. Mitigation: state-domain verifier and forced op fallback tests.
- R-22: **too many optimizer rewrites** — bounded graph-search compile-time explosion. Mitigation: bounded pure-region search and offline only, evidence before default.
- R-23: **3+ hybrid prefix groups** — naive two-group hit-intersection algorithm generalized incorrectly or partial compressor row. Mitigation: exact `CanResume`/alignment/restore per group and synthetic 3+ group tests.
- R-24: **false Rust-only conformance** — vendor C++/HIP escape imported without explicit user decision. Mitigation: Rust owned kernel source policy and external API/driver distinction.
- R-25: **unapproved product/release decisions** — historical research council "resolved" license/MRSV/version treated as user approval. Mitigation: decision provenance IDs and versioned rulings.
- R-26: **false universal model fallback** — unknown model math mapped to plausible but wrong dense ops. Mitigation: registered semantic op and reference oracle required, no unsupported custom code execution.
- R-27: **incorrect binary serialization** — historical research 48/54-byte KV header and magic conflict. Mitigation: checked schema-first ABI fuzz and no adoption by copying.
- R-28: **model/user hardware assumption** — old research 6GB 3060/27GB and 128GB Strix purchase treated current. Mitigation: real capability probe and user-confirmed availability only.

**Highest-upside hypotheses and test:** (1) SIR/RIR/CIR lets us optimize asynchronous expert/lookup/kernels without model fork—prove 2 model families share compiled region implementations; (2) state-aware prefix sharing across GDN/KDA/indexer significantly reduces long-agent follow-up TTFT—prove C8 shared-prefix state vs recompute; (3) GPU route-aware batch×expert residency×MTP boosts committed tokens/sec at quality/P95 parity—compare baseline sequential and grouped; (4) offline rewrite/tactic search finds stable wins under captured serving mode—compare serving-aware kernel-tuning research-style tactic timing; (5) new Rust GPU compilers close ISA gap without C++ kernel source—physical gfx1151/CUDA parity evidence.

**Maintainability test:** ask external agent to add a new model attention state domain and alternate GPU backend without touching HTTP, model store, sampler, worker core outside stable interfaces. Document file count changed, compile time, added tests and duplicated code. Failure triggers interface redesign even if early benchmark favorable.


## 2026-10-12 Gufo XDNA2 opportunity and risks

**H-I — cross-accelerator independent GEMM slicing:** RIR can select disjoint output rows for a known GEMM projection, overlapping GPU and NPU with first-use join. Gufo fork author reports +7.2% prefill on one Qwen UD-IQ4_XS 16K case; shape/precision/safety/clock matter. Test exact source checkpoint, held clock equal ON/OFF, output/logits/state parity, real async overlap and pinned LPDDR critical path. Kill default NPU use if full-request gain inconclusive or precision quality inadequate.

**R-29 Critical — machine-wide GPU/NPU hard-hang safety:** reported GPU fabric-clock transitions simultaneous with NPU produce machine checks and freezes. Not recoverable by ordinary process watchdog. Require opt-in, isolated physical soak, firmware/XRT/ROCm/FCLK matrix and no automatic privileged system changes; GPU-only always safe fallback at admission.

**R-30 High — Rust AIE compiler/driver viability:** Gufo source is C++/Python with executable `.xclbin`. A Rust wrapper is insufficient for user Rust-owned kernel policy. Must prove Rust-maintained AIE codegen or explicitly record unsupported XDNA until user resolves source policy.

**R-31 High — NPU BFP8 numeric drift:** 9-byte shared exponent/8 mantissa compute can perturb GDN/attention indexer output despite matching greedy test. Add strict reference full-logit/quality/target verifier gate and deny offload under unsupported numerical mode.

**R-32 Medium — multi-task/control loop explosion:** upstream Gufo ASR/TTS/image/video cannot be forced into autoregressive next-token state; TaskProgram with task-local effects/loop/output under shared SIR/CIR needed. Prevent per-model competing inference servers or huge unbounded generic framework.

**R-33 Medium — FCLK power/thermal confounding:** held-clock alone may improve GPU prefill but increase idle consumption and physical risk. Bench same-clock ABBA and energy/24h, not NPU ON held vs GPU OFF automatic. Do not claim all `gfx1151` machines have identical AGESA/driver safety.

## v4 critical-path and complexity risks (review R4-01…R4-24)

**New critical risks:** partial state publication despite individual StateDomain commits, device kernel "done" confused with TaskSession commit; host↔GPU router/sample sync each token; global allocation lock per kernel under C32; shape specialization explosion; compiled kernel caches keyed accidentally on live request IDs; unchecked numeric/top-k changes; distributed rank lost ACK and false network exactly-once; workflow DSL hidden inside core and unneeded training/autograd bloat.

**Mitigation:** prepare all domains then one logical PublishRoot; GPU-native StepProgram only for supported exact state/sampler modes; per-device suballocators under coarse global admission; bounded symbolic ShapeGuards and bundle cache; source quant/compute precision/task quality independently versioned; fencing epochs; sequenced output with cursor policy; trusted minimal TaskDriver and external tools above TaskRequest; one library+binary initial rather than many crates. Hardware safety and Rust-owned kernel requirements remain independent of mathematical correctness.

**Falsification:** H32–H43, Stage0 S0-V4-A…J and a second-model/new-device/non-token-task change-radius drill. A new AI model/domain that forces edits to unrelated service/runtime/physical allocator fails the modularity test; a high-performance kernel that adds unbounded CPU work, P95 regression or unsafe memory/clock changes fails optimization qualification. The review document preserves detailed severity, evidence and decisions.
