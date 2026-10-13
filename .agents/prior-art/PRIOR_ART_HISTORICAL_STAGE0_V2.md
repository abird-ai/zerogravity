# Prior-art/historical archive — INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md

**Historical v2 planning snapshot, archived 2026-10-12. NON-NORMATIVE RESEARCH.** External named projects and benchmark figures are preserved as source research, while actual current design and phase gates use project-neutral contracts at `../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`, `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`. User-provided instructions and decisions remain separately in DECISION_REGISTER. Never use historical source-project APIs as the new runtime's type or module names.

---

# zerogravity — Integrated feasibility, critical path, stage gates and delivery contract

**2026-10-10 v2 planning.** All activities NOT_STARTED. This is the authoritative **gate crosswalk** between the research program's S1–S12/G0–G4 and the project's ZG-P00…P23 phases. It is a staged engineering plan, not a date promise, and must not be interpreted as user approval of untested performance constants.

## 1. The core program contradiction we avoid

**Research risk:** building 20+ frozen crates and choosing CubeCL/C++ escape before proving Semantic IR/math and Rust gfx1151 codegen. **Project risk:** building a large symbolic/effect compiler before one GGUF model runs or a kernel reaches competitive GPU ISA. **Solution:** execute small orthogonal feasibility probes in parallel, then a thin end-to-end vertical slice, freeze interfaces only *after* both semantic and physical proof. Maintain a safe general path while specialized kernels are optional.

Canonical root: /home/pvl/spaces/abird/src/zerogravity. Source evidence: /home/pvl/spaces/abird/research/inference (read-only). Implementation still absent.

## 2. Integrated Stage 0 experiments, with mandatory proof

| ID | Path from original plans | Goal and smallest deliverable | Hard go/no-go | Hardware |
|---|---|---|---|---|
| **S0-00** | P00, research task1 | Lean Cargo workspace, rust toolchain+Nix shell+CI0, decision and source-pin ledger, clean CPU smoke | reproducible source/build, no unreviewed license or vendor runtime | CPU |
| **S0-01** | P01, research S1 | Rust GPU kernel compiler bakeoff: CubeCL gfx1151 low-bit GEMV/WMMA, Rust native AMD target alternatives; inspect device ISA/perf, graph capability | correct `gfx1151` results and competitive path or honest pivot | physical gfx1151 |
| **S0-02** | P01/P07, NEW | CUDA provider bakeoff: CubeCL versus new Rust tile compiler and SIMT-PTX compiler (alpha), same simple shapes and reference | Linux/CUDA toolchain feasibility, numeric and source language policy | NVIDIA compatible GPU |
| **S0-03** | P02/P03, research S2 | Safe GGUF+dialect/safetensors slice, tokenizer byte parity, CPU quant/attention + scalar oracle | complete tiny/real dense checkpoint, independent logits/tokenizer | CPU |
| **S0-04** | P04, NEW | Minimal SIR typed tensor + state effect operator, GDN and KV effects, snapshot/commit rule; verifier rejects invalid reorder | independent CPU state trace and invalid-effect negative tests | CPU |
| **S0-05** | P05, NEW | Bounded RIR regions and one kernel candidate registry, forced missing specialization; no per-token interpreted graph | real reference vs lowerer output and safe region fallback | CPU + GPU eventually |
| **S0-06** | P05/P07, NEW | Minimal CIR resource/timepoint/IO simulation: lease, dispatch, copy, acquire/release, commit, one fault | event DAG, alias, stale generation, cancellation/partial read fail closed | CPU fake driver |
| **S0-07** | P09/P10, research S3/S5 | EngineMsg/Event, StepPlan/WorkerDelta, paged KV pool/frontier+pure resource planner simulator | invariants, deterministic ledger/refusals, no global hot mutex | CPU simulator |
| **S0-08** | P02/P15, research S8/S9 | Full-logit ABBA when exact, tolerance/quality when permitted; speculative accept/replay across KV/GDN/RNG | no invisible mutations, exact target sampled distribution | CPU + GPU modes |
| **S0-09** | P03/P09, research S4/S6 | token template, grammar FSM, tool stream and HTTP mock clients, cancellation | official token vectors, API conformance smoke, no tool injection | CPU |
| **S0-10** | P01/P05, research S1b/S10 | ExecutableBundle+KernelPlan artifact, eager/captured graph lifecycle, old driver/shape refusal | no stale pointers/alloc under capture, deterministic bundle key | GPU |
| **S0-11** | P12/P20, research S7 | Offline expert-cache trace simulator: per-layer slots, LFU/recency/hotlist, route overlap, concurrent pressure | reproduce evidence or label mismatch, no policy promotion without gain | CPU trace |
| **S0-12** | P08/P09, NEW | **End-to-end vertical slice**: model artifact → SIR → RIR → CIR + kernel → EngineCore → generated token via CLI/server | one dense model correct on CPU and at least one GPU; disable extreme region and remain functional | CPU + GPU |
| **S0-13** | P10/P11, NEW | 3+ different hybrid state domains shared prefix, compressor alignment and fork/rollback | same causal frontier; exact compatible state reuse, safe partial hit/recompute | CPU then GPU |
| **S0-14** | P20, research S11/S12 | Optional portable Vulkan and zero-weight prefix-derived draft index, A/B against baseline | correct fallback, opt-in experimental policy only after measured result | Vulkan/CPU |

**Mandatory before architectural interface freeze:** S0-00/01/03/04/05/06/07/08/12. S0-01 can fail without killing portable product; it kills the extreme gfx1151 guarantee until a new Rust toolchain or kernel path proves viable. S0-02/09/10/11/13/14 can be independently scheduled when prerequisites ready; select a subset to avoid pretending the full ~93 engineer-days research program fits 4–6 weeks with two engineers.

**Important:** S0-12 requires enough basic artifact/model/runtime support to run one model; do not delay it until a giant compiler with all future state ops exists. S0-04/05/06 start tiny and expand as real ops demand.

## 3. Gates and hard evidence

### G0-A: correctness substrate
- GGUF safe parsing, tokenizer exact vectors, Q4/Q8 CPU quant oracle and at least one real small dense checkpoint; official/independent per-layer/logit tests; semantic state effects verified on synthetic GDN.
- Source/build provenance and license review; user Rust-only source policy not violated.
- Deterministic boundchecked artifact/schema/version; unknown op/quant returns typed refusal.
- No silent precision/layout changes.

### G0-B: codegen and async execution
- Physical `gfx1151` proof with generated ISA, register/LDS occupancy and comparison to selected licensed reference kernels, same dtype/shape.
- Rust CUDA compiler options evaluated when target available; no global CubeCL lock-in.
- CIR fake-resource events, IO-short-read/quarantine/generation tests pass; capture fallback safe.
- Runtime hot path references compiled bundle handles, not an IR interpreter.

### G0-C: first functioning vertical slice
- Real dense model CPU+one GPU verified; forced specialized operator disabled but runnable fallback.
- Same session state/tokenizer/API handling through EngineCore and direct worker outputs, no duplicate independent serving driver.
- Versioned minimal plan introspection and physical memory-fit report.
- Source, phase and tests documented from actual run.

**Architecture freeze only after G0-A/B/C** and only for proven external interfaces; new operator capability registry remains evolvable. If physical GPU not available, state that extreme gates are blocked; proceed CPU-only feasibility without claiming GPU qualification.

### G1: portable product
- one useful binary/SDK path, GGUF/safetensors supported matrix, correct common dense checkpoint(s), tokenizer/template/tools, CPU + demonstrated available GPU, typed refusals, beginner usability;
- no guessed license or model coverage;
- compare same hardware/checkpoint with llama.cpp at TTFT, decode, prefill with quality. Research proposed thresholds ≥70% decode/≥60% prefill; **not user-approved** and should become explicit experimental gates with selected reference.

### G2: serving parity
- paged KV/prefix, native C1/2/4/8/16/32 batching, preemption/fairness, mixed prefill/decode, optional spec and structured output;
- p95 TTFT/ITL, aggregate committed tokens, memory pressure and 24h soak;
- compare pinned vLLM/SGLang on same model/quant/hardware. Historical proposal ≥80% vLLM decode BS1/≥60% BS32, requires confirmation as release floor.

### G3: extreme models
- full model reference and modern checkpoint quant/quality, `gfx1151` strong MMVQ, QSA/GDN and GLM KDA/DSA/mHC, resident+SSD MoE, full hybrid snapshots, MTP;
- Gufo/Maya/DwarfStar matched benchmarks or separate unmatched published context, no fictitious win;
- honest path/capability records for 92/96/128GB and driver variants; no OS OOM.

### G4: breadth/distribution/future
- modern Gemma4/Qwen3.5+/DeepSeek, LoRA, embeddings/rerank, vision and audio as separately qualified paths;
- topology-aware TP/EP/PP and fail-closed peer agreement, optional CUDA/Metal/Vulkan/NPU;
- multi-model hosting and upgrade/ABI/migration, clear security and quality.

## 4. Program phase mapping, unchanged IDs P00–P23

| Existing project phase | Consolidated required adjustment |
|---|---|
| P00 | lean workspace and pinned source/decision evidence; **no mandatory 25 crates**, no Git init unless approved |
| P01 | CubeCL + AMD Rust codegen and NVIDIA Rust tile/SIMT-PTX compiler feasibility; never assume toolchain works from docs |
| P02 | CPU model operators and complete mutable reference state; add grammar/sampler golden as separate test |
| P03 | GGUF/safetensors, tokenizer pretypes, source hashes, imatrix/quant and optional derived pack plan |
| P04 | minimal SIR SSA+effects+state group codec; explicit novel-op rejection, same-plan replay |
| P05 | RIR pure/fusion/legalization + **KIR registry + CIR async resource/events**, bounded specialization |
| P06 | CPU executor direct compiled plan, tested SIMD where possible, correct fallback |
| P07 | GPU device capabilities, validated leases/events, graph safety and per-target Rust kernel provider |
| P08 | first real dense vertical slice and forced per-region fallback; first true kernel/perf evidence |
| P09 | EngineCore/typed worker protocol, CLI/HTTP, tokenizer/grammar/tool, event loops and mode receipts |
| P10 | paged+contiguous KV under one arbiter, optional radix, full-state prefix snapshots and 3+ group valid intersection |
| P11 | typed GDN/KDA/QSA/MLA/DSA/CSA and precise state codecs, long-context and encoder span alignment |
| P12 | MoE route+GPU grouping, expert slabs/lease/miss tiers, CPU lane only after measured benefit |
| P13 | modern Qwen3.8 specialized gfx1151 with ngram SSD tables, MTP and Gufo comparisons |
| P14 | GLM-5.3-Flash state and expert residency, Maya/DwarfStar matched quant and real RAM safety |
| P15 | target-authoritative exact stochastic verifier, CommitPlan/Frontier mechanics, full state rollback |
| P16 | 1–32+ concurrency, route-aware batching, joint speculation/residency cost and measured p95 |
| P17 | model breadth Gemma4/Qwen3.5+/DeepSeek **plus explicit LoRA and embedding/rerank runner contracts** |
| P18 | multimodal and modality-specific encoder budgets, video/audio as qualified extensions |
| P19 | tensor/expert/pipeline/P-D distributed strategies with model/state identity and failure atomicity |
| P20 | bounded offline/online tuning, KernelPlan provenance, eager/graph-mode tactics, rejected experiments |
| P21 | license policy, Rust vendor FFI decision, auth/privacy/plugin trust, cargo-deny, Nix/CI/release |
| P22 | replayable peer-matched bench with quality, hardware/quant, Pareto/performance claim ledger |
| P23 | versioned SemanticOp and KernelProvider registry, architecture evolution exercise, deprecation/migrations |

P00–P05 may overlap only where interface ownership is settled, but P08 vertical slice must not be delayed by implementing hypothetical P17 operators first. P13/P14 use P11/P12/P10 correctness; high-concurrency P16 may evolve independently of two huge model support implementations and must not postpone useful generic serving.

## 5. Actual user device versus dated research machine

Historical `research/inference` plan assumed RTX3060 6GB/Cezanne 27GB + purchase of gfx1151. **Do not treat this as verified current hardware.** At P00 physical `DeviceProbe` must query the available machines and drivers. Users may have a Strix Halo 92/96GB device and a separate NVIDIA notebook; if present, use them and do not recommend purchase. Device class selection by capability, not marketing/model names.

If actual Strix not accessible through connector and hardware cannot be tested, mark physical G0-B and G3 BLOCKED, not VERIFIED.

## 6. Source/kernel and licensing policy

- User chose Rust-authored engine. CubeCL not automatically the only compiler.
- New public Sept 2026 Rust tile and SIMT-PTX compilers need distinct capability/ISA tests. Avoid committing to rustc nightly as required for whole project because one experimental kernel provider needs it.
- Vendored C++/HIP kernel source allowed only if user explicitly resolves language policy; documented external source licenses still required for legal study/porting.
- GGUF input and derived acceleration pack: library packaging names open; research's `APK` confusing with Android .apk, avoid codifying in public ABI.
- Follow threat model/crypto-correct model/prefix ID before writing a disk-KV schema; research 48-byte header is invalid by arithmetic, build fresh checked versioned schema.

## 7. Test and evidence matrix

| Area | Small proof | Long proof | Release blocker |
|---|---|---|---|
| Tensor and state | scalar ref and shape fuzz | full official model/long context | missing op, wrong frontiers |
| Compiler | legal/illegal effect rewrite | GPU direct plan and fallback | semantic change, hot-path IR overhead |
| CIR and cache | lease/event simulator + injected IO | 24h stress/fragmentation/pressure | stale/overwritten device slot |
| Tokenizer/grammar | golden byte/token vectors | full tools/structured output | hidden control tokens/injection |
| Quant/format | block exact decode and manifest | NLL/KL quality bands | implicit downgrade |
| Batching | C1/2/4 goodput | C8/16/32 p95/ITL + fairness | wrong session state/starvation |
| Spec | accepted-length 0..N exact state | distribution preserving sampled multiworkload | rejected draft leaks into state |
| Hardware | probe + one real kernel | multiple driver/quant/context matrix | unsafe UMA overcommit |
| Multi-modal/LoRA | encoder/adapters reference | image span + LoRA concurrent replay | cache cross-use |
| Distributed | two-rank handshake/failure | multi-rank/transport fault soak | partial commit |
| Performance | paired ABBA matched run | full request replay/power+quality CI | unverified marketing claim |

## 8. Engineering estimates and kill criteria

Do not use a misleading linear roadmap. The original research's `~93 engineer-days Stage 0` is **18.6 engineer-weeks of labor**; with two full-time engineers it is ~9.3 elapsed weeks **before overhead/dependencies**, not a blanket 4–6 weeks claim. Reduce Stage 0 to the hard-gate set and schedule optional experiments separately rather than inventing faster parallelism. Project phases are milestone identifiers, not chronology guarantees.

**Kill/pivot conditions:**
1. Compiler/effect model cannot represent target model state without hidden writes → redesign SIR immediately.
2. Rust GPU toolchain cannot generate competitive target instruction code and no Rust-authored escape provider feasible → defer extreme claim; general engine continues.
3. No independent known checkpoint token/logit parity → stop specialization before baseline correct.
4. CIR introduces measurable unavoidable per-token overhead → compile plan to direct command/graph handles and simplify.
5. Memory allocator cannot safely refuse unfit model on UMA → block public release.
6. Repeated policy complexity gives no end-to-end p95 or committed t/s win → keep simpler policy.
7. Quant/model quality weak or stochastic verifier alters target distribution without opt-in → disable changed plan.
8. Kernel plan/version/cache invalidation is unsafe across driver/model schema upgrades → invalidate/rebuild, not continue.
9. Security parser/network tool vulnerability unresolved → block service release.
10. Superiority cannot be matched fairly with reference quant/model → report incomparable, no claim.

## 9. Exit bookkeeping

For every stage record `phase_id, model_sha/quant, operator contract version, plan_hash, source commit, command/test, actual hardware/ROCm/CUDA/driver/power, CPU/GPU numerical parity, quality results, memory headroom, latency distribution, fallback outcome, known skip/failure, next decision`. Mark statuses `NOT_STARTED/ACTIVE/VERIFIED/BLOCKED/DEFERRED` in current status document. Research and design writing alone never marks a phase VERIFIED.

## 2026-10-12 additional optional feasibility track: GPU + XDNA2 NPU and TaskProgram

**User decision U-016:** include the Gufo fork's GPU/iGPU+XDNA2 cooperative prefill and upstream Gufo ASR/TTS/image/video pipelines in our clean extensibility design. Keep S0-00…S0-14 unchanged; add **S0-NPU-A…S0-NPU-F as an independent gated experimental track**. This track does **not** block CPU/GPU model vertical slice or G1 portable release. It requires physical NPU hardware for qualification and *a feasible Rust-owned AIE source toolchain*. Refer to [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) for exact invariant and source links.

| New gate | Dependency and artifact | Objective proof / closure |
|---|---|---|
| **S0-NPU-A** | P04/P05 SIR+RIR+CIR; mock GPU/NPU backend | Model-independent output-row `ProjectionRowSlice`, complete row coverage, first-use dependency DAG, effect legality and canonical GPU-only fallback; fake device test no XRT |
| **S0-NPU-B** | S0-NPU-A and P01 device API | **Rust-owned** XDNA2/AIE kernel/compiler feasibility and bounded XRT runtime bindings, program/insts hash, supported shape and numeric encoding; if only non-Rust AIE source exists, mark blocked-by-Rust-policy |
| **S0-NPU-C** | physical qualified isolated `gfx1151+XDNA2` host | Firmware/AGESA/ROCm/XRT/FCLK/power/suspend, shared memory and safety preflight; no machine-check hangs during controlled concurrent testing; ≥24h soak including restart/resume; user-controlled profile, no auto root tweaks |
| **S0-NPU-D** | P11/P13 real Qwen model + source checkpoint | Reference versus GPU-only versus XDNA2 sliced GEMM for GDN z, QSA query/gate, shared gate/up separately and combined, full logits/state/quality, strict numerical-mode refusal, MTP/grammar |
| **S0-NPU-E** | S0-NPU-C/D and P20 performance harness | Reproduce/qualify source report UD-IQ4_XS 16K `1698→1821` author results under **identical fabric clock** in A/B, also 8K/32K/40K cold, C1/C4, p95, energy/idle watts, fixed+auto clock control, confidence |
| **S0-NPU-F** | S0-NPU-A/E; optional P17/P18 stage model | Add a second compatible projection family / encoder stage without edits to model-core/HTTP; unsupported NPU defaults GPU-only; TaskProgram ASR/TTS/media stage outputs have independent task reference/quality |

**Exit conditions:** S0-NPU-A may pass on CPU fake executor; S0-NPU-B remains BLOCKED if no Rust-authored AIE path; S0-NPU-C/E cannot be marked verified on simulated hardware; S0-NPU-D needs real model weights and numerical results. A frozen host is an unsafe outcome that blocks auto-enable until actual platform problem resolved; a software timeout does not guarantee recovery. Never perform destructive power/BIOS changes on the user's everyday machine to force a demonstration.

**Release/capability state:** NPU absent/unqualified/experimental/qualified. Default GPU-only; no implicit sudo/fabric hold/NPU keepalive. A supported TaskProgram (ASR, TTS, image, video) must separately pass model/codec/iteration/output/quality and resource state acceptance, not simply instantiate a language model decoder. Each capability has model+hardware+precision+source-code qualification. Partial value is fine: safe GPU-only fallback remains a genuine deliverable.

**Phase placement:** P01 NPU compiler feasibility, P05 RIR row-split and CIR first-use events, P07 DeviceProvider and pinned XRT BO, P10 physical budget, P11 hybrid math/reference, P13 Qwen GPU+NPU, P16 cost-aware scheduling, P17–18 TaskProgram media, P20 same-clock A/B autotuning, P21 SoC hard-hang security/power gate, P22 matched quality/performance. Keep P00–P23 IDs stable.