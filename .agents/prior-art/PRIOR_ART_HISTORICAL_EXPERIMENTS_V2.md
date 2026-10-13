# Prior-art/historical archive — EXPERIMENT_PROGRAM.md

**Historical v2 planning snapshot, archived 2026-10-12. NON-NORMATIVE RESEARCH.** External named projects and benchmark figures are preserved as source research, while actual current design and phase gates use project-neutral contracts at `../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`, `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`. User-provided instructions and decisions remain separately in DECISION_REGISTER. Never use historical source-project APIs as the new runtime's type or module names.

---

# zerogravity — Measured-performance and feasibility experiment program

**Status:** proposed. No zerogravity experiment has been run. Every numerical result from an external project is an attributed historical source claim and must not be used as an achieved result.

## Scientific protocol — same for every candidate

**Proposition:** a modification changes a specified independent variable; define the expected mechanism, affected critical-path stage, failure hazards and stop criteria *before* implementation. Evaluate A (locked baseline) and B (candidate) using identical tensor weights/checkpoints, tokenizer/templates, numerical policy, quantization/layout, runtime versions, context, prompt and output budget, cache warmness, power/thermals and exact measurement windows. If B alters precision or sampling distribution, classify as a different quality/speed trade-off rather than a speed-only improvement.

Artifacts per run: commit SHA, dirty diff or tree hash, Rust and kernel compiler SHA, OS/kernel and GPU driver/ROCm/CUDA/Metal versions, device/frequency/power cap, RAM/UMA carveout/GTT, quant file SHA and manifest, graph plan hash, response hashes/logits, workload corpus license, context depth/new tokens, concurrency, model format, prefill/AR/spec method, acceptance distribution, cold/warm state, expert residence/hits/misses, bytes SSD and memory transferred, queue and TTFT, per-token latency histogram (median/p95/p99), memory peak and resident set, errors, repetitions and raw JSON/CSV.

**Protocol:** warmup and stabilize; run mirrored experiments in alternated ABBA/BAAB order, multiple trials, measure noise and uncertainty; report total output tokens/time plus per-sequence latencies, not a lone maximum. Prefer wall-clock end-to-end HTTP/SDK benchmark plus isolated kernel attribution. Counter instrumentation can change scheduling; use separate diagnostic and non-instrumented performance runs. If data is inconclusive, label inconclusive and do not auto-enable.

**Safety:** never intentionally overcommit RAM to trigger Linux OOM/GPU reset on a valuable machine. Simulate quotas first; physical memory tests require watchdog, OS headroom, spare swap awareness and automatic abort. Do not run competing GPU work in the middle of a performance run.

## Baseline and comparison ledger (not apples-to-apples)

Research starting points from prior source documents:
- Gufo on 128 GB Strix Halo and Qwen3.8-Flash-Next UD-Q4_K_XL: single-user AR ~1700.52 tok/s prefill and ~25.95 tok/s decode at depth 0; long 128K AR ~1478.66 pp and ~24.92 tg; repetitive MTP ~60.39 tg, mixed MTP ~31.32 tg, C8 AR ~108.68 aggregate tg. Comparators pinned llama.cpp revisions; each table is typically one warmed sample.
- A specialized engine with native quant on 128 GB Strix Halo: ~1584 tok/s prefill at 8K, ~1517 at 131K, ~37.6 serial tg (short context) and ~55–56 coding-agent speculative tg (different workloads/versions); core engine source nonpublic.
- Maya on Strix Halo GLM-5.3-Flash Maya-S 2-bit: ~210–217 tok/s prefill and ~17.8 tok/s generation with all experts effectively resident; changes to chunking/hipBLASLt made substantial prefill gains; on discrete dual AMD ~33 tok/s with MTP, not a straight device-to-device comparison.
- DwarfStar ROCm on 128 GB Strix Halo DeepSeek V4 Flash Q2: tuned ~292.86 tok/s for warm 4K prefill (from ~187.26 prior); GLM/DeepSeek/other runs differ in weight quant and implementation. DwarfStar's Qwen3.8 has Metal/CUDA but no ROCm as of inspected snapshot.
- vLLM/SGLang dense serving throughput from per-model official benchmarks only; must remeasure with reproducible same device/models. An independent Rust serving project reports model-dependent gaps, especially Gemma 4 MoE.

**Important:** Qwen3.8 Flash-Next, GLM-5.3-Flash, DeepSeek V4 Flash and Gemma 4 are different computational models; their tokens/s cannot prove which engine is faster.

## Experiment H01 — Kernel compiler feasibility on gfx1151 (ZG-P01)

Hypothesis: Rust-authored GPU kernels can use wave32/MFMA/WMMA and implement Q8_0/Q4_K/IQ2_XXS decode and grouped-MoE efficiently enough to support model-family peak paths.

Test: pinned synthetic and real weight blocks for `M=2560/4096/8192, K=640/2048/4096/8192`, 1,2,4,8,16,128,256,1024,2048 rows where meaningful; baseline legal HIP implementations (own or licensable references), Rust CubeCL and optional direct backend. Compare generated ISA, VGPR/LDS, occupancy, instructions, alignment, wave32 vs wave64 and per-kernel latency. Hold numeric oracle stable; compare ragged K and tensor tail. Report whether a given backend loses because of memory traffic, compiler codegen or launch overhead. **Go/no-go:** correct operation + plausible direct path to parity without prohibitive compilation constraints.

## H02 — Semantic IR overhead removal (ZG-P04/P05)

Hypothesis: a lowered plan has no per-operation semantic interpretation overhead versus direct handwritten dispatch. Build same tiny model on CPU and GPU via (a) semantic interpreter, (b) optimized plan, (c) direct static control. Measure host launch overhead, GPU idle gaps, plans rebuilds and warmup; inspect callable/captured kernel sequences. Fail if IR effects enforce unnecessary global synchronization or numerous host callback barriers.

## H03 — Whole-region fusion (P05/P11/P13/P14)

Candidates: RMSNorm→quant→projection, GDN recurrence→state/conv, routing→selected-ID table, HC residual→gate, sparse attention select→compute, LM head/sampling. Check fusion's register spills, occupancy, scratch, gate stochastic ordering. Use synthetic shapes and full-model end-to-end. Some fusions intentionally rejected if quality or throughput regress.

## H04 — GPU route-aware batching of MoE (P12/P16)

Hypothesis: batching rows across requests/speculative candidates by selected expert reduces re-reading weights and improves grouped MMQ/GEMM without worse P95. Build deterministic workloads with uniformly random vs skewed route overlap: mean distinct experts per N assignments; batch rows 1/2/4/8/16/32; model-specific expert matrices. Compare ordered per-token, grouped-by-expert and scheduler-guided with fixed and pressure-limited cache. Measure router output transfer (must stay GPU-side), regroup/scatter cost, bandwidth and fairness. Abort experimental policy on meaningful worst-case latency or memory regressions.

## H05 — Residency × scheduling co-optimization (P10/P12/P16)

Compare fixed LFU/LRU/layer-partitioned cache, preloaded popularity, dynamic cache, GPU hit/no-miss and OS page-cache subsets; then jointly vary batch width and resident expert slots. Benchmark hit% is inadequate: measure miss latency critical path, transfer bytes, resident slab fragmentation and queue stalls. Test 128 GB `gfx1151` full-resident vs 92–96 GB with SSD fallback, separately from PCIe discrete. Disable CPU expert lane on UMA by default until contention/benefit proven; test PCIe variants with CPU lane separately.

## H06 — Disk-only n-gram/Engram (P03/P10/P13/P17)

Hypothesis: read-only row fetch from a huge lookup table can use a much smaller bounded RAM cache than mapping entire table. Test random/scattered/sequential/skewed row accesses, correct BF16 reconstruction, alignment, repeated IDs, out-of-order reads, corrupt/truncated source, cold and warm NVMe, OS page cache; compare `pread`, `io_uring`, suitable direct IO; quantify SSD throughput/latency and impact on main GPU inference. Never infer 95+ GiB disk table means 95+ GiB must be GPU resident.

## H07 — Hybrid prefix state sharing (P10/P11)

Hypothesis: shared immutable prefix includes recurrent GDN/KDA, indexer and token-history state, reducing prefill without expensive copies. Workload: 40K shared system/repo prefix with 8 diverging agents, branching in middle and tool-result rewinds. Measure full cache reconstruction vs immutable prefix COW vs eager copies vs deferred state materialization. Verify every state buffer, logits and RNG against fresh evaluation. Test eviction, file restoration, cancellation and mixed image/text layouts.

## H08 — Adaptive speculation economics (P15/P16)

Measure no speculation, MTP 1/2/3 depths, trained DSpark when supported, independent smaller drafter, prompt lookup and n-gram. Workloads: repetitive code, agent tool results, unpredictable prose, math/reasoning, nonzero temperature, long context and mixed concurrent sessions. Controller chooses by *net committed tokens per total draft + verification + rollback + idle cost*, not just raw acceptance. Compare switching and hysteresis; low-acceptance bypass vs always-on; correctness against exact target sampling. Publish acceptance distribution, rejection position, ratio of useful target forwards and quality differences.

## H09 — Prefill versus decode resource competition (P09/P16)

Workload matrix: C=1,2,4,8,16,32; requests arriving under Poisson/bursty schedules; 256, 2K, 8K, 32K prompts and 128/512 outputs. Compare pure FIFO, token-budget chunking, prefix-aware, mixed prefill/decode with 128/256/512/1024 quantum and bounded latency priority. Observe TTFT P95, inter-token P95, starvation, total tokens/s, GPU occupancy and expert residency. Tune for separate interactive vs throughput objectives.

## H10 — Compressed/sparse attention context depth (P11/P13/P14/P17)

Independent operator oracles and full models: dense GQA, sliding, QSA, KDA/MLA+DSA, CSA/HCA, compress boundary. Depths 0/4K/8K/16K/32K/64K/128K/256K and > where physically supported. Test masked index rows, selection ties, FP32 reductions and cache quantization. Compare GPU selectors vs CPU fallbacks; quality fixed-token logit/KL and retrieval. Discard alleged win if it relies on changing number of selected keys, precision or attention semantics without labeled policy.

## H11 — Unified-memory safety and throughput (P07/P10/P13/P14)

Profile Linux Strix Halo UMA carveout, HIP-reported allocatable VRAM/GTT, `MemAvailable`, `CmaFree`, physical working set, pinned stage limits, kernel/firmware and IOMMU settings. Capacity targets 64/92/96/128 GB as available; vary concurrent requests and prefill scratch. Require admission refusal before danger, stable cleanup, no system stalls, performance with/without redundant pinned RAM tier. Do not prescribe changing kernel boot parameters as a universal solution.

## H12 — Mixed quantization and model quality (P03/P12/P13/P14)

For same reference FP8/BF16 model, compare Q2/IQ2 and Q4/K plus Q8 dense/mHC choices, unchanged lossless layout repacking vs deliberate requantization. Quality gates include logit RMS/max error, KL divergence, next-token agreement, calibration case official continuations, coding/tool/long retrieval. Separate speed due to better layout, cache footprint and **changed numerical precision**. Refuse default quality upgrade/downgrade without recorded trade-off.

## H13 — Distributed topology and transport (P19)

Measurements: local 2/4/8 GPUs, heterogeneous PCIe, two 128 GB hosts with RDMA/TCP where available. Compare layer pipeline (prefill), tensor parallel and expert parallel (decode), and batching under constrained links. Quantify P2P/RDMA activation traffic, comm stalls, memory capacity, backpressure and workload scaling. Inject partial writes, timeout and wrong model/checkpoint handshake to test fail-closed. No unnecessary remote access/security downgrade.

## H14 — Long-lived agent workload (P09/P10/P15/P17)

Replay realistic coding agent conversations: shared tools/system instructions, full repo context, repeated tool results, JSON, reasoning boundaries, stream interruptions, multi-turn reuse and image attachment. Measure total task wall time and quality, not just kernel t/s. Identify repeated-prefix savings, hit rate, checkpoint storage and branch costs. Compare API semantics against native tokenizer/template references.

## Bench artifacts organization (future)

```
bench/
  corpora/         # hashes, origin, license (no secret transcripts)
  configs/         # exact comparator commands
  kernels/         # microbench kernel shapes
  models/          # same GGUF/safetensors checkpoints + hashes
  scenarios/       # arrival/concurrency/cache profiles
  results/YYYY-MM-DD/<machine>/<engine>/<model>/
     run.json  metrics.csv  logits.sha256  stdout.log  stderr.log
  reports/         # methodology, confidence, wins/losses
```

Scripts must never publish model weights or private prompts by default. End-to-end metrics originate from complete responses with EOS/stop semantics described, not a misleading concatenation of isolated calls. Any measurement with failed outputs, OOM, swap thrash or uncontrolled concurrent GPU activity is invalidated.

## Qualification decisions

- `PASS`: correctness gates pass; repeatable speed gain above noise on matched workload; safe memory; regression set green.
- `NEUTRAL`: correctness pass but benefit within noise or on only microbench; keep simplest existing path, or guarded opt-in.
- `FAIL`: numerical/state error, unsafe memory, latency regression beyond policy, quality decline or unsupported behavior; revert candidate.
- `INCONCLUSIVE`: insufficient trials, thermals/driver differences, incompatible quant/model, uncontrolled hardware; no performance claim.

Every optimization report links the source commit, test corpus, full benchmarks, reasons for retaining/rejecting and next investigation. That is part of zerogravity's intended transparent engineering discipline.

## 2026-10-10 integrated v2 research expansion (new experiments H15–H25)

**H15 — Rust compiler provider comparison:** CubeCL, direct target Rust HIP/AMDGPU option on gfx1151; CubeCL vs a Rust tile compiler and experimental a Rust SIMT/PTX compiler SIMT on suitable NVIDIA hardware. Compare generated ISA/LDS/register/tiles, Q8/Q4/IQ2 GEMV, grouped GEMM, attention and compile stability under exact same inputs/quality. Driver/FFI policy remains open; never pass external C++ engine as Rust-owned kernel result.

**H16 — Compiler effect legality:** synthetic full KV + GDN + sparse indexer + sampler RNG+grammar, randomly permute legal pure ops and deliberately illegal state reorders; SIR verifier and RIR pass reject illegal, reference state/logits remain stable. Compare full-model after chunk/snapshot/fork/rejected draft.

**H17 — CIR scheduling and resource reuse:** async simulated GPU/SSD/CPU timepoints, leases and alias check, forced stale slot generation, parallel copies and IO, fault injection; physical measure whether next-layer expert prefetch overlaps useful computation and whether scratch borrowing reduces resident pressure without p95 regression.

**H18 — Execution mode tactic selection:** same kernel algorithms under eager vs device graph capture (a 2026 kernel-dispatch lesson), measure full model and serving TTFT/P95, not just GPU event; persist mode-keyed candidate only after numerical and quality proof.

**H19 — Bounded equality search:** choose one pure norm/quant/projection or attention prelude and compare no rewrite vs heuristic vs bounded e-graph search (bounded graph-search lesson), report compile seconds, code size, memory, generated plan and actual end-to-end t/s. Kill if compile-time explosion or no meaningful gain.

**H20 — Three-plus domain prefix snapshots:** full/sliding KV + GDN/KDA+compressed indexer+token ngram/adapter/multimodal; compute exact common valid resume frontier under different alignments and storage tiers; compare replay vs state restoration, shared agent TTFT, byte identity in same fixed plan and full logits. Guard tenant salt and model/quant identity.

**H21 — Extensibility drill:** add a novel stateful attention op to SIR/reference, implement generic kernel and one target-tuned kernel, add a new device compiler provider. Count unrelated crates/files modified, run API/session unchanged; architecture fails if model-specific HTTP/allocator branching required.

**H22 — Full product correctness:** content-addressed model store/manifest/atomic partial download, tokenizer pre-type exact vectors, grammar mask/rollback, LoRA cache generation and embedding/rerank normalization, image encoder spans; fuzz inputs and staged release matrix, no unknown-op execution.

**H23 — Prevent misleading claimed improvements:** maintain separate tables of same-model same-quant benchmarks and unmatched source project literature. DwarfStar 8×L40S DeepSeek TP cannot stand in GLM matching cell; fix historical dataset attribution. Report uncertainties/incomparable rows as such.

**H24 — Unified-memory machine safety:** probe actual available devices instead of old 3060/Cezanne/27GB research assumption, validate physical RAM/GTT/cgroup/IO under 92/96/128GB (when present), no forced host OOM and full resident/streamed model fit. All RAM-heavy trials guarded; no blind firmware/kernel changes.

**H25 — Compile/hardware hot-path cost:** bench one real model as (a) CPU reference, (b) SIR interpreter for debugging, (c) RIR/CIR compiled bundle direct launches and (d) hand-coded direct dispatch baseline; demonstrate no meaningful per-token IR interpretation overhead. Include graph capture and worker diff overhead.

**Evidence discipline:** S0-00…S0-14 in integrated plan define Stage0 mandatory status. Every H15–H25 is a falsifiable proposal, not proof of performance. New benchmark cells include `semantic_schema, state_layout, cir_event_version, kernel_provider, execution_mode, model/quant/checkpoint/hash, context, new prefill tokens, power/driver, p95 and quality`. Do not automatically enable unbenchmarked heuristic or compile search in serving.


## 2026-10-12 research expansion: Gufo upstream and XDNA2 cooperative prefill (H26–H31)

**H26 — Same-model independent output-row projection partition**: synthetic CPU math for GEMM row split, GDN z, QSA query/gate, shared expert gate/up, odd widths, quant-block/bias/scales and downstream full state. RIR pass and CIR fake device first-consumer barrier. Proof: same semantics with GPU-only and mock 2-device variants, missing NPU fallback works. **Kill** if joint row map/state cannot be verified or compiler requires Qwen model-specific hardware code.

**H27 — XDNA2 program/toolchain and Rust source feasibility:** pinned fork's C++/Python AIE `v8bfp16ebs8` 4096×2560×1024 `.xclbin` is a research baseline **not a Rust-written kernel**. Evaluate a Rust-authored AIE/MLIR source generation and XRT driver FFI; document ABI, compile path, licenses, shape support, build reproducibility, whether compiled bitstream identity validated. **Blocked** if strict Rust source cannot be met; do not import source as a hidden wrapper.

**H28 — Cross-API memory/DMA safety and concurrency:** fake HIP mapped host BO/XRT import, generation-tagged buffers, NPU queue depth four, GPU stream host submit/wait callbacks (as source reference), first-use event join, short read/timeout/lost device/cancel and quarantine. Physical concurrent NPU/GPU firmware/AGESA/FCLK/XRT/ROCm soak on isolated device only. **No automatic privileged fabric/power changes or system hang attempts**; one crash invalidates auto-enable qualification.

**H29 — Source benchmark independent replication:** Qwen3.8-Flash-Next **UD-IQ4_XS** 16K 1698 GPU-only vs 1821 GPU+NPU prefill author claim, 32K 1783 NPU, 40K cold served 1637 NPU, 8192 chunk source conditions. ABBA same held fabric clock, XRT performance/NPU wake state and power/thermals in *both* arms, separate GPU-only default-clock comparison; measure TTFT/P95, actual overlap/stall, pinned memory and idle energy. Measure NPU projection subsets separately and zero-benefit modes, not just one headline. **No unqualified peer-engine ranking**.

**H30 — Mixed-compute numerical and quality:** v8bfp16ebs8 block-float nine-byte/eight-value encode/decode as separate ComputeEncoding, Q8_0 checkpoint stored quant unchanged. Compare source CPU/GPU outputs, per-layer full logits, GDN state/attention selected IDs, 30 greedy smoke + hundreds of held-out, exact sampler/MTP parity, longest context. Reject auto-enabled offload if quality changes outside approved validated numerical policy.

**H31 — General TaskProgram extension:** upstream Gufo Qwen3-ASR, Qwen3-TTS, Qwen-Image-2.1, MiniMax H3 research; model-independent task stage loop (ASR encoder, TTS vocoder, image denoise, video frame) compiles to SIR/KIR/CIR and ResourceArbiter, shares DeviceWorkers but has unique TaskProgram state/output. Test one fake nontext task plus one real accepted model reference when available; no claim all media models are supported before independent quality/latency tests.

**Evidence keys:** exact fork commit `8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`, Gufo upstream head `f37fb012281bbd58054b1082933153bbec303106`, source model/quant/compiled NPU program hashes, AIE toolchain, XRT/ROCm driver, BIOS/AGESA and held/default fabric clocks. Distinguish discrete GPU+iGPU expert ownership (a discrete-GPU+iGPU expert-ownership approach) from same-SoC NPU+iGPU projection slicing (Gufo fork). Sources in the two new `.agents/docs` files. Research and plan only, no H26–H31 runs completed.