# Prior-art/historical archive — RELEASE_ACCEPTANCE_GATES.md

**Historical v2 planning snapshot, archived 2026-10-12. NON-NORMATIVE RESEARCH.** External named projects and benchmark figures are preserved as source research, while actual current design and phase gates use project-neutral contracts at `../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`, `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`. User-provided instructions and decisions remain separately in DECISION_REGISTER. Never use historical source-project APIs as the new runtime's type or module names.

---

# zerogravity — Acceptance, verification and release gates

**Scope:** binds every proposed roadmap phase to objective, independently repeatable evidence. This document is a proposed governance standard, not an assertion of passing tests.

## A. Project lifecycle gates

| Gate | Earliest phase | Required evidence | Failure |
|---|---|---|---|
| G0: source/toolchain | P00 | fresh checkout builds, pinned tools, attribution and decision log | cannot reproduce |
| G1: kernel feasibility | P01 | real physical `gfx1151`, numerical references, disassembly/profiling, documented performance gap | no competitive Rust codegen path |
| G2: semantics/state | P02–P04 | CPU reference with tensors+effects, shape errors, snapshots and rollback | failed oracle/replay |
| G3: compiler legality | P05 | differential graph transformations, no incorrect reorder, fallback transitions | changed semantics / bad alias |
| G4: real model | P06–P08 | GGUF/safetensors and tokenizer parity, end-to-end dense outputs, CPU/GPU comparisons | plausible but wrong tokens |
| G5: sessions/serving | P09–P10 | streaming/tool/integration tests, memory safe, persistent context | hidden state or OOM corruption |
| G6: hybrid/MoE | P11–P12 | long-context GDN/KDA/indexer parity, route/cache miss tests | state corruption/unsafe eviction |
| G7: Strix specialist | P13–P14 | real Qwen/GLM reference and matched comparison, cold/warm | no model/quality proof |
| G8: speculation | P15 | rejected-prefix/RNG distribution, latency and benefit, adversarial stop | target mismatch |
| G9: high throughput | P16 | concurrency, fairness, p95 and no sequence cross-talk | stall or state leak |
| G10: breadth/scale | P17–P19 | per-model feature matrix, multi-device failure recovery | claims unsupported |
| G11: release | P20–P22 | signed provenance (policy TBD), performance and quality reports, security audit, QA | regression/critical risk |

## B. Correctness corpus levels

**Level 0:** parse/graph/operator shape/type/canonical reference with unit + property tests. **Level 1:** GPU microkernels against CPU FP32/FP64 as appropriate, quantized block exact decode, output tail, masks and invalid selectors. **Level 2:** synthetic complete model with layer-wise tensors/logits and state transitions. **Level 3:** official checkpoint and known tokenizer/template, 100+ held-out continuations and long-context adversarial fixtures. **Level 4:** HTTP/agent lifecycle, tools/reasoning, concurrent sessions, serialization/restore/cancel/VRAM exhaustion. **Level 5:** distributed two-rank partial failure and integrity.

A source project may report exact-logit equivalence to its own previous version; that is not automatically parity to an official HF model. Test both. Use same quant/artifact and normalized template whenever comparing runtime. Never claim quality preserved by small one-token agreement tests alone.

## C. Mandatory validation for execution plans

For every plan region record: semantic op IDs, value/state input/output types, tensor shape/strides, dtype & quant zero point/scales, allowed alignment, causal position, scratch bytes, all device/host leases, graph-capture flag, in-flight events, numerical policy, supported max batch/context and fallback plan. Programmatically reject violations. Test forced operator fallback and interrupted graph rollout (host/device failures) at legal commit boundaries.

## D. Resource safety minimum

- preflight total resident/model dense bytes + KV/SSM/indexer + optimizer and graph scratch + pinned staging + page/memory headroom;
- hard reservation per admitted request; dynamic scratch borrowed only via lease;
- device code never accesses expert slot after its generation changed;
- detect physical memory underestimation and bail safely before OS OOM where possible;
- user is warned before opt-in kernel/firmware memory configuration changes;
- corrupted SSD read invalidates request/state frontier instead of returning stale data;
- model-unload and server restart free or intentionally reuse all allocations;
- no unauthenticated public network exposure default.

## E. Serving service-level verification

For each supported model/hardware report max context, maximum admitted concurrency, typical TTFT P50/P95, intertoken latency P50/P95, aggregate tokens/s, peak RSS/HIP allocated, startup warm/cold, memory fit and quality contract. No universal SLO numbers defined before physical benchmarks; they are *open decisions*. Fault tests must cover malformed requests, stream resets, cancellations, tools, reasoning boundaries, slow clients, token limits and invalid cache files.

## F. Report schema recommendation

```json
{
  "project": "zerogravity",
  "result": "pass|neutral|fail|inconclusive",
  "phase": "ZG-Pxx",
  "date_utc": "...",
  "commit": "...",
  "model": {"id": "...", "artifact_sha256": "...", "quant": "...", "tokenizer_sha256": "..."},
  "hardware": {"gpu_arch": "...", "ram_bytes": 0, "driver": "...", "power_limit": "..."},
  "runtime": {"rustc": "...", "compiler": "...", "backend": "...", "plan_hash": "..."},
  "workload": {"prefill_tokens": 0, "prefix_depth": 0, "output_tokens": 0, "concurrency": 1, "sampling": "..."},
  "performance": {"ttft_ms_p50": 0, "itlat_ms_p95": 0, "prefill_tps": 0, "decode_tps": 0, "aggregate_tps": 0},
  "quality": {"reference_id": "...", "logit_rms": 0, "greedy_match": "...", "sampling_test": "..."},
  "memory": {"rss_peak_bytes": 0, "gpu_peak_bytes": 0, "expert_hits": 0, "ssd_bytes": 0},
  "repetitions": 0,
  "raw_artifact_paths": [],
  "known_limitations": []
}
```

The JSON above is an illustrative proposed schema and not a filled result. Unknown fields must be null/omitted in real reports, not misleading zeros.

## G. Documentation handoff closure

At phase finish update `START_HERE` references if changed, roadmap status, decision register with new decisions, relevant `.agents/docs`, benchmark evidence, risk register, and `MASTER_HANDOFF`. Verify live tree state and final test commands; cite exact files and commits. Agents must never say phase done without an executable proof and list of known blockers.

## 2026-10-10 integrated v2 acceptance gates

**New G0 baseline:** `G0-A` source/CPU artifact/math/state reference; `G0-B` physical gfx1151 Rust kernel feasibility and CIR event/resource safety; `G0-C` complete checkpoint→SIR→RIR→KIR/CIR→EngineCore→correct token vertical slice with forced per-region fallback. All are documented in `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`. Historical pure G0 kernel and tokenizer tests remain necessary, but a giant precompiled crate skeleton alone does not pass.

**Stage-specific proofs:**
- Compiler: illegal effect rewrite rejected, quant+position typed, no hidden stateful op removal; compiled bundle must not interpret SIR per output token.
- Device: Rust-authored kernel and target ISA reference/quality, bounded compile/JIT, Capture/Eager numeric/scratch/event compatibility, safe fallback on unavailable specialized region.
- CIR: async buffer lease generation, IO/transfer timepoint, short read and device loss abort; no in-flight expert overwrite, resource budget violation or sample/recurrent partial commit.
- State: 3+ hybrid groups cached frontier, full canonical immutable prefix, disk snapshot schema/hash (not broken original 48/54 byte), opt-in privacy and model/quant/tokenizer/adapter/modalities/salt identity.
- Serving: EngineMsg/StepPlan/WorkerDelta epochs, native vs ordered batching, cancel/stop/tool/grammar, no output ahead of committed state, p95 under C1..C32.
- Product: tokenizer, Jinja template, grammar, ModelStore, embedding/rerank runner, LoRA and multimodal only marked supported where conformance passes.
- Kernel tuning: eager-vs-captured actual serving tactic, micro plus whole request and quality band; no unbounded search.
- Distributed: peer model/state epoch handshake, partial collective fail-closed, no unauthenticated public transport by default.
- Research claims: source pins verified and matched checkpoint/quant/power; no universal superiority promised.

**Two corrected legacy errors:** research disk-KV fields total 54 bytes despite a 48 byte label and magic cannot be five characters in u16. Research line for 8×L40S DwarfStar DeepSeek TP cannot be a measured GLM-class benchmark. These are release/source integrity issues, not optional style edits.

**Release phases:** G1 portable (known dense model), G2 generic serving, G3 Qwen/GLM/DeepSeek extreme, G4 breadth/parallel. SLO percentages from earlier research are historical proposed target floors, **not** user-approved achievements; use meaningful repeated A/B and caveats. No gate marked VERIFIED before repository evidence.


## 2026-10-12 acceptance: optional XDNA2 and TaskProgram device/semantic qualification

**XDNA2 always optional:** no GPU/CPU release phase depends on an NPU or XRT. A hardware-qualified NPU region must pass (A) Rust-owned AIE source/toolchain and license rule, (B) full mathematical projection row-map/quant/compute precision and state parity, (C) CIR GPU↔XRT BO mapped/pinned lease/event/quarantine under cancellation and timeout, (D) controlled physical SoC firmware/driver/FCLK/power stability and long soak without hard hang, (E) same-clock GPU-only vs GPU+NPU model quality and matched full-request performance win, (F) clean GPU-only fallback for absent/misaligned/unqualified accelerator. No amount of generated token speed waives A–D.

The Gufo fork reports real **whole-machine hangs** when fabric clock transitions during concurrent NPU/GPU; **automatic** hardware tuning/activation is **not permitted** based on a handful of successful author runs. Do not call root shell/sysfs/systemd/IOMMU/BIOS changes from runtime. On unqualified hardware mark `NPU unavailable/experimental`, not performance-qualified, and execute GPU-only. A software timeout cannot recover a SoC hardware freeze; physical safety trials isolated and manually consented.

**Numeric/model truthfulness:** source `UD-IQ4_XS` 16K **1698→1821 prefill tok/s** is an external reported result, not reproduced by zerogravity. It uses BFP8 NPU *compute* format, not necessarily bit-equal to BF16/F32 kernel arithmetic. Qualify full logits, GDN/QSA index state, MTP sampled distribution and quality with original checkpoint. Require real model+hardware evidence before `NpuKernelProvider` becomes automatically selected on any profile.

**TaskProgram capability:** ASR/TTS/image/video model families carry their own task state semantics, encoder/acoustic/vocoder/denoise/temporal references, output format and privacy/security; task may be `reference`, `experimental` or `qualified` per checkpoint/backend/quant. Upstream Gufo implementation is evidence only, not proof of zerogravity production support. See `../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md`. All original G0–G4 acceptance gates remain applicable.