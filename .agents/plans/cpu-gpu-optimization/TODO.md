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
