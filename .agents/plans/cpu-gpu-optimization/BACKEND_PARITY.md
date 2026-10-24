# Vulkan / CUDA / HIP parity

Owner directive,23Oct: Vulkan reaches CUDA parity; AMD uses ROCm/HIP.
Live source/tests/results outrank this implementation map. No backend parity
or new model qualification is established by this document.

24Oct current: approved Qwen/Gemma HF sources and original references are now
imported on x2. Source20's actual shared HIP full64/full35 lifecycle unit passes;
natural Qwen comparison and learned Gemma PLE-hit coverage advance separately.
The earlier inventory/cache and whole-model blockers below retain their dates.
ROOT reviewed [Astra's next Vulkan unit](../../runs/cpu-gpu-optimization-astra-portable-next-20261024/DESIGN.md):
reuse full Gemma E2B traversal/oracle; add priced source-owned pipeline
preparation/retirement, full-K output-row tiling and effective binding limits.
Pi implements in a MAIN-readonly isolated candidate; no new Vulkan model or
performance qualification yet, and no heavy local build during quiet timing.

## Shared components and remaining implementations

| Component | Current shared scope | Remaining backend work |
|---|---|---|
| Source identity, authenticated windows, bounded small-window reuse | Core/artifacts, automatic for existing Qwen/Gemma callers | Same source/mutation/pressure/cancel checks on each actual route |
| Complete-cost residency policy | Pure host+device cost selection; qualified-provider availability separate | Each adapter derives its whole live inventory; unavailable implementations stay unavailable |
| Physical identity, arbiter, queue, leases, state roots, completion/quarantine | Existing common engine | Qualify actual provider behavior; HIP and Vulkan on one PCI GPU share one physical budget |
| Original-flight access and finite-certificate lookup | Shared resident/native adapters; full owners and exact ranges retained | Compile and exercise affected actual backends; CUDA proofs do not qualify HIP/Vulkan |
| Immutable prepared launches and RMS/GDN storage reuse | Implemented through existing CUDA launch adapters | HIPRTC and SPIR-V implementations under the same semantic plan and arithmetic contract |
| Source-derived BF16 matrix/row transport | Common geometry and verified-residency capsule; CUDA adapter and bounded actual HIP/Vulkan transport | Larger binding/page limits and actual learned-source consumers require their own admission and qualification |
| HF BF16 RMS, projection, attention, GDN and whole-model paths | Qualified finite CUDA source/profile tuples | Actual HIP/Vulkan implementations; preserve materialized casts/reductions and exact source/state semantics |
| Serving and benchmark profiles | Existing workers/publication, finite short-HF CUDA profile | Same source, natural feedback, full logits/state, cancellation/pressure/fault/API checks before timing |

Preparation chooses a qualified implementation once. Fresh attempts still check
source, session/root, physical generations, actual operands, epochs, cancellation
and completion. No second model loop, allocator, publisher or execution IR.
An advertised arithmetic capability is insufficient to select a kernel.

## Historical pvl-x2 readiness and subsequent finite results

- Actual PCI1002:1586/Radeon8060S/gfx1151. Existing amdgpu/KFD access;
  HIP wave32. Vulkan RADV reports default subgroup64: layouts cannot be retagged.
- Existing ROCm7.2.4 runtime/HIPRTC, hipClang/LLVM22, device libraries and BLAS;
  existing Mesa26.1.8/Vulkan loader/tools. Use per-job SDK paths before the first
  cached loader lookup; no persistent host or privilege changes are needed.
- Reported GPU pool96GiB, CPU-visible RAM about31GiB, disk about1.8TiB free.
  These are totals/readiness facts, not free-memory or model-fit proofs. Do not
  add GPU/GTT/host capacities into a fictitious combined budget.
- No current repo/approved HF cache at usual remote paths. Existing Ollama GGUF
  models are not approved HF source equivalents. Preserve exact managed source
  identity during any later authorized transfer. No new acquisition performed.
- Read-only inventory alone establishes no GPU execution, numerical or performance
  qualification. Actual bounded tests follow below. Retain gfx90c launch hold.

Current execution checkpoint: current MAIN202 +the tiny qualification delta
builds offline on pvl-x2 with installed HIP bindings53211; source/process/EOF
closure passes. First HIP attempt refused before source/client/kernel; Vulkan
found no adapter. Actual dlopen diagnosis: installed LLVM22 requiresGLIBC_2.44,
initial test process used2.42. Both failures closed cleanly and are preserved.
Existing matching glibc2.44 loader plus per-job installed library paths now
prove simultaneous HIP/HIPRTC/Vulkan/RADV load in one process. Runner-only
successor records exact loader/binary/path identities; no system/SDK/binary
mutation. Final six actual tests PASS on the same rebuilt qualification binary:
HIP/Vulkan healthy, each unknown, both shared-PCI fence directions. Initial
unknown test exposed a shadowed prior CPU oracle retaining Source; explicit
prior-owner drop fixed the fixture, preserving all weak/host/quarantine checks.
ROOT independently checked exact one-test markers, all owners/source/binary/EOF
and the final remote two-file source hashes against the stable donor. Two test
paths integrated MAIN/ISO after CUDA source closure; a declaration-format-only
correction passed assembled gate8b. Frozen CUDA binaries remain unchanged.

## Ordered complete units

- [x] Explicit compiled-provider/runtime admission before client/allocation;
  missing HIP libraries return a normal error. No automatic Vulkan/CPU fallback.
  Seven-path cold source merged MAIN/ISO; missing-library public/private tests
  and strict HIP+Vulkan/no-provider checks pass. Assembled isolated gate7 PASS
  1715Rust/0failed/88ignored/266suites plus scripted checks. Admission grants no
  arithmetic support; the actual tiny foundation is scoped below.
- [x] Foundational actual Q8 embedding→FP32Conv stateful execution, independent
  roots/completion/pressure/cancel/unknown quarantine on pvl-x2 HIP and Vulkan.
  Actual same-PCI dual-provider cohort must share one physical cap/fair queue;
  unknown work through either provider must fence both. Existing conservative
  policy may retain both provider envelopes; no new recovery policy. This
  tiny source establishes no BF16 HF primitive or learned-model qualification.
  Actual healthy compares32outputs/96history/three producer observations,
  independent initial/cached roots/stale/foreign/queue cancel/pressure/output
  lifetime and finalresident/events/physical/fixturehost0. Unknown each retains
  24real handles/64MiB; each cross-direction retains each provider's own handles
  and both128MiB envelopes until process retirement. HIP reports103079215104B,
  Vulkan79807275008B; both use the same256MiB operator cap/physical key/queue.
  [Actual source, results and limitations](../../../../zerogravity-worktrees/qwen27b-attention-tail-20261022/.agents/runs/model-execution-portable-service-qualification-20261023/RESULT.md).
- [x] Bounded actual BF16 transport through original HIP/Vulkan services on x2.
  Reviewed11path transport plus2path range correction integrated MAIN/ISO.
  Gate9 passes1715Rust/0failures/allscriptchecks before the range correction;
  corrected assembled gate10 also passes1715/0/allscriptchecks. Initial HIP whole-table comparison
  passed, then the whole-handle guard refused allocator views before row work.
  Original failed -1 remains immutable. Correction reuses original BufferRange
  exact used spans/service checks, rejects duplicateIDs/overlap and accepts
  disjoint views; no source/kernel/cap waiver. Six transport/range+two existing
  guards, strictHIPVulkan/CUDA and formatting pass. Corrected -2 actual12 modes
  PASS: both providers healthy, first/final upload unknown, row unknown, invalid
  device ID, and both shared-PCI quarantine directions. Three geometries
  1029x1024/685x1536/519x2048 cover all65536 raw BF16 patterns and repeated/
  first/middle/last selected rows byte-exact. Healthy lifecycle/root/pressure/
  cancel checks end at0; unknown retains actual handles/source/envelopes until
  original owned-process retirement. ROOT independently checked all12 exact
  test/outcome/cap/source/held-binary/group/reap/EOF records and final quiescence.
  Actual16B views have256B HIP/32B Vulkan backing. This establishes transport,
  not BF16 arithmetic, model execution, serving or performance.
  [Actual source, outcomes and preserved failure](../../../../zerogravity-worktrees/gemma4-text-end-to-end-20261022/.agents/runs/model-execution-portable-bf16-matrix-ranges-source-20261023/RESULT.md).
- [ ] Portable arithmetic assembled integration: existing recorded typed embedding→ordinaryRMS→
  projection slice, same ResidentProgram traversal/service/flight, explicit
  TorchCpuBf16Avx2MixedSourceV1 contract. Reuse the existing64lane projection
  shader; preserve every RMS materialization/reduction and check actual bits.
  B1/H1024or2048/W<=32MiB, provider64MiB/physical256MiB/host128MiB diagnostic
  bounds; no model/state-root/serving grant. CUDA SM86 rsqrt/native profiles
  remain separately unavailable. ROOT reviewed runtime/reference source;
  strictHIP/Vulkan/CUDA and focused guards pass. Both authentic pinned public
  HF/Torch captures plus independent validators pass all11 boundaries. Actual
  ten-process HIP/Vulkan cohort now PASS: both healthy H1024/H2048 exact public
  boundaries plus private CPU stages/packed RMS/FP32 projection,3nodes/28
  dispatches/17readbacks/callback1/finalresident-event-physical0. RMS/projection
  unknown cuts on each provider and both shared-PCI directions retain actual
  source/234664B/pending1/unpromoted outputs/64or128MiB envelopes;0D2H/Ready/
  callback. ROOT independently reproduced exact collector/outcome/source/
  binary/reference/cap/test/group/reap/EOF checks; finaljobsnone/VRAMbaseline
  observed, notpeak. Reviewed21paths merged MAIN with freshmtimes/backups;
  ISO held for CUDA diagnostic. Next combined fullgate remains pending.
  Finite synthetic3node CPU-profile proof only; no broad32MiBgeometry, learned
  model, CUDA-specific profile, serving or performance grant. Full model parity
  needs its actual backend/reference contract, not a provider retag.
  [Actual arithmetic results](../../../../zerogravity-worktrees/qwen27b-attention-tail-20261022/.agents/runs/model-execution-portable-bf16-arithmetic-source-20261023/RESULT.md).
- [ ] Whole-model HF and serving profiles on each backend, with full vocabulary,
  every state domain, private natural feedback and independent concurrent roots.
  [Gemma E2B whole35 bridge design](../../../../zerogravity-worktrees/gemma4-text-end-to-end-20261022/.agents/runs/model-execution-actor-review-gemma-gpu-bridge-20261023/GEMMA_GPU_BRIDGE.md)
  maps direct gamma1536/grouped256/512, actual call-M/PV contraction, source
  row/full-K tiles and numeric-zero-tie262144 greedy into the original engine.
  Reviewed13path source/call/observation/tile contract checkpoint merged MAIN;
  core183/Gemma32/Qwen79/strictClippy/fmt pass. Original whole35 traversal,
  authentic gamma/unit-V roles, callM/projection/PV geometry and <=1MiB frame
  recipes preserve old profiles. Inventory prices one4MiB call; every other
  simultaneous call/frame/native owner still needs actual admission. Lowerers
  and full model/state/physical qualification continue in isolated worktrees
  from current MAIN+portable21. No second traversal/IR or monolithic fit grant.
- [ ] Before any large Vulkan matrix consumer, use the device's effective buffer
  and storage-binding limits. CubeCL's Vulkan index64 path can expose heap size
  as max_page_size; that is not a single-binding limit. Existing actual
  transport<=3MiB and arithmetic<=32MiB callers have independent small bounds.
  Gemma E2B PLE table262144x8960 BF16=4697620480B exceeds x2's effective
  0xfffffffc buffer/0xffffffff storage-binding bounds; use authenticated
  selected-row reads or checked paging. No monolithic admission/fit claim.
- [ ] Product execution/declarative packaging must use a mutually compatible
  host-libc/compiler/backend SDK closure. Preserve the actual per-job loader
  requirement in qualification; successful prefixed tests do not qualify a
  differently launched run-rocm/serving binary. Avoid global environment repair
  or repository-external configuration changes without applicable authorization.
- [ ] Matched performance after correctness. Existing owner order remains
  .8BC1+C4 superior → C8 superior →2B; Qwen27B/Gemma integrate independently,
  with their GPU qualification/benchmarks on pvl-x2 HIP. No timing promotion
  from compilation, missing reference runtime or an unmatched source/codec.

Cross-backend parity includes supported tasks and failure behavior. Numerical
profiles, compiler/device versions, measured memory and performance remain
explicit per-target evidence; an implementation flag never substitutes for it.
