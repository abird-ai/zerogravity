# ZeroGravity current comprehensive handoff — 15 Nov 2026

Canonical cross-window entry point for `/home/pvl/spaces/abird/src/zerogravity`.

Full dated takeover snapshot: [`FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md`](FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md). The 15 Nov pack embeds the complete 14 Nov baseline so the October→November design and implementation history remains available from one file.

This file is the complete takeover map and a pointer into maintained source, tests
and plans. It is deliberately **not** the retired monolithic SHA/archive/evidence
handoff model. Git history retains superseded checkpoints; current source, tests
and live TODOs are authoritative.

## Resume here

### Repository/topology checkpoint

- **Repository/source authority:** use `dotlink---l5`; work only in
  `/home/pvl/spaces/abird/src/zerogravity` on `main`. Do not create another Git
  worktree.
- **AMD large-VRAM execution/test host:** use `dotlink---x2` only for running and
  testing, in the same path `/home/pvl/spaces/abird/src/zerogravity`. Do not use
  X2 as an independent source-authoring branch; source changes land on l5 first.
- Latest signed **implementation/qualification** checkpoint is
  `36253481d5e26d27ffb577187b21c338a211beaf`
  (`test(cuda): bound temporal qualifier host state`). Parent `12f7840`
  generalized the retained temporal qualifier across B2/B4/B8; this follow-up
  keeps full B1/CPU qualification at bounded host-state footprint by checking
  each row against a short-lived oracle instead of retaining B8-wide all42
  snapshots. Exact-current original0.8B B4 and B8 full-mode runs now close the
  B1 cross-check in addition to the native CUDA+CPU constituent. Production
  policy is still unchanged. Parent `1e0f4ebfc5b9650341814ad669d4101aea717623`
  (`feat(serving): enable qualified temporal B2 prefill`) remains the production
  B2 enablement: only B2 selects 4-token temporal prefill, while B4/B8 retain
  their accepted one-token policy until the remaining production gates close.
  B2 production HTTP still proves prompt10 executes as native chunks4+4+2 with
  three publications, two state-only + one greedy, exact CPU text/usage, zero
  failures/refusals and final41MiB. Parent `9dab004` added the reusable temporal candidate-chain
  primitive; `7dea460`/`ba7e376` establish fair aged decode/prefill scheduling.
  Earlier `9d6613c`/`599cb28` graph selector/timing remain accepted, but the
  shrinking B2/B4/B8 graph gain does not justify public graph serving; ordinary
  serving remains Native.
- Prefix-scheduling cold reconstruction is now complete for **0.8B and 2B**.
  Fresh-process/no-request-warmup ABBA compares accepted max32-prefix `d0a6084`
  against native teacher-prefix `f788929` with symmetric benchmark-only exact-SM
  cubin compatibility matching later `6a5c610`. 0.8B completes 32/32 exact requests:
  mean wave 17.4307s->15.4529s (-11.35%), TTFT p95 15.0368s->14.9527s and
  delivery-ITL p95 0.8188s->0.7804s. 2B also completes 32/32 exact requests with
  zero timeout/refusal/failure/cancel/discard: mean wave 29.5280s->25.0937s
  (-15.02%), rate 2.1693->2.5512tok/s (+17.60%) and delivery-ITL p95
  1.5598s->1.3234s (-15.15%), while TTFT p95 regresses 23.1625s->23.8931s
  (+3.15%). Before measurement, staged 2B C1/B2/B4/B8 diagnostics proved the
  rebuilt old provider healthy through exact B8; two earlier first-work process
  disappearances remain incomplete compiler-preconditioning attempts, with no
  cgroup OOM evidence. Every accepted physical scope retires to 41MiB. The original
  frozen benchmark binaries/workload are gone, so both cells use retained source-real
  `f788929` HTTP prompts and are reconstructed diagnostics, not byte-identical
  historical reruns, source-correct-goodput evidence or vLLM claims. The P12 prefix
  scheduling performance item is closed; continue with the remaining fused-serving
  paged-state/overlap/fairness and genuine equivalent-source vLLM work.
- **P12-C fair-interleaving sub-slice is accepted at `7dea460`.** Shared
  `zerogravity-engine` fair-round scheduling adds finite `round_tokens` and
  `prefill_age_rounds` above the accepted source/phase/frontier wave builder:
  every eligible decode row executes first, young prefill retains exact
  state/source owners while deferred, and the oldest prefill must make progress
  after two deferred rounds. Admission requires room for one maximum prefill wave
  plus every active decode row. Executor/Ready/publication/cancellation authority
  is unchanged. 18 wave tests, 14 worker tests, strict engine clippy and
  production CUDA/CLI composition pass. Physical unequal-arrival HTTP passes both
  approved 0.8B/2B CUDA sources: B arrives only after A's first actual ID; first
  shared B2 is no earlier than A index 4; both sockets drain; CPU IDs/text/usage
  remain exact; no failed/refused/cancelled/discarded work; final GPU 41MiB.
  Every `scripts/check.sh` command has a terminal-green result on these exact
  bytes under isolated `native-gpu`, including feature-boundary, CPU workspace,
  serving/lifecycle, acquisition and release/packaging blocks. Repeated attempts
  to keep the whole shell wrapper in one connector job disappeared during normal
  cached builds without a compiler/test error, surviving child, cgroup OOM or GPU
  residue; those wrapper attempts remain incomplete. The following P12-C slice
  is the actual multi-token temporal prefill work described below.
  **0.8B B2 temporal prefill is functionally/physically accepted in the
  debug qualification scope.** The retained `zerogravity-gpu` example is now the
  width-parameterized `temporal_prefill`: B2 still proves abandoned 4-token
  rollback, one-root 4-token state publication, one-root 2-token terminal greedy,
  CPU/B1/full-all42 exactness, exact packet/producer and final reserved0. The
  production HTTP path still passes prompt10 -> native chunks4+4+2 with CPU
  ID264/text ` a`, usage10/1, exactly3 B2 publications (2 state-only +1 greedy),
  zero failures/refusals and final41MiB.
  **15 Nov follow-up:** the same retained qualifier isolates width-specific CUDA
  work while full mode keeps B1 mandatory. Original0.8B B4 and B8 now each
  physically pass full mode on the exact `3625348` source: completed private
  4-token chains can be abandoned with roots/all42 unchanged; 4-token state and
  2-token greedy chunks each publish one root; every native row matches CPU and
  B1 all42 state plus greedy bits; packet producer lineage is exact; and final
  reservation returns to zero. B4 retained-band bytes were 1,230,916,660; B8
  1,802,356,148. Earlier full B4/oracle attempts and one exact-current B4
  confirmation disappeared before temporal work, with no Rust diagnostic, OOM,
  live child or GPU residue; the bounded retry passed and the disappeared runs
  remain incomplete, not failures. Two exact-current B2 qualifier confirmations
  likewise disappeared just after session setup before temporal work; the
  already-accepted B2 production/physical evidence is unchanged and those retries
  are not new failures.
  Production `NativeText` still selects temporal4 **only for B2**: B4/B8 keep
  their accepted one-token/unequal-frontier policy until their real production
  HTTP cells close. **2B temporal qualification is next and remains unqualified.**
  Do not widen B4/B8 beforehand through `native_rows` alone, because that could
  grant the 0.8B permit to an unqualified 2B route; either qualify the affected
  2B cells first or make the promotion source-bound. These physical
  proofs use the debug Rust profile; canonical thin-LTO/codegen-units=1 B2 is
  still open after external build losses without Rust/OOM evidence. **Next:
  qualify 2B temporal independently, then close B4/B8 production HTTP and widen
  only policies whose affected source/model cells are physically accepted. Retry
  release-profile B2 when the build host survives.** Paged hybrid state/COW and
  event-backed overlap remain separately open. **Post-check L5 baseline watch:**
  after the accepted B8 process had exited
  and no ZeroGravity/native-gpu process remained, the isolated connector view of
  `nvidia-smi` later reported ~1,785MiB/P0 instead of the initial 41MiB, while
  NVML/pmon exposed no process owner and the kernel log showed no Xid/OOM. A
  subsequent passive resample returned to 41MiB/0% without reset or intervention.
  Treat the transient as unexplained rather than a ZeroGravity leak; preserve any
  unrelated host owner and continue resampling the physical baseline before CUDA
  work.
- Earlier key implementation checkpoints include `f1ce532`
  (exact graph replay storage/access authority), `b0cf0b2`
  (`fix(cache): avoid duplicate Qwen snapshot hashing`), `134a571`
  (`fix(hip): isolate CUDA private KV qualification`), `874e67d`
  (`feat(cuda): retain shared-input projection pair`), `c282f20`
  (`fix(hip): restore source profile compilation`), `5bf440d` / `db7be60`
  (production CUDA graph substrate/resource admission), `6a5c610` (driver-safe
  nonpersistent NVRTC) and `81b7c71` (explicit graph-resource lifetime witness).
  Their docs checkpoints through `b12e8f5` are pushed.
- Pushed `origin/main` is `7dea460a80b420d0ce49a60f6c860b3502c67c38`
  (`feat(serving): age prefill behind decode`). It includes the earlier selector,
  graph timing and prefix-scheduling checkpoints that were previously local.
  This acceptance refresh may add one docs-only local commit; never push it unless
  the user explicitly asks. X2 is synced to the exact current local tip after each
  checkpoint even when its remote-tracking ref is stale.
- Before any new physical X2 run, sync it to the exact current l5 `main`
  bytes/commit without using a push merely as transport, then revalidate
  status/process/GPU state.
- Local docs `5c8f143` recorded the host-global X2 KFD observation and `3cadc3b`
  records rejection of the ordinary-Qwen `874e67d` consumer after completed
  loaded-model A/B; the primitive itself remains retained. Later host context
  corrected `5c8f143`'s ownership inference: X2 is shared with Ollama/model
  workloads, so that KFD owner is not attributable to ZeroGravity without host
  process evidence. `5bf440d` is the first broader post-micro-optimization CUDA
  implementation checkpoint.
- Important earlier accepted commits:
  - `60b048d` — canonical-boundary private B8 GDN recurrent core accepted.
  - `775dfa6` — ragged private B8 KV transition accepted.
  - `c12acab` — grouped B8 recurrent transport primitive retained, serving
    consumer rejected.
- At this refresh no Cargo/model process is active in the connector-visible
  command namespaces and l5 RTX3060 is idle at ~41 MiB. X2 is a **shared GPU host**
  whose baseline must be resampled every run: the latest observation is
  total103,079,215,104 bytes VRAM, used272,351,232/busy0 with no KFD owners, while
  earlier observations included unrelated Ollama/model residents around31GB. Do not
  infer ownership from a prior baseline or require an empty GPU. Record the current
  baseline, stay within available capacity, verify only our incremental VRAM/queue/
  process delta retires, and never reset/reclaim/restart/kill other owners without
  explicit authorization.
- Root `README.md` is user-authored. Leave it unchanged unless explicitly asked.
- Keep one persistent `main` checkout per host. Reuse canonical target/cache state;
  do not recreate old worktree pools or `.agents/runs`.
- Commits are signed. Inspect `git diff --cached --stat` before every commit and
  stage only owned paths.
- On l5, `git log --show-signature` reports a good ED25519 signature while also
  printing a missing local allowed-signers principal warning. That warning is not
  a bad signature.

### Read order / authority

A new agent should read in this order:

1. Current source/tests for the surface being changed.
2. `AGENTS.md`.
3. `.agents/plans/README.md`.
4. `.agents/docs/CURRENT_HANDOFF.md`.
5. `.agents/plans/TODO.md`.
6. `.agents/plans/model-execution/TODO.md`.
7. `.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md`.
8. The owning plan(s), especially:
   - `serving-and-speculation/TODO.md`
   - `cpu-gpu-optimization/TODO.md`
   - `memory-and-streaming/TODO.md`
   - `pipeline-composition/TODO.md`
   - `numerics-and-context/TODO.md`
   - `cli-device-routing/TODO.md`
   - `model-acquisition/TODO.md`
9. Relevant architecture docs under `.agents/docs/`.
10. Git history for removed experiments and superseded handoffs.

Current source/tests and live TODOs outrank this handoff if main advances.

### Tracking / handoff method

The old generated handoff/SHA/archive workflow is retired.

- `CURRENT_HANDOFF.md` is the canonical cross-window pointer.
- `.agents/plans/TODO.md` is the global queue.
- Per-plan TODOs are the detailed accepted/rejected/open-work ledgers.
- Git history is the archive and provenance record.
- Do not recreate `.agents/runs`, source snapshots, monolithic receipts, per-file
  hash catalogues or durable side-worktree coordination state.
- Actual model/content identities, numerical contracts and physical acceptance
  limits still belong in source/tests/plans where semantically required.
- Historical failures stay failures. A later pass/cleanup cannot relabel them.

## Owner direction / working style

The user has repeatedly steered the project toward implementation-first,
shared-primitives-first work:

- Prefer **implementation → quick real-model feedback → controlled comparison**.
- Solve cleanly at shared layers when the semantics genuinely benefit multiple
  models/backends/tools.
- Model/source policy belongs in model/source layers. Reusable numerical, layout,
  transport, state and physical execution primitives belong in shared runtime/GPU
  layers.
- Keep independently useful model-neutral primitives even when their first Qwen
  serving consumer is rejected; remove the slow consumer, not the clean primitive.
- Exact source/reduction/state/root/resource/cancellation semantics are never
  weakened to chase benchmark numbers.
- Unsupported capability fails closed. Never silently select another arithmetic
  profile, captured case or model-specific fallback.
- Rust preference: explicit ownership, simple robust APIs, zero-cost abstractions
  where natural, targeted hot-path optimization only when repeatably material.
- Do not over-engineer tests, evidence, receipts, pinning or phase granularity.
- Reuse accepted validation for unchanged primitives.
- Validate in coherent checkpoints rather than after every helper.
- Treat fewer launches/loads/instructions as a hypothesis, not acceptance evidence.
- For small expected gains use one-binary/one-model forward/reverse or ABBA
  measurement; cross-process clock/thermal drift repeatedly produced false wins.
- Remove all temporary selectors, profiling hooks, `_tmp` runners and include
  views after the decision is recorded.
- Recent CUDA work has used ~30-minute coherent checkpoints; continue that unless
  the user changes the cadence.
- Do not ask the user to reconfirm already-authorized continuation steps.
- Never push unless explicitly asked.

## Canonical architecture / non-negotiable invariants

ZeroGravity is one Rust-owned execution fabric, not a per-model collection of
engines/servers.

- One semantic/model definition per family/source contract.
- One logical `StateRoot` publication authority.
- Canonical roots remain checkpoint/rollback/publication authority even when
  private/native/derived representations exist.
- One process-shared physical `ResourceArbiter`.
- Device workers own real asynchronous compute/events, but cannot invent logical
  commits.
- Release follows actual completion/fence/event state; completion-unknown work is
  quarantined.
- Source/weight owners survive until the real final consumer retires.
- Cancellation cannot publish partial candidate state.
- Independent sessions/roots remain independent through physical batching.
- Physical batch width is separate from each source call's authenticated numerical
  reduction contract.
- Finite captured cases are qualification evidence, never production selectors.
- Metadata/registry slugs never grant execution support.
- CPU/CUDA/HIP/Vulkan and quant/source profiles remain separate qualification
  claims.
- Performance follows semantic/source/state/ownership correctness.

The Fabric transition is complete and now forms the shared architecture:
`zerogravity-core` owns semantic/state/provider/physical contracts;
`zerogravity-artifacts` owns immutable formats/source/acquisition/store; model
crates own family semantics; CPU/GPU crates own physical providers;
`zerogravity-engine` owns model-neutral runtime/session/scheduling/residency;
`zerogravity-serving` owns reusable protocol/server behavior; `zerogravity` is the
public composition root; CLI stays presentation/arguments/signals. Do not restart
the old Fabric S/F/Q transition sequence.

## Design/spec navigation

Do not read all design docs for every kernel change. Use the relevant map:

- Product/vision: `VISION_SCOPE_AND_PRINCIPLES.md`,
  `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`,
  `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`.
- Engine/IR: `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`,
  `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`,
  `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`,
  `SEMANTIC_IR_AND_COMPILER.md`.
- Model/formats/numerics: `MODEL_SEMANTICS_AND_COMPATIBILITY.md`,
  `FORMATS_QUANTIZATION_AND_LOADING.md`,
  `NUMERICS_QUALITY_AND_BENCHMARKS.md` and the correctness policy.
- Scheduling/state/memory: `SCHEDULER_BATCHING_AND_SPECULATION.md`,
  `STATE_CACHE_AND_TRANSACTIONS.md`, `MEMORY_RESIDENCY_AND_IO.md` and
  `UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md`.
- GPU/device: `DEVICE_KERNEL_AND_CODEGEN.md`,
  `ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md`,
  `OPERATOR_AND_FUSION_CATALOG.md`, `COST_MODEL_AND_AUTOTUNING.md`.
- Serving/routing: `RUNTIME_SERVING_AND_UX.md`,
  `ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md`,
  `MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md`.
- Composition: `COMPOSITION_AND_EXTENSION_CONTRACTS.md` and
  `pipeline-composition/STRUCTURE_AND_NAMING.md`.
- Distributed/multimodal: the distributed-placement/transport docs and the
  universal/realtime multimodal contracts.
- Historical Fabric rationale lives under `.agents/plans/fabric-vnext/` and
  `fabric-vnext-transition/`; those plans are closed.

## Global plan ownership / pending-work map

The global queue in `.agents/plans/TODO.md` is authoritative. At handoff creation
the work ledgers are:

- **serving-and-speculation** — active P1: serving correctness/context fit,
  population/batching/paged-state/speculation, matched vLLM campaigns and shutdown/
  cancellation behavior. Its TODO retains the full serving chronology from the
  initial CPU/OpenAI router through resident GPU cohorts.
- **numerics-and-context** — active P1: independent numerical/source acceptance,
  genuine context progression, reference capture, ordinary CUDA/HIP parity and
  remaining long-context/backend cells. The common acceptance rules are in
  `CORRECTNESS_POLICY.md`.
- **memory-and-streaming** — active P1: exact admission/fit, source windows,
  complete-domain state/prefix/COW/persistence, SSD streaming and residency
  boundaries.
- **cpu-gpu-optimization** — active P1: backend parity and measured performance,
  with correctness-first gates before matched optimization claims.
- **pipeline-composition** — active P1: recursive programs/profiles, component
  reuse, format/model/provider boundaries and production/test separation.
- **cli-device-routing** — active P1: unified run/config/source/device selection,
  exact device identity and future local spill/peer routing.
- **model-execution** — active P2: the detailed CUDA/HIP/Qwen implementation,
  qualification, performance and rejection ledger summarized above.
- **model-acquisition** — active P2: authenticated source location/cache/provider
  flows, HF/Xet/cloud/object acquisition, aliases and source rights.
- **multimodal** — icebox: multimodal execution/product work; do not infer current
  implementation from architecture docs.
- **distributed-execution** — icebox: remote/distributed placement/transport work.
- **platform-and-release** — icebox: assembled platform/release acceptance work.
- **fabric-vnext-transition** — closed: historical architecture/migration rationale
  and acceptance; do not reopen as the current queue.

When work crosses plans, update the lowest owning TODO(s) rather than creating a
parallel status ledger. The handoff is a navigation layer, not a replacement for
those work-item files.

## Current signed implementation state

### Fabric / shared runtime

Accepted milestones include `860ce38`, `fef7123`, `8d761c4` / `8c43844` and
`1c99929`. The transition is complete; expansion belongs to the active model,
serving, memory and composition plans.

### HIP / ordinary Qwen

- `8ffb511`: populated 0.8B/2B HIP serving accepted through qualified scopes.
- `d16f096`: explicit total HIP device-budget admission and resident 27B
  comparison. Resident 27B C1..C64 output/state/retirement is verified under the
  declared large budget.
- `c282f20`: fixes a real HIP-only cross-feature compile regression. The shared
  packed-BF16 numeric-zero batch-greedy kernel is model-neutral and no longer
  incorrectly hidden behind `cuda-cublas`; HIP-facing test/support profiles also
  state the existing singleton policy explicitly with `native_rows: 0`.
- X2 compile/link validation after `c282f20` passed `zerogravity-gpu` with
  `hip-source`, `qwen_hip_sdk` and `fabric_sdk` with `qwen35-hip,serve`, plus a
  temporary release 27B SDK/HTTP diagnostic link.
- The fresh 27B diagnostic still does **not** establish a 27B physical result:
  its isolated process was externally terminated during model load with no Rust
  panic/error. VRAM returned to baseline, no KFD client remained, and host RAM/swap
  did not establish an obvious OOM.
- Tiny physical HIP health is now re-established independently. X2 was synced to
  exact l5 `2caf304`; a temporary self-validating HIP C++ kernel built with the
  installed CLR 7.2.4 toolchain and its referenced ROCm device-library store ran
  only through `python3 -B scripts/native-gpu.py -- ...` and exited0 on AMD Radeon
  8060S / `gfx1151`: one device, 256 exact `i*17+5` words, checksum `556160`.
  Post-run VRAM returned to `272351232/103079215104` bytes, busy0, no KFD client
  remained and the dotlink cgroup still reported `oom=0`/`oom_kill=0`.
- The earlier tiny-smoke termination is therefore localized away from HIP runtime
  health. Focused Rust test builds could still disappear without an exit code both
  inside and outside `native-gpu.py`; no cgroup/kernel/systemd-oomd OOM evidence
  was found. The temporary diagnostic source-gate experiment was discarded.
- `134a571` fixes another cross-feature HIP qualification regression exposed by
  the tracked 27B context diagnostic: generic batch-KV qualification no longer
  unconditionally invokes the CUDA-only private-BF16 KV qualifier. Exact l5 HIP
  qualification check and X2 release context-diagnostic build pass.
- Exact X2 27B host-only `cache-preflight` passes on canonical snapshot
  `98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e`;
  source remains unchanged and 512/1K/2K capacity cells complete with no GPU.
- Runtime `open_source` is now split more precisely. Exact canonical27B
  tokenizer construction is independently green:12,809,320-byte `tokenizer.json`,
  SHA256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`,
  existing release tokenizer test pass0.71s.
- A smaller host-only exact-source process also authenticates all18 27B shards and
  completes the full Qwen binding/seal/1K plan:55,563,006,776 source bytes,1,199
  tensors,851 text/348 other roles,64 layers/states,64 plan boundaries/112 domains.
  The deterministic Qwen binding/plan path itself is therefore green.
- `b0cf0b2` fixes a concrete redundant verification boundary discovered here.
  Runtime previously fully hashed all31 snapshot payloads (55,586,113,293 bytes)
  in `open_snapshot`, then `VerifiedWindowPlan::open` authenticated the exact
  shards again. Qwen now binds a manifest-authenticated `SnapshotDescriptorLease`
  that claims no payload verification; every non-shard payload keeps its existing
  compiled SHA check and every shard keeps the authoritative whole-file
  `ExtentSource::scan_verified`. Normal `SnapshotLease` remains fully verified.
- The full canonical runtime host path is now green. A warmed optimized test-profile
  build produced the full runtime unit-test binary; running that exact binary under
  an external2s heartbeat wrapper completed the canonical source/tokenizer/final-
  reader test in75.20s with about16..37MiB RSS and `gpu=false`. The heartbeat only
  kept the connector execution envelope alive; it is not part of ZeroGravity.
  Therefore the earlier silent first-shard termination was not evidence of a Qwen
  source/runtime correctness bug.
- After that closure, one reference-free exact75GiB public-HTTP liveness probe was
  launched under `native-gpu.py`, private loopback and the unchanged180s server
  deadline. It reached real HIP allocation (~31.376GB VRAM, busy6%) but the entire
  isolated process disappeared after ~10..12s despite an external5s heartbeat,
  before any HTTP response. No numerical/reference acceptance follows.
- The earlier “no visible owner” note was incomplete because each dotlink shell
  has a Bubblewrap-private PID namespace. Host-global KFD sysfs exposes process
  entry `/sys/class/kfd/kfd/proc/2406838`, owning31,098,232,832 bytes VRAM and
  six queues with `pasid=0`, zero CU occupancy/faults. **The later host-context
  correction matters:** X2 also runs Ollama/model workloads outside the sandbox,
  so this KFD owner cannot be attributed to the failed ZeroGravity run. The old
  “stranded allocation / clean272MiB baseline” interpretation is retired; the
  observed KFD/VRAM state is a shared-host baseline unless ownership is positively
  established.
- Dotlink's exposed GPU mount sources are pinned with `O_PATH|O_NOFOLLOW|O_CLOEXEC`
  and used only as Bubblewrap `--dev-bind` sources; their fdinfo has no DRM memory
  accounting. A matched non-GPU durable-Job → Bubblewrap → `nix develop` →
  `native-gpu.py` → nested-unshare control ran35s and exited0. Therefore the
  physical run's10..12s loss is not a generic connector/native-gpu lifetime limit.
- The75GiB route selects resident storage for the canonical ~53.8GB source-weight
  extent and preparation uploads projection operands sequentially, but because the
  host already has shared GPU residents the observed ~31.376GB total usage does
  **not** localize how many bytes belonged to the failed ZeroGravity process. No
  ZeroGravity lease/drop defect is established.
- X2 is usable as a shared execution host. Before each physical run, snapshot
  total/available VRAM and KFD owners, launch only if the requested workload fits
  alongside existing residents, and compare after-run state to that exact
  pre-run baseline. Clean up only ZeroGravity's own delta. The KFD process subtree
  is read-only from the sandbox; the AMDGPU reset path is operator-level and was
  not invoked. Do not reset GPU, restart dotlink, kill unseen host processes or
  force KFD reclamation merely to create exclusive access. Running kernel7.3.1 vs
  current-system7.3.3 remains only an environment observation.
- Historical R9/R10 HTTP504 ran 5 Nov; `d16f096` explicit75GiB resident-budget
  plumbing landed 8 Nov. The old ~240s streamed path exceeded the unchanged180s
  deadline; the later resident C1 path was ~5.5s. Preserve that chronology rather
  than treating the old 504 as a resident-route result.
- The accepted 66-call 27B reference and populated1K/2K reference material are not
  present in the current X2 cache. Full numerical/context acceptance remains
  blocked until exact approved material is recovered or a properly supervised
  successor capture is established; never substitute/repin.
- ROCm2B C4 warmup plus C16/C32/C64 warmup/timed mismatches remain unlocalized.
  Preserve failed cells and expected IDs; do not repin expectations.
- These remaining campaigns are explicitly authorized X2 execution/test work;
  repository/source edits still originate on l5.

### CUDA composition / serving foundation

The CUDA route is a real source-derived resident path, not a finite-case
dispatcher.

Important foundation commits:

- `4f1e827` shared source-derived CUDA composition.
- `a57590e` output-reduction repair.
- `9d73d46` compact on-device greedy/token feedback.
- `9f45471` exact decode GEMV specialization.
- `55e195c` onward: genuine ordinary inter-session cohorts.
- `b191fc6`: exact independent source reductions survive physical row widening.
- `fa703fa` / `96a7ac7`: same-frontier chunked/ragged prefill and mutual cohort
  compatibility.

Current serving uses genuine B4/B8 physical cohorts with B2 partial regrouping.
Same-frontier ragged/chunked prefill is accepted. Unequal-frontier prefill remains
fail-closed. The retained `874e67d` shared-input Ordered+LaneBlocked B8 projection
pair remains model-neutral and exact, but its ordinary-Qwen serving consumer is now
**rejected**: public Qwen3.5-2B B8 prompts47..54/16 outputs preserved all128 ID264
tokens and native width8, while selector-off/on/reverse process-order measurements
had much larger side drift than the ~1.04% favorable two-sided candidate mean.
Keep the primitive; do not select the consumer from that noisy aggregate.

## CUDA execution/performance chronology

The exact measurements/tests live in `model-execution/TODO.md`. This is the
takeover map.

### 9–10 Nov — C1 to genuine B4/B8

- Compact greedy feedback moved to a tiny device packet instead of full-vocab host
  readback.
- Exact decode GEMV specializations landed.
- B4 ordinary source cohorts shared stateless projection/head while preserving
  private state.
- B4 GDN convolution, recurrent core and full attention became genuine physical
  batching primitives.
- BF16 finite certification was folded into GEMM stores where legal.
- B4 lane-strided output head shared weights across rows without changing per-row
  arithmetic.
- Full-prompt prefill cohorts landed.
- Native width widened to B8 while stateful GDN/attention initially remained
  composed from qualified B4 groups.

### 11 Nov — dense B8, chunking and partial populations

Accepted generic K64 families:
`67586eb` ordered large-K prefill row64, `77ffc23` ordered B8 decode,
`5a75a70` RoundedSegments B8 and `2b8e90f` interleaved-pair B8.

Other work:
`74f6422` populated small contexts; `597da3f` cross-profile measurement support;
`b95acdc` B2 partial regrouping; `fa703fa` chunked/ragged state-only intermediate
prefill; `0d9a56d` / `96a7ac7` mutual cohort compatibility.

Decisions:
- B4/B8 remain native widths; B2 handles partial populations.
- Larger offered populations become multiple B8 groups, not hidden B16.
- **B16 is rejected**: 0.8B was slightly worse than two B8 groups; 2B regressed
  catastrophically. Do not resurrect without materially different architecture.
- Unequal-frontier prefill previously diverged and remains held.

### 12 Nov — exact arithmetic + reusable stateless primitives

`b191fc6` fixed the key correctness boundary: physical widening must not select a
different source reduction. Exact B2/B4/B8 row kernels, ragged cohorts, inner-major
head packing, two-segment row split, sigmoid×multiply, stateless row spans,
RMS/residual-RMS, gate, Q/K RMS and source-exact rotary primitives followed.
Several consumers were rejected because lower dispatch count did not yield
repeatable serving gain; the clean primitives remain.

### 13 Nov — stateful primitives, roots and hot-path profiling

Important decisions before the current selected B8 state path:

- B8 gated GDN primitive retained; ordinary consumer rejected.
- LaneBlocked inner-major rejected once pack cost was included.
- Multiple dominant LaneStrided<16,512> CTA/load/unpack/warp/unroll variants were
  exact but flat/unstable/slower and removed.
- A shared-memory tiled recurrent transpose was ~12–13% faster in isolated copy
  timing but slower under sustained serving; removed.
- `fe3b122` strengthened transport oracles with an independent CPU index oracle.
- `0cdfaad` retained private ragged B8 attention; current ordinary selection held.
- `d196c1b` / `bad4713` qualified private cached-GDN state binding and in-place
  alias behavior.
- `16fba0e` retained an exact-root model-neutral derived-state cache; its first
  private-GDN serving representation was ~25.6% slower despite exact-root hits,
  so that consumer is rejected.
- `2f651b7` centralized session construction and removed first-lane prefill-binding
  coupling while keeping same-frontier admission.

### Selected: `60b048d` canonical-boundary private B8 recurrent core

Current serving selects this.

- Canonical F32 recurrent roots stay the only publication/checkpoint authority.
- Eight lanes still import exact canonical state, execute one private B8 recurrent
  core in-place, then export exact canonical candidates.
- Gated output remains two accepted B4 groups.
- No native recurrent state persists across tokens.
- Qwen3.5-2B B8 47+16 kept sixteen ID264 tokens/lane.
- Controlled candidate mean ~716.377 ms vs `2f651b7` baseline ~745.102 ms:
  ~3.86% lower latency / ~4.01% higher throughput in that short workload.

### Selected: `775dfa6` ragged private B8 KV

Current serving selects this.

- Extends packed-BF16 private KV from B4 to independent ragged B8 bases.
- Canonical BF16 K/V roots remain publication authority.
- Candidate fully writes prefix copy + append + future zero tail + nonfinite status.
- A single 32-root launch was rejected because it exceeded
  `MAX_BINDINGS=32`; the cap was not widened and no opaque pointer table was added.
- B8 therefore uses two reusable plane launches, key then value.
- Physical ragged bases/guards/nonfinite/multi-token/capacity1024 qualifier passes.
- Controlled candidate mean ~709.600 ms vs bracketed `60b048d` baseline
  ~722.432 ms: ~1.78% lower latency / ~1.80% higher throughput.

### Retained: `c12acab` grouped B8 recurrent transport; consumer rejected

This is the current remote origin.

- Exact model-neutral 8-root recurrent transport, 17 explicit bindings.
- Qualified against scalar transport and CPU index oracle.
- Serving grouped import/export was only ~0.24% better against a two-sided
  baseline and slower than the first baseline side; rejected.
- Primitive remains; serving does not select it.

### 13–14 Nov retained primitives; serving consumers rejected

These signed local commits are **retained shared primitives**, not selected
ordinary serving paths:

- `85f458f` packed BF16 batch greedy:
  exact B2/B4/B8 numeric-zero ordering; serving evidence only ~0.44% one-sided and
  lacked a clean reverse bracket, so not selected.
- `f303bca` exact B8 head→greedy partials:
  exact accepted logit arithmetic and packets; serial variants were ~8.8% and
  ~2.9% slower, CTA-preserving path ~0.12% slower/noise. Full head + greedy stays.
- `68ac810` grouped ConvB4 transport:
  exact grouped/scalar/CPU-oracle parity; ordinary 20→8 launch composition was
  flat/slightly slower (~688.920 vs 689.195 ms).
- `da3eae5` dual Q/K head RMS:
  exact physical composition; serving ~0.28% slower (~692.004 vs 693.917 ms).
- `ee347d9` certified recurrent export:
  exact RecurrentOut + gated BF16 certification, including exact status 0x0a
  combined fault case; serving composition ~0.33% slower (~679.122 vs 681.355 ms).

### Selected: `399e43a` B8 private-KV CTA256

Latest selected performance change.

- Only B8 private-KV CTA width changes 32 → **256 threads**; B4 remains 32.
- One U32/two BF16 values per thread remains unchanged.
- No arithmetic, root/state, ABI, prefix/append/tail/status change.
- Full guarded physical matrix passes at widths 32/64/128/256.
- Same-process Qwen3.5-2B B8 means:
  - 32: 687.760 ms
  - 64: 678.130 ms
  - 128: 678.374 ms
  - 256: **675.905 ms**, stable 675.148–676.997 ms
- Accepted benefit vs CTA32: ~1.72% lower latency / ~1.75% higher throughput.

### 14 Nov post-KV probes: `de2eed8` / `cb9e349`

These commits record rejections only; production source was restored.

Fresh relative post-CTA256 projection ranking from temporary serialized device
events:

- 24× 12288×2048 LaneStrided<16,512>: 5252.096 µs total / 218.837 mean.
- 24× 2048×6144 RoundedSegments<2048>: 3336.128 / 139.005 µs.
- 18× 8192×2048 Ordered: 2639.872 / 146.660 µs.
- 24× 2048×2048 LaneStrided<16,256>: 1148.160 / 47.840 µs.
- 18× 32×2048 LaneBlocked<32,64>: 1124.352 / 62.464 µs.
- 6× 5120×2048 LaneStrided<16,512>: 682.080 / 113.680 µs.
- Two observed 248320-head launches totalled ~7068.768 µs.

Use these as relative family ranking only; the probe serialized kernels.

Rejected after CTA256:
- recurrent transport CTA128/256/512: exact, no stable gain; CTA256 had a severe
  stall and CTA512 order effects crossed.
- Rounded paired BF16x2 segment conversion: exact, ~0.12% device-time difference.
- LaneBlocked<32,64> B2/B4/B8 shared-weight specialization: exact, but direct
  kernel timing showed accepted ~91.9 µs after first outlier vs candidate
  ~107.5 µs; apparent whole-model ~0.74% win was noise.
- weight-only `cp.async.cg` for K64 Ordered/Rounded: exact; Ordered repeatedly
  ~1.2% slower and Rounded flipped direction between repeats.
- All experimental helpers/selectors/tests/runners were removed.

### Retained: `874e67d` shared-input B8 projection pair; consumer rejected

- Adds one model-neutral B8 physical primitive that composes an exact Ordered K64
  projection with an independent `LaneBlocked<32,T>` projection under one launch.
  Both original reduction orders, row-major activation semantics, independent
  outputs/status and source/session/root/publication boundaries remain unchanged.
- The secondary projection remains one row per CTA; this is **not** the previously
  rejected LaneBlocked64 row-sharing specialization and is also distinct from the
  old rejected LaneStrided shared-input activation-staging experiment.
- RTX3060 physical qualification matches both accepted GEMMs bit-for-bit with zero
  status. Direct alternating timing at the real 2B hot pair
  `8192x8x2048 + 32x8x2048` measured separate launches at 168.230 µs mean versus
  155.752 µs composed (~7.42% lower primitive time).
- The missing loaded-model A/B is now complete. A temporary structural consumer
  composed only the real adjacent `Qkvz(normal)` then `Ba(normal)` B8 projections
  when both shared the exact SSA input, retained resident weights, and matched the
  primitive's Ordered + LaneBlocked<32,T> contracts. Public Qwen3.5-2B ragged
  prompts47..54 with16 outputs preserved all128 ID264 tokens and native width8.
- Reject ordinary-Qwen selection: selector-off first side mean665.690ms, candidate
  mean701.917ms, reverse baseline722.543ms, reverse candidate671.881ms. The
  two-sided aggregate favors candidate only ~1.04%, while the sign flips by order
  and side drift is much larger than the net effect. This does not meet the
  repeatability bar. Temporary consumer/selector/runner/include-view artifacts were
  removed; keep the model-neutral primitive for a materially different consumer.

## Current ordinary CUDA selected path

### Selected

- Exact independent source reduction contracts (`b191fc6`).
- B4/B8 cohorts with B2 partial regrouping.
- Same-frontier chunked/ragged prefill.
- Accepted K64 Ordered/Rounded/Interleaved GEMM families.
- Exact row-split GDN consumer (`d1ec723`).
- Canonical-boundary private B8 recurrent core (`60b048d`).
- Ragged private B8 KV (`775dfa6`).
- B8 private-KV CTA256 (`399e43a`).
- Existing B4 GDN gated path.
- Existing B4 attention composition for ordinary B8 serving.
- Full BF16 output head + existing greedy path.
- Canonical state/root publication.

### Retained but not selected in ordinary serving

- Private ragged B8 attention (`0cdfaad`).
- Cached-GDN B8 gated primitive.
- Exact-root derived-state cache (`16fba0e`).
- Grouped B8 recurrent transport (`c12acab`).
- Packed BF16 batch greedy (`85f458f`).
- Exact B8 head→greedy partials (`f303bca`).
- Grouped ConvB4 transport (`68ac810`).
- Dual Q/K head RMS (`da3eae5`).
- Certified recurrent export (`ee347d9`).
- Shared-input Ordered + LaneBlocked B8 projection pair (`874e67d`); ordinary-Qwen
  consumer rejected after order-sensitive loaded-model A/B, primitive retained.
- Other stateless row/RMS/gate/QK/rotary primitives whose current consumer
  economics were rejected.

Never infer “primitive exists” == “serving selects it”.

## Reusable shared primitives that must not be lost

Retain model-neutral:
exact BF16 GEMM reductions; B2/B4/B8 physical rows; K64
Ordered/Rounded/Interleaved staging; row-major↔inner-major BF16 transport;
StatelessRowExtent/RMS/residual-RMS/SiLU/gate/QK-RMS; sigmoid×multiply;
two-segment row split; B4/B8 packed-BF16 KV; source-exact rotary; private
B4/B8 GDN recurrent + alias properties; B4/B8 gated; ragged B4/B8 attention;
exact-root derived state; scalar/grouped recurrent and Conv transports; packed
BF16 greedy/continuation; head→greedy partials; dual Q/K RMS; certified recurrent
export; exact shared-input Ordered+LaneBlocked B8 projection composition;
compiler/transport/oracle utilities.

A rejected Qwen consumer is not a reason to delete these primitives.

## Important rejected experiments / do not casually repeat

Exact data is in the owning TODO. Major rejections:

- B16/native16.
- Unequal-frontier serving admission without new per-lane source qualification.
- Persistent/derived native GDN serving representation.
- Grouped B8 recurrent transport consumer.
- Tiled recurrent transpose consumer.
- recurrent CTA width sweep.
- grouped ConvB4 consumer.
- B8 gated consumer.
- private B8 attention consumer without materially new composition.
- certified export consumer.
- dual Q/K RMS consumer.
- packed BF16 greedy selection on weak/noisy evidence.
- existing head→greedy tilings.
- LaneBlocked inner-major with pack cost.
- LaneBlocked64 shared-weight specialization.
- Ordered M128/N16.
- Rounded M32/two-warp and paired-BF16 segment boundary.
- K64 weight-only `.cg` cache hint.
- dominant LaneStrided CTA-only/**old LaneStrided shared-input staging**/
  paired-column/warp-sharing/paired-unpack/unroll variants. Do not confuse this
  rejected family with retained `874e67d`, which composes two different exact
  projections sharing one activation boundary.
- full broad requalification of unchanged primitives.
- large CUDA acceptance from cross-profile NMSE/greedy-token equality alone.

## Current performance frontier

Do not generalize quick non-LTO/direct API numbers to HTTP, long context, graph
capture, another GPU or another model.

- Original C1 anchor: 0.8B native ~65.22 ms vs eager vLLM ~90.64 ms; 2B native
  ~114.29 ms vs eager vLLM ~84.34 ms.
- Warm 2B B8 cached-decode traversal has historically been ~25–28 ms/token
  depending clocks/build state.
- Dense projection remains the largest measured region; the fresh family ranking
  above is the current local guide.
- B8 recurrent core, B8 KV transition and CTA256 are accepted material wins.
- `874e67d` reduces one directly timed real 2B Ordered+LaneBlocked projection pair
  from 168.230 to 155.752 µs mean (~7.42%), but has no completed loaded-model A/B;
  it is a retained primitive opportunity, not a serving performance claim.
- Recent 47..54 +16 whole-model waves commonly land ~675–705 ms depending
  clock/thermal state.
- Cross-process timing has repeatedly reversed sub-1% conclusions.
- For sub-1% candidates, direct device timing + same-process ABBA/reverse order is
  the minimum useful evidence.

## Numerical / state / ownership rules

- Exact source identity/geometry explicit.
- No backend grant from metadata, registry names or another target.
- Every numerical profile owns its arithmetic contract; no universal epsilon.
- Physical widening preserves original source reduction order.
- Canonical roots remain publication/checkpoint/rollback authority.
- Derived representations are subordinate exact-root caches only.
- Restore/reset/replacement cannot reuse stale derivatives.
- Private in-place mutation is only a physical primitive property until a
  lifecycle explicitly qualifies/selects it.
- One ResourceArbiter; no hidden second allocator/budget system.
- Completion/event/fence governs release/quarantine.
- Cancellation cannot publish candidate state.
- Sessions remain independent through cohorts.
- Canonical future KV tail stays exact zero where required.
- Passing cleanup/final cases do not relabel earlier failures.
- Compilation, synthetic fixtures and token equality alone do not establish full
  model/backend acceptance.
- Operator correctness, model distribution quality and determinism are separate
  claims. Follow `CORRECTNESS_POLICY.md`.

## Current blockers / open work

### l5 source/repository work

1. **Do not force another CUDA micro-optimization.** Easy launch-count, CTA-width,
   cache-hint and instruction-count variants are heavily exhausted. `874e67d`
   remains a useful exact primitive, but its real adjacent ordinary-Qwen consumer
   has now been measured and rejected because order/thermal drift dwarfed the
   ~1.04% favorable two-sided aggregate. Do not retry the same selection without a
   materially different composition.
2. **Production CUDA graph selector is privately integrated at signed `9d6613c`.**
   `5bf440d`/`db7be60`/`6a5c610`/`81b7c71`/`f1ce532` establish the graph substrate,
   resource admission, driver-safe compilation, lifetime witness and exact captured
   storage/access authority; `63ee14e` adds the workspace-owned lifecycle/cache.
   `9d6613c` adds the fail-closed selector under that same workspace authority. Exact
   compatible live cache selects Graph; absent/cold/spent, incompatible profile or
   device-token route select Native before graph enqueue; identity/lifecycle drift,
   unresolved selection or replay failure is an error with no Native retry. External
   graph APIs remain qualification-only and ordinary serving still selects
   `FusedExecution::Native`.
3. The graph policy remains64KiB control +4KiB per prepaid source-dispatch slot +
   Rust graph/receipt/lease-access-vector metadata; actual CUDA driver bytes remain
   unqualified. The charge is acquired before `graph_prepare`, may grow only before
   capture starts, survives CaptureWindow/CapturedResidentGraph/quarantine ownership
   and releases only after trusted retirement. The current charge remains exactly
   **15,356,224 bytes** (32,768 bytes owned lease/access-vector metadata). The final
   supervised original0.8B B2 lifecycle run preserves warm `[3725,3725]`, body3632,
   six replay enqueues,14,900 committed kernels, full248320/all42 exactness and
   final41MiB/reserved0; it additionally proves compatible cache lookup,
   incompatible-profile miss, second-capture refusal, explicit retire->Spent/
   foreign0, fresh-workspace recapture, and B2->B4 trusted retire-before-replace->
   Cold/foreign0. One preceding fresh lifecycle invocation disappeared before a
   result with no retained process/VRAM delta; keep it incomplete.
4. One older post-`9dca4b` repeat failed closed at the first post-capture idle
   check: total resident bytes fell by exactly15,339,840 with pending_events=0 and
   an empty queue. It did not reproduce in four subsequent pre-`63ee14e` runs or
   the final lifecycle/selector runs. Do not weaken the idle invariant or call the
   transient explained. Driver-internal graph bytes/node enumeration remain
   unqualified. The `9d6613c` same-process B2 AB/BA diagnostic measured Native
   mean273,933us/median267,847us versus selected Graph mean264,881us/median263,238us
   (~3.3% mean/~1.7% median). That is positive but modest: do **not** enable or claim
   production graph serving from B2 alone. Next obtain matched B4/B8 evidence and
   decide whether capture/selection policy earns the retained graph complexity.
5. **Large CUDA same-contract distribution/state acceptance is open but blocked.**
   Cross-profile diagnostics are not acceptance. Approved original reference/archive
   material needed for the intended rerun is missing on l5; do not invent
   tolerances, repin or substitute another model/reference.
6. **Unequal-frontier prefill remains held.** First-lane coupling was hardened, but
   admission stays same-frontier. Per-lane source position/base/state semantics
   still need approved HF source plus private-singleton physical comparison before
   scheduler admission changes.
7. Broader HTTP/context/vLLM campaigns remain separate from quick CUDA direct
   feedback. Continue the larger runtime/serving capability rather than returning
   to endless sub-percent kernel churn.

### x2 AMD large-VRAM execution/testing — 27B host-load boundary next

1. **Tiny physical HIP health stays green.** The isolated `gfx1151` smoke passed
   and returned to baseline; do not repeat it unless host/runtime state changes.
2. **Canonical27B host loading is green.** Exact tokenizer construction,
   isolated source binding/seal/1K plan and the full runtime host test all pass.
   The full-runtime test needed only an external progress heartbeat to keep the X2
   connector execution alive; do not mistake that transport behavior for a model
   requirement or source bug.
3. **X2 is a shared GPU host.** Existing KFD owner `2406838` / ~31.1GB is
   not a blocker and cannot be attributed to ZeroGravity from the private PID
   namespace because Ollama/model workloads also run on the host. The old
   “recover owner before any model retry” rule is retired. A matched35s non-GPU
   native-gpu execution control remains green.
4. Before a new27B/ROCm run, record exact total/used/available VRAM and KFD baseline,
   then run only if the requested ZeroGravity configuration fits alongside
   existing workloads. Track our process and incremental VRAM/queues, and require
   those deltas to retire after normal exit. Never evict/reset/reclaim another
   owner. The last observed ~31.38GB shared baseline left ~71.7GB before overhead,
   so re-evaluate the exact75GiB route against **current available physical
   capacity** instead of assuming exclusive-card capacity. Preserve the unchanged
   180s HTTP deadline. Missing approved27B populated/reference material remains a
   separate numerical-acceptance blocker.
5. Before every X2 physical run, sync X2 to the exact current l5 `main`, confirm
   one checkout, record active workload/KFD/VRAM state, and avoid colliding with
   insufficient remaining capacity. X2 is not an independent source-authoring
   branch.

## Validation / execution cadence

1. Revalidate status, signed HEAD, one-worktree topology, active processes and GPU.
2. Read the latest owning TODO section.
3. Implement one substantial shared unit.
4. Use rustfmt/diff/compile/narrow checks during implementation.
5. Add physical qualification only for a genuinely new numerical/layout/ownership
   contract.
6. Get quick real-model feedback.
7. Use same-process ABBA/reverse order/device events when the expected effect is
   small.
8. Reject marginal/unstable complexity and remove its selector/consumer.
9. Keep independently useful shared primitives where justified.
10. Remove all temporary profiling/runners/include views.
11. Run strict Clippy/focused checks for changed surfaces.
12. Run `scripts/check.sh` only at an assembled acceptance phase.
13. Stage only owned paths, inspect cached diff, signed commit.
14. Never push without explicit user instruction.

## l5/x2 build and GPU operational notes

Defaults:
`CARGO_BUILD_JOBS=8`, `RAYON_NUM_THREADS=4`, `RUST_TEST_THREADS=1` where useful,
`CARGO_NET_OFFLINE=true`, `PYTHONDONTWRITEBYTECODE=1` and `python3 -B` where the
host environment provides Python.

Physical GPU checks use `python3 -B scripts/native-gpu.py -- <command>` from the
established host environment. l5 is the repository/source authority; X2 is the AMD
large-VRAM execution/test host only. X2's connector shell can have a reduced PATH,
so do not treat a missing `python3`/Cargo command in the raw connector environment
as proof that the established qualification environment is absent.

Connector behavior:
- dotlink bash timeout is adjustable; there is no fixed 114-second wall.
- Long Cargo/test calls can return ambiguous null exit/success false responses.
  Never treat that as success; check processes and binary timestamp/size/executable
  bit/marker strings.
- Prefer durable `job_start` / `job_get` / `job_output` for long builds.
- Heavy debug test links have occasionally been killed without a compiler error;
  `CARGO_PROFILE_TEST_DEBUG=0` is a build-only linker-pressure workaround.
- Warmed second link attempts often succeed; verify the actual executable.

Temporary CUDA include view, recreate only when required:
- cudart:
  `/nix/store/41094qrkxhkz9311pvpj2wj0hb9ygd8r-cuda13.4-cuda_cudart-13.4.92/include`
- CCCL:
  `/nix/store/gfmsqzs0x894ni7ddq4cp5wgpijddpvb-cuda13.4-cccl-13.3.4.3.1/include`
- NVRTC:
  `/nix/store/jjpa3v76wd7275agp79nscyvb2fdbnq2-cuda13.4-cuda_nvrtc-13.4.92-lib/lib`
- view: `tmp/_tmp_cuda_include_view`
- helper: `scripts/cuda-nix-include-view.sh`
- set CUDA_PATH to the view and LD_LIBRARY_PATH to NVRTC + driver lib.
- remove the view after use.

Do not download models/SDKs, alter privileges/recovery or activate held gfx90c
paths without applicable user authorization.

## Worktree / disk discipline

- `main` is the only persistent checkout **per host**.
- No second l5 or X2 Git worktree for implementation/profiling/review.
- Author/edit source on l5. Sync exact l5 commits to X2 only as needed for AMD
  execution/testing; do not let X2 drift into a separate implementation branch.
- Reuse canonical target/cache state on each host.
- Temporary `_tmp` runners and profiling files are disposable.
- Do not recreate old worktree pools or `.agents/runs`.
- Preserve unrelated dirty work.
- No destructive Git operations without explicit authorization.

## Next agent must not redo

Do not replay or reopen without materially new architecture/evidence:

- the completed Fabric vNext transition or cancelled recurrence/watch campaigns;
- B16 active/native serving;
- unequal-frontier **prefill** admission without new approved source qualification;
- persistent derived-GDN consumer, grouped recurrent transport consumer, tiled recurrent
  transpose, recurrent CTA sweeps, grouped ConvB4 consumer, B8 gated consumer,
  dual Q/K RMS consumer, certified recurrent-export consumer;
- existing head→greedy tilings, packed greedy selection on the old weak evidence,
  LaneBlocked inner-major/terms64, Ordered M128/N16, Rounded M32/paired BF16,
  K64 `.cg`, dominant LaneStrided CTA/old shared-input staging/paired-column/
  warp-sharing/paired-unpack/unroll variants;
- the `874e67d` ordinary-Qwen consumer: the exact Ordered+LaneBlocked pair primitive
  remains useful, but its loaded-model consumer was measured and rejected because
  order/thermal drift dwarfed the small two-sided aggregate;
- the CUDA graph PTX compatibility blocker: `6a5c610` already fixes non-persistent
  NVRTC by emitting exact-SM cubin/SASS;
- graph opaque-resource admission/lifetime: `db7be60` + `81b7c71` already price,
  retain, witness and retire the charge;
- same-card graph storage/event authority: `f1ce532` already requires every replay
  to prove the exact captured `(ResidentLease, DeviceAccess)` set against the armed
  `ResidentSubmission`;
- the old X2 "dirty GPU / wait for KFD owner to disappear" interpretation. X2 is
  shared with Ollama/model workloads; existing KFD/VRAM is baseline unless
  positively attributed to our process.

Do not broadly requalify unchanged primitives merely because a new phase starts.
On X2, do not reset/reclaim/restart/kill other GPU owners. Tiny physical HIP health
is already green; repeat it only if host/runtime state changes.

## Copy/paste next-agent prompt

> Use `dotlink---l5` for all repository/source changes and `dotlink---x2` only for
> AMD large-VRAM execution/testing. On both hosts work in
> `/home/pvl/spaces/abird/src/zerogravity` on `main`; do not create another Git
> worktree. First read, in order: `AGENTS.md`, `.agents/plans/README.md`,
> `.agents/docs/CURRENT_HANDOFF.md`,
> `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-14.md`,
> `.agents/plans/TODO.md`, `.agents/plans/model-execution/TODO.md`,
> `.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md`, and then the owning
> plan(s) for the task. Current source/tests/live TODOs outrank handoff text if
> `main` has advanced.
>
> Revalidate before editing: `git status --short --branch`, `git rev-parse HEAD`,
> `git rev-parse origin/main`, `git log -1 --show-signature`,
> `git worktree list --porcelain`, active Cargo/model/native-gpu processes, and
> physical GPU state. Latest implementation is signed
> `9d6613cb0bef84147a7d30453e5afb82ca83411b`
> (`feat(cuda): select compatible resident graph cache`), while pushed
> `origin/main` is `4502f5678e8cb1641848be62cd0fb5068bae1d09`.
> The handoff creation itself may add one newer docs-only signed commit, so trust
> live Git and sync X2 to that exact l5 tip by bundle before any new X2 physical
> execution. Never push unless the user explicitly asks.
>
> **Immediate continuation:** signed `63ee14e` owns the private workspace graph
> lifecycle/cache and signed `9d6613c` adds its fail-closed exact-compatible private
> selector. Exact live compatibility selects Graph; absent/cold/spent/incompatible
> profile/device-token routes select Native before graph enqueue; runtime identity/
> lifecycle drift, unresolved selection and replay failure error without Native retry.
> Public serving still constructs `FusedExecution::Native`. The supervised original
> 0.8B B2 selector qualifier preserves full248320/all42, the exact **15,356,224-byte**
> graph charge, trusted retirement/replacement and final reserved0/41MiB; it proves
> compatible selector replay and incompatible Native fallback without graph replay.
> One preceding fresh selector invocation disappeared without Rust error/exit code and
> remains incomplete. Driver-internal graph bytes/node enumeration remain unqualified.
>
> **Next phase:** keep public serving Native and extend the same matched selector
> evidence to B4/B8 before deciding any capture/selection policy. B2 AB/BA timing was
> positive but modest: Native mean273,933us/median267,847us versus selected Graph
> mean264,881us/median263,238us (~3.3% mean/~1.7% median), so it is not a serving-win
> gate by itself. Preserve all existing numerical/state/source/resource/lifetime
> authority and do not enable graph serving unless the broader matched evidence makes
> the complexity worthwhile.
>
> Preserve the graph invariants already established: one cold capture per live
> batch workspace; exact service/source/generation/stream/profile identity; batch
> workspace pinning across graph life; exact retained storage/access authority;
> graph resource charge acquired before `graph_prepare`, allowed to grow only
> before capture starts, retained through capture/replay/quarantine and released
> only after trusted retirement; terminal graph destruction only after trusted
> barriers; unknown completion fences/quarantines rather than freeing. One older
> repeat under the previous **15,339,840-byte** policy failed closed at the first
> post-capture idle check with an exact-policy-sized resident-byte delta. It did not
> reproduce in four subsequent runs, and the stronger `81b7c71` witness remains in
> place. Do not weaken the idle invariant or claim that transient is explained.
>
> Preserve the accepted ordinary CUDA path: exact independent source reduction
> contracts; genuine B4/B8 cohorts with B2 partial regrouping; same-frontier
> chunked/ragged prefill; selected canonical-boundary private B8 recurrent core
> `60b048d`; selected ragged private B8 KV `775dfa6`; selected B8 private-KV
> CTA256 `399e43a`; canonical roots remain checkpoint/rollback/publication
> authority. Keep reusable model-neutral primitives even where the first Qwen
> consumer was rejected: grouped recurrent transport, packed BF16 greedy,
> head→greedy partials, grouped ConvB4 transport, dual Q/K RMS, certified recurrent
> export, private B8 attention/gated GDN, stateless row/RMS/gate/rotary primitives,
> exact-root derived cache, and `874e67d`'s Ordered+LaneBlocked pair. Do not
> resurrect rejected B16/unequal-prefill/recurrent/transpose/CTA/head/Conv/QK/
> LaneBlocked/K64/Rounded/dominant-LaneStrided variants without materially
> different architecture and evidence.
>
> Large same-contract CUDA distribution/state acceptance remains blocked by missing
> approved reference/archive material on l5; do not invent tolerances, repin or
> substitute another reference. On X2, exact tokenizer/source binding and the full
> canonical27B host runtime path are green. X2 is a **shared GPU host**: existing
> KFD owners/VRAM (including Ollama/model workloads) are baseline, not a blocker.
> Before our run record current VRAM/KFD state, run only within available capacity,
> and verify only our incremental process/VRAM/queue delta retires. Never
> reset/reclaim/restart/kill another owner. The prior exact75GiB27B HTTP attempt
> disappeared before HTTP status; historical R9/R10 HTTP504 predates resident-budget
> support, and the accepted66-call plus populated1K/2K reference material is still
> missing. Preserve the existing180s contract; do not extend/repin to manufacture a
> pass.
>
> Work implementation-first: shared clean primitives where semantics genuinely
> generalize, model/source policy at model/source layers, Rust ownership/zero-cost
> abstractions where natural, correctness before performance. Reuse accepted
> validation for unchanged primitives. Use `CARGO_BUILD_JOBS=8`; physical GPU work
> only through `python3 -B scripts/native-gpu.py -- ...`; prefer durable jobs for
> long builds/runs. Remove every `_tmp` runner/include view/temporary selector or
> profiling hook after use. Inspect `git diff --cached --stat` before every signed
> commit. Keep each user turn to about30 minutes, checkpoint at the next coherent
> point, pause, and do not push unless explicitly asked.

---

## Historical design foundation — retained context, not current execution instructions

The sections below preserve the original product/architecture evolution and enduring design
context from earlier handoffs. **Do not treat old “next step”, worktree, campaign or status
phrasing inside historical material as current instructions.** The current continuation is
entirely above and in the live TODOs.

## 3. Original mission and enduring product design

ZeroGravity began as a Rust-native inference compiler/runtime and evolved into a broader execution fabric.

The enduring product goal is one Rust-owned execution fabric that can support:

- text generation;
- embeddings/reranking and encoder tasks;
- image generation/editing;
- video/temporal generation;
- ASR;
- TTS and audio codecs;
- realtime/full-duplex media;
- combined multimodal programs;
- future optional training/update tasks where explicitly designed.

The user-facing aspiration is simple:

- one binary;
- one Rust SDK;
- local-first operation;
- automatic hardware selection when safe;
- explicit fit/capacity explanations;
- OpenAI-compatible serving where implemented;
- no Python inference server dependency for qualified local paths;
- the same runtime for portable and highly specialized execution rather than a separate server per model or modality.

The performance goal is measured Pareto leadership where specialization is justified, never a blanket “always fastest” claim.

The architectural idea from the beginning has been:

- model mathematics and source semantics are defined once;
- state/effects are explicit;
- portable/reference execution exists first;
- optimized hardware paths replace closed semantic regions only when legal and qualified;
- one logical session/root authority publishes committed state;
- one shared physical resource authority owns real memory/device budgets;
- device workers own actual asynchronous compute/events but cannot invent logical commits;
- serving, model loading, scheduling and resource ownership are not cloned for each specialization.

The initial vision and full early architecture remain in:

- VISION_SCOPE_AND_PRINCIPLES.md
- ../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md
- the current public Engineering Specification and Whitepaper under ../../docs/
- current Fabric vNext plans for the next structural transition.

## 4. Engineering philosophy and non-negotiable invariants

Priority order:

1. semantic/state correctness;
2. source and quality fidelity;
3. safety, ownership and crash behavior;
4. compatibility and user clarity;
5. stable latency/throughput;
6. peak specialized speed;
7. experimental complexity only when evidence earns it.

Rust design preference:

- clean, robust, intuitive ownership;
- shared lowest layer when it genuinely benefits multiple callers;
- zero-cost abstractions where natural;
- hot-path optimization where simple and maintainable;
- prefer a clean design over convoluted optimization when the two conflict;
- no second runtime/allocator/state publisher hidden behind “optimization.”

Hard invariants:

- exact source/model identity remains explicit;
- model metadata or a registry slug never grants execution support;
- a finite captured testcase is qualification evidence, not general runtime authority;
- prompt IDs, prompt lengths, ordinals, packet contents, capture hashes or nearest-case tables must not select ordinary production math;
- one logical StateRoot publication authority;
- one process-shared physical ResourceArbiter;
- actual completion/event/fence governs release;
- completion-unknown work is quarantined, not silently recycled;
- source readers/owners survive until actual final consumers retire;
- cancellation cannot publish partial state;
- unsupported shape/source/backend is explicit Unsupported/refusal, never silent fallback to another implementation;
- a passing cleanup or final case cannot relabel an earlier failure;
- compile/source-only success does not imply physical inference support;
- performance follows correctness;
- no universal numeric epsilon; each operator/model/profile owns its declared contract.

The shared correctness policy is:

    ../plans/numerics-and-context/CORRECTNESS_POLICY.md

It deliberately combines the useful ideas from vLLM and llama.cpp without blindly copying either:

- exact semantics/source/state checks;
- independent operator numeric checks;
- model-level output/distribution/quality checks;
- determinism/batch-invariance as separate claims.

## 5. Important rejected patterns

Do not reintroduce:

- one model-specific server per family;
- duplicate resource wallets;
- duplicate StateRoot publishers;
- generic production dispatch based on testcase/capture identity;
- arbitrary Hugging Face remote Python execution;
- metadata-only “support” claims;
- universal tolerance constants;
- hidden precision changes to hit benchmark numbers;
- KV-only prefix reuse for models with additional recurrent/SSM/GDN state;
- benchmark claims before exact source/output/ownership gates;
- a giant permanent graph interpreter in the token hot path;
- per-kernel trait/object hierarchies with no demonstrated need;
- durable agent side branches/worktrees as coordination state;
- monolithic generated handoff archives or documentation SHA catalogues.

## 6. Architecture evolution from the beginning

### 10–12 Oct: initial architecture and v4 consolidation

The early design converged on:

- semantic/effect-aware model programs;
- portable CPU/reference execution;
- optimized device regions;
- extreme model/hardware specialization as optional qualified providers;
- TaskProgram/TaskDriver for non-autoregressive tasks;
- explicit model/session/resource/device boundaries;
- multi-model routing and OpenAI-compatible serving;
- local-first model store/cache/acquisition;
- future SSD and distributed placement.

The v4 minimality principle reduced durable representations to the semantic program and prepared executable/bundle; lower-level compiler/device views are transient.

### 13–16 Oct: source formats, quantization, model families and GPU ownership

Work established:

- GGUF/safetensors/sharded source handling;
- Qwen3.5 and Gemma source/model contracts;
- CPU source-exact math and quant paths;
- GPU backend separation for CUDA/HIP/Vulkan;
- physical ResourceArbiter and source/device ownership;
- exact source/state/fixture policies;
- XDG cache/source behavior;
- early model acquisition/cache foundations;
- multimodal/task architecture research.

Qwen .8B and 2B became the immediate correctness vehicles.

### 17–22 Oct: full-model numerics, GPU native paths and serving foundations

Work expanded through:

- full 24-layer Qwen execution/reference comparisons;
- all 42 Qwen state domains;
- full 248,320 logits;
- RMSNorm/GDN/attention/FFN numerical investigation;
- retained source/window ownership;
- GPU native image/launch qualification;
- cancellation/fault/quarantine;
- serving queues, streaming and shutdown;
- OpenAI-style API subset;
- preliminary matched vLLM comparisons.

Historical numeric differences remain evidence, not current acceptance.

### 23–26 Oct: integration, routing, acquisition and broader backend work

Work covered:

- device routing, including exact /dev selectors where appropriate;
- CLI/config/source precedence;
- acquisition providers and offline policies;
- cache transactions/reclamation;
- pipeline/config validation;
- Qwen27 and Gemma finite qualification lanes;
- HIP/Vulkan qualification work;
- serving concurrency/queue scaling;
- shared state/resource ownership audits;
- repeated fresh bug/regression reviews.

### 27 Oct: Fabric vNext design

The user requested a clean future architecture where the whole target structure becomes visible early in code, then current implementation migrates into it.

The first structural milestone is intended to introduce model-neutral engine and reusable serving crates, visible typed contracts and explicit unsupported stubs before deep migration.

That was the design-only checkpoint. Signed`7396984` now implements S0; functional migration remains open.

### 28 Oct: ordinary source-derived Qwen CUDA integration

Current source now has ordinary Qwen3.5 .8B/2B CUDA execution assembled under the existing resident owner:

- one call-free ForwardPlan;
- authenticated source owner;
- request-local materialization;
- cuBLAS projection path;
- source-exact attention;
- arbitrary-row parameterized GDN;
- F32/F32/BF16 two-root state;
- one queue/flight/completion/quarantine path;
- existing StateRoot publication authority.

It remains qualification-only. No production ordinary CUDA route is enabled.

### 28–29 Oct: evidence hardening, cache human aliases and repository normalization

Recent work also:

- made cache by-name aliases crash-consistent and explicitly non-authoritative;
- simplified ordinary parity evidence so the checked-in capture procedure defines the reference and only the evidence content digest crosses processes;
- removed code/doc SHA catalogues from parity evidence;
- made compare-existing independent of the historical native bundle;
- retired machine-local capture-only tests and dead run-artifact references;
- reduced topology to main-only;
- pruned superseded handoff archives;
- normalized current planning provenance;
- reviewed and fixed environment-sensitive PID1 serving signal tests;
- added Qwen27 layer9 source-contract coverage to the shared check gate;
- refreshed vNext handoff method to use this canonical pointer + maintained files + Git history.

## 7. Current repository/runtime structure

Current workspace owners remain:

- zerogravity-core
- zerogravity-artifacts
- zerogravity-cpu
- zerogravity-gpu
- zerogravity-model-qwen35
- zerogravity-model-gemma4
- zerogravity
- zerogravity-cli
- zerogravity-gpu-qualify
- zerogravity-engine (cold model-neutral shell; F0 migration pending)
- zerogravity-serving (cold embedded API; F2 protocol migration pending)

Signed`7396984` adds the two Fabric crates without enabling new model routes.

Current logical ownership:

- core: semantic/state/resource/device/pipeline contracts;
- artifacts: source formats, immutable sources, cache, acquisition;
- model crates: family math/source/state/tokenizer semantics;
- CPU/GPU: physical execution providers;
- zerogravity facade: current public composition surface;
- CLI: still owns more reusable application/serving policy than final vNext design wants;
- serving currently lives under CLI and is intended to move to a reusable library during vNext.

## 8. Current Qwen ordinary CUDA status

### What is implemented

Ordinary .8B/2B source-derived CUDA is integrated under the resident execution owner.

Source-free tests establish:

- ordinary cancellation checks before allocation, before flight, during source execution, after launch/completion and before commit;
- armed-flight cancellation enters existing unknown-completion/quarantine behavior;
- explicit provider/pool shutdown;
- unique model ownership before shutdown;
- zero resident bytes/events before namespace close;
- source/provider ownership survives until retirement;
- no alternate root publication path.

The ordinary production route remains intentionally refused until real-current-source qualification closes.

### Exact approved .8B source

Repository/model:

    Qwen/Qwen3.5-0.8B

Revision:

    2fc06364715b967f1860aea9cf38778875588b17

Visible approved source path:

    /home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17

Weight shard:

    model.safetensors-00001-of-00001.safetensors
    bytes: 1,746,942,600
    SHA256: 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696

The tracked source manifest is:

    crates/zerogravity/src/qwen35_cuda/source08.json

### Recovered reference and original blocker — 29 Oct

The old assembled bundle was purged, but its exact `.8B` N1 inputs are recovered
offline under XDG state. Retained canonical images plus two whole SM80
FlashAttention modules extracted from the pinned local reference image and the
original cuBLAS providers match current Rust identities. All 11 model members
verify; maintained bundle validation and isolated structural preflight pass.

Fresh phase-1 cached47→50 reference capture passes after selecting the existing
8 GiB reference-fixture host profile, with all 1,008 source windows retired.
The original 4 GiB refusal stays failed. The independently emitted evidence
content digest is retained in the owning numerics plan and external evidence.

Fresh ordinary comparison fails before model math: pinned `cublasCreate_v2`
returns status 1 after the owned allocator denies a private `cuGetExportTable`
call. Zero allocation attempts/charges and bounded resolver observations identify
this as a provider ownership blocker, not missing model/bundle bytes. Do not
forward opaque tables or preload cuBLAS around admission to make it run.

The current evidence and remaining work live in
`../plans/numerics-and-context/TODO.md`. `.8B` cached47→50 is exact; broader
lifecycle/context and production routing remain unqualified. Owner order is CUDA correctness first,
then ROCm on `pvl-x2`, with no subagents.

Maintained tooling:

- tools/qwen35-cuda-reference/
- tools/qwen35-cuda-bundle/
- scripts/qualify-qwen35-ordinary-cached47-50.sh

The Python discovery/packed-capture protocol still requires explicit retained
external capture sources. Exact offline bundle recovery is documented in the
bundle helper README; the Rust two-process parity wrapper uses that recovered
bundle. These are qualification inputs, not production authority.

### Owned execution checkpoint — `2983bb7`

Owner approved replacing ordinary opaque BLAS with shared driver-only GEMM.
The allocation-free tensor-core/SIMT primitive passes 180 synthetic cases and
eight authenticated prefill/cached projections exactly, including guards and
retirement. Typed SSA roles choose explicit reductions without captured node or
prompt lookup. Original compiler-context reservations and physical flights remain
the owners; pinned reference/source/resource limits are unchanged. Ordinary
comparison no longer requires cuBLAS paths.

Corrected the rotary admission omission (+280 bytes), GDN partition binding and
premature BF16 rounding of residual RMS sums. Row RMS retains Torch vector-four
FP32 boundaries; Q/K RMS instead follows the fused Triton adjacent-pair/warp
reduction. CUDA rotary now uses FP32 FMA and final BF16 rounding; the CPU-target
BF16 product contract is unchanged. All211 GPU host tests (43 ignored) and an
independent host-FMA rotary comparison at14 positions through32767 pass.

Owned tile-64 CUDA attention now preserves FlashAttention's BF16 unnormalized
probabilities, FTZ FP32 online softmax and tensor-core QK/PV. Runtime geometry,
not captured-case routing; no new device scratch or allocation authority. Both
independently authenticated prefill/cached attention captures match exactly.
Nine analytic causal/multitile cases through a32K active prefix and three
nonfinite refusals pass with guards and retirement. They are not long-context
model parity.

Fresh `.8B` cached47→48→49→50 passes exactly: all248,320 logits, all42 state
domains and natural feedback at every step, with normal model shutdown. The
reference bytes/digest are unchanged. The earlier229,459-logit /4,359,077-state
failure (max0.28125) remains failed historical evidence.

Replaced obsolete row/residual CPU-profile CUDA oracles with independent staged
CUDA/Torch comparisons:16 cases across widths1024/2048, rows1/2/17/64 and residual
modes pass exactly, including non-BF16 residual sums, guards and retirement.
Real post-launch cancellation and abandoned-future probes pass:396 original
handles quarantined, prior bits/tails intact, unchanged root, peer refusal and
retained charges after external drops. Normal lifecycle passes all13 admitted
capacities (`.8B` through32K,2B through16K), each at48 active tokens;2B32K refuses
at pure preflight. Long-active-context/2B numerical parity remains open. Final
isolated repository gate passes2,129 Rust executions/0fail/83ignored;14 Python
suites and ancillary gates pass. CUDA/native host675/0fail/54ignored and strict
CUDA/HIP/Vulkan feature checks pass.
The2B approved snapshot exists in the canonical cache; its extra reference image
must be recovered independently, not replaced by the `.8B` inventory.

Owned execution landed in signed main commit `2983bb7`; nothing is pushed and
production remains disabled. The extra exact2B reference image was not found
among1,252 extracted SM80 modules; retained original capture inputs/long-context
comparisons are also missing from the current qualification root. Do not replace
or repin them. ROCm is now active on `pvl-x2` gfx1151/96GiB, independently of
those CUDA input gaps. Current source is in bounded run-local execution scratch,
not another Git worktree. The exact31-member Qwen3.8-27B snapshot was imported and
verified in canonical XDG cache from existing inputs; no download was needed.
Fresh independent natural full64/fullV/all112-state reference capture validates
both seed11→220→16 cohorts and source/process/FD closure. Capture/validator
protocol repairs have14 Python controls;26 Rust comparator/admission controls and
narrow/shared HIP strict checks pass. Signed `99f0232` fixes rank3 Conv4 preflight
through the shared exact binding/30 source-window tests. SDK-matched child-only
loader resolves build-libc versus GLIBC2.44 runtime refusal without replacing
SDKs or changing shell libc. Actual teacher full4 and healthy natural9-call reports
match full64/fullV/all112-state exactly; all six repaired natural modes pass
semantic validation. Known reader paths retire121 modules; release faults retain
121 modules/original service/source/1,477 handles. Teacher unknown and independent
Pending/Snapshot final-owner scopes also pass. Earlier failed scopes stay failed.
The old healthy controller exit record was lost; that controller scope stays
incomplete. A fresh durable healthy-only replay passes with exit0 and no remaining
processes. Signed `825ade8` lands the bounded diagnostic repairs. Source gate
passes2,132 Rust/17 Python suites with explicit skips. Genuine ordinary contexts, small-model HIP, C1..C64
and performance remain open; production stays disabled. See the owning numerics
TODO for exact scopes and continuation.

Do not recreate deleted `.agents/runs` paths or treat old run names as provenance.

## 9. Ordinary cached47→50 parity protocol

Current parity design was deliberately simplified.

The checked-in reference capture procedure defines how phase 1 is produced.

Evidence schema v3 contains:

- source coordinates;
- prompt;
- 47→48→49→50 observations;
- full logits;
- all state domains;
- selected IDs/max-bit receipts;
- teardown/close receipt;
- an embedded corruption digest.

Cross-process trust:

- phase 1 emits the evidence content SHA256 in stdout;
- normal wrapper execution carries that digest in memory to phase 2;
- compare-existing requires the independently supplied trusted digest;
- the embedded digest is corruption detection, not a self-authenticating implementation identity.

Removed intentionally:

- reference implementation SHA graph;
- source-code SHA catalogues;
- migration pin catalogues;
- stage-receipt authority;
- code-hash-based evidence identity.

Phase 2 does not need the historical native bundle; it needs:

- captured evidence;
- trusted phase-1 evidence digest;
- current approved model source;
- ordinary CUDA providers.

Normal phase 1 still needs the qualified native/reference bundle.

The wrapper is:

    scripts/qualify-qwen35-ordinary-cached47-50.sh

Modes:

- normal: structural checks, ephemeral legacy staging if necessary, reference capture, digest handoff, ordinary compare;
- preflight-only: structural/environment checks only, no large copy;
- compare-existing: no historical bundle required.

## 10. Exact ordinary CUDA context admission

Under the unchanged 4608 MiB provider / 5120 MiB physical limits, totals include
cold model, two roots and retained conservative admission headroom. The owned
GEMM allocates no scratch; these are not claims of physical scratch usage.
The 30-Oct correction includes the previously omitted 280-byte rotary table.

### .8B

- 512: 1,652,916,585 B
- 1K: 1,667,662,185 B
- 2K: 1,697,153,385 B
- 4K: 1,756,135,785 B
- 8K: 1,874,100,585 B
- 16K: 2,110,030,185 B
- 32K: 2,581,889,385 B

All fit admission.

### 2B

- 512: 3,911,846,249 B
- 1K: 3,926,591,849 B
- 2K: 3,956,083,049 B
- 4K: 4,015,065,449 B
- 8K: 4,133,030,249 B
- 16K: 4,368,959,849 B
- 32K: 4,840,819,049 B

512 through 16K fit.

32K exceeds 4,831,838,208 B by 8,980,841 B and must refuse before physical allocation.

These are admission/fit results only. Every admitted context still needs real-model correctness, continuation, fault and ownership qualification.

## 11. Deferred Qwen mission

Fabric transition is now first. Do not rerun already accepted cached47→50,
bounded64 HIP, finite27B, lifetime or GDN scopes merely to restart the campaign.
After migration, resume genuine multi-row0.8B/2B HIP and call-free lowering,
then general27B/populated fit contexts/C1..C64/matched vLLM. Preserve the separate
CUDA2B/wider reference and ROCm baseline authority gaps. Exact current scope:
`../plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md` and owning
model-execution/numerics/serving TODOs. Production remains disabled.

## 12. Serving and router state

The project already has:

- bounded HTTP handling;
- health/models;
- OpenAI-style text/chat completions subset;
- JSON/SSE streaming;
- request/output usage accounting;
- bounded pending queue;
- cancellation/disconnect propagation;
- CPU route;
- finite/native CUDA routes where qualified;
- source/model route validation before side effects;
- graceful worker join/shutdown;
- source-free C16/C32/C64 queue/population coverage.

The source-free production router queue is not the blocker for ordinary Qwen CUDA; the model route is.

Recent fresh review found the process-signal fixture incorrectly assumed every environment could nest a bubblewrap PID1 namespace. The shared fixture now:

- always exercises ordinary SIGTERM/SIGINT process shutdown;
- preflights PID1 namespace capability;
- skips PID1 rows only for explicit namespace/permission unavailability;
- hard-fails unexpected bubblewrap/preflight errors;
- reports skip reason explicitly;
- Rust asserts ordinary rows cannot skip and PID1 rows pass when capability exists.

Current serve all-targets result after the fix:

- 89 passed;
- 0 failed;
- 4 intentionally ignored source/hardware fixtures.

## 13. Cache/acquisition current state

Canonical cache data is authoritative; human by-name naming is not.

Human projection design:

    models/
      snapshots/<digest>/...
      refs/...
      .by-name-initialized
      .by-name-state/
      by-name/

Key properties:

- runtime/model selection never reads by-name state;
- by-name projection is disposable;
- explicit aliases and automatic names live in durable human state;
- dirty modes recover projection/reverse/reconciliation;
- publication can leave a pending human update and later self-heal;
- missing state after prior initialization refuses rather than silently losing aliases;
- reset requires explicit alias discard;
- mount-boundary checks use mount identity, not just st_dev;
- process-exit/fault matrices exercise human metadata and reclamation recovery;
- canonical snapshot/ref availability is not downgraded by human metadata failure.

Acquisition/cache supports bounded, source-aware providers and offline exact-cache behavior for the implemented providers, including Hugging Face/GitHub/Xet/OCI/S3/GCS/Azure/ModelScope/object flows where the relevant feature is enabled.

No provider metadata or alias may upgrade source rights.

## 14. Device/CLI/config state

Current CLI/device work includes:

- explicit CPU/device routing;
- exact PCI/device identity;
- validated direct /dev selectors for supported DRM/NVIDIA nodes;
- NVIDIA minor is not treated as CUDA ordinal;
- aliases converge on one observed device identity;
- source/alias/device policy precedes execution side effects;
- bounded YAML pipeline/config checking;
- strict config/env/argv precedence;
- dry-run and refusal behavior.

Pipeline configuration is more complete than pipeline execution. A valid YAML graph does not imply the task pipeline is implemented.

## 15. Memory/SSD/state direction

Implemented foundations include:

- shared physical resource accounting;
- immutable source extents;
- bounded source windows;
- source descriptor retention;
- transfer staging admission;
- retained GPU readers through actual completion;
- state/root/cancellation/quarantine semantics.

Open work remains for:

- production paged complete-domain state;
- complete-domain prefix reuse/COW;
- product snapshot persistence/restore;
- real SSD model streaming;
- expert/source IO QoS;
- eviction under actual GPU completion;
- broader multi-model residency.

Do not call KV-only reuse a hybrid-state prefix hit when recurrent/GDN/SSM/other domains are missing.

## 16. Model-family status discipline

Qwen .8B/2B are the primary current ordinary CUDA correctness path.

Qwen27, Gemma4 and other families have substantial source/reference/finite backend work, but do not inherit .8B support automatically.

Keep these distinctions:

- source parser support;
- family mathematical support;
- CPU reference support;
- a finite backend qualification;
- a production route;
- long-context support;
- serving support;
- performance qualification.

They are separate claims.

Future/target families and tasks discussed in the design include broader Qwen, Gemma4, Llama/Mistral, GLM, DeepSeek, image, ASR, TTS, Whisper/Canary/Parakeet-style audio systems and multimodal pipelines. Treat these as roadmap targets unless current source/tests explicitly implement and qualify them.

## 17. Fabric vNext — historical design that led to the completed transition

Historical state when this section was written: S0 was signed in `7396984` and
S1–S3/F0–F4/Q0–Q2 were still open. The transition is now complete; use the
current resume section and closed `fabric-vnext-transition` plan for present status.

User intent:

- make the final architecture visible in code immediately;
- break pre-v1 internal/public APIs when that produces a cleaner design;
- move current working code into the new ownership model instead of carrying compatibility layers forever;
- allow private compile-safe placeholders/Unsupported where implementation is not yet migrated;
- no reachable todo panic on an advertised supported path;
- preserve all current exact source/numeric/state/resource safety;
- qualify the assembled transition once, then fix regressions in small focused commits.

Target dependency/ownership idea:

- zerogravity-core: semantic/numeric/state/provider/physical contracts;
- zerogravity-artifacts: containers/codecs/source/acquisition/store;
- model crates: family source/math/state/tokenization/specialization;
- CPU/GPU: physical providers;
- new zerogravity-engine: model/backend-neutral runtime, scheduling, sessions, residency;
- new zerogravity-serving: reusable server/protocol layer depending on engine/core, not facade/CLI;
- zerogravity facade: composition root/public SDK;
- CLI: arguments/presentation/signals only.

The facade is the composition root. Engine must not depend on Qwen/Gemma or physical CUDA/CPU implementations. Serving must not depend on facade/CLI. CLI should call the public SDK.

### Planned transition sequence

S0 — full structural skeleton first

- create engine and serving crates;
- expose top-level target ownership/contracts;
- compile;
- explicit Unsupported for pending operations;
- architecture checker updated;
- no heavyweight model qualification yet.

S1 — exact source/quant/encoding migration.

S2 — family math + GPU provider/qualification separation.

S3 — canonical state/root/resource publication and ownership.

F0 — model-neutral engine.

F1 — complete public Rust SDK.

F2 — reusable serving library.

F3 — thin CLI and deletion of shadow runtime copies.

F4 — feature parity matrix across all currently real behavior/refusals.

Q0 — one assembled broad qualification.

Q1 — focused bugfix queue at lowest shared layers.

Q2 — final full gate, physical/source qualification and capability matrix.

Only after transition:

- broader models;
- vLLM-style continuous batching/paged state;
- larger model residency;
- SSD execution;
- Gemma broad native routes;
- additional quant formats;
- sparse/MoE/MLA/Engram/MTP;
- multimodal ASR/TTS/image/video;
- NPU/remote distributed placement;
- calibrated autotuning/benchmarking.

The vNext documents no longer require or promise a monolithic archive or documentation SHA index. This handoff is the cross-window pointer. Git history is the archive.
