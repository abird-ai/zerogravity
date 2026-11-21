# ZeroGravity current comprehensive handoff — 21 Nov 2026

Canonical cross-window entry point for `/home/pvl/spaces/abird/src/zerogravity`.

Full dated takeover snapshot: [`FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md`](FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md). The 21 Nov pack embeds this live handoff, current owning-plan snapshots, the exact post-20-Nov signed chronology, and the complete 20 Nov pack; the 20 Nov pack in turn carries the 18 Nov/15 Nov/14 Nov/October design foundation.

This file is the complete takeover map and a pointer into maintained source, tests
and plans. It is deliberately **not** the retired monolithic SHA/archive/evidence
handoff model. Git history retains superseded checkpoints; current source, tests
and live TODOs are authoritative.

### Terminology architecture — 21 Nov 2026

Active naming follows [TERMINOLOGY.md](TERMINOLOGY.md). The first cleanup
separates model identity from runtime policy
(`QwenModelVariant::{Qwen800M,Qwen2B}` vs `QwenServingProfile`), shortens
arithmetic names to `NumericalProfile` / `NumericalContract`, replaces
`native_rows` and bare B-width scheduler language with explicit
`cohort_width` / fused cohorts, and stores historical GDN captures under
`reference/<variant>/<phase>/<tokens>`. The second cleanup is now complete:
the normal Qwen source-backed path is `QwenForwardPlan` /
`QwenForwardInvocation` with one backend-neutral `ForwardSourceSeal`; CUDA
mechanics live under `cuda_forward` / `CudaForward*`, and active qualification
uses semantic `prefill 47 + decode 48–50` names plus `FORWARD_*` env/receipt
labels. No compatibility aliases retain the superseded execution vocabulary.
Historical chronology/schemas/provenance remain intact. The third cleanup is now
complete too: the old `SourceCall*` family is replaced by the core semantic pipeline
`InvocationInput -> BoundInvocation -> TextInvocation { TokenSpan, InvocationPhase }`;
Qwen layers model semantics as `PromptInvocation`, `QwenGdnInvocation` and
`QwenModelInvocation`; live execution ownership is `ResidentText*`; the backend body is
`text_execution`; and observation is `TextFrame` / `TextObserver`. Active developer
interfaces use `TEXT_INVOCATION_*`, `RESIDENT_TEXT_*` and
`ZG_QWEN_TEXT_INVOCATIONS_MANIFEST_SHA256`. The legacy reference schema
`zg.original-hf-source-call-reference.v1` and captured status
`CAPTURED_ORIGINAL_SOURCE_CALL_REFERENCE_ONLY` remain unchanged compatibility values.

The fourth terminology cleanup is now complete: active committed-feedback execution uses
`autoregressive` throughout. The broad runtime family is rooted at
`native_fragment::autoregressive`; inputs/state/publication use `AutoregressiveInput`,
`AutoregressiveState`, `AutoregressiveCommit`, `AutoregressiveFeedback` and
`AutoregressiveSlot`; cold/reuse/fault/observer/receipt ownership follows the same prefix;
fixed-prompt paths use `FixedPromptAutoregressive*`; and public methods/modules use
`autoregressive_*`. The broader old `natural_publication` module was intentionally shortened
to `autoregressive` because it owns the whole feedback lifecycle, not only publication.
Frozen external Qwen27B/Gemma reference schemas, artifact filenames, receipt fields and the
historical `qwen27b_hip_natural` diagnostic entrypoint remain compatibility/provenance
vocabulary; its code now calls the new runtime API without rewriting the frozen evidence
format. Active Qwen35 CUDA/batching and Qwen text diagnostics now use autoregressive
feedback/history/IDs as well.

The fifth cleanup retires `all42` as an engineering term. Active APIs, tests, diagnostics,
comments and receipt labels now say **all state domains** / `all_state_domains`; the number
42 remains only an implementation/model property. The accepted Qwen35 reference archive
coordinates `plain/all42/...`, `all42-prior/...` and key
`all24_all42_post_validation` remain unchanged compatibility/provenance values and are not
runtime vocabulary.

The sixth cleanup retires model-size shorthand from active execution terminology: complete
model execution is **full model** / `full_model`, complete layer coverage is **all layers** /
`all_layers`, and complete vocabulary coverage is **full vocabulary** / `full_vocabulary`.
Active production diagnostics, errors, tests and receipt labels no longer use `full24`,
`all24` or `fullV`. The accepted Qwen35 key
`common_final_fullv_greedy_validation` and the older
`all24_all42_post_validation` key remain unchanged provenance/compatibility values. Real
numeric dimensions such as `full248320` are not terminology and remain untouched.

The seventh cleanup retires fixed-trajectory archaeology from active code: the 47-token
prefill is **prefill 47** / `prefill47`; decode continuation is named by explicit bases or
frontiers; and `canonical47`, `cached47`, `true47` and `case5` no longer appear in active
runtime/test terminology. The frozen batching oracle profile `canonical47-repeated-v1`
remains unchanged as a compatibility/provenance value.

The eighth cleanup retires provider/workload shorthand. The qualified max-eight fixed-prompt
lifecycle is the **eight-session benchmark profile** and keeps the explicit Rust variant
`FixedPromptLifecycleProfile::EightSessionBenchmark`; the smaller profile is `TwoSession`;
its tests/helpers use `eight_session_*` and
its resource contract is stated directly rather than as `active8`. Production temporal
prefill is described as **4-token equal-frontier prefill chunks** rather than `temporal4`.
The generic scheduler's real `prefill_quantum: 32` value remains unchanged; only the opaque
`quantum32` nickname was removed.

The ninth cleanup retires bare row-width/concurrency shorthand from the active semantic
surface. Scheduler/config code uses `cohort_width`; prose uses **singleton** and
**2/4/8-row cohort**; current workload prose uses `active=N` / `concurrency=N`.
`B4`/`B8` remains intentionally in fixed physical ABI/kernel/artifact names and historical
provenance, while `C*` remains only in historical benchmark coordinates. Public CLI help,
high-level scheduler/qualification assertions and the normative batching design now use
explicit row-count language; qualification receipt markers are preserved as evidence
coordinates rather than churned for style.

The tenth cleanup resolves the deferred `ordinary RMS` arithmetic name. The typed operator
is **zero-centered RMS**: the checkpoint stores an offset and runtime gamma is
`1 + checkpoint_weight`. Core contracts/plan nodes are `ZeroCenteredRmsContract` and
`TypedZeroCenteredRms`; CPU numerics are `ZeroCenteredRms` / `zero_centered_rms`; Qwen
operator slices/reference helpers use `zero_centered_*`. Qwen residual RMS and BF16-text /
gated direct-gamma RMS remain separate contract families. The pinned CPU golden bytes are
unchanged; only their owning module/test path moved from `ordinary` to `zero_centered`.
Historical milestone prose retains the old name where it records the terminology current at
the time.

Cumulative terminology validation remains green across strict core/Qwen/GPU/public-CUDA
Clippy plus public CUDA/HIP, CUDA qualification, 27B HIP qualification and CUDA CLI
compilation. Core invocation tests pass 10/10; the Qwen forward suite passes 5/5; focused
zero-centered RMS validation passes core typed-plan 3/3, pinned CPU goldens 2/2 and Qwen
slice/backend 4/4 with the golden SHA unchanged; the standard CUDA GPU lib-test and Qwen35
CUDA integration targets compile; and the frozen
`qwen27b_hip_natural` example compiles against the new autoregressive API. A combined
`cuda-native-qualification,cuda-cublas` GPU unit-test attempt gets past the stale invocation
test accessors fixed here and stops only on the pre-existing `CudaDiagnostic.compiler`
test-matrix drift in unrelated attention/GEMM/RMS tests. The renamed `cuda::text_forward`
static suite also retains its pre-existing 16/17 fixture drift: one assertion expects five
`!isfinite(mean)` / inverse-zero guards while the byte-identical kernel already contained
six at `b88d3b3`. A focused `qwen35-cuda-qualification` public lib-test attempt for the new
eight-session profile tests is additionally blocked before test execution by unrelated
pre-existing test-only drift: `gpu/cohort.rs` references a stale `native` variable and
`qwen_source/execute.rs` still calls the retired `fused_phase_compatible` helper. Production,
qualification and integration compilation remain green; do not repair those test-matrix
issues absent direct relevance. No physical GPU run was added for this behavior-free naming
slice.

### Error architecture — 21 Nov 2026

The cross-layer error contract is now intentionally small and typed:

- `Malformed` = invalid/corrupt external input or artifact.
- `Unsupported` = requested capability/semantic is not implemented.
- `Limit` = deterministic configured/static bounds and arithmetic overflow. It must
  not masquerade as a measured memory/VRAM shortage when requested/available bytes are
  known.
- `Resource(ResourceLimit)` = a real resource budget cannot fit the attempted work.
  It carries resource, operation, exact requested bytes, exact available bytes and an
  optional scope. Shared physical/resource admission includes device identity, owner,
  already-reserved bytes and capacity in that scope.
- `State` = invalid lifecycle/internal state; `Io` retains the underlying I/O cause.

Do **not** parse error prose to choose scheduling behavior. Prelaunch typed
`Resource` failures become capacity/backpressure where the scheduler explicitly owns
that policy; post-submission failures remain fatal. Resource pressure in optional
optimizations may fall back only at an already-defined typed capacity seam.

`ServiceError::inference` is the single public classification boundary:
`Resource -> 503/resource_exhausted`, deterministic `Limit -> 400/limit_exceeded`,
`Malformed/Unsupported -> 400`, and `Io/State -> 500`. HTTP 5xx bodies are
`server_error`, not `invalid_request_error`. Runtime/server reverse adapters use the
shared conversion and no longer wrap `ServiceError::to_string()`, avoiding duplicated
status/code prefixes. Real host/VRAM admission paths in the resource arbiter, resident
allocator/state/checkpoint, CUDA-forward/resident-text admission, HIP source budgets,
shared BF16 Qwen/Gemma paths and backing/model sub-admission now report concrete
requested/available bytes. Static shape/file/queue bounds deliberately remain ordinary
`Limit` errors. Preserve this distinction for new models/backends.

Validation for this slice is green on the default workspace, CUDA-native GPU tests,
Qwen CUDA serving, Qwen HIP serving, Gemma4 HIP and Gemma4 Vulkan shared-BF16 surfaces,
the existing resident-ledger concurrency/pressure tests, resident-text scheduler capacity
tests, focused core/engine/HTTP classification tests and strict Clippy for the changed
production crates. One combined Gemma HIP→Vulkan compile was externally SIGKILLed while
entering Vulkan dependencies; the HIP check had already passed and one cached
Vulkan-only retry passed, so no source failure follows.

## Resume here

### Repository/topology checkpoint

- **Repository/source authority:** use `dotlink---l5`; work only in
  `/home/pvl/spaces/abird/src/zerogravity` on `main`. Do not create another Git
  worktree.
- **AMD large-VRAM execution/test host:** use `dotlink---x2` only for running and
  testing, in the same path `/home/pvl/spaces/abird/src/zerogravity`. Do not use
  X2 as an independent source-authoring branch; source changes land on l5 first.
- Accepted native-GPU harness checkpoint is
  `6b920a0a91e534eae86f81e603cf419dea3b207b`
  (`fix(harness): reuse existing sandbox isolation`). The paged-state lineage
  remains `e6aac53` -> `3041457` -> `398e446`: `e6aac53` keeps the
  lifecycle qualifier cold-only, `3041457` corrected restore qualification,
  and `398e446` added the source-real lifecycle target; `1413f72` is the
  runtime/source-policy checkpoint; `b9b04aa` remains the accepted current-root
  gather/KvInPlace/scatter/Ready candidate execution; `cb57e1d` remains the
  physical pool substrate and `93468b1` remains snapshot/restore/inherited-Ready
  authority. Current main additionally carries the 19 Nov true in-flight
  cancellation implementation/qualification described below.
  `1413f72` added `CudaSourceStatePolicy::{Contiguous,Paged}` at the original
  resident-text loader seam; at that historical checkpoint the old loader spelling
  retained an exact Contiguous wrapper. Current source names the CUDA entrypoint
  `load_cuda_forward_source`; the explicit policy-bearing loader installs hybrid
  Vector/staging/page pools only for `Paged { page_tokens,
  max_append_tokens }`. At `1413f72` Paged was B1-only; current main now permits
  native rows 4/8 through the same hybrid state owner. Zero policy extents still
  fail closed. `HybridStateInventoryPlan` rejects max_append_tokens above state
  capacity, and runtime candidate construction returns `Limit` before checkout if
  a call exceeds the admitted append bound.
  At `1413f72` the high-level/public `ModelLoadRequest` was intentionally
  unchanged and production registration passed `CudaSourceStatePolicy::Contiguous`
  explicitly. The 19 Nov follow-up added an explicit policy-bearing high-level request
  while preserving `QwenCuda` as Contiguous **at that checkpoint**. That fixed default is
  superseded by the21 Nov generic smart-config layer recorded below; the explicit request
  now supplies a user value for that knob under Enabled mode. Pre-source-IO unit tests
  prove invalid paged/native/bound policies return
  `invalid_source_state_policy` before cache/device access.
  The real managed L5 cache is now visible at
  `/home/pvl/.cache/abird/zerogravity/models` with the exact approved0.8B
  snapshot `d7c7bf56...` and2B snapshot `88606494...`.
  The original-HF differential qualifier
  opens one authenticated source, runs Contiguous then Paged sequentially, and
  compares every full-vocabulary logit word plus all42 state words bit-for-bit at
  frontiers page_size-1/page_size/page_size+1/(2*page_size+1). Success marker:
  `P12_SOURCE_PAGED_B1_PASS ... full_vocab=exact all42=exact
  contiguous_reference=true final_shutdown=true`.
  On original0.8B with page_tokens4/max_append_tokens4 it now passes physically:
  `P12_SOURCE_PAGED_B1_PASS page_tokens=4 max_append_tokens=4
  frontiers=3,4,5,9 full_vocab=exact all42=exact
  contiguous_reference=true final_shutdown=true`.
  `398e446` adds a second ignored source-real paged B1 lifecycle qualifier on
  the same authenticated-source fixture. It proves, once inputs are present:
  prelaunch cancellation leaves root/all42 unchanged; dropping a completed Ready
  candidate publishes nothing; cancellation toggled after Ready yields
  `Discarded` with exact rollback; committed execution crosses page_size-1 ->
  page_size -> page_size+1; restore back to the saved pre-boundary and boundary
  roots reproduces exact all42 state; final model shutdown is clean. Success
  marker: `P12_SOURCE_PAGED_LIFECYCLE_PASS ... all42=exact
  final_shutdown=true`.
  Cold original0.8B paged lifecycle is accepted from retained dotlink job
  `9feb1394-1a3b-498b-8f92-fdd3e607c6b3`: normal `exit_code=0`, 168.08s,
  no contiguous preconditioning, full lifecycle marker
  `P12_SOURCE_PAGED_LIFECYCLE_PASS page_tokens=4 contiguous_warmup=false
  prelaunch_cancel=true abandoned_ready=true cancel_after_ready=true
  restore_before_boundary=true restore_at_boundary=true all42=exact
  final_shutdown=true`.
  `e6aac53` removes the temporary warmup branch and lifecycle stage spam; the
  retained qualifier is cold-only by construction and emits `cold_load=true`.
  The first completed warm run exposed a qualifier bug: restore intentionally
  publishes one fresh root and increments session epoch while restoring the saved
  frontier/state; the test incorrectly required equality with the old snapshot
  RootBinding. `3041457` fixes the assertion to require same source/session,
  root+1, epoch+1, saved frontier and exact all42 bytes.
  Several other0.8B cold attempts terminated intermittently with deployed
  dotlink **v0.17.3** reporting `status=completed`, `exit_code=null`,
  `success=false`, with no Rust failure, OOM/Xid/coredump, live child or VRAM
  residue. This is **not** a ZeroGravity cold-load blocker. Revalidation against
  the deployed `release/0.17` tree matters here: v0.17.3 does **not** contain
  main's newer `SharedProcessExit::Signal` projection. Its runner records
  `JobCompletion::completed(status.code(), status.success(), ...)`; Rust
  `ExitStatus::code()==None` means the process dotlink waited on was terminated
  by a signal, but v0.17.3 discards the signal number. The child dotlink waits on
  for a sandboxed shell command is Bubblewrap. Timeout/cancel/spawn/stdin/spool
  failure paths use distinct non-`Completed` statuses, and the normal
  `Completed` path disarms the process-tree guard before recording completion,
  so the observed null-exit jobs were not dotlink timeout/cancel results.
  Controlled probes further distinguish the case: SIGKILL of an inner child is
  observed by the surviving shell as status137, and SIGKILL of the sandbox shell
  is translated by surviving Bubblewrap to numeric exit137; the intermittent
  jobs instead have null exit code, meaning the Bubblewrap child itself was
  signalled. At the exact failures `dotlink.service` reported memory
  `oom=0`, `oom_kill=0`, pids `max=0`; no dotlink restart/poll failure was
  observed. The source of that external signal is not proven. A clean reproduction
  with no inner tmux/setsid/heartbeat supervisor
  (`native-gpu -> env -> test binary`) still produced the same null-exit shape,
  so the earlier tmux/setsid harness is not required to trigger it. Classify such
  runs as connector-infrastructure incomplete and require a full terminal pass for
  model evidence. Do not add loader workarounds or warmup.
  All Cargo compilation/check/clippy in this slice used
  `RUSTC_WRAPPER=/run/current-system/sw/bin/kache`.
  Final-current available validation is green: four hybrid admission units, all
  eight RTX3060 pool/paging/in-place cells, top-level `qwen35-hip`,
  CUDA tests and strict production clippy. A combined cross-feature wrapper
  disappeared after a terminal-green HIP command while entering the next build;
  separate CUDA commands completed green. One first GPU relink also disappeared
  late; one bounded warm lib-only retry completed.
  Production ordinary sessions still construct contiguous KV today; no serving
  selector, direct paged attention, prefix reuse or public capability changed.
  Parent `8e6f14d3defc0aa2e542c26ffb9ccc7c736f1fcb`
  (`feat(serving): enable qualified temporal B8 prefill`) remains the current
  production temporal checkpoint. The retained width-parameterized qualifier
  fully passes original0.8B and2B B2/B4/B8 with CPU/B1/all42,
  rollback/publication, greedy bits/producer and final reserved0. Production
  HTTP passes B4/B8 on both original sources and retained2B B2; B2/B4/B8 all
  use 4-token equal-frontier prefill chunks while decode remains unchanged.
  B2 production HTTP still proves prompt10 executes as native chunks4+4+2 with
  three publications, two state-only + one greedy, exact CPU text/usage, zero
  failures/refusals and final41MiB. Parent `9dab004` added the reusable temporal
  candidate-chain primitive; `7dea460`/`ba7e376` establish fair aged
  decode/prefill scheduling.
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
  **2B temporal B2/B4/B8 each pass full debug qualification** with the same
  private-candidate/root contract as0.8B. Two unsupervised B4 launches vanished
  before temporal work, but the materially different heartbeat-supervised run
  completed B4 exactly; this establishes an execution-envelope issue rather than
  a B4 semantic failure. On signed `8e6f14d`, original0.8B and2B B8 HTTP now
  each pass exact4+4+2 native B8 execution and accounting; the earlier vanished
  0.8B B8 attempt remains incomplete rather than being relabeled. The retained
  2B B2 HTTP cell also passes exact4+4+2. Production `NativeText` now selects
  4-token equal-frontier prefill chunks for **B2, B4 and B8**, with explicit
  native prefill cohort eligibility for all three widths. These physical proofs
  use the debug Rust profile; canonical thin-LTO/codegen-units=1 B2 is
  still open after another heartbeat-supervised release attempt disappeared
  while compiling/linking `zerogravity-cli`, before the test ran and without a
  terminal child status, Rust diagnostic, OOM/Xid, live child or GPU residue.
  Do not loop that unstable build. Paged-state control/ownership/materialization/
  typed-KV gather/scatter/domain publication/in-place execution/hybrid candidate/
  source-policy plumbing is now
  `0027c28` + `54340a6` + `1eb6e0f` + `3ac2059` + `dc6a188` +
  `53336b3` + `ef9e4d4` + `93468b1` + `cb57e1d` + `b9b04aa` +
  `1413f72` + `398e446` + `3041457` + `e6aac53`.
  **18 Nov original2B source-real paged B1 acceptance is complete.** Signed
  `6b920a0` (`fix(harness): reuse existing sandbox isolation`) now owns the
  native-GPU harness repair. `scripts/native-gpu.py` detects an existing
  Bubblewrap PID/user sandbox instead of nesting another user namespace. If
  Bubblewrap already owns a private network namespace it executes directly;
  otherwise it installs a fail-closed x86_64 seccomp socket-creation filter
  (including io_uring socket-creation bypass prevention) and executes directly.
  Signed `4688cba` extends this generically: if the caller has already entered a
  child netns, the harness accepts it only when `/proc/self/net/dev` is exactly
  loopback, PID1's network view differs, and no inherited socket FDs exist. This
  lets a shared-net connector supply `bwrap --unshare-net --bind / / --dev-bind
  /dev /dev --` without weakening `--require-private-network` or making the
  harness create another user namespace. Outside Bubblewrap it retains the original
  `unshare --user --map-root-user --net` route.
  `--isolation-info` reports the exact planned mode and
  `ZG_NATIVE_GPU_ISOLATION` records it in the child. Commands requiring TCP/UDP
  loopback must pass `--require-private-network`; the shared-network seccomp
  fallback refuses them rather than weakening isolation. `scripts/check.sh`
  now fails immediately in that fallback, and `AGENTS.md` uses
  `native-gpu.py --require-private-network -- bash scripts/check.sh`.
  Focused validation is now31/31 offline harness tests plus Python compile,
  real current-sandbox `--isolation-info`, AF_UNIX allowed/AF_INET denied,
  io_uring_setup denied with EPERM, exact direct exit73 propagation, direct TERM
  propagation as status143, shared-net private-network refusal, and real child-net
  `--require-private-network` loopback+CUDA smoke. Direct invocation still refuses
  this L5 outer shared-net profile; the explicit child-Bubblewrap wrapper is the
  accepted way to supply the required private loopback namespace.
  The upgraded connector now reports external signal numbers explicitly and, in
  v0.17.4, emits one lifecycle log line for every stop it initiates. Follow-up
  connector review found only the explicitly requested `job_cancel` stop in the
  relevant window; a silent `sleep 300` remained alive until that cancellation.
  Therefore the earlier SIGKILL9 observations must **not** be interpreted as a
  dotlink quiet-job watchdog or as evidence that periodic output is required.
  They are best classified as surrounding harness/orchestration artifacts outside
  dotlink's stop paths. Host/service memory limits were unbounded,
  OOM/high/max/pids events were0, PSI was0, and GPU retired cleanly, so they also
  remain non-model evidence. The heartbeat sidecar used below was an operational
  stabilization technique for our harness, not an accepted runtime requirement.
  Under that envelope and approved snapshot `88606494...`,
  page_tokens4/max_append_tokens4:
  - differential job `38c1f321-0347-4a80-b2dc-c6d9d46d0249` completed
    exit0 in703.17s with
    `P12_SOURCE_PAGED_B1_PASS page_tokens=4 max_append_tokens=4
    frontiers=3,4,5,9 full_vocab=exact all42=exact
    contiguous_reference=true final_shutdown=true`;
  - cold lifecycle job `744f36fd-b145-47b9-aa61-084e7d4b414b` completed
    exit0 in438.18s with
    `P12_SOURCE_PAGED_LIFECYCLE_PASS page_tokens=4 cold_load=true
    prelaunch_cancel=true abandoned_ready=true cancel_after_ready=true
    restore_before_boundary=true restore_at_boundary=true all42=exact
    final_shutdown=true`.
  Therefore both approved original0.8B and2B now pass source-real paged B1
  full-vocab/all42 boundary parity and cold lifecycle semantics.
  **19 Nov P12-C true in-flight CUDA cancellation is accepted on original0.8B
  paged B1.** A qualification-only launch witness pauses the original ordinary
  CUDA flight only after real gather/body/scatter enqueue and before Ready. Known
  cancellation waits for physical status readback, marks no candidate Write
  initialized and creates no Ready: root/all42 remain exact, Vector/staging/page
  resources stay held through completion, pool occupancy returns exactly, replay
  commits and final shutdown returns RTX3060 to37MiB. A paired deliberate drop at
  the same submitted cut proves the accepted fail-safe unknown path: root binding
  unchanged, service fenced, event pending,396 handles and source completion
  owners quarantined, no pool reuse/recovery. Two external-SIGKILL attempts before
  semantic completion remain incomplete; direct known-completion and final
  heartbeat-assisted unknown-completion runs exit0. Heartbeat is optional transport
  plumbing only. Run physical cells through `native-gpu`; reserve
  `--require-private-network` for genuine TCP/UDP loopback gates.
  **Historical19 Nov selector checkpoint (superseded for default selection on21 Nov).**
  Public `QwenCudaStatePolicy::{Contiguous,Paged}` was initially available only via
  `ModelLoadRequest::QwenCudaWithStatePolicy`, with `QwenCuda` then exactly Contiguous. **Native paged B2/B4/B8 state transport is now implemented**
  by reusing `OrdinaryCandidateState`: each lane gathers paged KV into its admitted
  staging, executes the existing native cohort body, scatters under the same flight,
  promotes only actual Vector/fresh-COW Writes, then materializes canonical paged
  Domains after completion. Contiguous B8 keeps its private-KV fast path; paged
  `KvInPlace` cleanly uses the existing per-lane append rather than adding another
  paging kernel. Focused policy tests pass, strict CUDA clippy and CUDA-serving
  composition checks pass. The compact original0.8B Contiguous-vs-Paged B2/B4/B8
  prefill+decode/all42 cell is now physically accepted: the same test completed
  exit0 in306.84s with `P12_SOURCE_PAGED_NATIVE_TRANSPORT_PASS widths=2,4,8
  prefill=exact decode=exact all42=exact contiguous_reference=true
  final_shutdown=true`. The earlier two external-SIGKILL attempts remain incomplete,
  not failures. Public SDK integration is also physically accepted:8 concurrent
  requests,4-token page-boundary prefill and2 output tokens produced actual native
  B8 publication in both Contiguous and Paged, identical token IDs and clean
  shutdown (`P12_SOURCE_PAGED_SDK_PASS ... contiguous_cohorts={8}
  paged_cohorts={8} tokens=exact final_shutdown=true`, exit0 in340.78s). Strict
  serving-test Clippy passes. **Public HTTP paged serving is now physically
  accepted too** under the explicit child-private-net wrapper:8 concurrent
  4-token/page-boundary requests with2 outputs observed actual native B8 for both
  Contiguous and Paged, returned identical token IDs and shut down cleanly.
  `P12_SOURCE_PAGED_HTTP_PASS ... contiguous_cohorts={8} paged_cohorts={8}
  tokens=exact final_shutdown=true`, exit0 in358.07s. Strict HTTP/SDK test-target
  Clippy passes. A fresh matched-vLLM attempt did not run: the exact pinned
  `f57d77f5...b4b0dbe` layer store is still present (~25GiB), but its image index is
  no longer registered for this connector's Podman view (`image not known` after a
  temporary user+mount namespace gets past the outer permission/mount issue). The
  canonical managed HF0.8B file tree remains exact at revision `2fc06364...` with
  safetensor `04b1c301...`; do not rebuild/re-register orphaned image layers just to
  manufacture a benchmark. The first full isolated `scripts/check.sh` attempt through the
  child-net envelope found one real source-boundary bug: `HybridStateInventoryPlan`
  is shared by production `cuda-native`, but `ResidentPlaneLayout` was not imported
  for that feature. The import cfg is corrected while CUBLAS-only map/lease imports
  stay gated; targeted `cargo check -p zerogravity-gpu --no-default-features
  --features cuda-native` passes. A warmed full-gate rerun then progressed beyond
  that boundary and was externally SIGKILLed during the later ordinary-Qwen feature
  compile with no compiler/test diagnostic, so the monolithic gate remains
  incomplete rather than failed. **Complete-domain prefix fork/COW substrate is now
  accepted:** `ResidentSession::fork_shared_prefix` only seeds a fresh destination,
  validates the same service/source/capacity/full domain set, publishes through the
  destination's own `begin_validated_replacement`, and clones immutable domain
  ownership without replay/readback/copy. `ResidentTextSession` additionally requires
  the exact same resident model. Original0.8B paged page4 physically passes a
  3-token partial-page prefix fork then1-token COW continuation versus independent
  replay: logits/all42 exact, cached snapshot immutable, independent root/session,
  clean shutdown (`P12_SOURCE_PAGED_PREFIX_FORK_PASS`,179.13s). **The bounded
  semantic prefix lookup/eviction layer is now implemented in the model-neutral
  engine.** `PrefixCompatibility` requires a bound source plus nonzero grouped
  fingerprints for execution (loader/quant/numerical), input
  (tokenizer/template/processor/adapter), position, state schema and sharing scope.
  `PrefixCache<U,T>` keeps immutable `Arc` payloads under explicit entry/weight
  bounds, does allocation-free newest-first lookup, returns the greatest exact
  semantic-unit prefix, refreshes exact duplicates and evicts FIFO deterministically.
  It never grants physical restore: the backend must still run the accepted full-
  domain validation/fork path. Focused state-cache tests, strict engine Clippy and
  production `qwen35-cuda` composition check pass. **The source-only common-frontier
  reconciliation primitive is now also in the same model-neutral engine layer.**
  `PrefixDomainCheckpoint` describes only a required domain's retained-through
  frontier and zero-based checkpoint alignment; `reconcile_prefix_frontier` computes
  the greatest request-bounded frontier legal for every domain using an allocation-free
  O(domains), overflow-safe bounded LCM. It returns0 when no nonempty aligned common
  checkpoint exists and grants no physical restore/copy authority. **The separate exact
  partial-tail COW ownership path is already complete.** Core `PagedBlockTable::share_prefix`
  shares the immutable physical page while narrowing only its logical valid extent;
  `fork_for_append` replaces a partial tail with a fresh target and a copy descriptor that
  retains both handles. Resident paged append exposes the old page as submitted Read and
  the fresh page as Write, and `ResidentSubmission` clones those exact leases for the
  armed flight until physical retirement. Candidate pages cannot publish before Ready;
  `publish_domain_swap` atomically installs the validated domains/root and displaced
  owners drop only after commit. Retained physical COW/materialization/page-boundary and
  source-real prefix-fork qualification already prove source immutability, fresh
  generations, exact valid-tail copy/append, zero tail and full retirement. No duplicate
  COW abstraction or new fixture is needed. The existing shared `ResidentPool` root path
  was re-audited at the same checkpoint:
  fresh logical lease IDs/generations are issued over retained backing, readbacks/Ready
  owners retain the checkout, and reuse occurs only after ledger-proven idle backing;
  unknown retirement poisons/fences/quarantines rather than returning a set idle.
  **Physical alias identity now has one shared core representation too.**
  `device_events::DeviceBufferIdentity` identifies an opaque process-unique ledger
  namespace + real allocation + exact byte range; direct buffers use their lease
  allocation while pooled/views use the retained backing allocation. The extra ledger
  namespace is required so independent services that both mint local ID1 cannot alias by
  value. The ledger is the sole constructor/query authority and its production
  alias-conflict path calls the same `overlaps` primitive, so there is no parallel
  hash/handle/range definition. Shared resident `overlaps_any` now checks lease/service
  ownership and resolves one reference against a lease slice under a single ledger lock,
  with no heap allocation or per-lease identity field. Both HIP/CUDA source-page alias
  guards use it, and the native B4 embedding preparation now uses the same primitive for
  output/table/status and feedback alias checks instead of reconstructing backend-handle
  ranges. The four-packet diagnostic pairwise check reuses `overlaps_any`. The production
  ordinary-CUDA body now also uses canonical identity for the complete prepared resident
  lease set via `ResidentService::pairwise_disjoint`: one ledger lock, no heap allocation
  inside the helper, and no backend handle inspection. This cleanly separates physical
  ownership truth from the still-required `BufferRange`/`OwnedRanges` kernel binding
  geometry. Strict CUDA Clippy/serving compile pass and the existing native-B8 public SDK
  Contiguous/Paged cell remains exact with clean shutdown after the migration; no new
  fixture was added. Six backing-ledger tests include cross-ledger separation. The next
  real transfer consumer is now migrated too: prepared streamed BF16 vocabulary-head
  binding validates the weight-staging/input/output lease triple with the same
  `pairwise_disjoint` ledger identity before its existing two-slot authenticated
  retirement ring can run. This closes the logical-ID-only alias hole without changing
  the ring, host-window retention, same-stream overwrite ordering or kernel-handle
  geometry. Production `cuda-cublas` and public `qwen35-cuda` checks pass. The retained
  public-SDK 0.8B context1024/prompt48 replay cell also passes physically after recreating
  the documented CUDA13.4 include view: exact two-token replay, distinct sessions, exact
  producer frontiers and clean shutdown in14.42s. The next production source consumer is
  now canonical too: HIP resident-text cold workspace leases use
  `pairwise_disjoint`, and request root validation uses
  `pairwise_disjoint_extension` to compare only new roots against the already-disjoint
  workspace and earlier roots under one ledger lock. This avoids a workspace² hot-path
  recheck while preserving exact handle ranges solely for execution-view geometry. HIP
  GPU/public composition checks compile, existing resident-text admission tests pass2/2,
  and the retained CUDA context1024/prompt48 public-SDK cell passes again in15.23s after
  the shared helper refactor. X2 HIP physical execution remains behind the recorded ROCm
  ABI environment gate. No new alias fixture was added. Remaining identity adoption is
  incremental for additional real graph/transfer consumers; do not cache identity into
  every lease merely for future diagnostics.
  **Ordinary CUDA source-page transfer now uses one retained authenticated owner.** The
  exact verified `TensorWindow`/window-cache result is wrapped only in an `Arc`, installed
  in the page's existing pending state before enqueue, and handed to the original CUDA
  actor's generic bounded owned-window H2D primitive. The unchanged page+consumer
  `sync_buffers` checkpoint therefore proves both source DMA and consumer completion before
  that owner can retire. This supersedes `b839920`'s source-side pinned staging: inspection
  found CubeCL pinned `Bytes` has no cheap `duplicate`, so the required clone became a Native
  deep copy and generic `Client::write` then staged to pinned again. The new path removes
  that payload `Vec`/`Bytes`/pinned chain, plus the now-unused transfer-staging reservation
  API, without adding a new pool or ring; HIP/non-CUDA serial storage is cfg-unchanged.
  Matched physical feedback on the same temporary 2B/32K Contiguous/Streamed prompt48 SDK
  cell: `b839920`39.36s; cache-promotion candidate41.00s rejected; moving the pinned owner
  into the generic writer/drop queue44.88s rejected; final retained-owner direct actor H2D
  **38.46s** (~2.3% faster than `b839920`). Exact replay/frontiers, independent sessions and
  clean shutdown remain intact. The older direct SSD→pinned/cache-bypass candidate remains
  rejected at41.77s. A 22 Nov follow-up tested the next simple overlap candidate without
  adding fixtures: reuse the existing two-slot retirement-ring state machine on CUDA with
  same-stream `OwnedCudaEvent` marks, the same single 1MiB device page, and exact retained
  `TensorWindow`/output owners. Strict CUDA/HIP checks and the existing retirement/accounting
  tests passed, but a same-tree matched opt1/codegen64/non-LTO explicit
  Contiguous/Streamed2B32K/prompt48 A/B measured **38.323s serial vs39.281s ring**; the
  Qwen one-page consumer/selector was ~2.5% slower and remains rejected. A separate24.267s
  Auto run selected paged state plus resident weights and therefore is not evidence for this
  streamed-path candidate; it does confirm the default smart policy already avoids this
  fallback when residency fits. The reusable mechanism is deliberately retained dormant:
  backend completion witnesses now implement the shared `RetirementMark` contract directly:
  CUDA `OwnedCudaEvent` and HIP `OwnedHipTransferRetirement`. The zero-cost aliases
  `CudaTransferRetirementRing<Owner>` / `HipTransferRetirementRing<Owner>` specialize the
  existing generic `TransferRetirementRing` without wrapper allocation. The ring owns only
  completion marks plus caller-supplied exact owners; model/session/publication policy stays
  with the future consumer. This preserves the primitive for Gemma, larger streamed models,
  other CUDA/HIP devices or different workload shapes without selecting it for Qwen. Temporary policy overrides/include view were removed
  and no new evidence fixture was retained. Do not retry the rejected Qwen one-page selector
  without new evidence; a future consumer may reuse the primitive after separately proving
  its owner graph/admission and a measured benefit. Meaningful Qwen streamed overlap beyond
  this still requires separately admitted extra device-page/copy-stream authority. Future
  direct/async IO must compose at the authenticated cache-miss boundary.
  **Scoped worker prefix reuse is
  now implemented and physically accepted.** `PrefixSharingScope` is a validated
  nonzero Copy ID. Existing `Request` stays unchanged; ordinary `request_text`
  remains isolated, while the explicit `request_text_scoped(request, scope)` path
  carries authorization alongside the queued request. Generic executor hooks restore
  only immediately after opening a fresh state, and the worker verifies restored
  cursor == actual executor frontier; cache insertion runs only after an already
  committed state-only prompt step. Qwen owns one model-local paged-only cache
  (`active` entry cap, `active*context` semantic-token weight); contiguous state pays
  no snapshot-retention cost. Because the cache instance is owned by one exact loaded
  Model and `ResidentTextSnapshot` additionally requires that exact resident model,
  loader/numerical/tokenizer/position/state-schema cannot cross instances; sharing
  scope is the only request-varying authorization class. The first physical 0.8B
  page4/max_append64 worker cell uses a65-token prompt: warm scopeA consumes65 model
  tokens, same-scope repeat restores the64-token complete-domain prefix and consumes
  only1, different scopeB consumes65 again; generated ID is identical and shutdown
  is clean (`P12_SOURCE_PAGED_PREFIX_CACHE_PASS`,172.98s). An initial run exposed a
  real lifecycle issue: cache-held snapshots outlived resident retirement; executor
  retirement now clears the cache before source shutdown. **Trusted HTTP scope
  injection is now accepted.** `ServerBuilder::prefix_sharing_scope` owns one optional
  validated scope for the whole server instance; connection captures carry only that
  Copy value. HTTP parsing exposes no scope field/header. `TextService` defaults to a
  fail-closed `prefix_sharing_unsupported` error for scoped admission, while the
  engine Runtime/Router explicitly implement it; the focused server suite is7/7 and
  strict serving Clippy passes. Original0.8B page4/max_append64 physical HTTP proof
  uses ordinary OpenAI JSON only: server scope7 warm consumes65 model tokens, repeat
  consumes1, a new server over the same runtime/cache with scope8 consumes65, all
  output IDs match and shutdown is clean (`P12_SOURCE_PAGED_PREFIX_HTTP_PASS`,
  171.97s). No client can select/override the scope through this HTTP API. **Bounded
  live-snapshot eviction and explicit stale-scope cache invalidation are now accepted.** The
  model-neutral cache exposes `PrefixCacheEviction { entries, weight }` and
  `invalidate_scope`; trusted control routes that through Executor -> typed Registry
  -> Router -> engine/public Runtime as `invalidate_prefix_scope`. Cache invalidation
  is explicit: stopping a server or selecting another scope does not purge prior
  state. The embedding owner must first stop issuing/drain a stale scope, then call
  `invalidate_prefix_scope`; this purges currently retained entries but does not
  retroactively cancel already-admitted work or invalidate a snapshot clone already
  owned by such work.
  Qwen now admits `2*active` cache entries while retaining only `active*context`
  semantic-token weight, so count and weight bounds are independently meaningful.
  Original0.8B page4/max_append64 physical `active=1` proof shows: three distinct
  prefix64 snapshots trigger FIFO count eviction (newer hit, oldest miss); explicit
  purge releases exactly2 entries/128 weight and the invalidated scope misses afterward;
  a prompt321 trajectory hits the512-weight bound, retains prefix320 for a1-token hit,
  but evicts prefix256 so prompt257 replays all257 tokens; final purge releases2
  entries/448 weight, repeated purge returns zero, and shutdown is clean
  (`P12_SOURCE_PAGED_PREFIX_EVICTION_PASS`,179.12s). This is cache-retention pressure.
  **Externally imposed resident page-pool pressure is now physically accepted too.**
  The qualification-only holder checks out real pages from the same finite
  `ResidentPagePool`; production only exposes read-only diagnostics. With original
  0.8B page4/max_append64, a cached prefix64 leaves192 pages occupied; the qualifier
  holds the remaining5952 pages and fills the6144-page pool. The next cached tail is
  refused prelaunch as `Capacity`, the root stays unchanged, vector/staging candidate
  leases unwind to their prior occupancy, and the cached snapshot remains valid. After
  releasing pressure, the same tail commits and its complete vocabulary logits match
  the pre-pressure baseline bit-for-bit; all allocated pages are idle before clean
  shutdown (`P12_SOURCE_PAGED_PREFIX_PAGE_PRESSURE_PASS`,168.51s). One earlier
  external SIGTERM at ~8s during startup remains incomplete only. Minimal production
  diagnostics now report current `prefix_cache` occupancy/bounds and `paged_pool`
  allocated/idle/occupied/max/page-bytes under each Qwen source model; there are no
  cumulative hot-path counters. This qualifies actual resident page-pool exhaustion
  and recovery, not host/device OOM.
  **Fresh 19 Nov review + overlap prerequisite:** the entire recent prefix/paging/
  serving lineage was re-read from the live tree and reviewed across engine/router/
  serving/GPU ownership, scope isolation, alias deduplication, cache invalidation,
  pressure unwind, cancellation and feature boundaries. No prefix correctness or
  authorization regression was found. The review did find the hard overlap boundary:
  `PhysicalQueue` still owns exactly one `active: Option<u64>` region per card, and
  `ResidentSubmission` only retires a physical region after its trusted readback/
  validation path completes. The current Qwen CUDA source also uses one pinned
  `OwnedCudaQueue` logical stream. Therefore worker-level or scheduler-level
  prefill/decode concurrency would still serialize physically (or would weaken
  completion safety if detached naively); do not claim overlap from more runnable
  rows/threads.
  As the first independent prerequisite, the patched CubeCL CUDA layer now supports
  `OwnedCudaQueue::fork_stream` for a same-service client clone and a non-clone
  `OwnedCudaEvent` that records/queries/waits/destroys one event on that exact actor/
  logical stream. No second CUDA service/context is created and cross-stream buffer
  dependencies remain the caller's responsibility. Physical qualification on L5
  enqueued disjoint8MiB writes on primary and fresh secondary streams, recorded both
  events before waiting either, observed the secondary event still not Ready at the
  first query, then waited both and read back both buffers exactly:
  `P12_CUDA_SECONDARY_STREAM_EVENT_PASS same_service=true streams_distinct=true
  disjoint_buffers=true events_recorded_before_wait=true primary_ready_before_wait=true
  secondary_ready_before_wait=false readback=exact`. This proves a live independent
  secondary stream/event substrate, **not** prefill/decode performance overlap and
  not multiple resident regions in flight. **The event/lease-backed resident in-flight
  owner is now accepted too.** `ResidentCudaInFlight<'a>` owns the original armed
  `ResidentSubmission<'a>` plus one same-service `OwnedCudaEvent`; it does not own the
  physical queue slot and creates no Ready/publication authority. Conversion is an
  explicit unsafe boundary: `into_cuda_in_flight(&OwnedCudaQueue)` records the event
  itself on the supplied queue, and the caller must prove that exact logical stream
  already received every command covered by the submission and that no later command
  will be appended. Known completion
  waits/queries the CUDA event, then still runs the existing trusted readback +
  validator path before write initialization/Ready. Cancellation marks the existing
  logical device event cancelled, waits the CUDA event, retires event pins + physical
  guard without initializing writes and without constructing Ready. Dropping unresolved
  ownership first drops the armed submission, preserving completion-unknown fencing,
  exact lease quarantine and completion-owner retention; the CUDA event token is
  destroyed only afterward. Physical L5 lifecycle proof is terminal-green:
  `P12_CUDA_RESIDENT_INFLIGHT_EVENT_PASS known_readback=exact known_initialized=true
  cancel_retired=true cancel_initialized=false unresolved_fenced=true
  unresolved_initialized=false completion_owner_retained=true quarantine_owner_delta=1
  queue_width_unchanged=true`. **The reusable bounded multi-flight substrate is now
  accepted.** `PhysicalQueue` uses a bounded active-ID set with configurable active
  capacity, while the production registry intentionally remains at active capacity1
  until model adoption. `PhysicalServiceBudget` now multiplexes its one prepaid
  envelope across overlapping known flights with a tiny shared counter: first flight
  marks the retained lease Submitted, intermediate flights add no charge/phase
  transition, and the last known completion marks it Ready; any unknown flight still
  fences the same envelope/card. Existing `DeviceEventLedger::submit` remains the
  access-conflict authority—no duplicate queue-level alias checker was added.
  Host contracts pass (queue6/6, physical-service5/5). Physical L5 proof acquires two
  active slots from a bounded2 queue, submits two disjoint resident outputs on primary
  and secondary streams under the same prepaid envelope, records both completion
  events, retires the second flight before the first and reads both8MiB outputs exactly:
  `P12_CUDA_RESIDENT_DUAL_FLIGHT_PASS active_capacity=2 same_prepaid_envelope=true
  streams_distinct=true disjoint_outputs=true events_before_completion=true
  reverse_retirement=true readback=exact outputs_initialized=true service_reusable=true`.
  Next move directly into the real Qwen/source execution path and measure actual
  prefill/decode multi-flight behavior; do not add another synthetic evidence matrix
  unless that integration finds a concrete missing invariant.
  **20 Nov realistic serving feedback:** rather than adding another harness, the existing
  public B8 SDK qualification now uses eight concurrent original0.8B requests at48
  prompt +3 output tokens, warms once on the loaded model, and times the second wave
  using existing token publication clocks. Both contiguous and paged form native B8
  and return exact-identical tokens. Steady measured wave: contiguous4888.231ms
  (4.910 output tok/s), mean token commits [4523.086,4705.276,4886.792]ms; paged
  5080.588ms (4.724 output tok/s), commits [4567.294,4818.167,5075.042]ms. Thus
  fresh prefill/TTFT (~4.5s) dominates; steady B8 decode intervals are ~182ms
  contiguous (~44 output tok/s) and ~251/257ms paged (~31-32 output tok/s). Warmup
  does not remove the regression. Retained historical eager-vLLM B8 is72.20 output
  tok/s on47-50 input/3 output, but the exact vLLM image is not currently runnable,
  so this is directional history rather than a fresh matched comparison.
  **Correction after interrupted-phase revalidation:** the previous residency diagnosis
  was wrong. Exact admission arithmetic for original0.8B/context512 shows the fully
  resident two-root plan at1,652,990,965B and the fully resident18-root `active=8` plan at
  2,076,877,813B. B8 cohort reserve is51,210,272B, greedy reserve11,904B, leaving a
  fit cap4,780,616,032B under the4,831,838,208B provider envelope; the fitted layout
  remains `resident_weights=true`. Streaming/root pressure therefore does **not** explain
  the ~4.5s TTFT. The real hot structure is `execute_prefill_cohort`: projections and
  stateless RMS are physically shared, but remaining stateful operators execute in a
  lane loop. Decode already has dedicated cached GDN cohort lowering and single-token
  B4/B8 attention; prefill has no general multi-token attention/GDN cohort primitive.
  The stateless QK-RMS/gate/multiply/SiLU cohort lowering was restored and retained:
  it is localized/composable, preserves lane-specific rotary/root authority, and the
  latest exact warm48+3 B8 run improved the retained no-extension baseline from
  4888.231→4696.591ms contiguous (~3.9%) and5080.588→4907.715ms paged (~3.4%).
  This does not solve the dominant stateful prefill bottleneck, but the gain is useful
  and compounds with later work. The old fast `native_cohort` path is also not a reusable general answer:
  its cold envelope is explicitly B4 and initial phase is fixed around47x4=188 rows;
  the serving loader remains qualification-gated. Next implementation target is shared,
  model-neutral **multi-token stateful prefill batching**. Multi-token attention is now
  retained: `Bf16CausalAttentionBatch4::new_ragged_rows` generalizes the existing B4
  kernel to equal multi-row private sequences, and ordinary B8 prefill composes two B4
  groups while incompatible/ragged cases keep the old per-lane fallback. The real
  warm48+3 B8 run stayed exact but TTFT was effectively flat versus the retained
  stateless baseline (contiguous4696.591→4701.433ms; paged4907.715→5022.809ms), so
  attention is not the dominant remaining cost.
  **Multi-token GDN prefill is now accepted.** The shared GDN runtime gained explicit
  `TextPrefillCohort { rows, lanes }` geometry, sequence-aware controls/cache/state
  workspace, strided packed recurrent gather/scatter, and a width-scoped cohort GDN
  workspace owned by `OrdinaryCudaCohortValues`. Ordinary B4/B8 prefill imports each
  private Conv/Recurrent root through existing typed `StateTransport`, executes one
  shared cohort GDN runtime over packed activations, then exports each candidate root;
  no scheduler/root/publication authority moved into the kernel layer. Runtime word
  transport keeps the old1,000,000-word global contract; only its dedicated runtime
  transport path has a2^21-word bound, exactly covering B8 recurrent state.
  Real warm48+3 B8 exact serving moved dramatically versus the attention checkpoint:
  contiguous4701.433→1007.301ms (~78.6% faster), paged5022.809→1160.106ms (~76.9%);
  mean TTFT4341.373→616.681ms contiguous (~85.8% lower) and4503.152→649.944ms paged
  (~85.6% lower). Those numbers were produced by Cargo's unoptimized test profile and
  are correctness/relative-debug feedback only; they must not be used as serving-perf
  baselines. Rebuilding the **same commit and same existing 8x48+3 SDK cell** with the
  historical non-LTO/opt1/codegen64 host profile (`CARGO_PROFILE_TEST_OPT_LEVEL=1`,
  `CARGO_PROFILE_TEST_CODEGEN_UNITS=64`, `CARGO_PROFILE_TEST_LTO=false`, debug info0)
  changes the result without any source change: contiguous183.932ms/130.483 output
  tok/s with commits [133.297,158.866,183.617]ms; paged242.488ms/98.974 tok/s with
  commits [153.943,198.058,241.371]ms. Exact tokens/native B8 remain unchanged.
  Contiguous is therefore already well above the retained historical eager-vLLM B8
  72.20 tok/s and old ZG76.77 tok/s on the same short47-50+3 workload class, although
  that vLLM number is still historical rather than a fresh matched rerun. Approximate
  decode-only cadence is ~25ms/B8 cohort contiguous (~318 output tok/s) and ~44ms paged
  (~183 output tok/s). Do **not** optimize against debug-profile timing; future serving
  performance feedback must use a matched optimized host profile.
  **Continuous refill/regroup is now physically exercised and one banding bug is fixed.**
  The production worker already refills freed active slots from the admitted FIFO before
  each wave and recomputes compatible native bands. A temporary public-SDK probe used
  four1-token rows + four8-token survivors + four queued8-token rows on an `active=8` / `cohort_width=8`
  model. It first exposed a real width-reduction defect: the width8 cohort arena retained
  only the B8 GDN compiled-ABI layout, so a legal B4 band bound `PackedBa` against a
  32768-byte B8 ABI while the B4 activation was correctly16384B. The fix retains both
  B4 and B8 `GdnRuntimeWorkspaceLayout` descriptors for a width8 arena while allocating
  physical GDN scratch once at the maximum width; active plans select the exact band
  descriptor with zero per-token allocation. B4 on a width8 model then passed with every
  token reporting native width4. The warmed refill probe passed contiguous and paged with
  survivor/queued width sets `{4,8}`, proving actual B8→B4→refill/prefill→B8 regrouping.
  Matched opt1/codegen64/non-LTO goodput for68 generated tokens was380.379ms /178.769
  output tok/s contiguous and498.212ms /136.488 paged. The temporary probe was removed.
  **The first optimized paged-gap slice is now implemented and physically measured.**
  Paged KV gather and fresh COW/page initialization previously allocated host-zero vectors
  and uploaded complete staging/page extents on every use. They now reuse the existing
  pinned runtime word-zero CUDA transport, chunked at its admitted2^21-word bound; zero
  work is counted as real dispatches, including Hybrid scatter accounting. No root/COW/
  Ready/publication rule or default state policy changed. On the same matched
  opt1/codegen64/non-LTO8x48+3 public SDK cell, exact native B8 passes at184.409ms /
 130.145 output tok/s contiguous and205.724ms /116.661 paged; retained pre-change results
  were183.932/130.483 and242.488/98.974. Mean first-token commits in the new run are
 133.988ms contiguous and133.531ms paged. Strict CUDA `zerogravity-gpu` Clippy passes.
  A focused low-level `physical_paged_kv_` attempt was externally SIGKILLed while
  compiling before any test body; a later exact-source SDK confirmation was externally
  SIGKILLed after compilation and test entry with no Rust/CUDA diagnostic. Both left no
  process/GPU residue and remain incomplete infrastructure evidence. The terminal-green
  measurement above predates only the dispatch-count accounting correction, not the
  compute/zero path. **The next paged transport slice is now physically accepted.**
  A source-neutral `RuntimePagedPairGather` uses the existing32-pointer pinned-kernel
  ceiling as15 key/value page pairs plus two destination planes. Page count, active words
  and destination offset are runtime scalars; longer prefixes chunk through the same ABI,
  so no page-count-specific kernel family or pointer-table allocation is introduced.
  Existing page/root/COW/access/completion authority is unchanged. With page_tokens4 the
 48-token workload has12 pages per plane and replaces24 page-copy launches per KV domain
  with one paired gather launch. The same matched8x48+3 SDK cell is terminal-green,
  exact/native B8: contiguous176.800ms/135.746 output tok/s and paged182.505ms/131.504,
  with mean commit clocks[129.424,153.479,176.519]ms versus
  [129.212,155.426,181.469]ms. That slice left ~3.2% matched wall-time overhead.
  **The final staging-zero launch gap is now removed on the bounded common path.** The
  paired gather copies its active prefix and zero-fills the remaining destination span in
  the same final launch whenever the staging plane fits the existing runtime-word bound.
  Oversized planes keep the previous bounded device-zero fallback, and empty prefixes
  explicitly keep it as well, preserving inactive-tail initialization without a larger
  ABI or new allocation. The same matched8x48+3 SDK cell is terminal-green at
  contiguous177.467ms/135.236 output tok/s and paged180.520ms/132.949, with commit clocks
  [129.797,153.805,177.178]ms versus[128.897,154.402,179.491]ms. Remaining matched
  wall-time overhead is ~1.7%, versus ~11.6% after device-zeroing and ~31.8% at the
  retained pre-paged-optimization baseline. The first physical attempt was externally
  SIGKILLed during compilation before test entry; process/GPU/kernel checks were clean and
  one cached retry passed. The connector still has no vLLM executable/module and Podman
  still reports the pinned image `image not known`; do not claim a fresh vLLM comparison.
  **Current continuous-refill goodput is now remeasured on unchanged `e72030c`
  source.** The same public-SDK4 one-token +4 eight-token survivor +4 queued eight-token
  shape (`active=8` / `cohort_width=8`,68 generated tokens) passes under the matched
  opt1/codegen64/non-LTO profile at400.383ms/169.837 output tok/s contiguous and
  422.572ms/160.919 paged, about5.5% paged wall-time overhead. This materially improves
  the older paged498.212ms/136.488 result while avoiding a new retained fixture; the
  temporary probe was removed. I also tested the obvious next shared scatter candidate:
  fuse device-to-device append copy with inactive-tail zeroing so fresh pages do not need
  a separate zero launch. The first post-change run was externally SIGKILLed after test
  entry with no residue/OOM/Xid; one cached retry completed at395.849ms/171.783
  contiguous and421.159ms/161.459 paged. That is only ~0.3% paged throughput gain versus
  accepted source, so the extra kernel path was rejected from production selection. The
  implementation is now retained as the model-neutral dormant CUDA candidate
  `cuda::performance_candidates::launch_copy_zero_tail`; source behavior stays `e72030c`. The connector still
  has no vLLM executable/module and Podman still reports the pinned image `image not
  known`; do not claim a fresh vLLM comparison. Owner now explicitly defers vLLM.
  Do not add refill fixtures, direct paged-attention complexity or debug-profile
  micro-tuning as the next step; advance the populated CUDA context ladder instead.
  **Post-check L5 baseline watch:**
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
- Pushed/tracked `origin/main` is `f0f06402c78d4511dea1559c73d4295f88790f93`
  (`feat(cuda): batch multi-token prefill attention`). The post-origin local lineage
  includes `1222f8a` (multi-token GDN prefill), `60ba956` (matched performance
  profile), `3afb990` (reduced-band GDN layouts/refill), docs-only `7beb1be`,
  `0dac5ab` (device-side paged zeroing), `0680b0b` (batched paired-page gather), and the
  current fused-tail-zero slice described above. Revalidate actual HEAD for the newest
  signed source checkpoint.
  Nothing after `f0f0640` is pushed unless explicitly requested. Never use push merely
  for X2 transport.
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
  command namespaces and l5 RTX3060 is idle at ~37 MiB. X2 is a **shared GPU host**
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
5. `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md`.
6. `.agents/plans/TODO.md`.
7. `.agents/plans/model-execution/TODO.md`.
8. `.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md`.
9. The owning plan(s), especially:
   - `serving-and-speculation/TODO.md`
   - `cpu-gpu-optimization/TODO.md`
   - `memory-and-streaming/TODO.md`
   - `pipeline-composition/TODO.md`
   - `numerics-and-context/TODO.md`
   - `cli-device-routing/TODO.md`
   - `model-acquisition/TODO.md`
10. Relevant architecture docs under `.agents/docs/`.
11. Git history for removed experiments and superseded handoffs.

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
- Physical batch width is separate from each text invocation's authenticated numerical
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
  genuine context progression, reference capture, CUDA-forward/HIP parity and
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
- **20 Nov current-source 27B HTTP recheck is environment-gated before HIP allocation.**
  X2 was synced to exact L5 `dfc3e39` bytes by Git bundle and had a clean passive
  baseline of274,448,384B VRAM /0% busy, so the explicit75GiB envelope fit. A temporary
  reference-free active1/context1K/output3/47-token HTTP probe compiled successfully
  with the already-accepted optimized test profile (opt3/codegen16/non-LTO/debug0);
  release thin-LTO compilation was again externally SIGKILLed and was not looped.
  The accepted child-net wrapper
  `bwrap --unshare-net --bind / / --dev-bind /dev /dev --` plus guarded loopback
  setup was recognized by `native-gpu.py --require-private-network` as
  `bubblewrap-child-network-namespace`. The runtime then failed cleanly before HIP
  allocation because the installed ROCm runtime closure cannot be loaded in the current
  locked shell: both installed `clr-7.2.4` roots pull `libLLVM.so.22.0` requiring
  `GLIBC_2.44`, while the shell provides glibc2.42. Exporting the exact HIP/ROCM root
  therefore does not repair the ABI mismatch. No HTTP/model result follows, no GPU delta
  occurred, and X2 returned exactly to274,448,384B/0%. The temporary probe and bundle
  were removed and X2 restored to its original `b599db3` main checkout. Do not spend
  model-execution turns reconstructing this host ROCm closure; repair the X2 environment
  separately before further27B/ROCm2B physical work. vLLM work is explicitly deferred by
  owner for now; continue L5-only non-vLLM implementation/correctness work next.
- **21 Nov dynamic mixed-prefill grouping was tested and deliberately rejected.**
  Current CUDA source already admits reduction-compatible ragged prompt chunks into one
  native cohort:48/49 share the authenticated source-derived projection reduction class,
  projections/GDN execute jointly, and logical scheduler order remains first-seen/fair.
  The remaining attention path batches only consecutive equal-row quartets. A temporary
  candidate kept scheduler order untouched, stably clustered at most8 private physical
  lanes by row count, temporarily permuted session refs only for physical publication,
  restored logical session/readback order, and allowed the existing B4 attention kernel
  to consume each equal-row quartet with per-lane fallback for leftovers. There was no
  new GPU allocation/kernel or changed root/state authority. A temporary public-SDK cell
  used singleton48/49 references and an interleaved native-B8
  [49,48,49,48,49,48,49,48] wave with3 outputs; every row remained singleton-exact and
  reported native width8. Under the matched opt1/codegen64/non-LTO profile, unchanged
  source measured299.985ms after warmup; the candidate measured324.231ms and313.585ms
  on two runs, about8.1% and4.5% slower. It remains rejected for this Qwen .8B selector,
  but its reusable allocation-free lane clustering/logical↔physical mapping/equal-group
  implementation is retained in `performance_candidates::LaneRegroupPlan`; only the
  temporary Qwen-specific wiring/probe was removed. Re-evaluate the retained candidate
  on other model/shape cells rather than reconstructing it. Next L5 implementation/
  qualification target was the genuine populated-context ladder rather than another
  batching fixture. **21 Nov current public CUDA serving now passes4K,8K and16K for both
  0.8B and2B.** The retained env-parameterized public-SDK qualifier loads only the
  approved small snapshots, uses active1/native_rows0 and the declared context capacity,
  submits an exact context-2 token prompt, observes two GPU-rooted output tokens, then
  repeats the request on a fresh session.4K/8K were already green; the16K slice now also
  passes0.8B and2B with exact replay, distinct session IDs, exact producer frontiers and
  clean shutdown. The first0.8B16K attempt was externally SIGKILLed during Rust compile,
  before the test binary ran; residue/GPU/kernel checks were clean and one cached retry
  passed in~62s.2B16K then passed in~76s. The qualifier remains retained in
  `fabric_sdk.rs`; no new fixture or token pin was added. This proves current public
  serving/chunk continuation through16K; it does **not** claim independent
  all42/full-logit numerical closure because the populated reference archive is absent
  on L5. **The old2B32K mandatory-refusal statement is now superseded.** It came from
  the28Oct fully-resident-weight calculation (~8.98MiB over4608MiB), but current shared
  source admission has supported streamed-weight fallback since `d16f096` on8Nov.
  A21Nov public2B32K `QwenCuda` load returned `Ok(())`, so current code must not be
  forced back to that historical refusal. The exact current load admission is now
  inspectable through the retained host-only `ResidentModel::ordinary_cuda_source_preflight`
  primitive, which shares the loader's actual fit helper rather than duplicating an
  estimator. With active1/native_rows0/provider4608MiB at32K,0.8B remains fully resident:
  provider=4,831,838,208B, fit=4,831,826,472B, source weights=1,503,789,056B resident,
  roots=4x422,854,656B, runtime reserve=11,736B, model admission=3,427,673,077B,
  total=3,427,684,813B.2B deliberately streams its3,762,552,832B source weights:
  resident weights=0, roots=4x422,854,656B, runtime reserve=11,736B,
  model admission=1,925,098,485B and total=1,925,110,221B. Both are far below the
  configured provider cap; the old2B refusal was purely a pre-streaming calculation.
  32K admission is closed; execution is now **partially closed**. Historical0.8B32K
  attempts were externally SIGKILLed, and2B32K loaded successfully but remained
  execution-open after a~230s external SIGKILL. The external termination also reproduced
  on compile-only current-source builds, proving it was not unique to inference. After
  the owner requested two additional trace runs, the exact already-built optimized
  0.8B32K public-SDK binary was executed directly under `native-gpu.py`, with Cargo
  absent from the process tree. **Trace run1 passed end-to-end in185.18s**:
  `P12_CUDA_POPULATED_CONTEXT_PASS model=08 context=32768 prompt_tokens=32766
  output_tokens=2 replay=exact independent_sessions=true producer_frontiers=exact
  final_shutdown=true independent_all42_reference=false`. This is current positive
  capability evidence for0.8B32K serving/chunk continuation. The **identical trace run2**
  then received explicit `SIGKILL from outside` at~80.7s after crossing the60s execution
  mark. Post-run state remained clean: GPU37MiB/0%, no process/kernel OOM/Xid,
  service-cgroup peak6,862,434,304B, memory.max/high unlimited, oom/oom_kill=0,
  pids-limit events=0, CPU quota unlimited, and no visible oomd/journal kill record.
  Therefore separate model capability from infrastructure reliability:0.8B32K has one
  full current-path pass; repeated execution is still host/external-SIGKILL flaky.
  The retained **2B32K** qualifier is now localized beyond the old external-SIGKILL
  ambiguity. With the matched opt1/codegen64/non-LTO profile, current execution builds
  cleanly and remains alive normally until the SDK's existing600s first-response
  deadline, then fails as `populated CUDA context response deadline: Timeout`
  (representative total615.82s). Multiple follow-up runs likewise remained healthy from
  the harness/process perspective and ended at~616–617s, not SIGKILL. This is a genuine
  long-context progress/performance boundary, not evidence of VRAM refusal, OOM or
  numerical mismatch. Simple placement hypotheses were measured and rejected from
  production: keeping~2.9GiB of3.76GiB source weights resident, prioritizing packs used
  by `state_nodes()`, and increasing the serial source page through16MiB did not
  materially move the deadline. A32MiB page experiment hit the then-current verified
  source-workspace ticket bound before inference. The underlying host-accounting bug is
  now fixed generically: the serial source path retains two independent page-sized host
  owners (`Bytes` payload + backend/bounce headroom), so it holds two separately bounded
  `WindowWorkspace` tickets instead of one artificial `2*n` ticket. The existing
  per-ticket32MiB source bound is unchanged; no bound was weakened. Partial-residency and
  sized-page production selection from the old experiment remain removed.
  One source-only admission cleanup is now accepted before further profiling: ordinary
  CUDA attention carried a capacity-sized `workspace_reserve` even though the retained
  tile-local attention kernel consumes no global scratch. Removing that dead lease from
  layout admission, cold allocation and access ownership cuts32K ordinary-CUDA workspace
  from151,012,757B to12,584,241B (−138,428,516B). The low-level two-root2B32K inventory
  therefore fits fully resident at4,702,464,913B under the4608MiB provider envelope.
  Public active1 resident-text execution still deliberately uses four logical root slots; do **not**
  reduce the `2*max_sessions+2` lifecycle pool merely to force residency—the extra
  ownership covers public Pending/snapshot/displaced-root lifetimes. The resolution is
  instead the existing paged-state policy: immutable committed prefix pages are shared,
  so admission now prices one full committed page set per active session plus only the
  physically reachable COW/appended pages for the remaining logical root slots. This
  preserves all four logical roots while avoiding four full contiguous32K KV slabs.
  Host qualification proves the paged2B32K four-root plan keeps all3,762,552,832B source
  weights resident. A matched public-SDK A/B at the **same2B/32K capacity and1024-token
  prompt** moved from283.22s contiguous/streamed to28.56s paged/resident (~9.9x faster),
  both exact with clean shutdown. The retained full32766-token paged qualifier then passed
  end-to-end in224.80s test execution /260.34s wall, inside the existing600s first-response
  contract, with exact replay, independent sessions, exact producer frontiers and clean
  shutdown. **Model-start smart configuration now owns the default choice.** The shared
  `smart_config` layer is model-neutral and resolves independent typed knobs once at cold
  model start; adapters can add/remove knobs without changing the core precedence/report
  contract. Modes are exactly: `Exclusive` = Auto owns supported knobs and user values for
  those knobs are ignored; `Enabled` (default) = user-provided values win per knob and Auto
  resolves the rest; `Disabled` = no Auto, use user values or static model defaults. Each
  Auto evaluation may consume a fresh read-only runtime/device observation, but ordinary
  resource admission remains authoritative and no decision enters the token hot path.
  Resolved choices emit one compact `smart-config` startup line with mode, observation time,
  source (`auto`/`user`/`default`) and rationale. Qwen CUDA is the first adapter and now
  resolves **two independent knobs jointly**: `state_policy` and `native_rows`. Candidate
  startup configs are tiny (at most `{Contiguous,Paged64/64} × {8,4,0}` filtered by active
  capacity/user constraints) and use the exact resident-text preflight against current CUDA
  ordinal0 free VRAM. Selection priority is deliberately simple: preserve resident weights
  first, then widest qualified native cohort, then Contiguous as the lower-overhead tie.
  This prevents smart batching from accidentally reintroducing weight streaming. Canonical
  `QwenCuda` uses `Enabled`; `QwenCudaWithStatePolicy` supplies only that user knob while
  native width remains Auto; `QwenCudaWithSmartConfig` exposes the global mode plus
  extensible `QwenCudaSmartOverrides`. Legacy nonzero `profile.native_rows` is folded into
  the override map; zero means unspecified under smart CUDA. Explicit `Some(0)` forces
  sequential execution without disabling Auto for other knobs.
  Physical default-Qwen acceptance on2B/32K/1024 logged Auto selecting paged because
  contiguous streamed weights while paged stayed resident, then passed in26.51s with exact
  replay/frontiers and clean shutdown. The existing `active=8` native SDK cell now leaves
  `native_rows=0` unspecified: Auto selected **8** for both halves while the state knob was
  Auto→Contiguous on one half and user→Paged4/64 on the other. Actual publication remained
  `{8}` with exact tokens/clean shutdown; matched opt1 measured176.498ms/135.979 tok/s
  contiguous and180.446ms/133.004 tok/s paged. No new benchmark fixture was added.
  **Authoritative live CUDA admission now follows the
  advisory smart decision.** `new_pinned_cuda` cleans reclaimable cold allocator/cache
  blocks, queries device-wide free VRAM from the original CUDA actor/context, preserves the
  configured `device_cap-provider_cap` headroom, and clamps the provider envelope to
  `min(configured_provider, free-headroom)`. If the cap falls, the loader reruns the same
  exact admission/placement helper under that lower cap before source binding/allocation;
  the resident ledger then sub-admits inside that clamped envelope. This is intentionally
  distinct from the read-only smart observation: a race or changed environment cannot
  overrule final resource admission. Pure arithmetic tests cover full-cap, reduced-cap,
  headroom exhaustion and invalid envelopes. A real default2B/32K/1024 run through this
  actor-side clamp passed in27.58s exact with clean shutdown on an idle card. The qualifier
  reuses the existing test; no duplicate fixture or token pin was added.
- **Retained-candidate policy:** reusable performance experiments are no longer deleted
  merely because one model/shape loses an A/B. Model-neutral candidates live dormant,
  with no production selector/callsite, so Qwen2B/27B, Gemma4 and future models can
  measure them independently. A negative result remains authoritative only for its
  measured cell.
- **21 Nov deleted-candidate restoration sweep is complete for the recoverable kernel
  mechanics.** Existing retained primitives such as ragged B4/B8 attention, packed
  private-KV transition, batched rotary, grouped state transport and stateless fused
  math were not duplicated. The dormant inventory now includes
  `performance_candidates::LaneRegroupPlan`,
  `cuda::performance_candidates::launch_copy_zero_tail`, thirteen
  `Bf16GemmCandidateKind` strategies, two recurrent-transpose directions and cached-GDN
  occupancy candidates for min-block hints8/16/24. The GEMM set covers the prior
  inner-major LaneBlocked B2/B4/B8, LaneStrided CTA64/CTA256, RoundedSegments M32/K64,
  paired-BF16 unpack, Ordered M128/N16, row-major shared-input/paired-column/split4 and
  inner-major warp-shared/paired-column/unroll8 strategies. Git history plus1,433
  unreachable blobs contained the historical measurements/docs but no surviving source
  blobs for those uncommitted A/B kernels, so they are clean reconstructions over the
  current accepted arithmetic source rather than falsely claimed byte recovery. The
  shared-input candidate is explicitly named `...SharedInputF32` because the old notes
  establish shared staging+widening but not the vanished literal implementation.
  Production selection is unchanged and there are zero callsites outside the dormant
  candidate modules.
  Physical compile-only SM86 checks passed first for the initial five GEMM candidates,
  then for7 GEMM +2 recurrent +3 GDN occupancy candidates, and finally for the expanded
  candidate translation unit after the remaining GEMM strategies were added. These are
  code-readiness results only, not numerical/performance acceptance for a new model.
- **GPU test/cfg drift exposed by the restoration sweep was repaired without changing
  runtime behavior.** `ResidentPlaneLayout` ownership is no longer hidden behind an
  unrelated feature gate; graph witness/test-support/quarantine observers are available
  under the test cfgs that consume them; old cached-GDN manifests explicitly reject the
  newer `TextPrefillCohort` phase; and stale GDN runtime tests pass the required lane
  count. Default, plain-`qualification` and CUDA-native `cargo check --tests` now
  type-check. The larger `cuda-native-qualification` test/library checks were repeatedly
  externally SIGKILLed immediately after entering the GPU crate with no Rust diagnostic;
  do not claim that matrix green or infer a source failure from those kills.
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
- 4-row forward-source cohorts shared stateless projection/head while preserving
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

## Current CUDA forward selected path

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
  `/nix/store/nhr5mz136csd28mn8h5gdrbq6cbmbg0m-cuda13.4-cuda_cudart-13.4.92/include`
- CCCL:
  `/nix/store/ramrzw37i39f9fbnj6agp8kym2g056y7-cuda13.4-cccl-13.3.4.3.1/include`
- NVRTC:
  `/nix/store/njh4s557p8z871vpqwcy7i23k6swv23m-cuda13.4-cuda_nvrtc-13.4.92-lib/lib`
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

> Use `dotlink---l5` as the only source-authoring connector. Continue in
> `/home/pvl/spaces/abird/src/zerogravity` on the single persistent `main`
> worktree. Use `dotlink---x2` only for AMD large-VRAM execution/testing;
> synchronize exact L5 commits by Git bundle rather than pushing merely for
> transport. Do not create another persistent worktree or branch. Preserve all
> unrelated dirty/staged work.
>
> Keep each implementation turn to roughly 30 minutes. Aim for the next coherent
> implementation checkpoint, update durable owning docs/plans, sign the commit,
> then pause. Do not push unless explicitly requested.
>
> Read first, in order:
> 1. `AGENTS.md`
> 2. `.agents/plans/README.md`
> 3. `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-21.md`
> 4. `.agents/docs/CURRENT_HANDOFF.md`
> 5. `.agents/plans/TODO.md`
> 6. `.agents/plans/model-execution/TODO.md`
> 7. `.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md`
> 8. `.agents/plans/serving-and-speculation/TODO.md`
> 9. `.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md`
> 10. `.agents/plans/memory-and-streaming/TODO.md`
> 11. Any owning docs/plans linked from those files.
>
> Revalidate before editing: `git status --short --branch`, `git rev-parse HEAD`,
> `git rev-parse origin/main`, remote GitHub `main` when network transport is
> available, `git log -1 --show-signature`, worktrees, active
> Cargo/rustc/model/native-gpu/Bubblewrap processes, physical GPU state,
> source/cache identities, disk space and `_tmp` artifacts.
>
> Latest signed implementation/source checkpoint at handoff creation is
> `6c68a361e843c73107a86aaedea5ca697a30258c`
> — `refactor(cuda): use ledger identity for prepared leases`.
> The signed handoff-consolidation docs commit(s) may sit above it; revalidate
> actual `HEAD` first. Pushed/tracked `origin/main` is
> `f86538345fe4ce412df81864542c5ec183446460`
> — `docs: record 32k trace pass and external kill`.
> The implementation/source lineage was 12 commits above origin before handoff
> consolidation. Do not push unless explicitly requested.
>
> For every Cargo invocation use `CARGO_BUILD_JOBS=8` and
> `RUSTC_WRAPPER=/run/current-system/sw/bin/kache`. Run physical GPU work only
> through `python3 -B scripts/native-gpu.py -- ...`. Commands genuinely requiring
> TCP/UDP loopback must use the accepted private-network envelope and
> `--require-private-network`; never weaken harness isolation.
>
> Preserve the current production decisions:
> - ordinary public CUDA graph serving remains Native; graph serving is disabled;
> - exact StateRoot publication, cancellation/unknown-completion authority,
>   ResourceArbiter ownership and physical lifetime semantics are non-negotiable;
> - canonical `QwenCuda` now uses the generic smart startup layer in Enabled mode;
>   user values override Auto per knob, Exclusive ignores supported user knob
>   values, Disabled uses user/static defaults only;
> - current Qwen smart knobs are `state_policy` and `native_rows`; exact startup
>   candidate scoring is resident weights first, then wider qualified native
>   cohort, then Contiguous as the lower-overhead tie;
> - smart config is advisory and load-time only; authoritative CUDA admission
>   re-observes live free VRAM from the original actor, preserves configured
>   device headroom and reruns exact admission under any lower live cap;
> - explicit paged 2B/32K with page64/append64 passes the full 32766-token public
>   SDK cell in 224.80s execution / 260.34s wall; same-capacity 1024-token A/B is
>   283.22s contiguous-streamed vs 28.56s paged-resident (~9.9x);
> - canonical default 2B/32K/1024 Auto-selects paged and passes in ~26–28s;
> - `active=8` SDK with native width unspecified Auto-selects B8 and physically
>   publishes `{8}`; matched opt1 is 135.979 tok/s Contiguous and 133.004 tok/s
>   user-Paged with exact tokens/clean shutdown;
> - 0.8B/32K has a complete retained current-path pass; independent populated
>   all42 numerical closure at larger contexts remains open because the approved
>   reference archive is absent;
> - exact partial-tail paged COW ownership is already closed; do not build another
>   COW abstraction;
> - canonical `DeviceBufferIdentity` is the physical allocation/range authority;
>   CUDA-forward prepared resident sets and source-page/native-embedding alias
>   safety already use ledger identity. Handle ranges remain execution binding
>   geometry, not ownership truth.
>
> Current reusable performance work must not be lost: dormant lane-regroup,
> copy+tail-zero, 13 BF16 GEMM strategies, recurrent transpose and GDN occupancy
> variants are retained without production consumers. Do not delete a clean
> reusable primitive merely because its first consumer lost an A/B.
>
> Fresh matched vLLM is still deferred by environment, not policy: the pinned
> `vllm-0.31.0-cu129` store exists, but this connector cannot enter the rootless
> Podman user namespace because `/etc/subuid`/`subgid` are unavailable. Do not
> rebuild/re-register/mutate the baseline merely for feedback. The retained
> eager-vLLM 72.20 tok/s figure is historical only.
>
> X2 ROCm physical work remains environment-gated; preserve other GPU owners and
> never reset/reclaim/restart/kill their workloads. L5 owns source changes.
>
> Highest-value next work should come from the live TODOs, not stale prose.
> Current likely lanes are: graph/transfer physical-identity adoption only where
> a real consumer exists, authenticated transfer rings/broader streaming tiers,
> independent larger-context numerical references when approved inputs become
> available, and measured smart-config knobs only when existing evidence supports
> a clean startup rule. Avoid speculative fixture matrices, debug-profile
> micro-tuning and duplicate paging/COW mechanisms.
>
> Preserve the project working style: implementation and quick real-model
> feedback first; reuse accepted evidence for unchanged primitives; keep
> model-neutral primitives clean/composable; favor zero-cost Rust abstractions
> where natural; optimize measured hot paths aggressively when the fast design
> stays simple; otherwise prefer the simpler maintainable design.
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
retained charges after external drops. At that historical pre-streaming checkpoint,
normal lifecycle passed all13 admitted capacities (`.8B` through32K,2B through16K),
each at48 active tokens;2B32K refused at pure preflight. This refusal is superseded by
the21Nov current-source streamed-admission note above. Long-active-context/2B numerical
parity remains open. Final
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

Historically, before streamed-weight admission became the current shared-source policy,
32K exceeded 4,831,838,208 B by 8,980,841 B and was expected to refuse before physical
allocation. The21Nov public2B32K load supersedes that refusal as current authority.

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