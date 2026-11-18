# Plan queue

Current source, tests, and this queue are authoritative; Git retains retired history.
Cross-window entrypoint: [CURRENT_HANDOFF.md](../docs/CURRENT_HANDOFF.md).
Full takeover context: [FULL_COMPREHENSIVE_HANDOFF_2026-11-14.md](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-14.md).

Latest implementation checkpoint is signed `9d6613c`
(`feat(cuda): select compatible resident graph cache`); signed qualification checkpoint
`599cb28` adds retained B4/B8 production-selector timing without changing ordinary
serving. Above signed `63ee14e`, the workspace-owned one-shot graph cache selects only
an exact compatible live graph; cold/spent/absent/incompatible/device-token cases are
pre-enqueue Native misses, while runtime identity/lifecycle drift or replay failure is
an error with no unsafe retry. Same-process AB/BA full-logit timing (4 warm pairs,
12 samples/path) now covers the accepted0.8B B2/B4/B8 graph bands: B2 Native273,933us
mean /267,847us median vs Graph264,881/263,238 (~3.30%/~1.72%); B4
273,801/273,910 vs270,766/270,805 (~1.11%/~1.13%); B8 362,559/360,951
vs359,915/359,649 (~0.73%/~0.36%). Graph policy charges are15,356,224 /
15,405,376 /15,458,624 bytes respectively. All supervised runs retain exact replay
counts, trusted retirement and final reserved0/41MiB. The gain shrinks at useful
widths and does not justify the extra serving/resource lifecycle, so public serving
deliberately remains `FusedExecution::Native`; retain the private selector/cache for
future materially different graph regions. Earlier interrupted invocations remain
incomplete, not passes. The older15,339,840-byte transient remains unexplained/
watch-only. Prefix scheduling follow-up is now complete as a bounded reconstructed
ZG-vs-ZG cold diagnostic for both original models. 0.8B old max32 (`d0a6084`)
versus native-prefix (`f788929`) improves mean wave 17.4307s->15.4529s (-11.35%),
TTFT p95 15.0368s->14.9527s and delivery-ITL p95 0.8188s->0.7804s. 2B completes
the same fresh-process ABBA contract with exact outputs/usage/finish and zero
failed/refused/cancelled/discarded work: mean wave 29.5280s->25.0937s (-15.02%),
reported output rate 2.1693->2.5512tok/s (+17.60%) and delivery-ITL p95
1.5598s->1.3234s (-15.15%), while TTFT p95 regresses 23.1625s->23.8931s
(+3.15%). Earlier rebuilt-old 2B first-work disappearances remain incomplete
preconditioning attempts; staged C1/B2/B4/B8 diagnostics and the exact ragged
scope subsequently pass with zero OOM events and final 41MiB. Both reconstructions
use symmetric benchmark-only exact-SM cubin behavior matching `6a5c610` and
retained source-real prompts because the original frozen binaries/workload are
gone. Do not promote these to byte-identical historical, source-correct-goodput
or vLLM claims. Pushed `origin/main` is `3a1596d`; current paged source
qualification work remains local until explicitly pushed.

The P12-C fair-interleaving sub-slice is now accepted: shared engine fair rounds
protect all eligible decode work, defer young prefill with original owners intact,
and force oldest prefill progress after two deferred rounds under a bounded
logical-token budget. Focused engine tests/clippy and production CUDA composition
pass; physical staggered-arrival HTTP passes both approved 0.8B/2B models with
exact IDs/text/usage, first shared B2 no earlier than A output index 4, zero
failed/refused/cancelled/discarded work and final 41MiB. Every command in
`scripts/check.sh` is terminal-green on the exact source under isolated
`native-gpu`; repeated monolithic connector-wrapper attempts disappeared during
ordinary cached builds with no compiler/test error, live child, OOM event or GPU
residue and remain explicitly incomplete. Multi-token temporal prefill has a
**functional physical 0.8B B2 acceptance cell** and production HTTP B2 remains
exact chunks4+4+2 with CPU ID264/text ` a`, usage10/1,3 publications, zero
failures/refusals and final41MiB. The retained qualifier is now
`temporal_prefill`, shared across B2/B4/B8. Original0.8B and2B each have full
CPU/B1/all42 temporal B2/B4/B8 acceptance with exact rollback/publication,
greedy/producer and final reserved0. Signed `eaec62d` promoted B4 and signed
`8e6f14d` now promotes **B8**. Final production HTTP passes original0.8B and2B
for both B4 and B8 as exact chunks4+4+2 with three matching native cohorts/
publications, mixed JSON/SSE CPU text/usage exact and clean retirement; retained
2B B2 HTTP also passes. B2+B4+B8 therefore all select temporal4 with
equal-frontier prefill. Earlier disappeared B4/B8 attempts remain incomplete,
not relabeled. Debug-profile evidence is real CUDA but not release packaging/
performance. Canonical release B2 remains open: a fresh heartbeat-supervised
thin-LTO attempt again disappeared during CLI compile/link before test execution,
with no terminal child/Rust/OOM/Xid/GPU-residue evidence. Do not loop it.
Signed `0027c28`, `54340a6` and `1eb6e0f` provide the immutable block-table
planner, finite resident page owner and actual COW materializer. Signed
`3ac2059` adds the first **typed paged-KV representation + bounded gather**
without selecting it in sessions. `ResidentPagedKvDomain` accepts only checked
`StateLayout::Kv` requirements, gives K/V distinct source/profile/domain/schema
compatibility identities, pairs their append transactions, and gathers the exact
committed frontier into fresh contiguous K/V staging under the original
`ResidentSubmission`. Final-byte RTX3060 proof uses BF16 row512/page4 with a
strict 4-page cap: frontiers3/4/5 gather byte-exact with dispatches2/2/4; 3->4
does two COW copies; 4->5 shares the full first pages and reuses retired backing;
prior snapshots are immutable; inactive tails zero; capacity refusal and final
retirement pass. Lower COW materialization and unknown-completion regression
cells remain green; production `qwen35-cuda` composition and strict clippy pass.
The first boundary run exposed post-Ready validation incorrectly reusing the
pre-submit uninitialized-target check; those lifecycle checks are now separated.
The next run proved data/gather correctness but showed only4 physical pages were
created because retired pages were reused; the qualifier was tightened from a
loose6-page ceiling to the actual4-page peak and passed. During linking,
`target/debug/incremental` (about29GiB) exhausted the filesystem; only that
disposable Rust build cache was removed, taking free space from about3.9GiB to
24GiB. One later exact relink wrapper disappeared without a compiler diagnostic;
the bounded warmed retry linked and the final-current physical result passed.
Signed `dc6a188` adds the missing **GPU-only staging/scatter bridge** required
before changing session ownership: `ResidentPagedKvStaging` presents transient
contiguous K/V leases as the unchanged `StateBuffers::Kv` ABI, and paged append
can scatter only newly produced candidate rows directly from those device
staging leases while preserving page-tail COW from the immutable prior. One
original `ResidentSubmission` now covers paged Reads, prior/candidate staging
Writes and private page Writes. RTX3060 exact proof prior3->candidate4 passes
with gather2, contiguous prefix-copy2, scatter-COW2 and scatter-append2; after
input the entire bridge is GPU-only, parent bytes remain exact and all pages
retire. The retained 3/4/5 boundary, lower materialization and unknown-completion
cells pass on the same final bytes; production `qwen35-cuda` composition and
strict clippy remain green. The implementation metric was renamed from the
host-specific `append_uploads` to transport-neutral `append_transfers`.
Signed `53336b3` adds the **canonical paged KV Domain/root substrate** without
selecting it in production ordinary sessions. `Domain` can now own
`ResidentPagedKvDomain` plus its exact page lease/generation cohort;
`StateBuffers::PagedKv` is a marker only. Validation checks declared
`StateRequirement::Kv` geometry, source/service generation, exact page-owner
coverage and readiness. Legacy native batch/range/diagnostic consumers explicitly
reject the marker until staging is bound. The existing single-session publication
sequence is factored into `publish_domain_swap` and still owns the only
`publish_if_usable -> prepare_publish -> swap -> commit` path. RTX3060 exact
proof publishes roots0->3->4, retains2 initialized page owners at each committed
frontier, keeps the displaced frontier3 byte-exact after frontier4 publication and
retires all pages. All four earlier paged physical cells pass on the final bytes;
production `qwen35-cuda` composition/strict clippy pass. Two unsupervised/late
link wrappers disappeared without Rust/OOM/Xid diagnostics; supervised/warmed
bounded retries completed and the final-current binary passed.

Signed `ef9e4d4` adds the **ordinary in-place KV execution contract** required
by the admitted one-staging design. `StateBindingMode::KvInPlace` is KV-only:
prior/candidate must be the exact same two resident leases (ID, allocation
generation and extent), staging must still be fresh/uninitialized at ordinary
validation, and all other bindings retain the old distinct-owner rule.
`KvAppend` now shares one `prepare_kv_candidate` path: Distinct is unchanged;
KvInPlace skips candidate zeroing and prior-prefix copies because the committed
prefix is already gathered into staging on the same original flight. RTX3060
physical pass uses the real ordinary KV append kernel at base3/capacity8:
`P12_ORDINARY_KV_INPLACE_PASS ... preparation_launches=0
prefix_preserved=true new_row_exact=true inactive_tail_zero=true
unique_write_access=true all_staging_retired=true`. The canonical paged
publication/bridge/boundary/COW/quarantine cells all pass on the same final
binary; production composition and strict clippy pass.

Signed `93468b1` closes the next **hybrid root semantic/admission gates**.
Paged snapshot/read no longer depends on contiguous `StateBuffers` handles:
it reads K then V page leases in block-table order, validates typed page geometry,
trims to valid rows and checks zero inactive tails. The same retained physical
root cell proves frontier3 partial + frontier4 full snapshots and 4->3->4 restore
through the original `publish_domain_swap`. Ready coverage now derives actual
Write outputs against the current root: shared pages are omitted only if they are
the exact initialized full owner at the same plane/index. Physical3->4 requires2
COW Writes;4->5 has4 candidate page leases but only2 Ready Writes because2 full
pages are inherited. `HybridStateInventoryPlan` takes explicit page_tokens and
max_append_tokens and computes vector-root, staging and page-pool maxima under
the original roots*state_bytes cap; it includes all-KV-domain candidate page
pressure and rejects insufficient headroom/heterogeneous page geometry. Three
unit gates pass. This is an admission plan only: no physical hybrid pool is
installed and no production page-token value is selected.

Signed `cb57e1d` adds the **physical hybrid pool substrate** without selecting
it in serving. `PooledStateStorage` now distinguishes the existing Contiguous
pool from a Hybrid owner containing vector-root, staging-set and page pools.
`install_window_hybrid_state_pools` reserves each metadata workspace from the
original verified source and constructs the three accepted pools. Initial hybrid
roots are zeroed Vector domains + empty paged KV; generic uninitialized candidate
allocation fails closed because candidate paging requires the current root.
Production page compatibility now uses fixed numerical-contract tags plus a
checked versioned physical/slab/capacity identity, not arbitrary qualifier IDs.
Physical RTX3060 proof forces vector_sets4 + staging_sets1 + max_pages24
simultaneously resident at exactly old_root_budget_bytes67584, verifies empty
paged roots and pool reuse, then drops the pools to final resident_bytes0.
All retained paged/in-place physical cells and production composition/clippy pass.

Signed `b9b04aa` adds the **current-root-aware hybrid candidate execution
path** without selecting it in the source loader. `HybridCandidateBundle` owns
one Vector candidate checkout, one complete KV staging checkout and all private
paged append plans. One exact-identity deduplicated submission covers current
Vector/page Reads, Vector/staging Writes and fresh/COW page Writes. Gather runs
before the ordinary source body; KV binds as accepted `KvInPlace`; scatter runs
after the body; Ready promotes only Vector candidates + actual fresh/COW pages;
paged `Domain`s are created only after physical completion.
Fresh/COW output ordering now matches `canonical_write_outputs` deterministically.
RTX3060 proof with one Vector + two KV domains passes base3->4 in one flight:
`P12_HYBRID_CANDIDATE_PASS ... gather_dispatches=4 scatter_dispatches=8
ready_outputs=5 canonical_ready_order=true paged_frontier=4 vector_ready=true
continuation_exact=true inactive_tail_zero=true old_root_budget_bytes=135168
final_resident_bytes=0`. Two earlier qualifier failures were test-owner lifetime
issues (retained access clones pinned staging; retained canonical comparison
clones pinned output backing), not execution failures. Two broad link wrappers
also disappeared late with no Rust/OOM/Xid status; lib-only linking succeeded and
the final-current eight-cell physical suite passes. Production check/clippy pass.

Signed `1413f72` adds the **explicit source-call paged state policy without
changing public/default serving selection**. `CudaSourceStatePolicy` is exported
only on the CUDA source boundary; the existing source loader remains a
Contiguous wrapper and the policy-bearing loader installs hybrid pools only for
explicit `Paged { page_tokens, max_append_tokens }`. Paged mode currently
requires B1; native rows and zero bounds refuse. Inventory also rejects an append
bound above state capacity, and runtime calls larger than the admitted bound
return `Limit` before checkout. Production registration explicitly passes
Contiguous, and public `ModelLoadRequest` is unchanged.
The pre-IO policy-refusal unit passes. The original-HF differential uses the same
authenticated source, Contiguous vs Paged, full248320 logits + all42 exact at
page-1/page/page+1/2page+1, final shutdown. The real managed L5 cache now exposes
the approved0.8B/2B snapshots. Original0.8B page4/append4 physically passes
`P12_SOURCE_PAGED_B1_PASS ... frontiers=3,4,5,9 full_vocab=exact all42=exact
contiguous_reference=true final_shutdown=true`. Four hybrid unit gates, all eight
exact-current RTX3060 paged cells, top-level qwen35-hip, CUDA tests and strict
clippy pass.

Signed `398e446` adds the **source-real paged B1 lifecycle qualification
target** without runtime changes. The shared approved-source fixture now drives a
second ignored target proving prelaunch cancel, abandoned Ready, cancel-after-
Ready discard, page-boundary commits and exact all42 restore to both the saved
pre-boundary and boundary roots, followed by shutdown. It compiles and links.
Signed `3041457` fixes the lifecycle qualifier's restore contract: restore
publishes a fresh root/epoch at the saved frontier; it does not resurrect the old
RootBinding. Same source/session, root+1, epoch+1 and exact all42 are now required.
Cold original0.8B lifecycle is physically accepted by retained job
`9feb1394-1a3b-498b-8f92-fdd3e607c6b3` with exit0 in168.08s and no
preconditioning:
`P12_SOURCE_PAGED_LIFECYCLE_PASS page_tokens=4 contiguous_warmup=false
prelaunch_cancel=true abandoned_ready=true cancel_after_ready=true
restore_before_boundary=true restore_at_boundary=true all42=exact
final_shutdown=true`.
Signed `e6aac53` removes the temporary warmup/stage diagnostics and leaves the
qualifier cold-only, emitting `cold_load=true`.
Intermittent alternative runs ending as deployed dotlink v0.17.3
Completed+exit_code=null+success=false are connector-infrastructure incomplete,
not model failures. On `release/0.17`, dotlink stores
`ExitStatus::code(): Option<i32>`; None means its Bubblewrap child was signalled,
but v0.17.3 does not retain the signal number (main's richer
`SharedProcessExit::Signal` is not deployed). Timeout/cancel/failure paths have
distinct statuses and the Completed path disarms containment before recording.
Controlled child/shell SIGKILL probes return numeric137 instead; cgroup OOM/pids
counters and dotlink restart/poll-failure evidence remain zero. A direct
`native-gpu -> env -> test binary` reproduction with no inner supervisor still
showed null exit, so tmux/setsid is not required. External signal origin remains
unproven; only full terminal passes count.

Admission review changes the next implementation shape: ordinary admission
budgets a finite inventory of complete state roots. A paged committed root plus
full prior staging plus full candidate staging would be a third state-sized
envelope at full context and must not be smuggled past admission. The Ready and
snapshot semantics are accepted at `93468b1`; physical pools at `cb57e1d`;
current-root execution at `b9b04aa`; explicit low-level policy at `1413f72`;
source-real lifecycle target at `398e446`; qualifier correction at `3041457`;
cold-only cleanup at `e6aac53`.
Original2B differential + cold lifecycle page4/append4 are now **accepted**.
Signed `6b920a0` owns the repaired execution harness: existing Bubblewrap
PID/user sandboxes are reused; private Bubblewrap netns executes directly;
shared-network Bubblewrap uses fail-closed socket/io_uring seccomp; outside
Bubblewrap the original user+net `unshare` path remains. `--isolation-info`
and `ZG_NATIVE_GPU_ISOLATION` expose the selected mode. Loopback/TCP callers
must use `--require-private-network`; `scripts/check.sh` refuses the seccomp
fallback immediately and AGENTS uses that flag. 29/29 harness tests plus current
sandbox AF_UNIX/AF_INET/io_uring/private-net refusal and silent 15s physical
smokes pass. The full workspace gate still needs an L5 connector profile with a
private network namespace; the current shared-network connector correctly
refuses it. Course correction from dotlink v0.17.4 forensics: periodic output is
**not** proven necessary. Dotlink now logs every stop it initiates; the relevant
window contains only the explicitly requested `job_cancel`, while a silent
long sleep survived until cancellation. Earlier SIGKILL9s are therefore
surrounding harness/orchestration artifacts outside dotlink stop paths, not a
quiet-job policy. The accepted runs happened to use a heartbeat sidecar while
keeping the model foreground:
- `38c1f321...` differential exits0 in703.17s with full-vocab/all42 exact at
  frontiers3/4/5/9 and clean shutdown;
- `744f36fd...` cold lifecycle exits0 in438.18s with
  `cold_load=true`, prelaunch cancel, abandoned Ready, cancel-after-Ready,
  both exact restores/all42 and final shutdown.
**Next: add true CUDA in-flight cancellation qualification on the ordinary source
path.** After-Ready discard is already proven but does not satisfy this gate.
Heartbeat output is optional harness plumbing, not an acceptance requirement.
Use ordinary `native-gpu` isolation for model/GPU cells and
`--require-private-network` only for loopback-dependent gates. Then consider
high-level paged serving selection. Default CUDA/native/HIP stay contiguous.
Complete-domain prefix identity/intersection follows that.
Event/lease-backed physical overlap remains after those state gates.

X2 remains the approved shared AMD large-VRAM execution/test host. Its baseline must
be resampled per run: at this checkpoint it is total103,079,215,104 bytes, used about
272,351,232/busy0 with no KFD owners, whereas earlier runs observed unrelated
Ollama/model residents around31GB. Never infer ownership from a prior baseline or
reset/reclaim another owner. Sync X2 to exact current l5 `main` before new AMD
physical work without using push as transport. Tiny-HIP runtime health, exact
tokenizer/source binding/plan and canonical27B host runtime remain green; missing
approved populated/reference material and previously recorded ROCm numerical cells
remain separate blockers. `main` is the only persistent worktree per host. Detailed
CUDA/HIP primitive history, exact measurements, rejections and blockers remain in
[model-execution/TODO.md](model-execution/TODO.md); do not duplicate them into a
second evidence ledger.

## Owner priority — current continuation

**Fabric functional transition accepted:** source `8c43844`, CUDA repair
`8d761c4`; scoped model/GPU/SDK/HTTP/cache acceptance and final isolated gate pass.
Original unsupported/unavailable cells and Qwen expansion remain explicit.
Main only; no new or resumed subagents or recurrence restart.
Approved x2 ROCm vLLM acquisition is complete. The former12GiB available-host
guard was waived with swap for fresh scopes; device safeguards remain unchanged.
Explicit75GiB-total HIP admission is signed`d16f096`; historical defaults unchanged.
Complete substantial implementation batches before combined validation; full gate last.
Prioritize working paths + early matched-vLLM feedback; reuse checks/evidence,
add fixtures only for concrete gaps. Baselines need not await all models/batching.
On l5, `5bf440d` opened the production CUDA graph ownership substrate,
`db7be60` added prepaid opaque graph-resource admission, `6a5c610` closed the
driver-PTX compatibility blocker, `81b7c71` made charge lifetime explicit,
`f1ce532` binds replay to exact resident-flight storage/access authority,
`63ee14e` adds the private workspace-owned production lifecycle/cache, `9d6613c`
adds the fail-closed exact-compatible private selector, and `599cb28` extends matched
selector timing through B4/B8. The original0.8B graph bands retain their qualified
resource/lifetime contracts; one older transient post-capture byte-accounting mismatch
remains a watch item. Public serving remains Native by decision: B2/B4/B8 mean gains
are only ~3.30%/~1.11%/~0.73% and median gains ~1.72%/~1.13%/~0.36%, so the graph
route does not earn its extra retained-resource/lifecycle complexity on this body.
Owner direction: l5 owns repository/source changes; x2 is only for AMD
large-VRAM execution/testing. Tiny HIP health and the full canonical27B host
runtime source/tokenizer path are green. X2 is a **shared GPU host** with Ollama
and other model workloads: existing KFD owner `2406838` / ~31.1GB is baseline,
not a launch blocker and not safely attributable to ZeroGravity from sandboxed
`/proc`. Before our physical work record the current VRAM/KFD baseline, run only
within available capacity, and verify our incremental delta retires afterward.
Never reset/reclaim/restart/kill other owners. Preserve the unchanged180s contract
and missing-reference blockers; no X2 build/device campaign is live at this
checkpoint.

Fabric migration is complete; the unfinished Qwen campaign remains next.
Preserve signed native GDN`e2559b6`/pause`1c35172` and all accepted bounded
model/lifetime evidence. Recurrence`17aafa71` remains cancelled; no implicit
hardware restart. S0 → S1–S3 → F0–F4 → Q0/Q1/Q2 are accepted within their named
scopes. Return to the Qwen model/context/C1..C64/vLLM mission without silently
activating held routes.

## Active

- [ ] P1 serving-and-speculation — @pvl — l5 CUDA `.8B`/`2B`, x2 ROCm `.8B`/`2B`/`27B`: correctness/context fit, C1..C64, then measured vLLM superiority; finite27B checkpoint passed; populated0.8B/2B HIP SDK/HTTP accepted; final gate passes; next CUDA/general27B/native batching; no recurrence restart — [details](serving-and-speculation/TODO.md)
- [~] P1 numerics-and-context — @pvl — exact cached47→50, lifecycle/capacity and final gate pass; active-context/2B reference inputs missing — [details](numerics-and-context/TODO.md)
- [~] P1 memory-and-streaming — @pvl — exact two-root admission: `.8B` fits through 32K, `2B` through 16K, `2B`32K refuses before allocation — [details](memory-and-streaming/TODO.md)
- [~] P1 cpu-gpu-optimization — @pvl — ordinary kernel/transport checks pass; exact whole-model parity before measured tuning and the matched benchmark campaign — [details](cpu-gpu-optimization/TODO.md)
- [~] P1 pipeline-composition — @pvl — recursive primitives/pipelines/profiles; shared execution adapters active, reuse current adapters in Fabric; Qwen expansion deferred — [details](pipeline-composition/TODO.md)
- [~] P1 cli-device-routing — @pvl — unified run/config/source/device selection; preserve behavior through Fabric SDK/CLI migration — [details](cli-device-routing/TODO.md)
- [~] P2 model-execution — @pvl — l5 CUDA2B B8 currently selects exact independent reductions, same-frontier ragged/chunked cohorts, private B8 recurrent core `60b048d`, ragged private B8 KV `775dfa6` and CTA256 KV `399e43a`; `874e67d` retains the exact shared-input Ordered+LaneBlocked B8 projection pair and its serving consumer remains rejected after non-repeatable loaded-model A/B. `6a5c610` fixes nonpersistent NVRTC driver compatibility; `63ee14e` physically closes the private workspace-owned graph lifecycle/cache above `db7be60`/`f1ce532`, `9d6613c` adds its exact-compatible private selector, and `599cb28` closes B2/B4/B8 timing with public graph serving deliberately left Native. Large same-contract CUDA numerical acceptance remains blocked by missing approved reference material. Populated small HIP is `8ffb511`; resident27B C1..C64 is `d16f096`; `c282f20`/`134a571` repair HIP cross-feature boundaries; `b0cf0b2` removes duplicate Qwen snapshot hashing with an explicitly non-authoritative descriptor lease. X2 exact tokenizer/source binding and full canonical host runtime pass. X2 is shared with Ollama/model workloads: existing KFD/VRAM is baseline, so physical27B/ROCm2B work may proceed when the requested configuration fits current available capacity, while preserving other owners and verifying only ZeroGravity's incremental resource delta retires — [details](model-execution/TODO.md)
- [~] P2 model-acquisition — @pvl — [details](model-acquisition/TODO.md)

## Icebox

- [ ] multimodal — [details](multimodal/TODO.md)
- [ ] distributed-execution — [details](distributed-execution/TODO.md)
- [ ] platform-and-release — [details](platform-and-release/TODO.md)

## Closed

- [x] fabric-vnext-transition — @pvl — source `8c43844`; S/F and scoped Q0/Q1/Q2 accepted, deferred expansion unchanged — [details](fabric-vnext-transition/TODO.md)
- [x] accepted-baseline — `6d74f76` — 2026-10-18
- [x] handoff-cleanup — `bfe672c` — 2026-10-18
- [x] historical-records — `9239719` — 2026-10-18
