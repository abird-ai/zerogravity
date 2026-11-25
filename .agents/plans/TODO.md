# Plan queue

**Current delivery order:** [Full implementation plan](FULL_IMPLEMENTATION_PLAN.md) — foundations → CLI/TUI/Web → ROCm parity → CUDA parity → Vulkan → optional compiled providers. This supersedes older ordering in linked designs; detailed owner TODOs and source/quality/lifetime constraints remain authoritative.

The historical queue narratives and P-level labels below preserve owner scopes, not a competing delivery order. Work the earliest incomplete phase of the full plan first. Do not promote an icebox item or bypass an explicit hold by following a cross-reference.

Current source, tests, and this queue are authoritative; Git retains retired history.
Cross-window entrypoint: [CURRENT_HANDOFF.md](../docs/CURRENT_HANDOFF.md).
Full takeover context: [FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md](../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md).
The Phase 1 exit pack links the complete dated history, Phase 2.1
UX-1 entry, original design and plan/open-work/Git inventories. The previous
foundations pack remains historical; live source and owning TODOs are authoritative.

**25 Nov Phase 1 checkpoint:** `7b36a879`/`0342e103`/`fdfb467f`
share authenticated matrix-window and resident/streamed CUDA transport.
L5 Qwen 0.8B resident/streamed B1 full logits/all42, native B4 token/score/
all42, known-completed cancellation/retry and CUDA SDK C1 pass; X2 native
HIP V2 C1/C8 plus alias chain pass. Fresh contexts compare actual outputs
without increasing the cumulative compiler budget.
Full isolated gate: 2393 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
**25 Nov parameter follow-up:** 687d9a24/06df027f/43928332/f4e09428
share authenticated materialization and original initialization ownership;
exact source/descriptor authority retained without transient backing.
Preserve all storage/rounding/padding policies and direct resident access.
Original logical Write/shared payload supports padded views without
weakening the native matrix/page primitive. 30 artifact + 3 conversion
tests, real CUDA resident/streamed exact B1/B4, all42, known-completed
cancel/retry and host staging retirement, HIP native V2 C1/C8 replay/
retirement pass. No new performance or unknown-fault claim.
X2 HIP portable typed slice 1024/2048 matches existing CPU/PyTorch
references and original retirement (test-only typed refusal correction).
Full isolated gate: 2393 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed 3ca16abd/a53a3446: direct typed source-region operands and explicit
compiled selection, distinct from native capture; original plan/shape,
admission and completion preserved. Engine preparation 9/9, windows 30/30,
strict Clippy/combined SDK and actual CUDA C1/HIP V2 C1/C8 replay/retirement
pass. No external SDK execution or new captured/GGUF physical claim.
Full isolated gate: 2394 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed 8eb17bc2: authenticated Qwen HF source independent of task/arithmetic,
backend-free approved snapshot discovery and common HF/GGUF cold eligibility.
Real CPU 0.8B/2B cached/direct and HF-import full vocabulary/all42 comparisons,
CUDA HF C1/GGUF pipeline+unload and HIP V2 C1/C8 pass with original retirement.
Test-only capacity/host fixture corrections 00f57e7a/9374e0ca.
Full isolated gate: 2396 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed 9ae6dd1e/e06213a5: Linux CUDA shared pre-Client original physical
envelope, exact PCI adoption, post-bootstrap live clamp releases unused
credit. Real HF peer refusal/no new compute owner/retry, actual pinned live
fit, CUDA HF C1/contiguous-paged C8/GGUF pipeline+unload and HIP V2 C1/C8 pass.
Full isolated gate: 2396 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed cbd2c119/197c2e26: aggregate host backing, common visible cgroup-v2
observation and cold fit, HIP driver/local VRAM intersection. Actual context
reservation replaces overconservative lifetime HOST upper-as-growth check;
CUDA HF pressure/retry/C1/GGUF and HIP V2 C1/C8 + 32k refusal pass.
Fresh release real GGUF CPU direct run passes. Full isolated gate: 2394 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Signed a6548863: ordinary Linux HIP pre-Client original owner/adoption,
guarded exact-adapter Vulkan local fit, same physical-domain placement
without inter-domain transfer. Actual portable HIP/Vulkan two-owner healthy/
unknown and peer refusal/retry pass; real HF HIP V2 C1/C8 replay/retirement.
Full isolated gate: 2394 Rust test executions, zero failures, 101 explicit skips; release/installer/scripts pass.
Generic repeated Vulkan initializer,
non-Linux, general UMA backing and cross-domain transport remain open.
Signed 1d5c4ada: optional production committed GGUF CPU checkpoint v2,
actual immutable source/geometry/numerical identity, bounded SHA256 file IO,
original host charge retained by state readers, original SessionAuthority
restore. Real 0.8B/2B full-vocab exact continuation after reloading source;
corrupt/wrong-model/budget/stale refusal and last-reader retirement pass.
Diagnostic v1 bytes and actual bridge unchanged. Full isolated gate: 2395
executions, zero failures, 102 explicit skips; release/installer/scripts pass.
Signed 804d4882/d9b25205: optional native HIP V2 F32 contiguous committed
persistence, stable source semantic revision, exact prepared compatibility,
original snapshot/root pool/upload/completion and shared bounded IO.
Actual HF 0.8B same/fresh-process full-domain/full-vocab exact restore,
corruption/stale refusal, clean retirement and unknown upload retention/fence.
Full isolated gate: 2395 Rust test executions, zero failures, 102 explicit skips; release/installer/scripts pass.
Signed ebda04c4: bounded source-declared language preparation, exact
selected graph model IDs, explicit local/HF policy and original loaders;
router + admitted plan publish atomically, binding/pressure rollback.
Actual CPU 0.8B/2B mismatch/retry/chain, CUDA GGUF two-owner three-stage
chain/selective unload, native HIP HF alias chain/C1/C8 and retirement.
Full isolated gate: 2396 Rust test executions, zero failures, 104 explicit skips; release/installer/scripts pass.
Signed b25b5c02: common borrowed model/plan text task, controls,
bounded receiver polling/cancel, structured refusal and selected original
registration/owner/config observation. Actual CPU 0.8B/2B and framed
nonstream task, CUDA GGUF distinct-owner chain and HIP HF alias chain/C1/C8
final observations/retirement pass. Full isolated gate: 2396 Rust test executions, zero failures, 104 explicit skips; release/installer/scripts pass.
Signed 7987f70a: shared local/cache GGUF resolution and held source/
family/target/context checks, TextLoadPolicy and declared SDK/pipeline,
exact snapshot observation and original verified raw lease.
Actual cached CPU/CUDA SDK and pipeline/reload/retarget refusal and
direct selected digest; local CPU/CUDA chain and native HF HIP V2 C1/C8
pass. 2ec06026 preserves selector refusal diagnostics. Full isolated gate: 2396 Rust test executions, zero failures, 106 explicit skips; release/installer/scripts pass.
Signed 8e3b8993/33118f16: original program/selected region descriptions,
actual retained CUDA projection, cold same-context queue/resource and
dense identical-layout copy compatibility; original completion/retention.
Actual CUDA resident/streamed B1/C4 exact, GGUF chain, HIP V2 C1/C8 and
private bridge refusal/interrupt pass. Full isolated gate: 2396 Rust test executions, zero failures, 106 explicit skips; release/installer/scripts pass.
Signed 3b5a520d/fc38db03/e1decff5: original named Vulkan service
lookup/atomic factory/exact Client, Linux PCI/local fit and original
preadmission. Actual RADV cold peer refusal, raced/warm reuse and
independent envelope retirement, portable healthy/unknown and HIP
healthy and real HF HIP V2 C1/C8 replay/retirement pass.
Full isolated gate: 2396 Rust test executions, zero failures, 106 explicit skips; release/installer/scripts pass.
Signed ce83cc18: actual captured profile/source/stream/receipt/cache/
pin/service and prepared cohort mode; actual CUDA GGUF B2 fullV/all42,
graph/native fallback and retirement/unknown retention, native HF
HIP V2 C1/C8 pass. Full assembled isolated gate: 2398 Rust test executions (2396 workspace gate + 2 focused graph tests), zero failures, 106 explicit skips; release/installer/scripts pass.
Signed 14bbbd48/5d952364/5a7535e7: original coherent Vulkan mapped
fresh private producer over exact backing/range/Write/queue/Ready.
Actual RADV full/subrange GPU consumption, original charge/retirement
and refusal; portable Vulkan healthy/unknown and HIP healthy pass.
Full isolated gate: 2396 Rust test executions, zero failures, 106 explicit skips; release/installer/scripts pass.
Final mapped accounting revision revalidated physically and by feature checks.
Minimum Phase 1 exit accepted; audit in FULL_IMPLEMENTATION_PLAN/
CURRENT_HANDOFF. Phase 2.1 UX-1 shared CLI presentation accepted:
2431 Rust executions pass, zero failures, 106 explicit skips; release PTY,
machine JSON and original CPU output equality pass. No execution policy
change. UX-2 typed command tree and shared cold model/config resolution:
canonical commands/globals/compatibility/help/completion, exact digest and
explicit by-name, one selected profile/document, original model/device/cache
owners. Isolated full gate 2451 pass/0 fail/106 skips; synthetic CPU output
equivalence and release completion/cold JSON smoke pass. No new execution
claim. UX-3 shared native ControlPlane: original owner snapshots/actions,
bounded transient notifications and resync, exact model registration and
listener generation guards, original output/cancellation and cache outcomes.
Actual local Qwen 0.8B GGUF two-client lifecycle/loopback and direct output
equality pass. Full isolated gate 2457 pass/0 fail/108 skips (+1 focused
model integration); architecture, feature and release checks pass.
UX-4 bounded canonical single-plan download lifecycle accepted: original
checkpoint/range pause/resume, exact operation control, cancellation,
cold partial records and PTY/plain/quiet/JSONL receipts. Full isolated gate
2459 pass/0 fail/109 skips; separate SDK integration and five monitor
fixture runs pass. Advanced provider forced modes and component aggregate
control remain open. UX-5 one-line run implemented: exact shared cached
name/source selection, positional/file/stdin/config prompts, independent
TTY requests retain one model owner, explicit eligible immutable 2B/27B
acquisition consent and precommit digest guard. Full gate 2465 Rust
executions pass/0 fail/109 skips; combined acquisition/serve lint and
CUDA/HIP/acquisition compile, release/installer/scripts and real Qwen
0.8B GGUF CPU PTY two-prompt equality/exit pass. No new remote or GPU
execution evidence. Next UX-6 stable serve CLI per §46A.28.
General UMA/foreign backing, device-write host read, unqualified model/
backend/media and other expansion holds remain explicit. Historical
CPU cache gate corrected at 00f57e7a/9374e0ca.
Continue [full plan](FULL_IMPLEMENTATION_PLAN.md). No second cache
hierarchy, allocator or publisher; mixed backend/multi-source HF pipeline,
other persisted GPU formats, cross-provider restore, scheduler-managed
persistent sessions and media/DAG/context remain open.

**20 Nov authoritative refresh:** pushed/tracked `origin/main` is `f0f0640`; local signed
source lineage above it includes `1222f8a` (multi-token GDN prefill), `60ba956` (matched
performance profile), `3afb990` (reduced-band GDN layouts), docs-only `7beb1be`,
`0dac5ab` (device-side paged zeroing) and `0680b0b` (batched paired-page gather). The
current fused-tail-zero slice is described below; revalidate actual HEAD first. The warmed public-SDK refill probe
demonstrated B8->B4->refill/prefill->B8 with opt1 goodput 178.769 output tok/s contiguous
and136.488 paged over68 generated tokens; the temporary probe was removed. The next
local source checkpoint removes full-capacity host-zero allocations/H2D from paged KV
gather and fresh-page materialization by reusing the bounded pinned device-side word-zero
transport. On the same matched8x48+3 cell, contiguous is184.409ms/130.145 output tok/s
and paged205.724ms/116.661 tok/s with exact native-B8 tokens. The next shared paired-page
gather then reduces the same matched cell to176.800ms/135.746 tok/s contiguous versus
182.505ms/131.504 paged by collapsing12 key/value pages into one launch per KV domain;
larger prefixes chunk through the same bounded ABI. The follow-on gather path now folds
inactive staging-tail zero-fill into the final paired launch whenever the full plane fits
the existing bounded transport; oversized planes retain the prior separate-zero fallback,
and empty prefixes still explicitly zero staging. The matched cell is terminal-green at
177.467ms/135.236 tok/s contiguous versus180.520ms/132.949 paged, only ~1.7% wall-time
overhead versus ~31.8% at the retained242.488ms/98.974 pre-optimization baseline. The
same mixed4-short+4-survivor+4-queued refill shape has now been rerun on unchanged
`e72030c` source after the paged transport work:400.383ms/169.837 output tok/s
contiguous and422.572ms/160.919 paged for68 generated tokens (~5.5% wall-time gap).
A device-scatter copy+tail-zero fusion was tested and rejected: the cached physical retry
measured395.849ms/171.783 contiguous and421.159ms/161.459 paged, only ~0.3% paged
throughput gain versus the accepted source run while adding another kernel path. The
production experiment wiring and temporary probe were removed; the reusable copy+tail-zero
candidate implementation is retained dormant for cross-model A/B. vLLM comparison is
owner-deferred for now;
do not spend the next slice on the longer-decode paged residual or direct paged-attention/
overlap complexity. Public graph serving remains Native. The older status narrative below is historical context unless
corroborated by current source/live plan sections.

Historical implementation checkpoint recorded below is signed `9d6613c`
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
or vLLM claims. This paragraph is historical; current pushed/tracked `origin/main`
is `f0f0640` and current signed local `main` is `3afb990` as recorded in the 20 Nov
refresh above.

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
2B B2 HTTP also passes. B2+B4+B8 therefore all use 4-token equal-frontier
prefill chunks. Earlier disappeared B4/B8 attempts remain incomplete,
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
fallback immediately and AGENTS uses that flag. Signed `4688cba` adds safe
recognition of a caller-supplied child netns only when self is loopback-only,
PID1's network view differs and no socket FDs are inherited.31/31 harness tests plus
real child-net loopback+CUDA smoke pass. Direct shared-net L5 still refuses; the
explicit `bwrap --unshare-net --bind / / --dev-bind /dev /dev --` wrapper now
supplies the private namespace without weakening `native-gpu`. Course correction from dotlink v0.17.4 forensics: periodic output is
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
**19 Nov P12-C in-flight cancellation gate is accepted on original0.8B paged B1.**
A qualification-only launch cut observes the already-armed ordinary CUDA flight
after real enqueue and before Ready. Known cancellation retires through the
original event ledger only after physical status readback, never initializes
candidate Writes or constructs Ready: root/all42 stay exact, Vector/staging/page
owners remain held until completion, retire to the preflight pool occupancy,
valid replay commits and final shutdown returns RTX3060 to37MiB. Deliberately
dropping the submitted future exercises the existing unknown-completion path:
root binding stays unchanged, the service fences, the pending event,396 submitted
handles and completion owners remain quarantined, and pool backing is not reused.
Two external-SIGKILL attempts before semantic completion remain incomplete; the
direct known-completion run and the final heartbeat-assisted unknown run both
exit0. Heartbeat output is transport plumbing only, not correctness.
**Historical19 Nov explicit high-level paged selector:** public
`QwenCudaStatePolicy::{Contiguous,Paged}` was first carried only by
`ModelLoadRequest::QwenCudaWithStatePolicy`; the fixed Contiguous `QwenCuda` default from
that checkpoint is superseded by the21 Nov generic smart-config layer. Native rows4/8 now use the same explicit
Paged policy and shared Hybrid state owner. Cohort preparation reuses
`OrdinaryCandidateState` lane-by-lane: gather -> unchanged native body -> scatter
under one flight, with actual write outputs promoted and canonical paged Domains
created only after completion. Contiguous 8-row execution retains its private-KV fast path;
paged `KvInPlace` falls back to the existing per-lane append. Focused policy tests,
strict `qwen35-cuda` clippy and `qwen35-cuda-serving` composition check pass. One
compact original0.8B 2/4/8-row Contiguous-vs-Paged prefill/decode/all-state-domain comparison
is physically accepted: exit0 in306.84s, exact prefill/decode/all-state-domain state and clean
shutdown. The two earlier external-SIGKILL attempts remain incomplete only. A
single public `fabric_sdk` cell then loads Contiguous and explicit Paged with
`cohort_width=8`, offers8 concurrent4-token/page-boundary requests with2 output tokens,
observes actual 8-row cohorts `{8}` on both, exact returned token IDs and clean
shutdown; exit0 in340.78s. Public HTTP is now accepted through the explicit
child-private-net wrapper: the same8 concurrent page-boundary workload observes
native 8-row cohorts `{8}` for both Contiguous/Paged, exact returned IDs and clean shutdown;
exit0 in358.07s. Strict serving-test Clippy passes. No local vLLM executable/module
exists here, and retained C8 numbers are not workload-matched with this SDK/HTTP
cell, so no fresh vLLM claim is made. The child-net full-gate attempt exposed and
fixed one generic `cuda-native` import boundary for `HybridStateInventoryPlan`;
targeted production `cuda-native` check passes. The warmed monolithic rerun was
externally SIGKILLed later during ordinary-Qwen feature compilation with no
diagnostic and remains incomplete. Do not add another fixture/evidence framework.
A matched-vLLM rerun is currently blocked by the retained pinned layer store losing
its runnable image index in this connector; exact HF source bytes remain present and
no image reconstruction was attempted. **Complete-domain prefix fork/COW substrate
is now accepted:** a fresh session can publish an immutable full-domain snapshot at
its own new root with no token replay/readback/copy; original0.8B paged3->4 COW
continuation matches independent replay for full logits/all42, cached snapshot stays
immutable and shutdown is clean (`P12_SOURCE_PAGED_PREFIX_FORK_PASS`,179.13s).
**Bounded semantic prefix key/lookup/eviction is now implemented** in
`zerogravity-engine::state_cache`: exact source + grouped execution/input/position/
state/sharing fingerprints, exact semantic-unit prefixes, greatest-prefix lookup,
newest tie-break, duplicate refresh, FIFO eviction and explicit entry/weight bounds.
Payloads are immutable `Arc`s and lookup allocates nothing; the cache never grants
restore authority. Two focused tests, strict engine Clippy and production CUDA
composition pass. **Scoped worker integration is now accepted:** `PrefixSharingScope`
is a nonzero Copy value carried beside the queued request; existing `Request` and
ordinary `request_text` remain no-sharing, while explicit `request_text_scoped`
opts in. Generic worker restore verifies the returned cursor equals the executor's
actual frontier; caching happens only after a committed state-only prompt step. Qwen
uses a model-local cache only for paged state, capped to `active` entries and
`active*context` semantic-token weight. Physical 0.8B page4/max_append64 proof with
prompt65 records model-token deltas65 warm /1 same-scope /65 different-scope and
identical output ID; shutdown is clean in172.98s. Cache-held snapshots are cleared
before resident retirement after the first run exposed that lifecycle requirement.
**Trusted HTTP scope injection is now accepted.** `ServerBuilder` carries one
optional validated `PrefixSharingScope`; HTTP clients have no body/header control for
it, and custom `TextService` implementations fail closed unless they implement
scoped admission. Runtime/Router do implement it. Focused server tests are7/7 and
strict serving Clippy passes. Physical original0.8B page4/max_append64 HTTP prompt65
uses ordinary JSON and records65 warm /1 same scope7 /65 new server scope8 with exact
output ID and clean shutdown in171.97s (`P12_SOURCE_PAGED_PREFIX_HTTP_PASS`).
**Bounded live-snapshot eviction + explicit stale-scope cache invalidation are accepted.**
`PrefixCacheEviction` reports removed entries/semantic weight; trusted
`invalidate_prefix_scope` is routed through the type-erased registry into executor-
owned caches and exposed on Runtime. It purges currently retained entries only; it
never cancels already-admitted work. Trusted owners must stop/drain a stale scope
first. Server stop/scope change alone never purges the cache.
Qwen uses `2*active` entry and `active*context` weight bounds. Physical0.8B active1
page4/max_append64 proves three prefix64 entries FIFO-evict the oldest while keeping
the newer hit; purge releases2/128 and forces a miss; prompt321 weight pressure
retains prefix320 for a1-token hit but evicts prefix256 so prompt257 costs257; final
purge releases2/448, repeated purge0, clean shutdown in179.12s. This qualifies
cache-retention pressure. **Actual external resident page-pool pressure is now
accepted as well:** the qualifier pins real pages from the same finite pool around a
cached prefix64. On original0.8B page4/max_append64 it holds5952 pages, reaches
6144/6144, gets a prelaunch Capacity with unchanged root and exact vector/staging
unwind, releases pressure, then reproduces the complete baseline logits exactly;
final allocated pages are all idle and shutdown is clean in168.51s. One startup
SIGTERM attempt is incomplete only. Qwen admission/health diagnostics now include
current prefix-cache entries/weight/maxima and paged-pool allocated/idle/occupied/max/
page-bytes only—no cumulative hot-path counters. Host/device OOM remains outside this
claim.

**Fresh overlap review:** the accepted prefix/paging/serving changes were re-audited
end-to-end; no correctness/security regression was found. The actual blocker is below
the worker: `PhysicalQueue` remains single-active per card and `ResidentSubmission`
retires only after trusted readback, while Qwen CUDA uses one pinned logical stream.
Do not enable scheduler concurrency and call it overlap. The first lower-level
prerequisite is now accepted: CubeCL `OwnedCudaQueue::fork_stream` creates a fresh
same-service logical/physical stream for a dedicated client clone, and non-clone
`OwnedCudaEvent` owns exact record/query/wait/destroy on that actor/stream. L5 physical
proof used disjoint8MiB writes, recorded both events before waiting either, observed
secondary still pending on first query and got exact dual readback
(`P12_CUDA_SECONDARY_STREAM_EVENT_PASS`). No physical-queue width, StateRoot or worker
policy changed. **Resident event+lease in-flight ownership is now accepted:**
`ResidentCudaInFlight` retains the original armed submission/leases/completion owners
plus one `OwnedCudaEvent`. Event completion alone never creates Ready: known completion
still runs existing readback/validation; cancellation waits the event and retires
without initializing writes; unresolved drop preserves completion-unknown fence +
quarantine and retains the attached owner. Physical L5 marker:
`P12_CUDA_RESIDENT_INFLIGHT_EVENT_PASS ... quarantine_owner_delta=1
queue_width_unchanged=true`. **Bounded multi-flight primitives are now accepted:**
`PhysicalQueue` has configurable bounded active slots (production registry still1),
and `PhysicalServiceBudget` shares one prepaid envelope across overlapping known
flights with first-Submitted/last-Ready accounting and no second charge. The existing
DeviceEventLedger remains the only access-conflict authority. Host queue6/6 and
physical-service5/5 pass; real CUDA bounded2 disjoint dual flights on primary/secondary
streams retire second-before-first with exact8MiB readbacks and reusable service
(`P12_CUDA_RESIDENT_DUAL_FLIGHT_PASS`). Next integrate the real Qwen/source path and
measure actual prefill/decode behavior; avoid another synthetic evidence matrix unless
integration exposes a specific missing invariant.

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
scopes. Return to the Qwen model/context `concurrency=1..64` mission without silently
activating held routes; vLLM comparison is owner-deferred for now.

- [x] Cross-cutting error architecture — real memory/resource exhaustion is now typed
  with exact requested/available bytes and scope, scheduler capacity decisions no longer
  depend on resource-error strings, and public SDK/HTTP classification is centralized as
  `resource_exhausted`/503 versus deterministic `limit_exceeded`/400. Static bounds
  remain lightweight `Limit`; lifecycle failures remain `State`. Preserve this
  distinction across new models/backends — [current contract](../docs/CURRENT_HANDOFF.md).

## Active

- [ ] P1 serving-and-speculation — @pvl — l5 CUDA `.8B`/`2B`, x2 ROCm `.8B`/`2B`/`27B`: source-real correctness/paged/prefix/fused 2/4/8-row cohorts accepted; multi-token prefill + autoregressive refill/regroup now fast under matched opt1. Public CUDA serving passes4K/8K/16K on both small models,0.8B32K has one full pass, and paged2B32K passes32766 tokens in224.80s. Same32K-capacity1024-token A/B is283.22s contiguous-streamed vs28.56s paged-resident (~9.9x). Generic cold-start smart config now jointly resolves Qwen `state_policy` + `cohort_width` from fresh runtime admission: Exclusive=ignore user knobs, Enabled(default)=user values override Auto per knob, Disabled=no Auto; score=resident weights > wider 8/4-row cohort > Contiguous tie. Default2B32K/1024 Auto-selects paged and passes in26.51s exact. Existing `active=8` SDK with cohort width unspecified Auto-selects8 for both Contiguous/user-Paged halves and publishes `{8}` at135.979/133.004 tok/s exact. Nix-realized vLLM0.33.0/cu130 correctness now passes exact local0.8B/2B on L5 without Podman; the frozen wheel graph is store-realized and an optimized CUDA-graph smoke passes, so the retained concurrency=1/4/8/16/32/64 paired matrix is next on L5 before the exact-commit X2 ROCm campaign. Independent populated all-state-domain closure remains open; X2 ROCm remains environment-gated — [details](serving-and-speculation/TODO.md)
- [~] P1 numerics-and-context — @pvl — exact prefill 47 + decode 48–50, lifecycle/capacity and final gate pass; active-context/2B reference inputs missing — [details](numerics-and-context/TODO.md)
- [~] P1 memory-and-streaming — @pvl — 32K storage/execution policy is materially closed:0.8B32K has one full public-SDK pass;2B32K passes the full32766-token retained cell with Paged64/64, and canonical default startup Auto-selects paged when contiguous would stream. Final CUDA admission independently clamps provider capacity to original-actor live free VRAM while preserving configured headroom. CUDA-forward source-page transfer now retains the verified `TensorWindow` itself through the existing checkpoint and uses the original actor's bounded owned-window H2D directly, removing the superseded pinned/Bytes copy chain. Matched 2B/32K prompt48 streamed feedback is38.46s vs39.36s at `b839920`; cache-promotion41.00s, moved-pinned44.88s and direct-SSD/cache-bypass41.77s variants were rejected. A same-tree one-page CUDA retirement-ring Qwen A/B is rejected at39.281s vs38.323s serial (~2.5% slower); production remains serial at that boundary and default smart config avoids it when paged state enables resident weights. The model-neutral primitive remains dormant/reusable across backends: CUDA `OwnedCudaEvent` and HIP `OwnedHipTransferRetirement` implement `RetirementMark`, with zero-cost CUDA/HIP ring aliases over the shared `TransferRetirementRing`. Shared completed-root reuse, allocation-free common-frontier reconciliation, exact partial-tail paged COW ownership and canonical physical allocation+range identity are active; CUDA-forward prepared sets, streamed-head transfer binding and production HIP resident-text workspace/root binding use canonical ledger identity. Remaining work is additional real identity consumers plus cache-aware direct/async IO and broader authenticated streaming tiers — [details](memory-and-streaming/TODO.md)
- [~] P1 cpu-gpu-optimization — @pvl — forward kernel/transport checks pass; exact whole-model parity before measured tuning and the matched benchmark campaign — [details](cpu-gpu-optimization/TODO.md)
- [~] P1 pipeline-composition — @pvl — recursive primitives/pipelines/profiles; shared execution adapters active, reuse current adapters in Fabric; Qwen expansion deferred — [details](pipeline-composition/TODO.md)
- [~] P1 cli-device-routing — @pvl — unified run/config/source/device selection; preserve behavior through Fabric SDK/CLI migration — [details](cli-device-routing/TODO.md)
- [~] P2 model-execution — @pvl — current Qwen CUDA forward owns source-real fused 2/4/8-row contiguous+paged transport, complete-domain prefix reuse, event-backed/multi-flight primitives, retained stateless + multi-token attention/GDN prefill, reduced 4/8-row GDN layout selection and bounded paired-page KV gather with fused inactive-tail fill. Latest matched opt1 short-wave is135.236 tok/s contiguous/132.949 paged (~1.7% wall-time gap); current68-token refill is169.837/160.919 tok/s. Rejected A/B mechanics are retained dormant rather than deleted: lane regroup, copy+tail-zero,17 BF16 GEMM strategies (including restored terms64 shared-row, paired-boundary and K64 weight-cache variants), recurrent tiled transpose and GDN occupancy variants, with zero production consumers. The restoration sweep also repairs default/qualification/CUDA-native test cfg drift; the largest native-qualification lane is external-SIGKILL incomplete. Public graph serving deliberately remains Native. Large same-contract CUDA numerical acceptance still depends on approved references; populated small HIP and resident27B work remain separate. X2 is shared: preserve other owners and verify only ZeroGravity's incremental resource delta retires — [details](model-execution/TODO.md)
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
