# ZeroGravity — Full Comprehensive Agent Handoff — 2026-10-25

Author: handoff assembled from the interrupted Codex thread "Review ZeroGravity handoff"
(Codex session `01a158d8-c120-7eb2-83f4-9a46420d9e9a`, started 2026-10-20, interrupted
2026-10-24T21:12:53Z by a ChatGPT/Codex usage-limit error; reset date Oct 31, 2026 1:41 AM),
plus independent revalidation of the live repository state at handoff time.

Everything below distinguishes:
- **[VERIFIED]** — independently confirmed in the live repo at handoff time.
- **[ACCEPTED]** — validated and accepted by the interrupted session's coordinator (ROOT)
  with evidence in `.agents/runs/`; not re-executed during this handoff.
- **[PENDING]/[INTERRUPTED]** — not done or stopped mid-flight; the resume point.

## 0. Orchestration change (user decision, this handoff)

- The previous delegation model (Sol xhigh coordinator / Astra high architect /
  CLI Pi `deepseek-v4.1-flash:1017` implementers, per-run Pi Bash guards, pidfd
  supervision chains, run-receipt ceremonies) is **DISCONTINUED**. The next agent
  uses its own orchestration rules. All orchestration-specific rules are dropped.
- **KEPT: the 5 parallel lanes.** The five isolated lane worktrees and their
  scopes remain the working structure:
  - `resume-all-20261024-qwen27b` (Qwen27B)
  - `resume-all-20261024-gemma4` (Gemma4)
  - `resume-all-20261024-hip` (HIP/AMD)
  - `resume-all-20261024-vulkan` (Vulkan)
  - `resume-all-20261024-queue` (shared queue/continuous batching)
- Keep the underlying AGENTS.md repo rules that are about safety and correctness,
  not orchestration: signed commits, no push/destructive ops without explicit user
  confirmation, preserve unrelated work, root README bytes never change,
  no model/SDK downloads or device privilege changes without user authorization,
  local gfx90c AMD launch hold, XDG paths, terse writing.

## 1. What ZeroGravity is and the overall goals

Rust workspace at `/home/pvl/spaces/abird/src/zerogravity`. Goal: the fastest and
most composable AI execution fabric — a single shared Rust engine, immutable
source identity, independent StateRoot ownership, physical ResourceArbiter, fair
workers, actual completion/quarantine — with model-family runtimes (Qwen 3.5
0.8B/2B, Qwen3.8 27B, Gemma4 E2B) reusing shared primitives, and multi-backend
support: CPU, CUDA (local RTX3060 Laptop 6GiB), HIP/ROCm (pvl-x2 gfx1151), Vulkan.

User's stated end goals across the thread:
- Full numerical parity with vLLM, then **strictly superior throughput** vs vLLM
  (measured, matched, honest).
- A three-layer composability architecture: **primitives layer → pipeline layer →
  profiles layer**; pipeline nodes may contain sub-pipelines; shared engine
  primitives → model-family primitives → per-model specializations, minimal
  duplication, zero-cost abstractions, buffer pools/rings/reusable buffers,
  optimized GPU–CPU boundary with smart avoidance of boundary trips.
- Adding a new model (another Qwen, another Gemma) or new backend (CUDA/HIP/Vulkan)
  should reuse nearly all prior work.
- CLI ergonomics: `zerogravity run <model|config-path|hf:…|gh:…|http(s):…|local path>`
  with automatic best-route device selection (NVIDIA → AMD → split-layer
  GPU+CPU → CPU; later NPUs, incl. speculative decoding on NPU when faster),
  plus forced `run-cpu`, `run-cuda`, etc.; everything flag-expressible must be
  config-expressible; default config `XDG_CONFIG_HOME/abird/zerogravity/config.yaml`
  falling back to `.config/…/config.yaml`.
- Serving: continuous request batching, unlimited default pending count (like
  vLLM) with an optional configured cap; graceful AND forced cancellation with
  timeout (deferred until after benchmarks — still on the list).
- Benchmarks ladder: C1/C4/C8 (done), C16, C32, C64 for both 0.8B and 2B;
  then Qwen27B and Gemma4 on pvl-x2; Vulkan on par with CUDA; ROCm/HIP for AMD.

## 2. Live repository state at handoff [VERIFIED]

- HEAD: `4bf3673` ("docs(plans): activate CLI, composition and compute").
  All 78 signed logical implementation commits are complete and in history.
- Index: empty (0 staged files). Root README byte-identical
  (SHA256 `6c00873e…da5b15`, matches the protected value).
- **Main worktree holds a large UNCOMMITTED assembled batch**: 70 modified +
  21 untracked paths, ~15,902 insertions. This is the five-lane + CUDA assembly
  described in §5. Nothing of it is committed.
- 6 registered worktrees: main + the 5 lanes (each still carrying its lane's
  uncommitted candidate changes: queue 29 files, vulkan 37, hip 11, qwen27b 9,
  gemma4 3 + untracked `tools/gemma4-hip-qualification/`).
- Unregistered but present trees under `worktrees/`: `gemma4-ple-pi-20261024`,
  `qwen27b-attention-tail-20261022`, `qwen27b-natural-pi-20261024`,
  `root-lowest-f32-20261020`, `vulkan-gemma-pi-20261024`. Keep: two reference
  roots (`root-lowest-f32`, `qwen27b-attention-tail`) while current checks read
  their files; three donor extracts until their work is recovered.
- Branches: main + 5 active lane heads + 1 reference head only; 30 obsolete
  merged local branches already removed (atomic, backed up, verified).
- No leftover pi/deepseek implementation processes running.
- Plan queue (`.agents/plans/TODO.md`): active plans serving-and-speculation,
  numerics-and-context, memory-and-streaming, cpu-gpu-optimization,
  pipeline-composition, cli-device-routing, model-execution, model-acquisition.

## 3. Progress report and validation status

### 3.1 Numerical parity and benchmarks [ACCEPTED]

- Full numerical parity with vLLM (pinned vLLM 0.31 eager/BF16 recipe) achieved
  for Qwen 0.8B and 2B: physical, full-2B numerical composition, API/HTTP,
  cold/whole compiler deltas 0, all owner/EOF closures.
- Fresh matched wins (source21, RTX3060 6GiB, `.agents/plans/cpu-gpu-optimization/SHORT_HF_RESULTS.md`):

  | Band | ZG tok/s | vLLM tok/s | Ratio |
  |---|---:|---:|---:|
  | .8B C1 | 67.7432 | 38.8470 | 1.74× |
  | .8B C4 | 95.6083 | 42.8328 | 2.23× |
  | .8B C8 | 92.9820 | 54.8241 | 1.70× |
  | 2B C1 | 45.0305 | 37.7431 | 1.19× |
  | 2B C4 | 80.8278 | 50.2514 | 1.61× |
  | 2B C8 | 80.7265 | 62.4442 | 1.29× |

- **C16 (frozen source21, active8/B4)**: 2B **loses**: ZG 78.7859 vs vLLM
  95.9974 tok/s (walls 1.62466s vs 1.33337s). Each arm completed 16 offers /
  128 tokens, 0 errors, clean closures — a rate loss, not a correctness failure.
  .8B C16 timings are diagnostic only (first overlapped an audit scan; the one
  repeat 105.7331 vs 96.6544 had an extra sampler). No third rerun authorized.
- **Root cause direction**: software-dispatch overhead at higher concurrency;
  the prepared-B4 certification/state-transport design
  (`.agents/runs/cpu-gpu-optimization-astra-c16-shared-gap-20261024/DESIGN.md`)
  targets up to 22.13% fewer software dispatches — a ceiling, NOT a measured
  wall-speedup claim. Implementation is **[PENDING]**; a fresh matched C16 is
  required after it; C32/C64 follow queue acceptance.
- No C2 band was ever requested.

### 3.2 Shared queue / continuous batching [ACCEPTED CPU, GPU pending]

- Design: unlimited default pending count (vLLM-style growable structure), optional
  configured cap, application-owned HOST admission (not process RSS).
- Real admission bug found and fixed: each queued request retained ~240MB
  temporary tokenizer scratch (≈224MB fixed VM scratch + ≈17MB cache growth per
  tokenizer); the 12th offer hit the 4 GiB HOST limit. Fix: per-tokenizer
  `Retained` owner (R+G+V, single private mutex, concurrency-1 proof), remove
  per-request duplicate charges, CLI direct-route preliminary-tokenizer
  reservation leak fixed.
- **CPU functional acceptance PASSED** [ACCEPTED]: real 16/32/64-request
  populations offered before draining, 4096MiB HOST, 8 active slots, full
  production P16 affinity (not the earlier taskset-4 evidence), plus cap,
  cancellation, and distinct 1024MiB retained-stage refusal cases. Release
  serve build + strict + 14+1 tokenizer text-feature tests pass.
- CUDA short-HF **8 GiB HOST policy** added as a selectable profile (CPU 4096
  and legacy HOST4 remain), because the new B4 allocation source bound
  (4,341,525,759B) exceeds the old 4 GiB host allowance. No 8GiB fit claim made.
- **Pending**: assembled GPU checks, HOST-fit validation for the 8 GiB profile,
  final full gate; several queue production-bound source findings were folded
  into the assembled batch (HTTP double-wrap/max_output, Pre-pool undercount,
  constructor coexistence, last-reader accounting, Rayon reuse semantics).

### 3.3 CLI device routing / run redesign [PARTIALLY DONE]

- Unified `run` with source selector (local paths, catalog/model names, `hf:`,
  `gh:`, `http:`, `https:`, config path, bare run with default config) and
  forced-device routes; config-default `XDG_CONFIG_HOME/abird/zerogravity/config.yaml`.
- Remaining open items in `.agents/plans/cli-device-routing/TODO.md`
  (Uncommitted): shared source-scheme precedence over `.yaml` suffix
  classification, ambiguous-variant rejection, dry-run read-only, shared
  pre-Client transactional admission (registration/current-free checks are not
  an OS hard reservation).

### 3.4 Qwen27B (Qwen3.8-27B) lane [SOURCE ONLY, capture blocked]

- Extensive reuse from Qwen 0.8B/2B shared+family primitives; specializations
  for 3.8 architecture incl. MTP support were required (user directive: reuse
  but keep family-specific optimizations, incl. MTP for 27B).
- vLLM-optimization-parity audit for the model performed; corrections folded in
  (external-pin, owner/helper preadmission, truncated argv, array capacity
  undercount, unchecked arithmetic, canonical 851-role, retained-5FD).
- **Natural reference capture remains BLOCKED**: Podman image identity could not
  be verified (`/run/lock/netavark.lock` permission, then overlay MS_PRIVATE
  EPERM). Decision recorded in
  `.agents/runs/resume-all-queue-production-bound-astra-review-20261024/Q_REFERENCE_ROUTE_DECISION.md`:
  stop nested-rootful/private-TmpDir retries; any native-rootless route needs
  complete original-container resource/identity/cleanup verification. No
  numerical acceptance for Qwen27B exists.

### 3.5 Gemma4 lane [SOURCE ONLY]

- Gemma4 E2B acquired via HF path (user approved HF cache quota increase to
  96 GiB; tokenizer+config downloaded).
- CPU path ready earlier; learned PLE hit/transfer coverage open; controller
  13 groups pass; physical 8 cases NOT executed.
- 22-path tracked `tools/gemma4-hip-qualification/` controller promoted into
  main (73 qualification-tool files total transferred to main, bytes/modes
  verified) [ACCEPTED].

### 3.6 HIP/AMD lane [ACCEPTED source, physical pending]

- Target: pvl-x2 gfx1151 (ROCm/HIP); **local gfx90c launch hold remains in force**.
- H14 pending-event source (14 files / 524,658B, incl. final sticky
  witness-error projection) transferred into main's working tree with fresh
  mtimes [ACCEPTED/VERIFIED present in main diff].
- Guarded CPU controls passed (19+1 owner/ring tests, HIP GPU lib + 12 example
  targets compile, 17 CPU tests). Serial staging-charge regression,
  markDrop-reentry, destroy-recheck corrections applied and reviewed.
- **Pending**: enabled-HIP typing, H6 serial physical cases (see §5 plan).

### 3.7 Vulkan lane [ACCEPTED source, physical pending]

- Vulkan-to-CUDA parity work: strict execution adapter, fresh-registry Cargo
  cache (approved inputs only), six registry checks passed (copying, missing
  inputs, checksum rejection, both cancellation cases, FD cleanup).
- Final five-fix strict-SSH host adapter (1,040 lines, SHA
  `1c12cfe11c45657fd6e5b707f44d9c0c51de36b9956c87b43c1d394a687833a6`)
  source-reviewed/accepted; five tiny fixes closed and accepted.
- **Vulkan source-census predicate fix** (pre-existing `__pycache__`/`.pyc`
  files must count in the full manifest census): closed source-only in
  `.agents/runs/resume-all-hgv-checkpoint-execution-sol-20261025/v-census-boundary-binding-1/`
  (2 main paths fixed; `V_BYTES`/`V_SHA` release constants still `None` —
  pending). Fresh final packages prepared under `hip-final/` and `gemma4-final/`;
  `vulkan-final/` package still to be produced.
- **Pending**: V10 serial physical cases; real registry/offline build/backend/SSH.

### 3.8 CUDA assembled batch [INTERRUPTED mid-checks]

- CUDA8 production corrections (83982 lineage; 96199 and 11638 drafts rejected
  with reasons: forced-cost-refusal, swallowed positive failures, cfg/type/
  capacity defects) accepted into main; CUDA11 assembled completion (11 files /
  317,979B incl. 3 test-oracle corrections) applied to main and formatted rc0.
- Affected build/strict batch
  (`.agents/runs/resume-all-cuda8-affected-execution-pi-20261025/`), run against
  the assembled main source [VERIFIED artifacts]:
  - build-cli rc0, build-core rc0, build-gpu rc0, build-product rc0
  - strict-core rc0, strict-gpu rc0
  - **strict-cli rc101 — FAILS, 2 clippy errors**:
    1. unused import `fixed_size` — `crates/zerogravity-cli/src/serve/gpu_engine.rs:9`
    2. collapsible_if — `crates/zerogravity-cli/src/run_cli/short_hf.rs:444`
       (arbiter snapshot `fenced || limit != 5120*1024*1024 || reserved != 0`)
  These two fixes are the immediate resume point for the assembled batch.

### 3.9 Housekeeping completed [ACCEPTED/VERIFIED]

- 43+5 registered obsolete worktrees retired (all source/evidence archived and
  rechecked); nine more trees retired with 18 TAR archives compared; two
  historical registrations deregistered with reference files preserved at
  exact paths; 30 obsolete local branches removed atomically with backups.
- Root README/index/HEAD preserved throughout. ~39.72GiB freed by tree cleanup.

## 4. The assembled main batch (uncommitted) — contents

The 91 paths in main's working tree combine: five-lane assembly (134-path Rust
unit incl. shared thread-control/Rayon fixes and the portable
`TransferAdmission` type in core), H14 transfer, CUDA11 corrections, queue
tokenizer/VM-credit repair, portable admission-type fix (CPU serving referenced
a Linux-only export), 73 qualification tools, plus new untracked files
(`tokenize_admission.rs`, `event_channel.rs`, `pending.rs`, `json_allocation.rs`,
`final_native.rs`, `vulkan_native_policy.rs`, gemma_text same_pci/vulkan owner
modules, vendor cubecl-hip source_page, `cubecl-wgpu/`, qwen35 `tokenize/`,
gemma4 vulkan examples, five `tools/*-qualification/` dirs).

**No `scripts/check.sh` full gate has been run on the assembled batch.**
Nothing is committed. Per repo rules, commit verified implementation+tests+plan
updates together with explicit paths, as logical signed commits.

## 5. Interruption point — exact resume queue

The thread died at 2026-10-24T21:12:53Z (usage limit; subagent
`astra_h_pending_final_review` also errored) mid-flight on:

1. **Fix the 2 strict-cli clippy failures** (§3.8), rerun strict-cli; then run the
   complete `scripts/check.sh` gate on the assembled main batch
   (`CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`).
2. **Finish the V package**: set `V_BYTES`/`V_SHA` from the final five-fix
   adapter file, produce `vulkan-final/` package (fresh; never retag old
   identities; old packages preserved).
3. **H/G/V physical checkpoint on pvl-x2** — fully prepared plan in
   `.agents/runs/resume-all-hgv-checkpoint-execution-sol-20261025/TODO.md`
   ("Released execution order"): source A/M guard + cold costs + local
   four-operand prepares; original H/G host admin + offline builds (3,600s each)
   + actual bind via strict SSH; V adapter `admin`/`bootstrap`/`build` once
   each; then SERIAL cases — H6 (known-qwen-reload; qwen-final-reader-unknown;
   gemma-feedback-unknown; source-page-query/wait/destroy-unknown), G8
   (resident; session-known; snapshot-known; resident-row-unknown; unknown;
   resident-unknown; session-release-fault; snapshot-release-fault), V10
   (resident; session-known; snapshot-known; resident-unknown;
   resident-row-unknown; session-release-fault; snapshot-release-fault;
   partial-native; peer-vulkan-then-hip; peer-hip-then-vulkan). Inspect EVERY
   case's semantic/source/native/binary/data70/cache/state/quarantine result and
   TERM/two-KILL/one-reap/group-absence/full-EOF cleanup before the next.
   Remote run names: `resume-all-five-{hip,gemma4,vulkan}-20261025-1`.
   Bounds: one x2 GPU owner PCI `0000:c6:00.0`; serial physical; caps H
   host/provider32GiB/physical64GiB, Q12/G16 cold, G/V host16/provider20/
   physical64GiB (G unknown provider16); no cap growth or forced streamed
   Gemma fallback; wall caps total 81,000s; free host RAM ≥16GiB; no weight
   transfer/import/download/cache copy.
4. **Commit the assembly** as logical signed units (only after the gate),
   reconcile plan TODOs (`## Uncommitted` → under-landing-commit sections),
   then diff each of the 5 lane worktrees against main and retire
   trees+branches only for fully landed lanes (preserve donor/reference trees).
5. **C16 fix**: implement shared prepared-B4 certification/state transport per
   `.agents/runs/cpu-gpu-optimization-astra-c16-shared-gap-20261024/DESIGN.md`
   (keep existing arithmetic, caps, independent roots), then fresh matched C16
   for both models; then C32, C64 (genuine offered populations, honest
   active8/B4; same-attempt source/profile/stock arm, no cherry-picking).
6. **Qwen27B natural capture** — resolve image identity per
   Q_REFERENCE_ROUTE_DECISION.md constraints (no nested-rootful retries without
   full original-container verification), then reference capture and numerical
   acceptance; Gemma4 learned-PLE coverage similarly open.
7. **CLI device routing finishing items** (§3.3).
8. Deferred user-approved backlog: graceful + forced cancellation with timeout
   and stuck-model loop detection (after benchmarks); NPU integration; DeepSeek
   and other model families on the TODO list; `zerogravity run` automatic
   device-routing behaviors beyond the designed subset.

## 6. User decisions (all, in order) — binding unless the user changes them

1. Planning/investigation first; implementation only on explicit resume
   (this was later superseded by repeated "continue/resume all tracks").
2. Fix at the root, lowest layer first; establish correctness upward.
3. Do end-to-end units in one turn without per-step user prompting; avoid
   too-fine validation; batch related work; validate assembled batches, not
   each helper.
4. If a task blocks on validation/correction, advance unrelated lanes in
   parallel; use worktrees; merge at completion of contained units or periodic
   checkpoints to limit drift. (Recorded generically in AGENTS.md.)
5. Research directives (parallel subagents, then reconcile into design):
   DeepSeek-v4-flash-on-GB10 serving recipe, DeepInfer, TensorFold, vLLM,
   llama.cpp, Halogen, Dwarfstar, Maya — optimization-by-optimization
   accounting; composable pipelines; integrate big speedups right after vLLM
   parity; add DeepSeek/others to TODO deferred. Primitives → pipelines →
   profiles layering, sub-pipelines per node.
6. Full vLLM parity, then benchmarks, then Qwen27B; keep binding work and
   benchmarks parallel.
7. Cancellation: both graceful and forced with timeout, stuck-loop breaking —
   AFTER benchmarks, not before.
8. Finish the full HF correctness + benchmark matrix; run 0.8B benchmarks in
   parallel with remaining impl; parallel subagents for 2B benchmark, Gemma4,
   Qwen27B.
9. Gemma4 source: use the HF path and pull it; HF cache quota → 96 GiB approved.
10. Benchmark ladder: quick 0.8B C1/C4 (then C8) vLLM vs ZG on both; investigate
    the huge C1 gap before wider runs; we want ZG faster than vLLM; cancel
    broader bands until the gap is fixed; then C8, then 2B; both engines.
11. Lift 0.8B optimizations engine-wide; shared-engine + family-specific
    profile work; rerun benchmarks; C1, C4, C8 all superior first; then 2B;
    then merge Qwen27B and Gemma4 and do the same for them on pvl-x2.
12. Vulkan on par with CUDA; ROCm/HIP for AMD.
13. Allocation-cost directive: buffer pools, reusable buffers, rings;
    zero-copy, idiomatic zero-cost abstractions; optimize the GPU–CPU boundary;
    clean minimal API from the shared engine.
14. 2B head-path bottleneck (1.017GB upload, 970 syncs/forward) is the same
    optimization set as 0.8B — implement shared: engine primitives → family →
    model specialization layers.
15. Astra-style agent for architecture/design review/guidance only, never
    implementation; implementation by the (now-discontinued) executor model —
    *the review-before-implementation discipline itself is worth keeping*.
16. Qwen27B/Gemma4 reuse 0.8B/2B code wherever possible; parameterize Qwen3.x
    modules across models; Qwen3.8-27B supports advanced architecture (MTP
    etc.) — add family specializations, don't blindly reuse.
17. CLI redesign (full routing spec quoted in thread, incl. config-path and
    bare-run default-config support; XDG default config location).
18. Benchmark bands: no C2; C16 after C8; then C32 and C64. Record best
    verified per band, same-attempt arms only.
19. Queue: unlimited pending by default like vLLM, optional config cap;
    growable deque/priority-heap subject to host memory acceptable — do better
    if possible.
20. Commit all main-worktree changes as logical units (many commits); then
    audit and delete completed/merged/defunct worktrees and unneeded branches.
21. Five lanes in parallel (qwen27b, gemma4, vulkan, hip, queue); finish each to
    a mergeable checkpoint; prep and merge everything into main at the next
    reasonable checkpoint; clean up and start fresh; don't transfer broken
    drafts to reduce tree count.
22. Orchestration (now discontinued): Sol xhigh coordinates; Astra high for
    complex architecture/math; Pi/deepseek for implementation and first-pass
    reviews; spin up many parallel executors; xhigh thinking, final reports
    only. → **Superseded by §0 of this handoff.**
23. Everything interrupted twice by usage/limits; user's standing directive:
    "continue all, all parallel tracks" — i.e., resume lanes autonomously to
    their mergeable units, don't wait per-lane.

## 7. Where the evidence lives

- Plan queue + live plan TODOs: `.agents/plans/` (read `README.md` rulebook
  first). Most current: cpu-gpu-optimization, model-execution, platform-and-release,
  serving-and-speculation, cli-device-routing.
- Historical comprehensive handoffs: `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17/19/20.md`,
  canonical sourcebooks and draft patches under `.agents/docs/handoff-2026-10-20/`.
- Run evidence: `.agents/runs/resume-all-*` (root coordination journal:
  `resume-all-parallel-root-20261024/TODO.md`; source supervision:
  `resume-all-parallel-source-supervision-sol-20261024/`; HGV physical plan:
  `resume-all-hgv-checkpoint-execution-sol-20261025/`; CUDA batch:
  `resume-all-cuda8-affected-execution-pi-20261025/`, `resume-all-cuda11-assembled-completion-pi-20261025/`).
- Benchmark numbers: `.agents/plans/cpu-gpu-optimization/SHORT_HF_RESULTS.md`.
- Raw interrupted-session JSONL (if ever needed):
  `~/.codex/sessions/2026/10/20/rollout-2026-10-20T18-26-02-01a158d8-*.jsonl` (399MB).

## 8. Standing constraints (unchanged)

- One Rust mathematical engine; immutable source identity; independent StateRoot
  ownership; physical ResourceArbiter; fair workers; real completion/quarantine.
- GPU/model changes require real source/state/cancellation/physical tests —
  compilation or mocks establish nothing.
- Local Renoir/gfx90c AMD launch hold remains.
- No downloads/SDKs/device-privilege changes without user authorization.
- Full gate: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  before accepting the assembled batch.
- When merging worktree source, copy bytes with fresh mtimes (stale timestamps
  can make Cargo reuse a stale library).
- Root README is user-authored; never change its bytes.