# ZeroGravity — Full Comprehensive Agent Handoff — 2026-10-25 (afternoon)

> **Successor audit (later 2026-10-25):** Read [`POST_4BF3673_AUDIT_AND_RESUME_2026-10-25.md`](POST_4BF3673_AUDIT_AND_RESUME_2026-10-25.md) **first**. It independently found and fixed CUDA-only feature leakage into HIP/Vulkan builds, the invalid H Gemma-feedback-refusal oracle, a G warmed-row-cache fixture assumption, and V false-permit fields missing from *four* report variants. It documents new regressions and fixes, a subsequently **complete fresh CPU-gate rc0** and cross-backend compile gate rc0, unchanged hardware blockers, and required rebinding. The source/pin/physical claims below describe the **prior snapshot**, not the successor's changed tree.

Author: takeover session `01a170a7` (main agent), assembled at session
wind-down per user directive. Supersedes
`FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25.md` (morning) for
everything below; the morning handoff remains valid for its
orchestration-change decision and history.

Everything distinguishes **[VERIFIED]** (independently confirmed in the live
repo), **[ACCEPTED]** (validated by lane agents with evidence in
`.agents/runs/`; ROOT inspected the results), **[PENDING]/[INTERRUPTED]**.

## 0. Orchestration (current, user-directed)

- Orchestration per user directive during this session: the ROOT main agent
  chairs; **glm-5.3 xhigh subagents** own each parallel lane (design,
  implementation, execution, review). The morning handoff's §0 rules still
  apply otherwise. **KEPT safety rules**: signed commits (`gpg.format=ssh`,
  key `pvl@prasannavl.com`; `SSH_AUTH_SOCK=/run/user/1000/gcr/.ssh` is
  required for commits), no push/tag/publish/history-rewrite without explicit
  user confirmation per action, root README bytes never change (sha256
  `6c00873ec0ac2c59bc73703b55a7bfa4d93703334a03fa16bcac2d204cda5b15`),
  no model/SDK downloads or device-privilege changes without user
  authorization, local Renoir/gfx90c AMD launch hold, XDG paths, GPU claims
  need real physical evidence, preserve failed/negative results.
- Subagent discipline learned this session: **every bash command needs a hard
  timeout and scoped searches** (rg with paths; `.agents/runs` is 433GB —
  never grep -r it; ROOT killed two runaway greps). Max 4 concurrent agents;
  the subagent queue drains as agents complete — steer running agents, resume
  (`resume:` param) completed ones; the launcher's spawned
  `pi --provider ollama-cloud --model deepseek-v4.1-flash:1017` inside
  `launch-host-phase.py` is fixed reviewed tooling, not an implementer.

## 1. What this session completed [VERIFIED]

### 1.1 The main checkpoint is formed and gated

- HEAD `6f1e094` on `main`. Signed commits this session (in order):
  `8f01635` (core portable admission/thread-control roots), `5d40cf5`
  (shared pending queue + ingress credit), `52f76a0` (short-HF tokenizer
  credit + HOST8), `efd7a34` (CUDA fragment inventory + host bound),
  `19ff059` (Vulkan native catalogue via cubecl-wgpu, 105 files),
  `29a98c1` (HIP source-page events + witness, H14), `e8db111` (Qwen27B
  HIP matrix + natural tools), `332038e` (Gemma4 HIP final-owner + matrix),
  `68e13a8` (lockfile), `74d0c71` + `1269b8e` (plan reconciliation + lane
  retirement docs), `06b89da` (HIP probe witness page-id typing E0308 fix),
  `c217c35` (V state-base parents fix + attempt-2 bases), `3941a3c`
  (Gemma resident report-phase envelope 20→4GiB fix), `e6ab074` (H/G source
  census skips build dirs + HIP attempt-3 base), `2304f8b` (CLI routing
  finish, cherry-picked), `d3d21a3` (B4 grouped transport + certification,
  cherry-picked), `1565683` + `6f1e094` (docs: queue GPU acceptance;
  checkpoint record).
- **Full CPU gate rc0 on the merged tree** (`check.sh` under
  `native-gpu.py`, jobs2/Rayon4): log at
  `.agents/runs/resume-all-takeover-root-20261025/checkpoint-gate.*`.
- Index clean; only `main` + `origin/main` refs; all lane worktrees/branches
  retired with evidence preserved under
  `zerogravity-worktrees/.preserved-lanes-20261025/` (SHA256SUMS, RECORDS.md,
  OLD-OIDs incl. the b4/cli merge branches `7765a3a`/`2ac9cb3`).
- Reference/donor trees kept: `root-lowest-f32-20261020`,
  `qwen27b-attention-tail-20261022`, `gemma4-ple-pi-20261024`,
  `qwen27b-natural-pi-20261024`, `vulkan-gemma-pi-20261024`,
  `.preserved-nine-20261024`.

### 1.2 Queue lane GPU acceptance [ACCEPTED]

22/22 rc0 (builds, stricts, GPU cuda_compiler_context 4/4 + loaded/inventory
4/4, CLI baseline/serve-short_hf, host-profile umbrella, four ignored route
cases alone per model, five actual source-cost cases). Measured at 8GiB
configured HOST domain: N1 .8B host_upper 4,075,700,811B; N1 2B
4,111,001,099B; B4 2B 4,741,293,239B; B4 .8B 4,736,259,779B; legacy 4GiB
refusal tied to authenticated host cost (excess 446,325,943B). Honest limits:
application-owned configured-domain fits, NOT process RSS / real RAM/cgroup /
device-serving execution. Evidence:
`.agents/runs/resume-all-takeover-queue-gpu-glm-20261025/`.
Anomaly recorded: `target/h24-cuda-local-sdk` include view stale (symlinks →
GC'd nix paths); live cuda12.9 store paths exist if a view rebuild is ever
approved; env unchanged.

### 1.3 CLI routing finished and merged [VERIFIED]

`2304f8b`: scheme precedence (pull-shaped namespaces with .yaml suffix stay
sources; `foo://` refuses at parse), ambiguous-variant rejection (exact
absent/ambiguous strings), dry-run read-only (pure sm86 scan + exclusion
traces + `dry_run`/`reservation` fields + cache-tree/ledger audits), and
transactional pre-Client card admission (`admit_pre_client_card`: 5120MiB
ProviderEnvelope hold via register_matching+admit, dropped at verified
identity BEFORE `original::load`). Verified on main: bin tests 99 passed,
run_unified 4/4, strict-cli clippy rc0. Serve wiring is a documented
follow-up (same startup gap, needs its own identity qualification).

### 1.4 B4 certification/state transport implemented and merged [ACCEPTED]

`d3d21a3` (26 paths, +2869/−217; branch was `7765a3a`): sealed
`FiniteCohortLeases` + `ProducerWitness::{Values,Cohort}` + cold
`FiniteCohortGroups`; `CertificationRun` off `Arc<ModelValues>`; N1 path
byte-identical; grouped typed transport (GDN 18×4→72 launches, KV 6×2→12)
via NEW `.cu` files (originals untouched); `CohortLibrary` cold definitions;
receipt `finite_certification` additive. Real ABI binding counts 9/22/30
(all ≤32). Unit green: 594 gpu lib tests + 17 new; clippy/check green on the
three selections. **PENDING**: the B4 physical GPU round (ignored matrices +
`actual_grouped_state_transport_oracle`) and fresh matched C16 — an agent
was launched and stopped at wind-down (see §4.1).

### 1.5 HGV physical checkpoint — three passes executed [ACCEPTED]

Bound plan: `.agents/runs/resume-all-hgv-checkpoint-execution-sol-20261025/`
(TODO.md + `hgv-execution-glm-20261025/LOG.md` = the complete evidence).
Driver: `v-census-boundary-binding-1/launch-host-phase.py <phase>` (spawned
deepseek Pi under the 298-line guard; launcher pins + REVIEWED-SCRIPT-PIN.json
released by ROOT).

**Pass 1** (M `92e96b9a…`): h-admin + g-admin PASS; h-build/g-build failed
cargo E0308 (`source_page.rs:612` — HIP-gated probe-witness typing, invisible
to local gates); v-admin failed (VS state base never created). Fixed:
`06b89da` (typing), `c217c35` (V admin parents=True + attempt-2 bases).
**Pass 2** (M `cfca1306…`): h-build PASS (E0308 fix confirmed on x2,
489.2s); h-known-qwen-reload re-attempt (authorized: attempt-1 was a missing
binding precondition) FAILED at the post-run source census (build `target/`
counted as source extras — landed H/G controller defect); fixed `e6ab074`
(census IGNORE policy) + `3941a3c` (Gemma envelope) → pass 3.
**Pass 3** (M `63f9cc5e…`, fresh `hip-final-5`/`gemma4-final-5`, HIP base
`-3`, G/V `-2`):
- PASS: h-admin, h-build (artifact `b2f006ae…`) + bind, **h-known-qwen-reload
  (737.1s, PASS_FINITE_SHARED_SOURCE_SCOPE)**, **h-qwen-final-reader-unknown
  (381.7s)**, g-admin, g-build (+bind), v-admin (**the parents fix works**),
  v-bootstrap, v-build (1,006.6s; 3 artifacts).
- FAIL (preserved): h-gemma-feedback-unknown, g-resident, v-resident (below).
- 19 cases stopped as sequence-dependents per stop policy; every failure kept
  full TERM/2-KILL/1-reap/group-absence/EOF closure and clean GPU retirement.

### 1.6 Lane evidence dirs (all gitignored, local)

`resume-all-takeover-root-20261025/` (coordination TODO + gate logs);
`-reconcile-prep-glm-20261025/` (COMMIT-SERIES.md);
`-qwen27b-capture-glm-20261025/` (ROUTE-DESIGN.md);
`-b4-design-glm-20261025/`; `-cli-routing-design-glm-20261025/`;
`-b4-impl-glm-20261025/`; `-cli-routing-impl-glm-20261025/`;
`-gemma4-ple-review-glm-20261025/`; `-queue-gpu-glm-20261025/`;
`-gemma-teardown-rc-glm-20261025/` (ROOT-CAUSE.md);
`-qwen27b-x2-capture-glm-20261025/`; `-b4-gpu-c16-glm-20261025/`.

## 2. THE THREE OPEN BUGS (successor's first work) [VERIFIED]

All three are landed-source or controller defects surfaced by real physical
runs on pvl-x2; all fix sites pinned; none affect the merged checkpoint's
CPU gate.

1. **Gemma HIP teardown/feedback assertion (h-gemma-feedback-unknown)**
   — child exit 1 at 385.0s, silent at keyword level. ROOT's discriminating
   read pinned the exact errors: `invalid session state: original native
   context identity/policy/fence differs` (×2, with `HIP native retirement
   retained after uncertainty`) and the final 46-byte assertion
   `scope closes only between completed proposals` (hip_shared_models main's
   `need()` after completed physical work; `model-report.json` never written).
   Root-cause agent's analysis + machine-code teardown elimination:
   `.agents/runs/resume-all-takeover-gemma-teardown-rc-glm-20261025/ROOT-CAUSE.md`
   (top suspect: the Gemma final-owner fault boundary
   `qualify_final_feedback_drop` in
   `crates/zerogravity-gpu/src/resident_program/gemma_text/final_owner.rs` —
   the only never-hardware-exercised code in this mode). Successor: read the
   H binary's gemma-feedback mode + the session-state identity/policy/fence
   validation + the `scope closes only between completed proposals`
   assertion site; determine expected-vs-actual for the injected unknown
   probe; fix in the qualification/engine source (never in the
   runner/bwrap/closure semantics); no product defect established
   (hardware-abort-probe gated, qualification-only).
2. **G parity row-upload fixture contract (g-resident)** — child exit 0
   with all markers; the runner's semantic check failed
   `fixture_semantic.py:555 require(up == len(call['input_ids']))`. Root
   cause: fixture contract bug — the capture cells repeat the 'One' token,
   and the resident PLE row page legitimately dedups already-resident rows
   (the binary's own `uploads+reuses==rows` contract passed). Fix design:
   deterministic expected up/hit computed from the cells in execution order
   against the 64-row dedup cache (STRONGER, not weaker), plus the vulkan
   fixture in lockstep. No engine change.
3. **V validator/producer field mismatch (v-resident)** — the physical
   case PASSED (child exit 0, PASS_EXACT_DIAGNOSTIC_MATRIX); the adapter
   validator rejected the report: controller.py FALSE_PERMITS/SCHEMAS
   require `trained_native_natural_serving_or_performance_qualified` which
   the producer never emits. Fix: add
   `"trained_native_natural_serving_or_performance_qualified": false` to the
   complete-qualification report producer at
   `crates/zerogravity/examples/gemma4_vulkan_matrix/run.rs:669` (the
   unknown-schema producer at :398 already matches its schema). 8 of 10 V
   modes affected; the two unknown modes are fine.

**Pass-4 procedure after fixes** (established, repeat of pass-3): commit
fixes → H14 manifest regen if any of the 14 H14 files changed
(`resume-all-h14-probe-typing-fix-20261025/` is the current accepted one) →
copy `freeze-prepare.py` to `local-prepare-final-6` (edit UH/UG to fresh
names + SPECS base for any consumed remote base; HIP base is now `-3`
(consumed), G/V `-2` (G consumed by pass-3 g-admin/bootstrap/build; V
consumed by v-admin/bootstrap/build — bump BOTH to `-3`-style fresh names
on the next pass, update the controllers' BASE constants + host-phase
H_BASE/G_BASE + the adapter copy's GB/VB/VS/BASE + re-pin V_BYTES/V_SHA +
launcher + REVIEWED-SCRIPT-PIN) → update host-phase.py pins (ROOT M +
H/G control/closure from the fresh PREPARED payloads) → resume the HGV
executor pattern (a fresh glm-5.3 xhigh agent with the pass-3 prompt
template, resume order: full serial from h-admin). Pass-3's stopped cases
resume in released order; failed cases are already preserved (their
physical evidence is real; only their controller/validator verdicts were
blocked).

## 3. Binding user decisions (unchanged + this session's)

1. **UD1 (this session)**: Qwen27B natural reference capture runs on
   **pvl-x2**, and the ~55.6GB image+weights establishment downloads on x2
   are AUTHORIZED. Route design + UD2–UD5 resolutions:
   `.agents/runs/resume-all-takeover-qwen27b-capture-glm-20261025/ROUTE-DESIGN.md`;
   the x2-adapted lane agent was launched and paused at wind-down (§4.2).
2. Benchmark ladder: no C2; fresh matched C16 both models on the checkpoint
   tree (B4 physical GPU round FIRST), then C32/C64 after acceptance;
   same-attempt source/profile/stock arms, active8/B4, honest reporting,
   22.13% is a dispatch ceiling never a speedup claim.
3. Convergence directive: one main checkpoint, no broken state merges;
   gemma/qwen merge without waiting for all optimizations; main lane
   (HGV) has priority for agent slots.
4. Wind-down (this): pause at the next reasonable point, hand off to the
   next agent with comprehensive notes.
5. All earlier decisions in the morning handoff §6 remain binding.

## 4. Paused/running lanes at handoff

1. **B4 physical GPU round + fresh matched C16** — agent
   `3d311ad1-d15f-405` (glm-5.3 xhigh) was stopped at wind-down in early
   Phase 1 (builds). Resume instructions: the full original prompt is in the
   session transcript; equivalently relaunch with the same brief — Phase 1:
   build the gpu test binary (jobs1), run the B4 physical matrices
   (enumerate `-- --list`, each ALONE fresh process, `--ignored` where
   marked) + N1 suite + `actual_grouped_state_transport_oracle`, using the
   approved source/bundle pins from
   `.agents/runs/resume-all-cuda8-affected-execution-pi-20261025/CUDA_RUNTIME_SCOPE.md`
   "Original input authorities"; Phase 2: fresh matched C16 per the accepted
   protocol (`.agents/runs/resume-all-c16-executor-pi-20261024/` TODO/PLAN/
   COMMANDS + the journal-fix accepted runner/profile pins; new run dir, new
   release-binary pin from the checkpoint tree, old runs untouched; quiet
   window). Local RTX3060 is free. Log dir:
   `.agents/runs/resume-all-takeover-b4-gpu-c16-glm-20261025/`.
2. **Qwen27B x2 capture** — agent `251d5839-6bb8-466` yielded its slot
   mid-preflight (progress/resume point in
   `.agents/runs/resume-all-takeover-qwen27b-x2-capture-glm-20261025/{TODO.md,LOG.md}`).
   NO x2 access, download, store write, container or GPU work has run yet —
   all work so far was read-only record review on pvl-l5. Gathered:
   (a) UD6 evidence: two foreign `ContainerExitCode` rows in the l5 store db
   (exit 0 @ 2026-10-24 18:54:17; exit 137 SIGKILL @ 18:54:51; no matching
   run dir; OOM/timeout-compatible, not yet interpretable); (b) the base
   image digest IS recorded — `docker.io/vllm/vllm-openai@sha256:b18abb2df97b…21e66ac12f`
   — plus the derived capture image identity (f57d77f5…/eec72f47…, torchcodec
   CPU wheel `de8e58e5…`, recipe `fac9b745…`) and all 31 weight pins in
   source-pin.json, so "no substitute image" is resolved; (c) OPEN DECISION
   at resume: establish the derived image on x2 by rebuild (config ID embeds
   a timestamp — identity determinism unproven) vs l5 `podman save` → x2
   `podman load` (preserves the content-addressed identity exactly) — pick
   the verifiable route and record it truthfully. Resume order is the 6-step
   list in that TODO.md (x2 strict-SSH preflight is network/disk-only while
   any HGV physical work runs; GPU capture only in a confirmed GPU-free
   window; own namespace `qwen27b-natural-capture-20261025-1`; never touch
   the qualification bases `resume-all-five-{hip,gemma4,vulkan}-20261025-*`).
3. HGV executor agent (`0767e210-aa87-412`) completed pass 3 and reported;
   reuse its prompt pattern for pass 4.
4. Gemma4 PLE regeneration (post-merge): the corrective review
   (`.agents/runs/resume-all-takeover-gemma4-ple-review-glm-20261025/`)
   verdict: regenerate from main's committed `tools/gemma4-hip-qualification/`
   (donor + isolated3 controllers unsalvageable); the PLE learned coverage +
   physical 8 cases remain open.
5. Deferred: forced cancellation with timeout + stuck-loop detection (after
   benchmarks), NPU integration, DeepSeek + other families, CLI serve
   pre-Client wiring, CUDA SDK include-view rebuild (if ever approved).

## 5. Successor's task queue (ordered)

1. Fix the three §2 bugs (one batch: `gemma_text/final_owner.rs` boundary
   analysis, g fixture deterministic contract, v producer false-permit field)
   + lockstep control fixtures; commit as logical signed units.
2. HGV pass 4 per §2's procedure; goal: all 24 cases green with full
   inspection (currently 2 PASS + 3 real failures + 19 stopped).
3. B4 physical GPU round → fresh matched C16 (§4.1) → C32/C64 after
   acceptance.
4. Qwen27B x2 capture (§4.2) → numerical acceptance for Qwen27B.
5. Gemma4 PLE regeneration + physical 8 cases; then Gemma4/Qwen27B
   benchmarks on x2 like 0.8B/2B.
6. Then the deferred list.

## 6. Standing constraints (unchanged)

One Rust mathematical engine; immutable source identity; independent
StateRoot ownership; physical ResourceArbiter; fair workers; real
completion/quarantine; GPU/model changes need real source/state/cancellation/
physical tests; HGV bounds and the strict-SSH discipline stay as in the HGV
run TODO; full `scripts/check.sh` gate before accepting any new batch;
fresh-byte merges (stale mtimes can make Cargo reuse a stale library);
`.agents/runs` is gitignored and 433GB — bounded searches only.