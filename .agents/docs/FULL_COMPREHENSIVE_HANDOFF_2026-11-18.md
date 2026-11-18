# ZeroGravity full comprehensive handoff — 18 Nov 2026

Canonical single-file takeover pack for /home/pvl/spaces/abird/src/zerogravity.

This document is intentionally exhaustive. The first sections are the current authority for takeover on 18 Nov 2026. Appendix A embeds the live CURRENT_HANDOFF snapshot verbatim. Appendix B embeds the full 15 Nov historical pack verbatim, which itself carries the complete 14 Nov and October design foundation. If an embedded older checkpoint conflicts with the current front section, the current front section, current source/tests and live TODOs win.

This pack does not revive the retired monolithic SHA/archive/evidence process. It is a human handoff and navigation document. Current source/tests, .agents/plans/TODO.md, per-plan TODOs and Git history remain maintained authorities.

## 0. Executive takeover summary

- Source-authoring host/connector: dotlink---l5 only.
- Repository: /home/pvl/spaces/abird/src/zerogravity.
- Persistent worktree: the single main worktree only.
- AMD large-VRAM execution host: dotlink---x2 only for physical execution/testing. Synchronize exact L5 commits by Git bundle; do not push merely for transport and do not author a divergent source branch on X2.
- Current signed HEAD: 2e0f5f8cd35e1e2896bb5b87fdae77915e566907.
- Pushed origin/main: 3a1596db12d0f2e5d104ed1980f57af9ae27a513.
- Creating this handoff may add one newer docs-only signed commit; trust live Git if HEAD is newer than the generation checkpoint above.
- Local main is intentionally ahead of upstream. Do not push unless the user explicitly asks.
- Commits must remain signed.
- Root README.md is user-authored; do not touch it without explicit request.
- Use CARGO_BUILD_JOBS=8.
- Every Cargo build/check/test/clippy command in this continuation must use RUSTC_WRAPPER=/run/current-system/sw/bin/kache.
- Physical GPU work must run through python3 -B scripts/native-gpu.py -- ...
- Remove every _tmp artifact/include view when the checkpoint finishes.
- Keep each implementation turn to roughly 30 minutes; stop at the next coherent, documented, signed checkpoint.
- Current RTX3060 baseline at the latest revalidation is approximately 37 MiB, 0% utilization, P8, with no live ZeroGravity/native-gpu process.
- Both approved managed original-HF source snapshots are present in the real L5 ZeroGravity cache:
  - Qwen3.5-0.8B by-name qwen3.5-0.8b-qwen-hf-2fc06364 -> snapshot d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c.
  - Qwen3.5-2B by-name qwen3.5-2b-qwen-hf-15852e8c -> snapshot 88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52.

Current P12-C state:

1. Fair aged decode/prefill interleaving is accepted in the shared engine.
2. Private graph lifecycle/cache/selector is retained, but public graph serving stays Native because B2→B8 gains shrink and do not justify the serving complexity.
3. Reconstructed cold prefix scheduling is accepted for its recorded scope.
4. Temporal B2/B4/B8 prefill is physically and production accepted on original 0.8B/2B under the recorded debug-profile scope; canonical release packaging/performance remains a separate open cell.
5. Paged-state logical COW, finite resident physical ownership, materialization, typed KV gather/scatter, canonical paged domains, one-staging in-place ordinary CUDA, hybrid Ready/snapshot semantics, physical hybrid pools, current-root-aware candidate execution and explicit low-level paged source policy are accepted.
6. Source-real paged B1 acceptance is complete for both original 0.8B and 2B: full-vocabulary logits and all42 state are exact at page boundaries/multi-page, and cold lifecycle cancel/abandon/restore/shutdown semantics pass.
7. scripts/native-gpu.py harness hardening is accepted at 6b920a0: it reuses an existing Bubblewrap PID/user sandbox instead of nesting a user namespace, uses an existing private netns when available, otherwise installs fail-closed socket/io_uring seccomp, and keeps the original user+net unshare path outside Bubblewrap.
8. Harness revalidation on current bytes passes 29/29 offline tests, AF_UNIX allow, AF_INET/io_uring denial, exact direct exit-code propagation, direct TERM signal propagation as 143, private-network fail-closed refusal, and a silent physical smoke.
9. Dotlink v0.17.4 forensics corrected an earlier interpretation: dotlink logs every stop it initiates; the relevant window had only the explicit user-requested cancel and a silent long sleep survived until cancellation. Heartbeat output is not a correctness requirement and earlier SIGKILL observations must not be described as a dotlink quiet-job watchdog. Accepted exit-0 model cells remain valid.

Next executable P12-C slice: true in-flight CUDA cancellation on the ordinary paged source path. The already accepted cancel-after-Ready path is not in-flight and does not satisfy this gate. Require a real launch-boundary observation, no partial publication, correct Ready/root authority, and unknown-completion/quarantine semantics. Do not jump to high-level paged serving, native B2/B4/B8 paged transport, complete-domain prefix reuse or event/lease physical overlap until this gate is accepted.

## 1. Mandatory read order for a new agent

Read this file first, then use it as a map into live authority. Before editing, read/revalidate in this order:

1. AGENTS.md
2. .agents/plans/README.md
3. .agents/docs/CURRENT_HANDOFF.md
4. .agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-18.md
5. .agents/plans/TODO.md
6. .agents/plans/model-execution/TODO.md
7. .agents/plans/numerics-and-context/CORRECTNESS_POLICY.md
8. .agents/plans/serving-and-speculation/TODO.md
9. .agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md
10. .agents/plans/memory-and-streaming/TODO.md
11. Any owning docs/plans linked from those files.
12. Current source/tests for the exact surface being changed.
13. Git history for rejected/superseded experiments when a removed path matters.

Current source/tests and live TODOs outrank prose in this handoff if main advances.

## 2. Non-negotiable user direction / working contract

- Implementation-first, shared-primitives-first work.
- Solve reusable semantics in shared layers only when they genuinely benefit more than one model/backend/tool.
- Keep Qwen/source policy in the source/model layer.
- Exact numerical/state/root/resource/cancellation semantics are never weakened for performance.
- Unsupported capability fails closed.
- Retain independently useful model-neutral primitives even if a first serving consumer is rejected.
- Do not resurrect rejected consumers without materially different architecture or evidence.
- Preserve accepted ordinary CUDA and HIP paths.
- Preserve public graph serving disabled.
- Preserve atomic Root/Ready/cancellation/resource authority.
- No paged-state overlap before event/lease, cancellation/unknown-completion and independent-root gates.
- One persistent worktree.
- Signed commits.
- Preserve unrelated dirt/staged work.
- Inspect git diff --cached --stat before every commit.
- No push unless explicitly requested.
- Use kache for Cargo: RUSTC_WRAPPER=/run/current-system/sw/bin/kache.
- Use CARGO_BUILD_JOBS=8.
- Physical GPU only through scripts/native-gpu.py.
- Remove _tmp.
- Roughly 30-minute coherent implementation turns and pause at the next clean checkpoint.

## 3. Enduring ZeroGravity mission / architecture

ZeroGravity is the AI execution fabric/runtime, not a single-model inference binary. Enduring architecture:

- source-verified model loading with explicit source/numerical contracts;
- typed execution IR and model-neutral primitives;
- CUDA, HIP/ROCm and Vulkan backends;
- OpenAI-compatible serving/router;
- continuous batching with source/phase/frontier ownership;
- exact resident state/root authority;
- finite admission and resource ownership;
- model pipelines, eventually including ASR -> LLM -> TTS and multimodal paths;
- single-process and per-model process deployment;
- SSD/source streaming and explicit cache/acquisition ownership;
- fail-closed device/routing/capability behavior;
- performance work only after correctness and ownership gates.

Shared numerical/layout/state/transport primitives stay model-neutral. Source/model-specific policy stays in source/model layers. Physical resource ownership is explicit. Publication is atomic and root-based. Reusable primitives survive a rejected first consumer.

The complete October→November architectural evolution is preserved verbatim in Appendix B.

## 4. Core correctness / ownership invariants

1. Source identity is authority. Never silently mix source snapshots, descriptors or numerical profiles.
2. Exact arithmetic profiles are explicit. Never replace a source/reference contract with close-enough arithmetic.
3. Ready is not publication. Physical completion may remain private.
4. Root publication is atomic. The accepted sequence remains publish_if_usable -> prepare_publish -> state swap -> commit.
5. Dropping an abandoned candidate cannot change committed root/state.
6. Cancellation phases are distinct: before launch, in flight, after Ready, after publication.
7. Submitted-but-unknown completion fails safe; retain/quarantine until authority is known.
8. Paged COW inheritance is exact: only the exact initialized full page at the same plane/index can be inherited without a new Write.
9. Transient staging is not canonical state authority.
10. Finite admission must cover simultaneous maxima; do not hide an extra state-sized envelope.
11. Never submit inherited full pages as fake Writes.
12. Paged snapshots are typed, bounded, K-then-V ordered and require zero inactive tails.
13. Restore republishes: same source/session and saved frontier/state, but a fresh root and incremented session epoch.
14. A retained primitive/private selector is not production selection.
15. Performance evidence never weakens correctness evidence.

See numerics-and-context/CORRECTNESS_POLICY.md for the independent-check policy.

## 5. Selected / retained / explicitly unselected state

### Selected / accepted

- Ordinary public serving remains Native.
- Temporal prefill B2/B4/B8 selects temporal-4 under its accepted source/frontier policy.
- Fair aged decode/prefill interleaving is in the shared engine.
- Source-real ordinary CUDA remains the correctness oracle for current paged B1 work.
- Paged logical/physical state primitives are accepted behind an explicit low-level paged source policy.
- Original 0.8B and 2B paged B1 source parity/lifecycle are accepted.
- Native-GPU harness isolation contract at 6b920a0 is accepted.

### Retained but not publicly selected

- Private CUDA graph lifecycle/cache/selector.
- Private B8 recurrent/KV/projection primitives listed in model-execution.
- Paged source policy and current-root execution.
- Hybrid pools and COW/gather/scatter adapters.
- Qualification-only source-real paged tests.

### Explicitly not selected / still open

- Public graph serving.
- High-level public paged serving selector.
- Native B2/B4/B8 paged state transport.
- Complete-domain prefix reuse/intersection on paged state.
- Direct paged attention replacing contiguous staging.
- Event/lease-backed physical overlap.
- True in-flight CUDA cancellation on paged ordinary source path.
- Canonical thin-LTO release-profile temporal B2 packaging/performance evidence.
- Genuine equivalent-source vLLM superiority work.
- Any silent public default page_tokens=4; it remains qualification data unless explicitly adopted.

## 6. Signed chronology from the original Nov-15 P12-C handoff

Every line below is an actual signed commit on current main after the original docs handoff checkpoint. Preserve this chronology; it is the exact implementation/documentation ledger.

~~~text
12f784083f4057dcf36fd343e408d5a7ed482589	test(cuda): qualify temporal B4/B8 prefill constituents
80409ac30a12c03f9ce4585068faa44aab791460	docs: checkpoint temporal B4/B8 qualification
330150fe23dd97be1fecac42de39b9058e2140e2	docs: record post-qualification GPU baseline
36253481d5e26d27ffb577187b21c338a211beaf	test(cuda): bound temporal qualifier host state
4d74d7f92bc4dd69b18bf947286dcb6a537755ff	docs: close temporal B4/B8 B1 qualification
eaec62d8a40c6179baa49e62be7a358b4b800128	feat(serving): enable qualified temporal B4 prefill
9347f5b838e389f68b4eb38f6d51b664b844704d	docs: accept temporal B4 production
8e6f14d3defc0aa2e542c26ffb9ccc7c736f1fcb	feat(serving): enable qualified temporal B8 prefill
fc9ba86567522b1a9171715ce520b820d7cad60c	docs: complete temporal serving foundation
0027c282fb28fcf846c141f032ea0e4e4fdee599	feat(state): add paged COW block-table primitive
fc99e9c3354500d1a8f946ad1c19d1f2e0b80d31	docs: checkpoint paged state COW foundation
54340a6d848ff5f19bb19333f550b1c057328881	feat(state): bind paged tables to resident pages
818cbb67a55518e3b44ad5c6393e85c67311b887	docs: accept resident paged ownership
1eb6e0f669f7bf6f31e4a0b7cdbdffa14ef73b7c	feat(state): materialize paged COW appends
2c908aef374d817825acd7cf4909358cfe8ec7ad	docs: accept paged COW materialization
3ac205991086e4d9e599993bc7960010d2f8706a	feat(state): add paged KV gather representation
5e6b391e8b6f417882affb54875439fc0fb19094	docs: accept paged KV gather boundary
dc6a188bad8d30a28d6871f0798ae98c4a6803bd	feat(state): add paged KV staging scatter
db3c155f52d6c2c16927208ff0010be03dd9d447	docs: checkpoint paged KV staging bridge
53336b3d961f2ffc59c0b845f876ff4e8e0033c9	feat(state): admit paged KV session domains
4e5aaade5bd83700f907d4f44e380dbbec20bbd7	docs: checkpoint paged KV domain publication
ef9e4d4c4364df423d4a6afad108a1c5b562d967	feat(cuda): add in-place paged KV staging mode
240e0f23def6d0993c9022a8c43d482cdeda58ff	docs: accept in-place paged KV staging
93468b1f5062804f009f817c6f5dd50bc7d0516e	feat(state): validate hybrid paged root semantics
4bafad794e36cb2c4b9647d8807bfc0f8818b4c4	docs: accept hybrid paged root semantics
cb57e1d8bd4d33b530c4efc771e19450df1ef55e	feat(state): construct hybrid paged state pools
9815c2abc63b81cf7ecbbc74041356d9e8a0f10e	docs: accept physical hybrid state pools
b9b04aafd9d10b0325ec5daac45cb3f6be8fedf0	feat(state): execute hybrid paged candidates
d944eae24cb0686d0d9eb21979f93984bac9c09f	docs: accept hybrid paged candidate execution
1413f729ac5db55131b8379752b8cdf0a7b98263	feat(state): add explicit paged source policy
919538ef25d338decfba1d5db54c5a2c3a5801e8	docs: checkpoint explicit paged source policy
398e4461fc3bd30a84e050951dbdf204b05f0cdd	test(state): qualify paged source lifecycle
3a1596db12d0f2e5d104ed1980f57af9ae27a513	docs: checkpoint paged source lifecycle qualifier
30414571c5998880d99c7f685227b83434e6c03c	test(state): fix paged restore qualification
30b311c1a0ca717a87d53a184b88fad5f29341fc	docs: accept 0.8b paged source parity
e6aac539edcf631199bb51628c7f9e361e65057f	test(state): keep paged lifecycle cold-only
cb35ce220edc1d69fa17d80050992f630dec997a	docs: accept cold 0.8b paged lifecycle
a21ef0fecdc9727bcd1e8f8dbd742975cb3b4a5a	docs: correct dotlink signal attribution
28890d1526538352423ff376f53042ad4636462e	docs: record 2b paged sandbox blocker
816e123c3fb68b2322aa3f23896a936a5d3c358f	docs: record unrestricted 2b qualification route
8b65a8ddfb55e20d948d700fe487ed03e94011fa	docs: accept 2b paged source lifecycle
c6ba1c658e77dee6315c77124ff9d15e8df284cd	docs: correct heartbeat harness attribution
6b920a0a91e534eae86f81e603cf419dea3b207b	fix(harness): reuse existing sandbox isolation
2e0f5f8cd35e1e2896bb5b87fdae77915e566907	docs: checkpoint native gpu harness isolation
~~~

### 6.1 Temporal B4/B8 continuation

Separate per-width qualification was mandatory. B4/B8 constituents were physically checked, qualifier host state was bounded, then B4 and B8 production temporal promotion landed independently. CPU/B1/all42 rollback/publication/greedy lineage remained mandatory. Earlier disappeared attempts remain incomplete, not failures. Debug-profile physical evidence does not close the separate release packaging/performance cell.

### 6.2 Paged logical COW — 0027c28

Introduced immutable logical block-table planning, model-neutral COW append plans and immutable prior roots before physical allocation.

### 6.3 Resident physical pages — 54340a6

Bound logical page tables to finite ResidentPagePool ownership with exact lease/generation identity and finite retirement/reuse.

### 6.4 Physical COW materialization — 1eb6e0f

Added real CUDA partial-tail copy, fresh-target zeroing and exact append-range writes with zero inactive tails and retained unknown-completion safety.

### 6.5 Typed paged KV + bounded gather — 3ac2059

ResidentPagedKvDomain accepts only checked KV requirements, keeps paired K/V frontiers and gathers committed paged K/V into bounded contiguous staging under the original submission.

### 6.6 GPU-only staging/scatter bridge — dc6a188

ResidentPagedKvStaging exposes the unchanged contiguous KV ABI transiently. Appended candidate rows scatter D2D back to private pages; the committed partial tail still COWs from immutable paged prior state. No host readback is required after input.

### 6.7 Canonical paged session domain/root substrate — 53336b3

StateBuffers::PagedKv became a marker while Domain owns actual paged state and exact page leases/generations. Legacy consumers fail closed. publish_domain_swap factors the original atomic publication sequence.

### 6.8 One-staging ordinary CUDA mode — ef9e4d4

Admission analysis rejected full prior + full candidate staging on top of paged committed KV because it creates a third state-sized envelope. Accepted design uses one full-capacity in-place K/V staging pair. KvInPlace skips candidate zeroing/prefix copy because the paged prefix is already gathered.

### 6.9 Hybrid root semantics/admission — 93468b1

Paged snapshots, restore, inherited Ready classification and HybridStateInventoryPlan landed. Only exact full same-plane/index current-root pages can be inherited. The plan proves vector/staging/page maxima under the historical root byte envelope.

### 6.10 Physical hybrid pools — cb57e1d

Materialized finite Vector root pool, complete K/V staging pool and homogeneous page pool from verified-source-owned metadata. Initial hybrid roots are zeroed Vector + empty canonical paged KV.

### 6.11 Current-root-aware hybrid execution — b9b04aa

HybridCandidateBundle owns Vector candidate, staging set, current paged owners and private append plans. One exact-identity access cohort gathers, runs the unchanged ordinary body via KvInPlace, scatters, completes only canonical outputs, then materializes paged candidate Domains after physical completion.

### 6.12 Explicit low-level paged source policy — 1413f72

CudaSourceStatePolicy keeps existing Contiguous loader behavior unchanged and allows explicit Paged policy internally. Paged is B1-only, bounds fail closed before source/device work, and public ModelLoadRequest/default serving remains unchanged.

### 6.13 Source-real lifecycle qualification — 398e446 / 3041457 / e6aac53

Source-real tests cover full-vocab/all42 parity, prelaunch cancel, abandoned Ready, cancel after Ready, boundary commits, restore and shutdown. Restore qualification was corrected to fresh root/epoch publication. Temporary warmup/stage debug paths were removed; lifecycle qualifier remains cold-only.

### 6.14 Source-real original 0.8B

Approved snapshot d7c7bf56..., page4/append4:
- full-vocab/all42 exact at frontiers 3,4,5,9 versus Contiguous;
- cold lifecycle passes prelaunch cancel, abandoned Ready, cancel-after-Ready, exact restores and shutdown.

### 6.15 Source-real original 2B

Approved snapshot 88606494..., page4/append4:
- differential job 38c1f321-0347-4a80-b2dc-c6d9d46d0249, exit0, 703.17s, full-vocab/all42 exact at 3/4/5/9;
- cold lifecycle job 744f36fd-b145-47b9-aa61-084e7d4b414b, exit0, 438.18s, cold_load=true, exact cancellation/abandon/restore/all42/shutdown.

Both approved small sources therefore clear source-real paged B1 state/lifecycle.

### 6.16 Native-GPU harness hardening — 6b920a0 / 2e0f5f8

Problem:
- old harness always attempted nested unshare --user --map-root-user --net;
- nested user namespace creation is not a valid assumption inside dotlink Bubblewrap;
- tmux/setsid/background-heartbeat experiments introduced unnecessary process-supervision ambiguity.

Accepted design:
- detect existing Bubblewrap by uid_map + PID1 bwrap + private PID namespace;
- existing Bubblewrap private netns -> direct exec;
- existing Bubblewrap shared network -> fail-closed seccomp socket-creation filter, AF_UNIX only, with io_uring blocked;
- caller needing loopback/TCP/UDP must use --require-private-network and fail closed on shared-net Bubblewrap;
- outside Bubblewrap retain original user+net unshare;
- direct execvpe removes an extra harness child and preserves direct status/signal propagation;
- --isolation-info reports the plan;
- ZG_NATIVE_GPU_ISOLATION records mode in child;
- scripts/check.sh is explicitly private-network-required.

Current revalidation:
- 29/29 offline harness tests;
- Bubblewrap-seccomp isolation plan;
- AF_UNIX allowed;
- AF_INET denied EPERM;
- io_uring setup denied EPERM;
- exit73 propagates as73;
- TERM propagates as143;
- private-network request refuses before child launch under shared-net Bubblewrap;
- silent15s physical smoke passes.

Dotlink forensic correction:
- deployed v0.17.4 logs every stop it initiates;
- dedicated review found only the explicit user-requested cancellation in the relevant window;
- a silent long sleep remained alive until explicit cancel;
- no accepted claim remains that dotlink kills quiet jobs;
- heartbeat output is optional harness plumbing, not correctness evidence.

## 7. Exact current evidence ledger

### Temporal/fairness

- fair aged decode/prefill scheduling accepted;
- original0.8B/2B temporal B2/B4/B8 debug physical qualification accepted;
- production HTTP temporal4 accepted under recorded B2/B4/B8 policy;
- exact CPU text/usage/publication accounting in recorded cells.

### Paged-state primitives

Accepted:
- logical block-table COW;
- finite resident page pool;
- physical COW append materialization;
- typed paged KV domain;
- bounded paged→contiguous gather;
- D2D staging→paged scatter;
- canonical paged Domain/root representation;
- root publication/restore authority;
- paged snapshots;
- inherited Ready classification;
- hybrid byte-admission plan;
- physical vector/staging/page pools;
- one-staging KvInPlace;
- current-root-aware one-flight hybrid execution;
- explicit low-level paged source policy.

### Source-real paged B1

Both original0.8B and2B:
- full-vocabulary exact versus Contiguous;
- all42 exact;
- page−1/page/page+1/multi-page exact;
- cold lifecycle;
- prelaunch cancel;
- abandoned Ready;
- cancel-after-Ready;
- restore before/at boundary;
- clean shutdown.

### Harness

Accepted:
- nested-sandbox-safe;
- mandatory network isolation fail closed;
- direct status/signal propagation;
- current shared-net Bubblewrap usable for non-loopback physical GPU work via seccomp;
- loopback-dependent gates require a private-net connector profile.

## 8. Rejected paths / do not repeat casually

- Do not infer temporal B4/B8 from B2.
- Do not select public CUDA graphs merely because private graph primitives exist.
- Do not revive graph serving without materially better B4/B8 evidence.
- Do not use separate full prior + full candidate staging on top of paged committed KV.
- Do not submit inherited full pages as fake Writes.
- Do not mutate canonical paged state through staging.
- Do not restore old RootBinding identity; restore republishes.
- Do not make page_tokens=4 a silent public default.
- Do not expose high-level paged serving before cancellation/native-transport gates.
- Do not jump to physical overlap before event/lease + unknown-completion + independent-root acceptance.
- Do not reinterpret interrupted connector jobs as model failures without terminal semantic evidence.
- Do not reintroduce nested user namespace creation inside existing Bubblewrap.
- Do not weaken network isolation to make a harness gate pass.
- Do not use heartbeat output as a correctness requirement.
- Do not recreate retired .agents/runs, monolithic receipt archives or side-worktree coordination.
- Do not push just to sync X2.
- Do not kill/reset/reclaim unrelated X2 GPU owners.

Historical rejected CUDA micro-optimizations and serving consumers remain in Appendix B and model-execution/TODO.md.

## 9. Current next execution order

### Immediate P12-C — true in-flight CUDA cancellation

Goal: cancellation after CUDA work has actually been submitted and before Ready/publication on the ordinary paged source path.

Acceptance must establish:
- observable launch/submission boundary;
- cancellation while physical work is in flight, not before launch and not after Ready;
- no partial committed root publication;
- current root/all42 exact if cancellation wins before publication;
- physical completion/unknown completion follows existing fail-safe rules;
- submitted page/vector/staging resources are not prematurely recycled;
- quarantine/retirement exact;
- subsequent valid work can continue;
- final model/resource shutdown retires cleanly.

After-Ready Discarded is already accepted and does not satisfy this gate.

Use original0.8B first to qualify the mechanism. Widen only after the mechanism is accepted. Use kache for Cargo and native-gpu for physical work.

### After true in-flight cancellation

1. high-level paged serving selector only if explicit source policy can be exposed without silently changing defaults;
2. native B2/B4/B8 state transport on canonical paged state;
3. source-real SDK/HTTP paged parity;
4. complete-domain paged prefix identity/intersection + COW;
5. direct paged attention only if it earns complexity/performance versus staging;
6. event/lease-backed physical overlap after state/cancellation/unknown-completion gates;
7. genuine equivalent-source vLLM comparison;
8. canonical release-profile temporal/paged packaging/performance cells.

## 10. Source/cache identities

Managed cache root:
/home/pvl/.cache/abird/zerogravity/models

Approved refs:
- qwen3.5-0.8b-qwen-hf-2fc06364 -> d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c
- qwen3.5-2b-qwen-hf-15852e8c -> 88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52

Do not substitute local GGUF evidence for source-real HF evidence.

## 11. L5 / X2 operational rules

### L5

- canonical authoring host;
- one main worktree;
- use kache for Cargo;
- CARGO_BUILD_JOBS=8;
- native-gpu for physical GPU;
- current shared-network Bubblewrap uses seccomp network isolation;
- loopback-dependent gates must request --require-private-network and currently refuse on this profile;
- missing local allowed_signers principal warning does not mean the Git signature is bad.

### X2

- execution/testing only for AMD large-VRAM work;
- sync exact L5 commit by Git bundle;
- resample VRAM/KFD baseline every run;
- shared host; unrelated Ollama/model residents may exist;
- never reset/reclaim/restart/kill unrelated owners;
- verify only ZeroGravity incremental resource delta retires.

## 12. Tracking model / canonical documents

Retired: generated monolithic receipts, .agents/runs, per-file hash catalogues and durable side-worktree coordination.

Use:
- CURRENT_HANDOFF.md — live cross-window pointer;
- this dated comprehensive pack — full takeover snapshot;
- global TODO — queue;
- per-plan TODOs — detailed evidence/decisions/open work;
- source/tests — executable authority;
- Git history — provenance/archive.

## 13. Clean restart procedure after a stuck/aborted turn

1. git status --short --branch
2. git rev-parse HEAD and origin/main
3. git log -1 --show-signature
4. git worktree list --porcelain
5. inspect dirty/staged paths and preserve unrelated work
6. ps for Cargo/rustc/model/native-gpu/bwrap
7. physical GPU baseline through native-gpu
8. inspect _tmp and disk
9. read CURRENT_HANDOFF, global TODO and owning plan
10. if a durable job exists, use job_get/job_output; never blindly relaunch it
11. classify interrupted runs as incomplete unless terminal semantic evidence exists
12. use kache for every Cargo invocation
13. physical GPU only through native-gpu
14. sign coherent commits and update owning docs before pausing

## 14. Next-agent copy/paste starting prompt

~~~text
Use dotlink---l5 as the only source-authoring connector. Continue in
/home/pvl/spaces/abird/src/zerogravity on the single main worktree. Use
dotlink---x2 only for AMD large-VRAM execution/testing; synchronize exact L5
commits by Git bundle rather than pushing merely for transport.

Read first, in order:

1. AGENTS.md
2. .agents/plans/README.md
3. .agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-18.md
4. .agents/docs/CURRENT_HANDOFF.md
5. .agents/plans/TODO.md
6. .agents/plans/model-execution/TODO.md
7. .agents/plans/numerics-and-context/CORRECTNESS_POLICY.md
8. .agents/plans/serving-and-speculation/TODO.md
9. .agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md
10. .agents/plans/memory-and-streaming/TODO.md
11. any owning docs/plans linked from those files.

Revalidate Git status, HEAD, origin/main, signature, worktrees, live processes,
physical GPU state, cache/source identities, disk space and _tmp before editing.
Preserve unrelated work. Do not create another persistent worktree or branch.
Commits must remain signed. Do not push unless explicitly requested.

Current signed HEAD is 2e0f5f8cd35e1e2896bb5b87fdae77915e566907. Pushed origin/main is 3a1596db12d0f2e5d104ed1980f57af9ae27a513. Current production
ordinary serving remains Native. Public graph serving remains disabled. Both
approved original Qwen3.5-0.8B and2B pass source-real paged B1
full-vocabulary/all42 boundary parity and cold lifecycle semantics at
page_tokens=4/max_append_tokens=4. Low-level paged source policy, hybrid finite
pools, COW/gather/scatter, canonical paged domains, one-staging KvInPlace,
Ready/root/snapshot/restore semantics and current-root candidate execution are
accepted. High-level paged serving is still not selected.

The native GPU harness is accepted at 6b920a0. It reuses an existing Bubblewrap
sandbox, uses private Bubblewrap netns when available, otherwise applies
fail-closed socket/io_uring seccomp, and falls back to user+net unshare only
outside Bubblewrap. Do not reintroduce nested user namespace assumptions.
Commands needing loopback must use --require-private-network and fail closed on
the current shared-net L5 profile. Heartbeat output is optional harness plumbing,
not a correctness requirement.

Use CARGO_BUILD_JOBS=8. Set
RUSTC_WRAPPER=/run/current-system/sw/bin/kache for every Cargo invocation.
Run physical GPU work only through python3 -B scripts/native-gpu.py -- ...
Remove all _tmp artifacts.

Continue with the next coherent P12-C slice: true in-flight CUDA cancellation on
the ordinary paged source path. After-Ready cancellation is already accepted and
does not satisfy this gate. Establish a real submitted/in-flight boundary,
preserve exact Root/Ready/cancellation/resource authority, no partial
publication, fail-safe unknown-completion/quarantine, clean continuation and
retirement. Do not jump to high-level paged serving, native B2/B4/B8 paged
transport, complete-domain prefix reuse or physical overlap before this gate is
accepted.

Keep this implementation turn to roughly30 minutes. Aim for the next coherent
checkpoint, update the owning TODO/handoff state, sign the commit, sync X2 only if
needed for physical AMD work, then pause. Do not push unless explicitly requested.
~~~

## 15. Appendix A — embedded live CURRENT_HANDOFF snapshot

The following is a verbatim snapshot of CURRENT_HANDOFF.md at generation time.
Its current-status sentences are superseded by the front sections above if they
differ, but it preserves every detailed accepted/rejected/open item in the live
handoff.

---

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
  `6b920a0a91e534eae86f81e603cf419dea3b207b`
  (`fix(harness): reuse existing sandbox isolation`). The paged-state lineage
  remains `e6aac53` -> `3041457` -> `398e446`: `e6aac53` keeps the
  lifecycle qualifier cold-only, `3041457` corrected restore qualification,
  and `398e446`
  added the source-real lifecycle target; `1413f72` is the
  runtime/source-policy checkpoint; `b9b04aa`
  remains the accepted current-root gather/KvInPlace/scatter/Ready candidate
  execution; `cb57e1d` remains the physical pool substrate and `93468b1`
  remains snapshot/restore/inherited-Ready authority.
  `1413f72` adds `CudaSourceStatePolicy::{Contiguous,Paged}` at the original
  source-call loader seam. Existing `load_ordinary_cuda_source` remains an exact
  Contiguous wrapper; the explicit policy-bearing loader installs hybrid
  Vector/staging/page pools only for `Paged { page_tokens,
  max_append_tokens }`. Paged policy is B1-only for now: native_rows !=0 and
  zero policy extents fail closed. `HybridStateInventoryPlan` also rejects
  max_append_tokens above state capacity, and runtime candidate construction
  returns `Limit` before checkout if a call exceeds the admitted append bound.
  The high-level/public `ModelLoadRequest` is intentionally unchanged:
  production registration passes `CudaSourceStatePolicy::Contiguous`
  explicitly, so no serving capability/default switched in this checkpoint.
  A pre-source-IO unit test proves invalid paged/native/bound policies return
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
  select temporal4 while decode remains unchanged.
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
  temporal4 for **B2, B4 and B8**, with equal-frontier prefill and explicit
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
  Outside Bubblewrap it retains the original
  `unshare --user --map-root-user --net` route.
  `--isolation-info` reports the exact planned mode and
  `ZG_NATIVE_GPU_ISOLATION` records it in the child. Commands requiring TCP/UDP
  loopback must pass `--require-private-network`; the shared-network seccomp
  fallback refuses them rather than weakening isolation. `scripts/check.sh`
  now fails immediately in that fallback, and `AGENTS.md` uses
  `native-gpu.py --require-private-network -- bash scripts/check.sh`.
  Focused validation:29/29 offline harness tests, Python compile + Bash syntax,
  real current-sandbox `--isolation-info`, AF_UNIX allowed/AF_INET denied,
  io_uring_setup denied with EPERM, private-network refusal before child launch,
  and a silent 15s direct-exec smoke all pass. The complete `scripts/check.sh`
  gate is not runnable under this particular L5 connector profile because its
  Bubblewrap shares networking; that is now an explicit execution-profile
  refusal, not a harness ambiguity or semantic test failure.
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
  **Next executable P12-C slice is true in-flight CUDA cancellation** on the
  ordinary source path. Keep the model child foreground and require an actual
  launch-boundary proof; periodic heartbeat output is optional harness plumbing,
  not part of the correctness contract. Run ordinary model/GPU physical cells
  through the accepted `native-gpu` isolation plan; reserve
  `--require-private-network` for gates that genuinely need TCP/UDP loopback.
  After-Ready cancellation is already accepted but is not a substitute. Only
  after that gate should high-level paged serving/native B2/B4/B8/HTTP work
  begin.
  Complete-domain prefix reuse follows these state gates; event/lease-backed physical
  overlap remains after those state gates and must not be pulled forward.
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
- Pushed `origin/main` is `3a1596db12d0f2e5d104ed1980f57af9ae27a513`
  (`docs: checkpoint paged source lifecycle qualifier`). Current
  `3041457` plus this acceptance refresh remain local until explicitly pushed.
  Never use push merely for X2 transport.
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

---

## 16. Appendix B — embedded full 15 Nov historical pack

The following is a verbatim copy of FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md.
It contains the complete 14 Nov baseline and October→November design foundation.
Treat its checkpoint and next instructions as historical; the current front
sections above win.

---

# ZeroGravity full comprehensive handoff — 15 Nov 2026

Canonical full takeover pack for `/home/pvl/spaces/abird/src/zerogravity`.

This file exists because the user explicitly wants one document a new agent can
read to recover the **full project context from the beginning**, including design
lineage, implementation phases, accepted/rejected experiments, exact evidence,
current source state, pending work, host/tool rules and the next executable step.

It is intentionally comprehensive. It is **not** a replacement for live source,
tests or TODOs:

1. current source/tests;
2. `AGENTS.md`;
3. `.agents/plans/README.md`;
4. `.agents/docs/CURRENT_HANDOFF.md`;
5. `.agents/plans/TODO.md`;
6. the owning plan TODO/spec;
7. this handoff for complete chronology/rationale;
8. Git history for superseded source and removed experiments.

If anything here conflicts with current source/tests/live TODOs, those live
authorities win. Git remains provenance. Do **not** resurrect the retired
generated-SHA/archive/evidence workflow.

---

## 1. Exact takeover checkpoint

### Repository and Git

- Source-authoring repository/host: `dotlink---l5`.
- Canonical path: `/home/pvl/spaces/abird/src/zerogravity`.
- Persistent branch/check-out: `main` only.
- No additional persistent worktree or side branch.
- Current signed local implementation tip at handoff preparation:
  `1e0f4ebfc5b9650341814ad669d4101aea717623`
  — `feat(serving): enable qualified temporal B2 prefill`.
- Immediate parents/new local checkpoints:
  - `9dab0043719e435ef3914f86f3966079deabc5ae`
    — `feat(cuda): add temporal fused prefill primitive`.
  - `ba7e3768197c417e42eecd7315f6043f1d5e9464`
    — `docs: accept fair serving interleaving`.
  - `7dea460a80b420d0ce49a60f6c860b3502c67c38`
    — `feat(serving): age prefill behind decode`.
- `origin/main` at handoff preparation:
  `7dea460a80b420d0ce49a60f6c860b3502c67c38`.
- Local `main` is therefore ahead by three commits before this handoff-doc commit.
- **Do not push unless the user explicitly asks.**
- Commits are signed. `git log --show-signature` reports the good ED25519 key
  `SHA256:4aB+NvRb1shCx4xjtZDr2Y7FzfHGph80eFtpF+pHuCI`; the missing local
  `~/.config/git/allowed_signers` principal warning is not a bad signature.
- Root `README.md` is user-authored and protected; do not modify unless explicitly
  requested.
- Before every commit inspect `git diff --cached --stat` and stage only owned
  paths.

### Current L5 physical state

At the latest accepted B2 temporal checkpoint:

- RTX 3060 Laptop GPU: about `41 MiB / 6144 MiB`, utilization `0%` after cleanup.
- No owned Cargo/rustc/ZeroGravity/native-gpu process remains.
- No temporal `_tmp` SDK/client/bundle artifacts remain.
- One worktree only.

Always resample before using this state; it is not a reservation.

### X2 execution host

- Use `dotlink---x2` only for AMD large-VRAM execution/testing.
- Same repository path:
  `/home/pvl/spaces/abird/src/zerogravity`.
- Source changes originate on L5 and are transported to X2 by bounded Git bundle;
  do not push merely to synchronize test bytes.
- X2 is synced to exact current implementation tip `1e0f4eb...`.
- Latest observed AMD baseline:
  - total VRAM `103079215104` bytes;
  - used about `272351232` bytes;
  - busy `0`;
  - no KFD owner observed at that checkpoint.
- X2 is a **shared host**. Older runs observed unrelated Ollama/model residents
  near 31 GiB and a host-global KFD owner hidden outside the Bubblewrap PID
  namespace. Never attribute or kill/reset/reclaim another owner's memory/process
  without explicit authorization.
- Record baseline before each run and verify only our incremental delta retires.

---

## 2. User direction and working contract

The user has repeatedly established these project-level preferences:

- Work implementation-first and correctness-first.
- Prefer clean, shared, model-neutral primitives where semantics genuinely
  generalize.
- Keep model/source-specific policy in model/source layers.
- Preserve reusable primitives even when a first serving consumer is rejected.
- Remove the slow/rejected consumer, not a clean shared primitive.
- Prefer Rust ownership and simple robust APIs; use zero-cost abstractions where
  natural.
- Prefer easy, maintainable hot-path performance; do not trade clarity and
  correctness for convoluted micro-optimization.
- Validate at coherent checkpoints, not after every helper.
- Do not repeatedly requalify unchanged primitives.
- Fewer launches, lower instruction counts, smaller copies or a microbenchmark are
  hypotheses, not acceptance.
- Numerical/source/state/Root/resource/cancellation semantics are never weakened
  to chase performance.
- Unsupported capability fails closed.
- Keep each implementation user/model turn to roughly **30 minutes**; aim for the
  next coherent checkpoint, record state, then pause.
- If a tool call/process disappears, revalidate exact source/process/GPU state and
  continue from the last established boundary; do not blindly relaunch large work.
- No unnecessary clarifying questions when the next step is already authorized.
- No push unless explicitly asked.

The user explicitly asked for this full handoff so future agents can start from
one prompt rather than reconstructing the history.

---

## 3. Mandatory read order for the next agent

Read this sequence before editing:

1. `AGENTS.md`.
2. `.agents/plans/README.md`.
3. `.agents/docs/CURRENT_HANDOFF.md`.
4. `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md` — this file.
5. `.agents/plans/TODO.md`.
6. `.agents/plans/model-execution/TODO.md`.
7. `.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md`.
8. Current owning plans, especially:
   - `.agents/plans/serving-and-speculation/TODO.md`
   - `.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md`
   - `.agents/plans/cpu-gpu-optimization/TODO.md`
   - `.agents/plans/memory-and-streaming/TODO.md`
   - `.agents/plans/numerics-and-context/TODO.md`
   - `.agents/plans/pipeline-composition/TODO.md`
   - `.agents/plans/cli-device-routing/TODO.md`
   - `.agents/plans/model-acquisition/TODO.md`
9. Relevant architecture docs under `.agents/docs/`.
10. Git history for removed/rejected implementations.

Do not read all architecture docs for every kernel tweak. Use the navigation map
later in this file and the owning plan.

---

## 4. Tracking model — do not recreate retired process

The repository deliberately retired monolithic generated handoff/SHA/archive/run
ledgers as a mandatory workflow.

Current tracking authority:

- `.agents/plans/TODO.md` — global queue.
- Per-plan `TODO.md` — accepted/rejected/open work.
- `CURRENT_HANDOFF.md` — concise cross-window pointer.
- This file — full human/agent onboarding context.
- Git history — provenance/archive.
- Current source/tests — implementation authority.

Do not recreate:

- `.agents/runs` as a durable evidence ledger;
- per-file hash catalogues for normal development;
- source snapshot archives;
- mandatory run receipts;
- durable side worktrees/branches as agent coordination state;
- generated handoffs after every minor commit.

Actual source/model/content hashes, numerical acceptance limits and physical
identities remain required where they are semantically part of correctness.

---

## 5. ZeroGravity mission and enduring architecture

ZeroGravity is an **AI execution fabric**, not a collection of per-model servers.

Product/runtime direction:

- one Rust-owned execution system;
- OpenAI-compatible serving/router;
- continuous/fused batching;
- CPU/CUDA/ROCm/Vulkan backend families;
- immutable source identity and authenticated source windows;
- SSD-streamed inference and tiered residency;
- multi-model/multimodal pipelines in one process where useful;
- YAML/configurable composition;
- ASR → LLM → TTS and other task programs;
- eventual speculation/MTP/draft/n-gram/Flash-Next variants under explicit source
  and correctness contracts;
- single-process and per-model process modes as product choices, not separate
  semantic engines.

Current crate architecture after the completed Fabric transition:

- `zerogravity-core`
  - semantic/state/provider/physical contracts;
  - StateRoot authority;
  - task/program abstractions;
  - resource contracts.
- `zerogravity-artifacts`
  - immutable formats;
  - source/cache/acquisition;
  - snapshot/artifact ownership.
- model crates, notably `zerogravity-model-qwen35`
  - model-family semantics;
  - source binding;
  - CPU mathematical/reference behavior.
- `zerogravity-cpu`
  - CPU physical execution.
- `zerogravity-gpu`
  - resident GPU execution;
  - shared CUDA/HIP/Vulkan physical primitives;
  - exact GPU state/lease/event/quarantine ownership.
- `zerogravity-engine`
  - model-neutral sessions/router/workers;
  - scheduling/fairness/admission;
  - cohort orchestration.
- `zerogravity-serving`
  - reusable protocol/HTTP behavior.
- `zerogravity`
  - public composition root/facade.
- `zerogravity-cli`
  - CLI/config/presentation/signals.

Do not restart the closed Fabric S/F/Q migration as if it were current work.
Historical Fabric rationale is retained in the embedded Nov 14 baseline and under
`.agents/plans/fabric-vnext*`.

---

## 6. Non-negotiable correctness / ownership invariants

These are the core rules behind almost every accepted/rejected decision:

- One semantic/model definition per family/source contract.
- One logical `StateRoot` publication authority.
- Canonical roots remain checkpoint/rollback/publication authority even when
  private/native/derived state exists.
- One process-shared physical `ResourceArbiter`.
- Device workers may own asynchronous compute/events but cannot invent logical
  commits.
- Source and weight owners survive until the actual final consumer retires.
- Release follows actual completion/fence/event state.
- Completion-unknown work is fenced/quarantined; it is never silently freed and
  retried.
- Cancellation cannot publish a partial candidate.
- Independent sessions remain independent through physical batching.
- Physical batch width and model/source numerical reduction profile are separate
  contracts.
- Finite captured cases are qualification evidence, not production selectors.
- A metadata/registry slug never grants implementation support.
- CPU/CUDA/HIP/Vulkan and source/quant profiles are separate qualification claims.
- Exact source identities, reductions, state domains, positions, masking, dtype,
  ownership and final-reader semantics are preserved.
- Performance work follows correctness and source/state/ownership acceptance.

Correctness policy:

- pinned vLLM practice is used for model-output/token comparisons;
- pinned llama.cpp operator practice is used for independently declared
  per-operator numerical contracts;
- quality/NLL/KL/PPL/top-token evidence is profile-specific;
- no blanket epsilon turns a failed exact source contract into a pass;
- interrupted/failed historical runs remain failed/incomplete;
- one backend/model/profile never automatically qualifies another.

See `.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md`.

---

## 7. Operational execution rules

- `CARGO_BUILD_JOBS=8` by default.
- `RAYON_NUM_THREADS=4` for broad gates where specified.
- Reduce build jobs only for observed resource pressure.
- Physical GPU execution only through:
  `python3 -B scripts/native-gpu.py -- ...`.
- Use `python3 -B` / `PYTHONDONTWRITEBYTECODE=1` for agent-side Python.
- For serving tests inside `native-gpu`, bring up loopback inside the isolated
  namespace when required.
- Long builds/runs should use durable jobs with explicit heartbeat output.
- A disappearing outer job with no compiler/test result is **incomplete**, not
  failed and not passed.
- Inspect actual semantic/test output before accepting.
- Remove temporary `_tmp` runners, selector hooks, SDK include views and bundles
  after the decision.
- `scripts/check.sh` is the final broad assembled gate; use focused tests during
  implementation.
- Do not repeat expensive unchanged gates unless a relevant boundary changed.
- Do not download models/SDKs or change device privileges/recovery unless already
  authorized for that exact scope.
- Local Renoir/gfx90c AMD launch hold remains historical/current where applicable.
- Keep the initiating agent supervising its durable job until result retrieval.

---

## 8. Current selected ordinary CUDA / serving path

The current public serving path remains native execution, not CUDA graph replay.

### Selected

- ordinary source-derived CUDA resident execution;
- source-exact packed BF16 kernels and retained shared primitives;
- B1/B2/B4/B8 native decode cohorts where accepted;
- unequal decode frontiers;
- chunked/ragged prefill foundation;
- fair aged prefill/decode scheduling;
- native teacher-prefix scheduling;
- **0.8B B2 temporal prefill quantum 4** after `1e0f4eb`;
- existing HTTP/router/Ready/publication/cancellation/backpressure semantics.

### Explicitly not selected

- public CUDA graph serving;
- unqualified B4/B8 temporal prefill;
- unqualified 2B temporal prefill;
- paged hybrid state;
- physical multi-stream overlap;
- speculative decoding strategies not independently accepted;
- finite diagnostic cases as production selectors.

### Current B2 temporal policy

`NativeText` now does:

- `native_rows == 2`:
  - prefill chunks of 4;
  - equal-frontier prefill (`unequal_prefill=false`);
  - explicit prefill cohort eligibility;
  - pairwise physical compatibility requires the same resident model owner,
    matching hidden width and matching chunk size.
- `native_rows == 4/8`:
  - existing one-token prefill remains;
  - existing unequal-frontier policy remains;
  - no temporal promotion yet.
- `native_rows == 0`:
  - ordinary non-native prefill remains chunk32 as before.

The B2-only restriction was deliberate: widen only after separate physical
qualification.

---

## 9. Post-Nov-14 continuation — exact chronology and decisions

This section is the delta beyond the complete 14 Nov historical handoff embedded
later.

### 9.1 Private production CUDA graph lifecycle/cache — `63ee14e`

A workspace-owned production graph lifecycle/cache was added without enabling
public graph serving.

`ProductionGraphState`:

- `Cold`
- `Capturing`
- `Live { graph, retirement_root }`
- `Spent`

`ProductionGraphCache` invariants:

- one cold capture per live batch workspace;
- exact source identity;
- exact resident service generation;
- exact CubeCL `ServiceId`;
- exact logical CUDA stream identity;
- complete `BatchGraphProfile`;
- exact captured storage/access authority;
- graph resource charge acquired before graph prepare;
- charge may grow only before capture;
- charge survives capture/replay/quarantine;
- release only after trusted retirement;
- graph destruction only after trusted terminal barriers;
- unknown completion fences/quarantines rather than frees;
- same workspace becomes Spent after retirement; recapture requires replacement.

The lifecycle/cache remains private qualification capability.

### 9.2 Production graph selector — `9d6613c`

`FusedExecution::Production` was integrated as a fail-closed private selector.

Behavior:

- exact compatible live cache → Graph;
- cold/spent/no-cache → Native miss;
- device-token/incompatible typed profile → Native miss;
- runtime identity/lifecycle drift → error;
- unresolved selector or graph replay failure never retries Native after launch.

Physical original0.8B B2 qualification proved:

- compatible replay;
- incompatible Logits→StateOnly Native fallback;
- unchanged roots/state;
- retained graph charge;
- explicit retirement;
- fresh-workspace recapture;
- trusted B2→B4 replacement;
- final reserved0 / 41MiB.

Public serving still did not select graph execution.

### 9.3 Matched graph timing — `599cb28`

Same process/model/workspace, full-logit readback, 4 AB/BA warm pairs and 12
samples/path:

- B2:
  - Native mean `273,933us`, median `267,847us`
  - Graph mean `264,881us`, median `263,238us`
  - gain about `3.30% mean / 1.72% median`
- B4:
  - Native `273,801 / 273,910us`
  - Graph `270,766 / 270,805us`
  - gain about `1.11% / 1.13%`
- B8:
  - Native `362,559 / 360,951us`
  - Graph `359,915 / 359,649us`
  - gain about `0.73% / 0.36%`

Exact replay deltas: 12.

Retained policy charges:

- B2 `15,356,224` bytes
- B4 `15,405,376` bytes
- B8 `15,458,624` bytes

Trusted retirement and final reserved0/41MiB passed.

**Decision:** the gain shrinks with useful width and does not justify public
graph-serving lifecycle/resource complexity. Ordinary serving remains
`FusedExecution::Native`. Retain the graph cache/selector for materially
different future graph regions.

Do not resurrect public graph serving based on these measurements alone.

### 9.4 Graph resource/compiler authority retained

Earlier graph substrate remains important:

- `5bf440d` — graph substrate in production native builds.
- `db7be60` — graph resources admitted through the same physical ledger.
- `6a5c610` — nonpersistent NVRTC emits exact-SM cubin/SASS instead of unsupported
  CUDA13.4 PTX for the installed driver; persistent cache path keeps PTX.
- `81b7c71` — explicit graph resource lifetime witness.
- `f1ce532` — replay binds exact `(ResidentLease, DeviceAccess)` authority.
- One transient post-capture resident-byte mismatch was observed once and did not
  reproduce in four strengthened physical reruns. Do not weaken the idle
  invariant; use the lifetime witness if it recurs.
- CUDA driver-internal graph byte/node enumeration remains unqualified.

### 9.5 Reconstructed prefix scheduling — `b19bdb4`, `f4fb04b`

Historical warm evidence remains accepted:

- accepted max32-prefix provider `d0a60844217fc106fe8c70bb3a99073eef0387c8`;
- accepted native teacher-prefix provider
  `f78892957f09a84838246aede4fef2be040a0d18`.

Historical warm reports:

- 0.8B wave `32.301205 → 18.33249s`;
  TTFT p95 `29.6884 → 15.7265s`;
  ITL p95 `0.3851 → 0.3832s`.
- 2B natural C8 wave `42.19858 → 28.457115s`;
  TTFT `35.6887 → 21.9980s`;
  ITL `0.9380 → 0.9312s`.

Original frozen binaries/workload were gone, so the cold campaign rebuilt exact
historical source commits under the current toolchain with the same benchmark-only
exact-SM cubin compatibility behavior.

Approved model hashes retained from the reconstruction:

0.8B:
- GGUF `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`
- tokenizer
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`

2B:
- GGUF `91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6`
- same tokenizer hash.

#### Reconstructed 0.8B cold diagnostic

8 source-real ragged prompts reconstructed from retained `f788929` HTTP tests,
all arrival0, alternating tenant labels, max_tokens8.

ABBA old→new→new→old:

- all `32/32` requests;
- 256 completion tokens exact across builds;
- exact token IDs/output SHA/reported usage/finish.

Means:

- old wave `17.4307123065s`
- new wave `15.4528811080s`
- wave change `-11.35%`
- output rate `3.6720 → 4.1416 tok/s` (`+12.79%`)
- TTFT p95 `15.0367732376 → 14.9527216314s` (~`0.56%` better)
- delivery ITL p95 `0.8188307785 → 0.7804418760s` (~`4.69%` better)

Zero timeout/refusal/failure/cancel/discard; final GPU 41MiB.

Important limitation: dispatch shape differs from the old warm report; this is a
distinct cold scheduler diagnostic, not a byte-identical reproduction.

#### Reconstructed 2B cold diagnostic

Before measurement, staged old-provider physical checks proved:

- C1 pass;
- ragged C2 pass;
- ragged C4 pass with B2;
- equal-frontier C4 forces B4 and passes;
- equal-frontier C8 forces B8 and passes;
- max observed GPU memory ~3261MiB;
- no cgroup OOM;
- final 41MiB.

Measured ABBA old-a/new-a/new-b/old-b:

old-a:
- wave `28.660851224s`
- rate `2.2330 tok/s`
- TTFT p95 `22.550742884s`
- ITL p95 `1.518029680s`
- bands B2/B4/B8 `3/1/6`

new-a:
- `24.662303997s`
- `2.5951 tok/s`
- TTFT `23.527454055s`
- ITL `1.303214931s`
- bands `14/11/17`

new-b:
- `25.525155984s`
- `2.5073 tok/s`
- TTFT `24.258651126s`
- ITL `1.343631052s`

old-b:
- `30.395075144s`
- `2.1056 tok/s`
- TTFT `23.774190073s`
- ITL `1.601505819s`

Means:

- old wave `29.527963184s`
- new wave `25.093729990s`
- wave `-15.02%`
- rate `2.1693 → 2.5512 tok/s` (`+17.60%`)
- TTFT p95 `23.162466478 → 23.893052590s` (**3.15% worse**)
- ITL p95 `1.559767750 → 1.323422992s` (`15.15% better`)

All requests/tokens/output hashes/usage/finish exact; zero failures/refusals/
cancel/discard; final 41MiB; cgroup OOM0.

**Decision:** P12 prefix scheduling performance item is closed for its recorded
scope. These are reconstructed diagnostics, not genuine equivalent-source vLLM
comparison and not source-correct SLO-goodput evidence.

### 9.6 Fair aged prefill/decode scheduling — `7dea460`, `ba7e376`

Shared engine implementation:

    pub struct Fairness {
        pub round_tokens: usize,
        pub prefill_age_rounds: usize,
    }

`WavePolicy` carries `fairness`.

`fair_round(...)` semantics:

- every ready row must appear exactly once in phase-homogeneous formed waves;
- Decode waves are prioritized;
- young Prefill is deferred while decode is live;
- Prefill becomes eligible when age reaches `prefill_age_rounds`;
- when no decode is live, Prefill is normally selectable;
- oldest age wins within class; original wave order breaks ties;
- selected logical-token total ≤ `round_tokens`;
- all active decode waves must fit or planning refuses;
- if aged prefill exists, at least one aged prefill wave must progress;
- multiple aged prefills may remain deferred so the scheduler does not overconstrain
  a round.

`Active<A>` owns `prefill_wait_rounds`.

Deferred rows retain exact original Active state/source owners and are requeued
with incremented age. Selected rows reset age. Executor/Ready/publication
semantics remain unchanged.

Configuration validation requires enough round credit for:

- one maximum admitted prefill wave;
- one decode token for every active session.

Production age = 2.

Focused evidence:

- 18 wave tests pass.
- 14 worker tests pass.
- strict engine clippy passes.
- production `qwen35-cuda` and CLI feature composition pass.
- physical unequal-arrival HTTP passes both approved 0.8B/2B:
  - B offered only after A's first actual ID;
  - first shared B2 no earlier than A output index4;
  - B output index at least1;
  - both sockets continuously drained;
  - CPU IDs/text/usage exact;
  - zero failed/refused/cancel/discard;
  - final 41MiB.

Final broad-gate nuance:

- every constituent command in `scripts/check.sh` has a terminal-green result on
  the exact scheduler source;
- formatting/architecture;
- full GPU feature-boundary;
- CPU clippy/workspace tests;
- minimal/core isolation;
- cache/source reference;
- serving/lifecycle;
- acquisition;
- release/installer/shell/Nix;
- qualification self-tests.

Repeated attempts to keep the entire monolithic `scripts/check.sh` shell wrapper
inside one connector job disappeared during ordinary cached builds with no
compiler/test error, no surviving child, no cgroup OOM and no GPU residue.
Those wrapper invocations remain **incomplete**, not passes. The complete
constituent command set is the accepted evidence for this scheduler slice.

`ba7e376` recorded that acceptance.

### 9.7 Temporal fused prefill primitive — `9dab004`

A reusable resident native temporal transaction was added below the existing
one-token fused body.

Scope:

- B2/B4/B8 equal-length chunks;
- 2..32 teacher tokens per row;
- host-token input;
- intermediate candidate domains remain private;
- original session roots remain unpublished;
- intermediate steps are state-only;
- final step returns one terminal `PreparedResidentCohort`;
- terminal mode may be state-only or greedy;
- dispatch/readback/state/logit receipts aggregate across the temporal chain;
- dropping/refusing the completed private chain cannot publish an intermediate
  logical root;
- ordinary cohort adapter understands whole-chunk state-only and terminal-greedy
  receipts;
- scheduler cursor advances by the whole chunk on publication.

This commit deliberately **did not** enable production temporal policy before a
physical acceptance cell existed.

### 9.8 0.8B B2 temporal physical acceptance and production integration — `1e0f4eb`

A small explicit qualifier was added:

`crates/zerogravity-gpu/examples/temporal_prefill_b2.rs`

Why:

- heavyweight `zerogravity`/CLI release test harness builds on L5 repeatedly lost
  the outer process while optimizing with workspace release
  `thin-LTO + codegen-units=1`;
- no Rust diagnostic;
- no cgroup OOM;
- so a narrow public-API qualifier avoids pulling the giant private test surface
  into one codegen unit.

Physical resident qualifier on RTX3060 proved:

- accepted one-token B2 workspace is prewarmed first so persistent batch workspace
  bytes are not mistaken for a leak;
- completed 4-token temporal candidate can be abandoned:
  - original roots unchanged;
  - all42 domains unchanged;
  - no pending events;
  - resident bytes return exactly to the warm workspace baseline;
- committed 4-token state-only chunk:
  - one logical root transition;
  - frontier +4;
  - CPU/B1/all42 exact;
- committed 2-token terminal greedy:
  - one logical root transition;
  - frontier +2;
  - exact packet ID/max bits;
  - exact producer binding;
  - CPU/B1/all42 exact;
- final ResourceArbiter reserved0;
- GPU returned to41MiB.

Marker:

`P12_NATIVE_TEMPORAL_B2_PASS source=qwen35-0.8b state_chunk=4 terminal_chunk=2 unpublished_abandon=true roots=atomic CPU_B1_all42=exact terminal_greedy=exact final_reserved=0`

#### Real integration bug discovered

The HTTP/scheduler path initially formed multi-row scheduler waves but executed
**zero native prefill cohorts**.

Health showed:

- `co_scheduled_members=6`;
- `gpu_fused_batches=0`;
- no native admission refusals.

Root cause:

- `NativeText` had never overridden `Executor::cohort_eligible`;
- the model-neutral default is `false`;
- decode can go directly to banding, but prefill passes through compatible-band
  eligibility first;
- therefore every prefill wave silently fell to singleton execution.

Fix:

- explicit B2-only prefill cohort eligibility;
- only while cursor is inside prompt;
- no device feedback present;
- pairwise compatibility requires exact same resident model Arc;
- same hidden width;
- both rows still in prefill;
- same current prefill chunk size.

This keeps physical/model ownership fail-closed.

#### Actual production HTTP proof

Independent CPU/tokenizer oracle for prompt `" a".repeat(10)`:

- prompt tokens: 10
- greedy ID: `264`
- text: `" a"`
- completion count:1
- finish reason:length.

Exact-current retained HTTP test:

`original_qwen35_08b_native_temporal_prefill_http`

Physical result:

- 1 test passed / 0 failed;
- two same-model requests;
- each prompt10 / completion1;
- actual B2 prompt chunks exactly `4 + 4 + 2`;
- exactly 3 B2 native cohorts;
- 2 state-only;
- 1 terminal greedy;
- 3 cohort publications / 6 rows;
- CPU text/usage exact;
- zero failed GPU quanta;
- zero native admission refusals;
- final GPU 41MiB /0% util.

Marker:

`P12_NATIVE_TEMPORAL_HTTP_PASS source=qwen35-0.8b B=2 prompt_tokens=10 chunks=4,4,2 native_batches=3 state_only=2 terminal_greedy=1 root_publications=3 CPU_text_usage=exact final_active=0 ...`

#### Deliberate rollout restriction

Production temporal promotion is **B2 only**:

- B2: chunk4 + equal-frontier prefill + explicit native prefill eligibility.
- B4/B8: stay on prior one-token prefill and unequal-frontier policy until their
  own physical qualification passes.
- Decode behavior remains unchanged.

The physical B2 cells used the Rust **debug profile** but execute the real CUDA
kernels, source ownership, state roots, Ready/publication and actual production
HTTP path.

They are functional/physical correctness evidence, **not canonical release-profile
packaging/performance evidence**.

Canonical release thin-LTO/codegen-units=1 builds remain operationally unstable on
L5; repeated build processes have disappeared without Rust diagnostics or cgroup
OOM. Keep release-profile B2 confirmation open.

---

## 10. Current validation/evidence ledger that matters now

### Accepted current scheduler/B2 source checks

On the current B2 rollout source:

- `cargo clippy -p zerogravity --features qwen35-cuda --lib -- -D warnings`
  passes.
- 18 `router::waves` tests pass.
- 14 `router::worker` tests pass.
- `cargo check -p zerogravity-cli --test serve_gpu --features serve,cuda`
  passes.
- `cargo check -p zerogravity-gpu --example temporal_prefill_b2 --features cuda-native-qualification`
  passes.
- `cargo test -p zerogravity --no-default-features --features qwen35-cuda,cpu --lib --no-run`
  passes, including the updated B2 temporal qualification accounting.
- Standalone B2 temporal resident physical qualifier passes.
- Exact-current B2 retained HTTP physical test passes.

### Do not overclaim

Not yet established from this B2 work:

- canonical release-profile B2 physical run;
- B4 temporal;
- B8 temporal;
- 2B temporal;
- temporal performance win;
- paged hybrid state;
- physical host/transfer/GPU overlap;
- release packaging performance;
- genuine equivalent-source vLLM parity/goodput.

---

## 11. Current graph/prefix/fairness/temporal decisions — concise table

| Area | Current decision |
|---|---|
| public CUDA graph serving | **Do not enable** from current B2/B4/B8 timing; keep private qualified cache/selector |
| native teacher-prefix scheduling | accepted in recorded warm and reconstructed cold scope |
| fair decode/prefill scheduling | accepted shared-engine foundation, age2 |
| temporal prefill B2 | accepted functional/physical for original0.8B, chunk4 |
| temporal prefill B4/B8 | not selected; still chunk1 |
| temporal prefill 2B | open |
| paged hybrid KV/recurrent | open |
| prefix COW/full-domain reuse | open |
| physical multi-stream overlap | open |
| vLLM matched goodput claim | open |
| canonical release-profile temporal B2 | open due L5 build instability |

---

## 12. Important rejected paths / do not casually repeat

The embedded Nov 14 baseline contains the full list. The most recent high-value
decisions are:

- Do not enable public graph serving from the small shrinking B2/B4/B8 gain.
- Do not weaken the transient graph idle invariant because one mismatch did not
  reproduce.
- Do not infer a temporal prefill pass from scheduler co-scheduling; before
  `cohort_eligible` was fixed, three 2-row waves still executed zero native prefill
  cohorts.
- Do not treat reusable native batch workspace retention as a candidate leak;
  prewarm the accepted band workspace before rollback-byte comparisons.
- Do not promote B4/B8 temporal policy because B2 passed.
- Do not call debug-profile physical correctness a release-profile or performance
  result.
- Do not relabel externally disappeared build/test jobs as pass/fail without a
  terminal semantic result.
- Do not resurrect rejected ordinary-Qwen consumers merely because their shared
  primitives remain useful.
- Do not use graph replay as a fallback after an unknown/failed graph execution.
- Do not substitute a different source/profile/backend for a missing acceptance
  cell.

---

## 13. Current open work / next execution order

The global and subsystem TODOs remain authoritative. For the exact continuation
from this handoff, the next coherent work is:

### Immediate P12-C temporal continuation

1. Revalidate L5:
   - clean status;
   - signed tip;
   - origin position;
   - one worktree;
   - no owned processes;
   - GPU baseline.
2. Read the current temporal section in:
   - `serving-and-speculation/TODO.md`;
   - `BATCHING_AND_PAGED_STATE.md`.
3. Keep 0.8B B2 accepted policy unchanged.
4. Build separate physical temporal cells for **B4 and B8**:
   - equal chunk state-only;
   - terminal greedy;
   - abandonment/rollback;
   - independent roots/all42;
   - exact producer;
   - final retirement;
   - no publication before final Ready.
5. Only after each width passes may its production prefill quantum change from1
   to4.
6. Qualify **2B** temporal independently; no permit inherits from 0.8B.
7. When L5 release build survives, rerun canonical release-profile B2 and the
   corresponding HTTP cell; keep functional debug-profile evidence distinct.
8. Re-run affected fair/HTTP/cancellation/EOS/slow-reader cases after wider
   temporal promotion.
9. Then return to the broader P12-C order:
   - paged hybrid state allocator;
   - complete-domain prefix reuse/COW;
   - event/lease-backed physical overlap;
   - only then overlap performance ablations.

### Broader still-open mission

- ordinary `.8B`/`2B` correctness across genuine admitted contexts;
- C1/C4/C8/C16/C32/C64 serving where fit;
- remaining ROCm2B / 27B cells;
- matched equivalent-source vLLM campaigns;
- committed SLO-goodput, TTFT/ITL tails, fairness and physical counts;
- memory/SSD/paged-state integration;
- speculation/MTP/n-gram/draft only after target correctness/state rollback
  contracts are established;
- broader model/backends/multimodal/distributed work according to global priority.

Do not let a serving optimization detour erase the correctness-first ordering in
the live TODOs.

---

## 14. Current subsystem map

### `serving-and-speculation`

Active P1.

Owns:

- OpenAI serving;
- continuous/fused batching;
- native cohorts;
- prefix scheduling;
- fairness;
- temporal prefill;
- paged state/overlap follow-up;
- speculation;
- matched serving benchmarks;
- cancellation/shutdown behavior.

Current hot section:
`Batching and paged-state follow-up — 21 Oct design only` plus the Nov15
fair/temporal continuation.

### `numerics-and-context`

Active P1.

Owns:

- independent numerical/source acceptance;
- genuine context progression;
- complete-domain state;
- backend/profile separation;
- vLLM/llama.cpp correctness policy;
- long-context cells.

Read `CORRECTNESS_POLICY.md` before claiming parity.

### `memory-and-streaming`

Active P1.

Owns:

- exact context fit/admission;
- source windows;
- completed-root backing reuse;
- paged hybrid KV/recurrent;
- complete-domain prefix COW;
- SSD streaming/residency;
- physical lifetime/eviction semantics.

### `cpu-gpu-optimization`

Active P1.

Owns backend parity/performance and controlled matched comparisons after
correctness.

### `pipeline-composition`

Active P1.

Owns recursive programs/profiles, component reuse, model/provider/format
boundaries, structure/naming and production-vs-qualification separation.

### `cli-device-routing`

Active P1.

Owns unified CLI/config/source/device selection, exact device identity and future
spill/peer routing.

### `model-execution`

Active P2.

Owns detailed CUDA/HIP implementation/qualification/performance chronology,
including the long retained/rejected primitive ledger. Do not rewrite that ledger
into another status system; use it when changing hot GPU execution.

### `model-acquisition`

Active P2.

Owns authenticated source/cache/provider/alias/acquisition behavior.

### Icebox

- multimodal;
- distributed execution;
- platform/release larger assembly.

Architecture docs describe future direction but do not grant current support.

---

## 15. Design/spec navigation

Use the smallest relevant set.

### Product and architecture

- `VISION_SCOPE_AND_PRINCIPLES.md`
- `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`
- `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`

### Engine / IR / compiler

- `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`
- `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`
- `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`
- `SEMANTIC_IR_AND_COMPILER.md`
- `ASYNC_RESOURCE_COMMAND_IR.md`

### Model / formats / numerics

- `MODEL_SEMANTICS_AND_COMPATIBILITY.md`
- `FORMATS_QUANTIZATION_AND_LOADING.md`
- `NUMERICS_QUALITY_AND_BENCHMARKS.md`
- `DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md`
- `numerics-and-context/CORRECTNESS_POLICY.md`

### Scheduling / state / memory

- `SCHEDULER_BATCHING_AND_SPECULATION.md`
- `STATE_CACHE_AND_TRANSACTIONS.md`
- `MEMORY_RESIDENCY_AND_IO.md`
- `UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md`
- `serving-and-speculation/BATCHING_AND_PAGED_STATE.md`

### GPU/device

- `DEVICE_KERNEL_AND_CODEGEN.md`
- `ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md`
- `OPERATOR_AND_FUSION_CATALOG.md`
- `COST_MODEL_AND_AUTOTUNING.md`
- `GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md`

### Serving / routing

- `RUNTIME_SERVING_AND_UX.md`
- `ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md`
- `MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md`
- `JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md`

### Composition

- `COMPOSITION_AND_EXTENSION_CONTRACTS.md`
- `pipeline-composition/STRUCTURE_AND_NAMING.md`
- `pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md`
- `pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md`

### Multimodal/distributed/future execution

- `UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md`
- `UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md`
- `REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md`
- `DISTRIBUTED_PARALLELISM.md`
- `DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md`
- `REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md`
- `NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md`

Historical Fabric design lives under:

- `.agents/plans/fabric-vnext/`
- `.agents/plans/fabric-vnext-transition/`

Those plans explain how the current architecture emerged; they are not the active
implementation queue.

---

## 16. Current source/model identities and benchmark facts agents commonly need

### Approved Qwen identities referenced by current work

Historical/current exact hashes are preserved in the embedded baseline and live
plans. Frequently reused current values:

0.8B reconstructed-prefix GGUF:
`57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`

2B reconstructed-prefix GGUF:
`91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6`

Tokenizer:
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`

Historical approved `.8B` HF source revision:
`2fc06364715b967f1860aea9cf38778875588b17`

Historical `.8B` safetensor:
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`

Canonical 27B snapshot:
`98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e`

27B tokenizer:
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`

Do not assume one source identity authorizes another route.

### B2 temporal HTTP oracle

For the retained 0.8B B2 temporal HTTP fixture:

- prompt: `" a"` repeated10;
- prompt tokens:10;
- independent CPU greedy ID:264;
- decoded text: `" a"`;
- completion count:1;
- finish reason:length.

The retained HTTP test verifies the actual response and physical cohort counters.

---

## 17. Clean restart procedure after a stuck/aborted turn

When resuming after tool/model interruption:

1. `git status --short --branch`
2. `git rev-parse HEAD`
3. `git rev-parse origin/main`
4. `git log -1 --show-signature`
5. `git worktree list --porcelain`
6. inspect live Cargo/rustc/ZeroGravity/native-gpu processes
7. inspect physical GPU baseline
8. inspect exact changed files/diffs before editing
9. read the current owning TODO section
10. continue from the last terminal semantic result

If a durable job disappeared:

- retrieve stdout/stderr if possible;
- inspect surviving children;
- inspect cgroup OOM events;
- inspect GPU residue;
- do not call the run a failure or pass without a terminal semantic result.

If X2 is needed:

- synchronize exact L5 bytes by bundle;
- resample host-global GPU/KFD baseline;
- preserve other owners.

---

## 18. Next-agent copy/paste starting prompt

Use this exact prompt to start the next agent:

> Use `dotlink---l5` as the only source-authoring connector. Continue in
> `/home/pvl/spaces/abird/src/zerogravity` on the single `main` worktree. Use
> `dotlink---x2` only for AMD large-VRAM execution/testing and synchronize exact
> L5 commits by Git bundle rather than pushing merely for transport.
>
> First read, in order:
> 1. `AGENTS.md`
> 2. `.agents/plans/README.md`
> 3. `.agents/docs/CURRENT_HANDOFF.md`
> 4. `.agents/docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-15.md`
> 5. `.agents/plans/TODO.md`
> 6. `.agents/plans/model-execution/TODO.md`
> 7. `.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md`
> 8. `.agents/plans/serving-and-speculation/TODO.md`
> 9. `.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md`
> 10. any owning plan/docs linked from those files.
>
> Revalidate Git status/HEAD/origin/signature/worktrees, live processes and physical
> GPU state before doing anything. Preserve unrelated work. Do not create another
> persistent worktree/branch. Commits must remain signed. Do not push unless I
> explicitly ask.
>
> Current implementation checkpoint before the handoff-doc commit is
> `1e0f4ebfc5b9650341814ad669d4101aea717623`
> (`feat(serving): enable qualified temporal B2 prefill`), with `origin/main`
> still `7dea460a80b420d0ce49a60f6c860b3502c67c38`.
>
> The current P12-C state is:
> - fair aged decode/prefill interleaving is accepted in the shared engine;
> - private production CUDA graph lifecycle/cache/selector is qualified but public
>   serving remains Native because B2/B4/B8 graph timing gains shrink with width;
> - reconstructed cold prefix scheduling is closed for its recorded scope;
> - reusable B2/B4/B8 temporal candidate-chain primitive exists;
> - **original0.8B B2 temporal prefill is physically accepted**:
>   resident rollback/state/greedy/all42 CPU/B1 proof passes, and exact-current
>   production HTTP passes prompt10 as chunks4+4+2 with3 B2 native cohorts,
>   2 state-only +1 greedy,3 publications, exact CPU text/usage, zero failures/
>   refusals, final41MiB;
> - production temporal promotion is deliberately **B2 only**; B4/B8 remain
>   one-token prefill and 2B is unqualified;
> - functional physical evidence is from the Rust debug profile; canonical
>   thin-LTO/codegen-units=1 release builds on L5 have repeatedly lost their outer
>   process without a Rust diagnostic or cgroup OOM, so release-profile B2 remains
>   open and must not be inferred from debug evidence.
>
> Continue with the next coherent P12-C slice: separately qualify temporal B4/B8,
> then 2B, preserving atomic Root/Ready/cancellation/resource semantics and only
> widening each production policy after its own physical acceptance. Retry
> canonical release-profile B2 when the build environment survives, but do not
> spend the entire turn blindly looping an operationally unstable build. Paged
> hybrid state/COW and event-backed physical overlap come after the temporal/fair
> foundation; do not jump to overlap before the required event/lease/cancel/
> unknown-completion/independent-root gates.
>
> Keep shared/model-neutral primitives clean and Qwen/source policy at the source
> layer. Preserve accepted ordinary CUDA/HIP paths and every retained useful
> primitive. Do not resurrect rejected consumers without materially different
> architecture/evidence. Keep public graph serving disabled. Use
> `CARGO_BUILD_JOBS=8`; run physical GPU work only through
> `python3 -B scripts/native-gpu.py -- ...`. Remove all `_tmp` artifacts.
>
> Limit this implementation turn to roughly 30 minutes. Aim for the next coherent
> checkpoint, update the owning TODO/handoff state, sign the commit, sync X2 if
> needed, then pause. Do not push unless explicitly requested.

---

## 19. Historical baseline policy

Everything below this line is the complete 14 Nov 2026 comprehensive handoff,
demoted only by one Markdown heading level. It is retained **word-for-word in
content** so the next agent has the full October→14 Nov design/implementation
history, including exact source identities, rejected paths, Fabric transition,
HIP/27B work, CUDA optimization chronology, benchmark evidence and historical
next-agent instructions.

Where the embedded baseline says “current”, “next”, gives a Git tip, or describes
host state, the **15 Nov sections above and live source/TODOs supersede it**.

---

## Appendix A — complete 14 Nov 2026 handoff baseline

## ZeroGravity full comprehensive handoff — 14 Nov 2026

Dated full takeover snapshot for `/home/pvl/spaces/abird/src/zerogravity`.

This is the one-file context pack for handing the project to another agent. It preserves:

- repository/host/workflow rules and exact current Git/GPU state;
- the original ZeroGravity mission and architecture evolution from the October design work;
- Fabric vNext structure, crate ownership and completed transition rationale;
- source/model/numerical/state/resource/cancellation invariants;
- Qwen ordinary CUDA/HIP integration, source/cache/acquisition and serving history;
- the full CUDA optimization chronology, selected primitives, retained primitives and rejected consumers;
- 27B/X2 host-load and HTTP history, including the corrected shared-GPU/KFD interpretation;
- production CUDA graph ownership/resource/compiler/lifetime/storage-event work through `f1ce532`;
- validation/build/physical-GPU rules, do-not-redo decisions, open blockers and exact next phase;
- the copy/paste prompt for the next agent; and
- the historical design foundation and deferred roadmap that led to the current implementation.

It is comprehensive context, not a frozen authority snapshot: **current source/tests and live TODOs outrank this file if `main` advances**. Git history remains the provenance/archive; do not recreate the retired SHA/archive/evidence workflow.

This file is the complete takeover map and a pointer into maintained source, tests
and plans. It is deliberately **not** the retired monolithic SHA/archive/evidence
handoff model. Git history retains superseded checkpoints; current source, tests
and live TODOs are authoritative.

### Resume here

#### Repository/topology checkpoint

- **Repository/source authority:** use `dotlink---l5`; work only in
  `/home/pvl/spaces/abird/src/zerogravity` on `main`. Do not create another Git
  worktree.
- **AMD large-VRAM execution/test host:** use `dotlink---x2` only for running and
  testing, in the same path `/home/pvl/spaces/abird/src/zerogravity`. Do not use
  X2 as an independent source-authoring branch; source changes land on l5 first.
- Latest signed **implementation** checkpoint is
  `f1ce532d49c4121324eb5f5a401f85ea900f1251`
  (`feat(cuda): bind graph replay to resident flight`). Above the accepted graph
  lifetime/resource substrate, a captured graph now retains the exact internal
  `(ResidentLease, DeviceAccess)` set and every replay must prove those same
  allocations with compatible access authority are retained by the current armed
  `ResidentSubmission` before the backend graph can enqueue. The graph is attached
  as a completion owner before that check, so refusal still fences/quarantines the
  same executable/storage authority. External graph APIs remain qualification-only
  and production serving still selects Native; no production graph-serving permit
  follows.
- Earlier key implementation checkpoints include `b0cf0b2`
  (`fix(cache): avoid duplicate Qwen snapshot hashing`), `134a571`
  (`fix(hip): isolate CUDA private KV qualification`), `874e67d`
  (`feat(cuda): retain shared-input projection pair`), `c282f20`
  (`fix(hip): restore source profile compilation`), `5bf440d` / `db7be60`
  (production CUDA graph substrate/resource admission), `6a5c610` (driver-safe
  nonpersistent NVRTC) and `81b7c71` (explicit graph-resource lifetime witness).
  Their docs checkpoints through `b12e8f5` are pushed.
- `origin/main` is `b12e8f5395af84bda5a5c0eab960efef2399f6c7`
  (`docs: record graph lifetime repeatability witness`). Local history above it
  currently starts with implementation `f1ce532` (exact graph replay
  storage/event authority); this refresh may add another docs-only commit. Earlier
  graph checkpoints through `81b7c71` and docs `b12e8f5` are already pushed.
  X2 is synced to the
  exact current local tip after each checkpoint even when its remote-tracking ref
  is stale. Never push unless the user explicitly asks.
- Before any new physical X2 run, sync it to the exact current l5 `main`
  bytes/commit without using a push merely as transport, then revalidate
  status/process/GPU state.
- Key pushed history now contained in `origin/main=b12e8f5` includes all
  earlier accepted CUDA/HIP work through `d68b649` plus `5c8f143`, `3cadc3b`,
  `5bf440d`, `bc9a1b3`, `db7be60`, `cbf1384`, `6a5c610`, `9dca4b`,
  `81b7c71` and `b12e8f5`. Do not describe those as unpushed local commits.
- Local docs `5c8f143` recorded the host-global X2 KFD observation and `3cadc3b`
  records rejection of the ordinary-Qwen `874e67d` consumer after completed
  loaded-model A/B; the primitive itself remains retained. Later host context
  corrected `5c8f143`'s ownership inference: X2 is shared with Ollama/model
  workloads, so that KFD owner is not attributable to ZeroGravity without host
  process evidence. `5bf440d` is the first broader post-micro-optimization CUDA
  implementation checkpoint.
- Important accepted commits immediately before origin:
  - `60b048d` — canonical-boundary private B8 GDN recurrent core accepted.
  - `775dfa6` — ragged private B8 KV transition accepted.
  - `c12acab` — grouped B8 recurrent transport primitive retained, serving
    consumer rejected.
- At this refresh no Cargo/model process is active in the connector-visible
  command namespaces and l5 RTX3060 is idle at ~41 MiB. X2 is a **shared GPU
  host**: the last baseline was total103,079,215,104 bytes VRAM,
  used31,376,207,872/busy0, with host-global KFD owner `2406838` holding
  31,098,232,832 bytes across six queues. The host also runs Ollama/model
  workloads outside the Bubblewrap PID namespace, so that owner is baseline and
  cannot be attributed to ZeroGravity from the sandbox. Do not require an empty
  GPU. Record the baseline before our run, stay within available capacity, and
  verify only our incremental VRAM/queue/process delta retires afterward. Never
  reset/reclaim/restart/kill other owners without explicit authorization.
- Root `README.md` is user-authored. Leave it unchanged unless explicitly asked.
- Keep one persistent `main` checkout per host. Reuse canonical target/cache state;
  do not recreate old worktree pools or `.agents/runs`.
- Commits are signed. Inspect `git diff --cached --stat` before every commit and
  stage only owned paths.
- On l5, `git log --show-signature` reports a good ED25519 signature while also
  printing a missing local allowed-signers principal warning. That warning is not
  a bad signature.

#### Read order / authority

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

#### Tracking / handoff method

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

### Owner direction / working style

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

### Canonical architecture / non-negotiable invariants

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

### Design/spec navigation

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

### Global plan ownership / pending-work map

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

### Current signed implementation state

#### Fabric / shared runtime

Accepted milestones include `860ce38`, `fef7123`, `8d761c4` / `8c43844` and
`1c99929`. The transition is complete; expansion belongs to the active model,
serving, memory and composition plans.

#### HIP / ordinary Qwen

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

#### CUDA composition / serving foundation

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

### CUDA execution/performance chronology

The exact measurements/tests live in `model-execution/TODO.md`. This is the
takeover map.

#### 9–10 Nov — C1 to genuine B4/B8

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

#### 11 Nov — dense B8, chunking and partial populations

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

#### 12 Nov — exact arithmetic + reusable stateless primitives

`b191fc6` fixed the key correctness boundary: physical widening must not select a
different source reduction. Exact B2/B4/B8 row kernels, ragged cohorts, inner-major
head packing, two-segment row split, sigmoid×multiply, stateless row spans,
RMS/residual-RMS, gate, Q/K RMS and source-exact rotary primitives followed.
Several consumers were rejected because lower dispatch count did not yield
repeatable serving gain; the clean primitives remain.

#### 13 Nov — stateful primitives, roots and hot-path profiling

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

#### Selected: `60b048d` canonical-boundary private B8 recurrent core

Current serving selects this.

- Canonical F32 recurrent roots stay the only publication/checkpoint authority.
- Eight lanes still import exact canonical state, execute one private B8 recurrent
  core in-place, then export exact canonical candidates.
- Gated output remains two accepted B4 groups.
- No native recurrent state persists across tokens.
- Qwen3.5-2B B8 47+16 kept sixteen ID264 tokens/lane.
- Controlled candidate mean ~716.377 ms vs `2f651b7` baseline ~745.102 ms:
  ~3.86% lower latency / ~4.01% higher throughput in that short workload.

#### Selected: `775dfa6` ragged private B8 KV

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

#### Retained: `c12acab` grouped B8 recurrent transport; consumer rejected

This is the current remote origin.

- Exact model-neutral 8-root recurrent transport, 17 explicit bindings.
- Qualified against scalar transport and CPU index oracle.
- Serving grouped import/export was only ~0.24% better against a two-sided
  baseline and slower than the first baseline side; rejected.
- Primitive remains; serving does not select it.

#### 13–14 Nov retained primitives; serving consumers rejected

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

#### Selected: `399e43a` B8 private-KV CTA256

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

#### 14 Nov post-KV probes: `de2eed8` / `cb9e349`

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

#### Retained: `874e67d` shared-input B8 projection pair; consumer rejected

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

### Current ordinary CUDA selected path

#### Selected

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

#### Retained but not selected in ordinary serving

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

### Reusable shared primitives that must not be lost

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

### Important rejected experiments / do not casually repeat

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

### Current performance frontier

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

### Numerical / state / ownership rules

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

### Current blockers / open work

#### l5 source/repository work

1. **Do not force another CUDA micro-optimization.** Easy launch-count, CTA-width,
   cache-hint and instruction-count variants are heavily exhausted. `874e67d`
   remains a useful exact primitive, but its real adjacent ordinary-Qwen consumer
   has now been measured and rejected because order/thermal drift dwarfed the
   ~1.04% favorable two-sided aggregate. Do not retry the same selection without a
   materially different composition.
2. **Production CUDA graph admission is the active broader l5 boundary.**
   `5bf440d` made graph identity/pin/replay/owner plumbing production-compilable;
   `db7be60` now pre-admits graph opaque resources through the same local ledger
   and global provider envelope. Policy is64KiB control +4KiB per prepaid
   source-dispatch slot + Rust graph/receipt/lease-access-vector metadata; actual CUDA
   driver bytes remain unqualified. The charge is acquired before `graph_prepare`,
   may grow only before capture starts, must cover the recorded body, survives
   CaptureWindow/CapturedResidentGraph/quarantine ownership and releases only after
   trusted retirement. Queue work uses prepaid `AllocationClass::Other`; there is
   no second global charge. External graph APIs remain qualification-only and
   production serving remains `FusedExecution::Native`.
3. Signed `6a5c610` closes the physical graph compiler blocker:
   `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` came from CUDA13.4 NVRTC PTX loaded by
   an older driver, so non-persistent C++ compilation now returns exact-SM
   cubin/SASS while persistent compilation continues to emit/cache PTX. Signed
   `81b7c71` strengthens the graph resource qualifier and signed `f1ce532`
   adds exact captured access authority to each retained lease. The current policy
   charge is **15,356,224 bytes** (32,768 bytes of owned lease/access-vector
   metadata); live foreign-ledger bytes must equal that charge, live charge owners
   must exist, and foreign bytes must reach0 immediately after trusted retirement.
   The original0.8B B2 path passes physically with warm `[3725,3725]`, captured
   body3632 launch sites, outside93/replay, six host enqueues,14,900 committed
   kernels, full248320 logits + all42 state domains exact, final ResourceArbiter
   reserved=0 and RTX306041MiB.
4. One post-`9dca4b` repeat failed closed at the first post-capture idle check:
   total resident bytes fell by exactly15,339,840 with pending_events=0 and an
   empty queue. It did not reproduce in **four consecutive** subsequent runs,
   including the final `81b7c71` witness run proving charge-retained-while-live
   and released-on-retire. Do not weaken the idle invariant or call the transient
   explained. If it recurs, use the warm/steady foreign-byte + charge-owner
   witness to localize it. `f1ce532` separately closes the same-card
   storage/event-authority seam: every replay is now rejected unless its armed
   resident flight retains the exact captured allocations and access modes. The
   physical B2 qualifier passes this path with final reserved=0 and41MiB→41MiB.
   Driver-internal graph bytes/node enumeration and graph performance remain
   unqualified. Production serving remains Native; production lifecycle/cache and
   selector integration are still open.
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

#### x2 AMD large-VRAM execution/testing — 27B host-load boundary next

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

### Validation / execution cadence

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

### l5/x2 build and GPU operational notes

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

### Worktree / disk discipline

- `main` is the only persistent checkout **per host**.
- No second l5 or X2 Git worktree for implementation/profiling/review.
- Author/edit source on l5. Sync exact l5 commits to X2 only as needed for AMD
  execution/testing; do not let X2 drift into a separate implementation branch.
- Reuse canonical target/cache state on each host.
- Temporary `_tmp` runners and profiling files are disposable.
- Do not recreate old worktree pools or `.agents/runs`.
- Preserve unrelated dirty work.
- No destructive Git operations without explicit authorization.

### Next agent must not redo

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

### Copy/paste next-agent prompt

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
> physical GPU state. At handoff construction the source/docs tip before the
> handoff-only commit is `9c86c0cd8396abd023d4504c6f2a559249d331fd`;
> latest implementation is signed `f1ce532d49c4121324eb5f5a401f85ea900f1251`
> (`feat(cuda): bind graph replay to resident flight`), and pushed
> `origin/main` is `b12e8f5395af84bda5a5c0eab960efef2399f6c7`.
> The handoff creation itself may add one newer docs-only signed commit, so trust
> live Git and sync X2 to that exact l5 tip by bundle before any new X2 physical
> execution. Never push unless the user explicitly asks.
>
> **Immediate continuation:** production CUDA graph admission is the active l5
> boundary. The graph identity/pin/replay substrate is production-compilable
> (`5bf440d`), opaque graph resources are prepaid in the existing resident
> ledger/provider envelope (`db7be60`), non-persistent CUDA NVRTC is driver-safe
> (`6a5c610`), graph resource lifetime is explicitly witnessed (`81b7c71`), and
> `f1ce532` now closes the same-card storage/event-authority seam by retaining the
> exact captured `(ResidentLease, DeviceAccess)` set and requiring the armed
> `ResidentSubmission` to prove those same allocations/accesses before every
> backend graph enqueue. The original0.8B B2 physical qualifier passes full248320
> logits/all42 state domains, six graph host enqueues, final reserved=0 and
> RTX3060 41MiB→41MiB. Current conservative graph policy charge is
> **15,356,224 bytes**, including **32,768 bytes** of owned lease/access-vector
> metadata. Driver-internal graph bytes/node enumeration remain unqualified.
>
> **Next phase:** add the private production graph lifecycle/cache and clean
> retirement/replacement policy around the existing `ResidentModel`/batch workspace.
> Keep external graph capture/replay wrappers qualification-only and keep public
> serving on `FusedExecution::Native` until the lifecycle/cache is complete and
> fail-closed. Do not jump directly to a router selector. Once lifecycle/cache is
> coherent and physically qualified, integrate a production selector that can only
> choose a compatible live graph and falls back/refuses according to the declared
> contract without widening numerical/state/resource semantics; then benchmark the
> selected path against the same native path before calling it a serving win.
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

### Historical design foundation — retained context, not current execution instructions

The sections below preserve the original product/architecture evolution and enduring design
context from earlier handoffs. **Do not treat old “next step”, worktree, campaign or status
phrasing inside historical material as current instructions.** The current continuation is
entirely above and in the live TODOs.

### 3. Original mission and enduring product design

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

### 4. Engineering philosophy and non-negotiable invariants

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

### 5. Important rejected patterns

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

### 6. Architecture evolution from the beginning

#### 10–12 Oct: initial architecture and v4 consolidation

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

#### 13–16 Oct: source formats, quantization, model families and GPU ownership

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

#### 17–22 Oct: full-model numerics, GPU native paths and serving foundations

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

#### 23–26 Oct: integration, routing, acquisition and broader backend work

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

#### 27 Oct: Fabric vNext design

The user requested a clean future architecture where the whole target structure becomes visible early in code, then current implementation migrates into it.

The first structural milestone is intended to introduce model-neutral engine and reusable serving crates, visible typed contracts and explicit unsupported stubs before deep migration.

That was the design-only checkpoint. Signed`7396984` now implements S0; functional migration remains open.

#### 28 Oct: ordinary source-derived Qwen CUDA integration

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

#### 28–29 Oct: evidence hardening, cache human aliases and repository normalization

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

### 7. Current repository/runtime structure

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

### 8. Current Qwen ordinary CUDA status

#### What is implemented

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

#### Exact approved .8B source

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

#### Recovered reference and original blocker — 29 Oct

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

#### Owned execution checkpoint — `2983bb7`

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

### 9. Ordinary cached47→50 parity protocol

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

### 10. Exact ordinary CUDA context admission

Under the unchanged 4608 MiB provider / 5120 MiB physical limits, totals include
cold model, two roots and retained conservative admission headroom. The owned
GEMM allocates no scratch; these are not claims of physical scratch usage.
The 30-Oct correction includes the previously omitted 280-byte rotary table.

#### .8B

- 512: 1,652,916,585 B
- 1K: 1,667,662,185 B
- 2K: 1,697,153,385 B
- 4K: 1,756,135,785 B
- 8K: 1,874,100,585 B
- 16K: 2,110,030,185 B
- 32K: 2,581,889,385 B

All fit admission.

#### 2B

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

### 11. Deferred Qwen mission

Fabric transition is now first. Do not rerun already accepted cached47→50,
bounded64 HIP, finite27B, lifetime or GDN scopes merely to restart the campaign.
After migration, resume genuine multi-row0.8B/2B HIP and call-free lowering,
then general27B/populated fit contexts/C1..C64/matched vLLM. Preserve the separate
CUDA2B/wider reference and ROCm baseline authority gaps. Exact current scope:
`../plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md` and owning
model-execution/numerics/serving TODOs. Production remains disabled.

### 12. Serving and router state

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

### 13. Cache/acquisition current state

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

### 14. Device/CLI/config state

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

### 15. Memory/SSD/state direction

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

### 16. Model-family status discipline

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

### 17. Fabric vNext — historical design that led to the completed transition

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

#### Planned transition sequence

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
