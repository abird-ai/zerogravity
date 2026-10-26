# ZeroGravity — next-agent starting prompt, 26 October 2026

Copy the text below. This is the current takeover prompt; older prompts are
historical provenance. The next agent must independently establish live state.

~~~text
Take over ZeroGravity at /home/pvl/spaces/abird/src/zerogravity.
Use GPT-6.1 Sol xhigh for all coordination, architecture, implementation,
execution and review; old Pi/Astra assignments are superseded.

Read ALL of .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-26.md.
Then read AGENTS.md, .agents/plans/README.md, the plan queue and owning TODOs,
and the handoff-2026-10-26 companions: USER_DECISIONS.md,
DESIGN_AND_DECISIONS.md, SOURCE_HISTORY.md, OPEN_WORK.md,
EVIDENCE_AND_CONTINUATION.md and SOURCE_CATALOG.md. Use their complete
272-commit history,374 live noncompleted plan blocks and264-original-document
catalog; retain the original245-commit/360-marker indexes as history.
Read large files in bounded chunks without treating truncated output as a
full read. Read the actual archived DECISION_REGISTER.md, master v4.3,
whitepaper/ES-00…15 and latest shared ForwardPlan/structure/correctness specs.
Older17/19/20/25AM/PM/NIGHT handoffs preserve history, not current status.

Independently inspect HEAD/status/index, signed commits, worktrees/stashes/jobs,
model permissions and current hardware/SDK/driver/source/binary identities.
Implementation cutoff was signed d74f59a18c6345dbea963861acd519c1282a26fa;
the new handoff documentation commit follows it. MAIN was clean.
The sole side worktree was synchronized0/0, with only11untracked64860B
GDN draft files preserved unstashed; inspect before touching or deleting it.
No owned model/build/device jobs remain running.

First finish the real architecture cleanup across Qwen .8B/2B/27B and Gemma4:
remove remaining captured-testcase dispatch and qualification/fault coupling
from ordinary loading/execution. Implement genuine parameterized GDN and
explicit numerical/kernel/ABI/resource contracts using the existing engine.
Do not rename a whitelist as a capability descriptor or append48–50 cases
to runtime tables. The parametric source proof/design is preserved, but
runtime implementation is unfinished. Tests/fixtures belong in Rust tests
or explicit diagnostic targets. Placement is a rule/review convention:
do not add convention-enforcement tests.

Then qualify current-source .8B fresh47/48/49/50 AND real cached continuation,
affected2B/B4/source/physical behavior and trained-model shutdown/faults;
proceed to genuine long contexts and original BF16 CPU/CUDA/ROCm/Vulkan routes.
Use the common pinned vLLM/llama.cpp correctness process: exact semantics/
ownership, declared per-op accuracy, independent model quality, separate
determinism; preserve existing exact source-bit contracts and every failure.
Full27B/Gemma model suites are waived for the structural refactor only.
After correctness, build an actually optimized release and run quiet
SAME-ATTEMPT C16 stock vLLM vs ZeroGravity for BOTH small models.
No C2; C32/C64 later. Qwen27B original natural full-window capture on x2
comes afterward, under exact source/image/resource/validator and trust gates.

All four fresh original47–50 full-ABI references now pass; native48–50
and current-source physical/C16 acceptance remain open. The latest source
gate passes, but is not inference qualification. Historical strict
single-request/batched failure stays failed; earlier A3C16 .8B win/2B loss
does not certify current code. Serving shutdown is fixed in new source,
with model-free PID1/container controls; trained-model replay remains.
Historical l5 benchmark container termination is probably cleanup escalation;
fatal signal/sender/OOM is unproven. Old l5 read-access and relay blockers
were overcome; x2 strict host trust was verified. Last x2 preflight instead
found missing exact model/image, insufficient available RAM and a validator
signature mismatch. Recheck time-sensitive facts without bypassing trust.

Preserve one ResourceArbiter, one ForwardPlan, one StateRoot publisher,
exact numerical/source identities and original physical owners.
Keep CUDA/HIP/Vulkan SDK/ABI/driver identities separate. The local
Renoir/gfx90c launch/recovery hold remains. Preserve README byte-for-byte
(SHA2566c00873ec0ac2c59bc73703b55a9bfa4d93703334a03fa16bcac2d204cda5b15).
No unapproved model/SDK/network downloads, privilege/driver/reset/recovery
changes, SSH host-key bypass, silent source/numerics/model substitution,
push/tag/release/history rewrite or loss of unrelated work.

Work in substantial batches; fix major source/refactor bugs first and
commit ready logical units promptly with signatures. Finish MAIN first,
then reconcile side work once after a useful batch, preserving unique work.
Use independent authorized subagents with explicit file/build/GPU ownership.
Use meaningful focused checks and required real source/state/physical/
serving validation, then full scripts/check.sh last on assembled source
through the approved native-gpu.py isolation and a supervised long session.
Reuse accepted unchanged-path evidence; avoid per-helper/per-commit rituals.
Set PYTHONDONTWRITEBYTECODE=1 and use python3 -B. Bound searches to named
source/run paths. Preserve full stdout/stderr, actual return code/timeout,
source/binary/model/SDK/driver pins, semantic failures and cleanup.
No interrupted-run green status or asynchronous/background completion claims.

Continue the next authorized substantial source unit. Report concrete
delivered behavior, source commits, validation, true blockers and next work.
~~~
