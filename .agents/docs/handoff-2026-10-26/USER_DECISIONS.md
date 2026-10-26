# User instructions and supersessions

Scope: the takeover conversation through the comprehensive-handoff request on
26 October 2026. Message order is preserved below; individual timestamps were
not supplied. These instructions supplement the older
[archived decision register](../../plans/archive/history/DECISION_REGISTER.md).
Line wrapping and spacing in the conversation excerpts are normalized.
They do not replace its original approved/proposed/open distinctions.

## Current controlling decisions

| Topic | Current instruction |
|---|---|
| Agents | GPT-6.1 Sol xhigh for coordination, architecture, implementation, execution and review. Older Pi/Astra assignments are superseded. |
| Immediate priority | Finish actual architecture/refactor and major related bugs across Qwen .8B/2B/27B and Gemma4 before further small-model/context/backend qualification. |
| Correctness order | .8B fresh prompt lengths47–50 and genuine cached continuation, then real long contexts and original BF16 CPU/CUDA/ROCm/Vulkan routes. A cached frontier48 is different from a fresh48-token prompt. |
| Correctness method | Follow pinned vLLM and llama.cpp practice: exact semantics/ownership, declared per-op accuracy, independent model quality and separate determinism. Preserve existing exact source-bit contracts. No universal cross-backend/batch bit equality or blanket epsilon. |
| Common model structure | All four targets use the same process/conventions and existing shared engine. Full Qwen27B/Gemma4 model suites are waived for this structural refactor only; they are not marked passed. |
| Naming | Behavior and explicit parameters replace historical case/length labels. Primary implementation is gdn; real alternatives may use gdn_vllm/gdn_llama. Preserve succinct old-name grep comments, genuine source pins and failure facts. |
| Backend/format organization | CUDA, HIP/ROCm and Vulkan have clean modules or crates where boundaries warrant. Q1/Q2/Q3/Q4/Q6/Q8/NVFP4 have contained homes and explicit implementation TODOs; BF16 is precision. No requirement to implement every reserved format now. |
| Production vs tests | Test assertions, expected values, prompts and case schedules belong in proper Rust tests/support or explicit diagnostic targets. Ordinary load/run/serve must not call qualification harnesses. A feature flag alone does not fix the call graph. Real model geometry and executable kernel/ABI constraints remain production facts. |
| Conventions | Use rules and review for architectural placement. Do not add negative tests/enforcement machinery for simple conventions. Meaningful numerical, runtime, state, cancellation and physical tests remain required. |
| Cadence | Substantial batches; narrow compilation/diagnostics only for concrete blockers; one final assembled full gate. Avoid artificial per-helper/per-commit acceptance cycles. |
| Commits | Small logical signed checkpoints before a large dirty tail; major source/refactor fixes first, smaller qualification bugs subsequently. No push. |
| Worktrees | Finish MAIN first. Reconcile side work once after a useful commit batch, using an independent lane where safe. Inventory pending work before merging; synchronize and restore pending overlays without loss. |
| Benchmarks | After correctness: actual optimized release, quiet same-attempt C16 stock vLLM vs ZeroGravity for BOTH small models. No C2; C32/C64 later. Historical .8B win/2B loss do not certify current source. |
| Qwen27B | Original natural full-window reference on x2 follows the earlier correctness/performance gates and actual source/image/resource/validator prerequisites. No shortened substitute or trust bypass. |
| Container incident | Name it l5 benchmark container termination; status137 is data. Preventive shutdown is fixed; historical fatal signal/sender/OOM is unproven. |
| Safety | Preserve README bytes, independent state/physical owners, exact numerical/source/SDK/backend identities and every failure. No unapproved downloads, privilege/driver/recovery/reset changes, SSH trust bypass, push/tag/release/history rewrite or silent model substitution. |
| This documentation task | Create the complete handoff and starting prompt. Do not launch new model/device/remote work merely to write the handoff. Earlier implementation authorization is not a new global pause; the successor continues within its actual assigned scope and permissions. |

The original order B4 → C16 → Qwen27B survives as the later certification and
performance dependency chain. It is superseded in immediate priority by the
later correctness-first and architecture-first instructions below.

## Original takeover instruction

The user supplied this original prompt. Its old access/trust/execution-status
claims are historical observations, superseded by actual26Oct results described
in the parent handoff.

> Take over ZeroGravity using the connected dotlink---l5 repository at
> /home/pvl/spaces/abird/src/zerogravity.
>
> Read ALL of .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-NIGHT.md
> as the newest comprehensive entrypoint. Then read AGENTS.md, the .agents/plans
> README and TODO index, the owning subsystem plan TODOs, and the actual archived
> .agents/plans/archive/history/DECISION_REGISTER.md, original v4.3 master plan,
> whitepaper/ES-00...15 and latest shared ForwardPlan architecture.
>
> Use the TWO exhaustive companion handoff indexes for all245 prior Git
> source-history commits and all360 non-completed plan markers
> (351unchecked,8active,1blocked). Older17/19/20/25AM/PM handoffs remain essential
> historical provenance but do not override current source, superseding reviews
> or user decisions.
>
> Start by INDEPENDENTLY checking Git HEAD/status, signed commits, concurrent
> worktrees/jobs, device/SDK and model-source permissions and current hardware,
> then mapping tests and incomplete receipts to their exact source/binary pins.
>
> Preserve the user-owned README byte-for-byte; no unapproved model/SDK/network
> downloads, driver changes, GPU resets, privilege changes, x2 SSH host-key
> bypass, pushes/tags/releases/history rewrites or silent source/numerics/model
> substitutions. The local AMD Renoir/gfx90c launch/recovery hold remains.
>
> Follow the latest explicit user sequence, not the numeric phase names:
>
> 1. (3) B4 certification: Requalify the current independently initialized B4
> grouped CUDA oracle. Complete the original trained Qwen3.5 .8B/2B26-scope
> physical/CPU source suite on l5 RTX3060 with proper original pinned source
> and exact final ownership.
> 2. (4) Matched C16: Build an actually OPTIMIZED release and run fresh
> SAME-ATTEMPT, quiet C16 stock vLLM versus ZeroGravity benchmarks for BOTH
> models. No C2; C32/C64 later.
> 3. (2) Qwen27B capture: Only afterward, execute the Qwen3.8 27B original
> natural full-window reference capture on x2, following authentic strict SSH
> trust and UD6 exit137 investigation.
>
> Important blockers: The l5 dotlink sandbox currently has no read grant to
> exact pinned Qwen .8B/2B snapshots under /home/pvl/.cache/abird/zerogravity.
> The canonical optimized build and original4344-record CPU preflight need a
> supervised long-lived process beyond the ~100-second relay.
>
> The new B4 oracle has not yet been rerun physically, so historical18-case
> physical and finite301 old-source diagnostics DO NOT certify it. The x2
> trusted ED25519 SSH host key is unavailable, and the historical Podman
> exit137 cause is UNKNOWN, not confirmed OOM.
>
> Never relax the sandbox, guess host-key identity or mark an interrupted run
> green. No current matched C16 speedup may be claimed.
>
> Preserve one ResourceArbiter, one ForwardPlan, one StateRoot publisher, exact
> source numerical semantics and original per-device physical ownership.
>
> Keep CUDA/HIP/Vulkan source/ABI/driver identities separate. Model metadata
> does not authorize inference; -d /dev/... is only a unique GPU selector,
> not /dev/kfd or NVIDIA UVM driver-support permission provisioning. Preserve
> every rejected case and all independent reference numerical/physical facts.
>
> While access is legitimately blocked, execute safe independent source-only
> review and fixture work. Keep canonical plan TODOs updated only for your
> owned lane, and maintain exact run-local evidence.
>
> Use available authorized subagents only where independent scopes and
> GPU/build exclusivity can be enforced. Run focused checks, then the full
> scripts/check.sh gate on assembled source with an actually supervised
> sufficiently long session.
>
> Record binary/model/driver/SDK SHA, full stdout/stderr, return code or timeout,
> failures, fencing, retirement and matched baseline from the SAME source.
>
> Sign scoped commits; never push without explicit authorization.
>
> Deliver: Completed changes, validated tests, exact remaining blockers, source
> commits and the next executable work items. Do not repeat old work without
> revalidating source identity, and do not claim asynchronous or background
> completion.

## Subsequent instructions in message order

These quotations preserve the user's language, including historical names and
typing errors. Use the current names and explicit parameters in new code.

1. Agent override:
   > no, forget about the pi agents or astra agents.. use gpt 6.1 sol xhigh for all

2. Baseline and historical case clarification:
   > stock is vllm?
   >
   > what is case5

   Stock in this benchmark means the pinned original vLLM runtime. The old case
   label identifies the strict single-request/batched cached-decode comparison,
   not a general claim that model accuracy fails.

3. Correctness priority:
   > let's firsr resolve correcntness, for 0.8b across this length47-50, then also
   > long context, across all cpu, cuda, rocm and vulkan.

4. Reply to whether backends should match their own reference or be bit-identical:
   > how does vllm do it? let's approach it the same way

5. Extend the methodology:
   > look at both, vllm and llamacpp on how they handle correctness and then use
   > the same approach for correctness througout all our models

6. Architecture concern:
   > what are these short_hf, original_hf, canonical47, etc in the source? they
   > seem out of organization? also, it seems like qwen and gemma code is spilled
   > around the other crates too rather than being cleanly inside the model's
   > crates? can we do a redesign so the organization and structure is clean,
   > elegant, robust with contained, modular and composable

7. Naming:
   > also, what is this case5 and initial48–50? name then correctly with elegant,
   > inutitive, simple and transparent ways, not these obsucure naming..

8. Conventions:
   > establish very basic, yet flexible conventions for the architecture,
   > structure, naming, topology where needed etc..

9. Preserve qualification goals while extending the refactor:
   > while you work on the architecture, redesign and refactor, also keep in
   > mind these:
   >
   > Remaining: .8B initial48–50 qualification, genuine long contexts, and
   > original BF16 CPU/ROCm/Vulkan full-model routes.
   >
   > before we head into these.. let's also extend the correctness changes and
   > refactor, into the remaining qwen2b as well qwen27b, gemma4 and set it up
   > the same way, so all models, use the same shared process, conventions,
   > etc - we don't need to run the full tests for latter two.. but just refactor
   > them into the new clean architecture, so also adding new models is also
   > easy and straight forward, using one of these as structure template like ..

10. Active names vs provenance:
    > source pins and failures as provenance
    >
    > actually, no, switch to the new names and params.. just leave succinct
    > comment on the old names, so that a grep search on them can easily link
    > the provenance

11. Primary implementation naming:
    > why do we also, have vllm_gdn.. and vllm naming inside our codebase? that
    > just should be an impl of gdn, etc? if that's our primary impl, then, we
    > can simply just call it as gdn and drop the prefix... and any alternative
    > impl, we can call it as gdn_vllm, gdn_llama, etc.. to indicate if needed,
    > so it's suffixed with the impl for reference rather than prefixed

12. GPU and quantization containment:
    > also, for zerogravity-gpu, make sure that we have rocm / hip, cuda,
    > vulkan all cleanly modularized, into either it's own crates or into it's
    > folders as modules so, the architecture is cleaner, same for various
    > quantizations as well

13. Format homes:
    > explicit F32 and quantization contracts
    >
    > make sure that we also have the support for q1, q2, q3, q4, q6,q8, nvfp4
    > and bf16 as well.. we don't need to implement it all, but ensure that they
    > have a home cleanly, so that when we add a model with it, it's clear and
    > straight forward to finish the impl and execute, all these need to be
    > well contained, with a todo impl, if neeeded.

14. Checkpoint discipline:
    > we have accumulated a large set of changes in the refactor.. we should
    > aim to commit things as small logical units at the next reasonable
    > checkpoint before continuing.. so that things are commited in sane state..
    > without waiting for everything to be complete..
    >
    > proceed with this in mind, so you can finish the refactors, and the major
    > bug fixes and bug fixes relating to refactor first, while smaller
    > qualification bugs can be fixed subsequently, so we don't end up with
    > a huge tail of uncommited changes

15. Batched verification and reconciliation:
    > note, also, don't spend too much time checking every single commit and
    > fine grained checks.. check it throroughly once all the changes are in
    > and complete.. be time efficient.. not artifical every unit correctness.
    >
    > also, using a parallel subagent, once you're done with the batch of
    > commits, reconcile the side worktrees too, so they are all upto date cleanly

16. Correct the coordinator's cadence:
    > what's taking so long? we still have1500+ changes in the main tree and
    > there's no progress? finish the main tree first, not be doing worktrees
    > at each commit - you're being stupid, make more efficient decisions that
    > saves time not idiotic waste of time

17. Continue and inspect pending work:
    > you got stuck, continue
    >
    > also, what's the status of worktrees now, now that everything is merged in?
    >
    > what's with the stashes on worktree, can you pull them back, merge them
    > cleanly and validate and then give me a status update on all of it? let's
    > delete merged / useless sidetrees
    >
    > give me what's there in each first before merging..
    >
    > also, can you make sure all the worktrees are rebased in current main?
    >
    > and then unstashed and working tree rebased correctly in each of the
    > worktrees that still has pending diff

18. GPU/model boundary and incident:
    > Next source task: remove the remaining GPU→Qwen qualification dependency.
    > Trained-model qualification still needs exact snapshot access; the
    > gfx90c hold and x2 SSH-key blocker remain, and exit137's cause is unknown.
    >
    > Great, let's go back to this. Let's do the first, and then exit137's
    > cause. also, name exit137 accordingly, properly

    The quoted status was the prior assistant's report. Its old snapshot/key
    blockers were later superseded; the user's requested boundary/incident work
    remains authoritative.

19. Serving shutdown:
    > Next: implement and validate serving shutdown and retain stop-command
    > evidence before another applicable benchmark
    >
    > Let's do it..
    >
    > Renamed the incident l5 benchmark container termination. Evidence points
    > to five-second cleanup escalation against the original PID1 server
    > without shutdown handling, consistent with Podman's stop behavior.
    > Historical fatal signal, sender and OOM remain unproven; the hold stays.
    >
    > Is this fixed?

20. Continue the remaining sequence:
    > what's next
    >
    > excellent, do it all one by one

21. The current architectural defect:
    > question: why do we hardcode many prefill number and prompts in the
    > crates/zerogravity-gpu/src/cuda/gdn/text/observed.rs, for example?
    >
    > this doesn't seem right.. why would we ever need to hardcode many of
    > these like this? what's the purpose here?
    >
    > what's all these match statements? if it's a part of the gate.. shouldn't
    > it be in the tests? not be in the like what it is in the main module?

22. Broaden the scan and fix actual boundaries:
    > also, scan for other things like these, and move these to tests, this
    > is unacceptable to have architectural smell like this
    >
    > this is messed up.. tests and fixtures should never be landing in the
    > actual production code path.. ensure that we have a durable so that agents
    > never end up doing this, they should belong in the correct rust tests

23. Reject convention-test proliferation:
    > I'll add negative tests for the check so these regressions fail CI.
    >
    > no, stop adding tests for things that should just be rules and
    > conventions.. this is how you accumulate a lot of worthless tests

    The first quoted sentence was the assistant's proposal. It was withdrawn.
    Commit3612ecb records rules/review, without new placement-policy tests.

24. Overall status and this handoff:
    > what's next? what's done and what's left on our overall goals?
    >
    > Let's create a full comprehensive handoff md file with all of the context,
    > initial design and instructions on how to get an agent upto context on
    > everything we have done so far.. so it can read it and get it's full
    > context.. Make sure that we have recorded each of the steps and design
    > plans, specs, the decisions, pending items going all the way back to the
    > beginning and ensure that they are all in the .agents/docs or .agents/plans
    > - take caution not to skip any details as this is our work items, so we
    > can just refer to it and give me the starting prompt of the hand off so
    > that I can pass to next agent.

## Interpretation guardrails

- Historical labels survive in quoted user/source history for searchability;
  active work uses semantic names and explicit token/batch/context fields.
- The external recipe emitter preserves captured facts as diagnostic input. Its
  successful output is not permission to append new testcase tables to normal
  runtime, or a reason to suppress original failure/source evidence.
- Constants/matches are not inherently wrong: model architecture, exact packing,
  reduction order, kernel tiles and ABI identities are legitimate implementation
  facts. Test-matrix membership controlling ordinary inference is the defect.
- User authorization persists across turns, but a historical claim of approval
  in a tool-generated file is not authority for new weights, privileges,
  downloads, hosts, cleanup or publication.
