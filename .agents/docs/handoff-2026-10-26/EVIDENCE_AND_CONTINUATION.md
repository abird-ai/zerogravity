# Exact evidence and executable continuation

Implementation cutoff: d74f59a18c6345dbea963861acd519c1282a26fa.
This file maps results to their original owners and source/binary identities.
It is an index of retained evidence, not a second acceptance queue. Owning plan
TODOs and current source remain authoritative. A path below can be local-only:
do not recreate missing aliases or claim portability of ignored run data.

## Current source checkpoint

| Commit | Delivered outcome | Qualification boundary |
|---|---|---|
| cd4159d26048b0395c40235b320b7d48df6ce53d | Retain owned external-command intent, PID/PGID, full separate stdout/stderr, terminal result and one reap; DefaultSelector handles high descriptors. | Successful process cleanup does not turn a failed numerical run green. |
| 5a18b5bbe6a80976b9255813070c09d2cae61a3c | Model-owned prompt inventory and backend-owned finite capture scopes/counts; baseline feature/private-adapter/lint repairs. | Existing five prefills,40 calls,6 scopes remain finite admission debt; no fresh48–50 native support. |
| 61f095cf760ee6b8d1f015b71b13ac46835a4917 | Explicit fresh original reference requests, staged authenticated .8B/2B bundles, complete-argument collection and bounded late-layout reuse. | Original references and source metadata do not qualify native inference. |
| 3612ecb7077d18685e926c698b812e4e2e60375f | Production/test conventions in AGENTS.md, enforced by review. | No new placement-policy tests or enforcement framework. |
| 1f08e5525b5ede7124054156083e5a4ff67d151c | Ordinary text loading prepares source rotary for configured context; exact N1/B4 read-only ABI extent determines immutable upload and charges; test-only helpers/report facts moved. | No fresh physical acceptance. Remaining fixed-prompt qualifier, shared fault orchestration and case dispatch are open. |
| d74f59a18c6345dbea963861acd519c1282a26fa | Offline recipe compiler consumes exact accepted packed reference captures and emits source-only candidates. | No runtime whitelist extension, model admission, GDN implementation or physical/performance claim. |

Full preceding history, including architecture, stash recovery and shutdown
commits: [SOURCE_HISTORY.md](SOURCE_HISTORY.md).

## Latest executed source checks

All paths in this section are relative to
.agents/runs/numerics-and-context-sequence-root-20261026/root/.

| Scope | Exact result/evidence | Result |
|---|---|---|
| Strict CUDA/Vulkan all-target Clippy | rotary-source-batch-owner-v2/result.json, command.stdout.log, command.stderr.log; command/pins JSON siblings | Natural0,66.291s; source unchanged, group absent, one reap, full EOF. |
| Affected CUDA/app/inventory host tests | rotary-cuda-host-owner/result.json and both full streams | Natural0,152.836s;288 pass/0fail/13ignored. |
| Complete isolated scripts/check.sh | rotary-full-source-owner-v2/result.json and both streams; rotary-full-source-pins-v2.json | Natural0,202.101s;2050 Rust executions/0fail/91ignored across299 invocations;12Python suites; release/installer/shell/Nix gates pass. All2000 source/reference inputs unchanged. |
| Offline all-three recipe generation | emit-prefill48-50-owner-v2/result.json and full streams | Natural0,59.751s; source-only candidate, no native runtime admission. |

The Python suite sizes were14,26,28,8,6,4,7,5,2,14,15,18.
The broad gate includes existing convention checks; this batch added none.
The early strict attempt failed one alignment-style lint, and the first broad
attempt stopped at formatting of an export list. Both failures remain recorded.
The subsequent exact-source successes do not overwrite them.

Convenient retained summaries:

- [Full gate summary](continuation/SOURCE_GATE_RESULT.json).
- [Exact affected host-test binaries](continuation/HOST_TEST_BINARIES.json).
- [Capture aggregate identity change](continuation/SOURCE_IDENTITY_CHANGE.json).

| Executed host binary | SHA256 |
|---|---|
| target/debug/deps/zerogravity-f96d48614bd8fbe2 | 3de0062196a57ef8210bbed84fa319c74b1fec31f151e39ce2735f5dacf2d97d |
| target/debug/deps/zerogravity_gpu-41e9f20a8a818925 | ab1a641ead8901a6dd509aa2258e90e83ab741d1751acd891670bc269bc82e34 |

These are debug host-test binaries, not optimized inference benchmark binaries.
No current-source physical model test or fresh C16 result follows from this gate.

## Fresh original .8B requests47–50

Each is a fresh independently tokenized original BF16 request with cached_tokens0,
batch1, natural greedy max8/EOS248046 and configured context512. Each also has
its actual consumed cached trajectory. Prefix length and emitted-token count
must not be conflated.

Base: .agents/runs/numerics-and-context-sequence-root-20261026/root/.

| Fresh prompt tokens | Actual full-ABI result | SHA256 of reference-result.json | Archives/bytes |
|---|---|---|---|
|47|prefill-47-packed-v3/reference-result.json|de7bbe616b67093c674c5a86cee4f76d13c5ad83477442c8b49dbee79987b243|102/672854512|
|48|prefill-48-packed-v3/reference-result.json|01691a654335c4c8f19f87efe8638113947a024518bfd9f204dc69b0a1a6b3fb|102/673440296|
|49|prefill-49-packed-v3/reference-result.json|5243d7d3369db9ae7b3c710994cd87775066f973e832c61e40086d78afc99ba6|102/674026080|
|50|prefill-50-packed-v3/reference-result.json|3979d27ce127b3c27ecf42f860465e8d69f2962d3f211cbd2434e85595b08793|102/674611864|

Outer original owners are prefill-N-packed-owner-v3/result.json.
Each requires its own inner owners and semantic reports as well as outer0.
Each completed plain+observed reference and exact same-request discovery joins:
24dual layer boundaries,42active state domains, full248320 BF16 vocabulary,
independent argmax, consumed priors, natural feedback, parameters before/after,
full image/layout/argument-source identity and cleanup.

Totals:408archives/46156raw records/2694932752bytes;
31008launch packets/156628reflected fields. Each case has8scopes/7752launches.
All four emitted the independently observed token sequence
[248068,198,90700,8340,25,271,16,13]. Matching tokens alone is not the acceptance.

[Independent packed-reference review](continuation/PACKED_REFERENCE_REVIEW.md)
and its original per-case JSONs under execution-prep/packed-reference-review/
join raw archives, source parameters, callbacks and actual owner streams.

### Failed/interrupted predecessors remain distinct

- First original47 controller failed on a report larger than its1MiB reader.
  The underlying original result was separately postvalidated; the outer failure
  was not relabeled. Fresh packed-v3 later supplied a fully closed run.
- First packed47 CPU-tokenizer attempt had a descriptor-reader cleanup failure.
  Tokenization returned0; no model GPU arm ran. The original failing FD was not
  retained. DefaultSelector repairs the likely select-FD limitation without
  claiming that missing observation was recovered.
- Packed47v2 collector INCOMPLETE64 missed six FlashAttention packets.
  It retained complete whole-image/formal reflection but lacked final worker/
  parameter-after receipts and a published small norm archive. The original
  failure stays failed. Only independently joined whole-image/formal layout facts
  entered a fresh retry; no missing packet, output or mathematical receipt was
  synthesized.
- First offline48 emitter refused an entry absent from its old discovery cohort.
  The successor requires the same held whole image, actual exported ELF symbol,
  live prelaunch function, full reflected packet, semantic pointer ranges and
  unchanged scalar classifier. Unknown images or missing source evidence still
  refuse. This is candidate-generation evidence, not a runtime policy.

### Collectors and observer are different artifacts

| Artifact | SHA256 |
|---|---|
| Explicit rebuilt48200B original-source observer | 51140d465ca240ca7bdb9bd3b5a2779945cd9c5a0cd0f60432ce848fdf1f2ad2 |
| Full-ABI build6 collector65128B | 92db60f231bc87c50f18393cb1a03a606f07cad31df48384e42cdb5f8a931132 |

Discovery authenticates42original source files; full-ABI capture authenticates50.
Their staged ancestry and shared20primary model/operator source pins remain
explicit. The discovery collector is not interchangeable with full-ABI build6.
Consult tools/qwen35-cuda-reference/README.md and its original-pins.json.
The current adapter has14files; any earlier README-only isolation change is an
explicit new documentation pin, not a claim that frozen adapter bytes stayed equal.

## Offline candidate generation and source identities

Latest output:
root/native-prefill48-50-candidate-v2/{generated.rs,records.rs,candidate-source.json}.
This generated81factories for actual fresh48/49/50,64semantic calls and
10752non-head sites; metadata61865984B; native limits[17,28,245].
Original factory bytes/site rows and2B baseline facts remain unchanged.
All admission/native/model/performance qualification flags remain false.

| Candidate body | Bytes | SHA256 |
|---|---:|---|
| generated.rs |3600850|07aeb2d5345a30e8af6cec7c9ebb52aa8733373e6b99a60e2379630e73ad3443|
| records.rs |172513|ed73fbee1bc14dbd0bb285505217a812d601a98b9967843b3da166430222f5b4|

The compiler deliberately uses the retained seven-file neutral4f427dd baseline,
under review/recipe-baseline/, plus13-file emitter ancestry and14-file reference
adapter. It must not pretend current post-refactor source is byte-identical:

- Current2B aggregate capture fingerprint changed from
  0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0
  to4d77cb6637843a57a17fc093ad1e5241ddf083bf8d9ade0c9bbaa118bc9448e2
  when an unused generated source-report constant was removed. Direct consumers
  were honestly updated. Executable packets and GDN CUDA payloads did not change.
- The offline neutral baseline still legitimately reports the old2B0938 identity;
  the all-three .8B candidate reports946678a9256d511b345075f9f10deadcaa44d46110055ad9994aab6b183415e7.
  Neither is the aggregate identity of the current production implementation.
- The .8B admitted native capture remains39b9a618…; do not substitute a generated
  candidate pin or a documentation-only alias as runtime authority.
- Original96GDN source tuples/12recipes/2catalogues remain exactly preserved.
  The newer pure history audit also checked123GDN CU/PTX/header payloads unchanged.
  These scopes count different source sets; they are not contradictory.

Do not integrate per-length candidate tables to claim the architecture defect
is fixed. Captured references belong in qualification support; genuine runtime
shapes need actual parameterized kernels/ABI/control/resource support.

## Current rotary change and remaining source debt

The ordinary text loader now calls prepare_text_rotary, with configured
context_capacity_tokens512, and checks completed ownership through the sealed
prefix instead of asserting12test launches,2test readbacks or a512MiB backend.
The original complete1M-position CUDA construction math remains unchanged.

Layout validates actual requested positions, byte arithmetic and source bounds.
Retained host cost is metadata + actual readback admission + three prefix
reservations; at512 it is1310720B. Real native packets presently require10112B
of immutable rotary input; the loader derives this exact extent from admitted
read-only N1/B4 bindings, uploads that real prefix slice and charges it.
The full configured CPU backing remains retained. Every call is bounded by both
the sealed source extent and uploaded lease. Equal byte lengths cannot substitute
FixedPrompt for Text: issuer/service/generation/route-kind checks remain.

This does not expand native accepted context. Other production tables still
restrict shape/call admission. Very large requested spans also encounter existing
32MiB host tickets: producer readback bounds at43463positions, while grouped
cost-only inspection bounds at28969. They refuse before device work. No long
context claim follows from larger rotary backing alone.

Remaining concrete source locations:

- cuda/native/source_rotary.rs still carries shared fault-path assertions and
  diagnostic policy plumbing; qualification.rs wrappers preserve old67/79
  diagnostic modes/statuses.
- Application qwen35_cuda/fixed_prompt.rs still calls its qualifier in loading.
- model-qwen35/plan/source_profile.rs and PromptWorkload admission expose the
  five-prefill/eight-output/512-context matrix to ordinary preparation.
- Model prepare/gdn.rs uses expected capture records and ordinal slices.
- cuda/gdn/text/observed.rs contains executable captured specialization selectors;
  CAPTURES/select and source catalogue remain runtime debt.
- Native packet keys/materializers, eager templates and vendor/cohort call limits
  still repeat finite capture coverage.
- Learned GDN parameter compatibility is tied to an aggregate catalogue. Replace
  it with authenticated operator/role/source/numerical/cache-layout contracts,
  not deletion of checks or a renamed whitelist.

Full audit and proposed parametric design:
[boundary audit](continuation/ARCHITECTURE_BOUNDARY_AUDIT.md),
[native policy audit](continuation/NATIVE_POLICY_AUDIT.md),
[GDN parameters](continuation/GDN_RUNTIME_PARAMETERS.md).
These are copied design/audit bodies; their internal relative references use the
original location listed in continuation/README.md.

### Parametric GDN design: proof, not implementation

The pure reversible source proof recovers all96original full CUDA bodies from
typed scalar operands. Those files contain29unique CUDA wrappers and17PTX bodies;
19parameterized wrappers reduce to17instruction/ABI variants because three
Solve export labels share a body. Ten prefill Conv specializations share one
body. Variant configuration still affects numerical behavior and resource costs.

Proposed transport reuses existing KernelArguments.with_info with a bounded
by-value trailing scalar block when grid_constants is true. A Conv24B block is
u32cache_lines,u32zero-padding,u64input_stride,u64output_stride. It rides the
existing queue/client/flight; do not create a parallel launcher or use
MetadataBindingInfo::custom, which would allocate device metadata.

Actual retained cache layout has5slots. Historical original scalar capacity
492/.8B and157/2B is upstream allocation metadata, not model geometry. A new ABI
derives capacity/strides from the real allocation and validates all control
indices/spans. It gets an honest new source/ABI identity, with legacy facts in
fixtures. No silent5-for492 substitution while claiming an old recipe pin.

KKT/Recompute/H/Output have genuine tile/warp/stage variants. Original tuner
choices are non-monotonic in prompt length; do not infer a threshold policy.
Select a declared numerical/kernel configuration once in cold preparation,
preserve exact epsilon/scale bits and check the complete family identity.
No default variant policy has been implemented or qualified.

The only unique pending worktree overlay is an untested11-file per-length
offline GDN tool,64860B. It is superseded as an integration strategy by this
parametric design and must not be merged blindly. A lossless text snapshot is
[preserved here](continuation/PENDING_GDN_DRAFT.md); originals remain in the
worktree and reconciliation archive.

## Earlier B4/SDK/API/C16 results are source-scoped

Original A3 implementation a857d45 and its frozen optimized binaries are
documented in .agents/runs/b4-takeover-codex-20261025/FINAL_REPORT.md.
They are not current d74f59a inference binaries.

- Correctly independent grouped oracle:12healthy closures,58grouped and232per-row
  launches; resident/events/arbiter0, unfenced. CPU4344 and finite301 pass.
- All26original scopes executed:24pass, one2B export is diagnostic/unqualified,
  and strict single-request-vs-batched cached-decode comparison fails101.
  The rejection deliberately fences before Ready. In-process retirement retains
  2469431001B/pending1/rootsets9/unusable; external process closure does not erase
  it. This is not all26green.
- Original same-prior N1/B4 also differs in six QKVZ BF16 words at frontier47.
  The default accuracy comparison and explicitly promised batch invariance are
  separate requirements. Preserve both independent facts and the original failure.
- Both-family original30 comparisons pass on their declared schedule:15transitions,
  35events,4344current +8112original sections,576unrounded sums/allstate/fullV/
  natural feedback. Same-state seeded composition is not an independent public
  original trajectory.
- SDK observer attempts1–4 remain rejected despite native passes. V5 admits a
  separately proved nonpublishing acquisition discard, then requires a fresh
  simultaneous live snapshot plus held supply checks. It does not prove universal
  continuous ABI equivalence.
- Old API both models24offers/192tokens per provider and HTTP both models
  5offers/36tokens pass with real N1→mixed B4/refill histories and compiler0.

| Historical same-attempt C16 | Stock tok/s | ZeroGravity tok/s | Wall stock/ZG | Result |
|---|---:|---:|---|---|
|.8B A3|98.2827|111.1738|1.30237/1.15135s|Rate and wall pass|
|2B A3|96.8627|84.1017|1.32146/1.52197s|Rate gate fails; pair/final incomplete|

Each arm completed16offers/128tokens/0request errors, client peak16/native8;
all four process/container/group/reap/EOF closures pass. One measured wave/model
is not a confidence-certified result. The failed2B pair cannot be reported as a
both-model win. The earlier source21 container incident is a different attempt.
No current-source matched C16, C2, C32 or C64 result is established.

## Source, SDK, hardware and model identities

Fresh read-only handoff observation on26Oct: pvl-l5, kernel7.2.9,
RTX3060 Laptop6144MiB, PCI0000:01:00.0,
UUID GPU-89d28939-1000-40b8-e243-b18f72b3c381, driver595.91.07,
41MiB reported used and no listed compute applications. This is a snapshot,
not a reservation or proof that a future benchmark is quiet. No GPU kernel ran
for this documentation task. Local AMD Renoir/gfx90c launch/recovery hold remains.

| Exact runtime supply | SHA256 |
|---|---|
| libcuda595.91.07 | dd2e55efb057b7846c39ef71c0eee1c4f028286d4e207b470e1a05163b86893f |
| Nix native NVRTC12.9.86 | 4b8c4db1354ffc8436905e87ed0b8a0f79cf398864f6571df559eb2945c32c90 |
| Nix native builtins12.9.86 | 62b65fb5bcc42872e2bf69bae64a47dd770928dd57cbe0c569734078061aefc6 |

Native NVRTC directory:
/nix/store/cs50fahgnln1k5cx7zs0dnyikzqzf4iq-cuda12.9-cuda_nvrtc-12.9.86-lib/lib.
Read root/native-nvrtc-fresh-hashes.json for exact files/resolution/stamps.
The88-row sdk-fresh-hashes.json includes the earlier overlay NVRTC identity;
do not confuse it with native Nix runtime identity. cudarc's compiled
cuda-13040 binding selection is another independent identity.

| Model | Exact approved source |
|---|---|
| Qwen3.5 .8B | /home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 |
| Qwen3.5 2B | /home/pvl/.cache/abird/zerogravity/models/snapshots/88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52/files |

.8B revision2fc06364715b967f1860aea9cf38778875588b17,
source manifestc4c1d349743c0e0ba51ab07f769348e6b13f9896c354303c7af5e839c11dbf09.
2B revision15852e8c16360a2fea060d615a32b45270f8a8fc.
All23original members,6341189643B, were freshly hashed in the continuation and
revalidated by original references; see A3 physical/a3-protocol-{08,2b}.json and
the fresh original reference held inputs. Historical l5 sandbox read denial
was overcome legitimately; there was no broad grant relaxation or model download.
Recheck actual permissions before reuse. Inference still needs all source gates.

Current-schema native .8B/2B N1 bundles are root/current-native-bundles/{0.8b,2b}/.
All resource bytes were preserved. The extra2B B4 image5857056B,
SHA ed95d9920ecf50d7a50a5f94e1986fec41a2555c683fabbe148bc36de40d01f5,
was located but not staged into a fresh complete B4 bundle.

## Serving shutdown and the historical container incident

Canonical incident:
[l5 benchmark container termination](../../plans/cpu-gpu-optimization/BENCHMARK_CONTAINER_TERMINATION.md).
Original local l5 .8B C16 PID1 server had no serving shutdown handler; owned
cleanup invoked stop with5seconds. Corrected journal evidence places container
death5.288610s after the stop actor. Cleanup escalation is the supported leading
inference; fatal signal, sender and OOM remain unproven.

Source50abaf7 handles TERM/INT by stopping admission, interrupting work, joining
HTTP handlers/workers and reporting physical retirement failure.
c84b987 retains actual stop intent/clocks/streams/terminal state; cd4159d hardens
generic command ownership. Four actual process/PID1 controls and a cached-image,
model-free Podman PID1 stop control pass0, ledger0/no escalation/exactCIDabsent.
Evidence: .agents/runs/serving-shutdown-20261026/root/validation.json.

That preventive source fix is complete. Fresh trained-model/backend shutdown
and fault scopes remain mandatory before a new applicable benchmark. The old
incident cannot be reclassified as confirmed OOM or solved historical signal
provenance. It is local l5 evidence, not an x2/Qwen27 capture failure.

## Qwen27B/x2: last authenticated preflight, no capture

Read .agents/runs/b4-takeover-codex-20261025/x2-current-preflight/root-resource-review.json,
the Qwen27 next-plan directory in that run and
the owning cpu-gpu-optimization TODO. No x2 query ran for this documentation task.

- Strict existing ED25519 alias/address trust was actually verified:
  SHA256:acPQLnW/MfwchrrYkM/TB9eVeTWrnaA5efNPrEJZ2U0.
  Existing known_hosts remained unchanged. Prescribed100.100.1.1/tailscale0 route;
  native UID1000. Old missing-key prose is superseded, but revalidation remains.
- Host identity: machine8280a2085c2247e4bb49a5ac1bcd1582;
  boot40a8634b-bd7c-4cb0-b127-d0fde2a4f37d, kernel7.2.10, AMD MAX+395.
- Last observed CPU-visible RAM33263382528B total,15343521792B available,
  below the16GiB gate. Sysfs VRAM103079215104B is separate; never double-count.
  Disk available1769526714368B. Capacity and ownership are time-sensitive.
- Exact model source parent and31members, and exact image store/runroot, were
  absent. A documented Nix Python3.13 was readable/executable and hashed, not
  installed or executed; prescribed PATH python3 was missing.
- Current owner source_open requires state, while the independent frozen validator
  supplies only plan,pins. Source caller/validator compatibility is an unresolved
  actual ABI blocker. Original9frozen files stay separate from the current caller.
- Prior authorization for the exact image/31-source establishment workflow exists;
  inspect that precise authority before action. It is not broad download,
  driver/privilege/memory repartition/recovery approval.
- Q27 source31 snapshot98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e,
  revision1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0,
  source-pin SHA4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a.
  Metadata is not established source or inference.

Do not start the original natural full-window capture until current sequence,
exact source/image/tool/validator compatibility, genuine>=16GiB/4CPU/GPU-owner
window and all original G1–G6 gates hold. Do not substitute a short or teacher
capture. This reference would not automatically qualify ZeroGravity27B serving.

## Supervision, isolation and commands

Never execute an old hardcoded build controller merely because it is imported
for its lifecycle helpers. Current generic driver:
.agents/runs/numerics-and-context-sequence-root-20261026/execution-prep/supervise.py.
It authenticates the earlier owner helper SHA
2db3137ee5a283df288d4b9ee2ec26b2f9374e7819c631bdcb141c7b1a1ce7f7
and lifetime helper SHA
e4f6dea9d10fb9bdbb3e98d4aff986ab7e3a6dbf83ce648f02f76d87b3ce08cd,
and requires exact argv/pin JSON digests, a fresh absolute output, explicit
timeout1..7200, README integrity, before/after input identities, CPUs0–3 and
group TERM/KILL-before-one-reap/full EOF. Imports do not execute its old build.

Inspect it before reuse; if local helpers are absent, establish a reviewed
supervised route rather than fabricating the old paths or detaching a job.
Use exec/write_stdin on a live owned tool session and retrieve actual completion.
An agent session ID alone is not an ownership transfer.

Source/build gate:

~~~bash
cd /home/pvl/spaces/abird/src/zerogravity
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  CARGO_NET_OFFLINE=true python3 -B scripts/native-gpu.py -- bash scripts/check.sh
~~~

The current broad argv also sets ZG_REFERENCE_SOURCE and ZG_CAPTURE_SOURCE to
the exact retained original-source/capture-source packages. Use
root/rotary-full-source-command.json for their actual absolute paths.
Strict CUDA features and checked native library environment are in
root/rotary-source-batch-command.json; affected host argv is
root/rotary-cuda-host-command.json. Inspect current Cargo features before reuse:
older handoff feature names were renamed.

For real original model runs, rootless Podman provides its own user/network
isolation. Do not nest it under native-gpu.py's mapped user namespace.
Preserve original model ownership, exact containers/CIDs/nonces and both complete
streams. A successful cleanup command is not mathematical acceptance.

An optimized inference build must use actual canonical release settings:
opt-level3, thin LTO, codegen-units1, stripped symbols; no diagnostic environment
overrides. Builds and physical tests are owned/serialized; quiet benchmarks have
no competing compilation, model/source scan or observer sampler. Validate exact
model/binary/source/driver/SDK/ABI identities and terminal results per scope.

## Worktree custody and next larger unit

MAIN and the sole side worktree both reached d74f59a, divergence0/0.
Main tracked/index clean. Side tracked/index clean; only the11-file GDN draft
is untracked and unstashed. No stashes, conflicts, unique side commits or active
build/model/device sessions remain.36prior overlay files933551B were archived
before the single post-batch ff-only reconciliation. Branch refs remain.

Evidence:
[worktree result](continuation/WORKTREE_RESULT.json),
original execution-prep/worktree-reconciliation/final-d74f59a/{manifest.json,
result.json,commands.json,overlay/,displaced-overlap/}.
ROOT independently checked all11pending bytes/modes against the archive.
The side tree must not be deleted as useless while its unique draft is pending.

The next source unit is the actual parametric GDN/normal-execution boundary,
including honest model admission and shared rotary/fault separation, across
the existing model targets. Reuse the one runtime, physical owners and scalar
transport. Finish MAIN's substantive batch, fix major refactor bugs, validate
affected semantics/ownership and the assembled gate, then sign scoped commits.
Only afterward reconcile side work once. The larger correctness/performance
sequence and all other pending work are in OPEN_WORK.md and the parent handoff.
