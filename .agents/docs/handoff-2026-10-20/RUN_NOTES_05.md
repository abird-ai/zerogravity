# Run Notes — volume 5

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/serving-and-speculation-cuda-graph-design-20261020/DESIGN.md

Original UTF-8 bytes: 15797; SHA256 `829f7f96eb8c7d9793dc0c86a930feea5b77e6aa84e7628b2a8e10527b5d270c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# BatchProgram CUDA Graph replay: first safe slice

Source-only proposal; no implementation/build/operator/model/GPU execution.
Owner: spine_review. Parent: serving-and-speculation P12. ROOT alone decides,
merges, runs gates and reconciles the owning plan. Existing full-HF guards stay
closed; this changes host dispatch of already accepted GGUF GPU mathematics.

## Recommendation

Capture ONLY the stable original BatchProgram node body. Keep the current
per-session gather/scatter, fresh candidate BufferLeases, input token/position
staging, sampler, status readback, Ready and atomic publisher outside the graph.
The existing packed prior/candidate workspace already solves device-pointer
stability; do not allocate a second B-shaped state staging pair.

First gate: private diagnostic B2, capacity512, Logits readout, max source AV
padded length256, one graph per currently live batch workspace. Full248320
logits/all42 domains against accepted CPU+B1 and ordinary native B2. No HTTP
mode or automatic graph fallback until this completes. Extend to StateOnly,
greedy/device feedback, padded512, B4/B8 and actual worker/HTTP only afterward.

## Actual source and API

Backend is the repository patch in
`crates/zerogravity-gpu/vendor/cubecl-cuda`, not upstream/latest assumptions:

- `src/compute/capture.rs:28`: `cuStreamBeginCapture_v2(...GLOBAL)`.
- `capture.rs:44-78`: instantiate/upload/replay; replay calls cuGraphLaunch.
- `capture.rs:96-133`: end capture on refusal, reject allocation/free nodes,
  instantiate and destroy the temporary graph. Node-query errors currently
  warn and skip that extra memory-node check; do not claim it is fail-closed.
- `src/compute/server.rs:261-310`: prepare/start/stop, recorded write-set
  ownership, replay through ExecuteScope. Replay resolves no dynamic inputs.
- `server.rs:311-335`: Graph destruction synchronizes its capture stream.
- `src/compute/stream.rs:27-44` and `server.rs:43-78`: capture lifecycle and
  exact-word metadata cache; warmup must populate identical metadata keys.
- `tests/graph.rs`: real-device tests cover normal replay, changed input in
  the same buffer, allocation refusal, intermediate page retention,150-launch
  dynamic metadata and cross-client capture visibility. These are existing
  source fixtures, not freshly executed evidence from this design task.

Cargo patches CUDA/HIP only; exact resolved0.11 shared types were also read at
`/home/pvl/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`:

- `cubecl-runtime-0.11.0/src/client.rs:46-160`: public Graph is cloneable,
  captures fixed pointers and StreamId; replay returns after enqueue, NOT Ready.
- Same file243-293: an unpinned Client uses StreamId::current(); unsafe
  set_stream pins it. Same file1331-1396: mandatory prepare/warm/start/stop;
  host writes/read/sync/profile during recording refuse.
- `cubecl-server-0.11.0/src/command/capture.rs:83-102,209-363`: Graph retains
  page guards, host-copy payloads and write set; warm prepare can relocate
  device pools, stop pins touched pages/info cache and drains deferred drops.

No public graph parameter-update API is exposed here. Do not reach through raw
CUDA pointers/GraphId, patch captured node arguments, or introduce another queue.

## Why whole launch_impl capture is wrong

`resident_program/batch.rs:621-696` gathers/scatters independent session handles
to/from fixed packed state. Those external handles change at every commit and
between cohorts; capturing them replays against an old request/root.

`batch.rs:450-520` clears status and feeds teacher token/device token and
positions. Teacher tokens and positions are host scalar launch arguments here;
capturing those launches freezes old values. Device-token gather also binds
each current packet lease. Leave these operations outside, writing the SAME
stable inner.token/positions handles before every replay.

`batch.rs:1081-1144` chooses attention-value iteration count on the host:
max over rows of ceil(valid_tokens/padding)*padding/32. The accepted cap512 and
padding256 give two graph topologies: padded256 or padded512. Exact per-row
positions remain device data. Never replay the larger topology as a substitute
for the smaller: inactive-FMA bit carry and signed zero are part of accepted
source mathematics. Existing row kernels preserve each row's own padded bound.

The fallback inner.launch_node receives parameters[0], but every position-using
operation is overridden by the batch path: RoPE, KV append, QK, softmax and AV.
Capture eligibility must explicitly enumerate the reviewed legacy operations;
new/typed/unreviewed operations refuse cold, rather than inherit an assumed
position-independent fallback.

`resident_inference/fused.rs` allocates one fresh12-byte greedy packet per row.
Keep the original4-launch sampler outside, writing those actual packets. Do not
add stable packet buffers or replace producer-root binding just to capture it.

## Minimal factoring, same arithmetic order

Split the existing synchronous BatchProgram launch into private phases:

1. Validate all original rows, domains, root/producer/source/service/epoch,
   token IDs, capacity and graph key; physical admission before any submission.
2. Original status clear, token/position staging and packed-prior gather.
3. Original node-prefix traversal and each node's existing certification,
   exactly in current order. This is the captured body.
4. Original packed-candidate scatter, per-candidate certification, quant and
   embedding status merges, plus existing sampler when requested.
5. Original status/logit/packet readback, complete_cohort_with_readback,
   slot.finish and independent checked all-root publication.

Retain all current external-domain proposal/handle checks. Body returns a
private immutable receipt containing its kernel-launch count and reviewed key;
the outside phases retain their actual counts. Do not make Graph construct a
ResidentReady, RootBinding, SessionAuthority or published state.

Private key minimum:
source identity + service generation + exact workspace generation/owned handles
+ B + capacity + readout node-prefix + AV max padded length + original plan/
numerical/prepared-block identity. Bound fields are derived from current
trusted objects, never accepted from caller hashes. Roots, token IDs and exact
frontiers are validated per replay but are not graph-cache keys.

## Mandatory stream and capture exclusion

Graph always replays on its capture StreamId. The service Client is currently
unpinned: loader-thread capture followed by worker-thread gather/replay/scatter
can silently race because StreamId::current() changes. Client identity alone
does not prove same-stream ordering.

First diagnostic may capture and replay entirely on one fixed thread. For a
shared production owner, pin the ORIGINAL service Client once to an explicit
capture stream and use that same client for every preparation, input refresh,
outside kernel, graph replay and physical completion/readback. Never make a
second client/queue whose work escapes the existing ResidentService. A new
clone pinned only for graph.replay while readback stays on the old unpinned
client is insufficient.

GLOBAL capture also requires quiescence against allocations/syncs from ALL
same-card clients. Current queue serializes submitted regions, but other
workers can allocate candidates before queue selection, and per-model
allocation locks do not exclude another model. A selected slot by itself is
not a capture barrier. First slice runs cold with no active workers/other
model construction. A later live-capture path needs a coordinator-owned
per-card lifecycle barrier covering allocation/upload/sync/destruction entry
points, following the existing allocation→queue→workspace order. Do not wait
for this barrier while holding a lock needed by an already selected B1 job.

Prepare/warm/capture must be a bounded owning transaction. Prepare marks a
capture active before recording; even warmup failure needs stop_capture to
close the prepared state. A private RAII guard closes only its own window on
every error/unwind. No await inside the actual start→record→stop section.
No source upload, candidate allocation, host copy, readback, graph destruction,
profiling or info-cache miss is allowed inside the recording.

Warmup is real work against private packed state, never a session commit.
Drive it through the original physical queue/work/readback retirement and
report its cold dispatches separately. Refeed current prior/input state before
the first real replay; recording itself executes zero model kernels. Capture
failure is an error with original taint/fence semantics, not an admitted None
or permission to silently rerun. Do not recover uncertain work on a timeout.

## Liveness and quarantine

Capture owns references to every original source/workspace ResidentLease it
uses, including status/info/prepared-block dependencies. Vendor PageGuards
prevent physical reuse but are not the engine's BufferLease, root or Ready.
Every replay flight still declares the full original read/write accesses:
shared weights, stable workspace, independent prior/candidates and any packets.

Add a narrow trusted graph-executable keepalive to the EXISTING in-flight/
physical-quarantine ownership, not another global recovery mechanism. A last
Graph clone Drop schedules synchronization/destruction; on unknown replay
completion the executable must remain alive together with all original lease
handles and provider envelope until process exit/trusted future recovery.
Current ResidentSubmission Drop retains handles/service, not an arbitrary
Graph executable: simply putting Graph inside BatchWorkspace is insufficient
if the model/workspace is dropped after an aborted replay.

Known Ready cancellation drops private candidate results and leaves the graph
cache reusable after actual terminal completion; fresh roots/packets are
revalidated on retry. No subset publication. Replay enqueue failure after
outside gather has started follows original unknown-work fencing; do not infer
whole-wave safety from the CUDA launch call alone saying it rejected the graph.

Bind graph lifetime to the CURRENT batch workspace. Existing capacity logic
adds an idle band's bytes as reclaimable before dropping it. A graph cache
holding the old band's leases/pages invalidates that assumption. Destroy and
establish terminal graph retirement before reclaiming its budget; do not cache
retired B2/B4/B8 workspaces behind the admission code. Normal graph destruction
must occur with no replay in flight and outside locks that its stream sync can
need. Unknown graphs are quarantined and never counted reclaimable.

## Footprint and finite limits

Extra explicit source/state/token/position/logit/packet device bytes for BODY
capture: ZERO; all are existing admitted buffers. Lease clones do not create
another raw payload allocation. They do extend lifetime, which must be visible
to existing resident capacity/reclamation accounting.

Let S=the actual original per-row state_bytes from preflight. Existing packed
workspace already charges2*B*S (prior+candidate), independent old/new roots
already own2*B*S, and queue staging admission remains B*S. For the accepted
24layer/16head128 GDN +6KV cap512 geometry, the shape-derived S is32,784,384B
(confirm against the live preflight, do not hardcode it as physical evidence):

| B | packed existing prior+candidate | fresh independent candidate |
|---|---:|---:|
| 2 |131,137,536B|65,568,768B|
| 4 |262,275,072B|131,137,536B|
| 8 |524,550,144B|262,275,072B|

Existing input parameters are8*B bytes; full logits are4*B*248320 bytes;
greedy packets12*B outside the graph. Retain original provider4GiB/device5GiB
caps, ResourceArbiter and all preflight refusal behavior. Graph replay is not
permission to widen them or spend another provider's reservation.

Graph executable, graph/page-guard vectors, retained info-cache entries and
CUDA driver internal allocations are ADDITIONAL control-plane footprint, not
zero. Public Graph exposes no exact executable-byte accounting/parameter update
or node-count receipt. Do not invent a metadata credit or claim all-VRAM/RAM
accounting from the zero explicit-tensor delta. Charge known owned host vector
capacities; first gate retains only ONE bounded graph, bounded plan/launch count,
and records actual host/CubeCL pool/NVIDIA memory peaks separately. Production
admission must explicitly reconcile persistent metadata/page retention with
the established provider envelope and actual available device headroom; this
is a qualification requirement, not solved by merely calling graph_prepare.

Later ceiling is at most4 graph shapes per LIVE band (2readouts×2AVpaddings),
not an unbounded per-root/token cache. Start with1 and measure before expansion.
The vendor's warning-only memory-node-query failure is an additional boundary:
all launches must be allocation-free by construction/warm cache, and a stronger
capture-allocation guarantee would need that query path to report/refuse errors.

## Counters and first physical acceptance

- On recording:0executed model kernels/row tokens/committed roots. Keep capture
  attempts, successful seals and warmup actual dispatches in cold diagnostics.
- On completed replay: add captured body kernel-launch count K plus actual
  outside counts to the existing dispatch receipt. A graph is not one kernel.
  Report graph host-enqueue calls separately:1replay enqueue versus Kkernel
  launch sites. Keep counter scope explicit unless actual driver node tracing
  independently proves it. Sampler4launches remain outside and fully counted.
- Preserve separate physically completed work, discarded Ready cohorts,
  committed roots and emitted tokens. Known discarded graph work still counts.

Root gate sequence:

1. Source-free key/coverage/refusal tests, lease/graph DropSpy quarantine tests,
   no replay on stale workspace/source/generation/wrong padding/readout/backend.
2. Real original0.8B B2/logits, same cap/source/raw kernels. First warm+record
   then change actual token/prior inputs before replay to prevent stale warmup
   bytes from passing. Every full-vocab word and all42 domains versus CPU/B1
   and ordinary native B2; multiple fresh sessions and swapped ordering.
3. Unequal frontiers in padded256; raw signed-zero/state tails; frontier256→257
   requires a separate padded512 graph or a PRELAUNCH refusal. Full StateOnly
   next-logit continuity and greedy/device-feedback producer bindings afterward.
4. Cancellation before launch, after actual Ready, atomic rollback, physical
   pool pressure/prelaunch None, originalqueue fairness, stale packet/root and
   abort-after-enqueue quarantine including executable lifetime. No recovery.
5. B4/B8 and2B same full-logit/all42 matrices; graph destruction/band eviction
   and final owned resource0 after known completion. Persistent metadata/pool
   retention measured and accounted before a production enable.
6. Strict/focused/full isolated gate, both accepted cache parity routes, then
   real worker+HTTP metrics and same-source before/after ABBA. No vLLM-equivalent
   throughput, matched vLLM result or original-HF math promotion from this gate.

## Disjoint implementation ownership suggestion

- Batch owner: private phase factoring in resident_program/batch.rs + new
  batch_graph child; unchanged mathematical kernels/prepared-block schedule.
- Root: ResidentService stream ownership, fused inference orchestration,
  original submission/quarantine executable pin and bounded cache retirement.
- Independent qualifier: new graph-only physical child and source-free owner/
  key/refusal fixtures; sole ROOT Cargo/GPU coordinator executes everything.

No plan/shared-source/README changes are part of this design preparation.

````

## Source: .agents/runs/serving-and-speculation-cuda-graph-design-20261020/TODO.md

Original UTF-8 bytes: 785; SHA256 `284d6c2f6279847e4067962a7402367869a2e673e85674e01102da82d7e0e3fb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUDA Graph replay design

- Parent: serving-and-speculation P12; owner: spine_review.
- Own: this ignored design run only; no production/shared-plan edits.
- [x] Read actual vendored CUDA capture/server/stream/physical graph tests.
- [x] Read exact resolved0.11 shared Client/Window source and current batch,
  queue, leases/quarantine paths; identify stream/global-capture/lifetime traps.
- [x] Prepare bounded B2 body-only slice, existing-buffer footprint/key/counts,
  ownership suggestions and physical acceptance boundaries in DESIGN.md.
- [ ] ROOT decision/review; implementation and every runtime gate remain open.

No build/Cargo/rustc/C++/native/operator/model/GPU/container/benchmark execution,
large payload/DSO hash scans, MAIN source edits, commits or publication here.

````

## Source: .agents/runs/serving-and-speculation-cuda-prebegin-retirement-20261020/DESIGN_DRAFT.md

Original UTF-8 bytes: 2534; SHA256 `ab1f204cfc05e80311f385aeefe949403121ae339c3c15784ff43aaed2ff2409`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pre-begin token draft — SOURCE DESIGN ONLY

Author: /root/spine_review. No implementation or acceptance.

- New move-only PreparedCapture owns preallocated original GraphHandle/GraphId,
  original DeviceHandle/explicit pinned stream and armed retirement slot BEFORE
  any begin. Private Graph cannot escape until RecordingCapture.stop succeeds.
- Client prepares task/slot and inserts an empty driver owner into SAME original
  Captures graph map before begin. Prepared.start consumes into Recording;
  Recording.stop consumes into public Graph. No Drop makes a driver call.
- Existing None start/stop/private qualifier APIs preserve original path. A
  registered gate requires explicit typed token; no late prepay inside capture.
- New Server prepare/begin/end prepaid methods default cold-refuse. Generic
  GraphDriver new optional methods default-refuse; no new model/math engine.
- SAME Captures entry holds original executable plus page/host/info/write owners
  across partial begin/end, logical refusal, panic, and unknown driver statuses.
- CUDA Executable needs distinct legacy and prepaid states. Prepaid container
  owns actual CUgraph/CUgraphExec outputs immediately after each call, before
  checking its status. Mark unknown BEFORE any driver effect; clear raw pointers
  only on checked successful explicit destruction. Prepaid Drop is driver-free.
- Generic Window prepaid seal uses original recorded data/math, transfers touched
  pages/host/write owners before driver end/instantiate. Unknown never drains
  drop queues/discards pins or destroys native objects. Known logical refusal
  with successfully ended raw graph can retire later on original actor.
- Checked cleanup of Prepared is pure/no driver; Recording/unknown refuse BEFORE
  sync; sealed uses original trusted barrier then checked explicit destruction.
- CUDA graph command must split original ctx/streams/graphs borrows directly or
  factor command construction; never remove the graph map entry into a temporary
  whose panic Drop could release unknown source/page owners.

Expected source paths: common handle/mod; runtime client +server/base; server
command/capture; CUDA compute/capture +server; own host/native-owner tests.
Actual QueueState cutoff/fairness/owned continuation, explicit event origin and
failed-wait destruction, actual storage/free kind, bootstrap and final service/
stream/context teardown still block complete production capture. None of this
issues Ready/CudaComplete/SessionAuthority or creates a second physical tracker.

````

## Source: .agents/runs/serving-and-speculation-cuda-prebegin-retirement-20261020/TODO.md

Original UTF-8 bytes: 945; SHA256 `398b238698269426c850c744614df8fd5baaae1197cf7979466a422ad6760777`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pre-begin retirement continuation — interrupted source design

Parent: serving-and-speculation. Owner: /root/spine_review.
No source implementation/build/runtime/MAIN changes. Parent reprioritized official
GGUF-plugin audit before edits; this is a continuation note, not qualification.

- [x] ROOT reviewed frozen retirement12 patch d4e875f7 without source blocker,
  under explicit closed/no-issuer boundary.
- [x] Review original Client/Server/Window/CUDA temporary graph paths.
- [x] ROOT style followup source review no blocker: docs/BootstrapAdmission alias/
  letchain only. Keep original firsthook/retirement freezes immutable.
- [ ] Implement explicit prepared capture token before begin_capture.
- [ ] Independent review/ROOT compile/tests; actual sameQueueState bridge,
  events/storage/bootstrap/final teardown and Complete issuer remain absent.

ROOT reported actual exact14 harness PASS, later strict/no_std running; host-only.

````

## Source: .agents/runs/serving-and-speculation-cuda-retirement-spine-20261020/CALLERS.md

Original UTF-8 bytes: 3188; SHA256 `338d707e86545ffedf221b7a0c6afeaf401f8f77fb02c35f39c7f5ad1d84b1ab`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Exact caller continuation map

Author: `/root/spine_review`. All physical completion/complete-participation rows
remain UNPROVEN. Paths below refer to actual pinned source in source-pins.json;
CUDA prefix is crates/zerogravity-gpu/vendor/cubecl-cuda/src.

| Owner/path | Source slice | Required continuation |
|---|---|---|
| common device/handle/retirement.rs | New fixed64 prepaid slots, move-only handles, original-quarantine callback | Actual QueueState authority, cutoff fairness, prepaid control charge, real actor binding |
| common device/handle/channel.rs custom Server | Same runner pumps after whole Task::run and while idle | Pre-close actor barrier/ack, final service detach before TLS drop, shutdown/unknown tests |
| common device/handle/base/mod/admission +device/base | Optional preparation/default Unproven | Original physical registration, actual cancellation/epoch/reentry and service/stream population seal |
| runtime client GraphHandle/stop_capture | Reserve/arm before returned-executable construction; last Drop publishes | Move prepay before begin_capture; error/temporary graph ownership; explicit prepared capture token |
| runtime server/base +server command/capture | Default-refusing checked destroy, map/pins retained on errors | All drivers and private error/abandonment destruction participate or refuse Complete |
| CUDA compute/capture +server graph_destroy_checked | Checked graph pointer destroy after successful original sync | Temporary graph/exec failures, all event/fence calls and original completion continuation |
| GPU physical_queue.rs /physical_quarantine.rs | No edits | SAME original QueueState/capture ticket/ResourceArbiter bridge; existing quarantine retains actual mailbox/service owners |
| GPU resident_leases init/allocate/upload/sync +resident weights/BufferLease Drop | No edits | Permit before actual physical effects; actual storage lease retained through known free or quarantine |
| CUDA compute/events.rs +server device_events/{base,event,fence,profiler} | No edits | Explicit authentic origin/prepaid event destroy; foreign wait/elapsed/GC, failed-wait destruction |
| server command/read_async +stream/event ResolvedStreams::Drop +drop_queue | No edits | Source/destination/future/stream owner through actual GPU terminal; no late fresh admission during close |
| CUDA compute/storage/{gpu,cpu} +pool relocation/drop drain | No edits | Exact async/sync alloc/free kind, pinned memory, relocation/flush/fallback paths before driver calls |
| CUDA compute/stream.rs +runtime.rs +server new/final drop | No edits | Pre-context original registration, stream create/final release and primary-context ownership/errors |
| common normal-channel/mutex/reentrant/custom transports | Unsupported retirement, default None unchanged | Actual prepaid original-executor mechanism or cold complete refusal; no ordinary enqueue from Drop |
| original OwnedCudaCommand /external same-card HIP/Vulkan | No edits | Original actor/native event/stream gate or cold complete refusal |

No new CUDA context, executor, physical registry, model loop, shader, numerical
profile, source identity, SessionAuthority publisher or byte-budget ledger.

````

## Source: .agents/runs/serving-and-speculation-cuda-retirement-spine-20261020/DESIGN.md

Original UTF-8 bytes: 8535; SHA256 `215893d7dc88d0e3ca723feafa15ab05a196f2c97cd7643a978b93c9bf084293`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Retirement integration design

Author: `/root/spine_review`. SOURCE ONLY. Preserve original math/queue/flight/
CudaComplete/SessionAuthority publishers. This does not establish GPU Ready.

## Implemented source slice

`DeviceHandle::prepare_retirement` asks the already cached original hook for a
retirement authority. None returns before allocating task or mailbox. Hook
presence alone is insufficient: default authority is Unproven; unsupported
transports refuse. Explicit service/Client stream metadata is not physical proof.

The custom channel lazily allocates 64 slots on its existing State. A slot owns
request, original authority and a preallocated task Arc. Reservation consumes a
checked generation/free bit before construction. Unarmed cancellation frees only
resources that must not yet have reached a driver. Arm occurs before end_capture;
its Drop publishes with atomics, never ordinary submit, queue-full wait, mutex,
allocation or driver call. Capacity refusal occurs before construction. Sixty-four
is this slice's explicit cap, NOT a claim that every model/event/pool fits it.

The original Server pumps at loop head and after Task::run, after the RefCell
service borrow has ended. A nonblocking atomic pump guard encodes exclusive
payload access even under accidental reentry. Each published slot either parks
unchanged, receives owned original cleanup work, or enters original quarantine.
Known terminal cleanup clears/free-reuses the slot only after the task returns
and the original execution owner reports terminal. Error/panic does not retry.
The whole bounded mailbox, original actor and all task resources stay retained
through the existing completion-quarantine owner type. A violated unsafe bridge
that panics while retaining uses a conservative Arc leak fallback; that fallback
is NOT a physical fence proof or recovery mechanism.

GraphRetirement owns the actual original DeviceHandle and explicit stream. Only
its checked actor call may clear the actual returned GraphId. An armed capture-end
failure with no returned id is UNKNOWN, not known no-resource. The runtime Server
checked method defaults to refusal; CUDA waits first and aborts on ANY wait error.
The shared graph map retains executable, host bytes and page/info pins until the
explicit driver destroy returns success. CUDA clears its private pointer only on
that success, preventing a second destroy from later Drop. Existing legacy None
paths and ordinary/native APIs remain source-equivalent.

## Next actual QueueState bridge — required, not implemented

1. Store retirement registration/close cutoff obligations in the SAME original
   QueueState, tied to actual ResourceArbiter physical key/epoch and registered
   actor. Do not create a second card registry, scheduler or resource ledger.
2. Capture uses its original fair ticket. Under the original queue mutex it
   closes new admission and records actor drain obligations. Release the mutex
   before contacting any actor. A prepaid actor barrier/pump acknowledges a
   bounded publication cutoff; each eligible retirement joins existing original
   work accounting before that actor's cutoff acknowledgement.
3. Before-close work and explicitly admitted drain continuations can execute
   while Closing; after-cutoff publications park with owners intact. Capture
   waits for cutoff acknowledgements AND all original trusted work terminals.
   Do not wait for an ActiveSlot required by cleanup while capture holds that
   same slot. Fair next-open admission must service pending retirement/cancelled
   construction obligations before repeated captures can starve them.
4. RetirementExecutionOwner::enter installs the exact actual actor scope, so
   the nested original checked call uses prepaid reentry, never new admission.
   Actor/key/epoch checks come from the actual registration, not numeric IDs.
5. Host callback return is not GPU completion. Existing initialization and
   ResidentSubmission flights retain original work through trusted event/readback
   and CudaComplete. A waiter needed by Closing has a pre-admitted continuation;
   it cannot request a fresh shared permit and wait behind its own capture.
6. Unknown fences the original physical key and transfers pre-admitted owners to
   original quarantine. It never calls a driver, waits for its own actor, retries
   destruction or declares recovery. Quarantine control storage must itself be
   prepaid; this proposal's task/mailbox bytes are not a VRAM/RAM attestation.

The pure QueueState 14 host tests and three intended compile negatives passed
in ROOT's isolated tree (reported result305070096Ba7a76529); this is host protocol
evidence only. It does not instantiate any of the unsafe retirement bridges.

## Actual remaining construction/completion/destruction plumbing

- **Capture before begin:** this slice prepays the returned graph before
  end_capture. Temporary CUgraph/CUgraphExec objects, failed instantiate paths and
  capture beginning itself still need an original pre-begin reserved owner.
  Prefer a move-only prepared capture/retirement token consumed by the original
  start/stop lifecycle, not a global map or a new reservation inside Capturing.
- **Events:** current EventApi has static methods/raw Copy stream; CudaEvent owns
  only a raw event. Event::Drop can call the driver from a host/GC thread, and a
  failed EventFence::wait_sync still drops/destroys the event. Checked graph
  cleanup does NOT cure this escape. Add explicit sealed original event origin
  and pre-created retirement/continuation before event_create, record/wait/elapsed/
  stream_wait and fallback synchronize. Thread/ordinal/ambient context lookup is
  insufficient. All generic fence/profiler/GC callers must propagate the origin.
- **Storage:** reserve actual allocation retirement before async or sync-fallback
  alloc and before pinned alloc. Preserve exact alloc/free kind, actual pointer/
  byte range/original stream and original backing resource lease. Populate the
  preallocated task only from actual successful constructor results. Pool logical
  dealloc/relocation/free/flush and every fallback must use the same owner. Known
  native free may release charge; unknown retains it. Managed handle Drop is not
  itself proof that physical storage was freed.
- **Final service/stream/context:** internal resources cannot all hold strong
  Clients forever (service -> task -> Client -> actor cycles). Explicit original
  actor shutdown must detach/retire service resources while the runner/pump still
  exists, then drain retirement before thread-local service destructors/stream/
  context release. A destructor cannot try to weak-upgrade a dead runner or join
  itself. Unknown retains original final service/storage owners in quarantine.
- **Bootstrap:** authenticate the existing physical registration before CUDA
  initialization/context/stream construction. Complete admission must forbid an
  unknown same-card actor from silently taking the legacy None path. DeviceId or
  CUDA ordinal alone is not a physical binding. Seal actual participating actors
  only after constructors/errors/final drop and alternate runtimes are covered.
- **Alternate transports:** dormant normal-channel, mutex/reentrant retirement,
  external pinned commands, same-card Vulkan/HIP/custom actors need coverage or
  cold Complete refusal. Default None stays legacy; no production issuer exists.

External-process exclusion remains an environmental qualification precondition.
No host mailbox can guarantee exclusion from other processes or GPU recovery.

## Required ROOT gates before any production close

- Compile source plus default/no_std native checks; execute eleven prepared host
  tests, exact firsthook14 and all three compile-negative boundaries.
- Implement actual queue cutoff/fairness/work adapter and test command/retirement/
  capture races, cancellation, stale epochs, queue capacity, admission before
  allocation and unknown retention with original owners.
- Instrument all listed real constructor/event/storage/final teardown paths;
  force bounded close/drain interleavings on the SAME physical card. No mutex
  guard spans await, driver wait, actor enqueue or Drop needed by drain.
- Real returned-graph known/error/unknown cases retain source/band/leases through
  original barriers; unknown does zero later driver calls and no recovery.
- Only after complete participation proof may the existing graph program use
  production capture. Root/session/math/readout publishers remain unchanged.

````

## Source: .agents/runs/serving-and-speculation-cuda-retirement-spine-20261020/README.md

Original UTF-8 bytes: 1808; SHA256 `6d80760d32475011c81b21f50cb9447539774a0bd4176f18f039f23530425896`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original-runner prepaid retirement — source proposal

Author: `/root/spine_review`. Parent: serving-and-speculation.
Status: **SOURCE ONLY, uncompiled/unexecuted; complete participation CLOSED**.

- Twelve proposed source paths in `proposal/`; existing source baseline pinned
  in `source-pins.json`. Firsthook/CallError-v2 and pure-queue freezes unchanged.
- Lazy fixed64 retirement slots on the **existing custom-channel runner**.
  Reserve before returned-executable construction, arm before `end_capture`;
  last handle Drop only publishes an already-owned slot with atomics.
- Original runner pumps outside the whole normal task/service borrow. No new
  worker, physical registry, model engine, CUDA context or byte ledger.
- Checked returned-graph cleanup keeps executable/page/source/info pins on any
  failed barrier or destroy. No returned id after armed construction is UNKNOWN.
- Explicit unsafe authority seam must join the SAME original QueueState,
  ResourceArbiter, fair admission, actor reentry and completion quarantine.
  No actual bridge or Complete issuer is supplied here.
- Eleven prepared host fixtures: capacity/generation/unarmed cancellation,
  close/park/known completion, error/panic quarantine, nested-pump exclusion,
  default None and actual original-channel borrow/full32 behavior. Not executed.
- Static source/hash/default-path assertions and rustfmt parsing only. No
  Ready/root publication, GPU completion, physical memory or capture safety claim.

See `DESIGN.md` for concrete continuation and `CALLERS.md` for remaining actual
lifetime/driver entry coverage. In particular events/storage/constructors and
final service teardown still escape this returned-graph slice. Production global
capture must stay disabled until those rows and the real queue bridge qualify.

````

## Source: .agents/runs/serving-and-speculation-cuda-retirement-spine-20261020/TODO.md

Original UTF-8 bytes: 1356; SHA256 `a3ab8d828b655b46176b6356a1879404098c878874ca0accccd189d4cf7f7344`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist — FROZEN source proposal

Parent: serving-and-speculation. Owner: /root/spine_review.
No MAIN/cache/vendor registration/Cargo.lock/download/build/runtime changes.

- [x] Inspect original channel, graph driver/cache, CUDA events/storage/teardown.
- [x] Pin firsthook + CallError-v2 + actual selected CUDA/server baseline copies.
- [x] Source draft fixed64 original-runner prepaid mailbox, atomic lastDrop,
  original completion-quarantine callback and checked returned-graph cleanup.
- [x] Read source/default path; armed end_capture without returned id is UNKNOWN;
  pump payload access guarded against accidental reentry.
- [x] Prepare eleven host fixture sources; no author test execution.
- [x] Document SAME QueueState cutoff/fairness, trusted device continuation and
  remaining events/storage/bootstrap/pre-begin/final-service teardown coverage.
- [ ] ROOT review/compile/host/no_std checks, actual bridge and all lifecycle
  plumbing. No complete issuer/production capture qualification in this slice.
- [ ] Physical original-card close/drain/cancel/fence/unknown tests, full gate.

Firsthook30 + additive typed4 and pure QueueState freezes unchanged. ROOT-reported
pureQueue14 +3 compile negatives are host-only evidence. Locked117 archive/cache
source proof ROOT028181fb is independently ROOT-owned, not an author execution.

````

## Source: .agents/runs/serving-and-speculation-cuda-retirement-spine-20261020/frozen/CALLERS.md

Original UTF-8 bytes: 3188; SHA256 `338d707e86545ffedf221b7a0c6afeaf401f8f77fb02c35f39c7f5ad1d84b1ab`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Exact caller continuation map

Author: `/root/spine_review`. All physical completion/complete-participation rows
remain UNPROVEN. Paths below refer to actual pinned source in source-pins.json;
CUDA prefix is crates/zerogravity-gpu/vendor/cubecl-cuda/src.

| Owner/path | Source slice | Required continuation |
|---|---|---|
| common device/handle/retirement.rs | New fixed64 prepaid slots, move-only handles, original-quarantine callback | Actual QueueState authority, cutoff fairness, prepaid control charge, real actor binding |
| common device/handle/channel.rs custom Server | Same runner pumps after whole Task::run and while idle | Pre-close actor barrier/ack, final service detach before TLS drop, shutdown/unknown tests |
| common device/handle/base/mod/admission +device/base | Optional preparation/default Unproven | Original physical registration, actual cancellation/epoch/reentry and service/stream population seal |
| runtime client GraphHandle/stop_capture | Reserve/arm before returned-executable construction; last Drop publishes | Move prepay before begin_capture; error/temporary graph ownership; explicit prepared capture token |
| runtime server/base +server command/capture | Default-refusing checked destroy, map/pins retained on errors | All drivers and private error/abandonment destruction participate or refuse Complete |
| CUDA compute/capture +server graph_destroy_checked | Checked graph pointer destroy after successful original sync | Temporary graph/exec failures, all event/fence calls and original completion continuation |
| GPU physical_queue.rs /physical_quarantine.rs | No edits | SAME original QueueState/capture ticket/ResourceArbiter bridge; existing quarantine retains actual mailbox/service owners |
| GPU resident_leases init/allocate/upload/sync +resident weights/BufferLease Drop | No edits | Permit before actual physical effects; actual storage lease retained through known free or quarantine |
| CUDA compute/events.rs +server device_events/{base,event,fence,profiler} | No edits | Explicit authentic origin/prepaid event destroy; foreign wait/elapsed/GC, failed-wait destruction |
| server command/read_async +stream/event ResolvedStreams::Drop +drop_queue | No edits | Source/destination/future/stream owner through actual GPU terminal; no late fresh admission during close |
| CUDA compute/storage/{gpu,cpu} +pool relocation/drop drain | No edits | Exact async/sync alloc/free kind, pinned memory, relocation/flush/fallback paths before driver calls |
| CUDA compute/stream.rs +runtime.rs +server new/final drop | No edits | Pre-context original registration, stream create/final release and primary-context ownership/errors |
| common normal-channel/mutex/reentrant/custom transports | Unsupported retirement, default None unchanged | Actual prepaid original-executor mechanism or cold complete refusal; no ordinary enqueue from Drop |
| original OwnedCudaCommand /external same-card HIP/Vulkan | No edits | Original actor/native event/stream gate or cold complete refusal |

No new CUDA context, executor, physical registry, model loop, shader, numerical
profile, source identity, SessionAuthority publisher or byte-budget ledger.

````

## Source: .agents/runs/serving-and-speculation-cuda-retirement-spine-20261020/frozen/DESIGN.md

Original UTF-8 bytes: 8535; SHA256 `215893d7dc88d0e3ca723feafa15ab05a196f2c97cd7643a978b93c9bf084293`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Retirement integration design

Author: `/root/spine_review`. SOURCE ONLY. Preserve original math/queue/flight/
CudaComplete/SessionAuthority publishers. This does not establish GPU Ready.

## Implemented source slice

`DeviceHandle::prepare_retirement` asks the already cached original hook for a
retirement authority. None returns before allocating task or mailbox. Hook
presence alone is insufficient: default authority is Unproven; unsupported
transports refuse. Explicit service/Client stream metadata is not physical proof.

The custom channel lazily allocates 64 slots on its existing State. A slot owns
request, original authority and a preallocated task Arc. Reservation consumes a
checked generation/free bit before construction. Unarmed cancellation frees only
resources that must not yet have reached a driver. Arm occurs before end_capture;
its Drop publishes with atomics, never ordinary submit, queue-full wait, mutex,
allocation or driver call. Capacity refusal occurs before construction. Sixty-four
is this slice's explicit cap, NOT a claim that every model/event/pool fits it.

The original Server pumps at loop head and after Task::run, after the RefCell
service borrow has ended. A nonblocking atomic pump guard encodes exclusive
payload access even under accidental reentry. Each published slot either parks
unchanged, receives owned original cleanup work, or enters original quarantine.
Known terminal cleanup clears/free-reuses the slot only after the task returns
and the original execution owner reports terminal. Error/panic does not retry.
The whole bounded mailbox, original actor and all task resources stay retained
through the existing completion-quarantine owner type. A violated unsafe bridge
that panics while retaining uses a conservative Arc leak fallback; that fallback
is NOT a physical fence proof or recovery mechanism.

GraphRetirement owns the actual original DeviceHandle and explicit stream. Only
its checked actor call may clear the actual returned GraphId. An armed capture-end
failure with no returned id is UNKNOWN, not known no-resource. The runtime Server
checked method defaults to refusal; CUDA waits first and aborts on ANY wait error.
The shared graph map retains executable, host bytes and page/info pins until the
explicit driver destroy returns success. CUDA clears its private pointer only on
that success, preventing a second destroy from later Drop. Existing legacy None
paths and ordinary/native APIs remain source-equivalent.

## Next actual QueueState bridge — required, not implemented

1. Store retirement registration/close cutoff obligations in the SAME original
   QueueState, tied to actual ResourceArbiter physical key/epoch and registered
   actor. Do not create a second card registry, scheduler or resource ledger.
2. Capture uses its original fair ticket. Under the original queue mutex it
   closes new admission and records actor drain obligations. Release the mutex
   before contacting any actor. A prepaid actor barrier/pump acknowledges a
   bounded publication cutoff; each eligible retirement joins existing original
   work accounting before that actor's cutoff acknowledgement.
3. Before-close work and explicitly admitted drain continuations can execute
   while Closing; after-cutoff publications park with owners intact. Capture
   waits for cutoff acknowledgements AND all original trusted work terminals.
   Do not wait for an ActiveSlot required by cleanup while capture holds that
   same slot. Fair next-open admission must service pending retirement/cancelled
   construction obligations before repeated captures can starve them.
4. RetirementExecutionOwner::enter installs the exact actual actor scope, so
   the nested original checked call uses prepaid reentry, never new admission.
   Actor/key/epoch checks come from the actual registration, not numeric IDs.
5. Host callback return is not GPU completion. Existing initialization and
   ResidentSubmission flights retain original work through trusted event/readback
   and CudaComplete. A waiter needed by Closing has a pre-admitted continuation;
   it cannot request a fresh shared permit and wait behind its own capture.
6. Unknown fences the original physical key and transfers pre-admitted owners to
   original quarantine. It never calls a driver, waits for its own actor, retries
   destruction or declares recovery. Quarantine control storage must itself be
   prepaid; this proposal's task/mailbox bytes are not a VRAM/RAM attestation.

The pure QueueState 14 host tests and three intended compile negatives passed
in ROOT's isolated tree (reported result305070096Ba7a76529); this is host protocol
evidence only. It does not instantiate any of the unsafe retirement bridges.

## Actual remaining construction/completion/destruction plumbing

- **Capture before begin:** this slice prepays the returned graph before
  end_capture. Temporary CUgraph/CUgraphExec objects, failed instantiate paths and
  capture beginning itself still need an original pre-begin reserved owner.
  Prefer a move-only prepared capture/retirement token consumed by the original
  start/stop lifecycle, not a global map or a new reservation inside Capturing.
- **Events:** current EventApi has static methods/raw Copy stream; CudaEvent owns
  only a raw event. Event::Drop can call the driver from a host/GC thread, and a
  failed EventFence::wait_sync still drops/destroys the event. Checked graph
  cleanup does NOT cure this escape. Add explicit sealed original event origin
  and pre-created retirement/continuation before event_create, record/wait/elapsed/
  stream_wait and fallback synchronize. Thread/ordinal/ambient context lookup is
  insufficient. All generic fence/profiler/GC callers must propagate the origin.
- **Storage:** reserve actual allocation retirement before async or sync-fallback
  alloc and before pinned alloc. Preserve exact alloc/free kind, actual pointer/
  byte range/original stream and original backing resource lease. Populate the
  preallocated task only from actual successful constructor results. Pool logical
  dealloc/relocation/free/flush and every fallback must use the same owner. Known
  native free may release charge; unknown retains it. Managed handle Drop is not
  itself proof that physical storage was freed.
- **Final service/stream/context:** internal resources cannot all hold strong
  Clients forever (service -> task -> Client -> actor cycles). Explicit original
  actor shutdown must detach/retire service resources while the runner/pump still
  exists, then drain retirement before thread-local service destructors/stream/
  context release. A destructor cannot try to weak-upgrade a dead runner or join
  itself. Unknown retains original final service/storage owners in quarantine.
- **Bootstrap:** authenticate the existing physical registration before CUDA
  initialization/context/stream construction. Complete admission must forbid an
  unknown same-card actor from silently taking the legacy None path. DeviceId or
  CUDA ordinal alone is not a physical binding. Seal actual participating actors
  only after constructors/errors/final drop and alternate runtimes are covered.
- **Alternate transports:** dormant normal-channel, mutex/reentrant retirement,
  external pinned commands, same-card Vulkan/HIP/custom actors need coverage or
  cold Complete refusal. Default None stays legacy; no production issuer exists.

External-process exclusion remains an environmental qualification precondition.
No host mailbox can guarantee exclusion from other processes or GPU recovery.

## Required ROOT gates before any production close

- Compile source plus default/no_std native checks; execute eleven prepared host
  tests, exact firsthook14 and all three compile-negative boundaries.
- Implement actual queue cutoff/fairness/work adapter and test command/retirement/
  capture races, cancellation, stale epochs, queue capacity, admission before
  allocation and unknown retention with original owners.
- Instrument all listed real constructor/event/storage/final teardown paths;
  force bounded close/drain interleavings on the SAME physical card. No mutex
  guard spans await, driver wait, actor enqueue or Drop needed by drain.
- Real returned-graph known/error/unknown cases retain source/band/leases through
  original barriers; unknown does zero later driver calls and no recovery.
- Only after complete participation proof may the existing graph program use
  production capture. Root/session/math/readout publishers remain unchanged.

````

## Source: .agents/runs/serving-and-speculation-cuda-retirement-spine-20261020/frozen/README.md

Original UTF-8 bytes: 1808; SHA256 `6d80760d32475011c81b21f50cb9447539774a0bd4176f18f039f23530425896`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original-runner prepaid retirement — source proposal

Author: `/root/spine_review`. Parent: serving-and-speculation.
Status: **SOURCE ONLY, uncompiled/unexecuted; complete participation CLOSED**.

- Twelve proposed source paths in `proposal/`; existing source baseline pinned
  in `source-pins.json`. Firsthook/CallError-v2 and pure-queue freezes unchanged.
- Lazy fixed64 retirement slots on the **existing custom-channel runner**.
  Reserve before returned-executable construction, arm before `end_capture`;
  last handle Drop only publishes an already-owned slot with atomics.
- Original runner pumps outside the whole normal task/service borrow. No new
  worker, physical registry, model engine, CUDA context or byte ledger.
- Checked returned-graph cleanup keeps executable/page/source/info pins on any
  failed barrier or destroy. No returned id after armed construction is UNKNOWN.
- Explicit unsafe authority seam must join the SAME original QueueState,
  ResourceArbiter, fair admission, actor reentry and completion quarantine.
  No actual bridge or Complete issuer is supplied here.
- Eleven prepared host fixtures: capacity/generation/unarmed cancellation,
  close/park/known completion, error/panic quarantine, nested-pump exclusion,
  default None and actual original-channel borrow/full32 behavior. Not executed.
- Static source/hash/default-path assertions and rustfmt parsing only. No
  Ready/root publication, GPU completion, physical memory or capture safety claim.

See `DESIGN.md` for concrete continuation and `CALLERS.md` for remaining actual
lifetime/driver entry coverage. In particular events/storage/constructors and
final service teardown still escape this returned-graph slice. Production global
capture must stay disabled until those rows and the real queue bridge qualify.

````

## Source: .agents/runs/serving-and-speculation-cuda-retirement-spine-20261020/frozen/TODO.md

Original UTF-8 bytes: 1356; SHA256 `a3ab8d828b655b46176b6356a1879404098c878874ca0accccd189d4cf7f7344`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist — FROZEN source proposal

Parent: serving-and-speculation. Owner: /root/spine_review.
No MAIN/cache/vendor registration/Cargo.lock/download/build/runtime changes.

- [x] Inspect original channel, graph driver/cache, CUDA events/storage/teardown.
- [x] Pin firsthook + CallError-v2 + actual selected CUDA/server baseline copies.
- [x] Source draft fixed64 original-runner prepaid mailbox, atomic lastDrop,
  original completion-quarantine callback and checked returned-graph cleanup.
- [x] Read source/default path; armed end_capture without returned id is UNKNOWN;
  pump payload access guarded against accidental reentry.
- [x] Prepare eleven host fixture sources; no author test execution.
- [x] Document SAME QueueState cutoff/fairness, trusted device continuation and
  remaining events/storage/bootstrap/pre-begin/final-service teardown coverage.
- [ ] ROOT review/compile/host/no_std checks, actual bridge and all lifecycle
  plumbing. No complete issuer/production capture qualification in this slice.
- [ ] Physical original-card close/drain/cancel/fence/unknown tests, full gate.

Firsthook30 + additive typed4 and pure QueueState freezes unchanged. ROOT-reported
pureQueue14 +3 compile negatives are host-only evidence. Locked117 archive/cache
source proof ROOT028181fb is independently ROOT-owned, not an author execution.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/CONTAINER_FUTURE_SEAM.md

Original UTF-8 bytes: 2138; SHA256 `c1ba0692eb6541368c4564bf2768d7f8d4673b84cd1fc05ffd7569e4ae7333ea`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Future container diagnostic seam — not implemented

Author: batch_design_review. Additive note after immutable source freezef9f5c522.

Current proposal is **host-native only**: direct native binary argv, same-UID
abstract UNIX control peer and exact host Popen/native PID. It does not establish
container collector support, and genuine matched benchmark timing remains
separate/unprofiled.

ROOT's actual matched native environment runs through a container Python wrapper
which ultimately `os.exec`s the original native program. A later diagnostic
derivative must set LD_PRELOAD and collector-only env **at that final native
exec**, leaving outer Python/container management unprofiled. Do not place the
collector on outer Python or benchmark-main argv and infer that exec preserved
a single initialized collector. An ordinary constructor can otherwise run
before and after exec, leaving a stale initialized control endpoint.

This frozen candidate already checks `/proc/self/exe` against the explicit
native target inode and is inert for another executable. That defensive host
guard is not container namespace/lifecycle qualification or a reason to skip the
explicit final-exec-only seam.

Future implementation requires an independently reviewed controller in the
original owned container's actual PID/network/user namespace, authenticated to
the exact original native process and nonce. Host Popen PID identifies Podman,
not its inner native PID; abstract UNIX sockets and SO_PEERCRED are namespace
dependent. Reuse original nonce/CID/owner/retirement receipts; no foreign process
lookup, setns privilege change, separate model engine or arbitrary context/stream
API. Controller placement/startup/flush/retirement must be physically qualified
before any container activity counts or matched hardware comparison claim.

Unsupported activity/driver/CUPTI permissions must refuse cold. No device
privilege, driver recovery, privileged counter enablement, SDK installation or
silent fallback is authorized by this profiler proposal. First ROOT execution
remains the limited host-native source freeze, after timing quiet/reservations.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/LIFECYCLE_V2.md

Original UTF-8 bytes: 4560; SHA256 `e0b8ccfe7f87f7286fe523f01bb6a974541018fbe9de7d80abafc105e06e46a0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive native owner V2

Original `f9f5c522…`/13 files/frozen copies remain immutable. This fixes the
original launcher's lifecycle defect; it does not change `collector.cpp`.
Original `serve-lifecycle.py` uses `poll()` during startup and `wait()` before a
later group signal. That can reap the leader before the last nonzero PGID
signal. V1 must not execute a server.

## Narrow composition

- `run_probe_v2.py`: canonical schema/endpoints/bench/preflight/warm→idle→START→
  workload→idle→STOP and plain→observed order unchanged. Actual native argv is
  still direct; no shell, model wrapper or alternate CUDA queue.
- `native_owner_v2.py`: subclasses only canonical Plan bookkeeping; replaces
  its `sample`/provider lifetime. Never calls original OwnedProvider, BoundedLog,
  `poll`, `wait`, `communicate`, start or close.
- `owned_container_v4.py`: exact reviewed `53ee4270…` source. Only public
  `leader_status_unreaped`, `cleanup_group` and bounded main-thread `Log` used.
  No Podman/container command is invoked by this native-only adapter.
- Unreaped child identity remains reserved through startup, canonical client
  calls and all nonzero group signals. Original `start_new_session=True` and
  authenticated SID/PGID/child checks bind both KILL attempts before ANY reap.
  Only signal 0 is used after reap; final group absence required.
- One non-daemon canonical client thread lets the main thread continuously
  drain bounded logs and observe the server with `waitid(WNOWAIT)`. No client
  function/math/HTTP policy is replaced. A deadline/unjoined client bypasses
  readiness retry; the original server is retired before joining that client.
- Effective retained-log cap = min(canonical declared cap, reviewed 4MiB cap).
  Any overflow/EOF/I/O/endpoint/unknown-client failure refuses acceptance and
  blocks a next provider. No output truncation accepted.
- Cleanup attempts group retirement, client join, complete log/EOF/descriptor
  closure and endpoint absence independently. Evidence I/O occurs AFTER actual
  cleanup. Primary and cleanup failures are retained together. SIGTERM raises
  the same cleanup path as an interrupt; SIGKILL/parent-death is not qualified.
- `validate_v2.py`: both actual receipts must bind this adapter/helper, ready
  success, two pre-reap KILL attempts, leader reap/group absence, complete log
  byte/hash/EOF, joined client and absent endpoints. No primary/evidence error.
  Original GPU-activity reducer is AST-identical to V1; API launch calls and
  real kernel records remain separate counts.

## ROOT-only gates

Author only read/edited source, parsed Python ASTs and hashed source/metadata.
No fixtures, build, child process, CUPTI/CUDA/DSO/provider/operator/model/GPU or
container executed by author. No native library payload hashed by author.

ROOT may run `python3 test_native_owner_v2.py` and original
`python3 test_contracts.py` under its serialized supervisor. Seven new fixtures
use real harmless owned Python children: normal/TERM-resistant, exited leader,
client pending until teardown, failed log construction, persistent postspawn
receipt failure, measurement interrupt, and a static no-early-reap check. These
are process-lifetime controls only; no CUDA or numerical proof.

Then ROOT uses `run_probe_v2.py` with the original arguments and a FRESH output,
followed by `validate_v2.py --output ABS`. Keep original V1 failures/output and
compiled artifacts unchanged. `root-build-1/build.json` is an actual V1 C++ build;
V2 explicitly binds its original freeze/source/artifact hash, because C++ bytes
are unchanged. V2 execution freeze is recorded separately; no invented new
compilation receipt or source rebuild claim.

A real native diagnostic still requires exact binary/CUPTI full-hash binding,
plain/observed wire transparency, positive declared-window GPU kernels, no loss
and complete actual retirement. Unsupported CUPTI/permissions fail closed; no
privilege change. B1/B2/B4/B8/graph/failure controls, hardware overhead, container
collection and vLLM collector qualification remain ROOT physical gates.
Unprofiled matched timings remain separate. This is not performance acceptance,
new numerical support, source/full-StateRoot proof or provider allocation
admission. The adapter proves only its own unreaped leader/PGID boundary; any
escaped process/session or unjoined thread is not silently claimed retired.

Future container seam stays in immutable `CONTAINER_FUTURE_SEAM.md`: preload
only final native exec, authenticated namespace controller, separate scope.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/README.md

Original UTF-8 bytes: 8920; SHA256 `c76986f2f28a6d8d3ab98eed71d5fb11d0cf6bf0f7e7189f1146ff4f0807a76a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Official CUPTI activity diagnostic

Author: batch_design_review. Source-only, uncompiled/unexecuted.
Parent: existing serving-and-speculation plan; diagnostic only.

Official CUPTI12.9.79 library **and headers are already present** in the approved
local image/cache. No SDK/tool/model download is needed. Host `nsys`/`ncu` are
absent from PATH; inspected image Nsight prefixes contain no candidate tools.
`LOCAL_AVAILABILITY.json` records actual local paths/metadata and header pins.
This is a CUPTI activity candidate, not an available Nsight CLI workflow.

## Exact available source

Cache site root:
`/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/7940bf3c960e46bcf6f5848ed00b3dadaa7d399d891836cd12dcd4746cb4c8cf/diff/usr/local/lib/python3.12/dist-packages`

- `nvidia/cuda_cupti/include/{cupti_activity,cupti_callbacks,cupti_version,cupti_driver_cbid,cupti_result}.h`
- `nvidia/cuda_cupti/lib/libcupti.so.12`:7,604,800B; accepted prior full SHA
  `2fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a`.
  Author checked current metadata/static ELF, not a new DSO payload hash/load.
  ROOT launcher must rehash the actual full library before and after collection.
- Public CUDA headers under `nvidia/cuda_runtime/include` and
  `nvidia/cuda_nvcc/include`, also already local.
- Package metadata: `nvidia_cuda_cupti_cu12-12.9.79.dist-info`; API macro28.
  Exact default exported symbols are versioned `@@libcupti.so.12`; candidate
  uses public `dlvsym(...,"libcupti.so.12")`, then checks same file dev/ino/size.
- Derived image `f57d77f5...` / manifest `eec72f47...` inherits this base layer;
  image metadata/source lookup establishes availability, not runtime support.

## Collector and queue boundary

- Diagnostic ELF `LD_PRELOAD` constructor, not a guessed CUDA injection entry.
  No interception or replacement of CUDA/model/math functions. Only exact target
  native executable dev/ino is observed; inherited compiler/helper execs are
  inert. ROOT original process ownership handles shutdown and reap.
- Official Activity API `CONCURRENT_KERNEL` + `DRIVER` are enabled before native
  server main/CUDA initialization. **Never KIND_KERNEL:** local header says it
  serializes kernels. No counters/metrics/PC sampling or profiler replay.
- Whole-process capture initialization preserves graph metadata visibility, but
  records retained to disk are selected by actual CUPTI START/STOP timestamps.
  Startup kernels alone cannot satisfy the window workload check.
- Native/Ready/ResourceArbiter/original queue and completion are unchanged. The
  collector never acquires another CUDA context/stream, submits GPU work, reads
  model/state memory, synchronizes the CUDA device or grants new authority.
- `FlushAll(FORCED)` blocks delivery **without CUDA synchronization**. ROOT must
  first obtain successful complete HTTP/SSE outputs and two original health
  snapshots with active/queued/parked0. Every retained record must then have
  complete nonzero timestamps, lie inside the declared window and match totals.
  Boundary-spanning GPU work/incomplete records are a refusal, not rounded away.
- Actual kernel activity records are distinct from host driver launch API calls.
  A CUDA Graph launch can yield many kernel records. Validator keeps both
  domains, public correlation IDs, graph/node IDs, grid IDs and original PID.

## Bounds, loss and lifetime

-16 reusable aligned1MiB callback buffers; no unbounded queue/record vector in
  the collector. Official requested device buffers1MiB, pool8, preallocate2,
  pinned-host allocation0; every attribute must read back byte-exact.
- Header explicitly warns actual device allocation can exceed requested buffer
  size. Device/RAM/control/CUPTI worker-thread overhead is diagnostic external
  overhead, **not a proved model-admission allocation bound**. No production
  memory/throughput claim follows from these buffer controls.
- Serialized output<=128MiB/1,000,000 records; names<512B; invalid/unknown kinds,
  null/oversized buffers, iterator errors, zero timestamps, dropped records,
  output cap/disk errors or nonzero CUPTI status make the trace refuse.
- Dropped counters are reset on read by CUPTI: callback+final values are summed;
  no silent loss allowed, including startup capture. Completion callbacks are
  global in these headers, so no deprecated per-stream-buffer assumption.
- One process-owned controller thread, private random abstract UNIX seqpacket
  name, same-UID SO_PEERCRED checks, exact nonce and two bounded commands.
 600s control deadline +5s peer I/O; ROOT client uses30s. No detached external
  helper, second model process, arbitrary CUDA command or exposed stream pointer.
- On STOP, kinds are disabled, forced flush completes, no buffer may remain
  busy, log is fsynced and exact metadata persisted/acknowledged. CUPTI DSO,
  registered callbacks/buffers and controller owner remain retained until the
  **original server process is actually reaped**. No unsafe mid-process
  `cuptiFinalize`/unload is attempted. Original lifecycle tears down every
  success/failure path with finite startup/shutdown and refuses incomplete
  owned group/log/readiness/endpoint cleanup.

## ROOT execution only

1. Review `collector.cpp`, Python source, SDK/API pins and lifecycle composition.
   Run `python3 test_contracts.py` only on ROOT's serialized source-free schedule.
2. Compile the new collector under ROOT's existing isolated compiler ownership,
   using `ROOT_BUILD.md`. No author compilation/runtime occurred. Retain actual
   compiler/source/library/artifact identity and errors; new attempt for fixes.
3. Supply an existing approved lifecycle spec whose native direct argv begins
   with the exact frozen native binary and whose second vLLM endpoint is unused.
   This helper launches **native only**, twice, serially. Full declaration/source
   acceptance stays in that original owning spec. No startup/measurement samples
   are added to the actual unprofiled timing run.
4. Execute `run_probe.py` with the documented args below, then independently
   `validate.py --output` using the same fresh absolute output directory.
   Both fresh server processes/warmups/HTTP workload must complete normally.

   ```bash
   python3 run_probe.py \
     --lifecycle-spec /absolute/approved-native-and-unused-vllm-spec.json \
     --binary /absolute/frozen/zerogravity --binary-sha256 ACTUAL_NATIVE_SHA \
     --collector /absolute/collector.so --collector-sha256 ACTUAL_COLLECTOR_SHA \
     --collector-build /absolute/ROOT-collector-build.json \
     --url http://127.0.0.1:18000 --unused-vllm-url http://127.0.0.1:18001 \
     --model APPROVED_NATIVE_ROUTE --requests 2 --concurrency 2 --max-tokens 4 \
     --new-output /home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/root-attempt-1
   ```

The ROOT build receipt must contain actual `status=ROOT_COMPILED_COLLECTOR_ONLY`,
freeze_sha256, collector_sha256 and successful compiler exit_code0; retain its
actual command/log beside it. The author supplies no invented binary receipt.

The helper preserves the original serving tool benchmark schema but labels all
results diagnostic-only. Plain/observed must match every request's text hash,
wire token IDs, finish reason and usage; all offers must complete. The workload
must increase actual native accepted output tokens between original idle
snapshots. Raw kernel records/launch correlations in the timestamp window then
establish activity for declared HTTP work, not source launch-site estimates.

## Acceptance and remaining scope

- Candidate first receipt: `PASS_NATIVE_CUPTI_WORKLOAD_ACTIVITY_DIAGNOSTIC_ONLY`.
  Original CPU/model numerical gates are not newly proved by output transparency.
- ROOT must physically test unsupported CUPTI/permissions, loss/cap/disk errors,
  killed/disconnected control, server error/timeout and failed cleanup. A failed
  collector never falls back to an unobserved successful trace.
- ROOT must qualify B1/B2/B4/B8 and graph replay activity cardinality where used,
  and separately the pinned real vLLM collector under its namespace/lifetime.
  This first host-native seqpacket/peer-PID design does not pretend to work
  unchanged across a container PID/network/user namespace.
- No new SDK/device privilege changes. Existing driver/device restrictions
  refuse unchanged. Hardware comparison remains blocked until both collectors
  and the matched model/source/profile/resource/performance gates pass.
- All timing claims use separate unprofiled samples. Tracing costs, extra
  driver API records, buffers and synchronization/delivery effects are reported
  as observer overhead; no speedup inferred from this diagnostic.

`startup-v1` provider containment freeze961d803b is untouched and unrelated to
this profiler's production eligibility. No MAIN/shared plan/root README edit.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/ROOT_BUILD.md

Original UTF-8 bytes: 1720; SHA256 `a686ec1fbe6b9f8c814e8a25c54198e3d62935238d7446fff3de471b5b527c54`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT collector build

Source-only preparation. Existing C++ compiler + already approved local headers;
no SDK install, GPU context, CUDA compiler or model is needed for this build.

Under ROOT's original isolated/serialized compiler supervisor, invoke direct
argv as represented below. `SOURCE`, `OUTPUT` and `SITE` are absolute task paths,
not shell text interpolated from an untrusted request.

```text
c++ -std=c++20 -O2 -fPIC -shared -Wall -Wextra -Werror -Wformat=2
  -fvisibility=hidden -Wl,-z,now
  -isystem SITE/nvidia/cuda_cupti/include
  -isystem SITE/nvidia/cuda_runtime/include
  -isystem SITE/nvidia/cuda_nvcc/include
  SOURCE/collector.cpp -ldl -pthread -o OUTPUT/collector.so
```

`SITE` is the exact absolute local image-layer site root in README. Root include
and dependency scan must remain bound to those local official header bytes;
headers are not copied/replaced. Symbols are resolved at runtime from the one
explicit exact CUPTI library, not statically linked to CUDA or another provider.

Record actual compiler/argv/log/exit and the compiled library full hash; verify
actual ELF dependencies before execution. These are profiling-tool dependencies
only, separate from the native model numerical source and unprofiled timing.
The launcher requires a ROOT actual build JSON:

```json
{
  "status": "ROOT_COMPILED_COLLECTOR_ONLY",
  "freeze_sha256": "ACTUAL_SOURCE_FREEZE_SHA",
  "collector_sha256": "ACTUAL_COMPILED_COLLECTOR_SHA",
  "exit_code": 0
}
```

Do not create that success receipt before the actual build. Failed build/runtime
attempts remain separate and unchanged; additive source corrections require a
new freeze/binary/output. Author ran no compiler, source-free fixture or CUPTI.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/TODO.md

Original UTF-8 bytes: 1372; SHA256 `ceb6c762953075f0bd0ae97a4312c545d2cf6839c5e6e106c92596e212b53e93`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUPTI diagnostic preparation

Parent: serving-and-speculation. Owner: batch_design_review.
Scope: this ignored source-only proposal, original native server unchanged.

- [x] Find existing official local CUPTI12.9.79 headers/library/API28.
- [x] Inspect local image ancestry, package metadata and public versioned ABI.
- [x] Confirm no host nsys/ncu PATH entry or inspected image-prefix candidate.
- [x] Prepare bounded concurrent-kernel/driver activity collector and window.
- [x] Reuse original owned lifecycle + serving benchmark; no second queue/engine.
- [x] Prepare independent cardinality/loss/transparency/cleanup validator tests.
- [x] Document exact overhead/flush/lifetime and remaining physical scope.
- [ ] ROOT independent full source review.
- [ ] ROOT source-free schema fixtures/strict C++ compile/ELF/payload binding.
- [ ] ROOT actual plain/observed workload, full wire transparency/cleanup.
- [ ] ROOT actual failure/loss/cap/unknown/permission and B1/2/4/8/graph controls.
- [ ] Independently qualify pinned vLLM collector/namespace/lifetime.
- [ ] Matched hardware comparison with separate unprofiled timing, owning gate.

Author results: source/header/metadata/static ELF reading, Python AST and small
source hashes only; no builds/tests/CUPTI load/operator/model/GPU/container
execution, payload hashing, SDK download, MAIN/plan edits or commits.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/README.md

Original UTF-8 bytes: 8713; SHA256 `2eee08c2526be0ef273c6b2ceb957b704f083f8c0e827ac7ee81322c5e949179`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native container activity diagnostic — source only

Author: batch_design_review. Parent plan: serving-and-speculation.

Own scope: this ignored child only. Immutable parents: CUPTI V1 f9f5c522, V2
932e471e and genuine matched-benchmark V8 b0c50e8a. No MAIN/parent edits, engine,
math, queue, source/model, timing workload or production profile changes.

## Files / ROOT execution

- `capacity-v1/`: separate bounded C++ revision + streamed validator + six
  synthetic fixtures. Capacity freeze4337ad63; new actual ROOT build required.
- `container_contracts.py`, `parent-bindings.json`: exact additive closure.
- `launch_derivative.py`: exact V8 argv + diagnostic-only mounts/env/bootstrap
  transport and finite file-size cap. Original model flags untouched.
- `native_exec.py`: run unchanged V8 bootstrap; its one final native exec uses
  `os.execve` with explicit collector env. Outer Python/Podman LD_PRELOAD remains
  empty. Stock always runs original bootstrap unchanged.
- `namespace_control.py`: original owner nonce/name/CID -> public inspect init
  HPID -> pidfd/start time/namespace readbacks -> host SO_PEERCRED. No signal,
  reap, setns, new privilege, foreign PID adoption, context or stream.
- `run_container.py`: four fresh original V8/53ee-owned processes: native plain,
  native observed, stock plain, stock public Torch-profiler observed.
- `container_activity.py`, `capacity_v1_loader.py`, `validate_container.py`:
  independent full raw activity/wire/namespace/owner/canonical-argv validation.
- `test_contracts.py`: seven prepared synthetic transport/control negatives.
  No synthetic fixture establishes physical collection or model mathematics.

ROOT first reviews/freezes, then executes only after timing quiet + reservation:

```text
python3 capacity-v1/test_activity.py
python3 test_contracts.py
```

Build `capacity-v1/collector_capacity.cpp` with immutable parent ROOT_BUILD.md
procedure and original approved headers/compiler. Record actual dependency/ELF
closure, argv/source/build/artifact hashes. Fresh artifact directory contains
`collector.so` and actual `build.json` with:

```json
{
  "status": "ROOT_COMPILED_COLLECTOR_ONLY",
  "freeze_sha256": "4337ad630552e0b46644cfb39d8198c6aad94cb3b48d85c8402a3a315a247bdd",
  "collector_source_sha256": "ACTUAL_CAPACITY_CPP_SHA256",
  "collector_sha256": "ACTUAL_COMPILED_ARTIFACT_SHA256",
  "exit_code": 0
}
```

Do not prewrite success. The accepted e918ff2f host V1 artifact is intentionally
refused for this revision. Build can preserve original callback/API/math bytes;
only fixed limits128→512MiB, records1M→4M, schema and explicit capacity metadata
change. No runtime environment override can raise limits.

Use an existing genuine V8 launch directory (exact protocol/workload/launches/
input-attestation), not a shortened host-control prompt. Set ROOT explicitly:

```text
ZG_BENCH_HOST_ROOT=/home/pvl/spaces/abird/src/zerogravity
python3 run_container.py --run --own-freeze OWN_FREEZE_SHA256
  --launch-directory ABSOLUTE_ORIGINAL_V8_LAUNCH_DIRECTORY
  --collector-artifact-directory ABSOLUTE_FRESH_ROOT_CAPACITY_BUILD_DIRECTORY
  --concurrency ORIGINAL_PROTOCOL_CONCURRENCY
  --output ABSOLUTE_THIS_DIRECTORY/root-container-1
python3 validate_container.py --own-freeze OWN_FREEZE_SHA256
  --output ABSOLUTE_THIS_DIRECTORY/root-container-1
```

Invocation is direct argv under ROOT's existing original process supervisor,
not shell interpolation; export the explicit host-root variable. Author has
not imported, run fixtures, compiled or executed any provider/container/workload.
Only Python AST parsing and small source/metadata hashing/freezing occurred.

## Identity, window, policy

- Actual container init HPID comes from nonce/name/CID-bound public inspect;
  Podman Popen PID remains its original cleanup owner, not the collector PID.
- Exact host peer HPID + ordinary host UID must match SO_PEERCRED. pidfd remains
  live; public container binding, proc start time and namespaces match before/
  after each START/STOP. Output retains actual UID/GID maps and namespace IDs.
- Reported `getpid()` is namespacePID1. Raw CUPTI driver IDs are never rewritten:
  exactly one ID, equal to authenticated HPID OR namespacePID1; reject mixed or
  foreign IDs and preserve which representation actually occurred. Metadata and
  driver identities use separate validator arguments.
- Original `--network=host` supplies abstract UNIX reachability; actual net-ns
  equality and user-namespace peer translation must pass. Any hidepid/proc,
  pidfd, mapped-file, permission or provider denial refuses; no privilege repair.
- Actual native/collector/CUPTI mapped inode/dev/full SHA are checked through
  `/proc/HPID/root`; CUPTI C++ still resolves only official versioned symbols
  from exact original provider. Remaining original native/provider dependencies
  retain existing V8/ROOT build closure; no substitute DSO.
- Both native modes inherit exact V8 bootstrap/source/affinity/cgroup checks.
  Stock uses original profiler-config and public `/start_profile`/`stop_profile`;
  no stock math/module/queue patches or CUPTI preload.
- `cold_process`: no collector warmup request. `warmed_process`: ONE unchanged
  V8 unmeasured request with first original prompt and original max_tokens.
  OS/storage/device caches are not evicted or claimed cold. Official collector
  initialization overhead itself is diagnostic, not a canonical timing sample.
- Native START follows original terminal/idle health; all canonical requests
  then complete; native terminal/idle health precedes STOP/forced flush.
  All source native dispatch counters remain distinct from CUPTI records.
- Same canonical request count/prompts/arrivals/concurrency/max_tokens, CPU4,
  RAM12GiB/swap-total12GiB, device, model/provider limits, stock eager/default
  route and generation policy. No one-token collector warmup or workload edit.
- Four actual owner receipts/monitor/containers/wholegroups/readers/EOF must
  retire under unchanged reviewed53ee WNOWAIT cleanup before next process.
  Primary/stop/close/post-source failures remain fatal with original receipts.

## Capacity and remaining physical gates

512MiB raw /4,000,000 records are finite, not a guarantee every canonical wave
fits. Any buffer refusal/drop/overflow/partial record/API/timestamp mismatch
refuses all acceptance. Entire native file is streamed twice, exact duplicate/
correlation/totals checked. Validation sets are finite but may use several GiB
of separate host RAM: ROOT reserves headroom before validation; not engine
provider admission. Kernel count, successful direct launch APIs, graph launch
APIs and kernels executed from graphs stay separate.

Diagnostic-only per-file soft=hard RLIMIT_FSIZE is576MiB. Each owned sample tree
must be<=768MiB (capacity README's native-sample bound); four-sample aggregate is
<=1GiB to hold512MiB native raw + original stock trace<=256MiB + metadata/logs.
Tree limits are post-write acceptance checks, not a kernel filesystem quota.
File limit is enforced by original Podman public ulimit; native wrapper records
actual readback. Original CPU/RAM/model limits are unchanged. Stock trace retains
original256MiB compressed/expanded bounds; exceeding any bound is refusal.

Before claiming a method: ROOT must qualify actual new build/13fixtures,
namespace/user-peer/mapped closure, exact plain/observed wire IDs/text/finish/
usage, real workload window and no activity loss, actual caps, final process/
container/group/pipe/monitor closure. Negative tests must include foreign peer,
wrong nonce/namespace/stale PID, provider permissions/cold refusal, failed STOP,
interruption/receipt I/O, actual overflow/drop behavior and no accepted partial
trace. Synthetic corrupted metadata fixtures validate refusal only; they do not
establish actual callback overflow/CUPTI drop behavior.

Root source estimate236270 dispatches for C1/48prompt/output8 suggests old128MiB
cap too small. This is source metadata, NOT a canonical hardware measurement or
speed result. Never run old undersized artifact and accept a prefix of records.

Actual small host V2 PASS64328 kernels remains scoped to that old method. No
container, larger-capacity, matched counts or new performance claim follows.
Stock public Torch profiler currently does NOT prove losslessness/all-worker/
complete initialization coverage; validator deliberately returns
`matched_complete_hardware_kernel_counts_qualified=false` even if both recorded
workload scopes pass. These diagnostic data do not establish numerical profile
parity, source-correct goodput or a timing speedup. Opaque CUPTI internal device
allocation admission remains unqualified and observed resource cost is separate.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/TODO.md

Original UTF-8 bytes: 1651; SHA256 `2debefd7c25d04d1648efabed5412ee198353a3096f4fb1929c9a34536d94cca`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUPTI container diagnostic / capacity revision

Parent: serving-and-speculation. Owner: batch_design_review.
Scope: ONLY this ignored child; no production/parent/plan/README changes.

- [x] Read immutable CUPTI V1/V2 + exact V8 canonical launch/bootstrap/owner/
  diagnostic/trace/cgroup source; keep original53ee lifecycle unchanged.
- [x] Prepare separate512MiB/4M C++ cap revision and six source-only negatives;
  exact callback/API/math unchanged, new compiled artifact mandatory.
- [x] Prepare final-native-exec-only seam; host PID/namespace PID/raw driver PID
  distinct; original owner/caps/source/full canonical workload/cold-warm policy.
- [x] Prepare four-process plain/observed native+stock transparency and independent
  complete activity/argv/owner validator; seven container transport fixtures.
- [x] AST parse only; no imports, fixtures, compilation or runtime by author.
- [ ] Independent source review / ROOT actual fixture+strict C++build+dependency
  closure / actual process namespace+permissions/metadata/STOP/loss qualification.
- [ ] ROOT actual canonical workload/negative failure+overflow/retirement tests.
- [ ] Stock collector loss/completeness gate; no matched complete HW claim yet.
- [ ] Any actual measured gain/numerical equivalence/fullmodel claim is external;
  none established by this source proposal.

Failures: no executed test/build/runtime. Source-only inspection corrected a
post-START receipt-I/O cleanup gap before freeze; STOP flag now arms immediately
on acknowledged START. Post-run source/affinity errors downgrade report before
independent validation rather than leaving an executable-success status.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/capacity-v1/README.md

Original UTF-8 bytes: 1818; SHA256 `be65f53ffb83f50481731d59b288fefa5b1c8a739f4213f792dab6e23ffdfed0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Bounded capacity revision — uncompiled

Author: batch_design_review. Parent: immutable CUPTI V1/V2.

- New C++ artifact required; accepted e918ff2f V1 binary cannot satisfy this method.
- Only fixed limits 128→512MiB / 1,000,000→4,000,000 records and explicit schema/
  limit metadata change. Callback, activity kinds, provider symbols, timestamps,
  window, loss/status, original queue and model arithmetic remain byte-identical.
- Same ROOT_BUILD.md compiler/header procedure; source collector_capacity.cpp.
  ROOT build receipt must bind THIS freeze, source, actual argv/dependencies and
  full compiled artifact. No prewritten success receipt.
- Original raw_row overflow sets failure8; dropped/error/partial rows still fatal.
  Capacity is finite, not guaranteed sufficient for every canonical workload.
- Streamed validator retains finite dedup/correlation sets up to four million
  records; this is separate diagnostic parent RAM, potentially several GiB, not
  model admission. ROOT must reserve that host headroom before validation. No
  claim that CUPTI opaque internal allocation has been admitted by the engine.
- Container diagnostic only: 576MiB per-file RLIMIT_FSIZE and 768MiB total artifact
  tree, with 512MiB trace + metadata/log reserve. Original12GiB RAM/swap and four
  CPUs/model/provider/device policies unchanged. These are profiling artifacts,
  never canonical timed-run resource changes.
- Source-estimated ~236270 native dispatches for canonical C1 is NOT a measured
  driver count and does not qualify this revision. ROOT must build, run fixtures,
  then actual lossless workload and plain/observed transparency before acceptance.

Prepared ROOT fixtures: `python3 test_activity.py` (six synthetic tests, no GPU).
Author only AST-parsed; no tests/build/provider/runtime executed.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/capacity-v1/frozen/README.md

Original UTF-8 bytes: 1818; SHA256 `be65f53ffb83f50481731d59b288fefa5b1c8a739f4213f792dab6e23ffdfed0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Bounded capacity revision — uncompiled

Author: batch_design_review. Parent: immutable CUPTI V1/V2.

- New C++ artifact required; accepted e918ff2f V1 binary cannot satisfy this method.
- Only fixed limits 128→512MiB / 1,000,000→4,000,000 records and explicit schema/
  limit metadata change. Callback, activity kinds, provider symbols, timestamps,
  window, loss/status, original queue and model arithmetic remain byte-identical.
- Same ROOT_BUILD.md compiler/header procedure; source collector_capacity.cpp.
  ROOT build receipt must bind THIS freeze, source, actual argv/dependencies and
  full compiled artifact. No prewritten success receipt.
- Original raw_row overflow sets failure8; dropped/error/partial rows still fatal.
  Capacity is finite, not guaranteed sufficient for every canonical workload.
- Streamed validator retains finite dedup/correlation sets up to four million
  records; this is separate diagnostic parent RAM, potentially several GiB, not
  model admission. ROOT must reserve that host headroom before validation. No
  claim that CUPTI opaque internal allocation has been admitted by the engine.
- Container diagnostic only: 576MiB per-file RLIMIT_FSIZE and 768MiB total artifact
  tree, with 512MiB trace + metadata/log reserve. Original12GiB RAM/swap and four
  CPUs/model/provider/device policies unchanged. These are profiling artifacts,
  never canonical timed-run resource changes.
- Source-estimated ~236270 native dispatches for canonical C1 is NOT a measured
  driver count and does not qualify this revision. ROOT must build, run fixtures,
  then actual lossless workload and plain/observed transparency before acceptance.

Prepared ROOT fixtures: `python3 test_activity.py` (six synthetic tests, no GPU).
Author only AST-parsed; no tests/build/provider/runtime executed.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/README.md

Original UTF-8 bytes: 6258; SHA256 `05b5a2b700fc6190e2a4e9b0c7623cd27ed788d00aa99700a35c38404f421669`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUPTI same-open-FD mapping identity — additive V3

- Parent: serving-and-speculation; source-only owned proposal. Original container07a86bf6, capacity4337ad63, finalization2b82ca5f and all failed ROOT attempts stay immutable.
- Motivation only: ROOT pre-START metadata and separate readonly-FD diagnostic show the native/collector's stat device and kernel VMA device differ while inode/full hashes agree. CUPTI's observed devices agree. Those observations do not independently qualify the target mappings or any GPU activity.
- Eight new source/metadata files; no MAIN, collector, queue, model, CLI, workload, policy, capacity or owner changes. Author performed source/AST/byte checks only: no tests, proposed-module imports, mmap, DSO/model reads/hashes, provider/build/runtime/GPU execution.

## Exact identity rule

1. Authenticate the same original nonce/name/CID, actual init HPID, retained pidfd, start time and four namespaces. Preserve original host-network and SO_PEERCRED/namespace-PID separation.
2. Open each exact `/proc/HPID/root/<declared-native-or-collector-or-CUPTI-path>` readonly/CLOEXEC/NOFOLLOW. Keep this FD through full bounded64KiB-chunk SHA256 and fstat checks. Require the original pathname still denotes that FD. No dlopen or executable mmap.
3. Create only one temporary `MAP_PRIVATE|PROT_READ`, offset0, up to one system page, from that exact retained FD. Compare bounded complete local proc-maps snapshots before/during/after. Require one newly introduced nonoverlapping `r--p` page with the FD's exact kernel readlink/inode. Derive its device/major/minor/inode from the actual VMA, then prove that mapping disappears after explicit close. Coalescing/ambiguity/unsupported geometry refuses; no guessed address or filesystem exception.
4. Require every actual target mapping at the original declared container path to match that derived kernel key and a bounded original file extent. At least one executable segment must exist. All target mapping rows must remain unchanged across the proof. Original stat-device identity is retained separately as `device`/`stat_device`; `mapping_device` is the independently observed kernel value. Never replace or ignore either value.
5. Rehash the entire same FD and recheck all stat/path/readlink identities before closing it. Native additionally authenticates the original `/proc/HPID/exe` magic link, same file identity and two complete hashes through its own retained readonly FD. Preserve actual before/after observations, not inferred metadata.
6. Check the original live process binding before/after every artifact. The original owner alone performs process/container retirement; this controller still never signals, reaps, setns, adopts or changes privileges.

Independent V3 validator reconstructs the temporary VMA/key from all retained bounded local snapshots and rechecks every target mapping, same-FD hash/stat/path/exe/close receipt. These are authenticated executed-controller observations; source preparation alone proves no physical mapping.

Bounds: original576MiB artifact-file cap; hashes64KiB scratch; at most one readonly page at a time; each complete local maps snapshot128KiB/2048rows; target maps retain original2MiB bound. Three snapshots per artifact remain in the bounded controller receipt. It must fit the original4MiB metadata reader before writing. Original1GiB aggregate,768MiB owner,256-file and two reserved final receipts remain exact; no artifact capacity increase.

## ROOT review/execution only

1. Read eight sources, `REVIEW.diff`, source pins and unchanged originals. Review full FD identity/cleanup and all-original native activity/stock profiler controls.
2. From this directory, ROOT may run `python3 -m unittest -v test_fd_mapping_v3.py`. Prepared synthetic negatives include unequal/equal devices, mixed target key/extent, writable/shared/executable/nonzero-offset pages, preexisting/coalesced/ambiguous VMAs, remaining mappings, hash/stat/path/exe mutations, malformed bounds, and original foreign-peer refusal. Two fixtures exercise only a small synthetic ordinary file's readonly metadata mapping/refusal; never an ELF/provider/model.
3. Existing original test_contracts.py/test_completion_v2.py remain unchanged. No collector rebuild is required for this host-only derivative; exact existing capacity collector/build and CUPTI provider bindings stay mandatory.
4. After review/fixtures and an otherwise quiet owned GPU interval, use a fresh output directly under the original container-v1 directory:

```sh
python3 fd-mapping-v3/run_container_v3.py --run \
  --own-freeze 07a86bf698dc8c7e5568b85c3096ce9a1221b7a67e265bf1fe84e65ed255adee \
  --mapping-freeze EXACT_NEW_FREEZE_SHA256 \
  --launch-directory ORIGINAL_V8_LAUNCH_DIRECTORY \
  --collector-artifact-directory ROOT_CAPACITY_BUILD_DIRECTORY \
  --concurrency 8 --output NEW_OWNED_OUTPUT
python3 fd-mapping-v3/validate_container_v3.py \
  --own-freeze 07a86bf698dc8c7e5568b85c3096ce9a1221b7a67e265bf1fe84e65ed255adee \
  --mapping-freeze EXACT_NEW_FREEZE_SHA256 --output NEW_OWNED_OUTPUT
```

New mandatory `--mapping-freeze` binds these exact host sources; original `--own-freeze`, final-exec transport and all provider argv remain unchanged. V3 provenance is checked before/after execution and final writes, in both report and controller receipt. All original nonce/pidfd/start/ns/peer/path/hash, driver-ID, full HTTP transparency, native losslessness, stock trace, cgroup/caps, finite owner cleanup and final artifact reservations remain enforced. Full native/stock outputs and original loss/cleanup guards remain required after actual runtime.

Acceptance stays diagnostic-only: no timing, source-correct goodput, identical numerical profile, complete matched hardware counts, stock profiler losslessness or device-profiler allocation admission claim. The new mapping source proposal does not close any of those gates.

## Owned continuation

- [x] Read actual ROOT metadata and original controller/runner/validator/contracts/finalization.
- [x] Prepare additive V3 and source-free fixtures; static source/AST review only.
- [ ] Independent peer + ROOT full source review.
- [ ] ROOT actual fixtures, then fresh plain/observed four-owner execution and independent validation.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v3/frozen/README.md

Original UTF-8 bytes: 6258; SHA256 `05b5a2b700fc6190e2a4e9b0c7623cd27ed788d00aa99700a35c38404f421669`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUPTI same-open-FD mapping identity — additive V3

- Parent: serving-and-speculation; source-only owned proposal. Original container07a86bf6, capacity4337ad63, finalization2b82ca5f and all failed ROOT attempts stay immutable.
- Motivation only: ROOT pre-START metadata and separate readonly-FD diagnostic show the native/collector's stat device and kernel VMA device differ while inode/full hashes agree. CUPTI's observed devices agree. Those observations do not independently qualify the target mappings or any GPU activity.
- Eight new source/metadata files; no MAIN, collector, queue, model, CLI, workload, policy, capacity or owner changes. Author performed source/AST/byte checks only: no tests, proposed-module imports, mmap, DSO/model reads/hashes, provider/build/runtime/GPU execution.

## Exact identity rule

1. Authenticate the same original nonce/name/CID, actual init HPID, retained pidfd, start time and four namespaces. Preserve original host-network and SO_PEERCRED/namespace-PID separation.
2. Open each exact `/proc/HPID/root/<declared-native-or-collector-or-CUPTI-path>` readonly/CLOEXEC/NOFOLLOW. Keep this FD through full bounded64KiB-chunk SHA256 and fstat checks. Require the original pathname still denotes that FD. No dlopen or executable mmap.
3. Create only one temporary `MAP_PRIVATE|PROT_READ`, offset0, up to one system page, from that exact retained FD. Compare bounded complete local proc-maps snapshots before/during/after. Require one newly introduced nonoverlapping `r--p` page with the FD's exact kernel readlink/inode. Derive its device/major/minor/inode from the actual VMA, then prove that mapping disappears after explicit close. Coalescing/ambiguity/unsupported geometry refuses; no guessed address or filesystem exception.
4. Require every actual target mapping at the original declared container path to match that derived kernel key and a bounded original file extent. At least one executable segment must exist. All target mapping rows must remain unchanged across the proof. Original stat-device identity is retained separately as `device`/`stat_device`; `mapping_device` is the independently observed kernel value. Never replace or ignore either value.
5. Rehash the entire same FD and recheck all stat/path/readlink identities before closing it. Native additionally authenticates the original `/proc/HPID/exe` magic link, same file identity and two complete hashes through its own retained readonly FD. Preserve actual before/after observations, not inferred metadata.
6. Check the original live process binding before/after every artifact. The original owner alone performs process/container retirement; this controller still never signals, reaps, setns, adopts or changes privileges.

Independent V3 validator reconstructs the temporary VMA/key from all retained bounded local snapshots and rechecks every target mapping, same-FD hash/stat/path/exe/close receipt. These are authenticated executed-controller observations; source preparation alone proves no physical mapping.

Bounds: original576MiB artifact-file cap; hashes64KiB scratch; at most one readonly page at a time; each complete local maps snapshot128KiB/2048rows; target maps retain original2MiB bound. Three snapshots per artifact remain in the bounded controller receipt. It must fit the original4MiB metadata reader before writing. Original1GiB aggregate,768MiB owner,256-file and two reserved final receipts remain exact; no artifact capacity increase.

## ROOT review/execution only

1. Read eight sources, `REVIEW.diff`, source pins and unchanged originals. Review full FD identity/cleanup and all-original native activity/stock profiler controls.
2. From this directory, ROOT may run `python3 -m unittest -v test_fd_mapping_v3.py`. Prepared synthetic negatives include unequal/equal devices, mixed target key/extent, writable/shared/executable/nonzero-offset pages, preexisting/coalesced/ambiguous VMAs, remaining mappings, hash/stat/path/exe mutations, malformed bounds, and original foreign-peer refusal. Two fixtures exercise only a small synthetic ordinary file's readonly metadata mapping/refusal; never an ELF/provider/model.
3. Existing original test_contracts.py/test_completion_v2.py remain unchanged. No collector rebuild is required for this host-only derivative; exact existing capacity collector/build and CUPTI provider bindings stay mandatory.
4. After review/fixtures and an otherwise quiet owned GPU interval, use a fresh output directly under the original container-v1 directory:

```sh
python3 fd-mapping-v3/run_container_v3.py --run \
  --own-freeze 07a86bf698dc8c7e5568b85c3096ce9a1221b7a67e265bf1fe84e65ed255adee \
  --mapping-freeze EXACT_NEW_FREEZE_SHA256 \
  --launch-directory ORIGINAL_V8_LAUNCH_DIRECTORY \
  --collector-artifact-directory ROOT_CAPACITY_BUILD_DIRECTORY \
  --concurrency 8 --output NEW_OWNED_OUTPUT
python3 fd-mapping-v3/validate_container_v3.py \
  --own-freeze 07a86bf698dc8c7e5568b85c3096ce9a1221b7a67e265bf1fe84e65ed255adee \
  --mapping-freeze EXACT_NEW_FREEZE_SHA256 --output NEW_OWNED_OUTPUT
```

New mandatory `--mapping-freeze` binds these exact host sources; original `--own-freeze`, final-exec transport and all provider argv remain unchanged. V3 provenance is checked before/after execution and final writes, in both report and controller receipt. All original nonce/pidfd/start/ns/peer/path/hash, driver-ID, full HTTP transparency, native losslessness, stock trace, cgroup/caps, finite owner cleanup and final artifact reservations remain enforced. Full native/stock outputs and original loss/cleanup guards remain required after actual runtime.

Acceptance stays diagnostic-only: no timing, source-correct goodput, identical numerical profile, complete matched hardware counts, stock profiler losslessness or device-profiler allocation admission claim. The new mapping source proposal does not close any of those gates.

## Owned continuation

- [x] Read actual ROOT metadata and original controller/runner/validator/contracts/finalization.
- [x] Prepare additive V3 and source-free fixtures; static source/AST review only.
- [ ] Independent peer + ROOT full source review.
- [ ] ROOT actual fixtures, then fresh plain/observed four-owner execution and independent validation.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/README.md

Original UTF-8 bytes: 5444; SHA256 `e16d5115fc1908f2113bf433f1da07c835a14a7644fc22ba4502fa8d05ca7196`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUPTI safe-signal owner checkpoints — additive V4

- Parent serving-and-speculation; ROOT-only source proposal. Preserve V3 `14bdbe64`, its eight sources, all earlier freezes/attempts and original V8 Plan/Provider +53ee cleanup byte-identically. V3 still has the inherited asynchronous SIGTERM entrypoint and must not be executed as the repaired launcher.
- This derivative changes only host cancellation/source binding and adds control fixtures. Same V3 FD/mapping code, collector, providers, argv, policy, workloads, generation, counters, capacities and numerical checks. No MAIN changes.
- Copy `SigtermCancellation`1428B exact `00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128`. TERM/INT handlers record a flag; never asynchronously raise during Popen return/assignment, monitor/log construction, original cleanup, signals or reaps.

## Safe checkpoints

- `checked_sample` checks before constructing an owner, inside a wrapped preflight after original Provider retained Popen/monitor/log, after readiness/yield, and only after original Plan.sample finally finishes retirement. Existing Provider.start catches Exception to retry readiness; synchronous pending-cancellation delivery is KeyboardInterrupt, which escapes that retry handler into the unchanged BaseException cleanup boundary.
- Check around actual measurement and before the next provider. A recorded signal during an original bounded HTTP/readiness/idle operation is delivered at the next checkpoint; original transport timeouts remain unchanged. No per-token callbacks, global Popen monkeypatch, alternate owner or executor.
- No checkpoints run inside cleanup. Repeated signals only record state; original TERM/grace/two-KILL-before-reap, independent reap, nonce/CID rediscovery, logs/EOF and GPU retirement complete. Existing primary/cleanup errors retain precedence. A recorded signal marks the diagnostic UNQUALIFIED; V4 validation requires false recorded state and exact guard/source metadata.
- Original complete owner/identity/loss/workload/transparency/caps/mapping/finalization checks remain. New mandatory `--signal-freeze` binds ten V4 source/metadata files, exact guard and immutable V3 closure before/after execution/final writes.

## ROOT sequence, after active timing

1. Review ten files and REVIEW.diff. Author performed AST/source-byte checks only; no tests/import execution/mmap/process/container/provider/model/GPU/build or payload hashing.
2. From this directory, ROOT may run `python3 -m unittest -v test_signal_v4.py` (three pure mock protocol methods). They establish only checkpoint/exception/cleanup ordering, no physical lifetime evidence.
3. ROOT may run the nine real harmless cases through the existing private rootless user/network wrapper:

```sh
python3 run_harmless_v4.py --run-control-only \
  --signal-freeze EXACT_V4_FREEZE_SHA256 --new-output FRESH_CHILD_OF_THIS_DIRECTORY
```

   These use the **actual original V8 Plan.sample/Provider and real53ee nonce/name/CID Podman cleanup**, exact existing pinned image, foreground CLI, bounded log and original EOF/TERM/KILL/reap/absence stages. No mocked container discovery or cleanup. Only GPU quiet/monitor observers are explicit stubs; their receipts cannot pass real diagnostic validation.

   Cases: TERM/INT before Popen (no child), at retained-handoff/readiness, during an actually open harmless POST, at the observer stage of original Provider.close; plus postspawn/final receipt-I/O failure. Cleanup-phase signals are injected only by the allowed GPU-observer stub; original53ee cleanup is neither wrapped nor mocked. The stdlib payload is an exact existing copy: no device/model/source/native-binary mount, CUDA/NVIDIA disabled,256MiB/1CPU configured controls, pullnever/read-only/capdrop/no-new-privileges. `/v1/models` always503; no token bytes, model/native readiness, generation or GPU proof. No-body-yield assertion prevents accepting the control as a real provider. Each non-prelaunch case must have actually observed the genuine nonce/CID and actual control health before its injected event. Even the expected I/O-negative must retain real cleanup evidence separately and leave the original Plan blocked on missing final evidence.

   The nine-case report is **control owner evidence only**, not provider/cgroup/math/hardware/throughput qualification. The first two cases intentionally have no child; the remaining seven must retire actual named containers, groups and pipes. Native/provider readiness must stay false in every case. No actual GPU probe is performed.
4. After all reviews/fixtures and ROOT's quiet-resource authorization, run the same V3 diagnostic commands using `run_container_v4.py`/`validate_container_v4.py` and add `--signal-freeze EXACT_V4_FREEZE_SHA256`. Keep `--own-freeze07a86bf6…` and `--mapping-freeze14bdbe64…` unchanged. Use fresh original container-v1 outputs and genuine existing capacity collector/build.

Acceptance remains diagnostic-only. No timing, complete matched hardware counts, stock-profiler losslessness, numerical equivalence or device-profiler memory admission follows from this signal fix or the control fixtures.

## Owned continuation

- [x] Preserve V3 and accepted guard; prepare signal-only V4 plus three mocks/nine harmless real cases.
- [ ] ROOT + independent peer full source review.
- [ ] ROOT actual mock/control fixtures after timing, then fresh four-owner diagnostic and independent validator.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/fd-mapping-v4/frozen/README.md

Original UTF-8 bytes: 5444; SHA256 `e16d5115fc1908f2113bf433f1da07c835a14a7644fc22ba4502fa8d05ca7196`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUPTI safe-signal owner checkpoints — additive V4

- Parent serving-and-speculation; ROOT-only source proposal. Preserve V3 `14bdbe64`, its eight sources, all earlier freezes/attempts and original V8 Plan/Provider +53ee cleanup byte-identically. V3 still has the inherited asynchronous SIGTERM entrypoint and must not be executed as the repaired launcher.
- This derivative changes only host cancellation/source binding and adds control fixtures. Same V3 FD/mapping code, collector, providers, argv, policy, workloads, generation, counters, capacities and numerical checks. No MAIN changes.
- Copy `SigtermCancellation`1428B exact `00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128`. TERM/INT handlers record a flag; never asynchronously raise during Popen return/assignment, monitor/log construction, original cleanup, signals or reaps.

## Safe checkpoints

- `checked_sample` checks before constructing an owner, inside a wrapped preflight after original Provider retained Popen/monitor/log, after readiness/yield, and only after original Plan.sample finally finishes retirement. Existing Provider.start catches Exception to retry readiness; synchronous pending-cancellation delivery is KeyboardInterrupt, which escapes that retry handler into the unchanged BaseException cleanup boundary.
- Check around actual measurement and before the next provider. A recorded signal during an original bounded HTTP/readiness/idle operation is delivered at the next checkpoint; original transport timeouts remain unchanged. No per-token callbacks, global Popen monkeypatch, alternate owner or executor.
- No checkpoints run inside cleanup. Repeated signals only record state; original TERM/grace/two-KILL-before-reap, independent reap, nonce/CID rediscovery, logs/EOF and GPU retirement complete. Existing primary/cleanup errors retain precedence. A recorded signal marks the diagnostic UNQUALIFIED; V4 validation requires false recorded state and exact guard/source metadata.
- Original complete owner/identity/loss/workload/transparency/caps/mapping/finalization checks remain. New mandatory `--signal-freeze` binds ten V4 source/metadata files, exact guard and immutable V3 closure before/after execution/final writes.

## ROOT sequence, after active timing

1. Review ten files and REVIEW.diff. Author performed AST/source-byte checks only; no tests/import execution/mmap/process/container/provider/model/GPU/build or payload hashing.
2. From this directory, ROOT may run `python3 -m unittest -v test_signal_v4.py` (three pure mock protocol methods). They establish only checkpoint/exception/cleanup ordering, no physical lifetime evidence.
3. ROOT may run the nine real harmless cases through the existing private rootless user/network wrapper:

```sh
python3 run_harmless_v4.py --run-control-only \
  --signal-freeze EXACT_V4_FREEZE_SHA256 --new-output FRESH_CHILD_OF_THIS_DIRECTORY
```

   These use the **actual original V8 Plan.sample/Provider and real53ee nonce/name/CID Podman cleanup**, exact existing pinned image, foreground CLI, bounded log and original EOF/TERM/KILL/reap/absence stages. No mocked container discovery or cleanup. Only GPU quiet/monitor observers are explicit stubs; their receipts cannot pass real diagnostic validation.

   Cases: TERM/INT before Popen (no child), at retained-handoff/readiness, during an actually open harmless POST, at the observer stage of original Provider.close; plus postspawn/final receipt-I/O failure. Cleanup-phase signals are injected only by the allowed GPU-observer stub; original53ee cleanup is neither wrapped nor mocked. The stdlib payload is an exact existing copy: no device/model/source/native-binary mount, CUDA/NVIDIA disabled,256MiB/1CPU configured controls, pullnever/read-only/capdrop/no-new-privileges. `/v1/models` always503; no token bytes, model/native readiness, generation or GPU proof. No-body-yield assertion prevents accepting the control as a real provider. Each non-prelaunch case must have actually observed the genuine nonce/CID and actual control health before its injected event. Even the expected I/O-negative must retain real cleanup evidence separately and leave the original Plan blocked on missing final evidence.

   The nine-case report is **control owner evidence only**, not provider/cgroup/math/hardware/throughput qualification. The first two cases intentionally have no child; the remaining seven must retire actual named containers, groups and pipes. Native/provider readiness must stay false in every case. No actual GPU probe is performed.
4. After all reviews/fixtures and ROOT's quiet-resource authorization, run the same V3 diagnostic commands using `run_container_v4.py`/`validate_container_v4.py` and add `--signal-freeze EXACT_V4_FREEZE_SHA256`. Keep `--own-freeze07a86bf6…` and `--mapping-freeze14bdbe64…` unchanged. Use fresh original container-v1 outputs and genuine existing capacity collector/build.

Acceptance remains diagnostic-only. No timing, complete matched hardware counts, stock-profiler losslessness, numerical equivalence or device-profiler memory admission follows from this signal fix or the control fixtures.

## Owned continuation

- [x] Preserve V3 and accepted guard; prepare signal-only V4 plus three mocks/nine harmless real cases.
- [ ] ROOT + independent peer full source review.
- [ ] ROOT actual mock/control fixtures after timing, then fresh four-owner diagnostic and independent validator.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/finalization-v2/README.md

Original UTF-8 bytes: 1274; SHA256 `80c9edee3e13b74b952f635e0a31c44551c37363ea33de5ae736eb53f493c8c1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive final receipt artifact bound

Author: spine_review. Original container07a86bf6 and capacity4337ad63 immutable.

- New run_container_v2.py and validate_container_v2.py retain original CLI,
  activity/source/window/model/capacity/cgroup/workload/lifecycle/math behavior.
- Before EACH final receipt write: serialize finite JSON <=4MiB; full existing
  tree <=1GiB-8MiB and <=254files, reserving two4MiB receipt/file slots. Then
  exclusive write and full final tree assertion, including exact observed
  byte/file increments. No post-limit receipt can become accepted evidence.
- Aggregate1GiB/per-owner768MiB/per-file576MiB and raw512MiB/4M remain unchanged.
  Reservation is acceptance bookkeeping, not filesystem quota or device/RAM cap.
- New reports bind this additive manifest; validator refuses old unreserved
  reports. Native bootstrap/collector still attest unchanged07a/4337, correctly.
- ROOT first runs test_completion_v2.py (four prepared synthetic methods), then
  uses v2 runner/validator with identical original arguments. New capacity C++
  still requires actual ROOT build; no author execution occurred.

Author validation: four Python ASTs parse only, small source/frozen hashes.
No fixtures/build/runtime/provider/GPU/model or payload hashing.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/finalization-v2/frozen/finalization-v2/README.md

Original UTF-8 bytes: 1274; SHA256 `80c9edee3e13b74b952f635e0a31c44551c37363ea33de5ae736eb53f493c8c1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive final receipt artifact bound

Author: spine_review. Original container07a86bf6 and capacity4337ad63 immutable.

- New run_container_v2.py and validate_container_v2.py retain original CLI,
  activity/source/window/model/capacity/cgroup/workload/lifecycle/math behavior.
- Before EACH final receipt write: serialize finite JSON <=4MiB; full existing
  tree <=1GiB-8MiB and <=254files, reserving two4MiB receipt/file slots. Then
  exclusive write and full final tree assertion, including exact observed
  byte/file increments. No post-limit receipt can become accepted evidence.
- Aggregate1GiB/per-owner768MiB/per-file576MiB and raw512MiB/4M remain unchanged.
  Reservation is acceptance bookkeeping, not filesystem quota or device/RAM cap.
- New reports bind this additive manifest; validator refuses old unreserved
  reports. Native bootstrap/collector still attest unchanged07a/4337, correctly.
- ROOT first runs test_completion_v2.py (four prepared synthetic methods), then
  uses v2 runner/validator with identical original arguments. New capacity C++
  still requires actual ROOT build; no author execution occurred.

Author validation: four Python ASTs parse only, small source/frozen hashes.
No fixtures/build/runtime/provider/GPU/model or payload hashing.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/finalization-v2/pre-review/finalization-v2/README.md

Original UTF-8 bytes: 1274; SHA256 `80c9edee3e13b74b952f635e0a31c44551c37363ea33de5ae736eb53f493c8c1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive final receipt artifact bound

Author: spine_review. Original container07a86bf6 and capacity4337ad63 immutable.

- New run_container_v2.py and validate_container_v2.py retain original CLI,
  activity/source/window/model/capacity/cgroup/workload/lifecycle/math behavior.
- Before EACH final receipt write: serialize finite JSON <=4MiB; full existing
  tree <=1GiB-8MiB and <=254files, reserving two4MiB receipt/file slots. Then
  exclusive write and full final tree assertion, including exact observed
  byte/file increments. No post-limit receipt can become accepted evidence.
- Aggregate1GiB/per-owner768MiB/per-file576MiB and raw512MiB/4M remain unchanged.
  Reservation is acceptance bookkeeping, not filesystem quota or device/RAM cap.
- New reports bind this additive manifest; validator refuses old unreserved
  reports. Native bootstrap/collector still attest unchanged07a/4337, correctly.
- ROOT first runs test_completion_v2.py (four prepared synthetic methods), then
  uses v2 runner/validator with identical original arguments. New capacity C++
  still requires actual ROOT build; no author execution occurred.

Author validation: four Python ASTs parse only, small source/frozen hashes.
No fixtures/build/runtime/provider/GPU/model or payload hashing.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/frozen/README.md

Original UTF-8 bytes: 8713; SHA256 `2eee08c2526be0ef273c6b2ceb957b704f083f8c0e827ac7ee81322c5e949179`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native container activity diagnostic — source only

Author: batch_design_review. Parent plan: serving-and-speculation.

Own scope: this ignored child only. Immutable parents: CUPTI V1 f9f5c522, V2
932e471e and genuine matched-benchmark V8 b0c50e8a. No MAIN/parent edits, engine,
math, queue, source/model, timing workload or production profile changes.

## Files / ROOT execution

- `capacity-v1/`: separate bounded C++ revision + streamed validator + six
  synthetic fixtures. Capacity freeze4337ad63; new actual ROOT build required.
- `container_contracts.py`, `parent-bindings.json`: exact additive closure.
- `launch_derivative.py`: exact V8 argv + diagnostic-only mounts/env/bootstrap
  transport and finite file-size cap. Original model flags untouched.
- `native_exec.py`: run unchanged V8 bootstrap; its one final native exec uses
  `os.execve` with explicit collector env. Outer Python/Podman LD_PRELOAD remains
  empty. Stock always runs original bootstrap unchanged.
- `namespace_control.py`: original owner nonce/name/CID -> public inspect init
  HPID -> pidfd/start time/namespace readbacks -> host SO_PEERCRED. No signal,
  reap, setns, new privilege, foreign PID adoption, context or stream.
- `run_container.py`: four fresh original V8/53ee-owned processes: native plain,
  native observed, stock plain, stock public Torch-profiler observed.
- `container_activity.py`, `capacity_v1_loader.py`, `validate_container.py`:
  independent full raw activity/wire/namespace/owner/canonical-argv validation.
- `test_contracts.py`: seven prepared synthetic transport/control negatives.
  No synthetic fixture establishes physical collection or model mathematics.

ROOT first reviews/freezes, then executes only after timing quiet + reservation:

```text
python3 capacity-v1/test_activity.py
python3 test_contracts.py
```

Build `capacity-v1/collector_capacity.cpp` with immutable parent ROOT_BUILD.md
procedure and original approved headers/compiler. Record actual dependency/ELF
closure, argv/source/build/artifact hashes. Fresh artifact directory contains
`collector.so` and actual `build.json` with:

```json
{
  "status": "ROOT_COMPILED_COLLECTOR_ONLY",
  "freeze_sha256": "4337ad630552e0b46644cfb39d8198c6aad94cb3b48d85c8402a3a315a247bdd",
  "collector_source_sha256": "ACTUAL_CAPACITY_CPP_SHA256",
  "collector_sha256": "ACTUAL_COMPILED_ARTIFACT_SHA256",
  "exit_code": 0
}
```

Do not prewrite success. The accepted e918ff2f host V1 artifact is intentionally
refused for this revision. Build can preserve original callback/API/math bytes;
only fixed limits128→512MiB, records1M→4M, schema and explicit capacity metadata
change. No runtime environment override can raise limits.

Use an existing genuine V8 launch directory (exact protocol/workload/launches/
input-attestation), not a shortened host-control prompt. Set ROOT explicitly:

```text
ZG_BENCH_HOST_ROOT=/home/pvl/spaces/abird/src/zerogravity
python3 run_container.py --run --own-freeze OWN_FREEZE_SHA256
  --launch-directory ABSOLUTE_ORIGINAL_V8_LAUNCH_DIRECTORY
  --collector-artifact-directory ABSOLUTE_FRESH_ROOT_CAPACITY_BUILD_DIRECTORY
  --concurrency ORIGINAL_PROTOCOL_CONCURRENCY
  --output ABSOLUTE_THIS_DIRECTORY/root-container-1
python3 validate_container.py --own-freeze OWN_FREEZE_SHA256
  --output ABSOLUTE_THIS_DIRECTORY/root-container-1
```

Invocation is direct argv under ROOT's existing original process supervisor,
not shell interpolation; export the explicit host-root variable. Author has
not imported, run fixtures, compiled or executed any provider/container/workload.
Only Python AST parsing and small source/metadata hashing/freezing occurred.

## Identity, window, policy

- Actual container init HPID comes from nonce/name/CID-bound public inspect;
  Podman Popen PID remains its original cleanup owner, not the collector PID.
- Exact host peer HPID + ordinary host UID must match SO_PEERCRED. pidfd remains
  live; public container binding, proc start time and namespaces match before/
  after each START/STOP. Output retains actual UID/GID maps and namespace IDs.
- Reported `getpid()` is namespacePID1. Raw CUPTI driver IDs are never rewritten:
  exactly one ID, equal to authenticated HPID OR namespacePID1; reject mixed or
  foreign IDs and preserve which representation actually occurred. Metadata and
  driver identities use separate validator arguments.
- Original `--network=host` supplies abstract UNIX reachability; actual net-ns
  equality and user-namespace peer translation must pass. Any hidepid/proc,
  pidfd, mapped-file, permission or provider denial refuses; no privilege repair.
- Actual native/collector/CUPTI mapped inode/dev/full SHA are checked through
  `/proc/HPID/root`; CUPTI C++ still resolves only official versioned symbols
  from exact original provider. Remaining original native/provider dependencies
  retain existing V8/ROOT build closure; no substitute DSO.
- Both native modes inherit exact V8 bootstrap/source/affinity/cgroup checks.
  Stock uses original profiler-config and public `/start_profile`/`stop_profile`;
  no stock math/module/queue patches or CUPTI preload.
- `cold_process`: no collector warmup request. `warmed_process`: ONE unchanged
  V8 unmeasured request with first original prompt and original max_tokens.
  OS/storage/device caches are not evicted or claimed cold. Official collector
  initialization overhead itself is diagnostic, not a canonical timing sample.
- Native START follows original terminal/idle health; all canonical requests
  then complete; native terminal/idle health precedes STOP/forced flush.
  All source native dispatch counters remain distinct from CUPTI records.
- Same canonical request count/prompts/arrivals/concurrency/max_tokens, CPU4,
  RAM12GiB/swap-total12GiB, device, model/provider limits, stock eager/default
  route and generation policy. No one-token collector warmup or workload edit.
- Four actual owner receipts/monitor/containers/wholegroups/readers/EOF must
  retire under unchanged reviewed53ee WNOWAIT cleanup before next process.
  Primary/stop/close/post-source failures remain fatal with original receipts.

## Capacity and remaining physical gates

512MiB raw /4,000,000 records are finite, not a guarantee every canonical wave
fits. Any buffer refusal/drop/overflow/partial record/API/timestamp mismatch
refuses all acceptance. Entire native file is streamed twice, exact duplicate/
correlation/totals checked. Validation sets are finite but may use several GiB
of separate host RAM: ROOT reserves headroom before validation; not engine
provider admission. Kernel count, successful direct launch APIs, graph launch
APIs and kernels executed from graphs stay separate.

Diagnostic-only per-file soft=hard RLIMIT_FSIZE is576MiB. Each owned sample tree
must be<=768MiB (capacity README's native-sample bound); four-sample aggregate is
<=1GiB to hold512MiB native raw + original stock trace<=256MiB + metadata/logs.
Tree limits are post-write acceptance checks, not a kernel filesystem quota.
File limit is enforced by original Podman public ulimit; native wrapper records
actual readback. Original CPU/RAM/model limits are unchanged. Stock trace retains
original256MiB compressed/expanded bounds; exceeding any bound is refusal.

Before claiming a method: ROOT must qualify actual new build/13fixtures,
namespace/user-peer/mapped closure, exact plain/observed wire IDs/text/finish/
usage, real workload window and no activity loss, actual caps, final process/
container/group/pipe/monitor closure. Negative tests must include foreign peer,
wrong nonce/namespace/stale PID, provider permissions/cold refusal, failed STOP,
interruption/receipt I/O, actual overflow/drop behavior and no accepted partial
trace. Synthetic corrupted metadata fixtures validate refusal only; they do not
establish actual callback overflow/CUPTI drop behavior.

Root source estimate236270 dispatches for C1/48prompt/output8 suggests old128MiB
cap too small. This is source metadata, NOT a canonical hardware measurement or
speed result. Never run old undersized artifact and accept a prefix of records.

Actual small host V2 PASS64328 kernels remains scoped to that old method. No
container, larger-capacity, matched counts or new performance claim follows.
Stock public Torch profiler currently does NOT prove losslessness/all-worker/
complete initialization coverage; validator deliberately returns
`matched_complete_hardware_kernel_counts_qualified=false` even if both recorded
workload scopes pass. These diagnostic data do not establish numerical profile
parity, source-correct goodput or a timing speedup. Opaque CUPTI internal device
allocation admission remains unqualified and observed resource cost is separate.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/frozen/TODO.md

Original UTF-8 bytes: 1651; SHA256 `2debefd7c25d04d1648efabed5412ee198353a3096f4fb1929c9a34536d94cca`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUPTI container diagnostic / capacity revision

Parent: serving-and-speculation. Owner: batch_design_review.
Scope: ONLY this ignored child; no production/parent/plan/README changes.

- [x] Read immutable CUPTI V1/V2 + exact V8 canonical launch/bootstrap/owner/
  diagnostic/trace/cgroup source; keep original53ee lifecycle unchanged.
- [x] Prepare separate512MiB/4M C++ cap revision and six source-only negatives;
  exact callback/API/math unchanged, new compiled artifact mandatory.
- [x] Prepare final-native-exec-only seam; host PID/namespace PID/raw driver PID
  distinct; original owner/caps/source/full canonical workload/cold-warm policy.
- [x] Prepare four-process plain/observed native+stock transparency and independent
  complete activity/argv/owner validator; seven container transport fixtures.
- [x] AST parse only; no imports, fixtures, compilation or runtime by author.
- [ ] Independent source review / ROOT actual fixture+strict C++build+dependency
  closure / actual process namespace+permissions/metadata/STOP/loss qualification.
- [ ] ROOT actual canonical workload/negative failure+overflow/retirement tests.
- [ ] Stock collector loss/completeness gate; no matched complete HW claim yet.
- [ ] Any actual measured gain/numerical equivalence/fullmodel claim is external;
  none established by this source proposal.

Failures: no executed test/build/runtime. Source-only inspection corrected a
post-START receipt-I/O cleanup gap before freeze; STOP flag now arms immediately
on acknowledged START. Post-run source/affinity errors downgrade report before
independent validation rather than leaving an executable-success status.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/frozen/README.md

Original UTF-8 bytes: 8920; SHA256 `c76986f2f28a6d8d3ab98eed71d5fb11d0cf6bf0f7e7189f1146ff4f0807a76a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Official CUPTI activity diagnostic

Author: batch_design_review. Source-only, uncompiled/unexecuted.
Parent: existing serving-and-speculation plan; diagnostic only.

Official CUPTI12.9.79 library **and headers are already present** in the approved
local image/cache. No SDK/tool/model download is needed. Host `nsys`/`ncu` are
absent from PATH; inspected image Nsight prefixes contain no candidate tools.
`LOCAL_AVAILABILITY.json` records actual local paths/metadata and header pins.
This is a CUPTI activity candidate, not an available Nsight CLI workflow.

## Exact available source

Cache site root:
`/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/7940bf3c960e46bcf6f5848ed00b3dadaa7d399d891836cd12dcd4746cb4c8cf/diff/usr/local/lib/python3.12/dist-packages`

- `nvidia/cuda_cupti/include/{cupti_activity,cupti_callbacks,cupti_version,cupti_driver_cbid,cupti_result}.h`
- `nvidia/cuda_cupti/lib/libcupti.so.12`:7,604,800B; accepted prior full SHA
  `2fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a`.
  Author checked current metadata/static ELF, not a new DSO payload hash/load.
  ROOT launcher must rehash the actual full library before and after collection.
- Public CUDA headers under `nvidia/cuda_runtime/include` and
  `nvidia/cuda_nvcc/include`, also already local.
- Package metadata: `nvidia_cuda_cupti_cu12-12.9.79.dist-info`; API macro28.
  Exact default exported symbols are versioned `@@libcupti.so.12`; candidate
  uses public `dlvsym(...,"libcupti.so.12")`, then checks same file dev/ino/size.
- Derived image `f57d77f5...` / manifest `eec72f47...` inherits this base layer;
  image metadata/source lookup establishes availability, not runtime support.

## Collector and queue boundary

- Diagnostic ELF `LD_PRELOAD` constructor, not a guessed CUDA injection entry.
  No interception or replacement of CUDA/model/math functions. Only exact target
  native executable dev/ino is observed; inherited compiler/helper execs are
  inert. ROOT original process ownership handles shutdown and reap.
- Official Activity API `CONCURRENT_KERNEL` + `DRIVER` are enabled before native
  server main/CUDA initialization. **Never KIND_KERNEL:** local header says it
  serializes kernels. No counters/metrics/PC sampling or profiler replay.
- Whole-process capture initialization preserves graph metadata visibility, but
  records retained to disk are selected by actual CUPTI START/STOP timestamps.
  Startup kernels alone cannot satisfy the window workload check.
- Native/Ready/ResourceArbiter/original queue and completion are unchanged. The
  collector never acquires another CUDA context/stream, submits GPU work, reads
  model/state memory, synchronizes the CUDA device or grants new authority.
- `FlushAll(FORCED)` blocks delivery **without CUDA synchronization**. ROOT must
  first obtain successful complete HTTP/SSE outputs and two original health
  snapshots with active/queued/parked0. Every retained record must then have
  complete nonzero timestamps, lie inside the declared window and match totals.
  Boundary-spanning GPU work/incomplete records are a refusal, not rounded away.
- Actual kernel activity records are distinct from host driver launch API calls.
  A CUDA Graph launch can yield many kernel records. Validator keeps both
  domains, public correlation IDs, graph/node IDs, grid IDs and original PID.

## Bounds, loss and lifetime

-16 reusable aligned1MiB callback buffers; no unbounded queue/record vector in
  the collector. Official requested device buffers1MiB, pool8, preallocate2,
  pinned-host allocation0; every attribute must read back byte-exact.
- Header explicitly warns actual device allocation can exceed requested buffer
  size. Device/RAM/control/CUPTI worker-thread overhead is diagnostic external
  overhead, **not a proved model-admission allocation bound**. No production
  memory/throughput claim follows from these buffer controls.
- Serialized output<=128MiB/1,000,000 records; names<512B; invalid/unknown kinds,
  null/oversized buffers, iterator errors, zero timestamps, dropped records,
  output cap/disk errors or nonzero CUPTI status make the trace refuse.
- Dropped counters are reset on read by CUPTI: callback+final values are summed;
  no silent loss allowed, including startup capture. Completion callbacks are
  global in these headers, so no deprecated per-stream-buffer assumption.
- One process-owned controller thread, private random abstract UNIX seqpacket
  name, same-UID SO_PEERCRED checks, exact nonce and two bounded commands.
 600s control deadline +5s peer I/O; ROOT client uses30s. No detached external
  helper, second model process, arbitrary CUDA command or exposed stream pointer.
- On STOP, kinds are disabled, forced flush completes, no buffer may remain
  busy, log is fsynced and exact metadata persisted/acknowledged. CUPTI DSO,
  registered callbacks/buffers and controller owner remain retained until the
  **original server process is actually reaped**. No unsafe mid-process
  `cuptiFinalize`/unload is attempted. Original lifecycle tears down every
  success/failure path with finite startup/shutdown and refuses incomplete
  owned group/log/readiness/endpoint cleanup.

## ROOT execution only

1. Review `collector.cpp`, Python source, SDK/API pins and lifecycle composition.
   Run `python3 test_contracts.py` only on ROOT's serialized source-free schedule.
2. Compile the new collector under ROOT's existing isolated compiler ownership,
   using `ROOT_BUILD.md`. No author compilation/runtime occurred. Retain actual
   compiler/source/library/artifact identity and errors; new attempt for fixes.
3. Supply an existing approved lifecycle spec whose native direct argv begins
   with the exact frozen native binary and whose second vLLM endpoint is unused.
   This helper launches **native only**, twice, serially. Full declaration/source
   acceptance stays in that original owning spec. No startup/measurement samples
   are added to the actual unprofiled timing run.
4. Execute `run_probe.py` with the documented args below, then independently
   `validate.py --output` using the same fresh absolute output directory.
   Both fresh server processes/warmups/HTTP workload must complete normally.

   ```bash
   python3 run_probe.py \
     --lifecycle-spec /absolute/approved-native-and-unused-vllm-spec.json \
     --binary /absolute/frozen/zerogravity --binary-sha256 ACTUAL_NATIVE_SHA \
     --collector /absolute/collector.so --collector-sha256 ACTUAL_COLLECTOR_SHA \
     --collector-build /absolute/ROOT-collector-build.json \
     --url http://127.0.0.1:18000 --unused-vllm-url http://127.0.0.1:18001 \
     --model APPROVED_NATIVE_ROUTE --requests 2 --concurrency 2 --max-tokens 4 \
     --new-output /home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/root-attempt-1
   ```

The ROOT build receipt must contain actual `status=ROOT_COMPILED_COLLECTOR_ONLY`,
freeze_sha256, collector_sha256 and successful compiler exit_code0; retain its
actual command/log beside it. The author supplies no invented binary receipt.

The helper preserves the original serving tool benchmark schema but labels all
results diagnostic-only. Plain/observed must match every request's text hash,
wire token IDs, finish reason and usage; all offers must complete. The workload
must increase actual native accepted output tokens between original idle
snapshots. Raw kernel records/launch correlations in the timestamp window then
establish activity for declared HTTP work, not source launch-site estimates.

## Acceptance and remaining scope

- Candidate first receipt: `PASS_NATIVE_CUPTI_WORKLOAD_ACTIVITY_DIAGNOSTIC_ONLY`.
  Original CPU/model numerical gates are not newly proved by output transparency.
- ROOT must physically test unsupported CUPTI/permissions, loss/cap/disk errors,
  killed/disconnected control, server error/timeout and failed cleanup. A failed
  collector never falls back to an unobserved successful trace.
- ROOT must qualify B1/B2/B4/B8 and graph replay activity cardinality where used,
  and separately the pinned real vLLM collector under its namespace/lifetime.
  This first host-native seqpacket/peer-PID design does not pretend to work
  unchanged across a container PID/network/user namespace.
- No new SDK/device privilege changes. Existing driver/device restrictions
  refuse unchanged. Hardware comparison remains blocked until both collectors
  and the matched model/source/profile/resource/performance gates pass.
- All timing claims use separate unprofiled samples. Tracing costs, extra
  driver API records, buffers and synchronization/delivery effects are reported
  as observer overhead; no speedup inferred from this diagnostic.

`startup-v1` provider containment freeze961d803b is untouched and unrelated to
this profiler's production eligibility. No MAIN/shared plan/root README edit.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/frozen/ROOT_BUILD.md

Original UTF-8 bytes: 1720; SHA256 `a686ec1fbe6b9f8c814e8a25c54198e3d62935238d7446fff3de471b5b527c54`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT collector build

Source-only preparation. Existing C++ compiler + already approved local headers;
no SDK install, GPU context, CUDA compiler or model is needed for this build.

Under ROOT's original isolated/serialized compiler supervisor, invoke direct
argv as represented below. `SOURCE`, `OUTPUT` and `SITE` are absolute task paths,
not shell text interpolated from an untrusted request.

```text
c++ -std=c++20 -O2 -fPIC -shared -Wall -Wextra -Werror -Wformat=2
  -fvisibility=hidden -Wl,-z,now
  -isystem SITE/nvidia/cuda_cupti/include
  -isystem SITE/nvidia/cuda_runtime/include
  -isystem SITE/nvidia/cuda_nvcc/include
  SOURCE/collector.cpp -ldl -pthread -o OUTPUT/collector.so
```

`SITE` is the exact absolute local image-layer site root in README. Root include
and dependency scan must remain bound to those local official header bytes;
headers are not copied/replaced. Symbols are resolved at runtime from the one
explicit exact CUPTI library, not statically linked to CUDA or another provider.

Record actual compiler/argv/log/exit and the compiled library full hash; verify
actual ELF dependencies before execution. These are profiling-tool dependencies
only, separate from the native model numerical source and unprofiled timing.
The launcher requires a ROOT actual build JSON:

```json
{
  "status": "ROOT_COMPILED_COLLECTOR_ONLY",
  "freeze_sha256": "ACTUAL_SOURCE_FREEZE_SHA",
  "collector_sha256": "ACTUAL_COMPILED_COLLECTOR_SHA",
  "exit_code": 0
}
```

Do not create that success receipt before the actual build. Failed build/runtime
attempts remain separate and unchanged; additive source corrections require a
new freeze/binary/output. Author ran no compiler, source-free fixture or CUPTI.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/frozen/TODO.md

Original UTF-8 bytes: 1372; SHA256 `ceb6c762953075f0bd0ae97a4312c545d2cf6839c5e6e106c92596e212b53e93`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUPTI diagnostic preparation

Parent: serving-and-speculation. Owner: batch_design_review.
Scope: this ignored source-only proposal, original native server unchanged.

- [x] Find existing official local CUPTI12.9.79 headers/library/API28.
- [x] Inspect local image ancestry, package metadata and public versioned ABI.
- [x] Confirm no host nsys/ncu PATH entry or inspected image-prefix candidate.
- [x] Prepare bounded concurrent-kernel/driver activity collector and window.
- [x] Reuse original owned lifecycle + serving benchmark; no second queue/engine.
- [x] Prepare independent cardinality/loss/transparency/cleanup validator tests.
- [x] Document exact overhead/flush/lifetime and remaining physical scope.
- [ ] ROOT independent full source review.
- [ ] ROOT source-free schema fixtures/strict C++ compile/ELF/payload binding.
- [ ] ROOT actual plain/observed workload, full wire transparency/cleanup.
- [ ] ROOT actual failure/loss/cap/unknown/permission and B1/2/4/8/graph controls.
- [ ] Independently qualify pinned vLLM collector/namespace/lifetime.
- [ ] Matched hardware comparison with separate unprofiled timing, owning gate.

Author results: source/header/metadata/static ELF reading, Python AST and small
source hashes only; no builds/tests/CUPTI load/operator/model/GPU/container
execution, payload hashing, SDK download, MAIN/plan edits or commits.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/frozen-future-seam/CONTAINER_FUTURE_SEAM.md

Original UTF-8 bytes: 2138; SHA256 `c1ba0692eb6541368c4564bf2768d7f8d4673b84cd1fc05ffd7569e4ae7333ea`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Future container diagnostic seam — not implemented

Author: batch_design_review. Additive note after immutable source freezef9f5c522.

Current proposal is **host-native only**: direct native binary argv, same-UID
abstract UNIX control peer and exact host Popen/native PID. It does not establish
container collector support, and genuine matched benchmark timing remains
separate/unprofiled.

ROOT's actual matched native environment runs through a container Python wrapper
which ultimately `os.exec`s the original native program. A later diagnostic
derivative must set LD_PRELOAD and collector-only env **at that final native
exec**, leaving outer Python/container management unprofiled. Do not place the
collector on outer Python or benchmark-main argv and infer that exec preserved
a single initialized collector. An ordinary constructor can otherwise run
before and after exec, leaving a stale initialized control endpoint.

This frozen candidate already checks `/proc/self/exe` against the explicit
native target inode and is inert for another executable. That defensive host
guard is not container namespace/lifecycle qualification or a reason to skip the
explicit final-exec-only seam.

Future implementation requires an independently reviewed controller in the
original owned container's actual PID/network/user namespace, authenticated to
the exact original native process and nonce. Host Popen PID identifies Podman,
not its inner native PID; abstract UNIX sockets and SO_PEERCRED are namespace
dependent. Reuse original nonce/CID/owner/retirement receipts; no foreign process
lookup, setns privilege change, separate model engine or arbitrary context/stream
API. Controller placement/startup/flush/retirement must be physically qualified
before any container activity counts or matched hardware comparison claim.

Unsupported activity/driver/CUPTI permissions must refuse cold. No device
privilege, driver recovery, privileged counter enablement, SDK installation or
silent fallback is authorized by this profiler proposal. First ROOT execution
remains the limited host-native source freeze, after timing quiet/reservations.

````

## Source: .agents/runs/serving-and-speculation-cupti-activity-20261020/frozen-v2/LIFECYCLE_V2.md

Original UTF-8 bytes: 4560; SHA256 `e0b8ccfe7f87f7286fe523f01bb6a974541018fbe9de7d80abafc105e06e46a0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive native owner V2

Original `f9f5c522…`/13 files/frozen copies remain immutable. This fixes the
original launcher's lifecycle defect; it does not change `collector.cpp`.
Original `serve-lifecycle.py` uses `poll()` during startup and `wait()` before a
later group signal. That can reap the leader before the last nonzero PGID
signal. V1 must not execute a server.

## Narrow composition

- `run_probe_v2.py`: canonical schema/endpoints/bench/preflight/warm→idle→START→
  workload→idle→STOP and plain→observed order unchanged. Actual native argv is
  still direct; no shell, model wrapper or alternate CUDA queue.
- `native_owner_v2.py`: subclasses only canonical Plan bookkeeping; replaces
  its `sample`/provider lifetime. Never calls original OwnedProvider, BoundedLog,
  `poll`, `wait`, `communicate`, start or close.
- `owned_container_v4.py`: exact reviewed `53ee4270…` source. Only public
  `leader_status_unreaped`, `cleanup_group` and bounded main-thread `Log` used.
  No Podman/container command is invoked by this native-only adapter.
- Unreaped child identity remains reserved through startup, canonical client
  calls and all nonzero group signals. Original `start_new_session=True` and
  authenticated SID/PGID/child checks bind both KILL attempts before ANY reap.
  Only signal 0 is used after reap; final group absence required.
- One non-daemon canonical client thread lets the main thread continuously
  drain bounded logs and observe the server with `waitid(WNOWAIT)`. No client
  function/math/HTTP policy is replaced. A deadline/unjoined client bypasses
  readiness retry; the original server is retired before joining that client.
- Effective retained-log cap = min(canonical declared cap, reviewed 4MiB cap).
  Any overflow/EOF/I/O/endpoint/unknown-client failure refuses acceptance and
  blocks a next provider. No output truncation accepted.
- Cleanup attempts group retirement, client join, complete log/EOF/descriptor
  closure and endpoint absence independently. Evidence I/O occurs AFTER actual
  cleanup. Primary and cleanup failures are retained together. SIGTERM raises
  the same cleanup path as an interrupt; SIGKILL/parent-death is not qualified.
- `validate_v2.py`: both actual receipts must bind this adapter/helper, ready
  success, two pre-reap KILL attempts, leader reap/group absence, complete log
  byte/hash/EOF, joined client and absent endpoints. No primary/evidence error.
  Original GPU-activity reducer is AST-identical to V1; API launch calls and
  real kernel records remain separate counts.

## ROOT-only gates

Author only read/edited source, parsed Python ASTs and hashed source/metadata.
No fixtures, build, child process, CUPTI/CUDA/DSO/provider/operator/model/GPU or
container executed by author. No native library payload hashed by author.

ROOT may run `python3 test_native_owner_v2.py` and original
`python3 test_contracts.py` under its serialized supervisor. Seven new fixtures
use real harmless owned Python children: normal/TERM-resistant, exited leader,
client pending until teardown, failed log construction, persistent postspawn
receipt failure, measurement interrupt, and a static no-early-reap check. These
are process-lifetime controls only; no CUDA or numerical proof.

Then ROOT uses `run_probe_v2.py` with the original arguments and a FRESH output,
followed by `validate_v2.py --output ABS`. Keep original V1 failures/output and
compiled artifacts unchanged. `root-build-1/build.json` is an actual V1 C++ build;
V2 explicitly binds its original freeze/source/artifact hash, because C++ bytes
are unchanged. V2 execution freeze is recorded separately; no invented new
compilation receipt or source rebuild claim.

A real native diagnostic still requires exact binary/CUPTI full-hash binding,
plain/observed wire transparency, positive declared-window GPU kernels, no loss
and complete actual retirement. Unsupported CUPTI/permissions fail closed; no
privilege change. B1/B2/B4/B8/graph/failure controls, hardware overhead, container
collection and vLLM collector qualification remain ROOT physical gates.
Unprofiled matched timings remain separate. This is not performance acceptance,
new numerical support, source/full-StateRoot proof or provider allocation
admission. The adapter proves only its own unreaped leader/PGID boundary; any
escaped process/session or unjoined thread is not silently claimed retired.

Future container seam stays in immutable `CONTAINER_FUTURE_SEAM.md`: preload
only final native exec, authenticated namespace controller, separate scope.

````

## Source: .agents/runs/serving-and-speculation-cupti-fd-mapping-review-batch-20261020/REVIEW.md

Original UTF-8 bytes: 2435; SHA256 `a53dd6601caa97772585c5aa69d6cbebbe418724f47c8da6f54129203bed7ee6`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUPTI FD mapping V3 independent source review

Author: batch_design_review. Parent: serving-and-speculation.
Reviewed14bdbe64496a25cecac4f8bd7144ec61c8e160c127c9c3273c0d066e14946a71.

- No blocker found in the new FD/mapping identity or independent replay scope.
- Read complete273-line mapping helper, controller/contract/independent validator,
  runner delta, all15 prepared fixtures and finalization integration.
- Exact retained readonly/NOFOLLOW FD regular identity + full before/after hashes,
  stable pathname and private nonexecuting offset0 map; full local map snapshots
  prove one fresh page and explicit retirement. Kernel device derives from that
  actual page; stat device remains separate. Native exe retains its own kernel
  magic-link FD/hash/stat identity; target rows require exact derived key/extent.
- Original nonce/CID, start/pidfd/namespaces, SO_PEERCRED and driver-ID separation,
  loss/whole-workload/plain-observed/physical-observer/EOF and final reserved
  receipts remain enforced; no privileges, model/math/queue changes.
- Independent validator rederives local witness/key, actual target rows and
  exe/hash/stat/path/close receipts. Target-row completeness still relies on the
  frozen executed controller capturing all matching rows; complete target proc
  maps are not retained. No continuous memory/code-page attestation is claimed.
- Bounds: local retained snapshots<=128KiB/2048 rows; target proc maps<=2MiB;
  full hashes64KiB chunks; total controller encoded metadata<=4MiB. maps_text
  reads cap+1 Unicode characters then checks encoded bytes: accepted metadata
  is byte-capped; transient decode allocation can exceed that byte cap (bounded
  by UTF8 expansion). No whole-controller physical RAM admission claim follows.
- Source IO checks: all8 live/frozen sources, review diff and29 old source/small
  diagnostic metadata pins exact; six Python ASTs/15 test methods parsed only.
  Original controller text/identity/live/peer/close and old owner/workload/driver
  validator functions AST-identical. No imports/mmap/tests/build/runtime or
  binary/DSO/model/trace payload reads/hashes.
- KNOWN BLOCKER outside mapping delta: inherited async SIGTERM across Popen or
  retirement remains unsafe; ROOT already assigned additive record-only V4.
  Preserve V3 and require V4 independent review/actual harmless tests before
  actual four-owner collector execution. No GPU mapping/count proof here.

````

## Source: .agents/runs/serving-and-speculation-cupti-fd-mapping-review-batch-20261020/TODO.md

Original UTF-8 bytes: 344; SHA256 `c9f3eda62b31523bd03d1fcdb4ea3e63ebe486bb4dad9ef00d74ce664d04377e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12 CUPTI FD mapping review

Parent: `.agents/plans/serving-and-speculation/TODO.md`; owner batch_design_review.

- [x] Full V3 source/metadata review, hashes and AST-only checks; no new blocker.
- [ ] Additive V4 lifecycle source review when frozen/assigned.
- [ ] ROOT actual fixtures and collector execution/validation; no author runtime.

````

## Source: .agents/runs/serving-and-speculation-cupti-fd-mapping-review-batch-20261020/V4_REVIEW.md

Original UTF-8 bytes: 2398; SHA256 `c9979563c0e71328093f52d8fc803599ca88b9753aeab9d808fc4adb39d87e14`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CUPTI V4 independent signal-only review

Author: batch_design_review. Source freeze9f9051cfa137d590faa437266d3cb3c24efed86102afe5c06d04f2e4e7bee149.

- No source blocker found in record-only guard/checkpoint and original owner
  composition. V3 mapping review remains applicable, immutable14bd bytes exact.
- Exact00d1 TERM/INT recorder spans main/owner handoff/retirement. Before-owner
  checkpoint + wrapped preflight only after original Popen/monitor/log retention
  + body boundary checks; KeyboardInterrupt escapes original readiness Exception
  retry to unchanged original Plan BaseException/finally. No raising handler.
- No checkpoint runs in original cleanup. Primary/cleanup failures propagate
  without a later checkpoint replacing them. Cleanup success plus recorded signal
  raises after retirement, stopping next provider; Plan blocked persists on lost
  final evidence. Original53ee WNOWAIT/signals-before-reap/rediscovery/EOF retained.
- Runner source checks and validator bind mandatory V4+V3+guard closure; recorded
  cancellation makes status UNQUALIFIED and is rejected by independent validator.
  Signals after report snapshot can cause a final nonzero exit; ROOT must require
  actual process success as always, not accept a standalone stale report marker.
- Three mock tests target checkpoint/primary/cleanup ordering only. Nine genuine
  harmless cases use original Plan/Provider/53ee with real container discovery and
  cleanup; only labelled GPU-observer quiet/Monitor stubs. Models always503, no
  tokens/output/native readiness; seven actual container cases/two prelaunch cases.
  Request signal follows real POST-open evidence. Cleanup signal injected through
  original observer stage; neither discovery nor53ee cleanup is wrapped/mocked.
  ReceiptIO fixture preserves actual retirement while Plan stays blocked.
- All10 live/frozen sources,24 old pins/reviewdiff verified; V3all8live/frozen
  unchanged; accepted guard/payload byte-exact. Eight Python ASTs/three methods
  parsed only, unchanged non-main runner/validator functions AST-identical;
  harmless namespace guard identical to accepted native V2. No imports/tests,
  process/container/mmap/provider/model/GPU/build/runtime or payload hashing.
- ROOT actual mock/harmless tests and four-owner collector/independent validation
  remain mandatory. No actual GPU mapping/count or memory admission proof here.

````

## Source: .agents/runs/serving-and-speculation-kernel-count-audit-20261020/AUDIT.md

Original UTF-8 bytes: 10180; SHA256 `0251543feb130460e66bc86a687b5fadcdce5ca427ba775d2655011fd72693f6`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Forward-dispatch accounting audit

2026-10-20; `/root/baseline_review`; read-only source/evidence review.
Reviewed MAIN `9815504a6975ce766766fc8089729be8121df3ac` plus current
uncommitted B7/token-observation source. Parent owns runtime qualification.

## Finding

No inflation or double counting found in the reviewed successful paths.
Original .8B unequal-arrival HTTP health total **204,414** is exactly
reconciled against explicit host-side kernel launch sites. It is neither
workgroup/element count nor an estimate of projection FLOPs.

This is a source-accounted completed-forward receipt metric, not an independent
CUPTI/Nsight/device-wide kernel trace. No tests or runtime launches performed
by this audit; existing root physical evidence was inspected.

## Receipt and accumulator boundaries

- `resident_program.rs::launch_impl`: two physical launches for status reset
  and original/device token installation, then the actual node-stage counts,
  output certificates and private proposed-state certificates. A launch with
  many workgroups contributes one; physical stage loops contribute one per
  actual launch. `grid(elements)` computes launch dimensions only.
- `device_activation.rs::launch_prepare`: seven preparation launches for
  unsigned-scale Q4_0/Q4_1/Q8_0; five for signed-scale Q6_K. Source projection
  then adds two status merges and one arithmetic projection kernel. Its
  separate finite output certificate adds another physical launch.
- `resident_greedy.rs::launch`: original vocabulary 248,320 uses four
  launches: initial selection, partial reductions 970→4→1, finish.
- `resident_program/batch.rs::launch_impl`: three initial launches
  (status/tokens/positions), actual state gather/scatter, actual node stages,
  output certificates and each row's proposed-state certificates.
  `copy_dense`/`certify` add one per actual <=1,000,000-word chunk, never one
  per copied word. Reviewed .8B B2 shapes need no extra split.
- `resident_batch_greedy.rs::launch`: four actual shared arithmetic sampler
  launches plus one physical private-packet copy per row. B2 total is six;
  native sampler subset is four.
- `resident_inference.rs::execute`: program and greedy dispatch counts are
  combined once. Readback receipt is constructed after physical completion,
  status/readback validation and successful original queue-slot finish.
- `ResidentSession::prepare_chunk`: sums all per-token forward receipts into
  the final chunk readback; it does not multiply the final-token count by the
  chunk size. State-only/logit step counts are summed independently.
- `resident_inference/fused.rs`: program + sampler counts are combined once,
  then completion/status/packet validation precedes the prepared cohort.
  Shared physical count is stored in row zero's readback; other rows contain
  zero. `PreparedResidentCohort::dispatches()` contains the shared count once.
- Serving `WaveStats::single_completion` adds B1 readback dispatches once
  after the public chunk/feedback API returned successfully. One B1 API call
  can contain many token forwards but only one final root publication.
- `gpu_engine/cohort.rs::record_completion` adds the cohort count once after
  Ready, before cancellation/commit scanning. Per-row installation does not
  invoke `single_completion`, so native rows are not charged twice.
- `serve/mod.rs` maps `completed_forward_kernel_dispatches` directly from
  this accumulator. It does not combine projection/sampler subsets into it
  again.

## Exact original .8B reconciliation

Physical evidence:
`.agents/runs/serving-and-speculation-parity-root-2026-10-19/b7-physical-http-final.log`
line 6, `P12_UNEQUAL_ARRIVAL_HTTP_PASS source=qwen35-0.8b`:

| Health field | Value |
| --- | ---: |
| completed_forward_kernel_dispatches | 204,414 |
| completed_model_token_steps | 42 |
| gpu_forward_calls_completed | 12 |
| gpu_fused_batches / gpu_fused_rows_completed | 5 / 10 |
| native_projection_dispatches_completed | 935 |
| native_sampler_dispatches_completed | 20 |
| committed_gpu_roots | 17 |
| accepted_output_tokens | 16 |
| active_sessions / queued_requests | 0 / 0 |
| failed / cancelled / discarded Ready cohorts | 0 / 0 / 0 |

`serve_gpu.rs::qualify_unequal_arrival_metrics` requires two actual original
CPU-oracle sequences of eight non-EOS outputs each, unequal prompt lengths,
and B arrival only after receiving A's first actual ID. Each generated sequence
needs seven feedback forwards; therefore 42 model-token rows comprise 28
prompt-token rows + 14 feedback rows. Two prompt tails and 14 feedback rows
need greedy output: 16 greedy rows, including 10 native B2 rows. The remaining
26 prompt rows are state-only B1 prefill.

Header-only inspection of the accepted `target/models/qwen35-0.8b/model.gguf`
directory/metadata establishes 24 layers: 18 GDN and six full attention.
There are 186 Q4_0 non-head projection operations plus the tied Q8_0 vocabulary
projection: 187 arithmetic projection kernels in a full greedy graph. No
weight payload/model execution was needed for this inspection.

Build artifact evidence `b7-final-serve-build.jsonl` records model features
`cpu,text`, GPU `cuda`, and CLI `cache,cpu,cuda,default,model-qwen35,serve,text`.
`diagnostic-gdn-trace` is absent; no trace-only extra GDN work is included.

For accepted source geometry and valid prefixes <=256, B1 state-only counts:

| Stage family, including listed output certificates | Launches |
| --- | ---: |
| 186 projection operations, preparation/status/certification | 2,046 |
| 96 RMS operations, four stages + certificate | 480 |
| SiLU/gated SiLU stages + certificates | 882 |
| 18 GDN conv operations + certificates | 90 |
| 18 GDN coefficient operations + three certificates each | 270 |
| 18 GDN recurrent operations + certificate each | 486 |
| Views/contiguous copies and certification | 96 |
| Residual adds + certificates | 96 |
| CheckFinite | 24 |
| RoPE stages + certificates | 276 |
| KV append | 6 |
| QK stages + certificates | 216 |
| Softmax stages + certificate | 234 |
| Value stages + certificate | 192 |
| Sigmoid gate stages + certificate | 42 |
| Initial status/token installation | 2 |
| Embedding gather/status + certificate | 3 |
| Candidate-state buffer certificates | 48 |
| **B1 state-only total** | **5,489** |

Full attention value operations use the actual plan padding **256**, so even
these short frontiers execute eight 32-token product blocks. Count per value
operation: two packing launches + 8×3 accumulation launches + five reduction
launches + one output certificate =32. Softmax is one operation/invocation
with 38 internal physical launches plus its output certificate, not one
physical dispatch.

B1 greedy adds 21: final RMS/certification five + vocabulary projection and
certification 11 + extra final-logits certificate one + sampler four.
**B1 greedy total: 5,510.**

Native B2 greedy adds 218 over B1 greedy: extra position setup one +
42 domains × two directions × two rows =168 physical state-transport
launches + 48 additional per-row candidate buffer certificates − one absent
B1-only duplicate final-logits certificate + two private packet copies.
**Native B2 greedy total: 5,728.**

```text
26 × 5,489 + 6 × 5,510 + 5 × 5,728 = 204,414
             6 B1 API calls + 1 state-only chunk + 5 native calls = 12 calls
             6 + 1 + 5×2 = 17 published roots
             5×187 = 935 final native projection arithmetic launches
             5×4   = 20 shared native sampler arithmetic launches
```

The large total is explained principally by source-exact staged projection
preparation/status/certification, staged SiLU/RMS/GDN math, attention stages
and state checks. It does not establish throughput parity with another engine.

## Subset labels

`native_projection_dispatches_completed` is a conservative physical subset:
**final native projection arithmetic kernel launches only**. Each admitted
`Project` node launches exactly one such kernel. For this source, nine other
physical launches precede that final arithmetic kernel, and output
certification follows it. Those launches appear in the total but not this
subset. The subset equals operator invocation count only because the current
closed projection path has one final arithmetic kernel per operator.

`native_sampler_dispatches_completed` counts the four shared native arithmetic
stages, excluding per-row private packet copies. Both subsets are already
contained in `completed_forward_kernel_dispatches`; do not add them again.

## Limits and benchmark use

- Cold source upload/preparation, state allocation/initialization, batch
  workspace admission/eviction, allocator `memory_cleanup`, readback DMA,
  completion events, and unrelated work are outside these forward receipts.
  This metric is not a count of every GPU activity in the process or device.
- A Ready native cohort later discarded for cancellation is deliberately
  included: the physical kernels already completed. Its root publications and
  output goodput remain separate counters. Logs contain such independent
  cancellation cases; the reconciled unequal-arrival sample has none.
- Error/abort/unknown-completion paths which return no successful readback or
  prepared cohort supply no completed receipt. Work submitted before a status,
  completion or commit failure can therefore be absent. In particular, B1
  serving charges only once its public committing API succeeds; native serving
  charges prepared Ready work before commit. This is not an all-submission
  counter for failed/adversarial workloads.
- Health loads separate atomics; it is not one atomic multi-field snapshot.
  Use quiescent before/after deltas and record warmup/process boundaries for
  benchmark accounting. The reviewed sample was quiescent with no failures.
- `completed_gpu_quanta` includes terminal presentation; `gpu_forward_calls`
  counts API calls/chunks; `completed_model_token_steps` counts token rows.
  They are distinct units. None is a substitute for committed output tokens.
- This audit did not inspect a matched vLLM hardware trace or run a benchmark;
  it creates no numerical, performance or vLLM-equivalence acceptance claim.

````

## Source: .agents/runs/serving-and-speculation-kernel-count-audit-20261020/TODO.md

Original UTF-8 bytes: 840; SHA256 `84ac1329c40babeacb739875d572bc09cd2bd2abb5caad518822dbfd6b969bc0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12 forward-dispatch counter audit

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Owner: `/root/baseline_review`
Scope: read-only source/evidence audit; own ignored notes only.

- [x] Trace B1 and native batch receipts to HTTP health.
- [x] Check actual launch accounting, shared-cohort ownership, chunk sums.
- [x] Reconcile original .8B unequal-arrival total exactly: 204,414.
- [x] Distinguish final projection/sampler subsets from all related kernels.
- [x] Record initialization, cancellation, error and sampling boundaries.
- [x] Report findings to root; no counter defect found in reviewed paths.
- [ ] Optional external profiler comparison remains a separate measurement;
      this audit did not run CUDA, models, Cargo, servers or profilers.

Notes: [AUDIT.md](AUDIT.md). No source changes or new qualification claim.

````

## Source: .agents/runs/serving-and-speculation-loaded-sass-peer-review-20261020/REVIEW.md

Original UTF-8 bytes: 4075; SHA256 `84455fff0125b6bb49e988fa27e72c4c6df2278e6ad54d4f5a6e1fc0d5d726c3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Source-only checkpoint: loaded-SASS ABI

Author: baseline_review. Parent: serving-and-speculation P12.
Paused at ROOT's user-directed checkpoint; no new scope or runtime started.

## Completed primary-ABI review

Reviewed source freeze
`serving-and-speculation-ordered-silu-loaded-sass-design-20261020/primary-abi-v2/freeze-manifest.json`,
existing manifest identity
`b4b7f125344fdb1c1425adb3c9a965573a29c34e3a6bd1c3368cdf1ba83a37f6`.
No file hashes recomputed by this review.

- Read complete exact NVIDIA12.9.79 `sass_source_map.cu`, CLOSURE, acquisition
  script/receipt and source-directory metadata descriptions. Original source
  copyright and full component license are retained. Windows source transport
  is not a Linux runtime/ABI proof, whole archive hash or binary acquisition.
- Actual sample lines117–118 choose ResourceData for resource cbdata;
  lines92–94 pass resourceDescriptor;64 casts that descriptor to
  ModuleResourceData;72–73 access pCubin/cubinSize within the callback.
  No direct-cbdata/dual-layout speculation is needed. This optional sample branch
  is disabled by default; source demonstrates intended ABI, not executed behavior.
- Read pinned Linux `cupti_callbacks.h` ResourceData/ModuleResourceData and
  generic callback/subscribe/unsubscribe definitions. The wrapper and member
  types agree with that source. Actual LP64 sizes/offsets still need ROOT compile
  assertions against those exact Linux headers; no layout is accepted from prose.
- Read generated `cuModuleLoadData_params`, `cuModuleGetFunction_params`,
  `cuModuleUnload_params`, `cuLaunchKernel_params` and cbids19/23/22/307.
  Successful API_EXIT is required before reading returned module/function
  handles. Callback pointers and pCubin must be copied while valid; launch
  kernelParams/extra are outside the selected observation scope.
- Read original CubeCL `load_ptx`/`execute_task` and pinned cudarc0.19.10 wrappers.
  They use original cuModuleLoadData, cuModuleGetFunction, cuLaunchKernel and
  preserve the submitted NUL-terminated PTX Vec through the load. No alternate
  driver entrypoint or model/executor is established.
- Read original DESIGN/ROOT_GATES/API facts. Primary-v2 supersedes only the
  original advisory's unresolved descriptor ambiguity; original files remain
  immutable. ActivityModule is explicitly not emitted by the Activity API.
  Callback-only extraction must stay separate from instrumented activity runs.

Verdict: **no source blocker in the primary wrapper→descriptor ABI closure**.
This review does not establish compilation, actual callback delivery, lossless
capture, module/function/launch association, copied cubin, loaded SASS, tool
identity/disassembly, numerical transparency, original Ready, or performance.

## Unfinished prototype review

`prototype-v3/collector.cpp` was reported WRITTEN BUT UNFROZEN. I did not read
moving prototype bytes or execute/import any prepared source. Runner, validator,
fixed-storage fixtures and exact prototype freeze were not delivered before pause.
No prototype source verdict is given.

Resume only from the author's exact frozen prototype, then check:

- Actual unique same-thread nested LoadData→resource module association; no
  equating moduleId with CUmodule and no forcing lazy-load policy to manufacture it.
- Context/generation/unload reuse, successful GetFunction and original launch
  metadata, bounded PTX NUL/name/cubin copies and explicit overflow/loss refusals.
- Preallocated storage and callback errno/no allocation/no I/O/no CUDA calls;
  callback state remains live through safely bounded process retirement.
- Original qualifier's all-output/status/read-only/canary/lease/final0 and Ready
  proof, plus plain/observed transparency and nonce/provider/FD ownership.
- No event/launch enqueue receipt substitutes for original GPU completion;
  unknown completion retains original quarantine and owners.

No tests/imports/compilers/providers/models/operators/containers/GPU, binary/DSO/
payload scans or hashes, MAIN/shared-plan edits or commits by this reviewer.

````

## Source: .agents/runs/serving-and-speculation-loaded-sass-peer-review-20261020/TODO.md

Original UTF-8 bytes: 732; SHA256 `8f5d2c861c18e7b729e6aeae4c18bf0279a7262425da9e145747ab959a8730cd`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: serving-and-speculation. Owned: ignored independent source-review notes.

- [x] Read exact NVIDIA12.9.79 primary sample +source acquisition/license scope.
- [x] Check wrapper/descriptor against pinned Linux callback/generated ABI headers.
- [x] Trace original CubeCL/cudarc load/function/launch source; source closure sound.
- [x] Report primary ABI no-source-blocker with compile/runtime/Ready limits.
- [x] Preserve user-directed pause checkpoint; prototype UNFROZEN/UNREVIEWED.
- [ ] On resume, obtain exact callback-only prototype freeze and review complete
  collector/launcher/validator/fixtures before ROOT compiles or launches it.

No execution, tests, source imports or payload/binary hashes. MAIN/plans unchanged.

````

## Source: .agents/runs/serving-and-speculation-loaded-sass-peer-review-20261020/frozen-checkpoint/REVIEW.md

Original UTF-8 bytes: 4075; SHA256 `84455fff0125b6bb49e988fa27e72c4c6df2278e6ad54d4f5a6e1fc0d5d726c3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Source-only checkpoint: loaded-SASS ABI

Author: baseline_review. Parent: serving-and-speculation P12.
Paused at ROOT's user-directed checkpoint; no new scope or runtime started.

## Completed primary-ABI review

Reviewed source freeze
`serving-and-speculation-ordered-silu-loaded-sass-design-20261020/primary-abi-v2/freeze-manifest.json`,
existing manifest identity
`b4b7f125344fdb1c1425adb3c9a965573a29c34e3a6bd1c3368cdf1ba83a37f6`.
No file hashes recomputed by this review.

- Read complete exact NVIDIA12.9.79 `sass_source_map.cu`, CLOSURE, acquisition
  script/receipt and source-directory metadata descriptions. Original source
  copyright and full component license are retained. Windows source transport
  is not a Linux runtime/ABI proof, whole archive hash or binary acquisition.
- Actual sample lines117–118 choose ResourceData for resource cbdata;
  lines92–94 pass resourceDescriptor;64 casts that descriptor to
  ModuleResourceData;72–73 access pCubin/cubinSize within the callback.
  No direct-cbdata/dual-layout speculation is needed. This optional sample branch
  is disabled by default; source demonstrates intended ABI, not executed behavior.
- Read pinned Linux `cupti_callbacks.h` ResourceData/ModuleResourceData and
  generic callback/subscribe/unsubscribe definitions. The wrapper and member
  types agree with that source. Actual LP64 sizes/offsets still need ROOT compile
  assertions against those exact Linux headers; no layout is accepted from prose.
- Read generated `cuModuleLoadData_params`, `cuModuleGetFunction_params`,
  `cuModuleUnload_params`, `cuLaunchKernel_params` and cbids19/23/22/307.
  Successful API_EXIT is required before reading returned module/function
  handles. Callback pointers and pCubin must be copied while valid; launch
  kernelParams/extra are outside the selected observation scope.
- Read original CubeCL `load_ptx`/`execute_task` and pinned cudarc0.19.10 wrappers.
  They use original cuModuleLoadData, cuModuleGetFunction, cuLaunchKernel and
  preserve the submitted NUL-terminated PTX Vec through the load. No alternate
  driver entrypoint or model/executor is established.
- Read original DESIGN/ROOT_GATES/API facts. Primary-v2 supersedes only the
  original advisory's unresolved descriptor ambiguity; original files remain
  immutable. ActivityModule is explicitly not emitted by the Activity API.
  Callback-only extraction must stay separate from instrumented activity runs.

Verdict: **no source blocker in the primary wrapper→descriptor ABI closure**.
This review does not establish compilation, actual callback delivery, lossless
capture, module/function/launch association, copied cubin, loaded SASS, tool
identity/disassembly, numerical transparency, original Ready, or performance.

## Unfinished prototype review

`prototype-v3/collector.cpp` was reported WRITTEN BUT UNFROZEN. I did not read
moving prototype bytes or execute/import any prepared source. Runner, validator,
fixed-storage fixtures and exact prototype freeze were not delivered before pause.
No prototype source verdict is given.

Resume only from the author's exact frozen prototype, then check:

- Actual unique same-thread nested LoadData→resource module association; no
  equating moduleId with CUmodule and no forcing lazy-load policy to manufacture it.
- Context/generation/unload reuse, successful GetFunction and original launch
  metadata, bounded PTX NUL/name/cubin copies and explicit overflow/loss refusals.
- Preallocated storage and callback errno/no allocation/no I/O/no CUDA calls;
  callback state remains live through safely bounded process retirement.
- Original qualifier's all-output/status/read-only/canary/lease/final0 and Ready
  proof, plus plain/observed transparency and nonce/provider/FD ownership.
- No event/launch enqueue receipt substitutes for original GPU completion;
  unknown completion retains original quarantine and owners.

No tests/imports/compilers/providers/models/operators/containers/GPU, binary/DSO/
payload scans or hashes, MAIN/shared-plan edits or commits by this reviewer.

````

## Source: .agents/runs/serving-and-speculation-loaded-sass-peer-review-20261020/frozen-checkpoint/TODO.md

Original UTF-8 bytes: 732; SHA256 `8f5d2c861c18e7b729e6aeae4c18bf0279a7262425da9e145747ab959a8730cd`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: serving-and-speculation. Owned: ignored independent source-review notes.

- [x] Read exact NVIDIA12.9.79 primary sample +source acquisition/license scope.
- [x] Check wrapper/descriptor against pinned Linux callback/generated ABI headers.
- [x] Trace original CubeCL/cudarc load/function/launch source; source closure sound.
- [x] Report primary ABI no-source-blocker with compile/runtime/Ready limits.
- [x] Preserve user-directed pause checkpoint; prototype UNFROZEN/UNREVIEWED.
- [ ] On resume, obtain exact callback-only prototype freeze and review complete
  collector/launcher/validator/fixtures before ROOT compiles or launches it.

No execution, tests, source imports or payload/binary hashes. MAIN/plans unchanged.

````

## Source: .agents/runs/serving-and-speculation-native-ab-peer-review-batch-20261020/REVIEW.md

Original UTF-8 bytes: 5192; SHA256 `72ef6ce82747d1f0b78267f2e0b7e640f5d1a026990f748c064d47440889cbf7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native before/after retained-result review

Author: batch_design_review. Parent: serving-and-speculation P12.

Verdict: no source/retained-evidence blocker for the declared native-only raw
accounting scope. ROOT alone executes/accepts/reconciles the phase.

Reviewed actual `serving-and-speculation-native-before-after-20261020/root-native-v2-1`,
its plan, V2 runner/validator, common accounting, original methodology and additive
namespace contract. Independently parsed all32 receipts/measurements/raw results,
all32 complete98-byte logs and actual bootstrap/resource/owner metadata. Own
sample-review.json retains small-file bindings and per-sample values. No proposed
validator import/execution or model/binary/DSO/raw-trace read performed here.

## Closure

- Exactly32 fresh native processes, ABBA then BAAB per model/C1 orC8 case;
 144 measured offers completed/HTTP200,1,152 output IDs,32 separate ordinary
 canonical max8 warmups. Model-specific complete IDs/text hashes/usage8/length
 agree across both executables/all rows/bands/warmups. Canonical prompt47 IDs;
 packet producer frontiers47..54. No EOS or length normalization.
- All144 responses deliver8 individual-ID groups and7 real client ITLs:
 1,008 intervals, no coalescing substitution. Provider commit observations retain
 their independent host-clock scope; mean ITL and SSE intervals are not used as
 actual token intervals. Generic provider-trace absence does not invalidate the
 separate response-bound native commit records.
- Distinct32 nonce/PID owners: no primary/cleanup/log error; actual group absent,
 leader reaped, every nonzero signal before reap, container absent, pipe EOF and
 untruncated log hash/length consistent. Expected owned retirement exit137 is
 recorded after successful measurement; it is not a numerical/request failure.
- All starts read back affinity0..3, CPU quota400000/100000, RAM12GiB/swap0.
 Pair launch argv differ only native binary mount and expected binary SHA;
 protocols differ only binary/path/SHA/source identity. Private rootless map,
 user-owned private net/loopback and same GPU UUID retained. Point monitor
 failures absent; its GPU PIDs belong to authenticated container listings.
- Before/after active/queued0. No failed/refused/discarded/cancelled GPU work.
 C1 has54 steps/roots and no native batching. C8 has432 steps/roots and exactly
 54 genuineB8 cohorts:46 StateOnly +8 greedy,368 StateOnly rows +64 greedy rows,
 32 sampler launches. Weighted row/phase/band and accepted-token accounting
 close; warmup excluded. Both local workload labels complete every offered row.

## Descriptive results

Medians over four fresh-process samples per native label, not pooled percentile
estimates or confidence-certified wins. ITL is median of each sample's client
token-delivery p95. Rates count usage tokens / all-offers-through-last-completion.

| Model/C | Rate baseline→candidate (tokens/s) | Offered TTFT p50 baseline→candidate (s) | ITL p95 baseline→candidate (s) | Completed source launches baseline→candidate |
| --- | --- | --- | --- | --- |
| .8B/C1 | 1.69640→1.69750 | 4.08717→4.09925 | .10324→.11145 | 236270→236270 |
| .8B/C8 | 2.25060→2.28945 | 25.77409→25.32761 | .40800→.40623 | 292116→239952 |
| 2B/C1 | 1.36985→1.37575 | 4.74065→4.71593 | .16056→.16569 | 228494→228494 |
| 2B/C8 | 1.62680→1.65430 | 32.66844→32.05865 | .97987→.97747 | 284340→232176 |

C8 source-launch reduction is52,164/sample (17.86%/.8B,18.35%/2B), while
descriptive median rate changes are only about1.73%/1.69%. C1 rate ranges overlap
and dispatches are unchanged. An18% kernel or latency speedup is unsupported.

Sampled total-device memory is equal within each model/band:1059/3045MiB for
.8B C1/C8 and1857/3509MiB for2B. Samples cover startup/warmup/measurement;
they are neither exact GPU peaks nor measured process/host allocation totals.
Thermal observations rise61..85°C over the phase and clocks vary; ordering limits
some drift but supplies no clock-lock or thermal-normalization proof.

## Limits retained

- This compares two Zerogravity executables only. No stock vLLM sample/slot,
 matched-vLLM performance result or vLLM-equivalent throughput follows.
- All launch counters are completed source launch sites, not enumerated CUDA
 activity or physical hardware kernel counts. Separate CUPTI runs are separate
 workload/method evidence and cannot replace these timing denominators.
- A single ordinary warmup does not fully warmB8 shape/graph/cache paths.
 Eight simultaneous offers are a finite burst, not sustained capacity. Same
 prompt/max8/local labels do not qualify tenant isolation or broad fairness.
- Client backlog remains explicit (largest29.47ms); C8 within-wave completion
 spread stays below2.45ms. All C1 responses meet configured client wire SLO;
 all128 C8 responses miss10s TTFT despite meeting2s ITL. No source-correct SLO
 goodput or production target SLO claim is admitted.
- No full-StateRoot/full-logit comparison is performed by this timing harness.
 Earlier physical acceptance gates are explicit antecedents; wire output equality
 does not independently establish them or enable full original-HF/graphs.

````

## Source: .agents/runs/serving-and-speculation-native-ab-peer-review-batch-20261020/TODO.md

Original UTF-8 bytes: 704; SHA256 `2f535a690fe5162bb99fd024a2f3a25a328c0a56882d0b12650f3c94eccdcb89`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native before/after peer review

Parent: serving-and-speculation P12. Owner: batch_design_review.

- [x] Read native V2 plan/runner/validator/common accounting and retained actual validation.
- [x] Independently inspect all32 receipt/raw-measurement/log metadata; compare output, lifecycle, resource and counter closure.
- [x] Record descriptive metrics and qualification limits in REVIEW.md/sample-review.json.
- [ ] ROOT decides phase acceptance/owning-plan reconciliation.

Scope: retained JSON/log/source only; stdlib metadata analysis. No proposed
validator/test import or execution, runtime/provider/model/GPU activity, build,
model/DSO/binary/raw-trace reads, MAIN/shared-plan change or commit.

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/README.md

Original UTF-8 bytes: 6481; SHA256 `64fe44a999122b592b0b848519671477788053637f1dfea33a0d6e5147ed6ced`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native baseline/candidate latency preparation

Author: batch_design_review. Parent: serving-and-speculation P12.
SOURCE ONLY; no author imports/tests, child/provider/model/GPU execution, builds,
binary/model/DSO payload scans or timing results.

- Explicit labels `native_baseline` (4a745527...) and `native_candidate`
  (87e1d727...). Both are Zerogravity. Neither occupies a vLLM comparison slot;
  no `serve-compare` labels, legacy ratios or `measure_same_engine` counters.
- Direct unchanged canonical `tools/serve-bench.py`. The immutable V8 native
  launch/bootstrap branch remains `provider=zerogravity`, its genuine ENGINE
  identity, for both new result labels. Stock/vLLM is never launched. Paired
  launch command normalization requires identical flags/mounts/environment
  except the immutable native executable and its expected hash.
- Both sources, warmed_process, C1 canonical solo and C8 canonical eight-offer
  wave, max8. Exact existing47 prompt IDs/common greedy/EOS policy. Each of four
  model/band cases runs ABBA then BAAB, eight fresh owned processes:32 measured
  samples,144 measured offers (16solo+128wave),72 offers/native label, plus32
  independent unmeasured warmup requests. ONE normal canonical prompt/max8
  warmup per process; no extra shape/full-wave warmup.
- Same exact four CPUs, CPU4 quota, RAM12GiB/no extra swap, original UUID/context,
  active8/native4GiB host/provider+5GiB device, read-only inputs and private user/
  network namespace; same V8 cgroup readback/point device observation. Native
  graph remains disabled in the accepted source. This is neither cold-process
  nor fully warmed shape/graph nor persistent-server steady-state measurement.

Gates and ownership:

- `bindings.json` pins the old actual accepted native metadata, candidate456
  source freeze, strict/release build, operator10/model46/full isolated+both
  approved routes and exact future worker14/HTTP12 receipt location. ROOT must
  supply its completed receipt SHA. Missing/RUNNING/partial/failed/unclean or
  unexpected-count receipt refuses preparation AND execution AND validation.
  ROOT should inspect all gate evidence before supplying the SHA; marker alone
  is insufficient. Later source drift or another runner version needs a fresh
  derivative, never a silently widened acceptance predicate.
- Original53ee nonce/name/CID container + reserved-leader WNOWAIT whole-group
  TERM/KILL-before-any-reap cleanup; original bounded log reader/EOF and V8
  ownership-authenticated point-memory monitor. Exact00d1 TERM+INT record-only
  guard spans provider lifetime, with explicit checks outside Popen handoff and
  retirement. No asynchronous exception crosses an unassigned child handle.
- Canonical HTTP retains its existing180s deadline/socket-timeout controls. A
  recorded cancel is checked after canonical return; this is no new hard
  asynchronous interrupt bound. Owner remains retained. Cleanup ignores recorded
  signals until all original stages close. No subsequent sample after any
  numerical, counter, transport, owner, log, memory-observer or receipt failure.
- Source/weights/binary binding scans run only in ROOT preparation/before+after
  the whole run, outside timed samples. Logs8MiB/process with truncation refused;
  whole output512MiB/1024files, every metadata write and final validation checks
  the tree INCLUDING its newly written receipt. These are artifact acceptance
  bounds, not a parent process physical-memory admission claim.

Metrics and acceptance:

- Every sample retains full raw benchmark BEFORE numerical/accounting checks.
  Exact full IDs/text SHA/usage/finish match all rows/versions/bands, including
  each ordinary warmup. No EOS/usage normalization or minimum-token workaround.
- Quiescent health before/after excludes warmup and cleanup. Every required
  counter is present, typed, monotonic; error/refusal/discard/cancel deltas0;
  weighted2/4/8 bands close actual fused rows, phases and sampler4/cohort. These
  are source launchsites, never CUDA/CUPTI enumerated kernel counts.
- C8 is client concurrency, not a promised nativeB8 cohort. Preserve actual row
  band deltas and packet cohort widths; no padding or forced cohort schedule.
  Actual packet frontiers47+index bind independent output progression. State/
  prefix cohorts have separate counters and are not reconstructed from SSE.
- Text TTFT and actual single-ID TTFT remain distinct. Keep offered/client-
  backlog TTFT and service TTFT. Individual-token ITL requires complete actual
  single-ID delivery; coalesced/unresolved rows are counted, never replaced by
  SSE chunk or meanITL proxies. Zero/one output ID has no ITL interval. Native
  commit observations remain a separate host clock, not device timestamps.
- Report offered/completed/usage throughput and all raw paired latencies plus
  local-label fairness, sampled memory, actual source counters. Four samples per
  label/band yield descriptive medians, no confidence-certified performance win.
  Eight finite simultaneous offers measure a burst, not sustained capacity.
  No source-correct SLO goodput, physical committed-token, fullHF, production
  graph or vLLM performance/parity claim follows.

ROOT sequence (not executed):

1. Complete/review current14worker/12HTTP source-frozen gate. Run the four pure
   `test_contracts.py` and four MOCK-only `test_owner_control.py` fixtures, then
   actual harmless-owner TERM/INT-before-Popen/during-readiness/during-
   request/during-cleanup failure-injection fixtures. Prior shadow/root evidence
   is not a test of this new adapter. Review all new source and bindings.
2. Set `ZG_BENCH_HOST_ROOT` to the original repository. Run
   `prepare_native_ab.py --hardware-protocol ABS_ACCEPTED_V8_PROTOCOL
   --hardware-protocol-sha256 SHA --accepted-serving-sha256 SHA
   --new-output ABS_THIS_RUN/root-plan-1`. That performs ROOT-only source scans,
   exact launch normalization and completed gate checks; no engine starts.
3. Under the original native-gpu user/network isolation, invoke Python with the
   original reviewed private-loopback helper (which execs Python), then
   `run_native_ab.py --run --plan ABS/root-plan-1/plan.json
   --plan-sha256 SHA --new-output ABS_THIS_RUN/root-native-1`.
4. `validate_native_ab.py --output ABS_THIS_RUN/root-native-1`. Independent ROOT
   review of full raw results/actual owner/point-memory/ITL coverage is required.
   Profiling/hardware counts and unprofiled timing remain separate runs.

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/README_V2.md

Original UTF-8 bytes: 5392; SHA256 `e5c6ea2b024fdc9334ff48f3766791dea4c85c003607d3e870f824c34afa1257`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native baseline/candidate V2 namespace correction

Author: batch_design_review. Parent: serving-and-speculation P12.
SOURCE ONLY; no author imports/tests, process/container/model/GPU execution,
builds or binary/model/DSO payload reads/hashes.

- Original12-file freeze6e5fd796 and original8 fixtures remain immutable.
  V2 adds exact full rootless Podman UID mapping/bootstrap and current-user-owned
  private network/loopback checks from accepted `root-private-loopback-exec-2.py`.
  Its full mapping preserves the custom store's overlay ownership. The original
  single-UID `native-gpu.py` invocation is unsuitable for this Podman driver.
- `prepare_native_ab_v2.py` changes only its source-binding import.
  `run_native_ab_v2.py` changes that import and namespace checks only.
  `validate_native_ab_v2.py` changes its binding/runner imports only.
  `native_ab_v2.py` checks both immutable V1 and additive V2 live/frozen bytes,
  plus exact accepted wrapper/support bytes. Original workload, launcher, caps,
  owner, cancellation recorder, numerical/accounting validation are unchanged.
- Candidate gate now physically exists: ROOT worker14/HTTP12 receipt18625960,
  source456, server87e1d727. Preparation still requires its explicit full SHA;
  no pending/partial gate is accepted. This derivative contains no timing result.
- Both native labels/models, warmed_process C1 and C8, ABBA+BAAB remain32 fresh
  measured processes /144 offered requests /32 ordinary canonical warmups.
  The canonical first prompt has47 IDs, max8. Neither label is a vLLM slot.
  Original README metrics/limitations still apply.

Prepared additional fixtures, ROOT executes only after source review:

- `test_namespace_v2.py`: six pure/mocked tests; full mapping accepted, known
  bad single UID/missing container bootstrap/foreign network/non-loopback/down
  loopback refused. These are contract tests, not actual namespace evidence.
- `test_owner_process_v2.py`: nine actual harmless-owner cases: TERM+INT before
  Popen, during readiness, during a real open HTTP POST, during original owner
  cleanup; one postspawn receipt-I/O failure. Only stdlib `harmless_http_v2.py`
  runs in the already-pinned local image, no pulls, devices, model/source/native
  binary mounts. Container256MiB/CPU1/pids32; no provider/native math.
- Target `/v1/models` ALWAYS503; POST emits no ID/token/text/usage. Fixture
  readiness is a separately labelled control health endpoint. Request signal
  requires actual POST-open marker and authenticated live nonce/CID/foreground;
  no delay-based inference. Original unchanged `serve-bench.stream_one` is used
  only as bounded transport (fixture timeout2s/max1); this is NOT a canonical
  benchmark sample. Its failure is mandatory.
- Original NativeOwner and53ee WNOWAIT/container/group/EOF cleanup are exercised.
  Signal goes only to the current supervisor, whose original recorder cannot
  raise across handoff/cleanup. Actual container cleanup signals only original
  authenticated owners. Stop/join precedes handler restoration; no next fixture
  after any unexpected failure. The target ignores TERM to exercise owner kill.
- GPU observation is an explicit `control_only` stub. Empty point samples and
  `ready!=true`, HARMLESS labels/non-native binary prevent native acceptance.
  No GPU quietness, engine/bootstrap/cgroup qualification follows these tests.
  Receipt-I/O failure cannot skip cleanup; actual closure is retained afterward
  via a restored fixture writer, with expected failures still explicit.

ROOT invocation (not executed here):

```bash
repo=/home/pvl/spaces/abird/src/zerogravity
own="$repo/.agents/runs/serving-and-speculation-native-before-after-20261020"
parent="$repo/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020"
export ZG_BENCH_HOST_ROOT="$repo" PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s "$own" -p 'test_namespace_v2.py'

# Exact accepted custom-store rootless wrapper2 chain, fresh private net/mount.
podman --root /home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers \
  --runroot /run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers \
  unshare unshare --net --mount --propagation private \
  python3 "$parent/root-private-loopback-exec-2.py" \
  python3 "$own/test_owner_process_v2.py" --run-control-only \
  --new-output "$own/root-harmless-v2-1"
```

After all fixtures/review and ROOT's exclusive GPU reservation:

1. Use `prepare_native_ab_v2.py` with unchanged original preparation arguments;
   fresh own plan directory and exact accepted V8 hardware-protocol SHA, plus
   `--accepted-serving-sha256
   186259602e1debf69df208c5f12be0f60687a6fb81c43a699c34f5a55dcd0431`.
   ROOT performs original source/binary/weight binding scans outside samples.
2. Repeat the exact Podman/unshare/wrapper2 chain above, replacing ONLY its final
   Python command with `run_native_ab_v2.py --run --plan ABS_PLAN/plan.json
   --plan-sha256 ACTUAL_SHA --new-output ABS_OWN/root-native-v2-1`.
3. `validate_native_ab_v2.py --output ABS_OWN/root-native-v2-1`; review all raw
   samples/owners/ITL coverage. Failures do not permit another sample/provider.

Own V2 sources only were AST parsed. Real fixtures, preparation, actual timings
and independent full receipt validation remain ROOT gates. No speed, hardware
count, source-correct goodput, full HF or vLLM parity claim.

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/TODO.md

Original UTF-8 bytes: 950; SHA256 `bd978523a1018a28be0a35e6689266a353badecd8a2ad36c2f47a859593af7e1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native before/after preparation

Parent: serving-and-speculation; owner: batch_design_review.
Owned: this new ignored derivative only; V8/old baseline/current456 unchanged.

- [x] Explicit native/native identities, exact existing launch/workload/caps.
- [x] Same canonical benchmark, complete32-sample/144-offer/32-warmup accounting.
- [x] Require completed actual candidate worker14/HTTP12 + prior source gates.
- [x] Prepare original owner cleanup +record-only TERM/INT checkpoints.
- [x] Full exact output/counter/band/ITL coverage and independent validation.
- [ ] ROOT/peer source review; actual pure/ownership failure fixtures.
- [ ] ROOT actual preparation +complete unprofiled native/native run+validation.
- [ ] Independent physical kernel diagnostic and measured performance acceptance.

Source-only AST and small source/metadata hashes by author. No tests, imports,
builds, native/provider/model/GPU execution or payload hashes/results.

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/TODO_V2.md

Original UTF-8 bytes: 855; SHA256 `0adb94256db2830289ca827fcbeae672a17b7e44dbd9f55ef248db2df3d20119`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12 native/native before-after V2

Parent: `.agents/plans/serving-and-speculation/TODO.md`; owner batch_design_review.

- [x] Preserve original6e5fd796 live/frozen12 files and ROOT original8 fixture PASS.
- [x] Full accepted rootless UID/bootstrap/network/loopback guard; V2 bound imports.
- [x] Prepare6 pure namespace and9 real harmless owner fixtures (not executed).
- [x] Bind actual POST-open evidence, original owner/EOF, explicit no-GPU stub.
- [x] Bounded review actual worker14/HTTP12 receipt/logs: no scope mismatch.
- [x] Source-only AST/read/freeze; no author runtime or payload scans.
- [ ] Independent V2 source review; ROOT pure and actual harmless fixture execution.
- [ ] ROOT preparation and actual32-process canonical warmed native/native run.
- [ ] Independent all samples/owners/numerics/accounting validation; no timing claim yet.

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/followup-v2/frozen/README_V2.md

Original UTF-8 bytes: 5392; SHA256 `e5c6ea2b024fdc9334ff48f3766791dea4c85c003607d3e870f824c34afa1257`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native baseline/candidate V2 namespace correction

Author: batch_design_review. Parent: serving-and-speculation P12.
SOURCE ONLY; no author imports/tests, process/container/model/GPU execution,
builds or binary/model/DSO payload reads/hashes.

- Original12-file freeze6e5fd796 and original8 fixtures remain immutable.
  V2 adds exact full rootless Podman UID mapping/bootstrap and current-user-owned
  private network/loopback checks from accepted `root-private-loopback-exec-2.py`.
  Its full mapping preserves the custom store's overlay ownership. The original
  single-UID `native-gpu.py` invocation is unsuitable for this Podman driver.
- `prepare_native_ab_v2.py` changes only its source-binding import.
  `run_native_ab_v2.py` changes that import and namespace checks only.
  `validate_native_ab_v2.py` changes its binding/runner imports only.
  `native_ab_v2.py` checks both immutable V1 and additive V2 live/frozen bytes,
  plus exact accepted wrapper/support bytes. Original workload, launcher, caps,
  owner, cancellation recorder, numerical/accounting validation are unchanged.
- Candidate gate now physically exists: ROOT worker14/HTTP12 receipt18625960,
  source456, server87e1d727. Preparation still requires its explicit full SHA;
  no pending/partial gate is accepted. This derivative contains no timing result.
- Both native labels/models, warmed_process C1 and C8, ABBA+BAAB remain32 fresh
  measured processes /144 offered requests /32 ordinary canonical warmups.
  The canonical first prompt has47 IDs, max8. Neither label is a vLLM slot.
  Original README metrics/limitations still apply.

Prepared additional fixtures, ROOT executes only after source review:

- `test_namespace_v2.py`: six pure/mocked tests; full mapping accepted, known
  bad single UID/missing container bootstrap/foreign network/non-loopback/down
  loopback refused. These are contract tests, not actual namespace evidence.
- `test_owner_process_v2.py`: nine actual harmless-owner cases: TERM+INT before
  Popen, during readiness, during a real open HTTP POST, during original owner
  cleanup; one postspawn receipt-I/O failure. Only stdlib `harmless_http_v2.py`
  runs in the already-pinned local image, no pulls, devices, model/source/native
  binary mounts. Container256MiB/CPU1/pids32; no provider/native math.
- Target `/v1/models` ALWAYS503; POST emits no ID/token/text/usage. Fixture
  readiness is a separately labelled control health endpoint. Request signal
  requires actual POST-open marker and authenticated live nonce/CID/foreground;
  no delay-based inference. Original unchanged `serve-bench.stream_one` is used
  only as bounded transport (fixture timeout2s/max1); this is NOT a canonical
  benchmark sample. Its failure is mandatory.
- Original NativeOwner and53ee WNOWAIT/container/group/EOF cleanup are exercised.
  Signal goes only to the current supervisor, whose original recorder cannot
  raise across handoff/cleanup. Actual container cleanup signals only original
  authenticated owners. Stop/join precedes handler restoration; no next fixture
  after any unexpected failure. The target ignores TERM to exercise owner kill.
- GPU observation is an explicit `control_only` stub. Empty point samples and
  `ready!=true`, HARMLESS labels/non-native binary prevent native acceptance.
  No GPU quietness, engine/bootstrap/cgroup qualification follows these tests.
  Receipt-I/O failure cannot skip cleanup; actual closure is retained afterward
  via a restored fixture writer, with expected failures still explicit.

ROOT invocation (not executed here):

```bash
repo=/home/pvl/spaces/abird/src/zerogravity
own="$repo/.agents/runs/serving-and-speculation-native-before-after-20261020"
parent="$repo/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020"
export ZG_BENCH_HOST_ROOT="$repo" PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s "$own" -p 'test_namespace_v2.py'

# Exact accepted custom-store rootless wrapper2 chain, fresh private net/mount.
podman --root /home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers \
  --runroot /run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers \
  unshare unshare --net --mount --propagation private \
  python3 "$parent/root-private-loopback-exec-2.py" \
  python3 "$own/test_owner_process_v2.py" --run-control-only \
  --new-output "$own/root-harmless-v2-1"
```

After all fixtures/review and ROOT's exclusive GPU reservation:

1. Use `prepare_native_ab_v2.py` with unchanged original preparation arguments;
   fresh own plan directory and exact accepted V8 hardware-protocol SHA, plus
   `--accepted-serving-sha256
   186259602e1debf69df208c5f12be0f60687a6fb81c43a699c34f5a55dcd0431`.
   ROOT performs original source/binary/weight binding scans outside samples.
2. Repeat the exact Podman/unshare/wrapper2 chain above, replacing ONLY its final
   Python command with `run_native_ab_v2.py --run --plan ABS_PLAN/plan.json
   --plan-sha256 ACTUAL_SHA --new-output ABS_OWN/root-native-v2-1`.
3. `validate_native_ab_v2.py --output ABS_OWN/root-native-v2-1`; review all raw
   samples/owners/ITL coverage. Failures do not permit another sample/provider.

Own V2 sources only were AST parsed. Real fixtures, preparation, actual timings
and independent full receipt validation remain ROOT gates. No speed, hardware
count, source-correct goodput, full HF or vLLM parity claim.

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/followup-v2/frozen/TODO_V2.md

Original UTF-8 bytes: 855; SHA256 `0adb94256db2830289ca827fcbeae672a17b7e44dbd9f55ef248db2df3d20119`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12 native/native before-after V2

Parent: `.agents/plans/serving-and-speculation/TODO.md`; owner batch_design_review.

- [x] Preserve original6e5fd796 live/frozen12 files and ROOT original8 fixture PASS.
- [x] Full accepted rootless UID/bootstrap/network/loopback guard; V2 bound imports.
- [x] Prepare6 pure namespace and9 real harmless owner fixtures (not executed).
- [x] Bind actual POST-open evidence, original owner/EOF, explicit no-GPU stub.
- [x] Bounded review actual worker14/HTTP12 receipt/logs: no scope mismatch.
- [x] Source-only AST/read/freeze; no author runtime or payload scans.
- [ ] Independent V2 source review; ROOT pure and actual harmless fixture execution.
- [ ] ROOT preparation and actual32-process canonical warmed native/native run.
- [ ] Independent all samples/owners/numerics/accounting validation; no timing claim yet.

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/followup-v3/README_V3.md

Original UTF-8 bytes: 2519; SHA256 `534b053305c147fcec5d7fcf4005c841be845f4c0dcf665311b40a12ad263bbc`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Fixture-only signal-thread retirement V3

Author: baseline_review. Additive SOURCE_ONLY; no author execution/import/tests.

- Canonical native/native preparation/runner/validator stay exact V2 b0a0371f.
  Original V1, V2 frozen sources, actual nine V2 harmless successes and failed
  benchmark attempts remain untouched. This adds only fixture prerequisites.
- Actual `owned_container()` authentication can issue TWO10s commands (`exists`,
  `inspect`), then fixture HTTP allows2s. Join25s derives22s+3s margin, replacing
  the old invalid15s/one-command assumption.
- Original NativeOwner and53ee container/group/log cleanup are unchanged.
  Original record-only SigtermCancellation is subclassed only for fixture exit:
  originals restore after a retired signal thread. If the25s retirement still
  fails, retain record-only handlers, propagate failure and stop the whole
  fixture run. Never restore a terminating handler while its producer is alive.
- Nine original actual harmless cases remain identical except imports, thread
  association and retirement. They still use no device/model/native payload,
  always503 native readiness and explicit non-GPU control observations.
- Three mocked source-free fixtures cover22s authenticated completion within25,
  exhausted join with no restoration, and original no-thread restoration. Mock
  arithmetic is not actual timing/namespace/owner evidence.
- Canonical benchmark may keep V2 identity; ROOT reviews this V3 fixture receipt
  as an additional prerequisite. No duplicated measurement runner or math.

ROOT commands (author did not execute):

```bash
repo=/home/pvl/spaces/abird/src/zerogravity
own="$repo/.agents/runs/serving-and-speculation-native-before-after-20261020"
parent="$repo/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020"
export ZG_BENCH_HOST_ROOT="$repo" PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s "$own/followup-v3" -p 'test_signal_retirement_v3.py'
podman --root /home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers \
  --runroot /run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers \
  unshare unshare --net --mount --propagation private \
  python3 "$parent/root-private-loopback-exec-2.py" \
  python3 "$own/followup-v3/test_owner_process_v3.py" --run-control-only \
  --new-output "$own/root-harmless-v3-1"
```

Any retirement or receipt failure stops; do not treat a subset as the full gate.
This author performed AST parsing and source diff/byte comparison only.

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/followup-v3/frozen/README_V3.md

Original UTF-8 bytes: 2519; SHA256 `534b053305c147fcec5d7fcf4005c841be845f4c0dcf665311b40a12ad263bbc`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Fixture-only signal-thread retirement V3

Author: baseline_review. Additive SOURCE_ONLY; no author execution/import/tests.

- Canonical native/native preparation/runner/validator stay exact V2 b0a0371f.
  Original V1, V2 frozen sources, actual nine V2 harmless successes and failed
  benchmark attempts remain untouched. This adds only fixture prerequisites.
- Actual `owned_container()` authentication can issue TWO10s commands (`exists`,
  `inspect`), then fixture HTTP allows2s. Join25s derives22s+3s margin, replacing
  the old invalid15s/one-command assumption.
- Original NativeOwner and53ee container/group/log cleanup are unchanged.
  Original record-only SigtermCancellation is subclassed only for fixture exit:
  originals restore after a retired signal thread. If the25s retirement still
  fails, retain record-only handlers, propagate failure and stop the whole
  fixture run. Never restore a terminating handler while its producer is alive.
- Nine original actual harmless cases remain identical except imports, thread
  association and retirement. They still use no device/model/native payload,
  always503 native readiness and explicit non-GPU control observations.
- Three mocked source-free fixtures cover22s authenticated completion within25,
  exhausted join with no restoration, and original no-thread restoration. Mock
  arithmetic is not actual timing/namespace/owner evidence.
- Canonical benchmark may keep V2 identity; ROOT reviews this V3 fixture receipt
  as an additional prerequisite. No duplicated measurement runner or math.

ROOT commands (author did not execute):

```bash
repo=/home/pvl/spaces/abird/src/zerogravity
own="$repo/.agents/runs/serving-and-speculation-native-before-after-20261020"
parent="$repo/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020"
export ZG_BENCH_HOST_ROOT="$repo" PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s "$own/followup-v3" -p 'test_signal_retirement_v3.py'
podman --root /home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers \
  --runroot /run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers \
  unshare unshare --net --mount --propagation private \
  python3 "$parent/root-private-loopback-exec-2.py" \
  python3 "$own/followup-v3/test_owner_process_v3.py" --run-control-only \
  --new-output "$own/root-harmless-v3-1"
```

Any retirement or receipt failure stops; do not treat a subset as the full gate.
This author performed AST parsing and source diff/byte comparison only.

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/frozen/README.md

Original UTF-8 bytes: 6481; SHA256 `64fe44a999122b592b0b848519671477788053637f1dfea33a0d6e5147ed6ced`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native baseline/candidate latency preparation

Author: batch_design_review. Parent: serving-and-speculation P12.
SOURCE ONLY; no author imports/tests, child/provider/model/GPU execution, builds,
binary/model/DSO payload scans or timing results.

- Explicit labels `native_baseline` (4a745527...) and `native_candidate`
  (87e1d727...). Both are Zerogravity. Neither occupies a vLLM comparison slot;
  no `serve-compare` labels, legacy ratios or `measure_same_engine` counters.
- Direct unchanged canonical `tools/serve-bench.py`. The immutable V8 native
  launch/bootstrap branch remains `provider=zerogravity`, its genuine ENGINE
  identity, for both new result labels. Stock/vLLM is never launched. Paired
  launch command normalization requires identical flags/mounts/environment
  except the immutable native executable and its expected hash.
- Both sources, warmed_process, C1 canonical solo and C8 canonical eight-offer
  wave, max8. Exact existing47 prompt IDs/common greedy/EOS policy. Each of four
  model/band cases runs ABBA then BAAB, eight fresh owned processes:32 measured
  samples,144 measured offers (16solo+128wave),72 offers/native label, plus32
  independent unmeasured warmup requests. ONE normal canonical prompt/max8
  warmup per process; no extra shape/full-wave warmup.
- Same exact four CPUs, CPU4 quota, RAM12GiB/no extra swap, original UUID/context,
  active8/native4GiB host/provider+5GiB device, read-only inputs and private user/
  network namespace; same V8 cgroup readback/point device observation. Native
  graph remains disabled in the accepted source. This is neither cold-process
  nor fully warmed shape/graph nor persistent-server steady-state measurement.

Gates and ownership:

- `bindings.json` pins the old actual accepted native metadata, candidate456
  source freeze, strict/release build, operator10/model46/full isolated+both
  approved routes and exact future worker14/HTTP12 receipt location. ROOT must
  supply its completed receipt SHA. Missing/RUNNING/partial/failed/unclean or
  unexpected-count receipt refuses preparation AND execution AND validation.
  ROOT should inspect all gate evidence before supplying the SHA; marker alone
  is insufficient. Later source drift or another runner version needs a fresh
  derivative, never a silently widened acceptance predicate.
- Original53ee nonce/name/CID container + reserved-leader WNOWAIT whole-group
  TERM/KILL-before-any-reap cleanup; original bounded log reader/EOF and V8
  ownership-authenticated point-memory monitor. Exact00d1 TERM+INT record-only
  guard spans provider lifetime, with explicit checks outside Popen handoff and
  retirement. No asynchronous exception crosses an unassigned child handle.
- Canonical HTTP retains its existing180s deadline/socket-timeout controls. A
  recorded cancel is checked after canonical return; this is no new hard
  asynchronous interrupt bound. Owner remains retained. Cleanup ignores recorded
  signals until all original stages close. No subsequent sample after any
  numerical, counter, transport, owner, log, memory-observer or receipt failure.
- Source/weights/binary binding scans run only in ROOT preparation/before+after
  the whole run, outside timed samples. Logs8MiB/process with truncation refused;
  whole output512MiB/1024files, every metadata write and final validation checks
  the tree INCLUDING its newly written receipt. These are artifact acceptance
  bounds, not a parent process physical-memory admission claim.

Metrics and acceptance:

- Every sample retains full raw benchmark BEFORE numerical/accounting checks.
  Exact full IDs/text SHA/usage/finish match all rows/versions/bands, including
  each ordinary warmup. No EOS/usage normalization or minimum-token workaround.
- Quiescent health before/after excludes warmup and cleanup. Every required
  counter is present, typed, monotonic; error/refusal/discard/cancel deltas0;
  weighted2/4/8 bands close actual fused rows, phases and sampler4/cohort. These
  are source launchsites, never CUDA/CUPTI enumerated kernel counts.
- C8 is client concurrency, not a promised nativeB8 cohort. Preserve actual row
  band deltas and packet cohort widths; no padding or forced cohort schedule.
  Actual packet frontiers47+index bind independent output progression. State/
  prefix cohorts have separate counters and are not reconstructed from SSE.
- Text TTFT and actual single-ID TTFT remain distinct. Keep offered/client-
  backlog TTFT and service TTFT. Individual-token ITL requires complete actual
  single-ID delivery; coalesced/unresolved rows are counted, never replaced by
  SSE chunk or meanITL proxies. Zero/one output ID has no ITL interval. Native
  commit observations remain a separate host clock, not device timestamps.
- Report offered/completed/usage throughput and all raw paired latencies plus
  local-label fairness, sampled memory, actual source counters. Four samples per
  label/band yield descriptive medians, no confidence-certified performance win.
  Eight finite simultaneous offers measure a burst, not sustained capacity.
  No source-correct SLO goodput, physical committed-token, fullHF, production
  graph or vLLM performance/parity claim follows.

ROOT sequence (not executed):

1. Complete/review current14worker/12HTTP source-frozen gate. Run the four pure
   `test_contracts.py` and four MOCK-only `test_owner_control.py` fixtures, then
   actual harmless-owner TERM/INT-before-Popen/during-readiness/during-
   request/during-cleanup failure-injection fixtures. Prior shadow/root evidence
   is not a test of this new adapter. Review all new source and bindings.
2. Set `ZG_BENCH_HOST_ROOT` to the original repository. Run
   `prepare_native_ab.py --hardware-protocol ABS_ACCEPTED_V8_PROTOCOL
   --hardware-protocol-sha256 SHA --accepted-serving-sha256 SHA
   --new-output ABS_THIS_RUN/root-plan-1`. That performs ROOT-only source scans,
   exact launch normalization and completed gate checks; no engine starts.
3. Under the original native-gpu user/network isolation, invoke Python with the
   original reviewed private-loopback helper (which execs Python), then
   `run_native_ab.py --run --plan ABS/root-plan-1/plan.json
   --plan-sha256 SHA --new-output ABS_THIS_RUN/root-native-1`.
4. `validate_native_ab.py --output ABS_THIS_RUN/root-native-1`. Independent ROOT
   review of full raw results/actual owner/point-memory/ITL coverage is required.
   Profiling/hardware counts and unprofiled timing remain separate runs.

````

## Source: .agents/runs/serving-and-speculation-native-before-after-20261020/frozen/TODO.md

Original UTF-8 bytes: 950; SHA256 `bd978523a1018a28be0a35e6689266a353badecd8a2ad36c2f47a859593af7e1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native before/after preparation

Parent: serving-and-speculation; owner: batch_design_review.
Owned: this new ignored derivative only; V8/old baseline/current456 unchanged.

- [x] Explicit native/native identities, exact existing launch/workload/caps.
- [x] Same canonical benchmark, complete32-sample/144-offer/32-warmup accounting.
- [x] Require completed actual candidate worker14/HTTP12 + prior source gates.
- [x] Prepare original owner cleanup +record-only TERM/INT checkpoints.
- [x] Full exact output/counter/band/ITL coverage and independent validation.
- [ ] ROOT/peer source review; actual pure/ownership failure fixtures.
- [ ] ROOT actual preparation +complete unprofiled native/native run+validation.
- [ ] Independent physical kernel diagnostic and measured performance acceptance.

Source-only AST and small source/metadata hashes by author. No tests, imports,
builds, native/provider/model/GPU execution or payload hashes/results.

````

## Source: .agents/runs/serving-and-speculation-native-forward-roi-audit-20261020/AUDIT.md

Original UTF-8 bytes: 11028; SHA256 `57b41522214d619038506374f4af977293d30ba1175e2361c638a574fa6d2eb3`.
Resolve embedded relative paths against the original path, not this volume.

````text
Native-forward optimization audit — source only, 2026-10-20

Recommend **batching existing state bit transport across rows first**
([actual owner](../../../crates/zerogravity-gpu/src/resident_program/batch.rs:680)). It removes
hundreds of launches per B8 cohort without changing one arithmetic operation,
token/root quantum, numerical profile or completion boundary. Next arithmetic
candidate is closed SiLU stage fusion; do not assume register casts preserve the
old physical stores. No new runtime, profiler, model or benchmark was executed.

Actual source: `9a199017f18f1bcb1a8053685be3c3c29236ec90`, pinned in
`source-pins.json`. Parent-owned current CPU-BF16 WIP was not touched. Warm
receipts below precede the newer RoPE body: they are not timings for this HEAD.
CUDA Graph remains qualifier-only; no live-capture admission is inferred.

Existing observations, not predicted speedups:
[.8B ABBA](../serving-and-speculation-native-prefix-before-after-20261020/actual-measurements/same-zg-56qiglta/report.json),
[2B naturalc8 BAAB](../serving-and-speculation-native-prefix-before-after-20261020/actual-natural-c8-measurements/same-zg-c8-q1d6tn3b/report.json).

| Source/workload | Old→native prefix, first pair | Kernel receipts | TTFT p50 | ID ITL p50 |
| --- | --- | --- | --- | --- |
| .8B,8 offers/B8 | same432 model rows,64 output IDs | 1,722,640→302,160 |29.625→15.748s|.3671→.3707s|
| 2B,8 offers/c8 | same432 model rows,64 output IDs | 1,668,496→294,384 |35.673→21.640s|.9227→.9035s|
| .8B,B1 |54 model rows,8 output IDs |246,314→246,314|3.910→4.052s|.0829→.0942s|
| 2B,B1 |54 model rows,8 output IDs |238,538→238,538|4.535→4.499s|.1580→.1570s|

All actual samples/second pairs, caps, exact binary IDs, warmed-process/cache
scope and health deltas are retained in `receipt-extracts.json`. Both endpoints
were Zerogravity. Source-correct SLO goodput remains unqualified. The original
strict2B B8 workload failure `same-zg-18c5xx5l` stays UNQUALIFIED; later natural
c8 is a separately declared workload, not a repaired/retried B8 claim. B1
kernel work stayed identical while calls/publications changed10→54; its timings
do not establish a scheduler regression or improvement without more evidence.

1. State gather/scatter: best low-risk next slice

- Actual caller: `resident_program/batch.rs:478` gathers, `:606` scatters;
  `transport_state:680` loops **rows then42 domains**.36 vector domains use one
  existing U32 copy each (`copy_dense:781`, current per-row lengths<1,000,000);
  six KV domains use one `transport_kv:1642` each, copying K+V together. Hence
  **84B launches**:168/336/672 atB2/4/8, outside any graph body, before/after
  the unchanged model nodes. These are actual launch sites counted in receipts,
  not workgroups, byte estimates or projection-dot subsets.
- New closed CUDA transport: one invocation per domain/direction accepts the
  actual2/4/8 private handles, source/target dense handle(s), explicit row stride,
  capacity and direction. Preserve exact U32 mapping `[T,B,H,D]` for KV, dense
  `[B,N]` for conv/recurrent; copy complete original capacity, including future
  zeros, zero signs, subnormals and opaque bits. No float conversion or arithmetic.
- At the new transport's explicit checked B×N bound, count can become84 per
  cohort (potential launch reductions84/252/588). **This is a proposed source
  count, not measured latency or a promise.** Existing generic copy helper caps
  one invocation at1M words; do not silently widen it. Either give the new
  multi-handle helper its own checked B<=8/per-row<=1M/total<=8M contract, or
  retain bounded chunks and report the actual larger number. Current recurrent
  `[8,262144]` exceeds1M total; a single old checked_copy_words call is invalid.
- Vector uses up to8 private bindings+dense+status, KV up to16 private K/V+
  denseK/V+status; check actual selected backend binding/argument capabilities
  cold. Inactive slots must never participate; no mutable alias or fake padding
  row. Keep non-CUDA original transport. No new persistent allocation/upload.
- Minimal ownership: new `resident_state_transport.rs` and qualifier;
  narrow `BatchProgram::transport_state`/runtime-preflight/counter wiring.
  `fused.rs` candidates, original access leases, queue, Ready/all-row publication,
  cancellation/quarantine and provider budget remain the owners. Keep all original
  prior+candidate handles in `handles_for_attempt`; no direct-root mutation.
- Gate: raw old/new transport all42/B2/4/8, independent per-row tags, unequal
  frontiers0/255/256/511, full-capacity KV; both approved models CPU+B1/native
  every teacher state/fullV+greedy, StateOnly no output; actual Ready discard,
  unknown completion, pressure and snapshot-outlives-provider/final0. Then the
  same bounded naturalc8/B1 ABBA/BAAB measure, preserving all wire/SLO metrics.
  No fused-model/state capability follows from a copy-only fixture.

2. Larger arithmetic launch pool: [SiLU](../../../crates/zerogravity-gpu/src/silu.rs:1), then small GDN stages

- `resident_program.rs:1617` still executes one interval check+12 SiLU launches
  +optional gate launch. `forward.rs:256,306,364` gives24 FFN+18 conv+18 GDN
  output-gate invocations:60 per token/body. Native batch falls through to SAME
  implementation (`batch.rs:1209`). Keeping interval/gate/certificates separate,
  a source-exact12→1 candidate could remove660 launches/body, independent ofB.
  Existing38-launch softmax adds228 across six attention layers;8 coefficient
  steps add144 across18 GDN layers; conv products+3 reductions add72. These are
  separate qualified math profiles, not interchangeable exponent formulas.
- Blocker is numerical, not API plumbing: `silu.rs:1` explicitly records failed
  compiler-local casts; old launches force real F32 stores. A future kernel must
  retain each exact source FMA/subtraction/product/quotient-correction/zero sign
  via explicit source-word rounding and independently prove every intermediate
  against the old physical12-launch oracle. No exp intrinsic, fast math, native
  FMA substitution, tolerance or BF16/HF helper reuse. CUDA-only cold selection;
  original Vulkan/AMD lowering/hold unchanged. This is a stronger qualification
  burden than row transport and is not yet an accepted optimization.

3. Repeated [activation quantization](../../../crates/zerogravity-gpu/src/resident_program/batch.rs:1353) / common-input projections

- Current CUDA quantized Project is **five launches before its output finite
  certificate**: fused block prep1 (`activation_quant_fused.rs:133`), composite1,
  block/row status merges2, original dot1 (`batch.rs:1353`; B1
  `device_activation.rs:271`+`resident_program.rs:1480`). Older6/4 block-prep
  launches are already gone on CUDA. Non-CUDA keeps its original stage sequence.
- Common inputs are real DAG identity, not repeated token IDs:
  `forward.rs:290` QKV/Z/alpha/beta,`:407` Q/K/V,`:256` FFN gate/up. Reuse only
  exact SourceIdentity+service+input ValueId/physical range+row layout+codec
  class/blockwidth/signed-scale/bias policy, within a single attempt. Q4_0/Q8_0
  block inputs can agree; Q6_K signed256 and Q8_1 status policy cannot be guessed
  equivalent. Matrix rows/weight scales/biases/composites remain distinct.
- Actual learned profile matters: alpha/beta may be F32 (V7 actual2B refused
  Q8_0 **ssm_out**, not alpha); do not budget four quantizers for every GDN input.
  At most3+2+1 duplicate inputs across18/6/24 layers gives90, but actual savings
  require cold plan+kind enumeration. For the common two-quantized GDN case it
  would be54 duplicate block preparations; no full-model count was executed here.
- Existing scratch0/1/2 is reused and per-operation status handles are distinct
  (`batch.rs:260`). Skipping prep naively leaves a consumer's block-status array
  stale; aliasing statuses invalidates the old unique-status proof. Minimal safe
  group must explicitly populate/retain each original consumer status or add a
  new checked shared-preparation certificate. Do not reuse weight-specific
  composite scales, skip SUM_NONFINITE/bias checks or rename invocation counts
  as physical kernels. This looks lower-impact than transport after the accepted
  cold fused preparation, though grouping dot work may later reduce enqueues.
- A later candidate can compute each matrix composite inside its original dot
  loop (`batch_composites:1715`, `batch_quant_projection.rs:75`) using the SAME
  rounded multiply/flags. It may avoid B×rows×blocks scale scratch traffic, but
  must prove all intermediate composite/status/source-dot bits; fusion is not
  justified by formula identity. Current dot kernels have not changed.

4. [Readout](../../../crates/zerogravity-gpu/src/resident_inference/fused.rs:608), embedding and scheduler boundaries

- Normal embedding is one native gather+status merge (`batch.rs:844,1761`), not
  a vocabulary projection/host transfer. Optimizing it first saves very few calls.
- Greedy still computes full original248320 logits once, then four native sampler
  stages plusB private packet copies (`resident_batch_greedy.rs:99`). Normal
  feedback reads **4+12B bytes**, StateOnly4 bytes (`fused.rs:608`); it already
  avoids fullV D2H and keeps original12B bound device packet authoritative.
  Do not optimize an imaginary per-token full-logit PCIe transfer. Packet copies
  can later use the same multi-handle bit-transport pattern (B→1, small savings).
- Final projection/greedy fusion would change the readout lowering and finite/
  max/tie certificate; weight bandwidth/FP64 dot cost remains. It is not the
  smallest next launch optimization and cannot replace fullV qualification.
- Every teacher quantum currently allocates private candidate states/packets,
  waits on the original fair queue, performs one compact completion and atomic
  root publication (`fused.rs:214–338,608–708`). Kernel count does not time CPU
  admission, allocator, enqueue or synchronization. No measured breakdown exists
  here; optimize those only after preserving physical budget and snapshot leases.
- Worker25ms receive timeout applies only when active is empty;2ms sleep only
  when all rows are credit parked (`gpu_engine.rs:731,772`). Neither is a generic
  per-token sleep to delete. One-teacher native credit is deliberate fairness,
  not an accidental32-token batching omission. Never hide32 consumes underReady1
  or cross the prefix/tail/EOS/credit/cancellation publication barriers.
- Projection source accumulation remains WideF64 (`source_accumulation.rs:1`),
  with NativeF32 diagnostic-only. Native CUDA FMA can be a distinct future
  source-bit qualification; device choice/IEEE intuition is not current proof.

No implementation/MAIN edit, source/profile promotion, vLLM-equivalence claim or
new benchmark result accompanies this audit. Largest arithmetic opportunities
remain experimental; the proposed row transport removes known redundant launch
work while preserving the accepted numerical graph.

````

## Source: .agents/runs/serving-and-speculation-native-forward-roi-audit-20261020/TODO.md

Original UTF-8 bytes: 605; SHA256 `2987d47544073c65b1fa975e01565e776171a04e211efd280a05145eb5f0aece`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: serving-and-speculation. Owner baseline_review. SOURCE ONLY.

- [x] Trace current original B1/native launch sites, state transport, quantization,
  readout and scheduler/physical completion ownership.
- [x] Extract prior ROOT warm receipts; preserve original strict2B B8 failure and
  separately declared naturalc8 result. No new runtime/performance measurement.
- [x] Recommend row-native U32 state transport, then separately qualified SiLU
  source-stage fusion; describe bounds, ownership and actual physical gates.
- [ ] ROOT chooses next implementation scope; no code/plan/profile promotion.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-before-after-20261020/NATURAL_C8.md

Original UTF-8 bytes: 6962; SHA256 `f7fb8161f04e2233805b55bb93419800468f31e5486789bcac61fb3439987003`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive natural concurrency8 workload

Parent: `.agents/plans/serving-and-speculation/TODO.md` (P12).
Author: `/root/baseline_review`; source-only preparation, root executes.

## Separate scope

- `measure_natural_c8.py`: explicit `--workloads c8,b1` (default), finite
  predeclared ABBA or BAAB. c8 means8 offers/client concurrency8/configured native
  maximum8. Actual native2/4/8 and B1 packet outputs are retained independently.
- Original strict B8 driver, six-file be0bbee8 freeze, all existing attempts and
  failure remain unchanged. No report-import/reclassification or retry mode.
- Fresh outputs use `actual-natural-c8-measurements/same-zg-c8-*`, a distinct
  schema/status and explicit workload semantics. Natural c8 may honestly show
  all B1 if that is what occurred; it cannot establish native width8 execution.
- The prior2B strict report `actual-measurements/same-zg-18c5xx5l/report.json`
  stays UNQUALIFIED, SHA256
  `5da6dc90184419ae53e467a4b86a9e35b117cc17255b6027ada4f9c19669b1c3`.
  Its complete8 offers/64 outputs did not meet that different fixed-B8 guard.

## Actual packet conservation

- Every output has explicit `gpu.native_cohort`: actual sequence/width2/4/8,
  or None for B1. No missing/unknown metadata is treated as B1.
- For each actual native sequence require one consistent width and exactly that
  many distinct request members. Retain each request/output index/producer
  frontier/token ID/wire ID/provider response ID. Different generation indices
  inside the same cohort are valid independent causal frontiers.
- Per-request native sequence order must increase. Every output belongs to
  exactly one complete native cohort or one listed B1 packet, including native
  tail→B1 and B1 tail→native transitions.
- Derive greedy band histogram only from observed packets. Subtract it from
  health total bands to obtain aggregate StateOnly bands for the new source.
  Require phase counts/rows to close. StateOnly membership is unobserved and is
  never reconstructed or claimed.
- Sampler count=4×observed native greedy cohorts, never4×all fused batches.
  Full kernel count covers disjoint native projection-dot and sampler subsets;
  actual kernel totals remain the source-bound physical receipts, not estimates.

For prompt frontier p and N requests (N=8 or1), model steps=N×(p+7), outputs=8N.

- New prefix: native StateOnly rows+B1 prefix tokens=N×(p−1). All B1 work consumes
  one token per scheduled pass. API calls=fused batches+(model steps−fused rows);
  committed roots=model steps. B1 greedy count is the actual listed B1 outputs.
- Accepted old source: prefix remains B1 max32 with last prompt tail separate;
  prefix API calls=N×ceil((p−1)/32). Calls=observed native greedy cohorts+B1
  outputs+prefix API calls; committed roots=8N+prefix API calls. Old phase fields
  remain absent/unobserved, never zero.

## Counter domain and namespace

The old compiled receipt identifies c5eb1f8 plus six exact overlays. GPU worker
and cohort source are not overlaid: their matching HEAD bytes are retained in
`natural-c8-counter-source/accepted/`. The exact health serializer overlay
matches d0a6084 bytes SHA84e30f0a. Current three source hashes are retained under
`current/` and match the source receipt in the preserved failed2B attempt.
`natural-c8-counter-contract.json` pins these small source files. Runtime checks
each reviewed build receipt against that exact worker/cohort/serializer contract
before starting providers. This is a source/build attestation, not an independent
reproducible-build proof.

In both inspected sources:

1. GpuServe creates one `Arc<WaveStats>` and passes its clone to the worker.
2. After successful `cohort.commit`, its `cohort_publications.fetch_add(1)+1`
   becomes `NativeCohortObservation.publication_sequence`.
3. Metrics reads that same atomic; serializer exposes
   `state_cohort_publications`. No earlier sequence reservation exists.
4. Abandoned, failed, cancelled-after-Ready and stale proposals do not increment
   it. B1 `single_completion` increments calls/model steps/roots, not this counter.
5. Samples reject failures/refusals/discards and use quiescent before/after
   health. Warmup is excluded. Each observed greedy sequence must lie in the
   exact same counter interval; its size closes against total native batches.

The counter belongs to the engine/worker stats, not a wire-exposed numeric GPU
ServiceId. Each sample namespace records owned foreground PID, measurement run
ID, singleton ready model alias, exact original source/tokenizer/binary hashes,
GPU UUID and source contract. The launcher declares only that one model and
requires readiness to expose exactly that alias. No SourceIdentity or ServiceId
number is fabricated; sequences are never merged across samples/providers/models.

## Preserved measurement/lifecycle rules

All underlying workload, settlement, signature, source/config/tokenizer/build
pins, provider argv/caps, greedy generation, memory sampling, lifecycle and
cleanup helpers remain byte-identical to the original. Health's text label now
says configured max8. Main changes only c8 naming/schema/output location,
source-domain binding and the new actual grouping/accounting closure.

- Same max8 tokens, context512, active/configured max8,4/4/5GiB caps, CPU4.
- Same full IDs/text hashes/usage/terminal/provider/wire/prompt/frontier guards
  and exact canonical equality across both providers and B1/c8 for each model.
- Same offered/completed/outcome denominators, true individual-ID ITL/commit
  clocks, fairness, fixed TTFT/ITL SLOs and raw health/memory/startup receipts.
- Same serialized owned lifecycle, no proxies/redirects, bounded logs,
  startup/warmup/measurement/shutdown receipts and GPU idle after cleanup.
- No production tool/source change, user process termination or privilege change.
- No claim of source-correct SLO eligibility, full model/state equivalence,
  causal speedup or vLLM parity from this measurement script.

## Prepared checks, not executed by author

`test_natural_c8.py` contains13 source-free methods: actual mixed layout and B1
remainder conservation; independent member indices/frontiers; original strict
B8 refusal; missing whole/partial cohorts; duplicate/width/type/sequence errors;
counter-domain interval; phase/band/sampler/API/root negatives; honest all-B1 c8;
old/new B1 policy; legacy observability; wire/outcome/usage/frontier guards;
source-contract refusal; unchanged underlying helpers.

The fixture's10-cohort membership is copied from the preserved failed capture;
other fixture payloads are explicitly synthetic. Tests do not promote that
failed attempt. Author only parsed ASTs and inspected static source/diffs;
no tests, providers, processes, models, operators, containers or GPU launches.

Root reviews, runs source-free fixtures, then chooses fresh pinned binaries and
an explicit finite ABBA/BAAB invocation. No automatic adjustment follows failure.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-before-after-20261020/NATURAL_C8_TODO.md

Original UTF-8 bytes: 857; SHA256 `f62e0a95a0da4c6d6acea2a86df878606684540bf3b0f708f5c157d8622ba2e1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Natural c8 variant

Parent: `.agents/plans/serving-and-speculation/TODO.md` (P12).
Owner: `/root/baseline_review`; additive source-only files in root-approved run.

- [x] Read original driver/tests progressively; preserve be0 six-file freeze.
- [x] Inspect preserved strict2B failure's small metadata without reclassifying.
- [x] Bind sequence counter domain to exact accepted/current source+receipts.
- [x] Prepare explicit c8 actual native2/4/8 +all B1 packet conservation.
- [x] Retain original lifecycle, source, wire, memory, SLO and B1/c8 equality.
- [x] Prepare13 source-free fixture methods; AST/static checks only, not run.
- [ ] Independent/root source review and root source-free test execution.
- [ ] Root fresh bounded c8+B1 ABBA/BAAB; no prior failure converted to PASS.
- [ ] Root measurement interpretation and shared-plan reconciliation.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-before-after-20261020/README.md

Original UTF-8 bytes: 8676; SHA256 `20e3a4ff0ca8251c3c907af3aa62e43d10a02ca5fa703c2d3af89311c0520388`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Same-engine native-prefix before/after — ROOT EXECUTES

Parent: .agents/plans/serving-and-speculation/TODO.md. Owner: spine_review.
This ignored source-only preparation changes no production/tools/shared plans.
ROOT owns all builds, tests, CPU/operator/model/container/GPU/provider execution.
Author performed source/AST/small metadata inspection only; no benchmark/test run.

Both slots ARE Zerogravity; there is no vLLM slot or matched-vLLM claim.

- A `accepted_d0prep_zerogravity`: exact preserved CUDA server SHA
  a896a455ec5d9093e88706751a9450170cb68b277658268ba30082f85e7ce598.
  Actual build receipt SHA148f990a06da8a94661304e2b89834c631fc6fc2662bd130d9e2f8dda9358ffb:
  compiled c5eb1f8e717c79146937e585a145a0052eff747b plus its six accepted
  source overlays, subsequently landed d0a6084. Never called a bare-d0 build.
- B `native_prefix_zerogravity`: ROOT must supply separately frozen actual
  binary SHA and reviewed source/build receipt path+SHA. Receipt must bind its
  server bytes/path, actual compiled40hex HEAD and source overlay hashes incl
  CLI worker/cohort/health plus resident fused implementation. Caller source-ID
  is labelled caller-supplied; receipt is attestation, not independent math or
  reproducible-build proof. No current binary or performance is invented here.
- Adapted reviewed measure_same_engine.py; pinned canonical serve-bench and
  serve-lifecycle modules unchanged. SameEnginePlan adapts initialization only;
  original sample/OwnedProvider/log/PGID/receipt-I/O/finite cleanup remain reused.
  Source pins record the exact original driver, accepted receipt and old worker
  source with max32 prefill scheduling. Old scripts/receipts stay untouched.

## Predeclared finite measurements

- Explicit ABBA or BAAB actual order, fresh owned process/sample. Run cold_process
  and warmed_process separately; repeat reverse order separately. Each invocation
  offers only selected b8 and/or b1 workloads, selected0.8B/2B, four samples each
  (maximum16 owned providers, never concurrent). No automatic retry after a
  failed sample, row-band miss, output mismatch, cleanup failure or SLO miss.
- Same approved original GGUF bytes, config/tokenizer hashes, model alias,
  prompt, source EOS, greedy temperature0/top_p1/n1 and output8 for A/B.
  No ignore-EOS policy. Early EOS refuses this bounded8-output comparison.
- Same selected GPU UUID; one worker, active8, configured native rows8/context512;
  host4096MiB/provider4096MiB/device5120MiB. Provider4GiB/device5GiB is the already
  physically qualified service envelope. Logical CPU affinity exactly4 allowed
  CPUs and Rayon/OMP/MKL/OpenBLAS4; not an OS CPU quota/physical RAM proof.
- b8 = eight actual HTTP offers/concurrency8, b1 = one actual offer/concurrency1,
  same prompt and fused-enabled server configuration. Workload arrivals all0,
  max_pending equals offers; no synthetic server barrier or dropped offers.
  Alternating local tenant labels are workload labels, not authenticated tenants.
- warmed_process uses one same-offer/max1 warmup wave, records every result, then
  one measured max8 wave. cold_process has no warmup. These names describe process
  policy only: OS/storage/device caches are untouched, not claimed cold/all warm.
  Source hashing occurs outside measured requests and can warm storage caches.
- Initial and post-owned-cleanup GPU idle: three no-compute samples, utilization
  <=5%, post memory<=initial+64MiB. Any other compute PID refuses without touching
  it. Owned executable must match frozen bytes/path and selected GPU must actually
  observe owned PID. Read-only .5s nvidia-smi/proc sampling can miss true peaks;
  device/RSS/HWM observations are not ResourceArbiter/exact allocator memory proof.
- Original bounded lifecycle: startup240s, shutdown30s, logs4MiB; request timeout
  <=180s, GPU-idle30s, health quiescence10s, sampler join8s. Cleanup precedes
  validation/evidence I/O/next spawn. Private unique outputs remain inside this run.
  Proxy-disabled and redirect-refusing loopback HTTP opener is installed before
  benchmark threads; no external service adoption or arbitrary stop commands.

## Evidence and accounting closure

- All offered, HTTP accepted/completed,429/503/rejected/cancelled/timeout/client
  overload/launch-deadline outcomes, exact per-request terminal usage and total
  usage must account. Every request must be actual HTTP200/completed/no error,
  retain exact offered/wire/provider identities, prompt SHA and final length.
- All8 actual token IDs/request require provider-bound complete ID status,
  individual client delivery and complete distinct provider commit observations.
  Same prompt must yield exact IDs/order/text SHA/usage/finish/frontiers across
  every request/sample and both B1/B8 workloads for each model. This is limited
  output evidence, not independent full-vocabulary/42-state mathematical proof.
- Actual packet-producing committed frontier P comes from provider observations,
  never tokenizer-length inference. Each request must report P..P+7, same P.
- b8 must physically observe exactly8 width8 greedy publication groups across
  all64 output IDs, each group containing every offered request exactly once at
  the same output index. Row-band miss refuses; it is not relabelled B8.
- New health requires all three phase counters: state-only batches/rows and
  greedy batches. Total fused batches/rows/weighted2/4/8 bands and actual atomic
  cohort publications/rows must close. There must be genuine state-only prefix
  cohorts plus8 greedy cohorts; sampler dispatches=4*greedy, never4*all prefix
  cohorts. New API calls=fused_batches+model_token_steps-fused_rows; committed
  roots=model_token_steps. B1 remainder work remains counted.
- Old phase fields MUST remain absent: report None/legacy_absent_not_observed,
  never fabricate observed zero. Its actual eight greedy cohorts/64 rows and
 4*8 sampler launches close separately; original max32 prefix API/root receipts
  are retained. Common counters, including state_cohort_rows, are observed in
  both accepted old and new health.
- b1 must have zero actual native multirow cohorts/projections/samplers in both.
  With actual P and8 outputs, model steps=P+7. Old prefix API count is
  ceil((P-1)/32), plus8 greedy/API/publication calls; new one-teacher policy is
  P-1 prefix+8=P+7. Report this explicit scheduling cost beside actual physical
  forward-kernel receipts, TTFT/ITL, wall/memory; do not infer a causal speedup.
- Actual completed full-forward kernels cover disjoint native projection/sampler
  subsets; projection means final-dot subset, sampler counts four native B>1
  kernels. B1 samplers, initialization/DMA/unknown completion have the original
  counter scope, not a false graph-as-one-kernel interpretation.
- No failed/refused/discarded/cancelled GPU work accepted in timing samples.
  Queued/active/parked fields remain before/after gauges; zero endpoints do not
  establish zero parking incidents. Full raw health and per-request records stay.
- TTFT<=10s and actual single-ID delivery ITL<=2s are predeclared maxima; CLI
  may tighten them, never widen. SLO misses remain recorded, not hidden/retried.
  Actual ID delivery ITL and provider commit-observation clocks stay distinct.
  Text TTFT/SSE chunk intervals are not true token ITL; no source-correct SLO
  goodput or vLLM equivalence is inferred from this same-engine experiment.

## ROOT commands after physical acceptance

Ten source-free accounting fixture methods are prepared, AST-only by author.
ROOT first executes fixtures, reviews the current actual source/build receipt,
freezes its binary, and waits for all CPU/build/GPU measurement holds to lift.

```sh
TASK_PREFIX_AB=.agents/runs/serving-and-speculation-native-prefix-before-after-20261020
python3 "$TASK_PREFIX_AB/test_driver.py"
python3 "$TASK_PREFIX_AB/measure_same_engine.py" --run \
  --current-server /ABSOLUTE/OWNED/FROZEN/NATIVE-PREFIX-SERVER \
  --current-sha256 ACTUAL_64HEX_BINARY_SHA \
  --current-source-id ACTUAL_SOURCE_BUILD_RECEIPT_ID \
  --current-receipt /ABSOLUTE/OWNED/ACTUAL_BUILD_RECEIPT.json \
  --current-receipt-sha256 ACTUAL_64HEX_RECEIPT_SHA \
  --gpu-uuid ACTUAL_GPU_UUID --models 0.8b --workloads b8,b1 \
  --order ABBA --policy warmed_process
```

Inspect first0.8B report before2B, reverse order and cold controls. Each invocation
creates private actual-measurements/same-zg-* with full raw observations and
original lifecycle receipts. Success means bounded same-Zerogravity observations
only. Unchanged outputs, physical prefix fusion and source/math/full-serving
acceptance remain distinct claims; no fabricated ratio or matched-vLLM result.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-before-after-20261020/TODO.md

Original UTF-8 bytes: 1297; SHA256 `ef8a7de1b362dc5b3b9c27c6f179cededeaea207750c04dbab11367c3f97055f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native-prefix before/after source preparation

Parent: .agents/plans/serving-and-speculation/TODO.md. Owner: spine_review.
ROOT alone executes all build/test/provider/CPU/model/operator/GPU work.

- [x] Preserve original accepted driver/actual receipts; identify exact a896
  accepted cold-preparation server and c5+six-overlay build attestation.
- [x] Adapt original owned lifecycle with honest same-engine labels, same caps,
  ABBA/BAAB order, cold/warm policies, actual8-offer and1-offer workloads.
- [x] Separate observed prefix/greedy metrics from absent legacy phase fields;
  close actual greedy output memberships, rows/bands/sampler/kernel/root counts.
- [x] Keep all offered outcomes, true ID delivery/provider commit clocks, exact
  outputs/frontiers/usage across samples and B1/B8; no wider SLO or vLLM claim.
- [x] Prepare10 source-free accounting fixture methods; Python AST only by author.
- [ ] Independent source review, ROOT fixture execution and fresh current-server
  actual artifact/source receipt after physical prefix gates.
- [ ] ROOT only actual serialized bounded cold/warm ABBA/BAAB after all competing
  build/CPU/GPU holds lift; report measured B1 scheduling cost and B8 gains/limits.
- [ ] Genuine matched pinned-vLLM/source-correct SLO gate remains separate/open.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-before-after-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 8676; SHA256 `20e3a4ff0ca8251c3c907af3aa62e43d10a02ca5fa703c2d3af89311c0520388`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Same-engine native-prefix before/after — ROOT EXECUTES

Parent: .agents/plans/serving-and-speculation/TODO.md. Owner: spine_review.
This ignored source-only preparation changes no production/tools/shared plans.
ROOT owns all builds, tests, CPU/operator/model/container/GPU/provider execution.
Author performed source/AST/small metadata inspection only; no benchmark/test run.

Both slots ARE Zerogravity; there is no vLLM slot or matched-vLLM claim.

- A `accepted_d0prep_zerogravity`: exact preserved CUDA server SHA
  a896a455ec5d9093e88706751a9450170cb68b277658268ba30082f85e7ce598.
  Actual build receipt SHA148f990a06da8a94661304e2b89834c631fc6fc2662bd130d9e2f8dda9358ffb:
  compiled c5eb1f8e717c79146937e585a145a0052eff747b plus its six accepted
  source overlays, subsequently landed d0a6084. Never called a bare-d0 build.
- B `native_prefix_zerogravity`: ROOT must supply separately frozen actual
  binary SHA and reviewed source/build receipt path+SHA. Receipt must bind its
  server bytes/path, actual compiled40hex HEAD and source overlay hashes incl
  CLI worker/cohort/health plus resident fused implementation. Caller source-ID
  is labelled caller-supplied; receipt is attestation, not independent math or
  reproducible-build proof. No current binary or performance is invented here.
- Adapted reviewed measure_same_engine.py; pinned canonical serve-bench and
  serve-lifecycle modules unchanged. SameEnginePlan adapts initialization only;
  original sample/OwnedProvider/log/PGID/receipt-I/O/finite cleanup remain reused.
  Source pins record the exact original driver, accepted receipt and old worker
  source with max32 prefill scheduling. Old scripts/receipts stay untouched.

## Predeclared finite measurements

- Explicit ABBA or BAAB actual order, fresh owned process/sample. Run cold_process
  and warmed_process separately; repeat reverse order separately. Each invocation
  offers only selected b8 and/or b1 workloads, selected0.8B/2B, four samples each
  (maximum16 owned providers, never concurrent). No automatic retry after a
  failed sample, row-band miss, output mismatch, cleanup failure or SLO miss.
- Same approved original GGUF bytes, config/tokenizer hashes, model alias,
  prompt, source EOS, greedy temperature0/top_p1/n1 and output8 for A/B.
  No ignore-EOS policy. Early EOS refuses this bounded8-output comparison.
- Same selected GPU UUID; one worker, active8, configured native rows8/context512;
  host4096MiB/provider4096MiB/device5120MiB. Provider4GiB/device5GiB is the already
  physically qualified service envelope. Logical CPU affinity exactly4 allowed
  CPUs and Rayon/OMP/MKL/OpenBLAS4; not an OS CPU quota/physical RAM proof.
- b8 = eight actual HTTP offers/concurrency8, b1 = one actual offer/concurrency1,
  same prompt and fused-enabled server configuration. Workload arrivals all0,
  max_pending equals offers; no synthetic server barrier or dropped offers.
  Alternating local tenant labels are workload labels, not authenticated tenants.
- warmed_process uses one same-offer/max1 warmup wave, records every result, then
  one measured max8 wave. cold_process has no warmup. These names describe process
  policy only: OS/storage/device caches are untouched, not claimed cold/all warm.
  Source hashing occurs outside measured requests and can warm storage caches.
- Initial and post-owned-cleanup GPU idle: three no-compute samples, utilization
  <=5%, post memory<=initial+64MiB. Any other compute PID refuses without touching
  it. Owned executable must match frozen bytes/path and selected GPU must actually
  observe owned PID. Read-only .5s nvidia-smi/proc sampling can miss true peaks;
  device/RSS/HWM observations are not ResourceArbiter/exact allocator memory proof.
- Original bounded lifecycle: startup240s, shutdown30s, logs4MiB; request timeout
  <=180s, GPU-idle30s, health quiescence10s, sampler join8s. Cleanup precedes
  validation/evidence I/O/next spawn. Private unique outputs remain inside this run.
  Proxy-disabled and redirect-refusing loopback HTTP opener is installed before
  benchmark threads; no external service adoption or arbitrary stop commands.

## Evidence and accounting closure

- All offered, HTTP accepted/completed,429/503/rejected/cancelled/timeout/client
  overload/launch-deadline outcomes, exact per-request terminal usage and total
  usage must account. Every request must be actual HTTP200/completed/no error,
  retain exact offered/wire/provider identities, prompt SHA and final length.
- All8 actual token IDs/request require provider-bound complete ID status,
  individual client delivery and complete distinct provider commit observations.
  Same prompt must yield exact IDs/order/text SHA/usage/finish/frontiers across
  every request/sample and both B1/B8 workloads for each model. This is limited
  output evidence, not independent full-vocabulary/42-state mathematical proof.
- Actual packet-producing committed frontier P comes from provider observations,
  never tokenizer-length inference. Each request must report P..P+7, same P.
- b8 must physically observe exactly8 width8 greedy publication groups across
  all64 output IDs, each group containing every offered request exactly once at
  the same output index. Row-band miss refuses; it is not relabelled B8.
- New health requires all three phase counters: state-only batches/rows and
  greedy batches. Total fused batches/rows/weighted2/4/8 bands and actual atomic
  cohort publications/rows must close. There must be genuine state-only prefix
  cohorts plus8 greedy cohorts; sampler dispatches=4*greedy, never4*all prefix
  cohorts. New API calls=fused_batches+model_token_steps-fused_rows; committed
  roots=model_token_steps. B1 remainder work remains counted.
- Old phase fields MUST remain absent: report None/legacy_absent_not_observed,
  never fabricate observed zero. Its actual eight greedy cohorts/64 rows and
 4*8 sampler launches close separately; original max32 prefix API/root receipts
  are retained. Common counters, including state_cohort_rows, are observed in
  both accepted old and new health.
- b1 must have zero actual native multirow cohorts/projections/samplers in both.
  With actual P and8 outputs, model steps=P+7. Old prefix API count is
  ceil((P-1)/32), plus8 greedy/API/publication calls; new one-teacher policy is
  P-1 prefix+8=P+7. Report this explicit scheduling cost beside actual physical
  forward-kernel receipts, TTFT/ITL, wall/memory; do not infer a causal speedup.
- Actual completed full-forward kernels cover disjoint native projection/sampler
  subsets; projection means final-dot subset, sampler counts four native B>1
  kernels. B1 samplers, initialization/DMA/unknown completion have the original
  counter scope, not a false graph-as-one-kernel interpretation.
- No failed/refused/discarded/cancelled GPU work accepted in timing samples.
  Queued/active/parked fields remain before/after gauges; zero endpoints do not
  establish zero parking incidents. Full raw health and per-request records stay.
- TTFT<=10s and actual single-ID delivery ITL<=2s are predeclared maxima; CLI
  may tighten them, never widen. SLO misses remain recorded, not hidden/retried.
  Actual ID delivery ITL and provider commit-observation clocks stay distinct.
  Text TTFT/SSE chunk intervals are not true token ITL; no source-correct SLO
  goodput or vLLM equivalence is inferred from this same-engine experiment.

## ROOT commands after physical acceptance

Ten source-free accounting fixture methods are prepared, AST-only by author.
ROOT first executes fixtures, reviews the current actual source/build receipt,
freezes its binary, and waits for all CPU/build/GPU measurement holds to lift.

```sh
TASK_PREFIX_AB=.agents/runs/serving-and-speculation-native-prefix-before-after-20261020
python3 "$TASK_PREFIX_AB/test_driver.py"
python3 "$TASK_PREFIX_AB/measure_same_engine.py" --run \
  --current-server /ABSOLUTE/OWNED/FROZEN/NATIVE-PREFIX-SERVER \
  --current-sha256 ACTUAL_64HEX_BINARY_SHA \
  --current-source-id ACTUAL_SOURCE_BUILD_RECEIPT_ID \
  --current-receipt /ABSOLUTE/OWNED/ACTUAL_BUILD_RECEIPT.json \
  --current-receipt-sha256 ACTUAL_64HEX_RECEIPT_SHA \
  --gpu-uuid ACTUAL_GPU_UUID --models 0.8b --workloads b8,b1 \
  --order ABBA --policy warmed_process
```

Inspect first0.8B report before2B, reverse order and cold controls. Each invocation
creates private actual-measurements/same-zg-* with full raw observations and
original lifecycle receipts. Success means bounded same-Zerogravity observations
only. Unchanged outputs, physical prefix fusion and source/math/full-serving
acceptance remain distinct claims; no fabricated ratio or matched-vLLM result.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-before-after-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1297; SHA256 `ef8a7de1b362dc5b3b9c27c6f179cededeaea207750c04dbab11367c3f97055f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native-prefix before/after source preparation

Parent: .agents/plans/serving-and-speculation/TODO.md. Owner: spine_review.
ROOT alone executes all build/test/provider/CPU/model/operator/GPU work.

- [x] Preserve original accepted driver/actual receipts; identify exact a896
  accepted cold-preparation server and c5+six-overlay build attestation.
- [x] Adapt original owned lifecycle with honest same-engine labels, same caps,
  ABBA/BAAB order, cold/warm policies, actual8-offer and1-offer workloads.
- [x] Separate observed prefix/greedy metrics from absent legacy phase fields;
  close actual greedy output memberships, rows/bands/sampler/kernel/root counts.
- [x] Keep all offered outcomes, true ID delivery/provider commit clocks, exact
  outputs/frontiers/usage across samples and B1/B8; no wider SLO or vLLM claim.
- [x] Prepare10 source-free accounting fixture methods; Python AST only by author.
- [ ] Independent source review, ROOT fixture execution and fresh current-server
  actual artifact/source receipt after physical prefix gates.
- [ ] ROOT only actual serialized bounded cold/warm ABBA/BAAB after all competing
  build/CPU/GPU holds lift; report measured B1 scheduling cost and B8 gains/limits.
- [ ] Genuine matched pinned-vLLM/source-correct SLO gate remains separate/open.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-before-after-20261020/natural-c8-frozen/NATURAL_C8.md

Original UTF-8 bytes: 6962; SHA256 `f7fb8161f04e2233805b55bb93419800468f31e5486789bcac61fb3439987003`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive natural concurrency8 workload

Parent: `.agents/plans/serving-and-speculation/TODO.md` (P12).
Author: `/root/baseline_review`; source-only preparation, root executes.

## Separate scope

- `measure_natural_c8.py`: explicit `--workloads c8,b1` (default), finite
  predeclared ABBA or BAAB. c8 means8 offers/client concurrency8/configured native
  maximum8. Actual native2/4/8 and B1 packet outputs are retained independently.
- Original strict B8 driver, six-file be0bbee8 freeze, all existing attempts and
  failure remain unchanged. No report-import/reclassification or retry mode.
- Fresh outputs use `actual-natural-c8-measurements/same-zg-c8-*`, a distinct
  schema/status and explicit workload semantics. Natural c8 may honestly show
  all B1 if that is what occurred; it cannot establish native width8 execution.
- The prior2B strict report `actual-measurements/same-zg-18c5xx5l/report.json`
  stays UNQUALIFIED, SHA256
  `5da6dc90184419ae53e467a4b86a9e35b117cc17255b6027ada4f9c19669b1c3`.
  Its complete8 offers/64 outputs did not meet that different fixed-B8 guard.

## Actual packet conservation

- Every output has explicit `gpu.native_cohort`: actual sequence/width2/4/8,
  or None for B1. No missing/unknown metadata is treated as B1.
- For each actual native sequence require one consistent width and exactly that
  many distinct request members. Retain each request/output index/producer
  frontier/token ID/wire ID/provider response ID. Different generation indices
  inside the same cohort are valid independent causal frontiers.
- Per-request native sequence order must increase. Every output belongs to
  exactly one complete native cohort or one listed B1 packet, including native
  tail→B1 and B1 tail→native transitions.
- Derive greedy band histogram only from observed packets. Subtract it from
  health total bands to obtain aggregate StateOnly bands for the new source.
  Require phase counts/rows to close. StateOnly membership is unobserved and is
  never reconstructed or claimed.
- Sampler count=4×observed native greedy cohorts, never4×all fused batches.
  Full kernel count covers disjoint native projection-dot and sampler subsets;
  actual kernel totals remain the source-bound physical receipts, not estimates.

For prompt frontier p and N requests (N=8 or1), model steps=N×(p+7), outputs=8N.

- New prefix: native StateOnly rows+B1 prefix tokens=N×(p−1). All B1 work consumes
  one token per scheduled pass. API calls=fused batches+(model steps−fused rows);
  committed roots=model steps. B1 greedy count is the actual listed B1 outputs.
- Accepted old source: prefix remains B1 max32 with last prompt tail separate;
  prefix API calls=N×ceil((p−1)/32). Calls=observed native greedy cohorts+B1
  outputs+prefix API calls; committed roots=8N+prefix API calls. Old phase fields
  remain absent/unobserved, never zero.

## Counter domain and namespace

The old compiled receipt identifies c5eb1f8 plus six exact overlays. GPU worker
and cohort source are not overlaid: their matching HEAD bytes are retained in
`natural-c8-counter-source/accepted/`. The exact health serializer overlay
matches d0a6084 bytes SHA84e30f0a. Current three source hashes are retained under
`current/` and match the source receipt in the preserved failed2B attempt.
`natural-c8-counter-contract.json` pins these small source files. Runtime checks
each reviewed build receipt against that exact worker/cohort/serializer contract
before starting providers. This is a source/build attestation, not an independent
reproducible-build proof.

In both inspected sources:

1. GpuServe creates one `Arc<WaveStats>` and passes its clone to the worker.
2. After successful `cohort.commit`, its `cohort_publications.fetch_add(1)+1`
   becomes `NativeCohortObservation.publication_sequence`.
3. Metrics reads that same atomic; serializer exposes
   `state_cohort_publications`. No earlier sequence reservation exists.
4. Abandoned, failed, cancelled-after-Ready and stale proposals do not increment
   it. B1 `single_completion` increments calls/model steps/roots, not this counter.
5. Samples reject failures/refusals/discards and use quiescent before/after
   health. Warmup is excluded. Each observed greedy sequence must lie in the
   exact same counter interval; its size closes against total native batches.

The counter belongs to the engine/worker stats, not a wire-exposed numeric GPU
ServiceId. Each sample namespace records owned foreground PID, measurement run
ID, singleton ready model alias, exact original source/tokenizer/binary hashes,
GPU UUID and source contract. The launcher declares only that one model and
requires readiness to expose exactly that alias. No SourceIdentity or ServiceId
number is fabricated; sequences are never merged across samples/providers/models.

## Preserved measurement/lifecycle rules

All underlying workload, settlement, signature, source/config/tokenizer/build
pins, provider argv/caps, greedy generation, memory sampling, lifecycle and
cleanup helpers remain byte-identical to the original. Health's text label now
says configured max8. Main changes only c8 naming/schema/output location,
source-domain binding and the new actual grouping/accounting closure.

- Same max8 tokens, context512, active/configured max8,4/4/5GiB caps, CPU4.
- Same full IDs/text hashes/usage/terminal/provider/wire/prompt/frontier guards
  and exact canonical equality across both providers and B1/c8 for each model.
- Same offered/completed/outcome denominators, true individual-ID ITL/commit
  clocks, fairness, fixed TTFT/ITL SLOs and raw health/memory/startup receipts.
- Same serialized owned lifecycle, no proxies/redirects, bounded logs,
  startup/warmup/measurement/shutdown receipts and GPU idle after cleanup.
- No production tool/source change, user process termination or privilege change.
- No claim of source-correct SLO eligibility, full model/state equivalence,
  causal speedup or vLLM parity from this measurement script.

## Prepared checks, not executed by author

`test_natural_c8.py` contains13 source-free methods: actual mixed layout and B1
remainder conservation; independent member indices/frontiers; original strict
B8 refusal; missing whole/partial cohorts; duplicate/width/type/sequence errors;
counter-domain interval; phase/band/sampler/API/root negatives; honest all-B1 c8;
old/new B1 policy; legacy observability; wire/outcome/usage/frontier guards;
source-contract refusal; unchanged underlying helpers.

The fixture's10-cohort membership is copied from the preserved failed capture;
other fixture payloads are explicitly synthetic. Tests do not promote that
failed attempt. Author only parsed ASTs and inspected static source/diffs;
no tests, providers, processes, models, operators, containers or GPU launches.

Root reviews, runs source-free fixtures, then chooses fresh pinned binaries and
an explicit finite ABBA/BAAB invocation. No automatic adjustment follows failure.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-before-after-20261020/natural-c8-frozen/NATURAL_C8_TODO.md

Original UTF-8 bytes: 857; SHA256 `f62e0a95a0da4c6d6acea2a86df878606684540bf3b0f708f5c157d8622ba2e1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Natural c8 variant

Parent: `.agents/plans/serving-and-speculation/TODO.md` (P12).
Owner: `/root/baseline_review`; additive source-only files in root-approved run.

- [x] Read original driver/tests progressively; preserve be0 six-file freeze.
- [x] Inspect preserved strict2B failure's small metadata without reclassifying.
- [x] Bind sequence counter domain to exact accepted/current source+receipts.
- [x] Prepare explicit c8 actual native2/4/8 +all B1 packet conservation.
- [x] Retain original lifecycle, source, wire, memory, SLO and B1/c8 equality.
- [x] Prepare13 source-free fixture methods; AST/static checks only, not run.
- [ ] Independent/root source review and root source-free test execution.
- [ ] Root fresh bounded c8+B1 ABBA/BAAB; no prior failure converted to PASS.
- [ ] Root measurement interpretation and shared-plan reconciliation.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-chunk-20261020/README.md

Original UTF-8 bytes: 7188; SHA256 `cac8180c3c68a5f82e483a3aab0dfee34b3c3add8f41ed5e21f5fb21ab7af7f3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed native teacher chunk T2/T4

Author: baseline_review. Parent: existing serving-and-speculation P12 plan.
Source-only proposal. ROOT owns MAIN, merge, Cargo/runtime/GPU, qualification and
shared plan reconciliation. No build/test/operator/provider/model/GPU execution.

## Exact change

- Existing BatchProgram, unchanged original ForwardPlan state prefix and original
  kernels, sequentially T times at B2/4/8. This is AR1 order for each row. There
  is NO B*T projection/MLP grouping, temporal BMM, new model schedule, engine,
  trained-HF arithmetic, CPU fallback or graph replay in this proposal.
- Qualification-only `ResidentModel::prepare_state_fused_chunk_admitted` accepts
  exact equal per-row T2/T4 teacher slices and immutable original root bindings.
  All IDs, source, model, original capacity <=512, exact independent frontiers,
  full span and first-token correspondence are checked. Ragged teacher lengths,
  duplicate/foreign/stale rows, device feedback/readout mixing and graph pins
  refuse. None remains only original physical capacity refusal before launch.
- Reuse the ALREADY admitted two packed state banks. Gather original independent
  roots once. Run the same state-only node body at each row's position+step.
  Original node certificates and embedding/quant status merges already OR into
  one status; clear it only once. Before each bank swap, certify ALL48 packed
  candidate fields, including full KV capacity, using the original finite U32
  predicate. Original1,000,000-word certificate chunks safely cover B8*262144
  state words without unchecked reuse of a one-million-word limit.
- Swap only private packed-bank metadata; RAII restores original orientation on
  every normal error/unwind. Final even-T state is in the originally prior bank,
  so a tiny existing transport preflight factor checks THAT exact final scatter
  bank before submission. Default swap_banks=false is byte/order-equivalent to
  accepted single-step lowering. Final certified scatter runs once into genuine
  independent candidate BufferLeases. No extra GPU state bank/output/sampler or
  warm source/state/activation upload is added.
- Original allocation->queue/workspace order, same ResidentService/client,
  original slot/flight/access leases, readback validator, Ready and atomic all-row
  publisher remain the owners. One original authority.begin_advance(T) per row;
  one terminal4B status readback/Ready; one root publication advances frontierT.
  T is honestly reported in state_only_steps. Physical dispatch count includes
  every body, parameter staging, intermediate certificate and gather/scatter.
  Native projection subset is summed across T. No graph enqueue/kernel1 claim.
- Unknown completion/status rejection retains original conservative fence and
  quarantine. Known-Ready caller cancellation is drop of ALL private candidates;
  a stale row refuses ALL publication. No recovery, output packet, generated ID,
  EOS handling, SSE, usage or Done exists in teacher chunk preparation.

## Source ownership and composition

`BASELINE.json` retains exact original source bytes, including accepted prefix,
RoPE, graph and certified-scatter changes. `NEW_DELTA.patch` owns six source paths:

- resident_inference/fused.rs: closed execution variant/entry, T authority/span
  and truthful readback counts; normal None/Greedy/Logits and graph paths kept.
- resident_batch.rs plus new resident_batch/chunk.rs: sealed bounded teacher
  metadata and two source-free authority/range tests.
- resident_program/batch.rs plus new resident_program/batch/chunk.rs: tiny
  default-preserving bank selector/module wiring and the existing-body T loop.
- One additional new qualification child under
  resident_inference/prefix_qualification/chunk.rs.

The list above is six source paths total: three existing parents/three new
children. ROOT alone applies the separate tiny `ROOT_WIRING.patch` mounting that
qualifier beneath existing prefix qualification (a seventh combined path). No lib.rs/CLI/HTTP/tools/plans/default feature/profile changes.
Batch's ordered-SiLU candidate edits launch_node and can be composed separately;
this proposal keeps that baseline math unchanged. Apply only reviewed hunks with
fresh modification times; never overwrite a newer root parent with old bytes.

## Prepared acceptance and limits

- Twelve ignored full-model cases: both original .8B/2B, B2/4/8, T2/4. Genuine
  distinct nonzero GPU/CPU histories and unequal frontiers, source tokenizer EOS
  as teacher, independent accepted CPU+B1 complete vocabulary/all42 AFTER EACH
  reference teacher, exact native all42 at final scope. Then real future native
  full248320 logits and accepted CPU/B1/native greedy packet ID/max/all42 parity.
- Exact4B total terminal readback, T steps/row, zero sampler/logit/packet during
  prefix, one root advance, source/model uploads unchanged, pending0/final0.
  Bad last teacher/ragged/duplicate rows cold-refuse; actual Ready abandonment,
  stale restore and all-sibling rollback preserve full prior state/leases.
- Two physical pressure cases: actual original Scratch lease prevents B8 before
  any queue work; explicit B4 scopes execute; B8 executes after release; no
  hidden fallback, full CPU/B1/all42/future fullV and final0.
- Four optional portable-checkpoint boundary cases (B2,T2/4,both models): genuine
  CPU-derived prefix states imported through original authority, cross256 and
  reach512; every reference teacher fullV/all42 and final nativeall42, capacity
  refusal unchanged. Scope is CPU-hydrated GPU decode, never GPU full prefill.
- Two fresh-process actual numerical-status negatives. A cfg(test) one-shot
  poisons ONLY an intermediate inactive packed KV word with NaN; a later original
  append heals final padding to positive zero. Original full-word intermediate
  certificate must latch status2 across later bodies. Direct raw original-handle
  readbacks prove status2/final padding0; no Ready/publication. Prior all42 raw
  bits exact, original conservative quarantine/fence/envelope retained. Synthetic
  fault is not an expected trained result and never exists in non-test builds.
- Two hardware-abort-feature fresh-process negatives: actual whole chunk queued,
  future dropped before terminal Ready, all original handles quarantined, every
  prior root/all42 bits unchanged and service/provider fence retained. This is
  omitted-terminal-proof safety, not a fabricated driver fault/recovery.

Still UNQUALIFIED: all prepared code compilation/runtime, actual driver counts,
latency/throughput, production scheduler use, aggregate host/control allocation
bounds, global graph capture, trained HF/FLA/27B and other families. The worker
still consumes its accepted one-teacher quantum. A future scheduler cut must
partition prefix vs final tail BEFORE bands, choose only equal real remaining
T2/T4 with each remaining prefix>=T, cap fair quantum<=4, and preserve cancellation,
parked/credit ownership and singleton truthfulness. Ragged requests must regroup
without zero/EOS padding; an actual stopped/final-tail row cannot enter this chunk.
No performance claim or production enablement follows from a source-only freeze.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-chunk-20261020/ROOT-GATES.md

Original UTF-8 bytes: 3522; SHA256 `e85d41e77b3b2242d5239c550fca5b2a2c874612c6870116db9993be40f03cb3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT qualification cuts

No author execution. Source-only candidate remains closed behind qualification
+cuda. Preserve every failed attempt. Existing original source/logit/state limits
and tolerance0 remain; do not substitute analytical/generated expected values.

1. Independent complete source review; compile default/non-CUDA +strict CUDA,
   Vulkan and existing legacy/HF cold guards. Check bank selector defaultfalse,
   original state-only body and lease/publisher tail against BASELINE exact bytes.
   Logical tests in `resident_batch::chunk::tests` are source-free, not CUDA proof.
2. Serial release model cases. New namespace:
   `resident_inference::fused_qualification::prefix_qualification::chunk::`.
   Run only twelve `qwen{08,2}_b{2,4,8}_t{2,4}` tests at first, using anchored
   exact filters or ROOT's test-list proof. Preserve sources/binary and actual
   counts. Existing original approved sources only; no new downloads/profile.
3. Run two exact pressure tests (`qwen08_chunk_pressure`, `qwen2_chunk_pressure`).
   These require prelaunch None/unchanged queue+root+resident bytes under genuine
   held lease, explicit B4 and B8 after release, no hidden retry.
4. With portable-checkpoint, four exact boundary tests
   `qwen{08,2}_chunk_boundary_t{2,4}`. Real independently computed CPU prefixes
   cross padded attention256 and end512. Imported actual state is labeled as such.
5. EACH numerical status refusal in a FRESH process:
   `qwen08_chunk_certificate_refusal`, `qwen2_chunk_certificate_refusal`.
   Actual original status validation intentionally fences the card/service and
   retains provider envelope; do NOT append a successful model test or recovery.
   Synthetic inactive-KV transient NaN must be caught even though final physical
   padding is positive zero. There is no status clear between teachers.
6. EACH hardware-abort test in a FRESH process, hardware-abort-probe enabled:
   `qwen08_chunk_abort`, `qwen2_chunk_abort`. Require actual queued original
   handles and omitted Ready, complete quarantine/retained caps/unchanged root
   and raw prior all42; no retry/device reset/privilege change.
7. Existing unchanged B1/native fullV/greedy/all42/unequal-frontier/mixed256/512,
   opaque+typed scatter canaries/status/offset/alias/refusal, graph owner/profile,
   original completion/quarantine, worker/HTTP/cancel/credit and approved original
   cache-direct/HF-import release routes must remain exact. Run full isolated:

   `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`

8. Before production scheduler integration, bind true Ready.tokens/model-token
   counters to consumed T rather than one. StateOnly model-cohort/readout-phase
   counts, API/readback calls, published roots, teacher frontiers, dispatch and
   fairness metrics differ and must stay separate. Existing worker/HTTP assumptions
   of one teacher per wave must get new qualified tests, never relaxed assertions.
   Reserve final prompt tail and original packet/presentation ownership barriers.
9. A separate actual CUPTI diagnostic can count real kernels/readbacks for this
   source/binary vs accepted per-teacher native path; record all input/profile and
   physical memory/process limits. Source receipt is not driver measurement.
   Unprofiled repeated same-engine A/B then matched genuine vLLM comparison remain
   separate. This proposal repeats every B-shaped original projection T times;
   no B*T stateless grouping or solo B1 acceleration is claimed.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-chunk-20261020/TODO.md

Original UTF-8 bytes: 1119; SHA256 `125bf8048981b671fdf140acedeab09f7431ca23107c36bb35d51153698d92bc`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native teacher chunk private run

- Parent: existing serving-and-speculation P12 plan; coordinator owns live plan.
- Owned: ignored proposal only, six GPU source paths +tiny qualifier wiring.
- Prepared: sealedB2/4/8,T2/4 metadata; one gather, original StateOnly bodyT,
  full intermediate certificates/persistentOR, private ping-pong/final scatter,
  one original Ready/readback/publisher and truthful T counts. No CLI change.
- Prepared:2 logical tests +22 optional ignored physical cases (12normal,
 2pressure,4portable boundary,2status refusal,2unknown completion).
- Static only: Rustfmt with skip_children=true and small source freeze/diff.
- Not run: Cargo/rustc/build/tests/operators/providers/models/containers/GPU.
- ROOT pending: independent review +all ROOT-GATES; no inference/performance or
  production admission claims. Numerical/unknown negatives each fresh process.
- Later: separately qualify scheduler equal-real-T/fairness/tail/credit/cancel
  accounting; then assess same-plan B*T stateless temporal grouping. Current
  sequentialT proposal intentionally leaves dot/kernel body count repeatedT.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-chunk-20261020/frozen/README.md

Original UTF-8 bytes: 7188; SHA256 `cac8180c3c68a5f82e483a3aab0dfee34b3c3add8f41ed5e21f5fb21ab7af7f3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed native teacher chunk T2/T4

Author: baseline_review. Parent: existing serving-and-speculation P12 plan.
Source-only proposal. ROOT owns MAIN, merge, Cargo/runtime/GPU, qualification and
shared plan reconciliation. No build/test/operator/provider/model/GPU execution.

## Exact change

- Existing BatchProgram, unchanged original ForwardPlan state prefix and original
  kernels, sequentially T times at B2/4/8. This is AR1 order for each row. There
  is NO B*T projection/MLP grouping, temporal BMM, new model schedule, engine,
  trained-HF arithmetic, CPU fallback or graph replay in this proposal.
- Qualification-only `ResidentModel::prepare_state_fused_chunk_admitted` accepts
  exact equal per-row T2/T4 teacher slices and immutable original root bindings.
  All IDs, source, model, original capacity <=512, exact independent frontiers,
  full span and first-token correspondence are checked. Ragged teacher lengths,
  duplicate/foreign/stale rows, device feedback/readout mixing and graph pins
  refuse. None remains only original physical capacity refusal before launch.
- Reuse the ALREADY admitted two packed state banks. Gather original independent
  roots once. Run the same state-only node body at each row's position+step.
  Original node certificates and embedding/quant status merges already OR into
  one status; clear it only once. Before each bank swap, certify ALL48 packed
  candidate fields, including full KV capacity, using the original finite U32
  predicate. Original1,000,000-word certificate chunks safely cover B8*262144
  state words without unchecked reuse of a one-million-word limit.
- Swap only private packed-bank metadata; RAII restores original orientation on
  every normal error/unwind. Final even-T state is in the originally prior bank,
  so a tiny existing transport preflight factor checks THAT exact final scatter
  bank before submission. Default swap_banks=false is byte/order-equivalent to
  accepted single-step lowering. Final certified scatter runs once into genuine
  independent candidate BufferLeases. No extra GPU state bank/output/sampler or
  warm source/state/activation upload is added.
- Original allocation->queue/workspace order, same ResidentService/client,
  original slot/flight/access leases, readback validator, Ready and atomic all-row
  publisher remain the owners. One original authority.begin_advance(T) per row;
  one terminal4B status readback/Ready; one root publication advances frontierT.
  T is honestly reported in state_only_steps. Physical dispatch count includes
  every body, parameter staging, intermediate certificate and gather/scatter.
  Native projection subset is summed across T. No graph enqueue/kernel1 claim.
- Unknown completion/status rejection retains original conservative fence and
  quarantine. Known-Ready caller cancellation is drop of ALL private candidates;
  a stale row refuses ALL publication. No recovery, output packet, generated ID,
  EOS handling, SSE, usage or Done exists in teacher chunk preparation.

## Source ownership and composition

`BASELINE.json` retains exact original source bytes, including accepted prefix,
RoPE, graph and certified-scatter changes. `NEW_DELTA.patch` owns six source paths:

- resident_inference/fused.rs: closed execution variant/entry, T authority/span
  and truthful readback counts; normal None/Greedy/Logits and graph paths kept.
- resident_batch.rs plus new resident_batch/chunk.rs: sealed bounded teacher
  metadata and two source-free authority/range tests.
- resident_program/batch.rs plus new resident_program/batch/chunk.rs: tiny
  default-preserving bank selector/module wiring and the existing-body T loop.
- One additional new qualification child under
  resident_inference/prefix_qualification/chunk.rs.

The list above is six source paths total: three existing parents/three new
children. ROOT alone applies the separate tiny `ROOT_WIRING.patch` mounting that
qualifier beneath existing prefix qualification (a seventh combined path). No lib.rs/CLI/HTTP/tools/plans/default feature/profile changes.
Batch's ordered-SiLU candidate edits launch_node and can be composed separately;
this proposal keeps that baseline math unchanged. Apply only reviewed hunks with
fresh modification times; never overwrite a newer root parent with old bytes.

## Prepared acceptance and limits

- Twelve ignored full-model cases: both original .8B/2B, B2/4/8, T2/4. Genuine
  distinct nonzero GPU/CPU histories and unequal frontiers, source tokenizer EOS
  as teacher, independent accepted CPU+B1 complete vocabulary/all42 AFTER EACH
  reference teacher, exact native all42 at final scope. Then real future native
  full248320 logits and accepted CPU/B1/native greedy packet ID/max/all42 parity.
- Exact4B total terminal readback, T steps/row, zero sampler/logit/packet during
  prefix, one root advance, source/model uploads unchanged, pending0/final0.
  Bad last teacher/ragged/duplicate rows cold-refuse; actual Ready abandonment,
  stale restore and all-sibling rollback preserve full prior state/leases.
- Two physical pressure cases: actual original Scratch lease prevents B8 before
  any queue work; explicit B4 scopes execute; B8 executes after release; no
  hidden fallback, full CPU/B1/all42/future fullV and final0.
- Four optional portable-checkpoint boundary cases (B2,T2/4,both models): genuine
  CPU-derived prefix states imported through original authority, cross256 and
  reach512; every reference teacher fullV/all42 and final nativeall42, capacity
  refusal unchanged. Scope is CPU-hydrated GPU decode, never GPU full prefill.
- Two fresh-process actual numerical-status negatives. A cfg(test) one-shot
  poisons ONLY an intermediate inactive packed KV word with NaN; a later original
  append heals final padding to positive zero. Original full-word intermediate
  certificate must latch status2 across later bodies. Direct raw original-handle
  readbacks prove status2/final padding0; no Ready/publication. Prior all42 raw
  bits exact, original conservative quarantine/fence/envelope retained. Synthetic
  fault is not an expected trained result and never exists in non-test builds.
- Two hardware-abort-feature fresh-process negatives: actual whole chunk queued,
  future dropped before terminal Ready, all original handles quarantined, every
  prior root/all42 bits unchanged and service/provider fence retained. This is
  omitted-terminal-proof safety, not a fabricated driver fault/recovery.

Still UNQUALIFIED: all prepared code compilation/runtime, actual driver counts,
latency/throughput, production scheduler use, aggregate host/control allocation
bounds, global graph capture, trained HF/FLA/27B and other families. The worker
still consumes its accepted one-teacher quantum. A future scheduler cut must
partition prefix vs final tail BEFORE bands, choose only equal real remaining
T2/T4 with each remaining prefix>=T, cap fair quantum<=4, and preserve cancellation,
parked/credit ownership and singleton truthfulness. Ragged requests must regroup
without zero/EOS padding; an actual stopped/final-tail row cannot enter this chunk.
No performance claim or production enablement follows from a source-only freeze.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-chunk-20261020/frozen/ROOT-GATES.md

Original UTF-8 bytes: 3522; SHA256 `e85d41e77b3b2242d5239c550fca5b2a2c874612c6870116db9993be40f03cb3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT qualification cuts

No author execution. Source-only candidate remains closed behind qualification
+cuda. Preserve every failed attempt. Existing original source/logit/state limits
and tolerance0 remain; do not substitute analytical/generated expected values.

1. Independent complete source review; compile default/non-CUDA +strict CUDA,
   Vulkan and existing legacy/HF cold guards. Check bank selector defaultfalse,
   original state-only body and lease/publisher tail against BASELINE exact bytes.
   Logical tests in `resident_batch::chunk::tests` are source-free, not CUDA proof.
2. Serial release model cases. New namespace:
   `resident_inference::fused_qualification::prefix_qualification::chunk::`.
   Run only twelve `qwen{08,2}_b{2,4,8}_t{2,4}` tests at first, using anchored
   exact filters or ROOT's test-list proof. Preserve sources/binary and actual
   counts. Existing original approved sources only; no new downloads/profile.
3. Run two exact pressure tests (`qwen08_chunk_pressure`, `qwen2_chunk_pressure`).
   These require prelaunch None/unchanged queue+root+resident bytes under genuine
   held lease, explicit B4 and B8 after release, no hidden retry.
4. With portable-checkpoint, four exact boundary tests
   `qwen{08,2}_chunk_boundary_t{2,4}`. Real independently computed CPU prefixes
   cross padded attention256 and end512. Imported actual state is labeled as such.
5. EACH numerical status refusal in a FRESH process:
   `qwen08_chunk_certificate_refusal`, `qwen2_chunk_certificate_refusal`.
   Actual original status validation intentionally fences the card/service and
   retains provider envelope; do NOT append a successful model test or recovery.
   Synthetic inactive-KV transient NaN must be caught even though final physical
   padding is positive zero. There is no status clear between teachers.
6. EACH hardware-abort test in a FRESH process, hardware-abort-probe enabled:
   `qwen08_chunk_abort`, `qwen2_chunk_abort`. Require actual queued original
   handles and omitted Ready, complete quarantine/retained caps/unchanged root
   and raw prior all42; no retry/device reset/privilege change.
7. Existing unchanged B1/native fullV/greedy/all42/unequal-frontier/mixed256/512,
   opaque+typed scatter canaries/status/offset/alias/refusal, graph owner/profile,
   original completion/quarantine, worker/HTTP/cancel/credit and approved original
   cache-direct/HF-import release routes must remain exact. Run full isolated:

   `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`

8. Before production scheduler integration, bind true Ready.tokens/model-token
   counters to consumed T rather than one. StateOnly model-cohort/readout-phase
   counts, API/readback calls, published roots, teacher frontiers, dispatch and
   fairness metrics differ and must stay separate. Existing worker/HTTP assumptions
   of one teacher per wave must get new qualified tests, never relaxed assertions.
   Reserve final prompt tail and original packet/presentation ownership barriers.
9. A separate actual CUPTI diagnostic can count real kernels/readbacks for this
   source/binary vs accepted per-teacher native path; record all input/profile and
   physical memory/process limits. Source receipt is not driver measurement.
   Unprofiled repeated same-engine A/B then matched genuine vLLM comparison remain
   separate. This proposal repeats every B-shaped original projection T times;
   no B*T stateless grouping or solo B1 acceleration is claimed.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-chunk-20261020/frozen/TODO.md

Original UTF-8 bytes: 1119; SHA256 `125bf8048981b671fdf140acedeab09f7431ca23107c36bb35d51153698d92bc`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native teacher chunk private run

- Parent: existing serving-and-speculation P12 plan; coordinator owns live plan.
- Owned: ignored proposal only, six GPU source paths +tiny qualifier wiring.
- Prepared: sealedB2/4/8,T2/4 metadata; one gather, original StateOnly bodyT,
  full intermediate certificates/persistentOR, private ping-pong/final scatter,
  one original Ready/readback/publisher and truthful T counts. No CLI change.
- Prepared:2 logical tests +22 optional ignored physical cases (12normal,
 2pressure,4portable boundary,2status refusal,2unknown completion).
- Static only: Rustfmt with skip_children=true and small source freeze/diff.
- Not run: Cargo/rustc/build/tests/operators/providers/models/containers/GPU.
- ROOT pending: independent review +all ROOT-GATES; no inference/performance or
  production admission claims. Numerical/unknown negatives each fresh process.
- Later: separately qualify scheduler equal-real-T/fairness/tail/credit/cancel
  accounting; then assess same-plan B*T stateless temporal grouping. Current
  sequentialT proposal intentionally leaves dot/kernel body count repeatedT.

````

## Source: .agents/runs/serving-and-speculation-native-prefix-chunk-20261020/temporal-grouping-design/DESIGN.md

Original UTF-8 bytes: 7986; SHA256 `9a3d1ce7532d70d4cfaff50b0b2d9616db0cb2eed4df509e04b15f00a9f6d9eb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Next qualification cut: true T2 stateless grouping

Author: baseline_review. Source-only feasibility; no code/runtime eligibility.
Parent cc3a62c9 sequential-T proposal remains immutable and distinct.

## What sequential T does and does not solve

The frozen proposal removes repeated gather/scatter/readback/publication while
running every original B-shaped projection/body T times. It does not reduce
those dot/activation/MLP traversals to one B*T traversal and does not change B1
serving. ROOT's actual solo-prefill bottleneck must not be declared solved by it.

The smallest next full-model scope is B2,T2, physical stateless rows4; original
native B4 kernels already exist. Represent physical activation rows explicitly
as `[teacher0.request0,teacher0.request1,teacher1.request0,teacher1.request1]`.
Only TWO genuine sessions/RootAttempts/Ready rows exist. The four activation rows
are not four sessions and must NEVER be passed to ResidentBatchRows or publisher
as invented/duplicated RootBindings. Each original authority advances2 once.

## Concrete existing seams and blockers

- `ForwardPlan::nodes/values`, PlanNode SSA inputs/outputs and StateEffect are the
  same immutable model program. State reads also live INSIDE KvPrefixGuard,
  AttentionQk, AttentionValue and ObserveState operations; `node.state.is_none()`
  is NOT a sufficient stateless classifier. Unknown/new typed ops cold-refuse.
- BatchProgram fixes rows throughout values/scratch/scaled node geometry and
  PreparedBlocks input bindings. Merely changing rows for selected launch_node
  calls is invalid. Existing native B2 and B4 programs can initially supply the
  two exact qualified layouts, under ONE original service/flight/workspace lock.
- ResidentProgram owns value buffers and prebound quantizer inputs. A raw alias
  replacement invalidates prepared bindings/OwnedRanges. Resident inference's
  lease_for requires exact original handle/range equality; broadening it to
  silently accept arbitrary subviews is unsafe. First slice should use explicit
  original U32 copies between B4 teacher slices and existing B2 IO buffers; keep
  original full leases and checked offsets. Optimize copies only after proof.
- B2 packed KV is `[token,request,head,width]`; a B4 temporal activation slice is
  contiguous, but a hypothetical `[token,teacher,request,...]` KV teacher slice
  is STRIDED. Reinterpreting a flat B4 KV range as a dense B2 cache is incorrect.
  Retain the genuine B2 state banks and original per-teacher QK/AV kernels.
- A read of teacher0 must use teacher0's actual KV snapshot, not final teacher1
  state under a mask. Raw QK future columns and AV padded zero-product/sign bits
  make the final-state shortcut numerically unproved. Preserve exact snapshot
  version selection even where probabilities are causal zero.
- Current Workspace holds one BatchWorkspace band. It cannot keep B2 state and
  B4 stateless programs simultaneously. A sealed temporal workspace needs both
  programs/leases/cold budgets and a joint access list before allocation/launch.
  Do not evade this by a second Client/service/queue/model/source.

## Bounded source-derived execution

1. Keep original ForwardPlan immutable. Build a SMALL closed lowering descriptor
   keyed by source, plan/value/state geometry, B2,T2, capacity512, GGUF/F32/readout.
   Expand only dependency metadata `(teacher,node/value/state-version)`, not a
   new Qwen layer/token engine. There is no trained-HF profile admission.
2. Prepare original B2 and B4 lowerings with the same original source handles.
   Pre-admit exact sum of both owned_bytes plus independent B2 final candidates,
   tokens/positions, any explicit copy scratch and original queue ticket. The
   unoptimized B4 packed state/greedy buffers are unused but STILL charged; do
   not claim free/subset workspace yet. Graph pins refuse, all typed ops refuse.
3. Factor existing launch_body's ONE node+original certifications into an internal
   checked-node helper; its existing full-body loop remains byte/order-equivalent.
   The temporal lowerer invokes that same helper, never rewrites dot/nonlinear
   math. Existing quantizer bindings remain attached to their genuine IO buffers.
4. For nodes whose inputs for BOTH teachers are ready and whose closed contract
   is stateless, run once as B4. Norms/projections/views/elementwise/MLP are prime
   candidates. Original embedding uses all4 true teacher IDs and per-row RoPE
   uses all4 true positions. Capture/observations/checks retain exact producer IDs.
5. Conv and recurrent effects run original B2 kernel at teacher0 then teacher1,
   with per-DOMAIN bank/version mapping, independent from other domains. Each
   candidate certificate/status merges BEFORE reuse. Copy resulting IO values
   into their B4 slots; later stateless consumers may regroup once both are ready.
6. Attention state read dependencies are explicit. A conservative first lowering
   keeps KV guard/append/QK/softmax/AV as one causal B2 region per teacher, with
   B4 projected/normed/rotated inputs and B4 output projection/MLP outside. This
   avoids storing extra snapshots or relying on masks to repair wrong operands.
   A finer DAG can later keep distinct T0/T1 snapshots for per-node grouping.
7. Each program's original status is cleared only once and OR-accumulates. Merge
   their two original statuses to one admitted4B transaction status before final
   original readback. No per-node/per-teacher host sync, error reset or fallback.
   All relevant state/value finite certificates remain before overwrite/reuse.
8. Scatter only final teacher1 states into original B2 candidates. Same original
   completion/Ready/atomic publisher; model_tokens4, independent rows/roots2,
   teacher steps2 per row. Physical projection rows4 is a KERNEL geometry, not a
   request/cohort-width counter. Final tail remains outside StateOnly scheduling.

This first proof admits explicit copy overhead. Source dot invocations for a
successfully grouped same-node pair become one B4 invocation instead of two B2;
no measured latency, driver-count or throughput reduction follows automatically.

## First production-facing extensions, separately qualified

- B1,T2 could use original B1 resident state kernels plus existing B2 stateless
  kernels under the SAME ownership scheme. This is the direct solo-prefill
  direction; BatchProgram(rows=1) is currently unsupported, so do not invent it.
  Reuse the original ResidentProgram B1 emitter/owner and original advance(T).
- B4,T2 ->stateless8 is an existing physical band, but needs its own complete
  geometry/state/lease/CPU+B1 proof. B8,T2 ->16 is unsupported. Tile into accepted
  groups with real offsets/ownership or cold-refuse; never pretend B16 exists.
- No T4 stateless grouping inferred from T2. Two banks cannot preserve arbitrary
  earlier snapshots if later teachers overwrite them before readonly consumers.
  Either use causal regions/tiledT2 or admit additional genuine snapshot storage.

## Required actual evidence

Source-free classifier/DAG tests must include hidden state operands, stale plans,
unsupported ops, teacher-tail boundaries, valid nonzero slices, offset/alias and
budget refusals. Then real .8B B2,T2 followed by2B: independent CPU/B1 fullV/all42
for each teacher, native private intermediate42 where captured and final42,
full future decode/greedy parity; exact original stage captures if divergence.
Mixed frontiers through256/512, real teacher EOS, genuine pool pressure, abandoned
Ready/stale/foreign/cancel and fresh unknown/status failures; all leases final0
on known completion. Same source/profile, original full gates and both routes.
Only then worker/HTTP fairness/credit/tail and honest model-token/row/root/kernel
counter integration. Actual CUPTI and unprofiled repeated performance follow,
with matched vLLM numerical-profile disposition unchanged until independently
qualified. Source inspection is not a benchmark or model-eligibility receipt.

````

## Source: .agents/runs/serving-and-speculation-native-solo-temporal-20261020/EMITTER-FACTOR.md

Original UTF-8 bytes: 767; SHA256 `8e792b81d34ebfb45d19f569d74905bb795fe78afaf596f5353af4e7b58e3ac8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original P1 factor source check

Author: batch_design_review. Static source IO only, not compiled/runtime proof.

- Original loop body vs new helper: equal after whitespace removal and sole
  control mapping `continue` (outer loop) → `return Ok(dispatches)` (helper).
- Original `fn launch_node` through end of ResidentProgram module is BYTE-EXACT.
  No original GPU node math, source quantizer, copy/certify helper or reduction
  implementation changes.
- Original outer loop visits unchanged output-prefix indices in original order.
  Final logit/proposal certificates and receipt body remain outside new helper.
- New helper adds only index lookup, local source-launch count and return.
- ROOT strict/all ordinary physical source/route gates remain mandatory.

````

## Source: .agents/runs/serving-and-speculation-native-solo-temporal-20261020/README.md

Original UTF-8 bytes: 7558; SHA256 `7683b6c4e64db27223981d18ef70184f11135784c21b1cbb4dc007d87f7c0dc3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed direct SOLO B1/T2 temporal grouping

Author: batch_design_review. Parent: serving-and-speculation P12.
SOURCE ONLY / UNQUALIFIED. No author tests, compilation, imports, child/provider/
model/GPU/container execution, binary/model/DSO payload reads/hashes or timings.

- Explicit excluded source foundations: sequential cc3a62c9019c5426656b9d27485bf1bee2b4ee853699970d6e0e6233e2000f0a,
  B2/T2 temporal6be63adfe19e649d60a5767b2066cc43ce9798d99b28a37a57fcc3687086cadf.
  Their physical qualification remains ROOT prerequisite, never inferred here.
  BASELINE.json identifies four exact parent bytes. Apply their reviewed deltas
  and required wiring first, then this proposal's narrow hunks with fresh mtimes.
  Never overwrite current MAIN parents wholesale or alter old frozen bundles.
- Original one Rust ForwardPlan/ResidentProgram, mathematical kernels, source
  identities/owners, TaskCoordinator, ResidentService/physical queue/actor,
  ResourceArbiter, ledger/Ready/quarantine and SessionAuthority remain owners.
  No source profile/model eligibility, CLI/HTTP, metrics, graph or default route
  is enabled. API exists only with qualification+CUDA and original pinned queue.

## Proposed source

- One actual session, two actual teachers, cap512. P2 activation rows are
  `[T0,T1]`, not two requests. No BatchProgram1, invented session, reconstructed
  device token, checkpoint/future root or second engine/queue is constructed.
- Original ResidentProgram ordered node loop delegates to a private one-node
  helper containing the same emitter and original output certificates. Original
  normal node order, View skip, Contiguous certificate, causal score predicate,
  final logit/proposal certificates and counts remain unchanged. A source-only
  normalized-body comparison is recorded; ROOT strict/physical routes required.
- Existing6be Dependencies/Stage are shared through narrow sibling visibility.
  Same original P1 plan-lineage/source Arc is retained in P2 cold lowering.
  Stateless nodes use existing genuine native B2 kernels once across teachers.
  Complete attention guard→append→QK→softmax→AV and GDN effects use ORIGINAL P1
  node emitter per teacher. No copied model/layer/token schedule or reducer.
- Explicit checked U32 copies bridge P1 values with exact T0/T1 P2 offsets.
  Source tensors, strides, prebound quantizer handles and original OwnedRanges
  are never rebound. Original dense token/head/dim B1 KV needs no repack.
- State path is committed prior→private T0 middle→private final candidate, per
  domain. All42 middle domains are persistent workspace scratch leases; all42
  final domains are fresh independently publishable leases. Middle is never a
  StateRoot or inference Ready output. Initial empty middle bytes are never
  consumed before sealed causal producers and full candidate certificates.
- Full intermediate state, including inactive capacity KV words, is certified
  before teacher1 consumption. P1 and P2 statuses each clear once; original
  merge_status OR plus original final certificates feed one terminal4B readback.
  One original authority begin_advance2 / real Ready / Session.commit advances
  frontier2 and root1. EOS is only a teacher input; no output packet/token/usage.

## Physical ownership/admission

- Original P1 storage remains entirely charged by original model load. Before
  own allocation or queue ticket, the original allocation lock checks all new
  P2 owned bytes (including unused private P2 state/greedy storage), every middle
  byte and final candidate byte against actual remaining capacity. P2 source
  bytes0 share original weights. Queue admits the complete two-state Scratch
  guard through original prepaid fair tickets, not scheduler byte estimates.
- P2 allocation handles/bytes and middle handle count/bytes reconcile exact
  preflight. Cold initialization uses original begun slot/initialization guard,
  memory cleanup, actual terminal sync; finishes/requeues before mathematical
  slot exactly like existing native cold paths. Existing pinned batch graphs
  refuse before allocation; no production capture exclusion is claimed.
- Both steps' all42 shapes, exact producer→consumer middle bindings, source/
  service/lineage, full nonaliasing prior/middle/final/P1/P2 spans and true bridge
  bounds validate before original mathematical submission. P1 original ABI
  word bound<=1M is closed explicitly; no relaxed kernel guard/scalar fallback.
- Original flight retains every accessed P1/P2/source/middle/final lease before
  any enqueue. Final Ready promotes only final candidates. Known Ready abandon
  or stale logical publication leaves committed roots untouched. Errors/unknown
  completion retain original service envelopes/handles/fences in quarantine.
- T0 inspection is a separately admitted original read-only flight AFTER real
  final Ready. Its actual middle leases and transfer ticket remain held through
  raw processing. It creates no root/event promotion and is outside terminal4B.
  T0 inspection is diagnostic and must precede any subsequent SOLO overwrite.
- Warm path creates no GPU storage or activation/model/state upload except the
  original fresh final private state allocation; it retains source-bound data
  on device. This is not a whole-host RAM admission claim for bounded metadata
  Vecs or CPU numerical oracle storage.

## Prepared ROOT qualification

Ten ignored tests, no author execution:

- Both sources: nonzero genuine B1 seed; independent CPU+B1 full248320 and all42
  after EACH teacher; actual source EOS as teacher1. Real native T0 all42 and
  canonical future KV zeros through private read-only flight, then committed T1
  all42. Later original B1 fullV and real CPU/B1/native greedy packet exactness.
- Exact P1 unchanged budget, all P2/middle bytes and ledger delta, every native
  projection source launch site, one4B transaction, one root/advance2, warm value
  uploads0, known-completion final provider/global reservation0.
- Original Ready abandon and stale epoch/refused commit preserve roots/all42;
  invalid length/token metadata refuses before original queue work.
- Genuine physical combined P2+middle+final capacity shortage returns None before
  queue/allocation; release then exact execution, no fallback or hidden budget.
- Actual CPU-hydrated original states at255/510 with original checkpoint imports;
  T2 crosses256/to512. First teacher native all42, each CPU/B1 complete reference,
  exhausted capacity prelaunch refusal. No full GPU prefill claim for hydration.
- Separate fresh-process unknown completion at original after-launch hook:
  actual accessed P1/P2/middle/final/source handles retained, card/service/provider
  fenced, other original service cannot recover, committed all42 raw unchanged.
- Separate fresh-process synthetic inactive middle KV NaN after T0, before its
  original full certificate. Actual T1 final tail +0 and latched status2 prove
  the intermediate decision survives overwrite. Roots/all42 priors unchanged,
  quarantine/fence persist. Test-only final lease retained for explicit raw
  post-refusal observation; no Ready/publication/recovery or trained NaN claim.

ROOT must run strict/default/CUDA/Vulkan gates, parent and new actual source
matrices, full isolated check and both original release routes. Source launch
counts are not driver-enumerated GPU work. Only later actual collector and
unprofiled controlled comparisons can establish latency impact. No throughput,
vLLM parity, full HF, production graph or serving qualification follows here.

````

## Source: .agents/runs/serving-and-speculation-native-solo-temporal-20261020/ROOT-GATES.md

Original UTF-8 bytes: 2162; SHA256 `9c3861626914ac33381371c76bf817336d349055e1f98bf275f7ace697459a06`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only SOLO gates

Author: batch_design_review. Author executed only source IO/rustfmt; no build/math.

1. Review/qualify cc3 sequential +6be temporal foundations first. Compose narrow
   seven source paths + ROOT_WIRING against current MAIN with fresh mtimes;
   preserve accepted typed/refusal/RoPE/row transport/queue and other agents' work.
2. Strict CUDA+qualification+portable-checkpoint+hardware-abort-probe, ordinary
   strict/default and Vulkan; original P1 body factor requires all accepted
   ordinary B1 source/physical routes unchanged. No formatter/type inference
   substitutes compilation. No author numerical/physical test receipt exists.
3. ROOT release ten ignored tests below; parent prefix qualification child path:
   `resident_inference::fused_qualification::prefix_qualification::chunk::solo`.
   Exact test names `qwen08_solo_b1_t2`, `qwen2_solo_b1_t2`,
   `qwen08_solo_pressure`, `qwen2_solo_pressure`,
   `qwen08_solo_boundary`, `qwen2_solo_boundary` (portable-checkpoint).
   Actual all42 domains/full248320 compare bits, no tolerance widening.
4. EACH `qwen08_solo_abort`, `qwen2_solo_abort` (hardware-abort-probe),
   `qwen08_solo_certificate`, `qwen2_solo_certificate` needs a fresh process;
   actual fences persist. Use --release --ignored --exact --test-threads=1.
   Diagnostic post-refusal raw reads never issue Ready or recovery.
5. Run complete isolated scripts/check.sh under original native-gpu wrapper,
   both approved original cache-direct/HF-import release parity routes and
   required worker/HTTP regressions before phase acceptance. No serving callsite
   is added; single-session API remains qualification-only/pinned queue.
6. Before any serving integration: explicit prefill-only T2 eligibility reserving
   actual final prompt tail, bounded fairness/cancellation, original counters'
   truthful tokens2/oneRoot/oneRequest, and no teacher Text/ID/usage/SSE. Existing
   GPU packets/B1 fallback must remain independent. This proposal enables none.
7. Later genuine hardware collector/unprofiled before-after only after numerical
   and physical acceptance; do not compare this source-only prototype as vLLM.

````

## Source: .agents/runs/serving-and-speculation-native-solo-temporal-20261020/TODO.md

Original UTF-8 bytes: 678; SHA256 `d56158a7d01217a4e9ac0d0c87af7415437cc3196c0610d652e7e6ab8cb6df54`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12 SOLO B1/T2 temporal proposal

Parent: `.agents/plans/serving-and-speculation/TODO.md`; owner batch_design_review.

- [x] Inspect original P1 emitter/physical single-session owner + frozen cc3/6be.
- [x] Report proposed shared emitter/state/lease/cold admission seams to ROOT.
- [x] Implement source-only P1 emitter factor + B2 stateless child + solo owner.
- [x] Prepare independent CPU/B1/all42/fullV and ownership/cancellation gates.
- [x] Source-only self-review/rustfmt/freeze; original emitter source-equivalence checked.
- [ ] ROOT/peer independent review, strict compilation and all actual physical gates.
- [x] No default/CLI/graph/profile/performance enablement.

````

## Source: .agents/runs/serving-and-speculation-native-solo-temporal-20261020/followup-v2/README.md

Original UTF-8 bytes: 1058; SHA256 `484133539030a017ea06dc8c040f44979f9226b59772e57b50bd5e81e6c08221`.
Resolve embedded relative paths against the original path, not this volume.

````text
# SOLO qualifier wiring packaging correction

Author: batch_design_review. Parent: serving-and-speculation.
Source-only additive repair; preserve original4f2a9d5f freeze/all source/artifacts.

- Original ROOT_WIRING.patch was malformed: appended multiline text was one
  difflib list element, leaving module line without '+' and wrong new hunk count.
- Use ONLY this CORRECT_ROOT_WIRING.patch in place of original wiring artifact.
  Context675..677 is exact parent6be chunk.rs EOF: closing brace/blank/mod temporal.
  Correct675,3→675,5 hunk adds separate '+' blank and '+mod solo;' lines.
- Exact intended resulting module source is original677 lines plus blank and
  mod solo; (679 lines). No production/qualifier Rust implementation changes.
- Original7 proposed/frozen sources and7 artifacts rechecked exact; old freeze
  and malformed patch remain immutable evidence. Source generation/count/context
  checks only; no patch application, tests, compilation or runtime by author.
- ROOT/peer review and actual compilation/physical gates remain pending.

````

## Source: .agents/runs/serving-and-speculation-native-solo-temporal-20261020/followup-v2/frozen/README.md

Original UTF-8 bytes: 1058; SHA256 `484133539030a017ea06dc8c040f44979f9226b59772e57b50bd5e81e6c08221`.
Resolve embedded relative paths against the original path, not this volume.

````text
# SOLO qualifier wiring packaging correction

Author: batch_design_review. Parent: serving-and-speculation.
Source-only additive repair; preserve original4f2a9d5f freeze/all source/artifacts.

- Original ROOT_WIRING.patch was malformed: appended multiline text was one
  difflib list element, leaving module line without '+' and wrong new hunk count.
- Use ONLY this CORRECT_ROOT_WIRING.patch in place of original wiring artifact.
  Context675..677 is exact parent6be chunk.rs EOF: closing brace/blank/mod temporal.
  Correct675,3→675,5 hunk adds separate '+' blank and '+mod solo;' lines.
- Exact intended resulting module source is original677 lines plus blank and
  mod solo; (679 lines). No production/qualifier Rust implementation changes.
- Original7 proposed/frozen sources and7 artifacts rechecked exact; old freeze
  and malformed patch remain immutable evidence. Source generation/count/context
  checks only; no patch application, tests, compilation or runtime by author.
- ROOT/peer review and actual compilation/physical gates remain pending.

````

## Source: .agents/runs/serving-and-speculation-native-temporal-grouping-20261020/README.md

Original UTF-8 bytes: 7611; SHA256 `f7b0c449af62b382d0ccf7b88c2afac854077ecdb4795d58aa2b2574013d0615`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed native temporal grouping

Author: baseline_review. Parent: serving-and-speculation P12.
Status: SOURCE_ONLY / UNQUALIFIED; no build or mathematical/device execution.

## Dependencies and application

- Immutable sequential-T foundation: `../serving-and-speculation-native-prefix-chunk-20261020/`
  freeze `cc3a62c9019c5426656b9d27485bf1bee2b4ee853699970d6e0e6233e2000f0a`.
- `BASELINE.json` preserves that proposal's exact source plus the original
  resident owner/program. It is an excluded baseline, not another implementation.
- ROOT applies reviewed cc3 + its qualifier wiring first, retaining current
  cold-prep/RoPE/certified transport/typed refusal changes. Then apply this new
  delta; parent hunks need narrow manual composition if MAIN moved. Copy bytes
  with fresh mtimes. Do not copy whole old parents over current MAIN.
- `PRODUCTION.review.patch` / `production-manifest.json` / `production-frozen/`
  are an immutable six-path review snapshot. The complete freeze adds the
  qualifier child and a separate one-line `ROOT_WIRING.patch`.
- No CLI route, scheduler, health field, public metrics, model eligibility or
  default serving behavior is enabled.

## Actual proposed source

- Closed B2 requests, T2 teacher tokens each, cap512; original pinned service.
  Four physical activation rows `[T0R0,T0R1,T1R0,T1R1]`, TWO authorities/roots.
  Independent `begin_advance(2)`, one final original all-row publisher.
- A cfg-only opaque lineage is issued by original ResidentProgram and cloned
  with its immutable plan into existing B2/B4 lowerings. SAME source Arc/lineage,
  original quantizer bindings, exact handles and leased ranges are retained.
- Dependencies derive original PlanNode SSA inputs/outputs and effects. Hidden
  KV reads in guard/QK/AV/ObserveState are explicit. Unknown typed ops refuse.
  Full attention guard→append→QK→softmax→AV stays causal B2 per teacher; the
  actual original StateOnly cut may end just after its last append.
- Existing `launch_body` factors only node+original output certifications; its
  normal node order/count/math stays unchanged. Stateless nodes call that same
  emitter once on B4; causal regions call B2 for T0 then T1. No model/token loop,
  transposed weight alias, reordered row reduction, fake B4 sessions or B1 retry.
- Explicit original U32 copies move causal IO between exact B2 and B4 buffers.
  T1 uses a checked nonzero activation offset. Neither prepared quantizer inputs
  nor OwnedRanges are rebound. Genuine B2 KV remains `[token,request,...]`;
  no temporal flat-KV reinterpretation or final-state causal-mask shortcut.
- Per-domain RAII bank flips preserve T0 in physical B and T1 in A; every state
  candidate is certified before later teacher consumption/reuse. Normal original
  value/candidate certificates retain their exact predicate and OR semantics.
- B2 and B4 status clear once each. Original merge_status ORs P4 into P2 before
  one final 4B transaction readback. Original final certified scatter, Ready,
  per-row physical candidates, cancellation/unknown quarantine and atomic root
  checks stay in the existing owner.
- Both COMPLETE original B2/B4 preflight budgets are checked before candidates
  or workspace allocate. Unused B4 state/greedy buffers are charged. Actual
  allocations/owned handles are reconciled. Band reclamation counts the sum.
  Same original cold initialization guard, fair queue and one mathematical
  physical flight; the existing cold initialization terminal sync/requeue remains.
- Warm operation does not allocate/upload activations/state/model bytes. Host
  bounded plan/receipt Vec/Arc metadata follows existing control-plane ownership;
  this is not a whole-host physical RAM admission claim.

## Prepared evidence; nothing executed by author

Four source-free descriptor fixtures: hidden-state reads/SSA producer refusal,
repeated or overwritten prior version refusal, separate temporal/state axes,
whole attention region plus clipped original cut.

Ten ignored ROOT-only physical cases (two sources each):

- Normal .8B then2B: distinct nonzero unequal histories, actual tokenizer EOS as
  second teacher, CPU and accepted B1 full248320/all42 after each teacher;
  native T0 packed all42 (including canonical unused KV) and T1 committed all42;
  future complete vocabulary and real B1/native greedy packet parity.
- Abandoned and stale Ready leave every root/domain unchanged. Invalid token,
  ragged length, T4 and duplicate-session metadata refuse before kernel work.
- Real combined P2+P4 budget pressure returns None before queue/allocation;
  release then exact execution, no fallback. Warm data-upload count unchanged,
  all known-completion leases/provider reservations final0.
- Optional portable-checkpoint .8B/2B: independent CPU-hydrated actual states at
  255/256 and510, actual T2 across padding256 and to512, then exhausted refusal.
  This labels hydrated source-real decode, not full GPU prefill of those prefixes.
- Fresh-process hardware abort retains both workspace/source/candidate handles,
  fences original service/provider and forbids commit/recovery. Read actual
  original all42 prior bytes only as an explicit diagnostic after quarantine.
- Fresh-process synthetic inactive KV NaN after T0, before original certificate:
  T1 restores final padding to +0; status2 must remain latched and refuse/fence.
  Original roots/all42 prior bytes stay exact. Synthetic poison is not a trained
  nonfinite-model result.

Native private T0 inspection occurs AFTER the real chunk Ready using a separate
original read-only flight/transfer-staging ticket held through byte processing.
It creates no StateRoot/checkpoint/session Ready and is excluded from the chunk's
single terminal4B receipt. CPU/B1 oracle steps and diagnostics are not timed work.

No HTTP/SSE acceptance is inferred: no serving callsite exists. No source or
hardware kernel-count, TTFT/ITL or throughput improvement is claimed. Receipt
`dispatches` counts source launch sites, not independently observed driver work.

## Concrete next cuts

1. ROOT source review/strict/default+CUDA+Vulkan builds, descriptor tests; qualify
   cc3 sequential ownership/status foundation before the temporal child.
2. Actual .8B normal/pressure/boundary, then2B; each unknown/numeric fence in a
   fresh process. Reject any bit difference; capture first original producer,
   do not relax tolerance or source profile. The double workspace may honestly
   refuse the existing cap; no implicit budget expansion or fallback is allowed.
3. Full required isolated check and both approved source routes, then actual
   kernel collection/unprofiled measurements if source/numerical gates pass.
4. Separate worker prefix reservation: every row must have at least T+1 prompt
   tokens remaining; reserve final tail before banding, equal real T2 only,
   truthful Ready.tokens2/model_tokens4/roots2, bounded fairness/credit policy.
5. Direct solo path B1/T2 requires factoring the ORIGINAL B1 node emitter and
   genuine B1 private state banks under the one-authority owner, plus existing
   B2 stateless lowering. BatchProgram1 is unsupported; do not pad a fake row.
   B1 KV stays dense token/head/dim, with two admitted private banks after one
   gather, and the existing single-session prepare/Ready/publisher advances2.
   This proposal implements no B1 throughput fix.
6. B4/T2→physical8 needs independent combined admission/full-model proof.
   B8/T2→physical16 is unsupported: tile an explicitly qualified geometry or
   refuse. No T4/full temporal BMM or production CUDA graph admission is inferred.

````

## Source: .agents/runs/serving-and-speculation-native-temporal-grouping-20261020/REVIEW_NATIVE_AB_V2.md

Original UTF-8 bytes: 2550; SHA256 `cd3f63e0222466ef35d7dfc1b90f68dd270ba3d0c040519a9f8b9c196ac03c2b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Independent native/native V2 source review

Author: baseline_review. ROOT assignment; read-only source/metadata review.

Reviewed full V2 namespace diff, binding shim, preparation/runner/validator,
namespace fixtures, nine harmless process fixtures/stub, original NativeOwner,
record-only cancellation, exact accepted wrapper2 and53ee ownership context.
Original12 and V2 eleven live/frozen sources compared byte-exact without imports.
No tests, runtime, containers, model/DSO payload reads/hashes or MAIN edits.

Finding: fixture signal-thread join15s assumed one10s owner inspect. Original
owned_container actually performs exists10 + inspect10, followed by HTTP2.
Thus a still-bounded authenticated observation could outlive join15 and reach
original handler restoration while its producer remains alive. ROOT accepted
fixture-only additive V3; actual V2 nine successes do not exercise this slow path.

No other concrete blocker found in the reviewed V2 derivative scope:

- Exact full rootless UID map/bootstrap, owning userNS ioctl and lo-only/up checks
  match accepted wrapper2. Read-only driver does not perform device/host recovery.
  Rootless custom-store ownership is retained; command recipe also supplies fresh
  private net/mount. Source tests alone do not prove actual mount/cgroup isolation.
- Canonical prepare/run/validator differ only V2 imports +namespace guard.
  Frozen workload/greedy/source/caps, ABBA/BAAB, full IDs/text/usage/frontiers,
  cleanup/error checks and source-vs-hardware count distinction stay unchanged.
- Only actual unreaped foreground, exact nonce/name/CID owner may be retired;
  original independent53ee container/group/EOF stages remain. Receipt-I/O failure
  is after independent cleanup. Any failure prevents a later benchmark sample.
- Harmless target mounts no source/native executable/device, emits no tokens and
  always503 model readiness. Explicit control GPU stub and missing real bootstrap
  cannot satisfy native acceptance. Open-POST cancellation requires actual marker,
  nonce/CID and unreaped leader before signalling only the supervisor itself.
- Canonical benchmark source remains V2 b0a0371f. V3 is an independent fixture
  prerequisite, not a relabelled measurement or numerical/provider qualification.

V3 source delivery: own benchmark followup-v3 freeze60d40c28b3225a887797e536d89f57e7e58fa9d348479877643c219f9f9fcc5e.
Author prepared25s retirement +no-handler-restore-on-live-thread +three mocked
contracts/nine unchanged harmless derivatives; ROOT review/execution still needed.

````

## Source: .agents/runs/serving-and-speculation-native-temporal-grouping-20261020/ROOT-GATES.md

Original UTF-8 bytes: 1828; SHA256 `6ec17ef093d9237c00a1ed972e77f5eeb39874fe95f99d7db8aa57be90fe055e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only gates

Author: baseline_review. No commands executed by author beyond rustfmt/static IO.

- Review original cc3 foundation plus new six-path production snapshot/full delta.
  Compose only owned hunks; preserve MAIN typed/cold-prep/RoPE/graph/transport math.
- Cargo strict CUDA+qualification+portable-checkpoint+hardware-abort-probe,
  strict non-qualification/default/Vulkan as relevant; no missing cfg/type seam
  can be inferred accepted from source formatting.
- Run source-free descriptor tests under
  `resident_program::batch::temporal::tests`.
- Actual release, source model qualifier order .8B then2B. Main child prefix:
  `resident_inference::fused_qualification::prefix_qualification::chunk::temporal`.
  Use exact tests and `--ignored --exact --test-threads=1`:
  `qwen08_temporal_b2_t2`, `qwen2_temporal_b2_t2`;
  `qwen08_temporal_pressure`, `qwen2_temporal_pressure`;
  portable-checkpoint `qwen08_temporal_boundary`, `qwen2_temporal_boundary`.
- Run EACH `qwen{08,2}_temporal_abort` and
  `qwen{08,2}_temporal_certificate_refusal` in a fresh process. They deliberately
  fence card/service/provider; no same-process continuation/recovery.
- Observe true T0/T1 complete42 and independent CPU/B1 fullV evidence, subsequent
  fullV/greedy, rootadvance2 once,4B global transaction status, complete P2+P4
  allocation coverage, uploads0/warm, known-completion final0. Never translate
  physical4 activation rows into four requests or sampler/model cohort counters.
- All current original qualification/routes remain mandatory; complete isolated
  gate: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`.
- Only after correctness: separate fair worker/HTTP integration, real device
  launch collector and unprofiled before/after. No vLLM-equivalence inference.

````

## Source: .agents/runs/serving-and-speculation-native-temporal-grouping-20261020/TODO.md

Original UTF-8 bytes: 999; SHA256 `de41914cced410cb5e14a99461bf45931ff3d922b28d198646141de5ff159880`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native temporal grouping

Parent: serving-and-speculation P12. Owner: baseline_review.

- [x] Preserve cc3 sequential-T and current original owner/program excluded baseline.
- [x] Prepare closed same-plan B2/T2 descriptor, B4 stateless emitter, B2 causal banks.
- [x] Prepare combined original workspace/admission/accesses and original Ready/rootadvance2 orchestration.
- [x] Prepare four source-free dependency fixtures and ten ignored actual-model cases.
- [x] Static rustfmt only; no Cargo/rustc/tests/operators/providers/models/GPU/runtime.
- [ ] ROOT independent source review, strict compilation and real numerical/resource/abort gates.
- [ ] ROOT full isolated check +both original release routes before implementation acceptance.
- [ ] Future solo B1/T2 and fair worker/HTTP integration, actual driver counts and performance.

Unqualified source proposal only. No MAIN/shared plan/README/commit edits. Unknown
and numerical refusal cases deliberately fence; execute each in fresh process.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/README.md

Original UTF-8 bytes: 9398; SHA256 `0707af4694473cfb277df9e1d09f616bcb8f8d893e2a5df4416a5d005f78f8bb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered CUDA SiLU candidate — uncompiled / unexecuted

Author: batch_design_review. Parent: serving-and-speculation.
Owned: this ignored run only. MAIN/current456 row-certificate sources unchanged.

## Exact scope

Three new proposed GPU paths, plus test-only module wiring. One source launch
reuses the original GGML Haswell polynomial, wideF64 accumulation policy,
manual subnormal packing and neighboring-quotient correction. It replaces the
current12 `silu_step` launches and optional `silu_gate`; original interval/finite
certificates, status, source/authority/roots, queue, leases, Ready and quarantine
remain original callers. No device exp, native F32 FMA, new model or profile.
Original-HF/Torch SLEEF SiLU is a DIFFERENT operation and remains untouched.
No production route or automatic runtime fallback is added. Existing accepted
CPU/CUDA/Vulkan/AMD paths stay byte-identical. Renoir/gfx90c hold remains.

Current accepted histogram64328 kernels includes10080 SiLU step kernels and588
gate kernels, for a short B2 warm control. It does NOT measure kernel durations,
canonical workloads, this candidate, latency savings or a throughput result.
Actual current source has12 stages, not32. Below are16 written stage results,
plus optional final gate. No historical phrase changes that executed/source scope.

## Source arithmetic / compiler boundary

`ordered_silu.rs` retains exact constants/order/branches from current `silu.rs`.
Existing `source_float::{widen_source,round_source,correct_quotient}` are called
unchanged. Every old F32 store boundary uses an explicitly rounded PTX F32
register result. Existing CubeCL `gpu_asm!` without `pure` lowers to volatile
inline PTX; no custom compiler/provider/direct CUDA launch is introduced.

- `cvt.rn.f32.f64` fixes narrowing; exact `cvt.f64.f32` widens.
- Explicit RN F64 FMA/add/sub/mul/div and RN F32 `0-input`, without `.ftz`.
- Finite F32 products fit exactly in F64. Original potentially contracted wide
  product+sum and explicit F64 FMA therefore have the same F64 rounding;
  this does NOT claim universal native-F32-FMA equivalence or change old policy.
- Original quotient estimate/subnormal logic passes through an opaque F32
  register-bit move before unchanged correction. Gate uses original manual
  rounding/sign policy; raw U32 final storage retains zero/subnormal bits.
- CubeCL's current NVRTC options are architecture/include paths/lineinfo, with
  no fast-math override. NVIDIA NVRTC12.9 defaults FTZfalse, FMADtrue; explicit
  PTX rounding controls this candidate independently of C++ local-cast folding.
- NVIDIA PTX8.8 documents RN-even and gradual underflow without FTZ on sm20+;
  explicit rounding modifiers constrain optimization. Actual generated PTX/SASS
  and compact-vs-traced physical parity still MUST be inspected/tested.

Primary sources (no SDK/model downloaded):
[NVIDIA PTX8.8, floating point/conversion](https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html),
[NVIDIA NVRTC12.9 compiler options](https://docs.nvidia.com/cuda/archive/12.9.0/nvrtc/index.html).
Local source pins include vendored InlineAsm lowering, original kernels/helpers,
NVRTC option builder and public CubeCL frontend. These establish a candidate
instruction contract, not executed compiler/mathematical qualification.

Observable reference mapping (per element; new trace17 U32 words):

| Original stage | Trace | Original written value |
|---|---|---|
|0|0|z|
|1|1|n|
|2|2|first reduced argument|
|3|3|reduced argument b|
|4|4,5,6,7|b², first/second polynomial seeds, final linear term|
|5|8|first Horner result|
|6|9|second Horner result|
|7|10,11|base / scaling factor|
|8|12|scaled exponential|
|9|13|denominator|
|10|14|physically rounded quotient estimate|
|11|15|corrected quotient|
|optional gate|16|final gated result, or unchanged quotient|

## Checked binding / original owners

`PreparedOrderedSilu::prepare(client,rows,width,Bindings)` accepts rows1/2/4/8,
positive per-row width<=65536, no request padding. Cold validation checks CUDA,
original F64 support, grid/binding bounds, exact handle service, checked aligned
spans, nonoverlapping output/trace writes vs all inputs/writes. READ/READ aliases
are safe. Optional multiplier/trace inactive ABI aliases have no accesses under
compile-time false branches; physical compact mode must verify this.

Prepare allocates/uploads no device storage and submits no physical work. Launch checks the original service
again and returns ONE source launch count. It neither owns SessionAuthority nor
publishes anything. Caller MUST retain all original BufferLease owners in its
original physical flight BEFORE launch, with complete original source/profile/
frontier preflight before model submission. Caller still runs original interval
and final finite certificates/status guard. Status is never cleared here.

No qualification-only graph admission/capture route is enabled. No second stream,
context, model loop, raw provider ABI, independent issuer or CPU fallback exists.

## Prepared ROOT gates

- Two source-free shape/checked-span tests (not executed by author).
- Ignored release CUDA operator qualifier:196 broad/read-alias cases plus8
  transition-neighborhood cases, B1/2/4/8, widths1/31/33/128/2048/6144/11008/65536.
- Original12 physical launches + copy-only U32 stage snapshots are independent
  oracle. Compare ALL16 intermediate writes + final gate, traced vs compact
  outputs, unchanged public CPU source exponential/denominator/SiLU/gate bits.
  Numerical pair counter22/element excludes separately checked input/canary/
  status words. No tolerance, expected-table fitting or signed-zero exception.
- Signedzero, smallest/subnormal/normal boundaries, ±3 subnormal quotient ties,
  ±256, signed scale/overflow branches, broad finite raw bits, dense mantissas,
  analytic ±128-ULP neighborhoods of source range-reduction ties. Synthetic
  recipes select inputs only; original CPU/GPU functions produce references.
- Original physical ResourceArbiter/ResidentService/init leases/fair ticket/
  actual readback/Ready used. Full prefix/suffix canaries and source-input bytes;
  original interval+finite status exact, no warm execution H2D, final normal0.
- Cold foreign-service/short/malformed/misaligned/trace-output/input-output
  aliases refuse without changing initialized ledger/uploads. Real unsubmitted
  fair-ticket cancellation leaves data/events untouched. Known Ready private
  candidate abandonment drops owners without any StateRoot publication.
- Fresh-process fault selectors `unknown`, `preexisting`, `nonfinite`, `range`:
  old/new status must match exactly; no numerical Ready; original pendingevent/
  handles/service/envelope retained and fenced. Unknown intentionally omits
  terminal readback; do not reset/recover or run another fault in same process.
- CPU oracle reads RN/FTZ/DAZ controls without changing them; sticky exception
  flags may change during arithmetic and are not treated as rounding changes.
- Source-call counts (old12/13, new1) are NOT driver enumeration. ROOT needs an
  official activity diagnostic after integration to establish actual reduction.

ROOT ONLY after source review/quiet GPU reservation: copy proposed paths with
fresh mtimes, apply ROOT_WIRING.patch, run focused strictCUDA/unit then original
isolated release ignored test. Example test selector:

```text
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py --
  cargo test -p zerogravity-gpu --release --features cuda,qualification --lib
  ordered_silu::qualification::physical_ordered_silu -- --ignored --exact --nocapture
```

Use same original supervisor/isolation for separate fault env selectors and
`known-ready-abandon`. Do not infer physical proof from preparation or compilation.
Root may retain `CUBECL_CUDA_DUMP_PTX=FRESH_OWNED_DIR` and
`CUBECL_DEBUG_LOG=FRESH_OWNED_LOG` under existing engine dump/log hooks. They may
silently fail or hit cache: missing actual witnesses REFUSE compiled-route
qualification. Bind compact/traced and multiplied/plain specializations through
actual compile log/test/binary/source. `audit_ptx.py --ptx ... --output ...` is a
bounded token witness only; inspect actual SASS too, never claim it executed from
an instruction's mere existence. No helper compiles/loads/runs code by itself.

## Remaining before production or speed claim

No build, unit, GPU, model, provider or timing work by author; only rustfmt and
small source/metadata inspection/freezing + Python AST parsing. If old physical
pipeline or new kernel fails any CPU adversary, keep both scopes unqualified;
do NOT widen tolerance, canonicalize zeros or silently switch numerical policy.

After exact operator+compiled gates: narrowly wire only qualified GGML CUDA
PlanOperation::Silu to existing B1/lifted batch binding, retaining old path
outside qualified backend/profile/geometry. Preflight all original buffers and
status before flight. Full original .8B/2B B1/2/4/8/mixedfrontier fullV/all42/
packets/real cancel/unknown/pressure/worker+HTTP, both original parity routes,
strict/full isolated check required. Actual CUPTI before/after on same accepted
canonical workload, then unprofiled paired timing required for any gain claim.
Session known-Ready cancellation/publication is a FULLMODEL gate; standalone
private-output abandonment above does not prove session cancellation semantics.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/TODO.md

Original UTF-8 bytes: 1545; SHA256 `e5f211a3971fb958b0e1e65fb25aab292cc0332168ab16bbdc7db6b501cb65a1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU source candidate

Parent: serving-and-speculation. Owner: batch_design_review.
Owned: this ignored run ONLY; no MAIN/shared plans/other freezes changed.

- [x] Read live12-stage source/helpers, B1/lifted batch callsites, original CPU
  GGML source function, actual short-window kernel histogram; not latency.
- [x] Read original CubeCL public asm frontend/vendor lowering/NVRTC options;
  primary NVIDIA12.9 ISA/compiler semantics; no SDK/model acquisition.
- [x] Prepare dormant new module with explicit RN/no-FTZ boundaries, same source
  polynomial/wide accumulation/packing/correction, checked original bindings.
- [x] Prepare independent physical12-stage snapshots,204 normal cases, CPU source
  comparison, status/alias/canary/cancel/private abandonment/unknown negatives.
- [x] rustfmt proposed3Rust paths; AST only Python witness checker. No imports,
  tests/builds/compile/operator/model/GPU/provider/runtime/payload scans.
- [ ] Independent source review; ROOT strict/unit/release/compiled PTX+SASS and
  actual physical operator/negative gates. No expected-output relaxation.
- [ ] Root-only production binding/inference/worker/HTTP/fullgate and source-bit
  all42/fullV acceptance; no production selection here.
- [ ] Actual driver activity dispatch reduction and unprofiled matched timing.

Source inspection corrections before freeze: source has12 stages/16 writes,
not32 launches. CPU oracle compares MXCSR control bits only, since legitimate
underflow/inexact sticky flags may change without changing math policy.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/followup-v2/README.md

Original UTF-8 bytes: 2483; SHA256 `c47d016cd7fd8f3447335c8392ff80006057050e06f06d214b38e4467822dde4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU qualifier V2 followup

Author: batch_design_review. Parent: serving-and-speculation P12.

- Original a7939ba7 freeze, all nine live/frozen files remain unchanged.
- `BYTE_BUDGET_ONLY.patch`: exactly two invalid `* *2` integer dereferences become
  intended explicit MiB products: resident256MiB/device640MiB. No budget increase
  relative to the documented intended values; original source never compiled.
- `INITIALIZATION_MATH_FLIGHTS.patch`: SAME original fair queue/service/leases.
  Initialization slot begins before cleanup/allocation/H2D, original initialization
  `complete_sync` produces Ready, then the initialization slot finishes. Moving
  this begin after H2D would violate original PhysicalQueue/ResidentService rules.
- All prepared bindings, alias/service/span negatives, queued-ticket cancellation,
  accesses/readbacks and CPU math-mode validation precede a NEW original math
  ticket. It uses the same prepaid envelope without a second resource charge;
  its acquired slot begins immediately before `service.submit`. No foreign stream,
  source authority, actor, event, lease, recovery, admission or production route.
- Original leases remain alive across init-slot retirement and cold validation.
  A cold refusal has no math submission/event; initialization already reached
  physical Ready. Initialization/unknown math completion still use original
  RAII quarantine. Both original flights retain their original guards.
- `QUALIFIER_V2.review.patch` is the composed two-fix delta; apply that OR the two
  ordered component patches, never both. New proposed copy is the final qualifier.
- Moving passive MXCSR validation before math flight changes no arithmetic.
  All 204 cases, 17 ordinal snapshots, statuses, canaries and fault recipes stay
  unchanged. Queue selected/completed totals now include distinct init/math jobs;
  these are diagnostic lifecycle counters, not driver kernel counts.

ROOT gates remain unexecuted: strict/default/Vulkan/CUDA compilation, two pure
Rust tests, compiled PTX and SASS review, all204 old-GPU/CPU exact-word cases and
fresh unknown/preexisting/nonfinite/range/known-Ready-abandon cases. No fullmodel,
HTTP, graph, numerical-profile eligibility or measured performance claim.

Author: source inspection; rustfmt final proposed qualifier; small source hashes
and immutable-parent comparisons only. No tests, Cargo/build, model/operator,
provider/GPU execution, PTX/SASS scan, DSO/raw trace/model payload hash.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/followup-v2/TODO.md

Original UTF-8 bytes: 889; SHA256 `70ca8a792c97d24f36b5c750dd36516ff90379ed9704755e8c41dc18a9a5e30b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU qualifier followup V2

Parent: serving-and-speculation P12; owner: batch_design_review.
Owned: fresh ignored followup only; original a7939 source and MAIN unchanged.

- [x] Preserve original strict compile blocker and frozen9-file source.
- [x] Repair exactly two byte-budget expressions in independent patch.
- [x] Source-check original queue/init contracts; separate original initialization
  and math fair slots, with all cold checks before math submission.
- [x] Rustfmt final qualifier; freeze composed/component diffs and source context.
- [ ] ROOT/independent source review, actual strict/unit/build/compiled shader gate.
- [ ] ROOT physical204 oldGPU+CPU/fault/canary/status/Ready/quarantine comparison.
- [ ] Fullmodel/worker/HTTP/parity/fullgate and real-count/unprofiled timing gates.

No author test/build/operator/model/provider/GPU runtime; no payload hashes.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/followup-v2/frozen/README.md

Original UTF-8 bytes: 2483; SHA256 `c47d016cd7fd8f3447335c8392ff80006057050e06f06d214b38e4467822dde4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU qualifier V2 followup

Author: batch_design_review. Parent: serving-and-speculation P12.

- Original a7939ba7 freeze, all nine live/frozen files remain unchanged.
- `BYTE_BUDGET_ONLY.patch`: exactly two invalid `* *2` integer dereferences become
  intended explicit MiB products: resident256MiB/device640MiB. No budget increase
  relative to the documented intended values; original source never compiled.
- `INITIALIZATION_MATH_FLIGHTS.patch`: SAME original fair queue/service/leases.
  Initialization slot begins before cleanup/allocation/H2D, original initialization
  `complete_sync` produces Ready, then the initialization slot finishes. Moving
  this begin after H2D would violate original PhysicalQueue/ResidentService rules.
- All prepared bindings, alias/service/span negatives, queued-ticket cancellation,
  accesses/readbacks and CPU math-mode validation precede a NEW original math
  ticket. It uses the same prepaid envelope without a second resource charge;
  its acquired slot begins immediately before `service.submit`. No foreign stream,
  source authority, actor, event, lease, recovery, admission or production route.
- Original leases remain alive across init-slot retirement and cold validation.
  A cold refusal has no math submission/event; initialization already reached
  physical Ready. Initialization/unknown math completion still use original
  RAII quarantine. Both original flights retain their original guards.
- `QUALIFIER_V2.review.patch` is the composed two-fix delta; apply that OR the two
  ordered component patches, never both. New proposed copy is the final qualifier.
- Moving passive MXCSR validation before math flight changes no arithmetic.
  All 204 cases, 17 ordinal snapshots, statuses, canaries and fault recipes stay
  unchanged. Queue selected/completed totals now include distinct init/math jobs;
  these are diagnostic lifecycle counters, not driver kernel counts.

ROOT gates remain unexecuted: strict/default/Vulkan/CUDA compilation, two pure
Rust tests, compiled PTX and SASS review, all204 old-GPU/CPU exact-word cases and
fresh unknown/preexisting/nonfinite/range/known-Ready-abandon cases. No fullmodel,
HTTP, graph, numerical-profile eligibility or measured performance claim.

Author: source inspection; rustfmt final proposed qualifier; small source hashes
and immutable-parent comparisons only. No tests, Cargo/build, model/operator,
provider/GPU execution, PTX/SASS scan, DSO/raw trace/model payload hash.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/followup-v2/frozen/TODO.md

Original UTF-8 bytes: 889; SHA256 `70ca8a792c97d24f36b5c750dd36516ff90379ed9704755e8c41dc18a9a5e30b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU qualifier followup V2

Parent: serving-and-speculation P12; owner: batch_design_review.
Owned: fresh ignored followup only; original a7939 source and MAIN unchanged.

- [x] Preserve original strict compile blocker and frozen9-file source.
- [x] Repair exactly two byte-budget expressions in independent patch.
- [x] Source-check original queue/init contracts; separate original initialization
  and math fair slots, with all cold checks before math submission.
- [x] Rustfmt final qualifier; freeze composed/component diffs and source context.
- [ ] ROOT/independent source review, actual strict/unit/build/compiled shader gate.
- [ ] ROOT physical204 oldGPU+CPU/fault/canary/status/Ready/quarantine comparison.
- [ ] Fullmodel/worker/HTTP/parity/fullgate and real-count/unprofiled timing gates.

No author test/build/operator/model/provider/GPU runtime; no payload hashes.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/frozen/README.md

Original UTF-8 bytes: 9398; SHA256 `0707af4694473cfb277df9e1d09f616bcb8f8d893e2a5df4416a5d005f78f8bb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered CUDA SiLU candidate — uncompiled / unexecuted

Author: batch_design_review. Parent: serving-and-speculation.
Owned: this ignored run only. MAIN/current456 row-certificate sources unchanged.

## Exact scope

Three new proposed GPU paths, plus test-only module wiring. One source launch
reuses the original GGML Haswell polynomial, wideF64 accumulation policy,
manual subnormal packing and neighboring-quotient correction. It replaces the
current12 `silu_step` launches and optional `silu_gate`; original interval/finite
certificates, status, source/authority/roots, queue, leases, Ready and quarantine
remain original callers. No device exp, native F32 FMA, new model or profile.
Original-HF/Torch SLEEF SiLU is a DIFFERENT operation and remains untouched.
No production route or automatic runtime fallback is added. Existing accepted
CPU/CUDA/Vulkan/AMD paths stay byte-identical. Renoir/gfx90c hold remains.

Current accepted histogram64328 kernels includes10080 SiLU step kernels and588
gate kernels, for a short B2 warm control. It does NOT measure kernel durations,
canonical workloads, this candidate, latency savings or a throughput result.
Actual current source has12 stages, not32. Below are16 written stage results,
plus optional final gate. No historical phrase changes that executed/source scope.

## Source arithmetic / compiler boundary

`ordered_silu.rs` retains exact constants/order/branches from current `silu.rs`.
Existing `source_float::{widen_source,round_source,correct_quotient}` are called
unchanged. Every old F32 store boundary uses an explicitly rounded PTX F32
register result. Existing CubeCL `gpu_asm!` without `pure` lowers to volatile
inline PTX; no custom compiler/provider/direct CUDA launch is introduced.

- `cvt.rn.f32.f64` fixes narrowing; exact `cvt.f64.f32` widens.
- Explicit RN F64 FMA/add/sub/mul/div and RN F32 `0-input`, without `.ftz`.
- Finite F32 products fit exactly in F64. Original potentially contracted wide
  product+sum and explicit F64 FMA therefore have the same F64 rounding;
  this does NOT claim universal native-F32-FMA equivalence or change old policy.
- Original quotient estimate/subnormal logic passes through an opaque F32
  register-bit move before unchanged correction. Gate uses original manual
  rounding/sign policy; raw U32 final storage retains zero/subnormal bits.
- CubeCL's current NVRTC options are architecture/include paths/lineinfo, with
  no fast-math override. NVIDIA NVRTC12.9 defaults FTZfalse, FMADtrue; explicit
  PTX rounding controls this candidate independently of C++ local-cast folding.
- NVIDIA PTX8.8 documents RN-even and gradual underflow without FTZ on sm20+;
  explicit rounding modifiers constrain optimization. Actual generated PTX/SASS
  and compact-vs-traced physical parity still MUST be inspected/tested.

Primary sources (no SDK/model downloaded):
[NVIDIA PTX8.8, floating point/conversion](https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html),
[NVIDIA NVRTC12.9 compiler options](https://docs.nvidia.com/cuda/archive/12.9.0/nvrtc/index.html).
Local source pins include vendored InlineAsm lowering, original kernels/helpers,
NVRTC option builder and public CubeCL frontend. These establish a candidate
instruction contract, not executed compiler/mathematical qualification.

Observable reference mapping (per element; new trace17 U32 words):

| Original stage | Trace | Original written value |
|---|---|---|
|0|0|z|
|1|1|n|
|2|2|first reduced argument|
|3|3|reduced argument b|
|4|4,5,6,7|b², first/second polynomial seeds, final linear term|
|5|8|first Horner result|
|6|9|second Horner result|
|7|10,11|base / scaling factor|
|8|12|scaled exponential|
|9|13|denominator|
|10|14|physically rounded quotient estimate|
|11|15|corrected quotient|
|optional gate|16|final gated result, or unchanged quotient|

## Checked binding / original owners

`PreparedOrderedSilu::prepare(client,rows,width,Bindings)` accepts rows1/2/4/8,
positive per-row width<=65536, no request padding. Cold validation checks CUDA,
original F64 support, grid/binding bounds, exact handle service, checked aligned
spans, nonoverlapping output/trace writes vs all inputs/writes. READ/READ aliases
are safe. Optional multiplier/trace inactive ABI aliases have no accesses under
compile-time false branches; physical compact mode must verify this.

Prepare allocates/uploads no device storage and submits no physical work. Launch checks the original service
again and returns ONE source launch count. It neither owns SessionAuthority nor
publishes anything. Caller MUST retain all original BufferLease owners in its
original physical flight BEFORE launch, with complete original source/profile/
frontier preflight before model submission. Caller still runs original interval
and final finite certificates/status guard. Status is never cleared here.

No qualification-only graph admission/capture route is enabled. No second stream,
context, model loop, raw provider ABI, independent issuer or CPU fallback exists.

## Prepared ROOT gates

- Two source-free shape/checked-span tests (not executed by author).
- Ignored release CUDA operator qualifier:196 broad/read-alias cases plus8
  transition-neighborhood cases, B1/2/4/8, widths1/31/33/128/2048/6144/11008/65536.
- Original12 physical launches + copy-only U32 stage snapshots are independent
  oracle. Compare ALL16 intermediate writes + final gate, traced vs compact
  outputs, unchanged public CPU source exponential/denominator/SiLU/gate bits.
  Numerical pair counter22/element excludes separately checked input/canary/
  status words. No tolerance, expected-table fitting or signed-zero exception.
- Signedzero, smallest/subnormal/normal boundaries, ±3 subnormal quotient ties,
  ±256, signed scale/overflow branches, broad finite raw bits, dense mantissas,
  analytic ±128-ULP neighborhoods of source range-reduction ties. Synthetic
  recipes select inputs only; original CPU/GPU functions produce references.
- Original physical ResourceArbiter/ResidentService/init leases/fair ticket/
  actual readback/Ready used. Full prefix/suffix canaries and source-input bytes;
  original interval+finite status exact, no warm execution H2D, final normal0.
- Cold foreign-service/short/malformed/misaligned/trace-output/input-output
  aliases refuse without changing initialized ledger/uploads. Real unsubmitted
  fair-ticket cancellation leaves data/events untouched. Known Ready private
  candidate abandonment drops owners without any StateRoot publication.
- Fresh-process fault selectors `unknown`, `preexisting`, `nonfinite`, `range`:
  old/new status must match exactly; no numerical Ready; original pendingevent/
  handles/service/envelope retained and fenced. Unknown intentionally omits
  terminal readback; do not reset/recover or run another fault in same process.
- CPU oracle reads RN/FTZ/DAZ controls without changing them; sticky exception
  flags may change during arithmetic and are not treated as rounding changes.
- Source-call counts (old12/13, new1) are NOT driver enumeration. ROOT needs an
  official activity diagnostic after integration to establish actual reduction.

ROOT ONLY after source review/quiet GPU reservation: copy proposed paths with
fresh mtimes, apply ROOT_WIRING.patch, run focused strictCUDA/unit then original
isolated release ignored test. Example test selector:

```text
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py --
  cargo test -p zerogravity-gpu --release --features cuda,qualification --lib
  ordered_silu::qualification::physical_ordered_silu -- --ignored --exact --nocapture
```

Use same original supervisor/isolation for separate fault env selectors and
`known-ready-abandon`. Do not infer physical proof from preparation or compilation.
Root may retain `CUBECL_CUDA_DUMP_PTX=FRESH_OWNED_DIR` and
`CUBECL_DEBUG_LOG=FRESH_OWNED_LOG` under existing engine dump/log hooks. They may
silently fail or hit cache: missing actual witnesses REFUSE compiled-route
qualification. Bind compact/traced and multiplied/plain specializations through
actual compile log/test/binary/source. `audit_ptx.py --ptx ... --output ...` is a
bounded token witness only; inspect actual SASS too, never claim it executed from
an instruction's mere existence. No helper compiles/loads/runs code by itself.

## Remaining before production or speed claim

No build, unit, GPU, model, provider or timing work by author; only rustfmt and
small source/metadata inspection/freezing + Python AST parsing. If old physical
pipeline or new kernel fails any CPU adversary, keep both scopes unqualified;
do NOT widen tolerance, canonicalize zeros or silently switch numerical policy.

After exact operator+compiled gates: narrowly wire only qualified GGML CUDA
PlanOperation::Silu to existing B1/lifted batch binding, retaining old path
outside qualified backend/profile/geometry. Preflight all original buffers and
status before flight. Full original .8B/2B B1/2/4/8/mixedfrontier fullV/all42/
packets/real cancel/unknown/pressure/worker+HTTP, both original parity routes,
strict/full isolated check required. Actual CUPTI before/after on same accepted
canonical workload, then unprofiled paired timing required for any gain claim.
Session known-Ready cancellation/publication is a FULLMODEL gate; standalone
private-output abandonment above does not prove session cancellation semantics.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-20261020/frozen/TODO.md

Original UTF-8 bytes: 1545; SHA256 `e5f211a3971fb958b0e1e65fb25aab292cc0332168ab16bbdc7db6b501cb65a1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU source candidate

Parent: serving-and-speculation. Owner: batch_design_review.
Owned: this ignored run ONLY; no MAIN/shared plans/other freezes changed.

- [x] Read live12-stage source/helpers, B1/lifted batch callsites, original CPU
  GGML source function, actual short-window kernel histogram; not latency.
- [x] Read original CubeCL public asm frontend/vendor lowering/NVRTC options;
  primary NVIDIA12.9 ISA/compiler semantics; no SDK/model acquisition.
- [x] Prepare dormant new module with explicit RN/no-FTZ boundaries, same source
  polynomial/wide accumulation/packing/correction, checked original bindings.
- [x] Prepare independent physical12-stage snapshots,204 normal cases, CPU source
  comparison, status/alias/canary/cancel/private abandonment/unknown negatives.
- [x] rustfmt proposed3Rust paths; AST only Python witness checker. No imports,
  tests/builds/compile/operator/model/GPU/provider/runtime/payload scans.
- [ ] Independent source review; ROOT strict/unit/release/compiled PTX+SASS and
  actual physical operator/negative gates. No expected-output relaxation.
- [ ] Root-only production binding/inference/worker/HTTP/fullgate and source-bit
  all42/fullV acceptance; no production selection here.
- [ ] Actual driver activity dispatch reduction and unprofiled matched timing.

Source inspection corrections before freeze: source has12 stages/16 writes,
not32 launches. CPU oracle compares MXCSR control bits only, since legitimate
underflow/inexact sticky flags may change without changing math policy.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-compiler-witness-review-20261020/REVIEW.md

Original UTF-8 bytes: 6956; SHA256 `70a5092bd59d509c5373102b2b93d38345354d62283bd4a25bfb574a35dfab95`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU compiler witness — source-only peer review

## Verdict and actual receipt scope

- No blocker in scalar-input CPP auditor or its five parser fixtures. The prior claimed missing assertRaises was retracted; actual source has it.
- `audit_generated_cpp.py` checks all parsed volatile asm calls against nine result/input constraint families and generated float/double input declarations; it requires all nine families and rejects unparsed calls, unsupported opcodes and pointer constraints. Synthetic parser success is explicitly not authentic compiler/numerical proof.
- `audit_ptx.py` checks explicit opcode presence, >=12 F32 rounding conversions, no FTZ, no approximate exp/reciprocal, no F32 FMA substitution. It explicitly qualifies tokens only, not full control/dataflow, kernel routing, SASS or performance. It currently scans an entire selected PTX module; unrelated entries/functions must not be mistaken for the selected ordered entry.
- Retained root-physical-2: six fresh processes exit0, original owned group/leader/EOF cleanup all pass. Normal log reports171,562,710 bit pairs; source loop gives204 cases. Known-Ready abandonment compares5,808 pairs; four other fault modes compare0 and exercise refusal/unknown/status only. Do not read their unconditional summary's CPU_source=exact as additional fault numerical comparison.
- ROOT compiler report honestly says all144 ordered CPP constraints passed but only one ordered PTX survives: compact gated count41120, entry ordered_6be19757. Other three selected count41120 records have no retained PTX. Original last180-character sanitized filenames overwrite settings-distinct artifacts. This blocks a claim of four/all144-specialization PTX coverage; it does not erase actual physical operator results.
- Source-only review: no validators/tests/imports/builds/hashes/CPP-or-PTX-payload scans, binary/DSO/model access or runtime by reviewer. Small actual receipts/logs and compiler/auditor sources only.

## Correct CPP → PTX binding

1. Bind immutable source freeze/build/executable/hardware/compiler options and original normal process receipt. Keep compilation logger full in its private CWD and profiling disabled; do not enable CUBECL_DEBUG_LOG, which changes profiling.
2. Select complete authentic START/END compilation blocks for exact `zerogravity_gpu::ordered_silu::ordered::Ordered`. Retain full Debug KernelId, U32/CubeDim/unchecked mode and count/multiplied/record flags, plus exact generated CPP and declared entry name. Match block identity, not label/order/filename suffix.
3. With the proposed diagnostic fix, join full KernelId sidecar byte-for-byte to that log id. Require exactly one complete raw PTX partner for each expected compile attempt, and exact matching `.entry` symbol from its CPP. Reject missing/unpaired/partial files, ambiguous entries and overwritten/duplicate identity assumptions. Same-id repeated loads are distinct attempts; a hash/filename is only an index.
4. Run scalar and instruction auditors separately on every selected specialization. At minimum all four plain/gated × traced/compact count41120, preferably all144 actual normal ordered variants. If a PTX contains multiple entries/functions, restrict the claimed instruction evidence to the exact selected entry and its called functions; module token presence alone is weaker.
5. Source route: full CPP log is emitted immediately before compile_to_ptx; its returned PTX is passed to load_ptx, dumped unchanged and then passed to cuModuleLoadData/get_function using that original entrypoint. Successful physical readback proves the relevant launch completed for tested cases. The diagnostic text dump itself occurs before module load and does not prove load success by itself.
6. Preserve PTX version/target/address-size and actual NVRTC/driver bindings. Claim **actual original-queue generated CPP + pre-JIT PTX instruction witness + physical bit parity**, never loaded SASS identity or every compiler optimization/correctness property. No driver binary disassembly is supplied by these scripts.

## Proposed narrow dump transport

- Only existing opt-in DUMP env path changes. Default arithmetic, NVRTC options, module data, original queue/stream, allocations/Ready/quarantine and source profiles are unchanged.
- Prefix: hash of full Debug KernelId (same text as logger), PID and process-global atomic sequence; bounded short filename. Hash is not authority. `create_new` on PTX and sidecar forbids overwrite even for same id/hash, PID reuse or sequence collision. Existing-path retries are finite64.
- Adjacent `.kernel-id.txt` retains full id, `.ptx` retains original bytes. Writes remain best-effort diagnostic I/O; missing/partial/unpaired artifacts must fail compiler-witness acceptance. A filesystem failure does not silently become proof or change inference math.
- One prepared CPU transport unit covers equal long suffixes and repeated exact id, raw byte conservation, sidecar identity, unique short names and six files. Synthetic file bytes are explicitly not compiler witnesses. ROOT must format/build/test it; author did not execute it.
- After ROOT review, use a fresh dump directory/new normal receipt and compare source-only diagnostic delta against the previous accepted source. Preserve root-physical-2 and earlier failed receipts unchanged. No need to rerun faults for numeric changes in this file-only diagnostic patch, unless ROOT's source/build identity or owner gate requires it; production integration still requires its full fresh gate below.

## Production integration beyond the operator sample

- Independent CPU and accepted B1/all42/full248320 for both approved models, plus nativeB2/B4/B8 fullV/all42 and mixed positions256/257/refills/canonical future-zero state. New B1/native paths share this leaf, so mutual agreement alone is insufficient.
- Exercise both gated FFN and plain GDN SiLU at genuine model widths, real quantization profiles and StateOnly skipped-tail paths. Preserve exact original interval status and outer certificates; no clearing preexisting errors. Explicit cold geometry fallback and Vulkan/CPU/typed-HF refusal tests; no hot Err fallback.
- Greedy full-vocabulary ties/real EOS, B1→native→B1 packet feedback, ragged retire/refill, Ready cancellation/regroup, stale roots, genuine admission pressure and fresh unknown-completion quarantine. Budget/physical buffer/scratch totals must remain unchanged; all roots/sessions retain independent publication authority.
- Both release parity routes, strict CUDA/Vulkan/default checks/full isolated gate, new worker14/HTTP12 physical suite. Diagnostic graph stays disabled.
- Source launch reductions must use actual executed SiLU nodes:11 per plain and12 per gated supported node; StateOnly counts only its executed prefix. These are not driver-enumerated counts or latency evidence. Separate real CUPTI diagnostic and unprofiled same-native A/B required before any speed claim.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-compiler-witness-review-20261020/ROOT.md

Original UTF-8 bytes: 1081; SHA256 `aa9f44f09d844275a02fd47d3e8448be624628b0b6f955447826f1b050b6cc9a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only next steps

- Review DUMP_COLLISION_ONLY.patch and original compiler route; do not edit old scalar/operator/integration frozen artifacts.
- ROOT format/strict + one vendor CPU unit `ptx_dumps_keep_suffix_collisions_and_repeated_ids_distinct`; fixture creates only its own fresh temp directory before RAII ownership, no CUDA calls.
- Compose diagnostic change in explicit new source/binary receipt. Run fresh normal204-case operator process with full compilation logger/profiling disabled and new PTX directory; compiler artifact overhead is diagnostic only.
- Pair full Debug KernelIds/CPP entries with new sidecars/raw PTX; require four/all144 selected entries complete and run existing auditors. Reject all incomplete pairs and no-retained-PTX variants; preserve root-physical-2/report.
- No loaded SASS claim. All production integration/fullmodel/state/HTTP/gate/A-B requirements remain in REVIEW.md and immutable9110ROOT.md.

Author did not execute code or calculate hashes per assigned boundary. This is an unexecuted source proposal, not a qualification receipt.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-compiler-witness-review-20261020/TODO.md

Original UTF-8 bytes: 801; SHA256 `35b65b9049f5cbaa5e7724362f00e2a40e1704bb817484f9ef86c6a18ac9061a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU compiler witness review

Parent: serving-and-speculation. Owner: batch_design_review. Scope: this ignored source-only review/proposal.

- [x] Read CPP/PTX auditors, parser fixtures, original logger/load/dump source and actual six small operator receipts/logs.
- [x] Identify original suffix180 collision and honest one-retained-PTX scope; report exactCPP entry/KernelId binding design.
- [x] Prepare vendor-only opt-in diagnostic non-overwrite transport and one source-only unit fixture.
- [ ] ROOT source review/format/strict/vendor unit/new actual normal operator/compiler witness gate.
- [ ] ROOT complete model/serving/routes/fullgate/A-B integration gates remain pending.

No imports/tests/builds/runtime/hashes/payload scans by author in this review. MAIN/shared plans unchanged.

````

