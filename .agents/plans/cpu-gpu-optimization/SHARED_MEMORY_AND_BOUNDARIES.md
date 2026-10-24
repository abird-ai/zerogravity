# Shared storage and CPU–GPU boundaries

Author: Codex. Owner: @pvl. User direction,23Oct.
Status: reconciled design plus historical causal investigation. Current
implementation/qualification: [TODO.md](TODO.md) and [measurements](SHORT_HF_RESULTS.md).
Extend the existing engine and recorded
pipeline; model runtimes supply mathematics/source bindings, not allocators or
new completion authorities.

Current24Oct: shared pools/rings, source-shaped residency and prepared bindings
are integrated. Source20 passes physical23/full2B numerical comparison and the
x2 finite Qwen/Gemma shared lifecycle unit. Its .8B cold-first-use API refusal
remains failed. Source21 prepares the two actually observed missing state
certificate recipes; CPU/strict/release/physical23/full2Bmath/bothAPIHTTP and
fresh .8/2B C1/C4/C8 pass. Compiler .8count448 and2Bcount396 remain unchanged
through their actual API/HTTP and timing units; ZG leads both models in allthree
bands for this pinned singlepass profile.
The2B4608MiB provider allowance remains
inside unchanged5120MiB physical admission. Historical measured C1 removed the
streamed-head gap; current2B cachedB4 leads the pinned stock C4/C8 arms. Gemma learned PLE-hit
traffic/full-state and natural Qwen27B checks advance independently. Historical
calculations and unresolved backend descriptions below retain their dated scope;
live source, current TODO and actual results govern capability claims.

## Historical causal evidence and priorities

- CubeCL already pooled physical allocations. The actor8 baseline requested48
  fresh state planes/forward:32,784,384B payload/root. Sampled allocator scans
  and coalescing are material CPU costs; they are not measured wall fractions.
- Cold value/RMS buffers already persisted. Baseline fresh binding cloned six
  immutable StorageRecipe tables and builds access/owner metadata. Share the
  immutable tables and reuse bounded working capacity, preserving fresh records,
  root/IDs/epochs and mutable traversal state.
- Production ShortHF already uses one terminal status/12B packet cohort.
  Full producer/logit/KV readbacks are qualification work. Keep that distinction.
- Gemma's draft waits on every source page and Observe. Bounded staging/frame
  retirement can remove repeated CPU waits. One same-stream device page can
  order upload→consume→overwrite; host DMA owners must still survive completion.
  Additional device pages or copy streams require separate actual admission.
- Current cold-preparation source batch adds shared content keys, cold GDN/FA
  copy definitions and immutable source-seal reuse. Focused strict checks and
  joint release pass. Original301edge+40natural/all5fault and matchedAPI checks
  pass with unchanged device counts. Quietpair9 observes C1+59.63%/C4+25.76%/
  C8+2.03%; singlepass, small C8 margin unreplicated. Full isolated gate11
  passes1724Rust/zero failures/allscriptchecks over260unchangedsourcepaths.

Actual actor8 capture:64forwards,258user-cycle samples/zero lost. Host bind
.138115s/enqueue.516009s; whole GPU timeline envelope.615946s, nested Head
.099558s. These scopes overlap and contain submission gaps; no subtraction or
sample percentage establishes removable time. Quiet pair9 results above remain
preliminary.2B actual301edges+40natural/all5faults and matchedAPI pass. Its first
quiet C1 pair was ZG0.5298 versus stock37.0680tok/s, about70x slower;16offers/
128tokens exact, both original owner/source/EOF closures pass. Further2B bands
were held for the shared storage/streaming repair. That2B provider source
bound4033269760B includes zero resident-head bytes. Resident2048-head capability
and complete revised fit must both be qualified before enabling residency.

Source-derived2B head cost:248320rows x2048columns x2B =1017118720B perforward,
970one-MiB authenticated source uploads/native launches/actual stream waits.
Production audit=None omits diagnostic W readback but still synchronizes each
window. These are source behavior/counts, not a complete timing attribution.
Current CUDA resident-head/source-inventory paths also contain .8B-only extents;
do not widen a capability flag or reuse TIED_BYTES for2B. Use exact source matrix
geometry and the genuine original2048 full-head packet through shared owners.

## Small shared API surface

User23Oct: finish pools/rings and lift the existing .8B optimizations through
the shared layers before further2B timings. The accepted units and remaining
qualification are tracked in the linked TODO. Keep these responsibilities in the
existing crates/modules, without facade-only traits or another pipeline IR:

| Layer | Owns | Reused by the next layer |
|---|---|---|
| Engine primitives | Charged resident backing, fresh logical leases, exact views/exclusive activation, bounded transfer retirement, immutable matrix geometry | Existing ResidentService/initialization/submission/completion APIs |
| Shared model operations | Recorded storage lifetimes, immutable layout/access tables, RMS/projection/row transport/greedy packets and source-shaped native capabilities | Existing prepared programs and lowerers |
| Family primitives | Original tensor roles/ties, configuration/geometry, mathematical traversal and family-specific operators | Qwen/Gemma source and binding adapters |
| Model specialization | Actual source/config and qualified implementation choice; only irreducible shape/backend specializations | Thin invocation binding, preserving current root/IDs/epochs |

Resident head eligibility follows the authenticated matrix geometry and exact
original native packet. The Qwen family authenticates its embedding/output tie;
the engine calculates rows/bytes/windows and handles residency/retirement.
Do not encode a .8B byte count in shared inventory or enable a larger model by
changing a boolean. Generic transport does not qualify another family's math.

Owner capacity clarification,23Oct:4GiB is the configured provider allowance,
not physical VRAM. Live local RTX3060Laptop reports6144MiB. New source threads
the explicit provider capacity through selection/service/queue;2B physical
qualification uses4608MiB within the same5120MiB physical cap, with original
4096MiB host and all native-owner charges retained. Complete provider bound
4598390272B fits4608MiB with233447936B spare; actual resident correctness and
the separately sourced C1 comparison now pass in the linked results.
Fresh .8B comparison retains4096MiB; later2B timing needs an explicit revised
protocol and must not claim unchanged4GiB/equal-resource comparison. The whole
provider envelope remains reserved even when actual live buffers use less.

The first conservative six-geometry value/RMS overlay derives450,949,632B
reclamation for2B. Its full resident-table provider bound remains303,422,976B
over4GiB; this source calculation is not actual fit or a measured speedup.
Further reclamation must follow real lifetime/capacity contracts. Preserve
the streamed fallback and complete source-window retirement when it cannot fit.

Use the existing prepare→bind→enqueue→complete/commit flow. Preparation derives
storage/layouts/limits from the original plan and source. Binding supplies the
actual current RootAttempt, IDs, M/frontier and leases. The original submission
owns every queued command; original Ready and SessionAuthority publish results.

Internal pool/ring types should remain behind these existing objects where
possible. Add a public type only for a real model-runtime need; no family-owned
pool, raw handle return API, second dispatcher or second mathematical IR.

| Shared concern | API responsibility | Required invariant |
|---|---|---|
| Reusable resident backing | Prepare an exact-layout bounded pool; checkout returns existing ResidentLease type | Fresh logical IDs/generations and uninitialized contents for each attempt |
| Immutable preparation | Share sealed layout, source, kernel/ABI and access schemas; bind current operands | Full content/collision checks; no cached request/root authority |
| Upload/readback | Original flight accepts owned authenticated source views or admitted writable transfer slots | Exact ranges, same service/stream, actual completion before host read or reuse |
| Transfer/frame retirement | Opaque marks and a bounded ready/drain interface | Unknown work retains source, host/device storage, events and original admission |
| Publication/feedback | Existing compact private device packet and terminal status cohort | Original current-root provenance; no logits/KV round trip in ordinary serving |
| Prepared native owners | Backend seals actual compiled/client/cache owners and reservations | A caller byte count, source length or pool report cannot establish complete fit |

Prefer borrowed slices/views, fixed arrays for fixed arity, Box<[T]> for sealed
tables and Arc<[T]> for shared immutable tables. Reuse Vec/VecDeque capacity
under exclusive ownership; enforce byte/count bounds before queueing. Avoid
per-element trait objects, heap boxes and growable hot queues. Dense plan slots
may index arrays; arbitrary monotone lease IDs require a checked mapping.

## Pool accounting and retirement

- Separate physical occupancy from logical event leases. The current ledger
  subtracts bytes when a lease drops; retaining that Handle in an idle Vec
  afterward would over-admit. A backing reservation must remain charged while
  storage is idle, checked out, snapshotted, pending or quarantined.
- Reuse only after successful actual event/read retirement and the last owner
  of the old logical use. GPU completion alone does not release a snapshot or
  Pending proposal. No public put_back(handle), refcount-only completion proof
  or cancellation-as-completion shortcut.
- Pool limits derive from the whole inventory. The current .8B14-root bound
  includes live+Pending+snapshots+displaced+idle+uncertain roots;14 is not a
  shared-engine constant. Reuse does not remove worst-case retained-root bytes.
- Start with exact-layout sets of separate planes. Shared slabs/subviews and
  cross-geometry storage need their own exact alias/range/escape proof. Six
  cold geometry owners sharing one workspace lock is an opportunity, not proof
  that every retained view/Ready can alias safely.
- Checkout/backpressure must preserve fair-worker progress and cancellation.
  Do not wait for pool storage while holding the physical workspace lock or
  blocking other models' workers. Unknown work fences and retains storage.

## Transfers and boundary policy

- Unify CUDA's existing retained TensorWindow upload with the portable route;
  the latter currently copies window→Vec/Bytes and can incur another backend
  staging copy. Preserve held-file/chunk authentication and original geometry.
- Then support verified reads directly into admitted reusable pinned slots
  where supported. Verify complete integrity chunks; a small tile may need
  additional charged scratch. Expose actual pinned/heap allocation kind.
- Begin with bounded same-stream host retirement. A2/3-slot ring can limit
  runahead without a per-page CPU wait; its size must follow real resource and
  overlap evidence. Adding slots alone does not prove overlap.
- Borrow completed readback bytes for parsing/comparison. A bounded diagnostic
  frame ring needs real taps before producer storage reuse and retains exact
  node/value/root/epoch identity. Required numerical observations remain intact.
- CUDA/HIP pinned staging and Vulkan queue-write/MAP_READ behavior differ.
  Same-stream ordering and overwrite/reclamation need actual backend proofs;
  do not retag CUDA command semantics as Vulkan support. AMD uses ROCm/HIP.
- Distinguish removed Rust copies, pinned DMA, device-resident feedback and
  host-visible mappings. Avoided copies do not imply absent H2D/D2H traffic.

## Backend owners and conservative admission

Existing PhysicalServiceBudget reserves an operator-bounded provider envelope;
it does not promise exact free VRAM or a complete process peak. Keep that
contract. Client ownership already reaches the compute server and compiled
cache; the original flight/service and quarantine retain the client. Extend
that real ownership boundary rather than asking model code to price a raw
native handle or enumerate opaque driver internals.

The installed Client::memory_report covers managed pools and reported auxiliary
storage, not every compiler/module/driver allocation. Use its scope honestly.
Seal source/service-bound prepared program owners, charge known tables/buffers
and conservative native overhead in the existing envelope, then qualify actual
finite-profile capacity and retirement. A measured peak is bounded evidence,
not a universal fit proof. Preserve charges for native owners beyond model drop:
the inspected HIP context does not unload modules, including cache resets.
Actual backend lifetime/reclamation needs a shared fix before reusable model
eviction can claim to return those resources. Gemma's current source draft
refuses construction at this unresolved boundary; it has no GPU grant.

## Ordered substantial units

1. Current assembled cold-preparation batch is verified: original physical/API,
   matched .8BC1/C4/C8 and final full gate pass above.2B now has a complete
   correctness/API checkpoint plus the small C1 gap baseline.
2. Shared storage unit: root-backing accounting/pool, immutable layout/access
   sharing, reusable binding workspace and exact mutually exclusive geometry
   backing. Derive actual cell maxima/escapes/certificate lifetimes before
   choosing an overlay. Include complete prepared identity reuse where useful.
   Recalculate full2B head residency from genuine source geometry/capability,
   preserving original4GiB caps. Root pooling alone does not remove the70x gap.
3. Shared authenticated upload/readback owners and bounded staging/frame
   retirement, with borrowed completion. Prioritize removal of970 streamed-head
   waits when residency is unavailable; retain actual sources/host buffers until
   real retirement. Qualify CUDA/HIP/Vulkan independently. The portable native
   owner boundary remains necessary before whole-model admission.
4. Reuse the qualified storage/transfer APIs for Gemma/other Qwen profiles;
   improve allocator free-bin/coalescing only if residual profiles justify it.
   Fusion/graphs preserve exact source materializations, observations, current
   bindings and original flight. No inferred fit or speedup from an overlay.

Batch verification over the assembled behavior: natural/full-state correctness,
fresh and stale leases/roots, Pending/snapshot escapes, pool/ring exhaustion,
cancel/unknown completion, live+idle/quarantined accounting, actual pinned/heap
fallback, overwrite ordering and multi-model fair progress. Instrument bounded
allocation/copy/retirement counters; compare matched unprofiled timing separately
from CPU samples/device envelopes. No per-helper acceptance phases.

## Independent investigations

- [Allocation/reuse audit](../../runs/cpu-gpu-optimization-allocation-reuse-gemma-math-20261023/REPORT.md).
- [GPU–CPU boundary audit](../../../../zerogravity-worktrees/gemma4-portable-gpu-whole35-20261023/.agents/runs/model-execution-gemma4-portable-gpu-whole35-20261023/GPU_CPU_BOUNDARY_REVIEW.md).
- [Current measured ShortHF results](SHORT_HF_RESULTS.md).
- [2B original paired C1 measurements](../../../../zerogravity-worktrees/root-lowest-f32-20261020/.agents/runs/serving-and-speculation-short-hf-08-bench-live-state-20261022/prepared2b-pair9-c1-comparison.json).
- [Backend parity and qualification boundaries](BACKEND_PARITY.md).
