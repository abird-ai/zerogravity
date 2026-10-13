# Session state, prefix caching, snapshots and transactional semantics

**15 Oct 2026 — generalized hybrid prefix frontier:** [Joint scheduling/prefix contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) requires the **largest source-complete, compatible prefix across every required StateDomain**, not just token hashes or an intersection of two KV groups: dense KV, SWA, GDN/KDA/SSM, MLA/QSA/DSA indexing, mHC/MTP/DFlash, processor/media state and sampler. Cache identity binds original/derived artifact, tokenizer/template/processor, positions and numerical mode, exact committed state version and tenant ACL; restore from RAM/SSD competes economically with recompute. Missing recurrent checkpoint means shorten reuse or replay, never claim all cached tokens reusable. Forks are COW, with encoded GPU readers pinned through actual completion. SERVOPT-12..18 and 26 test N-domain cache mismatch and tenant security.


**15 October 2026 — DISK STATE IS NOT EXPERT SOURCE:** [Virtual residency/SSD contract](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) shares physical IO/BufferLease but keeps **immutable routed expert bytes** separate from **mutable, tenant-private committed KV/GDN/SWA/prefix StateDomain snapshots**. Disk KV/prefix restoration needs exact model/quant/tokenizer/source/session/epoch compatibility, integrity and optional encrypted/private persistence; saved StateRoot cannot be restored merely by mapping a GPU pointer. A disk save after committed root may append asynchronously, but cannot publish uncommitted speculative state or free in-use GPU cache. DS4/specialized-engine disk resume prior art is not proof of source-compatible constant-time GPU restore. Tests SSD-27/28; unresolved retention/consent policy explicit.


**15 Oct hybrid cache/draft addendum:** [MiMo/GLM/Qwen state contract](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) requires independent source per-layer full paged KV, sliding-window **fixed per-session rings**, GDN/KDA/MLA/mHC/indexer and trained draft caches, with generation, source position, window lookahead and rollback. SWA ring bytes are **not KV per-token**; DFlash has its own ring/hidden tap and physical charge. Complete accepted StateRoot includes draft target verifier effects, RNG/grammar/tool response, and optional MTP/eh_proj identity. No sharing mutable cache among sessions; single TaskCoordinator retains publish authority. Tests HYB-06..12 and existing snapshot/version/ABA tests qualify.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


This specification unifies paged KV, token-prefix indexes, hybrid recurrent/indexer snapshots, serialized session histories and event-safe resource ownership under one full-state transaction. **A correct prefix snapshot must represent every required state domain at the same logical frontier.**

## 1. The distinction between state and model parameters

**Immutable model lifetime:** checkpoint tensors, converted packed layouts, dense/shared weights, routed expert source metadata and lookup table sources. Multiple sessions share these. Expert residency may change physically; tensor identity and quant semantics do not.

**Mutable session lifetime:** logical token history/frontier, KV/attention caches, recurrent state/conv buffers (GDN/KDA), sparse indexer/compressor state, n-gram/Engram history, vision/audio positions, RNG/sampler, draft/MTP state and cancellation epoch. No session can mutate another session's committed buffers.

**Short-lived invocation lifetime:** prompt batch activations, quantized activation staging, scratch, provisional speculative rows, in-flight GPU/IO events, draft proposal buffers. These may be pooled only under lease/generation rules.

## 2. Session data model (suggested, not final code)

```rust
struct SessionIdentity {
    model: ModelIdentity,
    semantic_schema: u32,
    state_layout_schema: u32,
    numeric_plan_hash: [u8; 32],
    rendered_prefix_hash: [u8; 32],
    position_policy_hash: [u8; 32],
}
struct SessionFrontier {
    committed_tokens: u64,
    request_epoch: u64,
    state_generation: u64,
}
struct SessionState {
    identity: SessionIdentity,
    frontier: SessionFrontier,
    kv: Vec<KvStateHandle>,
    recurrence: Vec<RecurrentStateHandle>,
    indexers: Vec<IndexerStateHandle>,
    ngrams: Vec<NgramHistoryHandle>,
    modalities: Vec<ModalityStateHandle>,
    sampler: SamplerState,
    draft: Option<DraftStateHandle>,
}
```

Each handle owns a typed lease and an exact `version`. A `Session` controls all mutation; cache and device managers own physical allocation. A finalized prefix is immutable and reusable across sessions if and only if model/quant/pos/template/layout/numeric policies are compatible.

## 3. State domains, required saved content and hazards

| Domain | Checkpoint payload | Typical hazard |
|---|---|---|
| Dense attention KV | page table, K/V blocks, position/mask/rope metadata | stale page after prefix fork, incorrect causal mask |
| Sliding-window KV | visible-ring position and overwritten region or reconstruction plan | parent/sibling ring corruption |
| Gated DeltaNet | per-head recurrent matrix, per-session short-conv history, gates as needed | speculative rejected tail changes later logits |
| Kimi Delta Attention | KDA fine-grain state and conv, clipping/decay mode | treating KDA as scalar GDN |
| Sparse QSA | raw/pooled index keys, attention KV, selection boundaries, positions | wrong top-k after rewind, stale compressed rows |
| Sparse MLA / DSA | latent compressed KV, indexer pools/keys, masks, NoPE assumptions | expanded vs absorbed layout mismatch |
| DeepSeek CSA/HCA | raw recent window, compressed historical rows, indexer for selected layers | wrong compression ratio/history |
| N-gram/Engram | recent token IDs, hashing/convolution history and metadata | wrong lookup rows after fork |
| Multimodal | encoder embeddings or fingerprinted source, visual spans/positions/rope maps | identical text prefix but different image |
| Sampler | PRNG algorithm/state, filtering history, penalties and logit frontier | repeated/consumed RNG after rejected draft |
| MTP/DSpark draft | trained predictor hidden/cache/history + proposals | draft state ahead of target |

A state snapshot that contains just ordinary KV is invalid for GDN/KDA/QSA/MTP models. A text-rendering cache key that ignores image identity is invalid. A SHA of rendered bytes alone cannot replace checking model/quant/position/format compatibility.

## 4. Transactional generation protocol

Each externally visible action is tied to a monotonically increasing committed token frontier. Session operations:
- `begin_step(parent_frontier)`: verify generation, allocate provisional state and scratch reservations.
- `execute_target`: launch state transitions into provisional/isolated writable buffers.
- `verify`: read target logits, proposals and sampling, decide accepted prefix `a`.
- `commit(a)`: publish all state-domain handles and updated RNG/token history for position `parent+a` **atomically at session level** after requisite GPU events.
- `abort`: invalidate all provisional buffers/events, restore prior committed state or force safe replay from immutable snapshot.
- `cancel`: prevent further commits after cancellation epoch; wait for or safely abandon in-flight device events before recycling buffers.

The commit must correspond to actual verified tokens; do not claim state was updated at `p+a` if one indexer or recurrence remains at `p+k`. All committed domains must have same logical position even if they store compressed representations with different physical row counts.

### Pseudocode

```text
frontier = session.committed_frontier
tx = begin(session, frontier)      // reserves buffers + branch versions
proposals = drafter.maybe_propose(frontier, policy)
verified = target.execute(frontier, proposals, tx.provisional_state)
if error OR cancelled:
    tx.abort_and_quarantine_inflight()
    return failure
decision = sampler.verify(proposals, verified.logits, tx.rng_branch)
if !tx.can_publish_all_domains(decision.accepted_prefix):
    tx.abort()
    return safe_failure
tx.commit(decision.accepted_prefix)
emit_only_committed_tokens()
```

Speculation may verify an initial seed and several proposals, so `accepted_prefix` indexing must unambiguously state whether the target seed is included. Test lengths 0..k, stop at tool/end boundaries and eos in middle.

## 5. Snapshot and copy-on-write (COW) policy

Snapshots are **immutable references to logical state**, not necessarily eager copies of all buffers. Techniques:
- KV pages frozen with reference counts; new continuation allocates new tail page or COW on mutation.
- Recurrent dense matrices: private per-session live state, deferred snapshot materialization; partially copied state tiles if proven correct; COW or simple copy chosen based on actual cost.
- Sparse indexer: append-only committed block keys can be referenced; mutable current block/ring needs private copy and exact row counts.
- Ngram/token-history: small value copy.
- Multimodal position/fingerprint: stable immutable descriptors; large vision embeddings may reuse frozen cache under matching fingerprint and model.
- Sampler state: exact value copy before proposals, but parallel draft RNG sequence must not leak into target RNG when rejected.

Snapshot lifetime extends through consumers' in-flight GPU events, disk serialization, branch forks, cache eviction and error handling. A released parent must not invalidate a still-used child.

### ABA protection

Leases include `(allocation_id, generation)`. Replacing expert/KV/snapshot slots increments generation; async callbacks must validate this generation, not just reuse address. COW mutation prohibited while frozen readers exist unless a new physical buffer is used. GPU event completion required before recycle.

## 6. Prefix cache identity and compatibility

A prefix key must include at least:
- model checkpoint and tokenizer fingerprint (not only human model name);
- quantization artifact/layout and numerical/position policy;
- exact rendered token IDs and prefix boundary, tool/historical bytes where semantics require;
- multimodal spans plus content fingerprints/encoder config and positional transforms;
- state schema and graph/effect version, compiler kernel plan compatibility constraints where exact restoration depends on them.

Plan-specific buffers may be converted to canonical state for reuse across different kernel plans. If conversion is not validated, reject cross-plan reuse and recompute from tokens; do not silently load invalid layout. Similar quantizations may yield different logits: cross-quant prefix reuse should be off by default unless deliberately qualified.

Prefix sharing needs a radix-tree or equivalent token-prefix index and a separate storage layer for *full state snapshots*. Indexing should be bounded; scores for eviction may consider reuse hits, prefix length, recompute cost, lifetime, in-memory and disk sizes, context relation and model popularity.

A model with a 40K shared prefix and eight agents should be able to reuse recorded final recurrent state rather than replaying GDN 40K times when compatible.

## 7. Snapshot persistence schema

Proposed file format:
```text
magic + schema/version + endian + checked sizes
model/checkpoint/quant/tokenizer/semantic hashes
numeric/position policy + modality fingerprint
rendered prefix identity + token IDs + commit frontier
buffer table: state domain, layer, dtype, layout, logical shape,
              codec, bytes, offset, checksum, version
optional sampler/RNG, draft state and tool/rendering history
end checksum/AEAD if policy requires
```

Do not use unchecked on-disk pointers, file-sized allocations before validating header, unbounded nesting or host-native layout without explicit version. Use strong hash to detect corruption and prevent model-mismatch reuse. Snapshot files may contain private user prompts/tool data: secure file permissions and opt-in persistence/retention policy are mandatory. Serialize without blocking GPU mutability beyond necessary event-backed snapshot freeze.

When a snapshot schema changes, either migrate with reference tests or explicitly invalidate and rebuild from text. Do not load stale states for the sake of speed.

## 8. Prefix replay and chunk boundaries

The canonical state for prefix `tokens[0..p]` must be equivalent under:
- initial prefill in one large chunk;
- chunks of 1/256/1024/2048/uneven sizes;
- snapshot load at p;
- full replay from tokens;
- branching and editing after an earlier checkpoint;
- waiting/serving queue interleaving;
- speculative verification with all rejected tokens discarded.

Numerical tolerances or plan-specific behavior may make *bit identity* unsupported across different chunk algorithms; record and separately test same-plan exact replay and cross-plan closeness/quality. Do not claim universal bitwise equivalence where tensor reduction order changes.

### Attention state compression

Some ops retain only compressed or sliding history; snapshot must hold sufficient compressed metadata and recent raw rows. A restored state that has correct physical KV allocation but wrong causal selected-index IDs is invalid. Test just before and after 4:1, 128:1 and QSA/DSA sparsity activation boundaries.

## 9. Tiered cache and memory interaction

Global arbiter can demote prefix state GPU → host RAM → disk, but must know:
- rebuild cost from token history vs bytes copied vs disk load;
- snapshot type and compression, latent vs expanded KV;
- other sessions' COW references;
- swap/staging overhead and physical available RAM;
- interleaving with expert cache and prefill scratch.

An expert cache stores model weights; its eviction cannot corrupt user state. Prefix cache stores mutable/logical context frozen at a position; its eviction may require recomputation. They share a global budget but **must not share one naive LRU policy**.

## 10. Failure and cancellation handling

- GPU launch error: mark provisional region invalid, prevent commit, quarantine GPU buffers until event resolution or device reset; do not emit unverified text.
- IO/truncated table: fail closed; re-fetch or restart safely only when state clear and row identity preserved.
- Corrupt disk snapshot: reject, delete/mark bad after logging, rebuild from canonical token history if available; no partial restore.
- Application stream disconnect: either finish a bounded committed step or abort; never reuse session slot before quiescence.
- Mid-speculation stop/tool marker: choose valid accepted prefix before state commit, rollback remainder. Retokenization of tool rendering cannot silently alter cache frontier.
- Distributed worker loss: all ranks fail and need consistent committed prefix or replay; a local snapshot is not a valid global commit by itself.
- Resource eviction in flight: lease forbids freeing until last GPU/IO reader completes; when memory demand exceeds reservation, backpressure/reject or use smaller plan.
- Power/driver reset: skip optimistic resumption unless driver and runtime can prove state integrity; replay from durable checkpoint if compatible.

## 11. Test suite design

Property-based generated token trees (fork depth, 0..k accept), deterministic scalar oracle, random batch interleavings, stale pointer slot reuse, page eviction during prefill, snapshot serialize/load corrupted byte positions, 40K shared prefix branching, image fingerprint mismatches, exact RNG and penalties. Debug mode logs state domain version/checksum at each commit and forbids invisible writes. Compare per-layer stable intermediate/last logits with reference.

## 12. API invariants for implementers

- `commit` consumes a transaction exactly once, immutable parent remains valid for all siblings.
- All held state pointers protected by a lease whose generation matches.
- One session can't be concurrently mutated by unrelated requests; request scheduler may read immutable prefix while a sibling mutates its own branch.
- Cache hit is usable only if fully validated; partial hits must reconstruct missing domains via a deterministic costed path.
- Any optimizer-generated state representation must declare canonical conversion at every fallback/snapshot boundary.
- No speculative or failed operation may advance external token counter or visible output beyond committed state.

## 13. Unresolved choices

COW granularity per state domain, durable snapshot codec, cache index persistence, quantitative thresholds for demotion and cross-plan reuse, maximum snapshot memory tax and exact cross-backend reproducibility mode. All require benchmarks and explicit policy decision; avoid hardcoding unmeasured constants in the core architecture.

## 2026-10-10 integrated v2 addendum — generalized hybrid prefix state + research frontier mechanics

The original research's `FrontierStore`/`CommitPlan { FullDirect, Prefix(n), Replay(from) }` and bounded frontier ring are now the **physical implementation strategies within** this document's complete state transaction. They are not a replacement for SIR's effect/state contract. A prefix can be committed or shared only if *every* GDN/KDA, KV, sparse/compressed-indexer, token-history, RNG, grammar, modality and LoRA-sensitive state group has an exact compatible representation at that logical frontier.

Implement `StateGroupCodec` with `state_schema_id`, `can_resume(candidate_frontier, stored_rows)`, `snapshot`, `restore`, `recompute_from`, `required_suffix`, `finalization_alignment` and `estimated_cost`; an intersection coordinator finds **common legal prefix frontier across arbitrary groups**. A design limited to two cache groups is insufficient for arbitrary hybrid attention/recurrent/indexer models; do not assume exactly two groups. Compressors that finalize only at 2048-token boundaries may require rounding down and recomputing; never claim an illegal partial snapshot hit. Canonical token IDs/rendered prompt policy, model/quant, adapter generation, multimodal fingerprint and tenant salt participate in collision-resistant identity.

**Paged and contiguous state pools:** historical research paged `KvPool` with intrusive block metadata, chained hashes, optional radix *index* over same pool and extreme contiguous reserve-all `KvPool` are complementary. Both allocate through one ResourceArbiter and are isolated by typed state leases. They do not imply separate user model runtimes; transitions happen at safe checkpoint/replay with layout compatibility.

O(1) `restore_prefix` means logical pointer/frontier change only where physical GPU buffer state has already been safely retained. GDN/KDA dense recurrence, compressed indexer rows and COW mutations may need copy/deferred materialization or replay. Rejected drafts cannot alter grammar FSM, sampled penalties or RNG. Tool/Reasoning template and hidden-state renderer must be snapshotted where necessary. Read `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md` for session owner and `ASYNC_RESOURCE_COMMAND_IR.md` for buffer/event/commit ordering.

**Persistent cache:** do not port research's defective "48-byte" disk-KV struct, which totals 54 stated bytes and declares a 5-character magic as u16. Design a new field-by-field serialization with explicit fixed byte counts and checked lengths, model+token+tenant/version identity, cryptographic hash and no raw pointers. Include corruption/truncation/partial-write/fuzz coverage and migration or explicit invalidation. Snapshot contains private user content; opt-in persistence/private permissions remain mandatory.


## v4 refinement — authoritative prepared-root publication

**Supersedes any independent StateDomain commit wording:** an individual domain returns a `PreparedDomainVersion`; it **must not mutate the committed root visible to other readers**. A `StateTransaction` starting at `StateRoot(epoch,frontier)` checks all required domains, a common legal resume/acceptance frontier and all GPU/IO readiness events. **Only the session TaskCoordinator publishes one new StateRoot.** Logical atomic publication does not mean a remote/GPU physical memory write is atomic; private/COW state, deferred copies or validated replay preserve old roots until publication. An unready/failed device blocks or aborts all new public state, not individual domain updates.

A full prefix hit must match exact source model/quant/adapter/task/token/pos/modality/tenant and a common valid `CanResume` frontier across KV/recurrent/indexer/grammar/RNG. Uncommitted sampler histories and media states never enter a shared prefix. Persist verified schema data, not raw GPU pointers/driver events. Old root references retire only after all device/host readers' generation-tagged leases and events quiesce.

**Required test:** inject failure during second domain Prepare after first succeeds; old root still produces exactly its previous output/state, and no emitted new token/audio chunk leaks. Then test accepted speculative prefix 0..N and cancellation with stale completion.

## v4.2 — remote mutable state owner is distinct from immutable shared cache

**StatePlacement is not CachePlacement.** Every active TaskSession per-layer KV/SSM/sparse-indexer/ngram/modal state has a versioned primary shard handle, owner peer, lease/generation/placement epoch and full StateCompatKey. TaskCoordinator alone publishes a StateRoot referencing those immutable committed versions; peers execute provisional updates and return Ready before publication. A peer directory may advertise a cached model prefix or encoded feature, but remote hit is only valid after content, schema, position, quant, adapter, tenant and *all required domains* have been confirmed and safely imported/recomputed.

**Layer co-location:** ordinary pipeline stage peer owns the layer's KV/recurrent state beside layer compute. **Layer and state decoupling:** if attention runs at state owner, send query and compact result; if compute remains local, migrate compatible state once; direct per-token remote KV read is a distinct, usually expensive topology. `PartitionAlgebra` is required to shard softmax attention KV by token range; ordinary GDN/KDA and global selected indexers don't inherit this automatically. A shared prefix cache cannot mutate a live authoritative state root, and an old peer's lease cannot survive StateRoot owner migration.

**Migration/checkpoint:** at committed TaskFrontier reserve destination, transfer entire compatible state group (KV, recurrent, indexer, modality, RNG if relevant), validate and fence old generation, publish new root/placement epoch or retain old root on failure. Network partition/crash may lose volatile proposed/committed-but-not-durable work; exactly-once client delivery and transparent coordinator failover are not promised. All remote DMA/registered buffers remain leased/quarantined until safe physical quiescence. Details: [placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [remote protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md).
