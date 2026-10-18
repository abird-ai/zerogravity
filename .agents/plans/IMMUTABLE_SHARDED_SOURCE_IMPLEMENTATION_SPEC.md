# Large immutable source loading: bounded basic-unit proposal

Status: PARTIALLY IMPLEMENTED (18 October 2026).
The verified owned-shard source-only implementation now lives in
`crates/zerogravity-artifacts/src/verified_shards.rs` and
`source_weights.rs` with tiny original-format regression fixtures at
`crates/zerogravity-artifacts/tests/verified_shards.rs`.
The later original trained Qwen3.8 27B model, SSD streaming,
full original tensor/mathematical binding and physical-device
qualification are NOT IMPLEMENTED. See the
[model-execution TODO](model-execution/TODO.md) for accepted gates.
The historical proposal below is retained for implementation context. The [accepted checkpoint](accepted-baseline/TODO.md) settles historical verification; [plan TODO index](TODO.md) is required for implementation. Revalidate source observations against the live workspace before editing. Artifact files below belong to `crates/zerogravity-artifacts/`.

Status of original review: design only. No production change,
limit increase, dependency addition, benchmark, model acquisition or GPU work.

Propose **verified owned shards first** in the existing SourceWeights owner.
Preserve the current4GiB GGUF `open` policy. Add an explicitly budgeted, expected-
digest constructor using the same Artifact parser, SourceIdentity issuer,
TensorSource and ResourceArbiter. Do not replace the model/runtime or add a
parallel source cache. Mapped residency is a later separately qualified storage
implementation, not an implicit way around host admission.

## Findings and boundaries

- `source_weights.rs` currently materializes one GGUF descriptor into Vec and
  issues exactly one identity. It rejects Safetensors and >4GiB. Its borrowed
  bytes remain stable for the owner lifetime; SourceIdentity is allocation
  identity, not a digest or a pathname. Existing loaded-model Arc sharing stays.
- Artifact::open already parses bounded F32/F16/BF16 Safetensors directories.
  It rejects duplicate names, shape/byte mismatches, overlapping/gapped ranges
  and unindexed data. into_bounded_source reuses the original descriptor and
  reparses copied bytes. Its documentation explicitly excludes concurrent-write
  snapshot guarantees. Reuse these checks, not another Safetensors parser.
- The merged Qwen metadata unit provides a bounded SafetensorsIndex parser with
  exact integer total_size and safe basenames. Metadata contracts do not
  establish full shard/header verification; reuse the existing parser.
- Current TensorSource returns lifetime-bound slices. Any implementation must
  keep the entire backing region alive and unchanged for every borrow and
  prepared GPU owner. An evicting per-tensor buffer cache cannot implement this
  interface by returning a slice whose storage later changes.
- Native HF matrices use output-first source shapes; existing GGUF CPU and
  prepared-GPU bindings expect their own explicit columns/rows interpretation.
  BF16/F16 codecs exist, but that does not authorize reinterpreting raw HF axes,
  Qwen grouped heads or MoE layouts. Preserve original header order and add a
  checked family binding; do not silently reverse metadata or transpose bytes.
- Local read-only meminfo sample reports28,581,696KiB physical RAM. The published
  Gemma26B index declares51,611,872,412 payload bytes; Qwen27B55,562,855,904.
  Neither fits a fully owned resident copy on this local host. Available RAM,
  swap, address space and sparse file length are different resources. Refusal
  is the correct result until an appropriate admitted host/profile is available.

## Immutability decision

Opening read-only, holding a descriptor, advisory locks, permissions, inode/size/
mtime checks, MAP_PRIVATE and map_copy_read_only do not freeze another writer's
inode. Path replacement changes future opens, while in-place writes/truncation
can affect the existing descriptor/mapping. Hashing once then mapping the
mutable original remains unsafe. memmap2 explicitly makes all file-backed map
constructors unsafe for this reason, including COW maps. [Primary safety docs](https://docs.rs/memmap2/0.9.11/memmap2/struct.MmapOptions.html#safety).

For the first unit, copy from the already opened descriptor into exclusively
owned Rust bytes; compute SHA256 of those actual copied bytes, compare the
caller-authorized expected whole-shard digest and length, and reparse those same
bytes. A concurrent writer can cause rejection; it cannot change a published
Vec. Before/after stat checks are useful diagnostics, not an authenticity proof.
A self-computed digest without an expected trusted manifest identifies captured
bytes but does not establish the intended official checkpoint or a coherent
multi-file revision. Require expected config/index/shard digests for the new
verified API. Caller authorization/provenance is separate from cryptographic
integrity; a digest supplied by an attacker authenticates nothing by itself.

Future mapped options require an enforceable immutable backing and expected
identity: supported fs-verity files with verified measured digest/metadata, or
an owned backing created/copied and sealed against write/grow/shrink before any
read-only mapping. fs-verity verifies blocks and makes enabled files read-only;
memfd seals protect its own copied backing, not the mutable source file. Neither
option automatically proves expected model identity, fitting working set,
platform support or an admission policy. No seal/fs-verity activation, privilege
change or filesystem configuration is proposed here. [Kernel fs-verity](https://docs.kernel.org/filesystems/fsverity.html), [memfd semantics](https://man7.org/linux/man-pages/man2/memfd_create.2.html).

## Smallest concrete next CPU source unit

1. Add an internal immutable source-storage enum to SourceWeights: retain single
   owned bytes, add owned shard allocations. Each tensor binding stores its
   shard index and original TensorInfo. Keep metadata/private backing immutable;
   issue one existing checked SourceIdentity only after the bundle validates.
2. Add checked SourceOpenLimits/ExpectedSourceManifest and an opaque prepared
   source plan owning the opened Artifacts, expected metadata and resource
   lease. Public API shape: prepare_verified_sources(opened_shards, index,
   expected_manifest, limits, admission) -> SourceLoadPlan; then
   materialize(self) -> SourceWeights. One synchronous cold transaction.
   Do not add a session/root issuer, generic runtime or loader registry.
3. Initially accept explicitly supplied opened descriptors rather than infer
   filesystem authority from untrusted index paths. Artifact needs a narrow
   descriptor constructor (declared format; parser reused). A later path
   convenience wrapper must open relative to an owned directory capability,
   refuse symlink/magic-link/escape races, and retain those exact descriptors.
   A check-then-open canonicalize/symlink_metadata sequence is insufficient.
4. Bind every shard's actual header names to its declared index shard; reject
   missing/extra/misassigned/duplicate cross-shard tensors, unsupported dtype,
   incomplete shard sets and inconsistent expected file/payload totals. Header
   payload sums exclude file headers; whole-file lengths include them. Hashes
   cover whole files. Validate model tensor roles separately using the newly
   merged Gemma/Qwen contracts. Do not treat index names as header evidence.
5. Reserve the checked total owned payload/file allocation plus metadata/header
   and bounded I/O staging envelope BEFORE materialization. Use fallible reserve
   and a fixed read bound; no read_to_end growth beyond the admitted maximum.
   Account allocator capacity/rounding conservatively; avoid an extra full-shard
   Vec-to-box copy. Sequential per-shard filling still retains previous shards:
   peak is total retained bytes plus bounded transient work, not largest shard.
6. Reuse process_shared ResourceArbiter with an explicitly configured host-RAM
   physical key and original lease held by SourceWeights until its last Arc
   owner dies. Synchronous CPU copy does not submit a GPU event: do not fake
   Ready, fence devices, mint GPU completion or release a still-live source.
   Keeping a conservative peak envelope charged for the source lifetime is a
   bounded first implementation; do not invent partial-lease splitting now.
7. Legacy SourceWeights::open keeps its default4GiB rejection. Before claiming
   process-wide host admission, route legacy source allocations and CPU state/
   scratch/host staging into the SAME configured host domain or reserve their
   conservative existing occupancy explicitly. A new-loader-only ledger cannot
   account for older untracked allocations. UMA and discrete staging need an
   attested shared-memory-pool policy; do not invent independent full budgets
   for CPU RAM and GPU RAM when they consume the same physical pool.
8. Publication happens only after all fallible source/hash/header/index checks.
   Failure drops private buffers and synchronous reservations, exposes no
   TensorSource/cache entry and changes no session roots. Existing mutable
   `bytes(&TensorInfo)` slicing must bind the name back to this owner's exact
   stored header/shard, refusing foreign altered metadata; no ambiguous absolute
   offset may accidentally select a different shard.

New materialized Safetensors storage alone does not enable the old GGUF CPU
model constructor. Family code must bind original names/axes and reuse the
single shared mathematical engine under an explicit numerical profile. Direct
F16/BF16 decode to FP32 is not proof of a framework's BF16 intermediate rounding.
No silently selected CPU/GPU fallback or trained support label is warranted.

## File/API ownership proposal

- artifacts/src/artifact.rs: descriptor constructor and bounded exact read/hash
  seam; preserve existing parser behavior and same-descriptor tests.
- artifacts/src/source_weights.rs: owned shard backing/binding, new explicit
  constructors, TensorSource implementation and retained host lease.
- artifacts/src/sharded.rs: reuse incoming index codec, add binding verification
  beside it; do not make the metadata parser open weights.
- artifacts/Cargo.toml: exact already-locked/cached sha2=0.10.9 if approved.
  No core dependency or unsafe mmap needed for this first unit.
- CPU adapter: only explicit original-layout bindings/borrowed source access;
  preserve current source-math bodies. Family integration is a separately
  verified follow-up, not a changed generic CPU runtime.
- Core arbiter: no duplicate type/authority. Host budget setup needs an explicit
  owner/policy decision before production enabling; source unit must not pick a
  machine limit from transient MemAvailable or treat swap as physical capacity.

## Acceptance with tiny/sparse fixtures

LS-B01: two tiny BF16/F16/F32 shards; strict index/header/name/dtype/offset/shape/
payload/file/digest agreement, source bytes and decoder bits. Wrong/extra/missing
shard, foreign header, duplicate cross-shard role, wrong hash and truncated read.
LS-B02: same-length pathname replacement after open; loaded bytes come only
from original descriptors. Writer mutates/truncates during copy: reject digest
or read; writer mutates AFTER successful load: borrowed source stays bit-exact.
LS-B03: mocked/controlled allocation/read failures, concurrent admitted loads,
insufficient envelope and checked u64/usize overflow; exact charge rollback,
last-Arc source lifetime and no root/cache publication on rejection.
LS-B04: sparse >4GiB file/header plans exercise offsets/checked limits and
pre-allocation refusal without reading/hashing/materializing4GiB. This is not
an actual >4GiB load qualification. Tiny explicit-limit overrides exercise the
same positive/failure paths; default4GiB policy remains unchanged.
LS-B05: source identity unique per independently loaded bundle, shared per Arc;
old GGUF API/source bytes/math/checkpoint tests stay exact; CPU-free family
compiles and dependency graph stays SDK-free. Test symlink escapes only when
the capability-based path wrapper exists; descriptor API must not claim it.
LS-B06: later explicitly authorized checkpoint access plus sufficient attested
host budget: actual >4GiB whole-shard digests/headers, measured allocation peak
and lifecycle, then independent trained model numerical qualification. No model
is available by implication; the existing0.8B/2B GGUFs cannot prove the new large
BF16/sharded profile. Acquisition and permission remain separate prerequisites.
