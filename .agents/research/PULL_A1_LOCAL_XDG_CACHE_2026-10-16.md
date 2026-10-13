# A1 local cache — protocol and API reference

Implemented and tested in `3787519`. Current implementation/test status and
remaining acquisition work live in [model-acquisition/TODO.md](../plans/model-acquisition/TODO.md).
This file is a technical reference, not another handoff or progress checklist.

## Protocol decisions from source/design review

1. Reuse optional already-locked/cached `sha2=0.10.9` and `rustix=1.1.5`; keep
   normal core/CPU/inspection dependencies unchanged. Linux capability-relative
   `openat`/NOFOLLOW operations protect every directory component. Read-only
   handles and cooperative immutability do not freeze same-UID external writes.
   A3 must hash actual owned source-copy bytes before SourceIdentity issuance.
2. Transfer admission uses a caller-supplied existing ResourceArbiter, registered
   domain and owner. Admit bounded staging before allocation/body reads; never
   instantiate or register another arbiter inside a pull. This does not establish
   global host/UMA or cross-process memory admission.
3. Disk quota covers conservative cooperative cache-owned usage and live-attempt
   reservations, including partials, orphan verified blobs and bounded metadata/
   allocation allowances. Persist root policy and refuse conflicting limits.
   Filesystem free space is an observation, not a reservation; unrelated writers
   are outside that accounting scope. Inject ENOSPC rather than filling the host.
4. Before atomic ref rename, a failed transaction leaves the previous binding
   exact. If rename succeeds but parent-directory sync fails, rollback cannot
   establish the old durable state. Return a cache-specific explicit
   DurabilityUnknown outcome containing the visible complete binding and cause.
   Never return ordinary success or pretend nothing was published. Child crash
   recovery may observe an old or new complete snapshot, never a partial closure.
   Existing core Error and GPU actualReady semantics remain unchanged.
5. Stable lock inodes are never unlinked/recreated while another process can
   hold or open them. Attempt-lock ownership, not PID existence, establishes
   whether a reservation is abandoned. Bound lock/metadata growth. Long object
   transfers do not hold the short global publication lock.
6. A1 has no HTTP/provider interface. Exercise actual network-isolated local
   hit/miss/import/inspect/verify and audit the selected dependency closure.
   The provider-constructor panic subfixture belongs to A2 when that interface
   exists; do not invent a provider runtime to manufacture this test result.
7. Snapshot leases retain validated metadata and the exact opened object handles.
   Removal/eviction remains typed Unsupported, including while leases exist.
   Artifact publication is distinct from executable eligibility or ModelReady.

## Public API and bounds

The existing artifact owner exports Linux-only `cache` and explicit
`diagnostic-cache` features. Use `CacheOptions::new(root, disk_limit_bytes)`,
`CacheStore::open`, `begin(manifest, &TransferAdmission, cancel)`,
`import_object`, then `commit(reference)`. `CommitOutcome` distinguishes Durable
from DurabilityUnknown. `resolve_ref`, `open_snapshot` and `verify_snapshot`
return a SnapshotLease retaining validated metadata and exact read-only files.
`usage`, `root_identity` and `filesystem_profile` expose bounded observations;
`remove_snapshot` returns Unsupported. No new public runtime/authority is added.

Defaults:1MiB manifest (8MiB hard ceiling),4096 files,1TiB/file,4TiB aggregate,
JSON depth8,64KiB transfer buffer and30-second lock deadline. These logical
artifact limits do not raise the existing4GiB executable-source limit. Paths
are unchanged case-sensitive UTF8 bytes, <=4096 bytes/255 per component/64
components; no case/Unicode normalization. Inventory is capped at131072 nodes
and8MiB retained path characters. Transfer admission reserves64MiB control
headroom +8×configured manifest bound +transfer buffer +64KiB verification
buffer before begin's control copies, transfer allocation or body read.
Filesystem block/fragment units must be nonzero powers of two <=4096 bytes;
other allocation profiles are explicitly Unsupported.

## Important implementation limits

- Disk accounting covers cooperating cache writers, partials/orphans/metadata
  and live future reservations; it does not reserve filesystem-wide capacity.
- An existing final snapshot cannot consume the reservation for a new private
  snapshot rebuild. Reference versions retain named backing even after alias
  replacement and on identical-record dedup, keeping live bytes charged.
- Verification hashes bounded bytes and checks length/inode/mtime. Hardlink-only
  ctime changes are allowed; same-UID external payload writes remain possible.
  A3 must hash actual owned source-copy bytes before source identity issuance.
- Publication tests covered18 operation types/38 callbacks with76 injected I/O
  and76 actual child-exit/restart cases. They establish same-kernel process
  recovery, not power-loss durability. Initial root policy creation was outside
  fault injection. Post-ref-rename sync uncertainty is explicitly reported.
- Entry/import cancellation and a genuinely held publication-lock deadline are
  tested. Cancellation inside an observed EWOULDBLOCK iteration was not claimed.
- A1 has no HTTP/provider/CLI pull/eviction/loading; cache data is not ModelReady,
  DeviceReady or StateRoot publication. No new checkpoint/SDK was downloaded and
  no GPU was launched. Actual network-isolated local hit/miss passed; provider
  constructor/credential offline fixtures belong to A2.
