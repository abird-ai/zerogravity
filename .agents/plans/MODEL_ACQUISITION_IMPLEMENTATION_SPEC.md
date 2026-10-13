# Acquisition next unit — bounded cache first, then HF/GitHub vertical slice

Status: A1 local cache implementation VERIFIED; A2/A3 remain planned. The
[A1 acceptance](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md) records actual API, source
snapshot, tests, findings and limits. This specification retains the original
full acquisition design; provider/loader/parity gates do not pass by implication.
The [accepted checkpoint](accepted-baseline/TODO.md) settles the earlier
baseline; [plan TODO](model-acquisition/TODO.md) tracks subsequent implementation and tests.

The original design below assumed no source changes at that time. A1 is now
implemented behind the optional Linux cache feature in `zerogravity-artifacts`.
No HTTP/provider, CLI pull, eviction, trained-model addition or download is
implemented by A1. The contention-first review remains complete with no LR
migration; do not put an advisory immutable index in charge of cache ownership.

## Bounded implementation sequence

A1 (now verified) in `zerogravity-artifacts`, behind an optional `cache` feature:
local import, bounded manifest/schema/path validation, private XDG cache,
content-addressed blobs, immutable complete snapshots, atomic mutable refs,
read-only offline inspect/verify and snapshot leases. No HTTP in this unit.
Use tiny local files and child-process fixtures, not new weights. Exclude eviction
until lease/accounting gates pass; return explicit Unsupported for that operation.
This is a concrete prerequisite, not complete PULL-A or upstream parity.

Immediately follow with A2: shared bounded HTTPS + native HF and GitHub resolvers,
explicit CLI pull/dry-run/offline, complete sidecar/shard closure, auth/redirect/
resume/cancel fixtures. Add A3 for acquisition-to-existing-loader and safe removal.
Keep optional cloud/OCI/an alternative model hub/Xet rows visible as remaining work, not silently
closed by the first two providers.

Ownership proposed for three parallel implementation agents AFTER cache API freeze:
1. Artifact store owner: new `src/cache/{mod,manifest,paths,store,leases}.rs`,
   cache tests and exact optional sha2/rustix dependencies. One file publication
   protocol and disk reservation authority; no inference/runtime/catalog rewrite.
2. Transfer/HF owner: new `src/acquisition/{mod,http,huggingface,resume}.rs` and
   transport fixtures. `http` exports the narrow verified stream API shared by GH;
   sole transfer owner sets TLS/auth/redirect/timeouts, not two HTTP implementations.
3. GitHub/CLI owner: new `src/acquisition/github.rs`, `zerogravity-cli/src/pull.rs`,
   command tests and facade exports. Coordinator reconciles manifests/lock/features.
   Do not parallel-edit store/protocol types; agree those cold interfaces first.

No extra crate is needed initially: optional transport dependencies fit the
existing artifact owner and preserve CPU/minimal inspection isolation. A separate
crate needs an observed platform/dependency boundary, not an architectural label.

## Cold API, authority and loader boundary

- `CacheStore::open(CacheOptions)` holds a validated root directory capability;
  options resolve absolute XDG/explicit root, quotas, lock deadlines and limits.
- `ResolvedPullPlan` has validated private fields: provider, endpoint/profile,
  immutable revision/object locator, exact relative paths, component roles,
  required/optional status, expected lengths/hashes, source-provenance strength.
  Never deserialize untrusted JSON directly into a trusted plan without validation.
- `CacheStore::begin(plan, admission) -> CacheTransaction`; stream each object
  through bounded memory + SHA256, then `commit -> PublishedSnapshot/PullReceipt`.
  A complete artifact snapshot is not ModelReady, DeviceReady or a StateRoot.
- `open_snapshot(digest) -> SnapshotLease` owns its exact parsed manifest and
  opened object handles; refs resolve once to one immutable manifest digest.
  No raw mutable path, catalog guard, credential or provider callback becomes
  lifetime/permission authority. Source publication remains the existing owner.
- `PullRequest -> resolve -> PullPlan -> execute` is explicit acquisition. Run,
  inspect and executable-catalog miss never trigger a download. Offline branches
  before provider/client/credential construction: no metadata, DNS or auth calls.
- Keep process-local SourceIdentity issued only by SourceWeights after a verified
  owned source allocation. Manifest digest is durable content identity, not that ID.
  SafetensorsIndex is metadata only; Qwen27B/Gemma contracts are not inference.
- Runtime bridge is A3, not a cache shortcut: add a descriptor-taking Artifact
  constructor plus a SourceWeights verified-owned-copy entry using the existing
  materialize/reparse path and hash the ACTUAL copied bytes against the manifest
  digest before issuance. Keep existing4GiB GGUF limit and mathematical schedule.
  Bounded config bytes must likewise be verified before the existing parser.
  Refactor one cold preparation helper if needed; do not create a second loader.
- A blob hash filename has no `.gguf` extension, while current Artifact::open
  dispatches by extension. Preserve original logical filename or pass an explicit
  checked format hint with the opened descriptor; do not reopen an unchecked path.
- Raw-path lazy catalog registration does NOT currently retain a cache lease.
  A1 offers no automatic GC claim for registered paths. A3 direct CLI cache-load
  holds a snapshot lease through the verified owned copy. Any later lazy cache
  registration needs an explicit retained lease, not a second executable catalog.

## XDG, locking and transaction details

Root: `${XDG_CACHE_HOME:-$HOME/.cache}/zerogravity/models`, ignoring relative XDG;
explicit absolute override wins. Root/attempt directories0700, records0600,
completed objects read-only by convention. Keep credentials/config outside cache.
Suggested names: `blobs/sha256/<digest>`, `manifests/<digest>.json`,
`snapshots/<digest>/`, `refs/<encoded-provider-repo-profile>/<encoded-selector>`,
`staging/<unique-attempt>/`, and stable `locks/` (not remote path strings).

Validate relative path components before allocation: no absolute/dot-dot/NUL/
backslash/alternate separator, duplicates, file-vs-directory collision or symlink
escape. Handle Unicode/case policy explicitly; Linux first, no untested Windows/
NFS guarantee. Root-relative openat/O_NOFOLLOW directory capabilities close the
check-then-open ancestor race; do not merely call canonicalize then open.
Bound JSON bytes/depth/entry counts/path bytes, per-file/aggregate u64 arithmetic
and required closure. Safetensors declared total_size is tensor payload bytes,
not the sum of shard file lengths including headers. Reuse its duplicate-key and
exact-u64 parser; validate required files separately. GGUF split filenames need
complete ordered1..N closure; acquisition still does not add split inference.

Use OS advisory file locks (Linux rustix is cached) on stable owner-private lock
inodes. Never unlink a lock inode while another process can hold/open it, which
would split the authority. Avoid holding multiple object locks; install each
verified blob independently, then serialize snapshot/ref commit. No catalog/read
guard spans disk/network/hash work. Separate long object-transfer single-flight
from short ref publication. Caller cancellation releases only its own attempt.
Use lock ownership + bounded waits, not PID-existence as crash/ABA authority.

Blob commit: stream into a same-filesystem private file; enforce bound/hash,
fsync file, install without replacing a conflicting existing object, fsync parent.
Dedup verifies existing object bytes from its opened handle, never just filename.
Build manifest and snapshot in private staging; sync files/directories, atomically
install the complete immutable snapshot, then atomically replace its small ref
and sync the ref parent. Ref readers see old complete or new complete snapshot.
Publish failure leaves the old ref exact; orphan verified blobs are not readiness.
Fault-inject every sync/link/rename and process death, then restart/validate.
Cache format/version/canonical manifest encoding must be explicit; original
config/tokenizer/license bytes are never normalized or rewritten for convenience.

Read-only files, hardlinks, mmap/MAP_PRIVATE and an open descriptor do NOT freeze
an externally writable inode. A same-UID writer can still change cache bytes.
Logical immutable naming/cooperative writer locks are not OS immutability. Verified
copied-byte digest at A3 closes the actual source-loading boundary; future mappings
need enforceable backing/leases, per the existing large-source audit. Hardlinked
snapshot aliases share blob bytes and must never be opened for writing.

Host TransferStaging uses the EXISTING process ResourceArbiter with a caller-bound
registered resource domain, bounded buffers/concurrency and explicit owner.
Do not instantiate a new arbiter per pull or invent per-request physical keys.
There is no current unified host-RAM or cross-process disk arbiter. Scoped transfer
reservations do not account all CPU models/UMA memory. State that limit plainly.
Disk quota/reservation records under cache-wide locking must include partials,
verified-but-unreferenced blobs, metadata and actual temporary/final peak; statvfs
is an observation, not reservation. Refuse unsupported cross-process quota scopes
rather than claiming the GPU ledger provides them. No unlink-based free-space
claim while descriptors/leases can retain bytes. Begin eviction only after that
separate physical-release/reservation protocol passes.

## A2 exact transport/provider contract

Use synchronous streaming HTTPS, not a new Tokio/runtime/background service.
Cached ureq3.4.2 +rustls is a candidate: defaults off (notably gzip), explicit
identity content encoding, TLS verification and capped redirects/timeouts.
Check complete selected TLS/C build closure; do not imply cached/locked means
qualified. No native OpenSSL/SDK dependency or optional network feature in core,
math, CPU/GPU or default inspection builds. sha2 0.10.9/rustix are already cached;
no dependency acquisition is needed for this audit. Keep Error ABI unchanged.

HF: resolve branch/tag once to exact commit; all listing/subfolder/filter/index/
sidecar requests use that pin. Failure to pin is an error or explicit offline pin,
not vLLM's best-effort mutable revision fallback. Discover exact requested quant;
show ambiguity/missing selection, never silently use first-file fallback. A
converted GGUF repo may lack official config/tokenizer: roles may name separate
explicitly authorized/pinned repos. Model-card URLs/config references do not grant
new authority. Optional auxiliaries stay data-only and separately pinned.

GitHub: resolve repository ref once; bound tree traversal and reject truncated
closure. Release asset ID plus expected digest (if available) binds content;
null-digest first-seen content is explicitly weaker. Git blob OID framing is not
raw SHA256. Recognize LFS pointer payload and either use a separately qualified
LFS resolver or typed Unsupported; never install it as requested model weights.

Resume: persist secret-free immutable source/validator/length/prefix metadata,
rehash saved bytes, request If-Range +Range, validate206 Content-Range start/total
and identity encoding, then append. A200 response restarts an OWNED partial or
fails, never concatenates. Changed revision/validator cannot mix snapshots. Final
full hash/size validation is mandatory. Bound retries/backoff/deadlines even for
unknown sizes; cancellation is observable between bounded reads/timeouts.

Credentials: env/profile inputs; avoid plaintext argv/receipts. Token scope is
provider+normalized origin/profile, never arbitrary redirect/mirror. No HTTPS
downgrade or forwarding Authorization across origin. Signed URL query/header
secrets must be scrubbed from errors/progress; do not persist expiring signed URLs
as object identity. Mirrors require explicit endpoint policy. Private/gated access
is not automatic license acceptance. Offline cannot check remote token revocation;
local private cached access policy must say so. No pickle/remote Python/hooks.

## Acceptance design (A1 results in model-acquisition/TODO.md; A2/A3 pending)

ACQ-A01 XDG unset/empty/relative/absolute, explicit override, permissions and roots.
ACQ-A02 duplicate/escaping/symlink-raced paths, JSON/size/count/arithmetic bounds.
ACQ-A03 tiny complete multi-file import, digest mismatch/truncation/unknown bounds;
         data receipt remains separate from executable eligibility.
ACQ-A04 every publication fault boundary + real child exit/restart: old/new/absent,
         never partial snapshot, mixed shard closure or unacknowledged ref success.
ACQ-A05 two processes same/different object and ref, dead publisher, lock-inode ABA,
         canceled waiter vs live owner, bounded waits and single logical binding.
ACQ-A06 snapshot lease through ref update and attempted removal; typed refusal
         until actual removal protocol qualifies, with accounting restored on error.
ACQ-A07 admission before body read, full disk/quota/metadata peaks and failure release;
         no fake global RAM/cross-process GPU admission claim.
ACQ-A08 real network-isolated offline hit/miss with a provider that panics if called.
ACQ-A09 default CPU/core/minimal inspect closure unchanged, optional cache builds,
         one source issuer/catalog/arbiter and no new inference family claim.
ACQ-A10 later A2 scripted+actual loopback HTTP fixtures cover redirects/auth/ranges/
         changed validators/error redaction; actual offline net isolation separate.
ACQ-A11 later A3 existing approved local .8B/2B import->same engine exact checks;
         no new checkpoint download,4GiB/sharded-runtime limits unchanged.
ACQ-A12 independent finite capability coverage audit and honest CLI supported/
         Unsupported table; no `every upstream capability` claim from A1/A2 alone.

## Inventory gaps / completion decision

The reviewed acquisition capability inventory exposes 57 capability rows and their first-unit
or deferred/loader/delegated status. None is implemented by this audit. Important
roadmap omissions/refinements: explicit HF Xet high-performance delegation,
non-weight/sparse-quant sidecars, cache-env compatibility, stable lock inode
retirement, release assets without digest, cached-token revocation semantics,
4GiB/same-descriptor verified-copy and raw-path catalog lease gaps.

Do not call57 exhaustive. Close finite frontiers before parity declaration:
all acquisition CLI/config/env/public entrypoints ->callsite/row; aliases and
intentional differences justified; selected Hub/Xet/alternative-model-hub/RunAI/serialized-tensor/
ModelExpress versions pinned and their concrete delegated API surfaces enumerated;
HTTP helper source location found (old common/http.cpp404 is not an absence proof);
source tests and provider protocol failure matrix mapped. Loader formats/plugins
remain visible in the separate execution ledger rather than invented transports.
The historical research review required no source fetch. Its raw captures are retired; the roadmap retains reference revisions and primary-source links. Revalidate source pins when closing future parity gates.
