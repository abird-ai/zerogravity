# Canonical Sourcebook — volume 5

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/plans/model-acquisition/README.md

Original UTF-8 bytes: 95268; SHA256 `b7c16440a4e8882b1c9e15a5ec2892eaf0c21b358e671b6f3bf9bd2175654fb4`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: GPT-6

# Model acquisition

Explicitly acquire bounded, pinned source files into the existing XDG cache.
This is data acquisition, separate from loading a model or granting execution
support. Implementation and tests are tracked in [TODO.md](TODO.md); the
[full acquisition specification](../MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md)
and [provider parity research](../../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md)
retain the broader scope.

## Design

One optional synchronous HTTPS owner serves the provider resolvers. Native Rustls TLS
verification, origin-scoped credentials, bounded metadata/headers/redirects and
identity encoding protect exact transferred bytes. Default CPU, core and source
inspection builds keep the transport dependency closure disabled.

ureq's TRACE formatting exposes signed URL queries and redirect Location values.
Selecting acquisition therefore caps the process-wide Rust `log` facade at DEBUG
at compile time, including when a runtime logger requests TRACE. Default builds
are unaffected; `tracing` and the engine's explicit JSON diagnostics are separate.
Downstream builds cannot select a conflicting `log/max_level_*` feature. This
explicit dependency constraint avoids silently leaking credentials through a
host application's logger; it can be revisited when upstream safely redacts TRACE.

Resolution uses a caller-owned transfer admission, pins a repository commit,
release asset, versioned object or explicit content contract, validates file
paths and selects only the applicable declared closure.
The returned plan has private validated fields; expiring download URLs stay in
memory. It grants neither SourceIdentity nor model/device readiness.

The existing CacheStore admits exact planned lengths or explicit full reservation
caps before object requests. Its acquisition transaction streams each file once,
checks the actual raw digest and preserves TrustedSha256 versus FirstSeen
provenance in the manifest. Actual
entries replace private accounting placeholders before the original snapshot/ref
publisher can run. Existing leases, stable locks, failure outcomes and quota
ownership remain authoritative; there is no second cache or allocator.

Durable partial recovery uses that same attempt owner. Only a synced proper
prefix (shorter than the complete object) can be saved, bound to immutable
source/object identity, exact total, strong ETag,
prefix digest and inode identity. Recovery rehashes and copies those bytes into
an exclusively created file under a new full-peak admission. It never reopens a
shared cache inode for writing. An exact206 response must match validator,
offset, total and identity encoding before append;200 or changed metadata refuse.
Even a cancellation at the final byte cannot bypass the final HTTP EOF check.

Checkpoints batch after8MiB or1second of accepted progress, plus interrupted-read
or cancellation cleanup. File sync precedes atomic record replacement and its
parent sync. Uncheckpointed crash tails remain charged but are not resumed.
Unchanged resumed prefixes create no second retained checkpoint. Once additional
progress is checkpointed, old retained attempts remain named and physically
charged; automatic reclamation awaits the separate removal protocol. Recovery
therefore needs space for the retained old files plus the new full transfer peak,
including24KiB/file for partial control records/lock/replacement. Repeated genuine
interruptions can consume quota; no global free-space or reclamation claim follows.

The stable partial guard serializes one source object's transfer at a time.
For a trusted checksum it also holds the original digest lock through staging,
so another publisher cannot create a named blob that falsely credits a private
copy's reservation. An already verified named object refuses the partial-write
API before file creation; callers use try_cached instead. Publication and digest
dedup remain A1's authority. The executor rechecks a known digest after acquiring
those locks. A concurrent fill returns a typed verified cached result before a
new private payload or HTTP
body is requested. It is counted as cached progress, with the same prefix/LFS
checks and original transaction completion. The legacy partial-write API still
refuses a known object; callers can explicitly select the cached-or-partial API.
Unknown-digest first-seen transfers do not gain a network single-flight guarantee.

## Decisions

- Pulls are explicit. Ordinary run, inspect and catalog misses do not acquire.
- Offline commands take the local branch before creating clients or credentials.
  They do not establish remote freshness or token-revocation state.
- Tokens come from the selected environment profile, never command-line values,
  cache manifests, progress text or upstream error strings. No remote code runs.
- Source-provided SHA256 and first-seen local content hashes have different trust
  strength. Neither automatically supplies a model license or execution contract.
- Exact lengths remain the default. Omitted listing lengths first use pinned LFS
  metadata or bounded body-free HEAD; explicit capped auxiliary streaming is
  described below. Chunked bodies can transfer against an established exact
  length or authorized cap. Host admission covers this process's transfer work,
  and disk quotas cover cooperating cache writers, not global OS resources.
- Linux local-filesystem semantics are qualified first. Open/read-only cache
  descriptors do not prevent arbitrary same-UID mutation; A3 must hash the bytes
  of its actual owned source copy before the existing source issuer runs.

## Interfaces and remaining scope

The optional `acquisition` feature exposes the library facade and explicit CLI
pull/cache commands. `diagnostic-acquisition` adds controlled fixture endpoints
and trust roots; it is not a production insecure-TLS option. Current supported
commands and limits are shown by `zerogravity pull --help` in that build.

Ordinary pulls retain strict same-repository model closure. Explicit component
plans can supply config/tokenizer from separate authorized sources. Each member
retains its own original provider/repository/immutable revision and manifest;
there is no merged-source manifest, composite ref or cross-source transaction.
Git LFS pointer payloads refuse. Lengths still unknown after bounded recovery
require the explicit auxiliary-only cap below; other unknown lengths refuse.
Explicit cache removal has the separate protected-reader contract below.
Acquiring an auxiliary file or unqualified model format grants no loader support.
The wider Hub/cloud/OCI/Xet/parity inventory remains separate unchecked work.

A borrowed source-bound progress callback extends the existing
executor, preserving its original entry point. Logical file/aggregate bytes count
accepted streamed bytes, verified cache hits and rehashed resumed prefixes;
cached/resumed bytes are distinct. Exact totals are optional and reservation
maxima are separate; a cap is never displayed as an observed or exact total.
A final file_complete event follows verified body/cache completion, including
when its accepted-byte count repeats the prior event. It is not publication.
The CLI throttles numeric progress on stderr; its final receipt stays on stdout.
Offline HF filters reuse the same bounded initial selector against verified
cached paths, before credentials/client/provider creation. This establishes no
remote freshness or additional loader eligibility.

Online CLI pulls register safe optional Linux signal-hook flags for SIGINT and
SIGTERM and unregister their callbacks on return. No handler is registered for
help/offline/cache commands or default CPU builds. signal-hook leaves its OS
dispatcher installed after callbacks are removed; this standalone single-command
CLI exits afterward rather than claiming restoration of prior dispositions.
Blocking HTTP reads remain bounded by the configured whole-request deadline;
cancellation is checked between reads and at deadlines. Online acquisition
threads that same flag through cached-object verification before/after each
64 KiB read, EOF and digest finalization. Additive cancellable snapshot/ref APIs
allow an existing caller flag to cover final verification. Legacy convenience
APIs without a cancellation input retain their existing behavior. Parsing,
inventory, filesystem syncs and blocked reads do not gain a real-time
cancellation guarantee.

## Metadata-only candidate discovery

`pull candidates hf|github OWNER/REPO --revision REV [--subfolder PATH]`
lists regular file choices from one exact pinned inventory. It does not select a
model, read an index or payload, probe lengths, reserve payload bytes or publish
a snapshot. Multiple GGUF families are useful discovery results rather than an
ambiguity error. Complete regular-file inventory validation precedes subfolder
projection, so a hidden regular ancestor cannot conceal a file/directory collision.

The additive `CandidateRequest`, `discover_candidates` and
`inspect_cached_candidates` APIs reuse the original provider pin/tree owner.
Metadata admission, aggregate control/file/path/body bounds and a cumulative
logical-call limit apply before any result. Existing HTTP independently bounds
physical retries and deadlines. Existing source identity, pull selection and model
closure remain with their original owners. Listed sizes may exceed transfer caps;
actual acquisition enforces its own file, snapshot and memory limits.

Results identify `pinned-remote-tree` or `verified-cached-subset` coverage. Each
repository-relative path has an optional size, filename role hint and optional
expected raw SHA. GitHub sizes describe stored Git blobs, including possible LFS
pointers; HF sizes describe tree/LFS metadata. Cached sizes describe fully verified
retained content. Unknown sizes stay unknown, Git object IDs are never raw SHA256,
and cached FirstSeen content is not promoted to a trusted expected hash. The JSON
`size_basis` makes those distinctions explicit.

Online credentials use only the existing origin-scoped `HF_TOKEN` or
`GITHUB_TOKEN`. Offline requires `--offline --snapshot DIGEST_OR_REF`, optionally
an existing explicit cache root. Source, current bounds, full managed hashes and
pointer policy are checked before credentials, clients, admission or signals.
Cached results describe retained data only; they do not prove a full remote tree,
current access, freshness, quantization, complete model closure or native execution.
No new endpoint, mutable publication, automatic selection or fallback is introduced.

## Explicit GGUF filename quant selection

`pull hf|github OWNER/REPO --revision REV --quant TAG [--subfolder PATH]`
chooses one complete GGUF filename family. Native `pull xet` accepts the same
explicit choice through its original HF inventory and discovery owner. `--quant`
and `--file` or release-asset selection are mutually exclusive. There is no default
tag, preferred model or fallback. A tag is only a filename label; it does not
establish tensor encoding, model eligibility or inference support.

Tags contain1..32ASCII bytes: a letter followed by letters, digits or underscores,
normalized to uppercase. Remove the exact lowercase `.gguf` suffix and a canonical
`-NNNNN-of-NNNNN` split trailer. The remaining basename must equal the tag or end
in `-TAG` or `.TAG`, with ASCII case-insensitive tag comparison. Original path and
family identity retain exact case. Underscores belong inside tags, so `Q4` does
not match `Q4_K_M`. Directory names never supply a tag or split trailer. Other
filename conventions remain available through explicit file selection.

Every matching split cohort must contain exactly1..N using the original strict
split parser. Missing/malformed/conflicting-total or nonregular members refuse;
a complete family cannot conceal an incomplete matching cohort. Singles, splits,
case variants and different model prefixes remain distinct families. Zero matches
or multiple families refuse and direct the caller to candidate discovery or an
explicit file; no second lookup chooses an alternative.

The additive private-field `GgufQuantRequest` and `quant::resolve` select inside the
same retained provider inventory, before original model closure, exact sizing and
source construction. One admitted metadata envelope and one logical-call budget
cover pin/tree/HEAD/index/native discovery; original HTTP independently owns
physical retries. Public `Selection` and `PullRequest` retain their original
shape. One shared CLI fetch path retains credentials, cancellation, progress,
transaction admission and typed publication completion.

New quant offline mode requires `--offline --ref SNAPSHOT_DIGEST`. Fully verified
managed bytes, exact source/current bounds, pointer refusal, complete selected
closure and required Weights/Config/Tokenizer roles are checked before environment,
client, admission or signal handling. The result describes the retained snapshot
and validates the requested family within it; other retained data is not removed.
It does not prove remote-global uniqueness, access, freshness or native execution.
Legacy Files offline retains its original selection and role semantics. Native
quant's new focused fixture qualifies inventory/announcement/plan selection;
existing native CAS execution gates remain separate from that fixture.

Remaining convenience work has explicit authority boundaries. Required sidecars
need family/format-backed schemas and positive, missing and collision tests;
config/tokenizer closure does not establish vocab/merges/templates, adapters or
sparse-quant support. Any first preset should be caller-selected bounded local
data mapped to existing component requests, with duplicate/unknown-field refusal.
It cannot add URLs, repositories, credentials, limits, scripts or serving options.
Mirror/environment compatibility needs a documented precedence table and an
explicit origin-bound credential/source profile before new aliases are consulted.
Observed401/403/404 cannot establish license acceptance, gating or model absence;
private services may mask access. No automatic login or license acceptance follows.

## Explicit full-path Unicode filters

An additive `FilterRequest` provides full-path, case-exact `FnmatchCaseV1`
selection on pinned HF/GitHub repositories and explicit native HF Xet. CLI:
`--include GLOB` (repeatable, at least one), optionally `--exclude GLOB`;
or `--quant TAG --exclude GLOB`. Filters conflict with legacy `--file`,
release assets and capped unknown lengths. Quant plus include is not qualified.
Other providers/components are unchanged.

The grammar follows Unicode scalar `fnmatchcase`: `*` and `?` may cross `/`,
brackets express scalar classes/ranges/negation, malformed `[` is literal,
and backslash does not escape. No case/Unicode normalization. Subfolder scope
is a literal directory-component boundary; match the suffix without treating
characters in the folder name as patterns. Legacy `--file` keeps its original
component-local byte `*`/`?` semantics.

Order: include union, exclude union, **then** family ambiguity and original
model closure. Excluding an entire competing family may disambiguate;
excluding a required index, shard, config or tokenizer refuses. Adjacent
indexes that original closure would inspect are vetoed before the index GET.
Optional README/license/tokenizer companions may be omitted. No silent
family/quant fallback, second pin/tree or alternative publisher is introduced.

Bounds: 4096 patterns of at most 4096 UTF-8 bytes each, at most 8MiB aggregate
pattern plus inventory path bytes, and one shared 16Mi matcher-work budget
with periodic cancellation. Nonrecursive two-row matching compiles one
pattern at a time; at most 512KiB scratch and extra 16MiB resolve-only control
reservation use the original TransferAdmission. Ordinary 128MiB and native
256MiB CLI defaults remain unchanged; smaller limits refuse before HTTP.
These are scoped buffers, not all-process-memory telemetry.

Offline filters require an exact snapshot digest and validate original source,
current limits, managed byte hashes, pointer refusal, required roles and exact
retained closed-snapshot equality before credentials/client/admission/signals.
They neither remove excluded retained bytes nor fabricate filtered snapshots.
Cached subsets cannot prove remote-global uniqueness, access/freshness or
native execution. Original `--file` and bare `--quant` semantics are retained.
The test suite includes frozen independent CPython-derived differential cases,
required/index veto, offline roles, actual private cache publication and
native announcement. See the live TODO for final acceptance status; filtering
does not grant new model loader or inference support.

## Exact length recovery

Resolution first uses valid size metadata for the immutable selected object.
HF LFS metadata can supply a missing tree length while retaining its original
SHA256 trust; contradictory lengths refuse. A selected/required file whose size
is still absent can use HEAD on its exact pinned HF commit, GitHub raw commit or
release-asset-ID URL. Unselected files require no speculative probe. One shared
HTTP owner retains TLS, origin credentials, redirect policy, identity encoding,
request deadlines/retries and error redaction. Custom transports that do not
implement HEAD return Unsupported; there is no fallback GET for size discovery.

An exact probe must return HTTP200 with a valid Content-Length within the file
limit. Missing lengths refuse by default. Overflowing, contradictory, encoded
or ranged/ambiguous framing always refuses. No response payload is read; dry-run
remains metadata-only. Probe counts are bounded per source by max_pages independently of listing-page counts.
Component GET and HEAD calls also share the set's single max_pages request
budget. Metadata body limits keep their existing separate authority.
The resulting exact lengths enter the original full-peak cache admission; no
zero placeholder, guessed total or unreserved growing payload is introduced.

A strong probe ETag, when supplied, must match the initial transfer and applicable
range responses. Weak/absent validators grant no extra resume authority. ETags
and Git object IDs do not become raw SHA256, and recovery does not strengthen
first-seen content provenance. Final original raw digest, exact bytes and terminal
EOF checks remain mandatory. The optional bounded path below retains these
source and integrity boundaries.

## Explicit capped auxiliary streaming

`--unknown-size-cap-bytes N` authorizes a positive per-file reservation cap,
no greater than `--max-file-bytes`. It applies only when pinned auxiliary files
still lack lengths after normal metadata and HEAD recovery. Weights, indexes,
config, tokenizer and license objects keep their exact-length requirement and
ordinary required-file closure. Model/index selections keep all their files
exact, including auxiliary closure members. The flag is rejected with offline
mode before cache, plan-file or credential work. Components use the same global cap without
changing their JSON schema or independent source identities.

The fallback is deliberately narrow: a validated HTTP200 HEAD with no length,
or HTTP405/501 indicating unsupported HEAD. Other status, transport, TLS,
malformed metadata, cancellation and budget failures do not authorize streaming.
A strong successful HEAD ETag still binds GET. Validators and lengths on405/501
describe the error representation and grant no object authority. HEAD never
reads a body; dry-run stays metadata-only.

The shared ByteLength distinguishes Exact from AtMost. Exact constructors retain
their behavior; explicit new_with_length constructors authorize a bounded entry.
Provider resolvers additionally require PullLimits.unknown_length_cap; a trusted
caller constructing a resolved plan directly supplies that authority through
AtMost itself. Resolved-file bytes(), plan/component total_bytes() and progress
exact-total getters now return Option rather than treating caps as exact lengths.
Callers needing admission bounds use length()/maximum_bytes() and the explicit
progress maximum getters. ManifestEntry and published manifest byte counts remain
exact. This is a source API change in the optional acquisition interface.

The existing attempt reserves full maxima before any payload I/O. Same trusted
digests with unequal caps reserve their maximum once; unrelated first-seen files
reserve independently. A private one-shot admission input feeds the original
reservation ledger and sole installer/publisher. Final manifest entries contain
only observed bytes and the actual raw SHA256. A known-digest hit verifies the
original opened file's actual length against the bound and hashes those actual
bytes; an insufficient cap or corrupt hit refuses before HTTP/private payload
staging. The admitted attempt and reservation already exist at that lookup.

The original partial guard performs ephemeral capped transfers without recovery,
saved partial validators or durable checkpoints. Its explicit finish_after_eof
requires the caller to have proved source completion. Cancellation, errors and saved bytes
never imply EOF. The reader consumes at most the remaining bound plus one byte
for overflow detection, including a nonempty sentinel read at the cap. Empty
and short payloads are valid after EOF and hash verification; LFS pointer bodies
still refuse before accepted progress. Existing exact resumable transfers retain
their stricter total/range/checkpoint contract.

For capped HTTP GET, require200, identity encoding, no Content-Range, and either
a valid Content-Length within the cap or sole chunked transfer framing. Conflicting
or stacked framing and ambiguous close-delimited bodies refuse. A declared GET
length must still be reached exactly, including for custom transports. The single
existing ureq decoder determines payload completion: the pinned version accepts
a zero-sized chunk line followed by connection close without independently
validating a complete trailer section. Tests retain this explicit boundary;
trailer metadata is never source or checksum authority. Missing zero chunks,
truncated data chunks and short declared lengths fail. No duplicate HTTP parser
or silent completion-on-read-error path is introduced.

Progress and final receipts count actual accepted/cached bytes. Reaching a cap
alone is not completion; file_complete follows the verified terminal read and
cache guard finish. Complete component receipts still require every separate
publication and descriptor verification. A later failed member can leave earlier
immutable data; no cross-source atomic publication is asserted. Unknown-length
required model objects and resumable capped transfers remain separate work.

## Explicit protected-cache removal

Fresh empty roots use persistent policy v2. New code still opens existing v1
roots without changing their policy bytes, but refuses their reclamation. Old
CacheStore openers reject v2 before they can open unprotected managed readers. A
policy-less root with existing data cannot adopt v2. There is no automatic or live migration.

Every managed regular-file open acquires a shared Linux flock on that exact open
file description and verifies its bound inode. File cloning, dup and inherited
child handles preserve that pin until their final close; wrapper lifetime alone
is insufficient. No explicit unlock runs on wrapper drop. Lazy Artifact readers
constructed from managed descriptor clones retain the same pin. Advisory locks
cover cooperating cache users. Generic local-path run/inspect, arbitrary raw
File::open, mappings or deliberate descriptor unlocks do not join this protocol;
raw access to cache-owned paths requires external quiescence during removal.
Use run-cache or the managed SnapshotLease API for concurrent loading.

In protected roots, a stable shared activity lock spans each complete transaction
and each snapshot or reference binding window. Legacy roots do not gain this
extra charged control inode. Maintenance takes activity exclusively, without
waiting, before publication. Active transfers therefore return Busy, including
first-seen/deduplicated blobs missing from planned checksum reservations. It
never waits on object or attempt locks under publication. Recovered partial
payloads close before their old attempt owner, and transaction cleanup finishes
before activity is released. Other metadata inventory operations hold publication
until their temporary descriptors close.

Under those existing authorities, one bounded graph captures current refs and
conservative shared-blob reachability through other manifests. Referenced targets
refuse. Stable locks, reference-record backing, dead/retained staging and general
orphans are retained. Maintenance probes separate actual candidate descriptors
with nonblocking exclusive locks and validates names/inodes; escaped readers
produce Busy before any target mutation. All local collector payload/control/tree
handles close before the final unlink. The scope is one exact target, with no
new cache owner, collector daemon or reservation ledger.

A fresh retirement needs the existing conservative 256 KiB control allowance to
fit under the cooperative disk policy. Dry-run and apply both check that
headroom; a full root can refuse without mutation. Activity and publication
exclude competing admissions throughout this bounded operation.

Apply atomically moves the snapshot into retirement/DIGEST and syncs the affected
parents before destructive cleanup. The moved manifest and then an empty retired
directory preserve exact retry through partial cleanup. A pending retirement
blocks publication of that same snapshot. Every mutation and directory sync has
diagnostic fault points. Errors after mutation return Incomplete, never implied
rollback. An already absent exact target is an idempotent zero-removal result;
apply syncs its parents, without collecting unrelated orphan manifests or blobs.

ReclaimReport counts named entries, including hard links, and reports rebuilt
unique-inode conservative cache charge before and after. An unavailable post-error
inventory is unknown. This is neither a physical-block-release attestation nor a
filesystem free-space reservation. Concurrent process creation can briefly inherit
collector descriptors until exec closes them, and kernel close effects can lag;
local close is not a process-global last-close attestation. An incomplete removal
can retain charged names; no optimistic release credit follows from a wrapper or
namespace change.

The independent cache-only CLI requires an existing explicit absolute root and
one mode:

```text
zerogravity cache remove DIGEST --dry-run --cache /absolute/cache
zerogravity cache remove DIGEST --apply --cache /absolute/cache
```

Use the root's existing disk-policy limit with --disk-limit-bytes when it differs
from the CLI default. No aliases, force option, hidden reference deletion,
automatic eviction, dead-stage sweep, credentials, transport or signal handlers
are involved. Dry-run observes eligibility and is not a reservation; apply always
revalidates. Tests and acceptance remain in TODO.md. Implementation fixtures use
private tiny caches; no live user cache is deleted or migrated by this work.

## Explicit components

`pull components PLAN.json` accepts the fixed `zerogravity.pull.components.v1`
schema, bounded to 1 MiB and 16 components. Each component declares a unique label,
role (`weights`, `config`, `tokenizer`, `auxiliary`), provider (`huggingface` or
`github`), repository and either revision/files/optional subfolder or release/asset.
Unknown/duplicate fields, URL/token fields and conflicting selectors refuse.
For example, independently authorized original files can be selected with:

```json
{
  "schema": "zerogravity.pull.components.v1",
  "components": [
    {"label":"weights","role":"weights","provider":"huggingface","repository":"owner/converted-model","revision":"main","files":["model.gguf"]},
    {"label":"config","role":"config","provider":"huggingface","repository":"owner/original-model","revision":"main","files":["config.json"]},
    {"label":"tokenizer","role":"tokenizer","provider":"huggingface","repository":"owner/original-model","revision":"main","files":["tokenizer.json"]}
  ]
}
```

These repository names illustrate the schema; they do not select an approved
source or imply format/model eligibility. Resolution pins aliases before payload
transfer. Repeated file-source provider/repository/requested-revision tuples reuse
the first exact commit, preventing alias movement between components. Repeated
GitHub release selections reuse the first validated release ID for the same
literal provider/repository/requested-tag tuple. Later components select assets
from that release ID through their own label-scoped transport, without resolving
the tag again. Separate file and release pin maps prevent equal selector strings
from crossing locator kinds. Ordinary single-asset resolution keeps its existing
behavior; no synthetic tag or public pin argument is introduced.

Every label still lists and validates assets through its own credentials and the
shared metadata/request budget. Listings retain path/name/ID, pagination and
count checks; the selected asset must be uploaded and satisfy digest validation
before supplying an asset-ID download URL. Each selected asset retains its original release-ID/asset-ID source
locator, exact or explicitly capped length and TrustedSha256/FirstSeen provenance.
A stable release ID does not freeze its mutable asset set: re-listing can observe
a renamed, missing or replaced asset, and identity/digest checks apply to the
actually selected object. This does not establish a cross-asset atomic snapshot.
Only an explicitly declared config/tokenizer supplier relaxes the corresponding
missing weight-repository sidecar; shard/index/split closure remains strict.

Actual metadata bytes and requests share aggregate budgets across the set, as do
resolved file counts, payload bytes and plan metadata. The caller's existing
transfer admission covers resolution and the same A1 executor handles each source.
Credential clients are bound to explicit labels and provider origins. A final
component receipt requires every individual publication to be durable and its
original snapshot descriptors verified. Earlier complete immutable snapshots can
remain after a later failure; no mutable reference is changed by component pulls.

`--dry-run` resolves metadata only. `--offline` requires an explicit `snapshot`
selector on every component and verifies the complete local set before any
credentials/client/provider work. Online plans reject those offline selectors.
The pure library `open_components` takes no transport. A shared canonical locator
parser requires positive decimal release and asset IDs without alternate spellings.
Repeated release bindings must agree on the release ID for the requested source
while permitting different asset IDs; repeated file bindings must still agree on
their exact commit. Offline verification cannot prove mutable selector freshness,
remote asset membership, token revocation, model compatibility or licensing.

Legacy HF/GitHub offline selection keeps its original data-only initial selection:
file roles and complete model closure are not inferred for ordinary cached data.
It now checks every retained managed descriptor for LFS pointer payloads after
original full cache verification. Generic CacheStore integrity alone grants no
acquisition eligibility. Explicit component weights additionally reuse the same
strict split/index closure as online resolution; independent config/tokenizer
suppliers relax only those sidecars. Reading retained indexes shares one cumulative
metadata-byte and index-request budget across the complete component set. These
checks use existing descriptors and parsers, with no transport or second cache.

## Explicit HTTPS objects

`acquisition::object::HttpsObjectSpec` describes one caller-authorized object:
a strict bare origin, logical path, role, exact byte count (including zero) and
raw SHA256. It stores no download URL. Its admitted `prepare(url, ...)` performs
local validation and returns the original `ResolvedPullPlan`; the existing
executor owns transfer, resume, hashing, staging and publication.

```text
zerogravity pull object --origin https://objects.example \
  --file payload.gguf --role weights --bytes 123 --sha256 EXPECTED_SHA256 \
  --url-file /absolute/private-url.txt --dry-run --cache /absolute/cache
```

The example selects no approved source. Online defaults to apply; omit
`--dry-run` only for an authorized fetch. `--url-file -` explicitly reads stdin.
URL files use no-follow, nonblocking opens, must be regular files and fit16KiB;
one final LF/CRLF is accepted. Other controls, whitespace, userinfo and fragments
refuse. Input errors omit the contents. Explicit stdin can block, and intermediate
path components are not a directory-capability ownership proof. No automatic
credential-file or source discovery occurs.

The expected origin is supplied separately and normalized by the existing URL
parser. The actual URL must match it before client/credential construction.
Production requires HTTPS; diagnostic HTTP remains literal-loopback only. Signed
URL paths and queries stay in memory within the existing redacted plan; changing
a signed locator on the same origin does not change the content contract.
Optional `--token-env NAME` reads only that named environment variable during
online execution and scopes the bearer to the expected origin. Existing redirect
stripping, TLS, framing, deadline, cancellation and progress rules apply.

The manifest provider is `https-object`. Its repository identity hashes the
versioned canonical origin; its revision hashes the versioned complete tuple of
origin, logical path, role, exact length and caller raw digest. Receipts/manifests
and Debug do not contain object URL paths/queries or credential values. The raw
checksum is independently caller-authorized, not inferred from an ETag, Git ID
or URL; normal raw-byte verification records TrustedSha256. These source labels
identify the declared data contract, not remote repository membership or a model
SourceIdentity. This first object frontend publishes no mutable reference.

`--dry-run` validates the caller contract and URL locally, without credentials,
client, remote metadata, payload transfer or cache initialization. Printed sizes
are caller-supplied exact bounds, not observations of a remote object. Preparation
admits1MiB of bounded operation-local control work; it does not retain a transfer
or TLS/global-memory reservation after returning the plan.

Offline needs no URL input at all:

```text
zerogravity pull object --origin https://objects.example \
  --file payload.gguf --role weights --bytes 123 --sha256 EXPECTED_SHA256 \
  --offline --snapshot SNAPSHOT_DIGEST --cache /absolute/cache
```

The API's `open_cached` and CLI offline path precede URL-file/stdin inspection,
credential/client construction and transfer admission. They open original managed
descriptors, verify all bytes and the exact single-entry source/path/role/length/
SHA/TrustedSha256 contract under caller limits, and reuse LFS-pointer refusal.
An existing ordinary cache ref may be selected instead of a snapshot digest;
that lookup grants no online freshness, license acceptance or inference support.
No repository sidecars or model closure are inferred for this explicit object.

## Native OCI raw GGUF layers

`acquisition::oci::OciRequest` resolves one OCI/Docker schema2 image manifest
by explicit tag or raw manifest SHA256. It hashes received manifest bytes before
parsing, checks a supplied manifest digest and any Docker-Content-Digest header,
and requires matching supported response/JSON media types. A recognized raw
`application/vnd.docker.ai.gguf.v3` layer supplies its exact size and SHA256 to
the original transfer/cache executor. Unique selection refuses multiple matching
layers; an explicit layer digest selects exactly one recognized layer.

```text
zerogravity pull oci --origin https://registry.example --profile anonymous \
  --repository owner/model --tag selected --file model.gguf --dry-run \
  --cache /absolute/cache
```

The example authorizes no source. Use `--manifest-sha256 RAW_SHA256` instead of
`--tag` for a caller-pinned manifest, and `--layer-sha256 RAW_SHA256` when selecting
a particular layer. Dry-run performs admitted broker/manifest metadata requests
but no layer transfer or cache initialization. Online defaults to apply.

Profiles are explicit anonymous access, preissued bearer via `--token-env NAME`,
or `dockerhub` public anonymous pull. Docker Hub fixes the registry to
`https://registry-1.docker.io` and token broker to `https://auth.docker.io/token`
with service `registry.docker.io` and the selected repository's pull-only scope.
No server-provided realm, arbitrary login, Basic credential, refresh, SDK or
ambient account discovery runs. The shared HTTP owner retains origin-scoped
credentials, TLS verification, secret redaction, redirects and bounded requests.
The broker consumes at most64KiB and half the metadata cap; manifest metadata
uses the remaining cap. Docker Hub requires a two-call logical metadata budget.

Provider `oci` records the raw manifest digest as revision. Its repository label
hashes the full versioned request contract: canonical origin, profile, repository,
requested tag-or-digest, logical file and layer selector. Secret values and URLs
are absent from the persisted identity. The original executor verifies actual
layer bytes against the descriptor's raw digest, including cache hits, framing,
EOF, LFS-pointer refusal and existing cancellation/rollback rules. No mutable
reference is published by this frontend.

Offline requires `--offline --snapshot DIGEST_OR_EXISTING_REF`, together with
the original request flags. It opens and hashes original managed cached bytes,
checks exact one-entry source/path/Weights/trusted-digest constraints and caller
limits, and branches before credentials, broker, client, transfer admission or
signal construction. The cache does not retain original raw manifest JSON:
offline contract matching cannot independently reauthenticate descriptor
membership, tag freshness, registry signatures or current remote access.

Indexes/platform selection, compressed/container layers, arbitrary descriptor
URLs, config extraction, generic authentication challenges and private Docker
login remain unsupported. Acquiring a GGUF layer supplies no model sidecars,
model closure, license acceptance or new inference eligibility. This is a bounded
native registry-data path, not full OCI conformance or container execution.

## Native GCS versioned objects

The native JSON API resolver accepts explicit objects or bounded auxiliary
prefixes at `https://storage.googleapis.com`. `GcsObjectRequest` resolves metadata
once; `GcsObjectPin` records generation, metageneration and exact size. Public
`resolve_with_pin`/`resolve_with_pins` expose those observed pins alongside the
original plans. Ordinary `resolve` still returns original plans alone.

```text
zerogravity pull gcs object --bucket selected-bucket --object selected/model.gguf \
  --file model.gguf --role weights --generation GENERATION --sha256 RAW_SHA256 \
  --dry-run --cache /absolute/cache
zerogravity pull gcs prefix --bucket selected-bucket --prefix selected/ \
  --dry-run --cache /absolute/cache
```

These examples authorize no bucket or data. Object roles are explicit; caller
SHA256 is optional and independently authorized. Without it the final original
raw hash remains FirstSeen: GCS MD5, CRC32C and ETag are not raw SHA256 authority.
Omitting generation resolves latest metadata once to a concrete generation.
Dry-run reads only admitted bounded metadata; online defaults to apply.

Media requests derive locally from the validated bucket/name and carry both
`generation` and `ifMetagenerationMatch`. Returned mediaLink values are ignored.
Pinning metadata version also protects stored representation: nonidentity
contentEncoding refuses before transfer, including a later representation change
via the metageneration precondition. Customer-key and soft-deleted profiles
remain unsupported. The original HTTP owner refuses all media redirects so a
Location cannot discard these query preconditions. Existing custom transports
must explicitly implement that direct-response capability or fail closed.

One private typed binding in the original executor checks contradicting PRESENT
x-goog-generation/metageneration headers before initial, resumed and live-retry
bodies. JSON API does not require those response headers; absence is allowed,
with query preconditions supplying the protocol binding. Original duplicate-
header, exact length/hash/EOF, strong-validator resume and publication rules
remain independent checks. There is no additional transfer engine.

Prefixes must be nonempty and slash terminated. The resolver bounds all page
bytes, calls, tokens, file count, control metadata and aggregate selected bytes;
cycles, duplicates, unsafe paths and file/directory collisions refuse. Every
selected object yields an independent Auxiliary/FirstSeen plan. Listing and
publication are not atomic: a later failure may leave earlier verified snapshots.
No config/tokenizer/index/model closure is inferred.

Anonymous and explicitly named bearer-environment profiles reuse the shared
HTTP owner. There is no ADC, metadata-server discovery, service-account minting,
SDK, refresh or requester-pays fallback. Custom production GCS endpoints remain
unsupported; diagnostic literal-loopback fixtures are separate.

Offline object selection requires the original bucket/name/logical path/role,
exact `--generation`, `--metageneration`, `--bytes` and snapshot digest, plus the
same caller SHA if used online. It verifies original managed bytes and exact
source/selection/trust under caller limits before credentials/client/admission/
signals. Receipts print the observed pin for later reuse. Prefix offline selection
refuses; choose each explicit pinned object instead. Offline byte/source matching
attests neither present remote access nor bucket membership/freshness.

## Native versioned S3 objects

The S3 resolver uses explicit regional AWS commercial endpoints or a caller's
bare compatible endpoint, region and path/virtual-host addressing. It resolves
HEAD once to an exact non-null object version and size; selected versions then
use the original transfer/cache executor. Typed pins expose actual version and
bytes for receipts/offline reuse. Null/unversioned and special bucket/partition
profiles remain outside this slice.

```text
zerogravity pull s3 object --endpoint-profile aws --region us-east-1 \
  --addressing path --bucket selected-bucket --key selected/model.gguf \
  --file model.gguf --role weights --version-id SELECTED_VERSION \
  --sha256 RAW_SHA256 --dry-run --cache /absolute/cache
```

The example authorizes no source. `--endpoint-profile compatible` requires an
explicit `--endpoint`; AWS profile rejects overrides. Virtual-host credentials
bind to the bucket-specific origin, and dotted virtual-host buckets refuse under
this TLS profile. Online defaults to apply; dry-run reads admitted metadata only.

Anonymous and explicit SigV4 profiles share the original HTTPS owner. SigV4 takes
only named `--access-key-env`, `--secret-key-env` and optional
`--session-token-env` inputs. It signs the actual method/Host/Range/If-Range and
amz headers on each attempt with checked UTC. No SDK, ambient AWS config/default
credential chain, identity service, region discovery/redirect or refresh runs.
Credentials stay in that origin's client; they never enter source labels or plans.

The opaque `HttpTarget` preserves the exact encoded request path/query through
that same HTTP owner. Its constructor is private to validated provider code, and
custom transports must explicitly support its default-refusing capability. S3
keys retain dot segments and repeated slashes; UTF8, percent and query values are
encoded once, with sorted canonical query pairs. Redirects refuse. This avoids
ordinary URL normalization silently selecting a different key or invalidating a
signature; no second socket or executor is introduced.

The original executor requires matching x-amz-version-id and refuses delete
markers before accepting body bytes at each initial/resume/retry response.
Strong ETag, exact range/framing/length/full raw SHA/terminal EOF and cache
publication retain their existing separate checks. Caller raw SHA gives
TrustedSha256; otherwise final content remains FirstSeen. ETag, MD5, CRC and
multipart/composite checksum labels never become raw SHA256 authority.

### Explicit AWS FULL_OBJECT SHA256

Versioned AWS-standard S3 object pulls may opt into `--aws-full-object-sha256`
(library: `S3ObjectRequest::with_aws_full_sha256()`). This changes only one
selected object's pinned HEAD: the existing typed target sends
`x-amz-checksum-mode: ENABLED`, signed as a literal in SigV4 SignedHeaders
(or sent literally for anonymous public HEAD), and forbids redirects. AWS
must return the exact non-null version, length, a canonical 32-byte base64
`x-amz-checksum-sha256`, and `x-amz-checksum-type: FULL_OBJECT` together.
The latter is essential: absent/COMPOSITE multipart checksums, missing
or noncanonical base64, mismatching caller `--sha256`, and contradictory
version/length or encoding refuse before payload GET. Without this explicit
flag, ordinary S3 pulls retain FirstSeen when no independent caller digest
is given—even if HEAD happens to include a checksum header.

On success the original pinned version/bytes/source/role/expected SHA256
contract is upgraded to TrustedSha256 and passed unchanged to the **single**
existing cache transfer, raw SHA/EOF verifier, and atomic publisher. Dry-run
remains HEAD metadata only and prints the observed full-object SHA256; it
never opens a payload or creates a snapshot. Offline requires `--sha256`
equal to that printed digest in addition to the same version, bytes and
snapshot digest, and validates only the exact cached contract—no HEAD,
credentials or inferred present-day AWS availability.

This slice is restricted to standard commercial AWS S3 endpoints with an
immutable non-null version and an explicit object selection. Compatible
endpoints, prefix scans, conditional FirstSeen mode, caller-only content
mode, null versions and assumed composite hashes **cannot** enter it.
An S3 KMS-encrypted object may require additional KMS permissions for checksum
access; the existing explicit credential/profile scope is unchanged and
permission failures do not fall back to weaker trust. No SDK, default AWS
credential discovery, region redirect, model loader or GPU action is added.
References: [AWS HeadObject API](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html) and [Checking object integrity](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity.html).

`pull s3 prefix` selects Auxiliary/FirstSeen objects independently. `--listing
current` uses ListObjectsV2 continuation tokens; `latest-versions` uses paired
key/version markers and selects latest live entries. Complete bounded XML and
HEAD validation precede returned plans, with aggregate request/metadata/control/
file/byte budgets, cycle/duplicate/path/collision refusal, and no DTD/entity
expansion. A prefix is not atomic, and later failure may retain earlier published
objects. Neither listing infers model closure.

Offline object mode needs the same endpoint/region/addressing/bucket/key/path/
role contract, `--version-id`, `--bytes`, snapshot digest and any original caller
SHA. It verifies original managed bytes/source/trust before credentials, client,
admission or signals. Prefix offline refuses; choose explicit object pins. Cached
bytes do not establish present listing membership, access or remote freshness.
Broader vendor features, presigning, archive restoration, customer keys,
delegated credential acquisition and non-SHA full-object checksums remain open.

## Native Azure BlockBlob revisions

`pull azure object` selects an explicit opaque version ID or snapshot on the
native public-cloud account endpoint. HEAD supplies exact size and a transient
strong ETag; GET and every retry/resume preserve the same revision query through
the existing opaque target and HTTP owner. Responses require a complete,
identity-encoded BlockBlob; structured bodies, customer-key representations and
incomplete copies refuse. A present version echo must agree, but the official
protocol does not require that echo. Snapshots and version IDs remain separate
namespaces.

```text
zerogravity pull azure object --origin https://ACCOUNT.blob.core.windows.net \
  --container selected-container --blob selected/model.gguf \
  --file model.gguf --role weights --version-id SELECTED_VERSION \
  --sha256 RAW_SHA256 --dry-run --cache /absolute/cache
```

The example authorizes no source. Caller SHA256 gives TrustedSha256; otherwise
original final hashing remains FirstSeen. ETag, MD5 and CRC64 are separate from
raw SHA authority. Source identity binds the exact origin/container/blob,
revision kind/value, logical path/role, observed bytes and optional caller SHA;
credentials and the transient ETag do not become offline source inputs.

Anonymous, preissued bearer, account SharedKey and a bounded ad-hoc serviceSAS
are explicit mutually exclusive profiles. SharedKey signs each actual request
with checked UTC, fixed REST version2023-11-03 and independently tested
HMAC-SHA256 canonicalization. Account keys come only from a named environment
variable. SAS comes from a bounded8KiB nofollow regular file or explicit stdin;
its escaped query stays private inside the same HTTP owner and is preserved
verbatim. The supported serviceSAS profile requires HTTPS/read access, an
appropriate version/snapshot/container resource, and sv2019-12-12 through
2023-11-03. The owner appends unsigned api-version=2023-11-03 to pin REST behavior.
Stored policies, account/delegation SAS, response overrides, ambient credential
chains, minting, SDK login and refresh remain outside this profile. Pinned
requests never redirect.

`pull azure prefix` uses bounded current-version XML listing and returns
independent Auxiliary/FirstSeen plans. It checks the entire selected listing,
continuations, version/type/copy properties, paths, aggregate controls and byte
budgets before returning. It grants no atomic container snapshot or model
closure; earlier complete publications may survive later failure.

Offline object mode requires the same source contract, revision kind/value,
exact bytes (including zero), cached snapshot digest and original caller SHA if
used. It verifies the original managed bytes and provenance before any secret
file/stdin/environment read, client, admission or signal setup. Prefix offline
selection and mutable refs refuse. Cached bytes attest neither current access nor
remote freshness. Implicit latest/no-SHA current objects, other endpoints/blob types and
broader credential profiles remain open work.

## Explicit unversioned cloud content contracts

S3 and Azure object commands accept an explicit `--content-contract --bytes N
--sha256 RAW_SHA256` profile. It supports current/unversioned locators without
inventing a remote version. Exact bytes (including zero) and the caller's raw
SHA256 are mandatory. Version/snapshot selectors, prefix mode, mutable refs and
unknown-size caps cannot be combined with this profile. A missing remote version
never silently selects it.

```text
zerogravity pull azure object --origin https://ACCOUNT.blob.core.windows.net \
  --container selected-container --blob selected/model.gguf \
  --file model.gguf --role weights --content-contract \
  --bytes EXACT_BYTES --sha256 RAW_SHA256 --dry-run
```

The example authorizes no source. S3 uses the same flag set with its existing
endpoint/region/addressing/bucket/key selectors. Preparation and dry-run are local
contract validation under bounded control admission: no HEAD, metadata, secret
input, client or cache publication. Apply uses the original exact transfer,
strong-validator resume, full raw SHA/EOF and sole publisher. A current locator
may change, but only bytes matching the complete authorized content contract can
install. The source providers s3-content and azure-content bind normalized source
selectors, logical path/role, exact size and SHA; they attest no remote version,
present membership or freshness.

Typed targets preserve S3 raw paths and cloud credentials through the same HTTP
owner; redirects refuse. S3 delete markers refuse. Azure retains identity-encoded
BlockBlob/copy/key/structured-body checks. Azure content SAS accepts read b or c
resources; version/snapshot requests retain their distinct bv/bs/c rules. Returned
version headers do not replace the caller's SHA. Offline requires the original
snapshot digest and identical content contract before credentials/client/signals
or transfer admission, using the shared managed-file verifier. HTTPS retains its
existing ref selection. Strong-conditional observations without caller SHA and
per-object digest-map prefixes remain separate work.

## Strong conditional cloud observations

The explicit object-only `--conditional-read` mode supports S3 and Azure current
objects when the caller supplies no raw SHA. The existing arbiter admits metadata
work before a body-free HEAD to the exact native target. An exact200 response must
provide complete identity-representation length and one strong ETag within the
original limits and provider representation checks. Missing/weak validators,
redirects, unsupported status/framing and exhausted bounds refuse.

A separate `s3-conditional` or `azure-conditional` source contract binds the full
normalized locator/profile, logical path/role, exact bytes and ETag. This is a
conditional observation. Actual raw SHA remains FirstSeen; neither ETag nor the
current locator supplies a trusted raw digest or immutable remote version.
Changing the declared observation changes its contract and retained-prefix key.

The original opaque HTTP target carries a private typed If-Match condition. Every
initial GET, range and retry sends it, with exact response ETag validation before
body reads. A differing resume validator refuses before wire/signing. SigV4 signs
the same If-Match bytes; Azure SharedKey uses the standard If-Match slot. Existing
anonymous, bearer and serviceSAS profiles retain their original ownership. HTTP412
refuses without refreshing metadata, downgrading the condition or choosing another
profile. The original exact length/hash/EOF/LFS/cancellation and sole cache
publisher remain authoritative.

Online dry-run resolves HEAD metadata only. Offline takes the same explicit
conditional request plus exact bytes, quoted ETag and snapshot digest; it checks
source/path/role/size/FirstSeen provenance, original full managed bytes and current
limits before credentials, file/FIFO/stdin, client, admission or signals. The pure
library pin constructor represents a caller-declared observation; offline access
asserts no present remote freshness or authorization. Receipts escape the ETag as
a JSON string. There is no conditional prefix/digest-map or implicit fallback;
existing remote-version and caller-SHA contracts keep their original semantics.

## Native ModelScope explicit commits

`pull modelscope OWNER/REPO --commit FULL40HEX --file PATH` uses the native
modelscope.cn API directly, without a Python SDK or persisted login. Repeat
`--file` for bounded selectors and use `--subfolder` for a repository directory.
Only a canonical lowercase40hex commit is accepted. Alias resolution, default
revisions and mirror endpoints remain separate work. Missing selected sizes use the bounded
HEAD profile below. The pinned
official client passes an explicit revision through its listing and raw-file
requests. This profile trusts that endpoint contract; it does not independently
verify a Git/Merkle tree or prove live server behavior from synthetic tests.

Resolution walks bounded shallow directories with Revision, Recursive=False and
relative Root. The official legacy API silently caps a directory at3000 rows
without pagination or completeness authority. A directory reaching that boundary
refuses, even when the requested file is already visible. One cumulative budget
bounds logical metadata calls and returned bytes across the inventory, HEAD probes
and required indexes. Original HTTP retry limits and per-request deadlines apply
independently; caller file/control limits also bound the inventory. No partial
walk becomes model closure.

A missing Size on a selected file or required index uses one body-free exact200
HEAD at the same full commit and opaque Direct target. The original admitted
metadata owner charges its request and returned header controls to the same
remaining budgets. Exact identity Content-Length is required; redirects, ranges,
transfer encodings, unsupported status and absent/invalid lengths refuse. Index
lengths are cached once before the index GET and reused by final planning.
Unselected absent sizes cause no HEAD. Explicit null or malformed present blob
sizes still refuse; the existing directory-null representation remains valid.
A strong HEAD ETag is checked before index/payload bytes; weak or absent validators
grant no resume authority. Listed raw SHA and FirstSeen distinctions remain intact.
Aliases still need authoritative alias-to-commit mapping; caller revision echoes,
file timestamps and last-modified commits are insufficient. CDN/mirror credentials
and unknown-size caps remain separate unqualified profiles.

A future file-only CDN profile must require exact bytes and an authorized raw
SHA before delegation. Its finite download-origin policy needs explicit authority;
one redirect cannot authorize a wildcard cloud suffix. The original HTTP owner
must strip Bearer and session cookies after every delegated hop, including a
bounce back, and keep signed locators ephemeral and redacted. Refresh must start
from the same native full-commit URL with unchanged hash, length and range rules.
Metadata Direct targets remain no-follow. Mirrors, ambient region discovery and
aliases do not inherit this potential profile; implementation and actual two-origin
credential/refresh/rollback tests remain open.

The original HF initial selector and extracted private GitHub closure logic
supply required config/tokenizer, indexed shards and split-GGUF checks, including
ambiguous-model refusal. GitHub retains its original metadata/size/raw-URL/LFS
owner around that pure callback. A canonical source-provided raw SHA gives
TrustedSha256; absent SHA stays FirstSeen. Git object IDs are not raw hashes.

Anonymous and explicit `--profile official-token --token-env NAME` are the only
profiles. The official-token client sends Bearer and m_session_id together within
one exact origin, with cookie-safe token validation and no jar/login/refresh.
Private Direct targets preserve the selected commit and refuse all redirects
for metadata and payloads; redirecting/CDN profiles remain unqualified. S3/Azure
credentials cannot authorize Direct, and official-token credentials cannot
authorize their targets. The sole HTTP/executor/cache/publication owner remains.

Offline requires the same repository/commit/selectors and a snapshot digest. It
verifies all original managed bytes, source, role/provenance, exact selection,
current bounds and pointer policy before credentials/client/admission/signals.
The source label includes normalized origin identity so diagnostic fixtures cannot
alias native data. Offline data does not prove present access, remote membership,
license or inference support. No mutable ref or capped required-file mode is
introduced.

## Xet hash and chunk primitives

The native reader shares exact hash/codec sources with the standalone primitive
test wrapper. The implementation follows protocol1.1.0 and its pinned xet-core
reference, with separate raw32-byte and limb-reversed textual
identifiers, keyed BLAKE3 chunks, in-place ordered Merkle aggregation, zero-key
file finalization and the special empty root. Raw SHA256 remains a separate
content/provenance authority. Download metadata supplies no expected per-term
verification hash; computing a shard/upload verification primitive does not
authenticate an absent field or a partial xorb root.

The safe decoder borrows caller buffers and allocates nothing. It bounds encoded
chunks to256KiB and decoded chunks to128KiB, validates the8-byte header and exact
consumption, and supports NONE, framed LZ4 and BG4/LZ4 including every tail
remainder. Standard frame/block checksums, linked versus independent history,
end markers and terminal EOF are checked. Unsupported dictionary, legacy,
skippable and concatenated frames refuse. Bounded cancellation checks cover
copying, checksumming, decoding and hashing. Literal upstream Merkle vectors and
independent hash/frame goldens exercise exact namespaces and malformed inputs.

Only two direct acquisition dependency edges, cached BLAKE3(std/pure) and
xxhash32, are added; CPU/cache-only transport isolation remains. The explicit
native route below uses these same primitives. Primitive tests alone grant no
native-transfer throughput or model capability.

## Native HF Xet

`pull xet OWNER/REPO --revision REV --file PATH` explicitly selects native
reconstruction through the existing optional acquisition owner. Original HF
pinning, complete bounded inventory, selectors, model/shard/sidecar closure and
source identity stay authoritative. At least one selected LFS object must announce
an Xet file identity. Discovery observes headers without following its redirect or
reading a compatibility payload. Declared ordinary Git sidecars retain the original
HTTP route, redirect credential stripping and strong-validator suffix resume.
Announced native files have no silent compatibility fallback on protocol failure.

A repository/commit-scoped read grant is obtained from a locally constructed Hub
route. The grant is ephemeral and restricted to the fixed CAS origin; arbitrary
refresh links do not supply authority. CAS uses only that grant, and signed range
requests suppress all Hub/cloud credentials even at the same origin. The original
HTTP owner preserves TLS, raw signed URI bytes, exact complete signed interval
sets, no-follow policy for native metadata/payloads, redaction and request deadlines.
Custom transports must explicitly implement the opaque target/accounting contract
or refuse; no secret getter or arbitrary-header interface is added.

Native reconstruction tries v2, with v1 only after404/501. Scoped401 and signed403
can refresh within one finite combined reissue budget. The file identity and
ordered semantic term contract must stay fixed. Parsing validates full-file offset,
exact aggregate lengths, unique maps, bounded terms/groups/ranges/chunks and full
coverage. Terms may repeat or overlap; only their selected logical chunks enter
the final file root, in file order. Fetched edge chunks are decoded and charged,
but do not enter that root or logical progress. Missing sibling data cannot prove
a full/partial-xorb root, and the protocol supplies no per-term verification hash.

The reader holds one decoded chunk and admitted codec/hash scratch. It requests
the complete signed range set, validates every multipart boundary, interval,
length, representation and terminal EOF, and handles parts in arbitrary order.
It can reissue a full set for later selected spans; fetched/decoded repeats remain
charged. Interrupted refetch compares already accepted leaves without duplicating
logical progress. This bounded reader has no throughput or fetch-coalescing claim;
measured coalescing remains future work.

One invocation bounds actual HTTP attempts, observed body/control bytes, decoded
chunks, logical leaves and refreshes. Accepted term/group/range counts accumulate
across files and metadata reissues, while each document is also bounded. At most
one observed overflow sentinel is charged; after overflow, the invocation refuses
further reads or attempts. Exact-budget zero-byte EOF probes remain possible.
Per-request deadlines plus finite work do not create a whole-operation elapsed
bound or kernel/TLS traffic accounting.

The original arbiter admits scratch before allocation and retains it alongside
the original cache transaction. Native CLI defaults to256MiB to cover both full
admission envelopes; ordinary pulls remain128MiB and explicit limits are unchanged.
The same cache locks, private staging, raw SHA/length/LFS/EOF checks and sole
publisher apply. Native objects use an exact ephemeral stream: HTTP prefixes are
neither recovered nor checkpointed, and process exit requires a new full native
transfer. The final Xet file root precedes successful source EOF and publication.
Cache hits remain verified cached data, with no assertion of native execution.

Dry-run performs Hub metadata/discovery only, with no CAS grant or signed body.
Offline opens an existing cache, checks original source/selection/current bounds
and pointer policy before credentials, client, admission or signals. It proves
cached data, not native protocol execution, freshness or present access. Measured
fetch coalescing, partial native prefixes and persistence remain open;
acquired bytes grant no new loader, inference, license or model qualification.

### Required native LFS/Xet safetensors index reconstruction

A required selected or adjacent `.safetensors.index.json` announced as native
Xet is now supported only through an explicit `XetOptions::with_index_reconstruction()`
builder. The default library resolution and metadata-only CLI dry-run preserve
their original **no CAS grant/signed body** contract and refuse the required
index before obtaining such a grant. Native CLI **apply** opts a cloned options
value into reconstruction; no new flag, second resolver, transport, source
issuer or cache publisher is introduced. Ordinary non-LFS source indexes retain
their original bounded GET route.

The original pinned HF tree, exact length, source index callback and model
closure supply the only source authority. A trusted LFS raw SHA256 is required
independently of the full keyed Xet file root. The existing original Access,
reader, scratch, signed-range and HTTP owners verify exact logical bytes,
complete root, terminal EOF sentinel, raw SHA and pointer refusal **before**
parsing index JSON or selecting any required shard. Index JSON remains bounded
by the original cumulative HF metadata budget and 8MiB parser ceiling.
Pre-plan reconstruction is ephemeral; publication reconstructs the index
again under the sole existing cache transaction, hashes/verifies it again and
reparses the staged index before a durable snapshot can replace a reference.

One original TransferAdmission keeps simultaneous metadata control and scratch
reservations, normally 80MiB + 90.5625MiB. The native CLI default256MiB
admits the real index path; a caller-provided128MiB bound refuses before
signed index payload. Every invocation's attempts, wire, metadata, decoded
chunks, leaves, refreshes, accepted terms/groups/ranges survives the
resolution-to-execution seed; poison and tightened limits refuse instead of
silently resetting counts. Native index planning/publication and later shard
work are cumulatively charged, including retransmitted/edge work. This is
not a model inference, throughput, remote freshness or whole-memory guarantee.

## Provider and compatibility frontier

The retired research captures once counted57 rows; that historical count is not
an extant exhaustive ledger. The current finite frontier below reflects inspected
native entrypoints and the pinned reference classes. A disposition is not a
passing provider gate. Open implementation and acceptance work stays in TODO.md.

| Surface | Current qualified scope and remaining work |
| --- | --- |
| HF repository/files/revisions | Native exact commit pinning, subfolder, bounded `*`/`?`, required index/shard and config/tokenizer closure. Metadata-only candidates share the same pinned tree. Explicit filename quant and full-path fnmatchcase include/exclude selection reuse original closure; broader required sidecar policy and endpoint/env compatibility remain open. |
| GitHub files/releases/components | Native commit/tree and release/asset pins, exact sizing, independent pinned components and alias pin reuse. Qualified LFS-pointer resolution is separate; pointers refuse. |
| Direct HTTPS objects | Explicit origin/role/path/exact size/caller SHA; original transfer, offline verification and cache publisher. Broader missing-size/digest or repository discovery is outside this slice. |
| Local HF caches | Explicit pinned selections and independent verified copies, original same-mount policy plus qualified Btrfs/tmpfs and distinct tmpfs pairs. Renamed/unnamespaced roots and other filesystems remain open. |
| XDG/offline/removal | Native XDG/explicit root, zero-network offline, inspect/list/verify and protected exact-snapshot removal. Compatibility aliases, legacy migration, automatic eviction/general orphan cleanup and unmanaged readers remain separate open work. |
| Auth/endpoints | Origin-scoped explicit bearer environment inputs, safe redirects and TLS. Mirror/profile policy, broader compatibility env aliases and provider-specific credentials remain open; offline bytes do not attest current remote access. |
| OCI/Docker | Native tag-or-manifest pin/raw-GGUF layer, public/preissued bearer and fixed Docker Hub anonymous broker. Index/compressed/container/signature/generic private-login profiles remain open; no model closure follows. |
| GCS | Native JSON object generation/metageneration media pins, auxiliary prefix/pageToken listing, anonymous/preissued bearer and explicit offline object pins. ADC/minting/requester-pays/custom endpoints and atomic/closure semantics remain outside this profile. |
| S3/compatible endpoints | Native non-null versions, regional/path/virtual-host GET/HEAD/ranges, bounded V2/latest-version prefixes and explicit SigV4. Current/unversioned caller SHA/size contracts and strong conditional FirstSeen observations remain separate. Explicit AWS-standard versioned object `--aws-full-object-sha256` qualifies typed HEAD FULL_OBJECT SHA-256; no COMPOSITE/compatible trust. Broader partitions/vendor/credential profiles and other checksum algorithms remain open; presigned HTTPS alone grants no native S3 parity. |
| Azure Blob | Native explicit BlockBlob versions/snapshots, bounded current-version auxiliary listing, pinned ranges and anonymous/bearer/SharedKey/ad-hoc serviceSAS. Explicit current/unversioned objects additionally support caller SHA/size contracts; Strong conditional FirstSeen observations are explicit; latest convenience, other endpoints/types/SAS and managed/default credentials remain open. |
| ModelScope | Native explicit40hex commit, bounded complete shallow walk, exact files/model closure, anonymous/official-token and digest-only offline. Selected missing sizes use exact pinned HEAD; aliases and redirect/CDN/mirror profiles remain open; no SDK/login runtime. |
| HF Xet | Explicit bounded native discovery, scoped CAS, signed ranges/multipart, ordered NONE/LZ4/BG4 reconstruction and original ephemeral importer/CLI. Explicit APPLY reconstructs required trusted-LFS indexed safetensors JSON under the same full-root/raw-SHA/EOF/cumulative-work owner; default/dry-run still refuse before CAS. Coalescing/persistent native prefixes/performance remain open; cached bytes grant no native execution proof. |
| Auxiliary/preset convenience | Explicit auxiliary/component bytes are supported data. Specialized mmproj/draft/MTP/DFlash/Eagle3/DSpark discovery and bounded presets remain open convenience work; a preset cannot authorize new sources or limits. |
| Transport/streaming | Exact strong-validator resume, bounded retry/deadline/cancel/progress and explicit capped Auxiliary streams are qualified. Required unknown lengths, capped resume, independent full HTTP trailer validation and broader concurrency/performance remain open. |
| Loader/runtime delegates | Safetensors/PT/serialized tensors, RunAI/ModelExpress internals, IPC/distributed/lazy loading, quant reconstruction and speculative/multimodal execution belong to the execution/streaming plans. Acquiring opaque bytes grants none of those capabilities. |

The pinned reference loader registry has separate execution responsibilities:

| Reference loader surface | Responsibility beyond native byte acquisition |
| --- | --- |
| auto / hf / safetensors / pt / mistral | Format choice, deserialization and family validation; cached PT/pickle is never executed by pull. |
| npcache | Derived cache representation and loading. |
| fastsafetensors / instanttensor | Loading, prefetch, placement and measured performance. |
| sharded_state / runai_streamer_sharded | Rank/shard selection, collectives and distributed loading. |
| runai_streamer | Tensor streaming, device buffers, broadcast and optional SDK/cache integration; its concrete cloud byte protocols remain acquisition work. |
| tensorizer | Serialization, encryption and per-tensor verification over acquired local/HTTPS/S3 bytes. |
| ipc_cache | Process/device ownership and CUDA IPC. |
| dummy | Synthetic parameter construction. |
| modelexpress | Inventory the selected package revision's finite source/auth contracts before native equivalence; its external delegation is not an open-ended native guarantee. |
| custom plugins / remote code | Add concrete missing protocols through the existing owner; arbitrary foreign loader execution is outside finite transport parity. |

## Explicit read-only HF cache import

The independent Linux `cache` feature exposes `cache::import_hf` and an explicit
local CLI. No HTTP, TLS, provider client, credential lookup or online signal
handler is involved:

```text
zerogravity cache import-hf \
  --source /absolute/hub/models--owner--repo --repository owner/repo \
  --commit 0123456789abcdef0123456789abcdef01234567 \
  --file weights:model.gguf --file config:config.json \
  --file tokenizer:tokenizer.json --dry-run --cache /absolute/destination
```

Use `--apply` for copying and optional `--sha256 PATH=DIGEST` for independently
authorized raw checksums. Each path and role is explicit; this is a data import,
not a model-closure resolver. The example names illustrate layout and select no
approved source. Ordinary run/inspect and cache misses do not import anything.
No environment-based cache discovery, branch/ref lookup or remote code runs.

Import requires a two-component repository name and a matching
`models--owner--repo` source basename, a40-digit lowercase commit and an existing
absolute destination directory with0700 permissions. An empty destination can be
initialized after validation; missing directories are never created. Renamed or
unnamespaced repository caches remain outside this slice. Source/destination
may use the original same-mount path or one narrowly qualified cross-filesystem
pair: Btrfs with tmpfs in either direction, or two distinct tmpfs filesystems.
Cross-filesystem roots must differ in both Linux mount ID and device ID, and
retained-FD fstatfs must report that permitted type pair. Each traversed node and
selected source descriptor must remain on its own root device and mount.
Across different mounts, same-device bind aliases, Btrfs-to-Btrfs pairs,
overlay/FUSE and other unqualified pairs refuse. Missing statx mount identity
also refuses. This allowlist is a bounded qualification policy, not a claim
that different device IDs alone
prove independent storage. The original same-mount behavior is unchanged.

Read-only source capabilities accept regular snapshot files or canonical HF leaf
links into flat40/64-digit lowercase blob names in the same repository root.
Intermediate/chained/absolute/escaping links, special selected files and incomplete
selected paths refuse. Positioned reads retain the original opened files rather
than reopening paths. Metadata checks bind device/inode, length, mtime and ctime
before/after copying and each read; a separate nonempty EOF sentinel proves the
exact bound. Git LFS pointer bodies refuse during streaming.

Preparation admits a bounded control envelope in the caller's existing physical
ResourceArbiter domain before opening source metadata. The plan retains that
lease and its source descriptors. Destination checking walks bounded source and
target directory metadata without reading refs contents, cached trees/*.json or
unselected file bodies. No symlink is followed during that topology walk.
Bidirectional ancestry, mount and inode checks reject overlapping trees and hardlink aliases, including
unselected source inodes and the destination root itself. This explicit
selected-repository safety inspection is not a cache-discovery scan. A successful dry-run is not a reservation or permission
for a later changed tree; apply repeats the checks.

Apply initializes the exact preflighted destination directory capability through
the original CacheStore initializer, then uses its existing full-peak admission,
private staging, hash/EOF validation, sole installer and snapshot publisher.
No source inode is locked, chmodded, linked, renamed or deleted by import. Source
bytes are copied into owned cache storage. A known-digest destination hit still
hashes and verifies the retained external source before commit; receipt
copied_bytes counts only bytes consumed by the importer, so such a hit reports
zero copied bytes. Normal copies reuse the original importer's SHA256 result.
No mutable reference is published, and typed DurabilityUnknown remains distinct
from a successful durable snapshot.

Manifests persist provider `local-hf-cache`, the caller-declared repository/commit
and actual exact lengths/hashes. They cannot satisfy online-HF cached-selection
checks as authenticated remote metadata. Content is FirstSeen unless an explicit
caller checksum verifies; cache filenames and commit-directory names supply no
raw checksum or membership authority. The existing run-cache bridge can consume
explicit resulting snapshots through its unchanged loader eligibility checks.

These are operation-local checks, not an atomic snapshot or a freeze against
arbitrary same-UID in-place writes, renames or mount changes. Blocked filesystem
operations do not gain real-time cancellation. The original root initializer
retains its existing lock-deadline behavior: cancellation after preflight may
leave initialized destination metadata before transaction admission refuses;
initialization metadata is not rolled back. Reading can update source atime;
tests preserve source names/content/modes/mtime. Ownership and mutation checks do
not establish global RAM/UMA accounting or filesystem free-space reservation.
A tmpfs destination remains volatile: the original publisher's successful fsync
acknowledgments do not promise survival across unmount, reboot or power loss.
No filesystem is mounted or privileges changed by import or by the selected
existing-filesystem fixtures. Real hidden bind/subvolume/overlay alias fixtures
remain separately unqualified; those ambiguous cross-filesystem pairs refuse.
Acceptance and remaining filesystem/layout work are tracked in TODO.md.

## Model pull v1 acceptance and execution cutover

The bounded original-source **data-acquisition path** is the prerequisite for
CPU-first model execution, not a model runtime. The optional acquisition CLI
covers HF/GitHub pinned files/index/quant/filter/native Xet; native ModelScope
explicit full-commit metadata; direct HTTPS, OCI, GCS, versioned S3/Azure and
explicit multi-source components. Each provider has separately qualified,
**narrow** source/version/auth/byte semantics. Unknown/gated/incompatible
profiles refuse rather than silently substituting an origin, checksum, family
or model loader. The unresolved provider-parity backlog remains in TODO.md but
is not a prerequisite to beginning source binding in the model-execution plan.

### Large HF safetensors examples (source discovery/metadata only)

The target source repositories include
[Qwen3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) and
[Gemma4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it).
Examples below **do not download anything by themselves** and are not license
or account-access authorization. A caller selects the exact intended
repository/revision, must hold any necessary access rights, and separately
authorizes APPLY and storage. Do not infer a learned model/format entitlement
from a repository name.

```sh
# The executable must be built with the optional acquisition feature.
# The exact safetensors index is an explicit source member, never a guessed
# selection among arbitrary weight families. 128GiB is a cooperative cache
# reservation policy, not proof of free space or a recommended RAM size.
zerogravity pull hf Qwen/Qwen3.8-27B \
  --revision SELECTED_REVISION --file model.safetensors.index.json \
  --disk-limit-bytes 137438953472 --dry-run

zerogravity pull hf google/gemma-4-26B-A4B-it \
  --revision SELECTED_REVISION --file model.safetensors.index.json \
  --disk-limit-bytes 137438953472 --dry-run

# Inspect the output pinned commit and exact selected bytes FIRST.
# Only then explicitly omit --dry-run to authorize a real transfer.
# Native HF Xet APPLY can reconstruct an Xet-backed index under the
# same admitted owner; native dry-run refuses that case without CAS grants.
# For GGUF, inspect candidates and explicitly choose one exact --quant tag
# or an exact --file (no implicit default, ranking, or fallback).
```

The same original pinned HF/GH model closure selects every shard referenced
by the validated safetensors index and the source's same-directory config,
tokenizer and known available companions. Besides the older vocab, generation,
license and documentation members, the finite optional set now retains
`chat_template.jinja`, `processor_config.json`,
`preprocessor_config.json`, `image_processor_config.json`,
`video_processor_config.json`, `video_preprocessor_config.json`,
`feature_extractor_config.json`, `audio_processor_config.json`,
`crc32.txt` (opaque; never checksum authority), `tokenizer.model`, `spiece.model`,
`sentencepiece.bpe.model`, `vocab.txt` and `tokenizer.tiktoken`.
They are original opaque bytes: neither template evaluation, processor syntax,
tokenizer parity, arbitrary nested-directory copying nor complete multimodal
configuration is implied. C3 excludes can omit optional companions; required
shard/index/config/tokenizer selection still fails closed.

The original 16GiB CLI disk default is deliberately too small for the cited
multi-tens-of-GB models; explicit `--disk-limit-bytes` controls only the
cooperating cache reservation. Full provider plan sizing precedes any weight
body, and disk/full-transfer peak admission refuses insufficient quotas.
Verify actual available storage and permitted source use independently.
Offline source verification must bind to the **pinned commit** shown by
resolution, not a floating `main` alias; original cache hashes, source
identity, roles and bounds are rechecked. Receipt and cache eligibility
prove retained bytes, **not** the ability to load them.

### October 18, 2026 — verified public metadata-only source plans

These observations used the actual optional acquisition CLI with **no HF_TOKEN**
or other account credentials and `--dry-run`. They performed pinned Hub
tree/index metadata resolution; no learned model shard GET, cache publication,
SDK/model download, license acceptance, or inference occurred:

| Original public HF source | Resolved immutable commit | Pinned files | Selected exact bytes |
| --- | --- | --- | ---: |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 18 safetensors shards + index + same-folder companions, including `video_preprocessor_config.json` and auxiliary `crc32.txt` | 55,586,113,293 |
| `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | 2 safetensors shards + index + config/tokenizer/chat/processor companions | 51,644,338,194 |
| `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | 1 unsharded `model.safetensors` + config/tokenizer/chat/processor companions | 10,278,848,001 |

The command line is the ordinary `pull hf REPO --revision main --file
model.safetensors.index.json --disk-limit-bytes 137438953472 --dry-run`;
the resolver prints the pinned commit. For the unsharded Gemma4 E2B target
use `--file model.safetensors` instead, with an explicit suitable disk policy
(e.g. `--disk-limit-bytes 34359738368`) and `--dry-run`.
Pinned index/sidecars not independently digested by the provider are explicitly
FirstSeen; weight and tokenizer LFS raw hashes are marked TrustedSha256. The trusted/first-seen distinction is not
license, execute-on-hardware or tensor-header qualification. Use the immutable
commits above only if the intended authorization and version match; never
assume that later `main` remains the same model. GitHub, ModelScope, native
Xet and cloud extended profiles retain their separately qualified boundaries.

**Execution handoff:** the source executor still has a 4GiB in-memory
executable-source cap, and large original BF16 checkpoints are not executable
merely because the acquisition plan supports their lengths. Next actions belong
to [model-execution](../model-execution/TODO.md): choose an authorized actual
checkpoint, admitted hardware and shard identity; implement large-source owned
shard reading, header/tensor/reference binding, qualified CPU outputs and then
GPU. No model weights, SDK, cloud object, live cache or held device is touched
by these source-only tests and examples.

## Cache-to-model preparation

The independent `cache` feature enables Linux cache loading without enabling
HTTP, TLS or signal handlers. With CPU/Qwen/text also selected, use:

```text
zerogravity run-cache MODEL_SNAPSHOT --model-file model.gguf \
  --config-snapshot CONFIG_SNAPSHOT --config-file config.json \
  --tokenizer-snapshot TOKENIZER_SNAPSHOT --tokenizer-file tokenizer.json \
  --prompt "Hello" --max-tokens 2
```

Selectors are explicit snapshot digests or existing refs. Missing sources refuse;
run-cache never acquires. Normal paths and cache preparation share `PreparedQwen`,
the family's geometry/tensor/tokenizer validation and the same CPU inference loop.
The family constructor accepts bounded config bytes and reuses the existing
parser; publicly mutable ModelGeometry is never a trusted construction argument.
A cheap family eligibility check rejects unsupported geometry before copying
weights, and the constructor still parses those verified config bytes itself.
The supported executable sources remain original approved Qwen3.5 .8B/2B GGUF;
new families, shards, formats or GPU eligibility do not follow from this bridge.

`SnapshotLease` supplies the original opened descriptors and original logical
format names. Positioned reads avoid OS cursor sharing across cloned descriptors.
The source bridge copies within the existing 4 GiB cap, verifies the SHA256 of that
actual owned allocation, reparses those same bytes and invokes the sole existing
SourceIdentity issuer. Config/tokenizer copies are likewise bounded and hashed
before parsing (1 MiB/64 MiB respectively). Path/ref replacement cannot retarget
these descriptors. Arbitrary same-UID inode mutation can still occur, so the
owned-copy digest is mandatory; descriptors alone are not immutable storage.

The caller's existing ResourceArbiter domain admits raw source bytes before
allocation. Its ImmutableWeights lease stays inside SourceWeights through all
shared model owners and releases after the actual raw allocation is dropped.
Sidecar staging admission lasts through copying/hashing/parsing; parsed metadata,
tokenizer structures, session memory and global host/UMA capacity are outside
this scoped accounting. The CLI registers one process-shared source domain with
a 5 GiB default cap, adjustable via `--source-memory-limit-bytes`. Legacy path-open
accounting keeps its existing boundary. Cache preparation grants a fresh source
allocation identity, while sessions on one loaded Arc share that immutable source
and retain independent StateRoots. The original-checkpoint gate also removes private cached snapshot/weight names
after source copying, then verifies both original models through the same exact
logit/state/session checks while their raw source reservation remains owned.
Numerical acceptance is tracked in TODO.md.

## References

- [Pinned llama.cpp arguments](https://github.com/ggml-org/llama.cpp/blob/2851b3bcbf8d993a2243ae751c0a7dfa8e944a40/common/arg.cpp) and
  [download selection](https://github.com/ggml-org/llama.cpp/blob/2851b3bcbf8d993a2243ae751c0a7dfa8e944a40/common/download.cpp),
  used to inventory filename quant, companions and presets. Their default/first-file
  fallback is not this project's explicit-source selection policy.
- [Pinned vLLM weight helpers](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/model_executor/model_loader/weight_utils.py),
  [registered loaders](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/model_executor/model_loader/__init__.py) and
  [Tensorizer wrapper](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/model_executor/model_loader/tensorizer.py),
  used for the finite acquisition versus runtime dispositions above.
- [Hub filtering](https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download) and
  [environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables),
  used to identify compatibility work, not authority to consult ambient external caches or credentials.

- [Xet protocol1.1.0](https://github.com/huggingface/hub-docs/blob/39aa2452668744552cc6690c75cd239771148e58/docs/xet/index.md),
  [pinned hash/codec reference](https://github.com/huggingface/xet-core/tree/c3c726bed5cf54ded92a63fab892cfb7857c751a/xet_core_structures/src), and
  [LZ4 v1.10.0 format](https://github.com/lz4/lz4/blob/ebb370ca83af193212df4dcbadcc5d87bc0de2f0/doc/lz4_Frame_format.md),
  used for exact hash namespaces, aggregation, framing and BG4 behavior.

- [Official ModelScope client0.4.6 source](https://github.com/modelscope/modelscope_hub/blob/65255a78b451c3785f2dfbfaff97172552fea3af/src/modelscope_hub/_legacy_api.py),
  [explicit revision flow](https://github.com/modelscope/modelscope_hub/blob/65255a78b451c3785f2dfbfaff97172552fea3af/src/modelscope_hub/_download.py), and
  [silent directory truncation tests](https://github.com/modelscope/modelscope_hub/blob/65255a78b451c3785f2dfbfaff97172552fea3af/tests/test_repo_files_truncation.py),
  pinned to65255a78b451c3785f2dfbfaff97172552fea3af for the native first profile.

- [Azure GET Blob](https://learn.microsoft.com/en-us/rest/api/storageservices/get-blob),
  [properties](https://learn.microsoft.com/en-us/rest/api/storageservices/get-blob-properties),
  [listing](https://learn.microsoft.com/en-us/rest/api/storageservices/list-blobs),
  [SharedKey](https://learn.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key),
  [serviceSAS](https://learn.microsoft.com/en-us/rest/api/storageservices/create-service-sas) and
  [REST versioning](https://learn.microsoft.com/en-us/rest/api/storageservices/versioning-for-the-azure-storage-services),
  used to bound revision echoes, canonical authentication and SAS protocol version.

- [S3 SigV4 canonicalization and official signature vectors](https://docs.aws.amazon.com/AmazonS3/latest/developerguide/sig-v4-header-based-auth.html),
  [HeadObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html) and
  [version listing](https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectVersions.html),
  used to bound the explicit native regional/versioned profile.

- [GCS JSON objects.get](https://docs.cloud.google.com/storage/docs/json_api/v1/objects/get),
  [objects.list](https://docs.cloud.google.com/storage/docs/json_api/v1/objects/list) and
  [decompressive transcoding](https://docs.cloud.google.com/storage/docs/transcoding),
  used for native generation/metageneration and representation preconditions.

- [OCI Distribution v1.1.1 manifest/blob protocol](https://github.com/opencontainers/distribution-spec/blob/v1.1.1/spec.md) and
  [Docker registry authentication](https://docs.docker.com/reference/api/registry/auth/),
  used to bound the explicit registry and fixed anonymous-broker profiles.

- [Hugging Face Hub API](https://huggingface.co/docs/hub/api),
  [official OpenAPI](https://huggingface.co/.well-known/openapi.md) and
  [official file-cache layout](https://huggingface.co/docs/huggingface_hub/guides/manage-cache),
  used for the explicit local snapshot/blob import contract.
- [GitHub Git trees](https://docs.github.com/en/rest/git/trees),
  [release tag lookup](https://docs.github.com/en/rest/releases/releases#get-a-release-by-tag-name)
  and [release-ID asset listing/download](https://docs.github.com/en/rest/releases/assets),
  checked against the official endpoint contracts for this component extension.
- [signal-hook flags](https://docs.rs/signal-hook/0.3.18/signal_hook/flag/) and
  [registry cleanup](https://docs.rs/signal-hook-registry/1.4.8/signal_hook_registry/fn.unregister.html),
  checked against the exact already-cached dependency sources.
- [ureq 3.4.2](https://docs.rs/ureq/3.4.2/ureq/), with its exact cached source
  used to check timeout, TLS and redirect behavior.
- [HTTP/1.1 framing and chunked completion, RFC9112](https://www.rfc-editor.org/rfc/rfc9112.html#section-6.3),
  with the pinned ureq/ureq-proto decoder sources checked locally.
- [Linux tmpfs](https://docs.kernel.org/filesystems/tmpfs.html),
  [Btrfs subvolumes](https://btrfs.readthedocs.io/en/latest/Subvolumes.html),
  [overlay identities](https://docs.kernel.org/filesystems/overlayfs.html) and
  [FUSE passthrough](https://docs.kernel.org/filesystems/fuse/fuse-passthrough.html),
  used to bound the qualified filesystem-pair policy. The pair policy follows
  from these filesystem semantics plus actual local fixtures; it is not a general
  cross-filesystem identity guarantee stated by those documents.
  Retained-identity review also checked Linux v6.17
  [anonymous superblocks](https://raw.githubusercontent.com/torvalds/linux/v6.17/fs/super.c),
  [tmpfs implementation](https://raw.githubusercontent.com/torvalds/linux/v6.17/mm/shmem.c)
  and [Btrfs inode identities](https://raw.githubusercontent.com/torvalds/linux/v6.17/fs/btrfs/inode.c).
- [Linux flock lifetime and advisory semantics](https://man7.org/linux/man-pages/man2/flock.2.html).
- [A1 local cache contract](../../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md),
  [current workflow](../../../AGENTS.md) and [TODO rules](../TODO-SPEC.md).

````

## Source: .agents/plans/model-acquisition/TODO.md

Original UTF-8 bytes: 44908; SHA256 `3da582c91a039efe72223ce09bcdacb1844de60ec14ed8a3a19d23d51d494b1f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — model acquisition

Status: active
Provenance: 9239719
Scope: P03; prerequisite to new acquisition/loader claims.
Design: [README.md](README.md).
Contracts: [A2/A3 specification](../MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md),
[pull/provider parity](../../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md).

## Committed

Completed phases through signed `fce1353` and their passed checks are in
[the completed checklist](archive/through-fce1353/TODO.md).

Completed phases from `e0073a5` through signed `a774ded` are in
[the next completed checklist](archive/through-a774ded/TODO.md).

### 837e77f — legacy offline acquisition eligibility

- [x] Reproduce five legacy offline policy failures before implementation using eight public-API tests: correct-hash LFS pointers passed repository/component eligibility; incomplete GGUF/index shards and reset index budgets passed component opening. All five failed at intended assertions, with three existing-semantics positives passing; generic CacheStore full verification legitimately accepted literal bytes.
- [x] Add one private managed-descriptor128B prefix helper using original pointer policy, cancellation checks and premature-EOF refusal. Legacy HF/GitHub offline API remains initial-selection/role-agnostic and keeps its public signature; generic cache inspect/verify remains literal-byte access. No second hash/cache/source issuer, new transport or realtime cancellation promise.
- [x] Reuse original pure split/index/sidecar closure only for explicit component Weights snapshots. Existing independent config/tokenizer suppliers, source/release coherence, roles and totals stay authoritative. Actual retained index reads share cumulative max_metadata_bytes/max_pages bounds across the whole component set; complete nested/split/release positives pass.
- [x] Focused production/diagnostic legacy8+8 and original API/lib/component/release/length/progress/transaction183 plus original CLI8+8 pass:215 actual executions/0 failures/2 intentional helpers ignored. All four strict artifact/CLI production/diagnostic configurations and scoped formatting pass. No unexpected compile/test/lint failure or post-GREEN source correction.
- [x] Full frozen isolated check.sh passes810 Rust executions/0 failures/68 ignored/174 suites plus18 namespace tests and complete lint/build/release/installer/comparator/resume gates. Original release cache_prepare passes both original routes in42.34s with both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation criteria exact. Actual Nix source-integrity includes the new test; protected README, lock and dependency manifests remain unchanged. No new model/context/GPU qualification.
- [x] Integrate all findings/results and retain native Xet, conditional/cloud/provider/convenience and separately authorized filesystem work open. Move eight completed groups and ten checked gate records through a774ded verbatim into a linked Markdown TODO archive, keeping every unfinished item live. Retire the completed content-contract worktree after14 exact source comparisons and its two integrated runs plus two primitive runs; preserve active worktrees/runs. No evidence archive, download, live account/cache/device/privilege/mount action or publication.

### 52e42fc — native HF Xet reconstruction

- [x] Wire the same exact Xet primitives into explicit native HF discovery, scoped CAS and signed-range reconstruction. Preserve original immutable HF source/pin/inventory/selection/closure and the sole HTTP/executor/cache/publisher. At least one announced native LFS object is required; discovery is body-free/no-follow, ordinary Git sidecars retain their original route and announced native failures never substitute compatibility HTTP.
- [x] Add sealed native Hub/CAS/signed-range capabilities in the original HTTP owner. Keep transient64K scoped grants separate from original bearer limits; fixed CAS origin/repository/commit and expiry apply. Native metadata and signed payloads never follow redirects; exact raw signed URI and complete interval sets are retained. CAS/signed calls suppress original credentials, including same-origin cases. Ordinary sidecar redirects use the current wire URL and strip cross-origin auth.
- [x] Validate ordered/repeated/overlapping terms and exact full-file lengths/coverage, unique bounded maps, arbitrary-order complete multipart sets, codecs, terminal source EOF and complete keyed Xet file root before publication. Fetched edge/refetched chunks count decode/work but not logical progress/root; accepted-leaf replay compares hash/length. Original raw SHA/size/LFS/cache-hit/publisher remain independent, with no invented per-term or partial-xorb authentication.
- [x] Enforce invocation-wide actual attempts/observed body+control bytes/decoded chunks/logical leaves/refreshes and cumulative accepted term/group/range counts across files/reissues. Overflow charges at most one sentinel then forbids later reads/attempts; exact-limit EOF probes work. Scratch is admitted before allocation and held alongside the original transaction. Native CLI default256MiB covers both envelopes; explicit limits and ordinary128MiB remain binding/unchanged. No global RAM/kernel/TLS traffic or whole-operation deadline claim.
- [x] Add exact ephemeral object streams through the existing cache locks/admission/private stage/installer: no native HTTP-prefix recovery, validators or checkpoints. Actual child exit73 preserves the old ref and a fresh process starts a full native transfer; ordinary sidecars keep qualified durable resume. Root review caught a draft blanket-range restriction affecting ordinary sidecars; narrow allow-only-OrdinaryPayload fix and actual retry/range/budget/ref tests pass.
- [x] Add native CLI through the original shared fetch path, metadata/discovery-only dry-run and existing-cache offline branch before credentials/client/admission/signals. Offline source/selection/current bounds/pointer/full bytes pass without claiming native execution/freshness/access. Private-source/limit/miss/pointer/corruption and feature-disabled subprocess checks pass. Existing provider routes/defaults and dependency/lock bytes are unchanged.
- [x] Final artifact551pass/0finalfail/7intentional helper ignores across diagnostic188, original compatibility179 and production184; both artifact strict all-target configurations pass. Final CLI175pass/0fail/2helpersignored/56suites across production/diagnostic/default/three disabled profiles; both CLI strict configurations pass. Actual native HTTP, default128refusal/256success, complete weights+ordinary sidecars, codecs, reordered multipart, replay, expiry/reissue, cancellation, process exit and old-ref gates pass. Independent long128KiB keyed hash/tree vectors follow six official scalar-oracle checks; no expected result was fitted to Rust.
- [x] Record all repairs: initial E0624 test access failure (outer tee exit0 hid Cargo status) fixed with test-only factory; exactly7 unused pure helpers become test-only, without arithmetic changes. Initial executable154lib/1externalpass/19fail arose from payload.bin correctly triggering model closure; change synthetic auxiliary path and add a complete weights closure case. Next154lib/20externalpass/1fail was original sanitized State versus raw Limit in an ordinary-route assertion. CLI4pass/1fail was grammatical do/does help mismatch. All fixes are narrow, final identical cohorts pass, and no runtime tolerance/redaction/closure was relaxed. Independent generator copied65/66hex XOF prefixes before Rust; correct official32byte prefixes then verify all six official vectors before freezing custom goldens.
- [x] Complete final frozen isolated check.sh867 Rust executions/0fail/69ignored/181suites plus18 namespace tests and all lint/build/release/installer/comparator/resume gates. Original release cache_prepare2routes43.26s (44.87s release rebuild) pass both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation exact. Actual offline Nix includes all9new Rust files;28 current local links, protected README and unchanged dependency/lock manifests pass. This adds no model/context/GPU qualification or live-provider throughput claim.
- [x] Integrate implementation, reviews, numerical/protocol findings, failures/results and all pending work atomically for signed local acceptance. Keep conditional cloud, selected ModelScope sizing, metadata candidate discovery and the wider provider/loader/runtime frontier open. Independent cold reviews find no conditional/sizing blocker but their execution remains unrun; do not promote alias echoes to immutable commits. No model/SDK/provider payload/account/cache/device/privilege/mount action, push, duplicate runtime or evidence archive.

### 46cf5a5 — strong conditional cloud observations

- [x] Add explicit object-only S3/Azure conditional requests and pure pins with separate s3-conditional/azure-conditional source contracts. Bind normalized endpoint/profile/locator/path/role/exact bytes/strong ETag; final original raw SHA stays FirstSeen. HEAD is admitted before transport, body-free/exact200 with complete identity metadata and original provider guards. No caller SHA, fabricated remote version, implicit fallback, prefix or cap profile.
- [x] Add one private typed If-Match condition to the existing opaque HttpTarget, None in all six original/native constructors. Every initial/range/retry emits one exact condition; original validator checks ETag before body. A mismatched resume validator refuses before signing/wire. SigV4 and Azure standard SharedKey slot match independent HMACs; HTTP412 refuses without refreshed HEAD or downgrade. Preserve all native Xet isolation/raw-target/ordinary-resume/counter tests and old cloud defaults.
- [x] Reuse the original one-entry managed verifier with private ExpectedObject/ExpectedDigest::TrustedSha256|FirstSeen, preserving HTTPS ref behavior and native digest-only pins. Exact source/path/role/bytes/provenance/current limits/full hash/pointer/cancel checks remain. No dummy expected SHA, second cache/hash/runtime/publisher or public source issuer.
- [x] Add --conditional-read to existing S3/Azure object handlers and exact offline --bytes/--etag/snapshot inputs. Online dry-run performs HEAD metadata only; offline finishes before env/credential file/FIFO/stdin/client/admission/signals. Escaped JSON ETag receipts preserve declared observation without asserting remote freshness/access. Explicit version/caller-SHA conflicts refuse; original auth/SAS scope and shared fetch/publisher remain unchanged.
- [x] Root and independent review find no protocol blocker; three literal HMAC vectors independently match Python stdlib before Rust gates. Reconcile conditional delta onto signed native52e42fc, preserving all original/native target code/tests; eighteen unowned baseline paths stay exact and only eleven approved source paths merge. A scoped SIDE formatting import-order failure was corrected before freeze; no MAIN compile/test/lint/source repair occurs.
- [x] Final focused872passes/0fail/16intentional ignores:714artifact (443diagnostic/271production),151CLI (75production/76diagnostic),4default-diagnostic and3disabled-profile tests. New conditional API10diagnostic/8production, all wire/signature/HEAD/resume/ETag-before-body/412/old-ref/offline tests pass. All four strict artifact/CLI configurations pass; original provider/native/legacy/cache/process/fault regressions stay green.
- [x] Complete final frozen isolated check.sh886 Rust executions/0fail/69ignored/188suites plus18 namespace and full lint/build/release/installer/comparator/resume gates. Actual Nix includes both new tests and exact Azure fixture;28 local links/protected README/unchanged lock/dependencies pass. Separate exact ignored release cache_prepare routes pass1test17.40s and1test23.69s (sum of test durations41.09s, not one invocation/wall time), both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation exact. No new model/context/GPU or live-cloud qualification.
- [x] Integrate all implementation/review/results and remaining provider work atomically for signed local acceptance. Retire two completed native worktrees after26 exact native source matches and CLI9current+2exact superseded-baseline matches; remove five integrated native runs. Preserve active conditional/ModelScope/candidate/quant work. No new dependency, download, account/cache/device/privilege/mount action, publication or evidence archive.

### 4db5efb — ModelScope selected exact sizing

- [x] Recover absent selected ModelScope file/index Size with one admitted body-free exact200 HEAD at the same fullcommit Direct target. Cache the exact length once before index GET/final planning, retain native raw SHA/provenance and source identity. Unselected missing sizes do not probe; malformed present/null blob sizes still refuse, with existing directory-null behavior preserved. Update the old malformed fixture to explicit null and clarify CLI sizing help.
- [x] Extend only the original provider metadata Budget for HEAD requests and returned header controls, shared with inventory/index bytes and logical call limits. Exact identity Content-Length, file/snapshot limits and no-follow status/framing rules apply. Strong ETag is checked before index/payload bytes; weak/absent validators grant no resume authority. Shared HTTP, native Xet, cloud profiles, cache, runtime, public APIs and dependencies remain unchanged.
- [x] Root proposal/source review and independent five-path review find no policy blocker. Native official-token GETtree/HEAD/GET and no-follow wire tests pass; selected/index size caching, unknown/unselected/zero/oversize/malformed fields, cancellation/budgets/ETags/rawSHA/old-ref/closure/source identity cases pass. Alias/CDN authority stays explicitly open; synthetic metadata grants no real provider or loader qualification.
- [x] Initial diagnostic all-target command exits101:156library tests and earlier providers pass, new sizing1pass/9fail. The old missing-size error is absent from live source but present in retained non-test rlib/executable; preserved SIDE mtimes predate previous builds. Refresh ONLY5merged-file timestamps after exact byte equality, then identical rerun453pass/0fail/13helpersignored confirms causal stale build. No Rust, fixture expectation, protocol or tolerance repair. Future copies use fresh modification times; record this Cargo constraint in AGENTS.md.
- [x] Final focused891passes/0fail/16intentional ignores:733artifact (453diagnostic/280production),151CLI,4default-diagnostic and3disabled. New sizing10diagnostic/9production pass and original provider/native/conditional/legacy/cache/process/resume/CLI tests stay green. All four strict artifact/CLI configurations pass with no lint/source correction.
- [x] Complete final frozen isolated check.sh896 Rust executions/0fail/69ignored/191suites plus18 namespace and complete lint/build/release/installer/comparator/resume gates. Actual Nix includes the new sizing test;28 current local links/protected README/unchanged lock/dependencies pass. Separate exact ignored original release routes pass17.64s and23.60s, both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation exact. No additional model/context/GPU qualification.
- [x] Integrate findings/failure/results and every unfinished task atomically for signed acceptance. Retire completed conditionalSIDE after29exact source/baseline matches to46cf5a5 and its three integrated runs; preserve active sizing/candidate/quant/filter work. No model/SDK/provider payload/account/cache/device/privilege/mount action, push, new dependency/runtime or evidence archive.

### f22f759 — metadata-only pinned file candidates

- [x] Add bounded metadata-only HF/GitHub candidates API/CLI through the original pinned-tree owner. List all regular choices without selecting a family, probing lengths, reading indexes/payloads or publishing. Full inventory collision/control validation precedes subfolder projection; metadata admission and logical-call/body budgets remain cumulative.
- [x] Expose honest pinned-tree versus verified cached-subset coverage, optional sizes and filename-role hints. JSON labels stored-Git versus HF tree/LFS versus verified-content lengths; Git OIDs and cached FirstSeen hashes never become trusted expected SHA. Offline source/current bounds/full hashes/pointers precede credentials/client/admission/signals; original Files/Release semantics remain.
- [x] Independent candidate review catches a regular-ancestor collision hidden by subfolder projection; fix complete-inventory ordering and add HF/GH negatives plus real-directory positive before execution. No executed RED claim. Final943focused passes/0fail/16helpers and four strict configurations pass without a compile/assertion/lint/source correction.
- [x] Frozen candidate full isolated `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passes915Rust/0fail/69ignored/198suites plus18namespace/all normal quality gates. Two separate exact release cache_prepare calls pass: approved_cached_models_match_direct_source_exactly17.47s and approved_hf_import_models_match_direct_source_exactly23.65s (`--release --locked --offline -p zerogravity --features cache --test cache_prepare NAME -- --ignored --exact --nocapture`, same wrapper). Both approved.8B/2B/all248320logits/all42domains/two tokens/session/rawlease/mutation/reclamation remain exact. These are individual test durations, not one combined invocation.
- [x] Integrate implementation/reviews/results and every unfinished task for signed candidate acceptance. Keep explicit quant, full filters/excludes, sidecar/preset/auth/provider/runtime frontier open. Preserve root README and dependencies; no model/SDK/provider payload/account/livecache/device/privilege/mount action, push or new runtime. Native source and cache bytes grant no additional model execution qualification.
- [x] Actual candidate source-integrity Nix passes with four new files included; all9source paths match frozen SIDE,26local documentation links/root README/dependencies pass. Retire completed ModelScope sizing SIDE after5exact signed4db5efb/MAIN matches and no process use; integrate future finite CDN authority/credential/refresh boundaries before deleting four completed sizing/review/frontier runs.

### f197c9c — explicit GGUF filename quant families

- [x] Add private-field GgufQuantRequest and explicit HF/GitHub/native filename-quant selection within the original retained provider tree and closure. Keep public Selection/PullRequest shape, original source identity, mandatory sidecars, same admission/logical counter and sole shared fetch/publisher. No default/rank/fallback or tensor eligibility follows.
- [x] Reuse strict canonical split decomposition, ASCII1..32letter-led tag grammar and terminal basename boundaries. Every matching family must be complete before uniqueness; malformed/incomplete/conflicting-total/case/single+split ambiguities refuse. Independent review's directory -of-/tag counterexample is repaired with basename-only inspection, preserving full path family identity; regression cases drafted before execution, no RED claim.
- [x] New quant offline requires explicit snapshot digest and verifies original source/current bounds/managed hashes/pointers/complete selected closure plus Weights/Config/Tokenizer roles before credentials/client/admission/signals. Report retained-subset scope, keep other retained data and original Files offline semantics. New native quant fixture is announcement/plan qualification only, not a CAS execution or inference test.
- [x] Quant focused975pass/0fail/16intentionalhelpers and all4strictartifact/CLI production/diagnostic configurations pass without compile/assertion/lint/source repair. New11API/5enabledCLI/disabled profiles and all original provider/component/native/legacy/cache/resume cohorts pass. After interruption, all11MAINsource paths still equal frozen draft; the sole existing Cargo owner continued.
- [x] Frozen full isolated `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passes934Rust/0fail/69ignored/205suites plus18namespace, all formatting/workspace/dependency-isolation/lint/build/release/installer/comparator/resume gates. Source remains unchanged. No new provider/model/SDK download; installer fetches are original owned local fixtures.
- [x] Both original exact release cache_prepare routes pass separately under jobs2/Rayon4/native isolation: approved_cached_models_match_direct_source_exactly17.60s; approved_hf_import_models_match_direct_source_exactly23.70s (`cargo test --release --locked --offline -p zerogravity --features cache --test cache_prepare NAME -- --ignored --exact --nocapture`). Each verifies both approved.8B/2B/full248320logits/all42domains/two tokens/sessions/rawsource lease/mutation/reclamation exact; individual reported test durations, not one aggregate call.
- [x] Wrap currentphase after explicit user pause: revalidate HEAD/worktrees/all11MAINfrozen sourcepaths and all15C3baselinepaths/C4exact3draftdeltas, originalrootREADME/deps and26local links; actualNix passes after3newpaths included and again after interruption. Both draft owners stopped/nochildren, full design/pending tests preserved in namedSIDEs. Quant validator returned exclusive Cargo after allcommands/children exit0. Integrate acceptance and all unfinishedwork for signed local commit, then pause; no push/download/model/GPU support change. Earlier C1 rolling-doc script expected7checkeditems but correctly stopped before mutation on actual6; inspected/corrected count and rolled exactly6, no implementation or test change.
- [x] Actual quant source-integrity passes after3newfiles add-N; all11source paths match frozen draft,26local links/root README/allCargo manifests+lock unchanged. Root approves C3 isolated source draft with fixed bounded fnmatchcase/exclusion/closure semantics and C4 native-index isolated source draft with explicit apply opt-in/default no-CAS and all-counter continuation; neither draft is execution evidence.

### 1a22597 — completed comprehensive handoff and source preservation

- [x] At the owner's documentation-only request, create the [comprehensive17Octoberhandoff](../../docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md) and [copyabletakeoverprompt](../NEXT_AGENT_PROMPT_2026-10-17.md), preserving runtimepause. Consolidate all U056–070 decisions, V1–V5.1/currentcontracts,77reachable commits, all121openplanitems, actualprofile/test/failure limits and complete document/research catalogs. Independentreview recovers an older94-source U040/G3b archive nested at7e61cb6; compare all94ranges and13+3 outersections exactly, retain explicit original33graph/later129/191 gaps. Preserve two byte-exact portable sourcepatches/fourunfinishedfiles plus pausedcontexts/inertoracle underdocs; actualapply and independentin-memory reconstruction pass. Links/coverage/README/appendonlydecisions/runtime/deps/index/pausedSIDEs remain intact. No runtime/model/build/GPU execution or publication; retire only completed documentationruns after integration.

- [x] After signedf197c9c acceptance, retire completed quantSIDE after15exact source/baseline matches toHEAD/no process use and exact cold-design copy in preserved indexSIDE; remove four integrated completed quant runs. Leave interrupted filters MAIN/SIDE runs and both unfinished source worktrees. This rolling TODO update belongs to the next commit; no new implementation or testing remains active.

### 04363c3 — bounded C3 Unicode filters and exclusion closure

- [x] C3 full-path filters resumed and source-qualified after the pause:
  exactly 11 intended Rust paths (3 new, 8 changed), fresh copy mtimes,
  one-pin HF/GitHub/native-Xet filtering, required-index-before-GET/shard/
  config/tokenizer veto, optional omission, exact retained offline closure,
  old Files/quant policy and shared original cache/admission/publisher.
  Independent CPython3.14 literal73 +33345-pair frozen digest, 4 filter
  matcher/adversarial units, 11 API integration tests (including actual
  private cache/ref publication on both providers), 4 enabled CLI tests
  and disabled-feature refusal pass. The original C2 candidate/quant suites
  and real private-loopback HTTP/TLS fixtures continue to pass.

- [x] Frozen C3 complete isolated check.sh: 956 Rust executions/0fail/
  69ignored/212 suites, plus 18 namespace fixtures, strict artifact/CLI
  production+diagnostic lint, workspace/dependency isolation, CPU build,
  installer, shell/Nix formatting and reference/comparator/resume self-tests.
  The first long run was interrupted during release compilation by a tool
  runtime failure (not an implementation test failure); the next frozen
  full run exited0 after the source-equal release build completed.

- [x] Actual Nix filtered-source integrity succeeds with all 3 new Rust
  paths included; checked current Markdown links, unchanged root README,
  Cargo.lock/manifests and no unrelated source differences. Two fresh
  separately invoked isolated original release cache_prepare tests pass:
  direct/cache 16.17s and HF-import 21.64s; each validates both approved
  Qwen3.5 checkpoints/all248320 logits/all42 state domains/2 tokens/sessions/
  original rawlease, mutation and removal behavior bit-for-bit. No new
  model, GPU, context or performance qualification is inferred.

- [x] Record the October17 user pause as a historical handoff only:
  C2 was signed at f197c9c, two preserved source drafts survived, and the
  explicit October18 resume moved C3 onto MAIN without losing C4.

- [x] Recover and complete the C3 paused source from
  /home/pvl/spaces/abird/src/zerogravity-filters-work. At takeover only
  filters.rs existed and was unwired/uncompiled; its frozen primary vectors
  and source/ownership policy are now implemented with API/CLI/native/closure
  regressions. Preserve the original paused source/run for traceability until
  reviewed retirement; no new SDK/model/provider/account payload fetched.

### 8195c18 — verified required native Xet indexes

- [x] C4 required native Xet indexes recovered from the preserved
  /home/pvl/spaces/abird/src/zerogravity-xet-index-work source-only draft.
  One isolated C3-base worktree performed a strict three-way merge of the
  three C4 production source deltas; exactly two HF/Xet dispatch conflicts
  were resolved without losing C3 selection/required-index veto. A cold
  typed IndexReader replaces the draft's uncompileable higher-ranked FnMut
  closure; original HF metadata parser, native Access/reader and publisher
  stay authoritative. Only seven reviewed Rust paths differ from C3 MAIN
  (five modified and two new integration tests). The original paused SIDE
  and its portable patch/design context remain untouched.
- [x] Preserve default library resolution and Xet CLI dry-run as body-free:
  required native LFS indexes refuse before CAS grants or signed body.
  Only CLI APPLY clones native options with explicit
  with_index_reconstruction; no new public flag, network client, cache,
  source issuer, device/runtime or parser. Original same-commit pinned
  index gets exact length+trusted LFS rawSHA, full keyed Xet root/EOF,
  raw SHA and LFS-pointer verification before JSON parse/shard closure.
  Planning uses one original admitted scratch; publication reconstructs
  and verifies the same index again in the existing cache transaction.
- [x] Carry ALL Xet consumed attempts, observed wire/control bytes, decoded
  chunks, logical leaves, refreshes and accepted terms/groups/ranges as a
  validated Result seed across resolution, publication and later weights;
  poisoned/tightened owners never silently zero. Native default256MiB
  admits index planning; explicit128MiB denies before signed index data.
  Ten new real original-owner API tests include independent abc Xet golden,
  exact index→shard→cache/ref publication, C3 native filtered positive
  and excluded-index-before-GET negative, wrong SHA/root/JSON/truncated
  signed body/cancel/old-ref, missing shard and aggregate budget refusal.
  Two private budget tests, two enabled CLI and one default-disabled CLI
  test also pass. Existing native Xet, C1 candidates, C2 quant, C3 filters,
  HTTP/TLS, offline and cache regressions remain green.
- [x] Final frozen isolated C4 check.sh exits0: 973 Rust executions,
  0 failures, 69 ignored, 219 suite runs, plus18 native namespace fixtures
  and full CPU/dependency/strict-lint/build/release/installer/Nix-format/
  comparator/resume gates. All four strict artifact/CLI production/diagnostic
  configurations pass after one test-only initializer style correction.
  An early 4/5 fixture failure selected an absent shard before reaching
  the required index; corrected fixture to select existing index. A
  separate truncated-body fixture initially ran out of scripted retries;
  bound its original request budget so refusal occurs before unscripted
  transport. Both corrected focused suites and final gate pass; no
  source arithmetic/protocol/tolerance relaxation or claimed RED test.
- [x] Actual final offline Nix filtered-source check includes both new Rust
  tests. Current local document links, original README, Cargo.lock and
  all dependency manifests are unchanged. Two separate current accepted
  Qwen3.5 release cache_prepare routes pass from the real cargo command:
  cache/direct16.08s and HF-import21.67s; each covers both original checkpoints,
  full248320 logits, all42 state domains, two tokens, independent sessions,
  source leases, mutation and reclamation exact. No new model/GPU/context,
  live provider/account/SDK/data download, privilege/mount/device operation,
  push or publication is implied.

### f56b574 — trusted AWS-standard FULL_OBJECT SHA256

- [x] C5 AWS-standard FULL_OBJECT SHA256 checksum admission:
  one six-file Rust slice with explicit --aws-full-object-sha256 and
  library builder. Only canonical AWS commercial-region immutable-version
  object HEAD requests x-amz-checksum-mode: ENABLED; original SigV4 signer
  includes the exact header, anonymous public profile sends it literally.
  HEAD redirects and absent/COMPOSITE/malformed/untyped checksum data refuse;
  only canonical base64 raw32 SHA256 from FULL_OBJECT plus matching pinned
  version/length becomes TrustedSha256. Independently supplied caller SHA
  must agree. No compatible endpoint, conditional/content mode, prefix or
  other algorithm gains remote SHA authority or fallback.
- [x] Preserve source/byte/provenance and original single publisher;
  dry-run remains metadata-only, offline requires the observed --sha256,
  exact version/length/snapshot before credentials. Seven dedicated
  original executor/cache API fixtures (exact AWS checksum, trusted vs
  FirstSeen, incompatible scope, malformed/composite/missing SHA, offline
  source, wrong GET/ref rollback, admission/cancellation) and one S3 CLI
  extended enabled test pass. One independent frozen Python-stdlib SigV4
  literal golden and two actual signed/unsigned private-loopback HTTP HEAD
  /redirect tests pass; real S3/C3/C4 regression suites remain green.
- [x] Final frozen C5 source/acceptance checks: 984 Rust executions /
  0 failures / 69 intentionally ignored / 222 suite executions in
  isolated check.sh, plus18 Python namespace fixtures, complete CPU,
  strict artifact/CLI production+diagnostic Clippy, release/installer,
  shell/Nix formatting and reference self-tests. Source-integrity passed
  using git add -N for new S3 API suite; local Markdown links, root README,
  Cargo.lock/manifests unchanged. Two independently invoked original
  release cache_prepare model routes pass: cache/direct16.03s and HF
  import21.45s, each both approved Qwen3.5 checkpoints/full248320 logits/
  all42domains/two tokens/independent sessions/source leases/mutation/
  reclaim exact. No inference/multimodal/GPU performance qualification
  is implied. Acceptance requires a signed local exact-path commit;
  no push or external provider payload testing is implied.

### d5aa5cb — original-model v1 source closure and execution cutover

- [x] Model-pull v1 execution cutover: close the finite companion gap
  for original index-first Qwen3.8 and Gemma4-like source layouts,
  using one shared bounded optional same-folder companion list in original
  HF and shared GH/cache closure. Known chat template, processor/image/audio/
  video metadata and alternate tokenizer files remain opaque source bytes;
  no arbitrary directory recursion, inferred model variant or auto-loader.
  Source-only two-provider synthetic >51GB indexed shard tests confirm
  no weight GET or cache publication when disk admission is insufficient,
  plus explicit GGUF/companion filtering/source closure and CLI index-first
  large-disk-budget argument validation. Keep 4GiB executable-source gate,
  original model numerical/CPU/GPU and authorization tasks in model-execution.
- [x] Final frozen model-pull v1 source/transfer gate: 988 Rust executions,
  0 failed, 69 intentionally ignored, 222 suite executions under exact
  original scripts/check.sh with private user/network namespace and guarded
  loopback. Strict artifact and CLI production+diagnostic lint, CPU workspace,
  release build/installer, Nix formatting and 18 Python namespace fixtures
  pass. Actual offline Nix filtered-source-integrity passes; root README,
  Cargo.lock/manifests unchanged, current local plan links checked.
  Independently reran original approved release cache_prepare sources:
  cache/direct16.38s and HF-import21.88s (both 0.8B+2B,
  all248320 logits/all42domains/2 tokens/source lease/mutation/reclaim exact).
  New source-closure tests cover original HF/GH GGUF+12 known metadata
  companions, filtered optional omissions, strict exact offline subsets
  and synthetic two-shard >51.6GB index plan refusing insufficient disk
  before any weight GET; no actual weight/model/SDK download and no license,
  trained-execution, GPU or general-provider compatibility qualification.
  Four bounded Rust files and four plan files only; remaining separately
  qualified provider/mirror/gating/filesystem/SSD throughput work stays
  explicitly unchecked without obstructing the model-execution CPU-first plan.

## Uncommitted

- [ ] Canonical source-pull execution order (18 Oct decision): Qwen3.5 .8B/
  2B, Qwen3.8 27B; Gemma4 E2B/26B-A4B/31B, each exact chosen GGUF
  quants or indexed original safetensors; QwenImage2.1 original
  Diffusers components; Qwen3 TTS/ASR; Whisper; NVIDIA Canary and
  NeMo Parakeet; Qwen3.8 Flash-Next original plus separately sourced
  Q2/~3.5-bpw/Q4 quants and independently verified MTP/n-gram
  embedding banks. Do not guess literal Q3.5 quant tags or conflate
  runtime speculative n-gram with trained n-gram checkpoint tensors.
  Implement exact typed presets/component assembly and run real
  metadata-only then authorized actual payload/sidecar/checksum
  publication gates. DeepSeek V4.1 Flash and GLM5.3 Flash have an
  explicit NO PAYLOAD download hold; prepare metadata pipeline only.
- [ ] Extend the new bounded HF/GitHub Auxiliary-only --bundle to
  source-verified Diffusers model_index nested shard sets, ASR/TTS
  processors/codecs, NeMo archive streaming/extraction and
  release-specific recursive dependencies under one pin/manifest.
  Current --bundle is a user-selected subset and cannot claim a
  complete model/codec assembly, support auto-loading, or run without
  newly qualified source parser and license checks.

- [x] Complete the observed public Qwen3.8+Gemma4 source metadata
  filenames in original pinned same-folder closure. 2026-10-18 actual
  tokenless HF --dry-run returned Qwen3.8-27B commit
  1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0,
  18 shards/55,586,113,293 selected bytes; Gemma4 26B-A4B-it commit
  4d7ae4984b7db7de8f8457170b3f1a419ee76d52,
  2 shards/51,644,338,194 selected bytes. A third tokenless
  HF --dry-run resolves Gemma4 E2B-it commit
  3e22461f65e89153144f8adb70e3b8c2cc9845a7,
  1 unsharded safetensors/10,278,848,001 selected bytes. The actual
  observed Qwen video_preprocessor_config.json and crc32.txt (opaque
  non-trust data) required two additive optional companion filenames.
  Both present in real result; no credentials, tensor weights, cache
  refs, SDK, model execution or GPU touched. Synthetic HF/GH model/quant/filter and
  offline exact closure tests cover both additional filenames.
  Final exact source cutover gates pass: isolated check.sh 988 Rust
  executions/0fail/69ignored/222 suites plus18 namespace fixtures and full
  strict production/diagnostic lint, CPU/release/installer/Nix-format/
  reference self-tests. Actual offline Nix source-integrity and local
  docs links pass; unchanged root README, Cargo.lock/dependency manifests.
  Separate original release source/cached model regressions both pass:
  cache/direct16.88s and HF-import21.72s, each both approved Qwen3.5
  checkpoints/all248320 logits/all42 state domains/two nonzero tokens/
  independent sessions/lease/mutation/reclaim bit-exact. This remains
  tokenless live metadata-only Hub resolution; there is no trained-source
  byte acquisition, inference, 8K drift closure, model-format or physical
  GPU qualification. Sign only the two additive Rust paths and relevant
  plan documentation; no push, tag or user-cache/device mutation.

- [ ] Extend cloud profiles beyond the narrow qualified AWS FULL_OBJECT SHA256 object route: broader S3 partitions/compatible vendors/credentials/checksum algorithms, Azure latest/SAS/managed chains and per-object digest-map prefixes. Keep explicit version, caller-SHA and conditional FirstSeen contracts distinct; never infer immutable versions or trusted raw hashes from unqualified validators or multipart COMPOSITE checksums.
- [ ] Extend ModelScope only through separately qualified alias authority and redirect/CDN/mirror/credential profiles. Selected missing sizes now use exact pinned HEAD; no alias-to-commit authority was established from revision echoes, timestamps or file last-modified IDs. The current profile does not close broader parity or independently prove server revision/membership behavior.
- [ ] Measure and qualify native Xet fetch coalescing/chunk caching and any persistent native-prefix protocol. The current bounded per-intersection complete-signed-set reader has counted overfetch/refetch and no performance claim; native process interruption restarts a full transfer. Keep per-term/partial-xorb proof limits, fixed CAS scope and existing elapsed-time/HTTP-trailer boundaries explicit.
- [ ] Close remaining finite convenience/compatibility gaps: required sidecar/adapter/sparse-quant policy, bounded auxiliary/presets, mirror/endpoint/env precedence and gate/license diagnostics. Preserve explicit source authority/no silent numeric fallback and separate loader/runtime plans.

- [ ] Qualify any future v1 migration only after separate authorization and proven reader/process quiescence; current removal supports originally protected roots only.
- [ ] Extend cleanup only through separately qualified exact ownership/recovery protocols for dead retained staging/general orphans and any later eviction policy. Current exact-snapshot operation deliberately retains them and every stable lock inode.
- [ ] Any additional lazy/raw-path/mmap consumer must adopt managed descriptor/open ownership or remain externally quiescent during removal. Observed named cache charge is not a process-global final-close/physical-free-space attestation.
- [ ] Qualify additional filesystem pairs before extending the explicit allowlist. Different st_dev values alone do not prove independent backing objects for Btrfs subvolumes, overlay/FUSE or virtualized views; retain branch-specific root device/mount checks and complete inode/ancestry exclusion. No unsupported pair gains acceptance from generic mocks.
- [ ] Separately authorize any needed namespace-only hidden-bind/subvolume/overlay filesystem fixtures before mount operations. Real target-root/unselected inode aliases and hidden bind aliases in both directions remain NOT RUN; no host mounts, block devices, new privileges or live cache actions are implied. Current tests use already existing distinct filesystems only.
- [ ] Independently qualify any broader local layouts, renamed/unnamespaced roots, missing mount identities or new destination-root creation before exposing support. Keep explicit selection, retained descriptors, bounded topology work and existing cancellation/rollback limits; real import still needs user-selected source/repository/commit/files and copying authorization.
- [ ] Finish the finite provider/loader parity ledger beyond qualified HF/GitHub/HTTPS/OCI/GCS/versioned-S3/Azure/ModelScope and explicit local-HF import slices: remaining external-cache layouts/filesystems, additional model hubs, unversioned/broader S3/Azure/compatible profiles, native Xet and broader OCI profiles. Opaque serialization/plugin loaders remain separate runtime work.
- [ ] Continue explicit source/license authorization and trusted-digest versus first-seen provenance through additional providers/loaders; acquisition grants no license or loading support.

## Tests and verification

- [ ] ACQ-A12: complete the finite provider/loader parity inventory and independently qualify remaining supported paths.

- [x] Legacy offline tests-first5RED/3positive, then215 focused passes/0 failures/2 helpers ignored and all4 strict configurations pass. Full frozen isolated check.sh810pass/0fail/68ignored/174suites and original cache_prepare2routes42.34s pass exact original criteria; actual source-integrity/current links/protected README/unchanged dependencies pass. No unexpected repair.
- [x] Native Xet final artifact551+CLI175 passes and all4 strict configs, documented test-access/fixture/grammar fixes only; full frozen isolated check.sh867pass/0fail/69ignored/181suites, actual source-integrity/28links/README/unchanged dependencies and original cache_prepare2routes43.26s pass exact original criteria. Native index closure/coalescing/persistence remain open.
- [x] Conditional cloud final872focused passes/all4strict/no MAIN repairs; full isolated886pass/0fail/69ignored/188suites, actual source-integrity/28links/README/unchanged dependencies and two original exact filtered routes17.40s+23.69s pass. Broader cloud authority/auth/prefix profiles remain open.
- [x] Selected ModelScope sizing891focused/all4strict and full896pass/0fail/69ignored/191suites pass; actual Nix/28links/README/unchanged dependencies and original exact routes17.64s+23.60s pass. Initial stale-library101 is resolved by byte-equal timestamp refresh, with no source/policy repair.
- [x] Candidate final943focused/all4strict and full915pass/0fail/69ignored/198suites+18namespace pass without repairs; actualNix/26local links/README/dependencies and original exact cache/direct17.47s+HF-import23.65s pass. Coverage remains metadata discovery/verified retained subset.
- [x] Quant final975focused/all4strict and full934pass/0fail/69ignored/205suites+18namespace pass without repairs; actualNix/26local links/rootREADME/dependencies and original exact cache/direct17.60s+HF-import23.70s pass. Interruption revalidation found no source drift; user requested pause and unfinished C3/C4 drafts remain preserved.
- [x] C4 exact native LFS-index all-counter/source/authorization validation,
  original HF/GH/Xet/C1/C2/C3 plus CLI/default-offline regressions,
  all four strict feature/lint configurations and final frozen source-integrity
  are qualified above. Full final check.sh 973/0fail/69ignored/219suites and
  two original release cache_prepare routes16.08s/21.67s pass separately.
- [ ] Each remaining provider phase requires focused API/CLI/protocol/auth/version/range/offline/fault gates, full frozen isolated check.sh, relevant source validation, actual source-integrity and documentation/dependency/root README checks before signing.

## Deferred

- [ ] Unsupported plans fail closed until qualified: unknown-length required/model-closure files, capped resumable transfers, unqualified external-cache layouts/filesystems and remaining cloud/OCI profiles, legacy-root removal and unmanaged-reader concurrent reclamation remain explicit above. Independent full HTTP trailer validation is unqualified under the documented decoder boundary, rather than a reason these payloads universally refuse. Independent component data grants no additional loader eligibility.
- [ ] Linux local-filesystem semantics are qualified first. HTTP cancellation obeys configured whole-request bounds; old retained attempts consume real quota and arbitrary same-UID mutation remains outside descriptor isolation.

````

## Source: .agents/plans/model-acquisition/archive/through-a774ded/TODO.md

Original UTF-8 bytes: 37926; SHA256 `4a178da79a9bc842b39d797f3a8141b28525e5a97cba3a11ba304c8edab27667`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — completed model acquisition phases through a774ded

Provenance: a774ded
Main plan: [model acquisition](../../TODO.md).
Earlier phases: [through fce1353](../through-fce1353/TODO.md).
This completed checklist does not close the remaining acquisition plan.

## Committed

### e0073a5 — pinned HTTPS objects

- [x] Add URL-free HttpsObjectSpec for a strict origin/logical path/role/exact bytes (including zero)/caller raw SHA256. Validate borrowed path/digest before cloning; reject raw nonbare origin/dot-segment paths, userinfo including empty @, fragments/controls and mismatched URL origins. Use existing URL normalization before original ResolvedPullPlan construction; production HTTPS and diagnostic literal-loopback policy remain.
- [x] Derive domain-separated versioned source IDs from normalized origin and the complete content contract; provider=https-object, origin hash and contract hash. URL paths/queries/token values never enter Debug/manifest/receipt metadata. Signed locator rotation within the declared origin preserves contract identity; ETags/Git IDs/URLs do not authenticate raw SHA.
- [x] Admit1MiB bounded operation-local prepare controls before URL parse/clones; perform no HEAD/metadata/payload request. Pure open_cached precedes URL/credential/client/admission input and verifies original managed snapshot/ref bytes, exact single-entry source/path/role/length/SHA/TrustedSha256, caller limits and original LFS-pointer refusal via positioned reads.
- [x] Expose explicit pull object with public origin/file/role/bytes/SHA and bounded --url-file ABS|- only. Regular no-follow/nonblock/CLOEXEC file opens refuse FIFO/device/leaf-symlink input;16KiB plus one optional LF/CRLF, sanitized errors; explicit stdin may block. Dry-run is local caller-contract validation before env/client/cache writes; offline --snapshot precedes URL-file/stdin/token/client/admission/signal setup. Intermediate path capabilities/global RAM/TLS lifetime reservation are not claimed.
- [x] Reuse existing exact/full-peak transfer admission, HTTP/redirect/range/resume/hash/EOF/LFS owner, installer/publisher, signal/progress and typed completion. No mutable ref first slice; independent review removed draft --ref support before compilation and added refusal coverage. Existing offline refs remain selectable. Optional --token-env NAME is origin scoped; generic help now scopes repository closure to HF/GitHub.
- [x] Add only acquisition-enabled Linux CLI dependency edge to already locked rustix1.1.5 for safe input. Offline Cargo metadata changes one lock edge, no package/version/download. Existing parse_url/reject_pointer helpers gain visibility only; CPU/core/cache-only dependency exclusions remain intact.
- [x] Focused API/regression cohort160 passed/0 failed/2 helpers ignored (library101 including8 new object units, capped12/components13/lengths12/progress11/transactions11); both artifact production/combined-diagnostic strict all-target lints pass. Independent cohort35 passed/0 failed/0 ignored: object5 diagnostic (actual trusted TLS success/untrusted-root refusal),4 production,8 production API units, CLI4new+8prior, default+diagnostic4 and minimal/cache-only disabled1 each; both CLI strict lints pass. No compile/test/lint failure or post-test source fix.
- [x] Full frozen isolated check.sh passed653 Rust executions/0 failed/68 ignored/126 suites plus full lint/build/release/installer,18 namespace Python tests and comparator/resume self-tests. Actual offline Nix source-integrity and25 current local links pass; root README unchanged and one tracked lock with only the reviewed CLI edge.
- [x] Both original cache_prepare routes passed2 selected tests in40.55s (release preparation0.14s), each approved Qwen3.5 .8B/2B over two tokens/nonzero state. All248320 logits/all42 domains, independent sessions/source IDs, foreign-source restore refusal, rawlease lifetime and private mutation/removal isolation remain exact. This adds no new loader/context/model/GPU qualification.
- [x] Move10 completed commit groups and37 checked gate records through fce1353 to a linked Markdown TODO archive, preserving completed text/link destinations and all unfinished work live. Two nonmutating documentation preflights expected wrong item counts (11vs10 groups;13vs12 open items); corrected before file writes. No evidence/source snapshots or hash inventories return.
- [x] Publish a current finite provider/compatibility frontier in README because the historical57-row captures were retired. Independently inspect native/pinned reference/official protocol surfaces; dispositions and Unsupported stubs do not close parity. Begin OCI/GCS work in detached worktrees during frozen-source verification and keep those phases unfinished.
- [x] Atomically integrate verified API/CLI outcomes/decisions/checks and retire completed HTTPS run records. No model/SDK/provider payload fetch, live-cache import, mount/device privilege change, GPU launch, push/history rewrite or root README edit. Continue authorized acquisition after the signed phase without waiting for another next prompt.

### e82d76b — native OCI raw GGUF layers

- [x] Implement native OCI/Docker schema2 manifest/raw-GGUF resolution through original transport/executor/publisher. Pin received raw manifest SHA before bounded parsing; validate caller pin, optional Docker-Content-Digest, response/JSON media agreement, known uncompressed GGUF media/descriptor SHA/exact size. Unique and explicit digest selectors refuse ambiguity/unrecognized layers/indexes/external URLs/compressed/container paths.
- [x] Bind versioned source identity to normalized origin/profile/repository/requested tag-or-manifest/logical path/layer selector; raw manifest digest is revision. Offline fully verifies original one-entry cached bytes/path/Weights/TrustedSha256/source/limits/LFS policy before credentials/client/broker/admission/signals. No retained raw manifest/independent descriptor membership, tag freshness, signatures or new model eligibility claim; no mutable ref.
- [x] Expose explicit anonymous/preissued bearer profiles and fixed public Docker Hub pull-only token broker; use original origin credential/redirect/TLS owner. Bound broker to min64KiB/half metadata and manifest to remainder with two-call Docker budget. Add static validated in-memory bearer factory; no arbitrary challenge realm/Basic/private login/refresh/ambient account discovery. No dependency/lock change.
- [x] Add explicit pull oci handler/options/help/offline receipt via existing limits/signals/progress/admission/executor. Preserve HTTPS route and existing source/runtime boundaries. Root precompile review caught missing real HTTP content-type/docker-content-digest retention; add exactly both under original duplicate-header guard and actual-loopback positive/negative regression before qualification.
- [x] API/source focused140 passed/0failed/2helpersignored, production OCI6 passed; both artifact acquisition-production/combined-diagnostic strict all-target clippy pass. Actual original HTTP positive metadata/payload and duplicate selected-header refusals, bounded auth/broker/digest/cancel/admission/old-ref/cache paths pass under isolated fixtures; no registry/model data fetched.
- [x] Independent CLI final23 passed/0failed/0ignored and both production/diagnostic strict all-target lints pass. First command13passed/3failed: one synthetic cache seed omitted TrustedSha256 provenance, two mutex-poison cascades. Correct fixture only to preserve resolved file provenance; identical16-test production cohort passes. OCI4 diagnostic and three minimal/cache-only/default-disabled cases also pass; production offline guard unchanged. No other compile/lint/test failure. Initial signed commit attempt lacked an agent socket (exit128, no commit); retry with the existing /run/user/1000/gcr/ssh socket signed successfully, no signing bypass.
- [x] Complete frozen isolated check.sh passed672 Rust executions/0failed/68ignored/133suites, full lint/build/release/installer,18 namespace Python tests and comparator/resume self-tests. Actual offline Nix source-integrity and25 local links pass; root README/Cargo.lock unchanged from e0073a5. Original cache_prepare isolated release regression2passed/0failed in41.86s (compile0.15s), both approved Qwen3.5 models/both routes/all248320 logits/all42domains/session/source/rawlease/mutation/reclamation exact. No new GPU/context/model qualification.
- [x] Merge detached OCI delta narrowly onto HTTPS e0073a5, review/stage10 explicit paths and integrate finished runs atomically. Document exact native profile/identity/offline limitations and primary OCIv1.1.1/Docker references; retain GCS/S3/research unfinished work. No root README change, model/SDK/provider payload download, live cache/privilege/device/mount action or publication. Continue next provider phase directly after signed local commit.

### f74a525 — native GCS generation/metageneration objects

- [x] Implement native GCS JSON metadata/object and bounded auxiliary prefix/pageToken resolution at fixed production storage.googleapis.com; exact bucket/name/generation/metageneration/size, independent caller raw SHA or FirstSeen. Ignore remote mediaLink/hash-label authority, reject nonidentity stored encoding/customer-key/soft-deleted profiles, zero-byte objects valid. Original resolve methods plus minimal typed resolve_with_pin(s) expose observed metadata without duplicated requests or source-string parsing.
- [x] Derive locally exact media generation/ifMetagenerationMatch pins and versioned endpoint/bucket/object/role/path/length/SHA source labels. Metadata can change within generation: metageneration protects representation. One private typed immutable response binding validates contradicting PRESENT x-goog-generation/metageneration before initial/resume/retry bodies; official JSON API permits absent headers. Exact length/SHA/EOF/ETag/206 remain separate mandatory guards.
- [x] Add fail-closed additive get_without_redirects capability to original HTTP owner, used only for bound media. Legacy custom transports return Unsupported, never silently delegate; original normal GET/HEAD behavior preserved. Refuse every pinned media redirect before Location lookup/body so query preconditions cannot disappear. Capture exactly two GCS version headers under original duplicate guard; no provider-specific executor or runtime.
- [x] Bound prefix JSON/pages/tokens/cycles/files/control/aggregate bytes, reject out-of-prefix/duplicate/unsafe/colliding/folder-marker selections and sort logical paths. Each object retains independent Auxiliary/FirstSeen source/pin/plan/publication, no atomic bucket snapshot/model closure. Return complete validated selection before transfer; later publication failure can leave earlier immutable data.
- [x] Expose explicit pull gcs object/prefix via original HTTP/admission/signals/progress/executor. Anonymous/default or explicitly named preissued bearer only; no ADC/service minting/metadata-server/SDK/refresh. Typed receipts print observed generation/metageneration/size. No refs; online defaults apply, metadata-only dry-run before cache initialization. Prefix offline Unsupported before I/O.
- [x] Pure GcsObjectPin/open_cached and digest-only object CLI offline require exact original bucket/name/path/role/generation/metageneration/bytes/optional SHA; original managed full-byte verification, trust/LFS/currentlimits checks precede env/client/admission/signal. No URL input, remote membership/freshness/revocation or checksum-strengthening claim. Caller bytes may zero; online rejects offline-only metadata flags rather than ignoring them.
- [x] Merge four new files/four shared seams from detached fce1353 onto signed OCI e82d76b preserving HTTPS/OCI routes/factory/header policy. First all-or-nothing three-way preflight hit competing CLI module additions (exit1) before any writes; review exact delta and reconcile only three GCS module/route/help additions. No source from failed preflight entered main. Precompile fixture split-write race and typed pin API refinements recorded before execution.
- [x] Focused artifact/source203passed/0failed/2helperignored: GCS14diagnostic/12production, library106, capped12/components13/length12/object5/OCI7/progress11/transactions11; both strict artifact configurations pass. Independent CLI27passed/0fail/0ignored (production20, default+diagnostic4, disabled3), both strict configurations pass. Actual original HTTP header/version/duplicate/query-preserving+stripping redirect fixtures pass. No compiler/test/lint failure or postmerge source fix.
- [x] Complete frozen isolated check.sh:693 Rust executions/0failed/68ignored/140suites plus full lint/build/release/installer/18 namespace/comparator/resume self-tests. Actual offline Nix source-integrity,25 current local links and unchanged root README/Cargo.lock pass. Original cache_prepare release2routes passed/0failed in41.83s, both approved models/all248320 logits/all42domains/source/session/rawlease/mutation/reclamation unchanged; no new model/context/GPU claim.
- [x] Integrate source/CLI/gate outcomes and remaining profiles into main TODO atomically, document primary GCS protocol/representation decisions, and preserve all unfinished S3/Azure/ModelScope/Xet/compatibility work. First source-format attempt found a missing brace (exit1), corrected before rustfmt pass; one adaptive patch found changed formatting and applied no changes, then corrected context succeeded. No evidence archives/SHA inventories/new dependencies, real cloud/model/SDK payload download, live cache action, mounts/privileges/device launch, root README edit or publication. Continue next provider phase after signed local commit.

### 833d68f — versioned S3 and SigV4

- [x] Implement versioned native S3 and explicit compatible bare endpoint/region profiles; canonical commercial AWS constructor, path-style/virtual-host addressing and bucket-specific credential origin. HEAD latest once or exact requested non-null version, exact size/identity metadata, original strong ETag when present. Null/unversioned/special partitions/directory/accesspoint/ambiguous TLS profiles refuse; no invented version or SDK/region/account discovery.
- [x] Add opaque private-constructor HttpTarget with typed private profile, additive default-Unsupported get_target/head_target and one original HTTP request/retry/TLS owner. Preserve raw key dot segments/repeated slashes, once-encoded UTF8/percent and sorted RFC3986 query through actual wire URI. No target redirects; existing normal GET/HEAD and GCS nofollow behavior preserved. Original executor dispatch changes only target-first branch; all existing response validation stays before bodies.
- [x] Add explicit redacted static/session SigV4 credentials and same-origin client factory; UTC re-sign actual method/Host/Range/If-Range/amz headers each attempt. Plain calls/extra credential origins refuse for signed profile. Require exact x-amz-version-id and no delete marker before initial/durable-resume/live-retry body, independently of ETag/206/framing/exact length/full raw SHA/EOF. Caller digest gives TrustedSha256, otherwise FirstSeen; ETag/MD5/CRC/composite labels never raw SHA.
- [x] Add admitted bounded ListObjectsV2 token and latest-live ListObjectVersions paired-marker prefix selection, exact per-object HEAD comparison and independent Auxiliary/FirstSeen plans. Validate whole selection, request/header/body/control/file/byte budgets, cycles/duplicates/path/collisions. Strict bounded UTF8/XML1.0 single document/declaration, no DTD/custom/external entities/PI. Later independent publication may retain earlier data; no atomic bucket/model closure.
- [x] Expose pull s3 explicit endpoint-profile/region/addressing/object or prefix and anonymous/SigV4 with named access/secret/session environment inputs only; no refs/caps/ambient AWS configuration. Exact version/bytes/digest offline pin verifies original managed bytes/source/role/path/trust/limits/LFS before env/client/admission/signals; zero bytes valid. Print escaped typed observed versions; metadata-only dry-run before cache initialization. Prefix offline Unsupported.
- [x] Add only acquisition optional cached hmac0.12.1/quick-xml0.41.0/time0.3.55 edges. Isolated metadata --no-deps was a successful lock no-op; targeted offline artifact update adds only cached HMAC/XML packages and existing digest/subtle feature edge. Structured comparison confirms all old versions/sources/checksums unchanged; no SDK/package download. Preserve default CPU/core/cache dependency boundaries.
- [x] Root/independent review catches XML outside-root entity/repeated declaration, owner catches XML1.0 raw/numeric controls and peer catches diagnostic parser incorrectly rejecting native HTTPS at3 new calls; correct all before first compile, with named negatives. Shared GCS reconciliation guarded preflights stop before writes, then extraction temporarily duplicates a block; declaration/static review removes it before compilation. Reconciled source preserves exact GCS module/guard/nofollow and HTTPS/OCI behavior. No numeric or production-policy relaxation.
- [x] Initial combined API/source197passes/0fail/2helpers, productionS3ninepasses; official3 literal signature vectors and actual signed GET/HEAD/range/raw-path/session/zero-redirect fixtures pass. Strict production lint initially101 on one MaxKeys collapsible_if; approve mechanical let-chain only, both strict artifact configs pass and final library115+S3nine plus productionnine rerun pass. Total339 actual artifact passing executions, initial lint failure retained separately. Durable/live exact-version success and changed/missing/delete-marker NoRead refusals pass, failed replacement targets actual keep ref.
- [x] Independent CLI final31passes/0fail/0ignored (production24/default+diagnostic4/disabled3) and both strict configs pass. Initial production20pass/4fail: missing n in help text caused one assertion plus3 mutex-poison cascades; correct one character only, identical24 passes. Final scoped formatter initially1 on root-merged module order; approved declaration-only reorder and exact before/after proof, finalformat/full gate pass. Tests/mutex/runtime unchanged; no hidden retries or tolerance changes.
- [x] Complete frozen isolated check.sh718 Rust executions/0failed/68ignored/147suites plus full lint/build/release/installer/18 namespace/comparator/resume self-tests. Actual offline Nix source-integrity repeated after both mechanical fixes/finalfreeze,25 current local links and protected root README pass. Original cache_prepare release2routes passed/0fail in41.06s, both approved Qwen3.5 models/all248320 logits/all42domains/source/session/rawlease/mutation/reclamation unchanged; no new model/context/GPU qualification.
- [x] Atomically integrate reviewed source/CLI results, all substantive precompile/lint/help/format failures and remaining unversioned/provider/frontier work; sign explicit verified paths only. Keep Azure/ModelScope/Xet source/research runs active, retire completed exact S3 side/run material only after useful outcomes integrated and signed source proof. No SDK/model/provider payload download, live cache/credential/account mutation, device/privilege/mount action, root README edit or publication. Continue next provider phase directly.

### e622222 — pinned Azure BlockBlob revisions

- [x] Add native Azure BlockBlob object/prefix APIs, checked opaque version/snapshot namespaces and pure typed offline pins. HEAD observes exact bytes and a transient strong ETag; full source contract binds origin/container/blob/revision kind/value/path/role/bytes/caller SHA. Caller SHA alone grants TrustedSha256; ETag/MD5/CRC64 remain separate. No mutable latest lookup/ref, encoded body, incomplete copy, customer-key or structured representation.
- [x] Extend the original private HttpTarget/auth/response-binding owner; preserve exact query and refuse redirects for properties, initial GET, durable resume and live retry. Require BlockBlob; reject contradictory present version echoes, while permitting the protocol's absent echo and keeping snapshot/version namespaces distinct. Reuse the sole original executor/cache/publisher, full raw hash/length/EOF/cancellation/old-ref boundaries.
- [x] Add explicit anonymous/bearer/account SharedKey and bounded ad-hoc serviceSAS. Reuse existing checked UTC/HMAC helpers; fixed REST2023-11-03, account-scoped64-byte key, exact actual method/range/resource/query signing. SAS stays private in the original client, preserves raw escapes, requires HTTPS/read/resource/version-window agreement and appends unsigned api-version. No SDK, ambient/managed credentials, mint/login/refresh, stored policy/delegation/account SAS or response overrides.
- [x] Bound complete current-version Auxiliary/FirstSeen XML listing, continuations, paths/collisions, copy/type/version properties and aggregate request/metadata/control/file/byte budgets. Prefix publications remain independent. Root review fixed omitted retained target/validator accounting before compilation; the meaningful two-long-ETag aggregate-budget negative passes.
- [x] Add explicit pull azure object/prefix, typed revision receipts and digest-only offline selection before every env/file/FIFO/stdin/client/admission/signal action. Parameterize the existing bounded nofollow/nonblock regular-secret helper for8KiB SAS without a second input owner; existing16KiB URL helper checks pass. Offline zero-byte/provenance/revision-kind/limits and hostile-input fixtures pass.
- [x] Add only the acquisition-enabled edge to already locked/cached base64. Offline update adds no packages; structured comparison proves all existing versions/sources/checksums unchanged. Root review found curated Nix omitted azure_wire.py; add exactly its allowlisted path and original-versus-filtered byte check. Final actual offline Nix source-integrity,25 current local links and protected root README pass.
- [x] Final focused isolated cohorts pass344 executions/0 failures/5 helpers ignored:258 API/source and86 CLI; artifact/CLI production+diagnostic strict all-target lints pass. Independent Python stdlib recomputes actual wire SharedKey signatures; public/bearer/SAS and no-redirect/duplicate-header fixtures pass. Independent peer cold review finds no blocker; no live Azure/account/provider payload fetch.
- [x] Record and resolve actual failures without relaxed policy: compile101 before tests (one module path and deprecated equivalent XML API);13-pass/1-fail short-body fixture omitted two legitimate original range retries; CLI one wrapped-help assertion plus three mutex-poison cascades; one manual-range-contains lint. Approve only narrow call/style and fixture corrections; identical cohorts subsequently pass.
- [x] Complete frozen isolated check.sh passes749 Rust executions/0 failures/68 ignored/154 suites, full lint/build/release/installer,18 namespace tests and comparator/resume self-tests. Source/API and filesystem/default-feature boundaries remain intact; no device launch, privilege/mount/live-cache action, model/SDK download, publication or root README change.
- [x] Original isolated release cache_prepare passes both routes in42.85s: approved Qwen3.5 .8B/2B, two tokens/nonzero state, all248320 logits/all42 domains exact, independent sessions/source identity/foreign-source restore/raw-lease/mutation/removal isolation unchanged. Atomically integrate results/decisions/failures and all pending tasks; sign explicit verified paths, then continue ModelScope/native Xet and broader object profiles. This grants no new model/context/GPU qualification.

### 87f6454 — native ModelScope explicit commits

- [x] Add native ModelScope explicit lowercase40hex commit/tree/file resolver and CLI using pinned official client0.4.6 protocol (65255a78b451c3785f2dfbfaff97172552fea3af). Fixed native origin, complete bounded shallow walk, direct-child/type/name/duplicate checks and exact sizes; refuse directories reaching the silent3000-row cap, exhausted walks and aliases. Endpoint revision semantics remain trusted, not independently proven Git/Merkle membership.
- [x] Reuse original HF selectors and extract private pure GitHub model/shard/sidecar closure with an admitted index callback. Preserve GitHub's original metadata/HEAD/raw URL/OID/regular-blob/length/LFS owner. Required sidecars, indexed/split shards and ambiguous-family refusal pass original regressions; source row SHA is TrustedSha256, absence FirstSeen, never Git OID authority. No dependency/lock change or new runtime.
- [x] Extend original opaque HttpTarget with no-redirect Direct profile and one-origin official-token client sending strict cookie-safe Bearer plus m_session_id. Metadata and payload redirects refuse; S3/Azure profiles cannot authorize Direct, and ModelScope credentials cannot authorize those targets. Actual isolated wire/header/redirect tests pass; no SDK, cookie jar, persisted login, refresh or live account/provider call.
- [x] Add digest-only offline exact source/selection/role/provenance/limits verification before env/client/admission/signals. Root review found generic manually seeded same-source data could contain an LFS pointer; a meaningful regression first failed at the intended assertion, then passed after bounded128B positioned reads on every original managed descriptor using existing pointer/cancellation checks. No path reopen, second full hash, source issuer or cache owner.
- [x] Final affected API9 production+9 diagnostic and HTTP/token1 production+3 diagnostic pass; original selected GH/HF/closure/components/release/length/cap/resume/cloud regressions pass. Seven-provider CLI matrix85 passes across production/diagnostic/three disabled profiles; final mechanical frontend change rechecks4. All four artifact/CLI strict configurations pass. These are actual isolated fixtures, with live provider/account and broader protocol profiles unrun.
- [x] Record narrow repairs: positive index fixture lacked mandatory total_size (199 prior+7 new passes/1 failure), one mutable-const warning fixed with a local reader, two actual-loopback tests needed diagnostic-only cfg to preserve production URL rules (125 passes/2 failures), pointer regression RED then GREEN, one test expect_err lint and one CLI let-chain lint. No parser, URL policy, assertion or numerical tolerance relaxation.
- [x] Complete final frozen isolated check.sh passes768 Rust executions/0 failures/68 ignored/161 suites, full lint/build/release/installer,18 namespace tests and comparator/resume checks. Actual final offline Nix source-integrity and25 current local links pass; root README, dependency manifest and lock remain unchanged. Original release cache_prepare2routes pass42.91s with both approved models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation exact; no new model/context/GPU qualification.
- [x] Atomically integrate all source/CLI/protocol/offline findings, failures, decisions and pending work for signed local acceptance. Retire the completed Azure worktree after exact equality/known final-correction review and remove its three integrated runs; preserve active ModelScope/Xet/unversioned/CLI work. No model/SDK/provider payload download, privileges/mount/device/live-cache action, push or root README edit; continue native Xet and explicit content contracts.

### 223ca79 — unwired Xet hash and chunk primitives

- [x] Add exact future Xet hash/codec sources with an exact-source test wrapper, presently unwired from production acquisition. Separate raw32-byte versus limb-reversed text identifiers, keyed chunks/internal aggregation, in-place ordered grouping/singleton tails, zero-key file finalization and empty-root rules. Xorb8192chunk/64MiB and file1M-leaf bounds/cancellation pass; raw SHA remains independent.
- [x] Add safe allocation-free NONE/framed-LZ4/BG4 decoding into caller buffers,256KiB encoded/128KiB decoded bounds, exact8-byte header and consumption, checksums, linked/independent history, BG4 remainders and terminal EOF. Reject bombs, malformed offsets/extensions/final sequences, truncations and dictionary/legacy/skippable/concatenated profiles. No network/cache/source issuer or allocator lives in these primitives.
- [x] Actual isolated cargo test --locked --offline -p zerogravity-artifacts --no-default-features --features acquisition --test acquisition_xet_primitives passes13/0fail/0ignored before and after the single mechanical fix. Literal saved upstream Merkle cases and independent scalar BLAKE3/XXH32/frame goldens test exact arithmetic/namespaces, corruption and cancellation. Independent peer source review finds no blocker; no invented per-term/partial-xorb authentication authority.
- [x] Pin protocol39aa2452668744552cc6690c75cd239771148e58, xet-core c3c726bed5cf54ded92a63fab892cfb7857c751a and LZ4 ebb370ca83af193212df4dcbadcc5d87bc0de2f0. Record pre-Rust diagnostics: official0..31 text example omitted byte0x10; use correct64char sequence plus verbatim saved source vector. Independent generator initially copied65/66 XOF characters into32-byte expectations; correct prefixes before custom vector freeze/Rust execution and verify four official checks. No expectation changed to fit new Rust output.
- [x] Add only cached optional acquisition edges blake3=1.8.2(std,pure) and xxhash-rust=0.8.19(xxh32). Offline lock resolution adds exactly arrayref0.3.9/blake3 1.8.2/constant_time_eq0.3.1 plus two artifact edges; every old package/version/source/checksum/dependency stays unchanged. Actual isolated cargo tree normal+build checks prove all four hash dependencies absent from CPU and CPU+cache feature closures.
- [x] First strict production lint exits101 only on codec match-loop manual_is_multiple_of; owner applies exact syntax equivalence in main/side, no math or vector change. Identical13 tests and artifact production/diagnostic all-target strict lints pass. Full final frozen check.sh passes781 Rust executions/0fail/68ignored/164suites plus complete lint/build/release/installer,18 namespace and comparator/resume checks.
- [x] Actual offline Nix source-integrity,25 current local links and protected root README pass. Original isolated release cache_prepare passes2routes41.70s with both approved trained models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation exact. Primitive tests do not enable native transfer, new model/context/GPU support or a throughput claim.
- [x] Integrate decisions/results/repairs and all pending native work atomically for signed local acceptance. Retire only the completed ModelScope side/two runs: all nine owned files equal signed87f6454; initial nonmutating whole-baseline equality catches three previously qualified Azure corrections, then all remaining22 source paths match. Preserve active native-wire/CLI/content worktrees and runs; no downloads, account/cache/device/privilege/mount action, push, evidence archive or root README edit.

### a774ded — explicit cloud content contracts

- [x] Add explicit S3ContentSpec/AzureContentSpec caller raw-SHA256/exact-size contracts (including zero), separate s3-content/azure-content source identities and typed current-object targets. Bind normalized source selectors/path/role/bytes/SHA; never invent remote versions or treat observed ETags/version/checksum headers as content authority. Existing versioned defaults remain unchanged and never silently fall back.
- [x] Prepare/dry-run locally under the original bounded control admission, with no HEAD/metadata/credential/client/cache publication. Apply reuses original exact peak admission, strong-validator durable/live resume, raw hash/length/EOF/LFS checks, cancellation and sole publisher. S3 delete markers refuse; Azure retains BlockBlob/identity/copy/key/structured checks. Actual malformed/current/mutated bodies preserve old refs.
- [x] Add explicit object-only --content-contract --bytes --sha256 CLI mode; reject revision/snapshot/ref/cap/prefix conflicts before I/O. Digest-only offline validates the same complete source/path/role/size/SHA/trust and actual managed bytes before env/file/FIFO/stdin/client/admission/signals. No no-SHA conditional or prefix digest-map fallback is introduced.
- [x] Reuse the original private trusted one-entry offline verifier for HTTPS/S3/Azure; keep existing HTTPS ref/cancellation/limits/positioned-prefix behavior. Rename only two extracted error labels to provider-neutral trusted-object text before compilation. No second cache/hash/publisher/source issuer or dependency/lock change.
- [x] Qualify Azure serviceSAS b/c only for Content, retaining bv/bs/c version/snapshot scopes, original raw escaped query/fixed api-version and independent SharedKey signing. Actual isolated versionless GET-only public/bearer/SAS/SharedKey and S3 SigV4 raw-target wire fixtures pass; old versioned wire fixtures remain intact through the same exact Nix-included Python input.
- [x] Final focused488 passes/0 failures/7 intentional helpers ignored:304 combined artifact/source,45 production,55/56 production/diagnostic CLI,4 default+diagnostic and24 disabled-provider executions. New content13 diagnostic/11 production and4 enabled CLI cases pass, all four strict artifact/CLI configurations pass. Zero compile/test/lint failures and no postmerge source/fixture repairs.
- [x] Complete final frozen check.sh802 Rust executions/0fail/68ignored/171suites plus full lint/build/release/installer,18 namespace and comparator/resume checks. Actual offline Nix source-integrity,25 local links, unchanged root README/lock/dependencies pass. Original release cache_prepare2routes40.55s passes both approved trained models/all248320 logits/all42 domains/source/session/raw-lease/mutation/reclamation exact; no new model/context/GPU or live-cloud/account claim.
- [x] Atomically integrate implementation/results/decisions and all pending conditional/cloud/native-Xet/legacy-offline work for signed local acceptance. Preserve original source/session/resource ownership and active side work; no download, provider payload/account/cache mutation, device/privilege/mount action, publication, evidence archive or root README edit.

## Tests and verification

- [x] HTTPS-object focused:160 API/regression executions/0fail/2helpersignored plus35 independent API/CLI/TLS/default-disabled executions/0fail/0ignored; artifact/CLI production+diagnostic strict lint passed. Actual trusted/untrusted TLS and child-process URL-file/FIFO/stdin/offline secrecy checks pass.
- [x] HTTPS-object full: CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh passed653 Rust/0fail/68ignored/126suites plus lint/release/installer/18 namespace/comparator/resume gates. Actual nix build --offline --no-link .#checks.x86_64-linux.source-integrity and25 current links pass; root README and one reviewed lock edge pass.
- [x] HTTPS original-source regression: isolated jobs2/Rayon4 cargo test --release --locked --offline -p zerogravity --features cache --test cache_prepare -- --ignored --test-threads=1 --nocapture passed2selected/both approved models/both routes in40.55s, original all-logit/all42-domain/session/source/resource/mutation/removal criteria unchanged.
- [x] OCI focused140+6 API and23 final independent CLI executions pass, strict production/diagnostic lint pass; initial one test-fixture provenance failure/two cascades corrected without production relaxation. Full isolated check.sh672passed/0fail/68ignored/133suites, offline Nix/25 links/README/unchanged lock pass; original cache_prepare2routes in41.86s pass exact original-model criteria.
- [x] GCS focused203 API/source+27 independent CLI executions pass, no postmerge source fix/failure; both artifact/CLI production+diagnostic strict lint pass. Full isolated check.sh693passed/0fail/68ignored/140suites, actual offline Nix/25 links/root README/unchanged lock and original cache_prepare2routes41.83s exact criteria pass.
- [x] Versioned S3 focused197+9 initial API/source and124+9 postmechanical reruns pass (339actualpasses); final independent CLI31pass, artifact/CLI strictprod+diagnostic configs pass after documentedlint/help/format repairs. Fullisolatedcheck.sh718passed/0fail/68ignored/147suites; finalactualofflineNix/25links/README/reviewedminimal lock and original cache_prepare2routes41.06s exactcriteria pass.
- [x] Azure focused344 passes/0fail/5helpersignored and all4 strict configs; documented compile/fixture/lint repairs retain original policy. Full frozen isolated check.sh749pass/0fail/68ignored/154suites, actual offline Nix/25links/root README/base64-only lock edge pass; original cache_prepare2routes42.85s passes all original exact source/state/session/resource/mutation/removal criteria.
- [x] ModelScope affected API/HTTP22passes plus CLI85matrix+4mechanical recheck; original selected regression cohort and all4 strict configs pass. Documented fixture/gating/style repairs and real offline-pointer RED→GREEN retain original authority. Full frozen check.sh768pass/0fail/68ignored/161suites, final actual offline Nix/25links/README/unchanged dependencies and original cache_prepare2routes42.91s exact criteria pass.
- [x] Xet unwired primitive tests13 pass before/after one mechanical lint fix; artifact strict production/diagnostic pass. Full frozen check.sh781pass/0fail/68ignored/164suites, actual offline Nix/25links/protected README/reviewed3cachedpackages+2edges/explicit CPU+cache closure exclusion and original cache_prepare2routes41.70s exact criteria pass. Native wire/CLI remain unqualified.
- [x] Unversioned caller-SHA content488focusedpass/0fail/7helpersignored and all4 strict configs, no repair. Full frozen check.sh802pass/0fail/68ignored/171suites; actual offline Nix/25links/README/unchanged dependencies and original cache_prepare2routes40.55s exact criteria pass. No-SHA conditional and broader native-cloud profiles remain open.

````

## Source: .agents/plans/model-acquisition/archive/through-fce1353/TODO.md

Original UTF-8 bytes: 58744; SHA256 `e0750281ddc39d99dac3e44542be9087d3fa18b305c0afa39c3785a502032d61`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — completed model acquisition through fce1353

Provenance: fce1353
Scope: completed implementation/results only; no unfinished work here.
Live plan: [TODO.md](../../TODO.md).

## Committed

### 3787519

- [x] Local Linux cache A1 accepted. Cooperative quotas are not global host/UMA reservation; descriptors do not prevent arbitrary same-UID mutation.

- [x] Actual A1 full gate:194 Rust executions passed,27 optional executions ignored; strict lint/release/installer/isolation checks passed. No inference/HTTP claim follows from cache publication.
- [x] Publication fault/recovery:18 operation types,38 callbacks,76 injected I/O and76 real child-exit fixtures; retained refs/leases and concurrent quota rebuild regressions passed.
- [x] Review fixes: retained named ref backing, private snapshot reservation accounting and hardlink-only ctime verification; deterministic regressions passed. Early readiness-marker and cfg-import harness issues were fixed before acceptance.

Technical API/bounds: [A1 cache reference](../../../../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md).

### fa6affa — bounded A2 acquisition

- [x] Add one optional synchronous verified HTTPS owner; bounded metadata/headers, redirects, origin-scoped credentials, identity encoding, retry/backoff and whole-request deadlines. Default CPU/core/inspection retain no HTTP/TLS/GPU dependency.
- [x] Extend the existing A1 attempt for exact-size unknown raw digests: one-pass admitted staging/hash, private non-publishable planning keys, actual inode accounting and persisted TrustedSha256/FirstSeen; legacy entries remain Unspecified.
- [x] Pin HF revision/tree and GitHub commit/tree or release/asset IDs; bound aggregate metadata, pages and filter work. Git object IDs never count as raw SHA256.
- [x] Close selected/adjacent safetensors indexes and shards, ordered GGUF splits and same-repository config/tokenizer/available sidecars. Missing closure, mixed weight families, changed exact pins and LFS pointer bodies refuse.
- [x] Add explicit optional facade/CLI pull, metadata-only dry-run, offline, list, inspect and verify. Offline branches precede credentials/client/provider; cache data grants no SourceIdentity or inference readiness.
- [x] Keep metadata and streaming work on the caller's existing ResourceArbiter TransferAdmission. Narrow production provider resolvers to the admitted entry point; prove denied/canceled resolution never requests metadata.
- [x] Validate actual loopback HTTPS chain/hostname/downgrade refusal, origin auth stripping, framing/limits, deadlines/cancellation, bounded retries and exact strong-ETag range append. Ignored200 or changed validators preserve prior refs.
- [x] Persist checksum trust strength, preserve exact original bytes and all A1 durability/lease/accounting limits; retain the concurrent user-authored6deebf2 XDG root abird/zerogravity/models.
- [x] Integrate independent cache/provider/transport/CLI reviews. Correct uppercase-pin checks, adjacent-index and weight-family omissions, public admission bypass and staged-index error redaction; regression tests cover refusal and exact old-ref preservation.
- [x] Correct ureq whole-body timeout interpretation and TRACE URL/Location disclosure. Header and total deadlines are distinct; acquisition-only log/max_level_debug caps process-wide log TRACE, with the downstream feature constraint documented.
- [x] Complete frozen-source gate:312 Rust executions passed,0 failed,30 optional/fixture entries ignored (counts include repeated feature builds); strict lint, production-only/diagnostic/combined features, CPU release/installer,18 namespace Python tests and comparator/resume self-tests passed.
- [x] Actual offline Nix source-integrity build passed, including exact synthetic TLS fixture retention;31 changed-document links, single tracked Cargo.lock and byte-identical user root README passed. Only four already-cached URL dependency packages were newly locked; no model/SDK/provider acquisition or device action.
- [x] Record incremental fixes: Read/Write by_ref ambiguity, BTreeSet iteration and fixture cfg scope; ordinary strict-lint borrow/tuple/as_ref/const/conditional reports were fixed without production allowances. The early four-boundary discovery trial was replaced by the complete40-boundary matrix.
- [x] Record gate/setup corrections: restricted /proc/1 namespace access replaced with mapped-UID plus NS_GET_USERNS ownership checks; safe private-loopback positive/host-map negative checks passed. First gate stopped130 for the reviewed redaction fix; the next run failed127 after editing its active Bash file. The final frozen script explicitly enables acquisition plus both diagnostic features and passed in full.

### 3c97c52 — A2 durable resume and pull UX

- [x] Add cache-owned durable proper-prefix records bound to immutable source/object/role/length/expected hash, strong ETag and actual inode/prefix SHA256. Recovery rehashes/copies into a new exclusive stage using the existing admitted attempt and sole A1 installer/publisher.
- [x] Batch checkpoints8MiB/1s plus interrupted-read/cancel paths; sync payload before atomic record replacement and parent sync. Ignore charged uncommitted tails and forbid saved prefixes at exact total until fresh HTTP terminal framing is proved.
- [x] Keep unchanged resumed prefixes ephemeral in the new attempt: repeated refused/no-progress retries preserve the original checkpoint without retaining duplicate copies. Genuine progressed old attempts remain named and charged; new full copy plus24KiB/file control peak must fit before transfer.
- [x] Close the public trusted-digest private-copy quota escape: hold one original digest lock through the guard, refuse already verified named content before stage creation, and reuse that lock during install. Direct same-hash alias/concurrent publisher tests validate physical charge and zero body reads on refusal.
- [x] Preserve execute and add borrowed source-bound execute_with_progress/PullProgress. Accepted file/aggregate bytes stay bounded and monotonic; cached, resumed and downloaded counts are distinct, and100% bytes is not publication.
- [x] Resume only exact If-Range/206 with matching validator/start/total/length/identity;200, changed source or malformed range refuse without changing the previous ref. Full raw digest and actual EOF remain mandatory, including empty objects.
- [x] Add optional Linux CLI SIGINT/SIGTERM flags and RAII callback cleanup, throttled numeric stderr progress and separate final stdout receipt; offline/help/cache/default CPU install no handler. Cached signal-hook0.3.18/registry1.4.8 are the only newly locked packages.
- [x] Share one bounded HF initial selector online and offline for exact/glob/subfolder/Unicode/missing/ambiguous/budget behavior. Offline verifies cached paths before credentials/client/provider; legacy/manual manifests gain no model closure or freshness claim.
- [x] Integrate three independent ownership/review streams. Fix exact-total EOF bypass, duplicate no-progress checkpoint copies, post-read/pre-write cancellation checkpoint loss and public known-digest quota escape; deterministic and real-process regressions pass.
- [x] Retain prior accepted bytes when a newly returned read is canceled before the cache write guard; checkpoint on both prefix and normal write errors. The deterministic case accepts128 of256 returned bytes, resumes at128 and preserves the old reference.
- [x] Verify final cache13, transfer11 and CLI/filter focused suites; preserve original A1 and discovery failure/process-exit regressions. Initial cache clippy needless-borrow report in the shared-installer refactor was fixed; strict rechecks pass without allowances.
- [x] Run the full frozen-source gate:353 Rust executions passed,0 failed,34 optional/helper entries ignored (feature repetitions included), strict lint/release/installer and namespace/comparator/resume self-tests passed. No numerical tolerance, inference operator, checkpoint model or device qualification changed.
- [x] Verify actual offline Nix filtered-source integrity,23 local documentation links, single tracked Cargo.lock and byte-identical root README. No dependency/model/SDK/provider acquisition, device action or publication occurred.
- [x] Integrate completed run outcomes into this checklist and design reference; retire temporary run checklists/logs. Keep independent A3, removal, component and parity gates open.

### baa9d88 — pinned components and A3 cached CPU preparation

- [x] Add explicit independently pinned component requests/plans, per-label credential transport adapters, bounded role suppliers and separate original verified snapshots. No composite source manifest, mutable ref, second publisher/runtime or model-readiness claim.
- [x] Reuse the first exact commit for repeated file-source aliases and preserve strict ordinary model closure. Relax only explicitly supplied config/tokenizer sidecars; retain missing-shard/index/split/role refusal. Repeated release aliases remain a separate unsupported frontier.
- [x] Bound actual aggregate component metadata bytes/requests, file counts, payload totals and plan metadata under the caller's existing transfer admission. Every component uses the original A1 executor with no ref; a complete receipt requires every publication durable and each descriptor lease verified. Earlier data can remain after later failure.
- [x] Add pure offline component opening before credentials/client/provider/admission, with original source/selection/exact-pin/role/shared-pin checks. Move the existing ordinary cached-selection validator into the common owner; no fabricated offline transfer receipt or freshness assertion.
- [x] Add original logical-name Artifact descriptor construction and Unix positioned reads. Cloned OS cursors cannot interfere; object/root/ref replacement cannot retarget the owned descriptor.
- [x] A3: verify the exact bounded owned source allocation against the manifest length/SHA256, reparse those same bytes and invoke the existing sole SourceIdentity issuer. Exact preallocation plus a separate EOF sentinel preserves the GGUF-only4GiB cap; mutation/digest/format/role/length/cap refusals release resources.
- [x] Admit raw ImmutableWeights bytes before allocation in the caller's existing ResourceArbiter domain and retain that lease inside SourceWeights until the actual allocation/all model/session Arc owners drop. Legacy ordinary open keeps its prior accounting boundary; metadata/tokenizer/session/global-RAM accounting is not asserted.
- [x] Add PreparedQwen shared cold preparation and bounded actual-copy config/tokenizer verification (1MiB/64MiB). Hold sidecar staging admission through parsing/construction; normal and cache paths reuse family validation and one unchanged CPU inference loop.
- [x] Keep the new family constructor safe: accept bounded config bytes through the existing parser rather than trusting publicly mutable ModelGeometry. Reuse a private checked constructor and cheap early family eligibility check before raw-source copying; no duplicate numerical validator or mathematical-body change.
- [x] Add explicit HTTP-free CLI cache feature and run-cache selectors, separate config/tokenizer snapshots, logical paths and raw-source/staging cap (default5GiB). Cache misses refuse, ordinary run never acquires, and default CPU/cache-only closures exclude HTTP/TLS/signal/GPU owners.
- [x] Add pull components PLAN.json with fixed schema,1MiB/16-component bounds, unknown/duplicate-field and conflicting-selector refusal, metadata-only dry-run, explicit offline bindings, separate label-scoped origin credentials and honest per-source output. Tokens/URLs are not plan fields and errors redact unrecognized input.
- [x] Integrate three independent implementation/review streams. Fix the broader-store/narrower-offline-file-limit gap with an actual public regression, unsafe mutable-geometry construction and late unsupported-family allocation. An early overly broad HF edit was corrected before compilation; no acceptance command failed or numerical tolerance changed.
- [x] Revalidate the six approved local checkpoint/config/tokenizer identities, CPU/RAM/disk and initial worktree. Qualify both original .8B/2B models through independently opened config/tokenizer snapshots: all248320 logits and all42 full state-domain vectors match direct CPU loading bit-for-bit over two tokens/nonzero state.
- [x] Verify actual independent sessions/unchanged sibling roots, distinct allocation source IDs, foreign-source restore refusal, ref retarget/private cache mutation isolation, denied staging/raw admission and exact retained raw-source charge until the last owner drops, then zero.
- [x] Run the frozen complete isolated gate:419 Rust executions passed,0 failed,54 optional/helper entries ignored (feature-build repetitions included). Strict minimal/default/cache/acquisition/diagnostic checks, CPU/cache HTTP exclusion, release/installer,18 Python namespace tests and comparator/resume self-tests pass; the ignored real-model gate was separately selected and passed for both models.
- [x] Verify actual offline Nix filtered-source integrity, changed-document local links, one tracked Cargo.lock and byte-identical user root README. No new dependency packages, model/SDK/provider fetch, device action or publication; optional CLI serde reuses the already locked package.
- [x] Integrate completed implementation run outcomes into this checklist/design reference and retire their temporary tracking. Keep source shards/new families, missing lengths, removal/reclamation and broader provider parity open.


### 20f449d — exact size recovery and cancellable cache verification

- [x] Recover omitted selected/required sizes from valid pinned HF LFS metadata or bounded HEAD on the exact HF commit/GitHub raw commit/release-asset-ID URL. Distinguish missing from zero, refuse conflicting LFS/tree sizes and avoid probes for unselected files.
- [x] Extend the existing synchronous HTTP owner with backward-compatible HEAD, preserving TLS/origin credentials/redirect/retry/deadline/identity/header limits and error redaction. HEAD reads no payload and unsupported custom transports never fall back to GET.
- [x] Require exact HTTP200 Content-Length within the file cap; refuse missing/malformed/overflow/conflicting/encoded/ranged metadata and contradictory HF commit headers, then use original full-peak A1 admission. Truly unknown-body streaming remains open.
- [x] Bind optional strong HEAD ETags to index metadata GET and initial/ranged transfer, preserving original digest/length/EOF and saved-prefix validation. Git OIDs/ETags never become trusted raw SHA256; changed validators cannot append older prefixes.
- [x] Bound per-source probes separately from listing pages and component GET/HEAD under one shared request budget. Preserve large object caps independently of JSON body limits; account transient validators in plan metadata and omit them/URLs from Debug.
- [x] Observe cancellation after failed HEAD before sanitized error mapping, preserving Limit/State categories. Keep offline before credentials/client/provider and body-free dry-run; CLI forwards HEAD through existing label/origin-scoped clients.
- [x] Add cached_or_partial_object/DiscoveredObject using original stable source/digest locks, reservation and publisher. A concurrent known-digest fill returns verified cached data before another private payload/request; legacy partial_object retains strict refusal.
- [x] Attribute those hits solely to cached progress with prefix/LFS rejection and immediate callback cancellation. Preserve original transaction completion, first-seen transfer limits and physically charged retained partials.
- [x] Thread transaction cancellation through bounded cache hashing, EOF/finalization, lookup/dedup/install/commit/partial verification and online final component open. Add cancellable snapshot/ref APIs; retain legacy no-flag wrappers and existing blocked-filesystem/inventory/parser boundaries.
- [x] Close control-copy cancellation: exact preallocation, bounded checked reads, separate EOF and SHA256 of actual owned control bytes before parsing. Raw A3 source copying/parsing and inference mathematics are unchanged.
- [x] Refresh CLI help for size recovery, body-free metadata, trust limits and accepted explicit A3 CPU loading; retain optional/default isolation and removal Unsupported. No new runtime, allocator, source issuer, inference format or CLI flags.
- [x] Integrate independent provider/HTTP/cache/CLI reviews. Fix actual development lint reports by boxing only the cold Partial guard (296B versus64B) and making the fixture Ordering import unconditional. Production/diagnostic rechecks pass without allowances/tolerance changes.
- [x] Verify12 new length tests,37 original provider/HTTP,10 component,11 progress and11 transaction tests;72 cache executions (33 library/4 maintenance/13 partial/7 source bridge/15 transactions);16 selected CLI executions and three final direct Cached-arm executor cases.
- [x] Requalify both approved original .8B/2B models:2 tokens each, all248320 logits/all42 state domains bit exact, independent sessions, exact raw-source lease lifetime and ref/private-mutation isolation. One selected ignored A3 test passed in21.61s; import enforced accepted source hashes and originals were untouched.
- [x] Run the complete frozen isolated gate:449 Rust executions passed,0 failed,54 optional/helper entries ignored across81 suites (feature repetitions included); strict minimal/default/cache/acquisition/diagnostic lint, CPU release/installer,18 namespace and comparator/resume checks passed.
- [x] Verify actual offline Nix filtered-source integrity, changed-document links, one unchanged tracked Cargo.lock and byte-identical user README. No dependency/model/SDK/provider acquisition, live cache deletion/migration, device action or publication.
- [x] Record dedicated removal requirements: escaped descriptor lifetime/old-reader policy compatibility, first-seen actual-blob reachability, partial file-before-owner close order, safe lock order and still-charged crash-safe retirement. Keep Unsupported until physical protocol qualification.
- [x] Integrate finished agent results/failures/frontiers into this checklist/design reference and retire temporary implementation checklists/scripts/logs after reconciliation.


### f44264c — protected snapshot removal

- [x] Introduce protected cache policy v2 only for fresh empty roots; serialize initialization, reject policy-less existing data, and add open_existing without root creation. Existing v1 policy/payload bytes and ordinary read/write semantics remain unchanged, without an extra charged activity inode; v1 removal refuses and no migration runs.
- [x] Pin actual managed regular-file open descriptions with shared Linux flock, revalidate name/inode, preserve try_clone/dup/inherited-child lifetime, and probe separate nonblocking exclusive descriptions. Managed lazy Artifact clones retain protection; arbitrary raw paths/mappings/unlock operations remain outside concurrent reclamation.
- [x] Hold protected activity through complete transactions and snapshot/ref binding; take maintenance exclusively before publication without waiting on attempt/object owners there. Exclude live first-seen/dedup actual blobs rather than infer reachability from planned hashes; close recovered partial File before its old owner and all transaction cleanup before activity.
- [x] Capture bounded current refs and conservative other global/snapshot/retired/abandoned-stage manifest reachability under original publication authority. Cap aggregate metadata64MiB and manifest edges131072, use sorted binary/prefix lookup, deduplicate actual probe inodes and retain shared blobs, staging, stable locks and reference backing.
- [x] Add exact-target retirement rename plus parent sync, identity-checked postorder unlink and retained-manifest/empty-marker restart. Pending same-digest retirement blocks publication; absent-target retries sync parents with zero removals and never sweep orphan blobs/manifests. Check the existing256KiB control allowance before fresh retirement.
- [x] Add typed reclaim_snapshot dry-run/apply outcomes: Busy/Referenced before mutation, Durable or Incomplete after work, named-entry counts and measured unique-inode/st_blocks charge. Lost authority/unavailable inventory reports unknown after-charge. Local collector handles close before unlink; fork-inherited descriptors/kernel close effects preclude a process-global physical-free guarantee. Legacy remove_snapshot remains Unsupported and directs callers to the typed method.
- [x] Add independent optional cache-only CLI cache remove DIGEST --dry-run|--apply --cache ABS with existing-root/disk-policy checks, exact selector, static argument refusal and honest observed-charge/busy/reference/incomplete output. Help states legacy refusal and raw-reader quiescence; no force, implicit ref deletion, transport, credentials or online signal setup. Correct stale pull-help text without provider changes.
- [x] Focused foundation82 tests passed twice, including36 library/7 descriptor-policy/4 existing maintenance/13 partial/7 source-bridge/15 transaction executions; strict production/diagnostic artifact lint passed. Collector14 tests passed, covering first-seen dedup, aliases/shared/abandoned stages, payload named manifest.json, control admission, cancellation, authority/regular/symlink replacement and blocked/canceled new opens.
- [x] Independent process/lazy/CLI25 focused executions passed across16 names; three CLI strict-lint cohorts passed. The34 observed removal callbacks were exercised with34 injected I/O failures and34 real exit86 children, exact survivor/ref preservation, restart/idempotency and independently reconstructed std::fs unique-inode/st_blocks charge. Same-OS exits do not establish actual power-loss behavior.
- [x] Record review refinements: preserve v1 quota compatibility without a new lock inode, avoid quadratic graph lookup, revalidate both stable authorities and candidate bindings after diagnostic Before hooks, admit retirement metadata peak, and distinguish absent no-op from proof of past removal. No duplicate cache/runtime/allocator or model-source authority was added.
- [x] Record development failures and fixes: E0373 in a scoped-opener test was fixed by moving senders/borrowing outer refs. First full gate failed101 on an immediate absent-target assertion; diagnostics reproduced callback19 ReclaimDirectorySync After Some(9) returning Busy because unrelated concurrent child spawning inherited activity descriptors until exec. A test-only first-acquired/last-dropped mutex serializes four unrelated fixture cases; all exact fault/intentional concurrency assertions remain. Ten default-parallel reruns passed; no production retry, locking, deadlines or numerical tolerances changed.
- [x] Final frozen full gate passed:518 Rust executions/0 failures/56 ignored across94 suite executions, including feature-build repetitions; strict lint, CPU release/installer,18 Python namespace tests and comparator/resume checks passed. Actual offline Nix source-integrity passed again after final corrected inputs were staged.
- [x] Enhanced original-checkpoint A3 gate passed both approved Qwen3.5 .8B/2B models: after alias replacement, private-copy corruption refusal and actual typed removal of the snapshot/weight blob, the owned model retains its raw reservation and matches all248320 logits/all42 full state domains across2 tokens, independent sessions/source IDs and final raw-lease release. Isolated release selection passed1 test in19.81s; no new format/family/context/GPU claim.
- [x] Integrate all agent changes, failures, results and boundaries into this main checklist/design; retire their completed run checklists/logs after review. Root README and Cargo.lock unchanged; no model/SDK/provider download, live cache deletion/migration, device action or publication.

### ea700f6 — capped auxiliary streaming

- [x] Carry explicit Exact/AtMost through existing cache/acquisition files and component plans. Keep optional exact totals separate from reservation maxima; published manifests and receipts contain observed exact bytes/hash. Existing exact constructors retain their meaning; acquisition exact-total getters now return Option, with explicit maximum getters for callers.
- [x] Add positive PullLimits.unknown_length_cap and CLI --unknown-size-cap-bytes, bounded by max-file-bytes and disabled by default. Only pinned Auxiliary selections without weights/index qualify; config/tokenizer/license/index/weights and every model-closure member remain exact. Explicit low-level AtMost constructors supply trusted caller authorization independently of provider policy.
- [x] Permit only validated HTTP200 HEAD missing a length or405/501 fallback after ordinary pinned metadata recovery. Keep all transport/TLS/cancel/pin/framing/budget errors fail-closed, bind strong successful HEAD ETags, and discard unsupported-HEAD error-representation lengths/validators. HEAD and dry-run read no payload.
- [x] Feed one-shot grouped maximum payload admission into the existing persistent reservation/attempt. Same trusted digest with unequal caps reserves the maximum; distinct first-seen files reserve independently. Preserve digest/activity locks, original sole installer/publisher, protected removal and legacy policy boundaries.
- [x] Verify known-digest hits on the original opened file using actual length/hash. Return actual cached counters and manifest bytes; a too-small cap or corrupt hit refuses before HTTP/private payload staging, while the admitted attempt/reservation already exists.
- [x] Reuse the existing partial guard and incremental raw SHA256 for ephemeral capped streaming, explicit terminal-EOF finish and cap+1 detection. Refuse durable checkpoint/validator/recovery for caps; read errors/cancellation never finalize or Range-retry an unknown body. Preserve exact resumable paths and accept empty/short payloads only after true source EOF.
- [x] Require200/identity/no-range and bounded Content-Length or sole chunked framing for capped GET. Enforce declared lengths even for custom transports; reject close-delimited/ambiguous framing, missing zero chunks and truncated data. Document pinned ureq zero-chunk-line-at-FIN payload completion without independent full trailer-section validation; trailer metadata grants no authority.
- [x] Keep accepted/cached/resumed/downloaded progress actual and monotonic, unknown exact totals optional and full reservation maxima explicit. Emit file_complete only after verified cache/body finish, even with a repeated byte count; publication and complete component receipts remain later boundaries.
- [x] Reject cap plus offline and invalid limits before cache/plan-file/credentials/client I/O. Preserve body-free dry-run and unchanged component schema/source identities; aggregate worst-case file/payload/metadata/request admission remains bounded. No new dependencies, source issuer, allocator, runtime or model-readiness claim.
- [x] Integrate independent ownership/reviews: cap+1-clamp custom body reads as well as production HTTP, distinguish payload staging from attempt creation, and document strict selection-wide scope/API/decoder limits. Cache scoped107 executions pass; transfer40 library/12 capped/12 lengths/10 components/11 progress/11 transactions pass, production capped12 and both strict all-target artifact lints pass. Final EOF-loop adjustment rechecked capped12 plus exact live-range1.
- [x] Record development corrections: wrong component-resolver fixture argument order failed compilation; .bin fixture names correctly classified as weights caused two test refusals and were changed to auxiliary .txt. The old progress vector failed because final EOF completion adds one same-count event; updated it to assert completion only on that final event. A while_immutable_condition lint was fixed by an explicit EOF transition. No production tolerances, rollback assertions or exact-resume checks were relaxed.
- [x] Independent checks pass:34 actual Rust executions/2 owned helpers ignored and both strict CLI all-target lint configurations. Actual HTTP/HEAD framing and capped completion/refusal fixtures pass;40 observed publication callbacks exercise80 I/O failures and80 real child exits across old/absent refs, followed by complete restart and reservation-release checks. Same-OS exits do not prove physical power-loss behavior.
- [x] Complete frozen isolated gate passed:553 Rust executions/0 failures/57 optional/helper entries ignored across103 suite executions (feature repetitions included); strict feature/lint checks, CPU release/installer,18 namespace Python tests and comparator/resume self-tests pass. Source/tests and active check.sh stayed frozen; no full-gate failure.
- [x] Actual offline Nix filtered-source integrity passed with final new Rust tests staged;23 local document links, one tracked unchanged Cargo.lock and byte-identical user root README pass.
- [x] Requalify both approved original Qwen3.5 .8B/2B models after cache changes using the existing isolated release cache_prepare gate:1 selected test passed in20.02s after43.49s compilation. Two tokens/model match all248320 logits/all42 full state domains bit-for-bit; independent sessions/source allocation, original raw-lease lifetime, private mutation and actual cache-removal isolation remain exact. No numerical/context/model/GPU eligibility extension.
- [x] Integrate all three completed agents' outcomes/failures/frontiers in this serialized checklist/design update and retire their temporary run tracking. Keep unknown model/required lengths, capped resume/full trailer validation and remaining parity/reclamation work explicit; one signed implementation commit carries actual provenance forward. Root README/lock unchanged; no new dependencies, model/SDK/provider fetch, live cache deletion/migration, device action or publication.

### 03b6948 — pinned multi-asset release components

- [x] Extract private ReleasePin backed by positive NonZeroU64, one canonical release-N-asset-M parser, checked GitHub source validation and a shared release-only resolver returning plan/pin. Ordinary pulls delegate with no prior pin; no synthetic tags, public pin argument, release-detail request or duplicate asset/hash/HEAD implementation.
- [x] Permit repeated release-alias component selections and reuse the first release ID for each literal provider/repository/requested-tag tuple. Keep file/release maps separate, later labels' own credential adapters and every original request/metadata/file/payload/role/sidecar bound. Preserve distinct original per-asset source manifests/receipts and the sole existing A1 executor/publisher.
- [x] Reuse the canonical parser for pure offline bindings: require positive canonical release/asset IDs, compare release IDs for repeated release tuples while allowing different asset IDs, and retain exact commit coherence in a separate file namespace. Mixed releases, alternate spellings, wrong source/locator kinds and missing selections refuse without clients.
- [x] Preserve exact length recovery, explicit Auxiliary-only capped streaming, trusted-SHA256 versus FirstSeen provenance, model closure, cancellation and admitted metadata work. A stable release ID does not freeze an asset set; selected IDs/digests remain the actual authority, and no compatible-model, cross-asset atomic snapshot or new execution capability follows.
- [x] Run combined scoped tests:147 executions/0 failures/3 owned helpers ignored (82 library,13 component,12 capped,6 independent capped,12 length,11 progress/resume,11 transactions). Six new provider cases and three added existing-component cases pass; strict artifact all-target lint passes for both production and diagnostic acquisition/cache.
- [x] Independently qualify seven release-component tests in both diagnostic and production builds:14 executions/0 failures/ignored. Cover one alias lookup despite movement, per-label release-ID metadata and asset-ID payloads, separate manifests/provenance, unrelated tags/repos and file/release namespaces, canonical offline coherence, duplicate/missing/state/digest refusal, denied/canceled admission, aggregate limits and100-row pagination with shared seven-request success/six-request refusal.
- [x] Prove later-member bad digest preserves an existing reference and earlier complete immutable source data, with original cache/host reservations released. Existing publisher crash/transport/credential matrices remain reused; no second publisher or redundant new crash protocol is introduced.
- [x] Integrate three independent ownership/review streams with no production blocker. Correct initial provisional Aux-only test fixtures to the existing required Weights/Config/Tokenizer suppliers before compiling; correct provisional test count from eight to seven. All scoped commands pass without source fixes, tolerance changes, validation relaxation, CLI/schema/dependency changes or new runtime.
- [x] Actual offline Nix source-integrity passes on staged final new Rust inputs.23 current local documentation links, byte-identical user root README/Cargo.lock and one tracked lock pass; production/source/math/active gate scripts remain frozen.
- [x] Complete frozen isolated repository gate passed:569 Rust executions/0 failures/57 optional/helper entries ignored across106 suites (feature repetitions included). Strict feature/lint matrix, CPU release/installer,18 namespace Python tests and comparator/resume self-tests pass. No scoped/full-gate failure or source/script edit after freeze.
- [x] Existing isolated original-model A3 release selection passed1 test/both approved original Qwen3.5 .8B/2B models in20.03s (cached compilation0.14s). Two tokens/model match all248320 logits/all42 state domains bit-for-bit, with independent sessions/source IDs, exact raw-lease lifetime and private mutation/removal isolation. This metadata-only phase changes no model/operator/context/GPU qualification.
- [x] Reconcile all completed agents' implementation/results/frontiers in this serialized main-plan/design update and retire finished run tracking. Preserve mutable asset-set/offline membership limitations and remaining provider/reclamation work. Sign the verified explicit-path phase with actual provenance; root README/lock unchanged, no dependencies, model/SDK/provider payload fetch, live cache inspection/import/deletion/migration, device action or publication.

### e44f4ee — explicit read-only HF cache import

- [x] Add explicit Linux cache-only `cache::import_hf` request/prepare/check_destination/apply/receipt API and CLI `cache import-hf --dry-run|--apply`. Require caller-selected repository root, exact40-lowercase-hex commit, paths/roles and optional authorized SHA256; no discovery, refs, provider client, network, credentials, new closure resolver or implicit run import.
- [x] Keep data provenance honest: persist `local-hf-cache`, caller-declared repository/commit, actual lengths/hashes and TrustedSha256 only for verified caller digests; otherwise FirstSeen. These local manifests cannot impersonate authenticated online-HF selections or grant loader eligibility.
- [x] Admit bounded source controls before metadata/FD opens in the caller's existing TransferAdmission, retain the lease/descriptors/cancel flag, and derive exact sizes from opened files. Preparation uses16MiB plus eight times the configured manifest cap; constructor count/path/schema limits precede large clones. Larger admitted caller manifest limits remain effective.
- [x] Read only regular snapshot files or canonical one-hop HF leaf links into same-root flat40/64-lowercase-hex blobs. No-follow intermediate components, retained positioned readers, per-read/precommit device/inode/length/mtime/ctime checks, independent nonempty EOF and streaming LFS-pointer refusal preserve original selected bytes.
- [x] Prove prewrite source/target disjointness through retained directory capabilities, bidirectional ancestry and complete source-metadata inode intersections, including unselected hardlinks. Bounded metadata-only DFS caps131072 nodes,8MiB cumulative path work and depth64; it reads no refs values, cached tree JSON or unselected bodies. Dry-run writes neither tree; apply repeats checks.
- [x] Initialize the exact preflighted destination RootDir through a narrow extraction of the original CacheStore initializer, without reopening its path. Preserve the existing attempt, full-peak admission, private copy, hash/EOF validation, sole installer, snapshot publisher and typed DurabilityUnknown. No external source locking/chmod/hardlink/rename/delete or mutable-ref publication.
- [x] Reuse the original SHA pass on ordinary copies; even a known-digest destination hit independently reads/hashes the original external source before commit. Report actual source/hash/provenance and zero copied bytes for that cache-hit path. Keep diagnostic hook injection within the original CacheHook boundary.
- [x] Qualify only matching `models--owner--repo` roots, two-component bounded ASCII repository names, available Linux statx mount identity and existing absolute0700 destinations on the same mount. Missing destinations, crossed/bind mounts, renamed/unnamespaced layouts and unavailable mount IDs refuse. This does not promise atomic external-tree freezing, global RAM/UMA/free-space reservation or realtime filesystem cancellation; original initializer metadata/lock-deadline behavior remains documented.
- [x] Integrate disjoint source/importer/CLI implementation and independent reviews. Scoped cache/source/import/prior regressions88 passed/0 failed/2 helpers ignored; independent production/diagnostic, CLI maintenance/import, facade tiny and disabled-feature cohort21 passed/2 model cases deliberately ignored. Strict production/diagnostic artifacts and cache-only CLI/facade all-target lint pass without allowances.
- [x] Record actual scoped failures and corrections: first library run42 passed/2 failed because intended40hex fixture blob ID had42 digits; corrected constant and chain fixture only. First strict diagnostic lint found two test-only field-reassign-with-default cases; explicit initializers fix them. First independent diagnostic run5 passed/1 fixture failed because Unix socket pathname exceeded SUN_LEN; bind at a short private path then rename the same socket inode to exercise selected-special-file refusal. Corrected suites pass; production acceptance bounds were not relaxed.
- [x] Revalidate original approved checkpoints/config/tokenizer identities and resource headroom. Add the ignored HF-import route to the existing source-model qualification using separate private copies of approved files, import with exact known hashes, then remove the synthetic HF source before preparation. Preserve the original numerical/session/resource/mutation/reclamation assertion tail and existing source issuer.
- [x] Run both original-model routes sequentially under isolation:2 selected tests passed/0 failed in45.15s, each covering original Qwen3.5 .8B and2B over two tokens/nonzero state. All248320 logits and all42 full state-domain vectors match direct loading bit-for-bit, with independent sessions/source identities, foreign-source restore refusal, exact rawlease lifetime and cache mutation/reclamation isolation.
- [x] Run the complete frozen isolated check.sh gate:625 Rust executions passed/0 failed/58 ignored across116 suite executions (feature repetitions included), strict lint/build/release/installer,18 namespace Python tests and comparator/resume self-tests pass. Actual offline Nix source-integrity passes; local document links, one unchanged Cargo.lock and byte-identical user root README pass.
- [x] Integrate actual agent results/failures/remaining qualification into this main checklist/design reference, then retire completed temporary run checklists/logs. No new dependency/model/SDK/provider fetch, live-cache import/mutation, mount/device privilege change, GPU launch, push or history rewrite; no new numerical/context/family/GPU claim.

### fce1353 — qualified cross-filesystem HF import

- [x] Qualify a bounded cross-filesystem HF import extension using the existing API. Preserve original same-mount behavior; differing mounts must also have distinct device IDs and actual retained-FD fstatfs types Btrfs↔tmpfs or tmpfs↔tmpfs. Across different mounts, same-device aliases, Btrfs↔Btrfs, overlay/FUSE/network and every other unqualified pair remain Unsupported.
- [x] Retain private root mount/device/inode identity and lazily inspect filesystem kind only for differing mounts. Recheck the original source root and selected descriptors; each cross-filesystem tree/selected file must remain on its own root device AND mount. Normalize filesystem magic through its low32bits for signed32-bit FsWord portability; no new dependency or public option.
- [x] Preserve bounded complete source-metadata inode inventory, both ancestry directions and target inode intersection. Explicitly check the destination ROOT before target children, then revalidate both retained root capabilities after topology inspection. No source path reopen or second destination initializer; arbitrary concurrent same-UID mutation is still not an atomic-tree guarantee.
- [x] Reuse the caller's original TransferAdmission, retained exact source readers, full hash/EOF/LFS checks, full-peak private staging, sole A1 installer/publisher and original cancellation/typed completion semantics. No external source lock/chmod/hardlink/rename/delete, cache owner/source issuer/math/runtime or numerical-limit change.
- [x] Add three policy-only unit cases for the complete allowed/refused kind matrix, missing kind, same-device mount-alias refusal and preserved same-mount device/kind behavior. Extend the actual opened-root replacement test to prove disjointness through the retained source capability. Synthetic identity tests are not physical qualification.
- [x] Add one ordinary public same-mount regression and five explicitly ignored physical cases (one diagnostic-only), selected separately with no runtime skips. Tiny private RAII-owned directories under existing repo target,/tmp,/dev/shm assert actual filesystem kind/dev/mount identities; no mount or privilege operation is used.
- [x] Physically copy in both Btrfs↔tmpfs and distinct tmpfs↔tmpfs directions: zero/direct/nested/canonical-HF-link files, exact manifest length/SHA/provenance, unchanged source names/content/mode/link-count/mtime/ctime (atime excluded), destination device and owned-byte survival after source deletion. FirstSeen and verified caller-hash paths preserve their existing meanings.
- [x] Physically verify trusted destination hits still hash actual external bytes and report copied0; corrupt same-length source bodies refuse. Wrong digest, denied admission, pre-copy cancellation, symlink/ancestor overlap and real reached post-copy ObjectInstall cancellation preserve exact old refs/snapshots/data and release live reservations. These alias fixtures do not claim hidden bind-mount qualification.
- [x] Complete focused isolated checks:91 source/library/public import/prior-cache executions passed/0 failed/2 helpers ignored; physical selected file6 diagnostic+5 production executions passed/0 failed/0 ignored (including one same-mount case in each configuration). Strict production and diagnostic artifact all-target lint pass. No acceptance command failed, no production/test fix after first compilation and no allowance/tolerance widening.
- [x] Complete frozen isolated repository gate:633 Rust executions passed/0 failed/68 ignored across119 suites (feature repetitions included), plus strict lint/build/release/installer,18 namespace Python tests and comparator/resume self-tests. Explicit physical selections above execute the relevant cases ignored by the portable generic gate.
- [x] Requalify both approved original Qwen3.5 .8B/2B checkpoints through original-cache and HF-import preparation:2 selected tests passed in42.20s after45.24s release compilation, two tokens/model/nonzero state. All248320 logits and all42 full state domains remain bit-exact, with independent sessions/source IDs, foreign-source restore refusal, exact rawlease lifetime and private cache mutation/reclamation isolation. These are existing loader regressions, not a new context/family/GPU qualification.
- [x] Actual offline Nix filtered-source integrity and23 local documentation links pass; root README and single Cargo.lock remain byte-identical. Integrate independent primary-source/kernel identity review and document tmpfs volatility: original successful fsync/publication acknowledgments do not promise persistence after unmount/reboot/power loss.
- [x] Reconcile all agents' results/limits and the next bounded direct-HTTPS proposal atomically into this checklist/design; retire completed temporary run files/logs. No new dependency/model/SDK/provider payload fetch, live-cache import/mutation, mount/device privilege change, GPU launch, push or history rewrite. Hidden bind/subvolume/overlay fixtures and additional filesystem pairs remain unfinished below.

## Tests and verification

- [x] ACQ-A01/A02/A04/A05/A06/A07: original A1 path/schema/admission/publication/crash/concurrency/lease regressions pass. Existing38-callback matrix retains76 injected I/O failures and76 real child exits.
- [x] ACQ-A03: discovered first-seen/trusted streaming, exact length/digest, LFS, closure, cancellation and ref rollback pass. Actual40 callback ordinals across absent/existing refs exercise80 I/O failures and80 real child exits, post-rename DurabilityUnknown, quota rebuild and subsequent publication. Final acquisition transactions:11 passed plus1 child helper ignored.
- [x] ACQ-A08: isolated offline hit/miss, source/selection/exact-pin/locator-kind refusal precede credentials/provider/client. Public metadata admission and pre-cancel tests use panic-on-request transport.
- [x] ACQ-A09: optional artifacts/facade/CLI strict lint with production-only and diagnostic acquisition; default CPU+acquisition compile and minimal/default CPU/core/inspection checks pass. Normal+build CPU closure excludes HTTP/TLS/GPU.
- [x] ACQ-A10 bounded slice: actual HTTP12/HF7/resume2 focused fixtures passed, including real verified TLS, redaction at requested runtime TRACE, framing, redirects, identity/range validators, retry/backoff/deadline/cancellation. The continuation extends durable resume/progress below; automatic reclamation remains open.
- [x] GitHub15 focused provider tests passed: exact commit/tree or release/asset pin/digest, truncation/missing/unsafe/LFS, complete model closure, uppercase identity and mixed-family refusal.
- [x] Focused provider/parser/CLI checks:23 executions passed across optional/default-disabled commands; full gate repeats relevant suites. Offline operations verify actual cached bytes; mutable branch/tag freshness is not asserted.
- [x] Full CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh passed on frozen sources with discovery crash diagnostics enabled; documentation integrity, original root README and actual Nix filtered-source gate passed.
- [x] ACQ-A11: actual A3 owned-copy original .8B/2B bridge passed. Isolated jobs2/Rayon4 `cargo test --release --locked --offline -p zerogravity --features cache --test cache_prepare approved_cached_models_match_direct_source_exactly -- --ignored --exact --nocapture`:1 selected test covers both approved models,2 tokens each, all248320 logits and all42 full domains exact; execution18.49s. No8K/new-family/GPU requalification claim.
- [x] Continuation cache:13 partial tests passed;16 I/O faults and16 real exits across8 initial/replaced checkpoint callbacks, uncommitted tails, binding/tampering, live process locks, no-progress cleanup, proper-prefix bounds and retained-old plus full-new peak admission. Original A1 transactions15 passed/one child helper ignored.
- [x] Continuation transport:11 combined tests passed/one child helper ignored;9 production tests passed. Actual loopback range/refusal matrix, live retry, weak/missing validators, full resumed digest, source/object mismatch, periodic-checkpoint child exit and exact-total/trailing-byte plus post-read cancellation regressions passed. Existing discovery11 tests and80 I/O/80 real-exit matrix remain green.
- [x] Continuation CLI:4 unit tests include actual SIGINT/SIGTERM children, callback unregister and real cached+streamed progress;6 cache/filter integration tests and8 HF tests pass, plus default-disabled refusal.19 distinct focused names/23 executions including the repeated4 after quota repair; full gate repeats final suites.
- [x] Continuation full gate:353 Rust executions passed,0 failed,34 optional/child-helper entries ignored (including feature-build repeats). Strict production/diagnostic/combined checks, default CPU signal/HTTP/TLS/GPU exclusion, release/installer,18 Python namespace tests and comparator/resume self-tests pass. Actual offline Nix source-integrity,23 local links, one tracked Cargo.lock and unchanged root README pass.
- [x] Components/A3 source gates:7 synthetic bridge tests,30 artifact library tests, original13 partial/15 transaction regressions,37 provider/transport tests,11 progress and11 acquisition transactions pass. Final component suite10 passed including the tighter public offline per-file bound; strict production/diagnostic artifact lint passed without allowances.
- [x] Components/A3 facade/CLI gates:21 selected executions passed, including8 frontend cases, tiny preparation refusal,10 corrected component tests, minimal cache-only refusal and1 explicitly selected real-model test exercising both checkpoints. Strict all-target default CPU+cache/acquisition and minimal cache lint/check passed; same-descriptor/cursor/path/mutation/lease guarantees received independent review.
- [x] Components/A3 full gate: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passed once on frozen source:419 Rust executions/0 failures/54 ignored across75 suite executions. Release/installer/lint/18 namespace and comparator/resume checks passed. Actual offline Nix source-integrity, all changed-document local links, single tracked lock and original root README passed.
- [x] Exact-size recovery:12 scripted/actual-loopback tests passed for pinned HF LFS/HEAD/GitHub raw/release-ID recovery; missing/zero/conflicting/encoded/ranged/oversize/status refusals; full index/shard closure; changed validators/saved-prefix refusal; cancellation/redaction; body-free redirects/auth/deadlines and shared request/large-object versus JSON budgets.
- [x] Cache verification:72 focused executions and strict production/diagnostic lint passed. Deterministic chunk/data/EOF cancellation, empty/truncated/oversize/wrong-digest control copies, concurrent fill/corruption/canceled waiter/ref/accounting and original source/fault/real-exit regressions passed.
- [x] Executor/CLI: final gate ran all three new direct Cached-arm tests (zero HTTP/payload, exact cached counters, immediate cancellation/ref preservation and LFS-before-progress refusal).16 scoped CLI executions/8 distinct names and final default-disabled/production/diagnostic checks passed.
- [x] Actual A3 requalification: isolated jobs2/Rayon4 cargo test --release --locked --offline -p zerogravity --features cache --test cache_prepare approved_cached_models_match_direct_source_exactly -- --ignored --exact --nocapture passed both approved original models in21.61s. Numerical/state/ownership criteria unchanged; no new model/context/GPU claim.
- [x] Recovery/cache full gate: CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh passed once on frozen source:449 Rust executions/0 failures/54 ignored across81 suites. Release/installer/lint/18 namespace/comparator/resume and actual offline Nix source-integrity passed; changed-document links, original README and one unchanged tracked lock pass.
- [x] Protected removal gates: actual descriptor/dup/inherited child/lazy Artifact lifetime, live first-seen/dedup exclusion, shared graph, legacy/fresh policy, new-open/cancel interleavings and post-hook authority/inode substitution pass;34 I/O/34 real-exit boundaries independently rebuild exact charge and preserve survivors.
- [x] Protected removal final gate: CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh passed on final frozen source:518 passed/0 failed/56 ignored/94 suites. Final actual offline Nix source-integrity and document links passed; no root README/lock changes.
- [x] Actual source after removal: isolated jobs2/Rayon4 cargo test --release --locked --offline -p zerogravity --features cache --test cache_prepare approved_cached_models_match_direct_source_exactly -- --ignored --exact --nocapture passed1 selected test/both original models in19.81s, with reclamation_isolated=1 and all existing exact numerical/state/ownership criteria.
- [x] Capped-streaming focused cache/transfer/CLI suites and independent framing/publication matrix passed as itemized above; no capped resume or full trailer-validation claim.
- [x] Capped-streaming complete gate: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passed on frozen sources:553 Rust executions/0 failures/57 ignored/103 suites, plus strict lint/release/installer/18 namespace/comparator/resume checks. Actual offline Nix source-integrity and document/root README/lock integrity pass.
- [x] Capped-streaming original-model A3 acceptance: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- cargo test --release --locked --offline -p zerogravity --features cache --test cache_prepare approved_cached_models_match_direct_source_exactly -- --ignored --exact --nocapture` passed1 selected test/both approved original models in20.02s; all-logit/all42-domain/session/resource/private-removal criteria unchanged.
- [x] Repeated-release qualification: isolated combined147 executions/3 helpers ignored and independent7 diagnostic+7 production pass, with strict production/diagnostic artifact all-target lint. New locator/namespace/budget/failure fixtures preserve unchanged executor/cache contracts.
- [x] Repeated-release full gate: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passed569 Rust executions/0 failures/57 ignored/106 suites plus full lint/release/installer/18 namespace/comparator/resume checks. Actual offline Nix filtered-source integrity,23 local links and original README/lock pass.
- [x] Repeated-release original-source regression: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- cargo test --release --locked --offline -p zerogravity --features cache --test cache_prepare approved_cached_models_match_direct_source_exactly -- --ignored --exact --nocapture` passed1 selected test/both original models in20.03s; original numerical/state/resource/source/private-removal boundaries unchanged.
- [x] HF-import focused gates: admitted source/public importer/prior cache suites88 passed/2 helpers ignored; independent diagnostic6 and production5, CLI import3/maintenance4, facade tiny1 and disabled-feature2 passed. Strict cache-only production/diagnostic artifacts and CLI/facade all-target lint pass; malformed paths/links/special files, mutation/EOF/digests/admission/cancel/overlap/hardlink and original publication-fault rollback remain covered.
- [x] HF-import full gate: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passed625 Rust executions/0 failed/58 ignored/116 suites, plus strict lint/release/installer/18 namespace/comparator/resume gates. `nix build --offline --no-link .#checks.x86_64-linux.source-integrity` passed on the same frozen source; current local links and unchanged root README/single Cargo.lock pass.
- [x] HF-import actual source gates: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- cargo test --release --locked --offline -p zerogravity --features cache --test cache_prepare -- --ignored --test-threads=1 --nocapture` passed both selected tests in45.15s after release compilation45.18s. Both original and imported-HF routes cover approved .8B/2B checkpoints with original all-logit/all42-domain/session/rawlease/private-mutation/removal acceptance, unchanged numerical limits and no source payload remaining in the synthetic HF tree during inference.
- [x] Cross-filesystem focused gates: isolated cache,diagnostic-cache --lib --test cache_import_hf --test cache_transactions --test cache_source_bridge --test cache_partials passed91/0fail/2helpersignored, including11 source tests; both strict artifact cache production/diagnostic all-target lints pass. Current API/source/old publisher and same-mount regressions remain intact.
- [x] Actual filesystem gate: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- cargo test --locked --offline -p zerogravity-artifacts --no-default-features --features cache,diagnostic-cache --test cache_import_hf_crossfs -- --include-ignored --test-threads=1 --nocapture` passed6/0fail/0ignored; same command with only cache passed5. Runtime checks observed Btrfs dev46/mount53 ↔ tmpfs dev39/mount43 and /tmp tmpfs dev53/mount196 ↔ /dev/shm tmpfs dev39/mount43, both directions. These are observed identities, not hardcoded acceptance IDs; source preservation, owned copies, actual hash, cache-hit rehash, rollback and reached post-copy cancellation pass.
- [x] Cross-filesystem full gate: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passed633 Rust executions/0 failed/68 ignored/119 suites, plus strict lint/release/installer/18 namespace/comparator/resume checks. `nix build --offline --no-link .#checks.x86_64-linux.source-integrity` and23 local links pass; root README/one unchanged Cargo.lock pass.
- [x] Cross-filesystem phase original-loader regression: `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- cargo test --release --locked --offline -p zerogravity --features cache --test cache_prepare -- --ignored --test-threads=1 --nocapture` passed2 selected tests covering both approved .8B/2B checkpoints through both original and imported-HF routes in42.20s. Original all-logit/all42-domain/session/source/rawlease/private-mutation/removal criteria remain unchanged; this does not claim whole-model execution from a newly qualified filesystem pair.

````

## Source: .agents/plans/model-execution/README.md

Original UTF-8 bytes: 24867; SHA256 `be399b51b723701045bc24395074e3fec870c81027cb48e256dc1075c5aabb05`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: Project owner
Co-authors: GPT-6

# Zerogravity: source-real model execution (CPU-first)

## Active order

1. **Qwen3.5 2B**, with 0.8B as the exact trained regression canary:
   preserve existing resident GGUF CPU/GPU StateRoot and complete logits,
   then optimize measured end-to-end prefill/decode on qualified hardware.
2. **Qwen3.8 27B:** original pinned source, immutable large shards, actual
   model tensor/head/axis binding and tokenizer, complete trained CPU
   logits/StateRoot parity, then GPU specialization and SSD streaming.
3. **Gemma4 E2B → 26B-A4B → 31B:** original learned PLE/MoE/dense
   geometry; source-accurate CPU/quant math first, then GPU physical gates.
4. **Qwen3.8 Flash-Next:** original GDN/QSA/MoE/Engram/MTP as actually
   source-qualified, real Q2/~3.5-bpw/Q4 quants; speculative suffix/n-gram
   decoding is a distinct algorithm from learned n-gram embeddings.

YAML single-process/multi-model pipeline scaffolds are already complete.
Live routing, audio/image/video, context providers, trained decisions
and non-critical cloud provider parity are deferred to their owners.

## LS-B01–B05: verified owned sharded-source transaction

Implementation: `crates/zerogravity-artifacts/src/verified_shards.rs`
and the original `SourceWeights` allocation/issuer in
`source_weights.rs`.

**Input:** one original index JSON plus the independently expected
whole-index SHA256, an exact set of already-opened Safetensors Artifacts
with caller-declared basename, whole-file length and independently
expected SHA256 per shard; `SourceOpenLimits` and the existing
caller-registered `TransferAdmission` for the actual host RAM domain.

**Prepare:** same bounded index/header parsers, exact indexed name→shard
bijection, dtype/shape/offset/size and payload-byte total agreement,
checked whole-file and metadata budgets. Reserve the entire original
file bytes (headers included) plus conservative metadata under ONE
physical ResourceArbiter lease *before source content allocation*.

**Materialize:** copy from the exact originally opened descriptors,
verify their whole-file SHA256 and exact lengths, reparse the copied
immutable bytes, re-check header geometry used during preflight,
and independently re-check owned payload sum against the index.
The already-existing `SourceIdentity::for_new_allocation` is called
only after every check. One `SourceWeights` owns all shard Vecs
plus source lease; its `TensorSource` borrows are tied to its
actual source lifetime. Caller-supplied foreign/altered TensorInfo
cannot select a different shard via offset/name/shape substitution.
Partial failures release admission without publishing any source,
cache entry or model StateRoot.

**Compatibility:** the original GGUF `SourceWeights::open` keeps
its **4GiB default ceiling**, CPU/GPU model code and source identity
semantics. Multiple model instances/Arc owners must share the
existing process-wide ResourceArbiter, not invent independent memory
domains. This source format constructor does **not** implement
original BF16/F16 mathematical schedules or convert Hugging Face
tensor axes to GGUF axes.

**Tested only with tiny owned shards and sparse metadata:** F32/F16/
BF16 reference vectors, exact headers and digests, wrong/incomplete
index, missing/duplicate/extra/misassigned shards, concurrent
admitted allocations, mutation/rename of original descriptors,
same-size different geometry and foreign tensor metadata refusal,
and sparse >5GiB header preflight refusal without reading the data.
A deliberate larger source limit still needs real physical RAM
admission. Neither synthetic 5GiB refusal nor two tiny shard loads
qualifies *real* 27B trained inference or SSD performance.

## Remaining execution gates

- Actual authenticated model/index/tokenizer sources on an admitted
  capable host, original Qwen3.8 head/permutation/numerical binding,
  GQA 16→48, all vocabulary and StateRoot values for trained weights.
- Source-real large-shard allocation/lifetime and actual >4GiB host
  memory admission before claiming large-resident execution. This first
  owned copy is **not** SSD streaming.
- An owned immutable SSD extent and streaming lease abstraction with
  bounded I/O, readiness, errors, cancellation, and **stable** source
  buffer borrows. Evictable tensor caches cannot implement the original
  borrowed `TensorSource` interface by returning movable slices.
- Measured Qwen3.5 2B CPU/GPU speed, accuracy, prefill and decode,
  source/StateRoot, per-session rollback and physical readiness,
  with no unauthorized GPU on a held device.
- Gemma/Flash original trained correctness and quant/SSD/GPU gates.
  No new model SDK/download or runtime process/serve path is implied.

See the [model execution TODO](TODO.md),
[immutable shard specification](../IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md),
[SSD and memory](../memory-and-streaming/TODO.md),
[CPU/GPU optimization](../cpu-gpu-optimization/TODO.md), and
[model acquisition](../model-acquisition/TODO.md).

## ME-P0 continuation — original Qwen3.8 text binding and CPU value path

The original Qwen3.8 **ordinary text** tensor contract is now bound
directly to the immutable `SourceWeights` from the accepted original
index/shard transaction. This is a source/value preparation stage, NOT
a complete original-model token decoder.

- `crates/zerogravity-artifacts/src/model/qwen_owned.rs`: borrow the
  original source with an independently expected config JSON SHA256
  and exact verified original indexed-source SHA256. Reuse the existing
  `QwenTextConfig` source-order tensor definitions. Required name,
  output-first shape and F32/F16/BF16 encoding must all agree with
  the immutable verified bytes. Missing/extra ordinary text tensors,
  untied output head mismatches and wrong source hashes refuse.
  Vision and MTP tensors remain separately unqualified.
- `crates/zerogravity-cpu/src/original_hf.rs`: one
  **[output,input]** source matrix adapter to the existing Rust
  source-typed CPU `MatVecRegion` and optional completed accelerator
  provider. Do not transpose/copy learned matrices or create a
  second mathematical backend. Bound the original rows, finite
  inputs and trusted physical completed outputs before dispatch.
- `crates/zerogravity/src/original_qwen_cpu.rs`:
  `QwenOriginalCpu::bind` exposes source-bound `embed`, `vector` and
  `project` cold value primitives using that one owner and CPU math.
  No new model SourceIdentity, session, token loop or resource issuer.

Small synthetic **complete original hybrid** source tests and original
F32/F16/BF16 projection/embedding integration tests prove the path;
wrong tensor shapes, missing output heads, added learned-text roles,
config/index digest disagreement and incompatible providers refuse.
These tests DO NOT qualify trained 64-layer Qwen3.8-27B inference.

**Remaining:** real learned weights/official tokenizer and reference
framework, original Q/K→V grouped replications, complete GDN/attention/
MLP forward schedule and BF16 rounding, full 248320-vocabulary logits,
session/StateRoot, real >4GiB resident loading and SSD streaming. The
existing 0.8B/2B trained GGUF oracles must remain exact.

## ME-P0 analytic original Qwen3.8 one-layer proposals (CPU)

**Scope:** source-bound, *tiny analytical FP32* proposals of GDN,
full attention and SwiGLU FFN, using the original verified HF tensor
names and the existing Qwen3.5 mathematical operators. This is NOT
a second model executor, trained Qwen3.8 output, production StateRoot,
model-loading CLI, or an HF BF16-equivalent source reference yet.

Location: `crates/zerogravity/src/original_qwen_math.rs`, opt-in
`cpu+cache+model-qwen35` only; default GGUF source/session path
is unchanged. `OriginalQwenLayerMath` borrows the previously
verified `QwenOriginalCpu` source and exercises:

- **Zero-centered RMSNorm:** original ordinary residual and Q/K
  RMSNorm weights are `1 + checkpoint_weight`. The GDN output's
  separately learned gated norm uses its raw weight. The same
  qualified `source_rms_normalize` and
  `source_grouped_rms_normalize` CPU math is reused.
- **Full attention:** HF `q_proj` interleaves queries and sigmoid
  gates **per head**. Split that layout before independent per-head
  Q/K normalization. Use source-derived head count, partial RoPE
  angle and only the configured rotary width. The existing
  `source_rope_kv`, `source_attention_qk`,
  `source_attention_softmax`, `source_attention_value`,
  `source_binary` and one original matrix projection produce an
  unpublished candidate. Original 24Q/4KV mapping repeats each
  KV head for six queries. An `OriginalFullPrefix` holds immutable
  Arc KV slabs, original SourceIdentity and exact layer ID; its
  constructor is available only from the authenticated source-bound
  numerics adapter. A candidate cannot borrow a different model's
  or layer's KV history; it never mutates a previously held prefix.
- **GDN:** use the original fused QKV, z, a and b projections,
  source channel-major four-tap convolution, SiLU, then split
  `[Q_16, K_16, V_48]` in **original** head ordering.
  Repeat each Q/K head three times to align with the 48 V heads.
  Perform independent head L2 normalization; transform stored
  `A_log` to `-exp(A_log)`, add source `dt_bias` in the
  existing `source_gdn_coefficients` and keep beta's sigmoid.
  Reuse the existing `source_gdn_recurrent_step` for value-major
  private-state updates, original GDN gated RMSNorm and the
  owned source out projection. `OriginalGdnPrefix` holds the
  private candidate convolution/recurrent state, source identity,
  layer and proposed position; only a source-bound adapter may mint
  or advance it. This stage explicitly supports the official
  **equal K/V widths and 4-tap** Qwen3.8 contract; unsupported
  geometries refuse instead of changing the kernel.
- **FFN:** source ordinary post-attention RMSNorm, projected
  gate/up, SiLU×up, projected down and candidate residual with
  finite output validation; no alternate token/layer loop.

All of these are *analytical FP32 source-operator diagnostics*.
They reuse the already qualified GGUF-style FP32 numerical
reduction order; **HF's explicit BF16 intermediate casts, FLA
kernel accumulation rules, variant-specific RoPE accuracy and
trained state/logits are not yet qualified**. No output may be
published through SessionAuthority or an API until those
independent gates pass. No GPU kernel is launched.

Bounded independent source-free tests cover 16→48 Q/K repeat with
signed-zero bit preservation, 24Q/4KV six-query-per-key mapping,
per-head interleaved gate split, ordinary `1+gamma`,
`A_log=0`/`alpha+dt=0` yielding gain 0.5 and beta 0.5,
one-token analytical full attention, two deterministic GDN
steps with unchanged original state, source/layer foreign-prefix
refusal, partial-RoPE tail invariance, SwiGLU residual and
malformed/nonfinite/capacity negatives. The original source
owner/leased tensor memory is not duplicated.

**Next:** wire the same operator plan into the *existing*
TaskDriver/SessionAuthority and a pinned HF BF16 numerical reference
(including round/cast positions, GDN/attention state, output heads).
Then validate original trained full logits/StateRoot, source-real
27B host capacity and SSD streaming. Do not infer those capabilities
from tiny analytical tests.

Upstream specification used for original tensor semantics:
[Hugging Face Qwen3.5 modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

## ME-P0 — original Qwen analytical layers and the existing TaskCoordinator

The original `zerogravity-core::state::TaskCoordinator` now has an
**opt-in immutable SourceIdentity binding** and a unique per-coordinator
process-local ID. Its prior generic `new`/`publish` API is preserved.
Source-bound coordinators require `publish_source_bound` and reject
unbound or foreign source identities: this extends the *existing*
publication authority without introducing a second scheduler/session.

The stateless `crates/zerogravity/src/original_qwen_state.rs` maps the
existing `ForwardPlan` state-slot vocabulary (`ConvDomain`,
`RecurrentDomain`, `KvDomain`) to fixed, versioned diagnostic state
domains for **one** original Qwen analytical layer at a time:

- Validate source, coordinator ID, layer, epoch, causal root/frontier
  and exact required domain schemas before reading a proposed step.
- Reconstruct finite, complete CPU values from the existing immutable
  `StateRoot` bytes, or an authentic zero-initialized first state.
  Run the original source-bound GDN/full-attention proposal using
  the already-qualified Qwen3.5 CPU mathematical operators.
- Stage an entire GDN convolution plus recurrence, or complete causal
  attention KV, as private `PreparedDomain` bytes. Reject
  nonfinite/malformed data, and bound diagnostic operation to 8K
  tokens and 64 MiB per-domain; check the *next* full KV budget
  before allocating another history-sized buffer.
- Commit exactly once using the original `TaskCoordinator` atomic
  publisher, after all validation. Abandonment, competing root
  attempts, stale epochs and foreign session/source identities cannot
  advance the state. Prior COW immutable roots remain unchanged.

Prepared proposals borrow the original verified Qwen source owner
until commit/abandonment, so their model weight ownership and retained
host ResourceArbiter lease cannot expire underneath publication.

These are **one-layer FP32 analytical diagnostics**, not a full
multi-layer token loop, compiled original 64-layer forward program,
durable portable StateRoot, BF16 reference, device-ready completion
or trained Qwen3.8 inference. Physical mutable state admission and
real cancellation are distinct production execution gates.
The **next** execution step is to integrate original HF operators into
the sole `forward_token`/`MathematicalBackend` and
`SessionAuthority` state owner, and prove trained BF16/FLA GDN
numerics and original full-vocabulary/StateRoot parity before real
Qwen3.8 27B/Gemma/SSD serving.

## ME-P0 — one full original Qwen FP32 diagnostic token, shared schedule

The existing Qwen3.5 `forward_token` and the original indexed-HF
source call the **same** `qwen35::run_layer_sequence` traversal
in `crates/zerogravity-model-qwen35/src/forward.rs`. That single
bounded, monomorphized iterator handles layer order and early
failure: GGUF `MathematicalBackend` owns its accepted operator
schedule and original HF delegates to source-bound GDN/full-
attention/FFN CPU operators for its profile-specific mathematics.
There is no separate model-load manager or public inference loop.

`OriginalQwenStateBridge::coordinator_for_model` registers all
typed `ConvDomain`, `RecurrentDomain` and `KvDomain` slots in
one existing source-bound `TaskCoordinator`. The four-layer
analytical source fixture has three GDN layers, one full-attention
layer and seven distinct immutable state domains. The explicitly
named `prepare_diagnostic_token_f32`:

1. Validates every domain schema and model source identity before
   numerical work; reads only one immutable root and one source-
   qualified embedding.
2. Uses the common Qwen layer traversal to stage the original
   analytical GDN, attention and FFN values and complete causal
   state for each layer, with the accepted FP32 residual operator.
3. Enforces 8K tokens, 128 layers and 128MiB of total diagnostic
   state, in addition to 64MiB per state domain. A missing or
   malformed late-layer state aborts the entire proposal.
4. Applies the source-bound final ordinary RMS and untied
   `lm_head.weight` projection; requires complete finite logits
   before returning an `OriginalPreparedToken`.
5. Commits all state domains and logits through the existing
   TaskCoordinator only after validation. Discarded, competing,
   stale or foreign proposed tokens cannot advance the root.

Tiny original Safetensors integration tests use a four-layer
hybrid source with all-one F32/F16/BF16 tensor values. The final
synthetic analytical logit is approximately **8**: four positive
uniform hidden lanes normalize to ~2 under original zero-centered
`(1+checkpoint_weight)` RMSNorm (checkpoint weight 1), and each
untied output-head row sums them. This independently derived
oracle checks more than simply uniform vocabulary results. The
complete forward's first GDN state domain matches the separate
one-layer bridge exactly. The shared GGUF loop test also proves
layer-order preservation and fail-fast behavior.

**Qualification boundary:** This is an FP32 **synthetic
whole-model diagnostic**, not source-real trained 27B execution,
not HF BF16/FLA numerical equivalence, not full per-op
`MathematicalBackend` integration for original sources and not
production `SessionAuthority`/GPU completion. We preserve the
GGUF trained source regression independently.

### Remaining production ownership gates

- Qualify original HF/FLA intermediate BF16 casting, accumulation
  and exact reference GDN/attention/FFN operators.
- Compose source profiles in the existing full per-op
  `MathematicalBackend` forward program and original
  `SessionAuthority` without a second token/state runtime.
- Verify trained checkpoint/official tokenizer, complete logits
  and StateRoot, physical memory and device completion.
- Admit original 27B host capacity and independently tested
  SSD streaming before user-facing model execution.

## ME-P0 — original BF16 source-dtype diagnostic (not trained parity)

The original Qwen math now offers two explicit source-bound modes:
`OriginalQwenLayerMath::new` (unchanged FP32 diagnostics), and
`bf16_diagnostic` (finite IEEE BF16 round-to-nearest-even after
source-identified dtype boundaries). The existing
`MathematicalBackend` layer traversal, `SourceIdentity`,
`TensorSource`, CPU kernels and `TaskCoordinator` remain shared.
There is **no BF16-specific token loop, model loader or state issuer**.

Source boundary operator: `crates/zerogravity/src/original_qwen_precision.rs`.

- IEEE BF16 quantization: finite f32 → nearest BF16 with ties-to-even,
  preserving signed zero. Nonfinite inputs/rounded overflow refuse.
- Ordinary zero-centered RMSNorm: promote BF16 input to FP32,
  normalize and multiply `1 + gamma` in FP32, then round once
  to input dtype. Independent Q/K head rows share this same operator.
- Source projection/output and SwiGLU: BF16 result per projection,
  SiLU, product, down-projection and residual. Untied vocabulary
  head also rounds after projection.
- Full attention: per-head interleaved Q/gate and K/V projections,
  headwise norms, source RoPE, FP32 dot with BF16 matmul-output cast,
  separate BF16 scale cast, FP32 softmax with BF16 return, value/GQA,
  sigmoid gate and source output projection. Exact actual GPU
  GEMM accumulation/RoPE cast scheduling remains unqualified.
- GDN: original BF16 QKV/z/a/b projections, four-tap conv and
  separate SiLU return casts, exact grouped Q/K16→V48 mapping.
  `A_log.float().exp` and `softplus(a.float()+dt)` remain FP32;
  beta is cast after sigmoid. The qualified existing recurrent
  CPU operator updates private **FP32** state; its output alone
  returns in BF16. Gated RMSNorm casts FP32 normalized core
  before learned gamma and again at BF16 gated output.
- Distinct immutable `StateRoot` schema versions for FP32 vs BF16
  diagnostic contexts. Neither may prepare a token using the
  other's root, even with the same original model allocation.

Source references: the published
[Qwen3.5 generated model code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
and [modular GDN/FLA fallback](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py).
No upstream SDK or weights are imported.

Independent source-free Python NumPy scalar mathematical fixtures
supply immutable BF16 bit patterns for zero-centered RMS and gated
RMS, signed-zero/tie-to-even cases and the two-step GDN recurrent
operator (BF16 output exact; unrounded FP32 state tolerance 2e-7).
Rust integration runs two full synthetic four-layer BF16 diagnostic
tokens on the existing TaskCoordinator; checks per-operation BF16
representation, 7 domains, two causal frontiers, rollback and
cross-dtype schema rejection. The GGUF 0.8B/2B trained full-logit/
StateRoot regressions remain separate unchanged acceptance gates.

**Still NOT IMPLEMENTED/QUALIFIED:** independent trained Qwen3.8-27B
BF16/FLA reference logits, per-token exact Qwen CUDA/FLA rounding,
torch GEMM accumulation/attention backend parity, performance,
real trained 27B payloads, official tokenizer and production
SessionAuthority/GPU/SSD execution. BF16 stage agreement is not a
claim of original-model learned checkpoint numerical parity.


## ME-P0 — pinned original-Qwen numerical reference observations

**Source identity:** `verified_shards.rs` now calculates a canonical
`source_bundle_sha256` over the independently verified index SHA256 and
every owned shard's basename, file length and whole-file SHA256.
Domain-separated SHA256 input: ASCII
`zerogravity.original.source-bundle.v1` followed by NUL,
the lowercase index hash, then basename-sorted
[LE32 basename length, basename bytes, LE64 shard length,
lowercase whole-shard hash]. Unlike index-only provenance,
changing learned bytes changes the bundle digest. No new source
issuer, loader or inference backend.

**Reference contract:** `original_qwen_trace.rs` reuses the shared
`PlanObservation` format. Explicit, source-bound diagnostics capture
embedding, GDN projections/conv/recurrent, QK/softmax/value,
FFN/residual and final logits with source, config/index/bundle SHA256,
coordinator/root/epoch, token/frontier and FP32/BF16 dtype.
Caps: 32 MiB, 2048 stage records, 1,048,576 values/stage.
Only ONE caller-selected GDN FP32 recurrent-state layer (or none)
is observed per token; capturing all 48 is too expensive.

`OriginalReferenceTrace::from_pinned_json` accepts only a
caller-SHA256-pinned, strictly typed versioned reference document.
Unknown/duplicate fields, incomplete/reordered stages, forged
source/config/bundle, dtype/token/frontier mismatch, nonfinite or
non-BF16 output words, invalid tolerance and over-budget content
refuse. First bad layer/stage/element and both IEEE words are
reported. BF16 outputs require exact bits; FP32 recurrence tolerances
are opt-in and bounded. `ReferenceComparison::exact()` means
bit-for-bit equal; `accepted()` includes explicitly tolerated
nonidentical FP32 values, counted by `tolerated_values`.
Comparing a candidate never publishes session StateRoot.

The exact typed JSON reference format is:

~~~json
{
  "schema": "zerogravity.qwen.original-reference.v1",
  "provenance": {
    "kind": "trained-framework",
    "framework_revision": "pinned-reference-revision",
    "reference_code_sha256": "<64 lowercase hex chars>",
    "config_sha256": "<64 lowercase hex chars>",
    "index_sha256": "<64 lowercase hex chars>",
    "bundle_sha256": "<64 lowercase hex chars>",
    "token": 0,
    "position": 0,
    "gdn_state_layer": 0,
    "dtype": "bf16"
  },
  "observations": [
    {"layer": null, "stage": "embedding", "bits": ["3f800000"]},
    {"layer": 0, "stage": "gdn_recurrent_fp32", "bits": ["3e4ccccd"]}
  ]
}
~~~

This is an **illustrative fragment**, not a complete valid matching
capture. A real reference needs every ordered observation and full
element array (including logits); each value is an exact 8-digit
lowercase IEEE-f32 bit word. The entire reference JSON is separately
pinned by its expected SHA256 at load time. The caller's
`trained-framework` claim does NOT independently prove provenance;
the actual external framework code/revision/source artifacts must be
validated separately. Model traces and weight contents are sensitive:
do not log/export them without source-owner authorization.

**Qualified here:** pinned manifest/parser/comparison, source-bundle
reordering invariant and same-index/different-weights difference;
small source-free four-layer BF16 diagnostics and analytical final
logit eight, first divergent stage/element, missing/reordered
records, bad source/position/dtype, malformed/duplicate JSON,
selected GDN state, bit-exact versus FP32-tolerated comparison,
and BF16 stage bit assertions. The full forward now uses its
existing source-aware BF16 residual cast, closing one previously
unrounded attention residual.

**Still open:** independently captured actual trained Qwen3.8 27B
checkpoint/reference trace, official tokenizer, original HF/FLA
operator and GEMM accumulation equivalence, complete trained
full-vocabulary logits/StateRoot, production physical GPU and SSD
streaming. No Qwen3.8 trained checkpoint, Torch/Transformers SDK or
reference traces were available locally for this milestone.

````

## Source: .agents/plans/model-execution/TODO.md

Original UTF-8 bytes: 45355; SHA256 `ec941d609bf1e5c0eca326e32f2e2a5f0c207b92b50d0ed1cd3173512f6be98a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Model execution

Provenance commit: `6d74f76`.
Scope: P02/P03/P08/P11–P14/P17; CPU truth before GPU specialization.
Contracts: [large immutable sources](../IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md),
[Qwen basics](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md),
[Gemma family](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md).

## Uncommitted

- [ ] **Independent trained BF16/FLA checkpoint gate:** original
  Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 acquired
  through the existing bounded Rust cache pipeline; independent31-file/
  18-shard full hashes and1199-tensor header/index checks passed.
  Whole-model27B execution remains unqualified; bounded trained operator
  qualifications below do not establish full logits or112 StateRoot domains.
  Genuine original layer0 GDN reference now validates initial chunk64 and true
  cached continuation for teachers11/353,48 value heads and original BF16
  parameters. Plain/observed complete outputs, convolution/recurrent state,
  raw source readsets, full before/after source hashes and original owned
  retirement match. The serializer's V2 stride failure remains preserved;
  V3 raw-bit transport passes12 real PyTorch layouts. This is an independent
  public target. An isolated native composition now passes the authentic H48
  cached slice:804960 input and798720 output words, including the complete
  recurrent state, match independently before/after callers drop. The distinct
  BF16-gamma gated RMS slice also passes61632 intermediate,12288 final and
  24576 retained input words. Source hashes, original ownership, cancellation,
  real pressure and final reservations0 pass; strict/release builds and237
  focused tests pass. These operator changes remain isolated from MAIN;
  initial chunk64, native exp, complete GDN,112-domain state and full inference
  remain unqualified. A separate genuine public reference passes32 pure
  operator cases over16 roles,881856 raw BF16 input/output words, including
  complete248320-row head and17408-column down projection. The isolated
  native wide-role replay now matches all881856 words exactly; real pressure,
  pre-cancel, retained head/source after caller drop and final reservations0
  pass. Strict/release builds and17 source/relocation fixtures pass. External
  probes for other roles do not establish trained producers or full-model
  parity; these changes also remain isolated from MAIN.
  Six actual bounded native-provider startup/lifetime cases pass with the
  provider unloaded. Physical provider allocation/math admission remains open.
  Original
  Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc is now acquired
  by the same bounded Rust pipeline; independent official Git/LFS full12-file
  hashes,632 headers/320 text shapes,596BF16+36F32 tensors pass. Its4.548GB
  shard remains above the default4GiB complete-source guard. Genuine pinned
  official CPU two-profile reference capture now passes archive and owned
  cleanup validation: tokens11/353, four full248320 logits,168 canonical
  domains and60 actual intermediates. Official BF16 loading has617 BF16
  parameters; restoring only36 original F32 tensors gives581 BF16+36 F32.
  The profiles differ in47/49 raw tensors per step despite identical argmax;
  neither establishes Rust2B HF numerical qualification. First capture's
  cleanup-name bug is preserved; the repaired fresh run proves actual nonce
  container absence and process reap. Original Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17 pinned
  Transformers5.17/Torch2.13 CPU captures passed archive validation:
  tokens11,353, full248320 logits/all42 domains, actual layer0 stages.
  Keep official473-BF16 loader and source437-BF16+36-F32 profiles distinct;
  first-token argmax321 vs198. Explicit new CPU64 diagnostic vs source-mixed
  reference remains UNQUALIFIED: embeddings/direct layer0 norms and both
  QKV projections exact; layer0 convolution state exact,41/42 domains still
  differ. Full248320 logits differ214245/214285 words, maxabs .22119140625/
  .1875. First token exposes conv/SiLU signedzero; second exposes GDN state.
  Complete pinned convolution, chunk64/cached recurrent, coefficients and
  gated RMS arithmetic, then original GPU/FLA/state/streaming gates before enabling
  trained HF execution or27B serving. User authorizes required runtime/model
  acquisition/testing; dated source-free slice restrictions below are history.
- [ ] **Next trained reference gate:** qualify exact original HF BF16
  rounding/FLA order and independent source-trained logits/StateRoot;
  then bind the original profile to the existing per-op
  MathematicalBackend/SessionAuthority, with full physical
  device/SSD resource admission. Earlier source-free slices establish no
  full27B logits,112-domain state, native provider admission or GPU support;
  current owner authorization covers the required acquisition and testing.

- [ ] **Following ME-P0 real trained integration:** fuse the
  accepted analytical original source operators into the ONE
  existing forward_token/MathematicalBackend and
  SessionAuthority execution schedule, qualify original HF BF16
  intermediate casts and FLA GDN accuracy, source-trained exact
  full logits and StateRoot, tokenizer, physical mutable-state
  memory accounting, cancellation and device completion. No
  SSD/physical inference claim follows from the diagnostic bridge.

- [ ] **ME-P0 next after diagnostics:** wire the analytically verified
  Qwen original operators into the ONE existing forward-plan and
  TaskDriver/SessionAuthority (no second token loop or state issuer),
  with exact source BF16 intermediate cast and original FLA GDN kernels,
  real trusted tokenizer and full 248320 logits/42-or-more StateRoot
  domains against the pinned checkpoint. Only then begin
  SSD-resident streaming and measured Qwen3.8-27B physical throughput.

- [ ] **ME-P0 next numerical stage:** source-exact original Qwen
  full attention, GDN/MLP, grouped Q/K→V replication, A_log/gate
  transformation and BF16 intermediate numerical schedule using the
  shared original TaskDriver/StateRoot. Qualify tiny independent
  analytic operator/state gates before real approved trained Qwen3.8
  27B original full-vocabulary reference. Measured original Qwen3.5
  2B CPU/GPU end-to-end performance also remains open.

- [ ] **ME-P0 next:** bind source-approved original Qwen3.8 27B
  tensor/index/config and tokenizer to the existing mathematical engine
  with authentic head/axis geometry, strict trained CPU full logits/
  StateRoot, then source-verifiable immutable SSD extents with stable
  leased memory and measured decode/prefill. First finish measured
  Qwen3.5 2B CPU/approved GPU end-to-end performance and
  cancellation/physical readiness. Preserve no model/SDK downloads
  and held device launch policies without explicit authorization.

- [ ] **ACTIVE NEXT: ME-P0 Qwen3.5 2B then Qwen3.8 27B.**
  Begin with original qualified Qwen3.5 .8B/2B CPU and resident GPU
  source/state oracles, inspect release source ownership and exact
  model-2B profiling on the permitted CPU/GPU devices. Keep tests
  bounded to existing 8K context now; 16K/32K extensions remain
  separate numerical gates. Qualify a stable single-model fast path,
  original tokenizer/math/KV/recurrent/StateRoot and actual device
  completion first. Then implement checked immutable source shard leases
  above the present 4GiB executable-source gate and Qwen3.8 27B
  complete trained CPU logits, including SSD streaming and quant-aware
  physical placement. Do not fetch weights or launch held Renoir/gfx90c
  hardware without the corresponding permission and source integrity.
- [ ] **NEXT after Qwen: ME-P1 Gemma4 E2B→26B-A4B→31B**, independently
  source-bind tensors/reference semantics, PLE/MoE/dense quantization
  and trained CPU→GPU correctness before measured throughput
  optimization. **ME-P2 Qwen3.8 Flash-Next** follows with separately
  verified learned n-gram embedding/4B MTP heads, QSA/GDN/MoE and
  explicit Q2/~3.5bpw/Q4 source quant choices (NOT guessed files),
  speculative n-gram as a distinct runtime optimization. Other OSS,
  ASR/TTS/image/video/decision execution and multi-model router remain
  on their own deferred source-real gates.
- [ ] Preserve direct one-model-per-process execution and one
  shared-instance-many-pipelines as **one Rust fabric**; the accepted
  YAML DAG/compiler and extension schemas add no inference backend.
  Do not resume context snapshot/provider/multimodal/router design
  work while ME-P0/P1 are open unless they genuinely block those models.

- [ ] Enforce new owner priority: Qwen3.5 .8B regression, Qwen3.5
  2B highest-perf trained CPU/GPU, Qwen3.8 27B with SSD
  streaming/owned shard tensors, Gemma4 E2B, 26B-A4B, dense 31B,
  then Qwen3.8 Flash-Next QSA/GDN/MoE/gated residual/MTP
  and 51B learned n-gram embeddings. Every additional DeepSeek,
  GLM/Muse/Mistral/Llama family has a typed NOT IMPLEMENTED
  source contract until distinct trained CPU and GPU gates pass.
  No subfamily is generalized from metadata-only source support.
- [ ] Treat single-model local GGUF execution and multi-model
  pipeline-qualified execution as two frontends of ONE TaskDriver/
  SourceIdentity/StateRoot fabric. A reusable model instance
  serves multiple routes without another model process or implicit
  source promotion; named replicas/adapters require separate
  physical memory admission.

- [ ] After the accepted source-acquisition v1 cutover, use the existing
  pinned HF index-first path to inspect explicit original Qwen3.8-27B and
  Gemma4 26B-A4B index/config/tokenizer/chat/processor identities
  (see [acquisition v1 handoff](../model-acquisition/README.md)).
  Select an authorized exact revision and admitted local host. Do not treat
  synthetic >51GB transfer-plan tests, public model pages or cached bytes
  as authorization, execution-ready source, or trained CPU parity.
  2026-10-18 tokenless live metadata-only HF source observation resolved
  Qwen3.8-27B to `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  (18 shards, 55,586,113,293 selected bytes) and Gemma4-26B-A4B-it
  to `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
  (2 shards, 51,644,338,194 selected bytes); Gemma4 E2B-it commit
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
  (1 unsharded safetensors, 10,278,848,001 selected bytes).
  No trained weights downloaded, locally owned, header-audited or
  model-execution-qualified yet.

- [ ] Acquire/bind authorized original config/index/tokenizer/checkpoint identities and choose a capable admitted host. Large BF16 sources exceed current local RAM; current executable source limit stays 4 GiB.
- [ ] Qualify real large (>4GiB) original trained shard payload on an
  authorized capable host using the already verified owned-shard
  implementation from 93a5622; independently measure resident
  allocation/physical RAM pressure and immutable lease lifetime. Next,
  develop the separately admitted SSD streaming extent architecture.
  The bounded source-only synthetic tests do not establish real large
  learned weight loading.
- [ ] Qwen3.8 27B: original HF grouped-head ordering and Q/K replication 16→48, explicit converted-profile permutations, untied head and source-derived norm/RoPE geometry.
- [ ] Gemma E2B/26B: shared GELU/multiply/scale/soft-cap, proportional RoPE, value norm/sliding/shared KV, E2B PLE/doubled FFN and 26B K=V/MoE branches; choose and qualify the exact reference-framework numerical schedule.
- [ ] Qualify actual trained CPU operators/full vocabulary/state/nonzero replay, official tokenization/templates/reasoning/stop behavior, sessions, rollback/checkpoints and resource refusal.
- [ ] Add the same source-bound forward plans and full physical GPU qualification per eligible device, preserving all state/actual-completion/cancellation guarantees.
- [ ] Keep distinct future heavy/hybrid recipes (Qwen Flash-Next, GLM, MiMo and source-defined QSA/MLA/KDA/indexers/hyperconnections), and broaden dense/encoder/pooling/reranking/LoRA support only through independent model/codec/backend gates.
- [ ] Preserve CPU-first family priorities; already-authorized parallel basics do not authorize model/SDK downloads or unsupported whole-model claims.

- [ ] Bind explicit original HF versus converted Qwen source profiles. Conversion permutes grouped/tiled V projections/constants/convolution/output, uses -exp(A_log) and adds1 to non-GDN norms; GDN swish is SiLU while full-attention gating remains sigmoid.
- [ ] Choose and independently qualify Gemma26B reference-framework routing/top-k/reduction/scale placement before production execution. Five analytic test-local routing cases are not trained source parity.

## Committed

### fff1842 — trained BF16 operator qualification

- [x] **Native CUDA CPU-profile BF16 projection:** both original Qwen3.5
  sources, layer3 q/k/v windows and teachers11/353, B1/independent B1/B2/4/8:
  163840 raw F32 accumulation words match the accepted CPU64 operator and
  163840 rounded BF16 words match authentic public archives exactly.
  Same sealed full-source windows, original CUDA queue/service/leases/Ready;
  pressure, pre-cancel, source/axis refusals, final physical0 and fresh actual
  abandoned-flight source/completion/fence retention pass. Synthetic328
  case/band pairs cover source widths64..8192, signed zero, subnormals, RNE,
  FMA/reduction order and canaries; eight fresh FP32/BF16 overflow flights
  withhold Ready and retain original quarantine. Strict Clippy/five focused
  tests and independent trained/synthetic raw-byte validators pass. This is
  the explicit CPU-target profile, not vLLM CUDA arithmetic, full HF/GDN,
  model StateRoot or a throughput claim. Original failed build/import checks
  and their additive harness corrections remain preserved.
- [x] **Original GDN convolution/SiLU fragment:** both authentic HF sources,
  layer0 teachers11/353, initial padding and cached-tail inputs: all122880
  BF16 words across raw convolution, activated output and canonical next3
  history match independent public Torch operators exactly. Public outputs
  first reproduce original activated/K4 archives. Same ForwardPlan/HostMath,
  sealed raw BF16[C,1,4] SourceParameter and externally authenticated inputs;
  actual pressure/pre-cancel/wrong-role, retained output after callers drop,
  full parameter stage accounting and final physical0 pass. Core10/artifact2/
  model48 tests and strict CPU/CUDA/Vulkan checks pass; trained replay has
  independent all12-file signed-bit validation. Standalone JSON-macro syntax
  failure is preserved; fresh ROOTV7 changes only harness parentheses.
  Full HF/session/checkpoint guards remain closed. Initial FLA chunk64,
  cached recurrent state, coefficients/gated norm and full logits/StateRoot
  remain required before enabling original2B/27B inference.
- [x] **Composed BF16/conv phase:** complete isolated1251 Rust tests,
  zero failures/70 ignored/248 suites,18 Python tests, release/installer/Nix
  checks pass. Composed strict CUDA/qualifier/Vulkan and CPU-free binding8/
  GPU cold-refusal2 tests pass. Both approved original release cache-direct
  and HF-import routes reproduce full248320/all42 for both models and retain
  original source/lease/mutation/reclamation guarantees.

### fe2a342 — original attention and reference operators

- [x] **Original CPU attention fragment:** both authenticated sources at
  layer3, teachers11/353, source-36-F32-restored profile: all47104 raw BF16
  words across seven projection/view/QK-norm outputs match authentic original
  captures, including signed zero. Same HostMath/ForwardPlan and sealed source
  windows; real source-pool pressure, pre/mid-cancel, foreign input/provider,
  retained output after callers drop and final reservations0 pass. Explicit
  external hidden inputs establish no preceding producer or StateRoot.
  Focused16 tests, strict CPU/CUDA/Vulkan, both approved original release
  routes and complete isolated1243-test phase pass. Fresh helper strict build,
  full source scans and owned process/group/EOF cleanup also pass.
  Earlier metadata-list and omitted second4MiB binding-accounting failures
  remain preserved; corrections change only the replay harness. RoPE, KV,
  attention BMM/softmax, GDN/FLA, full-HF logits/StateRoot and27B stay open.

- [x] **Authentic vLLM projection operands:** read-only extraction from the
  accepted four-call eager-B1 capture preserves all original full496640-logit,
  320-source-audit, provider and cleanup guards. Six raw BF16 files,36992 bytes,
  bind actual teacher11/353 inputs/results, public GEMM pointers/settings and
  learned QKVZ(qkv,z)/BA(b,a) row hashes. Ten transport tests and independent
  rederivation pass; result f0981e62. No new model/GPU execution or dtype cast.
  Native BF16 arithmetic, opaque provider allocation admission and full-HF
  execution remain unqualified.

- [x] **Original2B paired attention observation:** fresh plain/observed CPU
  captures preserve all196 raw tensors/four full248320 logits/168 canonical
  domains/60 prior intermediates against the accepted two-profile reference.
  Observed capture adds216 genuine BF16 tensors across six attention layers,
  nine stages and teachers11/353 in both original loader profiles,688128 bytes.
  Exact signed-bit/shape/stride/source checks, unchanged archive validator,
  all632 source/617 loaded parameter audits and owned group/container/EOF
  cleanup pass. Ten parser fixtures pass. Diagnostic elapsed44.56s/42.01s;
  no benchmark or physical RAM admission claim. Result e45e1903 qualifies
  transparent reference-stage availability, not Rust attention output,
  full HF execution/StateRoot, GPU/FLA, long context or trained27B.

- [x] **Genuine pinned vLLM V1 eager B1 public GEMM observation:** four
  unchanged original0.8B layer0 QKVZ/BA projections at teachers11/353 use
  cublasGemmEx, BF16 A/B/C, FP32 compute, requested DEFAULT_TENSOR_OP,
  default math/host pointers/atomics disallowed, alpha1/beta0 and explicit
  8519680-byte workspace. Actual full tensor/workspace allocation extents,
  original provider versions/mapped SHA and loaded320 source roles before/
  after pass; plain/observed all496640 logit words are exact. Strict observer
  build and all three owned process/container/group/EOF cleanups pass.
  Original header-build and generic resolver failures remain preserved;
  additive exact already-loaded provider binding has no alternative lookup
  or math fallback. Prior resolver failure cause remains unproven.
  This records public requests, not hidden selected kernels, all42 state,
  wider/default graphs, Zerogravity GPU math/profile or matched performance.
  Qualifier result55d68c7a; explicit workspace alone does not qualify opaque
  library allocation admission. Pinned default vLLM loader/quant registries
  have no GGUF entry; original BF16 remains the matched-source route.

### bfe62df — leased source-window execution

- [x] **Shared source-window HostMath fragment:** original27B851-role
  namespace and same PlanBuilder/HostMath embedding→ordinary RMS→full-column
  projection now execute genuine trained bytes without source materialization.
  Both tokens11/353 match all40,960 independent BF16-widened reference words,
  including full5120-wide embedding/norm and10240-row QKV. All18 shards/
 55,563,006,776B are verified through retained FDs. All80 authentic128-row
  windows preserve5120 input columns; maximum real QKV staging charge8,388,656B
  and raw norm staging4,194,328B are measured, not inferred from requested bytes.
  Each output retains admitted64KiB Output+960KiB Scratch and source issuer
  through last drop. Real stage-triggered mid-cancel, pre-cancel, pool pressure,
  foreign-source/config/index/token refusal and retained output comparisons pass;
  final reservations0/unfenced. Actual92.74s includes90.79s source scan;
 18,736KiB observed RSS is not physical free-RAM admission evidence.
  Fresh MAIN-linked original0.8B three-operator regression also matches all
 16,384 words/final source0. Focused17 artifact/1CPU/5model, strict CPU/CUDA/
  Vulkan/qualifier Clippy and full isolated1243 Rust/0failed/70ignored/248suites
  plus18Python/release/installer/Nix pass. Both approved original cache-direct
 17.22s/HF-import22.74s release routes pass fullV/all42/reclamation for both
  models. Only four pointless borrowed-value test drops needed repair. This is three pure state-free operators,
  frontier delta0: full112-domain27B logits/StateRoot and streamed inference
  remain unqualified.


### 5350a4d — typed ordinary math and source windows

- [x] **Typed ordinary RMS in the shared schedule:** the same PlanBuilder/
  HostMath embedding→raw InputNorm(0)→GdnQkv(0) fragment preserves source
  parameter identity, BF16 precision, epsilon1e-6 and explicit packing.
  Fresh MAIN replay matches all16384 genuine original0.8B reference words,
  two tokens11/353, with final source reservation0. Unaware providers and
  GPU plans refuse before callbacks/source admission; full HF session,
  forward and checkpoint guards remain closed.
- [x] **Sealed rank1 parameter windows:** BF16/F16/F32 raw ranges retain
  original full descriptors, issuer, opened source and actual staging charge;
  source/cancel/corruption/capacity/drop tests pass. Partial ranges never
  become complete TensorSource views. This is transport, not27B inference.
- [x] **Closed original BF16 row-window projection:** authentic original
  full-matrix descriptors and complete input columns reuse the accepted
  CPU64 dot without changing its arithmetic. Output-row partition, width5120,
  foreign source/range/dtype/nonfinite/cancel/provider and lease tests pass.
  No partial matrix/provider substitution or final BF16 cast; caller must
  admit private output/scratch. Same-HostMath27B integration remains open.
- [x] **Closed pinned F32 softmax:** source Vec8 maximum/sum order, unchanged
  SLEEF exponential and F32 reciprocal/multiply, width1..512. Genuine public
  full3599-case replay matches1838736 output words and3 refusals; authenticated
  default72-case golden preserves11280 expected words/3 refusals. NaN/+Inf/
  all-masked rows refuse. No legacy GGUF or full HF attention enablement.
- [x] **Composed acceptance:** focused core9/artifact16/CPU47/model6 pass,
  compact default5 pass; full isolated gate1241 Rust/0failed/70ignored/248
  suite runs plus18Python/release/installer/Nix passes. Strict composed CUDA+
  HF+CPUcache, Vulkan serving and CUDA qualifier pass; CPU-free HF binding8
  and GPU cold refusal2 pass. Both original cache-direct/HF-import RELEASE
  routes pass(18.88s/24.75s), full248320/all42 and final source reclamation.
  Full27B Rust reference comparison, streamed model state and CUDA/FLA remain
  open. Compilation/type and Clippy-only first failures were fixed; no
  mathematical tolerance or full-profile guard changed.


### c5eb1f8 — closed pinned sigmoid and ordinary RMS

- [x] **Closed pinned BF16 sigmoid:** explicit contiguous iterator spans,
  vector32 prefix plus the original scalar tail, private SLEEF and licensed
  Arm/glibc-FMA arithmetic. Genuine public Torch and Rust replay match all
  326753 output words, including all65280 finite BF16 values in vector/scalar/
  mixed call shapes and both actual original beta16 calls. No host exp,
  fitted output table, numerical-profile enablement or GGUF math change.
- [x] **Closed pinned ordinary RMS:** raw immutable rank1 BF16/F32 learned
  weights, exact BF16 input, source four-Vec8 reduction/carry order at widths
  256/1024/2048/5120, F32 division and runtime(1+raw weight), then final BF16
  cast after the F32 product. Actual original .8B122 norm calls preserve98
  prior full-logit/cache checks;32 public width/dtype fixtures cover the
  larger geometries. Rust matches all154 cases/1694 full stage fields/
  1680488 words exactly. Width fixtures do not qualify2B/27B inference.
- [x] **Composed source-free gate:** complete isolated check1218 Rust passed,
  0failed/69ignored/248 suite runs plus18Python, release/installer/Nix;
  ordinary RMS7 focused tests and strict CPU Clippy pass. Independent source
  review authenticates every6792 compact golden word from genuine public
  archives. Strict final composed CUDA+HF, Vulkan serving and CUDA qualifier
  Clippy pass; both approved cache-direct/HF-import release routes pass
  (20.09s/24.88s), both original models, full248320/all42, independent sessions,
  source leases and reclamation. Original-HF full session/forward/checkpoint
  guards remain closed; coefficient exp, initial chunk64, attention and
  physical model/state integration are still required.

### 2a32aff — typed source composition and vector softplus

- [x] **Typed original-source operator integration:** explicit program
  profile and activation precision, BF16 OutputInput embedding/projection
  contracts and raw SourceParameter metadata use the existing PlanBuilder
  and HostMath. Private operator slices carry no domains, logits receipt
  or frontier advance. Unknown physical providers and resident GPU plans
  refuse before source reads/admission. Raw learned parameters stay borrowed
  from the same owner; existing RootBinding descriptors distinguish initial/
  cached phase, BF16 conv/KV and FP32 recurrent axis layouts. These metadata
  views grant no freshness, state transport or publication capability.
  Complete original-HF session/forward/checkpoint guards remain closed.
- [x] **Genuine typed operator qualification:** pinned public Torch
  F.embedding for original tokens11/353 followed directly by independent
  GdnQkv(0), FfnGate(0) and tied Output projections matches every Rust bit,
  including all248320 words of both output projections. Eight reference
  payloads preserve518144 values; same loaded source, raw matrix pointers,
  two-node recorder, zero initialized state and final source reservation0
  are checked. No input normalization, complete layer/model logits or
  GPU/27B inference is claimed by this pure operator composition.
- [x] **Closed pinned FP32 vector softplus:** source beta1/threshold20,
  finite general-F32 contiguous spans divisible16, explicit SLEEF3.8
  compensated log1p/exp/FMA order and BSL notice. Actual32 original biased
  alphas and28464 adversarial F32 values, reordered values and16/32/48/64
  spans match public Torch bit for bit. No scalar-tail, MKL coefficient
  exp, general profile or default GGUF arithmetic change.
- [x] **Typed/softplus composed gate:** full isolated check1213 Rust
  passed/0failed/69ignored/248 suite runs plus18Python, installer/release/Nix;
  focused core/CPU/model158 passed/2ignored, CPU-free binding10 passed,
  resident cold refusal2 passed. Strict composed CUDA+HF, CPU-free HF,
  Vulkan serving and CUDA qualification Clippy pass. Both approved original
  cache-direct/HF-import release routes pass (17.15s/23.12s), both original
  models, full248320/all42, independent sessions/source lifetime/reclaim.

### cadcbd9 — closed pinned original-HF CPU operators

- [x] **Closed pinned CPU original-HF operators:** source-bound BF16
  four-tap convolution, FP32 width128 cached GDN, raw-F32-gamma gated RMS
  prefix and contiguous vector SiLU. No model/session/initial-chunk permit;
  default GGUF/scalar profiles stay unchanged. Convolution preserves four
  independent FMA partials followed by direct adds and BF16 output. Cached
  GDN uses direct query division by sqrt128, actual key-major FP32 state,
  materialized key reductions and separate update multiply/add; observed
  decay is an explicit input. RMS casts normalized core before raw original
  F32 gamma, retaining the FP32 product. SiLU ports pinned SLEEF3.8 AVX2
  exp/FMA/scaling and FP32 division with source attribution/license, only
  finite exactly BF16-widened contiguous inputs with length divisible16.
  Twenty-eight focused Rust tests pass including old projection/reducers.
- [x] **Actual closed-operator Rust qualification:** untouched pinned
  Torch2.13 public SiLU with observed SLEEF AVX2 provider supplies exact
  outputs for ALL65280 finite BF16 inputs plus4096 actual source gates;
  Rust matches every F32 bit. Full16-head cached replay matches262144
  state words plus5*2048 normalized/scaled/core/output words. Gated RMS
  prefix matches8256 words across both actual tokens/all16heads.
  Convolution and composed vector-SiLU/BF16 match all6144 channels at
  four actual causal positions (49152 output words). These are pure
  operator replays of genuine source operands, not full-model/profile,
  GPU/FLA or27B numerical acceptance. Initial chunk64, coefficient exp/
  sigmoid/softplus, ordinary RMS and eager attention remain open.
- [x] **Composed verification:** complete isolated check1202 Rust passed/
  0failed/69ignored/248 suite runs plus18Python/release/installer/Nix.
  Strict CUDA+HF-binding+portable, Vulkan serving and CUDA-qualification
  Clippy passed; CPU-free original binding6 and CPU serving22 passed.
  Both approved original0.8B/2B cache-direct/HF-import RELEASE routes
  retain all248320 logits/all42 domains, source lifetime and reclamation.
  Independent long-context GGUF GDN divergence remains a separate required
  gate in [numerics-and-context](../numerics-and-context/TODO.md).

### 9815504 — original HF roles and closed CPU projection/reducers

- [x] **Original HF source roles and closed CPU profiles:** authentic rank/axes,
  raw source dtype and tied/untied heads bind to the same loaded owner;
  Safetensors output/input rows remain untransposed. Typed original-HF
  metadata carries the actual config/index/bundle and independent state
  geometry. Its execution/session/checkpoint-v1 entry points refuse before
  allocation/backend/I/O; metadata is not an executable-model promotion.
  Explicit Torch CPU AVX2 BF16 projection diagnostics retain the pinned
  64-lane FMA/reduction order and separate state/trace schema; the original
  scalar/GGUF profiles remain distinct. Actual original .8B layer0 QKV
  is exact6144/6144 for each of tokens11,353. Closed FP32 width128 L2 and
  key-axis reducers pass authentic compact bitwise goldens and adversarial
  head/order/finite/overflow cases. Real attention-input norm observations
  now cover both full and GDN layers. Focused composed suites185/0/2ignored,
  CPU-free binding8/0, strict CPU/CUDA/Vulkan/qualifier Clippy and full
  isolated check1164/0/69ignored/245suites+18Python passed. Both approved
  original .8B/2B cache-direct/HF-import release routes remain exact full
  logits/all42. Full original HF mathematics remains UNQUALIFIED.


### f26232e — authentic tied original output heads

- [x] **Original HF tied output head:** verified `tie_word_embeddings`
  selects the original embedding or independent `lm_head` projection;
  no alias tensor or fallback source. Source-free authenticated tied/untied
  fixtures test opposite-sign independent heads, exact F32/BF16 diagnostic
  logits, immutable source identity and final lease0. Full isolated gate:
  1148 Rust passed/0 failed/69 ignored/245 suites,18 Python. Both approved
  original0.8B/2B cache-direct/HF-import release routes retain exact full
  logits/all42. This fixes the diagnostic head; no trained HF promotion.

### 1c0bb41 — pinned numerical reference seam

- [x] **ME-P0 pinned original numerical reference gate:**
  extend one immutable source owner with canonical source-bundle
  SHA256 over pinned index and all authenticated shard bytes.
  Capture a bounded subset of original Qwen GDN, attention, FFN
  and logits through shared PlanObservation, with opt-in 32MiB
  private trace and a single selected GDN recurrent state layer.
  Strict typed independently SHA256-pinned JSON comparisons bind
  config/index/bundle, exact stage order/geometry/position/dtype,
  detect first mismatched element, and distinguish truly bit-exact
  BF16 output from bounded, tolerated FP32 state deviations.
  Source-free adversarial tests verify same-index/different-weight
  digest changes, duplicate/missing/reordered reference content,
  bad token/source/dtype, selected GDN state and immutable roots.
  Fix full-model BF16 residual result-cast omission by reusing
  its source-typed existing residual binary operator.
  This stage DOES NOT prove independence merely because a
  reference JSON asserts trained-framework origin.
  Complete isolated scripts/check.sh passed 1,135 Rust tests, 0 failures,
  69 intentionally ignored across 238 suites plus 18 Python fixtures,
  release CLI/installer and CPU numeric comparators. Strict default and
  composed CPU/cache/Qwen/acquisition Clippy -D warnings pass.
  Independent approved Qwen3.5 .8B/2B release cache/direct 17.45s and
  HF-import 21.97s preserve exact full 248320 logits, all 42 StateRoot
  domains, two-token independent sessions and source/lease correctness.
  Offline Nix source-integrity passed with the checked trace files.
  No trained Qwen3.8 reference or Torch/FLA runtime was accessed.

### 6d74f76

- [x] Qwen3.8 27B and Gemma4 E2B/26B configuration, tensor and source-index contracts; no new trained execution, tokenizer or checkpoint loading.

### Source contracts — 023703f

- [x] Qwen3.8 metadata/shard roles and CPU-free Gemma4 family contracts,14 Gemma tests and official metadata/config cases passed within metadata scope. No tokenizer, new learned weights, physical shard-header binding or trained inference follows.
- [x] Preserve Qwen/Gemma source axes, metadata precision and routing/KV facts in the [model semantics reference](../../docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md) and [Gemma technical contract](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md). The executable-source4GiB cap remains.

### 93a5622 — original verified immutable source shard owner

- [x] **ME-P0 LS-B01–B05, verified owned sharded-source allocation:**
  implement `SourceWeights::prepare_verified_shards` in the existing
  artifacts owner with exactly opened original Safetensors file
  descriptors, caller-expected index SHA256 and whole-shard SHA256/length,
  one original SafetensorsIndex parser and checked header name→shard,
  dtype/shape/offset/payload binding. Preflight limits and reserve whole
  captured file allocation plus a conservative metadata envelope via the
  original caller-attested host `ResourceArbiter` before copying. Reparse
  exact owned bytes, compare authenticated headers against preflight
  (including same-length mutation) before issuing the single original
  SourceIdentity. Arc owner lifetime pins all original slices and RAM
  charge; default GGUF `SourceWeights::open` keeps the 4GiB cap and
  original Qwen execution path.
  10 source-only integration tests pass: positive F32/F16/BF16, complete
  index closure, foreign tensor metadata rejection, wrong SHA/length,
  missing/extra/duplicate/wrong shard, same-descriptor pathname
  replacement, original inode mutation, verified copied immutability,
  multiple admitted/shared original leases and exact resource rollback.
  Sparse >5GiB header tests refuse before reads under both 4GiB
  default and physically insufficient explicit >4GiB budgets.
  **No actual >4GiB weight load, SSD streaming, trained original
  Qwen27B/Gemma math, tokenizer or physical GPU support is claimed.**

### b796a2e — original Qwen text source geometry and CPU value binding

- [x] **ME-P0 original Qwen3.8 source/value binding (small original
  format, not trained 27B):** bind caller-authenticated original
  QwenTextConfig SHA256 and pinned indexed-source SHA256 on the *same*
  immutable SourceWeights allocation. Check every ordinary-text
  name and original [output,input] tensor shape/encoding, including
  the untied output head; distinguish source extra vision/MTP from
  unregistered text roles. Reuse the existing CPU source MatVecRegion
  for HF packed F32/F16/BF16 output-first matrices with no transpose,
  plus the existing embedding/parameter-vector decoder through a
  source-bound QwenOriginalCpu façade. Small complete 4-layer hybrid
  original-format integration tests and negative hash/shape/missing-
  head/source-provider tests pass. Oversized HF source matrices
  refuse before the optional accelerator is invoked (observed no-dispatch).
  The complete frozen isolated
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  passes 1,103 Rust tests, 0 failures, 69 ignored across 238 suites,
  plus 18 Python namespace fixtures, strict lint, installer/Nix
  formatting and CPU numeric selftests.
  Both independent Qwen3.5 0.8B/2B original trained
  source-parity release routes pass AFTER the final safety guard:
  cache/direct 16.70s and HF-import 22.88s, each full 248320-vocabulary
  logits, all 42 StateRoot domains, two tokens, independent sessions
  and lease/mutation/reclamation exact. Default and
  acquisition/cache-CPU strict `cargo clippy --locked --offline
  --all-targets -- -D warnings` checks also pass. The original
  GPT-family GGUF CPU and session path is unchanged.
  Does NOT run trained 27B, create a Qwen session, implement
  grouped GDN permutations, or stream SSD.

### b11e58c — original source-bound Qwen analytical one-layer proposals

- [x] **ME-P0 original Qwen3.8 analytic one-layer operators:**
  source-bound ordinary zero-centered RMS gamma (1+weight),
  HF head-interleaved full Q/gate projection, per-head Q/K norm,
  source-configured partial RoPE, 24Q/4KV grouped lookup,
  QK/softmax/AV/sigmoid/out projection and candidate KV slabs under a
  source-and-layer-bound immutable `OriginalFullPrefix`. Source GDN
  four-tap causal conv and SiLU, exact 16→48 repeat_interleave of
  original Q/K, per-head L2, stored A_log→-exp(A_log),
  dt_bias/softplus/sigmoid beta through the existing coefficients and
  recurrent step, output GDN gated norm/out projection. An immutable
  `OriginalGdnPrefix` holds exact source/layer and private tentative
  conv/recurrent state, never replacing SessionAuthority. Source FFN
  post-attention RMSNorm, SwiGLU and residual reuse old math.
  Small independently derived analytical checks include zero-centered
  norms, 16→48 head signed-zero mapping, 24Q/4KV GQA six-query
  grouping, ln2 decay beta0.5/gain0.5, full attention gates/partial
  RoPE, GDN two-token private repeat/branch, source identity mismatch,
  finite and dimensional errors. The original 8K diagnostic bound and
  original default GGUF behavior remain. **No trained HF BF16 intermediate
  rounding, full layer/model StateRoot, tokenization, actual 27B learned
  weights or GPU/SSD support claimed.** Strict source-config-specific
  numerical reference and official checkpoint tests remain open.
  Final isolated `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4
  python3 scripts/native-gpu.py -- bash scripts/check.sh`
  passed 1,111 Rust test executions, 0 failures, 69 ignored across
  238 suites, plus 18 Python namespace fixtures, release CLI/installer
  and CPU numeric-comparator scripts; strict default and composed
  CPU/cache/Qwen/acquisition Clippy `-D warnings` passed.
  Actual offline Nix source-integrity passed. Two independently run
  Qwen3.5 .8B and 2B trained-source release routes remain exact:
  cache/direct 17.36s and HF-import 22.49s, each original complete
  248320 logits/42 StateRoot domains/two tokens/source leases.
  This diagnostic-only milestone is recorded in signed commit b11e58c.

### 6329cff — source-bound Qwen one-layer TaskCoordinator

- [x] **ME-P0 source-bound original Qwen StateRoot admission
  (one-layer diagnostic only):** the sole existing TaskCoordinator
  now supports an opt-in original immutable source identity plus
  a unique coordinator/session ID while retaining its generic
  publication API. A stateless source-bound one-layer bridge maps
  canonical ForwardPlan Conv/Recurrent/Kv StateSlots to versioned
  GDN or full-attention StateRoot domain bytes, with exact model,
  epoch/session, layer, root, schema and frontier verification.
  Immutable source-pinned numerical proposals publish via the
  existing TaskCoordinator only after complete finite-domain
  preparation, 8K token/64MiB diagnostic caps and early next-KV
  admission. Independent tests cover GDN two-domain and causal
  attention KV commit, abandoned and stale competing steps,
  previous immutable roots, corrupted/nonfinite state refusal,
  source/session/schema mismatch and generic coordinator parity.
  There is NO separate multi-layer token loop, model scheduler or
  new state issuer; this is not yet trained BF16 reference parity.
  Final isolated full scripts/check.sh passed 1,116 Rust tests,
  0 failed, 69 intentionally ignored across 238 suites plus
  18 Python namespace fixtures, release CLI, installer, shell/Nix
  formatting and CPU comparator selftests. Strict default and
  composed CPU/cache/Qwen/acquisition Clippy -D warnings passed;
  independently repeated Qwen3.5 .8B/2B release cache/direct
  17.11s and HF-import 22.27s retained exact 248320 logits,
  42 StateRoot domains, two tokens and source lease behavior.
  Offline Nix source-integrity passed before signing.

### e43ba4c — original Qwen full-model FP32 analytical forward

- [x] **ME-P0 full original-HF model analytical forward:** extract
  the one Qwen layer traversal from the existing GGUF forward_token,
  preserving its complete MathematicalBackend operator order.
  Use it also for original source-bound GDN/full-attention/FFN,
  each layer's private Conv/Recurrent/Kv state and residual output,
  and final RMSNorm/untied head. One opt-in TaskCoordinator
  owns **all** model state domains, not one root per layer.
  Validate all source identities/schema/finite bytes, 8K/128-layer/
  128MiB diagnostic bounds and complete logits before atomic
  publication. Tests: complete synthetic four-layer original
  checkpoint with seven state domains/two tokens, matching
  independent one-layer GDN, abandoned/competing/foreign session
  and late-corrupted state refusal, derived final logit ~8, and
  shared GGUF traversal ordering/fail-fast. The method is explicitly
  named `prepare_diagnostic_token_f32`: this is **not** trained BF16/
  FLA inference, production TaskDriver or an alternate runtime.
  Complete isolated scripts/check.sh passed 1,119 Rust test
  executions, 0 failed, 69 intentionally ignored across 238
  suites, plus 18 Python namespace fixtures, release installer/CLI,
  Nix/shell formatting and numeric comparator selftests.
  Strict default and composed CPU/cache/Qwen/acquisition Clippy
  -D warnings passed. Both independent original Qwen3.5 .8B/2B
  trained-source release routes preserve exact 248320 full logits,
  42 StateRoot domains, two tokens and source/session leases:
  cache/direct 17.88s and HF-import 22.59s, both exit0.
  Offline Nix source-integrity passed before the signed commit.

### aaf9ba6 — original Qwen BF16 source dtype diagnostics

- [x] **ME-P0 explicit original BF16 dtype diagnostic:**
  implement finite nearest-even BF16 rounding with signed-zero and
  overflow refusal; introduce optional BF16 source stages into the
  existing original Qwen source binder/CPU operators, without
  mutating the default FP32 path. Cast after ordinary FP32
  (1+gamma) RMSNorm, per-head Q/K norm, HF output-first
  projection, SwiGLU, residual, RoPE, BF16 QK matmul *and*
  scaling, post-FP32 softmax, attention value and gates.
  GDN follows BF16 conv/SILU, FP32 A_log/softplus, BF16 beta,
  FP32 recurrent state and BF16 core output, source gated RMS
  casts before gamma and at gated output. StateRoot schema
  variants disallow cross-dtype replay even for one allocation.
  Two-token four-layer synthetic BF16 whole-model integration
  plus source-independent NumPy BF16 bit-golden RMSNorm and
  gated RMSNorm, FLA two-step recurrent output/FP32 state
  oracle, sign/tie/overflows and F32 parity are required.
  This is explicit BF16 **diagnostics** only: NOT trained
  reference equivalence, a new model-run route or GPU/SSD.
  Source-independent NumPy BF16 goldens match exact bit patterns
  for ordinary/gated RMS and two-step GDN returned cores, with
  FP32 recurrent state tested within 2e-7. The full isolated
  scripts/check.sh gate passed 1,128 Rust test executions with
  0 failures and 69 intentional ignores across 238 suite runs,
  plus 18 Python fixtures, release installer/CLI and CPU numeric
  comparators. Rerunning on the final implementation still
  passes 1,128/0 across 238 suites. Both independent Qwen3.5
  .8B and 2B trained release routes retain complete 248320
  logits, all 42 StateRoot domains and source lease/mutation:
  direct16.26s and HF-import21.78s on the final source.
  Strict default/composed Clippy and Nix source-integrity pass;
  production model run is not enabled.

## Deferred — after Qwen/Gemma and Flash execution

- [ ] Qualify independently selected NanoJev-like calibrated decision
  model sources under the same model/source/cache/residency authority.
  Register finite choice/boolean/score or include/omit heads without
  assuming a specific research checkpoint. Validate calibrated
  probabilities and per-question source-bound state, batching, thresholds,
  accepted/rejected context records and decision-fallback semantics
  before enabling generic context-filter execution; pipeline check alone
  provides only typed static intent.

````

## Source: .agents/plans/multimodal/TODO.md

Original UTF-8 bytes: 7625; SHA256 `774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Multimodal

Provenance commit: `6d74f76`.
Scope: P17/P18 and the existing v5/v5.1 extensions.
Contracts: [unified execution](../../docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md),
[model assemblies](../../docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md),
[realtime QoS](../../docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md),
[implementation gates](../../research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md).

## Uncommitted

- [ ] Opt-in, source-granted context-filter outputs as bounded,
  immutable, typed hints to ASR, acoustic-first diarization,
  transcript-text and separately source-qualified speaker-preserving
  translation stages. Default only prior committed session user/assistant
  records; opt-in tool call/result, app system/developer prompts and
  explicitly authorized shared data permitted. Generic include/exclude
  and typed decision-model choices operate after source ownership checks.
  Unsupported original ASR prompt dialect refuses or uses a separately
  verified correction stage. Actual audio A/B WER/CER/DER/translation
  adequacy, timestamp/partial-output correctness, consent/authorization
  and canceled-turn isolation are required before any runtime claim.

- [ ] Model each ASR (Qwen ASR/Whisper/Canary/Parakeet),
  diarization (separately sourced speaker recognition), LM, Qwen TTS,
  Qwen Image2.1 and future video as registered typed components,
  not separate per-node subprocesses. The ASR→speaker-tagged
  transcript→LLM→TTS single coherent pipeline must include rational
  clocks, codec validity, VAD, per-turn epochs, barge-in,
  bounded fanout/backpressure and correct playback ACK frontiers.
  Current TOML/audio/image graph skeleton has NO real media runtime.
- [ ] Enable native progressive image output only at complete
  source-valid VAE/tile/frame boundaries; partial latents or
  unverified solver steps are never playable. Actual media clocks,
  atomic commit barriers, ROS/transport and tested streaming
  quality/latency require source-real gates.

- [ ] Add typed sample/frame/codebook/latent/decision values, rational clocks, checked conversions and complete source-bound component/codec/processor assemblies.
- [ ] Prove a small real non-token encoder/decision task and bounded iterative task through the existing engine before broad media claims.
- [ ] Qualify CPU-first original ASR and non-AR TTS (Whisper/Parakeet and Kokoro candidates); then actual backend execution and measured streaming/audio quality.
- [ ] Add depth-AR/RVQ speech, dual-AR codebook TTS and genuine codec-based duplex neural audio (including Qwen-Omni) where source/license/resources permit; mock codecs or text cascades do not establish native support.
- [ ] Qualify seeded source-exact image flow/editing conditioner+VAE pipelines, then video temporal/3D-VAE/expert switching and synchronized AV commit groups; candidate families remain separately gated.
- [ ] Keep ternary PTQ/Hadamard codecs, schema/calibration decision heads and specialist encoders as independent codec/model/task gates, not aliases of Qwen.
- [ ] Implement RealtimeSession lanes/epochs, bounded media buffers, VAD/barge-in, presentation/ack/playback frontiers and transactional rollback without speculative/partial media leakage.
- [ ] Add per-model transport capabilities and authorized hosted TaskService adapters with privacy/cost/idempotency/Unknown reconciliation; no duplicate local runtime or guessed realtime guarantees.
- [ ] Complete MM-01..24 plus model-specific ASR/TTS/image/video/decision quality/clock/resource suites; no real media-model support follows from source-free mocks.

## Committed

### 6d74f76

- [x] Architecture/specifications and bounded synthetic foundations exist; actual new audio/image/video/decision model execution remains unqualified.

## Tests and verification

Detailed numbered source fixtures remain in the linked acceptance/dossier owners.
Each applicable suite must pass at its actual source/device/transport tier.

- [ ] MM-D1/MM-01/MM-15: checked rational sample/frame/codec clocks, 44.1/48 kHz to 16/24 kHz and 24000/1001 or 30000/1001 FPS, long-duration remainder, discontinuity, overflow and bounded parser/buffer refusal.
- [ ] MM-D2/MM-02: complete source/processor/codec/license assemblies; wrong conditioner/shape/mask, missing source transform and cross-session mutable leakage refuse before use.
- [ ] MM-D3/MM-03..09: seeded solver/RNG and direct finite decision state, CTC/RNNT/TDT distinction, complete RVQ depth and joint AV commit; interrupted/stale device work never publishes or exposes media.
- [ ] MM-ASR-01..12: actual frontend/window/blank/predictor/duration/language/timestamp/diarization source fixtures; offline-only checkpoint refuses streaming, malicious audio bounded before allocation, sessions isolated.
- [ ] MM-TTS-01..12: non-AR phoneme/duration/vocoder and exact PCM count; incomplete 9/10 RVQ frame unplayable, codec-rate swap refused, overlap/voice generation and duplex/barge-in/state/lease boundaries correct.
- [ ] MM-IMG-01..12: source conditioner/solver/CFG/seed/mask/VAE/color and generation-entrypoint traces; invalid source stage/crop/processor fails, cancel retains old root and unknown physical handles, source/license/hosted status truthful.
- [ ] MM-VID-01..15: temporal VAE/latent crop/frame count, Wan timestep experts, source tile masks/rectangular attention, AV cross-attention/joint clocks, invalid GOP/backpressure, hot reload, cancellation and whole-assembly admission.
- [ ] MM-DEC-01..06: finite and joint schema head options/order/null/refusal/calibration against original source; generated text is not source-equivalent, external egress refused without authorization.
- [ ] MM-BON-01..08: PTQ1_0/PQ2_0 packed source scales/signs/padding plus mandatory 1024 Hadamard activation transform; wrong codec or plain-Q2 path refused, original scalar/CPU/device source parity separately qualified.
- [ ] MM-13..17/MM-19: real per-session/turn/transport epochs; stale ACK ignored, duplicate reconnect cursor idempotent/authenticated/bounded, playback progress remains uncertain when missing, slow clients do not release device leases or grow queues indefinitely.
- [ ] MM-18/MM-22/MM-23: ten speech/five video mixed jobs, fair shared physical admission and legal solver/frame yield; no per-packet catalog/router/global snapshot or silent precision/step downgrade, coupled output stays unplayable until both tracks valid.
- [ ] MM-SRV-01..12/MM-20/21/24: exact versioned SDK/protocol/field/task conformance, hosted accepted-timeout Unknown/idempotent reconciliation, callback/signature/TTL/MIME bounds, malformed media/unauthorized egress/cross-tenant replay refused before side effects.
- [ ] MM-I0 change-radius drill: small encoder/finite task and iterative mock use existing source/state/resource owners; no central router/worker rewrite or duplicate runtime; original Qwen gates stay unchanged.
- [ ] Eligible source-real CPU→physical full-model→playable-stream→soft-QoS gates: exact licensed checkpoints/processors/codecs and original quality/state oracles, real negative cancellation/admission/replay tests and measured cold/warm tails/memory. Mock/reference/region/hosted results never promote another tier.
- [ ] Resolve realtime section18 policy choices explicitly (media numeric policy, first sources, SLO/clock uncertainty, mid-turn changes, persistence, biometric/asset consent, hosted cost/egress, transport packaging and safe physical recovery). User examples do not establish released model IDs or grant execution/provider/download authority.

````

## Source: .agents/plans/numerics-and-context/CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md

Original UTF-8 bytes: 45470; SHA256 `31a2f1280da6a5df2662c9c8747fc8d795a91a8ab87a1904c33b30dfe88de9f7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Correctness first, then measured vLLM performance

Author: pvl
Co-authors: Codex, baseline_review, spine_review, batch_design_review
Owner: @pvl
Date: 2026-10-20
Source checkpoint: `fb68d8fa165de16e21d1bd37afef3790b80da971`
Status: investigation and proposed implementation plan only; implementation paused.

## 1. Decision and scope

First make Qwen3.5-0.8B, Qwen3.5-2B and original Qwen3.8-27B numerically
correct under explicit reference arithmetic. Then optimize the same engine
and demonstrate a workload-specific advantage over pinned vLLM.

Use pinned vLLM as the primary target for the new CUDA execution profile.
Treat it as correct unless an independent counterexample establishes otherwise.
Keep the accepted llama.cpp reference and Torch CPU profiles as regression
oracles. Their differing reductions and casts must not be silently combined
into an invented fourth reference.

This document records the implementation sequence and acceptance decisions.
The existing [numerics TODO](TODO.md), [model execution TODO](../model-execution/TODO.md)
and [serving TODO](../serving-and-speculation/TODO.md) remain the only owning
queues. No implementation or numerical gate is completed by writing this plan.

**Fastest route:** reproduce the earliest differing operation with identical
inputs; port its actual reference arithmetic; immediately compose its real
producer and consumer; progress to a complete layer and model. Avoid extended
standalone helper work that never reaches a real token.

## 2. What the evidence establishes

| Area | Established | Still missing |
|---|---|---|
| Small-model GGUF reference profile | Fresh mixed/Unicode corpus through 8K; both models match pinned llama.cpp AR1 and explicit `use_ref` control at the recorded checkpoints: 730 selected records, full 248,320 logits, all 42 causal domains; 55,382,016 state words/model at 8K | Default bulk arithmetic and the unrecovered historical corpus |
| Native small-model CUDA | Complete B=2/4/8 model graphs, independent row positions, CPU/B1 logits and all 42 domains within accepted source/context scope | Broader independent numeric profiles, genuine long-context GPU, temporal prefill |
| Original HF CPU | Numerous source-bound operators and isolated compositions pass; original reference captures exist | Complete trained .8B/2B/27B model composition and full-state/logit parity |
| Original 27B | Original source acquired; H48 cached recurrence, gated norm and wide-role isolated tests pass | Live producers, complete initial chunk, complete model/112 domains, admitted physical execution |
| vLLM | Genuine pinned V1 and official GGUF plugin ran both small models | Cross-engine full numerical/state parity, optimized default-route warm/concurrent comparison |
| Performance | Bounded eager C1 comparison shows a large native deficit | Numerical qualification of compared profiles and representative performance advantage |

The original assertion that complete B>1 GPU fusion is absent is superseded
by accepted P12-B7 and later work. Cross-request full-model fusion exists.
Temporal prefill still executes a full model step per prompt token.

No evidence currently proves a single defective GDN recurrence equation is
the cause of every discrepancy. “First difference observed in GDN” is not
equivalent to “GDN produced the first numerical error.”

## 3. Divergence inventory and fixes

“Proven difference” below means the cited source or executed comparison
establishes that difference. It does not automatically establish a defect in
a separately declared arithmetic profile. This is the complete currently
identified inventory, not a claim that undiscovered bugs cannot remain.

| ID | Finding and confidence | Correct layer to change | Proposed fix and decisive test |
|---|---|---|---|
| N1 | **Executed:** fresh .8B first64 default/AR1 difference begins at layer3 attention AV, position2: 530/2,048 words. Probability operands and causal V values match. This precedes layer4 GDN. | F32 attention-value reduction and temporal dispatch selection | Reproduce original AV inputs through the pinned tinyBLAS reduction; compare every output. Select that contract only for the matching bulk profile. Then rerun layer3→layer4 and the complete prefix. |
| N2 | **Executed:** fresh 2B first difference is layer0 F32 alpha projection, position0: 13/16 words; beta also differs before recurrence. Earlier compared rows and alpha operands match. | F32 projection reduction | Same bulk primitive with original alpha weights/inputs. Require exact replay, then alpha/beta→coefficients→GDN state. |
| N3 | **Executed/source identified:** default bulk enters tinyBLAS's one-Vec8 accumulator path; AR1/reference uses four Vec8 partials. Float reassociation changes results. Only first named compiled entries were observed. | Numerical contract and schedule, not a global `gdn_step` replacement | Add closed bulk schedule without changing `GgufSourceV1` defaults. Reuse the prepared implementation only after compiling and replaying it. Full default-route model closure is a separate gate. |
| N4 | **Recorded route difference:** later 2B `ssm_out` uses Q8 weight dequantization followed by F32 BLAS in one route, versus activation-Q8/integer-dot arithmetic in AR1. | Quantized projection dispatch and temporary activation format | Reproduce both routes with the same raw source, explicit dequant/activation quantization, block scaling and accumulation. Fix route selection for the declared target; an F32 alpha fix alone is insufficient. |
| N5 | **Source-proven profile difference:** original-HF diagnostic rounds convolution to BF16 before SiLU; pinned vLLM fused convolution applies SiLU to the FP32 accumulator before BF16 store. | Typed causal-convolution/activation operation | Add a vLLM CUDA contract with the reference tap order, contraction and one output cast. Keep the independently qualified Torch CPU convolution→BF16→SiLU route intact. |
| N6 | **Source-proven profile differences:** vLLM prefill Q/K normalization stores BF16; its fused beta path stores F32. Current diagnostic keeps normalized Q/K in F32 and rounds beta to BF16. | Post-convolution normalization/coefficient preparation | Explicit phase-specific dtypes; compare Q/K, beta, log-decay and state using identical inputs. Do not change the Torch CPU target, where BF16 beta is intentional. |
| N7 | **Source-proven profile difference and diagnostic defect:** vLLM gated RMS keeps arithmetic F32 through gamma and SiLU until output store. The old diagnostic also rounds the weighted product even when raw F32 gamma should promote the CPU product to F32 (`original_qwen_precision.rs:140–143`). | Gated norm, with gamma dtype retained | Port selected vLLM norm; replace the diagnostic's unconditional weighted-product cast with the already qualified dtype-specific CPU contract when composing that CPU target. Keep these fixes separate. |
| N8 | **Executed/source-proven:** original HF initial chunk and cached step use different algorithms. Cached CPU query divides by rounded sqrt(128); initial CPU query multiplies rounded inverse sqrt. CPU initial cumsum has F64 carry/F32 stores. | Phase-aware GDN core, scale and prefix operations | Implement and select initial/continued-prefill/cached-decode contracts explicitly. A cached step on zero state is not an initial chunk qualification. |
| N9 | **Unfinished producer:** authentic recurrence replays import decay and other intermediates. Standalone Torch CPU `exp` uses a different provider from SLEEF expressions embedded in sigmoid/SiLU. | Coefficient producer and native math admission | Connect live `A_log→exp→negative`, alpha/dt/softplus, log-decay and decay; use the actual profile's exp implementation. Never substitute host `exp` or recorded lookup values. |
| N10 | **Executed/source-proven:** official BF16 loading and restoration of original F32 tensors produce different outputs. Accepted GGUF projections prepare Q8_0/Q8_1/Q8_K activations by codec; the stock plugin uses its own Q8_1 path. Storage equality does not imply matching activation/cast/reduction schedules. | Source binding, loader, projection and numerical profile identity | Pin effective loaded dtype and activation quantizer by role. Separate official-loaded, mixed-source, GGUF-AR1 and vLLM CUDA comparisons. Authentic HF cuBLAS operands do not qualify the GGUF-plugin projection. |
| N11 | **Integration gap:** isolated correct operators do not constitute a correct model. Ordinary norm, attention, residuals, MLP and final head can introduce the next difference after GDN. | Existing complete ForwardPlan and HostMath/resident backends | Trace in causal order and compose real producers. Check every layer's residual, complete logits and every state domain, then test the next teacher token. |
| N12 | **Geometry/layout risk, not newly demonstrated corruption:** 27B has 16 key heads→48 value heads, source/canonical state orientation differences, four-frame source conv cache versus canonical newest-three history. | Source adapter and typed layout/state codec | Explicit repeat-interleave, axis transpose and history mapping; asymmetric per-head fixtures and genuine 27B captures. No reinterpretation based merely on equal byte count. |
| N13 | **Evidence gap:** original historical 8K raw token corpus is unrecovered. Fresh exact 8K results cannot retrospectively replay it. | Reference corpus recovery/reporting | Make one bounded recovery attempt from retained original inputs; otherwise retain the historical item as unreproduced and close only the explicitly named fresh corpus. |
| N14 | **Measurement gap:** stock plain/observed exact logits prove observer transparency, not native/stock parity. Matching text or argmax hides large tensor differences. | Comparator and qualification boundary | Require independent full-vocabulary and canonical state comparison; bind benchmark validity to that qualification. |
| N15 | **Source-proven loader transform:** plugin reconstructs `A_log=log(-ssm_a)` from GGUF negative exponentiated decay; later exp need not recover identical stored bits. Ordinary norm subtracts one; GDN gamma is excluded. Grouped heads also require explicit permutations. | Effective loaded-parameter adapter, before GDN | Compare actual loaded parameters and faithfully reproduce the selected loader. Preserve raw source identity and derived representation separately; do not “fix” this with extra negation or in-place weight mutation. |

No new evidence justifies changing RoPE, global signed-zero treatment, the
accepted GGUF recurrence, or tolerances as a blanket “GDN fix.”

## 4. Numerical contracts

### 4.1 Keep four targets explicit

| Target | Purpose | Required identity |
|---|---|---|
| Accepted GGUF AR1/reference | Preserve working CPU/CUDA behavior and independently exact fresh8K results | Original GGUF, existing reduction/quant policy, F32 KV, FA off in reference |
| Pinned llama.cpp default bulk | Explain and reproduce the default comparison discrepancy | `d812350`, actual CPU/BLAS route, batch/ubatch, strides, thread count, temporary quant format |
| Pinned Torch CPU original HF | Independent trained-source oracle and existing CPU operator work | Effective loaded tensor dtypes, Torch/provider versions, vector width/tails, initial/cached schedule |
| Pinned vLLM CUDA | Primary new correctness and performance target | vLLM, plugin, source/loader, CUDA kernels, phase, activation/state dtype, compiler/math modes |

Extend `ProgramNumericalProfile` and typed operator contracts in the existing
core; the proposed vLLM and bulk profiles do not exist in MAIN today.
Profile identity belongs in compiled-plan/cache keys and state compatibility.
Reject incompatible reuse rather than choosing a dtype from a display name.

### 4.2 Mathematical GDN definition

For each value head, let `S[k,v]` be its FP32 recurrent matrix, with independent
convolution history and causal frontier. Original 27B uses 128 key/value
dimensions and repeats each of 16 Q/K heads into three adjacent value heads.
Converted GGUF layouts require their own verified mapping.
This mathematical notation is the transpose of vLLM's actual contiguous-key
`[head,value,key]` cache; it is not permission to reinterpret the bytes.

The semantic recurrence is:

```text
u_t = depthwise_conv4(project_qkv(x_t), prior_history)
(q_t, k_t, v_t) = split(SiLU(u_t))
qhat = q_t / sqrt(sum(q_t*q_t) + eps_l2)
khat = k_t / sqrt(sum(k_t*k_t) + eps_l2)
qscaled = qhat / sqrt(key_dim)
beta = sigmoid(project_b(x_t))
g = -exp(A_log) * softplus(project_a(x_t) + dt_bias)
d = exp(g)
Sdecayed[k,v] = d * Sprevious[k,v]
prediction[v] = sum_k(khat[k] * Sdecayed[k,v])
delta[v] = beta * (v_t[v] - prediction[v])
Snext[k,v] = Sdecayed[k,v] + khat[k] * delta[v]
core[v] = sum_k(qscaled[k] * Snext[k,v])
gated = RMS(core, eps_norm) * gamma * SiLU(project_z(x_t))
output = project_out(gated)
```

These equations specify algebra, not floating-point instruction order.
For the pinned target, the required implementation also includes each cast,
reduction tree, FMA boundary, exponential approximation and layout.
Pinned vLLM Q/K L2 epsilon is fixed `1e-6`; gated RMS uses
`config.rms_norm_eps`. These are separate fields.
Do not reuse GGUF grouped-RMS lowering as proof of the original FLA algorithm.

### 4.3 Exactness rules

- BF16 stores: round-to-nearest, ties-to-even at the reference boundary;
  preserve signed zero. Widening BF16 to F32 is exact and is not an extra
  rounding step.
- Raw learned parameters retain their actual dtype and source role. Ordinary
  Qwen RMS's `1 + weight` semantics do not apply to GDN's raw learned gamma.
- FP32 state remains FP32; BF16 core/output does not permit BF16 recurrent
  storage. Preserve source accumulation order and every materialized boundary.
- Source FMA is one rounding; separate multiply/add is two. Do not enable
  blanket fast-math or silently contract a CPU tensor expression.
- Padding is masked according to the reference. An intentional `-Inf`
  triangular mask is valid only inside that closed operator; it does not
  weaken general finite-value validation. Do not multiply poisoned future
  values by zero and assume the result is harmless.
- Pin denormal/FTZ behavior, exponential/reciprocal implementation, matrix
  compute/output types and TF32 settings. A cuBLAS API's “FP32 accumulation”
  does not reveal its reduction tree.
- Kernel source alone cannot establish the exact generated CUDA instruction
  stream. Where bit identity is promised, replay actual compiled kernels on
  authentic operands. Record unresolved instruction/provider details as
  unresolved rather than filling them in from algebra.

### 4.4 Initial prefill versus decode

Use the actual selected pinned vLLM kernel, not a kernel guessed from an
environment-variable name. On SM86 the inspected automatic prefill branch
selects Triton. With packed decode enabled, ordinary non-speculative decode
selects `_forward_core_decode_non_spec` and Triton packed recurrence.
`GDN_DECODE_KERNEL=cuda` alone does not prove the guarded C++ MTP path ran.
Confirm branch/environment at runtime when implementation resumes.

**Normal packed decode, all temporaries F32:** widen conv output; divide Q/K
by `sqrt(sum(square)+1e-6)`; multiply Q by `K**-0.5` before readout; compute
`x=a+dt_bias`, `softplus=select(x<=20,log(1+exp(x)),x)`,
`g=-exp(A_log)*softplus`, `beta=sigmoid(b)`; decay state, subtract its K dot
from V, multiply residual by beta, update state and reduce against scaled Q.
Store core BF16 and recurrent state in its explicitly configured FP32 cache.
The state-decay exp alias is `tl.exp` by default and `fast_expf` under
`FLA_USE_FAST_OPS=1`; coefficient exp/log/sigmoid are direct Triton operations.
Preserve the selected setting and validate generated arithmetic.

**CUDA initial/continued prefill, chunk64:**

1. Conv FP32 accumulation → `acc/(1+exp(-acc))` → BF16 store.
   Normalize Q/K in F32 using reciprocal sqrt, multiply, then store BF16.
   V remains BF16; beta and log-gates are F32.
2. Prefill softplus uses the stable positive branch
   `x+log(1+exp(-x))`, nonpositive `log(1+exp(x))`, then threshold20.
   This differs from the decode expression; do not collapse the two.
3. Compute chunk-local F32 prefix log-gates `G=cumsum(g)`.
   Form the strict-lower system
   `A[i,j]=dot(beta[i]*k[i],k[j])*exp(G[i]-G[j])` in the source graph.
   Beta multiplies the operand before the dot; moving it outside changes
   rounding. Keep the promoted operand dtype selected by the CUDA branch.
4. `solve_tril` forms `(I+A)^-1`, including its source negation and unit
   diagonal. Preserve the 16×16 solve and block-merge order rather than
   substituting a generic inverse. It returns **BF16 K dtype**.
   Pin `FLA_TRIL_PRECISION` (source default `ieee`) and actual dot policies.
   The WY construction casts scaled V/K operands to BF16 before its dots;
   `u` and `w` outputs are BF16.
5. Running state stays F32 `[head,value,key]`; each chunk's saved entering
   state `h` is BF16. Compute `v_new=u-w@h` and store BF16. Apply chunk decay
   to F32 state; cast the decay-adjusted delta BF16 before its K/delta dot.
   Final recurrent state is F32.
6. Output uses BF16 Q/K/h/v_new dots with F32 accumulators and causal masks;
   gate-adjusted attention coefficients are cast BF16 before the V dot.
   Scale both output terms, add and store BF16.
7. Gated RMS widens the BF16 core, computes norm/gamma/SiLU product in F32,
   then stores once to BF16. No CPU intermediate BF16 gamma-product cast.

This is a mixed-precision chunk/WY algorithm. Keep sequential recurrence as
an independent semantic oracle; it is not an instruction-exact substitute.
Reproduce actual reductions, masks and boundary state from the pinned kernels.
No newly executed vLLM→Rust operator comparison is asserted by this table.

The existing Torch CPU target has a distinct known schedule:

1. Widen and normalize Q/K; initial query multiplies F32 `0x3db504f3`.
   Cached query divides by rounded F32 sqrt(128).
2. Initial short input pads to64; materialize beta*K and beta*V.
3. Prefix scan uses F64 carry with an F32 store at each element.
4. Form masked prefix differences, exponentials, Gram products and
   unit-lower solves; retain materialized intermediate boundaries.
5. Preserve the six recorded BMM families and two solve groups, column-major
   solve outputs and final valid-row extraction. State remains FP32.
6. Cached update uses materialized products/reductions, separate subtraction,
   multiplication and addition. Its tested state uses key-major orientation.

This CPU schedule must not be copied into vLLM's Triton path merely because
both implement GDN. Conversely, vLLM's fused casts must not replace the
already qualified CPU profile.

### 4.5 Small-model default bulk arithmetic

The minimal existing candidate implements one eight-lane FP32 FMA accumulator
per output, processing K in eight-element groups, including the source's
zero-filled tail operations. Its horizontal sum follows the pinned low/high
lane additions, then `(low0 + low2) + (low1 + low3)`. AR1's four partial
accumulators remain a separate contract.

Validate the original strided AV and F32 alpha captures, together covering
132,096 output words. Then advance through downstream routes, including
Q8/dequant/BLAS. The prepared `root-default-bulk-20261020` worktree is
uncompiled and unqualified; its existence is not a fix.

## 5. Porting strategy and implementation boundaries

Port small mathematical operations and the reference's phase/layout logic.
Reuse the existing `MathematicalBackend`, `ForwardPlan`, source descriptors,
`HostMath`, resident GPU queue, `ResourceArbiter` and `SessionAuthority`.

| Existing area | Intended change after implementation resumes |
|---|---|
| `zerogravity-core/src/accelerator/program/typed.rs` and typed children | Explicit profile, phase, dtype and layout contracts; only qualified operations become selectable |
| `zerogravity-cpu/src/original_hf/` and `src/math/` | Independent semantic/source-profile reference operations; preserve accepted CPU schedules |
| `zerogravity-model-qwen35/src/forward.rs`, `host_math/`, source bindings | Compose actual trained producers in the existing plan; preserve source-derived geometry and state effects |
| `zerogravity-gpu/src/resident_program.rs` and operator modules | vLLM-profile CUDA operations and temporal kernels through the original queue |
| `resident_inference.rs`, `resident_batch.rs` | Phase-aware temporal prefill, independent rows, state/currentness and physical completion |
| `zerogravity/src/original_qwen_math.rs` | Keep diagnostic labeling explicit; do not promote it to a second production engine |

Do not port vLLM's Python model/session engine or allocator. If a provider
kernel is reused directly, it must fit the original physical admission,
buffer lifetime, event and quarantine model. A Python subprocess is an
independent oracle, not a production inference fallback.

Preserve upstream license/SPDX notices and exact source revision when porting.
Prefer the actual selected upstream kernel over a convenient unused helper.
No new broad plugin framework is needed for a handful of typed operations.

## 6. Fast implementation sequence, when resumed

### C0 — Lock the comparison, not another framework

Use existing raw teacher IDs, retained captures and pinned runtimes.
For each run record the model/source, effective loaded dtypes, phase,
chunk size, state layout and operation route in the existing test output.
Reuse existing process ownership and reference tools.

Choose the lane explicitly. Recommended primary lane: original .8B HF with
vLLM's actual loaded GPU dtypes, avoiding the extra GGUF log/exp and temporary
quantization transformations during the first GDN diagnosis. Use that same
profile for 2B/27B when admitted. Keep the existing GGUF-plugin benchmark lane
separate; it must reproduce its own loader/quantizer to qualify numerically.
Neither lane's pure operator captures may be relabeled as the other's.

For .8B, begin with teachers11/353 and the first64 corpus. Capture projected
QKV/Z/a/b before GDN to determine whether a projection mismatch already
precedes the known cast differences. Do not assume the diagnostic matches
vLLM at its GDN inputs.
Bind actual phase and partition metadata: teacher353 must really take the
claimed cached/continued route. Repeat the exact reference partition in each
comparison; different chunkings or GEMM shapes are not automatically bit-equal
because they share a profile name.

Exit: the first differing operation has identical compared inputs and source
weights, or the earlier source/loader/projection mismatch is precisely located.

### C1 — Close the earliest small-model default discrepancies

Compile and test the existing bulk primitive; replay AV and alpha.
Compose real producer→operator→consumer and run the same first64 tokens.
Follow the next actual difference, including beta and Q8/BLAS output routes.

Keep this a bounded CPU side task while the vLLM CUDA profile is developed.
Do not delay the primary vLLM route on recovering an unavailable historical
file or on completing an unrelated generic arithmetic framework.

Exit: exact authentic leaf replays and a declared complete default schedule
comparison; accepted AR1 results unchanged. Historical closure remains
separate if its raw inputs cannot be recovered.

### C2 — Complete one vLLM-profile GDN layer

After projection parity, implement convolution+SiLU → phase-specific Q/K
normalization/replication → coefficients. Qualify the smaller cached core
and gated RMS first using authentic nonzero prior state, then implement
initial/continued-prefill chunk64 and connect initial→cached continuation
plus output projection. Start with .8B, then use 2B and H48/27B for geometry.
Completing the separate Torch CPU MKL/BLAS provider is not a prerequisite
for the vLLM CUDA port.

For every fix require a failing-before/passing-after genuine-operand case.
Compare full intermediates and state, not just one output element.
Connect each accepted operation immediately; imported reference operands may
isolate a bug but cannot replace the native producer in the composed gate.

Exit: initial layer output/state followed by genuine cached continuation
matches the target; no reference intermediate is injected into the final test.

### C3 — Complete .8B, then 2B in the same plan

Compose all layers: embedding, norms, full attention, GDN, residuals, MLP,
final norm and head. Trace the first new mismatch rather than repeatedly
dumping every later error. Inspect the full 248,320 logits and all42 domains.

First B1, then B2/4/8, ragged positions, refill and B1↔batch transitions.
Do not carry an assumption that one-token and bulk GEMM use identical
reductions; qualify the actual selected algorithms/shapes.
The source-correct safe chunk path is itself part of correctness; its initial
implementation need not yet be the fastest temporal-prefill implementation.

Exit: small-model full correctness matrix below, including initial→decode
and whole-prefill→chunked-prefill continuations.

### C4 — Complete original27B without relaxing resource guards

Use the same recipe/operations with source-derived geometry: 64 layers,
48 GDN/16 full-attention layers, 112 state domains, hidden5120,
MLP17408, vocabulary248320 and an untied head.

Reuse verified source windows and isolated H48/wide-role work, including the
already tested full17408-column CPU down-projection role. Finish live
coefficients, initial chunk and trained producer/full-model composition.
The corresponding native GPU wide-column path remains unqualified;
geometry refusal must not be removed without the corresponding arithmetic.

Original BF16 weights are about55.56GB. They cannot fit the local6GiB GPU;
current executable-source and GPU-width guards also remain. First establish
complete CPU/reference execution on an admitted capable host or correctly
accounted source-window path. For GPU serving choose either a capable
matched resident device or complete physical RAM/SSD/GPU streaming.
Compare vLLM under the same capacity/offload constraints; an unsupported
comparison is reported as unavailable.

Exit: original27B full logits/all112 domains and continuation qualify.
Operator-only success, acquired metadata and a source window are not this gate.

### C5 — Accept and checkpoint

Run focused regressions, relevant strict Clippy, physical CUDA/state/cancel
tests, both approved original cache-direct/HF-import release routes and:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh
```

Use one MAIN Cargo/GPU coordinator; independent agents own disjoint source
cuts/reviews. Preserve root README, unrelated overlays and the Renoir/gfx90c
launch hold. Review and sign only owned verified changes. Reconcile existing
plan TODOs; no extra status ledger or source-inventory framework.

## 7. Numerical parity matrix and acceptance

| Axis | Small first gate | Expansion required for the claim |
|---|---|---|
| Sources | .8B original GGUF and official HF, explicitly separate | 2B equivalents; original27B, then individually qualified quant profiles |
| Teacher inputs | Tokens11/353; retained mixed/Unicode first64 | Fixed raw IDs for prose, code, Unicode, repeats, zeros/extreme operator inputs; held-out prompts |
| Phase boundaries | Empty→first prefill→cached step | 1/2/31/32/63/64/65/127/128/129 tokens; ragged chunks and continued prefill |
| Context | Short complete model | 512/1024/2048/4096/8192; 16K/32K only after actual capacity and predecessor gates |
| Batch | B1 | B2/4/8, independent positions, refill, row reorder, canceled neighbor |
| Outputs | Full affected tensor/state | Every layer residual, full248320 logits, all42 or112 domains and next-token continuation |
| State lifecycle | Initial and cached | Snapshot/restore, fork/COW, cancellation before/after submission, failure/quarantine, final reclaim |

Required distinguishing tests, using independently generated expected values:

- Conv sums straddling a BF16 rounding boundary before SiLU; one-store CUDA
  and two-store CPU results must remain distinguishable.
- Nonzero beta that changes under premature BF16 rounding; positive/negative
  softplus inputs and values on either side of20, separately for both phases.
- Zero/tiny/nontrivial Q/K sums; prefill BF16 versus cached F32 normalized
  vectors and fixed L2 epsilon independent of model norm epsilon.
- Asymmetric `HK=2,HV=6` sentinels proving `[0,0,0,1,1,1]` grouping; genuine
  16→48 source/canonical permutation and quantized output-layout checks.
- Nonuniform signed prior state and Q exposing transpose and scale-before/
  scale-after reduction; a zero-state-only fixture would miss both.
- F32 and BF16 raw gamma cases distinguishing promotion from intermediate
  BF16 rounding; compare the relevant CPU and CUDA targets separately.
- Length63/64/65 with nonzero initial state and invalid padded lanes;
  compare solved-A, u/w, saved h, v_new, final state and complete next output.

Three separate verdicts are required:

1. **Same source and same arithmetic contract:** exact bits for promised
   leaf replays, deterministic internal CPU/B1/batch state/logits and integer
   layouts. Zero-sign differences are reported, not silently normalized.
   Phase, shape, partition and selected kernel are part of that contract;
   cross-partition algorithm changes use a separately qualified comparison.
2. **Independent implementations with deliberately different arithmetic:**
   report max absolute/relative error, RMS/relative-L2, ULP distribution,
   full-vocabulary distribution metrics, top-k/rank and per-domain drift.
   Preserve existing numerical limits: historical functional shared-top20
   logprob0.45 and top-token probability0.05 are only those specific tests,
   not permission for arbitrary state error or a new universal tolerance.
3. **Model behavior:** teacher-forced loss/perplexity, continuation and task
   quality on held-out checkpoint-matched inputs; stochastic policy and
   sampling distribution separately when advertised.

There is no existing universal BF16 state/logit tolerance that can honestly
be asserted to qualify all three models against every vLLM kernel.
For a newly different arithmetic route, define its per-operator/state/error
budget from the pinned reference and sensitivity study **before evaluating
the candidate for promotion**; record it in the existing numerical contract.
Until that budget exists and passes, it is unqualified. Do not choose bounds
after seeing the candidate error or relabel a failed exact test as approximate.

For the fastest initial implementation, retain the target's own casts and
operation order and aim for exact genuine-operand replays. This minimizes
the need to justify new approximation budgets. A cuBLAS reduction difference
must still be investigated rather than hidden by the high-level equations.

## 8. Why performance is currently far behind

The actual eager first-request-after-readiness comparison measured median
TTFT13.400s versus1.561s for .8B and14.540s versus1.562s for2B.
The2B individual-token delivery p95 was approximately0.186–0.199s native
versus0.026–0.030s stock. This is a real observed deficit in that bounded
configuration, not evidence of numerical equivalence or every workload.
Separate native warm experiments show substantially lower C1 latency.
Their difference from the cold experiment is not a measured JIT duration:
profile lazy compilation, initialization and allocations explicitly.

Concrete causes/limitations to address:

- `resident_inference.rs` still iterates prompt tokens and awaits a complete
  model execution for each. Cross-request batching does not supply a
  token-matrix prefill or a chunk-parallel GDN algorithm.
- The canonical .8B C1 window launched236,270 native kernels for54 model
  steps. Native launch coverage was verified. Stock recorded9,318 kernel
  activities, but complete/lossless stock coverage was not established;
  do not claim an exact hardware-count ratio.
- Source launch attribution includes41,148 SiLU launches,58,144
  certification/merge launches and20,768 RMS launches. These are launch
  counts, not measured percentages of runtime.
- The latest accepted C8 change cut source dispatch counts by about18% but
  improved observed median raw token rates only about1.7%; C1 barely changed.
  Fewer launches alone are a poor predictor of the remaining speedup.
- Both compared routes were eager. Missing native production CUDA graphs
  cannot explain this particular eager-versus-eager deficit by itself.
- Tiny step projections, reduction kernels, synchronization, allocations and
  state movement are likely important; rank them by actual stage/device
  timing and bytes, not by kernel count alone.

There is no basis for saying all vLLM optimizations are already implemented.
Existing source/state safety and fusion foundations do not guarantee faster
execution. Correctness and a Rust host implementation alone imply no speedup.

## 9. Performance sequence after correctness

The order below targets structural work before local tuning. A bounded,
already-prepared fusion can land earlier if it proves a useful gain without
delaying the correctness or temporal-prefill path.
If measured host gaps dominate, graph admission may proceed alongside
projection work; the priorities are evidence-driven, not a fixed calendar.

| Priority | Change in shared machinery | Why / measurement | Required guard |
|---|---|---|---|
| P0 | Establish accepted numeric-profile baseline, phase timing, device events, bytes, CPU submission gaps and physical counts | Identify prefill/decode bottlenecks and separate GPU work from launch/host waits | Unprofiled paired timings remain the performance result; instrumented outputs match plain outputs |
| P1 | True temporal prefill: token-matrix projections/MLP, causal attention over a token block, chunked GDN, one bounded state transaction | Replace O(prompt tokens) host-driven full model steps; target TTFT first | Same-profile chunk/step parity or explicitly qualified chunk contract; no intermediate token publication |
| P2 | Efficient projection kernels/provider path, appropriate GEMV for decode and tensor-core GEMM for token/batch matrices | Improve actual arithmetic/memory throughput, not merely row count | Source codec/effective activation/output types and compute accumulation fixed; opaque workspace physically admitted |
| P3 | Fuse convolution+SiLU, Q/K normalize+coefficient preparation, gated RMS, residual+norm and SwiGLU where the profile permits | Reduce intermediate writes, launches and global synchronization | Preserve reference casts/reductions, finite checks and meaningful failure semantics |
| P4 | Reuse shape-bound scratch/commands; remove redundant scans/copies by constructing certified outputs and combining checks | Reduce certification, allocator and state-transport overhead | Retain required checks; one original authority, correct unknown-completion quarantine, no lease reuse before completion |
| P5 | Finish production CUDA graph admission and bounded decode graph replay | Reduce host overhead after the kernel structure is stable | Account for graph metadata, captured pointers/generations, all allocations/events and lifetime through retirement |
| P6 | Joint chunked prefill/decode scheduler, ragged refill, bounded overlap and per-tenant aging | Raise SLO goodput while controlling TTFT/ITL tails | Actual independent session/state/event ownership; no starvation or hidden dropped requests |
| P7 | Paged hybrid KV/GDN cache and exact complete-prefix reuse | Lower capacity/copy cost and exploit repeated work | Include recurrent/conv state, source/profile/tenant identity and causal frontier; cold no-hit baseline retained |
| P8 | Source-verified RAM/SSD/GPU streaming for capacity-bound27B | Enable the actual large model and overlap useful transfers | Stable source extents, bounded staging, true Ready events and physical resource accounting |
| P9 | Shape specialization and measured tuning of qualified kernels | Pursue a specific-device advantage after basic throughput is competitive | Offline bounded choices, held-out workload, no unbounded per-token JIT/search |
| P10 | Speculation/prefix combinations only after target execution is competitive | Additional useful committed tokens when acceptance pays for overhead | Exact target distribution/state rollback and full cost of rejected work; target-only comparison |

P1 has three distinct cuts: sequential T2/T4 amortizes gather/scatter,
readback and transaction overhead while repeating the mathematical body;
true temporal stateless lowering combines real teacher rows (including
solo B1/T2) across projections/MLP while respecting causal regions; the
fully qualified chunk attention/GDN path supplies larger temporal parallelism.
Measure each separately. A sequential loop is not token-matrix fusion.

For P2, preserve exact GGUF behavior by first investigating tiled/coalesced
weight loads, shared activation preparation and integer DP4A work within
the existing codec/block/lane reductions. Keep scale addition order and
overflow bounds. BF16 tensor-core GEMM belongs to the independently qualified
vLLM GPU profile; it is not a drop-in replacement for CPU-exact GGUF math.

Do not begin with speculative decoding to hide a slow or incorrect target.
Do not quantize the model/cache more aggressively and call the resulting
quality change a free speed improvement. Do not remove state isolation to
improve a benchmark.

Potential advantages over vLLM are hypotheses: lower host overhead on this
small GPU, efficient small-shape quantized kernels, fused state operations,
and workload-aware hybrid caching/scheduling. Promote only measured wins.

## 10. Two benchmark deliverables

### A. Model execution benchmark

Separate pure prefill and decode after a fixed prefix. Report model/source/
numeric profile, prompt length, teacher-input tokens/s separately from
generated/committed output tokens/s, per-step
latency, physical memory, kernel counts, transferred bytes and device time.
No HTTP/network overhead is attributed to kernels. A pure operator
microbenchmark remains a diagnostic, not the model result.

Preserve canonical47-input/8-output continuity. First admitted GPU matrix:
.8B and2B, B1/2/4/8, prompt lengths1/32/47/128/448 and output lengths8/64,
subject to the actual consumed-token frontier fitting capacity512.
Never run512 prompt plus32/128 generated tokens against that capacity or
silently truncate. After real capacity/numerical qualification, expand to
512/2048/8192 contexts and32/128 outputs within the enlarged envelope.
Cold-first-request and steady warm remain separate.
Add original27B only on a physically admitted, equivalent placement. Report
capacity-enabling streaming separately from resident performance.

### B. OpenAI serving benchmark

Use [serve-bench.py](../../../tools/serve-bench.py) and
[serve-compare.py](../../../tools/serve-compare.py). Preserve their existing
owned-provider lifecycle and alternating ABBA/BAAB sampling.

- Same exact source/quant, effective loader/numeric policy, tokenizer/template,
  raw IDs, generation/EOS/stop/sampling rules and output budget.
- Same device/driver, resource caps, power/thermal conditions and comparable
  residency/cache state; serialized arms on the shared GPU.
- Report startup separately from first request after readiness. Separate
  storage-cache cold, process cold, warmed model and prefix-cache hit/miss.
  One B1 warmup does not warm every B8/JIT/graph shape; declare an additional
  shape-warmed protocol symmetrically when that is what is measured.
- Compare eager routes for diagnosis and actual optimized/default vLLM V1
  routes for the competitive claim. Pin graph, prefix cache, chunked prefill,
  token/sequence budgets and attention/GDN backends.
- Start concurrency1/2/4/8 within current tool bounds. Use fixed open-loop
  arrival traces, short/long mixes and multiple tenants; sweep offered load
  through saturation. Expand concurrency only after admission/tool support.
  Workload labels alone measure class balance; authenticated tenant fairness
  additionally requires the actual tenant identity/admission boundary.
- Report offered/admitted/rejected/canceled/timed-out/completed requests,
  committed correct tokens, p50/p95/p99 TTFT and token ITL, SLO goodput,
  per-tenant completion/latency, memory and complete device counts.
- SSE chunk spacing is not individual-token ITL. When delivery coalesces,
  use authenticated token commit events for engine ITL; report client delivery
  independently and leave unavailable token-level measurements unset.

Current `serve-compare.py` deliberately leaves qualified source-correct
SLO-goodput null and equivalence `NOT_ESTABLISHED`; it also does not estimate
confidence intervals. The minimal follow-on is to bind an independently
accepted numerical qualification and physical commit evidence, then compute
paired statistics from retained raw runs. Merely supplying equal hashes or
matching generated text must not unlock that result.

Before measurement freeze workload/SLO/quality conditions. Retain the existing
10s TTFT/2s delivery-ITL workload as a historical comparator; define practical
interactive and long-prefill SLO classes separately before running them.
Use at least two balanced ABBA/BAAB cycles for screening. For promotion,
predeclare a larger fixed repeat budget, analyze independent run-level paired
ratios and95% confidence intervals, and do not treat correlated token samples
as independent repetitions.

**Win condition:** on a named qualified workload/hardware/configuration,
the95% confidence interval for correct committed SLO-goodput ratio is above1,
with no regression beyond predeclared latency, quality, rejection, completion
or fairness budgets. Report neutral/loss cases alongside wins. No engine-wide
“faster than vLLM” claim follows from one successful prompt or model.
If baseline goodput is zero, a ratio is undefined: use an absolute goodput
difference and its confidence interval, never an infinite-speedup claim.

## 11. Stop conditions and immediate next work

Stop promotion at the first incorrect tensor, incomplete state mapping,
unadmitted resource, unknown physical completion or missing independent
oracle. Continue diagnosis at that operation; do not advance a claim.

When the owner resumes implementation, the first concrete work is:

1. Revalidate paused overlays and accepted artifacts; use the existing bulk
   worktree for the bounded CPU reproduction.
2. Establish actual .8B vLLM projection→convolution inputs and select the
   first complete GDN layer's pinned phase/backend.
3. Port the smallest differing operation with genuine before/after evidence,
   immediately compose it, then walk forward to complete layer/state parity.
4. Expand to complete .8B,2B,27B and the parity matrix.
5. Implement temporal prefill and efficient projections before claiming that
   small launch reductions close the observed throughput gap.

The paused ordered-SiLU candidate remains unaccepted until its model/physical
gates pass. Do not use `target/release/zerogravity` as an accepted baseline
without checking which build produced it.

## 12. References and evidence boundaries

- [Numerical contract](../../docs/NUMERICS_QUALITY_AND_BENCHMARKS.md).
- [Independent llama.cpp runbook](../../../tests/reference/llama_cpp/README.md).
- [Joint serving/goodput contract](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md).
- [36 serving acceptance gates](../../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md).
- [Accepted batching and performance evidence](../serving-and-speculation/FUSED_GPU_BATCHING.md).
- Existing pinned runtime: vLLM0.31+cu129
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`, Torch2.13+cu129
  `cf30153c4…`, official GGUF plugin
  `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`; retained image
  `sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe`.
  Exact local runtime source is authoritative for this comparison; current
  online documentation is not a substitute for those pinned bytes.
- Pinned vLLM source paths: `model_executor/models/qwen3_5.py`;
  `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`;
  `model_executor/layers/mamba/ops/causal_conv1d.py`;
  `third_party/flash_linear_attention/ops/{fused_recurrent,
  fused_gdn_prefill_post_conv,chunk,chunk_scaled_dot_kkt,solve_tril,wy_fast,
  chunk_delta_h,chunk_o,layernorm_guard,op}.py`.
  Relevant source ranges: packed recurrence317–340; prefill prep78–149;
  gated norm119–176. Preserve Apache-2.0 and vendored FLA MIT notices
  (Songlin Yang, Yu Zhang) for later ports.
- Pinned plugin source: `weights_adapter/qwen3_5.py` and
  `quantization/layout.py`; loaded-role transforms are part of the comparison.
- Original27B revision:
  `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Numerical evidence and current open work are in the owning TODOs linked
  above. Run directories are supporting diagnostics, not additional queues.

This planning pass changes documentation only. It adds no runtime acceptance,
new benchmark result, implemented arithmetic profile or fixed-bug claim.

````

## Source: .agents/plans/numerics-and-context/TODO.md

Original UTF-8 bytes: 10220; SHA256 `f73cde3e405495eb3e33f7045d093fea51c797979bcaf82f462302d5fa17d4d7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Numerics and context

Provenance commit: `6d74f76`.
Scope: P02/P11/P22, independently of optimization.
Contracts: [numerical/source profile](../../docs/NUMERICS_QUALITY_AND_BENCHMARKS.md),
[independent reference runbook](../../../tests/reference/llama_cpp/README.md).

Planning-only continuation (20 Oct): [correctness fixes, arithmetic contracts,
three-model parity and subsequent vLLM performance plan](CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md).
Implementation is paused at the owner's request; this design adds no completed
numerical or performance gate. Existing owning TODOs remain authoritative.

## Uncommitted

- [ ] **Required three-model correctness gate (owner,20 Oct):** resolve
  independent original-source CPU8K GDN divergence for BOTH original
  Qwen3.5 models without loosening tolerances or changing declared source
  arithmetic. This is required alongside original Qwen3.8-27B correctness,
  matched serving/model benchmarks and shared-engine performance work.
  Short-context CPU/B1/native exactness and closed original-HF operator
  proofs do not close this independent GGUF long-context issue. Recover
  exact raw teacher prefixes and pinned native arithmetic/chunk settings,
  locate the first state/stage discrepancy, then rerun independent long
  context and the accepted short corpus after any implementation fix.
- [ ] Extend genuine resident GPU context/corpus qualification beyond the currently accepted 512 consumed tokens, including capacity/window/padding/replay boundaries.
- [ ] Run genuine whole-model 16K/32K CPU/GPU gates only after prerequisites; functional 8K success is not independent long-context numerical closure.
- [ ] Keep internal full-logit/all-state/trace bit parity distinct from independent native-source stage/top20 comparisons, including the known 2B 56 signed-zero distinctions.
- [ ] Broaden prompts/tokenizers/precision profiles and sampling quality only with independent oracles, exact history and explicit qualified ranges.

- [ ] Diagnose identical teacher-forced prefixes512/1024/2048/4096/8192 against pinned llama.cpp0.6.0/build11429 `d812350`, CPU/F32 KV and Flash Attention off. Fix the physical microbatch/chunk/arithmetic schedule and compare per-head GDN state, gain/beta, quantization and AR-versus-bulk recurrence; distinguish replay correctness from arithmetic.
- [ ] Compare native ubatch1/64/256/512 only within bounded feasible runs; the historical full ubatch1 capture did not finish. Use raw token IDs to avoid incomplete UTF-8 HTTP rows. Recheck short corpus, layers, state and actual8K after any fix; retain original functional0.45 shared-top20-logprob/0.05 top-token-probability limits and strict source-bit operator gates.

- [x] **Full-runtime reference-control prerequisite:** actual public metadata
  identifies the rejected constructor as pinned BLAS (ACCEL), before CPU
  initialization. The diagnostic keeps its original pre-setter exit125 and
  produces no numerical capture. Strict build, exact source/build/invocation
  binding and owned process/group/EOF cleanup pass. This diagnoses a control
  harness refusal; it does not explain or close the long-context arithmetic
  divergence. Guarded public compute-route coverage remains pending; no BLAS
  disable, source arithmetic or tolerance change.

## Committed

### fff1842 — complete two-model reference control

- [x] **Fresh2B whole-runtime reference control:** original8192-ID corpus,
  CPU8/F32 KV/FA off, batch/ubatch64, public use_ref and distinct CPU_Mapped
  intermediate placement pass all730 selected records/full248320 logits and
  all42 active causal domains,55,382,016 state words, zero bit differences
  versus accepted Rust. Plain/observed control runs preserve all730/fullV
  and whole serialized prior/post-state. Original CPU/BLAS providers remain
  live in their original order; no BLAS math is delegated. Actual public
  graph placement, buffer types, CPU plan/TLS route, source/build/invocation
  and owned process/group/EOF cleanup checks pass. V6/V7 pre-delegation
  exit125 failures remain preserved. This qualifies a declared reference
  control, not default bulk arithmetic or historical corpus recovery; the
  broader required numerical gate stays open. No production math or tolerance
  changed. Source freeze6f349a12; exact native state export38f96ba1.

### 9a19901 — original reference-control schedule

- [x] **Fresh0.8B whole-runtime reference control:** original8192-ID corpus,
  CPU8/F32 KV/FA off, batch/ubatch64 and public CPU use_ref control pass all730
  selected records/full248320 logits and all42 active causal domains,
  55,382,016 state words, zero bit differences including signed zero versus
  accepted Rust. Actual630 public CPU calls bind630 inner plans with use_ref
  true and exact TLS ownership; two BLAS lifecycle contexts execute no math.
  Original source/build/provider/observation and owned cleanup guards pass.
  This qualifies the explicit reference-control schedule, not default bulk
  arithmetic, historical corpus recovery or production arithmetic changes.

### bfe62df — independent source and reference-path qualification

- [x] Fresh mixed/Unicode teacher prefix: both original GGUF models at
  512/1024/2048/4096, pinned native AR1/F32 KV/FA off versus Rust: all730
  selected actual stage records/full248320 logits and independently decoded
  all42 active causal state domains bit-exact. Each4K state comparison covers
  30,216,192 words. Fresh native8K captures/55,382,016-word state exports are
  complete. BOTH fresh8K comparisons now pass all730/full248320 and all42
  active causal domains,55,382,016 words/model with zero differences;
  Rust6K captures alone are not independent6K comparisons. Historical8K raw
  corpus was not recovered; this fresh corpus does not establish its replay.
- [x] Unchanged0.8B first64 bulk-versus-AR1 capture narrows the first observed
  result gap to layer3 attention AV, position2 (530/2048 words), before the
  previously observed layer4 GDN gap. All131072 probability operands and
  1,064,960 causal V comparisons are exact;114432 future probabilities are
  positive zero. Earlier captured layers0–2/GDN states match. Plain/observed
  whole state and next AR1 full-vocabulary guards prove capture transparency.
- [x] Genuine public ggml AV replay:65 defaults/262144 words match their own
  captures. Public reference-control bulk differs57823 words from default
  bulk and matches all64 default AR1 outputs exactly. Transparent exported
  entry observation preserves all130 default/control outputs (524288 words)
  in both runs and records576 real calls:bulk64 takes64 true returns,
  AR1 takes512 false returns, reference control takes none. Exact opaque ABI,
  pinned DSO origin and bounded owned cleanup pass independent review.
  Internal compiled kernel/full-runtime cause is unqualified; no production
  arithmetic, source profile or tolerance changed.
- [x] Unchanged2B first64 bulk/AR1 capture: plain/observed whole post-state
  and next AR1 full248320/whole-state guards pass. All28 original physical
  learned operands remain unchanged after both modes; eight opaque CPU_REPACK
  payloads are transport-only, with no codec equivalence claim. The first
  observed result gap is layer0 F32 alpha projection, position0 (13/16 words).
  All960 preceding compared rows and131072 actual alpha-input words match;
  its full F32[2048,16] weight equals the original131072 GGUF bytes. Beta
  projection also differs before GDN. No production fix or tolerance change
  is inferred from this capture.
- [x] Genuine2B public alpha replay:65 defaults/2048 output words match their
  own original captures. Public reference-control bulk matches all64 default
  AR1 outputs exactly; reference AR1 outputs are unchanged. Transparent
  exported-entry observation preserves all130 default/control outputs,
 4096 words/run;520 calls comprise8 bulk true,512 AR1 false and zero control
  calls. Original inputs/full raw F32 weights, source/DSO identity and bounded
  owned cleanup pass. This narrows native batch-dependent arithmetic before
  GDN; internal compiled route/full-runtime cause and historical issue closure
  remain open. Rust/source arithmetic and numerical limits are unchanged.

- [x] Actual compiled-route observation for BOTH unchanged public replays:
  all130 default/control outputs, operand checks, exported-entry traces and
  complete before/after mapped-file sets/hashes remain exact. Each dataset
  records132 first named entries across392 stops. Default bulk enters the
  pinned tinyBLAS F32 outer/full/tail families; the other129 scopes enter
  ggml_vec_dot_f32. Static source/code pins identify one Vec8 partial/output
  versus four Vec8 partials/reduction, respectively. Only first named entries
  were observed, not every hit/instruction or private numerical arguments.
  Five actual owned-SID/pidfd/subreaper failure fixtures and actual debugger/
  inferior/group/EOF cleanup pass. First default-scheduler guard failure is
  preserved; additive explicit scheduler-locking off retains original math.
  This strengthens the fresh first64 batch-dependent arithmetic explanation;
  it does not recover the lost historical corpus, qualify full host-runtime
  closure or change production source math/tolerances.


### 6d74f76

- [x] Original CPU Phase 1 functional 16/16 through genuine 8K; bounded GPU genuine 512-token internal comparisons accepted.
- [x] Neither result closes the open independent numerical or larger-context gates.

### Accepted investigation findings — 6d74f76

- [x] Independent final-token divergence first appears in0.8B layer4 GDN and2B layer1 GDN, after preceding layers match. The separate historical AMD FMA failure is not a demonstrated CPU cause/fix.
- [x] At native ubatch512, shared-top20 logprob differences were0.20372924/0.19678696 (0.8B/2B); matching greedy7993 did not close numerical agreement.2B changes with native scheduling;0.8B did not. These partial-top20/activation observations are not full-vocabulary/KL/total-variation proof.
- [x] A large-position RoPE or padded-state fixture does not qualify a genuine long-context whole model. Canonical future-KV padding and zero signs remain source semantics.

````

## Source: .agents/plans/pipeline-composition/README.md

Original UTF-8 bytes: 39459; SHA256 `3d3737b7b956d2d5b6bf30a38f19b0b6cb651b49e1362d20b21472257894df4f`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: Project owner
Co-authors: GPT-6

# Zerogravity — one fabric, single-model and multi-model pipelines

**Decision date:** 18 October 2026. This is the canonical product intent.
**Implementation status:** source-neutral model registry, typed component
extension schemas, canonical YAML/CLI graph compiler, immutable in-memory
context-owner snapshots and source-grant filtering are implemented.
Neither multi-model execution, API serving, speech diarization, streaming
audio/image, large-model SSD inference nor other model-family registrations
are claims of working numerical inference. Unsupported operations refuse.

## Product contract

**Both modes use one engine and one single binary.**

1. **Direct / single-model:** the traditional
   `zerogravity run model.gguf --prompt "hello"` starts a qualified model
   in its own process without having to declare a pipeline, server or router.
   A one-stage pipeline can also expose that model's API once serving exists.
2. **One process / many models:** a configuration owns a pool of independently
   source-verified model instances, each with optional replicas/adapters,
   different placement and residency policies. Many graph pipelines and API
   routes reuse those instances. Loading a new pipeline does not imply a new
   OS process, duplicate model allocation or a second Rust inference runtime.
3. **Same configuration from YAML or repeatable flags:** provide a human-edited
   versioned YAML document with `model:`, `pipeline:` (including typed
   `step:` / `link:` lists) and `route:`, or pass `--model`,
   `--step`, `--link` and `--route` repeatedly with input/output/name
   flags. Both inputs lower into **one source-neutral graph**.
   CLI overrides of an existing config must have an explicit merge/precedence
   contract before being enabled: today mixing them refuses.
4. **One router publishes configured APIs:** an explicit route names a pipeline
   and a protocol with no guesses about conversions or source capability.
   Supported protocol *intent names* are `openai-text`, `openai-asr`, `sse`,
   `websocket` and `webrtc`; these currently validate but do not bind.
   Future serving adapters use the original TaskCoordinator/ResourceArbiter;
   no transport gets its own scheduler or permission to publish neural state.
5. **Every execution/migration/quality claim is separately qualified.**
   Source metadata, downloaded bytes, cache verification, model semantics,
   backend numerical parity, streaming packet delivery, and measured latency
   form distinct gates. A checked config starts no model and opens no socket.

The *same* declared model instance can be used by a chat endpoint, an
ASR→LLM→TTS voice assistant, a multimodal vision agent, or as a standalone
request. Shared immutable tensors and existing StateRoot copy-on-write are
reused, but each request/turn has private mutable state and authorization.
A model can also have named, separately admitted replicas: alias reuse and
replica count are never inferred from two component references.

### Intended engine flow

```text
CLI single-model run -------------------+
                                        |
YAML / repeated pipeline flags --------+--> validate one typed graph
                                               |
Verified snapshot + model assembly catalog ----+--> one model residency manager
                                               |    GPU / RAM / SSD / peers
                                               |    shared SourceIdentity/ResourceArbiter
HTTP / SSE / WebSocket / WebRTC / CLI --------->+--> one TaskCoordinator
                                                     + TaskDrivers & typed ports
                                                     + model/codec/solver execution
                                                     + shared admission/fair queues
                                                     + atomic StateRoot/AV publication
                                                     |
                                       one router -> configured endpoints
```

No central graph dispatcher may add mandatory work to each low-latency token,
audio frame, or resident tensor operation. Compile DAGs and type conversions
once per version/source; use prepared plans and physical completion-guarded
leases. Full resident model execution must not require SSD page faults or a
per-token central registry lock.

## Model priorities (binding order)

This order governs architecture and **source qualification**, not a
promise that an upstream title identifies one immutable weight/quant variant:

1. **Qwen families 3.5+, 3.5, 3.8:** highest optimization priority. The
   source-qualified resident Qwen3.5 0.8B/2B baseline remains unchanged.
   First finish Qwen3.5 2B, then Qwen3.8 27B execution. Qwen3.8 Flash-Next
   follows with independently bound QSA/Gated DeltaNet/MoE/gated residual,
   trained n-gram *embedding* tensors and MTP. Qwen3.5+ is a placeholder until
   an exact upstream release is qualified.
2. **Gemma4:** E2B, 26B-A4B (MoE) then dense 31B. Implement model-specific
   PLE/FFN/attention/routing/relative soft-cap semantics; avoid pretending a
   dense 31B source has 26B MoE kernels.
3. **DeepSeek V4.1 Flash**, then **GLM 5.3 Flash**: add checked source/provider
   pipeline and independent model architecture contracts, but **do NOT download
   their payloads** until the user expressly lifts the hold. The public
   repositories currently indicate DeepSeek FP8 `deepseek_v41` and
   GLM `glm5_next`, not a Qwen-compatible generic MoE.
4. **GLM 5.3 full**, **Meta Muse Spark Open** (including NVFP4 and license),
   **Mistral** and **Llama** families. Every special sparse/recurrent/
   multimodal/codec architecture gets its own source-real gates. A closed
   Muse variant is not automatically available through open Muse weights.
5. Other top OSS models (Phi, Kimi, MiniMax, other DeepSeek, FLUX/Wan/Kokoro)
   are discoverable placeholders; adding a name never adds model capability.

**Media targets:** Qwen Image 2.1, Qwen TTS, Qwen ASR, OpenAI Whisper,
NVIDIA Canary-Qwen 2.5B, NVIDIA Parakeet TDT 1.1B; add encoder, diarization,
codec and decision families independently. Qwen Image uses a pinned Diffusers
**text encoder + processor + transformer + scheduler + VAE assembly**, not
a GGUF or plain text-model auto-loader. Canary has NeMo/Qwen composite
frontend semantics; Parakeet is a native NeMo archive with TDT decoder.
ASR and TTS need their own feature extractors/codebooks and full media clocks.

The source-neutral registry exists at
`crates/zerogravity-core/src/targets.rs`; `zerogravity models list` and
`zerogravity models inspect SLUG` only report that intent. These operations
may not construct model/provider/GPU sessions. `models run/stream` fail
closed even when metadata-only source resolution is verified. The existing
loaded `ModelCatalog` and CPU/GPU `run` are separate source-bound owners.

## Pull sequence and quant policy

**Phase A — actual pinned source plan/pull:**

- Original Qwen3.5 0.8B/2B; Qwen3.8 27B; Gemma4 E2B/26B/31B,
  all source-bound and in available explicitly selected quants.
- Qwen Image2.1 original multi-component assembly; Qwen ASR/TTS exact
  original processors/codecs; Whisper; NVIDIA Canary and Parakeet.
- Qwen3.8 Flash-Next original BF16/FP8 and explicit third-party Q2,
  about 3.5 bits-per-weight, and Q4 **source variants**, including separately
  recognized learned n-gram embedding banks, MTP and other required files.
- DeepSeek V4.1 Flash and GLM 5.3 Flash: **metadata/pipeline only**.
  These two are expressly prohibited from payload download for now.

The existing `pull hf ... --file model.safetensors.index.json` handles one
pinned indexed source, with model sidecars; `pull hf ... --quant Q4_K_M`
selects one exact GGUF basename-tag family, not a quantization conversion.
An actual **3.5-bit** quality class does not imply a literal `Q3.5` tag
(the existing ASCII GGUF tag grammar uses names such as `Q3_K_M`). Source
repository, pin, exact encoding, shapes, quant group/scales and independent
trained inference/quality must be bound. No guessed quant repository,
dequant fallback, alias rewrite or automatic conversion. For a user-selected
quantizer, an explicit source-derived recipe must preserve all MTP/n-gram/
KV/recurrent components or refuse as incomplete.

**Source bundle for nonstandard multi-component models:**

`zerogravity pull hf REPO --revision REV --bundle --include GLOB ...`
or the corresponding GitHub choice collects **only** bounded explicitly
selected members of one pinned tree through the existing transport, transfer
admission, raw hashing and sole cache publisher. It assigns all members
`Auxiliary` under `huggingface-bundle` or `github-bundle`, rather than
inventing an executable model closure. Exact offline selection requires
the identical cached subset. Symlinks and uncontrolled directories refuse;
Xet-native bundle assembly, nested safetensors index automation and NeMo
unpacking are not implemented. Source-first checking and dry-run do NOT
verify nested shards, completeness, processors or codec correctness; assembly
qualification requires a separate validated component schema.

For instance QwenImage2.1 has a `model_index.json` tying typed
`text_encoder`, `transformer`, `vae`, `scheduler`, `processor` paths.
Unknown custom Python code must never execute from a model repository.
Native metadata-only pulls of all requested families are the near-term
acceptance target; payload fetch requires a selected destination, storage
admission and applicable source/license authorization.

## Pipeline API and typed composition

The reusable contract is registered in
`crates/zerogravity-core/src/pipeline.rs`; input/output ports have
**explicit semantic types**, currently `text`, `audio-pcm`,
`timed-transcript`, `speaker-transcript`, `image-rgba` and the
**non-user-constructible** `committed-chat-context` snapshot. Registered
component kinds: `asr`, `diarization`, `transcript-text`,
`speakers-text`, `translate`, `llm`, `tts`, `image-generate`.
ASR, diarization, transcript-text, speakers-text and translation
expose OPTIONAL read-only `context` ports, with `translate` producing
speaker-tagged text without mutating the acoustic source. A future
component adds one registered source-bound port contract and a
TaskDriver/codec implementation; it does **not** require a new server,
model catalog or generic unverified decoder path. Output that cannot
flow into an input needs an explicit registered conversion component.
The compiler validates graph bounds, duplicate IDs, required ports,
speaker/ASR/LLM/TTS model bindings, topological DAG, source-to-output
productive connections, route names/types and unique path declarations.

The ten human-facing YAML quickstarts for text/voice/context,
audio codecs, image assembly/video frames and decision filters are indexed
in [the examples README](../../../examples/pipelines/README.md).

```sh
# Only checks the pure graph; no file/source/model or socket is opened.
zerogravity pipeline check --config examples/pipelines/voice-assistant.yaml

# Same structural graph can be written using repeated flags:
zerogravity pipeline check --name chat --input text --output text \
  --model assistant=qwen3.5-2b --step reply=llm:assistant \
  --link input.value=reply.text --link reply.text=output.value \
  --route /v1/chat/completions=openai-text

# Explicit refusal until real source/backend/stream/transport gates pass.
zerogravity pipeline run --config examples/pipelines/voice-assistant.yaml
```

The YAML `model:` entries identify one shared instance per model ID;
`pipeline[].step` references those IDs, and `route:` declares an endpoint
bound to one pipeline. Validation is bounded (256 KiB file, 128 model
entries, 64 pipelines, 128 stages/512 edges each and 128 routes).
YAML is versioned with `schema: zerogravity.pipeline.v1`, a single
document, strict typed fields and no executable extensions. The loader
supports scalar anchors/aliases and shallow `<<` mapping merges,
but refuses custom tags, duplicate keys, recursive aliases, malformed
merges, depth above 32, more than 16,384 expanded AST nodes and
expanded scalar content above 256 KiB. CLI flags and YAML generate
the same immutable *shape*; loading a model,
accessing credentials, opening network ports, or making quality claims is
not part of validation. Port clocks/formats, compatible codec transforms,
auth and router SLOs require source-specific future contracts.

### Typed extension registry and future media operator contracts

The native `ComponentRegistry` in
`crates/zerogravity-core/src/pipeline.rs` permits a host/agent to register
a new **bounded immutable schema** for a custom stage and call
`Config::validate_with_registry` without modifying the central DAG
compiler or creating another serving process. The default CLI accepts
only built-in contracts; it cannot load arbitrary user-provided Rust/Python
code or infer a model executable from YAML. Each static extension
declares its task binding, input/output ports, semantic value kinds,
and optional input ports. Duplicate/malformed names, duplicate or
missing ports, and forged context outputs refuse. Every new backend
still needs independent source/codec/permission/trained-value/runtime
qualification before it can execute.

The current future-purpose stage descriptions (all **NOT IMPLEMENTED**)
include `audio-feature-extract` (PCM→features),
`audio-tokenize`/`audio-detokenize` (PCM↔codebook tokens),
`image-solve` (prompt→source-typed latents),
`image-decode` (latents→complete pixels),
`video-generate` (prompt→complete video frame stream) and
`decision` (text→typed verdict). They complement the earlier
ASR, diarization, transcript conversion, context-filter/translation,
LLM, TTS and image-generation scaffolds. The type registry contains
`audio-features`, `audio-tokens`, `image-latents`, `video-frames`,
and `decision-verdict` in addition to existing text/audio/image/context
values. Typed descriptions are **not** codecs, working operators,
scheduler jobs, API endpoints or streaming packet formats.

Future implementations must add source-specific memory placement,
modality clocks, completed audio codebook/frame boundaries, context
grant/epoch handling, legal image pixel/tile publication, video
presentation ordering, decision probability calibration and real
ResourceArbiter admission. A registered structural port never
authorizes a model download or creates an executable model instance.

The canonical YAML quickstarts cover direct text, ASR+diarization,
voice assistant, opt-in contextual voice, voice with tool/decision
filtering, audio codec, image generation, latent/decoder assembly,
video streaming intent and standalone decision models.
See [all examples](../../../examples/pipelines/README.md).

### Required pipeline examples

**Complete speech recognition and diarization:**

```text
audio samples --> ASR ------------------> timed transcript ---+
             \                                             |
              +--> diarization(audio, transcript) ---------+
                                            |
                                     speaker transcript
```

**Single coherent voice assistant:**

```text
audio -> ASR -> diarization -> speaker-text -> Qwen LLM -> TTS -> audio
                      \           /
                       timestamps
```

The same Qwen model can also serve `chat` in the same process. Speaker
identity, timestamped partial transcript and playback finality are typed,
not free-form text concatenation. A model may be configured for exclusive
or shared placement when proven; route selection never changes model
source/quant under the hood.

### Optional previous-chat context for ASR, diarization and translation

**Owner decision, 18 October 2026:** a single-process voice assistant may
optionally share its **text LLM's previous conversation context** with the
ASR, speaker-attribution and transcript-to-text/translation stages. This
is a first-class typed input, not a separate model server or a raw
conversation dump. The same `assistant` model instance may serve chat,
answer-generation and a separately source-qualified translation stage.

The context path is **OFF by default**. A human-editable config opts in:

```yaml
# Fragment of one entry under pipeline:; see full example for models/stages.
pipeline:
  - name: talk
    input: audio-pcm
    output: audio-pcm
    realtime: true
    chat_context: {enabled: true, from_model: assistant, max_bytes: 4096}
    # Context is read-only; optional stage inputs get None when disabled.
    link:
      - {from: context.value, to: listen.context}
      - {from: context.value, to: speaker.context}
      - {from: context.value, to: translate.context}
      - {from: context.value, to: prompt.context}
```

The equivalent repeated CLI flags are `--chat-context-from assistant`
(optionally `--chat-context-max-bytes 4096`) and
`--chat-context-enabled false` to turn sharing off without removing
edges. See `examples/pipelines/contextual-voice-assistant.yaml`
for the full typed ASR→speaker diarization→speaker-preserving translation→
text conversion→shared Qwen response→TTS pipeline and a separate chat
API in the **same process**.

**No same-turn cycle:** `context.value` is a synthetic, read-only source
of a previously committed context snapshot, or other app-authorized
configuration/shared source captured for this request. It is never an
output edge from a concurrently executing LLM/reply stage. The compiler reserves the
`context` name, validates one declared text-capable `from_model`
that appears in a configured `llm` stage (possibly another pipeline
in the same process), and rejects unknown providers, value spoofing,
cycles, duplicate inputs and snapshots outside **1..1048576 UTF-8 bytes**. A 4KiB snapshot
is the default; a stricter runtime/resource policy can impose a lower
limit, and larger explicit budgets need normal admission.
The snapshot has a fixed, source-neutral `CommittedChatContext` type:
it cannot be injected as an external `audio-pcm`, raw text, user
parameter, model KV address or arbitrary provider callback. A disabled
policy emits `None`, even when its optional typed edges are still in
the graph; a missing policy rejects any attempt to wire the special
context source.

### Generic ordered context filtering and decision models

The same reusable, typed filter pipeline can reduce any source-granted
context stream before it reaches ASR, diarization, text interpretation,
translation, an LLM, or other future components. It is **not** a special
speech-only boolean or an unconditional privacy whitelist.

The new `crates/zerogravity-core/src/context_filter.rs` defines a bounded
`ContextFilterPlan`: requested provenance-labelled record sources,
then ordered `include`/`exclude` predicates (record kind, exact
tool name, source origin, or literal text substring), and optionally
a source-bound calibrated **decision-model filter** that decides whether
each surviving record is relevant. Filtering is deterministic and
order-preserving; an excluded source cannot be recovered by any later
decision. Independent `tool_call` and `tool_result` kinds can be
retained or removed separately, preserving caller preferences.

Without explicit source selectors the default remains only
`session/user_message` and `session/assistant_message`.
The user can additionally request any authenticated context sources
such as `session/tool_call`, `session/tool_result`,
`application/system_prompt`, `application/developer_prompt` or
`shared:team-support/assistant_message`. The latter needs a real
named grant to access another user's shared history. This is an
**authorization check**, not a blanket prohibition on sharing.
Filters operate solely on the records actually provided under the
source owner's grant. This way the default is conservative but nothing
is hard-coded to refuse an application-owned prompt or an authorized
cross-user resource.

```yaml
# Fragment: config models include assistant and gate; see full example.
pipeline:
  - name: talk
    chat_context:
      enabled: true
      from_model: assistant
      max_bytes: 65536
      sources:
        - session/user_message
        - session/assistant_message
        - session/tool_call
        - session/tool_result
        - application/system_prompt
        - application/developer_prompt
        - shared:team-support/assistant_message
      filter:
        - kind: match
          action: exclude
          field: kind
          values: [tool_result]
        - kind: match
          action: exclude
          field: tool_name
          values: [database.lookup, filesystem.read]
        - kind: decision
          model: gate
          question: >-
            Does this record help interpret the current utterance?
          min_confidence_bps: 8000
          on_unavailable: omit
```

Equivalent repeated flags use
`--chat-context-source ORIGIN/KIND`,
`--chat-context-match exclude:kind:tool_result`, and
`--chat-context-decision gate:8000:omit:Relevant to audio?`.
A selected prompt/tool record can retain its exact structured provenance
in later runtime stages. No filter silently recasts a tool-result as a
higher-privilege instruction.

**Decision models are optional, composable components.** NanoJev-style
choice/boolean/score predictors are good examples, but several different
NanoJev implementations exist, so `nanojev-decision` is an *unqualified
model family intent*, not an implicitly selected checkpoint. It has a
typed include/omit result and confidence in 0..10000 basis points;
a configured threshold gates its verdict. `on_unavailable` accepts
`omit` (default), `reject` or `keep_authorized` (explicitly permissive).
Even the last policy may retain only records the owner already granted
and previous filters kept. No model weights, hosted API or decoder is
invoked by `pipeline check`; decision inference is a separate source
and runtime acceptance gate. Arbitrary model-supplied code and unbounded
prompts are not used as filters.

### In-memory source owner and snapshot selection (implemented in core)

`crates/zerogravity-core/src/context_snapshot.rs` now supplies an actual
in-process source boundary, not merely a filter over arbitrary caller views:

```text
trusted application's authenticated source owner (ContextOwner)
  ├── commit(tenant/session, source, committed_turn, tool_name, text)
  ├── grant / revoke(exact reader session, exact source)
  ├── erase(record_id)         # source version changes
  └── reader(authenticated SessionScope)
        └── capture(ChatContextPolicy, current_turn, DecisionAdapter)
              [short lock: only source-requested AND owner-authorized records]
              [release lock: ordered deterministic stages + typed decision]
              [short lock: verify exact source/grant epoch]
                → immutable original-order ContextSnapshot
                  + record IDs, source and publisher provenance
                  + included byte charge and truncation marker
                  + decision metadata (record ID/stage/model/verdict; no text)
```

The local source owner implements **one exact tenant/session reader
capability**, prior committed-turn eligibility for that session's messages,
explicit app prompt grants for the same tenant and explicit
`shared:resource` grants (including authorized cross-tenant readers).
Selection never equates a config request with a grant. A reader cannot
mint a different reader or mutate grants. Only a trusted host/application
holding `ContextOwner` can publish, grant, revoke or erase data; a
request's authenticated scope must be supplied by that trusted host.
No checkpoint/plugin/user-supplied executable code is a source owner.

The owner is **concurrency safe and optimistic**, not a per-token global
lock: record selection copies bounded complete candidates while holding
the short source lock, then runs filtering/decision callbacks *outside*
the lock. It checks the original epoch again after decisions. Any
intervening grant, revoke, erase or source commit refuses the entire
snapshot. A later router must call `ContextReader::require_fresh`
at its admission/publication boundary and bind the source epoch to the
original session/TaskCoordinator state; previously delivered bytes
cannot be recalled after a grant is revoked.

Logical bounded-resource contracts: store up to 4096 immutable records
and 16MiB logical retained bytes; stage at most the latest 1024 complete
granted records/4MiB for filtering, with a configurable final output
budget up to 1MiB. On output overflow the snapshot retains the
contiguous **newest complete** selected records in original order,
marks `truncated`, and never clips UTF-8 or fabricates half records.
Source owner's explicit `erase` handles removal and invalidates older
epochs. Physical allocator/RSS admission and persistent/remote source
synchronization remain separate runtime ownership work.

The `ContextDecisionAdapter` receives the original authorized record
view and typed question/model/stage/record ID for one filter stage.
For independently authored NanoJev-style decisions, the owner can
supply a calibrated backend later. No backend silently runs now:
`NoDecisionModel` returns unavailable, triggering the chosen
omit/reject/keep-authorized policy. A hard adapter error propagates
instead of being silently promoted to permissive fallback. Neither
inference nor an API listener is started by snapshot capture.

**What this does not establish:** there is no connected chat-history
provider, automatic conversation permissions, cross-process grant
persistence, source-real trained NanoJev checkpoint, ASR integration,
live router streaming or physical GPU/SSD model. The local tests use
a trusted in-memory fixture and source-free decision adapters. Adapter
authorization, owner-authenticated durable providers, multi-model
routing and ASR/decision fidelity remain separate acceptance gates.

**Failure and audit contract:** actual runtime implementation should
record the selected source version/grant, ordered filter steps, typed
decision model version, confidence, and included record IDs without
logging sensitive raw text by default. Authorization is checked before
filter execution and remains revocable. The context snapshot is frozen
at the declared turn boundary, then capped at the actual admitted
memory budget; a stale session, source revoke or missing required
decision causes the configured explicit refusal/omission, not a new
permission or guessed answer. For experiments, source-real A/B tests
should compare ASR accuracy, speaker DER, and latency with no-context,
filtered context, full authorized context and decision-filtered context.

**Runtime privacy/correctness contract (not implemented yet):**

- The *request's own authenticated conversation/session* supplies a
  bounded, immutable, previously committed history slice or any
  other **explicitly requested source** from the runtime's authorized
  application scope. Prior turns, tool calls/results, app-owned system
  and developer prompts, plus other users' records shared under a
  named, actually granted source scope, are all eligible. A config
  requests records; it does not grant permissions, invent hidden model
  state or access another person's private data without authorization.
  For example `application/system_prompt` means the application's
  own provided prompt, not another provider's inaccessible internal
  instructions. Source provenance and explicit, revocable owner grants
  must be retained. Treat included prompts and tool results as
  **untrusted data** for downstream ASR/decision models, not new
  privileged instructions. The caller's declared data scope, actual
  authorization and output consumer are separate checked contracts.
- Freeze the snapshot and its generation/turn ID **before** ASR for the
  next audio segment. Newly generated text for that segment cannot
  retroactively bias the same segment; a barge-in or canceled/stale
  epoch must not publish that segment's context or corrected transcript.
  Bound memory/copy time and avoid per-frame lookup/formatting.
- ASR may use history as vocabulary/terminology/language hints **only
  if its original model/processor supports those inputs**. Whisper,
  Qwen ASR, Canary and Parakeet each require independent capability
  qualification. An unsupported model does not receive a fabricated
  prompt; use a separately verified transcript rescoring/translation
  component where appropriate. Output must retain timings,
  uncertainty and traceable unmodified acoustic hypotheses.
- Diarization can condition speaker naming/continuity on prior known
  identities **only after** the acoustic segmentation/speaker evidence.
  It must not manufacture speakers/timestamps or treat chat names as
  biometric matches. Maintain consent/privacy rules for named voices.
- Translation/correction uses its own separately admitted model
  stage, with speaker-tagged time alignment and original transcript
  retained alongside any revised interpretation. Phrase-level
  look-back cannot silently rewrite already final/played output.
  Context-consistency can help technical terms but cannot override
  contradictory source speech.

**Acceptance before enabling runtime:** source-real A/B tests with
context ON/OFF, named-entity WER/CER, translation adequacy, diarization
DER, timestamp preservation, speaker consent, bounded fanout,
latency/real-time factor, stale-turn/cancel/tenant isolation and
incorrect-hint adversarial cases. A shape-valid `pipeline check` reports
the context toggle but **does not read conversations or start models**.
`pipeline run` continues to refuse with NOT IMPLEMENTED.

**Image generation:** prompt → pinned text processor/encoder → generated
latents via source-exact diffusion/flow solver → VAE decoder → complete
pixel frames. A preview image/latent tile is not a final playable image.
Image **streaming** may emit only registered complete tile/frame stages,
with explicit generations, cancellation and delivery-ACK bounds.

### Realtime, backpressure and failure atomicity

Each stream track uses rational clocks: input audio samples, ASR windows,
speaker-tagged transcript segments, output codec frames, image latent steps,
image tiles, video frames and solver steps are different semantic domains.
A session's StateRoot commits accepted model/codec work only after actual
device completion. **Playable** output is a separate frontier from neural
state. For duplex audio, support VAD, barge-in, playback ACK, output credit,
jitter buffering and stale epoch rejection; cancellation stops new
publication and retains fenced GPU allocations until genuine completion.
Never conflate "transport write succeeded" with "audio played." Coherent
image stream output requires complete validated pixels at a legal boundary.
Slow consumers impose bounded queue credits rather than unbounded media
accumulation. Hard realtime is not promised; soft SLO-gated admission
may reject a model whose cold SSD misses make live audio impossible.

Cross-model buffers are immutable/leased and may transfer as zero-copy
CPU/UMA/GPU views only where one verified placement and lifetime contract
permits. Otherwise copy through an explicitly accounted transfer stage.
There is **one** physical ResourceArbiter, fairness owner and readiness
ledger. Sharing within a process is not a reason to bypass copy isolation,
tenant boundaries or eventual GPU event retirement.

### API/router lifecycle

Loading a config produces a validated generation with model/source/version
pins, dependency graph and route definitions. Only after separate source and
backend qualification may the router provision model/source leases, warm
selected residents, admit queue/physical budgets and bind authorized APIs.
Hot reload requires an atomic new generation, old-session draining, route
version pinning, and safe physical reclamation, with no silent live model
swap. Streaming and batch replicas must use the same owner and report
separate model-load/compile time and per-request TTFT/ITL/p95 latency.
No implicit public bind, token configuration or cross-model credential
inheritance. Route APIs can target one pipeline or expose individual models,
but they cannot invent an unavailable OpenAI-compatible endpoint or live
WebRTC transport.

## Optimization sequence, success gates and non-goals

After the model-source plan:

- **Qwen3.5 2B:** CPU/GPU exact trained source and tokenization, then
  source-preserving quantized AVX/VNNI + native CUDA/HIP/Vulkan kernels,
  prefill/decode kernels, KV and shared-resident memory, real measured
  latency/throughput/power and physical cancellation; Qwen3.5 0.8B baseline
  remains the faster regression canary.
- **Qwen3.8 27B:** lift the executable 4GiB cap using source-real retained
  safetensors shard readers and typed tensor extents. Implement **SSD
  streaming as a first-class placement tier** with quant-aware prefetch,
  fused CPU/GPU stages and overlap. Compare complete trained logits,
  StateRoot, exact tokenization, I/O/OS headroom, cold/warm TTFT/ITL and
  failed reads before tuning any speed assertion.
- **Gemma4 E2B → 26B-A4B → 31B:** exact model geometry and source-bound
  routing/PLE/sliding/value norm/soft-cap, quant kernels and residency
  optimizations independently; use one shared scheduler and measured
  complete requests, not raw synthetic FLOPS.
- **Qwen3.8 Flash-Next:** learn QSA micro-block index and Gated DeltaNet,
  MoE/hyperconnection layout, resident/SSD learned n-gram embedding bank
  and 4B MTP heads with source-exact verification. Important distinction:
  trained n-gram **embeddings are checkpoint tensors**, while an
  n-gram/suffix speculative decoder is a **separate runtime strategy**
  that may not need any new weight download. Validate every accepted prefix
  and RNG/KV/recurrent rollback.
- **Other families/media:** scale via registered operators/components and
  source-real backend/codec gates, without runtime forks. Prioritize
  Whisper/Canary/Parakeet ASR and Qwen TTS/ASR streaming stages, then
  Qwen Image2.1 latent/pixel streaming; source-verified GPU and quality
  benchmarks before claiming interactive readiness.

**Do not:** spawn a separate executable/server for every pipeline node,
silently accept a new license, auto-download withheld DeepSeek/GLM payloads,
guess Q3.5 quant aliases, import arbitrary checkpoint Python, conflate
JSON shape compilation with a complete model assembly, or accept
unmeasured SSD/graph/speculation claims.

## Actual public metadata-only acquisition evidence — 18 Oct 2026

All rows below were checked using **this repository's original** built
`zerogravity pull hf ... --revision main --dry-run` resolver, with
`HF_TOKEN` and `GITHUB_TOKEN` unset. Selection was either one original
`--file model.safetensors.index.json` with automatically closed shards
or an explicitly selected `--bundle --include '*'` raw tree.
Use these pinned commits only after checking source permissions and license.
No weight bytes or SDK were downloaded and no cache snapshot was published.

| Target | Verified Hugging Face repository | Exact pinned commit | Selected bytes | Coverage |
| --- | --- | --- | ---: | --- |
| Qwen3.5 0.8B | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 1,769,978,895 | indexed weights |
| Qwen3.5 2B | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 4,571,272,453 | indexed weights |
| Gemma4 31B | `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | 62,578,684,366 | indexed weights |
| Qwen3.8 Flash-Next | `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | 360,023,349,944 | indexed weights; not a proof all MTP/n-gram groups have independent quant recipes |
| Qwen Image 2.1 | `Qwen/Qwen-Image-2.1` | `d26bb61231c349cf6b7896fa83353113880e1ba3` | 33,134,949,561 | opaque bundle only |
| Qwen3 ASR 1.7B | `Qwen/Qwen3-ASR-1.7B` | `7278e1e70fe206f11671096ffdd38061171dd6e5` | 4,703,114,308 | opaque bundle only |
| Qwen3 TTS 1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `0c0e3051f131929182e2c023b9537f8b1c68adfe` | 4,520,218,951 | opaque bundle only |
| Whisper large v3 turbo | `openai/whisper-large-v3-turbo` | `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | 1,622,466,054 | opaque bundle only |
| NVIDIA Canary-Qwen 2.5B | `nvidia/canary-qwen-2.5b` | `b1469e1bba1cfe140205529c79c434ca47180960` | 5,119,159,377 | opaque bundle only |
| NVIDIA Parakeet TDT 1.1B | `nvidia/parakeet-tdt-1.1b` | `53276c6469d1f17a1352e30c4d11be3d0d7e9575` | 4,283,149,969 | opaque .nemo bundle, no archive extraction |


The Qwen3.5 0.8B and 2B canonical weight inventories are **indexed
single-shard files** named `model.safetensors-00001-of-00001.safetensors`;
requesting `model.safetensors` alone does not match that repository.
The pinned index file correctly expands that unusual one-shard name.
The two Qwen originals already have separate trained local GGUF
execution gates; their HF metadata observations do not convert the
GGUF baseline to an automatically runnable safetensors model.

For an opaque bundle, the selected byte total is **not** an assembly
attestation: the original runtime does not yet recursively bind nested
Diffusers, NeMo codec archives or MTP calibration into one executable
model. Community Flash-Next GGUF variants may omit the 4B learned MTP
layer; a source-exact quantized core must not claim MTP execution until
the MTP weights and their distinct precision/semantics are verified.
Learned 51B n-gram embedding tables are separate from speculative
n-gram/suffix prefix matching, which is an algorithm and may use no
extra model weights.

The original tokenless Qwen3.8 27B and Gemma4 E2B/26B indexed
metadata checks from the preceding phase remain recorded in
[the acquisition source handoff](../model-acquisition/README.md).
Neither metadata source observation nor CLI config validation
authorizes physical model downloads or inference.

## References

- [Current model acquisition & source trust](../model-acquisition/README.md)
- [Model execution](../model-execution/TODO.md)
- [Serving & speculation](../serving-and-speculation/TODO.md)
- [Multimodal](../multimodal/TODO.md)
- [SSD & residency](../memory-and-streaming/TODO.md)
- [Unified multimodal execution contracts](../../docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md)
- [Realtime QoS](../../docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md)
- [Shared resources and serving](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md)
- [Official Qwen Flash-Next source](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Official Qwen Image2.1 component assembly](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/model_index.json)
- [NVIDIA Parakeet NeMo archive](https://huggingface.co/nvidia/parakeet-tdt-1.1b/tree/main)
- [Meta Muse Spark model/license](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4)

**Source binding reference:** The [TianyuCodings NanoJev architecture](https://github.com/TianyuCodings/NanoJev) and the independent [nano-jev MiniLM decision model](https://github.com/shubham10divakar/nano-jev) illustrate distinct candidate families. No one checkpoint is presumed and no trained local decision inference is claimed.

````

## Source: .agents/plans/pipeline-composition/TODO.md

Original UTF-8 bytes: 17912; SHA256 `771bbe4580bd690e88a1ca9930df458f66fcf5e647597621ab6022733af9f72c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — Single-process model and pipeline composition

Status: pipeline architecture/scaffolds complete; execution deferred to owner plans
Provenance: 4bf624e (original scaffolds); ac44e83/f28a62c (context source/filter)
Design: [README.md](README.md)
Owners: model-acquisition, model-execution, serving-and-speculation, multimodal,
memory-and-streaming. All model/source/media execution remains subject to
their existing numeric/source/physical acceptance gates.

## Uncommitted

- [x] Canonical YAML schema `zerogravity.pipeline.v1` replaces TOML
  for the single-process/multi-model CLI config. Repeatable model/step/link/
  route/context flags lower into the same typed validator. Strict YAML
  256KiB source/expanded scalar, 16,384 nodes/32 depth, single document,
  deny-unknown-fields/custom-tag/recursive-alias protections; bounded
  anchors/aliases/shallow merges allowed. Ten sample YAML configurations
  cover text, audio, ASR/speaker diarization, chat+context filtering,
  NanoJev-like decisions, latent/complete image stages and video frames.
  `pipeline check` only validates graph structure; `pipeline run` still
  refuses. A maintained YAML parser replacement is a separate dependency
  review, since the already-cached `serde_yaml 0.9.34` is deprecated.
- [x] Add finite native `ComponentRegistry` with independently registered
  immutable typed port contracts, pure `Config::validate_with_registry`,
  duplicate/invalid/forged-context port refusal and source-neutral future
  stage kinds for audio feature/codebook conversion, image solve/decode,
  video frames and calibrated decisions. Every stage and source remains
  NOT IMPLEMENTED for runtime use until separately qualified.
  No per-model binary/process, runtime scheduler, GPU, transport, extra
  licensed source or user code loaded.
- [x] Frozen source-neutral architecture/YAML acceptance passed:
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  exit0, 1,073 Rust executions/0 failures/69 ignored/230 suites
  plus 18 namespace Python fixtures, strict production/diagnostic
  Clippy, CPU comparator self-tests, release CLI, installer, shell
  and Nix formatting. Both original approved Qwen3.5 checkpoints
  retain full 248320-vocabulary logits, 42 StateRoot domains,
  nonzero tokens, independent sessions and source/reclamation leases:
  cache/direct18.43s and HF-import23.39s independent release routes
  pass exit0. Nix offline source-integrity includes the ten
  YAML examples, typed parser and extension registry; root README,
  workspace Cargo.toml and owned model math/source paths unchanged.
  Acceptance is **pure configuration/registry scaffolding only**;
  trained multimodel execution, audio/video/image streams, NanoJev
  inference, source-real decoded shards and production APIs remain
  in the deferred owner plans. Sign exactly reviewed source/examples/docs;
  no push, no device launch, no model download.

## Committed

- [x] Prior accepted source and family contracts remain in
  model-acquisition, model-execution, multimodal, serving, memory and
  earlier signed history; none of those prior unqualified model/runtime
  implementations are retroactively promoted.

### 4bf624e — accepted foundation

- [x] Source-neutral ordered model target matrix for Qwen3.5+/3.5/3.8,
  Gemma4, DeepSeek V4.1 Flash, GLM5.3 Flash/full, Muse Spark, Mistral/Llama,
  additional OSS, QwenImage2.1/QwenTTS/QwenASR/Whisper/Canary/Parakeet.
  Source name/layout/release status distinguished from executable support.
  Exact local 0.8B/2B Qwen GGUF baseline not promoted to arbitrary HF repo.
  DeepSeek V4.1 Flash/GLM5.3 Flash payload hold represented explicitly.

- [x] Pure typed ComponentContract, ModelInstance, Pipeline, Edge and ApiRoute
  types in existing neutral core. Compile bounded DAG with matching semantic
  ports, required producers, topology, source-to-output productivity,
  unambiguous exposed routes and shared model IDs. Output is NOT a server,
  loaded model or qualified media graph.

- [x] Default CLI `models [list|inspect]` static truthful registry and
  fail-closed `models run|stream`. initial TOML schema (now YAML) `zerogravity.pipeline.v1`
  and repeated `--model`/`--step`/`--link`/`--route` flags lower into
  exactly the same graph validator; config file size ≤256KiB. Both single-
  model and multi-model diagrams/example configs exist. `pipeline check`
  is pure and `pipeline run` is explicitly NOT IMPLEMENTED.

- [x] Bounded pinned HF/GitHub `--bundle --include` opaque Auxiliary-only
  data selection for Diffusers/multi-stage speech assets, reusing one Hub
  pin/tree, existing filters/admission/cache publisher and exact offline
  bundle-only source identity. No invented model/codec completeness, native
  Xet bundle, unknown-size cap, arbitrary subdir recursion or executable
  checkpoint interpretation.

- [x] Implement source-neutral explicit optional chat-context policy:
  per-pipeline enabled toggle (default absent/off), shared text LLM model
  instance, originally max1..16384 UTF-8 bytes and prior committed
  same-session visible turns (historical 4bf624e semantics, now superseded
  by the configurable source selection/filtering policy above).
  Typed optional context.value inputs reach ASR/diarization/
  transcript-text/speakers-text/translation. The graph still prevents
  same-turn cyclic feedback and unowned raw KV injection. Disabled policy leaves edges but delivers None.
  Source-real contextual speech/diarization/translation, privacy-safe
  snapshot capture and context A/B quality remain NOT IMPLEMENTED.
  initial TOML (now YAML) and repeated flags compile identical shape; include full
  contextual-voice-assistant example and adversarial negative tests.

- [x] Frozen source-only acceptance: exact MAIN
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  passed 1,032 Rust executions, zero failures, 69 intentional ignores
  across 230 suite runs, plus 18 namespace Python tests, release CLI,
  Nix/shell formatting, installer, CPU numeric self-tests and strict
  default/production/diagnostic Clippy. Actual Nix filtered source-integrity
  includes the new core/CLI modules and five initial TOML examples (migrated to YAML).
  Original Qwen3.5 release cache_prepare source/model parity independently
  passes for both models, all 248320 logits, all42 StateRoot domains,
  2 tokens, session/source leases, mutation/reclaim: direct 21.05s and
  HF-import 24.10s. Both successful after an initial aborted connector
  tool invocation and explicitly corrected two test-local shadowing errors
  plus a bad nested help quote; no numeric/source/runtime tolerances relaxed.
  Acceptance uses an exact-path signed local source commit. No device
  launch, data download, live server, license approval or real inference extension.

### ac44e83 — accepted foundation

- [x] Replace fixed prior-visible-only history with a reusable typed
  ContextFilterPlan for independently configurable user/assistant messages,
  tool calls, tool results, application-owned system/developer prompts and
  explicitly named shared/granted records. Default remains session
  user+assistant; filters include/exclude by record kind, exact tool name,
  source origin or text substring in deterministic order. Typed calibrated
  NanoJev-style decision stage references a separately declared Decision
  model, accepts threshold in basis points and configurable unavailable
  policy (omit/reject/keep_authorized). The latter never reintroduces
  excluded, missing or ungranted records. Supported initial TOML (now YAML) and repeatable CLI
  flags; bound source selectors/controls and configurable 4KiB default
  through 1MiB context budget. Configuration *requests* source rights;
  runtime source owner must verify actual authorization, not a hard-coded
  block against application prompts or authorized sharing. No model
  inference/automatic context history access is claimed.

- [x] Audited all 6 old detached side worktrees against signed C3/C4/C5,
  v1 and pipeline acceptance commits; preserved earlier unfinished portable
  source patches under .agents/docs and copied nine side-only run artifacts
  into existing MAIN .agents/runs before removing all six detached
  worktrees. One new active, clean worktree
  zerogravity-worktrees/agents-conventions at unmerged e0410c7 is
  intentionally retained. No branch or unrelated user work deleted.

### f28a62c — accepted foundation

- [x] Source-grant checked context owner/snapshot service: the Rust core
  now holds committed immutable records and exact revocable session,
  app-prompt and shared/cross-tenant source grants behind one trusted
  `ContextOwner`; separately minted read-only `ContextReader` capabilities
  cannot mutate grants or rebind to another principal. Per-session prior
  turn check, bounded latest-whole window, provenance IDs/source labels,
  source epoch retry/freshness validation, and filter-only exposure to
  verified candidates are implemented. Decision adapter receives record
  text only after source grant + previous filters; it returns a typed
  include/omit verdict or explicit unavailable/error. Snapshot records
  original-source attributes and decision audit (not raw text in audit).
  This is an in-process source owner, not an external chat connector
  or pretrained model executor. External source grants must be issued by
  the trusted application before minting a read-only reader. Separate
  tests prove that a policy selector alone cannot grant access.

## Tests and verification

- [x] Source-authoritative snapshot tests pass: exact tenant/application/
  session isolation, current-turn exclusion, cross-tenant shared grants,
  same-application prompts, independent tool-result filtering,
  authorized-only decision inputs, optimistic revoke during callbacks
  without deadlock, stale/foreign reader refusal, erasure/record provenance,
  bounded latest-whole 4MiB candidate and 1MiB output windows,
  full-or-empty source behavior, decision error propagation and
  secret-free decision audit. Full original scripts/check.sh frozen
  isolated gate passed 1,058 Rust executions/0 failures/69 ignored/230
  suite runs plus 18 Python namespace fixtures, strict lint, release
  build/installer, Nix-format and CPU comparator selftests.
  Independent release cache_prepare exact approved Qwen3.5 0.8B+2B
  parity passes full vocab/42 StateRoot domains/2 token sessions/leases:
  cache-direct17.36s and HF-import21.91s. Actual Nix offline
  source-integrity passed with new Rust module, root README and
  dependencies/lock unchanged; no live model/GPU/provider access.

- [x] Frozen isolated updated-context-filter source-only check.sh
  passed 1,047 Rust executions, 0 failed, 69 intentionally ignored
  across 230 suites, plus 18 Python namespace fixtures, four strict
  production/diagnostic feature lint gates, Nix formatting, release
  installer and CPU comparator gates. Two independently invoked original
  Qwen3.5 release cache_prepare routes pass both approved models,
  full248320 logits/all42 state domains/two tokens/source-lease/
  mutation/reclamation bit-exact: direct18.66s, HF-import23.24s.
  Real offline Nix filtered-source integrity includes two new paths;
  root README, workspace deps/Cargo.lock unchanged.
  This qualifies only config shape plus pure filtering over explicitly
  *provided* already-granted record views. Live source authorization,
  history retrieval, calibrated NanoJev model inference and real media
  execution are not implemented.

- [x] Filter pipeline: independent tool-call and tool-result opt-in/
  exclusion, context source selection for app prompts/shared grants,
  no silent permission acquisition, ordered predicates, user preference
  filters and decision confidence/fallback. Include/omit verdicts must
  not resurrect previously filtered entries or invent inaccessible text.
- [x] initial TOML (now YAML) + equivalent CLI filters and NanoJev-style typed decision
  references. No inferred upstream checkpoint or executable model. Reject
  duplicate/unknown scopes, malformed origin names, unsourced decision
  models, non-decision heads, invalid thresholds, invalid return variants,
  excessive source/filter/budget and unsupported executable pipelines.
- [x] Opt-in previous-turn context *shape* passes initial TOML (now YAML) and repeated flags,
  with toggling ON/OFF on the same graph, absent-by-default channel, and
  static bounds/unknown model/non-language source/no consumer/spoofed
  current-turn feedback refusal. Configuration checking has no
  data/transport access; a later runtime may include explicitly
  requested authorized app prompts or shared user records only under
  real source grants. Runtime content/biometric/clock tests remain open.
- [x] Core shape validation passes: one-stage Qwen, model sharing across
  two routes, ASR+diarization+optional contextual translation+LLM+TTS
  and image flow. Stages are semantic placeholders only.
- [x] Source-free negative tests pass: cycles, orphan stages, wrong
  typed ports, missing speaker source, duplicate model ID/API/path/input,
  unknown schema fields, unsupported model/media routes, invalid quant/role hold,
  context spoof/size/provider/current-turn linkage and disabled output.
- [x] Default CPU & acquisition CLI tests: offline `models list/inspect`,
  `pipeline check --config` and equivalent flags/ON-OFF context toggle;
  `run/stream` refuse without source/physical/API effects.
- [x] Acquisition and CLI gates: original HF/GH closed weights selection
  unchanged; `--bundle` separate Auxiliary role/source, original cache
  publisher and exact offline cache. Native Xet bundle refuses before
  Hub/CAS. No invented FirstSeen trust or guessed release payload;
  conflicting options fail before credentials.
- [x] Full frozen isolated `scripts/check.sh` 1,032/0 failures,
  four strict artifact/CLI profile lint and core CI, Nix filtered-source
  integrity and original two Qwen reference release regressions pass.

## Deferred

- [ ] OSS release/distribution naming and "latest" pointers remain
  observational until an exact immutable source and codec version is bound.
- [ ] Remote routed pipelines/peers, encrypted hosted gateways and
  implicit zero-copy cross-device are not enabled by source-free shapes.


### Runtime work deferred until after core model priorities

All entries below are still UNIMPLEMENTED. Their active owners are
[model-execution](../model-execution/TODO.md),
[serving-and-speculation](../serving-and-speculation/TODO.md),
[multimodal](../multimodal/TODO.md),
[memory-and-streaming](../memory-and-streaming/TODO.md) and
[model-acquisition](../model-acquisition/TODO.md).
The completed pipeline-scaffold plan is not a claim of serving/realtime
availability and does not keep pipeline development on the critical path.

- [ ] Connect authenticated application/conversation history providers to
  ContextOwner, with durable and revocable grants, owner-authenticated
  tenant+application+session identity, event/graph epoch binding to the
  existing TaskCoordinator and physical memory/residency admission. Wire
  source-real calibrated NanoJev/other decision heads and streaming media
  A/B tests before enabling an actual context-fed ASR/translation router.
  Do not infer inference support from the in-memory source fixture.

- [ ] Source execution: bind actual learned model, tokenizer, processor
  assembly and port formats in original ModelCatalog/TaskDriver before
  enabling any pipeline run or route listening.

- [ ] Single-process multi-model scheduler: immutable source residency,
  exact shared instance reuse/replicas, typed cross-model buffer leases,
  physical ready fencing, bounded ingress/admission/cancellation and
  model-specific batch/fairness. One TaskCoordinator publishes StateRoots.

- [ ] Pipeline runtime/hot reload: per-session model identity, graph generation,
  buffered media/source epoch, legal stage yields, failure rollback and
  atomic route transition. No stale audio, image or cross-tenant leakage.

- [ ] API serving: explicit auth/bind/route config and tested OpenAI
  chat, ASR, TTS, images; SSE/WS/WebRTC soft-realtime with content/size
  bounds and accurate NOT IMPLEMENTED capability reports until qualified.

- [ ] Context-aware source-real voice pipeline: stage-specific ASR prompt/
  vocab support or separate contextual transcript reranker, acoustic-first
  diarization and source-preserving contextual translation. Implement
  prior-committed same-session snapshot freezing, cancellation epochs,
  explicit consent, redaction, bounded history and no unauthorized
  cross-user disclosures.
  A/B compare WER/CER/DER/translation adequacy/latency for context
  ON/OFF; never override contradicted acoustic evidence. This does not
  follow from shape validation.

- [ ] Actual staged ASR + diarization (speaker-tagged timestamps), ASR
  -> Qwen -> TTS one coherent command; audio sample clocks/codebook
  playback/VAD/barge-in and image latent/complete preview frames with
  backpressure. Validate codec/latency/source on real media hardware.

- [ ] Optimize model execution: Qwen3.5 2B, Qwen3.8 27B
  resident+SSD, Gemma E2B/26B/31B, then Qwen Flash-Next QSA/GDN/MoE/
  learned n-gram embeddings/MTP and separate speculative n-gram/draft,
  with measured model-real speed/quality/physical safety gates.

- [ ] Complete remaining exact Q2/Q3.x/Q4 quant source pipelines,
  full native image/audio/NeMo assemblies, token/codec quality and
  range/resume/cancellation before declaring "all models supported".
  Keep DeepSeek V4.1 Flash and GLM5.3 Flash strictly pipeline-only now.

- [ ] Trained and streaming multi-model parity, real media correctness,
  device completion, SLO/fairness and zero-copy lease gates are owned by
  the model-execution/serving/multimodal/memory TODOs above.

````

## Source: .agents/plans/platform-and-release/TODO.md

Original UTF-8 bytes: 3497; SHA256 `121c75ca2c48de4b00620d1c4b4ebadb2f1851b7275b93bee8e4997b67f2c7b4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Platform and release

Provenance commit: `6d74f76`.
Scope: remaining P00/P04/P05/P08/P20/P21/P22/P23.
Contracts: [P00–P23 roadmap](../IMPLEMENTATION_ROADMAP.md),
[minimal execution spine](../../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md),
[compiler/task refinement](../../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md),
[release gates](../RELEASE_ACCEPTANCE_GATES.md).

## Uncommitted

- [ ] Generalize the working typed forward plan into verified effectful semantic regions/task control, closed provider substitution, bounded shape/kernel caches and explicit command/resource legality; no new persistent IR/runtime without a real need.
- [ ] Prove artifact→model→compiled task→coordinator→output and explain-plan for supported tasks, including source-qualified preselected fallback and no selected-GPU-error CPU substitution.
- [ ] Run extension drills for one stateful semantic op, device/provider, non-token TaskProgram, quant/storage/API adapter and optional ParameterVersion/TrainStep seam; future training is not thereby implemented.
- [ ] Finish per-file source/model/license/NOTICE/SBOM policy, malformed-input fuzzing and ownership/security review, authenticated APIs/private caches and qualified driver/device safety.
- [ ] Verify Nix/module/native/cross packages and installer/release behavior on actual Linux/Windows/macOS target access; packaging definitions alone do not prove platform support.
- [ ] Maintain minimal/default/composed CPU-free feature isolation, one lock/version authority and truthful unsupported capability behavior.
- [ ] Gate bounded compiler/autotune/AOT caches by source/device/driver/shape and retain negative candidates; no unbounded first-request search.
- [ ] Complete reproducible/signing/release policy and fair model/task/quality/performance matrices with cold/warm, p50/p95/p99, memory/energy and uncertainty. Push/tag/release remain separately authorized actions.
- [ ] Demonstrate compatibility/invalidation of operator/task/state/plan caches and bounded change radius before broader platform claims.

- [ ] Before a true zero-copy strided View is enabled, validate contiguity against stored value provenance and test public PlanValue retargeting. Current View physically materializes dense output; retain cached-element and dynamic range guards.
- [ ] Preserve one GPU-library service/queue/quarantine owner and cold generic-plan admission through packaging changes; verify actual final downstream links, not only compiler/type checks.

## Committed

### 1af2ac4

- [x] NixOS module eval check evaluates a complete base system (root filesystem and boot loader) so the whole-system assertions hold on current nixpkgs-unstable; the CI `quality` job was failing at this evaluation.

### 6d74f76

- [x] Cohesive workspace, ordinary CPU/product packaging, one shared GPU library and accepted narrow original-model vertical slice.
- [x] These completed slices do not mark every original P00–P23 goal complete.

## Tests and verification

- [ ] Qualify all six [CPU release definitions](../../../docs/DEVELOPMENT.md#cpu-cross-release-definitions) with actual target execution, installer checksum/version/refusal cases and platform signing/authenticity policy. Build definitions and same-host checks do not establish other-target support.
- [ ] Test enabled NixOS service lifecycle and runtime-secret handling after serving is implemented; the currently disabled service definition and module evaluation are not a running-server claim.

````

## Source: .agents/plans/qwen-parity/API_ROUTE_FRONTIER.csv

Original UTF-8 bytes: 7276; SHA256 `b749d2dd43067a507a4b679155a2d2c918efd593a77f8706da23c8bf4e25ae1d`.
Resolve embedded relative paths against the original path, not this volume.

````text
project,path,line,method,route,status
vllm,vllm/entrypoints/anthropic/api_router.py,51,POST,/v1/messages,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/anthropic/api_router.py,89,POST,/v1/messages/count_tokens,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/cohere/api_router.py,119,POST,/cohere/v2/chat,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/cohere/api_router.py,156,POST,/cohere/v2/chat/render,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/generate/generative_scoring/api_router.py,30,POST,/generative_scoring,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/generate/structured_decisions/api_router.py,33,POST,/v1/systemone,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/openai/chat_completion/api_router.py,43,POST,/v1/chat/completions,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/openai/chat_completion/api_router.py,88,POST,/v1/chat/completions/batch,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/openai/completion/api_router.py,37,POST,/v1/completions,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/openai/decisions/api_router.py,24,POST,/v1/decisions,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/openai/models/api_router.py,20,GET,/v1/models,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/openai/responses/api_router.py,110,POST,/v1/responses/{response_id}/cancel,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/openai/responses/api_router.py,48,POST,/v1/responses,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/openai/responses/api_router.py,80,GET,/v1/responses/{response_id},DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/pooling/classify/api_router.py,26,POST,/classify,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/pooling/embed/api_router.py,28,POST,/v1/embeddings,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/pooling/embed/api_router.py,46,POST,/v2/embed,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/pooling/pooling/api_router.py,27,POST,/pooling,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/pooling/scoring/api_router.py,102,POST,/v2/rerank,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/pooling/scoring/api_router.py,37,POST,/score,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/pooling/scoring/api_router.py,52,POST,/v1/score,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/pooling/scoring/api_router.py,69,POST,/rerank,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/pooling/scoring/api_router.py,84,POST,/v1/rerank,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,142,GET,/is_paused,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,159,POST,/init_weight_transfer_engine,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,171,POST,/shutdown_weight_transfer_engine,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,178,POST,/start_weight_update,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,185,POST,/start_draft_weight_update,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,192,POST,/update_weights,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,204,POST,/finish_weight_update,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,214,POST,/update_weight_version,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,223,GET,/weight_info,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,229,POST,/weight_checker,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,266,GET,/get_world_size,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,31,POST,/pause,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,75,POST,/resume,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/rl/online/api_router.py,94,POST,/abort_requests,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/scale_out/derender/api_router.py,31,POST,/v1/chat/completions/derender,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/scale_out/derender/api_router.py,81,POST,/v1/completions/derender,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/scale_out/render/api_router.py,101,POST,/v1/responses/render,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/scale_out/render/api_router.py,28,POST,/v1/chat/completions/render,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/scale_out/render/api_router.py,54,POST,/v1/messages/render,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/scale_out/render/api_router.py,78,POST,/v1/completions/render,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/scale_out/token_in_token_out/api_router.py,46,POST,/inference/v1/generate,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/dev/cache/api_router.py,20,POST,/reset_prefix_cache,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/dev/cache/api_router.py,47,POST,/reset_mm_cache,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/dev/cache/api_router.py,57,POST,/reset_encoder_cache,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/dev/rpc/api_router.py,23,POST,/collective_rpc,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/dev/server_info/api_router.py,43,GET,/server_info,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/dev/sleep/api_router.py,26,POST,/sleep,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/dev/sleep/api_router.py,37,POST,/release_kv_cache_memory,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/dev/sleep/api_router.py,44,POST,/wake_up,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/dev/sleep/api_router.py,58,GET,/is_sleeping,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/elastic_ep/api_router.py,27,POST,/scale_elastic_ep,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/elastic_ep/api_router.py,83,POST,/is_scaling_elastic_ep,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/fault_tolerance/api_router.py,37,POST,/fault_tolerance/apply,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/fault_tolerance/api_router.py,93,GET,/fault_tolerance/status,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/lora/api_router.py,63,POST,/v1/load_lora_adapter,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/lora/api_router.py,81,POST,/v1/unload_lora_adapter,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/profile/api_router.py,35,POST,/start_profile,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/profile/api_router.py,48,POST,/stop_profile,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/sagemaker/api_router.py,91,POST,/ping,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/sagemaker/api_router.py,92,GET,/ping,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/sagemaker/api_router.py,98,POST,/invocations,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/tokenize/api_router.py,35,POST,/tokenize,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/tokenize/api_router.py,61,POST,/detokenize,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/serve/tokenize/api_router.py,90,GET,/tokenizer_info,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/speech_to_text/realtime/api_router.py,17,WEBSOCKET,/v1/realtime,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/speech_to_text/transcription/api_router.py,31,POST,/v1/audio/transcriptions,DECLARED_ROUTE_UNREVIEWED
vllm,vllm/entrypoints/speech_to_text/translation/api_router.py,31,POST,/v1/audio/translations,DECLARED_ROUTE_UNREVIEWED

````

## Source: .agents/plans/qwen-parity/CLI_OPTION_FRONTIER.csv

Original UTF-8 bytes: 32856; SHA256 `979f13376516e471f8bd66aab1d057ee7019d40524d88cd315f198be245ec6fa`.
Resolve embedded relative paths against the original path, not this volume.

````text
project,path,line,name,status
llama.cpp,common/arg.cpp,1226,--spec-type,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1447,--help,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1447,--usage,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1454,--version,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1462,--cache-list,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1474,--completion-bash,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1481,--server-base,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1488,--verbose-prompt,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1495,--display-prompt,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1496,--no-display-prompt,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1503,--color,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1520,--threads,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1530,--threads-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1540,--cpu-mask,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1550,--cpu-range,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1560,--cpu-strict,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1567,--prio,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1577,--poll,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1584,--cpu-mask-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1594,--cpu-range-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1604,--cpu-strict-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1611,--prio-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1621,--poll-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1628,--lookup-cache-static,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1635,--lookup-cache-dynamic,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1642,--ctx-size,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1653,--kv-unified-per-slot,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1661,--n-predict,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1661,--predict,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1672,--batch-size,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1679,--ubatch-size,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1686,--keep,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1693,--swa-full,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1701,--ctx-checkpoints,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1701,--swa-checkpoints,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1709,--checkpoint-min-step,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1719,--cache-ram,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1728,--kv-unified,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1729,--no-kv-unified,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1736,--cache-idle-slots,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1737,--no-cache-idle-slots,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1744,--context-shift,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1745,--no-context-shift,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1752,--chunks,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1758,--flash-attn,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1774,--prompt,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1781,--system-prompt,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1788,--perf,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1789,--no-perf,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1797,--show-timings,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1798,--no-show-timings,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1805,--file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1817,--system-prompt-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1827,--in-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1840,--binary-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1856,--escape,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1857,--no-escape,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1864,--print-token-count,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1871,--prompt-cache,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1878,--prompt-cache-all,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1885,--prompt-cache-ro,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1892,--reverse-prompt,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1899,--special,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1906,--conversation,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1907,--no-conversation,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1917,--single-turn,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1926,--interactive,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1933,--interactive-first,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1940,--multiline-input,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1947,--in-prefix-bos,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1955,--in-prefix,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1963,--in-suffix,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1971,--warmup,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1972,--no-warmup,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1979,--spm-infill,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1989,--samplers,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,1998,--seed,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2005,--sampler-seq,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2005,--sampling-seq,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2012,--ignore-eos,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2019,--temp,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2019,--temperature,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2028,--top-k,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2036,--top-p,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2044,--min-p,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2052,--top-n-sigma,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2052,--top-nsigma,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2059,--xtc-probability,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2067,--xtc-threshold,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2075,--typical,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2075,--typical-p,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2082,--repeat-last-n,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2094,--repeat-penalty,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2108,--presence-penalty,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2119,--frequency-penalty,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2130,--dry-multiplier,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2137,--dry-base,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2148,--dry-allowed-length,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2155,--dry-penalty-last-n,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2165,--dry-sequence-breaker,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2191,--adaptive-target,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2201,--adaptive-decay,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2211,--dynatemp-range,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2218,--dynatemp-exp,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2225,--mirostat,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2234,--mirostat-lr,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2242,--mirostat-ent,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2250,--logit-bias,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2272,--grammar,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2279,--grammar-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2286,--json-schema,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2293,--json-schema-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2310,--backend-sampling,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2317,--pooling,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2329,--attention,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2338,--rope-scaling,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2348,--rope-scale,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2355,--rope-freq-base,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2362,--rope-freq-scale,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2369,--yarn-orig-ctx,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2376,--yarn-ext-factor,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2383,--yarn-attn-factor,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2390,--yarn-beta-slow,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2397,--yarn-beta-fast,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2404,--grp-attn-n,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2411,--grp-attn-w,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2418,--kv-offload,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2419,--no-kv-offload,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2426,--repack,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2427,--no-repack,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2434,--no-host,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2441,--cache-type-k,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2454,--cache-type-v,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2467,--hellaswag,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2474,--hellaswag-tasks,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2481,--winogrande,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2488,--winogrande-tasks,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2495,--multiple-choice,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2502,--multiple-choice-tasks,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2509,--kl-divergence,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2516,--kl-divergence-base,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2516,--save-all-logits,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2523,--ppl-stride,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2530,--ppl-output-type,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2537,--defrag-thold,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2548,--parallel,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2559,--parallel,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2567,--sequences,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2574,--cont-batching,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2575,--no-cont-batching,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2582,--mmproj,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2590,--mmproj-url,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2597,--mmproj-auto,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2598,--no-mmproj,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2598,--no-mmproj-auto,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2605,--mmproj-offload,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2606,--no-mmproj-offload,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2613,--rpc,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2614,--mmproj-device,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2633,--audio,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2633,--image,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2633,--video,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2642,--image-min-tokens,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2649,--image-max-tokens,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2656,--mtmd-batch-max-tokens,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2663,--video-fps,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2670,--video-timestamp-interval,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2677,--video-ffmpeg-dir,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2684,--rpc,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2695,--load-mode,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2714,--lazy-mode,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2727,--numa,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2742,--device,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2750,--list-devices,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2758,--override-tensor,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2764,--cpu-moe,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2771,--n-cpu-moe,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2781,--moe-cache-mib,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2791,--n-cpu-ffn,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2803,--gpu-layers,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2803,--n-gpu-layers,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2821,--split-mode,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2845,--tensor-split,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2872,--main-gpu,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2882,--fit,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2896,--fit-print,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2910,--fit-target,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2935,--fit-ctx,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2942,--check-tensors,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2949,--override-kv,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2961,--op-offload,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2962,--no-op-offload,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2969,--lora,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2979,--lora-scaled,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,2994,--control-vector,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3003,--control-vector-scaled,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3017,--control-vector-layer-range,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3025,--alias,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3037,--tags,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3049,--model,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3058,--model-url,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3065,--docker-repo,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3074,--hf-repo,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3084,--hf-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3091,--hf-token,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3098,--mtp,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3105,--dflash,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3112,--eagle3,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3119,--context-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3132,--chunk-size,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3139,--chunk-separator,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3146,--junk,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3153,--pos,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3160,--output,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3160,--output-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3168,--output-frequency,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3175,--output-format,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3184,--save-frequency,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3191,--process-output,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3198,--nextn,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3205,--ppl,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3206,--no-ppl,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3213,--chunk,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3213,--from-chunk,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3220,--show-statistics,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3227,--parse-special,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3234,--ids,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3241,--stdin,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3248,--no-bos,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3255,--no-parse-special,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3262,--show-count,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3307,--embd-normalize,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3314,--embd-output-format,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3321,--embd-separator,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3328,--cls-separator,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3335,--host,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3351,--port,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3358,--reuse-port,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3365,--path,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3372,--cors-origins,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3383,--cors-methods,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3390,--cors-headers,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3397,--cors-credentials,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3398,--no-cors-credentials,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3408,--api-prefix,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3415,--ui-config,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3415,--webui-config,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3422,--ui-config-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3422,--webui-config-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3429,--ui-mcp-proxy,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3429,--webui-mcp-proxy,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3430,--no-ui-mcp-proxy,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3430,--no-webui-mcp-proxy,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3437,--tools,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3447,--tools-runtime,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3458,--mcp-servers-config,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3466,--mcp-servers-json,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3474,--agent,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3475,--no-agent,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3490,--ui,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3490,--webui,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3491,--no-ui,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3491,--no-webui,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3498,--embedding,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3498,--embeddings,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3505,--rerank,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3505,--reranking,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3513,--api-key,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3524,--api-key-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3541,--ssl-key-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3548,--ssl-cert-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3555,--chat-template-kwargs,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3573,--timeout,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3581,--sse-ping-interval,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3588,--threads-http,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3595,--cache-prompt,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3596,--no-cache-prompt,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3603,--cache-reuse,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3613,--metrics,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3620,--props,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3627,--slots,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3628,--no-slots,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3635,--slot-save-path,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3649,--media-path,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3663,--models-dir,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3670,--models-preset,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3677,--models-max,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3684,--models-autoload,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3685,--no-models-autoload,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3692,--jinja,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3693,--no-jinja,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3700,--reasoning-format,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3711,--reasoning,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3729,--reasoning-effort,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3741,--reasoning-budget,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3749,--reasoning-budget-message,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3756,--reasoning-preserve,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3757,--no-reasoning-preserve,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3771,--chat-template,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3783,--chat-template-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3795,--skip-chat-parsing,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3796,--no-skip-chat-parsing,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3806,--prefill-assistant,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3807,--no-prefill-assistant,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3817,--slot-prompt-similarity,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3824,--lora-init-without-apply,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3831,--sleep-idle-seconds,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3841,--simple-io,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3848,--positive-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3855,--negative-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3862,--pca-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3869,--pca-iter,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3876,--method,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3885,--output-format,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3894,--log-disable,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3901,--log-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3908,--log-jsonl,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3909,--no-log-jsonl,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3916,--log-prompts-dir,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3928,--log-colors,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3945,--log-verbose,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3945,--verbose,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3953,--offline,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3960,--log-verbosity,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3960,--verbosity,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3975,--log-prefix,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3976,--no-log-prefix,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3983,--log-timestamps,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3984,--no-log-timestamps,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3996,--hf-repo-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,3996,--spec-draft-hf,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4003,--spec-draft-threads,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4003,--threads-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4013,--spec-draft-threads-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4013,--threads-batch-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4023,--cpu-mask-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4023,--spec-draft-cpu-mask,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4033,--cpu-range-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4033,--spec-draft-cpu-range,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4043,--cpu-strict-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4043,--spec-draft-cpu-strict,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4050,--prio-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4050,--spec-draft-prio,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4060,--poll-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4060,--spec-draft-poll,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4067,--cpu-mask-batch-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4067,--spec-draft-cpu-mask-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4077,--cpu-range-batch-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4077,--spec-draft-cpu-range-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4087,--cpu-strict-batch-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4087,--spec-draft-cpu-strict-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4094,--prio-batch-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4094,--spec-draft-prio-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4104,--poll-batch-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4104,--spec-draft-poll-batch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4111,--cache-type-k-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4111,--spec-draft-type-k,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4124,--cache-type-v-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4124,--spec-draft-type-v,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4137,--override-tensor-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4137,--spec-draft-override-tensor,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4143,--cpu-moe-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4143,--spec-draft-cpu-moe,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4150,--n-cpu-moe-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4150,--spec-draft-n-cpu-moe,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4150,--spec-draft-ncmoe,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4161,--spec-draft-n-max,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4171,--spec-draft-n-min,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4178,--spec-draft-n-ubatch,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4188,--spec-synth-len,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4201,--spec-synth-rates,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4221,--draft-p-split,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4221,--spec-draft-p-split,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4228,--draft-p-min,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4228,--spec-draft-p-min,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4235,--spec-draft-backend-sampling,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4236,--no-spec-draft-backend-sampling,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4244,--spec-draft-sampling,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4259,--device-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4259,--spec-draft-device,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4268,--gpu-layers-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4268,--n-gpu-layers-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4268,--spec-draft-ngl,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4287,--model-draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4287,--spec-draft-model,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4295,--spec-type,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4305,--spec-ngram-mod-n-min,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4315,--spec-ngram-mod-n-max,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4325,--spec-ngram-mod-n-match,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4336,--spec-ngram-simple-size-n,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4346,--spec-ngram-simple-size-m,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4356,--spec-ngram-simple-min-hits,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4367,--spec-ngram-map-k-size-n,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4377,--spec-ngram-map-k-size-m,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4387,--spec-ngram-map-k-min-hits,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4398,--spec-ngram-map-k4v-size-n,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4408,--spec-ngram-map-k4v-size-m,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4418,--spec-ngram-map-k4v-min-hits,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4433,--draft,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4433,--draft-max,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4433,--draft-n,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4440,--draft-min,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4440,--draft-n-min,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4447,--spec-ngram-size-n,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4454,--spec-ngram-size-m,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4461,--spec-ngram-min-hits,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4473,--tts-lang,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4481,--tts-speaker-file,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4493,--diffusion-steps,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4498,--diffusion-visual,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4503,--diffusion-eps,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4508,--diffusion-algorithm,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4516,--diffusion-alg-temp,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4521,--diffusion-block-length,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4526,--diffusion-cfg-scale,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4531,--diffusion-add-gumbel-noise,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4536,--learning-rate,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4540,--learning-rate-min,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4546,--learning-rate-decay-epochs,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4551,--weight-decay,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4556,--val-split,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4561,--epochs,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4566,--optimizer,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4575,--check,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4582,--save-logits,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4589,--logits-output-dir,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4596,--tensor-filter,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4606,--embd-gemma-default,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4636,--fim-qwen-3b-default,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4650,--fim-qwen-7b-default,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4664,--fim-qwen-7b-spec,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4680,--fim-qwen-14b-spec,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4696,--fim-qwen-30b-default,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4710,--gpt-oss-20b-default,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4729,--gpt-oss-120b-default,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4747,--vision-gemma-4b-default,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4758,--vision-gemma-12b-default,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,4769,--spec-default,LEXICAL_CANDIDATE_UNREVIEWED
llama.cpp,common/arg.cpp,833,--spec-type,LEXICAL_CANDIDATE_UNREVIEWED

````

