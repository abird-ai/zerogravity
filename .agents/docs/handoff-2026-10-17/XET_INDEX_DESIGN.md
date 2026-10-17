# TODO — native Xet required-index reconstruction design

Status: cold design; no source or runtime qualification.
Provenance: accepted native52e42fc; read current MAIN accepted implementation while C1 gates run. This run lives only in zerogravity-quant-work and is separate from frozen C2.
Main plan: [model acquisition](../../plans/model-acquisition/TODO.md).
Owned scope: this TODO only. No source/Cargo/MAIN/Git mutation, provider requests, payload/model/SDK/device/live-cache actions.

## Uncommitted

- [x] Read accepted native README and HF closure, Xet Hub/discovery/Access/Reader/Invocation/Budget and executor index validation.
- [x] Identify exact one-pin index-reader seam and verification/admission/control/cumulative work obligations below.
- [ ] Coordinator chooses dry-run/API policy and authorizes a source slice; no implementation authorized by this design.
- [ ] Implement/qualify only after that grant, reconciling accepted C1/C2 without changing legacy selection or native identities.

## Smallest useful slice

Support an HF LFS-backed selected/adjacent `.safetensors.index.json` required by ORIGINAL closure when its fixed commit URL announces native Xet. Reconstruct that small index through ORIGINAL Access + XetReader + hash/codec/multipart/HTTP owners before SafetensorsIndex parses it. Preserve the same one exact commit, pinned tree and remaining metadata bytes/logical requests. Keep non-LFS ordinary index GET behavior; no repin/relist, compatibility redirect/body fallback, new source issuer, cache/publisher, CAS URL/token authority or chunk runtime. Required model weights still transfer only after final closure and normal cache transaction admission.

### Private provider seam

1. Add an optional PRIVATE admitted index-reader callback at HF's existing exact-size/index GET point, after ensure_length and before original `SafetensorsIndex::from_json`. Supply a borrowed pinned context: original repository/commit/path/URL, exact length, LFS raw SHA and existing validator. Do not fake an HttpResponse/header or infer index status from an arbitrary URL in Hub::get.
2. Ordinary HF/compatibility Files and quant routes keep their original callback/GET path. Native entrypoint supplies the callback; it uses the SAME retained Hub budget and caller TransferAdmission. The callback cannot authorize another source/path or change the original selected/adjacent-index policy. GGUF quant normally needs no index; its shared initial-choice path must remain unchanged.
3. For native LFS index, factor/reuse the EXISTING announcement checker: exact fixed commit URL, header-only302, x-xet-hash parse, all present commit/size/rawSHA bindings checked. Require the pinned LFS rawSHA plus exact length; Git OID is not rawSHA. No follow/compatibility response body. A missing/changed announcement refuses.
4. Keep bounded path→XetSpec for an index already observed so final plan can attach that exact spec without another discovery call. Other final files use original discovery. No index-body cache or new persisted records. Original per-source plan/provenance/path identity stays unchanged.

### Exact bytes before parser

- Refuse before CAS/scratch allocation if index exact length exceeds caller max_file, remaining cumulative HF metadata bytes,8MiB parser bound, or addressable allocation. A selected unknown length must first use the already-approved exact metadata/HEAD policy; no cap/partial JSON mode.
- Lazily acquire ORIGINAL Scratch::reservation from caller's same registered transfer domain BEFORE constructing Scratch/Access/reconstruction vectors. Reuse one scratch sequentially for multiple indexes; retain the outer original resolution control lease. Do not create/register a new arbiter/key or weaken limits.
- Allocate the index Vec with exact checked capacity under the existing resolution control envelope, perform bounded reads/cancellation, incrementally compute actual SHA256. After exactly expected bytes, request the separate EOF sentinel: `take(expected)` alone is WRONG because it can skip XetReader's final full-root/drain/EOF verification. Require no extra logical byte and actual terminal native completion.
- The original XetReader must verify complete ordered file root and exact logical length, including extra edge decoding, signed complete interval set/drain/multipart terminal semantics and replay rules. Compare actual rawSHA with pinned LFS SHA independently. Neither Xet hash nor transport metadata replaces rawSHA.
- Reuse reject_pointer; only then return bytes to original static-error SafetensorsIndex parser and required-shard closure. Never parse tentative prefixes/unverified raw bytes; no shard URL/body before closure. Raw index bytes are metadata/data, not executable readiness or tensor total-size authority.
- Decrement ONE cumulative HF index/body byte budget and logical request budget across all selected/adjacent indexes. Charge decoded index bytes to the native semantic metadata budget (without pretending decompressed bytes are additional physical wire), while actual compressed/index/CAS/headers/retries remain charged by the original native HTTP work owner. Existing observed metadata budget includes headers, so limits may conservatively refuse before HF body-only budget is exhausted.

### Admission and lifetime accounting

Current defaults: outer metadata control envelope80MiB (`8*8MiB+16MiB`), native scratch90.5625MiB (includes40*262144 leaf storage, fixed scratch/control and8*metadata). During index resolution their simultaneous declared peak is170.5625MiB; existing native CLI256MiB can fit,128MiB must deny before signed index body. The bounded index Vec/parser is covered by the original outer metadata envelope; do not add an unaccounted second copy. Custom larger limits must use the actual checked formulas, not this default arithmetic. This scopes process transfer/control buffers, not all host RAM/model memory.

Drop resolution scratch, grants and index bytes after their last parser/closure use and before returning the final plan. The final payload invocation then uses original scratch+cache transaction envelopes (~162.6875MiB defaults). First slice may reconstruct the tiny required index again during publication using existing executor so it is actually included in the immutable snapshot; keep that second transfer bounded/charged and honest rather than inventing a temporary index-body cache. The same pinned rawSHA/spec must verify it again, and original executor reparses staged verified index closure before commit. No cache root/ref mutation during resolution; failures/cancellation leave old refs untouched.

### Mandatory cumulative seed change

CURRENT `budget::Seed` carries ONLY attempts/wire/metadata because accepted resolution has not decoded native bodies. Adding pre-plan index reconstruction without changing that seed would RESET terms/groups/ranges/decoded/leaves/refreshes at Invocation::new, exceeding declared whole-operation caps.

Extend the PRIVATE seed/snapshot to carry ALL consumed counters: attempts,wire,metadata,decoded chunks,accepted logical leaves,refreshes,accepted terms/groups/ranges. Budget::new validates every seeded count against the same options and starts from them. Budget::seed must return Result if reading the accepted Mutex can fail; never silently default/zero on poison. Final plan seed is captured only after all index reconstruction and final discovery. Reader scratch leaves can clear per file, but cumulative Budget leaf/decode counts cannot. Count retried/discarded edge chunks and accepted reconstruction entries on every reissue exactly as now. Every physical HTTP retry/redirect attempt retains original begin_attempt behavior; metadata logical calls are a distinct finite counter, not wire telemetry.

Public plan reuse still begins an independent admitted execution from its resolution seed, matching current semantics; do not claim one global cross-process grant/work counter. No grant/token/signed URL persisted in the plan.

## Root choices before source grant

1. **Recommended preserve existing dry-run promise:** existing `xet::resolve`/CLI dry-run remains Hub metadata+announcement only, no CAS grants/signed bodies. Add a named additive native resolver that explicitly permits required-index metadata reconstruction for APPLY (e.g. `resolve_with_indexes`, private shared implementation). CLI apply chooses it; dry-run that NEEDS native index returns explicit Unsupported before grant and cannot claim complete closure/plan. No new flag or silent dry-run network expansion. Alternatively root can deliberately revise dry-run to permit CAS for index metadata, but current help/README's explicit no-CAS/no-signed-body promise must change and be tested; that is a policy decision, not an implementation shortcut.
2. **Recommended first slice require trusted LFS rawSHA for native indexes.** Already qualified ordinary non-LFS index metadata remains allowed under existing pinned/TLS/FirstSeen rules. Native announced index lacking a pinned rawSHA refuses before CAS; do not promote a newly observed hash to expected authority.
3. **Recommended no pre-plan cache reuse/staging.** Reconstruct required index twice if needed (planning then normal immutable publication), charge both against cumulative limits, document the bounded cost. Avoid introducing a new cache ownership/transaction-before-closure protocol merely to save <=8MiB. Future retained verified index optimization is separate.

## Proposed source/test ownership after approval

Native/HF owner: HF private index callback and native entrypoints/discovery helper/Invocation scratch constructor; xet/budget.rs full seed; same reader/access APIs wherever a narrow helper permits reuse. Shared scratch/codec/hash/multipart stay exact, no duplicate parser or new transport. CLI owner: only apply versus metadata-only resolver choice and precise failure/help wording in existing shared run_fetch, preserving ordinary None and C2 Files|Quant source choice. Tests: new acquisition_xet_indexes.rs plus existing native/ordinary HF/GH/selection/legacy/candidate/quant suites. Root owns docs/gates/dependencies (none expected), integration/Git. Concrete paths/API finalized only after root's dry-run decision.

## Tests and verification — all proposed, UNRUN

- [ ] Golden tiny LFS-index native reconstruction produces verified JSON, original closure requires every same-commit shard+sidecar, native plans retain original per-file/source/provenance. Fixed one alias lookup and one tree listing; adjacent index uses same authority. No weight body before closure; required missing shard refuses without publication.
- [ ] Actual original HTTP owner synthetic Hub→read token→CAS reconstruction→signed ranges for index; Hub token never reaches CAS/ranges, CASgrant never Hub/ranges, signed URL auth suppression/no-follow unchanged; final model source roots/rawSHA valid independently. No live provider/account.
- [ ] Valid JSON with wrong rawSHA or wrong Xet root refuses BEFORE index parser/model body; correct root/SHA but malformed/unsafe/duplicate/overflow index refuses with static error. Truncated/missing signed interval/trailing data/replayed changed chunk/cancellation at exact final length refuse; original root/drain/EOF checks actually reached.
- [ ] Multiple required indexes share total logical request/body/cardinality limits; max_terms/groups/ranges/chunks/refreshes configured to allow each file individually but reject index+later payload cumulatively. Decode/refetch edge work and second planning/publication index copy consume remaining limits. Unit seed snapshot restores every counter and poisoned accepted owner returns error.
- [ ] Full-default128MiB admission fails BEFORE signed index body;256MiB succeeds same fixture and releases all declared memory after drops. Custom metadata/chunk bounds prove arithmetic, exact allocation/no geometric growth and no hidden globalRAM claim.
- [ ] Metadata-only dry-run performs zero CAS grants/signed bodies even when native index is required; apply explicitly reconstructs bounded metadata. Offline requires no credentials/client/provider/admission; existing cached data does not become native execution proof.
- [ ] Grant401/range403 retries charge original total attempts/reissues and preserve reconstruction contract; end-to-end old-ref preservation on each new preplan refusal/cancel. Existing publisher fault matrix remains unchanged; no redundant new generic archival/proof framework.
- [ ] Original HF/GH compatibility index/shard/sidecar, legacy offline, C1 discovery, C2 quant, native non-index and ordinary sidecar resume tests; strict feature/dependency matrices, full isolated gate/Nix, approved originals as root decides. No new inference/throughput claim.

## Current code anchors

- `acquisition/huggingface.rs::resolve_from` exact index length→metadata_checked→verify_metadata_digest→SafetensorsIndex parser, under one existing pinned tree/metadata_left.
- `xet/mod.rs::resolve`, `Invocation::new/reader` and `PlanState::seed`: existing original Hub/discovery and reader/admission owner.
- `xet/access.rs::Access::{reconstruction,fetch,refresh}`: scoped grants/typed target and cumulative accepted reconstruction owner.
- `xet/reader.rs::{Scratch::reservation,XetReader::finish_source}`: actual allocation envelope, terminal complete root/EOF.
- `xet/budget.rs::{Seed,Budget::new,seed,reconstruction,decode_chunk,file_leaf,refresh}`: seed currently partial, MUST extend before decoding during resolve.
- `acquisition/executor.rs::execute_with_progress`: sole transaction/importer and staged index reparse before commit. README explicitly leaves native LFS/index reconstruction open and says dry-run no CAS grants/signed bodies.
