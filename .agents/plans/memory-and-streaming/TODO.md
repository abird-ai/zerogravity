# Memory and streaming

Provenance commit: `6d74f76`.
Scope: P03/P05/P07/P10/P12/P16.
Contracts: [unified residency/SSD](../../docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md),
[SSD gates](../../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md),
[large-source admission](../IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md).

## Uncommitted

- [x] **Original-index owned tensor windows:** existing SourceWeights
  factory prepares a sealed VerifiedTensorSource with one original source
  identity, authenticated complete index/shard/header/bundle closure and
  retained opened descriptors. Whole SHA256 scanning is bounded4MiB;
  only full-width original OutputInput matrix rows can materialize.
  TensorWindow owns immutable staged bytes, exact original kind/shape/
  offset/row geometry and source owner through final drop. No borrowed
  whole TensorSource implementation or55GB resident allocation. Existing
  resident4GiB path remains unchanged. Original host ResourceArbiter
  admission precedes scan/window allocation; charge includes retained
  extent-offset metadata and carried descriptor vector capacity. Metadata
  is capped128MiB; labels/geometry/read counts are bounded. Cancellation
  checks retain synchronous IO owners until completed; pre-cancel/drop
  coverage does not prove asynchronous mid-read cancellation.
- [x] **Actual original Qwen3.8-27B source-window qualification:** exact
  revision1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, acquired snapshot
  98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e.
  One actual55,563,006,776-byte scan verifies all18 weights shards,
  original1199-tensor index/header closure and unchanged bundle digest.
  Twelve actual BF16 full-width row samples across QKV/embedding/last
  down-projection/head include four authenticated extent crossings;
  same-open-inode independent raw reads match every byte. Peak selected
  staging8,388,656B includes retained offsets; final reservation0 after
  the last window drops its source/metadata/FD ownership. Header/source/
  occupied-ledger refusals, pre-cancel, wrong axes/rank/name/ranges and
  attempted whole embedding materialization refuse without leaked charge.
  Root scan completed82.847s under512MiB address cap; measured process
  peakRSS17,328KiB. Configured metadata64MiB/host pool128MiB exercises
  the real shared ledger; it does NOT independently attest free RAM,
  account kernel page cache or qualify physical host admission/inference.
  Initial direct-rustc LTO link failed on dependency bitcode before payload
  execution; locked/offline isolated harness build and actual run passed.
- [x] **Window verification:** nine new synthetic ownership/source tests
  and seven SSD extent tests pass; exact mutation/short-read/rename,
  staging/metadata refusal and retained lifetime cases. Complete isolated
  check1202 Rust/0failed/69ignored/248 suites plus18Python/release/installer/
  Nix; strict CUDA/Vulkan/qualification Clippy and both approved original
  cache-direct/HF-import release routes pass full248320/all42. No GPU
  upload/Ready or model-root publication is performed by this source API.

- [ ] **Next SSD source execution:** integrate accepted original-index
  owned windows and an async I/O/readiness model with
  original CPU/GPU TensorSource and BufferLease, cancellation,
  cache/eviction/ABA and independently trained full-logit/StateRoot
  parity on real large weights.

- [ ] Shared one-process pipeline residency: source-exact immutable
  model/component buffers, per-session private StateRoots, one
  ResourceArbiter (UMA/GPU/RAM/SSD and device queue), cross-model
  lease/zero-copy only when physical placement, codec and lifetime
  permit it. Do not replicate weights per route or release a
  GPU buffer before actual completion/quarantine retirement.
- [ ] Prioritize Qwen3.5 2B resident then Qwen3.8 27B SSD
  streaming and Gemma4 E2B/26B/31B, then sparse Qwen Flash-Next
  learned 51B n-gram bank and 4B MTP head placements. Compare
  source-real logits, resident/streamed StateRoot, cold/warm
  TTFT/ITL, I/O/read amplification and throughput with full
  physical/cancel/error gates; live audio's soft deadline may
  demand resident codecs and refuse cold SSD layouts.

- [ ] Route legacy/new model allocations, CPU state/scratch/staging and APU/GPU shared backing into the same configured physical pools with OS/cgroup headroom; never sum overlapping UMA/VRAM/GTT counters.
- [ ] Implement paged/contiguous state policy and full-domain prefix identity/intersection/COW across KV/SWA/recurrent/indexer/draft/grammar/media state, source/template/adapter and tenant boundaries.
- [ ] Qualify committed snapshot persistence/restore, corruption/security/compatibility UX and restore-versus-recompute; diagnostic portable checkpoints are not product persistence.
- [ ] Add immutable source extents and bounded coalesced IO/materialization with stable logical tensor identity, genuine copy readiness and original IO/device reader leases.
- [ ] Protect encoded-but-unsubmitted/captured expert reads and generation ABA; support mixed-codec expert size classes under one byte cap, source-exact routing and hot-cache preservation.
- [ ] Add bounded PLE/Engram lookup rows and separately governed disk state/media tiers; fully resident execution must incur no mandatory IO or host cache dispatch.
- [ ] Test source-exact resident/streamed parity, short/corrupt IO, cancellation, safe victim refusal, SSD QoS and actual physical lifetime before eviction/free-space claims.
- [ ] Optional direct/async/platform IO needs equivalent semantics and explicit support; large memory-pressure/SSD experiments require approved hardware/source scope.

- [ ] Distinguish logical envelope retirement from driver allocator pages/pools, fragmentation, cross-process use and actual OS release. Driver VRAM/GTT usage/max counters are not reservations or immediately free bytes.
- [ ] Prove trusted teardown/recovery before reclaiming unknown-work handles or clearing a fence. Diagnostic service sync/readback permits inspection/exit only and cannot synthesize Ready, retire unknown events or authorize reuse.

## Committed

### 35b9093 — source-only verified SSD extents

- [x] **Verified source-only SSD extent foundation:** original opened
  descriptor, SHA256-pinned complete chunk manifest, checked source
  offsets and per-range hashes, 4MiB chunk/32MiB read bounds,
  coalesced positional file IO and original ResourceArbiter
  transfer-staging admission before allocating/reading. Exact
  byte-window materialization returns immutable leased slices, never
  unsafe borrowed mutable-file views. Tests cover exact reads,
  truncation/hash mutation, path rename, rollback, insufficient
  host memory and sparse >5GiB metadata-only source.
  Six SSD-focused tests pass, and the entire isolated workspace
  gate passes 1,147/0 across 241 Rust suites. The Nix filtered
  source-integrity derivation cannot run offline because the
  pinned nixpkgs flake input is missing (not a source test failure).
  No trained resident-versus-SSD model, device copy or eviction yet.

### 6d74f76

- [x] Physical arbiter, original GPU leases/quarantine and committed session/checkpoint foundations accepted; global host/UMA, full-state prefix manager and SSD model streaming remain open.

## Tests and verification

- [ ] SSD-01..32: run the [full source/residency suite](../../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) for each claimed backing/backend; mock tiers cannot qualify physical IO or trained streaming.
- [ ] SSD-14/15: encoded-but-unsubmitted, queued and captured readers prevent generation-ABA slot reuse through actual GPU completion/cancellation.
- [ ] SSD-21/22/24: mixed Q2/Q4 expert sizes share one byte cap with safe victims and fair progress; hot-cache retention cannot bypass admission.
- [ ] SSD-32: an authorized original trained streamed model matches complete resident logits/StateRoot; measure cold/warm IO, staging, OS headroom and genuine resource lifetime.
