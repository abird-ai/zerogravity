Author: Project owner
Co-authors: GPT-6
Owner: @pvl

# Source-preserving SSD residency

## Implemented: immutable original SSD source-extent IO

`zerogravity-artifacts::ssd_extent` under original `cache` feature.

- Already-opened file descriptor + independently pinned SHA256 of strict `zerogravity.extents.v1` JSON with complete contiguous chunk offsets, bounded bytes and per-chunk hashes; manifest also *claims* whole-file SHA256 but never pretends it is verified merely by opening.
- Sorted unique chunk or bounded arbitrary tensor-window requests. Coalesced exact positional IO, staging charged *before* allocation/read to existing caller-provided `ResourceArbiter`, hash every chunk of owned bytes before exposing any slice.
- Max 4 MiB/chunk, 32 MiB/read, 32 chunks/read, 16 MiB index, 200k chunks, 1 TiB file metadata. Returned `StagedExtents`/`StagedSlice` retains both immutable bytes and physical transfer-staging lease through final drop.
- Tests: exact first/last/window reads, damaged manifest, incomplete/overlapping geometry, original source inode mutation, source file rename, truncated reads and physical budget refusal; sparse >5 GiB metadata without materializing giant bytes.
- Resident model hot paths remain unchanged: there is no mandatory SSD read per token.

## Still pending

- Model execution through accepted original-index tensor windows, asynchronous prefetch and source-verified copy-to-GPU completion with stable lease ownership. Gemma source/runtime qualification remains separate.
- Eviction/ABA/cancellation/backpressure, per-expert codec placement, fair IO and physical GPU/RAM/SSD budget integration; never return a borrowed TensorSource slice from an evictable cache.
- Authorized original trained resident-versus-SSD complete logits/StateRoot correctness and cold/warm TTFT/ITL, IO amplification and memory-pressure benchmarks.

**This is source-byte IO, not yet working SSD-streamed inference.**


## Original-index tensor windows

`SourceWeights::prepare_verified_windows` authenticates the complete pinned
original index, whole shard hashes and actual headers using bounded scans.
Its sealed owner exposes original matrix descriptors and immutable owned
full-width row windows, each retaining source identity, actual staging and
metadata leases. It supplies no whole borrowed TensorSource or model permit.
Existing resident-source APIs retain their4GiB policy.

Actual original Qwen3.8-27B qualification scanned all18 weight files
(55,563,006,776B), matched12 raw BF16 row samples including4 extent splits
and released every reservation after the final window dropped. Process
peakRSS was17,328KiB under a512MiB address cap. The configured128MiB host
ledger is not independently attested physical RAM admission; kernel page
cache and generic allocator overhead are separate. Nine window and seven
extent fixtures, full isolated check and approved original release parity
routes pass. See [TODO.md](TODO.md) for exact scope and remaining work.

**Still source-byte IO: no streamed model, GPU upload/Ready or StateRoot
publication is qualified by these results.**
