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

- Exact original Qwen3.8/Gemma tensor→extent/shard index binding, asynchronous prefetch and source-verified copy-to-GPU completion with stable lease ownership.
- Eviction/ABA/cancellation/backpressure, per-expert codec placement, fair IO and physical GPU/RAM/SSD budget integration; never return a borrowed TensorSource slice from an evictable cache.
- Authorized original trained resident-versus-SSD complete logits/StateRoot correctness and cold/warm TTFT/ITL, IO amplification and memory-pressure benchmarks.

**This is source-byte IO, not yet working SSD-streamed inference.**
