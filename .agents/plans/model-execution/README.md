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
