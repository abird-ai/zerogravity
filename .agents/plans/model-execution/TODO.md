# Model execution

Provenance commit: `6d74f76`.
Scope: P02/P03/P08/P11–P14/P17; CPU truth before GPU specialization.
Contracts: [large immutable sources](../IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md),
[Qwen basics](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md),
[Gemma family](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md).

## Uncommitted

- [ ] Acquire/bind authorized original config/index/tokenizer/checkpoint identities and choose a capable admitted host. Large BF16 sources exceed current local RAM; current executable source limit stays 4 GiB.
- [ ] Implement explicitly admitted verified owned shards, exact complete index/header/name/dtype/offset/shape/hash binding and retained source leases; no unsafe mutable-file mapping shortcut or second issuer.
- [ ] Qwen3.8 27B: original HF grouped-head ordering and Q/K replication 16→48, explicit converted-profile permutations, untied head and source-derived norm/RoPE geometry.
- [ ] Gemma E2B/26B: shared GELU/multiply/scale/soft-cap, proportional RoPE, value norm/sliding/shared KV, E2B PLE/doubled FFN and 26B K=V/MoE branches; choose and qualify the exact reference-framework numerical schedule.
- [ ] Qualify actual trained CPU operators/full vocabulary/state/nonzero replay, official tokenization/templates/reasoning/stop behavior, sessions, rollback/checkpoints and resource refusal.
- [ ] Add the same source-bound forward plans and full physical GPU qualification per eligible device, preserving all state/actual-completion/cancellation guarantees.
- [ ] Keep distinct future heavy/hybrid recipes (Qwen Flash-Next, GLM, MiMo and source-defined QSA/MLA/KDA/indexers/hyperconnections), and broaden dense/encoder/pooling/reranking/LoRA support only through independent model/codec/backend gates.
- [ ] Preserve CPU-first family priorities; already-authorized parallel basics do not authorize model/SDK downloads or unsupported whole-model claims.

- [ ] Bind explicit original HF versus converted Qwen source profiles. Conversion permutes grouped/tiled V projections/constants/convolution/output, uses -exp(A_log) and adds1 to non-GDN norms; GDN swish is SiLU while full-attention gating remains sigmoid.
- [ ] Choose and independently qualify Gemma26B reference-framework routing/top-k/reduction/scale placement before production execution. Five analytic test-local routing cases are not trained source parity.

## Committed

### 6d74f76

- [x] Qwen3.8 27B and Gemma4 E2B/26B configuration, tensor and source-index contracts; no new trained execution, tokenizer or checkpoint loading.

### Source contracts — 023703f

- [x] Qwen3.8 metadata/shard roles and CPU-free Gemma4 family contracts,14 Gemma tests and official metadata/config cases passed within metadata scope. No tokenizer, new learned weights, physical shard-header binding or trained inference follows.
- [x] Preserve Qwen/Gemma source axes, metadata precision and routing/KV facts in the [model semantics reference](../../docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md) and [Gemma technical contract](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md). The executable-source4GiB cap remains.
