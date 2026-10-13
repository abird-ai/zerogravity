# Zerogravity — Extreme Hybrid Model Recipes, Native MTP and DFlash Source Contracts

**2026-10-15 • Design-only refinement of v4.3 ModelRecipe/TaskDriver and P11–P16.**
**Primary sources:** published serving recipes for Qwen, GLM and MiMo hybrid checkpoints. Source semantics must be re-audited against exact original model checkpoint revision. These projects are prior art, NOT an embedded runtime.

## 0. Governing rule

**ModelRecipe defines source math/state; DraftSource proposes; the target verifies/publishes; KernelCandidate executes a source-qualified closed region.** None owns another ModelRouter, TaskCoordinator, StateRoot publisher or hidden external/Python inference product. Chat/reasoning/tool templates are model-source inputs, not global string hacks.

To add another extreme hybrid model, register exact immutable assembly (original/derived checkpoint, attention and MTP/DFlash/vision sidecars), source semantic operators, session StateDomains, verifier/rollback, hardware kernels and truthful capability/quality receipts. Avoid model-name branching in central queue, HTTP or quant loader.

## 1. Three distinct registered model recipes

| Dimension | Qwen3.8-Flash-Next | GLM-5.3-Flash | MiMo-V2.6-Flash |
|---|---|---|---|
| Identity | separately versioned Qwen3.8 | GLM5 Next | new MiMo2 semantic adapter, not generic Qwen |
| Attention/recurrent | GDN plus sparse QSA indexed attention | KDA plus MLA/DSA and mHC | source-specific SWA and global attention |
| State | GDN, QSA index, full KV/lookup | KDA, MLA compressed KV, DSA index, mHC stream | 39 SWA / 9 global in the tested checkpoint, ring and global KV |
| Routing | source-verified top10 / 512 experts | source-verified top8 / 288 experts | load source top-k/expert counts, don't copy GLM/Qwen |
| Draft | trained MTP + optional lookup table | trained MTP, optional eh_proj precision sidecar | shipped DFlash auxiliary model with its own attention |
| Quant | EXL3 derived pack, PLE, optional vision | EXL3, optional BF16 eh_proj | EXL3 MoE plus source E8M0 block scales |
| Current Zerogravity | planned source math, not full native implementation | planned source math, not full native implementation | **source recipe not yet implemented**, needs detailed semantic registration |

These layer/expert counts are inspected source variants, not universal model-family invariants. Working Qwen3.5 0.8/2B native CPU and selected GPU tensor regions do not prove any of these three whole models.

## 2. Qwen3.8 source recipe

- GDN DeltaNet source short-convolution, decay/gates and causal state update are distinct from dense attention.
- Sparse QSA indexer-selected blocks preserve exact selected positions, mask, position embedding and causal order. Indexer state cannot be thrown away when speculating.
- Routed experts/top-k/shared expert and GatedResidual mixing require exact scale, accumulation and intermediate precision.
- Immutable hashed-row n-gram/PLE lookup table may be enormous and file-backed. Lookups are source parameter access, not KV or draft token histories. Only needed rows are paged and guarded by byte/checksum metadata.
- Trained MTP and prompt lookup are independent DraftSources. The model can choose an explicitly registered policy to combine proposals, while target verification remains authoritative.
- A session root includes GDN, QSA, KV, accepted history, sampler/RNG/grammar and complete state versions, even if MTP or PLE is disabled.

## 3. GLM5.3 source recipe

- KDA fine-grained gated recurrence and short convolution must not be silently substituted with Qwen GDN scalar gate math.
- MLA compressed latent KV, NoPE and DSA index selection require separate source attention operators and mask/position semantics.
- mHC manifold-constrained/Sinkhorn stream mixing and residual order are semantic, not arbitrary fusion reorderings.
- Trained MTP sidecar may include an optional higher-precision BF16 eh_proj. A published source report notes a 64MiB BF16 copy improving draft acceptance; bind source hash, shape, license and numerical/derived-artifact mode. A missing sidecar can be valid but must report lower draft capability; never discover it from an unrelated filesystem path.
- Source-specific mutable KDA/MLA/mHC target history grows with number of speculative rows. Global physical resource admission counts those history slots independently of KV.

## 4. MiMo2.6 required source adapter (currently a documentation gap)

A published MiMo adaptation demonstrates a distinct recipe rather than a generic Qwen decoder. Checklist:

1. Exact source HF checkpoint/config/tokenizer/template/processor and architecture revision; source tensor/head names, expert counts and layer topology.
2. E8M0 block scales as explicit numerical encoding; original source scale values and fused weight readers checked independently of EXL3 codebook.
3. Source-specific optional value-head dimension: a published source reports 128 logical v_head_dim inside 192 padded KV head slots. Logical output is NOT the padded layout; no phantom attention channels in source math.
4. Sliding window attention with exact masks/window, and global paged attention. The 39 SWA / 9 global split is **specific to their MiMo2 checkpoint**; ring/cache format determined by source, not one global magic constant.
5. Native MLP expert intermediate overflow and paired conversion/runtime compensation, verified for the exact derived pack; a blanket activation clamp changes model semantics.
6. Source chat/thinking/tool instructions and generation defaults versioned separately from generic API output streaming; alternate medium reasoning is a declared prompt/profile variant.
7. DFlash model and source attention formula plus auxiliary codebook/embedding, not a substitute trained MTP head or unrelated target.
8. Stateful target verifier/rollback; target-only valid source baseline and exact sampler/correction distribution if claiming lossless speculation.
9. Reject incompatible scale, attention, draft asset or quant data before physical submission.
10. Original CPU scalar/layer/checkpoint source oracle, then exact GPU source/vocab, long-context and independent stochastic tests.

### MiMo layer47 FP16 overflow: semantic and derived-format requirement

A published conversion reports one converted pack's expert intermediate product around **84,000**, exceeding finite FP16 maximum **65,504**. Their specific conversion applies an up-projection /128 rescale and compensating routed scale ×128 later in FP32 (called interm_div=128), making the intermediate safe to store in FP16. This is a **paired derived artifact transform** with source-relative numeric proof, not a universal MiMo or MoE activation clamp.

Test specific numeric-sensitive source rows, check nonfinite intermediates, shape/routed expert paths, full logits, target PPL and tool/coding/reasoning. A near-identical PPL alone can miss a broken expert warp or near-tie difference. No kernel may silently switch to FP32 or saturating FP16 without matching source quality/numeric contract and mode receipt.

## 5. DraftSource families and target-authoritative verification

~~~rust
enum DraftFamily {
    TrainedMtp(MtpSource),
    DFlash(DFlashSource),
    IndependentSmallModel(DraftModelSource),
    PromptLookup(HistoryLookupSource),
    OtherVerified(RegisteredDraftOp),
}
struct DraftSource {
    source_id: VerifiedArtifactId,
    target_compatibility: SemanticAndTokenizerKey,
    algorithm: VerifiedProposalAlgorithm,
    mutable_state: DraftStateSchema,
    max_depth: u16,
    sidecars: Vec<VerifiedComponent>,
}
struct VerifyBlock {
    target_parent_root: StateRootId,
    proposals: Vec<TokenId>,
    provisional_state: TargetStateLease,
    target_logits: TargetLogitReceipt,
    rng_grammar_stop: SamplerStateReceipt,
    accepted_prefix_len: usize,
}
~~~

Drafts are *proposals*, never committed media/text or true model tokens. Target runs actual source mathematical forward for verification, accepts only legal prefix and applies exact correction sampling when stochastic mode requires it. On zero acceptance, cancellation or mismatched sidecar all provisional draft and target branch state is discarded, committed TaskCoordinator root unchanged. An optional no-draft target path remains viable.

### Qwen MTP + n-gram lookup
- MTP is source-trained auxiliary head, with exact feature/tap/position; prompt lookup is deterministic token-history proposal, no logits authority.
- Optional mixed policy must preserve tokenizer/lookup row identity and causal template. Failing hash lookup is not justification to propose arbitrary tokens.
- Auxiliary GDN/indexer/hidden state is versioned/leased; offloading draft to another device includes transfer in measured cost.

### GLM MTP and BF16 sidecar
- Auxiliary eh_proj stored as optional BF16 separate artifact, with hashed checkpoint revision and exact model/head dimensions.
- MTP verify history reserves all KDA/MLA/mHC state that could be pending. Tiny auxiliary weights do not imply tiny per-request state.
- No new decode/sampling semantics just because eh_proj uses different physical storage precision.

### MiMo DFlash attention: exact semantic fields

The shipped MiMo auxiliary drafter cannot simply be loaded as generic DFlash with defaults from a different training convention. Validate:

1. Partial RoPE factor and rotated Q/K dimensions. Full-128 rotation instead of source-defined partial rotation is a semantic error.
2. Per-head sink bias and value-scale, including an explicitly verified absorption of value scale into output projection where source math allows it.
3. **Bidirectional draft block** with source attention left/right window. A source is_causal=false must not be normalized into a plain causal left-only window. A published source reports left=1024 and right=block_size-1 in a MiMo drafter variant.
4. Source-learned mask embedding at draft positions, not zeros or a generic BOS.
5. Hidden target layer-tap indexing convention. A published source reports tap_shift=0 for MiMo where older DFlash defaults use +1.
6. Model-specific draft KV ring and attention cache: draft state cannot alias target or another session.
7. Correct proposal length, confidence policy and shape/precision source constraints; caps and thresholds have versioned semantics.
8. Derived/staged DFlash artifact may be generated offline, but source tokenizer/adapter/drafter hashes must validate before load.

**Critical failure:** a broken draft might still produce correct **target** tokens if the verifier rejects it all. Therefore draft semantic validation needs independent intermediate/attention and acceptance golden fixtures, not merely target perplexity or final token parity.

## 6. Distinct state and clock budgets for hybrid models

Do not equate all attention to one paged full-context KV cache. A source SWA window W with needed lookahead/overlap/guard pages may have a fixed ring of W+bound; source-defined ring size varies by checkpoint. Draft DFlash ring is different from target SWA and global KV.

The MiMo source model reports:
- 39 sliding layers with ~175.5MiB fixed per-slot ring and 9 global layers with ~27KiB/token paged KV at one FP16 mode;
- keeping all sliding layers full would be ~261KiB/token and drastically reduce max context;
- its draft 5 sliding layers use a ~1792-token ring of ~35MiB fixed rather than 20KiB/token.

These are **author measurements for an exact pack**, not engine constants. State spec includes window, head, position, padding, ring generation/ABA, causal/source lookahead, acceptance frontier, snapshot/COW and cross-session isolation.

Physical budget = resident target + aux draft+eh_proj + optional vision/PLE + global KV + target SWA ring × sessions + draft KV/ring × sessions + GDN/KDA/mHC/MLA and indexer state × verifier history + scratch + DMA + file-page cache/OS reserve.

## 7. Kernel/route-aware speculation and batching

For B active sequences and per-sequence proposed depths d_i, total target verifier rows R = Σ(1 + d_i). Time is not merely proportional to R: unique routed experts, per-expert packing granularity, padded tiles, cache/state traffic, cold expert fetch and row-independent launch costs all matter. The published observed regime roughly fit **T_ms=32.4+9.04×R**, but this is only its measured source pack/device/quant/shape.

A packed MoE kernel with 16-row chunks yielded a cliff at batch5 × depth2 = 15 rows (one chunk) versus batch6 × depth2 = 18 rows (two, the second mostly empty). Engine optimization should choose among **measured source-qualified shape bands** and consider both single-stream latency and aggregate committed tokens/s, never hardcode 16-row chunk in SIR.

Model-specific gate:
- evaluate target-alone vs draft+target full wall time, including draft residency/compute, verifier rows, accepted tokens, state rollback, scheduling wait and physical cache.
- dynamic backoff or no-draft on low acceptance; mixing prompt classes matters.
- hysteresis avoids repeated candidate flips; no compile/search of entire cost graph in per-token hot path.
- prompts for chat/prose/code/tools/long context and near-tie numeric cases; sampled target distribution, not just greedy tokens.

## 8. Numerical and behavior policy

| Capability claim | Proof |
|---|---|
| Original source semantic model | official checkpoint/reference per-layer/output/state/logit test |
| Derived EXL3 artifact vs original | held-out original-vs-derived pack error/quality separately |
| Lossless target speculation | target-only vs target+draft on the same derived pack, sampler/RNG/grammar and accepted-prefix state |
| Draft algorithm accuracy | tap/attention/mask/source golden and acceptance across prompt classes |
| Batch deterministic output | source numeric reduction/near-tie tests at B=1,2,4,8, not implied by batching |
| Steering or added reasoning style | named **behavior-changing** variant and explicit user mode, not hidden acceleration |

A published report notes some batched multisequence decode that changes greedy near-tie tokens even with FP16 cache, due to different attention reduction order. Therefore strict equality between batch and solo must be explicitly gated by validated numerical policy; an untested batch optimization cannot advertise deterministic equivalence.

Optional residual activation steering is a model-behavior change with unique artifact hash, affected layer/direction, security policy and measured quality; it is **not a source-preserving fused kernel** and is OFF by default. No automatic removal of model safeguards or hidden transform under quantization.

## 9. Acceptance matrix P11/P13/P14/P15/P16

| ID | Fixture and required assertion |
|---|---|
| HYB-01 | Qwen GDN+QSA exact attention/index/StateRoot against original source |
| HYB-02 | GLM KDA/MLA/DSA/mHC full state and prefill/verify numerical checks |
| HYB-03 | MiMo SWA/global head/value/padding/mask valid; no padded-axis semantic state |
| HYB-04 | E8M0 scale codec and mixed EXL3 source golden; corrupt scale rejected pre-GPU |
| HYB-05 | MiMo transformed layer47 source pack finite and source/derived quality verified; wrong compensation fails |
| HYB-06 | Qwen MTP/prompt lookup/mixed policy exact target accepted prefix and sample RNG |
| HYB-07 | GLM eh_proj optional BF16 sidecar correct model identity and performance budget |
| HYB-08 | DFlash wrong tap/partial RoPE causes source golden failure |
| HYB-09 | DFlash missing sink bias, value scale, mask embedding or wrong bidirectional window causes source golden failure |
| HYB-10 | DFlash accepted prefix 0..k and sampled target correction/rollback full StateRoot |
| HYB-11 | ring wrap and speculation fork/replay 8K/128K/256K with source physical state |
| HYB-12 | four sessions share immutable draft/target weights but never mutable rings, history, RNG or grammar |
| HYB-13 | verifier R=15 vs R=18 and many expert counts; cost candidate changes only when measured |
| HYB-14 | prose/chat/code differing acceptance disables net-negative drafting without changing target |
| HYB-15 | derived quant vs original and draft vs derived-target equivalence reported separately |
| HYB-16 | unknown MiMo source op/DFlash sidecar rejects rather than falsely claiming full support |
| HYB-17 | source thinking/tool/template controls preserved; no hidden behavior prompt |
| HYB-18 | batch B1/2/4/8 near-tie numeric contract explicit, not unconditional byte parity |

## 10. Existing phase mapping, no new engine

- **P11–P14** register exact KDA/MLA/QSA/GDN/mHC/MiMo semantics and state, CPU oracle first.
- **P10/P11** implement verified target + draft ring/global cache and physical credits, not one KV rule.
- **P15** draft family variants MTP/DFlash/small model/lookup with exact verifier and fallback.
- **P16** measured MoE padding/row/expert-aware scheduling and dynamic draft policy.
- **P22** whole-model/derived quality, long contexts, numeric near ties, multiple tasks, device and real sampling.

All remain future design obligations; no new media/text model is implemented by this document.
