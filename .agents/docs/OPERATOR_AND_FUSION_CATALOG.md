# Operator/region contracts, fusion patterns and model execution catalog

**15 Oct 2026 — batching specialization boundaries:** [Joint serving contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) adds the physical shape dimension to each existing source-qualified closed region candidate: compatible active sessions and rows, padded expert partitions, optional target verification depth, routed expert histograms, layout/precision, device event dependencies and scratch. Merge matrix rows only when all source model/codec/operator effects and numeric mode match; other models may share scheduling and immutable encoder but not an invalid fused matmul. Quant variants, cache warmth and draft acceptance inform the cost model, not semantic expert choice. Test SERVOPT-21..26 against reference state and true device submissions.


**15 Oct candidate-registry refinement:** [extreme hybrid semantic and draft rules](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) and [quant source codec](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md) add **E8M0 source scale and paired convertible FP16 intermediate** as explicit tensor/semantic variants, plus derived-trellis/decode regions and separate MiMo attention. Norm→activation packing→quant GEMV, GDN/KDA recurrence and residual mixer, MLA indexer/tile, MoE dispatch→grouped quant expert→reduce can fuse only with original source effects, full state and legal reduction order. No whole-layer optimization can lose source mHC/Sinkhorn, QSA/DSA index or draft hidden-tap dependencies. Physical row tile size is KIR, not a model operator.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


This is the implementer's companion to `SEMANTIC_IR_AND_COMPILER.md` and `MODEL_SEMANTICS_AND_COMPATIBILITY.md`. It lists the semantic and execution choices a model adapter/kernel author must make, not fabricated finalized formulas. Exact mathematical equations must be derived from the pinned official model source and verified CPU oracles before implementation.

## 1. Canonical operator record

Every operator registry entry must include:
```text
operator_id, schema_version, aliases/model-family bindings
inputs and outputs: logical dims, dtype, quant stored vs compute layout
semantic parameters: eps, scale, mask, RoPE, gating, routing, compression...
state inputs/outputs + position/epoch and effects
preconditions: shape, sparse IDs, context, alignment, stable tensor identity
reference CPU function + external independent oracle fixture
backend kernel candidates and legal shape ranges
numerical rules: accumulation, rounding, selected-set invariants
allowed fusions/layout conversions/snapshot cuts
required scratch/lease/event and graph capture eligibility
test categories and performance profiles
```

No opaque "custom op" without a reference/state contract. If the full effect semantics are not representable, reject as unsupported instead of compiling a seemingly plausible graph.

## 2. Dense linear / embedding

**EmbeddingGather** maps token ID to a row; must handle model-tokenizer vocabulary, special token ID limits, quantized embedding storage where supported. N-gram/Engram lookups are separate ops.

**Matmul/Matvec/QuantizedProjection:** logical matrix `[out_dim,in_dim]` independent of stored block quant, padded K and transposed device layout. Backend may fuse dequant inside projection or prepack weights. Prefill GEMM, decode GEMV and verify grouped MMQ variants. Need A/B shape/error tests and explicit accumulator/rounding.

**RMSNorm/LayerNorm:** epsilon and reduction dtype, scale/bias order; fusion with quantization only if numeric policy passes independent oracle.

**Activation/GatedFFN:** GELU/SwiGLU/other exact model activation, gate/up/down row geometry and rounding, not interchangeably optimized.

**RoPE/MRoPE/Position:** position ID and axis/sections, static extension factor; long-context changes can affect quality. Need exact image/3D position mapping.

### Candidate region: Norm + Q8 activation packing + GEMV
Effect: pure tensor (unless intentionally fused with residual state). Inputs: current activation/weights. Output: same logical projected rows and declared numerical tolerance. Benefit: avoid extra activation write/read and kernel launch. Risk: register spill, FP32 rounding, block tails. Fallback: separate norm, pack and projection with explicit scratch.

## 3. Attention families

### Full causal attention / GQA
Q/K/V heads relation, causal mask, optional sliding window, position rotation, KV storage (paged). Prefill large attention tile, decode KV/gather, verification multirow. Fusion candidate QKV projection+RoPE+cache store where ordering/capture permits; dynamic KV page pointer reads require leases.

### Sliding-window / alternating local-global
Window is semantic (not arbitrary heuristic). Must preserve exact local positions and boundary; global layers cannot reuse same narrowed history. Gemma variants' attention scaling/positions declared by model recipe. Test sequence >window, uneven chunks and prefix fork.

### Qwen Sparse Attention (QSA)
Compressed/indexer rows, pooling, scoring, top-k block selection and masked attention; choose exact selection ID algorithm, tie ordering and scalar precision. At 128K sparse selection must remain on GPU for peak `gfx1151` plan. Separate candidate fused selected attention vs selector→attention; bounded scratch, no approximation to lower top-k without explicit quality mode.

### GLM latent MLA + DeepSeek Sparse Attention
Latent `c_kv` compressed state, NoPE where model says so, query absorbed projection option versus expanded K/V fallback (different numerical/physical layouts). Sparse indexer head geometry/top-k differs from QSA; do not use Qwen indexer kernel unmodified. Effect: append valid latent/indexer state and publish causal compressed rows. Legal conversion only at checkpointed frontier and test against model reference.

### DeepSeek time-compressed attention (CSA/HCA)
Raw most recent window and compressed rows indexed by time ratio (4 and 128 in DwarfStar research), some layers have indexer top-k. Distinct state update for compression pool and raw ring, exact causal reveal of completed compressed blocks. Never infer that ratio-128 compression is a performance tweak; it is trained model semantics.

### Candidate region: index score+topk+attention tile
Pros: fewer global scratch writes and launches. Cons: divergent dynamic top-k, register/LDS, numerical score ordering. Must test masked/empty selection, selection ties and long-context full logits, not just attention output relative error.

## 4. Recurrent and hybrid families

### Gated DeltaNet
State matrix and per-layer short-convolution history, gates, input projections; chunk prefill scan and one-step recurrence have distinct efficient kernels but semantically same transition. Model-specific heads and gating. Every speculative target row must have post-row checkpoint candidate, or an exact recomputation strategy. Causal dependency forbids parallelizing recurrent tokens as if independent ordinary attention.

### Kimi Delta Attention
Fine-grained per-channel gates/decay and potentially lower-bound scalar as released config requires; not GDN scalar-gate approximation. State large enough that eager snapshot per draft row may cost substantial memory. Fused scan should exploit algebraic associativity only if exact model recurrence and numerical policy allow. Long-context and chunk-edge parity mandatory.

### Hyperconnections
Qwen-style gated residual and GLM/DeepSeek mHC manifold-constrained mixing are distinct. mHC Sinkhorn iterations/normalization should be separate canonical op; fusion might change rounding; test full width, multiple stream slices and tool continuation quality.

### Candidate region: recurrent step + conv history + residual
Benefit: reduce small GPU launches and intermediate activations; preserve versioned recurrent state and snapshot access. Risk: no rollback snapshot at intermediate draft positions or scratch register spill. Legal only with explicit state outputs for each committed/provisional row.

## 5. MoE families

**Route:** logits→score transformation→top-k selected IDs/gates (signed/sigmoid/softmax/routed bias and scaling). Some models use auxiliary normalization; model recipe must dictate formula.

**Group:** token row assignments compacted into `expert -> row list`, with duplicate selected experts across rows computed together; preserve original row and gate association.

**RoutedExpert:** gate/up/down projections quantized per tensor; small decode and large prefill kernels can differ. Padded stored dims not semantic computation dims.

**SharedExpert:** often computed every token, may overlap selected expert IO/computation if dependencies permit; if output merged after routed contribution, reduction order/precision explicit.

**Aggregate:** deterministic or fast atomic reduce based on numerical contract. Atomic FP32 order may change logits, so exact-replay path uses stable ordering.

### Candidate fusion plans
- Router+selected IDs+GPU pointer table lookup: no host round trip on hit.
- Grouped expert gate/up and down: exploit input activation reuse and fewer launches, with scratch and occupancy budget.
- Shared expert running on independent stream while expert miss IO fetches missing routed weights.
- Prefill grouped GEMM vs decode GEMV/low-row MMVQ, choose measured row distribution.
- Speculative verification grouped experts across candidate rows and independent sessions.

**Hard guard:** expert-cache eviction cannot overwrite a slot with any still-queued reading kernel. Fused region must declare active leases and events or illegal.

## 6. N-gram / Engram row lookup

Operator has read-only huge backing file, small input token-history state and deterministic row-hash/lookup semantics. CPU may compute hash with bounded `pread`; GPU projection consumes tiny staged rows. Cache duplicate requests by `(artifact, row)`, no blindly mmap entire 95/189 GiB table as resident. A missing/corrupt row is an IO error that invalidates provisional model step; no zero-filled default or reused scratch.

## 7. Output/sampling/drafting

**LMHead:** large vocab projection with optional quant, shape-tail checks, model logit scaling and precision. **Sampler:** temperature/top-k/top-p/min-p/penalties and RNG, result emitted only upon state commit. **Drafter:** MTP/DSpark/small model/prompt lookup propose tokens, not authoritative outputs. **TargetVerifier:** computes logits/states for proposal rows, applies mathematically sound correction. Fusing sampling onto GPU is allowed only if matching numerical policy and RNG state/stop logic.

## 8. Operator capability matrix example (PROPOSED)

| Operator | CPU ref | Generic GPU | Special gfx1151 | Special CUDA | Notes |
|---|---|---|---|---|---|
| RMSNorm+project | required | candidate | wave32 fused | tensor-core suited | numerical rounding |
| Dense GQA | required | candidate | attention tile | Flash-style tile | paged KV/graph |
| GDN | required | candidate | recurrence+conv fused | recurrence fused | rollback every draft row |
| KDA | required | candidate | per-channel recurrence | per-channel recurrence | no scalar approximation |
| QSA | required | candidate | GPU top-k/sparse | sparse indexer | causal selection |
| Sparse MLA/DSA | required | candidate | latent/index tile | latent/index tile | NoPE and absorbed vs expanded |
| DeepSeek CSA/HCA | required | candidate | time-compressed tile | time-compressed tile | layer dependent ratios |
| Quant MoE | required | candidate | IQ2/Q2/Q4 grouped | FP4/INT8/Q kernels | route math fidelity |
| mHC | required | candidate | fused if valid | fused if valid | Sinkhorn/numeric policy |
| Ngram/Engram | required | CPU/IO candidate | async row stage | async row stage | disk table |
| MTP verifier | required | candidate | 2–8 grouped rows | grouped WMMA | exact commit |

All entries are implementation targets; **nothing in this table indicates zerogravity already runs them**.

## 9. Reference model and independent data flow

For each operation supply sample inputs and output SHA from official model where possible. If model code lacks a directly callable equivalent, implement independent pure scalar mathematical oracle and compare with at least two full-model reference points. Include state bytes or normalized state checksum after transitions. For quant, same stored bytes and dequant scales must be used by both candidates. Use official quality scores but avoid claiming official benchmark parity with different checkpoint/quant.

## 10. Region and fallback execution example

```text
SIR:
    norm -> gated_read -> GDN(state#7) -> gated_write -> router
        -> route_expert* + shared -> reduce -> residual

EIR candidate:
    fused_gdn_hc_region(state#7..state#8)
    routed_moe_gpu_region(quant IQ2, top10, rowcount 1..8)

If second kernel unavailable:
    preserve canonical activation at region cut
    generic routed MoE region executes known exact route/output
    state#8 remains owned and coherent
    continue next layer without full-model fallback
```

A fallback may require materialization and thereby lose performance, but **must not lose state correctness**.

## 11. Complexity budget

Add a new candidate only if it has a distinct valid hardware/workload regime. Avoid permanent flags for all experimental fusion combinations. Registry should expose a bounded priority list of winners and one portable fallback. Rejected variants may live in research fixtures but should not complicate release binary.

## 12. Missing official formula checklist

Before first implementation of GDN, KDA, QSA, DSA/MLA, mHC, CSA/HCA, n-gram/Engram and MTP:
- pin official model source/config to commit;
- derive precise equations and tensor shapes from released weight naming and reference executable;
- create tiny scalar oracle and actual-model vector fixture;
- document quantization sensitivity and state transition;
- only then write extreme kernels.

This stops model architecture names from becoming unsupported placeholders.

## 2026-10-10 integrated v2 addendum — effect-aware kernel and fusion ownership

Every operator catalog entry now has **four separate required contracts**: (1) exact SIR semantic formula and state effects, (2) RIR legalization/fusion/precision preconditions, (3) KIR backend candidate shapes/ISA/scratch and reference numerics, (4) CIR memory lifetime and async event/I/O dependencies. An operator is not performance-qualified simply because a generic graph can represent it.

Reusable operator families must remain first-class at SIR: ordinary GQA, sliding/full, GDN, KDA, Qwen sparse attention, GLM MLA/DSA, DeepSeek time-compressed attention and residual/mHC, routed+shared experts, model-native ngram/Engram, multimodal encoders and pooling. Only RIR may decompose these into lower-level primitives; a generic algebraic tensor rewrite must not erase a critical recurrent or indexing state effect.

**Candidate region boundaries:** fused norm→activation-quant→projection; QSA/DSA indexer select→masked attention; GDN/KDA recurrence+short-conv; router→stable counting-sort→grouped MMVQ/GEMM→deterministic reduction; trained MTP target verify+state snapshots. Each declares a canonical boundary and COW/convert fallback that preserves full state. For `gfx1151` grouped MoE, make `route` GPU-side, but don't assume greater group size wins when expert IDs are almost unique. For graph capture, dynamic expert table pointer generations must remain valid.

**Compile-stage tuning:** pure rewrite and target kernel candidates can be searched offline; avoid compiling dozens of combinatorial fusion variants per live request. A verified serving-mode-aware autotuning result means KernelPlan needs eager-versus-graph execution mode in its key. A selected candidate must beat baseline **at full-region or end-to-end level**, not merely microseconds in isolation. Custom Rust whole-model kernel accepted if and only if region semantics/state/numerics/resources are verified and the portable decomposition remains usable.

**New extension demonstration:** add a new trained sparse attention indexer under its own operator schema and CPU oracle; reuse existing GPU top-k/packed GEMM candidates where mathematically compatible, with no changes to `EngineCore` or tokenizer/HTTP.


## 2026-10-12 integration — `ProjectionRowSlice` and non-token stage optimization

New RIR candidate `ProjectionRowSlice` decomposes a pure linear projection's **disjoint output row groups** among capable compute owners. Precondition: all source rows covered once, compatible quant/scales/bias, independent per-output channel math until explicit join, no relocated causal state effect or whole-output layer norm before missing rows. `AssembleProjectionSlices` restores logical row order (possibly direct-to-layout), and downstream GDN, attention, shared FFN gates remain model-correct.

Source-demonstrated first slices: (1) GDN z rows in 16384-row SSM input, up to 6×1024 (default fork); (2) attention Q+gate of 16 heads via 8×1024 rows; (3) shared expert gate/up projections 2× eligible NPU GEMM. XDNA2 kernel is fixed 4096×2560×1024 BF8 example; GPU computes remaining rows. The exact NPU result first-use phase differs for each operator. Region choice may be GPU-only, GPU+NPU or other computation owners; no hidden hard-coded device name in SIR/ModelRecipe. Unsupported shape/batch/kernels/unsafe SoC → GPU-only lawful candidate.

**Separate mathematical correctness from quant encoding:** source model weights may be Q8_0, but NPU compute uses `v8bfp16ebs8` groups. Export per-op max error, first-use dependent state/logits and model quality under a declared numeric profile. Treat `TaskProgram::Vocoder`, `TaskProgram::DenoiseStep` and encoder pooling as future operator/task extensions with independent noncausal and iterative-state contracts, not as generic token projections by default.

## v4 refinement — useful op granularity and precise region substitution

Maintain high-level semantic op identity for stateful full/sparse/compressed attention, GDN/KDA/indexer, MoE routing, residual streams and model-specific media operators. Within a **closed region**, the compiler may lower into pure tensor primitives, fuse norms/quant/GEMM/router kernels, split independent projection output rows and use architecture-specific whole-layer kernels. **RIR/KIR are transient analysis/codegen views**; only canonical SIR and compiled ExecutableBundle need enduring identities initially.

A region candidate declares exact input/output, state effects, dynamic ShapeGuard/ragged offsets, tensor quant vs lossy compute encoding, scratch/leases and explicit safe fallback. A kernel may fuse multiple semantic operators but cannot hide state writes or publish TaskSession root. Numeric/effect legality and first consumer readiness precede adoption; measured whole model/TaskDriver performance dominates microkernel timing.
