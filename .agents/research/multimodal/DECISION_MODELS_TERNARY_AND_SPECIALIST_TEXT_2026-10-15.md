# Research Track F — Typed Decision Models, Ternary Formats and Specialist Text Architectures

**15 October 2026 · design/research only.**  
**Models:** hosted typed-decision services, encoder decision-model variants, Qwen-derived joint-schema decision models, ternary-packed low-bit models, other source-verified low-bit/hybrid/linear attention model families. No locally executable source for the hosted services is claimed.  
**Key architectural question:** can novel task heads and tensor formats coexist with a familiar Transformer backbone without creating another engine? **Yes, if source math, task algorithm, tensor codec and runtime activation transformation remain independent and verified.**

## 1. Why decision heads need a native task kind

Decision-focused models answer a **finite question** about a state, not necessarily a natural language response. Examples: Is this transaction suspicious? Which of five tools should execute? Score whether an image matches a medical document? A model may return an option logit vector or probability for multiple schema questions in a single forward pass.

A generative text model can be prompted to produce JSON, but that is **not the same** as a dedicated trained head with calibrated yes/no/choice/score semantics. Never route a typed-decision request into ordinary language sampling and claim source-equivalent option probabilities.

## 2. Source variants and execution family

| Model | Source-verified nature | Required model/task semantics | Official source |
|---|---|---|---|
| Hosted typed-decision service | proprietary hosted decision-only model with schema questions and typed responses | external TaskServiceAdapter; source input state, choice/score/noul schema, provider version, probability meaning | provider documentation |
| Encoder decision model (English) | ModernBERT-large 421M non-AR encoder | text encoder + direct decision classification/scoring head, finite schema and text normalization | model card |
| Encoder decision model (multilingual) | mmBERT-base 322M multilingual, checkpoint context differs | language/processor, source encoder, schema output and per-variant rate/limit | model card |
| Encoder decision model (typed decisions) | ModernBERT-large with typed choice/score/yes-no tasks | multiple typed question heads or shared classifier, calibrated score only when sourced | model card |
| Joint-schema decision model (27B) | multimodal Qwen3.8-27B-derived backbone with joint schema attention/decision head | Qwen multimodal processor and joint schema scoring, many questions per forward | model card |
| Joint-schema decision model (lighter) | lighter Qwen3.5-9B-derived multimodal backbone and decision head | exact different checkpoint, source processor/schema, not interchangeable with 27B | model card |
| Ternary-packed low-bit model (27B) | Qwen3.8-derived hybrid attention topology but PTQ1_0/PQ2_0 ternary weights with required runtime Hadamard transform | source register for codec + activation transform + accumulation, familiar Qwen semantic backbone | model card |

**Source caution:** The joint-schema source head layout and specific foundation variants must be bound to their exact weight config; do not infer every Qwen checkpoint automatically has a joint decision head. The hosted typed-decision service is opaque and absent released executable model weights; do not import a guessed neural formula. The encoder decision-model variants are distinct registered checkpoints, not simply one model with a language flag.

## 3. DecisionSchema model

~~~rust
struct DecisionTask {
    source_state: TypedStateRef,                 // text/media/structured context
    schema: VerifiedDecisionSchema,              // stable question/choice IDs
    output_policy: RawLogitsOrCalibrated,
    model: ResolvedModelBinding,                  // exact source/processor/semantic revision
}
struct QuestionSpec {
    id: QuestionId,
    semantic_prompt: BoundedText,
    answer: AnswerDomain,                        // Noul, finite choices, scalar score
}
enum AnswerDomain {
    Noul,                                        // source-defined yes/no probability semantics
    Choice(Vec<ChoiceId>),                       // choices ordered, source schema bound
    Score { minimum: f32, maximum: f32 },       // scale+direction defined by source
}
struct QuestionResult {
    question_id: QuestionId, schema_version: SchemaId,
    raw_logits: Option<Vec<f32>>,
    scores: Vec<ScoreWithProvenance>,
    calibration: Option<CalibrationId>,
    finality: DecisionFinality,
}
~~~

**Schema invariants:**
- Question IDs, answer option IDs and option order are versioned; sorting choices alphabetically may change model output.
- Raw logits are distinct from softmax probabilities and temperature/calibrated probabilities.
- Calibration metadata may include source calibration dataset, validation temperature/isotonic/logistic link, thresholds and task domain; absence means "not calibrated", never confidence=0.9 invented.
- A score may not be a probability; preserve units, range, direction and trained head source.
- 20 questions can be jointly processed only if checkpoint supports joint schema; otherwise register a bounded batch of separate score tasks.
- Nonfinite logits, invalid option count or missing schema answer must reject publication before StateRoot commits.

## 4. Encoder decision source trace

1. Bind exact variant (English, multilingual or typed decisions), tokenizer, language/profile and allowed schema.
2. Normalize bounded textual context and question descriptions per source processor; no arbitrary text truncation if it changes answer.
3. Execute ModernBERT/mmBERT encoder through shared attention+norm+GEMM providers.
4. Apply checkpoint-trained direct scoring or classification heads (not a causal token sampler), respecting question representations and answer type.
5. Produce option logits/score distribution and apply only source-validated calibration/normalization.
6. Validate schema cardinality and output value ranges, then atomic TaskCoordinator publish the decision result.
7. Benchmark batch sizes/memory, latency and actual per-domain accuracy. Device placement can be CPU/accelerator independently of model type.

**Negative:** no decoder LM available → no need to fabricate BOS/EOS text or KV decode loop. Router may still select between encoder decision models if caller authorized quality/language policy, but must report actual variant.

## 5. Joint-schema decision head trace

1. Bind Qwen-derived multimodal backbone+vision processor and the source-trained joint schema head as verified ModelAssembly.
2. Validate input text/image roles and finite schema (questions, options, scoring semantics).
3. Encode/source-pass multimodal context using checkpoint-correct attention/recurrent state.
4. Feed shared representations and all requested bounded schema questions to joint scoring head within one forward where source-trained.
5. Obtain raw per-option logits, apply source-specified normalization, calibration only if exact head supports it.
6. TaskCoordinator publishes complete question/answer schema atomically; output is a typed decision record rather than free-text JSON.
7. Share immutable foundation weights where absolutely identical to a local Qwen model, but separate joint-head/processor semantics and per-session state. Never assume plain Qwen3.8-27B results equal the joint-schema model.
8. Return device/model/mode receipt and schema revision for audit/replay.

**Specialization:** joint question scoring and medium/high batch matrix ops; source sparse/multimodal attentions may use existing Qwen providers if equivalence proven. A new decision head is a semantic op/reference plus model component, not an independent LLM inference server.

## 6. Hosted typed decisions

Current hosted service official description: state plus typed questions; source response may include probability/choice/score with provider model alias/version. API requires server-held bearer credentials and task-specific choice schema. Treat this as **opaque hosted TaskServiceAdapter**:
- consent for source state/image/doc egress and geography;
- model alias revision uncertainty; pin version when provider supports it;
- provider-calibration semantics only as officially promised;
- token/usage/cost, retries and tenant limits;
- typed response schema validation and actual service errors, rather than missing tensor logits invented locally;
- never mix local GPU source evidence with hosted API measurement.

A typed hosted decision and a local encoder decision model may share a public decision request/response schema where semantics truly align, but if one returns only a probability and the other returns raw logits, preserve that distinction.

## 7. Ternary low-bit source architecture: why simple Qwen format recognition is dangerous

The ternary low-bit model's source card describes 27.36B Qwen3.8-derived topology, about 75% linear attention / 25% full attention, with ternary packed weights in PTQ1_0 or PQ2_0; matching blockwise Hadamard activation transform is essential. Stock loaders can mishandle a superficially similar low-bit format and silently produce garbage without an error.

- Ternary values in {-1,0,+1}, g128 per-group scaling; codec forms have different bit packing and speed/memory tradeoffs.
- The source card states 1024-value block Hadamard rotation with fixed sign scheme folded into stored weights; a **matching transform MUST be applied to runtime activations** before dot.
- Codec-specific group scales/packing/unpacking, zero padding, signed representation and exact numerical accumulation are independently versioned.
- Vision encoder can be separate Q8_0 mmproj component; do not load/cache it for text-only tasks unless source assembly requires it.
- Source GGUF tensor type PTQ1_0/PQ2_0 must fail gracefully on generic stock Q4/Q2 kernels. A plausible-looking output is not an acceptable test.

This is a first-class proof of four orthogonal axes: **same semantic model topology; novel source physical codec; novel activation basis transformation; independent kernel backend specialization**.

## 8. Ternary TensorEncoding design

~~~rust
struct TernaryCodec {
    ternary_packing: TernaryPack,               // PTQ1_0 / PQ2_0 exact source
    group_size: u32,                           // source declared 128
    scale_type: ScaleEncoding,
    basis: HadamardBasisId,                    // source block 1024 and sign scheme
    activation_transform: VerifiedTransformId, // mandatory at runtime
    accumulation: NumericReductionContract,
}
struct QuantizedMatvecRegion {
    semantic_weight_id: TensorIdentity,
    encoded_weights: VerifiedByteSpan,
    input_activation: LogicalTensorRef,
    codec: VerifiedCodecId,
    transform: VerifiedTransformId,
    legal_kernel: SourceQualifiedKernelId,
}
~~~

Reference math:
- parse source weight blocks, scales and transform metadata with checked sizes and source license/version;
- evaluate original ternary + activation transform semantics in small scalar reference; verify independent golden vectors;
- on physical CPU/GPU kernels, match exact source numeric/quality contract and quantizer outputs; never omit Hadamard or convert it into an unqualified approximation;
- optimize direct bitpacked GEMV, activation Hadamard fusion, tensor packing and placements without expanding entire weights unless memory/perf justified;
- support optional separate vision component under exact processor identity and media port.

## 9. Extensibility to other specialist text models

Although the ternary low-bit model is a concrete adversarial target, the same architecture must accept:
- BitNet-style ternary/1.x bitlinear ops with source activation normalization;
- mixed FP4/NVFP4 weights with per-block/per-channel scales;
- linear/DeltaNet/KDA/GDN recurrent states distinct from Transformer KV;
- learned indexers/compression/non-token memory state (Qwen flash/GLM/DeepSeek-like variants);
- LoRA/adapter/rank-specific deltas with exact model/source/key identity;
- multimodal Qwen-derived variants with vision/acoustic processors;
- block-sparse experts and dynamic output projection where numerical identity preserved.

New semantics get a registered SemanticOp and ReferenceOp, new packed formats get a VerifiedCodec, new state gets a StateDomain; router, TaskCoordinator, server and physical arbiter remain unchanged.

## 10. Negative test matrix

**MM-DEC-01:** Encoder decision model 3 questions via encoder, no text sampling; exact option labels/order.  
**MM-DEC-02:** Joint-schema 20-question head, same source image/text representation and typed schema, atomic result.  
**MM-DEC-03:** option logits [-100,100] with stable normalized softmax; no NaN probability; calibration absent remains absent.  
**MM-DEC-04:** model lacking true probabilistic calibration cannot claim reliable confidence, even after softmax.  
**MM-DEC-05:** edited question option set after task start invalidates running schema; stale TaskRoot refused.  
**MM-DEC-06:** hosted typed-decision service unauthorized egress refused before API call; timeout job status Unknown instead of fabricated answer.  
**MM-BON-01:** source packed PTQ1_0/PQ2_0 blocks decode against golden reference and exact scales.  
**MM-BON-02:** missing Hadamard transform fails at model load/plan, not after poor generated tokens.  
**MM-BON-03:** wrong Hadamard block width/sign scheme fails source golden vector.  
**MM-BON-04:** generic Qwen architecture can share attention semantic graph without reusing wrong tensor codec.  
**MM-BON-05:** CPU scalar reference vs native bitpacked SIMD/GPU at multiple original source-trained vectors and all shape guards.  
**MM-BON-06:** two concurrent text sessions share immutable packed source weights but not recurrent/grammar/RNG state.  
**MM-BON-07:** vision tower absent text-only works where source permits; image input without verified vision mmproj refuses.  
**MM-BON-08:** memory reporting accounts exact compressed bytes AND Hadamard transform scratch, not just disk size.

## 11. Qualification and implementation order

1. Mock typed decision schemas and direct-logit normalization with source-free deterministic fixtures.
2. One small source-verified encoder+decision head CPU model, then source-qualified GPU; use no new serving engine.
3. Exact joint-schema model assembly with Qwen head integration and source multimodal state; verify original trained checkpoint on licensed hardware.
4. Ternary PTQ1/PQ2 source parsers, Hadamard/reference scalar kernel and source-golden vectors; then specialized devices/fast path.
5. Hosted service adapter only when user/admin explicitly connects account, grants egress/consent and job schema; no arbitrary external provider calls from native GPU.
6. Other low-bit text models follow same codec/transform/semantic policy.

Existing user-mandated practical Qwen3.5→Gemma4→Qwen3.8 source CPU/GPU order is preserved; this is a later extension plan and architecture acceptance sequence.

## 12. Original sources reviewed

- Hosted typed-decision service introduction.
- Hosted typed-decision service API quickstart.
- Encoder decision model source/config.
- Joint-schema decision model source card (27B).
- Joint-schema decision model source card (lighter variant).
- Joint-schema implementation reference.
- Ternary-packed low-bit model source card.
- Ternary-packed model demonstration source.

Source claims reviewed on 15 Oct 2026, subject to exact checkpoint/license/version verification. No decision or ternary low-bit model was run or downloaded in Zerogravity during this research.
