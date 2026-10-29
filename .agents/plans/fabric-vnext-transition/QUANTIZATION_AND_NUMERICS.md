# Fabric vNext transition — quantization, codecs, numerical equivalence and ownership

**Contract status:** specification for code movement and new clean Rust type boundaries; source code only created after separate user instruction. Informed by commit 405a3e6 format homes, live artifact/CPU/GPU modules, .agents/plans/pipeline-composition/FORMATS.md, and the original complete Fabric vNext/derived-quant design. A directory, parser, source codec or CUDA image is NOT whole-model execution proof.

## 1. The six independent dimensions

Do not collapse quantization into a single `QuantKind` label used to infer everything else.

1. **Source container** — GGUF, safetensors/index shards, NeMo/OCI source assembly, etc.; metadata, configuration, tensors and process/processor provenance. The container is not an inference algorithm or physical kernel.
2. **StoredTensorEncoding** — exact mathematical source tensor byte representation: ScalarDType (F32, F16, BF16) OR exact named QuantScheme such as Q4_0, Q4_1, Q4_K, Q6_K, Q8_0, future NVFP4 or EXL3 trellis/codebooks. Stores block length, bytes/block, endianness, signed codes, scale/zero tables, padding and exceptional bits. Never assume any format from bit width alone.
3. **NumericalContract + ActivationEncoding** — original mathematical operator and per-phase/batch reduction/cast/accumulation/profile, intermediate dynamic Q8_0/Q8_1/Q8_K coding, source-specific rounding, FMA/parallel lane order and nonfinite state. These are not implied by source encoding. The CPU reference can match GGUF's original arithmetic rather than a dequantized-f32 ideal dot.
4. **PreparedTensorLayout** — the *physical representation* after reversible lossless packing: row/column/group/stride, SIMD/tile/WMMA descriptors, device alignment and view lease/generation; a lossless repack changes buffer layout but not weight semantics. If a cast/transform changes math, it is a separately qualified numerical strategy or derived artifact.
5. **Physical QualifiedKernelCandidate** — selected source/semantic/quant/activation/numerical/shape/batch/device/driver/compiler/ABI/capture+state candidate; actual events, buffers/reader liveness, preflight memory and cancellation. A kernel source body existing does not make it production qualified.
6. **DerivedModelArtifact** — a *different* source artifact obtained by lossy weight quantization/calibration, exact source lineage, quantizer version, per-tensor transformation, held-out task quality and licensing. Must never silently replace the user's original BF16/FP8 source or reuse unrelated state/prefix keys.

**Physical tensor placement and IO** is orthogonal to these six: immutable SourceTensorId may materialize in device RAM/UMA/host/SSD/remote, but changes are separately chargeable TransferPlans with verified source event readiness. SourceIdentity for a live original issuer is distinct from content hash; exact matching hashes cannot mint the same source owner, resident pointer or qualified physical permit.

## 2. Proposed exact Rust types (sketch, keep small)

~~~rust
pub enum ScalarDType {
    F32, F16, BF16, // later exact scalar types when sources exist
}

pub enum QuantScheme {
    GgufQ4_0,
    GgufQ4_1,
    GgufQ4_K,
    GgufQ6_K,
    GgufQ8_0,
    // Add only implemented + versioned schemes. Q1/2/3 and NVFP4
    // remain *declared future capabilities*, NOT executable enum variants
    // until exact codec/source validation exists.
}

pub enum StoredTensorEncoding {
    Scalar(ScalarDType),
    Quantized(QuantScheme),
}

pub struct QuantTensorLayout {
    pub logical_shape: Shape,
    pub source_encoding: StoredTensorEncoding,
    pub blocks: BlockGeometry,
    pub scales: ScaleEncoding,
    pub source: AuthenticatedTensor,
}

pub struct NumericalContract {
    pub accumulation: AccumulationPolicy,
    pub activation: ActivationEncoding,
    pub reduction_order: ReductionPolicy,
    pub rounding: RoundingPolicy,
    pub nonfinite: NonfinitePolicy,
    pub semantic_profile: SemanticNumericIdentity,
}

pub struct PreparedTensorLayout {
    pub logical: SourceTensorId,
    pub layout: PhysicalLayoutId,
    pub destination: PhysicalMemoryDomain,
    pub source_owner: SourceOwner,
    pub lease: BufferLease,
    pub ready: EventSet,
}

pub struct DerivedModelArtifact {
    pub original: ArtifactDigest,
    pub derived: ArtifactDigest,
    pub quantizer: QuantizerRecipeIdentity,
    pub calibration: CalibrationProvenance,
    pub quality: QualityReceiptId,
}
~~~

These are conceptual fields/names, not a requirement for a heap object/trait per tensor, nor permission to claim an unsupported format. Store small exact enum/record contracts in shared core, real decoding in artifacts, CPU arithmetic in CPU, GPU kernels and source-specific ABI in GPU, and model role/geometry in family. Model-recipes use generic source and math interfaces; router/HTTP never parses Q4 blocks.

### Support must be multidimensional, not one boolean

~~~text
source: recognized / bytes verified / tensor shape validated
codec: scalar decode / supported activation + operator
backend: source and shape compile / real device Ready / state safe
model: trained source whole-forward / full task quality
serving: scoped request API, output and cancellation
performance: matched complete-request qualified
~~~

For example Q4_K currently has source recognized + scalar decoder, but source-exact CPU matvec deliberately Unsupported and no general trained model eligibility. It remains Unsupported at that level after refactor. A Q6_K GPU implementation on one backend does not qualify an independently different HIP/Vulkan backend or CUDA graph profile.

## 3. Present source implementation map at fadf2b1

| Exact scheme | Artifact bytes/decoder | CPU source arithmetic | GPU source work | Whole-model support |
| --- | --- | --- | --- | --- |
| F32 | Exact original scalar codec | General source projection/reference and ordered lane variants | Backend-selected, source/profile-specific | Conditional actual model/run |
| F16 | Exact scalar codec | Source/transform selected mathematical path | Some physical operations; numerical/ISA dependent | Conditional |
| BF16 | Scalar exact decode; real arithmetic located outside new precision folder | Explicit BF16 numerics and vector paths | Native/CubeCL/CUDA/HIP/Vulkan partial variants, not universal | Source/driver/shape dependent |
| Q4_0 | 32 elements / 18 bytes exact q4 decoder | Original Q4_0 x dynamic Q8_0 and 8-row repack | Existing q4_0 kernel, source qualifications | Qualified by exact model/backend tuple only |
| Q4_1 | 32 / 20, signed offset and scale | Q4_1 x Q8_1 | Existing specialized source projection | Model/backend/shape specific |
| Q4_K | 256 / 144, scalar decoder exists | **Deliberate Unsupported** for unqualified source Q4_K x Q8_K, despite reference dequantized diagnostic | No corresponding complete GPU kernel in source-format map | NOT granted |
| Q6_K | 256 / 210, q6 decoder | Signed dynamic Q8_K/laned original source arithmetic | Existing source projector q6_k | Scope-specific |
| Q8_0 | 32 / 34, q8 decoder | Q8_0 x Q8_0; distinct pure f32 reference/OpenBLAS contract | Existing q8_0 kernel | Scope-specific |
| Q1 family | README home only | No implementation | No implementation | No |
| Q2 family | README home only | No implementation | No implementation | No |
| Q3 family | README home only | No implementation | No implementation | No |
| NVFP4 | README home only | No implementation | No implementation | No |
| EXL3/QTIP/trellis/codebook | Design/research only | Not qualified | Not qualified | No |

Location notes:
- core/src/quant.rs currently recognizes F32/F16/BF16 and exact Q4_0/Q4_1/Q4_K/Q6K/Q8_0; from_ggml/from_safetensors have checked Unsupported for other formats.
- artifacts/src/quant.rs uses q4.rs, q6.rs, q8.rs and f16/bf16 converter functions. New q1/q2/q3/nvfp4/precision/bf16 directories hold README placeholders, not executable Rust.
- CPU source.rs/projection.rs handles original activation quantization and source-reduction, even though new cpu/src/source/quantization/q* directories hold only README. Do not invent implementability from those folder names.
- GPU src/quantization/* actually groups activation, fused, batch and exact Q4_0/Q4_1/Q6_K/Q8_0 source projection code. CUDA/HIP/Vulkan body and ABI proof is separate.
- Family source roles are in model-Qwen/model-Gemma, no quant format is allowed to invent valid tensor/model geometry.

## 4. Immediate structural milestone S0 for encoding

Create compile-visible types/contracts under core/encoding and core/numerics and exact parser/codec namespaces under artifacts. Existing QuantKind and old module names may remain internal while moving code in S1, but do not freeze them into a new public API. The facade should expose source/codec capability inspection and explicit refusal. All new placeholder paths must fail closed if called; don't add dummy "Q1" or "NVFP4" decoders returning fake zeros.

Recommended behavior during S0:
- Current working scalar decoder/CPU quant existing APIs still callable, or a narrow forwarding implementation to them under new public facade (temporary within tranche). Source parse/refusal unchanged.
- New codec/quant branches not yet migrated are module stubs with typed Unsupported, never todo! in an ordinary call path.
- Exact codec schema is a versioned type identity, not a string lookup at each matrix multiply; direct kernel selected once at prepared region.
- No codec-specific model/runtime engine, allocator, HTTP route or standalone scheduling actor.
- Add an inspect-only codec capability table that honestly reports recognized format, scalar decode, CPU kernel, GPU kernel and whole-model readiness as distinct fields.
- Keep old data/normalization/test conversions until next code moving units, no accidental change in numerical expression ordering.

## 5. Code migration S1 — physical implementation, not just renames

Small logical units (can commit per coherent group or as one S1 with passing focused checks):

S1.a Artifact codec exactness:
1. Introduce core ScalarDType vs QuantScheme/StoredTensorEncoding with explicit source GGML code mapping and checked byte-length.
2. Move original quant decoder q4/q6/q8 scalar functions into artifacts new codec namespace, preserving F16/BF16 nonfinite/round-to-even and all source byte-size bounds exactly.
3. Migrate parser/SourceTensorRef/source weights/quant tests in one source-integrated unit. Keep only one actual source decoder implementation, delete old aliases after callers updated.
4. Add unknown/malformed/truncated/overlap and original block-format fixtures; keep Q1/2/3/NVFP4 Unsupported.

S1.b CPU numerical contract:
1. Separate pure f32 dequantized reference from source-exact GGML Q4_0xQ8_0, Q4_1xQ8_1, Q6_KxQ8_K, Q8_0xQ8_0 and source BF16 calculations.
2. Move real dot/repack code into cpu/quantization or precise owned operation modules; preserve eight-lane reductions, cast/power/subnormals/negative-zero and AVX2 source profiles. No new numerical tolerance due to file move.
3. Retain deliberate Q4_K source execution refusal; its scalar decoder does not grant operator capability.
4. CPU tests compare actual original golden vectors and malformed/unqualified source guards, not merely converted f32 approximations.

S1.c GPU packed operation ownership:
1. Keep existing gpu/quantization implementation in place if it already meets owner contract; relocate only new module exports and source binding description, not every .cu file.
2. Implement adapter to semantic SourceTensorEncoding/ActivationEncoding + numerical profile + typed prepared kernel guard; no model named dispatch or hidden code-image change.
3. Retain q4/q6/q8 kernel unit fixtures and per-CPU/CUDA/HIP/Vulkan separate qualification marks; never transfer permits on feature-name equality.
4. Confirm encoded/captured queued readers and physical buffer generations; do not evict source-backing from a prepared direct packed kernel.
5. Any future Q1/2/3/NVFP4/EXL3 remains separate feature expansion AFTER transition acceptance.

## 6. CUDA GDN changes and kernel profile crosswalk

The 27-Oct CUDA GDN work explicitly separates a backend configuration (GdnKktKernel variants, GdnRecomputeKernel, GdnStateKernel, GdnOutputKernel) and TextGdnImplementation from historical finite qualification table recipes. This should remain **a source/phase/ISA/numerics-qualified physical candidate**, not an alternative model-specific execution engine.

- User-facing model family supplies Qwen source operator/phase/GDN geometry and parameter/buffer state roles.
- Backend supplies reusable parameterized GDN implementations and actual code/ABI identities.
- Compiler/preparation joins only compatible SourceCall and GdnKernelConfig and checks full phase stage implementation prior to H2D or root reservation; refuses missing implementation.
- Qualified source evidence may certify exact fresh47 or 48 or cached mode; a source-kernel implementation existing is not proof of general dynamic source.
- Backend code may retain explicit specialized geometric constants as a guarded implementation (e.g. QKV/Z width) but NOT falsely treat them as generic GDN mathematical law.
- Fresh-prefill capture added in ba7fb03 and oracle reference layout amended in fadf2b1. Treat as **in-progress source/qualification**, not promoted ordinary dynamic CUDA. Keep code/compiler/driver/ABI hashes for comparison when source moves.
- Historic finite source recipes and expected-output comparison remain in independent qualification/test support; code running in production must not lookup testcase-specific names or prompt lists.

## 7. Derived low-bit/approximate models after transition

Explicit EXL3/QTIP and ultra-low-bit/NVFP4 may materially improve extreme-model capacity but are not structural skeleton prerequisites. Before implementation, enforce:
- original source hash/model tensor complete identity, trained sidecars and source licensing;
- per-tensor exact packed coding, scale/codebook/trellis/basis transforms/exception handling, including MoE/embedding/head special tensors;
- optional offline calibration on independently held-out corpus; deliberate lossy conversion creates new ModelArtifactId;
- separately qualified CPU decoder/reference and direct GPU packed kernel, not mandatory full FP16 reconstruction;
- source checkpoint vs derived model quality; backend arithmetic vs derived source; target speculative losslessness relative to the *selected derived source* are THREE different quality claims;
- actual profile includes memory/compute, decoder/tile/register/LDS/capture cost, not just bits/weight; smaller bpw can run slower;
- model, quant and backend profile remain immutable within TaskSession/StateRoot/Bundle compatibility without explicit verified conversion/replay.

Source index: original ../fabric-vnext/README.md `quant, ../fabric-vnext/PRIOR_ART_AND_REVALIDATION.md, project .agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md and pipeline-composition/FORMATS.md. Follow actual source IDs and current laws, not guide README marketing.

## 8. Codec acceptance for the final consolidated Q0/Q2

**At every S1 commit:** fmt, architecture checker, exact artifact/CPU/GPU source package check and SMALL codec fixtures, no full trained/GPU qualification.

**At Q0 integrated transition:**
- Exact container source pack decoding byte-for-byte where defined; unknown/new variants refuse, safe parsed extents and overflow limits.
- Scalar F16/BF16 special values, signed zeros, scale/zero offsets, q4/q6/q8 blocks and negative malformed/corrupted source vectors.
- Q4_0/Q4_1/Q6_K/Q8_0 source-exact CPU output under original activation coding/reduction and explicitly distinct ideal F32 reference.
- Independent per-backend physical GPU source/ABI/version/shape/driver/Ready/retirement/unknown fault for qualified tuples only.
- End-to-end previously qualified Qwen3.5 GGUF CPU and native GPU source executions, same complete logits and all required state according to original mathematical contract.
- Separate original HF Qwen, Gemma4 E2B/HIP/Vulkan, Qwen27B and user-held device scopes; report unavailable/unexecuted instead of promoting.
- Finite CUDA fresh-prefill 47–50 original model profile checks only when matching original source and actual physical ownership; last changed capture cannot inherit earlier binary receipt.
- Confirm codecs/precision execute through the new model-neutral engine without codec-specific schedulers or HTTP switches.

**Never relax numerical/stochastic/determinism acceptance just to pass refactor.** Latest .agents/plans/numerics-and-context/CORRECTNESS_POLICY.md governs distinct semantic safety, operator accuracy, whole model quality, and deterministic/batch consistency claims. Original failed widths N1 vs B4 remain failed if genuinely different even when each separately fits reference.
