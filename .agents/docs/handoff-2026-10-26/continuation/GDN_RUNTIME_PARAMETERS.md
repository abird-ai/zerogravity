# GDN runtime parameters and qualification fixtures

Owner: numerical_failure_cleanup; ROOT owns production/build/device work.
26Oct; source-only. Supersedes REPORT.md's per-length integration proposal.
No production change or fresh48–50 native execution support.

## Executed source proof

- `parameter-proof.py`: one pure Python/source batch, exit0. Rehashed all actual
  old96 CUDA files against exact archive bcab1b4663df…5049; authenticated actual
  retained PTX by each tuple digest and renderer source7ce5b591…28b2 before import.
- Located scalar sites by original PTX **parameter index/load instruction**.
  Replacing only those load operands with typed runtime operands is reversible:
  specializing restores all96 complete original CU files byte-for-byte.
  No arbitrary numeric replacement, compilation, GPU or numerical acceptance.
-96 stored files contain29 exact CUDA sources and17 PTX bodies. Parameterizing
  yields19 wrappers but17 instruction/ABI bodies: Solve has three exported entry
  names for identical PTX/asm/config/formals. Old names stay fixture facts; a
  consolidated runtime wrapper gets a new source/ABI identity.
- Outputs: `parameter-proof/{result.json,stage-summary.json,
  all96-source-partitions.json,inputs.json}` contain complete source hashes,
  launch/scalar membership and the reversible transformation result.

| Stage | Actual minimal prefill bodies, both profiles | Genuine variation |
|---|---:|---|
| Conv |1|same PTX/config; captured cache-capacity scalar492/157|
| Prep |1|L and six source stride fields|
| Cumsum |1|T formal unused|
| KKT |3|BK32/4warps/3stages/8192B; BK32/8warps/3stages/16384B; BK64/8warps/4stages/24576B|
| Solve |1|128threads/10240B; only three entry labels differ|
| Recompute |2|BK64/BV64/8warps;3stages/32768B versus4stages/40960B|
| H |2|BV64/49412B versusBV32/41220B;4warps/2stages|
| Output |2|BK32/BV32/4warps;2stages/12288B versus3stages/20480B|
| Gated |1|M=16*token_rows, row strides and epsilon|
| Cached Conv/Decode/Gated |3total|one parameterized body each|

Actual PTX never loads T in Cumsum/KKT/Solve/Recompute/H/Output; native sequence
controls provide boundaries. Output loads scale bits1035273459; Gated loads eps
bits897988541. No host float recomputation. Actual tuner choices are non-monotonic:
.8B67 uses KKT BK64/Recompute3stages; .8B72 uses BK32/4stages.2B choices differ for
the same row counts. A guessed threshold cannot preserve those numerical choices.

## First mechanical candidate

`parameter-proof/conv-runtime-candidate.cu`, SHA
`2165f48fc19f67b7d6570086cf179feb5d00717f5a816772ee32db716df655c9`:

- Nine original pointer roles/order/types; append original scalar formals9/10/11:
  num_cache_lines:u32, stride_x_token:u64, stride_o_token:u64. This single body
  specializes to all ten old .8B/2B prefill Conv files exactly.
- Conv has no token-count scalar. GridX=ceil(M/8), gridY=24; native QueryStarts,
  ConvBatch/ConvChunkOffsets carry the true spans. Channels6144/kernel4/BLOCK_M8/
  BLOCK_N256/state strides are actual closed implementation constants.
- The five .8B Conv copies are already byte-identical, as are the five2B copies.
  Grouping identical bodies needs no math substitution. A runtime wrapper's new
  entry/scalar ABI must retain its new identity; do not alias an old recipe pin.
- Candidate has separate scalar formals to expose the reversible proof. Actual
  integration should use one packed by-value block through existing transport
  below, with another explicit source/ABI identity and relevant validation.

Original num_cache_lines492/157 is upstream allocation metadata, not a model or
prompt property. Current owned CACHE_SLOTS=5 (`cuda/gdn.rs:27–30`) with checked
slots1/4. A new runtime constructor derives capacity/strides from retained cache
allocation/layout, checks every control index/addressed span against retained
leases, and refuses incompatible views. Never infer492/157 from a model enum or
silently substitute5 while claiming the same old tuple. Preserve legacy scalar
facts exactly in Rust qualification fixtures.

## Concrete production interface

1. Model emits existing SourceConvGeometry/SourceGdnGeometry, SourceGdnContract,
   actual SourceCall and state semantics (core source_program.rs:160–183;
   Qwen plan/lower.rs:11–28). Remove ordinary preparation's EXPECTED capture hash
   and start0/6 six-record lookup.
2. Backend prepares actual GDN code from authenticated paired operators, geometry
   and an explicit numerical/kernel configuration contract. A small closed
   configuration names real BK/BV/warps/stages and source/ABI variants, never a
   testcase/length. Geometry derives validated controls/grid using checked math;
   it does not silently choose another numerical implementation.
3. Bind one of17 actual instruction bodies with typed runtime scalar bits. Exact
   source/compiler/options/full ABI define code identity; scalar/launch/binding/
   access/control bytes define the prepared arithmetic instance. A cross-case
   catalogue aggregate does not identify one implementation.
4. Learned parameter compatibility retains authenticated issuer, exact four
   roles/words/site and typed operator/numerical/cache-layout contract. Replace
   text_catalogue with that explicit checked contract while retaining existing
   ModelGdnParameters::require operator/role/source checks; do not simply drop it.
5. Ordinary runtime uses preparation/validation/execution with no qualification
   runner or test-matrix lookup. Unsupported configurations/shapes still refuse
   before private allocation or submission.

Concrete shape of the internal API, not a new registry or public code loader:

```rust
// Backend-owned, closed real implementations; names describe tile/config.
struct GdnKernelConfig {
    kkt: KktKernel,             // BK32W4S3 / BK32W8S3 / BK64W8S4
    recompute: RecomputeKernel, // BK64BV64W8S3 / BK64BV64W8S4
    state: StateKernel,         // BV32 / BV64
    output: OutputKernel,       // BK32BV32W4S2 / BK32BV32W4S3
}
struct GdnGeometry { /* validated semantic ops/call + retained cache layout */ }
struct GdnScalarBlock { /* bounded immutable typed u32/u64 bit words */ }
// Existing source factory retains issuer/roles/leases; arbitrary code is impossible.
fn prepare_gdn(operators: /* existing authenticated pair */,
               geometry: GdnGeometry, kernels: GdnKernelConfig, /* existing owners */)
    -> Result<Arc<TextGdnCode>>;
```

Actual application cold preparation selects a declared configuration contract
once, independently of fixture ordinals/length lists. Every prepared nine-stage
family retains that full configuration and its actual selected source/compiler/
schedule/ABI identity, so stage-mixing checks still compare one coherent family.
Decoded three-stage code has its own actual family under the same semantic
parameter contract. Epsilon/scale identity uses original bits, not float Debug
text. Cache layout records retained slot count/stride/offsets and addressed ranges.
No default is inferred or claimed qualified here; the next implementation unit
must explicitly choose its policy/configuration and validate the affected shapes.

Policy constraint: preserving all old variant behavior requires explicit variant
contracts. Their historical autotuner outcomes cannot be recreated by guessed
M thresholds. A deterministic new default across shapes is a separately declared
numerical profile/policy needing qualification. Old exact variant fixtures remain
regression inputs, including historical strict batch mismatch FAIL101.

## Existing scalar transport, same owner

- cuda/kernel.rs:51–66,153–164,267–279 stores/prepares a pointer-only ABI and
  appends only with_buffer_io. Add one internal bounded typed scalar-block ABI
  to PinnedKernel/PreparedPinnedLaunch; retain original pointer lease validation.
- Already present cubecl-runtime0.11 server/base.rs:1362–1433,1522–1534:
  KernelArguments::with_info(MetadataBindingInfo), data:Vec<u64>.
- CUDA runtime.rs:236–246 enables grid_constants for SM>=70; CUDA
  compute/server.rs:892–954 passes static words as a trailing by-value argument.
  dynamic_metadata_offset=data.len() creates **no** dynamic metadata buffer.
  Do not use custom(): it sets0 and creates a dynamic-tail device upload.
- One fixed64bit word per typed scalar is simple: u32 bits in low32/high32 zero;
  u64 uses the whole word. CUDA's final fixed struct contains the words; its
  inline r/l operand uses the declared width. FP32 fields use exact u32 bits.
  Conv's three words=24B. Validate type/order/count/padding, exact source/geometry
  and all pointer/control ranges before enqueue.
- Schema belongs to kernel ABI/cache identity; values belong to immutable prepared
  launch + StageArithmetic instance. Charge retained words and construction copies
  under existing checked metadata reservations. Reuse client.launch, original
  ResourceArbiter/flight/control uploads/fences/completion/quarantine/publisher.
  No second stream, allocator, queue, cache-state publisher or vendor driver path.

Conv packing example: host words `[cache_lines as u64, input_stride, output_stride]`;
one CUDA by-value `struct { unsigned long long words[3]; }` formal after the nine
pointers. PTX operand r uses `static_cast<unsigned int>(words[0])`; operands l use
words1/2. This is one24B CUDA argument, not three adjacent fake pointer arguments.
The existing server uses info.data.as_ptr() for that formal when the static-tail
offset is3. Reflected ABI size/offset and native packet tests must verify it.

## Production and fixture migration scope

| Path | Actual role and disposition |
|---|---|
| GPU cuda/gdn/text/observed.rs:16–128,152–1003 | Executable source/launch specialization dispatch. Replace with actual configuration/geometry factory + shared parametric bodies; cannot delete real code operands as fixtures. |
| observed.rs:130–149 | cfg(test) catalogue*6 helper; move beside Rust regression fixture data. |
| observed.rs:1005+; arithmetic.rs:84–142 | Full96 source catalogue/cross-case digest currently participates in runtime family identity. Replace with actual selected implementation/compiler/schedule/ABI identity. |
| Model source/binding/window_model/prepare/gdn.rs:7–108,140–163 | Duplicate capture pins/membership gate ordinary preparation. Keep semantic/operator issuer; move exact old hash matrix to Rust qualification fixtures. |
| Application qwen35_cuda/bindings.rs:60–66 | Replace model capture→backend recipe hash join with semantic prepared implementation binding. |
| GPU cuda/native/model_gdn.rs:81,109–163; model_gdn/text.rs:17–49 | Real learned-source/role owner with catalogue coupling. Keep ownership/checks, use typed compatibility contract. |
| GPU cuda/native/text_model.rs:51–79,351–456 | Retains/caches prepared recipe: use actual implementation/schema identity; preserve source/geometry owners. |
| GPU cuda/gdn/text.rs; text/workspace.rs | Real call binding/control leases/cold preparation remain; replace membership dependency with checked runtime geometry/configuration. |
| GPU cuda/gdn/assets.rs; qwen2b_assets.rs; fixed_prompt_assets.rs | Finite specialized numerical implementations, not generic fallbacks. Preserve distinct variants; duplicate original tuple assertions in FixedPrompt require_owned belong to Rust fixtures. |
| GPU cuda/gdn/resident.rs; resident/arithmetic.rs | Actual stage/copy execution/instance identity: add scalar transport/identity only; preserve original physical schedule. |

Minimal first actual unit: scalar-block transport + shared Conv body + checked
runtime geometry/control constructor, consumed by actual preparation; Rust
fixtures exercise the same constructor/body. Then migrate true remaining
variants through the same interface coherently. A rename/dedup that retains
CAPTURES as runtime authority does not solve the architecture.

## Validation, without convention tests

- Rust qualification data retains old96 source/entry/PTX/config/launch/scalar/role
  facts. Exact reverse specialization and actual packed argument bytes test the
  transformation; avoid directory/name/textual convention assertions.
- Real numerical/native checks compare independent actual operands/outputs under
  declared variants, actual full ABI packets, consumed control bytes, all state/
  fullV/natural histories and genuine fresh47–50. Existing capture's Conv/slot/
  has_initial byte-proof gaps remain explicit; formulas do not close them.
- Exercise wrong scalar schema/padding/count, overflow/control/pointer extents,
  wrong source/service/variant and failed/cancelled submitted work through original
  physical owner. Historical strict failure stays failed.
- This source proof establishes neither compilation, model accuracy, new shape
  support, invariance, physical fit nor performance.

## Preserved partial offline tool

Only NEW tools/cuda-gdn-recipes/ in reference WT was edited. Frozen14 and
ROOT-owned9 text-tool files were not edited by this lane. Partial tool performs
exact-byte reader bootstrap, original96/packet/PTX/control joins and emits only
qualification JSON/CU, no Rust dispatch tables. It is **unexecuted/untested**;
implementation paused at ROOT's architecture priority change. Do not integrate
it as a working emitter/admission route based on this report.
