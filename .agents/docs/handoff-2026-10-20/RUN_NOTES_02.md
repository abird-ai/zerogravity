# Run Notes — volume 2

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-hf-gdn-conv-root-v7-20261020/API.md

Original UTF-8 bytes: 4630; SHA256 `cb5b8c534eb1185ae93a9c9b387c1a6944641b2cf2315621fcd0f192732cceaf`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original GDN conv + SiLU private region

- SOURCE ONLY. No author build/test/operator/model/container/GPU execution.
- Parent bytes: `BASELINE.json`; 16 implementation/test paths + separate two-path
  GPU refusal proposal. ROOT reviews, merges with fresh mtimes, compiles/tests.
- Same `MathematicalBackend` → `PlanBuilder`/`HostMath` → `ForwardPlan` seam.
  No model/layer/token schedule, StateRoot owner, state effect or publication.
- Full original-HF forward/session/checkpoint guards remain CLOSED. No GGUF,
  old schema0x10/0x20, resident kernel or numerical provider behavior changes.

## Region and source

`CausalConvSiluContract::TorchCpuBf16Avx2` requires a bound raw
`SourceParameter`: original BF16 `[C,1,4]`, full `8C` bytes, `0<C<=16384`,
`C%32==0`; input C and canonical history3C are exact finite widened BF16.
Three private BF16 outputs: raw convolution C, final SiLU C, next history3C.
Plan recording requires the explicit CPU profile and a state-free slice;
foreign values, strided values, shape/precision/source mismatches refuse.
Unaware backends refuse before callbacks; GPU preflight/lowering cold-refuse.

`VerifiedTensorSource::conv4_parameter(name)` is an additional sealed original
rank3 descriptor. Existing `parameter()` stays rank1-only. Existing
`read_parameter*` preserves whole original metadata on partial ranges, same
issuer/location Arc, raw offsets/hash/FD, admission/cancel/drop. No fake rank1
matrix or partial `TensorSource` is created. Numerical code requires fullstage.

`OriginalWindowOperators::{append,execute}_gdn_convolution_silu_slice` validates
actual layer kind, original GdnConvolution role/name and full GdnQkv row count.
Loaded models only append existing same-builder operands; execution requires
window backing. Initial/cached external operands do not assert a producer,
committed root, epoch, token frontier or current cache ownership.

## Initial and cached operands

`GdnConvSliceInput` carries explicit raw BF16 activation/history bytes, bound
source/profile and `ConvInputPhase`:

- `InitialPadding`: actual original left3 padding must be supplied as +0 bytes;
  any negative-zero bit refuses, without canonicalization.
- `CachedTail`: exact original incoming `[C,4]` cache's last3 entries, oldest to
  newest. Signed zero is transported unchanged.

The accepted conv helper preserves the source four independent FMA partials
and three adds, then BF16 RNE; next3 is `[prior1,prior2,current]`. Accepted
contiguous16 BF16-input SiLU computes FP32, then the existing HostMath BF16 RNE
casts once. No stdlib exp substitution, zero canonicalization or tolerance.

IMPORTANT: accepted conv4 and contiguous SiLU are separate operator receipts.
Logical packing does NOT prove the original noncontiguous view selected the
same unary schedule. Fresh actual four-call comparisons below are mandatory;
this proposed region is UNQUALIFIED until ROOT executes them. No current
metadata/profile may be promoted on the basis of this source patch.

## Actual admitted backing

Both input Vecs share one reservation made BEFORE either allocation:
Output `4C*4` + Scratch960KiB. All three outputs share a separate aggregate
reservation BEFORE convolution/SILU allocations: Output`5C*4` + Scratch960KiB.
At C6144, live retained output payload is221184B and total input+result
reservations2187264B; full raw parameter49152B plus actual offset staging is
separately admitted. Scratch conservatively covers raw FP32, SiLU FP32,
rounding/output temporaries and small control allocations. Existing fixed
three-node/attention workspace policy is unchanged.

HostValue clones/views retain original backing and source owner. Borrowed
outputs only; `into_vec()` refuses leased storage. Buffers drop before their
shared reservation; all source/stage/input/result reservations retire after
last owners. These are actual ResourceArbiter charges, not a full-host RSS or
free-RAM admission claim.

## Prepared checks (not executed)

- Core: three BF16 outputs/no effects; raw kind/rank/byte/source/history guards;
  valid-geometry strided refusal then explicit pack; unaware callback refusal.
- Artifacts: full/partial metadata; foreign issuer; raw geometry/dtype;
  corruption, pre-cancel, actual pressure, retained window/final0.
- Model: original roles/same recorded backend; accepted helper wiring;
  signed-zero history; exact aggregate output counts; live clones/no Vec escape;
  pre-cancel and deterministic after-import cancellation; full-pool pressure
  and input-fit/result-one-byte-short pressure; foreign source/provider/strides;
  existing full-HF session/forward refusal and final0.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-root-v7-20261020/ROOT-QUALIFICATION.md

Original UTF-8 bytes: 5416; SHA256 `8a186c3ccb0ed0c7dee2631dea89156b6f5dafc39b2701230e17173f01d892c4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only genuine qualification

Status: PREPARED / UNQUALIFIED. Latest additive freeze: `freeze-manifest-v2.json`;
initial `freeze-manifest.json`/`frozen-prepared/` retained unchanged. Author parsed/formatted source only. No raw
model/tensor payload was hashed or operator/reference/harness executed.
`bindings.json` copies expected source/capture hashes from genuine existing
receipts; author independently read/pinned only small source/metadata.

1. Independently review/apply NEW_DELTA + GPU_REFUSAL to the exact parent
   bytes; do not overwrite intervening work. Refresh modification times.
   Run focused CPU/core/artifact fixtures and relevant strict default/CPU/CUDA
   Clippy plus full isolated check. New tests are currently uncompiled.
2. ROOT runs `prepare_inputs.py --output <ownrun>/root-inputs-1` bounded under
   CPU4/finite wall and memory. It scans full original files with1MiB transfer
   buffers before/after, verifies original source and existing actual raw
   capture hashes, selects full original BF16 conv0 weights, observed QKV,
   actual raw K4 caches and captured activated output. Four cases:
   .8B/2B × teacher11(initial)/353(cached), source-36-F32-restored profile.
   Raw conv expectation is NOT fabricated or reconstructed here.
3. ROOT uses its reviewed owned-container lifecycle with the acquired pinned
   CPU image closure, networknone/no devices/12GiB/4CPUs, private nonce/name/CID,
   bounded logs/deadline and complete owned cleanup. Mount prepared own run
   read-only `/probe`, prepared input directory read-only `/inputs`, and a
   fresh private output parent writable `/results`. Inner argv:

   `python3 /probe/reference.py --inputs /inputs/inputs.json --inputs-sha256 ACTUAL_INPUTS_SHA --output /results/root-reference-1`

   Required environment: `ZG_CONV_CPUS=actual,four,allowed,CPUs`,
   `ATEN_CPU_CAPABILITY=avx2`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
   `OPENBLAS_NUM_THREADS=4`, CUDA/NVIDIA visibility disabled. Script sets/checks
   affinity before Torch load; pins Torch2.13+cu129/git, official modeling
   source, CPU AVX2/mkldnn/thread policy. ROOT binds actual image/DSO closure,
   invocation, script/input/output hashes and cleanup separately; a script
   report alone is not a runtime-closure receipt. No new standalone container
   lifecycle is introduced by this patch.

   Genuine public `F.conv1d` and `F.silu`, no model load or foreign token loop:
   initial actual padded4 → padding3 convolution → first4 view → SiLU → last
   teacher; cached actual incomingK4+new → conv → last1 view → SiLU. Preserve
   actual original slices/strides. Public updated K4 and activated bytes MUST
   equal the original unchanged-inference archives before any raw conv target
   is written. Public next3 must equal actual current cache's last3 bytes.
   Fresh raw conv targets are separate public-operator evidence, never a
   missing live trace. Any mismatch fails; never fit expectations or widen a
   tolerance. Original F32 learned36 profile remains unchanged; this region's
   actual conv weights are raw original BF16.
4. ROOT copies the genuine reference directory intact under this own run and
   pins actual input/reference reports, then builds `Cargo.toml` in its normal
   offline isolated coordinator path AFTER source merge. Cargo paths target
   MAIN; Rust helper calls only existing OriginalWindowOperators/HostMath.
   Run via `run_replay.py --binary ACTUAL_HELPER --binary-sha256 SHA --inputs
   ownrun/root-inputs-1/inputs.json --inputs-sha256 SHA --reference
   ownrun/root-reference-1/reference.json --reference-sha256 SHA --new-output
   ownrun/root-rust-1`.

   Unchanged accepted native leader cleanup is copied in `owned_lifetime.py`;
   ROOT must inspect/source-bind it and run its existing cleanup fixtures.
   Runner caps actual helper address space512MiB, four CPUs, finite900s per
   model/3300s total,4MiB log, exact executable FD and full source before/after.
   Both independent sources scan completely through VerifiedTensorSource and
   full320-text-role binding. Compare all6144 raw +6144 activated +18432 history
   words per call, signed bits including zeros:122880 words total across four
   calls. Actual caller input/prior bytes are also compared and retained.
5. ROOT independently compares saved actual raw BF16 outputs to public oracle
   targets and validates report/source/runtime/lifecycle bindings. Retained
   result views must match after original callers/source descriptors drop.
   Real pre-cancel, wrong-layer and complete host-pool pressure must leave the
   original exact ledger; synthetic deterministic after-import cancellation
   test is separate and is not relabeled as a trained mid-operator interrupt.
   Record actual per-class live input/result/scratch/staging reservations,
   final0, source equality and unchanged default full-HF guards.

Only exact selected private CPU operator evidence may pass. No full forward,
StateRoot/logits, cached GDN, initial chunk, source-streamed whole-model, GPU,
serving or performance eligibility follows.

ROOT V6 composes immutable V2 implementation with reviewed test-only V3 non-Drop cleanup and V4 actual74MiB/4reservation harness ledger. Independent actual public reference/inputs copied byte-for-byte from original owned runs; originals retained. V5 source-preparation path failure preserved, no MAIN change/build/runtime in that failure.

````

## Source: .agents/runs/model-execution-hf-gdn-conv-root-v7-20261020/TODO.md

Original UTF-8 bytes: 1211; SHA256 `9bf6a9f764ca651fcc616833730bcf99c0940f306d4bfd6e0ea0170aaec3d99d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# GDN conv+SiLU shared-engine working checklist

- Parent: model-execution; owner baseline_review; ROOT sole build/test/runtime/merge.
- Scope: proposed source only under this run; no MAIN/shared-plan edits.
- [x] Read accepted closed conv/SiLU helpers and pinned source first/cached routes.
- [x] Inspect genuine .8B/2B two-profile metadata/header only; no tensor payload/hash.
- [x] Prepare typed default-refusing pure region + same HostMath/ForwardPlan/role binding.
- [x] Prepare sealed raw rank3 conv parameter stage, exact aggregate output/scratch leases.
- [x] Prepare sourcefree ownership/geometry/refusal/cancel/pressure/lifetime tests.
- [x] Prepare ROOT authentic selected-operand public oracle and replay, no expected reconstruction.
- [x] Freeze source-only proposal for ROOT review.
- [ ] Independent peer source review (Spine reprioritized; none claimed).
- [ ] ROOT focused/strict/full gates, genuine operator oracle and122880-word replay; no execution claimed.

Full HF forward/session/checkpoint/publication and GPU eligibility stay CLOSED.

Source-only checks: rustfmt18 + helper Rust syntax; Python AST6; whitespace18.
Prepared fixtures:11 Rust +2 Python. No fixture was executed by author.

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/README.md

Original UTF-8 bytes: 2661; SHA256 `80e87387d670e165d1ed1a14c2badc02f9bb59969c0ab2b377d1a87fe7519d41`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: spine_review

- Parent: model-execution. Ignored source proposal; MAIN untouched.
- Baseline: isolated ROOT composition `036ce0dc`, 521 Rust/manifest inputs.
- Production delta: `coefficient_roles` admits unevaluated raw BF16 A_log only
  at H48/96B, under the existing original-HF CPU profile, exact source/name/rank.
  Existing F32/head×4 metadata remains exact. Consumed raw dt_bias contract,
  sigmoid/add/softplus implementation, source issuer and admission unchanged.
- Three owned paths in `owned-delta.json`: one role guard, old test-module
  declaration and new sealed full-header synthetic H48 tests. No CPU arithmetic,
  core node, forward schedule, GPU refusal, initial/cached/exp or full-model guard
  changes.
- Synthetic tests prepared only: old F32 H16 accepted; BF16 H16 refused; H48
  real sealed BF16 descriptor; wrong codec/extent/shape/name/source/profile;
  cancel, foreign input, actual pressure, retained owners/final0; existing two
  preparation nodes only/no state/frontier/logits.

`replay/` uses accepted original27B V3 reference metadata. Both real calls contain
all required stages; no recapture or reconstructed expected values is needed.
Native ROOT replay calls only `execute_gdn_coefficient_preparation_slice`:

- Teachers 11 initial chunk64 and 353 true cached; H48, one [1,1,48] call each.
- Actual external a/b inputs: 192 words total. Three outputs: 288 words total.
- Compare all 480 private words before and after caller source/operators drop.
- Provenance diagnostics compare genuine raw BF16 A_log and dt_bias (48 words
  each) to both public call archives. A_log is never passed to coefficient math.
- Same original 851-role SourceWeights window factory, source identity, physical
  host ResourceArbiter and aggregate input/output leases. Real one-byte-short
  aggregate pressure, pre-cancel, foreign issuer, wrong layer and final0.
- Independent validator authenticates complete original archives/31 source files,
  compares every output bit (BF16 widening is codec transport only), checks full
  word counts/source extents/owner retention/cleanup. No tolerance.

CPU source math/profile only. No live projection producer, exp(A_log), g, decay,
initial chunk implementation, complete GDN, StateRoot/session/112-domain model,
GPU/provider or serving/performance eligibility. Existing four-case/320-word
reference and all previous validator failures/freezes remain untouched.

No author builds/tests/module imports/operators/runtime, archive/model/DSO/binary
reads/hashes. Author checks: source reads, Rust formatting, Python AST parsing,
source-only byte comparisons. ROOT executes after current timing hold.

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/TODO.md

Original UTF-8 bytes: 1112; SHA256 `290d409a06aa5d9538db641ec5a5066ad96dbddf119b49de1cc9ba172bde4104`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

- Parent: `.agents/plans/model-execution/TODO.md` (ROOT reconciles only).
- Own: this new ignored run; proposed three-path metadata/test delta + replay.
- [x] Inspect d744 coefficient contract/helper; A_log is metadata-only here.
- [x] Inspect authentic original27B V3 metadata; both H48 calls have all stages.
- [x] Prepare narrow BF16 H48 unevaluated descriptor accommodation; F32 unchanged.
- [x] Prepare sealed-source/profile/range/cancel/pressure/retention fixtures.
- [x] Prepare same typed native replay + independent full signed-bit validator.
- [x] Preserve original d744/320-word receipts/failures and ROOT036c baseline.
- [ ] ROOT + independent source review; merge only into explicitly isolated tree.
- [ ] ROOT strict/focused tests and standalone release helper build.
- [ ] ROOT metadata/reference preparation, actual native replay + independent raw
  validation, exact counts/negative/owner/final0/cleanup proof.
- [ ] ROOT reconcile accepted scope; no full-model/GPU promotion from this slice.

No tests/runtime/payload hashes executed by author. Unexecuted source proposal.

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/FOLLOWUP.md

Original UTF-8 bytes: 2703; SHA256 `3fe47e485af049efa47cb3c19f92017de6f5117d4fea343d239874955437d6e0`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: spine_review

- Harness-only additive followup. Original05095281 all28 files/frozen copies,
  production3-path proposal, authentic V3 archives, old4-case/320-word reference
  and all failures untouched.
- Actual original27B config has GDN0/1/2 and full-attention3 (interval4). External
  coefficient inputs are source/profile-bound, not layer-bound: layer1 was a
  legitimate operation and therefore an invalid negative.
- New native main.rs selects layer3 and verifies actual bound geometry marks
  layer3 FullAttention before either teacher. No coefficient/math/profile or
  positive execution path changes.
- Sixteen replay files copied:13 byte-identical; main.rs changes only the above
  guard/negative index, source_contract.py only authenticates immutable original
  layer and redirects BASELINE/owned-delta lookup, test_contracts.py adds one
  genuine config-metadata fixture. Exact copy map in followup.json.
- Same typed execute_gdn_coefficient_preparation_slice; same original source
  factory/source-pin/ResourceArbiter/leases/cancel/pressure/retention/final0,
  actual288 targets+192 retained input words. Same optional record-only TERM/INT
  and untouched owner cleanup. No full-model/exp/GDN/GPU eligibility.

ROOT sequence after review/hold:

- Use this `replay/` as the qualification directory; retained ROOT.md applies,
  replacing its source directory with this child. Build this child's Cargo.toml
  against the same explicitly isolated ROOT tree after the unchanged original
  three-path metadata/test proposal is installed. No MAIN merge is asserted.
- Actual build/source attestation MUST include this child's five compile inputs
  (Cargo/main/factory/source-pin/source-bindings), not old05095281 main.rs. The
  source contract verifies original05095281 + this followup before/after and
  binds the unchanged prior ROOT521 + original3-path proposal (522 source files).
  Local5 + optional generated lock makes527/528 actual build rows.
- Run `python3 -m unittest test_contracts`, prepare fresh metadata reference,
  write fresh actual source attestation after build, then run_replay.py and
  validate_replay.py exactly as retained ROOT.md, using scripts/output dirs in
  this child. Never overwrite old attempts or retag an old binary/receipt.
- Existing11 metadata fixtures + new real-config layer fixture prepared; no
  author tests/imports/builds/operators/processes/containers/models/GPU or
  payload/model/DSO/binary hashes. Formatting/AST/source checks only.

Acceptance still requires ROOT actual native wrong-layer refusal and unchanged
complete-word/source/lifetime/final0 independent gates. This corrects preparation,
not an executed failure or numerical result.

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/frozen-prepared/FOLLOWUP.md

Original UTF-8 bytes: 2703; SHA256 `3fe47e485af049efa47cb3c19f92017de6f5117d4fea343d239874955437d6e0`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: spine_review

- Harness-only additive followup. Original05095281 all28 files/frozen copies,
  production3-path proposal, authentic V3 archives, old4-case/320-word reference
  and all failures untouched.
- Actual original27B config has GDN0/1/2 and full-attention3 (interval4). External
  coefficient inputs are source/profile-bound, not layer-bound: layer1 was a
  legitimate operation and therefore an invalid negative.
- New native main.rs selects layer3 and verifies actual bound geometry marks
  layer3 FullAttention before either teacher. No coefficient/math/profile or
  positive execution path changes.
- Sixteen replay files copied:13 byte-identical; main.rs changes only the above
  guard/negative index, source_contract.py only authenticates immutable original
  layer and redirects BASELINE/owned-delta lookup, test_contracts.py adds one
  genuine config-metadata fixture. Exact copy map in followup.json.
- Same typed execute_gdn_coefficient_preparation_slice; same original source
  factory/source-pin/ResourceArbiter/leases/cancel/pressure/retention/final0,
  actual288 targets+192 retained input words. Same optional record-only TERM/INT
  and untouched owner cleanup. No full-model/exp/GDN/GPU eligibility.

ROOT sequence after review/hold:

- Use this `replay/` as the qualification directory; retained ROOT.md applies,
  replacing its source directory with this child. Build this child's Cargo.toml
  against the same explicitly isolated ROOT tree after the unchanged original
  three-path metadata/test proposal is installed. No MAIN merge is asserted.
- Actual build/source attestation MUST include this child's five compile inputs
  (Cargo/main/factory/source-pin/source-bindings), not old05095281 main.rs. The
  source contract verifies original05095281 + this followup before/after and
  binds the unchanged prior ROOT521 + original3-path proposal (522 source files).
  Local5 + optional generated lock makes527/528 actual build rows.
- Run `python3 -m unittest test_contracts`, prepare fresh metadata reference,
  write fresh actual source attestation after build, then run_replay.py and
  validate_replay.py exactly as retained ROOT.md, using scripts/output dirs in
  this child. Never overwrite old attempts or retag an old binary/receipt.
- Existing11 metadata fixtures + new real-config layer fixture prepared; no
  author tests/imports/builds/operators/processes/containers/models/GPU or
  payload/model/DSO/binary hashes. Formatting/AST/source checks only.

Acceptance still requires ROOT actual native wrong-layer refusal and unchanged
complete-word/source/lifetime/final0 independent gates. This corrects preparation,
not an executed failure or numerical result.

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/frozen-prepared/replay/ROOT.md

Original UTF-8 bytes: 3661; SHA256 `94f1b34a1eb40d0a933713619c7f4de311ebfeaa65d1f45d303dcc746ddfa12c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only execution

- Source-only preparation. Wait for timing/runtime hold to lift.
- Preserve original proposal/ROOT source freezes and old reference attempts.
- Review `NEW_DELTA.patch` then copy its exact three `proposed/` files with fresh
  modification times into the existing isolated
  `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020`.
  MAIN remains untouched. Source contract expects precisely the prior 521 inputs
  plus this one test child; only the two owned old files may differ.
- Run relevant strict/focused CPU/core/model gates under ROOT's single coordinator.
  No mathematical helper change is proposed. Genuine full-model/GPU guard closed.
- This directory's `factory.rs` and `source-pin.json` are byte-identical accepted
  original27B factory/report (`4c74fa28`, 7058B); no absent local include.
- Generate standalone `Cargo.lock` offline with this Cargo manifest, review that
  all external packages are an exact parent-lock subset (contract enforces it).
  Root's own target directory, two Cargo jobs, four threads; release only.
- Build and strict-check this standalone manifest; record actual compiler argv,
  source/binary identities and build status independently. `implementation.json`
  is explicitly source attestation, not proof the binary came from those bytes.

After reviewed source/tests/build, run from this `replay/` directory:

```sh
python3 write_implementation.py --new-output root-implementation-1.json
python3 -m unittest test_contracts
python3 prepare_reference.py --new-output root-reference-1
python3 run_replay.py --binary /ABS/ROOT/RELEASE/BINARY \
  --binary-sha256 ROOT_ACTUAL_BINARY_SHA \
  --implementation root-implementation-1.json \
  --reference-directory root-reference-1 --new-output root-native-1
python3 validate_replay.py --output root-native-1 \
  --reference-directory root-reference-1 --implementation root-implementation-1.json
```

- Metadata preparation opens only original capture/source metadata, not payload.
- Native/independent proof authenticates complete archives and actual original
  source payloads; ROOT only. Same sealed opened-FD source factory, no source copy,
  model materialization or complete-source 4GiB bypass.
- Native configured host ledger 128MiB, control8MiB/source metadata64MiB; original
  binding8MiB. Two original 1MiB workspace groups/call, all copies/stages inside
  admitted envelopes. Configured host budget is not total host-RAM attestation.
- Actual source scans stream on retained FDs. Selected archive reads ≤192B;
  per-archive ≤64MiB and header bounded before read. Native AS512MiB, four CPUs,
  child wall900s/log4MiB; full attempt tree8MiB. No GPU/container/model runtime.
- `owned_lifetime.py` is byte-identical accepted TERM/grace/twoKILL-before-reap
  group cleanup. `owned_native.py` adds only truthful scope and an optional safe
  record-only cancellation checkpoint after original Popen ownership handoff.
  Exact accepted TERM/INT guard `00d1efc1`; signals never asynchronously raise
  across Popen assignment/cleanup. No host-command scope is called GPU Ready.
- Full original-source hashing before/after is expensive; do not run during
  unprofiled benchmark intervals. A native nonzero exit is UNQUALIFIED.

Expected only after actual execution/independent validation: six stages/288
output words, 192 retained a/b input words, each bit exact before/after caller
drop; real original raw A_log/dt provenance, two phases' cancel/foreign/layer/
one-byte-short refusals; actual final0/unfenced and closed process/group/EOF.

Keep exp, initial/GDN/full-model/session/112StateRoots/GPU/performance unqualified.

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/ROOT.md

Original UTF-8 bytes: 3661; SHA256 `94f1b34a1eb40d0a933713619c7f4de311ebfeaa65d1f45d303dcc746ddfa12c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only execution

- Source-only preparation. Wait for timing/runtime hold to lift.
- Preserve original proposal/ROOT source freezes and old reference attempts.
- Review `NEW_DELTA.patch` then copy its exact three `proposed/` files with fresh
  modification times into the existing isolated
  `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020`.
  MAIN remains untouched. Source contract expects precisely the prior 521 inputs
  plus this one test child; only the two owned old files may differ.
- Run relevant strict/focused CPU/core/model gates under ROOT's single coordinator.
  No mathematical helper change is proposed. Genuine full-model/GPU guard closed.
- This directory's `factory.rs` and `source-pin.json` are byte-identical accepted
  original27B factory/report (`4c74fa28`, 7058B); no absent local include.
- Generate standalone `Cargo.lock` offline with this Cargo manifest, review that
  all external packages are an exact parent-lock subset (contract enforces it).
  Root's own target directory, two Cargo jobs, four threads; release only.
- Build and strict-check this standalone manifest; record actual compiler argv,
  source/binary identities and build status independently. `implementation.json`
  is explicitly source attestation, not proof the binary came from those bytes.

After reviewed source/tests/build, run from this `replay/` directory:

```sh
python3 write_implementation.py --new-output root-implementation-1.json
python3 -m unittest test_contracts
python3 prepare_reference.py --new-output root-reference-1
python3 run_replay.py --binary /ABS/ROOT/RELEASE/BINARY \
  --binary-sha256 ROOT_ACTUAL_BINARY_SHA \
  --implementation root-implementation-1.json \
  --reference-directory root-reference-1 --new-output root-native-1
python3 validate_replay.py --output root-native-1 \
  --reference-directory root-reference-1 --implementation root-implementation-1.json
```

- Metadata preparation opens only original capture/source metadata, not payload.
- Native/independent proof authenticates complete archives and actual original
  source payloads; ROOT only. Same sealed opened-FD source factory, no source copy,
  model materialization or complete-source 4GiB bypass.
- Native configured host ledger 128MiB, control8MiB/source metadata64MiB; original
  binding8MiB. Two original 1MiB workspace groups/call, all copies/stages inside
  admitted envelopes. Configured host budget is not total host-RAM attestation.
- Actual source scans stream on retained FDs. Selected archive reads ≤192B;
  per-archive ≤64MiB and header bounded before read. Native AS512MiB, four CPUs,
  child wall900s/log4MiB; full attempt tree8MiB. No GPU/container/model runtime.
- `owned_lifetime.py` is byte-identical accepted TERM/grace/twoKILL-before-reap
  group cleanup. `owned_native.py` adds only truthful scope and an optional safe
  record-only cancellation checkpoint after original Popen ownership handoff.
  Exact accepted TERM/INT guard `00d1efc1`; signals never asynchronously raise
  across Popen assignment/cleanup. No host-command scope is called GPU Ready.
- Full original-source hashing before/after is expensive; do not run during
  unprofiled benchmark intervals. A native nonzero exit is UNQUALIFIED.

Expected only after actual execution/independent validation: six stages/288
output words, 192 retained a/b input words, each bit exact before/after caller
drop; real original raw A_log/dt provenance, two phases' cancel/foreign/layer/
one-byte-short refusals; actual final0/unfenced and closed process/group/EOF.

Keep exp, initial/GDN/full-model/session/112StateRoots/GPU/performance unqualified.

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 2661; SHA256 `80e87387d670e165d1ed1a14c2badc02f9bb59969c0ab2b377d1a87fe7519d41`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: spine_review

- Parent: model-execution. Ignored source proposal; MAIN untouched.
- Baseline: isolated ROOT composition `036ce0dc`, 521 Rust/manifest inputs.
- Production delta: `coefficient_roles` admits unevaluated raw BF16 A_log only
  at H48/96B, under the existing original-HF CPU profile, exact source/name/rank.
  Existing F32/head×4 metadata remains exact. Consumed raw dt_bias contract,
  sigmoid/add/softplus implementation, source issuer and admission unchanged.
- Three owned paths in `owned-delta.json`: one role guard, old test-module
  declaration and new sealed full-header synthetic H48 tests. No CPU arithmetic,
  core node, forward schedule, GPU refusal, initial/cached/exp or full-model guard
  changes.
- Synthetic tests prepared only: old F32 H16 accepted; BF16 H16 refused; H48
  real sealed BF16 descriptor; wrong codec/extent/shape/name/source/profile;
  cancel, foreign input, actual pressure, retained owners/final0; existing two
  preparation nodes only/no state/frontier/logits.

`replay/` uses accepted original27B V3 reference metadata. Both real calls contain
all required stages; no recapture or reconstructed expected values is needed.
Native ROOT replay calls only `execute_gdn_coefficient_preparation_slice`:

- Teachers 11 initial chunk64 and 353 true cached; H48, one [1,1,48] call each.
- Actual external a/b inputs: 192 words total. Three outputs: 288 words total.
- Compare all 480 private words before and after caller source/operators drop.
- Provenance diagnostics compare genuine raw BF16 A_log and dt_bias (48 words
  each) to both public call archives. A_log is never passed to coefficient math.
- Same original 851-role SourceWeights window factory, source identity, physical
  host ResourceArbiter and aggregate input/output leases. Real one-byte-short
  aggregate pressure, pre-cancel, foreign issuer, wrong layer and final0.
- Independent validator authenticates complete original archives/31 source files,
  compares every output bit (BF16 widening is codec transport only), checks full
  word counts/source extents/owner retention/cleanup. No tolerance.

CPU source math/profile only. No live projection producer, exp(A_log), g, decay,
initial chunk implementation, complete GDN, StateRoot/session/112-domain model,
GPU/provider or serving/performance eligibility. Existing four-case/320-word
reference and all previous validator failures/freezes remain untouched.

No author builds/tests/module imports/operators/runtime, archive/model/DSO/binary
reads/hashes. Author checks: source reads, Rust formatting, Python AST parsing,
source-only byte comparisons. ROOT executes after current timing hold.

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1112; SHA256 `290d409a06aa5d9538db641ec5a5066ad96dbddf119b49de1cc9ba172bde4104`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

- Parent: `.agents/plans/model-execution/TODO.md` (ROOT reconciles only).
- Own: this new ignored run; proposed three-path metadata/test delta + replay.
- [x] Inspect d744 coefficient contract/helper; A_log is metadata-only here.
- [x] Inspect authentic original27B V3 metadata; both H48 calls have all stages.
- [x] Prepare narrow BF16 H48 unevaluated descriptor accommodation; F32 unchanged.
- [x] Prepare sealed-source/profile/range/cancel/pressure/retention fixtures.
- [x] Prepare same typed native replay + independent full signed-bit validator.
- [x] Preserve original d744/320-word receipts/failures and ROOT036c baseline.
- [ ] ROOT + independent source review; merge only into explicitly isolated tree.
- [ ] ROOT strict/focused tests and standalone release helper build.
- [ ] ROOT metadata/reference preparation, actual native replay + independent raw
  validation, exact counts/negative/owner/final0/cleanup proof.
- [ ] ROOT reconcile accepted scope; no full-model/GPU promotion from this slice.

No tests/runtime/payload hashes executed by author. Unexecuted source proposal.

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/replay/ROOT.md

Original UTF-8 bytes: 3661; SHA256 `94f1b34a1eb40d0a933713619c7f4de311ebfeaa65d1f45d303dcc746ddfa12c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only execution

- Source-only preparation. Wait for timing/runtime hold to lift.
- Preserve original proposal/ROOT source freezes and old reference attempts.
- Review `NEW_DELTA.patch` then copy its exact three `proposed/` files with fresh
  modification times into the existing isolated
  `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020`.
  MAIN remains untouched. Source contract expects precisely the prior 521 inputs
  plus this one test child; only the two owned old files may differ.
- Run relevant strict/focused CPU/core/model gates under ROOT's single coordinator.
  No mathematical helper change is proposed. Genuine full-model/GPU guard closed.
- This directory's `factory.rs` and `source-pin.json` are byte-identical accepted
  original27B factory/report (`4c74fa28`, 7058B); no absent local include.
- Generate standalone `Cargo.lock` offline with this Cargo manifest, review that
  all external packages are an exact parent-lock subset (contract enforces it).
  Root's own target directory, two Cargo jobs, four threads; release only.
- Build and strict-check this standalone manifest; record actual compiler argv,
  source/binary identities and build status independently. `implementation.json`
  is explicitly source attestation, not proof the binary came from those bytes.

After reviewed source/tests/build, run from this `replay/` directory:

```sh
python3 write_implementation.py --new-output root-implementation-1.json
python3 -m unittest test_contracts
python3 prepare_reference.py --new-output root-reference-1
python3 run_replay.py --binary /ABS/ROOT/RELEASE/BINARY \
  --binary-sha256 ROOT_ACTUAL_BINARY_SHA \
  --implementation root-implementation-1.json \
  --reference-directory root-reference-1 --new-output root-native-1
python3 validate_replay.py --output root-native-1 \
  --reference-directory root-reference-1 --implementation root-implementation-1.json
```

- Metadata preparation opens only original capture/source metadata, not payload.
- Native/independent proof authenticates complete archives and actual original
  source payloads; ROOT only. Same sealed opened-FD source factory, no source copy,
  model materialization or complete-source 4GiB bypass.
- Native configured host ledger 128MiB, control8MiB/source metadata64MiB; original
  binding8MiB. Two original 1MiB workspace groups/call, all copies/stages inside
  admitted envelopes. Configured host budget is not total host-RAM attestation.
- Actual source scans stream on retained FDs. Selected archive reads ≤192B;
  per-archive ≤64MiB and header bounded before read. Native AS512MiB, four CPUs,
  child wall900s/log4MiB; full attempt tree8MiB. No GPU/container/model runtime.
- `owned_lifetime.py` is byte-identical accepted TERM/grace/twoKILL-before-reap
  group cleanup. `owned_native.py` adds only truthful scope and an optional safe
  record-only cancellation checkpoint after original Popen ownership handoff.
  Exact accepted TERM/INT guard `00d1efc1`; signals never asynchronously raise
  across Popen assignment/cleanup. No host-command scope is called GPU Ready.
- Full original-source hashing before/after is expensive; do not run during
  unprofiled benchmark intervals. A native nonzero exit is UNQUALIFIED.

Expected only after actual execution/independent validation: six stages/288
output words, 192 retained a/b input words, each bit exact before/after caller
drop; real original raw A_log/dt provenance, two phases' cancel/foreign/layer/
one-byte-short refusals; actual final0/unfenced and closed process/group/EOF.

Keep exp, initial/GDN/full-model/session/112StateRoots/GPU/performance unqualified.

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/ROOT.md

Original UTF-8 bytes: 3661; SHA256 `94f1b34a1eb40d0a933713619c7f4de311ebfeaa65d1f45d303dcc746ddfa12c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only execution

- Source-only preparation. Wait for timing/runtime hold to lift.
- Preserve original proposal/ROOT source freezes and old reference attempts.
- Review `NEW_DELTA.patch` then copy its exact three `proposed/` files with fresh
  modification times into the existing isolated
  `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020`.
  MAIN remains untouched. Source contract expects precisely the prior 521 inputs
  plus this one test child; only the two owned old files may differ.
- Run relevant strict/focused CPU/core/model gates under ROOT's single coordinator.
  No mathematical helper change is proposed. Genuine full-model/GPU guard closed.
- This directory's `factory.rs` and `source-pin.json` are byte-identical accepted
  original27B factory/report (`4c74fa28`, 7058B); no absent local include.
- Generate standalone `Cargo.lock` offline with this Cargo manifest, review that
  all external packages are an exact parent-lock subset (contract enforces it).
  Root's own target directory, two Cargo jobs, four threads; release only.
- Build and strict-check this standalone manifest; record actual compiler argv,
  source/binary identities and build status independently. `implementation.json`
  is explicitly source attestation, not proof the binary came from those bytes.

After reviewed source/tests/build, run from this `replay/` directory:

```sh
python3 write_implementation.py --new-output root-implementation-1.json
python3 -m unittest test_contracts
python3 prepare_reference.py --new-output root-reference-1
python3 run_replay.py --binary /ABS/ROOT/RELEASE/BINARY \
  --binary-sha256 ROOT_ACTUAL_BINARY_SHA \
  --implementation root-implementation-1.json \
  --reference-directory root-reference-1 --new-output root-native-1
python3 validate_replay.py --output root-native-1 \
  --reference-directory root-reference-1 --implementation root-implementation-1.json
```

- Metadata preparation opens only original capture/source metadata, not payload.
- Native/independent proof authenticates complete archives and actual original
  source payloads; ROOT only. Same sealed opened-FD source factory, no source copy,
  model materialization or complete-source 4GiB bypass.
- Native configured host ledger 128MiB, control8MiB/source metadata64MiB; original
  binding8MiB. Two original 1MiB workspace groups/call, all copies/stages inside
  admitted envelopes. Configured host budget is not total host-RAM attestation.
- Actual source scans stream on retained FDs. Selected archive reads ≤192B;
  per-archive ≤64MiB and header bounded before read. Native AS512MiB, four CPUs,
  child wall900s/log4MiB; full attempt tree8MiB. No GPU/container/model runtime.
- `owned_lifetime.py` is byte-identical accepted TERM/grace/twoKILL-before-reap
  group cleanup. `owned_native.py` adds only truthful scope and an optional safe
  record-only cancellation checkpoint after original Popen ownership handoff.
  Exact accepted TERM/INT guard `00d1efc1`; signals never asynchronously raise
  across Popen assignment/cleanup. No host-command scope is called GPU Ready.
- Full original-source hashing before/after is expensive; do not run during
  unprofiled benchmark intervals. A native nonzero exit is UNQUALIFIED.

Expected only after actual execution/independent validation: six stages/288
output words, 192 retained a/b input words, each bit exact before/after caller
drop; real original raw A_log/dt provenance, two phases' cancel/foreign/layer/
one-byte-short refusals; actual final0/unfenced and closed process/group/EOF.

Keep exp, initial/GDN/full-model/session/112StateRoots/GPU/performance unqualified.

````

## Source: .agents/runs/model-execution-hf-initial-chunk-probe-20261020/TODO.md

Original UTF-8 bytes: 4140; SHA256 `6bebc53473944d97deb388abb5ceb3a4f607b67812bcaab45d48a791a0305c80`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pinned initial chunk64 public-op probe

Parent: model-execution. Owner: batch_design_review.
Own: this ignored run only. No model load, math replacement, MAIN production,
worktree/Cargo/GPU edits or execution. Root alone schedules the CPU probe.

- [x] Read pinned original source and actual step0 captured tensors/layouts.
- [x] Identify six BMM callsites, two unit-triangular solves, multiplicative
  query scaling, FP64-internal CPU cumsum and separate exp/VML gate.
- [x] Prepare untouched original-function/public-op replay with authentic
  physical layouts and source-free signed-zero/cancellation/underflow/dense cases.
- [x] Static review/AST+compile-syntax validation and freeze scripts; no execution.
- [ ] Root executes offline4CPU/12GiB probe and reviews source/dispatch closure.
- [ ] Pin actual MKL numerical topology before any closed initial-state helper.

Root-replay-1 failed at the observer's repeated source line364, before matrix
dispatch. Exact executed scripts are preserved under root-replay-1/executed-scripts;
its log/invocation remain unchanged. The observer now records each
(source-frame,label) point once, matching the original capture; tensor-name
duplicate checks and all50 bit/stride checks remain strict. This is an observer
fix only, no source math/expected-value changes. Fresh root-replay-2 required.

Root-replay-2 completed untouched-source frame checks/MKL calls, then refused a
constructed RHS unit-axis stride. Its exact scripts are preserved under
root-replay-2/executed-scripts. Replay now restores each actual SourceOps operand
shape/stride directly, after strict logical-callsite word validation; synthetic
templates use those exact recorded layouts. All stride assertions remain.
Completed original operands/results and50 frame-validation receipts are saved
before public replay, so later failures retain the authentic evidence. The sum
of both retained tensor archives is bounded64MiB (payload preflight + finalsize).
Fresh root-replay-3 required; no replay execution by this agent.

Root-replay-3 retained original-source archives/frame receipts but a redundant
same-shape reshape in the later validation normalized singleton-axis strides.
Exact scripts preserved under root-replay-3/executed-scripts. Validation now
checks restored operand shape directly and passes the unchanged operand to
strict bit/stride checks. Solve validation already uses the unchanged operand.
A source-free transport fixture asserts unusual recorded strides and records
same-shape reshape metadata for root execution. Fresh root-replay-4 required.

Prepared scripts are not qualification. Existing captures prove only the observed
token11/head16 case: solver outputs=RHS, v_new=new_values, state all+0 initially;
they do not establish a signed-zero-safe algebraic shortcut or serial GEMM.
The source64 solve/v_new buffers are column-major; capture restores those
strides before replay. All fifty original captured frame values/strides and
final outputs must match before public-op receipts are accepted.

Source-free cases record actual public-op outputs, never reconstructed targets.
They cover six callsites (UT/intra share a shape), both full solve shape/strides
and query multiplication. max60MiB tensor payload/max64MiB archive. Results
retain loaded binary hashes, source pins, optional CPU profiler and actual
MKL_VERBOSE SGEMM/STRSM evidence grouped by source/public callsite. The untouched
source block must itself contain both MKL dispatch families. Exclusive fresh
output directories retain failures; bounded cleanup targets only the runner's
recorded container ID and child process. No Rust/full-model/FLA/serving/27B claim.

Torch dispatch schemas are recorded dynamically; no inference-mode cast label
is assumed. Runtime schema/count/layout behavior is unexecuted and may refuse;
such a refusal is a preserved failed attempt, never qualification.

Root command after review, with a fresh output directory:

    python3 .agents/runs/model-execution-hf-initial-chunk-probe-20261020/run_probe.py \
      --profile --output .agents/runs/model-execution-hf-initial-chunk-probe-20261020/root-replay-4


````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/README.md

Original UTF-8 bytes: 3980; SHA256 `42d060a3fb843609a10d3e340fedd90a72a556e10b1fd8530af26495aa457cbd`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Initial-GDN query multiply +decay prefix

Owner baseline_review; parent model-execution. Source-only proposal based on
ROOT isolated036ce0dc/521 inputs. No worktree/MAIN/shared-plan changes or author
builds/tests/prepared imports/operators/providers/models/GPU/payload hashing.

## Exact primary source

- Torch cf30153 `ReduceOpsKernel.cpp:79–95`: accumulator starts from scalar+0,
  converts to `acc_type<scalar_t,false>`, adds each input in order and casts each
  stored prefix to scalar_t. `AccumulateType.h:64–66,155`: CPU float→double.
- Both complete primary files fetched/read at that exact immutable revision;
  copies +URL/SHA in `pinned-source/` and freeze. No reliance on the older audit's
  unretrieved links. Original BSD notice retained with the new CPU module.
- Pinned modeling762feb6c `torch_chunk_gated_delta_rule`: already-normalized
  query multiplies `128**-0.5`, rounded FP32 scalar0x3db504f3; padded decay
  cumsum dimension64. Cached query divides sqrt128 and is unchanged.

## Closed implementation

- Three new CPU Rust sources +one module declaration, no core/model/GPU changes.
- Explicit InitialChunk64/Cached phase, exact CPU profile, bound SourceIdentity,
  H16/H48. Query is ONE normalized `[heads,1,128]` FP32 call; prefix is ONE
  already-padded `[heads,1,64]` FP32 chunk. No arbitrary strides/T flattening.
- Allocation-free `initial_prelude_into` preserves multiplication/store order,
  signed zeros and unrounded FP64 carry. Passive x86/AVX2/MXCSR control guard:
  RNE, masked exceptions, FTZ/DAZ off; status bits ignored. No environment edits.
- Cache/Linux wrapper imports authenticated external raw FP32, checks same
  source identity, and reserves complete output +input/scratch before payload
  allocation. Exact Vec capacities; shared Arc output retains actual source and
  WindowWorkspace until lastdrop. Borrowed accessors only; no Vec escape.
- Query H48 charge49408B; prefix H48 charge24832B, including256B conservative
  per-result bookkeeping/scratch. Generic allocator/control-plane overhead and
  caller-owned external operands remain outside these payload figures.
- Invalid profile/Cached/source/shape/nonfinite/cancel fails closed. Prefix
  overflow returns error with private output; no partial state publication.
- Prepared tests: FP64 carry versus rounded store, signedzero/subnormal query,
  independent rows, bad source/profile/phase/extent/nonfinite, true sealed same-
  label foreign issuer, actual one-byte-short pool, retained clones and final0,
  deterministic completed-output cancellation. Existing synthetic source asset
  reused; no newly fabricated trained fixture.

No L2/decay/exp producer, provider, chunk solver, private state proposal,
SessionAuthority/StateRoot/checkpoint or full-HF eligibility is added. SAME
mathematical CPU module, not a model loop. Future typed HostMath integration is
separate after ROOT qualification; cached DIV stays unchanged.

## Genuine ROOT replay prepared

Original27B V3 step0 teacher11 contains exact complete input/output pairs:

- `chunk.qk_normalized.query`→`chunk.query_scaled.query`:6144 words.
- `chunk.cumulative_decay.decay`→`chunk.cumulative_decay.cum_decay`:3072 words.

Only original metadata selected by author; archives stayed unopened. ROOT
authenticates complete original archive and source FDs, then invokes both actual
new leaves;9216 outputs +9216 retained inputs compared raw before/after caller
drop. No expected math generation, query reconstruction or zero padding guess.

Original accepted source factory/supervisor/record-only TERM+INT handoff reused.
ROOT source attestation is524 inputs (521 parent +three Rust children), never
compiler/binary proof. Actual compile receipt remains separately required.
Configured host128MiB/control8MiB, AS512MiB/4CPU/900s, artifacts8MiB; source
verification reads55.563GB bounded RAM and must not overlap timed benchmarks.
No physical total-RAM/provider/GPU/performance claim follows these limits.

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/TODO.md

Original UTF-8 bytes: 672; SHA256 `40c1525761b5b741df7429b2830d96fcfefce9b8cd67d1cad92ee12316789330`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Initial query/prefix leaf proposal

Parent model-execution; owner baseline_review. MAIN/worktree untouched.

- [x] Retrieve/read exact pinned primary cumsum and accumulator source.
- [x] Prepare explicit InitialChunk64 CPU leaves +same-source admitted wrapper.
- [x] Prepare source/lifetime/refusal tests and genuine V3 raw replay/validator.
- [ ] ROOT independent review/build/tests, raw source/archive replay and validator.
- [ ] Later typed HostMath composition; provider/exp/initial core/fullmodel closed.

Author checks limited to source reads/generation, Rust formatting and Python AST.
No tests/compilers/imported operators/provider/model/GPU or payload hashing.

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 3980; SHA256 `42d060a3fb843609a10d3e340fedd90a72a556e10b1fd8530af26495aa457cbd`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Initial-GDN query multiply +decay prefix

Owner baseline_review; parent model-execution. Source-only proposal based on
ROOT isolated036ce0dc/521 inputs. No worktree/MAIN/shared-plan changes or author
builds/tests/prepared imports/operators/providers/models/GPU/payload hashing.

## Exact primary source

- Torch cf30153 `ReduceOpsKernel.cpp:79–95`: accumulator starts from scalar+0,
  converts to `acc_type<scalar_t,false>`, adds each input in order and casts each
  stored prefix to scalar_t. `AccumulateType.h:64–66,155`: CPU float→double.
- Both complete primary files fetched/read at that exact immutable revision;
  copies +URL/SHA in `pinned-source/` and freeze. No reliance on the older audit's
  unretrieved links. Original BSD notice retained with the new CPU module.
- Pinned modeling762feb6c `torch_chunk_gated_delta_rule`: already-normalized
  query multiplies `128**-0.5`, rounded FP32 scalar0x3db504f3; padded decay
  cumsum dimension64. Cached query divides sqrt128 and is unchanged.

## Closed implementation

- Three new CPU Rust sources +one module declaration, no core/model/GPU changes.
- Explicit InitialChunk64/Cached phase, exact CPU profile, bound SourceIdentity,
  H16/H48. Query is ONE normalized `[heads,1,128]` FP32 call; prefix is ONE
  already-padded `[heads,1,64]` FP32 chunk. No arbitrary strides/T flattening.
- Allocation-free `initial_prelude_into` preserves multiplication/store order,
  signed zeros and unrounded FP64 carry. Passive x86/AVX2/MXCSR control guard:
  RNE, masked exceptions, FTZ/DAZ off; status bits ignored. No environment edits.
- Cache/Linux wrapper imports authenticated external raw FP32, checks same
  source identity, and reserves complete output +input/scratch before payload
  allocation. Exact Vec capacities; shared Arc output retains actual source and
  WindowWorkspace until lastdrop. Borrowed accessors only; no Vec escape.
- Query H48 charge49408B; prefix H48 charge24832B, including256B conservative
  per-result bookkeeping/scratch. Generic allocator/control-plane overhead and
  caller-owned external operands remain outside these payload figures.
- Invalid profile/Cached/source/shape/nonfinite/cancel fails closed. Prefix
  overflow returns error with private output; no partial state publication.
- Prepared tests: FP64 carry versus rounded store, signedzero/subnormal query,
  independent rows, bad source/profile/phase/extent/nonfinite, true sealed same-
  label foreign issuer, actual one-byte-short pool, retained clones and final0,
  deterministic completed-output cancellation. Existing synthetic source asset
  reused; no newly fabricated trained fixture.

No L2/decay/exp producer, provider, chunk solver, private state proposal,
SessionAuthority/StateRoot/checkpoint or full-HF eligibility is added. SAME
mathematical CPU module, not a model loop. Future typed HostMath integration is
separate after ROOT qualification; cached DIV stays unchanged.

## Genuine ROOT replay prepared

Original27B V3 step0 teacher11 contains exact complete input/output pairs:

- `chunk.qk_normalized.query`→`chunk.query_scaled.query`:6144 words.
- `chunk.cumulative_decay.decay`→`chunk.cumulative_decay.cum_decay`:3072 words.

Only original metadata selected by author; archives stayed unopened. ROOT
authenticates complete original archive and source FDs, then invokes both actual
new leaves;9216 outputs +9216 retained inputs compared raw before/after caller
drop. No expected math generation, query reconstruction or zero padding guess.

Original accepted source factory/supervisor/record-only TERM+INT handoff reused.
ROOT source attestation is524 inputs (521 parent +three Rust children), never
compiler/binary proof. Actual compile receipt remains separately required.
Configured host128MiB/control8MiB, AS512MiB/4CPU/900s, artifacts8MiB; source
verification reads55.563GB bounded RAM and must not overlap timed benchmarks.
No physical total-RAM/provider/GPU/performance claim follows these limits.

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 672; SHA256 `40c1525761b5b741df7429b2830d96fcfefce9b8cd67d1cad92ee12316789330`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Initial query/prefix leaf proposal

Parent model-execution; owner baseline_review. MAIN/worktree untouched.

- [x] Retrieve/read exact pinned primary cumsum and accumulator source.
- [x] Prepare explicit InitialChunk64 CPU leaves +same-source admitted wrapper.
- [x] Prepare source/lifetime/refusal tests and genuine V3 raw replay/validator.
- [ ] ROOT independent review/build/tests, raw source/archive replay and validator.
- [ ] Later typed HostMath composition; provider/exp/initial core/fullmodel closed.

Author checks limited to source reads/generation, Rust formatting and Python AST.
No tests/compilers/imported operators/provider/model/GPU or payload hashing.

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/ROOT.md

Original UTF-8 bytes: 2017; SHA256 `95f0b357cd1f20cbd309627283c81bacfacf24a6418bbf13c4b63a53163a3058`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only qualification

1. Review proposed CPU sources, attribution and narrow module wiring. Parent is
   isolated036ce0dc/521 source inputs; preserve all existing work. Copy only
   owned files with fresh modification times into ROOT's isolated composition,
   never MAIN from this proposal. If the parent changes, retain this freeze and
   prepare an explicit source-contract derivative; do not retag old evidence.
2. Run focused CPU tests with/without cache and relevant strict checks. Full
   original-HF/session/provider guards remain closed. Expected parent plus
   three Rust children is524 inputs; license copy is separately frozen.
3. In this replay directory generate Cargo.lock offline. All external tuples
   must be an exact parent-lock subset. ROOT records actual argv/source/binary
   build receipt; write_implementation.py is source attestation only.
4. Prepare original metadata, then one fresh bounded native replay and an
   independent validator. Never run complete-source scans during timing.

```sh
python3 write_implementation.py --new-output root-implementation-1.json
python3 -m unittest test_contracts
python3 prepare_reference.py --new-output root-reference-1
python3 run_replay.py --binary /ABS/ROOT/RELEASE/BINARY \
  --binary-sha256 ROOT_ACTUAL_BINARY_SHA \
  --implementation root-implementation-1.json \
  --reference-directory root-reference-1 --new-output root-native-1
python3 validate_replay.py --output root-native-1 \
  --reference-directory root-reference-1 --implementation root-implementation-1.json
```

Expected only after actual execution: query6144 +prefix3072 original output
words and same9216 retained input words exact, both real pool/cancel/foreign/
Cached refusals, immutable full source/archive, cloned payload/source lifetimes,
actual charges74240B, final0 and owned group/EOF retirement. Genuine initial
input provenance is imported; no model producer or cached/FLA continuation is
asserted. Do not claim full GDN/state/provider/112-domain/model/GPU closure.

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/replay/ROOT.md

Original UTF-8 bytes: 2017; SHA256 `95f0b357cd1f20cbd309627283c81bacfacf24a6418bbf13c4b63a53163a3058`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only qualification

1. Review proposed CPU sources, attribution and narrow module wiring. Parent is
   isolated036ce0dc/521 source inputs; preserve all existing work. Copy only
   owned files with fresh modification times into ROOT's isolated composition,
   never MAIN from this proposal. If the parent changes, retain this freeze and
   prepare an explicit source-contract derivative; do not retag old evidence.
2. Run focused CPU tests with/without cache and relevant strict checks. Full
   original-HF/session/provider guards remain closed. Expected parent plus
   three Rust children is524 inputs; license copy is separately frozen.
3. In this replay directory generate Cargo.lock offline. All external tuples
   must be an exact parent-lock subset. ROOT records actual argv/source/binary
   build receipt; write_implementation.py is source attestation only.
4. Prepare original metadata, then one fresh bounded native replay and an
   independent validator. Never run complete-source scans during timing.

```sh
python3 write_implementation.py --new-output root-implementation-1.json
python3 -m unittest test_contracts
python3 prepare_reference.py --new-output root-reference-1
python3 run_replay.py --binary /ABS/ROOT/RELEASE/BINARY \
  --binary-sha256 ROOT_ACTUAL_BINARY_SHA \
  --implementation root-implementation-1.json \
  --reference-directory root-reference-1 --new-output root-native-1
python3 validate_replay.py --output root-native-1 \
  --reference-directory root-reference-1 --implementation root-implementation-1.json
```

Expected only after actual execution: query6144 +prefix3072 original output
words and same9216 retained input words exact, both real pool/cancel/foreign/
Cached refusals, immutable full source/archive, cloned payload/source lifetimes,
actual charges74240B, final0 and owned group/EOF retirement. Genuine initial
input provenance is imported; no model producer or cached/FLA continuation is
asserted. Do not claim full GDN/state/provider/112-domain/model/GPU closure.

````

## Source: .agents/runs/model-execution-hf-initial-prefix-peer-review-batch-20261020/REVIEW.md

Original UTF-8 bytes: 5741; SHA256 `7e2649b2e690a67f6532990c521548993bb6bc33f765f22b299400356ebf26ab`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Initial query/prefix leaf — independent source review

- Verdict: **no source blocker found** in the frozen3cda proposal. Compilation,
  authentic numerical replay and phase acceptance remain ROOT-owned/pending.
- Scope: production leaf, admitted wrapper, tests, pinned source, module wiring,
  genuine replay comparison/admission/source/owner paths. No imports, builds,
  tests, hashes, provider/model/GPU/runtime or payload reads by reviewer.
- MAIN/shared plans/root README and original freezes untouched.

## Source arithmetic

- Pinned modeling762feb6c normalizes query in FP32, then multiplies by
  `128**-0.5` before padding. The helper preserves the single FP32 multiply
  with scalar0x3db504f3; it neither substitutes cached sqrt division nor adds
  BF16 rounding or L2 math.
- Pinned Torch cf30153 `ReduceOpsKernel.cpp:79–95` initializes scalar+0,
  widens to `acc_type<float,false>`, adds each element in sequence and stores
  each prefix as FP32. `AccumulateType.h:155` maps CPU float to double.
  The helper retains an FP64 carry across64 elements and casts each store;
  it does not feed the rounded FP32 store back into the accumulator.
- Each contiguous head is independent. Query input is the real unpadded
  `[1,H,1,128]` normalized boundary; decay input is the real padded/reshaped
  `[1,H,1,64]` boundary. No padding values or hidden states are reconstructed.
- Pure multiply/add/cast need no opaque exp/GEMM implementation. Default strict
  Rust operations have no FMA or reassociation expression here. Passive MXCSR
  check requires RNE, masked exceptions, FTZ/DAZ off and AVX2; it ignores only
  exception status bits and does not alter thread controls.
- Finite source spans checked before writes; FP32 prefix overflow refuses with
  private partial output. Cancellation is checked before math, per head and
  after completion. Source-free cases distinguish FP64 carry, signed zeros,
  subnormal underflow and row independence; they are not authentic replay.

## Identity, admission and lifetime

- Contract fields are private; constructor closes wrong profile, Cached phase,
  unbound source and heads outside16/48. Import accepts exact full finite raw
  FP32 extent. This is explicitly external operator input, not producer proof.
- Admitted wrapper checks exact source identity before original source-pool
  admission. Same-source output/scratch requests precede both private Vecs;
  capacity checks prohibit a larger exposed payload capacity.
- Required payload envelope is `elements*8+256`: H48 query49,408B and
  prefix24,832B. Combined74,240B =36,864B Output +37,376B Scratch. External
  borrowed archive bytes and generic allocator/control overhead are explicitly
  outside this payload claim; no total-host physical admission is asserted.
- Backing declares input/output before WindowWorkspace, so payloads retire
  before their lease. Arc clones retain the same buffers and sealed source;
  borrowed accessors expose no Vec escape. Last-drop source retention and real
  one-byte-short source-pool refusal have meaningful prepared tests.
- This module adds no HostMath/model/token loop, StateRoot, SessionAuthority,
  public state/cache effect, checkpoint, GPU/provider route or full-HF permit.
  Initial exp/decay production and BMM/STRSM initial core remain separate/closed.

## Genuine replay and refusal scope

- Four exact authentic stage descriptors are bound to original accepted27B V3
  teacher11 step0/plain-observed equality, source/runtime identities and actual
  initial call count. Prefix input is tied to the recorded padded-decay bytes.
  Whole archive and retained FD extent checks precede selected reads; selected
  shape/dtype/range/raw-bit checks precede interpretation.
- Native comparison checks9216 returned input +9216 output words, including
  sign bits and full cardinality before zip. Targets are read for comparison;
  they never enter the helper. Independent validator compares emitted bytes
  directly to the unchanged original archive, without expected-value math.
- Direct leaf replay has only64MiB source +8MiB control before extraction:
 72MiB/two reservations is correct. It does not bind the two additional
  OriginalWindowOperators/QwenOriginalWindowText owners from older harnesses.
- Both operations exercise actual pre-cancel, wrong issuer, Cached refusal and
  aggregate one-byte-short refusal with exact ledger recovery. The native
  wrong-issuer case uses a fresh incompatible identity; the unit additionally
  opens a genuinely separate same-label sealed source.
- Results are cloned, caller source/config/result owners dropped, all imported
  inputs/outputs compared again, then clones dropped. Exact per-class ledgers
  and final0/unfenced are checked. Supervisor keeps WNOWAIT leader ownership
  through TERM/KILL before reap, original bounded EOF closure, and record-only
  TERM/INT handoff; no new model/runtime engine is introduced.

## Remaining physical gates

- ROOT compose against the joint current isolated source closure using the new
  attestor derivative; do not retag original521/524 source assertions or their
  compile status. Format/strict/unit/build and actual executable binding remain
  required.
- Run authentic H48 teacher11 initial replay plus independent raw validator;
  preserve signed-zero/subnormal/cancellation/pressure/retention results and
  source/FD/owner/finalization receipts. H16 trained replay is separate if that
  geometry is to be numerically promoted beyond source-free fixtures.
- Even a passing replay covers two pure imported initial leaves only. It does
  not close L2/coefficients/exp, initial chunk solver/state, trained activation
  producers, cached schedule, full27B/StateRoot, GPU, serving or performance.

````

## Source: .agents/runs/model-execution-hf-initial-prefix-peer-review-batch-20261020/TODO.md

Original UTF-8 bytes: 669; SHA256 `591d53003f91ce65ba63a3dc3df11600e5014c03f842455f40a74ff71fe1bb9c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Initial-prefix peer review

Parent: model-execution. Owner: batch_design_review.
Scope: this ignored review directory only.

- [x] Read production leaf/admitted wrapper/tests and module wiring.
- [x] Read pinned Torch cumsum/accumulator and original modeling schedule.
- [x] Read source/archive/replay comparison, physical ledger/retained owner and
  supervisor/refusal paths; distinguish pure leaf scope from full initial GDN.
- [x] Report no source blocker; compilation and runtime evidence remain pending.
- [ ] ROOT joint attestor composition, actual tests/replay and acceptance.

No execution/imports/builds/tests/hashes/payload reads; no MAIN/shared-plan edit.

````

## Source: .agents/runs/model-execution-hf-integration-design-20261020/DESIGN.md

Original UTF-8 bytes: 13950; SHA256 `39aaed1f735be8e9c88ff860768801c3bf0c39f31c703dec5697e4901c4adce8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original HF through the existing engine

Author: baseline_review. 2026-10-20. Source review of MAIN9815504.
Parent: model-execution. Design only; no implementation/admission change.

## Decision

Keep `LoadedQwenModel::require_execution_profile()` closed. Next integrate
**typed closed operators**, then parameter/state descriptors, then the SAME
`forward::forward_token` / `run_layer_sequence` through HostMath and PlanBuilder.
Do not turn the analytical facade into another production token engine.

Target is explicitly original mixed source + pinned Torch CPU AVX2 **eager
attention / slow GDN**. It is distinct from official all-BF16 loading, CUDA FLA,
GGUF, arbitrary BF16 kernels, and multi-token GEMM. Source bytes alone do not
identify these numerical policies.

## Current seams and blockers

- `model-qwen35/original_binding.rs`: genuine matrix and parameter roles already
  retain unchanged SourceWeights/SourceIdentity, OutputInput axes, original dtype,
  Nk/Nv/Dk/Dv/K, tied/independent head. Cold constructor intentionally leaves
  `layers`/`final_norm` empty; removing guard would index missing caches.
- `model-qwen35/forward.rs`: one existing token/layer schedule, but parameters
  still use GGUF names/CachedLayer F32 arrays. GDN assumes equal K/V heads/widths,
  qkv=3*width, GGUF grouped-RMS L2, packed GGUF coefficients and GGUF recurrence.
  RoPE has fixed64/1e7 constants. Genuine source config/roles must select them.
- `core/accelerator/program.rs`: SourceVector/StaticVector erase learned dtype;
  PlanValue/ForwardPlan have shape/source/state effects but no numerical profile
  or activation dtype. Project cannot distinguish CPU64 from default projection;
  Rms/Silu/Recurrent cannot express the actual additional boundaries below.
- `model-qwen35/host_math.rs`: embedding role checks literal token_embd.weight;
  project delegates legacy GGUF matvec; old fallback operators have GGUF math.
  These must not infer HF execution from BF16 packed storage.
- `zerogravity/original_qwen_math.rs` and precision/state modules: analytical
  proposals share source kernels and TaskCoordinator but still use legacy
  convolution/L2/recurrence/transcendentals in several places. They are useful
  probes, not qualified production semantics. Keep their diagnostic profile,
  trace names and schema offsets distinct until explicit replacement.
- `gpu/resident_program.rs`: non-F32 Project currently enters GGUF quantization
  planning/lowering. BF16 must get a separate qualified branch; never Q8-prepare
  an original BF16 projection as an implicit fallback. Unknown contracts fail
  preflight before physical admission/upload. Same applies to batch lowering.

## Required operation contracts

BF16 here means an exact ties-even16-bit roundtrip, not an accuracy tolerance.
F32 intermediate products/adds and signed zero are observable.

| Boundary | Required original/pinned CPU semantics | Remaining admission gate |
| --- | --- | --- |
| Embedding/head | Original BF16 row; tied embedding descriptor or actual lm_head. Final head returns BF16, then exact F32 widening for API/readback. | Whole vocabulary, both tokens; preserve separate F32/GGUF head behavior. |
| Linear | BF16 input, original output-first BF16 weights, pinned64-lane AVX2 FMA reduction, BF16 output. No transpose, quantization, or second learned payload. | Existing closed profile only; call-shape GEMV is not a multi-token GEMM policy. Unaware provider already refuses. |
| Ordinary/head/final RMS | x.float(), square/mean/rsqrt and multiply by1+w.float(), THEN BF16. Raw gamma is zero-centered; do not pre-adjust source bytes. | Existing GGUF norm uses F64 sum. Direct layer0 BF16 equality is local evidence, not all widths/layers/reduction topology proof. |
| Residual / FFN | BF16 add; SiLU(gate) -> BF16, multiply(up) -> BF16, down linear -> BF16. | Pin Torch elementwise/transcendental behavior; a fused F32 SiLU*up with only final cast changes policy. |
| Conv | Original rank3 BF16 [channels,1,K]; raw QKV BF16; precise4-tap CPU reduction and raw BF16 output, then SiLU BF16. Canonical next K-1 raw history. | Pure conv child frozen separately; actual Rust/full6144 replay pending coordinator. Do not silently change GGUF conv. |
| Q/K replication/L2 | Split2*Nk*Dk + Nv*Dv; repeat_interleave value-head ratio; exact BF16 widening to contiguous F32; FP32 L2 using eps1e-6 (not RMS eps/D). | Pure128 reducers are separate accepted operator scope; bind only qualified geometry. |
| GDN coefficients | beta=sigmoid(b) returns BF16; g=-exp(A_log.float())*softplus(a.float()+dt_bias) is FP32. exp(g) belongs to core phase. A_log raw source is not GGUF precomputed negative A. | Pinned sigmoid/softplus/exp still require closed actual-input/output replay. Cached helper accepts observed decay; that does not qualify coefficient generation. |
| Cached GDN | Key-major FP32 [head,key128,value128]; query divided by sqrt128 before products. Decay*state; materialized state*k -> exact key reduction; (v-memory)*beta; separate k*delta then separate add; state*q -> reduction. State stays F32; returned core rounds BF16. | New cached pure operator is isolated, not facade wiring; qualify full16 heads against actual capture and preserve explicit layout transport. No FMA substitution. |
| Initial/chunk GDN | Actual padded64 query/key/value/beta/decay; query * head_dim**-0.5; cumsum, pairwise decay,6 BMM +2 unit-triangular solves and source updates. | Separate unqualified algorithm. Equal real arithmetic does not permit replacing it with cached recurrence, even at zero prior state. MKL SGEMM_BATCH/STRSM topology remains open. |
| Gated RMS | FP32 normalization; BF16 normalized core; raw learned gamma multiplication promotes according to ORIGINAL kind; F32 SiLU(z.float()); final BF16. | Original .8B has36F32 learned A_log/norm parameters. F32 gamma must NOT acquire current diagnostic intermediate BF16 weighted cast. All-BF16 loader is a distinct profile. |
| RoPE / KV | Config-derived text RoPE; cos/sin computed F32 and returned BF16; q*cos and rotate_half(q)*sin each BF16, then BF16 add. Append original BF16-representable K/V. | Current analytic F32 rotation + one final cast does not establish this. Independently capture/qualify trig + products/add and position/frontier cases. |
| Eager attention | QK BMM -> BF16, scale -> BF16, mask in actual dtype, softmax FP32 -> BF16, value BMM -> BF16. Gate sigmoid -> BF16, multiply -> BF16, o_proj -> BF16. | Actual BMM layout/reduction, softmax and sigmoid; old GGUF source-dot/padding/FMA order is not an oracle for these shapes. |

Initial/no-cache vs cached is an explicit state phase; never infer it from an
all-zero tensor. Existing reference invokes one token with no cache, then one
with cache. Full prompt `seq_len>1` can select different GEMM/chunk code even
with a prior cache. Freeze call-shape/prefill policy in the numerical contract;
do not claim eager whole-prompt or CUDA FLA parity from tokenwise CPU tests.

## Small implementation slices and ownership

1. **Closed CPU operators (independent children).** `cpu/original_hf/{conv,
   recurrent,reduction}` remain pure source-bound operators; add coefficient,
   dtype-aware gated RMS and later full-attention/chunk operators there or in
   the corresponding existing shared primitive modules. No token/session loop.
   Reuse the same row kernel for CPU64. Actual observed inputs/outputs first;
   independent adversarial tails, FMA/separate-add and signed-zero tests second.
   Batch owner currently owns conv and initial-chunk probe; cached helper is a
   separate frozen delta. No coefficient/gated-norm owner yet.
2. **Core recording contract (one owner).** Add closed profile/cast/dtype-aware
   operation descriptors plus source-parameter kind/shape. Keep existing GGUF
   variants/default wrappers unchanged. Prefer new explicit operations (or
   explicit contracts) over a backend-global BF16 flag: profiled Project,
   RoundBF16, typed norm/gated norm, L2/replicate, phase-specific GDN. Record
   source/profile, input/output precision, state layout and phase; unknown
   semantics return Unsupported in every backend before allocation. Core must
   not depend on the Qwen model crate. Validate nodes against the selected
   profile; a string label or checksum is not mathematical capability.
3. **Loaded immutable parameter/state binding (one model owner).** Extend the
   existing role accessors to small raw learned parameter caches retaining
   original kind and shape. Use parameter_descriptor(role), never HF-to-GGUF
   aliases or A_log preconversion. Add profile-scoped state layout/phase metadata
   and geometry without session allocations. Matrix descriptors/source owner
   remain identical; only small decoded parameters, not duplicate matrices.
4. **Shared schedule + HostMath adapter (serialize after2/3).** Keep the single
   forward_token/run_layer_sequence control flow. Use role-based parameters,
   explicit source geometry and typed operations at the original boundaries.
   HostMath dispatches qualified CPU operators; PlanBuilder records the same
   operations/effects. Remove hardcoded embedding-name check in favor of the
   genuine model embedding descriptor. Do not call whole propose_gdn/propose_ffn
   inside a backend: that would hide the schedule from recording/GPU fusion.
   Keep full-model constructor/plan refusal while any reachable op is missing;
   operator/subplan qualification is permitted without making an HF session.
5. **State/codec validation (same model owner or narrow child).** Reuse QwenState,
   LayerState, SessionAuthority/RootAttempt, TaskCoordinator publication and
   StateSlot; no additional root/session owner. Canonical recurrent state stays
   value-major if retaining existing domain ABI; convert to/from the pure
   key-major CPU operator by explicit bit permutations with checked geometry.
   Conv canonical history is the newest K-1 positions; the official raw cache
   retains K. Label this semantic-state projection explicitly, rather than
   claiming byte identity for the discarded raw fourth slot. Keep raw oracle
   dumps as separate evidence. Typed state specs must include phase/layout and
   BF16-representability where required. Existing GGUF checkpointv1 stays
   closed to HF. New portable HF/profile format needs explicit schema/profile,
   source/config pins, layout/dtype/phase and finite/canonical-padding checks.
   Production algorithm revisions must never reinterpret existing diagnostic
   schema offsets0x10/0x20 or admit same-source checkpoints of another profile.
6. **GPU closed kernels/lowering (independent GPU owner after contracts).** New
   original BF16 Project/Cast entries in the existing ResidentProgram and
   BatchProgram; original packed16-bit weights, explicit activation/result
   casts, qualified reduction profile. Use original queue/ResidentService,
   physical ResourceArbiter reservations, BufferLease, Ready events and abort
   quarantine. Weight storage may share source identity, but compiled program,
   workspace/state specs and caches also key numerical profile/layout/dtype.
   No host fallback or CPU reference64 emulation may be advertised as CUDA FLA.
   Unknown new nodes must refuse rather than enter old F32/GGUF kernels.

CPU model activation comes **after** all reachable closed operators qualify,
then full original .8B all248320 logits/all42 semantic domains on independent
source-real histories, restore/stale/foreign/cancel/late-failure isolation and
resource reclamation. Only then narrow the existing guard to the exact
qualified profile/geometry/phase. GPU B1 and native B2/4/8 need separate physical
whole-model evidence through the same nodes, state authority and lease path.

## Streaming /27B / vLLM boundaries

The TensorWindow delta authenticates source bytes and owned bounded rows; it is
not a TensorSource or runnable model. Connect it later through one existing
source/row provider with retained owned windows, explicit HostMath/kernel use,
physical staging and completion lifetime. A streaming source must not satisfy
an API promising a whole borrowed tensor by assembling all55GiB in RAM. Preserve
existing SourceWeights source issuer, source binding and ResourceArbiter; no
27B-specific engine. Executable streamed27B needs the full source-verified math
and admission/placement/resource tests in addition to source-file acquisition.

The pinned .8B source-restored CPU oracle and genuine vLLM V1 runtime are
separate numeric routes. CPU64 GEMV, eager attention and chunk64 do not establish
vLLM CUDA FLA accumulation or throughput. Use explicit source/tokenizer/
generation/quantization/call-shape/resource declarations, real tokens/state
qualification and actual serialized pinned provider measurements. Never enable
an HF profile, fused mode or metadata-only family from this design.

## Reviewed source references

- `crates/zerogravity-model-qwen35/src/{lib,original_binding,forward,host_math,checkpoint}.rs`
- `crates/zerogravity-core/src/accelerator/program.rs`
- `crates/zerogravity-cpu/src/{weights,original_hf}.rs`
- `crates/zerogravity/src/original_qwen_{math,precision,state}.rs`
- `crates/zerogravity-gpu/src/{resident_program,resident_inference}.rs` and
  `resident_program/batch.rs`
- Pinned `modeling_qwen3_5.py` SHA762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b:
  RMS/gated norm218–234/841–855; conv252–291; GDN294–497/504–660;
  RoPE681–709; eager attention725–746/749–820; FFN824–838.
- Existing `captured-reference/manifest.json` and capture_reference.py under
  `.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/`:
  eager/use_kernels=false,437BF16+36sourceF32, all42 canonical domains, raw49
  cache/logit tensors. Full raw convK and canonical newestK-1 are separate.
- Current closed-op ownership/status supplied by batch_design_review; this
  source-only design executes no Rust, Torch, CUDA, model or server.

````

## Source: .agents/runs/model-execution-hf-integration-design-20261020/TODO.md

Original UTF-8 bytes: 720; SHA256 `79f456c8c6dd5995ea0490893180ddff4433ff5b22e255e63dd332c367d02c3f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original HF integration design

Parent: model-execution. Owner: baseline_review.
Scope: read-only source architecture; own ignored design only.

- [x] Inspect current MAIN9815504 loaded binding, shared schedule/backend,
  host session authority, analytical original numerics and pinned Torch source.
- [x] Inventory explicit dtype/reduction/layout/cache-phase gaps.
- [x] Recommend concrete implementation ownership/dependencies without a
  second engine or changing executable admission.
- [ ] Root decides implementation ownership and reconciles owning plan.

No MAIN/shared-plan/source edits, Cargo/GPU/runtime/model launches or commits.
This design is not implementation or numerical/physical acceptance evidence.

````

## Source: .agents/runs/model-execution-hf-transcendentals-audit-20261020/SOURCE_TOPOLOGY.md

Original UTF-8 bytes: 9842; SHA256 `bfb297d736103abebfc53b0a29bb8859d734efd6e5a74ba5a6f87c4b3206b9f7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pinned CPU unary arithmetic: source-only audit

Owner: batch_design_review. Parent: model-execution. Own ignored run only.
No Torch import/library load, tensor/model/operator replay, Cargo/GPU, production
edits or numerical-profile promotion. The executed containers only read files
and run nm/readelf/objdump; source fetches are immutable small upstream files.

## Result

The formula alone does not identify the source kernel. Activation-internal exp
uses SLEEF; standalone FP32 Tensor.exp uses MKL VML. Scalar tails use expf from
the container's libm. A single replacement exp cannot qualify all three.

Root's authentic gate comparison is retained at
../serving-and-speculation-parity-root-2026-10-19/silu-actual-comparison.json:
106/2048 and89/2048 rawFP32 SiLU words differ, maxima2.98e-8/5.96e-8. FinalBF16
products using captured weighted norms happen to match both tokens. Those final
matches do not establish exact raw SiLU or arbitrary subsequent products.

## Pins and provenance

- Image sha256 b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f.
- Torch2.13.0+cu129 git cf30153c4c131c8164ee7798e5022d810682e2cb.
- libtorch_cpu.so438418713B SHA256
  1b4ed646ce51a4c341808552426a0f9e2e2c04961955759701066d335c1631ab.
- Torch submodule SLEEF5a1d179df9cf652951b59010a2d2075372d67f68; installed
  header version3.8.0, header SHA256
  b4ddf0931fac015dfcaa7ed2ab452c7fb54b15ed908ad3c5388b64429d1d31ec.
- Original modeling SHA256
  762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b.
- Installed libm952616B SHA256
  fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0;
  libc2129424B SHA256
  3a15d66867d83762c7f2f1e37359cb8f6c5743edb369c65285cb0b1c4f7498bf.
  Package glibc2.39-0ubuntu8.9. Upstream2.39 source is separately pinned;
  installed Ubuntu binary is authoritative over an assumed unpatched release.
- pinned-source/{manifest,sleef-manifest,glibc-manifest}.json retain URLs/bytes/
  hashes. static-report.json and dependencies-report.json pin installed files.
  Installed vec256_float.h/vml.h/Loops.h contain the exact upstream header bytes
  inside the generated stable-ABI guard, with separately recorded full hashes.

All disassembly addresses below are ELF virtual addresses, not process pointers.
Compiled call paths and static dispatch initializers are established; no live
post-call function-pointer or instruction-level runtime trace was taken here.

## Activation dispatch and dtype/span dependence

| Operation | Source path | Compiled evidence |
| --- | --- | --- |
| FP32 SiLU | Activation.cpp1099–1125; direct division after exp(-x) and add1 | caller0x793d910; SLEEF calls0x793dafe/0x793db30; vaddps0x793db13/4e, vdivps0x793db1f/58 |
| FP32 sigmoid | UnaryOpsKernel.cpp32–60; reciprocal via actual division | caller0x6b86da0; SLEEF0x6b86f6b/82, vdivps0x6b86f9c/a0 |
| FP32 softplus | Activation.cpp921–961; beta/threshold scalar bindings, log1p after exp, direct division, threshold blend | caller0x7943c80; SLEEFexp0x7943e9b/eb; SLEEFlog1p0x7943ea0/ef0; vdivps0x7943ec2/0x7943f11 |
| BF16 variants | widen toFP32, same corresponding arithmetic, BF16RNE | compiled-caller-{28,35,52}.txt |
| Scalar tails/strided basic loop | std::exp/std::log1p specialization | all three compiled callers contain expf@plt, softplus also log1pf@plt |

SiLU's outer arithmetic matches existing math::silu: no reciprocal-multiply
rewrite explains the raw mismatch. Vector negation is sign-bit XOR; add/divide
are separate FP32 instructions. No zero canonicalization is justified.

Loops.h181–205 processes TWO vectors per iteration, then scalar basic_loop:
F32 vector8 =>16-element prefix; BF16 vector16 =>32-element prefix. Actual
contiguous iterator spans, not an arbitrary flattened tensor, determine tails.
Thus captured F32 norm gates2048 use the vector path. Original b.sigmoid()
(modeling617) receives BF16: .8B16heads =>all scalar;27B48heads =>32vector then
16scalar, assuming the source's contiguous single-token span. Full-attention
BF16 gates and BF16 convolution SiLU need their own observed spans as well.

## SLEEF AVX2 arithmetic that can be closed in Rust

vec256_float.h247–249 and391–392 select expf8_u10/log1pf8_u10.
Compiled expf8 AVX2 at0x116e5cd0 is251B; log1pf8 AVX2 at0x116e8ca0 is586B.
The latter has an identical-address finz alias; it remains the named u10 path.

SLEEF sleefsimdsp.c2029–2050 performs FP32 range reduction, two explicit FMA
corrections with split ln2, a degree7 FMA polynomial, separate add1 and two
power-of-two scalings. Its integer conversion is round-to-nearest under MXCSR,
not Rust's truncating float cast. Constants and bit construction are part of
the profile. Under/overflow masks are explicit; finite input may yield an
infinite private exp intermediate used by finite SiLU/sigmoid output.
helperavx2.h292–312 and compiled-selected-2.txt agree on FMA/division/negation.

SLEEF log1pf2848–2896 uses compensated two-FP32 arithmetic from df.h, including
FMA residuals and division, exponent extraction and a polynomial. It is not
equivalent to FP32 log(1+x) or a host f64 log. For softplus beta1/threshold20,
the needed exp outputs are nonnegative and bounded on the selected nonlinear
branch; this avoids the unrelated very-large-log fallback, but needs source
edge proof. Source softplus has no negative20 shortcut.

SLEEF generic entrypoints jump through writable pointers. Static pointer data
initializes to disp_expf8_u10 at0x116dd850 and disp_log1pf8_u10 at0x116db210.
Compiled/source dispatch tests AVX2 AND FMA, preferring these kernels over
AVX/FMA4. Exact dispatch assembly is retained under compiled-pnt_*.txt.
This audit did not read a live initialized pointer after an actual op.

## Standalone Tensor.exp is a separate MKL gate

UnaryOpsKernel.cpp664–740 packs strided FP32 into contiguous bounded blocks;
vml.h96–134 specializes float exp to vmsExp with
VML_HA|VML_FTZDAZ_OFF|VML_ERRMODE_IGNORE. Installed Config.h confirms MKL enabled.
Compiled float caller0x6b532d0 calls vmsExp at0x6b533f1 and other branches;
the passed mode immediate is0x140102. BF16 standalone exp instead widens through
the unspecialized vector map; do not confuse its SLEEF calls with float exp.

Compiled vmsExp0x8ba7c10 selects a CPU/mode table and dispatches through the VML
threader. dependencies-report.json lists the actual statically linked FP32
families E2/EX/H8/L9/Z0 and HA/LA/EP variants, table ELF0x14780a40. MKL2024.2
is inside the pinned libtorch CPU binary here; no separate libmkl filename is
needed to pin its bytes. This does NOT identify the selected microkernel's
arithmetic or prove SLEEF/libm equivalence. Mode/CPU/table-index/size/tail
observation and compiled-kernel derivation remain required before a Rust port.

Original GDN uses standalone FP32 exp for A_log and recurrent g; initial chunk
also applies it to cumulative, pairwise/masked and chunk-final decays. These
paths must preserve source dtype, shape/strides, signedzero and masked -inf.
The cached helper deliberately accepts decay as an already observed operand;
its acceptance cannot be stretched into an exp implementation claim.

## Recommended incremental shared seam

1. Pure original_hf::transcendentals child, same CPU crate and future typed
   MathematicalBackend profile. First closed public operation:
   silu_bf16_widened_f32_contiguous16(source,input)->privateVec<f32>.
   Bound source, finite exact-BF16-widened input, nonempty/max1M, span divisible16.
   Private exp port follows SLEEF constants/rounding/FMA/scalings exactly.
   No host exp/libTorch dependency, input-result lookup, tolerance or default
   GGUF changes. The span guard avoids claiming unqualified scalar-tail math.
2. Reuse that private exp for independently qualified vector sigmoid and
   beta1/threshold20 softplus; add the source compensated log1p core, not log.
   A future general operation must reproduce actual vector-prefix/scalar-tail
   policy and libm profile. Bind dtype/layout/span; never choose by tensor name.
3. Keep MKL-exp admission explicitly Unsupported until its actual kernel is
   derived and bit-verified. Only then compose original coefficients in source
   order: BF16 beta activation/cast; FP32 A_log exp, negation, biased alpha add,
   softplus, multiplication; separate FP32 exp(g) at the recurrent operator.
   Preserve trained raw parameter dtype/source bindings. No new token loop.

## Required numerical proof, not executed here

- New SiLU helper: every authentic4096 raw FP32 and finalBF16 product word;
  all65280 finite BF16 input patterns widened toFP32 in original vector spans;
  signs/underflows/overflow intermediates. Source-free test inputs never replace
  actual public-op outputs as oracle. Test output multiplication separately.
- Scalar-tail BF16 sigmoid: exhaustive finite-BF16 inputs in separate literal
  length16 calls, preserving the actual scalar path. One giant BF16 tensor
  would exercise a different backend. Check16/32/48/64, span boundaries and
  lane-placement invariance only within a verified path. Root must capture the
  resolved glibc expf IFUNC target and derive its compiled source arithmetic;
  a platform f32.exp() name is insufficient. libm export expf@GLIBC2.27 is IFUNC.
- Softplus: authentic biased-alpha inputs/results, threshold20 neighbors,
  large-negative/subnormal/signedzero, source log1p, parameter promotion/casts.
- MKL: capture actual exp operands/results/strides for A_log, g and initial
  chunk shapes, CPU/mode/table-selected kernel, actual binary pins. Derive that
  kernel before production equivalence. No CPU-libm/SLEEF substitution.
- After each closed helper: root Rust tests + complete authentic replay. Only
  then typed engine integration and full logits/all42 StateRoot acceptance.

The original source/profile remains unqualified at these arithmetic gaps.

````

## Source: .agents/runs/model-execution-hf-transcendentals-audit-20261020/TODO.md

Original UTF-8 bytes: 797; SHA256 `05b1771dbc76b7504bbdc42c2b767b7c0ab346e55a5739e1db2563c7036f246b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pinned CPU transcendental audit

Parent: model-execution. Owner: batch_design_review.
Scope: own ignored run, immutable source/header/ELF read-only; no Torch import,
operator/model/runtime replay, Cargo/GPU, production edits or profile promotion.

- [x] Read authentic SiLU comparison and source gates.
- [x] Trace SiLU/sigmoid/softplus vector and scalar paths.
- [x] Pin SLEEF source revision and installed compiled functions.
- [x] Distinguish direct aten.exp MKL VML from activation SLEEF exp.
- [x] Recommend shared closed helper and exact physical replay gates.
- [ ] Root independently reviews SOURCE_TOPOLOGY.md and chooses a closed slice.
- [ ] Actual live selected-provider/math replay and Rust qualification; not run
  by this source-only agent. No full-model/profile promotion claim.

````

## Source: .agents/runs/model-execution-hf-typed-oracle-20261020/README.md

Original UTF-8 bytes: 3733; SHA256 `970d6542cb4c348dc26339a13efd0724b6d30aaa4319925a5c395745331c12ad`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed embedding → projection replay

Owner: `/root/spine_review`; parent plan: model-execution. Root alone executes/builds. Prepared only; no Torch, model, Cargo, rustc, container or GPU execution by this agent.

- Genuine pinned Qwen3.5-0.8B original shard, revision `2fc06364715b967f1860aea9cf38778875588b17`; exact index/config/tokenizer pins in scripts.
- Tokens `11`, `353` independently: public `F.embedding([token], raw_embedding)` produces contiguous BF16 `[1,1024]`; public `F.linear` uses unchanged BF16 OutputInput `GdnQkv(0) [6144,1024]`, `FfnGate(0) [3584,1024]`, tied `Output [248320,1024]`, no bias.
- No input norm. QKV/FFN/head are independent raw projections of the same embedding, never complete layer0/model logits. Original F32 parameters are unused; this slice does not distinguish whole-model all-BF16-loader versus source-36-F32-restored profiles.
- Reference records actual operator metadata via a narrowly scoped TorchDispatchMode; each original operation is called once, with its result unchanged. Labels/geometry are framework dispatch observations, not machine-instruction traces or independently verified provider arithmetic.
- Original shard hashes before/after, raw matrix extents/digests, Torch commit/version/AVX2/four threads, actual compiled-library hashes and script SHA bind the observation. The receipt hash alone is no independent provenance proof.
- Eight F32LE payloads preserve exact widened BF16 words: `2,072,576` bytes total (<4MiB), including both complete 248320-wide outputs. Weight bytes remain raw BF16; no model loader, learned cache or weight cast.
- Root container: pinned acquired image; 12GiB memory+swap cap, four CPUs, four operator threads, no network/GPU/devices, source mount read-only; exclusive output/invocation refuses overwrite. Source/raw-matrix hashing streams ≤1MiB buffers; safetensors supplies original reference weights.
- Rust uses `SourceWeights::prepare_verified_shards(...).materialize()` with original retained composite raw+metadata lease ≤4GiB, then `LoadedQwenModel::from_original_source` and existing `execute_embedding_projection_slice`. It checks exact per-word outputs, raw descriptor/hash/pointer preservation, same recorder's two typed nodes/no domains/no frontier, zero initialized state, closed full session/plan and final source reservation zero.
- The 4GiB configured host ledger accounts for the existing source owner; this diagnostic does not claim whole-process physical RAM accounting. The reference's RSS/cgroup controls are separate.
- Any bit difference emits `UNQUALIFIED` and process exit2, without tolerances, arithmetic fixes or fallback. Errors also retain a diagnostic and require source reservation zero.

Root execution, when scheduling allows:

```sh
python3 .agents/runs/model-execution-hf-typed-oracle-20261020/run_probe.py
python3 .agents/runs/model-execution-hf-typed-oracle-20261020/validate_oracle.py
```

Compile `replay_operators.rs` against freshly built matching release artifacts for `zerogravity-artifacts(cache)`, `zerogravity-core`, `zerogravity-model-qwen35(cpu,original-hf-binding)`, serde_json and sha2. Root may use the ignored standalone Cargo.toml; no crate lock/dependencies have been acquired here. With rustc use root's current artifact map and consistent thin-LTO/bitcode/codegen flags; no stale library reuse.

```text
replay_operators SOURCE_DIRECTORY CAPTURED_OPERATORS_DIRECTORY ACTUAL_MANIFEST_SHA256 NEW_EXCLUSIVE_OUTPUT_DIRECTORY
```

Acceptance remains only the six pure typed operator cases. No SessionAuthority/StateRoot publication, trained complete graph, HF profile eligibility, GPU/FLA, Qwen3.8 27B, streaming transfer, serving, vLLM or performance gate follows from this replay.

````

## Source: .agents/runs/model-execution-hf-typed-oracle-20261020/TODO.md

Original UTF-8 bytes: 1406; SHA256 `0f0429710fabd0a2010264184122c57909de83458b86227999fbbad27c1151c3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Typed HF operator oracle preparation

Parent: model-execution. Owner: `/root/spine_review`. Scope: this ignored run only; root owns all execution, MAIN source/plans and commits.

- [x] Read new same-engine operator API, original source binding and accepted source-admission pattern.
- [x] Prepare pinned public BF16 embedding→three independent F.linear operator capture, bounded actual-dispatch metadata and <4MiB complete output payload.
- [x] Prepare exclusive no-network/no-GPU ≤12GiB/four-CPU root container launcher.
- [x] Prepare source-free byte/layout/receipt validator, no reconstructed arithmetic.
- [x] Prepare source-bound existing-engine Rust replay with full bit comparisons, two typed recorded nodes/no state/frontier, source composite ledger and final drop-zero checks.
- [x] Python syntax-only AST and owned Rust formatting pass; original source pins matched existing receipt using stdlib metadata reads only.
- [ ] Root executes capture and validator; inspect actual dispatch/payload/source/runtime receipts.
- [ ] Root compiles replay against fresh merged release crates, runs all six authentic cases and reviews mismatch/lease/guard results.

No runtime/Cargo/rustc/container/model/GPU execution, tracked edits, tolerance changes or profile promotion by this agent. Reciprocal raw parameter/state patch `74688961` review separately reported no blocker within metadata-only scope.

````

## Source: .agents/runs/model-execution-hf2b-attention-paired-20261020/README.md

Original UTF-8 bytes: 5720; SHA256 `dd70558e84a4e5fa26919eee6b9e30bf5ae58b9eccf9395712ef37127200f3a6`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original HF2B paired attention observation — ROOT executes

- Owner: baseline_review. Parent: existing model-execution plan; ignored preparation only.
- Source-only, unexecuted. No model/container/operator/test/build execution by author.
- Exact original Qwen/Qwen3.5-2B revision15852e8c16360a2fea060d615a32b45270f8a8fc,
  snapshot88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52.
- CPU Transformers eager/slow GDN in the already acquired pinned image. This is
  not vLLM inference or GPU evidence despite that image's installed packages.
- No production source, shared plans, existing run bytes, weights or math changed.

## Existing accepted reference

`../model-execution-hf2b-cpu-reference-20261020/root-probe-2/captured-reference`
is immutable. `parent-pins.json` pins its actual manifest/validation and original
invocation, accepted capture/validator/source metadata, Batch's frozen attention
hook and the reviewed GEMM `owned_container_v4.py` (SHA53ee4270…). Original raw
archive hashes come from that pinned accepted manifest; author did not scan or
load raw states, model payloads or large DSOs.

Both source profiles remain separate: `official-bf16-loader` then
`source-36-f32-restored`, each teacher11 then353 with its own fresh cache. The
original loader/casts, source/RECORD audits, parameter audit, module hooks,
token calls, snapshot codecs and unchanged archive validator remain in use.
`CAPTURE_INTEGRATION.review.diff` exposes every insertion. Original named
functions are AST-identical except delegation at the beginning of the existing
`observe_calls` callback. No replacement profiler, tensor, operator or model.

## New observation and transparency gates

- Fresh serialized plain then observed containers, each running the accepted
  two-profile/two-step procedure. No historical failed attempt is reused.
- Copied exact `AttentionProjectionObserver` observes six FullAttention layers
  3/7/11/15/19/23, nine BF16 tensors each, four profile/steps:216 tensors total.
  Stages: hidden, query/gate projection, K/V projections, raw query/key,
  normalized query/key, gate. Actual original module hooks and source-return
  gate copy only; original profiler call counts remain authoritative.
- Expected genuine2B shapes: hidden/gate2048, query/gate projection4096,
  query8×256, K/V2×256. Raw query stride512 preserves interleaved query/gate.
  These are stage transport descriptors, not reconstructed expected arithmetic.
- Every fresh mode must retain all196 raw tensors, four full248320 BF16 logits,
 168 canonical domains and60 old intermediates byte-identical to the accepted
  reference. Each mode also passes the unchanged complete archive/layout
  validator;632 source/617 loaded parameter audit bytes match exactly.
- Stdlib comparator reads bounded Safetensors descriptors and exact payloads,
  including signed zeros; no Torch import, rounding, tolerance or tensor math.
  Source/profiles/teachers/all-layer stage namespace, finite raw BF16,
  dtype/shape/strides/cumulative byte cap and all cleanup evidence are required.
- All base and new hooks are owned inside one finally boundary, every removal
  attempted, profiler retired, primary and cleanup failures retained. Observer
  module references are released before model disposal/archive validation.

## Bounds and lifetime

- Existing imageIDf57d77f5…/codec manifest eec72f47…; exact local inspect, no pull.
- No NVIDIA CDI/device, CUDA_VISIBLE_DEVICES=-1, NVIDIA_VISIBLE_DEVICES=void,
  no preload/audit, networknone, read-only source/probe/rootfs, cap-dropall.
- Memory+swap12GiB, CPUquota4, original4 Torch threads/interops1, AVX2. Affinity
  sets exactly four allowed CPUs before Torch import and validates afterward;
  no unavailable OCI cpuset controller is requested.
- Each capture300s outer/290s original alarm; original aggregate capture256MiB,
  per-file256MiB, helper log4MiB. Additional attention raw payload172032B/step,
  688128B total; its independent cumulative cap64MiB remains unchanged.
- Original4.548GB checkpoint/model memory estimate remains an estimate, not
  physical admission. Actual cgroup peak/RSS remain in each reference manifest.
- Reviewed helper is copied unchanged: WNOWAIT keeps owned PID reserved;
  independent TERM/grace/two KILL stages precede reap, exact nonce/name/CID
  discovery before and after, group/container absence and pipe EOF/closure.
  Postspawn receipt errors cannot bypass cleanup. Failed evidence prevents the
  next mode and final PASS. Existing processes/containers are never targeted.
- Final bounded stdlib comparison120s, block≤1MiB, tensor/archive geometry and
  complete EOF enforced. Separate fresh output directly beneath this run.

## ROOT-only commands after source review

```sh
cd /home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_validate_pair
PYTHONDONTWRITEBYTECODE=1 python3 run_pair.py --output "$PWD/root-pair-1"
```

Ten fixture methods are prepared, not run by author. Synthetic bytes test strict
raw comparison, missing stages, teacher/profile binding, query stride and finite
guards; they are never genuine trained expectations. ROOT should independently
authenticate all prepared source and parent pins before either command.

Final possible scope: `PASS_CPU_HF2B_PAIRED_ATTENTION_OBSERVATION_ONLY`. This
qualifies observation transparency and actual reference-stage availability for
the named two CPU profiles/two teacher tokens. It does not qualify Rust private
slice output, full HF execution/SessionAuthority/StateRoot, GPU/FLA, attention
RoPE/BMM/softmax integration, long context, trained27B or matched performance.

````

## Source: .agents/runs/model-execution-hf2b-attention-paired-20261020/TODO.md

Original UTF-8 bytes: 1186; SHA256 `27f2a0f06e897c887eba5c8f341391cd08e79e6c39ca215160ee51613e83a61e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Own run — original HF2B paired attention observation

- Parent: `.agents/plans/model-execution/TODO.md`; owner baseline_review.
- Owned scope: this ignored preparation directory only; no shared tracker.
- [x] Read accepted two-profile CPU capture/validator/root-probe-2 metadata and Batch frozen hook.
- [x] Preserve exact original source/parameter/semantic/raw math path; finally-owned observation integration.
- [x] Prepare exact reviewed owned-container lifecycle, serialized fresh plain/observed launcher and strict raw comparator.
- [x] Prepare216 actual-stage contract and10 source-free fixtures; retain explicit CPU-only/reference-only labels.
- [x] AST/static source checks and frozen source/parent pins only; no executed tests or model/runtime claims.
- [ ] Independent source review and ROOT fixture execution.
- [ ] ROOT paired capture; each mode all196 raw tensors/full4 logits/all168 domains/60 intermediates byte-exact to accepted reference.
- [ ] ROOT actual216 attention stages, source audit, cgroup/affinity and full owned cleanup validation.
- [ ] Later separate Rust operator-slice replay against these authentic stages; full HF/session/GPU guards remain closed.

````

## Source: .agents/runs/model-execution-hf2b-attention-paired-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 5720; SHA256 `dd70558e84a4e5fa26919eee6b9e30bf5ae58b9eccf9395712ef37127200f3a6`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original HF2B paired attention observation — ROOT executes

- Owner: baseline_review. Parent: existing model-execution plan; ignored preparation only.
- Source-only, unexecuted. No model/container/operator/test/build execution by author.
- Exact original Qwen/Qwen3.5-2B revision15852e8c16360a2fea060d615a32b45270f8a8fc,
  snapshot88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52.
- CPU Transformers eager/slow GDN in the already acquired pinned image. This is
  not vLLM inference or GPU evidence despite that image's installed packages.
- No production source, shared plans, existing run bytes, weights or math changed.

## Existing accepted reference

`../model-execution-hf2b-cpu-reference-20261020/root-probe-2/captured-reference`
is immutable. `parent-pins.json` pins its actual manifest/validation and original
invocation, accepted capture/validator/source metadata, Batch's frozen attention
hook and the reviewed GEMM `owned_container_v4.py` (SHA53ee4270…). Original raw
archive hashes come from that pinned accepted manifest; author did not scan or
load raw states, model payloads or large DSOs.

Both source profiles remain separate: `official-bf16-loader` then
`source-36-f32-restored`, each teacher11 then353 with its own fresh cache. The
original loader/casts, source/RECORD audits, parameter audit, module hooks,
token calls, snapshot codecs and unchanged archive validator remain in use.
`CAPTURE_INTEGRATION.review.diff` exposes every insertion. Original named
functions are AST-identical except delegation at the beginning of the existing
`observe_calls` callback. No replacement profiler, tensor, operator or model.

## New observation and transparency gates

- Fresh serialized plain then observed containers, each running the accepted
  two-profile/two-step procedure. No historical failed attempt is reused.
- Copied exact `AttentionProjectionObserver` observes six FullAttention layers
  3/7/11/15/19/23, nine BF16 tensors each, four profile/steps:216 tensors total.
  Stages: hidden, query/gate projection, K/V projections, raw query/key,
  normalized query/key, gate. Actual original module hooks and source-return
  gate copy only; original profiler call counts remain authoritative.
- Expected genuine2B shapes: hidden/gate2048, query/gate projection4096,
  query8×256, K/V2×256. Raw query stride512 preserves interleaved query/gate.
  These are stage transport descriptors, not reconstructed expected arithmetic.
- Every fresh mode must retain all196 raw tensors, four full248320 BF16 logits,
 168 canonical domains and60 old intermediates byte-identical to the accepted
  reference. Each mode also passes the unchanged complete archive/layout
  validator;632 source/617 loaded parameter audit bytes match exactly.
- Stdlib comparator reads bounded Safetensors descriptors and exact payloads,
  including signed zeros; no Torch import, rounding, tolerance or tensor math.
  Source/profiles/teachers/all-layer stage namespace, finite raw BF16,
  dtype/shape/strides/cumulative byte cap and all cleanup evidence are required.
- All base and new hooks are owned inside one finally boundary, every removal
  attempted, profiler retired, primary and cleanup failures retained. Observer
  module references are released before model disposal/archive validation.

## Bounds and lifetime

- Existing imageIDf57d77f5…/codec manifest eec72f47…; exact local inspect, no pull.
- No NVIDIA CDI/device, CUDA_VISIBLE_DEVICES=-1, NVIDIA_VISIBLE_DEVICES=void,
  no preload/audit, networknone, read-only source/probe/rootfs, cap-dropall.
- Memory+swap12GiB, CPUquota4, original4 Torch threads/interops1, AVX2. Affinity
  sets exactly four allowed CPUs before Torch import and validates afterward;
  no unavailable OCI cpuset controller is requested.
- Each capture300s outer/290s original alarm; original aggregate capture256MiB,
  per-file256MiB, helper log4MiB. Additional attention raw payload172032B/step,
  688128B total; its independent cumulative cap64MiB remains unchanged.
- Original4.548GB checkpoint/model memory estimate remains an estimate, not
  physical admission. Actual cgroup peak/RSS remain in each reference manifest.
- Reviewed helper is copied unchanged: WNOWAIT keeps owned PID reserved;
  independent TERM/grace/two KILL stages precede reap, exact nonce/name/CID
  discovery before and after, group/container absence and pipe EOF/closure.
  Postspawn receipt errors cannot bypass cleanup. Failed evidence prevents the
  next mode and final PASS. Existing processes/containers are never targeted.
- Final bounded stdlib comparison120s, block≤1MiB, tensor/archive geometry and
  complete EOF enforced. Separate fresh output directly beneath this run.

## ROOT-only commands after source review

```sh
cd /home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_validate_pair
PYTHONDONTWRITEBYTECODE=1 python3 run_pair.py --output "$PWD/root-pair-1"
```

Ten fixture methods are prepared, not run by author. Synthetic bytes test strict
raw comparison, missing stages, teacher/profile binding, query stride and finite
guards; they are never genuine trained expectations. ROOT should independently
authenticate all prepared source and parent pins before either command.

Final possible scope: `PASS_CPU_HF2B_PAIRED_ATTENTION_OBSERVATION_ONLY`. This
qualifies observation transparency and actual reference-stage availability for
the named two CPU profiles/two teacher tokens. It does not qualify Rust private
slice output, full HF execution/SessionAuthority/StateRoot, GPU/FLA, attention
RoPE/BMM/softmax integration, long context, trained27B or matched performance.

````

## Source: .agents/runs/model-execution-hf2b-attention-paired-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1186; SHA256 `27f2a0f06e897c887eba5c8f341391cd08e79e6c39ca215160ee51613e83a61e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Own run — original HF2B paired attention observation

- Parent: `.agents/plans/model-execution/TODO.md`; owner baseline_review.
- Owned scope: this ignored preparation directory only; no shared tracker.
- [x] Read accepted two-profile CPU capture/validator/root-probe-2 metadata and Batch frozen hook.
- [x] Preserve exact original source/parameter/semantic/raw math path; finally-owned observation integration.
- [x] Prepare exact reviewed owned-container lifecycle, serialized fresh plain/observed launcher and strict raw comparator.
- [x] Prepare216 actual-stage contract and10 source-free fixtures; retain explicit CPU-only/reference-only labels.
- [x] AST/static source checks and frozen source/parent pins only; no executed tests or model/runtime claims.
- [ ] Independent source review and ROOT fixture execution.
- [ ] ROOT paired capture; each mode all196 raw tensors/full4 logits/all168 domains/60 intermediates byte-exact to accepted reference.
- [ ] ROOT actual216 attention stages, source audit, cgroup/affinity and full owned cleanup validation.
- [ ] Later separate Rust operator-slice replay against these authentic stages; full HF/session/GPU guards remain closed.

````

## Source: .agents/runs/model-execution-hf2b-cpu-reference-20261020/README.md

Original UTF-8 bytes: 6335; SHA256 `509cace754f859d6e37444598fb47ca6b670ba228fec61c97386e95f7251003f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original HF 2B CPU reference — ROOT EXECUTES

Parent: model-execution / serving-and-speculation. Owner: spine_review.
Prepared source only; no Torch import, model/operator/container or GPU
execution by this agent. No production/model math or full-HF guard changes.

- Original Qwen/Qwen3.5-2B revision15852e8c16360a2fea060d615a32b45270f8a8fc;
  acquired snapshot88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52.
  ROOT independently authenticated all12 official Git/LFS files and632 tensor
  headers. Own source-pins.json binds that exact report, cache manifest, source
  files, original .8B capture/validator, runtime source and package RECORDs.
  No new acquisition or source-byte conversion.
- Raw source shard4,548,221,488B SHAaa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1;
  raw596 BF16+36 F32 tensors;320 ordinary text tensors. Tied embedding/head;
  no lm_head source tensor. Full conditional-generation class loads original
  visual parameters too, but invokes text-only inputs. Fifteen MTP parameters
  are excluded by unchanged official loading policy; every other source
  parameter is audited against actual loaded bytes after its declared cast.
- Exact already accepted derived imageIDf57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe;
  base official vLLM0.31.0-cu129 imageb18abb2d… with only official CPU TorchCodec
  packaging change. Pinned Torch2.13.0+cu129/Transformers5.17 and their RECORDs,
  five exact Transformers source files and all audited package versions are
  required before model calls and rechecked afterward. This is Transformers
  CPU inference, not a vLLM/GPU reference merely because image includes vLLM.

## Exact capture

1. Unmodified official Qwen3_5ForConditionalGeneration.from_pretrained with
   requested BF16, eager attention, use_kernels=False, local_files_only=True.
   No monkeypatch, replacement operator, synthesized expected or new engine.
2. Separate `official-bf16-loader` profile first; preserve its actual parameter
   dtype/cast audit. Then restore ONLY36 originally F32 A_log/GDN gamma tensors
   from exact source bytes and capture separately as `source-36-f32-restored`.
   Fresh cache for each profile. Never merge their numerical/profile claims.
3. Exact teacher token11 then353; one original cached call/token; no template,
   sampling, EOS inference or generation policy. Source/module observation
   requires18 actual slow chunk calls on first token,18 actual slow recurrent
   calls on next,18 original conv and6 eager attention calls each step.
4. Per step save full248320 raw BF16 logits and all48 original cache tensors:
   18 BF16 conv[1,6144,4],18 F32 recurrent[1,16,128,128] key-major,
   six BF16 K/V pairs[1,2,valid1or2,256]. Store actual dtype/shape/strides.
   Preserve all signed-zero bits in raw Safetensors.
5. Separately derive42 canonical semantic files: newest3 of raw4 conv history,
   recurrent key→value-major transpose, and each attention domain K thenV with
   token/head/width order. These are lossless documented codec/layout changes,
   not independently reconstructed cache mathematics. Unused future capacity
   and padding are not exported or claimed. No SessionAuthority/Ready/StateRoot
   execution claim from this external reference.
6. Fifteen actual original layer0/final outputs via non-replacing module hooks
   and slow-conv return observation, copied unchanged. No arithmetic synthesis.
   Archive validator requires all49 raw tensors/all42 UNIQUE domains per step,
   full raw bit equality of each codec derivation,60 actual stage tensors,
   four full logits and168 domains overall. It reports cross-profile changes
   without tolerance or numerical Rust/GPU acceptance.

## Bounds and cleanup

- Container12GiB memory+swap cap,4 CPUs/threads, networknone/offline, read-only
  source/probe mounts, no CDI/NVIDIA devices, cap-dropall/no-new-privileges.
  Explicit ATEN_CPU_CAPABILITY=avx2; same accepted runtime CPU path. No caches
  or inference engine files written outside own exclusive output.
- Whole source4.548GB; conservative source+owned model allowance2×shard plus
  2GiB runtime,128MiB active state/workspace and8MiB audit scratch totals
  11,386,532,960B versus12,884,901,888B cap (~1.50GB margin). This is an
  estimate, not a physical-memory guarantee. Actual cgroup peak/RSS are saved;
  OOM/refusal stays UNQUALIFIED. Safetensors pages are original read-only file
  mappings; no complete source Vec/preconverted checkpoint is created here.
- Parameter audit finite/equality/cast-delta checks chunk≤262144 elements
  (≤1MiB/F32 operand), avoiding a whole embedding temporary; official model
  arithmetic is untouched. Actual model intermediates remain library-owned.
- Capture artifacts≤256MiB aggregate; four step raw+canonical states expected
  ~160MiB plus observed tensors/JSON. Per-file RLIMIT_FSIZE256MiB. Raw source
  itself is mounted, never copied into output. Log≤8MiB. Outer wall300s,
  inner alarm290s; cleanup has separate finite deadlines and preserves failure.
- Every postspawn operation is inside finally cleanup. UUID nonce name +owner
  label +inspected exact CID bind all stop/kill/rm. Empty startup CID may use
  that inspected name/label; malformed nonempty CID refuses mutation. Owned
  CLI is reaped before final container rediscovery/absence check. Selector and
  stdout close separately. No foreign containers/processes touched.
- Validation shares the same bounded CPU container after capture and model
  release. Complete source/RECORD pins checked before/after; scripts pinned
  before/after by launcher. Old .8B artifacts and failed attempts unchanged.

## Root invocation

Review source-pins.json and freeze-manifest.json against their supplied hashes;
no build/install/acquisition needed. Invoke only when root CPU scheduling allows:

```bash
python3 .agents/runs/model-execution-hf2b-cpu-reference-20261020/run_probe.py \
  --output .agents/runs/model-execution-hf2b-cpu-reference-20261020/root-probe-1
```

A successful artifact validator proves genuine reference/archive closure in the
named two CPU profiles and two teacher tokens only. Full-HF engine execution,
GPU FLA, matched vLLM, throughput, quantization, long context and trained27B
remain outside this capture and cannot be inferred from its source audit.

````

## Source: .agents/runs/model-execution-hf2b-cpu-reference-20261020/TODO.md

Original UTF-8 bytes: 883; SHA256 `19d44f03bd908213dac34f21b392a8616a08f9ccb94930326aa41629f6f2b724`.
Resolve embedded relative paths against the original path, not this volume.

````text
# model-execution — original HF2B CPU reference

Owner: spine_review. Own ignored run only. ROOT schedules/executed all runtime.

- [x] Read authentic prior .8B two-teacher capture/validator and root actual2B
  acquisition/header receipt; keep profiles/source/runtime pins separate.
- [x] Prepare unchanged official2B capture with bounded audit temporaries,
  full logits/all raw42-domain layouts, source/hash/dtype/operator closure.
- [x] Prepare bounded owned offline12GiB/4CPU/300s runner and archive validator.
- [x] AST/source-only review. No Torch import/model/container executed.
- [ ] Independent source review; ROOT actual pinned CPU capture/validation.
- [ ] Actual memory/finite/cleanup and complete source recheck receipts.
- [ ] Independent shared-engine numerical acceptance/GPU FLA/matched vLLM:
  separate future gates, never implied by this reference preparation.

````

## Source: .agents/runs/model-execution-hf2b-cpu-reference-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 6335; SHA256 `509cace754f859d6e37444598fb47ca6b670ba228fec61c97386e95f7251003f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original HF 2B CPU reference — ROOT EXECUTES

Parent: model-execution / serving-and-speculation. Owner: spine_review.
Prepared source only; no Torch import, model/operator/container or GPU
execution by this agent. No production/model math or full-HF guard changes.

- Original Qwen/Qwen3.5-2B revision15852e8c16360a2fea060d615a32b45270f8a8fc;
  acquired snapshot88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52.
  ROOT independently authenticated all12 official Git/LFS files and632 tensor
  headers. Own source-pins.json binds that exact report, cache manifest, source
  files, original .8B capture/validator, runtime source and package RECORDs.
  No new acquisition or source-byte conversion.
- Raw source shard4,548,221,488B SHAaa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1;
  raw596 BF16+36 F32 tensors;320 ordinary text tensors. Tied embedding/head;
  no lm_head source tensor. Full conditional-generation class loads original
  visual parameters too, but invokes text-only inputs. Fifteen MTP parameters
  are excluded by unchanged official loading policy; every other source
  parameter is audited against actual loaded bytes after its declared cast.
- Exact already accepted derived imageIDf57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe;
  base official vLLM0.31.0-cu129 imageb18abb2d… with only official CPU TorchCodec
  packaging change. Pinned Torch2.13.0+cu129/Transformers5.17 and their RECORDs,
  five exact Transformers source files and all audited package versions are
  required before model calls and rechecked afterward. This is Transformers
  CPU inference, not a vLLM/GPU reference merely because image includes vLLM.

## Exact capture

1. Unmodified official Qwen3_5ForConditionalGeneration.from_pretrained with
   requested BF16, eager attention, use_kernels=False, local_files_only=True.
   No monkeypatch, replacement operator, synthesized expected or new engine.
2. Separate `official-bf16-loader` profile first; preserve its actual parameter
   dtype/cast audit. Then restore ONLY36 originally F32 A_log/GDN gamma tensors
   from exact source bytes and capture separately as `source-36-f32-restored`.
   Fresh cache for each profile. Never merge their numerical/profile claims.
3. Exact teacher token11 then353; one original cached call/token; no template,
   sampling, EOS inference or generation policy. Source/module observation
   requires18 actual slow chunk calls on first token,18 actual slow recurrent
   calls on next,18 original conv and6 eager attention calls each step.
4. Per step save full248320 raw BF16 logits and all48 original cache tensors:
   18 BF16 conv[1,6144,4],18 F32 recurrent[1,16,128,128] key-major,
   six BF16 K/V pairs[1,2,valid1or2,256]. Store actual dtype/shape/strides.
   Preserve all signed-zero bits in raw Safetensors.
5. Separately derive42 canonical semantic files: newest3 of raw4 conv history,
   recurrent key→value-major transpose, and each attention domain K thenV with
   token/head/width order. These are lossless documented codec/layout changes,
   not independently reconstructed cache mathematics. Unused future capacity
   and padding are not exported or claimed. No SessionAuthority/Ready/StateRoot
   execution claim from this external reference.
6. Fifteen actual original layer0/final outputs via non-replacing module hooks
   and slow-conv return observation, copied unchanged. No arithmetic synthesis.
   Archive validator requires all49 raw tensors/all42 UNIQUE domains per step,
   full raw bit equality of each codec derivation,60 actual stage tensors,
   four full logits and168 domains overall. It reports cross-profile changes
   without tolerance or numerical Rust/GPU acceptance.

## Bounds and cleanup

- Container12GiB memory+swap cap,4 CPUs/threads, networknone/offline, read-only
  source/probe mounts, no CDI/NVIDIA devices, cap-dropall/no-new-privileges.
  Explicit ATEN_CPU_CAPABILITY=avx2; same accepted runtime CPU path. No caches
  or inference engine files written outside own exclusive output.
- Whole source4.548GB; conservative source+owned model allowance2×shard plus
  2GiB runtime,128MiB active state/workspace and8MiB audit scratch totals
  11,386,532,960B versus12,884,901,888B cap (~1.50GB margin). This is an
  estimate, not a physical-memory guarantee. Actual cgroup peak/RSS are saved;
  OOM/refusal stays UNQUALIFIED. Safetensors pages are original read-only file
  mappings; no complete source Vec/preconverted checkpoint is created here.
- Parameter audit finite/equality/cast-delta checks chunk≤262144 elements
  (≤1MiB/F32 operand), avoiding a whole embedding temporary; official model
  arithmetic is untouched. Actual model intermediates remain library-owned.
- Capture artifacts≤256MiB aggregate; four step raw+canonical states expected
  ~160MiB plus observed tensors/JSON. Per-file RLIMIT_FSIZE256MiB. Raw source
  itself is mounted, never copied into output. Log≤8MiB. Outer wall300s,
  inner alarm290s; cleanup has separate finite deadlines and preserves failure.
- Every postspawn operation is inside finally cleanup. UUID nonce name +owner
  label +inspected exact CID bind all stop/kill/rm. Empty startup CID may use
  that inspected name/label; malformed nonempty CID refuses mutation. Owned
  CLI is reaped before final container rediscovery/absence check. Selector and
  stdout close separately. No foreign containers/processes touched.
- Validation shares the same bounded CPU container after capture and model
  release. Complete source/RECORD pins checked before/after; scripts pinned
  before/after by launcher. Old .8B artifacts and failed attempts unchanged.

## Root invocation

Review source-pins.json and freeze-manifest.json against their supplied hashes;
no build/install/acquisition needed. Invoke only when root CPU scheduling allows:

```bash
python3 .agents/runs/model-execution-hf2b-cpu-reference-20261020/run_probe.py \
  --output .agents/runs/model-execution-hf2b-cpu-reference-20261020/root-probe-1
```

A successful artifact validator proves genuine reference/archive closure in the
named two CPU profiles and two teacher tokens only. Full-HF engine execution,
GPU FLA, matched vLLM, throughput, quantization, long context and trained27B
remain outside this capture and cannot be inferred from its source audit.

````

## Source: .agents/runs/model-execution-hf2b-cpu-reference-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 883; SHA256 `19d44f03bd908213dac34f21b392a8616a08f9ccb94930326aa41629f6f2b724`.
Resolve embedded relative paths against the original path, not this volume.

````text
# model-execution — original HF2B CPU reference

Owner: spine_review. Own ignored run only. ROOT schedules/executed all runtime.

- [x] Read authentic prior .8B two-teacher capture/validator and root actual2B
  acquisition/header receipt; keep profiles/source/runtime pins separate.
- [x] Prepare unchanged official2B capture with bounded audit temporaries,
  full logits/all raw42-domain layouts, source/hash/dtype/operator closure.
- [x] Prepare bounded owned offline12GiB/4CPU/300s runner and archive validator.
- [x] AST/source-only review. No Torch import/model/container executed.
- [ ] Independent source review; ROOT actual pinned CPU capture/validation.
- [ ] Actual memory/finite/cleanup and complete source recheck receipts.
- [ ] Independent shared-engine numerical acceptance/GPU FLA/matched vLLM:
  separate future gates, never implied by this reference preparation.

````

## Source: .agents/runs/model-execution-hf2b-source-plan-20261020/HANDOFF.md

Original UTF-8 bytes: 7021; SHA256 `a7e6ffe0ac8d5f13539c0b6e23ccb77b59e479a3af0b79fb0eb53ad3a9ecc4d9`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original Qwen3.5 2B source acquisition handoff

Metadata-only audit for ROOT after benchmark timing; no weight/header download,
whole-weight/DSO read, model/operator/runtime execution, or source-code change.
Parent: serving-and-speculation + model-execution; owner: spine_review.
This is an acquisition handoff, not another work tracker or model qualification.

## Exact official source

- Repository `Qwen/Qwen3.5-2B`; public/ungated, Apache2.0.
- Revision `15852e8c16360a2fea060d615a32b45270f8a8fc` (official API at audit).
- Single indexed shard `model.safetensors-00001-of-00001.safetensors`:
  4,548,221,488B; provider LFS SHA256
  `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`.
  Git pointerOID `3868ead7f1658210f24643fa927afa7d9717083c`.
- Index 64,460B, actual fetched SHA256
  `aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9`,
  Git blob `69a41758e1138259916d08b99b56fada441fd502` authenticated to API.
- Index payload 4,548,144,832B; 632 names, all assigned to that shard:
  320 ordinary text +297 visual +15 MTP. No separate `lm_head.weight`.
- Complete original index-first closure: 12 files/4,571,272,453B. All repository
  files except .gitattributes; includes README/LICENSE and known sidecars.
  Header+length-prefix size 76,656B is only file-minus-index arithmetic;
  actual header/offset/dtype closure remains UNVERIFIED until acquisition.
- Official aggregate parameter metadata: BF16 2,274,067,232 elements;
  F32 2,592 elements; total 2,274,069,824. These exactly account for indexed
  payload bytes. This is mixed raw BF16/F32, not an all-BF16 checkpoint.
- Index has 18 A_log +18 linear_attn.norm.weight names; their familiar
  16/128 shapes would explain all2592F32 elements. This per-name dtype/shape
  assignment remains a source-derived EXPECTATION until authentic header audit.
  No range request or weight/header byte was fetched by this audit.

Official primary metadata:
- https://huggingface.co/api/models/Qwen/Qwen3.5-2B/revision/15852e8c16360a2fea060d615a32b45270f8a8fc?blobs=true
- https://huggingface.co/api/models/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc?recursive=false&expand=true
- https://huggingface.co/Qwen/Qwen3.5-2B/raw/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json
- https://huggingface.co/Qwen/Qwen3.5-2B/raw/15852e8c16360a2fea060d615a32b45270f8a8fc/model.safetensors.index.json

## Configuration/tokenizer profile

- Config 2,908B SHA256
  `ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4`;
  raw Git blob `d30a15be1bbfd610f58ad80f1fbeb6778f58a80f` verified.
- Qwen3_5ForConditionalGeneration/qwen3_5 wrapper; text qwen3_5_text,
  hidden2048/FFN6144/vocab248320/tied embedding,24layers (18GDN+6fullattn).
- Full attention: 8Q/2KV heads, head256, gated Q, biasfalse. GDN:
  16key/16value heads,width128,K4 convolution. RMS epsilon1e-6, SiLU;
  source dtype bfloat16, mamba_ssm_dtype float32. Max context262144;
  theta10000000/partialRoPE0.25/mrope interleaved sections11,11,10.
- Expected original text shapes, not yet header-verified: embedding
  [248320,2048], FFN gate/up [6144,2048], down [2048,6144], GDN QKV
  [6144,2048], Z/out [2048,2048], a/b [16,2048], convolution[6144,1,4];
  full-attention Q[4096,2048], K/V[512,2048], O[2048,2048], Q/K norms[256].
- Tokenizer LFS SHA256 (12,807,982B):
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.
  Same authenticated tokenizer Git/LFS IDs as acquired original0.8B.
- tokenizer_config 16,709B SHA256
  `49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c`
  and blob `fae3ce993e07c092ad024dde45e592379fde91bb`: actual metadata
  fetch verifies identity, matching0.8B receipt. Chat template blob
  `0ef09f214eaa6d9bca297988afc1454b5827b2c7` also matches0.8B.
- Qwen2Tokenizer, no BOS autoadd; config EOS/endoftext248044 versus tokenizer
  EOS/im_end248046, im_start248045. No generation_config.json listed.
  Matched benchmark must explicitly agree on prompt/template/thinking and both
  stop/EOS policies rather than silently inherit different defaults.

## Existing bounded Rust acquisition route (ROOT ONLY, AFTER timing)

No original HF2B named source receipt found in external-baselines/models; the
canonical managed cache currently holds only original Qwen3.8-27B. No cache
blob payloads were read. Existing cache policy is64GiB; retain that policy.
Use ROOT's independently verified acquisition-enabled release binary, e.g. its
existing zerogravity-acquire-release copy; no build/install is requested here.

```sh
TASK_HF2B_ACQUIRE=/absolute/ROOT_verified_acquisition_binary
"$TASK_HF2B_ACQUIRE" pull hf Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --file model.safetensors.index.json \
  --cache /home/pvl/.cache/abird/zerogravity/models \
  --disk-limit-bytes 68719476736 --memory-limit-bytes 134217728 \
  --max-file-bytes 5368709120 --max-snapshot-bytes 5368709120 \
  --max-files 16 --max-metadata-bytes 8388608 --max-pages 4 \
  --timeout-seconds 120 --dry-run
```

Inspect exact pinned revision,12files/4,571,272,453B and independent LFS pins.
APPLY is the identical command with only --dry-run removed. Index-first current
HF resolver closes the one referenced shard and same-folder known sidecars;
no --bundle, guessed wildcard, alternate downloader, SDK or extra model engine.
128MiB is existing ResourceArbiter transfer staging, not whole-RAM admission.
Current canonical XDG path is HOME fallback shown; if ROOT changes XDG base,
resolve the same abird/zerogravity/models project path under that absolute base.
License/README and all small Git blobs still need complete post-APPLY verification;
LFS digest is trusted provider SHA256, non-LFS cache digest alone is first-seen.

## Load/qualification consequences

- Shard alone exceeds4GiB by253,254,192B. Existing complete-source defaults
  refuse. Correct existing API is SourceWeights::prepare_verified_shards(...,
  SourceOpenLimits, admission); it has an explicit max_owned_bytes field.
  ROOT may choose a bounded >=4,548,221,488B owned cap plus separately charged
  metadata under a real host ResourceArbiter. No API named *_with_limits exists.
- Default4GiB GGUF/resident gates, full-HF execution/session/checkpoint guards
  stay CLOSED. Raw-window transport can authenticate original rows without
  whole-owned source; it does not create a fake complete TensorSource/bind.
- Original full632-tensor header/dtype/offset/index/EOF proof, same-token raw
  BF16/F32 parameter audit and loader/profile distinction, exact logits/all42
  state acceptance for2B, CPU/GPU operator scheduling and serving integration
  are future ROOT gates. Metadata acquisition proves none of these.
- Matched actual vLLM0.31.0/compiledFLA comparison requires this same original
  source and agreed dtype/cache/tokenizer/sampler/stop/hardware/resource profile.
  Approved mixed GGUF2B is a different quantized source; do not substitute it.
  No actual2B original-HF load, vLLM memory fit or throughput is claimed here.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/ADDRESS_SPACE.md

Original UTF-8 bytes: 5886; SHA256 `07ccb311a8b0a8a915fc1603c82f6db31be293ad350c3550a2465b4fe7537f27`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Strict worker address-space bound

Author: batch_design_review. Source-only proposal; all controls unexecuted here.

**Use exact RLIMIT_AS readback in addition to cgroup accounting.** Linux limits
brk/mmap/mremap and stack growth with RLIMIT_AS. Soft/hard limits are per process;
threads share them. RLIMIT_RSS is ineffective on current Linux. RLIMIT_NPROC
counts the real UID's threads, not this worker alone; it is not a substitute
for pids.max. [Linux resource-limit contract](https://man7.org/linux/man-pages/man2/getrlimit.2.html).

Inference: user-resident pages belonging to one address space cannot exceed
that address space's mapped-byte limit. This avoids using memory.max alone as
an exact user-page ceiling. It does not bound page tables, VMA/task metadata,
external file cache or parent/runtime-helper processes. These need separate
admission/constraints. No assertion that AS bounds all physical host RAM.

## Concrete first qualification budget using ROOT's28GiB planning domain

| Component | Candidate budget | Status |
| --- | ---: | --- |
| Worker RLIMIT_AS soft=hard |2GiB|actual getrlimit before dlopen required|
| Full immutable allowed closure/file bytes |512MiB|actual3 total455,641,931B fits; new complete closure must be counted|
| Parent IPC/control storage |128MiB|bound buffers/frames/receipts before allocation|
| Remaining guard for kernel/control costs |1,408MiB|available budget, NOT an established universal bound|
| Parent ProviderEnvelope reservation |4GiB|prepay on original host domain before ONE spawn|

This is a feasible budget candidate, leaving24GiB of a28GiB registered domain
for separately admitted source windows/state/other work. It is not measured
acceptance, a native allocator peak guarantee, or permission to use any unknown
guard cost. Physical admission remains closed until supported kernel/runtime
metadata and control costs fit the reserved guard with source/actual proof.
Do not shrink this budget to actual3's observed VmPeak or call that a bound.
ROOT must independently bind the actual host domain/limit before admission.

## Smallest launch variant

- Prefer the same native Rust worker executable with the already acquired
  exact GNU-loader/libc/provider closure staged readonly under one pinned root.
  Use original reviewed owned-child supervision, not per-operation Podman.
  Closure relocation changes file paths, not a licence to change bytes or let
  host Nix libc/libm/OpenMP enter. Independently reattest every mapped file and
  public entry. This packaging is not executed/qualified by actual3's container.
- In the owned pre-exec path, set AS soft/hard before exec using only supported
  async-signal-safe native operations. Parent admission is already held. Worker
  checks exact getrlimit immediately and before dlopen, then reports raw limits,
  PID/start/SID/PGID, VmSize and CPU affinity. Never let a label substitute for
  readback. A tiny bootstrap may set them before explicit loader exec if easier;
  it must not load the provider first or weaken ownership across exec.
- No CAP_SYS_RESOURCE in the initial namespace; no raising the hard limit.
  Bind no_new_privs and the exact worker confinement controls. Set bounded core
  dumps/files/descriptors/stack and deadline. Changing these does not change
  numerical VML mode, four-thread policy, provider bytes or reference contract.
- Disallow fork/vfork and clones with a separate address space. If threads are
  permitted, require CLONE_VM|CLONE_THREAD|CLONE_SIGHAND and reject namespace
  creation; shared-VM threads cannot multiply the AS budget. Deny subsequent
  execve/execveat. clone3 has an indirect argument struct: classic seccomp must
  not pretend it can inspect that pointer. A deliberate ENOSYS fallback to
  checked legacy clone is a candidate, requiring actual pinned pthread testing;
  otherwise reject this confinement choice rather than allow unchecked clones.
  [Clone semantics](https://man7.org/linux/man-pages/man2/clone.2.html),
  [seccomp argument model](https://man7.org/linux/man-pages/man2/seccomp.2.html).
- First EXP16-only worker can deny all thread creation if the original public
  call actually succeeds unchanged under its4-thread environment. This narrows
  a diagnostic, not a licence to run larger/parallel spans on another schedule.
  Matrix extension needs real bounded original four-thread creation/lifetime.
- Keep actual pids.max/cpu.max/memory.max/swap.max attestation when using a
  delegated worker cgroup/container. For direct-child launch without a usable
  delegated cgroup, thread/kernel/control bounds remain explicit gates; no
  guessed UID-wide limit or process-group label closes them.

## Minimum ROOT probe

1. Prepaid original host envelope; one child, no model. Exact AS soft/hard set
   before exec and read back before dlopen. Verify changing the child's soft
   limit above hard fails. Fresh child only; no host/global limit mutation.
2. Actual anonymous mmap/brk growth beyond AS refuses/terminates that child;
   retained parent owner remains until known retirement. This tests the control,
   not exp math. Forbid fork/newVM and observe original pthread route separately.
3. Execute actual3's four genuine n16 VML calls with all inputs/results/readsets
   exact, same closure and CPU/FP policy. Record AS, real kernel/cgroup controls,
   memory.stat/peak/events, mapping metadata and all owned process lifetimes.
4. Distinguish pressure/abort/cancel/malformed response from mathematical success.
   Prove parent windows and4GiB runtime charge survive a dropped future, source
   drop and unknown retirement; final0 only on complete known owned teardown.
5. Establish actual kernel/control guard coverage before production promotion.
   Then extend persistent RPC and accepted BLAS/STRSM families. No source/27B
   or full-HF/profile promotion from n16 EXP or containment fixtures alone.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/DESIGN.md

Original UTF-8 bytes: 11449; SHA256 `e9c39fe0050f2c7ed8c17df24e4663e1146078552865eb9f16ad146eedab4581`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Same-engine native CPU provider envelope

Author: batch_design_review. Source-only; no build, load, operator or process run.

**Recommend a bounded native Rust primitive worker.** Parent HostMath keeps the
one ForwardPlan, source/window owners, casts and state authority. Worker receives
only closed FP32 exp / grouped CBLAS / STRSM requests and has no source issuer,
model loader, token schedule, session, StateRoot or publication operation. No
Python/Torch computation or learned-weight cache. This is a proposed resource
boundary, not a qualified provider.

## Current exact boundary

- Native probe `root-actual-3`: 4 vmsExp,6 CBLAS batch-GEMM and32 STRSM calls,
  1,048,640 raw words,0 differences; real four-CPU policy and full mapped closure.
- It explicitly leaves `native_internal_memory_admission=UNQUALIFIED`.
  Cold mappings grew from4 files/3,396,784 file bytes to15/455,641,931 bytes;
  workers grew1→4. File totals and VmPeak are not live-allocation bounds.
- Reference oneMKL string is2024.0 Update2 build20240605, LP64 GNU-thread,
  CNRoff/Dyn0/FastMM1. Keep exact public signatures/mode0x140102, original
  grouping/layout/alignment policy and original provider/image closure.
- Acquired Torch DSO exposes documented allocator-pointer OBJECTs but lacks the
  exact i_malloc.h and documented memory-service symbol names. Those pointers
  do not cover the GNU loader, Torch constructors or OpenMP worker stacks.

## Why this boundary is smaller than allocator interception

Intel supports replacing i_malloc/i_calloc/i_realloc/i_free before the first
library call. It is a useful MKL suballocation seam, not whole-runtime ownership.
The pinned Torch closure runs about380 INIT_ARRAY entries on dlopen and loads
CUDA-runtime/CUPTI dependencies even for CPU math. Redirecting only four pointers
after dlopen cannot account for that cold work. [Public allocator contract](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/redefining-memory-functions.html).

`mkl_set_memory_limit` controls MCDRAM rather than arbitrary host heap. Disabling
FastMM routes calls to system allocation and changes the accepted policy; freeing
unused buffers is not a constructor/stack bound. Neither solves pre-admission.
[Memory limit](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/mkl-set-memory-limit.html),
[FastMM control](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/mkl-disable-fast-mm.html).

A parent-owned worker envelope can cover opaque allocation without changing
vendor math. Actual cgroup enforcement and lifecycle must be established first.
An argv/RSS receipt alone remains containment, not admission. See
`ADDRESS_SPACE.md` for the stronger AS-bounded worker candidate and concrete4GiB
prepaid budget; kernel/control closure remains separately required.

## Admission and startup

1. Borrow the existing `TransferAdmission`/process-shared ResourceArbiter and
   original registered physical host key. No new per-provider host pool or
   independent arbiter in the child. Admit `AllocationClass::ProviderEnvelope`
   BEFORE spawn, plus bounded parent transport/control storage. Admission failure
   performs no child spawn, DSO load or operator call.
2. Keep a parent runtime owner containing the envelope, exact runtime closure,
   owned process/container/cgroup identity, protocol endpoints and all active
   job owners. It is shared across source models; runtime identity is distinct
   from each source's sealed issuer. Never mint a model permit from the worker.
3. Initially use the exact pinned image/loader from actual3, offline/no GPU, immutable
   closure/worker mounts and private bounded writable output. No learned model
   mount is necessary. Reserve actual selected limits, not the old diagnostic's
   arbitrary12GiB containment allowance. The smaller production candidate uses
   one direct native child plus the exact pinned loader closure, preserving its
   byte identity and independently proving new mapped paths; no Podman per call.
4. Before dlopen, worker reports raw `/proc/self/cgroup`, mount/cgroup identity,
   its PID/start identity, actual memory.max/swap.max/pids.max/cpu.max and
   affinity. Parent independently binds these to its nonce/CID/owned cgroup.
   Verify swap.max0, no migration/controller mutation route,4 CPUs before DSO
   load and4 native threads. Missing delegation/controller is a cold refusal;
   preserve the known OCI cpuset-controller hold by using sched_setaffinity.
5. Kernel/OCI limits must apply before exec/constructors, not after a READY
   label. Parent issues no request until exact image, executable/DSO mappings,
   public symbols, MXCSR control and startup limits have been authenticated.

### Physical envelope completeness gate

- Cover the worker's anonymous allocations, native pools, stacks/TLS, mappings
  and bounded kernel costs. Parent input/output/transport buffers remain separately
  admitted before copying; shared mappings are not silently counted twice or
  assumed included in the worker's charge.
- Podman CLI/conmon/OCI helpers may live outside the container leaf. The first
  implementation needs an owned outer delegated cap covering those helpers, or
  a separately enforced and admitted control envelope. Do not equate the leaf's
  cap with all provider-associated host memory.
- Linux documents temporary memory.max overshoot. A reservation equal to that
  integer is not an absolute-byte proof. Bind a supported kernel/runtime policy
  with conservative admitted headroom and finite address-space/pid/fd/transport
  bounds; qualify pressure/OOM cases. Do not invent a universal overshoot margin.
- Already shared file-backed/IPC pages can have a different charge owner. Bind
  the allowed closure/shared backing explicitly; conservatively reserve their
  full known bytes if outside the worker envelope. Cgroup peak telemetry tests
  the chosen policy; it does not replace the preceding reservation.
- No cap writes, hierarchy migration or page-cache recovery by this agent.
  These conditions remain ROOT physical gates. [Kernel memory controller](https://www.kernel.org/doc/html/v6.6/admin-guide/cgroup-v2.html#memory),
  [Podman resource options](https://docs.podman.io/en/v5.6.1/markdown/podman-run.1.html).

## Smallest RPC contract

- Start with EXP only: source-site FP32 n16, exact mode0x140102, one contiguous
  transport span and private output. Reuse actual3 public call, pointer/closure
  attestation, redzones, immutable input check and exact expected-data separation.
  The provider never receives expected values.
- Fixed versioned opcode header, bounded payload lengths and checked products;
  one outstanding monotonically increasing request ID per worker. No shell,
  paths, arbitrary pointers or generic function address in a request. Logs use
  a separate bounded channel, never the math reply stream.
- Parent maps public operation/site/profile/layout contracts to the request.
  SourceIdentity and sealed WindowWorkspace issuer checks stay in HostMath.
  Worker validates its geometry independently but has no source-authority API.
- Initially use bounded framed copies, not shared writable parent mappings.
  This makes child death sufficient to stop all writes to parent-owned state.
  Parent prepays serialization staging and destination before request encoding;
  worker duplicates fit its whole envelope. A later shared-memfd transport is
  separate qualification and must retain all aliased owners through unmap/ack.
- Response is accepted only after full ID/opcode/shape/byte count, complete
  frame/EOF policy, exact input digest/redzone status and output status checks.
  No partial output or detached Vec escapes its original admitted owner.
- Extend to the accepted six CBLAS families and STRSM only after EXP ownership
  qualification. Preserve actual stride transport, transpose flags, group16,
  alpha1/beta0 and solves; no simplified zero-state recurrence or BLAS substitute.
  Head48/27B layouts need independent source-derived qualification.

## In-flight and retirement ownership

- Caller checks cancellation before admission/enqueue. Once written, a job owns
  all source windows/private arrays/runtime pins until actual terminal handling.
  Dropping the caller future signals cancellation but cannot drop the job owner.
- Original CPU worker/coordinator serializes the one native lane; no lock held
  across unrelated await or new model scheduling engine. TaskCoordinator and
  SessionAuthority remain parent-only. Native FFI is synchronous in the child;
  no asynchronous provider completion is inferred from pipe readability alone.
- Successful complete reply retires only that job's child copies. Parent result
  retains its WindowWorkspace. Cold/runtime/pool envelope remains charged for
  the persistent child, including between requests and after source-model drop.
- Canceled in-flight jobs drain and discard a valid completion, or retire the
  owned worker using the reviewed TERM/grace/KILL-before-reap lifecycle. Outputs
  from cancellation, OOM, malformed reply or EOF are never published/retried as
  a smaller numerical operation. Known stopped execution and mathematical
  success are distinct outcomes.
- Shutdown prevents new requests, joins/drains current work, closes stdin and
  proves exact owned worker/group/container/cgroup retirement plus pipe EOF.
  Only then release its envelope. Never merely drop a Ready GlobalResourceLease
  while a persistent child still owns memory.
- Unknown termination retains runtime envelope AND active job/source owners and
  fences the original host domain with existing GlobalResourceLease semantics.
  A new worker is not automatic recovery. A nonblocking Drop needs coordinator
  retirement ownership or retained quarantine, since async cleanup cannot run in
  ordinary Rust Drop. Ledger fences alone do not keep actual buffers alive.

## Exact existing seams / proposed ownership

- `core/resource_arbiter.rs`: admit/ProviderEnvelope/GlobalResourceLease already
  supply parent reservation, submitted/Ready and unknown fencing. No new budget
  authority needed. Keep the runtime lease alive through persistent child death.
- `artifacts/cache/mod.rs::TransferAdmission`: existing host domain/owner carrier;
  `source_weights/tensor_window.rs`: existing sealed source staging and aggregate
  output/scratch leases. Do not enlarge its32MiB per-operator workspace into an
  implicit foreign-runtime charge.
- `model-qwen35/host_math.rs`: source/profile/precision and retained workspace
  checks; add an explicitly qualified optional primitive route only after proof.
  Existing untyped MatVecKernel and full-HF refusal guards stay unchanged now.
- Worker reuses the public-ABI Rust probe's three call bodies, not a foreign
  model engine. A new internal native-provider worker subcommand/module may host
  that code later; it is not a second Qwen loop or a generic execution escape.
- Coefficient-preparation/cached-GDN proposals remain unchanged. Once qualified,
  native EXP supplies actual A_log exp/g-derived decay under typed site/shape
  contracts. Parent retains pointwise math, initial64 versus cached schedule,
  casts, canonical state codecs and all full-model acceptance gates.

No production API or source was added by this audit. Next implementation should
be the bounded EXP worker/owner prototype below, not an inert provider facade.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/ROOT_PROBE_PLAN.md

Original UTF-8 bytes: 3198; SHA256 `5ff6cbb1b41dc0fcd77605524a83bc3b725c54f125b2ea7bdf57f51b7207d73e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only next implementation / physical proof

Author: batch_design_review. Design prepared only; no script/process invocation.

1. Implement one-shot Rust EXP worker using exact actual3 pinned public ABI and
   genuine captured input16 words; expected archive stays parent-only. Keep old
   diagnostic source/receipts immutable. No model weights/token API in worker.
2. Parent borrows original host TransferAdmission; admits full runtime/control/
   transport envelopes before spawning. Exercise real insufficient budget and
   pre-cancel: no process, allocations, DSO constructors or native calls.
3. Startup stops before dlopen until parent validates exact nonce/CID/PID-start/
   cgroup binding, raw effective memory.max/swap.max/pids.max/cpu.max, actual
   four-CPU affinity and pinned executable/runtime closure. Cap observes real
   kernel controls, not echoed CLI config. No privileged controller repair.
4. Close DESIGN physical-envelope completeness conditions: outside helpers,
   shared bytes and kernel temporary-limit semantics. Record the chosen actual
   envelope separately from memory.max/RSS/peaks. No fabricated constant bound.
5. Normal and injected cold failure: prove initialization allocations remain
   charged until actual all-owned retirement. Force real cgroup/AS pressure in a
   fresh worker; native OOM/abort is terminal no-output, not caught Rust unwind.
6. Exact one-shot exp outputs versus all4 original captures, input readsets,
   signedzero/subnormal/range/nonfinite policy. Actual3 only tested n16; add
   authentic initial chunk exp layouts/spans and independent adversaries before
   widening any operator contract. Preserve exact VML mode and raw FP32 bits.
7. Inject cancellation after request write, caller-future drop, truncated/extra/
   stale response, worker death, slow/stuck call, cleanup-command errors and
   escaped owned-group child. Known retirement releases after proof; unknown
   retains source/runtime/job owners and host fence. No source/root publication.
8. Prove parent output retains WindowWorkspace after source/caller drop; actual
   final zero only after result and worker/control owners retire. Keep pending
   processes/groups/container/cgroup/EOF receipts on every failure path.
9. Then persistent EXP worker: multiple requests, fixed monotonic IDs, no cross-
   model/source results, bounded queue and actual idle pool envelope. Include
   cancellation between requests, ordinary model drop and final shutdown.
10. Reuse original six CBLAS+32 STRSM public calls/layouts for matrix extension,
    not CPU64 learned projection replacement. Qualify all1,048,640 actual3 words
    plus source-derived head48/27B shapes as separate scope. Only then wire typed
    per-operator provider into SAME HostMath/forward schedule and complete source
    fullV/all-state/initial/cached parity. Genuine vLLM GPU/performance separate.

Implementation isolation remains one coordinator. Do not repurpose frozen
coefficient or cached-GDN branches, loop schedules, mathematical profiles, or
already-reviewed lifecycle receipts. Reuse reviewed owned v4 cleanup logic with
actual new worker/cgroup retirement checks; no unrelated process signals.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/TODO.md

Original UTF-8 bytes: 1128; SHA256 `893c1432e936dfe135c7606b70f290837fe3c06daf784f54acdc5bdef6724bf0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist — native CPU provider envelope audit

Parent: shared original-HF execution; owner batch_design_review.

- [x] Inspect current ResourceArbiter, TransferAdmission, source windows and
  HostMath ownership; accepted actual3 public ABI/runtime/memory boundaries.
- [x] Recheck public Intel allocator/memory controls and kernel/Podman cap
  semantics. No new generic exp schedule, private symbol or guessed allocator.
- [x] Coordinate cached-GDN external-decay contract with baseline_review.
- [x] Recommend parent-admitted typed native Rust worker; document startup,
  in-flight ownership, conservative cap scope and actual retirement gates.
- [x] Audit RLIMIT_AS as user-page ceiling, fork/new-VM exclusion and separate
  kernel/control budget; propose4GiB first envelope, not a guessed peak bound.
- [x] Prepare next EXP-only implementation/physical matrix; freeze small source
  and metadata pins. No production code/API or numerical profile promotion.
- [ ] ROOT implementation/review and actual cap/admission/lifecycle/numerics.
- [ ] Matrix/head48/27B, all-HF and matched vLLM acceptance remain separate.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/frozen/ADDRESS_SPACE.md

Original UTF-8 bytes: 5886; SHA256 `07ccb311a8b0a8a915fc1603c82f6db31be293ad350c3550a2465b4fe7537f27`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Strict worker address-space bound

Author: batch_design_review. Source-only proposal; all controls unexecuted here.

**Use exact RLIMIT_AS readback in addition to cgroup accounting.** Linux limits
brk/mmap/mremap and stack growth with RLIMIT_AS. Soft/hard limits are per process;
threads share them. RLIMIT_RSS is ineffective on current Linux. RLIMIT_NPROC
counts the real UID's threads, not this worker alone; it is not a substitute
for pids.max. [Linux resource-limit contract](https://man7.org/linux/man-pages/man2/getrlimit.2.html).

Inference: user-resident pages belonging to one address space cannot exceed
that address space's mapped-byte limit. This avoids using memory.max alone as
an exact user-page ceiling. It does not bound page tables, VMA/task metadata,
external file cache or parent/runtime-helper processes. These need separate
admission/constraints. No assertion that AS bounds all physical host RAM.

## Concrete first qualification budget using ROOT's28GiB planning domain

| Component | Candidate budget | Status |
| --- | ---: | --- |
| Worker RLIMIT_AS soft=hard |2GiB|actual getrlimit before dlopen required|
| Full immutable allowed closure/file bytes |512MiB|actual3 total455,641,931B fits; new complete closure must be counted|
| Parent IPC/control storage |128MiB|bound buffers/frames/receipts before allocation|
| Remaining guard for kernel/control costs |1,408MiB|available budget, NOT an established universal bound|
| Parent ProviderEnvelope reservation |4GiB|prepay on original host domain before ONE spawn|

This is a feasible budget candidate, leaving24GiB of a28GiB registered domain
for separately admitted source windows/state/other work. It is not measured
acceptance, a native allocator peak guarantee, or permission to use any unknown
guard cost. Physical admission remains closed until supported kernel/runtime
metadata and control costs fit the reserved guard with source/actual proof.
Do not shrink this budget to actual3's observed VmPeak or call that a bound.
ROOT must independently bind the actual host domain/limit before admission.

## Smallest launch variant

- Prefer the same native Rust worker executable with the already acquired
  exact GNU-loader/libc/provider closure staged readonly under one pinned root.
  Use original reviewed owned-child supervision, not per-operation Podman.
  Closure relocation changes file paths, not a licence to change bytes or let
  host Nix libc/libm/OpenMP enter. Independently reattest every mapped file and
  public entry. This packaging is not executed/qualified by actual3's container.
- In the owned pre-exec path, set AS soft/hard before exec using only supported
  async-signal-safe native operations. Parent admission is already held. Worker
  checks exact getrlimit immediately and before dlopen, then reports raw limits,
  PID/start/SID/PGID, VmSize and CPU affinity. Never let a label substitute for
  readback. A tiny bootstrap may set them before explicit loader exec if easier;
  it must not load the provider first or weaken ownership across exec.
- No CAP_SYS_RESOURCE in the initial namespace; no raising the hard limit.
  Bind no_new_privs and the exact worker confinement controls. Set bounded core
  dumps/files/descriptors/stack and deadline. Changing these does not change
  numerical VML mode, four-thread policy, provider bytes or reference contract.
- Disallow fork/vfork and clones with a separate address space. If threads are
  permitted, require CLONE_VM|CLONE_THREAD|CLONE_SIGHAND and reject namespace
  creation; shared-VM threads cannot multiply the AS budget. Deny subsequent
  execve/execveat. clone3 has an indirect argument struct: classic seccomp must
  not pretend it can inspect that pointer. A deliberate ENOSYS fallback to
  checked legacy clone is a candidate, requiring actual pinned pthread testing;
  otherwise reject this confinement choice rather than allow unchecked clones.
  [Clone semantics](https://man7.org/linux/man-pages/man2/clone.2.html),
  [seccomp argument model](https://man7.org/linux/man-pages/man2/seccomp.2.html).
- First EXP16-only worker can deny all thread creation if the original public
  call actually succeeds unchanged under its4-thread environment. This narrows
  a diagnostic, not a licence to run larger/parallel spans on another schedule.
  Matrix extension needs real bounded original four-thread creation/lifetime.
- Keep actual pids.max/cpu.max/memory.max/swap.max attestation when using a
  delegated worker cgroup/container. For direct-child launch without a usable
  delegated cgroup, thread/kernel/control bounds remain explicit gates; no
  guessed UID-wide limit or process-group label closes them.

## Minimum ROOT probe

1. Prepaid original host envelope; one child, no model. Exact AS soft/hard set
   before exec and read back before dlopen. Verify changing the child's soft
   limit above hard fails. Fresh child only; no host/global limit mutation.
2. Actual anonymous mmap/brk growth beyond AS refuses/terminates that child;
   retained parent owner remains until known retirement. This tests the control,
   not exp math. Forbid fork/newVM and observe original pthread route separately.
3. Execute actual3's four genuine n16 VML calls with all inputs/results/readsets
   exact, same closure and CPU/FP policy. Record AS, real kernel/cgroup controls,
   memory.stat/peak/events, mapping metadata and all owned process lifetimes.
4. Distinguish pressure/abort/cancel/malformed response from mathematical success.
   Prove parent windows and4GiB runtime charge survive a dropped future, source
   drop and unknown retirement; final0 only on complete known owned teardown.
5. Establish actual kernel/control guard coverage before production promotion.
   Then extend persistent RPC and accepted BLAS/STRSM families. No source/27B
   or full-HF/profile promotion from n16 EXP or containment fixtures alone.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/frozen/DESIGN.md

Original UTF-8 bytes: 11449; SHA256 `e9c39fe0050f2c7ed8c17df24e4663e1146078552865eb9f16ad146eedab4581`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Same-engine native CPU provider envelope

Author: batch_design_review. Source-only; no build, load, operator or process run.

**Recommend a bounded native Rust primitive worker.** Parent HostMath keeps the
one ForwardPlan, source/window owners, casts and state authority. Worker receives
only closed FP32 exp / grouped CBLAS / STRSM requests and has no source issuer,
model loader, token schedule, session, StateRoot or publication operation. No
Python/Torch computation or learned-weight cache. This is a proposed resource
boundary, not a qualified provider.

## Current exact boundary

- Native probe `root-actual-3`: 4 vmsExp,6 CBLAS batch-GEMM and32 STRSM calls,
  1,048,640 raw words,0 differences; real four-CPU policy and full mapped closure.
- It explicitly leaves `native_internal_memory_admission=UNQUALIFIED`.
  Cold mappings grew from4 files/3,396,784 file bytes to15/455,641,931 bytes;
  workers grew1→4. File totals and VmPeak are not live-allocation bounds.
- Reference oneMKL string is2024.0 Update2 build20240605, LP64 GNU-thread,
  CNRoff/Dyn0/FastMM1. Keep exact public signatures/mode0x140102, original
  grouping/layout/alignment policy and original provider/image closure.
- Acquired Torch DSO exposes documented allocator-pointer OBJECTs but lacks the
  exact i_malloc.h and documented memory-service symbol names. Those pointers
  do not cover the GNU loader, Torch constructors or OpenMP worker stacks.

## Why this boundary is smaller than allocator interception

Intel supports replacing i_malloc/i_calloc/i_realloc/i_free before the first
library call. It is a useful MKL suballocation seam, not whole-runtime ownership.
The pinned Torch closure runs about380 INIT_ARRAY entries on dlopen and loads
CUDA-runtime/CUPTI dependencies even for CPU math. Redirecting only four pointers
after dlopen cannot account for that cold work. [Public allocator contract](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/redefining-memory-functions.html).

`mkl_set_memory_limit` controls MCDRAM rather than arbitrary host heap. Disabling
FastMM routes calls to system allocation and changes the accepted policy; freeing
unused buffers is not a constructor/stack bound. Neither solves pre-admission.
[Memory limit](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/mkl-set-memory-limit.html),
[FastMM control](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/mkl-disable-fast-mm.html).

A parent-owned worker envelope can cover opaque allocation without changing
vendor math. Actual cgroup enforcement and lifecycle must be established first.
An argv/RSS receipt alone remains containment, not admission. See
`ADDRESS_SPACE.md` for the stronger AS-bounded worker candidate and concrete4GiB
prepaid budget; kernel/control closure remains separately required.

## Admission and startup

1. Borrow the existing `TransferAdmission`/process-shared ResourceArbiter and
   original registered physical host key. No new per-provider host pool or
   independent arbiter in the child. Admit `AllocationClass::ProviderEnvelope`
   BEFORE spawn, plus bounded parent transport/control storage. Admission failure
   performs no child spawn, DSO load or operator call.
2. Keep a parent runtime owner containing the envelope, exact runtime closure,
   owned process/container/cgroup identity, protocol endpoints and all active
   job owners. It is shared across source models; runtime identity is distinct
   from each source's sealed issuer. Never mint a model permit from the worker.
3. Initially use the exact pinned image/loader from actual3, offline/no GPU, immutable
   closure/worker mounts and private bounded writable output. No learned model
   mount is necessary. Reserve actual selected limits, not the old diagnostic's
   arbitrary12GiB containment allowance. The smaller production candidate uses
   one direct native child plus the exact pinned loader closure, preserving its
   byte identity and independently proving new mapped paths; no Podman per call.
4. Before dlopen, worker reports raw `/proc/self/cgroup`, mount/cgroup identity,
   its PID/start identity, actual memory.max/swap.max/pids.max/cpu.max and
   affinity. Parent independently binds these to its nonce/CID/owned cgroup.
   Verify swap.max0, no migration/controller mutation route,4 CPUs before DSO
   load and4 native threads. Missing delegation/controller is a cold refusal;
   preserve the known OCI cpuset-controller hold by using sched_setaffinity.
5. Kernel/OCI limits must apply before exec/constructors, not after a READY
   label. Parent issues no request until exact image, executable/DSO mappings,
   public symbols, MXCSR control and startup limits have been authenticated.

### Physical envelope completeness gate

- Cover the worker's anonymous allocations, native pools, stacks/TLS, mappings
  and bounded kernel costs. Parent input/output/transport buffers remain separately
  admitted before copying; shared mappings are not silently counted twice or
  assumed included in the worker's charge.
- Podman CLI/conmon/OCI helpers may live outside the container leaf. The first
  implementation needs an owned outer delegated cap covering those helpers, or
  a separately enforced and admitted control envelope. Do not equate the leaf's
  cap with all provider-associated host memory.
- Linux documents temporary memory.max overshoot. A reservation equal to that
  integer is not an absolute-byte proof. Bind a supported kernel/runtime policy
  with conservative admitted headroom and finite address-space/pid/fd/transport
  bounds; qualify pressure/OOM cases. Do not invent a universal overshoot margin.
- Already shared file-backed/IPC pages can have a different charge owner. Bind
  the allowed closure/shared backing explicitly; conservatively reserve their
  full known bytes if outside the worker envelope. Cgroup peak telemetry tests
  the chosen policy; it does not replace the preceding reservation.
- No cap writes, hierarchy migration or page-cache recovery by this agent.
  These conditions remain ROOT physical gates. [Kernel memory controller](https://www.kernel.org/doc/html/v6.6/admin-guide/cgroup-v2.html#memory),
  [Podman resource options](https://docs.podman.io/en/v5.6.1/markdown/podman-run.1.html).

## Smallest RPC contract

- Start with EXP only: source-site FP32 n16, exact mode0x140102, one contiguous
  transport span and private output. Reuse actual3 public call, pointer/closure
  attestation, redzones, immutable input check and exact expected-data separation.
  The provider never receives expected values.
- Fixed versioned opcode header, bounded payload lengths and checked products;
  one outstanding monotonically increasing request ID per worker. No shell,
  paths, arbitrary pointers or generic function address in a request. Logs use
  a separate bounded channel, never the math reply stream.
- Parent maps public operation/site/profile/layout contracts to the request.
  SourceIdentity and sealed WindowWorkspace issuer checks stay in HostMath.
  Worker validates its geometry independently but has no source-authority API.
- Initially use bounded framed copies, not shared writable parent mappings.
  This makes child death sufficient to stop all writes to parent-owned state.
  Parent prepays serialization staging and destination before request encoding;
  worker duplicates fit its whole envelope. A later shared-memfd transport is
  separate qualification and must retain all aliased owners through unmap/ack.
- Response is accepted only after full ID/opcode/shape/byte count, complete
  frame/EOF policy, exact input digest/redzone status and output status checks.
  No partial output or detached Vec escapes its original admitted owner.
- Extend to the accepted six CBLAS families and STRSM only after EXP ownership
  qualification. Preserve actual stride transport, transpose flags, group16,
  alpha1/beta0 and solves; no simplified zero-state recurrence or BLAS substitute.
  Head48/27B layouts need independent source-derived qualification.

## In-flight and retirement ownership

- Caller checks cancellation before admission/enqueue. Once written, a job owns
  all source windows/private arrays/runtime pins until actual terminal handling.
  Dropping the caller future signals cancellation but cannot drop the job owner.
- Original CPU worker/coordinator serializes the one native lane; no lock held
  across unrelated await or new model scheduling engine. TaskCoordinator and
  SessionAuthority remain parent-only. Native FFI is synchronous in the child;
  no asynchronous provider completion is inferred from pipe readability alone.
- Successful complete reply retires only that job's child copies. Parent result
  retains its WindowWorkspace. Cold/runtime/pool envelope remains charged for
  the persistent child, including between requests and after source-model drop.
- Canceled in-flight jobs drain and discard a valid completion, or retire the
  owned worker using the reviewed TERM/grace/KILL-before-reap lifecycle. Outputs
  from cancellation, OOM, malformed reply or EOF are never published/retried as
  a smaller numerical operation. Known stopped execution and mathematical
  success are distinct outcomes.
- Shutdown prevents new requests, joins/drains current work, closes stdin and
  proves exact owned worker/group/container/cgroup retirement plus pipe EOF.
  Only then release its envelope. Never merely drop a Ready GlobalResourceLease
  while a persistent child still owns memory.
- Unknown termination retains runtime envelope AND active job/source owners and
  fences the original host domain with existing GlobalResourceLease semantics.
  A new worker is not automatic recovery. A nonblocking Drop needs coordinator
  retirement ownership or retained quarantine, since async cleanup cannot run in
  ordinary Rust Drop. Ledger fences alone do not keep actual buffers alive.

## Exact existing seams / proposed ownership

- `core/resource_arbiter.rs`: admit/ProviderEnvelope/GlobalResourceLease already
  supply parent reservation, submitted/Ready and unknown fencing. No new budget
  authority needed. Keep the runtime lease alive through persistent child death.
- `artifacts/cache/mod.rs::TransferAdmission`: existing host domain/owner carrier;
  `source_weights/tensor_window.rs`: existing sealed source staging and aggregate
  output/scratch leases. Do not enlarge its32MiB per-operator workspace into an
  implicit foreign-runtime charge.
- `model-qwen35/host_math.rs`: source/profile/precision and retained workspace
  checks; add an explicitly qualified optional primitive route only after proof.
  Existing untyped MatVecKernel and full-HF refusal guards stay unchanged now.
- Worker reuses the public-ABI Rust probe's three call bodies, not a foreign
  model engine. A new internal native-provider worker subcommand/module may host
  that code later; it is not a second Qwen loop or a generic execution escape.
- Coefficient-preparation/cached-GDN proposals remain unchanged. Once qualified,
  native EXP supplies actual A_log exp/g-derived decay under typed site/shape
  contracts. Parent retains pointwise math, initial64 versus cached schedule,
  casts, canonical state codecs and all full-model acceptance gates.

No production API or source was added by this audit. Next implementation should
be the bounded EXP worker/owner prototype below, not an inert provider facade.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/frozen/ROOT_PROBE_PLAN.md

Original UTF-8 bytes: 3198; SHA256 `5ff6cbb1b41dc0fcd77605524a83bc3b725c54f125b2ea7bdf57f51b7207d73e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only next implementation / physical proof

Author: batch_design_review. Design prepared only; no script/process invocation.

1. Implement one-shot Rust EXP worker using exact actual3 pinned public ABI and
   genuine captured input16 words; expected archive stays parent-only. Keep old
   diagnostic source/receipts immutable. No model weights/token API in worker.
2. Parent borrows original host TransferAdmission; admits full runtime/control/
   transport envelopes before spawning. Exercise real insufficient budget and
   pre-cancel: no process, allocations, DSO constructors or native calls.
3. Startup stops before dlopen until parent validates exact nonce/CID/PID-start/
   cgroup binding, raw effective memory.max/swap.max/pids.max/cpu.max, actual
   four-CPU affinity and pinned executable/runtime closure. Cap observes real
   kernel controls, not echoed CLI config. No privileged controller repair.
4. Close DESIGN physical-envelope completeness conditions: outside helpers,
   shared bytes and kernel temporary-limit semantics. Record the chosen actual
   envelope separately from memory.max/RSS/peaks. No fabricated constant bound.
5. Normal and injected cold failure: prove initialization allocations remain
   charged until actual all-owned retirement. Force real cgroup/AS pressure in a
   fresh worker; native OOM/abort is terminal no-output, not caught Rust unwind.
6. Exact one-shot exp outputs versus all4 original captures, input readsets,
   signedzero/subnormal/range/nonfinite policy. Actual3 only tested n16; add
   authentic initial chunk exp layouts/spans and independent adversaries before
   widening any operator contract. Preserve exact VML mode and raw FP32 bits.
7. Inject cancellation after request write, caller-future drop, truncated/extra/
   stale response, worker death, slow/stuck call, cleanup-command errors and
   escaped owned-group child. Known retirement releases after proof; unknown
   retains source/runtime/job owners and host fence. No source/root publication.
8. Prove parent output retains WindowWorkspace after source/caller drop; actual
   final zero only after result and worker/control owners retire. Keep pending
   processes/groups/container/cgroup/EOF receipts on every failure path.
9. Then persistent EXP worker: multiple requests, fixed monotonic IDs, no cross-
   model/source results, bounded queue and actual idle pool envelope. Include
   cancellation between requests, ordinary model drop and final shutdown.
10. Reuse original six CBLAS+32 STRSM public calls/layouts for matrix extension,
    not CPU64 learned projection replacement. Qualify all1,048,640 actual3 words
    plus source-derived head48/27B shapes as separate scope. Only then wire typed
    per-operator provider into SAME HostMath/forward schedule and complete source
    fullV/all-state/initial/cached parity. Genuine vLLM GPU/performance separate.

Implementation isolation remains one coordinator. Do not repurpose frozen
coefficient or cached-GDN branches, loop schedules, mathematical profiles, or
already-reviewed lifecycle receipts. Reuse reviewed owned v4 cleanup logic with
actual new worker/cgroup retirement checks; no unrelated process signals.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/frozen/TODO.md

Original UTF-8 bytes: 1128; SHA256 `893c1432e936dfe135c7606b70f290837fe3c06daf784f54acdc5bdef6724bf0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist — native CPU provider envelope audit

Parent: shared original-HF execution; owner batch_design_review.

- [x] Inspect current ResourceArbiter, TransferAdmission, source windows and
  HostMath ownership; accepted actual3 public ABI/runtime/memory boundaries.
- [x] Recheck public Intel allocator/memory controls and kernel/Podman cap
  semantics. No new generic exp schedule, private symbol or guessed allocator.
- [x] Coordinate cached-GDN external-decay contract with baseline_review.
- [x] Recommend parent-admitted typed native Rust worker; document startup,
  in-flight ownership, conservative cap scope and actual retirement gates.
- [x] Audit RLIMIT_AS as user-page ceiling, fork/new-VM exclusion and separate
  kernel/control budget; propose4GiB first envelope, not a guessed peak bound.
- [x] Prepare next EXP-only implementation/physical matrix; freeze small source
  and metadata pins. No production code/API or numerical profile promotion.
- [ ] ROOT implementation/review and actual cap/admission/lifecycle/numerics.
- [ ] Matrix/head48/27B, all-HF and matched vLLM acceptance remain separate.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/NEGATIVE_PLAN.md

Original UTF-8 bytes: 3359; SHA256 `c936c5c3e8cefe28808d0003e0a1ee1e1b4346e0678f520c34d09b79589736b4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Required physical negatives

Author: batch_design_review. Prepared, not executed by author.

## Included six-case first gate

| Case | Real action | Required result |
| --- | --- | --- |
| normal |admit, spawn same binary, readbacks, explicit EXIT|actual reaped0/group absent/EOF before release/final0|
| pre-cancel |cancel before request|no admission/no child/final0|
| admission-refusal |original host limit4GiB-1 vs real4GiB request|exact original typed capacity refusal/no child/final0|
| cancel-after-hello |drop caller Arc while real child blocks|same runtime ticket4GiB retained, child still alive, supervisor retires -15/-9 before release|
| as-pressure |mmap(PROT_NONE,AS+4096)|real ENOMEM; no RAM faulting, unchanged actual AS, normal retirement|
| bad-as |pre-exec AS=2GiB-4096|actual readback refuses before closure/provider work, child125, known retirement/final0|

Every successful child also must receive real EPERM trying to raise the AS hard
limit and invoke raw fork. An unexpectedly permitted fork child immediately
_exit126 and is reaped by its original parent before the probe refuses.

## Required before extending provider eligibility

These are not implemented or falsely passed by the six-case startup probe.
ROOT should assign narrow additive controls only after reviewing first results:

- Actual pthread/clone fallback route in pinned provider, fixed thread ceiling,
  no independent VM/exec/namespace path; explicit clone3 policy and x32 refusal.
  Current filter denies all clones and cannot qualify4-thread matrix calls.
- Inject malformed/oversized/partial frame, pipe EOF failure, startup failure,
  kill failure/timeout and parent-death paths; independently observe actual
  retained owner and fence until complete owned process/container retirement.
  A process-local leak is not a durable production quarantine implementation.
- Inject receipt I/O failures before/after spawn, before/after reap and during
  final report persistence. Preserve primary+cleanup failures; never release on
  uncertain process/group/pipe completion. Existing outer helper failure
  fixtures were independently accepted previously; this new Rust owner has
  not inherited actual execution qualification from those Python fixtures.
- Real absolute memory/control/thread/VMA/FD/cgroup accounting closure covering
  parent, worker, kernel metadata, file cache and any helper. Test denial under
  real arbiter pressure without spawning an unadmitted provider or relaxing cap.
- Only after closure: load exact prospective provider and reattest actual mapped
  full14DSO+binary closure, constructor/thread/task effects, FP policy and native
  ABI symbols. Any new mapping/allocation path refuses until independently bound.
- Replay actual3 public EXP16 first, then genuine accepted BMM/TRSM operands on
  original pinned binary/source schedule. Parent independently compares outputs;
  expected values are never sent to worker as an implementation substitute.
- Persistent optional MathematicalBackend provider adapter may issue only typed
  primitive requests; parent retains input/output source windows and physical
  leases. Worker must never become a model/token/session engine or state issuer.

Synthetic `test_contracts.py` fixtures test strict receipt rejection only.
They do not simulate or establish kernel/process/provider implementation.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/POLICY.md

Original UTF-8 bytes: 4419; SHA256 `d40a45cd0af8206459aa4254a3a70a6c8eebde8d5b517a205617fd08c7f7018d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Startup policy and ownership

Author: batch_design_review. Source-only; ROOT must execute every control.

## Before child spawn

- ROOT launcher: exact already acquired image ID/manifest, pull never, network
  none, read-only image, no GPU/device grant, all caps dropped, NNP,12GiB memory
  with total memory+swap12GiB (therefore swap0), pids64 and CPU quota4. No OCI
  cpuset requirement; explicit sched_setaffinity is set/read back in Rust.
- Only binary RO and fresh output RW mounts. No learned source/model or source
  issuer. Parent's configured original host key admits4GiB ProviderEnvelope
  before `Command::spawn`. Pre-cancel and capacity refusal spawn no child.
- Same exact binary is launched through original public GNU loader with original
  static14DSO search paths. Source pins and actual ELF/mapped-byte identity must
  remain exact. No DSO or provider acquisition by this code.

## Child controls

| Control | Exact policy | Proof scope |
| --- | --- | --- |
| Address space | AS soft=hard2GiB |set in pre-exec, actual readback before any prospective DSO hash/load; hard raise must EPERM|
| Core |0|actual soft/hard readback|
| FDs |32 hard; startup observed<=16|actual readback/count; no future provider claim|
| Stack |8MiB hard|actual readback|
| CPU time |60 seconds hard|actual readback; outer wall limit remains separate|
| Output files |4MiB hard|actual readback; parent output report<=128KiB|
| UID/caps |same parent effective UID, all four UID fields; all Eff/Prm/Bnd/Amb0|actual raw status + public UID calls; rootless namespace UID0 alone is not elevated host authority|
| NNP |1|pre-exec setting + actual prctl/raw readback|
| CPU affinity |exact four available CPU IDs|actual sched_getaffinity + raw status|
| Processes/threads |single child address space; all fork/vfork/clone/clone3 denied|actual tasks1 and real fork EPERM; pinned x86_64 seccomp UAPI rejects other arch/x32|
| Executables/namespaces |execve/execveat/unshare/setns/setpgid/setsid denied|source-derived seccomp; actual extra refusal fixtures still required below|
| Limit mutation |setrlimit denied; prlimit64 permits only NULL new_limit|source UAPI pointer-word checks, required for glibc getrlimit; no indirect pointer inspection claim|
| Other external allocations |SysV shmget/shmat, memfd, io_uring, userfaultfd, bpf, perf and ptrace denied|source policy only until actual negatives qualified|
| VMAs |observed<=4096; kernel max_map_count1..1,048,576|finite supported startup observation, not a per-worker physical metadata bound|

The filter intentionally permits ordinary file/pipe read/write, hashing,
allocation under AS and process exit. It is not a general hostile-code sandbox.
It does not currently permit actual native library thread initialization.
No provider is loaded before this filter or anywhere in this binary.

## Finite IPC and retained lifetime

- Child stdout: one JSON frame<128KiB plus newline; parent polls50ms with60s
  startup deadline, rejects EOF/oversized frames. Child accepts only5 bytes
  `EXIT\n` then EOF. There is no operator payload or shared memory transport.
- Parent report<=128KiB; immutable metadata reads<=128KiB each; prospective file
  hashing uses one1MiB buffer at a time and max512MiB full allowed closure bytes.
  DSO/kernel/cache physical overhead remains unqualified despite these bounds.
- Native sole `Child` is not reaped until after all TERM/grace/KILL attempts.
  Each nonzero group signal rebinds original unreaped PID/SID/PGID. After reap,
  only signal0 absence checks occur. Known retirement additionally requires EOF.
- Supervisor keeps same original Arc ticket while caller view is dropped. It
  marks original lease Ready/releases only after trusted complete retirement.
  Unknown retirement fences and retains actual child/pipes/charge in-process;
  no production recovery or parent-death durability is claimed.
- Outer exact reviewed v4 helper independently handles native parent/container
  lifetime, two bounded KILL attempts before reap, final nonce/CID absence and
  log EOF/close. Each case120s wall limit plus finite cleanup; six serial cases.
  All post-Popen receipt/log setup is under cleanup `finally`.

Primary control contracts are the original audit's Linux getrlimit/seccomp and
cgroup-v2 references. No public documentation guarantees that the remaining
1,408MiB guard covers all physical kernel/control costs; that gate stays false.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/README.md

Original UTF-8 bytes: 6238; SHA256 `b809ce160003a7a150b311e74134307295ecb338f41273ff0f977153cf7c84fb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native provider startup gate

Author: batch_design_review. ROOT-only proposal; uncompiled and unexecuted.

Parent plan: existing model-execution/shared-engine continuation. Derivative of
the unchanged envelope audit freeze
`ee7761fb1ac99d96d9d5af3215432716f23bcf2591b0dfa63ee04110594ba08b`.
No MAIN source, provider, source/model permission, token loop or eligibility
change. This is a startup-control probe, not an admitted native math provider.

## Scope

- One native Rust executable: original ResourceArbiter parent + same-binary
  child. Parent admits one real4GiB ProviderEnvelope before spawn. Child has
  exact2GiB AS soft/hard before exec, then independently reads effective UID,
  capabilities, NNP, CPU affinity, limits and actual cgroup controls.
- Native parent/child run in the already acquired actual3 base image and exact
  GNU loader. Original14DSO candidate manifest is copied byte-exact. Child
  authenticates all14 prospective files and the new ROOT binary, records the
  actual mapped subset, and refuses if the provider is already loaded.
- No dlopen, VML, CBLAS, STRSM, model/tensor source, source issuer, StateRoot,
  session, GPU device or native math call exists in this probe.
- The single-thread child additionally denies all fork/clone/exec and namespace
  changes. Actual fork and oversized nonresident mmap refusals are tested in
  their own cases. Four-thread native math remains a later gate.
- Same original GlobalResourceLease held by the supervisor survives dropping a
  caller Arc while the actual child waits. The lease is released only after
  owned child reap, group absence and stdout EOF. This tests a startup caller
  reference, **not a dropped asynchronous MathematicalBackend service future**.

## Explicit remaining admission gates

- Diagnostic host key/8GiB limit is configured inside the native process, using
  the same original process-shared ResourceArbiter API as existing isolated
  source-window proofs. It is not an attested production host domain/limit or
  a cross-process resource ledger.
- The outer container and Podman/conmon processes exist before native admission.
  Their host control costs are not covered/proven by this first ticket. No
  assertion that memory.max cannot overshoot or that AS bounds kernel metadata.
-4GiB is a candidate envelope:2GiB AS +512MiB full file cache allowance +128MiB
  IPC/control allowance +1,408MiB unproved guard. A guard allowance is not a
  physical bound. Status fields deliberately keep kernel/control and production
  physical provider admission false, even after all six startup cases pass.
- Single-thread startup policy, observed4096VMAs/16FDs and a bounded public
  vm.max_map_count readback do not qualify future provider/OpenMP/FastMM costs.
  Provider constructors, new thread control, exact mapped closure after load,
  actual exp/BLAS arithmetic and their whole physical lifetime remain closed.
- On unknown retirement, original ticket and actual child/pipe owner are leaked
  behind the original arbiter fence for the native parent's remaining lifetime.
  No retry or release occurs. Parent death/outer cleanup is not a persistent
  cross-process quarantine proof; the attempt fails, never passes this gate.

## Files

- `common.rs`: bounded observation/hash transport; original closure metadata.
- `worker.rs`: child readbacks, startup seccomp and real negative syscalls.
- `main.rs`: original admission, sole child ownership, finite retirement.
- `run_probe.py`: ROOT-only six-case launcher; exact copied owned-v4 cleanup.
- `validate.py`: independent raw/control/ledger/lifetime receipt checks.
- `test_contracts.py`: synthetic schema regressions, never runtime evidence.
- `POLICY.md`, `NEGATIVE_PLAN.md`: exact scope and required next physical gates.
- `freeze-manifest.json`, `frozen/`: source-only additive freeze.

## ROOT execution order

Review all files and source pins first. No author build, fixture run, container,
process/control probe, DSO hash/load or math execution has occurred.

1. Run prepared schema fixtures on ROOT's serialized CPU schedule:
   `python3 test_contracts.py` from this directory.
2. Generate the standalone effective lockfile, run strict Clippy/build under
   ROOT's original isolation/2-job coordination. Cargo resolves no new intended
   versions: libc0.2.190, serde_json1.0.151, sha2 0.10.9, original core path.
   Record actual source freeze, effective lock, successful build and binary hash.
   Build may refuse if host-linked symbol requirements exceed pinned glibc2.39;
   do not silently substitute another libc/provider/image.
3. Create a ROOT build JSON (not supplied/fabricated by the author):

   ```json
   {
     "status": "ROOT_COMPILED_STARTUP_ONLY",
     "freeze_sha256": "ACTUAL_STARTUP_FREEZE_SHA",
     "binary": "/absolute/path/to/native-cpu-startup-probe",
     "binary_sha256": "ACTUAL_COMPILED_BINARY_SHA",
     "effective_lock_sha256": "ACTUAL_EFFECTIVE_CARGO_LOCK_SHA",
     "build_exit_code": 0
   }
   ```

   Retain the actual command/log and effective lock with that receipt; a manually
   written success label is not compilation evidence. The launcher requires the
   new binary/receipt binding and checks its actual ELF requirements.
4. Run once with fresh output (replace only uppercase placeholders):

   ```bash
   python3 run_probe.py --binary /absolute/path/to/native-cpu-startup-probe \
     --binary-sha256 ACTUAL_COMPILED_BINARY_SHA \
     --build-attestation /absolute/path/to/ROOT-build.json \
     --new-output /home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/root-attempt-1
   ```

5. Independently validate the output with `python3 validate.py --output` and the
   same absolute new output path. Freeze failed attempts; use a fresh derivative
   and fresh output for source corrections. No automatic fallback/provider
   switch or model inference is allowed by this launcher.

Expected accepted scope: `PASS_STARTUP_OWNERSHIP_ONLY_PROVIDER_UNLOADED`.
After acceptance, first extend an independently admitted actual3 EXP16 call
with unchanged provider/mode/arithmetic and explicit lifecycle/resource proof.
No full-HF/FLA/27B/vLLM claim follows from this startup probe.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/SOURCE_AUDIT.md

Original UTF-8 bytes: 4223; SHA256 `d47ea76b9d53d056e763a07e70d8619be00b83cbc681db74700324888ea93eb0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Source boundary

Author: batch_design_review. Source reading only; no actual controls executed.

## Reused original contracts

- `resource_arbiter.rs`: process_shared, register_matching, exact typed Limit
  capacity refusal, ProviderEnvelope admission, mark_submitted/mark_ready/release
  and fence_unknown. Original Submitted-drop retains/fences uncertain work.
  No original source byte is changed; the new binary has no independent arbiter
  implementation. Configured diagnostic key is explicitly not production proof.
- Accepted actual3 native probe: copied exact static14DSO manifest, exact base
  image/ID and GNU loader/library path/cache inhibition. Actual3 arithmetic and
  memory receipts are read as prior evidence, not rerun/extended by author.
  That original binary had15 mapped files including its own executable; this
  startup probe authenticates a new ROOT-compiled binary as the fifteenth item.
- Reviewed `owned_container_v4.py` is byte-exact, copied from immutable GEMM v2.
  Native owner mirrors its reserved-leader WNOWAIT/bind-before-signal approach;
  copying the lifecycle design does not inherit physical test acceptance.

## Public Linux/libc boundary

- Pinned libc0.2.190 public x86_64 GNU bindings provide rlimit types, waitid,
  siginfo accessors, CPU_SET/ISSET, prctl and classic BPF sock_filter/fprog.
  Exact selected source bytes are pinned with this proposal.
- Classic Linux seccomp_data layout is nr@0, arch@4, instruction_pointer@8,
  six64-bit arguments@16. Therefore prlimit64 new_limit=args[2] is read at32/36;
  no user-pointer dereference or clone3 indirect-argument interpretation occurs.
  Exact x86_64 audit arch0xc000003e is required; syscall numbers>=0x40000000
  (including x32) are rejected before the native syscall-number table.
- The filter permits prlimit64 only with a NULL new_limit so glibc getrlimit
  can read actual values after installation. All setrlimit and non-NULL setters
  are denied. Parent pre-exec setters run before the additional child filter.
- Source references: [Linux seccomp UAPI](https://github.com/torvalds/linux/blob/v6.6/include/uapi/linux/seccomp.h),
  [seccomp system call](https://man7.org/linux/man-pages/man2/seccomp.2.html),
  [resource limits](https://man7.org/linux/man-pages/man2/getrlimit.2.html),
  [waitid/WNOWAIT](https://man7.org/linux/man-pages/man2/waitid.2.html),
  [cgroup-v2 controls](https://www.kernel.org/doc/html/v6.6/admin-guide/cgroup-v2.html).
  These explain the control semantics; actual running kernel behavior still
  requires ROOT's real readbacks and negative fixtures.

## Failure review

- Source parsing/startup failures enter independent native retirement. Cleanup
  stages do not skip both KILL attempts after a TERM/grace failure. Any unknown
  result stays fenced with actual child/pipes/charge retained in-process.
- Normal/refusal child completion is observed before cleanup signals; otherwise
  cleanup TERM could replace an expected0/125 status. Cancellation keeps stdin
  open through terminal retirement, avoiding EOF-error output racing TERM.
- All signals validate unreaped original PID/SID/PGID. No process/group-changing
  child path remains permitted after the startup filter. Unknown pre-filter
  failures still refuse; complete outer container teardown remains mandatory.
- Successful typed ticket retirement is checked after child reap/group absence/
  pipe EOF. Arc Weak verifies actual caller-reference ticket lifetime; it does
  not substitute for a real dropped service future or source-window ownership.
- Native output is bounded before JSON allocation; final reports/closure/proc
  reads have fixed caps. Large prospective closure SHA work is ROOT-only. Raw
  source/model/DSO payloads were not read/hashed by the author.
- Receipt persistence or parent-process death can prevent a complete native
  failure receipt. Outer log/cleanup is still required; no such attempt can
  pass. Whole production host admission/unknown durability remains unqualified.

Only Rust formatting, Python AST parsing and small source/metadata copy/hash
checks are author validation. No compilation, fixtures, process/control probe,
native provider load/math, Torch/model or GPU execution occurred.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/TODO.md

Original UTF-8 bytes: 1470; SHA256 `ea5b776e90e8980904ea3322d49ecb502b5de4e461ec751f836ed582fa4344b2`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native provider startup proposal

Parent: existing shared mathematical-engine original-HF continuation.
Owner: batch_design_review. Scope: this ignored derivative only.

- [x] Preserve audit freezeee7761fb and all original native actual3 evidence.
- [x] Prepare same Rust binary/original ResourceArbiter startup owner and child.
- [x] Prepare exact pre-exec limits + actual pre-provider readbacks/refusals.
- [x] Prepare bounded same-ticket cancellation/known retirement/final0 checks.
- [x] Copy original14DSO metadata and reviewed owned-v4 helper unchanged.
- [x] Prepare independent validator/schema regression source; no fixture run.
- [x] Document production/kernel/control/provider/thread/death gates explicitly.
- [ ] ROOT independent full review and source-free fixture execution.
- [ ] ROOT strict build/effective-lock/binary/ELF binding.
- [ ] ROOT actual six startup cases/independent validator/cleanup readbacks.
- [ ] ROOT actual new Rust-owner failure/unknown/death negatives.
- [ ] Establish whole production physical host envelope/domain/control coverage.
- [ ] Admit exact provider load/constructors/threads before actual3 EXP16 replay.
- [ ] Shared-engine provider integration + full-HF/FLA/27B acceptance, separate.

Author results: source-only Rust formatting/Python AST/source freeze permitted;
no builds, tests, processes, cgroup/control probes, DSO/model/payload hashes,
dlopen/provider/math/GPU execution. No MAIN/shared-plan edits or commits.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/followup-sigterm-v2/README.md

Original UTF-8 bytes: 6700; SHA256 `e24572a952a53cd07e76e7c515e86b2ac82d752187709324a2a8ac0af08d4496`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Startup SIGTERM adapter V2

Author: baseline_review. Source-only additive proposal. Original fifteen files,
original961d freeze, original Rust code, existing53ee owned helper, provider
closure and all native limits remain unchanged. No author fixture/build/process,
provider/container/operator/GPU execution occurred.

- A SIGTERM handler records cancellation; it never raises asynchronously across
  Popen assignment or the original retirement sequence.
- Original Log.drain is wrapped only during original Log.observe. Synchronous
  checkpoints before/after each original bounded drain enter the original
  except/finally on cancellation. The original drain is restored before helper
  cleanup/log.finish. Repeated SIGTERM cannot interrupt those cleanup stages.
- Checks bracket Popen. A signal during ownership handoff is deferred until the
  actual returned child is owned, then original cleanup retires it. A signal
  during cleanup is reported after cleanup and prevents another case.
- SIGTERM before owned launch is checked before Popen; preflight commands retain
  their original finite deadlines. Cancellation is cooperative, not an immediate
  kill/deadline or protection from SIGKILL/parent death/irrecoverable interpreter
  failure. Arbitrary provider descendants remain governed by the original helper.
- Original cleanup catches/receipts and failure policy are unchanged. No second
  lifecycle, signal_group implementation or new container target is introduced.
- `freeze_sha256` remains original961d and binds the unchanged native build.
  `execution_adapter_freeze_sha256` separately binds this Python adapter in the
  prepared invocation, actual invocation, complete and validation receipts.
  Cancellation produces an UNQUALIFIED failed invocation and no next case or
  complete receipt. Successful validation retains every original native check.

## ROOT qualification cut

All commands below are ROOT-only, serialized with existing Cargo/GPU schedules.
Do not run a benchmark while building or executing these probes.

1. Review `SIGTERM_V2.review.diff`, guard and six synthetic fixtures. Run fixtures
   from this directory with `python3 test_sigterm_v2.py`. They replace Popen,
   helper imports/commands and signal registration with in-memory fakes; invoke
   the captured callback directly; never send an OS signal or launch a process.
   They exercise original main's original finally, stdout closure, restored
   drain/handler, primary+cleanup errors and no next case/false complete. They
   establish control-flow coverage only, not real process retirement.
2. Revalidate original961d all fifteen files and twenty-four source pins, plus
   this separate freeze. Do not retag a changed native source as original961d.
3. Build the ORIGINAL manifest, not a copied Cargo.toml in this child (its
   relative original core path would then be wrong). From repository root:

   ```bash
   startup_probe_dir="$PWD/.agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1"
   CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
     cargo generate-lockfile --offline --manifest-path "$startup_probe_dir/Cargo.toml"
   CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
     cargo clippy --offline --locked --release --manifest-path "$startup_probe_dir/Cargo.toml" -- -D warnings
   CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
     cargo build --offline --locked --release --manifest-path "$startup_probe_dir/Cargo.toml"
   ```

   Preserve the effective standalone Cargo.lock and real command/log/exit. Its
   actual registry tuples must match approved MAIN versions/checksums; no fetch,
   lock bypass or new toolchain/provider. Locate the actual target artifact from
   ROOT's effective Cargo target configuration; do not assume target directory.
   Record the actual binary SHA and effective-lock SHA. The launcher independently
   requires max GLIBC <=2.39 and the original bounded DT_NEEDED set. It executes
   via the original image's GNU loader, never the host Nix interpreter/libm.
4. Write ROOT's actual build-attestation JSON ONLY after that build passes:

   ```json
   {
     "status": "ROOT_COMPILED_STARTUP_ONLY",
     "freeze_sha256": "961d803b4787cd80d63a5efe9df37f7a57eb7cdfc1fe0ed0b3b8b02afc284bf3",
     "binary": "/ABS/ACTUAL/native-cpu-startup-probe",
     "binary_sha256": "ACTUAL_BINARY_SHA256",
     "effective_lock_sha256": "ACTUAL_STANDALONE_LOCK_SHA256",
     "build_exit_code": 0
   }
   ```

   This adapter's new freeze must NOT replace the original build freeze field.
5. Use the existing ROOT private-network bootstrap with custom-store Podman
   unshare, not a new native-gpu single-UID namespace around Podman. That latter
   route already failed storage mount ownership. Same current ROOT wrapper:
   `.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/root-private-loopback-exec-2.py`.
   Example from repository root, replacing only actual uppercase values:

   ```bash
   podman --root /home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers \
     --runroot /run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers \
     unshare unshare --net --mount --propagation private \
     python3 "$PWD/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/root-private-loopback-exec-2.py" \
     python3 "$startup_probe_dir/followup-sigterm-v2/run_probe_v2.py" \
       --binary /ABS/ACTUAL/native-cpu-startup-probe \
       --binary-sha256 ACTUAL_BINARY_SHA256 \
       --build-attestation /ABS/ROOT-build.json \
       --new-output "$startup_probe_dir/root-sigterm-v2-attempt-1"
   ```

   Output MUST be a fresh direct child of original startup-v1, not a child of
   followup-sigterm-v2. Retain ROOT's actual wrapper/invocation/exit receipt. The
   handler can only receive signals delivered to this Python process; this is
   not propagation qualification through Podman/unshare ancestors.
6. Validate all SIX actual cases, without SIGTERM injection, with:

   ```bash
   python3 "$startup_probe_dir/followup-sigterm-v2/validate_v2.py" \
     --output "$startup_probe_dir/root-sigterm-v2-attempt-1"
   ```

   Existing original actual3 BASE image is pinned by original static closure;
   derived codec image is not interchangeable. `ZG_STARTUP_CPUS` is native
   startup's variable; `ZG_ALLOWED_CPUS` belongs to other Python probes. No host
   Nix mounts, downloads or GPU passthrough. Real startup/outer signal/unknown
   failure/physical host admission remain UNQUALIFIED until ROOT executes their
   explicit gates. Passing these six cases admits no native math/full HF engine.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/followup-sigterm-v2/TODO.md

Original UTF-8 bytes: 959; SHA256 `02e6d2083eb65479004563767c9ffd3219ed6ada3a391390817027a5080a0eaf`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Startup SIGTERM V2 working scope

- Parent: existing model-execution native-provider admission continuation.
- Owned: this additive ignored child directory only; original961d/53ee unchanged.
- Prepared: cooperative SIGTERM checkpoints into original finally; separate
  native-build/execution binding; unchanged independent native validation.
- Prepared: six harmless synthetic fixtures (all process/helper/signal calls
  faked), exact ROOT original-manifest build and six-case invocation instructions.
- Static only: AST parsing, small source hash/freeze/diff inspection.
- Not run: fixtures/tests/build/container/provider/operator/model/GPU.
- ROOT pending: independent source review; six fixtures; original strict/release
  build + actual lock/binary attestation; six real startup cases + full validator.
- Still closed: production physical admission, provider load/math, new threads,
  full HF/FLA/27B, parent-death/persistent unknown-completion safety.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/followup-sigterm-v2/frozen/README.md

Original UTF-8 bytes: 6700; SHA256 `e24572a952a53cd07e76e7c515e86b2ac82d752187709324a2a8ac0af08d4496`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Startup SIGTERM adapter V2

Author: baseline_review. Source-only additive proposal. Original fifteen files,
original961d freeze, original Rust code, existing53ee owned helper, provider
closure and all native limits remain unchanged. No author fixture/build/process,
provider/container/operator/GPU execution occurred.

- A SIGTERM handler records cancellation; it never raises asynchronously across
  Popen assignment or the original retirement sequence.
- Original Log.drain is wrapped only during original Log.observe. Synchronous
  checkpoints before/after each original bounded drain enter the original
  except/finally on cancellation. The original drain is restored before helper
  cleanup/log.finish. Repeated SIGTERM cannot interrupt those cleanup stages.
- Checks bracket Popen. A signal during ownership handoff is deferred until the
  actual returned child is owned, then original cleanup retires it. A signal
  during cleanup is reported after cleanup and prevents another case.
- SIGTERM before owned launch is checked before Popen; preflight commands retain
  their original finite deadlines. Cancellation is cooperative, not an immediate
  kill/deadline or protection from SIGKILL/parent death/irrecoverable interpreter
  failure. Arbitrary provider descendants remain governed by the original helper.
- Original cleanup catches/receipts and failure policy are unchanged. No second
  lifecycle, signal_group implementation or new container target is introduced.
- `freeze_sha256` remains original961d and binds the unchanged native build.
  `execution_adapter_freeze_sha256` separately binds this Python adapter in the
  prepared invocation, actual invocation, complete and validation receipts.
  Cancellation produces an UNQUALIFIED failed invocation and no next case or
  complete receipt. Successful validation retains every original native check.

## ROOT qualification cut

All commands below are ROOT-only, serialized with existing Cargo/GPU schedules.
Do not run a benchmark while building or executing these probes.

1. Review `SIGTERM_V2.review.diff`, guard and six synthetic fixtures. Run fixtures
   from this directory with `python3 test_sigterm_v2.py`. They replace Popen,
   helper imports/commands and signal registration with in-memory fakes; invoke
   the captured callback directly; never send an OS signal or launch a process.
   They exercise original main's original finally, stdout closure, restored
   drain/handler, primary+cleanup errors and no next case/false complete. They
   establish control-flow coverage only, not real process retirement.
2. Revalidate original961d all fifteen files and twenty-four source pins, plus
   this separate freeze. Do not retag a changed native source as original961d.
3. Build the ORIGINAL manifest, not a copied Cargo.toml in this child (its
   relative original core path would then be wrong). From repository root:

   ```bash
   startup_probe_dir="$PWD/.agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1"
   CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
     cargo generate-lockfile --offline --manifest-path "$startup_probe_dir/Cargo.toml"
   CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
     cargo clippy --offline --locked --release --manifest-path "$startup_probe_dir/Cargo.toml" -- -D warnings
   CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
     cargo build --offline --locked --release --manifest-path "$startup_probe_dir/Cargo.toml"
   ```

   Preserve the effective standalone Cargo.lock and real command/log/exit. Its
   actual registry tuples must match approved MAIN versions/checksums; no fetch,
   lock bypass or new toolchain/provider. Locate the actual target artifact from
   ROOT's effective Cargo target configuration; do not assume target directory.
   Record the actual binary SHA and effective-lock SHA. The launcher independently
   requires max GLIBC <=2.39 and the original bounded DT_NEEDED set. It executes
   via the original image's GNU loader, never the host Nix interpreter/libm.
4. Write ROOT's actual build-attestation JSON ONLY after that build passes:

   ```json
   {
     "status": "ROOT_COMPILED_STARTUP_ONLY",
     "freeze_sha256": "961d803b4787cd80d63a5efe9df37f7a57eb7cdfc1fe0ed0b3b8b02afc284bf3",
     "binary": "/ABS/ACTUAL/native-cpu-startup-probe",
     "binary_sha256": "ACTUAL_BINARY_SHA256",
     "effective_lock_sha256": "ACTUAL_STANDALONE_LOCK_SHA256",
     "build_exit_code": 0
   }
   ```

   This adapter's new freeze must NOT replace the original build freeze field.
5. Use the existing ROOT private-network bootstrap with custom-store Podman
   unshare, not a new native-gpu single-UID namespace around Podman. That latter
   route already failed storage mount ownership. Same current ROOT wrapper:
   `.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/root-private-loopback-exec-2.py`.
   Example from repository root, replacing only actual uppercase values:

   ```bash
   podman --root /home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers \
     --runroot /run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers \
     unshare unshare --net --mount --propagation private \
     python3 "$PWD/.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/root-private-loopback-exec-2.py" \
     python3 "$startup_probe_dir/followup-sigterm-v2/run_probe_v2.py" \
       --binary /ABS/ACTUAL/native-cpu-startup-probe \
       --binary-sha256 ACTUAL_BINARY_SHA256 \
       --build-attestation /ABS/ROOT-build.json \
       --new-output "$startup_probe_dir/root-sigterm-v2-attempt-1"
   ```

   Output MUST be a fresh direct child of original startup-v1, not a child of
   followup-sigterm-v2. Retain ROOT's actual wrapper/invocation/exit receipt. The
   handler can only receive signals delivered to this Python process; this is
   not propagation qualification through Podman/unshare ancestors.
6. Validate all SIX actual cases, without SIGTERM injection, with:

   ```bash
   python3 "$startup_probe_dir/followup-sigterm-v2/validate_v2.py" \
     --output "$startup_probe_dir/root-sigterm-v2-attempt-1"
   ```

   Existing original actual3 BASE image is pinned by original static closure;
   derived codec image is not interchangeable. `ZG_STARTUP_CPUS` is native
   startup's variable; `ZG_ALLOWED_CPUS` belongs to other Python probes. No host
   Nix mounts, downloads or GPU passthrough. Real startup/outer signal/unknown
   failure/physical host admission remain UNQUALIFIED until ROOT executes their
   explicit gates. Passing these six cases admits no native math/full HF engine.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/followup-sigterm-v2/frozen/TODO.md

Original UTF-8 bytes: 959; SHA256 `02e6d2083eb65479004563767c9ffd3219ed6ada3a391390817027a5080a0eaf`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Startup SIGTERM V2 working scope

- Parent: existing model-execution native-provider admission continuation.
- Owned: this additive ignored child directory only; original961d/53ee unchanged.
- Prepared: cooperative SIGTERM checkpoints into original finally; separate
  native-build/execution binding; unchanged independent native validation.
- Prepared: six harmless synthetic fixtures (all process/helper/signal calls
  faked), exact ROOT original-manifest build and six-case invocation instructions.
- Static only: AST parsing, small source hash/freeze/diff inspection.
- Not run: fixtures/tests/build/container/provider/operator/model/GPU.
- ROOT pending: independent source review; six fixtures; original strict/release
  build + actual lock/binary attestation; six real startup cases + full validator.
- Still closed: production physical admission, provider load/math, new threads,
  full HF/FLA/27B, parent-death/persistent unknown-completion safety.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/frozen/NEGATIVE_PLAN.md

Original UTF-8 bytes: 3359; SHA256 `c936c5c3e8cefe28808d0003e0a1ee1e1b4346e0678f520c34d09b79589736b4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Required physical negatives

Author: batch_design_review. Prepared, not executed by author.

## Included six-case first gate

| Case | Real action | Required result |
| --- | --- | --- |
| normal |admit, spawn same binary, readbacks, explicit EXIT|actual reaped0/group absent/EOF before release/final0|
| pre-cancel |cancel before request|no admission/no child/final0|
| admission-refusal |original host limit4GiB-1 vs real4GiB request|exact original typed capacity refusal/no child/final0|
| cancel-after-hello |drop caller Arc while real child blocks|same runtime ticket4GiB retained, child still alive, supervisor retires -15/-9 before release|
| as-pressure |mmap(PROT_NONE,AS+4096)|real ENOMEM; no RAM faulting, unchanged actual AS, normal retirement|
| bad-as |pre-exec AS=2GiB-4096|actual readback refuses before closure/provider work, child125, known retirement/final0|

Every successful child also must receive real EPERM trying to raise the AS hard
limit and invoke raw fork. An unexpectedly permitted fork child immediately
_exit126 and is reaped by its original parent before the probe refuses.

## Required before extending provider eligibility

These are not implemented or falsely passed by the six-case startup probe.
ROOT should assign narrow additive controls only after reviewing first results:

- Actual pthread/clone fallback route in pinned provider, fixed thread ceiling,
  no independent VM/exec/namespace path; explicit clone3 policy and x32 refusal.
  Current filter denies all clones and cannot qualify4-thread matrix calls.
- Inject malformed/oversized/partial frame, pipe EOF failure, startup failure,
  kill failure/timeout and parent-death paths; independently observe actual
  retained owner and fence until complete owned process/container retirement.
  A process-local leak is not a durable production quarantine implementation.
- Inject receipt I/O failures before/after spawn, before/after reap and during
  final report persistence. Preserve primary+cleanup failures; never release on
  uncertain process/group/pipe completion. Existing outer helper failure
  fixtures were independently accepted previously; this new Rust owner has
  not inherited actual execution qualification from those Python fixtures.
- Real absolute memory/control/thread/VMA/FD/cgroup accounting closure covering
  parent, worker, kernel metadata, file cache and any helper. Test denial under
  real arbiter pressure without spawning an unadmitted provider or relaxing cap.
- Only after closure: load exact prospective provider and reattest actual mapped
  full14DSO+binary closure, constructor/thread/task effects, FP policy and native
  ABI symbols. Any new mapping/allocation path refuses until independently bound.
- Replay actual3 public EXP16 first, then genuine accepted BMM/TRSM operands on
  original pinned binary/source schedule. Parent independently compares outputs;
  expected values are never sent to worker as an implementation substitute.
- Persistent optional MathematicalBackend provider adapter may issue only typed
  primitive requests; parent retains input/output source windows and physical
  leases. Worker must never become a model/token/session engine or state issuer.

Synthetic `test_contracts.py` fixtures test strict receipt rejection only.
They do not simulate or establish kernel/process/provider implementation.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/frozen/POLICY.md

Original UTF-8 bytes: 4419; SHA256 `d40a45cd0af8206459aa4254a3a70a6c8eebde8d5b517a205617fd08c7f7018d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Startup policy and ownership

Author: batch_design_review. Source-only; ROOT must execute every control.

## Before child spawn

- ROOT launcher: exact already acquired image ID/manifest, pull never, network
  none, read-only image, no GPU/device grant, all caps dropped, NNP,12GiB memory
  with total memory+swap12GiB (therefore swap0), pids64 and CPU quota4. No OCI
  cpuset requirement; explicit sched_setaffinity is set/read back in Rust.
- Only binary RO and fresh output RW mounts. No learned source/model or source
  issuer. Parent's configured original host key admits4GiB ProviderEnvelope
  before `Command::spawn`. Pre-cancel and capacity refusal spawn no child.
- Same exact binary is launched through original public GNU loader with original
  static14DSO search paths. Source pins and actual ELF/mapped-byte identity must
  remain exact. No DSO or provider acquisition by this code.

## Child controls

| Control | Exact policy | Proof scope |
| --- | --- | --- |
| Address space | AS soft=hard2GiB |set in pre-exec, actual readback before any prospective DSO hash/load; hard raise must EPERM|
| Core |0|actual soft/hard readback|
| FDs |32 hard; startup observed<=16|actual readback/count; no future provider claim|
| Stack |8MiB hard|actual readback|
| CPU time |60 seconds hard|actual readback; outer wall limit remains separate|
| Output files |4MiB hard|actual readback; parent output report<=128KiB|
| UID/caps |same parent effective UID, all four UID fields; all Eff/Prm/Bnd/Amb0|actual raw status + public UID calls; rootless namespace UID0 alone is not elevated host authority|
| NNP |1|pre-exec setting + actual prctl/raw readback|
| CPU affinity |exact four available CPU IDs|actual sched_getaffinity + raw status|
| Processes/threads |single child address space; all fork/vfork/clone/clone3 denied|actual tasks1 and real fork EPERM; pinned x86_64 seccomp UAPI rejects other arch/x32|
| Executables/namespaces |execve/execveat/unshare/setns/setpgid/setsid denied|source-derived seccomp; actual extra refusal fixtures still required below|
| Limit mutation |setrlimit denied; prlimit64 permits only NULL new_limit|source UAPI pointer-word checks, required for glibc getrlimit; no indirect pointer inspection claim|
| Other external allocations |SysV shmget/shmat, memfd, io_uring, userfaultfd, bpf, perf and ptrace denied|source policy only until actual negatives qualified|
| VMAs |observed<=4096; kernel max_map_count1..1,048,576|finite supported startup observation, not a per-worker physical metadata bound|

The filter intentionally permits ordinary file/pipe read/write, hashing,
allocation under AS and process exit. It is not a general hostile-code sandbox.
It does not currently permit actual native library thread initialization.
No provider is loaded before this filter or anywhere in this binary.

## Finite IPC and retained lifetime

- Child stdout: one JSON frame<128KiB plus newline; parent polls50ms with60s
  startup deadline, rejects EOF/oversized frames. Child accepts only5 bytes
  `EXIT\n` then EOF. There is no operator payload or shared memory transport.
- Parent report<=128KiB; immutable metadata reads<=128KiB each; prospective file
  hashing uses one1MiB buffer at a time and max512MiB full allowed closure bytes.
  DSO/kernel/cache physical overhead remains unqualified despite these bounds.
- Native sole `Child` is not reaped until after all TERM/grace/KILL attempts.
  Each nonzero group signal rebinds original unreaped PID/SID/PGID. After reap,
  only signal0 absence checks occur. Known retirement additionally requires EOF.
- Supervisor keeps same original Arc ticket while caller view is dropped. It
  marks original lease Ready/releases only after trusted complete retirement.
  Unknown retirement fences and retains actual child/pipes/charge in-process;
  no production recovery or parent-death durability is claimed.
- Outer exact reviewed v4 helper independently handles native parent/container
  lifetime, two bounded KILL attempts before reap, final nonce/CID absence and
  log EOF/close. Each case120s wall limit plus finite cleanup; six serial cases.
  All post-Popen receipt/log setup is under cleanup `finally`.

Primary control contracts are the original audit's Linux getrlimit/seccomp and
cgroup-v2 references. No public documentation guarantees that the remaining
1,408MiB guard covers all physical kernel/control costs; that gate stays false.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/frozen/README.md

Original UTF-8 bytes: 6238; SHA256 `b809ce160003a7a150b311e74134307295ecb338f41273ff0f977153cf7c84fb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native provider startup gate

Author: batch_design_review. ROOT-only proposal; uncompiled and unexecuted.

Parent plan: existing model-execution/shared-engine continuation. Derivative of
the unchanged envelope audit freeze
`ee7761fb1ac99d96d9d5af3215432716f23bcf2591b0dfa63ee04110594ba08b`.
No MAIN source, provider, source/model permission, token loop or eligibility
change. This is a startup-control probe, not an admitted native math provider.

## Scope

- One native Rust executable: original ResourceArbiter parent + same-binary
  child. Parent admits one real4GiB ProviderEnvelope before spawn. Child has
  exact2GiB AS soft/hard before exec, then independently reads effective UID,
  capabilities, NNP, CPU affinity, limits and actual cgroup controls.
- Native parent/child run in the already acquired actual3 base image and exact
  GNU loader. Original14DSO candidate manifest is copied byte-exact. Child
  authenticates all14 prospective files and the new ROOT binary, records the
  actual mapped subset, and refuses if the provider is already loaded.
- No dlopen, VML, CBLAS, STRSM, model/tensor source, source issuer, StateRoot,
  session, GPU device or native math call exists in this probe.
- The single-thread child additionally denies all fork/clone/exec and namespace
  changes. Actual fork and oversized nonresident mmap refusals are tested in
  their own cases. Four-thread native math remains a later gate.
- Same original GlobalResourceLease held by the supervisor survives dropping a
  caller Arc while the actual child waits. The lease is released only after
  owned child reap, group absence and stdout EOF. This tests a startup caller
  reference, **not a dropped asynchronous MathematicalBackend service future**.

## Explicit remaining admission gates

- Diagnostic host key/8GiB limit is configured inside the native process, using
  the same original process-shared ResourceArbiter API as existing isolated
  source-window proofs. It is not an attested production host domain/limit or
  a cross-process resource ledger.
- The outer container and Podman/conmon processes exist before native admission.
  Their host control costs are not covered/proven by this first ticket. No
  assertion that memory.max cannot overshoot or that AS bounds kernel metadata.
-4GiB is a candidate envelope:2GiB AS +512MiB full file cache allowance +128MiB
  IPC/control allowance +1,408MiB unproved guard. A guard allowance is not a
  physical bound. Status fields deliberately keep kernel/control and production
  physical provider admission false, even after all six startup cases pass.
- Single-thread startup policy, observed4096VMAs/16FDs and a bounded public
  vm.max_map_count readback do not qualify future provider/OpenMP/FastMM costs.
  Provider constructors, new thread control, exact mapped closure after load,
  actual exp/BLAS arithmetic and their whole physical lifetime remain closed.
- On unknown retirement, original ticket and actual child/pipe owner are leaked
  behind the original arbiter fence for the native parent's remaining lifetime.
  No retry or release occurs. Parent death/outer cleanup is not a persistent
  cross-process quarantine proof; the attempt fails, never passes this gate.

## Files

- `common.rs`: bounded observation/hash transport; original closure metadata.
- `worker.rs`: child readbacks, startup seccomp and real negative syscalls.
- `main.rs`: original admission, sole child ownership, finite retirement.
- `run_probe.py`: ROOT-only six-case launcher; exact copied owned-v4 cleanup.
- `validate.py`: independent raw/control/ledger/lifetime receipt checks.
- `test_contracts.py`: synthetic schema regressions, never runtime evidence.
- `POLICY.md`, `NEGATIVE_PLAN.md`: exact scope and required next physical gates.
- `freeze-manifest.json`, `frozen/`: source-only additive freeze.

## ROOT execution order

Review all files and source pins first. No author build, fixture run, container,
process/control probe, DSO hash/load or math execution has occurred.

1. Run prepared schema fixtures on ROOT's serialized CPU schedule:
   `python3 test_contracts.py` from this directory.
2. Generate the standalone effective lockfile, run strict Clippy/build under
   ROOT's original isolation/2-job coordination. Cargo resolves no new intended
   versions: libc0.2.190, serde_json1.0.151, sha2 0.10.9, original core path.
   Record actual source freeze, effective lock, successful build and binary hash.
   Build may refuse if host-linked symbol requirements exceed pinned glibc2.39;
   do not silently substitute another libc/provider/image.
3. Create a ROOT build JSON (not supplied/fabricated by the author):

   ```json
   {
     "status": "ROOT_COMPILED_STARTUP_ONLY",
     "freeze_sha256": "ACTUAL_STARTUP_FREEZE_SHA",
     "binary": "/absolute/path/to/native-cpu-startup-probe",
     "binary_sha256": "ACTUAL_COMPILED_BINARY_SHA",
     "effective_lock_sha256": "ACTUAL_EFFECTIVE_CARGO_LOCK_SHA",
     "build_exit_code": 0
   }
   ```

   Retain the actual command/log and effective lock with that receipt; a manually
   written success label is not compilation evidence. The launcher requires the
   new binary/receipt binding and checks its actual ELF requirements.
4. Run once with fresh output (replace only uppercase placeholders):

   ```bash
   python3 run_probe.py --binary /absolute/path/to/native-cpu-startup-probe \
     --binary-sha256 ACTUAL_COMPILED_BINARY_SHA \
     --build-attestation /absolute/path/to/ROOT-build.json \
     --new-output /home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/root-attempt-1
   ```

5. Independently validate the output with `python3 validate.py --output` and the
   same absolute new output path. Freeze failed attempts; use a fresh derivative
   and fresh output for source corrections. No automatic fallback/provider
   switch or model inference is allowed by this launcher.

Expected accepted scope: `PASS_STARTUP_OWNERSHIP_ONLY_PROVIDER_UNLOADED`.
After acceptance, first extend an independently admitted actual3 EXP16 call
with unchanged provider/mode/arithmetic and explicit lifecycle/resource proof.
No full-HF/FLA/27B/vLLM claim follows from this startup probe.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/frozen/SOURCE_AUDIT.md

Original UTF-8 bytes: 4223; SHA256 `d47ea76b9d53d056e763a07e70d8619be00b83cbc681db74700324888ea93eb0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Source boundary

Author: batch_design_review. Source reading only; no actual controls executed.

## Reused original contracts

- `resource_arbiter.rs`: process_shared, register_matching, exact typed Limit
  capacity refusal, ProviderEnvelope admission, mark_submitted/mark_ready/release
  and fence_unknown. Original Submitted-drop retains/fences uncertain work.
  No original source byte is changed; the new binary has no independent arbiter
  implementation. Configured diagnostic key is explicitly not production proof.
- Accepted actual3 native probe: copied exact static14DSO manifest, exact base
  image/ID and GNU loader/library path/cache inhibition. Actual3 arithmetic and
  memory receipts are read as prior evidence, not rerun/extended by author.
  That original binary had15 mapped files including its own executable; this
  startup probe authenticates a new ROOT-compiled binary as the fifteenth item.
- Reviewed `owned_container_v4.py` is byte-exact, copied from immutable GEMM v2.
  Native owner mirrors its reserved-leader WNOWAIT/bind-before-signal approach;
  copying the lifecycle design does not inherit physical test acceptance.

## Public Linux/libc boundary

- Pinned libc0.2.190 public x86_64 GNU bindings provide rlimit types, waitid,
  siginfo accessors, CPU_SET/ISSET, prctl and classic BPF sock_filter/fprog.
  Exact selected source bytes are pinned with this proposal.
- Classic Linux seccomp_data layout is nr@0, arch@4, instruction_pointer@8,
  six64-bit arguments@16. Therefore prlimit64 new_limit=args[2] is read at32/36;
  no user-pointer dereference or clone3 indirect-argument interpretation occurs.
  Exact x86_64 audit arch0xc000003e is required; syscall numbers>=0x40000000
  (including x32) are rejected before the native syscall-number table.
- The filter permits prlimit64 only with a NULL new_limit so glibc getrlimit
  can read actual values after installation. All setrlimit and non-NULL setters
  are denied. Parent pre-exec setters run before the additional child filter.
- Source references: [Linux seccomp UAPI](https://github.com/torvalds/linux/blob/v6.6/include/uapi/linux/seccomp.h),
  [seccomp system call](https://man7.org/linux/man-pages/man2/seccomp.2.html),
  [resource limits](https://man7.org/linux/man-pages/man2/getrlimit.2.html),
  [waitid/WNOWAIT](https://man7.org/linux/man-pages/man2/waitid.2.html),
  [cgroup-v2 controls](https://www.kernel.org/doc/html/v6.6/admin-guide/cgroup-v2.html).
  These explain the control semantics; actual running kernel behavior still
  requires ROOT's real readbacks and negative fixtures.

## Failure review

- Source parsing/startup failures enter independent native retirement. Cleanup
  stages do not skip both KILL attempts after a TERM/grace failure. Any unknown
  result stays fenced with actual child/pipes/charge retained in-process.
- Normal/refusal child completion is observed before cleanup signals; otherwise
  cleanup TERM could replace an expected0/125 status. Cancellation keeps stdin
  open through terminal retirement, avoiding EOF-error output racing TERM.
- All signals validate unreaped original PID/SID/PGID. No process/group-changing
  child path remains permitted after the startup filter. Unknown pre-filter
  failures still refuse; complete outer container teardown remains mandatory.
- Successful typed ticket retirement is checked after child reap/group absence/
  pipe EOF. Arc Weak verifies actual caller-reference ticket lifetime; it does
  not substitute for a real dropped service future or source-window ownership.
- Native output is bounded before JSON allocation; final reports/closure/proc
  reads have fixed caps. Large prospective closure SHA work is ROOT-only. Raw
  source/model/DSO payloads were not read/hashed by the author.
- Receipt persistence or parent-process death can prevent a complete native
  failure receipt. Outer log/cleanup is still required; no such attempt can
  pass. Whole production host admission/unknown durability remains unqualified.

Only Rust formatting, Python AST parsing and small source/metadata copy/hash
checks are author validation. No compilation, fixtures, process/control probe,
native provider load/math, Torch/model or GPU execution occurred.

````

## Source: .agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/frozen/TODO.md

Original UTF-8 bytes: 1470; SHA256 `ea5b776e90e8980904ea3322d49ecb502b5de4e461ec751f836ed582fa4344b2`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native provider startup proposal

Parent: existing shared mathematical-engine original-HF continuation.
Owner: batch_design_review. Scope: this ignored derivative only.

- [x] Preserve audit freezeee7761fb and all original native actual3 evidence.
- [x] Prepare same Rust binary/original ResourceArbiter startup owner and child.
- [x] Prepare exact pre-exec limits + actual pre-provider readbacks/refusals.
- [x] Prepare bounded same-ticket cancellation/known retirement/final0 checks.
- [x] Copy original14DSO metadata and reviewed owned-v4 helper unchanged.
- [x] Prepare independent validator/schema regression source; no fixture run.
- [x] Document production/kernel/control/provider/thread/death gates explicitly.
- [ ] ROOT independent full review and source-free fixture execution.
- [ ] ROOT strict build/effective-lock/binary/ELF binding.
- [ ] ROOT actual six startup cases/independent validator/cleanup readbacks.
- [ ] ROOT actual new Rust-owner failure/unknown/death negatives.
- [ ] Establish whole production physical host envelope/domain/control coverage.
- [ ] Admit exact provider load/constructors/threads before actual3 EXP16 replay.
- [ ] Shared-engine provider integration + full-HF/FLA/27B acceptance, separate.

Author results: source-only Rust formatting/Python AST/source freeze permitted;
no builds, tests, processes, cgroup/control probes, DSO/model/payload hashes,
dlopen/provider/math/GPU execution. No MAIN/shared-plan edits or commits.

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/README.md

Original UTF-8 bytes: 6132; SHA256 `432d9e563124dc6b6a86f2e96b700c48e3440cbb735cd2b6bf46b2e803127940`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only authentic CPU attention fragment replay

- Parent: existing model-execution plan. Source preparation only; root executes.
- Current MAIN APIs only: sealed original `VerifiedTensorSource` →
  `OriginalWindowOperators::bind/execute_attention_projection_norm_slice` plus
  explicit `AttentionSliceInput`. Same shared HostMath/ForwardPlan operations;
  no alternate mathematical implementation, token loop, state or input node.
- Sources: original .8B@2fc063… and 2B@15852…; raw original files/headers/source
  identity, BF16 matrices/learned QK gamma and epsilon1e-6 stay intact. All320
  text-role descriptors bind. Only `source-36-f32-restored` CPU AVX2 semantics;
  official all-BF16 loader and CUDA/vLLM mixed cast profile remain distinct.

## Independent reference and exact scope

- Two genuine prior teacher inputs,11 then353, at first FullAttention layer3.
  Pure fragments receive already-observed authentic hidden inputs; they do not
  produce those inputs or replay the preceding model/cache computation.
- .8B accepted root-probe-2: immutable whole archive pins plus actual call001
  original public schemas, full source parameters, input byte digests, split
  query frame/gate-flat frame and final type_as BF16 norm outputs. Parent all98
  logits/cache checks,12attention+2rotary and2802public checks stay mandatory.
- 2B accepted root-pair-1: e45e1903… paired validation/f969e1a4… observed manifest,
  both plain/observed196raw tensors/four fullV/168domains/60prior intermediates
  exact to accepted reference; raw source public-hook stage/layer/strides.
  No additional Torch/operator/model/container or D2H operation is required.
- Exporter copies only original raw BF16 words: four inputs plus seven expected
  outputs per case =106,496B. Expected tensors are never reconstructed from
  projections, rounded, zero-normalized or fitted. Independent validator
  rechecks exported words against the original authenticated archive spans.
- Rust compares all47,104 output words (23,552/source) as raw BF16 widened bits,
  including signed zero. Seven outputs: query/gate projection,K projection,V
  projection,query,gate,normalized query,normalized key. Full lengths required.
  Actual raw BF16 output files permit independent stdlib byte comparison.

## Real ownership and negative gates

- Same original full-shard/index FD/`prepare_verified_windows/open` path as the
  accepted root window factory; no materialized 1.7/4.5GB learned cache or fake
  reduced descriptors. Root performs complete source-file scans before/after;
  original window open independently authenticates the full weight shard.
- One128MiB host ResourceArbiter domain per sequential source:64MiB admitted
  source metadata,4MiB binding metadata,2MiB harness control/reference envelope.
  Existing per-value64KiB Output+960KiB Scratch retain original source issuer.
  Each case retains eight backings (input+seven outputs),16MiB for both cases.
- Actual pool exhausted through same-source workspace leases; execution must
  refuse and release partial private work. Pre-cancel and wrong GDN layer also
  refuse. Mid-cancel is triggered by actual >=2 Output backings plus a real
  TransferStaging charge. Missing this short physical boundary fails the case;
  it cannot be reported as passed merely because a timer elapsed.
- Both outputs survive dropping callers/operators/source, are compared again,
  then release all source and backing owners. Final reservations/bytes0 and
  no fence required. No SessionAuthority/StateRoot is allocated or published.
- Pool accounting covers named payloads/explicit metadata. Generic allocator,
  executable/JSON runtime bookkeeping, OS caches and RSS do not become physical
  free-RAM admission claims. No event-complete peak measurement is claimed.
- Native launcher uses the exact accepted root resolver WNOWAIT/owned SID/PGID
  cleanup functions. Independent TERM/grace/KILL stages occur before any reap;
  final reap, group absence, bounded pipe drain and closure required. Both
  helper invocations are serialized and use the same pinned executable FD.

## ROOT execution after independent source review

All commands are prepared, **unexecuted by author**. Keep old accepted archives,
source observations and prior attention proposal bytes immutable.

```sh
cd /home/pvl/spaces/abird/src/zerogravity
RUN="$PWD/.agents/runs/model-execution-original-attention-replay-20261020"
# Source-free parser/transport fixtures; no trained qualification.
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$RUN" -p test_contracts.py -v
# Genuine existing archive transport only; fresh directory directly under RUN.
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/prepare_inputs.py" --new-output "$RUN/root-inputs-1"
# Root records the actual Cargo argv, source freeze and fresh binary SHA.
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --offline --release --manifest-path "$RUN/Cargo.toml"
# Actual helper/inputs SHA values are obtained by ROOT, never placeholder pins.
python3 scripts/native-gpu.py -- python3 "$RUN/run_replay.py" \
  --binary ROOT_FRESH_BINARY --binary-sha256 ROOT_ACTUAL_SHA \
  --inputs "$RUN/root-inputs-1/inputs.json" --inputs-sha256 ROOT_INPUTS_SHA \
  --new-output "$RUN/root-replay-1"
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/validate_replay.py" --output "$RUN/root-replay-1"
```

Root must retain actual build stdout/argv and source/binary attestation; the
launcher cannot prove compilation provenance merely from a supplied SHA. Its
freeze binds the exact current MAIN source bytes and API dependencies. Cargo
lock normalization follows the accepted root helper policy; inspect any delta.
No dependency fetch or SDK/model acquisition is needed. Root schedules complete
source scans outside benchmarks and competing model/GPU phases.

Possible scope after all checks: **original CPU private attention fragment only**.
RoPE,KV/cache append,BMM/softmax/gating integration,initial/cached GDN coefficients
and MKL scheduling,full-HF session/checkpoint/logits,27B,GPU/native cuBLAS and
matched serving performance remain unqualified.

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/TODO.md

Original UTF-8 bytes: 985; SHA256 `ace505a350cb48c671c6dc4ed3827b45f8a543b1c617e9d9b66639bb8a64577d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

- Parent: existing model-execution plan; owner batch_design_review.
- Owned scope: this new ignored run only. MAIN source/shared plans unchanged.
- [x] Review narrow current MAIN attention compile/shape fixes; no blocker.
- [x] Prepare same-engine authentic .8B/2B layer3 two-teacher seven-output harness.
- [x] Keep genuine public source/schema/profile/archive and parent transparency binding.
- [x] Prepare real same-source pool pressure,cancel,retained owner/final0 checks.
- [x] Reuse exact accepted owned-native lifecycle functions; preserve original.
- [ ] Independent source review,source-free fixtures,actual archive export.
- [ ] ROOT build/current source+fresh binary binding and exact47,104-word replay.
- [ ] ROOT strict/full phase checks and acceptance; no implementation promotion here.

Author: AST/source inspection only. No tests,compilation,model/operator/GPU or
container execution; no large raw/model/DSO read/hash. No physical result claimed.

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/followup-v2/README.md

Original UTF-8 bytes: 7386; SHA256 `ef16c5e5f3624b13861104ca1c4ca6a1bee75623e74167fb219cc7f69db3c082`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only authentic CPU attention fragment replay — additive V2

- Parent: existing model-execution plan. Source preparation only; root executes.
- Current MAIN APIs only: sealed original `VerifiedTensorSource` →
  `OriginalWindowOperators::bind/execute_attention_projection_norm_slice` plus
  explicit `AttentionSliceInput`. Same shared HostMath/ForwardPlan operations;
  no alternate mathematical implementation, token loop, state or input node.
- Sources: original .8B@2fc063… and 2B@15852…; raw original files/headers/source
  identity, BF16 matrices/learned QK gamma and epsilon1e-6 stay intact. All320
  text-role descriptors bind. Only `source-36-f32-restored` CPU AVX2 semantics;
  official all-BF16 loader and CUDA/vLLM mixed cast profile remain distinct.

## V2 correction and preserved evidence

- Original12 prepared files/frozen copies and root-inputs-1 failure stay unchanged.
  Root found the original guard incorrectly compared a49-record metadata list
  to scalar49. The parent report counts remain scalar98/49; per-step metadata
  now requires the genuine full49 records, every bit_exact strictly true and
  the complete original ordered fullV/cache label namespace exactly once.
- No input/output/source/schema/math/shape/tolerance change. Rust/Cargo/bindings,
  exporter,validator and owned-lifetime implementations are copied byte-exact.
- Added metadata-only authentic49-record fixture and three regressions: genuine
  list accepted; scalar/incomplete list refused; onebad/nonboolean/missing flag
  or duplicate label refused. Root executes all10 fixtures; author AST only.
- Reinspected remaining actual small .8B/2B metadata types: report scalarcounts,
  replay_checks list, per-step tensor maps, public-call arguments,2B restored
  source names36, profile/step/token fields and both observed lifecycle schemas.
  No further metadata type mismatch found. No tensor payload read/hash or guard
  execution by author. Fresh ROOT extract/replay directories are required.

## Independent reference and exact scope

- Two genuine prior teacher inputs,11 then353, at first FullAttention layer3.
  Pure fragments receive already-observed authentic hidden inputs; they do not
  produce those inputs or replay the preceding model/cache computation.
- .8B accepted root-probe-2: immutable whole archive pins plus actual call001
  original public schemas, full source parameters, input byte digests, split
  query frame/gate-flat frame and final type_as BF16 norm outputs. Parent all98
  logits/cache checks,12attention+2rotary and2802public checks stay mandatory.
- 2B accepted root-pair-1: e45e1903… paired validation/f969e1a4… observed manifest,
  both plain/observed196raw tensors/four fullV/168domains/60prior intermediates
  exact to accepted reference; raw source public-hook stage/layer/strides.
  No additional Torch/operator/model/container or D2H operation is required.
- Exporter copies only original raw BF16 words: four inputs plus seven expected
  outputs per case =106,496B. Expected tensors are never reconstructed from
  projections, rounded, zero-normalized or fitted. Independent validator
  rechecks exported words against the original authenticated archive spans.
- Rust compares all47,104 output words (23,552/source) as raw BF16 widened bits,
  including signed zero. Seven outputs: query/gate projection,K projection,V
  projection,query,gate,normalized query,normalized key. Full lengths required.
  Actual raw BF16 output files permit independent stdlib byte comparison.

## Real ownership and negative gates

- Same original full-shard/index FD/`prepare_verified_windows/open` path as the
  accepted root window factory; no materialized 1.7/4.5GB learned cache or fake
  reduced descriptors. Root performs complete source-file scans before/after;
  original window open independently authenticates the full weight shard.
- One128MiB host ResourceArbiter domain per sequential source:64MiB admitted
  source metadata,4MiB binding metadata,2MiB harness control/reference envelope.
  Existing per-value64KiB Output+960KiB Scratch retain original source issuer.
  Each case retains eight backings (input+seven outputs),16MiB for both cases.
- Actual pool exhausted through same-source workspace leases; execution must
  refuse and release partial private work. Pre-cancel and wrong GDN layer also
  refuse. Mid-cancel is triggered by actual >=2 Output backings plus a real
  TransferStaging charge. Missing this short physical boundary fails the case;
  it cannot be reported as passed merely because a timer elapsed.
- Both outputs survive dropping callers/operators/source, are compared again,
  then release all source and backing owners. Final reservations/bytes0 and
  no fence required. No SessionAuthority/StateRoot is allocated or published.
- Pool accounting covers named payloads/explicit metadata. Generic allocator,
  executable/JSON runtime bookkeeping, OS caches and RSS do not become physical
  free-RAM admission claims. No event-complete peak measurement is claimed.
- Native launcher uses the exact accepted root resolver WNOWAIT/owned SID/PGID
  cleanup functions. Independent TERM/grace/KILL stages occur before any reap;
  final reap, group absence, bounded pipe drain and closure required. Both
  helper invocations are serialized and use the same pinned executable FD.

## ROOT execution after independent source review

All commands are prepared, **unexecuted by author**. Keep old accepted archives,
source observations and prior attention proposal bytes immutable.

```sh
cd /home/pvl/spaces/abird/src/zerogravity
RUN="$PWD/.agents/runs/model-execution-original-attention-replay-20261020/followup-v2"
# Source-free parser/transport fixtures; no trained qualification.
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$RUN" -p test_contracts.py -v
# Genuine existing archive transport only; fresh directory directly under RUN.
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/prepare_inputs.py" --new-output "$RUN/root-inputs-1"
# Root records the actual Cargo argv, source freeze and fresh binary SHA.
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --offline --release --manifest-path "$RUN/Cargo.toml"
# Actual helper/inputs SHA values are obtained by ROOT, never placeholder pins.
python3 scripts/native-gpu.py -- python3 "$RUN/run_replay.py" \
  --binary ROOT_FRESH_BINARY --binary-sha256 ROOT_ACTUAL_SHA \
  --inputs "$RUN/root-inputs-1/inputs.json" --inputs-sha256 ROOT_INPUTS_SHA \
  --new-output "$RUN/root-replay-1"
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/validate_replay.py" --output "$RUN/root-replay-1"
```

Root must retain actual build stdout/argv and source/binary attestation; the
launcher cannot prove compilation provenance merely from a supplied SHA. Its
freeze binds the exact current MAIN source bytes and API dependencies. Cargo
lock normalization follows the accepted root helper policy; inspect any delta.
No dependency fetch or SDK/model acquisition is needed. Root schedules complete
source scans outside benchmarks and competing model/GPU phases.

Possible scope after all checks: **original CPU private attention fragment only**.
RoPE,KV/cache append,BMM/softmax/gating integration,initial/cached GDN coefficients
and MKL scheduling,full-HF session/checkpoint/logits,27B,GPU/native cuBLAS and
matched serving performance remain unqualified.

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/followup-v2/TODO.md

Original UTF-8 bytes: 944; SHA256 `cac88c9601a498eb130e7ad0681b9c72a66e7e3eaa3d567e1f02c2c0f68dd004`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive V2 working checklist

- Parent: existing model-execution plan; owner batch_design_review.
- Owned scope: fresh followup-v2 only; original12/frozen/failure preserved.
- [x] Correct metadata prior_checks: actual49-record list, strict allbit_exact and exact full label cover; retain report scalar98/49.
- [x] Add genuine metadata-only listfixture plus three regressions; original seven fixtures unchanged.
- [x] Inspect remaining genuine small metadata schema/types without tensor payload/math/runtime.
- [x] Copy all other Rust/Cargo/bindings/export/validator/owned cleanup bytes unchanged.
- [ ] Independent ROOT/peer source review and10 sourcefree fixtures.
- [ ] ROOT fresh authenticated export, build/binary/source binding,47,104-word replay and independent byte validation.

Original actual root-inputs-1 export failure remains unqualified/preserved. No
author build/test/operator/model/GPU/container run or tensor/provider hash.

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/followup-v2/frozen/README.md

Original UTF-8 bytes: 7386; SHA256 `ef16c5e5f3624b13861104ca1c4ca6a1bee75623e74167fb219cc7f69db3c082`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only authentic CPU attention fragment replay — additive V2

- Parent: existing model-execution plan. Source preparation only; root executes.
- Current MAIN APIs only: sealed original `VerifiedTensorSource` →
  `OriginalWindowOperators::bind/execute_attention_projection_norm_slice` plus
  explicit `AttentionSliceInput`. Same shared HostMath/ForwardPlan operations;
  no alternate mathematical implementation, token loop, state or input node.
- Sources: original .8B@2fc063… and 2B@15852…; raw original files/headers/source
  identity, BF16 matrices/learned QK gamma and epsilon1e-6 stay intact. All320
  text-role descriptors bind. Only `source-36-f32-restored` CPU AVX2 semantics;
  official all-BF16 loader and CUDA/vLLM mixed cast profile remain distinct.

## V2 correction and preserved evidence

- Original12 prepared files/frozen copies and root-inputs-1 failure stay unchanged.
  Root found the original guard incorrectly compared a49-record metadata list
  to scalar49. The parent report counts remain scalar98/49; per-step metadata
  now requires the genuine full49 records, every bit_exact strictly true and
  the complete original ordered fullV/cache label namespace exactly once.
- No input/output/source/schema/math/shape/tolerance change. Rust/Cargo/bindings,
  exporter,validator and owned-lifetime implementations are copied byte-exact.
- Added metadata-only authentic49-record fixture and three regressions: genuine
  list accepted; scalar/incomplete list refused; onebad/nonboolean/missing flag
  or duplicate label refused. Root executes all10 fixtures; author AST only.
- Reinspected remaining actual small .8B/2B metadata types: report scalarcounts,
  replay_checks list, per-step tensor maps, public-call arguments,2B restored
  source names36, profile/step/token fields and both observed lifecycle schemas.
  No further metadata type mismatch found. No tensor payload read/hash or guard
  execution by author. Fresh ROOT extract/replay directories are required.

## Independent reference and exact scope

- Two genuine prior teacher inputs,11 then353, at first FullAttention layer3.
  Pure fragments receive already-observed authentic hidden inputs; they do not
  produce those inputs or replay the preceding model/cache computation.
- .8B accepted root-probe-2: immutable whole archive pins plus actual call001
  original public schemas, full source parameters, input byte digests, split
  query frame/gate-flat frame and final type_as BF16 norm outputs. Parent all98
  logits/cache checks,12attention+2rotary and2802public checks stay mandatory.
- 2B accepted root-pair-1: e45e1903… paired validation/f969e1a4… observed manifest,
  both plain/observed196raw tensors/four fullV/168domains/60prior intermediates
  exact to accepted reference; raw source public-hook stage/layer/strides.
  No additional Torch/operator/model/container or D2H operation is required.
- Exporter copies only original raw BF16 words: four inputs plus seven expected
  outputs per case =106,496B. Expected tensors are never reconstructed from
  projections, rounded, zero-normalized or fitted. Independent validator
  rechecks exported words against the original authenticated archive spans.
- Rust compares all47,104 output words (23,552/source) as raw BF16 widened bits,
  including signed zero. Seven outputs: query/gate projection,K projection,V
  projection,query,gate,normalized query,normalized key. Full lengths required.
  Actual raw BF16 output files permit independent stdlib byte comparison.

## Real ownership and negative gates

- Same original full-shard/index FD/`prepare_verified_windows/open` path as the
  accepted root window factory; no materialized 1.7/4.5GB learned cache or fake
  reduced descriptors. Root performs complete source-file scans before/after;
  original window open independently authenticates the full weight shard.
- One128MiB host ResourceArbiter domain per sequential source:64MiB admitted
  source metadata,4MiB binding metadata,2MiB harness control/reference envelope.
  Existing per-value64KiB Output+960KiB Scratch retain original source issuer.
  Each case retains eight backings (input+seven outputs),16MiB for both cases.
- Actual pool exhausted through same-source workspace leases; execution must
  refuse and release partial private work. Pre-cancel and wrong GDN layer also
  refuse. Mid-cancel is triggered by actual >=2 Output backings plus a real
  TransferStaging charge. Missing this short physical boundary fails the case;
  it cannot be reported as passed merely because a timer elapsed.
- Both outputs survive dropping callers/operators/source, are compared again,
  then release all source and backing owners. Final reservations/bytes0 and
  no fence required. No SessionAuthority/StateRoot is allocated or published.
- Pool accounting covers named payloads/explicit metadata. Generic allocator,
  executable/JSON runtime bookkeeping, OS caches and RSS do not become physical
  free-RAM admission claims. No event-complete peak measurement is claimed.
- Native launcher uses the exact accepted root resolver WNOWAIT/owned SID/PGID
  cleanup functions. Independent TERM/grace/KILL stages occur before any reap;
  final reap, group absence, bounded pipe drain and closure required. Both
  helper invocations are serialized and use the same pinned executable FD.

## ROOT execution after independent source review

All commands are prepared, **unexecuted by author**. Keep old accepted archives,
source observations and prior attention proposal bytes immutable.

```sh
cd /home/pvl/spaces/abird/src/zerogravity
RUN="$PWD/.agents/runs/model-execution-original-attention-replay-20261020/followup-v2"
# Source-free parser/transport fixtures; no trained qualification.
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$RUN" -p test_contracts.py -v
# Genuine existing archive transport only; fresh directory directly under RUN.
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/prepare_inputs.py" --new-output "$RUN/root-inputs-1"
# Root records the actual Cargo argv, source freeze and fresh binary SHA.
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --offline --release --manifest-path "$RUN/Cargo.toml"
# Actual helper/inputs SHA values are obtained by ROOT, never placeholder pins.
python3 scripts/native-gpu.py -- python3 "$RUN/run_replay.py" \
  --binary ROOT_FRESH_BINARY --binary-sha256 ROOT_ACTUAL_SHA \
  --inputs "$RUN/root-inputs-1/inputs.json" --inputs-sha256 ROOT_INPUTS_SHA \
  --new-output "$RUN/root-replay-1"
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/validate_replay.py" --output "$RUN/root-replay-1"
```

Root must retain actual build stdout/argv and source/binary attestation; the
launcher cannot prove compilation provenance merely from a supplied SHA. Its
freeze binds the exact current MAIN source bytes and API dependencies. Cargo
lock normalization follows the accepted root helper policy; inspect any delta.
No dependency fetch or SDK/model acquisition is needed. Root schedules complete
source scans outside benchmarks and competing model/GPU phases.

Possible scope after all checks: **original CPU private attention fragment only**.
RoPE,KV/cache append,BMM/softmax/gating integration,initial/cached GDN coefficients
and MKL scheduling,full-HF session/checkpoint/logits,27B,GPU/native cuBLAS and
matched serving performance remain unqualified.

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/followup-v2/frozen/TODO.md

Original UTF-8 bytes: 944; SHA256 `cac88c9601a498eb130e7ad0681b9c72a66e7e3eaa3d567e1f02c2c0f68dd004`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive V2 working checklist

- Parent: existing model-execution plan; owner batch_design_review.
- Owned scope: fresh followup-v2 only; original12/frozen/failure preserved.
- [x] Correct metadata prior_checks: actual49-record list, strict allbit_exact and exact full label cover; retain report scalar98/49.
- [x] Add genuine metadata-only listfixture plus three regressions; original seven fixtures unchanged.
- [x] Inspect remaining genuine small metadata schema/types without tensor payload/math/runtime.
- [x] Copy all other Rust/Cargo/bindings/export/validator/owned cleanup bytes unchanged.
- [ ] Independent ROOT/peer source review and10 sourcefree fixtures.
- [ ] ROOT fresh authenticated export, build/binary/source binding,47,104-word replay and independent byte validation.

Original actual root-inputs-1 export failure remains unqualified/preserved. No
author build/test/operator/model/GPU/container run or tensor/provider hash.

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/followup-v3/README.md

Original UTF-8 bytes: 8126; SHA256 `123266c1b7ba0e0061a30aed422c66b81de52575d9877b95047ecda7ef52309b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only authentic CPU attention fragment replay — additive V2

- Parent: existing model-execution plan. Source preparation only; root executes.
- Current MAIN APIs only: sealed original `VerifiedTensorSource` →
  `OriginalWindowOperators::bind/execute_attention_projection_norm_slice` plus
  explicit `AttentionSliceInput`. Same shared HostMath/ForwardPlan operations;
  no alternate mathematical implementation, token loop, state or input node.
- Sources: original .8B@2fc063… and 2B@15852…; raw original files/headers/source
  identity, BF16 matrices/learned QK gamma and epsilon1e-6 stay intact. All320
  text-role descriptors bind. Only `source-36-f32-restored` CPU AVX2 semantics;
  official all-BF16 loader and CUDA/vLLM mixed cast profile remain distinct.

## V2 correction and preserved evidence

- Original12 prepared files/frozen copies and root-inputs-1 failure stay unchanged.
  Root found the original guard incorrectly compared a49-record metadata list
  to scalar49. The parent report counts remain scalar98/49; per-step metadata
  now requires the genuine full49 records, every bit_exact strictly true and
  the complete original ordered fullV/cache label namespace exactly once.
- No input/output/source/schema/math/shape/tolerance change. Rust/Cargo/bindings,
  exporter,validator and owned-lifetime implementations are copied byte-exact.
- Added metadata-only authentic49-record fixture and three regressions: genuine
  list accepted; scalar/incomplete list refused; onebad/nonboolean/missing flag
  or duplicate label refused. Root executes all10 fixtures; author AST only.
- Reinspected remaining actual small .8B/2B metadata types: report scalarcounts,
  replay_checks list, per-step tensor maps, public-call arguments,2B restored
  source names36, profile/step/token fields and both observed lifecycle schemas.
  No further metadata type mismatch found. No tensor payload read/hash or guard
  execution by author. Fresh ROOT extract/replay directories are required.

## Independent reference and exact scope

- Two genuine prior teacher inputs,11 then353, at first FullAttention layer3.
  Pure fragments receive already-observed authentic hidden inputs; they do not
  produce those inputs or replay the preceding model/cache computation.
- .8B accepted root-probe-2: immutable whole archive pins plus actual call001
  original public schemas, full source parameters, input byte digests, split
  query frame/gate-flat frame and final type_as BF16 norm outputs. Parent all98
  logits/cache checks,12attention+2rotary and2802public checks stay mandatory.
- 2B accepted root-pair-1: e45e1903… paired validation/f969e1a4… observed manifest,
  both plain/observed196raw tensors/four fullV/168domains/60prior intermediates
  exact to accepted reference; raw source public-hook stage/layer/strides.
  No additional Torch/operator/model/container or D2H operation is required.
- Exporter copies only original raw BF16 words: four inputs plus seven expected
  outputs per case =106,496B. Expected tensors are never reconstructed from
  projections, rounded, zero-normalized or fitted. Independent validator
  rechecks exported words against the original authenticated archive spans.
- Rust compares all47,104 output words (23,552/source) as raw BF16 widened bits,
  including signed zero. Seven outputs: query/gate projection,K projection,V
  projection,query,gate,normalized query,normalized key. Full lengths required.
  Actual raw BF16 output files permit independent stdlib byte comparison.

## Real ownership and negative gates

- Same original full-shard/index FD/`prepare_verified_windows/open` path as the
  accepted root window factory; no materialized 1.7/4.5GB learned cache or fake
  reduced descriptors. Root performs complete source-file scans before/after;
  original window open independently authenticates the full weight shard.
- One128MiB host ResourceArbiter domain per sequential source:64MiB admitted
  source metadata,4MiB binding metadata,2MiB harness control/reference envelope.
  Existing per-value64KiB Output+960KiB Scratch retain original source issuer.
  Each case retains eight backings (input+seven outputs),16MiB for both cases.
- Actual pool exhausted through same-source workspace leases; execution must
  refuse and release partial private work. Pre-cancel and wrong GDN layer also
  refuse. Mid-cancel is triggered by actual >=2 Output backings plus a real
  TransferStaging charge. Missing this short physical boundary fails the case;
  it cannot be reported as passed merely because a timer elapsed.
- Both outputs survive dropping callers/operators/source, are compared again,
  then release all source and backing owners. Final reservations/bytes0 and
  no fence required. No SessionAuthority/StateRoot is allocated or published.
- Pool accounting covers named payloads/explicit metadata. Generic allocator,
  executable/JSON runtime bookkeeping, OS caches and RSS do not become physical
  free-RAM admission claims. No event-complete peak measurement is claimed.
- Native launcher uses the exact accepted root resolver WNOWAIT/owned SID/PGID
  cleanup functions. Independent TERM/grace/KILL stages occur before any reap;
  final reap, group absence, bounded pipe drain and closure required. Both
  helper invocations are serialized and use the same pinned executable FD.

## ROOT execution after independent source review

All commands are prepared, **unexecuted by author**. Keep old accepted archives,
source observations and prior attention proposal bytes immutable.

```sh
cd /home/pvl/spaces/abird/src/zerogravity
RUN="$PWD/.agents/runs/model-execution-original-attention-replay-20261020/followup-v3"
# Source-free parser/transport fixtures; no trained qualification.
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$RUN" -p test_contracts.py -v
# Genuine existing archive transport only; fresh directory directly under RUN.
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/prepare_inputs.py" --new-output "$RUN/root-inputs-1"
# Root records the actual Cargo argv, source freeze and fresh binary SHA.
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --offline --release --manifest-path "$RUN/Cargo.toml"
# Actual helper/inputs SHA values are obtained by ROOT, never placeholder pins.
python3 scripts/native-gpu.py -- python3 "$RUN/run_replay.py" \
  --binary ROOT_FRESH_BINARY --binary-sha256 ROOT_ACTUAL_SHA \
  --inputs "$RUN/root-inputs-1/inputs.json" --inputs-sha256 ROOT_INPUTS_SHA \
  --new-output "$RUN/root-replay-1"
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/validate_replay.py" --output "$RUN/root-replay-1"
```

Root must retain actual build stdout/argv and source/binary attestation; the
launcher cannot prove compilation provenance merely from a supplied SHA. Its
freeze binds the exact current MAIN source bytes and API dependencies. Cargo
lock normalization follows the accepted root helper policy; inspect any delta.
No dependency fetch or SDK/model acquisition is needed. Root schedules complete
source scans outside benchmarks and competing model/GPU phases.

Possible scope after all checks: **original CPU private attention fragment only**.
RoPE,KV/cache append,BMM/softmax/gating integration,initial/cached GDN coefficients
and MKL scheduling,full-HF session/checkpoint/logits,27B,GPU/native cuBLAS and
matched serving performance remain unqualified.

## ROOT additive V3 accounting correction

- V2 root-replay-2 refused before numerical operators: steady charge omitted
  QwenOriginalWindowText's4MiB metadata. OriginalWindowOperators retains a
  separate4MiB metadata charge. Exact steady total is64+8+2MiB/4reservations.
- Assert exact Scratch72MiB,Other2MiB,Output/Staging0; retain full final0 and
  every pressure/cancellation/lifetime/numerical guard. Add actual steady ledger
  to report. Both4MiB binding charges drop with callers; retained output/source
  charge remains64+2+16MiB. No MAIN/math/input/reference changes.
- OriginalV2 sources/frozen binary/failed invocation/report/log remain intact.
- Runtime status: ROOT must rebuild and execute; correction is not qualification.

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/followup-v3/TODO.md

Original UTF-8 bytes: 1054; SHA256 `88dfea41df0e48d628a371b7f2320196fcf530efa9d57ae98898401f7e64a4b9`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive V2 working checklist

- Parent: existing model-execution plan; owner batch_design_review.
- Owned scope: fresh followup-v2 only; original12/frozen/failure preserved.
- [x] Correct metadata prior_checks: actual49-record list, strict allbit_exact and exact full label cover; retain report scalar98/49.
- [x] Add genuine metadata-only listfixture plus three regressions; original seven fixtures unchanged.
- [x] Inspect remaining genuine small metadata schema/types without tensor payload/math/runtime.
- [x] Copy all other Rust/Cargo/bindings/export/validator/owned cleanup bytes unchanged.
- [ ] Independent ROOT/peer source review and10 sourcefree fixtures.
- [ ] ROOT fresh authenticated export, build/binary/source binding,47,104-word replay and independent byte validation.

Original actual root-inputs-1 export failure remains unqualified/preserved. No
author build/test/operator/model/GPU/container run or tensor/provider hash.

- [ ] ROOT V3: rebuild exact accounting correction, rerun both trained fragments and independent validation.

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/followup-v3/frozen-prepared/README.md

Original UTF-8 bytes: 8126; SHA256 `123266c1b7ba0e0061a30aed422c66b81de52575d9877b95047ecda7ef52309b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only authentic CPU attention fragment replay — additive V2

- Parent: existing model-execution plan. Source preparation only; root executes.
- Current MAIN APIs only: sealed original `VerifiedTensorSource` →
  `OriginalWindowOperators::bind/execute_attention_projection_norm_slice` plus
  explicit `AttentionSliceInput`. Same shared HostMath/ForwardPlan operations;
  no alternate mathematical implementation, token loop, state or input node.
- Sources: original .8B@2fc063… and 2B@15852…; raw original files/headers/source
  identity, BF16 matrices/learned QK gamma and epsilon1e-6 stay intact. All320
  text-role descriptors bind. Only `source-36-f32-restored` CPU AVX2 semantics;
  official all-BF16 loader and CUDA/vLLM mixed cast profile remain distinct.

## V2 correction and preserved evidence

- Original12 prepared files/frozen copies and root-inputs-1 failure stay unchanged.
  Root found the original guard incorrectly compared a49-record metadata list
  to scalar49. The parent report counts remain scalar98/49; per-step metadata
  now requires the genuine full49 records, every bit_exact strictly true and
  the complete original ordered fullV/cache label namespace exactly once.
- No input/output/source/schema/math/shape/tolerance change. Rust/Cargo/bindings,
  exporter,validator and owned-lifetime implementations are copied byte-exact.
- Added metadata-only authentic49-record fixture and three regressions: genuine
  list accepted; scalar/incomplete list refused; onebad/nonboolean/missing flag
  or duplicate label refused. Root executes all10 fixtures; author AST only.
- Reinspected remaining actual small .8B/2B metadata types: report scalarcounts,
  replay_checks list, per-step tensor maps, public-call arguments,2B restored
  source names36, profile/step/token fields and both observed lifecycle schemas.
  No further metadata type mismatch found. No tensor payload read/hash or guard
  execution by author. Fresh ROOT extract/replay directories are required.

## Independent reference and exact scope

- Two genuine prior teacher inputs,11 then353, at first FullAttention layer3.
  Pure fragments receive already-observed authentic hidden inputs; they do not
  produce those inputs or replay the preceding model/cache computation.
- .8B accepted root-probe-2: immutable whole archive pins plus actual call001
  original public schemas, full source parameters, input byte digests, split
  query frame/gate-flat frame and final type_as BF16 norm outputs. Parent all98
  logits/cache checks,12attention+2rotary and2802public checks stay mandatory.
- 2B accepted root-pair-1: e45e1903… paired validation/f969e1a4… observed manifest,
  both plain/observed196raw tensors/four fullV/168domains/60prior intermediates
  exact to accepted reference; raw source public-hook stage/layer/strides.
  No additional Torch/operator/model/container or D2H operation is required.
- Exporter copies only original raw BF16 words: four inputs plus seven expected
  outputs per case =106,496B. Expected tensors are never reconstructed from
  projections, rounded, zero-normalized or fitted. Independent validator
  rechecks exported words against the original authenticated archive spans.
- Rust compares all47,104 output words (23,552/source) as raw BF16 widened bits,
  including signed zero. Seven outputs: query/gate projection,K projection,V
  projection,query,gate,normalized query,normalized key. Full lengths required.
  Actual raw BF16 output files permit independent stdlib byte comparison.

## Real ownership and negative gates

- Same original full-shard/index FD/`prepare_verified_windows/open` path as the
  accepted root window factory; no materialized 1.7/4.5GB learned cache or fake
  reduced descriptors. Root performs complete source-file scans before/after;
  original window open independently authenticates the full weight shard.
- One128MiB host ResourceArbiter domain per sequential source:64MiB admitted
  source metadata,4MiB binding metadata,2MiB harness control/reference envelope.
  Existing per-value64KiB Output+960KiB Scratch retain original source issuer.
  Each case retains eight backings (input+seven outputs),16MiB for both cases.
- Actual pool exhausted through same-source workspace leases; execution must
  refuse and release partial private work. Pre-cancel and wrong GDN layer also
  refuse. Mid-cancel is triggered by actual >=2 Output backings plus a real
  TransferStaging charge. Missing this short physical boundary fails the case;
  it cannot be reported as passed merely because a timer elapsed.
- Both outputs survive dropping callers/operators/source, are compared again,
  then release all source and backing owners. Final reservations/bytes0 and
  no fence required. No SessionAuthority/StateRoot is allocated or published.
- Pool accounting covers named payloads/explicit metadata. Generic allocator,
  executable/JSON runtime bookkeeping, OS caches and RSS do not become physical
  free-RAM admission claims. No event-complete peak measurement is claimed.
- Native launcher uses the exact accepted root resolver WNOWAIT/owned SID/PGID
  cleanup functions. Independent TERM/grace/KILL stages occur before any reap;
  final reap, group absence, bounded pipe drain and closure required. Both
  helper invocations are serialized and use the same pinned executable FD.

## ROOT execution after independent source review

All commands are prepared, **unexecuted by author**. Keep old accepted archives,
source observations and prior attention proposal bytes immutable.

```sh
cd /home/pvl/spaces/abird/src/zerogravity
RUN="$PWD/.agents/runs/model-execution-original-attention-replay-20261020/followup-v3"
# Source-free parser/transport fixtures; no trained qualification.
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$RUN" -p test_contracts.py -v
# Genuine existing archive transport only; fresh directory directly under RUN.
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/prepare_inputs.py" --new-output "$RUN/root-inputs-1"
# Root records the actual Cargo argv, source freeze and fresh binary SHA.
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --offline --release --manifest-path "$RUN/Cargo.toml"
# Actual helper/inputs SHA values are obtained by ROOT, never placeholder pins.
python3 scripts/native-gpu.py -- python3 "$RUN/run_replay.py" \
  --binary ROOT_FRESH_BINARY --binary-sha256 ROOT_ACTUAL_SHA \
  --inputs "$RUN/root-inputs-1/inputs.json" --inputs-sha256 ROOT_INPUTS_SHA \
  --new-output "$RUN/root-replay-1"
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/validate_replay.py" --output "$RUN/root-replay-1"
```

Root must retain actual build stdout/argv and source/binary attestation; the
launcher cannot prove compilation provenance merely from a supplied SHA. Its
freeze binds the exact current MAIN source bytes and API dependencies. Cargo
lock normalization follows the accepted root helper policy; inspect any delta.
No dependency fetch or SDK/model acquisition is needed. Root schedules complete
source scans outside benchmarks and competing model/GPU phases.

Possible scope after all checks: **original CPU private attention fragment only**.
RoPE,KV/cache append,BMM/softmax/gating integration,initial/cached GDN coefficients
and MKL scheduling,full-HF session/checkpoint/logits,27B,GPU/native cuBLAS and
matched serving performance remain unqualified.

## ROOT additive V3 accounting correction

- V2 root-replay-2 refused before numerical operators: steady charge omitted
  QwenOriginalWindowText's4MiB metadata. OriginalWindowOperators retains a
  separate4MiB metadata charge. Exact steady total is64+8+2MiB/4reservations.
- Assert exact Scratch72MiB,Other2MiB,Output/Staging0; retain full final0 and
  every pressure/cancellation/lifetime/numerical guard. Add actual steady ledger
  to report. Both4MiB binding charges drop with callers; retained output/source
  charge remains64+2+16MiB. No MAIN/math/input/reference changes.
- OriginalV2 sources/frozen binary/failed invocation/report/log remain intact.
- Runtime status: ROOT must rebuild and execute; correction is not qualification.

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/followup-v3/frozen-prepared/TODO.md

Original UTF-8 bytes: 1054; SHA256 `88dfea41df0e48d628a371b7f2320196fcf530efa9d57ae98898401f7e64a4b9`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive V2 working checklist

- Parent: existing model-execution plan; owner batch_design_review.
- Owned scope: fresh followup-v2 only; original12/frozen/failure preserved.
- [x] Correct metadata prior_checks: actual49-record list, strict allbit_exact and exact full label cover; retain report scalar98/49.
- [x] Add genuine metadata-only listfixture plus three regressions; original seven fixtures unchanged.
- [x] Inspect remaining genuine small metadata schema/types without tensor payload/math/runtime.
- [x] Copy all other Rust/Cargo/bindings/export/validator/owned cleanup bytes unchanged.
- [ ] Independent ROOT/peer source review and10 sourcefree fixtures.
- [ ] ROOT fresh authenticated export, build/binary/source binding,47,104-word replay and independent byte validation.

Original actual root-inputs-1 export failure remains unqualified/preserved. No
author build/test/operator/model/GPU/container run or tensor/provider hash.

- [ ] ROOT V3: rebuild exact accounting correction, rerun both trained fragments and independent validation.

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/frozen/README.md

Original UTF-8 bytes: 6132; SHA256 `432d9e563124dc6b6a86f2e96b700c48e3440cbb735cd2b6bf46b2e803127940`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only authentic CPU attention fragment replay

- Parent: existing model-execution plan. Source preparation only; root executes.
- Current MAIN APIs only: sealed original `VerifiedTensorSource` →
  `OriginalWindowOperators::bind/execute_attention_projection_norm_slice` plus
  explicit `AttentionSliceInput`. Same shared HostMath/ForwardPlan operations;
  no alternate mathematical implementation, token loop, state or input node.
- Sources: original .8B@2fc063… and 2B@15852…; raw original files/headers/source
  identity, BF16 matrices/learned QK gamma and epsilon1e-6 stay intact. All320
  text-role descriptors bind. Only `source-36-f32-restored` CPU AVX2 semantics;
  official all-BF16 loader and CUDA/vLLM mixed cast profile remain distinct.

## Independent reference and exact scope

- Two genuine prior teacher inputs,11 then353, at first FullAttention layer3.
  Pure fragments receive already-observed authentic hidden inputs; they do not
  produce those inputs or replay the preceding model/cache computation.
- .8B accepted root-probe-2: immutable whole archive pins plus actual call001
  original public schemas, full source parameters, input byte digests, split
  query frame/gate-flat frame and final type_as BF16 norm outputs. Parent all98
  logits/cache checks,12attention+2rotary and2802public checks stay mandatory.
- 2B accepted root-pair-1: e45e1903… paired validation/f969e1a4… observed manifest,
  both plain/observed196raw tensors/four fullV/168domains/60prior intermediates
  exact to accepted reference; raw source public-hook stage/layer/strides.
  No additional Torch/operator/model/container or D2H operation is required.
- Exporter copies only original raw BF16 words: four inputs plus seven expected
  outputs per case =106,496B. Expected tensors are never reconstructed from
  projections, rounded, zero-normalized or fitted. Independent validator
  rechecks exported words against the original authenticated archive spans.
- Rust compares all47,104 output words (23,552/source) as raw BF16 widened bits,
  including signed zero. Seven outputs: query/gate projection,K projection,V
  projection,query,gate,normalized query,normalized key. Full lengths required.
  Actual raw BF16 output files permit independent stdlib byte comparison.

## Real ownership and negative gates

- Same original full-shard/index FD/`prepare_verified_windows/open` path as the
  accepted root window factory; no materialized 1.7/4.5GB learned cache or fake
  reduced descriptors. Root performs complete source-file scans before/after;
  original window open independently authenticates the full weight shard.
- One128MiB host ResourceArbiter domain per sequential source:64MiB admitted
  source metadata,4MiB binding metadata,2MiB harness control/reference envelope.
  Existing per-value64KiB Output+960KiB Scratch retain original source issuer.
  Each case retains eight backings (input+seven outputs),16MiB for both cases.
- Actual pool exhausted through same-source workspace leases; execution must
  refuse and release partial private work. Pre-cancel and wrong GDN layer also
  refuse. Mid-cancel is triggered by actual >=2 Output backings plus a real
  TransferStaging charge. Missing this short physical boundary fails the case;
  it cannot be reported as passed merely because a timer elapsed.
- Both outputs survive dropping callers/operators/source, are compared again,
  then release all source and backing owners. Final reservations/bytes0 and
  no fence required. No SessionAuthority/StateRoot is allocated or published.
- Pool accounting covers named payloads/explicit metadata. Generic allocator,
  executable/JSON runtime bookkeeping, OS caches and RSS do not become physical
  free-RAM admission claims. No event-complete peak measurement is claimed.
- Native launcher uses the exact accepted root resolver WNOWAIT/owned SID/PGID
  cleanup functions. Independent TERM/grace/KILL stages occur before any reap;
  final reap, group absence, bounded pipe drain and closure required. Both
  helper invocations are serialized and use the same pinned executable FD.

## ROOT execution after independent source review

All commands are prepared, **unexecuted by author**. Keep old accepted archives,
source observations and prior attention proposal bytes immutable.

```sh
cd /home/pvl/spaces/abird/src/zerogravity
RUN="$PWD/.agents/runs/model-execution-original-attention-replay-20261020"
# Source-free parser/transport fixtures; no trained qualification.
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$RUN" -p test_contracts.py -v
# Genuine existing archive transport only; fresh directory directly under RUN.
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/prepare_inputs.py" --new-output "$RUN/root-inputs-1"
# Root records the actual Cargo argv, source freeze and fresh binary SHA.
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --offline --release --manifest-path "$RUN/Cargo.toml"
# Actual helper/inputs SHA values are obtained by ROOT, never placeholder pins.
python3 scripts/native-gpu.py -- python3 "$RUN/run_replay.py" \
  --binary ROOT_FRESH_BINARY --binary-sha256 ROOT_ACTUAL_SHA \
  --inputs "$RUN/root-inputs-1/inputs.json" --inputs-sha256 ROOT_INPUTS_SHA \
  --new-output "$RUN/root-replay-1"
PYTHONDONTWRITEBYTECODE=1 python3 "$RUN/validate_replay.py" --output "$RUN/root-replay-1"
```

Root must retain actual build stdout/argv and source/binary attestation; the
launcher cannot prove compilation provenance merely from a supplied SHA. Its
freeze binds the exact current MAIN source bytes and API dependencies. Cargo
lock normalization follows the accepted root helper policy; inspect any delta.
No dependency fetch or SDK/model acquisition is needed. Root schedules complete
source scans outside benchmarks and competing model/GPU phases.

Possible scope after all checks: **original CPU private attention fragment only**.
RoPE,KV/cache append,BMM/softmax/gating integration,initial/cached GDN coefficients
and MKL scheduling,full-HF session/checkpoint/logits,27B,GPU/native cuBLAS and
matched serving performance remain unqualified.

````

## Source: .agents/runs/model-execution-original-attention-replay-20261020/frozen/TODO.md

Original UTF-8 bytes: 985; SHA256 `ace505a350cb48c671c6dc4ed3827b45f8a543b1c617e9d9b66639bb8a64577d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

- Parent: existing model-execution plan; owner batch_design_review.
- Owned scope: this new ignored run only. MAIN source/shared plans unchanged.
- [x] Review narrow current MAIN attention compile/shape fixes; no blocker.
- [x] Prepare same-engine authentic .8B/2B layer3 two-teacher seven-output harness.
- [x] Keep genuine public source/schema/profile/archive and parent transparency binding.
- [x] Prepare real same-source pool pressure,cancel,retained owner/final0 checks.
- [x] Reuse exact accepted owned-native lifecycle functions; preserve original.
- [ ] Independent source review,source-free fixtures,actual archive export.
- [ ] ROOT build/current source+fresh binary binding and exact47,104-word replay.
- [ ] ROOT strict/full phase checks and acceptance; no implementation promotion here.

Author: AST/source inspection only. No tests,compilation,model/operator/GPU or
container execution; no large raw/model/DSO read/hash. No physical result claimed.

````

## Source: .agents/runs/model-execution-original-attention-slice-20261020/README.md

Original UTF-8 bytes: 7651; SHA256 `74ced5fd8a23bfeed4413e846d4f0bea3e70b51ba6e0191a971be81832843c8a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original attention projections/QK ordinary RMS — source-only proposal

- Parent: existing model-execution/serving plans; no new plan/tracker.
- Base: MAIN HEAD f78892957f09a84838246aede4fef2be040a0d18 PLUS its preexisting merged/uncompiled15-path source-window resolver overlay. base-src bytes/digests define the exact baseline; that overlay is excluded from this delta.
- Scope: proposed copies + NEW_DELTA.patch only; tracked MAIN/root README untouched.
- Status: SOURCE_ONLY, RUSTFMT_PARSED, UNCOMPILED, UNEXECUTED. No model/profile/session eligibility.

## Shared engine contract

- Caller supplies an existing same-source/same-builder typed `PlanValue` to `LoadedQwenModel::append_attention_projection_norm_slice` or the same method on `OriginalWindowOperators`.
- `forward::attention_projection_norm_slice` appends only existing3 TypedProject,2 View,2 Contiguous,2 TypedOrdinaryRms operations. No external activation/constant node, second token loop, state domain, frontier advance, RoPE, cache or logits publication.
- Actual `FullAttention` layer required; fixed head width256; hidden256(synthetic),1024,2048,5120; GQA divisibility; full query/gate rows≤16384. Exact original namespace, source identity, BF16 raw OutputInput matrix axes, complete shapes and QueryNorm/KeyNorm descriptors checked before importing/recording. Epsilon1e-6. GDN gamma is refused even if width256.
- Query source rows retain `[head, query-or-gate, width]`; typed views have offsets0/256,stride512. Explicit contiguous copies preserve every BF16 bit.
- Ordinary RMS reuses accepted unchanged CPU primitive: borrowed raw learned BF16/F32 weight; runtime F32(1+weight), source reducer and final BF16. No learned cache mutation/new reducer/rounding.
- `AttentionSliceInput::from_bf16_bytes` is explicit qualifier input transport, never proof of a model producer. Caller must authenticate archive/source/profile. Bound source/profile, little-endian finite BF16, exact full-hidden byte length; no F32 rounding/canonicalization. Signed zero/subnormal bits widen unchanged.
- Window path reserves original same-issuer operator workspace BEFORE each private copy/math; retains original source/output/scratch through borrowed output. Input+7 result backings reserve8MiB (8×64KiB Output +8×960KiB scratch), plus preexisting source/metadata and genuine separately admitted source windows. No claimed SessionAuthority/physical GPU admission.
- Full-HF execution/session/checkpoint, legacy HostMath HF operations and unaware physical-provider refusals remain closed. Existing GGUF operations/math unchanged.

## Proposed source/tests

Eight paths under `proposed-src/` are frozen exact; six preexisting base copies under `base-src/` support byte comparison. Only those eight paths belong to NEW_DELTA.patch. Test fixture refactor preserves old default bytes and adds a synthetic full-header head256 source with distinguishable query/gate row labels.

Five new focused tests (NOT RUN):

- existing owned vs sealed-window outputs; exact row labels/raw learned norm unchanged;8MiB retained across caller/source drop; final physical reservation0;
- caller PlanBuilder composition/foreign-builder refusal;10nodes with synthetic TypedEmbed explicitly transport-only; no state/frontier;
- exact raw BF16 signed-zero/subnormal transport; source/profile/finite/length refusal before admission;
- same-width GDN gamma, unrelated matrix, forged descriptor, wrong per-head geometry, unaware provider and independent source issuer refusals;
- cancellation before import/after admitted input; genuine ResourceArbiter exhaustion/refusal; private owner release/final0.

ROOT only, after review/merge with fresh file modification times:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- cargo test -p zerogravity-model-qwen35 --features cpu,original-hf-binding operator_slice -- --test-threads=1
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- cargo clippy -p zerogravity-model-qwen35 --all-targets --features cpu,original-hf-binding -- -D warnings
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh
```

Also compile default +cpu without original-hf-binding feature; no acceptance based on syntax/source-free tests.

## Authentic replay gates

Existing accepted .8B observation, no new model run needed for inputs:

`/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-attention-observation-20261020/root-probe-2`

- step0/attention.safetensors2697148B SHA bdc827153bc1a7c667146b2831bb22840b7ad626ecd5a653888c95b33150e34a
- step1/attention.safetensors3089820B SHA fdf634861d2bb8a6c8aa0e41385c25567e2cb33de2077aa681f8e498306d7928
- First actual attention call `call001` is source layer3. Hidden: call001.hidden. Query/gate projection:op000.result; raw query:op002.result.0; actual gate:op003.result; key:op016.result; value:op029.result; normalized query:op014.result; normalized key:op027.result. These indices must be bound to actual recorded public operator labels/schema and tensor metadata, not guessed in a runtime parser. Other six-layer calls similarly have their own actual metadata.
- Every private output must compare raw BF16 words to authenticated source36-restored original observations. Current proposal has NOT replayed any trained output. Preserve98 accepted complete-logit/raw-cache checks. Do not infer producer/frontier/state support from a pure fragment.

Existing .2B reference has15 genuine GDN0 stages +all49 raw tensors/fullV per profile/teacher, but no genuine attention hidden/projection/norm input archive. `observation-prep/attention_observation_hook.py` supplies a PREPARED hook only (AST parsed, not imported/run), source-pinned original public pre/post hooks plus actual return-frame gate. It delegates observation from the existing profiler rather than replacing it and never returns replacement tensors. Integrate only into a fresh ROOT-owned accepted two-profile reference run, preserving ALL196 prior raw-reference tensor comparisons/complete semantic domains/source and parameter audits. All hooks must be removed in a finally; final hook removal failures stay failures. Outer launcher MUST use Spine's reviewed owned_container_v4 lifecycle (freeze248cde26); no new launcher/opaque lifetime is invented here. A fresh paired plain/observed capture and independent old-reference byte comparator still need ROOT preparation/review before execution. Hook alone is not an executable acceptance harness.

## Remaining full-HF blockers

- Typed pure prefix exists; complete forward/session guard deliberately refuses original HF.
- Initial GDN chunk64 uses padded source schedule, query reciprocal MULTIPLY, F64 cumulative sum→F32, six public BMM +two solve families; cached uses direct DIV and materialized ordered state arithmetic. No zero-state shortcut.
- Cached closed helper still accepts precomputed per-head decay; source Tensor.exp/MKL VML route and admitted public BMM/STRSM provider ownership remain separate gates. SLEEF SiLU/softplus do not qualify Tensor.exp.
- Eager attention RoPE and BF16 BMM need explicit typed contracts/cast boundaries. Valid1/valid2 may select different CPU dispatch; CPU64 projection reducer is not a BF16 BMM qualification. Accepted closed F32 softmax does not remove surrounding BF16 QK/scale/mask/probability/AV/gating rounds.
- Raw BF16 KV token/head/width and F32 key-major GDN state metadata do not enable payload publication. Full .8B/.2B/27B all-state/full-logit Rust parity remains unproven here; CUDA/FLA/profile/common-vLLM math/performance unclaimed.

````

## Source: .agents/runs/model-execution-original-attention-slice-20261020/frozen/README.md

Original UTF-8 bytes: 7651; SHA256 `74ced5fd8a23bfeed4413e846d4f0bea3e70b51ba6e0191a971be81832843c8a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original attention projections/QK ordinary RMS — source-only proposal

- Parent: existing model-execution/serving plans; no new plan/tracker.
- Base: MAIN HEAD f78892957f09a84838246aede4fef2be040a0d18 PLUS its preexisting merged/uncompiled15-path source-window resolver overlay. base-src bytes/digests define the exact baseline; that overlay is excluded from this delta.
- Scope: proposed copies + NEW_DELTA.patch only; tracked MAIN/root README untouched.
- Status: SOURCE_ONLY, RUSTFMT_PARSED, UNCOMPILED, UNEXECUTED. No model/profile/session eligibility.

## Shared engine contract

- Caller supplies an existing same-source/same-builder typed `PlanValue` to `LoadedQwenModel::append_attention_projection_norm_slice` or the same method on `OriginalWindowOperators`.
- `forward::attention_projection_norm_slice` appends only existing3 TypedProject,2 View,2 Contiguous,2 TypedOrdinaryRms operations. No external activation/constant node, second token loop, state domain, frontier advance, RoPE, cache or logits publication.
- Actual `FullAttention` layer required; fixed head width256; hidden256(synthetic),1024,2048,5120; GQA divisibility; full query/gate rows≤16384. Exact original namespace, source identity, BF16 raw OutputInput matrix axes, complete shapes and QueryNorm/KeyNorm descriptors checked before importing/recording. Epsilon1e-6. GDN gamma is refused even if width256.
- Query source rows retain `[head, query-or-gate, width]`; typed views have offsets0/256,stride512. Explicit contiguous copies preserve every BF16 bit.
- Ordinary RMS reuses accepted unchanged CPU primitive: borrowed raw learned BF16/F32 weight; runtime F32(1+weight), source reducer and final BF16. No learned cache mutation/new reducer/rounding.
- `AttentionSliceInput::from_bf16_bytes` is explicit qualifier input transport, never proof of a model producer. Caller must authenticate archive/source/profile. Bound source/profile, little-endian finite BF16, exact full-hidden byte length; no F32 rounding/canonicalization. Signed zero/subnormal bits widen unchanged.
- Window path reserves original same-issuer operator workspace BEFORE each private copy/math; retains original source/output/scratch through borrowed output. Input+7 result backings reserve8MiB (8×64KiB Output +8×960KiB scratch), plus preexisting source/metadata and genuine separately admitted source windows. No claimed SessionAuthority/physical GPU admission.
- Full-HF execution/session/checkpoint, legacy HostMath HF operations and unaware physical-provider refusals remain closed. Existing GGUF operations/math unchanged.

## Proposed source/tests

Eight paths under `proposed-src/` are frozen exact; six preexisting base copies under `base-src/` support byte comparison. Only those eight paths belong to NEW_DELTA.patch. Test fixture refactor preserves old default bytes and adds a synthetic full-header head256 source with distinguishable query/gate row labels.

Five new focused tests (NOT RUN):

- existing owned vs sealed-window outputs; exact row labels/raw learned norm unchanged;8MiB retained across caller/source drop; final physical reservation0;
- caller PlanBuilder composition/foreign-builder refusal;10nodes with synthetic TypedEmbed explicitly transport-only; no state/frontier;
- exact raw BF16 signed-zero/subnormal transport; source/profile/finite/length refusal before admission;
- same-width GDN gamma, unrelated matrix, forged descriptor, wrong per-head geometry, unaware provider and independent source issuer refusals;
- cancellation before import/after admitted input; genuine ResourceArbiter exhaustion/refusal; private owner release/final0.

ROOT only, after review/merge with fresh file modification times:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- cargo test -p zerogravity-model-qwen35 --features cpu,original-hf-binding operator_slice -- --test-threads=1
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- cargo clippy -p zerogravity-model-qwen35 --all-targets --features cpu,original-hf-binding -- -D warnings
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh
```

Also compile default +cpu without original-hf-binding feature; no acceptance based on syntax/source-free tests.

## Authentic replay gates

Existing accepted .8B observation, no new model run needed for inputs:

`/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-attention-observation-20261020/root-probe-2`

- step0/attention.safetensors2697148B SHA bdc827153bc1a7c667146b2831bb22840b7ad626ecd5a653888c95b33150e34a
- step1/attention.safetensors3089820B SHA fdf634861d2bb8a6c8aa0e41385c25567e2cb33de2077aa681f8e498306d7928
- First actual attention call `call001` is source layer3. Hidden: call001.hidden. Query/gate projection:op000.result; raw query:op002.result.0; actual gate:op003.result; key:op016.result; value:op029.result; normalized query:op014.result; normalized key:op027.result. These indices must be bound to actual recorded public operator labels/schema and tensor metadata, not guessed in a runtime parser. Other six-layer calls similarly have their own actual metadata.
- Every private output must compare raw BF16 words to authenticated source36-restored original observations. Current proposal has NOT replayed any trained output. Preserve98 accepted complete-logit/raw-cache checks. Do not infer producer/frontier/state support from a pure fragment.

Existing .2B reference has15 genuine GDN0 stages +all49 raw tensors/fullV per profile/teacher, but no genuine attention hidden/projection/norm input archive. `observation-prep/attention_observation_hook.py` supplies a PREPARED hook only (AST parsed, not imported/run), source-pinned original public pre/post hooks plus actual return-frame gate. It delegates observation from the existing profiler rather than replacing it and never returns replacement tensors. Integrate only into a fresh ROOT-owned accepted two-profile reference run, preserving ALL196 prior raw-reference tensor comparisons/complete semantic domains/source and parameter audits. All hooks must be removed in a finally; final hook removal failures stay failures. Outer launcher MUST use Spine's reviewed owned_container_v4 lifecycle (freeze248cde26); no new launcher/opaque lifetime is invented here. A fresh paired plain/observed capture and independent old-reference byte comparator still need ROOT preparation/review before execution. Hook alone is not an executable acceptance harness.

## Remaining full-HF blockers

- Typed pure prefix exists; complete forward/session guard deliberately refuses original HF.
- Initial GDN chunk64 uses padded source schedule, query reciprocal MULTIPLY, F64 cumulative sum→F32, six public BMM +two solve families; cached uses direct DIV and materialized ordered state arithmetic. No zero-state shortcut.
- Cached closed helper still accepts precomputed per-head decay; source Tensor.exp/MKL VML route and admitted public BMM/STRSM provider ownership remain separate gates. SLEEF SiLU/softplus do not qualify Tensor.exp.
- Eager attention RoPE and BF16 BMM need explicit typed contracts/cast boundaries. Valid1/valid2 may select different CPU dispatch; CPU64 projection reducer is not a BF16 BMM qualification. Accepted closed F32 softmax does not remove surrounding BF16 QK/scale/mask/probability/AV/gating rounds.
- Raw BF16 KV token/head/width and F32 key-major GDN state metadata do not enable payload publication. Full .8B/.2B/27B all-state/full-logit Rust parity remains unproven here; CUDA/FLA/profile/common-vLLM math/performance unclaimed.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/EXP_BOUNDARY.md

Original UTF-8 bytes: 4517; SHA256 `4e169bddd45a40016647f6944c2c62718b811414c2569b6f96175b4c37a5dd8e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Remaining exact exponential boundary

Author: batch_design_review. Source/accepted receipt audit only.

## Actual qualified boundary

- MAIN native ABI run
  `.agents/runs/model-execution-pinned-native-provider-probe-20261020/root-actual-3`
  reports `PASS_BOUNDED_NATIVE_PUBLIC_ABI_ONLY`, four genuine vmsExp calls,
  six public CBLAS batch-GEMMs and32 STRSM calls. Earlier affinity failures are
  preserved; actual3 verifies four CPUs before loading/calling providers.
- Its `validation.physical_memory_admission` is `UNQUALIFIED`; the Rust report
  separately leaves native-internal memory/model/profile execution unqualified.
  Do not turn this successful diagnostic into an admitted production provider.
- Pinned ATen cf30153c… UnaryOpsKernel.cpp877 invokes the FP32 VML exp path;
  `ATen/cpu/vml.h` selects public vmsExp mode0x140102 (HA | FTZDAZ_OFF |
  ERRMODE_IGNORE). Exact public ABI and selected provider/runtime closure were
  replayed. No portable implementation source/internal exp schedule is proved.
- Private accepted SLEEF exp exists only inside declared SiLU/softplus/sigmoid
  output contracts. It is not a public F32 Tensor.exp replacement. Neither CPU
  libm, HA accuracy guarantees nor a finite recorded set prove equality.

## Source schedule which remains distinct

- A_log: original rank1 F32 source → `.float()` no-op → public exp F32 → unary
  negation. Runtime a projection BF16 → F32 +raw BF16 dt_bias → softplus F32;
  separate multiplication produces g F32. Beta independently stays BF16.
- Initial slow chunk pads one valid token to64; it applies F64-accumulating
  cumsum→F32 materialization, then different exp operands/layouts, including
  masked pairwise differences. This is not cached recurrence on zero state.
- Cached slow core separately executes exp(g) line483 on `[1,16]` contiguous.
  Initial line415 takes a noncontiguous head span. Do not flatten/change those
  iterator calls or infer one admitted span covers every layout and size.
- Current coefficient-preparation output deliberately excludes g/exp/decay;
  the legacy original diagnostic's host `.exp()` is not promoted by this work.

## Shortest legitimate continuation

1. **Exact public VML, optional per-operator provider.** Reuse the proven public
   signature/DSO/mode and one original Rust schedule. Before any cold load/call,
   establish admitted provider-owned mappings, constructors, worker stacks,
   scratch/pools and caller operands/output lifetime under the existing resource
   owner. Synchronous return can retire per-call buffers only when provider
   persistent allocations remain charged. Cancellation retains all owners until
   return, then discards private output. Never unload an in-flight provider.
2. **Close allocation rather than assume it.** Public exported i_malloc/i_free/
   i_calloc/i_realloc are a potential vendor allocator seam after exact public
   ABI validation. They do not automatically cover Torch DSO constructors,
   loader/OpenMP/runtime allocations. Source-prove and physically account for
   every actual route, alignment, realloc ownership/OOM and cold/concurrent
   exclusion before enabling. Cgroup/RLIMIT/RSS containment is not admission.
   The current feasibility run records missing public memory-service exports;
   do not call similarly named private mkl_serv symbols by guess.
3. **Immutable A_log folding can hoist only an accepted operation.** After an
   exact admitted producer exists, compute its source-bound n16/n48 exp once
   per immutable full parameter/source/profile/runtime identity and fixed FP
   environment. Keep raw A_log unchanged and retain a separately admitted
   derived constant/source/runtime owner. It must be computed, never populated
   from expected archive outputs or per-input lookup. This saves repeated
   constant work but does not close dynamic exp(g)/initial-chunk exponentials.
4. **Portable alternative needs real implementation authority.** Obtain public
   or vendor-authorized exact implementation source and qualify it independently
   on original operands/adversaries. No fit, disassembly-derived vendor port,
   tolerance relaxation, or silent new SLEEF/libm numerical profile. A newly
   declared source target would require explicit separate qualification and
   still would not satisfy exact current pinned Torch CPU evidence by assertion.

No speculative production exp API/provider was added. This preparation proposal
requires ROOT execution; the exact missing exponential remains explicit.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/README.md

Original UTF-8 bytes: 7861; SHA256 `b7eea95e0b35d5c8f0846c36a610affbd46382c8894a2e6ec85e92f99f1d559c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original GDN coefficient preparation — source-only proposal

Author: batch_design_review

- Same MathematicalBackend/PlanBuilder/HostMath/OriginalWindowOperators.
- Compute beta BF16, biased-alpha F32 and softplus F32 from explicit authentic
  a/b input; no g, decay, A_log exponential, recurrent state or token loop.
- Source target: `TorchCpuBf16Avx2MixedSourceV1`, original BF16 matrices and raw
  mixed-source parameters. No default GGUF, vLLM CUDA or full-HF promotion.
- Author: source review, small metadata pins, AST parse and rustfmt only.
  No compilation, tests, operators, model, provider, GPU or container execution.
- Pending convolution proposal remains separate. Apply this minimal delta with
  fresh timestamps and reconcile its explicit GPU refusals with that proposal;
  do not replace either proposal's complete parent files blindly.

## Numerical and source contract

Official modeling SHA762feb6c…412c3b lines617/619:

1. `b.sigmoid()` produces BF16, retaining the original single-call span.
2. Widen `a` to F32, add original raw dt_bias in F32, materialize softplus F32.
3. Complete coefficients additionally need `-exp(A_log) * softplus`; this slice
   does not evaluate that expression. A_log descriptor remains original F32
   rank1 and explicitly unevaluated.

- One call has heads16/48, not a flattened collection of independent rows.
  Sigmoid preserves vector32 prefix/scalar tail. Softplus preserves the accepted
  finite contiguous/div16 beta1/threshold20 path. No new reducer or exp/log.
- Pure CPU helper accepts raw rank1 BF16/F32 dt_bias, privately widens its raw
  bytes, validates every operand before numerical work and preserves separate
  F32 addition and existing nonlinear calls. F32 synthetic raw-weight cases
  remain explicit; actual .8B/2B source dt_bias must be BF16.
- Model slice requires real LinearAttention layer; exact complete original
  in_proj_a/in_proj_b output-input BF16 matrix roles, source issuer, geometry,
  A_log F32 rank1 and dt_bias raw role. A_log is metadata-only in this prefix,
  not a validated exponential value. No raw learned storage mutation/cache.
- Two typed plan nodes preserve distinct result precisions and source order:
  beta first, then biased-alpha/softplus. Both inputs and all cold guards are
  checked before either node is recorded. No input node or StateRoot is invented.
- GPU preflight/lowerings explicitly refuse these new nodes. Existing full-HF,
  model/session, initial-chunk/cached recurrence and provider guards stay closed.

## Physical ownership

- Raw external a/b import is representation/source/profile checked; its actual
  provenance belongs to the qualifier. It asserts no hidden/model producer.
- One admitted same-source aggregate backs both private inputs. Another backs
  all three results. Every backing is retained by existing HostValue owners.
  At most five tiny buffers per case, with two real admission leases.
- Raw dt_bias is staged through the original complete ParameterWindow and held
  through synchronous math. Both input issuers must match the source owner.
- Existing unaware providers refuse before callbacks. Cancellation and real
  capacity refusal publish nothing. No Vec escape from leased window results.
- Native proof checks source+two separate4MiB bindings+control =74MiB/four
  reservations, +2MiB per retained case,70MiB after caller/source drop with both
  cases retained, then actual final0. No reused one-binding ledger assumption.

## Independent public reference

- `prepare_inputs.py`: ROOT-only stdlib full source SHA/FD audit and exact-byte
  export of original a/b matrices, raw A_log/dt_bias and each accepted teacher's
  actual layer0 input-norm hidden. No hidden reconstruction or tensor arithmetic.
- `reference.py`: unchanged pinned public F.linear, then literal source617/619
  public expression. All8 selected dispatches observed (only an undispatched
  already-F32 A_log no-op cast may be absent, with actual exp-input bits bound).
  Plain/observed beta/g, a/b read sets and learned source bytes must be exact.
- .8B: all11 produced stages, including a/b, must equal the existing untouched
  coefficient capture before targets are saved. Independent validator directly
  checks raw archive words again. All98 prior fullV/cache checks remain the
  accepted provenance; this is no new whole-model run.
- 2B: actual hidden and original learned bytes come from its independently
  accepted source/paired observation. New a/b/intermediates are explicitly
  public-fragment references; no missing original raw stage is invented. Its
  old core beta/g metadata has no raw tensor payload to compare in this scope.
- g/A_log exp are retained as diagnostic reference outputs only. They never
  enter production helper calls or establish exp eligibility.
- Four real cases ×16 heads ×(two imports +three preparation results) =320 raw
  comparison words. Native replay also rereads every result after callers drop.
  This is unrelated to full248320 logits/all42 Rust model execution.

## ROOT commands and gates

All outputs are fresh; preserve first failures and executed bytes. Standalone
lock normalization/builds are ROOT-owned/offline and must retain actual receipts.
Public reference may run before MAIN merge (`frozen(False)` authenticates own
source/metadata); native proof additionally requires exact intended MAIN bytes.

```bash
RUN=/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020
python3 "$RUN/test_contracts.py"
python3 "$RUN/prepare_inputs.py" --output "$RUN/root-inputs-1"
python3 "$RUN/run_reference.py" --inputs "$RUN/root-inputs-1/inputs.json" \
  --new-output "$RUN/root-public-1"
python3 "$RUN/validate_reference.py" --inputs "$RUN/root-inputs-1/inputs.json" \
  --public-output "$RUN/root-public-1"
```

- Container uses the already accepted immutable derived CPU image IDf57d77f5/
  manifesteec72f47, original exact CPU/libm binary pins, four explicit CPUs set
  before Torch,4threads,12GiB, networknone, no GPU/source model mount.
- `owned_container_v4.py` is byte-identical to the reviewed helper. Nonce/CID
  binding, WNOWAIT, whole-group TERM/KILL before reap, final absence and bounded
  non-daemon log EOF/close remain intact. No independent lifecycle reinvention.
- Then ROOT independently reviews/applies the minimal delta and runs focused
  core/CPU/model strict Clippy/tests, relevant no-HF/GPU refusal features and the
  full isolated check before accepting implementation. These are prepared only.

```bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --release --offline --manifest-path "$RUN/Cargo.toml"
python3 "$RUN/run_replay.py" --binary "$ACTUAL_ATTESTED_BINARY" \
  --binary-sha256 "$ACTUAL_BINARY_SHA" --inputs "$RUN/root-inputs-1/inputs.json" \
  --inputs-sha256 "$ACTUAL_INPUTS_SHA" \
  --reference "$RUN/root-public-1/reference/reference.json" \
  --reference-sha256 "$ACTUAL_PUBLIC_REFERENCE_SHA" --new-output "$RUN/root-native-1"
python3 "$RUN/validate_replay.py" --output "$RUN/root-native-1" \
  --inputs "$RUN/root-inputs-1/inputs.json" \
  --reference "$RUN/root-public-1/reference/reference.json"
```

Native launcher uses the unchanged reviewed cleanup functions, exec-FD digest,
four CPUs, finite wall/address-space/log caps, full source-before/after audit,
both models serially and immutable input/reference/source closure.

## Unfinished

- ROOT source review, compilation, strict/fixture tests, public oracle and native
  trained-source320-word proof are all unexecuted by the author.
- See `EXP_BOUNDARY.md`: admitted exact generic exp still required to finish
  coefficients and dynamic initial/cached core exponentials.
- No full GDN/FLA, all42 full-HF graph,27B model, device/serving or performance
  qualification follows from this prefix.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/TODO.md

Original UTF-8 bytes: 1375; SHA256 `49e85486edfce85ab664f348cb9acf5ef990f851e8d1460c0b49d70f6e1ba15f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist — coefficient preparation only

Parent: model-execution/shared original-HF engine; owner batch_design_review.

- [x] Read accepted sigmoid/softplus, actual .8B coefficient frames, official
  source617/619, actual 2B restored-source metadata and native ABI root-actual-3.
- [x] Identify exp boundary: direct vmsExp four-call replay PASS; resource
  admission/provider and portable source schedule remain unqualified.
- [x] Prepare same-engine CPU helper, two typed plan nodes, HostMath adapter,
  OriginalWindowOperators raw dt staging, exact-role a/b imports and outputs.
  No exp/g/decay or StateRoot/model eligibility changes.
- [x] Prepare source-free CPU/core/window ownership/pressure/cancel/provider
  tests. All tests unexecuted by author; rustfmt only.
- [x] Prepare bounded ROOT-only selected public stage oracle, inputs exporter,
  actual .8B stage checks and explicit 2B public-fragment provenance.
- [x] Prepare genuine two-source Rust raw-word replay/lease/final-zero harness.
- [x] Source-review complete delta, AST/rustfmt, freeze hashes/copies/commands.
- [ ] ROOT independent review, compilation/strict tests, public oracle and
  actual Rust two-source all-head comparison.
- [ ] Full coefficients still need legitimate admitted exact exp provider/math;
  initial chunk versus cached core/FLA/full-HF/profile/performance remain closed.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/COMPOSITION.md

Original UTF-8 bytes: 1114; SHA256 `1ac7370fab1e5115fc18cc46832e031063322fdbf60729acc7100f0baad55cb0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive source composition only

Author: batch_design_review

- Original freeze39ca13eb, all41 live/frozen files unchanged.
- Fresh baseline copies8 actual MAIN parents after reviewed GDN ROOTV6/V7 merge.
- Preserve all prior GDN causal_conv_silu nodes/methods/guards/tests. Add only
  coefficient preparation alongside those methods and refusal match arms.
- Existing record input loop is factored unchanged into validate_record_inputs;
  original GDN homogeneous result recording remains intact.
- The5 new coefficient modules, all numerical helpers, fixtures, reference and
  native proof scripts are byte-identical to the original freeze. README/TODO
  only adjust invocation location and describe composition.
- NEW_DELTA excludes2 explicit GPU-refusal parent files; GPU_REFUSAL is separate.
- main_after_merge union includes original CPU attention closure and accepted
  GDN ROOTV7 closure, rehashed current small source files, then13 proposal paths.
- No source/ABI/model/operator/build/GPU tests by author. AST/rustfmt only.
- ROOT must review/test/apply before any inference/profile eligibility claim.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/EXP_BOUNDARY.md

Original UTF-8 bytes: 4517; SHA256 `4e169bddd45a40016647f6944c2c62718b811414c2569b6f96175b4c37a5dd8e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Remaining exact exponential boundary

Author: batch_design_review. Source/accepted receipt audit only.

## Actual qualified boundary

- MAIN native ABI run
  `.agents/runs/model-execution-pinned-native-provider-probe-20261020/root-actual-3`
  reports `PASS_BOUNDED_NATIVE_PUBLIC_ABI_ONLY`, four genuine vmsExp calls,
  six public CBLAS batch-GEMMs and32 STRSM calls. Earlier affinity failures are
  preserved; actual3 verifies four CPUs before loading/calling providers.
- Its `validation.physical_memory_admission` is `UNQUALIFIED`; the Rust report
  separately leaves native-internal memory/model/profile execution unqualified.
  Do not turn this successful diagnostic into an admitted production provider.
- Pinned ATen cf30153c… UnaryOpsKernel.cpp877 invokes the FP32 VML exp path;
  `ATen/cpu/vml.h` selects public vmsExp mode0x140102 (HA | FTZDAZ_OFF |
  ERRMODE_IGNORE). Exact public ABI and selected provider/runtime closure were
  replayed. No portable implementation source/internal exp schedule is proved.
- Private accepted SLEEF exp exists only inside declared SiLU/softplus/sigmoid
  output contracts. It is not a public F32 Tensor.exp replacement. Neither CPU
  libm, HA accuracy guarantees nor a finite recorded set prove equality.

## Source schedule which remains distinct

- A_log: original rank1 F32 source → `.float()` no-op → public exp F32 → unary
  negation. Runtime a projection BF16 → F32 +raw BF16 dt_bias → softplus F32;
  separate multiplication produces g F32. Beta independently stays BF16.
- Initial slow chunk pads one valid token to64; it applies F64-accumulating
  cumsum→F32 materialization, then different exp operands/layouts, including
  masked pairwise differences. This is not cached recurrence on zero state.
- Cached slow core separately executes exp(g) line483 on `[1,16]` contiguous.
  Initial line415 takes a noncontiguous head span. Do not flatten/change those
  iterator calls or infer one admitted span covers every layout and size.
- Current coefficient-preparation output deliberately excludes g/exp/decay;
  the legacy original diagnostic's host `.exp()` is not promoted by this work.

## Shortest legitimate continuation

1. **Exact public VML, optional per-operator provider.** Reuse the proven public
   signature/DSO/mode and one original Rust schedule. Before any cold load/call,
   establish admitted provider-owned mappings, constructors, worker stacks,
   scratch/pools and caller operands/output lifetime under the existing resource
   owner. Synchronous return can retire per-call buffers only when provider
   persistent allocations remain charged. Cancellation retains all owners until
   return, then discards private output. Never unload an in-flight provider.
2. **Close allocation rather than assume it.** Public exported i_malloc/i_free/
   i_calloc/i_realloc are a potential vendor allocator seam after exact public
   ABI validation. They do not automatically cover Torch DSO constructors,
   loader/OpenMP/runtime allocations. Source-prove and physically account for
   every actual route, alignment, realloc ownership/OOM and cold/concurrent
   exclusion before enabling. Cgroup/RLIMIT/RSS containment is not admission.
   The current feasibility run records missing public memory-service exports;
   do not call similarly named private mkl_serv symbols by guess.
3. **Immutable A_log folding can hoist only an accepted operation.** After an
   exact admitted producer exists, compute its source-bound n16/n48 exp once
   per immutable full parameter/source/profile/runtime identity and fixed FP
   environment. Keep raw A_log unchanged and retain a separately admitted
   derived constant/source/runtime owner. It must be computed, never populated
   from expected archive outputs or per-input lookup. This saves repeated
   constant work but does not close dynamic exp(g)/initial-chunk exponentials.
4. **Portable alternative needs real implementation authority.** Obtain public
   or vendor-authorized exact implementation source and qualify it independently
   on original operands/adversaries. No fit, disassembly-derived vendor port,
   tolerance relaxation, or silent new SLEEF/libm numerical profile. A newly
   declared source target would require explicit separate qualification and
   still would not satisfy exact current pinned Torch CPU evidence by assertion.

No speculative production exp API/provider was added. This preparation proposal
requires ROOT execution; the exact missing exponential remains explicit.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/README.md

Original UTF-8 bytes: 7972; SHA256 `72a68157c7188fa7e6f8d568b7934eb1e55a87b8f81094fac61c4fba47f009e1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original GDN coefficient preparation — source-only proposal

Author: batch_design_review

- Same MathematicalBackend/PlanBuilder/HostMath/OriginalWindowOperators.
- Compute beta BF16, biased-alpha F32 and softplus F32 from explicit authentic
  a/b input; no g, decay, A_log exponential, recurrent state or token loop.
- Source target: `TorchCpuBf16Avx2MixedSourceV1`, original BF16 matrices and raw
  mixed-source parameters. No default GGUF, vLLM CUDA or full-HF promotion.
- Author: source review, small metadata pins, AST parse and rustfmt only.
  No compilation, tests, operators, model, provider, GPU or container execution.
- Additive followup-v2 composes only the coefficient wiring onto actual reviewed
  GDN ROOTV6/V7 MAIN source. Both original GDN typed operations and explicit GPU
  refusals remain. Original coefficient freeze39ca13eb and its41files stay exact.
  Apply the minimal derivative with fresh timestamps; no MAIN edits by author.

## Numerical and source contract

Official modeling SHA762feb6c…412c3b lines617/619:

1. `b.sigmoid()` produces BF16, retaining the original single-call span.
2. Widen `a` to F32, add original raw dt_bias in F32, materialize softplus F32.
3. Complete coefficients additionally need `-exp(A_log) * softplus`; this slice
   does not evaluate that expression. A_log descriptor remains original F32
   rank1 and explicitly unevaluated.

- One call has heads16/48, not a flattened collection of independent rows.
  Sigmoid preserves vector32 prefix/scalar tail. Softplus preserves the accepted
  finite contiguous/div16 beta1/threshold20 path. No new reducer or exp/log.
- Pure CPU helper accepts raw rank1 BF16/F32 dt_bias, privately widens its raw
  bytes, validates every operand before numerical work and preserves separate
  F32 addition and existing nonlinear calls. F32 synthetic raw-weight cases
  remain explicit; actual .8B/2B source dt_bias must be BF16.
- Model slice requires real LinearAttention layer; exact complete original
  in_proj_a/in_proj_b output-input BF16 matrix roles, source issuer, geometry,
  A_log F32 rank1 and dt_bias raw role. A_log is metadata-only in this prefix,
  not a validated exponential value. No raw learned storage mutation/cache.
- Two typed plan nodes preserve distinct result precisions and source order:
  beta first, then biased-alpha/softplus. Both inputs and all cold guards are
  checked before either node is recorded. No input node or StateRoot is invented.
- GPU preflight/lowerings explicitly refuse these new nodes. Existing full-HF,
  model/session, initial-chunk/cached recurrence and provider guards stay closed.

## Physical ownership

- Raw external a/b import is representation/source/profile checked; its actual
  provenance belongs to the qualifier. It asserts no hidden/model producer.
- One admitted same-source aggregate backs both private inputs. Another backs
  all three results. Every backing is retained by existing HostValue owners.
  At most five tiny buffers per case, with two real admission leases.
- Raw dt_bias is staged through the original complete ParameterWindow and held
  through synchronous math. Both input issuers must match the source owner.
- Existing unaware providers refuse before callbacks. Cancellation and real
  capacity refusal publish nothing. No Vec escape from leased window results.
- Native proof checks source+two separate4MiB bindings+control =74MiB/four
  reservations, +2MiB per retained case,70MiB after caller/source drop with both
  cases retained, then actual final0. No reused one-binding ledger assumption.

## Independent public reference

- `prepare_inputs.py`: ROOT-only stdlib full source SHA/FD audit and exact-byte
  export of original a/b matrices, raw A_log/dt_bias and each accepted teacher's
  actual layer0 input-norm hidden. No hidden reconstruction or tensor arithmetic.
- `reference.py`: unchanged pinned public F.linear, then literal source617/619
  public expression. All8 selected dispatches observed (only an undispatched
  already-F32 A_log no-op cast may be absent, with actual exp-input bits bound).
  Plain/observed beta/g, a/b read sets and learned source bytes must be exact.
- .8B: all11 produced stages, including a/b, must equal the existing untouched
  coefficient capture before targets are saved. Independent validator directly
  checks raw archive words again. All98 prior fullV/cache checks remain the
  accepted provenance; this is no new whole-model run.
- 2B: actual hidden and original learned bytes come from its independently
  accepted source/paired observation. New a/b/intermediates are explicitly
  public-fragment references; no missing original raw stage is invented. Its
  old core beta/g metadata has no raw tensor payload to compare in this scope.
- g/A_log exp are retained as diagnostic reference outputs only. They never
  enter production helper calls or establish exp eligibility.
- Four real cases ×16 heads ×(two imports +three preparation results) =320 raw
  comparison words. Native replay also rereads every result after callers drop.
  This is unrelated to full248320 logits/all42 Rust model execution.

## ROOT commands and gates

All outputs are fresh; preserve first failures and executed bytes. Standalone
lock normalization/builds are ROOT-owned/offline and must retain actual receipts.
Public reference may run before MAIN merge (`frozen(False)` authenticates own
source/metadata); native proof additionally requires exact intended MAIN bytes.

```bash
RUN=/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2
python3 "$RUN/test_contracts.py"
python3 "$RUN/prepare_inputs.py" --output "$RUN/root-inputs-1"
python3 "$RUN/run_reference.py" --inputs "$RUN/root-inputs-1/inputs.json" \
  --new-output "$RUN/root-public-1"
python3 "$RUN/validate_reference.py" --inputs "$RUN/root-inputs-1/inputs.json" \
  --public-output "$RUN/root-public-1"
```

- Container uses the already accepted immutable derived CPU image IDf57d77f5/
  manifesteec72f47, original exact CPU/libm binary pins, four explicit CPUs set
  before Torch,4threads,12GiB, networknone, no GPU/source model mount.
- `owned_container_v4.py` is byte-identical to the reviewed helper. Nonce/CID
  binding, WNOWAIT, whole-group TERM/KILL before reap, final absence and bounded
  non-daemon log EOF/close remain intact. No independent lifecycle reinvention.
- Then ROOT independently reviews/applies the minimal delta and runs focused
  core/CPU/model strict Clippy/tests, relevant no-HF/GPU refusal features and the
  full isolated check before accepting implementation. These are prepared only.

```bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --release --offline --manifest-path "$RUN/Cargo.toml"
python3 "$RUN/run_replay.py" --binary "$ACTUAL_ATTESTED_BINARY" \
  --binary-sha256 "$ACTUAL_BINARY_SHA" --inputs "$RUN/root-inputs-1/inputs.json" \
  --inputs-sha256 "$ACTUAL_INPUTS_SHA" \
  --reference "$RUN/root-public-1/reference/reference.json" \
  --reference-sha256 "$ACTUAL_PUBLIC_REFERENCE_SHA" --new-output "$RUN/root-native-1"
python3 "$RUN/validate_replay.py" --output "$RUN/root-native-1" \
  --inputs "$RUN/root-inputs-1/inputs.json" \
  --reference "$RUN/root-public-1/reference/reference.json"
```

Native launcher uses the unchanged reviewed cleanup functions, exec-FD digest,
four CPUs, finite wall/address-space/log caps, full source-before/after audit,
both models serially and immutable input/reference/source closure.

## Unfinished

- ROOT source review, compilation, strict/fixture tests, public oracle and native
  trained-source320-word proof are all unexecuted by the author.
- See `EXP_BOUNDARY.md`: admitted exact generic exp still required to finish
  coefficients and dynamic initial/cached core exponentials.
- No full GDN/FLA, all42 full-HF graph,27B model, device/serving or performance
  qualification follows from this prefix.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/TODO.md

Original UTF-8 bytes: 1531; SHA256 `2a5b61a994b97d2cf8c44961de30247d1eaa9791baee8ac6d1ecfa799409a50f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist — coefficient preparation only

Parent: model-execution/shared original-HF engine; owner batch_design_review.

- [x] Read accepted sigmoid/softplus, actual .8B coefficient frames, official
  source617/619, actual 2B restored-source metadata and native ABI root-actual-3.
- [x] Identify exp boundary: direct vmsExp four-call replay PASS; resource
  admission/provider and portable source schedule remain unqualified.
- [x] Prepare same-engine CPU helper, two typed plan nodes, HostMath adapter,
  OriginalWindowOperators raw dt staging, exact-role a/b imports and outputs.
  No exp/g/decay or StateRoot/model eligibility changes.
- [x] Prepare source-free CPU/core/window ownership/pressure/cancel/provider
  tests. All tests unexecuted by author; rustfmt only.
- [x] Prepare bounded ROOT-only selected public stage oracle, inputs exporter,
  actual .8B stage checks and explicit 2B public-fragment provenance.
- [x] Prepare genuine two-source Rust raw-word replay/lease/final-zero harness.
- [x] Source-review complete delta, AST/rustfmt, freeze hashes/copies/commands.
- [x] Preserve initial39ca13eb freeze; compose narrow parent wiring onto actual
  reviewed GDN ROOTV6/V7 bytes in separate followup-v2 only. No MAIN edits.
- [ ] ROOT independent review, compilation/strict tests, public oracle and
  actual Rust two-source all-head comparison.
- [ ] Full coefficients still need legitimate admitted exact exp provider/math;
  initial chunk versus cached core/FLA/full-HF/profile/performance remain closed.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/frozen/COMPOSITION.md

Original UTF-8 bytes: 1114; SHA256 `1ac7370fab1e5115fc18cc46832e031063322fdbf60729acc7100f0baad55cb0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive source composition only

Author: batch_design_review

- Original freeze39ca13eb, all41 live/frozen files unchanged.
- Fresh baseline copies8 actual MAIN parents after reviewed GDN ROOTV6/V7 merge.
- Preserve all prior GDN causal_conv_silu nodes/methods/guards/tests. Add only
  coefficient preparation alongside those methods and refusal match arms.
- Existing record input loop is factored unchanged into validate_record_inputs;
  original GDN homogeneous result recording remains intact.
- The5 new coefficient modules, all numerical helpers, fixtures, reference and
  native proof scripts are byte-identical to the original freeze. README/TODO
  only adjust invocation location and describe composition.
- NEW_DELTA excludes2 explicit GPU-refusal parent files; GPU_REFUSAL is separate.
- main_after_merge union includes original CPU attention closure and accepted
  GDN ROOTV7 closure, rehashed current small source files, then13 proposal paths.
- No source/ABI/model/operator/build/GPU tests by author. AST/rustfmt only.
- ROOT must review/test/apply before any inference/profile eligibility claim.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/frozen/EXP_BOUNDARY.md

Original UTF-8 bytes: 4517; SHA256 `4e169bddd45a40016647f6944c2c62718b811414c2569b6f96175b4c37a5dd8e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Remaining exact exponential boundary

Author: batch_design_review. Source/accepted receipt audit only.

## Actual qualified boundary

- MAIN native ABI run
  `.agents/runs/model-execution-pinned-native-provider-probe-20261020/root-actual-3`
  reports `PASS_BOUNDED_NATIVE_PUBLIC_ABI_ONLY`, four genuine vmsExp calls,
  six public CBLAS batch-GEMMs and32 STRSM calls. Earlier affinity failures are
  preserved; actual3 verifies four CPUs before loading/calling providers.
- Its `validation.physical_memory_admission` is `UNQUALIFIED`; the Rust report
  separately leaves native-internal memory/model/profile execution unqualified.
  Do not turn this successful diagnostic into an admitted production provider.
- Pinned ATen cf30153c… UnaryOpsKernel.cpp877 invokes the FP32 VML exp path;
  `ATen/cpu/vml.h` selects public vmsExp mode0x140102 (HA | FTZDAZ_OFF |
  ERRMODE_IGNORE). Exact public ABI and selected provider/runtime closure were
  replayed. No portable implementation source/internal exp schedule is proved.
- Private accepted SLEEF exp exists only inside declared SiLU/softplus/sigmoid
  output contracts. It is not a public F32 Tensor.exp replacement. Neither CPU
  libm, HA accuracy guarantees nor a finite recorded set prove equality.

## Source schedule which remains distinct

- A_log: original rank1 F32 source → `.float()` no-op → public exp F32 → unary
  negation. Runtime a projection BF16 → F32 +raw BF16 dt_bias → softplus F32;
  separate multiplication produces g F32. Beta independently stays BF16.
- Initial slow chunk pads one valid token to64; it applies F64-accumulating
  cumsum→F32 materialization, then different exp operands/layouts, including
  masked pairwise differences. This is not cached recurrence on zero state.
- Cached slow core separately executes exp(g) line483 on `[1,16]` contiguous.
  Initial line415 takes a noncontiguous head span. Do not flatten/change those
  iterator calls or infer one admitted span covers every layout and size.
- Current coefficient-preparation output deliberately excludes g/exp/decay;
  the legacy original diagnostic's host `.exp()` is not promoted by this work.

## Shortest legitimate continuation

1. **Exact public VML, optional per-operator provider.** Reuse the proven public
   signature/DSO/mode and one original Rust schedule. Before any cold load/call,
   establish admitted provider-owned mappings, constructors, worker stacks,
   scratch/pools and caller operands/output lifetime under the existing resource
   owner. Synchronous return can retire per-call buffers only when provider
   persistent allocations remain charged. Cancellation retains all owners until
   return, then discards private output. Never unload an in-flight provider.
2. **Close allocation rather than assume it.** Public exported i_malloc/i_free/
   i_calloc/i_realloc are a potential vendor allocator seam after exact public
   ABI validation. They do not automatically cover Torch DSO constructors,
   loader/OpenMP/runtime allocations. Source-prove and physically account for
   every actual route, alignment, realloc ownership/OOM and cold/concurrent
   exclusion before enabling. Cgroup/RLIMIT/RSS containment is not admission.
   The current feasibility run records missing public memory-service exports;
   do not call similarly named private mkl_serv symbols by guess.
3. **Immutable A_log folding can hoist only an accepted operation.** After an
   exact admitted producer exists, compute its source-bound n16/n48 exp once
   per immutable full parameter/source/profile/runtime identity and fixed FP
   environment. Keep raw A_log unchanged and retain a separately admitted
   derived constant/source/runtime owner. It must be computed, never populated
   from expected archive outputs or per-input lookup. This saves repeated
   constant work but does not close dynamic exp(g)/initial-chunk exponentials.
4. **Portable alternative needs real implementation authority.** Obtain public
   or vendor-authorized exact implementation source and qualify it independently
   on original operands/adversaries. No fit, disassembly-derived vendor port,
   tolerance relaxation, or silent new SLEEF/libm numerical profile. A newly
   declared source target would require explicit separate qualification and
   still would not satisfy exact current pinned Torch CPU evidence by assertion.

No speculative production exp API/provider was added. This preparation proposal
requires ROOT execution; the exact missing exponential remains explicit.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/frozen/README.md

Original UTF-8 bytes: 7972; SHA256 `72a68157c7188fa7e6f8d568b7934eb1e55a87b8f81094fac61c4fba47f009e1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original GDN coefficient preparation — source-only proposal

Author: batch_design_review

- Same MathematicalBackend/PlanBuilder/HostMath/OriginalWindowOperators.
- Compute beta BF16, biased-alpha F32 and softplus F32 from explicit authentic
  a/b input; no g, decay, A_log exponential, recurrent state or token loop.
- Source target: `TorchCpuBf16Avx2MixedSourceV1`, original BF16 matrices and raw
  mixed-source parameters. No default GGUF, vLLM CUDA or full-HF promotion.
- Author: source review, small metadata pins, AST parse and rustfmt only.
  No compilation, tests, operators, model, provider, GPU or container execution.
- Additive followup-v2 composes only the coefficient wiring onto actual reviewed
  GDN ROOTV6/V7 MAIN source. Both original GDN typed operations and explicit GPU
  refusals remain. Original coefficient freeze39ca13eb and its41files stay exact.
  Apply the minimal derivative with fresh timestamps; no MAIN edits by author.

## Numerical and source contract

Official modeling SHA762feb6c…412c3b lines617/619:

1. `b.sigmoid()` produces BF16, retaining the original single-call span.
2. Widen `a` to F32, add original raw dt_bias in F32, materialize softplus F32.
3. Complete coefficients additionally need `-exp(A_log) * softplus`; this slice
   does not evaluate that expression. A_log descriptor remains original F32
   rank1 and explicitly unevaluated.

- One call has heads16/48, not a flattened collection of independent rows.
  Sigmoid preserves vector32 prefix/scalar tail. Softplus preserves the accepted
  finite contiguous/div16 beta1/threshold20 path. No new reducer or exp/log.
- Pure CPU helper accepts raw rank1 BF16/F32 dt_bias, privately widens its raw
  bytes, validates every operand before numerical work and preserves separate
  F32 addition and existing nonlinear calls. F32 synthetic raw-weight cases
  remain explicit; actual .8B/2B source dt_bias must be BF16.
- Model slice requires real LinearAttention layer; exact complete original
  in_proj_a/in_proj_b output-input BF16 matrix roles, source issuer, geometry,
  A_log F32 rank1 and dt_bias raw role. A_log is metadata-only in this prefix,
  not a validated exponential value. No raw learned storage mutation/cache.
- Two typed plan nodes preserve distinct result precisions and source order:
  beta first, then biased-alpha/softplus. Both inputs and all cold guards are
  checked before either node is recorded. No input node or StateRoot is invented.
- GPU preflight/lowerings explicitly refuse these new nodes. Existing full-HF,
  model/session, initial-chunk/cached recurrence and provider guards stay closed.

## Physical ownership

- Raw external a/b import is representation/source/profile checked; its actual
  provenance belongs to the qualifier. It asserts no hidden/model producer.
- One admitted same-source aggregate backs both private inputs. Another backs
  all three results. Every backing is retained by existing HostValue owners.
  At most five tiny buffers per case, with two real admission leases.
- Raw dt_bias is staged through the original complete ParameterWindow and held
  through synchronous math. Both input issuers must match the source owner.
- Existing unaware providers refuse before callbacks. Cancellation and real
  capacity refusal publish nothing. No Vec escape from leased window results.
- Native proof checks source+two separate4MiB bindings+control =74MiB/four
  reservations, +2MiB per retained case,70MiB after caller/source drop with both
  cases retained, then actual final0. No reused one-binding ledger assumption.

## Independent public reference

- `prepare_inputs.py`: ROOT-only stdlib full source SHA/FD audit and exact-byte
  export of original a/b matrices, raw A_log/dt_bias and each accepted teacher's
  actual layer0 input-norm hidden. No hidden reconstruction or tensor arithmetic.
- `reference.py`: unchanged pinned public F.linear, then literal source617/619
  public expression. All8 selected dispatches observed (only an undispatched
  already-F32 A_log no-op cast may be absent, with actual exp-input bits bound).
  Plain/observed beta/g, a/b read sets and learned source bytes must be exact.
- .8B: all11 produced stages, including a/b, must equal the existing untouched
  coefficient capture before targets are saved. Independent validator directly
  checks raw archive words again. All98 prior fullV/cache checks remain the
  accepted provenance; this is no new whole-model run.
- 2B: actual hidden and original learned bytes come from its independently
  accepted source/paired observation. New a/b/intermediates are explicitly
  public-fragment references; no missing original raw stage is invented. Its
  old core beta/g metadata has no raw tensor payload to compare in this scope.
- g/A_log exp are retained as diagnostic reference outputs only. They never
  enter production helper calls or establish exp eligibility.
- Four real cases ×16 heads ×(two imports +three preparation results) =320 raw
  comparison words. Native replay also rereads every result after callers drop.
  This is unrelated to full248320 logits/all42 Rust model execution.

## ROOT commands and gates

All outputs are fresh; preserve first failures and executed bytes. Standalone
lock normalization/builds are ROOT-owned/offline and must retain actual receipts.
Public reference may run before MAIN merge (`frozen(False)` authenticates own
source/metadata); native proof additionally requires exact intended MAIN bytes.

```bash
RUN=/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2
python3 "$RUN/test_contracts.py"
python3 "$RUN/prepare_inputs.py" --output "$RUN/root-inputs-1"
python3 "$RUN/run_reference.py" --inputs "$RUN/root-inputs-1/inputs.json" \
  --new-output "$RUN/root-public-1"
python3 "$RUN/validate_reference.py" --inputs "$RUN/root-inputs-1/inputs.json" \
  --public-output "$RUN/root-public-1"
```

- Container uses the already accepted immutable derived CPU image IDf57d77f5/
  manifesteec72f47, original exact CPU/libm binary pins, four explicit CPUs set
  before Torch,4threads,12GiB, networknone, no GPU/source model mount.
- `owned_container_v4.py` is byte-identical to the reviewed helper. Nonce/CID
  binding, WNOWAIT, whole-group TERM/KILL before reap, final absence and bounded
  non-daemon log EOF/close remain intact. No independent lifecycle reinvention.
- Then ROOT independently reviews/applies the minimal delta and runs focused
  core/CPU/model strict Clippy/tests, relevant no-HF/GPU refusal features and the
  full isolated check before accepting implementation. These are prepared only.

```bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --release --offline --manifest-path "$RUN/Cargo.toml"
python3 "$RUN/run_replay.py" --binary "$ACTUAL_ATTESTED_BINARY" \
  --binary-sha256 "$ACTUAL_BINARY_SHA" --inputs "$RUN/root-inputs-1/inputs.json" \
  --inputs-sha256 "$ACTUAL_INPUTS_SHA" \
  --reference "$RUN/root-public-1/reference/reference.json" \
  --reference-sha256 "$ACTUAL_PUBLIC_REFERENCE_SHA" --new-output "$RUN/root-native-1"
python3 "$RUN/validate_replay.py" --output "$RUN/root-native-1" \
  --inputs "$RUN/root-inputs-1/inputs.json" \
  --reference "$RUN/root-public-1/reference/reference.json"
```

Native launcher uses the unchanged reviewed cleanup functions, exec-FD digest,
four CPUs, finite wall/address-space/log caps, full source-before/after audit,
both models serially and immutable input/reference/source closure.

## Unfinished

- ROOT source review, compilation, strict/fixture tests, public oracle and native
  trained-source320-word proof are all unexecuted by the author.
- See `EXP_BOUNDARY.md`: admitted exact generic exp still required to finish
  coefficients and dynamic initial/cached core exponentials.
- No full GDN/FLA, all42 full-HF graph,27B model, device/serving or performance
  qualification follows from this prefix.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/frozen/TODO.md

Original UTF-8 bytes: 1531; SHA256 `2a5b61a994b97d2cf8c44961de30247d1eaa9791baee8ac6d1ecfa799409a50f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist — coefficient preparation only

Parent: model-execution/shared original-HF engine; owner batch_design_review.

- [x] Read accepted sigmoid/softplus, actual .8B coefficient frames, official
  source617/619, actual 2B restored-source metadata and native ABI root-actual-3.
- [x] Identify exp boundary: direct vmsExp four-call replay PASS; resource
  admission/provider and portable source schedule remain unqualified.
- [x] Prepare same-engine CPU helper, two typed plan nodes, HostMath adapter,
  OriginalWindowOperators raw dt staging, exact-role a/b imports and outputs.
  No exp/g/decay or StateRoot/model eligibility changes.
- [x] Prepare source-free CPU/core/window ownership/pressure/cancel/provider
  tests. All tests unexecuted by author; rustfmt only.
- [x] Prepare bounded ROOT-only selected public stage oracle, inputs exporter,
  actual .8B stage checks and explicit 2B public-fragment provenance.
- [x] Prepare genuine two-source Rust raw-word replay/lease/final-zero harness.
- [x] Source-review complete delta, AST/rustfmt, freeze hashes/copies/commands.
- [x] Preserve initial39ca13eb freeze; compose narrow parent wiring onto actual
  reviewed GDN ROOTV6/V7 bytes in separate followup-v2 only. No MAIN edits.
- [ ] ROOT independent review, compilation/strict tests, public oracle and
  actual Rust two-source all-head comparison.
- [ ] Full coefficients still need legitimate admitted exact exp provider/math;
  initial chunk versus cached core/FLA/full-HF/profile/performance remain closed.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/frozen/EXP_BOUNDARY.md

Original UTF-8 bytes: 4517; SHA256 `4e169bddd45a40016647f6944c2c62718b811414c2569b6f96175b4c37a5dd8e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Remaining exact exponential boundary

Author: batch_design_review. Source/accepted receipt audit only.

## Actual qualified boundary

- MAIN native ABI run
  `.agents/runs/model-execution-pinned-native-provider-probe-20261020/root-actual-3`
  reports `PASS_BOUNDED_NATIVE_PUBLIC_ABI_ONLY`, four genuine vmsExp calls,
  six public CBLAS batch-GEMMs and32 STRSM calls. Earlier affinity failures are
  preserved; actual3 verifies four CPUs before loading/calling providers.
- Its `validation.physical_memory_admission` is `UNQUALIFIED`; the Rust report
  separately leaves native-internal memory/model/profile execution unqualified.
  Do not turn this successful diagnostic into an admitted production provider.
- Pinned ATen cf30153c… UnaryOpsKernel.cpp877 invokes the FP32 VML exp path;
  `ATen/cpu/vml.h` selects public vmsExp mode0x140102 (HA | FTZDAZ_OFF |
  ERRMODE_IGNORE). Exact public ABI and selected provider/runtime closure were
  replayed. No portable implementation source/internal exp schedule is proved.
- Private accepted SLEEF exp exists only inside declared SiLU/softplus/sigmoid
  output contracts. It is not a public F32 Tensor.exp replacement. Neither CPU
  libm, HA accuracy guarantees nor a finite recorded set prove equality.

## Source schedule which remains distinct

- A_log: original rank1 F32 source → `.float()` no-op → public exp F32 → unary
  negation. Runtime a projection BF16 → F32 +raw BF16 dt_bias → softplus F32;
  separate multiplication produces g F32. Beta independently stays BF16.
- Initial slow chunk pads one valid token to64; it applies F64-accumulating
  cumsum→F32 materialization, then different exp operands/layouts, including
  masked pairwise differences. This is not cached recurrence on zero state.
- Cached slow core separately executes exp(g) line483 on `[1,16]` contiguous.
  Initial line415 takes a noncontiguous head span. Do not flatten/change those
  iterator calls or infer one admitted span covers every layout and size.
- Current coefficient-preparation output deliberately excludes g/exp/decay;
  the legacy original diagnostic's host `.exp()` is not promoted by this work.

## Shortest legitimate continuation

1. **Exact public VML, optional per-operator provider.** Reuse the proven public
   signature/DSO/mode and one original Rust schedule. Before any cold load/call,
   establish admitted provider-owned mappings, constructors, worker stacks,
   scratch/pools and caller operands/output lifetime under the existing resource
   owner. Synchronous return can retire per-call buffers only when provider
   persistent allocations remain charged. Cancellation retains all owners until
   return, then discards private output. Never unload an in-flight provider.
2. **Close allocation rather than assume it.** Public exported i_malloc/i_free/
   i_calloc/i_realloc are a potential vendor allocator seam after exact public
   ABI validation. They do not automatically cover Torch DSO constructors,
   loader/OpenMP/runtime allocations. Source-prove and physically account for
   every actual route, alignment, realloc ownership/OOM and cold/concurrent
   exclusion before enabling. Cgroup/RLIMIT/RSS containment is not admission.
   The current feasibility run records missing public memory-service exports;
   do not call similarly named private mkl_serv symbols by guess.
3. **Immutable A_log folding can hoist only an accepted operation.** After an
   exact admitted producer exists, compute its source-bound n16/n48 exp once
   per immutable full parameter/source/profile/runtime identity and fixed FP
   environment. Keep raw A_log unchanged and retain a separately admitted
   derived constant/source/runtime owner. It must be computed, never populated
   from expected archive outputs or per-input lookup. This saves repeated
   constant work but does not close dynamic exp(g)/initial-chunk exponentials.
4. **Portable alternative needs real implementation authority.** Obtain public
   or vendor-authorized exact implementation source and qualify it independently
   on original operands/adversaries. No fit, disassembly-derived vendor port,
   tolerance relaxation, or silent new SLEEF/libm numerical profile. A newly
   declared source target would require explicit separate qualification and
   still would not satisfy exact current pinned Torch CPU evidence by assertion.

No speculative production exp API/provider was added. This preparation proposal
requires ROOT execution; the exact missing exponential remains explicit.

````

## Source: .agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/frozen/README.md

Original UTF-8 bytes: 7861; SHA256 `b7eea95e0b35d5c8f0846c36a610affbd46382c8894a2e6ec85e92f99f1d559c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original GDN coefficient preparation — source-only proposal

Author: batch_design_review

- Same MathematicalBackend/PlanBuilder/HostMath/OriginalWindowOperators.
- Compute beta BF16, biased-alpha F32 and softplus F32 from explicit authentic
  a/b input; no g, decay, A_log exponential, recurrent state or token loop.
- Source target: `TorchCpuBf16Avx2MixedSourceV1`, original BF16 matrices and raw
  mixed-source parameters. No default GGUF, vLLM CUDA or full-HF promotion.
- Author: source review, small metadata pins, AST parse and rustfmt only.
  No compilation, tests, operators, model, provider, GPU or container execution.
- Pending convolution proposal remains separate. Apply this minimal delta with
  fresh timestamps and reconcile its explicit GPU refusals with that proposal;
  do not replace either proposal's complete parent files blindly.

## Numerical and source contract

Official modeling SHA762feb6c…412c3b lines617/619:

1. `b.sigmoid()` produces BF16, retaining the original single-call span.
2. Widen `a` to F32, add original raw dt_bias in F32, materialize softplus F32.
3. Complete coefficients additionally need `-exp(A_log) * softplus`; this slice
   does not evaluate that expression. A_log descriptor remains original F32
   rank1 and explicitly unevaluated.

- One call has heads16/48, not a flattened collection of independent rows.
  Sigmoid preserves vector32 prefix/scalar tail. Softplus preserves the accepted
  finite contiguous/div16 beta1/threshold20 path. No new reducer or exp/log.
- Pure CPU helper accepts raw rank1 BF16/F32 dt_bias, privately widens its raw
  bytes, validates every operand before numerical work and preserves separate
  F32 addition and existing nonlinear calls. F32 synthetic raw-weight cases
  remain explicit; actual .8B/2B source dt_bias must be BF16.
- Model slice requires real LinearAttention layer; exact complete original
  in_proj_a/in_proj_b output-input BF16 matrix roles, source issuer, geometry,
  A_log F32 rank1 and dt_bias raw role. A_log is metadata-only in this prefix,
  not a validated exponential value. No raw learned storage mutation/cache.
- Two typed plan nodes preserve distinct result precisions and source order:
  beta first, then biased-alpha/softplus. Both inputs and all cold guards are
  checked before either node is recorded. No input node or StateRoot is invented.
- GPU preflight/lowerings explicitly refuse these new nodes. Existing full-HF,
  model/session, initial-chunk/cached recurrence and provider guards stay closed.

## Physical ownership

- Raw external a/b import is representation/source/profile checked; its actual
  provenance belongs to the qualifier. It asserts no hidden/model producer.
- One admitted same-source aggregate backs both private inputs. Another backs
  all three results. Every backing is retained by existing HostValue owners.
  At most five tiny buffers per case, with two real admission leases.
- Raw dt_bias is staged through the original complete ParameterWindow and held
  through synchronous math. Both input issuers must match the source owner.
- Existing unaware providers refuse before callbacks. Cancellation and real
  capacity refusal publish nothing. No Vec escape from leased window results.
- Native proof checks source+two separate4MiB bindings+control =74MiB/four
  reservations, +2MiB per retained case,70MiB after caller/source drop with both
  cases retained, then actual final0. No reused one-binding ledger assumption.

## Independent public reference

- `prepare_inputs.py`: ROOT-only stdlib full source SHA/FD audit and exact-byte
  export of original a/b matrices, raw A_log/dt_bias and each accepted teacher's
  actual layer0 input-norm hidden. No hidden reconstruction or tensor arithmetic.
- `reference.py`: unchanged pinned public F.linear, then literal source617/619
  public expression. All8 selected dispatches observed (only an undispatched
  already-F32 A_log no-op cast may be absent, with actual exp-input bits bound).
  Plain/observed beta/g, a/b read sets and learned source bytes must be exact.
- .8B: all11 produced stages, including a/b, must equal the existing untouched
  coefficient capture before targets are saved. Independent validator directly
  checks raw archive words again. All98 prior fullV/cache checks remain the
  accepted provenance; this is no new whole-model run.
- 2B: actual hidden and original learned bytes come from its independently
  accepted source/paired observation. New a/b/intermediates are explicitly
  public-fragment references; no missing original raw stage is invented. Its
  old core beta/g metadata has no raw tensor payload to compare in this scope.
- g/A_log exp are retained as diagnostic reference outputs only. They never
  enter production helper calls or establish exp eligibility.
- Four real cases ×16 heads ×(two imports +three preparation results) =320 raw
  comparison words. Native replay also rereads every result after callers drop.
  This is unrelated to full248320 logits/all42 Rust model execution.

## ROOT commands and gates

All outputs are fresh; preserve first failures and executed bytes. Standalone
lock normalization/builds are ROOT-owned/offline and must retain actual receipts.
Public reference may run before MAIN merge (`frozen(False)` authenticates own
source/metadata); native proof additionally requires exact intended MAIN bytes.

```bash
RUN=/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020
python3 "$RUN/test_contracts.py"
python3 "$RUN/prepare_inputs.py" --output "$RUN/root-inputs-1"
python3 "$RUN/run_reference.py" --inputs "$RUN/root-inputs-1/inputs.json" \
  --new-output "$RUN/root-public-1"
python3 "$RUN/validate_reference.py" --inputs "$RUN/root-inputs-1/inputs.json" \
  --public-output "$RUN/root-public-1"
```

- Container uses the already accepted immutable derived CPU image IDf57d77f5/
  manifesteec72f47, original exact CPU/libm binary pins, four explicit CPUs set
  before Torch,4threads,12GiB, networknone, no GPU/source model mount.
- `owned_container_v4.py` is byte-identical to the reviewed helper. Nonce/CID
  binding, WNOWAIT, whole-group TERM/KILL before reap, final absence and bounded
  non-daemon log EOF/close remain intact. No independent lifecycle reinvention.
- Then ROOT independently reviews/applies the minimal delta and runs focused
  core/CPU/model strict Clippy/tests, relevant no-HF/GPU refusal features and the
  full isolated check before accepting implementation. These are prepared only.

```bash
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo build --release --offline --manifest-path "$RUN/Cargo.toml"
python3 "$RUN/run_replay.py" --binary "$ACTUAL_ATTESTED_BINARY" \
  --binary-sha256 "$ACTUAL_BINARY_SHA" --inputs "$RUN/root-inputs-1/inputs.json" \
  --inputs-sha256 "$ACTUAL_INPUTS_SHA" \
  --reference "$RUN/root-public-1/reference/reference.json" \
  --reference-sha256 "$ACTUAL_PUBLIC_REFERENCE_SHA" --new-output "$RUN/root-native-1"
python3 "$RUN/validate_replay.py" --output "$RUN/root-native-1" \
  --inputs "$RUN/root-inputs-1/inputs.json" \
  --reference "$RUN/root-public-1/reference/reference.json"
```

Native launcher uses the unchanged reviewed cleanup functions, exec-FD digest,
four CPUs, finite wall/address-space/log caps, full source-before/after audit,
both models serially and immutable input/reference/source closure.

## Unfinished

- ROOT source review, compilation, strict/fixture tests, public oracle and native
  trained-source320-word proof are all unexecuted by the author.
- See `EXP_BOUNDARY.md`: admitted exact generic exp still required to finish
  coefficients and dynamic initial/cached core exponentials.
- No full GDN/FLA, all42 full-HF graph,27B model, device/serving or performance
  qualification follows from this prefix.

````

