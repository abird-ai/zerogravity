# Run Notes — volume 6

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/README.md

Original UTF-8 bytes: 4370; SHA256 `80f777aeb5541516307487bdf08a07ba91727dcbeda91b95a2750fc10230cb4a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU resident integration — source proposal

Author: batch_design_review. Parent: serving-and-speculation P12.
SOURCE ONLY. No MAIN changes, builds/tests/runtime/model/GPU/operator work,
binary/DSO/model/raw-trace payload reads or performance qualification.

Base: MAIN fb68d8fa plus ROOT's test-only ordered4591967e composition. Original
ordered freezea7939 and qualifierV2 70c594 remain immutable. This proposal is
conditional on ROOT completing their independent operator/PTX/SASS/fault gates.

## Narrow change

- Four-path `NEW_DELTA.patch`: original B1 and native batch parents, new shared
  resident SiLU emitter and2 source-free geometry fixtures. No model schedule.
- Separate `ORDERED_GEOMETRY_SEAM.patch`: pure geometry query +header wording in
  ROOT-owned ordered module. Checked binding/launch/shader body is byte-identical.
- Separate `ROOT_WIRING.patch`: ROOT-owned lib changes module cfg from test-only
  qualification+CUDA to CUDA. NEVER apply until physical promotion is approved.
- No persistent field/table/Vec/Box, new handle, payload, upload or metadata
  allocation. Existing ResourceArbiter classes/preflight/scratch/leases unchanged;
  added exact metadata reservation is0B. Temporary bindings use stack arrays and
  clones of existing Arc handles. Failure strings do not allocate payload storage.

## Cold choice and original flight

- Only original GGUF/F32 full plans qualify; existing typed/HF/operator-slice
  refusal remains. Native service selection is the same original CUDA Client.
  Vulkan and known out-of-geometry CUDA rows keep the exact original emitter.
- B1 construction preflights every original SiLU, including StateOnly-skipped
  tail nodes. Batch construction does the same with actual2/4/8 rows. Existing
  initialization has its own owner/Ready; cold checks inspect metadata, never
  uninitialized values or extra events.
- Existing attempt validation preflights all SiLU nodes BEFORE the enclosing
  original math submission. Batch checks once with actual rows and then invokes
  the original per-row state/range validation body. That body is byte-identical.
- Immediately before a SiLU node's first interval/body launch, repeat the SAME
  checked `PreparedOrderedSilu::prepare`. No cached pointer/foreign binding or
  mutable origin exists: same service, immutable plan IDs, whole ranges, widths
  and handles; actual allocator address remains resolved by the original queue.
  A surprising error propagates through existing submission failure/quarantine.
  Errors never select fallback, and nothing retries after a partial launch.
- Row width is packed input elements / actual rows. B8×65536 stays supported;
  aggregate524288 is never flattened into a fictitious oversized B1 row.
  Inconsistent partitions refuse. No request padding or new authority exists.

## Arithmetic and ownership

- Original interval certificate[-256,256] launches first and never clears status.
  Original outer finite certificates, causal state/version order and status OR
  remain untouched. Original exact independent per-row StateRoots, Ready event,
  cancellation, leases, source ownership and unknown quarantine remain owners.
- Supported CUDA node uses interval +one ordered kernel, incorporating the same
  optional multiplier. Reported source launches are2 rather than13/14. Per
  selected node this removes11/12 source sites; no hardware count or speed claim.
  Projection-only `fused_dispatches` meaning and sampler counters are unchanged.
- Out-of-geometry/Vulkan interval,12 steps, gate and return are source-identical
  modulo indentation to the original arm. STATIC.json records exact normalized
  fragments; original state-validator body and ordered checked_span→EOF compare
  byte-identically. Existing scratch remains allocated/retained even when unused.
- Graph diagnostic uses the same body if ROOT reruns it; this adds no production
  graph eligibility, capture/stream override, queue, fence or Ready authority.

Successful stack preparation was audited against current CubeCL Handle.clone /
ManagedMemoryHandle.clone / binding sources: existing Arc clones and POD only,
no Vec reserve/collect/upload or provider command. It retains no raw device pointer.

Root execution sequence/meaningful integration gates: ROOT.md. Source-only
formatting/static comparisons are not CUDA numerical or runtime evidence.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/ROOT.md

Original UTF-8 bytes: 3531; SHA256 `e2217183d9a9e0707cee9129ddd6c6356cf4c3147a547283df76002fde52f37d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT promotion gates — no execution by author

1. Finish immutable ordered a7939 / qualifier70c594 / ROOT459 operator gate:
   actual PTX/SASS instruction audit, all204 independent old-GPU/CPU raw-bit
   adversaries, signed zero/subnormal/NaN/status/canary/refusal and fresh unknown
   completion/cancellation controls. Keep failed attempts. No test marker alone
   enables this integration.
2. Source-review all3 separate patches against BASELINE.json. Preserve current
   parent changes when composing; create an explicit derivative if bytes differ.
   Copy source with fresh mtimes. ROOT owns lib/ordered and promotion decision.
   NEW_DELTA changes only4 resident paths. No new budget/class/lease or source
   profile is needed. Do not remove old scratch or old fallback in this phase.
3. Strict CUDA/Vulkan/default checks and original units, including2 new tests:
   `resident_program::silu::tests::packed_domains_keep_actual_rows_instead_of_flattening_to_b1`
   and `only_known_geometry_uses_original_route_and_partition_changes_refuse`.
   These test source transport/selection only, not physical numerical support.
4. Reuse existing meaningful independent integration qualifiers, not new mirrored
   arithmetic tests: both approved models' B1 full logits/ordered layer trace/
   all42 StateRoots (`resident_inference/qualification.rs`); native B2/B4/B8
   fullV/all42 +independent CPU/B1 in `resident_fused_qualification.rs`; original
   greedy packet/tie/EOS/feedback transitions in `resident_fused_greedy_qualification.rs`;
   unequal positions256/257/refills/future-KV zeros; StateOnly prefix/per-teacher
   all42/future fullV; genuine prelaunch pressure, stale roots, Ready cancellation
   and fresh unknown completion. Original46-case matrix/10-operator matrix are
   reusable recipes only: freeze a NEW source/binary and retain NEW receipts.
5. Keep independent old-GPU operator oracle distinct: new B1 and new native
   paths both use ordered SiLU, so agreement between those alone cannot qualify
   replacing the old leaf. Independent original CPU/component proof is required.
   Existing physical alias/range audit must still show unchanged exact logical
   allocation/scratch/lease totals; no uninitialized skipped tail becomes Ready.
6. Check actual source dispatch receipt deltas against active original SiLU nodes:
   saving =11×supported plain nodes +12×supported multiplied nodes per executed
   plan body. Use StateOnly's actual prefix node span when relevant. Original
   interval/outer certificates remain in both counts. Do not call launch-site
   arithmetic a driver-enumerated kernel count or infer speed from the saving.
7. Fresh serving worker14/HTTP12 regressions, both approved original cache-direct
   and HF-import release routes, and full isolated gate:

   `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`

   HTTP fused behavior/source profile/metrics and graph default stay unchanged.
   Preserve CPU/Vulkan/typed-HF refusal coverage and Renoir/gfx90c hold.
8. Only after those gates, fresh explicit native baseline/candidate warmed C1/C8
   A/B using unchanged canonical47-ID/max8 workload and original ownership/caps.
   Bind new completed serving/source/build receipts; never overwrite accepted
   old A/B or put either native executable in a vLLM label. Separate accepted
   CUPTI diagnostic can establish real activity counts; unprofiled timings remain
   separate. No improvement, vLLM parity or production full-HF claim beforehand.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/TODO.md

Original UTF-8 bytes: 994; SHA256 `5a39d10f00d37c1e59b29e11e8e2f8828714cebe08d6042b7ec578eb10d6b484`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU integration preparation

Parent: serving-and-speculation P12. Owner: batch_design_review.

- [x] Audit original B1/native batch emitter, cold validation, physical owners and CubeCL handle clones.
- [x] Prepare own no-table source proposal preserving exact old body and budgets.
- [x] Prepare2 source-free geometry tests and existing-qualifier integration recipe.
- [x] Rustfmt/source comparisons only; no compilation/test/runtime.
- [ ] Independent source review/ROOT composition.
- [ ] ROOT actual operator+PTX/SASS+fault prerequisite, strict/full-model/worker/HTTP/routes/full gate.
- [ ] ROOT new native A/B and separate hardware diagnostic before performance claim.

Owned scope only this ignored run. No MAIN/shared-plan/other freeze edits/commit.
Initial static comparator selected the binding multiplier condition rather than
the emitter condition; corrected selector then verified unchanged emitter source.
No arithmetic change or executed numerical result was involved.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/followup-visibility-v2/README.md

Original UTF-8 bytes: 884; SHA256 `215b8c4aa5611efcaf5a3c9e58df97cf2d17930f3bb196862c7680e51d628310`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU integration visibility followup

- Parent9110 is immutable. Source-only correction to its new child module.
- `preflight_silu` and `launch_silu`: private → `pub(super)` so original parent and sibling batch emitter can call them. `ordered_silu_binding` stays private. No body, arithmetic, buffers, queue, status, admission, Ready or quarantine changes.
- Apply9110's resident integration plus this patch. Compose its separate geometry seam onto ROOT's scalar-corrected ordered module3d789c; **never replace it with9110's old full ordered module**, which predates all15 scalar materializations.
- Parent and frozen source rows rechecked; removing exactly two visibility additions restores parent child bytes. No imports, test/build/operator/model/GPU/runtime by author. Compilation/full numerical/physical/gate/performance qualification remains ROOT-owned and pending.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/followup-visibility-v2/TODO.md

Original UTF-8 bytes: 463; SHA256 `7795b115cb73ce682f04ddb83186e1c292c90c069578cd492c3581afec33d0aa`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Visibility followup

Parent: serving-and-speculation. Owner: batch_design_review. Scope: this ignored followup only.

- [x] Preserve all20 parent live/frozen rows and9110 manifest.
- [x] Apply exactly two `pub(super)` method visibilities; private binding unchanged.
- [x] Reversal/static body identity and bounded diff checked.
- [ ] ROOT compile/physical integration acceptance remains pending.

No MAIN/shared-plan changes, commits, tests, builds or runtime.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/followup-visibility-v2/frozen/README.md

Original UTF-8 bytes: 884; SHA256 `215b8c4aa5611efcaf5a3c9e58df97cf2d17930f3bb196862c7680e51d628310`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU integration visibility followup

- Parent9110 is immutable. Source-only correction to its new child module.
- `preflight_silu` and `launch_silu`: private → `pub(super)` so original parent and sibling batch emitter can call them. `ordered_silu_binding` stays private. No body, arithmetic, buffers, queue, status, admission, Ready or quarantine changes.
- Apply9110's resident integration plus this patch. Compose its separate geometry seam onto ROOT's scalar-corrected ordered module3d789c; **never replace it with9110's old full ordered module**, which predates all15 scalar materializations.
- Parent and frozen source rows rechecked; removing exactly two visibility additions restores parent child bytes. No imports, test/build/operator/model/GPU/runtime by author. Compilation/full numerical/physical/gate/performance qualification remains ROOT-owned and pending.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/followup-visibility-v2/frozen/TODO.md

Original UTF-8 bytes: 463; SHA256 `7795b115cb73ce682f04ddb83186e1c292c90c069578cd492c3581afec33d0aa`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Visibility followup

Parent: serving-and-speculation. Owner: batch_design_review. Scope: this ignored followup only.

- [x] Preserve all20 parent live/frozen rows and9110 manifest.
- [x] Apply exactly two `pub(super)` method visibilities; private binding unchanged.
- [x] Reversal/static body identity and bounded diff checked.
- [ ] ROOT compile/physical integration acceptance remains pending.

No MAIN/shared-plan changes, commits, tests, builds or runtime.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/frozen/README.md

Original UTF-8 bytes: 4370; SHA256 `80f777aeb5541516307487bdf08a07ba91727dcbeda91b95a2750fc10230cb4a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU resident integration — source proposal

Author: batch_design_review. Parent: serving-and-speculation P12.
SOURCE ONLY. No MAIN changes, builds/tests/runtime/model/GPU/operator work,
binary/DSO/model/raw-trace payload reads or performance qualification.

Base: MAIN fb68d8fa plus ROOT's test-only ordered4591967e composition. Original
ordered freezea7939 and qualifierV2 70c594 remain immutable. This proposal is
conditional on ROOT completing their independent operator/PTX/SASS/fault gates.

## Narrow change

- Four-path `NEW_DELTA.patch`: original B1 and native batch parents, new shared
  resident SiLU emitter and2 source-free geometry fixtures. No model schedule.
- Separate `ORDERED_GEOMETRY_SEAM.patch`: pure geometry query +header wording in
  ROOT-owned ordered module. Checked binding/launch/shader body is byte-identical.
- Separate `ROOT_WIRING.patch`: ROOT-owned lib changes module cfg from test-only
  qualification+CUDA to CUDA. NEVER apply until physical promotion is approved.
- No persistent field/table/Vec/Box, new handle, payload, upload or metadata
  allocation. Existing ResourceArbiter classes/preflight/scratch/leases unchanged;
  added exact metadata reservation is0B. Temporary bindings use stack arrays and
  clones of existing Arc handles. Failure strings do not allocate payload storage.

## Cold choice and original flight

- Only original GGUF/F32 full plans qualify; existing typed/HF/operator-slice
  refusal remains. Native service selection is the same original CUDA Client.
  Vulkan and known out-of-geometry CUDA rows keep the exact original emitter.
- B1 construction preflights every original SiLU, including StateOnly-skipped
  tail nodes. Batch construction does the same with actual2/4/8 rows. Existing
  initialization has its own owner/Ready; cold checks inspect metadata, never
  uninitialized values or extra events.
- Existing attempt validation preflights all SiLU nodes BEFORE the enclosing
  original math submission. Batch checks once with actual rows and then invokes
  the original per-row state/range validation body. That body is byte-identical.
- Immediately before a SiLU node's first interval/body launch, repeat the SAME
  checked `PreparedOrderedSilu::prepare`. No cached pointer/foreign binding or
  mutable origin exists: same service, immutable plan IDs, whole ranges, widths
  and handles; actual allocator address remains resolved by the original queue.
  A surprising error propagates through existing submission failure/quarantine.
  Errors never select fallback, and nothing retries after a partial launch.
- Row width is packed input elements / actual rows. B8×65536 stays supported;
  aggregate524288 is never flattened into a fictitious oversized B1 row.
  Inconsistent partitions refuse. No request padding or new authority exists.

## Arithmetic and ownership

- Original interval certificate[-256,256] launches first and never clears status.
  Original outer finite certificates, causal state/version order and status OR
  remain untouched. Original exact independent per-row StateRoots, Ready event,
  cancellation, leases, source ownership and unknown quarantine remain owners.
- Supported CUDA node uses interval +one ordered kernel, incorporating the same
  optional multiplier. Reported source launches are2 rather than13/14. Per
  selected node this removes11/12 source sites; no hardware count or speed claim.
  Projection-only `fused_dispatches` meaning and sampler counters are unchanged.
- Out-of-geometry/Vulkan interval,12 steps, gate and return are source-identical
  modulo indentation to the original arm. STATIC.json records exact normalized
  fragments; original state-validator body and ordered checked_span→EOF compare
  byte-identically. Existing scratch remains allocated/retained even when unused.
- Graph diagnostic uses the same body if ROOT reruns it; this adds no production
  graph eligibility, capture/stream override, queue, fence or Ready authority.

Successful stack preparation was audited against current CubeCL Handle.clone /
ManagedMemoryHandle.clone / binding sources: existing Arc clones and POD only,
no Vec reserve/collect/upload or provider command. It retains no raw device pointer.

Root execution sequence/meaningful integration gates: ROOT.md. Source-only
formatting/static comparisons are not CUDA numerical or runtime evidence.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/frozen/ROOT.md

Original UTF-8 bytes: 3531; SHA256 `e2217183d9a9e0707cee9129ddd6c6356cf4c3147a547283df76002fde52f37d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT promotion gates — no execution by author

1. Finish immutable ordered a7939 / qualifier70c594 / ROOT459 operator gate:
   actual PTX/SASS instruction audit, all204 independent old-GPU/CPU raw-bit
   adversaries, signed zero/subnormal/NaN/status/canary/refusal and fresh unknown
   completion/cancellation controls. Keep failed attempts. No test marker alone
   enables this integration.
2. Source-review all3 separate patches against BASELINE.json. Preserve current
   parent changes when composing; create an explicit derivative if bytes differ.
   Copy source with fresh mtimes. ROOT owns lib/ordered and promotion decision.
   NEW_DELTA changes only4 resident paths. No new budget/class/lease or source
   profile is needed. Do not remove old scratch or old fallback in this phase.
3. Strict CUDA/Vulkan/default checks and original units, including2 new tests:
   `resident_program::silu::tests::packed_domains_keep_actual_rows_instead_of_flattening_to_b1`
   and `only_known_geometry_uses_original_route_and_partition_changes_refuse`.
   These test source transport/selection only, not physical numerical support.
4. Reuse existing meaningful independent integration qualifiers, not new mirrored
   arithmetic tests: both approved models' B1 full logits/ordered layer trace/
   all42 StateRoots (`resident_inference/qualification.rs`); native B2/B4/B8
   fullV/all42 +independent CPU/B1 in `resident_fused_qualification.rs`; original
   greedy packet/tie/EOS/feedback transitions in `resident_fused_greedy_qualification.rs`;
   unequal positions256/257/refills/future-KV zeros; StateOnly prefix/per-teacher
   all42/future fullV; genuine prelaunch pressure, stale roots, Ready cancellation
   and fresh unknown completion. Original46-case matrix/10-operator matrix are
   reusable recipes only: freeze a NEW source/binary and retain NEW receipts.
5. Keep independent old-GPU operator oracle distinct: new B1 and new native
   paths both use ordered SiLU, so agreement between those alone cannot qualify
   replacing the old leaf. Independent original CPU/component proof is required.
   Existing physical alias/range audit must still show unchanged exact logical
   allocation/scratch/lease totals; no uninitialized skipped tail becomes Ready.
6. Check actual source dispatch receipt deltas against active original SiLU nodes:
   saving =11×supported plain nodes +12×supported multiplied nodes per executed
   plan body. Use StateOnly's actual prefix node span when relevant. Original
   interval/outer certificates remain in both counts. Do not call launch-site
   arithmetic a driver-enumerated kernel count or infer speed from the saving.
7. Fresh serving worker14/HTTP12 regressions, both approved original cache-direct
   and HF-import release routes, and full isolated gate:

   `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`

   HTTP fused behavior/source profile/metrics and graph default stay unchanged.
   Preserve CPU/Vulkan/typed-HF refusal coverage and Renoir/gfx90c hold.
8. Only after those gates, fresh explicit native baseline/candidate warmed C1/C8
   A/B using unchanged canonical47-ID/max8 workload and original ownership/caps.
   Bind new completed serving/source/build receipts; never overwrite accepted
   old A/B or put either native executable in a vLLM label. Separate accepted
   CUPTI diagnostic can establish real activity counts; unprofiled timings remain
   separate. No improvement, vLLM parity or production full-HF claim beforehand.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/frozen/TODO.md

Original UTF-8 bytes: 994; SHA256 `5a39d10f00d37c1e59b29e11e8e2f8828714cebe08d6042b7ec578eb10d6b484`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU integration preparation

Parent: serving-and-speculation P12. Owner: batch_design_review.

- [x] Audit original B1/native batch emitter, cold validation, physical owners and CubeCL handle clones.
- [x] Prepare own no-table source proposal preserving exact old body and budgets.
- [x] Prepare2 source-free geometry tests and existing-qualifier integration recipe.
- [x] Rustfmt/source comparisons only; no compilation/test/runtime.
- [ ] Independent source review/ROOT composition.
- [ ] ROOT actual operator+PTX/SASS+fault prerequisite, strict/full-model/worker/HTTP/routes/full gate.
- [ ] ROOT new native A/B and separate hardware diagnostic before performance claim.

Owned scope only this ignored run. No MAIN/shared-plan/other freeze edits/commit.
Initial static comparator selected the binding multiplier condition rather than
the emitter condition; corrected selector then verified unchanged emitter source.
No arithmetic change or executed numerical result was involved.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/API_FACTS.md

Original UTF-8 bytes: 5681; SHA256 `db0611585928ce815ca3b3cdb67aaa05cb7b7a4cd86445094a40f661daddab4f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# API and source facts

## Original route

- `crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/context.rs:417`:
  compile_to_ptx supplies the source-selected `sm_XX` architecture, include
  options and `-lineinfo` to NVRTC, then calls get_ptx. It does not extract cubin.
- Same file:473–516: load_ptx retains the NVRTC PTX Vec through the original
  `module::load_data(ptx.as_ptr())` and `module::get_function(module,name)`,
  then retains the resulting function in the original kernel cache.
- Same file:528–566: execute_task uses that cached function, original stream,
  dimensions/shared-memory metadata and original resource pointers. It first
  calls cuFuncSetAttribute, then the original launch_kernel.
- Same file:588–609: CUBECL_CUDA_DUMP_PTX writes the submitted PTX, but the
  filename sanitizes and truncates the kernel key; I/O errors are ignored.
  A dump file alone is not a collision-free complete submission attestation.
- Original ordered source has one `ordered::launch_unchecked` dispatch, with
  compile-time count/multiplied/record. Plain/gated and compact/traced variants
  can produce distinct actual modules. Bind the actual specialization, not a
  substring, filename suffix or a hand-written replacement kernel.

The Cargo.lock-selected cached cudarc0.19.10 driver/result.rs:1165–1185
confirms load_data calls cuModuleLoadData and get_function calls
cuModuleGetFunction; :1325–1345 confirms launch_kernel calls cuLaunchKernel.
The wrapper source is pinned separately; no guessed alternate driver entry.

## Pinned CUPTI 12.9.79 / API 28 headers

Local header directory is in source-pins.json. No manually declared replacement
ABI should be used. A future ROOT build must include these exact headers and
static_assert the actual native layouts, rather than trust this LP64 sketch.

- cupti_callbacks.h:123,177–185: resource domain 3; MODULE_LOADED 6,
  MODULE_UNLOAD_STARTING 7, MODULE_PROFILED 8.
- :397–429 CUpti_ResourceData: context, union containing stream,
  resourceDescriptor. The descriptor comment says reserved for future use.
- :432–458 CUpti_ModuleResourceData: uint32_t moduleId; size_t cubinSize;
  const char *pCubin. Data lifetime ends with the callback. Expected LP64
  offsets 0/8/16 and size 24 are a proposal, NOT a compiled layout proof.
- :627–640 generic callback docs select CUpti_ResourceData for resource domain;
  the module-specific brief instead describes ModuleResourceData as cbdata.
  The exact wrapper/descriptor path therefore remains CLOSED here. Neither
  speculative direct casting nor trying two layouts is safe. Pin a genuine
  NVIDIA 12.9 sample showing the module callback dispatch before implementing
  any dereference. No exact-version primary sample was obtained in this audit.
- :319–394 CUpti_CallbackData exposes callbackSite, functionParams,
  functionReturnValue (only API_EXIT), context/contextUid, correlationData and
  correlationId. Only successful original API_EXIT values bind returned handles.
- :682–724 subscribe supports one subscriber; a conflict refuses the diagnostic.
  Unsubscribe promises no future callbacks, not an explicit in-flight drain.
  Retain callback state through owned process retirement unless safe drain has
  a separately proved contract. Do not call cuptiFinalize as casual cleanup.
- cupti_activity.h:4711–4752 Function records contain moduleId and functionIndex.
  :4758–4796 explicitly says ActivityModule is NOT emitted by the Activity API.
  Enabling MODULE activity is not an extraction implementation.

The public API documentation also states the callback lifetime. Its resource
and module descriptions preserve the same dispatch ambiguity as the header.
[ModuleResourceData](https://docs.nvidia.com/cupti/12.9/api/structCUpti__ModuleResourceData.html),
[ResourceData](https://docs.nvidia.com/cupti/12.9/api/structCUpti__ResourceData.html),
[callback API](https://docs.nvidia.com/cupti/12.9/api/group__CUPTI__CALLBACK__API.html)

## Original driver callback metadata

Generated CUDA parameter structs, not guessed ABI:

| Original call | Pinned callback ID | Permitted observation |
| --- | ---: | --- |
| cuModuleLoadData | 19 | input PTX identity; successful returned CUmodule |
| cuModuleGetFunction | 23 | original module/name; successful CUfunction |
| cuLaunchKernel | 307 | original function/stream/grid/block/shared bytes/status |
| cuModuleUnload | 22 | generation termination; never observer-triggered |

Parameter definitions are generated_cuda_meta.h:270,296,3348,288.
`kernelParams`/`extra` device inputs are not decoded, copied or altered. Return
values are observed without replacing them. A source/runtime change to another
entrypoint (e.g. ptsz, LaunchKernelEx, graph launch) needs a new audited route;
unknown callbacks cannot be counted as this route.

The pinned driver header has public module/function metadata APIs but no
identified public API returning a loaded module's whole cubin. No extra driver
query is required or authorized by this advisory. The callback is the selected
extraction route, with the original driver calls providing correlation.

## Tool availability boundary

Existing exact-image inspect metadata records installation of
`cuda-cuobjdump-12-9` alongside NVCC/NVRTC. This is package history, not proof of
an actual current tool file. ROOT may inspect the preinstalled
`/usr/local/cuda-12.9/bin/cuobjdump` path/target/version/hash and dependencies.
No tool invocation or binary read/hash occurred in this audit. nvdisasm is not
established present. Either accepted tool only disassembles the captured image;
NVRTC get_cubin, offline ptxas and recompilation do not prove driver-loaded SASS.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/DESIGN.md

Original UTF-8 bytes: 6455; SHA256 `a088031ff7bd088959f0938785dffc9a4baa8e56991eb3877de958ceccf83f6a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Smallest diagnostic boundary

## Keep the original execution

Use a fresh owned process running the existing ordered-SiLU physical qualifier.
Load the approved CUPTI observer before the original CUDA initialization/module
load, using existing ownership/identity/cleanup supervision. No alternate CUDA
context, queue, stream, module, kernel, provider, resource manager or executor.
No vendor/production capture changes and no privileged profiler changes.

Subscribe only to the reviewed resource callbacks and original driver metadata
callbacks. Enable no Activity API, PC sampling, SASS metrics, replay, range
profiling or kernel-timing instrumentation. The existing capacity collector
explicitly enables CONCURRENT_KERNEL; it must remain a separate diagnostic.
NVIDIA documents that concurrent-kernel timing inserts instrumentation at
module load, so its extracted image cannot silently stand in for an image from
this callback-only configuration. [CUPTI 12.9 tracing overhead](https://docs.nvidia.com/cupti/12.9/main/main.html#tracing-overhead)

Observer presence still affects host execution and allocation. Exact output
transparency is required; the configuration is not a timed-performance sample
and does not independently prove zero internal driver changes.

## Exact association, with refusal on ambiguity

Required identity chain:

`owned process nonce + exact provider/mapped-file identity + contextUid`
→ original successful LoadData and submitted PTX bytes
→ one associated resource callback moduleId/cubin
→ original successful GetFunction(CUmodule, exact entrypoint) returning CUfunction
→ original successful LaunchKernel(CUfunction, original stream and geometry)
→ the original physical completion/Ready + complete numerical/readset proof.

ResourceData/ModuleResourceData does not publicly supply a guaranteed CUmodule
handle relation in the inspected fields. Do not equate a moduleId with a handle.
For the smallest first slice, require the module callback to be uniquely nested
on the same observed driver-callback thread in the original LoadData interval;
record enter/exit/correlation and module generation. This nesting is a required
ACTUAL observation, not an assumed ordering guarantee. If lazy loading delays it
or callbacks are asynchronous/multiple/ambiguous, retain evidence and return
UNQUALIFIED. Never change CUDA_MODULE_LOADING or force a new load to manufacture
association. An expanded GetFunction/launch nesting rule needs separate review.

Copy returned module/function handles only on CUDA_SUCCESS. Track original
unload and context lifetime; handle or moduleId reuse creates a new generation.
Record every relevant module/function/launch event, not just the first name
containing ordered. Cold process + bounded explicit target specializations
prevents interpreting an already-cached/no-callback module as a capture.

Submitted PTX is known to be a NUL-terminated Vec in this audited original call.
A future observer must use a fixed bounded text scan only in this source-bound
route, require NUL before cap and reject non-PTX/foreign call origins. Copy while
the original Vec is live. Independently bind its exact entrypoint and bytes to
captured module/function metadata. Existing lossy dump filenames are at most a
cross-check; collision/missing/truncated/replaced files refuse acceptance.

## Bounded copy and ownership proposal

Predeclare before CUDA initialization (not adaptive after overflow):

- 16 module slots × 2 MiB copied cubin = 32 MiB.
- 16 PTX slots × 2 MiB = 32 MiB.
- 4 MiB fixed metadata/ring/name storage, at most 4096 metadata events.
- Total observer-owned arena 68 MiB; exact allocations charged/retained under
  a reviewed host ticket before subscription and copies. An explicit process
  RSS/AS/artifact bound contains unowned CUPTI/driver overhead; this arena is
  not a whole-process RAM claim. If foreign observer storage is not actually
  admitted to ResourceArbiter, label it separately and do not invent a lease.

These are proposed bounds for one operator diagnostic, not observed image sizes
or capacity qualification. A target larger than a bound produces an explicit
loss/overflow refusal. No partial cubin, overwritten slot or silent cap increase.

Callback body: preserve errno; reserve a fixed slot atomically, validate known
ABI/ID/size/non-null payload, bounded memcpy and scalar/name metadata copy,
publish with release ordering, restore errno. No malloc/vector growth, file I/O,
formatting, CUDA/CUPTI calls, waits, driver destruction or mutex needing another
callback. Copy pCubin before returning; never retain borrowed callback pointers.
If concurrent callbacks exceed a declared nesting/event bound, record failure.

After original qualifier completion and safe observer quiescence, the process
can hash/write copied bytes to exclusive files in a fresh owned directory.
Release observer state only after no callback can reference it. Prefer retaining
it until owned process exit for this first diagnostic; normal OS retirement is
bounded by the existing supervisor. No observer Drop performs driver calls or
attempts recovery after unknown completion. Original physical leases/Ready/
quarantine remain sole authorities, not the observer's callback records.

## Offline inspection and truthful claims

Validate a complete ELF cubin with checked headers/section bounds, expected
NVIDIA/SM86 metadata and exact target function symbol before disassembly. Pin
the copied file, tool/closure, argv and whole output. Run an already present
pinned cuobjdump/nvdisasm in a fresh CPU-only owned child on that file, without
recompiling or loading it into CUDA. Bound child wall/log/output, preserve error
and independent cleanup receipts, and reserve final metadata before writes.

Disassembly can inspect FMA/mul/add/cvt rounding and unwanted approximate
transcendentals for the loaded target image. Static opcode checks require the
complete target function/control-flow body and source mapping; opcode presence
alone cannot prove the arithmetic schedule or absence of alternate paths.

Allowed eventual claim: captured callback-provided JIT image, uniquely bound to
the original successful function launch and accepted original completion;
disassembled target function instruction schedule. It is not a PC/instruction
execution trace, proof every instruction executed, hardware kernel count,
unprofiled runtime identity, whole-model parity or a performance result.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/README.md

Original UTF-8 bytes: 1497; SHA256 `f1cbbbdb76240a781e71237d2e84e42c6b371af8cd3f9e847bd7c2b7f48899d7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original ordered-SiLU loaded SASS — source-only advisory

Author: spine_review. Parent: serving-and-speculation P12.

A callback-only CUPTI diagnostic can capture the JIT cubin associated with the
original CubeCL module, without compiling/loading another kernel. No collector,
launcher, provider or production change is proposed here. Current status:
**SOURCE_FEASIBLE / CALLBACK_ABI_AND_PHYSICAL_BINDING_UNQUALIFIED**.

NVIDIA documents the module resource callback as a route to the JIT cubin for
offline disassembly. This establishes feasibility, not the actual callback ABI,
module association, binary identity or successful instruction execution for
this repository. [CUPTI 12.9 SASS source correlation](https://docs.nvidia.com/cupti/12.9/main/main.html#sass-source-correlation)

Read API_FACTS.md for pinned local source/header anchors, DESIGN.md for the
small diagnostic boundary, and ROOT_GATES.md for the remaining source/runtime
requirements. source-pins.json hashes only source/header/metadata. The approved
CUPTI library identity is copied from existing metadata, not reread or rehashed.

No Cargo/compiler/Torch/model/provider/operator/GPU/container execution, binary
scan/disassembly, model/archive/payload read/hash, downloads, MAIN/vendor/shared
plan changes or old-freeze mutation by this author. Existing named tasks and
qualifiers remain authoritative; this advisory does not qualify ordered SiLU,
full-model numerics, throughput, kernel counts or production graph capture.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/ROOT_GATES.md

Original UTF-8 bytes: 2983; SHA256 `77a903d628bb06da5c46e1ec8190f7defcb6311a81d3a8c1e2431f680199be8d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Remaining ROOT gates

1. Source first: locate a genuine NVIDIA CUPTI 12.9 SDK sample/source showing
   exact module callback payload dispatch. Pin source/license/version and resolve
   ResourceData vs ModuleResourceData. Do not implement a dual caster. If no
   exact primary binding can be obtained, cubin pointer dereference stays CLOSED.
2. Review a minimal callback-only implementation against the actual approved
   headers. Compile real static_assert layouts and fixed-cap overflow checks;
   keep all notices. No copied third-party ABI accepted as primary closure.
3. Pin the actual helper/source/binary and approved CUPTI/driver mappings from
   verified retained FDs and actual mapping identity. No path-only/provider-name
   attestation. Single-subscriber conflict/lost callbacks/unknown IDs refuse.
4. ROOT may inspect actual preinstalled cuobjdump metadata, then separately
   authenticate its whole file/runtime closure before offline invocation. Image
   install history alone is not a verified tool. No acquisition is needed unless
   ROOT later determines the existing image lacks a usable tool.
5. Prepare source-free fixed-storage fixtures: oversize/zero/null/unknown module,
   slot overflow, missing/PTX NUL, reuse/unload/context generation, duplicate/
   foreign/missing association, failed original LoadData/GetFunction/Launch,
   concurrent/nested callback bounds, truncated file/ELF and wrong target symbol.
   Fixture arithmetic/driver stubs cannot qualify real loaded image semantics.
6. Fresh original plain qualifier, then fresh callback-only observed qualifier,
   identical source/input/profile/geometry/hardware/runtime/environment and
   original queue. All original output/intermediate/status/canary/read-only/
   leases/cancellation/Ready/final-budget proofs remain required. Do not label
   host launch callbacks as hardware activity or successful enqueue as Ready.
7. Actual trace must establish unique resource→original load/function/launch
   association and complete copied cubin/PTX; independently validate counts/
   generation/source bindings. Missing/ambiguous lazy-loading association is
   UNQUALIFIED, not a reason to force CUDA eager-loading or another module.
8. Unknown completion uses the original quarantine; no recovery, destroy or
   freeing original owners. A callback payload does not prove GPU completion.
   Fault/abandon tests remain original fresh-process routes; capture is not a
   requirement to extract an image from an unsafe unknown owner.
9. CPU-only disassembly of the exact captured file and bounded whole output,
   exact target function/control-flow inspection, transparent output validator,
   all owned process/group/pipe retirement and final evidence tree bounds pass
   before accepting loaded-image evidence. No timing or production promotion.

No ROOT step was run by the advisory author. Current evidence is pinned source
and primary documentation only; runtime/ABI/module-association/tool proof open.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/TODO.md

Original UTF-8 bytes: 1147; SHA256 `141c9c9ad4f2decdf626e742b93d9aa23ef2eee14bacd1e5968641a82f375dcd`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: serving-and-speculation P12. Owner: spine_review.
Scope: source-only advisory for actual original ordered-SiLU loaded SASS.

- [x] Inspect original CubeCL NVRTC→PTX→load_data→GetFunction→launch path.
- [x] Inspect approved CUPTI source headers/API28, actual callback parameter
  metadata and source-only existing collector/tool install metadata.
- [x] Check official NVIDIA12.9 documentation; identify supported extraction
  route, callback-only boundary and unresolved exact payload dispatch.
- [x] Prepare minimal bounded copy/correlation/lifetime/validation proposal.
- [x] Preserve all MAIN/vendor/old freeze bytes; source/header/metadata only.
- [ ] Exact NVIDIA12.9 sample binding resolves callback payload before coding.
- [ ] ROOT implementation review, layout compilation, fixtures and actual
  original plain/observed numerical/owner/association qualification.
- [ ] ROOT authenticates existing tool and disassembles actual captured image.

No builds/tests/runtime/provider/operator/model/GPU/binary/archive/payload scans
performed. No physical numerical, loaded-SASS or performance acceptance here.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/frozen/API_FACTS.md

Original UTF-8 bytes: 5681; SHA256 `db0611585928ce815ca3b3cdb67aaa05cb7b7a4cd86445094a40f661daddab4f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# API and source facts

## Original route

- `crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/context.rs:417`:
  compile_to_ptx supplies the source-selected `sm_XX` architecture, include
  options and `-lineinfo` to NVRTC, then calls get_ptx. It does not extract cubin.
- Same file:473–516: load_ptx retains the NVRTC PTX Vec through the original
  `module::load_data(ptx.as_ptr())` and `module::get_function(module,name)`,
  then retains the resulting function in the original kernel cache.
- Same file:528–566: execute_task uses that cached function, original stream,
  dimensions/shared-memory metadata and original resource pointers. It first
  calls cuFuncSetAttribute, then the original launch_kernel.
- Same file:588–609: CUBECL_CUDA_DUMP_PTX writes the submitted PTX, but the
  filename sanitizes and truncates the kernel key; I/O errors are ignored.
  A dump file alone is not a collision-free complete submission attestation.
- Original ordered source has one `ordered::launch_unchecked` dispatch, with
  compile-time count/multiplied/record. Plain/gated and compact/traced variants
  can produce distinct actual modules. Bind the actual specialization, not a
  substring, filename suffix or a hand-written replacement kernel.

The Cargo.lock-selected cached cudarc0.19.10 driver/result.rs:1165–1185
confirms load_data calls cuModuleLoadData and get_function calls
cuModuleGetFunction; :1325–1345 confirms launch_kernel calls cuLaunchKernel.
The wrapper source is pinned separately; no guessed alternate driver entry.

## Pinned CUPTI 12.9.79 / API 28 headers

Local header directory is in source-pins.json. No manually declared replacement
ABI should be used. A future ROOT build must include these exact headers and
static_assert the actual native layouts, rather than trust this LP64 sketch.

- cupti_callbacks.h:123,177–185: resource domain 3; MODULE_LOADED 6,
  MODULE_UNLOAD_STARTING 7, MODULE_PROFILED 8.
- :397–429 CUpti_ResourceData: context, union containing stream,
  resourceDescriptor. The descriptor comment says reserved for future use.
- :432–458 CUpti_ModuleResourceData: uint32_t moduleId; size_t cubinSize;
  const char *pCubin. Data lifetime ends with the callback. Expected LP64
  offsets 0/8/16 and size 24 are a proposal, NOT a compiled layout proof.
- :627–640 generic callback docs select CUpti_ResourceData for resource domain;
  the module-specific brief instead describes ModuleResourceData as cbdata.
  The exact wrapper/descriptor path therefore remains CLOSED here. Neither
  speculative direct casting nor trying two layouts is safe. Pin a genuine
  NVIDIA 12.9 sample showing the module callback dispatch before implementing
  any dereference. No exact-version primary sample was obtained in this audit.
- :319–394 CUpti_CallbackData exposes callbackSite, functionParams,
  functionReturnValue (only API_EXIT), context/contextUid, correlationData and
  correlationId. Only successful original API_EXIT values bind returned handles.
- :682–724 subscribe supports one subscriber; a conflict refuses the diagnostic.
  Unsubscribe promises no future callbacks, not an explicit in-flight drain.
  Retain callback state through owned process retirement unless safe drain has
  a separately proved contract. Do not call cuptiFinalize as casual cleanup.
- cupti_activity.h:4711–4752 Function records contain moduleId and functionIndex.
  :4758–4796 explicitly says ActivityModule is NOT emitted by the Activity API.
  Enabling MODULE activity is not an extraction implementation.

The public API documentation also states the callback lifetime. Its resource
and module descriptions preserve the same dispatch ambiguity as the header.
[ModuleResourceData](https://docs.nvidia.com/cupti/12.9/api/structCUpti__ModuleResourceData.html),
[ResourceData](https://docs.nvidia.com/cupti/12.9/api/structCUpti__ResourceData.html),
[callback API](https://docs.nvidia.com/cupti/12.9/api/group__CUPTI__CALLBACK__API.html)

## Original driver callback metadata

Generated CUDA parameter structs, not guessed ABI:

| Original call | Pinned callback ID | Permitted observation |
| --- | ---: | --- |
| cuModuleLoadData | 19 | input PTX identity; successful returned CUmodule |
| cuModuleGetFunction | 23 | original module/name; successful CUfunction |
| cuLaunchKernel | 307 | original function/stream/grid/block/shared bytes/status |
| cuModuleUnload | 22 | generation termination; never observer-triggered |

Parameter definitions are generated_cuda_meta.h:270,296,3348,288.
`kernelParams`/`extra` device inputs are not decoded, copied or altered. Return
values are observed without replacing them. A source/runtime change to another
entrypoint (e.g. ptsz, LaunchKernelEx, graph launch) needs a new audited route;
unknown callbacks cannot be counted as this route.

The pinned driver header has public module/function metadata APIs but no
identified public API returning a loaded module's whole cubin. No extra driver
query is required or authorized by this advisory. The callback is the selected
extraction route, with the original driver calls providing correlation.

## Tool availability boundary

Existing exact-image inspect metadata records installation of
`cuda-cuobjdump-12-9` alongside NVCC/NVRTC. This is package history, not proof of
an actual current tool file. ROOT may inspect the preinstalled
`/usr/local/cuda-12.9/bin/cuobjdump` path/target/version/hash and dependencies.
No tool invocation or binary read/hash occurred in this audit. nvdisasm is not
established present. Either accepted tool only disassembles the captured image;
NVRTC get_cubin, offline ptxas and recompilation do not prove driver-loaded SASS.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/frozen/DESIGN.md

Original UTF-8 bytes: 6455; SHA256 `a088031ff7bd088959f0938785dffc9a4baa8e56991eb3877de958ceccf83f6a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Smallest diagnostic boundary

## Keep the original execution

Use a fresh owned process running the existing ordered-SiLU physical qualifier.
Load the approved CUPTI observer before the original CUDA initialization/module
load, using existing ownership/identity/cleanup supervision. No alternate CUDA
context, queue, stream, module, kernel, provider, resource manager or executor.
No vendor/production capture changes and no privileged profiler changes.

Subscribe only to the reviewed resource callbacks and original driver metadata
callbacks. Enable no Activity API, PC sampling, SASS metrics, replay, range
profiling or kernel-timing instrumentation. The existing capacity collector
explicitly enables CONCURRENT_KERNEL; it must remain a separate diagnostic.
NVIDIA documents that concurrent-kernel timing inserts instrumentation at
module load, so its extracted image cannot silently stand in for an image from
this callback-only configuration. [CUPTI 12.9 tracing overhead](https://docs.nvidia.com/cupti/12.9/main/main.html#tracing-overhead)

Observer presence still affects host execution and allocation. Exact output
transparency is required; the configuration is not a timed-performance sample
and does not independently prove zero internal driver changes.

## Exact association, with refusal on ambiguity

Required identity chain:

`owned process nonce + exact provider/mapped-file identity + contextUid`
→ original successful LoadData and submitted PTX bytes
→ one associated resource callback moduleId/cubin
→ original successful GetFunction(CUmodule, exact entrypoint) returning CUfunction
→ original successful LaunchKernel(CUfunction, original stream and geometry)
→ the original physical completion/Ready + complete numerical/readset proof.

ResourceData/ModuleResourceData does not publicly supply a guaranteed CUmodule
handle relation in the inspected fields. Do not equate a moduleId with a handle.
For the smallest first slice, require the module callback to be uniquely nested
on the same observed driver-callback thread in the original LoadData interval;
record enter/exit/correlation and module generation. This nesting is a required
ACTUAL observation, not an assumed ordering guarantee. If lazy loading delays it
or callbacks are asynchronous/multiple/ambiguous, retain evidence and return
UNQUALIFIED. Never change CUDA_MODULE_LOADING or force a new load to manufacture
association. An expanded GetFunction/launch nesting rule needs separate review.

Copy returned module/function handles only on CUDA_SUCCESS. Track original
unload and context lifetime; handle or moduleId reuse creates a new generation.
Record every relevant module/function/launch event, not just the first name
containing ordered. Cold process + bounded explicit target specializations
prevents interpreting an already-cached/no-callback module as a capture.

Submitted PTX is known to be a NUL-terminated Vec in this audited original call.
A future observer must use a fixed bounded text scan only in this source-bound
route, require NUL before cap and reject non-PTX/foreign call origins. Copy while
the original Vec is live. Independently bind its exact entrypoint and bytes to
captured module/function metadata. Existing lossy dump filenames are at most a
cross-check; collision/missing/truncated/replaced files refuse acceptance.

## Bounded copy and ownership proposal

Predeclare before CUDA initialization (not adaptive after overflow):

- 16 module slots × 2 MiB copied cubin = 32 MiB.
- 16 PTX slots × 2 MiB = 32 MiB.
- 4 MiB fixed metadata/ring/name storage, at most 4096 metadata events.
- Total observer-owned arena 68 MiB; exact allocations charged/retained under
  a reviewed host ticket before subscription and copies. An explicit process
  RSS/AS/artifact bound contains unowned CUPTI/driver overhead; this arena is
  not a whole-process RAM claim. If foreign observer storage is not actually
  admitted to ResourceArbiter, label it separately and do not invent a lease.

These are proposed bounds for one operator diagnostic, not observed image sizes
or capacity qualification. A target larger than a bound produces an explicit
loss/overflow refusal. No partial cubin, overwritten slot or silent cap increase.

Callback body: preserve errno; reserve a fixed slot atomically, validate known
ABI/ID/size/non-null payload, bounded memcpy and scalar/name metadata copy,
publish with release ordering, restore errno. No malloc/vector growth, file I/O,
formatting, CUDA/CUPTI calls, waits, driver destruction or mutex needing another
callback. Copy pCubin before returning; never retain borrowed callback pointers.
If concurrent callbacks exceed a declared nesting/event bound, record failure.

After original qualifier completion and safe observer quiescence, the process
can hash/write copied bytes to exclusive files in a fresh owned directory.
Release observer state only after no callback can reference it. Prefer retaining
it until owned process exit for this first diagnostic; normal OS retirement is
bounded by the existing supervisor. No observer Drop performs driver calls or
attempts recovery after unknown completion. Original physical leases/Ready/
quarantine remain sole authorities, not the observer's callback records.

## Offline inspection and truthful claims

Validate a complete ELF cubin with checked headers/section bounds, expected
NVIDIA/SM86 metadata and exact target function symbol before disassembly. Pin
the copied file, tool/closure, argv and whole output. Run an already present
pinned cuobjdump/nvdisasm in a fresh CPU-only owned child on that file, without
recompiling or loading it into CUDA. Bound child wall/log/output, preserve error
and independent cleanup receipts, and reserve final metadata before writes.

Disassembly can inspect FMA/mul/add/cvt rounding and unwanted approximate
transcendentals for the loaded target image. Static opcode checks require the
complete target function/control-flow body and source mapping; opcode presence
alone cannot prove the arithmetic schedule or absence of alternate paths.

Allowed eventual claim: captured callback-provided JIT image, uniquely bound to
the original successful function launch and accepted original completion;
disassembled target function instruction schedule. It is not a PC/instruction
execution trace, proof every instruction executed, hardware kernel count,
unprofiled runtime identity, whole-model parity or a performance result.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/frozen/README.md

Original UTF-8 bytes: 1497; SHA256 `f1cbbbdb76240a781e71237d2e84e42c6b371af8cd3f9e847bd7c2b7f48899d7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original ordered-SiLU loaded SASS — source-only advisory

Author: spine_review. Parent: serving-and-speculation P12.

A callback-only CUPTI diagnostic can capture the JIT cubin associated with the
original CubeCL module, without compiling/loading another kernel. No collector,
launcher, provider or production change is proposed here. Current status:
**SOURCE_FEASIBLE / CALLBACK_ABI_AND_PHYSICAL_BINDING_UNQUALIFIED**.

NVIDIA documents the module resource callback as a route to the JIT cubin for
offline disassembly. This establishes feasibility, not the actual callback ABI,
module association, binary identity or successful instruction execution for
this repository. [CUPTI 12.9 SASS source correlation](https://docs.nvidia.com/cupti/12.9/main/main.html#sass-source-correlation)

Read API_FACTS.md for pinned local source/header anchors, DESIGN.md for the
small diagnostic boundary, and ROOT_GATES.md for the remaining source/runtime
requirements. source-pins.json hashes only source/header/metadata. The approved
CUPTI library identity is copied from existing metadata, not reread or rehashed.

No Cargo/compiler/Torch/model/provider/operator/GPU/container execution, binary
scan/disassembly, model/archive/payload read/hash, downloads, MAIN/vendor/shared
plan changes or old-freeze mutation by this author. Existing named tasks and
qualifiers remain authoritative; this advisory does not qualify ordered SiLU,
full-model numerics, throughput, kernel counts or production graph capture.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/frozen/ROOT_GATES.md

Original UTF-8 bytes: 2983; SHA256 `77a903d628bb06da5c46e1ec8190f7defcb6311a81d3a8c1e2431f680199be8d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Remaining ROOT gates

1. Source first: locate a genuine NVIDIA CUPTI 12.9 SDK sample/source showing
   exact module callback payload dispatch. Pin source/license/version and resolve
   ResourceData vs ModuleResourceData. Do not implement a dual caster. If no
   exact primary binding can be obtained, cubin pointer dereference stays CLOSED.
2. Review a minimal callback-only implementation against the actual approved
   headers. Compile real static_assert layouts and fixed-cap overflow checks;
   keep all notices. No copied third-party ABI accepted as primary closure.
3. Pin the actual helper/source/binary and approved CUPTI/driver mappings from
   verified retained FDs and actual mapping identity. No path-only/provider-name
   attestation. Single-subscriber conflict/lost callbacks/unknown IDs refuse.
4. ROOT may inspect actual preinstalled cuobjdump metadata, then separately
   authenticate its whole file/runtime closure before offline invocation. Image
   install history alone is not a verified tool. No acquisition is needed unless
   ROOT later determines the existing image lacks a usable tool.
5. Prepare source-free fixed-storage fixtures: oversize/zero/null/unknown module,
   slot overflow, missing/PTX NUL, reuse/unload/context generation, duplicate/
   foreign/missing association, failed original LoadData/GetFunction/Launch,
   concurrent/nested callback bounds, truncated file/ELF and wrong target symbol.
   Fixture arithmetic/driver stubs cannot qualify real loaded image semantics.
6. Fresh original plain qualifier, then fresh callback-only observed qualifier,
   identical source/input/profile/geometry/hardware/runtime/environment and
   original queue. All original output/intermediate/status/canary/read-only/
   leases/cancellation/Ready/final-budget proofs remain required. Do not label
   host launch callbacks as hardware activity or successful enqueue as Ready.
7. Actual trace must establish unique resource→original load/function/launch
   association and complete copied cubin/PTX; independently validate counts/
   generation/source bindings. Missing/ambiguous lazy-loading association is
   UNQUALIFIED, not a reason to force CUDA eager-loading or another module.
8. Unknown completion uses the original quarantine; no recovery, destroy or
   freeing original owners. A callback payload does not prove GPU completion.
   Fault/abandon tests remain original fresh-process routes; capture is not a
   requirement to extract an image from an unsafe unknown owner.
9. CPU-only disassembly of the exact captured file and bounded whole output,
   exact target function/control-flow inspection, transparent output validator,
   all owned process/group/pipe retirement and final evidence tree bounds pass
   before accepting loaded-image evidence. No timing or production promotion.

No ROOT step was run by the advisory author. Current evidence is pinned source
and primary documentation only; runtime/ABI/module-association/tool proof open.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/frozen/TODO.md

Original UTF-8 bytes: 1147; SHA256 `141c9c9ad4f2decdf626e742b93d9aa23ef2eee14bacd1e5968641a82f375dcd`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: serving-and-speculation P12. Owner: spine_review.
Scope: source-only advisory for actual original ordered-SiLU loaded SASS.

- [x] Inspect original CubeCL NVRTC→PTX→load_data→GetFunction→launch path.
- [x] Inspect approved CUPTI source headers/API28, actual callback parameter
  metadata and source-only existing collector/tool install metadata.
- [x] Check official NVIDIA12.9 documentation; identify supported extraction
  route, callback-only boundary and unresolved exact payload dispatch.
- [x] Prepare minimal bounded copy/correlation/lifetime/validation proposal.
- [x] Preserve all MAIN/vendor/old freeze bytes; source/header/metadata only.
- [ ] Exact NVIDIA12.9 sample binding resolves callback payload before coding.
- [ ] ROOT implementation review, layout compilation, fixtures and actual
  original plain/observed numerical/owner/association qualification.
- [ ] ROOT authenticates existing tool and disassembles actual captured image.

No builds/tests/runtime/provider/operator/model/GPU/binary/archive/payload scans
performed. No physical numerical, loaded-SASS or performance acceptance here.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/primary-abi-v2/CLOSURE.md

Original UTF-8 bytes: 3145; SHA256 `ae0f2c96dc5300568c8be88cc8d9dbaed6248ed0e2905641330dc884d95491bc`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Exact NVIDIA 12.9.79 module payload closure

Author: spine_review. Parent: serving-and-speculation P12.
Original advisory097cca66 and its six frozen files remain unchanged.

The module callback cast is now source-resolved by genuine NVIDIA
`samples/sass_source_map/sass_source_map.cu` from the exact CUPTI12.9.79
component published for CUDA12.9.1. The sample is6,152B, SHA
`4fff611da3ae8e1e9d886dcd1d79229c8a752b8403a6e0b46a03ecb52854940e`.
Original bytes/CRLF, NVIDIA copyright and full component LICENSE are retained.

- Lines117–118 select resource-domain cbdata as CUpti_ResourceData*.
- Lines92–94 pass its resourceDescriptor to DumpCudaModule.
- Line64 casts that descriptor to CUpti_ModuleResourceData*.
- Lines67–77 select MODULE_LOADED and read pCubin/cubinSize.
- Lines96–98 use the same descriptor for UNLOAD_STARTING.

This closes the header-description ambiguity: use one reviewed wrapper→
descriptor cast; never cast cbdata directly to ModuleResourceData or try both.
The sample ships with DUMP_CUBIN=0 and the cast/copy branch disabled by default.
It demonstrates the public intended ABI, not an author-executed layout, live
callback or numerical proof. Compile actual LP64 assertions against the already
pinned Linux12.9.79 headers before any ROOT run. No Windows binary is used.

Primary provenance: [NVIDIA exact-version manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json),
[exact CUPTI12.9.79 component](https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip).

The acquisition script requested only the22B EOCD, exact ZIP directory, local
headers and compressed source/LICENSE entries via HTTP206 ranges. All archive
requests returned the same ETag and Last-Modified; source CRC32/declared lengths
matched. Total HTTP body154,973B including54,256B official manifest,83,669B
ZIP directory,16,833B compressed source+license and215B headers. No library,
tool, model, executable or whole SDK archive entry was fetched/read/executed.
The full published component SHA is recorded but is NOT claimed independently
verified: whole archive bytes were never acquired. ZIP metadata does list binary
entry names/sizes; no binary member contents were read.

Preserve source-acquisition.json and the fetcher as exact acquisition evidence.
This source-only acquisition used standard-library HTTP/decompression only; no
CUPTI/driver/compiler/provider/operator/model/runtime import or execution.

The sample's file I/O inside callback and its instruction-execution activity are
NOT the proposed production/diagnostic pattern. A separate callback-only observer
will preallocate bounded storage and copy during the callback, write after a
closed observation boundary, and leave all original CUDA resources untouched.
No kernel/source activity, profiling, new queue/context/module or alternate math.
Actual original PTX→module→function→successful launch→original Ready correlation,
cubin extraction, tool identity/disassembly and full output transparency remain
UNQUALIFIED until ROOT executes reviewed gates.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/primary-abi-v2/TODO.md

Original UTF-8 bytes: 545; SHA256 `10213943ccdcdbed30775e37f2114cc6181768b9716585fb31425d923bb87fb2`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: serving-and-speculation P12. Owner: spine_review.

- [x] Exact NVIDIA12.9.79 primary source/license only acquired through ranges.
- [x] Resolve cbdata wrapper→resourceDescriptor→module payload.
- [x] Preserve original097c advisory; record honest partial archive provenance.
- [ ] ROOT layout compilation/live callback/image/correlation/transparency.
- [ ] Separate callback-only minimal collector/runner/validator source proposal.

No compiler/tests/provider/operator/GPU/model/runtime execution or binary scan.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/primary-abi-v2/frozen/CLOSURE.md

Original UTF-8 bytes: 3145; SHA256 `ae0f2c96dc5300568c8be88cc8d9dbaed6248ed0e2905641330dc884d95491bc`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Exact NVIDIA 12.9.79 module payload closure

Author: spine_review. Parent: serving-and-speculation P12.
Original advisory097cca66 and its six frozen files remain unchanged.

The module callback cast is now source-resolved by genuine NVIDIA
`samples/sass_source_map/sass_source_map.cu` from the exact CUPTI12.9.79
component published for CUDA12.9.1. The sample is6,152B, SHA
`4fff611da3ae8e1e9d886dcd1d79229c8a752b8403a6e0b46a03ecb52854940e`.
Original bytes/CRLF, NVIDIA copyright and full component LICENSE are retained.

- Lines117–118 select resource-domain cbdata as CUpti_ResourceData*.
- Lines92–94 pass its resourceDescriptor to DumpCudaModule.
- Line64 casts that descriptor to CUpti_ModuleResourceData*.
- Lines67–77 select MODULE_LOADED and read pCubin/cubinSize.
- Lines96–98 use the same descriptor for UNLOAD_STARTING.

This closes the header-description ambiguity: use one reviewed wrapper→
descriptor cast; never cast cbdata directly to ModuleResourceData or try both.
The sample ships with DUMP_CUBIN=0 and the cast/copy branch disabled by default.
It demonstrates the public intended ABI, not an author-executed layout, live
callback or numerical proof. Compile actual LP64 assertions against the already
pinned Linux12.9.79 headers before any ROOT run. No Windows binary is used.

Primary provenance: [NVIDIA exact-version manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json),
[exact CUPTI12.9.79 component](https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip).

The acquisition script requested only the22B EOCD, exact ZIP directory, local
headers and compressed source/LICENSE entries via HTTP206 ranges. All archive
requests returned the same ETag and Last-Modified; source CRC32/declared lengths
matched. Total HTTP body154,973B including54,256B official manifest,83,669B
ZIP directory,16,833B compressed source+license and215B headers. No library,
tool, model, executable or whole SDK archive entry was fetched/read/executed.
The full published component SHA is recorded but is NOT claimed independently
verified: whole archive bytes were never acquired. ZIP metadata does list binary
entry names/sizes; no binary member contents were read.

Preserve source-acquisition.json and the fetcher as exact acquisition evidence.
This source-only acquisition used standard-library HTTP/decompression only; no
CUPTI/driver/compiler/provider/operator/model/runtime import or execution.

The sample's file I/O inside callback and its instruction-execution activity are
NOT the proposed production/diagnostic pattern. A separate callback-only observer
will preallocate bounded storage and copy during the callback, write after a
closed observation boundary, and leave all original CUDA resources untouched.
No kernel/source activity, profiling, new queue/context/module or alternate math.
Actual original PTX→module→function→successful launch→original Ready correlation,
cubin extraction, tool identity/disassembly and full output transparency remain
UNQUALIFIED until ROOT executes reviewed gates.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/primary-abi-v2/frozen/TODO.md

Original UTF-8 bytes: 545; SHA256 `10213943ccdcdbed30775e37f2114cc6181768b9716585fb31425d923bb87fb2`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: serving-and-speculation P12. Owner: spine_review.

- [x] Exact NVIDIA12.9.79 primary source/license only acquired through ranges.
- [x] Resolve cbdata wrapper→resourceDescriptor→module payload.
- [x] Preserve original097c advisory; record honest partial archive provenance.
- [ ] ROOT layout compilation/live callback/image/correlation/transparency.
- [ ] Separate callback-only minimal collector/runner/validator source proposal.

No compiler/tests/provider/operator/GPU/model/runtime execution or binary scan.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/prototype-v3/PAUSED.md

Original UTF-8 bytes: 5997; SHA256 `504958be08905577597835dfb5d705bca7ad1875e3a0b12402a8169dc23442e4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Checkpoint — unfinished source proposal

Author: spine_review. Parent: serving-and-speculation P12.
Status: PAUSED_AT_USER_REQUEST / DRAFT_UNCOMPILED_UNREVIEWED.

## Frozen components

- Original source advisory097cca66357b43e91563f216f67d64746df57a8649288b762d671c494692289c unchanged.
- Sibling primary-abi-v2 b4b7f125344fdb1c1425adb3c9a965573a29c34e3a6bd1c3368cdf1ba83a37f6
  resolves exact NVIDIA12.9.79 wrapper→descriptor→module cast. Baseline completed
  independent source review without blocker; layout/live behavior remains open.
- Current collector.cpp is an INITIAL SOURCE DRAFT, not a reviewed executable.
  No runner, validator, build launcher or prepared fixture suite is written.

## Current draft boundary

- Public callbacks only: LoadData/GetFunction/LaunchKernel/Unload and module
  resource load/unload. No Activity API/profiling/PC sampling/kernel replay.
- Original calls are observed, not intercepted/replaced; kernelParams/extra and
  device data never dereferenced or altered. No new CUDA queue/context/module.
- Arena68MiB:16 selected PTX/cubin slots ×2MiB each plus bounded metadata;
 1024 module metadata /256 selected functions /4096 events. Arena preallocated
  and prefaulted before subscribe, retained until OS process retirement. This
  foreign diagnostic storage is NOT a fabricated Rust ResourceArbiter lease.
- Exact entrypoint allowlist<=16 (ROOT must derive authentic generated names),
  with bounded all-module metadata. It is selected-image scope, not complete
  hardware or whole-qualifier launch coverage. All limits overflow/refuse closed.
- Retained inherited target/CUPTI/output-directory FDs and nonce; constructor
  waits for same-parent SO_PEERCRED/ARM handshake before originalmain. The
  future runner must prove all actual mappings/pidfd/starttime/executable/source
  before ARM; no runner exists yet, so current environment fields are NOT a
  completed authenticity boundary by themselves.
- TLS LoadData nesting must provide exactly one resource module callback; lazy/
  asynchronous/reentrant/unknown routes fail. This is a proposed required actual
  observation, not a claimed ordering guarantee. Successful output handles bind
  original module→function→stream launch.
- Normal atexit closes callback writes, drains active writers then exports copied
  host data. No CUDA/CUPTI call, unsubscribe/finalize, driver destruction, recovery
  or owner free occurs at finish. Callback userdata/DSO/Fds remain until OS exit.
  Forced exits/missing/failed final export are UNQUALIFIED.

## Resume actions, in order

1. Review entire C++ draft before any compiler/run. Inspect source lexer bounds,
   exact status/entry-exit/context/handle generation checks, selected-vs-ignored
   accounting, thread-local callbacks, arena layout, output atomicity and
   callback finalization. Two lexical end checks and explicit includes were
   corrected before this freeze, without changing any existing parent source.
2. Decide whether failed/unselected ModuleLoadData and driver-generated callback
   routes should remain the current strict global refusal. No guessed fallback.
   The PTX NUL scan relies on the source-proven original NVRTC PTX call contract;
   arbitrary callers/binary images are outside scope and must refuse.
3. Implement a separate minimal ROOT runner around the EXISTING normal ignored
   ordered_silu physical qualifier (original args/source/Ready/math), fresh plain
   and observed processes. Preserve exact original qualifier output contract;
   do not add a replacement math/kernel loop just for the observer.
4. Reuse exact 53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d
   owned helper and00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128
   record-only TERM+INT guard; all signal stages before reap, bounded group/EOF
   cleanup, primary+cleanup errors independently retained. No async raise across
   Popen assignment. Parent ROOT owns all execution.
5. Reuse pinned fd-mapping-v3 pure same-FD mapping-key functions, with an explicit
   retained-FD lifetime adapter: observe/hash/map the SAME FDs passed to the
   collector, hold through child retirement and before/after hashing. Preserve
   stat-device vs kernel-mapping-device distinction. Do not copy container paths
   or call a different helper schema as if it were native source proof.
6. Implement an independent bounded validator: exact original normal Ready marker
   and test count/source identity; all selected entrypoint images; unique nested
   load event/correlation/TID/module generation; matching returned function and
   successful launch entry/exit geometry/context/stream; complete PTX/cubin files
   and strict sizes/ELF/symbol bindings; all overflow/error/drain counters zero;
   original canary/readset/final-budget and whole-process cleanup. No callback
   enqueue count is Ready or hardware activity. No exact math from SASS presence.
7. Prepare actual meaningful source fixtures for every bound, failed native
   status, duplicate/foreign/missing/asynchronous association, handle reuse,
   malformed PTX/ELF, callback drain and artifact-finalization failure. No
   fixture/stub can qualify real CUPTI/GPU/module semantics.
8. Only after ROOT source reviews/compiles may ROOT run normal plain+observed
   qualifier transparency, capture the actual image and authenticate an already
   present tool for offline disassembly. Scope ends at loaded image/successful
   original launch/completion; no PC execution trace/fullmodel/performance claim.

## Author actions

Only source/header/small-metadata reading, exact source-only HTTP ranges for
NVIDIA sample/license, source generation and snapshot hashing. No compiler,
tests, Torch/provider/operator/model/GPU/container/runtime execution or binary/
DSO/model/archive-member payload scan. ZIP directory/source member transfer is
recorded honestly; whole archive SHA was not independently verified. No MAIN,
vendor, shared plan, other owner or old freeze changed.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/prototype-v3/TODO.md

Original UTF-8 bytes: 968; SHA256 `23e5887731fbc31f77aea194e0ed4c995155c54102f10dca15f52c631881bd6a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Paused working checklist

Parent: serving-and-speculation P12. Owner: spine_review.
User pause received through ROOT; no new runtime/source scope started.

- [x] Exact primary NVIDIA12.9.79 payload ABI source closure in sibling
  primary-abi-v2 (freeze b4b7f125); independent Baseline source review NOBLOCKER.
- [x] Write initial separate callback-only collector.cpp proposal.
- [x] Freeze current draft and exact continuation; original097c unchanged.
- [ ] FULL independent source review of collector; no review verdict yet.
- [ ] ROOT-only strict compile/static layouts; no author compile or tests.
- [ ] Implement fresh runner, retained-FD mapping ownership adapter, independent
  validator and bounded source fixtures; NONE are present in this draft.
- [ ] ROOT-only live callback nesting/cubin/Ready/output transparency.
- [ ] ROOT-only existing tool authentication and exact loaded-image SASS.

No runtime/numerical/SASS/whole-model/performance qualification.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/prototype-v3/frozen/PAUSED.md

Original UTF-8 bytes: 5997; SHA256 `504958be08905577597835dfb5d705bca7ad1875e3a0b12402a8169dc23442e4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Checkpoint — unfinished source proposal

Author: spine_review. Parent: serving-and-speculation P12.
Status: PAUSED_AT_USER_REQUEST / DRAFT_UNCOMPILED_UNREVIEWED.

## Frozen components

- Original source advisory097cca66357b43e91563f216f67d64746df57a8649288b762d671c494692289c unchanged.
- Sibling primary-abi-v2 b4b7f125344fdb1c1425adb3c9a965573a29c34e3a6bd1c3368cdf1ba83a37f6
  resolves exact NVIDIA12.9.79 wrapper→descriptor→module cast. Baseline completed
  independent source review without blocker; layout/live behavior remains open.
- Current collector.cpp is an INITIAL SOURCE DRAFT, not a reviewed executable.
  No runner, validator, build launcher or prepared fixture suite is written.

## Current draft boundary

- Public callbacks only: LoadData/GetFunction/LaunchKernel/Unload and module
  resource load/unload. No Activity API/profiling/PC sampling/kernel replay.
- Original calls are observed, not intercepted/replaced; kernelParams/extra and
  device data never dereferenced or altered. No new CUDA queue/context/module.
- Arena68MiB:16 selected PTX/cubin slots ×2MiB each plus bounded metadata;
 1024 module metadata /256 selected functions /4096 events. Arena preallocated
  and prefaulted before subscribe, retained until OS process retirement. This
  foreign diagnostic storage is NOT a fabricated Rust ResourceArbiter lease.
- Exact entrypoint allowlist<=16 (ROOT must derive authentic generated names),
  with bounded all-module metadata. It is selected-image scope, not complete
  hardware or whole-qualifier launch coverage. All limits overflow/refuse closed.
- Retained inherited target/CUPTI/output-directory FDs and nonce; constructor
  waits for same-parent SO_PEERCRED/ARM handshake before originalmain. The
  future runner must prove all actual mappings/pidfd/starttime/executable/source
  before ARM; no runner exists yet, so current environment fields are NOT a
  completed authenticity boundary by themselves.
- TLS LoadData nesting must provide exactly one resource module callback; lazy/
  asynchronous/reentrant/unknown routes fail. This is a proposed required actual
  observation, not a claimed ordering guarantee. Successful output handles bind
  original module→function→stream launch.
- Normal atexit closes callback writes, drains active writers then exports copied
  host data. No CUDA/CUPTI call, unsubscribe/finalize, driver destruction, recovery
  or owner free occurs at finish. Callback userdata/DSO/Fds remain until OS exit.
  Forced exits/missing/failed final export are UNQUALIFIED.

## Resume actions, in order

1. Review entire C++ draft before any compiler/run. Inspect source lexer bounds,
   exact status/entry-exit/context/handle generation checks, selected-vs-ignored
   accounting, thread-local callbacks, arena layout, output atomicity and
   callback finalization. Two lexical end checks and explicit includes were
   corrected before this freeze, without changing any existing parent source.
2. Decide whether failed/unselected ModuleLoadData and driver-generated callback
   routes should remain the current strict global refusal. No guessed fallback.
   The PTX NUL scan relies on the source-proven original NVRTC PTX call contract;
   arbitrary callers/binary images are outside scope and must refuse.
3. Implement a separate minimal ROOT runner around the EXISTING normal ignored
   ordered_silu physical qualifier (original args/source/Ready/math), fresh plain
   and observed processes. Preserve exact original qualifier output contract;
   do not add a replacement math/kernel loop just for the observer.
4. Reuse exact 53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d
   owned helper and00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128
   record-only TERM+INT guard; all signal stages before reap, bounded group/EOF
   cleanup, primary+cleanup errors independently retained. No async raise across
   Popen assignment. Parent ROOT owns all execution.
5. Reuse pinned fd-mapping-v3 pure same-FD mapping-key functions, with an explicit
   retained-FD lifetime adapter: observe/hash/map the SAME FDs passed to the
   collector, hold through child retirement and before/after hashing. Preserve
   stat-device vs kernel-mapping-device distinction. Do not copy container paths
   or call a different helper schema as if it were native source proof.
6. Implement an independent bounded validator: exact original normal Ready marker
   and test count/source identity; all selected entrypoint images; unique nested
   load event/correlation/TID/module generation; matching returned function and
   successful launch entry/exit geometry/context/stream; complete PTX/cubin files
   and strict sizes/ELF/symbol bindings; all overflow/error/drain counters zero;
   original canary/readset/final-budget and whole-process cleanup. No callback
   enqueue count is Ready or hardware activity. No exact math from SASS presence.
7. Prepare actual meaningful source fixtures for every bound, failed native
   status, duplicate/foreign/missing/asynchronous association, handle reuse,
   malformed PTX/ELF, callback drain and artifact-finalization failure. No
   fixture/stub can qualify real CUPTI/GPU/module semantics.
8. Only after ROOT source reviews/compiles may ROOT run normal plain+observed
   qualifier transparency, capture the actual image and authenticate an already
   present tool for offline disassembly. Scope ends at loaded image/successful
   original launch/completion; no PC execution trace/fullmodel/performance claim.

## Author actions

Only source/header/small-metadata reading, exact source-only HTTP ranges for
NVIDIA sample/license, source generation and snapshot hashing. No compiler,
tests, Torch/provider/operator/model/GPU/container/runtime execution or binary/
DSO/model/archive-member payload scan. ZIP directory/source member transfer is
recorded honestly; whole archive SHA was not independently verified. No MAIN,
vendor, shared plan, other owner or old freeze changed.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/prototype-v3/frozen/TODO.md

Original UTF-8 bytes: 968; SHA256 `23e5887731fbc31f77aea194e0ed4c995155c54102f10dca15f52c631881bd6a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Paused working checklist

Parent: serving-and-speculation P12. Owner: spine_review.
User pause received through ROOT; no new runtime/source scope started.

- [x] Exact primary NVIDIA12.9.79 payload ABI source closure in sibling
  primary-abi-v2 (freeze b4b7f125); independent Baseline source review NOBLOCKER.
- [x] Write initial separate callback-only collector.cpp proposal.
- [x] Freeze current draft and exact continuation; original097c unchanged.
- [ ] FULL independent source review of collector; no review verdict yet.
- [ ] ROOT-only strict compile/static layouts; no author compile or tests.
- [ ] Implement fresh runner, retained-FD mapping ownership adapter, independent
  validator and bounded source fixtures; NONE are present in this draft.
- [ ] ROOT-only live callback nesting/cubin/Ready/output transparency.
- [ ] ROOT-only existing tool authentication and exact loaded-image SASS.

No runtime/numerical/SASS/whole-model/performance qualification.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-peer-review-20261020/REVIEW.md

Original UTF-8 bytes: 5250; SHA256 `9e03b1e03361395fbaebc2a58fb82575392e72fcb8016e780cdc8164c722852a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU integration peer review

Owner baseline_review; parent serving-and-speculation P12. Source only. No
MAIN/plan/frozen-source edits, builds/tests/prepared imports/model/operator/GPU
execution or model/DSO/binary/raw-trace payload scans.

**No remaining source blocker after visibility followup3bc2bf33 and scalar
repair16198590. Physical/operator/full-model/serving/performance gates remain
ROOT-owned; this review grants no production selection.**

## Exact reviewed layers

- Integration9110ef765e1e: all20 live/frozen source/artifact pins exact;
  all10 source-context pins still exact. Complete changed4-path patch, shared
  emitter/tests, separate geometry/lib patches and original owner/caller context.
- Valid blocker found: child `resident_program::silu` private methods were
  called by parent/sibling. Followup3bc2bf3316b1/patch01ecb17b exactly changes
  `preflight_silu` +`launch_silu` to `pub(super)`. All4 live/frozen followup pins
  exact; reversing the two keywords recovers original05aebe1a. Binding stays
  private. No arithmetic/ownership changes.
- Scalar16198590093e: all11 live/frozen pins exact. Exactly15 scalar loads
  across9 asm helpers; stripping those lines +explanation recovers baseline
  de0149c2 byte-for-byte. Actual MAIN ordered source is3d789c06,14175B, matching
  the ordered entry in ROOT's459-input freeze158be58c.
- Composition: apply the small geometry seam over scalar-corrected3d789c,
  **never copy9110's old full ordered module**, which would remove scalar fixes.
  Followup visibility README now records this requirement.

## Selection, bounds and original flight

- Original GGUF/F32/full-plan guard retained; typed/HF/operator slices refuse.
  CUDA identity is the original Client. Vulkan/known unsupported row widths
  select original body; capability/range/alias/service errors propagate, never
  select fallback or retry after partial work.
- Native2/4/8 use packed elements /actual rows, not aggregate-as-B1. Width65536
  atB8 remains8 real rows; inconsistent partitions refuse. No padding/session4
  invention, learned-source rewrite or metadata allocation/table/field.
- Constructor checks every immutable SiLU node, even skipped StateOnly tail.
  Attempt checks occur before original `begin_submission`/physical flight;
  same bindings are rechecked before the node's first interval launch. Initial
  construction remains under the original distinct initialization owner/Ready.
- Batch preflights with actual rows once, then reuses exact original per-row
  state validator. Original `validate_runtime` body equals renamed
  `validate_state_runtime` body byte-for-byte. No source/root/generation/state-
  range/candidate alias/Ready/cancel/quarantine ownership changes.
- Prepared spans use checked end/subtraction/alignment and exact original
  service; writes cannot overlap reads/other writes. Inactive multiplier/trace
  aliases have no compile-time-branch access. Handle/managed-binding clones are
  existing Arc/POD only; actual pointer resolution stays in original queue.
- No budget/class/scratch/lease changes. Successful prepare has stack arrays,
  Arc clones and borrowed properties only. Existing CubeCL compiler/launcher
  control-plane behavior is not a new total-host-memory claim.

## Arithmetic and source receipts

- Original[-256,256] interval remains first, status is never cleared by emitter.
  Original outer finite/state certificates and OR remain. This does not broaden
  KV certification to inactive capacity padding.
- Independent normalized comparison confirms exact original interval call,
  12-stage loop, optional gate and13+multiplied return for fallback.
- Selected ordered node counts interval1 +kernel1 =2 source launch sites;
  original13/14→2 saves11/12 sites. Outer certificates stay separate. Counts
  remain source receipts, not hardware-enumerated kernels or speed evidence.
- Geometry seam changes only support query/header; checked_span→EOF is exactly
  old candidate bytes. The scalar fix changes no templates, outputs, constants,
  math, round/store points, queue or status policy.
- Frontend inspected directly: gpu_asm `push_input` uses raw value; Reinterpret
  calls `read_value`, emitting LoadOp for pointer-backed mutable scalars; equal-
  width reinterpretation preserves bits. Outer float type yields'f'/'d' instead
  of pointer'l'. Optimizing inverse bitcasts cannot remove the required scalar
  load. Authentic generated CPP/PTX/SASS +old-GPU/CPU parity still qualify this.

## Parser correction and remaining gates

My initial missing-assertRaises report was a reading error and is **retracted**.
Actual +frozen test_generated_cpp.py2088B/4364c4ed has assertRaises at42–43.
ROOT already reported all5 fixtures PASS. No parser repair is required.

ROOT reported fresh6-process operator/fault gates including204 normal cases
PASS while review finished; no execution/independent raw numerical validation by
this reviewer. Authentic CPP/PTX/SASS closure, full trained CPU/B1/native all42/
fullV/feedback/prefix/boundary/pressure/abort, worker14/HTTP12, strict/allcfg/full
isolated gate/both approved routes and honest unprofiled native A/B remain the
explicit integration acceptance sequence. Existing graph remains diagnostic.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-peer-review-20261020/TODO.md

Original UTF-8 bytes: 518; SHA256 `f86c898f5fff941a07f3cfd32ed4ba72af2dcc2ba3f9a832a233febe90077125`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU independent source review

Parent serving-and-speculation; owner baseline_review.

- [x] Review9110 whole changed source +geometry/lib/owner context.
- [x] Report and verify narrow3bc2 visibility repair.
- [x] Review1619 scalar-load repair/frontend; verify3d789c in ROOT158be58c.
- [x] Retract false parser finding after exact actual/frozen pin recheck.
- [ ] ROOT remaining physical/CPP/PTX/SASS/full integration/serving/gates/A-B.

No MAIN/shared plans/frozen bytes or runtime changed by this review.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-runner-peer-review-batch-20261020/REVIEW.md

Original UTF-8 bytes: 3619; SHA256 `bc68740922152b0cbf4a198555326ec7b709eca06899ad753203972f920d11cf`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU physical runner — source-only review

- Verdict: **no source blocker found** before launch. Reviewed exact
  root-composition-1/root-physical-1.py, build-source binder, qualifier dispatch,
  exact test names/cfg, original53ee owner and00d1 cancellation guard.
- No imports/execution/tests/builds/hashes or binary/model/DSO/payload scans.
  Own ignored review only; MAIN/shared plans/production acceptance unchanged.

## Case/argument closure

- Exactly60 distinct cases: six operator +42 qualifier CLI +12 exact library
  tests. Per model: one B1; three widths × three normal modes; three widths ×
  three graph-unknown modes; two fixed-B2 native aborts =21 CLI. Prefix three
  bands +pressure/abort +admission-pressure =six library cases per model.
- CLI dispatch consumes original source, provider4096MiB, device5120MiB and
  rows2/4/8. Graph unknown then consumes state/logits/greedy. Native abort
  implementations are fixed B2 and the runner explicitly passes2.
- Exact prefix/admission test names match compiled module paths and cfg gates.
  `--exact` plus required `1 passed; 0 failed; 0 ignored` prevents a missing
  ignored test from passing as a zero-test process.
- HERE.parents[3] resolves repository root. Nonoperator cases run there, so
  approved relative model paths resolve correctly. Private operator CWD is
  used solely for full compilation logger/profiling-disabled configuration.

## Ownership, signal and source closure

- Fixed expected freeze plus exact successful release-build status/source
  binder precedes process creation. Every case rechecks intended source rows,
  all executable artifacts and root README before and after execution; quiet
  device check and original guard prevent starting a later failed case.
- Record-only TERM/INT guard spans Popen assignment and Log construction.
  No asynchronous handler raises across ownership transfer. observe restores
  original log.drain before cleanup, so recorded cancellation cannot block
  original53ee WNOWAIT ownership retirement.
- Popen/Log/observe are inside try/finally. Original cleanup signals bound
  unreaped group before any reap, reaps original child and checks absence.
  Log finish proves bounded evidence/EOF and closes selector/file/pipe.
  Partial Log-constructor failure still reaches group cleanup and closes the
  original stdout pipe. Cleanup failures reject the case and stop the matrix.
- Receipt writes and source checks occur after group/EOF retirement. A receipt
  I/O error cannot strand a live child or start another case. Final cancellation
  checkpoint precedes final successful summary.

## Scope/limitations

- Generic CLI `PASS` substring is deliberately weak as an evidence classifier;
  exit0 and exact fixed command dispatch are stronger current execution gates.
  ROOT must still review each expected per-mode marker/counter/domain result,
  complete60 case membership and owner/EOF receipts before accepting the phase.
- Source hashes checked by the runner are intended source closure, not an
  independent full dependency/DSO/provider attestation. Actual build receipt
  remains required. Reviewer did not calculate or validate hashes.
- Only operator-normal enables compilation logging/PTX dumping; all inherited
  debug/dump overrides are rejected. New witness file completeness and all
  selected entry identities require independent compiler audit afterward.
- 54 model cases do not substitute worker14/HTTP12, full isolated gate,
  both release parity routes, independent physical kernel counters or A/B.
  Diagnostic graph execution remains explicitly qualification-only.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-runner-peer-review-batch-20261020/TODO.md

Original UTF-8 bytes: 500; SHA256 `e5d3e49f0ce17d6b60f169632993ed74dd3e2a3ce65fac57333cf8d0d210998a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Physical runner peer review

Parent: serving-and-speculation. Owner: batch_design_review.
Scope: this ignored review directory only.

- [x] Review60-case enumeration, CLI arity/modes and exact library test cfg.
- [x] Review original owner/record-only signals, Popen handoff, finally/EOF and
  source/build checks; report no source blocker.
- [ ] ROOT execution, complete result review and remaining full phase gates.

No runtime/imports/tests/builds/hashes or MAIN/shared-plan changes by reviewer.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-scalar-input-20261020/README.md

Original UTF-8 bytes: 4653; SHA256 `30fd5a2c2a13f47940706abb125324b68d3880b7ca60cc925eb0228a524be50f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU scalar asm inputs — source-only repair

Author: batch_design_review. Parent: serving-and-speculation P12.

Actual ROOT first physical attempt failed CUDA compilation before numerical
Ready. This is not accepted operator arithmetic. Its owned cleanup/EOF/group
closure remains retained in original root-physical-1. All original a7939,
qualifier70c594 and integration9110ef76 freezes remain untouched.

## Causal source finding

- `cubecl-core/frontend/asm.rs` push_input<T:HasValue> calls value(), which can
  be the pointer backing a mutable scalar (including a prior asm output).
- `cubecl-ir/value.rs` read_value() instead emits LoadOp for such a pointer.
- `frontend/element/cast.rs` Reinterpret::__expand_reinterpret always calls
  read_value BEFORE reinterpret_value; latter checks exact equal byte size.
- Actual generated C++ has `float *v677` then cvt.f64.f32 input `"l"(v677)`,
  f64 FMA operands also `"l"`, and mov.b32 final float inputs `"l"`. CUDA CPP
  infer_constraint_letter selects'l' for pointers. Both final mov.b32 calls
  cause actual ptxas operand-size mismatch. Earlier64-bit operations can consume
  address bits without a size error, so fixing only mov.b32 is insufficient.

## Narrow fix

One ordered-module patch:15 scalar input materializations across9 asm helpers.
Each f32 input is `f32::reinterpret(u32::reinterpret(value))`; f64 uses f64/u64.
No asm template, output, arithmetic, literal, rounding, status, lease, dispatch,
queue or parameter changes. Removing those15 lines and one explanation restores
the original source byte-for-byte. No vendor or integration freeze is changed.

The first reinterpret forces the scalar load; subsequent same-size casts only
transport bits. Signed zero/subnormal/NaN payloads are not numerically converted.
An optimizer removing inverse casts must retain the scalar load/value type;
it cannot substitute the differently typed pointer. Actual generated CPP/PTX
and complete numerical proof remain mandatory, not inferred from this design.

A same-type `f32::reinterpret(value)`/f64 equivalent would also force read_value
in the current frontend. The explicit integer roundtrip makes the bit-only
intent visible. A general CPP pointer dereference is unsafe: real pointer and
memory asm inputs require addresses. A generic frontend fix needs a typed
value-input boundary or explicit modifier while preserving pointer semantics;
that broader change is deliberately outside this repair.

## Meaningful ROOT regression recipe

1. Preserve the original failed source/binary/log/receipt. Review/compose this
   patch into a fresh ROOT derivative of ordered test-only459; no production
   wiring or integration promotion. Copy fresh mtimes, strict/build again.
2. Run prepared parser tests only when ROOT schedules them. Actual two failed
   generated CPP sources are decoded from the original bounded54,053B log and
   must be refused. Synthetic successful declarations only test the parser.
3. Capture authentic corrected CubeCL generated C++ under original diagnostic
   compilation logging, not a hand-rewritten source. Source-proven seam:
   CUDA compute/context.rs load_transpiled sets DebugInformation("cpp") when
   ServerLogger compilation source is enabled; cubecl-runtime config accepts
   CUBECL_DEBUG_LOG and cubecl-server kernel Display emits fenced cpp source.
   ROOT must use a fresh owned bounded log/artifact envelope (not timing), bind
   selected ordered plain/gated and compact/traced specializations to actual
   compile argv/source/binary, and refuse truncation/loss. Do not silently exceed
   the old runner log cap to collect source.
4. `audit_generated_cpp.py --cpp AUTHENTIC_ORDERED_CPP [--cpp ...]
   --new-output NEW_JSON`: all9 opcode families must occur; every volatile asm
   must parse; actual float inputs use'f', double inputs'd', and scalar C++
   declarations must match. No pointer'l' input is accepted for these helpers.
   Bounded2MiB/file,16 selected files/8MiB total; static receipt only, ROOT routing
   mandatory. This catches f64 address-bit consumption even if NVRTC accepts it.
5. Original PTX/SASS checks +all204 exact original-GPU/CPU/intermediate/status
   cases and fresh fault/cancel/refusal tests must PASS on the corrected binary.
   Only then assess integration9110 source composition +full trained/serving/
   gates/A-B. No runtime/math/fullmodel/speed claim follows from this patch.

Author actions: source reading, bounded error-string decode/source extraction,
own source patch/formatting and static comparisons/AST only. No tests, compiler,
provider/operator/model/GPU/runtime execution or binary/DSO/payload read.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-scalar-input-20261020/TODO.md

Original UTF-8 bytes: 637; SHA256 `0992bb097e55c56dca4315ae51b41c3fdab031a84756125315fea7085ca79908`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered scalar-input continuation

Parent: serving-and-speculation P12. Owner: batch_design_review.

- [x] Confirm actual generated pointer constraints and source frontend cause.
- [x] Prepare15-input/9-helper bit-only repair, preserving all prior freezes.
- [x] Prepare bounded generated-CPP checker and5 parser tests; not executed.
- [ ] ROOT peer review, fresh composition/strict/build/CPP/PTX/SASS.
- [ ] ROOT actual204 numerical/status cases and fresh failure controls.
- [ ] ROOT assesses later production integration; remains unqualified here.

No MAIN/shared-plan edits, commits, test/build/runtime or payload scans by author.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-scalar-input-20261020/frozen/README.md

Original UTF-8 bytes: 4653; SHA256 `30fd5a2c2a13f47940706abb125324b68d3880b7ca60cc925eb0228a524be50f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered SiLU scalar asm inputs — source-only repair

Author: batch_design_review. Parent: serving-and-speculation P12.

Actual ROOT first physical attempt failed CUDA compilation before numerical
Ready. This is not accepted operator arithmetic. Its owned cleanup/EOF/group
closure remains retained in original root-physical-1. All original a7939,
qualifier70c594 and integration9110ef76 freezes remain untouched.

## Causal source finding

- `cubecl-core/frontend/asm.rs` push_input<T:HasValue> calls value(), which can
  be the pointer backing a mutable scalar (including a prior asm output).
- `cubecl-ir/value.rs` read_value() instead emits LoadOp for such a pointer.
- `frontend/element/cast.rs` Reinterpret::__expand_reinterpret always calls
  read_value BEFORE reinterpret_value; latter checks exact equal byte size.
- Actual generated C++ has `float *v677` then cvt.f64.f32 input `"l"(v677)`,
  f64 FMA operands also `"l"`, and mov.b32 final float inputs `"l"`. CUDA CPP
  infer_constraint_letter selects'l' for pointers. Both final mov.b32 calls
  cause actual ptxas operand-size mismatch. Earlier64-bit operations can consume
  address bits without a size error, so fixing only mov.b32 is insufficient.

## Narrow fix

One ordered-module patch:15 scalar input materializations across9 asm helpers.
Each f32 input is `f32::reinterpret(u32::reinterpret(value))`; f64 uses f64/u64.
No asm template, output, arithmetic, literal, rounding, status, lease, dispatch,
queue or parameter changes. Removing those15 lines and one explanation restores
the original source byte-for-byte. No vendor or integration freeze is changed.

The first reinterpret forces the scalar load; subsequent same-size casts only
transport bits. Signed zero/subnormal/NaN payloads are not numerically converted.
An optimizer removing inverse casts must retain the scalar load/value type;
it cannot substitute the differently typed pointer. Actual generated CPP/PTX
and complete numerical proof remain mandatory, not inferred from this design.

A same-type `f32::reinterpret(value)`/f64 equivalent would also force read_value
in the current frontend. The explicit integer roundtrip makes the bit-only
intent visible. A general CPP pointer dereference is unsafe: real pointer and
memory asm inputs require addresses. A generic frontend fix needs a typed
value-input boundary or explicit modifier while preserving pointer semantics;
that broader change is deliberately outside this repair.

## Meaningful ROOT regression recipe

1. Preserve the original failed source/binary/log/receipt. Review/compose this
   patch into a fresh ROOT derivative of ordered test-only459; no production
   wiring or integration promotion. Copy fresh mtimes, strict/build again.
2. Run prepared parser tests only when ROOT schedules them. Actual two failed
   generated CPP sources are decoded from the original bounded54,053B log and
   must be refused. Synthetic successful declarations only test the parser.
3. Capture authentic corrected CubeCL generated C++ under original diagnostic
   compilation logging, not a hand-rewritten source. Source-proven seam:
   CUDA compute/context.rs load_transpiled sets DebugInformation("cpp") when
   ServerLogger compilation source is enabled; cubecl-runtime config accepts
   CUBECL_DEBUG_LOG and cubecl-server kernel Display emits fenced cpp source.
   ROOT must use a fresh owned bounded log/artifact envelope (not timing), bind
   selected ordered plain/gated and compact/traced specializations to actual
   compile argv/source/binary, and refuse truncation/loss. Do not silently exceed
   the old runner log cap to collect source.
4. `audit_generated_cpp.py --cpp AUTHENTIC_ORDERED_CPP [--cpp ...]
   --new-output NEW_JSON`: all9 opcode families must occur; every volatile asm
   must parse; actual float inputs use'f', double inputs'd', and scalar C++
   declarations must match. No pointer'l' input is accepted for these helpers.
   Bounded2MiB/file,16 selected files/8MiB total; static receipt only, ROOT routing
   mandatory. This catches f64 address-bit consumption even if NVRTC accepts it.
5. Original PTX/SASS checks +all204 exact original-GPU/CPU/intermediate/status
   cases and fresh fault/cancel/refusal tests must PASS on the corrected binary.
   Only then assess integration9110 source composition +full trained/serving/
   gates/A-B. No runtime/math/fullmodel/speed claim follows from this patch.

Author actions: source reading, bounded error-string decode/source extraction,
own source patch/formatting and static comparisons/AST only. No tests, compiler,
provider/operator/model/GPU/runtime execution or binary/DSO/payload read.

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-scalar-input-20261020/frozen/TODO.md

Original UTF-8 bytes: 637; SHA256 `0992bb097e55c56dca4315ae51b41c3fdab031a84756125315fea7085ca79908`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered scalar-input continuation

Parent: serving-and-speculation P12. Owner: batch_design_review.

- [x] Confirm actual generated pointer constraints and source frontend cause.
- [x] Prepare15-input/9-helper bit-only repair, preserving all prior freezes.
- [x] Prepare bounded generated-CPP checker and5 parser tests; not executed.
- [ ] ROOT peer review, fresh composition/strict/build/CPP/PTX/SASS.
- [ ] ROOT actual204 numerical/status cases and fresh failure controls.
- [ ] ROOT assesses later production integration; remains unqualified here.

No MAIN/shared-plan edits, commits, test/build/runtime or payload scans by author.

````

## Source: .agents/runs/serving-and-speculation-p12-greedy-batch/TODO.md

Original UTF-8 bytes: 785; SHA256 `10ac87d38b92795d55912edbffbaf49662f7fb33c57b8c6f2b3d0911aeaef2ec`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Native batch greedy and device feedback

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Owner: batch-design agent
Scope: `resident_program/batch.rs`, `resident_batch_greedy.rs`

## Uncommitted
- [x] Read accepted greedy source and current batch lowering.
- [x] Agree native stage/private packet/workspace APIs with MAIN.
- [x] Implement native source-exact row reductions and packet scatter.
- [x] Implement row-bound physical packet token gather.
- [ ] MAIN: compile, strict Clippy and physical model qualification.
- [ ] Independent review and reconcile MAIN results.

## Tests
- No Cargo/GPU by this agent; MAIN owns those checks.
- `rustfmt --edition 2024` passed for both owned sources.
- Added source-free workspace-size and geometry-refusal tests; execution pending.

````

## Source: .agents/runs/serving-and-speculation-p12-greedy-qual/TODO.md

Original UTF-8 bytes: 1869; SHA256 `120f4cac2da8115d72b85dc65481a6d0891c983a1603759de00d405aac7d446a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12 native greedy qualification

Parent: [serving-and-speculation](../../plans/serving-and-speculation/TODO.md)
Owner: baseline_review
Scope: new `resident_fused_greedy_qualification.rs`; this run file only.

- [x] Read accepted B1/native source/state/greedy ownership paths.
- [x] Draft CPU/B1/native-full-logit references, host/EOS seed and four real
  feedback steps, tampered host metadata, exact compact packet/state receipts.
- [x] Draft swapped/duplicate/foreign/equal-ID/stale/restore/abandon refusals,
  packet/session reclamation and isolated synthetic tie/EOS selector fixtures.
- [x] Draft packet-aware physical future-abort qualifier: actual incoming/output
  packets retained, original state/packet bytes exact, roots unchanged, all future
  work/models refused, unknown-completion charge/fence/quarantine retained.
- [x] Correct first compile's tokenizer path and byte-slice checks; add actual
  accepted B1 GPU greedy packet oracle after restoring the full-logit prior.
- [x] Add actual native one-row nonfinite and preexisting-status poison cases;
  every row emits invalid ID/flags, no logical token/root is constructed.
- [x] Isolate same-epoch old-root refusal after first feedback commit: fresh
  siblings + old last-row packet, explicit producer epoch/root checks, unchanged
  complete roots/all42 state/admission/pending events.
- [x] Inspect retained .8B abort failure: CPU domains use layer order, physical
  domains use sorted StateSlot order (first mismatch Recurrent0/Conv1). Match by
  unique slot; preserve all42 exact values/full bytes/canonical KV tail checks.
- [ ] MAIN wires module/tool routes and actual sampler API.
- [ ] MAIN compiles/lints and runs physical .8B/2B B2/B4/B8, abort, full gate
  and independent original direct-cache/HF-import parity routes.

No Cargo, GPU work, model/SDK downloads or commits run by this agent.

````

## Source: .agents/runs/serving-and-speculation-p12-greedy-root-2026-10-19/TODO.md

Original UTF-8 bytes: 3722; SHA256 `77f49739ff0b00eac7942e4686cb52c3204456f548fc01129bea6b39b574d243`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12-B4 native batch greedy/device feedback

Parent: .agents/plans/serving-and-speculation/TODO.md
Base: 904a190cf39aca74e06c979dbaddb0db7f4404f1
Owner: root; MAIN Cargo/physical-test coordinator.
Scope: original ResidentModel/ResidentToken integration, module/tool wiring,
source/physical acceptance, owning plan and signed local commits.
Disjoint children: batch lowerer/sampler; new physical greedy qualifier;
independent Ready/token design and final review.

- [x] Revalidate live rules/plan, cleanmain+12 and sole worktree.
- [x] Existing NVIDIA RTX3060 idle52C/41MiB; rootREADME originalblob unchanged.
- [x] Re-read original B1 greedy/Ready and current native cohort implementation.
- [x] Integrate compact per-row original packet Ready and native GPU feedback.
- [x] Source-free GPU53passed/0failed/3ignored; strict CUDA qualification lint passed.
  Initial qualifier adapter compile failed3types; fixed and same strict gate rerun exit0.
  Default GPU and composedCUDA/Vulkan serve+cache lint passed; final qualifier
  pure sameepoch oldroot case added after review, final CUDA lint passed again.
- [x] Both approved original .8B/2B B2/B4/B8 final physical matrix exit0, six steps/fourdevicefeedback,
  full248320logitshadow+CPU+B1, actualB1greedyID/max/all42exact; sameepocholdroot refused.
- [x] Both final abortprobes exit0:1827/1722originalhandles,4GiBfencedretained,
  priorall42/fullKVtail+incomingpacketbits exact;7syntheticfullV cases perB passed.
  Both original full-logit B2 boundary regressions passed valid1/256/257/512.
  Finalfrozen isolatedcheck and both releaseparity routes rerun exit0; allsourcehashes unchanged.
- [x] Full isolatedcheck exit0:1147Rustpass/0fail/69ignored over241suites;18Python,
  releaseCLI/installer/Nixfmt. Both original cache-direct/HF-import release routes
  passed .8B/2B full248320logits/all42domains/leases/mutation. Comparator selftestpass.
- [x] Reviewed signed54fd580 implementation+tests+plan, signed570d865 substantive
  acceptance/docs+completedgroupreconciliation. Both signaturesG; mainclean+14,
  soleworktree; README/sourcehashes unchanged; GPUidle61C/41MiB; nopush.

Constraints: one Rust engine/source/root/resource authority; private12B packet
per row; producerNone before actual cohort commit; GPU first-word feedback,
public host ID/max fields non-authoritative; status4B+12B/row D2H, no full logits.
HTTP fused mode and vLLM performance remain unqualified. No SDK/model downloads,
AMD launches, device privileges/recovery, rootREADME changes or other-plan edits.
Dotlink connector remains absent from callable tool catalogue; local workspace.

Review: two independent frozen-source reviews no correctness/lease/event blocker.
Qualifier old-root negative was masked by epochrestore; added isolated last-row
sameepoch olderroot with fresh siblings, unchanged roots/all42state/bytes/events.
First matrix binary predates this added negative only; rebuild+final matrix pending.

Initial .8B feedbackabort launched actual model+sampler then failed verifier
StateSlotzip ordering (CPUlayer order vs physical BTreeMapslot order). Fixed
qualifier lookup via unique slot map; all42/fulllease/KVzerobit checks retained.
Own in-progress releasebuild interruptedSIGINT130 to avoid stale binary.
Strict+release rebuild/finalnormal+abort+boundary matrix pending on frozen repair.

Finalphysicalmatrix terminalexit0:6normal+2abort; separate2boundaryexit0.
Reviewed finaldocs counts againstactualreceipts by coordinator; independentdocs
review correctedexactly-once toat-most-once emission attempts andcreditwaitparking.
RootREADME blob unchangeda7be69c35cf61d14e8810f9cb5a6a43d1397bf27; sourceGGUFhashes unchanged.
Signedcommits/owningplanreconciliation next; nopush.

````

## Source: .agents/runs/serving-and-speculation-p12-http-network-2026-10-19/TODO.md

Original UTF-8 bytes: 2004; SHA256 `35a5e8a6bb263febd11b06ea2f46f8acf496ea9edc7ed86293c801a6a30bc733`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Physical CUDA HTTP integration

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Owner: batch-design agent
Scope: new `crates/zerogravity-cli/tests/serve_gpu.rs` only.

## Uncommitted
- [x] Read HTTP framing, shared output, native worker and source APIs.
- [x] Agree active=4 real gate request + paired source-alias admission.
- [x] Implement ignored release .8B/2B actual TCP/CPU-oracle qualification.
- [x] Require trained EOS within four approved assistant prefixes/eight CPU
  selections; actual JSON/SSE stop, EOS-excluded usage and native-batch delta.
- [ ] MAIN: compile and execute serial physical tests after public gate opens.

## Verification
- No Cargo/GPU by this agent; MAIN coordinates execution.
- `rustfmt --edition 2024` passed. Source tests are ignored release-only;
  compilation/physical execution pending MAIN and public gate qualification.

## B4/B8 continuation

- [x] Preserve both B2 tests; `start_rows` uses explicit native ceiling4/8,
  active8 and the existing bounded child-process owner (B2 retains active4).
- [x] Add source-specific wide tests looping B4 then B8 serially. All request
  sockets are written during observable real32-token GPU prefill; require all
  width rows queued before gate cancellation and response reads.
- [x] CPU-exact mixed JSON/SSE aliases, unequal maxima2..min(width+1,8),
  >=2 actual full-width receipts, and smaller odd-survivor band receipts.
- [x] Real trained-EOS JSON/SSE groups, EOS-excluded usage and single Done.
- [x] Authentic disconnect after source text while all width rows are still
  active; require gate+target cancellations, exact survivors and fresh replay.
- [x] Actual weighted native row accounting2*n2+4*n4+8*n8; sampler4/cohort;
  final active/queued/failed0 and no unexpected capacity refusals.
- [x] Rustfmt and diff check. Wire checks make no all42 or speed claim.
- [ ] MAIN: strict compile/Clippy and actual sequential source-wide HTTP tests;
  internal worker tests independently own all42 state proof.

````

## Source: .agents/runs/serving-and-speculation-p12-http-output-2026-10-19/TODO.md

Original UTF-8 bytes: 718; SHA256 `31ebb8e6cf3112b7aa843b8225b1db3cd03f5c7bd05c0ade6d392986b5679d3b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Bounded output presentation

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Owner: batch-design agent
Scope: `serve/engine.rs` GeneratedText and focused tests only.

## Uncommitted
- [x] Read current presentation and canonical root-first/at-most-once contracts.
- [x] Agree resumable presentation API and CPU fail-closed wrapper.
- [x] Implement bounded unsent Text/Done staging and terminal progress.
- [x] Add bounded-channel loss/duplication/EOS/UTF-8/CPU refusal tests.
- [ ] MAIN: execute focused tests and required gates.

## Verification
- No Cargo/GPU by this agent; MAIN coordinates execution.
- `rustfmt --edition 2024` passed. Eight focused bounded-channel tests added;
  execution pending MAIN.

````

## Source: .agents/runs/serving-and-speculation-p12-http-qual-2026-10-19/TODO.md

Original UTF-8 bytes: 1730; SHA256 `9b9892efbe4ea6563ffc4ea04ee62e13c69fc7cbf8fc8c59eb9e7f112bb98011`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12-B5 native GPU worker qualification

Parent: [serving-and-speculation](../../plans/serving-and-speculation/TODO.md)
Owner: baseline_review
Scope: new CLI `serve/gpu_engine/qualification.rs`; this run only.

- [x] Read existing worker, GPU packet/session, presentation and wave boundaries.
- [x] Draft ignored RELEASE .8B/2B physical tests over actual GpuActive/helpers.
- [x] Independent original CPU + accepted B1 packets/all42, B1/native/B1 handoff,
  unequal lengths, cancellation before feedback and after actual cohort Ready.
- [x] Real bounded-channel credit parking/resume and exact text/Done/usage.
- [x] Bounded source-real trained EOS search + GPU replay when found; explicit
  unqualified receipt/synthetic presentation policy only when no source EOS found.
- [x] Aligned final MemberStep/presentation/helper API and actual Ready hook;
  exact native model/sampler/commit counters, including discarded physical rows.
- [x] rustfmt parsed owned source; owned-path diff whitespace check clean.
- [ ] MAIN strict CUDA compile/review; source frozen pending requested repairs.
- First strict compile: fixed trait-object call coercion with source.clone().
  Matched worker empty-eligibility return (terminal/all-parked) before wave form;
  inspected actual Ready-cancel counter binding: one discarded declaration.
- Strict CUDA retry typechecked qualifier; changed release-only runtime refusal
  to if/panic for Clippy assertions_on_constants without an allow/const assert.
- [ ] MAIN release physical .8B/2B execution and separate actual HTTP wire gate.
- [ ] MAIN full isolated gate, strict feature gates, original direct/HF parity.

No Cargo, CPU/GPU inference, model/SDK downloads or commits executed by this agent.

````

## Source: .agents/runs/serving-and-speculation-p12-http-review-2026-10-19/TODO.md

Original UTF-8 bytes: 5393; SHA256 `e2c981fda08399f93f3dae16bec88375cda13c92eba63d6a44b53f2e570204ec`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12-B5 independent HTTP/state review

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Owner: `/root/spine_review`
Scope: read-only scheduler/transport/GeneratedText architecture review.

- [x] Read live B5 plan, joint serving contract and 36 acceptance gates.
- [x] Inspect existing GPU scheduler, config/metrics gates, shared output and
  HTTP disconnect/deadline/SSE lifecycle.
- [x] Send root/child recommendations: packet presentation survives rollback;
  pending Text/Done flushes resume without repeated token acceptance; parked
  rows excluded before feedback; B2-only waves; accurate physical/root counts.
- [x] Recommend cfg(test)-only one-shot after-Ready cancellation hook, with
  physical completion counted and all-row rollback before survivor replay.
- [x] Review final root/child implementation: packet presentation, staged
  output, Ready cancellation, per-row roots and completion metrics.
- [x] Review actual .8B/2B worker and HTTP receipts, trained EOS, all42
  internal state proofs, local benchmark smoke and documented limitations.
- [x] Identify and report exact499/cancelled classification; root repaired it.
- [x] Review public CUDA B2 mode gate, help, health and owning plan scope.

## Continued parity audit

Parent scope: same serving plan; root remains sole Cargo/GPU coordinator.

- [x] Read complete joint serving/goodput,36 SERVOPT acceptance, Qwen basics
  parity roadmap and vLLM prior-art reconciliation documents.
- [x] Inspect current native dispatch, scalar source dots, staged arithmetic,
  scratch/state transport, serving waves and benchmark/comparator sources.
- [x] Report comparator ABBA palindrome bug: reverse(ABBA) never emits BAAB.
- [x] Deliver prioritized remaining parity blockers and minimum implementation
  plus matched-benchmark evidence path, distinguishing text subset/full parity.

## Authorized measurement implementation

Root explicitly delegated ONLY `tools/serve-bench.py` and
`tools/serve-compare.py`; root retains MAIN tests and canonical plans.

- [x] Preserve positional APIs/legacy summary fields; add v2 raw records.
- [x] Open-loop scheduled workload with varied prompts, unique IDs/local tenant
  labels, bounded pending queue, launch deadline and accountable local refusal.
- [x] Classify HTTP200/429/503/499, cancellation, timeout and protocol/transport
  failure; retain every offered record and whole offered time denominator.
- [x] Separate TTFT/SSE chunk proxy from complete provider-bound token traces;
  no inferred token ITL, independent source proof or qualified SLO goodput.
- [x] Add opaque before/after provider snapshots on literal loopback HTTP;
  reject redirect/proxy escape; do not infer physical meaning from JSON.
- [x] Explicit ABBA/BAAB alternation with actual HTTP order fixture; compare
  varied prompts per identity, exact usage/finish and optional token IDs.
- [x] Add focused source-free fixtures for overload/denominator/SLO/trace
  binding and strict request validation. Files frozen for coordinator tests.
- [ ] MAIN coordinator self-tests/py_compile; repair any reported failure.
- [ ] Final read-only review of metric semantics and test receipts.

## B4/B8 admission review

- [x] Read-only review typed prelaunch `Result<Option<...>>`, allocation helper
  and dynamic8/4/2/1 regrouping. No source blocker: sole None precedes all GPU
  submission/allocation; all later failures remain Err; checked candidate+
  packet+new workspace bytes subtract only idle old workspace. Original queue
  retirement/requeue and actual Ready cancellation remain authoritative.
- [ ] Physical B4/B8 worker/HTTP receipts remain MAIN-owned and pending.

## Authorized GPU admission qualification

Root delegated ONLY new `resident_inference/admission_qualification.rs`;
root owns module wiring, compilation and all actual CUDA execution.

- [x] Add two ignored release .8B/2B tests, original CUDA device only.
- [x] Seed8 independent production/B1/CPU references; full vocabulary,
  compact ID/max and all42 domains with strict future KV zero checks.
- [x] Compute exact cold B4/B8 private state+packet+workspace bytes.
  Allocate actual uninitialized CUDA Scratch BufferLease leaving B4 bytes.
- [x] Cold/warm B8 host/device None, strict API Limit, invalid-host and
  swapped-producer Err. Check original queue counters, uploads, resident
  bytes, allocation-generation canaries, roots/state and idle handle IDs.
- [x] Execute two actual B4 feedback cohorts under pressure, release actual
  lease, then B8 continuation; independent CPU/B1 full vocabulary/packets/
  all42 and root invariants. Require final ledger0/globalreserved0/unfenced.
- [x] Standalone rustfmt passed; source frozen and parent notified.
- [ ] MAIN compile/Clippy and actual serial physical qualification.

Findings: HTTP remains native B2/equal frontier, active<=8/context512.
Quant projections cost8/10 launches each; SiLU13/14, softmax38, RoPE
half+11; source dots use F64-backed source mul_add. These are source
cost hypotheses, not measured kernel-time attribution. Current tools report
closed-loop identical-prompt throughput and chunk intervals, not true token
ITL/SLO goodput, tenant fairness or independently verified source identity.

Checks/results: source review only; no Cargo/GPU processes. Root owns all
implementation and test execution. No root README edits, downloads, privileges,
publication or AMD launches.

````

## Source: .agents/runs/serving-and-speculation-p12-http-root-2026-10-19/TODO.md

Original UTF-8 bytes: 1858; SHA256 `5f29a1c58d8f45b966463e7ac99fd96669b7c31a7b2313b91ce40f1a22b0c6e6`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12-B5 CUDA B2 HTTP integration

Parent: .agents/plans/serving-and-speculation/TODO.md
Base:570d865696ab398f0c5c37f08c5c951595eadd75
Owner:root; sole MAIN Cargo/physical GPU/test coordinator.
Root owns gpu_engine.rs/mod.rs and any public metrics getters; child output owns
engine.rs GeneratedText only; child qualifier owns newgpu_engine/qualification.rs;
independent reviewer read-only. Root alone reconciles owningplan/signedcommits.

- [x] Revalidate cleanmain+14/soleworktree/signedHEAD/idleRTX3060/unchangedREADME.
- [x] Re-read live AGENTS/queue/ownplan/canonicaljointcontracts and currentcode.
- [x] NativeB2 prompttail/decode with persistentpacketpresentation and finiteparking.
- [x] Actual original .8B/2B state/output/EOS/cancel/credits/handoff qualification.
- [x] Actual local HTTP/SSE nativeCUDA tests and tools/serve-bench.py exercise.
- [x] Strict relevantClippy/sourcefree/fullisolatedgate/bothreleaseparity routes.
- [x] Review, signedlocalcommits, owningplan reconciliation. No push.

No downloads/SDK/deviceprivileges/recovery/AMDlaunches/rootREADMEchanges.
Dotlink tools absent, local workspace used. FullvLLM benchmark remainsunavailable
until genuine pinned equivalentruntime; no throughputparity claim.

Physical: both worker tests all cases incl trainedEOS/CPU+B1 all42/finalreserved0.
Actual HTTP: both tests pass, native cohorts9/8, Ready discards1 each,
final active/queued/failed0; both real EOS JSON/SSE and disconnectsurvivorreplay.
Local release serve-bench:4/4complete16usage tokens/native8 each; noSLO/vLLMclaim.
Fullgate exit0:1147/0/69 over245 suites,18Python; CPUserve14/GPU53(3ignored).
Strict CPU/CUDA/Vulkan/qualification and both release sourceparityroutes passed.

Finished: signed implementation2677d8b +acceptance7fc9ad4. Cleanmain+16;
rootREADME/sourcehashes unchanged, idleRTX41MiB/0%, no externalpublication.

````

## Source: .agents/runs/serving-and-speculation-p12-native-bands/TODO.md

Original UTF-8 bytes: 1320; SHA256 `2b417948b38ed0faf98a795bf6ceed6d10d61075f6a3ee4447d6482f03caf570`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12 native HTTP bands

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Owner: `/root/batch_design_review`
Scope: `serve/gpu_batch.rs`, `serve/gpu_engine/cohort.rs`; no Cargo/GPU.

## Completed

- [x] Finite ordered 8/4/2/1 band planner; no padding; exhaustive 0..64
  rows across ceilings 1/2/4/8 plus survivor/capacity/refusal cases.
- [x] Both original prompt-tail and device-feedback cohort paths repartition
  eligible rows; retain phase, prior StateRoot/packet and presentation marks.
- [x] Dedicated admitted APIs: only prelaunch capacity `None` retries smaller
  bands; execution errors fail closed. Known-Ready cancellation discards all
  private proposals then regroups eligible survivors.
- [x] Native width counters require actual completed model/sampler receipts.

## Verification

- [x] Rustfmt and owned diff review; `git diff --check` clean.
- [ ] Main coordinator: focused tests, strict Clippy, release physical
  source/StateRoot/cancel/admission/output cases and complete isolated gate.
- [ ] Main coordinator: shared wiring/metrics, admitted GPU APIs and owning
  plan reconciliation. No physical support or performance claim from edits.

Attempted rows count pre-prepare admission attempts for in-flight visibility;
completed/native-band counters count actual terminal completion receipts.

````

## Source: .agents/runs/serving-and-speculation-p12-wide-worker/TODO.md

Original UTF-8 bytes: 1736; SHA256 `81b870ddf1f9d52587b1ec0c5d51926f842da82a981ab325e3a828d3d7c4cf3d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12 wide native worker qualification

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Owner: `/root/batch_design_review`
Scope: `serve/gpu_engine/qualification.rs`; no Cargo/GPU.

## Completed

- [x] Preserve B2 tests; configurable B4/B8 fixture and original worker rounds.
- [x] Four ignored release tests: approved original 0.8B/2B × B4/B8.
- [x] Actual native prompt tails; distinct input/session IDs; unequal maxima
  1..4/8; native packets consumed by B1 residuals; B1 packets consumed natively.
- [x] All live checked frontiers compare original CPU+B1 packets and all42
  physical domains. Actual per-width receipts and single-root transitions.
- [x] Known-Ready cancellation discards wide candidate; 3/7 survivors regroup
  2+1 / 4+2+1; preserved canceled snapshot and once-only output.
- [x] Real capacity1 output park; all healthy survivors advance while slow
  original packet/root stays unchanged; resume exact text/usage/single Done.
- [x] Bounded original CPU-selected trained EOS; native wide prompt tail;
  independent siblings; EOS-excluded usage and single terminal event.
- [x] Final original ResourceArbiter reservation must return to zero.

## Verification

- [x] Rustfmt and owned diff review; `git diff --check` clean.
- [ ] Main coordinator: compile/strict Clippy, sequential release physical
  tests for both sources and widths, full isolated gate. No new physical or
  throughput claim until executed.
- [ ] Main coordinator separately owns physical GPU-budget refusal probe,
  actual HTTP wire qualification and owning plan reconciliation.

The tests retain live B1 reference sessions in addition to each worker session;
their original physical leases count against the same finite provider budget.

````

## Source: .agents/runs/serving-and-speculation-parity-root-2026-10-19/TODO.md

Original UTF-8 bytes: 120546; SHA256 `42e961f462de77f84a032f7dd930a6bcb6a6b9b757ea5735981c8edb38f54c01`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Serving parity through trained Qwen3.8 27B

Parent: .agents/plans/serving-and-speculation/TODO.md;
model work also .agents/plans/model-execution/TODO.md.
Base:7fc9ad475a30a3d2120f307a7da2ff75a80e337a.
Owner:root, sole MAIN Cargo/physical GPU coordinator and plan reconciler.
User authorizes recommended implementation/runtime/model acquisition/testing
needed for actual vLLM parity, matched benchmarks and Qwen3.8 27B, no prompts.
Preserve rootREADME, existing one-engine ownership and AMD launch hold.
No push/tag/deploy/destructive history needed or planned.
Dotlink-l5 callable tools still absent; direct workspace tools available.

- [x] Revalidate cleanmain+16/signedHEAD/soleworktree/idleRTX3060.
- [x] Native B4/B8 serving, live survivor repartition, exact state/output proof: d9fa829.
- [ ] Unequal-frontier/native prefill/paged state and dispatch performance.
- [ ] Genuine pinned vLLM V1 source/quant/tokenizer-matched environment.
- [ ] True token events, SLO/fairness/resource/cold-warm matched benchmark.
- [ ] Remaining applicable canonical parity behavior and failure gates.
- [ ] Original trained Qwen3.8 27B source/BF16-FLA independent truth;
      physically admitted streaming via existing engine; actual model tests.
- [ ] Full isolated gate, physical qualification, release parity, reviewed
      signed commits and canonical owning-plan reconciliation for each phase.

## Current phase

- Native B4/B8 worker/HTTP survivor regrouping implemented, frozen for physical
  qualification. Typed actual-capacity None means only prelaunch refusal;
  GPU/unknown failures propagate and cannot choose smaller retries.
- Combined CLI/CUDA/cache + GPU/qualification strict Clippy passed.
  New real-lease admission tests compile under strict Clippy; release LTO build
  underway. No new physical result accepted yet.
- Measurement tools: bounded offered-arrival/tenant/status records, token-trace
  bindings and actual ABBA/BAAB order fix. Both isolated self-tests/pycompile
  passed. No provider traces/matched performance yet.
- Authentic original-HF tied head fix and focused source-free test frozen;
  no promotion of BF16 diagnostics to trained execution.
- Acquired genuine vLLM0.31.0-cu129 immutable image; Python3 imports and
  actual NVIDIA CDI context/device query passed. CLI imports fail because
  bundled TorchCodec requires libnvrtc.so.13. Resolving reproducible text-only
  runtime without fake modules or mathematical patches. No model execution.
- Original Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
  original Rust acquisition exit0, selected55,586,113,293B, snapshot
  98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e.
  Independent31-file/18-shard full hashes, exact1199-header/index contract
  passed. Source preparation only; no full27B RAM load or inference.
- Spine prepares official .8B BF16 CPU reference, full logits/all42 typed
  states. Explicitly distinguish official BF16 loader cast of36 originalF32
  parameters from source-mixed-precision reference. NoGPU/FLA equivalence.

## Executed / failure follow-up

- Pressure tests both original.8B/2B exit0 (75.22s): real2.54/1.94GB scratch
  leases; cold/warm B8 refusal leaves allocation/queue/upload/root/all42 and
  idleworkspace unchanged; actualB4 underpressure thenB8 afterrelease exact
  CPU/B1 fullvocab (independent B1), native compactpackets/all42; finalreserved0.
- First combined6 worker process: .8B B2/B4/B8 allpassed fivecases, then2B
  B2 failed actual418,381,824B CUDAallocationOOM; next2 cases refusedretained
 4GiB reservation. Whole runexit101/3pass3fail244.23s. No acceptance of2B
  orwiderHTTP fromthisrun. Failedservice remainedfenced/unreleased inprocess,
  process exitednormallytestfailure; cardidle41MiB afterwards, noreset.
- Added cold-load originalqueuedinitialization memory_cleanup to drain
  completeddeferred frees/cachedoldstream pools; never releaseslive/graphpinned
  orquarantinedowners, guarderrorsstillfence. CombinedstrictClippypass7.83s.
  Need rebuild/repeatSAMEcombined6process toprove causality/reclamation.
- Started isolatedoldbinary2B B8 negative/control check (newcleanupNOTinbinary).
- Fullisolatedcheck.sh beforecleanup exit0:1148pass/0fail/69ignored/245suites,
 18Python +benchselftest/release/installer/Nix. Includes newtiedheadtest.
  StrictCPU/Vulkan/qual/sourcefree+bothreleaseparityroutes inprogress.
- Independentreferencecaptures .8B official473BF16 vsoriginal437BF16+36F32
  profiles bothCPUtwoTokens11,353 actualfull248320/all42; completehasharchive
  validatorpass. Mainmanifest e007d5e8058a218614a6135d829d35ad5a3b72d4dbdc4916a39498426057cd76.
  Actualargmaxofficial321,599 vsrestored198,599; noRust/GPU/FLAparityyet.
- Twoindependentimplementationworktreesauthorized/created(base7fc9ad4+B6
 8fileoverlay+pressurefile): zerogravity-worktrees/p12-ragged-20261019 owns
 perrowfrontierfullnative; p12-hf-axes-20261019 ownsMatrixAxes/rawsourcevalidation.
  NoCargo/GPU/MAINeditsbyagents. Rootalone reviewsmergesverifiesnewdeltas
  afterB6acceptance. Preserve current rootcoldcleanup whenmergingparentfile.

## Continuation 20 Oct

- Signed f26232e tied-head source fix, reviewed3ownedpaths only.
- Signed6bfcdbe offered-work/token-trace/ABBA accounting tools+owningplan.
- Finalaftercleanup isolatedcheck1148/0/69/245+18Python; strictCPU,
 CUDA,Vulkan/qualifierchecks andbothapprovedreleaseparityroutes pass.
- RepeatedSAMEcombined6worker process withcleanup:6pass0fail587.63s;
 all.8B/2B B2/4/8 originalCPU/B1packets/all42 andfivescenario proofs.
 Priorpool2.51GB/2.13GB physically0aftercoldcleanup,no liveallocs.
- Finalbinarypressure2pass0fail44.08s: sameactualleases/prelaunch
 refusals/B4underpressure→B8afterrelease/fullvocab/all42/final0.
- FirstfinalHTTPattempt operationalfailure: fullcheckoverwrote
 target/release/zerogravity withnonserve build; no inference ran.
 Rebuildingexplicitserve,cuda,cache server thenrepeatall4tests.
- GenuinepinnedvLLM0.31 V1 actualCUDAweights/load/Torchcompile/graphs,
 health/model/twoteacherIDs+4requestwarmtools smoke passed. Owned
 containerstopped,cachepreserved. No matchedperformance/parity claim.
- OfficialHF.8B source-mixed captures+existingRustdiagnostic actually
 compared,UNQUALIFIED(embeddings/normexact,firstBF16projection1/3diff).
 NewexplicitCPU64laneprofile frozenisolated,independentreviewpending.
 AuthenticadditionalGDNslowchunk/recurrent50/40actualtensorcapture
 passes18.9MiB, fullvocab/allrawcache98hashesmatchpriorcapture.
- B7unequalfrontier andsourceaxes patches reviewedroot,awaitB6commit
 beforemerging. Allotheragentworktrees preserved, noagentCargo/GPU.

- B6 HTTPrebuiltrepeat4pass0fail318.19s. Bothmodels2/4/8 passactual
 JSON/SSE/EOS/liveDisconnect/survivor/replay, finalactive/queued/failed0.
 Signed13reviewedownedpaths d9fa829, canonicalserving3docsreconciled.
 Priorfullchecknonservebinaryfailure retainedseparately39.26s.
- CPU64profile8filepatch +Axes5filepatch mergedMAIN disjointB6scope;
 focusedCPU/Core/Model/facade147pass0fail1ignore, strictalltargetClippypass.
 Root corrected knowninaccurategatednormcomment withoutmathchange.
 FreshCPUreleasefacadelibs build21.95s, newexplicitCPU64harnessprepared
 underroot/cpu64-harness, needscompileandactualreferenceexecution.
- B7all1849lines root+independentspinereview no blockers, appliedMAIN
 preservingcoldcleanup/axesprefight, no compile/runtimeacceptanceyet.
- Spinefrozenmetrics9pathdelta reviewpending; authorized secondbounded
 producerfrontier/nativecohort metadata +actualunequalarrivalworker/HTTP
 tests inisolatedtree withB7overlay. No math/defaultmetrics changes.
- BaselineisolatedproductionHFsourcebinding/typedroles +explicitunqualified
 profile guard underway fromsameLoadedQwenModel/ForwardPlan; noexecclaim.
- Batchagent actualCPU128L2/recurrentproducts topology14checks12416values
 exact; newcompactpureCPUhelperpatch underway independently, defaultGGUF
 unchanged. NoagentCargo/GPU. Rootstillsolecoordinator.

- B7GPUreleasebuild3m30s successful, binaryzerogravity_gpu-d687351382267e66
 includesB7+Axes+CPU64+pure128reducers, beforeHFbinding/metrics/tracenorm.
 Actualtwo-modelunequalfrontierqualification running serially(session6817).
- CPU64harnessfirst actualexit2 numericalUNQUALIFIED (notoperational): both
6144layer0QKV exact vsold1/3diff, convolution0state nowexact;41/42domains
 stilldiffer. Fullvocab214245/214285different,maxabs.22119140625/.1875.
 Firstdifference token11convSiLU index2527 signedzero-0/+0; token353first
 GDNstate. Exactnonewlimits. Agentactualpreactivationpositive0 at2527,
 4separatemulproductsallnegative0, TorchSiLU(-0)negative0→conv cause;
 backendorderderivationpending,nocanonicalization.
- HFbinding6pathpatch+fixedmetrics9pathpatch mergedMAIN onlyafterroot/
 independentreviews. Cargo.lock exactlyoneexisting"sha2"modeldevdepentry.
 Newmodelsourceprofilemetadataonly;allHFexecution/checkpointv1refused.
- Addedrealattention_input_norm optintraceobservations atbothactualnorm
 sites; existinganalytictracetestassertsall4layers/firstBF16norm2.
 Noarithmeticchange. NeedrepeatactualCPU64harnesswithfreshtracecoverage.
- FirstcomposedstrictClippyfailedONLYmetricsnestedcollapsible_if; fixed
 letchainwithoutallow, repeatedlockedofflinechecks underway(session35384).
- MetricsindependentreviewnullIDsbugfixedbeforemerge; boundedIDbinding,
 monotonicdeliverywrites/receives, strictintegerchoiceindexfixturespass
 Pythonselftests(byagent), Rustnotyetexecuted. Phase2producercohortmetadata
+actualunequalarrivalworker/HTTPtestsstillisolatedspine.
- Closedpure128CPUreducers3paths merged; sourceobserved14checks/12416values
 fromactualTorch, Rustgoldentest8412bytespendingexecution. DefaultGGUF
 operations/profilesunchanged. CachedGDNpurecoreseparateagentunderway,
 explicitobserveddecayinput; coefficientexp/initialchunkunqualified.

- B7physical .8B B2/B4/B8 PASS full248320/all42 exactacrossmixed
 [255,256,0,31,32,1,510,511], actualinactiveAVpoison/signedzero/subnormal,
 Readyabandon/stalepacket/refill/B1handoffs/noWarmUploads/final0.
 2BsourceCPUhydration/physicalrun continues serially(session6817).
- Correctedfixture halfULP16384→16385 confirmedbyactualpinnedTorch
 positiveandnegativesigns; helpermathunchanged. Realcompactgoldensalready
 passed. Focusedfirstprivate-networkrun failedsocketloDown; rerununder
 existingprivateuser/netownerguard (nohostchange). Phase2merge thenfound
 missinggpu=Noneonlytestinitializer; fixed. FinalcomposedCPUtests185pass,
 0failed,2ignored across6suites, includingallmodelbindings/refusal/trace
 andmetricsSocket/UTF8/ID/clock/defaultoff tests.
- Finalphase2producer/frontier patch merged, root reviewednativecounter/
 retry/packetroot/HTTPmembership fields. New2worker+2HTTP testsnotyetrun.
 RootmetricsPythonselftests/pycompile pass beforephase2, finalrepeatpending.
 Strictcomposedlatestphase2CUDA+HF+portable alltargets running81314.
- Agentindependentconvprofiler6144slowconv calls,noMKLDNN; source+actual
 libtorchdisassemblyexcludesinternaloneDNN(m*n*k28<=4096), genericBF16NN
 fourtaps positive0accumulators,separateproducts. Pureclosedprofilehelper
 isolatedunderway, defaultGGUFandSiLUunchanged.
- CachedGDNpureCPUcorepatchfrozenisolated, notmerged. Actualquerydivide
 matches2048sourcewords; reciprocalmultiplydiff783. Compacthead0fixture
136276B +full16observedfixtureignored; needrootRustqualificationlater.

- B7 fullmodel unequalfrontiers .8B/2B BOTHPASS:2tests/805.11s;
 B2/4/8 all248320/all42 exactmixedpositions255/256/0/31/32/1/510/511,
 inactiveAVpoison/signedzero/subnormal, packet/refill/Readyrollback/final0.
 BinarypredatesHFmetadata/metrics; freshservingphysicalacceptancepending.
- Currentcomposedfullisolatedgate1164pass0fail69ignored245suites+18Python;
 strictCPU/CUDA/Vulkan/qualifier allpass; CPUserve22pass; GPU55pass3ignore.
 Bothapprovedreleasecache-direct17.22s/HF-import22.94s exactfull248320/all42
 fororiginal.8B/2B. RootCPU64actualHFstillUNQUALIFIED beyondexactQKV.
- FreshreleaseCUDAserve/tests buildsession79389 underwayafterfullcheck;
 planned8worker+6HTTP physicalserial. Newclosedconv/cachedcore/tensorwindow
 patchesfrozenisolated NOTmerged; benchmarklifecycleisolatedunderway.
- Signed9815504G bindsreviewedHFmetadata/Axes/CPU64/closedreducers/trace;22ownedpaths only. B7/metricsremainunstagedpendingphysicalgates;rootREADMEhashunchanged.

- FreshCUDAservebuildexit0 14m; currentCLIunit35pass0fail8physicalignored.
 Actual8workerphysicalcases launched25165,aftermatchingnewbinary; nootherGPU.
 Qualifierfreshabort-enabledreleasebuild23317 nowCPU-onlyparallel withworker.
- RootactualRustclosed128replay againstpinned4.3MBarchivePASS14checks/
12416F32values,0bitdiff (all16heads,2tokenL2/keyaxis memory/output), only
 committedpublicCPUhelperscalled, nogeneralmodel/FLAclaim.
- NewactualgatedRMSobservationattempt1failedobservercore_fp32matches[];
 retainedexclusiveinvocation/log/source. RoottinyactualTorchprobeexit0
 demonstratesinference_modeTrue dispatchcastaten.to.dtype vsordinary
 aten._to_copy.default. Agentpreparingattempt2matchingactualoplabels;
 noactualnewnormevidenceacceptedfromfailedcapture.
- IndependentartifactreviewfoundspareVeccapacity+perwindowoffsetmetadata
 accountinggaps; isolatedrevised7pathpatch0872794d...awaitsreview. Notmerged.
 Lifecyclev1receiptIOcouldskipcleanup; isolatedv2fix172b906d...retainedv1,
 sourcefreerealownedfailurecleanupfixturespass, rootreviewednotmerged.

- LatestB7releaseworkers8/0 PASS644.37s BOTHmodels B2/B4/B8 +actualunequal
 arrivalmetrics withGPUlongprefill/capacity1park/nativefrontiers/CPU-B1all42.
- LatestHTTP6/0PASS371.29s bothmodelsB2/B4/B8 JSON/SSE/EOS/disconnect/
 survivor/replay plusactualBofferAFTERAfirstID; sharednativeB2publication
 sequenceobservedatdistinctfrontiers, exactCPUallIDs/text/usage/oneterminal,
 healthyactive/queued/failed0. Freshcurrentbinarybeforethisphaseallsource.
- FreshqualificationCUDAabortbuildexit0 3m50; currentGPUlibreleasebuildexit0
3m48,58sourcefreepass0fail16ignored. Actualpressure2+abort4 nowrootserial.
- Newgatednormcaptureattempt2 actualPASS12stages×2tokens19,361,920B,
98priorfull-logit/cachechecks unchanged; validatorpass. ManifestSHA
225e612793d9896ee3ad843d8c3f845fde217e4d50c7eceba0bf377dcaa050b7.
- RootactualRustscalarSiLUon4096capturedBF16gates:F32bits106/89diff,
 max2.98e-8/5.96e-8; finalBF16withactualweightednorm0diffbothsteps.
 ThisdoesnotqualifyfullSLEEF/MKLtranscendentals orwholemodel.
- Initialchunkpublicprobeattempts1/2/3 retained;1failedduplicatedinline
 comprehensiontracepoint,2/3failedexactunitaxisstrides aftersame-shape
 reshape normalization. ActualsourceBMM/STRSMMKLdispatch observed2/3,
 butpublicreplaynotaccepted; agentfixingno-relaxationobserverforattempt4.
- Nextphasepatches allisolatedandreviewed exceptnewcoeffcaptureprep:
 conv2paths, cachedrecurrent4paths, norm3paths (preSiLUonly), tensorwindow
7paths0872794d..., lifecycle2paths172b906d... . DO NOTmergeblindlybefore
 B7verifiedcommit. Fullgates/currentstrict/approvedreleaseparityalreadypass.

- B7 finalphysical pressure2PASS73.78s; originalbinary-nameabortattempt127
  launchedNOabortGPUwork. Corrected4actualprobesPASS, originalhandles
 1824/1719 and1828/1723 allquarantined, roots/all42exact, no recovery.
- Actualinitialchunk root-replay-4PASS50frames/strides,6BMM+2STRSM public
  replays+4syntheticpatterns; noRust/MKLorfullmodelqualification.
- ActualGDNcoefficientcapturePASS19,389,504B+12normstages/step, all98prior
  fullrawlogit/cachechecksunchanged; validatorPASS. Closedexp/log pending.
- B7independentcounterauditexact204414=26*5489+6*5510+5*5728; no inflation.
  Rootreconciling3servingdocsandpreparingsignedreviewedB7/metricscommit.

- B7signeddbb5f4fG,28ownedpaths,cleanMAINbeforefollowingmerges.
- Closedconv/cached/norm/SiLU frozenpatches mergedserializedwithonlymodule
  declarationscomposed; focused28/0Rust. RootactualpublicSiLUoraclePASS
 65280finiteBF16+4096actual, observedSLEEFAVX2pointer. ActualRustclosed
 replaysALLPASS:SiLU69376words,cached272384words,norm8256words,
 conv6144channels×4positionspre+SiLU49152words. No fullmodelpermit.
- Tensorwindow7paths+lifecycle2pathsmerged. Sourcewindow9/0+SSDfocused6/0,
  toolsbench/comparator/lifecyclefixturesPASS. FullgatefirstfailedONLY
  literalreversedrangeClippytestfixture; preservedlog, Range{start,end}
  preservesnegativecase. Finalfullgate1202/0/69ignored248suites+18Python.
- StrictCUDA+HFbinding+portable/Vulkan/qualifierallPASS, CPUfreebinding8,
 CPUserve22PASS. BothapprovedreleaseparityroutesPASS17.28/23.21sfullV/all42.
- ActualserializedownedGPUproviderlifecycleB7ZG+immutablepinnedvLLMPASS
 startup/onewarmrequest/cleanup, bothPGIDs/endpoints/logreadersgone;
 VRAM1059/4203MiB live,41MiBaftereach. UNMATCHEDsources/math,NOTbenchmark.
- Usernowexplicitlyrequiresindependent8K GDNdivergenceclosurebothGGUFmodels,
3modelcorrectness,bothbenchmarks,genericmachineryoptimizationtowardsbeating
vLLM. numericsplanreactivatedP1; no tolerancewiden/promotion.
- Actual27BwindowharnessfirstdirectrustcLTOlinkfailedmissinglibcbitcode,
 NO payloadrun. IsolatedmanifestofflineCargo.lock+lockedbuild underway;
 pin/sourceharnessunchanged, failurelogretained. Nextwhole55GBboundedscan.

- Signedcadcbd9G:12ownedCPUoperator/planpaths; 1b599dfG:5ownedlifecycle/
  servingplanpaths. CPU-freebindingfirstfiltermatched0(retainedlog);
  actualunfiltered6PASS, composedCPUbinding23PASS2ignored. Correcteddoc6.
- Actualunchanged27BsourcewindowharnessPASS18shards55,563,006,776B,
 82.847s;12byteexactBF16row samples/4splits,maxstage8,388,656B,
 finalledger0, peakRSS17,328KiB under512MiBaddresscap. NO inference.
- Softplusnewfrozen3pathdelta rootread, actualpinnedpublicoraclePASS
 28464adversarialgeneralF32+32actual; Rustreplaypending/notmerged.
- Nextisolatedagents:Spine typedcore/profileproject+sameHostMath+GPUcold
 refusal; Baseline typedrawparams/state metadata thenlongcontextaudit;
 Batch sharedquantpreparationdispatchoptimizationdesign. RootsoleCargo/GPU.

## Typed source integration / activation preparation continuation

- Exact typed11path delta deaf2dd... + raw3path delta746889... root and
  independent reviewed, applied with fresh mtimes. Pure embedding→projection
  slice uses existing PlanBuilder/HostMath; HF full model/session/checkpoint
  admission stays closed. Composed focused checks pending.
- Closed softplus actual/adversarial public Torch→Rust bit replays passed
  (32 actual +28,464 F32 +reordered/spans16/32/48/64); final gate pending.
- Exact activation-prep candidate4c4daed... merged two new files with
  test-only CUDA/qualification wiring. Three source-free tests passed;
  272-case physical GPU differential qualification still pending.
- Separate numerical run has fresh teacher cases512/1024/2048/4096/8192,
  manifest-pinned raw IDs shared by both original GGUF models. Historical
  raw prefix unrecovered; no claimed replay of old8K evidence. Original
  native source tar verified against preexisting NAR source pin; installed
  runtime unchanged. Long-context GDN discrepancy remains open.

- Signed2a32aff typed/raw/sharedoperator+closedsoftplus18ownedpaths.
  Fullisolated1213/0/69/248+18Python,strictCUDA/HF/CPUfree/Vulkan/qual,
  focused158/0/2ignored+CPUfree10+GPUcoldrefusal2, actualpublic6operator
  replays518144referencewords/sourcelease0 andbothapprovedrelease
  parityroutes17.15s/23.12s pass. HFfullguard remainsclosed.
- Productioncoldactivation4pathb3642153 independently/rootreviewed,
  appliedwithfreshmtime; modulewiredproduction. Kernelbody/qualifier
  unchanged; initial272casesexact1483608words, newcachedAPIphysical
  andwholemodel/HTTP/cancellation/performanceproofpending.
- Sigmoid3pathc6553374 root+independentreviewed, applied. PublicTorch
  exhaustive16cases326753words/providerpointers+codehash/bytevalidator
  pass94ca701..., Rustreplay/focused/fullgatespending.
- Nativefresh512AR1capturesbothoriginalmodels730records/instrumentation
  fullV+wholestateexact. Firstfresh.8Bcut2CPU/native actualcomparison
  all730/11331616wordsexactincludingzeros; this isinstrumentationsmoke,
  notlongcontextclosure. Fresh.8B512Rustprefixnowrunning.

## Closed sigmoid/ordinary RMS + cold GPU preparation continuation

- Full original.8B/2B CUDA B2/4/8 unequal-frontier full248320/all42/B1/CPU/refill/abandon/stale-packet/final0 PASS:902.72s, frozen941dce53...; pressure2/0/final0 PASS50.27s. Serving/abort/A-B timing pending; no vLLM performance claim.
- Genuine BF16 sigmoid326753 output words0diff; ordinary RMS root-probe2 actual122calls/98priorchecks and32publicfixtures, regenerated full+compact bytes exact, Rust154cases/1694fields/1680488words0diff, norm7tests/strictCPUClippy PASS. Independent baseline review NO BLOCKER.
- Full isolated current composed gate1218passed/0failed/69ignored/248suites+18Python/release/installer/Nix PASS. Extra strict CUDA/Vulkan/qualifier and approved release routes repeated before RMS; repeating final composed checks next.
- Native ABI probe first --locked build refused standalone lock normalization; actualoffline regenerate has all registry tuples/checksums equalMAIN. Preserved original lock and failure log; fresh locked build underway. No runtime/math result yet.
- Fresh numerics bothmodels512/1024 raw730records exact; .8B2048 raw730exact. Independent rawpoststate42domains .8B+2B512exact. .8B1024bulk64 control FAIL543stages/9230360words, firstobservedlayer4priorrecurrent, nofirstcausalproof. Separatecut3shapecontrol730exact doesNOTbisect unchanged64row schedule.

## 20 Oct next P12 cold-preparation qualification

- [x] New CUDA prepared Q8 blocks:272 cases/all7 buffers/1,483,608 words exact;
  same original buffers/queue/flight/Ready, no new persistent device scratch.
- [x] Both fullmodels nativeB2/4/8 unequalfrontiers full248320/all42 exact vs
  independent accepted CPU AND B1;2pass/902.72s/final0. Pressure2/50.27s.
- [x] Frozen fresh serving worker8/0/756.11s + HTTP6/0/416.07s; exact trained
  IDs/packets/state/EOS/cancel/credit/disconnect/replay/final0 boundaries.
- [x] Four fresh fullgraph/sampler unknown-completion probes pass original
  quarantine1824/1719 and1828/1723 handles/unchangedroots/all42/4GiB/noRecovery.
- [x] MAIN fullisolated1218/0/69/248 +18Python/release/installer/Nix, strict
  CUDA/Vulkan/qualifier andbothoriginalreleaseparity passed currentphase.
- [ ] SameZG oldB7/newprep ABBA/BAAB cold/warm measurement inprogress, NOTvLLM.
  First actual attempt completed8/64tokens but0native/56typedrefusals;device4096
  equaledprovider4096 leaving0globalstagingcapacity. Preserved UNQUALIFIED.
  Root+independent source audit fixed BOTH providers device5120 to match accepted
  physicalHTTPcaps, provider4096 unchanged; allnumeric/nativeB8/no-refusal gates
  unchanged. Newdriverf06a088c, actualwarmABBA.8B rerunstarted20:11UTC.
- [ ] Review/owningservingplanreconcile/signedGPUphasecommit aftermeasurements.
- [x] Isolated typedordinaryv2 source reviewed root+Spine, core3/model6 focused
  tests pass. Genuine independent .8B public embedding/norm/QKV reference +same
  engine v2 replay ALL16384 BF16-widened words exact/finalsource0. Not MAINyet.
- [x] Actual nativepublicABI diagnostic12cases/42calls/1,048,640words exact with
  explicit child4CPUaffinity/fullDSOclosure, process0/cleanup[]; production native
  allocation/profile/fullGDN/FLA remains unqualified. Earlier affinityfailure kept.
- [x] Actual original27B public3operatorreference40960 words captured/validated,
  fulloriginalBF16 geometries/sourceFDhashbeforeafter, process0/cleanup[].
  Rust27B replay remains unqualified; sealedparameterwindowpatch frozen/reviewed.
- [x] Actual attentionoracle12calls+2rotary/98priorchecks/2802publicchecks pass;
  separatepublicmath36actual+432BF16BMM+3584softmax+3refusals pass CPU4/AVX2.
  Rustattention/compiledBMM/fullprofile remains closed. Initial staticnm probe
  refused full demangledtable>128MiB; streaming replacement prepared, unexecuted.

## Current continuation20Oct20:34UTC

- [x] SignedGPUphase d0a6084 after full actual CUDA/worker/HTTP/abort/gates; indexempty.
- [x] Warm sameZG ABBA.8B andBAAB2B PASS; each32offers/32completed/256tokens, allsamples8B8/zeroerrors/refusals/exactoutput. Meanwave39.14206→33.163665s and45.81464→43.04175s; dispatches2072360→1722640 and1949016→1668496. Both10sTTFTSLOmiss/goodput0, no vLLM claim. ColdABBA bothmodels currentlyrunning frozenbinary/cap-equivalent serialproviders.
- [x] Typedordinaryv2 +sealedrank1parameterwindows +closedsoftmax mergedMAIN after source reviews; freshMAIN executionpending. Actual.8B16384word replay had used same isolatedv2 bytes/finalsource0.
- [x] CPUrowwindow wrapper finala43709fa mergedMAIN after rootwholepatch+Spineindependent review. ExistingCPU64 reducer unchangedexceptvisibility; authentic fullmatrix/range/offset/issuer, no partialTensorSource/providerfallback or outputadmission claim. Wide5120 fixture chunk fix1024 only; production32chunkbound retained. No compile/runtime acceptance yet.
- [x] Nativefullprefix frozeneba53c63+counterc253cc68+wiring5a3896f0 source-reviewedroot+Spine, qualifiers batch-reviewed. Followup32068e3d explicitly covers futurefullV/greedy rowcounts/nonemptyHTTPfirstfrontier. Not merged/tested yet. Single-request1token-pass regression mustmeasure.
- [ ] Aftercoldtiming: authenticate/exportpublicsoftmax/fullRustreplay+compactdefaultgolden, freshCPU/core/model/artifactstrict/focused/full gates; actual27Bparameterwindow/rowmath and sameHostMathfragment integration.
- [ ] AV public replay65default/control, fresh6K/8Kbothmodels and2Bbulkcausalcontrols; independentlongcontextrequiredgate remainsopen.
- [ ] OriginalHF2B officialpin/index/config/tokenizermetadata audited bySpine; original4.548GBweightsnotyetacquired. Useexistingboundedindex-firstRustpull aftertiming.

## Current continuation20Oct21:06UTC

- [x] SignedCPU5350a4d: typedordinaryRMS/rank1sealedwindows/authenticBF16full-column row projections/closedsoftmax. MAIN-linked .8B16384words exact; fullsoftmax3599cases1838736words3refusals exact; compact72cases11280words3refusals default. Fullisolated1241Rust/0failed/70ignored/248suites+18Python/release/installer/Nix PASS; composedCUDA/HF+cache/Vulkanqualifier strictPASS; approved cache-direct18.88s/HF-import24.75s bothmodelsfullV/all42 PASS. FullHF/session/checkpointguardsclosed.
- [x] SameZG oldB7→newprep cold+warm16samples128offers/128completed/1024tokens actualPASS; all8B8cohorts,exactalloutputIDs/text/usage/finish,zeroerrors/refusals/discards. Coldmean.8B45.657195→39.34474s(13.8%),2B51.963075→48.845175s(6.0%); warm15.3%/6.1%. All10sTTFTSLOmiss/goodput0; nevermatchedvLLM.
- [x] OriginalHF2B officialsourceactualacquired12files4571272453B,snapshot88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52; independentall12officialGit/LFSdigests/all632headers/all320textshapes PASS596BF16+36F32. Explicit4.548GBshardcapneeded; default4GiBunchanged. GenuineCPUtwo-profile referenceactualstarted21:05UTCafterrootfullreview/freezeverification; no numericalclaim yet.
- [x] Nativefull-prefix4reviewedpatchesmergedfreshMAIN; composedstrictCUDA/qualifiersClippyPASS. GPUreleasefrozen0224d3da; B2physical.8Bthen2B nowrunning; servingreleasebuildstillactive. Native1teacher/pass B1regressionstillmeasure.
- [x] 27BsameHostMath3-node source-window resolver15fileb7a77a74 complete root+twoindependent source reviews NOBLOCKER; NOT MAIN/compiled/qualified yet.
- [ ] FinishactualnativeprefixB2/4/8 bothmodels/all42/futurefullV,pressure/isolatedunknowncompletion,worker/HTTP,counters,fullgate+approvedreleaseparity; freezeactualserverbeforeotherCargobuildsoverwrite.
- [ ] Fresh6K/8KRustbothmodelsongoing; native8Kcapturesand42domainexportsDONE, noRust8Kcomparison yet. AVpublicdefault/control+transparentexportedrouteproofPASS;2Bcausalhelper sourceprepactive.
- [ ] GenuineeagerV1CUDAoracle8files6cd494f3 sourceprepared; root+baseline reviewbeforeexecution. ActualcommonHF math/full42/batch/defaultgraph/benchmark andfull27Binferenceremainopen.

## Actual continuation20Oct21:22UTC

- [x] PrefixGPU6normalB2/4/8bothmodels all42eachteacherCPU+B1/futurefull248320 exact,2realpressurecases+2freshunknowncompletioncasesPASS. Unknownoriginalhandles1817/1712 retainedwithallpriorrootsunchanged/fence/provider4GiB/no recovery. Finalnormalreservation0.
- [x] PrefixreleaseCLIbuild11m13s frozenserverb1846c3e/worker59517ebf/HTTPf46f5b27; targetserverexactkeptforHTTP. 7sourcefreebandtestsPASS. BothB2worker casesPASS actualraggedteacherEOS/all42/Readycancel/survivoronce/parkeddecodefairness/nooutputfromprefix. RemainingB4/B8worker+allHTTP pending.
- [x] FreshprefixcomposedstrictCUDA/HF/cache,qualifier,VulkanClippyPASS; CPUfreebinding8/GPUcoldrefusal2PASS; approvedcache-direct22.20s/HF-import29.94s bothmodelsfullV/all42/source/leasesPASS. FullisolatedgatependingafterHTTPtoavoidtargetserverreplacement.
- [x] ActualHF2BCPUreference root-probe-1 archivePASS butlauncher successfulloopshadowednonceincleanup. Firstreceiptlifecycleinvalid; originalnonceabsenceindependentlyrecheckedroot-posthoc-owned-absence.json. Narrow2linefix c948254d/followup1492aba1 thenfreshroot-probe-2 PASS48.72s/all4fullV/168domains/60actualstages/cleanuporiginalnonce3checksprocessreaped/noerrors. ProfilesBF16loaderand36sourceF32restoredremainseparate;47rawtensorsdiffer4.785M/4.853Mwordspersnapshot. NoRustHFacceptance.
- [x] 2BRust6K selfcapturecomplete1823.91s/all730/all42/fullVobservationtransparent;8Kcontinuationactivefrom6K. .8BRust6Kstillactive. Native8Kbothalreadycaptured/exported;strictcrossengine8Kpending.
- [x] PublicAVroute proofacceptedbothindependentreviews. 2Bcausal strictCPPcompiled2a184a7e butactualfirstattemptSIGSEGV-11 after5.30s. OwnedchildGDB rootstackNULLgetter; pinnedrepack.cpp5168–5180 deliberatelyget_tensorNULL CPU_REPACK, ggml_backend.cpp502 blindlycallsgetter. Preserveallrawattempts; sourceverifiedopaquehostbuffertransportfixpending/no numericalclaim. No backend/math replacement.
- [x] GenuineVLLMoracle helperv3actual2sourcefreegroup/logcasesPASS; firstattemptprePopenbareimageIdsyntaxrefusal, secondactualpreengineexactmodulevsdistributionversionrefusal(cleanupPASS), bothpreserved. Additivev4/v5 sourceidentityrepairsverified. Actualroot-probe-3 plain V1/GPU/stockroutes complete2teachers/fullV+all320loadedsourceaudits+ownedcleanupPASS; observedfreshenginecurrentlystarting. Full42/defaultgraphs/wider/mixed/ZGmath/performanceunqualified.
- [ ] New27Bactualreplayharness00746966 preparedsourceonly; resolverb7stillNOTMAIN. Afterprefixphaseacceptance merge/qualify40960trainedwords+realstage/lease/cancel/pressure/final0.
- [ ] Twoagents now concrete originalactor/pinnedstream providerboundary and CUDAgraphB2qualificationprototype sourceprep, disjointvendor/servicevsgraph/completionownership; rootsolebuild/runtime/merge. No newprofile/servinggraphpromotion.

## Actual continuation21:39UTC

- [x] GenuinepinnedVLLM eagerB1 subset root-probe-3 actualPASS: plain+observed sequentialoriginalV1engines/full496640rawBF16logitwords exact/all320loadedparameters beforeafter unchanged; actualmodule0.31.0/distribution0.31.0+cu129/gitgdb9527a46 verified. Allownedcontainer/PGID/pipe cleanupPASS. Noall42/wider/defaultgraph/ZGHFmodel/matchedperformanceeligibility.
- [x] 2Bunchangedfirst64capture afternarrowCPU_REPACKtransportrepair PASS11.8965s/RSS3155188KiB; actualcomparator31.5387s/5570fieldrowdifferences. OriginalfailedNULLgetter/GDB preserved; firstactualgap F32alpha projection0/position0,13of16words, withall131072inputwords/full131072sourceweightbytes exact; all960earliercomparedrows exact. No productionmathchange/causeclosure.
- [x] Prefixfinalstrict composedCUDA+HF/cache+qualification/Vulkan,8CPUfree+2coldrefusals, approvedcache-direct/HF-importbothmodelsfullV/all42/source/reclaimPASS. PrefixB2workers2PASS(all3subscopes); remaining12worker+12HTTPphysicalpipelineongoing. FullisolatedgateafterHTTPto preservefrozencompiletimeCLIbinary.
- [x] PrefixA/B/B1source-onlydriver whole705lines/test131lines+pins reviewed, all6frozenfiles matchbe0bbee8; actual10sourcefreeaccountingfixturesPASS. NoactualtimingsuntilallCPUlong/Cargo/GPUdiagnosticsquiet.
- [ ] BothRust8Kcontinuationsongoing; native8K730/fullV+all42alreadycaptured butnocrossengine8Kcomparisonyet.

## HTTP fixture failure21:53UTC

- [!] InitialprefixHTTPsuite: bothB2newcasesrefusedBEFOREserverstartbecausefirsttwoactualprompttokenlengthsequal. New.8B/2B B4prefixcasespassand.8BB8pass; original6HTTPregressioncasespass. Thisisfixturequalificationfailure, notmodelruntimePASS. OriginalfrozenHTTP/failurerawlogpreserved; final2BB8pending.
- [x] NarrowROOT-ownedtest-onlyrepair swapssecond/fourthpreviouslyqualifiedprompts, retainingall8originalprompts/nonEOSoracles. Guardnowreportsactualtokenlengths/outputlengths. NeedrebuildHTTPonlywithunchangedb1846server, freshfreezeandrerunall6prefixHTTPcases; originalproduction/source11hashclosurewillbeupdatedexplicitly. No productionmathchange/tolerancewiden.

## HTTP prefix assertion correction22:01UTC

- [!] RepairedraggedB2.8B actualHTTP producesall5CPUboundcorrectSSEIDs/text/usage/frontiers,15nativeStateOnlyB2cohorts/30rows,no failed/refusedGPUwork/finalactivequeue0; finalunrelatedgreedy>0guardrefusesbecauseunequallengths+2/3outputlimitsfinishshortrowbeforelongtail. Current6rerunstillfinishing; allwider.8Bpass. Preserveallraw/frozenv2receipts.
- [x] IndependentBatchsourcereviewagreesremoveONLYgreedy>0fromPREFIXgate, explicitB1raggedtailcomment. KeeptrueStateOnlycohort/rowdeltas/allCPUwirefrontierchecks/physicalallbandsweightedrows/phaseclosure/sampler4×actualgreedy/noerrors. Original6matchingtail/wideHTTPand14workersprovegreedyseparately. Needfreshstrict/build/frozenv3andfocusedB2bothmodelsPASSbeforefullgate. No productioncode/math/limitschange.

## Fresh8K + reviewed source followups22:04UTC

- [x] BOTHoriginal.8B/2B fresh8192 compare nativeAR1 vsRust730stages/full248320/all42/55382016statewords/modelZEROdiff. .8B capture2103.161s/compare7.681s/state22.059s;2B2345.422s/10.589s/26.976s. Nativebulk/mathcausescope/historycorpus/gpu8K remainopen.
- [x] Initialgraph245/a67 FULLrootandBatchreviewsfoundonlylaunch_bodyvisibilityblocker; separatee7/ce9one-linepub(crate)repairreviewedbyroot+Batch+Spine. All10frozenfiles/6artifacthashesmatch, source-only/no compile/runtime/MAIN.
- [x] ActorBF16transport8da/895 fullrootreview9hashes+990linedelta NOBLOCKER, synthetic/no modelmathpromotion; completionownercompositionmustusegraphsource.
- [x] Original27Bactualharness007fullroot+Batchreviews; cleanupfailuregapfixedADD4299/226v2;4actualrootownedchildfailurefixturesPASS1.626s, independentBatch+SpinesourcereviewsNOBLOCKER. Original007frozenunchanged/unexecuted.
- [ ] NEXTgraphwide5c125/fd4source-onlyfrozen, independentBatchreviewrunning; initialB2physicalprereq. GenuineVLLMpublicGEMMobserver18filesb08preparedrootreviewpending, source-only.

## Final HTTP environment retry22:10UTC

- [!] FinalV3HTTP firstattempt exitsbeforeReady/CUDAoperation: launcheromittedCUDA_PATH, serverexplicitcuda_toolkit_missing503. Failurepreservedprefix-final-http-b2-08b.log; no numericalclaim/fullgate. Rootrerunsamefrozen8d395/b1846withapprovedCUDA_PATH/NVRTCpath, separaterawlogs.
- [x] GenuineGEMMsource-rootreviewall18filescomplete;3sourcefreepointer/span/queryrefusalfixturesPASS. Live/frozenhashverificationneedsactualfrozendirname; no runtime. IndependentSpinefullreviewongoing.

## Prefix acceptance22:13UTC

- [x] Samefinal8d395/b1846withapprovedCUDAenvbothB2HTTPPASS23.51s/34.77s; each15nativeStateOnlyB2/30rows,5correctoutputs/49modelroots/noerrors/final0. Earlieroriginal6+newwide4 accepted withsameproduction andretainedchecks; all12scopesPASScomposed. No pointlessrepeatofunchangedwidechecks.
- [x] Fullisolatedrequiredgateexit0:1241Rust/0failed/70ignored/248suites+18Python/release/installer/NixPASS. Exactsourceunchangedduringgate. Relevantstrict+bothapprovedreleaseparityalreadyPASS.
- [x] CompiledF32route ROOTfullsource26filesreview/alllive+frozenhashesexact;5actualownedSID/pidfd/subreaperfailurefixturesPASS3.131s, nooperator/GDByet. IndependentBatchreviewpending.
- [x] GenuineGEMM18livefrozenhashesexact;fiveacceptedv5filesbyteidentical, onlypublicengineextensionstringchanged. SpineindependentreviewfindsinheritedcleanupfailuregapandLtfailedattributehostscalarreadgap; additivev2sourcepreparationauthorized, original18immutable. No actualGEMMobserverbuild/load/runtimeyet.

- [x] Signedf788929 prefiximplementation/tests/owningservingdocs accepted,13ownedpaths/1645insertions/102deletions; indexempty/READMEblobunchanged. Numerics/modeldocsremainowneduncommitted; completedprefixitemsreconciledinsharedservingplanunderactualcommit.
- [ ] QuietprefixA/B .8BwarmABBA B8+B1 actualstarted22:14UTC, frozenA a896a455/Bb1846c3e/actualreceipt0c119; allCargo/GPUdiagnostic/longCPU workhelduntilmeasurementretirement. SameZGonly, no inventedvLLMslot.

- [x] Reviewedresolverb7a77a74 appliescleanly; MAIN15finalsourcehashesEXACTfrozenandfreshmtimes. NoCargo/runtimewhilequietmeasurementactive. Preparedfocused17artifact/1CPU/5modelwindowroutes+strictCPUscript; nestedtestfilterverifiedsourcebeforeexecution.

## Quiet .8B prefix comparison22:19UTC

- [x] .8B warmABBA B8+B1 actual8samplesPASS,measured36offers/36completed/288outputtokens, alldelivery/sourceIDs/text/usage/frontiers exact/finalcleanupidle. B8mean32.301205→18.33249s(-43.2%); TTFTp95mean29.688367→15.726534s,10sSLOstillmiss. Actualkernels1722640→302160(-82.46%);46StateOnlyB8+8greedyB8. B1mean4.43954→4.42998s, kernels246314both; no significantregression/causalwinclaimed. NewP-1+8 APIcostactualrecordedalongsideoldmax32. NotvLLM.
- [ ] 2BwarmBAAB B8+B1 actualstarted22:19UTC, all heavyCPU/GPU/model/compileholdcontinues; onlyafterall8samplesPASS/cleanupwillserializedCPUwindowfocused/strictscriptstart(session15686). NoCPUoperator/GDB/GEMMphysicsduringtiming.
- [x] GEMMadditivev2rootfullrepairdiff/newtests/11live+frozenhashesreviewed; independentBatchNOBLOCKER. Original18unchanged. Newv4helper53ee4270 andLtgettersvalidityguard9c83eb61; root13fixtures+actualbuild/plain+observedpendingquietwindow. No source/profile/mathpromotion.

## Actual CPU27B build and diagnostic failures22:25UTC

- [x] Resolveractualfocused17artifact/1CPU/5modelPASS. FirststrictClippyrefused4drop_non_dropHostMathtestlines; rootremovedonlyuselessdropsofborrow-only/noDropvalues, repeatedaffected5testsandstrictCPUalltargetsPASS3.59s. Production15pathbytesotherwiseexact.
- [x] Freshstandalonehelperofflinegeneratedlockall33externalpackagesEXACTMAINsubset;releasebuild19.11sPASS, originalsource/profileguardsclosed. Actualsource-window40960-word27Breplaynowrunningunderreviewedv2launcher512MiBAS/4CPU/900swall.
- [!] 2Bnaturalfirstnewprovider8offers/8complete/64tokens/noerrors/cleanupidle butstrictB8guardrefusesrealB2/B4/B8+B1mix. Unqualifiedreport18c5 preserved; noautomaticrerun/B8claim. Newc8actualband-accountingSOURCEONLYprepBaseline; sourceoutputsnotyetclosedbyoriginalvalidatorafterthatfailedguard.
- [x] GEMMv2ROOT13fixturesPASS4.117s, independentBatchNOBLOCKER. ActualfirstbuildfailsstrictheadersCUBLASAPIincludecontractbeforeanyGPU; allownedcleanupPASS, originalattemptpreserved; additiveSpinefixprep.
- [!] GDBfirstavactualfailedGNU17scheduler-lockingguardbeforeacceptedcompute;cleanupPASS/allgroupsabsent/EOF/0.807s. ROOTadditive2lineoffatfirstrealstop+launchernewsourcehashbeforeafterprepared, original26immutable; independentSpinenarrowreviewpending.

## Actual trained27B + compiled reference-route PASS22:27UTC

- [x] FreshMAINlinked27B3opb9cd7f6f underreviewedv2ownedlauncheractualPASS92.7447s/RSS18736KiB. ALL40960words/2tokens11+353/embedding5120/norm5120/QKV10240 ZEROdiff; all18/55.563GB/851roles realFDhashes(source scan90.785s). 80actual128row/full5120column transportwindows maxcharge8388656B/norm4194328B, actualstagemidcancel/pre-cancel/realpressure/refusals/borrowed-outputsource-retention/final0/unfenced. NoStateRoot/frontier/logits/full27B/physicalfreeRAM/performancequalification.
- [x] ExplicitGDBoffnarrowfollowuproot2line/fullbindingreviewedSpineNOBLOCKER. BOTHactualoperatorsrun/validatePASS: av2.1398s,alpha1.6669s,392stops/132firstentrieseach, all130outputs/operands/exportedtrace/maps/fullfilehashes exact/cleanup[]. Rootadditivescriptreceiptbindingcheckedroot-scheduler-validation-v2.json. NativebulkactualtinyBLASvsother129vecdotfamilies physicallyobserved; history/fullcause/profilestillopen.
- [x] GEMMv3root+Batchfullnarrowreview/frozen18+11+7hashesexact;strictactualcontainerbuildPASSwithpublicumbrellaheader/diagnosticwriteresultonly. GenuineplainGPUenginecurrentlyloading; observed+completevalidatorpending. v2badheaderattemptpreserved/allownedcleanupPASS.
- [ ] Resolvercompleteisolatedgate+composedstrictCUDA/Vulkan/coldrefusal/bothapprovedreleases nowserializedsession64453; nootherCargo. Afterfinish rebuildold.8BMAINlinkedtypedhelperfresh foractualregression beforeCPUcommit.

## Source-window phase accepted22:35UTC

- [x] Fullisolatedgate1243Rust/0failed/70ignored/248suites+18Python/release/installer/NixPASS. ComposedstrictCUDA/HF/cache20.72s/qualifier9.34s/Vulkan,8CPUfreebindings/2GPUcoldrefusalsPASS; approvedoriginalcache-direct17.22s/HF-import22.74s bothmodelsfull248320/all42/source/reclaimPASS.
- [x] FreshMAINlinkedold.8Btypedhelperreleasebuild2.03s +actualoriginalcheckpoint16,384trainedembedding/norm/QKVwords ZEROdiff/finalsource0PASS. Exact27B40960-word3op/all851realFD/lease/cancel/pressure acceptedseparately; full27B112-domaininferenceclosed.
- [x] NaturalscopeC8root407linediff/newtests/docs/sourcecountercontract/all13frozenlivefiles+2artifacts+3actualacceptedHEADblobs/failedreportdigestsreviewed; ROOT13actualsourcefreefixturesPASS0.036s. Batchindepreviewpending. NoC8measurementyet, nooriginalfailedreportpromotion.
- [x] RoPErootfull2newmodules/qualifier/call+wiring/delta/all6fullsource+9artifacthashes reviewed; Spinefindsonlyuncheckedoffsetblocker. Baselineadditivechecked-range4c0b/44c134preparedrootreviewNOBLOCKER, Spinenarrowreviewpending, noMAIN/build/runtime.
- [!] GenuineGEMMv3actualstrictbuild+plainoriginalenginePASSnormalcleanup; observedfailsbeforegenerationgenericexact-providerresolutionrefusal. Originalactualdoesnotidentifysymbol/proveRTLDcause. Spineadditivev4explicitEXACTalreadyloadedDSO NOLOAD/publicversionedbinder+boundeddiagnosticprep, no alternateprovider/mathfallback/SDKchange.

- [x] SignedCPUphasebfe62df68d763b9d5c6b87c891be6a62b3593de9 accepted18ownedpaths/1745insertions/148deletions; indexempty/READMEblobunchanged/cleanMAINbeforedocreconcile. Modelandnumericscompleteditemsnowreconciledunderactualbfe62df, noID-onlycommit. Full3model/logits/StateRoot/vLLMperformancegatesremainopen.

- [ ] Source-reviewed/root13fixtures/BatchindepNOBLOCKER newnaturalC8workloadactual2BwarmedBAAB C8+B1started22:38UTC, samefrozenA a896/Bb1846/receipt0c119. OriginalB8failedreport5da6unmodified/unqualified. AllheavyCPU/GPU/Cargo/diagnosticworkhelduntilall8samples/ownedcleanupretire; agentsstaticonly.

### 2026-10-19 22:45 UTC continuation
- Natural C8 2B warm BAAB: PASS8samples/36offers/36complete/288usage tokens; reportf6869712. Actual C8 new54B8/432rows/294384kernels vs old8B8+368B1steps/1668496kernels. Original strict-B8 failed5da6dc90 preserved. No vLLM parity.
- Merged source-reviewed PIN4d +initialgraph245 excluding old fused +visibilitye7 +wiringbaf/typefbe +currentprefixcomposition15cc; Spine independent no blocker, parentfused881 exact. No NEXT/RoPE/provider imports.
- Starting focused strict CUDA +nonCUDA qualifier checks/release GPU+qualifier build, physical graph not yet qualified. Serving graph remains private/qualification-only; production global exclusion/driver metadata admission open.

### 2026-10-19 22:49 UTC graph build and genuine provider continuation
- Initialgraph actual strict attempt1 refused bad DeviceHandle module; attempt2 refused dead unused init-owner setter/manual_contains/unit AppendKv pattern. Preserved original logs; corrected exactly4 locations, independent Baseline review no blocker. No math/order/limits change. CUDA strict now PASS12.59s; Vulkan qualifier strict PASS1m53s; release GPU/qualifier build ongoing.
- GEMMv4 source freezes original18+v2_11+v3_7+v4_9 hash exact; original full forwarding suffix byte exact v3, root+Baseline source review no blocker. Actual root-probe-v4-1 strict observer build PASS; fresh genuine plain/observed run active44058. No graph GPU qualification concurrent with that owned GPU lifetime. No retry/reclassification of v3failure.
- Attention source-only327line module/499line tests/9970Bhook and8proposed+6base+4artifact hashes fully root reviewed; MAIN exact6bases. No math/ownership blocker. Three HostMath non-Drop explicit drops to remove in additive testfix beforecompile. Actual2B paired capture preparation assigned Baseline; nativeuse_ref bulk controlled client assigned Batch. No attention merge/runtime yet.
- Capture exclusivity audit proves currentqueuealoneinsufficient (init/alloc/drop/actor/event transport escapes). Spine prepares source-only pure fair-owned-token state machine/default-closed participation proof plus exactintegrationmap; no productiongraphenable.

### 2026-10-19 22:54 UTC genuine public GEMM actual result
- root-probe-v4-1 ACTUAL PASS_EAGER_B1_PUBLIC_GEMM_API_ONLY, gemm-validation55d68c7a / validation2fa80108 / trace d56e802a / observedreport e060e753. All496640 words/320 loaded roles beforeafter unchanged; 4 real original QKVZ/BA cublasGemmEx calls. BF16type14/FP32compute68/algo99/transposeA1B0/math0/hostpointer0/atomics0/alpha1beta0; explicitworkspace8519680B, allfullspans actualcuMemquerychecked. Publicenummeaningverifiedexactpinnedheader.
- Strictbuild2.0499s/plain154.4348s/observed158.9084s are diagnostic elapsed times, NEVERbenchmark. Every3mode actualnoncecontainerabsent/groupgone/leaderreap/pipeEOF and cleanup_errors[]/stage_errors[]. Deviceback41MiB/noactivegenuinecompute. Newbinderpasses actualscope, earlier genericfailurecauseunknown; allpriorfailuresremain.
- Pinned defaultvLLM loader andquantization registries containnoGGUF entry; no attempt to compare originalquantZG withBF16vLLM as equivalent. Hiddenkernel, all42, wider/defaultgraphs, ZGfullHF/GPUnumerics, vendoropaqueallocadmission and genuine matchedperformance remainopen.

### 2026-10-19 23:01 UTC initial graph physical complete / bulk control first failure
- Initialgraph actualfrozenGPU1994404d / qualifier1d184cfb /sourcefreeze84441201. Sourcefree69PASS/25ignored/0fail, including3newkey/frontier/generationguards.
- Both normalinitialB2 qualifiers PASS CPU+B1/native full248320/all42,4stepsindependentfrontiers+swappedrows, invalid/stale/actualpressure/abandonedReady and final0. .8B warm4787/body4520/outside267/replay6/committed19148;2B4643/body4376/outside267/replay6/18572. These are source launchsites, notdriverenumeratednodes orspeed.
- Both freshunknowncompletion qualifiers PASS actualgraphenqueue1/unchangedroots2/executableoriginalownerretained/handles1824(.8B),1719(2B)/4GiBenvelope/fence/noDestroy/noRecovery/nopostabortreadback. Finalprocessreclamationdevice41MiB. InitialscopephysicalprerequisiteforNEXT closed, fullphasechecknotyetaccepted.
- Use_ref original14sourceFULLreview/hash exact;8parserPASS0.003s+5actualsupervisorPASS3.174s;strictC++buildPASS2.437s, binary121400B6ae8364e. Firstactual.8B8192ref64 FAILEDpreacceptedcompute/exit125 lexical exactloadedproviderpath guard duringtensorsload. Ownedleaderreap/groupsgone/EOF/cleanup[] PASS1.07086s, invocation/rawnative10147B retained. Noaccepted8Kcontrolcomparisonfromthisfailure. Batchprepares additivecanonicalexactfileidentity+boundednameddiagnostics, original14unchanged.
- RootFULLread NEXTcurrent1292line6pathpatch/wiring+README/all22files/all13MAINparenthashes; no sourceblocker. Spineindependentreviewpending, rootmergeonlyafter. RoPEoriginal+checkedrange alreadyindependentreviewed; willcomposeafterNEXT withfreshmtimes andphysicalgates beforeperformanceclaims.

### 2026-10-19 23:08 UTC wider/RoPE and original attention continuation
- Root+Spine FULLreviewNEXTcurrent1292line6pathse8ca/wiring4dad/all22proposal/all13MAINparent exact. Initialphysicalprerequisiteclosed; mergedNEXTthenRoPE9c8de+a604+checkedrange4c0b, originalfrozenartifactsunchanged. Actual17sourcefreeze2f7311e6. CUDAstrict11.86s/Vulkan8.66sPASS; releasequalifier3m31sPASS;frozenprobe36836400B40205ccd. GPUlibtestreleasebuildstillactive26820. Firstphysical170angles/trainedsource/CPU+B1/nativefullV/all42+greedyB2 onbothmodels active47852. No widergraphphysicalyet; productionbarrier/metadataadmissionUNQUALIFIED.
- HF2Bpaired actualroot-pair-1 PASS validatione45e1903. Freshplain44.5615s/observed42.0102s(notbench), eachall196raw/fourfull248320/all168canonicaldomains/60priorintermediates exacttoacceptedoldroot-probe-2;87728128raw+intermediatebytes/mode,216newBF16attentiontensors688128B. Ownedv4TERM+twoKILL-before-reap/groupsgone/containerabsent/EOF/cleanup[] PASS. maxRSS9719960/9691160KiB within12GiBdeclaredbound, notphysicalRAMadmission. TenfixturesPASS0.018s; initialrootunittestwrongcwdimportfailurepreserved. RootFULLreadnewrunner301linevalidator136tests/helpers+entire288lineintegrationdiff(re-readchunksafterinitialtruncatedoutput); all19frozenfiles+reusedparents match, SpineindepNOBLOCKER. RustattentionprivatefragmentstillNOTMAIN/compiled/run/fullHF.
- Use_refV2 strictbuild2.723sPASS125640B42b98af8 fromCPP8a5f1cab/freeze c5913da1. Freshactualref64-2 FAILEDexit125 nonCPUbackendduringcontextinitialization beforeanyacceptedcompute/trace. Newsource+build+actualinvocation bindingPASS(notmath); actual1.5193s/cleanup[]/groupgone/reap/EOF. No aliasfailurecauseclosureinferred. Baseline source-audits actualnonCPU publicidentity/accelerators beforeanyrepair; allfirst+secondfailedruns preserved.
- Batch found actualvLLMGEMMv4 rawarchivealreadyhas QKVZ/BA realinputs+outputs; prepares source-only compact stdlibexporter/validator overimmutableactualarchive, authenticpointer/callscope/packedrowrole binding, no newmodel/D2H. Spine puregatev2 drainfix independentlyacceptedsource-only; actualvendor/common/event/drop participation stillclosed/unimplemented andprotocoltests notrun.

### 2026-10-19 23:13 UTC physical-helper failure and correction
- New17sourcefullGPUunittestrelease3m14sPASS, frozen35885704B7c977e60;75testsPASS/25ignored/0fail0.07s.
- FirstcombinedRoPE08B qualifier:170physicalangles/28296words exactsignedbits/canaries/originalReady/final0 PASS, old4448+independentscalar6592 launchsites versus170candidate. Then composedlegacy trained rope_stress refusedconflictingphysicaldeviceattestation: anglecap5GiB vsoldoperatorcap1GiB (OPERATOR_CAP), so no fullmodel/2B claim fromthisattempt. Failedlogpreserved.
- Rootcorrectedcomposition2files: defaultlegacy rope_stress→newwith_device_cap(OPERATOR_CAP1GiB), oldbodyexactexceptargument; standaloneangleskeeps5GiBwrapper; qualify_modelpassesincomingcaptoangles+trainedstress+existingfused/greedy. Spine independentdiff/bodyreviewNOBLOCKER; nogatesremoved/mathchanged/ledgerbypass. New18sourcefreeze graph-wide-rope-cap-fixed-source.json; strictCUDA6.01sPASS, build18859ongoing. Earlier17sourcefrozenartifactsremainimmutable; helperfixedfreshphysicalgatespending.
- Use_refV2ref64-2 actualnonCPUguardrefusal/sourcebuildbinding/cleanupPASS; noacceptedplans. Baselinepinnedsourceaudit suggestsACCELinitializedbeforeCPU(BLASloadedinactual.log), is_cpu onlyCPUguid, notCPU_REPACK. V3 publicidentitydiagnosticprepared(no skip/disable/wrongsetter)beforeanypolicychange. Publicasync/plancompute canobservefuturewholebackendroutes, BLASbypassesinnerHaswellplan; op_offloadfalsealonecannotproveBLASunused. Causeactualbackendnamestillunobserved.

### 2026-10-19 23:19 UTC continuation
- Corrected18sourceGPUstrictCUDA6.01s/Vulkan+releasequalifierPASS; frozenprobe36836400B78ab3823/sourcefreezec54c0d1b. BOTHmodels170angles/28296words+trained32768stress+nativeB2CPU/B1all248320/all42+greedy6stepsPASS. Legacycomposedcapmismatchfailurepreserved, no math/gate bypass. B4/B8normal+greedy then6widergraphnormal+18freshunknown cases running serial22052. Productioncapturebarrier/metadataadmission/performance remainUNQUALIFIED.
- V3publicidentitydiagnostic rootbuild3 strictPASS2.620s127816Bab61418b. Actualref64-3 confirms backend/device/registryBLAS/device_type3/is_cpufalse/returneddevice==input. Sameexit125BEFOREsetter/acceptedcompute; no records/numericresult. Source/build/invocationbindingPASS. CLIinitialmissingcpus/thenintegercpus errors occurred beforeoutput/model, logs preserved where available; finalexactCPUlist0..7. NoBLASdisable/unknownskip/mathchange. Baselinepreparesguardedpublicasync/planCPU-onlycontrol, rejectsanyactualACCELcompute; freshAR1all8Kproofstillnotbulk/historicalclosure.

### 2026-10-19 23:27 UTC implementation and qualification
- Rootmergedreviewed8CPUattentionfragmentpathswithfreshmtimes afterall6parenthashesexact. InitialstrictcompilecaughtprivateSourceTensor::packed_bytes call; removedbecauseexactrole/BF16/geometry<=5120x16384/sourceguards subsumeitschecks/overflow; typedProjectstillvalidatesactualsource. Removed2nonDropHostMath testdrops. BatchindependentnarrowNOBLOCKER. Correctedstrictclippy1.77sPASS, default+CPUonlychecksPASS, focused16tests3.36sPASS; fullmodel43PASS/1ignored4.01s. AuthentictrainedBF16replaystillpending(Batchpreparesbothmodels), fullHFstillrefused.
- Purehostcaptureprotocol testedonlyfreshdetachedownedp12-capture-host-tests-20261020, MAINunchanged: actual14testsPASS0.00s/compile14.81s.3actualnegativesexactE0599/E0499/E0499PASS withsource restoredfinally. Frozenhosttest305070096Ba7a76529/resultcapture-host-protocol-result.json. FirstrelativeCWDcopyfailed andzero-casetestlogpreserved/excluded; correctedrunhas14nonvacuouscases. NoCUDAparticipation/terminal/retirement/completenessclaim.
- AuthenticvLLMexistingoperandexporterrootFULLreview658Py+README/contracts/provenance/originalunchangedvalidators, BaselineFULLpeerNOBLOCKER/all9new+49oldsource+20metadataexact.10actualsourcefreetestsPASS0.022s; fresh36,992Bexport/independentactualbindingqueuedafterGPUquiet. Nativevendoropaqueallocationandhiddenreductionunqualified; no speculativeunsupportedpublicAPIadded.
- CorrectedRoPE BOTHmodelsnativefullB2/4/8+greedy allfull248320/all42physicalPASS. Widegraph0.8BB2/4/8+2BB2/4PASS;2BB8+18freshunknowncasesrunningserial22052. Productioncaptureglobalbarrier/metadataadmissionUNQUALIFIED. Fullcomposedstrict/check/originalreleaseparity/finalworker+HTTPreleasebuildrunningunderoneserializedROOTCargo coordinator.

### 2026-10-19 23:31 UTC followups
- FirstcomposedstrictCUDA20.89sPASS then qualifier-onlyfeaturestrictFAILED deadcodeAttentionRoles.kv_heads (onlywindowfeatureusedit). Rootaddedexistingexpect_elements checksfor3actualprojectedshapes, boundedroleheadgeometry, noadditionalPlanOperation/math/allocation. BatchindependentnarrowNOBLOCKER. Oldfailurepreserved; new26sourcefreezed89cd668/currentpipeline75448. NewstrictCUDA10.94s/qualifier11.53s/Vulkan10.91sPASS;8CPUfreebindings+2GPUcoldrefusalsPASS; releaseoriginalroutes/fullisolated/servingbuildcontinue.
- ALL6widergraphnormalcasesbothmodelsB2/4/8PASS actualStateOnly2teachersinclEOS/futurefullV/outsidegreedy/CPU+B1+native/all42/pressure/Ready/stalereclamation. 18freshunknowncases14PASS/15started asof23:30:41, originalenvelope/quarantine/noafterabortread/destroy/recovery. Full26sourceCPUadditiondoesnotchangefrozen18GPUqualifier; composedfinalserving gatespending.
- FirsttransportproposalROOTFULLread9pathpatch/admission265/tests581/README/DESIGN/CALLERS andall30frozen+11selectedcachedhashesexact. RootcaughtCallError.into_panicfabricatedpayloadonrefusal;Spineadditiveclassificationfollowup763357e5/8e13ec56restoresoriginalpayloadandexplicitresume message+fixture. RootFULLreviewNOotherblockerunderUnproven/no-closeboundary. Copied119actualcachedcommon/runtimefiles+9proposal+2classificationfollowupONLYintoownedisolatedp12-capture-host-tests; MAIN/Cargo.lock/cacheunchanged. IsolatedCargo/14host/no_std/2negativespendingoneserializedCargoqueue. SpinepreparesoriginalGraphHandleprepaidnonblockingretirement.

### 2026-10-19 23:33 UTC completed actual physical batches
- All32freshphysicalcasesPASS (8nativeB4/B8normal+greedy,6widergraphB2/B4/B8bothmodels,18unknownallbands/allreadouts). Resultgraph-rope-32-physical-result.json SHA87b85effbfc2733ebed0856ab2eea3455dcaace1cacdcbd23f2a35cf2833ca70. FinalRTXUUIDunchanged/41MiB/82C/0util. No driverrecovery/deviceprivilegechange. GPUbinaries18source78ab/sourcec54c; composed26sourcesstrict/fullgate stillrunning, productiongraphglobalbarrier/metadatabudget/performanceunqualified. Originalreleasecache-direct19.03s/HFimport24.72s bothmodelsfullV/all42PASS.
- Existingcachedcommon/runtime .cratearchivesbothmatchcurrentCargo.lockchecksum: common002a1629...90171B;runtimea1c2bdd3...192733B. All41+76archivefilesbyteexacttoselectedcache; onlyextra2.cargo-okmarkers. No download. Rootproofcommand-hook-locked-crate-authenticity.json SHA028181fbf39f0d97d70ac0c0ff4515777455740eef80c52f7dd40346a90beb4d; improvesbaselineauthenticitywithoutmutatingoldfrozenproposal.
- AuthenticvLLMv4compactoperandexportactualPASS36,992B/6files andindependentfreshrederivationPASS. All4actualGEMM/API/input/output/packedlearnedrolehashes andexistingfull496640logit/320sourceaudit/provider/cleanupguards reexecutedunchanged; no newmodel/GPU/D2H. NativeBF16math/provideropaqueresourceadmission/fullHFremainUNQUALIFIED.

### 2026-10-19 23:54 UTC composed gate and reference followups
- Composed26source d89cd668 strict CUDA10.94s/qualifier11.53s/Vulkan10.91s PASS; release cache-direct19.03s/HF-import24.72s bothmodels fullV/all42 PASS. Full isolated check PASS1243Rust/0fail/70ignored/248suites +18Python/release/installer/Nix. Fresh release server/worker/HTTP built; server4a745527 / HTTP9df92fb0 / worker11ad10f4. ActualHTTP ALL12 PASS568.90s includingbothmodelsB2/B4/B8prefix, ragged/unequal arrivals/cancel/JSON+SSE/CPU-ID-text-usage-replay; originalreadoutownership/sourcecounters preserved. Fresh14workerphysical pending.
- Attention helper original prior_checks metadata incorrectlycompared49recordLISTwithscalar49; failedroot-inputs-1 preserved. AdditiveV2 freeze b458f320 / parentpatch1f5452e6; onlycontracts/tests/docs+authenticmetadatafixture differ, other8files exact. Root+Baseline narrowreview NOBLOCKER,10fixturesPASS. Fresh authentic extraction106496B/32files PASS input7adf3bbe; actual releasehelper8.88s/strictClippy9.41s PASS,106MAINpins unchanged,33normalizedexternalpackages exactMAINsubset. Frozen1517960B05eb0512. Actualbothmodels47,104word/privatefragment+sourcelease/cancel/pressure/lifetime replay active67388, nofullHFclaim.
- GenuineGEMM compactexport actualPASS36992B/6files, independent freshrederivationPASS f0981e62, old496640logit/320sourceaudits/providers/cleanup reexecutedUNCHANGED. Originalrawarchivebeforeafter exact. NativeCPUprofileBF16 projection andGDNconv+SiLU proposals assigneddisjointagents; noopaqueprovider/fullHFpromotion.
- Full-runtimeV4 controlactualFAILED125 beforecompute on legalfree(NULL); V5 boundedpublicdiagnostic ACTUALLYconfirms NULL/untracked/map0/compute0/nooriginalfreeforwarded. Bothfailedattempts retained, source/build/invocationbindings andownedcleanupPASS. V6 ONLY2nonNULLguardchanges+revision vsV4, ROOT+Batchreview NOBLOCKER, strictbuild3.184s160960B26178e02 PASS. ActualV6numericalrunpending afterattention; nohistorical/bulkclosure.
- Optionalcommon/runtimeisolatedhostattempt1failedpackage-not-member; explicitisolatedmembers thenofflinecriterionmissing; removeonlybenchcriterion thenofflineuncachedtest-log. All3logs preserved; noRustruntimecases, noMAIN/cachemutation/dependencyfetch. Originalcommonmanifest preserved andproductionRustsourceunchanged. Spinepreparesadditive exactfixture-onlycached-dependencyharness.

### 2026-10-20 00:13 UTC CPU and numerical results
- AttentionV2actualroot-replay-2 FAILEDbeforemath becauseharnessexpectedone4MiBbindinginsteadactualartifact4MiB+model4MiB. PreservedV2source/binary/failedreport/cleanup. AdditiveV3only4→8MiB/exact4reservations/Scratch72+Other2/Output&Staging0+steadyledgerreport; ROOT+BatchpeerNO BLOCKER. V3freeze d800a343, release39.32s/strict9.80s PASS, binary1517960Bb59a7614. Bothmodelsactual47,104BF16words all7outputs/teachers11+353 ZEROdiff; actualpressure/pre+midcancel/foreignsource/provider/lifetime/final0 PASS; independentvalidationPASS. Source/binary/input/helperFDpins andownedcleanup[]/groupsgone/EOF pass. Numerical-onlyCPU0..3 wasconcurrentwithnonbenchmarkworkerGPUCPU8..11; no elapsed/performanceclaim. SignedownedMAINcommit fe2a342 G, rootREADMEa7be69... unchanged, plancompletedmovedunderactualIDinpendingdocs; fullHF/GDN/27Bopen.
- Use_refV6actual0.8B8192ref64 capturePASS630publicCPU/630actualinneruse_reftrue/2CPU+2BLAScontexts/zeroBLASmath; originalwhole-state+next-logitobservationguardsunchanged. Compareall730/fullV and all42/55,382,016activecausalwords ZEROdiff vsacceptedfreshRust8K; comparelogs7.73s/export23.16s/statecompare22.87s(notbench). Fresh2BV6correctlyFAILED125actualBLAScompute beforedelegation, noacceptedcapture; source/build/provider/invocation/ownedcleanup[]bindingsPASS. BaselineauditsactualpublicschedulingandpreparesV7boundedpublicnodeidentitydiagnostic; noalpha-causeassumption, noBLASdisable/guardbypass/math/tolerancechange. FirstcomparisonCLIusedrelativepython3 andrefusedbeforechild; correctedabsoluteexecutable, nofailednumericalresultrewritten. Required2model/historicalgateOPEN.
- Optionalcommandhostharness49freeze9f333c6f ROOTfull3pathharnessdelta+51baseline/overlayhash/exactfixturesreview; all122normalizedexternalpackagesexactMAINsubset. Actual14PASScompile40.27s/0.00s. Firststrictcaught4ownedstylewarnings; additive2pathrootpatchbackticks/tuplealias/equivalentletchain ROOT+SpinereviewNO BLOCKER. Correctedstrict1.91s/actual14retestPASS/no_std23.86s. TwoactualcompilefailuresONLYE0599/E0499PASS, restoredexactsource; frozen22894416B5970ebdd; command-hook-host-result.json. NoMAINvendorregistration/Completeissuer/GPUcompletionproof.
- RetirementsourcefullROOTreview451linemodule+430fixtures+all12pathdelta+DESIGN/CALLERS: noidentifiedblockerunderclosedboundary; sourcefreeze de329f93/patchd4e875f7. Originalrunnerfixed64prepaidcleanup/knownerrretention prepared,11hostfixturesNOTRUN. Actualqueuecutoff/events/storage/bootstrap/finalteardownandpre-begintemporaries stillmissing; noMAINmerge. Spineprebegintokencontinuationdesignpreserved, reprioritizedofficialGGUFbaselineaudit.

### 2026-10-20 00:13 UTC official matched-GGUF baseline route
- OfficialvLLMdocsnowpointtoout-of-treevllm-gguf-plugin, READMEtestedQwen3.5. ROOTactualgitlsremoteHEAD/main e2b8ad532b8b5ea175100202c30430c1d2b5e6a8; boundedsourcearchive135853Bf51ee9cb,151regularfilesacquiredintoownedignoredserving-and-speculation-vllm-gguf-baseline-20261020/source. Source-onlyofficialregistrationroute; earliernoin-treeGGUFclaimstilltruebutdoesnotexcludeplugin.
- ROOT+Spinefullreviewfreshstockbuildscripts/acceptedexactownedhelper53ee4270. BuildV1failedbeforecontainerstartrootlesscpusetcontrollerunavailable, ownedcleanup[]/groupsgone/EOF. V2onlyremoveunsupportedcgroupscpusetandpinactualCPU0..3beforeTorchimport; cpus4/mem8G/MAX_JOBS2/networknone/pullnever/read-only/noGPU unchanged. Existingimagef57d/Torch2.13cu129/vLLM0.31/HFHub1.33/setuptools80.10.2/ninja1.13.2/CUDA12.9NVCC86; noSDKdownload.
- ActualunchangedupstreamwheelbuildPASS V2,1,096,140Baf9de736;129ZIP/RECORDmembersvalidated/111packagedPythonfilesbyteexactupstream151/sourceprepostunchanged; nativeextensionebb896a5. TorchCUDAuninitialized/nophysicalGPUorMODELrun; cleanup[]/leaderreap/groupsabsent/EOF PASS. PuredependencyGGUF0.19.0wheel118475B70bcd10e andwheel0.48.0wheel33320B3217dcc8 acquiredwithactualPyPIhashpins; nohostinstall.
- SourceauditstockQwenadapterrequiresrealmm_projwhenfullHFconfigcontainsvision. Spinefindslegitimatepinnedllamacpp --mmproj --outtypebf16 overauthenticHFvisualweights +stock --language-model-only StageMissingLayer/noGPUvisionparams; preparesROOT-onlyconversion/loadharness. Nofakeconfig/adapterfork/textweightconversion. Actual320textroles/noMM/source/tokenizer/profile/output/cleanup gatespendingbeforebenchmark. PluginuniformQ8_1 vsZGper-codecQ8_0/Q8_1/Q8_K/casts/reductionmaydiffer; sameGGUFweightsdoesNOTproveexecutionbitparity. Genuine matchedperfstillOPEN.

### 2026-10-20 00:34 UTC continuation
- Signed reviewed implementation checkpoints fe2a342 (original CPU attention fragment) and9a19901 (batched graph/RoPE qualification), G signatures; clean MAIN/index before next owned7path BF16qualifier composition. RootREADMEunchanged. All14worker+12HTTP+1243Rust/isolated/originalroutes passed previous phase; productiongraph/vLLM/performance remainopen.
- Publicplacement feasibility audit identifies genuine CPU_Mapped intermediatebuft + immediatepublicpresplit CPU assignments as distinct2B use_ref control candidate. OriginalCPUlast/BLASlive/order/buffers retained, actualBLASmath stillrefused125. Baselineprepares sourceonly; no numericalclosure.
- StockGGUFprojector V1 preflightFAILEDbeforecontainer (15MTPmistakenvisual); originalfreeze/failurepreserved. AdditiveV2 b14b8e90 actual8fixturesPASS. Genuineunchangedconverter+independentrawaudit PASS BOTHsources: .8B153visual→154projector207345856B6f67ce2f;2B297→298671372768B5ddf560e. ExactBF16orF32widening/noLM/sourceunchanged/ownedcleanup[]/EOF; stockGPUloader+math+bench stillpending.
- GDN35-sourceV2 plusroot+Batchreview;2sourcefreefixturesPASS, actualauthenticinputextractionPASS8.08s/64728KiB. Independentpublicconv1d/SiLUreference actualPASS bothmodels/teachers11+353 originalcapturedactivated/K4bytes exact BEFORE newrawtargetemission. Exactimage/helper/35source/inputs beforeafter+ownedcleanupclosed, noGPU/model/tokenloop. Sourcebyteexactreferencecopiedfreshroot-public-reference-1 forfutureRustcompare; nativeimplementationunmerged/unqualified.
- BF16 CUDACPUtarget7pathbase sourcefullyROOTreviewed/all371intendedsources verified/7sourcefreefixturesPASS. Firststrictfailedexistingunusedquarantineownersgetter andqualifierneedlessrange; V2checksactualabandon ownercountincrease/iterator instead ofwarningbypass, strict13.20s and2unitPASS. IndependentBatchreviewrequestscompleteCPUoraclelengthguard; ROOTV3sourcefreeze a30e1ba6 capturesallchanges, freshstrict/unit/build/physicalpending. Noarithmetic/productionenable/fullHFclaim.

## ROOT continuation 2026-10-20 01:06 UTC

- HEAD signed `9a19901`; root README blob unchanged. MAIN owns only additive CPU-target BF16 CUDA qualification files; no production profile enabled.
- ROOTV4 BF16 projection trained normal: both models, 163840 raw F32 words exact CPU and 163840 rounded BF16 words exact original public archives; 30 source-certified launch sites. Actual abandon fresh process per model retains original source/completion owner/fence/pending event and no Ready; independent validators PASS.
- BF16 adversary ROOTV3 strict GPU/helper + five focused units PASS. Actual release 1m15s, binary `4d67068232ff8f21c38ac03c08a18f0760e17baa7e7049ffc1d1f4fbeabad34c`, all372 source pins exact, 521 external lock packages exact MAIN subset. Physical normal328 source sites + eight fresh fault processes pending; GPU reserved exclusively. Peer review of two source-only corrections: no blocker.
- Genuine pinned stock vLLM .8B loaderV3 plain/observed PASS, all320 source roles and 496640 full-logit words zero differences. Native BF16/logit representation explicit. No matched performance, B>1/default graph, crossengine bit parity or all42 claim. Actual 2B loader pending.
- Distinct CPU_Mapped 2B8K plain completed, original730 raw stages/fullV serialized state and actual no-BLAS-math/public use_ref CPU policy checks PASS. Source/build/actual-run bound. Second observed same-control run active; full730/fullV/whole-state transparency and then independent Rust all42 remain pending. Historical divergence remains open until those comparisons.
- Authentic GDN public conv+SiLU reference PASS bothmodels/teachers; 122880 target BF16 words. Prepared ROOTV6 source overlay not merged while adversary source freeze is active.
- Agents source-only: generic U32 row-state transport, source-bound HF coefficient preparation, actual matched vLLM benchmark startup/policy/ownership bundle. No agent Cargo/GPU/runtime.

## ROOT continuation 2026-10-20 01:22 UTC

- Signed `fff1842594f8011b761d08fcc1690311b72543e3 G`:29owned source/test/plan paths,3509ins52del. Complete isolated1251Rust/0fail/70ignored/248suites+18Python/release/installer/Nix PASS. ComposedCUDA/qualifier/Vulkan strict +8CPUfree/2GPUcold-refusal PASS. Approved original release cache-direct18.32s/HF-import24.22s bothmodels full248320/all42/source/leases PASS. RootREADME bloba7be69c unchanged.
- BF16 adversary ROOTV3 actual normal328 source-launchsites/3690raw+3690rounded words PASS, then all8 fresh fp32/bf16overflow ×B1/2/4/8 processes PASS actualstatus/canary/readset/quarantine/noReady. Originalkernel/nativeCPUprofile unchanged. Trained ROOTV4 normal+abandon bothmodels previousPASS remains exact frozen evidence.
- GDN ROOTV6 releasehelper build failed JSONmacro arrayindex syntax; preserved untouched. FreshROOTV7 d5c385dd changes only parentheses, independentnarrowreviewNOBLOCKER. Actualnativebothsources all122880BF16words/publictargets/realpressure/precancel/retained/final0 +independentstdlibvalidator PASS. FullHF/session/checkpoint stillclosed.
- DistinctCPU_Mapped2B8K plain+observed publicpolicy/source/build/ownedcleanup PASS and exactsame-control730/fullV/wholeprior+poststate. Fresh independentRust comparison all730/full248320/all42/55382016statewords ZEROdiff. Freshcontrolledbothmodels8K nowPASS; losthistoricalcorpus/defaultbulk arithmetic stillopen.
- vLLM2B V3 actualrefused335vsassumed320storedtensors; sourceverified exact15MTP extras, originalstockadapter skipsentireblk24. ImmutableadditiveV4 83995212 withall40pins +actual19fixturesPASS. Fresh2BV4 root-loader-2b-2 activeexclusiveGPU; plaincompleteactual320roles/nativefullV, observed+pairpending. No matchedbenchmarkclaim.
- Row-nativeU32state transport e0f44d35 sourceFULLROOT+independentBatchreviewNOBLOCKER. Appliednarrowhunks+newfilesfreshmtime ontofff1842 preservingGDNcoldrefusal/BF16module. Firstsourceprevalidation printed filesdict-aslist KeyErrorbeforeanyMAINedit; correctedmetadataiteration thenall16pins exact/applychecksPASS. CurrentMAIN only6ownedrowtransportpaths; strict/focused/default/Vulkan pipelineactive. Physicalfullmodel/owner/graph/worker/HTTP/performance allpending.
- CoefficientprepfollowupV2 d744cb9d source-onlyfrozen; ROOTreadall5newmodules/corewiring/windowtests +publicinputs/reference/validator/lifecycles, nativehelper523linespendingreview. NoMAINmerge/actualpublicexecutionyet. CachedGDN/privateproviderphysical-envelopeagents continuing independently.

## ROOT continuation 2026-10-20 01:50 UTC

- Genuine stock vLLM2B V4 root-loader-2b-2 plain+observed+independent pair PASS: all335 stored roles inventoried/exact15MTP excluded by unchanged stock adapter/320backbone;496640nativeBF16 logit words zero observer differences. Both models loaders accepted; no matched benchmark or crossengine/full42 claim.
- Coefficient authentic inputs a730ac01 +actual public617/619 reference47efae46 PASS4cases; independent validator original FAILED tuple/list FDmetadata equality before math comparison. ROOTadditive validatorV3 f6fb7303 JSON-normalizes onlymetadata; peer review no blocker; corrected actualpublicvalidation PASS. MAIN/native unmerged/unbuilt; nativevalidator still needs additive importfix before actualnative gate.
- Row383-sourcefreeze d691d213 releasequalifier now built/frozen36987952B 0af306de; all383postbuildexact. Both new fullmodelB2prefix tests PASS fourteachers/EOS/unequalfrontiers/all42CPU+B1/futurefull248320/final0. Copy-onlyBoth previousPASS remains limited to copies. New46case fullmodel/native/greedy/privategraph/allreadoutunknown/B4+B8prefix/pressure/abort/admission serialGPU matrix running; worker/HTTP/fullgate/performance remainpending.
- BenchV2 e5e70923 all51live+frozen pins exact, independentreview no blocker, actual23fixtures PASS. First actualHTTP launch failed BEFOREengine private net loopbackdown; second privateownercheckedloopback fixed network but native-gpu singleUID namespace conflicts existing rootlessPodmanstorage (exit125/noGPU). Both failures retained; second in-namespace containerabsence audit refused, host actualownedname absence confirmed nextprobe. ROOT exactexistingPodmanunshare→freshprivate net+mount namespace wrapper2 passes sourcepeerreview and actuallyspawnscontainer; third attempt fails cgroup2overmountparser BEFORETorch/exec, ownedcleanup PASS. ROOTactualfiniteownednoGPUlayoutprobe confirms memory.max12884901888/swap.max0/cpu.max400000100000 and two cgroup2same-target mounts (hidden ancestor/visible readonlyroot). AdditiveV3 visibleFDmountID guard preparation assigned; no guardrelaxation/timing.
- OriginalRusttokenizer helper directstaleartifact link E0463 retained. FreshstandaloneCargo root-tokenizer-build-v2 uses actualMAIN83external locksubset, release1m24s PASS, sourcebyteexactaccepted tokenizer, frozen5641672B 5f1bc79e. Actualsoloworkload48promptIDs/EOS248046 PASS tokenizer-only. No model/GPU fromhelper.
- 27B actualmetadata sourceaudit confirms1199/851textallBF16,64layers/112domains/context512state224002048B. Concrete remaining source-dtype/gammaBF16rounding/windowroles/operation-sizedadmission/K16→V48replication/FLA/fullFA/MLP17408 columns/untiedhead/streamedsource gaps documented in owning ignored audit. Fullinference remainsclosed.

## ROOT continuation 2026-10-20 02:18 UTC

- Row transport ROOT full46case physicalmatrix PASS (6nativefullV/all42+6greedy+6privategraph+18freshunknown+4B4/B8prefix+2pressure+2abort+2admission). Bothmodels/allB2/4/8 exactCPU+B1/originalownership/Ready/rootrollback/source0hotupload/final0-ororiginalquarantine. All46loghashes/EOF/groupretirement reviewed; independentBatch finalsource+receipt review NO BLOCKER. All383GPU and454composed serving sources remainexact. Matrixscope does not promote productiongraph/globalcapture/driver admission.
- Newrow releaseserver7cf44de9361541cb4465e2f98f97ff6b0d73abf24e168eca3be48ff6356c9a76/36529336B, worker6bbb7609/36664432B, HTTPf0e10ad5/15708136B frozenbeforedefaultCPUbuild. CLI+CUDAServe strictPASS. BOTHapprovedreleasecache-direct18.75s/HFimport25.29s fullV/all42bothmodelsPASS. Completeisolatedgate PASS1251Rust/0fail/70ignored/248suites+18Python/release/installer/Nix. root-full-check-result-1.json bindsactual logs; freshworker14/HTTP12andbeforeafterperformance stillPENDING. Fullgateoverwrote target/release/zerogravity withCPUversion; restoreOWN frozenserver7cf44 bytes atthatpath BEFOREfrozenHTTP tests, noCargoafterrestorewhileHTTPactive.
- Reconciled3owningplans completedBF16/GDN/fresh2Bcontrol underactualfff1842; stock2Bloader pendingtext correctedtopairedPASS. Noqueue/rootREADMEchanges. Currentindexempty; MAINonly6ownedrowtransportpaths+3owningplans, no signedrowcommit yet.
- BenchV3 manifest6c3b8620/all78live+frozen/oldV1+V2exact/fullnewcgroup+tests+bootstrap+docsROOTreview andother14modules mechanicalnamingproof PASS; actual31fixturesPASS. ROOTactualnoGPUcontainer root-native-container-startup-1 confirmsFDboundreadonlycgroup12GiB/swap0/quota4/exactCPU0..3 plusactualfrozenNixnative --versionexit0. No model/GPU fromthisprobe.
- ActualV3policy root-policy-08b-eager-cold-solo-v3-1: nativefrozen4a745527 actualHTTPrequestall8tokenscompleted/ReadyandownedcleanupPASS. GenuinevLLM entersGPUworker thenfails FlashInfer mkdir/root/.cache/flashinfer readonly (XDGoverrideignoredbyoriginalframework). WholeownednonceCID/group/EOF/actualquietcleanupclosed; monitor correctlyrefusesdyingcontainerHPIDmembership, soattemptUNQUALIFIEDretained. No matchedperf.
- ROOTadditivebenchV4 manifest9097245f41294bd048b493a98247a4001f8b05e47f66e88b6433084696732815/all98files; oldfreezesunchanged. Soleexecutionchangebothproviders publicFLASHINFER_WORKSPACE_BASE=/tmp selects/tmp/.cache/flashinfer underexisting2GiBtmpfs; noHOMEoverride/source/math/CLIresources/workload/orderchange. Bootstraprecord+exactinstalledoriginalresolverpin added. Actual31fixturesPASS, independentBatchFULL356linediff+source/pinreviewNO BLOCKER. FreshactualpolicyV4 nativeHTTPall8complete; vLLMworkerinitializing activeROOT82508/soleGPU. Mainmatchedbenchmark stillnotrun.
- CUPTI officialSDKalreadyapprovedlocalimageheaders/library; hostnsys/ncuabsent. Source-onlycollector13filefreeze f9f5c522+29headers/sourcepinsROOTfullC++/runner/validator/contracts/docsreview. ROOTcaught originalcanonicalLifecyclePlanpoll/reap thenSIGKILL PGIDhazard beforeanycollector runtime; BatchpreparesadditiveV2 WNOWAIT+acceptedv4cleanupadapter, oldfreezeunchanged. ActualisolatedCXXstrictbuildPASS0.94s/e918ff2fa1fbb7df81e514cbed564dde1a0770c1464fd791bbc7bda377f5fe7a/32432B;308actualcompilerdeps/18SDKheaderspinned, actualELFdependenciesrecorded, fullapprovedCupti7.6MB2fdab19dhashPASS. SevenactualsyntheticvalidatorfixturesPASS. NOlibraryload/CUPTI/GPUcollector/runtime/HWkernelclaim. Futurecontainerseamseparatelydocsfinalnativeexec-onlypreload; noouterPythonpreload/namespacetrick.
- Coefficientnativehelper523lines+runreplay+validatorROOTfullreviewandSpineindependentreviewNOmath/lifetimeblocker. KnownadditiveJSONcanonicalizationV3publicvalidationPASS remains; nativevalidatorimportmustpointROOTV3 beforeactualnative. Unmerged/unbuiltwhile454rowfreezeactive.
- Source-onlycachedGDN independentSpine fullreview20pins/7parents/12sourcepins: onefixturemaskingbugF32querybeforestride, additive2linefollowup rows256→2 +bad.query→bad.decay frozen71e5e656/stride03dcd6ed. Preparedauthentic .8B H16archivedreplay931d3442 withmandatoryROOTactualsourcebuildreceipt, all266240outputs/268320imports/retention/pressure gates NOTRUN. 2B/H48cachedoperandsnotpresentinexistingarchives. Spinepreparesgenuine27Blayer0publicGDN initialchunk64→truecached11/353 usingACTUALdeclaredQwen3_5architecture, strictall9rawBF16params plusinputnorm; pagealignedparam-onlymaps<3GiB, native/full27Binferencestillclosed.
- Historicalnumericsaudit recoversoldartifactdigest24054920794c549afc0c678f58f6f44a1ecc3228bcfa793aeacc9c8516fb86a2 fromreachable7e61 nestedsource84 plusbatch2048/ubatch512/64/CPU8/F32KV/FAoff andtokenizer. SerializationUNKNOWN, so cannotcompareitdirectlytofreshu32LE44c935 toclaimdifferenttokens. ExacthistoricalIDsnotrecovered. Currentdefault .8B AV/2Balpha counterexamples precedeGDN: pinnedtinyBLASsingleVec8anddistincthorizontal tree differfrom4Vec8AR1/reference. NoevidenceforGDNrecurrencechange. Baselinepreparesclosedsource-only HaswellbulkF32primitive+authentic132096worddefaultreplays; productionfourVec8 pathsunchanged, noimplementation/runtimeyet.

## ROOT continuation 2026-10-20 02:30 UTC

- Actual V4 and V5 policy HTTP diagnostics: BOTH native and genuine vLLM completed one offer/one complete/eight tokens. Same48promptIDs and effectiveEOS248046/greedy fullsampling fields observed. Actual outputIDs differ; no crossengine bit/source-correct quality or accepted timing claim. V4 healthy-engine HPID failure disproves prior simple dying-container explanation. V5 retained exact actual public Podman listing rc0/HPID/five numeric hostPIDs/two literal question-mark rows. Both original nonceCID/group/nonzero-signals-before-reap/EOF/actual GPUquiet closed. Attempts remain UNQUALIFIED.
- ROOT additive V6 ba859c4eb706f3edc61b2920db22e27034b95457b48fca1f28edce10739b7821 parses only bounded positive unique numeric HPIDs plus literal unknown question marks. Unknown rows prove NO ownership; EVERY actual NVIDIA compute PID must match numeric authenticated-container subset. All-unknown/malformed/duplicate/oversized/failed listing refuses. Retain actualrows/unknowncount/subsetscope. Other15modules mechanical sibling/manifest names only, unchanged engines/math/resources/workload/cleanup/timing. All38 actual parser+priorbootstrap/protocol fixtures PASS. Fresh actual V6policy90428 active/soleGPU; no accepted matchedbenchmark. V1-V5 freezes unchanged.
- Source-only collector V2 932e471e frozen eight additivefiles; ROOT read new owner/run/contracts/validator/docs, Baseline independentreview pending. Original f9f5 C++ +actual root-build-1 unchanged. New runtime notexecuted.
- Source-only bulkF32 689221fe/e146 delta frozen threeCPU paths. IndependentBatch found exactupstream Mozilla2024 MIT notice omission; additivenotice correction requested, arithmetic/default pathsunchanged. Authentic132096word ROOT replay notbuilt/notrun.
- Source-only original27B publiclayer0GDN oracle09461af6 frozen11files, actualdeclaredQwen3_5 architecture/two realteacherstreams/strictall9trainedBF16params/11extent-onlymaps2.775GB/12GiBRAM+AS beforeTorch/CPU4/networknone. ROOT fullreview/actualpublicexecution pending. Native math/full27B112states/logits/GPU notqualified.

## ROOT continuation 2026-10-20 02:52 UTC

- ActualV6policy90428 EXIT0; independentoriginalRusttokenizer/effectivepolicy validatorPASS 48promptIDs/EOS248046/greedy/requestcap, bothnative+stockcompleted8rawtokens, BOTHoriginalnonceCID/group/reap/EOF/GPUpointownershipclosed. V6numericHPIDsubsetwithtwoactualunknownrows retained; everyGPUcomputePIDexactnumericmember. Generatedoutputs DIFFER, no crossengine/full42/perf acceptance. root-v6-policy-review-1.json independentlychecksallownerandGPUrows.
- ActualV6quality firstnativecasecompleted5tokens/stop/rawoutput5withthinkingmarkers, thenhelpererroneouslyrequiredstockresponse-IDecho fornativegeneratedchatcmpl-zg-N. Failure retained; ownedcleanupclosed. V7correctsprovideridentitybindingviaEXACTactualnative TokenMetricswireID/providerID/schema/complete/count/allorderedIDs, stockecho unchanged. Peerfoundfooterinttypegap; additiveV8 b0c50e8a/all172sources addsstrictint+float/boolnegatives, originalV1-V7 frozen. ROOT43V7+44V8fixturesPASS, BaselinefullnarrowreviewNOBLOCKER. ActualV8quality49388EXIT0/BOTH7cases complete/allrawSSE+JSON+IDs+usage+finish+owners+GPUmonitorclosed. IndependentROOTrawSSEparse/hash/terminalreviewPASS; all7stockactualpromptIDs independentlymatchoriginalRusthelper/sampling. Nativecodehasstrayclosingfence andprosepackingcontradiction; stockcopyusesentire128budgetthinking/noanswer. QUALITY UNQUALIFIED_DIFFERENT_NUMERICAL_PROFILE; nooutput/EOSusage rewriting. root-quality-review-08b-eager-v8-1.json +root-pre-benchmark-08b-eager-cold-solo-v8-1.json explicitlyretainlimitations.
- ROOT actualnativeCUPTI62536 andindependentvalidator97219 BOTH EXIT0: declaredtwo-request C2/output4window records64328realGPUkernels=64328successfuldirectdriverlaunchcalls=sourcecounterdelta64328,0dropped/APIerrors/parseerrors/graphs. All338169activityrecords accounted. Wholewindowincludesprefix+greedy/validation/transport, NOTdecode-only;14B2cohorts(10StateOnly+4Greedy),all8actualoutputcommitrecordsB2. Plain/observed tokenIDs/texthashes/finish/usage EXACT (requestnonceIDs intentionallydiffer). Bothoriginalnativechildren/groups/nonzero-before-reap/EOF/client/endpointretirementclosed. BaselineindependentboundedmetadatareviewNOBLOCKER. Originaltrace57029980B retained1e5b59cd; no newfullV/all42, perwindowUUIDattestation, vLLMcounts, performance, observeroverheadorphysicalallocatorqualification. root-native-b2-v2-1/validation.json methodPASS only. Sevenharmlessownerfixtures+sevenactivityfixturesPASS; firstprivateLOdownfixtureattemptfailedBEFOREanychildspawn, retained andcanonicalprivateLOsetupfixed.
- Original27BreferenceV2ROOTsourcefullreview+independentBatchreview and12sourcefreefixturesPASS. Actual81644genuineoriginaloperatorsran, firstobservedarchivewritten; failedmetadatahashafterward: BF16singleton-contiguouslaststride10240 cannotviewasuint8. FAIL_UNQUALIFIED manifest/actualsourcebefore/fullopenedFDs/observedarchive preserved; originalnonceCID/group/EOF/FDclosureallPASS. No numericalacceptance. AdditiveV3 d3614b48 same-sizeint16/int32reinterpretation+NumPyCorderbytestream only; no FPcast/source/operator/hook/archiveexpectedchange. ROOTfullV3read+BaselinepeerNOBLOCKER;192literalbits/12layout/canary+dtypefixtures andfullgenuinereference rerunqueuedAFTERtiming.
- ACTUAL UNPROFILED canonical0.8B/eager/cold_process/solo ABBAthenBAAB comparison23278 started02:46UTC, acceptednative4a7455(old9a19901source) versus genuineunchangedstock0.31/GGUF. Protocolfe3da80f/preparationb0c50e8a/prebenchdispositionexplicitlyUNQUALIFIED_DIFFERENT_NUMERICAL_PROFILE/sourcecorrectgoodputFALSE. No Cargo/build/test/model/GPUdiagnosticactive alongside;agentsremainquietsource-only. Firstnative+firststockownedcontextsclosed,nextstockinitializing at02:51. Full8sample/accountingvalidationnotfinished; no acceptedtiming/speedclaimfrompartialrun.
- NewP12typedscatter+certificate sourcecandidate fd06f5a7/091628b5,5paths/11files, onevectorlaunch/twoKVfieldlaunches10bindings avoiding19, sameoriginalpredicate/atomicstatusOR/all42fullcapacitypreflight/originalflight/Ready/authority/quarantine. Sourceestimate90/186/378fewerlaunchesperB2/4/8teacher, NOTmeasured. ROOTreadproductionwrapper/batchwiring/docs/partialqualifier; Baselinepeerreviewongoing. Notmerged/built/runtime; currentrow383/454 MAINfreezeunchanged.
- BulkF32 sourcecandidate689221fe+exactMozilla-attributiona7e8cc3b peerreviewotherwiseNOBLOCKER; no actualcompile/replayyet. Sourceintegrationproposal adac2aa8 readROOT: currentprefilltoken-major; needsealedmatrixoperator+temporalspan inSAMEschedule, notglobalreducerchange/fakeN64. Default2BalsoactualQ8dequant→BLASsgemmroute, distinctfromcurrentactivationQ8integerdots; primitivealonecannotclosefullDEFAULT8K. Existinguserfullapprovalcoversrequiredtechnicalcuts, no newpermissiongate.
- NativeCPUproviderstartup ROOTfullmain/worker/common/runner/validator/policy/negativefixture reviewcomplete +bothindependentagentsNOBLOCKER startup-only. ROOT14syntheticschemafixturesPASS. Unbuilt/unrun6actualcases; fullphysicalkernel/controlguard/nativeDLopen/math/thread/parentdeathadmissionstillUNQUALIFIED. Build/runtimequeuedaftertiming.

## ROOT continuation 2026-10-20 03:12 UTC

- Actual unprofiled .8B/eager/cold_process/solo comparison23278 EXIT0; original canonical8samples ABBA/BAAB, acceptednative4a7455 versus genuine stock0.31, all8offers/8complete/64tokens. Independentvalidator status BOUNDED_RAW_MATCHED_SOURCE_PER_PROVIDER_STABILITY_VALIDATED; source/tokenizer/policy/UUID/caps/ownedcleanup match scope. TTFTmedian13.400343s native versus1.560703s stock. Native .5701 versusstock4.64215 rawtokens/offeredsecond. Stockfaster. Internalprovideroutputs stable, crossengineIDs/text/usagefinish differ. One stocksample coalescedsingleIDdelivery→ITLunresolved; sourcecorrectSLOgoodputNULL. PointGPU1059MiBnative versus2143–2243stock, notpeaks/hardstock5GiBcap. Native236270source-dispatchsites eachsample, NOTcanonicalphysicalcount; stockhardwarecountunknown. Coldprocess firstafterready excludesstartupselfwarm/JIT; noOSstoragecacheeviction. No warm/C2/4/8/default-route/bestperformance/fullparity claim. Allownedprocess/container/group/EOF/GPUpointcleanupclosed; actualtiminghostwasfreeofCargo/model/diagnostics.
- RevalidatedMAINfff1842 signedG/indexempty/rootREADMEunchanged/onlyownedrow+plans. Cardquiet41MiB56C/nocompute, noCargo/providerprocess. Connector metadata search stillcontainsno dotlink-l5 callable tool; current sharedworkspace is within running dotlink99966. Next2BV8policy/quality and genuine27BV3transport/reference, reviewedscatter-certificate composition.

- Correction after independent fresh2B originalRust tokenizer execution: canonicalsolo prompt contains47 IDs, not48 as earlier prose stated. Fresh2B JSON is exactsame original0.8B47-ID sequence; actualpolicyvalidator compares COMPLETE arrays, so previous numerical policy acceptance unaffected. Failedposthelper48countassert retained, helperitself exit0 and authentic47-ID output preserved. Owningliveplan corrected to47; oldrunreceipts remain immutable.

## ROOT continuation 2026-10-20 03:23 UTC

- Native typedscatter+originalfinitecertificate fd06 reviewedROOT+BaselineNOBLOCKER, original383source/base3/pin11 exact thenonly5pathsappliedfresh. ROOTcomposed456closure c516760006fcefb85f1a2374c101a0eece8893b87e8d0a3da32b60967fe82d68; currentphase supersedes oldrowruntimefreeze, nooldreceiptpromotion. StrictCUDA/Vulkan/default and2geometryunits PASS; freshrelease libtest36201096B c66134e16dcc5fc597fc42b479f1b546e274f15a583d62faf3f8bca6a7d9a2c2 built247.97s. Physical10case operators and46+2fullmodel/worker14/HTTP12/fullgate/bothroutes/CUPTI/A-B stillpending. Qualifier/serving releasebuild34830 active.
- Genuine27B V3 actualtransport43429 EXIT0/12pinnedTorchBF16+F32layouts/192literalbits/allcanaries/3dtype refusals/ownerclosed. Fullreference66831 EXIT0, independentvalidator92579 PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY; all31sourcebeforeafter55GB/11mappedextents/rawreadsets/initialchunk64+truecached11→353/all48heads/plain-observedfull5120output/40960conv/786432recurrent/sourceowners/allgroupEOFretirement PASS. Actualpublicpayload80360448B;804960cachedinput/798720output wordsavailable. Full112StateRoots/nativeexp/nativeadmission/GPU/fullinference/performance stillFALSE. V2failureimmutable.
- 2BV8prepare94333 EXIT0 exactb0c; actualpolicy47082 EXIT0 +strictoriginalRusttokenizer/effectivepolicyvalidatorPASS; both1offer/1complete/8tokens/ownedGPUclosure. FreshoriginalRusttokenizer output exactprevious47-ID .8Bsequence; old48 prose correctedliveplan. Full2B7casequality82823 runningexclusiveGPU; no2Btimingyet.
- Original27Bpure-role qualification16809776 ROOTfullcapture/support/native/runners/validatorread +fourstdlibboundsfixturesPASS. IndependentBaseline found missing source-pin.json at actualnative CARGO_MANIFEST_DIR; exactold7058B4c74metadata exists, additivefixrequested, noactualwidepublic/native runyet. Narrowrolecancel7759599f+25d9270b ROOTfullread; noMAINmerge. Source-headF32workspaceestimate auditpending;12GiBASiscap/containment, notnativephysicaladmission.
- CPUstartup original961d source/runtimepending; additiveSIGTERM831986a6 fullROOTdiff/guard/contracts/6mockfixture review andactual6fixturePASS. Original53eehelper unchanged; handlerrecords/checkpointsaroundoriginalLogdrain toenteroriginalfinally, neverasync-raises acrossPopen/cleanup. NativeCargo stillnotbuilt; noeligibility.
- ROOT actualvalidatedB2CUPTI kernelnamehistogram fromsmallvalidationmetadata: SiLUsteps10080/64328(15.67%), finite9178(14.27%),statusmerge5230(8.13%); namecountsNOTlatency. Batchsource-only orderedSiLU fusioncandidate assigned, noarithmeticweakening. Container+largerfinitecapacity07a86bf6/4337ad63 prepared, peerreviewongoing/unbuilt/unrun; stocklosscompleteFALSE so matchedhardwarecountsclosed.

## ROOT continuation 2026-10-20 03:42 UTC

- Typedscatterphase all10 physicaloperator/prefixcases36798 EXIT0: bothopaque56346178U32words/canaries andbothnew128casecertificates536799936words exactoldphysicalcopy+oldfiniteoracle, source1848→196fixturelaunchsites; freshbothstatusrefusalsactualoriginalNONFINITE/readback/noReady/fence/handles+enveloperetained; freshbothabandonactualenqueue/noANYterminalreadback/originalunknownquarantine; bothB2prefix4teachers/EOS/unequalfrontiers/all42CPU+B1/future248320/final0PASS. ROOTallloghashes/EOF/groupretirementreviewPASS. Notmodel/latencyacceptance.
- New456source release artifacts34830 EXIT0: qualifier37020720B6869207e15112b0508d5ee294a6b246670c8a8674332edb0d3512c3209f2f94e; server36566200B87e1d727f908368f508eeb1e4dd53456f14348b8c72c4519cbee5e52b977c61e; worker36701296B108bdcc15255ebcb034234a5d1894862c25b0d9f4baf81fc7651f1a011d9922e; HTTP15708136B02c4b4eda6b6eedb8670b2ea68335645f15a0ca326e8be1d07f4466380a44f7c. StrictCLI+cudaPASS. Fresh46physicalmodelmatrix26118 started03:25 approx; first35casesPASSthrough2BB8graphunknownlogits, all0.8B B2/4/8 model/greedy/privategraph/allunknownreadoutsPASS. Remaining2BB8unknownGreedy +prefix/pressure/abort/admission, worker14/HTTP12, actualcounts/latencypending.
- Mandatoryfullgate54117 EXIT0:1251Rust/0failed/70ignored/248suites+18Python/release/installer/NixPASS fe98439fa6ac9fd3be936079ad9aeceead43464adc0342db5b2ad4d5e4849f55. Bothapprovedoriginalrelease cache-direct19.10s/HF-import actualfullV/all42bothmodelsPASS (log4252c9c1/210f03a7). All456sourcepostgateexact, root-full-check-result-1.json. Fullgateoverwrote target/release/zerogravity withCPUversion; restoreOWN87e1frozenserverbeforeHTTP/noCargoafterrestore. No signedrowcommit yet.
- Actual2B7casequality82823 EXIT0; ROOTnewroot-review-quality-2b-v8-1.py independentlyparsedALL14SSE/hash/events/text/IDs/finish/usageraw/originalnativefooter/allactualstockpolicy47-vs24etc inputIDs/originalRust2Btokenizer7prompts, originalowner2/container/PGID/signals-before-reap/EOF/actualcgroup12GiB/swap0/CPU4/everyGPUPIDnumericmembership/quietclosurePASS. Fiveofsevenfullrawtexts EXACT(nativeandstockminusoneEOSIDs/usageunchanged). Bothaddition128lengththinking/noanswer; stockcopy128lengththinking/noanswer/nativeBLUE. BothcodebalancedPythonfence+optionalASTadd/body, proseexactlowerdensityopenlattice. QualityUNQUALIFIED_DIFFERENT_NUMERICAL_PROFILE/sourcecorrectgoodputFALSE. Newroot-pre-benchmark-2b-eager-cold-solo-v8-1.json bindsactualpolicy/quality/tokenizer/loaders/baseline/caps, readyforREALunprofiled2B ABBAafterallCPU/GPUchecksquiet. No2Btimingyet.
- Original27Bpure16role/32casepublicreference21899EXIT0 +independentvalidator60149PASS_ORIGINAL27B_PURE_OPERATOR_REFERENCE_ONLY: all881856rawinput/outputBF16words, actualfull248320head/17408downcols/publicF.linear+originalRMS/full18shards55GBbeforeafter/rawreadsets/COW+FDsourceowners/53eeprocessgroupEOFretirement. Noall-roletrainedproducers/nativeparity/full112states/GPU. Rootmetadata-v2 e9581cf9 restoresbyteexact7058B4c74source-pin.json atactualCARGO_MANIFEST_DIR; ROOT3fixturesPASS+BaselineNOBLOCKER, original16809776unchanged.12GiBAS/RAM fixedcontainment, processpeakNOTqualified/VAallowancesestimates. Rolecandidate0f9+65cd+7759 stillnotMAINmerged/nativebuilt.
- CUPTIcontainer07a86bf6+capacity4337ad63 fullROOTsource/delta/namespace/rawPID/validator/lifecycle review +Spineindependentfoundfinalartifactbookkeepinggap. Additivefinalization2b82ca5f/2896b710 reserves2×4MiB/two slots BEFOREfinalreport/validationwrite and fullposttreeassert; old19filesimmutable. ROOTall17mockfixtures(6capacity+7transport+4finalization)PASS; peernarrowrepairreviewpending. ROOTactualstrictCXX6172EXIT0, newcapacitycollector32440Bd2f8346af7ef0a42b50c214152a8eb11c6c2e9c8753d138298d75ab994d2d1d2, actual308dependencyclosure e9ab33c1/ELFreadback, noGPUsdkdownload/providerexecution. PendingactualcontainerC1canonicalwindow/newcapacity/loss/peer/overhead/stocktrace; stocklosscompletenessFALSEalways so fullmatchedhardwarecountsclosed.
- CPUstartup originalactualstrict39418 FAILED oneClippyunnecessary_unwrapmain190; all21externallocktuples exactMAINsubset/effectivelockb716aa70, failurepreserved. ROOTmaterializedroot-native-v3: native5dc8e3b3fd806e9ccc13b77e6f19679da0ccc6fb5d7ebd590eebc685f0616b95 fivefiles changesONLYequivalentiflet+relocatedCargocorepath; common/worker/staticclosurebyteexact. Execution34563b94e5e80413a41ed246a9aa57d9000703ef00798ce2e677cd7d9b518957 fourfiles mechanicaloriginal831runner/validatorcontractname+byteexactguard+separateold961d/newnative/executionclosure binding; nooldnativefreezeretag. Actual52708strict+releasePASS,932256Bbinary92cd48dc635aba728294633937846bec50ba18629796358f4878d00ed98938b9. Peerreviewbefore6actualCPUonlystartupcasespending. NoCargoorheavyCPUjobsactive nowexceptGPUmatrixCPUoracles.
- Newsource-onlyH48typedcachedunusedgammafollowup26d9c6e7/6e41d1e4: cachedrecurrence consumesNOgamma; validateactualF32/512orBF16/256rank128 underexistingCPUprofile/issuer, noactualgatednorm/exp/Alog/GPU/modelguardwiden. H48genuineV3cachedreplay4ecc2873/5b54f199 bindsall1603680words, sameexecute_cached_gdn_slice/SourceWeights/factory/128MiBpool with16MiBcontrol, live118284672B/512MiBAS/900s; unbuilt/unrun/noMAIN. Baselinepeerreviewrequested. SpinepreparesdistinctBF16-gammaRNE-beforeSiLUgatecontract withsameengine/newstatefreeoperator, nofullmodelenablement. Batchsource-onlyordered12stageSiLUfusioncandidate/204adversarycases nearingfreeze, originalpathsunchanged. BaselinepreparesboundednativeT2/4teacherchunk/originalkernels/onefinalReady/allrowatomicadvanceT afterpriorityreviews, source-only.

### 2026-10-20 04:01 UTC — current phase and genuine baselines

- Current456-filec5167600 GPU state-transport/certification composition: strictCUDA/Vulkan/default +qualifier/CLI strict/release builds PASS; all10 operator/failure cases PASS, all46 complete-model/private-graph/prefix/pressure/unknown/admission physical cases PASS, full isolated1251 Rust/0fail/70ignored/248suites +18Python and both approved original cache-direct/HF-import release routes PASS.14 original worker/12 HTTP reruns still pending; source remains frozen. ROOT source-prepared root-serving-physical-1.py uses original53ee WNOWAIT owner and restores exact87e1d727 CUDA server after the CPU fullgate build; peer review requested. No benchmark or commit acceptance yet for this optimization.
- Genuine2B V8 eager/cold_process/solo/max8 comparison session80343 active, seven ofeight ordered fresh processes retired and finalstock starting. No Cargo, other GPU/model execution or payload diagnostics concurrent with actual timing. Exact prompt array has47 IDs; old48 prose is explicitly superseded without modifying old freezes.
- Native CPU startup corrected V3: strict/release52708 EXIT0; binary92cd48dc, allsix actual cases21649 EXIT0 and independentvalidation EXIT0. Scope PASS_STARTUP_OWNERSHIP_ONLY_PROVIDER_UNLOADED. Both original961d803b and SIGTERM831986a6 freezes preserved; narrow ROOT derivative5dc8e3b3/34563b94. Prospective DSOs remain unloaded; actual native provider/math/whole physical envelope still unqualified.
- Genuine original27B public pure-role reference21899 EXIT0 /independent60149 EXIT0:32cases/16roles/881856 raw BF16 words PASS_ORIGINAL27B_PURE_OPERATOR_REFERENCE_ONLY. Only hidden prefix producers are source-trained; other role inputs are explicit external probes. Native wide-role/cached-H48/gamma fragments unmerged/unrun;112 StateRoots/fullmodel/GPU still closed.
- New CUPTI capacity build6172 EXIT0, d2f8346a collector, original07a86bf6+4337ad63 source preserved. Final artifactreservation2b82ca5f reviewed ROOT+Baseline NO BLOCKER;17 actual source-free fixtures PASS. Four-process canonical container diagnostic NOTRUN; no matched physical count claim.
- New BF16-gamma candidate43c400a2 /patchc89eb6c0: ROOT CPU/core/window/slice/HostMath review and Batch full source review, conditional NO BLOCKER after two test-only drop_non_drop removals. Spine additive7f266003/86384ec0 preserves original; authentic replay preparation ongoing. No CPU/native/GPU execution yet. Ordered SiLU originala7939 untouched; additive70c5947f fixes invalid budgets and original init/math slot lifecycle; ROOT read composed delta, actual compilation/PTX/SASS/204physicalcases still pending.

### 2026-10-20 04:34 UTC — completed2B baseline and current serving regressions

- Actual803432B V8eager/cold_process/solo47prompt/max8 EXIT0. Original independentvalidator EXIT0; ROOTall8externalreceipts/loghashes/originalretirement+EOF/all8single-IDgroups/intervals/fullorderedmembershipreviewPASS. Fouroffers/fourcomplete/32rawtokens perprovider; medianTTFT14.5400237855s native vs1.5622418930s stock. Rawusage/offeredsec med0.51045 vs4.64985. Nativep95ITL .186266464–.198809186s vsstock .025755548–.030199654s; all8samplesindividualIDs. Nativepointdevice1857MiB,stock3473–3725MiB. Samefull8IDs/outputhash/usage/finish onthisoneprompt; sevenpromptqualitystillUNQUALIFIED, mathprofilesdifferent/sourcecorrectgoodputNULL/hardwarecountsNULL. Sourcecounter228494native eachsample, B1nofusedpadding. Rawreview root-compare-2b-eager-cold-solo-v8-1-root-review.json. Coldscope firstafterreadiness,notstoragecold/wholecoldstartup; only2ABBA/BAABblocks/noenginewideorconfidencewin.
- Current456c516 row/cert frozenserving12831 EXIT0:14workers953.94s and12HTTP593.68s ALLPASS. Source/87e1server/108bworker/02c4HTTP exact; ROOTbothfullsummary/loghash/knownexit/originalWNOWAITwholegroupnonzeros-before-reap/EOFclosedreviewPASS. root-serving-physical-2.json SHA186259602e1debf69df208c5f12be0f60687a6fb81c43a699c34f5a55dcd0431. OriginalSIGTERM/SIGINTrecordingguard12sourcefreemocksPASS; noGPIO/recovery/otherworktouched. Actualbefore/afterlatency remains; no currentphasecommit yet.
- ROOT own detachedFFF CPUmodelcomposition nowstrict+focused237pass/0fail/2ignored afterpreservedactualfailedintegration/fixtureattempts; MAINunchanged. Fullnotesinallowedownrun model-execution-root-bf16-composition-20261020/TODO.md, final521freeze7 036ce0dc. Genuine source-native replaysstillpending, nofull27Bpromotion.
- Nativebefore/after source6e5fd796 fullROOTreviewfound singleUIDnative-gpu namespace incompatiblewithacceptedrootlessPodmanstore; V2exactfullrootlessnamespace correction+actualharmlessownerfixtures requested (originalfreezeuntouched). Actual8sourcefreepure/mockedownerfixturesPASS, notactualchild/GPUproof.32samples/144offers/32unmeasuredordinarywarmups planned, notexecuted.
- SoleGPUactualcanonical.8BC1 CUPTI diagnostic80030 started04:33:30UTC, original07a+4337capacity+2b82finalization source/built d2f8346a; fourfresh nativeplain/observed,stockplain/publicprofilerobserved processes. Count/output/lifetime diagnostic ONLY, no performancetiming usable. NootherGPU/provider byagents.

### 2026-10-20 04:50 UTC — source-preserving replay and benchmark continuation

- MAIN revalidated fff1842 signedG, no staged paths, only owned GPU/three-plan changes; root README preserved. All26 worker/HTTP and prior56 physical operator/model checks reviewed ROOT+Batch NOBLOCKER. Genuine bounded two-model coldC1 benchmarks remain substantially slower than stock; no parity/speed claim.
- New canonical container CUPTI80030 failed beforeSTART at mapped-file identity, both owned native contexts closed cleanly, no accepted count. Intentional metadata-only44955 also refused beforeSTART and retired owners. Exact native/collector full hashes+inodes match; on actual Btrfs their stat device46 differs from kernel mapping device36. Independent SAME O_RDONLY FD readonly4096-byte mmap reproduces this distinction; CUPTI overlaydevice101 matches. Original freezes/failed receipts unchanged. Spine prepares additive controller+independent validator deriving kernel map identity from verified retained FD, preserving all hash/stat/nonce/pidfd/ns/peer/exe guards, no device exception or privilege change.
- NativeA/B namespace-onlyV2 b0a0371f reviewed ROOT+Baseline NOBLOCKER apart from fixture join bound. Actual6 pure namespace fixtures and9 real harmless owner cases51014 PASS: TERM/INT beforePopen/readiness/openPOST/cleanup plus receiptIO failure, original nonceCID/group/nonzeros-before-reap/EOF closed. Independent ROOT review records failure-free cleanup; initial review script used wrong log hash key and stopped, corrected reader uses sha256_retained. ActualV2 successes do not cover slow valid auth. Additive fixture-onlyV3 60d40c28 fixes join15→25 from TWO10s auth commands+HTTP2+3margin and retains record-only handlers if stillalive;3 mock retirement fixtures PASS, nine fresh actualV3 cases72668 running. Canonical prepare/run/validator remain unchangedV2; no native timing yet.
- ROOT isolated27B final521 sourcefreeze7 preserved. Both relocated H48 replay contracts be67c354/748349ea reviewed; 14new relocation fixtures plus original5/8 bytefixtures PASS (19cached/22gamma). Exact whole521source/old33&12 or25&9 requirements and8 narrow ROOT exceptions authenticate. Initial replaybuild1 metadata --no-deps did not normalize copiedlock; strict refused --locked BEFORE compile. Build2 adds actual offline generate-lockfile; every external package tuple/checksum remains approved MAINsubset. Both standalone strict/release PASS, binaries cached0ac56a44 (1332560B), gammaa381ac4c (1370336B); isolated source/binary/527input attestations retained, no MAIN merge. Authentic cached all1603680 words exported; gamma metadata-only reference prepared. Original owner/native/independent validators now running sequentially under native-gpu user/network isolation17890; complete27B112domains/initial/exp/fullmodel/GPU still unqualified.
- TrueB2T2 temporal source6be63adf/482b3e7a reviewed independentlyBatch NOBLOCKER: causalB2+statelessB4, exactly2 roots advance2, combined original admission/oneflight/4BReady/full42 qualifier. No compilation/runtime/serving/B1solo/throughput proof. Batch assigned directsoloB1T2 original emitter/owner design; Baseline smallest typed temporalspan DEFAULT arithmetic seam. Both source-only, ROOT sole runtime/Cargo coordinator.

### 2026-10-20 04:55 UTC — authentic native H48 slices pass; actual A/B starts

- Original27B isolated cached+BF16gamma native replays17890 EXIT0, both independent raw validators EXIT0. Cachedbe67 all1603680 signedwords/804960imports/798720outputs0diff inclfull786432state before+aftercallerdrop; gamma7483 14stages61632words/typed2final12288/retained245760diff. Complete sourcebefore/after55GB, actual4CPU/512MiBAS, originalWNOWAITgroup/EOF/leaderretirement, realpressure/precancel/lifetimes/finalreserved0 PASS. ROOTbuild2record05e4a911 bindsstrict+release/source036c521/527inputs, noMAINmerge/full112/initial/exp/fullmodel/GPUeligibility. OwningmodelTODOcorrected toactualisolatedslice scope.
- ActualfreshV3harmless72668 EXIT0/all9 realcases; independentROOTallreceipt/logSHA/fullEOF/groupabsence/nonzeros-before-reap/cleanup[] reviewPASS. Three mocked slow-retirement contractsPASS; originalcanonicalV2b0a037 untouched. root-harmless-v3-1-root-review.json recordsboundedcontrolscope only.
- NativebeforeafterV2prepare31442 EXIT0/planf0eeea28ce8b57d5177ea833f3a33f8c880a0682def01724e1682a7451cef49f, bothmodelsC1/C8/warmed_process/max8/sameoriginal47promptIDs/32freshprocesses144measuredoffers32ordinaryunmeasuredwarmups, explicitnativebaseline4a/nativecandidate87. Allactual14worker12HTTP/current456antecedents required. ROOTquiet preflight: RTX3060 UUIDsame6144MiB/41used/63C/util0/noothercompute/noCargo/originalnative modelprocs. SoleactualUNPROFILEDbenchmark6409 started04:55:12UTC underacceptedfullrootlessUIDmap/privateLOwrapper2. Nootherbuild/test/GPU/model/payloadhash jobs duringmeasurement; agentsquietsourceonly. No partialspeed/phaseacceptanceclaim.

### 2026-10-20 05:12 UTC — source-only next cuts while timing remains exclusive

- Actualnativebeforeafter6409 current24/32orderedprocessesclosed,no recordedfailure; both0.8BC1/C8 and2BC1 complete, last2BC8 bandactive. Do notclaimphase/speedfrompartialrun. NootherGPU/Cargo/tests/model/payloadhash work duringtiming.
- ROOT+Batch fullCUPTIV3mapped-FD source14bdbe64 reviewNOBLOCKERinmappedidentityscope: derivesactualkernelmappingdevicefromSAMEreadonlyFD whilepreservingstatdevice/fullhash/path/exe/nonce/pidfd/peer/loss/finalization; alloldsourceunchanged. ROOTidentifiedinheritedasyncSIGTERM __main__ gap acrossoriginalV8Provider.Popenassignmentandcleanup. Source-onlyadditiveV4safeoriginalPlan.samplecheckpoints+record-only00d1 guardandactualharmlessfixturepreparation inprogress; V3noteligibleforruntimeuntilfixed.
- Spineindependentboundedactual27Bcached+gamma receipts/logauditNOBLOCKER: complete851roles, rawBF16gamma256B, allcounts/negativeprobes/retainedinputs/final0,31full-sourcebeforeafterreceipts, allWNOWAITgroup/FD/selector/EOFclosures andisolatedsource036c521+527buildinputs align. Noarchive/binary/DSOhashreexecutionduringtiming.
- ROOTreadfullcc3 sequentialproduction/qualifier675+chunk307+metadata186 andfull6be temporal911/qualifier619/deltas; noconcretesourceblocker. PreparedONLYownignored future-temporal-source-1/ with11sourcefiles, exactcurrentMAIN4baselineparents andexcludedcc3+6be preserved; noMAIN/source-runtime claim.
- SOLO4f2a9d5f sourceprepared: originalP1emitter/certfactor, genuineP2statelessactivations, committed→privateT0middle→final denseB1statechain/all42fullcertificates, fullP2/middle/finaladmission/originaloneauthority advance2/one4BReady/quarantine; nofakeBatchProgram1orroot. ROOTfull470Program/375owner/598qualifier+factor/wiringread,noconcreteblocker. PreparedONLYownignored future-temporal-source-2/14files onexactprevioussnapshot; every4parentbaseline andnew7pathsourcefreezechecked,MAINbytesunchanged. BaselineindependentSOLOpeerpending. All42sequential+temporal+SOLOphysicalcases/compilation/defaultCUDA/Vulkan/fullgate/routes/servingintegration/drivercounts/latency remainUNRUN. Neitherpreparedtemporalrouteisenabled.
- Baselinesource-only3nodeclosedbulkF32typedsliceinprogress,sealedactualone-sequenceTeacherSpan/strides/fullK/sourceorexternaloperands,SAMEPlanBuilder/HostMath/Arbiterretainedoutput,notfullDEFAULTcontextfix.2BdequantQ8→F32BLASandhistcorpusremainopen.

### 2026-10-20 05:28 UTC — row/certificate phase accepted

- Native A/B32fresh processes complete:144/144 measured requests,1152 exact output IDs,32 ordinary B1 warmups; all144 requests yield1008 actual individual-ID intervals. Independent validator/ROOT/BATCH all pass; exact456-source gate retained.
- C8actualB8 median rawrates2.2506→2.28945/s(.8B,+1.73%) and1.6268→1.6543/s(2B,+1.69%); C1unchanged. All128 C8requests miss10sTTFT SLO; no source-correct goodput or vLLM qualification. Source-dispatch cuts52164/sample are not physical count or18%latency gain; thermal variation/pointmemory/four samples scope retained.
- Signed owned implementation+three plan reconciliation commit fb68d8fa165de16e21d1bd37afef3790b80da971, signatureG; MAINclean/rootREADMEa7be69c unchanged.
- CUPTI V3mapping15sourcefree/V4signal3mock +9actual originalPlan/Provider harmless owners pass. Actual V4canonical C1 diagnostic session36244 ongoing; nativeplain/observed OWNERS_CLOSED/noerrors, stock runs pending. No timings from profiled diagnostic.
- SOLOadditive17726 wiring independently reviewed; ROOTprepared14filecomposition chunk.rs matches intended e0cacfdc exactly. Temporal candidates remain uncompiled/unexecuted. Next ROOT GPUphase: exact ordered SiLU operator qualification before any production selection.

### 2026-10-20 05:37 UTC — canonical diagnostic and next dormant operator

- Actual V4canonicalC1 diagnostic36244 EXIT0; independentvalidator38663 EXIT0. ROOT+BASELINE receipt/metadata review PASS. Native0.8B exact47prompt/max8 completewindow236270actual kernels/directlaunches/sourcecounter,0drops/errors/graphs, raw439327905B/2839292records. Four actual original owners close with full EOF/group/CID cleanup and provider-internal plain/observed IDs/text/usage/finish exact.
- Stock publicprofiler records9318kernelactivity/8810hostlaunches; stocklosslessness/allprocesscoverage remainsFALSE. Countcomparison completequalification/timing/quality/goodput/opaqueprofilerallocationadmission remainFALSE. This is newdiagnostic evidence, not an accepted fast/parity route.
- ROOTcomposed orderedSiLU3newpaths plus test-onlylibwiring; source459freeze1967e8fd. No productionselection. SoleCargobuild93973: strictCUDA/Vulkan/default and2sourcefreeunits PASS sofar; release/actualoperator/compiledPTX/SASS/faults/fullmodel/serving/gate/AB remain pending. Oldcandidate+V2 artifacts immutable.

### 2026-10-20 05:57 UTC — genuine27B wide roles and scalar ABI repair

- Isolated wide-role relocation44de7b7a sourcecontract+17purefixtures PASS; source521036c unchanged. Serializedstrict/releasebuild36478EXIT0/effectiveofflinepackagesapproved; actualnative1453EXIT0. All32cases/16roles/881856input+outputwords match originalacceptedpublicreferences; complete248320-row head/17408-column down included. ActualpreCancel/realpressure/retained3035648Bheadinput+output aftercaller/source-drop/final0 allPASS. Nativebinary461af3178e1b185766f9345a455a166e3208569c90effceb301f51e717b8ca21, buildattestation66c51e47. OriginalWNOWAIT/TERM/twoKILLbefore-reap/groupabsent/EOF/FDclosure clean.111.17s diagnosticNOTbenchmark. IndependentBatchsmall-source/receipt review pending; allrolesremainprivateexternal/trained-inputscope, nofullmodel/112domains/GPU claim orMAINmerge.
- Original ordered459freeze1 strictCUDA/Vulkan/default+2units/release PASS (binary99a11b4d). Firstphysical97105 outerEXIT1/native101: NVRTCmov.b32 operand mismatch, no numericalReady qualification; originalgroup/EOF retirementclean. Retained54053Blog5ebd9ef6/root-physical-1 failure. RootreportoldstatusRUNNING_NOT_ACCEPTED +failure meansfailed/notlive. No laterfaultcasesran.
- Root+Batch exactsource diagnosis: BuildAsmExpand.push_input HasValue.value yieldsmutableprimitive pointers; ordinary reinterpret consumesReadValue.load. Immutablefix16198590 addsbit-preservingSSA materializations to15inputs/9helpers; templates/math/geometry/queue unchanged, no vendor changes. Appliedexact3d789csource, new459freeze158be58c. FiveparserfixturesPASS inclrejectionofbothrealfailedC++sources.
- CorrectedonlystrictCUDA/release17635EXIT0, originalunchangedgeometryunits2/priorVulkan/default retained. Freshphysical6-case97197 nowactive; normal204casebits/allintermed/canary/status +5freshfaults remainpending. PrivateCWDcompilation-onlylogging/profilingDisabled avoidsCUBECL_DEBUG_LOGprofiling side effects. RequireactualCPP/PTX/SASSwitness beforeintegration.
- Preparedproductionintegration9110(ROOTfull12kdelta review) remainsunapplied; independentBaseline found2childmethodvisibility errors (needpub(super)preflight_silu/launch_silu), authorpreparingadditivefix. H48coefficient050952reviewfoundactualFAwronglayernegative1→3; additivee4be9998 preparednotrun. Initialquery/cumsumleaf3cda106a source-onlypreparednotrun. Default8K/historicalandfull27B remainopen.

### 2026-10-20 06:09 UTC — ordered leaf physical success; unaccepted integration

- Corrected ordered459 source158be58c/build5f3ce7d3 frozenbinary2199224e passes all6 fresh actual physical processes97197 EXIT0. Normal204cases/171562710 exact numericalbitpairs/all17tracecheckpoints/compact-vs-trace/oldGPU+CPU/status/canaries; five fresh unknown/preexisting/nonfinite/range/knownReadyabandon controls retain original queue/envelope/quarantine, all owned process groups/EOF cleanup[]. No model/performance acceptance.
- ROOT actual compiler audit all144 orderedCPP scalar-asm constraints PASS; four final41120 specializations selected. Existing diagnostic suffix180 naming overwrote distinct PTX, leaving only compactgated ordered_6be19757 (22c9f0d2) retained; its explicit RN/noFTZ/noapprox/noF32FMA token checks PASS. Three other PTX witnesses and actual loadedSASS remain open; no complete instruction-proof claim. Narrow diagnostic-only collision fix source proposal pending.
- Baseline independent integration9110+visibility3bc2+scalar1619 full source review NOBLOCKER; false parser fixture issue explicitly retracted. ROOT composed reviewed four-path resident change+geometryseam+CUDA wiring+visibility by exact patches, preserving corrected3d789c checked_span-to-EOF arithmetic byteexact. Candidate is UNACCEPTED and for regression testing only; no accepted server rebuilt/enabled. ROOT461 strict/default/Vulkan/geometry build34168 ongoing under original native-gpu user/network isolation.
- Actual original27B wide32case881856word proof independently Batch-reviewed NOBLOCKER. H48coeff/initialprefix and DEFAULT bulk typed candidates remain isolated source-only/unrun. Full27B112domains/fullV/initial64/nativeEXP and DEFAULT8K/historical stillopen; boundedstock vLLM remains faster.

### 2026-10-20 06:23 UTC — owner requested pause; current release build draining

- Owner explicitly requested pause at reasonable next checkpoint. Finish only existingROOTrelease97420; DO NOTstart prepared60casephysicalmatrix, original routes, fullgate, worker/HTTP, benchmarks, model replay, newsource scope or GPU activity. All3 source agents receivedpause andpreserved owncheckpoints. No new implementation accepted orcommitted.
- ExactMAIN461unacceptedSiLU+diagnosticcomposition sourcefreeze30fdcfa2b67df26a82a272e14f80b22c315bae7b048784156adb04f52ccba244 remainsverified. StrictCUDA/Vulkan/default+2geometryunits PASS; productioncandidate loadedSASS/fullmodel/serving/fullgate/routes/performance remainopen. Existingphysicaloperator6process/204case171562710word-pair evidence is old correctedleaf2199224e, not new integratedbinaryacceptance.
- NarrowoptinPTXdumpfix exactsource+rustfmt reviewedROOT/Batch/Baseline, noarith/queue/NVRTCchange. RootCargo -p excludedvendorunit failedbeforecompile (featuresoutsideworkspace; retry devdepsoutsideworkspace), bothretained. Exactsameactualhelper/static+wholeactualunit extractedandcompiledagainstexistingrealcubecl_core StableHasher; isolatedunit22489PASS1test/noCUDA. Newrelease97420 libtestf603eb47/qualifierfe960465 built andstrictCLIpassed; finalservingbuild stillrunning. No freshPTX144orSASS/physics yet.
- ROOTprepared .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/root-physical-1.py:6operator+54model=60cases. BatchindependentSOURCEreviewNO_BLOCKER, ROOTASTvalid; NEVERexecuted. Usesnewsource30fd+buildrelease2/frozenartifacts, original53eeowner+00d1signals, fullCPPloggingonly/profilingdisabled anduniquedumps. Resumeonlyafterbuild/source/quietrevalidation; do notreportqueuedcasesasPASS.
- ExactNVIDIAprimaryABI sourceclosureb4b7f125 source-reviewedROOT+Baseline confirmsResourceData.resourceDescriptor→ModuleResourceData. No loadedimageproof. Spinepauseddraftcollectorfreeze5dc4e9b5 (collector+PAUSED+TODO) explicitlyUNCOMPILED/UNREVIEWED/no runner/validator/fixtures. Ownpeerreviewnotes .agents/runs/serving-and-speculation-loaded-sass-peer-review-20261020.
- ROOTowned newdetachedworktree /home/pvl/spaces/abird/src/zerogravity-worktrees/root-default-bulk-20261020 atfb68 hasonlyexactprimitive689/attributiona7e8+typed0d11+GPUrefusal composition, MAINuntouched. Actual463sourcefreezea4891041aa4fb57d3afac7f180212eddc65d735790dfea0f22f4e61eb7cca1b0 in .agents/runs/numerics-and-context-root-default-20261020; NOcompilation/replay/acceptance. Batchunfinishedfaithfulrelocationcheckpoint .agents/runs/numerics-and-context-default-bulk-isolated-20261020/CHECKPOINT.md (sixbaselinecopies only, norunnablederivative). NeedseparatemodelnamespaceMAIN/sourceROOT binding, original132096DEFAULTwordoracle preservation andnewfullbuildattestors. Default8K/historicalfix remainsOPEN.
- Original27B ROOT521036ce0 remainsUNCHANGED. Combined525source/harnessderivative e97eabda frozen in .agents/runs/model-execution-hf-combined525-attestation-20261020; ROOTreadREADME/fullcombined_sourcefirst195lines (remaining8 +609lineREVIEWmustfinishprogressively), noapply/build/run. Initial3cda independentBatchsource reviewNO_BLOCKER; Baselineitsauthoronlyselfaudit. H48wronglayer3corrected andseparateindependentreview. Bothleafreplays/fullinitial64/nativeEXP48/provider/full112/fullV/engineGPU remainpending. Donotreplaceoldsuccessful521nativeartifactsorretagthem525.
- Lastaccepted signedHEADfb68d8f state-scatter/certification phase: physicallyqualified2smallmodels/fullGPU2/4/8, fullgate/routes/servingpass andactualnativeA/B ~1.7% C8 gain. Boundedtwo-modelstockvLLM stillsubstantiallyfaster; no vLLMparity/no3modelcorrectnessclaim. RootREADMEa7be69c unchanged, indexempty, no push/tag/deploy/devicechanges. FinalpausecheckpointJSONwillrecordcompletedbuild/exactartifacts/quietprocessesafterdrain.

### 2026-10-20 06:31:23 UTC — PAUSED at completed release-build checkpoint

- Userpause honored after soleexistingrelease97420 EXIT0. Allfive steps (CUDAstrict,librelease,qualifierrelease,CLIstrict,servingrelease) PASS. Five frozen artifacts retained in integratedroot-composition-1; source46130fd verified unchanged. Compilation is NOT model/runtime acceptance.
- Exact final pause snapshot: [root-pause-20261020-063036.json](root-pause-20261020-063036.json), release/source/binarySHA/currentGitoverlay/targetserver identity included. New60casephysicalmatrix NEVERstarted; workerHTTP/routes/fullgate/bench/SASS NOTRUN for this candidate.
- Actual finalprocess scan: noCargo/rustc/test/inference jobs; nvidia-smi compute list empty. Allthree subagents stopped with preserved checklists/drafts. Indexempty/rootREADMEa7be69c unchanged/accepted signedHEADfb68; unacceptedownedsource remainsUNCOMMITTED. No push/tag/deploy. target/release/zerogravity now contains the newlybuiltUNACCEPTED candidate; accepted old frozen87e1 and4a artifacts remain untouched. Resume must use verified explicitfrozenartifacts, not assume target executable is prioracceptedserver.
- No further taskwork may start until ownerresumes. PendingSASSprototype5dc/sourceABI b4, DEFAULT463a489 relocationincomplete, combined525e97 unapplied; previousboundedvLLM/slower andfull27B/defaultcontext limits remain.

````

## Source: .agents/runs/serving-and-speculation-parity-root-2026-10-19/command-host-fixture-runtime/README.md

Original UTF-8 bytes: 277; SHA256 `45c38ed8d1a7c0341fb2e854ac26b1e3ea7097758fbb9015f7f8e984a2e4f476`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CubeCL Common

The `cubecl-common` package hosts code that _must_ be shared between cubecl packages (with `std` or
`no_std` enabled). No other code should be placed in this package unless unavoidable.

The package must build with `cargo build --no-default-features` as well.

````

## Source: .agents/runs/serving-and-speculation-performance-audit-batch-20261020/REPORT.md

Original UTF-8 bytes: 20837; SHA256 `e2f4d9a7ce75c269ceacc6d2cd15c15ffcbde149b57ddf621dc5b55a357bd114`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Shared-engine performance audit — planning only

Owner: batch_design_review. Parent: serving-and-speculation; ROOT consolidates.
No code, tests, builds, model/provider/GPU/container execution or payload scans.
This is an independent recommendation, not a new implementation tracker.

## Conclusion

The strongest immediate target is the repeated single-teacher execution path,
not a larger advertised batch flag. Native multi-request B2/4/8 is real, but
prefill still advances one teacher per request/wave and emits thousands of
kernels per model-token. A credible route toward vLLM performance combines
temporal prefill, exact leaf fusion, substantially better projection kernels
and eventually production-safe graph replay. None alone establishes a win.

Numerical correctness and physical ownership are release gates for every cut.
The current GGUF CPU-target profile and stock's loaded BF16 arithmetic are
different targets. Faster execution of one does not establish cross-engine
equivalence; the common GPU profile remains a critical-path requirement.

## What is measured, and what is inferred

| Observation | Evidence | Meaning / remaining uncertainty |
| --- | --- | --- |
| Cold-process eager C1 native TTFT13.400s/.8B and14.540s/2B; stock about1.56s | V8 cold-solo accepted raw comparisons, both models | Large raw latency gap. First request after readiness; startup/storage-cold latency excluded. Numerical/quality equivalence remains open. |
| 2B canonical IDs/usage/text match on one prompt, seven-prompt quality still unqualified | V8 policy/quality/pre-benchmark records | Matching one greedy response does not qualify hidden states or numerical profile. .8B canonical responses differ. |
| Warmed native C1 TTFT about4.09s/.8B and4.72s/2B |32-process native-only A/B | Cold/warm gap makes lazy compilation/allocation/initialization a necessary diagnostic. Different experiments; do not subtract them as a measured JIT duration. |
| Native prefix batching cuts earlier concurrent source launches about82%, improving finite-wave time43%/.8B and33%/2B | Native-prefix warm observations | Multi-request prefix fusion matters; C1 changes little. All concurrent samples still miss10s TTFT. |
| Latest C8 source launches292116→239952/.8B and284340→232176/2B; median rates+1.73%/+1.69% | Accepted native-only32-process before/after | About18% fewer launches did not yield18% speed. No causal per-family time breakdown or confidence-certified improvement. |
| 64328 actual native kernels in short B2 window; SiLU15.67%, finite14.27%, merge8.13%, copies5.76% of count | Validated native CUPTI +small family histogram | Real count inventory, **not latency fractions** and not canonical C1/current candidate. |
| Canonical native C1 has236270 actual kernels/direct launches; stock trace has9318 activities/8810 launches | Separate diagnostic records | Native window is qualified. Stock losslessness/all-process coverage remains open; no qualified hardware-count ratio. |
| Native prefill_count returns1; B1 and batch execute/certify the original node list | Current gpu_engine.rs/resident_program.rs/batch.rs | Actual source bottleneck opportunity. Sequential per-teacher body dispatch remains. Its critical-path share needs timing. |
| Existing projections map one request/output row to one GPU thread and reuse exact source dot loops | batch_quant_projection.rs/mixed_ops.rs | Genuine B batching, not a tiled high-throughput matrix multiply. TensorCore/CMMA use is absent from these paths. Compute/BW/occupancy dominance is not measured. |

The proposed ordered SiLU operator passed204 old-GPU/CPU cases and171562710 bit
comparisons; production integration remains unaccepted at the paused checkpoint.
Its original PTX filename collision invalidates complete specialization coverage,
not the retained physical comparisons. The collision-safe diagnostic and new
model/serving gates must close before measuring it as accepted production.

## Priorities and measurable gates

### 0. Close the comparable numerical target and benchmark accounting

- Keep native GGUF regression optimization against accepted independent CPU/B1
  full-vocabulary/all42 references. Keep original-HF official loader, restored
  F32 CPU profile and stock CUDA loaded-cast profile explicitly distinct.
- For genuine common-HF CUDA comparison, qualify the actual loaded parameter
  casts, BF16 projection reductions, GDN/FLA initial versus cached schedule,
  RoPE/attention, KV layout and complete state/logits. .8B B1 stock subset and
  public GemmEx settings are useful antecedents, not full-model equivalence.
- Extend existing comparison tools minimally with an authenticated independently
  validated parity record and original physical publication/committed-token
  binding. Record must bind weights/quantization, tokenizer/template, numerical
  profile, graph/program generation, generation policy, source/test scope and
  hardware. Output hashes or caller declarations alone cannot create it.
- Current serve-compare deliberately leaves source-correct SLO goodput unset and
  equivalence NOT_ESTABLISHED. Add paired process/block statistics after these
  gates; preserve legacy usage-throughput fields and never relabel them goodput.

Acceptance: full required mathematical/StateRoot/physical regression suites;
comparison rejects stale/incompatible parity records and unverifiable commits.
No source-correct performance claim until these prerequisites hold.

### 1. Obtain a critical-path decomposition before choosing further kernels

Use separate diagnostic processes/windows, never profiled main timing samples.
Derive kernel-duration totals and interval unions, launch/API blocking time,
host gaps, H2D/D2H and device-copy bytes, synchronization count, allocation,
compile/cache misses and shape reuse. Sum of overlapping durations is not wall
time. Associate data with prefill/decode/maintenance and exact request/window.

- Compare cold-process versus declared warm conditions; explicitly attribute
  compile, initialization and H2D rather than moving cost behind readiness and
  claiming it disappeared. Report startup-to-ready separately.
- Close native/stock collector coverage, losses, child-process/PID attribution
  and device/RAM overhead. Native counter equality applies only to its exact
  workload; graph enqueue1 is not underlying kernel count1.
- Establish a simple roofline for projection/state traffic from **measured**
  bytes and time. Counts, low point GPU utilization and code inspection alone
  cannot distinguish bandwidth, arithmetic, launch bubbles or host stalls.

Acceptance: diagnostic output transparency, zero lost/unowned activity and
closed owners; explain the dominant measured latency components before a major
kernel rewrite. Existing bounded trace may support offline analysis by ROOT;
this investigation did not scan its raw payload.

### 2. Finish small exact fusion already supported by source

- Qualify ordered SiLU in both original B1 and lifted B2/4/8 emitters, including
  plain/gated real widths, independent old operator/CPU arithmetic, status,
  canaries, source profile, exact buffer leases and unknown/Ready cancellation.
- Then rank RMS, coefficients/sigmoid, convolution/reduction, attention small
  chains and producer/certification/status merge by actual critical-path time.
  Fuse only while preserving every original rounding/store/FTZ/sign boundary
  and predicate, persistent error OR, cancellation and quarantine.
- Reuse fixed original workspace initially. Hoist validation/binding cold;
  avoid per-token metadata allocation. Never remove finite/range checks merely
  to improve counts, clear prior failure status, or silently retry after launch.

Acceptance: all intermediate/final bits versus independent original pipeline;
full model/all42, worker14/HTTP12, original release routes and isolated gate.
Ordered SiLU predicts11 plain or12 gated launch-site savings per executed node;
verify actual work and measure latency. No percentage speed forecast from count.

### 3. Temporal prefill, including the dominant solo case

Two distinct milestones must not be conflated:

1. Sequential T2/T4 state-only chunk: gather once, existing B-body repeatedT,
   certify every produced state before consumption, persistent status once,
   final scatter/one4B Ready and one begin_advance(T). This amortizes host,
   transport and terminal sync, **not projection/MLP kernel traversal count**.
2. True temporal stateless lowering: original B1/T2 through existing B2 math,
   B2/T2 through B4, then B4/T2 through B8. Batch stateless projections/MLP
   across actual teacher rows; execute complete attention/GDN causal regions
   in original teacher order. Same original plan lineage, no fake requests or
   authorities. This can reduce repeated mathematical dispatch and reuse weights.

Existing cc3 sequential,6be temporal and SOLO proposals are source-only. Qualify
them in that dependency order, including solo B1; optimizing only C8 cannot fix
the observed C1 gap. Do not label a B-loop or sequentialT loop native temporal
matrix batching. B8/T2 needs qualified tiling or new geometry; B16 is not implied.

Preserve real teacher EOS as input, final output-tail reservation, nonzero
histories, state versions, canonical future KV zeros, intermediate failure
latching, all-before-flight capacity admission and one independent authority
per real request. T2 first; bounded T4 only after evidence/fairness. Keep parked
outputs/cancellation responsive and cap the scheduler quantum by actual work.

Acceptance: CPU+B1 all42 after each teacher, native intermediate/final all42 and
future fullV, exact logits/packets, pressure/refusal/unknown completion and final0.
Measure C1 TTFT/host calls/readbacks/copied bytes and C2/4/8 mixed-load decode
tails; require measurable gain without correctness or declared-SLO regressions.

### 4. Rework the projection data path, not just its launch wrapper

- First exact GGUF route: tile output/request rows, coalesce packed weight loads,
  share activation codes/scales and retain the same per-block integer dots,
  zero points, scale arithmetic, eight FP32 lane accumulators and final tree.
  Investigate DP4A or bounded integer MMA only when integer subresults can match
  original lane/block grouping and overflow limits exactly. Do not reassociate
  FP scale reductions, replace activation quantization or narrow precision.
- Precompute immutable source scale metadata only under the source identity and
  physical budget; avoid persistent expanded weights exceeding admission.
- Separate common-HF CUDA route: native BF16 tiled/TensorCore projection in the
  SAME operator/provider seam is essential research for vLLM-class arithmetic
  intensity. Public GemmEx settings do not identify its hidden reduction kernel.
  Original-actor stream/handle/workspace/internal allocation/Ready ownership and
  authentic operand parity must close before a cuBLAS provider is admitted.
- Existing exact CPU-target BF16 CUDA kernel intentionally reproduces CPU64
  order and is not evidence of stock CUDA arithmetic or TensorCore throughput.

Acceptance: operator timing, achieved BW/occupancy/reuse, all codec scale/code/
intermediate/status bits, source-independent adversaries, complete model/state
and resource safety. TensorCore math requires its declared independently proven
GPU profile; it is not a universal replacement for CPU-exact GGUF reductions.

### 5. Production graph reuse and synchronization reduction

Private graph proof exists; production admission does not. Close participation
for every same-card command, constructor/init H2D/allocation, external event/
storage, sync and final Drop. Same original actor/stream and fair queue only;
no second context, global mutex held across await or caller isolation fiction.

- Capture fixed owned math workspace; dynamic original-root gather/candidate
  scatter and independent publication stay outside. Keys bind source/program
  generation, numerical profile, row band/capacity/readout and relevant shapes.
- Bound graph/executable/driver metadata, cache entries and retirement owners.
  Eviction waits for real completion; unknown completion quarantines owners.
- Device-token feedback and graph variants need their own proof; current private
  host-teacher proof does not authorize them. Cold admission refusal can choose
  another qualified path; post-launch errors cannot become fallback.

Acceptance: physical exclusivity under competing init/Drop/commands, fairness,
epoch revocation, no capture allocation leaks, CPU/B1/native all42/fullV,
abort/pressure/cache invalidation and final0. Measure host enqueue/bubbles/TTFT;
underlying kernels remainK per replay even when host graph API calls become1.

### 6. State traffic, attention and scheduling capacity

- Measure bytes first: current packed gather/scatter remains real state traffic
  even after row transport cut its kernel count. Consider direct per-row leased
  state bindings for uncaptured kernels, or immutable paged KV/prefix sharing
  under the original source/session lineage and physical ownership. GDN private
  candidate state still needs genuine independent writes. Do not alias writable
  roots or infer unpublished candidates are safe to overwrite.
- Graph fixed buffers and direct dynamic state bindings have different ownership
  constraints; benchmark both after qualification instead of silently capturing
  mutable published-state pointers.
- Fused attention/softmax and more parallel GDN/chunk prefill can be major long
  context work. FlashAttention/FLA cannot simply replace the CPU-target schedule:
  casts, reductions, masking, padded lanes and initial/cached order need their
  own independently verified common GPU contract.
- Continue ragged band/refill/age-aware decode-prefill scheduling; retain real
  typed capacity refusal and2/4/8→smaller bands, output credits, at-most-once SSE
  and prompt/decode fairness. Measure offered load and aging, not just full waves.
  Raise concurrency/geometry only after actual active-state/metadata/graph/cache
  admission; no silent bump to fill the GPU.

## Fair staged benchmark matrix

Use unchanged canonical serve-bench.py/serve-compare.py and a new explicit
protocol derivative when metrics/parity hooks are added. Existing compare caps:
requests<=128, concurrency1..8, repetitions1..4; bench has other bounds, which
do not authorize changing compare/admission. Start at C1/2/4/8. Keep all failures.

| Stage | Work | Conditions / purpose |
| --- | --- | --- |
| A: exact regression | Existing47-ID prompt/max8 plus seven-prompt quality, genuine EOS and unequal outputs | .8B/2B, complete state/logits first; preserve current comparability/history. |
| B: startup/solo | Canonical C1 cold-process and warm-process | First after readiness versus exactly one canonical ordinary warmup. Separately report process startup-to-ready. Neither means storage-cold. |
| C: concurrent | Canonical C2/4/8 finite waves, then bounded repeated waves | Test true fusion, shape/JIT warm effects and tail retirement; ordinary C1 warmup does not fully warm C8. Add an explicitly separate shape-warmed protocol if desired, symmetrically for both. |
| D: prefill/decode shape | Short1/32/47 input, medium128 input, long448 input; outputs8/64 where capacity fits | Current accepted GPU capacity512 constrains prompt+actual generated frontier. Do not schedule512+64 or advertise8K GPU serving. Report actual IDs/EOS-dependent work. |
| E: interference/SLO | Open-loop fixed arrivals, long-prefill plus short decode, ragged refills, cancellation/backpressure and two workload labels | Sweep offered rates through saturation, finite queue and fixed deadline; test starvation, overload and committed goodput, not a closed-loop throughput illusion. Labels are not authenticated tenant isolation. |
| F: best route | Genuine pinned stock default/graphs versus qualified ZG best path | Eager/eager is a diagnostic row. Do not handicap stock to eager and call it best vLLM, or enable private ZG graphs prematurely. |

Keep equivalent weights/quantization/loaded casts/tokenizer/template/generation,
actual raw token IDs, EOS/usage/finish, source/hardware/context and policy limits.
Use the same original rootless/private isolation, GPU UUID, CPU4/12GiB RAM/swap0
and declared device envelope as current .8B/2B protocol; collect actual readbacks.
Changing caps or admission is a new point for BOTH providers, not a rescue run.
Counterbalanced ABBA/BAAB fresh-process blocks remain useful. More evidence can
use repeated bounded outer blocks without silently raising tool limits.

Report offered/accepted/rejected/timed-out/cancelled/completed requests, committed
and delivered tokens, p50/p95/p99 TTFT and **actual individual-token** ITL,
SLO-qualified committed goodput per offered second, queue/client backlog,
memory allocation peaks versus point samples, power/clocks/temperature, fairness
and actual hardware kernels/direct APIs/graph enqueues separately. Coalesced IDs
leave per-token delivery unresolved; mean ITL or SSE chunk timing cannot repair it.

Add paired process/block confidence intervals for predeclared primary metrics.
Do not treat tokens within one process as independent repetitions or median
per-sample p95 as a pooled p95. Check thermal/order effects and all losing points.
A speed claim needs a confidence-supported improvement on its named workload
with source-correct SLO-goodput and predeclared latency/resource/no-starvation
constraints. A finite-wave win is not sustained capacity or broad vLLM parity.

## Model feasibility and dependencies

| Model | Honest next scope | Main blockers |
| --- | --- | --- |
| Qwen3.5 .8B | Accepted GGUF exact serving optimization and common-HF GPU qualification; bounded matched C1/2/4/8 after parity | Current cross-profile quality/state gap; token-major prefill, scalar projection path, production graphs/stock collector closure. |
| Qwen3.5 2B | Same engine/cuts, actual2B per-shape/resource proof rather than .8B extrapolation | HF shard4.548GB exceeds old4GiB complete-source guard; differing original/loaded casts, initial/attention/FLA closure; tighter active/workspace capacity. |
| Original Qwen3.8 27B | Finish initial exp/provider/core +common typed full model/112-domain CPU truth, then source-window streamed CPU/GPU and matched feasible offload | Isolated wide/cached/gamma operators pass but full model absent. Full55.563GB source cannot reside in current12GiB process cap or6GiB GPU. No stock exact-architecture/offload equivalence is established. |

Actual retained hardware is an RTX3060 **Laptop6GiB**, not a12GiB desktop card.
Current GPU operator envelope5GiB/provider4GiB is smaller again. Nominal27B
2-bit weights alone are about6.75GB before block metadata/state/workspace, so
even that is not an assumed fully resident solution. No such qualified source
is implied by a metadata listing. BF16 needs streamed/offloaded windows or a
separately authorized larger hardware configuration.

For27B, first prove the same engine's full original math and state; then measure
SSD/RAM/GPU bytes, throughput, source leases, I/O completion and overlap with
actual ResourceArbiter readiness. Multi-teacher prefill can amortize each staged
weight tile; streaming decode must disclose bandwidth/cache limits. Compare only
against stock's genuinely supported same source/profile/offload policy and caps.
If stock or ZG cannot execute that point, mark it infeasible/unqualified rather
than substitute another model/quantization/device or invent a benchmark.

## Source/evidence entry points

- `.agents/plans/serving-and-speculation/{TODO,FUSED_GPU_BATCHING}.md` and
  `.agents/plans/model-execution/TODO.md`: live gates and accepted scope.
- Current `crates/zerogravity-cli/src/serve/gpu_engine.rs:507`, cohort.rs and
  `crates/zerogravity-gpu/src/resident_program/{batch.rs,silu.rs}`: scheduling,
  repeated body, status/certification and candidate integration seams.
- `crates/zerogravity-gpu/src/{batch_quant_projection,mixed_ops,source_accumulation}.rs`:
  original scalar native source reductions; CPU-target BF16 kernel expressly
  excludes TensorCore math.
- Native-only32-process A/B and independent review:
  `.agents/runs/serving-and-speculation-native-before-after-20261020/root-native-v2-1`,
  `../serving-and-speculation-native-ab-peer-review-batch-20261020/REVIEW.md`.
- Stock raw V8 cold-solo validation/pre-benchmark/quality records under
  `.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020`.
- Actual short-window family histogram:
  `.agents/runs/serving-and-speculation-cupti-activity-20261020/root-native-kernel-families-1.json`.
- Sequential cc3, temporal6be and SOLO source proposals remain unexecuted;
  no implementation/performance acceptance follows this planning report.

````

## Source: .agents/runs/serving-and-speculation-performance-audit-batch-20261020/TODO.md

Original UTF-8 bytes: 791; SHA256 `79c86fd1f08b4c433c52379ea66cbce8f3192ef31dddc642cf1a211c8b7fe94f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Performance investigation checkpoint

Parent: serving-and-speculation. Owner: batch_design_review.
Scope: this ignored prose report only; ROOT consolidates the owning plan.

- [x] Read live serving/model gates, source scheduling/projections/status seams
  and retained benchmark/profiling metadata.
- [x] Separate measured latency/counts from bottleneck hypotheses and pending
  source-only optimization proposals.
- [x] Prioritize shared-engine cuts with dependencies/physical acceptance,
  truthful metrics/tool gaps, bounded matrix and27B feasibility.
- [ ] ROOT consolidated decisions; no implementation or execution authorized
  by this investigation-only task.

No code, builds/tests/provider/model/GPU/container execution, payload scans,
MAIN/worktree/shared-plan edits or commits.

````

## Source: .agents/runs/serving-and-speculation-pinned-reference-2026-10-19/REFERENCE_SCOPE.md

Original UTF-8 bytes: 6908; SHA256 `bdde01fd850d384b16363466e1d559c6c9325104823235def8fb05f6baeb6f79`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Official .8B CPU reference scope

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Owner: `/root/spine_review`

Source: `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`; every supplied source file is
SHA256-verified before and after capture against `source-receipt.json`.
Runtime image: `docker.io/vllm/vllm-openai@sha256:`
`b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f`.
Installed Transformers5.17.0/Torch2.13.0+cu129, CPU execution only. Installed
operator/config/cache/loader/dispatch source copies and SHA256 pins are in
`pinned-runtime-source/` and `runtime-source-manifest.json`.

## Distinct parameter profiles

- `official-bf16-loader`: unmodified official
  `Qwen3_5ForConditionalGeneration.from_pretrained(dtype=torch.bfloat16,
  attn_implementation="eager", use_kernels=False, local_files_only=True)`.
  Actual loaded parameter audit:473 BF16 tensors. All36 original F32 tensors
  are cast to BF16; the audit records each changed element count/max delta.
- `source-36-f32-restored`: same model/operators; only18 original
  `linear_attn.A_log` and18 original `linear_attn.norm.weight` parameters
  restored to their exact original F32 dtype/bytes. Distinct artifacts;
  this profile is not the unmodified official requested-BF16 loader.
- Checkpoint488 tensors:320 ordinary text,153 vision,15 auxiliary MTP.
  Official ordinary conditional-generation class omits MTP using
  `_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]` (modeling source925).
  Audit covers all488, explicitly excludes only those15; vision is loaded
  but never executed. No omitted ordinary text tensor.
- Official tied head is the same parameter object as
  `model.language_model.embed_tokens.weight` (source1762); no checkpoint
  `lm_head.weight`. Logits retain BF16 (source1865–1869).

## Actual operator protocol

Two separate cached forward calls consume exact IDs11 then353, with no chat
template, sampling, greedy feedback, loss or generation helper. First token
uses18 official slow chunk-GDN calls; second uses18 official slow recurrent-GDN
calls. Each step uses6 eager-attention calls. Transparent Python call
observation asserts counts at the actual official source function bodies;
no arithmetic implementation is replaced.

- Slow GDN casts q/k/v/beta/g to F32; state is F32 with layout
  `[batch,value_head,key_width,value_width]` (source301–497).
- Slow recurrent reads key width on axis-2, updates `k[...,None]*delta`, then
  reads query along axis-2. Output casts to original query dtype. This
  independently establishes state orientation; it does not establish the
  CUDA FLA reduction/rounding schedule.
- First-token chunk pads to64 and uses its official chunk algebra. That
  source path differs from simply starting the recurrent loop at zero.
- Gated RMS normalizes F32, casts normalized input before learned weight,
  multiplies SiLU(gate.float()), returns input dtype (source218–235).
  The source itself notes that FLA may differ in its cast behavior.
- Eager attention computes softmax in F32 and casts back to query dtype;
  full-attention cache stores original BF16 keys/values.
- Official convolution cache retains4 projection positions per channel;
  the next four-tap convolution only needs newest3 as prior history.

## Artifacts and layout transforms

Each profile/step contains full248320 BF16 logits and all48 raw official cache
tensors in `raw-official.safetensors` (36 GDN tensors plus12 K/V tensors).
`manifest.json` records original dtype/shape/stride and archive SHA256.
Exact F32 widenings are available as little-endian binary files:

- Convolution:18 domains, original `[B,C,4]` -> newest3 `[B,C,3]`,
  channel-major, oldest-to-newest within each channel.
- Recurrent:18 domains, original `[B,H,K,V]` -> `[B,H,V,K]`, preserving bits.
- Attention:6 paired domains, original K/V `[B,H,T,D]` -> `[B,T,H,D]`,
  all flattened keys followed by all flattened values; valid positions
  recorded. No invented capacity or padded tail is added.

These42 semantic domains are comparison material, not Rust `StateRoot` objects
or physical lease/Ready evidence. `validate_reference.py` independently loads
raw archives, checks every file hash/full finite tensor and compares all
derived F32 files bitwise with their documented layout transformations.

Unqualified: CUDA/FLA mathematics, actual vLLM model inference or loader dtype,
Rust numerical parity, GGUF quantization, serving/API/generation-policy
equivalence, latency/throughput, long context, .2B/27B or other models. CPU
diagnostic durations and profile-to-profile deltas are not acceptance limits
or benchmark results.

## Executed result

Capture exit0; maxRSS4,396,932KiB, cgroup memory12GiB/CPUs4, CUDA unavailable.
Both profiles completed exact inputs11 then353. Official BF16 argmax321,599;
source-F32-restored argmax198,599. Their logits differ222106 and223924 values
out of248320; maximum absolute deltas0.34375 and0.1875. These are reported
differences, with no cross-profile acceptance threshold.

All36 original F32 parameter tensors are audited. BF16 cast changes2009 values
across18 learned GDN norm vectors (maxabs0.00390625); A_log values happen to
remain exact while their loaded dtype changes. This distinction affects
arithmetic promotion even when parameter values are exactly representable.

Each step also has15 actual observed intermediate tensors in
`observed-intermediates.safetensors`: embedding; layer0 input norm, qkv,
source conv-SiLU return, attention output, residual, FFN input norm, gate,
up, actual down-projection input, FFN output, final residual, recurrent state;
final model norm and logits. The gate's valid-stage names omit FFN input norm
and final norm; no claim that this layer0 subset is a complete gate trace.

`observed-stage-subset.json` projects each actual archive into541696 exact
F32 words (~5.96MB): first-step conv return is cropped to its final1 position,
matching the official caller's crop; no arithmetic reconstruction. The
custom subset schema explicitly refuses a whole-model gate claim.

`diagnose_original_reference.rs` is a standalone, unexecuted harness for the
coordinator. Its only execution math is existing verified source/bf16 bridge
prepare-with-trace and publish. It authenticates the capture and subset pins,
compares all logits/all42 committed domains, and retains full Rust traces.
The existing original trace never emits its allowed `attention_input_norm`
stage; the harness records that coverage gap and compares the other12 mapped
observations. Source FFN-inputnorm/finalnorm extras have no existing stage.
KV codec transform is explicit: Rust original state interleaves each
position'sK,V; canonical reference groups allK then allV. Any numerical or
coverage gap yieldsUNQUALIFIED/exit2, with no tolerance; operational errors
exit1. Final source owner reservation must be0 after all borrows/owners drop.

````

## Source: .agents/runs/serving-and-speculation-pinned-reference-2026-10-19/TODO.md

Original UTF-8 bytes: 5684; SHA256 `266d0c28515a1ec103f7e0553b1b0dc29a27b8f9ff773d66628d0457d30f7510`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Official pinned .8B BF16 CPU reference

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Owner: `/root/spine_review`
Scope: ignored scripts/artifacts only; no production/tool edits, Cargo or GPU.

- [x] MAIN authorized isolated CPU reference: image digest pinned, source
  read-only, network none, no GPU devices, memory<=12GiB/CPUs4/threads4.
- [x] Inspect actual pinned Transformers model/cache/slow recurrence source.
- [x] Record source/runtime pins, tied head, cache schema and actual dtype/layout.
- [x] Capture authentic tokens11,353 full248320 logits and all42 state domains,
  separately for official BF16-loader and source36F32-restored profiles.
- [x] Copy actual layer0/final intermediate outputs with transparent hooks:
  embedding/norm/qkv/conv-SiLU/GDN-output/residual/FFN gate/up/fused/output/
  residual, final norm/logits and selected recurrent state;15 tensors/step.
- [x] Validate archive closure, finite values, raw hashes and independent scope:
  initial validator exit0,4 complete step archives/168 semantic domains/
  4 full248320 logits/60 observed intermediates. Runtime-source hash extension
  rerun exit0;5 installed source files match copied pins. No extra model run.
- [x] Project actual intermediate archives into four bounded stage-subset JSON
  files (~5.96MB each), preserving exact BF16→F32 words and source hash pins;
  first-step conv return cropped to newest1 as official caller does.
- [x] Write/format standalone `diagnose_original_reference.rs`; root owns
  compilation/run. Existing source loader+arbiter<=4GiB, BF16 diagnostic
  bridge prepare_with_trace/publish, exact all248320 logits/all42 state and
  existing observed stage comparisons; KV per-position(K,V) codec explicitly
  transformed to allK/allV. Difference/missing trace =>UNQUALIFIED/exit2;
  operational error exit1; verifies final source reservation0 after owners
  drop. No Rust compilation or inference by this agent.
- [x] Root compiles/runs harness against freshly built facade/artifact libs.
- [ ] Root resolves numerical differences using actual source evidence.
- [x] Root compiled harness successfully; first actual run exited1 before
  inference on overly narrow raw-only reservation assertion; finalreserved0.
  Fixed harness assertion to existing composite lease: immutable1,746,942,600
  bytes +exact metadata Scratch8,778,902 =1,755,721,502 <=4GiB. Opened-header
  calculation and per-AllocationClass snapshot must both match, actual
  SourceWeights.source_len remains exact shard bytes. Formatted/frozen,
  coordinator recompiles/runs; no engine changes or weakened resource check.
- [x] Proposed reusable capture/validator/projection tooling in
  `TOOL_PROMOTION_PROPOSAL.md`; read current owning scripts, no MAIN edits.
- [x] Root reports second harness run correctly exit2/UNQUALIFIED: all42
  typed state/logit/stage comparisons emitted and source/meta budgets return0.
  First observed arithmetic gap is layer0 GDN QKV:1/6144 at position0,
  3/6144 at position1; embedding exact. Independent baseline agent replayed
  actual Torch norm/linear against captured intermediates and source-derived
  reduction investigations; no tolerance/precision shortcut accepted.
- [x] At root request, add separate public-operator probe per step before
  prepare: existing math.ordinary_norm(embed_token(token), original layer0
  inputnorm gamma),1024F32LE; existing cpu.project(original layer0 qkv,&norm,
  None),6144rawF32LE. Descriptors explicitly noTraceRecord/no coverage
  acceptance. Existing engine trace gap remains. Formatted/frozen; root
  compiles/runs; this agent does not run CPU/GPU inference again.

Results:
- Capture exit0, maxRSS4,396,932KiB; actual cgroup12GiB/4CPUs; no CUDA.
- Ordinary official conditional-generation class loads473 tensors and omits
  15 auxiliary MTP under its own documented ignore rule. Header has488;
  ordinary text320 +vision153 +MTP15; no missing ordinary text weight.
- Official BF16 loader:473BF16. Cast changes18 learned GDN norm vectors,
  2009 elements,maxabs0.00390625;18 A_log cast dtypes but values unchanged.
- Restored profile:437BF16+36F32, all36 exact original tensor bytes.
- Official argmax321,599; restored198,599 for inputs11,353. Cross-profile
  logits differ222106/248320 then223924/248320 values;maxabs.34375/.1875.
  These differences are evidence that profiles must remain distinct.
- Actual first-token slow chunk18/causalconv18/eagerattention6; next-token
  slow recurrence18/convupdate18/eagerattention6, each profile.
- Attempts1/2 preserved: first exposed auxiliary MTP exclusions in audit;
  second completed official BF16 steps but failed JSON serialization of
  official loading-info sets before restored profile. Final capture includes
  explicit exclusions and supported set serialization; no math changes.
- Remaining: root independently compares the chosen pinned numerical profile
  with Rust stage/logit/state evidence. No numerical tolerance was introduced.
- Trace coverage: `attention_input_norm` is in the accepted stage vocabulary
  but live original math never records it. Harness emits missing coverage
  and continues other12 mapped stages/allstates rather than inventing values.
  FFN-inputnorm/finalnorm source extras are explicitly unmapped. Full existing
  Rust trace is retained; source reference remains layer0/model subset.

Official model: Qwen/Qwen3.5-0.8B
Revision: 2fc06364715b967f1860aea9cf38778875588b17
Runtime: vLLM0.31.0-cu129 image
sha256:b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f

This CPU Transformers reference does not qualify GPU FLA, GGUF quantization,
vLLM inference/throughput, trained27B or equivalence to the Rust engine.

````

## Source: .agents/runs/serving-and-speculation-pinned-reference-2026-10-19/TOOL_PROMOTION_PROPOSAL.md

Original UTF-8 bytes: 4833; SHA256 `83c0973982d0aabc28f7c90e7ed641777afb9047161ccf7c8e8b021b218e1bf9`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Reusable original-source reference utility — proposal only

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Owner: `/root/spine_review`
No MAIN tool/engine changes authorized or made for this proposal.

## Smallest useful promotion

Add `tools/qwen-original-reference.py` with `capture`, `validate`, `project`
subcommands. Consolidate this run's capture/validation/projection code rather
than retaining a second implementation of the same artifact schema.

- `capture`: official installed Transformers only; lazy Torch/Transformers
  imports. Local verified source directory + pinned source manifest + exact
  input IDs + explicit parameter profile + new output directory. Preserve
  original module arithmetic; observation hooks returnNone. Default CPU
  eager/slow path; actual operator-body call counts and tensor dtypes/layouts.
- `validate`: standard library only; source/runtime/profile/shape/domain
  closure, binary hashes, exact BF16/F32 words, finite checks, documented
  conv/recurrent/KV transforms. Validate completeness independently from
  numerical agreement. Existing raw Safetensors are the primary capture;
  derived layouts never erase raw cache positions or overwrite archives.
- `project`: bounded actual stage words for existing numerical diagnostic
  consumers; explicit complete/subset coverage, stable observation order,
  no invented layer/stage/StateRoot values. Keep FFN-inputnorm/finalnorm as
  extra observations until the existing engine has real matching trace sites.

Retain a short optional `tools/qwen-reference-container.sh` launcher:
caller supplies an already acquired immutable image digest and local source;
read-only source, offline environment/network none, no GPU device/CDI,
bounded CPU/memory/pids, no privilege changes and no automatic pulls. Record
actual cgroup limits and runtime/source file hashes in the artifact. Keep
large model captures/artifacts in ignored runs, not tracked source.

## Profiles are acceptance inputs

- Keep `official-bf16-loader` and `source-36-f32-restored` separate. Never
  interpret original-source identity as proof of identical loader policy.
- Generalize future restoration by a versioned, explicit GDN parameter rule
  and source-pinned expected parameter names/counts, not a universal count36.
  Reject unclassified original F32 text parameters; no silent broad restore.
- Audit all checkpoint tensors, actual loaded dtypes/cast deltas and exact
  exclusions. This .8B checkpoint has320 ordinary text +153 visual +15 MTP;
  ordinary class loads473 and ignores MTP by its own pinned rule.
- Capture genuine tied head and exact EOS/config/tokenizer differences.
  Fixed ID capture bypasses chat templates deliberately; a separate prompt
  profile must pin actual rendered IDs and generation policy.
- CPU eager/slow reference, CUDA FLA reference and actual vLLM execution are
  different operator/runtime profiles. A new profile requires a real pinned
  execution and independent math qualification; no relabeling existing data.

## Existing tools and engine remain authoritative

`scripts/compare-qwen-layers.py` compares an older JSONL/F32 stage schema with
predeclared tolerances; `scripts/compare-attention-value.py` is a strict,
checkpoint-specific GGUF attention gate. Neither currently captures original
HF state or loader profiles. Do not redirect or weaken those gates to accept
the new two-token subset. `scripts/cpu-qualification-gate.py` remains its
existing GGUF corpus/context acceptance gate.

Promote the standalone Rust harness later as a thin CPU diagnostic example of
the existing facade, compiled through Cargo with the same artifact versions.
It must keep SourceWeights/TransferAdmission/QwenOriginalCpu/math/bridge;
no copied inference arithmetic, loader, queue or root issuer. Avoid making
manual rustc/rlib selection the long-term reproducibility interface. Tracked
implementation should replace the one-off ignored harness when reviewed.

## Focused qualification before promotion

Source-free default selftests: BF16/F32 exact words including signed zero;
malformed/duplicate JSON, bounded dimensions and counts; source/runtime hash
changes; missing/duplicate/foreign stage/domain; incomplete versus complete
coverage; padding/crop and KV codec order at positions1/2; nonfinite words,
overlapping/out-of-file tensor ranges, altered binaries; profiles excluded
from equality when loader policy differs. No default model/SDK download or
hardware test. Add these source-free checks to the existing script gate.

Then rerun the existing authorized .8B physical CPU capture and whole-state
comparison through promoted tools. Preserve observed failures and original
tolerances. Capture closurePASS and Rust numericalUNQUALIFIED are independent
outcomes. Expand to2B/27B and GPU FLA only under their owning phase gates.

````

## Source: .agents/runs/serving-and-speculation-rope-angle-audit-20261020/DESIGN.md

Original UTF-8 bytes: 7485; SHA256 `22b90fc027ac374c072ad2aac6d54dbe03e44f0331d0e72100d9b7796999398f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Ordered RoPE angle launch reduction — source-only proposal

Parent: `.agents/plans/serving-and-speculation/TODO.md` (P12). No implementation,
compilation, operator/model execution or performance measurement in this run.

## Actual boundary

- Both approved Qwen3.5 GGUF models record `rotate=64`, `theta_scale` from the
  existing `10_000_000.0f32.powf(-2.0/64.0)` descriptor. Head width256;
  six full-attention layers (3/7/11/15/19/23), each with half32 angles.
- B1 `ResidentProgram` invokes `angle_step`32 times per RoPE node. Native
  `BatchProgram` invokes `angle_step_rows`32 times, each writing one stage for
  every row. Native B2/B4/B8 remains one model graph, not one angle launch/row.
- Each RoPE node also invokes three original input copies, original sincos,
  products/sums/subtract/apply, and three output copies: `half+11=43` internal
  launch sites. Output certifications outside this method remain unchanged.
- One ordered angle launch would make this internal count12, removing31/layer,
  186 across the six attention layers. The current192 angle launch sites become6.
  These are source launch-site counts, not timing or throughput evidence.
- Separate `CubeResidentProvider::try_rope_kv_async` also uses the stage loop.
  Preserve it initially as an accepted operator oracle; a later provider hunk
  needs its own reviewed accounting/qualification. No hidden provider substitution.

## Exact numerical contract

Existing batch stage0 writes `f32::cast_from(positions[row])`; B1 passes the
existing host `position as f32`. For stage1..31 the actual physical operation is:

```
angle = round_source(widen_source(previous) * widen_source(theta_scale), 0u32)
```

`widen_source` reconstructs FP32 subnormals; `round_source` explicitly packs
nearest-even FP32 subnormals. Preserve these exact helpers and sign argument0.
Do not replace them with plain multiplication/conversion, a closed-form power,
parallel prefix, reciprocal, trigonometric fusion or a different zero policy.

Proposed closed CUDA kernel: one active lane per row, stage0 copied/cast exactly
as today, then a sequential loop writing all stages into the same original
request-major angle buffer. Each iteration reads the prior stored FP32 value
through the original helper boundary. Complexity O(B×half); no inter-row
dependency, barrier, temporary persistent allocation or host upload. A kernel
per output angle recomputing all preceding stages costs O(B×half²) and offers no
reason to prefer it here.

The store/reload boundary must survive generated code. Source syntax alone does
not establish equivalence: root must inspect emitted code/PTX and compare every
angle bit against the original stage kernels before selection. Reuse theta's
already recorded FP32 bits; do not recompute its `powf` or use a host constant.

Preserve original sincos table, Q/K/V packing, head/row index mapping, product
rounding, source FMA repair, copied nonrotated lanes and current-token KV slab.
No source identities, PlanOperations, state layout, SessionAuthority, readout or
publisher change. StateOnly executes the same required state-producing RoPE.

## Selection, ownership and failure

- Select only exact CUDA server identity, following the existing typed
  `activation_quant_fused::is_cuda_service` pattern. Explicit non-CUDA selection
  retains the original loops. A CUDA capability/shape/compile/launch error must
  propagate; no error-based legacy fallback.
- Existing cold F64 capability and finite theta/geometry/position/source guards
  remain authoritative. Initial scope may be half32 and widths1/2/4/8 only;
  every rejected CUDA shape must fail before the first candidate launch.
- Use existing angle scratch, positions, service/client/stream, physical leases,
  original attempt accounting, Ready/status and unknown-completion quarantine.
  No new buffer, uncharged cache, transfer or warm initialization.
- Report one actual launch for the selected angle operator, preserve all outside
  launch counts and certifications, and verify real completed forward counts.
  Native projection-dot and sampler subsets do not change.
- Captured graph source/body K and executable identity change with this topology.
  Preserve frozen original/NEXT graph artifacts; rebuild and qualify a fresh
  graph against the new source. Never edit their old receipts or reuse an old K.
- Current CUDA NVRTC source lists architecture/include/lineinfo only, without a
  `--use_fast_math` flag. `supports_features.fast_math=true` is a capability field,
  not an equivalence guarantee. Do not change compiler flags or rely on implicit
  contraction; generated kernel inspection and raw-bit comparisons remain gates.

## Minimal disjoint implementation proposal

1. `rope_kv.rs`: new closed angle kernel(s), preserving old stage kernels.
2. New child `rope_kv/angle_qualification.rs`: actual admitted raw-buffer probe
   and finite synthetic fixtures; parent module declaration only.
3. Narrow `resident_program.rs` RoPE angle selection/count hunk and
   `resident_program/batch.rs::rope` corresponding hunk. Preserve current cold
   activation preparation/native-prefix and graph factoring; root merges fresh
   bytes instead of replacing these files wholesale.
4. Separate proposed qualifier route wiring; no serving-mode or model guard
   change. Existing provider stress path initially remains the old oracle.

## Root qualification gates, in order

1. **Angles:** real approved theta bits and positions0/1/2/3/7/31/119/120/121/
   127/128/255/256/511/512/1023/4095/4096/8191/8192/16383/16384/32767/32768.
   Full angle buffer old B1, old row kernels and candidate raw-bit equality;
   B2/B4/B8 unequal frontiers, permutation, repeat, padding/suffix canaries.
   Add scalar/theta boundary fixtures for subnormal rounding and signed zero,
   labeled synthetic and compared to old physical kernels. No tolerance widening.
2. **Independent source:** accepted CPU iterative FP32 angle sequence for real
   trained theta/position scope, then `source_rope_kv` full Q/K/V bits. Existing
   `rope-stress-cuda` captures actual trained inputs from `blk.3.attn_rope_kv`
   after teacher IDs11/353 and checks22 positions through32768, normal/edge QKV.
   Extend angle observations explicitly; do not treat CPU libm as a synthetic
   GPU rounding oracle outside the qualified scope.
3. **Runtime contracts:** invalid geometry/source/foreign service refuses before
   launch; cancellation before dispatch, actual Ready discard, original physical
   pressure, and pending-launch drop retain the original leases/quarantine with
   unchanged roots. No fresh allocations/uploads after cold preparation.
4. **Whole model:** both approved GGUFs, CPU + accepted B1 + ordinary native
   reference versus selected candidate B1/B2/B4/B8; all248320 logits and all42
   domains after every teacher step, unequal histories, native StateOnly/future
   decode and outside greedy feedback, lease return/cancel/pressure/abort.
   Rerun relevant worker/HTTP source-bound counter and no-early-output gates.
5. **Acceptance:** strict CUDA Clippy, focused checks, full isolated check and
   both approved original cache-direct/HF-import release routes. Root then runs
   bounded source-matched before/after measurements. Launch reduction alone
   establishes neither speedup nor vLLM equivalence.

No gates above have been executed by this audit. Root decides whether to
implement after reviewing this proposal and current critical-path ownership.

````

## Source: .agents/runs/serving-and-speculation-rope-angle-audit-20261020/TODO.md

Original UTF-8 bytes: 921; SHA256 `9bb9296057803bde3f8b364e08995f15d2bffae695fc2515280533d65c0e7836`.
Resolve embedded relative paths against the original path, not this volume.

````text
# RoPE angle audit

Parent: `.agents/plans/serving-and-speculation/TODO.md` (P12).
Owner: `/root/baseline_review`; source-only proposal, root executes/merges.

- [x] Read current B1/native/provider RoPE source and source-float helpers.
- [x] Check approved model geometry and actual32-stage/six-layer launch count.
- [x] Inspect same-engine CPU oracle, existing physical stress/abort routes,
  typed server selection and actual NVRTC flag construction.
- [x] Propose one ordered CUDA angle kernel with unchanged buffers/math/state.
- [x] Preserve frozen original/NEXT graph source/count qualification boundaries.
- [x] Record root-only physical/numerical/performance gates; no runtime claims.
- [ ] Root review/authorization of implementation ownership; no code yet.
- [ ] Root compilation, real angle/full-model/worker/HTTP/resource qualification.
- [ ] Root measured matched performance and shared-plan reconciliation.

````

## Source: .agents/runs/serving-and-speculation-row-state-transport-20261020/API.md

Original UTF-8 bytes: 4355; SHA256 `af7402927625b53580702caaeb6bf460cf09376fc838ce907e773123ccbf3f13`.
Resolve embedded relative paths against the original path, not this volume.

````text
SOURCE ONLY — row-native U32 state transport. Owner baseline_review.

- NEW generic primitive `resident_state_transport::{Shape,Direction,Row,
  PreparedTransport}`. No numerical profile, SourceIdentity issuer, session,
  StateRoot, allocation, upload, event, Ready or publication capability.
- Original source/generation authority stays in `ResidentModel`:
  `prepare_fused_execution` validates every RootBinding/token/source, exact
  per-session lease/service/generation and independent domains, then collects ALL
  original physical accesses and submits through the original queue/service.
  Raw Handle equality is not logical source/generation authorization. New
  primitive validates physical service, checked visible ranges and aliases only.
- `geometry(Shape,B)` admits EXACT B2/4/8,1..1M U32 words/row, <=8M total.
  KV slab>0,capacity1..512 with checked products; no old1M aggregate helper is
  reused. Actual8×262144=2097152 words fits. All byte arithmetic is checked.
- `ensure_capability` requires exact selected CUDA service and actual binding,
  cube/thread/grid capacity. Vector ABI9 buffers, KV18; no hidden dispatch split,
  retry, implicit device choice or backend fallback. Non-CUDA B1 unchanged.
- `prepare(client,rows,denseK,denseV,shape,direction)` checks every active private
  and dense buffer before binding/range helpers: same original physical ServiceId,
  checked allocation minus end trim then start,4B alignment, sufficient real span;
  rejects duplicate/overlapping visible ranges even for read-only private rows.
  Clones original handles only; no GPU buffer/source cache or copied values.
- Four kernels, one actual launch/domain/direction: vector gather/scatter
  `[B,N]`; KV gather/scatter `[T,B,H,D]`, same exact U32 mapping as original.
  Separate gather uses immutable private inputs; scatter uses immutable dense
  inputs. Exact raw signed-zero/subnormal/nonfinite payloads; no float conversion,
  canonicalization, math, finite-certificate removal or status-policy change.
- Fixed ABI inactive argument slots reuse row0 but are unreachable at actual
  checked B. All logical rows are real independent sessions; no fake model rows,
  padded tokens, duplicated active state or publication change.
- BOTH directions/ALL domains are preflighted inside existing
  `BatchProgram::validate_runtime`, before original `submit_cohort`. Capability
  checks also run before new batch workspace allocations. Original packet,
  candidate, staging, state, output and scratch admission/physical budget unchanged.
- Native and qualifier-only Graph share the SAME outside-body IO methods.
  `launch_inputs` now returns Result; Graph propagates it. Captured body nodes,
  kernel count, source math and graph completion owner are unchanged. Every
  original private/dense handle still appears in execution/physical flight
  accesses; unknown completion retains the same whole attempt/graph owner.
- Original per-row kernels are retained only for test+qualification+CUDA oracle;
  no runtime fallback. No resident_program.rs/public helper re-export is needed:
  physical tests mount as children of batch's oracle module. Only new module
  registration in lib.rs plus narrow batch.rs/graph.rs wiring.

Expected SOURCE launch sites for approved42 domains (not a timing claim):

| Operation | B2 | B4 | B8 |
| --- | --- | --- | --- |
| Original gather+scatter |168|336|672|
| Proposed native gather+scatter |84|84|84|
| Source count difference |84|252|588|

Existing per-candidate finite certificates remain separate and unchanged.
Projection/sampler/body/graph-enqueue counters are not relabeled. Total forward
kernel receipt adds the actual new per-domain launch returns once. B1 untouched.

ROOT integration:

1. Review `NEW_DELTA.patch` and `ROOT_WIRING.patch`. Baseline pins in BASE.json.
   Apply NEW_DELTA to current source; root's unmerged GDN change only appends
   typed cold-refusal matches, so compose the narrow batch hunks. Never copy the
   whole proposed batch/lib over concurrent work. Copy new files with freshmtime.
2. ROOT_WIRING adds only `mod resident_state_transport;` to current lib.rs;
   proposed full lib is a review aid, not an overwrite instruction.
3. Static formatting passed; no author compiler, test, model, provider, operator
   or GPU/runtime invocation. No source/profile eligibility or speed claim.

````

## Source: .agents/runs/serving-and-speculation-row-state-transport-20261020/ROOT-QUALIFICATION.md

Original UTF-8 bytes: 4361; SHA256 `26a170bc6159fe215b6705115f8deed87ad9a7e5d78d38700568c3dc329c6551`.
Resolve embedded relative paths against the original path, not this volume.

````text
ROOT ONLY — prepared physical gates, all UNEXECUTED by author.

Source-free and strict gates:

- Focus new `resident_state_transport::tests`: real2097152-word geometry, exact
  8M bound, refused overflow/invalidB/capacity, checked end/start/byte underflow and
  valid nonzerooffset views. Run relevant strict CUDA/default/Vulkan Clippy,
  approved model/cache release routes and full isolated `scripts/check.sh` as
  required by the owning implementation phase. No compilation establishes GPU
  support or graph/serving acceptance.

Actual copy-only qualifier:

- Fresh RELEASE CUDA library test binary with qualification feature. Run each
  ignored test alone under original isolation/coordinator:
  `resident_program::batch::state_transport_qualification::physical::original_qwen35_08b_native_state_word_transport`
  and corresponding `original_qwen35_2b_native_state_word_transport`.
- New pinned ORIGINAL ResidentService client/queue; original source identity;
  256MiB finite provider,5GiB device cap; physical ResourceArbiter reservations,
  initialization, actual Ready/readback and final reserved0/no fence/events.
- B2/4/8; vector13/18432/262144 and synthetic KV5×3/real512×512; complete payload
  plus prefix/suffix canaries on EVERY original private input, old/new packed and
  old/new scattered row/channel. All views have real nonzero offsets. Old physical
  vector/KV row kernels are independent oracles; synthetic tagged host words also
  require correct original row mapping. Includes both zero signs/subnormals,
  infinity/NaN payload transport; no numerical inference claim for those words.
- Actual foreign physical service metadata, duplicate/overlapping active rows,
  adequately sized dense/private alias, invalid end/start and short payloads
  refuse before launch without upload/lease/event change. Original source/epoch/
  root authorization is a separate preserved owner gate, not proved by Handle
  metadata. Original per-row dispatch count2B versus actual native2 for one
  gather+scatter fixture; no measured throughput/driver enqueue claim.

Mandatory complete model/owner matrix AFTER copy-only physical proof:

- Freshly compiled current source `.8B/2B × B2/4/8` accepted `resident-fused-cuda`
  full-logit and `resident-fused-greedy-cuda` feedback routes; independent accepted
  CPU+B1 full248320/all42 at every accepted step. Preserve negative rows, root/
  epoch/packet/snapshot ownership, abandoned preparation and physical lease return.
- Native teacher-prefix library tests for both sources at B2/4/8: actual StateOnly
  each teacher/all42, trained EOS suppressed, no generated packet/output; future
  greedy/fullV matches; ragged frontiers+256/257/512 boundary, tail readout barrier.
- Both-source pressure fallback tests and actual `hardware-abort-probe` StateOnly,
  full-logit and incoming/outgoing greedy abort routes. New module allocates no
  GPU buffers; prior original handles/readiness/quarantine and retained provider
  envelope must stay exact. No recovery after unknown completion.
- Re-run existing `resident-cuda-graph-wide-cuda` B2/4/8 for both sources:
  StateOnly/Logits/Greedy-outside body, independent CPU+B1/native/all42/fullV,
  stale/foreign/restore, abandon, pressure and actual graph abort. New copies are
  outside capture; old capture source/binary receipts do not qualify new IO.
  Production/global graph admission remains CLOSED.
- Fresh actual worker and HTTP B2/4/8+unequalarrival+prefix cases, credit parking,
  Ready cancellation/survivor retry, no duplicate source IDs/SSE/usage/Done, every
  row's exact final state and final physical0. Existing health conservation must
  use actual Native/StateOnly/Greedy/width and total kernel counts.
- Only after those pass, preserve the preceding accepted server as a real pinned
  baseline, then same-source/caps/temperature/tokenizer warmed ABBA/BAAB naturalc8
  AND B1 with canonical tools/owned lifecycle. Keep offered/completed/IDs/usage,
  TTFT/actual individual-ID ITL/fairness/SLO/memory/source-bound counter receipts.
  Previous `.8B/2B` warm timings predate this code. No speed/vLLM claim until
  independently measured; genuine matched vLLM acceptance is still separate.

Use existing accepted qualifiers/supervisors, not a second model/token loop.
No new tracker, model acquisition, driver change or shared plan edit is proposed.

````

## Source: .agents/runs/serving-and-speculation-row-state-transport-20261020/TODO.md

Original UTF-8 bytes: 2767; SHA256 `98e24f3874cefcd6fb5312565eda2d9797773217ced12aa9585f74721e39d3e7`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent serving-and-speculation. Owner baseline_review. SOURCE ONLY.

- [x] Read existing whole Native/Graph gather/scatter and physical owner paths.
- [x] Prepare bounded row-native U32 copies, prelaunch physical range/capability
  checks and minimal shared outside-body IO wiring; source math unchanged.
- [x] Prepare original per-row physical differential, opaque-bit/canary/refusal
  fixtures and ROOT complete model/graph/worker/HTTP/pressure/abort gate matrix.
- [x] Source formatting only; no author build, tests, operator/model/provider/GPU.
- [ ] ROOT+peer full source review, composition, strict build and focused tests.
- [ ] ROOT actual physical copy and complete model/owner matrix above; final0.
- [ ] Actual before/after measurement; no inferred speed/profile/graph admission.

## ROOT actual continuation 2026-10-20

- First MAIN merge preserved prior GDN coldrefusal/BF16module; original sourcefreeze unchanged. First strict failed4unsafe-context and missing physicalfixture imports before runtime. ROOT compile-fix-v2 e5caecdd corrects ONLY these, peer review no blocker; no shader/layout/math/ownership change.
- Corrected CUDAstrict12.32s, two geometry/range units, Vulkanstrict10.57s/defaultstrict8.97s PASS. Current383-sourcefreeze d691d213; sourcefiles stable. Release libtest0dd31b5a/36110984B built5m27s, postbuild source383 exact.
- Actual original.8B/2B U32 copy-only physical B2/4/8 PASS each56346178 complete words, all visibleprefix/suffixcanaries/signedzeros/subnormals/nonfinites exact vsoldphysicalkernels/hosttagmapping. Original140 per-row launchsites→30native across15fixtures, Ready/source/lease/queue/nohotupload/finalphysical0. This is NOT a full-model/performance claim.
- ROOT fullmodelB2 prefix comparison on BOTHsources active; B4/B8/full-logit/greedy/graph/abort/pressure/worker/HTTP/currentperformance remaining. Fresh qualifier releasebuild stillactive. Only ROOT Cargo/test/GPU coordination; no agent runtime.

- ROOT qualifier release built3m29s/frozen36987952B0af306de; source383 postbuild exact. Both B2prefix fullmodel PASS fourteachers/EOS/unequalfrontiers/all42CPU+B1/full248320future/final0. ROOT46case serialized native/logits/greedy/privategraph/unknown/prefix/pressure/admission matrix active; no performance claim/production graph admission.

- ROOT all46actualfull-model/graph/unknown/prefix/pressure/abort/admission cases PASS; all383/454sourcepinsstable; independentBatch finalreviewNO BLOCKER. NewreleaseCLIworker/HTTP/server frozen. Composedstrict +bothapprovedreleasefullV/all42routes +fullisolated1251Rust/18Python/release/installer/Nix PASS. Fresh14worker/12HTTPandactualbeforeafterperformance PENDING. DefaultCPUgateoverwrote targetreleaseCLI; restoreOWNnewfrozen7cf44server beforefrozenHTTP tests.

````

## Source: .agents/runs/serving-and-speculation-row-state-transport-20261020/frozen-prepared/API.md

Original UTF-8 bytes: 4355; SHA256 `af7402927625b53580702caaeb6bf460cf09376fc838ce907e773123ccbf3f13`.
Resolve embedded relative paths against the original path, not this volume.

````text
SOURCE ONLY — row-native U32 state transport. Owner baseline_review.

- NEW generic primitive `resident_state_transport::{Shape,Direction,Row,
  PreparedTransport}`. No numerical profile, SourceIdentity issuer, session,
  StateRoot, allocation, upload, event, Ready or publication capability.
- Original source/generation authority stays in `ResidentModel`:
  `prepare_fused_execution` validates every RootBinding/token/source, exact
  per-session lease/service/generation and independent domains, then collects ALL
  original physical accesses and submits through the original queue/service.
  Raw Handle equality is not logical source/generation authorization. New
  primitive validates physical service, checked visible ranges and aliases only.
- `geometry(Shape,B)` admits EXACT B2/4/8,1..1M U32 words/row, <=8M total.
  KV slab>0,capacity1..512 with checked products; no old1M aggregate helper is
  reused. Actual8×262144=2097152 words fits. All byte arithmetic is checked.
- `ensure_capability` requires exact selected CUDA service and actual binding,
  cube/thread/grid capacity. Vector ABI9 buffers, KV18; no hidden dispatch split,
  retry, implicit device choice or backend fallback. Non-CUDA B1 unchanged.
- `prepare(client,rows,denseK,denseV,shape,direction)` checks every active private
  and dense buffer before binding/range helpers: same original physical ServiceId,
  checked allocation minus end trim then start,4B alignment, sufficient real span;
  rejects duplicate/overlapping visible ranges even for read-only private rows.
  Clones original handles only; no GPU buffer/source cache or copied values.
- Four kernels, one actual launch/domain/direction: vector gather/scatter
  `[B,N]`; KV gather/scatter `[T,B,H,D]`, same exact U32 mapping as original.
  Separate gather uses immutable private inputs; scatter uses immutable dense
  inputs. Exact raw signed-zero/subnormal/nonfinite payloads; no float conversion,
  canonicalization, math, finite-certificate removal or status-policy change.
- Fixed ABI inactive argument slots reuse row0 but are unreachable at actual
  checked B. All logical rows are real independent sessions; no fake model rows,
  padded tokens, duplicated active state or publication change.
- BOTH directions/ALL domains are preflighted inside existing
  `BatchProgram::validate_runtime`, before original `submit_cohort`. Capability
  checks also run before new batch workspace allocations. Original packet,
  candidate, staging, state, output and scratch admission/physical budget unchanged.
- Native and qualifier-only Graph share the SAME outside-body IO methods.
  `launch_inputs` now returns Result; Graph propagates it. Captured body nodes,
  kernel count, source math and graph completion owner are unchanged. Every
  original private/dense handle still appears in execution/physical flight
  accesses; unknown completion retains the same whole attempt/graph owner.
- Original per-row kernels are retained only for test+qualification+CUDA oracle;
  no runtime fallback. No resident_program.rs/public helper re-export is needed:
  physical tests mount as children of batch's oracle module. Only new module
  registration in lib.rs plus narrow batch.rs/graph.rs wiring.

Expected SOURCE launch sites for approved42 domains (not a timing claim):

| Operation | B2 | B4 | B8 |
| --- | --- | --- | --- |
| Original gather+scatter |168|336|672|
| Proposed native gather+scatter |84|84|84|
| Source count difference |84|252|588|

Existing per-candidate finite certificates remain separate and unchanged.
Projection/sampler/body/graph-enqueue counters are not relabeled. Total forward
kernel receipt adds the actual new per-domain launch returns once. B1 untouched.

ROOT integration:

1. Review `NEW_DELTA.patch` and `ROOT_WIRING.patch`. Baseline pins in BASE.json.
   Apply NEW_DELTA to current source; root's unmerged GDN change only appends
   typed cold-refusal matches, so compose the narrow batch hunks. Never copy the
   whole proposed batch/lib over concurrent work. Copy new files with freshmtime.
2. ROOT_WIRING adds only `mod resident_state_transport;` to current lib.rs;
   proposed full lib is a review aid, not an overwrite instruction.
3. Static formatting passed; no author compiler, test, model, provider, operator
   or GPU/runtime invocation. No source/profile eligibility or speed claim.

````

## Source: .agents/runs/serving-and-speculation-row-state-transport-20261020/frozen-prepared/ROOT-QUALIFICATION.md

Original UTF-8 bytes: 4361; SHA256 `26a170bc6159fe215b6705115f8deed87ad9a7e5d78d38700568c3dc329c6551`.
Resolve embedded relative paths against the original path, not this volume.

````text
ROOT ONLY — prepared physical gates, all UNEXECUTED by author.

Source-free and strict gates:

- Focus new `resident_state_transport::tests`: real2097152-word geometry, exact
  8M bound, refused overflow/invalidB/capacity, checked end/start/byte underflow and
  valid nonzerooffset views. Run relevant strict CUDA/default/Vulkan Clippy,
  approved model/cache release routes and full isolated `scripts/check.sh` as
  required by the owning implementation phase. No compilation establishes GPU
  support or graph/serving acceptance.

Actual copy-only qualifier:

- Fresh RELEASE CUDA library test binary with qualification feature. Run each
  ignored test alone under original isolation/coordinator:
  `resident_program::batch::state_transport_qualification::physical::original_qwen35_08b_native_state_word_transport`
  and corresponding `original_qwen35_2b_native_state_word_transport`.
- New pinned ORIGINAL ResidentService client/queue; original source identity;
  256MiB finite provider,5GiB device cap; physical ResourceArbiter reservations,
  initialization, actual Ready/readback and final reserved0/no fence/events.
- B2/4/8; vector13/18432/262144 and synthetic KV5×3/real512×512; complete payload
  plus prefix/suffix canaries on EVERY original private input, old/new packed and
  old/new scattered row/channel. All views have real nonzero offsets. Old physical
  vector/KV row kernels are independent oracles; synthetic tagged host words also
  require correct original row mapping. Includes both zero signs/subnormals,
  infinity/NaN payload transport; no numerical inference claim for those words.
- Actual foreign physical service metadata, duplicate/overlapping active rows,
  adequately sized dense/private alias, invalid end/start and short payloads
  refuse before launch without upload/lease/event change. Original source/epoch/
  root authorization is a separate preserved owner gate, not proved by Handle
  metadata. Original per-row dispatch count2B versus actual native2 for one
  gather+scatter fixture; no measured throughput/driver enqueue claim.

Mandatory complete model/owner matrix AFTER copy-only physical proof:

- Freshly compiled current source `.8B/2B × B2/4/8` accepted `resident-fused-cuda`
  full-logit and `resident-fused-greedy-cuda` feedback routes; independent accepted
  CPU+B1 full248320/all42 at every accepted step. Preserve negative rows, root/
  epoch/packet/snapshot ownership, abandoned preparation and physical lease return.
- Native teacher-prefix library tests for both sources at B2/4/8: actual StateOnly
  each teacher/all42, trained EOS suppressed, no generated packet/output; future
  greedy/fullV matches; ragged frontiers+256/257/512 boundary, tail readout barrier.
- Both-source pressure fallback tests and actual `hardware-abort-probe` StateOnly,
  full-logit and incoming/outgoing greedy abort routes. New module allocates no
  GPU buffers; prior original handles/readiness/quarantine and retained provider
  envelope must stay exact. No recovery after unknown completion.
- Re-run existing `resident-cuda-graph-wide-cuda` B2/4/8 for both sources:
  StateOnly/Logits/Greedy-outside body, independent CPU+B1/native/all42/fullV,
  stale/foreign/restore, abandon, pressure and actual graph abort. New copies are
  outside capture; old capture source/binary receipts do not qualify new IO.
  Production/global graph admission remains CLOSED.
- Fresh actual worker and HTTP B2/4/8+unequalarrival+prefix cases, credit parking,
  Ready cancellation/survivor retry, no duplicate source IDs/SSE/usage/Done, every
  row's exact final state and final physical0. Existing health conservation must
  use actual Native/StateOnly/Greedy/width and total kernel counts.
- Only after those pass, preserve the preceding accepted server as a real pinned
  baseline, then same-source/caps/temperature/tokenizer warmed ABBA/BAAB naturalc8
  AND B1 with canonical tools/owned lifecycle. Keep offered/completed/IDs/usage,
  TTFT/actual individual-ID ITL/fairness/SLO/memory/source-bound counter receipts.
  Previous `.8B/2B` warm timings predate this code. No speed/vLLM claim until
  independently measured; genuine matched vLLM acceptance is still separate.

Use existing accepted qualifiers/supervisors, not a second model/token loop.
No new tracker, model acquisition, driver change or shared plan edit is proposed.

````

## Source: .agents/runs/serving-and-speculation-row-state-transport-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 791; SHA256 `403e592dbdd011103dbd11840a5644f60576d44c9afd2934b66d0dadac2d61bf`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent serving-and-speculation. Owner baseline_review. SOURCE ONLY.

- [x] Read existing whole Native/Graph gather/scatter and physical owner paths.
- [x] Prepare bounded row-native U32 copies, prelaunch physical range/capability
  checks and minimal shared outside-body IO wiring; source math unchanged.
- [x] Prepare original per-row physical differential, opaque-bit/canary/refusal
  fixtures and ROOT complete model/graph/worker/HTTP/pressure/abort gate matrix.
- [x] Source formatting only; no author build, tests, operator/model/provider/GPU.
- [ ] ROOT+peer full source review, composition, strict build and focused tests.
- [ ] ROOT actual physical copy and complete model/owner matrix above; final0.
- [ ] Actual before/after measurement; no inferred speed/profile/graph admission.

````

## Source: .agents/runs/serving-and-speculation-solo-peer-review-20261020/REVIEW.md

Original UTF-8 bytes: 3487; SHA256 `18ef19bffd544ed8e7e5a3a9da76a27c6d2dd0507b060fa310161612e913183e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# SOLO B1/T2 independent source review

Parent: serving-and-speculation. Owner: baseline_review.

Reviewed complete frozen `4f2a9d5f` production470-line lowering,375-line owner,598-line qualifier, all parent factor/wiring hunks, excluded cc3/6be dependency/state/flight contexts and new README/gates. Quiet source IO only: no imports/tests/builds/operators/models/providers/GPU or payload hashes. Seven proposed/frozen source copies byte-equal. Original `launch_node` through EOF byte-equal after factor; `ForwardPlan::state_nodes()` is the original contiguous prefix, so helper index lookup preserves ordinary ordered launch selection.

Finding + repair:

- Original `ROOT_WIRING.patch` malformed: unprefixed `mod solo;`, hunk3→4 inconsistent with intended two additions. ROOT/author notified; original immutable.
- Additive wiring `17726d0e` / `0dd48340` source-reviewed: exact parent EOF675..677 brace/blank/modtemporal, proper3→5 hunk adds blank and module. No Rust implementation change. No remaining source-review blocker with this wiring substitution.

Checked:

- Genuine original P1 causal emitter +original P2 stateless lowering. Same immutable plan/source-owner lineage; no BatchProgram1, fake session/root or copied model/token schedule. Full KV guard→rope→append→QK→softmax→AV remains teacher-ordered. Private middle uses actual P1 dense KV layout.
- Original state versions map committed→middle→final exactly per domain. Full geometry/source/service/ranges/producer-consumer equality and candidate-versus-prior/workspace/sibling nonaliasing before original math flight. P1 copy/cert word bounds<=1M explicitly closed; P2 retains accepted bounds/quant bindings.
- Complete P2 owned budget, unused allocations, all42 middle and fresh final bytes checked before allocations/ticket. Actual handle/byte reconciliation; original cold begun slot/initialization/sync/requeue. One allocation/workspace lock ordering; fair Scratch request covers both state banks, no fallback. Graph-pinned batch refuses.
- Every used original P1/P2/source/status/scratch/middle/final lease participates in original flight before enqueue. Middle Write access intentionally covers producer then consumer in same flight; final-only promotion. Unknown launch/drop retains original flight/quarantine/fence; no source/queue recovery path introduced.
- P1/P2 scalar statuses clear once. Original node certificates and immediate full state certificates preserve T0 decisions before reuse; final OR and full final-state certificates feed one actual4B terminal readback/Ready. Original authority begin_advance2/commit publishes one root; no token/EOS presentation.
- Qualifier actual per-teacher CPU/B1 fullV+all42, separately observed private T0 all42/canonical KV tail, final committed all42/future fullV+real greedy packets; source EOS only teacher. Invalid metadata, abandon/stale publication, genuine one-byte pressure,255/510 boundaries, actual after-launch cancellation, persistent inactive-middle-NaN certificate and final zero-padding check are prepared. Diagnostic reads hold original leases/transfer ticket and manufacture no Root/Ready.

Limits: source verdict only. cc3/6be/SOLO strict/all-cfg and actual both-model numerical/physical gates remain mandatory. No serving integration, temporal throughput, CUDA graph/global capture barrier or vLLM equivalence follows from this review. Hardware launch counts require a collector; source receipt counts remain source launch sites.

````

## Source: .agents/runs/serving-and-speculation-solo-peer-review-20261020/TODO.md

Original UTF-8 bytes: 550; SHA256 `461f07d3ba2ef2601ea12a0d1b2995cba48d157fe2e0d12d491031aabe5d5416`.
Resolve embedded relative paths against the original path, not this volume.

````text
# SOLO source review

Parent serving-and-speculation; owner baseline_review. Read-only scope.

- [x] Complete production/owner/qualifier/factor/admission/lineage/cert/Ready/quarantine source review.
- [x] Report malformed original wiring; independently review additive17726d0e repair.
- [x] No remaining source-review blocker with repaired wiring.
- [ ] ROOT compilation +both-model actual numerical/physical/ordinary-regression gates; no acceptance inferred here.

No MAIN/shared plans/author files/frozen bytes changed; no execution/payload scans.

````

## Source: .agents/runs/serving-and-speculation-spine-review/TODO.md

Original UTF-8 bytes: 1369; SHA256 `4df40ea42f9646294c58fb4f8a96e742caddebae1dfce234152262c0a8a59453`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12 shared completion / independent Ready rows

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Owner: `/root/spine_review`
Scope: `crates/zerogravity-gpu/src/resident_leases.rs` only.

- [x] Read complete canonical whitepaper, engineering specification, execution
  spine, task/compiler, invariants and state transaction contracts.
- [x] Review resident event retirement, unknown-completion quarantine and
  existing atomic cohort root publication.
- [x] Bind exact independent RootBindings before one physical submission.
- [x] Split one trusted terminal completion into original row Ready proofs;
  retain disjoint output leases and prohibit B1/cohort completion confusion.
- [x] Add source-free negative validation coverage (3 tests; not yet executed).
- [x] Standalone rustfmt and `git diff --check` on owned source file.
- [ ] Coordinator: focused tests, strict Clippy, full isolated gate and physical
  source-real CUDA B2/B4/B8 qualification. No Cargo run by this agent.

Results/failures: implementation and source review complete. B2/B4/B8 exact
bound roots, unique sessions/source/frontier and disjoint output promotion;
one physical terminal event, original per-row roots/leases. No Cargo run or
physical inference qualification performed by this agent. Main integration
must compare full returned row binding and exact candidate-domain coverage.

````

## Source: .agents/runs/serving-and-speculation-state-scatter-certification-20261020/README.md

Original UTF-8 bytes: 8587; SHA256 `3c63529ea07d5ad7978935c94b551184ef2ff4eec3663cff488977700e5d0220`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Typed state scatter + original finite certificate — source-only candidate

Parent: serving-and-speculation. Owner: batch_design_review. ROOT alone merges,
builds, runs tests/operators/models/GPU and reconciles the shared plan.

## Narrow candidate

The original native batch currently scatters42 state domains, then dispatches
`certify_finite_strided` separately for each row/physical candidate buffer.
Accepted Qwen0.8B/2B contain36 vector domains and6 two-buffer KV domains:48
candidate buffers per row, each within the original1,000,000-word certificate
bound. The proposed native-only scatter reads the exact U32 stored to each
candidate and ORs the original NONFINITE_VALUE bit for the original exponent
predicate. No FP instruction, cast, reduction, data canonicalization, new status,
new allocation, learned source, state authority or queue is introduced.

- `resident_state_transport/certified.rs`: checked typed wrapper around original
  PreparedTransport. Vector scatter1 launch; KV K/V scatter2 launches. Each
  launch has8 row handles +1 packed field +1 original status =10 bindings.
  Existing KV gather/copy remains18 bindings; never require19.
- Existing opaque `PreparedTransport::{prepare,launch}`/copy kernels and their
  original nonfinite-payload bit-transport qualifier remain unchanged. Opaque
  transport intentionally does NOT certify. Only native typed final-state
  scatter selects the new wrapper.
- Full shape/span/service/independent-buffer checks reuse original preflight.
  Additional original status word must have a valid4B-aligned visible span,
  same service and no overlapping visible range with ANY private/packed field.
  All42 gather/scatter/status bindings are checked by `validate_runtime` before
  original `slot.begin_submission`/`submit_cohort`. No launch in preflight.
- `BatchProgram::launch_outputs` retains the same complete proposal order but
  removes now-redundant separate candidate certificates. Node-body finite
  certificates and quantization/embedding status merges remain unchanged.
  Existing status is cleared only by original step initialization, once.
  Prior error bits are preserved by atomic OR; bothInf/NaN signs/payloads fail
  exactly like original certificate; signed zeros/subnormals stay raw U32.
- Inactive fixed-ABI handles reuse row0 exactly as accepted opaque transport.
  Checked rows2/4/8 and compile-time row bounds make those aliases unreachable.
- Original execution handles already retain status and every packed/private
  state lease in one original flight. There is no new allocation or lease.
  Failure between K/V launches leaves that same flight armed: original pending
  event/fence/quarantine retain ALL bindings and envelope; no root is published.
  Known-Ready cancellation still discards the private cohort and regroups the
  original roots/packets. Atomic all-row publisher remains unchanged.
- Graph math body remains cfg(qualification,cuda)/diagnostic. Typed scatter is
  outside capture in original launch_outputs; Result propagation is unchanged.
  No production capture, stream/client pin, graph cache or bypass is enabled.

Source launch-site estimate per native teacher, versus accepted row-native
copy +separate certificates: save `48*B-6` =90/186/378 atB2/B4/B8. KV uses one
extra copy launch/domain to stay within ABI. These are SOURCE estimates, not
actual driver counts or measured speed. ROOT's original official CUPTI native
control reports64,328 real kernels/64,328 direct API calls, zero loss, and many
finite certificates; its older binary is distinct from this unqualified change.
No57MiB trace was read; only small authenticated validation metadata inspected.

## ROOT physical gates

New nested qualifier compares SAME physical input/gathered buffers against
original per-row copy AND original `certify_finite_strided` (never a rewritten
CPU expected certificate). Synthetic scratch words are not trained outputs.
Both approved source identities create the original ResidentService; all
payloads/status/canaries use its queue, leases, initialization and real Ready.

Prepared cases B2/4/8 × vector13/18432/262144 and KV(5,3)/(512,512), every row and
KV field nonfinite in turn, +0/-0/subnormal/maxfinite/Inf/qNaN/sNaN rawbits, prior
status5, complete prefix/suffix/status canaries, no warm uploads, cold
foreign/status-alias/short/misaligned/duplicate-row refusal before flight,
original per-site dispatch receipts and final0. No extra source/profile support.

Two finite-status rejection tests use actual original readback, require both
old/new nonfinite status/canaries, refuse Ready under the original all-zero
policy, and require conservative original event/handle/envelope quarantine.
Two hardware-abort-feature tests enqueue actual original commands, then omit
ANY terminal readback: require pending event, fenced service, quarantined handles
and retained original envelope. ONE fresh process per negative case; no sync,
recovery, successful publication, released envelope or numerical acceptance is
claimed after those failures.

ROOT composes only NEW_DELTA.patch after independent review, with fresh mtimes.
No dependency/lock/Cargo changes. Suggested exact release filter namespace:

`resident_program::batch::state_transport_qualification::certified::`

- `original_qwen35_{08b,2b}_certified_state_scatter` (known complete/final0).
- `original_qwen35_{08b,2b}_certified_state_scatter_status_refusal` (fresh process).
- `original_qwen35_{08b,2b}_certified_state_scatter_abandon` with
  hardware-abort-probe (fresh process).

Do not select all tests by an ambiguous shared substring in one process.
ROOT must also execute the UNCHANGED opaque transport qualifier, B2/4/8 genuine
full-model logits/greedy/all42 CPU+B1 comparisons, mixed256→257/refill,
StateOnly teacher prefixes/futurefullV, stale/cancel/admission/unknown matrix,
worker14/HTTP12 and strict/full isolated check plus both approved release routes.
Then separately collect actual CUPTI kernel counts and unprofiled repeated A/B
latency on exact source/workload/binary closures. No speed/profile acceptance
follows from source inspection, formatter or synthetic predicate tests.

## Future token-major and graph dependencies

Current native serving consumes one teacher per member: gpu_engine::prefill_count
returns1; cohort source partition preserves teacher-vs-tail barrier; fused
preparation uses each original authority.begin_advance(1), all42 candidates,
real readback, then atomic all-row commit. B1 already begins_advance(T), but its
prepare_mode loops original execute/readback for each teacher with private
ping-pong and one final publication. Changing wave token counts alone does not
create token-major native math or remove per-teacher GPU dispatch.

A future exact AR1-order multi-teacher slice could begin_advance(T) independently
forB2/4/8, admit all private ping-pong/workspace/tokens before launch, gather once,
run the SAME original plan sequentiallyT times with U32 state/position advance,
and scatter once. Each intermediate must be certified before reuse; status must
accumulate across the entire chunk. No teacher-tail logits/sampler/SSE or EOS
retirement before the exact final readout barrier. Per-row roots gainT only at
one original atomic commit. BoundT/physical work quantum and preserve fair queue
service; split unequal remaining lengths without padding/stopped rows. This is
an additional physical transaction/math schedule, currently UNSUPPORTED; it
cannot be inferred from B1 chunk publication or present token-major tensor shapes.
Initial GDN bulk chunk math must not replace the accepted sequential arithmetic.

Production CUDA graph reuse is still blocked on actual same-card capture
exclusion across all original Clients, allocation/H2D/sync/drop/deallocation and
actor command admission, not just fair active model slots. Current graph source
requires caller cold isolation, fixed B2/4/8/cap512/AVpad256, original cold pin,
matching generation/body/readout and known terminal owner retirement. It excludes
device-token graph feedback. A graph cache must charge executable/driver storage
and keep all original workspace/stream/source/generation owners through every
Ready/unknown path; per-band pinned workspaces currently prevent replacement.
The pure exclusion/retirement proposals are not production issuers. Do not
remove cfg gates or treat one same-card worker as proof of global capture safety.
Once those contracts are actually qualified, graphs may collapse host enqueues
while retaining MANY actual kernel executions; do not report replay1 as kernel1.

````

## Source: .agents/runs/serving-and-speculation-state-scatter-certification-20261020/TODO.md

Original UTF-8 bytes: 1352; SHA256 `9fcca06587e9479aee38c36d5e164c961d611c56effb2f3bf6e3a49816c267b6`.
Resolve embedded relative paths against the original path, not this volume.

````text
# State scatter certification source proposal

Parent: `.agents/plans/serving-and-speculation/TODO.md`.
Owner: batch_design_review. Only this ignored run; no MAIN/shared-plan edits.

- [x] Read current native/graph/worker paths and small ROOT CUPTI validation.
- [x] Preserve original opaque U32 kernels/qualifier and source arithmetic.
- [x] Prepare checked10-binding typed field scatter and original status OR.
- [x] Prepare narrow native all42 preflight/output wiring; original owners intact.
- [x] Prepare real original-copy+certificate differential/status/unknown cases.
- [x] Record source-only token-major/capture contracts and still-closed gates.
- [ ] Independent ROOT/peer review; no claim before complete source review.
- [ ] ROOT strict/unit/default/Vulkan/full isolated check +both release routes.
- [ ] ROOT two-source operator/canary/status/unknown physical qualification.
- [ ] ROOT genuine full-model/all42/mixed/frontier/prefix/worker/HTTP matrix.
- [ ] ROOT exact-binary CUPTI counts and separate unprofiled A/B timing.

Author executed only source copying/formatting/metadata hashing. No tests,
Cargo/compiler/operator/model/provider/CUDA/GPU/container runtime. First rustfmt
attempt refused missing unowned child modules; rerun with skip_children=true
formatted only the five proposal paths. No source or numerical failure hidden.

````

## Source: .agents/runs/serving-and-speculation-state-scatter-certification-20261020/frozen/README.md

Original UTF-8 bytes: 8587; SHA256 `3c63529ea07d5ad7978935c94b551184ef2ff4eec3663cff488977700e5d0220`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Typed state scatter + original finite certificate — source-only candidate

Parent: serving-and-speculation. Owner: batch_design_review. ROOT alone merges,
builds, runs tests/operators/models/GPU and reconciles the shared plan.

## Narrow candidate

The original native batch currently scatters42 state domains, then dispatches
`certify_finite_strided` separately for each row/physical candidate buffer.
Accepted Qwen0.8B/2B contain36 vector domains and6 two-buffer KV domains:48
candidate buffers per row, each within the original1,000,000-word certificate
bound. The proposed native-only scatter reads the exact U32 stored to each
candidate and ORs the original NONFINITE_VALUE bit for the original exponent
predicate. No FP instruction, cast, reduction, data canonicalization, new status,
new allocation, learned source, state authority or queue is introduced.

- `resident_state_transport/certified.rs`: checked typed wrapper around original
  PreparedTransport. Vector scatter1 launch; KV K/V scatter2 launches. Each
  launch has8 row handles +1 packed field +1 original status =10 bindings.
  Existing KV gather/copy remains18 bindings; never require19.
- Existing opaque `PreparedTransport::{prepare,launch}`/copy kernels and their
  original nonfinite-payload bit-transport qualifier remain unchanged. Opaque
  transport intentionally does NOT certify. Only native typed final-state
  scatter selects the new wrapper.
- Full shape/span/service/independent-buffer checks reuse original preflight.
  Additional original status word must have a valid4B-aligned visible span,
  same service and no overlapping visible range with ANY private/packed field.
  All42 gather/scatter/status bindings are checked by `validate_runtime` before
  original `slot.begin_submission`/`submit_cohort`. No launch in preflight.
- `BatchProgram::launch_outputs` retains the same complete proposal order but
  removes now-redundant separate candidate certificates. Node-body finite
  certificates and quantization/embedding status merges remain unchanged.
  Existing status is cleared only by original step initialization, once.
  Prior error bits are preserved by atomic OR; bothInf/NaN signs/payloads fail
  exactly like original certificate; signed zeros/subnormals stay raw U32.
- Inactive fixed-ABI handles reuse row0 exactly as accepted opaque transport.
  Checked rows2/4/8 and compile-time row bounds make those aliases unreachable.
- Original execution handles already retain status and every packed/private
  state lease in one original flight. There is no new allocation or lease.
  Failure between K/V launches leaves that same flight armed: original pending
  event/fence/quarantine retain ALL bindings and envelope; no root is published.
  Known-Ready cancellation still discards the private cohort and regroups the
  original roots/packets. Atomic all-row publisher remains unchanged.
- Graph math body remains cfg(qualification,cuda)/diagnostic. Typed scatter is
  outside capture in original launch_outputs; Result propagation is unchanged.
  No production capture, stream/client pin, graph cache or bypass is enabled.

Source launch-site estimate per native teacher, versus accepted row-native
copy +separate certificates: save `48*B-6` =90/186/378 atB2/B4/B8. KV uses one
extra copy launch/domain to stay within ABI. These are SOURCE estimates, not
actual driver counts or measured speed. ROOT's original official CUPTI native
control reports64,328 real kernels/64,328 direct API calls, zero loss, and many
finite certificates; its older binary is distinct from this unqualified change.
No57MiB trace was read; only small authenticated validation metadata inspected.

## ROOT physical gates

New nested qualifier compares SAME physical input/gathered buffers against
original per-row copy AND original `certify_finite_strided` (never a rewritten
CPU expected certificate). Synthetic scratch words are not trained outputs.
Both approved source identities create the original ResidentService; all
payloads/status/canaries use its queue, leases, initialization and real Ready.

Prepared cases B2/4/8 × vector13/18432/262144 and KV(5,3)/(512,512), every row and
KV field nonfinite in turn, +0/-0/subnormal/maxfinite/Inf/qNaN/sNaN rawbits, prior
status5, complete prefix/suffix/status canaries, no warm uploads, cold
foreign/status-alias/short/misaligned/duplicate-row refusal before flight,
original per-site dispatch receipts and final0. No extra source/profile support.

Two finite-status rejection tests use actual original readback, require both
old/new nonfinite status/canaries, refuse Ready under the original all-zero
policy, and require conservative original event/handle/envelope quarantine.
Two hardware-abort-feature tests enqueue actual original commands, then omit
ANY terminal readback: require pending event, fenced service, quarantined handles
and retained original envelope. ONE fresh process per negative case; no sync,
recovery, successful publication, released envelope or numerical acceptance is
claimed after those failures.

ROOT composes only NEW_DELTA.patch after independent review, with fresh mtimes.
No dependency/lock/Cargo changes. Suggested exact release filter namespace:

`resident_program::batch::state_transport_qualification::certified::`

- `original_qwen35_{08b,2b}_certified_state_scatter` (known complete/final0).
- `original_qwen35_{08b,2b}_certified_state_scatter_status_refusal` (fresh process).
- `original_qwen35_{08b,2b}_certified_state_scatter_abandon` with
  hardware-abort-probe (fresh process).

Do not select all tests by an ambiguous shared substring in one process.
ROOT must also execute the UNCHANGED opaque transport qualifier, B2/4/8 genuine
full-model logits/greedy/all42 CPU+B1 comparisons, mixed256→257/refill,
StateOnly teacher prefixes/futurefullV, stale/cancel/admission/unknown matrix,
worker14/HTTP12 and strict/full isolated check plus both approved release routes.
Then separately collect actual CUPTI kernel counts and unprofiled repeated A/B
latency on exact source/workload/binary closures. No speed/profile acceptance
follows from source inspection, formatter or synthetic predicate tests.

## Future token-major and graph dependencies

Current native serving consumes one teacher per member: gpu_engine::prefill_count
returns1; cohort source partition preserves teacher-vs-tail barrier; fused
preparation uses each original authority.begin_advance(1), all42 candidates,
real readback, then atomic all-row commit. B1 already begins_advance(T), but its
prepare_mode loops original execute/readback for each teacher with private
ping-pong and one final publication. Changing wave token counts alone does not
create token-major native math or remove per-teacher GPU dispatch.

A future exact AR1-order multi-teacher slice could begin_advance(T) independently
forB2/4/8, admit all private ping-pong/workspace/tokens before launch, gather once,
run the SAME original plan sequentiallyT times with U32 state/position advance,
and scatter once. Each intermediate must be certified before reuse; status must
accumulate across the entire chunk. No teacher-tail logits/sampler/SSE or EOS
retirement before the exact final readout barrier. Per-row roots gainT only at
one original atomic commit. BoundT/physical work quantum and preserve fair queue
service; split unequal remaining lengths without padding/stopped rows. This is
an additional physical transaction/math schedule, currently UNSUPPORTED; it
cannot be inferred from B1 chunk publication or present token-major tensor shapes.
Initial GDN bulk chunk math must not replace the accepted sequential arithmetic.

Production CUDA graph reuse is still blocked on actual same-card capture
exclusion across all original Clients, allocation/H2D/sync/drop/deallocation and
actor command admission, not just fair active model slots. Current graph source
requires caller cold isolation, fixed B2/4/8/cap512/AVpad256, original cold pin,
matching generation/body/readout and known terminal owner retirement. It excludes
device-token graph feedback. A graph cache must charge executable/driver storage
and keep all original workspace/stream/source/generation owners through every
Ready/unknown path; per-band pinned workspaces currently prevent replacement.
The pure exclusion/retirement proposals are not production issuers. Do not
remove cfg gates or treat one same-card worker as proof of global capture safety.
Once those contracts are actually qualified, graphs may collapse host enqueues
while retaining MANY actual kernel executions; do not report replay1 as kernel1.

````

## Source: .agents/runs/serving-and-speculation-state-scatter-certification-20261020/frozen/TODO.md

Original UTF-8 bytes: 1352; SHA256 `9fcca06587e9479aee38c36d5e164c961d611c56effb2f3bf6e3a49816c267b6`.
Resolve embedded relative paths against the original path, not this volume.

````text
# State scatter certification source proposal

Parent: `.agents/plans/serving-and-speculation/TODO.md`.
Owner: batch_design_review. Only this ignored run; no MAIN/shared-plan edits.

- [x] Read current native/graph/worker paths and small ROOT CUPTI validation.
- [x] Preserve original opaque U32 kernels/qualifier and source arithmetic.
- [x] Prepare checked10-binding typed field scatter and original status OR.
- [x] Prepare narrow native all42 preflight/output wiring; original owners intact.
- [x] Prepare real original-copy+certificate differential/status/unknown cases.
- [x] Record source-only token-major/capture contracts and still-closed gates.
- [ ] Independent ROOT/peer review; no claim before complete source review.
- [ ] ROOT strict/unit/default/Vulkan/full isolated check +both release routes.
- [ ] ROOT two-source operator/canary/status/unknown physical qualification.
- [ ] ROOT genuine full-model/all42/mixed/frontier/prefix/worker/HTTP matrix.
- [ ] ROOT exact-binary CUPTI counts and separate unprofiled A/B timing.

Author executed only source copying/formatting/metadata hashing. No tests,
Cargo/compiler/operator/model/provider/CUDA/GPU/container runtime. First rustfmt
attempt refused missing unowned child modules; rerun with skip_children=true
formatted only the five proposal paths. No source or numerical failure hidden.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-audit-spine-20261020/TODO.md

Original UTF-8 bytes: 1102; SHA256 `52aef8a9ee4da67a1deeb77f64dc0096ab1ccf3f46205809545d5d60f22c2e02`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Stock official GGUF plugin audit

Parent: serving-and-speculation. Owner: /root/spine_review.
SOURCE/HEADER METADATA ONLY; no imports/build/operator/model/container/GPU runtime.

- [x] Browse official plugin/docs/PyPI metadata; use ROOT-acquired pinned151
  source files e2b8ad532b8b5ea175100202c30430c1d2b5e6a8.
- [x] Inspect loader/model mapping, head layout/parameter transforms, kernels,
  pinned0.31 API/requirements, legitimate original-mmproj language-only path.
- [x] Review ROOT CPU-only stock wheel-build scripts; no source blocker.
- [ ] Finish exact approved-header/role closure + codec/cast/reduction audit.
- [ ] Deliver exact ROOT-only runtime/benchmark acceptance gates and boundaries.
- [ ] Actual install/load/full logits/state/wire/performance gates ROOT-only.

Correction preserved: an early assertion that current ZG used unquantized F32
activations was WRONG and withdrawn. Current source uses Q8_0/Q8_1/Q8_K by weight
codec. Uniform plugin CUDA Q8_1 remains a different explicit execution profile.

Pre-begin retirement continuation preserved separately, design-only/no edits.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-baseline-20261020/TODO.md

Original UTF-8 bytes: 495; SHA256 `5bb49029552abeacfa4d3405e76ce0e0020673f51ce4f4d580f33fe222ef3ae5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# vLLM GGUF matched baseline

Parent: serving-and-speculation. Owner: /root.

- [x] Pin and acquire official plugin source; no runtime qualification.
- [ ] Independently audit same GGUF codecs/weights/mapping and pinned environment.
- [ ] Build immutable baseline plugin with accepted CUDA/Torch and owned lifetime.
- [ ] Prove genuine model loading/source/profile/tokenizer before measured runs.
- [ ] Use existing serve-bench/serve-compare with equivalent conditions and all required metrics.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-baseline-20261020/source/README.md

Original UTF-8 bytes: 3059; SHA256 `0394d081e2ad49976490de84747c02f620a2a99dd92791f8e3e4bd8873edcc2f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# vLLM GGUF Quantization Plugin

This plugin provides out-of-tree GGUF quantization support for vLLM after
in-tree support deprecation
([vllm-project/vllm#39583](https://github.com/vllm-project/vllm/issues/39583)).

## Installation

### Prerequisites

- CUDA toolkit or ROCm toolkit

We recommend [uv](https://docs.astral.sh/uv/) for package management. If you
don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From Source

1. Clone this repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-gguf-plugin
   cd vllm-gguf-plugin
   ```

2. If vLLM is not already installed, install it first:

   ```bash
   uv pip install vllm --torch-backend=auto
   ```

3. Build and install the plugin against the PyTorch installation used by
   vLLM:

   ```bash
   uv pip install -e . --no-build-isolation
   ```

   Disabling build isolation ensures that the CUDA extension is compiled
   against the same PyTorch installation used by vLLM at runtime.

## Development

After completing the editable source installation above, install and run the
development tooling:

```bash
uv pip install -e .[dev] --torch-backend=auto
pre-commit install
pre-commit run --all-files
```

The same hooks also run in GitHub Actions on every push and pull request.

## Usage

```bash
vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 --tokenizer Qwen/Qwen3-0.6B
```

Qwen 3.5 MTP speculative decoding loads the `nextn` block embedded in the same
GGUF; it does not download separate Hugging Face MTP weights:

```bash
vllm serve unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M \
  --tokenizer Qwen/Qwen3.5-4B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

For a GGUF without a `nextn` block, omit `--speculative-config`; the backbone
loads normally without MTP.

## Tested model coverage

The plugin uses vLLM's model implementations and a generic GGUF weight
adapter, so model compatibility is broader than a fixed allowlist. The models
below are covered by the repository's generation tests and are the best-known
starting points:

| Modality | Model family | Tested GGUF quantization |
| --- | --- | --- |
| Text | Qwen 2.5 | Q6_K |
| Text | Qwen 3 | Q8_0 |
| Text | Phi 3.5 | IQ4_XS |
| Text | GPT-2 | Q4_K_M |
| Text | StableLM | Q4_K_M |
| Text | Gemma 3 | Q4_0 |
| Text | OLMoE | Q4_0 |
| Vision-language | Gemma 3 | Q4_0 backbone with F16 projector |
| Vision-language | Gemma 4 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.5 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.6 | UD-IQ2_XXS backbone with BF16 projector |
| Image generation | Z-Image-Turbo | Q4_0 |
| Image generation | FLUX.2-klein | Q8_0 |

Other vLLM-supported architectures may work when their GGUF tensor names map
to the corresponding Hugging Face model. A model appearing in vLLM's general
supported-model list does not by itself guarantee GGUF compatibility. When
reporting an unsupported model, include the model repository, quantization,
plugin and vLLM versions, and the complete weight-mapping error.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/CONVERSION.md

Original UTF-8 bytes: 2661; SHA256 `fdae080bee5dd1b3784717b6e2abc7300373bd5653cb13338fec7a8d4508610a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Genuine projector preparation

- Parent: serving-and-speculation; author: spine_review.
- Source-only preparation. No conversion, container, model, Torch or GPU run by author.
- Exact existing llama.cpp0.6.0 conversion source; authentic full HF snapshots;
  original approved text GGUF mounted read-only and hashed before/after.
- Genuine public CLI: `convert_hf_to_gguf.py /source --mmproj --outtype bf16`.
  No patch, synthetic config, learned parameter substitution or LM reconversion.
- Converter accepts only original visual roles, splits the two temporal patch
  slices, forces patch tensors and some one-dimensional tensors to F32.
  Every emitted payload must equal original BF16 bits or their exact F32 widening.
  Torch-side audit plus independent stdlib GGUF/extent/source-byte audit.
- .8B:168 original visual roles→169 tensors; 2B:312→313. Both have320 text roles
  and24 text layers; visual depths are12/24. No text role may occur in projector.
- Read-only image/source/converter/backbone; exclusive output; networknone,
  no CUDA CDI/device; CUDA visibilitydisabled.12GiB memory,4CPU set inside
  before Torchimport,300s,1GiBperfile,1088MiB complete output tree,4MiB log.
- Exact reviewed `owned_container_v4.py`SHA53ee4270…; nonce/CID ownership,
  WNOWAIT, all signals before reap, independent cleanup stages, final group/
  container absence and EOF mandatory. No unavailable cpuset cgroup option.
- Successful conversion qualifies source-bound projector bytes only. Stock
  plugin loader, GPU math, all42 domains and performance remain unqualified.

ROOT, after reviewing the frozen manifest and six source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_conversion_fixtures.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_conversion.py \
  --model 0.8b --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-1 \
  --prepared-sha256 REVIEWED_CONVERSION_MANIFEST_SHA256
```

Repeat with `--model 2b` and a fresh `root-conversion-2b-1` output. Existing output
refuses; failed attempts remain immutable. Host validation rehashes authentic
source files and audits every projector payload, then proves final cleanup.

Stock loader followup must use this genuine projector with original full HF
config/tokenizer and `language_model_only=True`. It must prove actual320 loaded
text roles, zero learned vision parameters, stock CUDA plugin backend and
unchanged full-logit observations before any benchmark. Same GGUF source bytes
do not imply matching activation codec, parameter casts or execution arithmetic.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/CONVERSION_V2.md

Original UTF-8 bytes: 3114; SHA256 `3a48c37f209869c137a5f1c881442e061018152c31c3159028fbfb626dcff266`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Genuine projector preparation

- Parent: serving-and-speculation; author: spine_review.
- Source-only preparation. No conversion, container, model, Torch or GPU run by author.
- Exact existing llama.cpp0.6.0 conversion source; authentic full HF snapshots;
  original approved text GGUF mounted read-only and hashed before/after.
- Genuine public CLI: `convert_hf_to_gguf.py /source --mmproj --outtype bf16`.
  No patch, synthetic config, learned parameter substitution or LM reconversion.
- Converter accepts only original visual roles, splits the two temporal patch
  slices, forces patch tensors and some one-dimensional tensors to F32.
  Every emitted payload must equal original BF16 bits or their exact F32 widening.
  Torch-side audit plus independent stdlib GGUF/extent/source-byte audit.
- .8B:153 original visual roles→154 tensors; 2B:297→298. Both have320 text roles
  and24 text layers; visual depths are12/24. No text role may occur in projector.
- Read-only image/source/converter/backbone; exclusive output; networknone,
  no CUDA CDI/device; CUDA visibilitydisabled.12GiB memory,4CPU set inside
  before Torchimport,300s,1GiBperfile,1088MiB complete output tree,4MiB log.
- Exact reviewed `owned_container_v4.py`SHA53ee4270…; nonce/CID ownership,
  WNOWAIT, all signals before reap, independent cleanup stages, final group/
  container absence and EOF mandatory. No unavailable cpuset cgroup option.
- Successful conversion qualifies source-bound projector bytes only. Stock
  plugin loader, GPU math, all42 domains and performance remain unqualified.

ROOT, after reviewing the frozen manifest and eight source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_conversion_fixtures_v2.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_conversion_v2.py \
  --model 0.8b --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-1 \
  --prepared-sha256 REVIEWED_CONVERSION_MANIFEST_SHA256
```

Repeat with `--model 2b` and a fresh `root-conversion-2b-1` output. Existing output
refuses; failed attempts remain immutable. Host validation rehashes authentic
source files and audits every projector payload, then proves final cleanup.

Stock loader followup must use this genuine projector with original full HF
config/tokenizer and `language_model_only=True`. It must prove actual320 loaded
text roles, zero learned vision parameters, stock CUDA plugin backend and
unchanged full-logit observations before any benchmark. Same GGUF source bytes
do not imply matching activation codec, parameter casts or execution arithmetic.

V2 additive correction: V1 preflight refused before any container/conversion.
Both full snapshots also contain15 authentic BF16 MTP roles. `source-pins-v2.json`
pins their exact names/dtypes/shapes/file extents; source closure requires all15
and excludes them explicitly. The unchanged converter already filters `mtp.`.
V1 eight frozen files and its manifest remain byte-identical. No actual
conversion success is asserted by this preparation.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/LOADER.md

Original UTF-8 bytes: 5263; SHA256 `cfed78b82eaf9c90768d851bde2181a10de8367da822826f28e47ad02e747a69`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Stock GGUF language-only qualification preparation

Author: spine_review. Parent: serving-and-speculation. Source-only author work;
ROOT alone executes containers/operators/models/GPU. No MAIN changes.

- Stock official plugin `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`;
  ROOT stock wheel SHAaf9de736…, extension SHAebb896a5…; exact gguf0.19 wheel.
  Unmodified wheel members/RECORD extracted into ephemeral private site. No
  in-image install, import shim, numeric patch or alternate backend.
- Genuine pinned vLLM0.31.0+cu129/Torch2.13.0+cu129 in accepted imagef57d77….
  Selected original runtime source, all111 plugin Python files and native DSO
  pinned before/after. Actual identities and imported package path required.
- Original approved text GGUF unchanged; full original HF config/tokenizer;
  ROOT-verified genuine projectors bound by exact output and actual receipt hashes
  in `projector-pins.json`. Both authentic15-role MTP sets explicitly excluded;
  full HF source hash/metadata closure remains mandatory.
- Public `LLM` with `language_model_only=True` and actual
  `model_loader_extra_config={'mm_proj': genuine_projector}`. The stock vision
  guard remains. Require real `StageMissingLayer` and zero learned vision/MTP
  parameters; all320 text source roles must map to actual CUDA parameters.
- Same original public `WeightsMapper` metadata traversal; all packed shards,
  exact source geometry/codec bytes, padding, tied storage, auxiliary codec
  metadata, loader transforms and explicit casts checked before/after. Bounded
  2MiB row chunks; no checkpoint-sized audit copy or learned cache replacement.
- Stock native CUDA extension required, all required codec entrypoints present;
  no Triton substitute. Actual18 GDN/6FA2 backend selections recorded.
- Separate fresh plain/observed V1 engine lifetimes. Public STOCK sampler with
  allowed ID353 supplies two genuine forwards at11 then353. Both complete
  248320-word pre-sampler F32 logits must match bytewise across modes. This is a
  teacher policy, not an unconstrained greedy-output comparison.
- Observed mode installs bounded CPU profiler only. Require actual native
  `_C_gguf::` custom-op invocations. These counts are not hardware CUDA kernel
  counts, inter-token latency or throughput.
- Read-only image/inputs; networknone;12GiB RAM/4inside-affinedCPUs; one actual
  UUID-bound SM86 card. Requested provider4GiB/device-comparison5GiB; explicit
  512MiB KV cache, context512/B8 scheduler limits; eager execution. Actual GPU
  worker Torch allocator peak<=4GiB; physical endpoints and quiet process lists
  recorded. Neither sampled endpoints nor Torch peaks establish a complete
  physical GPU peak or hard device-allocation cap; report explicitly says so.
- Exact accepted53ee owned helper,600s/mode,4MiB logs,64MiB perfile,128MiB complete
  paired artifacts; finite independent nonce/CID/PGID cleanup/EOF mandatory.
  All compute owners must be absent before the next mode; no GPU recovery.

## Distinct arithmetic profiles

Same GGUF bytes prove source and weight-quantization identity only. Accepted
Zerogravity dynamically quantizes activations: Q8_0 forQ4_0/Q8_0, Q8_1 forQ4_1,
Q8_K forQ6_K, with its source-qualified rounding/scales/reducers. Stock plugin
CUDA uses Q8_1 block32 across these weight types, `roundf`, half scales/raw-input
sums and its own GEMV/GEMM schedule. It returns the activation dtype, ordinarily
BF16. No blanket claim of unquantized Zerogravity activations is valid.

Stock ordinary norms undo converted runtime+1 via subtraction; negative GGUF
SSM A is mapped through `log(-A)` into vLLM A_log. Loaded GDN gamma is BF16 while
A_log is F32. Embedding/projection outputs, convolution/state/cache dtypes and
FLA/FA2 arithmetic therefore require their own evidence. The loaded parameter
audit checks stock transformation/cast behavior; it is not an independent
arithmetic qualification of those transforms or a CPU AVX2 profile.

Both approved source configs have16 GDN key/value heads: ratio1, so the stock
converter value-head tiling restore is identity. General ratios remain outside
this preparation. Full42 cross-provider state, unconstrained generation/EOS,
B2/B4/B8 public workloads, default CUDA graphs, true token ITL/fairness/SLOs,
complete hardware kernel counts and matched performance remain unqualified.

## ROOT execution

After reviewing the frozen loader manifest and nine source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_loader_fixtures.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_loader.py \
  --model 0.8b --conversion .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-2 \
  --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-loader-08b-1 \
  --gpu-uuid APPROVED_ACTUAL_GPU_UUID --prepared-sha256 REVIEWED_LOADER_MANIFEST_SHA256
```

Repeat2B with its exact accepted `root-conversion-2b-2` and fresh output.
Existing output refuses. Every prior failed preparation/conversion stays intact.
Run only after ROOT releases the quiet GPU hold. This does not start serving or
claim vLLM parity. Benchmark schema/workload work follows actual loader gates.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/LOGITS_ABI_V3.md

Original UTF-8 bytes: 3680; SHA256 `e76ee0f24fd46d7b4ab3224fbde8d74c976cd654aa826a22dbb0856bf0b66d80`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive native-logit observation correction

Author: spine_review. SOURCE ONLY; no author runtime or operator execution.
V1loader freeze c8e157ac… and failed ROOT attempt `root-loader-08b-1` unchanged.
No retrospective dtype/shape observation is invented: V1 refused before saving
that metadata. Its successful initialization is not a completed paired gate.

Pinned actual primary source, now13 vLLM Python files:

- `model_executor/layers/logits_processor.py:98–120,136–199`: forward returns
  native head result; no universal F32 cast. Head projection uses configured
  dtype; TP1 trims vocabulary only.
- `config/model.py:2047–2073,2447–2466`: generation head defaults to model dtype.
  With requested BF16 model and no override, head dtype is BF16, not PythonNone.
- Stock plugin `quantization/linear.py:32–60,228–277` calls native GGUF operator.
  Its pinned `csrc/gguf/gguf_kernel.cu:104–116,228–232` allocates output using
  `X.scalar_type()`. Source predicts native BF16 for this actual profile; fresh
  actual metadata and bytes remain required.
- ACTUAL runner `v1/worker/gpu/model_runner.py:145–147,1578–1600` uses the GPU
  sampler at `v1/worker/gpu/sample/sampler.py`, not generic `v1/sample/sampler.py`.
- GPU sampler `apply_sampling_params:235–239` returns native logits when no
  processing is needed; otherwise copies into an F32 allocation. Allowed IDs
  set processing true through `sample/logit_bias.py:58–74`. Public353-only policy
  remains unchanged. This preparation does NOT observe that sampler copy.

V3 changes observation only:

- Write bounded actual Tensor dtype/device/shape/stride/head/caller metadata
  BEFORE any output dtype/device/shape refusal. Unknown/unsupported ABI refuses
  with diagnostic retained.
- Read back exact native BF16/F32 bytes, preserving signed-zero/subnormal bits.
  Persist native bytes and separately named exact F32 observer representation.
  BF16 widening is integer shift16, not a model/kernel replacement or improved
  arithmetic. Native F32 uses an explicit identity copy. Actual closed head
  profile still requires BF16, default scale1/no cap, matching hidden dtype.
- Both complete248320 native outputs and both exact observer representations
  must match between fresh plain and CPU-profiler observed modes. Neither the
  widened copy nor software custom-op counts claim a hardware kernel receipt,
  actual sampler stage, cross-engine parity, full42 state or performance proof.
- Unknown/nonfinite/foreign profile remains fail closed. Raw payloads within
  the already validated shape/cap remain retained even on finite-bit refusal.
- All320 source audits, true projector/tokenizer/source pins, noMM requirement,
  original allowed-token policy, memory/resource observations and53ee cleanup
  code remain unchanged. Original conversion source/freezes are immutable.

ROOT first reviews all additive files and runs12 source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_loader_fixtures_v3.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_loader_v3.py \
  --model 0.8b --conversion .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-2 \
  --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-loader-08b-2 \
  --gpu-uuid APPROVED_ACTUAL_GPU_UUID --prepared-sha256 REVIEWED_LOADER_V3_MANIFEST_SHA256
```

ROOT alone executes; fresh both-mode actual gates remain required. Repeat2B only
after original physical owners are absent. This correction cannot qualify the
failed V1 attempt or any unobserved original sampler arithmetic.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/MTP_INVENTORY_V4.md

Original UTF-8 bytes: 2657; SHA256 `f5c7752c3df8ab5682b2c09ec61e4c43ee43aaaf99d8e665494004f3d29db12a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive stock loader V4: exact MTP inventory

- V3/accepted `.8B` result and failed `2B root-loader-2b-1` remain unchanged.
- Actual original `2B` GGUF has335 stored tensors, not320: exactly320 backbone roles +15 `blk.24.*` MTP tensors; header declares25 blocks and one nextn layer. `.8B` has320/no MTP. Both original HF text backbones contain320 roles and24 layers, with tied output head.
- Source-only bounded descriptor read:10,961,541B `.8B` header SHA2010fccc0eb791715e8e02f49e44dd2f251ed6cc5e3a2a4a5a36c65f82d35517;10,962,754B `2B` header SHA32f64b1bf46ed9ec73cd235e4ecbbc97469b05471a97d995504f45da68fedab2. No author model/operator/runtime execution or full-weight scan.
- Stock unchanged `vllm_gguf_plugin/weights_adapter/qwen3_5.py:239–248` finds nextn and excludes the entire MTP block from backbone loading. No runtime/model/codec changes in V4.
- New inventory validates **every** original stored descriptor (name/shape/codec/absolute offset/byte length), complete bounded header digest, exact whole-source identity already authenticated before/after by original `source_check`, and exactly15 named MTP exclusions. Arbitrary skipped/missing/foreign tensors refuse.
- Actual `speculative_config is None` required. Complete320 source mappings, packed/raw payload readbacks, auxiliary codec metadata, tied head, zero learned MM/MTP parameters, native logits and all old paired transparency/lifecycle guards remain mandatory. Excluded MTP payloads are whole-file authenticated, not loaded/executed/operator-qualified.
- `loaded_roles_v4` retains the original parameter-audit and codec-audit function bodies byte-for-byte; only inventory selection replaces the erroneous stored-count assumption.
- V4 changes only ignored diagnostic scripts/manifest wiring. Native GPU math, plugin, source GGUF, tokenizer, projector and exact original teacher path remain unchanged.

ROOT commands (source-free first; no author execution):

```sh
python3 -m unittest discover -s .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020 -p 'test_loader_fixtures_v[34].py'
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_loader_v4.py --model 2b --conversion /home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-2b-2 --output /home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-loader-2b-2 --gpu-uuid ROOT_ACTUAL_UUID --prepared-sha256 ROOT_REVIEWED_V4_MANIFEST_SHA
```

No benchmark, all42 state, crossengine bit parity, default graph or performance qualification follows from this repair.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/TODO.md

Original UTF-8 bytes: 1887; SHA256 `a6ded62a7bffd30728a065f16610fbf7a05c6310edfaf989b8189fa36ff0173a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Genuine stock GGUF loader qualification preparation

Parent: serving-and-speculation. Owner: /root/spine_review.
No author Cargo/build/Torch/operator/model/container/GPU execution. MAIN unchanged.

- [x] Pin original HF .8B/2B full sources, exact approved GGUF, authentic converter,
  root-built stock plugin/gguf wheels, runtime source and exact owned_v4 helper.
- [x] Prepare bounded CPU-only genuine mmproj conversion + all raw visual roles.
  V1 preflight failed:15MTP incorrectly countedvisual; immutable freeze retained.
  Additive V2 pins all15MTP exact,153/297visual→154/298projector. ROOT8fixturesPASS.
- [x] ROOT both actual V2projectors independently allrawbitvalidated; source
  unchanged/cleanup[]/EOF/absence. Exact actual hashes/receipts projector-pins.json.
- [x] Prepare real stock language-only V1 loader/two-logit role/backend evidence.
- [x] Prepare independent paired validator/nine source-free refusals and freeze.
  Author AST/source only; no fixtures/container/operator/model/GPU execution.
- [ ] ROOT loader peer/source review, nine fixtures, actual plain/observed
  source/320roles/noMM/496640full-logit words/backend/worker-memory/cleanup gates.
- [x] ROOT firstloader:9fixturesPASS; actual .8B initialization/audits passed,
  then firstoutputrefused F32-only observer assertion. Attempt preserved and
  remains UNQUALIFIED; actual output dtype/shape was not recorded by V1.
- [x] Additive native-logit V3 sourceprep frozen04406c89: original28pins include
  old18+manifest unchanged; actual metadata beforeguard, native bits separate
  from exact observer widening; actual GPU sampler cast remains UNOBSERVED.
  Source13runtimefiles incl correct GPU sampler/caller/head-dtype default pinned.
- [ ] ROOT V3review/12fixtures/fresh actual paired native-bit gates forbothmodels.
- [ ] Full42 cross-provider/numerical profile/wire/performance gates still open.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/conversion-frozen/CONVERSION.md

Original UTF-8 bytes: 2661; SHA256 `fdae080bee5dd1b3784717b6e2abc7300373bd5653cb13338fec7a8d4508610a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Genuine projector preparation

- Parent: serving-and-speculation; author: spine_review.
- Source-only preparation. No conversion, container, model, Torch or GPU run by author.
- Exact existing llama.cpp0.6.0 conversion source; authentic full HF snapshots;
  original approved text GGUF mounted read-only and hashed before/after.
- Genuine public CLI: `convert_hf_to_gguf.py /source --mmproj --outtype bf16`.
  No patch, synthetic config, learned parameter substitution or LM reconversion.
- Converter accepts only original visual roles, splits the two temporal patch
  slices, forces patch tensors and some one-dimensional tensors to F32.
  Every emitted payload must equal original BF16 bits or their exact F32 widening.
  Torch-side audit plus independent stdlib GGUF/extent/source-byte audit.
- .8B:168 original visual roles→169 tensors; 2B:312→313. Both have320 text roles
  and24 text layers; visual depths are12/24. No text role may occur in projector.
- Read-only image/source/converter/backbone; exclusive output; networknone,
  no CUDA CDI/device; CUDA visibilitydisabled.12GiB memory,4CPU set inside
  before Torchimport,300s,1GiBperfile,1088MiB complete output tree,4MiB log.
- Exact reviewed `owned_container_v4.py`SHA53ee4270…; nonce/CID ownership,
  WNOWAIT, all signals before reap, independent cleanup stages, final group/
  container absence and EOF mandatory. No unavailable cpuset cgroup option.
- Successful conversion qualifies source-bound projector bytes only. Stock
  plugin loader, GPU math, all42 domains and performance remain unqualified.

ROOT, after reviewing the frozen manifest and six source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_conversion_fixtures.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_conversion.py \
  --model 0.8b --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-1 \
  --prepared-sha256 REVIEWED_CONVERSION_MANIFEST_SHA256
```

Repeat with `--model 2b` and a fresh `root-conversion-2b-1` output. Existing output
refuses; failed attempts remain immutable. Host validation rehashes authentic
source files and audits every projector payload, then proves final cleanup.

Stock loader followup must use this genuine projector with original full HF
config/tokenizer and `language_model_only=True`. It must prove actual320 loaded
text roles, zero learned vision parameters, stock CUDA plugin backend and
unchanged full-logit observations before any benchmark. Same GGUF source bytes
do not imply matching activation codec, parameter casts or execution arithmetic.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/conversion-frozen-v2/CONVERSION_V2.md

Original UTF-8 bytes: 3114; SHA256 `3a48c37f209869c137a5f1c881442e061018152c31c3159028fbfb626dcff266`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Genuine projector preparation

- Parent: serving-and-speculation; author: spine_review.
- Source-only preparation. No conversion, container, model, Torch or GPU run by author.
- Exact existing llama.cpp0.6.0 conversion source; authentic full HF snapshots;
  original approved text GGUF mounted read-only and hashed before/after.
- Genuine public CLI: `convert_hf_to_gguf.py /source --mmproj --outtype bf16`.
  No patch, synthetic config, learned parameter substitution or LM reconversion.
- Converter accepts only original visual roles, splits the two temporal patch
  slices, forces patch tensors and some one-dimensional tensors to F32.
  Every emitted payload must equal original BF16 bits or their exact F32 widening.
  Torch-side audit plus independent stdlib GGUF/extent/source-byte audit.
- .8B:153 original visual roles→154 tensors; 2B:297→298. Both have320 text roles
  and24 text layers; visual depths are12/24. No text role may occur in projector.
- Read-only image/source/converter/backbone; exclusive output; networknone,
  no CUDA CDI/device; CUDA visibilitydisabled.12GiB memory,4CPU set inside
  before Torchimport,300s,1GiBperfile,1088MiB complete output tree,4MiB log.
- Exact reviewed `owned_container_v4.py`SHA53ee4270…; nonce/CID ownership,
  WNOWAIT, all signals before reap, independent cleanup stages, final group/
  container absence and EOF mandatory. No unavailable cpuset cgroup option.
- Successful conversion qualifies source-bound projector bytes only. Stock
  plugin loader, GPU math, all42 domains and performance remain unqualified.

ROOT, after reviewing the frozen manifest and eight source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_conversion_fixtures_v2.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_conversion_v2.py \
  --model 0.8b --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-1 \
  --prepared-sha256 REVIEWED_CONVERSION_MANIFEST_SHA256
```

Repeat with `--model 2b` and a fresh `root-conversion-2b-1` output. Existing output
refuses; failed attempts remain immutable. Host validation rehashes authentic
source files and audits every projector payload, then proves final cleanup.

Stock loader followup must use this genuine projector with original full HF
config/tokenizer and `language_model_only=True`. It must prove actual320 loaded
text roles, zero learned vision parameters, stock CUDA plugin backend and
unchanged full-logit observations before any benchmark. Same GGUF source bytes
do not imply matching activation codec, parameter casts or execution arithmetic.

V2 additive correction: V1 preflight refused before any container/conversion.
Both full snapshots also contain15 authentic BF16 MTP roles. `source-pins-v2.json`
pins their exact names/dtypes/shapes/file extents; source closure requires all15
and excludes them explicitly. The unchanged converter already filters `mtp.`.
V1 eight frozen files and its manifest remain byte-identical. No actual
conversion success is asserted by this preparation.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/loader-frozen/CONVERSION_V2.md

Original UTF-8 bytes: 3114; SHA256 `3a48c37f209869c137a5f1c881442e061018152c31c3159028fbfb626dcff266`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Genuine projector preparation

- Parent: serving-and-speculation; author: spine_review.
- Source-only preparation. No conversion, container, model, Torch or GPU run by author.
- Exact existing llama.cpp0.6.0 conversion source; authentic full HF snapshots;
  original approved text GGUF mounted read-only and hashed before/after.
- Genuine public CLI: `convert_hf_to_gguf.py /source --mmproj --outtype bf16`.
  No patch, synthetic config, learned parameter substitution or LM reconversion.
- Converter accepts only original visual roles, splits the two temporal patch
  slices, forces patch tensors and some one-dimensional tensors to F32.
  Every emitted payload must equal original BF16 bits or their exact F32 widening.
  Torch-side audit plus independent stdlib GGUF/extent/source-byte audit.
- .8B:153 original visual roles→154 tensors; 2B:297→298. Both have320 text roles
  and24 text layers; visual depths are12/24. No text role may occur in projector.
- Read-only image/source/converter/backbone; exclusive output; networknone,
  no CUDA CDI/device; CUDA visibilitydisabled.12GiB memory,4CPU set inside
  before Torchimport,300s,1GiBperfile,1088MiB complete output tree,4MiB log.
- Exact reviewed `owned_container_v4.py`SHA53ee4270…; nonce/CID ownership,
  WNOWAIT, all signals before reap, independent cleanup stages, final group/
  container absence and EOF mandatory. No unavailable cpuset cgroup option.
- Successful conversion qualifies source-bound projector bytes only. Stock
  plugin loader, GPU math, all42 domains and performance remain unqualified.

ROOT, after reviewing the frozen manifest and eight source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_conversion_fixtures_v2.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_conversion_v2.py \
  --model 0.8b --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-1 \
  --prepared-sha256 REVIEWED_CONVERSION_MANIFEST_SHA256
```

Repeat with `--model 2b` and a fresh `root-conversion-2b-1` output. Existing output
refuses; failed attempts remain immutable. Host validation rehashes authentic
source files and audits every projector payload, then proves final cleanup.

Stock loader followup must use this genuine projector with original full HF
config/tokenizer and `language_model_only=True`. It must prove actual320 loaded
text roles, zero learned vision parameters, stock CUDA plugin backend and
unchanged full-logit observations before any benchmark. Same GGUF source bytes
do not imply matching activation codec, parameter casts or execution arithmetic.

V2 additive correction: V1 preflight refused before any container/conversion.
Both full snapshots also contain15 authentic BF16 MTP roles. `source-pins-v2.json`
pins their exact names/dtypes/shapes/file extents; source closure requires all15
and excludes them explicitly. The unchanged converter already filters `mtp.`.
V1 eight frozen files and its manifest remain byte-identical. No actual
conversion success is asserted by this preparation.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/loader-frozen/LOADER.md

Original UTF-8 bytes: 5263; SHA256 `cfed78b82eaf9c90768d851bde2181a10de8367da822826f28e47ad02e747a69`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Stock GGUF language-only qualification preparation

Author: spine_review. Parent: serving-and-speculation. Source-only author work;
ROOT alone executes containers/operators/models/GPU. No MAIN changes.

- Stock official plugin `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`;
  ROOT stock wheel SHAaf9de736…, extension SHAebb896a5…; exact gguf0.19 wheel.
  Unmodified wheel members/RECORD extracted into ephemeral private site. No
  in-image install, import shim, numeric patch or alternate backend.
- Genuine pinned vLLM0.31.0+cu129/Torch2.13.0+cu129 in accepted imagef57d77….
  Selected original runtime source, all111 plugin Python files and native DSO
  pinned before/after. Actual identities and imported package path required.
- Original approved text GGUF unchanged; full original HF config/tokenizer;
  ROOT-verified genuine projectors bound by exact output and actual receipt hashes
  in `projector-pins.json`. Both authentic15-role MTP sets explicitly excluded;
  full HF source hash/metadata closure remains mandatory.
- Public `LLM` with `language_model_only=True` and actual
  `model_loader_extra_config={'mm_proj': genuine_projector}`. The stock vision
  guard remains. Require real `StageMissingLayer` and zero learned vision/MTP
  parameters; all320 text source roles must map to actual CUDA parameters.
- Same original public `WeightsMapper` metadata traversal; all packed shards,
  exact source geometry/codec bytes, padding, tied storage, auxiliary codec
  metadata, loader transforms and explicit casts checked before/after. Bounded
  2MiB row chunks; no checkpoint-sized audit copy or learned cache replacement.
- Stock native CUDA extension required, all required codec entrypoints present;
  no Triton substitute. Actual18 GDN/6FA2 backend selections recorded.
- Separate fresh plain/observed V1 engine lifetimes. Public STOCK sampler with
  allowed ID353 supplies two genuine forwards at11 then353. Both complete
  248320-word pre-sampler F32 logits must match bytewise across modes. This is a
  teacher policy, not an unconstrained greedy-output comparison.
- Observed mode installs bounded CPU profiler only. Require actual native
  `_C_gguf::` custom-op invocations. These counts are not hardware CUDA kernel
  counts, inter-token latency or throughput.
- Read-only image/inputs; networknone;12GiB RAM/4inside-affinedCPUs; one actual
  UUID-bound SM86 card. Requested provider4GiB/device-comparison5GiB; explicit
  512MiB KV cache, context512/B8 scheduler limits; eager execution. Actual GPU
  worker Torch allocator peak<=4GiB; physical endpoints and quiet process lists
  recorded. Neither sampled endpoints nor Torch peaks establish a complete
  physical GPU peak or hard device-allocation cap; report explicitly says so.
- Exact accepted53ee owned helper,600s/mode,4MiB logs,64MiB perfile,128MiB complete
  paired artifacts; finite independent nonce/CID/PGID cleanup/EOF mandatory.
  All compute owners must be absent before the next mode; no GPU recovery.

## Distinct arithmetic profiles

Same GGUF bytes prove source and weight-quantization identity only. Accepted
Zerogravity dynamically quantizes activations: Q8_0 forQ4_0/Q8_0, Q8_1 forQ4_1,
Q8_K forQ6_K, with its source-qualified rounding/scales/reducers. Stock plugin
CUDA uses Q8_1 block32 across these weight types, `roundf`, half scales/raw-input
sums and its own GEMV/GEMM schedule. It returns the activation dtype, ordinarily
BF16. No blanket claim of unquantized Zerogravity activations is valid.

Stock ordinary norms undo converted runtime+1 via subtraction; negative GGUF
SSM A is mapped through `log(-A)` into vLLM A_log. Loaded GDN gamma is BF16 while
A_log is F32. Embedding/projection outputs, convolution/state/cache dtypes and
FLA/FA2 arithmetic therefore require their own evidence. The loaded parameter
audit checks stock transformation/cast behavior; it is not an independent
arithmetic qualification of those transforms or a CPU AVX2 profile.

Both approved source configs have16 GDN key/value heads: ratio1, so the stock
converter value-head tiling restore is identity. General ratios remain outside
this preparation. Full42 cross-provider state, unconstrained generation/EOS,
B2/B4/B8 public workloads, default CUDA graphs, true token ITL/fairness/SLOs,
complete hardware kernel counts and matched performance remain unqualified.

## ROOT execution

After reviewing the frozen loader manifest and nine source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_loader_fixtures.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_loader.py \
  --model 0.8b --conversion .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-2 \
  --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-loader-08b-1 \
  --gpu-uuid APPROVED_ACTUAL_GPU_UUID --prepared-sha256 REVIEWED_LOADER_MANIFEST_SHA256
```

Repeat2B with its exact accepted `root-conversion-2b-2` and fresh output.
Existing output refuses. Every prior failed preparation/conversion stays intact.
Run only after ROOT releases the quiet GPU hold. This does not start serving or
claim vLLM parity. Benchmark schema/workload work follows actual loader gates.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/loader-frozen-v3/CONVERSION_V2.md

Original UTF-8 bytes: 3114; SHA256 `3a48c37f209869c137a5f1c881442e061018152c31c3159028fbfb626dcff266`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Genuine projector preparation

- Parent: serving-and-speculation; author: spine_review.
- Source-only preparation. No conversion, container, model, Torch or GPU run by author.
- Exact existing llama.cpp0.6.0 conversion source; authentic full HF snapshots;
  original approved text GGUF mounted read-only and hashed before/after.
- Genuine public CLI: `convert_hf_to_gguf.py /source --mmproj --outtype bf16`.
  No patch, synthetic config, learned parameter substitution or LM reconversion.
- Converter accepts only original visual roles, splits the two temporal patch
  slices, forces patch tensors and some one-dimensional tensors to F32.
  Every emitted payload must equal original BF16 bits or their exact F32 widening.
  Torch-side audit plus independent stdlib GGUF/extent/source-byte audit.
- .8B:153 original visual roles→154 tensors; 2B:297→298. Both have320 text roles
  and24 text layers; visual depths are12/24. No text role may occur in projector.
- Read-only image/source/converter/backbone; exclusive output; networknone,
  no CUDA CDI/device; CUDA visibilitydisabled.12GiB memory,4CPU set inside
  before Torchimport,300s,1GiBperfile,1088MiB complete output tree,4MiB log.
- Exact reviewed `owned_container_v4.py`SHA53ee4270…; nonce/CID ownership,
  WNOWAIT, all signals before reap, independent cleanup stages, final group/
  container absence and EOF mandatory. No unavailable cpuset cgroup option.
- Successful conversion qualifies source-bound projector bytes only. Stock
  plugin loader, GPU math, all42 domains and performance remain unqualified.

ROOT, after reviewing the frozen manifest and eight source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_conversion_fixtures_v2.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_conversion_v2.py \
  --model 0.8b --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-1 \
  --prepared-sha256 REVIEWED_CONVERSION_MANIFEST_SHA256
```

Repeat with `--model 2b` and a fresh `root-conversion-2b-1` output. Existing output
refuses; failed attempts remain immutable. Host validation rehashes authentic
source files and audits every projector payload, then proves final cleanup.

Stock loader followup must use this genuine projector with original full HF
config/tokenizer and `language_model_only=True`. It must prove actual320 loaded
text roles, zero learned vision parameters, stock CUDA plugin backend and
unchanged full-logit observations before any benchmark. Same GGUF source bytes
do not imply matching activation codec, parameter casts or execution arithmetic.

V2 additive correction: V1 preflight refused before any container/conversion.
Both full snapshots also contain15 authentic BF16 MTP roles. `source-pins-v2.json`
pins their exact names/dtypes/shapes/file extents; source closure requires all15
and excludes them explicitly. The unchanged converter already filters `mtp.`.
V1 eight frozen files and its manifest remain byte-identical. No actual
conversion success is asserted by this preparation.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/loader-frozen-v3/LOADER.md

Original UTF-8 bytes: 5263; SHA256 `cfed78b82eaf9c90768d851bde2181a10de8367da822826f28e47ad02e747a69`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Stock GGUF language-only qualification preparation

Author: spine_review. Parent: serving-and-speculation. Source-only author work;
ROOT alone executes containers/operators/models/GPU. No MAIN changes.

- Stock official plugin `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`;
  ROOT stock wheel SHAaf9de736…, extension SHAebb896a5…; exact gguf0.19 wheel.
  Unmodified wheel members/RECORD extracted into ephemeral private site. No
  in-image install, import shim, numeric patch or alternate backend.
- Genuine pinned vLLM0.31.0+cu129/Torch2.13.0+cu129 in accepted imagef57d77….
  Selected original runtime source, all111 plugin Python files and native DSO
  pinned before/after. Actual identities and imported package path required.
- Original approved text GGUF unchanged; full original HF config/tokenizer;
  ROOT-verified genuine projectors bound by exact output and actual receipt hashes
  in `projector-pins.json`. Both authentic15-role MTP sets explicitly excluded;
  full HF source hash/metadata closure remains mandatory.
- Public `LLM` with `language_model_only=True` and actual
  `model_loader_extra_config={'mm_proj': genuine_projector}`. The stock vision
  guard remains. Require real `StageMissingLayer` and zero learned vision/MTP
  parameters; all320 text source roles must map to actual CUDA parameters.
- Same original public `WeightsMapper` metadata traversal; all packed shards,
  exact source geometry/codec bytes, padding, tied storage, auxiliary codec
  metadata, loader transforms and explicit casts checked before/after. Bounded
  2MiB row chunks; no checkpoint-sized audit copy or learned cache replacement.
- Stock native CUDA extension required, all required codec entrypoints present;
  no Triton substitute. Actual18 GDN/6FA2 backend selections recorded.
- Separate fresh plain/observed V1 engine lifetimes. Public STOCK sampler with
  allowed ID353 supplies two genuine forwards at11 then353. Both complete
  248320-word pre-sampler F32 logits must match bytewise across modes. This is a
  teacher policy, not an unconstrained greedy-output comparison.
- Observed mode installs bounded CPU profiler only. Require actual native
  `_C_gguf::` custom-op invocations. These counts are not hardware CUDA kernel
  counts, inter-token latency or throughput.
- Read-only image/inputs; networknone;12GiB RAM/4inside-affinedCPUs; one actual
  UUID-bound SM86 card. Requested provider4GiB/device-comparison5GiB; explicit
  512MiB KV cache, context512/B8 scheduler limits; eager execution. Actual GPU
  worker Torch allocator peak<=4GiB; physical endpoints and quiet process lists
  recorded. Neither sampled endpoints nor Torch peaks establish a complete
  physical GPU peak or hard device-allocation cap; report explicitly says so.
- Exact accepted53ee owned helper,600s/mode,4MiB logs,64MiB perfile,128MiB complete
  paired artifacts; finite independent nonce/CID/PGID cleanup/EOF mandatory.
  All compute owners must be absent before the next mode; no GPU recovery.

## Distinct arithmetic profiles

Same GGUF bytes prove source and weight-quantization identity only. Accepted
Zerogravity dynamically quantizes activations: Q8_0 forQ4_0/Q8_0, Q8_1 forQ4_1,
Q8_K forQ6_K, with its source-qualified rounding/scales/reducers. Stock plugin
CUDA uses Q8_1 block32 across these weight types, `roundf`, half scales/raw-input
sums and its own GEMV/GEMM schedule. It returns the activation dtype, ordinarily
BF16. No blanket claim of unquantized Zerogravity activations is valid.

Stock ordinary norms undo converted runtime+1 via subtraction; negative GGUF
SSM A is mapped through `log(-A)` into vLLM A_log. Loaded GDN gamma is BF16 while
A_log is F32. Embedding/projection outputs, convolution/state/cache dtypes and
FLA/FA2 arithmetic therefore require their own evidence. The loaded parameter
audit checks stock transformation/cast behavior; it is not an independent
arithmetic qualification of those transforms or a CPU AVX2 profile.

Both approved source configs have16 GDN key/value heads: ratio1, so the stock
converter value-head tiling restore is identity. General ratios remain outside
this preparation. Full42 cross-provider state, unconstrained generation/EOS,
B2/B4/B8 public workloads, default CUDA graphs, true token ITL/fairness/SLOs,
complete hardware kernel counts and matched performance remain unqualified.

## ROOT execution

After reviewing the frozen loader manifest and nine source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_loader_fixtures.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_loader.py \
  --model 0.8b --conversion .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-2 \
  --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-loader-08b-1 \
  --gpu-uuid APPROVED_ACTUAL_GPU_UUID --prepared-sha256 REVIEWED_LOADER_MANIFEST_SHA256
```

Repeat2B with its exact accepted `root-conversion-2b-2` and fresh output.
Existing output refuses. Every prior failed preparation/conversion stays intact.
Run only after ROOT releases the quiet GPU hold. This does not start serving or
claim vLLM parity. Benchmark schema/workload work follows actual loader gates.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/loader-frozen-v3/LOGITS_ABI_V3.md

Original UTF-8 bytes: 3680; SHA256 `e76ee0f24fd46d7b4ab3224fbde8d74c976cd654aa826a22dbb0856bf0b66d80`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive native-logit observation correction

Author: spine_review. SOURCE ONLY; no author runtime or operator execution.
V1loader freeze c8e157ac… and failed ROOT attempt `root-loader-08b-1` unchanged.
No retrospective dtype/shape observation is invented: V1 refused before saving
that metadata. Its successful initialization is not a completed paired gate.

Pinned actual primary source, now13 vLLM Python files:

- `model_executor/layers/logits_processor.py:98–120,136–199`: forward returns
  native head result; no universal F32 cast. Head projection uses configured
  dtype; TP1 trims vocabulary only.
- `config/model.py:2047–2073,2447–2466`: generation head defaults to model dtype.
  With requested BF16 model and no override, head dtype is BF16, not PythonNone.
- Stock plugin `quantization/linear.py:32–60,228–277` calls native GGUF operator.
  Its pinned `csrc/gguf/gguf_kernel.cu:104–116,228–232` allocates output using
  `X.scalar_type()`. Source predicts native BF16 for this actual profile; fresh
  actual metadata and bytes remain required.
- ACTUAL runner `v1/worker/gpu/model_runner.py:145–147,1578–1600` uses the GPU
  sampler at `v1/worker/gpu/sample/sampler.py`, not generic `v1/sample/sampler.py`.
- GPU sampler `apply_sampling_params:235–239` returns native logits when no
  processing is needed; otherwise copies into an F32 allocation. Allowed IDs
  set processing true through `sample/logit_bias.py:58–74`. Public353-only policy
  remains unchanged. This preparation does NOT observe that sampler copy.

V3 changes observation only:

- Write bounded actual Tensor dtype/device/shape/stride/head/caller metadata
  BEFORE any output dtype/device/shape refusal. Unknown/unsupported ABI refuses
  with diagnostic retained.
- Read back exact native BF16/F32 bytes, preserving signed-zero/subnormal bits.
  Persist native bytes and separately named exact F32 observer representation.
  BF16 widening is integer shift16, not a model/kernel replacement or improved
  arithmetic. Native F32 uses an explicit identity copy. Actual closed head
  profile still requires BF16, default scale1/no cap, matching hidden dtype.
- Both complete248320 native outputs and both exact observer representations
  must match between fresh plain and CPU-profiler observed modes. Neither the
  widened copy nor software custom-op counts claim a hardware kernel receipt,
  actual sampler stage, cross-engine parity, full42 state or performance proof.
- Unknown/nonfinite/foreign profile remains fail closed. Raw payloads within
  the already validated shape/cap remain retained even on finite-bit refusal.
- All320 source audits, true projector/tokenizer/source pins, noMM requirement,
  original allowed-token policy, memory/resource observations and53ee cleanup
  code remain unchanged. Original conversion source/freezes are immutable.

ROOT first reviews all additive files and runs12 source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_loader_fixtures_v3.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_loader_v3.py \
  --model 0.8b --conversion .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-2 \
  --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-loader-08b-2 \
  --gpu-uuid APPROVED_ACTUAL_GPU_UUID --prepared-sha256 REVIEWED_LOADER_V3_MANIFEST_SHA256
```

ROOT alone executes; fresh both-mode actual gates remain required. Repeat2B only
after original physical owners are absent. This correction cannot qualify the
failed V1 attempt or any unobserved original sampler arithmetic.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/loader-frozen-v4/CONVERSION_V2.md

Original UTF-8 bytes: 3114; SHA256 `3a48c37f209869c137a5f1c881442e061018152c31c3159028fbfb626dcff266`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Genuine projector preparation

- Parent: serving-and-speculation; author: spine_review.
- Source-only preparation. No conversion, container, model, Torch or GPU run by author.
- Exact existing llama.cpp0.6.0 conversion source; authentic full HF snapshots;
  original approved text GGUF mounted read-only and hashed before/after.
- Genuine public CLI: `convert_hf_to_gguf.py /source --mmproj --outtype bf16`.
  No patch, synthetic config, learned parameter substitution or LM reconversion.
- Converter accepts only original visual roles, splits the two temporal patch
  slices, forces patch tensors and some one-dimensional tensors to F32.
  Every emitted payload must equal original BF16 bits or their exact F32 widening.
  Torch-side audit plus independent stdlib GGUF/extent/source-byte audit.
- .8B:153 original visual roles→154 tensors; 2B:297→298. Both have320 text roles
  and24 text layers; visual depths are12/24. No text role may occur in projector.
- Read-only image/source/converter/backbone; exclusive output; networknone,
  no CUDA CDI/device; CUDA visibilitydisabled.12GiB memory,4CPU set inside
  before Torchimport,300s,1GiBperfile,1088MiB complete output tree,4MiB log.
- Exact reviewed `owned_container_v4.py`SHA53ee4270…; nonce/CID ownership,
  WNOWAIT, all signals before reap, independent cleanup stages, final group/
  container absence and EOF mandatory. No unavailable cpuset cgroup option.
- Successful conversion qualifies source-bound projector bytes only. Stock
  plugin loader, GPU math, all42 domains and performance remain unqualified.

ROOT, after reviewing the frozen manifest and eight source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_conversion_fixtures_v2.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_conversion_v2.py \
  --model 0.8b --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-1 \
  --prepared-sha256 REVIEWED_CONVERSION_MANIFEST_SHA256
```

Repeat with `--model 2b` and a fresh `root-conversion-2b-1` output. Existing output
refuses; failed attempts remain immutable. Host validation rehashes authentic
source files and audits every projector payload, then proves final cleanup.

Stock loader followup must use this genuine projector with original full HF
config/tokenizer and `language_model_only=True`. It must prove actual320 loaded
text roles, zero learned vision parameters, stock CUDA plugin backend and
unchanged full-logit observations before any benchmark. Same GGUF source bytes
do not imply matching activation codec, parameter casts or execution arithmetic.

V2 additive correction: V1 preflight refused before any container/conversion.
Both full snapshots also contain15 authentic BF16 MTP roles. `source-pins-v2.json`
pins their exact names/dtypes/shapes/file extents; source closure requires all15
and excludes them explicitly. The unchanged converter already filters `mtp.`.
V1 eight frozen files and its manifest remain byte-identical. No actual
conversion success is asserted by this preparation.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/loader-frozen-v4/LOADER.md

Original UTF-8 bytes: 5263; SHA256 `cfed78b82eaf9c90768d851bde2181a10de8367da822826f28e47ad02e747a69`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Stock GGUF language-only qualification preparation

Author: spine_review. Parent: serving-and-speculation. Source-only author work;
ROOT alone executes containers/operators/models/GPU. No MAIN changes.

- Stock official plugin `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`;
  ROOT stock wheel SHAaf9de736…, extension SHAebb896a5…; exact gguf0.19 wheel.
  Unmodified wheel members/RECORD extracted into ephemeral private site. No
  in-image install, import shim, numeric patch or alternate backend.
- Genuine pinned vLLM0.31.0+cu129/Torch2.13.0+cu129 in accepted imagef57d77….
  Selected original runtime source, all111 plugin Python files and native DSO
  pinned before/after. Actual identities and imported package path required.
- Original approved text GGUF unchanged; full original HF config/tokenizer;
  ROOT-verified genuine projectors bound by exact output and actual receipt hashes
  in `projector-pins.json`. Both authentic15-role MTP sets explicitly excluded;
  full HF source hash/metadata closure remains mandatory.
- Public `LLM` with `language_model_only=True` and actual
  `model_loader_extra_config={'mm_proj': genuine_projector}`. The stock vision
  guard remains. Require real `StageMissingLayer` and zero learned vision/MTP
  parameters; all320 text source roles must map to actual CUDA parameters.
- Same original public `WeightsMapper` metadata traversal; all packed shards,
  exact source geometry/codec bytes, padding, tied storage, auxiliary codec
  metadata, loader transforms and explicit casts checked before/after. Bounded
  2MiB row chunks; no checkpoint-sized audit copy or learned cache replacement.
- Stock native CUDA extension required, all required codec entrypoints present;
  no Triton substitute. Actual18 GDN/6FA2 backend selections recorded.
- Separate fresh plain/observed V1 engine lifetimes. Public STOCK sampler with
  allowed ID353 supplies two genuine forwards at11 then353. Both complete
  248320-word pre-sampler F32 logits must match bytewise across modes. This is a
  teacher policy, not an unconstrained greedy-output comparison.
- Observed mode installs bounded CPU profiler only. Require actual native
  `_C_gguf::` custom-op invocations. These counts are not hardware CUDA kernel
  counts, inter-token latency or throughput.
- Read-only image/inputs; networknone;12GiB RAM/4inside-affinedCPUs; one actual
  UUID-bound SM86 card. Requested provider4GiB/device-comparison5GiB; explicit
  512MiB KV cache, context512/B8 scheduler limits; eager execution. Actual GPU
  worker Torch allocator peak<=4GiB; physical endpoints and quiet process lists
  recorded. Neither sampled endpoints nor Torch peaks establish a complete
  physical GPU peak or hard device-allocation cap; report explicitly says so.
- Exact accepted53ee owned helper,600s/mode,4MiB logs,64MiB perfile,128MiB complete
  paired artifacts; finite independent nonce/CID/PGID cleanup/EOF mandatory.
  All compute owners must be absent before the next mode; no GPU recovery.

## Distinct arithmetic profiles

Same GGUF bytes prove source and weight-quantization identity only. Accepted
Zerogravity dynamically quantizes activations: Q8_0 forQ4_0/Q8_0, Q8_1 forQ4_1,
Q8_K forQ6_K, with its source-qualified rounding/scales/reducers. Stock plugin
CUDA uses Q8_1 block32 across these weight types, `roundf`, half scales/raw-input
sums and its own GEMV/GEMM schedule. It returns the activation dtype, ordinarily
BF16. No blanket claim of unquantized Zerogravity activations is valid.

Stock ordinary norms undo converted runtime+1 via subtraction; negative GGUF
SSM A is mapped through `log(-A)` into vLLM A_log. Loaded GDN gamma is BF16 while
A_log is F32. Embedding/projection outputs, convolution/state/cache dtypes and
FLA/FA2 arithmetic therefore require their own evidence. The loaded parameter
audit checks stock transformation/cast behavior; it is not an independent
arithmetic qualification of those transforms or a CPU AVX2 profile.

Both approved source configs have16 GDN key/value heads: ratio1, so the stock
converter value-head tiling restore is identity. General ratios remain outside
this preparation. Full42 cross-provider state, unconstrained generation/EOS,
B2/B4/B8 public workloads, default CUDA graphs, true token ITL/fairness/SLOs,
complete hardware kernel counts and matched performance remain unqualified.

## ROOT execution

After reviewing the frozen loader manifest and nine source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_loader_fixtures.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_loader.py \
  --model 0.8b --conversion .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-2 \
  --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-loader-08b-1 \
  --gpu-uuid APPROVED_ACTUAL_GPU_UUID --prepared-sha256 REVIEWED_LOADER_MANIFEST_SHA256
```

Repeat2B with its exact accepted `root-conversion-2b-2` and fresh output.
Existing output refuses. Every prior failed preparation/conversion stays intact.
Run only after ROOT releases the quiet GPU hold. This does not start serving or
claim vLLM parity. Benchmark schema/workload work follows actual loader gates.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/loader-frozen-v4/LOGITS_ABI_V3.md

Original UTF-8 bytes: 3680; SHA256 `e76ee0f24fd46d7b4ab3224fbde8d74c976cd654aa826a22dbb0856bf0b66d80`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive native-logit observation correction

Author: spine_review. SOURCE ONLY; no author runtime or operator execution.
V1loader freeze c8e157ac… and failed ROOT attempt `root-loader-08b-1` unchanged.
No retrospective dtype/shape observation is invented: V1 refused before saving
that metadata. Its successful initialization is not a completed paired gate.

Pinned actual primary source, now13 vLLM Python files:

- `model_executor/layers/logits_processor.py:98–120,136–199`: forward returns
  native head result; no universal F32 cast. Head projection uses configured
  dtype; TP1 trims vocabulary only.
- `config/model.py:2047–2073,2447–2466`: generation head defaults to model dtype.
  With requested BF16 model and no override, head dtype is BF16, not PythonNone.
- Stock plugin `quantization/linear.py:32–60,228–277` calls native GGUF operator.
  Its pinned `csrc/gguf/gguf_kernel.cu:104–116,228–232` allocates output using
  `X.scalar_type()`. Source predicts native BF16 for this actual profile; fresh
  actual metadata and bytes remain required.
- ACTUAL runner `v1/worker/gpu/model_runner.py:145–147,1578–1600` uses the GPU
  sampler at `v1/worker/gpu/sample/sampler.py`, not generic `v1/sample/sampler.py`.
- GPU sampler `apply_sampling_params:235–239` returns native logits when no
  processing is needed; otherwise copies into an F32 allocation. Allowed IDs
  set processing true through `sample/logit_bias.py:58–74`. Public353-only policy
  remains unchanged. This preparation does NOT observe that sampler copy.

V3 changes observation only:

- Write bounded actual Tensor dtype/device/shape/stride/head/caller metadata
  BEFORE any output dtype/device/shape refusal. Unknown/unsupported ABI refuses
  with diagnostic retained.
- Read back exact native BF16/F32 bytes, preserving signed-zero/subnormal bits.
  Persist native bytes and separately named exact F32 observer representation.
  BF16 widening is integer shift16, not a model/kernel replacement or improved
  arithmetic. Native F32 uses an explicit identity copy. Actual closed head
  profile still requires BF16, default scale1/no cap, matching hidden dtype.
- Both complete248320 native outputs and both exact observer representations
  must match between fresh plain and CPU-profiler observed modes. Neither the
  widened copy nor software custom-op counts claim a hardware kernel receipt,
  actual sampler stage, cross-engine parity, full42 state or performance proof.
- Unknown/nonfinite/foreign profile remains fail closed. Raw payloads within
  the already validated shape/cap remain retained even on finite-bit refusal.
- All320 source audits, true projector/tokenizer/source pins, noMM requirement,
  original allowed-token policy, memory/resource observations and53ee cleanup
  code remain unchanged. Original conversion source/freezes are immutable.

ROOT first reviews all additive files and runs12 source-free fixtures:

```sh
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/test_loader_fixtures_v3.py
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_loader_v3.py \
  --model 0.8b --conversion .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-08b-2 \
  --output .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-loader-08b-2 \
  --gpu-uuid APPROVED_ACTUAL_GPU_UUID --prepared-sha256 REVIEWED_LOADER_V3_MANIFEST_SHA256
```

ROOT alone executes; fresh both-mode actual gates remain required. Repeat2B only
after original physical owners are absent. This correction cannot qualify the
failed V1 attempt or any unobserved original sampler arithmetic.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/loader-frozen-v4/MTP_INVENTORY_V4.md

Original UTF-8 bytes: 2657; SHA256 `f5c7752c3df8ab5682b2c09ec61e4c43ee43aaaf99d8e665494004f3d29db12a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive stock loader V4: exact MTP inventory

- V3/accepted `.8B` result and failed `2B root-loader-2b-1` remain unchanged.
- Actual original `2B` GGUF has335 stored tensors, not320: exactly320 backbone roles +15 `blk.24.*` MTP tensors; header declares25 blocks and one nextn layer. `.8B` has320/no MTP. Both original HF text backbones contain320 roles and24 layers, with tied output head.
- Source-only bounded descriptor read:10,961,541B `.8B` header SHA2010fccc0eb791715e8e02f49e44dd2f251ed6cc5e3a2a4a5a36c65f82d35517;10,962,754B `2B` header SHA32f64b1bf46ed9ec73cd235e4ecbbc97469b05471a97d995504f45da68fedab2. No author model/operator/runtime execution or full-weight scan.
- Stock unchanged `vllm_gguf_plugin/weights_adapter/qwen3_5.py:239–248` finds nextn and excludes the entire MTP block from backbone loading. No runtime/model/codec changes in V4.
- New inventory validates **every** original stored descriptor (name/shape/codec/absolute offset/byte length), complete bounded header digest, exact whole-source identity already authenticated before/after by original `source_check`, and exactly15 named MTP exclusions. Arbitrary skipped/missing/foreign tensors refuse.
- Actual `speculative_config is None` required. Complete320 source mappings, packed/raw payload readbacks, auxiliary codec metadata, tied head, zero learned MM/MTP parameters, native logits and all old paired transparency/lifecycle guards remain mandatory. Excluded MTP payloads are whole-file authenticated, not loaded/executed/operator-qualified.
- `loaded_roles_v4` retains the original parameter-audit and codec-audit function bodies byte-for-byte; only inventory selection replaces the erroneous stored-count assumption.
- V4 changes only ignored diagnostic scripts/manifest wiring. Native GPU math, plugin, source GGUF, tokenizer, projector and exact original teacher path remain unchanged.

ROOT commands (source-free first; no author execution):

```sh
python3 -m unittest discover -s .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020 -p 'test_loader_fixtures_v[34].py'
python3 .agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/run_loader_v4.py --model 2b --conversion /home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-conversion-2b-2 --output /home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-loader-prep-20261020/root-loader-2b-2 --gpu-uuid ROOT_ACTUAL_UUID --prepared-sha256 ROOT_REVIEWED_V4_MANIFEST_SHA
```

No benchmark, all42 state, crossengine bit parity, default graph or performance qualification follows from this repair.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/BOOTSTRAP_V2.md

Original UTF-8 bytes: 3418; SHA256 `3fe4e1b6f21913beffe0e8e1bd1943767e59f467fbb4a75e8053e19821312eec`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive V2: import boundary and actual OS caps

Original c2fe4546…32-file preparation/frozen copies remain byte-identical. No V1 container was executed; ROOT/peer source review found the startup blocker first.

- Root cause: shared `protocol.py` eagerly indexed `HERE.parents[2]`; `/probe` has no such ancestor. Both stock/native imports failed before engine start.
- `protocol_v2` imports without resolving any host repository. `host_root()` is a separate host-only operation: explicit absolute `ZG_BENCH_HOST_ROOT`, exact ownership of this run and original repository tool locations; `/probe` refuses before reading that variable. No inferred repository/alternate-root fallback. Only host launch/tool functions call it.
- All V2 entrypoints/imports point to additive V2 siblings. Shared legacy support/wheel/lifecycle helpers remain unchanged. Measurement/workload/generation/math/order/lifecycle policy is unchanged.
- Bootstrap now reads actual `/proc/self/cgroup`, cgroup2 mount identity and corresponding `memory.max`, `memory.swap.max`, `cpu.max` before Torch/native exec. Fail closed unless memory.max12GiB, additional swap0 (Podman RAM+swap total12GiB), quota exactly4CPU periods and actual scheduler affinity the same four requested IDs. Readback is bounded and retained in bootstrap; owned readiness independently validates it before accepting the provider.
- This proves actual startup kernel limits/affinity; it is not GPU byte authority, physical RAM availability, exact peak memory, post-exec per-thread affinity or inference evidence. Unsupported/mismatched cgroup layouts/refusals remain visible, without cap bypass or guessed paths.

Atomic writes: unchanged `support_v3.write` opens **only temporary `.tmp`** with `xb`, flushes/fsyncs, then replaces destination via `os.replace`. Successful replacement removes temporary pathname, so bootstrap's two writes and serialized policy updates can replace destination repeatedly. An interrupted/stale `.tmp` intentionally refuses; it does not overwrite the last valid report. New real temporary-directory fixtures cover both behaviors. No helper change needed.

ROOT only, no author execution:

```sh
export ZG_BENCH_HOST_ROOT=/home/pvl/spaces/abird/src/zerogravity
python3 -m unittest discover -s .agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020 -p '*_v2.py'
```

The23 prepared methods include the original14 schema fixtures plus9 new bootstrap fixtures. The isolated-import fixture starts a real `python3 -I` child with modules compiled at actual `/probe/...` source filenames, asserts no eager ROOT, imports shared protocol/bootstrap/policy observer without Torch/vLLM, and proves host_root refuses there. This is a source-free import/mount-layout fixture, **not** actual OCI proof. `container_import_smoke_v2.py` is also prepared for ROOT's existing owned no-GPU image runner; no engine/model/GPU/runtime import, no device exposure required.

Use fresh V2 launch/output directories and `prepare_launch_v2.py`, `run_diagnostic_v2.py`, `run_compare_v2.py`, validators suffixed `_v2`, and reviewed `prepared-manifest-v2.json` SHA. Original V1 launch specs must not be reused because they name old bootstrap. Set explicit host-root environment within ROOT's existing approved isolation wrapper; it is not passed into containers. The true mounted-container startup/OS-cap gate remains mandatory before performance/quality claims.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/BOOTSTRAP_V3.md

Original UTF-8 bytes: 5030; SHA256 `fb8c80c223eabffde6ee6257065a4061deeddb9b1f9cda17f6f5652d2bb5dc4f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive V3: opened cgroup mount identity

Author: baseline_review. ROOT alone executes fixtures, namespaces, providers and benchmarks. V1 c2fe4546 and V2 e5e70923 plus all their frozen/live source bytes remain unchanged.

## Actual cause and evidence

- V2 correctly refused ambiguous mount-path enumeration before engine import; no benchmark result follows from either failed attempt.
- ROOT's no-engine/no-GPU `root-cgroup-layout-1` showed two cgroup2 mountinfo entries at `/sys/fs/cgroup`: hidden1109 root`/../../../../../..`, visible read-only1357 root`/`. Both exact cap words were memory.max12884901888, swap.max0, cpu.max400000/100000.
- ROOT's independent `root-cgroup-layout-2` additionally opened the visible directory: actual fdinfo mnt_id1357, inode66373, st_dev23. `cgroup-layout-fixture-v3.json` copies actual proc/mount/control/FD metadata only. The fixture's four-CPU affinity and mode/read-only FD fields are explicitly source-free synthetic inputs; the original no-engine diagnostic had affinity0–15.
- `bootstrap-root-prerequisites-v3.json` pins both raw metadata logs/invocations and `root-private-loopback-exec-2.py`. No model payload, native DSO or binary was hashed by this author.

## Narrow binding

1. Read bounded actual `/proc/self/cgroup` and mountinfo; open `/sys/fs/cgroup` with O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC.
2. Read that actual directory FD's `/proc/self/fdinfo/FD`; select **exactly its mnt_id** in mountinfo, independent of list position. Require selected cgroup2, exact mountpoint/device, canonical selected root, read-only mount flags and actual fstatvfs read-only state. Hidden other mounts are retained as raw evidence, never selected by first/last order.
3. Resolve actual membership beneath the selected root. Walk canonical child components via openat directory FDs; refuse symlinks/cross-mount directories. Retain all directory FDs through readback.
4. Open each named memory/CPU control relative to the process-directory FD, O_NOFOLLOW. Check regular mode, same mount ID/device and bounded complete text. Require unchanged12GiB RAM, additional swap0, quota exactly4CPU periods and startup affinity exactly the four requested IDs.
5. Reopen the visible base path and require identical FD metadata. Readiness independently validates the complete stored proof. All owned FDs close independently, including failures; no descriptor escapes.

`server_boot_v3` requests `/output/cgroup-startup-readback.json`. Failures retain collected actual proc/stat/raw FDinfo/control data before parser refusal; evidence-I/O or cleanup failures remain fatal. Original owned nonce/CID/WNOWAIT provider cleanup still runs outside the bootstrap. No cap fallback or alternate cgroup path was added.

The selected controls establish the process's **startup local cgroup limits/affinity**, not exclusive four-CPU capacity, absence of tighter ancestors, physical available RAM, an exact peak, post-exec per-thread affinity, GPU allocation authority or inference correctness. ROOT's actual V3 startup remains mandatory.

## Unchanged execution and ROOT usage

- V3 siblings replace only V2 sibling imports, manifest selection, bootstrap pathname and FD-bound cgroup readback. Stock wheels/source, original frozen native binary, common prompt/EOS/generation policy, provider math, canonical measurement, ABBA/BAAB order, resource amounts and owned cleanup are unchanged.
- Use fresh V3 launch/output directories and `_v3.py` entrypoints, reviewed `prepared-manifest-v3.json` SHA and explicit `ZG_BENCH_HOST_ROOT` as in V2. Do not reuse V1/V2 launch JSON because it names older bootstrap/preparation files.
- ROOT's outer route is the exact custom-store `podman unshare -> unshare --net --mount --propagation private -> root-private-loopback-exec-2.py -> unchanged V3 host entrypoint`. No second single-UID user namespace, rootless egress namespace, driver/privilege/global config change or storage rewrite. This is an explicitly scoped benchmark launcher; mandated core `scripts/native-gpu.py -- ... scripts/check.sh` is unchanged. A pinned wrapper source is not proof it was invoked; retain the actual ROOT command/namespace receipts.
- ROOT fixture command: `python3 -m unittest discover -s RUN -p 'test_*_v3.py'`. New bootstrap fixtures cover actual duplicate-path layout/FD identity, reordered mount records, wrong/duplicate/missing IDs, wrong device/filesystem/read-only metadata, outside membership, unchanged cap negatives, true dirfd/no-symlink reads, partial parser failure evidence, independent FD closes, import isolation and unchanged atomic writer behavior. Original14 schema methods are copied with sibling imports only.
- `container_import_smoke_v3.py` remains import-only; no Torch/vLLM/model/GPU. No author tests/builds/engines/providers/namespaces were executed. Static AST parsing and own source/metadata hash checks only.

Hardware kernel enumeration/P22, numerical/full-state parity, source-correct goodput, stock hard5GiB device cap and whole benchmark acceptance remain UNQUALIFIED.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/EOS_COMPATIBILITY_FINDING.md

Original UTF-8 bytes: 3837; SHA256 `041a9211c255d9087f3d88baaea7dff24d8278ece12472f1c237b712179f34d8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive source finding — terminal EOS accounting

The32-file benchmark freeze c2fe4546… remains immutable; this note is not an accepted runtime change.

**Pinned-vLLM numeric compatibility requires counting sampled terminal EOS in completion usage and the returned token-ID extension. Useful decoded output remains a separate count.** The repository contract specifies useful accepted committed output, not an explicit wire-level terminal-EOS counting rule; no external OpenAI normative EOS rule is established by this review. This is a concrete mismatch against the chosen genuine vLLM baseline, not evidence of a model-math error.

Source facts:

- ZG `engine.rs:343–357`: EOS receives no TokenObservation and is never pushed into `generated`; `generated_token_count` is `generated.len()`, and Done uses that same length at396. Tests921/1011 intentionally preserve EOS-excluded behavior. HTTP usage and final observer validation use that count; this is an established policy.
- ZG `gpu_engine.rs:530–545`: `accepted_output_tokens` increases only by that non-EOS presented-count delta. It means staged committed-packet **useful non-EOS output**, not all sampled IDs or delivery acknowledgement. Resume/Ready retry avoids presenting a packet twice.
- `committed_gpu_roots` counts successful original publications (`gpu_engine.rs:647`, `cohort.rs:246`), including teacher/prefill work. A root that produced an EOS prediction is published, but EOS is never fed back. Roots/model-input steps are not generated-output counts.
- Genuine pinned `v1/core/sched/utils.py:104–109` stops after the selected EOS is already present in output IDs. `v1/engine/detokenizer.py:106–126` excludes terminal stop from detokenized text but appends the original ID to token_ids. `chat_completion/serving.py:838–845` sums those generated-ID counts into terminal completion usage; token-ID extension retains the stop ID.
- Same effective stop set does not erase this wire/accounting difference. The frozen benchmark preserves raw original usage/IDs and conservative exact-comparison results; no timed normalization or EOS suppression is permitted.

Smallest candidate, if ROOT chooses numeric vLLM compatibility:

1. Add a separate bounded `sampled_token_count` in existing GeneratedText, increment once per newly presented committed packet, including EOS. Preserve `generated` non-EOS decoded content and existing useful-output getter; no extra forward/state publication/feedback or model loop.
2. Expose sampled count in HTTP `usage.completion_tokens`/total and actual return_token_ids/observer metadata. Emit terminal EOS as an empty-text TokenText observation (or combine it with the final flush delta), followed by once-only Done. Preserve <=2 queued output events, <=512 observations, real producer root/commit time, UTF-8 flush, and Blocked/Ready retry ownership.
3. Preserve health `accepted_output_tokens` non-EOS meaning. If sampled accounting is exposed in health, add an explicitly named separate counter; do not relabel root/model-step/source-dispatch counters or retroactively reinterpret old receipts.
4. Required source-free and actual gates: CPU and native B1/2/4/8 trained EOS, EOS-first usage1/text-empty/ID[EOS], non-EOS+EOS usageN+1, length termination unchanged, stop-before-feedback, Full/Text+Done resumptions without duplicate ID/usage/counters, disconnect/Ready cancellation and exact original all42/private source state. Reconcile existing EOS-excluded tests as an explicit compatibility-policy change.
5. Keep the current accepted native server frozen for initial observations. Any candidate accounting fix needs its own reviewed/verified signed phase and a new binary/benchmark profile; it cannot alter already measured counts.

No implementation, test, provider/model/operator/GPU run was performed for this note.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/PROFILING.md

Original UTF-8 bytes: 3631; SHA256 `c0a80f2f568eef47e6dacce633766de6fa81ef6b96ae03853885215c3adaefda`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Separate real CUDA activity diagnostic — not executed

Pinned primary sources are in `bench-source-pins.json` (cached installed genuine vLLM/Torch sources; no registry assumptions):

- `vllm/entrypoints/serve/profile/api_router.py`: public POST `/start_profile` and `/stop_profile`; router attaches only with configured profiler. HTTP200 acknowledges engine calls; it proves no CUDA/CUPTI activity.
- `vllm/config/profiler.py`: public torch config, CPU/CUDA activities, trace output, no-delay/no-max/no-warmup schedule. `ignore_frontend=True` excludes frontend CPU trace duplication, not the GPU worker.
- `vllm/profiler/wrapper.py`: maps CUDA to `torch.profiler.ProfilerActivity.CUDA`; constructs genuine `torch.profiler.profile`/tensorboard Chrome handler; `_start`/`_stop` invoke original profiler. Worker base catches/logs start/stop exceptions, hence **do not accept successful HTTP alone**. `profiler='cuda'` is only a CUDA profiler range and needs an external collector; it is not a count source itself.
- Stock graph mode source: `config/compilation.py` documents FULL_AND_PIECEWISE V1 default, but actual backend resolution can differ; no unconditional graph-runtime assertion from that comment. Retain source/default runtime decisions and actual CUDA graph-launch activity separately.

`run_diagnostic.py --stage cuda-activity` starts a fresh exact stock route with the sole additive public `--profiler-config`, same output and CPU/GPU/workload/resource policy, same declared warmup, then starts profiler → invokes unchanged serve-bench → stops profiler in finally → fully closes owned process/container → audits original worker trace. Startup/compilation/capture/warmup outside this interval are excluded. All main timing samples are **unprofiled**.

The stdlib auditor accepts actual `cat='kernel'` complete Chrome activities with actual device0/stream metadata; CPU `aten`/`_C_gguf` counts are refused. Every recorded graph-replay kernel is counted individually; host cudaGraphLaunch/cudaLaunchKernel API events remain separate. Full compressed/expanded trace bounds256MiB, maximum32 artifacts, whole diagnostic tree512MiB. TP1 requires exactly one GPU-worker activity trace; no silently truncated/missing/duplicate activity file. Preserve input/output IDs and raw measurement receipts for identical-workload binding to the unprofiled population before using counts. Profiler overhead affects these diagnostic timings, which cannot enter speed ratios.

A genuine trace establishes **recorded CUDA activity**, not automatically collector losslessness/all-process coverage. Inspect native logs for CUPTI failure/dropped buffers and exact source/runtime/card/workload interval. `audit_cuda_trace.py` deliberately leaves workload binding/losslessness unqualified; ROOT closes them only with actual receipts. Total durations may overlap; summing does not equal wall time.

**Current remaining boundary:** this public Torch profiler observes its genuine vLLM worker, not an unrelated Rust process. ZG health supplies accepted source-certified dispatch-site counts; this bundle does not pretend they are independent CUDA/CUPTI enumeration. A same-workload genuine public process-level CUPTI/Nsight collector for original ZG still requires explicit verified available tooling/owned lifecycle and actual execution. No SDK download, private Torch ABI, process adoption, privilege change or profiling sysctl change is prepared here. Both providers' hardware enumeration stays **UNQUALIFIED** until that evidence exists; do not close the complete P22 kernel-count requirement from the stock CPU-op or source counters.

````

## Source: .agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/PROTOCOL.md

Original UTF-8 bytes: 11299; SHA256 `59e5449bf0825ab04189f08a5518b01f8d0f05877f9d24c02e0d977561ac35da`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Genuine stock GGUF matched serving observations — prepared only

Author: spine_review. Parent: serving-and-speculation. ROOT is sole build/model/container/GPU/benchmark coordinator. No author runtime/fixtures executed. Existing conversion/loader freezes and MAIN are unchanged.

## Inputs and exact scope

- Initial native baseline: `accepted-native-baseline.json` binds actual36,398,264B server SHA4a7455277b89aa48e92bd9d30b15a939935017368918c0e304a4f699687aae3f, ROOT source freeze d89cd668…/declared checkpoint9a19901, original14worker+12HTTP receipts. Artifact, result/log and binary identities were statically checked. Production graph admission remains closed. New row-state transport is excluded until ROOT's full physical/performance acceptance.
- Genuine vLLM0.31.0-cu129/Torch2.13/imagef57d…; stock official GGUF-plugin e2b8ad5 wheel af9de736… plus pure gguf0.19.0 wheel70bcd10e…. Ephemeral RECORD-verified site, no source/engine/weight/kernel patches. Stock public `vllm.entrypoints.openai.api_server` owns inference. No teacher/allowed-ID constraints in **any** benchmark/quality request.
- Exact existing `.8B` GGUF57d19977… and `2B`91c102fc…; identical original HF tokenizer5f9e4d49… and full config; authentic ROOT projectors6f67ce2f…/5ddf560e…. No text reconversion. ROOT accepted source-role/loader evidence is prerequisite: `.8B` V3; `2B` requires additive V4 complete335-stored/320-backbone/15-excluded-MTP audit. Projectors are stock guard inputs; no actual MM parameters/task inference is claimed.
- Weight-source quantization is identical. **Execution profiles differ:** ZG Q4_0/Q8_0→Q8_0, Q4_1→Q8_1, Q6_K→Q8_K activation preparation/reducers; stock plugin uniform Q8_1, roundf/half scales, CUDA GEMV/GEMM and BF16/F32 cast profile. No inference of bit/all42 numerical equivalence from source/format identity, same text, CPU-op events or speed.

## Common public request policy

Canonical `tools/serve-bench.py` sends one user message to `/v1/chat/completions`, temperature0/top_p1/n1/stream/include_usage/max_tokens and optional actual `return_token_ids`. `common-chat.jinja` exactly matches existing original ZG tokenizer rendering:

`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`

No automatically appended think prefix/closed-think block, no reasoning parser or tool mode. Generated think/text tokens remain genuine output; this policy does not force the model not to reason. Original HF template bytes remain authenticated but are deliberately overridden **via stock public policy flag**, explicitly for matching existing ZG policy.

`common-generation/generation_config.json` sets EOS248046 only, with neutral greedy/penalty defaults. **`--generation-config vllm` alone is insufficient:** original `InputProcessor` separately loads generation metadata and can add HF textconfig248044; an extra `stop_token_ids=[248046]` does not remove that implicit stop. Actual pinned return-frame observation must prove tokenizer EOS, generation EOS, `all_stop_token_ids=={248046}`, `ignore_eos=False`, no allowlist/min_tokens, actual prompt IDs and output cap for the real OpenAI requests. The Rust tokenizer-only helper uses original `QwenTokenizer`, never a second model engine. `validate_policy.py` binds original Rust prompt IDs to every actual returned EngineCoreRequest and HTTP provider response ID.

**Raw EOS accounting differs:** current ZG excludes sampled terminal EOS from output IDs/usage; stock vLLM retains it in token IDs/usage while omitting it from decoded text. Preserve raw per-provider IDs/counts/finish reasons. Never normalize timed throughput/goodput, ignore EOS, force minimum output or choose a special prompt to hide that difference. A separate full-output review may annotate terminal EOS without altering any timing/count report.

## Resources and runtime routes

- Both foreground containers: same immutable image,12GiB RAM and total swap allowance,4CPU quota, same exact four inside affinity IDs before runtime import, TP1/PP1, one actual pinned NVIDIA UUID/SM86, context512/max8sequences/max512batchedtokens, no speculative decoding or prefix cache.
- Native container executes **exact frozen binary**, readonly original `/nix/store`, driver/CUDA view and source mounts. This transport has a mandatory actual startup gate; no host-native/container timing mix is substituted if it fails.
- ZG real physical software admission: host4GiB/provider4GiB/device5GiB. Stock vLLM: requested4GiB provider fraction of actual worker total bytes, explicit512MiB KV, BF16 KV/conv and F32 SSM. Stock vLLM has **no hard5GiB device cap** here; actual point samples/worker evidence remain distinct from a hard admission guarantee. Record configured differences. Both actual OS cgroup memory/CPU settings must be inspected before accepting resource-control equivalence.
- `eager`: explicit stock `--enforce-eager`, controlled diagnostic/performance population.
- `stock-default`: supplies **no compilation/cudagraph overrides** and no eager flag. Retain actual resolved configuration/startup/logs; actual default graph activity needs the separate CUDA diagnostic. If stock startup/graph/resource use refuses, retain the refusal; no automatic eager/lower-context/lower-sequence/dtype/backend fallback. Eager results never establish best vLLM performance.
- Both share ROOT's approved isolated private loopback namespace (`--network=host` inside that namespace); offline model flags, proxy/redirect refusal, no downloads/privilege/recovery changes. `--network=host` itself is not an egress-isolation proof; ROOT runs under the existing native-gpu isolation wrapper.

## Workload, cold/warm and acceptance

- `solo`: exactly one offered request, concurrency1. `wave`: eight offers at t0, concurrency2/4/8. `ragged-burst`: sixteen offers, four genuinely distinct prompt lengths and staggered0/.15/.30/.45s arrivals, concurrency2/4/8. Source bytes are fixed in `protocol.workload`; do not drop excess offers or labels. Main outputs8 or32; no source context beyond512.
- Two full repetitions per population = actual ABBA then BAAB. Both cold_process and warmed_process; fresh owned process/ephemeral caches each sample. Warmed uses exactly canonical one unmeasured request. Cold means first offered request after stock readiness, **not** cold OS/storage/driver caches; no cache eviction. Stock startup warmups/graph capture are runtime behavior and must be retained.
- Predeclared TTFT10s/actual single-ID delivery ITL2s, request/launch deadlines180s, pending bound equals offered count. SSE chunk intervals never token ITL; unresolved/coalesced IDs remain unqualified. Provider aggregate mean ITL and ZG observer commit times are different clock/authority scopes. No vLLM physical commit proof is inferred from its wire metrics.
- All planned/offered/HTTP accepted429/503/rejected/cancelled/timed-out/completed requests and raw usage/IDs/text hashes/finish reasons retained by canonical tools. Local tenant labels show workload-group fairness, not authenticated server quota isolation. No failed run can be relabelled or skipped into a passing subset.
- Exact ZG output regression across repetitions/bands is required. Stock per-provider stability and crossengine exact IDs/text/usage/finish are reported separately. Full untimed response bytes/text/IDs/usage plus predeclared arithmetic/copy and code/prose/EOS tasks are in `quality_probe.py`; all outputs need independent ROOT quality disposition. Distinct numeric profiles do not imply zero-bit equality; differences cannot be hidden by throughput ratios.
- `serve-compare` retains its conservative exact-output comparison and caller-attestation warnings. The adapter changes only owned launch/cleanup and supplementary quiescent ZG counter observations; original measurement function, workload arguments and ABBA/BAAB loop remain unchanged. Default old lifecycle cleanup is replaced with exact accepted v4 WNOWAIT/independent signal/container cleanup; bounded log drain is reused. No actor/engine/kernel changes.
- Physical memory/clocks/power/thermal/driver samples include startup/warmup/measurement;0.5s sampling can miss peaks. Actual GPU compute PIDs must belong to the nonce/CID-authenticated container. No competing workload is stopped. Source-certified ZG successful dispatch sites, forwards, model-token steps, roots, accepted output IDs and row bands are distinct units; CUDA hardware counts remain unset absent real activity.
- Two order blocks are limited statistics. Keep raw paired runs; no confidence-certified superiority/engine-wide or vLLM-equivalent claim from medians. The initial report remains raw matched-source observations; P22 source-correct SLO goodput remains unset until complete numeric/functional/commit/resource/fairness/statistical gates close.

## ROOT workflow

1. Review this freeze; run `python3 -m unittest discover -s RUN -p test_protocol.py` and AST/byte compilation. No author tests ran.
2. `prepare_launch.py --model {0.8b,2b} --route {eager,stock-default} --policy {cold_process,warmed_process} --workload {solo,wave,ragged-burst} --gpu-uuid ACTUAL --gpu-total-mib ACTUAL_NVIDIA_DISPLAY --gpu-total-bytes ACTUAL_WORKER_MEM_GET_INFO_TOTAL --cpus ACTUAL_FOUR --prepared-sha256 REVIEWED_SHA --output RUN/FRESH_LAUNCH`. Latest frozen native baseline defaults are source-attested. Generator performs ROOT-only full input hashes, then emits strict protocol/commands; no provider starts.
3. Compile `tokenizer_probe.rs` only against ROOT's accepted facade/serde_json/sha2 artifacts (same rustc artifact/target compatibility as prior helpers; ROOT sole coordinator); pass original tokenizer and generated workload, save original IDs. No model/CUDA execution in that helper.
4. `run_diagnostic.py --run --stage policy --concurrency DECLARED --launch-directory … --prepared-sha256 … --output RUN/FRESH_POLICY`; run `validate_policy.py` with exact protocol/diagnostic/original Rust tokenizer output. Review actual cgroup inspect/startup/stock source-role prerequisites and actual resolved config. Run separate `--stage quality`, inspect full outputs and actual original Rust decoded-ID/text identity. No policy/quality timing is promoted.
5. ROOT writes inspected `zerogravity.stock-gguf.pre-benchmark-qualification.v1` disposition with model/route/card/protocolSHA, effective_policy_and_prompt_ids_exact, stock_320_source_roles_accepted, both_container_startups_accepted, OS_RAM_CPU_caps_observed and same_workload_functional_quality_disposition (`PASS` or explicit `UNQUALIFIED_DIFFERENT_NUMERICAL_PROFILE`). This is ROOT disposition, **not** cryptographic mathematical proof; raw performance observations may proceed with that explicit quality limitation.
6. `run_compare.py --run --launch-directory … --prepared-sha256 … --qualification-receipt … --output RUN/FRESH_MAIN`; then `validate_results.py REPORT --output …`. Repeat full matrix; keep every failed/refused case. Initial actual preparation/startup/negative fixtures still require ROOT execution; source-only fixtures do not certify this orchestration.
7. Separate `--stage cuda-activity` on identical workload/config after main timing and cleanup, as below. Do not run during benchmarks. No teacher allowed IDs, prefix surgery or source/model substitutions at any step.

````

