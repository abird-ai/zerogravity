# Run Notes — volume 4

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/numerics-and-context-bulk-causality-20261020/frozen-prepared/AUDIT.md

Original UTF-8 bytes: 9032; SHA256 `43491a40c56118ad97e300d38182b0866be18917bb3c7657637be90ec46efdc5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Fresh0.8B1024 bulk64 causality audit

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: `/root/spine_review`. Source-only preparation; root alone builds/runs.
No production/frozen-helper/capture modifications. No Cargo/C++/native/model/
operator/runtime/GPU execution by author.

## Current actual evidence

- Parent fresh AR1 controls512/1024 both originals compare730 records exactly.
- Fresh0.8B1024 prefix_batch64/ubatch64 has543 differing stages/9,230,360 words.
  First *observed* selected-token gap: layer4 incoming recurrent state171,058
  words, maxabs0.00047385692596435547. All current layer4 projection/conv/
  coefficient/q/k/v inputs and conv prior/proposal exact; current core/state/
  later layers then differ. Layers0–2 whole incoming/proposed recurrence exact;
  layer3 selected-token broad results exact. This does not identify an earlier
  causal token/operator or prove all preceding row outputs exact.
- Capture helper used original1023-prefix segmentation15×64+63, then actual
  selected token1023 as AR1. Its callback was disabled for the prefix. Last-token
  records do not observe the shape64 execution that produced the inherited gap.

## Source-derived candidates, not runtime attribution

- `llama-context.cpp:235–237` sets fused_gdn_ar/ch=true, auto_fgdn=false.
  `models/delta-net-base.cpp:373–446` dispatches both widths to the same
  GGML_OP_GATED_DELTA_NET, K=1 final snapshot. The CH label does **not** establish
  the separate chunked-FLA graph. Native CPU `ops.cpp:11107–11275` copies prior
  state then loops actual token time, scalar expf(g), vector decay, dot, delta,
  FMA update and output dot. Head/thread partitioning is outside that time loop.
- `ggml-cpu.c:1269` tries tiled quant kernels first. `tiled/tiled.cpp:605–632`
  excludes F32 src0, so this cannot supply F32 attention QK/AV.
- `ggml-cpu.c:1309–1327` tries llamafile for contiguous src1. F32 src1/KV requires
  no conversion. `llamafile/sgemm.cpp:3909–3934` rejects n<2, while F32 AVX2 n≥2
  may use tinyBLAS if its geometry gates pass. QK query views may be strided;
  do not assume contiguity. AV softmax output is contiguous. Capture actual
  ne/nb and operands instead of claiming a backend from requested ubatch.
- tinyBLAS (`sgemm.cpp:559–670`) uses one eight-lane F32 accumulator per output
  for each k+=8, then hsum. Native fallback `vec.cpp:107–135` uses four eight-lane
  accumulators, k+=32, pairwise accumulator reduction and lane reduction. These
  are distinct source rounding schedules, not proof which one produced a bit.
- Attention graph `llama-graph.cpp:2755–2808` is F32 QK→masked scaled softmax→AV.
  `ops.cpp:5813–5910` softmax runs each query row independently; `vec.cpp:531+`
  exp lanes/reduction depend on padded KV width. Native KV get_n_kv (`kv-cache
  :1284`) pads to256. Batch rows also see look-ahead cache values hidden by the
  causal mask; comparing those unmasked future QK columns as causal results
  would be wrong. Retain all raw bits, classify future columns explicitly.
- Sigmoid/SiLU/softplus and quant projection paths can also depend on actual
  spans/layout/provider. Their row inputs/results are captured. No HF helper,
  new tolerance, profile switch or fallback is proposed from this audit.

## Minimal actual continuation

1. First try **original first64-token block from zero**. It preserves the actual
   width64 arithmetic and has a common exact initial state. No prefix search is
   needed if its actual row evidence already isolates an upstream operation.
2. If first block remains exact, inspect subsequent aligned blocks0,64,…,960
   preserving64 rows, with final original block63 rows. Existing helper cut65,
   129,… can expose the original corresponding prefix boundary in its *prior*
   state; use aligned boundaries only. Do not call arbitrary-cut binary search
   causal bisection: it changes the batch shape, and bit divergence need not be
   monotone. Sequential first-boundary scan avoids that assumption.
3. At chosen aligned block, new probe computes independent native bulk/AR1
   prefixes using original segmentation. It **requires their entire serialized
   incoming state equal**. A differing inherited prior refuses and directs root
   earlier. Bulk plain64 (or63) and observed64 restore identical actual prior;
   AR1 plain/observed independently replay the same actual teacher rows.
4. Both observation paths must match their own plain path's complete post-state
   exactly. Then both plain/observed paths consume the same next actual teacher
   token as AR1; require full248320 logits and whole serialized state exact.
   Those continuation guards do not replace the captured original block.
5. Capture actual layers0–4 full rows, all projection operands/results, attention3
   Q/K/V/projection/norm/RoPE/cache operands/QK/mask/softmax/AV/gate/output, and
   GDN0/1/2/4 actual initial state, q/k/v/g/b, core rows and final proposed state.
   No exp(g) reconstruction or coefficient expected generation. Original K=1
   snapshot policy unchanged: intermediate native recurrent states unobserved.
6. Compare matching actual row slices. Preserve full raw padded caches; compare
   whole-cache operands at final AR1 frontier. Report future unmasked QK raw
   differences separately from causal positions<=query. This is exact bit
   coverage selection by actual causal mask, not tolerance/zero normalization.
7. Causal attribution still requires matching complete actual inputs and prior
   to the *specific* first differing result. Then root may replay that operator
   through existing shared engine/public native path, observe dispatch if
   necessary and derive a source-correct implementation. No production change
   or long-context/model eligibility follows automatically from this probe.

## New files and bounds

- `native_window.cpp` reuses byte-pinned old public-client helpers by textual
  include with renamed unused main; old bytes remain untouched. Actual source
  ne/nb and exported dense-view shape/slice offset are recorded separately.
- `compare_window.py` compares native bulk/AR1 actual observations only. It
  validates complete metadata/file shapes, repeated reads and byte hashes;
  records first *observed* field difference, never auto-labels a cause or Rust
  parity. Exit3=malformed/failed guard; exit0=diagnostic emitted, not parity.
- Original0.8B geometry only; teacher stream exactly8192 u32LE IDs, vocabulary
  bound248320; selected aligned start0..960; original case cut1024. CPU threads
  1..8; cooperative≤7200s deadline; original n_ctx8320/F32 KV/FAoff/CPU settings.
- ≤128MiB tensor payload+16MiB metadata/provider, field≤2M F32 words, records
  ≤16384/provider; no full graph dump/learned weight copies. Three raw snapshots
  at≤1023 causal tokens add<136MiB; two full-next-logit files<2MiB. Combined
  artifacts<512MiB excluding bounded external process logs. Root applies finite
  actual process RAM/CPU/wall/log caps. Four contexts share one original model;
  these bounds are not ResourceArbiter/physical RAM qualification.
- Completion file is written only after all transparency guards. Failure keeps
  an incomplete exclusive directory and root's failure log. No existing output
  is overwritten; no batch width/backend/operator/privilege is changed.

## Root-only preparation/execution

Root must verify `prepared-manifest.json`/`source-pins.json`, original model SHA
57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf and common raw-ID
SHA44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d independently.
Use same compiler/include/link/glibc closure as existing native_raw; freeze/hash
the fresh executable and retain actual argv/library/source receipt. No downloads.

```sh
# ROOT ONLY; existing root supervisor enforces finite process/resource/log caps.
probe=.agents/runs/numerics-and-context-bulk-causality-20261020
src=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
lib=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
glibc=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
c++ -std=c++17 -O2 -Wall -Wextra -Werror -I "$src/include" -I "$src/ggml/include" \
  "$probe/native_window.cpp" -L "$lib" -L "$glibc" \
  -Wl,-rpath,"$lib:$glibc" -Wl,--dynamic-linker,"$glibc/ld-linux-x86-64.so.2" \
  -lllama -lggml -lggml-base -lm -o "$probe/native_window"
# ROOT review first, then run one original first64-row block only.
"$probe/native_window" target/models/qwen35-0.8b/model.gguf \
  .agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le \
  0 8 900 "$probe/root-first64"
python3 "$probe/compare_window.py" --capture "$probe/root-first64" \
  --output "$probe/root-first64-comparison.json"
```

Do not automatically launch the complete block ladder or call these timings a
serving benchmark. Instrumentation controls establish numerical transparency,
not performance transparency or internal microkernel observation.

````

## Source: .agents/runs/numerics-and-context-bulk-causality-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1003; SHA256 `15770ec049e7d657a9192b0c90449f1be60ac9b409c8283938bb5b77e3bdc6bb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: `/root/spine_review`; root owns execution/shared-plan reconciliation.

- [x] Audit current1024 bulk64 report and pinned graph/CPU dispatch source.
- [x] Separate inherited layer4 prior-state difference from causal attribution.
- [x] Preserve original15×64+63; document why arbitrary-cut binary search changes
  source arithmetic and cannot identify the original first divergent row.
- [x] Prepare source-only unchanged-block/native-AR1 observer + bit comparator;
  all input/result rows and independent state/next-logit transparency guards.
- [ ] Root/independent review, compile and actual first64-row bounded control.
- [ ] Actual first causal row/operator with complete equal inputs/prior.
- [ ] Source-correct existing-engine fix; actual operator/full-state/long gates.

No native/model/operator/runtime/Cargo/C++compile/GPU execution by author.
Original frozen helpers/captures and all production source unchanged.

````

## Source: .agents/runs/numerics-and-context-bulk-causality-20261020/frozen-span-coverage/AUDIT.md

Original UTF-8 bytes: 9196; SHA256 `87a2ccef1168264e9fd95ecb35a45563210f0b49f49e8c98bcbff26b60fa80f8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Fresh0.8B1024 bulk64 causality audit

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: `/root/spine_review`. Source-only preparation; root alone builds/runs.
No production/frozen-helper/capture modifications. No Cargo/C++/native/model/
operator/runtime/GPU execution by author.

First64 root observation refused the old16MiB backing-span cap before completion;
see `SPAN_FIX.md` for preserved failure and exact V-cache logical-run follow-up.

## Current actual evidence

- Parent fresh AR1 controls512/1024 both originals compare730 records exactly.
- Fresh0.8B1024 prefix_batch64/ubatch64 has543 differing stages/9,230,360 words.
  First *observed* selected-token gap: layer4 incoming recurrent state171,058
  words, maxabs0.00047385692596435547. All current layer4 projection/conv/
  coefficient/q/k/v inputs and conv prior/proposal exact; current core/state/
  later layers then differ. Layers0–2 whole incoming/proposed recurrence exact;
  layer3 selected-token broad results exact. This does not identify an earlier
  causal token/operator or prove all preceding row outputs exact.
- Capture helper used original1023-prefix segmentation15×64+63, then actual
  selected token1023 as AR1. Its callback was disabled for the prefix. Last-token
  records do not observe the shape64 execution that produced the inherited gap.

## Source-derived candidates, not runtime attribution

- `llama-context.cpp:235–237` sets fused_gdn_ar/ch=true, auto_fgdn=false.
  `models/delta-net-base.cpp:373–446` dispatches both widths to the same
  GGML_OP_GATED_DELTA_NET, K=1 final snapshot. The CH label does **not** establish
  the separate chunked-FLA graph. Native CPU `ops.cpp:11107–11275` copies prior
  state then loops actual token time, scalar expf(g), vector decay, dot, delta,
  FMA update and output dot. Head/thread partitioning is outside that time loop.
- `ggml-cpu.c:1269` tries tiled quant kernels first. `tiled/tiled.cpp:605–632`
  excludes F32 src0, so this cannot supply F32 attention QK/AV.
- `ggml-cpu.c:1309–1327` tries llamafile for contiguous src1. F32 src1/KV requires
  no conversion. `llamafile/sgemm.cpp:3909–3934` rejects n<2, while F32 AVX2 n≥2
  may use tinyBLAS if its geometry gates pass. QK query views may be strided;
  do not assume contiguity. AV softmax output is contiguous. Capture actual
  ne/nb and operands instead of claiming a backend from requested ubatch.
- tinyBLAS (`sgemm.cpp:559–670`) uses one eight-lane F32 accumulator per output
  for each k+=8, then hsum. Native fallback `vec.cpp:107–135` uses four eight-lane
  accumulators, k+=32, pairwise accumulator reduction and lane reduction. These
  are distinct source rounding schedules, not proof which one produced a bit.
- Attention graph `llama-graph.cpp:2755–2808` is F32 QK→masked scaled softmax→AV.
  `ops.cpp:5813–5910` softmax runs each query row independently; `vec.cpp:531+`
  exp lanes/reduction depend on padded KV width. Native KV get_n_kv (`kv-cache
  :1284`) pads to256. Batch rows also see look-ahead cache values hidden by the
  causal mask; comparing those unmasked future QK columns as causal results
  would be wrong. Retain all raw bits, classify future columns explicitly.
- Sigmoid/SiLU/softplus and quant projection paths can also depend on actual
  spans/layout/provider. Their row inputs/results are captured. No HF helper,
  new tolerance, profile switch or fallback is proposed from this audit.

## Minimal actual continuation

1. First try **original first64-token block from zero**. It preserves the actual
   width64 arithmetic and has a common exact initial state. No prefix search is
   needed if its actual row evidence already isolates an upstream operation.
2. If first block remains exact, inspect subsequent aligned blocks0,64,…,960
   preserving64 rows, with final original block63 rows. Existing helper cut65,
   129,… can expose the original corresponding prefix boundary in its *prior*
   state; use aligned boundaries only. Do not call arbitrary-cut binary search
   causal bisection: it changes the batch shape, and bit divergence need not be
   monotone. Sequential first-boundary scan avoids that assumption.
3. At chosen aligned block, new probe computes independent native bulk/AR1
   prefixes using original segmentation. It **requires their entire serialized
   incoming state equal**. A differing inherited prior refuses and directs root
   earlier. Bulk plain64 (or63) and observed64 restore identical actual prior;
   AR1 plain/observed independently replay the same actual teacher rows.
4. Both observation paths must match their own plain path's complete post-state
   exactly. Then both plain/observed paths consume the same next actual teacher
   token as AR1; require full248320 logits and whole serialized state exact.
   Those continuation guards do not replace the captured original block.
5. Capture actual layers0–4 full rows, all projection operands/results, attention3
   Q/K/V/projection/norm/RoPE/cache operands/QK/mask/softmax/AV/gate/output, and
   GDN0/1/2/4 actual initial state, q/k/v/g/b, core rows and final proposed state.
   No exp(g) reconstruction or coefficient expected generation. Original K=1
   snapshot policy unchanged: intermediate native recurrent states unobserved.
6. Compare matching actual row slices. Preserve full raw padded caches; compare
   whole-cache operands at final AR1 frontier. Report future unmasked QK raw
   differences separately from causal positions<=query. This is exact bit
   coverage selection by actual causal mask, not tolerance/zero normalization.
7. Causal attribution still requires matching complete actual inputs and prior
   to the *specific* first differing result. Then root may replay that operator
   through existing shared engine/public native path, observe dispatch if
   necessary and derive a source-correct implementation. No production change
   or long-context/model eligibility follows automatically from this probe.

## New files and bounds

- `native_window.cpp` reuses byte-pinned old public-client helpers by textual
  include with renamed unused main; old bytes remain untouched. Actual source
  ne/nb and exported dense-view shape/slice offset are recorded separately.
- `compare_window.py` compares native bulk/AR1 actual observations only. It
  validates complete metadata/file shapes, repeated reads and byte hashes;
  records first *observed* field difference, never auto-labels a cause or Rust
  parity. Exit3=malformed/failed guard; exit0=diagnostic emitted, not parity.
- Original0.8B geometry only; teacher stream exactly8192 u32LE IDs, vocabulary
  bound248320; selected aligned start0..960; original case cut1024. CPU threads
  1..8; cooperative≤7200s deadline; original n_ctx8320/F32 KV/FAoff/CPU settings.
- ≤128MiB tensor payload+16MiB metadata/provider, field≤2M F32 words, records
  ≤16384/provider; no full graph dump/learned weight copies. Three raw snapshots
  at≤1023 causal tokens add<136MiB; two full-next-logit files<2MiB. Combined
  artifacts<512MiB excluding bounded external process logs. Root applies finite
  actual process RAM/CPU/wall/log caps. Four contexts share one original model;
  these bounds are not ResourceArbiter/physical RAM qualification.
- Completion file is written only after all transparency guards. Failure keeps
  an incomplete exclusive directory and root's failure log. No existing output
  is overwritten; no batch width/backend/operator/privilege is changed.

## Root-only preparation/execution

Root must verify `prepared-manifest.json`/`source-pins.json`, original model SHA
57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf and common raw-ID
SHA44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d independently.
Use same compiler/include/link/glibc closure as existing native_raw; freeze/hash
the fresh executable and retain actual argv/library/source receipt. No downloads.

```sh
# ROOT ONLY; existing root supervisor enforces finite process/resource/log caps.
probe=.agents/runs/numerics-and-context-bulk-causality-20261020
src=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
lib=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
glibc=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
c++ -std=c++17 -O2 -Wall -Wextra -Werror -I "$src/include" -I "$src/ggml/include" \
  "$probe/native_window.cpp" -L "$lib" -L "$glibc" \
  -Wl,-rpath,"$lib:$glibc" -Wl,--dynamic-linker,"$glibc/ld-linux-x86-64.so.2" \
  -lllama -lggml -lggml-base -lm -o "$probe/native_window"
# ROOT review first, then run one original first64-row block only.
"$probe/native_window" target/models/qwen35-0.8b/model.gguf \
  .agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le \
  0 8 900 "$probe/root-first64"
python3 "$probe/compare_window.py" --capture "$probe/root-first64" \
  --output "$probe/root-first64-comparison.json"
```

Do not automatically launch the complete block ladder or call these timings a
serving benchmark. Instrumentation controls establish numerical transparency,
not performance transparency or internal microkernel observation.

````

## Source: .agents/runs/numerics-and-context-bulk-causality-20261020/frozen-span-coverage/SPAN_FIX.md

Original UTF-8 bytes: 3046; SHA256 `407900e18043551c033aeb8fb6afd601acb6535b445c2222394f115e4959f2dc`.
Resolve embedded relative paths against the original path, not this volume.

````text
# First64 observation span refusal: narrow follow-up

Initial frozen manifest564ab0e0/helper639fc525 and root-first64 output remain
unchanged. Root compiled that helper strictly; four source-free comparator
fixtures passed. Actual run exited3 before transparency completion, therefore
establishes no accepted native window/model inference result.

- Last completed field152: actual `kqv-3`, ne[256,64,8,1]. Next field reads its
  V-cache input. Previously inherited `dense()` refused any backing span>16MiB.
- Pinned `llama-context.cpp:292` pads requested8320 context to256→8448. Actual
  common contexts and earlier captured K-cache nb3=17301504 independently agree.
- `llama-kv-cache.cpp:1329/:1514`: transposed V storage is
  `[capacity,head*dimension]`; view+permute gives ne[visibleKV,256,2,1] with
  nb[4,8448*4,8448*256*4,8448*512*4]. At visibleKV256, exact minimum backing
  span is `((512-1)*8448+256)*4=17268736B`, while logical data is524288B.
  `ggml.c:1298` computes exactly4+sum((ne-1)*nb). No larger tensor was requested.

`window_dense()` is local to this new diagnostic only:

- Existing native_raw.cpp/dense() bytes and all frozen/captured attempts unchanged.
- F32 innerstride4; positive checked dimensions/product≤2M words; checked stride
  products/span additions; computed span must equal actual public ggml_nbytes.
- Ordinary spans retain≤16MiB policy. The only larger accepted span is exactly
  the named attention3 transposed V-cache operand with actual context8448,
  visibleKV256/512/768/1024, heads2×dim256 and exact above strides/formula.
  Its range remains inside the actual full cache allocation17301504B.
- Allocate only logical≤8MiB result. Public `ggml_backend_tensor_get` copies each
  inner contiguous run into old dense order; read offsets/counts checked against
  the exact tensor span, no capacity-gap read/copy and no scalar math operation.
- Refusals now print bounded call/site/name/op/ne/nb/context/logical/span details.
  Original exact full-state/next-token guards, limits and source math unchanged.

Independent review also found mandatory-field coverage should be explicit.
New C++ checks each actual call; comparator independently requires every call's
embedding, first5-layer broad boundaries, exact original projection role
inputs/outputs, attention3 QK/mask/softmax/AV/cache inputs and all GDN0/1/2/4
q/k/v/g/b/core plus initial/final states at their actual boundaries. No omitted
attention field may silently reduce scope. Runtime callback labels still must
match actual source; no guessed expected value or numerical exception.

Conv source history/kernel inputs remain unobserved. A convolution-result gap
would still need a narrow actual operand capture before causal attribution.
The comparator stays diagnostic-only and never auto-promotes a cause/profile.

Author performed source/hash/AST checks only. Root independently reviews,
compiles a fresh binary and runs a **new** exclusive attempt directory; no
native/model/operator/runtime/Cargo/C++compile/GPU execution by author.

````

## Source: .agents/runs/numerics-and-context-bulk-causality-20261020/frozen-span-coverage/TODO.md

Original UTF-8 bytes: 1298; SHA256 `bd53e75b091c2ef655fa42d4a1e0d1dceb989c6d0dba4b02d71b3f4e313fc59e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: `/root/spine_review`; root owns execution/shared-plan reconciliation.

- [x] Audit current1024 bulk64 report and pinned graph/CPU dispatch source.
- [x] Separate inherited layer4 prior-state difference from causal attribution.
- [x] Preserve original15×64+63; document why arbitrary-cut binary search changes
  source arithmetic and cannot identify the original first divergent row.
- [x] Prepare source-only unchanged-block/native-AR1 observer + bit comparator;
  all input/result rows and independent state/next-logit transparency guards.
- [x] Preserve root first64 span refusal; prepare local logical-run V-cache
  copier with exact context/head/stride/span guards and refusal context.
- [x] Require projection/attention/GDN semantic families per actual call in
  both emitter and comparator; no silently partial observation scope.
- [ ] Root/independent review, compile and actual first64-row bounded control.
- [ ] Actual first causal row/operator with complete equal inputs/prior.
- [ ] Source-correct existing-engine fix; actual operator/full-state/long gates.

No native/model/operator/runtime/Cargo/C++compile/GPU execution by author.
Original frozen helpers/captures and all production source unchanged.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-integration-20261020/INTEGRATION.md

Original UTF-8 bytes: 10682; SHA256 `b8d0d51590af05b28886a62566c61b6c41f47b6837c63a2bc14c2a93495a8e98`.
Resolve embedded relative paths against the original path, not this volume.

````text
# GGUF temporal bulk integration — source-only proposal

Parent: numerics-and-context. Owner: baseline_review. ROOT alone reviews/merges/runs.
No production, default, numerical, session, checkpoint or GPU change is proposed
for immediate activation. The frozen primitive689221fe and attributiona7e8cc3b
remain unchanged, uncompiled and unqualified. `HaswellTinyBlasF32V1` below is a
proposed core contract name, not a claim that MAIN exposes/qualifies it; the
candidate currently calls its enum GgmlD812350HaswellTinyBlas.

## Conclusion

The next useful code cut is a **sealed temporal matrix operation plus explicit
span shape/state effects in the existing engine**. There is no real CPU bulk
ForwardPlan callsite to switch today. Implement/qualify the generic closed
matrix slice first; extend the SAME forward schedule to a bounded teacher span
only afterward. Do not change dot_f32_source_reduction or source_attention_value
globally, label the current token loop bulk, or use independent GPU cohort B as
native temporal N. Native-default long-context closure remains open.

## Actual seams and missing contracts

- model lib.rs1175–1215: prefill_transaction_inner takes one RootAttempt(N),
  clones a private draft, then calls forward_token N times, publishing only at
  the end. This is a valid atomic transaction, with one-token arithmetic.
- forward.rs22–36/217–277: one run_layer_sequence owns all layers. Retain it.
  forward_token embeds one ID; project/expect_elements/attention/GDN shapes are
  vectors; advance_frontier advances one token. No duplicate model/token loop.
- core program.rs185,407–417,460–552,1321–1516: scalar input parameters,
  Vector/CausalRows values and one KV append/version per domain cannot describe
  temporal queries with per-row causal lengths or a whole-block KV proposal.
- host_math.rs540–545/850–917: project calls the existing one-input GEMV;
  QK/AV see a causal prefix of token slabs, and AV pads that prefix to256.
  lib.rs1305–1331/1423–1471 and cpu weights.rs592–634 use four Vec8 partials.
- core accelerator.rs64–76/313–365/465–500: MatVecRegion, AttentionQkRegion,
  AttentionValueRegion encode one temporal row. Their provider methods cannot
  receive a new bulk numerical contract without an explicit new seam.
- ProgramNumericalProfile/ProjectionContract currently expose only source-v1
  and the distinct closed HF CPU profile. HostMath legacy_precision refuses
  other profiles; preserve that refusal. HF operators are unrelated to this
  GGUF Haswell contract.

## Exact first-gap bindings

| Authentic native operation | A ne | B ne | output ne | Existing mismatch |
|---|---|---|---|---|
| .8B layer3 AV | [256,256,2,1] | [256,64,8,1] | [256,64,8,1] | first observed pos2,530/2048 words |
| 2B layer0 alpha F32 | [2048,16,1,1] | [2048,64,1,1] | [16,64,1,1] | first observed pos0,13/16 words |

AV A's actual byte strides are [4,33792,8650752,17301504] with17,268,736B
used span and capacity8448. P is contiguous. Alpha's unchanged raw GGUF F32
weight is131072B; X is the actual normalized2048x64 input. All default replay
outputs and compiled first-entry receipts already exist independently of the
candidate; they must remain expected values. No candidate-generated goldens.

Pinned ggml-cpu.c1308–1333 selects tinyBLAS only when use_ref is false,
RHS contiguous and the real public kernel accepts the operation. sgemm.cpp3908
rejects n<2;559–617 requires an accepted M tile;627–670 uses one Vec8 FMA chain
and252–270 the even/odd hsum. A physical ubatch limit64 does not mean every
actual graph has N64: partial chunks, the final AR1 observation, selected output
rows and noncontiguous operands have their own real shapes/routes.

## Smallest implementation cuts, in order

1. **Primitive gate (ROOT first).** Apply original candidate plus attribution
   follow-up, source-free strict tests and actual unchanged default AV/alpha
   replay. This only admits the closed operator/profile and bounded geometry.
2. **Generic sealed matrix slice (core/CPU/model seam).** Add a closed
   F32 matrix contract under existing MathematicalBackend/PlanOperation and a
   read-only operator slice evaluated by the SAME PlanBuilder and HostMath.
   Constructor privately binds source owner + operand roles, full ne/nb,
   offset/backing, actual source kind, F32 activation precision, causal/span
   origin and numerical contract. SourceTensor.validate_source validates
   learned rank2 raw F32 bytes; dynamic V/P/Q bindings retain original state
   domain version/issuer and exact buffer range. F32View names+SourceIdentity
   alone are NOT that seal. No fake TensorSource/partial learned descriptor,
   metadata dtype retag or silently dequantized matrix.
   A new backend/provider method defaults to Unsupported before any callback;
   an existing MatVecKernel or GPU backend must not execute it as Source.
   Contract refusal/error is fatal, not an invitation to old arithmetic.
3. **Temporal span in the SAME family schedule (core/model/HostMath).** Factor
   forward_token into one span-aware schedule with the existing token method
   as span1 wrapper. Keep run_layer_sequence and role selection single-owned.
   Add temporal rows/causal-matrix shapes and immutable ForwardSpanInputs:
   source, profile/route version, source context capacity, base frontier,
   actual consecutive teacher IDs/positions, per-query valid lengths,
   actual N, output-row selection and KV padding/window. First private scope:
   one sequence, fixed full N64, CPU/F32 KV/FA off, no serving admission.
   Head/conv/GDN work may iterate teacher positions INSIDE their closed math
   operators in original order; no backend acquires a Qwen layer/token loop.
   Bulk attention appends all actual block K/V to the private candidate before
   QK/mask/softmax/AV, retaining per-row causal masks and future value words.
   Never replace masked future V by +0 or shorten the reduction because its
   P is +0 without an independent bit proof, including signed zero. Raw
   canonical token slabs may be explicitly packed/transposed into leased
   workspaces; retain both original provenance geometry and actual packed
   geometry instead of claiming native strides describe a new allocation.
   Preserve source GGUF conv, norm, sigmoid/softplus, coefficient and GDN
   recurrence; bind actual native fused_GDN choices rather than infer chunk
   math from ubatch. No HF FLA/SLEEF substitution.
4. **Full default route closure, not name-based overrides.** Route all actual
   eligible F32 project/QK/AV nodes through explicit contracts, not only names
   ssm_alpha/attn_value. Keep quantized row kernels/activation encoding exactly
   as accepted until their own native bulk route is established. Add a
   pre-execution closed route table covering EVERY node shape/kind/provider;
   reject unknown routes before allocation/state effects. Only a proven
   non-tinyBLAS source row route uses existing four-Vec8 math. Do not fabricate
   a tinyBLAS n=1 exception to execute a singleton/tail.

A real additional default2B blocker already exists: V7 refused actual BLAS
MUL_MAT linear_attn_out-0, src0 Q8_0[2048,2048], src1 F32[2048,64]. Pinned
BLAS supports_op396–432 can accept large contiguous quantized matrices;
59–145 dequantizes weights to F32 then invokes cblas_sgemm. The existing Rust
Q8_0 projection dynamically quantizes its activation and executes packed integer
dots. Neither a TinyBLAS F32 change nor the successful CPU_Mapped reference
control implements this default BLAS arithmetic. Keep that route refused until
separately source-derived/qualified; do not unload/disable BLAS, change buffer
placement or call such a control default. Primitive success cannot close it.

## Admission and ownership seals before fullmodel use

- Candidate caps: K<=8192,M<=4096,2<=N<=64,logical operands<=1M words,
  backing<=8M words,outputs<=1M,work<=128Mi FMAs. At K8192 full AV logical A
  has4M words and full B4M words; work is1,073,741,824 FMAs. An unsplit call
  MUST refuse. No unchecked increase or inner-K partial reduction.
- A reviewed tile scheme can split independent output channels/heads while
  preserving each complete K chain. Example M64/query-head1/N64 at K8192:
  A524288 words,B524288 words,output4096 words,33,554,432 FMAs. Four channel
  tiles per query head,8 query heads. RHS head packs are explicit exact copies
  with original full operand provenance; do not falsify contiguous stride
  metadata. Root must prove all tile boundaries/output coverage/default bits.
  QK can similarly tile output key rows; K is still the full head width256.
- Admit aggregate staging + packed operands + full output + tile output/scratch
  BEFORE allocating each payload; retain original shared ResourceArbiter
  tickets. Current HostStorage Vec cannot absorb BulkF32Completed without
  dropping its lease/copying: add a private retained backing variant or an
  explicitly charged copy, preserve views/clones, deny bare Vec escape for
  leased results as the existing window path does. No full host-RAM claim.
- Reuse RootAttempt(N), independent SessionAuthority and private COW state;
  verify all required42 domains and finite outputs before one publication.
  Cancellation, bad token, pressure, provider refusal or later-layer failure
  leaves old root/all domains unchanged and drops staging/output tickets.
  CPU is synchronous; do not invent Ready events or GPU completion. GPU graph,
  queue, original leases and unknown quarantine remain unchanged/cold-refuse.
- Bind profile/span provenance to session/drafts/snapshots/plans. SourceIdentity
  stays the SAME immutable source owner, not an invented numerical identity.
  Existing QwenState/QwenCheckpoint v1 only bind source/frontier and cannot
  express the new family/route/span contract. Keep v1 export/import refused for
  new-profile state until a reviewed explicit compatible encoding/binding
  exists; never reinterpret current v1 bytes. Old GGUF v1 remains unchanged.

## Proposed disjoint ownership

- core: new closed bulk contract/temporal shape/state validation; parent wiring.
- CPU: original primitive adapter/raw F32 operand/tile transport; no row math edits.
- model forward/HostMath: SAME role/layer schedule span factor, sealed operands,
  leased backing, original root publication and focused source-free tests.
- ROOT: profile/checkpoint policy, GPU cold refusal, composition and ALL runs.

NEXT permission should cover cuts1–2 first. Cut3 requires a reviewed span/API
patch and default native route/readout evidence; cut4 BLAS/quantized additions
must not be implied by a numerical F32 primitive qualification. See ROOT-GATES.md.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-integration-20261020/ROOT-GATES.md

Original UTF-8 bytes: 5972; SHA256 `ac4d6e32de3032fb1434a4e1be8938eac3eafb4c29326bfd78422e589e04a513`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only default-bulk qualification gates

Source proposal only; no command below was executed/prepared as a passing test.
Use existing native_raw/causal-window/public replay/Rust export comparators,
source identities and SessionAuthority; no independent model/token engine.
Retain all previous native/default/reference-control/AR1 failures/successes.

1. **Build gate before model work.** Reviewed exact primitive+Mozilla notice,
   focused adversarial Rust tests, strict CPU/core/model Clippy, unaware-provider
   zero-callback and resident/GPU cold refusal, complete mandated isolated check.
   Real default AV/alpha payload hashes/replay performed only by ROOT. Candidate
   must match132096 authentic default words, pressure/pre/mid cancellation,
   retained outputs after borrowed input release and final ledger0. Any mismatch
   is failure, never tolerance/zero-sign normalization.
2. **Generic plan-fragment gate.** Record/evaluate real sealed F32 alpha and AV
   via SAME PlanBuilder/HostMath callback paths. Check contract in recorded node,
   exact source full metadata/raw source owner and input provenance; every output
   bit vs independent unchanged public default. Wrong profile/source/dtype/rank,
   role/range/stride/overlap/overflow, stale state version, foreign issuer, bad
   extent/head broadcast, N1 and unsupported provider refuse before callbacks or
   state writes. Canary gaps/nonzero offsets/tail/signed-zero remain meaningful.
   Leased output/view clones survive input/root scope only under intended owner;
   pressure/cancel returns every new charge, no uncharged Vec or hidden fallback.
3. **Native-default first64 gates, BOTH original GGUF models.** Use unchanged
   raw first64 IDs and exact incoming native/Rust state. Native CPU8/F32 KV,
   FlashAttention off, exact source/provider closure and actual temporal block64;
   capture all existing selected rows/early layers. Plain→restore same prior→
   observed must preserve FULL logits and whole serialized post-state; observer
   requests alone are not evidence. .8B all131072 probabilities and1,064,960
   causal V words must match before comparing complete layer3 AV131072 words;
   include all masked-future P signs and actual future V/padding. 2B actual
   normalized131072 input words/full32768 raw F32 alpha weight words must match
   before all1024 alpha outputs. Then full all64 early-layer trajectories,
   beta/gain/GDN states and ALL42 block-end domains/full248320 logits.
   Native state export must independently decode six KV +18conv +18recurrent,
   not compare only GDN or infer all42 from final logits.
4. **Full default route/refusal gate.** Log actual node population, real m/n/k,
   ne/nb/dtype, contiguous predicate, selected arithmetic and output-row count.
   Include final norm/head readout whose N can differ from prefix N. Explicitly
   retain native BLAS route evidence; candidate must either implement a separately
   qualified exact contract or refuse BEFORE publishing a block. An explicit
   CPU_Mapped placement/use_ref run is a separately labeled reference control;
   it cannot supply expected default results or default pass status.
5. **Boundaries with original numerical schedule preserved.** Independent native
   runs at N1/2/3/31/32/63/64 and block boundaries63/64/65,255/256/257,511/512/513,
   then1024/2048/4096/8192. First support only fixed64 and its declared final
   AR1 observer; other widths remain refused until their actual source routes
   qualify. Compare equal actual block partitions, causal lengths/Kpadding,
   context8448 and output-row selection. Do NOT assert different partitions
   have equal bits: each native route is its own oracle. Never change the
   actual reduction K when bounding work; full-K output tiles must match an
   independently captured unsplit default operator. At8K enforce checked tile
   coverage and aggregate admission, not the primitive's old small call budget.
6. **Original wholemodel ladder.** Reuse fresh-prefix-cases.json raw IDs and
   same pinned llama0.6.0/d812350 closure, CPU8/F32 KV/FA off, batch/ubatch64.
   Existing native_raw warm loop processes cut-1, then observes one AR1 token;
   record ACTUAL prefix graphs (e.g.511=7x64+63), not fictitious64 at every step.
   Compare full730 selected stage records/full248320 logits AND every42-domain
   active state word at512/1024/2048/4096/8192;8K scope55,382,016 state words/model.
   No partial-top20 surrogate, approximate norm or signed-zero exception. First
   differing operator stops acceptance and becomes next source-derived cut.
   Only after64 passes consider native256/512 microbatch populations: candidate
   currently N<=64 and scheduling64 inside512 is a different source schedule.
7. **Publication/replay regressions.** Invalid last row, deterministic cancellation
   after actual completed math but before publication, pressure on a later layer,
   wrong profile/stale/foreign snapshot, context exhaustion, nonfinite/short
   provider output all preserve old root/all42 and old checkpoint bytes. Success
   publishes exactly one RootAttempt(N), correct frontier and no prefix generated
   token/EOS/output usage. Trained teacher EOS is ordinary input. Snapshot/restore
   preserves same-profile timeline/epochs; do not accept untagged new-profile v1
   checkpoints. Rerun accepted short corpus CPU/B1/native B2/4/8 fullV/all42,
   cancellation/Ready/quarantine/resource final0 and both original parity routes.
8. **Independent claims.** Keep separately: existing AR1/reference-control8K
   success; new operator-only TinyBLAS parity; new default64 wholemodel parity;
   default256/512; historical input replay. No old240549 artifact encoding was
   recovered; fresh raw-token/JSON digest mismatch alone proves no different
   token history. Preserve original0.45/0.05 functional limits while internal
   operator/state gates stay bit-exact. This proposal qualifies nothing and
   makes no timing/throughput claim.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-integration-20261020/TODO.md

Original UTF-8 bytes: 725; SHA256 `7a1d47f62f5908bdbfc22c93843b65ec417cb4e3d34dd1c22aee0a0ae963ab13`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working scope

Parent: ../../plans/numerics-and-context/TODO.md. Owner: baseline_review.

- [x] Read current source/plan and bounded independent default replay metadata.
- [x] Trace one-token CPU transaction, same ForwardPlan/HostMath, provider/profile
  and checkpoint boundaries; identify missing temporal contracts/8K bounds.
- [x] Prepare narrow staged integration and full default-control replay gates.
- [ ] ROOT review/authorization of any implementation; primitive remains unqualified.

No MAIN/shared-plan edits, builds/tests/operators/providers/model/GPU runtime,
model/DSO payload hashing or numerical/default/profile activation performed.
Original primitive/candidates/failures unchanged. Source-only evidence.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/README.md

Original UTF-8 bytes: 7558; SHA256 `afa11f70d6b06d6838b23f2a05bca2e221f3a1d8b38fa080a7a3d5851a2dce90`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed Haswell bulk-F32 primitive — source preparation only

Parent: numerics-and-context. Owner: baseline_review. ROOT alone builds/runs.
No MAIN/shared-plan edits, compiler/test/operator/provider/model/GPU execution.

The proposed CPU math child implements ONE explicit numerical primitive,
`GgmlD812350HaswellTinyBlas`, without selecting it for any production projection,
attention, model, session, ForwardPlan or checkpoint. Existing four-Vec8 GGUF
paths and all current core/model numerical profiles remain byte unchanged.
No foreign library/model/token loop or second engine is introduced.

## Exact source arithmetic

Pinned original llama.cpp0.6.0/d812350 `ggml/src/ggml-cpu/llamafile/sgemm.cpp`:

-252–270: Vec8 low+high, then the actual `movehl`/`movehdup` tree:
  `(low0+low2)+(low1+low3)`. This is NOT vec-dot's adjacent-pair tree.
-407–409,627–670: one Vec8 +0 accumulator/output, ordered FMA every8 elements;
  both operands in a partial final Vec8 are zero-filled and ALL8 FMAs execute.
-3908–3934,559–617: actual Haswell F32 path refuses n<2; m divisible4 is a
  supported closed tile population. Scalar output iteration preserves each
  original output's FMA order; it does not claim native tile/thread scheduling.
- Full original ggml MIT notice is retained in the new math module.

Prepared source-free tests isolate one-chain versus four-chain cancellation,
actual even/odd horizontal order, subnormal/signed-zero tail padding, nonzero
views/NaN gap canaries/broadcast, source/stride/span/overflow/AR1 refusal,
nonfinite output refusal, actual capacity, pre-cancel and deterministic
mid-output cancellation. No test has been executed by the author.

## API and resource boundary

`math::haswell_bulk_f32::{BulkF32Profile,F32View,BulkF32Request,
BulkF32Admission,BulkF32Completed,execute}`.

- Borrowed F32 views retain original4D `[inner,row,head,sequence]` geometry,
  checked strides/offset/full backing extent, same nonzero SourceIdentity and
  bounded names. RHS must be genuinely contiguous; left may retain V strides.
- SourceIdentity is a local allocation identity, NOT a model digest/issuer.
  This low-level view cannot authenticate learned bytes or grant model support.
  Caller must keep original input owners/admission for synchronous execution.
  ROOT fixture transport separately verifies original capture/source provenance.
- K<=8192,m<=4096,2<=n<=64; <=1M logical words/operand/output, <=32MiB physical
  backing/operand, <=128Mi ordered FMAs/call. Checked products before access.
  Nonfinite used words refuse; unobserved gap words are never read by math.
- Existing supplied host ResourceArbiter admits output payload +256B numerical
  scratch BEFORE output allocation. Private result retains that original ticket;
  no public Vec escape/Clone. Payload drops before lease. Synchronous CPU work
  has no submitted GPU event/unknown completion to reinterpret.
- These are logical payload/scratch charges, not allocator/kernel/entire physical
  host accounting. The diagnostic host128MiB registration is configured only.
  No provider or program fallback; unsupported geometry is an error.
- ROOT runtime must prove RNE/masked/noFTZ/noDAZ controls. Harness reads MXCSR
  before/after; the primitive does not mutate CPU FP policy or canonicalize zeros.

## Genuine ROOT replay protocol

1. Review NEW_DELTA.patch and all source/provenance pins. ROOT composes only
   the3 proposed CPU paths, with fresh modification times. No default callsite.
2. ROOT runs source-free tests/strict checks and builds this standalone helper
   offline under its sole Cargo schedule. Root effective lock must reuse exact
   current tuples; no acquisition. Cargo paths point to MAIN after composition.
3. ROOT `prepare.py --new-fixtures ABS/FRESH` authenticates unchanged AV/alpha
   parent captures/fixtures/source pins and accepted public DEFAULT outputs,
   then bounded byte-copies TWO original bulk64 cases. It never computes an
   expected word. Parent authenticators can hash whole model/DSO files: ROOT only.
4. ROOT records a real successful build log/effective lock and attestation:

   ```json
   {"status":"ROOT_COMPILED_BULK_F32_PRIMITIVE_ONLY","build_exit_code":0,
    "freeze_sha256":"ACTUAL_FREEZE","binary":"ABS_BINARY",
    "binary_digest":{"bytes":0,"sha256":"ACTUAL_BINARY_SHA"},
    "effective_lock_sha256":"ACTUAL_LOCK_SHA",
    "main_digest":{"bytes":0,"sha256":"ACTUAL_HELPER_SHA"},
    "production_files":[{"path":"ABS_MERGED_SOURCE","digest":{"bytes":0,"sha256":"ACTUAL_SHA"}}]}
   ```

   Fill real lengths/hashes; production_files contains all3 proposed `.rs`
   paths, sorted exactly as run_replay.py enumerates. A success label alone is
   not a build receipt. ROOT keeps actual command/log/lock alongside it.
5. Under ROOT's already approved user/network isolation, execute:

   ```sh
   python3 run_replay.py --binary ABS_BINARY --binary-sha256 ACTUAL_SHA \
     --build-attestation ABS_BUILD_JSON --fixtures ABS_FIXTURES --new-run ABS_FRESH_RUN
   python3 validate.py --run ABS_FRESH_RUN --fixtures ABS_FIXTURES
   ```

   Only ROOT runs this. Source/fixture/binary bindings precede and follow the
   actual child; owned WNOWAIT/group/EOF cleanup completes before final receipt
   I/O. The copied accepted53ee helper is unchanged; only Log/cleanup_group are
   used. No Podman command, GPU grant or provider is invoked by this runner.
   Actual4CPU affinity,1GiB AS,300s wall,4MiB log and bounded fresh output.

Actual inputs: .8B layer3 AV V `[256,256,2,1]` at genuine word strides
`[1,8448,2162688,4325376]`, P `[256,64,8,1]`; 2B original raw F32 alpha W
`[2048,16,1,1]`, X `[2048,64,1,1]`. AV logical capture is restored into its
original17,268,736B physical span; uncaptured gaps are zero-filled and remain
outside the arithmetic. No fake TensorSource, transposed alias or source weight
repack is created. Whole source authenticity remains ROOT's parent-pin check.

The helper uses the same original ResourceArbiter: metadata4MiB and staging32MiB
before input payload allocation, primitive actual output/scratch admission,
real pre-cancel and capacity refusal, retained output after input release,
final0. It calls the proposed public primitive, then opens expected payloads.
Independent validator compares ALL132096 output words byte-exact to authentic
captured DEFAULT outputs AND parent public default outputs. No tolerance,
expected fitting, source arithmetic replacement or signed-zero exception.
AR1 is deliberately refused here; existing four-Vec8 AR1 is a DIFFERENT profile.

## Gates that remain CLOSED

- All results initially UNQUALIFIED: source prep/AST/rustfmt are not replay.
- ROOT first needs both actual default-operator outputs, strict/adversarial
  tests and retained ticket/cleanup proof before any profile integration.
- Full default-bulk parity requires explicit immutable plan/profile/physical
  chunk selection and state/restore binding in the SAME engine. Never replace
  the existing four-Vec8 reducer globally or relabel its v1 checkpoints.
- After integration: unchanged first64 model captures, full short corpus,
  existing CPU/B1/native regression, full logits/all42 at bounded long-context
  cuts BOTH models; inspect further quantized/BLAS route gaps rather than
  assuming these two primitive fixes close the whole native schedule.
- Historical replay remains separate. HISTORICAL-RECOVERY.json records14
  permitted bounded artifact/serializer hashes,0matches to240549; unknown old
  serialization prevents a claim that fresh and historical token IDs differ.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/TODO.md

Original UTF-8 bytes: 1474; SHA256 `03c99e42754eb17860ec6cf7203736c7d654ff26b3c0d66233fae1961b91a63c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed bulk-F32 primitive preparation

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: baseline_review. Owned scope: this ignored proposal only.

- [x] Re-read actual730/fullV/all42 fresh8K controls and retained default gaps.
- [x] Recover nested historical240549 artifact digest/tokenizer/chunk settings;
  retain serialization caveat;14bounded fresh artifact/serializer checks0matches.
- [x] Prepare explicit source-derived singleVec8 +even/odd horizontal/tail math.
- [x] Preserve default fourVec8 CPU/model/GPU paths and current profile guards.
- [x] Prepare checked original admission/private result/cancel/refusal fixtures.
- [x] Prepare authentic transport +ROOT public-helper replay/independent validator.
- [x] Rust formatting/Python AST/source-only freeze; no tests/compiler/operator.
- [ ] Independent ROOT/peer full source review.
- [ ] ROOT source-free tests/strict Cargo/full isolated gate.
- [ ] ROOT actual two-default replay132096words/exactsignedbits/ledger/cleanup.
- [ ] Separate future explicit shared-engine schedule integration and complete
  BOTH-model short/long/fullV/all42/state/cancel/restore/default-route gates.
- [ ] Recover exact historical task/IDs/serialization/generator if available.

Unqualified: model/session/checkpoint/default/GPU eligibility, whole physical
host admission, historical token-sequence recovery, full default8K closure and
performance. No MAIN/shared-plan edits, commits, downloads or runtime by author.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/followup-attribution/README.md

Original UTF-8 bytes: 683; SHA256 `244319f470fb0fee1e1d47cf4fba43d8532f454f1ec956e4a022d942eac51c30`.
Resolve embedded relative paths against the original path, not this volume.

````text
Attribution-only follow-up to immutable bulk-F32 source freeze689221fe / deltae1460ed2.

Apply original NEW_DELTA.patch, then ATTRIBUTION_ONLY.patch; full replacement module is under proposed/. Exact Mozilla Foundation MIT notice is copied verbatim from pinned llamafile/sgemm.cpp, before the retained umbrella ggml MIT notice. Removing this added comment reproduces the original module byte-for-byte. No numerical code, tests, helper, default policy or runtime changes.

Source-only preparation; no builds, tests, operators, providers, models or GPU execution. ROOT owns review/composition/qualification. Original freeze, all16 owned bytes and all failed receipts remain unchanged.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/README.md

Original UTF-8 bytes: 7558; SHA256 `afa11f70d6b06d6838b23f2a05bca2e221f3a1d8b38fa080a7a3d5851a2dce90`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed Haswell bulk-F32 primitive — source preparation only

Parent: numerics-and-context. Owner: baseline_review. ROOT alone builds/runs.
No MAIN/shared-plan edits, compiler/test/operator/provider/model/GPU execution.

The proposed CPU math child implements ONE explicit numerical primitive,
`GgmlD812350HaswellTinyBlas`, without selecting it for any production projection,
attention, model, session, ForwardPlan or checkpoint. Existing four-Vec8 GGUF
paths and all current core/model numerical profiles remain byte unchanged.
No foreign library/model/token loop or second engine is introduced.

## Exact source arithmetic

Pinned original llama.cpp0.6.0/d812350 `ggml/src/ggml-cpu/llamafile/sgemm.cpp`:

-252–270: Vec8 low+high, then the actual `movehl`/`movehdup` tree:
  `(low0+low2)+(low1+low3)`. This is NOT vec-dot's adjacent-pair tree.
-407–409,627–670: one Vec8 +0 accumulator/output, ordered FMA every8 elements;
  both operands in a partial final Vec8 are zero-filled and ALL8 FMAs execute.
-3908–3934,559–617: actual Haswell F32 path refuses n<2; m divisible4 is a
  supported closed tile population. Scalar output iteration preserves each
  original output's FMA order; it does not claim native tile/thread scheduling.
- Full original ggml MIT notice is retained in the new math module.

Prepared source-free tests isolate one-chain versus four-chain cancellation,
actual even/odd horizontal order, subnormal/signed-zero tail padding, nonzero
views/NaN gap canaries/broadcast, source/stride/span/overflow/AR1 refusal,
nonfinite output refusal, actual capacity, pre-cancel and deterministic
mid-output cancellation. No test has been executed by the author.

## API and resource boundary

`math::haswell_bulk_f32::{BulkF32Profile,F32View,BulkF32Request,
BulkF32Admission,BulkF32Completed,execute}`.

- Borrowed F32 views retain original4D `[inner,row,head,sequence]` geometry,
  checked strides/offset/full backing extent, same nonzero SourceIdentity and
  bounded names. RHS must be genuinely contiguous; left may retain V strides.
- SourceIdentity is a local allocation identity, NOT a model digest/issuer.
  This low-level view cannot authenticate learned bytes or grant model support.
  Caller must keep original input owners/admission for synchronous execution.
  ROOT fixture transport separately verifies original capture/source provenance.
- K<=8192,m<=4096,2<=n<=64; <=1M logical words/operand/output, <=32MiB physical
  backing/operand, <=128Mi ordered FMAs/call. Checked products before access.
  Nonfinite used words refuse; unobserved gap words are never read by math.
- Existing supplied host ResourceArbiter admits output payload +256B numerical
  scratch BEFORE output allocation. Private result retains that original ticket;
  no public Vec escape/Clone. Payload drops before lease. Synchronous CPU work
  has no submitted GPU event/unknown completion to reinterpret.
- These are logical payload/scratch charges, not allocator/kernel/entire physical
  host accounting. The diagnostic host128MiB registration is configured only.
  No provider or program fallback; unsupported geometry is an error.
- ROOT runtime must prove RNE/masked/noFTZ/noDAZ controls. Harness reads MXCSR
  before/after; the primitive does not mutate CPU FP policy or canonicalize zeros.

## Genuine ROOT replay protocol

1. Review NEW_DELTA.patch and all source/provenance pins. ROOT composes only
   the3 proposed CPU paths, with fresh modification times. No default callsite.
2. ROOT runs source-free tests/strict checks and builds this standalone helper
   offline under its sole Cargo schedule. Root effective lock must reuse exact
   current tuples; no acquisition. Cargo paths point to MAIN after composition.
3. ROOT `prepare.py --new-fixtures ABS/FRESH` authenticates unchanged AV/alpha
   parent captures/fixtures/source pins and accepted public DEFAULT outputs,
   then bounded byte-copies TWO original bulk64 cases. It never computes an
   expected word. Parent authenticators can hash whole model/DSO files: ROOT only.
4. ROOT records a real successful build log/effective lock and attestation:

   ```json
   {"status":"ROOT_COMPILED_BULK_F32_PRIMITIVE_ONLY","build_exit_code":0,
    "freeze_sha256":"ACTUAL_FREEZE","binary":"ABS_BINARY",
    "binary_digest":{"bytes":0,"sha256":"ACTUAL_BINARY_SHA"},
    "effective_lock_sha256":"ACTUAL_LOCK_SHA",
    "main_digest":{"bytes":0,"sha256":"ACTUAL_HELPER_SHA"},
    "production_files":[{"path":"ABS_MERGED_SOURCE","digest":{"bytes":0,"sha256":"ACTUAL_SHA"}}]}
   ```

   Fill real lengths/hashes; production_files contains all3 proposed `.rs`
   paths, sorted exactly as run_replay.py enumerates. A success label alone is
   not a build receipt. ROOT keeps actual command/log/lock alongside it.
5. Under ROOT's already approved user/network isolation, execute:

   ```sh
   python3 run_replay.py --binary ABS_BINARY --binary-sha256 ACTUAL_SHA \
     --build-attestation ABS_BUILD_JSON --fixtures ABS_FIXTURES --new-run ABS_FRESH_RUN
   python3 validate.py --run ABS_FRESH_RUN --fixtures ABS_FIXTURES
   ```

   Only ROOT runs this. Source/fixture/binary bindings precede and follow the
   actual child; owned WNOWAIT/group/EOF cleanup completes before final receipt
   I/O. The copied accepted53ee helper is unchanged; only Log/cleanup_group are
   used. No Podman command, GPU grant or provider is invoked by this runner.
   Actual4CPU affinity,1GiB AS,300s wall,4MiB log and bounded fresh output.

Actual inputs: .8B layer3 AV V `[256,256,2,1]` at genuine word strides
`[1,8448,2162688,4325376]`, P `[256,64,8,1]`; 2B original raw F32 alpha W
`[2048,16,1,1]`, X `[2048,64,1,1]`. AV logical capture is restored into its
original17,268,736B physical span; uncaptured gaps are zero-filled and remain
outside the arithmetic. No fake TensorSource, transposed alias or source weight
repack is created. Whole source authenticity remains ROOT's parent-pin check.

The helper uses the same original ResourceArbiter: metadata4MiB and staging32MiB
before input payload allocation, primitive actual output/scratch admission,
real pre-cancel and capacity refusal, retained output after input release,
final0. It calls the proposed public primitive, then opens expected payloads.
Independent validator compares ALL132096 output words byte-exact to authentic
captured DEFAULT outputs AND parent public default outputs. No tolerance,
expected fitting, source arithmetic replacement or signed-zero exception.
AR1 is deliberately refused here; existing four-Vec8 AR1 is a DIFFERENT profile.

## Gates that remain CLOSED

- All results initially UNQUALIFIED: source prep/AST/rustfmt are not replay.
- ROOT first needs both actual default-operator outputs, strict/adversarial
  tests and retained ticket/cleanup proof before any profile integration.
- Full default-bulk parity requires explicit immutable plan/profile/physical
  chunk selection and state/restore binding in the SAME engine. Never replace
  the existing four-Vec8 reducer globally or relabel its v1 checkpoints.
- After integration: unchanged first64 model captures, full short corpus,
  existing CPU/B1/native regression, full logits/all42 at bounded long-context
  cuts BOTH models; inspect further quantized/BLAS route gaps rather than
  assuming these two primitive fixes close the whole native schedule.
- Historical replay remains separate. HISTORICAL-RECOVERY.json records14
  permitted bounded artifact/serializer hashes,0matches to240549; unknown old
  serialization prevents a claim that fresh and historical token IDs differ.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/TODO.md

Original UTF-8 bytes: 1474; SHA256 `03c99e42754eb17860ec6cf7203736c7d654ff26b3c0d66233fae1961b91a63c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed bulk-F32 primitive preparation

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: baseline_review. Owned scope: this ignored proposal only.

- [x] Re-read actual730/fullV/all42 fresh8K controls and retained default gaps.
- [x] Recover nested historical240549 artifact digest/tokenizer/chunk settings;
  retain serialization caveat;14bounded fresh artifact/serializer checks0matches.
- [x] Prepare explicit source-derived singleVec8 +even/odd horizontal/tail math.
- [x] Preserve default fourVec8 CPU/model/GPU paths and current profile guards.
- [x] Prepare checked original admission/private result/cancel/refusal fixtures.
- [x] Prepare authentic transport +ROOT public-helper replay/independent validator.
- [x] Rust formatting/Python AST/source-only freeze; no tests/compiler/operator.
- [ ] Independent ROOT/peer full source review.
- [ ] ROOT source-free tests/strict Cargo/full isolated gate.
- [ ] ROOT actual two-default replay132096words/exactsignedbits/ledger/cleanup.
- [ ] Separate future explicit shared-engine schedule integration and complete
  BOTH-model short/long/fullV/all42/state/cancel/restore/default-route gates.
- [ ] Recover exact historical task/IDs/serialization/generator if available.

Unqualified: model/session/checkpoint/default/GPU eligibility, whole physical
host admission, historical token-sequence recovery, full default8K closure and
performance. No MAIN/shared-plan edits, commits, downloads or runtime by author.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-peer-review-batch-20261020/REVIEW.md

Original UTF-8 bytes: 5495; SHA256 `4152b898498524a50e04bb26917bf5cad352f9eedf6b0fbf0d4797f43d0bd620`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed bulk F32 typed slice — independent source review

Author: batch_design_review. Parent: numerics-and-context.

Verdict: no blocker found in the declared source-only pure-operator scope.
Compilation, actual source numerical replay and phase acceptance remain ROOT's.

Reviewed `numerics-and-context-bulk-f32-typed-slice-20261020`:

- Original freeze `0d11d476e2955a40eeb1dbe45c478b4502dede72a92ebb2e4f092e7481e7a6e1`.
  All33 listed live files and33 frozen copies match; all10 dependency/source pins match.
- Nine-path core/model delta `79c065c2669ba31003ab65594af82b8f4dea45d2e652a978db323c0960afab0f`;
  separate2-path GPU refusal `8d24045e9855533e4793721122c74b0e2d380a0bd83bec2265f5a0b000a4ea2f`.
- Execution followup `fce99df1aba6fff0416fdbbf60535ea24d18127740a4fbf50a11053af66ad003`;
  all5 live/frozen execution files and2 docs match. Original source remains immutable.
- Read all new production children, every parent delta, source-free core fixtures,
  Rust replay main, Python prepare/contracts/validator/runner/owned lifetime and
  signal fixtures. Twelve Python sources AST-parsed only; nothing imported/run.

## Contract and arithmetic

- One same-source `TeacherSpan` uses Arc identity; equal IDs/labels cannot merge
  independently created spans. N2..64 is temporal teacher count, not requestB.
- Full K, exact allocation backing/offset/strides, checked overflow/end and
  contiguous RHS are enforced. Head broadcast requires exact divisibility.
  Complete learned F32 matrix retains original source/name/InputOutput descriptor;
  no fake rows, dequantization, transpose, learned cache or K splitting.
- PlanBuilder records exactly two typed inputs +one typed matrix node. Owner,
  producer, precision, span, side and layout are checked. Generic legacy consumer,
  full-plan/state/frontier/logit paths do not gain eligibility. GPU preflight and
  both execution emitters refuse the new operations explicitly.
- SAME HostMath calls the unchanged closed primitive. No copied token/layer loop
  or MatVec fallback/provider callback. Existing four-Vec8 default paths unchanged.
- Source context confirms tinyBLAS one Vec8 FMA chain, positive-zero initialized
  accumulator, zero-padded remainder FMAs and even/odd horizontal tree. Typed
  wrapper introduces no new reducer/cast. RNE/noFTZ/noDAZ controls are observed
  before/after math and are never changed. Logical finite checks precede output
  allocation; arbitrary storage gaps remain outside arithmetic.

## Ownership and resource boundary

- Host metadata ticket precedes span/descriptor allocation; exact aggregate
  external private staging ticket precedes either payload copy. Actual Vec
  capacity equality is required. Learned raw matrix remains borrowed from its
  retained immutable model owner, with byte geometry/alignment revalidation.
- Runtime HostValues retain the exact bound owner; foreign/partial/retagged views
  and overlapping actual input ranges refuse. Original primitive admits result
  plus256B scratch before output allocation and retains them through completion.
- Output exposes borrowed slices only; clones share the same retained result and
  source owner. Input staging can retire independently after math. Payload fields
  precede their tickets in drop order. No SessionAuthority/StateRoot publication.
- Replay's six live reservations and four retained-output reservations are
  consistent with its original external source/meta/staging and private tickets.
  It verifies exact classes, real one-byte pressure, pre-cancel, provider zero
  callbacks, cloned borrowed output/source lifetime and final0.
- Its32MiB diagnostic transport and full-source+16MiB metadata charge are honest
  diagnostic envelopes, not claimed proofs of cached metadata capacity or whole
  process physical RAM. No actual mid-operation cancellation claim is added.

## Genuine replay and lifecycle

- Unchanged accepted default public AV/alpha captures supply all132,096 expected
  F32 words. Preparation verifies earlier transparent public/default outputs and
  source/readset closure, then copies bytes only. Both original8192 teacher files
  must match completely; actual first64 are used. No expected generation/fitting.
- AV logical values are restored into original strided transport; uncaptured gaps
  are explicitly reconstructed, not described as actual complete cache storage.
  Alpha checks all original learned raw bytes, then same model source borrows them.
- Expected raw arithmetic words open only after HostMath returns. Independent
  validator requires every actual byte/hash, complete geometry, same source and
  all ledger/lifetime facts. Failed math reports remain failures without tolerance.
- New runner binds ROOT build/source/dependency attestations and original frozen
  bytes. Exact accepted00d1 TERM+INT handlers only record through child handoff and
  cleanup; checkpoints invoke original53ee WNOWAIT observation/TERM/KILL-before-
  reap/group absence/EOF retirement. Primary and cleanup failures prevent success.
  Signals arriving at the very end can still yield nonzero exit after a receipt;
  ROOT must require actual runner success as well as validator receipts.

Remaining gates: ROOT composition/strict/unit/full checks, primitive and typed
actual replay, actual cancellation/failure controls as applicable. No default
bulk model scheduler, full8K parity, historical teacher recovery, StateRoot,
GPU/provider, physical whole-RAM or performance eligibility follows from review.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-peer-review-batch-20261020/TODO.md

Original UTF-8 bytes: 719; SHA256 `6cdb21768915b0b5a084f3ff54931e41c539d6fa678d33ca9ab2f4a95782d8d3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed bulk F32 typed peer review

Parent: numerics-and-context. Owner: batch_design_review.

- [x] Read all9 core/model paths +2 cold GPU guards, unchanged primitive and source context.
- [x] Read ROOT helper, authentic transport/provenance, independent validator, lifetime and synthetic fixtures.
- [x] Review additive two-signal execution fce99; check immutable source/frozen bindings.
- [x] Record source verdict/physical limits in REVIEW.md.
- [ ] ROOT composes, builds and executes numerical/resource/full-gate checks.

Review only: source/retained metadata and12 Python ASTs, no imports/tests,
runtime/model/provider/GPU/build activity or binary/DSO/model payload read.
No MAIN/shared-plan changes or commits.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/README.md

Original UTF-8 bytes: 3409; SHA256 `6e1b6ba729fa7c6efae9711f76d654026419a14e9a69b740aa21aa6352840345`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed default-bulk F32 typed slice — source proposal only

Parent: numerics-and-context. Owner: baseline_review.

- Dependency: primitive `689221fe` + exact Mozilla MIT attribution `a7e8cc3b`; integration audit `adac2aa8`. All unchanged. Primitive and this slice remain uncompiled/unqualified.
- Generic core: one sealed, consecutive teacher span (`N=2..64`, one sequence), exact source/full-K/ne/nb/offset/backing metadata and explicit `HaswellTinyBlasF32V1` operation contract. Equal IDs/names do not merge independent span issuers. Physical `ValueShape::Vector` capacity is distinct from temporal matrix axes carried by the operation.
- SAME `MathematicalBackend`/`PlanBuilder`/`HostMath`: exactly input/input/matrix, no model or token loop. Full plans/state/logits/publication are refused; existing AR1 four-Vec8 math and source/session/checkpoint guards stay unchanged.
- AV: explicit external observed operands. Reconstructed transport gaps are declared, not claimed to be original physical cache bytes/current state. Original actual strides and full K remain.
- Alpha: exact original raw F32 matrix role/name/source/axes, reborrowed from the same retained `LoadedQwenModel`; no learned copy, dequantization or adjusted cache. RHS remains explicit external observed input.
- Existing host `ResourceArbiter`: bounded 16KiB context metadata, aggregate actual private external payload capacities before allocation; unchanged primitive admits output+256B scratch before result allocation. Borrowed/clone output retains tickets/source. No bare Vec escape. Generic plan/control-plane allocation and pre-existing source allocation are outside this context reservation; no whole-host physical RAM claim.
- Default/unaware backend/provider and GPU cold routes refuse. RNE/noFTZ/noDAZ is observed before/after math, never mutated. Source descriptor, issuer, side, full view, full K, exact current backing lengths, checked ranges and no overlapping runtime input buffers are checked before the original primitive.

Prepared evidence only:

- Five source-free core fixtures: actual AV N/strides, independent issuers/full K, offset/overflow/overlap bounds, same three-node plan/foreign values/legacy refusal, raw source dtype/axes/bytes/issuer.
- ROOT helper: genuine accepted .8B AV +2B alpha complete default-native outputs (132,096 F32 words), independent expected bytes opened after math, raw alpha learned byte equality, actual original teacher IDs, same three-node plan/no state, pre-cancel constructor+execution, provider zero callbacks, real one-byte output/scratch pressure, cloned borrowed output and source lifetime, actual per-class ledgers/final0.
- Nine synthetic Python parser/refusal fixtures; they establish no trained arithmetic.
- Author ran only rustfmt and AST parsing. No tests, compilation, provider/model/operator/GPU execution or weight/DSO payload hashes.

Not closed: actual mid-operation cancellation at this HostMath boundary, full default bulk forward/scheduling, default 8K parity, historical input recovery, 2B Q8_0→F32 BLAS `linear_attn_out`, GPU execution and performance. Existing primitive's cancellation/unit gates remain a separate prerequisite; this replay honestly qualifies pre-cancel only. After primitive/slice exactness, next qualification cut is the same-engine temporal layer scheduler and producer/state/profile sealing described in the prior integration audit.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/ROOT.md

Original UTF-8 bytes: 3227; SHA256 `fedacad825f561a648e618104502e5cefe8cece180c22407b7709400c48b8c8f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only execution sequence

1. Review/compose primitive `689221fe` + attribution `a7e8cc3b` first. Run its meaningful unit/default-operand replay gates. Do not promote this uncompiled candidate from source review.
2. Apply `NEW_DELTA.patch` (core/model only) against the exact `BASELINE.json` bytes. `GPU_REFUSAL.patch` is separate narrow exhaustive cold-refusal wiring; preserve newer unrelated GPU code. Copy source with fresh mtimes. Frozen full proposed bytes permit exact review; if parents require composition, retain a new ROOT derivative/source recipe rather than retagging this freeze.
3. ROOT appropriate isolated strict CPU/core/default/CUDA/Vulkan checks plus required full gate. Five new core tests are in `program::bulk_f32::tests`. Full model execution/session/checkpoint stays closed.
4. ROOT `python3 OWN/prepare.py --new-fixtures OWN/root-fixtures-1` only when CPU/I/O allowed. This reauthenticates unchanged old public captures, source/DSO/readsets and whole GGUFs; copies original inputs/expected outputs; proves both original8192 teacher files byte-equal and binds first64. No operator or expected generation. Failed preparation directory is retained, never reused.
5. Build this standalone `Cargo.toml` offline, ROOT-owned finite jobs. Its explicit paths resolve to MAIN crates. Use current effective lock normalized only by ROOT; preserve build argv/log/status. Source helper loads approved GGUF raw source +existing small caches, not forward/session/model output.
6. Write a fresh actual build-attestation JSON with `status=ROOT_COMPILED_SAME_ENGINE_BULK_F32_ONLY`, `build_exit_code=0`, this freeze SHA, absolute binary path/digest, effective lock SHA, `main_digest`, `production_files` and `primitive_dependency_files` arrays exactly as `run_replay.py` computes (sorted full proposed source paths; primitive dependency order is math.rs/tests.rs/corrected primitive.rs). Record actual compilation; no fabricated attestation.
7. `python3 OWN/run_replay.py --binary ABS --binary-sha256 ACTUAL --build-attestation ABS --fixtures OWN/root-fixtures-1 --new-run OWN/root-replay-1`. Bounds: CPU affinity exactly4,300s wall/1200s CPU,4GiB address space,64FDs/8MiB files, original WNOWAIT owned PGID cleanup and bounded4MiB EOF drain. Copied accepted SIGTERM record/checkpoint guard spans original cleanup; no asynchronous raise across Popen assignment, no signal after reap. Receipt I/O after retirement cannot strand child.
8. `python3 OWN/validate.py --run OWN/root-replay-1 --fixtures OWN/root-fixtures-1`. Must compare all132,096 actual bytes, all class/retention/final0 facts and original default source closure. Results are pure captured operator scope only. Failed math/build/refusal/cleanup artifacts remain UNQUALIFIED, no fallback/tolerance/retry to convert failure.

Configured diagnostic source charge: full raw source byte count +conservative16MiB cached/metadata before `LoadedQwenModel::open`; no proof of exact source metadata/allocator capacity or outer physical host admission. Inputs use separate32MiB diagnostic transport charge; operator context independently owns exact private payload charges. Expected/report/control-plane memory is diagnostic, not a new production admission claim.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/TODO.md

Original UTF-8 bytes: 1023; SHA256 `258533a86524bdbe7b2a9c909e32cac5cce35000aced221b1ba714cbfae52074`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed bulk F32 typed slice

Parent numerics-and-context. Owner baseline_review.

- [x] Recover immutable primitive/integration constraints; preserve both freezes.
- [x] Generic sealed temporal span/layout/full-K matrix and SAME pure3-node plan.
- [x] SAME HostMath source borrowing +aggregate admitted payload/result backing; original AR1/default/state guards unchanged.
- [x] Five core fixtures +ROOT original default AV/alpha byte replay/lifetime/refusal/ledger preparation; nine synthetic validator fixtures.
- [x] Rustfmt +Python AST only. No imports/tests/builds/math/model/provider/GPU/payload hashes by author.
- [x] Freeze proposed bytes/deltas for ROOT/independent source review.
- [ ] ROOT compile/strict/unit/allcfg gates; primitive+slice genuine132096-word replay and independent validation.
- [ ] Full default bulk schedule/producer/state sealing and2B Q8→F32 BLAS remain separate unqualified cuts.

No MAIN/shared plans/other agent freezes changed. No production/default/performance acceptance inferred.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/followup-signal-v2/README.md

Original UTF-8 bytes: 1380; SHA256 `9745511acf738c549101dae00a33cb65725ab6a006c2e4ec0e94f187d00c1113`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive execution-only two-signal guard

Preserve source/math freeze0d11d476/all33 files. V1 reused the accepted startup SIGTERM-only guard; SIGINT remained asynchronous across Popen assignment. V2 copies the already accepted native A/B SIGTERM+SIGINT record-only guard BYTE-EXACT. Original owner/log/retirement, math/native helper, prepared fixtures, production deltas and validator remain unchanged.

ROOT use `run_replay_v2.py` instead of V1 runner and `validate_v2.py` instead of V1 validator, with identical documented CLI arguments. Original ROOT build-attestation continues binding source/math freeze0d11; V2 receipt independently binds this execution freeze +two exact cancellation signals. Require BOTH original `validation.json` and new `validation-execution-v2.json`. Historical field `sigterm_received` now records either accepted interruption; successful receipts must be false. No actual signal-injection qualification is inferred from a successful numeric replay.

Three new mocked source-free fixtures exercise handler installation/restoration, record-only Popen assignment/finally and drain restoration. Author AST-only, no imports/tests/real signals/children/runtime. ROOT can run these before the actual replay; failure or incomplete execution evidence remains UNQUALIFIED. No lifecycle rewrite, background thread, math/provider/model route or timing claim.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/README.md

Original UTF-8 bytes: 3409; SHA256 `6e1b6ba729fa7c6efae9711f76d654026419a14e9a69b740aa21aa6352840345`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed default-bulk F32 typed slice — source proposal only

Parent: numerics-and-context. Owner: baseline_review.

- Dependency: primitive `689221fe` + exact Mozilla MIT attribution `a7e8cc3b`; integration audit `adac2aa8`. All unchanged. Primitive and this slice remain uncompiled/unqualified.
- Generic core: one sealed, consecutive teacher span (`N=2..64`, one sequence), exact source/full-K/ne/nb/offset/backing metadata and explicit `HaswellTinyBlasF32V1` operation contract. Equal IDs/names do not merge independent span issuers. Physical `ValueShape::Vector` capacity is distinct from temporal matrix axes carried by the operation.
- SAME `MathematicalBackend`/`PlanBuilder`/`HostMath`: exactly input/input/matrix, no model or token loop. Full plans/state/logits/publication are refused; existing AR1 four-Vec8 math and source/session/checkpoint guards stay unchanged.
- AV: explicit external observed operands. Reconstructed transport gaps are declared, not claimed to be original physical cache bytes/current state. Original actual strides and full K remain.
- Alpha: exact original raw F32 matrix role/name/source/axes, reborrowed from the same retained `LoadedQwenModel`; no learned copy, dequantization or adjusted cache. RHS remains explicit external observed input.
- Existing host `ResourceArbiter`: bounded 16KiB context metadata, aggregate actual private external payload capacities before allocation; unchanged primitive admits output+256B scratch before result allocation. Borrowed/clone output retains tickets/source. No bare Vec escape. Generic plan/control-plane allocation and pre-existing source allocation are outside this context reservation; no whole-host physical RAM claim.
- Default/unaware backend/provider and GPU cold routes refuse. RNE/noFTZ/noDAZ is observed before/after math, never mutated. Source descriptor, issuer, side, full view, full K, exact current backing lengths, checked ranges and no overlapping runtime input buffers are checked before the original primitive.

Prepared evidence only:

- Five source-free core fixtures: actual AV N/strides, independent issuers/full K, offset/overflow/overlap bounds, same three-node plan/foreign values/legacy refusal, raw source dtype/axes/bytes/issuer.
- ROOT helper: genuine accepted .8B AV +2B alpha complete default-native outputs (132,096 F32 words), independent expected bytes opened after math, raw alpha learned byte equality, actual original teacher IDs, same three-node plan/no state, pre-cancel constructor+execution, provider zero callbacks, real one-byte output/scratch pressure, cloned borrowed output and source lifetime, actual per-class ledgers/final0.
- Nine synthetic Python parser/refusal fixtures; they establish no trained arithmetic.
- Author ran only rustfmt and AST parsing. No tests, compilation, provider/model/operator/GPU execution or weight/DSO payload hashes.

Not closed: actual mid-operation cancellation at this HostMath boundary, full default bulk forward/scheduling, default 8K parity, historical input recovery, 2B Q8_0→F32 BLAS `linear_attn_out`, GPU execution and performance. Existing primitive's cancellation/unit gates remain a separate prerequisite; this replay honestly qualifies pre-cancel only. After primitive/slice exactness, next qualification cut is the same-engine temporal layer scheduler and producer/state/profile sealing described in the prior integration audit.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/ROOT.md

Original UTF-8 bytes: 3227; SHA256 `fedacad825f561a648e618104502e5cefe8cece180c22407b7709400c48b8c8f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT-only execution sequence

1. Review/compose primitive `689221fe` + attribution `a7e8cc3b` first. Run its meaningful unit/default-operand replay gates. Do not promote this uncompiled candidate from source review.
2. Apply `NEW_DELTA.patch` (core/model only) against the exact `BASELINE.json` bytes. `GPU_REFUSAL.patch` is separate narrow exhaustive cold-refusal wiring; preserve newer unrelated GPU code. Copy source with fresh mtimes. Frozen full proposed bytes permit exact review; if parents require composition, retain a new ROOT derivative/source recipe rather than retagging this freeze.
3. ROOT appropriate isolated strict CPU/core/default/CUDA/Vulkan checks plus required full gate. Five new core tests are in `program::bulk_f32::tests`. Full model execution/session/checkpoint stays closed.
4. ROOT `python3 OWN/prepare.py --new-fixtures OWN/root-fixtures-1` only when CPU/I/O allowed. This reauthenticates unchanged old public captures, source/DSO/readsets and whole GGUFs; copies original inputs/expected outputs; proves both original8192 teacher files byte-equal and binds first64. No operator or expected generation. Failed preparation directory is retained, never reused.
5. Build this standalone `Cargo.toml` offline, ROOT-owned finite jobs. Its explicit paths resolve to MAIN crates. Use current effective lock normalized only by ROOT; preserve build argv/log/status. Source helper loads approved GGUF raw source +existing small caches, not forward/session/model output.
6. Write a fresh actual build-attestation JSON with `status=ROOT_COMPILED_SAME_ENGINE_BULK_F32_ONLY`, `build_exit_code=0`, this freeze SHA, absolute binary path/digest, effective lock SHA, `main_digest`, `production_files` and `primitive_dependency_files` arrays exactly as `run_replay.py` computes (sorted full proposed source paths; primitive dependency order is math.rs/tests.rs/corrected primitive.rs). Record actual compilation; no fabricated attestation.
7. `python3 OWN/run_replay.py --binary ABS --binary-sha256 ACTUAL --build-attestation ABS --fixtures OWN/root-fixtures-1 --new-run OWN/root-replay-1`. Bounds: CPU affinity exactly4,300s wall/1200s CPU,4GiB address space,64FDs/8MiB files, original WNOWAIT owned PGID cleanup and bounded4MiB EOF drain. Copied accepted SIGTERM record/checkpoint guard spans original cleanup; no asynchronous raise across Popen assignment, no signal after reap. Receipt I/O after retirement cannot strand child.
8. `python3 OWN/validate.py --run OWN/root-replay-1 --fixtures OWN/root-fixtures-1`. Must compare all132,096 actual bytes, all class/retention/final0 facts and original default source closure. Results are pure captured operator scope only. Failed math/build/refusal/cleanup artifacts remain UNQUALIFIED, no fallback/tolerance/retry to convert failure.

Configured diagnostic source charge: full raw source byte count +conservative16MiB cached/metadata before `LoadedQwenModel::open`; no proof of exact source metadata/allocator capacity or outer physical host admission. Inputs use separate32MiB diagnostic transport charge; operator context independently owns exact private payload charges. Expected/report/control-plane memory is diagnostic, not a new production admission claim.

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/frozen/TODO.md

Original UTF-8 bytes: 1023; SHA256 `258533a86524bdbe7b2a9c909e32cac5cce35000aced221b1ba714cbfae52074`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Closed bulk F32 typed slice

Parent numerics-and-context. Owner baseline_review.

- [x] Recover immutable primitive/integration constraints; preserve both freezes.
- [x] Generic sealed temporal span/layout/full-K matrix and SAME pure3-node plan.
- [x] SAME HostMath source borrowing +aggregate admitted payload/result backing; original AR1/default/state guards unchanged.
- [x] Five core fixtures +ROOT original default AV/alpha byte replay/lifetime/refusal/ledger preparation; nine synthetic validator fixtures.
- [x] Rustfmt +Python AST only. No imports/tests/builds/math/model/provider/GPU/payload hashes by author.
- [x] Freeze proposed bytes/deltas for ROOT/independent source review.
- [ ] ROOT compile/strict/unit/allcfg gates; primitive+slice genuine132096-word replay and independent validation.
- [ ] Full default bulk schedule/producer/state sealing and2B Q8→F32 BLAS remain separate unqualified cuts.

No MAIN/shared plans/other agent freezes changed. No production/default/performance acceptance inferred.

````

## Source: .agents/runs/numerics-and-context-compiled-f32-routes-20261020/README.md

Original UTF-8 bytes: 7233; SHA256 `f93e4fb73b6f1f65700e08eda3a7d6f34bfd83a33026dfba7a4ed9def76892b3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Compiled F32 route observation — ROOT execution only

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owned scope: this ignored run only. No production, mathematical engine, model,
source weight, execution profile, checkpoint or serving changes.

Prepared, not executed by the author. Static ELF/source inspection and Python
AST parsing establish no operator or GDB qualification.

## Purpose and limits

Run each already accepted public ggml replay unchanged under a fresh owned GDB
inferior. Retain its original public-entry LD_PRELOAD observer. Observe only the
first entry into each named compiled function per original compute scope.
There are no private ABI calls, argument reads, FP expressions, memory writes,
return changes, numerical substitutions or attachments to existing processes.

The original public client performs default bulk64 + 64 AR1 operations before
the separately named public `use_ref` controls. Every one of the 130 full outputs,
whole operand invariance, existing exported-entry trace, and actual mapped-file
set/hash guards must pass before accepting a named-entry observation. Nothing
here proves every hit/instruction, anonymous mappings, full host-runtime closure,
long-context or historical-corpus causality, or model/profile eligibility.

## Pinned source and compiled entries

`source-pins.json` binds all original fixtures, accepted plain outputs, source,
original observer/binary, accepted public traces and mapped regular files.
`datasets.json` names both exact accepted datasets and original argv; the new
launcher changes only the output-directory argument. Inputs remain external and
read-only by use; full fixture hashes are checked before and after execution.

`static-audit.json` records the exact Haswell DSO, ELF PT_LOAD ranges, 13 SYMTAB
entries, their code ranges/hashes, and bounded retained nm/objdump output. These
static inspections were performed without loading or running the provider.

Pinned primary source: acquired llama.cpp 0.6.0 tree, `ggml/src/ggml-cpu/`:

- `ggml-cpu.c:1308`: default contiguous RHS may call public `llamafile_sgemm`;
  `use_ref` skips it. A returned boolean alone is insufficient internal evidence.
- `llamafile/sgemm.cpp:559–730`: F32 `tinyBLAS<8>` uses one Vec8 accumulator per
  output and KN=8 FMA steps. The source-selected bulk outer family is
  `gemm<4,3,4>` for AV (m=256) and `gemm<4,3,2>` for alpha (m=16), with actual
  full/tail block functions `gemm_bloc<4,3>` and `gemm_bloc<4,2>` at n=64.
- `vec.cpp:104–127`, `simd-mappings.h:587–631`: the Haswell F32 dot family uses
  four Vec8 partials across 32-element steps, pair reduction then horizontal
  reduction. This is a source description, not a new arithmetic implementation.

The proposed exact observation is 3 named bulk entries followed by 129
`ggml_vec_dot_f32` entries: 132 first-entry records and 260 original begin/end
stops, total 392. All 12 candidate F32 symbols are watched inside each actual
scope, so an unexpected candidate family fails validation. Later hits after the
first one are explicitly unobserved. Native probes start disabled and are
enabled only between the unchanged observer's begin/end callbacks.

Each named breakpoint resolves through the actual ELF symbol. Its sole location,
PC and mapped DSO must equal the exact pinned offset. A pure public symbol-address
lookup of `ggml_vec_dot_f32` supplies the load base; no target function is called.

## Ownership and bounds

- GDB: pinned GNU 17.2, `--nx --nh --batch`, autoload/debuginfod/shell startup off,
  target function calls disabled, all-stop and scheduler-locking off. Breakpoint
  changes occur in a top-level stopped loop, never `Breakpoint.stop` callbacks.
- Fresh Popen session only; GDB PID=PGID=SID, live parent/starttime/UID bound. The
  actual inferior PID/starttime/SID/UID is published atomically by the observer.
- Public process-local PR_SET/GET_CHILD_SUBREAPER; no host setting changes.
  The supervisor only scans its authenticated fresh SID (max32 members). Every
  nonzero signal uses a public pidfd after before/after starttime/PPID/SID/UID/
  PGID checks. No `killpg` nonzero signals, broad process search, foreign adoption
  or numeric-PID retargeting.
- TERM, grace, two KILL stages and two leader-reap opportunities are independent;
  all signal stages precede any supervisor reap. Known adopted inferior identities
  are reaped separately. All known groups must be absent and the pipe at EOF.
  Primary and every cleanup failure remain visible and make the run unqualified.
- This is owned-SID supervision of the pinned source-reviewed client, not a
  general containment facility. An observed SID/UID escape or unavailable exact
  process identity fails closed; no unsafe fallback signal is issued. If initial
  Session construction fails, the wrapper records incomplete cleanup, refuses
  unbound signals, and cannot claim successful retirement.
- Original public client deadline remains300s, observer wall600s, outer wall900s,
  <=2048 named stops, combined log<=8MiB, metadata<=1MiB/file. RLIMIT_AS is8GiB
  **per process**, not an aggregate host-RAM claim. Whole new run tree<=512MiB,
  including a128KiB final-report reserve. No model loading or GPU access.

## ROOT commands after independent source review

Use the same already approved user/network isolation and CPU policy as the
accepted public replay. These are source preparation instructions, not a receipt.

```bash
task_run=/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-compiled-f32-routes-20261020
python3 "$task_run/test_supervisor.py"
python3 "$task_run/run_gdb.py" --dataset av --new-output "$task_run/root-av-gdb-1"
python3 "$task_run/validate.py" --run "$task_run/root-av-gdb-1" --new-report "$task_run/root-av-validation-1.json"
python3 "$task_run/run_gdb.py" --dataset alpha --new-output "$task_run/root-alpha-gdb-1"
python3 "$task_run/validate.py" --run "$task_run/root-alpha-gdb-1" --new-report "$task_run/root-alpha-validation-1.json"
```

Five root-only real-process fixtures cover a separate inferior PGID in the same
owned SID, actual pidfd signaling/adoption/reaping, and independently injected
TERM/grace/first-KILL/first-reap failures. Importing the fixture or supervisor
does not launch or signal anything. No fixtures have been executed by the author.

Failure to resolve exact named locations, startup/debugger API differences,
deadline failures, output differences or closure differences is UNQUALIFIED.
Preserve the attempt; do not silently weaken expected outputs or route labels.

Primary API references:

- [GNU GDB Python breakpoints](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Breakpoints-In-Python.html)
- [GNU GDB Python events](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Events-In-Python.html)
- [Linux pidfd_open](https://man7.org/linux/man-pages/man2/pidfd_open.2.html)
- [Linux pidfd_send_signal](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html)
- [Linux PR_SET_CHILD_SUBREAPER](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html)

`UPSTREAM_LICENSE` preserves the original llama.cpp MIT notice. No upstream
arithmetic implementation is copied into these observation scripts.

````

## Source: .agents/runs/numerics-and-context-compiled-f32-routes-20261020/TODO.md

Original UTF-8 bytes: 1515; SHA256 `93a4917a716709bb236c6ebcd40859ee7f72fbaaa0e68eebe7f910d63e74b6a3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Compiled F32 route source preparation

- Parent: numerics-and-context; coordinator owns shared plan and execution.
- Owned: this ignored run only. Original AV/alpha helpers and receipts unchanged.

## Prepared

- [x] Inspect exact primary F32 SGEMM/vector source and bounded pinned ELF symbols.
- [x] Retain13 symbol/code ranges and bounded nm/objdump artifacts; no arithmetic port.
- [x] Bind both accepted datasets, all130 outputs and full actual mapped-file sets.
- [x] Prepare named entry observer without private ABI calls/arguments/FP evaluation.
- [x] Prepare owned-SID subreaper/pidfd supervisor, independent signal/reap stages.
- [x] Prepare5 real-process failure fixtures; no author execution.
- [x] Require exact output/operand/maps/public-trace gates before132 named entries.
- [x] Parse Python AST only; source preparation is not runtime qualification.

## ROOT pending

- [ ] Independent source review and any explicit additive repair/freeze.
- [ ] Execute real-process supervisor fixtures in approved isolation.
- [ ] Execute fresh original AV and alpha public clients under owned GDB.
- [ ] Validate all130 complete outputs for each, actual file maps and public traces.
- [ ] Accept only bounded first-entry evidence; preserve unexpected/failed attempts.
- [ ] Coordinator reconcile qualification boundaries in the existing owning plan.

No GDB, operator, model, container, build or test was executed by the author.
No full internal instruction, historical/lost corpus or model eligibility claim.

````

## Source: .agents/runs/numerics-and-context-compiled-f32-routes-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 7233; SHA256 `f93e4fb73b6f1f65700e08eda3a7d6f34bfd83a33026dfba7a4ed9def76892b3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Compiled F32 route observation — ROOT execution only

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owned scope: this ignored run only. No production, mathematical engine, model,
source weight, execution profile, checkpoint or serving changes.

Prepared, not executed by the author. Static ELF/source inspection and Python
AST parsing establish no operator or GDB qualification.

## Purpose and limits

Run each already accepted public ggml replay unchanged under a fresh owned GDB
inferior. Retain its original public-entry LD_PRELOAD observer. Observe only the
first entry into each named compiled function per original compute scope.
There are no private ABI calls, argument reads, FP expressions, memory writes,
return changes, numerical substitutions or attachments to existing processes.

The original public client performs default bulk64 + 64 AR1 operations before
the separately named public `use_ref` controls. Every one of the 130 full outputs,
whole operand invariance, existing exported-entry trace, and actual mapped-file
set/hash guards must pass before accepting a named-entry observation. Nothing
here proves every hit/instruction, anonymous mappings, full host-runtime closure,
long-context or historical-corpus causality, or model/profile eligibility.

## Pinned source and compiled entries

`source-pins.json` binds all original fixtures, accepted plain outputs, source,
original observer/binary, accepted public traces and mapped regular files.
`datasets.json` names both exact accepted datasets and original argv; the new
launcher changes only the output-directory argument. Inputs remain external and
read-only by use; full fixture hashes are checked before and after execution.

`static-audit.json` records the exact Haswell DSO, ELF PT_LOAD ranges, 13 SYMTAB
entries, their code ranges/hashes, and bounded retained nm/objdump output. These
static inspections were performed without loading or running the provider.

Pinned primary source: acquired llama.cpp 0.6.0 tree, `ggml/src/ggml-cpu/`:

- `ggml-cpu.c:1308`: default contiguous RHS may call public `llamafile_sgemm`;
  `use_ref` skips it. A returned boolean alone is insufficient internal evidence.
- `llamafile/sgemm.cpp:559–730`: F32 `tinyBLAS<8>` uses one Vec8 accumulator per
  output and KN=8 FMA steps. The source-selected bulk outer family is
  `gemm<4,3,4>` for AV (m=256) and `gemm<4,3,2>` for alpha (m=16), with actual
  full/tail block functions `gemm_bloc<4,3>` and `gemm_bloc<4,2>` at n=64.
- `vec.cpp:104–127`, `simd-mappings.h:587–631`: the Haswell F32 dot family uses
  four Vec8 partials across 32-element steps, pair reduction then horizontal
  reduction. This is a source description, not a new arithmetic implementation.

The proposed exact observation is 3 named bulk entries followed by 129
`ggml_vec_dot_f32` entries: 132 first-entry records and 260 original begin/end
stops, total 392. All 12 candidate F32 symbols are watched inside each actual
scope, so an unexpected candidate family fails validation. Later hits after the
first one are explicitly unobserved. Native probes start disabled and are
enabled only between the unchanged observer's begin/end callbacks.

Each named breakpoint resolves through the actual ELF symbol. Its sole location,
PC and mapped DSO must equal the exact pinned offset. A pure public symbol-address
lookup of `ggml_vec_dot_f32` supplies the load base; no target function is called.

## Ownership and bounds

- GDB: pinned GNU 17.2, `--nx --nh --batch`, autoload/debuginfod/shell startup off,
  target function calls disabled, all-stop and scheduler-locking off. Breakpoint
  changes occur in a top-level stopped loop, never `Breakpoint.stop` callbacks.
- Fresh Popen session only; GDB PID=PGID=SID, live parent/starttime/UID bound. The
  actual inferior PID/starttime/SID/UID is published atomically by the observer.
- Public process-local PR_SET/GET_CHILD_SUBREAPER; no host setting changes.
  The supervisor only scans its authenticated fresh SID (max32 members). Every
  nonzero signal uses a public pidfd after before/after starttime/PPID/SID/UID/
  PGID checks. No `killpg` nonzero signals, broad process search, foreign adoption
  or numeric-PID retargeting.
- TERM, grace, two KILL stages and two leader-reap opportunities are independent;
  all signal stages precede any supervisor reap. Known adopted inferior identities
  are reaped separately. All known groups must be absent and the pipe at EOF.
  Primary and every cleanup failure remain visible and make the run unqualified.
- This is owned-SID supervision of the pinned source-reviewed client, not a
  general containment facility. An observed SID/UID escape or unavailable exact
  process identity fails closed; no unsafe fallback signal is issued. If initial
  Session construction fails, the wrapper records incomplete cleanup, refuses
  unbound signals, and cannot claim successful retirement.
- Original public client deadline remains300s, observer wall600s, outer wall900s,
  <=2048 named stops, combined log<=8MiB, metadata<=1MiB/file. RLIMIT_AS is8GiB
  **per process**, not an aggregate host-RAM claim. Whole new run tree<=512MiB,
  including a128KiB final-report reserve. No model loading or GPU access.

## ROOT commands after independent source review

Use the same already approved user/network isolation and CPU policy as the
accepted public replay. These are source preparation instructions, not a receipt.

```bash
task_run=/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-compiled-f32-routes-20261020
python3 "$task_run/test_supervisor.py"
python3 "$task_run/run_gdb.py" --dataset av --new-output "$task_run/root-av-gdb-1"
python3 "$task_run/validate.py" --run "$task_run/root-av-gdb-1" --new-report "$task_run/root-av-validation-1.json"
python3 "$task_run/run_gdb.py" --dataset alpha --new-output "$task_run/root-alpha-gdb-1"
python3 "$task_run/validate.py" --run "$task_run/root-alpha-gdb-1" --new-report "$task_run/root-alpha-validation-1.json"
```

Five root-only real-process fixtures cover a separate inferior PGID in the same
owned SID, actual pidfd signaling/adoption/reaping, and independently injected
TERM/grace/first-KILL/first-reap failures. Importing the fixture or supervisor
does not launch or signal anything. No fixtures have been executed by the author.

Failure to resolve exact named locations, startup/debugger API differences,
deadline failures, output differences or closure differences is UNQUALIFIED.
Preserve the attempt; do not silently weaken expected outputs or route labels.

Primary API references:

- [GNU GDB Python breakpoints](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Breakpoints-In-Python.html)
- [GNU GDB Python events](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Events-In-Python.html)
- [Linux pidfd_open](https://man7.org/linux/man-pages/man2/pidfd_open.2.html)
- [Linux pidfd_send_signal](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html)
- [Linux PR_SET_CHILD_SUBREAPER](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html)

`UPSTREAM_LICENSE` preserves the original llama.cpp MIT notice. No upstream
arithmetic implementation is copied into these observation scripts.

````

## Source: .agents/runs/numerics-and-context-compiled-f32-routes-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1515; SHA256 `93a4917a716709bb236c6ebcd40859ee7f72fbaaa0e68eebe7f910d63e74b6a3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Compiled F32 route source preparation

- Parent: numerics-and-context; coordinator owns shared plan and execution.
- Owned: this ignored run only. Original AV/alpha helpers and receipts unchanged.

## Prepared

- [x] Inspect exact primary F32 SGEMM/vector source and bounded pinned ELF symbols.
- [x] Retain13 symbol/code ranges and bounded nm/objdump artifacts; no arithmetic port.
- [x] Bind both accepted datasets, all130 outputs and full actual mapped-file sets.
- [x] Prepare named entry observer without private ABI calls/arguments/FP evaluation.
- [x] Prepare owned-SID subreaper/pidfd supervisor, independent signal/reap stages.
- [x] Prepare5 real-process failure fixtures; no author execution.
- [x] Require exact output/operand/maps/public-trace gates before132 named entries.
- [x] Parse Python AST only; source preparation is not runtime qualification.

## ROOT pending

- [ ] Independent source review and any explicit additive repair/freeze.
- [ ] Execute real-process supervisor fixtures in approved isolation.
- [ ] Execute fresh original AV and alpha public clients under owned GDB.
- [ ] Validate all130 complete outputs for each, actual file maps and public traces.
- [ ] Accept only bounded first-entry evidence; preserve unexpected/failed attempts.
- [ ] Coordinator reconcile qualification boundaries in the existing owning plan.

No GDB, operator, model, container, build or test was executed by the author.
No full internal instruction, historical/lost corpus or model eligibility claim.

````

## Source: .agents/runs/numerics-and-context-cpu-mapped-placement-control-20261020/README.md

Original UTF-8 bytes: 6414; SHA256 `77d34d9afaaecbaac127d93efede939a678b05c81746ea3a0ea39426540d617b`.
Resolve embedded relative paths against the original path, not this volume.

````text
Distinct CPU_Mapped + public allCPU placement control, SOURCE ONLY.

- Parent: numerics-and-context. Owner: baseline_review. ROOT alone compiles/runs.
- Genuine original d812350 GGUF/native tokenloop, same `[BLAS,CPU]` handles/order;
  CPU remains last. Both original providers remain loaded and live. Existing V6
  public CPU `use_ref=true` guards remain. Every actual BLAS compute still exits125
  BEFORE delegation. This changes a declared scheduler placement policy; it is
  neither default native execution nor a production numerical repair.
- The only changed constructor argument is CPU compute buft. A real owned
  aligned64 allocation is passed through public `cpu_buffer_from_ptr`; its
  genuine CPU_Mapped type is used through public `buft_alloc_buffer` with a second
  64B owned probe. Public alignment/host/support/type/base/size checks precede use.
  No fabricated private buft, modified backend, tensor payload, or source weight.
- Before each real public split, including the stock direct fused-op probe,
  reserve, reserve_size and allocation routes, public supported-op checks and
  `sched_set_tensor_backend` assign all existing nodes to the original CPU.
  Different compute-buft identity prevents stock pass3's equal-buft upgrade to
  higher-priority BLAS. Original source/op headers and weight buffer/data pointer
  identities on graph nodes/direct sources remain byte exact before/after setters,
  original split, allocation and compute. No graph nodes or numerical parameters
  are constructed, replaced or reordered by the control.
- Metadata scope: <=4096 actual nodes, <=8192 graph-node/direct-source headers,
  <=4 live/64 total scheduler generations. Header snapshots include raw public
  dtype/op/flags/ne/nb/op_params/name/src/view metadata and original observed
  WEIGHTS buffer/data identities. They exclude mutable activation payload/data
  pointers and all private backend/scheduler fields. Full source files are pinned
  before/after by ROOT. This is not a readback of all in-memory learned bytes.
- Every actual scheduler compute must use the same allocated graph, mappings and
  header metadata. Actual CPU_Mapped tensor buffers are required after allocation
  and rechecked before every compute. Its exact contiguous publicCPU call range/backend is recorded
  and independently reconciled against the unchanged V4 public/TLS/use_ref trace.
  Sync is not counted as additional math. Reuse is observed/count-based; a zero
  reuse or reserve_size count never qualifies that route physically.
- Stock context member destruction retires backend objects before scheduler.
  The control permits this original order only outside complete math/preparation,
  records referencing live scheduler generations, and never dereferences a retired
  backend handle. Later preparation/compute requires the original backend alive.
  Only copied header metadata is discarded during scheduler destruction. Both own
  mapped buffers/backing are released after ALL schedulers retire.
- Unknown owners, nested/uncovered routes, changed source/op headers, unsupported
  ops, BLAS math, allocations or public-use_ref failures stay fail closed125.
  There is no fallback, BLAS disable/unload, CPUfirst reorder, buffer override on
  model weights, changed kernel, offload toggle or tolerance/signed-zero exception.

ROOT protocol (fresh output names, one case at a time):

1. Independently review `freeze-manifest.json` and original V6-policy diff.
   Strict build using `build-argv.json`, actual pinned compiler and unchanged
   ROOT `requested.json`/`result.json` schema. Bind all compile inputs before AND
   after, real exit code, fresh output path/bytes/SHA. No source-prep claim is a
   build or provider receipt.
2. ROOT may run prepared parser fixtures: `python3 -m unittest
   test_control_v4 test_placement_trace test_supervisor` from this run. Source-free
   fixtures do not establish actual public route coverage.
3. Run `run_placement.py --mode plain --model 2b --prefix 64 --cpus CPU8
   --seconds 7200 --binary ABS/native_cpu_placement --binary-sha256 ACTUAL_SHA
   --new-output ABS/root-2b8192-plain-1`; then a separate fresh `--mode observed`.
   Explicit eight available CPU IDs; 12GiB address cap; eight native threads;
   finite original/outer deadline;8MiB logs;512MiB individual artifacts. The
   existing ROOT-accepted SID/subreaper/pidfd supervisor is copied unchanged.
   It preserves failed diagnostics and always completes owned cleanup first.
4. `bind_placement.py --manifest-sha256 REVIEWED_SHA --build ABS/root-build-1
   --run ABS/CASE --new-binding ABS/CASE/source-build-binding.json`; it binds
   actual fresh CPP+header+tokenloop/compiler inputs, binary, invocation, provider
   and actual public/placement reports/trace, including failed runs.
5. `validate_placement.py --capture-run ABS/CASE --new-report ABS/CASE/policy.json`.
   Requires all original730 file geometries, actual fullV/whole-state internal
   observer transparency, same mapped/runtime/source pins, exact placement and
   publicCPU ownership/counter conservation, and complete owned cleanup.
6. `compare_same_control.py --plain ABS/PLAIN --observed ABS/OBSERVED
   --new-report ABS/PAIR.json`: both modes execute SAME placement/guards/math.
   Only observed mode adds <=8 public node metadata samples. Complete730 raw
   stage words (including full248320 logits) AND complete native prior/post
   serialized bytes must match bitwise across modes. This comparison never
   asserts default-route transparency. Existing tokenloop also compares its own
   plain/observed tail fullV and whole serialized state inside EACH process.
7. After paired-control transparency, ROOT compares genuine selected-cut730
   stage/fullV bytes to the accepted SAME Rust engine using unchanged
   `numerics-and-context-long-context-audit-20261020/compare_raw.py`. Decode actual
   native post-state using unchanged pinned `state-codec/native_state.py`, export
   accepted Rust checkpoint domains and use unchanged `compare_state.py` for
   ALL42 domains. These downstream comparisons are mandatory, never inferred
   from a successful compute/report or pair. No changed historical-prefix claim.

All V1..V7 source freezes, build/failure receipts and prior control captures stay
immutable. No model/operator/runtime/build/test execution by author. Production
source, MAIN plans and full-model eligibility remain untouched.

````

## Source: .agents/runs/numerics-and-context-cpu-mapped-placement-control-20261020/TODO.md

Original UTF-8 bytes: 1083; SHA256 `15ba25987698b0f40d8e1542d540c5101b55d13c9b39328eeeefcadda0c3e18b`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: `.agents/plans/numerics-and-context/`. Owner: baseline_review.

- [x] Source audit genuine public CPU_Mapped allocation and original CPUlast
  scheduler split/placement/retirement closure; source-only selected ELF evidence.
- [x] Prepare distinct control from immutable V6 with unchanged original tokenloop,
  mapped buffers, metadata/owner guards and exact publicCPU call conservation.
- [x] Prepare ROOT build/binding/owned launcher/policy/paired byte validators and
  bounded source-free refusal fixtures. Author did not execute them.
- [ ] ROOT independent review, strict build and prepared fixtures.
- [ ] ROOT two fresh SAME-CONTROL plain/observed captures; verify original all730,
  fullV, whole serialized states and physical public-use_ref/placement evidence.
- [ ] ROOT independent accepted Rust all730/fullV/ALL42 comparisons. BLAS actual
  math or any unknown source/owner/route remains refused, not an accepted result.
- [ ] Default native transparency, historical lost-prefix reproduction, production
  repair and model eligibility remain outside this control.

````

## Source: .agents/runs/numerics-and-context-cpu-mapped-placement-control-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 6414; SHA256 `77d34d9afaaecbaac127d93efede939a678b05c81746ea3a0ea39426540d617b`.
Resolve embedded relative paths against the original path, not this volume.

````text
Distinct CPU_Mapped + public allCPU placement control, SOURCE ONLY.

- Parent: numerics-and-context. Owner: baseline_review. ROOT alone compiles/runs.
- Genuine original d812350 GGUF/native tokenloop, same `[BLAS,CPU]` handles/order;
  CPU remains last. Both original providers remain loaded and live. Existing V6
  public CPU `use_ref=true` guards remain. Every actual BLAS compute still exits125
  BEFORE delegation. This changes a declared scheduler placement policy; it is
  neither default native execution nor a production numerical repair.
- The only changed constructor argument is CPU compute buft. A real owned
  aligned64 allocation is passed through public `cpu_buffer_from_ptr`; its
  genuine CPU_Mapped type is used through public `buft_alloc_buffer` with a second
  64B owned probe. Public alignment/host/support/type/base/size checks precede use.
  No fabricated private buft, modified backend, tensor payload, or source weight.
- Before each real public split, including the stock direct fused-op probe,
  reserve, reserve_size and allocation routes, public supported-op checks and
  `sched_set_tensor_backend` assign all existing nodes to the original CPU.
  Different compute-buft identity prevents stock pass3's equal-buft upgrade to
  higher-priority BLAS. Original source/op headers and weight buffer/data pointer
  identities on graph nodes/direct sources remain byte exact before/after setters,
  original split, allocation and compute. No graph nodes or numerical parameters
  are constructed, replaced or reordered by the control.
- Metadata scope: <=4096 actual nodes, <=8192 graph-node/direct-source headers,
  <=4 live/64 total scheduler generations. Header snapshots include raw public
  dtype/op/flags/ne/nb/op_params/name/src/view metadata and original observed
  WEIGHTS buffer/data identities. They exclude mutable activation payload/data
  pointers and all private backend/scheduler fields. Full source files are pinned
  before/after by ROOT. This is not a readback of all in-memory learned bytes.
- Every actual scheduler compute must use the same allocated graph, mappings and
  header metadata. Actual CPU_Mapped tensor buffers are required after allocation
  and rechecked before every compute. Its exact contiguous publicCPU call range/backend is recorded
  and independently reconciled against the unchanged V4 public/TLS/use_ref trace.
  Sync is not counted as additional math. Reuse is observed/count-based; a zero
  reuse or reserve_size count never qualifies that route physically.
- Stock context member destruction retires backend objects before scheduler.
  The control permits this original order only outside complete math/preparation,
  records referencing live scheduler generations, and never dereferences a retired
  backend handle. Later preparation/compute requires the original backend alive.
  Only copied header metadata is discarded during scheduler destruction. Both own
  mapped buffers/backing are released after ALL schedulers retire.
- Unknown owners, nested/uncovered routes, changed source/op headers, unsupported
  ops, BLAS math, allocations or public-use_ref failures stay fail closed125.
  There is no fallback, BLAS disable/unload, CPUfirst reorder, buffer override on
  model weights, changed kernel, offload toggle or tolerance/signed-zero exception.

ROOT protocol (fresh output names, one case at a time):

1. Independently review `freeze-manifest.json` and original V6-policy diff.
   Strict build using `build-argv.json`, actual pinned compiler and unchanged
   ROOT `requested.json`/`result.json` schema. Bind all compile inputs before AND
   after, real exit code, fresh output path/bytes/SHA. No source-prep claim is a
   build or provider receipt.
2. ROOT may run prepared parser fixtures: `python3 -m unittest
   test_control_v4 test_placement_trace test_supervisor` from this run. Source-free
   fixtures do not establish actual public route coverage.
3. Run `run_placement.py --mode plain --model 2b --prefix 64 --cpus CPU8
   --seconds 7200 --binary ABS/native_cpu_placement --binary-sha256 ACTUAL_SHA
   --new-output ABS/root-2b8192-plain-1`; then a separate fresh `--mode observed`.
   Explicit eight available CPU IDs; 12GiB address cap; eight native threads;
   finite original/outer deadline;8MiB logs;512MiB individual artifacts. The
   existing ROOT-accepted SID/subreaper/pidfd supervisor is copied unchanged.
   It preserves failed diagnostics and always completes owned cleanup first.
4. `bind_placement.py --manifest-sha256 REVIEWED_SHA --build ABS/root-build-1
   --run ABS/CASE --new-binding ABS/CASE/source-build-binding.json`; it binds
   actual fresh CPP+header+tokenloop/compiler inputs, binary, invocation, provider
   and actual public/placement reports/trace, including failed runs.
5. `validate_placement.py --capture-run ABS/CASE --new-report ABS/CASE/policy.json`.
   Requires all original730 file geometries, actual fullV/whole-state internal
   observer transparency, same mapped/runtime/source pins, exact placement and
   publicCPU ownership/counter conservation, and complete owned cleanup.
6. `compare_same_control.py --plain ABS/PLAIN --observed ABS/OBSERVED
   --new-report ABS/PAIR.json`: both modes execute SAME placement/guards/math.
   Only observed mode adds <=8 public node metadata samples. Complete730 raw
   stage words (including full248320 logits) AND complete native prior/post
   serialized bytes must match bitwise across modes. This comparison never
   asserts default-route transparency. Existing tokenloop also compares its own
   plain/observed tail fullV and whole serialized state inside EACH process.
7. After paired-control transparency, ROOT compares genuine selected-cut730
   stage/fullV bytes to the accepted SAME Rust engine using unchanged
   `numerics-and-context-long-context-audit-20261020/compare_raw.py`. Decode actual
   native post-state using unchanged pinned `state-codec/native_state.py`, export
   accepted Rust checkpoint domains and use unchanged `compare_state.py` for
   ALL42 domains. These downstream comparisons are mandatory, never inferred
   from a successful compute/report or pair. No changed historical-prefix claim.

All V1..V7 source freezes, build/failure receipts and prior control captures stay
immutable. No model/operator/runtime/build/test execution by author. Production
source, MAIN plans and full-model eligibility remain untouched.

````

## Source: .agents/runs/numerics-and-context-cpu-mapped-placement-control-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1083; SHA256 `15ba25987698b0f40d8e1542d540c5101b55d13c9b39328eeeefcadda0c3e18b`.
Resolve embedded relative paths against the original path, not this volume.

````text
Parent: `.agents/plans/numerics-and-context/`. Owner: baseline_review.

- [x] Source audit genuine public CPU_Mapped allocation and original CPUlast
  scheduler split/placement/retirement closure; source-only selected ELF evidence.
- [x] Prepare distinct control from immutable V6 with unchanged original tokenloop,
  mapped buffers, metadata/owner guards and exact publicCPU call conservation.
- [x] Prepare ROOT build/binding/owned launcher/policy/paired byte validators and
  bounded source-free refusal fixtures. Author did not execute them.
- [ ] ROOT independent review, strict build and prepared fixtures.
- [ ] ROOT two fresh SAME-CONTROL plain/observed captures; verify original all730,
  fullV, whole serialized states and physical public-use_ref/placement evidence.
- [ ] ROOT independent accepted Rust all730/fullV/ALL42 comparisons. BLAS actual
  math or any unknown source/owner/route remains refused, not an accepted result.
- [ ] Default native transparency, historical lost-prefix reproduction, production
  repair and model eligibility remain outside this control.

````

## Source: .agents/runs/numerics-and-context-default-bulk-isolated-20261020/CHECKPOINT.md

Original UTF-8 bytes: 4119; SHA256 `4dba3db7e5213577353db028abedc1f606929987122b27e1695503fcb408c701`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Unfinished relocation — not runnable or qualified

ROOT requested a faithful relocation, then relayed the user's pause before
implementation. This checkpoint records source reading/design only. It is not
a prepared-manifest, executable helper, numerical receipt or accepted change.

## Exact owning inputs

- Isolated tree: `/home/pvl/spaces/abird/src/zerogravity-worktrees/root-default-bulk-20261020`.
- Detached base: fb68d8fa165de16e21d1bd37afef3790b80da971.
- ROOT own run: `.agents/runs/numerics-and-context-root-default-20261020`.
- `composition-1.json`: ROOT applied original primitive689221 +attributiona7e8
  +typed0d11 +GPU cold refusals. Uncompiled; MAIN unchanged.
- ROOT `source-freeze-1.json`: supplied SHA
  a4891041aa4fb57d3afac7f180212eddc65d735790dfea0f22f4e61eb7cca1b0.
  Schema fields: status ROOT_ISOLATED_DEFAULT_BULK_UNCOMPILED_SOURCE, root/head,
  source_count463, files[{file,bytes,sha256}], MAIN_merged=false and
  model_or_long_context_qualified=false. The coordinator computed this digest;
  reviewer only read metadata, without recomputing it.
- Original typed run: `.agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020`,
  freeze0d11d476e2955a40eeb1dbe45c478b4502dede72a92ebb2e4f092e7481e7a6e1.
- Original signal followup: same run/followup-signal-v2, freeze
  fce99df1aba6fff0416fdbbf60535ea24d18127740a4fbf50a11053af66ad003.
- Original primitive: numerics-and-context-bulk-f32-primitive-20261020,
  freeze689221fe60b982e93e4684012b5c4bf8e3dc061546e86fc0ed4be7f63c78f61f;
  attributiona7e8cc3b9c2addb2c64ec994efb2368a79cbb359bcac9f7989a9d550ecfe6eb4.
- Original53ee owner and00d1 two-signal record-only guard remain immutable.

## Required narrow adapters

- Local main.rs must remain byte-identical: actual original .8B AV +2B alpha
  default-reference132096 output words, same HostMath three-node slice, full K,
  source/raw learned matrix binding, expected bytes opened after math, real
  pre-cancel/one-byte pressure, retained clone/source and final0.
- Cargo currently uses relative MAIN crate roots. Change only core/model paths
  to the absolute isolated tree. Do not rewrite original source/model paths.
- contracts.py REPO remains original MAIN for approved model files and genuine
  raw/default captures. Add a distinct SOURCE_ROOT for actual compile sources.
  PRIMITIVE remains the original run; parents()/verify_transport() remain exact.
- run_replay_v2.py contains two HERE.parents[2] source joins and assumes a local
  proposed tree. Both must map actual source to isolated SOURCE_ROOT while
  retaining original proposed-file expectations. Preserve runner lifecycle,
  argv/body/caps/signals and original authentic fixture population.
- Source inventory must equal all463 ROOT Rust/TOML/root lock entries, plus
  exact local Cargo/main/effective Cargo.lock. Authenticate all11 typed proposed
  paths (including two GPU refusals) and three primitive paths; seven are new
  Rust children. No extra code, symlink or source alias accepted.
- frozen() and execution_v2.verify_execution() currently assert old local
  preparation identities. New wrappers must authenticate the immutable old
  identities/files/pins separately and return the new relocated identity,
  never pretend modified helper bytes still have old0d11/fce99 hashes.
- New build attestation needs a mandatory separate actual ROOT compiler receipt
  bound to this exact preparation, source freeze, local lock and executable
  digest. A schema claiming source-only/compiled=false cannot qualify a binary.
- Old validators/prepare should stay byte-identical where possible through local
  imports. New scope/provenance checks may wrap them; no expected generation,
  reducer substitutions, tolerance, layer schedule or full8K claims.

## Prepared versus remaining

Only baseline copies and this design/checklist exist here. No new contracts,
runner, Cargo variant, freeze generator or compilation manifest is ready.
No original/raw payload was copied or read. ROOT/user pause takes precedence;
resume from this checklist after authorization resumes, preserving all old
freezes and unrelated work.

````

## Source: .agents/runs/numerics-and-context-default-bulk-isolated-20261020/TODO.md

Original UTF-8 bytes: 1793; SHA256 `50061ea2ed7b6d5bf4bdebea7ba3e1bd1a91a0be367c19548610665d591235f7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Isolated DEFAULT bulk replay — paused source checkpoint

Parent: numerics-and-context. Owner: batch_design_review.
Scope: this ignored directory only. User pause relayed by ROOT; no runnable
relocated replay or source/build acceptance is prepared.

- [x] Read ROOT composition-1.json/TODO and immutable typed/primitive replay
  helpers, original source/transport/validator and execution-v2 contracts.
- [x] Receive explicit ROOT463 source freeze a4891041 metadata; do not hash or
  modify the worktree, MAIN, source payloads or existing frozen artifacts.
- [x] Preserve exact original helper baselines and continuation design here.
- [ ] Copy main/prepare/validate/owner/record-only guard/parser fixtures unchanged
  into a fresh proposed harness; local Cargo only remaps two crate roots.
- [ ] Separate immutable source/model/reference namespace (MAIN) from actual
  build-source namespace (ROOT isolated tree); adapt both runner source joins.
- [ ] Implement whole463 source closure +14 exact primitive/typed affected-file
  checks, local Cargo/main/effective-lock attestation and separate mandatory
  ROOT compiler/binary receipt. Do not treat a source attestation as a build.
- [ ] Authenticate original0d11/689221/a7e8/fce99 files and pins without retagging;
  new preparation/execution provenance must have its own identity.
- [ ] ROOT review and source-only seal/freeze; no author hashes were calculated.
- [ ] ROOT source-free fixtures/strict/build, actual132096-word DEFAULT replay,
  independent unchanged raw-output validator and owner/EOF/final0 review.

No imports/tests/build/runtime/provider/model/GPU execution or binary/DSO/model
payload reads/hashes. No MAIN/worktree/shared-plan edits or commits. Pause is
explicit; do not resume dependent work until ROOT/user resumes.

````

## Source: .agents/runs/numerics-and-context-long-context-audit-20261020/AUDIT.md

Original UTF-8 bytes: 4098; SHA256 `9720d8079d1a81208bc95aae14de002405237af2cc659a52a6fc88c87ec50807`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Surviving evidence and next diagnostic

- Historical queue records independent8K disagreement for BOTH originals: native512 shared-top20 logprob maxima0.20372924/0.19678696; greedy7993 agreement is insufficient. Recorded first differing observed GDN layers4/1 are investigation leads, not fresh bit-level attribution.
- Historical0.8B native256/512 broad observations matched158/158;2B differed149/158. This does not establish AR1 parity; old ubatch1 run did not finish.
- Targeted old `target/qualification` evidence/header/binary folders are empty. No old raw teacher prefix or complete8K state/stage capture was recovered. Fresh root cases in `numerics-and-context-root-20261020/fresh-prefix-cases.json` are explicitly new.
- Surviving `llama_reference_layers.cpp` re-tokenizes UTF8, reports selected broad tensors, and reconstructs unnamed post-FFN residual in its historical converter. Broad comparison defaults allowed tolerances. This cannot establish earlier source bits or causal GDN inputs. New helper captures actual `l_out` tensor; no residual reconstruction.
- Existing `trace_tokens` final-token filter applies after tracing every token. New harness does state-only prefix chunks and enables existing trace only for the selected last token; no duplicated token/layer/math engine.
- Current GGUF recurrent formula matches visible pinned fused-AR structure: expf decay, source FMA dot, delta beta, FMA update, dot then scale. Actual native op choice and inputs still need execution proof. Native cached-state memory and Rust canonical recurrence use value-major rows with key contiguous.
- Native standalone coefficients use pinned glibc2.44 expf/log functions; Rust host libm is separate. Capture actual beta/softplus/gate/raw operands first. New native `gdn_gain` is an explicitly separate direct expf(gate) probe, not a live intermediate or proof of original kernel's exact evaluation. `gdn_scale` similarly is a separate pinned-source width probe. Do not reuse HF SLEEF helpers to alter GGUF silently.
- Callback selection splits native graph work. New plain→save prior→decode selected token versus restored prior→observed selected-token runs MUST have exact full248320 logits AND entire serialized native post-state. Rust Unsupported observer MUST match full logits and all42 complete domain words of its plain selected-token run. Any guard failure invalidates that capture; do not normalize/remove differing bytes without independent evidence.

Bounds: raw IDs≤8192 u32LE; source original only0.8B/2B; selected-token captures≤64MiB, record≤2M words/16MiB native view span, opaque native snapshots≤512MiB each, cuts≤8192, deadline≤7200s, threads1..8, native prefix/ubatch1..512. Root should additionally impose external CPU/RSS/address/time limits; in-process deadlines are cooperative between decode calls. No tracing every prefix token, whole-source copy or GPU use.

Comparison coverage:146 broad live boundaries,187 actual projection input operands,22 GDN fields×18 layers,248320 full logits=730 records. GDN includes actual incoming/proposed recurrent state and conv history/cache. Shape/name/source/file bounds are strict. All F32 bits including zero signs are compared; native gain/scale probe differences remain separately labeled. Comparator reports first OBSERVED boundary only; incoming state differences can originate earlier in the prefix. Cross-provider KV state is not exported; full-state instrumentation guards do include each provider's KV. This is diagnosis, not full inference acceptance.

Smallest next causal controls after512: bisect the earliest observed divergent raw prefix frontier; compare native AR1 versus native bulk with the same selected-token capture; when q/k/v/coefficient/prior operands match, replay only the existing shared recurrent operator against the independent proposed state/core. If any upstream word differs, attribute that gap before touching recurrence. Native ubatch changes may choose different operators; capture actual `GGML_OP_GATED_DELTA_NET` count and never infer formula from a requested number alone.

````

## Source: .agents/runs/numerics-and-context-long-context-audit-20261020/README.md

Original UTF-8 bytes: 4539; SHA256 `90877cd44fdf6dc500bc69d633cce93a45cda08a0e2157bf8ac97c646403d2e1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only fresh raw-prefix diagnostic

Prepared only. No model/native/Cargo/GPU execution by child. No production edits.

Pinned closure/source and original source hashes are in `source-pins.json`; parent's matching-source receipt is `.agents/runs/numerics-and-context-root-20261020/source-acquisition.json`. Do not substitute an unpinned native library or modern math backend.

From repository root, parent builds once (its Cargo coordinator owns all commands):

```sh
raw_run=.agents/runs/numerics-and-context-long-context-audit-20261020
raw_src=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
raw_lib=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
raw_libc=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I "$raw_src/include" -I "$raw_src/ggml/include" "$raw_run/native_raw.cpp" \
  -L "$raw_lib" -L "$raw_libc" \
  -Wl,-rpath,"$raw_lib:$raw_libc" \
  -Wl,--dynamic-linker,"$raw_libc/ld-linux-x86-64.so.2" \
  -lllama -lggml -lggml-base -lm -o "$raw_run/native_raw"
CARGO_BUILD_JOBS=2 cargo build --offline --release --manifest-path "$raw_run/Cargo.toml"
python3 "$raw_run/compare_raw.py" --self-test
```

Standalone manifest has its own lockfile; parent may first generate it offline, then use `--locked`. No download required. The ignored standalone target is independent from shared production output.

Initial0.8B512 AR1 commands, root supervised under finite memory/CPU/time caps (e.g.8GiB address/RSS budget,7200s external deadline; cooperative internal deadline does not interrupt an in-flight kernel). Never start the entire ladder automatically:

```sh
raw_cases=.agents/runs/numerics-and-context-root-20261020
"$raw_run/native_raw" target/models/qwen35-0.8b/model.gguf \
  "$raw_cases/fresh-mixed-v1-512.u32le" 512 1 1 8 7200 "$raw_run/native-0.8b-512-ar1"
RAYON_NUM_THREADS=8 "$raw_run/target/release/rust_raw" \
  target/models/qwen35-0.8b/model.gguf target/models/qwen35-0.8b/config.json \
  "$raw_cases/fresh-mixed-v1-512.u32le" 512 7200 - "$raw_run/rust-0.8b-512"
python3 "$raw_run/compare_raw.py" \
  --native "$raw_run/native-0.8b-512-ar1" --rust "$raw_run/rust-0.8b-512" \
  --model target/models/qwen35-0.8b/model.gguf \
  --model-sha256 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf \
  --raw-ids "$raw_cases/fresh-mixed-v1-512.u32le" \
  --raw-sha256 17892761d801c08052bc5de82c6e4d978796a6a721c77134b71095c3be0eff1c \
  --cut 512 --output "$raw_run/compare-0.8b-512-ar1.json"
```

Repeat2B using its exact original GGUF/config and source SHA91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6. Before/after each native launch, root records model/binary/installed-library hashes and actual argv/env-name declarations in its owned log. Comparator verifies current original model SHA, Rust source SHA, same canonical native/Rust model path and identical raw input bytes; it does not independently authenticate which FD the external library opened during its past invocation.

Native effective batch capacity may be clamped by the library. AR1 here means actual prefix calls carry ONE token and requested ubatch1; effective settings and actual selected-token fused GDN op count are recorded. Initial threads8 preserves the historical setting; explicit4 is a separate control. Bulk64/256/512 changes native prefix-batch and ubatch arguments, with the selected last token always replayed alone. Independent GDN state capture must remain actual fused-operator output, not reconstructed expected state.

Both tools refuse existing output directories and emit a completion marker only after their own instrumentation equality guards. Completion is NOT cross-provider acceptance. Comparator exit0=all730 selected-token records exact; exit2=actual source bit differences (report retained); exit3=malformed/incomplete/unqualified evidence. Signed-zero differences are failures. Native gain and scale are separate probes in the report. No GPU/full-state cross-provider/historical8K claim is inferred.

Rust optional resume requires a checkpoint bound to the same model SHA AND complete raw-ID byte stream SHA. To reuse checkpoints across cuts, pass the SAME8192-ID file and vary `cut` from the beginning; per-cut files have different hashes and intentionally refuse reuse. Root should first validate ordinary fresh512 runs with resume`-`. Native resume is not implemented; opaque prior/post files are diagnostic artifacts, not authenticated reusable checkpoints.

````

## Source: .agents/runs/numerics-and-context-long-context-audit-20261020/REVIEW-20261020.md

Original UTF-8 bytes: 4378; SHA256 `df22539545a5ec6c81246b730dd8b907584d6c374f375b5ffe8b192af0e350cf`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Source review after preparation freeze

Owner: `/root/baseline_review`. No Cargo/compiler/native/model/GPU/operator execution by this reviewer. Python artifact parsing/syntax and Rust formatting only.

## Typed/raw shared engine

No blocker in frozen typed patch `deaf2ddf` or composed raw/state patch `74688961`: exact unchanged source metadata/borrow, private value owner/profile/precision through views/packing, source-exact BF16 OutputInput CPU64/RNE, no unaware-provider fallback, absent session/state authority refusal, pure slice cannot write domains/advance, resident/batch cold refusal before source/allocation/upload. Default GGUF Source/F32 unaffected. Original HF full session/checkpoint guards remain closed. Parent/spine received verdict.

## Closed softplus

No blocker in current MAIN closed softplus child/parent export/golden. Pinned Torch Activation.cpp948–989 FP32 vector branch and SLEEF xlog1pf2848–2896/df.h compensated FMA topology match. Strict `>20`, beta1 identities and length divisible16 avoid scalar tails. Nonlinear `exp(x<=20)` is nonnegative, keeps `dp1>=1` normal and is below `LOG1PF_BOUND=1e38`; omitted negative/huge/subnormal-dp1 branches cannot select. Finite general FP32, source binding/max1M and full BSL notice are explicit; no coefficient/MKLVML/full-profile promotion.

Independently parsed both authentic safetensors headers/payloads and verified archive SHA against manifests: all32 checked-in golden input/output pairs exactly match actual `coefficient.dispatch.006.args.0/result`. Archives `0ff40c27...6028fb` and `8070911a...baf46`. This is golden authenticity inspection, not another runtime replay. Parent's existing actual32/adversarial28464/reordered28464 zero-difference replay log was read, scope unchanged.

## Fresh long-context diagnostic

No blocker for bounded512 investigation:

- Native scheduler executes callback(false) after selected graph interval completion+synchronize. It can alter graph partitioning; complete full logits AND whole serialized post-state equality are required and present.
- Actual projection operand read occurs immediately after its MUL_MAT. Matrix outputs do not alias an operand in place; no later operation has executed yet.
- Actual prior conv view is captured before cache COPY. Actual COPY output gives newest3 channel history; no reconstructed expected cache.
- Native fused GDN src5 is separate from packed output; ops.cpp copies srcstate into output before mutation. Capture source prior and actual proposed-state subspan after2048 core words. Flattened native key-fast axes match existing Rust value-major transport. No transposition or expected-state construction.
- Rust observer returns Unsupported for every implemented hook; ordinary existing fallback/schedule/authority runs. Full-vocab/all42 numerical guard compares plain selected token to restored-prior observed selected token. Post-state capture selects semantic domains by StateSlot.
- Comparator requires730 unique expected names with exact source geometry, matching raw teacher bytes, original model SHA/path, complete instrumentation guards, bounded files/aggregate and finite values. Signed zeros remain failures. Incoming prior mismatch prevents first-current-operator attribution; report says first OBSERVED boundary only. Native gain/scale are separate probes.
- Parent actual native cut2 metadata was read:730 records,45,326,464B, actual n_ctx8448/batch1/ubatch1/threads8/glibc2.44, both guards true. This independently checks emitted metadata/coverage against source only; reviewer did not launch/replay it. Parent owns actual receipt and any runtime acceptance.
- Explicit limitations: cross-provider KV is not exported; native model opened-FD source binding relies on parent's before/after pinned invocation hashes, not retrospective current-file hashing alone; cooperative deadline cannot interrupt an in-flight decode; no authenticated native resume. None may become an implicit whole-model/8K claim.

Root-requested comparator follow-up preserves `frozen-prepared/` and accepts strict integer cuts1..8192 for smoke/bisection. `compare_raw.v2.frozen.py` SHA8ecd7a0b0bf5a4f31e63886362825e56051c22a91337f57e3ded4c85b2a6ffd2; patch6bbd719a95f8fd76eed10e5b25ee5ff87233a4617a9a2958b795faeae4f9cca8. Tiny scopes remain labeled fresh selected-token source bits. Bounds fixtures authored; parent executes.

````

## Source: .agents/runs/numerics-and-context-long-context-audit-20261020/TODO.md

Original UTF-8 bytes: 2174; SHA256 `04528422da35b06678866e142f0b9a254f6e0982b541ce1b8d1eeef0d63f5a08`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Fresh long-context source diagnosis

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: `/root/baseline_review`; parent alone compiles/runs/merges.

- [x] Read canonical numerics queue, surviving native helpers and current CPU/GDN math.
- [x] Locate pinned installed d812350 runtime and verified matching source.
- [x] Inspect missing historical raw-prefix/output/header directories; original 8K input/evidence remains unrecovered.
- [x] Prepare bounded raw-ID native selected-token capture with plain/observed full logits + whole serialized native-state equality guards.
- [x] Prepare same-engine Rust selected-token observer with plain/observed full logits + all42 domain bit guards; every provider hook returns Unsupported.
- [x] Prepare exact730-stage geometry comparator, explicit signed-zero differences and separately labeled native gain/scale probes.
- [x] Rustfmt / Python syntax compilation only. No C++/Rust build, model/provider, native runtime, Cargo or GPU executed by this agent.
- [ ] Parent: compile both helpers + run source-free comparator fixtures.
- [ ] Parent: genuine fresh512 original0.8B AR1 (actual prefix calls1, ubatch requested1, threads8), then2B; inspect actual callback coverage and all instrumentation guards first.
- [ ] Compare all upstream observed bits; do not infer current first causal operator from a differing incoming prior state or historical layer4/layer1 prose.
- [ ] Extend only after bounded512 evidence:1024/2048/4096/8192, same raw-ID prefix and pinned native closure.
- [ ] Native bulk controls64/256/512 separately; record actual GDN operator choice, effective context/batch/ubatch and thread settings.
- [ ] Close actual causes with shared-engine changes + strict operator/state/regression/long-context reruns. No tolerance change or HF/GGUF numerical-profile substitution.

No numerical closure is claimed by preparation or capture completion. Parent's fresh teacher corpus is distinct from the lost historical corpus. Cross-provider42-state equality is not yet provided: this diagnostic exports all18 conv/recurrent domains; each provider's own instrumentation guard covers its complete state, including KV.

````

## Source: .agents/runs/numerics-and-context-long-context-audit-20261020/frozen-prepared/AUDIT.md

Original UTF-8 bytes: 4098; SHA256 `9720d8079d1a81208bc95aae14de002405237af2cc659a52a6fc88c87ec50807`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Surviving evidence and next diagnostic

- Historical queue records independent8K disagreement for BOTH originals: native512 shared-top20 logprob maxima0.20372924/0.19678696; greedy7993 agreement is insufficient. Recorded first differing observed GDN layers4/1 are investigation leads, not fresh bit-level attribution.
- Historical0.8B native256/512 broad observations matched158/158;2B differed149/158. This does not establish AR1 parity; old ubatch1 run did not finish.
- Targeted old `target/qualification` evidence/header/binary folders are empty. No old raw teacher prefix or complete8K state/stage capture was recovered. Fresh root cases in `numerics-and-context-root-20261020/fresh-prefix-cases.json` are explicitly new.
- Surviving `llama_reference_layers.cpp` re-tokenizes UTF8, reports selected broad tensors, and reconstructs unnamed post-FFN residual in its historical converter. Broad comparison defaults allowed tolerances. This cannot establish earlier source bits or causal GDN inputs. New helper captures actual `l_out` tensor; no residual reconstruction.
- Existing `trace_tokens` final-token filter applies after tracing every token. New harness does state-only prefix chunks and enables existing trace only for the selected last token; no duplicated token/layer/math engine.
- Current GGUF recurrent formula matches visible pinned fused-AR structure: expf decay, source FMA dot, delta beta, FMA update, dot then scale. Actual native op choice and inputs still need execution proof. Native cached-state memory and Rust canonical recurrence use value-major rows with key contiguous.
- Native standalone coefficients use pinned glibc2.44 expf/log functions; Rust host libm is separate. Capture actual beta/softplus/gate/raw operands first. New native `gdn_gain` is an explicitly separate direct expf(gate) probe, not a live intermediate or proof of original kernel's exact evaluation. `gdn_scale` similarly is a separate pinned-source width probe. Do not reuse HF SLEEF helpers to alter GGUF silently.
- Callback selection splits native graph work. New plain→save prior→decode selected token versus restored prior→observed selected-token runs MUST have exact full248320 logits AND entire serialized native post-state. Rust Unsupported observer MUST match full logits and all42 complete domain words of its plain selected-token run. Any guard failure invalidates that capture; do not normalize/remove differing bytes without independent evidence.

Bounds: raw IDs≤8192 u32LE; source original only0.8B/2B; selected-token captures≤64MiB, record≤2M words/16MiB native view span, opaque native snapshots≤512MiB each, cuts≤8192, deadline≤7200s, threads1..8, native prefix/ubatch1..512. Root should additionally impose external CPU/RSS/address/time limits; in-process deadlines are cooperative between decode calls. No tracing every prefix token, whole-source copy or GPU use.

Comparison coverage:146 broad live boundaries,187 actual projection input operands,22 GDN fields×18 layers,248320 full logits=730 records. GDN includes actual incoming/proposed recurrent state and conv history/cache. Shape/name/source/file bounds are strict. All F32 bits including zero signs are compared; native gain/scale probe differences remain separately labeled. Comparator reports first OBSERVED boundary only; incoming state differences can originate earlier in the prefix. Cross-provider KV state is not exported; full-state instrumentation guards do include each provider's KV. This is diagnosis, not full inference acceptance.

Smallest next causal controls after512: bisect the earliest observed divergent raw prefix frontier; compare native AR1 versus native bulk with the same selected-token capture; when q/k/v/coefficient/prior operands match, replay only the existing shared recurrent operator against the independent proposed state/core. If any upstream word differs, attribute that gap before touching recurrence. Native ubatch changes may choose different operators; capture actual `GGML_OP_GATED_DELTA_NET` count and never infer formula from a requested number alone.

````

## Source: .agents/runs/numerics-and-context-long-context-audit-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 4539; SHA256 `90877cd44fdf6dc500bc69d633cce93a45cda08a0e2157bf8ac97c646403d2e1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only fresh raw-prefix diagnostic

Prepared only. No model/native/Cargo/GPU execution by child. No production edits.

Pinned closure/source and original source hashes are in `source-pins.json`; parent's matching-source receipt is `.agents/runs/numerics-and-context-root-20261020/source-acquisition.json`. Do not substitute an unpinned native library or modern math backend.

From repository root, parent builds once (its Cargo coordinator owns all commands):

```sh
raw_run=.agents/runs/numerics-and-context-long-context-audit-20261020
raw_src=/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0
raw_lib=/nix/store/0wcm8r7hr02vj7w4l3dlpyrhzsknf5mi-llama-cpp-0.6.0/lib
raw_libc=/nix/store/h4wfwic161kxrr74jlzla5lsm28hgary-glibc-2.44-25/lib
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I "$raw_src/include" -I "$raw_src/ggml/include" "$raw_run/native_raw.cpp" \
  -L "$raw_lib" -L "$raw_libc" \
  -Wl,-rpath,"$raw_lib:$raw_libc" \
  -Wl,--dynamic-linker,"$raw_libc/ld-linux-x86-64.so.2" \
  -lllama -lggml -lggml-base -lm -o "$raw_run/native_raw"
CARGO_BUILD_JOBS=2 cargo build --offline --release --manifest-path "$raw_run/Cargo.toml"
python3 "$raw_run/compare_raw.py" --self-test
```

Standalone manifest has its own lockfile; parent may first generate it offline, then use `--locked`. No download required. The ignored standalone target is independent from shared production output.

Initial0.8B512 AR1 commands, root supervised under finite memory/CPU/time caps (e.g.8GiB address/RSS budget,7200s external deadline; cooperative internal deadline does not interrupt an in-flight kernel). Never start the entire ladder automatically:

```sh
raw_cases=.agents/runs/numerics-and-context-root-20261020
"$raw_run/native_raw" target/models/qwen35-0.8b/model.gguf \
  "$raw_cases/fresh-mixed-v1-512.u32le" 512 1 1 8 7200 "$raw_run/native-0.8b-512-ar1"
RAYON_NUM_THREADS=8 "$raw_run/target/release/rust_raw" \
  target/models/qwen35-0.8b/model.gguf target/models/qwen35-0.8b/config.json \
  "$raw_cases/fresh-mixed-v1-512.u32le" 512 7200 - "$raw_run/rust-0.8b-512"
python3 "$raw_run/compare_raw.py" \
  --native "$raw_run/native-0.8b-512-ar1" --rust "$raw_run/rust-0.8b-512" \
  --model target/models/qwen35-0.8b/model.gguf \
  --model-sha256 57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf \
  --raw-ids "$raw_cases/fresh-mixed-v1-512.u32le" \
  --raw-sha256 17892761d801c08052bc5de82c6e4d978796a6a721c77134b71095c3be0eff1c \
  --cut 512 --output "$raw_run/compare-0.8b-512-ar1.json"
```

Repeat2B using its exact original GGUF/config and source SHA91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6. Before/after each native launch, root records model/binary/installed-library hashes and actual argv/env-name declarations in its owned log. Comparator verifies current original model SHA, Rust source SHA, same canonical native/Rust model path and identical raw input bytes; it does not independently authenticate which FD the external library opened during its past invocation.

Native effective batch capacity may be clamped by the library. AR1 here means actual prefix calls carry ONE token and requested ubatch1; effective settings and actual selected-token fused GDN op count are recorded. Initial threads8 preserves the historical setting; explicit4 is a separate control. Bulk64/256/512 changes native prefix-batch and ubatch arguments, with the selected last token always replayed alone. Independent GDN state capture must remain actual fused-operator output, not reconstructed expected state.

Both tools refuse existing output directories and emit a completion marker only after their own instrumentation equality guards. Completion is NOT cross-provider acceptance. Comparator exit0=all730 selected-token records exact; exit2=actual source bit differences (report retained); exit3=malformed/incomplete/unqualified evidence. Signed-zero differences are failures. Native gain and scale are separate probes in the report. No GPU/full-state cross-provider/historical8K claim is inferred.

Rust optional resume requires a checkpoint bound to the same model SHA AND complete raw-ID byte stream SHA. To reuse checkpoints across cuts, pass the SAME8192-ID file and vary `cut` from the beginning; per-cut files have different hashes and intentionally refuse reuse. Root should first validate ordinary fresh512 runs with resume`-`. Native resume is not implemented; opaque prior/post files are diagnostic artifacts, not authenticated reusable checkpoints.

````

## Source: .agents/runs/numerics-and-context-long-context-audit-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 2174; SHA256 `04528422da35b06678866e142f0b9a254f6e0982b541ce1b8d1eeef0d63f5a08`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Fresh long-context source diagnosis

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: `/root/baseline_review`; parent alone compiles/runs/merges.

- [x] Read canonical numerics queue, surviving native helpers and current CPU/GDN math.
- [x] Locate pinned installed d812350 runtime and verified matching source.
- [x] Inspect missing historical raw-prefix/output/header directories; original 8K input/evidence remains unrecovered.
- [x] Prepare bounded raw-ID native selected-token capture with plain/observed full logits + whole serialized native-state equality guards.
- [x] Prepare same-engine Rust selected-token observer with plain/observed full logits + all42 domain bit guards; every provider hook returns Unsupported.
- [x] Prepare exact730-stage geometry comparator, explicit signed-zero differences and separately labeled native gain/scale probes.
- [x] Rustfmt / Python syntax compilation only. No C++/Rust build, model/provider, native runtime, Cargo or GPU executed by this agent.
- [ ] Parent: compile both helpers + run source-free comparator fixtures.
- [ ] Parent: genuine fresh512 original0.8B AR1 (actual prefix calls1, ubatch requested1, threads8), then2B; inspect actual callback coverage and all instrumentation guards first.
- [ ] Compare all upstream observed bits; do not infer current first causal operator from a differing incoming prior state or historical layer4/layer1 prose.
- [ ] Extend only after bounded512 evidence:1024/2048/4096/8192, same raw-ID prefix and pinned native closure.
- [ ] Native bulk controls64/256/512 separately; record actual GDN operator choice, effective context/batch/ubatch and thread settings.
- [ ] Close actual causes with shared-engine changes + strict operator/state/regression/long-context reruns. No tolerance change or HF/GGUF numerical-profile substitution.

No numerical closure is claimed by preparation or capture completion. Parent's fresh teacher corpus is distinct from the lost historical corpus. Cross-provider42-state equality is not yet provided: this diagnostic exports all18 conv/recurrent domains; each provider's own instrumentation guard covers its complete state, including KV.

````

## Source: .agents/runs/numerics-and-context-root-20261020/TODO.md

Original UTF-8 bytes: 7264; SHA256 `3c3f9d4844cb963d1d9a90b71fc0a0af7cc4f4d04d0acf27aca90b86ae93d600`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Numerics and context root continuation

Parent: .agents/plans/numerics-and-context/TODO.md.
Owner: root coordinator; source/benchmark/math implementation scope authorized.

- [x] Re-read current numerical TODO/runbook and recorded8K divergence boundaries.
- [x] Promote existing plan to ActiveP1 per owner's20Oct required correctness scope.
- [x] Confirm original pinned runtime/CPUplugins/separate glibc still installed;
  matching devheaders and priorraw8Ktraces/checkpoints are absent in named dirs.
- [x] Acquire exact v0.6.0 source37,874,349B into projectXDGcache; independently
  match the PREEXISTING Nix recursiveSHA256 before trusting headers/source.
  No installedlibrary/driver/device privileges changed.
- [ ] Baseline agent prepares bounded raw-ID native/Rust GDN operand/state/stage
  comparator using current source, explicitthreads/n_batch/n_ubatch/F32KV/nonflash.
- [ ] Root review/compile; genuine <=512AR1 control first, then1024/2048/4096/8192
  and64/256/512chunk controls as needed to locate first cause, notjustfinalstage.
- [ ] Implement source-derived fix in SAME CPU/GPU engine with no tolerancewiden.
- [ ] Rerun shortcorpus and realindependentlongcontext forbothapprovedGGUFmodels;
  source-bit, cross-plan numeric and quality scopes remain separate.
- [ ] Full3modelcorrectness/serving+modelbenchmarks/sharedmachineryperformance
  gates remain open in owning model/serving plans; no vLLM win claimed.

Fresh prefixes must be identified separately; historical exact8192-ID source lost.
Known historical .8Blayer4/2Blayer1 mismatch is a diagnostic lead, not current
first-bit proof. Internal CPU/B1/native exactness does not close independent drift.

## Fresh actual continuation through4K

- [x] Both original models cuts512/1024/2048/4096 nativeAR1 vsRust selected
  730actualrecords/full248320 allbits exact; independently decoded ALL42 active
  causal post-state domains exact.4K each30,216,192 words/zero differences.
  .8BRust4K961.52s,2B1203.20s; diagnostics, never servingbenchmarks.
- [x] Fresh.8B1024nativebulk64 differs543selectedstages/9,230,360words, first
  observed selectedlayer4incomingGDNstate. Tinycut3control exact; no arbitrary-cut
  causal bisection because changingcut changes actual chunk shape.
- [x] Unchangedfirst64nativebulk/AR1 transparentcapture nowcomplete: commonwhole
  prior exact, bothownplain/observedfullpoststate+nextAR1fullV/wholeState exact.
  Initial attempts refused physicalVspan thenAR1artifactcap; both preserved.
  Narrow read-only logicalVstride copy and artifact-only AR192MiB/bulk128MiB,
  combinedsnapshots136MiB/wholeattempt512MiB fixed instrumentation only.
- [x] FirstOBSERVED resultgap layer3AV(kqv) actualposition2:530/2048 words.
  Independentrawbyteaudit: all64probabilityinputs131072words exact,114432future
  probabilitywords positive0; allcausalV1064960 comparisons exact; finalfullVcache
  hash exact. Earliercapturedlayers0–2/allselectedGDNprior+finalstates exact.
  248rowfielddifferences overall; internal reducer CAUSEstillunqualified.
- [ ] Rootreview/compile/execute prepared publicggml AV replay65default cases
  againstactualcapturedoutputs BEFORE any separatelylabeled publicuse_ref control.
  Source/numericaltolerances unchanged. No historical8Kcorpus recovered.
- [ ] Fresh6K/8K and2Bbulk causal controls after current serialized performance
  waves; do not contaminatebench with CPU/model/Cargo/GPU diagnostics.

## Actual public AV and fresh8K continuation21:06UTC

- [x] Genuinepublicggml AV65defaults262144words matchowncapturedoutputs0diff. Publicuse_refbulk57823wordsdifferentfromdefaultbulk,0difffromall64defaultAR1outputs; publiccontrolonly,noproductionmathchange.
- [x] Transparentpublicentryobserverrootactualplain/observed all130outputs524288wordseach matchoriginalacceptedoutputs.576genuinepubliccalls:bulk64true8/head,64AR1×8falsehead0,65refcasesnone. ExactopaqueCABI/pinnedCPUorigin/boundedintegerrecording/ownedcleanup; independenttwoagentreviewsNOBLOCKER. Internalcompiledkernel/fullruntime/longcontextcauseclosureunqualified.
- [x] NativeAR1fresh8192bothmodelswholeplain/observedfullV/stateexact; decoded42domains55382016words/model RAW_EXPORT_ONLY. Rust6144continuationsbothrunningfromaccepted4K,8Knotcomparedyet.
- [ ] On6144successstart8192fromnewcheckpoint; strict730stage/fullV +independentall42comparison.
- [ ] 2Bunchangedfirst64causalhelper sourceprepared/executeafterrootreview. Olderindependentlongcontextrequiredgatestaysopen.

## Actual2B causal continuation21:39UTC

- [x] BothRust6144selfcaptures complete730/fullV/all42 transparent;8192continuationsrunningfromnewcheckpoints. Notanindependent6144comparison.
- [x] Actual2Bunchangedfirst64nativecapture complete11.8965s afterreviewedopaqueCPU_REPACKtransportfix; originalSIGSEGV/GDBattemptspreserved. All28physicalsourceowners/readsetexactafterbothmodes,plain/observedwholepost+nextAR1fullV/wholeStateexact. EightopaqueCPU_REPACKweightsstaycodecUNQUALIFIED.
- [x] Actualstrictfieldcomparator31.5387s/5570differingfieldrows. Firstobservedgap layer0alpha F32MUL_MAT position0,13/16words; all960precedingcomparedrows/all131072alphainputwords+full131072sourceweightbytesexact. Betaalso differsbeforeGDN. Spinepreparingsource-onlygenuinepublicalphareplay/control/entryobserver; noproductmath/tolerancechange.
- [ ] On8Kcompletionstrict730/full248320 plusindependentlydecodedall42activecausalstatecomparison;requiredlongcontextgatestaysopen.

## Actual2B8192 and alpha results21:51UTC

- [x] Rust2B8192capturecomplete2345.422s/RSS1986156KiB,730/fullV/all42transparent. FreshnativeAR1 comparison10.589s/730stages/all248320ZEROdiff; exportedactualcheckpoint221528064B; strictdecoded42domains/55382016wordsZEROdiff26.976s. Nativebatch1/F32KV/FAoff/CPU8 explicit; freshcorpus, notlosthistoricalreplay.
- [x] Original2Bpublicalpha ROOTcompiledstrictGCC15.3/glibc2.44 binary92f41219/observer519b1f77. Originalee78wrapperheldunexecuted; additive3496/0e79reviewedand4actualfailurefixturesPASS1.599s. Bothactualplain1.850s/observed2.007s process0/allownedcleanupPASS. Validatorall65defaults2048words matchcaptures/all1304096wordsplain==observed. Publicrefbulkexactall64defaultAR1/refAR1unchanged. Actual520entrycalls8bulktrue/512ARfalse/refzero; internalkernel/causeclosurefalse. Productionmath/limitsunchanged.
- [ ] .8B8Kstillrunning; same730/fullV+decodedall42checks oncompletion. Nativeinternalroute source/GDBobservationpreparationassignedSpine; nohiddenFPdebugcalls/privateABIguessing.

## BOTH fresh8192 comparisons PASS22:04UTC

- [x] .8B8192 actualcapture2103.161s/RSS1288160KiB/730/fullV/all42transparent. Strict730/full248320nativeAR1comparison7.681sZEROdiff. Realcheckpointexport221528064B2.971s; independentall42causalstate55382016wordsZEROdiff22.059s.
- [x] BOTHoriginalGGUFmodels freshmixedUnicode8192-IDcorpus exactagainstpinnednativeCPU8/AR1/F32KV/FAoff. Actualprefix_requested1/ubatch1/context8320rounded8448 explicit. Thisdoesnotestablishlosthistoricalcorpusreplay, nativebulkbit equivalence, fullruntimeclosure, GPU8K orHFmath. Noproductionfix/sourcearithmetic/tolerancechange.
- [ ] Compiledinternalrouteobservation sourcepreparedSpine/GDBnamedentries/pidfdsubreaperboundedlauncher; executeonlyafterreview. Remainingnativebulk/referenceprofile/historyissueexplicitlyopen.

````

## Source: .agents/runs/numerics-and-context-root-default-20261020/TODO.md

Original UTF-8 bytes: 797; SHA256 `791a050e06617cb7a60a0bf582daa014f2706f5f87d1815f6206ba70ebd00f5e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# ROOT isolated DEFAULT bulk continuation

Parent: numerics-and-context. Owner: ROOT.

- [x] Reviewed primitive689221/attributiona7e8 +typed0d11/signalv2 source previously; independent Batch review no blocker.
- [x] Fresh ROOT-owned detached fb68 worktree; four exact source patches only, preserving MAIN and all other trees.
- [ ] Source-free strict/unit checks, complete source freeze and faithful relocated replay/build attestors.
- [ ] Actual authentic132096-word DEFAULT operand replay/independent validator; no expected generation.
- [ ] Full temporal/source-profile schedule, real model contexts and historical recovery remain open.

No build, replay, numerical acceptance or MAIN merge yet. ROOT sole Cargo/test coordinator; current CUDA release job97420 remains exclusive until complete.

````

## Source: .agents/runs/numerics-and-context-scheduler-placement-audit-20261020/AUDIT.md

Original UTF-8 bytes: 7073; SHA256 `c199d35ad02136a5b9df8d01a3b97ecde6cc95f9e2616bc170a994e5c2faa949`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pinned public CPU-placement feasibility

- Owner baseline_review; parent numerics-and-context. SOURCE/SELECTED-ELF ONLY.
- No author provider/model/operator/C++ build/test/runtime; no MAIN/plan edits.
- Pinned source: verified v0.6.0/d812350 acquisition receipt at
  `../numerics-and-context-root-20261020/source-acquisition.json`.
- Actual V7 ROOT trace identifies first refused2B BLAS graph: Q8_0
  blk.0.ssm_out.weight[2048,2048] × final_output-0 F32[2048,64], producing
  linear_attn_out-0 F32[2048,64], followed by RESHAPE. No alpha attribution.
  V7 still refused125 before delegation; neither a capture nor a math fix.

Source paths below relative to that verified llama.cpp directory.

## Stock controls that do not guarantee this route

- `src/llama-context.cpp344–361`: adds every registered ACCEL including BLAS,
  then CPU last. `llama_model_params.devices` GPU selection does not omitACCEL.
- `ggml/src/ggml-blas/ggml-blas.cpp408–436`: MUL_MAT requires both contiguous,
  input F32 and output rows>=32, batch>=32, input width>=32; source Q8_0 has
  to_float, so the actual2048×64 operation qualifies. Earlier alpha output16
  does not. `supports_buft`452 accepts ANY host buffer type.
- `op_offload=false` only disables one optional offload branch1110. Ordinary
  highest-priority compatible-weight selection1034–1042 still picks BLAS.
- `include/llama.h314–324` public tensor_buft_overrides changes weight location,
  not operation priority. `src/llama-model-loader.cpp1239–1243` CPU override
  even reselects compatible CPU extra bufts. Plain CPU/CPU_Mapped both host;
  neither alone excludes BLAS or preserves a CPU operation assignment.
- CPU_REPACK is nonhost but is NOT a generic F32 escape. Repack.cpp5191–5225
  requires an actual supported optimized repack type; Q8_0 branch5105 has
  NEON/RISCV candidates, no Haswell/AVX2 candidate. Do not force F32/thisQ8 into
  an unsupported codec or reclassify raw/source bytes to force eligibility.
- Public sched_set_tensor_backend(CPU) BEFORE allocation is not sufficient:
  scheduler pass3 (`ggml-backend.cpp1378–1401`) upgrades assigned nodes to
  higher priority when their DEFAULT SCHEDULER buft pointers are equal and
  sources supported. CPU/BLAS defaults are both the same CPU buft.
- CPU-first[CPU,BLAS] is INVALID: constructor2013 requires CPU last. Inputs1086
  and expansion1279/1300 assume lastCPU. Selected actual BASE ELF confirms
  38aba→get_device@PLT→dev_type@PLT, test38acb/jne38e4c→abort38e77. Do not
  evade this with a duplicate CPU-handle entry: no duplicate-backend scheduler
  ownership/semantics contract has been established.
- Setting placement after allocation changes the public mapping, not already
  stored split ownership. Do not claim that retargets actual computation.

No stock llama.h knob found that guarantees bulk64 pureCPU with both original
backends live and original buffers. Smaller microbatches are a legitimate but
DIFFERENT predeclared workload; they cannot stand in for the same bulk64 proof.

## Feasible DISTINCT public-API placement control

A source-derived candidate, NOT IMPLEMENTED/QUALIFIED:

1. Keep exact original backend handles/list/order[BLAS,CPU], CPU last, every
   model buffer/payload/dtype/ne/nb, flags, threads and op_offload unchanged.
   Retain V6 legal-null/live-owner/use_ref/TLS/plan guards. BLAS remains live
   and registered; every actual BLAS compute must still refuse125.
2. Obtain the genuine built-in CPU_Mapped buft through public
   cpu_buffer_from_ptr(aligned own64B)→buffer_get_type. Retain that temporary
   buffer/backing until all schedulers retire; do not rely on private layout
   or fabricate a buft. Public get-name/is-host/alignment and supports_buft on
   originalCPU must be checked. CPU_Mapped is host, distinct from CPU default.
3. At original PUBLIC sched_new, copy its tiny buft argument array and change
   ONLY the CPU intermediate-allocation buft to that genuine CPU_Mapped type.
   Preserve both original backends/order, BLAS buft, graphsize/parallel/offload
   and caller-owned arrays. Delegate original constructor. Public scheduler
   getters must confirm exact retained handles/bufts; all unknowns coldrefuse.
4. Immediately BEFORE original PUBLIC sched_split_graph (not merely reserve
   entry), enumerate actual graph via public node APIs and assign every actual
   node to originalCPU using public sched_set_tensor_backend. Check op support
   and finite node-count bounds; do not edit tensor fields/source/operation.
   Pass1 preserves user assignment; pass2 leaves assigned nodes; pass3 cannot
   upgrade CPU to BLAS because scheduler buft identities now differ; pass4/5
   preserve those node assignments. Query each public node mapping afterward
   and before compute, require CPU. Original source/weight buffers stay put.
5. This covers reserve_size's INTERNAL reset: reserve_size resets then splits,
   so assigning only before reserve_size would be discarded. Actual selected
   BASE reserve_size3c3b5 and alloc_graph3c561 call PUBLIC split_graph@PLT17ca0;
   relocationdb548 targets that export. Reserve normal path must also be
   compiled/runtime-observed; reused graphs require validated prior ownership.
6. Keep original allocator/split/graph execution. Do not duplicate token/layer
   loops, copy/reinterpret weights, route graphs through another engine, turn
   off BLAS, reorder invalidly, fake Hadamard hints, or allow BLAS delegation.

Why the candidate is possible: public scheduler constructor accepts per-backend
bufts and checks supports_buft; CPU supports any host buft (`ggml-cpu.cpp489`).
Pinned BASE CPU_Mapped type2651–2666 has original CPU aligned-malloc alloc_buffer,
original alignment and is_host=true. It is not allocation-null. Selected actual
ELF public buffer_from_ptr3e413 references type dc040; second pointer dc048 is
35f40, the original CPU alloc callback (static bytes only). That callback passes
through the supplied genuine buft into ordinary CPU buffer initialization.
No private table or offset would be read by the proposed RUNNING control.

This changes declared reference CONTROL placement and intermediate buffer-type
identity. It is not default-native transparency, a serving optimization, or an
arithmetic acceptance claim. CPU_Mapped does not itself disable BLAS: explicit
pre-split CPU assignments are also necessary. Unexpected opaque/graph/plan or
BLAS compute routes must keep failing closed.

ROOT acceptance before any causal conclusion: exact public-provider/PLT/runtime
mapping and args, public constructor+split/free coverage for both contexts,
all unchanged model/source bytes and op metadata, CPU8/thread/resource/lifetime
bounds, plain-versus-observed SAME-CONTROL fullV+serializedstate guards, then
all730/fullV/all42 against accepted Rust. A separate pass-through observer mode
would need its own genuine default transparency proof; never merge a failed
V6/V7 default schedule into a placement-control PASS. No source edit proposed
until ROOT selects this explicit control and its bounded evidence contract.

````

## Source: .agents/runs/numerics-and-context-scheduler-placement-audit-20261020/TODO.md

Original UTF-8 bytes: 501; SHA256 `11e35c78bb040960afca781a4c520412b0daa9d128e5ee0e343fa48170231cc5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Scheduler placement source audit

- Parent numerics-and-context; baseline_review own ignored notes only.
- [x] Read actual2B V6/V7 metadata and pinned public scheduler/BLAS/loader code.
- [x] Confirm selected compiled CPU-last guard and genuine Mapped allocator.
- [x] Describe stock-control refusals and distinct public placement candidate.
- [ ] ROOT decision/implementation/source review/runtime evidence; nothing enabled.
- Author no native/model/operator/math/test/build/runtime or MAIN edits.

````

## Source: .agents/runs/numerics-and-context-state-codec-20261020/README.md

Original UTF-8 bytes: 6889; SHA256 `273e674eac1954805f3f129d9e230083475053aa13f6ee88664c5bfd27e8930f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Actual native/Rust post-state codec export

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: `/root/spine_review`. Root alone builds/runs exporters and comparisons.

PREPARED ONLY. No Cargo, native/model/operator/runtime/GPU execution by author.
No production files, frozen helpers or existing capture output changed.

- `native_state.py`: pinned public `llama_state_get_data` raw-byte decoder.
  No external library is loaded. Exact original0.8B/2B GGUF+config profile only;
  private sequence0, contiguous text positions0..cut-1, cut1..8192.
- `export_rust.rs`: existing `LoadedQwenModel::open` source binder + public
  `load_checkpoint`/`QwenCheckpoint::domain_values`. No session/root/forward or
  model arithmetic is executed. Existing decoder validates all finite values,
  typed domains, source/input digests, state bytes/fingerprint and exact EOF.
- `compare_state.py`: all42 exact complete active numerical domains; no
  tolerance, including signed zero. Exit0=all42 exact; exit2=bit differences;
  exit3=malformed/unqualified. First mismatch coordinates/bits + maxabs retained.
- `test_codec.py`: source-free synthetic metadata/axis/nonfinite/short/trailing
  fixtures; unexecuted by author. Distinct token/head/dimension markers test V
  transpose instead of relying on zeros.

## Source-derived wire closure

Pinned llama.cpp0.6.0/build11429/d812350 source was independently acquired by
root; `source-profile.json` binds its receipt, files and installed libraries.
Raw API is architecture-length+architecture, without session-file magic/version.
The outer profile is mandatory: raw format lacks model SHA, layer IDs and head
counts, so it is insufficient as a self-describing/authenticated checkpoint.

1. `llama-context.cpp:3400`: u32-length `qwen35` string; hybrid memory follows.
   `llama-io.cpp:5` fixes length encoding; native capture enforces little endian
   and four-byte float. Types `llama_pos`, `llama_seq_id`, `llama_token` are i32.
2. `llama-memory-hybrid.cpp:190`: attention first, recurrent second.
3. `llama-kv-cache.cpp:2071`: u32 stream count1, u32 occupied cell count=cut.
   Each cell (`:2217`) stores pos:i32,n_seq:u32, IMRoPE extension
   x:i32,y:i32,tok:i32, then sequence:i32. Refuse anything except actual private
   seq0, pos=x=y=index and exact teacher token. `llama-batch.cpp:125` expands
   text position[p]→[p,p,p,0]; cache `:1149` stores actual extension/token.
4. `llama-kv-cache.cpp:2250`: v_trans:u32=1,n_layer:u32=6,n_rot_k/v:u32=0.
   In ascending filtered layer order3/7/11/15/19/23: each K has type:i32=F32(0),
   row bytes:u64=2048, then occupied rows in physical cell order. This fixture
   requires that order to equal contiguous logical positions; no sorting or
   inferred holes. Each V has type:i32=0, element bytes:u32=4,width:u32=512,
   then actual channel-major occupied cells. Copy V bits to token-major.
   `:1355` K merges contiguous[dimension,head]; `:1514` V index is
   channel*capacity+cell. No dequantization, rotations or RoPE recomputation.
5. Layers/geometry come from original GGUF/config digests and model sources:
   `models/qwen35.cpp:4`, `llama-model.cpp:2705/:2772`, KV allocation ascending
   layer loop `llama-kv-cache.cpp:165`. Both originals have2KV heads×256.
6. `llama-memory-recurrent.cpp:784/:896/:915`: one recurrent cell at cut-1,
   seq0; s_trans=0,n_layer=24. Ascending18 filtered layers: F32 R rows73728B,
   then F32 S rows1048576B. No PLE rows in this pinned Qwen profile. Exact EOF.
   `llama-hparams.cpp:208/:236` fixes counts. Conv row channels×oldest3 history
   from `models/delta-net-base.cpp:449`; GDN S is[head,value,key], key contiguous
   (`ggml-cpu/ops.cpp:11227`). Copy both unchanged, not a recurrence replay.

Canonical KV domain is all K[token,head,dimension] followed by all V in same
order. Canonical convolution and recurrent layout match existing Rust domain
descriptors. All42 domains come from actual post-state bytes, including the six
KV domains omitted by the earlier730-record callback capture.

## Boundaries

- Both portable encodings contain only occupied causal KV, not unused physical
  future capacity. No future padding equality or GPU completion/resource claim.
- Native opaque file itself authenticates no source or historical execution.
  Export checks current source bytes, teacher bytes, outer native completion
  record and pins. Parent retains actual binary/library/argv/instrumentation
  receipts; no claim that a caller-supplied hash independently proves provenance.
- Existing730-record captures and their whole-native-state instrumentation
  equality remain unchanged. This adds a separate direct raw post-state check;
  logits and prior/operator evidence retain their existing independent reports.
- File/numerical output cap512MiB per exporter; at8192 numeric payload is
  221528064B/model/provider. Raw native reader retains at most six16MiB K slabs
  plus bounded working buffers; no full-model/source mmap or second engine.
  Rust public decoder and `domain_values` can hold two complete state payloads
  plus its existing source owner. These are logical bounds, not physical host
  ResourceArbiter/RAM admission. Root applies actual process/time limits.
- Fresh512/1024/etc inputs are distinct from lost historical8K inputs. No long
  GDN/source-profile cause closure or whole-model eligibility from export alone.

## Root-only commands

Use independently verified actual teacher-stream digest, not a per-cut digest
when captures/checkpoints were bound to the common8192-ID file. Require fresh
exclusive output directories. No exporter starts a provider/model evaluation.

```sh
codec=.agents/runs/numerics-and-context-state-codec-20261020
raw=.agents/runs/numerics-and-context-long-context-audit-20261020
ids=.agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le
ids_sha=44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d
# ROOT build coordinator only, offline; alternatively use existing rustc artifacts.
CARGO_BUILD_JOBS=2 cargo build --offline --release --manifest-path "$codec/Cargo.toml"
# ROOT only: source-free fixtures first.
python3 "$codec/test_codec.py"
python3 "$codec/native_state.py" --run \
  --capture "$raw/native-0.8b-512-ar1" --model target/models/qwen35-0.8b/model.gguf \
  --config target/models/qwen35-0.8b/config.json --raw-ids "$ids" --raw-sha256 "$ids_sha" \
  --cut 512 --output "$codec/root-native-0.8b-512"
"$codec/target/release/export_rust" target/models/qwen35-0.8b/model.gguf \
  target/models/qwen35-0.8b/config.json "$ids" "$ids_sha" 512 \
  "$raw/rust-0.8b-512/rust.post-checkpoint.bin" "$codec/root-rust-0.8b-512"
python3 "$codec/compare_state.py" --native "$codec/root-native-0.8b-512" \
  --rust "$codec/root-rust-0.8b-512" --output "$codec/root-compare-0.8b-512.json"
```

Inspect actual source-bound result before any2B/longer case. No automatic ladder.

````

## Source: .agents/runs/numerics-and-context-state-codec-20261020/TODO.md

Original UTF-8 bytes: 918; SHA256 `21f066d5f92e8b6fcdef11b99c95456dc89e21b38612881e7f16ab38db570838`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: `/root/spine_review`; root owns all execution/shared-plan reconciliation.

- [x] Read pinned public native serializer and current public Rust decoder.
- [x] Source-derive actual text cell metadata, K/V axes, filtered layer order,
  complete R/S row geometry and occupied-causal-only scope.
- [x] Prepare exclusive bounded native exporter, existing public Rust decoder
  exporter, strict42-domain comparator and source-free negative/axis fixtures.
- [ ] Root/independent source review, compilation and source-free fixtures.
- [ ] Root actual existing512 native/Rust export and exact42-domain comparison.
- [ ] Root2B/longer source-bound actual proofs and GDN source-profile cause closure.

No Cargo/provider/model/math/runtime/GPU execution, no production edits.
Historical/frozen captures unchanged. No numerical acceptance result yet.

````

## Source: .agents/runs/numerics-and-context-state-codec-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 6889; SHA256 `273e674eac1954805f3f129d9e230083475053aa13f6ee88664c5bfd27e8930f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Actual native/Rust post-state codec export

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: `/root/spine_review`. Root alone builds/runs exporters and comparisons.

PREPARED ONLY. No Cargo, native/model/operator/runtime/GPU execution by author.
No production files, frozen helpers or existing capture output changed.

- `native_state.py`: pinned public `llama_state_get_data` raw-byte decoder.
  No external library is loaded. Exact original0.8B/2B GGUF+config profile only;
  private sequence0, contiguous text positions0..cut-1, cut1..8192.
- `export_rust.rs`: existing `LoadedQwenModel::open` source binder + public
  `load_checkpoint`/`QwenCheckpoint::domain_values`. No session/root/forward or
  model arithmetic is executed. Existing decoder validates all finite values,
  typed domains, source/input digests, state bytes/fingerprint and exact EOF.
- `compare_state.py`: all42 exact complete active numerical domains; no
  tolerance, including signed zero. Exit0=all42 exact; exit2=bit differences;
  exit3=malformed/unqualified. First mismatch coordinates/bits + maxabs retained.
- `test_codec.py`: source-free synthetic metadata/axis/nonfinite/short/trailing
  fixtures; unexecuted by author. Distinct token/head/dimension markers test V
  transpose instead of relying on zeros.

## Source-derived wire closure

Pinned llama.cpp0.6.0/build11429/d812350 source was independently acquired by
root; `source-profile.json` binds its receipt, files and installed libraries.
Raw API is architecture-length+architecture, without session-file magic/version.
The outer profile is mandatory: raw format lacks model SHA, layer IDs and head
counts, so it is insufficient as a self-describing/authenticated checkpoint.

1. `llama-context.cpp:3400`: u32-length `qwen35` string; hybrid memory follows.
   `llama-io.cpp:5` fixes length encoding; native capture enforces little endian
   and four-byte float. Types `llama_pos`, `llama_seq_id`, `llama_token` are i32.
2. `llama-memory-hybrid.cpp:190`: attention first, recurrent second.
3. `llama-kv-cache.cpp:2071`: u32 stream count1, u32 occupied cell count=cut.
   Each cell (`:2217`) stores pos:i32,n_seq:u32, IMRoPE extension
   x:i32,y:i32,tok:i32, then sequence:i32. Refuse anything except actual private
   seq0, pos=x=y=index and exact teacher token. `llama-batch.cpp:125` expands
   text position[p]→[p,p,p,0]; cache `:1149` stores actual extension/token.
4. `llama-kv-cache.cpp:2250`: v_trans:u32=1,n_layer:u32=6,n_rot_k/v:u32=0.
   In ascending filtered layer order3/7/11/15/19/23: each K has type:i32=F32(0),
   row bytes:u64=2048, then occupied rows in physical cell order. This fixture
   requires that order to equal contiguous logical positions; no sorting or
   inferred holes. Each V has type:i32=0, element bytes:u32=4,width:u32=512,
   then actual channel-major occupied cells. Copy V bits to token-major.
   `:1355` K merges contiguous[dimension,head]; `:1514` V index is
   channel*capacity+cell. No dequantization, rotations or RoPE recomputation.
5. Layers/geometry come from original GGUF/config digests and model sources:
   `models/qwen35.cpp:4`, `llama-model.cpp:2705/:2772`, KV allocation ascending
   layer loop `llama-kv-cache.cpp:165`. Both originals have2KV heads×256.
6. `llama-memory-recurrent.cpp:784/:896/:915`: one recurrent cell at cut-1,
   seq0; s_trans=0,n_layer=24. Ascending18 filtered layers: F32 R rows73728B,
   then F32 S rows1048576B. No PLE rows in this pinned Qwen profile. Exact EOF.
   `llama-hparams.cpp:208/:236` fixes counts. Conv row channels×oldest3 history
   from `models/delta-net-base.cpp:449`; GDN S is[head,value,key], key contiguous
   (`ggml-cpu/ops.cpp:11227`). Copy both unchanged, not a recurrence replay.

Canonical KV domain is all K[token,head,dimension] followed by all V in same
order. Canonical convolution and recurrent layout match existing Rust domain
descriptors. All42 domains come from actual post-state bytes, including the six
KV domains omitted by the earlier730-record callback capture.

## Boundaries

- Both portable encodings contain only occupied causal KV, not unused physical
  future capacity. No future padding equality or GPU completion/resource claim.
- Native opaque file itself authenticates no source or historical execution.
  Export checks current source bytes, teacher bytes, outer native completion
  record and pins. Parent retains actual binary/library/argv/instrumentation
  receipts; no claim that a caller-supplied hash independently proves provenance.
- Existing730-record captures and their whole-native-state instrumentation
  equality remain unchanged. This adds a separate direct raw post-state check;
  logits and prior/operator evidence retain their existing independent reports.
- File/numerical output cap512MiB per exporter; at8192 numeric payload is
  221528064B/model/provider. Raw native reader retains at most six16MiB K slabs
  plus bounded working buffers; no full-model/source mmap or second engine.
  Rust public decoder and `domain_values` can hold two complete state payloads
  plus its existing source owner. These are logical bounds, not physical host
  ResourceArbiter/RAM admission. Root applies actual process/time limits.
- Fresh512/1024/etc inputs are distinct from lost historical8K inputs. No long
  GDN/source-profile cause closure or whole-model eligibility from export alone.

## Root-only commands

Use independently verified actual teacher-stream digest, not a per-cut digest
when captures/checkpoints were bound to the common8192-ID file. Require fresh
exclusive output directories. No exporter starts a provider/model evaluation.

```sh
codec=.agents/runs/numerics-and-context-state-codec-20261020
raw=.agents/runs/numerics-and-context-long-context-audit-20261020
ids=.agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le
ids_sha=44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d
# ROOT build coordinator only, offline; alternatively use existing rustc artifacts.
CARGO_BUILD_JOBS=2 cargo build --offline --release --manifest-path "$codec/Cargo.toml"
# ROOT only: source-free fixtures first.
python3 "$codec/test_codec.py"
python3 "$codec/native_state.py" --run \
  --capture "$raw/native-0.8b-512-ar1" --model target/models/qwen35-0.8b/model.gguf \
  --config target/models/qwen35-0.8b/config.json --raw-ids "$ids" --raw-sha256 "$ids_sha" \
  --cut 512 --output "$codec/root-native-0.8b-512"
"$codec/target/release/export_rust" target/models/qwen35-0.8b/model.gguf \
  target/models/qwen35-0.8b/config.json "$ids" "$ids_sha" 512 \
  "$raw/rust-0.8b-512/rust.post-checkpoint.bin" "$codec/root-rust-0.8b-512"
python3 "$codec/compare_state.py" --native "$codec/root-native-0.8b-512" \
  --rust "$codec/root-rust-0.8b-512" --output "$codec/root-compare-0.8b-512.json"
```

Inspect actual source-bound result before any2B/longer case. No automatic ladder.

````

## Source: .agents/runs/numerics-and-context-state-codec-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 918; SHA256 `21f066d5f92e8b6fcdef11b99c95456dc89e21b38612881e7f16ab38db570838`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: `.agents/plans/numerics-and-context/TODO.md`.
Owner: `/root/spine_review`; root owns all execution/shared-plan reconciliation.

- [x] Read pinned public native serializer and current public Rust decoder.
- [x] Source-derive actual text cell metadata, K/V axes, filtered layer order,
  complete R/S row geometry and occupied-causal-only scope.
- [x] Prepare exclusive bounded native exporter, existing public Rust decoder
  exporter, strict42-domain comparator and source-free negative/axis fixtures.
- [ ] Root/independent source review, compilation and source-free fixtures.
- [ ] Root actual existing512 native/Rust export and exact42-domain comparison.
- [ ] Root2B/longer source-bound actual proofs and GDN source-profile cause closure.

No Cargo/provider/model/math/runtime/GPU execution, no production edits.
Historical/frozen captures unchanged. No numerical acceptance result yet.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/README.md

Original UTF-8 bytes: 7463; SHA256 `39275e251d3094568573bffe0d84cff21a074b8e559283fb9f6dd0cd07376c1c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Public native CPU use_ref bulk control

Parent: `.agents/plans/numerics-and-context/TODO.md`. Owner: `/root/batch_design_review`.
PREPARED SOURCE ONLY; ROOT alone compiles, tests and runs. No production change.

Fresh8192 teachers, approved original GGUF0.8B/2B, CPU8, F32 KV, FA off,
prefix=batch=ubatch64 or512. One explicit case per invocation. Existing native
AR1/Rust evidence is preserved; this control is not yet executed or accepted.

## Public binding

- `native_raw_original.cpp` is byte-identical to the accepted16560B client
  (`387e0fdf...`). Its token loop, model/context construction, selected-token
  observer,18 actual GDN callback guards,730 records, full248320 logits and
  whole serialized plain/observed-state checks are unchanged.
- New executable exports four public symbols. Original public device init
  creates each real CPU backend, then the documented CPU registry extension
  `ggml_backend_cpu_set_use_ref(backend,true)` configures it. Context's existing
  documented `ggml_backend_set_abort_callback` callback is delegated unchanged,
  then sets the same reference policy on each real backend again.
- Pinned `llama-context.cpp:335–375` constructs CPU via public dev-init and
  calls that callback; `:1206–1219` visits every context backend. No private
  `llama_context`, `ggml_backend` or CPU context fields are read or written.
- Original public `ggml_graph_compute(graph,cplan)` receives the original
  public `ggml_cplan`; every observed plan must already have `use_ref=true`
  before delegation and retain it after. Both backend callbacks, four setter
  calls, all plans and two final frees are actual receipts, not predictions.
  The minimum plan-count bound includes each real prefix decode plus both
  selected-token decodes; extra scheduler split computations remain counted.
- CPU backend has **no public use_ref getter**. Plan-to-backend ownership is
  also unavailable publicly. Receipt says so. Actual plan flag readback plus
  both genuine setter/callback routes is the proposed policy proof; unseen
  routes, non-CPU backends, unexpected lifetimes or false flags refuse.
- Original implementations resolve only through exact already-mapped absolute
  `NOLOAD|NOW|LOCAL` providers; exact public symbol/path/inode/length checked.
  Source/runtime/provider files are independently hashed by ROOT before/after.
  Initial small ELF audit found preemptible public exports/PLT routes; that is
  feasibility evidence only. Actual interposition/count checks must pass.

## Numerical boundaries

`use_ref` is the sole intentional native execution-policy change. It bypasses
some generic CPU fusions/tinyBLAS/Hadamard/tiled paths according to pinned source.
The original extra-buffer dispatch precedes ordinary op selection; CPU_REPACK
codec/weight packing/dispatch remains untouched. No opaque packed data is
decoded, disabled, replaced or supplied to a new operator. Fused GDN remains
the original graph operation. This may still differ from AR1; no bit result is
assumed and no tolerance changes are allowed.

Plain/observed equality covers the original selected tail token and complete
post-state after the same plain8191-token bulk prefix. It is not a second
independent observed8192-token bulk run or an every-frontier observation.
Independent accepted comparisons are still required for all730 final-token
stages/full vocabulary and all42 complete occupied causal state domains.
Gain/scale records keep their existing separate-probe labels. Neither fresh8K
comparison recovers the lost historical teacher stream or closes every cause.

## Ownership and bounds

- Fresh exclusive output, original native deadline≤7200s, outer deadline+60s,
  address space12GiB, CPU time≤8×wall+64s, file cap512MiB, log cap8MiB.
- Explicit eight available CPU IDs; affinity installed before exec/DSO load,
  checked before/after native execution and every actual public compute call.
  Actual context thread settings remain8; individual plans may need1..8 tasks.
- Exact reviewed `supervisor.py` retained unchanged. Public pidfd/WNOWAIT,
  process-local subreaper, authenticated fresh SID, independent TERM/grace/
  two KILL stages before any reap, first/final reap, group absence and pipe EOF.
  Scope is owned SID only, not containment of a descendant creating a new SID.
- All post-Popen receipt/input checks run inside cleanup `finally`. Failure,
  partial incremental trace and original outputs remain; no retry/overwrite.
- This is a bounded diagnostic host process, not physical ResourceArbiter
  admission or any GPU/profile/serving/performance qualification.

## ROOT execution, after quiet measurements

Inspect `build-argv.json`; compile its exact argv under ROOT's bounded build
coordinator. It links the original providers, adding only `dl`, `pthread` and
four executable public exports. Retain compiler/argv/source/binary identity;
the caller-supplied binary SHA alone cannot prove compilation provenance.
No author build, linker, native/operator/model or lifecycle test was run.

```sh
run=.agents/runs/numerics-and-context-use-ref-bulk-control-20261020
# ROOT only: prepared source-free parser fixtures first.
python3 "$run/test_control.py"
python3 "$run/test_supervisor.py"
# ROOT authenticates the fresh compiled binary and chooses eight available CPUs.
python3 "$run/run_ref_control.py" --model 0.8b --prefix 64 \
  --cpus 0,1,2,3,4,5,6,7 --seconds 7200 \
  --binary "$run/native_ref_control" --binary-sha256 ROOT_FRESH_BINARY_SHA \
  --new-output "$run/root-0.8b-8192-ref64-1"
python3 "$run/validate_control.py" --capture-run "$run/root-0.8b-8192-ref64-1" \
  --new-report "$run/root-0.8b-8192-ref64-1/control-validation.json"
```

ROOT reviews actual policy/ownership/provider evidence before any next case.
Use fresh names for2B or512; no automatic ladder. An unobserved route is a
refusal to investigate, never authorization to fake a getter or force the flag
inside compute. Numeric differences are recorded, not suppressed.

Existing exact independent comparisons, ROOT only:

```sh
raw=.agents/runs/numerics-and-context-long-context-audit-20261020
codec=.agents/runs/numerics-and-context-state-codec-20261020
ids=.agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le
ids_sha=44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d
case="$run/root-0.8b-8192-ref64-1"
model=target/models/qwen35-0.8b/model.gguf
config=target/models/qwen35-0.8b/config.json
model_sha=57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf
# Supply the already accepted original Rust8192 capture directory; do not rebuild/relabel it.
python3 "$raw/compare_raw.py" --rust ROOT_ACCEPTED_RUST_0_8B_8192_CAPTURE \
  --native "$case/capture" --model "$model" --model-sha256 "$model_sha" \
  --raw-ids "$ids" --raw-sha256 "$ids_sha" --cut 8192 --output "$case/raw-comparison.json"
python3 "$codec/native_state.py" --run --capture "$case/capture" --model "$model" \
  --config "$config" --raw-ids "$ids" --raw-sha256 "$ids_sha" --cut 8192 \
  --output "$case/native-state"
python3 "$codec/compare_state.py" --native "$case/native-state" \
  --rust "$codec/root-rust-0.8b-8192" --output "$case/state-comparison.json"
```

Raw/state comparator exit2 means actual bit differences; malformed/refused
evidence remains distinct. Both reports must be inspected and bound to this
new capture before any control-result claim. No production arithmetic change
follows automatically from either outcome.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/SOURCE_AUDIT_V3.md

Original UTF-8 bytes: 6523; SHA256 `25507375032083b576af446c1e37c67b5cb13502c5ea1b8964432009bf7799c4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive V3 backend identity diagnostic — ROOT only

Owner: baseline_review, root-authorized additive files only. All original14,
V2 source/manifest/frozen copies and both failed cases remain unchanged.
No build, test, model/operator, container/GPU runtime or large payload hash by
author. No source/profile/math/StateRoot/long-context acceptance.

## Actual refusal and source inference

Fresh `root-0.8b-8192-ref64-2` used actual V2 binary42b98af8… and failed125
after context geometry n_ctx8448/n_outputs64, before any accepted capture:
records.jsonl is0B, only raw teacher file exists, no public setter/plan trace.
Its log explicitly loads BLAS and Haswell CPU. Cleanup completed without errors.

Pinned d812350 source:

- llama-context.cpp334–366 initializes model devices, then every ACCEL backend
  (explicit example BLAS), then CPU. n_gpu_layers0/offload flags do not suppress
  the ACCEL construction loop.
- ggml-blas.cpp214/322/478 public metadata names BLAS;356 returns device type
  ACCEL;281–307 constructs its distinct BLAS backend GUID.
- ggml-cpu.cpp249 recognizes CPU GUID specifically. Thus BLAS legitimately
  returns false. Current control applies set_reference to EVERY dev_init return.
- CPU_REPACK is an extra CPU buffer type (repack.cpp5162/5238), not a separately
  initialized backend. It cannot itself identify this failed backend predicate.

BLAS is a source-supported hypothesis, not actual backend identity. No v2
diagnostic printed the returned backend/device/registry; do not close the cause.

## V3 observation only

`native_ref_control_v3.cpp` adds one fatal metadata helper, only on the original
failed backend&&is_cpu check. Public BASE symbol lookup retains exact V2
NOLOAD/canonical regular-file dev/inode/length/export attestation. No new DSO,
RTLD_GLOBAL, fallback, private structure read or backend/provider replacement.

The <=4095B named diagnostic prints origin, actual is_cpu boolean, opaque
backend/device/registry identities, public names/device type, original input
device identity, whether it is known and equals the returned backend device.
Public getters resolve from the original BASE; existing predicate remains the
same exact Haswell public is_cpu. Known actual backend is read through public
APIs only. Null metadata is marked rather than dereferenced.

Then SAME exit125, BEFORE setter invocation for that attempt. It never skips or
disables a backend, never sets a CPU flag on a nonCPU object, and never proceeds
to computation after refusal. Successful CPU predicate path retains the same
setter/callback forwarding/plan checks/token loop; only the source revision
marker changes for build/run binding. No new accelerator policy is implemented.

## ROOT actual compile and invocation binding

Build V3 with build-argv-v3.json argv from MAIN repository CWD using the exact
pinned compiler. Preserve old binaries/build directories. Fresh build receipts
requested.json and result.json must both include actual `cwd`, `compiler`,
`compiler_sha256`, `argv`, reviewed V3 `source_freeze_sha256`, exact `source_inputs`
mapping from manifest compile_inputs, and runtime_or_math_qualified=false.
Record source_inputs before/after compilation. Result also records real exit,
binary_path/bytes/SHA. This closes the relative-argv CWD gap explicitly.

Use unchanged run_ref_control.py with new binary path/digest and a fresh output
name. The generic invocation correctly pins V1 launcher/helpers; it does not
claim that V1 CPP was compiled into this V3 binary. After actual invocation:

```sh
python3 .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/bind_v3.py \
  --manifest-sha256 REVIEWED_V3_SHA \
  --build ABS_FRESH_BUILD --run ABS_FRESH_CASE --new-binding ABS_FRESH_BINDING_JSON
```

Recipe authenticates all prior/new frozen sources, failed metadata, compiler,
exact new CPP/includes/CWD/argv/build exit/binary and actual runner inputs. A
failed diagnostic may bind source/build/invocation while remaining UNQUALIFIED.
Actual diagnostic interpretation is ROOT review; it is not a numeric verdict.

## Future policy feasibility only — NOT implemented

Pinned public scheduler graph routes:

- ggml-backend.cpp595: sync graph_compute delegates graph_compute_async.
- 601–604: graph_compute_async dispatches the selected backend iface directly.
- 1955/1977: both ordinary split and observed graph-view scheduler paths call
  that public async entry. 588–592: public graph_plan_compute is another direct
  iface route; checking only inner CPU ggml_graph_compute does not cover it.
- CPU cpp139–153 stores source cplan.use_ref at plan creation;162–190 both CPU
  plan/ordinary callbacks converge on ggml_graph_compute, whose current wrapper
  checks actual cplan.use_ref before/after. Old plans must not be silently fixed.
- BLAS cpp226–263 directly invokes CBLAS and bypasses inner CPU graph_compute.

If actual identity later proves a genuine ACCEL is initialized but unused,
unmodified construction could be recorded separately while a NEW reviewed
control rejects every nonCPU public async/plan compute BEFORE delegation.
Unknown/foreign backends remain cold refusals. Known CPU public calls could
carry a thread-local scope through original forwarding, bind actual inner CPU
plan observations to that backend, and require every original delegated plan
use_ref=true. Nested sync→async must count one compute rather than two math
invocations; capture every entry/return/success and all initialization/retirement.
Opaque plan handles need public create/free association if plan route is used;
no private plan/backend layout reinterpretation. Existing two-CPU counts cannot
be relabeled as all-backend coverage; constructed ACCELs require explicit records.

This is source feasibility, not compiled route closure. ROOT must prove actual
public interposition paths/DSO closure for both plain and observed modes before
allowing construction to continue. Direct/internal bypass or any unmatched
compute must refuse. Numerical all730/fullV/all42 and observation guards remain.

Crucially, op_offload=false does NOT prove ACCEL unused: backend.cpp1028 chooses
the highest priority compatible buffer/op; pass3 around1380 upgrades compatible
buffer nodes independently of op_offload. BLAS supports host-buffer contiguous
F32 GEMM with all relevant dimensions>=32 (blas.cpp397–459). Bulk64 QK/AV may
qualify. A future guard may correctly refuse; do not disable BLAS, alter buffers,
or silently downgrade math merely to make the control pass.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/SOURCE_AUDIT_V4.md

Original UTF-8 bytes: 7843; SHA256 `df90f645366d93660a06c8decbb11e30e42b18b4aa5ca10dc7c9e35fbc87e9c0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# V4: exact BLAS lifecycle; CPU-only `use_ref` computation

- Owner: baseline_review. Parent: numerics-and-context.
- Additive source preparation only; no author build/test/native/model/runtime.
- V1/V2/V3, original token loop, generic runner, failures and source pins unchanged.
- V3 actually identified BLAS (backend/device/registry, ACCEL=3, `is_cpu=false`, original input device equality). This proves the refusing backend identity; it does not prove BLAS unused.

## Closed policy

- Same public `ggml_backend_dev_init` forwarded once. Classify returned backend using exact BASE public metadata, original input-device equality, CPU `is_cpu` or BLAS `is_blas`, exact public provider registry singleton and registry device0. Require original names/type. Constructor trace records actual checked identities, names, type and predicate.
- Same canonical-file/regular device+inode/length/exact-export V2 attestation; only `RTLD_LOCAL|RTLD_NOLOAD|RTLD_NOW`. BLAS adds actual ROOT pin: regular non-symlink64136B, dev38/inode2939302343, SHA `ab748c85dadb66ff128c1eda20a506bd6de94646cd06a7fbfd5b7a3881225e64`. ROOT hashes before/after; C++ checks immutable same-file identity and pinned inode/size before obtaining original public symbols. No new provider load or global promotion.
- Exactly two original CPU and two BLAS contexts; original loop keeps both contexts alive simultaneously. CPU init and original abort callback each apply the same V1 public `use_ref=true` setter. BLAS receives neither setter nor changed arguments/policy. Record constructor and completed retirement separately. Both CPU owners must actually compute; BLAS owners must only construct/retire.
- Public async, sync and opaque-plan compute wrappers reject every BLAS/unknown/retired target BEFORE original delegation. A real BLAS assignment causes exit125 with `nonCPU_compute_refused`, never a successful capture. Plan creation/free also require CPU: only BLAS construction/retirement is admitted.
- CPU async/opaque-plan call creates one thread-local public owner scope. Actual public CPU `ggml_graph_compute` must occur once inside that scope, with `cplan.use_ref=true` before/after, unchanged1..8 threads and successful status. Async binds exact graph pointer. Opaque plans use actual public create/free owner+generation associations; no private plan/backend layout is read. CPU implementation's graph copy is deliberately not confused with original public graph pointer.
- Sync must delegate once through same-backend/same-graph guarded async; its count is separate from math. Actual public/inner counts conserve globally and per CPU, with nonvacuous coverage. Missing/nested/duplicate routes refuse. Opaque-plan route is explicitly UNOBSERVED if count0; successful async evidence does not qualify it.
- Callback arguments, graph pointers, buffers, offload/repacking policies and original compute remain unchanged. No disableBLAS, wrong-backend setter, CPU forcing, math fallback, signed-zero exception, tolerance change or alternate token loop.

## Source and actual selected compiled routes

Pinned d812350 source in original acquisition receipt:

- `ggml-backend.cpp`: scheduler ordinary split and callback graph-view split call public `ggml_backend_graph_compute_async`; public sync calls async; public plan compute dispatches opaque backend callback. Device/registry getters are public metadata.
- `ggml-cpu.cpp`: ordinary callback creates public cplan with context `use_ref`, then calls public inner compute; opaque create stores same policy and opaque compute calls same public inner compute. Opaque internals are source-audited only, not accessed by V4.
- `ggml-blas.cpp`: BLAS ordinary compute invokes CBLAS directly, bypassing inner CPU compute. Thus old inner-only control was insufficient once BLAS exists. Device type ACCEL, names BLAS, actual public `is_blas`/registry functions available.
- `llama-context.cpp`: constructs ACCEL providers in addition to CPU. `op_offload=false` does not forbid all scheduler assignment: backend/buffer compatibility and scheduler upgrade rules still apply. V4 preserves them and refuses if they actually compute.

ROOT-produced `root-v4-elf-audit/` excerpts are pinned in `static-route-pins-v4.json`:

- BASE `ggml_backend_sched_graph_compute_async`: actual selected calls at `3cb18` (callback graph view) and `3d12b` (ordinary split) -> public async PLT `18840`.
- BASE sync: `375fc` -> same public async PLT. BASE async/opaque-plan entries dispatch original interface callback.
- CPU ordinary `1f34a` CALL and opaque-plan `1f384` TAILJMP -> public inner PLT `15a40`.
- Matching `R_X86_64_JUMP_SLOT`: BASE `0xdbb18` async; CPU `0x14de28` inner. Both selected providers lack dynamic SYMBOLIC.
- Pinned loader source `ggml-backend-dl.cpp` uses `RTLD_NOW|RTLD_LOCAL`, without DEEPBIND. This source observation does not substitute for actual runtime routing.
- These are selected-site static evidence for these exact compiled providers. They are **not** proof of every call in every DSO, complete dynamic binding, or runtime acceptance. ROOT must review them, compile actual exported wrappers and observe nonvacuous public-to-inner conservation. No private function offsets are called; no GOT patch, callback replacement, dispatch forcing or LD-preload arithmetic.

## ROOT build / run / validation

1. Independently review exact V4 manifest plus original freezes and selected compiled-route evidence. Do not execute old or newly broadened policy by inference.
2. Fresh finite strict build from MAIN CWD using `build-argv-v4.json` with actual pinned g++ path from manifest. Record requested/result JSON as prior V3: `cwd`, `compiler`, `compiler_sha256`, exactargv, THIS `source_freeze_sha256`, full seven-entry `source_inputs` before/after, `runtime_or_math_qualified=false`; result adds exit_code and actual output absolute path/bytes/SHA. Preserve all old builds/binaries.
3. Same unchanged `run_ref_control.py`, fresh output, actual new binary+SHA, `.8B` then2B, fresh8192 IDs, prefix/ubatch64 or512 and explicitCPU8/caps. This generic helper still binds its original V1 freeze; it is never relabeled as V4 source proof.
4. `bind_v4.py --manifest-sha256 REVIEWED_V4_SHA --build ABS_NEW_BUILD --run ABS_NEW_CASE --new-binding ABS_NEW_CASE/source-build-binding-v4.json`. Records actual new source, seven inputs/build/output, invocation/native log, selected static scope and extra provider. Can bind a failure; does not convert it into PASS. Unknown or actual BLAS compute remains exit125/UNQUALIFIED, even if its construction was legitimate.
5. ROOT-only source-free `python3 -m unittest test_control_v4.py` from this run; eleven prepared cases. Author AST only. Then `validate_control_v4.py --capture-run ABS_NEW_CASE --new-report ABS_NEW_REPORT` for a successful capture. New validator duplicates old capture/lifecycle/source guards, adds strict V4 trace reconciliation and required separate source/build binding, hashes actual original/extra providers before/after and verifies final exact BLAS mappings.
6. A control PASS establishes policy/route evidence and original plain-vs-observed full-logit/whole-serialized-state transparency only. Separate authentic raw730, full vocabulary and all42 domain comparisons against Rust remain mandatory. Historical prefix unrecovered; no historical/whole-cause/performance conclusion.

## Bounds

- Same native12GiB address/CPU8/finite deadline, teacher8192, batch64/512, original state/capture caps, owned supervisor cleanup.
- New maps: <=4 backends, <=64 simultaneous opaque plans, <=65536 total public plan generations; trace <=65536 records, <2048B per record, <=8MiB total. Record buffers/metadata are bounded; no learned matrices/copies/new native allocator.
- No production engine/profile/serving change, no historical8K divergence closure, no inference support claim.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/SOURCE_AUDIT_V5.md

Original UTF-8 bytes: 6216; SHA256 `b9811856fc6b474dbb1fb8dd69ef8f1d86ebbb8b35830908d9626ed46e363790`.
Resolve embedded relative paths against the original path, not this volume.

````text
# V5: observe the SAME V4 free refusal

- Owner baseline_review; source only. No author build/test/native/model execution.
- All V1–V4 sources, freezes, binaries and actual failed attempts remain unchanged.
- Sole diagnostic change: at V4's failing `ggml_backend_free` membership predicate, print fixed API origin, actual raw pointer, `is_null`, map-only tracked/retired status and already-observed owner/compute counts. No unknown/null/retired backend dereference, metadata getter, new DSO load, delegation, policy continuation or constructor allowance. Same `_exit(125)`.

## Actual V4 evidence

- ROOT strict build PASS3.332s, actual160960B SHA `41ccc52ad37f26077d2370e6101fbfe705ee45bd778210c3334ba1d341a3a357`; eleven pure fixtures PASS; source/build/invocation/provider binding PASS.
- Actual case `root-0.8b-8192-ref64-4`: native125 after model buffer loading, no public constructor trace/control report/accepted compute. Native log50345B, SHA `40f44109073c82d8e6063fa380666da75eb179cd877455c6015378af66e8beeb`; only BLAS and Haswell CPU load messages. Original raw capture records still empty.
- Actual owned cleanup receipt reports leader reaped, groups absent, nonzero signals before reap, EOF/selector/pipe closure and no cleanup errors. Failed case remains UNQUALIFIED.
- The actual failing pointer/API was not printed by V4. It is not yet established as NULL or a direct untracked CPU backend.

## Construction/free closure audit, pinned d812350

- `ggml/src/ggml-backend.cpp:380–386`: public `ggml_backend_free(NULL)` is explicitly a legal no-op; nonnull calls original interface free.
- V3's wrapper guarded `armed && backend`. V4's wrapper changed this to `armed`, then called `known(backend)`, which rejects NULL. This introduced an incompatible null-free guard in the control itself.
- `src/llama-model-loader.cpp:1493–1597`: each `load_all_data` initializes local `upload_backend` from a lambda. It returns nullptr for mmap/check_tensors; otherwise requires actual buffer/default device, async+host_buffer+events capability and pinned staging/events. Only a successful async-upload branch constructs a real backend, via PUBLIC `ggml_backend_dev_init` at1589.
- Same function1760 UNCONDITIONALLY calls public `ggml_backend_free(upload_backend)` after loading that tensor context. This includes null upload_backend. `src/llama-model.cpp:1950–1954` calls `load_all_data` once for each original tensor context/buffer map. Thus model-loading cleanup is separate from two inference contexts' CPU/BLAS owners and must never be counted as a live backend if NULL.
- Selected CPU provider `ggml-cpu.cpp:394–400`: actual CPU device props declare async=false, host_buffer=false, events=false. Even non-mmap CPU uploads cannot take that successful async constructor branch. Actual V4 model log reports CPU_Mapped and CPU_REPACK buffers, then refusal, consistent with null upload cleanup. This is a source-supported causal lead; actual V5 pointer observation is still required.
- CPU_REPACK is an extra buffer type and repacked tensor storage/operations. Source search through `ggml-cpu/repack.cpp` found no `ggml_backend_cpu_init`, `ggml_backend_dev_init` or `ggml_backend_free` call. Repacking is not evidence of an independently constructed backend. No repacking/buffer/math policy changes are proposed.
- CPU `ggml_backend_cpu_init` is a genuine direct public constructor at217; selected device init callback405 invokes it. Normal context route PUBLIC `ggml_backend_dev_init` therefore records it after return. No direct CPU init call is present in `llama-model-loader.cpp` or `llama-model.cpp`. Direct CPU compare/debug code exists elsewhere, but this helper does not invoke that API; it must not be silently admitted if actually encountered.
- BLAS direct public init286 is called by selected BLAS device callback377. Context construction `llama-context.cpp:335–362` adds actual model devices, every ACCEL via public dev_init, then CPU via public init_by_type. Registry convenience functions `ggml-backend-reg.cpp:366–389` all route to public dev_init.
- Original helper holds plain and observed contexts simultaneously, later explicitly destroys observed then plain. Each context owns its actual backend list via `ggml_backend_ptr`; header deleter is public free. CPU/BLAS actual interface free functions delete their own objects and do not recursively invoke public backend free. V4's exact two CPU/two BLAS count concerns these actual nonnull contexts only; no model-load null cleanup belongs to that count.
- Other source backend families (META/RPC/debug CPU compare) have constructors, but are not an excuse to broaden this control. Current selected log loads only pinned BLAS/CPU; V4/V5 refuse every unknown nonnull owner. Source inspection does not replace an actual dynamic constructor/caller observation.

## Prepared ROOT primitive

1. Fresh strict build `build-argv-v5.json`, same seven headers/included source and public exports, with manifest's actual g++ path. Retain requested/result metadata as V4, using THIS reviewed V5 SHA and actual new `native_ref_control_v5` output; preserve older builds.
2. Same unchanged generic runner, fresh case, same model/IDs/caps. Expected diagnostic still125; do not reinterpret it as numerical progress.
3. `bind_v5.py --manifest-sha256 REVIEWED_V5_SHA --build ABS_FRESH_BUILD --run ABS_FRESH_CASE --new-binding ABS_FRESH_CASE/source-build-binding-v5.json`. Root-only existing provider/route/source binding reused; report only source/build/invocation.
4. Read the actual `free_refusal_diagnostic_v5`. If NULL with zero tracked owners/compute, it proves this refusal is the wrapper's legal-null-free incompatibility, distinct from model backend construction. Only then consider the separate narrow restoration of original null-free forwarding with no owner/setter/compute count. If nonnull, retain failure and audit actual public origin; never dereference unknown/retired state or add a blanket constructor allowance.

No V5 success/control validator or numerical eligibility is claimed. Source/static normalization confirms removal of only diagnostic + revision marker yields exact frozen V4 CPP. No new refusal framework or policy changes.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/SOURCE_AUDIT_V6.md

Original UTF-8 bytes: 3211; SHA256 `187d1c28a86b2694f919c0f357bb10a06fcdd0a6a107c2d1fe89ea138b159502`.
Resolve embedded relative paths against the original path, not this volume.

````text
# V6: restore original legal NULL-free forwarding only

- Source-only additive; owner baseline_review. V1–V5 and all failed attempts immutable.
- ROOT V5 actually observed `api=ggml_backend_free backend=(nil) is_null=true tracked=false retired=false tracked_backend_count=0 actual_public_compute_calls=0 actual_inner_calls=0 original_free_delegated=false policy_continued_after_refusal=false`. Thus the model-loading refusal was V4's null-free guard regression, not an observed untracked CPU constructor.
- Pinned BASE public `ggml_backend_free(NULL)` immediately returns. Loader unconditionally frees its possibly-null `upload_backend`; this is separate from real context backend lifetimes.
- V6 CPP is frozen V4 plus TWO `&& backend` conditions in the public free wrapper (before and after original free), plus the revision string. Original NULL free is forwarded exactly once, outside tracking/counters. No no-op backend identity is created; no null setter/compute/retirement trace or count is invented. Every NONNULL V4 classification, lifetime/count, BLAS refusal, public/TLS/inner use_ref, plan association and original compute remains byte-identical.
- Removing those two conditions and changing only revision back to V4 is statically asserted to produce exact V4 bytes. No V5 diagnostic is imported into this control.
- New successful report/validator revision: `public_use_ref_exact_blas_cpu_compute_legal_null_free_v6`. The exact V4 trace parser and eleven pure fixtures are reused unchanged. Original native token loop, source/profile, buffers/CPU_REPACK, offload flags, public BLAS lifecycle/never-compute policy, all arithmetic and generic runner are unchanged.

## ROOT follow-through

1. Independently review tiny `NULL_FREE_FORWARDING_V6.patch` and new source/binding/validator revision. Build from MAIN CWD with actual g++ and `build-argv-v6.json`; same seven input and fresh requested/result/binary receipts as V4/V5, using THIS reviewed V6 SHA. Preserve all old builds/binaries.
2. Same unchanged generic runner, actual fresh `native_ref_control_v6` SHA/fresh case and original source/teacher/CPU8/8192/prefix caps. No accelerator disable or inference-policy workaround if BLAS actually computes.
3. `bind_v6.py --manifest-sha256 REVIEWED_V6_SHA --build ABS_NEW_BUILD --run ABS_NEW_CASE --new-binding ABS_NEW_CASE/source-build-binding-v6.json`. Requires immutable V1–V5, actual build/seven inputs/output/invocation plus original runtime/static provider evidence. It can bind a failure; it cannot qualify arithmetic.
4. Reuse `python3 -m unittest test_control_v4.py` unchanged. A successful fresh case additionally requires `validate_control_v6.py --capture-run ABS_NEW_CASE --new-report ABS_NEW_REPORT`; correct new revision and actual V6 binding, SAME strict V4 trace equations and original full730 metadata/observer guards. Complete dynamic route/policy evidence still requires ROOT observation. Separate raw730/full-vocabulary/all42 comparisons remain mandatory for both models.

Author: two Python ASTs and small-source/metadata normalization only; no tests/build/native/model/GPU/runtime. No long-context/cause/full-model/performance qualification is implied by this control repair.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/SOURCE_AUDIT_V7.md

Original UTF-8 bytes: 2518; SHA256 `0226a058721323d17437908172cd8a751128ac63cfaf8d6f088068dd0b3a9029`.
Resolve embedded relative paths against the original path, not this volume.

````text
# V7: metadata at the SAME nonCPU refusal

- Source-only; no author build/test/native/operator/model/runtime execution.
- V1..V6 source/freezes, failed attempts and ROOT .8B V6 result unchanged.
- Actual ROOT2B V6 refused a pinned BLAS public compute before delegation.
  Actual operator identity is not yet established. In particular F32 alpha
  W[2048,16] has output16 and fails BLAS's pinned MUL_MAT minimum-output32
  condition; do not attribute this refusal to alpha from a split count alone.
- V7 passes only the already available public graph argument into that same
  guard. Before its unchanged exit125, emit actual graph count and up to128
  nodes, then every nonnull public src[0..9] header for those nodes. Node/input
  records contain opaque identity, bounded byte-escaped name, type/op IDs and
  public names, and four integer dimensions. No payload/data/buffer/plan/backend
  private field, callback, copy, source value, flag, scheduling or arithmetic.
- graph_n_nodes/graph_node/op_name/type_name resolve through the same exact
  pinned BASE NOLOAD/canonical/inode/length/export attestation. Public pinned
  ggml.h defines the tensor header and these APIs. ROOT actual BASE symbol
  listing confirms all four exports; this is static metadata, not proof of
  new runtime/provider route coverage.
- Summary states actual_node_count and retained_nodes; >128 or absent public
  graph is explicitly incomplete, never silently a complete graph. Opaque plan
  APIs expose no public graph argument: emit unavailable metadata, never peek.
- Existing2048B/record,8MiB trace and65536-record bounds still apply. At most128
  node +1280 input records; each header independently escaped/bounded. Exact
  same known/live backend, CPU setter/use_ref/TLS/plan and legal NULL-free
  policy; no BLAS allowance/delegation, unload, disable, op_offload or buffer
  change. Removing diagnostic helper/call/public graph argument and revision
  reproduces frozen V6 byte-for-byte.

ROOT: compile fresh build-argv-v7.json with pinned compiler under the SAME
bounded build receipt schema/source_inputs pre/post; preserve V6 failed paths.
Run fresh2B case via unchanged generic V1 owned runner, actual V7 binary SHA.
`bind_v7.py` binds THIS source/build/binary/invocation/trace to fresh receipts.
Actual refusal remains UNQUALIFIED; read bounded V7 graph metadata before any
new routing policy. `validate_control_v7.py` retains the unchanged V4 trace
parser/11 fixtures for a future successful control; no new numeric acceptance.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/TODO.md

Original UTF-8 bytes: 1927; SHA256 `290485060e47a7ae31141f6c9e85622b6ff54f5f939535040a77ceef285da526`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Public CPU use_ref fresh8K bulk control

Parent: `.agents/plans/numerics-and-context/TODO.md`. Owner: `/root/batch_design_review`.
Scope: this new ignored diagnostic run only; no MAIN source/shared-plan edits.

- [x] Source-audit public actual context backend creation/abort extension and
  reference setter; actual public cplan flag exists, no backend getter claimed.
- [x] Small ELF metadata confirms candidate exported/PLT routes; no runtime-route claim.
- [x] Prepare additive wrapper over byte-identical original16560B native client;
  only policy=true, original operator/token schedule/CPU_REPACK unchanged.
- [x] Prepare explicitCPU8/source-pinned finite one-case launcher using unchanged
  reviewed pidfd/WNOWAIT/subreaper owned-SID supervisor; original failures retained.
- [x] Prepare strict actual route/lifetime/count/affinity/capture validator and
  source-free negative fixtures. Python AST only; no fixtures executed by author.
- [x] Source/own-artifact freeze and ROOT-only build/capture/comparison instructions.
- [ ] Independent source review and ROOT strict C++ build/ELF export inspection.
- [ ] ROOT prepared parser/lifecycle fixtures; this launcher's cleanup unexecuted.
- [ ] ROOT genuine new8192 refbulk capture: actual setter/plan flags, provider pins,
  original730 records + plain/observed fullV/whole serialized state, allcleanup.
- [ ] ROOT independent unchanged raw730/full248320 and all42 causal-state comparisons
  against the accepted exact Rust AR1 capture/checkpoint, each source/scope explicit.
- [ ] Only after reviewing the first result choose another model/64or512 case.

Unqualified: all builds/runtimes/arithmetic results, full bulk cause closure,
historical teacher recovery, CPU_REPACK layout decoding, GPU/HF/serving support.
No existing source or accepted receipt was modified; original client/supervisor
copies were compared byte-for-byte. No model/DSO payload hashes by author.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/TODO_V4.md

Original UTF-8 bytes: 1483; SHA256 `c9685704b9319c65a88d76cf8bfe5ad63c339a7dc1c1979c8c89d16072c99fb5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# baseline_review V4 working checklist

- Parent: `.agents/plans/numerics-and-context/`; root sole execution/merge coordinator.
- Scope: additive ignored public use_ref diagnostic/control only; no production changes.
- [x] Read V3 actual BLAS identity and ROOT actual selected ELF excerpts.
- [x] Preserve original14, V2/V3 freezes, original helper/runner/validators/tests and failed cases.
- [x] Prepare exact pinned BLAS constructor/retirement policy; refuse every actual nonCPU compute before delegation.
- [x] Prepare CPU public/TLS/inner pre/post use_ref conservation, opaque public plan owner/lifecycle and nonvacuous per-CPU checks.
- [x] Prepare strict source-only parser fixtures and source/build/invocation/static-provider binding.
- [x] AST parse new Python; small-source/excerpt hashes only. NO author tests/build/model/native/operator/GPU/runtime.
- [ ] Independent ROOT/peer source review.
- [ ] ROOT strict fresh C++ build and explicit new binary/source/actual build binding.
- [ ] ROOT execute eleven pure parser fixtures.
- [ ] ROOT fresh bounded actual model case; if any BLAS compute, preserve exit125/UNQUALIFIED, no policy forcing.
- [ ] If actual capture succeeds, ROOT V4 validator, exact raw730/fullV/all42 comparison and explicit scope disposition for both models.
- Unresolved: complete dynamic route closure until actual observation; opaque-plan runtime eligibility if unused; historical prefix and full historical8K cause. No inference/performance claim.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 7463; SHA256 `39275e251d3094568573bffe0d84cff21a074b8e559283fb9f6dd0cd07376c1c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Public native CPU use_ref bulk control

Parent: `.agents/plans/numerics-and-context/TODO.md`. Owner: `/root/batch_design_review`.
PREPARED SOURCE ONLY; ROOT alone compiles, tests and runs. No production change.

Fresh8192 teachers, approved original GGUF0.8B/2B, CPU8, F32 KV, FA off,
prefix=batch=ubatch64 or512. One explicit case per invocation. Existing native
AR1/Rust evidence is preserved; this control is not yet executed or accepted.

## Public binding

- `native_raw_original.cpp` is byte-identical to the accepted16560B client
  (`387e0fdf...`). Its token loop, model/context construction, selected-token
  observer,18 actual GDN callback guards,730 records, full248320 logits and
  whole serialized plain/observed-state checks are unchanged.
- New executable exports four public symbols. Original public device init
  creates each real CPU backend, then the documented CPU registry extension
  `ggml_backend_cpu_set_use_ref(backend,true)` configures it. Context's existing
  documented `ggml_backend_set_abort_callback` callback is delegated unchanged,
  then sets the same reference policy on each real backend again.
- Pinned `llama-context.cpp:335–375` constructs CPU via public dev-init and
  calls that callback; `:1206–1219` visits every context backend. No private
  `llama_context`, `ggml_backend` or CPU context fields are read or written.
- Original public `ggml_graph_compute(graph,cplan)` receives the original
  public `ggml_cplan`; every observed plan must already have `use_ref=true`
  before delegation and retain it after. Both backend callbacks, four setter
  calls, all plans and two final frees are actual receipts, not predictions.
  The minimum plan-count bound includes each real prefix decode plus both
  selected-token decodes; extra scheduler split computations remain counted.
- CPU backend has **no public use_ref getter**. Plan-to-backend ownership is
  also unavailable publicly. Receipt says so. Actual plan flag readback plus
  both genuine setter/callback routes is the proposed policy proof; unseen
  routes, non-CPU backends, unexpected lifetimes or false flags refuse.
- Original implementations resolve only through exact already-mapped absolute
  `NOLOAD|NOW|LOCAL` providers; exact public symbol/path/inode/length checked.
  Source/runtime/provider files are independently hashed by ROOT before/after.
  Initial small ELF audit found preemptible public exports/PLT routes; that is
  feasibility evidence only. Actual interposition/count checks must pass.

## Numerical boundaries

`use_ref` is the sole intentional native execution-policy change. It bypasses
some generic CPU fusions/tinyBLAS/Hadamard/tiled paths according to pinned source.
The original extra-buffer dispatch precedes ordinary op selection; CPU_REPACK
codec/weight packing/dispatch remains untouched. No opaque packed data is
decoded, disabled, replaced or supplied to a new operator. Fused GDN remains
the original graph operation. This may still differ from AR1; no bit result is
assumed and no tolerance changes are allowed.

Plain/observed equality covers the original selected tail token and complete
post-state after the same plain8191-token bulk prefix. It is not a second
independent observed8192-token bulk run or an every-frontier observation.
Independent accepted comparisons are still required for all730 final-token
stages/full vocabulary and all42 complete occupied causal state domains.
Gain/scale records keep their existing separate-probe labels. Neither fresh8K
comparison recovers the lost historical teacher stream or closes every cause.

## Ownership and bounds

- Fresh exclusive output, original native deadline≤7200s, outer deadline+60s,
  address space12GiB, CPU time≤8×wall+64s, file cap512MiB, log cap8MiB.
- Explicit eight available CPU IDs; affinity installed before exec/DSO load,
  checked before/after native execution and every actual public compute call.
  Actual context thread settings remain8; individual plans may need1..8 tasks.
- Exact reviewed `supervisor.py` retained unchanged. Public pidfd/WNOWAIT,
  process-local subreaper, authenticated fresh SID, independent TERM/grace/
  two KILL stages before any reap, first/final reap, group absence and pipe EOF.
  Scope is owned SID only, not containment of a descendant creating a new SID.
- All post-Popen receipt/input checks run inside cleanup `finally`. Failure,
  partial incremental trace and original outputs remain; no retry/overwrite.
- This is a bounded diagnostic host process, not physical ResourceArbiter
  admission or any GPU/profile/serving/performance qualification.

## ROOT execution, after quiet measurements

Inspect `build-argv.json`; compile its exact argv under ROOT's bounded build
coordinator. It links the original providers, adding only `dl`, `pthread` and
four executable public exports. Retain compiler/argv/source/binary identity;
the caller-supplied binary SHA alone cannot prove compilation provenance.
No author build, linker, native/operator/model or lifecycle test was run.

```sh
run=.agents/runs/numerics-and-context-use-ref-bulk-control-20261020
# ROOT only: prepared source-free parser fixtures first.
python3 "$run/test_control.py"
python3 "$run/test_supervisor.py"
# ROOT authenticates the fresh compiled binary and chooses eight available CPUs.
python3 "$run/run_ref_control.py" --model 0.8b --prefix 64 \
  --cpus 0,1,2,3,4,5,6,7 --seconds 7200 \
  --binary "$run/native_ref_control" --binary-sha256 ROOT_FRESH_BINARY_SHA \
  --new-output "$run/root-0.8b-8192-ref64-1"
python3 "$run/validate_control.py" --capture-run "$run/root-0.8b-8192-ref64-1" \
  --new-report "$run/root-0.8b-8192-ref64-1/control-validation.json"
```

ROOT reviews actual policy/ownership/provider evidence before any next case.
Use fresh names for2B or512; no automatic ladder. An unobserved route is a
refusal to investigate, never authorization to fake a getter or force the flag
inside compute. Numeric differences are recorded, not suppressed.

Existing exact independent comparisons, ROOT only:

```sh
raw=.agents/runs/numerics-and-context-long-context-audit-20261020
codec=.agents/runs/numerics-and-context-state-codec-20261020
ids=.agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le
ids_sha=44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d
case="$run/root-0.8b-8192-ref64-1"
model=target/models/qwen35-0.8b/model.gguf
config=target/models/qwen35-0.8b/config.json
model_sha=57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf
# Supply the already accepted original Rust8192 capture directory; do not rebuild/relabel it.
python3 "$raw/compare_raw.py" --rust ROOT_ACCEPTED_RUST_0_8B_8192_CAPTURE \
  --native "$case/capture" --model "$model" --model-sha256 "$model_sha" \
  --raw-ids "$ids" --raw-sha256 "$ids_sha" --cut 8192 --output "$case/raw-comparison.json"
python3 "$codec/native_state.py" --run --capture "$case/capture" --model "$model" \
  --config "$config" --raw-ids "$ids" --raw-sha256 "$ids_sha" --cut 8192 \
  --output "$case/native-state"
python3 "$codec/compare_state.py" --native "$case/native-state" \
  --rust "$codec/root-rust-0.8b-8192" --output "$case/state-comparison.json"
```

Raw/state comparator exit2 means actual bit differences; malformed/refused
evidence remains distinct. Both reports must be inspected and bound to this
new capture before any control-result claim. No production arithmetic change
follows automatically from either outcome.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1927; SHA256 `290485060e47a7ae31141f6c9e85622b6ff54f5f939535040a77ceef285da526`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Public CPU use_ref fresh8K bulk control

Parent: `.agents/plans/numerics-and-context/TODO.md`. Owner: `/root/batch_design_review`.
Scope: this new ignored diagnostic run only; no MAIN source/shared-plan edits.

- [x] Source-audit public actual context backend creation/abort extension and
  reference setter; actual public cplan flag exists, no backend getter claimed.
- [x] Small ELF metadata confirms candidate exported/PLT routes; no runtime-route claim.
- [x] Prepare additive wrapper over byte-identical original16560B native client;
  only policy=true, original operator/token schedule/CPU_REPACK unchanged.
- [x] Prepare explicitCPU8/source-pinned finite one-case launcher using unchanged
  reviewed pidfd/WNOWAIT/subreaper owned-SID supervisor; original failures retained.
- [x] Prepare strict actual route/lifetime/count/affinity/capture validator and
  source-free negative fixtures. Python AST only; no fixtures executed by author.
- [x] Source/own-artifact freeze and ROOT-only build/capture/comparison instructions.
- [ ] Independent source review and ROOT strict C++ build/ELF export inspection.
- [ ] ROOT prepared parser/lifecycle fixtures; this launcher's cleanup unexecuted.
- [ ] ROOT genuine new8192 refbulk capture: actual setter/plan flags, provider pins,
  original730 records + plain/observed fullV/whole serialized state, allcleanup.
- [ ] ROOT independent unchanged raw730/full248320 and all42 causal-state comparisons
  against the accepted exact Rust AR1 capture/checkpoint, each source/scope explicit.
- [ ] Only after reviewing the first result choose another model/64or512 case.

Unqualified: all builds/runtimes/arithmetic results, full bulk cause closure,
historical teacher recovery, CPU_REPACK layout decoding, GPU/HF/serving support.
No existing source or accepted receipt was modified; original client/supervisor
copies were compared byte-for-byte. No model/DSO payload hashes by author.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/frozen-v3/SOURCE_AUDIT_V3.md

Original UTF-8 bytes: 6523; SHA256 `25507375032083b576af446c1e37c67b5cb13502c5ea1b8964432009bf7799c4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive V3 backend identity diagnostic — ROOT only

Owner: baseline_review, root-authorized additive files only. All original14,
V2 source/manifest/frozen copies and both failed cases remain unchanged.
No build, test, model/operator, container/GPU runtime or large payload hash by
author. No source/profile/math/StateRoot/long-context acceptance.

## Actual refusal and source inference

Fresh `root-0.8b-8192-ref64-2` used actual V2 binary42b98af8… and failed125
after context geometry n_ctx8448/n_outputs64, before any accepted capture:
records.jsonl is0B, only raw teacher file exists, no public setter/plan trace.
Its log explicitly loads BLAS and Haswell CPU. Cleanup completed without errors.

Pinned d812350 source:

- llama-context.cpp334–366 initializes model devices, then every ACCEL backend
  (explicit example BLAS), then CPU. n_gpu_layers0/offload flags do not suppress
  the ACCEL construction loop.
- ggml-blas.cpp214/322/478 public metadata names BLAS;356 returns device type
  ACCEL;281–307 constructs its distinct BLAS backend GUID.
- ggml-cpu.cpp249 recognizes CPU GUID specifically. Thus BLAS legitimately
  returns false. Current control applies set_reference to EVERY dev_init return.
- CPU_REPACK is an extra CPU buffer type (repack.cpp5162/5238), not a separately
  initialized backend. It cannot itself identify this failed backend predicate.

BLAS is a source-supported hypothesis, not actual backend identity. No v2
diagnostic printed the returned backend/device/registry; do not close the cause.

## V3 observation only

`native_ref_control_v3.cpp` adds one fatal metadata helper, only on the original
failed backend&&is_cpu check. Public BASE symbol lookup retains exact V2
NOLOAD/canonical regular-file dev/inode/length/export attestation. No new DSO,
RTLD_GLOBAL, fallback, private structure read or backend/provider replacement.

The <=4095B named diagnostic prints origin, actual is_cpu boolean, opaque
backend/device/registry identities, public names/device type, original input
device identity, whether it is known and equals the returned backend device.
Public getters resolve from the original BASE; existing predicate remains the
same exact Haswell public is_cpu. Known actual backend is read through public
APIs only. Null metadata is marked rather than dereferenced.

Then SAME exit125, BEFORE setter invocation for that attempt. It never skips or
disables a backend, never sets a CPU flag on a nonCPU object, and never proceeds
to computation after refusal. Successful CPU predicate path retains the same
setter/callback forwarding/plan checks/token loop; only the source revision
marker changes for build/run binding. No new accelerator policy is implemented.

## ROOT actual compile and invocation binding

Build V3 with build-argv-v3.json argv from MAIN repository CWD using the exact
pinned compiler. Preserve old binaries/build directories. Fresh build receipts
requested.json and result.json must both include actual `cwd`, `compiler`,
`compiler_sha256`, `argv`, reviewed V3 `source_freeze_sha256`, exact `source_inputs`
mapping from manifest compile_inputs, and runtime_or_math_qualified=false.
Record source_inputs before/after compilation. Result also records real exit,
binary_path/bytes/SHA. This closes the relative-argv CWD gap explicitly.

Use unchanged run_ref_control.py with new binary path/digest and a fresh output
name. The generic invocation correctly pins V1 launcher/helpers; it does not
claim that V1 CPP was compiled into this V3 binary. After actual invocation:

```sh
python3 .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/bind_v3.py \
  --manifest-sha256 REVIEWED_V3_SHA \
  --build ABS_FRESH_BUILD --run ABS_FRESH_CASE --new-binding ABS_FRESH_BINDING_JSON
```

Recipe authenticates all prior/new frozen sources, failed metadata, compiler,
exact new CPP/includes/CWD/argv/build exit/binary and actual runner inputs. A
failed diagnostic may bind source/build/invocation while remaining UNQUALIFIED.
Actual diagnostic interpretation is ROOT review; it is not a numeric verdict.

## Future policy feasibility only — NOT implemented

Pinned public scheduler graph routes:

- ggml-backend.cpp595: sync graph_compute delegates graph_compute_async.
- 601–604: graph_compute_async dispatches the selected backend iface directly.
- 1955/1977: both ordinary split and observed graph-view scheduler paths call
  that public async entry. 588–592: public graph_plan_compute is another direct
  iface route; checking only inner CPU ggml_graph_compute does not cover it.
- CPU cpp139–153 stores source cplan.use_ref at plan creation;162–190 both CPU
  plan/ordinary callbacks converge on ggml_graph_compute, whose current wrapper
  checks actual cplan.use_ref before/after. Old plans must not be silently fixed.
- BLAS cpp226–263 directly invokes CBLAS and bypasses inner CPU graph_compute.

If actual identity later proves a genuine ACCEL is initialized but unused,
unmodified construction could be recorded separately while a NEW reviewed
control rejects every nonCPU public async/plan compute BEFORE delegation.
Unknown/foreign backends remain cold refusals. Known CPU public calls could
carry a thread-local scope through original forwarding, bind actual inner CPU
plan observations to that backend, and require every original delegated plan
use_ref=true. Nested sync→async must count one compute rather than two math
invocations; capture every entry/return/success and all initialization/retirement.
Opaque plan handles need public create/free association if plan route is used;
no private plan/backend layout reinterpretation. Existing two-CPU counts cannot
be relabeled as all-backend coverage; constructed ACCELs require explicit records.

This is source feasibility, not compiled route closure. ROOT must prove actual
public interposition paths/DSO closure for both plain and observed modes before
allowing construction to continue. Direct/internal bypass or any unmatched
compute must refuse. Numerical all730/fullV/all42 and observation guards remain.

Crucially, op_offload=false does NOT prove ACCEL unused: backend.cpp1028 chooses
the highest priority compatible buffer/op; pass3 around1380 upgrades compatible
buffer nodes independently of op_offload. BLAS supports host-buffer contiguous
F32 GEMM with all relevant dimensions>=32 (blas.cpp397–459). Bulk64 QK/AV may
qualify. A future guard may correctly refuse; do not disable BLAS, alter buffers,
or silently downgrade math merely to make the control pass.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/frozen-v4/SOURCE_AUDIT_V4.md

Original UTF-8 bytes: 7843; SHA256 `df90f645366d93660a06c8decbb11e30e42b18b4aa5ca10dc7c9e35fbc87e9c0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# V4: exact BLAS lifecycle; CPU-only `use_ref` computation

- Owner: baseline_review. Parent: numerics-and-context.
- Additive source preparation only; no author build/test/native/model/runtime.
- V1/V2/V3, original token loop, generic runner, failures and source pins unchanged.
- V3 actually identified BLAS (backend/device/registry, ACCEL=3, `is_cpu=false`, original input device equality). This proves the refusing backend identity; it does not prove BLAS unused.

## Closed policy

- Same public `ggml_backend_dev_init` forwarded once. Classify returned backend using exact BASE public metadata, original input-device equality, CPU `is_cpu` or BLAS `is_blas`, exact public provider registry singleton and registry device0. Require original names/type. Constructor trace records actual checked identities, names, type and predicate.
- Same canonical-file/regular device+inode/length/exact-export V2 attestation; only `RTLD_LOCAL|RTLD_NOLOAD|RTLD_NOW`. BLAS adds actual ROOT pin: regular non-symlink64136B, dev38/inode2939302343, SHA `ab748c85dadb66ff128c1eda20a506bd6de94646cd06a7fbfd5b7a3881225e64`. ROOT hashes before/after; C++ checks immutable same-file identity and pinned inode/size before obtaining original public symbols. No new provider load or global promotion.
- Exactly two original CPU and two BLAS contexts; original loop keeps both contexts alive simultaneously. CPU init and original abort callback each apply the same V1 public `use_ref=true` setter. BLAS receives neither setter nor changed arguments/policy. Record constructor and completed retirement separately. Both CPU owners must actually compute; BLAS owners must only construct/retire.
- Public async, sync and opaque-plan compute wrappers reject every BLAS/unknown/retired target BEFORE original delegation. A real BLAS assignment causes exit125 with `nonCPU_compute_refused`, never a successful capture. Plan creation/free also require CPU: only BLAS construction/retirement is admitted.
- CPU async/opaque-plan call creates one thread-local public owner scope. Actual public CPU `ggml_graph_compute` must occur once inside that scope, with `cplan.use_ref=true` before/after, unchanged1..8 threads and successful status. Async binds exact graph pointer. Opaque plans use actual public create/free owner+generation associations; no private plan/backend layout is read. CPU implementation's graph copy is deliberately not confused with original public graph pointer.
- Sync must delegate once through same-backend/same-graph guarded async; its count is separate from math. Actual public/inner counts conserve globally and per CPU, with nonvacuous coverage. Missing/nested/duplicate routes refuse. Opaque-plan route is explicitly UNOBSERVED if count0; successful async evidence does not qualify it.
- Callback arguments, graph pointers, buffers, offload/repacking policies and original compute remain unchanged. No disableBLAS, wrong-backend setter, CPU forcing, math fallback, signed-zero exception, tolerance change or alternate token loop.

## Source and actual selected compiled routes

Pinned d812350 source in original acquisition receipt:

- `ggml-backend.cpp`: scheduler ordinary split and callback graph-view split call public `ggml_backend_graph_compute_async`; public sync calls async; public plan compute dispatches opaque backend callback. Device/registry getters are public metadata.
- `ggml-cpu.cpp`: ordinary callback creates public cplan with context `use_ref`, then calls public inner compute; opaque create stores same policy and opaque compute calls same public inner compute. Opaque internals are source-audited only, not accessed by V4.
- `ggml-blas.cpp`: BLAS ordinary compute invokes CBLAS directly, bypassing inner CPU compute. Thus old inner-only control was insufficient once BLAS exists. Device type ACCEL, names BLAS, actual public `is_blas`/registry functions available.
- `llama-context.cpp`: constructs ACCEL providers in addition to CPU. `op_offload=false` does not forbid all scheduler assignment: backend/buffer compatibility and scheduler upgrade rules still apply. V4 preserves them and refuses if they actually compute.

ROOT-produced `root-v4-elf-audit/` excerpts are pinned in `static-route-pins-v4.json`:

- BASE `ggml_backend_sched_graph_compute_async`: actual selected calls at `3cb18` (callback graph view) and `3d12b` (ordinary split) -> public async PLT `18840`.
- BASE sync: `375fc` -> same public async PLT. BASE async/opaque-plan entries dispatch original interface callback.
- CPU ordinary `1f34a` CALL and opaque-plan `1f384` TAILJMP -> public inner PLT `15a40`.
- Matching `R_X86_64_JUMP_SLOT`: BASE `0xdbb18` async; CPU `0x14de28` inner. Both selected providers lack dynamic SYMBOLIC.
- Pinned loader source `ggml-backend-dl.cpp` uses `RTLD_NOW|RTLD_LOCAL`, without DEEPBIND. This source observation does not substitute for actual runtime routing.
- These are selected-site static evidence for these exact compiled providers. They are **not** proof of every call in every DSO, complete dynamic binding, or runtime acceptance. ROOT must review them, compile actual exported wrappers and observe nonvacuous public-to-inner conservation. No private function offsets are called; no GOT patch, callback replacement, dispatch forcing or LD-preload arithmetic.

## ROOT build / run / validation

1. Independently review exact V4 manifest plus original freezes and selected compiled-route evidence. Do not execute old or newly broadened policy by inference.
2. Fresh finite strict build from MAIN CWD using `build-argv-v4.json` with actual pinned g++ path from manifest. Record requested/result JSON as prior V3: `cwd`, `compiler`, `compiler_sha256`, exactargv, THIS `source_freeze_sha256`, full seven-entry `source_inputs` before/after, `runtime_or_math_qualified=false`; result adds exit_code and actual output absolute path/bytes/SHA. Preserve all old builds/binaries.
3. Same unchanged `run_ref_control.py`, fresh output, actual new binary+SHA, `.8B` then2B, fresh8192 IDs, prefix/ubatch64 or512 and explicitCPU8/caps. This generic helper still binds its original V1 freeze; it is never relabeled as V4 source proof.
4. `bind_v4.py --manifest-sha256 REVIEWED_V4_SHA --build ABS_NEW_BUILD --run ABS_NEW_CASE --new-binding ABS_NEW_CASE/source-build-binding-v4.json`. Records actual new source, seven inputs/build/output, invocation/native log, selected static scope and extra provider. Can bind a failure; does not convert it into PASS. Unknown or actual BLAS compute remains exit125/UNQUALIFIED, even if its construction was legitimate.
5. ROOT-only source-free `python3 -m unittest test_control_v4.py` from this run; eleven prepared cases. Author AST only. Then `validate_control_v4.py --capture-run ABS_NEW_CASE --new-report ABS_NEW_REPORT` for a successful capture. New validator duplicates old capture/lifecycle/source guards, adds strict V4 trace reconciliation and required separate source/build binding, hashes actual original/extra providers before/after and verifies final exact BLAS mappings.
6. A control PASS establishes policy/route evidence and original plain-vs-observed full-logit/whole-serialized-state transparency only. Separate authentic raw730, full vocabulary and all42 domain comparisons against Rust remain mandatory. Historical prefix unrecovered; no historical/whole-cause/performance conclusion.

## Bounds

- Same native12GiB address/CPU8/finite deadline, teacher8192, batch64/512, original state/capture caps, owned supervisor cleanup.
- New maps: <=4 backends, <=64 simultaneous opaque plans, <=65536 total public plan generations; trace <=65536 records, <2048B per record, <=8MiB total. Record buffers/metadata are bounded; no learned matrices/copies/new native allocator.
- No production engine/profile/serving change, no historical8K divergence closure, no inference support claim.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/frozen-v4/TODO_V4.md

Original UTF-8 bytes: 1483; SHA256 `c9685704b9319c65a88d76cf8bfe5ad63c339a7dc1c1979c8c89d16072c99fb5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# baseline_review V4 working checklist

- Parent: `.agents/plans/numerics-and-context/`; root sole execution/merge coordinator.
- Scope: additive ignored public use_ref diagnostic/control only; no production changes.
- [x] Read V3 actual BLAS identity and ROOT actual selected ELF excerpts.
- [x] Preserve original14, V2/V3 freezes, original helper/runner/validators/tests and failed cases.
- [x] Prepare exact pinned BLAS constructor/retirement policy; refuse every actual nonCPU compute before delegation.
- [x] Prepare CPU public/TLS/inner pre/post use_ref conservation, opaque public plan owner/lifecycle and nonvacuous per-CPU checks.
- [x] Prepare strict source-only parser fixtures and source/build/invocation/static-provider binding.
- [x] AST parse new Python; small-source/excerpt hashes only. NO author tests/build/model/native/operator/GPU/runtime.
- [ ] Independent ROOT/peer source review.
- [ ] ROOT strict fresh C++ build and explicit new binary/source/actual build binding.
- [ ] ROOT execute eleven pure parser fixtures.
- [ ] ROOT fresh bounded actual model case; if any BLAS compute, preserve exit125/UNQUALIFIED, no policy forcing.
- [ ] If actual capture succeeds, ROOT V4 validator, exact raw730/fullV/all42 comparison and explicit scope disposition for both models.
- Unresolved: complete dynamic route closure until actual observation; opaque-plan runtime eligibility if unused; historical prefix and full historical8K cause. No inference/performance claim.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/frozen-v5/SOURCE_AUDIT_V5.md

Original UTF-8 bytes: 6216; SHA256 `b9811856fc6b474dbb1fb8dd69ef8f1d86ebbb8b35830908d9626ed46e363790`.
Resolve embedded relative paths against the original path, not this volume.

````text
# V5: observe the SAME V4 free refusal

- Owner baseline_review; source only. No author build/test/native/model execution.
- All V1–V4 sources, freezes, binaries and actual failed attempts remain unchanged.
- Sole diagnostic change: at V4's failing `ggml_backend_free` membership predicate, print fixed API origin, actual raw pointer, `is_null`, map-only tracked/retired status and already-observed owner/compute counts. No unknown/null/retired backend dereference, metadata getter, new DSO load, delegation, policy continuation or constructor allowance. Same `_exit(125)`.

## Actual V4 evidence

- ROOT strict build PASS3.332s, actual160960B SHA `41ccc52ad37f26077d2370e6101fbfe705ee45bd778210c3334ba1d341a3a357`; eleven pure fixtures PASS; source/build/invocation/provider binding PASS.
- Actual case `root-0.8b-8192-ref64-4`: native125 after model buffer loading, no public constructor trace/control report/accepted compute. Native log50345B, SHA `40f44109073c82d8e6063fa380666da75eb179cd877455c6015378af66e8beeb`; only BLAS and Haswell CPU load messages. Original raw capture records still empty.
- Actual owned cleanup receipt reports leader reaped, groups absent, nonzero signals before reap, EOF/selector/pipe closure and no cleanup errors. Failed case remains UNQUALIFIED.
- The actual failing pointer/API was not printed by V4. It is not yet established as NULL or a direct untracked CPU backend.

## Construction/free closure audit, pinned d812350

- `ggml/src/ggml-backend.cpp:380–386`: public `ggml_backend_free(NULL)` is explicitly a legal no-op; nonnull calls original interface free.
- V3's wrapper guarded `armed && backend`. V4's wrapper changed this to `armed`, then called `known(backend)`, which rejects NULL. This introduced an incompatible null-free guard in the control itself.
- `src/llama-model-loader.cpp:1493–1597`: each `load_all_data` initializes local `upload_backend` from a lambda. It returns nullptr for mmap/check_tensors; otherwise requires actual buffer/default device, async+host_buffer+events capability and pinned staging/events. Only a successful async-upload branch constructs a real backend, via PUBLIC `ggml_backend_dev_init` at1589.
- Same function1760 UNCONDITIONALLY calls public `ggml_backend_free(upload_backend)` after loading that tensor context. This includes null upload_backend. `src/llama-model.cpp:1950–1954` calls `load_all_data` once for each original tensor context/buffer map. Thus model-loading cleanup is separate from two inference contexts' CPU/BLAS owners and must never be counted as a live backend if NULL.
- Selected CPU provider `ggml-cpu.cpp:394–400`: actual CPU device props declare async=false, host_buffer=false, events=false. Even non-mmap CPU uploads cannot take that successful async constructor branch. Actual V4 model log reports CPU_Mapped and CPU_REPACK buffers, then refusal, consistent with null upload cleanup. This is a source-supported causal lead; actual V5 pointer observation is still required.
- CPU_REPACK is an extra buffer type and repacked tensor storage/operations. Source search through `ggml-cpu/repack.cpp` found no `ggml_backend_cpu_init`, `ggml_backend_dev_init` or `ggml_backend_free` call. Repacking is not evidence of an independently constructed backend. No repacking/buffer/math policy changes are proposed.
- CPU `ggml_backend_cpu_init` is a genuine direct public constructor at217; selected device init callback405 invokes it. Normal context route PUBLIC `ggml_backend_dev_init` therefore records it after return. No direct CPU init call is present in `llama-model-loader.cpp` or `llama-model.cpp`. Direct CPU compare/debug code exists elsewhere, but this helper does not invoke that API; it must not be silently admitted if actually encountered.
- BLAS direct public init286 is called by selected BLAS device callback377. Context construction `llama-context.cpp:335–362` adds actual model devices, every ACCEL via public dev_init, then CPU via public init_by_type. Registry convenience functions `ggml-backend-reg.cpp:366–389` all route to public dev_init.
- Original helper holds plain and observed contexts simultaneously, later explicitly destroys observed then plain. Each context owns its actual backend list via `ggml_backend_ptr`; header deleter is public free. CPU/BLAS actual interface free functions delete their own objects and do not recursively invoke public backend free. V4's exact two CPU/two BLAS count concerns these actual nonnull contexts only; no model-load null cleanup belongs to that count.
- Other source backend families (META/RPC/debug CPU compare) have constructors, but are not an excuse to broaden this control. Current selected log loads only pinned BLAS/CPU; V4/V5 refuse every unknown nonnull owner. Source inspection does not replace an actual dynamic constructor/caller observation.

## Prepared ROOT primitive

1. Fresh strict build `build-argv-v5.json`, same seven headers/included source and public exports, with manifest's actual g++ path. Retain requested/result metadata as V4, using THIS reviewed V5 SHA and actual new `native_ref_control_v5` output; preserve older builds.
2. Same unchanged generic runner, fresh case, same model/IDs/caps. Expected diagnostic still125; do not reinterpret it as numerical progress.
3. `bind_v5.py --manifest-sha256 REVIEWED_V5_SHA --build ABS_FRESH_BUILD --run ABS_FRESH_CASE --new-binding ABS_FRESH_CASE/source-build-binding-v5.json`. Root-only existing provider/route/source binding reused; report only source/build/invocation.
4. Read the actual `free_refusal_diagnostic_v5`. If NULL with zero tracked owners/compute, it proves this refusal is the wrapper's legal-null-free incompatibility, distinct from model backend construction. Only then consider the separate narrow restoration of original null-free forwarding with no owner/setter/compute count. If nonnull, retain failure and audit actual public origin; never dereference unknown/retired state or add a blanket constructor allowance.

No V5 success/control validator or numerical eligibility is claimed. Source/static normalization confirms removal of only diagnostic + revision marker yields exact frozen V4 CPP. No new refusal framework or policy changes.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/frozen-v6/SOURCE_AUDIT_V6.md

Original UTF-8 bytes: 3211; SHA256 `187d1c28a86b2694f919c0f357bb10a06fcdd0a6a107c2d1fe89ea138b159502`.
Resolve embedded relative paths against the original path, not this volume.

````text
# V6: restore original legal NULL-free forwarding only

- Source-only additive; owner baseline_review. V1–V5 and all failed attempts immutable.
- ROOT V5 actually observed `api=ggml_backend_free backend=(nil) is_null=true tracked=false retired=false tracked_backend_count=0 actual_public_compute_calls=0 actual_inner_calls=0 original_free_delegated=false policy_continued_after_refusal=false`. Thus the model-loading refusal was V4's null-free guard regression, not an observed untracked CPU constructor.
- Pinned BASE public `ggml_backend_free(NULL)` immediately returns. Loader unconditionally frees its possibly-null `upload_backend`; this is separate from real context backend lifetimes.
- V6 CPP is frozen V4 plus TWO `&& backend` conditions in the public free wrapper (before and after original free), plus the revision string. Original NULL free is forwarded exactly once, outside tracking/counters. No no-op backend identity is created; no null setter/compute/retirement trace or count is invented. Every NONNULL V4 classification, lifetime/count, BLAS refusal, public/TLS/inner use_ref, plan association and original compute remains byte-identical.
- Removing those two conditions and changing only revision back to V4 is statically asserted to produce exact V4 bytes. No V5 diagnostic is imported into this control.
- New successful report/validator revision: `public_use_ref_exact_blas_cpu_compute_legal_null_free_v6`. The exact V4 trace parser and eleven pure fixtures are reused unchanged. Original native token loop, source/profile, buffers/CPU_REPACK, offload flags, public BLAS lifecycle/never-compute policy, all arithmetic and generic runner are unchanged.

## ROOT follow-through

1. Independently review tiny `NULL_FREE_FORWARDING_V6.patch` and new source/binding/validator revision. Build from MAIN CWD with actual g++ and `build-argv-v6.json`; same seven input and fresh requested/result/binary receipts as V4/V5, using THIS reviewed V6 SHA. Preserve all old builds/binaries.
2. Same unchanged generic runner, actual fresh `native_ref_control_v6` SHA/fresh case and original source/teacher/CPU8/8192/prefix caps. No accelerator disable or inference-policy workaround if BLAS actually computes.
3. `bind_v6.py --manifest-sha256 REVIEWED_V6_SHA --build ABS_NEW_BUILD --run ABS_NEW_CASE --new-binding ABS_NEW_CASE/source-build-binding-v6.json`. Requires immutable V1–V5, actual build/seven inputs/output/invocation plus original runtime/static provider evidence. It can bind a failure; it cannot qualify arithmetic.
4. Reuse `python3 -m unittest test_control_v4.py` unchanged. A successful fresh case additionally requires `validate_control_v6.py --capture-run ABS_NEW_CASE --new-report ABS_NEW_REPORT`; correct new revision and actual V6 binding, SAME strict V4 trace equations and original full730 metadata/observer guards. Complete dynamic route/policy evidence still requires ROOT observation. Separate raw730/full-vocabulary/all42 comparisons remain mandatory for both models.

Author: two Python ASTs and small-source/metadata normalization only; no tests/build/native/model/GPU/runtime. No long-context/cause/full-model/performance qualification is implied by this control repair.

````

## Source: .agents/runs/numerics-and-context-use-ref-bulk-control-20261020/frozen-v7/SOURCE_AUDIT_V7.md

Original UTF-8 bytes: 2518; SHA256 `0226a058721323d17437908172cd8a751128ac63cfaf8d6f088068dd0b3a9029`.
Resolve embedded relative paths against the original path, not this volume.

````text
# V7: metadata at the SAME nonCPU refusal

- Source-only; no author build/test/native/operator/model/runtime execution.
- V1..V6 source/freezes, failed attempts and ROOT .8B V6 result unchanged.
- Actual ROOT2B V6 refused a pinned BLAS public compute before delegation.
  Actual operator identity is not yet established. In particular F32 alpha
  W[2048,16] has output16 and fails BLAS's pinned MUL_MAT minimum-output32
  condition; do not attribute this refusal to alpha from a split count alone.
- V7 passes only the already available public graph argument into that same
  guard. Before its unchanged exit125, emit actual graph count and up to128
  nodes, then every nonnull public src[0..9] header for those nodes. Node/input
  records contain opaque identity, bounded byte-escaped name, type/op IDs and
  public names, and four integer dimensions. No payload/data/buffer/plan/backend
  private field, callback, copy, source value, flag, scheduling or arithmetic.
- graph_n_nodes/graph_node/op_name/type_name resolve through the same exact
  pinned BASE NOLOAD/canonical/inode/length/export attestation. Public pinned
  ggml.h defines the tensor header and these APIs. ROOT actual BASE symbol
  listing confirms all four exports; this is static metadata, not proof of
  new runtime/provider route coverage.
- Summary states actual_node_count and retained_nodes; >128 or absent public
  graph is explicitly incomplete, never silently a complete graph. Opaque plan
  APIs expose no public graph argument: emit unavailable metadata, never peek.
- Existing2048B/record,8MiB trace and65536-record bounds still apply. At most128
  node +1280 input records; each header independently escaped/bounded. Exact
  same known/live backend, CPU setter/use_ref/TLS/plan and legal NULL-free
  policy; no BLAS allowance/delegation, unload, disable, op_offload or buffer
  change. Removing diagnostic helper/call/public graph argument and revision
  reproduces frozen V6 byte-for-byte.

ROOT: compile fresh build-argv-v7.json with pinned compiler under the SAME
bounded build receipt schema/source_inputs pre/post; preserve V6 failed paths.
Run fresh2B case via unchanged generic V1 owned runner, actual V7 binary SHA.
`bind_v7.py` binds THIS source/build/binary/invocation/trace to fresh receipts.
Actual refusal remains UNQUALIFIED; read bounded V7 graph metadata before any
new routing policy. `validate_control_v7.py` retains the unchanged V4 trace
parser/11 fixtures for a future successful control; no new numeric acceptance.

````

## Source: .agents/runs/p12-full-fusion-batch-design/TODO.md

Original UTF-8 bytes: 1136; SHA256 `e60cb978dd1e83593c1ad801abffac2e150006cba1d2051d0316c1bebaf10740`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12 batch lowering

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Owner: batch-design agent
Scope: `crates/zerogravity-gpu/src/resident_program/batch.rs` only.

## Uncommitted
- [x] Inspect original ForwardPlan, resident lowering, state ownership and batch operators.
- [x] Agree shared-source BatchProgram seam with MAIN.
- [x] Implement native B2/4/8 lowering of the original operation traversal.
- [ ] MAIN: compile, physical trained parity and full isolated gate.
- [ ] Independent review and reconcile results with MAIN.

## Tests
- No Cargo or physical tests by this agent; MAIN is the sole coordinator.
- `rustfmt --edition 2024` passed for the new source.
- MAIN's first compile found allocator closure lifetime inference errors;
  added explicit closure argument/result types.
- Review found the original transport/certificate 1,000,000-word dispatch
  bound; added bounded dense copies and certificates for complete batch
  logits/KV, preserving original attention kernels/padding. MAIN rerun pending.
- Greedy/device-token batching and HTTP fused enablement remain outside this
  diagnostic full-logit milestone.

````

## Source: .agents/runs/p12-full-fusion-qualification/TODO.md

Original UTF-8 bytes: 1590; SHA256 `c28021fb2446c44b1d98e2fa4ac6a66c6d0db1a8d02f76c079579f17952b48bf`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12 full-model fused qualification

Parent: [serving-and-speculation](../../plans/serving-and-speculation/TODO.md)
Owner: baseline_review subagent
Scope: new `resident_fused_qualification.rs`; this run file only.

- [x] Revalidate signed clean baseline, source hashes, hardware and old receipts.
- [x] Draft physical full-model qualifier against independent CPU and accepted B1 GPU.
- [x] Distinct nonzero two-token histories, complete 248320 logits/all42 domains.
- [x] Private/drop, invalid/duplicate/reordered row, stale epoch/root rollback.
- [x] Count real native dispatch receipts; check Ready/lease/resource return.
- [x] Draft source-real B2 future-abort qualifier with original in-flight RAII,
  both immutable nonzero priors, quarantine/charge/fence and refusal checks.
- [x] Report native dispatch subset accurately as projection dispatches.
- [x] Draft portable-checkpoint B2 boundary qualifier: source-real CPU prefixes
  0/255/511, independently imported B1/native lanes, exact valid1/256/257/512
  logits/state, dispatch boundary, capacity refusal and complete lease return.
- [ ] MAIN wires child module/CLI and batch API/counters.
- [ ] MAIN compiles/lints; run physical B2 before B4/B8 and 2B.
- [ ] Physical CUDA execution, isolated full gate and both release parity routes.
- [ ] MAIN runs fused abort in its own process; no recovery/reuse after abort.
- [ ] MAIN wires/runs `resident-fused-boundary-cuda`; CPU hydration is setup,
  not evidence of GPU continuous full-prefix prefill.

No Cargo, tests, GPU launches, dependencies or model downloads run by this agent.

````

## Source: .agents/runs/p12-full-fusion-root-2026-10-19/TODO.md

Original UTF-8 bytes: 4718; SHA256 `d582c59b7dcd3ab62791d3b7e60a0ebf42f271519bbc0f3b0c4d2a6328dd48bc`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12 batch-native resident forward

Parent: `.agents/plans/serving-and-speculation/TODO.md`
Base: `c85ad250334f590a87c99710e95c71a3deb28323`
Owner: root; sole MAIN Cargo/test coordinator.
Scope: resident program/inference integration, module/tool wiring, plan reconciliation.
Independent owners: batch lowering child; physical qualification child; lease completion child.

- [x] Read full comprehensive handoff progressively; inspect live rules/plans.
- [x] Revalidate clean main/+10, sole worktree, no Cargo/server; healthy idle RTX3060.
- [x] Approved local model hashes match; installed CUDA12.9 inputs available.
- [x] Integrate B-native full ForwardPlan and independent candidate state/Ready.
- [x] First CUDA .8B B2 physical exit0: all248320 logits/all42 domains per row exact CPU+B1;
  2 decode steps: 11442 actual launches, 374 native projections versus22024 separate B1 launches;
  abandoned/invalid/duplicate/reordered/foreign/stale epoch/root refusals, sibling replay,
  no warm source uploads, events/leases/final logical envelope reclaimed.
- [x] Independent final lowerer/Ready/qualifier review; strict CUDA Clippy exit0.
- [x] .8B B4/B8 and2B B2 physical exit0, CPU+B1 exact full logits/all42 domains.
- [x] Remaining2B B4/B8 physical matrix; boundary/abort; default/CUDA/Vulkan feature gates.
- [x] Frozen isolated check.sh and separate original cache-direct/HF-import release gates.
- [x] Reconcile owning plan and signed exact-path commit; root README unchanged.

Limits: dotlink connector not exposed in tool catalogue/resources; direct workspace used.
vLLM absent; no reference benchmark, SDK/model downloads or AMD launches authorized.

Implementation/review fixes:
- First cargo check exit101: allocator closure HRTB inference; explicit closure types fixed.
  Same check then exit0. No numeric expectation/tolerance changes.
- Cold init lock cycle fixed: allocation -> queue -> workspace, terminal init before requeue.
- Warm validation precedes submission; no await after submitting except actual completion/abort hook.
- B8 vocab and lifted KV exceed old1M-word transport helper; chunk in new lowering,
  retaining the original bounded ABI and counting each actual chunk launch.

Active validation: serial physical matrix session96549, now2B B4; logs in this directory.
Boundary/abort qualifiers wired but not yet executed; all MAIN Cargo remains root-owned.

- Source freeze note: diagnostic-only MixedBatchStorage now cfg(qualification/test);
  default strict lint will check it. Effective CUDA qualifier/kernel math unchanged.

- Comparator self-test first isolated invocation exit1: private loopback is DOWN
  (URLError101), before endpoint requests. No harness/source numerical failure.
  The remaining-gates runner repeats after check.sh enables ONLY guarded private lo.
- Release qualifier build session40903 overlaps only source-real numerical B8 work;
  root remains sole compiler/test coordinator and B8 is the sole GPU workload.

- [x] All6 short-prefix physical cases exit0; .8B and2B B2/B4/B8.
  Each all248320 logits/all42 per row exact CPU+B1; two decode steps and negatives.
- [x] Source-free GPU unit51pass0fail3ignored; strict default GPU Clippy exit0.

- [x] Strict CLI serve+cache CUDA and Vulkan Clippy exit0; no Vulkan/AMD launches.
- [x] Release qualifier build exit0 (13m16s); active boundary .8B via runner85847.

- [x] .8B release B2 boundaryvalid1/256/257/512 exact CPU+B1/all42, refusal/reclaim.
- [x] .8B release B2 actual launch/drop quarantine1820handles, unchanged prior roots/all42,
  retainedfull4GiB envelope, stale Ready/nextstep/othermodel refused; no recovery.

- [x] 2B release B2 boundaryvalid1/256/257/512 exact CPU+B1/all42, refusal/reclaim.
- [x] 2B release B2 launch/drop quarantine1715handles, unchanged prior roots/all42,
  full4GiB retained/fenced; stale Ready/nextstep/othermodel refused; no recovery.
- Active: full-check in runner85847. All requested physical cases passed.

- [x] Full isolated check exit0:1147pass0fail69ignored/241Rust suite runs;18Python,
  releaseCLI/installer/Nixfmt. Both original .8B/2B RELEASE cache-direct17.69s
  and HF-import22.74s exit0. Comparator self-test exit0 after private loopback setup.
- [x] Final independent doc/source review; evidence scope corrected and verified.
- [x] Signed implementation+owningTODO commit; then substantive acceptance docs
  and commit-group reconciliation, preserving root README and all other plans.

Finished: signed implementation35ebc5b and acceptance904a190; both signatureG.
MAIN clean/+12 localorigin; no push. RootREADME originalblob unchanged.
All source/tests/remaining scope reconciled into the owning plan.
GPU idle56C/41MiB after all qualification processes exited.

````

## Source: .agents/runs/p12-q8-fusion-2026-10-19/TODO.md

Original UTF-8 bytes: 1026; SHA256 `5207612e2bf0f0bd027298ce6945ce722a1e77e229ac9f83399edda8b60bc596`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12-B2 source GGUF quantized batch projection

Base: 08be9c6
Plan: [P12](../../plans/serving-and-speculation/FUSED_GPU_BATCHING.md)

- [x] One original B1/B>1 source dot for Q4_0/Q4_1/Q8_0/Q6_K.
- [x] All 12 codec×batch B2/4/8 physical RTX3060 CUDA
  cases bit exact with one native fused projection launch.
- [x] Typed source-only metadata and malformed geometry tests.
- [x] Strict default/CUDA all-targets Clippy -D warnings.
- [x] Isolated scripts/check.sh: 1,147 Rust tests 0 failures,
  69 ignored across 241 suites; 18 Python fixtures.
- [x] Original trained Qwen3.5 .8B/2B direct-cache and
  HF-import exact full248320 logits and 42 state domains.
- [!] Offline Nix source-integrity blocked by uncached pinned
  nixpkgs/CA trust; Nix formatting itself passed.
- [x] SSH-signed commit 3ec431a, eight reviewed source/docs
  paths, post-commit rustfmt and acceptance receipts pass;
  no push or model downloads.
- [ ] Full native GPU resident multi-row forward and vLLM
  performance comparison are not yet qualified.

````

## Source: .agents/runs/p12-resident-cohort-2026-10-19/TODO.md

Original UTF-8 bytes: 1378; SHA256 `2e947c1fbd807744b3a5ad46657649d7851b1d43821787520ebdb79b65f64e19`.
Resolve embedded relative paths against the original path, not this volume.

````text
# P12-A1 resident GPU state cohort — current run

Base: 2869f87
Plan: [P12](../../plans/serving-and-speculation/FUSED_GPU_BATCHING.md)

- [x] Source-specific RootBinding cohort, actual per-row GPU Ready
  checks and one fence-guarded original SessionAuthority commit.
- [x] Physical CUDA B2/4/8 each exact trained Qwen3.5 0.8B full
  248320 logits and all42 state domains against CPU.
- [x] B2 dropped/invalid/stale rollback; surviving session works.
- [x] Live HTTP CUDA B2 one shared cohort publication and
  identical short responses, zero failed/leaked sessions.
- [x] Strict Clippy CPU/CUDA/Vulkan all-targets -D warnings.
- [x] Complete isolated scripts/check.sh exit0: 1,147 Rust
  passed, 0 failed, 69 ignored over 241 suites; release CLI,
  installer, 18 Python fixtures, Nix formatting and CPU comparators.
- [x] Approved original trained Qwen3.5 .8B/2B direct-cache and
  HF-import independent release parity: exact 248320 logits
  and all42 StateRoot domains, leases/sessions intact.
- [!] Separate offline Nix filtered source-integrity could NOT
  evaluate: locked nixpkgs GitHub input uncached and connector
  SSL trust path invalid (CA certificate error 77). This is an
  environment blocker; Nix formatting itself passed.
- [ ] Scoped SSH-signed commit, post-commit source checks; no push.
- [ ] Full model native fused B>1 kernels and vLLM parity still open.

````

## Source: .agents/runs/qwen-original-bind-2026-10-18/TODO.md

Original UTF-8 bytes: 1406; SHA256 `011bec29c16a1ae23a2ed65d68f180695117a5d2195146bad8064e63ce8a0845`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Original Qwen source binding — run checklist

Provenance: 93a5622.
Main plan: [model execution](../../plans/model-execution/TODO.md).
Scope: original Qwen text source geometry, source-digest binding,
output-first cold CPU embedding/vector/projection and regression.
No model weights, SDK, held GPU launches, or pushes.

- [x] Exact indexed-source SHA and caller-authorized config SHA
  bind to one source; all original ordinary-text tensors checked;
  vision/MTP cannot be promoted into this executable text profile.
- [x] Original HF output-first math reuses Rust CPU MatVecRegion.
- [x] F32/F16/BF16, missing/extra/wrong source roles, bad config/hash,
  unsupported accelerators, finite input and oversized-before-dispatch
  refusal tests pass.
- [x] Full frozen scripts/check.sh 1,103 Rust passes/0 failed/69
  ignored/238 suites plus 18 Python fixtures.
- [x] Independently reran BOTH Qwen3.5 .8B/2B trained
  original source-parity release routes after final guard:
  cache/direct 16.70s, HF-import 22.88s, full logits and 42 StateRoot
  domains/2 tokens/independent lease and mutation correctness.
- [x] Strict default and CPU/acquisition/cache Clippy --all-targets
  -D warnings passed after guard.
- [ ] Post-commit Nix source-integrity and scoped signed checkpoint.
- [ ] Later: actual trained 27B full-logit/StateRoot, tokenizer,
  GDN/GQA math/rounding, SSD streaming and GPU qualification.

````

## Source: .agents/runs/retired-side-worktrees-2026-10-18/zerogravity-filters-work/.agents/runs/model-acquisition-filters/TODO.md

Original UTF-8 bytes: 22153; SHA256 `8fb40a46d132879ae46ea597a4c74ed9a9e16cfd1f3e680dabde9734918144de`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — additive filters and exclusions cold design

Status: PAUSED at user request; partial unwired SIDE source draft preserved, all Rust checks UNRUN.
Provenance: C1 frozen9path candidate inventory + C2 source-only quant11paths at46cf5a5 overlay; MAIN conditional/candidate gates coordinated separately.
Main plan: [model acquisition](../../plans/model-acquisition/TODO.md).
Owned scope: approved11Rustpaths only in /home/pvl/spaces/abird/src/zerogravity-filters-work (4db5efb +exact C1/C2 overlays) plus this own MAIN run. No MAIN source/Cargo/Git/canonical writes, payload/provider/account/model/SDK/cache/device actions.

## Uncommitted

- [x] Read original byte-component */? matcher, closure and current candidate/quant initial-choice seams without treating drafting code as frozen proof.
- [x] Verify primary Python/HF/vLLM filename matching requirements and choose exact additive full-path UTF8 grammar, brackets/ranges/negation/malformed literals/slash/case.
- [x] Define deterministic nonexponential bounded work/memory/cancel, includeunion->exclude->familyambiguity and required-versus-optional closure exclusion policy.
- [x] Propose additive API/CLI, explicit quant/native composition and meaningful tests, preserving legacy Files grammar and sole provider inventory/budget/publisher.

## Primary requirements and chosen grammar

[Python3.14 fnmatch documentation](https://docs.python.org/3.14/library/fnmatch.html) establishes full-string, case-sensitive fnmatchcase semantics: separators and leadingperiods have no special status; ? consumes one character, * any sequence. [HF download documentation](https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download) supports combined allow/ignore lists. These justify an additive Unicode path matcher, not changing existing component-local byte Files behavior.

Use named fixed grammar `FnmatchCaseV1` on valid UTF8 scalar strings; no locale/case/Unicode normalization. `*` includes slash; consecutive stars collapse with no specialglobstar/recursive-directory operator. Literal slash remains literal, and ? may itself consume slash. Exact repository paths remain validated separately; patterns only choose among owned pinned paths, never authorize URL/path traversal or discovery of a new source. With a subfolder, enforce its literal component-boundary scope first and match the remaining full relative suffix; never interpolate the folder text into pattern syntax (folders can themselves contain brackets). Without it, match full repository-relative paths.

[CPython v3.14.0 matcher source](https://github.com/python/cpython/blob/v3.14.0/Lib/fnmatch.py) and [its fixed tests](https://github.com/python/cpython/blob/v3.14.0/Lib/test/test_fnmatch.py) define bracket edge behavior: [seq], leading! negation, Unicodeordinal ranges; ^ is literal. Initial ] inside a class is literal. Unclosed [ is literal, then remaining pattern parses normally. Hyphens at class ends are literal; descending closed ranges contribute nothing, and negating an empty normalized set matches any scalar. Nested/set-operation punctuation has no POSIX/intersection meaning. Backslash is literal, never a general escape. Port the bounded class normalization into token/range data, not a regex runtime, and validate unusual multiple-hyphen cases against the pinned tests before claiming parity.

Patterns: nonempty <=4096UTF8bytes each; combined include+exclude count<=4096, aggregate pattern bytes plus inventory path bytes<=8MiB. Reject NUL/control inputs before HTTP and do not support bytes/lone-surrogate Python inputs; actual sourcepaths already exclude them. No requirement that every include pattern matches individually: include lists are union, and unmatched excludes are harmless. Empty final selection is an explicit refusal, never select-all or first-file fallback. Excludes-only without an explicit quantbase also refuses; callers can deliberately request --include '*'.

Pinned [vLLM source a8be1500bad93d3c459e435bf71194aa8062e49d](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/model_executor/model_loader/weight_utils.py#L399) forwards ignorepatterns and may choose a first matching format. Zerogravity retains explicit selection and refuses ambiguity; it does not inherit that numeric/source fallback.

## Ordered selection and closure

1. Validate bounded request/patterns before provider activity; acquire original operation controls and matcher scratch before allocation. Pin/list once through the same complete private inventory used by C1/C2; validate completeinventory before subfolder projection, preserving accepted collision fix.
2. Match all include patterns as a union over scoped regularpaths, then subtract union of excludes. Determine family ambiguity only AFTER this subtraction; calling current select_paths first would reject before exclusion can intentionally narrow choices. Reuse original GGUF family parser/complete1..N and safetensors/index logic, not new filename heuristics.
3. Apply original model closure with private exclusion policy/reason awareness. Required config/tokenizer, complete split/shard cohort and source index cannot be excluded: fail before any forbidden payload/plan/publication. Excluded optional known sidecars (README/LICENSE/tokenizer_config/etc) are skipped; they cannot be silently re-added by generic closure. If any path is both optional and structurally required, required wins. Include patterns constrain initialchoice; they do not forbid required dependencies. Excludes are the explicit veto.
4. Excluding every file of a whole competingfamily may disambiguate deliberately; excluding only one member of a stillselected splitfamily must fail requiredclosure, not produce a partial model. Excluding an index while its shards remain cannot erase the dependency. Preserve original ambiguousindex/family/regularmode/path checks and explicit external sidecar supplier authority.
5. Final exact per-file bounds/provenance/source identity, rawSHA/EOF/LFS/progress/cancel and sole original publisher remain unchanged. Selectionpolicy is not new content authority or tensor/loader support. No filteredplan with placeholders or postpublication file deletion.

Index policy decision for root before source: safest smallslice forbids reading an explicitly excluded index body. Existing closure discovers relevant indexes by reading adjacent candidates, so an excluded adjacent candidate needed to establish membership must cause typed refusal before GET, even when later parsing might have proved it unrelated. Do not skip it and silently weaken closure. Record this conservative excluded-candidate-index boundary explicitly; broader membership discovery can be separately qualified if useful. No user permission gate or new source authority is implied.

Private code seam: existing require(path) gets Required reason+policy; optional additions use add_if_present_allowed(path). Old Files/quant/component/default paths use AllowAll policy byte-for-byte behavior. HF's original closure and shared GH/ModelScope closure have differences (including legacy .bin roles), so do not unify all implementations under this feature merely to share excludes. Share only the new pure matcher and small exclusion decision; preserve each old owner and provider metadata callbacks.

## Bounded matcher and admission

- Compile one pattern at a time to Literal/Any/Star/Class tokens; no global cache of compiled patterns, no regex crate/runtime and no recursive backtracking. Two DP/NFA state rows scan Unicode scalar filename input with star epsilon transitions. Work polynomial in patternunits*pathscalars, with class-range comparisons charged too; parse/normalize including malformed bracket lookahead must consume work before proceeding.
- Fixed cumulative16Mi state/character/range/compile transitions per complete operation, never reset perpattern/path/include/exclude/closure/refresh. Exhaustion returns typed Limit, not false/no-match/fallback. Cancellation checked before allocations/HTTP and at most4096transitions apart. Collapse repeated stars and bound unmatched-bracket/range normalization work; avoid naive exponential star matching.
- Scratch ceiling512KiB for one <=4096byte token/class representation, two state rows and Unicode path buffer. Precompute exclusion membership once over inventory, then reuse checked lookup during required/optional closure rather than reevaluating patterns each callback. Bounds must account actual retained path keys/collectionnodes; use compact inventory indices or separately charged bounded copies, no uncharged second full inventory.
- Recommended conservative resolve-only extra16MiB control allowance in the SAME existing TransferAdmission/arbiter owner, covering scratch +worstbounded excludedpath keys/nodes. Combine with original operation reservation before new allocations; no new resource domain/allocator/ledger and no silent reduction of existing metadata buffers. Default original80MiB metadata+16MiB remains below ordinary128MiB, but explicit lower caller limits still legitimately refuse. This number must be reconciled with actual chosen representation and tested, not treated as physical all-RAM telemetry. Drop matcher/control state before payload/cache transaction; native256MiB policy unchanged.
- All existing logicalmetadata/body/probe/file/path/manifest bounds remain cumulative across the one pin/tree/closure operation; original HTTP still owns actual retries/deadlines. Never discover_candidates()->resolve(), reset budget inside a selectionpass or materialize a second catalog/cache.

## Additive public and CLI seam

Proposed `acquisition::filters` public private-field type:

```rust
FilterRequest::new(source: CandidateRequest, includes: Vec<String>, excludes: Vec<String>) -> Result<Self>
FilterRequest::for_quant(quant: GgufQuantRequest, excludes: Vec<String>) -> Result<Self>
filters::resolve(transport, &FilterRequest, &PullLimits, &TransferAdmission, &AtomicBool) -> Result<ResolvedPullPlan>
filters::validate_cached_selection(&SnapshotLease, &FilterRequest, &PullLimits, &AtomicBool) -> Result<()>
xet::resolve_filtered(transport, &FilterRequest, &PullLimits, &XetOptions, &TransferAdmission, &AtomicBool) -> Result<ResolvedPullPlan>
```

Names are a proposal until root freezes C2 interfaces. No Selection enum/public PullRequest/PullLimits change. Private initialchoice hook grows from C2 Files/Quant to Filtered(Patterns or Quantbase,excludes), sharing inventory and finish closure; normal old Files and quant route untouched. One new patternengine implementation serves HF/GH/newnative/offline paths.

CLI proposed additions on explicit HF/GH/native Xet file routes: repeated --include GLOB with optional repeated --exclude GLOB; require at least one include. New flags reject any --file/asset/release ambiguity before I/O, so old --file byte*/? grammar never silently changes. Explicit --quant TAG --exclude GLOB is a second base mode: match all requested-tag candidates, subtract excludes BEFORE familychoice, then enforcecompletefamily. Ban --quant+--include in the firstslice rather than invent their intersection/auxiliary semantics. Pure --quant without excludes retains qualified C2 behavior. --exclude alone is invalid. Candidates/release/components/cloud/ModelScope modes do not acquire new semantics implicitly.

Native HF applies filtering INSIDE original Hub/pin/tree metadata pass before discovery; reuse its exact request/body/control budgets, read grants/credential suppression/Xetroot/rawSHA/sourceEOF/ephemeral guard. Must retain >=1announcednative file and current native-index refusal; excluded nativeobjects cannot trigger compatibility fallback. No ordinaryfiltersresolve then second nativealiasresolve. If this thin path cannot be qualified in C3, reject its flags explicitly and keep nativefilter item unfinished.

Offline terminal branch precedes credentials/client/admission/signals, using retained fully verified original source lease/currentlimits/pointer policy and the same bounded matcher/strict closure under honest cachedsubset scope. It neither removes excluded cached data nor publishes a filtered snapshot. Validate whole requested result consistently with C2's final offline contract; if returning a whole SnapshotLease as exact result, reject manifestextras outside finalclosure instead of implying excludedbytes vanished. No remote-global-family completeness/freshness/privateaccess/nativeexecution proof from retainedsubset.

## Meaningful future qualification (all UNRUN)

- [ ] Golden primary fnmatchcases: UTF8scalar? vs original byte?, slashcrossing*/?/classes, dots, caseexactness, composed/decomposedUnicode distinction, collapsed**; classpositive/negation/ranges/nonASCII/leading]/literal^/hyphens/backslash/metacharquoting/unclosedbrackets/descendingranges and setpunctuation. Fixed expected cases from primarytests/independent Python oracle BEFORE Rust implementation, never regenerate expectations to fit new engine.
- [ ] Bounded exhaustive/differential short-string/alphabet tests versus pinned fnmatchcase oracle, with fixed seed/finite cases; no Python runtime dependency or compiler-generated expected values at productruntime. Reject malformedinputpolicy deviations explicitly in docs.
- [ ] Adversarial repeatedstar/no-finalmatch, longmissingbrackets, largeclasses and manyindividuallyfit patterns exhaust one sharedwork cap with typedLimit, bounded measured buffers and cancellation, not exponential hang or false no-match. Admission/explicit tiny-RAM refusal before HTTP/new allocations and default128/native256 envelope positives.
- [ ] HF/GH includeunion thenexclude narrows twoGGUFfamilies correctly, multipleunmatchedinclude patterns tolerated, emptyresult nofallback; old --file still componentlocal bytegrammar exactly. Split partialexclusion/missingmembers/conflictingtotals and sourceindex/config/tokenizer exclusions refuse; optionalREADME/tokenizer_config/LICENSE exclusion stays omitted, never re-added.
- [ ] Adjacentexcludedindex policy refuses before itsGET, no unqualified membershipskip; malformed/ambiguous/incomplete source closures retain oldcriteria. Exactcase/subfolders containingbrackets treated literally; samefolder componentboundary and fullinventory regularancestor collision refusal preserved.
- [ ] Movingbranch/fulltreepagination prove one pin/list and sourcecommit through HEAD/index/body; cumulativelogicalrequests/metadatabytes/header/probes/path/file/control checked once, no reset/extra catalog. Old provenance/digest/source identities equal corresponding explicitfile acquisition.
- [ ] Actual originalowner/native flow tokenisolation/metadataonlydryrun/defaults/cancel/root/fullSHA/EOF/oldref/rollback; cacheddata choice and explicitoffline before env/FIFO/stdin/client/admission/signals, source/currentlimit/pointer/closure/extras negatives, no data deletion/new snapshot.
- [ ] New CLI modes/conflicts/JSONsafe diagnostics/optionalfeatures and old API/exhaustivematch compatibility; strict production/diagnostic, original provider/component/native/legacy regressions, full isolated gate/sourceNix and original model support evidence stay coordinator-owned. No loader/tensor quant support claim.

## Remaining choices before implementation

- [ ] Root freeze final source-qualified C2 initialchoice/API names, excluded-adjacent-index policy and exact matcher control allowance. This is normal design ownership, not a requirement for new user authorization.
- [ ] Source grant, implementation, independent review and actual tests remain pending. C3 is not closed by this proposal or by typed Unsupported boundaries; broader filter compatibility must follow verified implementation.

## Approved C3 source grant

- [x] Root approved exact11path/API outline: artifact acquisition/{filters.rs NEW,quant.rs,selection.rs,huggingface.rs,github.rs,mod.rs,xet/mod.rs}, tests/acquisition_filters.rs NEW; CLI src/{pull.rs,pull_xet.rs}, tests/pull_filters.rs NEW. candidates.rs only if concrete need arises (none planned). API uses private FilterRequest::new(source,Vec includes,Vec excludes), ::for_quant(quant,Vec excludes), source/includes/excludes/quant_request getters; filters resolve/cachedvalidator and native resolve_filtered.
- [x] Root approved excluded-index refusal only where original closure would attempt that index; keep unrelated/GGUF paths unaffected. Fixed matcher bounds and conservative combined extra16MiB SAME admission; offline new mode exact closed snapshot subset and digest-only.
- [ ] Implement matcher and provider/closure/CLI/native reuse, independent fixture coverage, static formatting and independent source review. All Rust compile/tests/lints/gates UNRUN until MAIN grant.
- [ ] Oracle setup initial attempt failed before freezing any output: pinned Python3.14 fnmatch source uses new strict-end regex spelling \z; installed Python3.13 rejects it because initial replacement accidentally looked for two backslashes. Correct only the spelling adaptation to synonymous strict-end \Z, then freeze literals before Rust matcher. This is oracle tooling, not a Rust test failure/RED or fitted expected value.

## Paused interruption handoff — 2026-10-17

- [x] Coordinator relayed explicit user request to wrap/revalidate interruption and pause. Stopped implementation immediately after first new-file scaffold; no additional integration attempted.
- [x] Exact edited Rust path: crates/zerogravity-artifacts/src/acquisition/filters.rs NEW, only in zerogravity-filters-work. Other approved10 Rust paths (quant/selection/HF/GH/mod/Xet +API test and CLI3) remain supplied4db5efb+C1/C2 baseline; candidates.rs not edited. Root is qualifying C2 independently, whose future actual fixes must be synchronized before resuming C3.
- [x] Existing own MAIN run and disposable oracle.py were written before pause; no MAIN writes after pause. This own SIDE TODO copies their useful design/interruption context. No canonical/main source/dependency/lock/Git modifications by C3.
- [x] Frozen independent expected cases BEFORE Rust matcher implementation: CPython v3.14.0 Lib/fnmatch.py read as primary source and executed on installed Python3.13 with ONLY strict-end regex spelling \z -> synonymous \Z.73 literal class/Unicode/slash cases plus585 patterns *57 names =33345 pairs produce literal SHA256 c9f3b0c631f703ae44a4f2dafb5c8c71cb2715bfaa3249751cd299c73a40445f. Initial oracle setup failed on mistaken two-backslash replacement; corrected once before any expected output, then successful oracle command exit0. Expected values were not derived from new Rust or altered to fit it. Oracle is a disposable research generator, not a runtime dependency.
- [x] New file draft contains private fixed-work matcher with Unicode-scalar NFA rows, CPython-style class normalization, no recursive backtracking/regex/cache; FilterRequest constructors/getters, computed Exclusions and public resolve/cachedvalidator SCAFFOLDS. These references intentionally await private integration and make the file NOT COMPILE-READY. It is not wired from acquisition/mod.rs and establishes no filter support.
- [x] Read-only interruption comparison: all8 existing approved integration paths byte-match frozen C2 SIDE after basename repair; both new integration test paths are absent. New filters.rs is the only Rust delta.
- [x] Scoped rustfmt --edition2024 --config skip_children=true filters.rs exited0; own-file Python trailing-whitespace/final-newline check passed. No Cargo/rustc/test/feature checks and no execution of new Rust matcher, oracle assertions or imports. No active command/session/child process is retained.
- [ ] Resume by independently auditing matcher/class normalization and fixed oracle; check compiler/lint only after serialized target grant. Validate actual scratch bound with size_of and class/token worst case; additional adversarial work/cancel/input tests not yet written. Extra16MiB same-admission rationale currently an unverified design comment, not physical qualification.
- [ ] Add missing quant private seams validate_inventory/select_excluding/metadata_admission_extra; InitialChoice::Filtered; default-empty Exclusions behavior. Existing C2 stays unchanged until reconciled qualified baseline is available.
- [ ] Add filtered HF/GH entrypoints inside original one pin/tree owner; required veto and optional skip in BOTH original HF closure and shared selection closure; conservative excluded index refusal only where original closure would attempt reading it. Avoid blanket unrelated/GGUF index refusal. Preserve original Files and pure Quant behavior.
- [ ] Add thin xet::resolve_filtered and combined control admission before HTTP, preserving native >=1announcement/exact source/index boundary/actual budgets and no second runtime/publisher. New filters API currently references nonexistent seams, so do not wire the module prematurely or claim compile success.
- [ ] Implement offline exact final closed subset, source/current limits/roles/pointers/index budgets and digest-only CLI before credentials/client/admission; verify no excluded retained extras are presented as a filtered view. Scaffold uses shared pure closure/read_verified but remains uncompiled/unreviewed.
- [ ] Add CLI Action::Filtered/private FetchRequest extension, includes/excludes vs legacy --file/quant conflicts, literal subfolder suffix matching, same shared native/ordinary fetch defaults and completion. No CLI source edited yet.
- [ ] Author tests/acquisition_filters.rs and CLI tests/pull_filters.rs: include union/exclusion-before-ambiguity, quant complete/partial family, optional omission/required veto/index-before-body, source/budgets/pin equality, offline exact subset/pointer/role/source/limit/secrets, tiny RAM preHTTP/native defaults and unchanged legacy byte grammar. No integration fixture files exist yet.
- [ ] Independent cold review and all compile/tests/lints/full gate/source-integrity/model checks remain UNRUN and coordinator-owned after resume. No C3 support/parity/inference or native execution claim. Preserve SIDE and this unfinished run until recovered; no merge/commit/cleanup requested.

````

## Source: .agents/runs/serving-and-speculation-activation-before-after-20261020/README.md

Original UTF-8 bytes: 6743; SHA256 `5f3b66f11fdf8f069e02a31215987f6a450cd12879b37c32f8f5d3f1d972bab5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only same-engine before/after observations

Parent: `.agents/plans/serving-and-speculation/TODO.md`.
Owner: `/root/baseline_review`; root alone runs actual providers/GPU.

PREPARED ONLY. This agent executed no provider, model, GPU, benchmark, Cargo or runtime test. Python syntax/AST and source/hash inspection only. No production source changes.

Both providers ARE Zerogravity:

- A: root's preserved accepted B7 CUDA server, source checkpointdbb5f4f, actual binary SHA d7975d1da2eb7392df947a0034b646a1e091e7604a5d5deece22a75b277820c1.
- B: root's future separately frozen current CUDA server, actual SHA and source/build receipt ID REQUIRED at invocation. Filename never supplies provenance.

This driver does not call `serve-compare.py`, populate a vLLM slot or change shared measurement tools. A small initialization adapter gives existing `LifecyclePlan` honest same-engine labels; inherited `sample`/`OwnedProvider` startup/readiness/log/PGID/cleanup/receipt-I/O paths are reused unchanged. No global PROVIDERS mutation. Launches are direct bare-metal argv; there are no arbitrary stop commands or existing-process signals.

Exact measurement policy:

- Same approved original GGUF, sidecar config and tokenizer file hashes, same alias/prompt/workload/greedy policy for A/B. Both originals0.8B and2B, serialized per model. Separate process per ABBA or BAAB sample; defaultABBA is four samples per model. Repeat with BAAB separately as the counterbalanced control.
- CUDA, one worker, active8, native rows8, context512, output cap8(default) or16. Host/provider/device configured caps4096MiB. Four currently allowed logical CPUs via process affinity, Rayon/OMP/MKL/OpenBLAS4. No claim of an OS CPU quota or physical RAM admission from these settings.
- `warmed_process`: one unmeasured eight-offer/max1 wave, then measured eight-offer/max8(or16) wave. Actual warmup results/bands are retained; this does not promise every kernel/JIT/cache is warm. `cold_process`: fresh process with no warmup; OS/storage/device caches remain untouched and are not claimed cold.
- Eight actual offered requests, concurrency8/max-pending8, same bounded prompt, two alternating workload fairness labels (not authenticated tenants). Source EOS remains active; there is no ignore-EOS policy. Every measured request must complete with>=2 actual IDs, exact usage, complete commit metadata and individual token-ID delivery; early EOS/ambiguous delivery makes that run insufficient for this ITL comparison. Never relabel SSE chunk spacing as token ITL.
- Existing `serve-bench.py` supplies actual per-request metrics, token IDs/sequence, output hashes, TTFT/individual client-delivery ITL, separate provider commit-observation ITL, SLO observations and workload fairness. Full raw records remain in each original lifecycle measurement artifact and the final report.
- Quiescent health snapshots around measured work supply actual successful forward launch counts, model rows, output tokens and native B2/B4/B8 counters. Require native B8>0 and no failed/cancelled/admission-refused/discarded work. Projection counter remains final-dot subset; initialization/DMA/unknown-completion are excluded from these receipts.
- Queued, active and output-credit-parked request fields are current gauges: retain explicit before/after values, never subtract them as monotonic counters or infer zero parking incidents from zero endpoints. Full health snapshots remain available.
- Identical request IDs, prompt hash, token IDs/order, usage, output hash and finish reason across every A/B wave per model. Mismatch stops after owned cleanup; it never produces a successful performance comparison. This limited output check is not full-vocabulary/all-state numerical qualification.
- Process logs capped4MiB and continuously drained. Private0600 receipts use the existing lifecycle implementation. Startup≤240s, shutdown≤30s, request timeout≤180s, GPU-idle wait≤30s. Readiness performs bounded actual `/v1/models`+`/health` checks and verifies owned `/proc/PID/exe` against the frozen binary after readiness.
- This private driver installs a proxy-disabled, redirect-refusing urllib opener before benchmark threads start; benchmark/readiness/health traffic remains on the explicit loopback endpoints. Canonical measurement tools are unchanged.
- Before/after each sample: no existing listeners are adopted/stopped; no other compute app is permitted on caller-pinned GPU UUID. Read-only nvidia-smi/proc sampling starts before spawn; selected GPU must actually observe the owned server PID. Other compute PIDs cause refusal without touching them. Owned PGID/endpoints/drainers finish before next provider; post-cleanup GPU idle requires three no-compute observations, utilization≤5%, memory within initial idle+64MiB.
- Memory is raw sampled device/process observation: GPU device memory and owned process RSS/HWM. Sampling every.5s can miss peaks; provider health exposes no GPU budget-byte snapshot. Do not call this physical ResourceArbiter admission or exact allocator peak.

Only root's GPU coordinator may execute, after strict/physical/full gates for the fresh implementation. First freeze the freshly built binary to an owned stable path outside shared `target` output and retain its exact build/source receipt. Driver checks both binary hashes before each launch and after shutdown. Both measurement/lifecycle tool hashes are pinned; tool drift refuses execution until root reviews/repins.

```sh
# ROOT ONLY; values below must come from actual fresh build/NVIDIA receipts.
python3 .agents/runs/serving-and-speculation-activation-before-after-20261020/measure_same_engine.py \
  --run \
  --current-server /ABSOLUTE/OWNED/FROZEN/CURRENT-CUDA-SERVER \
  --current-sha256 ACTUAL_64_LOWERCASE_HEX_SHA256 \
  --current-source-id ACTUAL_SOURCE_OR_BUILD_RECEIPT_ID \
  --gpu-uuid ACTUAL_GPU_UUID \
  --models 0.8b --order ABBA --policy warmed_process --max-tokens 8
```

Inspect the first bounded0.8B ABBA report before2B or further controls. Then use `--models 2b`, BAAB, or explicit `--policy cold_process` as separate recorded experiments. Default `--models 0.8b,2b` is available but does not bypass per-sample cleanup/refusal.

Each invocation creates a unique private `actual-measurements/same-zg-*` directory. `report.json` retains every raw measurement, output identity, health delta and sampled memory/lifecycle receipt. Exit success means bounded same-Zerogravity observations only. No vLLM, full-model/source-correct SLO goodput or production eligibility claim is inferred, and no speed ratio is fabricated from kernel-count reduction.

Root independently corrected the CLI frontier wording during preparation. No help patch is needed or included here.

````

## Source: .agents/runs/serving-and-speculation-activation-before-after-20261020/TODO.md

Original UTF-8 bytes: 1857; SHA256 `ab59c72f38dbd000fc1ceb0b18b318e27f90cb1b92c35192f1cc7096d0dec7f8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Activation preparation before/after harness

Parent: `.agents/plans/serving-and-speculation/TODO.md`.
Owner: `/root/baseline_review`; no shared plan changes.

- [x] Read actual measurement/lifecycle modules and root's successful owned-provider launch script.
- [x] Prepare honest same-Zerogravity labels with unchanged underlying owned cleanup.
- [x] Pin accepted frozen server, future current server required actual SHA/source-ID, measurement tools, both original source/config/tokenizers.
- [x] Prepare ABBA/BAAB, fresh-process/warmup policy, eight actual offers/rows8/context512/4096MiB caps/four logical CPUs.
- [x] Preserve raw individual IDs/true client-delivery ITL/provider commit clocks, usage/output hash, fairness, health dispatch deltas and sampled GPU/process memory.
- [x] Cleanup-before-validation/receipt-I/O, selected GPU owned PID observation, no competing compute PIDs and idle-before-next policy.
- [x] Python syntax/AST only; no provider/GPU/model/test/Cargo/runtime execution.
- [x] Root review follow-up: parked requests are a current gauge, retained before/after alongside active/queued; no parking-incident delta. Original frozen-prepared bytes preserved.
- [x] Independent review follow-up: private-process urllib opener disables ambient proxies and redirects before benchmark threads; canonical tools unchanged.
- [ ] Root independently review adapter/cleanup and measurement claims before launch.
- [ ] Root current implementation strict/operator/full-model/cancel/reclaim/release/full gates; freeze actual current binary/build receipt.
- [ ] Root first bounded0.8B warmed ABBA/max8, actual tool/output/health/cleanup validation.
- [ ] Root2B and BAAB/cold-process controls as useful; no automatic speed/qualification promotion.
- [ ] Reconcile owning serving plan with actual measured scope; vLLM comparison remains separate.

````

## Source: .agents/runs/serving-and-speculation-activation-before-after-20261020/frozen-gauge-proxy-followup/README.md

Original UTF-8 bytes: 6743; SHA256 `5f3b66f11fdf8f069e02a31215987f6a450cd12879b37c32f8f5d3f1d972bab5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only same-engine before/after observations

Parent: `.agents/plans/serving-and-speculation/TODO.md`.
Owner: `/root/baseline_review`; root alone runs actual providers/GPU.

PREPARED ONLY. This agent executed no provider, model, GPU, benchmark, Cargo or runtime test. Python syntax/AST and source/hash inspection only. No production source changes.

Both providers ARE Zerogravity:

- A: root's preserved accepted B7 CUDA server, source checkpointdbb5f4f, actual binary SHA d7975d1da2eb7392df947a0034b646a1e091e7604a5d5deece22a75b277820c1.
- B: root's future separately frozen current CUDA server, actual SHA and source/build receipt ID REQUIRED at invocation. Filename never supplies provenance.

This driver does not call `serve-compare.py`, populate a vLLM slot or change shared measurement tools. A small initialization adapter gives existing `LifecyclePlan` honest same-engine labels; inherited `sample`/`OwnedProvider` startup/readiness/log/PGID/cleanup/receipt-I/O paths are reused unchanged. No global PROVIDERS mutation. Launches are direct bare-metal argv; there are no arbitrary stop commands or existing-process signals.

Exact measurement policy:

- Same approved original GGUF, sidecar config and tokenizer file hashes, same alias/prompt/workload/greedy policy for A/B. Both originals0.8B and2B, serialized per model. Separate process per ABBA or BAAB sample; defaultABBA is four samples per model. Repeat with BAAB separately as the counterbalanced control.
- CUDA, one worker, active8, native rows8, context512, output cap8(default) or16. Host/provider/device configured caps4096MiB. Four currently allowed logical CPUs via process affinity, Rayon/OMP/MKL/OpenBLAS4. No claim of an OS CPU quota or physical RAM admission from these settings.
- `warmed_process`: one unmeasured eight-offer/max1 wave, then measured eight-offer/max8(or16) wave. Actual warmup results/bands are retained; this does not promise every kernel/JIT/cache is warm. `cold_process`: fresh process with no warmup; OS/storage/device caches remain untouched and are not claimed cold.
- Eight actual offered requests, concurrency8/max-pending8, same bounded prompt, two alternating workload fairness labels (not authenticated tenants). Source EOS remains active; there is no ignore-EOS policy. Every measured request must complete with>=2 actual IDs, exact usage, complete commit metadata and individual token-ID delivery; early EOS/ambiguous delivery makes that run insufficient for this ITL comparison. Never relabel SSE chunk spacing as token ITL.
- Existing `serve-bench.py` supplies actual per-request metrics, token IDs/sequence, output hashes, TTFT/individual client-delivery ITL, separate provider commit-observation ITL, SLO observations and workload fairness. Full raw records remain in each original lifecycle measurement artifact and the final report.
- Quiescent health snapshots around measured work supply actual successful forward launch counts, model rows, output tokens and native B2/B4/B8 counters. Require native B8>0 and no failed/cancelled/admission-refused/discarded work. Projection counter remains final-dot subset; initialization/DMA/unknown-completion are excluded from these receipts.
- Queued, active and output-credit-parked request fields are current gauges: retain explicit before/after values, never subtract them as monotonic counters or infer zero parking incidents from zero endpoints. Full health snapshots remain available.
- Identical request IDs, prompt hash, token IDs/order, usage, output hash and finish reason across every A/B wave per model. Mismatch stops after owned cleanup; it never produces a successful performance comparison. This limited output check is not full-vocabulary/all-state numerical qualification.
- Process logs capped4MiB and continuously drained. Private0600 receipts use the existing lifecycle implementation. Startup≤240s, shutdown≤30s, request timeout≤180s, GPU-idle wait≤30s. Readiness performs bounded actual `/v1/models`+`/health` checks and verifies owned `/proc/PID/exe` against the frozen binary after readiness.
- This private driver installs a proxy-disabled, redirect-refusing urllib opener before benchmark threads start; benchmark/readiness/health traffic remains on the explicit loopback endpoints. Canonical measurement tools are unchanged.
- Before/after each sample: no existing listeners are adopted/stopped; no other compute app is permitted on caller-pinned GPU UUID. Read-only nvidia-smi/proc sampling starts before spawn; selected GPU must actually observe the owned server PID. Other compute PIDs cause refusal without touching them. Owned PGID/endpoints/drainers finish before next provider; post-cleanup GPU idle requires three no-compute observations, utilization≤5%, memory within initial idle+64MiB.
- Memory is raw sampled device/process observation: GPU device memory and owned process RSS/HWM. Sampling every.5s can miss peaks; provider health exposes no GPU budget-byte snapshot. Do not call this physical ResourceArbiter admission or exact allocator peak.

Only root's GPU coordinator may execute, after strict/physical/full gates for the fresh implementation. First freeze the freshly built binary to an owned stable path outside shared `target` output and retain its exact build/source receipt. Driver checks both binary hashes before each launch and after shutdown. Both measurement/lifecycle tool hashes are pinned; tool drift refuses execution until root reviews/repins.

```sh
# ROOT ONLY; values below must come from actual fresh build/NVIDIA receipts.
python3 .agents/runs/serving-and-speculation-activation-before-after-20261020/measure_same_engine.py \
  --run \
  --current-server /ABSOLUTE/OWNED/FROZEN/CURRENT-CUDA-SERVER \
  --current-sha256 ACTUAL_64_LOWERCASE_HEX_SHA256 \
  --current-source-id ACTUAL_SOURCE_OR_BUILD_RECEIPT_ID \
  --gpu-uuid ACTUAL_GPU_UUID \
  --models 0.8b --order ABBA --policy warmed_process --max-tokens 8
```

Inspect the first bounded0.8B ABBA report before2B or further controls. Then use `--models 2b`, BAAB, or explicit `--policy cold_process` as separate recorded experiments. Default `--models 0.8b,2b` is available but does not bypass per-sample cleanup/refusal.

Each invocation creates a unique private `actual-measurements/same-zg-*` directory. `report.json` retains every raw measurement, output identity, health delta and sampled memory/lifecycle receipt. Exit success means bounded same-Zerogravity observations only. No vLLM, full-model/source-correct SLO goodput or production eligibility claim is inferred, and no speed ratio is fabricated from kernel-count reduction.

Root independently corrected the CLI frontier wording during preparation. No help patch is needed or included here.

````

## Source: .agents/runs/serving-and-speculation-activation-before-after-20261020/frozen-gauge-proxy-followup/TODO.md

Original UTF-8 bytes: 1857; SHA256 `ab59c72f38dbd000fc1ceb0b18b318e27f90cb1b92c35192f1cc7096d0dec7f8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Activation preparation before/after harness

Parent: `.agents/plans/serving-and-speculation/TODO.md`.
Owner: `/root/baseline_review`; no shared plan changes.

- [x] Read actual measurement/lifecycle modules and root's successful owned-provider launch script.
- [x] Prepare honest same-Zerogravity labels with unchanged underlying owned cleanup.
- [x] Pin accepted frozen server, future current server required actual SHA/source-ID, measurement tools, both original source/config/tokenizers.
- [x] Prepare ABBA/BAAB, fresh-process/warmup policy, eight actual offers/rows8/context512/4096MiB caps/four logical CPUs.
- [x] Preserve raw individual IDs/true client-delivery ITL/provider commit clocks, usage/output hash, fairness, health dispatch deltas and sampled GPU/process memory.
- [x] Cleanup-before-validation/receipt-I/O, selected GPU owned PID observation, no competing compute PIDs and idle-before-next policy.
- [x] Python syntax/AST only; no provider/GPU/model/test/Cargo/runtime execution.
- [x] Root review follow-up: parked requests are a current gauge, retained before/after alongside active/queued; no parking-incident delta. Original frozen-prepared bytes preserved.
- [x] Independent review follow-up: private-process urllib opener disables ambient proxies and redirects before benchmark threads; canonical tools unchanged.
- [ ] Root independently review adapter/cleanup and measurement claims before launch.
- [ ] Root current implementation strict/operator/full-model/cancel/reclaim/release/full gates; freeze actual current binary/build receipt.
- [ ] Root first bounded0.8B warmed ABBA/max8, actual tool/output/health/cleanup validation.
- [ ] Root2B and BAAB/cold-process controls as useful; no automatic speed/qualification promotion.
- [ ] Reconcile owning serving plan with actual measured scope; vLLM comparison remains separate.

````

## Source: .agents/runs/serving-and-speculation-activation-before-after-20261020/frozen-prepared/README.md

Original UTF-8 bytes: 6259; SHA256 `33def35ede5d184c7a2fe7f271d3554e24158482d5148cd4418861c3039c31d1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Root-only same-engine before/after observations

Parent: `.agents/plans/serving-and-speculation/TODO.md`.
Owner: `/root/baseline_review`; root alone runs actual providers/GPU.

PREPARED ONLY. This agent executed no provider, model, GPU, benchmark, Cargo or runtime test. Python syntax/AST and source/hash inspection only. No production source changes.

Both providers ARE Zerogravity:

- A: root's preserved accepted B7 CUDA server, source checkpointdbb5f4f, actual binary SHA d7975d1da2eb7392df947a0034b646a1e091e7604a5d5deece22a75b277820c1.
- B: root's future separately frozen current CUDA server, actual SHA and source/build receipt ID REQUIRED at invocation. Filename never supplies provenance.

This driver does not call `serve-compare.py`, populate a vLLM slot or change shared measurement tools. A small initialization adapter gives existing `LifecyclePlan` honest same-engine labels; inherited `sample`/`OwnedProvider` startup/readiness/log/PGID/cleanup/receipt-I/O paths are reused unchanged. No global PROVIDERS mutation. Launches are direct bare-metal argv; there are no arbitrary stop commands or existing-process signals.

Exact measurement policy:

- Same approved original GGUF, sidecar config and tokenizer file hashes, same alias/prompt/workload/greedy policy for A/B. Both originals0.8B and2B, serialized per model. Separate process per ABBA or BAAB sample; defaultABBA is four samples per model. Repeat with BAAB separately as the counterbalanced control.
- CUDA, one worker, active8, native rows8, context512, output cap8(default) or16. Host/provider/device configured caps4096MiB. Four currently allowed logical CPUs via process affinity, Rayon/OMP/MKL/OpenBLAS4. No claim of an OS CPU quota or physical RAM admission from these settings.
- `warmed_process`: one unmeasured eight-offer/max1 wave, then measured eight-offer/max8(or16) wave. Actual warmup results/bands are retained; this does not promise every kernel/JIT/cache is warm. `cold_process`: fresh process with no warmup; OS/storage/device caches remain untouched and are not claimed cold.
- Eight actual offered requests, concurrency8/max-pending8, same bounded prompt, two alternating workload fairness labels (not authenticated tenants). Source EOS remains active; there is no ignore-EOS policy. Every measured request must complete with>=2 actual IDs, exact usage, complete commit metadata and individual token-ID delivery; early EOS/ambiguous delivery makes that run insufficient for this ITL comparison. Never relabel SSE chunk spacing as token ITL.
- Existing `serve-bench.py` supplies actual per-request metrics, token IDs/sequence, output hashes, TTFT/individual client-delivery ITL, separate provider commit-observation ITL, SLO observations and workload fairness. Full raw records remain in each original lifecycle measurement artifact and the final report.
- Quiescent health snapshots around measured work supply actual successful forward launch counts, model rows, output tokens and native B2/B4/B8 counters. Require native B8>0 and no failed/cancelled/admission-refused/discarded work. Projection counter remains final-dot subset; initialization/DMA/unknown-completion are excluded from these receipts.
- Identical request IDs, prompt hash, token IDs/order, usage, output hash and finish reason across every A/B wave per model. Mismatch stops after owned cleanup; it never produces a successful performance comparison. This limited output check is not full-vocabulary/all-state numerical qualification.
- Process logs capped4MiB and continuously drained. Private0600 receipts use the existing lifecycle implementation. Startup≤240s, shutdown≤30s, request timeout≤180s, GPU-idle wait≤30s. Readiness performs bounded actual `/v1/models`+`/health` checks and verifies owned `/proc/PID/exe` against the frozen binary after readiness.
- Before/after each sample: no existing listeners are adopted/stopped; no other compute app is permitted on caller-pinned GPU UUID. Read-only nvidia-smi/proc sampling starts before spawn; selected GPU must actually observe the owned server PID. Other compute PIDs cause refusal without touching them. Owned PGID/endpoints/drainers finish before next provider; post-cleanup GPU idle requires three no-compute observations, utilization≤5%, memory within initial idle+64MiB.
- Memory is raw sampled device/process observation: GPU device memory and owned process RSS/HWM. Sampling every.5s can miss peaks; provider health exposes no GPU budget-byte snapshot. Do not call this physical ResourceArbiter admission or exact allocator peak.

Only root's GPU coordinator may execute, after strict/physical/full gates for the fresh implementation. First freeze the freshly built binary to an owned stable path outside shared `target` output and retain its exact build/source receipt. Driver checks both binary hashes before each launch and after shutdown. Both measurement/lifecycle tool hashes are pinned; tool drift refuses execution until root reviews/repins.

```sh
# ROOT ONLY; values below must come from actual fresh build/NVIDIA receipts.
python3 .agents/runs/serving-and-speculation-activation-before-after-20261020/measure_same_engine.py \
  --run \
  --current-server /ABSOLUTE/OWNED/FROZEN/CURRENT-CUDA-SERVER \
  --current-sha256 ACTUAL_64_LOWERCASE_HEX_SHA256 \
  --current-source-id ACTUAL_SOURCE_OR_BUILD_RECEIPT_ID \
  --gpu-uuid ACTUAL_GPU_UUID \
  --models 0.8b --order ABBA --policy warmed_process --max-tokens 8
```

Inspect the first bounded0.8B ABBA report before2B or further controls. Then use `--models 2b`, BAAB, or explicit `--policy cold_process` as separate recorded experiments. Default `--models 0.8b,2b` is available but does not bypass per-sample cleanup/refusal.

Each invocation creates a unique private `actual-measurements/same-zg-*` directory. `report.json` retains every raw measurement, output identity, health delta and sampled memory/lifecycle receipt. Exit success means bounded same-Zerogravity observations only. No vLLM, full-model/source-correct SLO goodput or production eligibility claim is inferred, and no speed ratio is fabricated from kernel-count reduction.

Root independently corrected the CLI frontier wording during preparation. No help patch is needed or included here.

````

## Source: .agents/runs/serving-and-speculation-activation-before-after-20261020/frozen-prepared/TODO.md

Original UTF-8 bytes: 1517; SHA256 `f659b9474a4efd48f74daf62754a86e56d07039144ed23c5721f2b450731e035`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Activation preparation before/after harness

Parent: `.agents/plans/serving-and-speculation/TODO.md`.
Owner: `/root/baseline_review`; no shared plan changes.

- [x] Read actual measurement/lifecycle modules and root's successful owned-provider launch script.
- [x] Prepare honest same-Zerogravity labels with unchanged underlying owned cleanup.
- [x] Pin accepted frozen server, future current server required actual SHA/source-ID, measurement tools, both original source/config/tokenizers.
- [x] Prepare ABBA/BAAB, fresh-process/warmup policy, eight actual offers/rows8/context512/4096MiB caps/four logical CPUs.
- [x] Preserve raw individual IDs/true client-delivery ITL/provider commit clocks, usage/output hash, fairness, health dispatch deltas and sampled GPU/process memory.
- [x] Cleanup-before-validation/receipt-I/O, selected GPU owned PID observation, no competing compute PIDs and idle-before-next policy.
- [x] Python syntax/AST only; no provider/GPU/model/test/Cargo/runtime execution.
- [ ] Root independently review adapter/cleanup and measurement claims before launch.
- [ ] Root current implementation strict/operator/full-model/cancel/reclaim/release/full gates; freeze actual current binary/build receipt.
- [ ] Root first bounded0.8B warmed ABBA/max8, actual tool/output/health/cleanup validation.
- [ ] Root2B and BAAB/cold-process controls as useful; no automatic speed/qualification promotion.
- [ ] Reconcile owning serving plan with actual measured scope; vLLM comparison remains separate.

````

## Source: .agents/runs/serving-and-speculation-cuda-capture-exclusion-spine-20261020/CALLERS.md

Original UTF-8 bytes: 6101; SHA256 `85997c67259a624bbe61cb76fb2da0eba0bcdb2ac9e2a9aa8f8c34dfd15ff91b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Required integration map — source, not participation proof

Owner: `/root/spine_review`. This list is an implementation audit, not a new
production issuer. Every row remains unwired by this pure host-only slice.
Registry crate paths below are the actual pinned 0.11.0 source under
`/home/pvl/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`.

| Required boundary | Current source anchors | Necessary participation |
|---|---|---|
| Physical fair work | `zerogravity-gpu/src/physical_queue.rs::submit/poll_acquire/WaitingTicket/ActiveSlot` | Normal owned work begins before allocation/init; capture uses this same RR ticket/epoch, no priority queue/second byte ledger |
| Provider construction | `resident_leases.rs::ResidentService::new/new_pinned_cuda`; `resident_inference.rs` capability query through `device.client()` | Constructor permit before first runtime creation; bind actual physical key and original actor/stream; refuse if no complete gate proof |
| Candidate/packet/source/init | `ResidentService::allocate_uninitialized`; `ResidentInitialization::{allocate_empty,upload_bytes,complete_sync}`; `resident_inference.rs::allocate_state`; fused pre-ticket allocations | Work/command permit covers empty/create/write/sync before enqueue, including failure and readback future lifetime |
| Lease/cache release | `resident_leases.rs::LeaseInner::drop`; `resident_provider.rs::ResidentWeightLease::drop` | Last original handle and provider Client cannot be released into capture; pre-admitted bounded deferred ownership, original quarantine on unknown |
| Generic transport | `cubecl-common/src/device/handle/{base,channel,mutex,reentrant}.rs`: submit, submit_blocking, send/run_scoped, exclusive | Admit before closure enqueue; queued closure retains permit; actor never waits on gate; reentrancy requires exact owned epoch scope |
| Service creation/shutdown | `cubecl-common/src/device/handle/channel.rs::ChannelDeviceState::init/DeviceRunner/shutdown_device`; corresponding mutex service lifecycle | Bootstrap permit before service constructor; no new same-card service races Closing; final service/runner state teardown is admitted/deferred |
| Client/graph/profile | `cubecl-runtime/src/client.rs`: init/load, get_resource, create/empty/staging/write, launch/check/flush, graph lifecycle, sync, profiles, memory cleanup/report, collectives; `GraphHandle::drop:151` | All device-touching and implicit allocation/destruction calls participate; final GraphHandle drop is not a pure refcount operation |
| CUDA server entry | vendored `compute/server.rs`: staging105, read118, initialize_memory138, write158, launch, check243, flush251, graph_prepare261, begin266, end271, replay293, destroy311, sync343, profiles360/367/385, memory391/404/409, collectives420/429/466/478/513, get_resource1357 | A pre-enqueue token is required; gate before compile_failed/implicit stream, metadata buffer, pool relocation and fallback operations, not merely command() |
| CUDA bootstrap | vendored `runtime.rs::DeviceService::init:76`, DeviceProbe106, primary context retain, comm stream391, server416; `compute/server.rs::new:562` | Conservative bootstrap gate before context/stream creation; seal reported PCI identity, then same-card gate; ordinal is not a physical key |
| CUDA events outside actor | vendored `compute/events.rs::Cuda` all EventApi methods; `cubecl-server/src/device_events/{event,fence,profiler}.rs` | Originating card/gate attached to each event/fence/profile owner; wait/destroy/elapsed/fallback sync participate even on foreign host/GC thread |
| Async read/sync/GC | `cubecl-server/src/command/base.rs::read_async:334`; `stream/event.rs::ResolvedStreams::drop:249`; drop_queue worker | Copies are enqueued before returned future; future waits/destroys events outside actor; drain includes original retained source/destination/fence and posted GC |
| Allocation/free/relocation | vendored `compute/storage/{gpu,cpu}.rs`, `compute/driver.rs`, `compute/stream.rs`; common memory managers/drop queues | Every alloc/free (including sync fallback), host-pinned allocation, copy/relocation, stream/event creation and shutdown must be admitted or deferred |
| CUDA graph lifetime | vendored `compute/capture.rs::Executable::drop`; server graph_destroy synchronizes before destroy; original `CapturedResidentGraph` quarantine/retire | Preserve executable+band+service leases across all unknowns; no driver destruction in unadmitted Drop; keep original two-barrier explicit retirement |
| Original owned CUDA commands | accepted pinned-queue/owned_command adapter | Same original actor/stream and gate; adding an owned command must not create an ungated alternative entry |
| Same-card other runtimes | Vulkan/HIP/common clients sharing reported PCI; custom DeviceHandle services | Participate in common command/lifecycle contract or production capability refuses cold; a CUDA-only registry cannot prove their exclusion |

Pure cached Client properties/features/target metadata need no command permit.
Queries that lazily construct a Client, stream, pool or profile are not pure.
The scope excludes external processes/foreign runtimes; qualification must own
that environmental exclusion. No discovery scan is being called proof here.

Suggested disjoint implementation ownership after this protocol review:

1. GPU coordinator: physical_queue/physical_service and resident work ordering;
   original ResourceArbiter/fence bridge, existing all42/Ready/lease publisher.
2. Common/vendor transport owner: explicit shared generic command/lifecycle gate
   contracts, pre-enqueue ownership, bootstrap/shutdown and exclusive scope.
3. CUDA lifecycle owner: existing physical identity, events/fences, storage,
   graph/stream teardown and bounded original-owner deferral.
4. Independent qualifier: deterministic protocol negatives, then root-only
   actual two-model/two-actor same-card construction/alloc/drop/cancel capture.

No single owner may issue Complete before the other paths are source-closed and
physically qualified. No module should acquire another scheduler or VRAM ledger.

````

## Source: .agents/runs/serving-and-speculation-cuda-capture-exclusion-spine-20261020/DESIGN.md

Original UTF-8 bytes: 7852; SHA256 `6111e5b9f2a682c48a8b498924190c6ac3395de032ab1ec88c5affa9632993b1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Complete per-card capture exclusion: minimal honest boundary

Author: `/root/spine_review`; source-only proposal, 2026-10-20.
Production graphs remain closed. Initial graph is a caller-isolated qualifier.

## Finding

PhysicalQueue serializes one selected inference region per reported physical
card, and PhysicalWork retains/fences the original provider envelope. Neither
prevents device commands outside that region. Candidates/packets are allocated
before queue selection; initialization helpers do not require an ActiveSlot;
raw Client methods are unrestricted. CUDA fences can wait/destroy events on
other threads. Graph final drop posts destruction, whose server method first
synchronizes. Server construction creates a primary context and communication
stream before the GPU PhysicalQueue can be attached.

A CUDA-Server-method mutex is insufficient. Blocking there can park the actor
behind a capture-close operation. Refusing there can fail a normal command that
was already asynchronously enqueued. The admission decision and owned command
permit must precede enqueue; its lifetime must include the queued closure.
`Client::exclusive` is reentrant actor serialization, not a physical-card lock.

## Required layers

1. Keep existing fair PhysicalQueue and original prepaid ResourceArbiter. Add a
   capture request as a typed fair work item, not a priority bypass. Queue wait
   occurs before model allocation/workspace locks or any device command. A normal
   owned WorkPermit spans allocation/init/gather/launch/readback/terminal event;
   early drop after possible submission preserves original fence/quarantine.
2. One process-shared command gate per canonical reported physical identity.
   Do not key by CUDA ordinal, DeviceId, ServiceId, source or model. Vendor-level
   registration is needed before primary-context retain/stream construction;
   a short global bootstrap admission closes the unknown-card discovery race.
   Bootstrap releases only after an actual reported PCI/LUID binding is sealed.
3. Common transport admission before every `submit`, `submit_blocking`, scoped
   `exclusive`, service creation and shutdown. A command owns its permit through
   actor execution and any returned device-touching future. Normal pre-close
   queued closures finish; post-close external callers park/cancel outside actor.
   Never block a CUDA actor, destructor, executor poll or gate mutex on this gate.
4. CUDA event/storage/stream/graph lifecycle participates even outside transport.
   Events carry their originating gate/card owner. Wait/elapsed/destruction and
   GC/drop-queue work cannot silently call the driver while capture is active.
   Keep original buffers/events/graph/actor owners alive when deferring teardown.
5. Exclusive owned CaptureEpoch enters only after admission is closed, all
   admitted command permits and deferred work are drained, and each registered
   same-card actor has a trusted terminal stream barrier. No lock guard crosses
   await, a driver wait, actor call or Drop. A typed token holds the epoch instead.
6. Enter original Client::exclusive synchronously on the pinned actor/stream.
   Privately install the exact epoch/actor/stream scope for reentrant body calls.
   Only reviewed capture lifecycle/hot-body methods may use it; the token is
   not a generic allocation/sync/destruction bypass. No await inside that scope.
7. Stop/end capture, retain the original executable/leases in original flight
   owners, then leave actor scope. Resume queued/deferred destruction only after
   the epoch has a trusted closed result. Original all42 roots, Ready, publisher,
   byte accounting and kernel counts remain unchanged.

## Owned-state protocol

State: Open(epoch) -> Closing(ticket,epoch) -> Draining -> Capturing ->
Retiring -> Open(next_epoch), or Fenced. All transitions use short host locks.

- Capture request must win the existing fair queue; a stream of new ordinary
  requests cannot pass a pending capture forever. Ordinary command permits
  admitted before Closing remain valid and counted until executed/retired.
- Closing rejects no already-admitted native task. New async requests park with
  bounded tickets/wakers; a queued cancellation removes only that request.
- Capture token is private/noncloneable. A borrowed actor scope is derived only
  on its original actor, bound to card/service/stream/epoch and reviewed op kinds.
  Token/receipt from another card/service/epoch is a cold error, no device work.
- Cancelling before any capture/device effect reopens admission after draining
  its own bookkeeping. No GPU fence is invented for a never-submitted ticket.
- Cancelling after start/unknown stop retains every graph/workspace/source,
  event, pending destructor and original Client/actor owner; fence both gate and
  original ResourceArbiter card. Never clear by timeout or synchronizing in Drop.
- Gate/module mutex guards are never stored in permits. No actor callback or
  destructor is invoked while holding the state lock. Wakers run after unlock.
- Deferred destructor transport must be bounded and admitted before a resource
  becomes droppable. If its slot cannot be proved, refuse capture cold; do not
  drop the resource, allocate an unbounded closure queue, or block its destructor.
- A destructor originating during Capturing transfers its original resource and
  actor keepalive into its pre-admitted slot. It must not call graph_destroy,
  stream/event destruction or release the final actor. Replay Ready does not
  alone authorize executable destruction; keep the existing two-barrier retire.

## Why the narrow three-file edit cannot enable production

`cubecl-common` owns asynchronous/scoped transport and service shutdown;
`cubecl-runtime` owns Client and GraphHandle drop; `cubecl-server` owns events,
GC/drop workers and stream storage teardown. These are current registry crates,
not this repository's vendored CUDA crate. CUDA events carry no gate/card field.
A physical_queue/new GPU module/CudaServer-only change cannot intercept all
these entry points. The minimum complete integration needs explicit common
transport and lifecycle contracts (vendored/patch-owned source), plus CUDA
bootstrap/event/driver plumbing. Source-compatibility/default paths may stay
unchanged, but an unproven participant must make capture capability unavailable.

Cross-backend actors reporting the same PCI card must participate or make
production capture refuse cold. The present Vulkan/HIP/common Client paths do
not establish that participation. A PCI queue is not proof that every raw actor
or same-process external CUDA user is covered. External processes and foreign
CUDA runtimes remain outside this in-process contract; do not claim to exclude
those by a Rust permit. Physical qualification must explicitly control them.

## Smallest next implementation slice

After root agrees ownership, implement a pure per-card gate/state module and
opaque traits in the vendor common transport boundary, with a default-closed
`CaptureParticipation` capability. No actual capture method accepts a token
until every required hook is installed. Share the same gate authority with
PhysicalQueue; avoid two independently ordered schedulers or a second byte
ledger. Tests first cover owned tickets, phase/epoch/actor binding, fairness,
cancellation before effect, fence/unknown retention and bounded deferred drops.
This is preparatory infrastructure, not production graph support.

Do not patch only launches or release a command permit immediately after host
enqueue. Both leave actual posted device work/destruction outside the boundary.
Do not hold a permit for every buffer's entire lifetime: the intended graph's
own resident source/workspace buffers are live, and capture could never start.
Gate their device-affecting use/drop, preserving original physical owners.

````

## Source: .agents/runs/serving-and-speculation-cuda-capture-exclusion-spine-20261020/README.md

Original UTF-8 bytes: 3617; SHA256 `c7350e2dc18f7a02763326bafcf266c6d0a48a483b7d5aa50d70b98f35d7c794`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pure physical-queue capture protocol proposal

Owner: `/root/spine_review`. Ignored preparatory slice only. No MAIN or graph
worktree tracked edits; no provider discovery, actual capture or math changes.

Two proposed source paths:

- `crates/zerogravity-gpu/src/physical_queue.rs`: only seven cfg(test) lines add
  a child module/field/initializer to its actual existing QueueState.
- `crates/zerogravity-gpu/src/physical_queue/capture.rs`: host protocol using that
  same queue mutex, original FairDeviceQueue ticket/ActiveSlot and process-shared
  ResourceArbiter physical key/fence. No registry, second scheduler or byte ledger.

The original default/runtime file projection is byte-identical when test-only
insertions are removed. No production completeness issuer exists. Participation
is CLOSED by default; synthetic complete issuance lives only inside host tests.
No existing command/constructor/drop is protected by this unwired prototype.
Production graph capability must remain unavailable until CALLERS.md is closed.

Protocol: pre-enqueue shared command permits -> fair original capture ticket ->
close new admission -> drain all earlier admitted command permits -> move-only
capture epoch -> exclusive borrowed actor scope -> ordered Begin/Body/End ->
trusted terminal -> original fair slot retirement. Every token binds actual
queue Arc, physical key, queue epoch and exact service/stream metadata. Guards
never cross await/driver/actor calls. Wakers run outside the queue mutex.

Queued/selected/closing cancellation before effects reopens without a device
fence. A started unknown command or capture keeps the gate fenced and calls the
same original ResourceArbiter fence. Actual device owners/quarantine remain the
existing ResidentSubmission/lease protocol; this slice does not implement those
hooks or claim that a host token is physical Ready. Deferred destructor slots,
bootstrap registration, actual actor scope installation and full participation
proof are intentionally unimplemented; see DESIGN.md and CALLERS.md.

13 deterministic host tests are PREPARED, NOT RUN: default/foreign completeness;
interleaved pre-close commands/drain; original RR fairness; queued/unstarted and
drain cancellation; unknown command/capture fence; stale epoch refusals; foreign
equal-numbered ticket/slot; epoch change after start and after known terminal;
unlocked wake; bounded permits/phase order and owned token consumption. These
use synthetic host-only physical keys and actual existing queue/arbiter policy,
with no CUDA client or invented physical memory claim.

`noncloneable-negative.rs` and `exclusive-borrow-negative.rs` are separate ROOT
compile-negative snippets for insertion INSIDE capture::tests in a fresh isolated
checkout. Require exactly E0599 at clone and E0499 at the overlapping mutable
borrow; do not treat unrelated resolution/privacy errors as a pass. Normal test
source deliberately does not include these failing snippets.

ROOT, after full source review and merging only into an isolated check tree,
can run the focused existing crate test gate:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 cargo test -p zerogravity-gpu physical_queue::capture::tests -- --test-threads=1
```

No source-only result marks production graph exclusion, metadata admission,
full inference or performance qualified. Parent alone owns wider integration,
strict/full checks and actual same-card multi-actor/constructor/drop/abort proof.

Author: rustfmt parser/format only; all13 tests and both negative snippets remain
unexecuted. Source/patch hashes are small local files, no model/DSO scan/runtime.

````

## Source: .agents/runs/serving-and-speculation-cuda-capture-exclusion-spine-20261020/TODO.md

Original UTF-8 bytes: 1311; SHA256 `e61c2c54e3a0258855ff2d8f08789d275b2be83b6f672608c024113a0b3fcdea`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Owned capture-exclusion preparation

Parent: `.agents/plans/serving-and-speculation/TODO.md` P12.
Owner: `/root/spine_review`. Ignored source-only proposal; MAIN/graph tracked
sources untouched. Root alone builds/tests/runs and reconciles the parent plan.

- [x] Read current physical queue, resident construction/leases, CUDA actor,
  runtime bootstrap, graph lifecycle, common channel/exclusive and event paths.
- [x] Identify concrete pre-queue allocations, enqueue, outside-actor event,
  deferred-drop/teardown and cross-backend participation gaps.
- [x] Specify ownership/order/cancellation/fence protocol and required callers.
- [x] Root approved pure protocol scope; wider vendor/common integration stays open.
- [x] Prepare first pure owned-token/fair capture state-machine slice;
  default-closed production participation capability, no capture enablement.
- [x] Prepare 13 host tests + two compile-negative ownership snippets, NOT RUN.
- [ ] Root review/compile/run pure slice.
- [ ] Wire every caller and issue actual complete participation only after proof.
- [ ] Root strict/full and actual multi-actor/card capture/abort/teardown gates.

No compilation, fixtures, native/operator/model/GPU/container/debugger runtime,
tracked source edits or large payload hashes performed by this preparation.

````

## Source: .agents/runs/serving-and-speculation-cuda-capture-exclusion-spine-20261020/followup-v2/README_V2.md

Original UTF-8 bytes: 2693; SHA256 `94ac2844a3fbc38c371bd88a6c046069c979167a85b6e1f654c396fc8e255292`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive host-protocol drain liveness repair

Owner: `/root/spine_review`; original315b84b3/c9f7c034 and all11 original files
remain immutable. This followup is source-only; no MAIN/runtime/graph changes.

Independent Batch review found two modeled issues:

- Original `advance_epoch` woke fair queue waiters but not the selected capture
  drain's waker. Revocation could leave a pending drain asleep until an unrelated
  command retired. A cfg(test) hook now takes that waker into the ORIGINAL wake
  vector while holding the original state lock. Existing wake iteration remains
  outside that lock. No second scheduler, queue or epoch authority.
- A single stored waker was inconsistent with `&self` drain methods: multiple
  futures could overwrite it. Both poll_drained and wait_drained now require
  `&mut self`, enforcing one drainer by the token's Rust borrow. Existing tests
  only gain the required mutable bindings; no behavior assertion is relaxed.

The new real Count-waker fixture leaves an admitted command unretired, polls its
capture drain Pending, revokes the original queue epoch, requires exactly one
wake and a repoll returning stale Err BEFORE command retirement. The Wake checks
that the original queue mutex is unlocked. Unstarted cancellation remains unfenced.
There are now14 prepared deterministic host tests, not executed by the author.

`single-drain-negative.rs` is a separate ROOT compile-negative snippet. Include
inside capture::tests in a fresh isolated check tree and require E0499 at the
second wait_drained borrow. The original move-only epoch/actor-scope negative
snippets remain immutable. Do not accept unrelated errors as negative evidence.

Binding wording correction to the original README: CommandPermit binds the
original queue Arc/physical key/epoch. CaptureTicket/Epoch additionally bind
caller-supplied ServiceId/stream metadata. Those fields and ActorScope do NOT
prove actual actor execution; the original Client::exclusive/vendor hook remains
unimplemented. Complete participation has no production issuer; production
capture remains CLOSED. No deferred GPU-owner or actual Ready proof is added.

Apply DRAIN_LIVENESS_FOLLOWUP.patch only after original PURE_PROTOCOL.patch in
an isolated tree. Proposed target file copies are under followup-v2/proposal.
All new code remains under the parent's cfg(test); removing test-only insertions
still gives the byte-identical original runtime physical_queue.rs.

Root alone reviews/compiles/runs the14 host fixtures and ownership negatives.
Author performed rustfmt parser/format and small source/hash projection checks
only. No Cargo, tests, GPU/provider/operator/model/container/debugger execution.

````

## Source: .agents/runs/serving-and-speculation-cuda-capture-exclusion-spine-20261020/followup-v2/TODO_V2.md

Original UTF-8 bytes: 614; SHA256 `6af16194b2ed3e65d57ca66fd71d2f9189246b89eec0827747c2601174c474d5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Owned additive drain repair

Parent: existing P12 pure capture-exclusion preparation. Owner: spine_review.

- [x] Preserve original315b/c9f freeze/all11 files.
- [x] Wake selected capture drain on epoch change through original unlocked loop.
- [x] Encode a single drainer using mutable token borrow.
- [x] Prepare real revocation Count-waker test + separate E0499 negative.
- [x] Correct command-token vs caller-supplied capture actor metadata scope.
- [ ] Root independent review/compile/14test execution; no author execution.
- [ ] All vendor/common/runtime hooks and actual production capability remain open.

````

## Source: .agents/runs/serving-and-speculation-cuda-capture-exclusion-spine-20261020/followup-v2/frozen-v2/README_V2.md

Original UTF-8 bytes: 2693; SHA256 `94ac2844a3fbc38c371bd88a6c046069c979167a85b6e1f654c396fc8e255292`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive host-protocol drain liveness repair

Owner: `/root/spine_review`; original315b84b3/c9f7c034 and all11 original files
remain immutable. This followup is source-only; no MAIN/runtime/graph changes.

Independent Batch review found two modeled issues:

- Original `advance_epoch` woke fair queue waiters but not the selected capture
  drain's waker. Revocation could leave a pending drain asleep until an unrelated
  command retired. A cfg(test) hook now takes that waker into the ORIGINAL wake
  vector while holding the original state lock. Existing wake iteration remains
  outside that lock. No second scheduler, queue or epoch authority.
- A single stored waker was inconsistent with `&self` drain methods: multiple
  futures could overwrite it. Both poll_drained and wait_drained now require
  `&mut self`, enforcing one drainer by the token's Rust borrow. Existing tests
  only gain the required mutable bindings; no behavior assertion is relaxed.

The new real Count-waker fixture leaves an admitted command unretired, polls its
capture drain Pending, revokes the original queue epoch, requires exactly one
wake and a repoll returning stale Err BEFORE command retirement. The Wake checks
that the original queue mutex is unlocked. Unstarted cancellation remains unfenced.
There are now14 prepared deterministic host tests, not executed by the author.

`single-drain-negative.rs` is a separate ROOT compile-negative snippet. Include
inside capture::tests in a fresh isolated check tree and require E0499 at the
second wait_drained borrow. The original move-only epoch/actor-scope negative
snippets remain immutable. Do not accept unrelated errors as negative evidence.

Binding wording correction to the original README: CommandPermit binds the
original queue Arc/physical key/epoch. CaptureTicket/Epoch additionally bind
caller-supplied ServiceId/stream metadata. Those fields and ActorScope do NOT
prove actual actor execution; the original Client::exclusive/vendor hook remains
unimplemented. Complete participation has no production issuer; production
capture remains CLOSED. No deferred GPU-owner or actual Ready proof is added.

Apply DRAIN_LIVENESS_FOLLOWUP.patch only after original PURE_PROTOCOL.patch in
an isolated tree. Proposed target file copies are under followup-v2/proposal.
All new code remains under the parent's cfg(test); removing test-only insertions
still gives the byte-identical original runtime physical_queue.rs.

Root alone reviews/compiles/runs the14 host fixtures and ownership negatives.
Author performed rustfmt parser/format and small source/hash projection checks
only. No Cargo, tests, GPU/provider/operator/model/container/debugger execution.

````

## Source: .agents/runs/serving-and-speculation-cuda-capture-exclusion-spine-20261020/followup-v2/frozen-v2/TODO_V2.md

Original UTF-8 bytes: 614; SHA256 `6af16194b2ed3e65d57ca66fd71d2f9189246b89eec0827747c2601174c474d5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Owned additive drain repair

Parent: existing P12 pure capture-exclusion preparation. Owner: spine_review.

- [x] Preserve original315b/c9f freeze/all11 files.
- [x] Wake selected capture drain on epoch change through original unlocked loop.
- [x] Encode a single drainer using mutable token borrow.
- [x] Prepare real revocation Count-waker test + separate E0499 negative.
- [x] Correct command-token vs caller-supplied capture actor metadata scope.
- [ ] Root independent review/compile/14test execution; no author execution.
- [ ] All vendor/common/runtime hooks and actual production capability remain open.

````

## Source: .agents/runs/serving-and-speculation-cuda-capture-exclusion-spine-20261020/frozen/CALLERS.md

Original UTF-8 bytes: 6101; SHA256 `85997c67259a624bbe61cb76fb2da0eba0bcdb2ac9e2a9aa8f8c34dfd15ff91b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Required integration map — source, not participation proof

Owner: `/root/spine_review`. This list is an implementation audit, not a new
production issuer. Every row remains unwired by this pure host-only slice.
Registry crate paths below are the actual pinned 0.11.0 source under
`/home/pvl/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`.

| Required boundary | Current source anchors | Necessary participation |
|---|---|---|
| Physical fair work | `zerogravity-gpu/src/physical_queue.rs::submit/poll_acquire/WaitingTicket/ActiveSlot` | Normal owned work begins before allocation/init; capture uses this same RR ticket/epoch, no priority queue/second byte ledger |
| Provider construction | `resident_leases.rs::ResidentService::new/new_pinned_cuda`; `resident_inference.rs` capability query through `device.client()` | Constructor permit before first runtime creation; bind actual physical key and original actor/stream; refuse if no complete gate proof |
| Candidate/packet/source/init | `ResidentService::allocate_uninitialized`; `ResidentInitialization::{allocate_empty,upload_bytes,complete_sync}`; `resident_inference.rs::allocate_state`; fused pre-ticket allocations | Work/command permit covers empty/create/write/sync before enqueue, including failure and readback future lifetime |
| Lease/cache release | `resident_leases.rs::LeaseInner::drop`; `resident_provider.rs::ResidentWeightLease::drop` | Last original handle and provider Client cannot be released into capture; pre-admitted bounded deferred ownership, original quarantine on unknown |
| Generic transport | `cubecl-common/src/device/handle/{base,channel,mutex,reentrant}.rs`: submit, submit_blocking, send/run_scoped, exclusive | Admit before closure enqueue; queued closure retains permit; actor never waits on gate; reentrancy requires exact owned epoch scope |
| Service creation/shutdown | `cubecl-common/src/device/handle/channel.rs::ChannelDeviceState::init/DeviceRunner/shutdown_device`; corresponding mutex service lifecycle | Bootstrap permit before service constructor; no new same-card service races Closing; final service/runner state teardown is admitted/deferred |
| Client/graph/profile | `cubecl-runtime/src/client.rs`: init/load, get_resource, create/empty/staging/write, launch/check/flush, graph lifecycle, sync, profiles, memory cleanup/report, collectives; `GraphHandle::drop:151` | All device-touching and implicit allocation/destruction calls participate; final GraphHandle drop is not a pure refcount operation |
| CUDA server entry | vendored `compute/server.rs`: staging105, read118, initialize_memory138, write158, launch, check243, flush251, graph_prepare261, begin266, end271, replay293, destroy311, sync343, profiles360/367/385, memory391/404/409, collectives420/429/466/478/513, get_resource1357 | A pre-enqueue token is required; gate before compile_failed/implicit stream, metadata buffer, pool relocation and fallback operations, not merely command() |
| CUDA bootstrap | vendored `runtime.rs::DeviceService::init:76`, DeviceProbe106, primary context retain, comm stream391, server416; `compute/server.rs::new:562` | Conservative bootstrap gate before context/stream creation; seal reported PCI identity, then same-card gate; ordinal is not a physical key |
| CUDA events outside actor | vendored `compute/events.rs::Cuda` all EventApi methods; `cubecl-server/src/device_events/{event,fence,profiler}.rs` | Originating card/gate attached to each event/fence/profile owner; wait/destroy/elapsed/fallback sync participate even on foreign host/GC thread |
| Async read/sync/GC | `cubecl-server/src/command/base.rs::read_async:334`; `stream/event.rs::ResolvedStreams::drop:249`; drop_queue worker | Copies are enqueued before returned future; future waits/destroys events outside actor; drain includes original retained source/destination/fence and posted GC |
| Allocation/free/relocation | vendored `compute/storage/{gpu,cpu}.rs`, `compute/driver.rs`, `compute/stream.rs`; common memory managers/drop queues | Every alloc/free (including sync fallback), host-pinned allocation, copy/relocation, stream/event creation and shutdown must be admitted or deferred |
| CUDA graph lifetime | vendored `compute/capture.rs::Executable::drop`; server graph_destroy synchronizes before destroy; original `CapturedResidentGraph` quarantine/retire | Preserve executable+band+service leases across all unknowns; no driver destruction in unadmitted Drop; keep original two-barrier explicit retirement |
| Original owned CUDA commands | accepted pinned-queue/owned_command adapter | Same original actor/stream and gate; adding an owned command must not create an ungated alternative entry |
| Same-card other runtimes | Vulkan/HIP/common clients sharing reported PCI; custom DeviceHandle services | Participate in common command/lifecycle contract or production capability refuses cold; a CUDA-only registry cannot prove their exclusion |

Pure cached Client properties/features/target metadata need no command permit.
Queries that lazily construct a Client, stream, pool or profile are not pure.
The scope excludes external processes/foreign runtimes; qualification must own
that environmental exclusion. No discovery scan is being called proof here.

Suggested disjoint implementation ownership after this protocol review:

1. GPU coordinator: physical_queue/physical_service and resident work ordering;
   original ResourceArbiter/fence bridge, existing all42/Ready/lease publisher.
2. Common/vendor transport owner: explicit shared generic command/lifecycle gate
   contracts, pre-enqueue ownership, bootstrap/shutdown and exclusive scope.
3. CUDA lifecycle owner: existing physical identity, events/fences, storage,
   graph/stream teardown and bounded original-owner deferral.
4. Independent qualifier: deterministic protocol negatives, then root-only
   actual two-model/two-actor same-card construction/alloc/drop/cancel capture.

No single owner may issue Complete before the other paths are source-closed and
physically qualified. No module should acquire another scheduler or VRAM ledger.

````

## Source: .agents/runs/serving-and-speculation-cuda-capture-exclusion-spine-20261020/frozen/DESIGN.md

Original UTF-8 bytes: 7852; SHA256 `6111e5b9f2a682c48a8b498924190c6ac3395de032ab1ec88c5affa9632993b1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Complete per-card capture exclusion: minimal honest boundary

Author: `/root/spine_review`; source-only proposal, 2026-10-20.
Production graphs remain closed. Initial graph is a caller-isolated qualifier.

## Finding

PhysicalQueue serializes one selected inference region per reported physical
card, and PhysicalWork retains/fences the original provider envelope. Neither
prevents device commands outside that region. Candidates/packets are allocated
before queue selection; initialization helpers do not require an ActiveSlot;
raw Client methods are unrestricted. CUDA fences can wait/destroy events on
other threads. Graph final drop posts destruction, whose server method first
synchronizes. Server construction creates a primary context and communication
stream before the GPU PhysicalQueue can be attached.

A CUDA-Server-method mutex is insufficient. Blocking there can park the actor
behind a capture-close operation. Refusing there can fail a normal command that
was already asynchronously enqueued. The admission decision and owned command
permit must precede enqueue; its lifetime must include the queued closure.
`Client::exclusive` is reentrant actor serialization, not a physical-card lock.

## Required layers

1. Keep existing fair PhysicalQueue and original prepaid ResourceArbiter. Add a
   capture request as a typed fair work item, not a priority bypass. Queue wait
   occurs before model allocation/workspace locks or any device command. A normal
   owned WorkPermit spans allocation/init/gather/launch/readback/terminal event;
   early drop after possible submission preserves original fence/quarantine.
2. One process-shared command gate per canonical reported physical identity.
   Do not key by CUDA ordinal, DeviceId, ServiceId, source or model. Vendor-level
   registration is needed before primary-context retain/stream construction;
   a short global bootstrap admission closes the unknown-card discovery race.
   Bootstrap releases only after an actual reported PCI/LUID binding is sealed.
3. Common transport admission before every `submit`, `submit_blocking`, scoped
   `exclusive`, service creation and shutdown. A command owns its permit through
   actor execution and any returned device-touching future. Normal pre-close
   queued closures finish; post-close external callers park/cancel outside actor.
   Never block a CUDA actor, destructor, executor poll or gate mutex on this gate.
4. CUDA event/storage/stream/graph lifecycle participates even outside transport.
   Events carry their originating gate/card owner. Wait/elapsed/destruction and
   GC/drop-queue work cannot silently call the driver while capture is active.
   Keep original buffers/events/graph/actor owners alive when deferring teardown.
5. Exclusive owned CaptureEpoch enters only after admission is closed, all
   admitted command permits and deferred work are drained, and each registered
   same-card actor has a trusted terminal stream barrier. No lock guard crosses
   await, a driver wait, actor call or Drop. A typed token holds the epoch instead.
6. Enter original Client::exclusive synchronously on the pinned actor/stream.
   Privately install the exact epoch/actor/stream scope for reentrant body calls.
   Only reviewed capture lifecycle/hot-body methods may use it; the token is
   not a generic allocation/sync/destruction bypass. No await inside that scope.
7. Stop/end capture, retain the original executable/leases in original flight
   owners, then leave actor scope. Resume queued/deferred destruction only after
   the epoch has a trusted closed result. Original all42 roots, Ready, publisher,
   byte accounting and kernel counts remain unchanged.

## Owned-state protocol

State: Open(epoch) -> Closing(ticket,epoch) -> Draining -> Capturing ->
Retiring -> Open(next_epoch), or Fenced. All transitions use short host locks.

- Capture request must win the existing fair queue; a stream of new ordinary
  requests cannot pass a pending capture forever. Ordinary command permits
  admitted before Closing remain valid and counted until executed/retired.
- Closing rejects no already-admitted native task. New async requests park with
  bounded tickets/wakers; a queued cancellation removes only that request.
- Capture token is private/noncloneable. A borrowed actor scope is derived only
  on its original actor, bound to card/service/stream/epoch and reviewed op kinds.
  Token/receipt from another card/service/epoch is a cold error, no device work.
- Cancelling before any capture/device effect reopens admission after draining
  its own bookkeeping. No GPU fence is invented for a never-submitted ticket.
- Cancelling after start/unknown stop retains every graph/workspace/source,
  event, pending destructor and original Client/actor owner; fence both gate and
  original ResourceArbiter card. Never clear by timeout or synchronizing in Drop.
- Gate/module mutex guards are never stored in permits. No actor callback or
  destructor is invoked while holding the state lock. Wakers run after unlock.
- Deferred destructor transport must be bounded and admitted before a resource
  becomes droppable. If its slot cannot be proved, refuse capture cold; do not
  drop the resource, allocate an unbounded closure queue, or block its destructor.
- A destructor originating during Capturing transfers its original resource and
  actor keepalive into its pre-admitted slot. It must not call graph_destroy,
  stream/event destruction or release the final actor. Replay Ready does not
  alone authorize executable destruction; keep the existing two-barrier retire.

## Why the narrow three-file edit cannot enable production

`cubecl-common` owns asynchronous/scoped transport and service shutdown;
`cubecl-runtime` owns Client and GraphHandle drop; `cubecl-server` owns events,
GC/drop workers and stream storage teardown. These are current registry crates,
not this repository's vendored CUDA crate. CUDA events carry no gate/card field.
A physical_queue/new GPU module/CudaServer-only change cannot intercept all
these entry points. The minimum complete integration needs explicit common
transport and lifecycle contracts (vendored/patch-owned source), plus CUDA
bootstrap/event/driver plumbing. Source-compatibility/default paths may stay
unchanged, but an unproven participant must make capture capability unavailable.

Cross-backend actors reporting the same PCI card must participate or make
production capture refuse cold. The present Vulkan/HIP/common Client paths do
not establish that participation. A PCI queue is not proof that every raw actor
or same-process external CUDA user is covered. External processes and foreign
CUDA runtimes remain outside this in-process contract; do not claim to exclude
those by a Rust permit. Physical qualification must explicitly control them.

## Smallest next implementation slice

After root agrees ownership, implement a pure per-card gate/state module and
opaque traits in the vendor common transport boundary, with a default-closed
`CaptureParticipation` capability. No actual capture method accepts a token
until every required hook is installed. Share the same gate authority with
PhysicalQueue; avoid two independently ordered schedulers or a second byte
ledger. Tests first cover owned tickets, phase/epoch/actor binding, fairness,
cancellation before effect, fence/unknown retention and bounded deferred drops.
This is preparatory infrastructure, not production graph support.

Do not patch only launches or release a command permit immediately after host
enqueue. Both leave actual posted device work/destruction outside the boundary.
Do not hold a permit for every buffer's entire lifetime: the intended graph's
own resident source/workspace buffers are live, and capture could never start.
Gate their device-affecting use/drop, preserving original physical owners.

````

## Source: .agents/runs/serving-and-speculation-cuda-capture-exclusion-spine-20261020/frozen/README.md

Original UTF-8 bytes: 3617; SHA256 `c7350e2dc18f7a02763326bafcf266c6d0a48a483b7d5aa50d70b98f35d7c794`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Pure physical-queue capture protocol proposal

Owner: `/root/spine_review`. Ignored preparatory slice only. No MAIN or graph
worktree tracked edits; no provider discovery, actual capture or math changes.

Two proposed source paths:

- `crates/zerogravity-gpu/src/physical_queue.rs`: only seven cfg(test) lines add
  a child module/field/initializer to its actual existing QueueState.
- `crates/zerogravity-gpu/src/physical_queue/capture.rs`: host protocol using that
  same queue mutex, original FairDeviceQueue ticket/ActiveSlot and process-shared
  ResourceArbiter physical key/fence. No registry, second scheduler or byte ledger.

The original default/runtime file projection is byte-identical when test-only
insertions are removed. No production completeness issuer exists. Participation
is CLOSED by default; synthetic complete issuance lives only inside host tests.
No existing command/constructor/drop is protected by this unwired prototype.
Production graph capability must remain unavailable until CALLERS.md is closed.

Protocol: pre-enqueue shared command permits -> fair original capture ticket ->
close new admission -> drain all earlier admitted command permits -> move-only
capture epoch -> exclusive borrowed actor scope -> ordered Begin/Body/End ->
trusted terminal -> original fair slot retirement. Every token binds actual
queue Arc, physical key, queue epoch and exact service/stream metadata. Guards
never cross await/driver/actor calls. Wakers run outside the queue mutex.

Queued/selected/closing cancellation before effects reopens without a device
fence. A started unknown command or capture keeps the gate fenced and calls the
same original ResourceArbiter fence. Actual device owners/quarantine remain the
existing ResidentSubmission/lease protocol; this slice does not implement those
hooks or claim that a host token is physical Ready. Deferred destructor slots,
bootstrap registration, actual actor scope installation and full participation
proof are intentionally unimplemented; see DESIGN.md and CALLERS.md.

13 deterministic host tests are PREPARED, NOT RUN: default/foreign completeness;
interleaved pre-close commands/drain; original RR fairness; queued/unstarted and
drain cancellation; unknown command/capture fence; stale epoch refusals; foreign
equal-numbered ticket/slot; epoch change after start and after known terminal;
unlocked wake; bounded permits/phase order and owned token consumption. These
use synthetic host-only physical keys and actual existing queue/arbiter policy,
with no CUDA client or invented physical memory claim.

`noncloneable-negative.rs` and `exclusive-borrow-negative.rs` are separate ROOT
compile-negative snippets for insertion INSIDE capture::tests in a fresh isolated
checkout. Require exactly E0599 at clone and E0499 at the overlapping mutable
borrow; do not treat unrelated resolution/privacy errors as a pass. Normal test
source deliberately does not include these failing snippets.

ROOT, after full source review and merging only into an isolated check tree,
can run the focused existing crate test gate:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 cargo test -p zerogravity-gpu physical_queue::capture::tests -- --test-threads=1
```

No source-only result marks production graph exclusion, metadata admission,
full inference or performance qualified. Parent alone owns wider integration,
strict/full checks and actual same-card multi-actor/constructor/drop/abort proof.

Author: rustfmt parser/format only; all13 tests and both negative snippets remain
unexecuted. Source/patch hashes are small local files, no model/DSO scan/runtime.

````

## Source: .agents/runs/serving-and-speculation-cuda-capture-exclusion-spine-20261020/frozen/TODO.md

Original UTF-8 bytes: 1311; SHA256 `e61c2c54e3a0258855ff2d8f08789d275b2be83b6f672608c024113a0b3fcdea`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Owned capture-exclusion preparation

Parent: `.agents/plans/serving-and-speculation/TODO.md` P12.
Owner: `/root/spine_review`. Ignored source-only proposal; MAIN/graph tracked
sources untouched. Root alone builds/tests/runs and reconciles the parent plan.

- [x] Read current physical queue, resident construction/leases, CUDA actor,
  runtime bootstrap, graph lifecycle, common channel/exclusive and event paths.
- [x] Identify concrete pre-queue allocations, enqueue, outside-actor event,
  deferred-drop/teardown and cross-backend participation gaps.
- [x] Specify ownership/order/cancellation/fence protocol and required callers.
- [x] Root approved pure protocol scope; wider vendor/common integration stays open.
- [x] Prepare first pure owned-token/fair capture state-machine slice;
  default-closed production participation capability, no capture enablement.
- [x] Prepare 13 host tests + two compile-negative ownership snippets, NOT RUN.
- [ ] Root review/compile/run pure slice.
- [ ] Wire every caller and issue actual complete participation only after proof.
- [ ] Root strict/full and actual multi-actor/card capture/abort/teardown gates.

No compilation, fixtures, native/operator/model/GPU/container/debugger runtime,
tracked source edits or large payload hashes performed by this preparation.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/CALLERS.md

Original UTF-8 bytes: 4737; SHA256 `53d53194c020b2d3034dfeb5f4f2a25043428c03618ef944be0ab6359c6877a5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Remaining complete-participation map

Author: `/root/spine_review`. All physical rows remain UNPROVEN. Proposed host
transport interception is not device-terminal or whole-lifecycle coverage.
Vendor paths are below `crates/zerogravity-gpu/vendor/cubecl-cuda/src/`; common/
runtime/server refer to actual cached0.11.0 sources, not registry assumptions.

| Boundary | First slice | Required before closing admission |
|---|---|---|
| physical_queue.rs/ResourceArbiter | No edits; pure14 fixture dependency | Bridge same actual QueueState/key/epoch/original fair ticket; no new registry, byte ledger or scheduler |
| common DeviceHandle submit/blocking/exclusive | Optional preenqueue host owner, retained in original closure | Bridge original work flight; cancellation/epoch/fence after posting; default None native regression |
| channel::ChannelDeviceState::init | Static bootstrap BEFORE CHANNELS; owned actor initializer | Original physical bootstrap identity before CUDA init; exact original context/actor registration and all creation failures |
| mutex::new/insert | Static bootstrap BEFORE registry lock | Actual same-card constructor/insert lifetime and physical identity |
| reentrant::locate/with_lock | Static metadata before GLOBAL; cold init permit after metadata lock released | Actual original actor scoped reentry/cancellation; source-default/no_std and races |
| Client explicit stream/exclusive | Mirror only actual set_stream; scope checks hook/service/explicit stream | Original pinned Client token + SAME capture epoch, not numeric metadata or an ordinary exclusive callback |
| common raw transports/custom services | No Complete issuer; three original cached paths only | Audit every custom service/alternate direct transport entry; reject production capability if an escape remains |
| common shutdown/runner/state Drop | Unchanged | Nonblocking prepaid retirement, posted-work drain and unknown fence; no actor join required by its own drain |
| ResidentService construction/init/allocations | No registration | Work permit BEFORE device.client/bootstrap and candidate/packet/scratch alloc/upload/sync; same physical source/BufferLease ownership |
| LeaseInner/ResidentWeightLease last Drop | Unchanged | Prepaid deferral retains actual original BufferLease/provider/Client through capture and original quarantine |
| Client stop_capture/GraphHandle Drop | Unchanged | Reserve slot/action BEFORE executable creation, nonblocking defer WITHOUT ordinary re-admission; original two barriers/executable owner retained |
| Client create/staging/write/read/launch/check/flush/profile/collective/memory | Most host closures cross new optional hook | Complete constructor/internal-event/allocation/failure paths; callback return is NOT terminal |
| CUDA Server all entries | No vendor delta | staging/read/init/write/launch (including compile_failed), check/flush/capture/replay/destroy/sync/profile/memory/collective/get_resource BEFORE any indirect allocation/effect |
| CUDA runtime::init/server::new | No driver delta | Pre-context/stream constructor token, authentic original DeviceProbe PCI identity, all error and primary-context release paths |
| CUDA EventApi and common event/fence/profiler | Unchanged | Actual card owner on create/record/wait/elapsed/stream-wait/destroy, including foreign host and GC thread |
| cubecl-server command::read_async/ResolvedStreams::drop | Unchanged | Copies enqueued before future exists; retain future/fence/source/destination/GC work to terminal/unknown |
| CUDA storage GPU/CPU and pool relocation/free | Unchanged | async/sync fallback alloc/free, pinned host memory, copy, relocation, deferred pool drain; no driver call in unadmitted Drop |
| CUDA stream/primary context/final actor teardown | Unchanged | Actual creation/final owner destruction participates; explicit bounded teardown without blocking Drop needed by drain |
| original pinned queue/OwnedCudaCommand | Unchanged | SAME original actor/stream gate; no alternative native command escape |
| same PCI card Vulkan/HIP/custom actors | Unchanged | Cross-runtime physical participation or cold production refusal; CUDA-only service registry is insufficient |

External-process exclusion remains an environmental qualification precondition,
not a guarantee of a process-local gate. Pure cached feature/property queries may
remain outside; a query that lazily initializes a Client, stream or event is not
pure. No discovery scan, constructor label or Arc clone is a completeness proof.

Disjoint next ownership: root physical-queue/flight bridge; common transport and
prepaid retirement owner; CUDA event/storage/constructor/destruction owner;
independent same-card qualifier. No owner can issue Complete alone.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/DESIGN.md

Original UTF-8 bytes: 6513; SHA256 `71db396b5f7cc65c1832b039c003eaea8eb71fa9bf9c6017a1375bdab2f7660c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Minimal pre-enqueue transport contract

Author: `/root/spine_review`. This advances source integration one boundary,
not production eligibility or an independent scheduler.

## Exact existing owner

The future CommandAdmission adapter must borrow the SAME physical QueueState,
ResourceArbiter key/epoch, existing fair ticket and original flight/quarantine
owners. Common ServiceId distinguishes original actor services, not physical
cards. No card lookup, completeness issuer, queue mutex, fairness algorithm or
byte counter is supplied by this new common module.

The earlier pure QueueState protocol stays test-only. Root has now reported
all14 host tests PASS in a fresh isolated tree plus all3 exact compile negatives
(E0599 clone, E0499 actor borrow, E0499 drain borrow), restoring exact source. Its original315b84/c9f7 and additive123684/d1e756 files
remain intact. Those tests/negative fixtures are a dependency for any bridge. An
actor/stream numeric pair is metadata until original Client/actor participation
and actual retained epoch are wired and qualified.

## Host lifetime

1. Read cached hook metadata without reaching the original actor/driver.
2. Obtain an owned permit before registry locks and enqueue. Any typed refusal
   means no callback/enqueue/driver effect; CallError identifies it separately
   from task panic and channel disconnection.
3. Move the permit into the original queued closure. Transport discard drops an
   unentered HOST owner; the caller returning from submit does not retire it.
4. Enter on the original actor/thread. The guard owns actor-local reentry and
   remains through callback execution/unwind. Enter/Drop never wait or invoke a
   driver. The guard is intentionally not required to be Send.
5. Its destruction retires HOST ownership only. Original asynchronous command
   flights/fences/GC/storage keep the separate posted-device ownership through
   known terminal Ready or original quarantine. This part is NOT implemented.

An admission-closing adapter may not be installed yet. The unsafe bridge
contract describes future waiting/known/unknown requirements, but does not
manufacture physical coverage. In particular, cancellation/epoch revocation
after enqueue still needs the original typed execution/failure/flight path;
tests here cover only refusal before enqueue and host owner cleanup.

## Constructor bootstrap and reentry

DeviceService::bootstrap_command_admission is cached metadata only, called
before CHANNELS/DEVICE_REGISTRY/GLOBAL locks. The channel initializer owns the
prepaid permit across the actual S::init actor closure. Mutex init enters on the
same calling thread. The lazy reentrant handle stores the cached bootstrap
hook, checks cold metadata under its existing lock, releases it, obtains the
bootstrap permit, then reacquires for actual lazy S::init. A raced duplicate
initializer remains ordered by the original lock. No gate may wait while a
registry/actor lock is held. An inserted S was already constructed: Install
does not claim to cover that external construction.

The cached actual service hook must equal bootstrap's Arc. A changed/missing
bridge fails cold instead of silently switching physical owners. Metadata
getters and hook Drop must remain nonblocking; no init/discovery in a getter.

Client::set_stream alone mirrors the actual explicit stream into its original
DeviceHandle. Original default stream resolution remains unchanged; admission
metadata None stays None. Common code never treats ambient StreamId as a pin.
CommandScope stores exact hook/service/explicit stream plus an original epoch
owner; it is not Clone and exclusive_with_command_scope requires &mut scope.
Only its actual entered permit may install reentry for nested original actor
commands. The future bridge must refuse foreign service/stream, stale/fenced
epochs and unowned actor reentry before effects. Client::exclusive remains the
old original actor scope; it does not become a physical-card exclusion lock.

## Required retirement slice before closing admission

Current GraphHandle::drop uses ordinary submit; final lease, event, pool and
service destruction have additional original paths. Waiting there can deadlock
a capture draining those same owners. This proposal DOES NOT reroute those
Drops or pretend their host command permit proves completion.

For GraphHandle, the smallest followup is reserve a bounded retirement slot and
preallocate its original-actor action BEFORE end_capture can create an owned
executable. Fill the returned graph ID without allocating. Last Drop moves that
prepaid action/actor owner to the original gate's deferred queue without
blocking or re-admitting; release only after the capture epoch permits it and
the original stream barriers establish destruction safety. Failure retains
original executable/band/service/source owners under original quarantine.
Apply the same ownership rule to event/storage/free/stream/context and final
actor teardown. No blocking destructor, fallback driver free, second drop worker
or foreign actor may bypass the same physical gate.

Only after all CALLERS.md rows are source-closed and physically tested may a
separately reviewed capability issuer replace Unproven. No command-presence
flag, driver label, same ordinal, ordinary Client::exclusive or test scope can
issue it.

## Root-only checks, after isolated registration is approved

- Root reported original pure protocol14 + clone/exclusive/drain negatives PASS;
  that host-only dependency does not establish any physical/actor issuer.
- Copy full current cached common/runtime sources to an isolated check tree,
  overlay ONLY these nine proposed files, retain exact base/package metadata;
  use already cached dependencies, no model/SDK downloads.
- Run common's focused admission fixtures plus original channel/mutex/reentrant
  suites under std/multi_threading and its supported no_std compile surface.
- Compile the two supplied negatives against that exact common build; require
  intended E0599/E0499, no unrelated failures.
- Strict common/runtime/CUDA/Vulkan/default checks and complete isolated repo
  gate after any accepted registration. Default None must not start capture,
  allocate a permit, reject existing work or alter original output semantics.
- Later: actual same-card independent actor/constructor/allocation/free/event/
  final-drop/cancel/fence/ragged-serving tests before production graph admission.

No root commands were run by the author; this is an implementation proposal.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/README.md

Original UTF-8 bytes: 4288; SHA256 `e84bdbb71ebd44659ab314c8c83e5c25097833d3640875b9fac4f132ed6bcc8a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# First optional common transport hook — SOURCE ONLY

Author: `/root/spine_review`; parent: `serving-and-speculation`.

Owned ignored proposal only. No MAIN, cached-crate, Cargo.lock, vendor registry,
CUDA driver, model math, graph implementation or shared plan edits. Original
pure-protocol/followup freezes remain immutable.

Nine proposed source paths:

- Common: `src/device/base.rs`; `src/device/handle/{base,mod,channel,mutex,reentrant}.rs`.
- New common: `src/device/handle/admission.rs` and `admission/tests.rs`.
- Runtime: `src/client.rs`.

They are copied under `proposal/{cubecl-common,cubecl-runtime}-0.11.0/`.
`baseline/` and `source-pins.json` bind the actual selected cached bytes and VCS
metadata. Both unpacked crates lack `.cargo-checksum.json`; their VCS metadata
and Cargo.lock checksums do not independently prove unpacked-byte authenticity.
No new upstream/registry equivalence claim is made. The existing cached package
license/metadata remains unchanged; no arithmetic/provider source is copied.

The seam obtains a cached optional hook before transport enqueue. The resulting
move-only owner is captured by the original queued closure and enters on its
original execution thread. Host execution/discard/panic retirement does not mean
GPU completion. Constructors obtain bootstrap admission BEFORE CHANNELS or
DEVICE_REGISTRY; lazy reentrant initialization releases its metadata lookup lock
before bootstrap admission. Cached service/bootstrap hook identity must match.

Default services return None. Their original submit/blocking/exclusive call
expressions remain the None branches; no hook, permit allocation or driver call
is introduced there. Client's existing explicit set_stream mirrors the SAME
logical stream into command metadata; None is never inferred from ambient
StreamId. Clone/seen_as preserve that metadata. Existing GraphHandle/Client APIs,
original graph source/ownership and default math stay in place.

An explicitly owned original capture scope is move-only and mutably borrowed
through the original actor exclusive call. Hook Arc, original service, explicit
stream and original epoch-owner validation precede enqueue. Nested reentry comes
from the entered permit's bridge, not logical IDs or a new runtime. Its unsafe
factory has no implementation here that owns an actual CUDA capture epoch.
Public contract types are exported through already-owned device/base.rs; the
existing device_handle public facade stays unchanged.

**Complete CUDA participation stays CLOSED.** The only participation variant is
Unproven, including when a hook exists. This source slice MUST NOT register an
admission-closing bridge. Graph final Drop, final service/actor teardown, external
events/storage/GC and device-terminal ownership remain unwired. An installed
bridge may not park those Drops on work needed by capture drain. CALLERS.md and
DESIGN.md give the required next integration, rather than claiming this seam
protects production capture.

13 host fixtures are PREPARED, NOT RUN: default None, queued lifetime, discard,
pre-enqueue refusal, unwind, ambient-vs-explicit stream, exact/stale/foreign
scope, entered nested reentry, clone/seen_as, permanent Unproven, and all THREE
original common transports with actual host constructor/reentrant dispatch.
Those are host tests, with synthetic epoch owners and no CUDA/physical proof.
Two compile-negative snippets require E0599 (scope clone) and E0499 (overlapping
mutable scope borrow); unrelated resolution/privacy errors are not a pass.

Author validation: rustfmt parser/format and bounded source/byte assertions only.
No author Cargo, rustc, tests, CUDA/container/model/operator/runtime execution.
Root has since reported all14 separate pure QueueState host tests PASS in its
fresh isolated tree (14.81s compile, 0.00s test) and all3 intended compile negatives
(E0599 clone; E0499 actor borrow; E0499 drain borrow), restoring exact proposal
bytes afterward. Root's capture-host-protocol-result.json binds that host-only
result; no production issuer or physical completeness was promoted. Root must review this proposal and wire copied
cached crates in an isolated check tree. No Cargo/vendor registration was prepared or changed here. All
runtime/physical/performance acceptance remains pending.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/TODO.md

Original UTF-8 bytes: 1846; SHA256 `8c2bc7312fc318de38ff11dab32bba60502328c1176a3eadaaf7e6dcde2511a8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: `.agents/plans/serving-and-speculation/TODO.md`.
Owner: `/root/spine_review`. Ignored source-only integration proposal.

- [x] Pin actual seven cached source files + package/VCS metadata; preserve
  missing checksum-file preparation failure (no source copies written then).
- [x] Prepare optional preenqueue owned host permit and typed refusal seam.
- [x] Keep default None original calls; participation only Unproven.
- [x] Bootstrap before registry locks; retain original initializer owner.
- [x] Explicit pinned Client metadata + move-only mutable scope; no ambient
  logical-stream/physical-card identity inference.
- [x] Prepare13 host fixture sources and2 compile negatives, not executed.
- [x] Document actual physical/retirement/constructor/event integration gaps.
- [ ] Root source review + independent review; do not merge/register yet.
- [x] Root reported pure protocol14 actual PASS plus all3 intended compile
  negatives; original isolated bytes restored. Author did not execute them.
- [ ] Root compile this focused common proposal and supported default/std/no_std/
  runtime checks in isolated tree; this proposal's13 fixtures remain unrun.
- [ ] Implement actual original QueueState/flight bridge only after dependency
  and API review; no independent parallel tracker.
- [ ] Close constructor/external event/storage/GC/final Drop map, prepaid
  retirement and actual original pinned actor scope before closing admission.
- [ ] Root strict/full/physical multi-actor same-card gates before any complete
  production graph participation/capture/performance claim.

No author Cargo/rustc/build/test/operator/Torch/model/container/GPU execution,
vendor registration, shared plan edit or commit. Current root17 graph/RoPE source
and later root-owned cap-only helper correction were not edited by this run.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/followup-v2/README_V2.md

Original UTF-8 bytes: 727; SHA256 `238c573fb1aaddf3b16909e26b0b504d77b1b0666a5c8aa539e4ef378f91d038`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive CallError classification repair — SOURCE ONLY

Original bb7de6de/cabcf725 and all30 files remain immutable.
Overlay only two files after original proposal. into_panic returns ONLY the
actual stored actor panic payload. Admission refusal still has its typed
admission_error/message; resume reports it explicitly on the caller rather
than synthesizing an actor panic payload. Existing actual panic path unchanged.
One new fixture checks refusalSome/into_panicNone, untouched actual u32 payload,
and exact caller-side refusal panic text; proposed focused test count now14.
Author rustfmt/source/diff only. No tests/build/runtime/MAIN/vendor edits.
Complete participation remains Unproven; no bridge closes admission.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/followup-v2/frozen-v2/README_V2.md

Original UTF-8 bytes: 727; SHA256 `238c573fb1aaddf3b16909e26b0b504d77b1b0666a5c8aa539e4ef378f91d038`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Additive CallError classification repair — SOURCE ONLY

Original bb7de6de/cabcf725 and all30 files remain immutable.
Overlay only two files after original proposal. into_panic returns ONLY the
actual stored actor panic payload. Admission refusal still has its typed
admission_error/message; resume reports it explicitly on the caller rather
than synthesizing an actor panic payload. Existing actual panic path unchanged.
One new fixture checks refusalSome/into_panicNone, untouched actual u32 payload,
and exact caller-side refusal panic text; proposed focused test count now14.
Author rustfmt/source/diff only. No tests/build/runtime/MAIN/vendor edits.
Complete participation remains Unproven; no bridge closes admission.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/frozen/CALLERS.md

Original UTF-8 bytes: 4737; SHA256 `53d53194c020b2d3034dfeb5f4f2a25043428c03618ef944be0ab6359c6877a5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Remaining complete-participation map

Author: `/root/spine_review`. All physical rows remain UNPROVEN. Proposed host
transport interception is not device-terminal or whole-lifecycle coverage.
Vendor paths are below `crates/zerogravity-gpu/vendor/cubecl-cuda/src/`; common/
runtime/server refer to actual cached0.11.0 sources, not registry assumptions.

| Boundary | First slice | Required before closing admission |
|---|---|---|
| physical_queue.rs/ResourceArbiter | No edits; pure14 fixture dependency | Bridge same actual QueueState/key/epoch/original fair ticket; no new registry, byte ledger or scheduler |
| common DeviceHandle submit/blocking/exclusive | Optional preenqueue host owner, retained in original closure | Bridge original work flight; cancellation/epoch/fence after posting; default None native regression |
| channel::ChannelDeviceState::init | Static bootstrap BEFORE CHANNELS; owned actor initializer | Original physical bootstrap identity before CUDA init; exact original context/actor registration and all creation failures |
| mutex::new/insert | Static bootstrap BEFORE registry lock | Actual same-card constructor/insert lifetime and physical identity |
| reentrant::locate/with_lock | Static metadata before GLOBAL; cold init permit after metadata lock released | Actual original actor scoped reentry/cancellation; source-default/no_std and races |
| Client explicit stream/exclusive | Mirror only actual set_stream; scope checks hook/service/explicit stream | Original pinned Client token + SAME capture epoch, not numeric metadata or an ordinary exclusive callback |
| common raw transports/custom services | No Complete issuer; three original cached paths only | Audit every custom service/alternate direct transport entry; reject production capability if an escape remains |
| common shutdown/runner/state Drop | Unchanged | Nonblocking prepaid retirement, posted-work drain and unknown fence; no actor join required by its own drain |
| ResidentService construction/init/allocations | No registration | Work permit BEFORE device.client/bootstrap and candidate/packet/scratch alloc/upload/sync; same physical source/BufferLease ownership |
| LeaseInner/ResidentWeightLease last Drop | Unchanged | Prepaid deferral retains actual original BufferLease/provider/Client through capture and original quarantine |
| Client stop_capture/GraphHandle Drop | Unchanged | Reserve slot/action BEFORE executable creation, nonblocking defer WITHOUT ordinary re-admission; original two barriers/executable owner retained |
| Client create/staging/write/read/launch/check/flush/profile/collective/memory | Most host closures cross new optional hook | Complete constructor/internal-event/allocation/failure paths; callback return is NOT terminal |
| CUDA Server all entries | No vendor delta | staging/read/init/write/launch (including compile_failed), check/flush/capture/replay/destroy/sync/profile/memory/collective/get_resource BEFORE any indirect allocation/effect |
| CUDA runtime::init/server::new | No driver delta | Pre-context/stream constructor token, authentic original DeviceProbe PCI identity, all error and primary-context release paths |
| CUDA EventApi and common event/fence/profiler | Unchanged | Actual card owner on create/record/wait/elapsed/stream-wait/destroy, including foreign host and GC thread |
| cubecl-server command::read_async/ResolvedStreams::drop | Unchanged | Copies enqueued before future exists; retain future/fence/source/destination/GC work to terminal/unknown |
| CUDA storage GPU/CPU and pool relocation/free | Unchanged | async/sync fallback alloc/free, pinned host memory, copy, relocation, deferred pool drain; no driver call in unadmitted Drop |
| CUDA stream/primary context/final actor teardown | Unchanged | Actual creation/final owner destruction participates; explicit bounded teardown without blocking Drop needed by drain |
| original pinned queue/OwnedCudaCommand | Unchanged | SAME original actor/stream gate; no alternative native command escape |
| same PCI card Vulkan/HIP/custom actors | Unchanged | Cross-runtime physical participation or cold production refusal; CUDA-only service registry is insufficient |

External-process exclusion remains an environmental qualification precondition,
not a guarantee of a process-local gate. Pure cached feature/property queries may
remain outside; a query that lazily initializes a Client, stream or event is not
pure. No discovery scan, constructor label or Arc clone is a completeness proof.

Disjoint next ownership: root physical-queue/flight bridge; common transport and
prepaid retirement owner; CUDA event/storage/constructor/destruction owner;
independent same-card qualifier. No owner can issue Complete alone.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/frozen/DESIGN.md

Original UTF-8 bytes: 6513; SHA256 `71db396b5f7cc65c1832b039c003eaea8eb71fa9bf9c6017a1375bdab2f7660c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Minimal pre-enqueue transport contract

Author: `/root/spine_review`. This advances source integration one boundary,
not production eligibility or an independent scheduler.

## Exact existing owner

The future CommandAdmission adapter must borrow the SAME physical QueueState,
ResourceArbiter key/epoch, existing fair ticket and original flight/quarantine
owners. Common ServiceId distinguishes original actor services, not physical
cards. No card lookup, completeness issuer, queue mutex, fairness algorithm or
byte counter is supplied by this new common module.

The earlier pure QueueState protocol stays test-only. Root has now reported
all14 host tests PASS in a fresh isolated tree plus all3 exact compile negatives
(E0599 clone, E0499 actor borrow, E0499 drain borrow), restoring exact source. Its original315b84/c9f7 and additive123684/d1e756 files
remain intact. Those tests/negative fixtures are a dependency for any bridge. An
actor/stream numeric pair is metadata until original Client/actor participation
and actual retained epoch are wired and qualified.

## Host lifetime

1. Read cached hook metadata without reaching the original actor/driver.
2. Obtain an owned permit before registry locks and enqueue. Any typed refusal
   means no callback/enqueue/driver effect; CallError identifies it separately
   from task panic and channel disconnection.
3. Move the permit into the original queued closure. Transport discard drops an
   unentered HOST owner; the caller returning from submit does not retire it.
4. Enter on the original actor/thread. The guard owns actor-local reentry and
   remains through callback execution/unwind. Enter/Drop never wait or invoke a
   driver. The guard is intentionally not required to be Send.
5. Its destruction retires HOST ownership only. Original asynchronous command
   flights/fences/GC/storage keep the separate posted-device ownership through
   known terminal Ready or original quarantine. This part is NOT implemented.

An admission-closing adapter may not be installed yet. The unsafe bridge
contract describes future waiting/known/unknown requirements, but does not
manufacture physical coverage. In particular, cancellation/epoch revocation
after enqueue still needs the original typed execution/failure/flight path;
tests here cover only refusal before enqueue and host owner cleanup.

## Constructor bootstrap and reentry

DeviceService::bootstrap_command_admission is cached metadata only, called
before CHANNELS/DEVICE_REGISTRY/GLOBAL locks. The channel initializer owns the
prepaid permit across the actual S::init actor closure. Mutex init enters on the
same calling thread. The lazy reentrant handle stores the cached bootstrap
hook, checks cold metadata under its existing lock, releases it, obtains the
bootstrap permit, then reacquires for actual lazy S::init. A raced duplicate
initializer remains ordered by the original lock. No gate may wait while a
registry/actor lock is held. An inserted S was already constructed: Install
does not claim to cover that external construction.

The cached actual service hook must equal bootstrap's Arc. A changed/missing
bridge fails cold instead of silently switching physical owners. Metadata
getters and hook Drop must remain nonblocking; no init/discovery in a getter.

Client::set_stream alone mirrors the actual explicit stream into its original
DeviceHandle. Original default stream resolution remains unchanged; admission
metadata None stays None. Common code never treats ambient StreamId as a pin.
CommandScope stores exact hook/service/explicit stream plus an original epoch
owner; it is not Clone and exclusive_with_command_scope requires &mut scope.
Only its actual entered permit may install reentry for nested original actor
commands. The future bridge must refuse foreign service/stream, stale/fenced
epochs and unowned actor reentry before effects. Client::exclusive remains the
old original actor scope; it does not become a physical-card exclusion lock.

## Required retirement slice before closing admission

Current GraphHandle::drop uses ordinary submit; final lease, event, pool and
service destruction have additional original paths. Waiting there can deadlock
a capture draining those same owners. This proposal DOES NOT reroute those
Drops or pretend their host command permit proves completion.

For GraphHandle, the smallest followup is reserve a bounded retirement slot and
preallocate its original-actor action BEFORE end_capture can create an owned
executable. Fill the returned graph ID without allocating. Last Drop moves that
prepaid action/actor owner to the original gate's deferred queue without
blocking or re-admitting; release only after the capture epoch permits it and
the original stream barriers establish destruction safety. Failure retains
original executable/band/service/source owners under original quarantine.
Apply the same ownership rule to event/storage/free/stream/context and final
actor teardown. No blocking destructor, fallback driver free, second drop worker
or foreign actor may bypass the same physical gate.

Only after all CALLERS.md rows are source-closed and physically tested may a
separately reviewed capability issuer replace Unproven. No command-presence
flag, driver label, same ordinal, ordinary Client::exclusive or test scope can
issue it.

## Root-only checks, after isolated registration is approved

- Root reported original pure protocol14 + clone/exclusive/drain negatives PASS;
  that host-only dependency does not establish any physical/actor issuer.
- Copy full current cached common/runtime sources to an isolated check tree,
  overlay ONLY these nine proposed files, retain exact base/package metadata;
  use already cached dependencies, no model/SDK downloads.
- Run common's focused admission fixtures plus original channel/mutex/reentrant
  suites under std/multi_threading and its supported no_std compile surface.
- Compile the two supplied negatives against that exact common build; require
  intended E0599/E0499, no unrelated failures.
- Strict common/runtime/CUDA/Vulkan/default checks and complete isolated repo
  gate after any accepted registration. Default None must not start capture,
  allocate a permit, reject existing work or alter original output semantics.
- Later: actual same-card independent actor/constructor/allocation/free/event/
  final-drop/cancel/fence/ragged-serving tests before production graph admission.

No root commands were run by the author; this is an implementation proposal.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/frozen/README.md

Original UTF-8 bytes: 4288; SHA256 `e84bdbb71ebd44659ab314c8c83e5c25097833d3640875b9fac4f132ed6bcc8a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# First optional common transport hook — SOURCE ONLY

Author: `/root/spine_review`; parent: `serving-and-speculation`.

Owned ignored proposal only. No MAIN, cached-crate, Cargo.lock, vendor registry,
CUDA driver, model math, graph implementation or shared plan edits. Original
pure-protocol/followup freezes remain immutable.

Nine proposed source paths:

- Common: `src/device/base.rs`; `src/device/handle/{base,mod,channel,mutex,reentrant}.rs`.
- New common: `src/device/handle/admission.rs` and `admission/tests.rs`.
- Runtime: `src/client.rs`.

They are copied under `proposal/{cubecl-common,cubecl-runtime}-0.11.0/`.
`baseline/` and `source-pins.json` bind the actual selected cached bytes and VCS
metadata. Both unpacked crates lack `.cargo-checksum.json`; their VCS metadata
and Cargo.lock checksums do not independently prove unpacked-byte authenticity.
No new upstream/registry equivalence claim is made. The existing cached package
license/metadata remains unchanged; no arithmetic/provider source is copied.

The seam obtains a cached optional hook before transport enqueue. The resulting
move-only owner is captured by the original queued closure and enters on its
original execution thread. Host execution/discard/panic retirement does not mean
GPU completion. Constructors obtain bootstrap admission BEFORE CHANNELS or
DEVICE_REGISTRY; lazy reentrant initialization releases its metadata lookup lock
before bootstrap admission. Cached service/bootstrap hook identity must match.

Default services return None. Their original submit/blocking/exclusive call
expressions remain the None branches; no hook, permit allocation or driver call
is introduced there. Client's existing explicit set_stream mirrors the SAME
logical stream into command metadata; None is never inferred from ambient
StreamId. Clone/seen_as preserve that metadata. Existing GraphHandle/Client APIs,
original graph source/ownership and default math stay in place.

An explicitly owned original capture scope is move-only and mutably borrowed
through the original actor exclusive call. Hook Arc, original service, explicit
stream and original epoch-owner validation precede enqueue. Nested reentry comes
from the entered permit's bridge, not logical IDs or a new runtime. Its unsafe
factory has no implementation here that owns an actual CUDA capture epoch.
Public contract types are exported through already-owned device/base.rs; the
existing device_handle public facade stays unchanged.

**Complete CUDA participation stays CLOSED.** The only participation variant is
Unproven, including when a hook exists. This source slice MUST NOT register an
admission-closing bridge. Graph final Drop, final service/actor teardown, external
events/storage/GC and device-terminal ownership remain unwired. An installed
bridge may not park those Drops on work needed by capture drain. CALLERS.md and
DESIGN.md give the required next integration, rather than claiming this seam
protects production capture.

13 host fixtures are PREPARED, NOT RUN: default None, queued lifetime, discard,
pre-enqueue refusal, unwind, ambient-vs-explicit stream, exact/stale/foreign
scope, entered nested reentry, clone/seen_as, permanent Unproven, and all THREE
original common transports with actual host constructor/reentrant dispatch.
Those are host tests, with synthetic epoch owners and no CUDA/physical proof.
Two compile-negative snippets require E0599 (scope clone) and E0499 (overlapping
mutable scope borrow); unrelated resolution/privacy errors are not a pass.

Author validation: rustfmt parser/format and bounded source/byte assertions only.
No author Cargo, rustc, tests, CUDA/container/model/operator/runtime execution.
Root has since reported all14 separate pure QueueState host tests PASS in its
fresh isolated tree (14.81s compile, 0.00s test) and all3 intended compile negatives
(E0599 clone; E0499 actor borrow; E0499 drain borrow), restoring exact proposal
bytes afterward. Root's capture-host-protocol-result.json binds that host-only
result; no production issuer or physical completeness was promoted. Root must review this proposal and wire copied
cached crates in an isolated check tree. No Cargo/vendor registration was prepared or changed here. All
runtime/physical/performance acceptance remains pending.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-admission-spine-20261020/frozen/TODO.md

Original UTF-8 bytes: 1846; SHA256 `8c2bc7312fc318de38ff11dab32bba60502328c1176a3eadaaf7e6dcde2511a8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: `.agents/plans/serving-and-speculation/TODO.md`.
Owner: `/root/spine_review`. Ignored source-only integration proposal.

- [x] Pin actual seven cached source files + package/VCS metadata; preserve
  missing checksum-file preparation failure (no source copies written then).
- [x] Prepare optional preenqueue owned host permit and typed refusal seam.
- [x] Keep default None original calls; participation only Unproven.
- [x] Bootstrap before registry locks; retain original initializer owner.
- [x] Explicit pinned Client metadata + move-only mutable scope; no ambient
  logical-stream/physical-card identity inference.
- [x] Prepare13 host fixture sources and2 compile negatives, not executed.
- [x] Document actual physical/retirement/constructor/event integration gaps.
- [ ] Root source review + independent review; do not merge/register yet.
- [x] Root reported pure protocol14 actual PASS plus all3 intended compile
  negatives; original isolated bytes restored. Author did not execute them.
- [ ] Root compile this focused common proposal and supported default/std/no_std/
  runtime checks in isolated tree; this proposal's13 fixtures remain unrun.
- [ ] Implement actual original QueueState/flight bridge only after dependency
  and API review; no independent parallel tracker.
- [ ] Close constructor/external event/storage/GC/final Drop map, prepaid
  retirement and actual original pinned actor scope before closing admission.
- [ ] Root strict/full/physical multi-actor same-card gates before any complete
  production graph participation/capture/performance claim.

No author Cargo/rustc/build/test/operator/Torch/model/container/GPU execution,
vendor registration, shared plan edit or commit. Current root17 graph/RoPE source
and later root-owned cap-only helper correction were not edited by this run.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-host-harness-20261020/README.md

Original UTF-8 bytes: 1902; SHA256 `07ab5d382a7859b0da6a4457b383e4ecf77f60fe58481ac1c1954bd880456678`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Exact14 optional-command host harness

Author: `/root/spine_review`. Parent serving-and-speculation. SOURCE ONLY; no
Cargo/rustc/test/actor/driver execution by author. No fake dependency/shim.

`harness/` copies genuine cached common0.11.0 plus exact firsthook and typed-v2
Rust overlays. Only three files differ from that composed source:

- Cargo.toml: omit upstream bench target and all three dev-dependencies
  (criterion/tempfile/test-log), add standalone empty workspace. Normal features,
  dependencies/build-dependencies/build.rs remain unchanged. Preserved original
  manifest and current ROOT lock snapshot make this selection reviewable.
- format.rs and benchmark.rs: exclude ONLY their logging test modules when
  zg_host_admission_only is set. All module/test function bodies unchanged;
  default cfg projection unchanged. No `test_log` substitute exists.

All14 admission fixture bodies remain byte-exact in their original private
module. Actual DeviceFixture/ticker/transport helpers remain byte-exact. All
other upstream library test code can compile using normal dependencies; exact
module filtering runs only the14 requested tests. This does not claim the two
excluded logging modules or whole upstream test suite passed.

ROOT-only command (adapt supervision/isolation and preserve required flags):

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
RUSTFLAGS='--cfg zg_host_admission_only --check-cfg=cfg(zg_host_admission_only)' \
cargo test --offline --manifest-path "$RUN/harness/Cargo.toml" --lib \
  device::handle::admission::tests:: -- --test-threads=1
```

Check actual listed/result count14 against source-pins.json; zero tests is not
acceptance. The copied whole ROOT lock may need a ROOT-recorded harness-local
resolution update; never edit MAIN lock, enable network or infer cache availability
from this preparation. No CUDA/device terminal/complete participation claim.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-host-harness-20261020/TODO.md

Original UTF-8 bytes: 488; SHA256 `bebcf324f32eb6c828dec2b35d10e22c2e85934f39a0f3fc09096190a34de5a2`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: serving-and-speculation. Owner: /root/spine_review.

- [x] Preserve composed source/normal deps/exact14 bodies/helpers.
- [x] Prepare only two unrelated test-module selection cfg attributes and a
  bench/dev-dependency-free standalone manifest; no fake test-log package.
- [x] Source checks/manifest parse only.
- [ ] ROOT review and offline Cargo run, nonvacuous exact14 result.
- [ ] Any missing normal cache dependency remains an explicit operational gap.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-host-harness-20261020/frozen/README.md

Original UTF-8 bytes: 1902; SHA256 `07ab5d382a7859b0da6a4457b383e4ecf77f60fe58481ac1c1954bd880456678`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Exact14 optional-command host harness

Author: `/root/spine_review`. Parent serving-and-speculation. SOURCE ONLY; no
Cargo/rustc/test/actor/driver execution by author. No fake dependency/shim.

`harness/` copies genuine cached common0.11.0 plus exact firsthook and typed-v2
Rust overlays. Only three files differ from that composed source:

- Cargo.toml: omit upstream bench target and all three dev-dependencies
  (criterion/tempfile/test-log), add standalone empty workspace. Normal features,
  dependencies/build-dependencies/build.rs remain unchanged. Preserved original
  manifest and current ROOT lock snapshot make this selection reviewable.
- format.rs and benchmark.rs: exclude ONLY their logging test modules when
  zg_host_admission_only is set. All module/test function bodies unchanged;
  default cfg projection unchanged. No `test_log` substitute exists.

All14 admission fixture bodies remain byte-exact in their original private
module. Actual DeviceFixture/ticker/transport helpers remain byte-exact. All
other upstream library test code can compile using normal dependencies; exact
module filtering runs only the14 requested tests. This does not claim the two
excluded logging modules or whole upstream test suite passed.

ROOT-only command (adapt supervision/isolation and preserve required flags):

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
RUSTFLAGS='--cfg zg_host_admission_only --check-cfg=cfg(zg_host_admission_only)' \
cargo test --offline --manifest-path "$RUN/harness/Cargo.toml" --lib \
  device::handle::admission::tests:: -- --test-threads=1
```

Check actual listed/result count14 against source-pins.json; zero tests is not
acceptance. The copied whole ROOT lock may need a ROOT-recorded harness-local
resolution update; never edit MAIN lock, enable network or infer cache availability
from this preparation. No CUDA/device terminal/complete participation claim.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-host-harness-20261020/frozen/TODO.md

Original UTF-8 bytes: 488; SHA256 `bebcf324f32eb6c828dec2b35d10e22c2e85934f39a0f3fc09096190a34de5a2`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Working checklist

Parent: serving-and-speculation. Owner: /root/spine_review.

- [x] Preserve composed source/normal deps/exact14 bodies/helpers.
- [x] Prepare only two unrelated test-module selection cfg attributes and a
  bench/dev-dependency-free standalone manifest; no fake test-log package.
- [x] Source checks/manifest parse only.
- [ ] ROOT review and offline Cargo run, nonvacuous exact14 result.
- [ ] Any missing normal cache dependency remains an explicit operational gap.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-host-harness-20261020/frozen/harness/README.md

Original UTF-8 bytes: 277; SHA256 `45c38ed8d1a7c0341fb2e854ac26b1e3ea7097758fbb9015f7f8e984a2e4f476`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CubeCL Common

The `cubecl-common` package hosts code that _must_ be shared between cubecl packages (with `std` or
`no_std` enabled). No other code should be placed in this package unless unavoidable.

The package must build with `cargo build --no-default-features` as well.

````

## Source: .agents/runs/serving-and-speculation-cuda-command-host-harness-20261020/harness/README.md

Original UTF-8 bytes: 277; SHA256 `45c38ed8d1a7c0341fb2e854ac26b1e3ea7097758fbb9015f7f8e984a2e4f476`.
Resolve embedded relative paths against the original path, not this volume.

````text
# CubeCL Common

The `cubecl-common` package hosts code that _must_ be shared between cubecl packages (with `std` or
`no_std` enabled). No other code should be placed in this package unless unavoidable.

The package must build with `cargo build --no-default-features` as well.

````

