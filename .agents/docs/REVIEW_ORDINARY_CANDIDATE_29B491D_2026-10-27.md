# Review record — ordinary Qwen CUDA candidate `29b491d` (interrupted)

Author: @pvl
Status: **review interrupted by usage limit; candidate neither integrated nor rejected**

## 1. What exists

Signed branch `pi-web-agent-8026e968-eb45-46fd-a2a1-d669a232fe6b`:

| commit | title |
|---|---|
| `52bfb51` | base — current `main` |
| `ec91ad1` | Add dynamic CUDA ordinary text primitives |
| `3a91a97` | fix(cuda): constrain ordinary text candidate |
| `29b491d` | fix(cuda): close ordinary Qwen review blockers |

Diff `52bfb51..29b491d`: 9 files, +2852/−5.

- `crates/zerogravity-gpu/src/cuda/ordinary_text/{mod.rs,kernels.cu,tests.rs}` (new)
- `crates/zerogravity-gpu/src/cuda/{cuda.rs,kernel.rs}`, `lib.rs`,
  `prepared_kernel_key.rs`
- `crates/zerogravity/src/qwen35_cuda.rs` and new
  `crates/zerogravity/src/qwen35_cuda/ordinary.rs`

The worktree that produced it was removed. Re-read with:

```bash
git -C /home/pvl/spaces/abird/src/zerogravity worktree add /tmp/rev-29b491d 29b491d
# or: git show 29b491d:<path>
```

Do not assume `main` contains this work: `main` is `52bfb51` and does **not**
include `ordinary_text` or `qwen35_cuda/ordinary.rs`.

## 2. Why the review stopped

The independent review agent was terminated by a Codex usage limit
(`Codex error: The usage limit has been reached`) after finishing the layout/stride
sweep, the sigmoid/multiply split, and part of the `kernel.rs` ABI read. It never
opened `qwen35_cuda/ordinary.rs` or audited the `kernels.cu` status sites, and no
verdict was issued. The coordinator did not integrate the branch.

Therefore: **no independent acceptance exists for `29b491d`.** The commit's own
passing test/clippy/build report is implementer evidence, not acceptance.

## 3. Implementer claims to re-verify, not trust

The implementer reported: `cargo fmt --check`; CUDA-native and qualification
all-target checks; Clippy `-D warnings`; `cuda-native` 157 passed / 3 ignored + 7
integration; `cuda-native-qualification` 646 passed / 42 ignored + 7;
`zerogravity --features qwen35-cuda-qualification` 27 passed / 7 ignored, plus
`qwen35_cuda` 3/4 and `qwen35_batching` 7/8. All 0 failed. No GPU/model run.

Feature-matrix gap: no executed command covered plain
`cargo check -p zerogravity --features qwen35-cuda` (without
`-qualification`), the default build, or `--all-features`. Treat those
configurations as unverified until run.

Correct a stale earlier note: the tree does compile and did pass. The `−1` lines
in `kernel.rs` and `prepared_kernel_key.rs` are the removal of duplicated inner
`#![cfg_attr(not(feature = "cuda-native-qualification"), allow(dead_code))]`
attributes (`duplicated_attributes` under stricter Clippy, already applied by
`cuda.rs:9` and `lib.rs:43`). They are not removed safety checks. Do not
re-litigate compilation; verify semantics.

## 4. Reviewer checklist state at interruption

Legend: **V** = verified by the interrupted reviewer; **C** = additionally
re-checked by the coordinator after interruption; **O** = still open.

### A. Byte extents and strides — V, C

- Extents compute through `checked_mul`/`checked_add`.
- Both profiles share `query_width = 8 × 256 = 2048`; therefore
  `attention_row_stride = 2*query_width + 2*kv_width = 5120` and
  `value_offset = 4608` for both `.8B` and `2B`. Only `hidden_width`,
  `intermediate_width`, `vocabulary` differ.
- `C`: `MAX_RUNTIME_ROWS = 64` bounds rows in every plan; `MAX_CONTEXT_CAPACITY =
  32768` bounds capacity; the plan rejects rows outside `1..=64` and capacity
  outside `1..=32768`. Scalar `u32` words cannot truncate at those bounds.
- Still open: state explicitly whether `MAX_CONTEXT_CAPACITY * attention_row_stride`
  element count and its `× BF16_BYTES` byte product are asserted to stay in range
  for every geometry rather than only relying on the bound being small.

### B. `require_strided_view` exclusive-end arithmetic — V

`end = (rows−1)*row_stride + offset + (groups−1)*group_stride + width` is an
exclusive end. Packed cases land exactly on `total_elements` (gate 4096, K RMS
4608, value 5120). Same shape for `SiluUp`
(`gate_offset = 0`, `up_offset = intermediate_width`, `row_stride = 2*intermediate_width`).

- **O**: `zg_ot_silu_up_v1` scalar word order vs the Rust `u32_words(&[…])` call
  is the one remaining unverified ABI-order pair. Every other order
  (sigmoid, head RMS, kv_append, multiply, residual, greedy) matched.

### C. Sigmoid → BF16 → multiply split — V, C

- `zg_ot_sigmoid_v1` stores `zg_store(sigmoid)`; `zg_ot_multiply_v1` reloads that
  BF16 value and multiplies. The fused `x / (1 + expf(-gate))` attention-gate form
  is gone; `tests.rs` asserts both the presence of the separate sigmoid line and
  the absence of the fused form.
- `C` correction to an earlier reviewer doubt: `zg_ot_silu_up_v1` rounds the
  activation to BF16 and then computes `f32(bf16_act) * f32(bf16_up)`. Because the
  product of two BF16 values is exact in FP32 (8-bit significands, 16-bit product
  < 24-bit mantissa), the single BF16 store equals a BF16 multiply. This is not a
  deviation. The same argument applies to the gated multiply. State this
  explicitly rather than recording it as an accepted deviation.
- **O**: declare the reduction order. The RMS `sum` is thread-strided accumulation
  then a shared-memory tree reduction, which is one specific FP32 order. The
  Qwen evidence records its own unrounded FP32 sum order. Either match the
  declared order or declare a distinct numerical profile; do not claim bit
  equality across different summation orders.

### D. Nonfinite / status / fence / read / publication — V, C (one real gap)

- **Confirmed gap (C)** in `zg_ot_rms_v1`, `zg_ot_residual_rms_v1` and the head-RMS
  macro: inputs, `gamma`, `inverse` and `y` are checked, but the reduction result
  is not. Finite inputs whose squares overflow (`x*x`, or the accumulated `sum`)
  give `sum = +inf`, `rsqrtf(inf) = 0`, hence `inverse = 0` (finite) and `y = 0`
  (finite), so status stays 0 and the silent zero row is accepted. Fix by
  flagging `!isfinite(sum)` (and ideally `inverse == 0`) as a status fault.
- Greedy (`zg_ot_greedy_v1`) does flag any nonfinite logit and `continue`s, so a
  poisoned vocabulary fails closed.
- Status is read only after `complete_and_read`; nothing publishes before
  completion.
- `C`: `StatusProtocol::{cleared, submitted, fenced_read}` is internal to one
  submit call and therefore enforces nothing beyond the caller trait. There is no
  production `CallerStatusContract` implementation in this tree, so the shipped
  code contains no fence/read path at all — only inert scaffolding. Do not
  describe it as a working completion contract.

### E. Pointer ABI correspondence — V, partial

Order and count matched for positions, rope, embedding, attention, head RMS,
sigmoid, multiply, silu_up, residual, kv_append, greedy and the private attention.

- **O**: confirm `diagnose_parameter_layout`'s `#[cfg]` compatibility. It calls
  `StaticScalarSchema::byte_size()`, which is `#[cfg(test)]` in the base tree; the
  branch must widen it to `any(test, feature = "cuda-native-qualification")`.
  Likely present because qualification Clippy passed, but verify in the blob.
- **O**: confirm `used_bytes == expected.bytes` exact-size checks and the
  `overlap`/`same_range` alias rules: only exact same-range aliasing is allowed;
  partial overlap is rejected; and Read+Read, Write+Write and cross-role aliases
  are all rejected.

### F. Authority and execution claim — V, C

- `Definition::prepare`, `Program::{…}` execution paths and
  `CallerStatusContract` are `pub(crate)`; `BufferRole`/`Access`/`BufferBinding`
  are inert plain data.
- `Program::contract` and `greedy_host_semantics` compute plans only.
- `zerogravity::qwen35_cuda::require_ordinary_invocation_capability` is `pub` but
  always returns `Err`: `ordinary::plan_primitives` routes Projection /
  GatedDelta / FlashAttention to missing, and `ordinary::require_binding`
  unconditionally appends the physical-wiring refusal.
- `require_ordinary_capability()` in `main` still refuses before discovery.
- **O**: grep every call site for any path reaching `launch_on`.
- **C**: `main`'s refusal text is
  `"Qwen CUDA has no ordinary dynamic GDN implementation; finite captured
  execution requires the explicit qwen35-cuda-qualification diagnostic feature"`.
  After `fe228ab`/`52bfb51` this wording is stale: a parameterized dynamic GDN
  runtime now exists. The branch's replacement text also overstates, because it
  calls dynamic GDN incomplete. Refresh the refusal to name exactly what is
  missing — source-bound non-GDN projection/attention lowering and physical
  queue/flight/completion/`StateRoot` ownership — while keeping the refusal
  itself before allocation.

### G. Lowering-audit weakness (new) — V

`crates/zerogravity/src/qwen35_cuda/ordinary.rs` encodes the expected
`KvAppend`/`QkRmsRotaryGated` packed `TensorView` with hard-coded `rows: 64`,
`offset: 4608`, `row_stride: 5120` instead of comparing against
`plan.max_chunk_rows()`. It validates `epsilon == 1e-6` but never validates the
RoPE theta, even though the inverse-frequency buffer is an input. Both fail
closed, so there is no safety issue, but the "source-plan lowering audit" claim is
weaker than it reads. Either derive these from the plan or narrow the claim.

### H. Untested feature matrix — O

See §3. No command covered plain `qwen35-cuda`, default, or `--all-features`.

## 5. Working position at interruption

Integration of `29b491d` as **unselected primitives with no backend or launch
authority** looks acceptable, conditional on:

1. closing the RMS/head-RMS reduction-overflow status gap (§4.D);
2. completing the `SiluUp` scalar-word-order check (§4.B);
3. writing the `byte_size()` `#[cfg]`, alias-rule and `ordinary.rs` findings into
   the record (§4.E, §4.G);
4. running the untested feature matrix (§4.H);
5. refreshing the stale refusal wording in `qwen35_cuda.rs` so it truthfully
   names non-GDN lowering plus physical ownership, not "no dynamic GDN";
6. making no backend, launch, numerical or performance claim.

Nothing in the candidate removes or weakens the ordinary refusal, and the
qualification gates are unchanged.

## 6. Housekeeping

- The branch ref must be preserved until accepted or explicitly rejected.
- No `pi-web-agent-*` worktree currently exists; `git worktree list` shows only
  `main` and the preserved `qwen35-prompt-reference-20261026` extraction overlay.
- A prior draft commit `68cbef5` (synthetic queue test) stays unintegrated.

## 7. Resumed independent review, closure and integration

The interrupted review was resumed from the recorded source state. The original
`/tmp/rev-29b491d` materialization was verified, but dotlink's per-command
sandbox makes `/tmp` ephemeral; after pruning only that stale registration the
same commit was rematerialized at
`/home/pvl/spaces/abird/src/.zerogravity-worktrees/rev-29b491d` and attached to
`pi-web-agent-8026e968-eb45-46fd-a2a1-d669a232fe6b`. No conclusion below relies
on the interrupted reviewer's verdict.

### A. Independently re-derived findings

- **Authority remains absent.** There is no `CallerStatusContract`
  implementation for ordinary text, no production ordinary primitive selector,
  and no call site that can turn the new definitions into an ordinary backend.
  Run/serve still call `require_ordinary_capability()` and refuse before source,
  device or driver discovery. The only ordinary `launch_on` remains inside the
  crate-private `PreparedLaunch::submit`; the unqualified online-FP32 attention
  primitive remains absent from selectable definitions.
- **`SiluUp` scalar ABI is correct.** CUDA consumes
  `[rows, gate_offset, up_offset, row_stride]`; Rust emits
  `[rows, 0, intermediate_width, 2 * intermediate_width]`. The source-derived
  packed-plan test covers both admitted geometries and every runtime row 1..64.
- **`StaticScalarSchema::byte_size()` widening is correct.** It is exposed only
  under `cfg(any(test, feature = "cuda-native-qualification"))`, exactly where
  CUDA formal-parameter reflection needs it; it is not widened into ordinary
  production API surface.
- **Alias validation is fail-closed.** Every overlapping pair is checked
  regardless of access mode. Only an explicitly listed role pair with the exact
  same allocation/range may alias. Partial overlap is rejected even for an
  allowed pair. New controls also reject read/read and write/write overlap.
- **The lowering audit had two additional source-identity weaknesses.**
  `KvAppend` hard-coded the 64-row view instead of the invocation row count, and
  QK/RoPE authenticated epsilon but not the declared `theta=10_000_000`. Both
  were fail-closed but made the audit weaker than claimed.

### B. Review fixes

Signed branch commit `b6aae7fe1656c249d89af76e2fb7d469bd328977`
(`fix(cuda): close ordinary primitive review gaps`) closes the review blockers:

1. RMS, residual-RMS and Q/K head-RMS now set nonfinite status if the completed
   sum-of-squares reduction is nonfinite, and also if the computed inverse is
   nonfinite or exactly zero. A finite input whose square/reduction overflows can
   therefore no longer produce a silently accepted zero row; publication still
   depends on the existing fenced status protocol.
2. The Qwen audit derives the packed KV view from geometry plus the actual
   invocation row count, and authenticates the exact RoPE theta.
3. Dynamic GDN is no longer described as missing. It stays owned by the existing
   runtime schedule; the audit names only source-bound **non-GDN**
   projection/attention lowering plus physical
   queue/flight/completion/`StateRoot` ownership as incomplete.
4. Ordinary refusal text now names that same real gap and still refuses before
   allocation.
5. The CUDA source no longer depends on host `<math.h>`/`<stdint.h>` or the host
   `INFINITY` macro. It uses the admitted CUDA runtime headers and constructs
   IEEE negative infinity from exact bits, which is accepted by the existing
   Nix CUDA/NVRTC environment.

### C. Independent verification

All commands were run from the reviewed tree with locked/offline Cargo where
applicable:

- `cargo test --locked --offline -p zerogravity-gpu --features cuda-native ordinary_text::tests --lib`
  — **12 passed, 0 failed**.
- `cargo test --locked --offline -p zerogravity --no-default-features --features qwen35-cuda qwen35_cuda::ordinary --lib`
  — **3 passed, 0 failed**.
- `cargo clippy --locked --offline -p zerogravity-gpu --features cuda-native --lib -- -D warnings`
  — **passed**.
- `cargo clippy --locked --offline -p zerogravity --features qwen35-cuda --lib -- -D warnings`
  — **passed**.
- Plain `cargo check --locked --offline -p zerogravity --features qwen35-cuda`
  — **passed**.
- Default `cargo check --locked --offline -p zerogravity` — **passed**.
- `cargo check --locked --offline -p zerogravity --all-features` — **passed**;
  only three pre-existing dead-code warnings outside this candidate were emitted.
- `cargo fmt --all -- --check` and `git diff --check` — **passed**.
- SM86 CUDA formal compile/load reflection under the already accepted local CUDA
  12.9 include view and pinned Nix NVRTC/driver libraries — **1 passed, 0
  failed**, with no compute process before or after. The first attempt correctly
  refused for absent `CUDA_PATH`; after using the existing admitted SDK view,
  reflection exposed the host-header/`INFINITY` portability defects above.
  Those were fixed in-source; no SDK download, host permission change or fallback
  compiler path was used.

The reflection result is compilation/ABI evidence only. It is **not** ordinary
kernel launch, numerical inference, model support or performance evidence.

### D. Verdict and main integration

**Accepted only as unselected primitives with no backend or launch authority.**
The remaining ordinary gap is source-bound non-GDN projection/attention lowering
through the existing physical queue/flight/completion/quarantine owner and sole
`StateRoot` publication path.

The reviewed four-commit series was integrated onto documentation-only main
`901d23e` as signed cherry-picks:

- `ea66b3a` — `Add dynamic CUDA ordinary text primitives`
- `68393e5` — `fix(cuda): constrain ordinary text candidate`
- `6b4a2ec` — `fix(cuda): close ordinary Qwen review blockers`
- `441fa64` — `fix(cuda): close ordinary primitive review gaps`

The integrated owned source is byte-for-byte identical to reviewed branch tip
`b6aae7f`. No root README, Fabric vNext path, extraction overlay, model/source
identity, captured recipe/table, ordinary serving route, numerical permit or
performance claim was changed or granted.
