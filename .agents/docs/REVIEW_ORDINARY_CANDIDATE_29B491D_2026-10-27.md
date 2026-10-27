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
