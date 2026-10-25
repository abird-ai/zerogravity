# Fresh source/design/correctness review — 2026-10-25

Scope: `main` at entry `3d9866d` (27 signed commits ahead of `origin/main`), especially changes since `4bf3673`. Corrective implementation landed as signed `a39ca96` (grouped GPU) and `7362699` (serving). Review performed through `dotlink---l5`. Examined `AGENTS.md`, active plans, the existing [shared architecture](../plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md), resource arbiter, CPU/GPU serving router, request queue, HTTP framing, GPU scheduler, B4 grouped transport Rust/CUDA ABI, and the model-free physical oracle. This is a **bounded code/design review**, not a proof of every model, shader, cancellation edge, or supported platform. Source and actual results outrank historical handoffs. No full build, weight download, model inference, x2 access, GPU physical re-run, C16 timing, push, release or history rewrite occurred in this review.

## Verdict

The fundamental separation is worth keeping: family-authenticated mathematical `ForwardPlan` → prepared physical operations → one owned submission/event/fence → genuine Ready → independent `SessionAuthority` publication. One `ResourceArbiter` per real physical domain (process-shared), immutable source, resource-credit/backpressure, composable typed state transport, and closed qualified backend selection are the correct primitives. **Do not introduce another global scheduler, second state authority, model-specific execution fabric or untyped universal task abstraction.**

The review found **four actionable correctness problems** and an unnecessarily expensive oracle allocation. The contained fixes below preserve public API, source arithmetic, capacity limits, kernel source, and explicit device fences. Old physical evidence must **not** be promoted to the changed source without new physical receipts.

## Confirmed bugs and changes

| Priority | Source / problem | Fix | Acceptance still required |
| --- | --- | --- | --- |
| P0 | `resident_program/native_cohort/transport_qualification.rs`: per-row reference and grouped implementation ran sequentially on **the same already-mutated GPU buffers**. A no-op/incomplete grouped write could inherit the reference's exact output and pass its comparison. | Run two complete **independently initialized and retired** `qualify_mode` flights, each checked word-by-word against the same deterministic host oracle. Require source, owner and original physical reservations fully retired between them; only return matching case counts and the actual reduced launch population. | Fresh pinned real RTX3060 oracle, one isolated release-profile physical owner, including final device0 and **both** mode receipts. Historical grouped physical pass under `03f2e66` is not current-source acceptance. |
| P0 | `resident_state_transport/typed.rs`: grouped KV `prepare` bound **only `rows[0].native`**; `rows[1..].native` could differ in memory, byte view or stream without refusal. | `require_shared_kv_cache` validates full `BufferRange` (same physical service, memory identity, start/end) and stream of every row before any launch; focused test rejects different allocation, truncated view, different stream. | Test target type-check complete, negative unit test still needs actual execution in correct GPU-feature unit build. GPU requalification before serving promotion. |
| P1 | Same oracle callback retained a second `RefCell<Vec<Vec<u8>>>` copy of every readback and a cloned `fixture.expected`, then discarded them; not required for any assertion, inflating transient unaccounted host memory. | Verify against borrowed expected slices in-place; remove `seen` and the extra clone. | Fresh physical run to retain complete byte-coverage proof. |
| P1 | `serve/pending.rs`: a cancellation could arrive **after `reserve` but before `enqueue`**; `sweep` on empty pending queue consumed its hint. Later enqueued cancelled item escaped compaction until another hint. Counter could also theoretically wrap to zero. | A boolean notification flag, preserve it through an empty sweep, retain first nonempty compaction and existing outside-lock payload destruction. No new queue or per-request allocation. | 20/20 pending-queue tests PASS; new exact reserve→cancel→empty sweep→enqueue→sweep regression PASS. |
| P1 | `serve/gpu_batch.rs`: decode at frontier512 + 1 token passed the per-member capacity check (prefill had a full-span check). | Unified `checked_add(frontier,tokens) <= 512` for **both** phases. Preserve first-seen grouping and original per-source credit. | 14/14 GPU-batch tests PASS including frontier511 success and 512/overflow refusals. |

**No GPU kernel arithmetic or source signatures were widened**, and no accepted unknown/submitted recovery was converted to fallback. Input/model/source identity and terminal physical ownership remain mandatory.

## Focused verification run on changed code (no full build)

Evidence root `.agents/runs/architecture-review-20261025/`.

- `cargo check --locked --offline -p zerogravity-gpu --no-default-features --features vllm-cuda-qualification --tests`: **PASS** on revised GPU/fixture and negative cache-view test (`review-cuda-tests-fixed.rc=0`). A first compile caught a test-local Rust closure lifetime, fixed by a typed helper function; no runtime product regression.
- `cargo test --locked --offline -p zerogravity-cli --features serve --bin zerogravity serve::pending::tests`: **20/20 PASS** (`pending-tests.rc=0`).
- `cargo test --locked --offline -p zerogravity-cli --features serve,cuda --bin zerogravity serve::gpu_batch::tests`: **14/14 PASS** after final refinement (`gpu-batch-tests-final.rc=0`).
- `cargo test --locked --offline -p zerogravity-cli --features serve --bin zerogravity serve::http::tests`: **8/8 PASS** (`http-tests.rc=0`); no HTTP code change.
- `cargo check --locked --offline -p zerogravity-cli --features serve,cuda --tests`: **PASS** (`gpu-scheduler-typecheck-fixed.rc=0`).
- `bash scripts/check-gpu-feature-boundaries.sh`: **PASS**, HIP-only, Vulkan-only and CUDA-only compilation, 5 Gemma controls and 2 Vulkan report tests (`gpu-feature-boundaries.rc=0`).
- CUDA+HIP+Vulkan+qualification+hardware-abort+portable checkpoint **union** compile `cargo check`: **PASS** (`all-gpu-union.rc=0`).
- `cargo fmt --all -- --check` and `git diff --check`: PASS.
- Warnings: unused GPU probe methods / unused `bf16_bands` / GPU route variables in limited feature combinations; not new correctness failures. No full `scripts/check.sh`, optimized release build, GPU model tests or benchmark. These remain required when physical access/build window is restored.

## Maintainability and composition review: concrete prioritized direction

### Keep the existing four responsibilities

1. **Primitive:** verified immutable source/typed values; registered physical memory, checked extents and lease generations; device events/fences.
2. **Semantic family operation:** source-specific matrix/numerical contract, typed state effects, independent frontier; no hardware queue ownership.
3. **Prepared implementation:** finite backend/kernel/liveness/bindings, source/ABI/shape/admitted resource identity, optional fused region; never a second math or state authority.
4. **Serving/pipeline:** shared fair admission queue, request credits, model/session route and HTTP exposure; real producer completion and publication only. Source-specific tokenizer/chat framing stays in adapters.

These are *responsibility boundaries*, not a recommendation to add four new crates, trait objects per tensor, or a generic runtime per modality. The current architecture plan already selects one `ForwardPlan`, and this review agrees.

### Next architectural changes, in impact order (review proposals, NOT applied now)

1. **Make eligibility declarative:** one typed immutable prepared `KernelContract` / operation descriptor to keep CUDA ABI binding order, byte extents, access modes and backend/source policy adjacent. Rust `PinnedKernel` metadata and CUDA signatures currently require manual synchronization; the prior wrong base/end/block order proves the risk. Retain literal source signature tests, not unchecked codegen or generic shader relabeling.
2. **Keep shared primitives model-neutral:** core still has Qwen-family geometry/profile names; specialize verified `FamilySource` / `TextCallGeometry` in Qwen/Gemma modules and expose ordinary typed `ForwardPlan` operations. Never merge arithmetic by approximate dtype alone: FMA, reduction/rounding, state shape and source identity remain exact separate contracts.
3. **Keep GPU serving adapters finite:** route `Gguf`, canonical47 and short-HF through one prepared model execution interface where a **second real family** demonstrates an interchange, preserving original source session/attempt/publisher. Do not start an untyped TaskDriver, second scheduler or visitor trait hierarchy just for future hypothetical modalities.
4. **Simplify user configuration without relaxing certification:** stable ergonomic `run/serve --model …` / clear device-auto selection by *actually qualified* capabilities; keep the current exact native/cache/context flags under an advanced or benchmark profile. No implicit HIP/Vulkan/large-model permit from compiled features, and no fallback after submitted unknown work.
5. **Reduce build/qualification maintenance overhead:** isolate the large optional CubeCL persistence/`turso_core` dependency and vendor modifications from source-only or CI gates where feasible, with locked identities and no altered runtime compiler semantics. Canonical optimized thin-LTO/codegen1 B4 builds exceed short connector limits; keep a fast diagnostic profile explicitly separate from acceptance/performance binaries.
6. **Separate cross-process VRAM policy from process-local arbiter:** `ResourceArbiter::process_shared()` coordinates owners **within one process**, not all processes using the same PCI device. A distributed router must add authenticated device-level admission/arbitration rather than assume process-local free VRAM proves global exclusivity.
7. **Retain strict negative tests and narrow receipts:** source-free negative shape/alias/fence/cancel tests, model-free physical checks, original full-model/source/state comparison, and matched C16 each grant **different** eligibility. Record small structured evidence rather than adding another manually synchronized handoff or parallel allocator.

## Before next full build or task

- **Fresh B4 physical oracle under this revised source is mandatory**: no prior two-flights-on-one-fixture result is sufficient. Original N1 finite301 acceptance is historical on its unchanged function; artifact provenance must still be checked with the assembled binary.
- **Step 3:** source-bound model tests remain blocked from this connector by read permissions on the actual `/home/pvl/.cache/abird/zerogravity/` .8B/2B snapshots; original 4344-record preflight also timed out. No weight substitution.
- **Step 4:** optimized fresh matched C16 (same-attempt source/profile/stock) stays blocked until full B4 model/physical gate, correct optimized binary and original input receipts. No claimed speedups.
- **Step 2:** Qwen27B x2 natural capture follows 3/4 as requested, still requires trusted strict SSH host-key access and resolution of UD6 exit137; no bypass.

**Review sign-off is limited to design, code-type correctness, scoped test contracts and feature compilation. Full-build, real physical, trained model and performance sign-offs remain OPEN.**
