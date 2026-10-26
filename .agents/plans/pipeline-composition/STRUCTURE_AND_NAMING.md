Author: Project owner
Co-authors: GPT-6.1 Sol xhigh
Owner: @pvl; ROOT coordinates migration

# Structure and naming

26 Oct 2026. Conventions for new work and the current Qwen/Gemma refactor.
The migration status belongs in [TODO.md](TODO.md). These rules establish
organization; they grant no new numerical, model or device qualification.

Retain the [shared execution architecture](SHARED_EXECUTION_ARCHITECTURE.md)
and [correctness policy](../numerics-and-context/CORRECTNESS_POLICY.md).

## Basic rules

1. **One obvious owner.** Put a concept beside the code that defines it.
   Model architecture belongs to its model crate; a device queue belongs to its
   backend. Move callers and tests with a responsibility, not just its file.
2. **Dependencies point down.** Shared code does not import its model consumers.
   Application wiring may select models; operators and allocators may not.
3. **Names describe purpose.** Use ordinary domain terms and explicit units.
   Prompt lengths, case ordinals, development milestones and reference runtime
   names do not define model architectures or application routes.
4. **Variation is data.** Sizes, prompt/context lengths, batch sizes, supported
   shapes and device placement are checked configuration/capability fields.
   Different mathematical architectures get different family implementations;
   matching dimensions alone never establishes interchangeable behavior.
5. **Share demonstrated behavior.** Reuse the existing operators, source readers,
   admission and execution spine. Extract an abstraction at a real consumer
   boundary; do not create a crate/trait/registry for each concept or future use.
6. **Keep authority explicit.** One mathematical engine/ForwardPlan, the existing
   ResourceArbiter, original completion/fence and sole StateRoot publisher.
   Composing modules cannot introduce another allocator, scheduler or publisher.

## Ownership and dependencies

Keep the existing crates unless a concrete dependency or lifetime boundary
requires another. Directories express ownership; crates enforce dependencies.

| Owner | Contains |
|---|---|
| `zerogravity-core` | Typed semantic operators/values, source/state identities, numerical contracts, ForwardPlan, session publication and resource contracts |
| `zerogravity-artifacts` | Generic GGUF/safetensors/container metadata, authenticated source bytes/windows, snapshots, cache and bounded reads |
| `zerogravity-cpu`, `zerogravity-gpu` | Operator implementations, device compilation/ABI, physical memory, transfers, queues/events, completion, fencing and retirement |
| `zerogravity-model-qwen35` | Qwen family config, tensor roles/ties, source interpretation, exact traversal/state, tokenization, variants and specialized operation recipes |
| `zerogravity-model-gemma4` | Equivalent Gemma responsibilities, including PLE/shared-KV and variant-specific semantics |
| `zerogravity` | Shared loading/execution orchestration and one cold registration point for available families/backends |
| `zerogravity-cli` | Argument/config parsing, presentation and protocol adapters; delegates model decisions to registered adapters |
| Tests/reference tools | Independent reference adapters, cases, comparison policies and model/device qualification |

Target dependency direction (`A → B` means A uses B):

```text
CLI → application → model crates → core + artifacts
                   model crates → optional CPU/GPU implementation adapters
      application → CPU/GPU backends → core + artifacts
                    artifacts → core
```

- Core imports no model, backend or application crate. Artifact readers do not
  interpret Qwen/Gemma architecture, tensor roles or inference eligibility.
- Backends do not import model crates. Model-specific adapters may use backend
  preparation APIs after the reverse dependency is removed. Never add a cycle.
- Family-specific kernel definitions and ABI recipes stay with their model;
  shared kernels stay with the backend. Device FFI, compiler policy, actual
  source/ABI validation and physical lifetime remain backend responsibilities.
- Native source, descriptor or caller byte counts cannot mint an execution
  permit. Preserve typed authenticated source retention and validated bindings;
  `Any`, strings or dimensions alone cannot replace them.
- Registration is application wiring, not a second runtime/plugin framework.
  Model names are legitimate there, in their model crate and in integration
  tests. They do not belong in a generic allocator or operator dispatcher.

Within the GPU crate, use `cuda/`, `hip/` and `vulkan/` for provider-specific
compiler, native ABI and runtime policy. ROCm is the software stack; `hip`
names the actual API/backend. Keep common operators, admission, queues and
retirement shared. Provider namespaces remain feature-gated independently;
CUDA source/ABI and HIP source/ABI never confer permission on each other.

Use `quantization/` for GPU packed-format execution, with format-specific
submodules where code exists. Generic packed-byte decoding stays in artifacts;
CPU arithmetic stays in CPU. A precision such as BF16 is not a quantization
format. Source codecs, scale/zero-point rules, packing and accumulation contracts
remain explicit: reorganizing files cannot replace one quantization with another.
Add a crate only when it enforces an actual dependency or build boundary.
Use the [format implementation map](FORMATS.md) for Q1/Q2/Q3/Q4/Q6/Q8/NVFP4
homes, separate BF16 precision, exact implemented variants and remaining work.

## Model template

Use the same small structure in each existing model crate; create only modules
that contain real code. One family may cover several sizes sharing architecture.

```text
src/
  lib.rs              small public API
  config.rs           architecture and variant interpretation
  tensors.rs          model tensor roles, ties and shape checks
  source/             bind authenticated generic source bytes to those roles
  forward.rs          one mathematical traversal
  state.rs            family state layout; original session authority publishes
  tokenize/           tokenizer and chat/input framing, when applicable
  execution/          optional family-specific CPU/CUDA/HIP/Vulkan adapters
tests/                model contracts and fixtures
```

These are responsibilities, not mandatory empty folders or a mass file split.
An existing `text/` subtree may group text-specific responsibilities. Generic
format readers remain in artifacts. Shared numerical arithmetic remains with
operators/backends, even if Qwen was its first consumer. Shared model operations
such as projection/head streaming already exist and must be reused.

Build the same family traversal against the existing mathematical backend to
evaluate or record a ForwardPlan. Prepare source bindings, immutable topology,
finite implementations and costs once; bind request inputs/state/epochs late.
Keep family-specific source seals and capabilities next to that family's adapter.

The application adapts a prepared model once for request validation, session,
step, committed output and cancellation. Use the smallest concrete interface
that serves actual consumers. Avoid repeated family switches through HTTP,
workers and per-node execution; no vtable per scalar/tensor element.

## Naming and reports

- Modules/functions name behavior: `config`, `projection`, `prefill`, `decode`,
  `source`, `state`, `prepare`, `submit`, `complete`, `retire`.
- The primary implementation uses the operation name: `gdn`, `attention`,
  `rms_norm`. Distinct alternatives use an implementation suffix when useful:
  `gdn_vllm`, `gdn_llama`. Backend context is supplied by the enclosing module,
  e.g. `cuda::gdn`; do not repeat it in every name. Reference adapters and
  numerical-contract metadata explicitly identify their actual upstream/version.
  Renaming never merges different reduction, cast, rounding or state contracts.
- Public types are specific within their module (`qwen35::Model`, `Config`);
  avoid broad names such as `FullModelProfile` for two fixed Qwen variants.
- Use units in scalar fields: `prompt_tokens`, `cached_tokens`,
  `context_capacity_tokens`, `request_batch_size`, `generated_tokens`,
  `hidden_width`, `host_bytes`. Distinguish request batching from token rows.
- `prefill` means processing a new prompt/prompt chunk; `decode` means processing
  a generated token with an existing prefix. Reports explain these in words.
  Prompt length, consumed context and emitted output count remain separate.
- `cpu/cuda/hip/vulkan` identify execution backends. Use “ROCm (HIP)” in prose
  when that distinction helps. Source format and numerical contract are separate
  fields; neither “HF” nor “BF16” names a complete mathematical implementation.
- Keep precise upstream/version names when they identify a real numerical
  contract or compatibility implementation. Do not rename such identities to
  vague `fast`, `native`, `original` or `generic` profiles.

| Concept | Meaning | Active representation |
|---|---|---|
| Model text execution | A verified model and its prepared implementation | Model-owned adapter; supported prompt/output/context bands are explicit capability data |
| Source arithmetic | Output-first matrix layout and precisely matched numerical operations | Layout/operation modules and explicit numerical contracts |
| Cached decode agreement | Compare the same request alone and batched | `cached_decode_matches_single_request`, explicit `comparison: bit_exact` and batch size |
| Prompt processing | Fresh prompts containing 48, 49 or 50 tokens | `phase: prefill`, `prompt_tokens: 48/49/50`, `cached_tokens: 0` |
| Prompt continuation | A 47-token prompt followed by token-by-token continuation | Explicit cached/input/processed token counts per step |

<!-- Previous labels: short_hf, canonical47, original_hf, case5, initial48–50, vllm_gdn. -->

Tests name the behavior/assertion, not their order or debugging history. If a
case tests unrelated behaviors, use separately named scenarios under its shared
fixture. Keep its exact numerical comparison policy visible.

Example case data, not a new registry/schema requirement:

```yaml
check: cached_decode_matches_single_request
model: qwen3.5-0.8b
prompt_tokens: 47
request_batch_size: 4
comparison: bit_exact
```

For a fresh prompt use `phase: prefill`, `prompt_tokens: 48`, `cached_tokens: 0`.
A 47-token prompt followed by decode is a different case. The last emitted token
is not automatically consumed; report actual counts per phase.

Use only new names and explicit parameters in active code, plans and reports.
Leave succinct old-name comments where grep can connect source history; do not
retain compatibility aliases or parallel naming systems. Preserve actual source/
model hashes, failure facts and what historical binaries executed. New names
never turn a failed/interrupted scope into a pass. Check Rust type/kernel/cache
identity after moves/renames and requalify changed identities; old binary evidence
cannot be attached to new source merely because the arithmetic body is unchanged.

## Three different topologies

1. **Code dependencies:** the crate/module ownership graph above.
2. **Mathematical/task composition:** typed operator and pipeline edges, existing
   ForwardPlan and declared state effects; family modules record the math.
3. **Physical placement:** explicit devices, memory domains, streams, transfers
   and completion dependencies in the existing prepared physical plan.

Do not infer one from another. A module boundary does not require a copy or
wait. Model size/source layout does not choose a device. Several model instances
may share immutable verified weights while retaining independent session roots.
Future pipeline/sharding edges carry actual source/profile/producer bindings and
typed ports; no new generic topology framework is required by this refactor.

## Migration and verification

- Apply to Qwen .8B/2B/27B and Gemma4 together. Preserve their actual architecture,
  variants, source roles, numerical contracts, finite capabilities and physical
  owners. A model README/template is not a support declaration.
- Move family source interpretation out of artifacts; neutralize shared-core
  family descriptors; then remove GPU→model dependencies before enabling the
  reverse optional adapter edges. Update real consumers in coherent units.
- Retain the existing source lifecycle, complete costs, cancellation/unknown
  behavior, one physical execution loop and original publication path.
- Use the common correctness workflow for all families: identity/semantic/state
  invariants, declared operator accuracy, independent model/task comparisons and
  separate determinism. Keep every failed/unsupported/unexecuted cell explicit.
- Source/compile/feature checks cover all migrated families. User26Oct waives
  full model suites for Qwen27B/Gemma4 during refactoring; this grants no fresh
  inference qualification. Required affected .8B/2B checks and the final assembled
  `scripts/check.sh` remain distinct from those unrun model suites.
- Enforce architecture with Cargo dependencies, small boundary checks and
  extension drills. Adding a family should touch its crate, one registration
  entry and its tests; adding a shape/kernel should touch its responsible
  operator/adapter. Shared code changes only for genuinely new shared semantics.
