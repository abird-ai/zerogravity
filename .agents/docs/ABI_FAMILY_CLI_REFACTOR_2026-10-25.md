# Three-seam refactor: kernel ABI, Qwen family, local run UX — 2026-10-25

Parent checkpoint: `5229e53`. Scope: contained implementation only; unchanged root README, model binaries, trained weights, device settings and numerical acceptance limits. Original source, ResourceArbiter, completion quarantine and StateRoot publisher remain authoritative.

## Design decisions

### Single kernel ABI without hot-path allocation

`crates/zerogravity-gpu/src/resident_state_transport/typed/abi.rs` owns the finite, typed B4 CUDA binding role order. One `Abi` carries:
- family entrypoint, up to 30 original pointer arguments in a checked stack array (not a fresh per-token `Vec`);
- each role's exact byte extent and read/write access;
- the mapping to the original checked buffer handle for all four rows.

`gdn_definition` and `kv_definition` derive their `PinnedKernel` metadata from this table. `PreparedTypedTransport::prepare` binds actual handles **from the identical table**, validates service/byte bounds/disjointness/shared KV source, and verifies the cold native entry+entire extent/access signature **before launch**. The source-only native-signature regression parses both literal CUDA `.cu` pointer parameter lists, including KV import/export preprocessor differences, and compares exact names/types/order to the table. Thus three former independent Rust descriptions are one, with a separate CUDA source guard. No new compiler, IR, registry or kernel rewrite; original CUDA text/source hashes/launch math unchanged.

### Family-specific semantics without changing stable type identity

Exact Qwen3.5 `full_model_profile`, `short_hf`, `vllm_model` and `vllm_native` source and tests are grouped under `program/family_qwen35/`. `program.rs` uses explicit `#[path]` module declarations to **preserve the original Rust canonical type paths** (and associated type IDs); existing imports and closed model-profile names remain intact. Generic `ForwardPlan`, `TextCallGeometry`, tensors, resource/queue ownership and model-independent operations stay where they are. This is a source ownership boundary and maintenance improvement, not a claim that every Qwen descriptor has been removed from core. Future families supply their source-verified mathematical adapters instead of enlarging these Qwen recipes.

### One model/device selection policy, simpler spelling

Existing `zerogravity run MODEL` stays canonical. A single `-d`/`--device` controls normal hardware selection: omitted for automatic selection, `-d cpu` to select CPU, and `-d gpu:0` or `-d pci:...` to constrain the GPU. The redundant `--on` alias was removed in a follow-up review. `--backend` is retained only as an advanced override when specifying the execution technology (e.g. CUDA versus Vulkan on the same physical GPU), using the same typed qualified routing policy. `-d cpu` or `--device local/cpu` normalizes to `cpu:0`; GPU selection still needs an enumerated index or a physical PCI identity, no ambiguous GPU magic. `devices list` shows human-readable total/available MiB or `unknown`, and explicitly labels inventory as observation, **never model execution qualification**. Help leads with two concise run forms; advanced flags remain available. Existing source/cache identity, config provenance, pre-client admission, forced GPU refusal and `--dry-run` stay on the original `run_policy`/CLI path. No downloader, routing engine or silent CPU fallback added.

Example commands:
```sh
zerogravity run ./model.gguf --prompt 'Hello'
zerogravity run ./model.gguf -d cpu --prompt 'Hello'
zerogravity run ./model.gguf -d pci:0000:01:00.0 --backend cuda --explain
zerogravity run ./config.yaml --dry-run
zerogravity devices list
```
Unsupported source/backend tuples continue to return explicit refusals.

## Verified on the assembled source

Evidence: `.agents/runs/abi-family-cli-refactor-20261025/`.

- Core family and resource semantics: `cargo test -p zerogravity-core --lib` **208/208 PASS**.
- CLI normalization/aliases/config/source priority: `cargo test -p zerogravity-cli --bin zerogravity run_cli::tests` **10/10 PASS**; equivalent advanced/concise flags, contradiction refusal, unknown-inventory formatting.
- GPU typed ABI: `cargo test -p zerogravity-gpu --no-default-features --features vllm-cuda-qualification --lib resident_state_transport::typed::` **7/7 PASS**; includes CUDA native signature guards, wrong entry/access refusal, existing byte/extents and disjoint source tests.
- HIP-only, Vulkan-only, CUDA-only source qualification build boundaries, Gemma HIP controls and Vulkan schema tests **PASS**.
- Combined CUDA/HIP/Vulkan + qualified fault/checkpoint feature set `cargo check` **PASS**.
- Real CLI `run --help` **rc0**; read-only `devices list` **rc0**, nonqualified GPU inventories shown without claiming readiness.
- Full CPU/workspace `scripts/check.sh` attempted twice under mandatory bounded command lifetimes: **rc124 timeout** after 113s and 112s; second progressed through optional artifact/CLI suites into 4-case actual CPU `run_unified` (2 cases PASS before cutoff). No observed assertion failure. **The whole gate has NOT completed** and must not be marked passing. Receipts: `full-cpu-gate*`. A separately supervised long-running owner is required for full acceptance.
- Root `README.md` SHA256 unchanged: `6c00873ec0ac2c59bc73703b55a7bfa4d93703334a03fa16bcac2d204cda5b15`.

## Deliberately still open

- The strengthened separate-fixture grouped B4 physical oracle and the model-bound 26-case B4 acceptance under newly built original artifacts; no physical promotion from this source refactor.
- Optimized canonical release build and matched fresh C16 throughput / vLLM numbers (source/physical/pinning conditions unchanged).
- Qwen27B x2 capture only **after** B4/C16, retaining strict SSH trusted host-key and UD6 exit137 forensics blockers.

Implementation changes are intentionally narrow; more generic family registries, unqualified GPU auto-routing, automatic layer splitting and second runtime abstractions would be premature.
