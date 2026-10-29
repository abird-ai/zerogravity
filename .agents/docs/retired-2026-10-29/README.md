# Retired artifacts — 2026-10-29

Cleanup pass on `pvl-l5`. Nothing here is load-bearing for `cargo build` or
`cargo test`; every removal was checked for references first.

Links elsewhere in `.agents/**` that point into `zerogravity-worktrees/…` or at
`examples/qwen27b_{full_window,output_tail,attention_tail}` are now dead links.
They live in dated hand-offs and research records and were deliberately left
untouched — treat them as historical text, not as current paths.

## Removed

| What | Size | Why safe |
|---|---|---|
| `zerogravity-worktrees/` (5 trees + hidden dirs) | 17 G | plain copies with no `.git`; a blob-level audit showed no content absent from `main` (one file was a strict subset). The 4 runtime-read reference JSONs were the only live data |
| Reference replay harness: `crates/zerogravity-model-qwen35/examples/qwen27b_{full_window,output_tail,attention_tail}*` incl. 5 `_pin.rs`, plus the 3 `[[example]]` blocks in that crate’s `Cargo.toml` | 24 tracked files | not built by default, not run by `scripts/check.sh`; no other crate imported the modules |
| `crates/zerogravity-gpu/src/cuda/gdn/{fixed_prompt,qwen2b}-pins.json` | ~50 KB | records only — no reader anywhere in `crates/`, `tools/`, `scripts/` |
| `*.agents/runs` `target*` dirs (main repo + worktrees) | 325 G apparent | cargo build output |
| 228 agent runs dated 2026-10-24 | 16 G | see `pruned-runs-20261024.txt` |
| Two HF snapshot tarballs in `.preserved-nine-20261024/` | 63 G | all 39/39 files verified present in `~/.cache/abird/zerogravity/models/blobs/sha256/` at the same HF revisions |
| 36 git branches (14 merged + 22 whose patch-ids all exist in `main`) | — | see `pruned-branches-20261029.txt`; 7 branches with unique patches were kept |

## Not removed but worth knowing

- `zerogravity/.agents/runs/` still holds ~116 G of capture evidence and is linked
  from ~144 tracked docs.
- The three `tools/qwen27b-*/[full|prefix]-reference-pin.json` records still carry
  absolute paths into the retired worktrees. They are **inputs** to
  `validate_reference.py` (accepted anchor) and are listed in `contract.py:FILES`,
  so they were kept. Re-sealing after a fresh capture overwrites them.

## Resuming parity work

1. **Capture** with the independent implementation:
   `tools/qwen27b-initial-core-reference/{capture.py, run_reference.py, archive.py, validate_reference.py}`
   and the `qwen27b-layer9-prefix-reference/` twin. Producer identity is pinned in
   `contract.py`: image `sha256:f57d77f5…` (digest `eec72f47…`), `MODEL_SHA`,
   `CACHE_SHA`, `TORCH_GIT cf30153c…`, `REGISTRY_SHA`.
2. **Weights** are content-addressed in `~/.cache/abird/zerogravity/models`
   (`zerogravity.cache.snapshot.v1` manifests carry `repository`, `revision` and
   per-file `sha256`). Present: `Qwen/Qwen3.8-27B` rev `1d4bf0f2…`,
   `google/gemma-4-E2B-it` rev `3e22461f…`, `Qwen/Qwen3.5-2B`.
3. **Re-seal** by re-running the capture + validation, which rewrites the
   `*-reference-pin.json` records and retires the stale absolute paths.
4. **Compare** with a script, not a compile-time-sealed example: address fixtures
   by `{repository, revision, sha256}` (or commit them — they are 13–30 KB) and
   report bitwise/tolerance tables. This is the pattern `source-pin.json` +
   `--source ABS` already uses for model inputs.

## `.agents/runs` purge (same day)

All 260 run directories were deleted (~116 G), leaving only
`qwen27b-evidence-hardening-review-20261028` (915 M) which was being written by a
live agent at the time.

This directory was **never part of the repository** — `.gitignore:3` ignores
`/.agents/runs/*` and no file under it has ever been tracked, so it could not be
cloned, reviewed or CI-tested. It was a machine-local evidence store.

Consequences, for the record:

- **2493 cited paths** across tracked docs now point at nothing
  (`runs-cited-paths-20261029.txt`, `runs-inventory-20261029.txt` keep the
  shape and sizes for future readers).
- The four fixture dirs read by `#[ignore = "requires pinned retained original 2B
  captures; ROOT runs serialized"]` tests are gone:
  `numerics-and-context-{2b-,}{bulk-causality,av-replay,2b-alpha-replay}-20261020`
  (711 M + 367 M + 38 M + 3.6 M), used by
  `crates/zerogravity-model-qwen35/tests/bulk_{f32_source,gdn_coefficients,gdn_gated,gdn_recurrent,gdn_ssm_out,gdn_upstream,gguf_prefix}.rs`
  and `crates/zerogravity-cpu/tests/{haswell_bulk_source,zen_blas_source}.rs`.
  Those tests never ran in normal `cargo test` (already `#[ignore]`) and could
  never pass on any other machine; re-running them now requires re-capturing the
  original 2B captures (see "Resuming parity work" — same producer identity).
- Raw tensor/kernel payload that was deleted: `.bin` 64 G / 9,421 files,
  `.cubin` 15.6 G / 36,180, extensionless 9.1 G, `.12` blobs 8 G, `.f32le` 4.3 G /
  55,839, plus 61,411 `.json` summaries (3.8 G).
