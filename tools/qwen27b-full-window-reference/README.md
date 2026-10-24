# Original27B full text reference

Expected-only producer; no Rust/runtime authority. Uses the approved pinned HF
source and original Transformers5.17/Torch2.13 CPU code in the existing immutable
image. No download, GPU, whole-model `from_pretrained`, model dtype conversion,
replacement mathematical loop or expected-value input.

- Two independent original64-layer runs: teacher11 at empty cache, then353 using
  the same actual cache. Fixed teachers; no natural-feedback claim.
- Hook-bind67 original modules in their actual traversal: embedding,64 decoder
  layers, final RMS and full vocabulary head. All851 raw text roles; source BF16
  and F32 stay distinct. Vision/MTP is excluded explicitly.
- Meta construction only; one learned module mapped at a time, at most3GiB.
  Original kernels consume these direct source views. No full55.6GB parameter
  allocation. Per-module raw mapping SHA/version checks precede unmapping.
- Keep authentic HF Conv4, FP32 recurrent state and head-major KV.112 logical
  domains serialize128 actual tensors. Initial state is absent, never fabricated.
  Runtime minimal Conv3 is not silently relabeled as complete HF cache storage.
- Capture64 layer outputs, embedding/positions/final RMS/full248320 logits,
  all initialized before/after cache tensors and module/source events.1040 records,
  934,776,928B total;4MiB per-record and1GiB total limits precede transport.
- Full held source31-file hashes before/after original execution;18 shard FDs
  remain owned through process exit. Fresh validator independently repeats source,
  header/role/range, full finite raw coverage, state carry, repeat, library and
  owned process/container/group/EOF gates. It pins the same held metadata/raw FDs.

Limits: this is an original raw-source reference. Source roles are actually all
BF16; recurrent/math state is F32. No all-BF16 loader equivalence, initial64 chunk
qualification, Rust parity, native/serving support, performance, or solution to
the MKL-HA exp mismatch is claimed. Existing fatal
2/48 decay and5,539 state-word evidence remains unchanged. No reference pin exists.

## Commands

Cheap source-free checks:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  env PYTHONDONTWRITEBYTECODE=1 python3 tools/qwen27b-full-window-reference/test_contract.py
```

ROOT only, after reserving a heavy CPU/I/O window (not run by the author):

```sh
python3 tools/qwen27b-full-window-reference/run_reference.py --new-output /ABS/FRESH-OUTPUT
python3 tools/qwen27b-full-window-reference/validate_reference.py \
  --output /ABS/FRESH-OUTPUT --receipt /ABS/SEPARATE-FRESH-RECEIPT.json
```

Original owned lifecycle is reused unchanged.4CPU/12GiB OS limit,16GiB worker
address-space limit,2h worker deadline/3h total,4MiB retained stdout. These are
bounds, not measured capture runtime/RSS. Whole-source and per-role scans are
heavy and must be scheduled separately from benchmark timing.

## Integration

All files in this directory are additive tools; no Cargo/core/CPU/GPU/runtime
changes. Keep earlier final63/fullCPU/exp-causal donors and failures unchanged.
Before any learned replay, ROOT must execute and independently validate this
original producer. A future Rust adapter must compare source851 roles and each
actual coherent boundary; Conv4→diagnostic last3 mapping must be explicit. The
existing full64 diagnostic schedule and ordinary public refusal remain intact.
Do not turn an accepted original reference into an exp/native permission.

Source API was checked against the approved image's exact modeling/cache files
(`762feb6c...12c3b`/`702144bb...743c`): TextModel1240–1310 is the original loop,
ForCausalLM1664–1748 applies the real Head, Cache906–912 stores Conv/recurrent
owners, and DynamicCache1756–1856 constructs all source layer kinds. Positions
are the real four-axis expansion split to3; no hand-generated replacement.
