# Original27B final63 FFN → complete vocabulary

ROOT-only learned reference and replay of the existing shared CPU engine. This
unit starts from four **external BF16 post-attention boundary probes**, grouped
1+3 independent rows. Each original and Rust call has one row, so this does not
introduce a batched GEMM numerical profile or claim a trained final-layer input.
No AutoModel, model loop, StateRoot, GPU, serving or performance permit.

The six genuine roles are final63 FFN norm/gate/up/down, final model norm and
untied full248320 head. They are selected from approved shards17/16/18 by the
actual pinned index/header. The public pinned Qwen MLP runs normally; read-only
hooks retain its actual gate/up/Down-input/branch tensors. Final residual, norm
and full head are computed in that same sequence. Both genuine repetitions are
archived and compared for each input and all eight computed stages:72 complete
records,2,609,152 BF16 words /5,218,304B. Earlier layer0 references are not inputs
or expected outputs for this unit.

## ROOT execution

Use the existing direct owned-container launcher procedure (nested Podman in
`native-gpu.py` changes its user namespace). Source files are opened with
NOFOLLOW, complete hashes and same held descriptors before/after; immutable
approved image, network-none/read-only/cap-drop,4CPU/12GiB and bounded owned
TERM/KILL-before-reap/EOF cleanup. No pull or download.

```sh
python3 tools/qwen27b-output-tail-reference/run_reference.py \
  --new-output /ABS/NEW_CAPTURE
python3 tools/qwen27b-output-tail-reference/validate_reference.py \
  --output /ABS/NEW_CAPTURE --receipt /ABS/NEW_SEPARATE_REFERENCE.json
```

After independent ROOT validation, the actual expected-only identity is sealed
in `examples/qwen27b_output_tail/reference_pin.rs`. Actual capture and release
replay passed all2,609,152 BF16 words, source/lifetime checks and finalhost0; see
`.agents/runs/model-execution-qwen27b-output-tail-root-20261022/TODO.md`. This
qualifies only the external-boundary tail described above. Build/replay serially
with existing approved offline Cargo inputs:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo run --locked --offline --release -p zerogravity-model-qwen35 \
    --features cpu,original-hf-binding --example qwen27b_output_tail -- \
    --source /ABS/APPROVED_HF_SNAPSHOT_FILES --report /ABS/FRESH_REPLAY.json
```

The Rust example calls `OriginalWindowOperators::execute_output_tail`; no new
mathematics. It opens the full18-shard indexed source through the existing64MiB
metadata/4MiB scan source-window factory, in an actual128MiB host arbiter. Its
2MiB control reservation and512KiB sequential expected read lease are separate
from each existing source-owned math output/scratch admission. Complete learned
matrices are never packed into host Vecs. Each synchronous computation returns
before expected stage bytes are read. Input bytes are explicit leaf operands;
full reference SHA scans establish identity, not producer authority.

Replay includes pre-cancel/real exhausted-pool refusal, all72 defined streams,
full source SHA after computation and after caller owners drop, retained complete
last output, immutable reference FD, and final actual host reservation0/unfenced.
A mismatch reports case/repeat/stage/first word and bits, without a tensor dump.

## CPU fixtures only

```sh
python3 scripts/native-gpu.py -- bash -c \
 'cd tools/qwen27b-output-tail-reference && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_contract.py'
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo test --locked --offline -p zerogravity-model-qwen35 \
    --features cpu,original-hf-binding --example qwen27b_output_tail
```

These fixtures open tiny fabricated files and metadata only. They do not import
Torch or read learned/native payloads. Actual learned capture/replay remains
ROOT-only, subject to shared compiler/model resource scheduling.
