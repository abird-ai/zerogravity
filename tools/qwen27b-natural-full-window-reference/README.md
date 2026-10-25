# qwen27b natural full-window reference

Narrow derivative of `tools/qwen27b-full-window-reference/`. It recomputes the
original Qwen27B full-window trajectory for **two natural positions per cohort**
instead of the retired teacher-forcing pair, and it is the only producer of the
`accepted-original-natural-full-window-1.json` receipt.

## Why

The runtime natural two-position path feeds its own selected token back. The
approved teacher-forcing reference (`[11, 353]`) cannot validate that path: its
call1 state follows token 353, which the natural run never produces. Its call1
logits are therefore not a prediction for the natural continuation. The natural
reference is a genuinely new trajectory: seed 11, then the original model's own
call0 argmax. No expected value, Rust-selected id or teacher call1 state reaches
this producer.

## Derivative scope (nothing else changed)

Unchanged: original image/digest, model/snapshot pins, source registry, 851
source roles, 64 layers, full HF conv4 width 4, raw-source-assign-no-cast
parameter mode, mixed BF16/F32 source codecs, 112 cache domains, record/archive
caps, container argv/mounts/limits, TERM/KILL/EOF lifetime policy.

Changed (natural only):

- `contract.py`
  - `scope()`: `teacher_forcing_only: false`, `natural_feedback_qualified: true`.
  - `specs()`: schema `zg.original27b-natural-full-window-inputs.v1`; requires
    `seed=11`, `calls=2`, `capacity=2`, `repetitions=2`,
    `independent_call1_input=call0-original-argmax`; rejects any `teachers` key.
  - `record_specs()` no longer carries a teacher column; the actual per-call
    input id joins the record through `record_metadata(..., input_id)`.
  - `argmax_bf16()`: independent integer total-order selection over widened
    BF16 source bits. Both signed zeros rank as one numeric zero (first, i.e.
    lowest, index wins and keeps that word's bits); subnormals keep source bits;
    non-finite words are refused.
- `capture.py`: per-cohort inputs are `[11]` then the original Torch
  `logits.argmax()` of call0 (`greedy_id_diagnostic_only`,
  `selected_max_bits`, `logits_record_sha256`, `predecessor_logits_sha256`);
  call1 asserts its input is that same cohort's call0 selection. `natural_inputs`
  and the status are natural-specific; `status` is
  `CAPTURED_ORIGINAL27B_NATURAL_FULL_WINDOW_REQUIRES_VALIDATION`.
- `validate_reference.py`: validates the natural cohort (seed call0, feedback
  call1, predecessor fullV sha join), decodes every `logits` record with the
  independent `argmax_bf16`, and requires the decoded id/bits to equal the
  reported call row and each cohort's call1 input to equal its independently
  decoded call0 argmax. Receipt schema/status are natural and carry
  `natural_inputs`, `image_id`, `image_digest`.
- `inputs.json`: natural schema/scope, `seed`, `independent_call1_input`,
  `calls`, `capacity`; identical source/model/cache/record geometry
  (`expected_records 1040`, `expected_raw_bytes 934776928`).
- `run_reference.py`: natural status, `zg-original27-natural-full-window-` name
  prefix; same image, entrypoint, mounts, 12g/swapless, `--cpus 4`,
  `--pids-limit 256`, `fsize` ulimit, nonce and CPU policy.
- `test_contract.py`: natural geometry, cohort fixtures, negative cases
  (teacher-forced cohort, wrong frontier/prior/order, missing feedback,
  predecessor mismatch, wrong bits at the join), integer argmax total-order
  cases, held-pin replacement/mutation, no model import at import time.
- `archive.py`, `source_owner.py`, `owned_container_v4.py`, `source-pin.json`:
  byte-identical to the approved package (single owned lifetime/source owner).

## Validation

```sh
cd tools/qwen27b-natural-full-window-reference
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest test_contract -v
PYTHONDONTWRITEBYTECODE=1 python3 -B run_reference.py            # capture host only
PYTHONDONTWRITEBYTECODE=1 python3 -B validate_reference.py --output <retired> \
    --receipt <fresh>/accepted-original-natural-full-window-1.json
```

`validate_reference.py` writes the fresh receipt and pins the held
`manifest`, `raw`, `invocation`, `image_inspect` files it actually read. The
runtime lease expects those five files (plus the receipt) staged under one root
by basename; transport of the 934776928 B corpus to the device host is a later
scope and never passes through the controller.

## Not qualified here

Capture, transport, runtime natural comparison against this corpus, physical
two-position execution and performance are later scopes. Nothing in this
directory is evidence of Rust numerical parity or inference support.
