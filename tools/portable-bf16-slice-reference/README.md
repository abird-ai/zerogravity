# Independent portable BF16 operator reference

- Synthetic original source:3 BF16 tensors, embedding17×H/gammaH/W97×H,
  H1024/2048, token16. Same immutable byte generator as Rust fixtures.
- Original pinned HF ordinary RMS + Torch embedding/linear, CPU AVX2/B1.
  No full model, Rust math in original process, expected operands or GPU.
-11 actual public boundaries; original Torch owns mean/inverse/projection
  dispatch. Rust CPU separately checks internal sum/sqrt/FP32 accumulation.
- Byte-exact equality only. Different legitimate provider arithmetic refuses
  this explicit profile; no tolerance or SM86 retagging.
- Reuse unchanged owned-container-v4: offline approved image,4CPU/12GiB OS,
  16GiB AS/240s/1MiB files, original TERM+twoKILL before reap/EOF/source closure.
- Two source-only protocol fixture tests execute no Torch/container/model.

ROOT coordinates original invocations and all hardware:

```sh
python3 tools/portable-bf16-slice-reference/run_reference.py --width 1024 --new-output /absolute/fresh/original1024
python3 tools/portable-bf16-slice-reference/validate_reference.py --original /absolute/fresh/original1024 --new-accepted /absolute/fresh/accepted1024.json
```

Repeat H2048 within the same qualification unit. Accepted references are
expected-only and keep all portable/model/state/serving/performance flagsfalse.
The hardware test requires `ZG_ARITHMETIC_REFERENCE_1024` and
`ZG_ARITHMETIC_REFERENCE_1024_SHA256` (and2048) from independently reviewed
accepted outputs. It opens/hashes expected bytes only after original physical
completion and device status/finite checks. No implicit cold reference fallback.

Hardware test: `resident_program::typed_slice::tests::actual_portable_typed_bf16_slice`;
`ZG_ARITHMETIC_BACKEND=hip|vulkan`, fresh process per
`ZG_ARITHMETIC_MODE=healthy|rms-unknown|projection-unknown|cross-rms|cross-projection`.
Actual PCI0000:c6:00.0,64MiB/provider/256MiB physical. Peer same-PCI faults
retain both128MiB envelopes under original conservative quarantine.
No hardware/model/performance support follows from building this source.
