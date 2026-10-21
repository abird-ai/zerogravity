# Zerogravity resident GPU library

The production implementation lives in `crates/zerogravity-gpu`. It executes the
engine's existing `ForwardPlan` with source-bound GPU buffers, independent
immutable session roots, physical admission, fair workers and actual completion.
The library owns one queue/service/quarantine implementation. It contains no
Qwen forward loop or mandatory model, CPU executor, tokenizer or facade dependency.

Provider features are explicit and default off: `vulkan`, `cuda`, `hip`; `all`
selects those production providers only. Native providers select the existing
CubeCL C++ NVRTC/HIPRTC path. The selected CPP graph does not require the unused
LLVM SDK/JIT dependency or executable-only link shim. Existing compatible native
runtime/compiler libraries and headers remain required. No setup command installs
or downloads a model or SDK. LLVM-only upstream compiler paths remain optional
and are not qualified as Zerogravity device backends.

## Source and owning API

`ResidentModel::load(Device, Arc<dyn TensorSource>, ForwardPlan, provider_cap,
device_cap)` retains the immutable source and one prepared physical workspace.
Source trait calls occur only during cold validation/upload; token execution uses
prepared direct bindings and the plan's cached source identity. The constructor
rejects source/layout mismatches, invalid logits/token embedding vocabularies and
capacities outside the existing 256/512 resident profile before allocation.
Raw device values naming compiled-out providers or APIs other than explicit
Vulkan, CUDA or HIP return a typed refusal before client initialization.

Public `ResidentSession`, `ResidentSnapshot`, `PreparedInference`, `ResidentToken`
and `InferenceReadback` retain private physical fields. Existing prepare/commit,
forward, GPU greedy feedback, snapshot/restore/reset and readback semantics remain.
The optional `portable-checkpoint` bridge adds the family v1 codec without a CPU
executor or tokenizer. It takes the caller's typed codec owner explicitly and
verifies its source identity before physical readback. Qualification separately
selects CPU and text oracles; production `all` selects only GPU providers.

## Production and qualification commands

The sole text frontend is the workspace `zerogravity` CLI. From the repository
root, with the existing selected runtime configured:

```sh
cargo run --release --locked --offline -p zerogravity-cli --features vulkan -- \
  run-vulkan-nvidia target/models/qwen35-0.8b/model.gguf --raw --prompt 'Hi.' --max-tokens 8
python3 scripts/native-gpu.py -- cargo run --release --locked --offline \
  -p zerogravity-cli --features cuda -- \
  run-cuda target/models/qwen35-0.8b/model.gguf --raw --prompt 'Hi.' --max-tokens 8
```

Use `hip`/`run-rocm` only on a healthy qualified AMD profile. Existing H24 local
`gfx90c` initialization failure remains open; no local AMD launch or reset is
implied. Model/profile qualification remains bounded to the original approved
0.8B/2B checkpoints and 512 consumed tokens, not arbitrary contexts or models.
The new workspace release profile preserves CPU thin LTO/codegen-units=1 and
therefore differs from the former independent probe's default release profile.
Actual newly packaged binaries require their own physical regression gates.

The explicit qualification tool links this same library once:

```sh
cargo build --release --locked --offline -p zerogravity-gpu-qualify \
  --features qualification,vulkan
python3 scripts/native-gpu.py -- cargo build --release --locked --offline \
  -p zerogravity-gpu-qualify --features qualification,cuda,hip
python3 scripts/native-gpu.py -- target/release/zerogravity-gpu-probe --help
```

`resident-qualify-BACKEND MODEL` and `resident-context-BACKEND MODEL PROMPT IDS`
retain the existing physical/source gates. `hardware-abort-probe`,
`portable-checkpoint` and `diagnostic-gdn-trace` are separate diagnostic flags;
none is in production `all`. Destructive cancellation cases need disposable
processes. Old probe `run-*` commands direct users to the sole production CLI.
See [qualification tool](../../tools/gpu-qualify/README.md) and
[vendor provenance/consumer patches](vendor/README.md).

The native command guard always requires actual user/network isolation and
Cargo offline mode. Default CPP commands do not require LLVM. Explicit
`--require-llvm` and `--verify-only` verify the existing pinned cache; they never
acquire it. Accepted qualification and limits are recorded in the
[baseline TODO](../../.agents/plans/accepted-baseline/TODO.md).

## Accepted qualification baseline

The [owner-approved checkpoint](../../.agents/plans/accepted-baseline/TODO.md)
records completed physical qualification and limits. Its raw records are retired.
Follow the [plan TODO index](../../.agents/plans/TODO.md) for future
changes; use the current commands above rather than restoring old probe paths.
