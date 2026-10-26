# Explicit GPU qualification frontend

This tool links `zerogravity-gpu` once and invokes its feature-gated diagnostic
entry point. It contains no copied GPU modules, model schedule or physical queue
statics. Production text generation belongs to the `zerogravity` CLI.

From the repository root:

```sh
cargo run --locked --offline -p zerogravity-gpu-qualify \
  --features qualification,vulkan -- resident-qualify-vulkan-nvidia MODEL
python3 scripts/native-gpu.py -- cargo run --release --locked --offline \
  -p zerogravity-gpu-qualify --features qualification,cuda -- resident-qualify-cuda MODEL
```

Select `hip` for the qualified ROCm profile. Use `resident-context-BACKEND MODEL
PROMPT RAW_IDS` for the retained independent genuine512 corpus. Complete GDN
traces require `diagnostic-gdn-trace`; checkpoint diagnostics require
`portable-checkpoint`; actual cancellation requires `hardware-abort-probe` and
a disposable process. These features are absent from production `all`.

The tool intentionally depends on CPU/family oracles through qualification;
selecting a production GPU backend alone does not. All numerical comparisons,
source identity checks, physical readbacks and quarantine gates remain in the
existing implementation. Native LLVM device code generation, broader contexts
and local wedged gfx90c recovery are not implied. See the
[resident library runbook](../../crates/zerogravity-gpu/README.md).

## Source projection fixture helper

The developer-only [Q4 fixture extractor](extract-qwen-q4.py) verifies the full
local GGUF SHA before packaging original Q4 bytes, source embedding values and
the retained scalar fixture oracle. It remains a diagnostic helper, separate
from the Rust inference engine and whole-model source qualification. From the
repository root:

```sh
python3 tools/gpu-qualify/extract-qwen-q4.py --help
```

Supply the existing approved local checkpoint, an ignored output path and its
exact `--sha256`; the helper downloads nothing. The [accepted checkpoint](../../.agents/plans/accepted-baseline/TODO.md)
records historical qualification. New runs follow the
[plan TODO index](../../.agents/plans/TODO.md).
