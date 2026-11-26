# Zerogravity CLI

This package owns the sole `zerogravity` executable, argument handling and text
rendering. Default workspace `cargo build`, `cargo run -- ...` and source
installation (`cargo install --path crates/zerogravity-cli`) select the full
production CLI: all current models and CPU, CUDA, HIP/ROCm and Vulkan backends,
acquisition, serving, TUI and Web. This enables compilation, not an unconditional
model load or device permit; native runtime availability, source compatibility,
qualification and resource admission still govern execution.

The binary also builds with `--no-default-features`: help, version and bounded
artifact/config inspection remain available. Inference requires its selected
backend, `model-qwen35` and `text`; disabled capabilities refuse before source
I/O or device initialization.

Backend commands are explicit:

- `run` uses CPU execution.
- `run-vulkan-amd` selects Vulkan integrated GPU0.
- `run-vulkan-nvidia` selects Vulkan discrete GPU0.
- `run-cuda` selects CUDA device0.
- `run-rocm` selects HIP/ROCm device0.

The Vulkan command names preserve the qualified frontend's device-kind mapping;
they do not discover or prove a vendor identity. Qualification records bind the
actual physical device separately. Execution never falls back to another backend.

Features forward through the library facade: `cpu`, `model-qwen35`,
`model-gemma4`, `text`, `vulkan`, `cuda`, `hip`, `all-models`,
`all-backends` and `all`. CLI default is `all`, the production superset.
It excludes diagnostic/qualification, fault-injection and experimental native
attention providers. Model source-contract support is not a claim of inference
qualification (e.g. Gemma4 remains source-contract-only in the production set).
The library facade keeps its own smaller defaults. For the portable CPU CLI:

```sh
cargo run --release -p zerogravity-cli --no-default-features --features cpu,model-qwen35,text,tui,web
```

Existing diagnostic feature names remain explicit forwarders for API examples.

The CPU frontend preserves its existing argument/audit limits. The resident GPU
frontend accepts at most 512 consumed tokens, checks worst-case prompt plus
generation before GPU preparation, and requires colocated original GGUF,
`config.json` and `tokenizer.json`. It uses the existing generic GPU model/session
API and preserves EOS handling and the device-token loop.
