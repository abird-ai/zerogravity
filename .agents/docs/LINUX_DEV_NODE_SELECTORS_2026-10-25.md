# Linux GPU device-node selectors

Scope: `-d /dev/...` on Linux, implemented through the existing `run_policy::DeviceRef` + `run_cli::devices::allowed` pathway. No alternative execution backend, allocator, GPU context or model permit.

Examples:

```sh
zerogravity run model.gguf -d /dev/dri/renderD128
zerogravity run model.gguf -d /dev/dri/by-path/pci-0000:05:00.0-render
zerogravity run model.gguf -d /dev/nvidia0
```

- Accept bounded normalized absolute /dev paths alongside `cpu`, `gpu:N` and `pci:...`.
- For DRM cards/render nodes, verify real character-device major/minor, `/sys/class/drm/<node>/dev`, and an exact PCI device link under `/sys/devices`.
- For per-GPU `/dev/nvidiaN`, verify char major 195 and `Device Minor`/full `Bus Location` from NVIDIA kernel proc metadata. NVIDIA minor is **not** assumed equal to CUDA ordinal; duplicate or missing mappings refuse.
- Canonicalized symlinks must resolve to those supported nodes under /dev. No direct device open, ioctl, driver context, privilege change or model/network/cache action.
- Reject ambiguous/global nodes like `/dev/kfd`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, regular files, generic character nodes and paths with unsafe components.
- Reuse the existing full `pci:` inventory `Observation` exactly, preserving device holds/exclusions, backend/memory/source qualification, and alias deduplication; non-Linux /dev paths refuse. Optional advanced `--backend` remains a filter, not a replacement for -d.
- Path resolution means *identity selection*, not permission to run an unsupported model. For AMD select a DRM render node; `/dev/kfd` identifies a shared KFD service, not a unique GPU.

Verification (logs in `.agents/runs/dev-device-selector-20261025/`): **15/15** focused CLI tests and **208/208** core tests pass, including synthetic DRM and NVIDIA mismatches, exact physical observation equivalence and live PCI alias consistency. Default CLI builds and actual CLI accepts both tested DRM render nodes and `/dev/nvidia0`, then reaches the original GGUF GPU capability refusal. `/dev/kfd` and `/dev/null` refuse as non-unique GPUs. CUDA/short-HF CLI feature **typecheck passes**; default core/CLI **Clippy -D warnings passes**, formatting and diff checks pass. The combined all-backend cold compile timed out without a compiler error and is **not** claimed passing. No GPU model inference or GPU-performance claim; full workspace gate not inferred.
