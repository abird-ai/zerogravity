# Zerogravity facade

The `zerogravity` library reexports the concrete engine owners. Existing public
paths and type identities remain stable; this crate adds no scheduler, numerical
engine, source identity issuer or physical admission authority.

The default features are `cpu`, `model-qwen35` and `text`. Selecting a GPU backend
(`vulkan`, `cuda`, `hip`) adds `zerogravity::gpu` without selecting a model family,
tokenizer or CPU executor. `all-models`, `all-backends` and `all` aggregate only
implemented production capabilities. Diagnostic features remain explicit.

Core state/resource contracts and bounded artifact inspection are available
without default features. Model source ownership belongs to artifacts and the
family adapter; CPU math and GPU execution remain optional independent owners.

API integration tests and the existing diagnostic examples live here. Their
checkpoint fixtures retain workspace-root coordinates, independent of Cargo's
per-package test working directory. The default workspace members include this
crate and the CLI, preserving existing named `--example` commands.

The executable is owned by [zerogravity-cli](../zerogravity-cli/README.md).
