# Prior art, provenance and architecture neutrality

**15 October 2026 — LICENSE SELECTION:** The user selected [MIT for Zerogravity's
own source and documentation](../../LICENSE) and approved root and GPU package
metadata `license = "MIT"`, with founder/contributor copyright. This **does not**
automatically relicense externally sourced files, model weights, tokenizers,
codecs, datasets, precompiled GPU kernels, or dependencies. [Contribution and AI
development policy](../plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md)
records human-led design and AI-assisted implementation with a docs-first
contribution workflow.

**21 October 2026 — OWNER REUSE DIRECTIVE:** For every implementation and
correctness harness, prefer suitable audited MIT/Apache-2.0 components that can
be reused through Rust or statically linked native C ABI, including a narrow
C++ shim where practical. If direct linkage is unavailable or incompatible,
directly port the useful implementation and fixtures, establish faithful
behavior, then fit it into our architecture. This supersedes older from-scratch
implementation and Rust-only foreign-kernel restrictions; architecture neutrality,
license/NOTICE obligations and original runtime ownership remain. See the
[integration contract](../plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md).
This directive supplies the reuse preference within authorized work; do not
request a second permission merely because suitable code is externally authored.

**Date:** 2026-10-12. **Authority:** project policy for engineering design
documentation, reflecting explicit user directive U-017. **Not legal advice.**

## 1. Original architecture and reusable implementations

The design is an **original modular AI computation/runtime architecture**. It must
retain its own semantic/runtime contracts rather than require an external model
server or proprietary engine architecture. Private native adapters may bind
pinned component APIs/layouts; those details do not become the engine's public
semantic or ownership model.

Zerogravity owns its architecture while reusing suitable audited implementations
under the owner directive above. Previously vendored components (for example,
CubeCL) and new linked/ported components retain applicable upstream licenses.
Named project references are limited to the
acknowledged inspiration projects in the root README (llama.cpp, vLLM, SGLang,
Ollama, gufo, strata, maya, CubeCL, pi and dwarfstar4) and to the mechanism names
below for architectural naming. License, NOTICE and provenance records must name
the actual upstream authors/projects of linked or ported components.

- **Canonical normative design** (master architecture, composition spec,
  contracts, implementation roadmap, acceptance criteria, agent prompt) must use
  **mechanism/capability/interface names**, not "Project-X-style runtime" or
  `ProjectXPlan` as a type. Legitimate backend APIs and model/hardware
  identifiers (e.g., ROCm, CUDA, Vulkan, GGUF, MLIR as a source language, model
  checkpoint ID, XDNA2, gfx1151) identify actual protocols and deployment
  targets, not the architecture's parentage.
- **Research-only files** keep original technical analysis, exact numbers, limits
  and negative findings, written as our own ideas; external implementation
  specifics are not automatically the design to copy.
- **Code reuse**: any candidate source must be audited at the exact pinned
  revision/file for third-party notices (GPL/Apache/BSD/LLVM exceptions) before
  any import; model weights/device binaries have separate rights. Prefer suitable
  MIT/Apache native reuse, otherwise a faithful attributed port; a root license
  alone does not establish the complete component/dependency disposition.
- **Proprietary/nonpublic work**: black-box published performance/behavior can
  motivate experiments, but no illegal source access, cloning or fabricated
  knowledge. Patents/trademarks remain independent.
- **No vendor dependency surprise**: XRT, ROCm, CUDA driver/maths FFI and compiler
  toolchains are optional device adapters, never semantic or server dependencies.
  Rust owns integration/contracts; audited native linked or ported kernels are
  permitted under the 21 October directive and the same physical/numerical gates.

## 2. Reuse and license review

A source repository being permissively licensed at its root is not sufficient to
import a specific kernel, transport/RDMA vendor binding, model checkpoint, device
program or tunnel client. Before reusing any source:

1. Confirm the exact upstream license file, `NOTICE` and any file-level header at
   the pinned revision.
2. Determine suitable Rust/static C ABI linkage (a narrow C++ shim where practical)
   or a direct native port; audit transitive runtime/toolchain and ownership needs.
3. Record attribution/NOTICE/SBOM; treat model weights, tokenizers and datasets as
   separately licensed.
4. Apply the owner-approved MIT/Apache reuse-first rule without redundant
   source-language approval. Resolve other license dispositions under applicable
   existing authorizations; a port preserves upstream derivative obligations.

Link/port the smallest useful component, including its executable fixtures and
negative controls, before optimizing it through our interfaces. No embedded
Python/PyTorch runtime or whole foreign inference server is implied. Existing
independent reference tooling remains valid; inspect final binary dependencies
instead of assuming that a C++ extension is a standalone static library.

Vendored components retain their own license texts in place (for example, the
vendored CubeCL packages under `crates/zerogravity-gpu/vendor/`). The
Zerogravity-authored source remains MIT.

## 3. Mechanism taxonomy: use these instead of project names in design

| Prior-art mechanism | Canonical neutral architecture term | Component owner |
|---|---|---|
| serving-token scheduler / continuous batching | **BudgetedWorkScheduler** and `StepPlan` | Runtime policy |
| paged/prefix KV and ranked prefix index | **StatePagePool + PrefixIndex** | State/Memory |
| general graph math execution | **PortableSemanticExecutor** | Semantic/Kernels |
| single-model extreme fused kernel | **SpecializedRegionProvider** | Region optimizer / KernelProvider |
| GPU/RAM/SSD expert pool | **TieredSparseParameterStore** | Resource arbiter + IO |
| GPU+GPU routed experts | **AsymmetricExpertOwnership** | Placement/CIR |
| GPU+NPU partial GEMM | **SplitProjectionCompute** | RIR/KIR/CIR |
| GPU graph eager/capture | **CompiledDispatchBundle** | CIR device execution |
| high-level/low-level math compiler | **TypedSemanticGraph + RegionKernelLowering** | SIR/RIR/KIR |
| async stream/resource compiler | **ResourceCommandGraph** | CIR |
| bounded equality rewrite search | **PureRegionRewriteSearch** | Offline optimizer |
| speculative branches | **DraftVerifier + StateTransaction** | TaskProgram/runtime |
| hybrid state prompt reuse | **FullStatePrefixSnapshot** | State codec/coordinator |
| content-addressed model downloads | **ArtifactStore + TrustedManifest** | Product/artifact |
| multimodal ASR/TTS/image/video pipelines | **TypedTaskProgram** | Task orchestrator |

**Enforcement:** before implementation or design edits, a reviewer should be able
to delete the entire prior-art and research corpus without losing any needed
mathematical/semantic/device/runtime specification. If code depends on a named
project's proprietary type, source layout, cross-project environment variable or
process lifecycle, the architecture is not yet generalized.

## 4. Prior-art and research preservation

Research documents live under [`.agents/prior-art/`](.) and
[`.agents/research/`](../research/) and keep their technical analysis, numbers,
limits and negative findings. They are **non-normative**: a source
implementation's particular flags or defaults are not automatically the design to
copy. They are retained as searchable engineering context, written as our own
analysis. Actual linked or ported files are separately identified by their
source/license/NOTICE records; a research reference alone does not imply reuse.

## 5. Concrete developer checklist

1. Define **capability/mechanism** in project-neutral language and identify
   semantic/non-semantic responsibility.
2. Cite named prior art only where the acknowledgement policy permits it.
3. If reusing actual code, verify **that file's** license/NOTICE/derivative
   attribution and dependencies; prefer suitable MIT/Apache native static reuse,
   otherwise directly port implementation and fixtures with retained notices.
4. Implement behind our own versioned capability interfaces; do not create adapter
   wrappers around a whole external model server.
5. Check fallback and new-model/new-device/new-task extensibility tests.
6. Revalidate numerical quality, memory safety and performance against an
   independent baseline; a source author's results are hypothesis targets.
7. Update the decision register when user policy changes, benchmark records when
   evidence changes and the license table when actual source pins change.

**Final principle:** external work is prior art and a validation benchmark; the
executable architecture belongs to Zerogravity and must remain coherent if every
external project stops developing tomorrow.
