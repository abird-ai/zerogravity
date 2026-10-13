# Prior art, provenance and architecture neutrality

**15 October 2026 — LICENSE SELECTION:** The user selected [MIT for Zerogravity's
own source and documentation](../../LICENSE) and approved root and GPU package
metadata `license = "MIT"`, with founder/contributor copyright. This **does not**
automatically relicense externally sourced files, model weights, tokenizers,
codecs, datasets, precompiled GPU kernels, or dependencies. [Contribution and AI
development policy](../plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md)
records human-led design and AI-assisted implementation with a docs-first
contribution workflow.

**Date:** 2026-10-12. **Authority:** project policy for engineering design
documentation, reflecting explicit user directive U-017. **Not legal advice.**

## 1. From-scratch architecture rule

The design is an **original modular AI computation/runtime architecture**. It must
**not** depend conceptually, organizationally or programmatically on a named
external engine, library's internal data structures, model-specific server, or
proprietary project.

Zerogravity is a from-scratch rebuild informed by accumulated prior-art concepts,
learnings and research. **No source from any prior-art project was copied or
linked unless it is directly vendored (for example, CubeCL (vendored)), which
retains its upstream license.** Named project references are limited to the
acknowledged inspiration projects in the root README (llama.cpp, vLLM, SGLang,
Ollama, gufo, strata, maya, CubeCL, pi and dwarfstar4) and to the mechanism names
below.

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
  any import; model weights/device binaries have separate rights. No automatic
  copy/import.
- **Proprietary/nonpublic work**: black-box published performance/behavior can
  motivate experiments, but no illegal source access, cloning or fabricated
  knowledge. Patents/trademarks remain independent.
- **No vendor dependency surprise**: XRT, ROCm, CUDA driver/maths FFI and compiler
  toolchains are optional device adapters, never semantic or server dependencies.
  The requirement for Rust-owned kernel source remains in force.

## 2. Reuse and license review

A source repository being permissively licensed at its root is not sufficient to
import a specific kernel, transport/RDMA vendor binding, model checkpoint, device
program or tunnel client. Before reusing any source:

1. Confirm the exact upstream license file, `NOTICE` and any file-level header at
   the pinned revision.
2. Confirm the user's Rust source-ownership policy for any foreign device kernel.
3. Record attribution/NOTICE/SBOM; treat model weights, tokenizers and datasets as
   separately licensed.
4. Obtain explicit user approval where source-language/licensing constraints
   require it.

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
ideas, not as links to incorporated code.

## 5. Concrete developer checklist

1. Define **capability/mechanism** in project-neutral language and identify
   semantic/non-semantic responsibility.
2. Cite named prior art only where the acknowledgement policy permits it.
3. If reusing actual code, verify **that file's** license/NOTICE/derivative
   attribution; confirm Rust-owned source and user consent for foreign device
   kernels.
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
