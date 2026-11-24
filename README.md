# zerogravity

**The blazingly fast AI execution fabric - built for composable multi-modal inference specialization at every layer, from individual kernels to distributed systems**

A single binary Rust execution fabric for **text, image, video, audio, real-time, and combined multimodal inference**. Model-specific mathematics, hardware kernels, memory placement, scheduling, and serving compose under one engine—not a separate runtime for every model or modality.

Run _models locally_ without compromise - *one engine to run them all*.

> **From scratch, informed by prior art.** No source from any prior-art project
> was copied or linked unless it is directly vendored (for example, CubeCL
> (vendored)) and therefore retains its upstream license. Zerogravity is an
> independent Rust rebuild that learns from the concepts, research and
> engineering of the projects acknowledged under [Related projects](#related-projects).

## Intro

- **Any model architecture.** Transformers, hybrid recurrent models, MoE, diffusion/flow, speech codecs, direct decision heads, and model assemblies—each with verified semantics.
- **Specialization at every layer.** Source-aware quantization, fused operators, ISA-specific kernels, batching, speculative decoding, and distributed execution.
- **One memory fabric.** GPU VRAM, unified memory, system RAM, and SSD-streamed tensors share a logical model identity and safe physical leases.
- **High-concurrency serving.** Continuous batching, chunked prefill, paged/hybrid state, prefix reuse, multi-model routing, and latency-aware admission.
- **Joint optimization.** Choose compatible batch, draft, kernel, residency, and placement strategies together—not as isolated optimizations.
- **Correct by construction.** Source-verified mathematics, explicit state effects, actual device completion, and one authoritative session-state publisher.
- **Simple interface.** CLI, TUI, Web UI, and Rust API share one execution fabric, with OpenAI-compatible HTTP, WebSocket, and WebRTC serving.

## Quick start

Build on Linux with the Nix development environment:

~~~sh
nix develop
cargo build --release --locked
~~~

Run a local **Qwen3.5 0.8B or 2B GGUF** with its matching `config.json` and `tokenizer.json`:

~~~sh
./target/release/zerogravity inspect /path/to/model.gguf
./target/release/zerogravity run /path/to/model.gguf \
  --prompt "Explain Rust ownership in three sentences" --max-tokens 32
~~~

Run the repository's quality checks:

~~~sh
./scripts/check.sh
~~~

These commands use local model files without implicit downloads or execution of third-party model code.

## Architecture

![ZeroGravity execution architecture](docs/architecture-overview.svg)

**Reuse optimized mathematics; own the execution fabric.** Native execution
composes shared operators, mature vendor libraries, specialized kernels, and
captured command graphs. Compiled providers—MIGraphX, TensorRT, IREE, and
ONNX Runtime—execute closed regions or whole models under the same resource,
scheduling, and state authority.

**Define a model once.** Original Hugging Face checkpoints (Safetensors)
and GGUF feed source-bound Rust model definitions. The semantic program
describes the computation, tensor layouts, numerical contracts, and state
effects. Native operators and compiled providers lower that program into
prepared executables. ONNX is an independent graph-import format, not a
required intermediate representation.

**Separate computation from placement.** CPU, CUDA, HIP, and Vulkan provide
physical execution and completion. Model semantics remain independent of
the selected implementation, hardware, and memory tier.

**Four independent concerns:**

| Concern | Contract |
|---|---|
| **Model** | Verified source, tensor encodings, components, operators, and task-specific state |
| **Execution** | Compiled, guarded regions and kernels; CPU/GPU/NPU/peer placement |
| **Resources** | Shared physical admission; leased weights, state, caches, and GPU/RAM/SSD materialization |
| **Serving** | Concurrent requests, stream events, routing, QoS, and single-authority state commitment |

A model is defined once. Its **compute, immutable weights, mutable session state, and reusable caches** can be placed independently when source semantics and physical completion allow it. Fully resident tensors take the direct device path; SSD misses use bounded asynchronous materialization.

### Joint optimization

~~~mermaid
flowchart LR
    B["Batch · Prefill"]
    D["Draft · Verify"]
    K["Kernel · Quant"]
    M["Cache · Residency"]
    L["Device · Placement"]
    P["Bounded, source-qualified planner"]
    O["SLO goodput · Latency · Fairness"]

    B --> P
    D --> P
    K --> P
    M --> P
    L --> P
    P --> O
~~~

**Choose the best validated combination**, subject to exact model semantics, numerical policy, resource budgets, and deadlines:

~~~text
best_plan = argmax SLO_goodput(plan)
            over source-qualified, physically feasible plans
~~~

A larger batch can waste padded MoE rows. A longer draft can touch more experts. A smaller quantization can decode more slowly. An SSD cache hit can still be expensive if it copies full weights each step. **The best plan optimizes these interactions together**, with a simple verified fallback.

### Design decisions

- **One execution engine**, not one backend server per architecture, quantization, device, or modality.
- **One authoritative StateRoot per session**; speculative output cannot publish before target verification.
- **One physical resource authority**; buffer reuse requires real IO/device completion.
- **Composable specialization** from tensor format and fused operator through task, model, hardware, and cluster.
- **Native and hosted models are distinct**; external APIs use explicit service adapters, not counterfeit local kernels.

## Engineering

- [Architecture and subsystem specifications](.agents/docs/)
- [Joint serving and optimization contract](.agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md)
- [Multimodal and real-time architecture](.agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md)
- [Unified GPU/RAM/SSD residency](.agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md)
- [Whitepaper](docs/WHITEPAPER.md) · [Engineering specification](docs/ENGINEERING_SPECIFICATION.md)
- [Build, releases, NixOS, and previous implementation details](.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md)

## Interfaces

| Surface | Purpose |
|---|---|
| **CLI / Rust** | Local model inspection, execution, and embedded workflows |
| **TUI / Web** | Interactive model management, configuration, serving, and observability |
| **HTTP / OpenAI-compatible** | Text, embeddings, model routing, tools, and structured outputs |
| **WebSocket / WebRTC** | Streaming text, speech, native duplex audio, and live multimodal sessions |
| **Media tasks** | Image, video, speech synthesis/transcription, and asynchronous generation |
| **Distributed** | Qualified compute, model weights, session state, and cache placement across devices or peers |

## Implementation status and roadmap

| Available now | In progress / planned |
|---|---|
| Native Rust **Qwen3.5 0.8B / 2B** GGUF text CPU execution and inspection | Complete native accelerator model execution and additional model families |
| Original-model CPU state, snapshots, prefill, and source-reference checks | Closed-region preparation and compiled execution providers (MIGraphX, TensorRT, IREE, ONNX Runtime) |
| Shared immutable model loading and model catalog foundations | Unified CLI, TUI and Web; broader OpenAI-compatible serving and speculation |
| Native source-backed **HIP Qwen 0.8B / 2B / 27B** execution on qualified gfx1151; bounded CUDA and Vulkan paths | Optimized backend parity, broader model coverage, optional compiled providers and distributed execution |
| Bounded physical GPU admission, fair worker primitives, and real-event safety probes | First-class SSD execution, multimodal media/real-time models, and full hardware tuning |

**Qualification:** the original Qwen CPU Phase 1 gate passed through 8K. Whole-model HIP source execution, shared cohorts and native rocBLAS projection/GDN have passed bounded physical model tests. Exact-reference execution remains available. Optimized vLLM parity, wider hardware/model qualification and multimodal specialization remain implementation work. See the [current baseline](.agents/docs/CURRENT_BASELINE.md) for the measured scope.

**Delivery order:** foundations → CLI, TUI and Web → ROCm parity with vLLM →
CUDA parity with vLLM → Vulkan → compiled execution providers. The
implementation plan defines the phased work, acceptance criteria, and
remaining qualifications.

**Roadmap:** [implementation plan](.agents/plans/FULL_IMPLEMENTATION_PLAN.md) · [current baseline](.agents/docs/CURRENT_BASELINE.md) · [current progress](.agents/plans/CURRENT_STATUS_AND_TRACKING.md) · [phases P00–P23](.agents/plans/IMPLEMENTATION_ROADMAP.md) · [acceptance](.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md)

## FAQ

**How are models defined?** Each model family supplies verified source semantics, tensor encodings, task definitions, and state requirements. Shared operators and execution providers realize those definitions without a separate inference engine for each family.

**Does it require a GPU or a Python inference engine?** No. The core is native Rust and supports CPU execution; accelerator providers are optional. Zerogravity does not wrap a separate Python/C++ model-serving engine.

**How does SSD execution work?** One logical tensor spans GPU VRAM, unified memory, host RAM, and SSD. Resident access takes the direct path; cache misses use bounded asynchronous materialization under the same resource and lifetime contracts.

**How does it differ from vLLM?** Zerogravity combines high-concurrency inference with joint specialization of model semantics, kernels, quantization, state, memory tiers, hardware, and serving QoS across text and multimodal pipelines.

**How do multiple models run together?** One runtime manages independently admitted model instances and composes them into reusable task pipelines. Immutable weights and compatible caches can be shared; mutable session state remains private. Placement and residency follow the physical resource budget.

## Related projects

Zerogravity would not exist without the open-source inference and systems
community. We thank and acknowledge the following projects as inspiration and
prior art; their ideas, research and engineering shaped this work:

- **llama.cpp** — portable GGUF execution and the dead-simple local single-binary experience.
- **vLLM** — high-throughput serving, continuous batching and paged state.
- **SGLang** — structured program execution and prefix-aware cache reuse.
- **Ollama** — one-command local model UX and resident model management.
- **gufo** — model-family specialization, accelerator kernels and multimodal tasks.
- **strata** and **maya** — extreme model specialization and VRAM→RAM→SSD expert tiering.
- **CubeCL (vendored)** — the Rust GPU kernel IR, vendored with its upstream license.
- **pi** and **dwarfstar4** — agentic engineering and extreme-model execution research.

These names are acknowledged as inspiration and interoperability references only.
No source code from any prior-art project is copied or linked unless it is
directly vendored and retains its upstream license; Zerogravity is an
independent, from-scratch rebuild.

## Contributing

We build zerogravity in the open and would love your ideas, architectural
challenges, research, and focused improvements. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the best ways to take part.

## AI disclosure

**Human-centric design. AI-centric implementation.** Zerogravity is built with
extensive AI assistance for research, code, tests, and documentation. We believe
the best software comes from human-led design and AI-assisted implementation:
humans own the architecture, priorities, and acceptance criteria, while AI helps
implement, test, and iterate. We don't hand direction to autonomous models or
claim every generated line is manually reviewed — correctness rests on explicit
designs, reproducible tests, and continuous validation.

## License

[MIT](https://github.com/abird-ai/zerogravity/blob/main/LICENSE) © 2026 Prasanna Loganathar and zerogravity contributors. Zerogravity-authored
text and software are MIT-licensed. Vendored third-party components (for example,
CubeCL) and other assets retain their respective upstream licenses. Model
weights, third-party libraries, and other assets retain their respective licenses.
