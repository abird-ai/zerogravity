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
- **Simple interface.** Local CLI and Rust API, with HTTP/OpenAI-compatible, WebSocket, WebRTC, and distributed serving in the architecture.

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

Only local files are used; no model weights are automatically downloaded or executed as third-party model code. See [Implementation status](#implementation-status-and-roadmap) for the currently supported execution paths.

## Architecture

~~~mermaid
flowchart TB
    I["CLI · SDK · HTTP · WebSocket · WebRTC"]
    R["Model catalog · Router · Admission"]
    S["Verified ModelAssembly + TaskDriver"]
    P["SemanticProgram → ExecutableBundle"]
    W["CPU · GPU · NPU · Peer workers"]
    M["GPU VRAM · Unified RAM · Host RAM · SSD"]
    C["Real device events → TaskCoordinator → StateRoot"]

    I --> R --> S --> P --> W --> C
    M <--> W
    C --> I
~~~

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
| **HTTP / OpenAI-compatible** | Text, embeddings, model routing, tools, and structured outputs |
| **WebSocket / WebRTC** | Streaming text, speech, native duplex audio, and live multimodal sessions |
| **Media tasks** | Image, video, speech synthesis/transcription, and asynchronous generation |
| **Distributed** | Qualified compute, model weights, session state, and cache placement across devices or peers |

These are **architectural targets**, not a claim that every interface or model family is implemented today.

## Implementation status and roadmap

| Available now | In progress / planned |
|---|---|
| Native Rust **Qwen3.5 0.8B / 2B** GGUF text CPU execution and inspection | Complete native accelerator model execution and additional model families |
| Original-model CPU state, snapshots, prefill, and source-reference checks | General compiled SemanticProgram / ExecutableBundle |
| Shared immutable model loading and model catalog foundations | Continuous batching, exact speculation, and OpenAI-compatible multi-model server |
| Source-qualified **selected** Vulkan GPU projections on NVIDIA and AMD | Whole-model GPU recurrent/attention/KV, CUDA/HIP, and distributed execution |
| Bounded physical GPU admission, fair worker primitives, and real-event safety probes | First-class SSD execution, multimodal media/real-time models, and full hardware tuning |

**Qualification:** the Qwen CPU Phase 1 acceptance gate passed through 8K; 16K/32K full-model cases remain deferred. The GPU evidence covers selected original-model operations—not a complete GPU model run. vLLM-class serving and multimodal specialization remain design/implementation work, not measured product capabilities.

**Roadmap:** [phases P00–P23](.agents/plans/IMPLEMENTATION_ROADMAP.md) · [current progress](.agents/plans/CURRENT_STATUS_AND_TRACKING.md) · [acceptance and evidence](.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md)

## FAQ

**Which models does zerogravity support?** The architecture is model-family agnostic, but execution requires verified source semantics, codecs, and kernels. See [Implementation status](#implementation-status-and-roadmap) for what runs today.

**Does it require a GPU or a Python inference engine?** No. The core is native Rust and supports CPU execution; accelerator providers are optional. Zerogravity does not wrap a separate Python/C++ model-serving engine.

**Can a model use SSD when it does not fit in GPU memory?** The design supports one logical tensor across GPU, unified RAM, host memory, and asynchronously streamed SSD. Resident access remains direct; actual SSD misses have measurable latency. Native SSD execution is still on the roadmap.

**How does it differ from vLLM?** Zerogravity targets the same concurrency fundamentals while jointly specializing model semantics, kernels, quantization, state, memory tier, hardware, and serving QoS. Performance superiority must be established by comparable benchmarks, not assumed.

**Can I run several models at once?** The model catalog and shared-weight foundation exist. Full multi-model inference serving and cross-model admission are planned; registering models does not imply they all fit in physical memory.

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

**We are building zerogravity in the open with you.** We'd love your ideas, architectural challenges, research, and thoughtful improvements. We believe the best way to develop software is with human-led design and AI-led implementation. [More on the contributing guide here.](CONTRIBUTING.md)

## AI disclosure

**Human-centric design. AI-centric implementation.** Zerogravity is built with extensive AI assistance for research, code, and tests. Humans lead the architecture, priorities, and acceptance criteria. We neither hand over direction to autonomous models nor claim that every generated line is manually reviewed. We rely on explicit designs, reproducible tests, engineering evidence, and continuous validation.

## License

[MIT](https://github.com/abird-ai/zerogravity/blob/main/LICENSE) © 2026 Prasanna Loganathar and zerogravity contributors. Zerogravity-authored
text and software are MIT-licensed. Vendored third-party components (for example,
CubeCL) and other assets retain their respective upstream licenses. Model
weights, third-party libraries, and other assets retain their respective licenses.
