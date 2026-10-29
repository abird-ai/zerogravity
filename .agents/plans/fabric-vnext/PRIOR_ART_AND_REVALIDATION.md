# Fabric vNext — full comparative research and zerogravity revalidation

**Research provenance:** Source-level investigation of ggml-org/llama.cpp and vllm-project/vllm performed 26 October 2026 through the GitHub connector; read-only inspection of zerogravity via dotlink-l5 on 26–27 October. The source research initially targeted HEADs of upstream default branches; they are moving. Revalidate exact upstream SHAs on future measurements. Historical zerogravity checkout used for file counts: HEAD 195099ca583ace766d075682927f590360c823a5 plus 26 October uncommitted working state. The later 27 October working tree is NOT included in the analysis. Current reference must not be mistaken for a new source audit.

**Preservation goal:** This document preserves detailed earlier research (file trees, architectural ownership, new-model integration, router/batching/cache layers, favorable/unfavorable patterns, measured code-size observations) and the second round of zerogravity architecture/plan review. It is not a new implementation status certification.

## 1. The central comparison

| Engine | Core abstraction | Main strength | Main limitation relevant to zerogravity |
| --- | --- | --- | --- |
| llama.cpp | Model-specific GGML tensor graph evaluated by GGML backends | Portable local execution, simple API and CLI, GGUF ecosystem, heterogeneous hardware | Shared model registry/converter coordination; less general joint multi-model/task resource scheduling |
| vLLM | Request scheduler + KV cache + workers/model runners | High-throughput GPU serving, continuous batching, async engine/worker split, rich model registry | CUDA/PyTorch-centric design and large model runner complexity; serving-first task assumptions |
| vLLM-Omni | Multiple modality stage executors/pipelines | Explicit non-AR modalities, audio/video/diffusion/codec stage diversity | Another framework's decomposition, not one Rust-native resource/state fabric |
| zerogravity (intended) | Verified semantic program + prepared physical executable + TaskDriver + transaction/state/resource authorities | Source-correct composable model/device/task/memory/serving specialization with one execution substrate | Complexity and unimplemented general compiler/serving/device/remote/media capabilities must be proven |
| zerogravity (26-Oct inspected implementation) | Rust ForwardPlan + model family crates + physical CUDA/HIP/Vulkan qualification + bounded serving | Real state/lease/source and GPU/CPU implementations beyond scaffolding | Still carries family/source-specific execution branches in GPU and serving, and incomplete production generic engine |

### The four most reusable lessons

1. **From llama.cpp:** mathematics constructs execution graphs; generic backend knows how to execute but does not need to know Qwen/Gemma identity.
2. **From vLLM:** EngineCore, SchedulerOutput, KVCacheManager, Executor, GPUModelRunner and ModelRegistry are distinct responsibilities; high throughput requires coherent scheduling, cache and physical execution design. The newer GPU MRV2 notes explicitly warn about expansion of a monolithic shared runner.
3. **From vLLM-Omni and zerogravity v5.1:** a model's internal audio/video codec stages and an application workflow spanning separate models are different. Non-text task work units cannot be shoehorned into language token rows.
4. **From zerogravity v4–v5:** semantic/effect legality, one logical publication root, one coarse physical resource arbiter, real completion/fencing and independent placement of compute, weights, state and cache distinguish a rigorous execution fabric from a serving-only runtime.

## 2. llama.cpp architecture: actual file tree and ownership

Source repository: https://github.com/ggml-org/llama.cpp
Upstream guide (moving default branch): https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md
Public llama interface: https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h
GGML backend interface: https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h
Model source directory: https://github.com/ggml-org/llama.cpp/tree/master/src/models
Server source: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
Multimodal source: https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd

### Structural layers

~~~text
llama.cpp/
├── include/
│   └── llama.h                      Public C inference API
├── src/                             libllama
│   ├── models/
│   │   ├── models.h                 Model base classes and helpers
│   │   ├── llama.cpp                Llama graph
│   │   ├── qwen3.cpp                Qwen3 graph
│   │   ├── qwen35.cpp               Qwen3.5/hybrid graph
│   │   ├── gemma4.cpp               Gemma4 graph
│   │   └── ...                      Architecture implementations
│   ├── llama-arch.{h,cpp}           Architecture and metadata mapping
│   ├── llama-model.cpp              Model lifecycle and selection
│   ├── llama-model-loader.cpp       GGUF reader and tensor binding
│   ├── llama-graph.{h,cpp}          Shared graph construction
│   ├── llama-context.cpp            Inference context and graph execution
│   ├── llama-cparams.cpp            Context parameters
│   ├── llama-batch.cpp              Batch representation
│   ├── llama-kv-cache*.cpp          KV policies/variants
│   ├── llama-memory*.cpp            State and recurrent memory
│   ├── llama-vocab.cpp              Vocabulary
│   ├── llama-sampler.cpp            Sampling
│   └── llama-quant.cpp              Quantization entrypoints
├── ggml/
│   ├── include/
│   │   ├── ggml.h                   Tensor/operation API
│   │   ├── ggml-backend.h           Backend API
│   │   └── gguf.h                   Container format
│   └── src/
│       ├── ggml.c / ggml.cpp
│       ├── ggml-backend.cpp         Backend scheduling
│       ├── ggml-backend-reg.cpp     Backend/device registry
│       ├── ggml-alloc.cpp           Allocation
│       ├── ggml-quants.c            Quant types
│       ├── ggml-cpu/
│       ├── ggml-cuda/
│       ├── ggml-hip/
│       ├── ggml-metal/
│       ├── ggml-vulkan/
│       └── ...                      SYCL, OpenCL, RPC, others
├── common/                          Application-level shared utilities
├── conversion/                      HF conversion support
├── gguf-py/
├── convert_hf_to_gguf.py
├── tools/
│   ├── server/
│   │   ├── server-context.cpp       Server inference execution
│   │   ├── server-queue.cpp         Request queue
│   │   ├── server-models.cpp        Model routing/child lifecycle
│   │   ├── server-http.cpp          HTTP
│   │   └── ...                      Streaming, chat, APIs
│   ├── cli/
│   ├── mtmd/
│   │   ├── mtmd*.cpp                Media/processor stage
│   │   └── models/
│   │       ├── gemma4v.cpp          Vision encoder
│   │       ├── gemma4a.cpp          Audio encoder
│   │       └── ...
│   └── ...
├── tests/
├── benches/
├── examples/
├── docs/
└── CMakeLists.txt
~~~

The source in 26-Oct lookup showed a dedicated src/models directory via src/CMakeLists.txt: file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp"), with a distinct llama core source list. This separation means a model graph builder can emit standard GGML operations, while the backend implements CPU/GPU memory and kernels without embedding that model's full forward loop.

**Conceptual inference sequence:** checkpoint → GGUF metadata and tensor decoder → model-specific graph builder → GGML graph → backend graph allocator/scheduler → chosen CPU/CUDA/HIP/Vulkan/etc. backend → context/state/sample → application. GGML need not know a Qwen architecture name.

**Two distinct schedulers:**
- GGML graph/backend scheduler partitions physical graph work and handles backend selection, buffers and copies.
- llama-server slot/request machinery assembles compatible sequences into decode/prefill batches and drives llama_decode.
The current server also has a router mode that manages multiple child serving instances; this provides useful lifecycle UX but is NOT one-process shared physical multi-model/task scheduling by default.

**New-model integration path:**
1. Interpret source HF model config and tensors through converter; GGUF support where applicable.
2. Extend architecture/metadata naming in llama-arch.h/cpp and gguf-py/gguf/constants.py.
3. Add src/models/<arch>.cpp: load_arch_hparams, load_arch_tensors, graph builders and model-specific state.
4. Extend llama-model.cpp/model declarations/factory appropriately.
5. Add backend primitives only if existing GGML ops cannot express actual math efficiently/at all.
6. Extend tools/mtmd where vision, audio or processor components require it.
7. Regression tests for schema/CPU/GPU/quant/long context/server.
The full guide is linked above. Integrating a nearby variant is cheap *after* all shared operations exist; a novel recurrent/attention math family requires larger changes.

**Concrete models inspected:**
- src/models/qwen35.cpp (actual graph plus recurrent memory use)
- src/models/gemma4.cpp
- src/models/qwen3.cpp
- tools/mtmd/models/gemma4v.cpp
- tools/mtmd/models/gemma4a.cpp

**Measured historical source inventory examples (NOT invariant upstream current totals):**
- llama-arch.h around 914 lines; llm_arch enum contained roughly 165 entries including UNKNOWN at lookup.
- src/models/models.h ~3,115 lines with model class declarations/helpers.
- src/llama-model.cpp ~4,022 lines; shared model dispatch/lifecycle.
These figures illustrate that even a relatively well-separated model/graph system can accumulate central registries and classes.

**What zerogravity should adopt:** per-architecture mathematical ownership; one stable public API; backend-independent model descriptions; GGML-like portable primitives and physical backend capability negotiation; simple local UX. **Avoid:** making GGUF conversion universally mandatory, sprinkling model-name switches throughout core model factory and graph scheduling, treating GGML op availability as trained model support, replicating child server processes to get logical model separation.

## 3. vLLM architecture: actual file tree and ownership

Source repository: https://github.com/vllm-project/vllm
Architecture design: https://docs.vllm.ai/en/latest/design/arch_overview/
New-model guide: https://docs.vllm.ai/en/latest/contributing/model/basic/
Model Runner V2: https://docs.vllm.ai/en/latest/design/model_runner_v2/
Registry: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py

### Source tree mapped by responsibility

~~~text
vllm/
├── vllm/
│   ├── entrypoints/
│   │   ├── openai/                   HTTP/OpenAI APIs, model endpoints
│   │   ├── cli/                      CLI parsing
│   │   └── launchers/
│   ├── v1/
│   │   ├── engine/
│   │   │   ├── core.py               EngineCore
│   │   │   ├── core_client.py        IPC
│   │   │   └── llm_engine.py         Engine-facing abstraction
│   │   ├── core/
│   │   │   ├── sched/
│   │   │   │   ├── scheduler.py       Schedule requests, token budgets
│   │   │   │   └── output.py          New/continuing/stopped step output
│   │   │   ├── kv_cache_manager.py   KV page ownership
│   │   │   ├── kv_cache_utils.py
│   │   │   └── ...                   Hybrid cache planning
│   │   ├── executor/
│   │   │   └── multiproc_executor.py Worker dispatch
│   │   ├── worker/
│   │   │   ├── gpu_worker.py         Physical worker lifecycle
│   │   │   ├── gpu_model_runner.py   Earlier V1 model runner
│   │   │   └── gpu/
│   │   │       ├── model_runner.py   Newer modular V2 path
│   │   │       └── input_batch.py    Persistent slot/batch structures
│   │   ├── attention/
│   │   │   ├── backend.py            AttentionBackend
│   │   │   └── backends/registry.py
│   │   ├── kv_cache_interface.py    Cache layouts / type schemes
│   │   └── spec_decode/              Draft/speculative strategies
│   ├── model_executor/
│   │   ├── models/
│   │   │   ├── registry.py           Model architectures mapped to implementations
│   │   │   ├── llama.py
│   │   │   ├── qwen3_5.py
│   │   │   ├── qwen3_next.py
│   │   │   ├── gemma4.py
│   │   │   └── ...                   Numerous model-specific types
│   │   └── layers/
│   │       ├── attention/
│   │       ├── quantization/
│   │       ├── linear/
│   │       └── ...                   Reusable neural layers
│   ├── multimodal/
│   │   └── registry.py              Processors/dummy inputs / frontend
│   ├── distributed/
│   ├── platforms/
│   │   └── interface.py             Device/platform capabilities
│   ├── compilation/
│   ├── config/
│   └── ...
├── csrc/                              Native kernels C++/CUDA/HIP
├── rust/
│   ├── Cargo.toml                     Separate Rust frontend workspace
│   └── src/ ...
├── tests/
├── benchmarks/
├── examples/
├── docs/
└── CMakeLists.txt
~~~

**Logical serving sequence:** HTTP/LLM frontend → tokenization/input processing → EngineCore → request Scheduler and KVCacheManager → scheduler output → Executor/Worker → shared GPUModelRunner → architecture-specific PyTorch model.forward() → shared attention/linear/MoE/quant layers + Triton/CUDA/HIP kernels → sampler/outputs → protocol streaming.

**Central vLLM distinctions:**
- Model mathematical implementation is model_executor/models/<family>.py.
- Reusable neural layers are model_executor/layers/.
- Hardware-specific attention and kernels are separate.
- Scheduler and KV cache allocate/admit state independent of model code.
- Executor communicates work to device workers and model runners.
- Frontend protocols do not define model math.
- vLLM-Omni extends stage structure beyond text, with modality-specialized computation.

**New-model options:**
A. Existing Transformers-compatible model using Transformers modeling backend (lowest-friction, may not be optimal).
B. Native optimized vLLM model implementation: forward pass, layer reuse, load_weights, architecture registry, multi-GPU/attention/quant integration, tests.
C. Out-of-tree model plugin via ModelRegistry registration, provided it complies with runtime contracts.
Novel attention/cache/state mechanisms and media processors can require modifications beyond simple model file addition. Support in registry does not guarantee arbitrary backend or quant modes.

**Concrete upstream model evidence:** vllm/model_executor/models/qwen3_5.py subclasses/uses Qwen3-Next components; gemma4.py contains its own model, attention, router, decoder structures. At 26-Oct lookup:
- llama.py about 586 lines;
- qwen3_5.py about 756 lines;
- gemma4.py about 1,507 lines;
- v1/core/kv_cache_manager.py about 930 lines;
- v1/worker/gpu/model_runner.py about 2,506 lines with prominent comment that it should contain ONLY behavior common to all models and should remain minimal.
- models/registry.py ~1,636 lines / ~67 KB with separate generation/embedding/late-interaction/reward/classification/multimodal/speculative mappings.
Historical registry excerpts contained 124 text-generation and 127 multimodal entries; these are category entries with overlaps/variants and MUST NOT be interpreted as 251 distinct fully qualified models.

**vLLM cache strategy:** paged logical KV block table separated from physical cache backing. Block pool, cache keys/refcounts and reusable prefix blocks reduce duplicate prefix computation and enable flexible admission/preemption. That should inform zerogravity's generic domain-specific state pool, but it is not a complete substitute for GDN/SSM/MLA/media history.

**vLLM Model Runner V2 lessons:** The old V1 runner grew substantial interleaving of persistent batch state, async control and model-specific hacks. MRV2 targets stable per-request slot assignments, sparse/incremental metadata updates such as staged writes, GPU-side index preparation/sampling where appropriate, reduced GPU/CPU synchronization, consistent graph capture, and single common runner with specialized behaviors elsewhere. Its documentation describes evolving/partially qualified functionality, not universal completion. This is a *design lesson*, not a suggestion to copy Python code.

**What zerogravity should adopt:** explicit scheduler output/data contracts; engine/worker separation; cache-aware admission and paged blocks; one source-independent model runner; persistent slots; multi-modal model processor reuse; source-independent public serving protocols; asynchronous CPU/GPU overlap; real task and request QoS/metrics. **Avoid:** PyTorch or CUDA as hard foundation; model or device assumptions in the request scheduler; huge Python model runner; extra process/network hops for direct embedded CPU inference; a server-named mandatory control-plane runtime for all modalities.

## 4. Comparative new-model and physical execution pathways

| Situation | llama.cpp | vLLM | zerogravity target |
| --- | --- | --- | --- |
| Another checkpoint of same architecture | Converter/metadata + compatible graphs | Same model class/weight loader | Source manifest + family adapter + scoped capability |
| Variant with new layer geometry | Model graph changes | Inheritance/shared layer/config | Extend family math, operation data, guards |
| New dense transformer | GGML model graph + metadata + tests | Optimized module or HF Transformers | Verified definition, CPU reference, canonical program, existing provider |
| Novel GDN/KDA/SSM or sparse attention | New graph and state handling, perhaps ggml kernels | New vLLM layer/cache/backend | New exact semantic op + domain schema + reference, add qualified provider |
| New quantization codec | GGUF types/converter, kernels | Config/method and kernels | Source/derived artifact codec, explicit compute encoding + independent quality |
| Vision/audio | mtmd processor and encoder graph | multimodal registry/models/Omni stages | Native verified assembly + typed media processor/task/clock |
| New GPU backend | GGML backend implementation | Platform + native kernels | RegionExecutionProvider/DeviceBackend with actual state bridge, lease/events |
| Multi-model service | Router/child processes | independent engines/replicas/frontends | One process native catalog/router/global resource + model-neutral engine |
| SDK embed | C library & integration | Python/Rust frontend paths | Single public zerogravity crate reexporting all capabilities, including serving |

**Extensibility ranking caveat:** vLLM is easiest for Transformers-compatible fallbacks, llama.cpp is direct for closely related architectures with existing GGML ops, zerogravity today has significant finite implementation/qualification setup. The target is not "zero code for any new model", but measured minimum change radius for known semantics and simple native source-trusted extension.

## 5. Historic zerogravity checkout and observed file tree (26-Oct baseline)

Historic read-only source root: /home/pvl/spaces/abird/src/zerogravity on dotlink-l5.
Original inspected SHA: 195099ca583ace766d075682927f590360c823a5, with hundreds of uncommitted refactor modifications/deletions. DO NOT use old path/line counts as current after resync.

~~~text
zerogravity/
├── Cargo.toml
├── README.md
├── AGENTS.md
├── crates/
│   ├── zerogravity-core/
│   │   └── src/
│   │       ├── accelerator.rs
│   │       ├── accelerator/program.rs
│   │       ├── resource_arbiter.rs
│   │       ├── session_root.rs
│   │       ├── device_events.rs
│   │       ├── device_queue.rs
│   │       ├── pipeline.rs
│   │       ├── targets.rs
│   │       └── ...
│   ├── zerogravity-artifacts/
│   │   └── src/
│   │       ├── artifact.rs
│   │       ├── source_weights.rs
│   │       ├── sharded.rs
│   │       ├── quant.rs
│   │       ├── acquisition/
│   │       └── cache/
│   ├── zerogravity-model-qwen35/
│   │   └── src/
│   │       ├── config.rs
│   │       ├── forward.rs
│   │       ├── plan/
│   │       ├── source/
│   │       ├── reference/
│   │       └── tokenize/
│   ├── zerogravity-model-gemma4/
│   │   └── src/
│   │       ├── config.rs
│   │       ├── tensors.rs
│   │       ├── variant_26b.rs
│   │       ├── variant_31b.rs
│   │       ├── text/
│   │       └── execution/
│   ├── zerogravity-cpu/
│   ├── zerogravity-gpu/
│   │   └── src/
│   │       ├── cuda/
│   │       ├── hip/
│   │       ├── vulkan/
│   │       ├── backend/
│   │       ├── resident_program/
│   │       ├── resident_inference/
│   │       ├── resident_provider.rs
│   │       ├── resident_leases/
│   │       └── physical_service/
│   ├── zerogravity/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model.rs
│   │       └── prepare.rs
│   └── zerogravity-cli/
│       └── src/
│           ├── main.rs
│           ├── run_cli/
│           ├── serve/
│           ├── pull/
│           └── pipeline.rs
├── tools/gpu-qualify/
├── tests/
├── docs/
└── .agents/
    ├── docs/
    ├── plans/
    ├── research/
    ├── prior-art/
    └── runs/
~~~

**Historical source counts**, source .rs under crate src (not production-only, generated/qualification included; vendor excluded):
| Crate | Rust files | Source lines | Other note |
| --- | ---: | ---: | --- |
| zerogravity-gpu | 590 | ~315,159 | ~120,412 from generated*.rs |
| zerogravity-model-qwen35 | 88 | ~34,276 | concrete family state/plan and tests |
| zerogravity-artifacts | 57 | ~32,246 | acquisition/store/format |
| zerogravity-cli | 43 | ~25,225 | current HTTP and source-specific request orchestration |
| zerogravity-core | 43 | ~19,316 | generic and historically special-case ops |
| zerogravity-cpu | 42 | ~11,686 | numerical/CPU source/reference |
| zerogravity-model-gemma4 | 24 | ~6,677 | source contract and candidate paths |
| zerogravity public facade | 16 | ~3,464 | model/config and source-specific entry |

Other historically inspected lengths: core/src/accelerator/program.rs ~3,002; gpu/src/resident_provider.rs ~2,576; gpu/src/resident_inference.rs ~2,081; gpu/src/resident_program.rs ~3,668; cli/src/serve/engine.rs ~2,473; model-qwen35/src/lib.rs ~2,298; cli/src/serve/http.rs ~1,648. Large generated files alone are not an architectural failure; *family-specific dependencies and duplicate lifecycle owners* are.

### Existing working architectural strengths

- Generic core exports MatVec/attention/RMS/GDN operator regions, MathematicalBackend, PlanOperation, ForwardPlan, value IDs, source/numeric precision, state effects and typed domains.
- Core includes ResourceArbiter with global physical key/reservations, DeviceEventLedger/quarantine, FairDeviceQueue, SessionAuthority/RootAttempt/RootBinding and typed pipeline structural validator.
- Artifact layer offers GGUF/safetensors/checkpoint/cache/immutable original-shard source ownership and Xet/HF/other transport.
- Qwen family forward.rs establishes one source-verified family traversal that evaluates or records typed math through MathematicalBackend; source roles, profiles and checkpoints are separate concerns.
- Gemma family is separated into config/tensors/variants/text/execution and source-binding candidates.
- CPU source and numerics are independent from optional CUDA/HIP/Vulkan providers.
- GPU has real resident inference programs, physical services, leases, typed ABI, source qualification and bounded actual CUDA cohorts.
- CPU router and GPU router have bounded pending queues, model aliases, source memory charges, context and streaming, with source-specific evidence and limits.
- Pipeline config is source-neutral structurally typed, with audio/image/video/text/decision ports; it explicitly refuses graph execution until qualified.

### Historical architectural coupling / changes proposed

| Historic source | Coupling observed | Better owner after redesign |
| --- | --- | --- |
| core/targets.rs | model-named catalog/status in generic core | cold builtins/catalog assembly; core keeps generic schemas |
| zerogravity/src/model.rs | Qwen/Gemma inspection enum | model definition registry at application composition |
| core/accelerator/program.rs | generic plan alongside distinct source-specific arithmetic contracts | model-specific exact recipes in family; shared math/effect types in core |
| gpu → model-qwen35 Cargo dep | reverse dependency from physical provider to semantic family | remove, bind both in facade or family specialization adapter |
| gpu/resident_program/{gemma_text,qwen_*} | source-specific recipes amid shared native physical workers | family eligibility/recipes outside shared provider; backend owns kernel/image/lifetime |
| gpu/resident_inference/{qwen_source,gemma_text} | source-specific session progression in physical crate | generic physical step + family adapter + engine SessionAuthority |
| cli/serve/gpu_engine/route.rs | Gguf/FixedPrompt/Qwen35Cuda model/request enums | model-neutral executable-model/session interface in engine |
| cli/serve/engine.rs | LoadedQwenModel and QwenTokenizer directly in HTTP request engine | model/session orchestration engine; HTTP protocol in serving lib |
| cli/serve/mod.rs | CPU/GPU routers, source profile and backend model switching | server crate for HTTP mapping, model/placement engine for execution |
| cli/serve/gpu_batch.rs | finite source-aware waves and widths | qualified task-specific batching policy and prepared provider bands in engine |
| core/resource_arbiter.rs | global live resource service and contracts in foundational crate | shared IDs/lease semantics in core; one live physical authority in engine |
| core/session_root.rs | live session logic in core | core root schema/effects, engine sole TaskCoordinator |
| core/pipeline.rs | typed ports and structural config plus model intent | core task/multimodal type contracts; library pipeline checker/preparer/runner |
| CLI acquisition, devices, configuration | reusable business functions embedded in CLI | public zerogravity library models/store/devices/config; CLI only presentation |

Those findings derive from the historic refactor state. They are a migration hypothesis requiring a fresh source read later; they are NOT instructions to remove now-useful code blindly.

**Crate dependency issue inspected:** core had no model/backend dependency; artifacts→core; cpu→core+artifacts; gpu→core+artifacts and optional model-qwen35; model-gemma4→core and optional gpu; facade→models/cpu/gpu; CLI→facade. Model→GPU plus GPU→Qwen is a dangerous extension-pattern shape even if Cargo features did not produce an unconditional cycle. Remove reverse dependency to maintain model-neutral provider.

**What existed vs target:** ForwardPlan and typed effects exist; the general "SemanticProgram" and "ExecutableBundle" names did not appear as public defined Rust types in the inspected source. A TaskDriver was mostly a design/typed-pipeline concept, not a generalized production virtual execution framework. Tests/selected Qwen GPU cohorts were meaningful but did not certify general C1..C64, all models, distributed, media or fully qualified vLLM-equivalent performance. Do not confuse accepted finite captures with arbitrary source-run support.

## 6. Zerogravity's design documentation validation (26-Oct round)

Research root: .agents/docs, .agents/plans, docs. The review inventoried around 56 non-archive docs entries in .agents/docs, 45 plan entries and six public docs. It read governing documents, heading maps, active TODOs and selected critical sections; NOT every historical 15K-line handoff verbatim. Below are the key authority relationships and decisions for future agents.

### Governing reference docs

- .agents/docs/VISION_SCOPE_AND_PRINCIPLES.md — model/task/hardware user goals and no unknown semantics.
- .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md — single execution fabric; minimal vocabulary; SIR+bundle.
- .agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md — exactly five logical authorities; domain Prepare not commit; one PublishRoot; device completion real.
- .agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md — semantic operators vs pure subsets, TaskDriver, compiled StepProgram, bounded optimization, no workflows.
- .agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md — typed model/provider/state/task extension seams and effect-driven fusion, not multiple engines.
- .agents/docs/SEMANTIC_IR_AND_COMPILER.md — detailed SIR/RIR/KIR/CIR mathematical legality; latest v4 two durable representations override earlier three/quadruple persistent IR wording.
- .agents/docs/ASYNC_RESOURCE_COMMAND_IR.md — liveness/aliasing, copies, readiness, events, coherence, no semantic device Commit.
- .agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md — one RegionExecutionProvider for NativeKernel/CapturedGraph/AOT/Dataflow/Partitioned; state bridge and code lifecycle essential.
- .agents/docs/DEVICE_KERNEL_AND_CODEGEN.md — portable kernel contracts, Rust-authored policy, GPU/ISA-specific fusion with physical qualification.
- .agents/docs/COST_MODEL_AND_AUTOTUNING.md — critical path, not naive sum; three timescales (load/offline/online bounded); goodput constraints; topology/profiles.
- .agents/docs/OPERATOR_AND_FUSION_CATALOG.md — attention, GDN/KDA, MoE, lookup, output, stateful fused region alternatives.
- .agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md — four truth claims, independent reference and full model/task quality, sampling, quant and bench comparisons.
- .agents/docs/STATE_CACHE_AND_TRANSACTIONS.md — state schema, COW, prefix frontier intersection, snapshot/recompute, cancellation.
- .agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md — EngineCore/Worker separation, scheduler outputs, persistent batch metadata and async event lifecycle.
- .agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md — token/work budgeting, group eligibility, target-authoritative speculation, state-safe prefill/decode.
- .agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md — vLLM-class serving, admission layers and joint cost/SLO selection.
- .agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md — catalog/registered model/resident deployment/replica/aliases, exact route binding and shared resource ownership.
- .agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md — one-command local defaults, progressively exposed config, exact APIs and failures.
- .agents/docs/RUNTIME_SERVING_AND_UX.md — native SDK, streaming, endpoint conformance, user diagnostics.
- .agents/docs/MEMORY_RESIDENCY_AND_IO.md and UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md — resident/streamed tensors under same lease/identity/arbiter, real asynchronous IO.
- .agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md — lossy new source artifact separate from reversible device packing.
- .agents/docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md — coherent host/GPU resources, OS page cache, WMMA and safety.
- .agents/docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md — native MTP/n-gram/Engram, target-authoritative verification, derived-source checks.
- .agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md and HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md — row-slice decomposition, true event first-use, no unsafe GPU+NPU system changes.
- .agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md, UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md, VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md — native model components, codec/time/typed state, non-AR task drivers.
- .agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md — five output frontiers, clocks, codecs, turn/barge-in, session/reconnect and QoS.
- .agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md and REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md — four placement maps, transfer plan and remote epoch/failure trust.
- .agents/docs/SECURITY_AND_RELIABILITY.md, INVARIANTS_AND_FAILURE_SEMANTICS.md — required trust/event/quarantine/lease/tenant protocol.
- .agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md, MODEL_SEMANTICS_AND_COMPATIBILITY.md, FORMATS_QUANTIZATION_AND_LOADING.md — source math, models, codec, task and extension proof.

### Current (historic) owner-plan documents

- .agents/plans/README.md / TODO.md — one owner/current implementation queue, NOT duplicate this new design file into active queue until authorized.
- .agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md — one owner, dependencies down, family source/math in model crates, base operator/allocator shared; pre-v1 can break APIs, per later user instruction.
- .agents/plans/pipeline-composition/REGIONS_AND_PROFILES.md — semantic primitive→nested regions/tasks→orthogonal workload profile, one flat canonical semantics + one prepared physical plan, no recursive interpreter.
- .agents/plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md — source to semantic ForwardPlan to prepared physical plan to original attempt/Ready/SessionAuthority; remove Qwen/Gemma leakage and duplicate pools.
- .agents/plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md — allowed explicit approximation profiles, replaceable allocator/implementation under common accounting; no unsafe silent change.
- .agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md and FUSED_GPU_BATCHING.md — B2/B4/B8 finite evidence, temporal vs request batch, paged hybrid state, graph readiness, release gates.
- .agents/plans/cpu-gpu-optimization/BACKEND_PARITY.md, SHARED_MEMORY_AND_BOUNDARIES.md, SHORT_HF_RESULTS.md — independent CUDA/HIP/Vulkan semantics and physical safety; no copied performance claims.
- .agents/plans/numerics-and-context/CORRECTNESS_POLICY.md — semantic, per-op, full model quality, determinism distinct; exact independent reference.
- .agents/plans/cli-device-routing/README.md — -d device, backend vs physical identity, fit/UMA domains, all-or-none reservations, qualified split.
- .agents/plans/memory-and-streaming/TODO.md — source leases, pooled backings, SSD/windows, paged/hybrid state/eviction.
- .agents/plans/model-execution/README.md — Qwen2B→Qwen27B→Gemma E2B/26B/31B→Flash; metadata-only vs qualified source.
- .agents/plans/multimodal/TODO.md, distributed-execution/TODO.md — important unimplemented scopes, no premature claims.
- .agents/plans/IMPLEMENTATION_ROADMAP.md P00–P23; EXPERIMENT_PROGRAM.md H01+; RELEASE_ACCEPTANCE_GATES.md; INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md.

### Conflicts resolved in the vNext design

1. Older document prose suggests three permanent IRs (SIR/EIR/DPI) or four named stages; **latest v4 contract wins: SIR and Bundle only enduring**.
2. Older StateDomain commit examples vs atomic whole-root safety; **domain Prepare only, TaskCoordinator PublishRoot once**.
3. Some scheduler narratives imply CPU per-token host state update; **device-native bounded execution islands permitted under exact state/effect/sampling and real completion constraints**.
4. Some architecture sketches imply one default model/one server; **one model-neutral engine and independent serving library, public lib as only ordinary SDK entry, standalone direct run does not require server**.
5. Older 12-Oct public engineering specification had packaging-only status; historic 26-Oct repository contained much more actual source and tests. **Source/tests and latest plan completion outrank old document status.**
6. Model registry placeholders/metadata-only entries are not executable model authorization.
7. No general autograd/training or unrestricted external plugins just to future-proof.
8. Local run should not block on remote, NPU, wafer or SSI proof gates.
9. Enormous specialized native code is acceptable as a bounded qualified physical candidate, not an architecture-specific duplicate session/server.
10. The 27-Oct user explicitly permits breaking ANY pre-v1 APIs, yet design-only archival phase must not touch active source.
11. The 27-Oct user's new library/serving requirements supersede any previous file tree showing all HTTP code inside CLI.

## 7. Comparative ownership table (historic source vs new design)

| Concern | llama.cpp | vLLM | Historic zerogravity | Fabric vNext |
| --- | --- | --- | --- | --- |
| Family math | src/models | model_executor/models | model-* (partial) | model-* only |
| Source loader | llama-model-loader/gguf-py | model_loader/load_weights | artifacts + family | artifacts generic bytes; family verifies roles |
| Semantic ops | GGML | PyTorch/vllm layers | core/accelerator/program | core/semantic/program, model-specific recipes outside |
| CPU kernel | ggml-cpu | CPU platform | zerogravity-cpu | CPU reference/provider |
| GPU kernel | ggml-cuda/hip etc | csrc/Triton/attention | zerogravity-gpu | GPU backend/kernel providers, model eligibility elsewhere |
| Cache/state | llama-memory/kv | v1/core/kv_cache | core/session_root + family + gpu | generic state transaction in engine, paged KV domain strategy |
| Scheduler | server-context | v1/core/sched | device_queue + cli/gpu_batch | engine task-aware scheduler |
| Physical worker | GGML backend | v1/worker/executor | gpu/physical_service | physical provider/worker + engine admission |
| HTTP/stream | tools/server | entrypoints/openai | cli/serve | **zerogravity-serving standalone library** |
| Public Rust SDK | C lib; bindings separate | Python/Rust frontends | zerogravity facade incomplete | **zerogravity public library exposes all APIs** |
| Multi-model router | server-models/process | engines and replicas | CPU/GPU model-specific routers | engine catalog/residency/router, exact immutable binding |
| Multimodal | mtmd | multimodal/Omni | pipeline validation/family | typed ModelAssembly + TaskDriver + serving |
| Distributed | RPC backend | distributed | plans | one Bundle placement + trusted remote worker |

## 8. Original three-way recommendations retained

**Keep/adapt:**
- Model family as explicit unit of ownership and minimal diff radius.
- Canonical source-aware tensor math and independently selected hardware execution.
- vLLM-class continuous scheduling, KV block accounting, async dispatch and slots.
- A model-neutral executable-model interface; request and output protocols not family-specific.
- One co-located local process and service; no mandatory external proxies.
- Shared immutable weights but private session roots.
- Hardware capabilities/evidence, independent CPU+CUDA/HIP/Vulkan qualification.
- Correctness then performance on full mixed request/service workload.
- Fused whole region, batch/row and source-specific mathematical optimizations as optional guarded candidates.
- Simple command-line run and serve UX, with all implementation in libraries.

**Avoid:**
- mandatory inference process per model, or independent runtime per backend/modality.
- growing core family-name enum/registry and shared GPU runner.
- CPU per-op tensor copies or mandatory host sync between device-resident stages.
- false native model support from test fixture/shape or metadata-only entry.
- training/autograd/equality saturation in token critical path.
- global GPU scratch memory lock at each kernel.
- universal LRU spanning weights, state and SSD.
- external text APIs misrepresented as native region backend.
- silent output/model quality changes, wrong quant source or converted checkpoint identity.
- collapsing queued work, dispatched work, physical Ready, StateRoot publish and client delivery into a single success flag.

## 9. Remaining research/evidence gaps

The 26-Oct comparative analysis was a source-tree and architecture review, not a complete line-by-line verification of every upstream module or a throughput benchmark. Some upstream docs and code have changed by 27-Oct and will change again. Before performance implementation:
- Pin exact llama.cpp and vLLM commit/feature branch and vLLM-Omni release; verify MRV2 readiness and active scheduler/cache APIs, not merely design draft.
- Check precise model support and quantization/attention in the actual selected upstream versions.
- Verify current zerogravity tree/implementation after explicit resync; previous 590 GPU Rust file count/paths are historical only.
- Independently run semantics, state, physical/resource, API and matched performance gates.
- Avoid equating source code counts with maintainability or number of supported models.
- Compare vLLM M1/M2 C1/C4/C8/C16/C32/C64 and llama.cpp parallel slot/context runs only when same source/quality/task and hardware device/driver are admissible; include requests offered/rejected, TTFT/ITL, throughput, memory/energy and tails.

This is enough research to design the next-generation abstraction; it is not yet enough to claim zerogravity equals or exceeds other engines in performance.

## 10. Primary external links and exact file anchors

### llama.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/CMakeLists.txt
- https://github.com/ggml-org/llama.cpp/blob/master/src/models/models.h
- https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma4.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend-reg.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/CMakeLists.txt
- https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/CMakeLists.txt
- https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md

### vLLM
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma4.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/output.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/input_batch.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py
- https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py
- https://github.com/vllm-project/vllm/blob/main/vllm/platforms/interface.py
- https://github.com/vllm-project/vllm/blob/main/rust/Cargo.toml
- https://docs.vllm.ai/en/latest/design/arch_overview/
- https://docs.vllm.ai/en/latest/design/model_runner_v2/
- https://docs.vllm.ai/en/latest/contributing/model/basic/

These URLs are references to upstream default branches, NOT immutable benchmarks. Freeze actual commit SHA before quoting exact per-line source in future evaluations. Confirm source licenses before adapting code.

**Next step for another agent:** DO NOT start refactoring from these historic paths. Read HANDOFF.md and wait for the user's explicit next-phase resynchronization request.
