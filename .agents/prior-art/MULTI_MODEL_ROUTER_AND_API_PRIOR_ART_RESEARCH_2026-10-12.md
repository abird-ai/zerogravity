# Multi-model local runtime, router, CLI and API prior-art research

**2026-10-12 · NON-NORMATIVE SOURCE RESEARCH.** Public projects and API providers are named here for attribution, licensing and test-comparison only; core zerogravity architecture is independently specified in `MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md` and `ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md`. The source research tree `/home/pvl/spaces/abird/research/inference` is **read-only**. No remote API request, model download, server install or source code import was performed.

## 1. User goal and immediately important prior art

The user wants one experience that is **as simple as one command to run a model** while also operating a built-in model router, load balancing and concurrent multi-model serving with per-model hardware/quant/context/sampler/configuration, exposing OpenAI-compatible APIs and preserving the distributed/heterogeneous compiler architecture.

That request should not be reduced to a conventional API gateway proxy:
- model identity and model artifact/configuration aliases are separate;
- immutable loaded model weights, physical replicas, device placements, draft/adapters and *mutable session state* have different lifecycles;
- limited physical RAM/GPU HBM imposes actual simultaneous-residency limits even if the registry contains thousands of model names;
- switching a model mid-session generally changes logits/tokenizer/state and cannot be silent;
- a service/router only adds value when using the **same TaskCoordinator, ResourceArbiter, RegionExecutionProvider and TaskDriver** that the local CLI uses.

### 1.1 Ollama CLI and lifecycle: simplicity baseline

- [Official Ollama CLI reference](https://docs.ollama.com/cli) (accessed Oct 12, 2026) documents `ollama run <model>`, `pull`, `ls`, `ps`, `rm`, `stop`, `serve`, and customization through Modelfile. It also describes multimodal prompting and embeddings from CLI.
- [Official Ollama FAQ](https://docs.ollama.com/faq) says it supports **multiple resident models when memory allows**, multiple parallel requests for a model, queueing when a new model cannot load until another idle one unloads, a configurable maximum resident model count and a `keep_alive` control. Its guidance explicitly ties parallel context to additional memory use. These are **product behavioral precedents**, not fixed counts/settings zerogravity must copy.
- [Modelfile reference](https://docs.ollama.com/modelfile) includes `FROM`, `PARAMETER`, `SYSTEM`, `TEMPLATE`, `ADAPTER`, `LICENSE` and other customization rules. Its parameter overlay and model variant alias are useful convenience concepts; copying its exact syntax/API without source/license review is **not necessary** for an original Rust design.
- [Ollama API OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) says its compatibility is a **subset** of OpenAI endpoints, including local `/v1/chat/completions` and `/v1/responses` examples, vision and stream concerns. Local SDK examples use a placeholder API key ignored by server; zerogravity should make **local-vs-remote authentication explicit** rather than assuming any token is secure.
- [Ollama native API](https://docs.ollama.com/api/introduction) separates its API from OpenAI compatibility; zerogravity likewise should have a small native diagnostics/model-control API distinct from interoperability endpoints.

**Generalized lessons:** stable single binary/CLI, free-standing run vs long-lived daemon behavior, model snapshots/inspect, content-addressed ModelStore, automatic on-demand load/unload under physical memory safety, configurable model profiles and optional keepalive pins, and one API endpoint over many resident/prepared models. A common anti-pattern is silently overcommitting parallel context allocations.

**License note:** Ollama is publicly MIT at its repository root historically, but verify the exact code/files/revision and notices before reusing any source; documentation and model artifacts have separate rights. Model code and remote registries are not user-approved implicit download permissions.

### 1.2 Lightweight native runner/router prior art

- [llama.cpp CLI docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md) show a simple model path / prompt CLI and optional connection to a local model server.
- [llama.cpp server main README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) documents an integrated **router mode** with model source cache/dir/presets, on-demand load/unload, a max concurrently loaded models limit and OpenAI-compatible chat/completion/responses/embeddings. The README distinguishes a router's model selection by API `model` from execution in an instance.
- [llama.cpp server developer docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md) describe multiple inference server instances behind one router. **Zerogravity's direction differs deliberately:** support many model handles/resident plans/replicas through one shared inference engine/ResourceArbiter/TaskCoordinator; a model-specific subprocess may be an optional isolation/fault-domain policy, not a required C++ or per-model inference engine clone.
- Other native runner products and Rust model serving libraries under the original prior-art catalog may support quant formats/tokenizer/hardware backends, but are not canonical engine owners.

**Generalized lesson:** multi-model registry, loadable resident instances and API routing are **distinct but composable interfaces**. Model-level routing should never preclude native multi-request batching *within* a model, or cross-model scheduling when safe.

## 2. OpenAI-compatible APIs must be pinned to an explicit tested subset

- [OpenAI Responses API reference (accessed Oct 2026)](https://developers.openai.com/api/reference/resources/responses/methods/create) includes structured inputs, streaming and optional store/background/sequence semantics; endpoint evolution is fast. [OpenAI Responses overview](https://developers.openai.com/api/reference/responses/overview) differentiates stateful conversations/tool requests. [Streaming events reference](https://developers.openai.com/api/reference/resources/responses/streaming-events) defines typed `response.*` events, not simply the same chat `delta` SSE format.
- [OpenAI Python client](https://developers.openai.com/api/reference/python) and [official streaming guide](https://developers.openai.com/api/docs/guides/streaming-responses) provide useful client-conformance targets. The first-party SDK may send an API-key header even against a localhost clone; do not interpret that as proof remote caller is authorized or leak in log.
- Compatibility contract has **endpoint-level, modality-level and parameter-level coverage**, not a single boolean "OpenAI-compatible". Chat Completions and Responses differ in schema, tool event handling, conversation store, usage and streaming events; multimodal ASR/TTS/image APIs have additional format/codec and task rules.
- Remote API compatibility is **not** permission for the runtime to execute tools, invoke built-in web search, retrieve private files, or store conversations; native source TaskDriver can generate structured tool calls but executing tools is an external permissioned application boundary.
- Some API parameters are model or backend capability dependent (e.g. response format/JSON schema, seed reproducibility, logprobs, parallel tool calls, audio output, context, requested sampling). Unsupported fields should receive **typed errors** or have explicitly documented safe defaults, never be silently reinterpreted.

**Generalized lesson:** canonical typed `TaskRequest/StreamContract` with protocol adapters that preserve semantics, streaming/usage/finish reason, auth and explicit per-endpoint conformance fixtures. Default privacy-first state retention differs from a cloud provider's current storage defaults.

## 3. Multi-model lifecycle and routing architecture research

**Resident model vs registered model:** even if thousands of manifests/aliases are visible in a catalog, only models admitted by physical ResourceArbiter can have weights, compiled programs and KV/SSM state resident simultaneously. An alias is not a physical replica.

**Routing is not one algorithm:**
- **Exact model resolution:** deterministic `model` alias → pinned checkpoint+quant/adapter/numeric identity, no silent substitution of another family, checkpoint or quant.
- **Replica balancing:** choose among semantically equivalent prepared model replicas using queue/SLO/cache affinity/admitted physical capacity and same source artifact; avoid naive round-robin migration of stateful sessions.
- **Task and phase routing:** embedding/reranking/ASR/TTS/vision roles select *task-compatible model definitions* and TaskDrivers, with explicit task/model policy; some compound tasks run multiple named models under one TaskProgram.
- **Alias/policy selection:** a special configured `auto` alias may select **different model identities** by cost/capability/quality if the caller explicitly permits it, with actual selected model and semantic change visible. This is not equivalent to replica balancing.
- **Device/machine placement:** once model selected, choose compatible executable/cost topology from existing placements; do not copy an inference request to a separate gateway runtime.

**Load balancing and capacity:** model warm/load JIT and quant/layout conversion cost, cache affinity and session stickiness, device power/firmware and memory, multiple concurrent model and model-state reservations, fairness and queue backpressure, on-demand load/unload (only idle/unpinned) and prewarming. Protect against thrashing from alternating large model choices. A scale-to-zero model may need an explicit saved/snapshot/replay guarantee before evicting its live session's mutable cache.

## 4. Security, model provenance and user experience

Simple local CLI and API defaults should be **loopback only**, with remote exposure only on explicit bind and authenticated access. Remote peer/tunnel enrollment is a **separate authorization** from serving the HTTP API. A model alias cannot escalate model license/data locality restrictions or select an unauthorized remote node. The local user might have downloaded proprietary/private weights: automatic model pulls and cross-host artifact copies need explicit authorization and verifiable hashes.

**Default usability** means:
- `run <model>` handles model resolution from local catalog/path and safe backend auto-selection; if remote artifact needed, prompt/explicit pull before any network download, with license and size exposed.
- all behavior adjustable through **profile/config/CLI/approved request fields**, and `inspect --explain` reveals actual effective settings, source and reason for fallback.
- avoid a mandatory admin dashboard, cluster scheduler or per-model `serve` process; any resident daemon/service is an optional session-sharing deployment form, not the definition of the runtime.
- keep CLI default concise; use `--verbose`, `--explain`, `config show --effective` for progressive disclosure.

## 5. Source distinction from requested deliverable

This document identifies **external references only**, including MIT source candidates. It does not make the project a wrapper around another runtime or API gateway, and published model-server behaviors are not internal zerogravity tests.

**Normative destination:**
- `MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md` — explicit catalog/model-profile/resident-replica/router/scheduler/memory/placement ownership.
- `ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md` — proposed one-command local UX, configuration, OpenAI endpoint subset and conformance/failure policy.

**Status:** research + designs only; no code, model downloads, API tests or benchmark runs executed.
