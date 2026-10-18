Author: Project owner
Co-authors: GPT-6

# Zerogravity — one fabric, single-model and multi-model pipelines

**Decision date:** 18 October 2026. This is the canonical product intent.
**Implementation status:** source-neutral model registry, typed component
extension schemas, canonical YAML/CLI graph compiler, immutable in-memory
context-owner snapshots and source-grant filtering are implemented.
Neither multi-model execution, API serving, speech diarization, streaming
audio/image, large-model SSD inference nor other model-family registrations
are claims of working numerical inference. Unsupported operations refuse.

## Product contract

**Both modes use one engine and one single binary.**

1. **Direct / single-model:** the traditional
   `zerogravity run model.gguf --prompt "hello"` starts a qualified model
   in its own process without having to declare a pipeline, server or router.
   A one-stage pipeline can also expose that model's API once serving exists.
2. **One process / many models:** a configuration owns a pool of independently
   source-verified model instances, each with optional replicas/adapters,
   different placement and residency policies. Many graph pipelines and API
   routes reuse those instances. Loading a new pipeline does not imply a new
   OS process, duplicate model allocation or a second Rust inference runtime.
3. **Same configuration from YAML or repeatable flags:** provide a human-edited
   versioned YAML document with `model:`, `pipeline:` (including typed
   `step:` / `link:` lists) and `route:`, or pass `--model`,
   `--step`, `--link` and `--route` repeatedly with input/output/name
   flags. Both inputs lower into **one source-neutral graph**.
   CLI overrides of an existing config must have an explicit merge/precedence
   contract before being enabled: today mixing them refuses.
4. **One router publishes configured APIs:** an explicit route names a pipeline
   and a protocol with no guesses about conversions or source capability.
   Supported protocol *intent names* are `openai-text`, `openai-asr`, `sse`,
   `websocket` and `webrtc`; these currently validate but do not bind.
   Future serving adapters use the original TaskCoordinator/ResourceArbiter;
   no transport gets its own scheduler or permission to publish neural state.
5. **Every execution/migration/quality claim is separately qualified.**
   Source metadata, downloaded bytes, cache verification, model semantics,
   backend numerical parity, streaming packet delivery, and measured latency
   form distinct gates. A checked config starts no model and opens no socket.

The *same* declared model instance can be used by a chat endpoint, an
ASR→LLM→TTS voice assistant, a multimodal vision agent, or as a standalone
request. Shared immutable tensors and existing StateRoot copy-on-write are
reused, but each request/turn has private mutable state and authorization.
A model can also have named, separately admitted replicas: alias reuse and
replica count are never inferred from two component references.

### Intended engine flow

```text
CLI single-model run -------------------+
                                        |
YAML / repeated pipeline flags --------+--> validate one typed graph
                                               |
Verified snapshot + model assembly catalog ----+--> one model residency manager
                                               |    GPU / RAM / SSD / peers
                                               |    shared SourceIdentity/ResourceArbiter
HTTP / SSE / WebSocket / WebRTC / CLI --------->+--> one TaskCoordinator
                                                     + TaskDrivers & typed ports
                                                     + model/codec/solver execution
                                                     + shared admission/fair queues
                                                     + atomic StateRoot/AV publication
                                                     |
                                       one router -> configured endpoints
```

No central graph dispatcher may add mandatory work to each low-latency token,
audio frame, or resident tensor operation. Compile DAGs and type conversions
once per version/source; use prepared plans and physical completion-guarded
leases. Full resident model execution must not require SSD page faults or a
per-token central registry lock.

## Model priorities (binding order)

This order governs architecture and **source qualification**, not a
promise that an upstream title identifies one immutable weight/quant variant:

1. **Qwen families 3.5+, 3.5, 3.8:** highest optimization priority. The
   source-qualified resident Qwen3.5 0.8B/2B baseline remains unchanged.
   First finish Qwen3.5 2B, then Qwen3.8 27B execution. Qwen3.8 Flash-Next
   follows with independently bound QSA/Gated DeltaNet/MoE/gated residual,
   trained n-gram *embedding* tensors and MTP. Qwen3.5+ is a placeholder until
   an exact upstream release is qualified.
2. **Gemma4:** E2B, 26B-A4B (MoE) then dense 31B. Implement model-specific
   PLE/FFN/attention/routing/relative soft-cap semantics; avoid pretending a
   dense 31B source has 26B MoE kernels.
3. **DeepSeek V4.1 Flash**, then **GLM 5.3 Flash**: add checked source/provider
   pipeline and independent model architecture contracts, but **do NOT download
   their payloads** until the user expressly lifts the hold. The public
   repositories currently indicate DeepSeek FP8 `deepseek_v41` and
   GLM `glm5_next`, not a Qwen-compatible generic MoE.
4. **GLM 5.3 full**, **Meta Muse Spark Open** (including NVFP4 and license),
   **Mistral** and **Llama** families. Every special sparse/recurrent/
   multimodal/codec architecture gets its own source-real gates. A closed
   Muse variant is not automatically available through open Muse weights.
5. Other top OSS models (Phi, Kimi, MiniMax, other DeepSeek, FLUX/Wan/Kokoro)
   are discoverable placeholders; adding a name never adds model capability.

**Media targets:** Qwen Image 2.1, Qwen TTS, Qwen ASR, OpenAI Whisper,
NVIDIA Canary-Qwen 2.5B, NVIDIA Parakeet TDT 1.1B; add encoder, diarization,
codec and decision families independently. Qwen Image uses a pinned Diffusers
**text encoder + processor + transformer + scheduler + VAE assembly**, not
a GGUF or plain text-model auto-loader. Canary has NeMo/Qwen composite
frontend semantics; Parakeet is a native NeMo archive with TDT decoder.
ASR and TTS need their own feature extractors/codebooks and full media clocks.

The source-neutral registry exists at
`crates/zerogravity-core/src/targets.rs`; `zerogravity models list` and
`zerogravity models inspect SLUG` only report that intent. These operations
may not construct model/provider/GPU sessions. `models run/stream` fail
closed even when metadata-only source resolution is verified. The existing
loaded `ModelCatalog` and CPU/GPU `run` are separate source-bound owners.

## Pull sequence and quant policy

**Phase A — actual pinned source plan/pull:**

- Original Qwen3.5 0.8B/2B; Qwen3.8 27B; Gemma4 E2B/26B/31B,
  all source-bound and in available explicitly selected quants.
- Qwen Image2.1 original multi-component assembly; Qwen ASR/TTS exact
  original processors/codecs; Whisper; NVIDIA Canary and Parakeet.
- Qwen3.8 Flash-Next original BF16/FP8 and explicit third-party Q2,
  about 3.5 bits-per-weight, and Q4 **source variants**, including separately
  recognized learned n-gram embedding banks, MTP and other required files.
- DeepSeek V4.1 Flash and GLM 5.3 Flash: **metadata/pipeline only**.
  These two are expressly prohibited from payload download for now.

The existing `pull hf ... --file model.safetensors.index.json` handles one
pinned indexed source, with model sidecars; `pull hf ... --quant Q4_K_M`
selects one exact GGUF basename-tag family, not a quantization conversion.
An actual **3.5-bit** quality class does not imply a literal `Q3.5` tag
(the existing ASCII GGUF tag grammar uses names such as `Q3_K_M`). Source
repository, pin, exact encoding, shapes, quant group/scales and independent
trained inference/quality must be bound. No guessed quant repository,
dequant fallback, alias rewrite or automatic conversion. For a user-selected
quantizer, an explicit source-derived recipe must preserve all MTP/n-gram/
KV/recurrent components or refuse as incomplete.

**Source bundle for nonstandard multi-component models:**

`zerogravity pull hf REPO --revision REV --bundle --include GLOB ...`
or the corresponding GitHub choice collects **only** bounded explicitly
selected members of one pinned tree through the existing transport, transfer
admission, raw hashing and sole cache publisher. It assigns all members
`Auxiliary` under `huggingface-bundle` or `github-bundle`, rather than
inventing an executable model closure. Exact offline selection requires
the identical cached subset. Symlinks and uncontrolled directories refuse;
Xet-native bundle assembly, nested safetensors index automation and NeMo
unpacking are not implemented. Source-first checking and dry-run do NOT
verify nested shards, completeness, processors or codec correctness; assembly
qualification requires a separate validated component schema.

For instance QwenImage2.1 has a `model_index.json` tying typed
`text_encoder`, `transformer`, `vae`, `scheduler`, `processor` paths.
Unknown custom Python code must never execute from a model repository.
Native metadata-only pulls of all requested families are the near-term
acceptance target; payload fetch requires a selected destination, storage
admission and applicable source/license authorization.

## Pipeline API and typed composition

The reusable contract is registered in
`crates/zerogravity-core/src/pipeline.rs`; input/output ports have
**explicit semantic types**, currently `text`, `audio-pcm`,
`timed-transcript`, `speaker-transcript`, `image-rgba` and the
**non-user-constructible** `committed-chat-context` snapshot. Registered
component kinds: `asr`, `diarization`, `transcript-text`,
`speakers-text`, `translate`, `llm`, `tts`, `image-generate`.
ASR, diarization, transcript-text, speakers-text and translation
expose OPTIONAL read-only `context` ports, with `translate` producing
speaker-tagged text without mutating the acoustic source. A future
component adds one registered source-bound port contract and a
TaskDriver/codec implementation; it does **not** require a new server,
model catalog or generic unverified decoder path. Output that cannot
flow into an input needs an explicit registered conversion component.
The compiler validates graph bounds, duplicate IDs, required ports,
speaker/ASR/LLM/TTS model bindings, topological DAG, source-to-output
productive connections, route names/types and unique path declarations.

The ten human-facing YAML quickstarts for text/voice/context,
audio codecs, image assembly/video frames and decision filters are indexed
in [the examples README](../../../examples/pipelines/README.md).

```sh
# Only checks the pure graph; no file/source/model or socket is opened.
zerogravity pipeline check --config examples/pipelines/voice-assistant.yaml

# Same structural graph can be written using repeated flags:
zerogravity pipeline check --name chat --input text --output text \
  --model assistant=qwen3.5-2b --step reply=llm:assistant \
  --link input.value=reply.text --link reply.text=output.value \
  --route /v1/chat/completions=openai-text

# Explicit refusal until real source/backend/stream/transport gates pass.
zerogravity pipeline run --config examples/pipelines/voice-assistant.yaml
```

The YAML `model:` entries identify one shared instance per model ID;
`pipeline[].step` references those IDs, and `route:` declares an endpoint
bound to one pipeline. Validation is bounded (256 KiB file, 128 model
entries, 64 pipelines, 128 stages/512 edges each and 128 routes).
YAML is versioned with `schema: zerogravity.pipeline.v1`, a single
document, strict typed fields and no executable extensions. The loader
supports scalar anchors/aliases and shallow `<<` mapping merges,
but refuses custom tags, duplicate keys, recursive aliases, malformed
merges, depth above 32, more than 16,384 expanded AST nodes and
expanded scalar content above 256 KiB. CLI flags and YAML generate
the same immutable *shape*; loading a model,
accessing credentials, opening network ports, or making quality claims is
not part of validation. Port clocks/formats, compatible codec transforms,
auth and router SLOs require source-specific future contracts.

### Typed extension registry and future media operator contracts

The native `ComponentRegistry` in
`crates/zerogravity-core/src/pipeline.rs` permits a host/agent to register
a new **bounded immutable schema** for a custom stage and call
`Config::validate_with_registry` without modifying the central DAG
compiler or creating another serving process. The default CLI accepts
only built-in contracts; it cannot load arbitrary user-provided Rust/Python
code or infer a model executable from YAML. Each static extension
declares its task binding, input/output ports, semantic value kinds,
and optional input ports. Duplicate/malformed names, duplicate or
missing ports, and forged context outputs refuse. Every new backend
still needs independent source/codec/permission/trained-value/runtime
qualification before it can execute.

The current future-purpose stage descriptions (all **NOT IMPLEMENTED**)
include `audio-feature-extract` (PCM→features),
`audio-tokenize`/`audio-detokenize` (PCM↔codebook tokens),
`image-solve` (prompt→source-typed latents),
`image-decode` (latents→complete pixels),
`video-generate` (prompt→complete video frame stream) and
`decision` (text→typed verdict). They complement the earlier
ASR, diarization, transcript conversion, context-filter/translation,
LLM, TTS and image-generation scaffolds. The type registry contains
`audio-features`, `audio-tokens`, `image-latents`, `video-frames`,
and `decision-verdict` in addition to existing text/audio/image/context
values. Typed descriptions are **not** codecs, working operators,
scheduler jobs, API endpoints or streaming packet formats.

Future implementations must add source-specific memory placement,
modality clocks, completed audio codebook/frame boundaries, context
grant/epoch handling, legal image pixel/tile publication, video
presentation ordering, decision probability calibration and real
ResourceArbiter admission. A registered structural port never
authorizes a model download or creates an executable model instance.

The canonical YAML quickstarts cover direct text, ASR+diarization,
voice assistant, opt-in contextual voice, voice with tool/decision
filtering, audio codec, image generation, latent/decoder assembly,
video streaming intent and standalone decision models.
See [all examples](../../../examples/pipelines/README.md).

### Required pipeline examples

**Complete speech recognition and diarization:**

```text
audio samples --> ASR ------------------> timed transcript ---+
             \                                             |
              +--> diarization(audio, transcript) ---------+
                                            |
                                     speaker transcript
```

**Single coherent voice assistant:**

```text
audio -> ASR -> diarization -> speaker-text -> Qwen LLM -> TTS -> audio
                      \           /
                       timestamps
```

The same Qwen model can also serve `chat` in the same process. Speaker
identity, timestamped partial transcript and playback finality are typed,
not free-form text concatenation. A model may be configured for exclusive
or shared placement when proven; route selection never changes model
source/quant under the hood.

### Optional previous-chat context for ASR, diarization and translation

**Owner decision, 18 October 2026:** a single-process voice assistant may
optionally share its **text LLM's previous conversation context** with the
ASR, speaker-attribution and transcript-to-text/translation stages. This
is a first-class typed input, not a separate model server or a raw
conversation dump. The same `assistant` model instance may serve chat,
answer-generation and a separately source-qualified translation stage.

The context path is **OFF by default**. A human-editable config opts in:

```yaml
# Fragment of one entry under pipeline:; see full example for models/stages.
pipeline:
  - name: talk
    input: audio-pcm
    output: audio-pcm
    realtime: true
    chat_context: {enabled: true, from_model: assistant, max_bytes: 4096}
    # Context is read-only; optional stage inputs get None when disabled.
    link:
      - {from: context.value, to: listen.context}
      - {from: context.value, to: speaker.context}
      - {from: context.value, to: translate.context}
      - {from: context.value, to: prompt.context}
```

The equivalent repeated CLI flags are `--chat-context-from assistant`
(optionally `--chat-context-max-bytes 4096`) and
`--chat-context-enabled false` to turn sharing off without removing
edges. See `examples/pipelines/contextual-voice-assistant.yaml`
for the full typed ASR→speaker diarization→speaker-preserving translation→
text conversion→shared Qwen response→TTS pipeline and a separate chat
API in the **same process**.

**No same-turn cycle:** `context.value` is a synthetic, read-only source
of a previously committed context snapshot, or other app-authorized
configuration/shared source captured for this request. It is never an
output edge from a concurrently executing LLM/reply stage. The compiler reserves the
`context` name, validates one declared text-capable `from_model`
that appears in a configured `llm` stage (possibly another pipeline
in the same process), and rejects unknown providers, value spoofing,
cycles, duplicate inputs and snapshots outside **1..1048576 UTF-8 bytes**. A 4KiB snapshot
is the default; a stricter runtime/resource policy can impose a lower
limit, and larger explicit budgets need normal admission.
The snapshot has a fixed, source-neutral `CommittedChatContext` type:
it cannot be injected as an external `audio-pcm`, raw text, user
parameter, model KV address or arbitrary provider callback. A disabled
policy emits `None`, even when its optional typed edges are still in
the graph; a missing policy rejects any attempt to wire the special
context source.

### Generic ordered context filtering and decision models

The same reusable, typed filter pipeline can reduce any source-granted
context stream before it reaches ASR, diarization, text interpretation,
translation, an LLM, or other future components. It is **not** a special
speech-only boolean or an unconditional privacy whitelist.

The new `crates/zerogravity-core/src/context_filter.rs` defines a bounded
`ContextFilterPlan`: requested provenance-labelled record sources,
then ordered `include`/`exclude` predicates (record kind, exact
tool name, source origin, or literal text substring), and optionally
a source-bound calibrated **decision-model filter** that decides whether
each surviving record is relevant. Filtering is deterministic and
order-preserving; an excluded source cannot be recovered by any later
decision. Independent `tool_call` and `tool_result` kinds can be
retained or removed separately, preserving caller preferences.

Without explicit source selectors the default remains only
`session/user_message` and `session/assistant_message`.
The user can additionally request any authenticated context sources
such as `session/tool_call`, `session/tool_result`,
`application/system_prompt`, `application/developer_prompt` or
`shared:team-support/assistant_message`. The latter needs a real
named grant to access another user's shared history. This is an
**authorization check**, not a blanket prohibition on sharing.
Filters operate solely on the records actually provided under the
source owner's grant. This way the default is conservative but nothing
is hard-coded to refuse an application-owned prompt or an authorized
cross-user resource.

```yaml
# Fragment: config models include assistant and gate; see full example.
pipeline:
  - name: talk
    chat_context:
      enabled: true
      from_model: assistant
      max_bytes: 65536
      sources:
        - session/user_message
        - session/assistant_message
        - session/tool_call
        - session/tool_result
        - application/system_prompt
        - application/developer_prompt
        - shared:team-support/assistant_message
      filter:
        - kind: match
          action: exclude
          field: kind
          values: [tool_result]
        - kind: match
          action: exclude
          field: tool_name
          values: [database.lookup, filesystem.read]
        - kind: decision
          model: gate
          question: >-
            Does this record help interpret the current utterance?
          min_confidence_bps: 8000
          on_unavailable: omit
```

Equivalent repeated flags use
`--chat-context-source ORIGIN/KIND`,
`--chat-context-match exclude:kind:tool_result`, and
`--chat-context-decision gate:8000:omit:Relevant to audio?`.
A selected prompt/tool record can retain its exact structured provenance
in later runtime stages. No filter silently recasts a tool-result as a
higher-privilege instruction.

**Decision models are optional, composable components.** NanoJev-style
choice/boolean/score predictors are good examples, but several different
NanoJev implementations exist, so `nanojev-decision` is an *unqualified
model family intent*, not an implicitly selected checkpoint. It has a
typed include/omit result and confidence in 0..10000 basis points;
a configured threshold gates its verdict. `on_unavailable` accepts
`omit` (default), `reject` or `keep_authorized` (explicitly permissive).
Even the last policy may retain only records the owner already granted
and previous filters kept. No model weights, hosted API or decoder is
invoked by `pipeline check`; decision inference is a separate source
and runtime acceptance gate. Arbitrary model-supplied code and unbounded
prompts are not used as filters.

### In-memory source owner and snapshot selection (implemented in core)

`crates/zerogravity-core/src/context_snapshot.rs` now supplies an actual
in-process source boundary, not merely a filter over arbitrary caller views:

```text
trusted application's authenticated source owner (ContextOwner)
  ├── commit(tenant/session, source, committed_turn, tool_name, text)
  ├── grant / revoke(exact reader session, exact source)
  ├── erase(record_id)         # source version changes
  └── reader(authenticated SessionScope)
        └── capture(ChatContextPolicy, current_turn, DecisionAdapter)
              [short lock: only source-requested AND owner-authorized records]
              [release lock: ordered deterministic stages + typed decision]
              [short lock: verify exact source/grant epoch]
                → immutable original-order ContextSnapshot
                  + record IDs, source and publisher provenance
                  + included byte charge and truncation marker
                  + decision metadata (record ID/stage/model/verdict; no text)
```

The local source owner implements **one exact tenant/session reader
capability**, prior committed-turn eligibility for that session's messages,
explicit app prompt grants for the same tenant and explicit
`shared:resource` grants (including authorized cross-tenant readers).
Selection never equates a config request with a grant. A reader cannot
mint a different reader or mutate grants. Only a trusted host/application
holding `ContextOwner` can publish, grant, revoke or erase data; a
request's authenticated scope must be supplied by that trusted host.
No checkpoint/plugin/user-supplied executable code is a source owner.

The owner is **concurrency safe and optimistic**, not a per-token global
lock: record selection copies bounded complete candidates while holding
the short source lock, then runs filtering/decision callbacks *outside*
the lock. It checks the original epoch again after decisions. Any
intervening grant, revoke, erase or source commit refuses the entire
snapshot. A later router must call `ContextReader::require_fresh`
at its admission/publication boundary and bind the source epoch to the
original session/TaskCoordinator state; previously delivered bytes
cannot be recalled after a grant is revoked.

Logical bounded-resource contracts: store up to 4096 immutable records
and 16MiB logical retained bytes; stage at most the latest 1024 complete
granted records/4MiB for filtering, with a configurable final output
budget up to 1MiB. On output overflow the snapshot retains the
contiguous **newest complete** selected records in original order,
marks `truncated`, and never clips UTF-8 or fabricates half records.
Source owner's explicit `erase` handles removal and invalidates older
epochs. Physical allocator/RSS admission and persistent/remote source
synchronization remain separate runtime ownership work.

The `ContextDecisionAdapter` receives the original authorized record
view and typed question/model/stage/record ID for one filter stage.
For independently authored NanoJev-style decisions, the owner can
supply a calibrated backend later. No backend silently runs now:
`NoDecisionModel` returns unavailable, triggering the chosen
omit/reject/keep-authorized policy. A hard adapter error propagates
instead of being silently promoted to permissive fallback. Neither
inference nor an API listener is started by snapshot capture.

**What this does not establish:** there is no connected chat-history
provider, automatic conversation permissions, cross-process grant
persistence, source-real trained NanoJev checkpoint, ASR integration,
live router streaming or physical GPU/SSD model. The local tests use
a trusted in-memory fixture and source-free decision adapters. Adapter
authorization, owner-authenticated durable providers, multi-model
routing and ASR/decision fidelity remain separate acceptance gates.

**Failure and audit contract:** actual runtime implementation should
record the selected source version/grant, ordered filter steps, typed
decision model version, confidence, and included record IDs without
logging sensitive raw text by default. Authorization is checked before
filter execution and remains revocable. The context snapshot is frozen
at the declared turn boundary, then capped at the actual admitted
memory budget; a stale session, source revoke or missing required
decision causes the configured explicit refusal/omission, not a new
permission or guessed answer. For experiments, source-real A/B tests
should compare ASR accuracy, speaker DER, and latency with no-context,
filtered context, full authorized context and decision-filtered context.

**Runtime privacy/correctness contract (not implemented yet):**

- The *request's own authenticated conversation/session* supplies a
  bounded, immutable, previously committed history slice or any
  other **explicitly requested source** from the runtime's authorized
  application scope. Prior turns, tool calls/results, app-owned system
  and developer prompts, plus other users' records shared under a
  named, actually granted source scope, are all eligible. A config
  requests records; it does not grant permissions, invent hidden model
  state or access another person's private data without authorization.
  For example `application/system_prompt` means the application's
  own provided prompt, not another provider's inaccessible internal
  instructions. Source provenance and explicit, revocable owner grants
  must be retained. Treat included prompts and tool results as
  **untrusted data** for downstream ASR/decision models, not new
  privileged instructions. The caller's declared data scope, actual
  authorization and output consumer are separate checked contracts.
- Freeze the snapshot and its generation/turn ID **before** ASR for the
  next audio segment. Newly generated text for that segment cannot
  retroactively bias the same segment; a barge-in or canceled/stale
  epoch must not publish that segment's context or corrected transcript.
  Bound memory/copy time and avoid per-frame lookup/formatting.
- ASR may use history as vocabulary/terminology/language hints **only
  if its original model/processor supports those inputs**. Whisper,
  Qwen ASR, Canary and Parakeet each require independent capability
  qualification. An unsupported model does not receive a fabricated
  prompt; use a separately verified transcript rescoring/translation
  component where appropriate. Output must retain timings,
  uncertainty and traceable unmodified acoustic hypotheses.
- Diarization can condition speaker naming/continuity on prior known
  identities **only after** the acoustic segmentation/speaker evidence.
  It must not manufacture speakers/timestamps or treat chat names as
  biometric matches. Maintain consent/privacy rules for named voices.
- Translation/correction uses its own separately admitted model
  stage, with speaker-tagged time alignment and original transcript
  retained alongside any revised interpretation. Phrase-level
  look-back cannot silently rewrite already final/played output.
  Context-consistency can help technical terms but cannot override
  contradictory source speech.

**Acceptance before enabling runtime:** source-real A/B tests with
context ON/OFF, named-entity WER/CER, translation adequacy, diarization
DER, timestamp preservation, speaker consent, bounded fanout,
latency/real-time factor, stale-turn/cancel/tenant isolation and
incorrect-hint adversarial cases. A shape-valid `pipeline check` reports
the context toggle but **does not read conversations or start models**.
`pipeline run` continues to refuse with NOT IMPLEMENTED.

**Image generation:** prompt → pinned text processor/encoder → generated
latents via source-exact diffusion/flow solver → VAE decoder → complete
pixel frames. A preview image/latent tile is not a final playable image.
Image **streaming** may emit only registered complete tile/frame stages,
with explicit generations, cancellation and delivery-ACK bounds.

### Realtime, backpressure and failure atomicity

Each stream track uses rational clocks: input audio samples, ASR windows,
speaker-tagged transcript segments, output codec frames, image latent steps,
image tiles, video frames and solver steps are different semantic domains.
A session's StateRoot commits accepted model/codec work only after actual
device completion. **Playable** output is a separate frontier from neural
state. For duplex audio, support VAD, barge-in, playback ACK, output credit,
jitter buffering and stale epoch rejection; cancellation stops new
publication and retains fenced GPU allocations until genuine completion.
Never conflate "transport write succeeded" with "audio played." Coherent
image stream output requires complete validated pixels at a legal boundary.
Slow consumers impose bounded queue credits rather than unbounded media
accumulation. Hard realtime is not promised; soft SLO-gated admission
may reject a model whose cold SSD misses make live audio impossible.

Cross-model buffers are immutable/leased and may transfer as zero-copy
CPU/UMA/GPU views only where one verified placement and lifetime contract
permits. Otherwise copy through an explicitly accounted transfer stage.
There is **one** physical ResourceArbiter, fairness owner and readiness
ledger. Sharing within a process is not a reason to bypass copy isolation,
tenant boundaries or eventual GPU event retirement.

### API/router lifecycle

Loading a config produces a validated generation with model/source/version
pins, dependency graph and route definitions. Only after separate source and
backend qualification may the router provision model/source leases, warm
selected residents, admit queue/physical budgets and bind authorized APIs.
Hot reload requires an atomic new generation, old-session draining, route
version pinning, and safe physical reclamation, with no silent live model
swap. Streaming and batch replicas must use the same owner and report
separate model-load/compile time and per-request TTFT/ITL/p95 latency.
No implicit public bind, token configuration or cross-model credential
inheritance. Route APIs can target one pipeline or expose individual models,
but they cannot invent an unavailable OpenAI-compatible endpoint or live
WebRTC transport.

## Optimization sequence, success gates and non-goals

After the model-source plan:

- **Qwen3.5 2B:** CPU/GPU exact trained source and tokenization, then
  source-preserving quantized AVX/VNNI + native CUDA/HIP/Vulkan kernels,
  prefill/decode kernels, KV and shared-resident memory, real measured
  latency/throughput/power and physical cancellation; Qwen3.5 0.8B baseline
  remains the faster regression canary.
- **Qwen3.8 27B:** lift the executable 4GiB cap using source-real retained
  safetensors shard readers and typed tensor extents. Implement **SSD
  streaming as a first-class placement tier** with quant-aware prefetch,
  fused CPU/GPU stages and overlap. Compare complete trained logits,
  StateRoot, exact tokenization, I/O/OS headroom, cold/warm TTFT/ITL and
  failed reads before tuning any speed assertion.
- **Gemma4 E2B → 26B-A4B → 31B:** exact model geometry and source-bound
  routing/PLE/sliding/value norm/soft-cap, quant kernels and residency
  optimizations independently; use one shared scheduler and measured
  complete requests, not raw synthetic FLOPS.
- **Qwen3.8 Flash-Next:** learn QSA micro-block index and Gated DeltaNet,
  MoE/hyperconnection layout, resident/SSD learned n-gram embedding bank
  and 4B MTP heads with source-exact verification. Important distinction:
  trained n-gram **embeddings are checkpoint tensors**, while an
  n-gram/suffix speculative decoder is a **separate runtime strategy**
  that may not need any new weight download. Validate every accepted prefix
  and RNG/KV/recurrent rollback.
- **Other families/media:** scale via registered operators/components and
  source-real backend/codec gates, without runtime forks. Prioritize
  Whisper/Canary/Parakeet ASR and Qwen TTS/ASR streaming stages, then
  Qwen Image2.1 latent/pixel streaming; source-verified GPU and quality
  benchmarks before claiming interactive readiness.

**Do not:** spawn a separate executable/server for every pipeline node,
silently accept a new license, auto-download withheld DeepSeek/GLM payloads,
guess Q3.5 quant aliases, import arbitrary checkpoint Python, conflate
JSON shape compilation with a complete model assembly, or accept
unmeasured SSD/graph/speculation claims.

## Actual public metadata-only acquisition evidence — 18 Oct 2026

All rows below were checked using **this repository's original** built
`zerogravity pull hf ... --revision main --dry-run` resolver, with
`HF_TOKEN` and `GITHUB_TOKEN` unset. Selection was either one original
`--file model.safetensors.index.json` with automatically closed shards
or an explicitly selected `--bundle --include '*'` raw tree.
Use these pinned commits only after checking source permissions and license.
No weight bytes or SDK were downloaded and no cache snapshot was published.

| Target | Verified Hugging Face repository | Exact pinned commit | Selected bytes | Coverage |
| --- | --- | --- | ---: | --- |
| Qwen3.5 0.8B | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 1,769,978,895 | indexed weights |
| Qwen3.5 2B | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 4,571,272,453 | indexed weights |
| Gemma4 31B | `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | 62,578,684,366 | indexed weights |
| Qwen3.8 Flash-Next | `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | 360,023,349,944 | indexed weights; not a proof all MTP/n-gram groups have independent quant recipes |
| Qwen Image 2.1 | `Qwen/Qwen-Image-2.1` | `d26bb61231c349cf6b7896fa83353113880e1ba3` | 33,134,949,561 | opaque bundle only |
| Qwen3 ASR 1.7B | `Qwen/Qwen3-ASR-1.7B` | `7278e1e70fe206f11671096ffdd38061171dd6e5` | 4,703,114,308 | opaque bundle only |
| Qwen3 TTS 1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `0c0e3051f131929182e2c023b9537f8b1c68adfe` | 4,520,218,951 | opaque bundle only |
| Whisper large v3 turbo | `openai/whisper-large-v3-turbo` | `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | 1,622,466,054 | opaque bundle only |
| NVIDIA Canary-Qwen 2.5B | `nvidia/canary-qwen-2.5b` | `b1469e1bba1cfe140205529c79c434ca47180960` | 5,119,159,377 | opaque bundle only |
| NVIDIA Parakeet TDT 1.1B | `nvidia/parakeet-tdt-1.1b` | `53276c6469d1f17a1352e30c4d11be3d0d7e9575` | 4,283,149,969 | opaque .nemo bundle, no archive extraction |


The Qwen3.5 0.8B and 2B canonical weight inventories are **indexed
single-shard files** named `model.safetensors-00001-of-00001.safetensors`;
requesting `model.safetensors` alone does not match that repository.
The pinned index file correctly expands that unusual one-shard name.
The two Qwen originals already have separate trained local GGUF
execution gates; their HF metadata observations do not convert the
GGUF baseline to an automatically runnable safetensors model.

For an opaque bundle, the selected byte total is **not** an assembly
attestation: the original runtime does not yet recursively bind nested
Diffusers, NeMo codec archives or MTP calibration into one executable
model. Community Flash-Next GGUF variants may omit the 4B learned MTP
layer; a source-exact quantized core must not claim MTP execution until
the MTP weights and their distinct precision/semantics are verified.
Learned 51B n-gram embedding tables are separate from speculative
n-gram/suffix prefix matching, which is an algorithm and may use no
extra model weights.

The original tokenless Qwen3.8 27B and Gemma4 E2B/26B indexed
metadata checks from the preceding phase remain recorded in
[the acquisition source handoff](../model-acquisition/README.md).
Neither metadata source observation nor CLI config validation
authorizes physical model downloads or inference.

## References

- [Current model acquisition & source trust](../model-acquisition/README.md)
- [Model execution](../model-execution/TODO.md)
- [Serving & speculation](../serving-and-speculation/TODO.md)
- [Multimodal](../multimodal/TODO.md)
- [SSD & residency](../memory-and-streaming/TODO.md)
- [Unified multimodal execution contracts](../../docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md)
- [Realtime QoS](../../docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md)
- [Shared resources and serving](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md)
- [Official Qwen Flash-Next source](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Official Qwen Image2.1 component assembly](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/model_index.json)
- [NVIDIA Parakeet NeMo archive](https://huggingface.co/nvidia/parakeet-tdt-1.1b/tree/main)
- [Meta Muse Spark model/license](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4)

**Source binding reference:** The [TianyuCodings NanoJev architecture](https://github.com/TianyuCodings/NanoJev) and the independent [nano-jev MiniLM decision model](https://github.com/shubham10divakar/nano-jev) illustrate distinct candidate families. No one checkpoint is presumed and no trained local decision inference is claimed.
