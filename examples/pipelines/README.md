# Pipeline YAML examples (structural, not runnable yet)

These files describe one-process/single-model and multi-model configurations
for Zerogravity's single binary. The built
`zerogravity pipeline check --config FILE.yaml` command validates the model
target names, stage ports, graph connectivity, optional context source
policies and route declarations. It **does not** download trained weights,
execute a model/codec, start an API server, or stream images/audio/video.
`pipeline run` explicitly returns **NOT IMPLEMENTED** until the original
source/physical/runtime gates pass.

| YAML example | Source-neutral intent |
| --- | --- |
| [text-model.yaml](text-model.yaml) | Single Qwen model, OpenAI-style chat route |
| [voice-assistant.yaml](voice-assistant.yaml) | ASR → diarization → Qwen → TTS, concurrent chat route |
| [transcribe-diarize.yaml](transcribe-diarize.yaml) | Audio transcript with per-speaker diarization |
| [contextual-voice-assistant.yaml](contextual-voice-assistant.yaml) | Optional chat hints to speech/translation |
| [contextual-voice-filtered.yaml](contextual-voice-filtered.yaml) | Configurable tool/app/shared source and typed decision filters |
| [audio-codec.yaml](audio-codec.yaml) | Audio PCM ↔ codebook tokens |
| [image-generation.yaml](image-generation.yaml) | Original QwenImage2.1 image-generation stage |
| [image-latent-assembly.yaml](image-latent-assembly.yaml) | Prompt → solver latents → image pixels |
| [video-stream.yaml](video-stream.yaml) | Text → future video frame stream and SSE route declaration |
| [decision.yaml](decision.yaml) | NanoJev-style typed decision verdict |

## Quickstart

```sh
zerogravity pipeline check --config examples/pipelines/text-model.yaml
zerogravity pipeline check --config examples/pipelines/contextual-voice-filtered.yaml

# Same typed graph expressed using repeated flags:
zerogravity pipeline check --name chat --input text --output text \
  --model assistant=qwen3.5-2b --step answer=llm:assistant \
  --link input.value=answer.text --link answer.text=output.value \
  --route /v1/chat/completions=openai-text

# Does not run until source-real trained execution and serving are qualified:
zerogravity pipeline run --config examples/pipelines/voice-assistant.yaml
```

## Configuration rules

Canonical input is **YAML**, one `schema: zerogravity.pipeline.v1` document.
Use `model:`, `pipeline:`, and `route:` lists. Each pipeline has typed
`step:` and `link:` lists. A model referenced from several pipelines
describes one shared instance; that never implies a new process or
a duplicate model allocation.

YAML comments, flow mappings, scalar aliases, anchors, shallow `<<` merge
keys and block/folded strings are accepted when the resulting typed schema
is valid. Duplicate keys, unknown schema fields, custom tags, recursive
aliases and multiple documents fail closed. The loader caps config files
at 256 KiB, expanded AST nodes at 16,384, nesting at 32, and expanded
scalar bytes at 256 KiB, then checks the typed graph/model/context bounds.
An anchor cannot grant context, start a model, or import executable code.
**TOML is not a supported pipeline config format.**

Users may explicitly include tool calls/results, application-owned prompts
and granted shared-user records; this is a *source request*, not a grant.
The trusted source owner decides actual permissions, and the future
runtime must revalidate the snapshot at delivery.

The architecture and deferred execution TODOs are in the
[composition contract](../../.agents/plans/pipeline-composition/README.md)
and the [model-execution](../../.agents/plans/model-execution/TODO.md),
[serving](../../.agents/plans/serving-and-speculation/TODO.md),
[multimodal](../../.agents/plans/multimodal/TODO.md), and
[memory/SSD](../../.agents/plans/memory-and-streaming/TODO.md) plans.

The YAML parser is the already-cached, version-pinned `serde_yaml 0.9.34`,
which upstream has marked deprecated. A maintained replacement requires a
separately verified dependency upgrade; it is not a justification for
unreviewed package downloads in the priority Qwen/Gemma execution phase.
