# Zerogravity plans

Provenance commit: `d5aa5cb`.
Workflow: [AGENTS.md](../../AGENTS.md). Each linked TODO holds that plan's
implementation, tests, commit history and unfinished work. Read specifications
only as needed for the task; old handoff/status documents are historical.

The October 17 implementation pause was lifted, and original source-pull
v1 plus the one-process pipeline architecture were accepted in later signed
commits. The **current critical path is model execution**: Qwen3.5 2B
(with the accepted .8B reference), then Qwen3.8 27B owned shards/SSD,
then Gemma4 E2B/26B-A4B/31B, then Qwen3.8 Flash-Next. Only after those
source-real CPU/GPU and performance gates should pipeline serving,
context-history integrations, media streaming, NanoJev decisions and
non-blocking provider parity resume. All their exact TODOs remain
preserved in their owning plans. Source metadata/cached bytes and YAML
config validation never imply trained model or live API capability.

Full context: [17 October comprehensive handoff](../docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md).
Takeover text: [next-agent starting prompt](NEXT_AGENT_PROMPT_2026-10-17.md).

- [x] [Accepted implementation baseline](accepted-baseline/TODO.md) — `6d74f76`.
- [x] [Simplify agent handoffs](handoff-cleanup/TODO.md) — `bfe672c`.
- [x] [Historical records — archive retirement complete](historical-records/TODO.md) — `9239719`.
- [ ] [Model acquisition — pinned model-pull v1 accepted; separately qualified provider parity remains open](model-acquisition/TODO.md) — `d5aa5cb`.
- [x] [Pipeline architecture and YAML typed scaffolds](pipeline-composition/TODO.md) — complete as source-neutral contracts; trained models/router/media execution remain explicitly deferred in their owner plans.
- [ ] [**ACTIVE NEXT — Model execution: Qwen3.5 2B → Qwen3.8 27B/SSD → Gemma4 E2B/26B/31B → Qwen Flash-Next**](model-execution/TODO.md).
- [ ] [Numerics and context — independent8K drift, broader GPU and16K/32K](numerics-and-context/TODO.md).
- [ ] [CPU/GPU optimization — five GPUs/three CPUs after acquisition](cpu-gpu-optimization/TODO.md).
- [ ] [Serving, complete pinned Qwen parity and advanced decoding](serving-and-speculation/TODO.md).
- [ ] [Physical memory, full-state reuse and RAM/SSD/expert streaming](memory-and-streaming/TODO.md).
- [ ] [Multimodal model/task execution and realtime operation](multimodal/TODO.md).
- [ ] [Heterogeneous and authorized distributed execution](distributed-execution/TODO.md).
- [ ] [Compiler/platform extension, security, packaging and release](platform-and-release/TODO.md).
