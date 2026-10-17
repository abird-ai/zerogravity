# Zerogravity plans

Provenance commit: `d5aa5cb`.
Workflow: [AGENTS.md](../../AGENTS.md). Each linked TODO holds that plan's
implementation, tests, commit history and unfinished work. Read specifications
only as needed for the task; old handoff/status documents are historical.

The prior October 17 pause was lifted by the user's explicit resumption.
The current cutover is to finish the bounded primary model-pull v1 workflows
and proceed to model execution. The acquisition plan retains separately
qualified provider-compatibility/security extensions; a retrieved cache snapshot
never establishes trained model, license or GPU execution support.

Full context: [17 October comprehensive handoff](../docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md).
Takeover text: [next-agent starting prompt](NEXT_AGENT_PROMPT_2026-10-17.md).

- [x] [Accepted implementation baseline](accepted-baseline/TODO.md) — `6d74f76`.
- [x] [Simplify agent handoffs](handoff-cleanup/TODO.md) — `bfe672c`.
- [x] [Historical records — archive retirement complete](historical-records/TODO.md) — `9239719`.
- [ ] [Model acquisition — pinned model-pull v1 accepted; separately qualified provider parity remains open](model-acquisition/TODO.md) — `d5aa5cb`.
- [ ] [Model execution — CPU-first Qwen3.8 and Gemma4, then GPU](model-execution/TODO.md).
- [ ] [Numerics and context — independent8K drift, broader GPU and16K/32K](numerics-and-context/TODO.md).
- [ ] [CPU/GPU optimization — five GPUs/three CPUs after acquisition](cpu-gpu-optimization/TODO.md).
- [ ] [Serving, complete pinned Qwen parity and advanced decoding](serving-and-speculation/TODO.md).
- [ ] [Physical memory, full-state reuse and RAM/SSD/expert streaming](memory-and-streaming/TODO.md).
- [ ] [Multimodal model/task execution and realtime operation](multimodal/TODO.md).
- [ ] [Heterogeneous and authorized distributed execution](distributed-execution/TODO.md).
- [ ] [Compiler/platform extension, security, packaging and release](platform-and-release/TODO.md).
