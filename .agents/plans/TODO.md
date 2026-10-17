# Zerogravity plans

Provenance commit: `f22f759`.
Workflow: [AGENTS.md](../../AGENTS.md). Each linked TODO holds that plan's
implementation, tests, commit history and unfinished work. Read specifications
only as needed for the task; old handoff/status documents are historical.

Execution is paused at the user's request after wrapping the current acquisition
phase. The acquisition TODO records the verified checkpoint and preserved drafts.

- [x] [Accepted implementation baseline](accepted-baseline/TODO.md) — `6d74f76`.
- [x] [Simplify agent handoffs](handoff-cleanup/TODO.md) — `bfe672c`.
- [x] [Historical records — archive retirement complete](historical-records/TODO.md) — `9239719`.
- [ ] [Model acquisition — native transfers verified; selection/provider parity in progress](model-acquisition/TODO.md) — `f22f759`.
- [ ] [Model execution — CPU-first Qwen3.8 and Gemma4, then GPU](model-execution/TODO.md).
- [ ] [Numerics and context — independent8K drift, broader GPU and16K/32K](numerics-and-context/TODO.md).
- [ ] [CPU/GPU optimization — five GPUs/three CPUs after acquisition](cpu-gpu-optimization/TODO.md).
- [ ] [Serving, complete pinned Qwen parity and advanced decoding](serving-and-speculation/TODO.md).
- [ ] [Physical memory, full-state reuse and RAM/SSD/expert streaming](memory-and-streaming/TODO.md).
- [ ] [Multimodal model/task execution and realtime operation](multimodal/TODO.md).
- [ ] [Heterogeneous and authorized distributed execution](distributed-execution/TODO.md).
- [ ] [Compiler/platform extension, security, packaging and release](platform-and-release/TODO.md).
