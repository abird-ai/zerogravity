# GPU tokenization, execution control and text decoding roadmap

16 October 2026. User-requested roadmap addition following the discussion of
GPU tokenization, device execution scheduling and detokenization.

**Status: planned; none of the capabilities below is qualified by this document.**
This addendum extends the existing phase definitions without renumbering them.
Current implementation ordering and completion live in the
[CPU/GPU optimization TODO](../plans/cpu-gpu-optimization/TODO.md); this design
does not reopen completed backend work or qualify the capabilities below.

## Existing architecture and phase ownership

Use the [implementation roadmap](../plans/IMPLEMENTATION_ROADMAP.md),
[backend execution contract](../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md)
and [single-owner execution spine](../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md).
This makes existing device-native StepProgram and graph-execution ideas concrete;
it does not introduce another model runtime, scheduler authority or public DSL.

| Existing phase | Added deliverables |
| --- | --- |
| ZG-P03: artifacts, codecs and rendering | Exact tokenizer/decoder contracts, asset identity, CPU oracle and GPU-compatible vocabulary/merge representations. |
| ZG-P05: legal regions and async commands | Reusable command graphs and persistent execution candidates with explicit effects, resources, guards and rounding boundaries. |
| ZG-P07: accelerator providers | Capability-qualified CUDA/HIP graph replay, equivalent Vulkan command reuse where supported, device queues and persistent execution. |
| ZG-P08: complete task slice | End-to-end prompt bytes to committed output bytes through the same model engine, with declared CPU/GPU text placement. |
| ZG-P09: coordinator and workers | Bounded device-driven generation, asynchronous output delivery, CPU/GPU overlap and authoritative root/output publication. |
| ZG-P10: physical memory | Admission and original leases for tokenizer tables, text buffers, graph captures, device queues, decoder state and output rings. |
| ZG-P15: speculation | GPU stop/grammar/verification work where qualified, preserving target authority, RNG semantics and accepted-prefix rollback. |
| ZG-P16: batching and placement | Batch tokenization/detokenization and measured CPU/GPU placement across prompt sizes, concurrency and live model load. |
| ZG-P20/P22: tuning and qualification | Bounded profiling, negative results and complete latency/throughput/quality/resource comparisons. |

## Shared execution contract

Keep one Rust mathematical engine, immutable source/tokenizer identity,
independent per-session StateRoots and the sole SessionAuthority publisher.
Workers may compute private state/output proposals and execute prepared work;
GPU execution does not independently publish a root or a client response.

Reuse the existing physical ResourceArbiter, fair workers, original buffer
leases, actual completion and unknown-completion quarantine. Budget all added
device tables, graph storage, input/output staging and queue capacity. Report
host metadata and driver allocations separately where measurement is available.

Keep exact token IDs, output bytes and model numeric contracts. Graph replay or
fusion must preserve required FP32 rounding, physical stores and state ordering.
Never widen numerical tolerances to qualify a placement or execution strategy.

CPU and GPU text processing are explicit, separately qualified placements of
the same semantics. Choose placement before dispatch and report the choice.
An explicitly selected GPU path propagates failures; it does not silently
substitute CPU work. Qualified CPU paths remain available for short inputs,
unsupported tokenizer configurations and other declared placement decisions.

## 1. Profile the complete request before selecting optimizations

Measure tokenization, normalization/template work, admission/queue delay,
host submission, GPU execution, synchronization, sampling/stop processing,
detokenization and response delivery separately and end to end.

Cover short interactive prompts, long fresh prompts, reused prefixes, Unicode,
multiple simultaneous sessions and sustained batches. Report cold versus warm
tables/graphs, first-use compilation, host core usage, logical and observed
physical memory, TTFT, inter-token latency and completed useful throughput.
Include p50/p95/p99, cancellation, rejection and fairness in serving workloads.
Measure GPU text processing while model inference is active, not only in an
isolated tokenizer benchmark where compute and memory bandwidth are idle.

**Gate:** identify a reproduced critical-path cost and retain an equivalent
baseline. Kernel timing, host-only lookup timing and GPU utilization alone
cannot establish a request-level speedup.

## 2. Reuse prepared execution through captured command graphs

Prepare bounded graph variants from the existing typed model/operation plan.
Use CUDA Graphs and HIP graphs on qualified native providers; evaluate recorded
commands/replay on Vulkan according to actual capabilities. Record unsupported
features explicitly rather than equating native and Vulkan mechanisms.

Preallocate admitted parameter/input/output buffers. Bind token, position,
valid-prefix length and current private state safely for each execution. Guard
source identity, numeric policy, geometry/capacity, device/context, compiler and
all captured allocation generations. Invalidate or safely rebind graphs after
restore, reset, buffer replacement, device fencing or incompatible configuration.
Retain original leases through replay and actual completion, including graphs
whose contents refer to buffers skipped by a state-only prefix.

**Gate:** compare replay with the existing uncaptured path on the same trained
sources: full logits, all required state domains, ordered diagnostic traces and
zero signs. Exercise nonzero causal state, changing valid lengths, shape-band
boundaries, multiple sessions, snapshots, rollback, admission refusal and real
post-launch cancellation/quarantine. Report graph construction/cache cost and
extra memory alongside end-to-end latency; preserve qualified context limits.

## 3. Move bounded generation and task scheduling onto the device

Progress from graph replay to bounded device-side decode loops/conditional
graphs where supported, then evaluate persistent kernels with device task
queues and dependency events when profiling justifies them. Keep implementations
as provider execution variants of the same semantic plan.

Keep sampled token feedback, supported EOS/stop checks, per-step parameter
updates and eligible grammar/sampling work on device. Research device-side
batch/KV metadata updates within already admitted resource grants. GPU workers
must not invent global memory admission or a second session-state authority.

Use bounded work windows and output queues so cancellation, fairness and
backpressure remain observable. Define actual completion at the authorized
transaction frontier; return ordered provisional state/output receipts to the
host SessionAuthority. Publish only validated committed prefixes. Preserve
exact speculative acceptance, rejection, RNG and all-domain rollback semantics.
Evaluate CPU/GPU overlap as a separate strategy when host control remains useful.

**Gate:** exact results against the existing step-by-step executor, including
early EOS, token limits, stop/grammar boundaries, partial accepted prefixes,
concurrent sessions, queue saturation, stale epochs and real cancellation at
multiple device-loop points. Unknown completion retains all referenced originals
and fences the device. Measure latency, useful throughput and fairness; an
unbounded persistent kernel or fewer host wakeups is not an acceptance result.

## 4. Add exact GPU tokenization as an optional placement

Define source-bound prepared vocabulary, merge ranks and byte/normalization
tables from the original tokenizer assets. Qualify the complete configured
pipeline: normalization, added/special-token matching, pre-tokenization, BPE or
other selected algorithm, token-ID assembly, BOS/EOS and request policies.
Separate GPU BPE primitives from a qualified complete model tokenizer.

Start with an explicit supported configuration and extend by semantic rules,
not model-name guesses. Preserve official chat-template behavior. Any retained
host normalization or request-policy stage must be visible in execution receipts
and included in complete-call measurements. Feed GPU-produced IDs directly to
GPU prefill where possible; CPU ID readback is a diagnostic or declared caller
requirement, not an automatic dependency of the GPU consumer.

Batch independent inputs without merging their semantic boundaries. Evaluate
declared CPU placement for short inputs and GPU placement for long/batched
inputs. Base any adaptive crossover on measured complete-call costs, concurrency,
transfer/topology and available resources; no universal character-count threshold.

**Gate:** complete ordered token-ID equality with the official CPU tokenizer
loaded from the same exact assets. Cover empty/short/long text, all byte classes,
Unicode normalization and combining marks, whitespace/newlines, digits, special
tokens adjacent to ordinary text, byte fallback, merge-rank ties, batch seams,
changing/reused workspaces and malformed-input policy. Test relevant normalization
and pre-tokenizer variants rather than assuming that a similarly named Qwen
family shares identical assets. Qualification on one vocabulary or GPU does not
qualify arbitrary tokenizer.json files, Qwen3.5, AMD or another device.

## 5. Add exact GPU detokenization and output processing

Evaluate vocabulary-byte lookup, output-length prefix scans, byte assembly and
eligible incremental decoder/stop processing on GPU. Keep tokens and prepared
decoder tables resident where useful. Transfer bounded completed output chunks
at the API/console boundary, accounting for queueing and copies in measurements.
Batch independent sessions while retaining separate decoder state and frontiers.

Match the source decoder's byte mapping, special-token options, byte fallback,
normalization/cleanup behavior and incomplete UTF-8 handling. Stop strings can
span token or output-chunk boundaries. Decoder state must follow the authorized
committed prefix through rewind, speculative rejection, snapshot and cancellation;
never expose speculative text or emit a prefix twice after replay.

Keep the reference CPU decoder as a declared placement when it is cheaper.
Distinguish neural-network autoregressive decoding from detokenization: moving
token IDs into text bytes does not add or remove model GPU compute coverage.

**Gate:** byte-identical final text and equivalent documented streaming behavior
against the source CPU decoder across adversarial token/chunk boundaries,
Unicode fragments, special tokens, cleanup settings, stop trimming and rejected
drafts. Require bounded output admission, no session cross-talk and correct
rollback/cancellation. Prefer complete-request gains over isolated bulk byte-copy
speedups that increase interactive latency.

## 6. Joint placement and release acceptance

Retain independent choices for text encoding, model execution, execution control
and text decoding. A fully resident GPU model, GPU tokenizer, GPU detokenizer,
captured graph and device-driven generation are separate reported capabilities.
Avoid a single full-GPU flag that implies unqualified frontend/control support.

Compare relevant CPU/GPU text placements, eager versus replayed execution,
bounded device loops and persistent scheduling under identical model/tokenizer
assets, numeric policy, workload, hardware, power and cold/warm conditions.
Report native CUDA, native HIP and Vulkan results separately. Preserve failed
experiments and measured regressions; retain simpler qualified strategies where
they win. Do not claim that these studies already demonstrate an improvement
over another runtime or permit broader contexts/model families.

Implementation order is profiling, captured replay, bounded device execution,
then measured GPU tokenizer/decoder candidates and joint qualification. Text
codec research may proceed independently when it does not interfere with native
backend bring-up. Persistent scheduling follows a measured need and complete
ownership/fairness gates. Track each bounded implementation slice, source/model
identities, actual hardware/numerical results, failures and unfinished gates in
the owning plan/run TODOs under [TODO-SPEC.md](../plans/TODO-SPEC.md). Commit
verified implementation, tests and plan updates together; no generated handoff,
archive or documentation SHA closeout is required. Existing phase IDs remain.

## Research evidence and its limits

These sources motivate execution constructs, not dependencies to install or
foreign runtimes to embed. Recheck versions, licenses, actual source behavior
and measured claims before implementation; no download or SDK/device-access
authorization is granted by this roadmap.

- [vendor GPU tokenization](https://docs.rapids.ai/api/libcudf/legacy/group__nvtext__tokenize)
  supplies GPU BPE/WordPiece operations. Its primitives alone do not establish
  exact full-pipeline support for our tokenizer assets.
- [GraphTok research paper, v1](https://arxiv.org/html/2610.13325v1) describes GPU
  pre-tokenization, BPE and ID assembly with vLLM and other serving-stack
  integrations. Its tested
  configurations, CPU routing for short inputs, host Qwen NFC normalization and
  H100 NVL evaluation limit what can be inferred for our models/hardware.
- [llama.cpp CUDA Graphs](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)
  demonstrates reduced submission overhead through graph replay. It does not
  mean that tokenization or the complete serving scheduler runs on GPU.
- A vendor overlap scheduler
  overlaps CPU scheduling/response/stop work with GPU computation; evaluate
  overlap independently from actually moving that work onto the device.
- [Mirage Persistent Kernel paper, v1](https://arxiv.org/html/2512.22219v1)
  demonstrates in-kernel dependency scheduling and device batch/KV metadata
  updates. Reuse the ideas only under our numerical/resource/state contract;
  its results do not qualify source-exact fusion or our GPUs automatically.
- [vLLM tokenizer configuration](https://docs.vllm.ai/en/stable/configuration/optimization/)
  documents its standard Hugging Face tokenizer and optional Rust replacement;
  that does not imply default GPU tokenization.
- [SGLang detokenizer implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/detokenizer_manager.py)
  illustrates a host detokenization process with incremental text state. It is
  a semantic/placement reference, not evidence of a GPU text decoder.

## Implementation integration

Use the current plan TODO for ownership and applicable gates, with separate
agent run TODOs until coordinator integration. Preserve unrelated work and the
user-authored root README. Commit only verified, scoped changes and carry every
unfinished item forward. This roadmap remains a design reference; reading or
linking it does not establish implementation or qualification.
