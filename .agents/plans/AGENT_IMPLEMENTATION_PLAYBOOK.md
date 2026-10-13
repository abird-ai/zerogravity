# Zerogravity — implementation design reference

Read [the plan TODO index](TODO.md) for current work and
[AGENTS.md](../../AGENTS.md) for the complete work-tracking process. This file
retains phase/invariant guidance; its historical ownership sketches are design
references, not another handoff, tracker or mandatory document-update list.

## 2. Begin with a bounded phase contract

For a new agent task, record a small **work ticket** before changes:

```text
Ticket:
  phase / requirements IDs:
  user authorization and scope:
  exact current source revision / dirtiness:
  owner modules and files:
  upstream model/source/quant or device capabilities:
  expected new interface and invariants:
  portability/reference fallback:
  positive tests and deliberate negative tests:
  source permissions and physical safety:
  measurable performance hypothesis, if any:
  blocked policy/hardware requirements:
  rollout/status and handoff artifacts:
```

Do not spend weeks scaffolding all proposed backends, every TaskKind, a generic plugin ABI or all model families before a **real one-model CPU reference** passes. The project wants **a small number of coherent modules with strong tests**, not a sprawling architecture visualization implemented as empty traits.

## 3. Minimal Rust module ownership (first release, not frozen public ABI)

A healthy starting structure is one library crate and one CLI binary (or a second GPU/toolchain crate only if actual build isolation requires). These module names are illustrative:

| Module | Owns | Must NOT own |
|---|---|---|
| `artifact` | bounded GGUF/safetensors source/manifest/quant/tokenizer/token mapping | GPU execution or HTTP routing |
| `semantic` | registered exact SemanticOps, state effects, reference evaluator and model recipes | backend compiler choice or session mutability |
| `task` | trusted TaskDrivers, typed input/output frontiers and stage control | arbitrary external tool execution or driver memory |
| `compile` | legal region transformations, shape guards, RegionExecutionProvider selection and bundle metadata | HTTP, active session root or global model registry |
| `device` | CPU/GPU/NPU optional compiled handles, streams/events and local fast memory pools | TaskSession state publication |
| `resource` | global physical admission/reservations/pressure and typed resource lease policy | per-kernel global mutex |
| `runtime` | TaskCoordinator/StateRoot prepare and publish, request batching, StepPlan/WorkerReply | mutable checkpoint source or second model-specific server |
| `catalog` | immutable source and alias/profile registry + resident model lifecycle | per-token sampling/math |
| `router` | exact source/replica selection, queue/admission policy and state/cache affinity | its own device workers/sampler |
| `service` | CLI/Rust SDK and optional HTTP/SSE/API/admin adapters | model math or external tool execution |
| `network` *optional* | peer identity, topology, TransportProvider and RemoteRegionWorker messages | new semantic IR or second state writer |
| `validation` *test-only* | golden vectors, compatibility, property/fault tests and benchmark evidence | shipped driver/runtime dependency in CPU-only build |

Avoid literal per-module crates before compiler and source tests justify it. Types may live together to avoid circular dependencies. Make code review focus on **ownership** rather than the number of modules created.

## 4. Implementation sequence, Phase P00–P23

Every phase describes intended work, **not a completion claim**. The authoritative [Roadmap](IMPLEMENTATION_ROADMAP.md) has expanded tasks; this playbook states *first deliverable, required test and safe skip* per phase.

### P00 — reproducible project foundation

**Deliver:** minimal Cargo CLI binary (optional library added only with first actual engine), pinned toolchain/Cargo.lock, CPU-only build, `rustfmt/clippy/test` gates, dev-shell/Nix compatibility, test fixture/data rights policy and source/provenance ledger. **The packaging subset now exists and is locally tested (U-023); P00 remains PARTIAL until Nix release/module verification and remaining source/rights gates. Do not misclassify `--help` as an inference runtime.** Probe actual CPU/OS/physical memory **without altering drivers or root services**.

**Pass:** clean checkout builds and tests without GPU/NPU SDK, no foreign inference engine linked, documentation status updated honestly. **Do not:** create 12 empty crates or auto-download a model.

### P01 — physical accelerator source/codegen feasibility

**Deliver:** tiny Rust-authored native GPU source kernel on actually available hardware for representative GEMV/GEMM/quant/recurrent update shapes; record driver/compiler ISA and golden CPU reference.

**Pass:** actual GPU output/logits for tested kernel are correct; register/occupancy/kernel timing reported with source revision. If driver/compiler/hardware inaccessible mark **BLOCKED** and continue portable CPU reference. Additional AOT/wafer/NPU SDKs are optional, not gatekeeping CPU.

### P02 — reference operator/state truth

**Deliver:** exact math for the selected model's attention/projection/position/sampler and one effectful state domain, CPU independent oracle. Extend to reference recurrent/sparse as models demand.

**Pass:** source model tensor/vector/logit tests, branch/rollback and invalid state effect-order tests. **Do not:** substitute plausible but different attention or reduce routed top-k for speed.

### P03 — verified artifact, quant and tokenizer

**Deliver:** safe bounded real model source parser and tokenizer/template, tensor manifest and source SHA, relevant one quant codec. Explicit file limits and allocation overflow checks; no custom model Python execution.

**Pass:** official token pretypes/chat template, exact tensor dims/quant dequant/source vectors and malformed corpus; unsupported type returns typed error.

### P04 — effect-aware SemanticProgram and trusted TaskDriver

**Deliver:** minimal typed values/shape/effects and one text-generation TaskDriver, plus fake second StateDomain. The reference can execute a simple semantic program.

**Pass:** illegal effect reordering rejected; task loop/frontier valid; a second synthetic non-token task can be represented without fake target sampling. **Do not:** implement a generic agent workflow DSL.

### P05 — region candidates, dynamic guards and async compiled plan

**Deliver:** one legal optimized region, an explicit portable fallback, ShapeEnv/stride/ragged checks, prepared native kernel candidate and typed resource/event command DAG within ExecutableBundle.

**Pass:** legal optimized vs reference output/state, illegal shape rejected before launch, mock invalid lease/event/order tests, **no IR walk per token** in compiled path.

### P06 — useful CPU-only model executor

**Deliver:** source-verified small real model full forward/prefill/decode, CPU quant kernels and sampler, TaskSession state and simple CLI through same TaskCoordinator.

**Pass:** meaningful reference-checked user output/stop/usage, source quant/dtype parity and cancellation. **Do not:** declare GPU speed metrics or unsupported quant/model success.

### P07 — real GPU worker, local pools and event safety

**Deliver:** actual optional GPU backend with buffer generations, stream/dispatch/event callbacks, local subpool within global reservation and first direct compiled model regions.

**Pass:** full-source GPU/CPU parity for eligible shapes; delayed DMA and stale event generation tests; no global per-token allocator lock or obligatory CPU sampled-token round trip where native StepProgram supports it.

### P08 — first complete real-model vertical slice

**Deliver:** `run ./model.gguf` or equivalent from trusted model file through exact tokenizer, SIR, legal compiled region, device worker, state Prepare→PublishRoot and committed terminal output.

**Pass:** real checkpoint+reference, CPU and qualified GPU if feasible, explicit fallback when fast kernel disabled, no mandatory server/cloud/config or model-source download. **This is the key architecture viability gate.**

### P09 — canonical service/SDK plus model router entry

**Deliver:** same Engine library serving CLI and one loopback HTTP endpoint, exact model alias binding, ModelCatalog with immutable revision and basic on-demand local load. Implement tested `/v1/models` and nonstream/stream `/v1/chat/completions` as supported.

**Pass:** official SDK pinned-version conformance, semantic/source parity between direct CLI and HTTP, SSE usage/stop/UTF-8 and bounded cancellation, local binding/admin/auth baseline. **Do not:** add separate model API gateway.

### P10 — physical residency + full-state snapshots

**Deliver:** global source/resident context budget and page/slab pools, safe state snapshot/partial cache prefix and model load/warm/idle drain/evict. Distinguish registered model, resident model, replica and mutable TaskSession.

**Pass:** 3+ full state domains resumed correctly, real memory reservations and pressure/eviction tests, no active unsnapshotable KV or graph freed to load another model.

### P11 — complex hybrid attention/recurrent semantics

**Deliver:** source-exact registered latent/sliding/selected sparse and recurrence/GDN/KDA operators as demanded by chosen source fixture, state codecs and legal chunk vs decode.

**Pass:** long-context official logits/quality, snapshot/replay/speculative accepted-prefix, invalid position/indexer/tied state rejection. **Do not:** assume ordinary softmax partition algebra for SSM/learned indexers.

### P12 — routed sparse weights and tiered IO

**Deliver:** exact routed/shared expert math, grouped device assignments, resident expert slabs/generation and bounded host/SSD misses with correct state/IO.

**Pass:** randomized route overlaps, cold/expert cache failure/replacement and source model reference math. **Do not:** silently change trained router selection count to improve speed.

### P13 — first extreme hybrid model region

**Deliver:** one very different complex model's legal source operators plus hand-tuned/compiled whole-region candidates, realistic prefill/decode/verify and state/layout fallback.

**Pass:** full model/task quality and resource state, independent matched performance where tested, forced generic fallback without a new model runtime.

### P14 — second complex family confirms modularity

**Deliver:** second mathematically different model, e.g. different sparse/recurrent/attention family, using same Artifact/Task/Region/Device/StateRoot and service.

**Pass:** real source quality, bounded new operator/recipe code change and no modifications to unrelated model server/allocator. Track module/line delta.

### P15 — target-authoritative speculative decoding

**Deliver:** optional trained/independent/prompt draft source, exact target verifier/correction and all-domain provisional/rollback with per task RNG/grammar.

**Pass:** sampling distribution/reference tests, rejected tail state exact, no preliminary draft output and net full-task gain before default promotion.

### P16 — high-concurrency multi-model and stage-aware scheduling

**Deliver:** native batched rows, model-specific queues, warm/cold priority fairness, equivalent replica selection, state-affine routing and typed stage budgets (token/encoder/audio/denoise), resource/tunnel cost-aware placement policy.

**Pass:** C1/2/4/8/16/32 P95 and useful outputs/energy, many resident model load/eviction under pressure, no global per-token router dispatch or starvation. Revert scheduling complexity that loses against simple policy.

### P17 — broad model roles, adapters, embeddings

**Deliver:** more validated ModelDefinitions, adapters with exact keys and encoder/pooling/rerank task interfaces; composed target+draft and encoder+rerank model roles with separately reserved artifacts.

**Pass:** new source/adapter/model quality, independent reference vectors and model alias/task/endpoint capability matrix. Reject incompatible KV source/adapters.

### P18 — ASR/TTS/image/video TaskDrivers

**Deliver:** one true supported source model per claimed task, trusted stage loop, task-specific stream/out units, codec/state and source quality vectors on same executor.

**Pass:** task-specific official/independent reference and cancellation/output validity; do not fake audio/image endpoints with LM tokens.

### P19 — networked multi-device, layer/state and fabric

**Deliver in safe order:** fake remote closed-region peers then **authorized** real two-node layer+KV colocated pipeline, full state transfers/remote prefix, P/D and attention/FFN when cost-qualified, token-range attention with registered algebra, optional RDMA/multi-GPU/graph fabric and remote state/epoch recovery. Coarse router may balance equivalent replicas or select a sharded deployment.

**Pass:** reference logits/full state, wrong quant/peer/lost ACK/timeout, no split brain, physical link direct/relay P95 and capacity-vs-speed honesty. **Do not:** connect actual machines or tunnels absent explicit user authorization, nor build full cluster consensus as first implementation.

### P20 — bounded offline optimization and cost model

**Deliver:** measured variants, full task/region cost and shape/program cache, cold/warm/placement optimizer with hysteresis and negative experiments. Separate Compile/Load/Warm from StepPlan dispatch.

**Pass:** stable matched model/task real performance improvement; bounded optimizer memory/compile overhead and no per-token JIT/search. Unqualified candidate remains disabled.

### P21 — security, source rights and reproducible release

**Deliver:** hardened parser, model source/license policies, HTTP loopback/auth/TLS/admin model control, remote peer identity/privacy, rust-owned kernels and vendor SDK/foreign code provenance, Nix/OS build/release packaging and optional model state retention policy.

**Pass:** fuzz/denied remote/SSRF/prompt secrecy/driver error recovery and release/source NOTICE evidence. No hidden cloud account or auto privileged power changes.

### P22 — release-quality numerical and physical performance matrix

**Deliver:** full source checkpoint/operator/quant/driver/hardware/task/serialization quality tests and benchmark harness with raw confidence, TTFT/ITL/P95/committed output, memory/energy, local/run/API overhead and eligible multi-peer cost.

**Pass:** published wins/ties/losses with matched baselines, no untested TPU/NPU/wafer model “supported” badges, official SDK and verified model source. Hardware not available is **BLOCKED**, not silently assumed pass.

### P23 — long-term extension and future technology proof

**Deliver:** deliberately add a new ModelDefinition, new registered SemanticOp/StateDomain, another device/program provider, new model/alias profile, a non-token TaskDriver and a TransportProvider/topology while keeping old code functional. Optional synthetic TrainStep/ParameterVersion seam demonstrates future training compatibility without training product scope.

**Pass:** measured change radius, no duplicated serving engine/session writer/memory manager, safe source/state migrations or cache invalidation and complete regression suite.

---

## 5. Test programs as grouped work, not 86 mandatory day-one tasks

The [Integrated Stage0](INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md) includes:
- **S0-00…S0-14:** fundamental source kernel/format/semantic/state/compiler/end-to-end proofs;
- **S0-NPU-A…F:** optional hardware source/compiler/safety/quality/perf;
- **S0-V4-A…J:** prepared StateRoot, device-native hot loop, hierarchical allocation, shape and minimality;
- **S0-FABRIC-A…H:** fake graph-only/static dataflow provider then optional real compiler/hardware;
- **S0-DIST-A…L:** independent placements, peer state/tunnel and failure;
- **S0-SERVE-A…J:** one-command local, model/alias, residency, equivalent replica, config and API conformance.

The [H01…H86 Experiment Program](EXPERIMENT_PROGRAM.md) contains research hypotheses and benchmark matrices. **Do not run every experiment by default**; choose only relevant hypotheses for the phase, with a documented baseline, code version, hardware and quality controls. Simulated resource/transport and source model physical tests have different promotion levels.

The [Requirements Traceability](REQUIREMENTS_AND_TEST_TRACEABILITY.md) links important ES/INV and U requirements to stages and test obligations.

## 6. Implementation priority / hard dependencies

```text
P00 (lean Rust CPU build + fixture rights)
   +-- P01 (GPU source feasibility, independent optional blocker)
   +-- P02 (math oracle) -- P03 (real format/tokenizer) 
                    \      /
                     P04 (effects + TaskDriver)
                       |
                     P05 (compiled plan + leases)
                       |
                    P06 -> P07
                        |
                       P08 (REAL model through CLI)
                        |
                 P09 model router/API
                        |
             P10 physical context/residency
                        |
        P11/P12 hybrid + sparse correct math
                        |
             P13/P14 extreme models
                        |
             P15/P16 speculate/batch
                        |
             P17/P18 breadth of model/tasks
                        |
             P19 distribution, optional SDK fabrics
                        |
              P20/P21/P22 release & qualification
                        |
              P23 re-run extension drill
```

This is a **rough dependency spine**. Some work can parallelize when owned interfaces and tests are independent: parser fuzz and CPU source math, shape guard and resource event simulator, SDK adapter mock conformance and model residency simulation, CLI documentation and source license audit. Avoid freezing interface prematurely or stubbing all phases at once.
