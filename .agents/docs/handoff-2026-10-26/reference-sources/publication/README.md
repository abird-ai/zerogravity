# Zerogravity documentation

The whitepaper and engineering specifications describe the architecture. Current
implementation, tests and remaining work are tracked in
[the plan TODO index](../.agents/plans/TODO.md). Agents use
[AGENTS.md](../AGENTS.md), their plan TODO and their own run TODO.

## Read according to your purpose

| Audience | Start here | Then |
|---|---|---|
| **Technical reader, collaborator or reviewer** | [Whitepaper](WHITEPAPER.md) | [References](REFERENCES.md) and [Publication/Review Checklist](PUBLICATION_AND_REVIEW_CHECKLIST.md) |
| **Systems/ML engineer implementing the runtime** | [Engineering Specification](ENGINEERING_SPECIFICATION.md) | [Agent Implementation Playbook](../.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md) and the relevant subsystem contract |
| **Coding LLM or autonomous agent** | [Plan TODO index](../.agents/plans/TODO.md) | [AGENTS.md](../AGENTS.md), the selected plan TODO and its linked specifications |
| **Architecture/product lead** | [Whitepaper](WHITEPAPER.md) + [Master Architecture](../.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) | [Open decisions](../.agents/plans/DECISION_REGISTER.md), [Critical Review](../.agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md), [Final Architecture/Publication Review](../.agents/prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md), [Release Gates](../.agents/plans/RELEASE_ACCEPTANCE_GATES.md) |
| **Performance or correctness reviewer** | [Engineering Spec ES-11…13](ENGINEERING_SPECIFICATION.md) | [Benchmark/Numerics](../.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md), [State/Failure Invariants](../.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md), [Experiments](../.agents/plans/EXPERIMENT_PROGRAM.md) |

## Documentation categories and authority

### A. Publication-facing

- [**Whitepaper**](WHITEPAPER.md): cohesive, externally readable technical manuscript. Research context, architectural contributions, falsifiable hypotheses, explicit current limitations. **Proposed system**, not benchmark results.
- [**Engineering Specification**](ENGINEERING_SPECIFICATION.md): integrated normative requirements **ES-00…ES-15**, including interfaces, state machine, physical memory, models, task control, routing, distributed placement, API and acceptance tests.
- [**References**](REFERENCES.md): numbered primary papers and official sources, with source author benchmarking and license caveats.
- [**Publication and Review Checklist**](PUBLICATION_AND_REVIEW_CHECKLIST.md): technical/claims/editorial/policy gates for a public whitepaper version and future results revision.

### B. Canonical current core design (project-neutral)

The public Engineering Specification governs cross-component consistency. For more exact detail, use these normative source files:

| Concern | Source |
|---|---|
| Minimal ontology and ownership | [Modular AI Architecture](../.agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md) |
| Typed Rust extension seams | [Composition and Extension Contracts](../.agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md) |
| SIR and legal compiler transforms | [Semantic IR and Compiler](../.agents/docs/SEMANTIC_IR_AND_COMPILER.md), [Task/Compiler Refinement](../.agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) |
| Device commands and lifecycle | [Async Resource Command IR](../.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md), [Device Kernels](../.agents/docs/DEVICE_KERNEL_AND_CODEGEN.md) |
| Native, graph and static fabric devices | [Accelerator Execution Families](../.agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) |
| Logical StateRoot, workers and failure | [Minimal Execution/State Protocol](../.agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), [State/Cache Transactions](../.agents/docs/STATE_CACHE_AND_TRANSACTIONS.md) |
| Physical resource and cost | [Memory Residency and IO](../.agents/docs/MEMORY_RESIDENCY_AND_IO.md), [Cost Model/Autotuning](../.agents/docs/COST_MODEL_AND_AUTOTUNING.md) |
| Scheduler, batching, speculation | [Scheduler and Speculation](../.agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md) |
| Mathematical model source/quant | [Model Semantics](../.agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md), [Formats and Quantization](../.agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md), [Operator Catalog](../.agents/docs/OPERATOR_AND_FUSION_CATALOG.md) |
| Multi-model routing/residency | [Model Router Contract](../.agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) |
| CLI, configuration and API | [Zero-Config CLI/API](../.agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md), [Runtime Serving](../.agents/docs/RUNTIME_SERVING_AND_UX.md) |
| Distributed topology, remote state | [Distributed Placement](../.agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [Remote Peer/Tunnel State Handoff](../.agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md) |
| Multimodal tasks and NPU split | [Product/Model Pipeline](../.agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md), [GPU/NPU Cooperative Tasks](../.agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) |
| Security and fidelity | [Security and Reliability](../.agents/docs/SECURITY_AND_RELIABILITY.md), [Numerical Quality](../.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md), [Invariants](../.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md) |
| Rust workspace minimality | [Rust Engineering](../.agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md) |

**Authority rule:** latest explicit user decision and actual verified code/tests outrank static plans; for design-only work, the [Engineering Specification](ENGINEERING_SPECIFICATION.md) and current [Master Plan](../.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) take precedence over old subsystem appendices when inconsistent. Register contradictions rather than silently creating incompatible interfaces.

### C. Implementation plans and tests

- [Plan TODO index](../.agents/plans/TODO.md): one line per plan.
- [Agent workflow](../AGENTS.md): per-plan and per-agent checklists, commit
  provenance, coordinator integration and carrying unfinished items forward.
- [Architecture roadmap](../.agents/plans/IMPLEMENTATION_ROADMAP.md),
  [requirements/test design](../.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md)
  and [decision register](../.agents/plans/DECISION_REGISTER.md): design references.
- [Release gates](../.agents/plans/RELEASE_ACCEPTANCE_GATES.md) and
  [experiments](../.agents/plans/EXPERIMENT_PROGRAM.md): planned checks; completion
  and actual results belong in the selected plan TODO.

### D. Non-normative historical design and cited prior art

Retained for context and legal/provenance review, **not as additional architectural authorities**:

- [Prior-art and License Policy](../.agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md) and [Original Source Atlas](../.agents/prior-art/RESEARCH_SOURCE_ATLAS.md).
- [Compiler/Runtime Research](../.agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md), [Historical Reconciliation](../.agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md), [Strix Halo Dossier](../.agents/research/STRIX_HALO_OPTIMIZATION_DOSSIER.md).
- [GPU/NPU Source Investigation](../.agents/prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md), [Accelerator Platform Research](../.agents/prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md), [Distributed/Tunnel Research](../.agents/prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md), [Router/API Research](../.agents/prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md).
- Retired design snapshots are available in Git history. Their useful rejected-design rationale is retained in the [critical review](../.agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md); current work is tracked in [plan TODOs](../.agents/plans/TODO.md).
- The external pre-existing research library is **READ-ONLY**.

## Invariants in one paragraph

The model's actual trained mathematics and source quantization are authoritative. Device kernels and compiled whole-regions may be specialized without creating another inference engine or silently changing the model. Device/IO workers produce **prepared** state; only one TaskCoordinator publishes each full StateRoot and its corresponding ordered output. Global physical admissions and fast local device pools share one budget; an unavailable device or invalid shape uses a legal fallback. A ModelRouter resolves exact source and compatible replicas, not an unrequested alternative target. Network peers require explicit authorization, real link measurement and epoch-fenced state handoff. API compatibility is an independently tested feature/endpoint subset, not a promise of all cloud tools.

## Publication and implementation status

Consult [the plan TODOs](../.agents/plans/TODO.md) for implemented/tested scope,
failures and remaining work. A design, proposed experiment or packaging definition
does not establish measured quality, physical backend support or a release.
Publication/tagging/release still require the owner's authorization.
