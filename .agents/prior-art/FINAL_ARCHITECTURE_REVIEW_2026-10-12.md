# Zerogravity — Final Architecture, Editorial and Agent-Handoff Review

**Review snapshot:** 12 October 2026 · Architecture v4.3 · Publication dossier 0.1.  
**Status:** **DESIGN-ONLY REVIEW; NO IMPLEMENTED RUNTIME OR BENCHMARKS.**  
**Purpose:** provide the final, honest engineering/editorial verdict for the requested publication-ready whitepaper and complete LLM-agent implementation specification. This file is an issue/authority and residual-risk ledger, **not a second architectural source of truth**.

**Canonical root:** `/home/pvl/spaces/abird/src/zerogravity/`. The independent `/home/pvl/spaces/abird/research/inference` tree is read-only prior art. The repository has not been initialized with Git and the source has no Rust engine as of this review.

## 1. Final deliverables and ownership

| Deliverable | Current path | Review purpose |
|---|---|---|
| Whitepaper manuscript | `../../docs/WHITEPAPER.md` | Externally readable proposal: motivation, novel *composition* argument, architecture, research context, evaluation, risks and limitations |
| Primary engineering spec | `../../docs/ENGINEERING_SPECIFICATION.md` | **Cross-cutting normative ES-00…ES-15** ownership, math/state, compiler/worker, physical memory, distributed and router/API rules |
| Bibliography | `../../docs/REFERENCES.md` | Numbered academic/first-party source references and attribution, no invented original-source credit |
| Publication checklist | `../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md` | Claims, independent results, code/whitepaper license, author signoff and editorial release gates |
| Public docs index | `../../docs/README.md` | Routes readers/reviewers/agents, classifies normative versus source history |
| Implementation playbook | `AGENT_IMPLEMENTATION_PLAYBOOK.md` | Every stable P00–P23 phase has bounded goal, owned modules, reference/negative tests and handoff rules |
| Requirements traceability | `REQUIREMENTS_AND_TEST_TRACEABILITY.md` | U-001…U-022 → ES + invariants + P phases and Stage0 proof requirements |
| Current concise agent entry | the agent playbook | Minimal mandatory reading order and existing project/hardware/privacy status |
| Full handoff | the retired master handoff | Latest publication takeover at top, all previous v4.3/v4.2/v4.1 history preserved |
| Core architecture detail | `ZEROGRAVITY_MASTER_PLAN_2026-10-10.md` and current `../docs` contracts | Full specialized technical source, subordinated on cross-cutting ambiguity to integrated engineering spec |
| Source/historical evidence | `../prior-art/PRIOR_ART_*`, `../prior-art/DESIGN_EVOLUTION_*`, platform investigations | Attributed research, legal provenance, old plans **not** another runtime interface authority |
| Document integrity manifest | the retired integrity manifest | Byte-exact SHA records of every other current Markdown file, regenerated **last** |

**Rationale:** The project had become a very large chronologically accumulated specification collection. These entrypoints **avoid** asking every implementation agent to reinterpret hundreds of pages of overlapping historical notes before making a small code change. The detailed original plans and research remain accessible; explicit authority and traceability prevent accidental silent loss of decisions.

## 2. Cross-cutting technical findings and dispositions

| Finding | Risk/failure if implemented literally | Final disposition and normative source |
|---|---|---|
| **F-01: Per-domain commit versus global transaction** | incomplete KV/GDN/RNG state published during speculative failure | **RESOLVED IN DESIGN:** StateDomains `Prepare`; TaskCoordinator alone publishes an immutable whole StateRoot after all physical readiness. [Engineering ES-06](../../docs/ENGINEERING_SPECIFICATION.md); runtime source test NOT RUN |
| **F-02: Device compiler abstraction too narrow** | TPU/wafer accelerator simulated with thousands of host kernel RPC calls | **RESOLVED IN DESIGN:** one RegionExecutionProvider can own kernel, captured graph, AOT stage, or static dataflow at legal granularity, with explicit StateBridge and lifecycle. [ES-04](../../docs/ENGINEERING_SPECIFICATION.md); vendor SDK support unverified |
| **F-03: Overly many enduring IR/runtime formats** | enormous code/ABI and per-token traversal overhead | **RESOLVED IN DESIGN:** SemanticProgram and ExecutableBundle are enduring concepts; RIR/KIR transient, CIR physical dependency DAG inside bundle; source-specific full-runtime adapters not required |
| **F-04: Global resource manager as hot per-kernel lock** | GPU idle, central contention at 32 concurrent tasks | **RESOLVED IN DESIGN:** coarse global admission and local fast pools with generation/event leases; must measure actual O(1) hot path and pressure failure |
| **F-05: TaskProgram tempted to become agent tool workflow** | arbitrary process/file/network side effects in model kernels | **RESOLVED IN DESIGN:** small trusted TaskDriver, external authorized application/agent layer; no unrestricted ExternalAction in inference |
| **F-06: Same source mathematical operators across modes** | silent top-k/quant/attention semantic changes, false throughput wins | **RESOLVED IN DESIGN:** distinct semantic, numerical, task quality and sampling-distribution evidence; source model artifact separate from temporary physical compute |
| **F-07: Compute and cache location conflation** | remote full KV reads every token over slow network, or source-invalid prefix hits | **RESOLVED IN DESIGN:** independent compute/parameters/mutable state/immutable cache maps plus costed TransferPlan and StateCompatKey; remote attention/hydrate/direct-read alternatives are distinct |
| **F-08: Multi-machine state authority** | split-brain coordinator, network timeout frees active DMA, duplicate output | **RESOLVED IN DESIGN:** session/placement epochs, remote shard leases, Prepare/Ready cohort, fail-stop/checkpoint replay; durable HA/consensus not implemented |
| **F-09: More than one model means physical infinity** | uncontrolled out-of-memory, cold model thrash, active session eviction | **RESOLVED IN DESIGN:** RegisteredModel vs aliases vs ResidentModel/Replicas vs TaskSession; safe load/warm/pin/evict under ResourceArbiter |
| **F-10: Model load balancing could change target** | wrong checkpoint/quant/adapter for a user-requested model | **RESOLVED IN DESIGN:** exact model identity pinned, equivalent replicas only; optional explicit multi-model policy transparent |
| **F-11: OpenAI API facade mistaken for cloud parity** | wrong SSE/Responses schema, unexpected tool execution, private stored data | **RESOLVED IN DESIGN:** endpoint/task/field/version SDK test matrix, model-generated tool-call data only, privacy-first store and strict unsupported semantics |
| **F-12: Publication text overclaims implementation** | misleading “fastest”, generic any-model, supported TPU/wafer/tunnel capability | **RESOLVED IN MANUSCRIPT:** all performance ambitions labeled hypotheses and physical tests absent; exact capabilities classified PLANNED/REFERENCE/COMPILING/FUNCTIONAL/QUALIFIED only from evidence |
| **F-13: Project-named source architecture dependencies** | incompatible licenses or fragile upstream API ownership | **RESOLVED IN DESIGN:** project-neutral normative types, named sources with dates/links preserved in separate prior art; exact-file license review before any reuse |
| **F-14: Agent takeover requires hidden conversation** | next agent renumbers phases, copies source examples or repeats decisions | **RESOLVED IN DOCUMENTATION:** public whitepaper+engineering spec, the agent playbook, Traceability, the retired master handoff and negative-test matrix |

**Technical outcome:** a coherent and falsifiable set of ownership contracts is specified. **Implementation correctness or performance has not been demonstrated**; nothing above should be promoted to physical VERIFIED until code/fixtures/benchmarks pass.

## 3. Editorial and citation quality review

- The **whitepaper** is separate from the engineering specification. It presents a stand-alone technical argument and does not require a reader to inspect private conversation history. It includes abstract, problem, design principles, mathematical/runtime/distributed/serving architecture, source-backed related work, evaluation hypotheses, security, limitations and conclusion.
- The **engineering spec** is implementation-focused and organized as ES-00 through ES-15; illustrative Rust interfaces are clearly marked as *not stable public ABI or existing source*.
- The **references** are numbered and source attributed. Peer-reviewed reference R07 was corrected to **Ansor at OSDI 2020**, R05 to **DistServe at OSDI 2024**, and the OpenAI Responses documentation link corrected to the dedicated Responses overview. Primary proceedings and current API references were independently checked. These support research context only, not zerogravity benchmark claims.
- The initial integrated diagram incorrectly suggested ModelRouter/admission happens **downstream of** physical worker execution; this was corrected to clarify request admission → compiled execution → StateRoot publication, while resource reservations remain cross-cutting. A structural review also found three accidentally concatenated section headings (`ES-05`, `ES-08` and `ES-11`) caused by interrupted incremental writing. These were separated into valid headings and included in the navigable ES-00…ES-15 contents; a compact ES-15 glossary was added.
- Historical design/version appendices are **preserved, not silently deleted**; active authority ordering resolves contradictions without rewriting prior evidence. External project references are acceptable in whitepaper/bibliography/prior-art dossiers as citations, but not core architectural type names.
- The manuscript intentionally omits author attribution, unapproved original project license, DOI, public Git URL and experimental graphs. Those require owner and independent reviewer inputs rather than invention.
- The public manuscript is **technical-review-ready**; an actual public distribution still requires user signoff, license/byline and external link QA. Its claims should not be marketed as software features until implemented.

## 4. Open research and implementation risks that documentation cannot close

| ID | Risk / unknown | Concrete blocking proof or policy |
|---|---|---|
| **RISK-A — real Rust GPU code generation** | Can the chosen Rust-authored device kernel path match mature low-bit kernels under register/shared-memory constraints? | One actual source-verified model, independent reference GEMV/prefill/decode/quant and profiling on target GPUs |
| **RISK-B — enough generic semantic ops** | Novel future models may require new operators/effects not representable by current registry | Second truly different recurrent/sparse model with portable reference and closed-region fallback |
| **RISK-C — GPU StepProgram control** | Exact grammar/RNG/cancellation may force host synchronization despite graph capture | Device-native multistep with exact acceptance/stop reference and measured host sync, or honest host-loop fallback |
| **RISK-D — economical full-state COW** | Logical root swap may need costly physical recurrent matrix copy and multi-domain snapshot | Three-domain failure replay/copy benchmark and bounded physical memory under C1/8/32 |
| **RISK-E — mixed model residency** | Cold model load/eviction and program memory could thrash unified memory | True physical memory admits, 100-load dedup, 1–8 model workload P95/energy, active KV lease safety |
| **RISK-F — distributed network performance** | WAN/tunnel RTT can overwhelm benefits of remote attention/FFN/sequence shards | Authorized actual two-peer direct and relayed tests at matched model/quant/long context; capacity-only wins labeled honestly |
| **RISK-G — remote failure/retention** | No durable coordinator quorum or exactly-once output mechanism | Explicit product durability target and implementation of lease fencing/checkpoints/ACK replay if needed, otherwise fail-stop |
| **RISK-H — vendor SDK/source language** | AOT TPU/wafer/ASIC may lack legal direct Rust device compiler or StateBridge | Specific vendor toolchain/license/SDK/physical source proof or unsupported mode; keep generic CPU/GPU |
| **RISK-I — OpenAI API evolution** | Cloud schema, models, streaming events and SDK behavior change over time | Pinned SDK/endpoint/parameter capability matrix, conformance tests, explicit unsupported fields |
| **RISK-J — model/training release breadth** | Broad task names are not runnable model/codec support and optional autograd may bloat inference | One real qualified checkpoint+TaskDriver per advertised modality; training optional ParameterVersion seam only |
| **RISK-K — publication/provenance** | No byline, document/code license, stable venue or independent review yet | Resolve O-067…O-071 and run external link/license/peer review before public release |
| **RISK-L — architecture complexity growth** | Extensive docs tempt coding agents to implement every proposed abstraction up front | Playbook P00–P08 real vertical slice first; measure crates, code churn, build time and hot-path overhead before expanding |

These are **not excuses to add hypothetical frameworks now**. Each risk should become a small code/test/evidence ticket only when relevant to the authorized phase.

## 5. Requirements and work preservation

- User decisions **U-001…U-022** remain in `DECISION_REGISTER.md` with all P working decisions and O open items retained.
- Stable original implementation phases **ZG-P00…P23** remain unchanged and NOT_STARTED.
- Experiments **H01…H86** remain a research/proof program, not a to-do list that must all be implemented in the first release.
- Original Stage0 S0-00…S0-14, optional NPU S0-NPU-A…F, V4 S0-V4-A…J, optional fabric S0-FABRIC-A…H, distributed S0-DIST-A…L and serving S0-SERVE-A…J remain traceable.
- The complete previous v4/v4.1/v4.2/v4.3 handoffs, v2/v3 historical snapshots and source-license/source-specific research remain intact.
- The **current intended first executable** remains one trusted checkpoint/tokenizer/quant CPU reference, one TaskDriver, direct compiled region and resource event lease, full Prepare→PublishRoot and local terminal output; then same-engine API and model router.

## 6. Verification ledger and final closure procedure

The final review should run:
1. existence and nonempty length of whitepaper, integrated engineering spec, bibliography, public index, publication checklist, agent playbook, traceability, status/handoff and decision/roadmap;
2. all Markdown relative links, internal section anchors, code fences/UTF-8 and H1 headings;
3. exact user decision IDs U-001…U-022, proposal P-001…P-091, open O-001…O-071 without lost historical IDs;
4. exact unique phase headings ZG-P00…P23 and experiment H01…H86 headings;
5. all expected Stage0 gate IDs and separated source-named research vs normative architecture (brand names cited as prior art are fine);
6. no new implementation source files, Git, installed network services, model downloads or changed read-only source corpus;
7. regenerate the integrity manifest **last** and verify exact SHA-256 for every other project Markdown file.

**Observed final structural review (2026-10-12, prior to manifest refresh):** **70 Markdown files**, **672 Markdown links**, **56 internal anchors**, **16 ES sections**, **25 integrated invariants**, **23 numbered references**, **24 stable roadmap phases**, **86 experiment IDs**, **61 Stage0 gates**, **U-001…U-022**, **P-001…P-091** and **O-001…O-071** all verified present with **zero structural/link/anchor/normative source-branding errors**. A full archive file-content SHA verification is the final follow-up, recorded in the manifest and last tool output. These tests establish documentation consistency only, not physical model execution. The absence of source files is verified on the available project filesystem but does not prove anything about external model/hardware systems.

## 7. Publication and engineering verdict

**Architecture coherence for design review:** ready to submit for peer technical review, with known unimplemented risks explicitly named. **Executable performance/resource/security correctness:** **NOT YET VERIFIED**. **External/public distribution of manuscript:** pending owner-selected authorship and licensing/peer signoff. **Agent implementation handoff:** complete as a design-to-code contract, provided the new agent respects phase scope/actual source state and independently validates physical source/math before promoting capabilities.

**Success after implementation will mean**: a supported model runs simply and correctly, the runtime admits independent specializations without multiplying engines, and compatible heterogeneous/distributed/model-router workloads improve measured practical quality/latency/memory/energy under the exact stated safety and fidelity constraints. The present files describe a route to that goal, not an achieved benchmark result.
