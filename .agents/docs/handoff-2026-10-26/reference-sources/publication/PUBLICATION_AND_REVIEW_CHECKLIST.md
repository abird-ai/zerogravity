# Whitepaper publication and final architecture review checklist

**Publication candidate:** [Zerogravity Whitepaper v0.1](WHITEPAPER.md), [Engineering Specification edition 1](ENGINEERING_SPECIFICATION.md).  
**Draft date:** 12 October 2026. **Status:** reviewable **design manuscript**, **NOT a results paper**, not a fully cleared/legal-public release.  
**Scope:** technical accuracy, architecture coherence, source attribution, reproducibility, editorial style, accessibility, software/hardware qualification and agent handoff.

This checklist prevents a polished design document from being mistaken for a proven implementation. Exact completed/pending status should only be marked after the checks are run, not because an earlier agent said a manuscript looked complete.

This dated manuscript checklist is not current runtime status. Use the [plan index](../.agents/plans/TODO.md) and [TODO spec](../.agents/plans/TODO-SPEC.md) for current work, passing gates and committed completion.

## 1. Technical claims and source honesty

- [x] Whitepaper title/abstract clearly identify the work as a **proposal**, not implemented runtime or measured fastest serving engine.
- [x] Distinct claims for mathematical semantics, numerical agreement, checkpoint/task quality and target sampling distribution.
- [x] No first-invention claim for prior-art paging, tiling, speculative verification, disaggregation, source-compiler layering or external topology technologies.
- [x] No universal tokens/s, throughput/energy or percentage superiority is claimed without reproducing source/model/task/hardware.
- [x] Compatibility status is per model checkpoint/quant/task/shape/device/driver and event/state safety.
- [x] Proprietary vendor compiler runtime and Rust-authored native device source remain distinct; no fake universal TPU/wafer kernel support.
- [x] Transparent distinction between architecture-compatible, simulated, functional, performance-qualified and production-qualified.
- [x] A recognized model file isn't claimed to execute unknown trained operators.
- [x] No source or training/fine-tuning functionality claimed to exist currently.
- [x] The spec disclaims global durable exactly-once streaming and automatic distributed leader failover.
- [ ] When experimental code exists, add a **separate reproducibility/results appendix** with real model/quant/driver/OS/link/thermal/workload, independent quality, p95/energy, matched baseline and raw data. *Pending by design: no engine implementation.*

## 2. Architecture consistency

- [x] **One SemanticProgram and one ExecutableBundle** as enduring compilation/execution concepts; region/kernel/command views are internal, not four mandatory public dialects.
- [x] Trusted TaskDriver controls computation stages; external tools/agent network/file actions belong to a separately authorized application boundary.
- [x] One TaskCoordinator writer per TaskSession/shard; StateDomains only **Prepare**, device/CIR only Ready/Failed, TaskCoordinator alone **PublishRoot**.
- [x] ResourceArbiter coarse physical budget and fast device-local pools with generation/event leases; no GPU virtual memory mistaken for available RAM.
- [x] One RegionExecutionProvider handles native kernel/captured graph/AOT/dataflow/limited managed-state program as qualified.
- [x] Distributed placement separates compute, parameters, live mutable state and immutable prefix/cache, with costed TransferPlan and trusted peers.
- [x] ModelCatalog/ModelRouter/ResidencyManager use same model/session/device engine as direct CLI and HTTP; no model-specific duplicate runtimes.
- [x] Exact model routing never silently substitutes checkpoint/quant/adapter; user-authored explicit model-choice policies separate from same-source replica balancing.
- [x] Local one-command run is independent of server, cloud account, mandatory cluster or remote model pull.
- [x] OpenAI compatibility explicitly defined as tested endpoint/field/SDK/SSE subset.
- [x] First source-verified CPU/GPU vertical slice is a prerequisite to ambitious multi-model/distributed qualification, not a framework-only launch.
- [ ] When real source exists, confirm architecture with working multi-model, non-token and second backend **extension drills**, profiling and resource-fault injection. *Pending by design.*

## 3. Publication manuscript and bibliography

- [x] Manuscript includes abstract, problem statement, design axioms, component architecture, state/physical resource/distributed/router descriptions, numerical/security boundaries, falsifiable evaluation and limitations.
- [x] Related prior work is cited in a separate numbered [References](REFERENCES.md) dossier with primary official links; non-MIT project names appear there/source research, not as core architecture types.
- [x] Useful design rationale and attribution remain in current research/references; retired document versions remain in Git history.
- [x] Diagram includes a textual equivalent explaining node/edge semantics; no image or copyrighted figure imported.
- [x] Tables have explicit headings and enough surrounding prose; Markdown renders in ordinary repository viewers.
- [ ] **Author credit/byline, institutional affiliation and contact** need the project's owner to select a public identity. No personal or organization attribution was invented.
- [ ] Confirm the publication artifact's licensing and third-party permissions before public release, following the owner's [MIT project policy](../.agents/plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md) and [LICENSE](../LICENSE). Separate third-party material and formal publication clearance still require review.
- [ ] **Trademark, third-party material and publication legal review** remain owner responsibility; prior-source code/weights/license rights must be reassessed if implementation incorporates them.
- [ ] **Persistent publication date/version/DOI/canonical URL** to be assigned when publicly submitted; internal paths are local developer references, not external URLs.
- [ ] **External link rot audit at actual publication time** and citation-style proofreading should be repeated; references were researched as of 12 October 2026.
- [ ] **PDF/HTML publication artifact, accessibility/print QA and figure export** optional for release; no PDF was requested, generated or claimed in this design pass.
- [ ] **Independent technical peer review** and copyedit/author signoff still required for a formal release.

## 4. Implementation planning and verification

- [x] Public [Engineering Specification](ENGINEERING_SPECIFICATION.md) includes ES-00…ES-15, identity, interfaces, state/lease/worker protocols, fallbacks, network and API, negative tests, six end-to-end traces and release conditions.
- [x] [Implementation Playbook](../.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md) lists stable P00–P23 with bounded deliverables/independent proof and stop/blocked instructions.
- [x] [Requirements Traceability](../.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md) maps explicit user decisions U-001…U-022 to technical source/phase/test requirements and integrated invariants.
- [x] [plan TODO index](../.agents/plans/TODO.md), root README and [Document Index](README.md) provide ordered navigation and authoritative/research separation.
- [x] No implementation phase marked complete merely because an acceptance test was described.
- [x] Remaining publication, release model/task/backend, public API and network-trust choices stay explicit. Follow the selected project license and qualified vendor-runtime scope rather than reopening settled decisions.
- [ ] Before implementation, inspect the live source revision, working changes and agent ownership; follow [AGENTS.md](../AGENTS.md).
- [ ] Record actual test commands/results, relevant model-content identities, fixtures/hardware, failures and unfinished work in the plan/run TODOs. Group completed work under its landing commit after its gates pass, following [TODO-SPEC.md](../.agents/plans/TODO-SPEC.md); no generated handoff or separate proof inventory is required.

## 5. Automated integrity checks

Current publication documents should pass:

- headings and balanced fenced code blocks across active Markdown;
- all relative Markdown file references (and internal section anchors when possible);
- no accidental stale root path or broken canonical architecture references;
- all stable phase IDs ZG-P00…P23, experiments H01…H86, required S0 gate groups, U/P/O decision coverage;
- no non-MIT source project names used as core Rust type/interface names in normative documents, while named sources remain in prior-art research and bibliography;
- zero source/runtime/Git/network changes unless separately authorized.

Record applicable checks in the owning plan TODO and commit verified changes together. Structural checks establish document integrity only; they do not establish runtime correctness, current external URLs or publication approval. No generated archive or documentation SHA manifest is required.

## 6. Publication release decision

**Current recommended classification:**

- **Design manuscript ready for external technical review:** YES, subject to the final integrity audit and editorial check.
- **Ready for public distribution with credited authors/license:** **PENDING** public byline, publication clearance, owner approval and final citation/link QA under the selected project license.
- **Ready to publish experimental performance/reliability results:** **NO** — each publication claim still needs its matched source/task/device measurements and review; accepted implementation slices do not pass every publication gate.
- **Ready for implementing LLM agents to start a bounded authorized phase:** YES within the user-authorized scope, following the live repository checks and [current plan TODOs](../.agents/plans/TODO.md).
- **Guaranteed highest-performing or supports every hardware/model:** NO; those remain engineering goals.

An eventual results edition should retain the design/limitations, add independent reproducibility, source/driver/test matrices, exact negative results and measured comparison against contemporaneous baselines. It should not retroactively treat this design document as already proving those results.
