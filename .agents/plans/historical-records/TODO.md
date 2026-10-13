# Retire old evidence and prepare historical archive removal

Provenance commit: `aeff567`.

## Uncommitted

### Historical archive deletion — prepared, not performed

- [ ] Retire [V3 design snapshot](../../prior-art/DESIGN_EVOLUTION_V3_SNAPSHOT.md) after retaining useful rejected-design rationale in the [current critical review](../../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md).
- [ ] Retire [V3 handoff snapshot](../../prior-art/DESIGN_EVOLUTION_V3_HANDOFF_SNAPSHOT.md) after updating inbound links. StateDomains prepare; the current coordinator alone publishes roots.
- [ ] Retire [V2 master plan](../../prior-art/PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md), using [the current master](../ZEROGRAVITY_MASTER_PLAN_2026-10-10.md).
- [ ] Retire [V2 master handoff](../../prior-art/PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md), keeping current user/open decisions and source attribution in the decision register and research/license documents.
- [ ] Retire [V2 roadmap](../../prior-art/PRIOR_ART_HISTORICAL_ROADMAP_V2.md), using [P00–P23](../IMPLEMENTATION_ROADMAP.md).
- [ ] Retire [V2 Stage0](../../prior-art/PRIOR_ART_HISTORICAL_STAGE0_V2.md), using [current delivery gates](../INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md).
- [ ] Retire [V2 experiments](../../prior-art/PRIOR_ART_HISTORICAL_EXPERIMENTS_V2.md), using [current experiments](../EXPERIMENT_PROGRAM.md).
- [ ] Retire [V2 acceptance](../../prior-art/PRIOR_ART_HISTORICAL_ACCEPTANCE_GATES_V2.md), using [current release gates](../RELEASE_ACCEPTANCE_GATES.md).
- [ ] Retire [old distributed parallelism](../../prior-art/PRIOR_ART_HISTORICAL_DISTRIBUTED_PARALLELISM_V4_1.md), using the current placement/peer contracts; preserve one PublishRoot across participants.
- [ ] Retire [the multimodal aggregate](../../research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md) after accounting for its useful user/open decisions against the separately retained v5/v5.1 contracts, acceptance plan and seven research dossiers. Do not reproduce embedded old status/version copies.
- [ ] Migrate useful current build/release/Nix guidance from [the pre-product snapshot](../../research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md), then replace its body with a useful pointer. Keep that exact path valid because the immutable root README links it. Separate GPU lock/probe instructions are obsolete.
- [ ] Retire [SSD handoff wrapper](../../research/SSD_STREAMING_FIRST_CLASS_ARCHITECTURE_AGENT_HANDOFF_2026-10-15.md) after moving live links to its retained residency contract, SSD acceptance and source atlas. Preserve queued-reader and mixed-codec quota regressions.
- [ ] Retire [specialization handoff wrapper](../../research/EXTREME_MODEL_SPECIALIZATION_DESIGN_AGENT_HANDOFF_2026-10-15.md) after moving live links to its retained integration/source atlas and EXL3/QTIP/hybrid/GFX contracts. Preserve SYK proposal/open status and negative findings.
- [ ] Retire [the old multimodal research briefs](../../research/MULTIMODAL_PARALLEL_RESEARCH_WORKSTREAMS_2026-10-14.md) after retaining their useful unanswered questions/negative fixtures in the active multimodal TODO and dossiers.
- [ ] Retire [v5.1 research/agent briefs](../../research/MULTIMODAL_V5_1_RESEARCH_WORKSTREAMS_AND_AGENT_BRIEFS_2026-10-15.md) after preserving any distinct modality acceptance gates.
- [ ] Before archive removal, update active inbound links from public engineering/index documents, the master/critical review/composition/placement specs, phase/release gates and decisions. Leave the root README bytes unchanged.

Keep historical rank/state commit, four enduring IR and obsolete archive-generator
instructions superseded. Current API/auth/trust/SLO/plugin/quality questions stay
in their active plan TODOs. No archive in this section is deleted by this phase.

## Committed

### aeff567 — retire legacy evidence reports

- [x] Translate completed evidence into concise committed checkboxes and real pending tasks in the existing plan TODOs.
- [x] Preserve useful source/numerical/ownership details in current technical references.
- [x] Delete obsolete phase reports, working notes and duplicate completion records after their useful content is integrated.
- [x] Update references to deleted records; keep root README and implementation/fixtures unchanged.
- [x] List historical archive deletion candidates, replacement references and any extraction still needed. Prepare only; preserve unique archives until this is done.
- [x] Run link/source checks, independent reviews and the complete repository check; record actual results briefly.
- [x] Commit this migration and carry unfinished archive work forward.
- [x] Identify the archive set and current replacement contracts; Git at `bfe672c` retains the original source history. Current specs/research and production/reference fixtures are outside this evidence deletion.
- [x] Remove the obsolete rejected-LR metadata JSON files; both were owned unchanged JSON with no active descriptor observed. Production/reference inputs and model/SDK/build caches are untouched.

#### Migration details

- [x] Retire 70 obsolete Markdown records: 42 GPU phase/working-note reports, 20 CPU/model/concurrency/checkpoint reports, six obsolete entrypoint aliases and two completed takeover/layout records. Their original contents remain in Git at `bfe672c`; no copies or new archives are created.
- [x] Integrate known accepted completion/results into accepted-baseline, model-execution and CPU/GPU TODOs. Preserve actual model content hashes, source512/internal-bit scope, native profiles and no-migration outcome.
- [x] Carry actionable 8K/source-schedule work, deadline/recovery, real physical memory release and future strided View provenance guards into current unchecked TODO items.
- [x] Preserve unique source rounding/quantization/padding and new-model axes/routing semantics in the current numerical/model technical references. Keep Gemma official revision URLs; remove obsolete result/evidence prose from that technical contract.
- [x] Redirect current documentation and test/tool runbook links to the plan owning the surviving results/tasks. Keep root-README-linked current-status and pre-product paths valid. Historical archive bodies are unchanged and their inbound/outbound link cleanup belongs to the queued archive step.

#### Validation

- [x] `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`: exit 0; 194 Rust executions passed, 0 failed, 27 explicitly ignored. Strict lint, release build, installer, shell/Nix formatting, CPU/reference self-tests and 18 native-wrapper fixtures passed. No model/GPU launch or download occurred.
- [x] Current first-party navigation: 114 Markdown documents have no missing local targets. The three unchanged missing links in the vendored CubeCL README are outside this cleanup. Static historical archive/brief links stay queued for their removal pass.
- [x] Independent GPU review: 144 links across 23 selected TODO/spec/runbook files resolve; math/ownership/View guard extraction retained and no completed phase reopened.
- [x] Independent CPU/model review: preserve geometry/source/math details. Corrected two wording ambiguities: shard byte-count parsing refuses all exponent notation; WideF64 GEMV has no residual buffers, while separately qualified FMA operators use FastTwoSum correction.
- [x] All 15 historical archive/brief bodies and root README match `bfe672c`; all non-Markdown implementation/dependency/test-fixture/vendor/CI/Nix files are unchanged.
- [x] Ignored target top-level audit found no further old evidence after the two LR inventories were removed. Hardware discovery file listings were stale and are removed; source/model/reference/SDK/build caches remain outside deletion.
- [x] Finished agent findings are integrated here and into their owning plans; completed run files and temporary logs are retired. No new raw-evidence/archive/provenance pipeline is introduced.

### bfe672c — starting workflow

- [x] Single plan index, per-plan TODOs and separate agent run TODOs exist.
- [x] New work uses these checklists; old duplicated evidence does not become a fresh verification requirement.
