# History, design and coverage audit — 20 October 2026

Author: `/root/spine_review`.
Scope: documentation-only historical/coverage audit and preservation advice.
No implementation, build, tests, provider/operator/model execution, dependency
acquisition, source-weight/DSO/binary scans, Git mutation or shared-plan edits.
The coordinator owns the aggregate, archives/catalogue and current checkpoint.

## Preservation conclusion

Preserve **the exact October 19 handoff**, full current plan TODO snapshots,
the current canonical design bodies, historical recovered bodies and new
continuation notes. A short executive handoff alone cannot retain the original
V1–V5.1 rationale, decisions, rejected approaches, detailed model/media and
hardware requirements, all unfinished work and numerical qualification limits.

The coordinator's new sourcebook/run-note/recovered-history volumes are the
appropriate dated preservation layer. Original source paths and live TODOs
remain authoritative. The new archive must be labeled **historical text**, not
a second tracker, a new runtime proof, or authority to execute every embedded
command. Current user permission and current source/tests still govern.

This audit inspected the October 19 structure, current-state/design/decision
crosswalk, historical recovery sections and catalogues; the master design,
roadmap, decision register's categories/supersession rows and current navigation;
and the relevant research/prior-art catalogues. It is a coverage audit, not a
claim to have independently re-reviewed every line of every preserved historical
source or re-executed its results. Byte-complete archival validation belongs to
the coordinator's actual copy/catalogue checks.

## What the October 19 handoff already preserves

Source: `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md`,
5,776 lines / 602,982 bytes at inspection.

| Section/range | Context retained; do not replace with a current-status summary |
| --- | --- |
| Front matter, lines 1–429 | October 19 checkpoint, one-engine vision, V1–V5.1 chronology, owners/invariants, source/CPU/GPU/acquisition/pipeline boundaries, priority and command/test scope. Dated, with many subsequent supersessions. |
| Appendix A, lines 430–3869 | Complete October 17 handoff body: original architecture and implementation chronology, failures/corrections, decisions, recovery limits, exact 94-source nested reading catalogue and earlier open-work snapshots. |
| Appendix B, lines 3870–5506 | Full then-live plan/work-item texts. Keep these as dated snapshots even when the latest snapshot differs; they record what was unfinished at takeover. |
| Appendix C, lines 5507–5679 | Public/core/plan/research/prior-art/run-note catalogue. It names sources outside `.agents/docs` and `.agents/plans`; preserving only those two roots loses substantial design context. |
| Appendix D, lines 5680–5776 | Exact October 19 starting prompt. Commands, restrictions and implementation assertions are historical, not an override of current instructions. |

Appendix A is especially important: the early architecture and original
requirements are not reconstructible from only current code or a commit list.
The master filename contains October 10 but its body is the later **V4.3
synthesis**, not a frozen V1 specification.

## Durable design lineage and explicit supersessions

| Earlier design/status | Later ruling that must be stated beside archival text |
| --- | --- |
| V1 original engine and extreme-model ambition | Retained: one Rust-owned semantic execution fabric, ordinary and specialized models, legal fallback, source/state correctness, simple UX and demonstrated efficiency. It was research/design, not a claim all models ran. No universal speed percentage or deadline was approved. |
| V2/V2.1 independent compiler/resource/model sketches | Unified model/task/compiler/physical source semantics. Storage codec, temporary compute encoding, hardware placement and task control are distinct. GPU/NPU cooperation is an eligible specialization, not an automatic accelerator or performance claim. |
| V3 broad StateDomain `commit` sketches | V4 domains prepare private candidates. One TaskCoordinator/SessionAuthority publishes the whole compatible root after actual completion. A device command, peer, Ready event, speculative draft or individual domain cannot independently publish visible state. |
| Four permanent IR engines / model-owned kernel plan | V4 keeps two enduring representations: semantic program and compiled executable bundle. RIR/KIR are transient views; CIR is physical dependency/resource lowering. Model source describes math, not a second hardware scheduler. |
| General privileged workflow in inference | Minimal trusted TaskDriver expresses model-relative bounded control. Tool execution, app workflows and external authority remain outside the mathematical kernel engine. |
| One native-kernel-shaped provider for every machine | V4.1 admits natural closed kernel/graph/AOT/static-dataflow granularity through one provider seam; opaque hosted text services remain separate task-service adapters. Architecture-compatible is not physically supported. |
| Rank-local/distributed commit and one shared placement map | V4.2 separates compute, immutable parameters, mutable state and immutable cache placement plus actual transfer costs; remote workers prepare, coordinator publishes. No automatic peer enrollment, WAN speed illusion or transparent HA. |
| Independent per-model server/router hot loop | V4.3 distinguishes registered source, versioned alias/profile, physical resident replica and mutable TaskSession. Route/admit at coarse boundaries; preserve one executor, allocator and sampler. |
| Text-only token ontology | V5/V5.1 add typed multimodal values/assemblies, model parts, clocks, media frontiers, realtime/duplex/QoS and explicit Unknown external/playback outcomes. They extend the same ownership spine; most tasks remain unimplemented. |
| Early proposed TOML user configuration | Later owner chose **YAML** for pipeline/serving configuration. Master §14.5's proposed TOML is dated design, not current instruction. Direct simple CLI remains valid. |
| Packaging-only U-023 and early implementation pauses | Later explicit instructions resumed implementation in bounded phases. The most recent correctness investigation was then explicitly narrowed to **planning/investigation only**; the current comprehensive handoff is documentation only. An old resume prompt does not lift this pause. |
| All 16K/32K checks before any GPU work | U-027 deferred those extended-context prerequisites while retaining independent short/functional-8K gates and unresolved numerical investigation. Functional context, exact numeric context and physical resident context are different claims. |
| “Qwen 3.7 27B” conversational spelling | Observed source is **original Qwen3.8-27B**; do not bind a nonexistent/different revision by name. Actual source/config remains authoritative. |
| Probes directory as a parallel product | Reusable implementation moved into crates, qualification into tools, one workspace/CLI. No permission to resurrect probes or another model engine. |
| Broad left-right/cold-catalog optimization proposal | Contention-first review rejected speculative migration of an uncontended cold catalog. A proposal's repetition is not approval or measured benefit. |
| Mandatory archives/SHA inventories/generated handoffs | U-066–U-068 and current AGENTS/plan README use ordinary per-plan/per-run TODOs. The present explicit all-context request authorizes **this dated preservation copy only**, not permanent archival bureaucracy. Actual model/content integrity and numerical acceptance remain required. |
| Old instruction not to restore retired archives | Superseded only for the current explicitly requested documentation preservation: copy into the new dated archive, not old active paths; no historical task/branch/runtime is revived. |

Stable phase names require namespace qualification: roadmap **ZG-P12 is routed
sparse/expert execution**, while serving-plan **P12 is native GPU batching**.
Likewise the multimodal ledger's historical U-040/U-041 are not the canonical
decision register's different U-040/U-041. The records must retain their source
document namespace rather than merge matching numerals.

## October 19 current-state assertions now requiring explicit correction

These are supersessions to be supplied by the coordinator's current checkpoint
and executed receipts; this audit does not independently rerun their acceptance.

- “Full model B>1 fusion absent / fused HTTP refused” became obsolete after
  actual full-model B2/B4/B8 lowering, private roots, greedy feedback, dynamic
  ragged cohorts and accepted HTTP work. Do not repeat the old serialized
  cohort facts as the final implementation state.
- Cross-request batching does **not** imply temporal token-matrix prefill.
  State-only native prefix batching still processes a model step per prompt
  token. Current performance deficits and unfinished chunk-prefill work remain.
- “vLLM absent” is obsolete. Genuine pinned V1 and official GGUF plugin were
  acquired/built and both small loaders qualified. Plain/observed full-logit
  equality establishes observer transparency, not native/vLLM all-state parity.
- “Original 27B metadata only / learned bytes not acquired” is obsolete.
  Original source and bounded source-window/typed wide-role and H48 reference/
  replay work exist. Full 27B inference, live complete producers and 112-domain
  session/CPU/GPU/streamed execution remain unqualified.
- Original .8B/2B HF source and CPU references are now available. Official
  all-BF16 loading, restored source-F32 loading, and vLLM's actual loaded/cast
  profile differ. No single checkpoint hash proves all their operator outputs.
- The old “first observed GDN layer4/layer1” drift descriptions are not causal
  proof. Later captures located earlier default-bulk AV (.8B) and alpha/beta
  projection (2B) discrepancies. Fresh AR1/reference-control 8K closure does
  not retroactively replay the unrecovered historical token corpus or default
  bulk arithmetic.
- Accepted isolated CUDA graph correctness does not enable production capture:
  complete same-card command/constructor/event/storage/drop participation and
  safe capture/retirement exclusion remain separate work. Host-token tests are
  not GPU Ready or complete-participation proof.
- New source-only SiLU/SASS, provider, coefficient and graph proposals retain
  their exact per-proposal frozen/compiled/physical status. A later release
  build does not qualify a paused unrun matrix or the loaded-SASS prototype.
- Debug interleaved benchmark numbers were superseded by genuine unprofiled
  measurements. They still do not imply source-equivalent math, lossless stock
  activity coverage or a throughput win. SSE chunks are not token ITL.

Gemma4, broader modalities, speculation and distribution remain their separate
plan gates. Metadata/architecture/parsers must not be promoted to trained
inference support by copying a handoff.

## Historical recovery: positive evidence and limits

The October 19 handoff's recovery sections (559–591, 1075–1115, 1380–1447,
1919–1945) distinguish text preservation from Git ancestry:

- Reachable `7e61cb63efd0c1429aa96044f2f7d28fda1e59aa` historically contains
  the 23,979-line multimodal aggregate and earlier nested **94-source handoff
  through U-040/P07-G3b**, plus 13 appendices and three then-live snapshots.
  The coordinator is preserving these actual bodies, not manufacturing reports.
- Fourteen deleted historical paths and one replaced pre-product README body
  are recoverable there/at the documented `be95424` reference. “Missing current
  standalone path” never means every embedded copy was lost.
- The original **33-node commit graph**, later **129/191-source aggregate
  versions**, and the complete **70-report set** were not recovered in that
  historical audit. Some early members of the latter do survive in the 94-source
  nested archive. H4/H5–H25 verbatim reports were not found there; H15's exact
  original title/ID remained unknown. Retained summaries are not new originals.
- Historical labels such as `6d74f76` and `bfe672c` record approvals but were
  unresolvable in the rewritten graph. Preserve their text; do not synthesize
  ancestry or tell a successor to rely on a nonexistent Git object.
- This documentation audit did not perform new fsck/reflog recovery. A newly
  checked recoverable ref is evidence only for its actual listed bodies.

Preserved historical test logs/counts are reports of execution at their own
source/date. No archive copy or reference hash proves a test executed now.

## Catalogue and remaining coverage risks

1. **Outside-root context.** The existing October 19 catalogue names all six
   public Markdown documents, 24 research sources and 12 prior-art sources.
   They carry multimodal model/QoS decisions, NPU/UMA hazards, SSD requirements,
   specialization, Qwen/vLLM parity and licensed-source reasoning not fully
   repeated under the two originally requested roots. Keep the coordinator's
   complete sourcebook copies, not just links.
2. **Non-Markdown input.** Current roots contain five substantive text sources:
   `plans/qwen-parity/{CLI_OPTION_FRONTIER,API_ROUTE_FRONTIER}.csv` and
   `docs/handoff-2026-10-17/{paused-filters-oracle.py.txt,paused-filters.patch,paused-xet-index.patch}`
   (all under `.agents`). At inspection they total 1,795 lines / 90,561 bytes.
   The CSVs preserve explicit public/internal feature inventory; the old
   acquisition patches are superseded drafts. The new archive catalogue now
   lists all five; this closes the Markdown-only omission.
3. **Current source and external library.** The read-only source research
   library `/home/pvl/spaces/abird/research/inference/{analysis,refs/COMMITS.txt}`
   is external evidence referenced by the reconciliation dossier. The new
   document archive does not copy its whole source tree, model cache, SDKs or
   later baseline libraries. Preserve attributable pins/paths/license limits;
   do not describe this as reproducing every external dependency or benchmark.
4. **Run-note versus source coverage.** All available run Markdown is useful
   for failures/partial phases, but is not a snapshot of every ignored Rust,
   C++, Python, JSON/raw archive or patch candidate. The coordinator has now
   closed the identified paused-source preservation gap: at this inspection the
   archive catalogue has 1,440 entries, including 442 selected text items from eight
   unique unexecuted candidate roots, three MAIN/ROOT worktree diffs, untracked
   source and small teacher JSON. Exact unpatched
   `...ordered-silu-loaded-sass-design-20261020/prototype-v3/collector.cpp`
   (26,706 bytes) is preserved in `PAUSED_CONTINUATION_TEXT_21.md`, lines
   4442–4930, along with its frozen copy. It remains uncompiled/unreviewed;
   preserving it does not qualify it. Together with all 624 run Markdown,
   220 patch, 133 canonical and 15 recovered historical bodies plus six upstream
   notices, these dated copies retain the important paused continuation text.
   The separately replaced pre-product README body is now preserved from
   `be95424837deb7919adf34503db71d27d3ad3adb` in `RECOVERED_HISTORY_04.md`
   (40,447 bytes); the later pointer remains a separate current source. This
   closes the earlier 14-versus-15 historical-body discrepancy.
   This is selected textual preservation, not an assertion that all raw/binary payloads or
   executable external environments are included.
5. **Stale navigation.** Current `docs/README.md`/master still name retired
   `.agents/plans/{DECISION_REGISTER,AGENT_IMPLEMENTATION_PLAYBOOK,REQUIREMENTS_AND_TEST_TRACEABILITY}.md`
   and `accepted-baseline/TODO.md`; their current bodies are under
   `.agents/plans/archive/history/`. Deleted `TODO-SPEC.md` is superseded by
   AGENTS + `.agents/plans/README.md`. Final architecture review lives in
   `.agents/prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md`. Supply a current
   navigation/remap table without editing historical embedded bodies.
6. **Open IDs and proposals.** Preserve canonical U-001…U-070, P-001…P-096,
   O-001…O-074 and A-001…A-006 with their USER/PROPOSAL/OPEN categories; preserve
   separate D-01…D-20, R4, DMM/OMM and modality ledgers. Do not close an old OPEN
   item from a nearby later implementation unless its actual question was
   answered. Latest user decisions after those ranges need prose with the
   current authority, not retroactively invented IDs or approvals.
7. **Actual artifacts.** Raw execution/capture files can be intentionally
   omitted from a documentation deliverable. State that receipts/notes and
   test source describe them; do not claim the self-contained Markdown contains
   full binary logs, checkpoints, learned weights or executable environments.

## Latest GDN investigation must survive

`GDN_VLLM_ARITHMETIC_AUDIT.md` in this directory preserves the new source-only
report and all its observed small-source hashes verbatim. It records:

- Pinned vLLM `db9527a4…` and official plugin `e2b8ad53…` actual flat QKVZ/BA
  mapping, grouped heads, state axes, conv, norm epsilon, coefficients and casts.
- Distinct prefill BF16-Q/K + chunk64/WY and normal decode FP32-Q/K recurrence,
  pre-scaled Q, FP32 beta, FP32 fused norm and final storage boundaries.
- The first conditional postprojection source difference: old diagnostic rounds
  raw conv BF16 before SiLU, whereas vLLM applies SiLU before its BF16 store.
  This does not prove the first actual trained-model mismatch or an algebra bug.
- Correct separation of accepted GGUF/GGML, pinned Torch CPU and targeted vLLM
  CUDA profiles. In particular dynamic Q8 activation preparation must not be
  mislabeled a universal F32-activation dot route.
- Contraction/reduction/transcendental compiled-kernel proof still missing;
  no full-model parity, relaxed tolerance or new runtime support is claimed.

The current coordinator-authored correctness-first plan owns the proposed
implementation sequence. This audit is a preservation/coverage companion,
not a second implementation plan or permission to resume.
