# Complete reachable source and documentation history — 26 October 2026

Author: comprehensive-handoff history lane, coordinated by `/root`.
Scope: documentation and read-only Git/source inspection; no implementation, builds,
models, device work or qualification reruns.

## Frozen boundary and complete coverage

- Implementation cutoff: `d74f59a18c6345dbea963861acd519c1282a26fa`.
- **272 reachable commits, 272 unique full SHAs, one root, no merge commits**.
  Every reachable commit through the cutoff appears exactly once below, including
  documentation, plans, tests, tooling, packaging and vendored source.
- The [original 245-entry lineage](../HANDOFF_ALL_COMMIT_LINEAGE_2026-10-25.md)
  ends at `d84853337bbd2830f50a9c4e704d266990fd2ce4`. Entries **1–245 below
  preserve its ordinal, SHA and subject exactly**; entries **246–272 are the
  27 subsequent commits**. The original archive was committed as entry246,
  `d0fff261904fcb7836c0f71d0682028a12fd53b4`, after the source snapshot it indexes.
- This new handoff's documentation commit necessarily follows the frozen cutoff;
  it is not part of this 272-commit source snapshot. The live plan queue remains
  authoritative; this is a historical reading index, not another tracker or gate.
- Order is oldest-to-newest **Git ancestry**, generated with
  `git log --reverse --topo-order`. Imported dates are not execution dates:
  entries29–46 carry recorded dates27 October–5 November2026 even though they
  precede much of the October implementation in ancestry. No future execution
  is inferred. Dates below are exact ISO8601 author metadata, with offset.
- Author and committer dates match except entry92 (committer
  `2026-10-18T15:29:42+08:00`), entry226 (`2026-10-25T12:05:19+08:00`) and
  entry227 (`2026-10-25T12:29:28+08:00`).

## Signature and classification interpretation

- Current local verification: **272 `G`, zero other statuses**, signer
  `pvl@prasannavl.com`, SSH ED25519
  `SHA256:4aB+NvRb1shCx4xjtZDr2Y7FzfHGph80eFtpF+pHuCI`.
  Git uses the configured `/home/pvl/.config/git/allowed_signers` in this
  environment. `G` records a good signature under that configuration, not a
  source, numerical, physical or release acceptance.
- The [25 October NIGHT handoff](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-NIGHT.md)
  reported `U` in its restricted environment. That remains a historical trust
  observation; the current metadata audit can verify the same reachable objects.
- `S` = source/test/tool/build/configuration paths only (**98 commits**);
  `D` = documentation/provenance/historical context only (**60 commits**);
  `M` = both (**114 commits**). Classification uses actual changed paths, not the
  subject prefix. `.agents/` preservation material is documentation even when it
  contains a draft patch, raw source quotation or JSON archive catalogue.
  README/license/provenance bodies are documentation. A test or source import is
  not a new production support claim.
- Entry1 has a `docs(design)` subject but also includes the Rust workspace,
  installers, Nix and GitHub build scaffold: it is deliberately `M`. Entry13's
  vendored licenses/provenance and entry46's oracle notes are `D` despite their
  `feat`/`test` subjects.
- The owner/path column summarizes every changed path by actual repository
  directory, with path counts; a single/two-path change names exact files.
  `A`, `M`, `D`, `T` mean added, modified, deleted, type-changed paths. Counts
  use `git diff-tree --root -r --no-renames`; a moved file therefore contributes
  its old and new paths. They are not line counts or independent implementations.
- Recorded subjects such as “qualify” are preserved verbatim. They describe
  historical intent; inspect the actual diff, owning TODO and original executed
  evidence before accepting any particular claim. This lane has not rerun them.

## Before the rewritten root: preserved beginning and recovery limits

The project vision predates the first reachable commit. The October10–12
V1/V2/V3/V4 designs, original user decisions, earlier phase reports and changes
of direction survive in retained text. The reachable root is the later synthesis,
not the project's first conversation and not a reconstructed original graph.

- [Master plan](../../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) is the later
  V4.3 synthesis despite its filename. [Archived decision register](../../plans/archive/history/DECISION_REGISTER.md)
  distinguishes user-approved U rows, proposals P, open O and assumptions A.
  Preserve their categories and supersessions rather than reading a proposal as
  an authorization or merging unrelated number namespaces.
- The [17 October handoff](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md)
  preserves the historical recovery audit and exact94-source nested reading
  catalogue; the [19 October handoff](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md)
  embeds that earlier handoff and its then-live TODOs. The
  [20 October historical coverage audit](../handoff-2026-10-20/HISTORY_DESIGN_AND_COVERAGE_AUDIT.md)
  explains the sourcebooks, recovered-history, run-note and draft preservation
  volumes committed in entries203–208.
- Reachable `7e61cb63efd0c1429aa96044f2f7d28fda1e59aa` (entry52, before archive
  retirement entry53) contains the retired23,979-line multimodal aggregate,
  including an older complete **94-source** handoff through U-040/P07-G3b,
  plus the fourteen later-deleted historical bodies and replaced pre-product
  README context described by those audits. Read through Git or the preserved
  [recovered-history volume](../handoff-2026-10-20/RECOVERED_HISTORY_01.md);
  absence of a former standalone path is not proof that its text is lost.
- The original **33-node Git graph**, later **129/191-source aggregate versions**
  and full historical **70-report set** were not recovered in that historical
  audit. Some early reports survive inside the94-source text. Historical approval
  labels `6d74f76` and `bfe672c` remain in plan prose, but must not be presented
  as commits in this rewritten reachable graph. No missing original SHA,
  ancestry, report title or later verbatim aggregate is fabricated here.
- Entry53 retires archive bodies; entry93 moves historical plans under
  `.agents/plans/archive/history/`; entries78 and203–208 preserve surviving
  history/drafts as dated context. Those commits do not recreate the unavailable
  original graph or authorize executing old prompts. Old source hashes remain
  historical identities, not current acceptance or mandatory inventory duties.

## Recent consequences and exact identity boundaries

The original NIGHT snapshot stops before the26 October source and evidence
changes. The following interpretation preserves the larger changes without
turning their recorded checks into fresh validation by this lane.

- **246–250, archival checkpoint → B4 evidence → all-model correctness.**
  `a857d45` repairs fixture lifetime/accounting and adds independent finite/sticky/
  status controls. `6610b61` records the complete attempted original26 scope:
  24 passes, an unqualified diagnostic export and the retained strict N1/B4
  numerical rejection with intentional quarantine; the whole original26 gate
  remains false. Later source-scoped finite API/HTTP and original30 comparisons
  must remain distinct. Its same-attempt C16 records a passing .8B comparison
  and a slower2B result, not an all-model speedup. `6d85559` adds common pinned
  vLLM/llama.cpp correctness tooling; `195099c` adopts the all-model policy while
  preserving operator accuracy, model quality and determinism as distinct tests.
  Current policy/owner order supersedes NIGHT's older performance-first prompt.
- **251–263, containment and naming.** `405a3e6` creates explicit format homes
  without implementing absent codecs. `d9a8ecf` replaces ambiguous correctness
  output-v1 names with phase/prompt/input/cached-token/request-batch geometry.
  `a5ed7de` moves family recipes/config/source roles into model crates and groups
  CPU operations; `c90b8cd` separates GPU provider/binding owners; `41e9337`
  retains projection/capture guards after those moves; `fc73457` migrates actual
  application/serving/export callers. These are assembled ownership changes,
  not independent runtime engines. `5c2fe53` restores two missing source-admission
  regressions recovered from preserved snapshots; `9c2c51a` records reviewed
  side-worktree retirement. `fe17625` moves the unchanged four-tap GDN scalar
  operation to CPU ops; `263ff79` removes GPU production model dependencies,
  retaining dev-only model fixtures and enabling the strict production DAG gate.
- **263–267, terminal ownership and truthful forensics.** `e7fc315` diagnoses
  the old local l5 container-stop incident as likely cleanup-timeout escalation,
  with historical sender/fatal signal/OOM still unproven. `50abaf7` handles
  TERM/INT and joins HTTP and inference workers before process exit;
  `c84b987` retains stop intent, streams, terminal/container observations and
  cleanup history. `4f427dd` records bounded shutdown validation. `cd4159d`
  extends shared owned-command evidence to references, including high-numbered
  descriptors, single reap and complete separate streams/EOF. Historical137
  alone still proves neither OOM nor which actor sent the signal.
- **268–272, capture/runtime separation and offline source candidates.**
  `5a18b5b` separates model-issued prompt inventory from actual native capture
  coverage; actual finite runtime admission stays unchanged. `61f095c` adds
  authenticated original .8B prompt47–50 discovery/full-ABI capture and exact
  current-schema N1 bundle staging. Original failed/incomplete parents stay
  failed; layout-only reuse does not transfer packets or numerical acceptance.
  `3612ecb` formalizes production/test boundaries. `1f08e55` uses shared rotary
  initialization for configured context, derives immutable upload extent from
  real ABI bindings, and moves test reporters/helpers into Rust test support.
  Remaining qualifier/fault/case-driven runtime dependencies stay open.
  `d74f59a` emits offline .8B48–50 recipe candidates from complete passing
  same-request reference captures. It does not install them or admit new native
  shapes, GDN source tuples, model execution, serving or performance.

### Equality claims refer to different objects

| Object / scope | Exact historical relationship | What it does not establish |
|---|---|---|
| Trained model and original reference semantics | Reference adapter distinguishes the retained42-file discovery source from50-file packed-capture source; ten changes/eight additions are explicit stage ancestry. The20 primary runtime/model/operator pins and original model/all24/all42/fullV/source/lifetime validators remain separately bound. See [reference contract](../../../tools/qwen35-cuda-reference/README.md). | Whole adapter/collector/package identity is not “unchanged original source”; discovery is not complete packed capture, and a failed parent is not accepted by sharing its images. |
| Original CUDA GDN executable tuples | `5a18b5b` records all96 original tuples/12recipes/2catalogues unchanged. The subsequent `1f08e55` diff moves a `#[cfg(test)]` helper out of `observed.rs`; actual `.cu`/PTX payloads remain separate executable identities. | Changed Rust selection/binding source or rebuilt binaries do not inherit old physical qualification. |
| .8B aggregate native capture | `39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d` remains the current finite capture identity. | Existing finite47/57/62/67/72 inventory does not admit new48–50 runtime shapes. |
| 2B aggregate native capture before `1f08e55` | `0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0`. | It cannot be relabeled as the post-refactor source identity. |
| 2B aggregate native capture at cutoff | `4d77cb6637843a57a17fc093ad1e5241ddf083bf8d9ade0c9bbaa118bc9448e2`. `1f08e55` removes only the unused `SOURCE_INPUT_SHA256` constant from `cohort_native/generated_2b.rs`; executable factory/packet rows and GDN source bytes remain unchanged, while the honest aggregate fingerprint changes. | Byte-identical packets do not make the whole changed Rust source/binary identical or prove fresh physical validation. |
| Offline emitter's neutral baseline | [Baseline pins](../../../tools/cuda-text-recipes/baseline-pins.json) bind the actual retained seven-file `4f427dd814f734afdad798b053141946c998347a` representation, with the old .8B/2B fingerprints, all6720sites per source and original39,845,888/34,603,008B metadata charges. The current source is allowed to differ; do not rewrite those pins to hide it. | An old baseline is not a current runtime loader/admission permit. |
| New offline candidate output | [Emitter](../../../tools/cuda-text-recipes/README.md) authenticates13-file ancestor,14-file reference tools and7-file neutral baseline. Actual48/49/50 additions produce81 new factories while preserving old packet/site bytes and independent ownership charges. Exact same-request reference/state/prior/fullV, whole-image, reflected ABI and exported entry checks remain required. | Candidate files are qualification data. No production file installation, native48–50 admission, GDN PTX/control extraction, model/worker/HTTP/fault proof or performance qualification follows. |

## Complete indexed chronology

### 1–53 — Design synthesis, engine bootstrap, oracle support and archive retirement

The rewritten root preserves the initial vision/research and build scaffold. Core source/state/resource owners, artifacts, CPU/Qwen/Gemma, audited CubeCL backends, facade/CLI and independent oracle tooling follow. Original phase numbering and earlier approvals survive as historical text; this ancestry is the rewritten graph.

| # | Full commit SHA | Author date (ISO8601) | Sig | Class | Recorded subject | Affected owners/paths; change counts |
|---:|---|---|:---:|:---:|---|---|
| 1 | `be95424837deb7919adf34503db71d27d3ad3adb` | `2026-10-13T12:00:00+08:00` | G | M | docs(design): product, architecture, prior-art and research plans | `.agents/docs/` ×41; `.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md` ×1; `.agents/plans/CURRENT_STATUS_AND_TRACKING.md` ×1; `.agents/plans/DECISION_REGISTER.md` ×1; `.agents/plans/EXPERIMENT_PROGRAM.md` ×1; `.agents/plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md` ×1; `.agents/plans/IMPLEMENTATION_ROADMAP.md` ×1; `.agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md` ×1; `.agents/plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md` ×1; `.agents/plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md` ×1; `.agents/plans/RELEASE_ACCEPTANCE_GATES.md` ×1; `.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md` ×1; `.agents/plans/TODO.md` ×1; `.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md` ×1; `.agents/plans/accepted-baseline/` ×1; `.agents/plans/cpu-gpu-optimization/` ×1; `.agents/plans/distributed-execution/` ×1; `.agents/plans/handoff-cleanup/` ×1; `.agents/plans/historical-records/` ×1; `.agents/plans/memory-and-streaming/` ×1; `.agents/plans/model-acquisition/` ×1; `.agents/plans/model-execution/` ×1; `.agents/plans/multimodal/` ×1; `.agents/plans/numerics-and-context/` ×1; `.agents/plans/platform-and-release/` ×1; `.agents/plans/qwen-parity/` ×2; `.agents/plans/serving-and-speculation/` ×1; `.agents/prior-art/` ×21; `.agents/research/` ×29; `.gitattributes` ×1; `.github/` ×1; `.gitignore` ×1; `AGENTS.md` ×1; `CONTRIBUTING.md` ×1; `Cargo.toml` ×1; `LICENSE` ×1; `README.md` ×1; `docs/` ×5; `flake.lock` ×1; `flake.nix` ×1; `install.ps1` ×1; `install.sh` ×1; `nix/` ×10; `rust-toolchain.toml` ×1; **A146** |
| 2 | `1e10b5cc1d5cf34832477bc83ef39117a765c6a0` | `2026-10-13T12:00:00+08:00` | G | S | feat(core): immutable source identity, state roots and physical resource ownership | `Cargo.lock` ×1; `Cargo.toml` ×1; `crates/zerogravity-core/` ×1; `crates/zerogravity-core/src/` ×10; **A12,M1** |
| 3 | `af9a5d3969c9a87951990223b6d289aba69c63fb` | `2026-10-14T12:00:00+08:00` | G | S | feat(artifacts): sharded source artifacts and model identity | `Cargo.lock` ×1; `Cargo.toml` ×1; `crates/zerogravity-artifacts/` ×1; `crates/zerogravity-artifacts/src/` ×12; `crates/zerogravity-artifacts/tests/` ×1; **A14,M2** |
| 4 | `b276fd01e421f1b7e416ae773610b5823b9da36a` | `2026-10-14T12:00:00+08:00` | G | S | test(artifacts): cache transactions | `crates/zerogravity-artifacts/tests/cache_transactions.rs`; **A1** |
| 5 | `586d440b144727bdd9acea7455afb27f6c38ae25` | `2026-10-15T12:00:00+08:00` | G | S | test(artifacts): qwen text contract | `crates/zerogravity-artifacts/tests/qwen_text_contract.rs`; **A1** |
| 6 | `782e6fc48f0924cc055232d56338da62b68b1a79` | `2026-10-15T12:00:00+08:00` | G | S | feat(cpu): portable CPU math and weights | `Cargo.lock` ×1; `Cargo.toml` ×1; `crates/zerogravity-cpu/` ×1; `crates/zerogravity-cpu/src/` ×3; **A4,M2** |
| 7 | `1fd12da7840151ba47ab43368b4e4f435aac3d3b` | `2026-10-16T12:00:00+08:00` | G | S | feat(model-qwen35): source-verified Qwen3.5 forward and tokenizer | `Cargo.lock` ×1; `Cargo.toml` ×1; `crates/zerogravity-model-qwen35/` ×1; `crates/zerogravity-model-qwen35/src/` ×6; **A7,M2** |
| 8 | `901ba9ae2f2ed531d992a6d657a37dc362518bd0` | `2026-10-16T12:00:00+08:00` | G | S | test(model-qwen35): source layout contract | `crates/zerogravity-model-qwen35/tests/source_layout.rs`; **A1** |
| 9 | `e6ecf02bb7d951e01e66b430a24553c690b4ac5b` | `2026-10-17T12:00:00+08:00` | G | S | feat(model-gemma4): Gemma contract, MoE and 26B variant | `Cargo.lock` ×1; `Cargo.toml` ×1; `crates/zerogravity-model-gemma4/` ×1; `crates/zerogravity-model-gemma4/src/` ×5; `crates/zerogravity-model-gemma4/tests/` ×2; **A8,M2** |
| 10 | `81642a9af4d88d928327a1012af1594168704956` | `2026-10-17T12:00:00+08:00` | G | S | test(model-gemma4): e2b contract | `crates/zerogravity-model-gemma4/tests/e2b_contract.rs`; **A1** |
| 11 | `7aa19c268026069ad916f6a69440dab602aaa94a` | `2026-10-18T12:00:00+08:00` | G | S | test(model-gemma4): family contract | `crates/zerogravity-model-gemma4/tests/family_contract.rs`; **A1** |
| 12 | `65b026f902501e85afad5fd31f3ca53968a3e836` | `2026-10-18T12:00:00+08:00` | G | S | test(model-gemma4): variant 26b contract | `crates/zerogravity-model-gemma4/tests/variant_26b_contract.rs`; **A1** |
| 13 | `b03c16451d65c33a05b3c495c1205240db4ccac0` | `2026-10-19T12:00:00+08:00` | G | D | feat(data): vendored CubeCL license and provenance | `crates/zerogravity-gpu/` ×3; **A3** |
| 14 | `f491e34fe55e67eed434ac577c32d44eca51430e` | `2026-10-19T12:00:00+08:00` | G | M | feat(data): vendor CubeCL cubecl-ir 0.11.0 | `Cargo.lock` ×1; `Cargo.toml` ×1; `crates/zerogravity-gpu/vendor/cubecl-ir/` ×60; **A60,M2** |
| 15 | `627c58ec80ff7b33fa1ce5086a729946c5b1cd44` | `2026-10-20T12:00:00+08:00` | G | M | feat(data): vendor CubeCL cubecl-cpp 0.11.0 | `Cargo.lock` ×1; `Cargo.toml` ×1; `crates/zerogravity-gpu/vendor/cubecl-cpp/` ×77; **A77,M2** |
| 16 | `3664f975e79242d8b1a12bc4eefe886f81a33c2f` | `2026-10-20T12:00:00+08:00` | G | M | feat(data): vendor CubeCL cubecl-cuda 0.11.0 | `Cargo.lock` ×1; `Cargo.toml` ×1; `crates/zerogravity-gpu/vendor/cubecl-cuda/` ×23; **A23,M2** |
| 17 | `f77ae37ec0317f39a2e1ce4466d82c645cd344da` | `2026-10-21T12:00:00+08:00` | G | M | feat(data): vendor CubeCL cubecl-hip 0.11.0 | `Cargo.lock` ×1; `Cargo.toml` ×1; `crates/zerogravity-gpu/vendor/cubecl-hip/` ×23; **A23,M2** |
| 18 | `7d768a080a84b6132595f7cadbc84b75b69106e4` | `2026-10-21T12:00:00+08:00` | G | M | feat(gpu): source-exact resident GPU execution and physical ownership | `Cargo.lock` ×1; `Cargo.toml` ×1; `crates/zerogravity-gpu/` ×3; `crates/zerogravity-gpu/src/` ×62; **A65,M2** |
| 19 | `a9723bceb8ad4aa4df639dbe660efcae6c22711b` | `2026-10-22T12:00:00+08:00` | G | M | feat(runtime): facade, semantic program, task drivers and state publication | `Cargo.lock` ×1; `Cargo.toml` ×1; `crates/zerogravity/` ×2; `crates/zerogravity/examples/` ×6; `crates/zerogravity/src/` ×1; `crates/zerogravity/tests/` ×1; **A10,M2** |
| 20 | `c3d0d0b06fb1548c8bf3db10f2965e5af52be9bc` | `2026-10-22T12:00:00+08:00` | G | S | test(runtime): gemma contract | `crates/zerogravity/tests/gemma_contract.rs`; **A1** |
| 21 | `4add50a023d4fb5dd989989cab75ac4298d46d29` | `2026-10-23T12:00:00+08:00` | G | S | test(runtime): layout api | `crates/zerogravity/tests/layout_api.rs`; **A1** |
| 22 | `b4a36b829031fba498e45bebb71a6cd8f01e7ef9` | `2026-10-23T12:00:00+08:00` | G | S | test(runtime): multimodal mm0 | `crates/zerogravity/tests/multimodal_mm0.rs`; **A1** |
| 23 | `a691e21c040d2dcf46a52f2cbcf0e0eb934c8369` | `2026-10-24T12:00:00+08:00` | G | S | test(runtime): qwen real model | `crates/zerogravity/tests/qwen_real_model.rs`; **A1** |
| 24 | `25ec3e27e3c208158e9763944026003005456461` | `2026-10-24T12:00:00+08:00` | G | S | test(runtime): qwen shared model | `crates/zerogravity/tests/qwen_shared_model.rs`; **A1** |
| 25 | `047e3b70bbcbd7acbb5176712208cc4bcc02dd29` | `2026-10-25T12:00:00+08:00` | G | M | feat(cli): single-binary inspect and run | `Cargo.lock` ×1; `Cargo.toml` ×1; `crates/zerogravity-cli/` ×2; `crates/zerogravity-cli/src/` ×2; **A4,M2** |
| 26 | `52e37313bff57c2a7e831f50e9236fa983e13d02` | `2026-10-25T12:00:00+08:00` | G | S | test(cli): CLI contract | `crates/zerogravity-cli/tests/cli.rs`; **A1** |
| 27 | `22e0cc5abb4a067a95324206c2881de9e3327427` | `2026-10-26T12:00:00+08:00` | G | M | feat(tooling): workspace GPU qualification crate | `Cargo.lock` ×1; `Cargo.toml` ×1; `tools/gpu-qualify/` ×4; **A4,M2** |
| 28 | `4d82c6704dad47875fa42bdec65f9a65de79722e` | `2026-10-26T12:00:00+08:00` | G | S | test(tooling): independent reference oracle llama_reference_layers | `tests/reference/llama_cpp/llama_reference_layers.cpp`; **A1** |
| 29 | `a2d6a02f1577554ea8dba669f77ba7b903c1bb29` | `2026-10-27T12:00:00+08:00` | G | S | test(tooling): independent reference oracle llama_reference_logits | `tests/reference/llama_cpp/llama_reference_logits.cpp`; **A1** |
| 30 | `a8ea8926b53d7b4a9c6d00f3d9be04906f8d64bb` | `2026-10-27T12:00:00+08:00` | G | S | test(tooling): independent reference oracle llama_reference_qk | `tests/reference/llama_cpp/llama_reference_qk.cpp`; **A1** |
| 31 | `6a0079eb167d1f1593fe2a9372039138cd949770` | `2026-10-28T12:00:00+08:00` | G | S | test(tooling): independent reference oracle llama_reference_recurrent | `tests/reference/llama_cpp/llama_reference_recurrent.cpp`; **A1** |
| 32 | `be88e30733f9459eda5f150bdbd70d9d0f8c6684` | `2026-10-28T12:00:00+08:00` | G | S | test(tooling): independent reference oracle llama_reference_rope | `tests/reference/llama_cpp/llama_reference_rope.cpp`; **A1** |
| 33 | `77778a2c258751c64c56c4ffd3dba2ff70f7296c` | `2026-10-29T12:00:00+08:00` | G | S | test(tooling): independent reference oracle llama_reference_softmax | `tests/reference/llama_cpp/llama_reference_softmax.cpp`; **A1** |
| 34 | `6015241a5f5c2bb7ee65715a998988a026ff800c` | `2026-10-29T12:00:00+08:00` | G | S | test(tooling): independent reference oracle llama_reference_value | `tests/reference/llama_cpp/llama_reference_value.cpp`; **A1** |
| 35 | `e190aff1fdd392696e8086010a7bbc1f12d3df4f` | `2026-10-30T12:00:00+08:00` | G | S | test(tooling): independent reference oracle llama_reference_value_padding | `tests/reference/llama_cpp/llama_reference_value_padding.cpp`; **A1** |
| 36 | `0aea8de66e3eb3b704e7d26687731a3e5b07d3bb` | `2026-10-30T12:00:00+08:00` | G | S | test(tooling): build-release-artifacts.sh | `scripts/build-release-artifacts.sh`; **A1** |
| 37 | `057987203d5de45ad268ba02cfc5084f76f48229` | `2026-10-31T12:00:00+08:00` | G | S | test(tooling): check.sh | `scripts/check.sh`; **A1** |
| 38 | `c74450a27b1e3080e9576b0f95922f17c7cdca8b` | `2026-10-31T12:00:00+08:00` | G | S | test(tooling): compare-attention-value.py | `scripts/compare-attention-value.py`; **A1** |
| 39 | `9f54d2435fe6af15b3eaf28980bc1039ad02b559` | `2026-11-01T12:00:00+08:00` | G | S | test(tooling): compare-qwen-layers.py | `scripts/compare-qwen-layers.py`; **A1** |
| 40 | `7458ffc286f7e72c70ac18acfd3f776186aa204a` | `2026-11-01T12:00:00+08:00` | G | S | test(tooling): cpu-qualification-gate.py | `scripts/cpu-qualification-gate.py`; **A1** |
| 41 | `73cb6e55e1795e2c13533f60621dc29619cf5e2c` | `2026-11-02T12:00:00+08:00` | G | S | test(tooling): native-gpu.py | `scripts/native-gpu.py`; **A1** |
| 42 | `80145a7f91724fb34a6f67e9972850a279982bf3` | `2026-11-02T12:00:00+08:00` | G | S | test(tooling): test-install.sh | `scripts/test-install.sh`; **A1** |
| 43 | `ca1138b8cf866c81bbfb276358d03a62576df16a` | `2026-11-03T12:00:00+08:00` | G | S | test(tooling): test-native-gpu.py | `scripts/test-native-gpu.py`; **A1** |
| 44 | `5a0636c3f9c744c2f979ded40943a269ba5a1176` | `2026-11-03T12:00:00+08:00` | G | S | test(tooling): test-numeric-resume.py | `scripts/test-numeric-resume.py`; **A1** |
| 45 | `9a6c504ebef037e09bd187d65e2bec7e2e456eef` | `2026-11-05T12:00:00+08:00` | G | S | chore(workspace): restore the locked full workspace manifest | `Cargo.lock`; **M1** |
| 46 | `57a9ad7fe78bf7c5e64d3870a6ba176ceba4cc07` | `2026-11-05T12:00:00+08:00` | G | D | test(tooling): independent reference probe notes | `tests/reference/llama_cpp/README.md`; **A1** |
| 47 | `1af2ac434279cb5f4b635a51884c936f19b54aa3` | `2026-10-16T21:39:45+08:00` | G | M | fix(nix): give module check a complete base system | `.agents/plans/platform-and-release/TODO.md`; `nix/tests/module.nix`; **M2** |
| 48 | `0452c594d99df6d3b4c972ceaae8df6ea56925e3` | `2026-10-16T23:45:32+08:00` | G | D | docs(plan): record module eval check fix | `.agents/plans/platform-and-release/TODO.md`; **M1** |
| 49 | `2afa32aa74aff7f35ec1b52e4caa7079a5f68980` | `2026-10-16T23:45:32+08:00` | G | D | docs(readme): reorder engineering and FAQ sections | `README.md`; **M1** |
| 50 | `a054d396e1ec12fc5e9b8e2ca60e8060954870c5` | `2026-10-17T08:33:07+08:00` | G | D | docs(agents): document plan layout and TODO spec | `.agents/plans/TODO-SPEC.md`; `AGENTS.md`; **A1,M1** |
| 51 | `cc3cab27a2713fdb64bb3f1a42c8138c060c5dce` | `2026-10-17T08:33:07+08:00` | G | D | docs(contributing): align guide and AI disclosure | `CONTRIBUTING.md`; `README.md`; **M2** |
| 52 | `7e61cb63efd0c1429aa96044f2f7d28fda1e59aa` | `2026-10-17T08:41:24+08:00` | G | D | docs(agents): gate pushes and destructive actions | `AGENTS.md`; **M1** |
| 53 | `9239719c25128028fd78a6a60f817ce6b626517f` | `2026-10-17T09:54:54+08:00` | G | D | docs: retire historical archive bodies | `.agents/docs/` ×4; `.agents/plans/DECISION_REGISTER.md` ×1; `.agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md` ×1; `.agents/plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md` ×1; `.agents/plans/RELEASE_ACCEPTANCE_GATES.md` ×1; `.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md` ×1; `.agents/plans/TODO.md` ×1; `.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md` ×1; `.agents/plans/cpu-gpu-optimization/` ×1; `.agents/plans/historical-records/` ×1; `.agents/plans/memory-and-streaming/` ×1; `.agents/plans/multimodal/` ×1; `.agents/plans/platform-and-release/` ×1; `.agents/prior-art/` ×10; `.agents/research/` ×13; `docs/` ×4; **A1,M28,D14** |

### 54–83 — Pinned acquisition, source cache, providers and companion closure

One immutable cache/acquisition implementation grows bounded resume, explicit components, HF import, HTTPS/OCI/GCS/S3/Azure/ModelScope and Xet. Offline/size/hash/range/selection guards are source behavior; a successful provider or metadata test is not model execution or a new download authorization.

| # | Full commit SHA | Author date (ISO8601) | Sig | Class | Recorded subject | Affected owners/paths; change counts |
|---:|---|---|:---:|:---:|---|---|
| 54 | `6deebf22228315863f563723b5e6761ac6b51c5c` | `2026-10-17T10:11:15+08:00` | G | M | refactor(cache): namespace model cache under abird/zerogravity | `.agents/plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md` ×1; `.agents/research/` ×1; `crates/zerogravity-artifacts/src/` ×1; **M3** |
| 55 | `fa6affa425089fdf9a15bfa10b330e03f738fec8` | `2026-10-17T10:30:09+08:00` | G | M | feat(artifacts): add pinned model pulls | `.agents/plans/TODO.md` ×1; `.agents/plans/historical-records/` ×1; `.agents/plans/model-acquisition/` ×2; `Cargo.lock` ×1; `crates/zerogravity-artifacts/` ×1; `crates/zerogravity-artifacts/src/` ×10; `crates/zerogravity-artifacts/tests/` ×5; `crates/zerogravity-cli/` ×1; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; `crates/zerogravity/` ×1; `crates/zerogravity/src/` ×1; `nix/` ×2; `scripts/` ×1; **A14,M16** |
| 56 | `3c97c52075563f65928c564818ce73502168778f` | `2026-10-17T13:44:29+08:00` | G | M | feat(artifacts): resume interrupted pulls | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `Cargo.lock` ×1; `crates/zerogravity-artifacts/src/` ×6; `crates/zerogravity-artifacts/tests/` ×2; `crates/zerogravity-cli/` ×1; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; `scripts/` ×1; **A4,M13** |
| 57 | `baa9d8818061958a63eabeb61a7239a103a738f0` | `2026-10-17T14:12:22+08:00` | G | M | feat(artifacts): load pinned cache components | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `Cargo.lock` ×1; `crates/zerogravity-artifacts/src/` ×7; `crates/zerogravity-artifacts/tests/` ×2; `crates/zerogravity-cli/` ×1; `crates/zerogravity-cli/src/` ×4; `crates/zerogravity-cli/tests/` ×2; `crates/zerogravity-model-qwen35/src/` ×2; `crates/zerogravity/src/` ×2; `crates/zerogravity/tests/` ×1; `scripts/` ×1; **A8,M18** |
| 58 | `20f449d20320b6f2912698c7b97cfdfd82472ba2` | `2026-10-17T14:37:44+08:00` | G | M | feat(artifacts): recover missing pull sizes | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×9; `crates/zerogravity-artifacts/tests/` ×2; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A2,M15** |
| 59 | `f44264c017856a015584a094566a077638939e55` | `2026-10-17T15:03:41+08:00` | G | M | feat(cache): add protected snapshot removal | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×6; `crates/zerogravity-artifacts/tests/` ×3; `crates/zerogravity-cli/src/` ×3; `crates/zerogravity-cli/tests/` ×2; `crates/zerogravity/tests/` ×1; **A6,M12** |
| 60 | `ea700f67fca7ff7871613044a989e03b8ad6f9d7` | `2026-10-17T15:23:55+08:00` | G | M | feat(pull)!: add capped auxiliary streaming | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×9; `crates/zerogravity-artifacts/tests/` ×6; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A3,M18** |
| 61 | `03b69482a5fb10b8c1cc7032e7cbdf39e2e5fead` | `2026-10-17T15:41:45+08:00` | G | M | feat(pull): pin multi-asset release components | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×2; `crates/zerogravity-artifacts/tests/` ×2; **A1,M6** |
| 62 | `e44f4ee8020d5661dd8167cb07dfcb085906fcb7` | `2026-10-17T16:16:36+08:00` | G | M | feat(cache): import local HF snapshots | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×4; `crates/zerogravity-artifacts/tests/` ×2; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; `crates/zerogravity/tests/` ×1; **A6,M7** |
| 63 | `fce1353e649df2eb73fe6a39437043fc8551aef5` | `2026-10-17T16:33:28+08:00` | G | M | feat(cache): qualify cross-filesystem imports | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×1; `crates/zerogravity-artifacts/tests/` ×1; **A1,M4** |
| 64 | `e0073a5593e5641494c0f48e8d2dcc9e9e61a119` | `2026-10-17T17:05:07+08:00` | G | M | feat(pull): add pinned HTTPS objects | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×3; `Cargo.lock` ×1; `crates/zerogravity-artifacts/src/` ×4; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/` ×1; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A5,M9** |
| 65 | `e82d76b8f8e91f7d4b49c211e26baa081ff523e5` | `2026-10-17T17:14:19+08:00` | G | M | feat(pull): add pinned OCI GGUF layers | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×3; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A4,M6** |
| 66 | `f74a525366e6280a9806ecaabd7e1f4df46a6a97` | `2026-10-17T17:26:28+08:00` | G | M | feat(pull): add version-pinned GCS objects | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×4; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A4,M7** |
| 67 | `833d68f716783c255b24691552a081a1d2b88e76` | `2026-10-17T17:51:00+08:00` | G | M | feat(pull): add versioned S3 and SigV4 | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `Cargo.lock` ×1; `crates/zerogravity-artifacts/` ×1; `crates/zerogravity-artifacts/src/` ×6; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A6,M9** |
| 68 | `e62222276e0dac62ed554a3fec2f2ae86edad497` | `2026-10-17T18:13:08+08:00` | G | M | feat(pull): add pinned Azure blobs | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `Cargo.lock` ×1; `crates/zerogravity-artifacts/` ×1; `crates/zerogravity-artifacts/src/` ×6; `crates/zerogravity-artifacts/tests/` ×2; `crates/zerogravity-cli/src/` ×3; `crates/zerogravity-cli/tests/` ×1; `nix/` ×2; **A7,M12** |
| 69 | `87f6454fb7c4832262d36104a806baad9a9734a7` | `2026-10-17T18:24:38+08:00` | G | M | feat(pull): add pinned ModelScope files | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×5; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A5,M7** |
| 70 | `223ca794c9d45a0e246ad1cd9ba8613f2f203cd4` | `2026-10-17T18:33:14+08:00` | G | M | feat(pull): add Xet hash and chunk primitives | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `Cargo.lock` ×1; `crates/zerogravity-artifacts/` ×1; `crates/zerogravity-artifacts/src/` ×2; `crates/zerogravity-artifacts/tests/` ×1; **A3,M5** |
| 71 | `a774ded154ed066c8f4b134ce7f5cba32ccdd09c` | `2026-10-17T18:45:39+08:00` | G | M | feat(pull): add cloud content contracts | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×6; `crates/zerogravity-artifacts/tests/` ×2; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A2,M12** |
| 72 | `837e77f5080c0c2c07fee90b2ef9fb791068a7d3` | `2026-10-17T18:55:23+08:00` | G | M | fix(pull): enforce offline acquisition rules | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×3; `crates/zerogravity-artifacts/src/` ×2; `crates/zerogravity-artifacts/tests/` ×1; **A2,M5** |
| 73 | `52e42fcdf0e7bead1ebefe55d8103501c7bd4f64` | `2026-10-17T19:15:13+08:00` | G | M | feat(pull): add native Xet reconstruction | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×13; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A9,M11** |
| 74 | `46cf5a5830e7e8d8096234341ef51fafdf63cc16` | `2026-10-17T19:24:22+08:00` | G | M | feat(pull): add conditional cloud reads | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×6; `crates/zerogravity-artifacts/tests/` ×2; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A2,M12** |
| 75 | `4db5efbcce973d21bf1c8ca750471e6b2ab956ab` | `2026-10-17T19:34:10+08:00` | G | M | feat(pull): recover pinned ModelScope sizes | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `AGENTS.md` ×1; `crates/zerogravity-artifacts/src/` ×1; `crates/zerogravity-artifacts/tests/` ×2; `crates/zerogravity-cli/src/` ×1; `crates/zerogravity-cli/tests/` ×1; **A1,M8** |
| 76 | `f22f75972e63ea70771c59fbce065e371d5f29d4` | `2026-10-17T19:43:00+08:00` | G | M | feat(pull): list pinned file candidates | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×5; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A4,M8** |
| 77 | `f197c9cf4a44e06a3c1cf3c532855545bed945fa` | `2026-10-17T19:52:21+08:00` | G | M | feat(pull): select explicit GGUF quant families | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×7; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A3,M11** |
| 78 | `1a22597d9fda50ad71c43ae08f8660fb5b014218` | `2026-10-17T17:44:06Z` | G | D | docs: preserve complete October 17 takeover and paused sources | `.agents/docs/` ×1; `.agents/docs/handoff-2026-10-17/` ×6; `.agents/plans/DECISION_REGISTER.md` ×1; `.agents/plans/NEXT_AGENT_PROMPT_2026-10-17.md` ×1; `.agents/plans/TODO.md` ×1; **A8,M2** |
| 79 | `04363c3e83368517475b6dccd4894ff437639927` | `2026-10-17T17:44:24Z` | G | M | feat(pull): qualify bounded full-path filters and exclusion closure | `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×7; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A3,M10** |
| 80 | `8195c18e43bd2aba51d727c03ebe03df6d1fd166` | `2026-10-17T18:12:21Z` | G | M | feat(pull): reconstruct verified required native Xet indexes | `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×3; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-cli/tests/` ×1; **A2,M7** |
| 81 | `f56b57400eb913240d6e2b2eeea5d60d9ab6f35e` | `2026-10-17T19:30:00Z` | G | M | feat(pull): authenticate versioned AWS FULL_OBJECT SHA256 | `.agents/plans/model-acquisition/` ×2; `crates/zerogravity-artifacts/src/` ×3; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/src/` ×1; `crates/zerogravity-cli/tests/` ×1; **A1,M7** |
| 82 | `d5aa5cb0ba10d2603ae34488e54755c5495f63b3` | `2026-10-17T19:48:40Z` | G | M | feat(pull): finish v1 original-model companion closure | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `.agents/plans/model-execution/` ×1; `crates/zerogravity-artifacts/src/` ×2; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/src/` ×1; **M8** |
| 83 | `fee0bb2e2d690956069a0de151770c21168bea63` | `2026-10-17T19:57:15Z` | G | M | feat(pull): close official Qwen and Gemma source metadata | `.agents/plans/TODO.md` ×1; `.agents/plans/model-acquisition/` ×2; `.agents/plans/model-execution/` ×1; `crates/zerogravity-artifacts/src/` ×1; `crates/zerogravity-artifacts/tests/` ×1; **M6** |

### 84–100 — Typed pipelines, source authorization and original-model CPU semantics

Pipeline/filter/context controls compose under existing authority. Immutable safetensors and source-bound Qwen CPU layer/model/dtype/reference plans expand. Historical plan relocation and concise ownership rules occur alongside source work, and remain indexed rather than omitted as “docs only.”

| # | Full commit SHA | Author date (ISO8601) | Sig | Class | Recorded subject | Affected owners/paths; change counts |
|---:|---|---|:---:|:---:|---|---|
| 84 | `4bf624e300254f845e9e047fcb344026c9252d5d` | `2026-10-17T21:31:11Z` | G | M | feat(pipeline): scaffold shared multimodel routing and opt-in chat context | `.agents/plans/TODO.md` ×1; `.agents/plans/cpu-gpu-optimization/` ×1; `.agents/plans/memory-and-streaming/` ×1; `.agents/plans/model-acquisition/` ×1; `.agents/plans/model-execution/` ×1; `.agents/plans/multimodal/` ×1; `.agents/plans/pipeline-composition/` ×2; `.agents/plans/serving-and-speculation/` ×1; `Cargo.lock` ×1; `crates/zerogravity-artifacts/src/` ×4; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/` ×1; `crates/zerogravity-cli/src/` ×3; `crates/zerogravity-cli/tests/` ×3; `crates/zerogravity-core/src/` ×3; `crates/zerogravity/src/` ×1; `examples/` ×5; **A12,M19** |
| 85 | `a9984a47052baaf1aeda558154256d361dcc64af` | `2026-10-18T09:26:30+08:00` | G | S | chore(gitignore): share one canonical ignore block | `.gitignore`; **M1** |
| 86 | `ac44e836b46f6c5aff44afa298bdb342ff65433e` | `2026-10-18T02:23:14Z` | G | M | feat(pipeline): compose source-scoped context filters and typed decisions | `.agents/plans/model-execution/` ×1; `.agents/plans/multimodal/` ×1; `.agents/plans/pipeline-composition/` ×2; `.agents/plans/serving-and-speculation/` ×1; `crates/zerogravity-cli/src/` ×1; `crates/zerogravity-cli/tests/` ×1; `crates/zerogravity-core/src/` ×4; `crates/zerogravity/src/` ×1; `examples/` ×1; **A2,M11** |
| 87 | `f28a62ceb47229279b70283f4e88747d1719e2ed` | `2026-10-18T03:23:06Z` | G | M | feat(context): snapshot authorized committed history for pipelines | `.agents/plans/pipeline-composition/` ×2; `.agents/plans/serving-and-speculation/` ×1; `crates/zerogravity-core/src/` ×3; `crates/zerogravity/src/` ×1; **A1,M6** |
| 88 | `f0746d1d7e22b94b55c93cb734a8c4ad3b2ab67a` | `2026-10-18T04:38:18Z` | G | M | feat(pipeline): finish YAML-first typed multimodal scaffolding | `.agents/plans/TODO.md` ×1; `.agents/plans/model-execution/` ×1; `.agents/plans/pipeline-composition/` ×2; `Cargo.lock` ×1; `crates/zerogravity-cli/` ×1; `crates/zerogravity-cli/src/` ×1; `crates/zerogravity-cli/tests/` ×1; `crates/zerogravity-core/src/` ×1; `examples/` ×17; **A11,M9,D6** |
| 89 | `93a562288f0e00240da92c37c9dd7f4b846d62d3` | `2026-10-18T05:21:30Z` | G | M | feat(source): bind verified immutable safetensors shards | `.agents/plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md` ×1; `.agents/plans/model-execution/` ×2; `crates/zerogravity-artifacts/src/` ×3; `crates/zerogravity-artifacts/tests/` ×1; **A3,M4** |
| 90 | `b796a2e17a4f20086c4db59743aea9558ae1a892` | `2026-10-18T06:27:12Z` | G | M | feat(qwen): bind original source tensors to shared CPU value operators | `.agents/plans/model-execution/` ×2; `Cargo.lock` ×1; `crates/zerogravity-artifacts/src/` ×4; `crates/zerogravity-artifacts/tests/` ×2; `crates/zerogravity-cpu/src/` ×3; `crates/zerogravity/` ×1; `crates/zerogravity/src/` ×2; `crates/zerogravity/tests/` ×1; **A5,M11** |
| 91 | `b11e58cb9ef7dd0d607f301d90c1870897cebd34` | `2026-10-18T07:23:47Z` | G | M | feat(qwen): qualify original hybrid layer CPU proposals | `.agents/plans/model-execution/` ×2; `crates/zerogravity/src/` ×2; `crates/zerogravity/tests/` ×1; **A1,M4** |
| 92 | `1b761119c2de2690723b9531b1d43660a66c1001` | `2026-10-18T15:29:41+08:00` | G | D | docs(agents): unify shared conventions and plan model | `.agents/plans/README.md` ×1; `.agents/plans/TODO-SPEC.md` ×1; `.agents/plans/TODO.md` ×1; `AGENTS.md` ×1; **A1,M2,D1** |
| 93 | `631e59fd614fa6576985e8ff8bf5503d88f99359` | `2026-10-18T15:30:21+08:00` | G | D | docs(plans): migrate historical plans to the unified model | `.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md` ×1; `.agents/plans/CURRENT_STATUS_AND_TRACKING.md` ×1; `.agents/plans/DECISION_REGISTER.md` ×1; `.agents/plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md` ×1; `.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md` ×1; `.agents/plans/accepted-baseline/` ×1; `.agents/plans/archive/history/` ×8; `.agents/plans/handoff-cleanup/` ×1; `.agents/plans/historical-records/` ×1; **A8,D8** |
| 94 | `6329cffbb3e1b93811235867fa697f5f76f3d27a` | `2026-10-18T08:18:40Z` | G | M | feat(qwen): admit source-bound original layer states atomically | `.agents/plans/model-execution/` ×2; `crates/zerogravity-core/src/` ×1; `crates/zerogravity/src/` ×3; `crates/zerogravity/tests/` ×1; **A1,M6** |
| 95 | `e43ba4c63106bab9804ff0f581457ce4f370741e` | `2026-10-18T09:16:57Z` | G | M | feat(qwen): compose original full-model FP32 diagnostic forward | `.agents/plans/model-execution/` ×2; `crates/zerogravity-model-qwen35/src/` ×2; `crates/zerogravity/src/` ×2; `crates/zerogravity/tests/` ×1; **M7** |
| 96 | `28e47f5a4728679e33659298db043da728f225c1` | `2026-10-18T17:17:44+08:00` | G | D | docs(agents): add terse-writing guidance | `AGENTS.md`; **M1** |
| 97 | `7e328411a8de3c0cc2b9cb9690f2e4a0aa9a54c9` | `2026-10-18T17:24:36+08:00` | G | D | docs(agents): strengthen ownership and commit-scope rules | `AGENTS.md`; **M1** |
| 98 | `aaf9ba6ec94a9d1bbd4e09b747d9338637fc9e87` | `2026-10-18T10:09:54Z` | G | M | feat(qwen): qualify source-bound BF16 dtype diagnostics | `.agents/plans/model-execution/` ×2; `crates/zerogravity/src/` ×4; `crates/zerogravity/tests/` ×1; **A1,M6** |
| 99 | `1c0bb41f2e7b2f635b255d5285040ed15b9c25b6` | `2026-10-18T11:26:23Z` | G | M | feat(qwen): bind pinned source bundles to numerical reference traces | `.agents/plans/model-execution/` ×2; `crates/zerogravity-artifacts/src/` ×3; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity/` ×1; `crates/zerogravity/src/` ×4; `crates/zerogravity/tests/` ×1; **A1,M11** |
| 100 | `e90ff8868c46f321ff8bd0bf4c2154a71606aab2` | `2026-10-18T19:38:06+08:00` | G | D | docs(agents): add terse-output guidance | `AGENTS.md`; **M1** |

### 101–131 — Serving, source-exact native batches and independent reference controls

Bounded CPU serving, fair resident GPU waves, batched operators/state, fused feedback, ragged frontiers and source-aware benchmarks develop together. Independent original/HF/AR1/BF16 controls and source-window operators preserve profile distinctions; qualified slices do not imply all contexts or universal model parity.

| # | Full commit SHA | Author date (ISO8601) | Sig | Class | Recorded subject | Affected owners/paths; change counts |
|---:|---|---|:---:|:---:|---|---|
| 101 | `35b90932f9bdf93127059df426e7d7eeaee54cf5` | `2026-10-19T00:19:08Z` | G | M | feat(serve): add bounded OpenAI CPU router and verified SSD extents | `.agents/plans/TODO.md` ×1; `.agents/plans/memory-and-streaming/` ×2; `.agents/plans/serving-and-speculation/` ×2; `Cargo.lock` ×1; `crates/zerogravity-artifacts/src/` ×2; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cli/` ×1; `crates/zerogravity-cli/src/` ×4; `crates/zerogravity-cli/tests/` ×1; `examples/` ×1; **A8,M8** |
| 102 | `9790fa2ef6c4993be0b93014ad1e63d5db5cabbb` | `2026-10-19T02:24:07Z` | G | M | feat(serve): admit resident GPU requests with fair wave scheduling | `.agents/plans/serving-and-speculation/` ×2; `crates/zerogravity-cli/src/` ×5; `examples/` ×1; `scripts/` ×2; `tools/serve-bench.py/` ×1; **A5,M6** |
| 103 | `114bba566bb2d150afe5f8a0fb5701880cae713f` | `2026-10-19T03:59:32Z` | G | M | feat(gpu): qualify native multi-row F32 projection on CUDA | `.agents/plans/serving-and-speculation/` ×2; `crates/zerogravity-gpu/src/` ×4; **A2,M4** |
| 104 | `2869f8784d986dd80973da03afafde89da34c947` | `2026-10-19T05:11:09Z` | G | M | feat(gpu): qualify batched attention GDN KV and elementwise operators | `.agents/plans/serving-and-speculation/` ×2; `crates/zerogravity-cli/src/` ×1; `crates/zerogravity-gpu/src/` ×8; `tools/serve-bench.py/` ×1; `tools/serve-compare.py/` ×1; **A6,M7** |
| 105 | `08be9c64ea60b5dcf4391c54cc183d63666e4a70` | `2026-10-19T07:43:08Z` | G | M | feat(gpu): qualify resident B2-B8 state cohort publication | `.agents/plans/serving-and-speculation/` ×3; `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-gpu/src/` ×4; **A1,M8** |
| 106 | `3ec431a2df62bbcae5091a048f285801f151da7e` | `2026-10-19T08:04:35Z` | G | M | feat(gpu): qualify native batched GGUF quantized projections | `.agents/plans/serving-and-speculation/` ×3; `crates/zerogravity-gpu/src/` ×5; **A2,M6** |
| 107 | `c85ad250334f590a87c99710e95c71a3deb28323` | `2026-10-19T08:41:20Z` | G | D | docs(agents): preserve complete Zerogravity history and P12 handoff | `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md`; `.agents/plans/NEXT_AGENT_PROMPT_2026-10-19.md`; **A2** |
| 108 | `35ebc5b343fd82201eac768edf86ba7a81c41846` | `2026-10-19T18:08:46+08:00` | G | M | feat(gpu): fuse native resident model batches | `.agents/plans/serving-and-speculation/` ×1; `crates/zerogravity-gpu/src/` ×11; **A3,M9** |
| 109 | `904a190cf39aca74e06c979dbaddb0db7f4404f1` | `2026-10-19T18:09:43+08:00` | G | D | docs(agents): record native batch acceptance | `.agents/plans/serving-and-speculation/` ×3; **M3** |
| 110 | `54fd580815dbcfcf0916e4e737beaf21728818b2` | `2026-10-19T19:51:39+08:00` | G | M | feat(gpu): keep batch greedy feedback on device | `.agents/plans/serving-and-speculation/` ×1; `crates/zerogravity-gpu/src/` ×7; **A2,M6** |
| 111 | `570d865696ab398f0c5c37f08c5c951595eadd75` | `2026-10-19T19:52:16+08:00` | G | D | docs(agents): record batch feedback acceptance | `.agents/plans/serving-and-speculation/` ×3; **M3** |
| 112 | `2677d8b664c78b657c26e7bf46c835317b003913` | `2026-10-19T22:33:08+08:00` | G | M | feat(serve): run native CUDA B2 waves | `.agents/plans/serving-and-speculation/` ×1; `crates/zerogravity-cli/src/` ×6; `crates/zerogravity-cli/tests/` ×1; **A3,M5** |
| 113 | `7fc9ad475a30a3d2120f307a7da2ff75a80e337a` | `2026-10-19T22:33:38+08:00` | G | D | docs(serve): bound native B2 acceptance | `.agents/plans/serving-and-speculation/` ×3; **M3** |
| 114 | `f26232e8b03bf5ab3c13e00ed641eb0d602360d2` | `2026-10-20T00:28:17+08:00` | G | M | fix(qwen): honor tied original output heads | `.agents/plans/model-execution/` ×1; `crates/zerogravity/src/` ×1; `crates/zerogravity/tests/` ×1; **M3** |
| 115 | `6bfcdbe72f0133a6c79e2e7de169c3de74b3da8a` | `2026-10-20T00:30:04+08:00` | G | M | feat(bench): account for offered serving work | `.agents/plans/serving-and-speculation/` ×1; `tools/serve-bench.py/` ×1; `tools/serve-compare.py/` ×1; **M3** |
| 116 | `d9fa829ceba092c7f8a3fa9c2746a11a0906eaba` | `2026-10-20T00:40:08+08:00` | G | M | feat(serve): qualify native CUDA rows 4 and 8 | `.agents/plans/serving-and-speculation/` ×3; `crates/zerogravity-cli/src/` ×5; `crates/zerogravity-cli/tests/` ×1; `crates/zerogravity-gpu/src/` ×4; **A1,M12** |
| 117 | `9815504a6975ce766766fc8089729be8121df3ac` | `2026-10-20T01:13:36+08:00` | G | M | feat(qwen): bind original HF diagnostic profiles | `.agents/plans/model-execution/` ×1; `Cargo.lock` ×1; `crates/zerogravity-core/src/` ×1; `crates/zerogravity-cpu/src/` ×4; `crates/zerogravity-gpu/src/` ×2; `crates/zerogravity-model-qwen35/` ×1; `crates/zerogravity-model-qwen35/src/` ×6; `crates/zerogravity/src/` ×5; `crates/zerogravity/tests/` ×1; **A4,M18** |
| 118 | `dbb5f4f4cc42e28c91b12e906f9091ef76510c03` | `2026-10-20T01:53:12+08:00` | G | M | feat(serve): fuse unequal decode frontiers | `.agents/plans/serving-and-speculation/` ×3; `crates/zerogravity-cli/src/` ×8; `crates/zerogravity-cli/tests/` ×1; `crates/zerogravity-gpu/src/` ×14; `tools/serve-bench.py/` ×1; `tools/serve-compare.py/` ×1; **A2,M26** |
| 119 | `cadcbd9d1216b00ca4b69cae9bc04a409e347193` | `2026-10-20T02:07:20+08:00` | G | M | feat(qwen): qualify pinned CPU HF operators | `.agents/plans/TODO.md` ×1; `.agents/plans/model-execution/` ×1; `.agents/plans/numerics-and-context/` ×1; `crates/zerogravity-cpu/src/` ×9; **A8,M4** |
| 120 | `1b599df06d8d7c8c4912838836b272e28e04aca1` | `2026-10-20T02:08:49+08:00` | G | M | feat(bench): isolate owned provider lifecycles | `.agents/plans/serving-and-speculation/` ×3; `tools/serve-compare.py/` ×1; `tools/serve-lifecycle.py/` ×1; **A1,M4** |
| 121 | `d41163432fd6233c7bd88f07b64fb71b1086a902` | `2026-10-20T02:11:18+08:00` | G | M | feat(artifacts): own verified tensor windows | `.agents/plans/memory-and-streaming/` ×2; `crates/zerogravity-artifacts/src/` ×5; `crates/zerogravity-artifacts/tests/` ×2; **A2,M7** |
| 122 | `2a32aff615fc4f7f6718f14a15187825730d6897` | `2026-10-20T02:43:37+08:00` | G | M | feat(qwen): compose typed source operators | `.agents/plans/model-execution/` ×1; `crates/zerogravity-core/src/` ×3; `crates/zerogravity-cpu/src/` ×3; `crates/zerogravity-gpu/src/` ×3; `crates/zerogravity-model-qwen35/src/` ×8; **A8,M10** |
| 123 | `c5eb1f8e717c79146937e585a145a0052eff747b` | `2026-10-20T03:22:04+08:00` | G | M | feat(cpu): qualify source sigmoid and RMS | `.agents/plans/model-execution/` ×1; `crates/zerogravity-cpu/src/` ×7; **A5,M3** |
| 124 | `d0a60844217fc106fe8c70bb3a99073eef0387c8` | `2026-10-20T04:19:25+08:00` | G | M | perf(gpu): fuse source activation preparation | `.agents/plans/serving-and-speculation/` ×2; `crates/zerogravity-cli/src/` ×1; `crates/zerogravity-gpu/src/` ×6; **A2,M7** |
| 125 | `5350a4d40b535106240d48fec8aedd459d5cd17a` | `2026-10-20T04:56:45+08:00` | G | M | feat(cpu): qualify typed source windows | `.agents/plans/model-execution/` ×1; `crates/zerogravity-artifacts/src/` ×2; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-core/src/` ×3; `crates/zerogravity-cpu/` ×1; `crates/zerogravity-cpu/src/` ×12; `crates/zerogravity-gpu/src/` ×2; `crates/zerogravity-model-qwen35/src/` ×4; **A9,M17** |
| 126 | `f78892957f09a84838246aede4fef2be040a0d18` | `2026-10-20T06:13:20+08:00` | G | M | feat(gpu): fuse native teacher-prefix serving | `.agents/plans/serving-and-speculation/` ×2; `crates/zerogravity-cli/src/` ×6; `crates/zerogravity-cli/tests/` ×2; `crates/zerogravity-gpu/src/` ×3; **A3,M10** |
| 127 | `bfe62df68d763b9d5c6b87c891be6a62b3593de9` | `2026-10-20T06:35:48+08:00` | G | M | feat(cpu): execute leased source-window plans | `.agents/plans/model-execution/` ×1; `.agents/plans/numerics-and-context/` ×1; `.agents/plans/serving-and-speculation/` ×1; `crates/zerogravity-artifacts/src/` ×3; `crates/zerogravity-artifacts/tests/` ×1; `crates/zerogravity-cpu/src/` ×3; `crates/zerogravity-model-qwen35/` ×1; `crates/zerogravity-model-qwen35/src/` ×7; **A4,M14** |
| 128 | `fe2a34213009cc160f4d82068bc7dd1f49d05c6c` | `2026-10-20T08:12:19+08:00` | G | M | feat(cpu): qualify original attention fragments | `.agents/plans/model-execution/` ×1; `crates/zerogravity-model-qwen35/src/` ×8; **A2,M7** |
| 129 | `9a199017f18f1bcb1a8053685be3c3c29236ec90` | `2026-10-20T08:21:04+08:00` | G | M | feat(gpu): qualify batched graph replay | `.agents/plans/model-execution/` ×1; `.agents/plans/numerics-and-context/` ×1; `.agents/plans/serving-and-speculation/` ×2; `crates/zerogravity-gpu/src/` ×15; `crates/zerogravity-gpu/vendor/cubecl-cuda/` ×3; **A7,M15** |
| 130 | `fff1842594f8011b761d08fcc1690311b72543e3` | `2026-10-20T09:20:27+08:00` | G | M | feat: qualify trained BF16 operator slices | `.agents/plans/model-execution/` ×1; `.agents/plans/numerics-and-context/` ×1; `.agents/plans/serving-and-speculation/` ×1; `crates/zerogravity-artifacts/src/` ×2; `crates/zerogravity-artifacts/tests/` ×2; `crates/zerogravity-core/src/` ×3; `crates/zerogravity-gpu/` ×1; `crates/zerogravity-gpu/src/` ×9; `crates/zerogravity-model-qwen35/src/` ×9; **A11,M18** |
| 131 | `fb68d8fa165de16e21d1bd37afef3790b80da971` | `2026-10-20T13:27:39+08:00` | G | M | perf: combine GPU state scatter certification | `.agents/plans/model-execution/` ×1; `.agents/plans/numerics-and-context/` ×1; `.agents/plans/serving-and-speculation/` ×1; `crates/zerogravity-gpu/src/` ×8; **A5,M6** |

### 132–187 — 24 October assembled core, model, backend and serving source

Run policy, backing/workspace/topology and CPU source access lead to large-model reference tools, Gemma/Qwen source roles, CUDA/HIP kernel/qualification assets, facade loaders and finite canonical/short-HF serving. Each source/feature/physical claim remains bound to its actual tuple; scaffold imports are not broad model support.

| # | Full commit SHA | Author date (ISO8601) | Sig | Class | Recorded subject | Affected owners/paths; change counts |
|---:|---|---|:---:|:---:|---|---|
| 132 | `bb089823653f89d64bc97935cd7b11b7c3e66fd3` | `2026-10-24T16:55:25+08:00` | G | S | feat(core): add run policy request validation | `crates/zerogravity-core/src/` ×4; **A2,M2** |
| 133 | `bd277cf7667381333d54ff3c7f249dae92b8efa6` | `2026-10-24T16:55:25+08:00` | G | S | feat(core): add reserved domains and workspace | `crates/zerogravity-core/src/` ×3; **A2,M1** |
| 134 | `35c9962d8633599a8b2393b29e1fa26f02677272` | `2026-10-24T16:55:26+08:00` | G | S | feat(core): admit device backing views | `crates/zerogravity-core/src/` ×3; **A2,M1** |
| 135 | `eb49fa0e945bec802db28472deb50ab94b715284` | `2026-10-24T16:55:26+08:00` | G | S | feat(core): record plan topology and source programs | `crates/zerogravity-core/src/` ×43; **A39,M4** |
| 136 | `ec30569bc6b921bfc09af888978d22d23e957944` | `2026-10-24T16:55:26+08:00` | G | M | refactor(cpu): share reference math lanes | `crates/zerogravity-cpu/src/` ×8; **A4,M4** |
| 137 | `9dd1ebfb18623e27cb5ef07940f5e6839f1ecaed` | `2026-10-24T16:55:26+08:00` | G | S | feat(cpu): add original torch CPU operators | `Cargo.lock` ×1; `crates/zerogravity-cpu/` ×1; `crates/zerogravity-cpu/src/` ×8; **A5,M5** |
| 138 | `a3716fd0de505b5bcf1e7e43ff3da7154c1ac8ff` | `2026-10-24T16:55:27+08:00` | G | S | feat(artifacts): select cached sources by locator | `crates/zerogravity-artifacts/src/` ×8; **A2,M6** |
| 139 | `77c48b315feec501082c2609077033331cc9fcb7` | `2026-10-24T16:55:27+08:00` | G | S | feat(artifacts): authenticate reusable source access | `crates/zerogravity-artifacts/src/` ×6; `crates/zerogravity-artifacts/tests/` ×2; **A4,M4** |
| 140 | `7ba9278b0a1ab97514f9511fc7a72ae90eeccf1a` | `2026-10-24T16:55:27+08:00` | G | S | feat(cpu): evaluate reusable text source windows | `crates/zerogravity-cpu/src/` ×8; **A7,M1** |
| 141 | `dbd836351d40289eb8d99f7e1f648c139530de34` | `2026-10-24T16:55:27+08:00` | G | S | feat(cpu): add haswell and zen bulk kernels | `crates/zerogravity-cpu/src/` ×5; `crates/zerogravity-cpu/tests/` ×2; **A6,M1** |
| 142 | `b026d0f9d5024d06f5b83768191f70258c1aa306` | `2026-10-24T16:55:28+08:00` | G | M | feat(tools): add qwen27b output-tail reference | `tools/qwen27b-output-tail-reference/` ×10; **A10** |
| 143 | `85d86b3c5863d9d78b6bed0b7db6e36ed28686cf` | `2026-10-24T16:55:28+08:00` | G | S | feat(tools): add qwen27b attention-tail reference | `tools/qwen27b-attention-tail-reference/` ×10; **A10** |
| 144 | `0c7a4a9770f64d3de1e3c920a438c88e088668c3` | `2026-10-24T16:55:28+08:00` | G | M | feat(tools): add qwen27b full-window reference | `tools/qwen27b-full-window-reference/` ×11; **A11** |
| 145 | `af5aa8498550fb98e69734c18769cd2c84299278` | `2026-10-24T16:55:29+08:00` | G | S | feat(tools): add qwen27b layer9-prefix reference | `tools/qwen27b-layer9-prefix-reference/` ×11; **A11** |
| 146 | `6dd5970776b7b5839ebc6b60c3694049d31f406b` | `2026-10-24T16:55:29+08:00` | G | S | feat(tools): add qwen27b initial-core reference | `tools/qwen27b-initial-core-reference/` ×14; **A14** |
| 147 | `2c70ccb66c780874998e77bbdb2c9e33387bdf2f` | `2026-10-24T16:55:29+08:00` | G | S | feat(tools): add gemma4 tokenizer reference | `tools/gemma4-tokenizer-reference/` ×8; **A8** |
| 148 | `c870725898c9a0595609f53f7310182c5bec4fac` | `2026-10-24T16:55:30+08:00` | G | M | feat(tools): add gemma4-e2b text reference | `tools/gemma4-e2b-reference/` ×13; **A13** |
| 149 | `8b0cc32c951027123ffa2a9dde1e5f7dd6c2a57a` | `2026-10-24T16:55:30+08:00` | G | M | feat(tools): add gemma4-e2b CPU matrix | `tools/gemma4-e2b-ordinary-reference/` ×9; **A9** |
| 150 | `283765cff8bdbb69d4d6674d7fd28360238ea693` | `2026-10-24T16:55:30+08:00` | G | M | feat(tools): add portable BF16 slice reference | `tools/portable-bf16-slice-reference/` ×7; **A7** |
| 151 | `b5015e7a77e37172236c602b65a25c74a3caead4` | `2026-10-24T16:55:31+08:00` | G | S | feat(gemma4): admit dense 31B text metadata | `crates/zerogravity-model-gemma4/src/` ×3; `crates/zerogravity-model-gemma4/tests/` ×2; **A3,M2** |
| 152 | `bebfeaf2d81722ebca4c11613e8a40d5834e18df` | `2026-10-24T16:55:31+08:00` | G | S | feat(gemma4): add E2B text reference and tokenizer | `Cargo.lock` ×1; `crates/zerogravity-model-gemma4/` ×1; `crates/zerogravity-model-gemma4/src/` ×14; **A13,M3** |
| 153 | `3002089997fcee9e43abc3dc60404cbeba1f52ab` | `2026-10-24T16:55:32+08:00` | G | S | feat(gemma4): add E2B learned-source diagnostics | `crates/zerogravity-model-gemma4/` ×1; `crates/zerogravity-model-gemma4/examples/` ×10; **A10,M1** |
| 154 | `225fc5e1a520637b45c3c76f36e4b58963c1de08` | `2026-10-24T16:55:32+08:00` | G | S | feat(qwen35): record window, bulk and CPU tail sources | `crates/zerogravity-model-qwen35/src/` ×43; `crates/zerogravity-model-qwen35/tests/` ×7; **A42,M8** |
| 155 | `9af407bf3ff5763c20f4e9fddbf8061fa16f8179` | `2026-10-24T16:55:32+08:00` | G | S | feat(qwen35): add qwen27b window replays | `Cargo.lock` ×1; `crates/zerogravity-model-qwen35/` ×1; `crates/zerogravity-model-qwen35/examples/` ×22; **A22,M2** |
| 156 | `c4cbff17e6c431ee1d60e67cd725b3bf0f2fc4a7` | `2026-10-24T16:55:32+08:00` | G | S | feat(cuda): observe device metadata | `crates/zerogravity-gpu/vendor/cubecl-cuda/src/lib.rs`; `crates/zerogravity-gpu/vendor/cubecl-cuda/src/observation.rs`; **A1,M1** |
| 157 | `d9d4f8b9bb500dcf632b37efae50c2926152adf3` | `2026-10-24T16:55:33+08:00` | G | S | feat(cuda): own compiler, source and cubin actor paths | `Cargo.lock` ×1; `crates/zerogravity-gpu/vendor/cubecl-cuda/` ×73; **A68,M6** |
| 158 | `4bc7c429d585d720600a9a33f0da25da4baceff4` | `2026-10-24T16:55:33+08:00` | G | S | feat(hip): expose availability and prepared native sets | `crates/zerogravity-gpu/vendor/cubecl-hip/` ×11; **A6,M5** |
| 159 | `248bcec54c6e3d94992e379b509d84fd5864cbbc` | `2026-10-24T16:55:33+08:00` | G | S | build(gpu): add qualification feature surface | `Cargo.lock`; `crates/zerogravity-gpu/Cargo.toml`; **M2** |
| 160 | `68c2f5c050136978f5e7c01958fc08e6f2504d60` | `2026-10-24T16:55:33+08:00` | G | S | feat(gpu): add shared kernel source primitives | `crates/zerogravity-gpu/src/` ×6; **A5,M1** |
| 161 | `abca70fdebda8e395b513133e79f938a9b2d84cd` | `2026-10-24T16:55:33+08:00` | G | S | feat(gpu): add original source arithmetic kernels | `crates/zerogravity-gpu/src/` ×14; **A13,M1** |
| 162 | `63583681ae8c3f850f1b6e00dad4de8a7f421c56` | `2026-10-24T16:55:33+08:00` | G | S | feat(gpu): add native policy and physical identity | `crates/zerogravity-gpu/src/` ×10; `crates/zerogravity-gpu/tests/` ×1; **A9,M2** |
| 163 | `fe95ecdde7800a6a9079f70ff7aa90f6780cad15` | `2026-10-24T16:55:33+08:00` | G | M | feat(gpu): add vLLM input RMS qualification kernels | `crates/zerogravity-gpu/src/` ×37; **A37** |
| 164 | `405e2462ff310c6ceae1e43ee54c54e2b13907c0` | `2026-10-24T16:55:34+08:00` | G | M | feat(gpu): add vLLM GDN instruction kernels | `crates/zerogravity-gpu/src/` ×46; **A46** |
| 165 | `71c59964149f280787359063f1c1e17eb52fef98` | `2026-10-24T16:55:34+08:00` | G | S | feat(gpu): add Qwen2B GDN instruction assets | `crates/zerogravity-gpu/src/` ×8; **A8** |
| 166 | `00d4bbabde0491f6f0be513f0223558bc813edd6` | `2026-10-24T16:55:34+08:00` | G | S | feat(gpu): add .8B short-HF GDN call catalog | `crates/zerogravity-gpu/src/` ×112; **A112** |
| 167 | `d031166b726a1ab07ef88ea8c1401e01df8285ee` | `2026-10-24T16:55:35+08:00` | G | S | feat(gpu): add vLLM CUDA qualification scaffolding | `crates/zerogravity-gpu/src/` ×38; **A38** |
| 168 | `8dbe361c1e53db00738f1e25ab11dca44b5f5840` | `2026-10-24T16:55:35+08:00` | G | S | feat(gpu): add layer, attention and GDN cubin checks | `crates/zerogravity-gpu/src/` ×56; **A56** |
| 169 | `13ae2ffb7d0510073fd36dfc137693a9de2f3bb8` | `2026-10-24T16:55:35+08:00` | G | S | feat(gpu): add canonical47 model qualifications | `crates/zerogravity-gpu/src/` ×29; **A29** |
| 170 | `bbafb1de8c769c0ad1d158ffc138b7db0144f319` | `2026-10-24T16:55:35+08:00` | G | S | feat(gpu): add Qwen2B qualification leaves | `crates/zerogravity-gpu/src/` ×17; **A17** |
| 171 | `f619f7eed8377b8c8bf9838a642b86d049881180` | `2026-10-24T16:55:35+08:00` | G | S | feat(gpu): add short-HF and full-model checks | `crates/zerogravity-gpu/src/` ×34; **A34** |
| 172 | `3e89f1919993081535b5fe2820da05ea9d840b82` | `2026-10-24T16:55:36+08:00` | G | S | feat(gpu): wire resident leases, program and native sets | `crates/zerogravity-gpu/src/` ×246; **A230,M16** |
| 173 | `136a4a8692d9aefe19e1d7ef25eab4e2fc48d561` | `2026-10-24T16:55:37+08:00` | G | S | feat(facade): expose CPU routing preparation | `crates/zerogravity/src/lib.rs`; `crates/zerogravity/src/prepare.rs`; **M2** |
| 174 | `08b30b571e4b3557be78318b66365030c1bf5bb2` | `2026-10-24T16:55:37+08:00` | G | S | feat(facade): add canonical47 original-HF loader | `Cargo.lock` ×1; `crates/zerogravity/` ×1; `crates/zerogravity/src/` ×8; **A7,M3** |
| 175 | `bc66207ce9c5a1ca96e609e975151541579c4390` | `2026-10-24T16:55:37+08:00` | G | S | feat(facade): add short-HF serving source loader | `crates/zerogravity/` ×1; `crates/zerogravity/src/` ×6; **A5,M2** |
| 176 | `e44a53f75320e21df9642f5c3fe711efe8fccc04` | `2026-10-24T16:55:37+08:00` | G | S | test(facade): add short-HF physical suite | `crates/zerogravity/tests/` ×5; **A5** |
| 177 | `08fc5935e124f6af33c5010fc4309648539e1641` | `2026-10-24T16:55:37+08:00` | G | S | test(facade): add short-HF cohort suite | `crates/zerogravity/tests/` ×7; **A7** |
| 178 | `a4a5b6322f343004d1ecdf13b264654da466424f` | `2026-10-24T16:55:37+08:00` | G | S | feat(facade): add Gemma4 E2B CPU candidate | `crates/zerogravity/` ×1; `crates/zerogravity/examples/` ×5; `crates/zerogravity/src/` ×6; **A10,M2** |
| 179 | `2058dcb1ea3f8053b6906561d0698df392c91731` | `2026-10-24T16:55:37+08:00` | G | S | feat(facade): add qwen27b HIP teacher source | `crates/zerogravity/` ×1; `crates/zerogravity/examples/` ×6; `crates/zerogravity/src/` ×4; **A9,M2** |
| 180 | `71f496b190dc2c7ced495f9d4c1f5c5da9495f33` | `2026-10-24T16:55:37+08:00` | G | S | feat(facade): add Gemma4 HIP drivers | `crates/zerogravity/` ×1; `crates/zerogravity/examples/` ×5; **A5,M1** |
| 181 | `752642361cb4ad2f769b868f071a3307e3b01651` | `2026-10-24T16:55:37+08:00` | G | S | feat(facade): add shared HIP lifecycle driver | `crates/zerogravity/Cargo.toml`; `crates/zerogravity/examples/hip_shared_models.rs`; **A1,M1** |
| 182 | `2bd8167db2c1d8817d38e4851dd45f9c2c9aa6e4` | `2026-10-24T16:55:37+08:00` | G | S | build(cli): gate finite HF serving and run features | `Cargo.lock`; `crates/zerogravity-cli/Cargo.toml`; **M2** |
| 183 | `b3c5a532d72a266fc3d66e13c5c1a0f7ad1bd084` | `2026-10-24T16:55:37+08:00` | G | S | feat(cli): route run requests and add gemma4 candidate | `crates/zerogravity-cli/src/` ×12; `crates/zerogravity-cli/tests/` ×1; **A11,M2** |
| 184 | `45570bdb3d7f21f531b8cbc840633b8d2eb83e4c` | `2026-10-24T16:55:38+08:00` | G | S | perf(serve): plan atomic source prefill waves | `crates/zerogravity-cli/src/serve/gpu_batch.rs`; **M1** |
| 185 | `e6b07efdbb2d6ec56d979ff09b025c8071038f06` | `2026-10-24T16:55:38+08:00` | G | S | feat(serve): route and serve canonical47 sources | `crates/zerogravity-cli/src/` ×9; **A2,M7** |
| 186 | `9999adb1a66d646f76cd1152c32da51b22c9eded` | `2026-10-24T16:55:38+08:00` | G | S | feat(serve): serve short-HF BF16 B4 cohorts | `crates/zerogravity-cli/src/serve/gpu_engine/short_hf.rs`; `crates/zerogravity-cli/src/serve/gpu_engine/short_hf/cohort.rs`; **A2** |
| 187 | `ef9cffd413dfe3a02d8ed54e4a2ff3a5e317e10b` | `2026-10-24T16:55:38+08:00` | G | S | chore(scripts): extend gate for HF features | `scripts/check.sh`; **M1** |

### 188–209 — Design/research decisions and lossless historical-context checkpoint

These documentation commits retain execution cadence, reuse/provenance, composition, pools/state, numerical/model/acquisition checkpoints, distributed/platform/CLI designs and the20 October archive. Paused text and draft patches are preserved as context, not silently promoted into implementations.

| # | Full commit SHA | Author date (ISO8601) | Sig | Class | Recorded subject | Affected owners/paths; change counts |
|---:|---|---|:---:|:---:|---|---|
| 188 | `2225af34b82e0e60da0f50dd4c5ec76eb827ca48` | `2026-10-24T17:01:02+08:00` | G | D | docs(agents): codify parallel execution cadence | `.agents/plans/README.md`; `AGENTS.md`; **M2** |
| 189 | `abb6b3b3b233e6887e182acb7c95ed26718601e4` | `2026-10-24T17:01:02+08:00` | G | D | docs(research): compare optimization mechanisms | `.agents/research/` ×17; **A17** |
| 190 | `4b44d313bedecc6b82ecd44e317d786db2e08cd5` | `2026-10-24T17:01:02+08:00` | G | D | docs(policy): adopt reuse-first provenance rule | `.agents/plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md`; `.agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`; **A1,M1** |
| 191 | `46996b66e39e832f05d0174419e9c8f69367128f` | `2026-10-24T17:01:02+08:00` | G | D | docs(plans): design composition and batch state | `.agents/plans/numerics-and-context/` ×1; `.agents/plans/pipeline-composition/` ×3; `.agents/plans/serving-and-speculation/` ×1; **A4,M1** |
| 192 | `6ef74b048e830f1075d4d5ca4296cc801b8b6871` | `2026-10-24T17:01:02+08:00` | G | D | docs(cpu-gpu): record shared memory and parity | `.agents/plans/cpu-gpu-optimization/` ×3; **A3** |
| 193 | `fb7ea9a9a77b95543544ae131fb44da26522e7a7` | `2026-10-24T17:01:02+08:00` | G | D | docs(numerics): record F32 closure checkpoints | `.agents/plans/numerics-and-context/TODO.md`; **M1** |
| 194 | `424c672768b388f1738187baffafba7973c4cd08` | `2026-10-24T17:01:02+08:00` | G | D | docs(models): record source20 and Qwen audit | `.agents/plans/model-execution/TODO.md`; **M1** |
| 195 | `da28f851448b8cebdd8b5c132f1a7371a6b39380` | `2026-10-24T17:01:03+08:00` | G | D | docs(acquisition): record Gemma E2B acquisition | `.agents/plans/model-acquisition/TODO.md`; **M1** |
| 196 | `0c966799d7830c6b0a3d6b354fcf5318d36add9d` | `2026-10-24T17:01:03+08:00` | G | D | docs(cpu-gpu): record current checkpoints | `.agents/plans/cpu-gpu-optimization/TODO.md`; **M1** |
| 197 | `c57648ac54ff02a5370c3a26e4dc0c1634cee7da` | `2026-10-24T17:01:03+08:00` | G | D | docs(memory): adopt shared pools and paged state | `.agents/plans/memory-and-streaming/TODO.md`; **M1** |
| 198 | `b552e2198d66400c94fbb56d4dc5deeb7604115b` | `2026-10-24T17:01:03+08:00` | G | D | docs(composition): record shared HIP checkpoints | `.agents/plans/pipeline-composition/TODO.md`; **M1** |
| 199 | `6075a4386589ca5d6e0d857d14f3ef4c6a4879b7` | `2026-10-24T17:01:03+08:00` | G | D | docs(serving): certify transport and add batching | `.agents/plans/serving-and-speculation/TODO.md`; **M1** |
| 200 | `3b9da530d922814e8e5dea75574c2d4fdbc11462` | `2026-10-24T17:01:03+08:00` | G | D | docs(distributed): defer mechanism reconciliation | `.agents/plans/distributed-execution/TODO.md`; **M1** |
| 201 | `38e3d77eba85875f0a3b2131a74c0de42933690b` | `2026-10-24T17:01:03+08:00` | G | D | docs(platform): add Vulkan parity and reuse gates | `.agents/plans/platform-and-release/TODO.md`; **M1** |
| 202 | `a765f221279c9f803f0dacbee314c6ad4126905f` | `2026-10-24T17:01:03+08:00` | G | D | docs(cli): add device routing plan | `.agents/plans/cli-device-routing/README.md`; `.agents/plans/cli-device-routing/TODO.md`; **A2** |
| 203 | `309b17d81bfbe7878fc8734adee424cb7785f533` | `2026-10-24T17:01:03+08:00` | G | D | docs(handoff): archive history and run notes | `.agents/docs/handoff-2026-10-20/` ×11; **A11** |
| 204 | `ac17d49c07b2cb0e4653db19dc8621c6f0591279` | `2026-10-24T17:01:03+08:00` | G | D | docs(handoff): preserve paused continuation text | `.agents/docs/handoff-2026-10-20/` ×29; **A29** |
| 205 | `7115754b639aa8bfbaab3a9231b97458fe945df5` | `2026-10-24T17:01:04+08:00` | G | D | docs(handoff): preserve draft patches | `.agents/docs/handoff-2026-10-20/` ×29; **A29** |
| 206 | `11a47c74ecc55d92749dfd581d90254f02defdd5` | `2026-10-24T17:01:04+08:00` | G | D | docs(handoff): preserve takeover and audit reports | `.agents/docs/handoff-2026-10-20/` ×7; `.agents/plans/NEXT_AGENT_PROMPT_2026-10-20.md` ×1; **A8** |
| 207 | `b5d814f8fcab784514b4c6d9bfb1a11a7fa635d2` | `2026-10-24T17:01:04+08:00` | G | D | docs(handoff): preserve canonical sourcebooks | `.agents/docs/handoff-2026-10-20/` ×7; **A7** |
| 208 | `e11fb986091f8fef71abf188e15177242084e596` | `2026-10-24T17:01:04+08:00` | G | D | docs(handoff): index the 20 Oct handoff packet | `.agents/docs/` ×1; `.agents/docs/handoff-2026-10-20/` ×4; **A5** |
| 209 | `4bf36731cfc4f66e02aff2f63fc32b653f6e363c` | `2026-10-24T17:01:04+08:00` | G | D | docs(plans): activate CLI, composition and compute | `.agents/plans/TODO.md`; **M1** |

### 210–245 — 25 October integrated lanes, physical/ABI corrections and device UX

Portable admission, pending ingress, backend catalogues, H/G/V tools and B4 transport are assembled. Subsequent fixes strengthen fixture retirement, SDK/feature separation, independent fresh-flight oracles, cancellation/context bounds and typed ABI closure. CLI aliases and Linux character-device→PCI resolution preserve existing physical admission and hardware holds.

| # | Full commit SHA | Author date (ISO8601) | Sig | Class | Recorded subject | Affected owners/paths; change counts |
|---:|---|---|:---:|:---:|---|---|
| 210 | `8f01635305af920eadba96a7d4f1d6b9e8f588cf` | `2026-10-25T10:00:19+08:00` | G | S | feat(core): portable admission and thread-control roots | `crates/zerogravity-artifacts/src/` ×3; `crates/zerogravity-core/src/` ×1; `crates/zerogravity-cpu/src/` ×2; `crates/zerogravity/src/` ×2; **A2,M6** |
| 211 | `5d40cf507a2e8f5f159b664ddc83876d73745607` | `2026-10-25T10:00:23+08:00` | G | S | feat(cli): shared pending queue and ingress credit | `crates/zerogravity-cli/` ×1; `crates/zerogravity-cli/src/` ×9; **A2,M8** |
| 212 | `52f76a0e0dc3ce8e09476af1460193a03a551227` | `2026-10-25T10:00:27+08:00` | G | S | fix(cli): short-HF tokenizer credit and HOST8 | `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-model-qwen35/` ×1; `crates/zerogravity-model-qwen35/src/` ×2; `crates/zerogravity/src/` ×4; **A1,M8** |
| 213 | `efd7a34220ce168b5a5c851226b400e07e95a095` | `2026-10-25T10:00:32+08:00` | G | S | fix(gpu): CUDA fragment inventory and host bound | `crates/zerogravity-cli/src/` ×2; `crates/zerogravity-gpu/src/` ×4; **M6** |
| 214 | `19ff0598dd79d12ade182d90a855ecf9c60b1f59` | `2026-10-25T10:00:38+08:00` | G | M | feat(gpu): Vulkan native catalogue via cubecl-wgpu | `Cargo.toml` ×1; `crates/zerogravity-gpu/src/` ×19; `crates/zerogravity-gpu/` ×1; `crates/zerogravity-gpu/vendor/cubecl-hip/` ×1; `crates/zerogravity-gpu/vendor/cubecl-wgpu/` ×60; `crates/zerogravity/` ×1; `crates/zerogravity/examples/` ×6; `crates/zerogravity/src/` ×1; `tools/vulkan-gemma-qualification/` ×15; **A86,M19** |
| 215 | `29a98c1c7c086d6367308ad9468c427046008cc5` | `2026-10-25T10:00:44+08:00` | G | M | feat(gpu): HIP source-page events and witness | `crates/zerogravity-gpu/` ×1; `crates/zerogravity-gpu/src/` ×8; `crates/zerogravity-gpu/vendor/cubecl-hip/` ×5; `crates/zerogravity/examples/` ×1; `tools/hip-shared-models-qualification/` ×20; **A21,M14** |
| 216 | `e8db111158991339f76e5b06ed3acb8b370f302a` | `2026-10-25T10:00:48+08:00` | G | M | feat(examples): Qwen27B HIP matrix, natural tools | `crates/zerogravity/examples/` ×2; `tools/qwen27b-hip-qualification/` ×25; `tools/qwen27b-natural-full-window-reference/` ×11; **A36,M2** |
| 217 | `332038ea78ec8b1e69a2a29d3ca09370d712ef6c` | `2026-10-25T10:00:52+08:00` | G | M | feat(examples): Gemma4 HIP final-owner and matrix | `crates/zerogravity/examples/` ×2; `tools/gemma4-hip-qualification/` ×22; **A22,M2** |
| 218 | `68e13a8114e130ab89ac4c55ee9ad14172faa42f` | `2026-10-25T10:00:52+08:00` | G | S | chore(cargo): lock assembled five-lane graph | `Cargo.lock`; **M1** |
| 219 | `74d0c71998709de48115db16507ea0109b23f65b` | `2026-10-25T10:02:44+08:00` | G | D | docs(plans): record assembly, C16 and handoff | `.agents/docs/` ×1; `.agents/plans/NEXT_AGENT_PROMPT_2026-10-25.md` ×1; `.agents/plans/cpu-gpu-optimization/` ×2; `.agents/plans/model-execution/` ×1; `.agents/plans/platform-and-release/` ×1; **A2,M4** |
| 220 | `1269b8eae908a40e79d747e923e29b708d84ba12` | `2026-10-25T10:04:00+08:00` | G | D | docs(plans): retire five landed lane trees | `.agents/plans/cpu-gpu-optimization/TODO.md`; `.agents/plans/platform-and-release/TODO.md`; **M2** |
| 221 | `06b89da455655ef039be2a87fd1e29917ce4d2c3` | `2026-10-25T10:28:19+08:00` | G | S | fix(gpu): HIP probe witness page id typing | `crates/zerogravity-gpu/src/resident_leases/source_page.rs`; **M1** |
| 222 | `c217c35d480eb33b04eb981496d552a816b169e9` | `2026-10-25T10:28:24+08:00` | G | M | fix(qualification): V state base parents, attempt-2 bases | `tools/gemma4-hip-qualification/` ×9; `tools/hip-shared-models-qualification/` ×8; `tools/vulkan-gemma-qualification/` ×5; **M22** |
| 223 | `3941a3c9fe0ee5dec0b9f836a86e992161f5f8ba` | `2026-10-25T11:02:16+08:00` | G | S | fix(qualification): Gemma resident report-phase envelope 4GiB | `tools/gemma4-hip-qualification/control-checks.py`; `tools/gemma4-hip-qualification/payload/physical-runner/fixture_semantic.py`; **M2** |
| 224 | `e6ab074e3118a7200ea0a620a92b3a0990bbcac5` | `2026-10-25T11:33:24+08:00` | G | M | fix(qualification): H/G source census skips build dirs | `tools/gemma4-hip-qualification/` ×1; `tools/hip-shared-models-qualification/` ×8; **M9** |
| 225 | `1565683442e80f529d810d98c63ca2583c5b01d7` | `2026-10-25T11:51:52+08:00` | G | D | docs(plans): record queue GPU acceptance results | `.agents/plans/cpu-gpu-optimization/TODO.md`; **M1** |
| 226 | `2304f8b4d2079c5312a8020f14af49c2c3832ab3` | `2026-10-25T12:03:16+08:00` | G | S | fix(cli): precedence, ambiguity, pre-client hold | `crates/zerogravity-artifacts/src/` ×1; `crates/zerogravity-cli/src/` ×5; `crates/zerogravity-cli/tests/` ×2; `crates/zerogravity-core/src/` ×1; **A1,M8** |
| 227 | `d3d21a35c5b75272d82947ccd9f351bc7888e3c3` | `2026-10-25T12:07:48+08:00` | G | S | feat(gpu): B4 grouped transport and certification | `crates/zerogravity-gpu/src/` ×25; `crates/zerogravity/tests/` ×1; **A4,M22** |
| 228 | `6f1e0947cdf7673cbe4626b983fe894fbad1744d` | `2026-10-25T12:40:33+08:00` | G | D | docs(plans): record main checkpoint and B4 merge | `.agents/plans/cpu-gpu-optimization/TODO.md`; **M1** |
| 229 | `9fbe15e9e45c0d9c71ce00e73dfbdde343a77513` | `2026-10-25T13:08:47+08:00` | G | D | docs(handoff): record checkpoint, HGV pass 3, successor queue | `.agents/docs/` ×1; `.agents/plans/model-execution/` ×1; `.agents/plans/platform-and-release/` ×1; **A1,M2** |
| 230 | `5749b732dc755c24d2179ba7dd7e8abc9380805e` | `2026-10-25T13:28:53+08:00` | G | D | docs(handoff): record Qwen27B capture lane yield detail | `.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md`; **M1** |
| 231 | `488e0ea2c21e2301c2fdeaf82dab6de32789ea08` | `2026-10-25T08:14:56Z` | G | M | fix: repair native qualification contracts and GPU feature isolation | `.agents/docs/` ×2; `crates/zerogravity-gpu/src/` ×2; `crates/zerogravity/examples/` ×3; `scripts/` ×2; `tools/gemma4-hip-qualification/` ×2; `tools/vulkan-gemma-qualification/` ×1; **A4,M8** |
| 232 | `93b611acaa657b6999a19239e4596d4de47b4de3` | `2026-10-25T08:26:58Z` | G | S | fix: prevent Gemma HIP source-control payload shadowing | `tools/gemma4-hip-qualification/control-checks.py`; `tools/gemma4-hip-qualification/test_row_traffic.py`; **M2** |
| 233 | `6edc6e2c99d9a2353d16a85ac689e76af3628be5` | `2026-10-25T08:33:50Z` | G | D | docs: pin refreshed H G packages and completed GPU-free controls | `.agents/docs/POST_4BF3673_AUDIT_AND_RESUME_2026-10-25.md`; **M1** |
| 234 | `9e67b74000bc085a844d8844d954d82f8e196493` | `2026-10-25T08:59:57Z` | G | D | docs(audit): independently validate post-4bf GPU boundaries and resume holds | `.agents/docs/TAKEOVER_6EDC6E2_INDEPENDENT_AUDIT_2026-10-25.md`; **A1** |
| 235 | `03f2e662f219797da524d0e4d6e2aabef979471b` | `2026-10-25T11:08:08Z` | G | M | fix(gpu): certify grouped B4 CUDA ABI and exact fixture retirement | `.agents/docs/` ×1; `crates/zerogravity-gpu/src/` ×3; **A1,M3** |
| 236 | `3d9866dfd6cf66009c9aa6efe0967e4a134e4e41` | `2026-10-25T11:09:42Z` | G | D | docs(audit): distinguish 23 model-dependent B4 scopes | `.agents/docs/B4_C16_EXECUTION_CONTINUATION_2026-10-25.md`; **M1** |
| 237 | `a39ca967ea2029400649284285ca18620544152c` | `2026-10-25T12:34:50Z` | G | S | fix(gpu): qualify grouped transport from independent fresh state and seal KV cache views | `crates/zerogravity-gpu/src/` ×3; **M3** |
| 238 | `7362699b7023b30c815482a71900af40b0a0688f` | `2026-10-25T12:35:00Z` | G | S | fix(serve): preserve prepublication cancellations and bound decode context | `crates/zerogravity-cli/src/serve/gpu_batch.rs`; `crates/zerogravity-cli/src/serve/pending.rs`; **M2** |
| 239 | `5229e53d7b46c0fd322217b216b2e11a8b94a433` | `2026-10-25T12:37:00Z` | G | D | docs(review): record architectural and correctness findings and qualification gates | `.agents/docs/ARCHITECTURE_CORRECTNESS_REVIEW_2026-10-25.md`; **A1** |
| 240 | `83a31747679385cd5951487e6bcdfb9a2f2d1f24` | `2026-10-25T13:51:15Z` | G | S | refactor(gpu): one typed closed CUDA ABI for metadata and bound handles | `crates/zerogravity-gpu/src/` ×3; **A1,M2** |
| 241 | `095bf8a4b97dbd7a0561c7ea4316ef7ff8167bdd` | `2026-10-25T13:53:06Z` | G | M | refactor(core): collocate Qwen3.5 recipes without changing type identity | `crates/zerogravity-core/src/` ×27; **A13,M2,D12** |
| 242 | `2a47d25181f1a8d91a9489d9f2d621d3807bb329` | `2026-10-25T13:53:16Z` | G | S | feat(cli): concise run placement aliases and observed-device inventory | `crates/zerogravity-cli/src/` ×4; `crates/zerogravity-core/src/` ×1; **M5** |
| 243 | `d9c5872c73ee4d7b2bbbe1079446ac7634f8d919` | `2026-10-25T13:53:41Z` | G | D | docs: scope ABI family and run UX refactor acceptance | `.agents/docs/ABI_FAMILY_CLI_REFACTOR_2026-10-25.md`; **A1** |
| 244 | `5399ab93929eab8e6448078ae8126144b147b238` | `2026-10-25T14:31:01Z` | G | M | refactor(cli): use device selection without redundant on alias | `.agents/docs/` ×1; `crates/zerogravity-cli/src/` ×3; **M4** |
| 245 | `d84853337bbd2830f50a9c4e704d266990fd2ce4` | `2026-10-25T14:59:13Z` | G | M | feat(cli): resolve Linux GPU device nodes to verified physical PCI identities | `.agents/docs/` ×1; `crates/zerogravity-cli/src/` ×4; `crates/zerogravity-core/src/` ×1; **A2,M4** |

### 246–250 — NIGHT archive, B4/C16 outcomes and all-model correctness policy

The original245-source snapshot is committed unchanged, followed by real fixture repairs and the recorded26-scope/finite-serving/C16 distinctions. Common correctness tooling and policy explicitly prevent one passing numerical or benchmark slice from standing in for all models.

| # | Full commit SHA | Author date (ISO8601) | Sig | Class | Recorded subject | Affected owners/paths; change counts |
|---:|---|---|:---:|:---:|---|---|
| 246 | `d0fff261904fcb7836c0f71d0682028a12fd53b4` | `2026-10-25T16:26:13Z` | G | D | docs(handoff): preserve complete ZeroGravity history, open work and takeover prompt | `.agents/docs/` ×4; `.agents/plans/NEXT_AGENT_PROMPT_2026-10-25-NIGHT.md` ×1; **A5** |
| 247 | `a857d45ddea827457fb59823ea6597f3eca03d22` | `2026-10-26T02:55:15+08:00` | G | M | test(gpu): repair B4 qualification fixtures | `.agents/plans/cpu-gpu-optimization/` ×1; `crates/zerogravity-gpu/src/` ×1; `crates/zerogravity/tests/` ×7; **A1,M8** |
| 248 | `6610b6195505d7d8fb17640660461310fa1c3d41` | `2026-10-26T07:57:31+08:00` | G | D | docs: record B4 and C16 outcomes | `.agents/plans/cpu-gpu-optimization/TODO.md`; **M1** |
| 249 | `6d855599531e8f67ec6cd1e1e00f7e1b186ff7f3` | `2026-10-26T08:47:53+08:00` | G | M | test: add shared model correctness checks | `scripts/` ×3; `tools/model-correctness/` ×5; **A6,M2** |
| 250 | `195099ca583ace766d075682927f590360c823a5` | `2026-10-26T08:48:41+08:00` | G | D | docs: set all-model correctness policy | `.agents/docs/` ×1; `.agents/plans/TODO.md` ×1; `.agents/plans/cpu-gpu-optimization/` ×1; `.agents/plans/model-execution/` ×1; `.agents/plans/numerics-and-context/` ×2; **A1,M5** |

### 251–263 — 26 October semantic ownership, provider containment and recovery

Format/geometry naming, model-owned recipes, shared CPU ops, neutral GPU provider/bindings and application callers move as one source batch. Guard fixes, recovered missing source tests, preserved-side-work retirement and strict production dependency checks retain existing evidence while changing actual source identities.

| # | Full commit SHA | Author date (ISO8601) | Sig | Class | Recorded subject | Affected owners/paths; change counts |
|---:|---|---|:---:|:---:|---|---|
| 251 | `405a3e6358c85917506f76f780094de89eaec01b` | `2026-10-26T13:39:06+08:00` | G | M | refactor(formats): group codecs and format homes | `.agents/plans/pipeline-composition/` ×2; `crates/zerogravity-artifacts/src/` ×13; `crates/zerogravity-cpu/src/` ×9; `crates/zerogravity-gpu/src/` ×9; **A31,M2** |
| 252 | `d9a8ecf6187216c8b909c60e44927b06f256afdc` | `2026-10-26T13:40:43+08:00` | G | M | refactor(correctness): name input geometry | `.agents/plans/numerics-and-context/` ×1; `tools/model-correctness/` ×3; **M4** |
| 253 | `fa844a663ff9ab4f701146be61bf2353cf580385` | `2026-10-26T13:41:20+08:00` | G | D | docs: define module ownership and naming | `.agents/plans/pipeline-composition/` ×3; `AGENTS.md` ×1; **A1,M3** |
| 254 | `a5ed7de9e0f0682d86aca09d0ec73336a5f2ecf1` | `2026-10-26T15:07:57+08:00` | G | M | refactor(models): contain source plan ownership | `.agents/plans/pipeline-composition/` ×1; `Cargo.lock` ×1; `crates/zerogravity-artifacts/src/` ×7; `crates/zerogravity-artifacts/tests/` ×3; `crates/zerogravity-core/src/` ×39; `crates/zerogravity-cpu/src/` ×96; `crates/zerogravity-cpu/tests/` ×2; `crates/zerogravity-model-gemma4/` ×2; `crates/zerogravity-model-gemma4/src/` ×15; `crates/zerogravity-model-qwen35/` ×2; `crates/zerogravity-model-qwen35/examples/` ×6; `crates/zerogravity-model-qwen35/src/` ×101; `crates/zerogravity-model-qwen35/tests/` ×6; **A119,M72,D90** |
| 255 | `c90b8cd7e2fc8b05784d1e27c907347ea4319f97` | `2026-10-26T15:30:04+08:00` | G | M | refactor(gpu): separate providers and bindings | `.agents/plans/pipeline-composition/` ×1; `crates/zerogravity-gpu/` ×1; `crates/zerogravity-gpu/src/` ×1103; `crates/zerogravity-gpu/vendor/cubecl-cuda/` ×55; `tools/gpu-qualify/` ×15; **A509,M160,D506** |
| 256 | `41e93370b7b3b9edc45cd9575eaa6b9d6845bed2` | `2026-10-26T15:41:21+08:00` | G | M | fix(gpu): preserve plan and capture guards | `.agents/plans/pipeline-composition/` ×1; `crates/zerogravity-gpu/src/` ×8; **M9** |
| 257 | `fc734571e6e342b376f406aac7fc2ef4e42e69f2` | `2026-10-26T15:53:31+08:00` | G | M | refactor(app): compose model execution routes | `.agents/plans/pipeline-composition/` ×1; `crates/zerogravity-cli/` ×1; `crates/zerogravity-cli/src/` ×21; `crates/zerogravity-cli/tests/` ×3; `crates/zerogravity-model-qwen35/src/` ×5; `crates/zerogravity/` ×1; `crates/zerogravity/examples/` ×20; `crates/zerogravity/src/` ×49; `crates/zerogravity/tests/` ×30; `scripts/` ×4; `tools/hip-shared-models-qualification/` ×1; `tools/qwen27b-output-tail-reference/` ×1; **A42,M55,D40** |
| 258 | `4a0f4552354ad8f5931b7cb45bd85d34a0026eeb` | `2026-10-26T16:01:43+08:00` | G | D | docs: record completed source checkpoint | `.agents/plans/pipeline-composition/TODO.md`; **M1** |
| 259 | `5c2fe532c68e98a0c750b5ff27969ae965ec3f9d` | `2026-10-26T17:13:15+08:00` | G | M | test(cuda): restore source admission checks | `.agents/plans/pipeline-composition/TODO.md`; `crates/zerogravity-gpu/src/cuda/native/fixed_prompt_model/tests.rs`; **M2** |
| 260 | `9c2c51ab291580a5c47d9528f8ff2ed87bd8228a` | `2026-10-26T17:15:22+08:00` | G | D | docs: record side-worktree retirement | `.agents/plans/pipeline-composition/TODO.md`; **M1** |
| 261 | `fe1762568c96eead3893f338d0181b8634a98303` | `2026-10-26T18:26:57+08:00` | G | M | refactor(cpu): own shared GDN convolution | `.agents/plans/pipeline-composition/` ×1; `crates/zerogravity-cpu/src/` ×2; `crates/zerogravity-gpu/src/` ×1; `crates/zerogravity-model-qwen35/src/` ×3; `tools/gpu-qualify/` ×1; **A1,M7** |
| 262 | `263ff791d04eb50fd1aeca85249b0f7aed3cb1ad` | `2026-10-26T18:27:45+08:00` | G | M | refactor(gpu): remove model dependency | `.agents/plans/pipeline-composition/` ×2; `crates/zerogravity-gpu/` ×1; `crates/zerogravity-gpu/src/` ×45; `crates/zerogravity-model-qwen35/src/` ×4; `crates/zerogravity/` ×1; `crates/zerogravity/src/` ×5; `scripts/` ×1; `tools/gpu-qualify/` ×19; **A20,M54,D4** |
| 263 | `e7fc3150927104b2f22dd020f63efb8817a75d82` | `2026-10-26T18:31:38+08:00` | G | D | docs: explain benchmark container termination | `.agents/plans/cpu-gpu-optimization/` ×2; `.agents/plans/pipeline-composition/` ×1; **A1,M2** |

### 264–272 — Owned shutdown, finite original captures and offline recipe compilation

TERM/INT shutdown and benchmark/reference command ownership gain full stream/reap evidence. Native capture scope is separated from model prompt inventory; actual original47–50 packed references, rotary/test separation and an offline recipe emitter finish this cutoff. Fresh native48–50 and broader source/state/physical/model/serving gates remain open.

| # | Full commit SHA | Author date (ISO8601) | Sig | Class | Recorded subject | Affected owners/paths; change counts |
|---:|---|---|:---:|:---:|---|---|
| 264 | `50abaf71121a2ffbb6a649f16aafe185650c1388` | `2026-10-26T19:20:42+08:00` | G | M | fix(serve): join workers on process shutdown | `.agents/plans/serving-and-speculation/` ×1; `crates/zerogravity-cli/` ×1; `crates/zerogravity-cli/src/` ×6; `crates/zerogravity-cli/tests/` ×1; `nix/` ×1; **A3,M7** |
| 265 | `c84b98744955c43f7a6a992289c7d8a977a0ee9e` | `2026-10-26T19:21:14+08:00` | G | M | fix(bench): retain owned stop evidence | `.agents/plans/serving-and-speculation/` ×1; `scripts/` ×1; `tools/serve-lifecycle.py/` ×1; `tools/test-serve-lifecycle.py/` ×1; **A1,M3** |
| 266 | `4f427dd814f734afdad798b053141946c998347a` | `2026-10-26T19:28:49+08:00` | G | D | docs: record serving shutdown validation | `.agents/plans/cpu-gpu-optimization/` ×2; `.agents/plans/serving-and-speculation/` ×1; **M3** |
| 267 | `cd4159d26048b0395c40235b320b7d48df6ce53d` | `2026-10-26T21:58:07+08:00` | G | M | fix(lifecycle): retain owned command streams | `.agents/plans/numerics-and-context/` ×1; `tools/serve-lifecycle.py/` ×1; `tools/test-serve-lifecycle.py/` ×1; **M3** |
| 268 | `5a18b5bbe6a80976b9255813070c09d2cae61a3c` | `2026-10-26T21:58:36+08:00` | G | M | refactor(cuda): derive capture scope ownership | `.agents/plans/numerics-and-context/` ×1; `crates/zerogravity-gpu/src/` ×32; `crates/zerogravity-model-qwen35/src/` ×4; `crates/zerogravity/src/` ×4; **M41** |
| 269 | `61f095cf760ee6b8d1f015b71b13ac46835a4917` | `2026-10-26T21:59:12+08:00` | G | M | feat(reference): capture finite Qwen prompts | `.agents/plans/numerics-and-context/` ×1; `scripts/` ×1; `tools/qwen35-cuda-bundle/` ×4; `tools/qwen35-cuda-reference/` ×14; **A18,M2** |
| 270 | `3612ecb7077d18685e926c698b812e4e2e60375f` | `2026-10-26T23:14:16+08:00` | G | D | docs: define production and test boundaries | `AGENTS.md`; **M1** |
| 271 | `1f08e5525b5ede7124054156083e5a4ff67d151c` | `2026-10-26T23:23:45+08:00` | G | M | refactor: separate rotary and test preparation | `.agents/plans/numerics-and-context/` ×1; `.agents/plans/pipeline-composition/` ×1; `crates/zerogravity-gpu/src/` ×20; `crates/zerogravity-model-qwen35/src/` ×1; `crates/zerogravity/src/` ×3; `crates/zerogravity/tests/` ×1; **A5,M22** |
| 272 | `d74f59a18c6345dbea963861acd519c1282a26fa` | `2026-10-26T23:24:36+08:00` | G | M | feat(reference): emit captured CUDA recipes | `.agents/plans/numerics-and-context/` ×1; `.agents/plans/pipeline-composition/` ×1; `tools/cuda-text-recipes/` ×9; **A9,M2** |

## Reproducible metadata checks for this snapshot

The history lane executed only read-only Git/source and documentation checks:

```bash
git rev-list --count d74f59a18c6345dbea963861acd519c1282a26fa
git rev-list --merges d74f59a18c6345dbea963861acd519c1282a26fa
git log --reverse --topo-order \
  --format='%H%x09%aI%x09%cI%x09%G?%x09%GS%x09%GF%x09%s' \
  d74f59a18c6345dbea963861acd519c1282a26fa
# Per indexed commit; --root includes initial source/bootstrap contents:
git diff-tree --root --no-commit-id --name-status -r --no-renames COMMIT
```

Executed results:272 indexed/reachable/unique;0merges;272 good signatures;
245/245 original ordinal+SHA+subject joins exact; classification totals
98S+60D+114M=272. Table row/order/count and full-SHA equality were checked against
`git rev-list --reverse --topo-order` at the frozen cutoff. The seven emitter-baseline pins match their actual `4f427dd` Git bytes; the
2B generated-file diff is exactly the unused constant removal; all123 CUDA
GDN `.cu`/PTX/header payloads present at that baseline remain byte-identical
through the cutoff. These are historical coverage/source-equality checks, not
inference, numerical, physical, build or release tests.
