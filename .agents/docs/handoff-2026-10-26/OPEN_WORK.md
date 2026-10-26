# Open work — complete plan-marker inventory, 26 October 2026

Code cutoff: `d74f59a18c6345dbea963861acd519c1282a26fa`. Documentation-only snapshot. Live source, executed results,
latest user decisions and owning plan TODOs remain authoritative. This requested
handoff index creates no new queue, receipts, acceptance phases or tracking duties.
Do not tick its copied markers or treat a historical checkbox as current priority
or authorization. No plan/source edits, builds, model/device runs, downloads,
Git mutations or worktree changes were performed by this inventory lane.

## Coverage and identity

- All **17** `TODO.md` files recursively under `.agents/plans/` are covered:
  **11 live owning plans**, **1 queue**, **5 archival TODOs**.
- Live owning plans: **374 noncompleted occurrences = 361 `[ ]` + 12 `[~]` +
  1 `[!]`**. Of these, **295** occur under `Uncommitted`, **57** under tests,
  **16** under deferred headings and **6** under `Committed`.
- Queue: **11** noncompleted plan-level entries = 3 `[ ]` + 8 `[~]`; these
  refer to the live plans and are not 11 additional implementation tasks.
- Archives: **0** noncompleted, **338** completed historical occurrences.
- All17 sources contain **1,148** checkbox occurrences:385 noncompleted
  (including queue) and763 completed. No unchecked archive work was excluded.
- Each copied occurrence has an identity `path:Lstart-Lend@TODO-SHA256`,
  full heading path, exact source span and full unabridged Markdown block.
  The per-entry `block_sha256` identifies the copied UTF-8 bytes, including
  their final newline. Both source and block identities are snapshot-specific.
- No semantic deduplication: tasks may overlap across owners or repeat old
  decisions. All374 full-block hashes and first lines are distinct. Neither
  a shared concept nor a later successful sub-scope erases an earlier marker.
- Block ends at the next checkbox or Markdown heading outside fenced code;
  trailing empty lines are excluded. Attached unmarked paragraphs/bullets remain
  included, so failures, exceptions, dependencies and later scope notes survive.
  Markdown links inside literal copied blocks retain their original text;
  resolve them from the owning TODO, not from this handoff's directory.
- File links give start–end source line spans at the cutoff; later edits may
  move lines. The full copied text and SHA256 preserve the checkpoint.

The sorted source-set digest is `dca96f5ab9480561ad2714d946d02487ee7da990907605f99b5c7718ca992468`, computed over one UTF-8 record
per file: `repository_relative_path + NUL + source_SHA256 + LF`. It covers all17
sources in the tables below. It identifies this requested index only and is not
a new runtime or workflow requirement.

## Current priority and dependency matrix

This matrix interprets the latest owner direction; it does not promote icebox
plans or authorize a device/source action. Older conflicting prose is retained
verbatim below and called out separately.

| Order / work | Prerequisites and completion boundary | Owners / continuation |
|---|---|---|
| Architecture, model containment and production/test separation | Finish remaining rotary/fault/qualifier and captured-case dispatch debt; actual consumers/dependencies/exports/features must follow the same boundary. Test whitelists cannot become runtime capability descriptors. Preserve real source/ABI/arithmetic/geometry/ownership facts. | Pipeline composition + numerics; shared/model/backend code owners. Applies to .8B,2B,27B and Gemma together, before new .8B47–50, long-context or backend expansion. |
| All-model correctness workflow | One shared process: semantics/source/state/physical safety; independently declared per-op accuracy; pinned original model/task accuracy; determinism as a separate claim. All present/future registry targets, including2B,27B and Gemma, follow it. | GPT-6.1 Sol xhigh in every coordination, architecture, implementation, execution and review lane. Previous Pi/Astra/Sol-HIGH allocations are historical. |
| .8B fresh47/48/49/50 + cached47→50 | Fresh prompt lengths are distinct from cached frontiers. Reference packed captures now exist; native48–50 remain unsupported pending real source/ABI/math/state/physical qualification. Preserve FAIL101, original six-word N1/B4 difference and quarantine. | Numerics + model execution; real-model serving shutdown/fault scopes remain separate required evidence. |
| Genuine long context and backend routes | Actual512/1024/2048/4096/8192, then16K/32K prefixes/continuations and all required state/fault checks; original BF16 CPU/CUDA/ROCm(HIP)/Vulkan cells qualify independently. Padding/large rotary positions, GGUF results and synthetic operators do not fill missing cells. | Numerics + model/backend owners. No new timing/27B capture before the current correctness order. Local Renoir/gfx90c launch/reset/recovery hold remains. |
| Affected assembled validation | Meaningful numerical/source/state/cancellation/physical/serving checks for changed behavior; full isolated `scripts/check.sh` last on the coherent batch. Source/compile/feature/CPU checks do not establish fresh physical inference. | Coordinator inspects every semantic and cleanup outcome. Full27B/Gemma model suites are waived for the current refactor only; record unrun, never passed. No new policy/convention placement tests. |
| Matched optimization and larger models | After correctness: source-bound B4 physical/CPU coverage, optimized same-attempt quiet C16 for both small models; C32/C64 remain queue-dependent. Continue27B/Gemma contained source integration where independent, but actual model/backend qualification and performance remain separate. | CPU/GPU optimization + model execution + serving. Historical C1/C4/C8 wins/losses remain tied to their exact source/profile/attempt; no C2 in the current C16 sequence. |
| CLI, storage and serving foundations | Preserve the one mathematical engine, immutable SourceIdentity, independent StateRoots, one physical ResourceArbiter/fair queue and authentic completion/quarantine. Pre-Client admission, complete peaks, paging/COW/prefix, source streaming, ingress/output credit and public route qualification stay with their owners. | Active CLI/memory/serving/model-acquisition plans; completed finite slices do not close broad plan markers. |
| Icebox and deferred extensions | Multimodal, distributed, platform/release and future sparse/codec/speculation/hosted work retain exact source/license/numerical/physical/quality gates. Acquisition/registry/provider metadata grants no execution eligibility. | Do not elevate archived/deferred checklist wording into present-day work; source/device/publication authorization remains separate. |

Current policy/spec owners:

- [.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md](../../plans/numerics-and-context/CORRECTNESS_POLICY.md)
- [.agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md](../../plans/pipeline-composition/STRUCTURE_AND_NAMING.md)
- [.agents/plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md](../../plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md)
- [.agents/plans/cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md](../../plans/cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md)
- [.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md](../../plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md)

## Live owning plan source counts

| Plan | Lines | Bytes | Source SHA256 | `[ ]` | `[~]` | `[!]` | Total |
|---|---:|---:|---|---:|---:|---:|---:|
| [cli-device-routing](../../plans/cli-device-routing/TODO.md) | 288 | 18691 | `db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162` | 60 | 0 | 0 | 60 |
| [cpu-gpu-optimization](../../plans/cpu-gpu-optimization/TODO.md) | 1219 | 89395 | `66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038` | 48 | 7 | 0 | 55 |
| [distributed-execution](../../plans/distributed-execution/TODO.md) | 53 | 3876 | `033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff` | 12 | 0 | 0 | 12 |
| [memory-and-streaming](../../plans/memory-and-streaming/TODO.md) | 174 | 12346 | `d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0` | 25 | 0 | 0 | 25 |
| [model-acquisition](../../plans/model-acquisition/TODO.md) | 353 | 46012 | `861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86` | 18 | 0 | 0 | 18 |
| [model-execution](../../plans/model-execution/TODO.md) | 1449 | 99671 | `ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f` | 43 | 3 | 0 | 46 |
| [multimodal](../../plans/multimodal/TODO.md) | 73 | 7625 | `774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208` | 27 | 0 | 0 | 27 |
| [numerics-and-context](../../plans/numerics-and-context/TODO.md) | 713 | 87328 | `2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa` | 29 | 2 | 0 | 31 |
| [pipeline-composition](../../plans/pipeline-composition/TODO.md) | 846 | 55159 | `0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b` | 52 | 0 | 1 | 53 |
| [platform-and-release](../../plans/platform-and-release/TODO.md) | 100 | 7632 | `ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1` | 17 | 0 | 0 | 17 |
| [serving-and-speculation](../../plans/serving-and-speculation/TODO.md) | 978 | 66926 | `82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0` | 30 | 0 | 0 | 30 |

## Queue and archival TODOs — separate scopes

| Role / source | Lines | Source SHA256 | `[ ]` | `[~]` | `[!]` | `[x]` |
|---|---:|---|---:|---:|---:|---:|
| queue: [.agents/plans/TODO.md](../../plans/TODO.md) | 24 | `721aebc8d047b7bdeae0dfa380982fb34e61ed22788d187dfd66b0b95d1a96b9` | 3 | 8 | 0 | 3 |
| archive: [.agents/plans/archive/history/accepted-baseline/TODO.md](../../plans/archive/history/accepted-baseline/TODO.md) | 38 | `968831b1b3183e027222842127791b88509781adce2e08432acf0cc1eaef0a51` | 0 | 0 | 0 | 13 |
| archive: [.agents/plans/archive/history/handoff-cleanup/TODO.md](../../plans/archive/history/handoff-cleanup/TODO.md) | 57 | `0d92b720ff0ece86e003d4235dde95683eae408bbf2def229d2839459130433e` | 0 | 0 | 0 | 14 |
| archive: [.agents/plans/archive/history/historical-records/TODO.md](../../plans/archive/history/historical-records/TODO.md) | 95 | `2ac60f20d2823af7d309c7867a3cb2cb64469dc42d6f82c011a834b655c3ecda` | 0 | 0 | 0 | 53 |
| archive: [.agents/plans/model-acquisition/archive/through-a774ded/TODO.md](../../plans/model-acquisition/archive/through-a774ded/TODO.md) | 120 | `4a178da79a9bc842b39d797f3a8141b28525e5a97cba3a11ba304c8edab27667` | 0 | 0 | 0 | 85 |
| archive: [.agents/plans/model-acquisition/archive/through-fce1353/TODO.md](../../plans/model-acquisition/archive/through-fce1353/TODO.md) | 218 | `e0750281ddc39d99dac3e44542be9087d3fa18b305c0afa39c3785a502032d61` | 0 | 0 | 0 | 173 |

The five archival TODOs are closed historical records. They add no unresolved
marker and no current priority. Live plan headings labelled `Committed` may still
contain unresolved markers; all six are copied in their real locations below.

### Exact queue markers

<a id="queue-l5"></a>

#### ~ · [L5–5](../../plans/TODO.md#L5)

Heading: `Plan queue > Active`.
Identity: `.agents/plans/TODO.md:L5-L5@721aebc8d047b7bdeae0dfa380982fb34e61ed22788d187dfd66b0b95d1a96b9`.
Block SHA256: `87a648f6ae0efdd5aa6d2e8cb9bd6f980ef6d28a58aeb3f461e8c02d0f795b79`.

````markdown
- [~] P0 serving-and-speculation — @pvl — [details](serving-and-speculation/TODO.md)
````

<a id="queue-l6"></a>

#### ~ · [L6–6](../../plans/TODO.md#L6)

Heading: `Plan queue > Active`.
Identity: `.agents/plans/TODO.md:L6-L6@721aebc8d047b7bdeae0dfa380982fb34e61ed22788d187dfd66b0b95d1a96b9`.
Block SHA256: `c6dc2d22c9a90ffa1b2e1a0f0e2f6ce66422d473b9dccbf9e58597a52f9fa0d9`.

````markdown
- [~] P1 numerics-and-context — @pvl — shared vLLM/llama.cpp correctness; .8B47–50 then genuine long contexts — [details](numerics-and-context/TODO.md)
````

<a id="queue-l7"></a>

#### ~ · [L7–7](../../plans/TODO.md#L7)

Heading: `Plan queue > Active`.
Identity: `.agents/plans/TODO.md:L7-L7@721aebc8d047b7bdeae0dfa380982fb34e61ed22788d187dfd66b0b95d1a96b9`.
Block SHA256: `090dd3a89dc8e38dadfda1f5a70238a633bd04de9a6ed3f923727f1da5107dd6`.

````markdown
- [~] P1 memory-and-streaming — @pvl — [details](memory-and-streaming/TODO.md)
````

<a id="queue-l8"></a>

#### ~ · [L8–8](../../plans/TODO.md#L8)

Heading: `Plan queue > Active`.
Identity: `.agents/plans/TODO.md:L8-L8@721aebc8d047b7bdeae0dfa380982fb34e61ed22788d187dfd66b0b95d1a96b9`.
Block SHA256: `fc043ee415152971a43780a5a92dff1c5d55a4c27dc8ce897aa2216ee80dfb4f`.

````markdown
- [~] P1 cpu-gpu-optimization — @pvl — correctness before performance; separate CUDA/HIP/Vulkan qualification — [details](cpu-gpu-optimization/TODO.md)
````

<a id="queue-l9"></a>

#### ~ · [L9–9](../../plans/TODO.md#L9)

Heading: `Plan queue > Active`.
Identity: `.agents/plans/TODO.md:L9-L9@721aebc8d047b7bdeae0dfa380982fb34e61ed22788d187dfd66b0b95d1a96b9`.
Block SHA256: `99c054855b1ee09f76743d99936b01d5fd982407eabd9cfb2280b194c90bcbc3`.

````markdown
- [~] P1 pipeline-composition — @pvl — recursive primitives/pipelines/profiles; shared execution adapters active — [details](pipeline-composition/TODO.md)
````

<a id="queue-l10"></a>

#### ~ · [L10–10](../../plans/TODO.md#L10)

Heading: `Plan queue > Active`.
Identity: `.agents/plans/TODO.md:L10-L10@721aebc8d047b7bdeae0dfa380982fb34e61ed22788d187dfd66b0b95d1a96b9`.
Block SHA256: `a62bcb4e526df886b64712473c260f0f266eaa3f5025d3a2c215bf4508955b70`.

````markdown
- [~] P1 cli-device-routing — @pvl — unified run/config/source/device selection; parallel implementation authorized24Oct — [details](cli-device-routing/TODO.md)
````

<a id="queue-l11"></a>

#### ~ · [L11–11](../../plans/TODO.md#L11)

Heading: `Plan queue > Active`.
Identity: `.agents/plans/TODO.md:L11-L11@721aebc8d047b7bdeae0dfa380982fb34e61ed22788d187dfd66b0b95d1a96b9`.
Block SHA256: `835e7a892b44c222cab782dbb4694669774abebb4bbe89966aa962ee390199eb`.

````markdown
- [~] P2 model-execution — @pvl — all-model correctness policy; missing full-model/backend adapters remain open — [details](model-execution/TODO.md)
````

<a id="queue-l12"></a>

#### ~ · [L12–12](../../plans/TODO.md#L12)

Heading: `Plan queue > Active`.
Identity: `.agents/plans/TODO.md:L12-L12@721aebc8d047b7bdeae0dfa380982fb34e61ed22788d187dfd66b0b95d1a96b9`.
Block SHA256: `146230b10478f3f44668ee416f134c4ddf0e6eb4e5e857d948cbbeb23d4c8dfe`.

````markdown
- [~] P2 model-acquisition — @pvl — [details](model-acquisition/TODO.md)
````

<a id="queue-l16"></a>

#### open · [L16–16](../../plans/TODO.md#L16)

Heading: `Plan queue > Icebox`.
Identity: `.agents/plans/TODO.md:L16-L16@721aebc8d047b7bdeae0dfa380982fb34e61ed22788d187dfd66b0b95d1a96b9`.
Block SHA256: `cf88b46e208de1f848fe40d18220080c61c7ebb259a30147f3d31c5cb56cb0d8`.

````markdown
- [ ] multimodal — [details](multimodal/TODO.md)
````

<a id="queue-l17"></a>

#### open · [L17–17](../../plans/TODO.md#L17)

Heading: `Plan queue > Icebox`.
Identity: `.agents/plans/TODO.md:L17-L17@721aebc8d047b7bdeae0dfa380982fb34e61ed22788d187dfd66b0b95d1a96b9`.
Block SHA256: `8560e1ec03d71af5cd991cb0281ece484a4bdebc82049f602b1e820d65136beb`.

````markdown
- [ ] distributed-execution — [details](distributed-execution/TODO.md)
````

<a id="queue-l18"></a>

#### open · [L18–18](../../plans/TODO.md#L18)

Heading: `Plan queue > Icebox`.
Identity: `.agents/plans/TODO.md:L18-L18@721aebc8d047b7bdeae0dfa380982fb34e61ed22788d187dfd66b0b95d1a96b9`.
Block SHA256: `447eaad9cf73883df78083ac14bc2e42e51faa84e3bc1943b793e806bf5a8d8e`.

````markdown
- [ ] platform-and-release — [details](platform-and-release/TODO.md)
````

## Historical360 companion — preserved, reconciled by occurrence

- Original: [.agents/docs/HANDOFF_ALL_OPEN_PLAN_ITEMS_2026-10-25.md](../HANDOFF_ALL_OPEN_PLAN_ITEMS_2026-10-25.md). SHA256 `cb8b18c456a9161e7d7a2c742507e2420a33bbb7305d00e6d3cd785c8dcc3972`; source HEAD `d84853337bbd2830f50a9c4e704d266990fd2ce4`. Its464 lines and360 first-line entries were read; its bytes are unchanged.
- Historical totals:351 open,8 active,1 blocked across11 live owners. It
  excludes queue/archive TODOs and intentionally abbreviates descriptions at
  their first line. This new handoff includes full multiline blocks.
- Reconciliation key: `(owning TODO path, marker status, HTML-unescaped exact
  first-line text)`. All360 historical keys still occur among the374 current
  markers; **0 historical-only keys**, **14 new keys**. This confirms marker
  retention, not unchanged continuation paragraphs or accepted task completion.
- Six TODO source hashes remain identical; five changed: CPU/GPU optimization,
  model execution, numerics/context, pipeline composition, serving/speculation.
  New headings/lines move historical references; read current spans below.

### Fourteen added occurrences

| Plan | Current span / heading | Exact first line |
|---|---|---|
| cpu-gpu-optimization | [L19–22](../../plans/cpu-gpu-optimization/TODO.md#L19-L22) · Current correctness-first order — 26 Oct | `[~]` &#91;Shared vLLM/llama.cpp correctness policy&#93;(../numerics-and-context/CORRECTNESS_POLICY.md) |
| cpu-gpu-optimization | [L23–26](../../plans/cpu-gpu-optimization/TODO.md#L23-L26) · Current correctness-first order — 26 Oct | `[ ]` Preserve case5 FAIL101 and quarantine. Original default vLLM N1/B4 also |
| cpu-gpu-optimization | [L30–34](../../plans/cpu-gpu-optimization/TODO.md#L30-L34) · Current B4 takeover — 25–26 Oct | `[ ]` Current B4 → fresh optimized matched C16 (.8B +2B) → x2 original27B, |
| model-execution | [L13–17](../../plans/model-execution/TODO.md#L13-L17) · Current owner direction — 26 Oct | `[~]` Every existing/future registry target follows the shared |
| model-execution | [L18–41](../../plans/model-execution/TODO.md#L18-L41) · Current owner direction — 26 Oct | `[ ]` Implement/qualify missing full-model/context/backend cells, beginning |
| numerics-and-context | [L85–89](../../plans/numerics-and-context/TODO.md#L85-L89) · Reference capture and remaining runtime separation — 26 Oct | `[ ]` Remove remaining shared rotary fault assertions/fixed-prompt qualifier |
| numerics-and-context | [L93–102](../../plans/numerics-and-context/TODO.md#L93-L102) · Current owner order — 26 Oct | `[~]` Sequential continuation after serving shutdown: fresh .8B prompt47/48/49/50 |
| numerics-and-context | [L108–110](../../plans/numerics-and-context/TODO.md#L108-L110) · Current owner order — 26 Oct | `[~]` Apply &#91;all-model correctness policy&#93;(CORRECTNESS_POLICY.md), based on |
| numerics-and-context | [L111–116](../../plans/numerics-and-context/TODO.md#L111-L116) · Current owner order — 26 Oct | `[ ]` .8B initial47/48/49/50 and cached47→50: actual independent original and |
| numerics-and-context | [L117–121](../../plans/numerics-and-context/TODO.md#L117-L121) · Current owner order — 26 Oct | `[ ]` Genuine .8B512/1024/2048/4096/8192, then16K/32K context/state/feedback; |
| pipeline-composition | [L142–144](../../plans/pipeline-composition/TODO.md#L142-L144) · Production/test separation — 26 Oct | `[ ]` Finish remaining qualifier/fault and captured testcase dispatch boundaries |
| pipeline-composition | [L154–156](../../plans/pipeline-composition/TODO.md#L154-L156) · Model containment and naming — 26 Oct | `[ ]` Extend source-bound .8B qualification to fresh47–50-token prompts, genuine |
| pipeline-composition | [L157–162](../../plans/pipeline-composition/TODO.md#L157-L162) · Model containment and naming — 26 Oct | `[ ]` After source-bound correctness and legitimate source access: current |
| serving-and-speculation | [L13–14](../../plans/serving-and-speculation/TODO.md#L13-L14) · Shutdown qualification — 26 Oct | `[ ]` Replay applicable original-model/backend shutdown and fault scopes under |

## Stale or inconsistent prose — retained, not silently resolved

| Location | Discrepancy / interpretation at cutoff |
|---|---|
| Plan rules README, older plan checkpoints and linked design authorship | Older Pi/DeepSeek/Astra/Sol-HIGH role instructions conflict with the latest owner instruction. All current lanes use GPT-6.1 Sol xhigh; historical authorship/evidence stays intact. |
| CPU/GPU TODO `Current B4 takeover`, older C16 source checkpoint; model execution24Oct priority text; older benchmark/backend specs | Performance-first / next27B wording is superseded by architecture/test separation and all-model correctness-first order. Preserve those marker occurrences, historical wins/losses and no-C2 constraint; do not execute their stale next action. |
| Pipeline composition model containment; numerics reference capture; all-model policy | Full27B/Gemma model suites were waived only for the source refactor. Source/compile/full-CPU-gate facts cannot become physical or whole-model passes. Fresh native48–50 remain unsupported; captured cached frontiers or reference-only success do not change that. |
| CLI TODO88–288 versus `Committed`12–80 | Many original unit1 design/test markers remain open despite later named implementation/finite validation. Keep them as source occurrences; do not report60 independently missing implementations or re-run all completed coverage by default. Shared pre-Client transactional admission is explicitly still open. |
| CPU/GPU TODO801/846/856/880,1194,1218–1219 | Open markers contain later pass records or older “running”/“last accepted whole gate”/pending retirement statements. Their exact scopes differ from later26Oct assembled gate and retirement records. A checklist occurrence alone is not proof that a job remains live or that later source received physical coverage. |
| Numerics TODO313–315 and702–713 | Historical SiLU formatting ownership and ISO-only/natural-feedback statements coexist with later code/qualification records. They remain unresolved source markers; current caller/source/result review must decide what survives. Never erase failed parent gates because a later leaf passed. |
| Model execution TODO49–66 andpipeline TODO846 | Historical H/G/V pass3 failures and blocked HIPv4/controlv5 controller record are not proof of a current product failure or a running job. Later review must reconcile exact sources, final feedback/PLE fixture and validator fields; preserve stopped dependents and failed receipts. |
| Model execution original “no weights downloaded” and4GiB wording versus later acquired/window/replay records | Dated source acquisition and source-window support advanced. Different source-only, finite CPU replay, learned HIP and current runtime eligibility scopes coexist; no broad whole-model/native/serving support follows. |
| Pipeline TODO old dependency-check failure/full-gate notes versus later package-boundary commits | Older checkpoint failures remain historical. Later source boundary and full-gate passes cover their named source; remaining qualifier/captured-case dispatch debt remains open. No new convention/policy placement tests are requested. |

No owning TODO, plan priority, old marker or source prose was edited to resolve
these discrepancies. The latest user decision and exact executed scope take
precedence when continuing. Broken links are registered below as provenance
problems, not repaired by guessed paths.

## Complete exact live-marker blocks

The following374 blocks are in owning-file/source order, including tests,
deferred work and unresolved markers within committed history. Heading paths
are copied from the source; a heading's date is historical context, not a fresh
authorization or priority assignment.

### cli-device-routing — 60 noncompleted occurrences

<a id="cli-device-routing-l88"></a>

#### open · [L88–90](../../plans/cli-device-routing/TODO.md#L88-L90)

Heading: `CLI and device routing > Uncommitted > Reviewed source checkpoint — 24 Oct`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L88-L90@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `5ec45dc455a1fcef1686cec922d7af285554fa6b588f6a015b72e24846c6ba5f`.

````markdown
- [ ] Shared pre-Client transactional admission remains open and explicitly
  reported by the finite adapter; registration/current-free checks are not an
  OS hard memory guarantee or a reservation before Client creation.
````

<a id="cli-device-routing-l94"></a>

#### open · [L94–95](../../plans/cli-device-routing/TODO.md#L94-L95)

Heading: `CLI and device routing > Uncommitted > Review and authorization`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L94-L95@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `4bb5990865c749248d09bb316ce88845aecf14ac5ba2c20e98aaf7993ffac4d9`.

````markdown
- [ ] Recheck live source, ownership and accepted model/backend evidence before
  implementation; preserve root README, other plans and active implementation.
````

<a id="cli-device-routing-l99"></a>

#### open · [L99–102](../../plans/cli-device-routing/TODO.md#L99-L102)

Heading: `CLI and device routing > Uncommitted > Explicit source-form addition — 24 Oct`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L99-L102@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `f0871df51710e099e0b2919a7b083d20a50b46a0c19425d2b090178b4c1d51ef`.

````markdown
- [ ] Accept local paths, catalog/model names, `hf:`, `gh:`, `http:` and `https:`
  through one shared source selector and the existing acquisition/cache identity
  contracts. Recognized source schemes take precedence over local YAML suffix
  classification; only a local `.yaml`/`.yml` selector implies a run document.
````

<a id="cli-device-routing-l103"></a>

#### open · [L103–106](../../plans/cli-device-routing/TODO.md#L103-L106)

Heading: `CLI and device routing > Uncommitted > Explicit source-form addition — 24 Oct`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L103-L106@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `6abc3408784cdd201c24cc14e675bfec92b1bea57a5c48eb9c606c9a79d5f8cc`.

````markdown
- [ ] Resolve already acquired immutable exact sources, reject ambiguous variants
  and preserve explicit acquisition guidance when absent. URI parsing/resolution
  is not inference qualification. Reuse provider parsing/source identity rather
  than a second downloader or format-specific run loop; dry-run stays read-only.
````

<a id="cli-device-routing-l107"></a>

#### open · [L107–109](../../plans/cli-device-routing/TODO.md#L107-L109)

Heading: `CLI and device routing > Uncommitted > Explicit source-form addition — 24 Oct`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L107-L109@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `7d54185e9816b64a2fe8ee20a4eaee3a3251392d5afa8d7cb5fab80afac06906`.

````markdown
- [ ] Every new CLI semantic option has a typed v2 field. Unit1 implementation
  and tests precede ROOT guarded fresh-mtime merge; final assembled gate remains
  required. Future split/NPU/remote units retain separate qualification.
````

<a id="cli-device-routing-l117"></a>

#### open · [L117–118](../../plans/cli-device-routing/TODO.md#L117-L118)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L117-L118@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `656df4c5e2cff2341fd5e3682fcaf8d767a2f90d12ad6f0b6a0612399c10b5a8`.

````markdown
- [ ] Normalize `run`, backend aliases and common task flags into one library
  request; preserve compatible existing invocations and actionable diagnostics.
````

<a id="cli-device-routing-l119"></a>

#### open · [L119–121](../../plans/cli-device-routing/TODO.md#L119-L121)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L119-L121@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `740981bf3d8fb72ae647b0f40a7fc3c295e8d110178334e69c0058ea9d6b3ab2`.

````markdown
- [ ] Extend the canonical pipeline document/compiler to proposed v2 with one
  model table, explicit model/pipeline entrypoint and typed run policy/task/
  diagnostics; do not introduce a separate config interpreter or runtime.
````

<a id="cli-device-routing-l122"></a>

#### open · [L122–124](../../plans/cli-device-routing/TODO.md#L122-L124)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L122-L124@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `51155ce0dcd1fb9313f0cfb2f8daae5de9a20874656800515bfc21fc254e788e`.

````markdown
- [ ] Implement unambiguous positional model/config and explicit selectors;
  selector-free run loads only absolute-XDG/HOME `abird/zerogravity/config.yaml`.
  Missing/unreadable/invalid config fails without fallback or initialization.
````

<a id="cli-device-routing-l125"></a>

#### open · [L125–127](../../plans/cli-device-routing/TODO.md#L125-L127)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L125-L127@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `da0f75cbc89d6cb9a453080c8b4ddaf75fd2832776ec171578b91f5e864fbd57`.

````markdown
- [ ] Implement CLI > selected config > built-in resolution with per-field
  provenance, origin-correct paths, whole-list replacement and explicit boolean/
  zero/empty semantics. Preserve alias equivalence and final constraint checks.
````

<a id="cli-device-routing-l128"></a>

#### open · [L128–129](../../plans/cli-device-routing/TODO.md#L128-L129)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L128-L129@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `1db64ce57e572a1429ae2edbf7d00e4dd9ea313ef7a020a84b3e1008d5236865`.

````markdown
- [ ] Migrate model metadata to `--model-config`; cover every existing task,
  tokenizer, teacher-ID, prefill and audit option with typed config equivalents.
````

<a id="cli-device-routing-l130"></a>

#### open · [L130–132](../../plans/cli-device-routing/TODO.md#L130-L132)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L130-L132@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `aee7ef1dac5073037c10f95e4e30730f82312b44236f2be94dd68ce23abc7fe5`.

````markdown
- [ ] Retain bounded strict YAML/graph validation; check the complete merged
  document before effects. Declared routes remain inert under run; graph and
  recursive execution still need their independently qualified implementation.
````

<a id="cli-device-routing-l133"></a>

#### open · [L133–134](../../plans/cli-device-routing/TODO.md#L133-L134)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L133-L134@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `176b9866a4d166c33876fe14742bd243c2f8c2b5000b7ba2cea78880e5f30b8c`.

````markdown
- [ ] Resolve exact local source and bounded task/model requirements through
  existing adapters; catalog/metadata status grants no executable capability.
````

<a id="cli-device-routing-l135"></a>

#### open · [L135–136](../../plans/cli-device-routing/TODO.md#L135-L136)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L135-L136@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `6ba29c72745182c53d1c8e9f37b152450de5dd37e4c7ff3073e1869c085caa96`.

````markdown
- [ ] Add read-only inventory and stable physical identity resolution; derive
  actual vendor/backend support and deduplicate cross-backend aliases.
````

<a id="cli-device-routing-l137"></a>

#### open · [L137–138](../../plans/cli-device-routing/TODO.md#L137-L138)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L137-L138@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `cd07eb4ed3865d1111b0d56fe3afd86aa105319b8ac6ac982c061fffb7f1a733`.

````markdown
- [ ] Implement strict ordered allowlists, CPU-terminal semantics, local-only
  auto and all-neural-stage forced-backend checks before preparation.
````

<a id="cli-device-routing-l139"></a>

#### open · [L139–141](../../plans/cli-device-routing/TODO.md#L139-L141)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L139-L141@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `1a566397fa24ba44db820df33fdb11c7b17c2cb28ac6fb07f87900d1af16b0a8`.

````markdown
- [ ] Implement current-memory observation with explicit provenance/age/unknown
  status, OS/cgroup constraints and host/UMA domain relationships. Do not reuse
  `max_memory` as free memory or mutate established operator ceilings.
````

<a id="cli-device-routing-l142"></a>

#### open · [L142–144](../../plans/cli-device-routing/TODO.md#L142-L144)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L142-L144@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `75a00d8e9761c1bdc8d97f4803ab8623e07e34b877ce834b0a0ec06b3890bee6`.

````markdown
- [ ] Resolve a safe default operator-ceiling/bootstrap policy once for a new
  process before context effects, so bare `run` needs no manual resource caps;
  preserve established domains and benchmark/serving controls.
````

<a id="cli-device-routing-l145"></a>

#### open · [L145–147](../../plans/cli-device-routing/TODO.md#L145-L147)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L145-L147@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `39719511963e7ba9036a0f55c38a3ba5abdb0ffac62b136c5aa5ea977ec86a29`.

````markdown
- [ ] Integrate complete cold/prefill/decode/native/compiler/state/output costs
  with original ledger snapshots and authentic existing backing/reuse. Close or
  expose pre-admission context-bootstrap gaps; unknown costs cannot claim fit.
````

<a id="cli-device-routing-l148"></a>

#### open · [L148–149](../../plans/cli-device-routing/TODO.md#L148-L149)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L148-L149@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `b1dcf990bad665243c99651447d35fb6b0b8cf5e0658ad1f051ca13727a544cc`.

````markdown
- [ ] Implement configurable headroom, all-GPU full-fit scan and qualified CPU
  fallback; explain missing split support when a model would otherwise spill.
````

<a id="cli-device-routing-l150"></a>

#### open · [L150–151](../../plans/cli-device-routing/TODO.md#L150-L151)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L150-L151@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `66b50e403b401ac196da49ee39e7095a4eaf891e661df8d5f041233daa02abf1`.

````markdown
- [ ] Revalidate and admit through original owners; bound safe startup replanning,
  preserve unknown-work fences and retain all submitted owners.
````

<a id="cli-device-routing-l152"></a>

#### open · [L152–153](../../plans/cli-device-routing/TODO.md#L152-L153)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L152-L153@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `ef71ec9beeb72faaf5413bbca6b8ce8f82cd4089bdd90730c0e00cca4d38f8ca`.

````markdown
- [ ] Provide `devices list`, `--dry-run`, `--explain` and structured refusals
  without hidden execution/network/cache mutation during diagnostics.
````

<a id="cli-device-routing-l154"></a>

#### open · [L154–155](../../plans/cli-device-routing/TODO.md#L154-L155)

Heading: `CLI and device routing > Uncommitted > Proposed unit 1 — useful local run policy`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L154-L155@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `fecc159eb6a271ef0fde71321d54a718b3b5171a859872219302a8eb8f23d32a`.

````markdown
- [ ] Validate the assembled unit on approved source/hardware/backend tuples;
  document actual supported tuples and unavailable current-resource probes.
````

<a id="cli-device-routing-l164"></a>

#### open · [L164–165](../../plans/cli-device-routing/TODO.md#L164-L165)

Heading: `CLI and device routing > Uncommitted > Proposed unit 2 — local layer spill with real state`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L164-L165@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `aca2aa8ab55fc6c679f2653ce97048205e73d5531da92d598eaa8d1895841d1e`.

````markdown
- [ ] Define source-specific indivisible groups and embedding/head/shared-weight/
  MTP ownership; produce one complete feasible placement before any preparation.
````

<a id="cli-device-routing-l166"></a>

#### open · [L166–167](../../plans/cli-device-routing/TODO.md#L166-L167)

Heading: `CLI and device routing > Uncommitted > Proposed unit 2 — local layer spill with real state`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L166-L167@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `e51cb3aa58c2b31ba80734e4da0309374d9dfa836123911b32d010ecc3defcae`.

````markdown
- [ ] Extend original arbiter with all-or-none multi-domain admission if needed;
  collapse aliases and UMA domains, preserve prepaid-envelope/sublease rules.
````

<a id="cli-device-routing-l168"></a>

#### open · [L168–170](../../plans/cli-device-routing/TODO.md#L168-L170)

Heading: `CLI and device routing > Uncommitted > Proposed unit 2 — local layer spill with real state`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L168-L170@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `bded1e2ccd35fdc9c472baf81cc5fdc5113b2ef3b71c228f12059292848c291f`.

````markdown
- [ ] Integrate original state shards, typed activations and actual event/reader
  transfers into the existing prepared physical plan and fair workers; join all
  Ready events before the original SessionAuthority publishes one logical root.
````

<a id="cli-device-routing-l171"></a>

#### open · [L171–172](../../plans/cli-device-routing/TODO.md#L171-L172)

Heading: `CLI and device routing > Uncommitted > Proposed unit 2 — local layer spill with real state`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L171-L172@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `4de9262a13c078fbb8f6598574d04ece9ecccd55e97a4971ac346712cd3fc83f`.

````markdown
- [ ] Prove same-backend multi-GPU and GPU↔CPU compositions separately before
  enabling each; qualify mixed CUDA/ROCm/Vulkan only with its own evidence.
````

<a id="cli-device-routing-l173"></a>

#### open · [L173–174](../../plans/cli-device-routing/TODO.md#L173-L174)

Heading: `CLI and device routing > Uncommitted > Proposed unit 2 — local layer spill with real state`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L173-L174@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `c02509b9a6106cbf022efecb8ba6935e651b57fa34d15d98c0fae38a271d39c6`.

````markdown
- [ ] Verify full-logit/complete-state following-token parity, cancellation,
  partial preparation, transfer failure, unknown work and complete retirement.
````

<a id="cli-device-routing-l175"></a>

#### open · [L175–176](../../plans/cli-device-routing/TODO.md#L175-L176)

Heading: `CLI and device routing > Uncommitted > Proposed unit 2 — local layer spill with real state`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L175-L176@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `b967b3fc58abb2e7a9d345fd663eedc506733e00340746fbcec5850bb00178a0`.

````markdown
- [ ] Compare full CPU and each feasible single/split route with complete cold/
  warm request costs; show capacity enabling separately from speedup.
````

<a id="cli-device-routing-l182"></a>

#### open · [L182–183](../../plans/cli-device-routing/TODO.md#L182-L183)

Heading: `CLI and device routing > Uncommitted > Proposed unit 3 — qualified roles and explicit peers`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L182-L183@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `b61b20f431c07bfd4f3abc0df6613278e2e42b24f8e6047a898c4eaf9ed4c676`.

````markdown
- [ ] Introduce semantic stage-role constraints without model-family CLI routing
  branches; preserve full-run forced-backend policy and common allowlist.
````

<a id="cli-device-routing-l184"></a>

#### open · [L184–185](../../plans/cli-device-routing/TODO.md#L184-L185)

Heading: `CLI and device routing > Uncommitted > Proposed unit 3 — qualified roles and explicit peers`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L184-L185@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `55a91cacb6a641d5e3368fc1fcfc0190db8a589d7936fe0b4d1a1647cb03634f`.

````markdown
- [ ] Qualify actual trained draft/MTP source and recursive state semantics;
  distinguish native MTP depth from supported proposal length.
````

<a id="cli-device-routing-l186"></a>

#### open · [L186–188](../../plans/cli-device-routing/TODO.md#L186-L188)

Heading: `CLI and device routing > Uncommitted > Proposed unit 3 — qualified roles and explicit peers`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L186-L188@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `92606a85bfd12966f56ae23c183ffeee5052d6a1fbd8b77f8cf2fed83b6c4847`.

````markdown
- [ ] Add an NPU provider only through existing region/state/lifetime boundaries;
  prove end-to-end benefit versus available GPU draft and no-speculation routes,
  including transfer/verification/rejection/shared-domain costs.
````

<a id="cli-device-routing-l189"></a>

#### open · [L189–190](../../plans/cli-device-routing/TODO.md#L189-L190)

Heading: `CLI and device routing > Uncommitted > Proposed unit 3 — qualified roles and explicit peers`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L189-L190@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `b72b2ec367e4adb25514b8a0c22ad464133523bb117d219d9a065b375d965c64`.

````markdown
- [ ] Resolve explicitly enrolled peer aliases to authenticated identities and
  capability/topology epochs; standalone auto performs no remote discovery.
````

<a id="cli-device-routing-l191"></a>

#### open · [L191–192](../../plans/cli-device-routing/TODO.md#L191-L192)

Heading: `CLI and device routing > Uncommitted > Proposed unit 3 — qualified roles and explicit peers`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L191-L192@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `090ae3fce461d2658f0809aca06714c969c14c70b91ac5a7f8092bcb12bfd996`.

````markdown
- [ ] Coordinate bounded all-or-none holds through original per-host ledgers;
  implement authenticated transfers and exact state shard publication/fencing.
````

<a id="cli-device-routing-l193"></a>

#### open · [L193–194](../../plans/cli-device-routing/TODO.md#L193-L194)

Heading: `CLI and device routing > Uncommitted > Proposed unit 3 — qualified roles and explicit peers`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L193-L194@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `13c0bcbc8538beeb6a46778543b969451ce9f2b9c5ac1ea3a1c250989fcb40af`.

````markdown
- [ ] Qualify first two-host contiguous layer/local-state composition on approved
  sources/hosts; report directional network/copy/crypto/queue costs and loss.
````

<a id="cli-device-routing-l195"></a>

#### open · [L195–196](../../plans/cli-device-routing/TODO.md#L195-L196)

Heading: `CLI and device routing > Uncommitted > Proposed unit 3 — qualified roles and explicit peers`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L195-L196@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `79650604c9a83a2efd90fe35048275c562fa0ada63fe959841b3b905151db90c`.

````markdown
- [ ] Keep remote inference APIs that hide region/state contracts outside native
  device routing; no SSH/shell or implicit artifact downloads for placement.
````

<a id="cli-device-routing-l204"></a>

#### open · [L204–206](../../plans/cli-device-routing/TODO.md#L204-L206)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L204-L206@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `ea46b8a75d8042db4386eea0bcd46580f73f4aaa8dd5cd3c08165e25e9c1456b`.

````markdown
- [ ] **Syntax equivalence:** canonical/alias/repeated flags normalize identically;
  conflicting backend, mixed `auto`, empty device, unknown host/ID and CPU before
  GPU refuse. Unsupported backend is reported without model preparation.
````

<a id="cli-device-routing-l207"></a>

#### open · [L207–209](../../plans/cli-device-routing/TODO.md#L207-L209)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L207-L209@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `ca2b9b0425b03bcaf3373c0dae6369488f5f2d32d3f656b0367b7914b117b82f`.

````markdown
- [ ] **Default config:** present/missing/unreadable/invalid default; absolute,
  relative and unset XDG base; valid/absent HOME fallback; no cwd/parent search.
  Explicit model/config/graph skips default overlay; help/version do not load it.
````

<a id="cli-device-routing-l210"></a>

#### open · [L210–213](../../plans/cli-device-routing/TODO.md#L210-L213)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L210-L213@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `1e62e896062abf6e817e947d8dcdd7d0798521347f6881d9d374f664e29afd60`.

````markdown
- [ ] **Config discrimination:** YAML suffix, extensionless explicit --config,
  --model, odd model filename, malformed/unsupported schema, multiple positional
  selectors and old model-JSON --config produce the specified result; no model/
  config fallback, automatic pull, initialization or side effects.
````

<a id="cli-device-routing-l214"></a>

#### open · [L214–217](../../plans/cli-device-routing/TODO.md#L214-L217)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L214-L217@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `78fa62123469263b1c60a925f0463597c3ce46d8a77bc120578654880c32babc`.

````markdown
- [ ] **Config/CLI parity:** direct YAML and flags yield equal typed requests;
  graph YAML and declarations share one instance table/compiler. Source metadata,
  tokenizer, prompt/context/output/raw/teacher/prefill/audit/numerical/speculation/
  role/device/diagnostic fields and semantic validation round-trip equally.
````

<a id="cli-device-routing-l218"></a>

#### open · [L218–221](../../plans/cli-device-routing/TODO.md#L218-L221)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L218-L221@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `62976151b1bdf1885fefc4ae2aeff99f182e8aebbbab06a15b4d3c47e44f3905`.

````markdown
- [ ] **Precedence/provenance:** CLI scalar wins, alias equals canonical backend,
  default only fills absent values; false/zero/empty/auto are preserved and
  validated. Repeated CLI devices replace config list as a whole; graph list
  replacement/clear detects dangling references and duplicate IDs.
````

<a id="cli-device-routing-l222"></a>

#### open · [L222–224](../../plans/cli-device-routing/TODO.md#L222-L224)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L222-L224@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `76b5d6b59b518787bc7884328659785235d37d26c00cd31a4b90760419216975`.

````markdown
- [ ] **Path bases:** config-origin source/sidecar/audit paths use selected config
  directory; CLI replacements use invocation cwd. Moving cwd, config symlinks and
  same relative spelling with different origins cannot silently retarget fields.
````

<a id="cli-device-routing-l225"></a>

#### open · [L225–228](../../plans/cli-device-routing/TODO.md#L225-L228)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L225-L228@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `6f931d1961be9efbe7b9ecd585e916a5e2d5f5298b07df2ce7b40ccc04d9e82e`.

````markdown
- [ ] **Config safety/entrypoint:** unknown/duplicate fields, expansion/depth/file
  limits, custom tags/includes/interpolation, missing/ambiguous entrypoint,
  contradictory final backend/device policy and legacy model-config spelling
  refuse before execution. Selected graph runs once; declared routes do not bind.
````

<a id="cli-device-routing-l229"></a>

#### open · [L229–231](../../plans/cli-device-routing/TODO.md#L229-L231)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L229-L231@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `0559c767eb61c26dac4263bd71b02e305d0db21ea311e472a3d0af410b7937a1`.

````markdown
- [ ] **Identity:** CUDA/Vulkan aliases of one card count once; AMD discrete and
  NVIDIA integrated classifications cannot invert vendor routing. Snapshot
  indices never survive reorder/restart without physical revalidation.
````

<a id="cli-device-routing-l232"></a>

#### open · [L232–234](../../plans/cli-device-routing/TODO.md#L232-L234)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L232-L234@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `bba73d3bb4d1d9e3134db50d246f09a0ed7e7fd8b13d0c2a6ff8ddb7fe09e15f`.

````markdown
- [ ] **Forced scope:** CUDA+AMD rejects before allocation/compiler activity;
  forced GPU+CPU spill or incompatible NPU draft rejects; forced CPU runs every
  requested neural stage on CPU or refuses a missing capability.
````

<a id="cli-device-routing-l235"></a>

#### open · [L235–237](../../plans/cli-device-routing/TODO.md#L235-L237)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L235-L237@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `d1474cb10f0cbe37a36090164f000d3d7625f00904e51141530325f6b362a501`.

````markdown
- [ ] **Full fit:** both vendors fit→NVIDIA; NVIDIA insufficient/occupied and AMD
  fits→AMD; later listed GPU full fit precedes first-card split; GPU-only explicit
  list cannot reach CPU. NVIDIA unavailable/fenced is a precise exclusion.
````

<a id="cli-device-routing-l238"></a>

#### open · [L238–240](../../plans/cli-device-routing/TODO.md#L238-L240)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L238-L240@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `ac40c7920975ef66b29a635cf1a040b6bfe3fdd6733629bcbae5d0f8b1bd012c`.

````markdown
- [ ] **CPU fallback:** no GPUs, missing GPU build, unqualified GPU model and no
  qualified split each reach CPU only if allowed, qualified and currently fitting.
  An unqualified model everywhere gives no invented inference route.
````

<a id="cli-device-routing-l241"></a>

#### open · [L241–243](../../plans/cli-device-routing/TODO.md#L241-L243)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L241-L243@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `f5c8bf4fa6c47978af2f4762f85b6001ab952613fe3553dfc8ed00f9fad6ef53`.

````markdown
- [ ] **Current pressure:** large total VRAM with low free bytes does not fit;
  cross-process occupancy, OS/cgroup pressure, missing/stale counters and
  fragmentation produce conservative proposals and bounded allocation failures.
````

<a id="cli-device-routing-l244"></a>

#### open · [L244–246](../../plans/cli-device-routing/TODO.md#L244-L246)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L244-L246@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `3ee3eb90d847a06615c94ac27555e75cbd59881f5391280837b733b4eb7e1b7d`.

````markdown
- [ ] **Accounting:** original leases plus actual backing plus unbacked promises;
  prepaid envelope and native partition; idle retained pools; authentic reuse;
  unknown/captured readers. No duplicate charge/free credit or cap enlargement.
````

<a id="cli-device-routing-l247"></a>

#### open · [L247–248](../../plans/cli-device-routing/TODO.md#L247-L248)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L247-L248@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `86aef09127392cc7862cd9e9c15f43f4ff928d8ce0a965fa05c84852b8c612d3`.

````markdown
- [ ] **UMA:** CPU/GPU/NPU alias one memory domain; cross-device views/transfers
  retain proper fences and charge physical backing once. Headroom applied once.
````

<a id="cli-device-routing-l249"></a>

#### open · [L249–251](../../plans/cli-device-routing/TODO.md#L249-L251)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L249-L251@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `2b608c336fcec8e3a72f8b1cf56f6673916f089ab7717f23d03e3a670940ead4`.

````markdown
- [ ] **Complete peaks:** weights fit but prefill/KV/output/compiler/native/capture/
  draft costs do not; reject. Disjoint phase maxima are not blindly added; actual
  overlapped lifetimes, COW, transfer and old/new state are not omitted.
````

<a id="cli-device-routing-l252"></a>

#### open · [L252–255](../../plans/cli-device-routing/TODO.md#L252-L255)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L252-L255@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `546e57ae1b0a3880e37523181231bc39111c0a8219b72bff6804e958fb75069d`.

````markdown
- [ ] **Headroom:** percentage/bytes/default/floor/overflow validation; resolved
  amount shown; reducing margin never raises a physical/operator ceiling or
  waives source/shape/native qualification. Bare-run bootstrap records a safe
  default once; later live-free changes never resize the registered ceiling.
````

<a id="cli-device-routing-l256"></a>

#### open · [L256–258](../../plans/cli-device-routing/TODO.md#L256-L258)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L256-L258@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `fa44ecde645893c5aea3e0c81b64174808e4e70fccd0897e26db84120f888f85`.

````markdown
- [ ] **TOCTOU:** external allocation and concurrent internal admission between
  proposal/admission/preparation; all-or-none holds and known-safe rollback;
  at most one fresh replan. Dry-run and cached plan never grant reservation.
````

<a id="cli-device-routing-l259"></a>

#### open · [L259–261](../../plans/cli-device-routing/TODO.md#L259-L261)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L259-L261@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `a1f2033963962c6525dda59c1d256e37bd2c6dbe1fd0025890d20e3f28bf3ab5`.

````markdown
- [ ] **Unsafe preparation/flight:** unknown module load/unload, transfer/capture/
  launch retains/fences original owners; no alternate-backend replay, partial
  root/output publication or reuse based on cancellation/timeout alone.
````

<a id="cli-device-routing-l262"></a>

#### open · [L262–264](../../plans/cli-device-routing/TODO.md#L262-L264)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L262-L264@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `45bde5506e18db22931b1c68b2a8d168c99faa563d7575ee9f7812f90c1f41d5`.

````markdown
- [ ] **Split feasibility:** largest legal prefix leaves a feasible remainder;
  indivisible tail/tied embedding/head cases; GPU→GPU→CPU exact state and
  activation transfer. Unqualified mixed numerics refuse rather than retag.
````

<a id="cli-device-routing-l265"></a>

#### open · [L265–267](../../plans/cli-device-routing/TODO.md#L265-L267)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L265-L267@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `dc4d522e40f2d774e6c36b65d81e7f597b7725a44398d6f0556a9c430b5af156`.

````markdown
- [ ] **Runtime equivalence:** real full affected logits/tensors, every required
  state domain, natural following-token continuation and task outputs match the
  selected numerical contract for each admitted source/profile/composition.
````

<a id="cli-device-routing-l268"></a>

#### open · [L268–270](../../plans/cli-device-routing/TODO.md#L268-L270)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L268-L270@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `71f5b81fb1e45185019171dbcad366b4c560609e95fa69386ad0aeb943784501`.

````markdown
- [ ] **Fairness/lifetime:** overlapping requests/models preserve independent
  roots, original fair progress and complete reader retirement under pressure;
  successful cleanup cannot hide an earlier semantic failure.
````

<a id="cli-device-routing-l271"></a>

#### open · [L271–274](../../plans/cli-device-routing/TODO.md#L271-L274)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L271-L274@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `f8058a28b8585f490adbd8a72f92b5921c556f41d0426779f805d1149e98922b`.

````markdown
- [ ] **Diagnostics:** list/dry-run leave model, GPU context/compiler, resource
  ledger, cache and network unchanged; report unknown rather than probe by
  execution. Explain identifies each rejection and final route without corrupting
  output or exposing prompts/secrets.
````

<a id="cli-device-routing-l275"></a>

#### open · [L275–279](../../plans/cli-device-routing/TODO.md#L275-L279)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L275-L279@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `6ee531b1d4bcdc785325c3728317108a55b672c9fa8b523e346382706480b3b0`.

````markdown
- [ ] **NPU role, future:** qualified NPU still loses when transfer/verification/
  rejection/queue costs dominate; wins only with applicable full-request benefit.
  Stale evidence rejects automatic speed claim; default auto selects ordinary
  decode without proven benefit, optional draft-device does not force drafting,
  and required MTP/draft/source features are not omitted.
````

<a id="cli-device-routing-l280"></a>

#### open · [L280–282](../../plans/cli-device-routing/TODO.md#L280-L282)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L280-L282@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `5468bf3f255e1a8be21e85a0bda9cf096514f46d4f316b86228cdf67e4a68449`.

````markdown
- [ ] **Remote, future:** unknown enrollment, identity/epoch mismatch, denied
  source/data grant, partial holds, slow/relayed link, late Ready, lost ACK and
  uncertain DMA preserve original authority/leases; no implied discovery or HA.
````

<a id="cli-device-routing-l283"></a>

#### open · [L283–285](../../plans/cli-device-routing/TODO.md#L283-L285)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L283-L285@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `e72d8c5ebf887276df0ac438fdcc277678c4eff91d4c3ce77089dc2cc18cabf4`.

````markdown
- [ ] **Performance:** authorized paired whole-request comparisons, cold/hot
  preparation, TTFT/ITL and total latency, memory/IO/transfers and fair load;
  compare against CPU/single-GPU baseline and include slower-split negative cases.
````

<a id="cli-device-routing-l286"></a>

#### open · [L286–288](../../plans/cli-device-routing/TODO.md#L286-L288)

Heading: `CLI and device routing > Tests and verification`.
Identity: `.agents/plans/cli-device-routing/TODO.md:L286-L288@db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162`.
Block SHA256: `8835af787021ab230a785ee1a2ff532d8f33f4657904fee2d12a566c07707ef0`.

````markdown
- [ ] **Final assembled gate:** after applicable unit qualification, run
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`.
  This design-only draft does not run the gate or any GPU/model tests.
````

### cpu-gpu-optimization — 55 noncompleted occurrences

<a id="cpu-gpu-optimization-l19"></a>

#### ~ · [L19–22](../../plans/cpu-gpu-optimization/TODO.md#L19-L22)

Heading: `Cpu gpu optimization > Uncommitted > Current correctness-first order — 26 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L19-L22@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `f6aa74c620be89c82d1a4981355c02d03625d932aa0f4cda7afd1aaac2d27b78`.

````markdown
- [~] [Shared vLLM/llama.cpp correctness policy](../numerics-and-context/CORRECTNESS_POLICY.md)
  across all models; .8B initial/cached47–50, then genuine long contexts and
  CPU/CUDA/ROCm/Vulkan. All agents GPT-6.1 Sol xhigh. Earlier B4/C16/27B order
  below is historical; no new performance or27B capture before correctness.
````

<a id="cpu-gpu-optimization-l23"></a>

#### open · [L23–26](../../plans/cpu-gpu-optimization/TODO.md#L23-L26)

Heading: `Cpu gpu optimization > Uncommitted > Current correctness-first order — 26 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L23-L26@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `a8a773dee877a48e37a239a2da5c11202055bf86ad751b61c98fb73a68062736`.

````markdown
- [ ] Preserve case5 FAIL101 and quarantine. Original default vLLM N1/B4 also
  has the same six QKVZ BF16 differences at frontier47 with identical inputs,
  input RMS and48 prior planes. Width-specific original agreement and model
  accuracy are distinct; no arithmetic substitution or failed-run relabel.
````

<a id="cpu-gpu-optimization-l30"></a>

#### open · [L30–34](../../plans/cpu-gpu-optimization/TODO.md#L30-L34)

Heading: `Cpu gpu optimization > Uncommitted > Current B4 takeover — 25–26 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L30-L34@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `e5af7d7aa389621adc066963cf38e7f9b601eb200fbb8b80d0a63f9f17ee29f3`.

````markdown
- [ ] Current B4 → fresh optimized matched C16 (.8B +2B) → x2 original27B,
  latest owner order; no C2. Base `d0fff261`, all246 signatures verified;
  README unchanged. All current lanes: GPT-6.1 Sol xhigh, owner override.
  [Owned run](../../runs/b4-takeover-codex-20261025/TODO.md).
  [Final evidence index](../../runs/b4-takeover-codex-20261025/FINAL_REPORT.md).
````

<a id="cpu-gpu-optimization-l217"></a>

#### ~ · [L217–230](../../plans/cpu-gpu-optimization/TODO.md#L217-L230)

Heading: `Cpu gpu optimization > Uncommitted > Current source checkpoint — 24 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L217-L230@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `5b45126b0dba0712fefabe8bb625939b7bd5169d0ac56030923776f1aa855010`.

````markdown
- [~] User extends both-model sequence to C16, thenC32 andC64; NO C2. Implement
  genuine populations and honest active8/B4 observations. Latest user direction:
  resume all parallel tracks now. The78 signed logical commits are complete;
  C16 correction, Qwen27B, Gemma4, Vulkan, HIP and shared queue have independent
  Pi lanes. ROOT coordinates builds/physical ownership and quiet timings.
  Default pending count unlimited, optional configured cap, actual HOST
  admission unchanged. C32/C64 follow queue acceptance. C16 correction and
  grouped controls run separately from product source; old drafts preserved.
  Audit/store latest and best verified matching historical C1/C4/C8 plus new
  bands, with source/profile and stock arm from SAMEattempt, no cherry-picking.
- Historical source20 first-use refusal remains failed:446→448 attempts after
  readiness. Source21 fixes the two observed state recipes and passes actual
  both-model API/HTTP/C1/C4/C8 with cold/whole compiler0; see Committed above.
  Preserve original failed20/source15 artifacts; no warmed substitution.
````

<a id="cpu-gpu-optimization-l231"></a>

#### ~ · [L231–340](../../plans/cpu-gpu-optimization/TODO.md#L231-L340)

Heading: `Cpu gpu optimization > Uncommitted > Current source checkpoint — 24 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L231-L340@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `c874e22a1289e33fcf8a367bb03eb294d9267cc8cbc723fb8f4e8009048bfe6e`.

````markdown
- [~] C16 both executed on frozen21. Clean2B: ZG78.7859tok/s,1.62466s;
  vLLM95.9974tok/s,1.33337s. Each16offers/16completed/128tokens/0errors,
  original semantic/source/owner closure healthy, actual active8/B4 and compiler0.
  Rate refusal is a measured performance loss, not numerical/cleanup failure.
  First.8B timing overlaps audit scan; one repeat105.7331vs96.6544tok/s has an
  extra sampler and remains diagnostic, not quiet/best evidence. No third rerun.
  [Scope review](../../runs/resume-all-parallel-source-supervision-sol-20261024/C16_FINAL_SCOPE_REVIEW.json).
Historical source13 checkpoint (superseded by source20 above): ROOT adopted models118 plus source-bound2B
cachedB4 as one guarded173path unit (172effective/46new), fresh mtimes and
README/lock preserved. Full1629/selected492 source equality verified MAIN/ISO/
donor; optimized joint CLI/N1/B4 release13 CLOSED0, both-family actual nonlaunch
admission CLOSED0. Current23 CLOSED1: original19 and genuine2B trajectory/export/
healthyretirement pass; first new2B cached-conv fault fails an unchanged compound
retention assertion without its actual snapshot, last2 faults unrun. Source-first
fixture diagnosis continues; original SAMEPRIOR comparison of the healthy
closed trajectory now passes all15transitions/30plain+observed calls,8112sections/
576unrounded sums/fullstate/fullV/natural bits. Source13 only; serving/benchmarks
stay held until fault qualification. Combined strict,20sharedCPU,
5ABI-source controls, format and authentic2B inventory/reopen pass; excluded
vendor-test Cargo refusals stay explicit. Source-derived provider upper
4,767,472,640B within4608MiB and HOST4,080,276,698B within4GiB; fullhead remains
resident and only cachedB4 is admitted for2B. Its actual successful trajectory
fits this finite admission and retires healthy; whole fault/math/serving/speed
qualification remains open. Diagnostic14 only prints actual fault snapshots,
retains all assertions and changes no production source. Source11/12 below
remains the last accepted measured checkpoint.

Earlier numerical checkpoint24Oct: fixture-only source8/412paths exported all15
genuine transitions/35events/1,648,529,280B, with healthy source/native retirement
and final host/device0. Original same-prior reference3 now PASS: all15transitions/
30plain+observed calls, fullstate/fullV/24sums/natural bits and all controls exact.
Loaded320roles unchanged; final source/engine/process/container/group/EOF closure
healthy. Explicit state-initialized compositional proof, not a separately generated
original public trajectory. Earlier adapter failures and cross-width N1 diagnostic
remain preserved. Fresh API8 PASS24offers192tokens/provider, wave16actualB4
flights/64rows; ragged N1. Separate event-driven HTTPv2 now PASS: all5 genuine
N1prefills, A→mixed cachedB4, completedA→replacementE/refilledB4,5publications/
20rows/36natural tokens exact against actual same-cap original stock outputs.
All original source/owners close; v1's metadata-only adapter failure preserved.
Finite A4+BCD/E8 correctness scope, not a timing or general cancellation grant.
Quiet C1/C4
CLOSED0: ZG66.7151/17.5611 versus vLLM37.3407/49.0326tok/s. All32offers256tokens,
four original owner/source closures exact; singlepass preliminary. C4 regresses,
C8 hard-held. Separate direct diagnostic4 CLOSED0:185compiler trips/1480spans,
all8offers64tokens/source/owner/clock closure exact. In its3.72236s wave,79first-use
trips consume3.10167–3.10218s parent union; NVRTC2.18533–2.18585s and module load
.84863–.84915s are nested. Later decode/second wave have0compiler spans. This is
instrumented attribution, not a fraction/updated result of the quiet benchmark.
Shared admitted exact-recipe readiness now passes14distinct CPU contracts and
intended all-target strict configurations; independent Sol review finds no
must-fix. ROOT merged52guarded paths with fresh mtimes into MAIN/ISO source9
(420selected/1583complete build files); rootREADME unchanged. Optimized joint
CLI/N1/B4 release9 CLOSED0. Its actual18-case physical batch CLOSED1:16 cases
refused before inference at the compiler ABI guard; two N1 matrices first passed
their301 standalone finite checks, then correctly refused installing a fresh
compiler policy on that already-used actor. All18 original owner/group/one-reap/
fullEOF closures healthy. Installed CubeCL deliberately sets
`CompiledKernel.io=None` for handwritten kernels (conservative access analysis),
while the private PinnedKernel factory already enforces physical ABI<=32. The
new policy conflated optional access metadata with the actual ABI. Reviewed
four-path successor10 now merged MAIN/ISO; nine contracts and intended full
integration strict pass. ROOT verified all1583build files MAIN/frozenWORK plus
420selected MAIN/ISO and unchanged rootREADME; optimized joint build88097 CLOSED0.
The unchanged301-case qualifier now has its own original-owned fresh process;
reviewed19-case adapter requires its success/closure before all18model scopes.
Preserve release9/failures. Actual19physics97239 CLOSED1: standalone301, both
whole numerical matrices and all4B4 scopes PASS;12N1faults stop at stale fixture
accounting after expected retained/fenced/no-publication outcomes. All19 original
owner/group/one-reap/fullEOF closures healthy. N1 deliberately admits4608MiB
provider even for.8B; actual512MiB backend adds to5GiB, rather than the old
provider-only expectation. Reviewed fixture-only successor11 derives exact sum
from actual healthy leases, checks bothDEVICEowners/classes plus originalHOST
source/backend owners before/after. One guarded file applied MAIN/ISO/WORK;
optimized affected N1build93562 CLOSED0, unchanged CLI10/B4/libraries reused
byte-exactly. Fresh19batch25323 CLOSED0: all19 source/state/fault/lifetime scopes
and original owners pass; all12 exact HOST/DEVICE fault ledgers emitted. API94011
CLOSED0:24offers192tokens/provider exact, original sources/owners close. Genuine
HTTP70470 CLOSED0:5N1prefills→mixedB4/refill,5offers36natural tokens exact;
same-cap actual stock reuse explicit, compiler attempts/keysource/PTX deltas0.
Quiet current11 C1/C4 unit24047 CLOSED0: ZG67.7996/101.984 versus
vLLM37.4208/49.023tok/s; C8 gate passed, unit39371 CLOSED0:100.46vs61.9783.
All48offers384tokens/source/sixoriginalowners exact, compilerattempt/keysource/
PTX delta0; actualC4/C8 each16B4flights64rows,C8peak8. Single-pass preliminary,
pinned0.31 eager/BF16 recipe, no full parity/confidence grant. Fullgate17 stops
only at owned fixture formatting; rustfmt-only source12 applied MAIN/ISO/WORK,
CLI11/production unchanged. Fullgate18session56242 CLOSED0:1756Rust/0failed/
89explicitignored/271suites/allscriptchecks; full1583MAIN/WORK and420selectedISO/
README unchanged after gate. Current.8B checkpoint accepted within its finite
source/profile/singlepass scope. No budget or
production change from fixture correction/formatting. Isolated2B source-bound
cachedB4 family specialization now active; N1prefill/partialfallback retained,
authentic2B RMS/projection/catalog/reference and exact fit required.
Original2B B4 discovery passes all8calls/all24boundaries/all42state domains.
Capture1's exact fullimage layout refusal64/48absent FApackets stays failed.
Reviewed capture-only catalog pins54discovery+twoactual late-reflected fullimages;
fresh capture2 CLOSED0, all selected arguments present/driverfail0/all8scopes/
fullV/natural4×8 plain-observed exact. ROOT verifies source320/632/fullpayload,
all64original package members and both original process/container/group/reap/
fullEOF closures. No image hash normalization. Expected-only original reference
and complete native metadata do not grant ZG math/fit/serving or N1→B4
continuation; the shared2B implementation and independent SAMEPRIOR proof remain
open. Original failed capture is unchanged.
No new physical/performance grant. Actual detached
backend384MiB HOST/512MiB DEVICE uses the original domains; model quotas remain
unchanged. Exact pooled RMS definitions replace duplicated source metadata;
anchored bounds fit, fresh live inventory remains required. Live same-actor
compiler counters and actual retained lease charges are exposed for qualification.
CLI6 measurements above are historical; current CLI10/11 results and fullgate18
are the accepted checkpoint. Run the next complete gate last on the assembled
2B/model contribution; gate16 covers preceding316 only.
````

<a id="cpu-gpu-optimization-l457"></a>

#### open · [L457–463](../../plans/cpu-gpu-optimization/TODO.md#L457-L463)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L457-L463@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `4403d27ead646959dcbbf89e4efbdd0f71410012cf7750e029904427c40ae356`.

````markdown
- [ ] Shared compiler lifetime/readiness: fix checked all-exit raw NVRTC cleanup,
  retain/fence uncertain destruction and partial module loads; establish actual
  context-lifetime host/device ownership before eager exact-recipe preparation.
  No global DryRun, dummy roots or unpriced cache. Existing model-local128MiB
  loader is host scratch, not a persistent CUDA context charge. Numeric ceilings/
  complete fit remain unresolved; Sol HIGH implements the contained foundation.
  [Astra architecture](../../runs/architecture-astra-compiler-readiness-20261024/DESIGN.md).
````

<a id="cpu-gpu-optimization-l464"></a>

#### open · [L464–599](../../plans/cpu-gpu-optimization/TODO.md#L464-L599)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L464-L599@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `67202f53546827dc3eadf0201a10b785cbe50b22cf7feeb69d5e434448c52f4b`.

````markdown
- [ ] After current C8/refill and source113 qualification: shared prepared matrix/
  compact IO reuse, temporal work, optimized eligible GEMM/attention/GDN, pages/
  full-domain prefix/COW, admitted prefetch and graph replay. Preserve numerical
  profiles; source113 CPU-order dot is not TensorCore/BLAS parity. Family and new
  model gaps are queued in model-execution; no hardware eligibility by name.

Gemma44 diagnostic source now integrated MAIN/ISO343 after binder316 gate16.
Frozen MAIN314+44 actual x2 HIP matrix PASS7cases/55completed/52commits/all35/
fullV/all15KV, zero mismatches; fresh unknown0roots/full16GiB fence and original
source/process closures PASS. One1MiB synchronous source page remains a
correctness baseline. New source-neutral physical owner and simultaneous
prepared sets need their own actual checks; no Gemma serving/performance grant.
Current HIP actor still refuses a second live native set; bounded lifecycle
extension remains in the next shared native-consumer unit.
The [Astra follow-up](../../runs/architecture-astra-shared-native-20261023/DESIGN.md)
selects nonzero, never-reused logical streams through every actor cache/IO/driver
lookup, immutable source-service Client views, explicit cold32GiBprovider/
64GiBdevice/32GiBhost with two12GiB quotas and conservative compiler retention.
These are provisional design ceilings; complete cost/actual fit is unproved.
The [Astra residency/set-key follow-up](../../runs/architecture-astra-residency-and-set-key-20261023/DESIGN.md)
accepts a private key carrying logical set plus the FULL original KernelId,
bounded full-Eq actual-driver routing and immutable SingleCompat/Scoped modes.
Sol HIGH implements separately from single-source Qwen numerics. Gemma's next
parallel unit retains repeated projection weights, widens exact row launches,
skips optional diagnostics before readback construction and uses original private
feedback/publication. Exact source selection/complete cold peaks precede fixed
profile admission; old16GiB proof remains distinct. Tied head768MiB plus2MiB
scratch is design arithmetic only; PLE full table4480MiB is not needed to stop
repeated projection uploads. No measured speedup or new fit claim.

Latest accepted shared-head extraction:290selected source paths, real exact
issuer/member/expert-range binding; both models' complete physical/state/fault/
API checks PASS. Full isolated gate13 PASS1737Rust/0failed/88ignored/269suites/
allscriptchecks. Shared implementation owns operand/residency/native invocation;
family adapter owns roles/tie/SSA/publication. Timings below remain the earlier
storage/rings release2. Native14+core2 and opt-in Qwen full4 convolution-state
transport now accepted301selected paths:198core/86Qwen/549GPU/8nativeartifact
tests and strict complete-feature checks PASS. Required release both models'
complete physical/state/fault/API scopes plus forced-streamed2B faults PASS.
Full isolated gate14 LAST PASS1747Rust/0failed/88ignored/269suites/allscriptchecks;
source/rootREADME unchanged. Actual x2 native3 +affected HIP/Vulkan transport12
also PASS, scoped to four native kernels and BF16 transport. No Gemma/Qwen27B
inference or fresh performance grant.

Successor shared CPU arithmetic now reviewed/integrated302selected paths:
explicit original normal-domain exp profile closes authentic H48 cached producer
and all786432state words bitexact, preserving legacy negative control. Actual
full64 replay3 now passes two independent sessions, each one INITIAL token ID11
then one cached token ID353,
all1040records/240896000float words/fullV/all112domains/ALL4Conv frames bitexact;
source/owned closures and finalhost0 pass. Four qualification-example corrections
are integrated MAIN/ISO. Natural/long/GPU/native/serving/performance remain open.
Existing native301 qualification/timings stay scoped to their immutable sources.

Prepared recipe successor now reviewed/integrated314selected source paths.
Sol HIGH implements; Astra HIGH architecture-only review accepts/no must-fix.
Same record/ForwardPlan: shared topology/source semantics, genuine per-request
operation/provenance and original physical/state authority. Exact source cache40
retains4MiB/shape, conservatively160MiB; request record bound512KiB replaces4MiB.
Warm binding runs outside cacheMutex. Core201/Qwen87/GPU549/default+provider
strict PASS; original joint release and both models' complete real source/state/
fault/API scopes plus forced-streamed2B faults PASS. Fresh .8B C1/C4/C8:
ZG64.8742/63.5759/60.7789 vsstock37.9990/49.3562/62.3861tok/s
(+70.73%/+28.81%/−2.58%). All48offers384tokens exact/all6owner/source/group/
container/EOF closures healthy. Singlepass/no confidence claim; C8 superiority
still unmet. Source/site/native checks and wrapper allocations remain; no
zero-allocation claim. Full assembled gate15 LAST PASS1752Rust/0failed/89ignored/
269suites/allscriptchecks after the four qualifier example corrections;
314selected sourcepaths/rootREADME unchanged. Frozen release/timing retain
their original pre-qualifier snapshot, with unchanged CLI/library bytes.

Shared binder unit Sol8 now reviewed/integrated MAIN/ISO316:
existing ColdModel owns bounded40 immutable analyses; warm actual-call/operator/
producer checks run outside cacheMutex before scope/attempt. Original source/
member/SSA/role/head checks stay in cold analysis, downstream physical/state/
native authority remains late. Retained first-record witness +site metadata
guarded inside512KiB/entry, full20MiB+64KiB host inventory. GPU551/0failed/
54ignored/provider+default strict PASS. Original joint release and both models'
complete real source/state/fault/API scopes plus forced-streamed2B faults PASS.
Fresh .8B C1/C4/C8 ZG67.2018/63.8830/60.9713 vsstock37.6230/50.4311/61.6415
tok/s (+78.62%/+26.67%/−1.09%). All48offers384tokens exact; six original
source/owner/group/container/full-EOF closures healthy. Singlepass only; C8
superiority remains unmet. Final assembled isolated gate16 PASS1752Rust/0failed/
89ignored/269suites/allscriptchecks;316sourcepaths/rootREADME unchanged. Source
and performance claims remain scoped to this frozen release. B4 loaded/inventory
changes reconcile against this donor. Genuine batching precedes further N1
microtuning.

Previous timed frozen storage/rings release2:287 MAIN/ISO selected source paths;
strict6,193core+543GPU tests and standalone default/CUDA/HIP/Vulkan/all-provider
targets pass. Both actual301finite+40natural/fullV/all42state/all5fault matrices
and matchedAPI24offers192tokens perprovider pass. Explicit forced-streamed2B
mid-head unknown/cancel also verifies original transfer retention/quarantine.
Fresh .8B C1/C4/C8 ZG62.3484/60.7822/57.4249 vsstock38.0862/49.9539/61.8017
tok/s (+63.70%/+21.68%/−7.08%).2B C1 ZG43.3738 vsstock37.6126tok/s
(+15.32%): full1017118720B resident head, explicit native4608MiB vsstock
request4096MiB+KV512MiB/no harddevicecap. Host4096/device5120MiB unchanged.
All timing owners/source/group/EOF closures closed. Singlepass/no confidence
claim. Native remains N1; C8 superiority is unmet. Full isolated gate12 PASS
1734Rust/0failed/88ignored/266suites/allscriptchecks;287sourcepaths/rootREADME
unchanged. Genuine B4/B8 batching remains open.

Previous checkpoint: frozenpair9 shared prepared content keys, cold GDN/FA copy
definitions and immutable source-seal reuse assembled with portable21/Gemma13/
diagnostic6,260MAIN/ISO sourcepaths exact. Strict focused checks/joint release
and original301edge+40natural/all5fault/matchedAPI checks PASS. Device counts/
fault scalars unchanged; cold cohort metadata +1400B is priced. Quiet C1ZG61.5227/
stock38.5419(+59.63%),C4ZG62.4174/stock49.6312(+25.76%),C8ZG62.5204/
stock61.2769(+2.03%)tok/s. All48offers384tokens exact/all6 originalowner/source/
EOFclosures reconciled. Singlepass only; small C8 margin is not stable confidence.
Full isolated gate11 PASS1724Rust/0failed/88ignored/266suites/allscriptchecks,
260sourcepaths/rootREADME unchanged.2B actual301edges+40natural/all5faults and
matchedAPI24offers192tokens eachprovider PASS. Quiet2BC1 ZG0.5298/stock37.0680
tok/s (stock69.97x);16offers128tokens exact/both originalclosures reconciled.
Further2B bands held for shared storage/streaming repair. Source shows1.017GB/
970uploads+streamwaits perforward; full attribution/fit remains open. Parallel
pool/boundary audits and shared APIs linked above.

Previous source checkpoint: shared ordered-enqueue actor6 passes frozen pair8
original301edge+40natural/all5fault/API checks; actual dispatch/finite/fault
scalars equal pair7. Full isolated gate10 PASS1715Rust/0failed/88ignored/
266suites/allscriptchecks over220selectedchangedpaths, source/rootREADME
unchanged. Quiet pair8 C1ZG60.8262/stock38.2050(+59.21%), C4ZG60.9697/
stock50.6970(+20.26%), C8ZG60.4256/stock63.2563(−4.47%)tok/s. All48offers/
384tokens exact; ROOT checked all6 originalowner/source closures. Preliminary
singlepass, no confidence claim or2B promotion. Current actor CPU/GPU-envelope
diagnostic precedes next optimization; earlier pair7 remains historical.
Portable corrected transport12 actual HIP/Vulkan modes separately pass on x2;
arithmetic10 actual HIP/Vulkan modes pass exact finite CPU-profile boundaries
and original fault/ownership closure; reviewed21paths merged MAIN, combined
gate pending. Gemma13 source contracts also merged; whole-model GPU remains
unqualified. Actor8 diagnostic closed with64exactforwards/432steps/all owners,
whole device-clock envelope.615946s/nestedHead.099558s, hostforward.860439s.
258usercycle samples/0lost identify allocator scans, kernel-key hashing and
driver submission; no wall/GPU percentage attribution. Prepared immutable key
unit proceeds without math/cap changes; no2B timing.
````

<a id="cpu-gpu-optimization-l601"></a>

#### open · [L601–613](../../plans/cpu-gpu-optimization/TODO.md#L601-L613)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L601-L613@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `dc9cda5060dfdaad39bff56a1b59bc174167d9570765fde171a7702cea135ad0`.

````markdown
- [ ] **Parallel implementation, third merge — genuine BF16/HF request batching
  (owner directive,23Oct).** Develop the physical multi-request unit now; merge
  after current storage/residency/architecture fixes, before further N1 microtuning.
  Start with genuine canonical47 B4/B8 initial prefill (188/376 total tokens)
  through SAME independent roots into decode and correct partial-batch fallback;
  extend mixed frontiers through shared engine
  cohort/binding/workspace APIs. Reuse existing root-cohort publication, but
  implement compatible BF16 attention/GDN/projection lowering rather than
  enabling the older GGUF/F32 path. Preserve independent row states, natural
  feedback, fair work, cancellation/quarantine and current resource caps.
  Validate row permutations/full logits/all states and physical/API behavior,
  then matched C1/C4/C8; preserve the C1 fast path. Request batching is distinct
  from operation fusion; no speedup or superiority claim before measurement.
````

<a id="cpu-gpu-optimization-l614"></a>

#### open · [L614–617](../../plans/cpu-gpu-optimization/TODO.md#L614-L617)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L614-L617@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `1ccae60c9fe57c8e2578e4105f1d82a28b82e17034195f91b16eb15786f56dc1`.

````markdown
- [ ] Lift reusable source identity, bounded immutable reads, prepared bindings,
  actual-flight indices and resident matrix transport into shared components.
  Select qualified implementations using complete host/device live costs;
  a family adapter may restrict mathematics without duplicating common policy.
````

<a id="cpu-gpu-optimization-l618"></a>

#### open · [L618–624](../../plans/cpu-gpu-optimization/TODO.md#L618-L624)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L618-L624@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `5dfe2e9f35c5fa53c0f443cc4885d753bb207e3e76ffff90c33f8e74af205927`.

````markdown
- [ ] Shared resident-backing pool + immutable layout/access metadata + bounded
  transfer/frame retirement, from23Oct parallel source audits. Keep existing
  model traversal/publication and current root/lease freshness; price live,
  idle and unknown owners until actual retirement. Expose minimal common APIs
  through existing prepare/bind/submission/completion objects. CUDA/HIP pinned
  staging and Vulkan queue/mapping need their own physical proof; no inferred
  speedup or zero-copy claim. Follow the coherent units in the linked design.
````

<a id="cpu-gpu-optimization-l625"></a>

#### open · [L625–630](../../plans/cpu-gpu-optimization/TODO.md#L625-L630)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L625-L630@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `2f0854fe6f49d0b5c227375e90eeef4cbe2b4b21eec4db1f120804eb23dc2fc9`.

````markdown
- [ ] Demonstrate superior matched .8B C1 AND C4, then run C8. Only after all
  three wins, optimize/benchmark2B. Keep original quality/source/caps and report
  single-pass uncertainty. Latest prepared-binding release1 observes C1+78.62%/
  C4+26.67%/C8−1.09%; C8 superiority remains open.2B head residency is fixed;
  actual physical/API pass and C1+15.32% with the explicit4608MiB profile.
  Genuine batching precedes wider fresh timing bands; stable superiority is unproved.
````

<a id="cpu-gpu-optimization-l631"></a>

#### open · [L631–632](../../plans/cpu-gpu-optimization/TODO.md#L631-L632)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L631-L632@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `5ffb96c59903f1c8d280a818351d079d6e1cc99312846d3a9937b540071a294e`.

````markdown
- [ ] Merge Qwen27B/Gemma contained units as ready in parallel. Their later GPU
  qualification/benchmarks use pvl-x2's AMD gfx1151 through ROCm/HIP.
````

<a id="cpu-gpu-optimization-l633"></a>

#### open · [L633–638](../../plans/cpu-gpu-optimization/TODO.md#L633-L638)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L633-L638@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `dec21a494aee3b7423463956716304385e1d9ee17c39bff1a2fe118efd16c84b`.

````markdown
- [ ] Bring Vulkan implementations to CUDA parity independently; AMD uses HIP,
  not an implicit Vulkan/CPU substitute. Preserve one engine/physical service,
  independent roots, fair work and completion/quarantine through each provider.
  Read-only pvl-x2 inventory confirms96GiB GPU memory, existing ROCm7.2.4 and
  RADV/Vulkan. Availability/compilation is not numerical or performance parity.
  [Shared components and backend implementation units](BACKEND_PARITY.md).
````

<a id="cpu-gpu-optimization-l640"></a>

#### open · [L640–649](../../plans/cpu-gpu-optimization/TODO.md#L640-L649)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L640-L649@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `947583213ae8acc6e18ab6e5de3e69a9b26c87b51d56136f704e29d197566346`.

````markdown
- [ ] **OPT-01 matched short baseline:** after parity, established same-source
  C1/C2/C4/C8 profiles, original ABBA+BAAB/offered-load controls, real TTFT/true
  token ITL/E2E tails, correct committed/delivered SLO-goodput and failures.
  Separate unprofiled timing from CPU/driver/kernel/copy/IO diagnostics; pin
  actual binary/profile/caps/EOS rather than retagging old GGUF workloads.
  Quick .8B canonical wave8 C1/C4 baseline completed: ZG0.9557/0.9511 output
  tok/s vsstock38.5021/49.6408. Owner23Oct now requires fixing the gap and
  beating vLLM first;2B/full repeated/ragged timing deferred. C8 pair6 is now
  completed above; earlier C8 cancellation is superseded. Preliminary
  single-pass result, no repeated-run confidence or stable-tail claim.
````

<a id="cpu-gpu-optimization-l650"></a>

#### open · [L650–688](../../plans/cpu-gpu-optimization/TODO.md#L650-L688)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L650-L688@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `1a14986d08a9caa9d8ccb0a7d8f9dde1a7cb2af2f5518edac4b5a9a66b18e9aa`.

````markdown
- [ ] **OPT-02 reusable bindings:** profile then qualify stable leased row/
  workspace metadata, bounded diff uploads, compact device feedback and removal
  of redundant source/state/readout copies in the original actor. Source/root/
  epoch/physical generations and unknown-completion ownership stay authoritative.
  After the current short baseline, measure the1MiB streamed vocabulary head's
  actual source reads/copies/launches. Qualify admitted larger windows or full
  residency where the complete model/root/workspace inventory fits; retain
  streaming when it does not. Investigate sharing serial cold-geometry storage
  by genuine liveness/alias proofs rather than summing six separate cohorts.
  Both are performance hypotheses, not measured gains or permission to raise caps.
  Current source uses485/970 authenticated1MiB reads+H2D+kernel+same-stream
  checkpoints per .8B/2B Head. Existing .8B original full-table selected-head
  actor offers a reuse path. Raw508,559,360B .8B table addition stays below the
  present4GiB provider/host upper bounds, before deriving new exact overheads;
  original estimate is now superseded by actual .8B validation below. 2B cannot
  inherit that fit claim; evaluate larger tiles and proved workspace reuse.
  Active isolated resident-head unit reuses original full-table actor, keeps
  same arithmetic/source/private ownership and actual4GiB/5GiB caps. Separate
  warmed-process CPU profile now closed healthy: sampled user CPU cycles57.573%
  SHA256,15.202% BF16 finite scanning,10.195% actor spin,7.341% memcmp. Private
  stripped addresses classified from actual instructions. These are CPU cycle
  fractions, not wall/GPU percentages; perf-stat file was empty. Static30.31GiB/
  31,040sync per64forwards is not a measured time percentage.
  Assembled resident .8B passes actual40 natural calls/all5 prompt lengths with
  exact boundaries/fullV/all-state/feedback and pressure/stale/precancel/retirement.
  Actual resident head508,559,360B; source-derived provider upper2,625,406,976B/
  host upper3,397,702,556B fit unchanged4GiB limits. Conv/recurrent/lastFA fault
  cases pass; head-unknown exposed verifier's old streamed-only error recognizer.
  Exactly2 resident strings added behind unchanged error/index254/fault20|21/
  actual-enqueue gates; two affected Head faults await rebuilt fixture. Vendor117
  CPU guards pass. No speedup or complete new-batch acceptance claim yet.
  Follow-up physical Headunknown+cancel PASS; matrix40+all5faults complete.
  Quiet resident CLI2 diagnostic8requests/64forwards exact, owned source/retirement
  closed: forward6.907854s, bind_program3.629112s, enqueue3.021396s,
  complete/readback0.148894s. Nested Head0.005158s,0H2D/0checkpoints/1native per
  call; prepare includes forward, so totals must not be added. Instrumented
  wave7.115s is not an unprofiled performance win. Next substantial host unit
  addresses reusable cold construction and attempt-local state proofs while
  preserving dynamic root/session/generation/fence/cancel/ownership checks.
````

<a id="cpu-gpu-optimization-l689"></a>

#### open · [L689–693](../../plans/cpu-gpu-optimization/TODO.md#L689-L693)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L689-L693@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `8ed8b4e69edd91a090ef581b6d0037081d0118c4a22b00811c959f678d5cc1fa`.

````markdown
- [ ] **OPT-03 prepared optimized regions:** close existing production graph
  admission in the serving plan; qualify exact producer/consumer fusion,
  prologue/epilogue layouts, finite decode/prefill/verify shape variants and
  source-derived packs. Preserve materialized casts/reductions, aliases and
  effective fused read lifetimes. Fewer graph submissions are not fewer kernels.
````

<a id="cpu-gpu-optimization-l694"></a>

#### open · [L694–698](../../plans/cpu-gpu-optimization/TODO.md#L694-L698)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L694-L698@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `d6762f844ca6f4bf8330546797984c7e9d33d2fc2d83d9fa9300abd82d9e20a5`.

````markdown
- [ ] **OPT-04 temporal/fair work:** extend existing native sequence batches
  with actual multi-token dense/GDN prefill and fair mixed-work chunks; retain
  independent state and profile-specific arithmetic. Serving owns scheduling
  integration; memory owns pages/prefix/COW. Re-run short concurrent baselines
  after singles and meaningful combinations before larger contexts.
````

<a id="cpu-gpu-optimization-l699"></a>

#### open · [L699–702](../../plans/cpu-gpu-optimization/TODO.md#L699-L702)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L699-L702@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `abe3b66a6d453050b1ef50d2d0ccf3ce052d5574130b669db07ddaddc18d0b97`.

````markdown
- [ ] **OPT-05 state/residency composition:** full-domain compatible prefix
  intersection, pages/COW, exact source chunk/pack sharing, safe scratch reuse
  and bounded transfer readiness through existing owners. Benchmark cache hits,
  cold arrivals, long prefill versus decode, memory displacement and fair tails.
````

<a id="cpu-gpu-optimization-l703"></a>

#### open · [L703–706](../../plans/cpu-gpu-optimization/TODO.md#L703-L706)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L703-L706@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `1640c00d1f3d49197fadf9f92ea6a59d3b807f5b423c2c3457c309c4d497b9f6`.

````markdown
- [ ] **OPT-06 larger dense source:** after long-context prerequisites, original
  27B execution using authenticated layer/row windows, coalescing/batch reuse
  and admitted prefetch. Price every required dense matrix plus activation IO;
  a sparse expert cache cannot turn this checkpoint into a sparse model.
````

<a id="cpu-gpu-optimization-l707"></a>

#### open · [L707–711](../../plans/cpu-gpu-optimization/TODO.md#L707-L711)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L707-L711@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `8c7446ec70e63042b17d1907c4fb87ec104d52cbd990a70c6b057d1dc8543164`.

````markdown
- [ ] **OPT-07 optional proposals:** target verifier/all-prefix rollback first,
  then bounded n-gram/Recall/suffix and compatible trained MTP/block drafters.
  Joint finite cost policy considers batch/graph/depth/IO/retained state and
  selects K=0 when it loses. Exact distribution, seeded trace, numeric profile
  and quality changes are distinct gates. No acquisition/training implied.
````

<a id="cpu-gpu-optimization-l712"></a>

#### open · [L712–716](../../plans/cpu-gpu-optimization/TODO.md#L712-L716)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L712-L716@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `dace1b0ab9f7a32906ac91f3eee537eb4528ac3ba6a5eef9089e0f7f92688296`.

````markdown
- [ ] **OPT-08 nested profiles:** reconcile primitives→recursive pipelines→
  profiles with the pipeline-composition plan. Compile/in-line/specialize closed
  regions through the original engine; no per-token interpreter/search, new
  serialized IR, model loop, allocator or publisher. Structural design can
  proceed in parallel; new framework implementation must not delay parity.
````

<a id="cpu-gpu-optimization-l717"></a>

#### open · [L717–720](../../plans/cpu-gpu-optimization/TODO.md#L717-L720)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L717-L720@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `143a1ae5e5369933951d1662d7cb9ad94a5ead4f4ef20c2a237f65292d7b1402`.

````markdown
- [ ] **OPT-09 deferred capabilities:** typed sparse selection, grouped experts/
  protected SLRU/hotsets/replicas, deterministic PLE/Engram IO, source-dependent
  shared-KV/liveness, hardware-specific MMA/PDL and derived weight/state codecs.
  Activate only for qualified Qwen/Gemma variants, then other model/device work.
````

<a id="cpu-gpu-optimization-l721"></a>

#### open · [L721–724](../../plans/cpu-gpu-optimization/TODO.md#L721-L724)

Heading: `Cpu gpu optimization > Uncommitted > Ordered adoption — 21 Oct`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L721-L724@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `7a883652c169cb3dbfab1ea1330a6eaed03df72b5cdf57dbb6860d085defc442`.

````markdown
- [ ] **OPT-10 promotion:** source/full-logit/all-state/natural-output, actual
  cancel/pressure/unknown-completion/fair serving and complete isolated check.sh;
  repeated complete-request benefit including cold/setup/memory/energy and
  negative cases. Do not multiply external gains or promise universal superiority.
````

<a id="cpu-gpu-optimization-l731"></a>

#### open · [L731–736](../../plans/cpu-gpu-optimization/TODO.md#L731-L736)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L731-L736@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `917d1b903802a61723ccc6035cfc684840c1c0ba0256c99091f22c3436928f20`.

````markdown
- [ ] Qualify [cross-stage residency/fusion](../pipeline-composition/REGIONS_AND_PROFILES.md#resident-values-and-cross-stage-lowering--owner21-oct)
  through the original engine after baseline: compare ordinary, resident-chain,
  graph and fused paths; retain intermediates/feedback on device where eligible,
  report actual H2D/D2H/device traffic, host waits and reasons for required
  bridges. Preserve arithmetic/declared approximation, original commits,
  lifetimes/cancel/fairness and measure complete-task benefit/negative cases.
````

<a id="cpu-gpu-optimization-l737"></a>

#### open · [L737–743](../../plans/cpu-gpu-optimization/TODO.md#L737-L743)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L737-L743@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `d52dd95b7dae815d54485880e06eadfc1192fc705c5c19ec1fd665b640f86b1a`.

````markdown
- [ ] Complete the dedicated [ExLlamaV3 investigation](../../research/engine-optimization-comparison-2026-10-21/exllamav3-adoption-scope.md):
  pinned kernel/packing/transform/codec consumers, shape/model/backend limits,
  KV, batching/graphs, CPU offload and partition seams; exact file/dependency
  licenses and static C/Rust ABI versus direct-port disposition. Add independent
  fixtures and map mechanisms into existing plans; existing EXL3/QTIP calibration
  design is not missing and must not be duplicated. Initial scoping is not a
  completed audit or a new entry in the completed416-mechanism inventory.
````

<a id="cpu-gpu-optimization-l744"></a>

#### open · [L744–749](../../plans/cpu-gpu-optimization/TODO.md#L744-L749)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L744-L749@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `e89668efe6c16069f382485517bbc9b529f94263d3ce7138d87de045f038c4a8`.

````markdown
- [ ] Implement native kernel-contract comparison/fixture support, reusing or
  directly porting suitable TensorFold harness/leaf code after license/ABI
  review. Add planted rounding/reduction/NaN/tail/no-write/alias/source defects
  and prove every expected rejection. Invalid hardware configurations refuse
  before launch. Keep broken controls test-only; independent oracles, actual
  model/all-state/physical checks and full isolated check.sh remain required.
````

<a id="cpu-gpu-optimization-l750"></a>

#### open · [L750–753](../../plans/cpu-gpu-optimization/TODO.md#L750-L753)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L750-L753@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `f8f5c3c2a359c1fb73a38e045e28084171f5558f30151e3ac77b8e4c147dae64`.

````markdown
- [ ] Implement the useful kernels/components identified by these investigations
  through existing providers, using static-native reuse first and faithful
  direct ports when unsuitable for linking. Qualify the implementation as well
  as the harness; an upstream passing test is not local runtime acceptance.
````

<a id="cpu-gpu-optimization-l754"></a>

#### open · [L754–758](../../plans/cpu-gpu-optimization/TODO.md#L754-L758)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L754-L758@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `af46b55b1d3e71ea316ca2e6e72d4b55808c3a68909cc53c194256c252e336d7`.

````markdown
- [ ] Specialize first for the NVIDIA cards on both existing machines, then
  prioritize Strix Halo/gfx1151 ahead of all other new hardware and cluster
  work. Keep broad model/device/platform compatibility as the long-term target,
  with portable paths and actual per-target gates; retain the separate gfx90c
  hold. Hardware order does not alter parity→short→long→dense27B model order.
````

<a id="cpu-gpu-optimization-l760"></a>

#### open · [L760–767](../../plans/cpu-gpu-optimization/TODO.md#L760-L767)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L760-L767@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `9a63e8205b7986b98c1a018339a7a57e719a9235448fb91fb008a57d40c352b5`.

````markdown
- [ ] Bench and hyper-optimize in strict user order:
  Qwen3.5 2B, dense Qwen3.8 27B including SSD/coalesced layer-window streaming;
  Gemma4 E2B/26B/31B; Qwen3.8 Flash-Next Q2/~3.5-bpw/Q4
  and MTP/drafter/n-gram pipeline. Optimize complete
  request goodput with the same model semantics, source quant
  calibration and StateRoot precision. Do not change the
  Renoir/gfx90c launch hold or assume measured superiority.
  Routed-expert streaming applies only to later source-defined MoE variants.
````

<a id="cpu-gpu-optimization-l769"></a>

#### open · [L769–769](../../plans/cpu-gpu-optimization/TODO.md#L769)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L769-L769@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `04e7aefc61e641d1c5147225169e6867d245284bf1766cd8da30bbcfeff871e7`.

````markdown
- [ ] Keep the local Renoir/gfx90c hold: 2B initialization aborted and automatic resets wedged it; no further launch/reset/recovery or Drop-retirement proof is implied.
````

<a id="cpu-gpu-optimization-l770"></a>

#### open · [L770–770](../../plans/cpu-gpu-optimization/TODO.md#L770)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L770-L770@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `5cadab572f96f321ab5e97df27de7dec6974450bf9ae11d9d3b8685a5b2cbbc5`.

````markdown
- [ ] Establish controlled cold/warm request baselines across all five GPU/three CPU optimization targets; accepted five backend profiles do not mean five cards or fleet-wide maximal capability use.
````

<a id="cpu-gpu-optimization-l771"></a>

#### open · [L771–771](../../plans/cpu-gpu-optimization/TODO.md#L771)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L771-L771@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `eae77cd93ecb982e521887d63c8fe5263b2ff8e0c17af0669d5179f51f71c2cb`.

````markdown
- [ ] Qualify capability-selected AVX2/AVX512/VNNI quantized kernels, cache/thread/topology policies and source-derived repacking while preserving reduction/rounding/zero/subnormal rules.
````

<a id="cpu-gpu-optimization-l772"></a>

#### open · [L772–772](../../plans/cpu-gpu-optimization/TODO.md#L772)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L772-L772@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `088205480ab059a9aa383cf3900053685135fcd95e7208b2605ecca9edc8fba7`.

````markdown
- [ ] Measure native/Vulkan tiling, subgroup/vector loads, packing, source-exact fusion/prepared dispatch; gate tensor/MMA/WMMA/integer-dot use by exact arithmetic, not advertised support.
````

<a id="cpu-gpu-optimization-l773"></a>

#### open · [L773–773](../../plans/cpu-gpu-optimization/TODO.md#L773)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L773-L773@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `75a40aa8caea6bd8fa1aaeab92d9113906869ec1851391f5db078082abfa1594`.

````markdown
- [ ] Evaluate explicit NativeF32 projection policy only after physical FMA edge tests, all-codec trained vectors and whole-model/cancellation/replay gates; WideF64 stays default meanwhile.
````

<a id="cpu-gpu-optimization-l774"></a>

#### open · [L774–774](../../plans/cpu-gpu-optimization/TODO.md#L774)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L774-L774@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `351119f3af1696c4b7c188eb5c0cfb6d3b17d8b12d45e9c4dcc60b0c595446d3`.

````markdown
- [ ] Profile then qualify captured graphs/replay, bounded device loops/persistent scheduling, CPU/GPU overlap and exact optional GPU tokenization/detokenization; prepared plans alone are not graph capture.
````

<a id="cpu-gpu-optimization-l775"></a>

#### open · [L775–775](../../plans/cpu-gpu-optimization/TODO.md#L775)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L775-L775@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `b4052f2df1d7accb36ae10f5c28c431af31ab2d2fb9fd678d2a787e44a03c58c`.

````markdown
- [ ] Keep separate LLVM code-generation paths and broader accelerator backends unqualified until independently built and physically tested; accepted native paths use CPP NVRTC/HIPRTC.
````

<a id="cpu-gpu-optimization-l776"></a>

#### open · [L776–776](../../plans/cpu-gpu-optimization/TODO.md#L776)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L776-L776@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `ce9af64a6ca4ac5332ed95f0f767cf61e96c48a2f816344d783ef0d5159a83b3`.

````markdown
- [ ] Qualify optional EXL3/QTIP/trellis/codebook/low-bit derived packs and calibration identity/quality separately from original source semantics. No hidden precision substitution or default steering.
````

<a id="cpu-gpu-optimization-l777"></a>

#### open · [L777–777](../../plans/cpu-gpu-optimization/TODO.md#L777)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L777-L777@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `92f92e2317dc659742505c0d77550ac1feda4e83ecef5f6a3821875630669799`.

````markdown
- [ ] Promote only repeated complete-request gains with cold/warm cost, tails, memory, power, source correctness, negative results and qualified capability/version guards.
````

<a id="cpu-gpu-optimization-l779"></a>

#### open · [L779–779](../../plans/cpu-gpu-optimization/TODO.md#L779)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L779-L779@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `242065e9c8d376e9fa779bb4f89338eeb91dc8bc088cd82f4fbb0fad49e28239`.

````markdown
- [ ] Gate future compiler/driver/libm and fast-arithmetic profiles with scalar fingerprints, physical edge/stress and trained full-logit/state tests. WideF64 is qualified on measured inputs, not a universal IEEE FMA emulation.
````

<a id="cpu-gpu-optimization-l780"></a>

#### open · [L780–780](../../plans/cpu-gpu-optimization/TODO.md#L780)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L780-L780@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `9f803a90a6ae02d876897226807278020579e025864476fdfb93063cbbfe780f`.

````markdown
- [ ] Qualify real deadline wakeups, concurrent/multi-stream scheduling and actual driver failure/recovery before broader worker claims; queue deadlines currently need scheduling/poll opportunities.
````

<a id="cpu-gpu-optimization-l781"></a>

#### open · [L781–781](../../plans/cpu-gpu-optimization/TODO.md#L781)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L781-L781@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `5dcdd13c2e1b55364dcf8a3cc1dc3c209639c7fb4afd029ec0881688b6be88c0`.

````markdown
- [ ] Keep the local gfx90c hold. If a healthy/recovered profile is separately authorized, localize retained allocation/H2D/final-sync initialization before changing kernels; natural SIGABRT establishes neither Rust Drop quarantine nor reclaimed storage.
````

<a id="cpu-gpu-optimization-l782"></a>

#### open · [L782–782](../../plans/cpu-gpu-optimization/TODO.md#L782)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L782-L782@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `e44705c55d6edeef35353b4c7f2187a253d436edb87cc16e635712c7657a8dfe`.

````markdown
- [ ] Consider read-mostly publication only after representative contention and ownership evidence. If left-right fits, test deterministic dual replay without IO/RNG/admission/state/device effects, one writer, persistent non-Sync readers, short guards without await/load/device work and explicit publish/drain/shutdown. Account for both indexes/logs while sharing one payload; do not reopen the rejected cold-catalog migration.
````

<a id="cpu-gpu-optimization-l784"></a>

#### open · [L784–784](../../plans/cpu-gpu-optimization/TODO.md#L784)

Heading: `Cpu gpu optimization > Uncommitted > Owner follow-up — 21 Oct, reuse/harness/target coverage`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L784-L784@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `b37fb4c87fb9bb08624453b544c0a10f81583b0a52b3b1f3cbf7d1aadacb2bf3`.

````markdown
- [ ] Resolve applicable SYK-011..016 source/derived-quality/LLVM/SLO/pressure/steering choices from the [specialization reference](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) before that slice; proposed SYK-002..010 are not frozen user approvals. Preserve negative candidates and the default disabled steering policy.
````

<a id="cpu-gpu-optimization-l801"></a>

#### ~ · [L801–815](../../plans/cpu-gpu-optimization/TODO.md#L801-L815)

Heading: ``Cpu gpu optimization > Committed > `5d40cf5` — shared pending queue and ingress credit``.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L801-L815@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `9f73ebcbaed540f3fc35c6f47fb76dbf025f604ce8acb915a11a5a15fc91a7cf`.

````markdown
- [~] Shared queue checkpoint preserves application-owned HOST admission,
  unlimited default pending count and optional cap. Final batch corrects
  tokenizer construction, original Rayon reuse, HTTP framing/serialization
  and worker/last-reader ownership. HTTP handles and connection credit survive
  through join; original default thread stacks remain runtime-selected.
  Platform stacks/TLS/caches and OS/driver headroom remain separate, unquantified
  scope: application reservations are not process RSS or a full-process fit.
  [Contract decision](../../runs/resume-all-queue-production-bound-astra-review-20261024/MERGE_SCOPE_DECISION.md).
  CPU functional acceptance is complete; the final full CPU gate passed
  rc0 25Oct on the committed tree. Assembled GPU checks PASSED 25Oct
  (22/22 rc0: builds/stricts, cuda_compiler_context 4/4, loaded/inventory
  4/4, CLI baseline/serve-short_hf, host-profile umbrella, four ignored
  route cases alone, five actual source-cost cases) —
  [execution](../../runs/resume-all-takeover-queue-gpu-glm-20261025/LOG.md).
  Physical device scopes (CLI direct route, serving) remain open.
````

<a id="cpu-gpu-optimization-l846"></a>

#### ~ · [L846–855](../../plans/cpu-gpu-optimization/TODO.md#L846-L855)

Heading: ``Cpu gpu optimization > Committed > `52f76a0` — short-HF tokenizer credit and HOST8``.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L846-L855@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `bbfa11ddb6893018f36687fcac45166d93b8125e97812e9a5438f92e63dd4203`.

````markdown
- [~] Direct short-HF route lifetime correction: preliminary tokenizer drops
  before model loading, but its static retained-credit vector never releases.
  Return tokenizer plus its actual lease; validate the exact legitimate
  process-global bootstrap baseline through route/model closure. No VM9/H14
  source overlap. Closed11-path source draft has the typed owner; ROOT/Astra
  required collected corrections for request payload/credit lifetimes,
  unconditional original close, exact bootstrap/backend baselines and tests.
  Those source repairs landed in `52f76a0`; the affected optimized/strict
  batch and the final full CPU gate passed rc0 25Oct; real source-cost/
  lifetime and GPU serving/ownership checks remain pending.
````

<a id="cpu-gpu-optimization-l856"></a>

#### ~ · [L856–869](../../plans/cpu-gpu-optimization/TODO.md#L856-L869)

Heading: ``Cpu gpu optimization > Committed > `52f76a0` — short-HF tokenizer credit and HOST8``.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L856-L869@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `9f8bb77b32912e999fbd14d5ba7f355be0b144ae768bd11b684a22e93b7121a4`.

````markdown
- [~] CUDA B4 HOST requirement needs correction before fresh acceptance.
  Source-derived P16 lower bound4,341,525,623B already exceeds4GiB by46,558,327B
  before bootstrap/native/other metadata; old source21 HOST fits omit the new
  R+G+V credit. Cost-only inventory currently bypasses that owner too. Preserve
  all real charges, GPU caps and CPU4096MiB population controls; review honest
  configured HOST requirements versus real owner/phase savings. Explicit
  short-HF8192MiB policy landed in `52f76a0`; CPU/default4096 and legacy4GiB,
  provider/physical and H/G/V caps remain unchanged. Measured 25Oct
  (application-owned configured-domain accounting at 8192MiB, gpu_executed
  false, no process-RSS or device claim): N1 .8B host_upper 4,075,700,811B,
  N1 2B 4,111,001,099B, B4 2B 4,741,293,239B, B4 .8B 4,736,259,779B — all
  fit 8GiB; legacy 4GiB refusal tied to authenticated host cost (excess
  446,325,943B, provider within cap). Real RAM/cgroup availability and
  device/serving execution checks remain open.
````

<a id="cpu-gpu-optimization-l880"></a>

#### ~ · [L880–886](../../plans/cpu-gpu-optimization/TODO.md#L880-L886)

Heading: ``Cpu gpu optimization > Committed > 25Oct assembly and evidence records — `docs(plans)` (this commit)``.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L880-L886@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `233e6b20cae607ecdd208730bc66a0058c34d7f0bb2bae20c6d9268988b224a2`.

````markdown
- [~] Five-lane source is assembled in MAIN; the final affected batch completes
  portable core admission, actual spawn-capture ownership and direct-lease
  metadata. Original private-network HTTP checks now execute without a fallback.
  The final source freeze/full CPU gate passed rc0 25Oct and the five-lane
  batch landed as signed commits (`8f01635`..`68e13a8`); assembled GPU/
  physical checks, H6/G8/V10 cases and lane-tree retirement remain pending;
  no new benchmark yet.
````

<a id="cpu-gpu-optimization-l1194"></a>

#### open · [L1194–1204](../../plans/cpu-gpu-optimization/TODO.md#L1194-L1204)

Heading: `Cpu gpu optimization > Tests and verification`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L1194-L1204@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `f4119978082a2f05d797d10bc9fcffb30e2ef0081a3ccb9929b5e1293ffd534b`.

````markdown
- [ ] Assembled .8B host batch: authenticate immutable114-weight cohort once
  per binding, reuse exact cold allocation analysis/certificates, seal fresh
  native roles once, index actual lease/access coverage. Current source/root/
  permit/generation/fence/cancel and physical range checks remain. Existing
  pre-node0 status clear permits removal of redundant4B upload/sync; no new
  sticky-status semantics or cap changes. Ten paths merged into ROOT ISO;
  frozen physical fixture passes full40+five-fault requalification, actual final
  host0/device0 and original owned closure. All14 pure metadata guards PASS,
  including three new lifetime/rebind/SSA/private-effect/alias guards. Matched
  API wave/ragged checks PASS full IDs/text/usage/finish and actual owned/source
  closure. Clean C1/C4 performance remains pending CPU/IO quiet clearance.
````

<a id="cpu-gpu-optimization-l1205"></a>

#### open · [L1205–1206](../../plans/cpu-gpu-optimization/TODO.md#L1205-L1206)

Heading: `Cpu gpu optimization > Tests and verification`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L1205-L1206@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `28ed15547a3cf350a91268ac237436768268559227a3e88cf72eb4425f480650`.

````markdown
- [ ] Native harness passes positive fixtures and rejects all named planted
  defects; independent oracle and selected-profile equivalence matrix pass.
````

<a id="cpu-gpu-optimization-l1207"></a>

#### open · [L1207–1209](../../plans/cpu-gpu-optimization/TODO.md#L1207-L1209)

Heading: `Cpu gpu optimization > Tests and verification`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L1207-L1209@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `a4264b2a6358ee8f4be1f1889f6cc3aff19e31737aecd5e692e4653cd828addc`.

````markdown
- [ ] Reused/ported implementation passes exact/approximate declared-contract,
  held-out quality where applicable, full-state/cancel/physical and native
  packaging/link checks independently of the harness's own controls.
````

<a id="cpu-gpu-optimization-l1210"></a>

#### open · [L1210–1210](../../plans/cpu-gpu-optimization/TODO.md#L1210)

Heading: `Cpu gpu optimization > Tests and verification`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L1210-L1210@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `f6986fa69902a196ce22c383252fcb9963ac1671075e181a47281dd06fdec362`.

````markdown
- [ ] QTIP-01..14: derived source codec/pack identity, transforms, held-out quality and physical kernel correctness pass independently of original model semantics.
````

<a id="cpu-gpu-optimization-l1211"></a>

#### open · [L1211–1211](../../plans/cpu-gpu-optimization/TODO.md#L1211)

Heading: `Cpu gpu optimization > Tests and verification`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L1211-L1211@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `48d8f35190be16e98001b374cbc16abec3259f6525f380698c80872d2a613109`.

````markdown
- [ ] HYB-01..18: exact source routing/draft/tap/mask/reduction and target-authoritative verification pass, including cases where a wrong draft is rejected or deeper drafting slows work.
````

<a id="cpu-gpu-optimization-l1212"></a>

#### open · [L1212–1212](../../plans/cpu-gpu-optimization/TODO.md#L1212)

Heading: `Cpu gpu optimization > Tests and verification`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L1212-L1212@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `99d45b13a91b90574a89ad12b63586ea3ae7c95acbb1f61f1da0dc2cb5a2c703`.

````markdown
- [ ] GFX-01..18: WMMA/LDS caller/shape coverage, driver/compiler numeric gates and coherent-memory physical headroom pass; require matched complete-request gains. See the [50-test register](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md#4-acceptance-register-50-planned-tests-not-implemented).
````

<a id="cpu-gpu-optimization-l1218"></a>

#### open · [L1218–1218](../../plans/cpu-gpu-optimization/TODO.md#L1218)

Heading: `Cpu gpu optimization > Tests and verification > 24 Oct — current 2B readiness checkpoint`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L1218-L1218@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `6d178c8608f854c6b5a7f2e3a5d7d0977909075cf365dbc10917dcfbebe2d60a`.

````markdown
- [ ] Complete current15 fresh matched API/HTTP for .8B and2B; ROOT1254 running. Then quiet C1/C4 and C8 only after both rate+wall wins, no hot compiler activity; previous .8B measurements remain historical.
````

<a id="cpu-gpu-optimization-l1219"></a>

#### open · [L1219–1219](../../plans/cpu-gpu-optimization/TODO.md#L1219)

Heading: `Cpu gpu optimization > Tests and verification > 24 Oct — current 2B readiness checkpoint`.
Identity: `.agents/plans/cpu-gpu-optimization/TODO.md:L1219-L1219@66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038`.
Block SHA256: `dda09131b6b0ed587f1fca8764fe849ee7b95111af36694579d9100ac1bd4121`.

````markdown
- [ ] Full assembled check.sh phase gate after CLI/source reconciliation; source12 gate18 remains the last accepted whole gate.
````

### distributed-execution — 12 noncompleted occurrences

<a id="distributed-execution-l21"></a>

#### open · [L21–24](../../plans/distributed-execution/TODO.md#L21-L24)

Heading: `Distributed execution > Uncommitted > Deferred mechanism reconciliation — 21 Oct`.
Identity: `.agents/plans/distributed-execution/TODO.md:L21-L24@033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff`.
Block SHA256: `780820b48e112fb324c97a0727ea02e1ead48f980b4d18a3fdb15f9bee3a5ac2`.

````markdown
- [ ] Evaluate layer, tensor/head, context and expert partitions as independent
  placement choices under source-specific partition algebra. Distributed full
  softmax summaries and recurrent state are different contracts; changed reduction
  or8/16-bit activation wire formats need explicit numerical profiles.
````

<a id="distributed-execution-l25"></a>

#### open · [L25–27](../../plans/distributed-execution/TODO.md#L25-L27)

Heading: `Distributed execution > Uncommitted > Deferred mechanism reconciliation — 21 Oct`.
Identity: `.agents/plans/distributed-execution/TODO.md:L25-L27@033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff`.
Block SHA256: `6293e48eb40c71c8ffeae83bbc2f20b0935a6f5bf8b13ce9e48be36df6bbec0b`.

````markdown
- [ ] Admit shared collective/IPC/RDMA backing, staging, graph and encoded-reader
  ownership through original per-physical-resource ledgers. UMA host-visible
  transfer is not discrete GPUDirect; port ratings/tunnels do not attest throughput.
````

<a id="distributed-execution-l28"></a>

#### open · [L28–31](../../plans/distributed-execution/TODO.md#L28-L31)

Heading: `Distributed execution > Uncommitted > Deferred mechanism reconciliation — 21 Oct`.
Identity: `.agents/plans/distributed-execution/TODO.md:L28-L31@033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff`.
Block SHA256: `1942c99efa409d3ed8be576f73c4642f7c289afce9d8fb64fc668b60cf5774ae`.

````markdown
- [ ] Later MoE EP/replicas/EPLB/communication overlap: preserve logical expert
  source IDs, topology/version guards, old/new migration readers and rank-agreed
  batch/graph/verification shapes. PDL requires actual eligible hardware; DBO is
  not generic one-GPU prefill/decode overlap.
````

<a id="distributed-execution-l32"></a>

#### open · [L32–35](../../plans/distributed-execution/TODO.md#L32-L35)

Heading: `Distributed execution > Uncommitted > Deferred mechanism reconciliation — 21 Oct`.
Identity: `.agents/plans/distributed-execution/TODO.md:L32-L35@033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff`.
Block SHA256: `2b11db09928ad9a33245ac96c9806428a89a57446bdca223eafac1aae2794ec0`.

````markdown
- [ ] Later produced-state P/D/encoder separation and remote prefix hydrate:
  complete hybrid source/profile/tenant/root-frontier handoff, authenticated
  producer/consumer epochs, completion notifications and failure fencing. Heartbeat
  expiry or peer silence cannot reclaim unknown DMA or publish partial roots.
````

<a id="distributed-execution-l36"></a>

#### open · [L36–39](../../plans/distributed-execution/TODO.md#L36-L39)

Heading: `Distributed execution > Uncommitted > Deferred mechanism reconciliation — 21 Oct`.
Identity: `.agents/plans/distributed-execution/TODO.md:L36-L39@033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff`.
Block SHA256: `a4eef7f5c42cb3770059f39915109c5eff3d70ff159b26ab8b490d3a3351d23c`.

````markdown
- [ ] Compare placement against local batching and capacity needs using actual
  directional links/host copies/crypto/collectives/stragglers and offered-load
  tails/goodput. Retain two-device/tiny-model and grouped-pipeline negative cases;
  no universal speedup from more ranks or deferred global-service descriptions.
````

<a id="distributed-execution-l41"></a>

#### open · [L41–41](../../plans/distributed-execution/TODO.md#L41)

Heading: `Distributed execution > Uncommitted > Deferred mechanism reconciliation — 21 Oct`.
Identity: `.agents/plans/distributed-execution/TODO.md:L41-L41@033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff`.
Block SHA256: `06519db3933ca823f50c11dfb2a84a7511489aa1713a1b0295f83fd5e19ff660`.

````markdown
- [ ] Preserve zero-network standalone execution; add capability/transport/placement seams through the same model/task/coordinator/resource authorities.
````

<a id="distributed-execution-l42"></a>

#### open · [L42–42](../../plans/distributed-execution/TODO.md#L42)

Heading: `Distributed execution > Uncommitted > Deferred mechanism reconciliation — 21 Oct`.
Identity: `.agents/plans/distributed-execution/TODO.md:L42-L42@033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff`.
Block SHA256: `291288b5be81cf4418b947a18ae3fd4fcc945e34ee579090b24f230604db832b`.

````markdown
- [ ] Qualify optional local CPU/GPU/NPU split projection, expert ownership and prefill/decode placement with actual event/memory/power-fabric safety; Stage N is independent of CPU/GPU release.
````

<a id="distributed-execution-l43"></a>

#### open · [L43–43](../../plans/distributed-execution/TODO.md#L43)

Heading: `Distributed execution > Uncommitted > Deferred mechanism reconciliation — 21 Oct`.
Identity: `.agents/plans/distributed-execution/TODO.md:L43-L43@033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff`.
Block SHA256: `db84f81e0c16ccfc984aab55fc116d49a954ecf577fde58e098faf9dd287d4fa`.

````markdown
- [ ] Begin authorized two-node layer-group plus local-state execution and immutable full-state hydration; then evaluate P/D and attention+state versus FFN/expert disaggregation.
````

<a id="distributed-execution-l44"></a>

#### open · [L44–44](../../plans/distributed-execution/TODO.md#L44)

Heading: `Distributed execution > Uncommitted > Deferred mechanism reconciliation — 21 Oct`.
Identity: `.agents/plans/distributed-execution/TODO.md:L44-L44@033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff`.
Block SHA256: `85ae1c5d77acf3875fe40884d46a2155504e8f7d0e070eb59cd612a351c3bd48`.

````markdown
- [ ] Extend tensor/head/pipeline/data/context/expert/sequence parallelism and collectives only where exact state partition algebra, source numerics and transport cost support them.
````

<a id="distributed-execution-l45"></a>

#### open · [L45–45](../../plans/distributed-execution/TODO.md#L45)

Heading: `Distributed execution > Uncommitted > Deferred mechanism reconciliation — 21 Oct`.
Identity: `.agents/plans/distributed-execution/TODO.md:L45-L45@033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff`.
Block SHA256: `a177fc602da032fcbc4c348bd99cc915be2936e71ee132eff539955209090761`.

````markdown
- [ ] Authenticate/version source/state/tenant/session epochs and complete transfers; loss/timeout/stale peers cannot publish partial roots or release unknown in-flight storage.
````

<a id="distributed-execution-l46"></a>

#### open · [L46–46](../../plans/distributed-execution/TODO.md#L46)

Heading: `Distributed execution > Uncommitted > Deferred mechanism reconciliation — 21 Oct`.
Identity: `.agents/plans/distributed-execution/TODO.md:L46-L46@033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff`.
Block SHA256: `7b5194709666d0a2d0d5c4c017463e966805652afeff7b680fef2d4c6e7fad83`.

````markdown
- [ ] Qualify topology changes only at committed frontiers and compare collocation against fast/slow/tunneled links, including real transfer/copy/crypto cost and fairness.
````

<a id="distributed-execution-l47"></a>

#### open · [L47–47](../../plans/distributed-execution/TODO.md#L47)

Heading: `Distributed execution > Uncommitted > Deferred mechanism reconciliation — 21 Oct`.
Identity: `.agents/plans/distributed-execution/TODO.md:L47-L47@033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff`.
Block SHA256: `c8765e7ef6d5a46861035694dd9d20a584a7254c7617cd5d72f21ff1f26da4de`.

````markdown
- [ ] Add a peer/transport/placement without rewriting model, task, coordinator or API; fleet SSH qualification alone is not distributed inference.
````

### memory-and-streaming — 25 noncompleted occurrences

<a id="memory-and-streaming-l14"></a>

#### open · [L14–19](../../plans/memory-and-streaming/TODO.md#L14-L19)

Heading: `Memory and streaming > Uncommitted`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L14-L19@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `f1916bc803ff28038e99f72d0bd18dc792ec3f0324cc3b38c0caf1c27f6dc3df`.

````markdown
- [ ] Adopt bounded completed-root backing reuse with fresh logical leases,
  shared immutable layouts/access schemas, owned authenticated transfer slots
  and borrowed completed readbacks through existing resident APIs. Charge idle,
  Pending/snapshot/readers and unknown work until actual retirement. Shared
  rings must expose real pinned/heap and backend ordering; no model-owned pool
  or implied zero-copy/overlap. Coordinate substantial units with optimization.
````

<a id="memory-and-streaming-l26"></a>

#### open · [L26–29](../../plans/memory-and-streaming/TODO.md#L26-L29)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L26-L29@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `248136f5afb1afbe2b5723d1192ec06857a6310e95b9bfb4bb88190c800448fb`.

````markdown
- [ ] Bind physical backing identity across aliases, prepared packs, graph views,
  source windows and transfer staging. Charge one actual allocation once and
  separate real copies/metadata; equal hashes/chunk coordinates do not prove
  shared storage. Include encoded/captured/queued readers in retirement.
````

<a id="memory-and-streaming-l30"></a>

#### open · [L30–32](../../plans/memory-and-streaming/TODO.md#L30-L32)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L30-L32@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `a49c5c17fa11a5df3f4c9277100f62e067a5402fae40c8a6d8cf97526c85f08a`.

````markdown
- [ ] Add measured lifetime-aware scratch reuse/phase lending only when no live
  root, graph, transfer or reader needs the range. Price borrowed capacity,
  old/new COW copies and following decode residency; do not import a hidden arena.
````

<a id="memory-and-streaming-l33"></a>

#### open · [L33–36](../../plans/memory-and-streaming/TODO.md#L33-L36)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L33-L36@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `1372a728a9c21da768b26fb5bbb12a551bc2de6598fec4bcf69b1ab62c6268a3`.

````markdown
- [ ] Protect demand/decode hotsets from transient prefill and unused predicted
  reads; compare protected segmented/recency/frequency policies under topic shifts.
  Different source codecs use byte-size classes under one cap. Cache policy may
  change residency, never source routing, expert count or arithmetic silently.
````

<a id="memory-and-streaming-l37"></a>

#### open · [L37–40](../../plans/memory-and-streaming/TODO.md#L37-L40)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L37-L40@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `ee39485098927014c34d676c0a0866c32030a7a21bf621f41480fc6e9b2b4fd2`.

````markdown
- [ ] Separate known source lookup IDs, dense next-layer windows, produced-state
  restore and predicted expert prefetch, while sharing fair bounded IO/host/device
  admission. Optional direct/async IO retains the same hash/short-read/cancel
  contract. A fully resident path performs no mandatory cache-miss bookkeeping.
````

<a id="memory-and-streaming-l41"></a>

#### open · [L41–44](../../plans/memory-and-streaming/TODO.md#L41-L44)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L41-L44@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `d0901b5d1339005c654fe5e5b7d402c5b83521aa5af5a87ec2e46da4d7289b4a`.

````markdown
- [ ] Implement common-frontier prefix reconciliation as a bounded monotone
  intersection over every required domain's legal checkpoint/alignment rules;
  exact partial-tail COW retains both endpoints until real copy/read completion.
  Approximate truncated replay and moved-message reuse stay outside exact mode.
````

<a id="memory-and-streaming-l45"></a>

#### open · [L45–48](../../plans/memory-and-streaming/TODO.md#L45-L48)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L45-L48@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `a27c0962cc300b83e7639d1f3e89c555fbfb22485afcbe9673376b8118648d38`.

````markdown
- [ ] For dense27B, compare layer-major/row-cohort source-window reuse against
  extra activation spill, retained states, IO amplification and fair decode
  progress. First correct execution may itself require streaming; no full-resident
  runtime is assumed feasible under the declared5GiB target.
````

<a id="memory-and-streaming-l50"></a>

#### open · [L50–54](../../plans/memory-and-streaming/TODO.md#L50-L54)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L50-L54@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `3a6b8aa7928b186eaee0102e8ed90d89b1e2f94c9d6bd82416dfa2d17e625c7e`.

````markdown
- [ ] **Next SSD source execution:** integrate accepted original-index
  owned windows and an async I/O/readiness model with
  original CPU/GPU TensorSource and BufferLease, cancellation,
  cache/eviction/ABA and independently trained full-logit/StateRoot
  parity on real large weights.
````

<a id="memory-and-streaming-l56"></a>

#### open · [L56–61](../../plans/memory-and-streaming/TODO.md#L56-L61)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L56-L61@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `6945964a3bc7020f5215841c8c0d0e779602f3c9bffb190f9b67efd7bb0e3b7d`.

````markdown
- [ ] Shared one-process pipeline residency: source-exact immutable
  model/component buffers, per-session private StateRoots, one
  ResourceArbiter (UMA/GPU/RAM/SSD and device queue), cross-model
  lease/zero-copy only when physical placement, codec and lifetime
  permit it. Do not replicate weights per route or release a
  GPU buffer before actual completion/quarantine retirement.
````

<a id="memory-and-streaming-l62"></a>

#### open · [L62–68](../../plans/memory-and-streaming/TODO.md#L62-L68)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L62-L68@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `46b77c5f81dab690e3a3cc20e918a7b50cff4fde98c3f3ef7f500b2b4582786e`.

````markdown
- [ ] Prioritize Qwen3.5 2B resident then Qwen3.8 27B SSD
  streaming and Gemma4 E2B/26B/31B, then sparse Qwen Flash-Next
  learned 51B n-gram bank and 4B MTP head placements. Compare
  source-real logits, resident/streamed StateRoot, cold/warm
  TTFT/ITL, I/O/read amplification and throughput with full
  physical/cancel/error gates; live audio's soft deadline may
  demand resident codecs and refuse cold SSD layouts.
````

<a id="memory-and-streaming-l70"></a>

#### open · [L70–70](../../plans/memory-and-streaming/TODO.md#L70)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L70-L70@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `b61a2379b50edd79d27905468009a0408e711a8126b9e190cc8b6c9581684d8d`.

````markdown
- [ ] Route legacy/new model allocations, CPU state/scratch/staging and APU/GPU shared backing into the same configured physical pools with OS/cgroup headroom; never sum overlapping UMA/VRAM/GTT counters.
````

<a id="memory-and-streaming-l71"></a>

#### open · [L71–76](../../plans/memory-and-streaming/TODO.md#L71-L76)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L71-L76@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `7176796fd6218283dfbf96e1aa5a413f458b85e738ed29c0dcb0c63fafef4706`.

````markdown
- [ ] **Paged hybrid state allocator:** extend original leases/StateRoot with
  finite page pools/block tables, generations, valid tails, charged metadata
  and peak COW/candidate memory. Retain a qualified contiguous policy and typed
  recurrent/conv checkpoints. Pin encoded/captured/queued/in-flight readers;
  reclaim only after real completion, preserving cancellation/quarantine.
  Design and physical acceptance: [batching/paged state](../serving-and-speculation/BATCHING_AND_PAGED_STATE.md#paged-state-allocator--memory-plan).
````

<a id="memory-and-streaming-l77"></a>

#### open · [L77–85](../../plans/memory-and-streaming/TODO.md#L77-L85)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L77-L85@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `2414dca5cd47d45093d935f127b07d68df4673d57235404c8ecfb8e805fb9472`.

````markdown
- [ ] **Complete-domain prefix reuse/COW:** source/loader/quant/numerical-profile,
  token/template/processor/adapter, position/schema and tenant-safe identity;
  greatest jointly restorable frontier across KV/SWA/recurrent/indexer/draft/
  grammar/media domains. Share only immutable authorized state; fork private
  roots and copy changed shared domains. Missing recurrent state requires
  earlier checkpoint/recompute, not a full KV-only hit. Qualify hit/miss/fork,
  pressure/eviction/stale epochs, page boundaries and full-logit/all-state
  continuations against cache-disabled replay. Exact gates and serving owner:
  [prefix design](../serving-and-speculation/BATCHING_AND_PAGED_STATE.md#complete-domain-prefix-reuse-and-cow--memory-plan).
````

<a id="memory-and-streaming-l86"></a>

#### open · [L86–86](../../plans/memory-and-streaming/TODO.md#L86)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L86-L86@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `b64ba9619e4c901ae0758361c6d40c5be8d9dba0d2c3ac208f0803772e654374`.

````markdown
- [ ] Qualify committed snapshot persistence/restore, corruption/security/compatibility UX and restore-versus-recompute; diagnostic portable checkpoints are not product persistence.
````

<a id="memory-and-streaming-l87"></a>

#### open · [L87–87](../../plans/memory-and-streaming/TODO.md#L87)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L87-L87@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `259ab068953cc919c4b554fa51f9b212017da46e98b5a80cbdc119f85227d617`.

````markdown
- [ ] Add immutable source extents and bounded coalesced IO/materialization with stable logical tensor identity, genuine copy readiness and original IO/device reader leases.
````

<a id="memory-and-streaming-l88"></a>

#### open · [L88–88](../../plans/memory-and-streaming/TODO.md#L88)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L88-L88@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `b061d39d9369bde61f8476d692ec9a8bde4aa785ecc134c8c3f2456c36e81050`.

````markdown
- [ ] Protect encoded-but-unsubmitted/captured expert reads and generation ABA; support mixed-codec expert size classes under one byte cap, source-exact routing and hot-cache preservation.
````

<a id="memory-and-streaming-l89"></a>

#### open · [L89–89](../../plans/memory-and-streaming/TODO.md#L89)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L89-L89@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `162ea40ef635aaa46d5106a1fc1fe1662290f9955c239e911703f16c9a4a8a53`.

````markdown
- [ ] Add bounded PLE/Engram lookup rows and separately governed disk state/media tiers; fully resident execution must incur no mandatory IO or host cache dispatch.
````

<a id="memory-and-streaming-l90"></a>

#### open · [L90–90](../../plans/memory-and-streaming/TODO.md#L90)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L90-L90@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `823e3d870c760f8e6bb3592748720c451746787d54c5917a26910c2a8f4d5fcd`.

````markdown
- [ ] Test source-exact resident/streamed parity, short/corrupt IO, cancellation, safe victim refusal, SSD QoS and actual physical lifetime before eviction/free-space claims.
````

<a id="memory-and-streaming-l91"></a>

#### open · [L91–91](../../plans/memory-and-streaming/TODO.md#L91)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L91-L91@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `684a56ec6d25782c9977a69d365ef2ff3241727c74a52ac52b0767059f7a9744`.

````markdown
- [ ] Optional direct/async/platform IO needs equivalent semantics and explicit support; large memory-pressure/SSD experiments require approved hardware/source scope.
````

<a id="memory-and-streaming-l93"></a>

#### open · [L93–93](../../plans/memory-and-streaming/TODO.md#L93)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L93-L93@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `93b3e52bf67062efc882171ae34c71560c588ac4329f020ce0d6630ed99e342d`.

````markdown
- [ ] Distinguish logical envelope retirement from driver allocator pages/pools, fragmentation, cross-process use and actual OS release. Driver VRAM/GTT usage/max counters are not reservations or immediately free bytes.
````

<a id="memory-and-streaming-l94"></a>

#### open · [L94–94](../../plans/memory-and-streaming/TODO.md#L94)

Heading: `Memory and streaming > Uncommitted > Mechanism adoption — 21 Oct, unimplemented follow-up`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L94-L94@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `e7c73db7ab0cd40df1ac0fcf9c5fd0802d56bdec8c44ad2159c7755da3c8e0b8`.

````markdown
- [ ] Prove trusted teardown/recovery before reclaiming unknown-work handles or clearing a fence. Diagnostic service sync/readback permits inspection/exit only and cannot synthesize Ready, retire unknown events or authorize reuse.
````

<a id="memory-and-streaming-l171"></a>

#### open · [L171–171](../../plans/memory-and-streaming/TODO.md#L171)

Heading: `Memory and streaming > Tests and verification`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L171-L171@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `db84638f5bd0d2eae0a1332b4b19e07ea073b61860c3350dce38ef95cceb8ec3`.

````markdown
- [ ] SSD-01..32: run the [full source/residency suite](../../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) for each claimed backing/backend; mock tiers cannot qualify physical IO or trained streaming.
````

<a id="memory-and-streaming-l172"></a>

#### open · [L172–172](../../plans/memory-and-streaming/TODO.md#L172)

Heading: `Memory and streaming > Tests and verification`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L172-L172@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `23e6987cb8d1ed407d043a63b2b5cc9c912508025e8c0af6b24ac21355c5af8c`.

````markdown
- [ ] SSD-14/15: encoded-but-unsubmitted, queued and captured readers prevent generation-ABA slot reuse through actual GPU completion/cancellation.
````

<a id="memory-and-streaming-l173"></a>

#### open · [L173–173](../../plans/memory-and-streaming/TODO.md#L173)

Heading: `Memory and streaming > Tests and verification`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L173-L173@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `767e0a2b28716b41e7f996be77fe4fde650dda595800d1549932af2028e7f63d`.

````markdown
- [ ] SSD-21/22/24: mixed Q2/Q4 expert sizes share one byte cap with safe victims and fair progress; hot-cache retention cannot bypass admission.
````

<a id="memory-and-streaming-l174"></a>

#### open · [L174–174](../../plans/memory-and-streaming/TODO.md#L174)

Heading: `Memory and streaming > Tests and verification`.
Identity: `.agents/plans/memory-and-streaming/TODO.md:L174-L174@d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0`.
Block SHA256: `4a837613b566d377aec9d3568b4170a85277e8ff4dfc0645a69c7018664134ae`.

````markdown
- [ ] SSD-32: an authorized original trained streamed model matches complete resident logits/StateRoot; measure cold/warm IO, staging, OS headroom and genuine resource lifetime.
````

### model-acquisition — 18 noncompleted occurrences

<a id="model-acquisition-l299"></a>

#### open · [L299–310](../../plans/model-acquisition/TODO.md#L299-L310)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L299-L310@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `eb7a0b34ce96e7ef4a2a64da14d0ad18e3a71d7a9e02091be664649c9bba00c3`.

````markdown
- [ ] Canonical source-pull execution order (18 Oct decision): Qwen3.5 .8B/
  2B, Qwen3.8 27B; Gemma4 E2B/26B-A4B/31B, each exact chosen GGUF
  quants or indexed original safetensors; QwenImage2.1 original
  Diffusers components; Qwen3 TTS/ASR; Whisper; NVIDIA Canary and
  NeMo Parakeet; Qwen3.8 Flash-Next original plus separately sourced
  Q2/~3.5-bpw/Q4 quants and independently verified MTP/n-gram
  embedding banks. Do not guess literal Q3.5 quant tags or conflate
  runtime speculative n-gram with trained n-gram checkpoint tensors.
  Implement exact typed presets/component assembly and run real
  metadata-only then authorized actual payload/sidecar/checksum
  publication gates. DeepSeek V4.1 Flash and GLM5.3 Flash have an
  explicit NO PAYLOAD download hold; prepare metadata pipeline only.
````

<a id="model-acquisition-l311"></a>

#### open · [L311–317](../../plans/model-acquisition/TODO.md#L311-L317)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L311-L317@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `f13e5b1b1a209e9303e741ca37fe25b0fceeddea8f60efc1ffa984474430d175`.

````markdown
- [ ] Extend the new bounded HF/GitHub Auxiliary-only --bundle to
  source-verified Diffusers model_index nested shard sets, ASR/TTS
  processors/codecs, NeMo archive streaming/extraction and
  release-specific recursive dependencies under one pin/manifest.
  Current --bundle is a user-selected subset and cannot claim a
  complete model/codec assembly, support auto-loading, or run without
  newly qualified source parser and license checks.
````

<a id="model-acquisition-l319"></a>

#### open · [L319–319](../../plans/model-acquisition/TODO.md#L319)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L319-L319@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `a6c9985c51c9beb679bd6a150dc1389fb3891a07847b640aed706da5258ef21e`.

````markdown
- [ ] Extend cloud profiles beyond the narrow qualified AWS FULL_OBJECT SHA256 object route: broader S3 partitions/compatible vendors/credentials/checksum algorithms, Azure latest/SAS/managed chains and per-object digest-map prefixes. Keep explicit version, caller-SHA and conditional FirstSeen contracts distinct; never infer immutable versions or trusted raw hashes from unqualified validators or multipart COMPOSITE checksums.
````

<a id="model-acquisition-l320"></a>

#### open · [L320–320](../../plans/model-acquisition/TODO.md#L320)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L320-L320@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `6f99da26352cd5667c6e6d5b6f0a89058a38889558de331db48f3be12ac53cbd`.

````markdown
- [ ] Extend ModelScope only through separately qualified alias authority and redirect/CDN/mirror/credential profiles. Selected missing sizes now use exact pinned HEAD; no alias-to-commit authority was established from revision echoes, timestamps or file last-modified IDs. The current profile does not close broader parity or independently prove server revision/membership behavior.
````

<a id="model-acquisition-l321"></a>

#### open · [L321–321](../../plans/model-acquisition/TODO.md#L321)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L321-L321@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `5ab20ff0b6ec8f8b5fd0d7b01a7a0b2443d75e5fd71f7453bb6ee29f46e5b494`.

````markdown
- [ ] Measure and qualify native Xet fetch coalescing/chunk caching and any persistent native-prefix protocol. The current bounded per-intersection complete-signed-set reader has counted overfetch/refetch and no performance claim; native process interruption restarts a full transfer. Keep per-term/partial-xorb proof limits, fixed CAS scope and existing elapsed-time/HTTP-trailer boundaries explicit.
````

<a id="model-acquisition-l322"></a>

#### open · [L322–322](../../plans/model-acquisition/TODO.md#L322)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L322-L322@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `e8d58f52f4091e951ced3d0b7d356ab86cecd83bc2f9eb6cf2a0cfecfcc3423f`.

````markdown
- [ ] Close remaining finite convenience/compatibility gaps: required sidecar/adapter/sparse-quant policy, bounded auxiliary/presets, mirror/endpoint/env precedence and gate/license diagnostics. Preserve explicit source authority/no silent numeric fallback and separate loader/runtime plans.
````

<a id="model-acquisition-l324"></a>

#### open · [L324–324](../../plans/model-acquisition/TODO.md#L324)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L324-L324@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `b80318586130ff636aaa7729f69fe20440a317c798c951f918b84be4f346c092`.

````markdown
- [ ] Qualify any future v1 migration only after separate authorization and proven reader/process quiescence; current removal supports originally protected roots only.
````

<a id="model-acquisition-l325"></a>

#### open · [L325–325](../../plans/model-acquisition/TODO.md#L325)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L325-L325@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `37a0c88172d360c5110e9fa7b0cf25778cd746413f9704e5e6acc2aba6573512`.

````markdown
- [ ] Extend cleanup only through separately qualified exact ownership/recovery protocols for dead retained staging/general orphans and any later eviction policy. Current exact-snapshot operation deliberately retains them and every stable lock inode.
````

<a id="model-acquisition-l326"></a>

#### open · [L326–326](../../plans/model-acquisition/TODO.md#L326)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L326-L326@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `354aa12e07978d718b5c65391146270b5e8b5d8daca77a174790f49f3abc886d`.

````markdown
- [ ] Any additional lazy/raw-path/mmap consumer must adopt managed descriptor/open ownership or remain externally quiescent during removal. Observed named cache charge is not a process-global final-close/physical-free-space attestation.
````

<a id="model-acquisition-l327"></a>

#### open · [L327–327](../../plans/model-acquisition/TODO.md#L327)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L327-L327@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `2d4d42792061e303f10ea02947df3287853da06e2ed0ef6577585d8643a95924`.

````markdown
- [ ] Qualify additional filesystem pairs before extending the explicit allowlist. Different st_dev values alone do not prove independent backing objects for Btrfs subvolumes, overlay/FUSE or virtualized views; retain branch-specific root device/mount checks and complete inode/ancestry exclusion. No unsupported pair gains acceptance from generic mocks.
````

<a id="model-acquisition-l328"></a>

#### open · [L328–328](../../plans/model-acquisition/TODO.md#L328)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L328-L328@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `e2aa9c77358ea5b21c4a92baf182bbbfcf73426d701b4baf7f00bc7ee47615a4`.

````markdown
- [ ] Separately authorize any needed namespace-only hidden-bind/subvolume/overlay filesystem fixtures before mount operations. Real target-root/unselected inode aliases and hidden bind aliases in both directions remain NOT RUN; no host mounts, block devices, new privileges or live cache actions are implied. Current tests use already existing distinct filesystems only.
````

<a id="model-acquisition-l329"></a>

#### open · [L329–329](../../plans/model-acquisition/TODO.md#L329)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L329-L329@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `7455f7439f2869c66a788b90e003a8867d3840b3efb431c4d58d70259f1f5f14`.

````markdown
- [ ] Independently qualify any broader local layouts, renamed/unnamespaced roots, missing mount identities or new destination-root creation before exposing support. Keep explicit selection, retained descriptors, bounded topology work and existing cancellation/rollback limits; real import still needs user-selected source/repository/commit/files and copying authorization.
````

<a id="model-acquisition-l330"></a>

#### open · [L330–330](../../plans/model-acquisition/TODO.md#L330)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L330-L330@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `258618e66f5f45e51592c65c0deaebc0bb9d8fd43f1b71bbf5e82d25baa8cc38`.

````markdown
- [ ] Finish the finite provider/loader parity ledger beyond qualified HF/GitHub/HTTPS/OCI/GCS/versioned-S3/Azure/ModelScope and explicit local-HF import slices: remaining external-cache layouts/filesystems, additional model hubs, unversioned/broader S3/Azure/compatible profiles, native Xet and broader OCI profiles. Opaque serialization/plugin loaders remain separate runtime work.
````

<a id="model-acquisition-l331"></a>

#### open · [L331–331](../../plans/model-acquisition/TODO.md#L331)

Heading: `TODO — model acquisition > Uncommitted`.
Identity: `.agents/plans/model-acquisition/TODO.md:L331-L331@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `1bd475208508ac69b87a257b4bb9eb61ddc95a80a6facc91d9f3e25811f01e33`.

````markdown
- [ ] Continue explicit source/license authorization and trusted-digest versus first-seen provenance through additional providers/loaders; acquisition grants no license or loading support.
````

<a id="model-acquisition-l335"></a>

#### open · [L335–335](../../plans/model-acquisition/TODO.md#L335)

Heading: `TODO — model acquisition > Tests and verification`.
Identity: `.agents/plans/model-acquisition/TODO.md:L335-L335@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `e7c3595304615dbf61513aae0d4cfcaa1fab827057c01026b514e542af91a178`.

````markdown
- [ ] ACQ-A12: complete the finite provider/loader parity inventory and independently qualify remaining supported paths.
````

<a id="model-acquisition-l348"></a>

#### open · [L348–348](../../plans/model-acquisition/TODO.md#L348)

Heading: `TODO — model acquisition > Tests and verification`.
Identity: `.agents/plans/model-acquisition/TODO.md:L348-L348@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `0b1a3ce05bee417c1380bdd3b8b348283f40a0761be297d05deab8911a6dccdd`.

````markdown
- [ ] Each remaining provider phase requires focused API/CLI/protocol/auth/version/range/offline/fault gates, full frozen isolated check.sh, relevant source validation, actual source-integrity and documentation/dependency/root README checks before signing.
````

<a id="model-acquisition-l352"></a>

#### open · [L352–352](../../plans/model-acquisition/TODO.md#L352)

Heading: `TODO — model acquisition > Deferred`.
Identity: `.agents/plans/model-acquisition/TODO.md:L352-L352@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `bcbe663e6b1320eed162f1ff2c3dbe284c31aaa7a50ff4d8d77d458c870436de`.

````markdown
- [ ] Unsupported plans fail closed until qualified: unknown-length required/model-closure files, capped resumable transfers, unqualified external-cache layouts/filesystems and remaining cloud/OCI profiles, legacy-root removal and unmanaged-reader concurrent reclamation remain explicit above. Independent full HTTP trailer validation is unqualified under the documented decoder boundary, rather than a reason these payloads universally refuse. Independent component data grants no additional loader eligibility.
````

<a id="model-acquisition-l353"></a>

#### open · [L353–353](../../plans/model-acquisition/TODO.md#L353)

Heading: `TODO — model acquisition > Deferred`.
Identity: `.agents/plans/model-acquisition/TODO.md:L353-L353@861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86`.
Block SHA256: `bdb59d13d3ae27849b2c707e96afb0af2cfe1c15e26a83d1b292f3551f239946`.

````markdown
- [ ] Linux local-filesystem semantics are qualified first. HTTP cancellation obeys configured whole-request bounds; old retained attempts consume real quota and arbitrary same-UID mutation remains outside descriptor isolation.
````

### model-execution — 46 noncompleted occurrences

<a id="model-execution-l13"></a>

#### ~ · [L13–17](../../plans/model-execution/TODO.md#L13-L17)

Heading: `Model execution > Uncommitted > Current owner direction — 26 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L13-L17@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `1cdda01eb9f590a0a89d8a67cfe3c008002bb941e99d6541ac1c26051560d605`.

````markdown
- [~] Every existing/future registry target follows the shared
  [correctness policy](../numerics-and-context/CORRECTNESS_POLICY.md): declared
  per-op accuracy, pinned independent task/model quality, separate determinism
  and exact source/state/physical ownership. Registry metadata is no execution
  or model-access authorization; unsupported adapters remain unqualified.
````

<a id="model-execution-l18"></a>

#### open · [L18–41](../../plans/model-execution/TODO.md#L18-L41)

Heading: `Model execution > Uncommitted > Current owner direction — 26 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L18-L41@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `2e4de0ee31ee0f4eed2b89e39121b9e0688b31b3a4a3eba0aee127e5b1155a9d`.

````markdown
- [ ] Implement/qualify missing full-model/context/backend cells, beginning
  .8B47–50 then genuine long context on CPU/CUDA/ROCm/Vulkan. ROOT coordinates
  exclusively GPT-6.1 Sol xhigh lanes. Older Pi/Astra assignments and performance
  sequence below are superseded; no new timing/27B capture precedes correctness.

Current24Oct: CLI and shared model units are integrated; Pi implements/executes,
Sol coordinates/reviews/merges, Astra designs/reviews architecture and math.
Local source21 shared state-certificate preparation passes the complete1793 CPU
gate, combined strict, five readiness contracts and optimized release build.
Current physical23/full2Bmath and both .8B/2B API/HTTP pass ROOT review;
current .8B/2B C1/C4/C8 singlepass rate AND wall wins are accepted. These
finite scopes do not grant fullmatrix/generalparity. Source20 x2's complete
shared HIP unit passes below. Independent Qwen natural feedback and learned
Gemma PLE hit/transfer coverage remain open; next whole-unit design proceeds in
parallel with local .8B→2B benchmark work. All78 signed commits are complete.
Latest user direction resumes all tracks now: separate Pi Qwen27B/Gemma4/
Vulkan/HIP/queue lanes alongside external C16 correction. C32/C64 follow shared
queue acceptance; physical jobs and quiet timings retain coordinated ownership.

Latest owner merge direction: complete the current Qwen27B/Gemma4/HIP/Vulkan/
queue units to a tested, mergeable checkpoint, assemble and reconcile shared
source, then integrate into main and retire completed worktrees. Keep original
reference roots while current checks use them. Broken source or a draft
controller is not a merge checkpoint; preserve pending physical scopes honestly.
````

<a id="model-execution-l46"></a>

#### ~ · [L46–61](../../plans/model-execution/TODO.md#L46-L61)

Heading: `Model execution > Uncommitted > H/G/V physical checkpoint — 25 Oct pass 3`.
Identity: `.agents/plans/model-execution/TODO.md:L46-L61@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `f949b437fde2ebc39c3266f1f47ea12ed0ebbd1744d4621a1d54c3d159dd7617`.

````markdown
- [~] Three passes executed on pvl-x2 (full evidence:
  ../../runs/resume-all-hgv-checkpoint-execution-sol-20261025/LOG.md).
  Root causes of passes 1-2 fixed and verified on hardware
  (`06b89da` E0308, `c217c35` V parents, `e6ab074` census,
  `3941a3c` Gemma envelope). Pass 3: builds/admins/binds all PASS;
  h-known-qwen-reload + h-qwen-final-reader-unknown PASS (full
  semantic closures); three real failures preserved
  (h-gemma-feedback-unknown: session-state identity/policy/fence
  assert + `scope closes only between completed proposals`;
  g-resident: parity row-upload fixture contract vs legitimate PLE
  row dedup; v-resident: adapter validator requires a false-permit
  field the producer never emits — physical case itself passed);
  19 cases stopped as dependents. Fix sites pinned in
  [the PM handoff](../docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md) §2;
  pass-4 procedure §2 there. No product defect established
  (qualification-gated paths).
````

<a id="model-execution-l65"></a>

#### ~ · [L65–73](../../plans/model-execution/TODO.md#L65-L73)

Heading: `Model execution > Uncommitted > Current shared source20 hardware checkpoint — 24 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L65-L73@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `e8d5505c9d1457a4954940bd2a10aee591c84e4c0bc9630eba20f760917271d5`.

````markdown
- [~] Qwen natural2-position independent original comparison and Gemma learned
  PLE row-hit/H2D/full-state scope, retaining existing quotas/owners. ROOT reviewed
  Astra's complete next-unit design; two isolated Pi implementation/controller
  preparation lanes active, no GPU/reference execution before review. Qwen old
  teacher capture requires independently verified original fullV/prior/source
  join; otherwise fresh natural reference. Gemma reuses its genuine natural
  matrix and adds an actual-hit full35/fullV comparison plus final-owner scopes.
  No new public
  serving, source21 x2 artifact, general-fit or speed grant from source20 fixture.
````

<a id="model-execution-l74"></a>

#### open · [L74–77](../../plans/model-execution/TODO.md#L74-L77)

Heading: `Model execution > Uncommitted > Current shared source20 hardware checkpoint — 24 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L74-L77@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `4af55848e4b9f7b4f04f8fa6c00d70e01f86bc380193df68bd0f6a6a7dee0300`.

````markdown
- [ ] Gemma PLE twofixture source draft and prepared controller need corrective
  review before merge/physical work. ROOT finds diagnosticD2H exactly1B gate
  instead of positive and regressed dataFD/bounds. Original valid source20
  math/lifetime results unchanged; new all-hit/final-reader scope unexecuted.
````

<a id="model-execution-l82"></a>

#### open · [L82–92](../../plans/model-execution/TODO.md#L82-L92)

Heading: `Model execution > Uncommitted > Qwen27B and Gemma4 optimization audit — 24 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L82-L92@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `099bd4c7ee44b21b67961000c9aeb0014c7fae1fd001abb8ecbc8d51ae554ce5`.

````markdown
- [ ] Close isolated source113 as one checked model unit, then actual x2 source/
  state/fault/lifetime qualification: Qwen full64/all112, Gemma resident20/all35/
  fullV/all15KV and observer-free natural feedback. Grouped isolated CPU/strict
  checks PASS:5strict configurations,907CPU library tests/2comparators,46explicitly
  ignored cases. Exact source113/P1/P2 and original400v4 base remain unchanged;
  contextual410 continuous/compiler reconciliation is reviewed and affected
  strict/166boundary/7compiler-owner checks pass with all jobs closed;21physical
  cases explicitly ignored. Later RMS3/source6 excluded. Gemma's finite actual
  Resident20 and Qwen full64/armed-producer fault results follow below; Qwen's
  final Pending/Snapshot reader fault coverage is now closed on frozen469 below.
  Keep current C8 assembly separate.
````

<a id="model-execution-l93"></a>

#### open · [L93–95](../../plans/model-execution/TODO.md#L93-L95)

Heading: `Model execution > Uncommitted > Qwen27B and Gemma4 optimization audit — 24 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L93-L95@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `b80cb5c27a9396c11c28e27b4d74d0568344ecb7ce88ba50e11c6a54876d1e86`.

````markdown
- [ ] Reconcile frozen models113/P1 hooks and Gemma final-feedback fix with current
  shared CUDA readiness source, preserve arithmetic/backend boundaries, perform
  affected assembled checks and final complete gate before model merge acceptance.
````

<a id="model-execution-l96"></a>

#### open · [L96–101](../../plans/model-execution/TODO.md#L96-L101)

Heading: `Model execution > Uncommitted > Qwen27B and Gemma4 optimization audit — 24 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L96-L101@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `6ab78736ed669b84ed8a329d074044b19e4e05d9c4e813eb019cdf8538e80239`.

````markdown
- [ ] Qwen27: qualify natural and temporal decode/prefill, residency/admitted
  prefetch, packed projections, eligible optimized GEMM/attention/recurrent GDN,
  hybrid prefix checkpoints and serving through shared engine primitives.
  Current native route requires streamed weights and finite teacherfrontiers0/1;
  shared64-lane CPU-order BF16 dot is not TensorCore/BLAS. Preserve source/cast/
  reduction profiles rather than inheriting small-model CUDA permits.
````

<a id="model-execution-l102"></a>

#### open · [L102–146](../../plans/model-execution/TODO.md#L102-L146)

Heading: `Model execution > Uncommitted > Qwen27B and Gemma4 optimization audit — 24 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L102-L146@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `bfe8fbf561beb4cb0d9fd4adc6b1481f1d6548aa80b73fbad69999aa188c575a`.

````markdown
- [ ] Gemma E2B final-reader ownership: source466 review finds Session/Snapshot
  drop `loaded` before final committed feedback's independent12-byte Output
  lease. Native/modules/managed-source guard can retire before a true final
  feedback release refusal. Pending already has safe ordering. Isolated Sol
  fix+fixture unit authorized: place `loaded` after inner+feedback; Session/
  Snapshot × healthy/exact final Output-lease refusal, Weak-only diagnostics and
  original actual service/native/source/quarantine owners. Preserve frozen466,
  completed Qwen P1 and readiness sources. Frozen469 ten-path fix+fixture passes
  grouped6control/4probe/all-target strict checks; independent Sol review and ROOT
  complete source/control/cost/archive review pass. ROOT authorized one bounded
  fresh admin/offlinebuild/four-case unit. Admin33537/offline GPU-free build73520
  CLOSED0; first Session-known28952 FAILED_CLOSED at semantic verifier's exact
  `device_with_six_roots_bytes` formula after native child exited0. Actual groups/
  one-reap/fullEOF/source35FD/cache-parent closure healthy. ROOT and independent
  Sol identify verifier-only omission: greedy12336B+six256B alignment guards
  missing from both exact expected totals; production sums/caps are correct.
  Corrected local verifier6controls PASS; new saved Session-known revalidation
  PASS without GPU replay, original FAILED receipts unchanged. ROOT reviewed
  full225line local collector delta/17pins and authorized remaining3 serial
  native cases, exact-known-error-only/source35/cache/binary/originalowner
  closure required. All4actual scopes now close with corrected local acceptance;
  ROOT independently verifies every report/source/cache/original+reader owner.
  Known:184→0modules/sourcecloseOK/model0/context4GiBdevice+2GiBhost. Fault:
  actual12Bfinalreader refusal precedes native retirement,184modules/617handles
  retained/full20GiBfenced/sourceclose refused. Every full35/30planes/fullV/
  one natural commit; original failed remote collectors and SSH1 remain explicit.
  No Session replay/Rust rebuild/serving or performance grant. Existing source466
  cache reused with only actual publication/activity locks writable; no second
  10GB import. Current11+118path isolated combined8package/alltarget strict and
  189CPU PASS/33explicitphysicalignored; original source/owner/EOF closures pass,
  first controller-only compilerfile cap failure preserved. Current12 differs
  only by test-fixture rustfmt outside118. ROOT now adopted models118 plus the
  source-bound2B cachedB4 unit together:173selected/172effective/46new paths,
  fresh mtimes, original README/lock and all unrelated source preserved. Current13
  has1629full files/492selected; combined strict/20sharedCPU/5ABI-source controls/
  authentic2B CPU inventory/reopen pass. Original excluded-vendor Cargo refusals
  remain failed, not executed vendor-test evidence. Optimized13 CLOSED0 and all
  original19physical scopes/healthy2Btrajectory PASS; original2B same-prior30calls/
  8112sections/fullstate/fullV exact. New2B fault global-vs-flight count mismatch
  is diagnosed; fixture-only15 correction builds, preserving every graph/source/
  fence guard. Current assembled phase and final
  assembled gate still required. Prior.8B measured checkpoint remains source11/12.
  [Combined assembly results](../../runs/model-execution-current11-assembled-checks-sol-high-20261024/REPORT.md).
  [Four-case report](../../runs/model-execution-gemma-final-feedback-verifier-v2-sol-high-20261024/REPORT.md).
  [Independent source evidence](../../runs/model-execution-gemma-final-owner-review-sol-high-20261024/REVIEW.md).
````

<a id="model-execution-l147"></a>

#### open · [L147–150](../../plans/model-execution/TODO.md#L147-L150)

Heading: `Model execution > Uncommitted > Qwen27B and Gemma4 optimization audit — 24 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L147-L150@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `39a6371e5b493ec2c39b5f231f2f7927a3ed3e863d098a5f224d3e53008990be`.

````markdown
- [ ] Gemma E2B/E4B: dependency-based shared-tail fast prefill, PLE row reuse,
  optimized packed projections and eligible attention, physical SWA reclamation.
  E2B15owners/20shared tail exists; E4B needs24/18 and a new functional route.
  Optional vLLM fast prefill is off by default, not a universal baseline feature.
````

<a id="model-execution-l151"></a>

#### open · [L151–152](../../plans/model-execution/TODO.md#L151-L152)

Heading: `Model execution > Uncommitted > Qwen27B and Gemma4 optimization audit — 24 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L151-L152@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `9bd7381c369617390981e40e8010993a4725fc46c63caa84452f2e7c91e57d70`.

````markdown
- [ ] Add functional Gemma E4B and12B routes;12B uses Unified geometry/MM mapping,
  no PLE/shared tail. Extend source/profile contracts before learned qualification.
````

<a id="model-execution-l153"></a>

#### open · [L153–159](../../plans/model-execution/TODO.md#L153-L159)

Heading: `Model execution > Uncommitted > Qwen27B and Gemma4 optimization audit — 24 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L153-L159@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `ac6f455d59d2d7ba6bc54bb8310a8ced5f3434d44cd9f51694461a5deae2ec0b`.

````markdown
- [ ] Gemma26B-A4B: learned router/grouped experts/dense+expert composition, then
  admitted expert sharing/hotsets/EP. Gemma31B is dense; qualify its learned dense
  path separately. Reuse the existing single raw K projection for configured
  K=V layers, preserving distinct K/V transforms and state. Select the exact
  optimized-router tie/reduction/scale profile; full-softmax fallback is not an
  identical floating-point schedule. Geometry/analytic contracts alone do not
  implement either learned route.
````

<a id="model-execution-l160"></a>

#### open · [L160–163](../../plans/model-execution/TODO.md#L160-L163)

Heading: `Model execution > Uncommitted > Qwen27B and Gemma4 optimization audit — 24 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L160-L163@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `8204f68781df1238c03fceadfd61a40997557eb4b72fab29de26e6b7162b2987`.

````markdown
- [ ] Both families: shared continuous serving/temporal work, paged KV/COW,
  full-domain compatible prefix intersection and graph/capture ownership after
  actual source qualification. Shared target verifier/all-prefix rollback before
  Qwen trained MTP or Gemma assistant/KV sharing/sparse draft vocabulary.
````

<a id="model-execution-l164"></a>

#### open · [L164–168](../../plans/model-execution/TODO.md#L164-L168)

Heading: `Model execution > Uncommitted > Qwen27B and Gemma4 optimization audit — 24 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L164-L168@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `f78436887ea1a6e3bb3642cc99c5a6ae78a6353ab302d4b3ddfdaded199da048`.

````markdown
- [ ] Later explicit codec/distributed/MM/LoRA profiles: do not substitute lossy
  FP8/FP4/KV codecs for original BF16 or assume source/hardware eligibility.
  CUDA-only fusion and CDNA/gfx12 AITER are not unconditional gfx1151 gaps;
  current ROCm Triton speculative GDN needs actual eligibility checks. Generic
  MXFP4 registry falls back for dense Linear; DBO rejects relevant hybrid/MM routes.
````

<a id="model-execution-l256"></a>

#### open · [L256–261](../../plans/model-execution/TODO.md#L256-L261)

Heading: `Model execution > Uncommitted > Parallel model work — contained integration, 22 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L256-L261@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `f65a8600a6a699004255c93a66d72d48d2d2545e11bb61def149f0d4eeea8d5b`.

````markdown
- [ ] Continue Qwen27B/Gemma source/runtime integration independently; merge
  contained completed units promptly. Use pvl-x2 Radeon8060S/gfx1151 ROCm/HIP
  for their GPU qualification/benchmarks; preserve exact HF source identities,
  source-derived complete allocations and full logits/state/cancellation gates.
  Existing remote Ollama GGUF inputs are not approved-HF equivalents. Current
  Vulkan/CUDA parity work is a separate implementation/qualification obligation.
````

<a id="model-execution-l262"></a>

#### open · [L262–264](../../plans/model-execution/TODO.md#L262-L264)

Heading: `Model execution > Uncommitted > Parallel model work — contained integration, 22 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L262-L264@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `623f1330a654df2785bbcee40002260c2bea23648f455f82fa61de75598583ef`.

````markdown
- [ ] Gemma follow-on: qualify tokenizer/ordinary learned route and broader
  prompt/cached geometry plus learned cancellation/state lifecycle through the
  same engine; selected diagnostic exactness alone grants no public route.
````

<a id="model-execution-l265"></a>

#### open · [L265–268](../../plans/model-execution/TODO.md#L265-L268)

Heading: `Model execution > Uncommitted > Parallel model work — contained integration, 22 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L265-L268@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `3a26c9157e6fc146ed2a1f170e4f4b66553e17e3c9dfca3f47e88a084ad8fce7`.

````markdown
- [ ] Independent follow-on units continue while short-HF validation runs:
  Qwen authentic H48/128 GDN producer and full64/112-domain schedule, plus Gemma
  family source-bound CPU completion. Preserve actual source geometry and exact
  numerical limits; unavailable learned arithmetic must not use legacy fallback.
````

<a id="model-execution-l269"></a>

#### open · [L269–271](../../plans/model-execution/TODO.md#L269-L271)

Heading: `Model execution > Uncommitted > Parallel model work — contained integration, 22 Oct`.
Identity: `.agents/plans/model-execution/TODO.md:L269-L271@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `6b84b09eff98f2ff16b2d3995bb1d5d21efc25f74576e67096d61b018675f602`.

````markdown
- [ ] Gemma trained Rust execution requires exact learned BF16 arithmetic and
  independent source/state comparison. Payload/tokenizer/original CPU oracle
  now exist; native/serving/performance acceptance remains separate.
````

<a id="model-execution-l280"></a>

#### open · [L280–283](../../plans/model-execution/TODO.md#L280-L283)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L280-L283@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `86c85662c193721eb56137ef172752d303f2d86bd277fb3fc3140d9f97edafd5`.

````markdown
- [ ] Dense27B: bind full trained source/state through bounded row/layer windows;
  amortize immutable reads across compatible prompt/request rows, account
  activation spill and genuine upload readiness. MTP metadata/tensors alone
  do not qualify target or drafter runtime; expert hotsets do not sparsify it.
````

<a id="model-execution-l284"></a>

#### open · [L284–288](../../plans/model-execution/TODO.md#L284-L288)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L284-L288@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `b7ac6297ddd65cf0d3b150da59a2a7c795784519e94a79c796f0afbd40bac6ec`.

````markdown
- [ ] Gemma4 variants: source-real PLE, normalization/logit caps, SWA/global
  attention, configured K=V/cross-layer sharing, head256/512 and26B-A4B routing.
  Check graph padding cannot route experts or advance roots. Matching trained
  assistants may share target KV/embedding only under actual architecture maps;
  first qualify target execution and complete branch/precision ownership.
````

<a id="model-execution-l289"></a>

#### open · [L289–292](../../plans/model-execution/TODO.md#L289-L292)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L289-L292@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `d28fc8611b8d812e09648c5e843fafa9a35101257fe8dc2080df7b8f4c15ffa1`.

````markdown
- [ ] Later sparse Qwen: stable finite top-k/tie/block ordering, source-selected
  expert grouping, hyperconnections and deterministic learned PLE lookup lane;
  runtime n-gram proposals are a separate stage. Source-dependent sparse masks
  and reduction/cast profiles cannot be grafted onto ordinary dense attention.
````

<a id="model-execution-l293"></a>

#### open · [L293–297](../../plans/model-execution/TODO.md#L293-L297)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L293-L297@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `60e73c73441f3a4aa5b7112d70a97c6460a487b101ebf537ddfffdb47de6e28a`.

````markdown
- [ ] Later DeepSeek: exact CSA/HCA/MLA/indexer/compressor/mHC/YOCO/Engram and
  checkpoint-matched DSpark/Markov/confidence contracts; typed dependency/liveness
  proof may eliminate dead prompt work only after every future state read is
  covered. Truncated bounded replay and perturbed sampling are explicit changed
  profiles, not exact defaults. Existing payload acquisition hold persists.
````

<a id="model-execution-l298"></a>

#### open · [L298–301](../../plans/model-execution/TODO.md#L298-L301)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L298-L301@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `1c08cf75eec3dea86d666f008f1759edede9001820a25c1ff75bea5c19fe7b6d`.

````markdown
- [ ] Later GLM: distinguish KDA from GDN, MLA/DSA index/open-tail state, mHC and
  own trained MTP/taps. CPU/GPU placement, cached-expert group order and FP16
  prefill must have declared arithmetic; geometry or source names do not prove
  interchangeability. Existing payload/device holds persist.
````

<a id="model-execution-l303"></a>

#### open · [L303–357](../../plans/model-execution/TODO.md#L303-L357)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L303-L357@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `5395c64a415375d5b85368510de4d0af7d26a0a3cde52345b279df0a564b8fb5`.

````markdown
- [ ] **Independent trained BF16/FLA checkpoint gate:** original
  Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 acquired
  through the existing bounded Rust cache pipeline; independent31-file/
  18-shard full hashes and1199-tensor header/index checks passed.
  Whole-model27B execution remains unqualified; bounded trained operator
  qualifications below do not establish full logits or112 StateRoot domains.
  Genuine original layer0 GDN reference now validates initial chunk64 and true
  cached continuation for teachers11/353,48 value heads and original BF16
  parameters. Plain/observed complete outputs, convolution/recurrent state,
  raw source readsets, full before/after source hashes and original owned
  retirement match. The serializer's V2 stride failure remains preserved;
  V3 raw-bit transport passes12 real PyTorch layouts. This is an independent
  public target. An isolated native composition now passes the authentic H48
  cached slice:804960 input and798720 output words, including the complete
  recurrent state, match independently before/after callers drop. The distinct
  BF16-gamma gated RMS slice also passes61632 intermediate,12288 final and
  24576 retained input words. Source hashes, original ownership, cancellation,
  real pressure and final reservations0 pass; strict/release builds and237
  focused tests pass. These operator changes remain isolated from MAIN;
  initial chunk64, native exp, complete GDN,112-domain state and full inference
  remain unqualified. A separate genuine public reference passes32 pure
  operator cases over16 roles,881856 raw BF16 input/output words, including
  complete248320-row head and17408-column down projection. The isolated
  native wide-role replay now matches all881856 words exactly; real pressure,
  pre-cancel, retained head/source after caller drop and final reservations0
  pass. Strict/release builds and17 source/relocation fixtures pass. External
  probes for other roles do not establish trained producers or full-model
  parity; these changes also remain isolated from MAIN.
  Six actual bounded native-provider startup/lifetime cases pass with the
  provider unloaded. Physical provider allocation/math admission remains open.
  Original
  Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc is now acquired
  by the same bounded Rust pipeline; independent official Git/LFS full12-file
  hashes,632 headers/320 text shapes,596BF16+36F32 tensors pass. Its4.548GB
  shard remains above the default4GiB complete-source guard. Genuine pinned
  official CPU two-profile reference capture now passes archive and owned
  cleanup validation: tokens11/353, four full248320 logits,168 canonical
  domains and60 actual intermediates. Official BF16 loading has617 BF16
  parameters; restoring only36 original F32 tensors gives581 BF16+36 F32.
  The profiles differ in47/49 raw tensors per step despite identical argmax;
  neither establishes Rust2B HF numerical qualification. First capture's
  cleanup-name bug is preserved; the repaired fresh run proves actual nonce
  container absence and process reap. Original Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17 pinned
  Transformers5.17/Torch2.13 CPU captures passed archive validation:
  tokens11,353, full248320 logits/all42 domains, actual layer0 stages.
  Keep official473-BF16 loader and source437-BF16+36-F32 profiles distinct;
  first-token argmax321 vs198. Explicit new CPU64 diagnostic vs source-mixed
  reference remains UNQUALIFIED: embeddings/direct layer0 norms and both
  QKV projections exact; layer0 convolution state exact,41/42 domains still
  differ. Full248320 logits differ214245/214285 words, maxabs .22119140625/
  .1875. First token exposes conv/SiLU signedzero; second exposes GDN state.
  Complete pinned convolution, chunk64/cached recurrent, coefficients and
  gated RMS arithmetic, then original GPU/FLA/state/streaming gates before enabling
  trained HF execution or27B serving. User authorizes required runtime/model
  acquisition/testing; dated source-free slice restrictions below are history.
````

<a id="model-execution-l358"></a>

#### open · [L358–364](../../plans/model-execution/TODO.md#L358-L364)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L358-L364@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `e3b35349a5c5cc709a5403bcca5822074e5e30228233287c705ccd121f63065f`.

````markdown
- [ ] **Next trained reference gate:** qualify exact original HF BF16
  rounding/FLA order and independent source-trained logits/StateRoot;
  then bind the original profile to the existing per-op
  MathematicalBackend/SessionAuthority, with full physical
  device/SSD resource admission. Earlier source-free slices establish no
  full27B logits,112-domain state, native provider admission or GPU support;
  current owner authorization covers the required acquisition and testing.
````

<a id="model-execution-l366"></a>

#### open · [L366–373](../../plans/model-execution/TODO.md#L366-L373)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L366-L373@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `23376a38bab633032b6d2ee30f3a4b75859666370dcd4aecbfb5f7483e30cb96`.

````markdown
- [ ] **Following ME-P0 real trained integration:** fuse the
  accepted analytical original source operators into the ONE
  existing forward_token/MathematicalBackend and
  SessionAuthority execution schedule, qualify original HF BF16
  intermediate casts and FLA GDN accuracy, source-trained exact
  full logits and StateRoot, tokenizer, physical mutable-state
  memory accounting, cancellation and device completion. No
  SSD/physical inference claim follows from the diagnostic bridge.
````

<a id="model-execution-l375"></a>

#### open · [L375–381](../../plans/model-execution/TODO.md#L375-L381)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L375-L381@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `996c516f4207b72047f28263bc411a0741e7107b7dbc20e2ad6a40ea730a5c8d`.

````markdown
- [ ] **ME-P0 next after diagnostics:** wire the analytically verified
  Qwen original operators into the ONE existing forward-plan and
  TaskDriver/SessionAuthority (no second token loop or state issuer),
  with exact source BF16 intermediate cast and original FLA GDN kernels,
  real trusted tokenizer and full 248320 logits/42-or-more StateRoot
  domains against the pinned checkpoint. Only then begin
  SSD-resident streaming and measured Qwen3.8-27B physical throughput.
````

<a id="model-execution-l383"></a>

#### open · [L383–389](../../plans/model-execution/TODO.md#L383-L389)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L383-L389@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `51ae5cb6d0c1465c33f01de6623ed0e3011a986a2140165a2906f3c210278236`.

````markdown
- [ ] **ME-P0 next numerical stage:** source-exact original Qwen
  full attention, GDN/MLP, grouped Q/K→V replication, A_log/gate
  transformation and BF16 intermediate numerical schedule using the
  shared original TaskDriver/StateRoot. Qualify tiny independent
  analytic operator/state gates before real approved trained Qwen3.8
  27B original full-vocabulary reference. Measured original Qwen3.5
  2B CPU/GPU end-to-end performance also remains open.
````

<a id="model-execution-l391"></a>

#### open · [L391–398](../../plans/model-execution/TODO.md#L391-L398)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L391-L398@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `b521dd2cab14f54032139bd0d01c551db253cdb082769a20d9fe0255eaff1844`.

````markdown
- [ ] **ME-P0 next:** bind source-approved original Qwen3.8 27B
  tensor/index/config and tokenizer to the existing mathematical engine
  with authentic head/axis geometry, strict trained CPU full logits/
  StateRoot, then source-verifiable immutable SSD extents with stable
  leased memory and measured decode/prefill. First finish measured
  Qwen3.5 2B CPU/approved GPU end-to-end performance and
  cancellation/physical readiness. Preserve no model/SDK downloads
  and held device launch policies without explicit authorization.
````

<a id="model-execution-l400"></a>

#### open · [L400–411](../../plans/model-execution/TODO.md#L400-L411)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L400-L411@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `5611b6a2ff5bbc3cdcb089cedd10615022f644ae2dffc7d08224c90c7726f7ab`.

````markdown
- [ ] **ACTIVE NEXT: ME-P0 Qwen3.5 2B then Qwen3.8 27B.**
  Begin with original qualified Qwen3.5 .8B/2B CPU and resident GPU
  source/state oracles, inspect release source ownership and exact
  model-2B profiling on the permitted CPU/GPU devices. Keep tests
  bounded to existing 8K context now; 16K/32K extensions remain
  separate numerical gates. Qualify a stable single-model fast path,
  original tokenizer/math/KV/recurrent/StateRoot and actual device
  completion first. Then implement checked immutable source shard leases
  above the present 4GiB executable-source gate and Qwen3.8 27B
  complete trained CPU logits, including SSD streaming and quant-aware
  physical placement. Do not fetch weights or launch held Renoir/gfx90c
  hardware without the corresponding permission and source integrity.
````

<a id="model-execution-l412"></a>

#### open · [L412–420](../../plans/model-execution/TODO.md#L412-L420)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L412-L420@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `36122c86e1272fe627400d81b0e3070d72dd6261be4f69f14fa28fca08ea8b57`.

````markdown
- [ ] **NEXT after Qwen: ME-P1 Gemma4 E2B→26B-A4B→31B**, independently
  source-bind tensors/reference semantics, PLE/MoE/dense quantization
  and trained CPU→GPU correctness before measured throughput
  optimization. **ME-P2 Qwen3.8 Flash-Next** follows with separately
  verified learned n-gram embedding/4B MTP heads, QSA/GDN/MoE and
  explicit Q2/~3.5bpw/Q4 source quant choices (NOT guessed files),
  speculative n-gram as a distinct runtime optimization. Other OSS,
  ASR/TTS/image/video/decision execution and multi-model router remain
  on their own deferred source-real gates.
````

<a id="model-execution-l421"></a>

#### open · [L421–426](../../plans/model-execution/TODO.md#L421-L426)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L421-L426@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `8cd3bd8d704d026f5a4fd823c41cea06711bc17e961221e89fe4820383c8cafe`.

````markdown
- [ ] Preserve direct one-model-per-process execution and one
  shared-instance-many-pipelines as **one Rust fabric**; the accepted
  YAML DAG/compiler and extension schemas add no inference backend.
  Latest21Oct owner request authorizes parallel recursive pipeline/profile
  design after independent engine research. It must not delay ME-P0/P1 or
  imply new trained context/media/router execution.
````

<a id="model-execution-l428"></a>

#### open · [L428–435](../../plans/model-execution/TODO.md#L428-L435)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L428-L435@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `ed9f54d5f3579a8716a5e709dfb27fbff3a8ddfab96e9bf57070c72eba3c76fa`.

````markdown
- [ ] Enforce new owner priority: Qwen3.5 .8B regression, Qwen3.5
  2B highest-perf trained CPU/GPU, Qwen3.8 27B with SSD
  streaming/owned shard tensors, Gemma4 E2B, 26B-A4B, dense 31B,
  then Qwen3.8 Flash-Next QSA/GDN/MoE/gated residual/MTP
  and 51B learned n-gram embeddings. Every additional DeepSeek,
  GLM/Muse/Mistral/Llama family has a typed NOT IMPLEMENTED
  source contract until distinct trained CPU and GPU gates pass.
  No subfamily is generalized from metadata-only source support.
````

<a id="model-execution-l436"></a>

#### open · [L436–441](../../plans/model-execution/TODO.md#L436-L441)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L436-L441@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `fc4543c84488ccdd16c49c492ea6e13e0e61f0e61ee5714d80df42417fd71d9c`.

````markdown
- [ ] Treat single-model local GGUF execution and multi-model
  pipeline-qualified execution as two frontends of ONE TaskDriver/
  SourceIdentity/StateRoot fabric. A reusable model instance
  serves multiple routes without another model process or implicit
  source promotion; named replicas/adapters require separate
  physical memory admission.
````

<a id="model-execution-l443"></a>

#### open · [L443–458](../../plans/model-execution/TODO.md#L443-L458)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L443-L458@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `ff08ff829e3ab0abda9693feab07d16d243937d499d3a8dac79fb236a0292582`.

````markdown
- [ ] After the accepted source-acquisition v1 cutover, use the existing
  pinned HF index-first path to inspect explicit original Qwen3.8-27B and
  Gemma4 26B-A4B index/config/tokenizer/chat/processor identities
  (see [acquisition v1 handoff](../model-acquisition/README.md)).
  Select an authorized exact revision and admitted local host. Do not treat
  synthetic >51GB transfer-plan tests, public model pages or cached bytes
  as authorization, execution-ready source, or trained CPU parity.
  2026-10-18 tokenless live metadata-only HF source observation resolved
  Qwen3.8-27B to `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  (18 shards, 55,586,113,293 selected bytes) and Gemma4-26B-A4B-it
  to `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
  (2 shards, 51,644,338,194 selected bytes); Gemma4 E2B-it commit
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
  (1 unsharded safetensors, 10,278,848,001 selected bytes).
  No trained weights downloaded, locally owned, header-audited or
  model-execution-qualified yet.
````

<a id="model-execution-l460"></a>

#### open · [L460–460](../../plans/model-execution/TODO.md#L460)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L460-L460@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `ecce834eaf1649bd6f9012c8efa7b31d75827b9f60401f74ad87c256ce7851b1`.

````markdown
- [ ] Acquire/bind authorized original config/index/tokenizer/checkpoint identities and choose a capable admitted host. Large BF16 sources exceed current local RAM; current executable source limit stays 4 GiB.
````

<a id="model-execution-l461"></a>

#### open · [L461–467](../../plans/model-execution/TODO.md#L461-L467)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L461-L467@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `9204cf6bf7d607dbac6c0f8a5eb86d62fd11a0e0e969ac932ef5a2d378a2e63f`.

````markdown
- [ ] Qualify real large (>4GiB) original trained shard payload on an
  authorized capable host using the already verified owned-shard
  implementation from 93a5622; independently measure resident
  allocation/physical RAM pressure and immutable lease lifetime. Next,
  develop the separately admitted SSD streaming extent architecture.
  The bounded source-only synthetic tests do not establish real large
  learned weight loading.
````

<a id="model-execution-l468"></a>

#### open · [L468–468](../../plans/model-execution/TODO.md#L468)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L468-L468@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `1d43fb0bd575ff332ba44eab667ece46f1cc185c26b74c6d662483e88f2b6eda`.

````markdown
- [ ] Qwen3.8 27B: original HF grouped-head ordering and Q/K replication 16→48, explicit converted-profile permutations, untied head and source-derived norm/RoPE geometry.
````

<a id="model-execution-l469"></a>

#### open · [L469–469](../../plans/model-execution/TODO.md#L469)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L469-L469@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `f18fb72c2f30cb035e9affc6e4a520d78be71689cd229669958a7cd9658deff6`.

````markdown
- [ ] Gemma E2B/26B: shared GELU/multiply/scale/soft-cap, proportional RoPE, value norm/sliding/shared KV, E2B PLE/doubled FFN and 26B K=V/MoE branches; choose and qualify the exact reference-framework numerical schedule.
````

<a id="model-execution-l470"></a>

#### open · [L470–470](../../plans/model-execution/TODO.md#L470)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L470-L470@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `5c588c488f34d624ac58fabf41c5c9a86f09066740ab338d2830855551e201c0`.

````markdown
- [ ] Qualify actual trained CPU operators/full vocabulary/state/nonzero replay, official tokenization/templates/reasoning/stop behavior, sessions, rollback/checkpoints and resource refusal.
````

<a id="model-execution-l471"></a>

#### open · [L471–471](../../plans/model-execution/TODO.md#L471)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L471-L471@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `8a81420e9c002dec3913d6ec9c2daac8d5f8b4f89e3e9bb7cd1ca84c90911a6f`.

````markdown
- [ ] Add the same source-bound forward plans and full physical GPU qualification per eligible device, preserving all state/actual-completion/cancellation guarantees.
````

<a id="model-execution-l472"></a>

#### open · [L472–472](../../plans/model-execution/TODO.md#L472)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L472-L472@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `0867306b7ef0c2c037ba962b82275c0b1ec7731e5d49b09625393bb5b607e4b3`.

````markdown
- [ ] Keep distinct future heavy/hybrid recipes (Qwen Flash-Next, GLM, MiMo and source-defined QSA/MLA/KDA/indexers/hyperconnections), and broaden dense/encoder/pooling/reranking/LoRA support only through independent model/codec/backend gates.
````

<a id="model-execution-l473"></a>

#### open · [L473–473](../../plans/model-execution/TODO.md#L473)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L473-L473@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `98af65f8caa091708a2258a5445f4246b12f73ec672abc1bd6267e8b843bc1a2`.

````markdown
- [ ] Preserve CPU-first family priorities; already-authorized parallel basics do not authorize model/SDK downloads or unsupported whole-model claims.
````

<a id="model-execution-l475"></a>

#### open · [L475–475](../../plans/model-execution/TODO.md#L475)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L475-L475@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `f43b93aebd5b2c2da086cd64c8670be42aecc70b41e480dd27448e2f7755db4e`.

````markdown
- [ ] Bind explicit original HF versus converted Qwen source profiles. Conversion permutes grouped/tiled V projections/constants/convolution/output, uses -exp(A_log) and adds1 to non-GDN norms; GDN swish is SiLU while full-attention gating remains sigmoid.
````

<a id="model-execution-l476"></a>

#### open · [L476–476](../../plans/model-execution/TODO.md#L476)

Heading: `Model execution > Uncommitted > Source-specific optimization backlog — 21 Oct, deferred behind current order`.
Identity: `.agents/plans/model-execution/TODO.md:L476-L476@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `1b47e1d265cf43f6bd980394b89ccd22003e4aa10dcb9afad69f8377fb879c23`.

````markdown
- [ ] Choose and independently qualify Gemma26B reference-framework routing/top-k/reduction/scale placement before production execution. Five analytic test-local routing cases are not trained source parity.
````

<a id="model-execution-l1442"></a>

#### open · [L1442–1449](../../plans/model-execution/TODO.md#L1442-L1449)

Heading: `Model execution > Deferred — after Qwen/Gemma and Flash execution`.
Identity: `.agents/plans/model-execution/TODO.md:L1442-L1449@ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f`.
Block SHA256: `b27a3472fd25b73a2927f4860b0062e840bc59049ba81b62752fd549de4068a6`.

````markdown
- [ ] Qualify independently selected NanoJev-like calibrated decision
  model sources under the same model/source/cache/residency authority.
  Register finite choice/boolean/score or include/omit heads without
  assuming a specific research checkpoint. Validate calibrated
  probabilities and per-question source-bound state, batching, thresholds,
  accepted/rejected context records and decision-fallback semantics
  before enabling generic context-filter execution; pipeline check alone
  provides only typed static intent.
````

### multimodal — 27 noncompleted occurrences

<a id="multimodal-l12"></a>

#### open · [L12–22](../../plans/multimodal/TODO.md#L12-L22)

Heading: `Multimodal > Uncommitted`.
Identity: `.agents/plans/multimodal/TODO.md:L12-L22@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `2ac18a2151941c17a204e6f032e48181e9103924b4dccb2db70cd302910fbde0`.

````markdown
- [ ] Opt-in, source-granted context-filter outputs as bounded,
  immutable, typed hints to ASR, acoustic-first diarization,
  transcript-text and separately source-qualified speaker-preserving
  translation stages. Default only prior committed session user/assistant
  records; opt-in tool call/result, app system/developer prompts and
  explicitly authorized shared data permitted. Generic include/exclude
  and typed decision-model choices operate after source ownership checks.
  Unsupported original ASR prompt dialect refuses or uses a separately
  verified correction stage. Actual audio A/B WER/CER/DER/translation
  adequacy, timestamp/partial-output correctness, consent/authorization
  and canceled-turn isolation are required before any runtime claim.
````

<a id="multimodal-l24"></a>

#### open · [L24–31](../../plans/multimodal/TODO.md#L24-L31)

Heading: `Multimodal > Uncommitted`.
Identity: `.agents/plans/multimodal/TODO.md:L24-L31@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `072d81d706c49a39cdd264003c448a4bf85851c798e9ba2680ea52c0c45b09fb`.

````markdown
- [ ] Model each ASR (Qwen ASR/Whisper/Canary/Parakeet),
  diarization (separately sourced speaker recognition), LM, Qwen TTS,
  Qwen Image2.1 and future video as registered typed components,
  not separate per-node subprocesses. The ASR→speaker-tagged
  transcript→LLM→TTS single coherent pipeline must include rational
  clocks, codec validity, VAD, per-turn epochs, barge-in,
  bounded fanout/backpressure and correct playback ACK frontiers.
  Current TOML/audio/image graph skeleton has NO real media runtime.
````

<a id="multimodal-l32"></a>

#### open · [L32–36](../../plans/multimodal/TODO.md#L32-L36)

Heading: `Multimodal > Uncommitted`.
Identity: `.agents/plans/multimodal/TODO.md:L32-L36@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `6e99a3521955e1a1edfc16425dcb99db7c078b83cc71b7a5dcc8dc267435076c`.

````markdown
- [ ] Enable native progressive image output only at complete
  source-valid VAE/tile/frame boundaries; partial latents or
  unverified solver steps are never playable. Actual media clocks,
  atomic commit barriers, ROS/transport and tested streaming
  quality/latency require source-real gates.
````

<a id="multimodal-l38"></a>

#### open · [L38–38](../../plans/multimodal/TODO.md#L38)

Heading: `Multimodal > Uncommitted`.
Identity: `.agents/plans/multimodal/TODO.md:L38-L38@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `998014231451b4d853d4015614fbbae40d4bcb4b2ef3b5e0cf62cb7dfb7c147d`.

````markdown
- [ ] Add typed sample/frame/codebook/latent/decision values, rational clocks, checked conversions and complete source-bound component/codec/processor assemblies.
````

<a id="multimodal-l39"></a>

#### open · [L39–39](../../plans/multimodal/TODO.md#L39)

Heading: `Multimodal > Uncommitted`.
Identity: `.agents/plans/multimodal/TODO.md:L39-L39@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `3dc2d2f213b7349b4c8ba825e3c5def3c23d6a9f8df122e02b1ee2a7d9083ce9`.

````markdown
- [ ] Prove a small real non-token encoder/decision task and bounded iterative task through the existing engine before broad media claims.
````

<a id="multimodal-l40"></a>

#### open · [L40–40](../../plans/multimodal/TODO.md#L40)

Heading: `Multimodal > Uncommitted`.
Identity: `.agents/plans/multimodal/TODO.md:L40-L40@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `fbcdcb2c400ba19cc682898c163ab9a9953ec6fc3c52cbcd042885e23f52e4dd`.

````markdown
- [ ] Qualify CPU-first original ASR and non-AR TTS (Whisper/Parakeet and Kokoro candidates); then actual backend execution and measured streaming/audio quality.
````

<a id="multimodal-l41"></a>

#### open · [L41–41](../../plans/multimodal/TODO.md#L41)

Heading: `Multimodal > Uncommitted`.
Identity: `.agents/plans/multimodal/TODO.md:L41-L41@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `1c2e2accc3d394f55c5e41d62110a6439484dc800c9f8f2c8e2b4a29fb3ed1c4`.

````markdown
- [ ] Add depth-AR/RVQ speech, dual-AR codebook TTS and genuine codec-based duplex neural audio (including Qwen-Omni) where source/license/resources permit; mock codecs or text cascades do not establish native support.
````

<a id="multimodal-l42"></a>

#### open · [L42–42](../../plans/multimodal/TODO.md#L42)

Heading: `Multimodal > Uncommitted`.
Identity: `.agents/plans/multimodal/TODO.md:L42-L42@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `6dc163f057ed523ea560cc106217c1c8ee8f263234b63b2b0e84f0f4723b7336`.

````markdown
- [ ] Qualify seeded source-exact image flow/editing conditioner+VAE pipelines, then video temporal/3D-VAE/expert switching and synchronized AV commit groups; candidate families remain separately gated.
````

<a id="multimodal-l43"></a>

#### open · [L43–43](../../plans/multimodal/TODO.md#L43)

Heading: `Multimodal > Uncommitted`.
Identity: `.agents/plans/multimodal/TODO.md:L43-L43@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `cd695fc6621394a940ceb3f91695bb48386024d371d607cdcaba5812fda63525`.

````markdown
- [ ] Keep ternary PTQ/Hadamard codecs, schema/calibration decision heads and specialist encoders as independent codec/model/task gates, not aliases of Qwen.
````

<a id="multimodal-l44"></a>

#### open · [L44–44](../../plans/multimodal/TODO.md#L44)

Heading: `Multimodal > Uncommitted`.
Identity: `.agents/plans/multimodal/TODO.md:L44-L44@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `2567c1dada8abcdc054700717cda43304d9259d90aff5fe55d51263451b6b728`.

````markdown
- [ ] Implement RealtimeSession lanes/epochs, bounded media buffers, VAD/barge-in, presentation/ack/playback frontiers and transactional rollback without speculative/partial media leakage.
````

<a id="multimodal-l45"></a>

#### open · [L45–45](../../plans/multimodal/TODO.md#L45)

Heading: `Multimodal > Uncommitted`.
Identity: `.agents/plans/multimodal/TODO.md:L45-L45@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `37950f9db60e938fba0072f402563b77b2fe5e004901d0b8d072f7923f4e63ff`.

````markdown
- [ ] Add per-model transport capabilities and authorized hosted TaskService adapters with privacy/cost/idempotency/Unknown reconciliation; no duplicate local runtime or guessed realtime guarantees.
````

<a id="multimodal-l46"></a>

#### open · [L46–46](../../plans/multimodal/TODO.md#L46)

Heading: `Multimodal > Uncommitted`.
Identity: `.agents/plans/multimodal/TODO.md:L46-L46@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `1373dcfc7bd9868e94b467fec11dfd14ea5cfb48bf5ee2f314058d26c317ea5d`.

````markdown
- [ ] Complete MM-01..24 plus model-specific ASR/TTS/image/video/decision quality/clock/resource suites; no real media-model support follows from source-free mocks.
````

<a id="multimodal-l59"></a>

#### open · [L59–59](../../plans/multimodal/TODO.md#L59)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L59-L59@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `0225b92340c359a375b87946d133631354b689445d404f5ae15213c891999259`.

````markdown
- [ ] MM-D1/MM-01/MM-15: checked rational sample/frame/codec clocks, 44.1/48 kHz to 16/24 kHz and 24000/1001 or 30000/1001 FPS, long-duration remainder, discontinuity, overflow and bounded parser/buffer refusal.
````

<a id="multimodal-l60"></a>

#### open · [L60–60](../../plans/multimodal/TODO.md#L60)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L60-L60@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `fc02ee724f24ae317c490873c27d61659a43766ce9f49c569fa2e37fc0bb2ac5`.

````markdown
- [ ] MM-D2/MM-02: complete source/processor/codec/license assemblies; wrong conditioner/shape/mask, missing source transform and cross-session mutable leakage refuse before use.
````

<a id="multimodal-l61"></a>

#### open · [L61–61](../../plans/multimodal/TODO.md#L61)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L61-L61@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `acb92e36e34222f4e8a589987902a5df65097307b2d65e876ed37ec522cba575`.

````markdown
- [ ] MM-D3/MM-03..09: seeded solver/RNG and direct finite decision state, CTC/RNNT/TDT distinction, complete RVQ depth and joint AV commit; interrupted/stale device work never publishes or exposes media.
````

<a id="multimodal-l62"></a>

#### open · [L62–62](../../plans/multimodal/TODO.md#L62)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L62-L62@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `3f2f19f76c495fc96497990dc1a9bc4d8d07a5327c8dd0ea2133d54eac3027c3`.

````markdown
- [ ] MM-ASR-01..12: actual frontend/window/blank/predictor/duration/language/timestamp/diarization source fixtures; offline-only checkpoint refuses streaming, malicious audio bounded before allocation, sessions isolated.
````

<a id="multimodal-l63"></a>

#### open · [L63–63](../../plans/multimodal/TODO.md#L63)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L63-L63@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `5d05ef1f6ceeb458f342eaefa45e941357c6075db71f5126ce2695d3c02bbe4d`.

````markdown
- [ ] MM-TTS-01..12: non-AR phoneme/duration/vocoder and exact PCM count; incomplete 9/10 RVQ frame unplayable, codec-rate swap refused, overlap/voice generation and duplex/barge-in/state/lease boundaries correct.
````

<a id="multimodal-l64"></a>

#### open · [L64–64](../../plans/multimodal/TODO.md#L64)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L64-L64@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `3961e37acc67a0f2e7210a12fc9cb6dde7efaddf3a82c5eb0b48d61b4c195800`.

````markdown
- [ ] MM-IMG-01..12: source conditioner/solver/CFG/seed/mask/VAE/color and generation-entrypoint traces; invalid source stage/crop/processor fails, cancel retains old root and unknown physical handles, source/license/hosted status truthful.
````

<a id="multimodal-l65"></a>

#### open · [L65–65](../../plans/multimodal/TODO.md#L65)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L65-L65@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `9a849aca0f2a25384e3c628b388202f713523ca0a4d982c018f8ccfba658d52e`.

````markdown
- [ ] MM-VID-01..15: temporal VAE/latent crop/frame count, Wan timestep experts, source tile masks/rectangular attention, AV cross-attention/joint clocks, invalid GOP/backpressure, hot reload, cancellation and whole-assembly admission.
````

<a id="multimodal-l66"></a>

#### open · [L66–66](../../plans/multimodal/TODO.md#L66)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L66-L66@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `ecf3f0fc71e6e299ab60da9b2c556fe051e57907337da52ef6636d45b418e654`.

````markdown
- [ ] MM-DEC-01..06: finite and joint schema head options/order/null/refusal/calibration against original source; generated text is not source-equivalent, external egress refused without authorization.
````

<a id="multimodal-l67"></a>

#### open · [L67–67](../../plans/multimodal/TODO.md#L67)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L67-L67@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `13452d8187c4695d9b04903b9a5806e255b07ae796367b4cda3cc273936ad2f3`.

````markdown
- [ ] MM-BON-01..08: PTQ1_0/PQ2_0 packed source scales/signs/padding plus mandatory 1024 Hadamard activation transform; wrong codec or plain-Q2 path refused, original scalar/CPU/device source parity separately qualified.
````

<a id="multimodal-l68"></a>

#### open · [L68–68](../../plans/multimodal/TODO.md#L68)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L68-L68@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `0f08bb96b8f75215507884ead873e5eacf4eab8769d414e093b25b0f8749d6e3`.

````markdown
- [ ] MM-13..17/MM-19: real per-session/turn/transport epochs; stale ACK ignored, duplicate reconnect cursor idempotent/authenticated/bounded, playback progress remains uncertain when missing, slow clients do not release device leases or grow queues indefinitely.
````

<a id="multimodal-l69"></a>

#### open · [L69–69](../../plans/multimodal/TODO.md#L69)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L69-L69@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `bed2935e80d7a7b7014e999b76ec7c941b34d692266433e83dc301cc6cf2917b`.

````markdown
- [ ] MM-18/MM-22/MM-23: ten speech/five video mixed jobs, fair shared physical admission and legal solver/frame yield; no per-packet catalog/router/global snapshot or silent precision/step downgrade, coupled output stays unplayable until both tracks valid.
````

<a id="multimodal-l70"></a>

#### open · [L70–70](../../plans/multimodal/TODO.md#L70)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L70-L70@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `688e3b1a7adad9e9fa1b05ba1b3c1e4fed66094ec8d5373ab25f3a30bb3926ef`.

````markdown
- [ ] MM-SRV-01..12/MM-20/21/24: exact versioned SDK/protocol/field/task conformance, hosted accepted-timeout Unknown/idempotent reconciliation, callback/signature/TTL/MIME bounds, malformed media/unauthorized egress/cross-tenant replay refused before side effects.
````

<a id="multimodal-l71"></a>

#### open · [L71–71](../../plans/multimodal/TODO.md#L71)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L71-L71@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `caa74c66358d5688bc7cae0c81d61b343e03290c386266ad7b45bfceeba64ee6`.

````markdown
- [ ] MM-I0 change-radius drill: small encoder/finite task and iterative mock use existing source/state/resource owners; no central router/worker rewrite or duplicate runtime; original Qwen gates stay unchanged.
````

<a id="multimodal-l72"></a>

#### open · [L72–72](../../plans/multimodal/TODO.md#L72)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L72-L72@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `69bcef6990848e37b9c8a5e48ebb53b8d71caa11674ade292c92b833b787e0c9`.

````markdown
- [ ] Eligible source-real CPU→physical full-model→playable-stream→soft-QoS gates: exact licensed checkpoints/processors/codecs and original quality/state oracles, real negative cancellation/admission/replay tests and measured cold/warm tails/memory. Mock/reference/region/hosted results never promote another tier.
````

<a id="multimodal-l73"></a>

#### open · [L73–73](../../plans/multimodal/TODO.md#L73)

Heading: `Multimodal > Tests and verification`.
Identity: `.agents/plans/multimodal/TODO.md:L73-L73@774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208`.
Block SHA256: `84e8f204d5b7b420f1aecedc87281a67533a251e112fe3967531ff1ae406824f`.

````markdown
- [ ] Resolve realtime section18 policy choices explicitly (media numeric policy, first sources, SLO/clock uncertainty, mid-turn changes, persistence, biometric/asset consent, hosted cost/egress, transport packaging and safe physical recovery). User examples do not establish released model IDs or grant execution/provider/download authority.
````

### numerics-and-context — 31 noncompleted occurrences

<a id="numerics-and-context-l85"></a>

#### open · [L85–89](../../plans/numerics-and-context/TODO.md#L85-L89)

Heading: `Numerics and context > Uncommitted > Reference capture and remaining runtime separation — 26 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L85-L89@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `d5c93a263753043f468786a0e241fa5ffe15811192b860a8301761faff5f1ef0`.

````markdown
- [ ] Remove remaining shared rotary fault assertions/fixed-prompt qualifier
  dependency and case-driven model/GDN/native dispatch. Implement genuine
  parameterized kernels and explicit numerical configurations; do not move or
  rename test whitelists into runtime descriptors. Original CUDA scalar/source
  facts stay Rust fixtures. Parametric source proof/design is available only.
````

<a id="numerics-and-context-l93"></a>

#### ~ · [L93–102](../../plans/numerics-and-context/TODO.md#L93-L102)

Heading: `Numerics and context > Uncommitted > Current owner order — 26 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L93-L102@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `45e4fb1cd0a66ae63d63a4c1241022bee711b2667e85fa3e8d423cc5006acf64`.

````markdown
- [~] Sequential continuation after serving shutdown: fresh .8B prompt47/48/49/50
  originals, matching native profiles, real-model shutdown/faults, genuine long
  contexts/backend routes, then optimized matched C16 and27B. All23 approved
  model members freshly SHA-verified. New parameterized reference collector
  preserves original math/state/fullV/source validators and exact owner; source
  controls pass. Fresh47/48/49/50 original pairs now pass all24/all42, fullV,
  consumed priors and natural feedback;47 uses explicit separate postvalidation
  after its original controller report-size failure. Fresh complete packed
  v3 captures47–50 now pass; native comparisons remain pending. Native fresh48–50 remains
  unsupported until actual source/ABI/physical qualification.
````

<a id="numerics-and-context-l108"></a>

#### ~ · [L108–110](../../plans/numerics-and-context/TODO.md#L108-L110)

Heading: `Numerics and context > Uncommitted > Current owner order — 26 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L108-L110@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `d317fe0cf0676f83998581d28a5125acab2215428c70b0853439cf8a9ba24fd3`.

````markdown
- [~] Apply [all-model correctness policy](CORRECTNESS_POLICY.md), based on
  pinned vLLM/llama.cpp. Sol6.1 xhigh owns every lane; this supersedes older
  Pi/Astra assignments and performance-first continuations below.
````

<a id="numerics-and-context-l111"></a>

#### open · [L111–116](../../plans/numerics-and-context/TODO.md#L111-L116)

Heading: `Numerics and context > Uncommitted > Current owner order — 26 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L111-L116@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `9b4012de32ca0b25c1594bd77002ad5ad2ee083c0b2ce39a1978f6e6e739d560`.

````markdown
- [ ] .8B initial47/48/49/50 and cached47→50: actual independent original and
  native outputs/state/physical closure. Original case5 strict FAIL101 remains;
  original N1/B4 QKVZ itself differs by six BF16 words at identical frontier47
  operands. The historical fifteen-transition archive lacks paired N1 frontier50;
  fresh47–50 original trajectories are now independently available. This explains
  the diagnostic, not native model acceptance.
````

<a id="numerics-and-context-l117"></a>

#### open · [L117–121](../../plans/numerics-and-context/TODO.md#L117-L121)

Heading: `Numerics and context > Uncommitted > Current owner order — 26 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L117-L121@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `938a3874afa15a1a46dcd2ca1078326ce7e5063e8665f0dc07193d08837af09d`.

````markdown
- [ ] Genuine .8B512/1024/2048/4096/8192, then16K/32K context/state/feedback;
  corresponding original BF16 CPU/ROCm/Vulkan full-model routes are missing.
  GGUF, synthetic operators and finite other-model proofs stay separate. Local
  Renoir/gfx90c launch/recovery hold remains. Performance and27B follow this
  correctness sequence; earlier finite timing facts remain historical.
````

<a id="numerics-and-context-l123"></a>

#### open · [L123–129](../../plans/numerics-and-context/TODO.md#L123-L129)

Heading: `Numerics and context > Uncommitted > Current owner order — 26 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L123-L129@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `e9bbf5f3382c4a0b80bf70e71a59f90c38cfbe071a37e065d15002b09f39b69f`.

````markdown
- [ ] **Next one short-HF unit, then matched benchmarks:** both approved .8B/2B
  sources, canonical/ragged47/57/62/67/72 plus genuine cached feedback, original
  active8/caps and existing worker. Stage generic bounded storage/liveness,
  checked runtime logical views/epochs, complete lowerer access/tap ownership and
  pooled primitive scratch; current metadata planner alone grants no alias/fit.
  Preserve old source/precision/native gates. Benchmark the whole qualified short
  matrix together, then extend long contexts; unrelated27B/Gemma continue.
````

<a id="numerics-and-context-l131"></a>

#### open · [L131–131](../../plans/numerics-and-context/TODO.md#L131)

Heading: `Numerics and context > Uncommitted > Current owner order — 26 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L131-L131@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `e9d4c87ea688de30fedb166bd53acbe5715a31ef34ba5c97b21679bd5f78ce9f`.

````markdown
- [ ] Actualcanonical47projectiondiscovery1 completed original PLAIN+OBSERVED/source/ownedcleanup, but exactcommon state.recurrent.after call0 differs. NewOBS cumsum/KKT/H/output selections differacceptedreference1; selectionvariance is verified, specificcause unproved. Independent actual metadata identity/layout gate now passes separately; no fullmodel/packedargs/native/perf acceptance.
````

<a id="numerics-and-context-l132"></a>

#### open · [L132–132](../../plans/numerics-and-context/TODO.md#L132)

Heading: `Numerics and context > Uncommitted > Current owner order — 26 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L132-L132@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `0cb76e64a5d5fad4459b958e0913a4915c40e34ce85f7917d7f22af2fbfeb016`.

````markdown
- [ ] Qualify actual canonical short47input/max8output/EOS248046 and ragged57/62/67/72 prompts; exact source/profile/numerical schedule and original all24/all42/fullV/actual sampler. Extend genuine rotary/ABI/control coverage as required, then original worker/HTTP and C1/2/4/8. Old GGUF diagnostic shapes do not grant HF parity.
````

<a id="numerics-and-context-l133"></a>

#### open · [L133–133](../../plans/numerics-and-context/TODO.md#L133)

Heading: `Numerics and context > Uncommitted > Current owner order — 26 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L133-L133@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `1528fe34f66fa511b59575508afa6dea13c7faf8e42523b98ddc7c1f46ad68fe`.

````markdown
- [ ] Distinct native2B source opening/2048wide ordinary+dualRMS/projection+6144MLP/head/970window tied table+4096B source-row actor. Actual2B prefill PTX differs in cumsum, scaled-dot KKT, recompute W/U and chunk output; matching GDN geometry and different cache keys are not an execution permit. Preserve actual reference3/source/relative5GiB boundary; derive physical fit, then same fullmodel/natural/fault/serving closure. Matched short benchmarks, long contexts and dense27B follow actual parity. End-to-end benchmarks remain unexecuted.
````

<a id="numerics-and-context-l134"></a>

#### open · [L134–134](../../plans/numerics-and-context/TODO.md#L134)

Heading: `Numerics and context > Uncommitted > Current owner order — 26 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L134-L134@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `2f85efdb30db258ca74318b5199ba64a0a6ccce39ffd23f0cc5d55651839568b`.

````markdown
- [ ] Investigate actual2B discovery2 PLAIN outlier: both lanes finish/source/ownedcleanup pass, OBSERVED matches acceptedreference3, but PLAIN first differs at call2/layer8 recurrent state and branch/residual, then consumed downstream/fullV. Conv.after is only its last3 cache rows; no proof all upstream conv outputs match. Ordinary layer0 RMS10boundaries/565248B match independently. Fullmodel discovery remains unaccepted; source audit confirms timing-based autotuning but no specific cause; unchanged PLAINrepeat1 completed with exact reproduction. Independent RMS leaf reference accepted after complete source/IR/byte gates (565248B boundaries/sourceembedding,4096B gamma); native2048 component physical comparisons/faults/fullgate20 now pass separately, explicit parent-failure boundary retained. Unchanged PLAINrepeat1 reproduces discovery2PLAIN exactly across113713152all42+11168652common+6117576final+565248RMS B, still548/21/3different records versusOBSERVED/reference3. Python-only actual selection2 diagnostic now completes18records/lane and strict source/common/final/prior/all42/RMS gates PASS despite four differing selected stageconfigs. Different configuration alone is insufficient; the earlier reproducible outlier remains unresolved.
````

<a id="numerics-and-context-l137"></a>

#### open · [L137–137](../../plans/numerics-and-context/TODO.md#L137)

Heading: `Numerics and context > Uncommitted > Binding continuation — 21 Oct 09:30 UTC`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L137-L137@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `0bdf650adbbfedfa9864be0ec15004998ecb23b280b6ff082d03b7d5cb7c0e92`.

````markdown
- [ ] Native continuation: genuine47 current-root/position/provider/cache coverage, cached2B histories and whole-model admission remain open. Isolated2B QKVZ/BA phase51 and actual six-node source prefix/GDN phase52 are accepted below; initial1/64 stops before GdnOut. Original full24 permit stays guarded.
````

<a id="numerics-and-context-l138"></a>

#### open · [L138–138](../../plans/numerics-and-context/TODO.md#L138)

Heading: `Numerics and context > Uncommitted > Binding continuation — 21 Oct 09:30 UTC`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L138-L138@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `00bdad31fe560d09cb2ff61bd1d9906c5f9b0b581c3367b4948a22bcabf603bc`.

````markdown
- [ ] Current original2B KKT causal leaf1:29frozenchildren/23pure guards, actual original source/ownedcleanup and ordinaryRMS565248B pass. Four KKT configs/two repeats produce bit-identical A/F32+Ai/BF16 (all72comparisons0), but every solve uses the SAME2warps/4stages selected kernel: this does not compare solve variants. Strict original model reference still FAILS; current run reproduces prior PLAIN outlier byte-for-byte131564628B/27files. No tuner override or failure waiver; KKT config alone does not explain this case. Physical allocator peak remains unknown.
````

<a id="numerics-and-context-l139"></a>

#### open · [L139–139](../../plans/numerics-and-context/TODO.md#L139)

Heading: `Numerics and context > Uncommitted > Binding continuation — 21 Oct 09:30 UTC`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L139-L139@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `cc93be45c494cafb2960631571a1c8fc5b036c77f53fc7ec58a7596a930988d2`.

````markdown
- [ ] Canonical47 failure narrowed in parallel: three separate PLAIN runs share the same all42 record differences; their OBSERVED lanes match acceptedreference1. Original late-H pair1 now reproduces oldbad PLAIN/good OBSERVED actual state bytes under healthy source/owned cleanup; original raw comparison remains FAIL. Inputs to H already differ: W59BF16words, U9BF16words (token34/head4); K/G/H0/history exact, Vnew=U, finalHT1002F32words differ. Later failed-pair solve-counterfactual1 directly captures equal operands/F32A and first solve.Ai differences, below. This earlier late-H capture alone did not localize the cause.
````

<a id="numerics-and-context-l140"></a>

#### open · [L140–140](../../plans/numerics-and-context/TODO.md#L140)

Heading: `Numerics and context > Uncommitted > Binding continuation — 21 Oct 09:30 UTC`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L140-L140@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `440f08fda01f0531154a5f3aeb26cde76ba7749cbc7e26b1073fd67fe0ce21af`.

````markdown
- [ ] Post-recompute47 investigation: attempt1 observer refused copied callable-grid assumption; original caller uses literal(1,16), failed attempt/cleanup retained. Narrowv2/28guards/peer review +48files/33parents/21runtime/3historypins PASS; actual attempt2 source/owned cleanup and original pair/common/fullV/all42 gates PASS. Seven actual K/V/beta/G/Ai/W/U records872320B/lane all equal good35; neither lane reproduces badW/U, so no historical cause/fix. Same-input original solve/recompute counterfactual proposed to test the actual observed warp2/warp4 solve selections without production tuner mutation.
````

<a id="numerics-and-context-l141"></a>

#### open · [L141–141](../../plans/numerics-and-context/TODO.md#L141)

Heading: `Numerics and context > Uncommitted > Binding continuation — 21 Oct 09:30 UTC`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L141-L141@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `78c2039f074bc2b3dd3826ad0b429be84bc585d3306f1acffd8f95fed01fb7f5`.

````markdown
- [ ] Repair47 exact arithmetic binding at existing provider/preparation seam; authenticate declared candidate and actual original code/ABI/launch histories, qualify finite owned solve variants, then connected full47+cached histories. Broad geometry/source equality cannot establish byte equivalence across two proven unequal solve implementations. No tuner forcing, expected-word selection, tolerance relaxation or failed-parent waiver; all18 GDN layer/eight-call identity coverage and fullmodel gates remain open.
````

<a id="numerics-and-context-l143"></a>

#### open · [L143–143](../../plans/numerics-and-context/TODO.md#L143)

Heading: `Numerics and context > Uncommitted > Binding continuation — 21 Oct 09:30 UTC`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L143-L143@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `8dbfad8f4d98d8d5644e4ffba00c8dd055fc3687b46d4661c2e745d1c1d319cd`.

````markdown
- [ ] Retired2B postAttention1 parent gate remains CLOSED: bothlanes healthy/newlayer0pairedboundaries/source/native14op PASS. Fresh L8 pair3 source/lifetimes/full37paired archives verified after22observer+5host-schema guards: dual/RMS/QKVZ/BA/fullConv/preparedQKV/coefficients/cumulativeG/KKT A exact; FIRST solveAi14BF16differences, W96/U17/H2048. Actualsolve configs2warps/3stages versus4/4 and fullPTX differ; currentHT hashes reproduce prior same-lane split. Its original fullV/all42 failures remain fatal, RMSPASS. Separate new original MLP/ACT reference success and fixedOBS-only component successor do not accept this failedparent. Retired same-input originalsolve/cross-recompute leaf3 PASS: same actual794640B inputs; solve2warps/3stages reproducesPLAIN and4/4 reproducesOBS, Ai14/W96/U17 word split exact under BOTH recompute8/3 and8/4, twice; all12 returned completePTX/cachehashes equal actualoriginal variants. This isolates that split to solve configuration; it does not repair the original parent or prove earlier uncaptured inputs. Previous72 comparisons did not vary solve configuration.
````

<a id="numerics-and-context-l144"></a>

#### open · [L144–144](../../plans/numerics-and-context/TODO.md#L144)

Heading: `Numerics and context > Uncommitted > Binding continuation — 21 Oct 09:30 UTC`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L144-L144@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `bb638393d00be6a58280c6b012fc2553611033766c2ac364acb421bda243d0be`.

````markdown
- [ ] Next: all24/all42/fullV/feedback/cached47 histories and workerHTTP parity. New original2B MLP+activationcapture1 passes strict original common/fullV/prior/all42/RMS/postAttention/MLP reference gates and captures two6144activation packets; ROOTpostvalidation/native2B path/fullmodel remainpending. Earlier failed postAttention parents remainfailed; predeclared fixedOBSpost1 successor target unchanged. MatchedC1/2/4/8 benchmarks follow parity, then longcontexts and27B.
````

<a id="numerics-and-context-l145"></a>

#### open · [L145–145](../../plans/numerics-and-context/TODO.md#L145)

Heading: `Numerics and context > Uncommitted > Binding continuation — 21 Oct 09:30 UTC`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L145-L145@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `eadfeb092c115b008f7d1e7b63056565a5afcffb0db8f732984202df54430fa3`.

````markdown
- [ ] Post-parity overhead follow-up: consider immutable GdnArithmetic same-object pointer shortcut before structural fallback; source-valid review only, no measured speedup or code change.
````

<a id="numerics-and-context-l301"></a>

#### open · [L301–306](../../plans/numerics-and-context/TODO.md#L301-L306)

Heading: `Numerics and context > Uncommitted > Bottom-up F32 repair — 20 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L301-L306@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `4111b020f472faaf15e96e0acf223931fb7e996411bf2a2dc4bf5abea3760244`.

````markdown
- [ ] Qualify original-row Q4_0_8x8 QKV/Z using existing source projection math
  against native fields6/38 before changing any equation. Native operands are
  CPU_REPACK Q4_0, distinct from Q8_0/Zen BLAS; actual inner Q4 dispatch is
  source-derived, not freshly instrumented. QKV M6144 needs separate closed
  geometry; do not raise tinyBLAS limits. Then compose existing convolution,
  SiLU and grouped Q/K normalization with retained raw6144×3 history.
````

<a id="numerics-and-context-l307"></a>

#### open · [L307–312](../../plans/numerics-and-context/TODO.md#L307-L312)

Heading: `Numerics and context > Uncommitted > Bottom-up F32 repair — 20 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L307-L312@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `38d39b1d3ebccfe14ab0e0915f9b65e2db3a403af27186d8648a8dda0a20d5c9`.

````markdown
- [ ] Nonzero cached-prior, live producer, full temporal/default-bulk route/
  profile/state binding and default8K qualification. Retained native bulk64
  versus AR1 upstream inputs match2097152 words plus1572864 four-tap operand
  words; that comparison is not a Rust producer proof. Current
  Q/K/V/prior/Z remain explicit observations; gain lacks separate capture.
  Sliced CPU proofs confer no complete-model/vLLM or whole-host RAM qualification.
````

<a id="numerics-and-context-l313"></a>

#### open · [L313–315](../../plans/numerics-and-context/TODO.md#L313-L315)

Heading: `Numerics and context > Uncommitted > Bottom-up F32 repair — 20 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L313-L315@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `292985dc1924dd0b36d47e8db2aab49cf91d7aeab97e5a94e26043e0e414b461`.

````markdown
- [ ] MAIN full formatting still flags only the preserved SiLU draft
  `resident_program/silu.rs:91`; bytes match its pre-existing checkpoint.
  Do not alter another owner's draft.
````

<a id="numerics-and-context-l319"></a>

#### open · [L319–324](../../plans/numerics-and-context/TODO.md#L319-L324)

Heading: `Numerics and context > Uncommitted > Remaining qualification`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L319-L324@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `65812451aa9ea06e4d85b962a15a694a17f0a619de3cb716f3370f32c495892b`.

````markdown
- [ ] **vLLM parity remains open:** implement and physically qualify its own
  CUDA arithmetic/cast/layout profile; complete initial/continued prefill and
  cached-decode GDN under that profile; close full .8B/2B/27B logits/state and
  long-context gates, then matched default/warm/concurrent serving performance.
  Current GGUF CPU operator proofs and historical cold-eager TTFT observations
  establish no completion percentage, cross-engine equivalence or new speedup.
````

<a id="numerics-and-context-l326"></a>

#### open · [L326–335](../../plans/numerics-and-context/TODO.md#L326-L335)

Heading: `Numerics and context > Uncommitted > Remaining qualification`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L326-L335@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `2878032d6b4081f5e87e0ba23241d1e0cb4d60f0b344d7734bf242b124e85910`.

````markdown
- [ ] **Required three-model correctness gate (owner,20 Oct):** resolve
  independent original-source CPU8K GDN divergence for BOTH original
  Qwen3.5 models without loosening tolerances or changing declared source
  arithmetic. This is required alongside original Qwen3.8-27B correctness,
  matched serving/model benchmarks and shared-engine performance work.
  Short-context CPU/B1/native exactness and closed original-HF operator
  proofs do not close this independent GGUF long-context issue. Recover
  exact raw teacher prefixes and pinned native arithmetic/chunk settings,
  locate the first state/stage discrepancy, then rerun independent long
  context and the accepted short corpus after any implementation fix.
````

<a id="numerics-and-context-l336"></a>

#### open · [L336–336](../../plans/numerics-and-context/TODO.md#L336)

Heading: `Numerics and context > Uncommitted > Remaining qualification`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L336-L336@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `f71b7079433050989fe5fec98344efb4d7cd94e50669f64017ae6eed32aa9d74`.

````markdown
- [ ] Extend genuine resident GPU context/corpus qualification beyond the currently accepted 512 consumed tokens, including capacity/window/padding/replay boundaries.
````

<a id="numerics-and-context-l337"></a>

#### open · [L337–337](../../plans/numerics-and-context/TODO.md#L337)

Heading: `Numerics and context > Uncommitted > Remaining qualification`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L337-L337@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `65c91fd85b02df88297514805dd23a987f828ed9d7d71b743258dc64cdc35280`.

````markdown
- [ ] Run genuine whole-model 16K/32K CPU/GPU gates only after prerequisites; functional 8K success is not independent long-context numerical closure.
````

<a id="numerics-and-context-l338"></a>

#### open · [L338–338](../../plans/numerics-and-context/TODO.md#L338)

Heading: `Numerics and context > Uncommitted > Remaining qualification`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L338-L338@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `d3b4202de9855294307008ddb3253d80deecda2da3f0d8fb15d6e4d86f349be4`.

````markdown
- [ ] Keep internal full-logit/all-state/trace bit parity distinct from independent native-source stage/top20 comparisons, including the known 2B 56 signed-zero distinctions.
````

<a id="numerics-and-context-l339"></a>

#### open · [L339–339](../../plans/numerics-and-context/TODO.md#L339)

Heading: `Numerics and context > Uncommitted > Remaining qualification`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L339-L339@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `f772c0f0941cdc083887690f5a49594d3c2e2de7408063ba72695f2cce0b2c52`.

````markdown
- [ ] Broaden prompts/tokenizers/precision profiles and sampling quality only with independent oracles, exact history and explicit qualified ranges.
````

<a id="numerics-and-context-l341"></a>

#### open · [L341–341](../../plans/numerics-and-context/TODO.md#L341)

Heading: `Numerics and context > Uncommitted > Remaining qualification`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L341-L341@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `9ea6d76c555d28ad962cea8f0d8fc365856a16c8ad658de82b6c6e269996dd80`.

````markdown
- [ ] Diagnose identical teacher-forced prefixes512/1024/2048/4096/8192 against pinned llama.cpp0.6.0/build11429 `d812350`, CPU/F32 KV and Flash Attention off. Fix the physical microbatch/chunk/arithmetic schedule and compare per-head GDN state, gain/beta, quantization and AR-versus-bulk recurrence; distinguish replay correctness from arithmetic.
````

<a id="numerics-and-context-l342"></a>

#### open · [L342–342](../../plans/numerics-and-context/TODO.md#L342)

Heading: `Numerics and context > Uncommitted > Remaining qualification`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L342-L342@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `6731e42ed726fce38c1849bbba4baad654fde9e7fedb8eb6faeeaa8a97d6b460`.

````markdown
- [ ] Compare native ubatch1/64/256/512 only within bounded feasible runs; the historical full ubatch1 capture did not finish. Use raw token IDs to avoid incomplete UTF-8 HTTP rows. Recheck short corpus, layers, state and actual8K after any fix; retain original functional0.45 shared-top20-logprob/0.05 top-token-probability limits and strict source-bit operator gates.
````

<a id="numerics-and-context-l702"></a>

#### open · [L702–702](../../plans/numerics-and-context/TODO.md#L702)

Heading: `Numerics and context > Committed > Full24 forced-input numerical closure — 21 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L702-L702@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `41319d6c43106d6f80bfbe0457564e07234b3c239f41f47825361df638fa13b7`.

````markdown
- [ ] Naturalfeedback/current-root publication and serving integration through originalSession/ResidentProgram. Forced353 diagnostics remainfrontier0/unpublished; candidate IDs[198,599,8,72,18] were not fed back. Separate source-staged per-call binding/publication unit in progress. No serving/performance/fullmodel-default-route acceptance; newimplementation remainsISOonly.
````

<a id="numerics-and-context-l704"></a>

#### open · [L704–713](../../plans/numerics-and-context/TODO.md#L704-L713)

Heading: `Numerics and context > Committed > Full24 forced-input numerical closure — 21 Oct`.
Identity: `.agents/plans/numerics-and-context/TODO.md:L704-L713@2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa`.
Block SHA256: `2898c6e7d4f05f256612f957ab8457ad57bdd7c43ff5e64325e3972d64d74a78`.

````markdown
- [ ] Native2B original-source all-state/natural-feedback qualification and serving integration. Reference success does not establish Zerogravity inference, a driver hard cap or unknown transient highwater; actual baseline is separate from requested relative memory. Benchmarks/longcontexts/27B runtime remain OPEN.

- 22Oct freshattentioncapture2 retired both original lanes exit0/all56scopes64launches/full declared packets/12layouts/source33roles320/ownedCLI-group-container-EOF closure. Metadata remainsREFUSED:17projection+23FAprivate pointer candidates,7cachedcopies and3KVloader typedrefusals unresolved. Original common/all42/attention_raw pairFAIL remainsfatal. ROOT held-wholeSHA/recordSHA all24 archive diagnostic: newOBS==accepteddiscovery2OBS all3096records/166576128B; newPLAIN firstdiff layer0/call0H1002F32words(word66816), then1850records. Same8emittedIDs. OBSselectedKKT/H configurations changed but OBSoutputs areidentical; PLAINselection/firstinternaldivergence unobserved, no causal claim/newreference/math/native permit. Actual observedcall0GDN launches exist despite empty worker.selected; this does not seal all3layers/PLAIN tuples. First3 acceptedfixedOBS scope unaffected; complete47attention/fullmodel/benchmarks remainopen.

- Source diagnostic precision: capture2 preserved-input903 includes a COMPLETE1428640B b0ddb21d…420d blob and actual exactresource equality, alongside its separate64B headerdiagnostic. The earlier raw-header-only wording applies only to that diagnostic, not the whole input evidence. This does not prove KV0781 LoadData64 or DSO-carrier membership; source/native/replay flags remainfalse.

- Canonical47 semanticmetadata successor:14pureguards/source review PASS; actual retiredcapture2 revalidation exits1 with explicit REFUSED report, classifying40attention launches/23FAprivate pointers/7copies/KVregistration chain. Original three numerical failures and24projection/17workspace refusals remainfatal. Current fixed diagnostic does not grant native/math/model/perf acceptance; exact fresh-key schema guard required before generic reuse.
- Initial47call0 QKV/reducer ROOT resource inspection: wholeheld-image identity/ELF bank descriptors match actual reflected spans/base0x160; NVIDIA CPU reader confirms producer static49152 +actual dynamic24576=required73728B, reducer static0. Producer relocatable bank remains exactname/section-info join only, not a defined-symbol/linker claim. Four pure address/coverage guards PASS; reducer60..63 is unobserved ABI padding. Separate additive binding/numerical/physical gates pending; no new native/model/perf permit.

- New47 QKV fixedOBS reference prerequisite PASS:12sourceguards and actual offline source/33primary/320roles/twooriginalowners/44metadata/16wholearchives checks; values39/40/41 plus actual inputdual F32sum866304B already captured, no newobserver. New15312B/8ff503d8…fe1b receipt preserves original discovery2 and first3 receipts; all42 acceptance inherited, no newfull24 recompare. Packet source11children/52deps and allnonpointer/60B/21parameter comparisons match actual. Real33node/42value source composed in ISO; strictClippy/48canonical/90vendor checks PASS before final helper, six budget guards and independent326856780Bdevice/719156132Bhost bounds justify finite320/704MiB testing envelopes;319 is conservativeupper, not actualhandles. Two captured complete ABI seals materialized, actual partial-main faults authored. Physical33nodeQKV/source/state/fault/fullgate successor acceptedphase62 above; laterQK/KV/FA/fullmodel/serving/perf gates remainopen.
````

### pipeline-composition — 53 noncompleted occurrences

<a id="pipeline-composition-l142"></a>

#### open · [L142–144](../../plans/pipeline-composition/TODO.md#L142-L144)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Production/test separation — 26 Oct`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L142-L144@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `3f1a70562e38c08684352b2fa65893fb3f720175022bca27a299b073b3d0ab28`.

````markdown
- [ ] Finish remaining qualifier/fault and captured testcase dispatch boundaries
  across model/GDN/native paths. Parametric GDN source proof/design is ready;
  actual runtime implementation and source/state/physical validation remain.
````

<a id="pipeline-composition-l154"></a>

#### open · [L154–156](../../plans/pipeline-composition/TODO.md#L154-L156)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Model containment and naming — 26 Oct`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L154-L156@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `76f2de44d940dc8e76577367c98f48e2625b62e124a1be5c67cddf6fa1b98da0`.

````markdown
- [ ] Extend source-bound .8B qualification to fresh47–50-token prompts, genuine
  long contexts and original BF16 CPU/HIP/Vulkan routes. Existing cached
  frontiers48–50 and source fixtures do not establish those capabilities.
````

<a id="pipeline-composition-l157"></a>

#### open · [L157–162](../../plans/pipeline-composition/TODO.md#L157-L162)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Model containment and naming — 26 Oct`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L157-L162@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `4a8d828a3f6f80bf282b08cd989586cf2df1da663732d53b699ebc1db196e94f`.

````markdown
- [ ] After source-bound correctness and legitimate source access: current
  independent B4 physical/CPU suite, then optimized release +quiet same-attempt
  C16 for both small models; then trusted x2 original27B full-window capture
  under its remaining gates. [l5 benchmark container termination](../cpu-gpu-optimization/BENCHMARK_CONTAINER_TERMINATION.md)
  now has a supported cleanup-escalation inference; historical signal/OOM proof
  remains unavailable. No failed scope or capture gate is waived.
````

<a id="pipeline-composition-l166"></a>

#### open · [L166–167](../../plans/pipeline-composition/TODO.md#L166-L167)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Shared HIP / model composition checkpoint — 24 Oct`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L166-L167@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `cb20de3effd0ff1d0c80085603101a71fceaa7d4d0b208e7964e95a5cfd84e3b`.

````markdown
- [ ] Natural independent-reference and explicit Gemma row-hit/transfer scopes;
  no public serving, current source21 x2, general-fit or performance grant.
````

<a id="pipeline-composition-l175"></a>

#### open · [L175–180](../../plans/pipeline-composition/TODO.md#L175-L180)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Model optimization reconciliation — 24 Oct`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L175-L180@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `64b42911d90e545e296b6d9714964f598cab9b030958e098b5ad970c57ad5d60`.

````markdown
- [ ] Map each applicable strategy to existing primitive→model operation→family→
  model/profile owners: shared matrix storage/layout, compact outputs, prepared
  recipes, temporal work, pages/COW/prefix, graph ownership and target verifier
  once; Qwen GDN/state and Gemma PLE/KV-sharing/liveness select eligibility.
  No second registry/IR/runtime or per-token feature search. Reuse map/callers,
  source/math/backend/phase limits and complete costs precede profile selection.
````

<a id="pipeline-composition-l181"></a>

#### open · [L181–191](../../plans/pipeline-composition/TODO.md#L181-L191)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Model optimization reconciliation — 24 Oct`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L181-L191@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `e1c5e1cf466224e2813c79a20e91f72539564311f1a985a2043978adaaf3e0dc`.

````markdown
- [ ] Preserve distinct numerical strategies: source113 CPU-order BF16 versus
  BLAS/TensorCore/split-K, packed casts and explicit lossy codecs. A common
  operation does not share its source permit, arithmetic grant or StateRoot.
  Genuine N1→B4 cross-width diagnostic first differs at QKVZ2; GdnOut7 has a
  separate projection difference even with exact input. Original same-prior B4
  comparison now passes all15transitions/30plain+observed/fullstate/fullV/24sums/
  natural bits, with genuine HTTP N1→mixedB4/refill separately exact. Explicit
  compositional proof, not an independently generated original public trajectory
  or universal N1/B4 bit equality. Preserve distinct strategies/source permits;
  no production reference imports, weakened limits or cross-width grant.
  [Astra contract review](../../runs/architecture-astra-cross-width-contract-20261024/DESIGN.md).
````

<a id="pipeline-composition-l192"></a>

#### open · [L192–194](../../plans/pipeline-composition/TODO.md#L192-L194)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Model optimization reconciliation — 24 Oct`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L192-L194@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `24ee622e6074fca26ec5adacf1e61a517b0a9bf5517bf5c643256e942360ddb7`.

````markdown
- [ ] Use temporal/ragged pipeline liveness for Gemma E2B/E4B shared-tail fast
  prefill; retain requested prompt logits/taps and every future KV dependency.
  It is inapplicable to12B/26B/31B. Request B4 is not temporal prefill.
````

<a id="pipeline-composition-l195"></a>

#### open · [L195–213](../../plans/pipeline-composition/TODO.md#L195-L213)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Model optimization reconciliation — 24 Oct`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L195-L213@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `a6a0d5f3b3499954b39951bda20056b68b532cbde7fa4b1b88aee6e75b72ac77`.

````markdown
- [ ] Adopt source-bound phase selection for2B cachedB4 through the existing
  family/native adapter: one selector governs geometry/storage/inventory,
  native/GDN/RMS definitions, readiness, input admission and scheduler. Genuine
  N1prefill/partialfallback; unsupported2BinitialB4 refused before preparation.
  Derive width4 root18 from actual8current+4candidate+2Pending+2snapshot+
  2displaced, preserving sole retained capacity and physical readers. Exact
  selected-phase costs within existing4608MiBprovider+512backend/4GiBHOST;
  authentic2B native/RMS/catalog and same-prior numerical proof required.
  Guarded source13 adoption now includes the shared model foundation and2B
  specialization together. Authentic CPU inventory/reopen reports root18,
  fullhead1,017,118,720B, provider upper4,767,472,640B and HOST4,080,276,698B;
  remaining64,365,568B/214,690,598B respectively. These are complete source-derived
  bounds, not themselves GPU evidence. Optimized13 CLOSED0; original19physical
  scopes and genuine2B15transition trajectory/export/healthyretirement PASS.
  Original2B SAMEPRIOR30plain+observed/8112sections/fullstate/fullV/natural bits
  now exact; first newcached fault fails an unchanged compound retention check,
  last2faults unrun. Fixture-only14 logs full actual snapshots without relaxed
  checks; API/HTTP/timing/full assembled acceptance remain held.
  [Astra architecture verdict](../../runs/model-execution-astra-phase-profiles-review-20261024/REVIEW.md).
````

<a id="pipeline-composition-l226"></a>

#### open · [L226–239](../../plans/pipeline-composition/TODO.md#L226-L239)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 23 Oct, Astra high architecture review`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L226-L239@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `42716834d66e4b81e4f83a092c4775695a021b4fc44b5de1f6cbe61e5823f0fd`.

````markdown
- [ ] Implement the [Astra next-native redesign](../../runs/architecture-astra-next-native-review-20261023/DESIGN.md)
  through Sol HIGH: source-neutral physical owner plus source-bound services,
  same envelope/queue/ledger/fence, exact model sub-admissions and independent
  source/close/reload lifetimes. No per-model bootstrap/cap increase or source
  equality weakening. Source work isolated while frozen single-source Gemma
  actual qualification and binder316 checks continue.
  Physical13 now integrated MAIN/ISO345 with guarded original parents/modes/
  README; logical564/provider+default strict PASS, Astra architecture accepts/no
  concrete must-fix. Original local required release/full both-model CUDA/state/
  fault/API checks PASS; x2 native3/portable6/rawtransport12 and corrected shared4
  HIP/Vulkan healthy/unknown PASS. Fixture Oracle shadow aliases corrected only;
  production unchanged in345v3. Interrupted original controller and missing
  local13 closure preserved separately from successful fresh replay. Final gate
  pending after coherent contributions; measured release remains316.
````

<a id="pipeline-composition-l240"></a>

#### open · [L240–285](../../plans/pipeline-composition/TODO.md#L240-L285)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 23 Oct, Astra high architecture review`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L240-L285@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `1958f17442147daa13808a01054f4f39334cef1238f08c4b289c310da05818b7`.

````markdown
- [ ] Extend the original HIP actor's bounded prepared-set lifecycle for multiple
  independently source-bound models on one physical owner. Current actor
  deliberately refuses a second live set/legacy/graph; shared admission alone
  does not grant simultaneous Gemma/Qwen native execution. Preserve exact source/
  arithmetic/specialization identities, original envelope/global fence and actual
  per-set module/flight retirement. No learned payload sharing by code identity.
  [Astra selected seam](../../runs/architecture-astra-shared-native-20261023/DESIGN.md):
  nonzero never-reused logical StreamId through all compile/cache/IO/driver joins,
  immutable source-service Client including internal enqueue paths, maxphysical
  streams1 retained. Cold32/64/32GiB provider/device/host remains a candidate;
  derive quotas from complete costs before bootstrap. The earlier two12GiB
  draft must not force Gemma's resident projections back to streamed storage.
  Current candidate Qwen12GiB/Gemma16GiB plus one4GiBcontext fits the32GiB
  provider envelope arithmetically; complete host/init/scoped-key peaks remain
  to qualify. Qwen9,917,743,073B is model-only; Gemma13,174,234,632B plus the
  shared context gives27,386,945,001B before new metadata. No duplicate context
  subtraction, live cap growth or optimistic compiler-host release. Sol HIGH
  implements; actual multi-set remains open.
  [Concrete ABI review](../../runs/architecture-astra-residency-and-set-key-20261023/DESIGN.md):
  private scoped key nests the FULL original KernelId; bounded full-Eq routing
  reaches actual driver lookup despite Command dropping logical StreamId.
  Fixed SingleCompat/Scoped mode preserves original callers without fallback
  mixing. Per-set source/module/admission and global unknown fence remain.
  [Astra HIGH draft-source/API review](../../runs/pipeline-composition-astra-hip-multiset-review-20261024/REPORT.md)
  approves the bounded direction: shared-only Qwen reload after actual final
  reader/native/service retirement; exact pinned-client preflight with a cold
  Arc/Weak identity, without cache backedges. Sol implements both corrections.
  Cold full-Eq routes and actual family consumers were inspected; completed
  fixes, full costs, checks and two-source physical lifecycle remain unqualified.
  ROOT reviewed the complete scoped-source delta and compiler successor: Shared
  retains up to three admitted8GiB compiler leases on the original persistent
  context (two initial sets plus one reload), including cancelled attempts; model
  unload gives no compiler-cache credit. Reserve before queue submission so a
  fourth ordinary refusal cannot fence a healthy sibling. SingleCompat keeps its
  existing policy. Conditional complete HOST upper33,668,886,252B includes two
  scoped-metadata maxima and128MiB fixture allowance, leaving690,852,116B under
  fixed32GiB; actual costs/fit and two-source lifecycle remain open. Grouped
  GPU-free successor checks run separately; no physical or performance grant.
  [Compiler lifetime correction](../../runs/pipeline-composition-hip-multiset-sol-high-20261024/compiler-successor-v2/REVIEW.md).
  Frozenv4 whole7stage CPU unit now PASS: all3strict,171resident/12core/7compiler
  tests and2actual-type source-expression controls. Actual compiled scoped metadata
  is observed per set and joined exactly, not inferred as a residual. Corrected
  fullHOST upper34,205,757,164B includes separate512MiB supervisor, leaving
  153,981,204B under32GiB; startup checks actual coldplan+fixture+controller.
  ROOT complete source/control/owner review; isolated x2 optimizedbuild active,
  genuine same-actor healthy reload/globalunknown cases still pending.
````

<a id="pipeline-composition-l286"></a>

#### open · [L286–289](../../plans/pipeline-composition/TODO.md#L286-L289)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 23 Oct, Astra high architecture review`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L286-L289@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `30a3abd090fd77392df50748f0f63e3f3a8b45fa907592237c8ec84319a37363`.

````markdown
- [ ] Record Qwen27B's one family traversal through the existing backend/plan
  with explicit MKL arithmetic revision,112logical domains/128ordinary planes/
  fullConv4, before native/multi-model serving. Current scalar-exp/H16/short-HF
  binders refuse; accepted CPU INITIAL1→cached1 proof cannot grant their reuse.
````

<a id="pipeline-composition-l291"></a>

#### open · [L291–293](../../plans/pipeline-composition/TODO.md#L291-L293)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 23 Oct, Astra high architecture review`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L291-L293@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `540580c12672bab38b7fec83159f0411ff97199a5f4414280462aa19927a5dc6`.

````markdown
- [ ] Qualify the integrated `TextCallGeometry` extraction and actual Gemma
  consumer. It separates source-bound call metadata from model geometry and
  finite numerical/backend eligibility; six copied-core tests pass only.
````

<a id="pipeline-composition-l294"></a>

#### open · [L294–296](../../plans/pipeline-composition/TODO.md#L294-L296)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 23 Oct, Astra high architecture review`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L294-L296@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `de2e505634f859c5038762fd18c0a3f895ecb116d171abbac47eddb2b88e2efd`.

````markdown
- [ ] Move remaining closed family builders/geometry out of core after generic
  semantic consumers migrate. Separate family, numerical and implementation
  identity; retain explicit rounding/reduction contracts and actual shape gates.
````

<a id="pipeline-composition-l297"></a>

#### open · [L297–298](../../plans/pipeline-composition/TODO.md#L297-L298)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 23 Oct, Astra high architecture review`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L297-L298@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `8c46ac335a8a0356f872f2cb627928f53da1b6726680878db4d08573eefa573b`.

````markdown
- [ ] Use one prepared-model serving adapter for validation/session/accepted
  step/committed output/cancel, retaining the original router/workers/roots.
````

<a id="pipeline-composition-l299"></a>

#### open · [L299–305](../../plans/pipeline-composition/TODO.md#L299-L305)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 23 Oct, Astra high architecture review`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L299-L305@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `24b9ec297251edc1fc0ebed482fd6a0c7badb7e8b499399d63b3fa0bafa69bdc`.

````markdown
- [ ] Implement sealed NativeCohort in the SAME node loop: authentic perrow
  record/runtime/prior/candidate/borrowedphase joins through validation, View/
  storage/finite certification and terminal grouped proposals, original physical
  cohort flight/Ready and single all-or-none publisher. Physical188/376 is an
  explicit request axis, not temporal tokens. Call-local body and enforcedmax8
  sessions preserve22root bound. Actual canonical47B4 source discovery passes;
  complete native arguments and genuine Rust188prefill→same-rootB4 remain open.
````

<a id="pipeline-composition-l306"></a>

#### open · [L306–328](../../plans/pipeline-composition/TODO.md#L306-L328)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 23 Oct, Astra high architecture review`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L306-L328@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `52ac3f94c24d48c276b4720b9af296970d5918833112ee2deaac8e599541d11a`.

````markdown
- [ ] Share immutable recipe/topology/SSA/storage/native candidates inside the
  existing record API; bind small source-issued invocation/root/phase data late.
  Current percall record charge4MiB must not multiply into permanent Bplan
  rebuild/hash overhead. No new template IR or firstrow provenance shortcut.
  Sol18 now integrated MAIN/ISO314selected source paths; Astra HIGH completed
  architecture-only review and accepted with no must-fix. Source/profile/shape
  cache40 retains original host charges; topology/source semantics Arc-shared,
  actual operation/provenance remains invocation-local; warm bind outsideMutex.
  Per-call record bound512KiB, conservative fullcache160MiB. Core201/Qwen87/
  GPU549/default+provider strict PASS. Original source/site/native validation and
  wrapper allocations remain. Both models' complete actual CUDA/state/fault/API
  plus streamed2B faults PASS; .8B quiet C1+70.73%/C4+28.81%/C8−2.58%, singlepass.
  Full assembled gate15 LAST PASS1752Rust/0failed/89ignored/269suites/allchecks;
  314sourcepaths/rootREADME unchanged. No zero allocation/general pipeline claim.
  [Architecture review](../../runs/architecture-astra-prepared-review-20261023/DESIGN.md).
  Residual Sol8 binder now integrated MAIN/ISO316: concrete existing ColdModel
  cache40, genuine invocation validation outside lock, original retained
  witness/site charges512KiB each +64KiB owner. GPU551/provider+default strict
  pass; original joint release and both models' full real source/state/fault/API
  plus streamed2B faults PASS. Quiet .8B C1+78.62%/C4+26.67%/C8−1.09%, singlepass;
  six original owner closures healthy. Final isolated gate16 PASS1752Rust/0failed/
  89ignored/269suites/allchecks;316sourcepaths/rootREADME unchanged. Astra
  next-native pass is architecture/planning only; Sol HIGH applies the redesign.
````

<a id="pipeline-composition-l329"></a>

#### open · [L329–338](../../plans/pipeline-composition/TODO.md#L329-L338)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 23 Oct, Astra high architecture review`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L329-L338@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `00b8125af50a8494b9235c568dd4d45ec919ac935defb24a3cec8fc6cea86d5e`.

````markdown
- [ ] Before Gemma HIP merge: retain authentic source/snapshot lifecycle through
  native unknown/retirement; unsafe lowlevel boundary with one validated safe
  facade caller, no safe Any-owner proof. Keep fixed hardware/context/compiler/
  module policy in shared backend preparation. Sol44v5 implements these Astra
  corrections and complete source/state/lifetime qualifier in donor; reachable
  strict/source tests pass. Original frozen MAIN314+44 on x2 now passes full
  seven-case/all35/state/fullV/natural/lifetime matrix and fresh actual unknown
  retention. Guarded Gemma44 integrated MAIN/ISO343; successor shared physical
  owner/serving/performance stay unqualified. Prior316 final gate16 passed;
  next assembled gate follows coherent successor/affected physical checks.
````

<a id="pipeline-composition-l339"></a>

#### open · [L339–341](../../plans/pipeline-composition/TODO.md#L339-L341)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 23 Oct, Astra high architecture review`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L339-L341@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `4303b5c2eedaa9bd7b22101def027f1e153cfcff82b77f70a07e5e39b2560d83`.

````markdown
- [ ] Demonstrate one forced ordinary versus optimized region/profile slice
  with joint liveness/observers and device-resident typed stage edges. Build on
  existing region metadata and physical lowering, without a universal framework.
````

<a id="pipeline-composition-l342"></a>

#### open · [L342–346](../../plans/pipeline-composition/TODO.md#L342-L346)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 23 Oct, Astra high architecture review`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L342-L346@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `ac0f6a303b10927a7d434008e2fe1f686da6ba959be4e94cd20a94b0cac165b7`.

````markdown
- [ ] Later multidevice/MoE reuse: source/range/codec identity separate from
  physical slot/generation/placement; pin queued/captured/inflight readers,
  charge migration old+new/staging, preserve fairness and complete root shards.
  Backend tuples qualify CUDA/HIP/Vulkan separately. No current sparse,
  distributed or media execution claim follows from this review.
````

<a id="pipeline-composition-l358"></a>

#### open · [L358–361](../../plans/pipeline-composition/TODO.md#L358-L361)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 21 Oct, resident stages and fusion`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L358-L361@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `8ab55547c9e9afe69b97c97331a469745a7fadfb39986ab9d07caf292e7fe069`.

````markdown
- [ ] Carry eligible stage outputs through typed leased device bindings with
  source/profile/root, owner/device/context/generation/range, layout/extents,
  access and actual producer dependencies. Stage boundaries alone introduce no
  host return, serialization, allocation, payload copy or synchronization.
````

<a id="pipeline-composition-l362"></a>

#### open · [L362–365](../../plans/pipeline-composition/TODO.md#L362-L365)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 21 Oct, resident stages and fusion`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L362-L365@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `3565eefe903840c886d0555e9c86b62e4f1d212cf26ce60507bb5d0da71ac046`.

````markdown
- [ ] Prepare cross-stage resident handoff, device chains/graphs, exact fused
  regions and explicit bridges through existing ports/providers; force the
  accepted decomposition for comparison. Resolve finite compatible candidates
  outside the hot loop; no new interpreter, tensor runtime or execution IR.
````

<a id="pipeline-composition-l366"></a>

#### open · [L366–369](../../plans/pipeline-composition/TODO.md#L366-L369)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 21 Oct, resident stages and fusion`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L366-L369@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `a0de9260078611455a180810079cd0f536c329a2326ce03169ea65a4374fef93`.

````markdown
- [ ] Plan joint placement/layout/liveness and admitted scratch/graph/staging
  across eligible producers and all consumers. Qualify caller-owned native
  component buffers and stream hooks; expose hidden pools/copies/waits and
  retain cases where a copy or smaller region improves complete-task cost.
````

<a id="pipeline-composition-l370"></a>

#### open · [L370–374](../../plans/pipeline-composition/TODO.md#L370-L374)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 21 Oct, resident stages and fusion`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L370-L374@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `57dadbb91a75d6cf5a00cb7a6926d86082e4620cf150e23f680cb319207ead93`.

````markdown
- [ ] Keep qualified intermediates, model state and compact generation feedback
  on device. Use real event dependencies and bounded control/status transfers;
  preserve sampler/grammar, context grants, independent model publication,
  cancellation/fairness and output-credit boundaries. Require typed trained/
  codec bridges between models; graph composition does not erase host decisions.
````

<a id="pipeline-composition-l375"></a>

#### open · [L375–377](../../plans/pipeline-composition/TODO.md#L375-L377)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 21 Oct, resident stages and fusion`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L375-L377@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `56030f1472e204f048b072332c043cac87bf4031c31e9c43e762aaabfcdbea6a`.

````markdown
- [ ] Explain and attribute each required materialization/copy/readback/wait and
  graph split. Compare forced ordinary, resident-chain, graph and fused paths
  under the same numerical/approximation contract, source and output consumers.
````

<a id="pipeline-composition-l378"></a>

#### open · [L378–383](../../plans/pipeline-composition/TODO.md#L378-L383)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 21 Oct, resident stages and fusion`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L378-L383@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `218aa608246c76be65cd8cf62c3e3c8bec8e41fdc8c523414b3f73c056c4f1f5`.

````markdown
- [ ] Prove zero intermediate H2D/D2H payload traffic and no stage-only host wait
  on a qualified compatible fixture; test incompatible bridges/refusals,
  tails/ragged/multiple-consumer/debug/alias, cancel/backpressure/reload and
  unknown completion. Separately qualify trained complete-task results and
  actual physical peak/tails; start NVIDIA, then Strix Halo, without assuming UMA
  implies zero-copy. Appropriate real-source/physical and full check.sh gates.
````

<a id="pipeline-composition-l390"></a>

#### open · [L390–393](../../plans/pipeline-composition/TODO.md#L390-L393)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 21 Oct, approximation/components/shared policy`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L390-L393@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `ab9256f39c71f6d6b099eefe3aec45f602954b1ad55891485d4e46e42be2ba68`.

````markdown
- [ ] Support explicitly approximate profiles beyond derived weight precision:
  declared methods/ranges, changed semantics, held-out quality limits and
  permitted adaptive choices; separate variant correctness from original-model
  fidelity. Bind effective variant/state formats into cache/snapshot/graph keys.
````

<a id="pipeline-composition-l394"></a>

#### open · [L394–398](../../plans/pipeline-composition/TODO.md#L394-L398)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 21 Oct, approximation/components/shared policy`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L394-L398@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `247693ac06a859306cbc82b8015928dddd654ef24d95d4921a047a8eb7200289`.

````markdown
- [ ] Permit interchangeable allocator/pool, provider, codec, scheduling,
  residency and transfer implementations under the original ownership spine.
  Define replacement boundaries and old-allocation retirement; test old/new
  coexistence, failure, graph readers, cancellation and unknown completion.
  Replace/rebuild unsuitable modules when real implementations justify it.
````

<a id="pipeline-composition-l399"></a>

#### open · [L399–402](../../plans/pipeline-composition/TODO.md#L399-L402)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 21 Oct, approximation/components/shared policy`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L399-L402@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `1c176bc63713e0d4b6881779403d78adcc06e27393e93e4a59daaa371027ea32`.

````markdown
- [ ] Resolve one shared versioned configuration visible to participating
  modules, with explicit local overrides, bounded operational observations and
  explainable choices. Existing coordination admits joint batch/graph/draft/
  prefetch cost; test conflicts, stale observations and reload boundaries.
````

<a id="pipeline-composition-l403"></a>

#### open · [L403–406](../../plans/pipeline-composition/TODO.md#L403-L406)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 21 Oct, approximation/components/shared policy`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L403-L406@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `e04fc8a21942df0fcbbfc991c712cd1f46c3e1228d999850690b6e3beb6abfb0`.

````markdown
- [ ] Apply audited MIT/Apache static-native reuse first to every implementation
  and harness; otherwise directly port the smallest useful implementation and
  fixtures, establish faithful behavior, then integrate through our contracts.
  Preserve upstream notices; no Python/PyTorch production runtime is implied.
````

<a id="pipeline-composition-l407"></a>

#### open · [L407–410](../../plans/pipeline-composition/TODO.md#L407-L410)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 21 Oct, approximation/components/shared policy`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L407-L410@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `733fac58579e89a4cb8df2aa9c8156f725296420ef004adc8de5cf181c238292`.

````markdown
- [ ] Consolidate profile/phase/backend equivalence acceptance: solo/concurrent,
  B1/batched reorder/refill, draft off/on/every keep, fresh/restored/COW,
  ordinary/graph/fused and contiguous/paged. State required bitwise/error/
  distribution/changed-semantic relation; reject incompatible combinations.
````

<a id="pipeline-composition-l411"></a>

#### open · [L411–414](../../plans/pipeline-composition/TODO.md#L411-L414)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Owner additions — 21 Oct, approximation/components/shared policy`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L411-L414@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `69d397bc5209d45acb94045ebc1ff63dbad0c2e5fdf8727475347e0a7e518798`.

````markdown
- [ ] Exercise the first useful region/profile with one reused or ported
  component and the native correctness harness; preserve the baseline, complete
  state/physical gates and measured complete-request results. New abstractions
  must not delay current parity or require a second execution stack.
````

<a id="pipeline-composition-l423"></a>

#### open · [L423–424](../../plans/pipeline-composition/TODO.md#L423-L424)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Recursive composition and profiles — owner request21 Oct, design first`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L423-L424@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `79e5e3bf970c301354e0d323fc1ec6b3f325e53c230dee80821d75f15bda5a6e`.

````markdown
- [ ] Define primitive semantics separately from their kernel implementations;
  typed ports/shapes/source/numerical policy and declared state/read/write effects.
````

<a id="pipeline-composition-l425"></a>

#### open · [L425–427](../../plans/pipeline-composition/TODO.md#L425-L427)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Recursive composition and profiles — owner request21 Oct, design first`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L425-L427@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `52b0acec3b029e01c5bfc99f58cb8a1847bdd35e5d930fcc44e731fe28abc666`.

````markdown
- [ ] Define a pipeline node that may reference a bounded subpipeline: one
  inference's operator/kernel regions and compound multimodal task stages use
  consistent composition laws while retaining their distinct value/frontier types.
````

<a id="pipeline-composition-l428"></a>

#### open · [L428–431](../../plans/pipeline-composition/TODO.md#L428-L431)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Recursive composition and profiles — owner request21 Oct, design first`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L428-L431@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `461975ad8a3b57d227eae8a1a4b7c73478095687794d3b8ed93dbd43fe534af5`.

````markdown
- [ ] Define immutable profiles combining named pipelines and permitted source/
  arithmetic/workload/placement strategies. Separate defaults/preferences from
  non-overridable source, authorization, physical and completion requirements.
  A saved profile is not a live SourceIdentity, Ready or allocation permit.
````

<a id="pipeline-composition-l432"></a>

#### open · [L432–435](../../plans/pipeline-composition/TODO.md#L432-L435)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Recursive composition and profiles — owner request21 Oct, design first`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L432-L435@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `3ee2990c70dbd82b5dc38f4253de9aa17001a10aee2da403a04a051f71983e4b`.

````markdown
- [ ] Compile hierarchical views into existing semantic/physical plans; legal
  flatten/in-line/out-line, exact region replacement, prepared shape selection,
  graph replay and optional fused kernels. Preserve two enduring representations,
  no new interpreter/runtime/model loop/allocator or per-token plan search.
````

<a id="pipeline-composition-l436"></a>

#### open · [L436–440](../../plans/pipeline-composition/TODO.md#L436-L440)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Recursive composition and profiles — owner request21 Oct, design first`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L436-L440@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `1ba640cdf4b2f4a83b393b72084b4cf69a5c0d3babe2a64f46ad65c18adb7db9`.

````markdown
- [ ] Specify child-region effects/exports, alias/liveness after fusion, late
  source/root/epoch/generation binding, cancellation, backpressure and completion.
  Inner mathematical regions prepare; original session writer publishes once.
  Compound task boundaries must not imply global rollback of already committed
  model/media stages or allow recursive independent publishers.
````

<a id="pipeline-composition-l441"></a>

#### open · [L441–443](../../plans/pipeline-composition/TODO.md#L441-L443)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Recursive composition and profiles — owner request21 Oct, design first`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L441-L443@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `c31f3e8006943e8a15850e56efb5c69e8de8934b0a8c9f96c2e2e6e2760aa721`.

````markdown
- [ ] Profile examples: resident interactive, concurrent throughput, long/shared
  prefix, memory-bounded dense27B; future sparse/derived-precision/media/distributed.
  Mark prerequisites, conflicts, unsupported/fallback and actual qualification.
````

<a id="pipeline-composition-l444"></a>

#### open · [L444–446](../../plans/pipeline-composition/TODO.md#L444-L446)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Recursive composition and profiles — owner request21 Oct, design first`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L444-L446@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `65422a677ff0f95ed22b72d51b962dd7831e7e9090164824aa224bef2d43ba82`.

````markdown
- [ ] Compare minimal efficient kernel paths with ordinary decomposition and
  captured schedules; retain cases where fusion loses occupancy or overlap.
  Distinguish semantic nodes, submissions and actual device kernel invocations.
````

<a id="pipeline-composition-l447"></a>

#### open · [L447–450](../../plans/pipeline-composition/TODO.md#L447-L450)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Recursive composition and profiles — owner request21 Oct, design first`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L447-L450@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `0e100932cab43d672f6a817f959dbf3eecff9362cf55e1b37714156508279594`.

````markdown
- [ ] After parity and matched baseline, measure source-only preparation
  (e.g. exact gamma widening/shift) and legal normalization-region fusion as
  early candidates; preserve casts, original model ownership and admitted bytes.
  Compare complete-request cost and actual dispatches before promotion.
````

<a id="pipeline-composition-l451"></a>

#### open · [L451–453](../../plans/pipeline-composition/TODO.md#L451-L453)

Heading: `TODO — Single-process model and pipeline composition > Uncommitted > Recursive composition and profiles — owner request21 Oct, design first`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L451-L453@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `2029415fc67ac2e204b4f8818b7f6b6f49810854a155a57c53dd7894d2556bee`.

````markdown
- [ ] Review extension drills for a new primitive/model/provider/task without
  unrelated scheduler/API changes; define exact-source/all-state/physical and
  complete-request performance gates before implementing these abstractions.
````

<a id="pipeline-composition-l772"></a>

#### open · [L772–773](../../plans/pipeline-composition/TODO.md#L772-L773)

Heading: `TODO — Single-process model and pipeline composition > Deferred`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L772-L773@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `4de1afcccd6353c1221dccd0f20dce0f0555a0339c4b54804778e16c66d7dd4d`.

````markdown
- [ ] OSS release/distribution naming and "latest" pointers remain
  observational until an exact immutable source and codec version is bound.
````

<a id="pipeline-composition-l774"></a>

#### open · [L774–775](../../plans/pipeline-composition/TODO.md#L774-L775)

Heading: `TODO — Single-process model and pipeline composition > Deferred`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L774-L775@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `f41b739bd3f4846c62044b3a7e027d9bf4fedab5e1b2c1d26c05fb55f5e6a014`.

````markdown
- [ ] Remote routed pipelines/peers, encrypted hosted gateways and
  implicit zero-copy cross-device are not enabled by source-free shapes.
````

<a id="pipeline-composition-l789"></a>

#### open · [L789–795](../../plans/pipeline-composition/TODO.md#L789-L795)

Heading: `TODO — Single-process model and pipeline composition > Deferred > Runtime work deferred until after core model priorities`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L789-L795@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `63e4f2375855ad5856aee319964b1b483e5616f730f28d0342ce3e869f1cdd96`.

````markdown
- [ ] Connect authenticated application/conversation history providers to
  ContextOwner, with durable and revocable grants, owner-authenticated
  tenant+application+session identity, event/graph epoch binding to the
  existing TaskCoordinator and physical memory/residency admission. Wire
  source-real calibrated NanoJev/other decision heads and streaming media
  A/B tests before enabling an actual context-fed ASR/translation router.
  Do not infer inference support from the in-memory source fixture.
````

<a id="pipeline-composition-l797"></a>

#### open · [L797–799](../../plans/pipeline-composition/TODO.md#L797-L799)

Heading: `TODO — Single-process model and pipeline composition > Deferred > Runtime work deferred until after core model priorities`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L797-L799@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `e8a0278bf7227bdf96e86408db0886c9a116a5a2abf792aed39f5020f49fb248`.

````markdown
- [ ] Source execution: bind actual learned model, tokenizer, processor
  assembly and port formats in original ModelCatalog/TaskDriver before
  enabling any pipeline run or route listening.
````

<a id="pipeline-composition-l801"></a>

#### open · [L801–804](../../plans/pipeline-composition/TODO.md#L801-L804)

Heading: `TODO — Single-process model and pipeline composition > Deferred > Runtime work deferred until after core model priorities`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L801-L804@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `4712ff7e6650214857126528c112e01cd7157854352deccb37634b6b5b8df319`.

````markdown
- [ ] Single-process multi-model scheduler: immutable source residency,
  exact shared instance reuse/replicas, typed cross-model buffer leases,
  physical ready fencing, bounded ingress/admission/cancellation and
  model-specific batch/fairness. One TaskCoordinator publishes StateRoots.
````

<a id="pipeline-composition-l806"></a>

#### open · [L806–808](../../plans/pipeline-composition/TODO.md#L806-L808)

Heading: `TODO — Single-process model and pipeline composition > Deferred > Runtime work deferred until after core model priorities`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L806-L808@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `720e5fb8978175868e330a42decb9ef013901eb867f5ce70652aa1701b01cc7e`.

````markdown
- [ ] Pipeline runtime/hot reload: per-session model identity, graph generation,
  buffered media/source epoch, legal stage yields, failure rollback and
  atomic route transition. No stale audio, image or cross-tenant leakage.
````

<a id="pipeline-composition-l810"></a>

#### open · [L810–812](../../plans/pipeline-composition/TODO.md#L810-L812)

Heading: `TODO — Single-process model and pipeline composition > Deferred > Runtime work deferred until after core model priorities`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L810-L812@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `26c925f8efaaf9cb11d72e113ca2d726026f3a8cc70b788a265bf9532065b20d`.

````markdown
- [ ] API serving: explicit auth/bind/route config and tested OpenAI
  chat, ASR, TTS, images; SSE/WS/WebRTC soft-realtime with content/size
  bounds and accurate NOT IMPLEMENTED capability reports until qualified.
````

<a id="pipeline-composition-l814"></a>

#### open · [L814–822](../../plans/pipeline-composition/TODO.md#L814-L822)

Heading: `TODO — Single-process model and pipeline composition > Deferred > Runtime work deferred until after core model priorities`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L814-L822@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `b3d99e81a5fff31130b1980e31b915a1dc065a0c7c54fb380f726d881abe5d82`.

````markdown
- [ ] Context-aware source-real voice pipeline: stage-specific ASR prompt/
  vocab support or separate contextual transcript reranker, acoustic-first
  diarization and source-preserving contextual translation. Implement
  prior-committed same-session snapshot freezing, cancellation epochs,
  explicit consent, redaction, bounded history and no unauthorized
  cross-user disclosures.
  A/B compare WER/CER/DER/translation adequacy/latency for context
  ON/OFF; never override contradicted acoustic evidence. This does not
  follow from shape validation.
````

<a id="pipeline-composition-l824"></a>

#### open · [L824–827](../../plans/pipeline-composition/TODO.md#L824-L827)

Heading: `TODO — Single-process model and pipeline composition > Deferred > Runtime work deferred until after core model priorities`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L824-L827@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `6c9b4cff370b09d403dfda461408cc90d8d2c622ef168efc26f22e154e427ec6`.

````markdown
- [ ] Actual staged ASR + diarization (speaker-tagged timestamps), ASR
  -> Qwen -> TTS one coherent command; audio sample clocks/codebook
  playback/VAD/barge-in and image latent/complete preview frames with
  backpressure. Validate codec/latency/source on real media hardware.
````

<a id="pipeline-composition-l829"></a>

#### open · [L829–832](../../plans/pipeline-composition/TODO.md#L829-L832)

Heading: `TODO — Single-process model and pipeline composition > Deferred > Runtime work deferred until after core model priorities`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L829-L832@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `5bfb58e87809eed0f649ef082f60141e24405ecedc99b07e569bb3833a1b464a`.

````markdown
- [ ] Optimize model execution: Qwen3.5 2B, Qwen3.8 27B
  resident+SSD, Gemma E2B/26B/31B, then Qwen Flash-Next QSA/GDN/MoE/
  learned n-gram embeddings/MTP and separate speculative n-gram/draft,
  with measured model-real speed/quality/physical safety gates.
````

<a id="pipeline-composition-l834"></a>

#### open · [L834–837](../../plans/pipeline-composition/TODO.md#L834-L837)

Heading: `TODO — Single-process model and pipeline composition > Deferred > Runtime work deferred until after core model priorities`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L834-L837@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `983d4b9fd849bc5654790f27142bf1433dbf9bc1c1fbf1ae50f56d4136362b7c`.

````markdown
- [ ] Complete remaining exact Q2/Q3.x/Q4 quant source pipelines,
  full native image/audio/NeMo assemblies, token/codec quality and
  range/resume/cancellation before declaring "all models supported".
  Keep DeepSeek V4.1 Flash and GLM5.3 Flash strictly pipeline-only now.
````

<a id="pipeline-composition-l839"></a>

#### open · [L839–841](../../plans/pipeline-composition/TODO.md#L839-L841)

Heading: `TODO — Single-process model and pipeline composition > Deferred > Runtime work deferred until after core model priorities`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L839-L841@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `f5baf375f093c52ec8cba3976f0c995da65e3d7e0e59c7794e754b738c083fae`.

````markdown
- [ ] Trained and streaming multi-model parity, real media correctness,
  device completion, SLO/fairness and zero-copy lease gates are owned by
  the model-execution/serving/multimodal/memory TODOs above.
````

<a id="pipeline-composition-l846"></a>

#### ! · [L846–846](../../plans/pipeline-composition/TODO.md#L846)

Heading: `TODO — Single-process model and pipeline composition > Deferred > 24 Oct — parallel implementation checkpoint`.
Identity: `.agents/plans/pipeline-composition/TODO.md:L846-L846@0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b`.
Block SHA256: `8a3a0f9a1e0c4d2e72e5cf8384931bab562c06e9a5340140879bf972e117ac01`.

````markdown
- [!] HIPv4 first healthy57492 refused before child/model/GPU at old-Q import receipt schema; actual source/binary/owner closure preserved, no inference result. Controlv5 real metadata-only preflight accepted both actual receipt schemas but independent review found missing finalizer import alarm_until; distinct control-only successor required before fresh actual serial cases. No source rebuild/import/copy/download needed.
````

### platform-and-release — 17 noncompleted occurrences

<a id="platform-and-release-l12"></a>

#### open · [L12–20](../../plans/platform-and-release/TODO.md#L12-L20)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L12-L20@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `8d50accce3340f2fdb8f07a3795e8a359f54ccba623428ebac1fb774c9cb41b2`.

````markdown
- [ ] Owner23Oct: bring Vulkan execution to CUDA feature/correctness parity;
  use ROCm/HIP for AMD including pvl-x2 gfx1151. Vulkan source catalogue
  landed in `19ff059`; the real offline build PASSED on x2 25Oct (v-build,
  1,006.6s, three artifacts) and v-admin/v-bootstrap closures are green;
  V cases are blocked on the v-resident validator/producer field fix
  (PM handoff §2.3) — no Vulkan parity or performance claim yet. Reuse existing physical
  identity/arbiter/service/queue/quarantine. Separate compiled, available and
  qualified source/layout/math capabilities; refuse before allocation/launch
  when unsupported. Preserve gfx90c launch hold and source/SDK acquisition rules.
````

<a id="platform-and-release-l21"></a>

#### open · [L21–24](../../plans/platform-and-release/TODO.md#L21-L24)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L21-L24@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `014940655dc2b25572c29334a825badf911cc14dd9834751343b0016605345d9`.

````markdown
- [ ] Apply the owner-approved [static-native reuse/direct-port rule](../pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md#reuse-first-across-all-implementations-and-harnesses)
  to all implementations and correctness tooling: exact MIT/Apache file and
  transitive license/NOTICE review, suitable Rust/static C ABI library first,
  otherwise faithful smallest-component port with attribution and fixtures.
````

<a id="platform-and-release-l25"></a>

#### open · [L25–29](../../plans/platform-and-release/TODO.md#L25-L29)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L25-L29@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `9a434e3c01e37964df49f0d9ed7d862ba54f2dd02f09d9a40503c91a8be0bf2f`.

````markdown
- [ ] Prove actual final binary linkage for selected reused components across
  supported targets: native toolchain and ABI/error/thread/allocator boundaries,
  no accidental Python/PyTorch/foreign-server runtime, declared optional device
  interfaces, native fixture/diagnostic support and feature-isolated packaging.
  Statically linked components do not alone prove the entire binary is static.
````

<a id="platform-and-release-l30"></a>

#### open · [L30–32](../../plans/platform-and-release/TODO.md#L30-L32)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L30-L32@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `e6d256ebcb84f24de9ea659ebf950a1a96f1d4e1c127848a1297f81c7eb0154c`.

````markdown
- [ ] Verify shared effective-configuration explanation/revision handling and
  replacement compatibility across allocator/provider implementations; preserve
  original resource/session/completion authorities and old-reader retirement.
````

<a id="platform-and-release-l33"></a>

#### open · [L33–33](../../plans/platform-and-release/TODO.md#L33)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L33-L33@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `8bcfc2a3d8fd6f969721dfc2ea330fc795a9fa75b9ee0bbf57aec4e442419f2d`.

````markdown
- [ ] Generalize the working typed forward plan into verified effectful semantic regions/task control, closed provider substitution, bounded shape/kernel caches and explicit command/resource legality; no new persistent IR/runtime without a real need.
````

<a id="platform-and-release-l34"></a>

#### open · [L34–34](../../plans/platform-and-release/TODO.md#L34)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L34-L34@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `5e0b0878e996d0f5fab54367776111a03c1d1248f853ddcfe046099964212374`.

````markdown
- [ ] Prove artifact→model→compiled task→coordinator→output and explain-plan for supported tasks, including source-qualified preselected fallback and no selected-GPU-error CPU substitution.
````

<a id="platform-and-release-l35"></a>

#### open · [L35–35](../../plans/platform-and-release/TODO.md#L35)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L35-L35@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `a12499da57dd7012a587db956bc9c781e46cf3828e7a158d99b7d20673d50381`.

````markdown
- [ ] Run extension drills for one stateful semantic op, device/provider, non-token TaskProgram, quant/storage/API adapter and optional ParameterVersion/TrainStep seam; future training is not thereby implemented.
````

<a id="platform-and-release-l36"></a>

#### open · [L36–36](../../plans/platform-and-release/TODO.md#L36)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L36-L36@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `e8fb2e0215b11b7e3275c040774c79dbac415249e2267616e3582de41706a483`.

````markdown
- [ ] Finish per-file source/model/license/NOTICE/SBOM policy, malformed-input fuzzing and ownership/security review, authenticated APIs/private caches and qualified driver/device safety.
````

<a id="platform-and-release-l37"></a>

#### open · [L37–37](../../plans/platform-and-release/TODO.md#L37)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L37-L37@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `3679d7e1f8ad23a22c6fa99c42f4d2a2bfc53f8dee3194ae05e65cbef0619ea3`.

````markdown
- [ ] Verify Nix/module/native/cross packages and installer/release behavior on actual Linux/Windows/macOS target access; packaging definitions alone do not prove platform support.
````

<a id="platform-and-release-l38"></a>

#### open · [L38–38](../../plans/platform-and-release/TODO.md#L38)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L38-L38@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `02cf539eae05ccf6ea5e2fca39acd492e9749bf6b86a150702b58c084a0816a0`.

````markdown
- [ ] Maintain minimal/default/composed CPU-free feature isolation, one lock/version authority and truthful unsupported capability behavior.
````

<a id="platform-and-release-l39"></a>

#### open · [L39–39](../../plans/platform-and-release/TODO.md#L39)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L39-L39@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `c46f215fa7e5119b476bbc66d88fb7f5b20ec1dd0241892e9d4ff35f4aeff447`.

````markdown
- [ ] Gate bounded compiler/autotune/AOT caches by source/device/driver/shape and retain negative candidates; no unbounded first-request search.
````

<a id="platform-and-release-l40"></a>

#### open · [L40–40](../../plans/platform-and-release/TODO.md#L40)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L40-L40@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `597cd7f71517af2f3f8f8dc28748295fd778a9707287e84a71376a5785704f50`.

````markdown
- [ ] Complete reproducible/signing/release policy and fair model/task/quality/performance matrices with cold/warm, p50/p95/p99, memory/energy and uncertainty. Push/tag/release remain separately authorized actions.
````

<a id="platform-and-release-l41"></a>

#### open · [L41–41](../../plans/platform-and-release/TODO.md#L41)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L41-L41@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `93cf14ff370b9b8b9e082a5d0ddb4f84b2bfdb4f439bdb80c8171f798b2266a7`.

````markdown
- [ ] Demonstrate compatibility/invalidation of operator/task/state/plan caches and bounded change radius before broader platform claims.
````

<a id="platform-and-release-l43"></a>

#### open · [L43–43](../../plans/platform-and-release/TODO.md#L43)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L43-L43@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `6864dfa6cd822b525793bec852613ef5f8fa748800eb25b94c268c0c75e69266`.

````markdown
- [ ] Before a true zero-copy strided View is enabled, validate contiguity against stored value provenance and test public PlanValue retargeting. Current View physically materializes dense output; retain cached-element and dynamic range guards.
````

<a id="platform-and-release-l44"></a>

#### open · [L44–44](../../plans/platform-and-release/TODO.md#L44)

Heading: `Platform and release > Uncommitted`.
Identity: `.agents/plans/platform-and-release/TODO.md:L44-L44@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `cbeecc5e904c9ede8fc9c27c3719f7439b447777fb6deca131c873ee01061262`.

````markdown
- [ ] Preserve one GPU-library service/queue/quarantine owner and cold generic-plan admission through packaging changes; verify actual final downstream links, not only compiler/type checks.
````

<a id="platform-and-release-l99"></a>

#### open · [L99–99](../../plans/platform-and-release/TODO.md#L99)

Heading: `Platform and release > Tests and verification`.
Identity: `.agents/plans/platform-and-release/TODO.md:L99-L99@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `77160b54cd7d6a7d19c875c231559c51838e632689262447b123217f9d7f03d7`.

````markdown
- [ ] Qualify all six [CPU release definitions](../../../docs/DEVELOPMENT.md#cpu-cross-release-definitions) with actual target execution, installer checksum/version/refusal cases and platform signing/authenticity policy. Build definitions and same-host checks do not establish other-target support.
````

<a id="platform-and-release-l100"></a>

#### open · [L100–100](../../plans/platform-and-release/TODO.md#L100)

Heading: `Platform and release > Tests and verification`.
Identity: `.agents/plans/platform-and-release/TODO.md:L100-L100@ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1`.
Block SHA256: `863d69625c7f6b369335c0f469bb19d689bf7039bee93d5a33a5b84c1002f88f`.

````markdown
- [ ] Test enabled NixOS service lifecycle and runtime-secret handling after serving is implemented; the currently disabled service definition and module evaluation are not a running-server claim.
````

### serving-and-speculation — 30 noncompleted occurrences

<a id="serving-and-speculation-l13"></a>

#### open · [L13–14](../../plans/serving-and-speculation/TODO.md#L13-L14)

Heading: `Serving and speculation > Uncommitted > Shutdown qualification — 26 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L13-L14@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `f6fa1d6e2ab2f75025e5f7b581f26cf82e6b356724e0e9de1770819042a449e5`.

````markdown
- [ ] Replay applicable original-model/backend shutdown and fault scopes under
  their existing source/device gates before a benchmark qualification claim.
````

<a id="serving-and-speculation-l209"></a>

#### open · [L209–244](../../plans/serving-and-speculation/TODO.md#L209-L244)

Heading: `Serving and speculation > Uncommitted > Current parity and post-parity optimization order — 23 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L209-L244@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `ab63638aad4f3b0e19f41e38c6c974230bb5b0302e87abc8b1032591497299e4`.

````markdown
- [ ] Profile and fix the .8B gap first, then beat the same matched vLLM C1/C4
  baseline without changing source/quality/caps. Broader2B/C8/ragged/repeated
  timing deferred by explicit owner priority; no unnecessary baseline sweep.
  Resident-head candidate passes all40 actual numerical calls/unchanged limits
  and all five fault scopes after narrow verifier recognizer repair. Quiet
  host phase diagnostic8requests/64forwards exact: binding3.629112s/enqueue
  3.021396s, nestedHead0.005158s/zeroH2D/zero checkpoints; original owned closure
  clean. Follow-on host unit passes fresh40+all5faults/14metadata guards/API24
  per-provider exact; strict GPU Clippy PASS. Fresh unprofiled canonical wave8:
  C1 native9.7748 vsstock38.0061 tok/s (6.54748s vs1.68394s); C4 native9.7972
  vsstock49.1130 (6.53247s vs1.30312s). All32 offers/256tokens exact, actual
  peaks1/4, batchmax1, all original owners/source/EOF close. Preliminary one
  pass only; still3.89x/5.01x slower, no certified win. Combined CPU+phase
  diagnostic finds memcmp70.66% of532 user-cycle samples/0lost; actual process
  6.72user+0.66system over6.77396wall. Nested bind3.074109s/enqueue2.945825s,
  complete/readback0.149683s; Head64launches/zeroH2D/checkpoints. CPU fractions
  are not GPU or wall fractions. Investigate deep GDN static-source identity
  comparisons at their shared metadata owner; do not weaken equality/authority.
  Exact static-source equality now preserves all field/content semantics while
  bypassing same-pointer+length text scans. Metadata10/strictGPU Clippy/joint
  release PASS; frozen pair4 passes fresh actual matrix40/all5faults and matched
  API24/24 per provider, exact output/state/owner closure. Fresh unprofiled pair4:
  C1 ZG40.4089 vsstock37.0951 tok/s (1.58381s vs1.72530s), native+8.93%;
  C4 ZG39.2740 vsstock48.2272 (1.62958s vs1.32705s), native18.57% lower.
  All32/32 offers/256tokens exact, original owners/source/EOF closed. One
  preliminary pass, no confidence/tail/fullparity claim. Actual HF batchmax1;
  source-safe B4 integration is under separate investigation, not yet a causal
  percentage attribution. Subsequent pair5 residentembedding reuse passes
  fresh40calls/allfivefaults/matchedAPI24offers192tokens perprovider. Quiet
  C1 ZG48.9151 vsstock38.8841tok/s (+25.80%); C4 ZG49.9998 vsstock50.4721
  (-0.94%), all32offers/256tokens exact/allfour owners/sourceclosed. Native
 64roots=forwards=outputs/432steps perarm/peak1|4/batchmax1. Singlepass only,
  no C4win/confidence claim. Periodic146path MAIN↔ROOT convergence preserves
  parents/freshmtimes; assembled gate5 PASS1645Rust/0failed/86ignored/265suites
  plus complete scripted checks. Later common preparation/cache work requires
  its own assembled gate and fresh physical/API checks. Models parallel.
````

<a id="serving-and-speculation-l245"></a>

#### open · [L245–255](../../plans/serving-and-speculation/TODO.md#L245-L255)

Heading: `Serving and speculation > Uncommitted > Current parity and post-parity optimization order — 23 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L245-L255@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `bae24e7523ef00a709c6a5de67286b4c221d447f23f3fe63ee4459042172e99f`.

````markdown
- [ ] Remaining .8B C4 optimization: source-safe HF physical coalescing/batching.
  Current CLI rejects native_rows, HFloaded batchNone/native_batchfalse and
  worker excludes originalHF; existing B2/B4/B8 BatchProgram requires GGUF F32.
  Reuse root-cohort validation/publication, not incompatible math admission.
  Unchanged N1 completioncohort needs genuine private lanes or registered
  cohort epochs/status taps; shared RMS/value/readback reuse must await actual
  completion. True BF16 B4 needs its own typed arithmetic/source/state lowering.
  Existing genuine47/57/62/67 B1 prefills then four distinct cached frontiers
  provide strong mixed-row/current-root oracles without another capture; test
  row permutations/all24boundaries/all42state/fullV/F32/feedback/fault closure.
  This is source planning; no B4 implementation or causal gap percentage claim.
````

<a id="serving-and-speculation-l256"></a>

#### open · [L256–260](../../plans/serving-and-speculation/TODO.md#L256-L260)

Heading: `Serving and speculation > Uncommitted > Current parity and post-parity optimization order — 23 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L256-L260@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `d2713825210f05507d6762ac5c3fddc1b2a142f2e834c27be9a12a950a88625e`.

````markdown
- [ ] After source/state/natural-serving parity, execute established short
  C1/C2/C4/C8 comparisons with unchanged source/profile/EOS/caps and original
  lifecycle controls. Then apply measured substantial generic improvements and
  rerun those profiles before long contexts and dense27B. Research alone adds
  no throughput/TTFT/goodput result.
````

<a id="serving-and-speculation-l261"></a>

#### open · [L261–270](../../plans/serving-and-speculation/TODO.md#L261-L270)

Heading: `Serving and speculation > Uncommitted > Current parity and post-parity optimization order — 23 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L261-L270@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `1c6154ae8b4ed4dffb9bf49b77a6b366c819fe6093b968e3308f5e8b335961f7`.

````markdown
- [ ] After the already authorized matched benchmarks complete, implement
  graceful cancellation plus explicit forced/immediate cancellation with bounded
  escalation deadlines. Add hooks for detecting stalled progress, repeated model
  loops and runaway CPU work; allow the supervisor to break them cleanly. Define
  request/session/model scope, cooperative drain versus forced termination,
  timeout policy and observability. Release GPU/native/source resources only
  after confirmed completion/termination; keep unknown work fenced/quarantined.
  Cover disconnect, stuck execution, timeout/escalation and healthy-survivor
  recovery with real physical tests. Owner directive22Oct: defer this extension
  until after benchmarks; it must not delay the active parity/benchmark path.
````

<a id="serving-and-speculation-l271"></a>

#### open · [L271–275](../../plans/serving-and-speculation/TODO.md#L271-L275)

Heading: `Serving and speculation > Uncommitted > Current parity and post-parity optimization order — 23 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L271-L275@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `ac863ccd80a7fe59c5f799e16802983ae5f7d85300dbd0a59a3e51f10e87f22a`.

````markdown
- [ ] Compose stable leased rows/diff metadata, compact readout, qualified
  production graph regions, exact fusion and true temporal prefill inside the
  existing worker. Price extra graph pools/padding, output backpressure and
  prefill/decode fairness; unknown capture/launch completion cannot eager-fallback
  through a fenced service.
````

<a id="serving-and-speculation-l276"></a>

#### open · [L276–280](../../plans/serving-and-speculation/TODO.md#L276-L280)

Heading: `Serving and speculation > Uncommitted > Current parity and post-parity optimization order — 23 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L276-L280@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `150982ba69e9afd036f24f9775a5b31c1aa04615d3444706c604ace8a4b641b3`.

````markdown
- [ ] Extend full-state branch verification before any proposer: prefix0..K,
  target correction, all recurrent/conv/KV/indexer/draft/RNG/grammar domains,
  EOS/stop and canceled/stale epochs. Compare n-gram/Recall/suffix, compatible
  trained MTP/block drafters and K=0 using useful complete-cycle work, not assumed
  acceptance or source kernel counts. Keep explicit sampler/distribution profiles.
````

<a id="serving-and-speculation-l281"></a>

#### open · [L281–288](../../plans/serving-and-speculation/TODO.md#L281-L288)

Heading: `Serving and speculation > Uncommitted > Current parity and post-parity optimization order — 23 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L281-L288@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `64c9e58cc31213e2a82c0f9325ed0cd506e6dffbc68e36696972da9ed430802d`.

````markdown
- [ ] Retain profile/shape/layout/source/context-grant compatibility at every
  batch/prefix/graph/proposal boundary; dynamic policy chooses only already
  qualified compatible variants. The user-facing profile combines pipelines,
  while original roots, physical admission and actual completion authorize work.

Research/priorities: [mechanism comparison](../../research/engine-optimization-comparison-2026-10-21/README.md),
[optimization tasks](../cpu-gpu-optimization/TODO.md),
[recursive composition](../pipeline-composition/TODO.md).
````

<a id="serving-and-speculation-l298"></a>

#### open · [L298–300](../../plans/serving-and-speculation/TODO.md#L298-L300)

Heading: `Serving and speculation > Uncommitted > Batching and paged-state follow-up — 21 Oct design only`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L298-L300@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `a368646cdb1cb10e907d7bc78a930c065a463096e246ec3f0f06f161639bd143`.

````markdown
- [ ] Integrate both accepted GGUF/reference and newly qualified vLLM numerical
  profiles through the same session/worker/HTTP path. Bind profile/phase/schema
  to compiled plans and cache compatibility; reject incompatible state reuse.
````

<a id="serving-and-speculation-l301"></a>

#### open · [L301–305](../../plans/serving-and-speculation/TODO.md#L301-L305)

Heading: `Serving and speculation > Uncommitted > Batching and paged-state follow-up — 21 Oct design only`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L301-L305@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `2da5e61670bda0ecce61b0dd7e1d09f1f115e0f3f38da94bd968c8169b6b7f8b`.

````markdown
- [ ] Implement original-plan paged KV append and block-table QK/softmax/AV,
  preserving each profile's masks, positions, dtype/reductions and ownership.
  Qualify independent B1/B2/B4/B8, page boundaries and full logits/state;
  allocator/hybrid prefix/COW dependencies live in the
  [memory plan](../memory-and-streaming/TODO.md).
````

<a id="serving-and-speculation-l306"></a>

#### open · [L306–309](../../plans/serving-and-speculation/TODO.md#L306-L309)

Heading: `Serving and speculation > Uncommitted > Batching and paged-state follow-up — 21 Oct design only`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L306-L309@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `2d76ad7455e6b1ce482f1cca397c954ee400b73a77a370fff91178ff4a7c32b5`.

````markdown
- [ ] Extend bounded refill/regroup into fair aged chunked prefill/decode
  scheduling; qualify actual multi-token temporal prefill separately from
  request batching. Admit any physical GPU overlap only after event/lease,
  cancellation/unknown-completion and independent-root tests.
````

<a id="serving-and-speculation-l310"></a>

#### open · [L310–315](../../plans/serving-and-speculation/TODO.md#L310-L315)

Heading: `Serving and speculation > Uncommitted > Batching and paged-state follow-up — 21 Oct design only`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L310-L315@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `3a427042a52d634a69f2d3a99a70c05ec92f2e9bdf50938de6887318d5015f9f`.

````markdown
- [ ] Run profile-matched batching/paging/prefix-hit/overlap ablations and genuine
  pinned-vLLM comparisons: established short C1/2/4/8 first, then qualified long
  contexts. Record quality/full-state acceptance, true ITL/TTFT tails, committed
  SLO-goodput, offered/rejected/canceled work, fairness, actual dispatch/copy
  counts and physical memory/fragmentation. Preserve old-profile regressions;
  code presence or fewer launches cannot close serving/performance parity.
````

<a id="serving-and-speculation-l337"></a>

#### open · [L337–350](../../plans/serving-and-speculation/TODO.md#L337-L350)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L337-L350@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `10d6159192e0db0caf45d84ede3c8765c545638a483c83ad5ba246a92765ad05`.

````markdown
- [ ] **P12 ordered SiLU integration:** private CUDA B1/B2/B4/B8 operator
  passes204 physical old-GPU/CPU differential cases,171562710 exact bit
  comparisons and five fresh failure/abandonment processes. All17 intermediate
  checkpoints, status, canaries and original lease/quarantine ownership pass.
  Independent source review and candidate CUDA/Vulkan/default strict checks
  plus two row-partition tests pass. Candidate library, qualifier, server,
  worker and HTTP release artifacts compile; the exact-source diagnostic
  file-transport unit passes. Paused at the owner's request before starting
  the prepared60-case physical matrix. Full-model integration remains unaccepted:
  complete PTX/loaded-code witnesses, model/StateRoot/serving regressions, both
  original release parity routes, isolated full gate and fresh timing/count
  comparisons remain pending. Existing truncated PTX dump filenames overwrite
  specializations; retain that failed evidence and qualify a collision-safe
  diagnostic before claiming complete instruction coverage.
````

<a id="serving-and-speculation-l352"></a>

#### open · [L352–381](../../plans/serving-and-speculation/TODO.md#L352-L381)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L352-L381@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `5f73dc0476988ee6a00f7aa35615f40d926980c2a2e0b6a8829d7d95e7429b60`.

````markdown
- [ ] **Stock2B loader and matched benchmark:** genuine pinned stock2B
  V4 plain/observed loader and independent comparison now pass: all335
  stored tensors inventoried, exact15 MTP exclusions and320 backbone roles;
  both complete vocabularies,496640 native BF16 logit words, remain exact.
  Source/runtime and owned process/group/container/EOF cleanup pass; retain
  the original V3 observer refusal. Both model loaders are prerequisites,
  not cross-engine/full42/default-graph or throughput qualification.
  Actual0.8B eager HTTP policy passes for both providers: identical47 prompt
  IDs, original Rust tokenizer, effective greedy/EOS settings, observed
  cgroup12GiB/swap0/CPU4 caps and authenticated GPU process membership.
  Original owned cleanup closes. Podman unresolved HPIDs establish no
  ownership; every sampled GPU PID must match the numeric owned subset.
  Generated IDs differ under the declared activation/cast profiles. Actual
  0.8B full-output quality covers seven prompts; quality equivalence remains
  unqualified. First unprofiled eager/cold-process/solo comparison completes
  all eight ABBA/BAAB samples: four requests/32 raw tokens per provider.
  Median TTFT13.400s native versus1.561s stock; stock is faster here.
  Cold-process means first request after readiness, not storage-cache cold
  or complete startup latency. Both providers are internally stable, but
  cross-engine IDs/text/usage differ. One stock sample coalesces delivery;
  individual-token ITL and source-correct SLO-goodput remain unqualified.
  The2B eager/cold-process/solo run also completes all eight samples:
  four requests/32 tokens per provider, identical raw IDs/output hashes/
  usage/finish on that one prompt. Median TTFT14.540s native versus1.562s
  stock; all samples have individual token delivery, native ITL p95
  0.186–0.199s versus stock0.026–0.030s. Seven-prompt2B quality remains
  unqualified despite five matching full texts; this prompt establishes no
  numerical equivalence. Finish default-route, warm/concurrent workloads
  and physical counts;
  this bounded observation does not establish numerical or serving parity.
````

<a id="serving-and-speculation-l383"></a>

#### open · [L383–390](../../plans/serving-and-speculation/TODO.md#L383-L390)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L383-L390@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `4bd234257c0da0ae6eb5af17c8849ad7ed3aa0f987d950844a789f3454dc27c4`.

````markdown
- [ ] **Production CUDA graph admission:** close every same-card constructor,
  transport, allocation, external event/storage and final-drop path through
  original queue/completion ownership; admit graph metadata/driver resources.
  Private graph replay does not enable production graph selection. Test-only
  host protocol14 cases/three ownership negatives pass. Separate original
  transport14 cases/two intended ownership failures, strict Clippy and no_std
  also pass in an isolated harness. Prepaid retirement and complete integration
  remain separate work, with participation unproven.
````

<a id="serving-and-speculation-l392"></a>

#### open · [L392–395](../../plans/serving-and-speculation/TODO.md#L392-L395)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L392-L395@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `df821b07299e66d915ce6ed30b45dd407529d4fea5e602ad08939ed9e9da11e4`.

````markdown
- [ ] **P12 prefix scheduling performance:** measure concurrent native prefix
  cold-process throughput/TTFT against the accepted max32-prefix implementation.
  Warm observations below cover both models plus explicit B1 scheduling cost;
  genuine equivalent-source vLLM comparison remains open.
````

<a id="serving-and-speculation-l397"></a>

#### open · [L397–419](../../plans/serving-and-speculation/TODO.md#L397-L419)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L397-L419@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `655f0bb0f6fe379fb5e66d311fe6d8c15a97860b4453b89090655f6efd294177`.

````markdown
- [ ] **P12 fused continuous GPU serving:** extend the accepted CUDA B2/B4/B8
  HTTP path with paged hybrid KV/recurrent cache, physically safe
  prefill/decode overlap and fairness. Unequal decode frontiers and
  compatible refill are accepted within the B7 source/context scope.
  Extend full-logit/StateRoot qualification to other advertised trained
  profiles/backends and full GPU prefill. Measure actual p50/p95 TTFT,
  true token ITL, committed SLO-goodput, fairness, memory and physical
  dispatch counts against genuine pinned vLLM V1 with equivalent
  source/quant/tokenizer/hardware, cold/warm conditions and caps.
  Local release smoke and fewer launches do not qualify performance.
  Separate native0.8B B2 CUPTI diagnostic records64328 actual kernels and
  64328 successful direct launch calls, zero dropped records and exact source
  counter delta; plain/observed complete outputs and owned retirement match.
  A fresh accepted-baseline0.8B canonical C1 diagnostic also validates all
  four native/stock plain/observed processes:236270 actual native kernels
  and direct launches equal the source delta, zero dropped records/errors;
  exact provider-internal outputs and original owned retirement pass.
  Stock public profiler records9318 kernel activities/8810 host launches,
  but its losslessness/all-process coverage remains unverified. Independent
  validators and ROOT/peer metadata reviews pass. These observations qualify
  the native windows only; complete matched hardware counts, profiler device
  allocation admission, source-correct goodput and profiled timing remain
  unqualified. Equivalent default-route/warm/concurrent vLLM timing stays open.
````

<a id="serving-and-speculation-l421"></a>

#### open · [L421–431](../../plans/serving-and-speculation/TODO.md#L421-L431)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L421-L431@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `c7a7c902da529b12f9e5e5b8c2c692956ba600e144ec8ba96c9c50126d9b7251`.

````markdown
- [ ] Connect accepted core `ContextOwner`/`ContextReader` snapshot
  capability to the real router's authenticated session/turn frontier:
  source-scoped application/shared grants, per-application and per-tenant
  ownership, immutable complete context records, explicit ON/OFF context
  selection, short-lock optimistic source epochs and cancel/revoke
  invalidation. Recheck `require_fresh` at TaskCoordinator delivery and
  bind source/graph generations to model/media StateRoot publication.
  DecisionAdapter is a typed callback boundary, not trained NanoJev
  logits or actual ASR context support. Cross-session/host source sync,
  bounded CPU staging and QoS, media quality A/B and revocation/stream
  cancellation remain open.
````

<a id="serving-and-speculation-l433"></a>

#### open · [L433–444](../../plans/serving-and-speculation/TODO.md#L433-L444)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L433-L444@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `76b89960e6da82a0af573e833f90f6c9dcd722d9b9b36cf1f1996370d1b1ac92`.

````markdown
- [ ] One router must snapshot each enabled, source-granted context
  stream at the declared turn/graph epoch before live ASR. Default only
  committed user/assistant session content; explicitly selected tool
  call/result, application-owned system/developer prompts and authorized
  shared/cross-user records are eligible. Strict source ownership and
  requested grants, not a blanket category blacklist, govern access.
  No same-turn LLM feedback loop or unauthorized sources/tenants, raw
  unowned KV pointers or tool-secret leaks. The ON/OFF toggle leaves
  typed graph topology stable; OFF performs no retrieval. Bound
  configurable 4KiB..1MiB plus physical admission, ordered filters,
  typed decision models, credit/backpressure, stale ACKs and source-real
  accuracy/consent tests before live mode.
````

<a id="serving-and-speculation-l446"></a>

#### open · [L446–454](../../plans/serving-and-speculation/TODO.md#L446-L454)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L446-L454@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `b2af4c127e5ffdfb19db2f59851875df60864953384dd5240ace624086503845`.

````markdown
- [ ] Make YAML/repeatable-flag compiled pipeline configs a first-class
  source-bound router input: many model IDs, many typed component DAGs,
  many explicit protocol/route bindings within ONE process; one model
  can serve standalone chat and ASR→diarization→LLM→TTS. Preserve
  direct one-model CLI execution with no config requirement. Current
  pipeline check is pure scaffolding; implement model residency,
  admission, TaskCoordinator execution, generation/hot reload and
  authenticated OpenAI/SSE/WebSocket/WebRTC transport, then enable
  pipeline run/serve only after source/physical tests.
````

<a id="serving-and-speculation-l455"></a>

#### open · [L455–459](../../plans/serving-and-speculation/TODO.md#L455-L459)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L455-L459@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `4bab40e18e5b92393068b8e260f6e54c17ad8bf53f52c9ef05f5a28aa505bea2`.

````markdown
- [ ] Distinguish model-trained MTP weights and Qwen Flash-Next
  learned n-gram embedding bank from runtime n-gram/suffix
  speculative decoding (no extra source required). Verify
  accept/reject and complete rollback/grammar RNG across
  QSA/GDN/MoE before any speed claim.
````

<a id="serving-and-speculation-l461"></a>

#### open · [L461–461](../../plans/serving-and-speculation/TODO.md#L461)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L461-L461@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `58c4784cffb0b49a6bbf7f671aa0289a1df66143fd315451a881afa800de129d`.

````markdown
- [ ] Implement canonical TaskRequest/coordinator workers and CLI/SDK/API integration, model routing/residency lifecycle, explicit supported OpenAI/Anthropic interfaces, streaming/tools/reasoning/schema/sampling and truthful capabilities.
````

<a id="serving-and-speculation-l462"></a>

#### open · [L462–462](../../plans/serving-and-speculation/TODO.md#L462)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L462-L462@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `cd0ad88c9ada7a9fcb41fa743132f617d2944a8ef3d88c6dd9286d30aec7c38f`.

````markdown
- [ ] Enforce separate API/tenant payload, queued-prefill/residency and physical active-step admission; bounded overload, cancellation, disconnect, stale epochs and output backpressure.
````

<a id="serving-and-speculation-l463"></a>

#### open · [L463–463](../../plans/serving-and-speculation/TODO.md#L463)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L463-L463@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `3dc2d4b9474f15e36fa34e195e7dc8a3ad5247db72708850142df2557f39725a`.

````markdown
- [ ] Add real compatible continuous/grouped batches, aged chunked prefill, decode/media fairness and finite joint batch/draft/expert/kernel/placement plans; no per-token catalog/JSON/global-lock work.
````

<a id="serving-and-speculation-l464"></a>

#### open · [L464–464](../../plans/serving-and-speculation/TODO.md#L464)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L464-L464@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `c3ec9702a9bf5a40153c54df594bb4ccd9327c38357b6969907efbb41d76b874`.

````markdown
- [ ] Finish Qwen basics before optimizations; enumerate the complete pinned llama.cpp/vLLM union, public and internal surfaces. Preserve 105 provisional groups/145 document seeds and CLI/API frontiers as unreviewed inputs, not exhaustive parity.
````

<a id="serving-and-speculation-l465"></a>

#### open · [L465–465](../../plans/serving-and-speculation/TODO.md#L465)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L465-L465@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `2879c397cad2e76d733f2b0b8bd16ffbb8f337b01823e4486397a5cbd439e55a`.

````markdown
- [ ] Implement every applicable sampler/logits-processor/RNG/logprob/grammar/tool/stop/reasoning composition and its independently verified outputs.
````

<a id="serving-and-speculation-l466"></a>

#### open · [L466–466](../../plans/serving-and-speculation/TODO.md#L466)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L466-L466@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `46a846656ee3d6da14d63869b5193cd246ee76c1df85aa0134a06136c8a14b30`.

````markdown
- [ ] Qualify target-authoritative MTP, draft model, n-gram/suffix, EAGLE, DFlash/DFlash2, DSpark and related registered strategies; DSpark2 exact primary source/version remains unresolved.
````

<a id="serving-and-speculation-l467"></a>

#### open · [L467–467](../../plans/serving-and-speculation/TODO.md#L467)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L467-L467@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `4e972d0e946d7564cdb4be60adfc3ba6017b4accdf5376f20a107fef6807c0c4`.

````markdown
- [ ] Test all accepted-prefix lengths, target-distribution correction, target/draft RNG/grammar ownership, full KV/recurrent/index/media rollback, physical branch admission and authentic cancellation/quarantine.
````

<a id="serving-and-speculation-l468"></a>

#### open · [L468–468](../../plans/serving-and-speculation/TODO.md#L468)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L468-L468@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `9e220b7f5057a820fee8ee9347d53ef508e2c6c2092c995822ba88ed1b3c8749`.

````markdown
- [ ] Jointly benchmark material combinations, adaptive depth/backoff and all offered/rejected/canceled work: committed SLO-goodput, TTFT/ITL tails, fairness, memory and energy versus equivalent pinned vLLM/V1/Omni.
````

<a id="serving-and-speculation-l469"></a>

#### open · [L469–469](../../plans/serving-and-speculation/TODO.md#L469)

Heading: `Serving and speculation > Uncommitted > Resumed correctness-first continuation — 21 Oct`.
Identity: `.agents/plans/serving-and-speculation/TODO.md:L469-L469@82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0`.
Block SHA256: `66ed316b2921c45d310097b69221ffa14da1309f8fb3a6cb73dfaa68239297a5`.

````markdown
- [ ] Preserve zero untriaged frontier/disposition requirements before any full-parity claim; synthetic acceptance, fewer launches and competitor measurements do not establish local parity or speed.
````

## Direct linked-document and evidence registry

Every Markdown link outside fenced code in all17 TODOs is registered here,
deduplicated by resolved local path (fragments stay in source references) or exact
external URI. This includes design/specification/acceptance owners, current
readmes, run designs and evidence, source runbooks, cross-plan/archive links,
historical handoffs and missing targets. It is a direct TODO-link registry;
transitive references remain in the named documents and broader source catalog.
Existing directories are distinguished from absent files. No symlink target was
created, file downloaded or dead link silently redirected. Hashes identify local
file bytes, not claim freshness, correctness or executed qualification. External
URIs are preserved references and were not fetched for this documentation task.

| Target / role | State / SHA256 | Referring TODO spans |
|---|---|---|
| `/home/pvl/spaces/abird/src/zerogravity-worktrees/gemma4-portable-gpu-whole35-20261023/.agents/runs/model-execution-native-sol-high-20261023/GEMMA_APPLY-v5-2.md` · run evidence | **MISSING at cutoff** | [model-execution/TODO.md:227](../../plans/model-execution/TODO.md#L227) |
| `/home/pvl/spaces/abird/src/zerogravity-worktrees/model-source410-checks-sol-high-20261024/.agents/runs/model-execution-x2-readiness-sol-high-20261024/READINESS.md` · run evidence | **MISSING at cutoff** | [model-execution/TODO.md:555](../../plans/model-execution/TODO.md#L555) |
| `/home/pvl/spaces/abird/src/zerogravity-worktrees/model-source410-checks-sol-high-20261024/.agents/runs/model-execution-x2-readiness-sol-high-20261024/execute-source466-v1/PHYSICAL-SETUP-RESULT.md` · run evidence | **MISSING at cutoff** | [model-execution/TODO.md:597](../../plans/model-execution/TODO.md#L597) |
| `/home/pvl/spaces/abird/src/zerogravity-worktrees/model-source410-checks-sol-high-20261024/.agents/runs/model-execution-x2-readiness-sol-high-20261024/execute-source466-v1/QWEN-PHYSICAL-RESULT.md` · run evidence | **MISSING at cutoff** | [model-execution/TODO.md:624](../../plans/model-execution/TODO.md#L624) |
| `/home/pvl/spaces/abird/src/zerogravity-worktrees/qwen-final-owner-alias-sol-high-20261024/.agents/runs/model-execution-qwen-final-owner-alias-sol-high-20261024/REPORT.md` · run evidence | **MISSING at cutoff** | [model-execution/TODO.md:638](../../plans/model-execution/TODO.md#L638) |
| [/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/.agents/runs/model-execution-gemma4-root-integration-20261022/TODO.md](../../../../zerogravity-worktrees/root-lowest-f32-20261020/.agents/runs/model-execution-gemma4-root-integration-20261022/TODO.md) · run evidence | file `99e37027975a2f0d11d8c5cd93863dc3f0f8edbd7a0233d44420a7350a47f715` | [model-execution/TODO.md:939](../../plans/model-execution/TODO.md#L939) |
| [/home/pvl/spaces/abird/src/zerogravity-worktrees/root-lowest-f32-20261020/.agents/runs/model-execution-qwen27b-root-integration-20261022/TODO.md](../../../../zerogravity-worktrees/root-lowest-f32-20261020/.agents/runs/model-execution-qwen27b-root-integration-20261022/TODO.md) · run evidence | file `990b9d585c1cb8017de39b62e00d58a617345da9a8a5cf068319ea20c43d3f8a` | [model-execution/TODO.md:935](../../plans/model-execution/TODO.md#L935) |
| [.agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md](../ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) · design/specification/runbook | file `33ec1a0b63efff4fa71fc0c8e56766ed707680ac8ae9fc470328fa133719ba08` | [distributed-execution/TODO.md:7](../../plans/distributed-execution/TODO.md#L7) |
| [.agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md](../DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md) · design/specification/runbook | file `bf2100d929ed6f2fd6b18981b13470f49dc3d6c17fd9162dfb125ceba69ca55e` | [cpu-gpu-optimization/TODO.md:8](../../plans/cpu-gpu-optimization/TODO.md#L8) |
| [.agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md](../DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) · design/specification/runbook | file `2c47bae7d0dae9391c87e34335e9cc76b5b4b7b17815ad9ff31d99d5e72fa5c1` | [distributed-execution/TODO.md:5](../../plans/distributed-execution/TODO.md#L5) |
| [.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md](../FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md) · historical handoff | file `e4fed9cebd90a2f4d193fa73f8637c5254944e1ae2e1c295557961ea574c2ebb` | [model-acquisition/TODO.md:131](../../plans/model-acquisition/TODO.md#L131) |
| [.agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md](../JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) · design/specification/runbook | file `d62f507a2a3bac341b753d0ff512c7f7a2f81c15cf95779903bebd4359272dde` | [serving-and-speculation/TODO.md:5](../../plans/serving-and-speculation/TODO.md#L5) |
| [.agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md](../MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) · design/specification/runbook | file `72a1f1941239b77e2dc3c0a0eb08d505d024e819a37ec49c777aeb94f25aa845` | [platform-and-release/TODO.md:6](../../plans/platform-and-release/TODO.md#L6) |
| [.agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md](../MODEL_SEMANTICS_AND_COMPATIBILITY.md) · design/specification/runbook | file `ffabf256a01a3c29b455593ba5e0162e1cb6a417bedce7fb7b13e00b15594341` | [model-execution/TODO.md:1254](../../plans/model-execution/TODO.md#L1254) |
| [.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md](../NUMERICS_QUALITY_AND_BENCHMARKS.md) · design/specification/runbook | file `d419ec8016f967c123e174292b1b0595011bfc8eb2a6588aa5fb27bacb3d017a` | [numerics-and-context/TODO.md:5](../../plans/numerics-and-context/TODO.md#L5) |
| [.agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md](../REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md) · design/specification/runbook | file `0a5412cf92701b7c6a4a8ab22fd75d8746a31b44ee882890c7e7509856ab6352` | [multimodal/TODO.md:7](../../plans/multimodal/TODO.md#L7) |
| [.agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md](../REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md) · historical handoff | file `bd35928aa22cfd61e7e089c16a7f3899bfacb01dac9d73a5556edbfcd4f78715` | [distributed-execution/TODO.md:6](../../plans/distributed-execution/TODO.md#L6) |
| [.agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md](../TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) · design/specification/runbook | file `61411e66c812dc40a9ed37617cb9deff54ee209b9f548aed91409b3275a45142` | [platform-and-release/TODO.md:7](../../plans/platform-and-release/TODO.md#L7) |
| [.agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md](../UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md) · design/specification/runbook | file `55fec7a0edcd1b48a2e47eee17c8f7f63f63c6c6f5c5c98a0e34cb13001f7d01` | [multimodal/TODO.md:5](../../plans/multimodal/TODO.md#L5) |
| [.agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md](../UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) · design/specification/runbook | file `d39bfcaaf68165aeb9c3ac4d053a7332d840cac925dce2562df6b0236a2a284f` | [memory-and-streaming/TODO.md:5](../../plans/memory-and-streaming/TODO.md#L5) |
| [.agents/docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md](../VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md) · design/specification/runbook | file `01057dffdccd1d9a92bf3c4451cb80784ed7b84b5b90f0c70c9ad0bc754dea1e` | [multimodal/TODO.md:6](../../plans/multimodal/TODO.md#L6) |
| [.agents/plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md](../../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md) · design/specification/runbook | file `dda399a4b79a677d6b2ae1e15b790257720dcdcd345a786c01d613f8821a40b1` | [memory-and-streaming/TODO.md:7](../../plans/memory-and-streaming/TODO.md#L7); [model-execution/TODO.md:5](../../plans/model-execution/TODO.md#L5) |
| [.agents/plans/IMPLEMENTATION_ROADMAP.md](../../plans/IMPLEMENTATION_ROADMAP.md) · design/specification/runbook | file `a976ec6153dd882d37c5478944c7aa10bb63d28e55d9c236ce49b5f8ff6381fa` | [platform-and-release/TODO.md:5](../../plans/platform-and-release/TODO.md#L5) |
| [.agents/plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md](../../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md) · design/specification/runbook | file `73d4eb29a98eebd39166b179b8cb647eec961b849a79b7a749220a5dc3c96829` | [model-acquisition/TODO.md:7](../../plans/model-acquisition/TODO.md#L7) |
| [.agents/plans/NEXT_AGENT_PROMPT_2026-10-17.md](../../plans/NEXT_AGENT_PROMPT_2026-10-17.md) · historical handoff | file `15734cf3bef345bed0b121768d695d0ffbad0974c758b781e8db40bec20520b8` | [model-acquisition/TODO.md:131](../../plans/model-acquisition/TODO.md#L131) |
| [.agents/plans/RELEASE_ACCEPTANCE_GATES.md](../../plans/RELEASE_ACCEPTANCE_GATES.md) · design/specification/runbook | file `84377b654b99fe828442debb5810f13feac0600e961c34b04358fe3fa9a9b64c` | [platform-and-release/TODO.md:8](../../plans/platform-and-release/TODO.md#L8) |
| `.agents/plans/archive/history/EXPERIMENT_PROGRAM.md` · historical archive | **MISSING at cutoff** | [archive/history/historical-records/TODO.md:58](../../plans/archive/history/historical-records/TODO.md#L58) |
| `.agents/plans/archive/history/IMPLEMENTATION_ROADMAP.md` · historical archive | **MISSING at cutoff** | [archive/history/historical-records/TODO.md:56](../../plans/archive/history/historical-records/TODO.md#L56) |
| `.agents/plans/archive/history/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md` · historical archive | **MISSING at cutoff** | [archive/history/historical-records/TODO.md:57](../../plans/archive/history/historical-records/TODO.md#L57) |
| `.agents/plans/archive/history/RELEASE_ACCEPTANCE_GATES.md` · historical archive | **MISSING at cutoff** | [archive/history/historical-records/TODO.md:59](../../plans/archive/history/historical-records/TODO.md#L59) |
| `.agents/plans/archive/history/TODO.md` · historical archive | **MISSING at cutoff** | [archive/history/accepted-baseline/TODO.md:8](../../plans/archive/history/accepted-baseline/TODO.md#L8) |
| `.agents/plans/archive/history/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md` · historical archive | **MISSING at cutoff** | [archive/history/historical-records/TODO.md:54](../../plans/archive/history/historical-records/TODO.md#L54) |
| `.agents/plans/archive/history/model-acquisition/TODO.md` · historical archive | **MISSING at cutoff** | [archive/history/handoff-cleanup/TODO.md:7](../../plans/archive/history/handoff-cleanup/TODO.md#L7); [archive/history/historical-records/TODO.md:85](../../plans/archive/history/historical-records/TODO.md#L85) |
| `.agents/plans/archive/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md` · historical archive | **MISSING at cutoff** | [archive/history/historical-records/TODO.md:52](../../plans/archive/history/historical-records/TODO.md#L52) |
| `.agents/plans/archive/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md` · historical archive | **MISSING at cutoff** | [archive/history/historical-records/TODO.md:62](../../plans/archive/history/historical-records/TODO.md#L62) |
| [.agents/plans/cli-device-routing/README.md](../../plans/cli-device-routing/README.md) · design/specification/runbook | file `cda9b856cb8024fe51e144e8174967c9e76ccd84a4cc8ae127c8049aa3b117ec` | [cli-device-routing/TODO.md:6](../../plans/cli-device-routing/TODO.md#L6) |
| [.agents/plans/cli-device-routing/TODO.md](../../plans/cli-device-routing/TODO.md) · owning queue | file `db95b7feea9e1bef0efa7747dbef8edb6d20850f14fd7970e57be8fe60b8b162` | [TODO.md:10](../../plans/TODO.md#L10) |
| [.agents/plans/cpu-gpu-optimization/BACKEND_PARITY.md](../../plans/cpu-gpu-optimization/BACKEND_PARITY.md) · design/specification/runbook | file `f77795ab496a8b52cf1eca1c0bfb43037d7e4d6596f1e040b37568359612b307` | [cpu-gpu-optimization/TODO.md:638](../../plans/cpu-gpu-optimization/TODO.md#L638) |
| [.agents/plans/cpu-gpu-optimization/BENCHMARK_CONTAINER_TERMINATION.md](../../plans/cpu-gpu-optimization/BENCHMARK_CONTAINER_TERMINATION.md) · design/specification/runbook | file `b6a00da62238468fe4d877900d6c02ca38a5f69ee2a58d04566b4eacbf7c05d9` | [cpu-gpu-optimization/TODO.md:181](../../plans/cpu-gpu-optimization/TODO.md#L181); [pipeline-composition/TODO.md:160](../../plans/pipeline-composition/TODO.md#L160) |
| [.agents/plans/cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md](../../plans/cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md) · design/specification/runbook | file `27f7e8893f37306511cc07de29bbf4514330bdc44edcdcb2586e68c7f1fdc28b` | [cpu-gpu-optimization/TODO.md:12](../../plans/cpu-gpu-optimization/TODO.md#L12); [memory-and-streaming/TODO.md:11](../../plans/memory-and-streaming/TODO.md#L11); [pipeline-composition/TODO.md:353](../../plans/pipeline-composition/TODO.md#L353) |
| [.agents/plans/cpu-gpu-optimization/SHORT_HF_RESULTS.md](../../plans/cpu-gpu-optimization/SHORT_HF_RESULTS.md) · design/specification/runbook | file `2e840e6f53d9462997fb698652a402eed167d7628ee3e5e787132da551482b76` | [cpu-gpu-optimization/TODO.md:11](../../plans/cpu-gpu-optimization/TODO.md#L11); [cpu-gpu-optimization/TODO.md:956](../../plans/cpu-gpu-optimization/TODO.md#L956); [serving-and-speculation/TODO.md:116](../../plans/serving-and-speculation/TODO.md#L116) |
| [.agents/plans/cpu-gpu-optimization/TODO.md](../../plans/cpu-gpu-optimization/TODO.md) · owning queue | file `66b049686f7f6b1910242a6ee7b56dfec3947bc016d9975c65dde36c93a1f038` | [TODO.md:8](../../plans/TODO.md#L8); [serving-and-speculation/TODO.md:287](../../plans/serving-and-speculation/TODO.md#L287) |
| [.agents/plans/distributed-execution/TODO.md](../../plans/distributed-execution/TODO.md) · owning queue | file `033606bb464b7b556d4a7d4cf859290856d8ac017af70641ab67b4ce53d2c9ff` | [TODO.md:17](../../plans/TODO.md#L17) |
| `.agents/plans/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md` · historical handoff | **MISSING at cutoff** | [model-execution/TODO.md:59](../../plans/model-execution/TODO.md#L59) |
| [.agents/plans/memory-and-streaming/TODO.md](../../plans/memory-and-streaming/TODO.md) · owning queue | file `d3ec590c55c88127ecb77324f1d8cc86b14f003a4d38a73ce9e31b0d99351fb0` | [TODO.md:7](../../plans/TODO.md#L7); [pipeline-composition/TODO.md:784](../../plans/pipeline-composition/TODO.md#L784); [serving-and-speculation/TODO.md:305](../../plans/serving-and-speculation/TODO.md#L305) |
| [.agents/plans/model-acquisition/README.md](../../plans/model-acquisition/README.md) · design/specification/runbook | file `b7c16440a4e8882b1c9e15a5ec2892eaf0c21b358e671b6f3bf9bd2175654fb4` | [model-acquisition/TODO.md:6](../../plans/model-acquisition/TODO.md#L6); [model-execution/TODO.md:446](../../plans/model-execution/TODO.md#L446) |
| [.agents/plans/model-acquisition/TODO.md](../../plans/model-acquisition/TODO.md) · owning queue | file `861f26b54d59580d3922b1e83c9d61e1b6bad9ae1a51cd5e03f5d6aec32b1d86` | [TODO.md:12](../../plans/TODO.md#L12); [model-acquisition/archive/through-a774ded/TODO.md:4](../../plans/model-acquisition/archive/through-a774ded/TODO.md#L4); [model-acquisition/archive/through-fce1353/TODO.md:5](../../plans/model-acquisition/archive/through-fce1353/TODO.md#L5); [pipeline-composition/TODO.md:785](../../plans/pipeline-composition/TODO.md#L785) |
| [.agents/plans/model-acquisition/archive/through-a774ded/TODO.md](../../plans/model-acquisition/archive/through-a774ded/TODO.md) · historical archive | file `4a178da79a9bc842b39d797f3a8141b28525e5a97cba3a11ba304c8edab27667` | [model-acquisition/TODO.md:64](../../plans/model-acquisition/TODO.md#L64) |
| [.agents/plans/model-acquisition/archive/through-fce1353/TODO.md](../../plans/model-acquisition/archive/through-fce1353/TODO.md) · historical archive | file `e0750281ddc39d99dac3e44542be9087d3fa18b305c0afa39c3785a502032d61` | [model-acquisition/TODO.md:61](../../plans/model-acquisition/TODO.md#L61); [model-acquisition/archive/through-a774ded/TODO.md:5](../../plans/model-acquisition/archive/through-a774ded/TODO.md#L5) |
| [.agents/plans/model-execution/TODO.md](../../plans/model-execution/TODO.md) · owning queue | file `ce7768c1b42630d9e92c3cd348a99cfaf3f3759b91a0c04acd2650a9eaaedb7f` | [TODO.md:11](../../plans/TODO.md#L11); [pipeline-composition/TODO.md:781](../../plans/pipeline-composition/TODO.md#L781) |
| [.agents/plans/multimodal/TODO.md](../../plans/multimodal/TODO.md) · owning queue | file `774818e6addf486ea8f64ca8a9c43dfb700c04c8c80954da99eae51cf04ae208` | [TODO.md:16](../../plans/TODO.md#L16); [pipeline-composition/TODO.md:783](../../plans/pipeline-composition/TODO.md#L783) |
| [.agents/plans/numerics-and-context/CORRECTNESS_POLICY.md](../../plans/numerics-and-context/CORRECTNESS_POLICY.md) · design/specification/runbook | file `7af352ae65cb0605a6fda99efd05d7418d66d4eb7153ee9daa748e2a8127fa26` | [cpu-gpu-optimization/TODO.md:19](../../plans/cpu-gpu-optimization/TODO.md#L19); [model-execution/TODO.md:14](../../plans/model-execution/TODO.md#L14); [model-execution/TODO.md:486](../../plans/model-execution/TODO.md#L486); [numerics-and-context/TODO.md:108](../../plans/numerics-and-context/TODO.md#L108) |
| [.agents/plans/numerics-and-context/TODO.md](../../plans/numerics-and-context/TODO.md) · owning queue | file `2d6ba97765e6fabeb589ee0bcef620c2db11410ff81eb729f6b8ee3737e456fa` | [TODO.md:6](../../plans/TODO.md#L6); [model-execution/TODO.md:1184](../../plans/model-execution/TODO.md#L1184); [serving-and-speculation/TODO.md:204](../../plans/serving-and-speculation/TODO.md#L204) |
| [.agents/plans/pipeline-composition/FORMATS.md](../../plans/pipeline-composition/FORMATS.md) · design/specification/runbook | file `5036c213f6c819c3eb58dab81e585da3efb63cc21cf13e9dc42fc3de6923ce3d` | [pipeline-composition/TODO.md:23](../../plans/pipeline-composition/TODO.md#L23) |
| [.agents/plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md](../../plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md) · design/specification/runbook | file `79c92f930edd63267a13c49a41ccf90371fa82c5d7ad6f53d7c09cef08f117b7` | [cpu-gpu-optimization/TODO.md:728](../../plans/cpu-gpu-optimization/TODO.md#L728); [pipeline-composition/TODO.md:387](../../plans/pipeline-composition/TODO.md#L387); [platform-and-release/TODO.md:21](../../plans/platform-and-release/TODO.md#L21) |
| [.agents/plans/pipeline-composition/README.md](../../plans/pipeline-composition/README.md) · design/specification/runbook | file `e1dc4d316c91bc1bbc65ef9158335242fdd5bac24c56a28809a0764b99cba4e4` | [pipeline-composition/TODO.md:6](../../plans/pipeline-composition/TODO.md#L6) |
| [.agents/plans/pipeline-composition/REGIONS_AND_PROFILES.md](../../plans/pipeline-composition/REGIONS_AND_PROFILES.md) · design/specification/runbook | file `afc847466f74e63f1d91340d42193c4c1b1c6f5b66810cb0e13ecc1eec9a51e7` | [cpu-gpu-optimization/TODO.md:731](../../plans/cpu-gpu-optimization/TODO.md#L731); [pipeline-composition/TODO.md:350](../../plans/pipeline-composition/TODO.md#L350); [pipeline-composition/TODO.md:421](../../plans/pipeline-composition/TODO.md#L421) |
| [.agents/plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md](../../plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md) · design/specification/runbook | file `ce7f4aa5d0225ea4a8a41990ec4754128a45f16ae92b677b2209f15919b1dba1` | [cpu-gpu-optimization/TODO.md:13](../../plans/cpu-gpu-optimization/TODO.md#L13); [distributed-execution/TODO.md:11](../../plans/distributed-execution/TODO.md#L11); [pipeline-composition/TODO.md:217](../../plans/pipeline-composition/TODO.md#L217) |
| [.agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md](../../plans/pipeline-composition/STRUCTURE_AND_NAMING.md) · design/specification/runbook | file `238200c1834d03e1ff8fade285d256a820f811af5c08d06ce8c5c5cd927b2f98` | [pipeline-composition/TODO.md:148](../../plans/pipeline-composition/TODO.md#L148) |
| [.agents/plans/pipeline-composition/TODO.md](../../plans/pipeline-composition/TODO.md) · owning queue | file `0a1ee475ffa8a61acebec376b122597ebda7e746a905f7aa66ab364a3a324e5b` | [TODO.md:9](../../plans/TODO.md#L9); [serving-and-speculation/TODO.md:288](../../plans/serving-and-speculation/TODO.md#L288) |
| [.agents/plans/platform-and-release/TODO.md](../../plans/platform-and-release/TODO.md) · owning queue | file `ad7ce7b692685a262a4acdc23702afda484f61a320d8ec0990ff12abed80cbd1` | [TODO.md:18](../../plans/TODO.md#L18) |
| [.agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md](../../plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md) · design/specification/runbook | file `4b4bb5b486b2519b761228c4e7c61f0abe3bd0089f03be3addf32d8b2a5b25ec` | [memory-and-streaming/TODO.md:76](../../plans/memory-and-streaming/TODO.md#L76); [memory-and-streaming/TODO.md:85](../../plans/memory-and-streaming/TODO.md#L85); [serving-and-speculation/TODO.md:292](../../plans/serving-and-speculation/TODO.md#L292) |
| [.agents/plans/serving-and-speculation/TODO.md](../../plans/serving-and-speculation/TODO.md) · owning queue | file `82abb6331d61088b93deb2713d605bbf03076113ce4802f09fd75ca363e2a4a0` | [TODO.md:5](../../plans/TODO.md#L5); [pipeline-composition/TODO.md:782](../../plans/pipeline-composition/TODO.md#L782) |
| [.agents/research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) · design/specification/runbook | file `30a30d0dc41a4aa37829ac92687c0cb4feaa0c2c3cc368767b18ddfa17b7bb02` | [cpu-gpu-optimization/TODO.md:784](../../plans/cpu-gpu-optimization/TODO.md#L784); [cpu-gpu-optimization/TODO.md:1212](../../plans/cpu-gpu-optimization/TODO.md#L1212) |
| [.agents/research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md) · design/specification/runbook | file `4a7a0da4210cf82c6c69521875e0a6271199f9899127bbb593b0ecc77ba994f5` | [model-execution/TODO.md:7](../../plans/model-execution/TODO.md#L7); [model-execution/TODO.md:1254](../../plans/model-execution/TODO.md#L1254) |
| [.agents/research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md](../../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md) · design/specification/runbook | file `fd3dfc14e583780556b32eaa5ad65e0003f39a7ff9152d1f2e6429382e837661` | [cpu-gpu-optimization/TODO.md:6](../../plans/cpu-gpu-optimization/TODO.md#L6) |
| [.agents/research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md](../../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md) · design/specification/runbook | file `5449aae6e719ee9b314c94cd588373f3dfadc4251034d55432c4fc4e00060745` | [cpu-gpu-optimization/TODO.md:5](../../plans/cpu-gpu-optimization/TODO.md#L5) |
| [.agents/research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md](../../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md) · design/specification/runbook | file `0a6d40f9e51aafb016b8379f5485bab3b6d29f30a0c524147db7e2a2617d77c1` | [model-acquisition/TODO.md:8](../../plans/model-acquisition/TODO.md#L8) |
| [.agents/research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md](../../research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md) · design/specification/runbook | file `cdbdd47321034fdab00031c21541c7f49dfddbe89e52d8c826fb7b0dbf471953` | [multimodal/TODO.md:8](../../plans/multimodal/TODO.md#L8) |
| [.agents/research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md](../../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md) · design/specification/runbook | file `10aa7ce134a6d8876833a2483f3826d599c74e51ae73c563b13ee1fa33dfa445` | [cpu-gpu-optimization/TODO.md:7](../../plans/cpu-gpu-optimization/TODO.md#L7) |
| [.agents/research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md](../../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md) · design/specification/runbook | file `ef8fc19168c602ce843dc19e95842294579bc0985e625c28dd5b84a19788d5ab` | [model-acquisition/archive/through-fce1353/TODO.md:17](../../plans/model-acquisition/archive/through-fce1353/TODO.md#L17) |
| [.agents/research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md) · design/specification/runbook | file `b45d031e0c058de6777bf652e2b0092ac1ea72c6c6c48b9157dc561b388b2839` | [model-execution/TODO.md:6](../../plans/model-execution/TODO.md#L6); [serving-and-speculation/TODO.md:7](../../plans/serving-and-speculation/TODO.md#L7) |
| [.agents/research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md](../../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) · design/specification/runbook | file `cf46b1e593ed5cf118ede4b676ef187f4dcec1987b7f58ae8d3e3b8f63cd10f3` | [memory-and-streaming/TODO.md:6](../../plans/memory-and-streaming/TODO.md#L6); [memory-and-streaming/TODO.md:171](../../plans/memory-and-streaming/TODO.md#L171) |
| [.agents/research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md](../../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md) · design/specification/runbook | file `064b63c2cd342a464f8ff3325189df5ca35b83803e5c5bc45ed84e9738119d17` | [serving-and-speculation/TODO.md:6](../../plans/serving-and-speculation/TODO.md#L6) |
| [.agents/research/engine-optimization-comparison-2026-10-21/README.md](../../research/engine-optimization-comparison-2026-10-21/README.md) · design/specification/runbook | file `ce1ba5c54531dd65f436d99fd608d829ef19ff0519a2443610c4b4c7d09ae916` | [cpu-gpu-optimization/TODO.md:9](../../plans/cpu-gpu-optimization/TODO.md#L9); [distributed-execution/TODO.md:18](../../plans/distributed-execution/TODO.md#L18); [memory-and-streaming/TODO.md:23](../../plans/memory-and-streaming/TODO.md#L23); [model-execution/TODO.md:276](../../plans/model-execution/TODO.md#L276); [pipeline-composition/TODO.md:418](../../plans/pipeline-composition/TODO.md#L418); [serving-and-speculation/TODO.md:286](../../plans/serving-and-speculation/TODO.md#L286) |
| [.agents/research/engine-optimization-comparison-2026-10-21/composition-review.md](../../research/engine-optimization-comparison-2026-10-21/composition-review.md) · design/specification/runbook | file `591ca60be3eb97265e76d4e718583161131c069b4d9729df684e93f260bd0b16` | [pipeline-composition/TODO.md:551](../../plans/pipeline-composition/TODO.md#L551) |
| [.agents/research/engine-optimization-comparison-2026-10-21/exllamav3-adoption-scope.md](../../research/engine-optimization-comparison-2026-10-21/exllamav3-adoption-scope.md) · design/specification/runbook | file `f0876f1d566fef69b2dd3a701daec875122581919b89415f6060ba4bd23f2ec7` | [cpu-gpu-optimization/TODO.md:737](../../plans/cpu-gpu-optimization/TODO.md#L737) |
| [.agents/runs/architecture-astra-compiler-readiness-20261024/DESIGN.md](../../runs/architecture-astra-compiler-readiness-20261024/DESIGN.md) · run design | file `5b866c123cb09184dfce6966eb7467fc5053aa8a02be0f8097ba9694e8d9e386` | [cpu-gpu-optimization/TODO.md:463](../../plans/cpu-gpu-optimization/TODO.md#L463); [pipeline-composition/TODO.md:523](../../plans/pipeline-composition/TODO.md#L523) |
| [.agents/runs/architecture-astra-cross-width-contract-20261024/DESIGN.md](../../runs/architecture-astra-cross-width-contract-20261024/DESIGN.md) · run design | file `9bfd63ca8a0000cc97c5550249f81a1cc3ffe90aad0b4111a20a0fa7f44273bd` | [pipeline-composition/TODO.md:191](../../plans/pipeline-composition/TODO.md#L191) |
| [.agents/runs/architecture-astra-next-native-review-20261023/DESIGN.md](../../runs/architecture-astra-next-native-review-20261023/DESIGN.md) · run design | file `6a6dd6637211c012b3eb2a949288839fbb619cfdfce0ab444ce648fbea888c35` | [pipeline-composition/TODO.md:226](../../plans/pipeline-composition/TODO.md#L226) |
| [.agents/runs/architecture-astra-prepared-review-20261023/DESIGN.md](../../runs/architecture-astra-prepared-review-20261023/DESIGN.md) · run design | file `1b6432b9856aaace83c0ec69670867ee364789f8db1d3a6f455ed374c74bb388` | [pipeline-composition/TODO.md:320](../../plans/pipeline-composition/TODO.md#L320) |
| [.agents/runs/architecture-astra-qwen-pool-review-20261023/TODO.md](../../runs/architecture-astra-qwen-pool-review-20261023/TODO.md) · run evidence | file `20a4f31cfa25aed93021ec0e174e0e89ed6421c846b4824890e29ffd52fe155a` | [model-execution/TODO.md:184](../../plans/model-execution/TODO.md#L184) |
| [.agents/runs/architecture-astra-residency-and-set-key-20261023/DESIGN.md](../../runs/architecture-astra-residency-and-set-key-20261023/DESIGN.md) · run design | file `3ea9389818f0b631011d6ce472d303827e7ddecd60965495b97f9aa533122f12` | [cpu-gpu-optimization/TODO.md:483](../../plans/cpu-gpu-optimization/TODO.md#L483); [model-execution/TODO.md:198](../../plans/model-execution/TODO.md#L198); [pipeline-composition/TODO.md:258](../../plans/pipeline-composition/TODO.md#L258) |
| [.agents/runs/architecture-astra-shared-native-20261023/DESIGN.md](../../runs/architecture-astra-shared-native-20261023/DESIGN.md) · run design | file `5df41f3dcdfe6f9b87384e819fd29f0cd3f74f5516ad65daf8c7d3cd104dde0a` | [cpu-gpu-optimization/TODO.md:478](../../plans/cpu-gpu-optimization/TODO.md#L478); [pipeline-composition/TODO.md:246](../../plans/pipeline-composition/TODO.md#L246) |
| [.agents/runs/b4-takeover-codex-20261025/FINAL_REPORT.md](../../runs/b4-takeover-codex-20261025/FINAL_REPORT.md) · run evidence | file `53ce29896e837fefd6c3d4464a31591e704abad3085a17d93ab37266c88dc154` | [cpu-gpu-optimization/TODO.md:34](../../plans/cpu-gpu-optimization/TODO.md#L34) |
| [.agents/runs/b4-takeover-codex-20261025/TODO.md](../../runs/b4-takeover-codex-20261025/TODO.md) · run evidence | file `26a447ce0d93eb40688f14da49b03663f68513ebc52fff6281f353a44a43f704` | [cpu-gpu-optimization/TODO.md:33](../../plans/cpu-gpu-optimization/TODO.md#L33) |
| [.agents/runs/b4-takeover-codex-20261025/matched-c16-v6/root-both-closed-review.json](../../runs/b4-takeover-codex-20261025/matched-c16-v6/root-both-closed-review.json) · run evidence | file `51e609adf76c40462031352faa93691cc6cc42800d854625c07221a7d1d5481e` | [cpu-gpu-optimization/TODO.md:168](../../plans/cpu-gpu-optimization/TODO.md#L168) |
| [.agents/runs/b4-takeover-codex-20261025/x2-current-preflight/root-resource-review.json](../../runs/b4-takeover-codex-20261025/x2-current-preflight/root-resource-review.json) · run evidence | file `6a231fccd32f53846424f30de01b9d2103af66b8377c0f9eae98ce5432f07fbd` | [cpu-gpu-optimization/TODO.md:180](../../plans/cpu-gpu-optimization/TODO.md#L180) |
| [.agents/runs/cpu-gpu-optimization-astra-c16-shared-gap-20261024/DESIGN.md](../../runs/cpu-gpu-optimization-astra-c16-shared-gap-20261024/DESIGN.md) · run design | file `77c52ea46af9378ed5edf1ddb0cb1729f35f6113828358f470648b075bca4b75` | [cpu-gpu-optimization/TODO.md:894](../../plans/cpu-gpu-optimization/TODO.md#L894) |
| [.agents/runs/cpu-gpu-optimization-cohort-b4-root-20261023/TODO.md](../../runs/cpu-gpu-optimization-cohort-b4-root-20261023/TODO.md) · run evidence | file `3d707853307152532c66339bf56974918a4dfe90b9150953d85b5d1cab0974b9` | [cpu-gpu-optimization/TODO.md:455](../../plans/cpu-gpu-optimization/TODO.md#L455) |
| [.agents/runs/cpu-gpu-optimization-physical-owner-root-20261023/TODO.md](../../runs/cpu-gpu-optimization-physical-owner-root-20261023/TODO.md) · run evidence | file `694f1b9886f9586b56d88aa4348b1a46f7978ed390cd39bc4c64107f37e0756d` | [cpu-gpu-optimization/TODO.md:394](../../plans/cpu-gpu-optimization/TODO.md#L394) |
| [.agents/runs/gpu-model-boundary-20261026/root/validation.json](../../runs/gpu-model-boundary-20261026/root/validation.json) · run evidence | file `d79e103d932e765041ec144bccca9732b80ed21fddb9f04074abe7e3ef072579` | [pipeline-composition/TODO.md:108](../../plans/pipeline-composition/TODO.md#L108); [pipeline-composition/TODO.md:123](../../plans/pipeline-composition/TODO.md#L123) |
| [.agents/runs/gpu-model-boundary-20261026/worktree-reconcile/retirement.json](../../runs/gpu-model-boundary-20261026/worktree-reconcile/retirement.json) · run evidence | file `38d681bcc45206878051ac03fe8ddcb5876545807f88708452466cb751bb1a21` | [pipeline-composition/TODO.md:130](../../plans/pipeline-composition/TODO.md#L130) |
| [.agents/runs/model-execution-astra-phase-profiles-review-20261024/REVIEW.md](../../runs/model-execution-astra-phase-profiles-review-20261024/REVIEW.md) · run evidence | file `43ea4dbf29f497ea35b584808e9304f07a4ec2fdbf364ed701bdce8c24fa832c` | [pipeline-composition/TODO.md:213](../../plans/pipeline-composition/TODO.md#L213) |
| [.agents/runs/model-execution-astra-qwen-natural-capture-correction-20261024/DESIGN.md](../../runs/model-execution-astra-qwen-natural-capture-correction-20261024/DESIGN.md) · run design | file `11a4887388ff201dbec803598ec9619226a283ccdb503cde6d6948d1027aa769` | [model-execution/TODO.md:532](../../plans/model-execution/TODO.md#L532) |
| [.agents/runs/model-execution-current11-assembled-checks-sol-high-20261024/REPORT.md](../../runs/model-execution-current11-assembled-checks-sol-high-20261024/REPORT.md) · run evidence | file `e8eca6d2e8046cafaae5b82a2e2da96c4dce66df6a1a5cd5cc83dd08fc0fca94` | [model-execution/TODO.md:144](../../plans/model-execution/TODO.md#L144) |
| [.agents/runs/model-execution-gemma-final-feedback-verifier-v2-sol-high-20261024/REPORT.md](../../runs/model-execution-gemma-final-feedback-verifier-v2-sol-high-20261024/REPORT.md) · run evidence | file `fbd93a48ae7708525494fb04fb60ac44ca8dac51b9a66754919d5a68b40adbb1` | [model-execution/TODO.md:145](../../plans/model-execution/TODO.md#L145) |
| [.agents/runs/model-execution-gemma-final-owner-review-sol-high-20261024/REVIEW.md](../../runs/model-execution-gemma-final-owner-review-sol-high-20261024/REVIEW.md) · run evidence | file `24f22cc6633eadaf1e19b308334581af42178d90445e85e32b790c93c2ef2702` | [model-execution/TODO.md:146](../../plans/model-execution/TODO.md#L146) |
| [.agents/runs/model-execution-gemma-series-audit-review-sol-high-20261024/REPORT.md](../../runs/model-execution-gemma-series-audit-review-sol-high-20261024/REPORT.md) · run evidence | file `ae174bf172bbf569b9063f70d5e72e1c212f4c08cfc1e6bd7744227331f32cb4` | [model-execution/TODO.md:550](../../plans/model-execution/TODO.md#L550) |
| [.agents/runs/model-execution-qwen-gemma-vllm-audit-20261024/REPORT.md](../../runs/model-execution-qwen-gemma-vllm-audit-20261024/REPORT.md) · run evidence | file `b5d4465aea4678c44ac99d0120d390420af64676124b468ff4764f73af41dd49` | [cpu-gpu-optimization/TODO.md:1031](../../plans/cpu-gpu-optimization/TODO.md#L1031); [model-execution/TODO.md:538](../../plans/model-execution/TODO.md#L538); [pipeline-composition/TODO.md:171](../../plans/pipeline-composition/TODO.md#L171) |
| [.agents/runs/modular-architecture-20261026/root/final-coverage.json](../../runs/modular-architecture-20261026/root/final-coverage.json) · run evidence | file `23678323e7dc8412e8856e7fad1a8e8c4c137566ce745f599c72ea9775e8b089` | [pipeline-composition/TODO.md:64](../../plans/pipeline-composition/TODO.md#L64) |
| [.agents/runs/modular-architecture-20261026/worktree-reconcile-sol/final-batch-result.json](../../runs/modular-architecture-20261026/worktree-reconcile-sol/final-batch-result.json) · run evidence | file `57e8491eca16724c822ca3a476fd6127920186d992bb4059abb89ec1de41b2de` | [pipeline-composition/TODO.md:78](../../plans/pipeline-composition/TODO.md#L78) |
| [.agents/runs/numerics-and-context-upstream-cuda-root-20261020/TODO.md](../../runs/numerics-and-context-upstream-cuda-root-20261020/TODO.md) · run evidence | file `ae6663614458c3a1f6e400f2d3aae3120bef01f2cee5dacedafacbbf7d5c8bda` | [numerics-and-context/TODO.md:151](../../plans/numerics-and-context/TODO.md#L151); [serving-and-speculation/TODO.md:333](../../plans/serving-and-speculation/TODO.md#L333) |
| [.agents/runs/pipeline-composition-astra-hip-multiset-review-20261024/REPORT.md](../../runs/pipeline-composition-astra-hip-multiset-review-20261024/REPORT.md) · run evidence | file `1c482245379f751050b867f6a4e47a5d0d8fbe22682ba2fab8ee1c7919c59fcc` | [pipeline-composition/TODO.md:263](../../plans/pipeline-composition/TODO.md#L263) |
| [.agents/runs/pipeline-composition-hip-multiset-sol-high-20261024/compiler-successor-v2/REVIEW.md](../../runs/pipeline-composition-hip-multiset-sol-high-20261024/compiler-successor-v2/REVIEW.md) · run evidence | file `437ed2b641a84d19b221eb17fae278dca7a156a75671d540ee3437d90d028d72` | [pipeline-composition/TODO.md:278](../../plans/pipeline-composition/TODO.md#L278) |
| [.agents/runs/q08-correctness-20261026/TODO.md](../../runs/q08-correctness-20261026/TODO.md) · run evidence | file `6fb859cd6a803691cf06d9d0bc5a2486492d7ec977884f0d9e09f2b74b60e245` | [cpu-gpu-optimization/TODO.md:794](../../plans/cpu-gpu-optimization/TODO.md#L794); [model-execution/TODO.md:487](../../plans/model-execution/TODO.md#L487); [numerics-and-context/TODO.md:360](../../plans/numerics-and-context/TODO.md#L360) |
| [.agents/runs/resume-all-parallel-root-20261024/WORKTREE_CLEANUP.md](../../runs/resume-all-parallel-root-20261024/WORKTREE_CLEANUP.md) · run evidence | file `45ad117dcecf32e4f1f3deda4c2316752fbaab523d941939f09739416015dd26` | [cpu-gpu-optimization/TODO.md:879](../../plans/cpu-gpu-optimization/TODO.md#L879); [platform-and-release/TODO.md:54](../../plans/platform-and-release/TODO.md#L54) |
| [.agents/runs/resume-all-parallel-source-supervision-sol-20261024/C16_FINAL_SCOPE_REVIEW.json](../../runs/resume-all-parallel-source-supervision-sol-20261024/C16_FINAL_SCOPE_REVIEW.json) · run evidence | file `db17e813fedfff01064d57a11733b49ee9d68cb1cc75292e58af5ff66a7777a4` | [cpu-gpu-optimization/TODO.md:237](../../plans/cpu-gpu-optimization/TODO.md#L237) |
| [.agents/runs/resume-all-queue-production-bound-astra-review-20261024/MERGE_SCOPE_DECISION.md](../../runs/resume-all-queue-production-bound-astra-review-20261024/MERGE_SCOPE_DECISION.md) · run evidence | file `a6f16b9a0e1b0928f5d7ede4cd914fd21b1528e3d5f79dddac52fc3ef13e19c2` | [cpu-gpu-optimization/TODO.md:808](../../plans/cpu-gpu-optimization/TODO.md#L808) |
| [.agents/runs/resume-all-queue-production-bound-astra-review-20261024/TOKENIZER_GROWTH_CREDIT_PLAN.md](../../runs/resume-all-queue-production-bound-astra-review-20261024/TOKENIZER_GROWTH_CREDIT_PLAN.md) · run design | file `7420e2f7776707f4b4b1da3d3bd6bd1cabf7b2bea241289c5c657e9d8dd4d063` | [cpu-gpu-optimization/TODO.md:835](../../plans/cpu-gpu-optimization/TODO.md#L835) |
| [.agents/runs/resume-all-takeover-queue-gpu-glm-20261025/LOG.md](../../runs/resume-all-takeover-queue-gpu-glm-20261025/LOG.md) · run evidence | file `5b77d61901e5ab8b6feebe44816cab827de030be450c33df282cf1f8b1c2e83e` | [cpu-gpu-optimization/TODO.md:814](../../plans/cpu-gpu-optimization/TODO.md#L814) |
| [.agents/runs/resume-all-tokenizer-text-fixed-coverage-pi-20261025](../../runs/resume-all-tokenizer-text-fixed-coverage-pi-20261025) · run evidence | existing directory; no file hash | [cpu-gpu-optimization/TODO.md:845](../../plans/cpu-gpu-optimization/TODO.md#L845) |
| [.agents/runs/resume-all-tokenizer-vm-cpu-continuation-prep-pi-20261025](../../runs/resume-all-tokenizer-vm-cpu-continuation-prep-pi-20261025) · run evidence | existing directory; no file hash | [cpu-gpu-optimization/TODO.md:824](../../plans/cpu-gpu-optimization/TODO.md#L824) |
| [.agents/runs/serving-shutdown-20261026/reconcile/retirement.json](../../runs/serving-shutdown-20261026/reconcile/retirement.json) · run evidence | file `4d5dd2e8b6d4fe7d2e23513ae11edf91e3841c77783da9e102c713c881a0ecea` | [serving-and-speculation/TODO.md:490](../../plans/serving-and-speculation/TODO.md#L490) |
| [.agents/runs/serving-shutdown-20261026/root/validation.json](../../runs/serving-shutdown-20261026/root/validation.json) · run evidence | file `97b84f2050de7cd8c6904580188c7548a9c39d8f45be1cc1e1e6e712c3d697ce` | [serving-and-speculation/TODO.md:487](../../plans/serving-and-speculation/TODO.md#L487); [serving-and-speculation/TODO.md:504](../../plans/serving-and-speculation/TODO.md#L504) |
| [.agents/runs/source20-x2-actual-namespace2-pi-20261024/REPORT.md](../../runs/source20-x2-actual-namespace2-pi-20261024/REPORT.md) · run evidence | file `394e19926e85fd3dd1344914ca8610460ace0a41b0e9d4f23b8cad406084542f` | [model-execution/TODO.md:523](../../plans/model-execution/TODO.md#L523) |
| [.agents/runs/stash-recovery-20261026/root/REPORT.md](../../runs/stash-recovery-20261026/root/REPORT.md) · run evidence | file `7f8a0c6036846bdbe8fd39ef145295880b052c394b2fc9a3851fb1dd5b5fb43e` | [pipeline-composition/TODO.md:91](../../plans/pipeline-composition/TODO.md#L91); [pipeline-composition/TODO.md:100](../../plans/pipeline-composition/TODO.md#L100) |
| [docs/DEVELOPMENT.md](../../../docs/DEVELOPMENT.md) · design/specification/runbook | file `3427b2041dd19bc3053ef79a11d6bade5eefc493131216de0dc9810f082a63a5` | [platform-and-release/TODO.md:99](../../plans/platform-and-release/TODO.md#L99) |
| [tests/reference/llama_cpp/README.md](../../../tests/reference/llama_cpp/README.md) · design/specification/runbook | file `9a42a49a5e6ec12dfc22be5d3f29476c6adf690b2a733d83b47c669ed566a058` | [numerics-and-context/TODO.md:6](../../plans/numerics-and-context/TODO.md#L6) |

Registry totals: **119 distinct targets**, **173 direct source links**, **15 missing local targets**. Missing archive-relative paths and retired worktree references remain historical provenance gaps. The live model-execution PM-handoff link resolves to nonexistent `.agents/plans/docs/…`; its literal source spelling is retained.
