# Mechanisms organized into profiles

Planning crosswalk, 21 Oct2026. All profiles and new recursive behavior remain
unimplemented/unqualified. The [region/profile contract](../../plans/pipeline-composition/REGIONS_AND_PROFILES.md)
defines legal composition; the [416-entry comparison](README.md) retains source
dispositions. This is a view of that research, not another execution queue.

## Reusable pipeline components

| Component | Nested regions/stages | Original owner |
|---|---|---|
| Source preparation | Verify original roles/bytes → optional exact pack → admitted residency/upload | Source owner + physical arbiter |
| Mathematical inference | Embedding → repeated source-defined layers → optional norm/head; each layer exports typed subregions/effects | Existing model traversal and numerical closures |
| Request phases | Tokenize/render → prefill/state-only → decode; optional propose/verify/accepted-prefix preparation | Trusted task control + independent original sessions |
| State reuse | Compatible complete-prefix lookup → joint frontier → restore/COW → private continuation | StateRoot/session authority and original reader leases |
| Capacity/IO | Bounded source/state dependency reads → staging → actual copy readiness → consume/retire | Same physical/IO/device owners |
| Readout/delivery | Exact selected sampler/grammar → actual Ready → original root publication → bounded ordered output | Original session writer + output credits |
| Compound AI | Typed processor/model/codec stages, each optionally a subpipeline | Trusted task driver; explicit staged commits/frontiers |

These are composition roles, not newly implemented modules. A primitive or
closed subpipeline can lower to a kernel, several kernels or a captured region.
Hierarchy imposes no physical launch. Mathematical regions never publish roots.

## Profile recipes

| Profile | Combines | Measurement target / principal conflict |
|---|---|---|
| RI `resident-interactive` | Prepared inference/decode + compact readout + bounded fair small cohorts; exact local fusion/replay when eligible | True-token ITL/TTFT; large chunks/deep drafts/extra pools can lose |
| RT `resident-throughput` | True temporal prefill + native independent rows + stable metadata + aged mixed phase budgets + qualified replay | Correct concurrent SLO-goodput; wider batches cost scratch and tails |
| LP `long-shared-prefix` | Long source-qualified inference + paged complete state + exact prefix/COW + optional restore tier | Avoided actual work/capacity and tails; KV-only or shifted/truncated hits invalid |
| MD `memory-bounded-dense` | Every required dense layer/window + row reuse + admitted bounded IO/copy/compute; first correct path may be synchronous | Dense27B correctness/bandwidth/working-set feasibility; activation spill and rereads can dominate |
| SC `sparse-capacity` | Source-selected routing/grouped experts + protected byte-budgeted hotsets + deterministic learned lookup + admitted transfers | Actual sparse source goodput; predictions, replicas and optional drafting compete for IO/memory |
| DP `derived-precision` | Explicit selected source/calibration recipe + its exact codec kernels + separately qualified activation/state precision | Declared quality/memory/performance tradeoff; incompatible with original exact-profile state/cache |
| MI `media-interactive` | Source-real processor/ASR/language/TTS or independently selected native speech pipeline + clocks/credits/barge-in | Trained media quality and complete-chunk latency; already committed/delivered/played output cannot roll back |
| DX `distributed-placement` | Same qualified regions + authorized topology/partition/transfer/collective/state-handoff stages | Complete task goodput/capacity; link/epoch/reduction/straggler costs can erase gains |

RI/RT express workload emphasis; source/arithmetic remain independent bindings.
LP/MD/SC/DP/MI/DX add capabilities with explicit prerequisites, not an arbitrary
inheritance graph or permission to combine every recipe. Resolve a finite
canonical choice once; reject conflicting/unsupported combinations. Current
baseline never automatically enables DP/approximation to meet capacity or speed.

## Family coverage

Each source entry maps through the [21 mechanism families](README.md#cross-engine-mechanism-comparison)
to these recipes. **Base** = invariant/common facility; **candidate** = potential
strategy requiring the actual model/shape/device and its independent gates;
**deferred** = future source/device/task scope. None means promoted runtime.

| Family | Profile use |
|---|---|
|01 Source/pack identity | Base all; exact prepared repacks candidates RI/RT/MD/SC; no source-issuer sharing by hash |
|02 Numerical/quality contracts | Base all; changed arithmetic/quant/context/sampling only explicitly selected DP or distinct declared semantics |
|03 Dense/quant kernel regimes | Candidates RI/RT/LP/MD; SC/MI/DX only where their actual child operators match |
|04 Legal fusion | Candidates all inside qualified closed child regions; no fusion across unpreserved state/control/media boundaries |
|05 Attention/recurrence/positions | RI/RT/LP/MD by source; SC uses actual sparse/indexer semantics; MI/DX only registered matching child operations |
|06 Graphs/device programs | Candidate RI/RT and eligible LP/MD/SC/MI/DX; every captured reader/pool/shape charged; unknown failure refuses |
|07 Metadata/feedback overlap | Candidate all; stable source-owned preparation plus fresh root/epoch/row bindings, no fabricated future input |
|08 Fair native/temporal work | Base all; RT primary batching/temporal focus, RI/MI protected latency, LP/MD/SC bounded long/IO work |
|09 Physical/liveness/phase lending | Base all; actual joint live set after strategy selection, no sum/max of child peaks as proof |
|10 Full hybrid pages/prefix/COW | LP primary; RI/RT/MD/SC candidates; MI typed state and DX complete handoff only after model/task qualification |
|11 Persistence/restore economics | LP/MD/SC candidates; optional other profiles; portable diagnostic snapshots do not enable product persistence |
|12 Bounded IO/prefetch | MD/SC primary; LP state tiers; MI/DX explicitly selected dependencies. RI/RT resident path has no mandatory IO scaffolding |
|13 MoE/expert placement | SC and actual MoE child models only; never MD dense27B; workload RI/RT can select compatible sparse child pipelines |
|14 Verification/rollback | Optional RI/RT/MD/SC/LP after target verifier; MI/DX only qualified task/topology. Every state/RNG/grammar branch participates |
|15 Proposers/adaptive budget | Optional same eligible profiles, actual target/source compatible; K=0 fallback. No forced acceptance or hidden distribution change |
|16 Sampling/grammar/readout | Base only as task requests; RI/RT/LP/MD/SC text lanes, MI typed output; DX transfers committed outputs without extra publisher |
|17 CPU/topology | Candidates all on actual available topology; exact cross-placement arithmetic first, existing workers retain authority |
|18 Distribution | DX deferred; no hardware qualification from research/config. Other profiles can use DX only with complete compatible bindings |
|19 Context/models/media/profiles | Base source/grant/isolation; MI compound tasks deferred. Shared preprocessing only with exact contracts; private neural state remains per session |
|20 Model-specific/approximate semantics | Deferred matching SC/MI/DX source tasks; explicit changed DP/semantic variants only. Trained sparsity/shared-KV is not a general optimization |
|21 Measurement/cost/energy/negative cases | Base all; independent changes then material profile combinations; all offered correct useful work/tails/physical cost |

## First implementation and acceptance

Current .8B binding/publication and native2B parity stay first. Then establish
matched short C1/2/4/8 before selecting one significant generic bottleneck.
The first profile increment can combine **one existing source/profile/shape,
one closed region with forced decomposition and one accepted optimized variant**.
It needs no general recursive runtime or multimodal framework first.

After real source/state/cancel/physical and complete check.sh gates, compare
unprofiled complete requests, then a material second combination. Count actual
device kernels separately from graph submissions. Retain losing combinations,
startup/capture/pack cost, true-token tails, all offered work, memory/IO/energy
and fairness. Continue long contexts→dense27B→Gemma4 in the user's order.
