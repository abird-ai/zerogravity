# Fabric vNext transition — settled decisions, risks, unresolved choices and acceptance law

This is an implementation-focused decision register made 27 October 2026. It **supplements** the fully preserved historical user/architecture decision ledger at ../fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md and the exact latest user instructions in this plan's README.md. It does not supersede source truth or active owner plans.

## 1. Explicitly binding user decisions

- User fully permits pre-v1 breaking of Rust internal/public APIs, crates, module names, implementations and old artifacts to achieve cleaner scalable maintainable design. **No compatibility shims** required at final transition. Correct source/math/device/state/lifecycle and existing supported user operations must survive.
- The architecture must be a **single reusable AI execution fabric**, not cloned inference engines per model/backend/format/modality; vLLM and llama.cpp lessons are incorporated but neither imported wholesale.
- **Dedicated independently usable zerogravity-serving Rust library** owns HTTP/OpenAI/Responses/SSE and future realtime protocols, with no serving implementation buried in CLI.
- **All functionality exported through one public zerogravity Rust library** for arbitrary external programs: model/load/inspect/pull/cache, devices, run, sessions, pipeline, diagnostics, server start/stop, eventually modalities/distributed. CLI only flags, terminal and presentation/signals.
- The **complete target architecture must be visible immediately as actual compiled Rust code** in the FIRST structural source milestone, with typed TODO stubs where necessary. File tree alone is not enough.
- Then physically move current source implementations into this structure, make current functionally supported feature set work and stay correct, delete old paths, commit in logical units as implementation proceeds.
- Each phase is committed and tested with cheap targeted checks. **Do not repeatedly run heavyweight model/GPU/long-context/release qualification before the large structural + functional transition is complete.** Run one consolidated complete qualification afterward; follow with specific bugfix commits and targeted tests; one final full gate when stable.
- User's **current request is to write detailed plan and handoff only**, not actually implement/refactor/commit Rust now. The ready-to-paste successor prompt will direct a future explicitly approved agent how to begin.

## 2. Accepted technical design refinements after 41-commit source review

| Decision | Reason and implementation rule |
| --- | --- |
| Add zerogravity-engine as a real model-neutral library | Extract model/session scheduler/router and task/physical preparation from CLI/facade; engine does not import model family or GPU directly |
| Add zerogravity-serving as independent library | Protocol and socket lifecycle are reusable; server→engine/core, facade→server, CLI→facade, no cycle |
| Existing physical ResourceArbiter stays low-level | CPU/GPU/artifacts need real lease/physical identity; moving arbiter wholesale to engine would create a reverse dependency, duplicate wallets |
| CPU TaskCoordinator and GPU SessionAuthority converge logically | One RootAttempt/prepared domains/physical Ready/PublishRoot protocol; preserve original physical signoffs, no second publisher |
| Keep existing ForwardPlan as foundation | One canonical SemanticProgram rather than a new unproved IR+interpreter |
| Only SemanticProgram and ExecutableBundle enduring | RIR/KIR transient, CIR embedded physical commands; no graph interpretation per token |
| Model families own source math | Qwen/Gemma config/source roles/traversal/state/specialization; GPU implements physically reusable kernels/ABI only |
| Existing GPU→Qwen dependency removal is already DONE | Commit 263ff79 and checker; don't waste time recreating that refactor |
| Quant/codec system is not one QuantKind | Container vs stored scalar/quant vs activation encoding vs numerical behavior vs packed layout vs derived lossy checkpoint vs physical qualification |
| Q4_K scalar decode is not CPU source-operator support | Preserve explicit source dot refusal until independently qualified |
| Q1/Q2/Q3/NVFP4 current dirs are placeholders | Keep capability false and typed Unsupported despite visible code homes |
| GPU CUDA GDN parameterized kernel implemented source ≠ ordinary model capable | Model source + backend exact implementation + numeric/ABI/shape proofs and physical Ready required, finite captures remain diagnostic |
| Resource and task schedulers are separate levels | One global fair admission policy, stage-specific work units; no separate capacity ledger or token-only scheduler for all media |
| Complete original functionality before future features | Existing GGUF/CPU/OpenAI/GPU finite/codec/acquisition/cache/device/pipeline checker semantics restored; new codecs/media/distributed are later |
| Heavy qualification once after major changes | S/F Tier L/M quick source checks and signed commits, Q0 merged broad gate, Q1 focused bugfix, Q2 final full gate |
| Source/user rights and hardware holds remain binding | No dangerous gfx90c/clock reset/unknown source download/remote peers from design or skeleton |

## 3. Major architectural rejection decisions

**Reject**:
- Separate per-model/per-format/per-API engine or "legacy worker" kept in parallel just to avoid breaking pre-v1 imports.
- A full runtime scheduler/ResourceArbiter in CLI or serving crate.
- Moving low-level ResourceArbiter *implementation* upwards without solving CPU/GPU/artifact dependency cycles; duplicating pools instead.
- A fake general ModelRunner that repeatedly calls a model-specific short-HF CUDA fixed schedule under the hood while claiming dynamic production.
- Generic Q1/Q2/Q3/Q4/NVFP4 codec recognition based on bits-per-weight without exact released format and numerical operators.
- Treating BF16 numerical precision and quantized storage as the same thing, or lossless source pack and lossy derived artifact as one checkpoint.
- Treating model checkpoint metadata/registry YAML/typed port schema as trained runtime capability.
- Reusing a passing finite GDN CUDA captured test to authorize arbitrary fresh 48–50/context/source GPU calls.
- Full SemanticProgram rewrite that duplicates current ForwardPlan or adds four durable compiler IRs without a proven use.
- CPU/GPU per-op host readback or generic stage serialization when qualified device-resident handoff is possible.
- Broad autograd/training/framework hierarchy, agent workflow DSL, dynamic executable plugins before real vertical slices.
- Running full scripts/check.sh and long GPU model/reference suites after every moved file.
- Using todo!() to deliver panics from supported SDK/HTTP/CLI call paths.
- Treating all generated .cu source as an architecture flaw that must be rewritten; generated source is fine with correct ABI/code provenance.
- Publishing/pushing/releasing based on a planning request alone.

## 4. Risks that can derail a fast transition

| Risk | Severity | Mandatory mitigation |
| --- | --- | --- |
| Skeleton becomes non-functional second runtime | High | It is typed structure; existing routes may remain temporarily but no new claims; F0–F4 fully migrate/delete old |
| Two ResourceArbiters give false available memory | Critical | Process-shared core authority; engine policies reserve original ledger; alias and UMA accounts once |
| CPU and GPU each publish separate StateRoot semantics | Critical | Canonical root/epoch/frontier/physical Ready gate and one publisher |
| Model math changes under a crate rename | Critical | Preserve original source, cast/reduction, exact operator fixtures; Q0 actual model proof |
| CUDA captures/encoded buffers freed early | Critical | Real physical lease retirement from encoding/queue/capture through device Ready, unknown quarantine |
| New codec enum falsely advertises unsupported Q4_K/Q1/Q2/NVFP4 | Critical | Separate decode/operator/backend/whole-model evidence, Unsupported-first |
| vLLM performance goals lost in generic abstraction | High | Prepared executable/row slots, no per-token name/JSON/plan/compiler/global lock, measured E02 |
| Serving facade cycle / CLI still owns HTTP | High | Enforced acyclic Cargo DAG and public Rust embed tests |
| Acquisition and security edges lost in CLI extraction | High | Full pull S3/GCS/Azure/HF/Xet/OCI/cache tests, source/privacy rights |
| New stub panic in supported runtime | Critical | Explicit Unsupported errors or delegate current path; unreachable todo only in internal gate |
| Broad full suites repeatedly block transition | Medium-high | Tier L/M focused checks per commit, Q0 one main run, Q1 fixes, Q2 final |
| Heavy qualification delayed hides obvious broken import or source guard | High | Every phase compile+small actual invariant tests; do not commit broken source safety |
| Historical binary evidence retagged after code image changes | Critical | Preserve source/driver/model fingerprints, fresh physical qualification when allowed |
| Live source moved by other agents while plan executed | High | Recheck HEAD/active TODO/owner, explicit file ownership, one integration coordinator on main |
| gfx90c/NPU safety hold bypassed by "full qualification" script | Critical | Review hardware access, no unsafe/unauthorized device launches or system tuning |
| Published UI claims all planned models/formats run | High | Capability matrix per source/quant/task/shape/backend/numerical/quality |
| One giant unreviewable commit | High | Single full structural S0 commit, subsequent smallest coherent logical signed commits |

## 5. Deliberately unresolved implementation details

Do not freeze illustrative Rust names if source suggests cleaner pre-v1 implementation:
- exact Runtime/Server/PreparedModel/ExecutionProvider public naming and sync/async method return types;
- whether all new structural top-level modules are physically separate files or grouped files, as long as ownership clear and compiled;
- whether resource identity warrants lower-level dedicated zerogravity-resource crate vs existing core module; default KEEP core to avoid unnecessary new package;
- whether a broad RegionExecutionProvider trait is required vs static prepared enum and narrow common contract; avoid trait zoo;
- how much of fixed-source CUDA dynamic parameterization can generalize without numerical/ABI changes; finite captures remain diagnostic;
- existing pipeline YAML v1/v2 schema changes allowed pre-v1 if old useful validation preserved; no forced backward syntax compat;
- whether serving HTTP uses current httparse-based implementation or a dedicated async HTTP library; reuse existing robustness until measured reason to replace;
- exact CLI flags/config precedence and error presentation; user functionality preserved but legacy spelling not binding;
- how to represent multiple Runtime handles over one process-global physical ResourceArbiter and preserve source/tenant visibility; avoid duplicated budgets;
- which short Tier M suite is inexpensive enough per phase; use measured build cost/availability and hard safety constraints;
- precise real hardware/driver/source access for Q0/Q2; source/user restrictions and previous holds remain;
- whether all proposed task/media/remote typed skeleton modules warrant public exports in S0 or private typed API; structure must remain visible and refuse;
- use of future workflows/provider apps or dynamic plugin ABI; not initial structural deliverable.

Any deviation from mandatory latest user decisions needs explicit agreement. These open choices may be resolved by the successor based on current Rust source and measured maintenance/perf/correctness evidence.

## 6. Phase acceptance law

**S0 Architecture-visible acceptance:**
- Two new crates engine/serving, public SDK and CLI target shape compile with correct Cargo DAG and strict source-only tests.
- All semantic, codec/numerical, session, physical, model/TaskDriver, placement, public API, server and CLI roles visible as real Rust modules.
- Future stubs Unsupported or private todo; no fake model/device/task support.
- Only cheap checks, one signed coherent commit; NOT declaring previous features migrated.

**S3 Structural relocation acceptance:**
- Source exact codec/CPU/GPU and model/forward semantics in correct owners; physical provider/qualification separate; one root/resource protocol.
- Exact source/quant/shape/numeric and known device GPU decoupling preserved; no duplicate source math.
- Each phase signed, source-only tests and regression/refusal checks, long model/driver suite deferred.

**F4 Functional migration acceptance:**
- All currently implemented features in FEATURE_PARITY_MATRIX functioning via engine, public zerogravity SDK, independent serving and thin CLI, or truthfully held at originally unsupported/unsafe scope.
- No old shadow runtime, no permanent compatibility alias tree, no missing CLI-to-lib operation, no supported-path todo panic.
- Focused checks only; ready for Q0.

**Q2 Final transition acceptance:**
- One Q0 broad gate performed, bugfixes per Q1 targeted/tests/commits, one final complete Q2 gate on stable source; safety/quality/physical and actual hardware/driver scopes separately recorded.
- Previously qualified exact model/source/quant/task/backend ranges independently verified or explicitly marked unexecuted/blocked, not retagged from old binary.
- No performance claims absent comparable matched whole-request tests; future E-series capabilities remain clearly separate.

## 7. Planning/documentation control

- New transition side plan is independent of .agents/plans/TODO.md active owner queue until a coordinator explicitly promotes it during authorized implementation.
- Existing older frozen ../fabric-vnext files and full archive remain unchanged; successor must read them as historical architecture/research, with user-requested updated phase ordering here.
- At actual code handoff, verify current source HEAD, AGENTS.md, active TODO plans and concurrent agent ownership. Do not automatically push/tag or include unrelated agent files.
- The 27-Oct planning turn was documentation-only. Current cross-window handoff starts at `.agents/docs/CURRENT_HANDOFF.md`; this directory remains the architecture/transition detail set.
