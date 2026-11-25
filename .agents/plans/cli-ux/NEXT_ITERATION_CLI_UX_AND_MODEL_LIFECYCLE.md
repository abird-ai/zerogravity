# Next implementation iteration — CLI, model lifecycle and terminal UX

**24 Nov delivery order:** the [full implementation plan](../FULL_IMPLEMENTATION_PLAN.md)
sets phase priority. Shared foundations (including optional compiled-provider
scaffolding), CLI/TUI/Web and serving, ROCm, CUDA, Vulkan, then concrete
compiled providers. Native Hugging Face/Safetensors and GGUF are primary.
Older ordering below is context; valid owner constraints remain authoritative.

Status (26 Nov): **Phase 2.1 UX-1 through bounded UX-8C accepted; UX-8D Profiles / Config / Logs next (§46A.28).**
The bounded Phase 1 minimum exit is accepted at `d5030ad4`.
See the [Phase 1 exit takeover](../../docs/FULL_COMPREHENSIVE_HANDOFF_2026-11-25_PHASE1_EXIT.md)
and [live full plan](../FULL_IMPLEMENTATION_PLAN.md). The original design below
remains authoritative for UX scope; reconcile historical prerequisite wording
against live source. See the [CLI checkpoints](../cli-device-routing/TODO.md)
for validated scope. Shared presentation, typed command/global grammar,
static completions, exact cold source/config selection and the shared native
ControlPlane are implemented. Original machine outputs preserved; selected
metadata/dry-run JSON supported. Native snapshots and typed actions reuse
original owners; bounded event cursors resync on lag. Direct SDK/independent
HTTP/external cache activity requires fresh observations. Canonical explicit
single-plan HF/GH/Xet online transfers now share progress and cooperative
pause/resume/cancel, cold partial records, PTY/plain/quiet/JSONL receipt
presentation. Advanced provider forced modes and aggregate component control
remain open. General JSONL events, unified serving/profile management and
frontend surfaces remain ordered work.

This iteration runs before the OpenAI-compatible serving and backend-parity
campaigns. It changes no backend qualification or execution authority.

The objective is to make the entire ZeroGravity product surface feel like one
small, coherent tool even though the underlying engine supports a large amount of
model, cache, device, execution, serving and pipeline complexity.

This document is a UX/product/CLI design. It grants no model, backend, network,
cache, serving or numerical capability by itself.

---

## 1. Product goal

A new user should be able to do the obvious thing in one line:

~~~text
zg run qwen3.5-2b "Explain ownership in Rust"
~~~

or:

~~~text
zg serve qwen3.5-2b
~~~

without learning:

- snapshot digests;
- cache roots;
- source profiles;
- provider/device envelopes;
- sidecar paths;
- by-name projections;
- backend aliases;
- acquisition transports;
- serving worker internals.

An expert must still be able to inspect and control every one of those details.

The UX therefore has two simultaneous goals:

1. **minimal by default**;
2. **transparent on demand**.

The CLI should feel modern, native, calm and highly polished:

- beautiful interactive terminal presentation;
- concise colors and symbols;
- sparing emoji in rich interactive mode;
- responsive progress;
- useful pause/resume/cancel controls;
- strong model-management UX;
- consistent configuration/profile management;
- one-line run and serve;
- deterministic plain output in scripts;
- stable structured output for automation;
- verbose/explain modes for full transparency.

The complexity belongs in the resolver and SDK, not in the user's muscle memory.

---

## 2. Iteration ordering

The intended sequence is:

1. **current live implementation iteration closes**;
2. **this CLI/model-lifecycle/UX iteration**;
3. **full OpenAI-compatible serving iteration**;
4. **backend-parity/execution-performance iteration**;
5. later model/pipeline expansion.

Do not pull physical-execution optimization work forward into this iteration.

This iteration may expose already-qualified capabilities cleanly, but must not
manufacture backend support or use UX changes to weaken current execution gates.

---

## 2A. Immediate prerequisite — production feature/build/run convergence

This prerequisite is **higher priority than the later UX slices** and should be completed immediately.

### Backend capability × model contribution

Cargo features use intersection semantics:

~~~text
cuda + model-qwen35  -> Qwen CUDA contribution
hip  + model-qwen35  -> Qwen HIP contribution
cuda alone           -> CUDA substrate, no Qwen
model-qwen35 alone   -> Qwen model/task/source support, no CUDA/HIP
~~~

Backends are independent system capabilities (`cpu`, `cuda`, `hip`, `vulkan`). Models are independently additive (`model-qwen35`, `model-gemma4`, ...). A model contributes an implementation only when both that model and the matching backend are selected.

Use Cargo weak optional-dependency features where appropriate so a backend can enable a selected optional model's backend contribution without pulling that model into the binary. Existing pair names such as `qwen35-cuda` / `qwen35-hip` may remain compatibility aliases temporarily, but they are not the architecture.

Frontend features are orthogonal too. `serve` means the server/frontend exists; it must not implicitly select CPU or Qwen. Compiled models/backends determine what it can actually serve.

### Production all

`all` means all normal production capabilities: normal models, CPU/CUDA/HIP/Vulkan, cache/acquisition and serving. It deliberately excludes qualification captures, fault injection, hardware-abort probes and diagnostic/evidence features.

Canonical production build:

~~~text
cargo build --locked --release \
  -p zerogravity-cli \
  --no-default-features \
  --features all
~~~

### Mixed-GPU development

Provide one mixed backend shell:

~~~text
nix develop .#gpu
~~~

It owns CUDA runtime headers, CCCL, NVRTC, cuBLAS/nvcc as needed plus HIP/ROCm, hipcc, hipBLAS/hipBLASLt/rocBLAS and runtime library paths. CUDA's unfree Nix policy stays isolated to this optional shell; ordinary CPU package evaluation remains unchanged.

### One production run path

Normal one-shot GPU execution uses the same source-bound `Runtime::models().load(...)` and `Runtime::request_text(...)` path used by SDK/serving. No second GPU executor and no finite qualification loader.

~~~text
zerogravity run-cuda qwen3.5-0.8b --prompt hello
zerogravity run-rocm qwen3.5-0.8b --prompt hello
zerogravity run      qwen3.5-0.8b --prompt hello
~~~

`run-cuda` means CUDA or fail. `run-rocm` means HIP/ROCm or fail. `run` may choose automatically only from the intersection of selected model, compiled backend contribution, observed physical device, resource admission and actually qualified implementation.

Friendly model names resolve through the model contribution to the authenticated immutable source snapshot. Users should not need revision-qualified cache aliases.

Backend compilation is not execution qualification. For example, an SM86-closed CUDA Qwen kernel family must not be advertised on SM89 merely because CUDA compiled; integrated AMD capacity must use authoritative HIP runtime pool facts rather than treating DRM GTT as invented VRAM.

### Immediate acceptance

Before later TUI/Web UX work, prove:

1. backend-only features compile without pulling optional models;
2. model-only features compile without pulling GPU backends;
3. model + backend automatically compiles the model's available contribution;
4. `serve` alone does not pull CPU/Qwen;
5. production `all` builds one CPU + CUDA + HIP + Vulkan binary;
6. `nix develop .#gpu` supplies CUDA and ROCm/HIP together;
7. `run-cuda` and `run-rocm` reach the production Runtime seam;
8. `run` performs qualified automatic selection;
9. a real small Qwen CUDA canary executes through the public CLI;
10. HIP executes where the driver-admitted pool fits, otherwise returns the factual admission refusal;
11. ordinary production `all` requires no qualification/debug feature.

This prerequisite makes the rest of the CLI/TUI/Web iteration a presentation and lifecycle problem rather than another execution-routing migration.

---
## 3. Current surface: what already exists

The existing CLI is not empty. A substantial amount of correct machinery already
exists behind a fragmented presentation layer.

Current top-level commands include:

- \`inspect\`;
- \`run\` plus backend aliases;
- \`devices\`;
- \`pull\`;
- \`models\`;
- \`pipeline\`;
- \`serve\`;
- \`run-cache\`;
- \`cache\`;
- reserved \`config\`.

The current unified run resolver already understands:

- local source paths;
- run YAML documents;
- cached snapshot/reference selectors;
- provider/catalog-style selectors;
- backend selection;
- multiple device selectors;
- headroom;
- context and output limits;
- numerical profile;
- speculation policy;
- draft devices;
- direct model or pipeline entrypoints;
- graph definitions;
- dry-run and explain;
- XDG default config resolution;
- provenance of configuration values.

The acquisition/cache stack already owns:

- Hugging Face;
- GitHub;
- Xet;
- ModelScope;
- OCI;
- S3;
- GCS;
- Azure;
- generic object acquisition;
- immutable snapshot publication;
- explicit source/revision selection;
- partial downloads;
- suffix/range resume where qualified;
- cancellation;
- progress events;
- verification;
- cache aliases/by-name entries;
- exact snapshot reclamation;
- local HF import.

The serving surface already owns:

- typed serving configuration;
- model IDs;
- backend selection;
- device/provider/host budgets;
- active/request limits;
- OpenAI-compatible completions/chat;
- health;
- graceful shutdown.

The primary UX defect is therefore not lack of machinery.

It is that the machinery currently appears to users as several unrelated expert
interfaces.

This iteration should **compose and present**, not replace the underlying
authorities.

---

## 4. Core UX principles

### 4.1 One concept has one normal command

Users think in terms of:

- models;
- running;
- serving;
- devices;
- pipelines;
- profiles;
- configuration.

They should not need to think in terms of:

- cache snapshots;
- pull providers;
- run-cache;
- by-name projections;

for normal operation.

Those remain available as advanced/storage concepts where needed.

### 4.2 Rich for humans, plain for machines

TTY interaction should be delightful.

Non-TTY output should be boring, stable and easy to parse.

The same command semantics must exist underneath both.

### 4.3 No hidden network in automation

Interactive convenience must not turn into nondeterministic scripts.

Example:

~~~text
zg run qwen3.5-2b
~~~

If the model is cached, run it.

If it is missing:

- **interactive TTY:** offer to download it;
- **non-interactive:** refuse with an exact remediation unless the caller passed
  an explicit network/download flag.

A script must never unexpectedly download tens of gigabytes because stdin/stdout
changed.

### 4.4 No silent backend fallback

Existing backend truthfulness remains.

If the user forces HIP, CUDA or Vulkan, failure stays failure.

\`auto\` may select a backend, but rich/verbose/explain output must make the
selection visible.

### 4.5 Progressive disclosure

The normal path should expose approximately five concepts:

- model;
- prompt;
- backend/device if desired;
- profile if desired;
- output length if desired.

Everything else should remain reachable without dominating help or examples.

### 4.6 Primary data stays clean

Generated text is primary output.

Progress, diagnostics and status do not get mixed into stdout unless the selected
machine-output format explicitly asks for event streaming.

### 4.7 Interactive is optional

Every interactive action has a command-line equivalent.

No important operation requires arrow keys or a TTY.

---

## 5. Binary naming

Keep \`zerogravity\` as the canonical product/binary name.

Also install a first-class short alias:

~~~text
zg
~~~

Both names invoke the exact same binary.

Documentation should prefer \`zg\` for interactive examples and use
\`zerogravity\` where an explicit canonical executable name matters.

Do not introduce a second implementation or wrapper script.

---

## 6. Top-level command model

The normal command tree should be:

~~~text
zg
zg run [MODEL] [PROMPT...]
zg serve [MODEL...]
zg model ...
zg device ...
zg pipeline ...
zg profile ...
zg config ...
zg doctor
zg completion ...
zg help [COMMAND...]
zg version
~~~

Compatibility aliases may remain, but main help should not teach legacy
fragmentation.

### Normal nouns are singular

Prefer:

- \`model\`;
- \`device\`;
- \`profile\`;
- \`config\`;
- \`pipeline\`.

This makes the hierarchy predictable.

### Old commands become compatibility aliases

Conceptually:

~~~text
pull        → model pull
models      → legacy/catalog compatibility
inspect     → model inspect for model/artifact targets
run-cache   → run with a cache reference
cache       → advanced storage/repair surface
run-cuda    → run --backend cuda
run-rocm    → run --backend rocm
run-vulkan  → run --backend vulkan
~~~

Do not remove compatibility in the first UX release.

Hide old aliases from primary help once the replacement is complete.

---

## 7. No-argument behavior

### Interactive TTY

Running:

~~~text
zg
~~~

opens the native full-screen ZeroGravity TUI.

The TUI is the **default interactive product surface**. It may use the terminal
alternate screen because it owns the whole interactive session, but exiting it
must restore the caller's terminal cleanly.

The default TUI is the control panel specified in §46A:

~~~text
Home
Running
Serve

Models
Downloads
Devices

Profiles
Config
Logs
~~~

It should open directly into Home, hydrate expensive device/cache/runtime facts
asynchronously, and remain usable immediately rather than displaying a blank
startup screen.

Use symbols/emoji sparingly as semantic accents, not decoration.

Direct subcommands such as `zg run`, `zg serve`, `zg model` and `zg config`
remain direct command surfaces and do not enter the TUI unless explicitly asked.

### Non-TTY

No arguments retain predictable text behavior:

- print concise help;
- no prompt;
- no terminal control sequences;
- success exit is acceptable for compatibility.

---

## 8. Presentation modes

One presentation subsystem owns all command output.

### 8.1 Auto mode

Default:

~~~text
--format auto
--interactive auto
--color auto
--progress auto
~~~

For ordinary direct commands, if stdin/stderr are suitable TTYs:

- interactive prompts allowed;
- rich colors;
- Unicode;
- concise emoji/symbols;
- in-place progress;
- keyboard controls.

For the bare `zg`/`zerogravity` command on a capable TTY, this auto-detection
selects the native full-screen TUI instead of the ordinary direct-command
renderer.

Otherwise:

- plain text;
- no ANSI;
- no cursor rewriting;
- no prompt;
- no emoji;
- no dynamic terminal UI.

### 8.2 Global output flags

All normal commands should understand a shared global surface:

~~~text
-q, --quiet
-v, --verbose
-vv
--format auto|text|json|jsonl
--color auto|always|never
--interactive auto|always|never
--progress auto|always|never
--yes
--offline
~~~

\`-v\` is useful detail.

\`-vv\` is diagnostic detail.

Raw internal tracing remains separately controllable by the existing logging
mechanism/environment; do not turn normal verbose UX into an internal trace dump.

### 8.3 NO_COLOR

Honor the established \`NO_COLOR\` convention.

\`--color always\` may explicitly override it if the user requests that behavior;
otherwise a non-empty \`NO_COLOR\` disables ANSI color. The NO_COLOR convention
exists specifically so users can disable color consistently across CLI tools. citeturn369441search0

### 8.4 Rich mode

Rich mode may use:

- color;
- bold/dim;
- Unicode box/arrow characters;
- semantic symbols;
- small emoji accents;
- spinners/progress bars;
- terminal line replacement.

It should remain calm.

Avoid rainbow output or an emoji on every line.

### 8.5 Plain mode

Plain mode uses stable ASCII-ish text.

Example:

~~~text
model: qwen3.5-2b
status: ready
backend: hip
device: gpu:0
~~~

No ANSI.

No cursor movement.

No hidden lines.

---

## 9. stdout/stderr contract

This must be explicit.

### stdout

Primary command result:

- generated text;
- requested JSON;
- requested JSONL event stream;
- a path/digest where the command's purpose is to return one;
- table/list output when that is the primary result.

### stderr

Human operational state:

- resolution;
- download progress;
- model loading;
- selected backend/device;
- server startup/shutdown;
- warnings;
- hints;
- verbose diagnostics.

This allows:

~~~text
zg run qwen3.5-2b "hello" > answer.txt
~~~

without the answer file containing progress noise.

---

## 10. Structured output

GitHub CLI is a useful precedent: human text by default and explicit JSON for
automation. It also allows further machine filtering instead of making the human
format parseable. citeturn369441search1turn369441search5

ZeroGravity should follow the same principle.

### Final-object JSON

~~~text
zg model list --format json
zg device list --format json
zg model inspect qwen3.5-2b --format json
~~~

stdout contains one JSON value.

Progress is suppressed by default for final-object JSON unless explicitly
requested elsewhere.

### Event JSONL

Long-running operations may expose:

~~~text
--format jsonl
~~~

with typed events:

~~~json
{"event":"resolve.started","model":"qwen3.5-2b"}
{"event":"download.progress","bytes":104857600,"total":2147483648}
{"event":"verify.completed","snapshot":"..."}
{"event":"model.ready","name":"qwen3.5-2b"}
~~~

Do not invent a separate monitoring protocol merely for CLI JSONL.

Serialize the same typed events that drive the human renderer.

---

## 11. Internal presentation architecture

Remove scattered direct presentation responsibility from command modules over
time.

The desired structure is:

~~~text
command / SDK operation
        │
        ▼
typed semantic events + final result
        │
        ├──────────────┬───────────────┐
        ▼              ▼               ▼
 RichRenderer     PlainRenderer    JsonRenderer
~~~

### Event examples

~~~text
ResolveStarted
ResolveCandidate
ResolveCompleted
DeviceSelected
DownloadPlanned
DownloadFileStarted
DownloadProgress
DownloadPaused
DownloadResumed
DownloadCompleted
VerificationStarted
VerificationCompleted
PublicationCompleted
ModelLoadStarted
ModelLoadCompleted
ServerReady
ServerStopped
Warning
Hint
~~~

These events describe product semantics.

Do not expose internal thread/task implementation details.

### Reporter interface

The CLI should own a small reporter abstraction.

SDK/core code should not know about:

- colors;
- emoji;
- terminal width;
- spinners;
- prompts.

The reporter must be cheap enough that commands can emit semantic events without
turning presentation into a performance concern.

---

## 12. Model reference: one resolver everywhere

\`run\`, \`serve\`, \`model inspect\`, \`model pull\`, \`model verify\` and
\`model remove\` should resolve the same model syntax.

Create one typed model-reference parser/resolver rather than command-specific
interpretations.

### User-facing reference classes

#### Catalog/cache name

~~~text
qwen3.5-2b
gemma4-e2b
~~~

#### Exact cache snapshot

~~~text
sha256:<digest>
<full-digest>
~~~

#### Explicit local path

~~~text
./model
/path/to/model
./model.gguf
~~~

#### Explicit provider locator

~~~text
hf:Qwen/Qwen3.5-2B@REV
gh:owner/repo@REV
~~~

Provider-specific advanced selection may remain available.

### Resolution order

Names must never unexpectedly turn into network locations.

A safe conceptual order is:

1. explicit local path syntax;
2. exact snapshot digest;
3. exact cache alias/by-name entry;
4. known local/catalog model slug;
5. explicit provider locator.

If a string is ambiguous, refuse and show explicit alternatives.

### Immutable identity

A resolved remote model must become an immutable revision/snapshot before
publication.

Human aliases remain convenience pointers to immutable cached snapshots.

Do not make aliases runtime source authority.

---

## 13. One-line run UX

The canonical path:

~~~text
zg run qwen3.5-2b "Explain ownership in Rust"
~~~

### Prompt sources

Support, in priority order:

~~~text
zg run MODEL "prompt text"
zg run MODEL --prompt "prompt text"
zg run MODEL --prompt-file prompt.txt
echo "prompt text" | zg run MODEL
~~~

Do not silently concatenate contradictory prompt sources.

### No prompt + TTY

~~~text
zg run qwen3.5-2b
~~~

opens an interactive text session.

### No prompt + piped stdin

Read stdin as the prompt.

### Missing model

If the model resolves to a known remote/catalog source but is not cached:

#### Interactive

~~~text
📦 qwen3.5-2b is not cached
   Qwen/Qwen3.5-2B · BF16 · 3.9 GiB
   source revision: 4ba...

Download and run?  [Y/n]
~~~

#### Non-interactive

Refuse:

~~~text
error[model-not-cached]: qwen3.5-2b is not cached
hint: zg model pull qwen3.5-2b
hint: or rerun with --pull
~~~

### Explicit one-line download + run

~~~text
zg run qwen3.5-2b "hello" --pull
~~~

This is deterministic explicit network intent.

### Backend/device transparency

Normal rich startup may show one compact line:

~~~text
⚡ qwen3.5-2b · HIP · Radeon 8060S · 32K context
~~~

\`--quiet\` removes it.

\`-v\` shows:

- resolved snapshot;
- selected backend/device;
- selected config/profile document;
- memory policy;
- source/numerical profile.

\`--explain\` shows the full resolved decision/provenance before execution.

---

## 14. Interactive run/chat

The interactive session should be intentionally small.

Example:

~~~text
⚡ Qwen3.5 2B
HIP · Radeon 8060S · profile auto

You › Explain ownership in Rust

Assistant › ...
~~~

Useful slash commands:

~~~text
/help
/new
/model
/profile
/stats
/clear
/exit
~~~

Do not turn this into a separate chat application.

The same model/session APIs remain underneath it.

If multi-turn chat state is not yet a qualified product capability, the UI must
not imply it; interactive mode may initially be repeated one-shot requests until
real session-history semantics are implemented.

---

## 15. Model management: normal surface

Normal users should use:

~~~text
zg model list
zg model search [QUERY]
zg model inspect REF
zg model pull REF
zg model verify REF
zg model remove REF
zg model import PATH
zg model alias ...
~~~

### 15.1 model list

Default means **locally usable/cached model snapshots**, not static engineering
intent entries.

Example rich output:

~~~text
📦 Models

NAME             FORMAT   SIZE      STATUS       BACKENDS
qwen3.5-0.8b     BF16     1.6 GiB   ✓ verified   CUDA HIP Vulkan
qwen3.5-2b       BF16     3.9 GiB   ✓ verified   CUDA HIP Vulkan
gemma4-e2b       BF16     5.1 GiB   ✓ verified   HIP Vulkan
qwen3.5-27b      BF16    55.6 GiB   ✓ verified   HIP
~~~

Columns must reflect actual available metadata.

Do not invent "ready" backend claims from a static model family registry.

### Useful filters

~~~text
--cached
--partial
--backend hip
--task text
--format json
~~~

Only add filters backed by real metadata.

### 15.2 model search

Searches the known model catalog first.

Remote provider-wide search is a separate capability and should not be implied
until implemented.

Interactive:

~~~text
zg model pull
~~~

may open the same searchable catalog picker.

### 15.3 model inspect

This replaces the fragmented mental model of:

- top-level local artifact inspect;
- pull inspect;
- models inspect;
- cache/by-name lookup.

The resolver determines the reference type and shows what is actually known.

Default rich card:

~~~text
Qwen3.5 2B                         ✓ verified · cached

Source      Hugging Face · Qwen/Qwen3.5-2B
Revision    4ba787ec…
Snapshot    88606494…
Format      safetensors · BF16
Size        3.9 GiB
Task        text generation
Aliases     qwen3.5-2b
Cache       ~/.cache/abird/zerogravity/models/…

Execution on this host
  HIP       ✓ qualified
  Vulkan    ✓ qualified
  CUDA      unavailable on selected device
~~~

The exact execution-capability wording must come from real model/backend
qualification state, not from file-format inspection.

### Verbose inspect

\`-v\` may add:

- exact digest;
- repository/revision;
- manifest/files;
- tensor count/layout;
- tokenizer/config identities;
- source profile;
- numerical contract;
- native artifact bundle;
- cache charge;
- aliases/references;
- backend refusal reasons;
- estimated/admitted memory where a cold plan can truthfully calculate it.

\`-vv\` may add low-level provenance and paths.

### 15.4 model verify

Verify the immutable cached snapshot and relevant manifest/source identity.

Do not imply remote freshness unless the command explicitly performs remote
resolution.

### 15.5 model remove

Normal model removal is a friendly frontend over exact snapshot reclamation.

Interactive:

~~~text
Remove qwen3.5-0.8b?
  snapshot  d7c7bf56…
  charge    1.63 GiB
  aliases   qwen3.5-0.8b
  shared blobs may remain

[Remove]  [Cancel]
~~~

Referenced/busy models still refuse.

No "force live delete."

Non-interactive destructive use should require explicit intent such as:

~~~text
zg model remove qwen3.5-0.8b --yes
~~~

A dry-run remains available.

### 15.6 model alias

Friendly alias management:

~~~text
zg model alias list
zg model alias set coding qwen3.5-2b
zg model alias remove coding
~~~

Aliases point to immutable snapshot identity.

They remain human convenience metadata, not runtime provenance authority.

---

## 16. Advanced cache surface remains

\`zg cache\` remains available for storage experts and repair:

- exact digest reclamation;
- by-name refresh/reset;
- cache diagnostics;
- low-level partial/reclamation state.

Main help should describe it as **advanced cache maintenance**, not the normal way
to manage models.

Do not remove current exact-cache semantics merely because \`model remove\` is
friendlier.

---

## 17. Model acquisition UX

The normal command is:

~~~text
zg model pull MODEL
~~~

Legacy \`zg pull\` remains a compatibility alias.

### 17.1 Phases

A download is presented as clear semantic phases:

~~~text
resolve → plan → transfer → verify → publish
~~~

Do not show a spinner saying "downloading" while hashing or publishing.

### 17.2 Rich progress

Example:

~~~text
📦 Qwen3.5 2B · 3.9 GiB

Resolving  Qwen/Qwen3.5-2B @ 4ba787ec…                 ✓
Files      3 selected · 1 cached                       ✓
Download   model-00001-of-00002.safetensors
           1.21 / 1.84 GiB  ███████████░░░  66%
           82.4 MiB/s · 7s remaining
Overall    2.48 / 3.91 GiB · cached 612 MiB · resumed 384 MiB

space pause/resume · c cancel · ctrl-c cancel
~~~

Multiple concurrent files may use multiple lines if actual transfer concurrency
exists.

Do not fake concurrency in the UI.

### 17.3 Plain progress

Non-rich stderr:

~~~text
download qwen3.5-2b 2.48/3.91 GiB 63% 82.4 MiB/s cached=612MiB resumed=384MiB
~~~

Rate-limit progress output similarly to the current 200 ms event cadence.

### 17.4 Quiet

\`--quiet\` prints no progress.

On success, the requested command result still appears.

### 17.5 No progress

Follow the useful convention exposed by modern tools such as uv, which offers
explicit progress suppression in addition to quiet/verbose modes. citeturn369441search2

---

## 18. Pause, resume and cancel

The download UX needs real control semantics, not cosmetic key bindings.

### State model

Conceptually:

~~~text
Queued
  ↓
Resolving
  ↓
Transferring ↔ Paused
  ↓
Verifying
  ↓
Publishing
  ↓
Completed

Any cancellable phase → Cancelled
Any phase            → Failed
~~~

### Pause

Pause is cooperative.

It must not suspend an arbitrary thread while it owns an inconsistent mutable
transaction.

At safe bounded transfer points:

- stop scheduling new ranges/chunks;
- finish or safely abandon the current bounded operation;
- retain valid partial state;
- remain resumable.

### Resume

Resume revalidates the same immutable source/revision and existing partial state
before continuing.

### Cancel

Cancel preserves resumable partial bytes by default where the existing cache
contract allows it.

It must preserve the previously durable reference.

An explicit "cancel and discard partial" may be offered only if safe and
separately confirmed.

### Signals

First SIGINT/Ctrl-C requests graceful cancellation.

Use conventional exit 130 for user cancellation when appropriate.

SIGTERM requests cancellation and orderly retirement.

### Process restart

A second invocation of the same immutable pull should naturally reuse verified
partials and report resumed bytes.

A daemon is **not required** merely to support resume.

---

## 19. Async operation architecture

The UI should be event-driven and responsive, but this iteration should not force
an unnecessary async runtime rewrite of hardened acquisition code.

### Shared operation abstraction

Long-running CLI work should expose conceptually:

~~~text
OperationHandle
  pause()
  resume()
  cancel()

OperationEventStream
  phase
  item
  bytes
  total
  cached
  resumed
  rate
  message
~~~

This can initially be backed by:

- current synchronous hardened HTTP/cache operations;
- bounded worker threads;
- channels;
- cooperative pause/cancel state.

The terminal event loop remains responsive even if the current transport is
blocking inside a bounded worker.

If acquisition later becomes native async I/O, the UI contract does not change.

### Do not add Tokio only for appearance

The project already has a small async/futures footprint.

Do not pull a large runtime into core just to make a spinner asynchronous.

Use the simplest executor/channel design that gives:

- responsive terminal input;
- bounded transfer workers;
- cancellation;
- pause/resume;
- progress events;
- predictable cleanup.

### Bounded concurrency

Any parallel download concurrency is a resource policy, not a UI trick.

Respect:

- memory admission;
- open-file limits;
- provider constraints;
- cache transaction ownership;
- network limits.

---

## 20. Serve UX

The canonical path:

~~~text
zg serve qwen3.5-2b
~~~

Multiple models:

~~~text
zg serve qwen3.5-0.8b qwen3.5-2b
~~~

Advanced explicit IDs remain possible:

~~~text
zg serve --model fast=qwen3.5-0.8b --model quality=qwen3.5-2b
~~~

### Missing model

Same resolution/network rules as \`run\`.

Interactive may offer download.

Non-interactive requires explicit \`--pull\`.

### Startup

Rich TTY:

~~~text
🌐 ZeroGravity

Resolving   qwen3.5-2b                                  ✓
Device      HIP · Radeon 8060S · 96 GiB                ✓
Profile     balanced
Loading     3.9 GiB                                    ✓

● Ready
  API       http://127.0.0.1:8000/v1
  model     qwen3.5-2b
  context   32K
  active    8
  backend   HIP

Ctrl-C to stop · -v for detailed startup
~~~

### Running status

Default rich mode should be compact, not a full dashboard.

An updating status line is enough:

~~~text
● 00:42 · requests 128 · active 5/8 · queued 0 · 154 tok/s
~~~

Only show metrics already available cheaply and truthfully.

\`-v\` may display more.

Do not add high-frequency monitoring work that changes serving performance.

### Plain mode

~~~text
serving model=qwen3.5-2b backend=hip listen=127.0.0.1:8000 context=32768 active=8
~~~

### Shutdown

Rich:

~~~text
✓ Server stopped · requests drained · model retired
~~~

Plain:

~~~text
server stopped; handlers and inference workers joined
~~~

### Foreground first

Foreground serving remains the default.

Do not build a daemon/service manager just because the interactive UI is modern.

A future \`--detach\` requires its own explicit process lifecycle design.

### OpenAI-compatible serving boundary

The CLI iteration owns the **user-facing serve interface**, not the full protocol
rewrite.

It must leave behind a stable:

~~~text
zg serve MODEL...
~~~

surface with:

- the same ModelRef resolver as run/model management;
- the same ConfigDocument/profile selection model;
- backend/device resolution;
- missing-model interactive pull / explicit non-interactive --pull;
- rich/plain startup and shutdown presentation;
- stable serve configuration;
- clean handoff into the serving crate.

The current server is only partially OpenAI-compatible: it implements models,
legacy Completions and Chat Completions with JSON/SSE, but Responses, Embeddings
and Realtime are explicitly unsupported and the bootstrap HTTP transport is
intentionally narrow.

Do **not** expand that protocol surface inside this CLI iteration.

The immediately following dormant iteration is:

~~~text
../serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md
~~~

That iteration owns:

- mature async HTTP/WebSocket transport;
- the shared semantic request/event seam;
- Responses as the primary text/multimodal API;
- full Chat/Completions compatibility adapters;
- response/conversation protocol state;
- Embeddings;
- Audio;
- Images;
- Realtime WebSocket;
- OpenAI SDK conformance.

The CLI design should not introduce choices that make that server iteration
awkward, but it should not absorb the protocol implementation itself.

---

## 21. Devices UX

Normal surface:

~~~text
zg device list
zg device inspect DEVICE
~~~

Compatibility:

~~~text
zg devices list
~~~

### Default rich list

~~~text
◈ Devices

GPU 0  AMD Radeon 8060S      96.0 GiB  95.7 GiB avail   HIP ✓  Vulkan ✓
GPU 1  NVIDIA RTX 3060        6.0 GiB   5.9 GiB avail   CUDA ✓ Vulkan ✓
CPU    AMD Ryzen ...                                  CPU ✓
~~~

"Available" remains observation, not execution admission.

### Inspect

Show:

- stable selector IDs;
- PCI/node identity;
- vendor/class;
- total/observed available memory;
- compiled backends;
- runtime availability;
- exclusions;
- model-specific eligibility only when a model is supplied.

Example:

~~~text
zg device inspect gpu:0 --model qwen3.5-27b
~~~

may show the cold admission proposal/refusal without loading the model.

---

## 22. Profiles

A profile is simply a **named ZeroGravity configuration document**.

There is no separate profile schema, profile database, selected-profile state or
inheritance system.

The default profile is the normal user configuration document:

~~~text
~/.config/abird/zerogravity/config.yaml
~~~

Named profiles use the exact same schema:

~~~text
~/.config/abird/zerogravity/profiles/laptop.yaml
~/.config/abird/zerogravity/profiles/large-context.yaml
~~~

They are not model identities and they are not model source profiles.

Examples:

~~~text
zg profile list
zg profile show laptop
zg profile create laptop
zg profile copy laptop travel
zg profile edit laptop
zg profile remove travel

zg --profile laptop run qwen3.5-2b
zg --profile laptop serve qwen3.5-2b
~~~

### One schema, one loader

config.yaml, every profiles/<name>.yaml file and an arbitrary document selected
with --config FILE all deserialize to the same typed ConfigDocument.

A document may contain both experience preferences and execution policy:

~~~yaml
ui:
  color: auto
  interactive: auto
  progress: auto
  symbols: auto

downloads:
  ask_before_pull: true
  concurrency: auto

backend: auto
device: auto
headroom: 10%

run:
  context: 32768
  max_tokens: 512

serve:
  active: 8
  max_pending: 128
~~~

A named profile can describe the same complete experience:

~~~yaml
# profiles/laptop.yaml
ui:
  color: auto
  progress: auto

downloads:
  concurrency: 2

backend: cuda
device: gpu:0
headroom: 20%

run:
  context: 8192
  max_tokens: 256

serve:
  active: 4
  max_pending: 32
~~~

Do not create a separate settings file merely because some fields are UI
preferences.

### Exactly one selected config document

Keep resolution deliberately simple.

For one invocation the base document is exactly one of:

1. explicit --config FILE;
2. explicit --profile NAME, resolving only to profiles/<name>.yaml;
3. implicit config.yaml;
4. built-in defaults when the implicit config.yaml does not exist.

--config and --profile are mutually exclusive.

A selected named profile **replaces config.yaml** for that invocation. It is not
layered on top of config.yaml.

Missing fields inherit only built-in defaults.

Then explicit command-line arguments override the selected document.

The complete precedence is:

~~~text
built-in defaults
       ↓
one selected config document
       ↓
CLI overrides
~~~

There is no:

- profile inheritance;
- profile stacking;
- persistent selected-profile pointer;
- hidden merge with config.yaml;
- second UI/settings overlay.

### Profile names are identifiers

--profile laptop is not a path lookup.

It resolves only to:

~~~text
profiles/laptop.yaml
~~~

Use a strict bounded identifier grammar such as lowercase letters/digits plus
._-, with no slash, dot-dot component or path interpretation.

Arbitrary config documents use --config FILE.

### Interactive profile manager

zg profile with a TTY may open:

~~~text
◇ Profiles

  default        config.yaml
  laptop         profiles/laptop.yaml
  large-context  profiles/large-context.yaml
  low-memory     profiles/low-memory.yaml

enter view · n new · c copy · e edit · d delete
~~~

The default entry is always config.yaml.

There is no separate "make default" action because config.yaml already *is* the
default profile.

---

## 23. Config management

config manages the default ZeroGravity configuration document: config.yaml.

It contains normal UX preferences, download behavior and execution defaults in
one typed document.

Normal commands:

~~~text
zg config
zg config show
zg config path
zg config get ui.color
zg config set ui.color auto
zg config unset ui.color
zg config validate [FILE]
zg config edit
zg config explain
zg config init
~~~

### config show

Shows config.yaml.

If it does not exist, show the effective built-in defaults clearly as defaults;
do not silently write a file.

### config get/set/unset

Operate on typed schema paths in config.yaml.

Mutations are:

- schema validated;
- written to a temporary file;
- atomically replaced;
- refused for unknown keys or invalid values.

This is the settings manager too. There is no second key/value settings database.

### config explain

Shows the resolved execution decision plus provenance.

Example:

~~~text
backend       hip             profiles/laptop.yaml
device        gpu:0           auto selector
context       32768           profiles/laptop.yaml
max_tokens    512             built-in default
headroom      10%             CLI
color         auto            profiles/laptop.yaml
model         qwen3.5-2b      CLI
~~~

The existing run resolver already tracks provenance; extend and expose that
instead of creating another provenance system.

### config edit

Open config.yaml in the configured editor.

Write changes to a temporary file, validate the full typed document and atomically
replace the old file only after validation succeeds.

An invalid edit must not replace the current config.

zg profile edit NAME uses the exact same editor, parser, validator and atomic
write path for profiles/<name>.yaml.

### config init

Generate a minimal commented config.yaml, not a giant exhaustive file.

---

## 24. Unified configuration document

Configuration and former "settings" are one concept.

The schema may include sections such as:

~~~yaml
ui:
  color: auto
  interactive: auto
  progress: auto
  symbols: auto

downloads:
  ask_before_pull: true
  concurrency: auto

backend: auto
device: auto
headroom: 10%

run:
  context: 32768
  max_tokens: 512

serve:
  active: 8
  max_pending: 128
~~~

The exact schema should stay compact and typed. Add fields only when they control
real product behavior.

A named profile may contain any of these same fields. That is useful when an
execution environment also wants different download concurrency, verbosity or
interactive defaults.

Do not force every profile to repeat everything. Omitted fields use built-in
defaults.

Do not put into config.yaml, profiles or project config documents:

- API tokens;
- HF tokens;
- GitHub tokens;
- signed URLs;
- source secrets.

Credentials remain environment/credential-provider concerns.

A portable/project document selected with --config ./project.yaml uses the same
schema. It may simply omit UI/download fields when they are irrelevant.

The important rule is **one configuration language and one ConfigDocument type**,
not one mandatory physical file.

---

## 25. Configuration file layout

Keep the filesystem understandable:

~~~text
~/.config/abird/zerogravity/
  config.yaml
  profiles/
    laptop.yaml
    large-context.yaml
    low-memory.yaml
~~~

config.yaml is the implicit default profile.

Every file below profiles/ has the same schema as config.yaml.

There is no settings.yaml.

Do not put cache state, downloaded model state, credentials or runtime receipts
in the config directory.

---

## 26. Pipeline UX

Keep the existing typed pipeline compiler.

Normal surface:

~~~text
zg pipeline list
zg pipeline inspect NAME
zg pipeline check FILE
zg pipeline run NAME|FILE
~~~

Until actual pipeline execution exists, \`run\` continues to refuse truthfully.

### Interactive pipeline view

For a checked pipeline:

~~~text
⛓ voice-assistant

audio
  ↓
ASR  canary-qwen
  ├──→ diarization
  ↓
LLM  qwen3.5-2b
  ↓
TTS  qwen3-tts
  ↓
audio

Models      4 instances
Routes      /v1/realtime · websocket
Context     enabled · 4 KiB max
Status      structure valid · execution not yet qualified
~~~

Do not imply runnable media stages from structural validity.

---

## 27. Interactive model manager

\`zg model\` on a TTY opens a model manager.

It should remain lightweight and searchable.

Example:

~~~text
📦 Models                                   / search

Cached
  ✓ qwen3.5-0.8b       1.6 GiB   HIP CUDA Vulkan
  ✓ qwen3.5-2b         3.9 GiB   HIP CUDA Vulkan
  ✓ qwen3.5-27b       55.6 GiB   HIP

Available
  ↓ gemma4-e2b
  ↓ qwen3-tts-1.7b

Partial
  ◐ qwen-image         4.2 / 12.8 GiB

enter inspect · r run · s serve · d download · x remove
~~~

Only show Available entries backed by the real catalog.

No fake internet search.

---

## 28. Interactive acquisition flow

Running \`zg model pull\` with no model and a TTY opens a catalog picker.

Flow:

1. choose/search model;
2. show source/revision/size;
3. show cache space impact;
4. choose relevant supported variant if required;
5. confirm;
6. download with live controls;
7. verify;
8. publish alias;
9. offer Run / Serve / Done.

Do not ask unnecessary questions when the catalog already has one unambiguous
supported artifact set.

---

## 29. Download partials

Partials should be visible but not noisy.

\`zg model list --partial\` shows interrupted/resumable transfers.

Interactive model manager groups them under Partial.

A normal repeated pull resumes automatically if immutable identity and partial
validation match.

Do not automatically delete partials on cancel.

Advanced cleanup belongs under \`zg cache\` or an explicit model-partial remove
operation.

---

## 30. Errors and hints

Errors should be typed and rendered consistently.

Rich example:

~~~text
✖ Model is not cached

  qwen3.5-2b

  Run:
    zg model pull qwen3.5-2b

  or:
    zg run qwen3.5-2b --pull
~~~

Plain:

~~~text
error[model-not-cached]: qwen3.5-2b is not cached
hint: zg model pull qwen3.5-2b
~~~

### Error structure

Conceptually:

~~~text
code
summary
detail?
context?
hints[]
causes[]
~~~

Normal mode shows summary + the most useful hint.

\`-v\` adds context.

\`-vv\` adds cause chain and low-level detail.

### Never echo secrets

Preserve the current acquisition design that refuses to echo unknown raw argv,
because a mistyped token must not enter logs.

A future typed parser must not regress this behavior.

---

## 31. Exit codes

Keep exit codes small and stable.

Suggested broad categories:

~~~text
0    success
2    invalid usage/configuration
3    not found / unresolved model
4    unsupported capability
5    resource/admission failure
6    network/acquisition failure
7    verification/integrity failure
130  user cancellation
~~~

Do not create dozens of exit codes.

Machine callers can use structured error codes for finer distinctions.

Migration from the current mostly-\`2\` behavior should be deliberate and
documented.

---

## 32. Help UX

Top-level help should fit comfortably on one terminal screen.

Example:

~~~text
ZeroGravity — AI execution fabric

Usage:
  zg <command> [options]

Commands:
  run         Run or chat with a model
  serve       Start an OpenAI-compatible API
  model       Download, inspect and manage models
  pipeline    Check and run pipelines
  device      Inspect available hardware
  profile     Manage named configuration profiles
  config      Manage default configuration and preferences
  doctor      Diagnose the local installation

Run 'zg help <command>' for details.
~~~

Do not print the entire compiled feature matrix in normal help.

### Detailed help

\`zg help run\` gives normal options and examples.

Advanced engine/developer options appear under an "Advanced" section or an
explicit extended-help mode.

### Completions

Provide generated completions:

~~~text
zg completion bash
zg completion zsh
zg completion fish
zg completion powershell
~~~

Model/profile names may be dynamically completed only if this can be done quickly
and without network access.

---

## 33. Parser architecture

The current CLI uses several hand-written parsers.

For the coherent command tree, a typed command grammar is preferable.

\`clap\`/derive is a reasonable CLI-layer choice because:

- help is generated from the actual grammar;
- aliases become explicit;
- global flags are consistent;
- shell completion becomes straightforward;
- subcommands stop duplicating parsing behavior.

This is CLI code, not a token hot path; parser micro-performance is irrelevant.

### Important security caveat

The acquisition parser currently deliberately does not echo arbitrary unknown
argv because it may contain a secret.

If using a parser library:

- capture parser errors;
- render through the ZeroGravity error system;
- do not blindly print library-generated raw argv diagnostics for acquisition
  commands;
- credentials remain environment-only;
- sensitive values are redacted.

If preserving the hand parser is simpler for a sensitive acquisition subgrammar,
do so.

The goal is coherent UX, not ideological parser uniformity.

---

## 34. Terminal implementation stack

Prefer a small inline terminal stack.

A likely implementation shape:

- typed command parser;
- ANSI/style layer with TTY detection;
- inline progress/multi-progress;
- small interactive selection/input library or minimal crossterm use;
- existing signal handling;
- event channels from background operations.

Avoid a full \`ratatui\` alternate-screen application unless a real later UX
requires it.

The initial UI should look good in normal shell scrollback.

That keeps:

- logs readable;
- copy/paste useful;
- SSH behavior simple;
- terminal bugs low;
- implementation maintenance small.

---

## 35. Color and symbol grammar

Use a restrained semantic palette.

Conceptually:

- success: green;
- active/running: cyan/blue;
- warning: yellow;
- error: red;
- muted metadata: dim;
- model/source names: bold/default;
- identifiers/digests: dim/monospace terminal default.

Symbols:

~~~text
✓ success
✖ error
! warning
● running
◐ partial/in progress
↓ available/download
▶ run
◈ device
◇ profile
⚙ config
~~~

Emoji may appear in rich menus/cards:

~~~text
⚡ 📦 🌐
~~~

but never become required to understand status.

Plain mode replaces all symbols with words.

---

## 36. Verbosity design

### Quiet

~~~text
-q
~~~

Only primary result and errors.

### Normal

Concise human state.

### Verbose

~~~text
-v
~~~

Adds:

- resolved source;
- exact snapshot short digest;
- backend/device;
- config/profile;
- resource plan;
- transfer source/revision;
- verification summaries.

### Diagnostic

~~~text
-vv
~~~

Adds:

- full digests;
- paths;
- provenance;
- detailed refusal reasons;
- selected execution/profile internals;
- timing phases.

Internal developer tracing remains separate.

Do not make \`-vv\` dump secrets or arbitrary HTTP headers.

---

## 37. run/serve explain and dry-run

Unify \`--explain\` and \`--dry-run\` behavior across execution commands.

### dry-run

Resolve and validate without:

- model load;
- GPU allocation;
- network unless explicitly required/allowed by the command;
- serving bind.

Return the proposed typed plan.

### explain

Show why the final choices were made:

- model reference resolution;
- config provenance;
- profile;
- backend/device selection;
- resource admission policy;
- state policy;
- cohort/batching policy;
- reasons alternatives were rejected.

Rich human presentation by default.

JSON available.

The existing run decision/provenance machinery should be reused.

---

## 38. Doctor

Add:

~~~text
zg doctor
~~~

This is not a giant qualification suite.

It is a fast local UX diagnostic:

- config/profile document parse;
- cache root accessibility;
- device discovery;
- compiled backends;
- backend runtime libraries present;
- cache usage;
- obvious source/runtime path issues.

Example:

~~~text
✓ config        valid
✓ cache         64.1 GiB / 256 GiB
✓ HIP           ROCm 7.2.4 · Radeon 8060S
✓ Vulkan        RADV · Radeon 8060S
- CUDA          no NVIDIA device
! profile       'laptop' selects missing device gpu:1
~~~

Do not run real models unless the user explicitly requests a deeper check.

---

## 39. Config/profile transparency

The user should never wonder "where did this value come from?"

Every resolved field should have provenance.

Use the existing provenance model and extend it to:

- selected config/profile document;
- command line;
- built-in default;
- auto selection.

Interactive forms may show the source in dim text.

---

## 40. Backward-compatible migration

The first UX release should preserve existing commands.

### Compatibility aliases

Examples:

~~~text
zerogravity run-cuda ...       → run --backend cuda
zerogravity run-rocm ...       → run --backend rocm
zerogravity pull ...           → model pull ...
zerogravity run-cache ...      → run CACHE_REF ...
zerogravity inspect PATH       → model inspect PATH
zerogravity devices list       → device list
~~~

### Deprecation behavior

Do not spam deprecation warnings on every invocation initially.

Main help/documentation should move to the new surface first.

A later release may add one concise warning for commands intended for removal.

### No semantic migration without authority

Legacy \`models list\` currently means an offline static target matrix, not cached
models.

Do not silently change that command's meaning if scripts may depend on it.

Instead:

- new \`model list\` means local model management;
- legacy \`models\` may remain hidden/compatible;
- catalog/static intents become \`model search\` / \`model catalog\` when the
  replacement is complete.

---

## 41. Security and trust boundaries

UX convenience may not weaken existing trust.

### Remote model acquisition

Before transfer, the resolver should make clear:

- provider;
- repository;
- exact revision when resolved;
- selected files/variant;
- expected/maximum bytes where known.

### Credentials

Do not display tokens.

Do not store tokens in config/profile documents.

### Local cache

Aliases are convenience metadata.

Snapshot digest/provenance remains canonical.

### Automatic pull

Interactive auto-offer is a UI prompt, not implicit SDK network behavior.

Non-interactive network remains explicit.

### Destructive operations

Model/cache deletion revalidates liveness/reference constraints.

No force-delete of active model owners.

---

## 42. Performance characteristics of the CLI

The CLI itself should feel instant.

Targets:

- help/version: no model/cache/device initialization;
- config/profile list: filesystem-only;
- model list: local cache metadata only;
- device list: device discovery only;
- model inspect cached: no model load;
- interactive home: no heavyweight model load.

Do not initialize CUDA/HIP/Vulkan merely to draw the home screen.

Lazy-load expensive information only when selected.

---

## 43. Model metadata index

For a good \`model list\` UX, do not repeatedly reopen tens of gigabytes of
artifacts.

Use canonical cache manifest/provenance/by-name metadata where sufficient.

If additional human-friendly metadata is required, store a small derived index
that can always be rebuilt from canonical snapshots.

It must not become:

- execution authority;
- source authentication;
- reclamation authority.

A stale derived index should degrade presentation, not correctness.

---

## 44. UX operation lifecycle

Long operations should use a shared operation lifecycle.

Examples:

- pull/import;
- verify;
- model load;
- server startup;
- future compilation/native-artifact preparation.

Conceptually:

~~~text
Operation
  id
  label
  phase
  progress
  cancellable
  pausable
  final_result
~~~

Do not create a generic distributed job system.

This is a local CLI orchestration primitive.

---

## 45. Initial implementation slices

This iteration should itself be implemented in coherent slices.

### UX-1 — shared presentation foundation

Implement:

- terminal capability detection;
- rich/plain mode;
- color policy + NO_COLOR;
- quiet/verbose;
- typed reporter/events;
- unified errors/hints;
- stdout/stderr contract.

No command-tree redesign yet.

Close with existing commands rendered through the new layer where practical.

### UX-2 — typed top-level command tree

Implement:

- canonical command hierarchy;
- global options;
- concise help;
- compatibility aliases;
- completion generation.

Do not change underlying model/cache semantics.

### UX-3 — unified model resolver and model inspect/list

Implement:

- one ModelRef;
- cached model list;
- unified inspect;
- exact aliases/digests/local paths;
- structured output.

Keep remote pull explicit.

### UX-4 — model pull + rich progress/control

Implement:

- \`model pull\`;
- progress events;
- interactive progress;
- pause/resume/cancel;
- resumed-partial reporting;
- quiet/plain/JSONL modes.

Use existing hardened acquisition/cache transaction code.

Accepted 25 Nov for canonical explicit single-plan HF/GH/Xet. Original
strong-validator proper-prefix checkpoint/range continuation; other bodies
park at safe boundaries. Cold partial metadata is not authenticated or
necessarily inactive. Exact operation IDs and bounded invalidation events;
actual durable/uncertain receipts. Direct provider progress shared, advanced
forced modes and aggregate component controls open. Full gate 2459 pass,
0 fail, 109 skips; separate SDK acquisition and five monitor fixture runs pass.

### UX-5 — one-line run

Implement:

- cached model refs;
- positional prompt;
- stdin;
- interactive no-prompt session;
- missing-model interactive offer;
- explicit \`--pull\`;
- model/backend/device summary.

Retire the need for normal \`run-cache\`.

Accepted 25 Nov: exact durable name selection with conflict refusal and
approved original Qwen default only if absent; one quoted positional
prompt, prompt file, config prompt or bounded stdin; independent interactive
requests reuse one prepared owner, not persistent chat history. Canonical
run does not need legacy run-cache for supported cached GGUF. --pull and an
affirmative eligible terminal offer use exact 2B/27B source recipes and the
original download operation; require the final snapshot digest before commit.
Legacy 0.8B manifest recreation, arbitrary unregistered/floating sources and
unsupported routes refuse; no live remote acquisition or new backend
qualification. Full isolated gate 2465 pass, 0 fail, 109 skips; release,
installer, scripts, final combined lint/feature compilation and exact-recipe
fixture pass. Real Qwen 0.8B GGUF CPU release PTY two-prompt equality and
clean exit pass with no cache creation.

### UX-6 — serve UX and stable interface

Implement:

- \`serve MODEL...\`;
- same ModelRef resolver as run/model management;
- same ConfigDocument/profile/device resolution;
- missing-model interactive offer / explicit \`--pull\`;
- stable serve configuration passed into the serving crate;
- rich startup;
- compact live status;
- clean shutdown presentation.

Do not rewrite the HTTP protocol in this slice.

UX-6 implementation (25 Nov): positional models, explicit IDs, shared selected
document/profile and typed serve section; CLI overrides first and explicit
route lists replace document model table. CPU GGUF and one approved original
GPU snapshot shared by aliases; other route sets refuse, legacy advanced
serving retained. Cold route-set selection before eligible original acquisition;
same operation and exact commit guard, shared shutdown flag. Original
ControlPlane listener, actual startup and coarse changed-counter status,
listener retirement before runtime join. Focused real cached 0.8B GGUF
release alias equality/one owner/SIGTERM and cold refusal pass; combined
lint/feature compile pass. Full isolated gate 2466 pass/0 fail/109 skips,
release/installer/scripts pass. No new protocol, physical GPU
qualification or live remote transfer. UX-7 follows before native TUI/Web.

### UX-7 — unified config/profile documents

Implement:

- config.yaml as the implicit default profile;
- profiles/<name>.yaml using the identical schema;
- mutually exclusive --profile/--config selection;
- config get/set/unset/edit and profile create/copy/edit/remove;
- provenance/explain;
- interactive config/profile managers;
- one parser/validator and atomic typed edits.

Do not add profile inheritance.

Implemented 25 Nov: extend existing typed v2 pipeline Document (ConfigDocument
alias); keep run.task.context etc. as actual schema paths, no parallel
flattened settings schema. One implicit/default/named/explicit file, built-in
fallback only for absent implicit default, CLI overrides. Model/device UI
preferences and run/serve execution settings share selected file. Typed UI
color/progress/interactive and download ask_before_pull are real settings;
false offer is not consent, no unused concurrency/symbol controls. SDK
ConfigFile validates complete candidate with original parser/model/graph/
policy and serving bounds, compares original bytes, cooperatively locks
short commit, atomically publishes. Raw editor draft, typed mapping edits,
profile copy/create/remove, JSON read/explain; no inherited profiles or
secret fields. Typed edits normalize YAML. Same-base copy retains bytes,
cross-base copy anchors relative paths. Basic line managers share actions;
rich TUI UX-8D remains. Focused SDK/CLI/actual PTY and combined strict lint/
feature compile pass. Corrected complete isolated gate 2473 pass/0 fail/
109 skips; release/installer/scripts pass. Serial combined release
run_unified 6/0/0, including real cached Qwen 0.8B GGUF serving at the
unchanged 8 GiB admission bound. Earlier invalid-source error-category
regression corrected and build-pressure refusal resolved by serial execution.
UX-7 accepted; UX-8A next.

### UX-8 — interactive home/model manager

Implement:

- no-argument rich launcher;
- model manager;
- model picker;
- run/serve transitions;
- config/profile entrypoints.

Keep it inline rather than a full-screen application.

### UX-9 — pipeline/device/doctor polish

Bring:

- device;
- pipeline;
- doctor;

onto the same presentation/event system.

### UX-10 — close compatibility surface

Update normal help/docs.

Keep hidden compatibility aliases.

Remove duplicated normal paths only when their replacements are proven.

---

## 46. Minimal validation policy

This iteration must not create another huge CLI fixture suite.

### Parser/presentation

Use focused tests for:

- command grammar;
- TTY/plain mode selection;
- no-color;
- stdout/stderr routing;
- structured JSON;
- secret redaction.

### Interactive

Test state machines and key actions as pure logic where possible.

Do not snapshot thousands of ANSI escape sequences.

One or two bounded renderer snapshots are enough for accidental regressions.

### Acquisition

Reuse current acquisition/cache correctness tests.

Add only:

- pause/resume control;
- cancellation presentation;
- resumed progress;
- TTY/non-TTY behavior.

Do not duplicate the acquisition protocol matrix.

### Run/serve

Reuse existing real-model and serving tests.

Add only the new resolver/CLI path needed to prove the same backend was reached.

---

## 46A. Default TUI and SolidJS web control plane

The normal ZeroGravity product has **two first-class interactive clients** over
one shared control-plane model:

1. native terminal TUI;
2. embedded SolidJS web interface.

Neither client is runtime authority.

Both render and invoke the same typed ZeroGravity control state/actions/events.

This intentionally follows the strongest AW V1.5 architecture lesson:

> presentation state points at existing runtime owners and exact capabilities; a
> view never becomes a second runtime, scheduler, model owner or permission
> engine.

### 46A.1 Default interactive behavior

On a capable TTY:

~~~text
zerogravity
zg
~~~

opens the native TUI.

The TUI is the default interactive ZeroGravity experience.

Explicit commands remain direct:

~~~text
zg run ...
zg serve ...
zg model ...
zg profile ...
zg config ...
~~~

They do not enter the TUI unless explicitly requested.

On non-TTY stdin/stdout the bare command prints concise normal help rather than
emitting terminal control codes.

### 46A.2 Web entrypoint

Provide an explicit local web control-plane command:

~~~text
zg web
~~~

It starts the ZeroGravity control host on loopback and serves the embedded
SolidJS application.

Useful options may include:

~~~text
zg web --listen 127.0.0.1:7860
zg web --no-open
zg web --profile workstation
~~~

The default may open the browser when an interactive graphical environment is
available.

Do not call this \`serve\`; \`zg serve\` remains the OpenAI-compatible inference
API.

The TUI may expose an **Open Web UI** action which starts the same local control
surface in the current process and opens it in the browser.

### 46A.3 One control-plane model

Introduce one small reusable Rust control layer below both frontends.

Conceptually:

~~~text
                       ZeroGravity Runtime
                               │
                 model/cache/device/config owners
                               │
                               ▼
                         ControlPlane
                   ┌───────────┼───────────┐
                   │           │           │
             snapshots      actions      events
                   │           │           │
             ┌─────┴─────┐     │     ┌─────┴─────┐
             ▼           ▼     ▼     ▼           ▼
         Native TUI   SolidJS Web   CLI      JSON/API
~~~

The exact crate/module boundary should stay small. A dedicated
\`zerogravity-control\` crate is appropriate if it prevents the CLI/TUI/web from
duplicating model lifecycle and configuration orchestration; otherwise keep the
same typed module in the public facade.

The control model owns **orchestration**, not inference semantics.

It may expose typed concepts such as:

~~~text
ControlSnapshot
ControlAction
ControlEvent

ModelSummary
ModelInstance
DownloadSummary
DeviceSummary
ServeSummary
ProfileSummary
ConfigSummary
LogEvent
~~~

Do not create frontend-specific model managers.

### 46A.4 Control actions

Both TUI and Web invoke the same actions:

- run/invoke a model;
- load/unload a model;
- start/stop OpenAI serving;
- pull/pause/resume/cancel a model download;
- verify/remove a model;
- inspect model/device/runtime details;
- create/copy/edit/remove named profiles;
- edit the default config document;
- validate config/profile documents;
- select/view logs;
- open diagnostics/doctor;
- start the web UI from the TUI.

Every effectful action goes through the same typed validation and lifecycle
authority used by the direct CLI.

No UI gets a special "trusted" execution path.

### 46A.5 Events and snapshots

Use two complementary concepts.

**Snapshot** gives bounded current state needed to render a screen.

**Events** provide incremental progress/activity:

~~~text
ModelLoaded
ModelUnloaded
RequestStarted
RequestCompleted
ServeStarted
ServeStopped
DownloadProgress
DownloadPaused
DownloadCompleted
ConfigChanged
DeviceChanged
LogEvent
~~~

Events do not replace authoritative state. On reconnect or missed events, the
client reloads a fresh snapshot.

This keeps the browser resilient without requiring event-history replay as a
runtime authority.

### 46A.6 Client-local presentation state

Following AW V1.5, presentation state stays local to each client:

- selected navigation item;
- current tab/view;
- search text;
- scroll position;
- expanded rows;
- split ratios;
- maximized pane;
- table sorting/filtering;
- transient dialogs;
- command-palette state.

Closing a tab or pane never unloads a model, stops the OpenAI server or cancels a
download unless the user invokes an explicit effectful action.

Selection and execution ownership are separate notions.

### 46A.7 One visual language across TUI and Web

TUI and Web should clearly feel like the same product.

Share:

- information architecture;
- nouns;
- ordering;
- semantic colors;
- status vocabulary;
- icons/symbol intent;
- command names;
- detail-card structure;
- keyboard concepts;
- command-palette destinations.

Do **not** chase pixel identity.

The TUI is cell-native.

The Web interface is spatially richer and can use typography, whitespace,
hover/focus, resizing and browser navigation naturally.

### 46A.8 Visual thesis

Borrow AW V1.5's strongest principles:

- one title owner;
- no redundant nested headers;
- one main workspace;
- contextual secondary pane only when it materially helps;
- panes may maximize and restore without changing runtime state;
- narrow viewports collapse rather than lose state;
- selection is not an effect;
- one command palette rather than several search UIs;
- restrained status colors with text/shape fallbacks;
- exact resource identity beneath friendly names.

ZeroGravity should be even more minimal than AW because its primary objects are
models, devices, server state and configuration rather than an IDE workspace.

The visual target is:

> quiet instrument panel, not observability dashboard.

### 46A.9 Native TUI information architecture

The default navigation is:

~~~text
Home
Running
Serve

Models
Downloads
Devices

Profiles
Config
Logs
~~~

Optional/contextual destinations such as Doctor or Pipelines may be reached from
the command palette and appear in the rail when relevant.

#### Home

Home answers only:

- is ZeroGravity healthy?;
- what is running?;
- what can I do immediately?;
- is anything asking for attention?

It contains:

- one quick-run prompt;
- loaded-model summary;
- server state;
- active/queued request summary;
- download/attention summary;
- selected-model detail only when useful.

No giant telemetry grid.

#### Running

Shows actual loaded model instances.

Selection exposes:

- backend/device;
- memory;
- context/state policy;
- active/queued work;
- throughput;
- load time;
- explicit Run / Inspect / Logs / Unload actions.

#### Serve

Shows the OpenAI-compatible server:

- endpoint/base URL;
- running/stopped state;
- served model IDs;
- active/queued requests;
- compact traffic metrics;
- Start/Stop;
- Copy URL;
- Open logs;
- Open Web UI.

Detailed OpenAI endpoint/capability compatibility belongs in a drill-down, not
the default status screen.

#### Models

Model lifecycle manager:

- Cached;
- Available/catalog;
- Partial.

Actions:

- Run;
- Serve;
- Pull;
- Resume;
- Verify;
- Inspect;
- Alias;
- Remove.

#### Downloads

Live transfer state with real:

- rate;
- bytes;
- ETA;
- cached/resumed bytes;
- phase;
- pause;
- resume;
- cancel.

#### Devices

Observed hardware and backend availability.

Model-specific execution eligibility appears only after a model is selected.

#### Profiles

Rows are simply:

~~~text
default        config.yaml
laptop         profiles/laptop.yaml
workstation    profiles/workstation.yaml
~~~

Open/edit/copy/delete actions share the same typed ConfigDocument editor.

#### Config

Inspired by Whisrs settings UX:

- searchable row list;
- grouped semantic sections;
- inline current values;
- Enter edits the selected value;
- contextual description/help;
- optional raw YAML view;
- validation before write;
- no separate settings file.

#### Logs

One searchable/filterable operational stream.

Do not mix model output with logs.

### 46A.10 TUI interaction grammar

Default keys:

~~~text
↑↓ / j k     move
enter        open / activate selected row
esc          back / close transient view
/            search/filter current surface
:            command palette
?            contextual help
1..9         jump common section where unambiguous
tab          move between panes
shift-tab    reverse pane focus
q            quit when no modal owns it
~~~

Effectful destructive actions require explicit activation/confirmation.

Real text-entry fields own normal typing.

Do not globally steal keys from an active prompt/editor widget.

### 46A.11 TUI responsive geometry

Model the layout in real terminal cells, as AW does.

Starting policy:

| Width | Layout |
| --- | --- |
| <70 cols | one pane, nav overlay |
| 70–99 | compact rail + one workspace pane |
| 100–139 | rail + workspace, detail as overlay or narrow pane |
| 140+ | rail + workspace + contextual detail pane |

Preserve:

- selected resource;
- scroll anchor;
- split ratio;
- search;
- maximized/restored state;

across resize.

No information should disappear permanently because the terminal narrows.

### 46A.12 TUI rendering implementation

Use a native Rust TUI stack, likely Ratatui + Crossterm unless a materially
simpler qualified stack exists at implementation time.

Keep TUI dependencies in the product/frontend layer.

Do not leak terminal widget types into the control/runtime crates.

The TUI event loop should multiplex:

- keyboard/mouse/resize;
- control events;
- periodic lightweight refresh only where necessary.

Avoid high-frequency polling when semantic events already exist.

### 46A.13 Web architecture — borrow AW directly

The Web UI should intentionally follow AW's browser architecture.

Use:

- SolidJS;
- Vite;
- JSX/JavaScript unless TypeScript materially improves the implementation;
- signals/memos/effects for local view state;
- small controller modules for complex view models;
- a thin API client module;
- committed/reproducible production build output where the repository policy
  chooses the same model as AW;
- compiled assets embedded and served by the Rust binary.

There is **no Node runtime in production**.

Conceptually:

~~~text
web/
  package.json
  vite.config.js
  src/
    main.jsx
    api.js
    control-state.js
    command-palette.jsx
    shell.jsx
    home.jsx
    running.jsx
    serve.jsx
    models.jsx
    downloads.jsx
    devices.jsx
    profiles.jsx
    config.jsx
    logs.jsx
    styles.css
  dist/
    ...
~~~

Do not copy AW-specific Agent/File/Git/PTY concepts.

Borrow its shell/state/presentation architecture and visual discipline.

### 46A.14 Web shell

Wide browser layout:

~~~text
┌─────────────────────────────────────────────────────────────────────┐
│ ⚡ ZeroGravity       default · config.yaml     HIP · Radeon 8060S   │
├───────────────┬───────────────────────────────┬─────────────────────┤
│ Home          │                               │                     │
│ Running       │       primary workspace       │ contextual detail   │
│ Serve         │                               │ when useful         │
│               │                               │                     │
│ Models        │                               │                     │
│ Downloads     │                               │                     │
│ Devices       │                               │                     │
│               │                               │                     │
│ Profiles      │                               │                     │
│ Config        │                               │                     │
│ Logs          │                               │                     │
├───────────────┴───────────────────────────────┴─────────────────────┤
│ status / context-sensitive shortcuts                               │
└─────────────────────────────────────────────────────────────────────┘
~~~

Like AW:

- the selected resource can open as its own focused tab/view where useful;
- a detail pane may maximize;
- maximize is visual only;
- restore preserves previous geometry/state;
- one title owner avoids nested repeated headings.

Unlike AW, do not make generic tiling a primary feature. ZeroGravity is a control
panel, not a free-form IDE.

### 46A.15 Web responsive behavior

Desktop:

- navigation rail;
- main workspace;
- optional detail pane.

Tablet:

- compact/icon navigation;
- main workspace;
- details become a drawer.

Mobile/narrow:

- one main pane;
- bottom or overlay navigation;
- details drill in as a page;
- no horizontal table dependence.

Responsive collapse preserves selection and state.

### 46A.16 SolidJS state model

Follow AW's useful separation:

~~~text
server snapshot/event state   authoritative observation
        │
        ▼
small Solid controllers
        │
        ├── filtered/sorted lists
        ├── selection
        ├── tabs/details
        ├── command palette
        └── optimistic transient UX where safe
        │
        ▼
Solid components
~~~

Do not mirror the entire Rust Runtime object graph in JavaScript.

Use stable opaque IDs for:

- model instance;
- download;
- request;
- server;
- device;
- profile/config document.

Friendly names are labels, not identities.

### 46A.17 Web control transport

The Web control API is **not** the OpenAI /v1 inference API.

Keep namespaces separate:

~~~text
/v1/...          OpenAI-compatible inference protocol
/zg/control/...  ZeroGravity control actions/snapshots
/zg/events       ZeroGravity control event stream
/ui/...          embedded static web assets
~~~

Exact routes can be refined later.

Prefer:

- ordinary HTTP JSON for snapshots/actions;
- SSE for one-way control/event updates initially;
- WebSocket only if a genuinely bidirectional control feature requires it.

Do not choose WebSocket merely because it looks interactive.

The control surface stays loopback-only by default.

### 46A.18 TUI and Web simultaneous use

If the TUI chooses **Open Web UI**, start the web control listener in the same
ZeroGravity process and point it at the same ControlPlane/Runtime.

Both clients may observe and control the same model/server/download owners.

They do not duplicate them.

Example:

~~~text
one Runtime
  ├── TUI client
  └── browser client
~~~

A model loaded from the browser appears in the TUI via the same control event.

A download paused from the TUI immediately updates in the browser.

### 46A.19 Client action consistency

The following should mean the same thing everywhere:

~~~text
Run
Serve
Load
Unload
Pull
Pause
Resume
Cancel
Verify
Remove
Edit config
Edit profile
Start server
Stop server
~~~

The direct CLI, TUI and Web call the same typed operation.

Never reimplement business rules in JSX or terminal widgets.

### 46A.20 Command palette

Borrow AW's one-palette rule.

TUI:

~~~text
:
~~~

Web:

~~~text
Ctrl+Shift+P / F1
~~~

Both search the same destination/action catalog conceptually:

- actions;
- models;
- running instances;
- devices;
- profiles;
- views.

Do not create a second global quick switch.

Palette selection of a resource is inert until an explicitly effectful action is
chosen.

### 46A.21 Logs and observability

Logs are an operational view, not an observability product.

Provide:

- bounded recent log ring;
- level;
- component;
- model/request correlation;
- text search;
- filters;
- pause/follow.

Do not ship Prometheus/Grafana concepts into the default UI.

More advanced metrics can live under diagnostics later.

### 46A.22 UI persistence

Persist only useful presentation preferences:

- last section;
- column widths/split ratio;
- collapsed rail;
- theme;
- table sort where valuable.

Do not persist:

- transient modal state;
- selected destructive action;
- stale "running" facts;
- model/runtime authority.

The browser may use localStorage for local presentation preferences.

The TUI may use the same UI section in ConfigDocument for durable user
preferences where appropriate.

Do not make frontend persistence a second settings system.

### 46A.23 Theme

Initial target:

- dark default;
- light supported;
- monochrome/high-contrast fallback;
- terminal color capability fallback;
- restrained semantic accent colors.

Web and TUI share semantic tokens but render them independently.

Do not hard-code a huge design-system framework.

### 46A.24 Accessibility

Web:

- semantic HTML;
- keyboard complete;
- visible focus;
- ARIA only where native semantics are insufficient;
- no color-only status;
- reduced-motion support.

TUI:

- Unicode/ASCII fallback;
- no color-only status;
- terminal-cell width measured correctly;
- usable at 80×24 even if visually richer at larger sizes.

### 46A.25 Security boundary

The Web control interface is powerful.

Default:

- loopback only;
- same-process random control credential/cookie;
- no remote bind without explicit secure exposure design;
- CSRF-safe action semantics;
- no model/provider tokens in client-readable state;
- no raw config secrets.

OpenAI API authentication and control-plane authentication are separate
concerns.

Do not assume an OpenAI API key authorizes model deletion/config edits.

### 46A.26 Web build integration

Mirror AW's useful production model:

- web source is developed with SolidJS/Vite;
- build is deterministic/reproducible;
- Rust embeds the generated assets;
- production requires no npm/node process;
- dev mode may run Vite against a local control API.

Keep build integration out of model/backend crates.

### 46A.27 Testing philosophy

Do not create a giant frontend fixture estate.

Shared control logic:

- focused Rust state/action tests.

TUI:

- reducer/navigation tests;
- a few fixed-cell snapshots at representative widths;
- key interaction tests.

Web:

- controller/unit tests for important state transitions;
- a small browser smoke matrix;
- responsive/layout checks for representative widths;
- no pixel-perfect screenshot explosion.

Cross-client acceptance should prove only a few critical shared-owner flows:

1. TUI loads model → Web sees same instance.
2. Web starts OpenAI serve → TUI reflects it.
3. TUI pauses download → Web sees Paused → Web resumes → TUI updates.
4. profile edited in Web → same validated file visible in TUI.
5. closing either frontend does not silently kill shared owners unless that
   frontend owns the whole process and shutdown was explicitly requested.

### 46A.28 Implementation ordering inside the CLI/UX iteration

Expand the UX implementation slices to:

~~~text
UX-1   shared presentation/event foundation
UX-2   typed command tree + unified model/config resolver
UX-3   shared ControlPlane snapshot/action/event model
UX-4   model lifecycle + downloads
UX-5   one-line run
UX-6   stable serve CLI surface
UX-7   ConfigDocument/profile management

UX-8A  native TUI shell
UX-8B  TUI Home / Running / Serve
UX-8C  TUI Models / Downloads / Devices
UX-8D  TUI Profiles / Config / Logs

UX-9A  SolidJS/Vite embedded web shell
UX-9B  Web Home / Running / Serve
UX-9C  Web Models / Downloads / Devices
UX-9D  Web Profiles / Config / Logs
UX-9E  shared control events + simultaneous TUI/Web acceptance

UX-10  doctor/pipeline polish + compatibility cleanup
~~~

UX-8A implementation (26 Nov): read-only native shell with Ratatui 0.30.0
and one Crossterm 0.29.0, optional frontend-only tui feature selected in
ordinary defaults/all and product release. Bare capable TTY and selected
document/CLI UI policy, direct commands and non-TTY help unchanged. Nine
destinations, responsive rail/narrow overlay, destination palette, focus
and scroll, scoped raw/alternate screen restoration and Unix stop signals.
Borrow original Runtime/ControlPlane; read-only live snapshot/event cursor
and cold cache/device/config/profile observations. No implicit load,
acquisition, listener or task. Cold views synchronous, Logs only bounded
client-local control notifications. Fixed-cell/reducer and actual Linux PTY
navigation/resize/quit/SIGTERM/plain fallback, strict lint pass; full
isolated gate 2477 pass/0 fail/109 skips, optimized release PTY including
profile/CLI preference and installer/scripts pass. UX-8B–D add actual
owner actions, background operation presentation,
rich model/config/log views; no duplicated runtime or settings authority.

UX-8B implementation (26 Nov): shared CLI canonical serve/native UI
resident planner, exact original cache/local/device/host/native route
policy and original control-owned registration/listener. Home load/run,
Running exact owner selection/invoke/unload, Serve start/stop generation,
same runtime and one owned action worker, cooperative original stream
cancel and bounded attributed output copy. Confirmed process quit retires
owned work/listener/runtime; modal input and bracketed paste do not
activate shortcuts. No implicit acquisition, new execution grant or
chat history. Existing optional serve feature; configured one cold route
set at a time, cold inventory sync, Logs only transient notifications.
First serial real cached GGUF TUI/HTTP lifecycle and combined release
6/0/0 (24.59s), strict lint and combined CUDA/HIP/acquisition/serve
compile pass. Final repeat-invoke/paste real PTY/HTTP rerun 35.16s and
other five release tests pass, after aligning fixture HTTP read timeout
with existing 60s bound; no relaxed admission or expected output.
Full isolated gate passes (2,479 passed, zero failed, 109 ignored);
release PTY, installer and script contracts pass. UX-8B accepted.
UX-8C/D then Web.

UX-8C implementation (26 Nov): typed exact cached digest/catalog/partial
rows, contextual list/detail, inspect/verify/load/run/serve, non-replacing
alias and original preview/confirmed reclaim. Physical device observation,
cold selected-source/device policy check or load; one coalesced inventory
reader and original action worker. Explicit immutable HF/GitHub file/
quant intent or approved Qwen recipe, scrollable consent, bare offline,
same runtime original controlled transfer, observed bytes/rate/phase/
durability and exact operation pause/resume/cancel. No automatic quota
increase, alias retarget, resume from opaque checkpoint key or advanced
provider auto-selection. Focused reducers/intent and actual original
controlled-transfer native PTY pass, strict lint pass; actual cached lifecycle
and remaining release tests pass, offline shell PTY passes. Full isolated
gate passes (2,481/0/109), release PTY/installer/scripts pass. UX-8C
accepted. UX-8D config/profile/logs then Web.

Build TUI before Web so the shared control surface proves it is not browser-shaped.

Then build the SolidJS frontend against the same control model.

Do not defer Web until after execution-performance work; both interactive
surfaces should be stable before the OpenAI-serving and performance iterations.

### 46A.29 Acceptance

This CLI/UX iteration does not close until:

- bare ZeroGravity opens the TUI on a capable TTY;
- non-TTY remains clean CLI;
- all normal direct commands remain direct;
- TUI can manage current models/downloads/devices/profiles/config/logs/server;
- \`zg web\` starts the embedded SolidJS control UI;
- TUI can open the Web UI over the same process/runtime;
- Web provides the same high-value management surfaces;
- both clients use one typed control-plane action/snapshot/event model;
- no frontend duplicates runtime/model/download owners;
- visual selection never implies an effect;
- responsive TUI and Web preserve state on resize;
- command palette is singular per client;
- ConfigDocument remains the one configuration language;
- OpenAI /v1 serving is architecturally separate from the /zg control surface;
- browser control auth is distinct from OpenAI inference auth;
- the UI stays minimal enough that a user can understand the system at a glance.

---

## 47. Acceptance criteria

The iteration closes when the following are true.

### Simplicity

A user can:

~~~text
zg run qwen3.5-2b "hello"
zg serve qwen3.5-2b
zg model pull qwen3.5-2b
zg model list
zg model inspect qwen3.5-2b
~~~

without learning cache internals.

### Interactive UX

On a capable TTY:

- no-argument launcher works;
- model selection is searchable;
- progress is responsive;
- pause/resume/cancel work;
- colors/symbols are tasteful;
- terminal scrollback remains usable.

### Script UX

With pipes/non-TTY:

- no prompts;
- no ANSI;
- no emoji;
- stable stdout;
- progress on stderr or suppressed;
- JSON/JSONL available;
- explicit network/destructive intent required.

### Transparency

\`-v\`, \`-vv\`, \`--explain\` and inspect expose enough information to understand:

- source;
- snapshot;
- backend/device;
- configuration provenance;
- resource policy;
- refusal reason.

### Correctness

No UX path weakens:

- source authentication;
- immutable snapshot identity;
- backend qualification;
- device admission;
- StateRoot authority;
- cache lifecycle;
- cancellation/retirement.

### Maintainability

- one normal model resolver;
- one presentation system;
- one global output grammar;
- one ConfigDocument schema and one-document selection model;
- no daemon required;
- no second cache or acquisition engine;
- legacy aliases do not drive new architecture.

---

## 48. Recommended user experience examples

### First run, missing model

~~~text
$ zg run qwen3.5-2b "Why is the sky blue?"

📦 qwen3.5-2b is not cached
   Qwen/Qwen3.5-2B · 3.9 GiB · revision 4ba787ec…

Download and run?  [Y/n] y

↓ Downloading qwen3.5-2b
  2.8 / 3.9 GiB  █████████████░░░░  72%  86 MiB/s  13s
  cached 128 MiB · resumed 0 B

✓ Verified · snapshot 88606494…
⚡ HIP · Radeon 8060S

The sky appears blue because ...
~~~

### Cached one-line run

~~~text
$ zg run qwen3.5-2b "hello"
⚡ qwen3.5-2b · HIP · Radeon 8060S
Hello! How can I help?
~~~

### Quiet pipeline use

~~~text
$ zg run qwen3.5-2b "hello" -q > answer.txt
~~~

### Machine use

~~~text
$ zg model inspect qwen3.5-2b --format json
~~~

### Serve

~~~text
$ zg serve qwen3.5-2b
🌐 ZeroGravity
✓ qwen3.5-2b · HIP · Radeon 8060S
● http://127.0.0.1:8000/v1
  active 0/8 · queued 0
~~~

### Explain

~~~text
$ zg run qwen3.5-2b --dry-run --explain

Model      qwen3.5-2b
Snapshot   88606494…
Backend    HIP                 auto
Device     gpu:0               auto · best admitted fit
Context    32768               profiles/laptop.yaml
Headroom   10%                 CLI
State      paged64             smart config
~~~

---

## 49. What not to do

Do not:

- build a separate GUI-like terminal application for every command;
- require interactive mode;
- put progress on stdout;
- auto-download in scripts;
- silently fall back across backends;
- expose snapshot digests as the normal model UX;
- duplicate cache authority in a "model database";
- create a background daemon only for download controls;
- store tokens in config/profile documents;
- create profile inheritance;
- create multiple model-reference parsers;
- snapshot every colored line in tests;
- use UX work to broaden model/backend qualification;
- make help a dump of every advanced engine flag.

---

## 50. Relationship to the following OpenAI-serving iteration

The immediately following iteration is
`../serving-and-speculation/NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md`.

That server iteration should assume this UX is stable and should not redesign the
normal command/config/model lifecycle while adding protocol compatibility.

The interface iteration should leave behind:

- one model resolver;
- one backend/device resolver;
- one ConfigDocument schema with config.yaml as default and named profile files;
- one explain/provenance surface;
- one model lifecycle;
- one stable `zg serve MODEL...` command/config surface;
- one structured event/output layer.

The OpenAI-serving iteration then makes the server behind that stable interface
fully compatible before physical execution-performance work begins.

---

## 51. First activation prompt

When the current live iteration is closed, the next agent should begin:

> Read the current CLI source, current model/cache/acquisition/run/serve SDK
> surfaces and
> \`NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md\`. This is the UX/interface
> iteration and comes immediately before the OpenAI-compatible serving iteration,
> which in turn precedes backend execution-performance work. Preserve
> all existing source/cache/device/state authorities. Treat config.yaml as the
> implicit default ConfigDocument and profiles/<name>.yaml as same-schema named
> alternatives; never layer them or create a separate settings store. Start with
> UX-1 only:
> introduce one small presentation/event layer with rich TTY and plain non-TTY
> renderers, global quiet/verbose/color/progress policy, stable stdout/stderr
> behavior and typed errors/hints. Do not change model resolution or acquisition
> semantics in that first slice. Preserve the design requirement that the bare
> interactive command will ultimately open the native full-screen TUI and that
> the SolidJS web client will consume the same ControlPlane; do not create a
> duplicate frontend runtime, daemon or authority model.