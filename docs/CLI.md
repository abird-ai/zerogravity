# CLI and local control

ZeroGravity uses one runtime, typed model-source resolver, device policy,
cache and configuration authority. Use `zerogravity help COMMAND` for
the compiled binary's current options. Feature availability is not a
promise of model or hardware qualification.

### Native Nix package

On x86-64 Linux, the default Nix package builds the normal CPU, CUDA,
ROCm and Vulkan features and the supported model/control frontends.
Its launcher supplies the pinned runtime libraries, CUDA/CCCL headers
and HIP device libraries without a development shell. CUDA compiler
bindings use the version of the pinned SDK. Host driver modules, device
access, source support and resource admission remain independent
requirements.

```sh
nix run .# -- --offline
nix run .# -- --format json doctor
nix build .#default
```

The `portable` Nix package and existing cross-platform release assets
retain their portable CPU feature set. Native package realization may
need additional pinned Nix dependencies; it does not acquire model
weights. Explicit GPU runtime environment overrides remain supported.

## Start and run

```sh
zerogravity --help
zerogravity
zerogravity device list
zerogravity run ./model.gguf "Hello" -d cpu:0
zerogravity run ./model.gguf --dry-run --format json
zerogravity device inspect cpu:0 --model ./model.gguf --format json
```

Bare invocation opens the native control UI on a capable TTY when built
with that feature; otherwise it prints help. Direct commands remain direct.
A run accepts one positional prompt, an explicit prompt/file, a configured
prompt or piped stdin. On a capable terminal, a prompt-free direct run
supports repeated independent requests. Source and backend selection stay
explicit: a bare name resolves through the durable local name/catalog
authority, not a speculative remote search. Absolute paths and `./` paths
remain local. A local YAML positional argument selects a run document.

Device listing/inspection reports observations, not model admission.
The optional model check uses the original run dry-run planner with the
selected physical identity, never a second allocator or loader. Dry-run
output is a JSON decision, including in text mode; it does not acquire or
reserve. A refusal may still produce a decision report. Explicit backend
and device constraints must be satisfied or refused.

## Cached models

```sh
zerogravity model search qwen
zerogravity model list --format json
zerogravity model inspect REF
zerogravity model verify REF
zerogravity model remove REF --dry-run
zerogravity help model pull
```

Here REF denotes a listed canonical reference or exact snapshot digest.
Search is a static family intent catalog, not remote search or readiness.
Acquisition is explicit; the normal hub form is
`model pull hf:REPO@REV --file PATH` (or the corresponding `gh:` form).
Choose the intended revision and file; prefer a full immutable commit.
No implicit download is implied by search, inspection or dry-run.
`run --pull` is explicit consent for supported acquisition.
`--offline` prohibits acquisition. Supported online hub/Xet transfers
have pause/resume/cancel and structured JSONL progress. Removal requires
`--dry-run` or explicit `--yes`; referenced or busy data refuses.
Cache operations require the relevant Linux build features.

## Configuration and profiles

```sh
zerogravity config path
zerogravity config show
zerogravity config validate
zerogravity config explain
zerogravity profile list
zerogravity profile create laptop
zerogravity --profile laptop config set run.backend cpu
zerogravity --profile laptop config get run.backend
zerogravity --profile laptop run ./model.gguf "Hello"
```

Default configuration lives under
`$XDG_CONFIG_HOME/abird/zerogravity/config.yaml` (fallback
`$HOME/.config/abird/zerogravity/config.yaml`).
Named profiles are separate documents, not inherited overlays. Explicit
`--config FILE` or `--profile NAME` selects the document. CLI overrides
take precedence over selected configuration and built-in defaults.
An explicit positional model selects a fresh direct route while retaining
run policy defaults; explicit `--model` with a configured entrypoint
retains the existing override contract. Use `config edit` for local YAML
editing; typed set/unset validates and atomically replaces the document.
Profile creation and copy never overwrite existing profiles.

## Cold diagnostics and pipeline structure

```sh
zerogravity doctor
zerogravity --profile laptop doctor --format json
zerogravity pipeline check examples/pipelines/text-model.yaml
zerogravity --config examples/pipelines/text-model.yaml pipeline list
zerogravity --config examples/pipelines/text-model.yaml pipeline inspect chat --format json
```

Doctor inspects selected configuration policy, declared local path metadata,
cache metadata and existing physical inventory. Compiled GPU backends use
their original loader probes (CUDA driver, HIP runtime/RTC, Vulkan entry
loader). Library loadability is not device binding, compiler or execution
qualification. Uncompiled probes report unknown, not missing libraries.
No device client, model, payload verification or provider request is started.
Cache observations reuse the original cooperative inode charge and live
reservation calculation, under the existing nonblocking publication lock.
Busy publication reports unknown charge; no cache, lock inode or directory
is created. Logical snapshot bytes may include shared blobs; charged and
reserved bytes are quota accounting, not filesystem free space. Errors
return exit status 2 with the report; warnings alone do not fail. Neither
an `ok` diagnostic nor structural pipeline validation grants execution.
Pipeline list, inspect and check use the same typed structural checker,
including declared stages, links, context policy and routes. General
pipeline execution remains explicitly refused; declared routes are not
listening endpoints. See [pipeline examples](../examples/pipelines/README.md).

## Native UI, Web and serving

On a capable terminal, bare `zerogravity` opens the native control panel.
Press **T** to open the active document's appearance setting, Enter to edit,
arrow keys to preview, Enter to apply or Esc to cancel. The Web header
opens the same setting; changes are previews until saved. To set it directly:

```sh
zerogravity config set ui.theme dark
zerogravity config set ui.theme light
zerogravity config set ui.theme auto
```

Auto follows the browser's system appearance, or the terminal's
`COLORFGBG` hint (dark if unknown). Explicit themes work without detection;
`--color never`/existing color policy remains separate. Appearance changes
do not reconfigure admitted models, transfers or listeners.

Native mouse capture is enabled by default on a capable terminal. Click
navigation, model shelves, visible rows and pane controls; drag the rail
and inner dividers, or scroll/drag a pane's scrollbar independently.
Tab moves focus, F6 switches main/detail, F7 maximizes/restores,
Alt+Left/Right resizes the rail, Ctrl+arrows resize inner splits. F8
toggles mouse capture for this session; set `ui.mouse: never` (or
`zg config set ui.mouse never`) to opt out persistently. Terminal text
selection is emulator-owned (often Shift+drag, or disable capture first).
Selection alone does not execute. No action toolbar: the selected
row's ellipsis, right-click or F4 opens contextual actions. Enter opens
model/device/registration actions; profiles open their editor and Config
retains direct YAML value editing. The relevant pane has one action menu
when the selected row is not visible or a list is absent. The footer
Menu (F10) holds Web, Appearance, Refresh and Quit.

In searchable views, press s or Ctrl-F, or click the inline field to
focus search. Only focused search consumes typing and paste. Ctrl-U
clears; Esc, Enter or Tab returns to list focus while retaining the
filter. Arrow keys select/scroll results. Bare 1–9 switches views
outside search; Alt+1–9 also works from search. Alt+letter invokes
the indicated operation shortcut (Models Load for Serve is Alt+s).
F1/Alt+? shows help, Alt+: opens the destination palette, Ctrl-Q
quits outside dialogs. The wide rail contains the product and active
profile; narrow terminals retain their compact navigation control.
Models show human-readable sizes (B/KiB/MiB/GiB and larger units),
distinguish snapshots without weights, and show exact-source task
stages, elapsed time, observed verification/download counters and
the most recent matching action outcome. Related repository pulls
are not exact snapshot operations. Verification reports bytes hashed
and files authenticated separately; inventory/verification does not
grant execution. The canonical text planner and Models view share a
metadata-only route classification: missing weight payloads, unsupported
original Safetensors source, role-bound GGUF candidate, or approved
original Qwen source. Known structural refusals disable Load/Run and
Load for Serve but leave inspection, verification and cache actions
available. Model-specific execution choices omit unsupported backends;
a candidate is never a memory-fit or authentication guarantee.
Auto checks the actual CUDA compiler/runtime binding and HIP device
identity before choosing a route; native numerical selection requires
ROCm. After selection, admission and source loading may still refuse
without silently trying a different device or numerical contract.
Actual loaded backend/device comes from the runtime registration. No fabricated load percentage or persistent task history. Details in the context menu (Alt-D)
opens exact metadata, full digest/revision, raw bytes and last action
result without executing inspection or verification. Disabled actions
cannot activate; changed observations require choosing again.
Form and confirmation buttons use the same typed actions as
Enter/Ctrl-S and y/n/Esc. The Running detail title switches between
model details and Response (F3); each has its own viewport. New
responses follow the tail, manual scroll pauses and returning to the
tail or F9 resumes. Selected item/registration and log sequence
anchors preserve scroll through refresh and resize; model shelves
keep their own positions.

Running shows the selected registration's average decode TPS, engine
TTFT, recent aggregate-output trend, committed output tokens and completed
requests. Averages are since registration; the trend samples actual
counter changes over elapsed wall time, so it includes idle and concurrent
requests. TTFT runs from engine request acquisition to first committed
non-EOS token (queue/prefill included), not HTTP first byte. Decode uses
within-request commit intervals, including active and ended requests.
Unavailable measurements are shown as an em dash, not a fake zero.
Aliases share a registration's observations; reload resets them.
Performance details (Alt+D or selected-row context menu) shows sample
counts, last completed request, reported layer execution and original
memory bounds. Web provides the same details in a disclosure. CPU and
original prepared single-device Qwen routes report exact layer ranges;
unknown maps stay unreported. Compute placement does not imply weight
residency or GPU utilization. Trends are client-local, bounded, and not
persistent benchmark logs.


In Config, Edit from the context menu or Alt+e opens a finite setting
picker when available; choose a row and apply explicitly. F2 opens its
selected value in the original YAML form. Enter retains direct YAML
value editing; raw document repair remains available. Logs Level and
Component choices are in the pane menu (Alt+l/f, uppercase Alt+L/F
retain cycling). Click a log row or select it with arrows; Enter,
its ellipsis or right-click opens the exact operational entry (bounded local
causes are native-only; browser details use the safe projection).
Selecting/scrolling off tail pauses the bounded view; End resumes.
Filtering is client-local and does not trigger any
provider acquisition or runtime action.

### Operational logs

One-off commands keep their primary output on stdout and diagnostics on
stderr; they do not create a session log by default. Interactive TUI and
standalone Web control hosts write private JSONL session logs under
`$XDG_STATE_HOME/abird/zerogravity/logs` (default
`$HOME/.local/state/abird/zerogravity/logs`). The native Logs view shows
its current destination and any sink failure. A standalone Web host
reports the destination on stderr. Attached Web clients share the host's
log owner, not another writer or runtime.

Each session owns a separate locked directory and four rotating 4 MiB
segments. New sessions prune recognized older closed sessions beyond
16; live sessions and unrelated files are preserved. Files are flushed
for live readers and synchronized at rotation and joined shutdown.
Records include UTC timestamps, process-local sequence, elapsed time,
component, action/source correlation, real progress and typed failure
details where available. The 128-entry UI ring is a bounded recent view,
not a history browser. Inspect older retained JSONL segments directly.
A transient event-stream gap is reported rather than concealed. No log
can reconstruct errors from older sessions that never persisted them.

Foreground `serve` emits continuous plain timestamped stderr, including
request failures even with no UI attached. The NixOS service directs
stdout/stderr to journald; use `journalctl -u zerogravity -f` for live
service diagnostics. Service enablement and journal retention remain
operator-owned.

Optional environment overrides:

- `ZEROGRAVITY_LOG_FILE=/absolute/private/path` selects an explicit
  append-only file (one concurrent writer, operator-managed retention).
  `stderr` selects the stream; redirect it away from the TUI terminal
  before use. The parent directory must already exist for explicit files.
- `ZEROGRAVITY_LOG_FORMAT=jsonl|text` chooses an explicit file or stream
  format. Automatic interactive session files always use JSONL.
- `ZEROGRAVITY_LOG_LEVEL=info|warn|error` sets the destination threshold
  (default `info`); the live view can filter its own bounded ring.

Local causes are bounded and sanitized; generation/download/settings
failures that could contain private input use safe typed details instead.
Prompts, generated responses and credentials are not event payloads.
The Web/HTTP projection never exposes the local cause chain or session
file path. These logs are operational observations, not an exhaustive
audit trail. Unknown load totals remain unknown.

Text forms and the raw editor accept click-to-place caret, drag selection,
Shift+arrow selection, bounded paste/replacement, PageUp/PageDown and
mouse wheel/scrollbar scrolling. Ctrl-U clears, Ctrl-A/E move to the
start/end of the value; raw Ctrl-S saves, Esc cancels. An editor scroll
never changes an underlying model/setting selection or saves the draft.
Dragging outside a captured editor extends selection and scrolls one
visual row per actual drag event; releasing over another control does
not activate it. At short terminal heights dialogs use the full usable
screen and their text/results scroll. Mouse capture does not install a
clipboard bridge.

```sh
zerogravity web
zerogravity serve --model local=./model.gguf --listen 127.0.0.1:8000
```

The optional Web feature starts a local embedded control UI with a private
URL. Native and attached Web clients share the same foreground runtime,
action ownership, model/cache state and settings; their view state is local.
Control authentication is independent of optional inference API
authentication. Closing an attached browser does not stop the native host.
Ctrl-C stops the foreground host and joins its work.

Serving requires the serve feature and an eligible loaded model route.
The local API supports Models, Responses and conversation resources,
Chat/legacy Completions, capability-gated embeddings and finite speech,
and qualified text Realtime. Unsupported routes, models, parameters and
backends refuse. It is loopback-only; optional `ZEROGRAVITY_API_KEY` protects
inference. See the [OpenAI-compatible inference guide](OPENAI.md) for the
actual endpoint matrix, model qualification boundaries and SDK example.
Public serving compatibility and performance qualification are separate work.
See [original-source GPU serving policy](OPENAI.md#original-source-gpu-serving-policy)
for automatic physical cohorts, explicit overrides and ROCm memory
envelopes.
The existing [ordinary benchmark client](OPENAI.md#ordinary-serving-benchmarks)
measures Chat, Responses or raw Completions through this same listener.

### Error diagnostics

HTTP and SSE errors keep their status/code and a safe public message. The
original typed failure remains in-process; missing backends, unsupported
devices, invalid input, resource shortages and malformed artifacts are
distinct. Native and attached Web Logs retain bounded structured failure
metadata (category, code, HTTP status when applicable, measured requested
and available bytes, OS error code when available) without arbitrary
diagnostic text, prompts, paths or credentials.

For trusted local debugging only, explicitly enable diagnostic responses:

```sh
zerogravity config set serve.enable_error_details true
# Or use --enable-error-details on an explicit serve invocation.
```

The listener requires a configured `ZEROGRAVITY_API_KEY` and a valid bearer
credential on the request. It remains loopback-only. Errors before successful
authentication (including framing and connection admission) never reveal
details. Authorized diagnostic responses add an `error.details` object
with the structured metadata and bounded original diagnostic/cause text.
That text can contain internal information: never enable it on a public
or reverse-proxied service intended for untrusted callers. The setting
defaults to false; the browser-safe configuration editor cannot enable it.
Changing the document applies to subsequent listener starts, not the
already-owned listener. To disable:

```sh
zerogravity config set serve.enable_error_details false
```


## Scripts and compatibility

Use `--format json` on supported structured read commands, `--color never`
for plain output and `--quiet` to reduce presentation noise. Generate shell
completion with `zerogravity completion bash` (also zsh, fish and powershell).
Normal help uses canonical subcommands. Existing `devices list`, static
`models`, `inspect PATH`, `run-cache`, provider `pull` forms and explicit
backend run aliases remain available; they are not new execution owners.
