# Gemma4 HIP qualification controller

Promoted G two-artifact/eight-mode controller. This checkout is a **source-only
checkpoint**: prepare, build, controls and physical qualification have not run
here. ROOT owns assembly, build, `scripts/check.sh` and GPU qualification.

## Inputs

Prepare requires an explicit COMPLETE ROOT assembly, never a G-only fixture tree
or `.git` autodiscovery:

- `--source-root ABS`: ROOT's assembled all-five workspace
  (`zerogravity-core`, `zerogravity-model-qwen35`, `zerogravity-model-gemma4`,
  `zerogravity-gpu`, `zerogravity`).
- `--source-manifest ABS`: ROOT-owned fresh complete manifest with
  `files[relative_path] = {kind: "file", bytes: int, mode: int, sha256: hex}`.
  Count is `len(files)`; there is no fixed historical inventory. The unchanged
  extraction bounds are 64 MiB per member and 192 MiB for the whole source.
  The root `README.md` is a required record like `Cargo.toml`; its bytes stay
  pinned by the supplied record.
- `--payload ABS` and `--archive ABS`: both fresh, absolute, nonexistent and
  outside this tool tree. The archive must be outside the payload. Outputs under
  `ROOT/.agents/runs/<fresh-run>` are allowed and are excluded from input
  discovery. The source root may be staged without `.git`.

Prepare copies the exact manifest bytes to `payload/assembled-source.json`, every
declared source byte/mode to `payload/source/`, the tracked static templates from
`payload/` here, and packages the unchanged control/source-closure/archive wire
format. It verifies every record plus Cargo workspace/path/vendor/build-script
and literal include inputs, deriving counts and bytes from the supplied records.
These are literal mechanical checks: they do not resolve arbitrary Rust macros
or dynamic `build.rs` outputs, and they do not replace ROOT's authoritative
manifest, which alone owns the complete input set. It does not build or execute
anything.

## Workflow

```sh
python3 -B tools/gemma4-hip-qualification/prepare.py \
  --source-root "$ROOT_ASSEMBLED" \
  --source-manifest "$ROOT_MANIFEST" \
  --payload "$RUN/payload" \
  --archive "$RUN/payload.tar"

python3 -B "$RUN/payload/runner/admin.py" \
  --payload "$RUN/payload" --archive "$RUN/payload.tar" \
  --remote-base /home/pvl/.cache/abird/zerogravity/qualification/resume-all-five-gemma4-20261025-1 \
  --output "$RUN/admin-1"

python3 -B "$RUN/payload/runner/launch.py" \
  --payload "$RUN/payload" \
  --remote-base /home/pvl/.cache/abird/zerogravity/qualification/resume-all-five-gemma4-20261025-1 \
  --remote-output /home/pvl/.local/state/abird/zerogravity/resume-all-five-gemma4-20261025-1/build-1 \
  --output "$RUN/build-1" --mode build

python3 -B tools/gemma4-hip-qualification/bind-build.py \
  --payload "$RUN/payload" --build-dir "$RUN/build-1" \
  --output "$RUN/binding-1.json"

python3 -B tools/gemma4-hip-qualification/control-checks.py \
  --payload "$RUN/payload" --archive "$RUN/payload.tar" \
  --output "$RUN/control"

python3 -B tools/gemma4-hip-qualification/check-boundary.py \
  --source "$ROOT_ASSEMBLED" --payload "$RUN/payload" \
  --wrapper-log "$RUN/native-gpu-wrapper.log" --output "$RUN/control"
```

Run the eight physical modes through `payload/runner/launch.py --binding
"$RUN/binding-1.json"`. `payload/check-commands.json` is the complete command
plan. `control-checks.py` and the prepare/bind steps take explicit paths only,
never a fixed script-parent or `payload.tar` name. `check-boundary.py` has no
`--archive`: it takes `--payload`, `--source`, `--wrapper-log` and `--output`.
Both use the same `--output "$RUN/control"`: `control-checks.py` creates the
fresh control result there and `check-boundary.py` reads
`CONTROL-RESULT.json` from its own `--output`, so a separate `$RUN/boundary`
cannot work.

## Caller lifecycle (source)

`payload/runner/launch.py` and `payload/physical-runner/run.py` use one
settled pattern; no new framework or helper module:

- Managed TERM/INT/HUP/ALRM are record-only across the real `Popen` acquisition,
  child/PID assignment, sampler start and original `Log` construction, and across
  every mandatory original close. A recorded signal is re-raised immediately after
  that ownership critical section, so the child is already owned before failure.
- Outside critical ownership ALRM raises and the caller's absolute phase deadline
  bounds preflight, reads, held full SHA, `Held.after`, authority re-checks, EOF
  and the receipt. SIGTERM/SIGINT keep the existing checkpoint/drain behaviour.
- Cleanup runs under record-only handlers, so it cannot be abandoned mid-way: the
  original sampler join, group TERM -> two authenticated KILL before one normal
  reap/absence, bounded `Log` drain/EOF/all-close and every original held
  data/cache/binary FD plus the joint lease close independently, even when an
  earlier hash or authority check already failed. Each close is attempted on its
  own; one failure cannot skip the rest.
- `run.py` always closes `child.stdout` directly when the original `Log` was never
  constructed (sampler or `Log` constructor failure) and closes every binary FD it
  opened, including an artifact acquired by `bind_build`.
- The final `invocation.json` write has its own bounded deadline. A failed write is
  recorded and an optional rewrite gets its own bounded deadline; neither can turn
  a recorded failure into a pass. A signal landing between verification phases is
  recorded as a cleanup error, so the run still ends in a bounded failure receipt
  rather than an unreported exit. Handlers and the interval timer are restored
  unconditionally afterwards, even when the receipt write fails.

## Invariants

- Two artifacts and eight modes; resident calls stay source-derived, cached-row
  unknown is 2, fresh unknown is 1, and each Session/Snapshot known/fault mode is 2.
- Strict fixture semantics, contextual producer and five Ready bindings unchanged.
- Original `owned_container_v4.py` and `sigterm_cancellation.py` bytes unchanged;
  existing data/cache/helper pins, 70 distinct FDs, two cache locks, held binary,
  offline hydration, both artifact receipts, TERM -> two KILL -> one Reap/group/EOF
  and the current remote BASE join with the original wire fields are unchanged.
- Source20/21 manifests and pins are immutable **historical provenance only**.
  They are never the current file list, replacement authority or a fixed count.
- Current remote BASE `resume-all-five-gemma4-20261025-1` and the original wire
  labels stay intentionally unchanged. The old `gemma4-ple-source21-20261024-1`
  name remains historical provenance only. Existing remote prepared state at that
  historical base was not inspected: ROOT must resolve any old prepared-closure
  collision before any reuse, never silently reusing or overwriting it.

## Status and pending scope

CPU controls are accepted; nothing here is qualified. Accepted evidence:

- 81 strict caller controls: production main plus the original real `Popen`
  facade, four managed signals at every caller, real PIPE/`Log` EOF, repeated
  signals during cleanup, missing `Log`/raw-pipe fallback, ALRM between hashes and
  inside the all-held `finally`, and non-critical deadline with bounded receipt
  failure.
  Evidence: `.agents/runs/resume-all-qvg-caller-assembled-cpu-pi-20261024/FINAL-VERIFICATION.json`
  (`cases: 81`, `caller_controls_pass: true`, `fd_leaks: 0`, `rescue_needed: 0`).
- 33 complete-manifest metadata controls over the supplied complete ROOT
  manifest (Cargo workspace/path/vendor/build-script and literal include inputs),
  plus the default awaiting and accepted external receipt paths with fresh
  outputs. Evidence: the same `FINAL-VERIFICATION.json` (`prepare_cases: 33`,
  `prepare_controls_pass: true`).
- 12 late cleanup controls: owner TERM stage and pre-assignment ALRM with real
  owner retirement, held FDs closed, one child, one invocation, bounded seconds.
  Evidence: `.agents/runs/resume-all-qvg-owner-error-final-pi-20261024/VERIFICATION-OWNER.json`
  (`passed: 12`, `total: 12`, `all_ok: true`, `sources_ok: true`).

These are CPU controls only: no model, source-provenance or physics acceptance.

Still pending, owned by ROOT's single guarded executor:

- ROOT assembly of all five units, compilation of both artifacts, the
  controls/full `scripts/check.sh` gate and real source/state/physical
  qualification.
- Resolution of any old prepared-closure collision at the historical
  `gemma4-ple-source21-20261024-1` base before any reuse. Successor execution
  uses the fresh `resume-all-five-gemma4-20261025-1` namespace.

No GPU, model, network, cache or remote state was touched by these CPU controls.
