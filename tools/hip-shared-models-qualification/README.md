# H — shared Qwen35/Gemma4 HIP source qualification

Fresh `resume-all-five-hip-20261025-2` controller unit for the current complete
ROOT-assembled all-five source closure. It is a source/template namespace only:
`prepare.py` emits the source payload, archive, control and source closure into an
explicit fresh run directory. No generated payload, archive, control, closure,
binding or result is stored in this tree.

## Preparation

`prepare.py` consumes the same four operands as the qualified full-manifest
preparers:

```sh
python3 -B prepare.py \
  --source-root <ROOT> \
  --source-manifest <ROOT_MANIFEST.json> \
  --payload <fresh-run>/payload \
  --archive <fresh-run>/source-payload.tar
```

The supplied manifest is authoritative: it is copied byte-for-byte to
`payload/assembled-source.json`, and every declared file is copied with its
manifest bytes and mode. There is no fixed 1650-file or source20 assumption.
Source20/source19 manifests are historical provenance only.

## One artifact and six physical modes

The build argv still builds exactly one `hip_shared_models` example. The original
three finite modes are unchanged:

- `known-qwen-reload`
- `qwen-final-reader-unknown`
- `gemma-feedback-unknown`

The H6 unit also declares three bounded injected-policy-unknown modes:

- `source-page-query-unknown`
- `source-page-wait-unknown`
- `source-page-destroy-unknown`

They observe one actual selected source-page event and the same original
`HipSourcePageActor` lifecycle without inventing a driver-error claim, a generic
fault report, recovery, or a new framework/protocol/artifact/test phase. Each keeps
the existing 1800 s unknown wall, 64 GiB physical domain, 70 held data/cache FDs,
two cache locks, three compiler generations and original retirement/quarantine.

The pending-event probe contract is the source supervisor's
`H-PENDING-EVENT-CONTRACT.json`; the own run's `H-CONTRACT-QUESTIONS.md` records
the exact implemented keys and the remaining Rust-side blocker. Rust mode
declarations are deliberately not edited here.

## Ownership and cleanup

`payload/runner/{admin,bootstrap,launch}.py` and
`payload/physical-runner/run.py` use the established `ManagedSignals` / `_LIVE`
pattern: TERM/INT/HUP/ALRM are record-only inside mandatory ownership and close
sections, raw stdout is a fallback close, held data/cache/binary FDs close
independently, and handlers/timer are restored after a partial setup, failed
write, or failed receipt. The immutable 53ee owner helper and 00d1ef cancellation
helper are used unchanged.

## Commands

`payload/check-commands.json` lists the local admin/launch/build/bind commands and
the six remote physical modes. A separate combined HGV admin/bootstrap/
package-boundary validation is required after this source batch; this tree makes
no test or qualification claim by itself.
