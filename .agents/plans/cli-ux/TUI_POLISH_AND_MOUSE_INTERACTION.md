# ZeroGravity — native TUI interaction and visual polish

Author: Project owner
Co-authors: GPT-6.2
Owner: @pvl
Decision: 26 Nov 2026 — polish before public serving
Status: UX-P1/P2/P3 implemented and focused-qualified (26 Nov); UX-P4 final ergonomics/acceptance next.

This is the follow-on to the accepted UX-1–10 iteration at `1b8de1cb`.
It supersedes only the relevant interaction/presentation details of
[NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md](NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md)
§46A.6–12 and §46A.29. Keep the [full plan](../FULL_IMPLEMENTATION_PLAN.md),
[CLI/device owner TODO](../cli-device-routing/TODO.md), [current handoff](../../docs/CURRENT_HANDOFF.md)
and accepted SDK/control/runtime/config contracts. This is a bounded
product polish phase (UX-P), not a redesign of the execution fabric.
Complete it before §2.3 public serving/ordinary benchmarks.

## 1. Intent

A quiet, beautiful instrument panel that feels like a native application,
not an assortment of terminal printouts. Elegant because the hierarchy is
obvious, not because everything is decorated. Comfortable at 80x24 and
useful across a wide terminal. Mouse-first discoverability and equally
complete keyboard operation.

- Click any visible navigable destination, row, tab, field, button, picker,
  confirmation, scrollable pane and resize handle. No action requires
  remembering a shortcut or navigating by an unrelated key.
- Resize and maximize/restore meaningful panes; scroll each inner pane
  independently. Preserve selection, search, scroll anchor and preferred
  split through resize, navigation, refresh and temporary collapse.
- Use one clean semantic color vocabulary, with restrained accents,
  readable typography/cell spacing and clear focus. Dark, Light, Auto.
- Keep exact source/registration/generation/operation/document identity
  and truthful current-state labels; never hide a refusal behind styling.
- Reuse original effects, read workers, confirmation flows, snapshots
  and typed configuration writes. No new runtime, cache, scheduler,
  provider, execution or persistent presentation owner.

## 2. Inspected baseline and reference

### Current ZeroGravity

Source inspected at the accepted checkpoint:

- `crates/zerogravity-cli/src/tui.rs`: dirty/event-driven foreground loop
  handles resize, paste and keys; no mouse event dispatch.
- `tui/terminal.rs`: scoped raw mode, alternate screen and bracketed paste;
  no mouse capture. Restore on ordinary errors, quit and unwind.
- `tui/view.rs`: two fixed rail widths, fixed 50/50 library splits,
  single per-view scroll counter, header/footer full of key hints,
  independent keyboard modal handling. Cyan bold and DarkGray are the
  primary semantic styles. Library list offsets derive from selection;
  detail scrolling counts unwrapped source lines rather than visual rows.
- `tui/library.rs`: stable typed row keys and original bounded observation
  worker already exist. `tui/logs.rs` observes the bounded host ring.
- `control_host/{settings,work,actions}.rs`: exact ConfigFile snapshot,
  typed mutation, original shared action slot and runtime lifetimes.
- `pipelines/document.rs` UiDoc and configuration introspection:
  color/progress/interactive settings exist; no theme.
- Web `src/styles.css`: already uses slate/teal dark and light palettes;
  `main.jsx` has a local binary light/dark toggle, initially dark.

Do not rebuild the UI from scratch or create another widget/runtime framework.

### Whisrs (actual source, not the desktop overlay mistaken for its TUI)

The local Nix configuration pins Whisrs v0.2.1,
`5f5be26e8886e575caa1464133e961b449561fa3`. Reviewed the corresponding
[TUI theme](https://github.com/y0sif/whisrs/blob/5f5be26e8886e575caa1464133e961b449561fa3/src/cli/tui/theme.rs),
[palette](https://github.com/y0sif/whisrs/blob/5f5be26e8886e575caa1464133e961b449561fa3/src/overlay/palette.rs),
[Home layout](https://github.com/y0sif/whisrs/blob/5f5be26e8886e575caa1464133e961b449561fa3/src/cli/tui/screens/home.rs)
and [manifest](https://github.com/y0sif/whisrs/blob/5f5be26e8886e575caa1464133e961b449561fa3/Cargo.toml).

Borrow the visual discipline, not its application architecture:
Carbon's soft white, charcoal/gray and teal success; Ember's warm amber
and Cyan's blue accent are examples of the same semantic roles. Whisrs
uses terminal-native backgrounds, two levels of muted text/rules,
light-aware roles, truecolor/ANSI16/NO_COLOR fallback and a bounded
terminal background probe outside its input loop. Its narrow home stacks
information instead of overcrowding columns. ZeroGravity already has
a compatible teal/slate Web identity, so adopt one original coordinated
dark/light palette instead of copying those named themes. No Whisrs
code, theme dependency, agent bridge or extra frontend framework needed.

## 3. Visual system

### 3.1 Palette

Proposed baseline tokens (hex RGB; tune by inspecting real terminal cells
during UX-P1, not by adding a theme engine). Colors express meaning rather
than object type; a GPU vendor is not a color.

| Semantic role | Dark | Light | Use |
| --- | --- | --- | --- |
| Canvas | #111318 | #F5F6F8 | Whole shell |
| Surface | #171A20 | #FFFFFF | Main pane, navigation |
| Raised | #1E232B | #EEF1F5 | Dialogs, input/action chrome |
| Hover | #202731 | #E9EEF2 | Pointer affordance, optional |
| Selection | #233735 | #DCEFEA | Selected resource/navigation |
| Text | #E8EBF0 | #26313C | Body/value/output |
| Secondary | #ADB6C3 | #4F5D6D | Labels, supporting metadata |
| Muted | #929EAD | #596879 | Help/idle text, still readable |
| Rule | #343D48 | #D7DEE6 | Noninteractive hairline |
| Handle | #7C8998 | #788797 | Divider, scrollbar, affordance |
| Accent | #80D1C5 | #116B63 | Focus, active destination, primary action |
| Success | #83CFA8 | #206C4B | Ready, durable, saved |
| Warning | #E5BC79 | #8A560E | Paused, caution, incomplete |
| Danger | #F1979C | #A33743 | Failed, destructive confirmation |
| Info | #9DB8E6 | #315E99 | In-progress/status information |
| On accent | #111D1D | #FFFFFF | Primary button text |

Use solid background only on small semantic regions: one active selection,
focused input, primary action, modal. No gradients, rainbow badges,
bright borders on every panel, full-page cyan, excessive iconography or
fake dashboard cards. Whitespace and alignment are the main hierarchy.
Use compact 1-cell separators, one-cell padding where geometry allows,
aligned label/value columns and consistent units. Soft neutral titles;
accent only active/focused elements. One clear primary action per view,
ordinary secondary buttons are low-key, destructive action styled only
when relevant. No flashing selection, animated decorative chrome or
attention-grabbing progress when idle.

Provide a small copyable palette value and role-to-Style functions in
`tui/theme.rs`, not per-widget RGB or a configurable theme framework.
Only Dark and Light palettes; Auto is a selector, not a third palette.
Use the same tokens/vocabulary in Web CSS (small parity pass; no wholesale
Web redesign). Primary readable foregrounds against painted surfaces
target WCAG AA 4.5:1; actionable dividers/focus indicators 3:1. Passive
decorative hairlines need not be prominent. Initial RGB token pairs were
checked for contrast; verify actual state combinations, ANSI fallback
and readability in real terminal renders.

### 3.2 Appearance settings and terminal depth

Add optional `ui.theme: auto|dark|light` to the existing strict UiDoc,
default auto if absent. A separate typed ThemeMode enum, not UiSwitch
(always/never). Preserve old documents. Suggested `ui.mouse` reuses
UiSwitch (auto default; never disables capture; always requests capture
only on a capable terminal). No extra YAML schema, persistent UI database,
theme directory, downloads or user palettes now. Example:

~~~yaml
ui:
  theme: auto
  color: auto
  mouse: auto
~~~

Show Theme as a three-choice picker in Config → Appearance and a compact
header control. Click or keys preview; Apply/Cancel explicit, active target
shown. Preview is client-local and reversible. Apply uses the original
exact observed ConfigFile/typed action slot, validates and saves once.
No implicit document creation on launch/preview. Missing default document
can be created only by an explicit authorized save. Inactive profile
changes affect only that profile; active document changes repaint the
native UI and attached Web through configuration invalidation, without
reloading models or rebinding the listener. Failed or stale saves retain
the draft and allow reload/retry; never silently retry writes.
`zg config set ui.theme dark` is the CLI path. No extra global flag needed
for v1. Extend existing safe Web settings schema/typed picker for theme;
replace binary browser toggle with Auto/Dark/Light; browser Auto follows
prefers-color-scheme, including changes. Do not allow appearance settings
to bypass or alter existing CLI color/progress/interactive policy.

Native Auto resolves terminal appearance once at session start:
1. Known local terminal background (safe bounded OSC 10/11 probe before
   raw mode/input reader, never while keys are being consumed).
2. Recognized COLORFGBG background hint.
3. Dark fallback when unknown, stated in appearance help.
Explicit Dark/Light skips detection. Cache the observation for the session;
switching to Auto uses it or the hint/fallback. No ongoing terminal query,
no background process. Skip active probe on SSH/multiplexer or incapable
terminal, where delayed replies can become input. The Whisrs approach
uses terminal-colorsaurus; consider that small audited TUI-only dependency
for reliable local detection, subject to existing dependency/offline
authorization. It was not found in the accessible local Cargo source cache.
Do not download/install it during this planning work or hand-roll a risky
blocking OSC parser. If not approved/available, implement the conservative
environment fallback and document/track the local query limitation
honestly; do not claim true automatic detection on all terminals.

Explicit Dark/Light and resolved Auto use the selected full canvas in
truecolor for predictable contrast. This is an intentional difference
from Whisrs' transparent terminal-native Carbon. Truecolor when supported,
256-color mapping if available, role-specific ANSI16 fallback otherwise;
no assumption that user-defined ANSI colors have exact RGB values.
Unknown terminal capabilities stay conservative. `--color never` and
NO_COLOR (respect existing explicit --color always precedence) suppress
RGB/background color; semantic markers, bold/reverse and text carry the
same information. TERM=dumb and non-TTY direct output are unchanged.
Do not change emulator palette via OSC or set global terminal colors.
Theme choice persists even when color is disabled.

### 3.3 Semantic status

Always pair hue with text/shape: Ready ✓, Listening ●, Loading/Running ◐,
Paused ‖, Pending …, Stopped ○, Failed !, Cancelled ×, Unavailable —.
ASCII-compatible markers where glyph width/terminal support is uncertain.
Use actual typed status, not inferred success from a progress percentage
or absence of events. Distinguish selected item, keyboard-focused pane,
active destination, active profile and admitted owner. Disabled actions
are dimmed but labeled, with a reason (not compiled, offline, busy,
stale, not applicable). Do not make disabled areas clickable. Unknown
memory/total/progress remains Unknown/—, never 0. User/model output remains
body text; do not interpret it as terminal escape sequences.

## 4. Layout

Keep nine destinations and their existing key numbers. Group rail labels
only when height permits: Workspace (Home, Running, Serve), Resources
(Models, Downloads, Devices), Settings (Profiles, Config, Logs).
Header: ZeroGravity / current view; active profile, compact appearance
control and connection/owner status if space. Rail has one active marker.
Workspace has one title and context actions, not a title inside a
title. Footer: current operation/notice (higher priority than help),
then compact context hints and clickable Help/Quit. No redundant
navigation instructions in every content pane.

Representative wide view (not literal exact cell artwork):

~~~text
 ZeroGravity / Models                    laptop •     Theme: Auto ▾
───────────────────────────────────────────────────────────────────────
 WORKSPACE    │ Models                [ Cached ] Catalog  Partial
   Home       │ Search models…                     [Pull model]
   Running    │
   Serve      │ NAME                  SIZE   STATUS │ Qwen3.5-0.8B
 RESOURCES    │ ▎ Qwen3.5-0.8B        0.5 G  Cached │ GGUF · snapshot
 ▎ Models     │   Qwen3.5-2B          1.4 G  Cached │ Source / digest
   Downloads  │   …                                 │ Files / bytes
   Devices    │                                     │ Verification
 SETTINGS     │                                     │
   Profiles   │                                     │ [Load] [Run]
   Config     │                                     │ [Verify] […]
   Logs       │                                     │
───────────────────────────────────────────────────────────────────────
 Ready · 2 cached models             Tab Focus  / Search  : Go to  ? Help
~~~

The rail and list/detail boundary are draggable. Header maximize/restore
and collapse/expand controls appear on meaningful panes, not on every
decorative surface. Keep minimum sizes and exactly one title owner.
Proposed defaults, measured against the actual terminal Rect and content
minima rather than hard width branches scattered in views:

| Terminal / workspace | Default |
| --- | --- |
| <20x7 usable minimum | Resize hint + quit, never panic or invoke hidden controls |
| Narrow (<70 cols) | Workspace only, clickable nav overlay, detail drill-in with Back |
| 70–99 cols | Compact rail + primary; stack list/detail if enough height, else detail drill-in |
| 100–139 cols | Rail + primary and contextual split when both fit; otherwise vertical/collapsed |
| >=140 cols | Rail + list/workspace + detail where useful |
| Short heights | Collapse chrome/action overflow, then secondary pane; no lost state |

Initial rail 22 cols (compact ~18, clamp min 15/max 32); list/details
default ~55/45 with list min 32, detail min 28 cells; vertical split
min ~6 rows per pane plus headers. These are starting design constants;
fit actual content and terminal geometry. Never squeeze below minima.
User drag adjusts preferred rail width or split ratio in current
orientation. Terminal resize clamps actual geometry, not preferred ratio;
when space returns restore it. Separate remembered preferences per
layout family/orientation, not per-resource. Maximize one pane and
restore exact prior preferred geometry. Collapse and reopen preserves
selection, filters and anchor. Pane layout/focus/scroll stays session
local, no persistence file in this phase.

## 5. Unified mouse and keyboard interaction

### 5.1 Event routing and geometry

Use existing Crossterm 0.29 MouseEvent (Down, Up, Drag, Moved, ScrollUp,
ScrollDown, ScrollLeft/Right where emitted) and scoped EnableMouseCapture /
DisableMouseCapture in terminal::Session. Restore terminal raw/alternate
screen/paste/mouse/cursor on normal exit, error, unwind, signals and any
supported suspend path. Unsupported events harmless; no global mouse
capture outside foreground TUI. Disable capture when requested by setting.
Terminal selection/copy remains terminal-emulator owned (typically
Shift+drag bypass); document variance, keyboard way to disable capture,
no automatic OSC clipboard writes or unrelated command execution.

One layout pass yields pane Rects, visible viewports, splitter tracks and
visible hit targets derived from actual rendered cells. A compact
`HitTarget { rect, target }` collection in paint order, with typed semantic
targets for pane, nav, row key, field, action, modal button, tab, scrollbar
or splitter. Reuse exact existing typed identity and existing action
handlers. Do not build a second geometry model, per-cell map, widget tree,
runtime store, or string-command dispatch. Modal layer exclusively owns
its hit regions; nothing clicks through.

Press records one target and drag state. Ordinary effect buttons activate
on release only if the same enabled semantic target is still visible and
valid; cancel on different target, resize, modal/route transition or
invalidated owner. Revalidate original owner/request identity on dispatch
even if a new snapshot appeared. A drag never activates a button; release
clears drag. Mouse move only affects hover/drag presentation, never starts
read workers, mutations or model tasks. Ignore right/middle by default;
optional context menu only if it reuses the visible action menu and offers
no unique or hidden effect. Avoid essential double-click semantics: optional
double-click may drill into read-only detail/open an editor, never confirm,
load, pull, start, remove or submit. Single click and visible actions are
sufficient. Keyboard equivalent for all controls.

### 5.2 Focus, selection and scrolling

- Click rail item navigates immediately (inert view change), click pane
  focuses it, click visible row selects exact Key/Owner and updates detail;
  click a non-action row never executes. Tab/Shift-Tab cycles logical
  panes/controls, arrows/j/k select/scroll where appropriate; maintain
  old shortcuts except in actual input fields. Focus indicator and
  selection marker differ; Enter invokes a safe drill-in or the focused
  explicit button.
- Separate list, detail, output, modal, editor and log viewport offsets.
  Wheel routes to the innermost scrollable pane under pointer, not the
  current keyboard focus, and does not change selected identity. Keyboard
  moves only the focused viewport; selection movement reveals selected
  row if requested. Page/Home/End and scrollbars reach the full content.
  Draggable thumb/track click, clamped position and visible-range math.
  Scroll only in actual visible area; no scrolling the parent when a
  child is active unless explicitly targeting it.
- Compute scroll bounds from real visible/wrapped visual rows with cell
  width (Unicode wide/combining characters, graphemes where appropriate),
  not bytes, code points or unwrapped line count. Reuse Ratatui primitives
  or existing locked Unicode utilities; no editor framework. Maintain
  stable item/log sequence or output-line anchor on refresh/resize; clamp
  when data disappears. Distinguish selection-reveal from manual browsing.
  Logs/output follow only at tail; manual scroll away pauses auto-follow,
  visible Follow button returns to tail. Bounded ring drops show gap;
  never invent history.
- Drag splitter on its visible track with pointer captured through the drag,
  clamp against minima; horizontal and vertical as applicable. Release or
  modal/resize cancels drag safely. Keyboard focus splitter + arrows
  resizes, Enter/double-click splitter may reset to default. Maximize and
  restore controls have keyboard equivalents. Pointer does not reorder
  panes or alter runtime resources.
- For text inputs and raw YAML: click to place caret in visible cell,
  drag selection within editor, keyboard edits apply to selection,
  scrolling independent of underlying view. Keep valid UTF-8/grapheme
  cursor boundaries and existing bounded paste/input semantics. No
  click outside to submit; focus changes do not implicitly save. Prompt
  has explicit Send/Run and chat/raw picker, raw editor Save/Cancel,
  existing Ctrl-S/Ctrl-J/Enter semantics where applicable. Preserve
  drafts across resize and modal-local scrolling; discard only explicit
  Cancel/Esc (confirm unsaved changes where needed). No broad text-editor
  rewrite, syntax-highlighter dependency or clipboard bridge.
- Modal forms/pickers have visible buttons, clear default action, click
  rows and scroll. Confirmations display exact target and effect;
  destructive/default action is Cancel; Apply/Confirm is explicit.
  Backdrop clicks never confirm or silently discard. Escape and keyboard
  confirmation retain original behavior. Palette one instance; clickable
  matching destinations and Appearance/Help shortcuts, not a second
  command interpreter.

### 5.3 Responsive action bar

Every action currently exposed solely by key hints gets a visible labeled
button, tab, menu item or picker. Wide bars show primary + most useful
secondary actions; narrow views put the rest behind one `More…` button
with full mouse/keyboard access. Click hitbox includes padding and label,
not just a glyph. Context hints (shortcut next to action) replace repeated
long key lists. Keep action visibility and enabled predicate from the
original typed owner/status/selection, not a parallel policy table.
Unavailable optional feature has clear state and no fake enabled button.

## 6. View-by-view design

| View | Main layout and mouse actions | Ownership boundary |
| --- | --- | --- |
| Home | Compact live overview: resident models, API listener, active transfer, profile. Quick Run, Load, Open Models, Start/Manage Serve, Open Web, refresh/help. Click status summary navigates to related view. Empty state explains next action. | No auto-load/serve/pull; status is current snapshot. |
| Running | Registration list + selected owner details and task response. Click selects exact registration; Load, Invoke, Unload/Retire, Cancel active action, chat/raw selector, prompt Send, copyable identity. Independent response scrolling/follow. | Selected owner is not a new registration. Unload confirmation and cancellation use exact original IDs. No persistence of prompts/output or new history. |
| Serve | Listener status/address, registered routes, counters, selected policy; explicit Start (form), Stop (exact generation), Open Web, Manage Models. Show disabled reasons and terminal join status. | No implicit listener on selection; no server config change applies to an admitted listener. Don't expose inference key. |
| Models | Click Cached/Catalog/Partial tabs, local search/filter, exact row selection, details with source/digest/verification/bytes. Load, Run, Load for Serve, Pull, Verify, Alias, Reclaim preview/Remove via action bar. | Static catalog not remote search/readiness; no implicit acquisition. Partial controls only if original owner supports them. Remove exact digest, preview and confirm. |
| Downloads | Operation and phase, observed vs durable/checkpoint bytes, known-total progress (indeterminate otherwise), file counts/status. Pause/Resume/Cancel buttons gated by exact live operation; transfer remains visible while browsing other pages. | Preserve original safe-pause vs request distinction. No synthetic progress, no duplicate transfer scheduler. |
| Devices | Observed physical rows, vendor/class and memory, stable selector and backend metadata/exclusion; click details, Check model policy and Load on selected device via form. | Only observed hardware names/data. Cold fit is not admission. Original device/source planner unchanged. |
| Profiles | Active marker separate from selected row; click selects, Open editor, Create, Copy, Activate, Remove (confirm). Show defaults and relative path context. | Open inert; activation affects future operations only. Original exact document/path guards. |
| Config | Grouped searchable rows, current value + provenance; detail with description, constraints and active/editing target. Click value or Edit opens typed picker/editor. Enum choices, bool toggle draft, bounded numeric/text input; explicit Apply/Unset/Validate, Copy/Activate/Remove, Raw View/Raw Edit. Appearance group Theme/Color/Mouse. | Exact original snapshot and typed validation/atomic save. Unsupported/complex fields keep raw repair path. No silent policy mutation or second schema. |
| Logs | Search, level/component pickers, Follow/Pause, scrollable operational entries, dropped count. Click row reveals safe existing correlation/details. | Host-owned bounded ring only; no model output, raw prompt, secrets or durable log store. |

All views support focus, scroll, refresh, narrow navigation and contextual
empty/loading/error states. Keep progress visible across view changes in
the footer, but not an extra persistent status dashboard. Use clear
action verbs and consistent units (B/KiB/MiB/GiB; never pretend byte or
time estimates). Long exact identities/URLs remain available through
detail, wrap/scroll; not truncated into ambiguous actions. Browser URL
modal remains private; close it without stopping host.

## 7. Implementation boundaries (simple Rust)

Preserve `App::handle`, existing Navigation/Form/Confirmation, Library
stable keys, shared control_host Actions/Selection/Document, work/read
workers and lifecycle. Add only minimal presentation modules:

- `tui/theme.rs`: semantic palette, depth/effective theme and style helpers.
- `tui/layout.rs`: pure responsive geometry and preferred split/focus/viewport state.
- `tui/input.rs` (or small existing navigation extension): mouse capture state,
  typed target dispatch into the same existing Action path; keyboard
  and mouse converge on view intent. No dynamic dispatcher/event bus.
- Shared row/button/input/scroll helper functions in `tui/widgets.rs` if
  useful. Split oversized view.rs by views only when it simplifies
  ownership; no generic declarative UI framework or TUI trait hierarchy.

Render stores only visible targets and small state per pane. No unbounded
layout/hit histories, polling thread, per-mouse-move inventory refresh or
full snapshot cloning. Dirty redraw only on actual presentation/event
change; coalesce drag/motion when useful without losing release/press
ordering. Prefer zero-cost typed enums/Rects, direct rendering and
existing bounded observations. No unsafe code or new dependencies except
separately approved terminal background helper if needed; no renderer
dependencies in SDK or CLI-only build. Do not optimize non-hot code at the
expense of clarity. Device/cache/serving owners remain original.

Shared config changes go through UiDoc, introspection, safe Web typed
whitelist and original Settings request; no new persistence path. Browser
scope is visual token/theme parity only; keep existing responsive
functional Web acceptance. No new network API/serving protocol work until
UX-P exit. Reuse actual accepted runtime tests, no massive evidence
framework, new pinned artifacts or test snapshots.

## 8. Ordered implementation checkpoints

One coherent signed, validated checkpoint per turn (~30 min upper
target); prefer practical slices, not a multitude of tiny commits. Do
not mark an untested or interrupted phase accepted.

- [x] **UX-P1 — visual foundation and appearance:** semantic roles,
  restrained chrome, Dark/Light/Auto, config/typed appearance selector,
  depth/NO_COLOR and safe Auto fallback, minimal Web token/theme parity.
  Actual rendered preview in each supported mode; agree final tokens
  against real terminal view. Preserve all original keyboard functions.
  Landed: typed optional ui.theme (Auto/Dark/Light), semantic RGB and
  ANSI256/16/no-color styles, quieter pane rules, focused selection,
  native T shortcut → exact observed Config theme picker with preview,
  explicit apply/cancel, coordinated Web CSS and active theme observation.
  Web Auto follows matchMedia; native Auto captures COLORFGBG at startup,
  otherwise dark. No active OSC probe or new dependency. Mouse setting,
  full control affordances and geometry remain UX-P2/P3.
  Strict minimal/Web-only/combined lint; 81/0/3 focused Rust tests
  (including 51/0/3 optimized CLI units), JS 3/0, actual PTY shell,
  settings and appearance (truecolor/256/16/no-color, stale Web edit
  refusal, inactive-profile isolation, persisted reopen), real Firefox
  desktop/tablet/mobile theme preview/cancel/save and existing settings
  passed. Initial narrow header overflow fixed by compact mobile glyph;
  test-only PTY repaint assertion corrected to wait for selected choice.
  Full isolated workspace gate and real resident/transfer acceptance
  deferred to assembled UX-P4, no new GPU/performance claim.
- [x] **UX-P2 — input/layout foundation:** scoped mouse capture/restoration,
  one render-derived hit layout, pane focus, rail navigation, drags,
  maximize/restore, responsive collapse and independent scroll primitives.
  Actual PTY click/nav/resize/scroll with exact geometry; no effectful
  click path until identity/confirmation mapping is proven.
  Landed: scoped Crossterm capture/restore and typed ui.mouse policy,
  F8 session override; one render-derived bounded visible hit list,
  exact row/ready-registration identity with release matching and original
  owner validation. Inert navigation, shelf and row selection, theme
  previews, palette destinations and modal isolation. Preferred rail
  width and horizontal/vertical split ratios, per-view focus, maximize,
  scroll offsets, narrow nav/collapsed detail controls. Keyboard Tab,
  F6 pane, F7 maximize, Alt+Left/Right rail, Ctrl+arrows inner resize.
  Actual wrapped visual row count uses the existing pinned Ratatui
  rendered-line-info feature; no new dependency. Scrollbars, independent
  list/detail and main/output scroll, logs pause when scrolling off tail.
  Running selected details and response share the secondary viewport
  for now. Stable anchors across reordered/refreshed content, complete
  editor/modal scrolling and Unicode editing remain P3/P4.
  Strict minimal, Web-only, TUI/Web and combined lint; focused Rust
  84/0/3 (54/0/3 optimized CLI units), actual SGR PTY click, wheel,
  scrolled exact row, both splitter orientations, maximize/restore,
  narrow nav, raw detail scrollbar/selection independence, modal mask,
  capture opt-out and restoration. Original native shell/settings/theme
  and actual Firefox desktop/tablet/mobile/settings smoke pass. Initial
  fixture assertions corrected for ANSI differential updates; no
  production interaction relaxed. Full workspace and actual resident/
  transfer acceptance remain assembled UX-P4.
- [x] **UX-P3 — full view controls:** action bars/overflow and mouse row/tab/
  filter selection for all nine views; pickers/modal buttons and typed
  settings/editor mouse interaction. Shared original actions and guards,
  no keyboard-only remaining visible operation. Verify controlled actual
  cache/download and configuration mutation plus explicit confirmations.

  - [x] UX-P3A: width-stable semantic action strips and one F4/More
    menu across nine views, visible unavailable reasons, direct typed
    Command -> original Action mapping. One shared Rc binding per
    surface, exact observed selected row/file/owner, configuration
    revision, listener/transfer/action identity and availability.
    Release must match the visible binding; fresh live snapshot and
    slot revalidation precede dispatch. Conservative refusal on
    observation changes rather than retargeting. No motion effects.
    Explicit modal apply/submit/confirm/cancel and prompt chat/raw mode,
    same keyboard submission path; busy keyboard submit retains draft.
    Menu-local wheel/keyboard scrolling, disabled rows no click hit.
    Real isolated optimized Rust 82/0/3, strict minimal/Web-only/
    TUI-Web/combined lint. Native controls, P2 mouse, shell, settings,
    appearance PTY pass. Actual controlled transfer mouse pause/
    resume/durable/cancel passed. Actual approved local Qwen 0.8B
    combined browser/native/HTTP lifecycle 1/0/0 (52.81s, five filtered)
    passed with native mouse device check, cache inspect/verify/alias,
    declined removal, exact digest run/submit, listener start/stop and
    retirement; original Web stale owner and shared policy checks pass.
    Width instability on disabled labels fixed; initial PTY assertion
    failures corrected to use actual modal labels/selected rows, original
    file and transfer effects unchanged. No new dependency, remote
    download, GPU or performance claim.
  - [x] UX-P3B: typed enum/boolean setting and log filter pickers,
    local per-shelf model search, grapheme caret/selection and independent
    editor scroll. Effective finite choices derive from the observed
    setting, not invented defaults. Edit/e opens picker, Enter retains
    original YAML scalar editor, F2 converts selected choice into a YAML
    draft; nonlisted/custom values retain lossless original editor.
    Explicit Apply/Cancel uses the same exact observed ConfigFile and
    original Action::Submit. Log l/f pickers are local; L/F cycles.
    Model Rows retains original observations and visible indices,
    no duplicate catalog or provider reads. Row identity remains exact.
    One bounded Editor owns caret/anchor/visual viewport; GraphemeCursor,
    shared width calculation, rendered row ranges and visible-only hit
    targets. No per-cell map, clipboard, secondary document store or
    execution owner. Wheel/drag scrollbar and PageUp/Down are local;
    selection replacement/paste preserves UTF-8 and input limits. Shared
    Viewport uses usize offsets; editor reaches >65,535 visual rows;
    ordinary Ratatui Paragraph scroll retains its explicit u16 bound.
    Already locked Unicode packages 1.13.3/0.2.2 are now direct optional
    TUI dependencies (only two dependency edges changed, no version
    resolution change or download). Compact forms reserve input and
    control rows; dialog identifies document or prompt registration.
    Strict minimal/Web-only/TUI-Web/combined lint, optimized Rust
    86/0/3 (58/0/3 CLI units), real editor/choices PTY with exact saved
    Unicode YAML bytes, stale finite-choice refusal, mouse filters,
    local shelf search, original mouse/controls/shell/settings/theme
    passed. Cached Qwen browser/native/HTTP 1/0/0 (55.96s, five filtered)
    passed with unchanged admitted scheduler/registration/generation
    and exact output; original cache and listener owner unchanged.
    Actual controlled transfer mouse pause/resume/durable/cancel and
    default CPU/TUI/Web release native fixtures also passed.
    Initial fixture assertions corrected for ANSI differential rendering
    and the expected logged error after deliberate stale refusal.
    Full visual/focus and remaining view affordance review belongs P4.

- [ ] **UX-P4 — finish ergonomics and acceptance:** Unicode wrapping/caret,
  scrollbar/follow/selection edge cases, small viewports, accessibility,
  focus order, no-color/ANSI, busy/disabled/stale/error/empty states,
  cross-client theme revision, browser retirement and terminal cleanup.
  Real cached resident inference/serve/download/config simultaneous
  regression through mouse and keyboard, full offline isolated gate,
  updated CLI guide/help and owner handoff. No physical GPU perf claim.
  Suggested coherent exit slices: P4A remaining ergonomics/visual review
  (stable item/log/output anchors across refresh/resize/reorder, distinct
  Running selected-detail/output affordance, safe log-row detail, compact
  modal navigation/scroll and editor edge-of-viewport selection, keyboard
  focus/shortcuts and minimal remaining view actions). Verify actual
  rendered cells and terminal PTY at representative sizes/themes; no
  golden matrix. P4B final simultaneous cached resident/transfer/config
  regressions and full offline isolated workspace gate. No premature
  acceptance; keep native Auto hint fallback truthful unless separately
  authorized bounded probe. Resume public serving only after P4 exit.

Use these as order, not four compulsory isolated branches. Preserve
working keyboard controls during every slice. Continue from the live
source and TODO rather than reopen accepted foundations.

## 9. Acceptance (minimal, high-value)

- Actual wide/normal/narrow terminal renders (e.g. 150x40, 110x30, 80x24,
  50x16, 30x8) reviewed for legibility, no duplicated headings or clipped
  essential controls. Dark/Light/Auto, truecolor and limited/no-color.
  Theme switching and persisted reopen of active/inactive profile; original
  config bytes not changed by preview/cancel or stale refused mutation.
- Focused pure geometry/navigation tests for bounds, hit translation from
  rendered viewport, modal precedence, press/release/drag, stable identity,
  wide chars and wrapped scroll. Actual Crossterm SGR mouse sequences via
  existing PTY fixture, not just direct reducer calls: nav click, row
  select after list scroll, wheel over two panes without changing focus,
  splitter drag both orientations, maximize/restore and resize. Click
  settings picker then save, cancel modal and stale release. Keyboard
  parity and correct teardown of capture on quit/error/signal.
- Actual source-bound cached Qwen test and original controlled transfer
  fixture: explicit mouse Run/Load/Unload/Serve Start/Stop and pause/resume/
  cancel, exact target identity and stale refusal. No selection/scroll/
  resize/theme effect on admitted registration, generation, operation or
  resource ownership; browser close independent, host quit joins. No
  provider download or fresh physical GPU qualification necessary.
- Existing shared config and attached Web tests, terminal-free Web
  feature isolation, CLI-only/no-default features, strict lint, normal
  full workspace/release gate. Check generated assets only if CSS/JS
  changed. Keep evidence concise (one screenshot/transcript per useful
  layout/theme and actual action results), no golden per-cell test matrix.
  Note limitations (e.g. unknown terminal auto probe or emulator capture
  bypass), do not claim full mouse/auto support where unverified.

Exit: all normal foreground TUI controls are mouse-accessible and
keyboard-accessible; resizable panes and scroll behave coherently;
the UI is polished in Dark/Light/Auto and graceful in restricted
terminals; existing runtime and configuration semantics unchanged.
Public serving resumes only after this bounded polish acceptance.
Additional named/custom themes, persistent layout history, dedicated
pipeline graph visualization, generalized terminal clipboard or remote
agent widget API are deliberately deferred.
