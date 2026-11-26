# Live model performance and execution inspection

Status: UX-P-M1 implemented and focused-qualified (26 Nov); optional M2 and assembled P4B pending.
User direction 26 Nov 2026: Running models should show elegant live
performance, average TPS/TTFT since the run began, and optionally detailed
performance, layers and their devices. Keep the object-first toolbar-free
UX and existing execution/resource authority.

## Presentation

Running is the dashboard. Keep one search field, quiet owner rows, and the
existing selected-row context menu. No permanent toolbar, monitoring app,
extra process, or dashboard of unrelated cards.

Selected model overview:
- Registration age, two prominent values: average decode tok/s and engine
  TTFT. No samples means an em dash, not zero. Distinguish completed
  requests from the existing admitted-reader count.
- Short recent aggregate-output sparkline and latest actual sampled rate;
  one calm accent, subdued labels, readable in no-color. No chart when no
  sample exists. Zero is actual observed idle output, not missing data.
- Compact total output and completion counts, provider/device, context and
  reported layer compute summary. Retain response/model switch (F3) and
  independent viewports. Do not imply last completed request belongs to
  a particular visible response if concurrent requests can run.
- Performance details via exact selected owner menu/Alt-D, read-only,
  scrollable; Web uses a disclosure in the existing Running panel.
  Include sample denominators, last completed request, prepared execution
  facts, layer spans, resident weights/state/capacity and context budgets.
  Explain actual measurement boundaries in details, not the normal pane.

Illustrative placement, not a new hardware qualification:

    Layer execution · 28 layers
    ━━━━━━━━━━━━━━━━━━━━━━━━  CUDA · physical device
    0–27                       original device

A future genuinely split owner can show proportional contiguous spans and
a short legend/ranges per physical device. Never infer a split by dividing
layers by VRAM, model name or desired configuration. Distinguish compute
placement, weight/storage placement, state reserve, model budget and actual
physical usage. Layer IDs are zero-based, end-exclusive in the contract.
Unknown placement remains unreported; host assistance does not mean host
layer execution. Avoid percentages unless actual comparable bounds exist.

## Measurement contract

Use one fixed-size observation aggregate on the original registry Route,
not on alias, source hash, CLI process, HTTP client or a second scheduler.
Each new registration has its own lifetime/clock and counters. Aliases
share it. Reader carries a small request-local timing sample, exactly the
existing ModelRead lifetime. Nothing retains an extra model reader.

- loaded_ms: monotonic registration age, includes idle; not GPU busy time.
- output_tokens: non-EOS committed tokens observed when the existing
  presenter advances its generated ID frontier. Count once even if
  decoder/transport subsequently fails; no duplicate on parked resume.
- TTFT: engine request-reader acquisition through first observed committed
  non-EOS token (includes validation, queue, tokenizer, prefill). Not
  HTTP-ingress-to-first-byte, socket delivery or GPU-only timing. Every
  request with a valid first observation contributes one sample, including
  an ultimately cancelled/failed run. No first-token sample for EOS-only,
  preflight failure or pre-output cancellation.
- Decode TPS: sum of valid within-request inter-commit interval counts
  divided by summed interval durations, token weighted; never subtract
  clocks of two requests or divide by time idle between requests. Live
  aggregates include active and ended requests. No rate for zero intervals
  or zero total time. Includes scheduler/backpressure delays between
  commits; not per-kernel utilization or wire delivery.
- Completed request: original Done receipt accepted by bounded presenter.
  Last completed sample preserves that request's TTFT, decode intervals,
  duration and output count; zero-output completion is legitimate.
  Existing scheduler completed counts describe work steps, not requests.
- Per-token path reuses published observed_at and existing generated count
  frontier. Relaxed atomic observations, no additional per-token clock,
  mutex, allocation, synchronization or transport event. Terminal sample
  uses small mutex once per completed request. Counters and durations may
  advance between fields in live snapshots, as existing owner snapshots.
  Existing sizeof-based route/queue/active reservations price fields.
- Recent rate: client delta of cumulative committed tokens divided by
  elapsed registration ms. Sample through existing native 1s/Web 2s
  refreshes, suppress duplicate/sub-750ms updates, clear at gaps >6s,
  regression or replacement. At most 24 observations for selected
  registration per client, no persistence or extra polling. Normalize
  plot to observed peak, display actual latest rate, don't compare its
  height across different windows/models. Aggregate throughput can exceed
  per-request decode TPS with concurrent requests.

## Ownership and exposure

Engine router/performance.rs; original ModelSnapshot owns value-only
performance. ModelDetails.layer_placement is adapter-supplied, exact
layer spans from prepared owner. Direct Qwen CPU reports all geometry
layers on Host CPU; original single-device prepared Qwen text reports
all plan layer boundaries on its actual resident service device.
Older resident adapter without reported layer map stays None. No change
to physical route admission, selected provider, source validation,
cache, scheduler, numerical semantics or public inference wire. Existing
private authenticated Web snapshot explicitly projects numeric performance,
reported layer compute ranges and safe memory bounds. No prompts, output
strings, paths, token IDs, credentials or diagnostic history in samples.

## Implementation and acceptance

- [x] UX-P-M1 registration metrics, live TUI/Web summary and bounded
  trends, exact read-only performance details, known single-device
  layer spans and memory/lease detail. Reuse original publication
  timestamps and owner lifetime. Tests deterministic interleaving,
  weighted rate, zero-token completion, alias sharing, queued cancel,
  retirement and history reset. Actual cached model control fixture
  compares native/HTTP/Web usage totals, exact new registration reset,
  CPU placement and visible native/browser performance. Final isolated
  engine 109/0/0, combined CPU CLI 64/0/3, Web 4/0 and asset build, strict
  engine/minimal/combined CPU lint, full production cargo check passed.
  Actual approved cached Qwen CPU/browser/native/HTTP 1/0/0 (55.61s,
  five filtered): two native and one HTTP requests share exact counters,
  Web adds the next request, reload resets lifetime, 24 CPU layers and
  visible native/F3/details scroll and Firefox disclosure. Exact output,
  original 8 GiB admission, stale owners and shared settings retained.
  Initial PTY assertions corrected for ANSI differential output and
  capitalization. Last visual refinement hides full digest aliases and
  secondary technical facts in Details. GPU placement compiled only;
  no new physical GPU qualification or instrumentation-overhead claim.
  Logs target/ux-live-performance-final.log,
  target/ux-live-performance-web-final.log,
  target/ux-live-performance-resident-final.log.
- [ ] UX-P-M2 optional richer request and phase inspection after M1:
  exact current response/request identity, per-request run progress,
  prompt/prefill/queue and decode timings where original boundaries
  provide them; optional bounded recent request samples/latency
  distribution only if justified, no unbounded history. Detailed
  layer/device topology across actual adapters including future mixed
  placement, layer roles/types from original model geometry, reported
  memory by owner class. Per-layer time is explicitly profiled, never
  estimated from model TPS or enabled via per-layer GPU synchronization
  by default. Reuse authoritative provider completion/timing contracts;
  qualify observer overhead and physical execution on relevant hardware.
  Keep unreported observations explicit. No requirement to invent
  unsupported split/offload or per-layer telemetry to fill a dashboard.
- [ ] UX-P4B comprehensive isolated workspace gate on assembled source,
  actual cached resident and simultaneous transfer/config qualification,
  default full production link/launch; hardware placement qualification
  on original existing routes before claim. Preserve prior GPU lint debt
  and feature boundaries. No performance comparison claim from these UX
  observations. Public serving remains after UX-P exit.

M1 is the minimal useful dashboard; M2 expands detail only on request and
according to actual capability. Do not delay UX-P4B for imaginary hardware
placement or new performance profiler architecture.
