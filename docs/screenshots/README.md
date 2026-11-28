# Interface snapshots

Captured 28 November 2026 from actual ZeroGravity v0.1.0 development builds.

- `cli.png`: native terminal UI, Models → Catalog → qwen3.5-0.8b; Kitty in a
  private Xvfb display, 1600 × 950.
- `web.png`: full native Web UI, Models → Catalog, filtered to qwen3.5- and
  qwen3.5-0.8b selected; Firefox headless capture, 1440 × 1000.

Both captures use offline mode and isolated temporary configuration, cache, data
and state directories. Catalog metadata is actual application output. No model
was loaded or downloaded; no inference or performance figures were simulated.
The Web capture includes the acquisition form from the full native build; the
portable build omits acquisition. These are development snapshots, not published
release screenshots.

To refresh, launch the corresponding interface from the intended build with
isolated XDG directories, navigate to the same view, and capture its actual pixels.
Keep credentials, private model paths and browser chrome outside the image.
Review the resulting image before replacing the asset; retain the current support
labels and describe any build difference here.
