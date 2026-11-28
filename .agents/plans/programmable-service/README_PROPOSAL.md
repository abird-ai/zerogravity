Author: Project owner
Co-authors: GPT-6 Astra Pro; GPT-6.1 Sol (editorial and SDK review)
Owner: @pvl
Status: Root README and architecture diagram applied by explicit user instruction; remaining metadata proposals pending
Date: 28 November 2026 (Asia/Singapore)

# README proposal — live programmable inference

## Applied outcome — 28 November 2026

The user explicitly authorized the README update and requested the deep-execution
authoring examples. The [root README](../../../README.md) is now the authoritative
applied copy. It includes the quick SDK, measured live revision, a supported fused
mapping, and an expandable typed RMS kernel chain with preparation. SDK examples
remain clearly labeled as planned APIs. A subsequent user request moves the
installed-binary CLI quick start to the beginning, adds authentic terminal/Web
snapshots, and authorizes committing and pushing the plans and README updates.

The [architecture SVG](../../../docs/architecture-overview.svg) now includes MCP,
the reusable Service, semantic programs, prepared mappings, shared execution
fabric, observations and saved revisions. The original proposal below is retained
as design history; package descriptions, Web title and external metadata have not
been changed by this documentation update.

The release audit found no published GitHub releases on 28 November. The README
therefore describes commands for an installed `zerogravity` executable and links
the source-install guide. It does not advertise a working latest-release download.
The portable recipe includes CPU/Qwen text, TUI and Web; acquisition, serving and
GPU support require the applicable full native build. Publishing a first binary
release remains separate release work.

## Positioning

Keep the agreed tagline: **The fully programmable, blazingly fast inference stack.**
Use **live programmable inference stack** as the short product description.
The opening should explain ordinary usage first, then show a tested live change.

Keep the name `zerogravity`, commands and crate names. Use `ZeroGravity` in prose.
Retain execution fabric as a precise architecture term; make the programmable
inference stack the public product identity.

The original proposal below replaced the former hero and Intro. The applied copy
also includes deep-execution examples and more explicit qualification boundaries.
The provenance notice is beside Related projects, preserving its substance;
the installed-binary CLI Quick start now precedes the SDK previews.

## Original proposed opening

<!-- BEGIN PROPOSED README OPENING -->
# zerogravity

**The fully programmable, blazingly fast inference stack.**

Run models with a few lines of Rust. Compose pipelines, inspect execution, and
change how inference runs—live, from model definitions to the kernels beneath them.

## SDK at a glance

> **SDK preview:** these examples show the proposed API. For current runnable
> commands, see Quick start below.

```rust
use zerogravity::Service;

let service = Service::embedded().await?;
let zg = service.client();
let source = zg.pull("catalog:qwen3.5-0.8b").await?;
let model = zg.load(source).await?;

println!("{}", model.run("Explain Rust ownership.").await?.text());
service.shutdown().await?;
```

Load local files or pull a model into the cache. Stream responses or serve the
same model through an OpenAI-compatible endpoint.

### Program the running stack

Inspect a running model, try another implementation, and measure it against a
representative workload. Activate a passing revision for new requests while
in-flight work keeps its original revision.

This separate example uses a client connected to an active `assistant` route.
`region_id` comes from inspection; `fused_mapping` is a supported implementation
recipe; `workload` includes the inputs and comparison requirements.

```rust
let base = zg.revisions().active("assistant").await?;
let region = base.program().region(region_id)?;
let candidate = zg.revisions().derive(&base)
    .map_region(&region, &fused_mapping)
    .prepare().await?;

let report = zg.experiments().compare(&base, &candidate)
    .workload(workload)
    .check_inherited_contract()
    .record("kernel-comparison").run().await?;
report.require_acceptable()?;

zg.activate("assistant", &candidate)
    .if_generation(base.activation_generation())
    .for_new_requests().await?;
```

The experiment keeps the recipe and results so you can reopen, rerun, and
iterate. **MCP exposes the same native operations to agents.**

<!-- END PROPOSED README OPENING -->

## Reviewed README treatment

| Current section | Proposed treatment |
| --- | --- |
| Hero and seven Intro bullets | Replace with the opening above; the SDK appears immediately |
| Quick start | Keep the runnable commands distinct from SDK previews; use the accepted UX-11D packaging/source guidance when it lands |
| Architecture | Keep one compact diagram and the short copy below |
| Joint optimization diagram, formula, four-concern table | Move detailed explanation behind existing engineering/design links |
| Interfaces | Consolidate with the support/roadmap section; avoid another broad feature catalogue |
| Implementation status | Keep a short supported-scope summary linked to the current baseline; do not infer availability from architectural copy |
| Delivery order | Replace the stale serial ROCm→CUDA sentence with the current foundation/parallel-lane order |
| Related projects, provenance, contribution, disclosure, license | Preserve the acknowledgements and legal/disclosure substance |
| FAQ | Keep the few questions that add information; move duplicated architecture explanation into linked docs |

### Architecture copy

**One service, one execution fabric.** CLI, TUI, Web, Rust SDK and MCP use the same
native operations. Typed task pipelines and source-bound model programs map to
prepared kernels, libraries and device execution.

The service owns lifecycle and revisions. The execution fabric owns scheduling,
state and physical resources. Run embedded or attach to an existing service;
choose implementations without changing the model's declared meaning.

The updated architecture diagram adds MCP alongside the Rust SDK, distinguishes
the reusable Service from Runtime/execution, and shows observation and revision
flow. It preserves the restrained visual language of the original SVG.

### Delivery-order copy

**Roadmap:** shared service and SDK foundations, then parallel ROCm, CUDA,
MCP and composition work. Vulkan and optional compiled providers follow the
same contracts.

Link this line to the [implementation phases](IMPLEMENTATION_PHASES.md);
the live plan retains exact prerequisites and acceptance criteria.

## Subtle branding changes

The README tagline and architecture alt text are applied. The other entries remain
proposed text changes, not implementation or external metadata updates.

| Surface | Suggested copy |
| --- | --- |
| README tagline | The fully programmable, blazingly fast inference stack. |
| GitHub About / short product description | Live programmable inference in Rust. |
| Public Rust package description | Live programmable inference in Rust |
| CLI package description | CLI, TUI and Web for the ZeroGravity inference stack |
| Web page title | ZeroGravity · Live inference |
| Architecture image caption/alt text | ZeroGravity live programmable inference stack |

The inspected public-package description currently says “Rust-native composable
AI execution fabric”; the CLI description says “Explicit CPU and resident GPU
text frontend for Zerogravity”; the Web title is “ZeroGravity · Local control.”
Their future wording remains proposed here. Do not rename internal ownership
types, the binary or packages to accomplish the positioning change.

## Publication and example boundaries

- The API preview labels cover SDK, live-revision and deep-mapping examples
  and their planned MCP projection.
  Keep it adjacent to the examples while those APIs are still planned.
- The examples are independent excerpts. The first shuts down its embedded
  Service; the second assumes another live Client and an existing named route.
- `pull` is explicit acquisition; `load` does not silently download.
  An application also needs joined shutdown on error paths.
- The selected mapping must come from supported inspection/implementation
  capabilities. A placeholder name cannot invent fusion support.
- `require_acceptable` represents declared correctness/state and workload
  acceptance. A failed or losing candidate does not become active.
- The recorded experiment preserves reproducible inputs/recipe/outcomes;
  this sentence does not promise arbitrary in-flight checkpoint resume.
- Activating a revision affects new admissions; already admitted tasks/sessions
  retain their exact revision. Avoid blanket “zero downtime” or unrestricted
  hot-patching claims.
- Keep “blazingly fast” as positioning; benchmark and parity claims belong with
  measured results and the supported-scope summary.
- Before promoting an example from preview to executable quick start, verify
  it against the actual public SDK and supported source/target scope.

## References

- [Service and SDK target](README.md)
- [Composition and adaptive execution](COMPOSITION_AND_ADAPTIVE_EXECUTION.md)
- [Foundation phases and parallel lanes](IMPLEMENTATION_PHASES.md)
- [Live checklist](TODO.md)

Inspected source locations, under `/home/pvl/spaces/abird/src/zerogravity`:
`README.md`, `crates/zerogravity/Cargo.toml`,
`crates/zerogravity-cli/Cargo.toml` and
`crates/zerogravity-cli/web/index.html`.
