# Zerogravity — Human-Centric Design, AI-Centric Implementation, and Open Contributions

**15 October 2026 · Public contributor policy.** User-approved README, MIT and contribution guidance; no inference implementation change.

## 1. How we build

**Zerogravity is deeply human-designed and extensively AI-implemented.**

- Humans own the vision, architectural boundaries, tradeoffs, technical decisions, and acceptance standards.
- AI agents assist with source research, implementation, testing, benchmarking, review and documentation.
- We do **not** let autonomous agents drive architectural direction unchecked, and do **not** claim a human manually reads every generated line.
- Source correctness, reproducible tests, reviewed design contracts, independent real model/hardware evidence and transparent provenance are our quality controls. AI output is not proof on its own.
- The development process is open to contributors regardless of whether they use AI tools; collaboration is judged on thoughtful design, correctness and evidence.

The README states this succinctly as **human-centric design, AI-centric implementation**.

## 2. Contribution workflow

The user explicitly wishes to build Zerogravity publicly together with everyone. Invite friendly architectural debate, questions, source research and small well-contained improvements.

1. Open a **design/architecture issue** explaining the problem, source references, constraints and proposed approach.
2. For nontrivial code, contribute a concise plan under `.agents/plans/<topic>.md` FIRST: alternatives, affected interfaces/files, existing invariants, source/numeric/physical safety, tests, and fallback/rollback.
3. Send a focused implementation PR with source references, tests and relevant performance/hardware evidence. Avoid unrelated rewrites.
4. Update the plan/evidence when a finding changes. Source-free mocks are not original model tests; partial GPU kernels are not full model inference.
5. Documentation fixes and small typos can go straight to a PR. Negative benchmark findings and better simpler designs count as valuable contributions.

No requirement that contributors personally author every line. Never overwrite concurrent agents' staging, and avoid hidden model/server backends that duplicate core Rust ownership.

## 3. License selection

- The user approved the **MIT license** for the Zerogravity project; see root `LICENSE` and Rust package manifest `license = "MIT"`.
- Copyright (c) 2026 Prasanna Loganathar and contributors.
- Model weights, processors, external assets, dependencies and imported source retain their independent licenses. MIT does not automatically relicense third-party code.
- For external source reuse, pin the upstream revision, inspect per-file notices and provide attribution/licensing evidence; architectural study does not imply source import authorization.

## 4. README and implementation evidence policy

Root README is a **product entry point**, not the running implementation log. Include the exact user tagline, clear text/image/video/audio/realtime ambitions, minimal highlights, simple local quick start, compact fabric and joint optimization diagrams, FAQ, high-level honest status/roadmap, contribution invite, AI disclosure and MIT license.

The root-README-linked `.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md` remains a navigation pointer to current build/release/Nix guidance and plan TODOs. The old snapshot body remains in Git history. Current plan checklists and verified code/tests govern implementation status.

No proposed architecture, benchmark or generated explanation establishes support for original trained image/audio/video, full GPU Qwen or production vLLM-class serving absent source-real tests.

## 5. Decision scope

The license/AI disclosure/contribution approach is **explicit user direction**, unlike proposed future model ABI and scheduling options. User authorization here concerns documentation, repository license metadata and public-facing contributor guidance—not driver permissions, downloaded model weights, external provider account access or live deployment.

Maintain original P00–P23 phase order, one Rust engine, one resource authority and one TaskCoordinator publisher.
