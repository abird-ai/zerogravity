# Contributing to zerogravity

You're welcome here. We're building zerogravity in the open, and we'd love to build it with you. We believe the best way to collaborate is through human-led design and AI-assisted implementation, guided by clear plans, tests, and shared decisions.

## Best ways to contribute

- **Design and architecture:** open an issue describing the problem, alternatives, correctness constraints, and expected impact. Propose a plan at `.agents/plans/<plan>/TODO.md` and link it from `.agents/plans/TODO.md` so others and AI coding agents can reason from the same decisions.
- **Contained implementation:** start with a short `.agents/plans/<plan>/TODO.md` covering the provenance commit, implementation steps, required tests and unfinished work. Parallel agents keep their own checklist under `.agents/runs/<plan>-<agent-id>/TODO.md` until the coordinator integrates it. Then submit a focused PR with source references, tests, and measured results where applicable.
- **Research and verification:** contribute reproducible model/source analysis, failing cases, hardware measurements, counterexamples, or simpler architectural options. Clearly separate reproduced evidence from hypotheses.
- **Documentation fixes:** direct small PRs are welcome; a dedicated plan is unnecessary for a typo or broken link.

## Engineering principles

Keep one Rust-native engine, exact model/quantization semantics, one session-state publisher, and safe physical resource ownership. Prefer minimal, composable changes over new infrastructure. Never label planned support as implemented or benchmark results as source-equivalent without evidence.

**AI-assisted contributions are welcome.** Share the design decisions, tests and validation evidence; contributors aren't expected to manually author or inspect every line. Humans remain responsible for the goal and acceptance criteria, and AI can help with implementation and iteration.

Thank you for helping make fast, open inference better for everyone.
