Author: Project owner
Co-authors: GPT-6
Owner: @pvl

# Native serving and request scheduling

## Implemented: first OpenAI HTTP/CPU router slice

- **One process:** optional `zerogravity serve` feature on original Qwen3.5 GGUF CPU only. Each canonical model path loads once under one immutable SourceIdentity and ResourceArbiter lease. Multiple aliases share original bytes/tokenizer; each request owns an independent StateRoot.
- **HTTP/1.1:** loopback-only; remote access must use a separate TLS/auth proxy. Optional `ZEROGRAVITY_API_KEY`. `GET /health` exposes configured CPU admission-budget counters; `GET /v1/models`; `POST /v1/completions` and `POST /v1/chat/completions`. Reject user chat strings containing raw Qwen role delimiters.
- **Supported semantics:** greedy temperature 0, top_p 1, n 1 and zero penalties, bounded max_tokens or max_completion_tokens; plain string chat/content only. JSON/SSE with finish reason and optional usage; strictly reject unsupported sampling/tools/structured output/roles, duplicate JSON keys and HTTP framing ambiguities. **Not full OpenAI API parity.**
- **Concurrency:** finite CPU worker threads, per-worker round-robin quanta of ≤32 prefill tokens or one decode token, bounded active and queued requests. Distinct sessions share immutable weights. One configured host-resource ledger charges weight residency and per-request state/scratch. This is a configured budget, **not measured available physical RAM**.
- **Cancellation:** dropped request receivers stop future worker quanta, including queued tasks. HTTP disconnect and SSE keepalives with deadlines, bounded response backpressure. CPU worker session is released on completion. No GPU work launched.
- **Bounds:** default 2 workers × 2 active requests per worker, 16 queued per worker, 64 connections, 1 MiB JSON body, maximum configurable context 8192/output 512 tokens. Distinct long-context accuracy limits still apply.

## Run

From the repository root, with approved GGUF/config/tokenizer already present:

~~~bash
cargo run --release --locked --offline -p zerogravity-cli --features serve -- \
  serve --config examples/serving/qwen-cpu.yaml
~~~

Or explicit model and resource caps:

~~~bash
cargo run --release --locked --offline -p zerogravity-cli --features serve -- \
  serve --model qwen3.5-0.8b=target/models/qwen35-0.8b/model.gguf \
  --workers 2 --active-per-worker 2 --max-context 512 --max-output 32 \
  --host-mib 4096 --listen 127.0.0.1:8000
~~~

~~~bash
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"Hello!"}],"temperature":0,"max_tokens":8}'
~~~

Set configured host budget responsibly based on physical RAM and other host/cgroup workloads. The server does not automatically measure all OS headroom. With `ZEROGRAVITY_API_KEY` set, every request requires an Authorization Bearer token.

## Not implemented / next

- vLLM-style fused continuous GPU batching, chunked cross-request GEMMs, paged KV/cache, GPU buffer completion/quarantine, cross-machine load balancing and any vLLM goodput parity claim.
- All sampling/grammar/reasoning/tool/audio/image/video/embedding/provider-specific API shapes beyond the above bounded subset; real pipeline/media execution, multi-tenant shared context, hot reload or persistent sessions.
- Gemma4/Qwen3.8 trained execution, SSD model inference, exact trained 27B StateRoot or tokenizer support.

Next: group compatible model/source/request-position tasks under one `MathematicalBackend` and physically admitted GPU batch plan. Preserve independent per-request StateRoots, finite occupancy and original BufferLease completion; benchmark p50/p95 TTFT/ITL and committed goodput against pinned vLLM under equivalent inputs.
