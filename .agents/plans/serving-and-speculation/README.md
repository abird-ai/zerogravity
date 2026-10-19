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

## Resident GPU OpenAI serving — accepted CUDA slice

- One `ResidentModel` per canonical GGUF; aliases share source.
  One independent `ResidentSession`/StateRoot per request and the
  existing physical service/queue/BufferLease readiness.
- Explicit `serve,cuda`, `serve,vulkan` or `serve,hip`. No CPU
  inference fallback. GPU greedy token feedback stays on the
  original device path; the host handles tokenization and text.
- One finite GPU worker, ≤8 active/16 queued. Compatible
  source/phase/frontier waves with ≤32-token prefill quanta and
  per-request decode; each member still executes a SEPARATE
  GPU program. `gpu_fused_batches=0`, not vLLM fusion.
- Shared CPU/GPU output owner handles EOS, UTF-8, SSE, usage,
  backpressure/cancel. Health distinguishes attempted/completed
  quanta, GPU forward API calls (not individual kernel launches),
  queued/active/canceled/error work.
- CUDA fail-fast prerequisite: `CUDA_PATH` contains runtime
  headers and CCCL device headers. Split installed Nix packages
  can be staged using `scripts/cuda-nix-include-view.sh`:
  link existing header paths; no SDK download/driver modification.

### Run qualified NVIDIA CUDA HTTP source

```bash
# Supply already-installed CUDA runtime and CCCL include directories
# as CUDART_INCLUDE and CCCL_INCLUDE; no packages are downloaded.
export CUDA_PATH="$(scripts/cuda-nix-include-view.sh \
  "$CUDART_INCLUDE" "$CCCL_INCLUDE" \
  "$HOME/.cache/abird/zerogravity/cuda-toolkit-view")"

# Put the locally installed NVIDIA driver and NVRTC library directories
# on LD_LIBRARY_PATH, then launch:
cargo run --release --locked --offline \
  -p zerogravity-cli --features serve,cuda -- \
  serve --config examples/serving/qwen-cuda.yaml
```

The GPU YAML profile specifies backend=cuda, workers=1,
active=2, context=512 and bounded provider/device memory.
Use your XDG cache root instead of HOME/.cache if configured.
See the CPU YAML example for unchanged CPU serving.
Local Renoir/gfx90c AMD launch hold remains intact.

### Physical evidence: RTX 3060 Laptop / CUDA, 19 Oct 2026

- Qwen3.5 0.8B: resident device model, HTTP 200, GPU-generated
  greedy token. Initial debug process readiness ~7.65s.
- Two requests, two aliases, one immutable source: both produced
  identical deterministic 2-token completions. 6 completed
  scheduling quanta, 3 source-compatible scheduling groups,
  4 successful model-forward API calls (NOT single kernel launches),
  0 failed, 0 fused, 0 active
  sessions after return. ~16.56s debug wall in first test.
- Disconnect while active: in-flight GPU step finished, session
  canceled and retired, no further launches; fresh session worked.
- `tools/serve-bench.py` tested 2 offered/2 complete SSE requests:
  4 usage-reported tokens, 20.02s debug wall, ~0.1998
  tokens/offered-second, p95 first streamed text ~19.78s.
  No pinned vLLM baseline or trained accuracy comparison;
  SSE chunk intervals are NOT verified per-token ITL.
- CUDA 12.9 split packages were already in Nix store. Vulkan
  builds/tests pass, but CubeCL exposed 0 usable local discrete
  adapters despite `vulkaninfo` seeing NVIDIA. No Vulkan physical
  acceptance, AMD launch or GPU reset performed.

## Pending: P12/P16/P22 vLLM-equivalent fused GPU execution

1. One source-bound batch-native `ForwardPlan` and
   `ResidentProgram` for B>1. Preserve each lane's immutable
   source, session/epoch/causal frontier and root publication.
2. Native multi-row fused GPU projection, norm, RoPE, GDN,
   attention and FFN kernels; one physical dispatch per stage
   covers multiple lanes, not serial member dispatches.
3. Device-resident paged hybrid KV + GDN/COW state with
   actual buffer leases, safe cancellation/quarantine.
4. Finite fair chunked prefill/decode and physical GPU/host/SSD
   occupancy, no speculative phantom memory credit.
5. Source-real CPU↔GPU full logits and StateRoot for B1/2/4/8;
   real per-token TTFT/ITL/SLO goodput benchmark against pinned
   equivalent vLLM V1 (same weights/quant/tokenizer/hardware).
   No performance parity claim before these gates pass.

Current GPU HTTP servicing is physically real and concurrent;
source-compatible waves are scheduler co-location ONLY.
