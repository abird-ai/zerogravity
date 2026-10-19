Author: Project owner
Co-authors: GPT-6
Owner: @pvl

# Native serving and request scheduling

Current P12: opt-in CUDA B2 HTTP prompt-tail/decode waves execute the
original native full-model ForwardPlan. Same-source/frontier eligible
rows share immutable weights; each retains its own SessionAuthority,
StateRoot, device packet and physical leases. Unmatched rows and
multi-token prefill use the accepted B1 GPU path. Default batch mode
remains `interleaved`.

Approved original Qwen3.5 0.8B/2B passed real release JSON/SSE,
trained EOS, unequal output limits, disconnect/survivor/replay and
internal CPU/B1 exact packets/all42 state domains. The earlier native
B2/B4/B8 engine diagnostics retain full248320-logit acceptance;
HTTP currently dispatches native B2 only. Wider/unequal-frontier
serving, full GPU prefill and matched vLLM performance remain open.
Exact scopes: [TODO.md](TODO.md), [FUSED_GPU_BATCHING.md](FUSED_GPU_BATCHING.md).

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

- Wider/unequal-frontier native GPU serving, paged KV/cache, cross-machine load balancing and any vLLM goodput parity claim.
- All sampling/grammar/reasoning/tool/audio/image/video/embedding/provider-specific API shapes beyond the above bounded subset; real pipeline/media execution, multi-tenant shared context, hot reload or persistent sessions.
- Gemma4/Qwen3.8 trained execution, SSD model inference, exact trained 27B StateRoot or tokenizer support.

Next: widen the existing native serving path to B4/B8 and unequal frontiers. Preserve independent per-request StateRoots, finite occupancy and original BufferLease completion; benchmark p50/p95 TTFT/true ITL and committed SLO goodput against pinned vLLM under equivalent inputs.

## Resident GPU OpenAI serving — accepted CUDA slice

- One `ResidentModel` per canonical GGUF; aliases share source.
  One independent `ResidentSession`/StateRoot per request and the
  existing physical service/queue/BufferLease readiness.
- Explicit `serve,cuda`, `serve,vulkan` or `serve,hip`. No CPU
  inference fallback. GPU greedy token feedback stays on the
  original device path; the host handles tokenization and text.
- One finite GPU worker, ≤8 active/16 queued. Default `interleaved`
  uses B1 graphs. Opt-in `--batch-mode fused` requires CUDA and
  ≥2 active slots: matching-source/frontier prompt tails and decode
  use actual B2 kernels. Other prefill uses ≤32-token B1 quanta;
  unmatched rows use B1. No padding stopped or blocked requests.
- Shared CPU/GPU output owner handles EOS, UTF-8, SSE, usage,
  backpressure/cancel. Credit-blocked GPU rows park before feedback
  with one retained packet; a 30s deadline bounds stalls. Ready-cohort
  cancellation discards every candidate before retrying survivors.
  Health distinguishes scheduler quanta, completed model-token work,
  actual kernel dispatches, native cohorts, committed roots and
  queued/active/canceled/error work; accepted output is not goodput.
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

### Initial interleaved evidence: RTX 3060 Laptop / CUDA, 19 Oct 2026

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

Detailed implementation/acceptance contract: [FUSED_GPU_BATCHING.md](FUSED_GPU_BATCHING.md).

## Native CUDA B2 HTTP acceptance — P12-B5

Implementation: `2677d8b`.

Release tests over BOTH approved original models prove native
source-alias JSON/SSE, exact independent CPU text/usage, real trained
EOS stop with EOS excluded from usage, one terminal/usage/Done,
active disconnect, survivor and fresh replay. Internal real GPU
worker tests additionally compare compact token ID/max bits and all42
state domains against CPU and B1, B1↔native packet handoffs,
unequal output limits, cancellation before feedback/after Ready,
actual bounded-channel parking and final reservation0.

| Original source | Completed native cohorts | Native projections | Native sampler stages | Discarded Ready cohorts | Final active/queued/failed |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5 0.8B | 9 | 1683 | 36 | 1 | 0/0/0 |
| Qwen3.5 2B | 8 | 1496 | 32 | 1 | 0/0/0 |

Cohort counts include actual completed work discarded before root
publication. They do not measure delivered tokens or throughput.
HTTP wire checks do not expose private StateRoot/lease contents;
those exact checks belong to the internal worker qualification.

For the optional native mode, use the same qualified CUDA environment
and resource caps, for example:

~~~bash
cargo run --release --locked --offline -p zerogravity-cli \
  --features serve,cuda -- serve --backend cuda --batch-mode fused \
  --model qwen3.5-0.8b=target/models/qwen35-0.8b/model.gguf \
  --workers 1 --active-per-worker 4 --max-context 512 --max-output 8 \
  --provider-mib 4096 --device-mib 5120 --listen 127.0.0.1:8000
~~~

## Pending: P12/P16/P22

1. Native HTTP B4/B8, unequal frontiers, dynamic compatible refill
   and full-prefix GPU prefill; qualify additional trained profiles
   and backends before advertising them.
2. Device-resident paged hybrid KV + GDN/COW state, source-safe
   prefix reuse, physical admission and cancellation/quarantine.
3. Fair prefill/decode/media occupancy and safe overlap across
   GPU/host/SSD; no phantom memory credit.
4. Actual p50/p95 TTFT/token ITL, committed SLO goodput, fairness,
   memory and dispatch measurements against a genuine pinned,
   source/quant/tokenizer/hardware-matched vLLM V1 environment.
   No vLLM executable/module is installed in the checked environment.

Earlier B0/B1/A1/B2 operator and serialized-cohort evidence is retained
in [FUSED_GPU_BATCHING.md](FUSED_GPU_BATCHING.md); its historical
limitations are superseded only by the later named acceptance scope.
