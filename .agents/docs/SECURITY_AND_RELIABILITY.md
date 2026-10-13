# Security, reliability, resilience and safe defaults

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


zerogravity accepts untrusted model artifacts, tokenizer metadata, prompts, images and potentially network requests. GPU drivers, memory mapping and high-memory MoE loads create host-safety risks beyond ordinary HTTP services. Security must be designed from the beginning, not added after fast kernels ship.

## 1. Threat model

Assets: host files/config/secrets, model checkpoints/provenance, GPU/CPU memory, user prompt/tool histories, cached token/recurrent state, network service/auth credentials, benchmark/model licensing, device stability and availability.

Potential adversaries:
- malicious/untrusted GGUF/safetensors/tokenizer/config/vision assets;
- unauthorized local/remote HTTP clients, very large or deliberately pathological prompts;
- code injected through user/tool content attempting to trigger host execution (not a model-runtime privilege);
- compromised network/distributed worker or stale RDMA memory key;
- crafted cache/snapshot file causing oversized allocations or state reuse under different model;
- untrusted optional plugin/kernel bundle trying to execute arbitrary host/device code;
- non-malicious workloads that exhaust physical RAM/GTT and hang an APU host.

## 2. Untrusted artifact parsing

Every input integer and offset is validated before allocating or dereferencing:
- limit tensor count, dimensional rank, string length and metadata nested structures;
- check all `count * elem_size`, offset+size, shard total and alignment with 64-bit checked arithmetic;
- reject overlapping or contradictory regions and duplicate mandatory tensor names;
- reject unrecognized quant type or unknown architecture rather than guessing;
- handle invalid UTF-8/template/tokenizer tables without UB;
- canonicalize local paths and apply user authorization rules to symlinks/traversal;
- no executing HF `trust_remote_code`, model postprocessing scripts or embedded executable payload;
- downloads to temporary files, verify source/checksum, atomic publish; protect from partial reads.

Use property/fuzz tests with ASan/UBSan where FFI is involved and Rust fuzz/prop tests for parsers. GPU kernel dispatch must not take unchecked dimensions from user model file.

## 3. GPU memory lifetime and physical safety

The GPU can read mapped/pinned host memory directly. A stale pointer or generation mismatch can corrupt model state and in severe cases destabilize driver. Enforce typed BufferLeases across IO/CPU/GPU events, checked state versions and no overlap between in-flight source/evict destination.

On Strix Halo: GPU-visible GTT may be larger than safely free RAM. Respect adjusted `MemAvailable`, `CmaFree` where relevant, OS/cache margin and worst-case prompt scratch. Physical benchmarks require watchdog and exclusive GPU use. Avoid causing host OOM/forced reboot during standard tests. Never auto-edit `amdgpu.gttsize`/TTM kernel parameters, disable IOMMU or change firmware UMA without explicit user action and hardware-specific warning.

Kernel spin waits for expert misses must have bounded progress/timeouts and failure notification; driver error results must propagate. No unsafe memory overcommit to try to fit 321B model.

## 4. Service exposure and request admission

Default bind localhost only; `--host 0.0.0.0` requires explicit choice and warning about authentication/TLS. Protect remote endpoints with authentication, rate limits, per-client token/context/quota controls and bounded upload sizes. CORS headers are **not** authentication.

Request parsers limit JSON nesting, string sizes, tool lists, multimodal bytes/pixel count, context tokens, max output, concurrent sessions and timeouts before allocation. A slow client should not retain unbounded GPU memory or block all request scheduling. Streaming may be cancelled safely after committed frontier.

No arbitrary file reading through `image_url`, local model loaders, tools or `/read` commands without explicit policy. Model serving and an optional agent tool executor are different security principals. If an agent is added, tools must be sandboxed with separate consent and filesystem/network limits.

## 5. Model and prompt privacy

Full rendered prompts, hidden reasoning, tool results, embedding data and KV/recurrent state can contain sensitive conversation data. Debug traces and durable prefix snapshots are therefore private by default. Use restrictive file permissions, configurable expiry/size limit, versioned cache identity and opt-in persistence. No uploading benchmark/model data or prompts to external telemetry by default. If content hashes are used, avoid logging material that allows easy correlation with private inputs.

When exposing traces, redact credentials and secret request fields. Hidden reasoning must not leak through streaming even if model-native template encodes it in output.

## 6. Snapshot integrity and cache poisoning

Prefix cache keys must bind model weights, quant, tokenizer/rendered message structure, multimodal fingerprint, position/numeric policy and state schema. A hash collision or corrupted metadata cannot result in unsafe arbitrary state restore. Serialization bounds check offsets and checksums before memory use, with no persisted raw device pointers. Reject snapshot from different schema/backend layout unless a tested conversion exists. Cached state persistence may need encryption based on deployment threat model (OPEN).

If snapshot loading fails, reconstruct from trusted token history or reject; never continue with half-restored recurrence/KV.

## 7. Distributed transport

RDMA/TCP protocols require authenticated peers or explicitly restricted trusted deployment, checked frame lengths and model/plan/quant identity handshake, monotonically sequenced operation epochs, bounded allocations and timeouts. Registered memory region keys must be scoped and revoked at disconnect. No default public unauthenticated RDMA/TCP cluster on shared internet. Remote worker failures require atomic frontier failure and rebuild, not accept incomplete allreduces.

Do not log credential/session payload or NIC keys. Malicious or stale peer commands cannot induce arbitrary GPU memory writes; validate region keys, offsets and expected tensor shapes.

## 8. Plugins and generated kernel trust

A Rust-authored GPU kernel may compile to device code with potentially serious memory safety consequences. Do not auto-download and execute arbitrary third-party kernels or executable compiler IR from model repositories. Ship built-in signed/built code; if plugins added, define a strict user-consented trust/sandbox/signature scheme, versioned operator ABI and resource limits. Runtime compiler inputs must be validated and code cache isolated to trusted/user-owned directories with checksums.

A model checkpoint is **data**, not an executable plugin.

## 9. Reliability and operational state

- crash-safe model artifact download and cache snapshot writes via temporary file + fsync/atomic rename where needed;
- device error marks provisional state invalid; no silent continuation;
- empty/masked attention selection handled as defined, not NaNs or uninitialized memory;
- bounded IO queues and host-mapped event rings with credits/backpressure;
- strict startup memory-fit and fallback plan selection;
- thread and device teardown waits on in-flight events and cleans cached slabs;
- no incompatible model hot reload while active session running;
- independent operation metrics and structured errors, not hung promise/worker.

## 10. Supply chain and licensing

All third-party code (GGML quant code, Gufo, Maya, DwarfStar, CUDA snippets), model checkpoints, conversion recipes and evaluation datasets require license/provenance review before redistribution. DwarfStar license was inspected as MIT with GGML notices, but other repos' licenses must be rechecked at pinned commit. Some engine sources are not public; do not copy or derive from proprietary implementation. Document source/attribution when independently porting legal algorithms into Rust.

Avoid installing unofficial external binaries or unsigned models without review. Nix/Cargo lockfiles and release SBOM are proposed controls. Avoid committing large proprietary checkpoint data or access tokens in test fixtures.

## 11. Reliability testing before release

**Model inputs:** fuzz GGUF/safetensors/quant scales/shards and broken vision encoder; no crashes/hangs/excess RSS.  
**Runtime:** simulate out-of-memory, HIP device lost, failed event, cancelled kernel, poisoned buffer generation and per-session race.  
**Storage:** partial expert read, corrupted Engram row, truncated snapshot, disk full, read-only file, temp rename failure.  
**Serving:** malformed JSON, 1000 simultaneous invalid requests, tool boundary parse, HTTP disconnect, auth rejects, stream backpressure, large uploads.  
**Distributed:** wrong model handshake, dropped frames, TCP half-close, RDMA reject, worker crash during speculation.  
**APU:** physical memory watchdog under increasing session/context/scratch pressure; admit/reject predictably without system-wide OOM.  
**Numerics:** every safety fallback returns correct result (if safe) or clear error, not plausible invalid tokens.

## 12. Release security checklist

- [ ] Defined model file trust and network resolver policy.
- [ ] All external source/data licenses reviewed.
- [ ] No unchecked parser arithmetic or format downgrades.
- [ ] Public HTTP exposure is explicit and protected.
- [ ] Model-state snapshots private, bounded and versioned.
- [ ] Physical RAM admission and APU watchdog exercised on hardware.
- [ ] Fault injection on expert slots, snapshots, kernels and cancellation.
- [ ] No execution of arbitrary Python model/plugin scripts.
- [ ] Distributed transport identity/credential and fail-closed policy.
- [ ] GPU memory/FFI unsafe blocks audited and sanitizer tests recorded.
- [ ] Secrets/model traces excluded from CI and release artifacts.
- [ ] Performance and quality claims link reproducible evidence.

## 13. Open security decisions

TLS/auth implementation and tokens, snapshot encryption/retention defaults, model download allowlist, kernel/plugin extension policy, optional agent filesystem/network grants, remote execution deployment, release signing keys and public bug reporting/security contact. These must not be silently assumed done.

## 2026-10-10 integrated v2 addendum — compiler, async executor and plugin threat boundaries

The new pipeline creates trust boundaries in addition to the original artifact reader/API: `ArtifactReader` may parse malicious model files, `SemanticOp` adapter may define unknown math, `RewritePass` may perform illegal state change, `KernelProvider` may generate unsafe ISA, `CIR` may schedule stale device/SSD buffers, and distributed/LoRA/vision paths may reuse state across tenants. Only trusted built-in Rust code providers should execute initially; dynamic third-party executable plugins require source/signing/trust and driver memory isolation policy before exposure. Model files are **data**, never permission to download/run HF Python remote code.

**CIR security:** all allocations consume ResourceArbiter reservations, every async DeviceEvent/Timepoint has source epoch and buffer lease generation, and expert IO `pread` validates artifact identity/offset/bytes before publish. Graph capture must not dereference host pointers whose lifetimes expired. No infinite GPU spin while waiting for host ring completion; host service error/timeout leads to safe provisional abort. Model cache and prefix snapshot lifetimes are private, tenant-salted and copied/serialized only under validated state schema.

**Binary format correction:** research's listed "48-byte" disk-KV header is actually 54 bytes with incompatible five-character-u16 magic. Reconstruct a new checked versioned format and fuzz it before writing or accepting snapshot files; never persist process pointers. Hash identity must include exact token and model/quant/pos/adapter/modality/renderer state, cryptographic digest rather than bare SHA1 prompt bytes.

**Source language / license risk:** historical vendored HIP/C++ kernel proposal must not bypass user Rust-owned source requirement. Optional Rust GPU tile/SIMT compiler backends each bring version/ISA attack surfaces; audit codegen artifacts, driver load, unsupported dynamic compiler cache and kernel source trust. Using vendor GPU math via FFI is an open user policy, not permission to load an entire untrusted inference runtime.

**Control API:** default localhost, bounded authentication/authz, model download checksum, prompt/hidden reasoning privacy, tool/MCP permission separation, per-tenant cache namespace, encrypted/permissioned durable state policy, no remote RDMA without explicit trust. A new device/IO transport must pass event/lease and peer-epoch fault tests as well as ordinary kernel parity. Add a compiler pass negative corpus that intentionally moves `ConsumeRng` or `GdnTransition` across commit and proves rejection.


## 2026-10-12 integration — GPU+XDNA2 SoC hard-hang risk and NPU artifact trust

**Critical external report:** the user's Reddit source `1x3gtdh` and `mikealanni/gufo` fork report **whole-machine freezes / machine-check errors** when Radeon GPU fabric clock changes while the XDNA2 NPU operates concurrently. Fork workaround pins `pp_dpm_fclk` to top, configures XRT NPU performance mode, disables runtime suspend and sends keepalive; 12 clean author test runs **do not prove** general production safety. Some community reports note increased idle power and BIOS/AGESA behavior.

**Safe default and rule:** NPU offload stays **disabled by default** in zerogravity until an exact hardware/firmware/driver and compiler program combination passes controlled physical qualification. Detect and expose `Supported/Experimental/Qualified/Unsafe` capabilities, with typed refusal/fallback GPU-only. **Never issue privileged automatic clock/power/BIOS/sysfs/systemd/IOMMU modifications.** A diagnostic may describe manual opt-in commands, their scope, increased power, reversibility and risks; auto mode must not silently lock clocks. Avoid testing machine-check triggers on an everyday workstation: use an isolated, recoverable bench box and explicit user permission. A daemon timeout cannot recover a frozen SoC.

**XRT/HIP memory and malicious bytecode:** XRT `.xclbin` and instruction stream are **executable device programs** with potential DMA and fault risks. Only trusted bundled/checksummed hardware-qualified programs with declared source/compiled ISA provenance; a downloaded GGUF must never specify an executable kernel URL. A Rust runtime wrapper around Gufo's C++/Python AIE source is **not** proof of Rust-owned kernels; honor source policy and verify licensing per file. GPU-pinned BOs imported into XRT are resources requiring generation-tagged ownership and explicit device sync. Failed/timed-out commands may keep DMA alive; quarantine pages rather than freeing them. Bound BO dimensions, stride, program size, queue depth, device temperature and power.

**Security tests:** software-level simulations for stale buffer generation, XRT BO map failure/short import, HIP callback ordering, worker crash/cancel, partial NPU writes; physical safe tests for driver/firmware contention, 24h run, boot/resume/thermal/power and fallback. CPU/GPU model remains correct with `--npu off` or absent XRT, and no new network or filesystem privilege becomes necessary for normal serving.

## v4 refinement — one logical root and output delivery security

Every provisional state update is isolated; **no independently mutating per-StateDomain commit** or CIR device-side session publish exists. TaskCoordinator checks prepared versions of KV/recurrent/indexer/RNG/media/grammar and required GPU/IO fences then atomically swaps the logical StateRoot, after which a sequenced committed output may be attempted for delivery. A lost remote ACK means exactly-once client receipt is **not** guaranteed; optional replay cursor requires explicit authenticated retention and dedup. Never expose tentative tokens, audio frames or latent state.

The global ResourceArbiter is admission/pressure authority; local device pools own fast allocations and event-backed retirement under its reservation, with stale generation quarantine. External memory pressure can still occur; fail safely rather than claiming OS memory cannot change. No model-supplied code, device bytecode or privileged agent actions in untrusted inputs. GPU/NPU power/fabric risk is a hard device qualification veto; no automatic root/clock/IOMMU/firmware setting. Gradient/parameter updates (if ever supported) require new immutable ParameterVersion and inference session isolation.

## v4.1 — compiled graph/wafer programs, remote sessions and license trust

A graph compiler or programmed fabric produces **executable device code/program images** with model/data access; it is not safe just because our Rust host passed a typed program to the SDK. Verify provenance/driver/compiler/program signatures or trusted build source, op/shape/quant validity, buffer limits and device safety before load. Runtime device access may require vendor closed binaries/FFI; user's Rust-authored kernel source policy still applies separately and remains open for graph targets—**do not claim a native Rust TPU/wafer/LPU compiler exists without proof**.

A managed accelerator service that only offers model input/output and no trusted snapshot/KV/RNG/gradient and effect semantics is not a Native `RegionExecutionProvider`; classify as an external authenticated **TaskServiceAdapter** with explicit user consent, tenant/session/output contracts, remote data-privacy and egress/security cost. Do not leak prompts, checkpoint weights, embeddings or media to a remote device implicitly. Device program/collective groups must be fenced on failures and versioned against actual model/driver/firmware; no exactly-once network delivery claims. Safe host and internal firmware/power restrictions still override optimizer placement.

An opaque graph with unrecoverable in-place state cannot be selected for general speculative/fork/resumable session execution. A stateless encoder subgraph can still qualify with correct input/output and client privacy constraints. Unknown op/quant or unsafe toolchain => fail closed/fallback. Source citations stay in prior-art research.

## v4.2 — trusted network peers, tunnel isolation and remote state safety

**Explicit enrollment before any remote execution**: each peer has cryptographically authenticated identity, tenant/model/weight rights and allowed compute/data classes. Tunnel discovery and IP reachability do not imply access; a new node cannot execute arbitrary binaries, host shell commands or read a model/store outside approved artifacts, nor can the runtime implicitly export prompts, hidden activations, KV cache, embeddings, audio/video or checkpoint weights. Remote cache/task placement must honor source licenses and sensitive-data policy.

Control/metadata and bulk tensor state transfer are **distinct logical planes** with identity, versioned bounded frames, deadlines/idempotency, encrypted transport and optional physically qualified RDMA. A private WireGuard-type tunnel does not provide GPU RDMA or reliable low RTT; direct/relay/cryptographic cost must be measured. Registered RDMA/RNIC memory keys, GPU pointers and tenant secrets cannot be in logs; source/destination memory lease generations and DMA fence must remain valid through timeouts. Remote machine compromise is a separate trust assumption even if link encrypted.

**Single session publisher across network:** remote workers Prepare state, coordinator root contains versioned owned StateShard refs and epoch; only it publishes. Cluster membership/lease fencing prevents old rank/coordinator writes after partition; no false exactly-once client network delivery or durable transparent failover promise. Without verified quorum/heartbeat/checkpoint, fail-stop and reconstruct last known committed state safely; session availability is conditional. Cache directory entries can be stale hints and must be revalidated. Missing transfer bytes, bad model/version or slow peer cannot become a committed target token. [Remote protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md).

## v4.3 model router, registry and API security

**Serving listener defaults to loopback**, and one local `run model` needs no public server. A remote/API listener requires deliberate bind plus authentication/TLS/proxy policy, tenant quotas, model alias visibility and native admin separation. A local SDK sending a dummy key for syntactic compatibility does **not** authenticate a remote caller; never ignore externally presented invalid keys or log credentials. Admin model pulls/imports/profiles/loaded model controls must not be available to every chat completion caller.

Model names/aliases can reveal private checkpoint research and tenant data. Expose only authorized `/v1/models` aliases; don't reveal local artifact file paths, peer/tunnel addresses, quant private sidecars, GPU topology or last user prompts to an unauthorized client. Model source import/pull requires user intent/allowlisted source, content hash/license/size and cannot run arbitrary Python/model code. Unknown names never trigger automatic cloud model guessing/download; ModelRouter must not leak source prompt/hidden KV/weights to an unauthorized remote peer or TaskServiceAdapter.

Model changing mid-session is a correctness and security risk: pinned immutable checkpoint/quant/adapter/template/alias revision and StateCompatKey remain valid through streaming, request config can't bypass numerical/tenant/memory controls. Router load/unload must hold BufferLease/event fences and shouldn't evict an active unsnapshotable session. Resource exhaustion attacks via model-switch thrash and many cold loads are bounded by queue/load dedup/headroom/per-tenant fairness, with typed rate/overload errors.

OpenAI-compatible fields don't grant inference code tool/web/file execution; **tool calls are structured generated data only** until an external app separately authorizes and executes a tool. Stateless Responses first; durable store, previous-response state and cloud built-ins only with explicit implemented security/privacy/conformance. Local persistent conversation memory is privacy-first opt-in. CORS/media URL fetch/JSON/body sizes and streaming backpressure validated. [Full CLI/API contract](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md), [router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md).
