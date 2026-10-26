# Research Track E — Video, Temporal Diffusion, Synchronized Audio-Video and Frontier Providers

**15 October 2026 · design/source research only; no original video model executes in Zerogravity.**  
**Scope:** Wan 2.1/2.2, original/1.5 dual-stream video DiTs, asymmetric-attention video DiTs, coupled audio/video model families and hosted frontier video providers. User intent includes both open native video and hosted frontier providers.  
**Dependencies:** v5.1 typed media clocks/model assembly, source-registered solver and joint StateRoot/CommitGroup; realtime-serving for live AV output.

## 1. Verdict

Video workloads require the same shared execution spine, but they stress it differently from text:
- a denoising transformer iterates over **spatiotemporal latent grids** rather than a text token prefix;
- source attention may be temporally sparse, sliding, asymmetric or dual-stream;
- a 3D VAE introduces temporal compression, receptive-field overlap, pixel/frame reconstruction and potentially separate tile state;
- Wan 2.2 expert choice is **timestep/denoising-phase routing**, not token-level sparse FFN MoE;
- Coupled audio/video model families and advanced hosted systems can generate coupled **audio and video** with cross-modal neural state;
- output can be valid only at complete decoded frame/segment and synchronizable audio clocks, not at arbitrary latent iteration.

No universal FPS, fixed video latent temporal compression or universal sampler assumption is safe. TaskDriver and ModelAssembly are exact per checkpoint while CUDA/Vulkan/GPU tensor kernels and physical resource ownership are reused.

## 2. Source-backed family matrix

| Model/checkpoint | Distinct semantic path | Required native extension | Primary source |
|---|---|---|---|
| Wan 2.1 T2V/I2V 1.3B/14B variants | spatiotemporal diffusion/DiT + source video VAE, prompt conditioning | temporal latent positions, source flow solver, 3D VAE, image-reference conditioning, exact stages | https://github.com/Wan-Video/Wan2.1 |
| Wan 2.2 A14B / TI2V variants | high-noise vs low-noise denoising-phase experts selected by timestep; model variants differ | step-phase expert/sigma switch, verified phase-resident experts, source 3D VAE/schedule | https://github.com/Wan-Video/Wan2.2 |
| Original dual-stream video DiT | text→large conditioner, dual-stream→single-stream video DiT, causal 3D VAE | distinct attention phase, temporal compression/reconstruction and source prompt optimization | — |
| Dual-stream video DiT 1.5 | ~8.3B DiT, 3D causal VAE (4 temporal/16 spatial compression), selective/sliding tile attention and optional SR | exact sparse attention masks, causal/tiled VAE, SR component and phase+latency planning | — |
| Asymmetric-attention video DiT | ~10B AsymmDiT, non-square QKV/output projections, large visual stream, temporal AsymmVAE | asymmetric hidden dimensions, joint-attention layouts, temporal 3D VAE and context-parallel candidates | — |
| Coupled audio/video dual-stream model | audio and video **coupled** asymmetric DiT streams, two source VAEs, vocoder, bidirectional cross-modal attention | two latent clocks/time-dependent cross-attn, exact sync barriers/AV state root, separate decoder placement | — |
| Hosted video generation service (3.1 and related) | hosted generation of video/audio via long-running API jobs; no public exact neural source | explicitly authorized TaskServiceAdapter with job status, asset retrieval and rights | provider documentation |
| Hosted creative media service | hosted creative media API with async submission/status | provider task lifecycle, auth, references/media asset boundaries, cost/privacy | provider documentation |
| Hosted joint audiovisual service (1.5 Pro) | closed audiovisual generation; public report describes joint AV denoising branches | hosted service today; source-independent joint A/V solver contract for future verified checkpoint | provider documentation |
| Hosted creative video service | hosted creative video/storyboarding/AV generation | provider API version, async jobs, input/reference media and rights; internal neural math Unknown | provider documentation |

**Source distinction:** The requested frontier models are **capability/API targets**, not verified open native weights. Different video model releases have distinct licenses and potentially incompatible VAE/tile/sampler schemas. The listed upstream author claims are not Zerogravity measurements.

## 3. Video value and clock contract

~~~rust
struct VideoInput {
    frames_or_packets: TimedVideoValue,
    fps: RationalRate,                    // e.g. 24000/1001, 30000/1001
    orientation: Orientation,
    color: ColorSpaceDescriptor,
    track_epoch: TrackEpoch,
}
struct VideoLatent {
    batch: u32, latent_channels: u32,
    temporal_length: u32,
    spatial_width: u32, spatial_height: u32,
    vae_identity: VerifiedDecoderId,
    temporal_compression: VerifiedTemporalStride,
    tile_plan: Option<VerifiedTileGeometry>,
    solver_step: SolverCursor,
}
~~~

Pixel/latent/media packet timebases differ: true image frame count, temporal VAE latent frame number, solver step index and codec packet DTS/PTS are separate dimensions. For 30000/1001 FPS, a single frame spans 1001/30000 s, not a floating 33.33ms rounding repeated forever. Audio 48k samples cannot be aligned by assuming integer samples/frame. Resampler/codec delay and fractional remainder must participate.

Temporal VAE output cannot be considered final until receptive-field overlap and crop on all necessary adjacent windows are complete. Frame set is publishable only when all component readiness events and source numerical/output constraints pass.

## 4. Source-valid video-generation execution trace

1. Validate task kind (T2V, I2V, V2V, TI2V, extension, keyframes), exact checkpoint variant, aspect, number of frames, total decoded byte budget, and license.
2. Decode input/keyframes/references with exact color/alpha/orientation; bind source processor identity and references in assembly.
3. Encode text and image conditioning via verified source encoders; prompt rewriter is separate component (and privacy/egress policy if remote).
4. Seed initial spatiotemporal latent grid using the exact registered noise RNG and scheduler config; no silent replacement of seed or time bounds.
5. Run DiT/flow/denoiser iterative steps, preserving timestep/sigma order, sparse mask, cross-modal attention and source guidance.
6. Under Wan2.2 choose high- or low-noise expert according to declared source switch threshold and each step's actual timestep. Do not choose expert by token/latent index.
7. Source VAE decodes valid latent temporal windows with causal/tile overlap, optional registered upscaler and pixel transform.
8. Encode output video packets (and optional audio) under source target frame rate, stream codec, color and requested container; track DTS/PTS and mux metadata.
9. TaskCoordinator publishes complete, playable frames/segments only after all required model/codec events Ready, and only in correct output generation/track epoch.
10. Long job status/progress is independent of model state and transport: queued, running, completed/canceled/failed/unknown; no fabricated partial final video before VAE/mux completion.

## 5. Wan 2.1 specifics

- Source spatiotemporal DiT/3D VAE with T2V/I2V variants; validate exact trained length/resolution and input image conditioning.
- Denoise/rectified flow schedule is model/variant specific and must be captured by solver identity.
- Device-native step loop may avoid repeated host roundtrips only while preserving source RNG/timestep/guidance and bounded latent/event memory.
- VAE time dimension may require padding/extra frame semantics at start/end; shape guard must include causal window and reassembly crop.
- Long video with high resolution requires tile plans, temporal window and offload accounting; no model "fits in VRAM" claim based on weights alone.

## 6. Wan 2.2 denoising-phase MoE

Wan 2.2 introduces MoE by splitting the denoising trajectory into specialized experts with high-noise and low-noise responsibilities. This is **whole denoising-phase** switching, unlike LLM sparse FFN expert selection per token.

Required source semantics:
1. Registered timestep/sigma schedule in TaskDriver.
2. Source-defined threshold/phase selection rule; no heuristic "choose the expert that is currently warm" override.
3. Proper dual expert weight source identities, potentially different LoRA/quant or model variants.
4. Residency pipeline can prefetch the later expert when schedule safely predicts phase, but cannot evict an earlier expert while GPU buffers using it remain pending.
5. Actual memory plan includes transition spike (both expert sets may be simultaneously resident), not merely max(single expert).
6. Denoiser output continuity and RNG across phase boundary must be source-equivalent; no ad-hoc blending or interpolating the expert parameters.
7. TI2V variants that are dense do not automatically use the same MoE plan.

**Negative:** With source timetable [high,high,low], a worker executing high at the last step should fail the source variant quality gate even if output visually plausible.

## 7. Dual-stream video DiT original and 1.5

Original model combines textual conditioning with dual-to-single stream multimodal video transformer and causal 3D VAE. The 1.5 source highlights:
- selective/sliding tile attention (SSTA): attention mask/selection is a mathematical constraint; cannot silently replace with dense attention whose outputs differ numerically or a different sparse window;
- causal 3D VAE compression and stateful tile/overlap behavior;
- optional super-resolution and distillation steps are separate component/quality mode;
- source attention backend/FP8 GEMM can be optimized only within validated output numerical/quality contract.

Proposed specialization: source-trained sparse-tile region with precompiled allowed tile shapes and safe fallback. High-level SparseAttn registry needs explicit selected block IDs/mask and spatial/temporal axes; a generic "local window" key is insufficient.

## 8. Asymmetric-attention DiT and AsymmVAE

The original asymmetric-attention model has unequal text and video hidden widths, non-square QKV and output projections, and asymmetrically sized visual/text subnets. This is a direct test of composability: a runtime with an implicit equality constraint on cross-modal hidden dimensions cannot support it correctly.

- Semantic model operator must allow rectangular QKV and cross-stream attention projections.
- Source tokenizer/T5-XXL text conditioner and video VAE are separate verified components.
- 3D VAE temporal/spatial compression and causal decode/dense attention masks are checkpoint-specific.
- Context/tensor parallel candidate requires correct cross-device rank/ordering, scratch and source numerical proof; not inferred from vendor benchmark claims.
- Precompiled shapes can specialize common spatial/temporal sizes without editing TaskCoordinator or public HTTP API.

## 9. Coupled joint audio/video, exact synchronized group

Source documentation describes **asymmetric dual-stream audio and video DiT** with bidirectional attention, audio and video VAEs, vocoder and Gemma-3-based text conditioner in certain source variants. Internal model dimensions must be loaded from exact checkpoint rather than copying one technical paper's parameter split.

~~~text
Text + optional image/audio/video references
       -> verified conditioner embedding(s)
       -> video noisy latents, video spatial/temporal positions
       -> audio noisy latents, audio timebase and neural codec representation
       -> JOINT dual-stream transformer iteration:
             video self-attention + audio self-attention
             text conditioning + bidirectional A↔V cross-attention
             timestep-conditioned heads / source solver
       -> synchronized complete audio+video latent group
       -> 3D Video VAE + Audio VAE/vocoder
       -> aligned complete A/V packet groups with rational PTS
       -> ONE TaskCoordinator synchronized PublishRoot
~~~

Neither output may publish as independent final AV if source requires cross-attention through an unfinished iteration. The task frontier includes (video latent timestep, audio latent timestep, video PTS, audio sample/frame cursor, decode readiness watermark) and a CommitGroupId with legal cross-track readiness.

**Important counterexample:** generating Wan video and attaching unrelated Kokoro speech afterward is a valid *different multimodel application* but not a native inference implementation of the coupled A/V model.

## 10. Frontier proprietary video services

A public hosted report describes joint audiovisual diffusion; exact internal neural source weights/math are not available in this design inventory. Hosted adapters preserve:
- request model/version, text/images/references, duration/frame rate/audio flags and license/rights;
- async job id, progress, expected polling intervals, provider output asset ref and expiry;
- upload staging and authenticated callback/webhook verification;
- idempotency keys, rate quotas, cost controls and unknown status after connection failure;
- no hidden external egress or implied local GPU/VRAM budget.

Other hosted providers similarly provide API observable semantics, but not a native executable SIR recipe unless separately verified open weights exist. TaskServiceAdapter must not claim numerical equivalence to local Wan or coupled A/V.

## 11. Physical memory and QoS

Whole video assembly budgets must cover text/vision conditioner, denoiser experts, VAE/codec, temporal state, CFG branches, latent+tiling scratch, video+audio output packetization, buffers and physical transfer leases. In mixed workloads prioritize interactive duplex audio at bounded soft deadlines without allowing unbounded video starvation; schedule video at **legal solver-step boundaries**.

Cost/perf metrics:
- first valid playable video frame/segment latency, not just first noisy latent;
- fps after reconstruction, real frame timestamps and A/V sync skew;
- peak real HBM/VRAM/UMA usage, offload and multi-device transfer;
- noise-step latency, denoiser/latent residency, VAE tile cache hit and SR overhead;
- overall video quality on pinned source prompts/seeds and desired resolution;
- context/temporal window drift and looping boundary artifacts;
- cost, job egress and retries for hosted services.

Any "realtime video" request must have a measured admit/refuse policy; a 50-frame batch finishing eventually is not real-time interactive video.

## 12. Test matrix for independent agent implementation

**MM-VID-01:** Wan1 source temporal latent shape/3D VAE reconstruction with exact final frame count and crop.  
**MM-VID-02:** Wan2.2 source fixed high/low-noise phase transition and dual expert residency; wrong step expert rejected.  
**MM-VID-03:** source original dual-stream video DiT dual→single stream transition and conditioning masks, valid compiler region split.  
**MM-VID-04:** Dual-stream video DiT 1.5 SSTA mask/selected tiles equality against slow reference; no silent sparse-mask relaxation.  
**MM-VID-05:** Asymmetric-attention DiT non-square text/video projection and source-specific VAE temporal compression; rectangular dimensions valid.  
**MM-VID-06:** Coupled two-stream A/V solver with audio/video cross-attention dependencies and legal A/V CommitGroup barrier.  
**MM-VID-07:** 48k PCM + 30000/1001 FPS video duration alignment over 60 minutes, exact residual arithmetic.  
**MM-VID-08:** slow codec/mux returns backpressure, prepared video frames retained but not exposed as final.  
**MM-VID-09:** cancel one video denoise task after physical submit; no unrecovered buffer, preserved old StateRoot.  
**MM-VID-10:** 5 video jobs + live ASR/TTS and text requests share physical GPU budget without false QoS guarantee.  
**MM-VID-11:** hosted video provider timeout after create reconciles idempotent job; no duplicate billable submission.  
**MM-VID-12:** streamed video no invalid GOP/packet timestamps; clock discontinuities are explicit.  
**MM-VID-13:** model source decoder/source font/text/video language license verified independently of repository.  
**MM-VID-14:** artifact hot reload cannot reuse old temporal VAE state/codec buffers with new semantics.  
**MM-VID-15:** cross-model Wan video + external Kokoro TTS is identified as compound task, not a native coupled A/V model.

## 13. Implementation order, not a changed user model priority

1. Source-free spatiotemporal latent/3D VAE contracts and tiny fake noise-step loop.
2. One small Wan2.1 or similar open-video original checkpoint with source-trained CPU reference; favor feasible shape/hardware.
3. Wan2.2 two-expert denoising transition and actual measured residency/physical GPU scheduler pressure.
4. Asymmetric attention and dual-stream video DiT sparse tile masks, separate numerical equivalence tests.
5. Coupled joint synchronized audio/video for real coupled generation, not a post-hoc mux.
6. Hosted provider adapters independently behind explicit authorization.

No step implies original GPU Qwen3.5/Gemma4/Qwen3.8 implementation priority was altered.

## 14. Primary sources reviewed 15 October 2026

- Wan2.1: https://github.com/Wan-Video/Wan2.1
- Wan2.2: https://github.com/Wan-Video/Wan2.2
- Original dual-stream video DiT source model.
- Dual-stream video DiT 1.5 source model.
- Asymmetric-attention video DiT source model.
- Coupled audio/video source model.
- Coupled audio/video component architecture.
- Hosted video generation provider documentation.
- Hosted creative media provider documentation.
- Hosted joint audiovisual provider release documentation.
- Hosted creative video provider API documentation.

No video models were downloaded or run. Release, license, Tensor role, scheduler and artifact source code must be revalidated at implementation time.
