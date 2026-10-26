# Research Track C — Text-to-Speech, Neural Audio Codecs, Voice and Full-Duplex Conversation

**15 October 2026 · research/design only.** This track owns source-specific phoneme/prosody/acoustic/codebook/codec semantics and full-duplex state. Serving transport is separately specified; no model is claimed to run natively in Zerogravity now.

## 1. Verdict

A universal "text → wav" routine conceals at least four radically different implementations:
1. **Duration/prosody acoustic synthesis** (Kokoro): G2P+prosody+vocoder without decoder LM.
2. **Autoregressive acoustic codebook generation** (a dual-AR codebook model, some Qwen TTS): one or several nested time/depth loops with a decoder.
3. **Streaming acoustic generation with flow/diffusion chunks** (VibeVoice, model-specific Qwen paths): solver loops inside acoustic frame batches.
4. **Native full-duplex audio-language models** (a native full-duplex speech model, Qwen-Omni): model simultaneously ingests ongoing speech and generates speech/text; not necessarily an ASR→text-LM→TTS sequence.

Each is a trusted TaskDriver/ModelAssembly under the same source semantics, physical buffer/event leases, StateRoot and media serving system. Do NOT create a separate TTS backend server per family.

## 2. Primary source and method matrix

| Exact model/variant | Source method | Mandatory model/codec boundaries | Primary references |
|---|---|---|---|
| Kokoro 82M / voice variants | style-based text/phonemes, prosody and duration modeling, acoustic synthesis/ISTFT vocoder | an independent G2P engine or verified alternative, phoneme tokenization, speaker embeddings, duration/F0, PCM reconstruction | https://huggingface.co/hexgrad/Kokoro-82M |
| Dual-AR speech model (source variant) | Dual AR: slow temporal transformer predicts primary codebook, fast depth transformer predicts nine residual codebooks; 10-codebook RVQ codec ~21Hz | nested temporal/depth clocks, codec frame completion, voice/prosody instructions, waveform decoder | — |
| Qwen3-TTS | variant-specific streaming text/audio decoder; Qwen acoustic tokenizers 12.5 Hz/25Hz and potentially multi-codebook/flow chunk reconstruction | token frames+sample rate, codebook depth, speech tokenizer checksum, chunk/overlap, model-specific voice conditioning | https://github.com/QwenLM/Qwen3-TTS |
| VibeVoice TTS | long-form speaker-consistent generation with source-specific acoustic decoder/flow | 4-speaker or variant-defined voice state, long-duration chunking, semantic acoustic state | https://github.com/microsoft/VibeVoice |
| VibeVoice-Realtime 0.5B | streaming text input, low-latency acoustic synthesis, checkpoint-specific codec/flow | incremental text cursor, bounded output chunks, model-specific lookahead/reconstruction | https://github.com/microsoft/VibeVoice |
| Native full-duplex speech model + neural codec | native simultaneously inbound/outbound audio and text streams, 24kHz codec to ~12.5Hz audio frames, two-stage temporal/depth models | one full-duplex TaskSession with independent capture, generated codebook, text and playback cursors | — |
| Qwen3-Omni Thinker/Talker | separate perception/reasoning Thinker and speech Talker components, coupled text/audio output | verified Thinker→Talker edge, text/acoustic synchronization, option to omit Talker on input-only tasks | https://github.com/QwenLM/Qwen3-Omni |
| Hosted streaming TTS | hosted streaming TTS, voice/speaker selection and dialogue APIs with model-specific WebSocket semantics | explicit provider credentials/egress/consent/cost, text flush/end, cancel and provider voice lifecycle | provider documentation |

**Evidence note:** The dual-AR source model documentation describes its Dual-AR architecture and codec; model weights may use a research-only license different from code. For Qwen-TTS, 12.5Hz and 25Hz refer to **specific tokenizer variants**, not a universal single model rate. For VibeVoice, original repo release/availability has changed historically: inspect exact checkpoint and code location before loading. Hosted voice-provider internals are not independently verifiable; represent endpoint behaviors only.

## 3. Audio state and clocks: minimum trusted schema

~~~rust
struct SpeechSynthesisRequest {
    text: TimedTextStreamOrTextRef,
    voice: VerifiedSpeakerOrStyleId,
    language: Option<LanguageId>,
    sample_rate: RequestedOrSourceRate,
    output_codec: RegisteredAudioOutput,
    prosody: ModelSpecificProsodyControls,
    seed: OptionalModelRngSeed,
    streaming: StreamingCapabilityRequest,
}
struct AcousticFrontier {
    input_text_position: TextTokenCursor,
    time_frame: AcousticFrameCursor,     // 12.5Hz, 21Hz, 25Hz etc
    codebook_depth: Option<RvqDepthCursor>,
    codec_delay: SampleOffset,
    decoded_sample_cursor: SampleCursor,
    speaker_style_version: SpeakerStateVersion,
}
~~~

Time-domain frame and depth-domain codebook count MUST NOT be collapsed into one scalar position; 10 codebooks at 21Hz does not mean 210 acoustic frames per second. Codebook completion and PCM sample clock finality are different. A codec frame is publishable only when its declared number/dependencies of codebooks and waveform reconstruction history are valid.

Separate clocks:
- text Unicode/grapheme/token/phoneme positions (not necessarily aligned);
- acoustic semantic frames and residual RVQ depth;
- resampler input/output sample counts with checked rational ratios and fractional residue;
- vocoder overlap/add state and emitted samples;
- output codec packets and PTS; transport transmitted/acknowledged/played positions.

Voice/speaker state persists within a source-qualified session; changing voice mid-chunk may require explicit new session, codec reset, or model-defined transition. External voice cloning requires verified consent/rights policy, not just a model parameter.

## 4. Kokoro: non-token-AR TTS execution trace

1. Validate raw text language, Unicode normalization, punctuation and length before allocating weights.
2. Run source-required grapheme-to-phoneme frontend (an independent G2P engine or validated equivalent) and phoneme tokenizer; record G2P model/version/license separately from Kokoro weights.
3. Apply verified phoneme-aware text conditioning and speaker/style embedding.
4. Compute duration/alignment/F0/prosody with source-specific inference; normalize and bound predicted frame length.
5. Run acoustic generation and vocoder with source-defined convolution, phase/ISTFT transform, overlap and output sample rate.
6. Apply postprocessing only if source-defined or explicitly chosen audio-quality policy (normalization, loudness, trim).
7. Publish complete PCM/encoded packet at safe vocoder overlap boundary and account for latency.
8. On cancellation, discard uncommitted samples; retain submitted GPU buffers until Ready; playback ACK ledger remains separate.

**Adversarial fixture:** an input with no whitespace (numbers/foreign language) must exercise correct text normalization, not silently strip characters. Changing G2P or voice embedding changes source semantics even if PCM sounds plausible. Two independent listeners share immutable Kokoro weights but never the other person's speaker or text state.

## 5. Dual-AR codebook generation trace

1. Bind text+speaker/style/reference voice, source codec and exact source checkpoint.
2. Slow AR creates primary semantic RVQ codebook tokens along time, approximately 21Hz per source version.
3. Fast AR produces residual codes per temporal frame (nine residual codebooks in the source model's official description). This inner loop is **not** nine extra time frames.
4. Only when the full required depth group is complete may the audio codec decoder consume that frame.
5. Neural waveform decoder reconstructs sample intervals under its overlap/receptive field; resulting PCM cursor may lag codebook generation.
6. Commit and deliver complete audio chunks with exact timestamps. Separate text output, if any, is not implicitly guaranteed to align word-perfectly to codec frames.
7. Barge-in cancels uncommitted future codebook/PCM frames by output generation; already heard output requires playback-boundary accounting.
8. A source reference codec version change requires new assembly identity, not silent reuse of cached frame embeddings.

**Specializations:** device-native slow+fast batch, small depth loops, direct codebook embedding gather, fused acoustic decoder; each must preserve codebook probabilities and state. A fast GPU kernel that reorders RVQ depth will change speech.

## 6. Qwen3-TTS: variant/streaming cautions

Upstream publishes 12.5Hz and 25Hz neural speech tokenizers; checkpoint-specific inference can use distinct codec models, temporal chunk policies and acoustic reconstruction. A single CLI model alias is insufficient to identify the tokenizer/codec and semantics.

- Verifiable components: text tokenizer, speaker/style/reference encoder, acoustic token predictor, speech tokenizer/decoder, optional diffusion/acoustic refinement.
- Codec sample rate, quantizer count, token frame rate, chunk latency and special text/audio sentinel positions come from exact source model.
- A model README may claim streaming-capable architecture while a particular inference wrapper buffers full output before decoding. Zerogravity may advertise playable streaming only when implemented and measured source-compatible codec chunk completion.
- Audio/text synchronized generation requires declared alignment mapping and TaskFrontier coordinates. Do not assert one text token maps to one speech frame.
- Voice design and voice cloning must be opt-in capability with privacy/consent and locally bound reference-audio identity; never silently send source voice to cloud service.

## 7. VibeVoice long-form and realtime

- Long-form voice state may include speaker consistency, conversation context and prosodic trajectory across multiple speech chunks.
- Realtime 0.5B model uses streaming text input: incomplete text words/punctuation and interruptive edits must be tracked, not simply concatenated after generation.
- Acoustic diffusion/flow stages (where present per selected checkpoint) need source solver/timestep; do not substitute direct waveform vocoder.
- Long generation cannot accumulate every PCM sample and every GPU latent in resident memory. Stream complete chunks, preserve bounded overlap/prosody state and release verified retired buffers.
- Speaker changes are explicit track events; multiple voices may share one model but not mutable per-speaker generation state.
- Profile real-time factor, voice consistency and first audible latency separately from GPU batch throughput.

## 8. Native full duplex with a neural codec

A native full-duplex model is a critical test of architecture rather than merely another TTS implementation. Its neural codec publishes ~12.5Hz neural audio frames from 24kHz audio with streaming causal behavior, and the model maintains simultaneous user speech input and its own output plus text stream.

**Trace:**
1. Capture incremental user audio and decode/encode with the neural codec into acoustic input frames, source sample/capture clock and latent delay.
2. Model consumes incoming audio even while its earlier response is still being generated or played, under the source-trained duplex attention/state semantics.
3. Model predicts text and outgoing acoustic tokens, potentially through a temporal backbone and depth decoder.
4. Neural codec transforms only complete output acoustic groups into continuous audio samples with known overlap.
5. Commit neural state and complete output chunks at legal source boundaries. Delivery and playback are tracked separately.
6. Barge-in advances the affected outgoing turn generation, not the entire input acoustic state and not the shared physical device epoch.
7. Model-specific conversation state update on interruption is mandatory. It may **not** be possible to reconstruct an exact state by splicing a text transcript into the stream.
8. Treat user silence, VAD activity, echo and dropped PCM packets as distinct input events; do not infer all responses from text-only prompts.

The neural codec has a real Rust implementation in an independent source repository, useful as **prior art**. It is not permission to embed a separate independent Rust inference engine under the Zerogravity serving stack; adopt mathematical/codec semantics through independently reviewed source/licensing.

## 9. Qwen3-Omni Thinker/Talker

A verified Qwen3-Omni assembly binds audio/video frontend, Thinker, Talker, speech codec/tokenizers and typed text/acoustic outputs. Thinker may have expert routing and state differing from Talker; if input-only ASR/perception is requested, a compatible assembly may omit Talker residency. Conversely, audio output requires valid Thinker→Talker edge and codec output semantics, not a generic language model + speaker file.

Candidate optimization:
- separate residency/offload of Talker and codec decoder for non-audio tasks,
- overlap source-permitted Thinker computation with output codec preparation,
- reuse shared text conditioning under exact artifact identity,
- handle interleaved temporal audio+video inputs and independent clocks.
Any pipelined generation must preserve source causal dependencies and single committed root semantics.

## 10. Hosted voice providers

A hosted voice provider is an explicit TaskServiceAdapter:
- validated voice ID/model ID, output format, language/latency options;
- WebSocket streaming input/output frames and text flush/end markers per provider schema;
- provider-side rate, quota/cost, API key and data-egress permissions;
- client disconnect, barge-in and cancellation receipts; remote operation may continue after local network error;
- source voice consent and user privacy classification;
- provider-reported bytes/seconds, not a fabricated neural StateRoot or local GPU runtime.

Do not label API-hidden model internals "known codec-AR" without official source. Native portability requires actual licensed weights, math and source codec.

## 11. Serving contracts / output finality

- /v1/audio/speech: bounded TTS request with declared output codec/sampling, source-bound voice, zero unsupported-field swallowing.
- Stream events: TextAccepted, AcousticFramePrepared, CodecFrameComplete, PcmChunkCommitted, AudioPacketDelivered, PlaybackReported, InterruptionConfirmed, ResponseTerminated.
- WebRTC: media track and control channel; WS: binary media + control metadata; CLI: file output and optional progressive chunk callback, all from the same committed media event.
- Explicit output-side backpressure: never unbounded PCM buffer. A slow speaker/network can pause at frame boundary or cancel; model cannot overwrite immutable already-committed output.
- Exact sample duration and output decoder delay are required for speech/video synchronization and lived playback history.

## 12. Latency, quality and risk

Measure:
- TTFA (time to first audible valid PCM) and capture-to-first-response,
- P50/P95/P99 chunk gap/jitter, real-time factor (<1 means faster-than-audio-duration, state definition),
- perceptual metrics when allowed plus transcript alignment, speaker similarity, artifacts at chunk boundaries,
- normal/interrupt/low-resource long-form memory consumption,
- default and degraded modes with exact model/codec/numeric receipt,
- voice data retention/security and explicit user permissions.

A vendor-reported 300ms first audio on one hardware class is not a guarantee Zerogravity achieves it on Strix Halo or RTX. Performance evidence must include checkpoint, device, load/temperature, transport, codec and setup.

## 13. Required adversarial acceptance fixtures

**MM-TTS-01:** Kokoro phoneme/duration path generates exact sample count and valid wav without any decoder text LM.  
**MM-TTS-02:** Dual-AR frame with nine of ten codebooks remains unplayable; final depth completes one frame, not ten time frames.  
**MM-TTS-03:** Qwen12.5Hz and 25Hz tokenizer assemblies refuse cross-swapped codec components; clocks round exactly.  
**MM-TTS-04:** voice style updated mid-chunk under an unsupported variant must fail before state corruption.  
**MM-TTS-05:** Full-duplex simultaneous user audio in + assistant audio out preserves independent cursors; no obligatory transcript.  
**MM-TTS-06:** user barge-in truncates unplayed speech, does not erase consumed user input or cancel other GPU sessions.  
**MM-TTS-07:** learned/vocoder overlap buffers persist across samples, but not across unrelated sessions; exact PCM continuity.  
**MM-TTS-08:** slow client backpressure bounded for 10-minute long generation; no infinite output queue.  
**MM-TTS-09:** disconnect/resume playback cannot claim user heard delivered bytes without ACK; maintain UnknownPlaybackProgress.  
**MM-TTS-10:** no untrusted remote voice-cloning/hosted-provider egress without explicit authorization/consent.  
**MM-TTS-11:** partial codec GPU command failure fences physical handle if unknown and refuses StateRoot publication.  
**MM-TTS-12:** source-trained multiprompt/sample-rate and speaker tests on actual hardware before production media claim.

## 14. Incremental implementation plan

1. Introduce source-free distinct sample and acoustic-frame clocks + complete codec-frame finality mocks (no actual TTS weights).
2. Native Kokoro small non-AR TTS with original checkpoint and verifiable G2P/vocoder, CPU reference then qualified GPU.
3. Dual-AR small codec/RVQ fixtures to prove time×depth; then original licensed model if allowed.
4. Qwen3-TTS exact variant assembly and chunked source codec on CPU; only qualified realtime playback advertised.
5. Native full-duplex neural-codec end-to-end latency/interrupt evidence, or Qwen3-Omni Thinker/Talker with original codecs.
6. Hosted voice-provider TaskServiceAdapter independently tested against sandbox/mock API policy, not a second native model engine.

Any new TTS architecture should add a source model assembly and TaskDriver, never a duplicate GPU ResourceArbiter/TaskCoordinator.

## 15. Primary sources checked

- Kokoro: https://huggingface.co/hexgrad/Kokoro-82M
- Rust-language independent source study.
- Dual-AR speech source model.
- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- VibeVoice: https://github.com/microsoft/VibeVoice
- Native full-duplex/neural-codec source model.
- Qwen3-Omni: https://github.com/QwenLM/Qwen3-Omni
- Hosted real-time TTS WebSocket documentation.
- Hosted TTS versus text-to-dialogue documentation.

Checked against upstream docs and earlier model atlas on 15 Oct 2026. All detailed model internals/weights and licensing must be independently bound to the exact implementation/version before declaring Zerogravity support.
