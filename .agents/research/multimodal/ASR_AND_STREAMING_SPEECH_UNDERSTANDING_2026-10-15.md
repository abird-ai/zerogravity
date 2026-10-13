# Research Track B — ASR, Speech Translation and Streaming Speech Understanding

**Date:** 15 October 2026 | **Mode:** source-grounded DESIGN/RESEARCH, not Zerogravity implementation.  
**Owner scope:** original mathematical source/processor/decoder models, stream/clock/state semantics and acceptance fixtures. Do not own transport server, GPU allocator, TaskCoordinator or model routing.  
**Read first:** .agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md; .agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md.

## 1. Verdict and central lesson

A speech-to-text task is not universally a causal text generator. **Whisper's encoder-decoder autoregression, CTC blank collapse, RNN-T streaming joiner/predictor, TDT duration transduction, source-language speech translation and joint diarization have different mathematical effects and legitimate partial-output boundaries.** They share frontend DSP, convolution/matmul/attention/normalization kernels, resource admission and typed transcript events, but require separate trusted TaskDriver algorithms. Native ASR must support both file-window and continuous streaming semantics without inventing timestamps.

## 2. Authoritative source matrix

| Family / exact variant risk | Original method and task | State/protocol consequence | Official source |
|---|---|---|---|
| Whisper (size/language/English vs multilingual vary) | log-mel encoder + cross-attending AR text decoder, language/task/time tokens, 30-second source window method | rolling windows, prompt carry, exact time offset merging, timestamp token grammar | original public model release and research |
| NVIDIA Parakeet FastConformer (CTC, RNNT, TDT vary BY CHECKPOINT) | acoustic FastConformer encoder; CTC blank collapse OR recurrent transducer predictor/joiner OR token/duration transducer | separate decoder and state schema; align/cached convolution and timestamp rules | public ASR toolkit and checkpoint matrix |
| NVIDIA Parakeet TDT 0.6B v3 | offline multilingual TDT checkpoint, not automatically cache-aware streaming | **do not advertise live streaming** simply because the FastConformer family has other streaming variants | public model-variant documentation |
| NVIDIA Canary variants | multitask encoder-decoder speech recognition and speech translation, language task slots | source-target language conditioning, translation/not just ASR, task label and capitalization | public ASR toolkit |
| NVIDIA Nemotron streaming speech variants | cache-aware FastConformer RNNT with limited lookahead, specific streaming configurations | persistent acoustic encoder/predictor cache and chunk/latency policy | public model-variant documentation |
| VibeVoice-ASR | long-form ASR with diarization and timestamping (who/when/what) | speaker identity, interval alignment, long acoustic context and structured transcript | public model card |
| VibeVoice-ASR-Streaming | incremental transcription while PCM arrives; chunk/lookahead from checkpoint's preprocessor config | emit provisional text/chunk revisions; do not hardcode arbitrary chunk length | public streaming ASR documentation |
| Qwen3-Omni Thinker | native multimodal audio/video/text understanding, MoE architecture varies by variant | synchronized input media clocks and perception state; optional Talker separate | public model documentation |

**Source caution:** Model names and checkpoints are not interchangeable. Some Parakeet models support full-utterance only; some use source-trained chunked caches. Whisper large versus tiny changes weights/performance, not the generic driver algorithm. Canary task/translation pairing is a model capability, not inferred from a request string. VibeVoice ASR format and streaming lookahead must come from its versioned preprocessor config, not a global 20ms constant.

## 3. Whisper reference execution trace

1. Accept PCM/file under size/duration/rights restrictions. Decode compressed audio via registered/permissioned codec with bounded decompression and checked channels/sample format.
2. Resample to the source-required audio rate using a registered resampler; record delay, source sample clock, exact rational sample positions and resampling remainder.
3. Compute STFT/log-mel features with exact window/hop/window function, normalization, pad/truncate/window contract. Codec/processor version is part of assembly identity.
4. Execute encoder using standard conv/attention/GEMM regions; encoder output is immutable within the window and may be cached under exact source audio feature hash.
5. Decoder starts with model-specific start/language/task/prompt header tokens and performs AR token generation with encoder cross attention; tokenizer is source-verified.
6. Timestamp token grammar, no-speech probability, temperature fallback/beam behavior, compression/segment selection and window merging follow exact upstream behavior or a named alternative task quality mode.
7. Prepare transcript segments with exact window/sample offsets and possible overlapping-word adjudication; publish committed segment only when source policy says stable.
8. Stream interim transcript if the checkpoint/TaskDriver actually supports it; an offline-only window result should not be deceptively streamed as source-native word-by-word real-time recognition.
9. Device memory includes encoder activations, decoder KV, mel buffers, source text history and bounded overlap; pending frontend buffers have their own SampleClock domain.

**Negative cases:** wrong sample rate interpreted without resampling; timestamp token treated as natural-language text; reused encoder cache with different mel preprocessing; missing language task header; late segment from stale capture epoch; bad audio frame counts before GPU allocation.

## 4. CTC source-specific execution trace

1. Capture, validate and window audio. Run source-trained Conv/FastConformer with cached/recurrent state if model supports it.
2. Produce per-frame logits with declared vocabulary including **blank** class.
3. Apply model-specific CTC decoding: collapse repeats and blanks with correct ordering. Optional beam/LM fusion changes decoding semantics and must have named task options.
4. Align label frames to exact source feature-frame and audio-sample clocks including subsampling and lookahead delays.
5. Emit provisional segment revisions while frames may revise, then publish final stable transcript segment according to decoder/checkpoint policy.
6. Validate WER, time alignment and stream chunk invariance against original source fixture. Do not write output through a causal-text decoder unless the model truly contains one.

**Counterexample:** source frame predictions [a,a,blank,a] collapse to [a,a] under CTC rules, not a GPT token stream. Blank id and padding mask semantics must not change.

## 5. RNNT and TDT source-specific trace

1. Feature frontend yields frames and validated subsampled encoder state.
2. RNN-T uses a prediction network conditioned on previous labels and a joiner combining acoustic and predictor representations.
3. Decoder can emit multiple labels per acoustic frame until blank; beam state, blank probability and recurrent prediction caches belong in provisional StateDomains.
4. TDT includes token **duration**, permitting semantic frame jumps; its duration-cursor semantics cannot be simulated with a fixed frame increment.
5. A source checkpoint may allow cache-aware streaming with bounded lookahead or may only support full-utterance inference; advertised realtime capabilities are evidence-gated.
6. Time stamps are formed from trained subsampling, duration and observed sample-clock positions, never solely from token index.
7. Mid-frame failure reverts entire proposed acoustic/predictor state; only a valid transcript frontier publishes.

**Performance candidates:** fused encoder conv+norm/GEMM, cached streaming attention, predictor/joiner scalar/tiny-batch GPU vs CPU selection, duration-aware stage batching. Fusing is legal only if blank/duration probabilities and state are preserved.

## 6. Canary ASR + AST / language conditioning

- A request must explicitly name task and source/target language constraints if required; never silently translate when caller requested verbatim transcription.
- Language prompts/task slots, tokenizer, audio segment sizes, punctuation/casing and acoustic features are source processor components.
- Translate with source checkpoint decoder where actually trained; if a separate text translation model is used, describe it as a multi-model composition with a different ModeReceipt.
- Return transcription/translation output type with language version, alignment policy and confidence provenance.

## 7. VibeVoice and diarization

- Long audio requires chunked/bounded feature residency; do not allocate an entire hour-long unbounded GPU attention matrix as an accidental general frontend.
- Speaker IDs are provisional within a recording/session and often not equivalent to universal real-world identity. Merge/renumber only through explicit diarization versioned correction.
- ASR content, word/time spans and speaker turns are distinct event streams that may require an atomic committed segment group.
- Streaming chunk/lookahead must come from each checkpoint preprocessor_config rather than a family-wide fixed number.
- A change in speaker segmentation after text has been streamed requires explicit revision events; it cannot silently rewrite a previously committed transcript.

## 8. Qwen-Omni audio input and live perception

The Thinker is a verified multimodal model component using audio and potentially video inputs. Its frontend codec, source text/audio representation and time alignment must be bound to its exact source variant. A recognition-only task can choose to omit/unload a Talker when assembly permits, without inventing speech output. If internal audio tokens and video frames are interleaved, the TaskDriver maintains the appropriate source order and typed timebases; do not collapse everything to text.

## 9. Reusable ASR contract (illustrative)

~~~rust
struct SpeechRecognitionRequest {
    source_audio: TimedPcmOrEncodedRef,
    task: TranscribeOrTranslate,
    languages: SpeechLanguageSelection,
    timestamps: TimestampGranularity,
    diarization: DiarizationPreference,
    interim: InterimPolicy,
    decoder: RegisteredDecoderConfig,
    deadline: Option<SoftDeadline>,
}
struct TranscriptSegment {
    segment: StableSegmentId,
    revision: u64,
    text: String,
    sample_range: CheckedSampleRange,
    speaker: Option<SpeakerTrackId>,
    language: Option<LanguageId>,
    confidence: Option<ConfidenceWithProvenance>,
    finality: InterimOrCommitted,
}
~~~

**Capabilities cannot be inferred from enum alone.** Exact model binding reports which decoder, translation languages, timestamps, diarization and streaming modes have source/test evidence. External recognizer service is TaskServiceAdapter, not a native ASR DeviceBackend.

## 10. State, scheduling and memory

| State domain | Source-relevant contents | Safe commit |
|---|---|---|
| AudioIngress | capture sample cursor, gaps, channels | consumed window/ack |
| Resampler/STFT | fractional residue, overlap, sample/window index | complete exact frontend window |
| EncoderCache | attention/conv cache, subsampling, mask | source-defined encoder chunk |
| DecoderBeam | RNNT predictor/joiner, TDT duration, or Whisper KV and tokenizer | validated text hypothesis version |
| Diarization | speaker cluster labels/embeddings and interval versions | stable speaker assignment |
| Transcript | segment, revision, word timestamps and language | complete declared transcript segment |

Low-latency ASR uses bounded cache-aware frames and CPU/GPU crossover cost, not maximum GEMM throughput only. Whole-file ASR can batch windows and prioritize throughput. Model/router should expose task-specific QoS and streamed/whole-file modes separately. Other ASR sessions have disjoint mutable state but can share immutable model weights and processor kernels.

## 11. Independent verification and negative fixtures

**MM-ASR-01**: mono 16kHz audio and 44.1k→16k resampled exact time mapping across many windows with residue; no drift.  
**MM-ASR-02**: Whisper 30s window boundary and valid timestamp token header, exact overlap/merge.  
**MM-ASR-03**: synthetic CTC frames with blanks/repeats, beam selection and timestamp alignment; no token AR flattening.  
**MM-ASR-04**: RNNT predictor/joiner emits multi-label frame then blank, cancellation rolls back both caches.  
**MM-ASR-05**: TDT source duration step advances correct number of acoustic frames; wrong duration schema fails.  
**MM-ASR-06**: checkpoint that disallows streaming explicitly rejects WS live request, despite family name Parakeet.  
**MM-ASR-07**: Canary source→target language/task flags correct, no silent transcript/translation change.  
**MM-ASR-08**: VibeVoice ASR streaming revised hypothesis with speaker update; immutable committed past unaffected.  
**MM-ASR-09**: interrupted input includes stereo gap, sample discontinuity and frontend overlap; reject impossible clocks.  
**MM-ASR-10**: long audio bounded input buffers; malicious decompression ratio rejected before large allocation.  
**MM-ASR-11**: ModelSession=two recordings same weights; no cache/diarization leakage.  
**MM-ASR-12**: mismatched processor/feature normalization to checkpoint rejected rather than producing plausible text.

Before marking model implemented: obtain licensed exact checkpoint, source feature extractor/decoder configuration, CPU independent original oracle, token/time WER tests, actual GPU source parity and latency. Recognition of model metadata is not execution support.

## 12. Implementation sequencing and priorities

1. Tiny source-validated log-mel/CTC mock TaskDriver and typed transcript events.
2. One real small Whisper or Parakeet checkpoint, original source feature/tokenizer/decoder verification, CPU first.
3. Add transducer variant only when RNN-T/TDT semantics have isolated test oracle; do not approximate with AR generator.
4. Add Canary multilingual translation and VibeVoice rich/long ASR with licensed code/weights and real tests.
5. Join ASR with a separately modeled LLM/TTS under trusted user application composition, then native Omni perception/duplex when verified.

Maintain one ModelDefinition, one TaskDriver per algorithm, one TaskCoordinator. No separate ASR router/allocator/server.

## 13. Primary sources and provenance

- Original Whisper implementation.
- Whisper research publication.
- Parakeet/FastConformer ASR toolkit source.
- Parakeet/FastConformer checkpoint matrix.
- Parakeet/FastConformer model-variant and streaming caveat documentation.
- VibeVoice model card.
- VibeVoice streaming ASR documentation.
- Qwen3-Omni model documentation.

Read/tested as publicly available documentation on 15 Oct 2026. **Variant-specific code/config and licenses must be rechecked** when choosing real artifacts. No tests/downloads of the listed checkpoints were performed during this design research.
