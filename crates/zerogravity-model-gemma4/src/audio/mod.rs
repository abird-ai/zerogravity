//! Native finite audio encoder over a verified original Gemma 4 checkpoint.
//! Produces language-space soft-token rows; not a text-generation, ASR, TTS
//! or public serving capability by itself. F32 reference math with BF16
//! learned coefficients, not a claim of native BF16 device parity.
mod config;
mod frontend;
mod language;
#[cfg(feature = "audio-transcription")]
mod transcription;
#[cfg(feature = "audio-transcription")]
pub use transcription::{
    Gemma4Transcriber, Gemma4Transcript, Gemma4Transcription, TranscriptFinish,
};
mod layer;
pub use language::{AudioLanguageSession, Gemma4AudioLanguage};
mod source;

use crate::{Error, Result};
use config::Config;
use frontend::MelFrontend;
use layer::Layer;
use source::{Binder, Linear, Norm, Subsampling};
use std::sync::atomic::AtomicBool;
use zerogravity_artifacts::{cache::SnapshotLease, source_weights::VerifiedTensorSource};
use zerogravity_core::{
    identity::SourceIdentity,
    resource_arbiter::{
        AllocationClass, AllocationRequest, GlobalResourceLease, TransferAdmission,
    },
};
use zerogravity_cpu::ops::{audio::ImageRows, encoder::check_cancel};

fn overflow() -> Error {
    Error::Limit("audio allocation extent overflow".into())
}
fn reserve(
    a: &TransferAdmission,
    kind: AllocationClass,
    bytes: u64,
) -> Result<GlobalResourceLease> {
    a.arbiter.admit(
        &a.resource,
        a.owner.clone(),
        &[AllocationRequest { kind, bytes }],
    )
}
fn bytes(n: usize) -> Result<u64> {
    (n as u64).checked_mul(4).ok_or_else(overflow)
}

pub struct AudioFeatures {
    values: Vec<f32>,
    frames: usize,
    width: usize,
    source: SourceIdentity,
    _memory: GlobalResourceLease,
}
impl AudioFeatures {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn frames(&self) -> usize {
        self.frames
    }
    pub fn width(&self) -> usize {
        self.width
    }
}
pub struct AudioTokens {
    values: Vec<f32>,
    rows: usize,
    width: usize,
    source: SourceIdentity,
    _memory: GlobalResourceLease,
}
impl AudioTokens {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
}

pub struct Gemma4AudioEncoder {
    source: VerifiedTensorSource,
    config: Config,
    frontend: MelFrontend,
    subsampling: Subsampling,
    positions: Vec<f32>,
    layers: Vec<Layer>,
    output: Linear,
    bias: Vec<f32>,
    projection_norm: Norm,
    language_projection: Linear,
    admission: TransferAdmission,
    _parameters: GlobalResourceLease,
}
impl Gemma4AudioEncoder {
    /// Authenticate the original managed checkpoint once, with bounded
    /// source windows rather than another complete resident checkpoint.
    pub fn load(
        snapshot: &SnapshotLease,
        max_source_bytes: u64,
        admission: &TransferAdmission,
    ) -> Result<Self> {
        let source = crate::managed_source::open(snapshot, admission, max_source_bytes, None)?;
        Self::bind(snapshot, source, admission, None)
    }
    pub(crate) fn bind(
        snapshot: &SnapshotLease,
        source: VerifiedTensorSource,
        admission: &TransferAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        let _metadata = reserve(admission, AllocationClass::Scratch, 4 * 1024 * 1024)?;
        let config = Config::parse(
            &snapshot.read_verified("config.json", 64 * 1024)?,
            &snapshot.read_verified("processor_config.json", 64 * 1024)?,
        )?;
        let c = &config.audio;
        // Small decoded conv/norm/scalar parameters and source descriptors,
        // relative-position table and FFT/filter plans. Full matrices remain
        // in the original charged source backing.
        let h = c.hidden_size as u64;
        let n = c.num_hidden_layers as u64;
        let [a, b] = c.subsampling_conv_channels.map(|n| n as u64);
        let small_floats = n * h * (14 + c.conv_kernel_size as u64)
            + a * 9
            + b * a * 9
            + a
            + b
            + (c.attention_context_left as u64) * h
            + 3 * c.output_proj_dims as u64;
        let parameter_bytes = small_floats
            .checked_mul(4)
            .and_then(|n| {
                n.checked_add(
                    8 * (config.frontend.feature_size as u64)
                        * (config.frontend.fft_length as u64 / 2 + 1),
                )
            })
            .and_then(|v| v.checked_add(n * 128 * 1024 + 1024 * 1024))
            .ok_or_else(overflow)?;
        let parameters = reserve(
            admission,
            AllocationClass::ImmutableWeights,
            parameter_bytes,
        )?;
        let frontend = MelFrontend::new(&config.frontend)?;
        let mut binder = Binder::new(&source, cancelled);
        let subsampling = Subsampling::bind(&mut binder, &config)?;
        let positions = layer::relative_positions(c);
        let layers = (0..c.num_hidden_layers)
            .map(|i| Layer::bind(&mut binder, i, c))
            .collect::<Result<Vec<_>>>()?;
        let output = binder.linear(
            "model.audio_tower.output_proj",
            c.hidden_size,
            c.output_proj_dims,
            false,
        )?;
        let bias = binder.decoded("model.audio_tower.output_proj.bias", &[c.output_proj_dims])?;
        let projection_norm = Norm::unit(c.output_proj_dims, c.rms_norm_eps);
        let language_projection = binder.linear(
            "model.embed_audio.embedding_projection",
            c.output_proj_dims,
            config.text_width,
            false,
        )?;
        binder.finish()?;
        Ok(Self {
            source,
            config,
            frontend,
            subsampling,
            positions,
            layers,
            output,
            bias,
            projection_norm,
            language_projection,
            admission: admission.clone(),
            _parameters: parameters,
        })
    }
    pub fn source(&self) -> SourceIdentity {
        self.source.source_identity()
    }
    pub(crate) fn verified_source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    #[cfg(feature = "audio-transcription")]
    pub(crate) fn validate_admission(&self, admission: &TransferAdmission) -> Result<()> {
        self._parameters
            .bytes_in_domain(&admission.arbiter, &admission.resource)?;
        Ok(())
    }
    #[cfg(feature = "audio-transcription")]
    pub(crate) fn token_count(&self, samples: usize) -> Result<usize> {
        if samples > self.max_samples() {
            return Err(Error::Limit("Gemma audio sample extent".into()));
        }
        Ok(MelFrontend::frames(&self.config.frontend, samples)?.div_ceil(4))
    }
    pub fn max_samples(&self) -> usize {
        self.config.max_samples
    }
    /// Real mono normalized PCM samples, 16 kHz. One unpadded utterance,
    /// no silent resampling, truncation or synthetic audio tokens.
    pub fn features(
        &self,
        samples: &[f32],
        sample_rate: u32,
        cancelled: Option<&AtomicBool>,
    ) -> Result<AudioFeatures> {
        let c = &self.config.frontend;
        if sample_rate != c.sampling_rate
            || samples.len() > self.max_samples()
            || !samples
                .iter()
                .all(|v| v.is_finite() && (-1.0..=1.0).contains(v))
        {
            return Err(Error::InvalidInput("audio PCM rate/length/range".into()));
        }
        check_cancel(cancelled)?;
        let frames = MelFrontend::frames(c, samples.len())?;
        let _scratch = reserve(
            &self.admission,
            AllocationClass::Scratch,
            (c.fft_length as u64) * 64 + 64 * 1024,
        )?;
        let memory = reserve(
            &self.admission,
            AllocationClass::Output,
            bytes(frames.checked_mul(c.feature_size).ok_or_else(overflow)?)?,
        )?;
        let values = self.frontend.extract(c, samples, frames, cancelled)?;
        if !values.iter().all(|v| v.is_finite()) {
            return Err(Error::Malformed("nonfinite audio features".into()));
        }
        Ok(AudioFeatures {
            values,
            frames,
            width: c.feature_size,
            source: self.source(),
            _memory: memory,
        })
    }
    pub fn encode(
        &self,
        features: &AudioFeatures,
        cancelled: Option<&AtomicBool>,
    ) -> Result<AudioTokens> {
        let c = &self.config.audio;
        if features.source != self.source() || features.width != self.config.frontend.feature_size {
            return Err(Error::InvalidInput(
                "audio features belong to another source/profile".into(),
            ));
        }
        check_cancel(cancelled)?;
        let first = ImageRows {
            height: features.frames,
            width: features.width,
            channels: 1,
        }
        .subsampled(c.subsampling_conv_channels[0]);
        let second = first.subsampled(c.subsampling_conv_channels[1]);
        let tokens = second.height;
        let conv = first
            .elements()
            .and_then(|n| n.checked_add(second.elements()?))
            .ok_or_else(overflow)?;
        // Max live convolution stages or transformer row activations,
        // relative/one-row GEMV, FFN, projection and reduction temporaries.
        let scratch_elements = tokens
            .checked_mul(24 * c.hidden_size + 4 * c.output_proj_dims + 2 * self.config.text_width)
            .and_then(|n| n.checked_add(conv))
            .and_then(|n| n.checked_add(8 * self.positions.len() + 32 * c.hidden_size))
            .ok_or_else(overflow)?;
        let scratch = bytes(scratch_elements)?
            .checked_add(4 * 1024 * 1024)
            .ok_or_else(overflow)?;
        let _scratch = reserve(&self.admission, AllocationClass::Scratch, scratch)?;
        let memory = reserve(
            &self.admission,
            AllocationClass::Output,
            bytes(
                tokens
                    .checked_mul(self.config.text_width)
                    .ok_or_else(overflow)?,
            )?,
        )?;
        let mut hidden = self.subsampling.run(
            &self.source,
            &self.config,
            &features.values,
            features.frames,
            cancelled,
        )?;
        for layer in &self.layers {
            check_cancel(cancelled)?;
            hidden = layer.run(&self.source, c, hidden, &self.positions, cancelled)?;
        }
        let mut output = self.output.rows(&self.source, &hidden, cancelled)?;
        drop(hidden);
        for row in output.chunks_exact_mut(c.output_proj_dims) {
            for (v, bias) in row.iter_mut().zip(&self.bias) {
                *v += bias;
            }
        }
        self.projection_norm.apply(&mut output)?;
        let values = self
            .language_projection
            .rows(&self.source, &output, cancelled)?;
        check_cancel(cancelled)?;
        if values.len() != tokens * self.config.text_width || !values.iter().all(|v| v.is_finite())
        {
            return Err(Error::Malformed(
                "audio language projection extent/values".into(),
            ));
        }
        Ok(AudioTokens {
            values,
            rows: tokens,
            width: self.config.text_width,
            source: self.source(),
            _memory: memory,
        })
    }
    pub fn encode_pcm(
        &self,
        samples: &[f32],
        sample_rate: u32,
        cancelled: Option<&AtomicBool>,
    ) -> Result<AudioTokens> {
        let features = self.features(samples, sample_rate, cancelled)?;
        self.encode(&features, cancelled)
    }
}

#[cfg(test)]
mod tests;
