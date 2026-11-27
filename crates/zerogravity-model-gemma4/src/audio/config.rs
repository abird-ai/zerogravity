//! Original Gemma 4 USM audio and feature-extractor contracts.
use crate::{Error, Result};
use serde::Deserialize;

const MAX_CONFIG: usize = 64 * 1024;

#[derive(Deserialize)]
struct Root {
    model_type: String,
    audio_config: Audio,
    text_config: Text,
}
#[derive(Deserialize)]
struct Text {
    hidden_size: usize,
}
#[derive(Deserialize)]
pub(super) struct Audio {
    model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub output_proj_dims: usize,
    pub subsampling_conv_channels: [usize; 2],
    pub attention_chunk_size: usize,
    pub attention_context_left: usize,
    attention_context_right: usize,
    pub attention_logit_cap: f32,
    attention_invalid_logits_value: f32,
    pub conv_kernel_size: usize,
    hidden_act: String,
    pub residual_weight: f32,
    pub gradient_clipping: f32,
    pub rms_norm_eps: f32,
    pub use_clipped_linears: bool,
    #[serde(default)]
    chunk_size_feed_forward: usize,
}
#[derive(Deserialize)]
struct Processor {
    audio_seq_length: usize,
    audio_ms_per_token: usize,
    feature_extractor: Frontend,
}
#[derive(Deserialize)]
pub(super) struct Frontend {
    feature_extractor_type: String,
    pub feature_size: usize,
    pub sampling_rate: u32,
    pub frame_length: usize,
    pub hop_length: usize,
    pub fft_length: usize,
    fft_overdrive: bool,
    dither: f64,
    preemphasis: f64,
    input_scale_factor: f64,
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub mel_floor: f64,
    padding_side: String,
    padding_value: f64,
    per_bin_mean: Option<Vec<f64>>,
    per_bin_stddev: Option<Vec<f64>>,
    return_attention_mask: bool,
}
pub(super) struct Config {
    pub audio: Audio,
    pub frontend: Frontend,
    pub text_width: usize,
    pub max_samples: usize,
}
fn parse<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    if bytes.is_empty() || bytes.len() > MAX_CONFIG {
        return Err(Error::Limit("Gemma audio config extent".into()));
    }
    serde_json::from_slice(bytes).map_err(|e| Error::Malformed(format!("Gemma audio config: {e}")))
}
impl Config {
    pub fn parse(root: &[u8], processor: &[u8]) -> Result<Self> {
        let root: Root = parse(root)?;
        let p: Processor = parse(processor)?;
        let a = root.audio_config;
        let f = p.feature_extractor;
        let positive = |n: f32| n.is_finite() && n > 0.;
        // Original causal relative-position convention: 12-position left
        // horizon, 12-query blocks, no future context. This is the actual
        // installed Gemma4 reference's relative table, not generic RoPE.
        if root.model_type != "gemma4"
            || a.model_type != "gemma4_audio"
            || a.hidden_act != "silu"
            || a.attention_chunk_size != 12
            || a.attention_context_left != 13
            || a.attention_context_right != 0
            || a.chunk_size_feed_forward != 0
            || !positive(a.attention_logit_cap)
            || a.attention_invalid_logits_value > -1e6
            || !positive(a.gradient_clipping)
            || !positive(a.rms_norm_eps)
            || !positive(a.residual_weight)
            || !(1..=4096).contains(&a.hidden_size)
            || !a.hidden_size.is_multiple_of(2)
            || !(1..=64).contains(&a.num_hidden_layers)
            || !(1..=128).contains(&a.num_attention_heads)
            || !a.hidden_size.is_multiple_of(a.num_attention_heads)
            || !(1..=8192).contains(&a.output_proj_dims)
            || !(1..=8192).contains(&root.text_config.hidden_size)
            || !(1..=64).contains(&a.conv_kernel_size)
            || f.feature_extractor_type != "Gemma4AudioFeatureExtractor"
            || f.sampling_rate != 16000
            || f.frame_length != 320
            || f.hop_length != 160
            || f.fft_length != 512
            || f.fft_overdrive
            || f.dither != 0.
            || f.preemphasis != 0.
            || f.input_scale_factor != 1.
            || f.padding_side != "right"
            || f.padding_value != 0.
            || !f.return_attention_mask
            || f.per_bin_mean.is_some()
            || f.per_bin_stddev.is_some()
            || !f.min_frequency.is_finite()
            || !f.max_frequency.is_finite()
            || f.min_frequency < 0.
            || f.min_frequency >= f.max_frequency
            || f.max_frequency > 8000.
            || !f.mel_floor.is_finite()
            || f.mel_floor <= 0.
            || !(4..=512).contains(&f.feature_size)
            || !f.feature_size.is_multiple_of(4)
            || a.subsampling_conv_channels[0] != f.feature_size
            || !(1..=512).contains(&a.subsampling_conv_channels[1])
            || p.audio_ms_per_token != 40
            || !(1..=750).contains(&p.audio_seq_length)
        {
            return Err(Error::Unsupported(
                "Gemma 4 audio profile outside native F32 reference contract".into(),
            ));
        }
        let max_samples = p.audio_seq_length * 4 * f.hop_length;
        Ok(Self {
            audio: a,
            frontend: f,
            text_width: root.text_config.hidden_size,
            max_samples,
        })
    }
}
