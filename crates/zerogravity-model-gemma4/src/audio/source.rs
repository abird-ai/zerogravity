//! Bind the original BF16 audio closure. Matrices borrow authenticated
//! source bytes; only small affine/conv parameters are decoded.
use crate::{Error, Result};
use std::{collections::BTreeSet, sync::atomic::AtomicBool};
use zerogravity_artifacts::{SourceWeights, artifact::TensorInfo, codecs::dequantize};
use zerogravity_core::{
    accelerator::MatVecRegion,
    encoding::{ScalarDType, StoredTensorEncoding},
};
use zerogravity_cpu::ops::audio::{ImageRows, subsample_conv2d};
use zerogravity_cpu::{
    ops::{
        encoder::{check_cancel, layer_norm},
        rms_norm,
    },
    source::source_reference_projection,
};

const BF16: StoredTensorEncoding = StoredTensorEncoding::Scalar(ScalarDType::Bf16);

pub(super) struct Binder<'a> {
    pub source: &'a SourceWeights,
    used: BTreeSet<String>,
}
impl<'a> Binder<'a> {
    pub fn new(source: &'a SourceWeights) -> Self {
        Self {
            source,
            used: BTreeSet::new(),
        }
    }
    fn tensor(&mut self, name: &str, shape: &[usize], finite: bool) -> Result<TensorInfo> {
        let t = self.source.tensor(name)?;
        if t.kind != BF16 || !t.shape.iter().copied().eq(shape.iter().map(|&n| n as u64)) {
            return Err(Error::Malformed(format!(
                "Gemma audio tensor {name}: dtype/shape mismatch"
            )));
        }
        if finite
            && !self
                .source
                .bytes(t)?
                .chunks_exact(2)
                .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80)
        {
            return Err(Error::Malformed(format!(
                "nonfinite learned audio tensor {name}"
            )));
        }
        if !self.used.insert(name.to_owned()) {
            return Err(Error::Malformed(
                "duplicate Gemma audio tensor binding".into(),
            ));
        }
        Ok(t.clone())
    }
    pub fn decoded(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let t = self.tensor(name, shape, true)?;
        let n = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| Error::Limit("audio decoded tensor extent".into()))?;
        dequantize(BF16, self.source.bytes(&t)?, n)
    }
    fn scalar(&mut self, name: &str) -> Result<f32> {
        let t = self.tensor(name, &[], false)?;
        let raw = self.source.bytes(&t)?;
        let v = f32::from_bits(u32::from(u16::from_le_bytes([raw[0], raw[1]])) << 16);
        if v.is_nan() {
            return Err(Error::Malformed(format!("NaN clip bound {name}")));
        }
        Ok(v)
    }
    pub fn norm(&mut self, name: &str, width: usize, epsilon: f32) -> Result<Norm> {
        Ok(Norm {
            weight: self.decoded(&format!("{name}.weight"), &[width])?,
            epsilon,
        })
    }
    pub fn linear(
        &mut self,
        name: &str,
        input: usize,
        output: usize,
        clipped: bool,
    ) -> Result<Linear> {
        let (matrix, clip) = if clipped {
            let bounds = Clip {
                input: [
                    self.scalar(&format!("{name}.input_min"))?,
                    self.scalar(&format!("{name}.input_max"))?,
                ],
                output: [
                    self.scalar(&format!("{name}.output_min"))?,
                    self.scalar(&format!("{name}.output_max"))?,
                ],
            };
            if bounds.input[0] > bounds.input[1] || bounds.output[0] > bounds.output[1] {
                return Err(Error::Malformed(format!(
                    "inverted audio clip bound {name}"
                )));
            }
            (format!("{name}.linear.weight"), bounds)
        } else {
            (format!("{name}.weight"), Clip::UNBOUNDED)
        };
        Ok(Linear {
            weight: self.tensor(&matrix, &[output, input], true)?,
            input,
            output,
            clip,
        })
    }
    pub fn finish(self) -> Result<()> {
        if self.source.tensor_names().any(|name| {
            (name.starts_with("model.audio_tower.") || name.starts_with("model.embed_audio."))
                && !self.used.contains(name)
        }) {
            return Err(Error::Unsupported(
                "unbound original Gemma audio tensor".into(),
            ));
        }
        Ok(())
    }
}

struct Clip {
    input: [f32; 2],
    output: [f32; 2],
}
impl Clip {
    const UNBOUNDED: Self = Self {
        input: [f32::NEG_INFINITY, f32::INFINITY],
        output: [f32::NEG_INFINITY, f32::INFINITY],
    };
}
pub(super) struct Linear {
    weight: TensorInfo,
    input: usize,
    output: usize,
    clip: Clip,
}
impl Linear {
    pub fn rows(
        &self,
        source: &SourceWeights,
        rows: &[f32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        if rows.is_empty() || !rows.len().is_multiple_of(self.input) {
            return Err(Error::InvalidInput("audio projection input extent".into()));
        }
        let data = source.bytes(&self.weight)?;
        let mut out = Vec::with_capacity(rows.len() / self.input * self.output);
        for row in rows.chunks_exact(self.input) {
            check_cancel(cancelled)?;
            let clipped;
            let input = if self.clip.input == Clip::UNBOUNDED.input {
                row
            } else {
                clipped = row
                    .iter()
                    .map(|v| v.clamp(self.clip.input[0], self.clip.input[1]))
                    .collect::<Vec<_>>();
                &clipped
            };
            // Shared exact source GEMV; original HF [out,in] geometry,
            // F32 activation with BF16 coefficients, no second weight copy.
            let values = source_reference_projection(MatVecRegion {
                source_identity: source.source_identity(),
                tensor_name: &self.weight.name,
                kind: BF16,
                columns: self.input,
                rows: self.output,
                packed_weights: data,
                activation: input,
            })?;
            out.extend(
                values
                    .into_iter()
                    .map(|v| v.clamp(self.clip.output[0], self.clip.output[1])),
            );
        }
        Ok(out)
    }
}
pub(super) struct Norm {
    weight: Vec<f32>,
    epsilon: f32,
}
impl Norm {
    pub fn unit(width: usize, epsilon: f32) -> Self {
        Self {
            weight: vec![1.; width],
            epsilon,
        }
    }
    pub fn apply(&self, rows: &mut [f32]) -> Result<()> {
        if !rows.len().is_multiple_of(self.weight.len()) {
            return Err(Error::InvalidInput("audio RMSNorm extent".into()));
        }
        for row in rows.chunks_exact_mut(self.weight.len()) {
            row.copy_from_slice(&rms_norm(row, &self.weight, self.epsilon)?);
        }
        Ok(())
    }
}

struct Subsample {
    weights: Vec<f32>,
    norm: Vec<f32>,
    channels: usize,
}
impl Subsample {
    fn bind(b: &mut Binder<'_>, name: &str, input: usize, output: usize) -> Result<Self> {
        Ok(Self {
            weights: b.decoded(&format!("{name}.conv.weight"), &[output, input, 3, 3])?,
            norm: b.decoded(&format!("{name}.norm.weight"), &[output])?,
            channels: output,
        })
    }
    fn run(
        &self,
        input: &[f32],
        shape: ImageRows,
        epsilon: f32,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(Vec<f32>, ImageRows)> {
        let (mut rows, shape) =
            subsample_conv2d(input, shape, &self.weights, self.channels, cancelled)?;
        let bias = vec![0.; self.channels];
        for row in rows.chunks_exact_mut(self.channels) {
            layer_norm(row, &self.norm, &bias, epsilon)?;
            for x in row {
                *x = x.max(0.);
            }
        }
        Ok((rows, shape))
    }
}
pub(super) struct Subsampling {
    first: Subsample,
    second: Subsample,
    projection: Linear,
}
impl Subsampling {
    pub fn bind(b: &mut Binder<'_>, c: &super::config::Config) -> Result<Self> {
        let root = "model.audio_tower.subsample_conv_projection";
        let [a, z] = c.audio.subsampling_conv_channels;
        Ok(Self {
            first: Subsample::bind(b, &format!("{root}.layer0"), 1, a)?,
            second: Subsample::bind(b, &format!("{root}.layer1"), a, z)?,
            projection: b.linear(
                &format!("{root}.input_proj_linear"),
                (c.frontend.feature_size / 4) * z,
                c.audio.hidden_size,
                false,
            )?,
        })
    }
    pub fn run(
        &self,
        source: &SourceWeights,
        c: &super::config::Config,
        features: &[f32],
        frames: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        let shape = ImageRows {
            height: frames,
            width: c.frontend.feature_size,
            channels: 1,
        };
        let (first, shape) = self
            .first
            .run(features, shape, c.audio.rms_norm_eps, cancelled)?;
        let (second, _) = self
            .second
            .run(&first, shape, c.audio.rms_norm_eps, cancelled)?;
        drop(first);
        self.projection.rows(source, &second, cancelled)
    }
}
