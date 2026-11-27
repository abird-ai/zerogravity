//! Semicausal periodic-Hann magnitude log-mel, original HTK filter bank.
use super::config::Frontend;
use crate::{Error, Result};
use std::sync::atomic::AtomicBool;
use zerogravity_cpu::ops::{audio::MagnitudeFft, encoder::check_cancel};

pub(super) struct MelFrontend {
    window: Vec<f32>,
    filters: Vec<f64>,
    fft: MagnitudeFft,
}
impl MelFrontend {
    pub fn new(c: &Frontend) -> Result<Self> {
        let window = (0..c.frame_length)
            .map(|i| {
                (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / c.frame_length as f64).cos())
                    as f32
            })
            .collect();
        let bins = c.fft_length / 2 + 1;
        let to_mel = |hz: f64| 2595. * (1. + hz / 700.).log10();
        let lo = to_mel(c.min_frequency);
        let hi = to_mel(c.max_frequency);
        let centers: Vec<_> = (0..c.feature_size + 2)
            .map(|i| {
                let mel = lo + (hi - lo) * i as f64 / (c.feature_size + 1) as f64;
                700. * (10f64.powf(mel / 2595.) - 1.)
            })
            .collect();
        let mut filters = Vec::with_capacity(c.feature_size * bins);
        for triple in centers.windows(3) {
            for i in 0..bins {
                let hz = f64::from(c.sampling_rate) / 2. * i as f64 / (bins - 1) as f64;
                filters.push(
                    ((hz - triple[0]) / (triple[1] - triple[0]))
                        .min((triple[2] - hz) / (triple[2] - triple[1]))
                        .max(0.),
                );
            }
        }
        Ok(Self {
            window,
            filters,
            fft: MagnitudeFft::new(c.fft_length)?,
        })
    }
    pub fn frames(c: &Frontend, samples: usize) -> Result<usize> {
        samples
            .checked_sub(c.frame_length / 2 + 1)
            .map(|n| n / c.hop_length + 1)
            .ok_or_else(|| {
                Error::InvalidInput("audio too short for original analysis window".into())
            })
    }
    pub fn extract(
        &self,
        c: &Frontend,
        samples: &[f32],
        frames: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        let bins = self.fft.size() / 2 + 1;
        let mut frame = vec![0.; c.frame_length];
        let mut scratch = vec![[0.; 2]; self.fft.size()];
        let mut magnitude = vec![0.; bins];
        let mut out = Vec::with_capacity(frames * c.feature_size);
        for t in 0..frames {
            check_cancel(cancelled)?;
            for (i, (v, &w)) in frame.iter_mut().zip(&self.window).enumerate() {
                let sample = (t * c.hop_length + i)
                    .checked_sub(c.frame_length / 2)
                    .map_or(0., |index| samples[index]);
                *v = sample * w;
            }
            self.fft.magnitude(&frame, &mut scratch, &mut magnitude)?;
            out.extend(self.filters.chunks_exact(bins).map(|filter| {
                (c.mel_floor
                    + filter
                        .iter()
                        .zip(&magnitude)
                        .map(|(w, v)| w * v)
                        .sum::<f64>())
                .ln() as f32
            }));
        }
        Ok(out)
    }
}
