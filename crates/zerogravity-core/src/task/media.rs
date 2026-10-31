//! Descriptive media clocks/frontiers, not decoded payloads or publication receipts.
use super::TimeBase;
use crate::{Error, Result};
use std::num::NonZeroU128;

/// Future source clock domain, never a device queue epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaClockId(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IngressMilestone {
    Received,
    Decoded,
    Consumed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaPosition {
    pub clock: MediaClockId,
    pub time_base: TimeBase,
    pub ticks: u64,
    pub turn_generation: u64,
    pub output_generation: u64,
    pub revision: u64,
}
/// Exact fractional target tick; callers must retain, not silently round, residue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvertedTicks {
    pub whole: u64,
    pub remainder: u128,
    pub denominator: NonZeroU128,
}
impl TimeBase {
    pub fn convert_ticks(self, ticks: u64, target: Self) -> Result<ConvertedTicks> {
        let numerator = u128::from(ticks)
            .checked_mul(u128::from(self.numerator.get()))
            .and_then(|n| n.checked_mul(u128::from(target.denominator.get())))
            .ok_or_else(|| Error::Limit("media clock conversion numerator".into()))?;
        // Product of two nonzero u64 values always fits nonzero u128.
        let denominator = u128::from(self.denominator.get()) * u128::from(target.numerator.get());
        let whole = u64::try_from(numerator / denominator)
            .map_err(|_| Error::Limit("media clock converted ticks".into()))?;
        let denominator = NonZeroU128::new(denominator)
            .ok_or_else(|| Error::State("media clock denominator".into()))?;
        Ok(ConvertedTicks {
            whole,
            remainder: numerator % denominator.get(),
            denominator,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn base(numerator: u64, denominator: u64) -> TimeBase {
        TimeBase {
            numerator: numerator.try_into().unwrap(),
            denominator: denominator.try_into().unwrap(),
        }
    }
    #[test]
    fn fabric_media_conversion_retains_fraction_and_refuses_overflow() {
        let sample = base(1, 48000).convert_ticks(1, base(1, 44100)).unwrap();
        assert_eq!(
            (sample.whole, sample.remainder, sample.denominator.get()),
            (0, 44100, 48000)
        );
        let second = base(1, 48000).convert_ticks(48000, base(1, 44100)).unwrap();
        assert_eq!((second.whole, second.remainder), (44100, 0));
        assert!(
            base(u64::MAX, 1)
                .convert_ticks(u64::MAX, base(1, u64::MAX))
                .is_err()
        );
    }
}
