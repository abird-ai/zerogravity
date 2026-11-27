//! Original HIP device envelope; context is a partition, never an extra charge.
use super::{CONTEXT_DEVICE, DEVICE_CAP, GIB};
use zerogravity_core::{Error, Result};

/// Cold source-service limits. Host and native compiler/cache policy stay unchanged.
/// Default preserves the historical 24GiB envelope and 64GiB admission ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HipSourceBudget {
    total: usize,
    admission: u64,
}
impl Default for HipSourceBudget {
    fn default() -> Self {
        Self {
            total: 24 * GIB,
            admission: DEVICE_CAP,
        }
    }
}
impl HipSourceBudget {
    /// Explicit total device envelope, including the original 4GiB context.
    /// Physical admission is capped at the same total; actual hardware must fit it.
    pub fn new(total_device_bytes: u64) -> Result<Self> {
        let total = usize::try_from(total_device_bytes)
            .map_err(|_| Error::Limit("HIP source device envelope overflow".into()))?;
        if total_device_bytes <= CONTEXT_DEVICE || total > isize::MAX as usize {
            return Err(Error::Limit(
                "HIP source total device envelope must exceed context".into(),
            ));
        }
        Ok(Self {
            total,
            admission: total_device_bytes,
        })
    }
    /// Fit an explicit envelope or adopt the observed post-headroom capacity.
    /// Explicit values cannot silently shrink, including during a comparison.
    pub fn within_available(usable: u64, requested: Option<u64>) -> Result<Self> {
        let envelope = requested.unwrap_or(usable);
        if envelope > usable {
            return Err(Error::resource(
                "HIP device memory",
                "explicit serving envelope after headroom",
                envelope,
                usable,
            ));
        }
        Self::new(envelope)
    }

    pub fn total_device_bytes(self) -> usize {
        self.total
    }
    pub fn physical_admission_bytes(self) -> u64 {
        self.admission
    }
    pub fn context_device_bytes(self) -> usize {
        CONTEXT_DEVICE as usize
    }
    pub fn model_device_bytes(self) -> usize {
        self.total - self.context_device_bytes()
    }
    pub(crate) fn inventory(self, model_upper: usize) -> Result<(usize, usize)> {
        let context = self.context_device_bytes();
        let total = model_upper
            .checked_add(context)
            .ok_or_else(|| Error::Limit("resident source physical upper overflow".into()))?;
        if total > self.total {
            return Err(Error::resource(
                "device memory / VRAM budget",
                "resident source pre-client device/native/root admission",
                total as u64,
                self.total as u64,
            ));
        }
        Ok((context, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn context_is_inside_each_total_and_defaults_do_not_change() {
        let legacy = HipSourceBudget::default();
        assert_eq!(
            (legacy.total_device_bytes(), legacy.model_device_bytes()),
            (24 * GIB, 20 * GIB)
        );
        assert_eq!(legacy.physical_admission_bytes(), 64 * GIB as u64);
        for gib in [24, 64, 75] {
            let budget = HipSourceBudget::new((gib * GIB) as u64).unwrap();
            assert_eq!(budget.context_device_bytes(), 4 * GIB);
            assert_eq!(budget.model_device_bytes(), (gib - 4) * GIB);
            assert_eq!(budget.physical_admission_bytes(), (gib * GIB) as u64);
            assert_eq!(
                budget.inventory((gib - 4) * GIB).unwrap(),
                (4 * GIB, gib * GIB)
            );
            assert!(budget.inventory((gib - 4) * GIB + 1).is_err());
            assert!(budget.inventory(usize::MAX).is_err());
        }
        for total in [0, CONTEXT_DEVICE - 1, CONTEXT_DEVICE, u64::MAX] {
            assert!(HipSourceBudget::new(total).is_err());
        }
        let available = (96 * GIB) as u64;
        let cap = (24 * GIB) as u64;
        assert_eq!(
            HipSourceBudget::within_available(available, Some(cap))
                .unwrap()
                .physical_admission_bytes(),
            cap
        );
        assert_eq!(
            HipSourceBudget::within_available(available, None)
                .unwrap()
                .physical_admission_bytes(),
            available
        );
        assert!(matches!(
            HipSourceBudget::within_available(cap - 1, Some(cap)),
            Err(Error::Resource(_))
        ));
        assert_eq!(HipSourceBudget::default(), legacy);
    }
}
