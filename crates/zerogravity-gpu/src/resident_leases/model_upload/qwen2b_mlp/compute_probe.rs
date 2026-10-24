//! Closed deliberate cuts after completed source comparison; no GPU callback/hook.
use super::*;
use std::sync::atomic::AtomicUsize;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Qwen2bMlpComputeReadCut {
    UnknownFirst,
    UnknownMiddle,
    UnknownLast,
    CancelFirst,
    CancelMiddle,
    CancelLast,
}
impl Qwen2bMlpComputeReadCut {
    fn point(self) -> (usize, bool) {
        match self {
            Self::UnknownFirst => (1, false),
            Self::UnknownMiddle => (37, false),
            Self::UnknownLast => (73, false),
            Self::CancelFirst => (1, true),
            Self::CancelMiddle => (37, true),
            Self::CancelLast => (73, true),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Qwen2bMlpComputeProgress {
    pub completed_reads: usize,
    pub compared_source_bytes: usize,
    pub trusted_validator_entered: bool,
}
pub(super) struct ComputeReadProbe {
    cut: Option<Qwen2bMlpComputeReadCut>,
    completed: AtomicUsize,
    bytes: AtomicUsize,
    validator: AtomicBool,
}
impl ComputeReadProbe {
    pub(super) fn new(cut: Option<Qwen2bMlpComputeReadCut>) -> Self {
        Self {
            cut,
            completed: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            validator: AtomicBool::new(false),
        }
    }
    pub(super) fn progress(&self) -> Qwen2bMlpComputeProgress {
        Qwen2bMlpComputeProgress {
            completed_reads: self.completed.load(Ordering::Acquire),
            compared_source_bytes: self.bytes.load(Ordering::Acquire),
            trusted_validator_entered: self.validator.load(Ordering::Acquire),
        }
    }
    pub(super) fn after_completed_read(&self, count: usize, bytes: usize) -> Result<()> {
        if !(1..=73).contains(&count)
            || self.validator.load(Ordering::Acquire)
            || self
                .completed
                .compare_exchange(count - 1, count, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(Error::State(
                "compute source plan cannot replay/reorder/reset".into(),
            ));
        }
        let total = self
            .bytes
            .fetch_add(bytes, Ordering::AcqRel)
            .checked_add(bytes)
            .ok_or_else(|| Error::Limit("compute source comparison counter".into()))?;
        if total != compared_at(count)? {
            return Err(Error::State(
                "actual complete compute source prefix bytes".into(),
            ));
        }
        if let Some(cut) = self.cut
            && cut.point().0 == count
        {
            if cut.point().1 {
                let flag = AtomicBool::new(true);
                not_cancelled(Some(&flag))?;
            }
            return Err(Error::State(
                "deliberate qwen2b compute unknown after completed source read".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn enter_validator(&self) -> Result<()> {
        let p = self.progress();
        if p.completed_reads != 73
            || p.compared_source_bytes != 75_497_472
            || self
                .validator
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(Error::State(
                "all73 source reads precede one compute validator".into(),
            ));
        }
        Ok(())
    }
}
fn compared_at(count: usize) -> Result<usize> {
    match count {
        0..=48 => Ok(count * 1024 * 1024),
        49..=72 => Ok(48 * 1024 * 1024 + (count - 48) * 85 * 12288),
        73 => Ok(75_497_472),
        _ => Err(Error::State("finite73 source read itinerary".into())),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn bytes(count: usize) -> usize {
        compared_at(count).unwrap() - compared_at(count - 1).unwrap()
    }
    #[test]
    fn source_cut_is_after_real_progress_and_never_resets() {
        for cut in [
            Qwen2bMlpComputeReadCut::UnknownFirst,
            Qwen2bMlpComputeReadCut::UnknownMiddle,
            Qwen2bMlpComputeReadCut::UnknownLast,
            Qwen2bMlpComputeReadCut::CancelFirst,
            Qwen2bMlpComputeReadCut::CancelMiddle,
            Qwen2bMlpComputeReadCut::CancelLast,
        ] {
            let p = ComputeReadProbe::new(Some(cut));
            for n in 1..cut.point().0 {
                p.after_completed_read(n, bytes(n)).unwrap();
            }
            assert!(
                p.after_completed_read(cut.point().0, bytes(cut.point().0))
                    .is_err()
            );
            assert_eq!(p.progress().completed_reads, cut.point().0);
            assert_eq!(
                p.progress().compared_source_bytes,
                compared_at(cut.point().0).unwrap()
            );
            assert!(!p.progress().trusted_validator_entered);
            assert!(p.after_completed_read(1, bytes(1)).is_err());
        }
    }
    #[test]
    fn all_source_words_before_single_validator_and_wrong_tail_refuses() {
        let p = ComputeReadProbe::new(None);
        assert!(p.enter_validator().is_err());
        for n in 1..=73 {
            p.after_completed_read(n, bytes(n)).unwrap();
        }
        p.enter_validator().unwrap();
        assert!(p.enter_validator().is_err());
        assert!(p.after_completed_read(1, bytes(1)).is_err());
        let p = ComputeReadProbe::new(None);
        for n in 1..73 {
            p.after_completed_read(n, bytes(n)).unwrap();
        }
        assert!(p.after_completed_read(73, bytes(73) - 2).is_err());
        assert!(p.enter_validator().is_err());
    }
}
