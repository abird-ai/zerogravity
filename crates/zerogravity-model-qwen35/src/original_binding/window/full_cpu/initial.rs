//! Bounded copies of borrowed initial-core observations, charged before math.
use super::*;
pub use zerogravity_cpu::original_hf::gdn::InitialCoreStage;
const CHARGE: usize = 1024 * 1024;
pub(crate) struct InitialCoreTape {
    data: Vec<(InitialCoreStage, Vec<f32>)>,
    _charge: WindowWorkspace,
}
impl InitialCoreTape {
    pub(crate) fn prepare(source: &OriginalWindowOperators) -> Result<Arc<std::sync::Mutex<Self>>> {
        let (charge, _) = source.source().reserve_operator_buffers_split(CHARGE, 0)?;
        let mut data = Vec::new();
        data.try_reserve_exact(10)
            .map_err(|_| Error::Limit("initial core metadata".into()))?;
        Ok(Arc::new(std::sync::Mutex::new(Self {
            data,
            _charge: charge,
        })))
    }
    pub(crate) fn require_source(&self, source: &OriginalWindowOperators) -> Result<()> {
        if !source.source().same_issuer(self._charge.source()) {
            return Err(Error::State(
                "initial core tape original source admission".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn capture(&mut self, stage: InitialCoreStage, values: &[f32]) -> Result<()> {
        let n = match stage {
            InitialCoreStage::Convolution | InitialCoreStage::Activated => 10240,
            InitialCoreStage::QueryKey => 48,
            _ => 6144,
        };
        let used = self
            .data
            .iter()
            .map(|(_, v)| v.capacity() * 4 + 64)
            .sum::<usize>();
        if values.len() != n
            || values.iter().any(|v| !v.is_finite())
            || self.data.len() >= 10
            || self.data.iter().any(|(s, _)| *s == stage)
            || used + n * 4 + 64 > CHARGE
        {
            return Err(Error::State(
                "initial core exact finite stage/once-only capacity".into(),
            ));
        }
        let mut copy = Vec::new();
        copy.try_reserve_exact(n)
            .map_err(|_| Error::Limit("initial core admitted copy".into()))?;
        if copy.capacity() != n {
            return Err(Error::Limit("initial core exact copy capacity".into()));
        }
        copy.extend_from_slice(values);
        self.data.push((stage, copy));
        Ok(())
    }
    pub(crate) fn complete(&self) -> Result<()> {
        if self.data.len() != 10 {
            return Err(Error::State(
                "complete initial core observation cohort".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn values(&self, stage: InitialCoreStage) -> Result<&[f32]> {
        self.data
            .iter()
            .find(|(s, _)| *s == stage)
            .map(|(_, v)| v.as_slice())
            .ok_or_else(|| Error::State("initial core stage not observed".into()))
    }
}
