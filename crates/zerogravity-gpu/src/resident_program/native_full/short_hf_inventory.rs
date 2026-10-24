//! Exact original root/readback shapes, shared with real allocation admission.
use super::*;
use crate::vllm_cuda_qualification::short_hf_model::ShortHfModelBinding;
impl FullModelProgram {
    pub(crate) fn short_hf_root_layout(scope: &ShortHfModelBinding) -> Result<Vec<usize>> {
        let requirements = state_requirements(scope.source().source(), scope.record().plan())?;
        let mut layout = Vec::with_capacity(48);
        for requirement in requirements {
            let (count, elements) = requirement.layout.planes()?;
            let bytes = elements
                .checked_mul(4)
                .ok_or_else(|| Error::Limit("short-HF root plane overflow".into()))?;
            layout.extend((0..count).map(|_| bytes));
        }
        if layout.len() != 48 {
            return Err(Error::State("short-HF42 domains/48 original planes".into()));
        }
        Ok(layout)
    }
    pub(crate) fn short_hf_readback_reservation(
        scopes: &[Arc<ShortHfModelBinding>],
    ) -> Result<usize> {
        value_storage::require_geometry_cohort(
            &scopes
                .iter()
                .map(|s| s.call().span().len())
                .collect::<Vec<_>>(),
        )?;
        let first = scopes
            .first()
            .ok_or_else(|| Error::State("short-HF no actual readback scopes".into()))?;
        let mut maximum = 0;
        for scope in scopes {
            if !Arc::ptr_eq(first.source().source(), scope.source().source())
                || first.call().model_profile() != scope.call().model_profile()
            {
                return Err(Error::State(
                    "short-HF readback exact source/profile".into(),
                ));
            }
            maximum = maximum.max(value_storage::actual_diagnostic_bytes(scope)?);
        }
        value_storage::readback_chunks(maximum)?
            .checked_mul(32 * 1024 * 1024)
            .ok_or_else(|| Error::Limit("short-HF shared D2H ticket overflow".into()))
    }
}
