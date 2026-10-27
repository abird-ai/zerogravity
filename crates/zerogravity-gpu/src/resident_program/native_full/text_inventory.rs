//! Exact original root/readback shapes, shared with real allocation admission.
use super::*;
use crate::cuda::native::text_model::TextModelBinding;
impl FullModelProgram {
    pub(crate) fn text_root_layout(scope: &TextModelBinding) -> Result<Vec<usize>> {
        let requirements = state_requirements(scope.record())?;
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
    pub(crate) fn text_readback_reservation(scopes: &[Arc<TextModelBinding>]) -> Result<usize> {
        let capture = scopes
            .first()
            .ok_or_else(|| Error::State("short-HF no actual readback scopes".into()))?
            .native_capture();
        if scopes.iter().any(|scope| scope.native_capture() != capture) {
            return Err(Error::State(
                "short-HF readback scopes mix native captures".into(),
            ));
        }
        value_storage::require_geometry_cohort(
            capture,
            &scopes
                .iter()
                .map(|s| s.call().rows().len())
                .collect::<Vec<_>>(),
        )?;
        let first = scopes
            .first()
            .ok_or_else(|| Error::State("short-HF no actual readback scopes".into()))?;
        let mut maximum = 0;
        for scope in scopes {
            if !first.source().same_issuer(scope.source())
                || first.hidden_size() != scope.hidden_size()
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
