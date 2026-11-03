//! Source-authenticated parameters widened exactly under the original initialization.
use super::{ResidentInitialization, ResidentLease};
use std::{collections::BTreeMap, sync::Arc};
use zerogravity_artifacts::source_weights::{
    ParameterWindow, VerifiedTensorSource, WindowWorkspace,
};
use zerogravity_core::{
    Error, Result, accelerator::program::SourceParameter, device_events::BufferRole,
    encoding::StoredTensorEncoding,
};
fn aligned(n: usize) -> Result<usize> {
    n.checked_add(255)
        .map(|n| n & !255)
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Limit("source parameter aligned extent".into()))
}
struct ParameterUpload {
    _window: ParameterWindow,
    _bytes: Vec<u8>,
    _staging: WindowWorkspace,
}
pub(crate) fn upload_parameters(
    source: &VerifiedTensorSource,
    declared: &BTreeMap<String, SourceParameter>,
    init: &mut ResidentInitialization<'_>,
) -> Result<BTreeMap<String, ResidentLease>> {
    let mut parameters = BTreeMap::new();
    for (name, p) in declared {
        let window = source.read_declared_parameter(p, None)?;
        let bytes = aligned(window.parameter().elements() * 4)?;
        let staging = source.reserve_workspace(3 * bytes)?;
        let mut widened = vec![0u8; bytes];
        match p.kind {
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::F32) => {
                widened[..p.bytes].copy_from_slice(window.as_slice())
            }
            StoredTensorEncoding::Scalar(zerogravity_core::encoding::ScalarDType::Bf16) => {
                for (out, input) in widened
                    .chunks_exact_mut(4)
                    .zip(window.as_slice().chunks_exact(2))
                {
                    out.copy_from_slice(
                        &(u32::from(u16::from_le_bytes(input.try_into().unwrap())) << 16)
                            .to_le_bytes(),
                    );
                }
            }
            _ => return Err(Error::State("source parameter physical dtype".into())),
        }
        if widened[..window.parameter().elements() * 4]
            .chunks_exact(4)
            .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
        {
            return Err(Error::Malformed("source raw parameter nonfinite".into()));
        }
        let owner = Arc::new(ParameterUpload {
            _window: window,
            _bytes: widened,
            _staging: staging,
        });
        // The original initialization retains the authentic window BEFORE copy.
        init.retain_completion_owner(owner.clone())?;
        parameters.insert(
            name.clone(),
            init.upload_bytes(&owner._bytes, BufferRole::ImmutableWeights)?,
        );
    }
    Ok(parameters)
}
