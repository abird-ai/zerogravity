//! Preparation remains cold and fail-closed until qualified provider adaptation.
use crate::{Error, Result, executable::ExecutableBundle};
use zerogravity_core::{placement::PlacementRequest, semantic::ForwardPlan};
pub fn prepare(_program: &ForwardPlan, _placement: &PlacementRequest) -> Result<ExecutableBundle> {
    Err(Error::Unsupported(
        "Fabric executable preparation awaits S2/F0 adapters".into(),
    ))
}
