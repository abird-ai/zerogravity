//! One pooled Initial47 full-model union; predecessor scopes remain closed.
use super::{CubinKernelManifest, VerifiedCubin, failure};
use cubecl_core::server::ServerError;
const WHOLE_MEMORY: Option<()> = Some(());
fn require_gate(gate: Option<()>) -> Result<(), ServerError> {
    gate.ok_or_else(|| failure("canonical47 whole memory/source view unqualified"))
}
pub(super) fn require(
    image: &VerifiedCubin,
    manifest: &CubinKernelManifest,
) -> Result<(), ServerError> {
    require_gate(WHOLE_MEMORY)?;
    if super::canonical47_layer3_policy::require(image, manifest).is_ok() {
        return Ok(());
    }
    super::head_policy::require(image, manifest)
}
#[cfg(test)]
mod tests {
    use super::super::policy::CubinSetPolicy as P;
    #[test]
    fn full_capacity_is_distinct_from_all_predecessors() {
        assert!(super::require_gate(None).is_err());
        assert!(P::Canonical47Full.limits(12, 15, 15, false, false).is_ok());
        for (m, f, r, nm, nf) in [
            (12, 15, 16, false, false),
            (13, 15, 15, false, false),
            (12, 16, 15, false, false),
            (12, 15, 15, true, false),
            (12, 15, 15, false, true),
        ] {
            assert!(P::Canonical47Full.limits(m, f, r, nm, nf).is_err());
        }
        for old in [
            P::Canonical47Layer3,
            P::Canonical47Fa,
            P::Canonical47QkKv,
            P::Canonical47Qkv,
            P::Canonical47Mlp,
        ] {
            assert!(old.limits(12, 15, 15, false, false).is_err());
        }
    }
}
