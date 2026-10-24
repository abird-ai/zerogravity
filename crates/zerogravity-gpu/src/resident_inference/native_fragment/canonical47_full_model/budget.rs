use super::*;
pub(super) const PROVIDER: Option<usize> = Some(2_147_483_648);
pub(super) const HOST: Option<usize> = Some(1_369_203_644);
const MIB: usize = 1024 * 1024;
pub(super) const MATH_BYTES: usize = 164_249_876;
pub(super) const DESCRIPTORS: usize = 753;
fn require_gates(provider: Option<usize>, host: Option<usize>) -> Result<(usize, usize)> {
    Ok((
        provider.ok_or_else(|| {
            Error::Unsupported("canonical47 whole physical envelope unqualified".into())
        })?,
        host.ok_or_else(|| Error::Unsupported("canonical47 whole host bound unqualified".into()))?,
    ))
}
pub(super) fn require(cut: &Canonical47PrefixCut) -> Result<usize> {
    let (cap, host) = require_gates(PROVIDER, HOST)?;
    if !cut.is_whole()
        || cut.nodes().len() != 255
        || cut.values().len() != 327
        || cut.duals() != 48
        || cut.weight_groups() != 114
        || cut.source_windows() != 1008
        || !cut.carries().is_empty()
        || cut.scope().source().source().source().staging_chunk_bytes() != MIB
    {
        return Err(Error::State(
            "canonical47 whole exact source-bound capacity geometry".into(),
        ));
    }
    let public = cut
        .scope()
        .record()
        .plan()
        .values()
        .iter()
        .try_fold(0usize, |sum, value| {
            sum.checked_add(
                value
                    .capacity_elements()?
                    .checked_mul(2)
                    .ok_or_else(|| Error::Limit("whole public size overflow".into()))?,
            )
            .ok_or_else(|| Error::Limit("whole public sum overflow".into()))
        })?;
    if public != 78_999_424
        || crate::vllm_cuda_gdn::model_metadata_bound(
            crate::vllm_cuda_gdn::VllmGdnPhase::Initial47,
        )? != 2_178_088
        || !(1_974_915_812..=2 * 1024 * MIB).contains(&cap)
        || !(1_369_203_644..=1536 * MIB).contains(&host)
    {
        return Err(Error::Limit(
            "canonical47 whole reviewed envelope/source dimensions differ".into(),
        ));
    }
    Ok(cap)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whole_source_bound_counts_native_and_certificate_dispatches_separately() {
        let native = 18 * 8 + 6 * 12 + 1;
        let dispatches = 1 + 49 * 10 + native + 18 * 19 + 327 + 48 + 6;
        assert_eq!((native, dispatches), (217, 1431));
        assert_eq!(
            22 + 327 + 48 * 4 + 1 + 6 + 96 + 72 + 12 + 1 + 24,
            DESCRIPTORS
        );
        let math = 76
            + 194560
            + 78999424
            + 48 * 387072
            + 1560
            + 24
            + 65568768
            + 891072
            + 6144
            + 8576
            + 216;
        assert_eq!(math, MATH_BYTES);
        assert!(math > 128 * MIB && math < 5 * 32 * MIB);
        assert_eq!(
            (78_999_424 + 65_568_768 + 48 * 387072 + 194560 + MIB).div_ceil(32 * MIB),
            5
        );
        assert!(require_gates(None, None).is_err());
        assert!(require_gates(Some(2 * 1024 * MIB), None).is_err());
        assert!(require_gates(None, Some(1536 * MIB)).is_err());
    }
}
