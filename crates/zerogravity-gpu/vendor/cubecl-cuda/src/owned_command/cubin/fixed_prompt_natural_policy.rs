//! Finite two-geometry/current-call union. Old policies remain exact.
use super::{CubinKernelManifest, VerifiedCubin, failure};
use cubecl_core::server::ServerError;
const CAPACITY: Option<(usize, usize, usize)> = Some((15, 18, 38));
// Exact (whole image SHA, complete ABI digest), never entry/shape-only matching.
mod records;
const RECORDS: Option<&[([u8; 32], [u8; 32])]> = Some(records::RECORDS);
pub(super) fn limits(m: usize, f: usize, r: usize, nm: bool, nf: bool) -> Result<(), ServerError> {
    let (mm, ff, rr) = CAPACITY
        .ok_or_else(|| failure("canonical natural complete union capacity pending ROOT"))?;
    if m > mm || f > ff || r >= rr || (nm && m == mm) || (nf && f == ff) {
        return Err(failure("canonical natural finite source union capacity"));
    }
    Ok(())
}
pub(super) fn require(
    image: &VerifiedCubin,
    manifest: &CubinKernelManifest,
) -> Result<(), ServerError> {
    if image.derived_from().is_none() && image.captured_from().is_none() {
        return Err(failure(
            "canonical natural requires original fixed executable issuer",
        ));
    }
    require_record(image.sha256(), manifest, RECORDS)
}
fn require_record(
    image: [u8; 32],
    m: &CubinKernelManifest,
    records: Option<&[([u8; 32], [u8; 32])]>,
) -> Result<(), ServerError> {
    let records = records
        .ok_or_else(|| failure("canonical natural complete source ABI cohort pending ROOT"))?;
    let abi = super::attention_policy::digest(m);
    if records
        .iter()
        .filter(|(i, a)| *i == image && *a == abi)
        .count()
        != 1
    {
        return Err(failure("canonical natural complete immutable ABI differs"));
    }
    Ok(())
}
pub(crate) fn require_n1_copy(
    image: &VerifiedCubin,
    m: &CubinKernelManifest,
) -> Result<(), ServerError> {
    require(image, m)?;
    let pin = super::PinnedCapturedBf16Cubin::AttentionFaSplitSm80;
    if image.captured_from() != Some(pin) || image.derived_from().is_some() {
        return Err(failure("canonical cached FA exact selected resource"));
    }
    pin.require_image(image.byte_len(), image.sha256(), image.binary_sm())?;
    if m.parameters.len() != 1
        || m.parameters[0].offset != 0
        || m.parameters[0].template.len() != 472
        || m.grid != [1, 1, 2]
        || m.block != [128, 1, 1]
        || m.dynamic_shared_bytes != 98304
    {
        return Err(failure("canonical cached FA source copy record"));
    }
    let k = i32::from_le_bytes(m.parameters[0].template[184..188].try_into().unwrap());
    if !(48..=54).contains(&k) {
        return Err(failure("canonical cached FA exact seven key lengths"));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_union_and_cross_image_never_authorize() {
        let m = super::super::head_policy::Bf16VocabularyHead::manifest();
        let image = [1; 32];
        assert!(require_record(image, &m, None).is_err());
        let r = [(image, super::super::attention_policy::digest(&m))];
        require_record(image, &m, Some(&r)).unwrap();
        assert!(require_record([2; 32], &m, Some(&r)).is_err());
        let mut n = m.clone();
        n.parameters[0].template[0] ^= 1;
        assert!(require_record(image, &n, Some(&r)).is_err());
        let dup = [r[0], r[0]];
        assert!(require_record(image, &m, Some(&dup)).is_err());
        assert!(limits(15, 18, 37, false, false).is_ok());
        assert!(limits(15, 18, 38, false, false).is_err());
        assert!(limits(15, 18, 0, true, false).is_err());
        assert!(limits(15, 18, 0, false, true).is_err());
    }
}
