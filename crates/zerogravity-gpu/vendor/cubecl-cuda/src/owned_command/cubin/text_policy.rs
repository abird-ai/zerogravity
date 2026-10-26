//! Compiled genuine profile/workload/call packets; no runtime manifest authority.
use super::{CubinKernelManifest, VerifiedCubin, failure, policy::CubinSetPolicy};
use cubecl_core::server::ServerError;
mod cohort2b;
mod cohort4;
mod records;
const COHORT4: records::Cohort = records::Cohort {
    calls: 48,
    limits: (17, cohort4::FUNCTIONS, 164 + cohort4::RULES.len()),
    rules: cohort4::RULES,
};
const COHORT2B: records::Cohort = records::Cohort {
    calls: 47, // original five×eight N1 calls +seven actual cached B4 calls
    limits: (
        cohort2b::MODULES,
        cohort2b::FUNCTIONS,
        cohort2b::ORIGINAL_RULES + cohort2b::RULES.len(),
    ),
    rules: cohort2b::RULES,
};
fn cohort(policy: CubinSetPolicy) -> Result<&'static records::Cohort, ServerError> {
    match policy {
        CubinSetPolicy::Text08 => Ok(&records::QWEN08),
        CubinSetPolicy::Text08Cohort4 => Ok(&COHORT4),
        CubinSetPolicy::Text2B => Ok(&records::QWEN2B),
        CubinSetPolicy::Text2BCachedCohort4 => Ok(&COHORT2B),
        _ => Err(failure("short-HF exact separate source policy")),
    }
}
pub(crate) fn require_complete(policy: CubinSetPolicy) -> Result<(), ServerError> {
    let c = cohort(policy)?;
    let base = match policy {
        CubinSetPolicy::Text08Cohort4 => Some(&records::QWEN08),
        CubinSetPolicy::Text2BCachedCohort4 => Some(&records::QWEN2B),
        _ => None,
    };
    let calls = match policy {
        CubinSetPolicy::Text08Cohort4 => 48,
        CubinSetPolicy::Text2BCachedCohort4 => 47,
        _ => 40,
    };
    if (policy == CubinSetPolicy::Text2BCachedCohort4
        && cohort2b::ORIGINAL_RULES != records::QWEN2B.rules.len())
        || c.calls != calls
        || c.rules.is_empty()
        || c.limits.0 == 0
        || c.limits.1 == 0
        || c.limits.2 != c.rules.len() + base.map_or(0, |source| source.rules.len())
    {
        return Err(failure(
            "short-HF all five-workload/eight-call source ABI cohort unavailable",
        ));
    }
    Ok(())
}
pub(super) fn limits(
    policy: CubinSetPolicy,
    m: usize,
    f: usize,
    r: usize,
    nm: bool,
    nf: bool,
) -> Result<(), ServerError> {
    require_complete(policy)?;
    let (mm, ff, rr) = cohort(policy)?.limits;
    if m > mm || f > ff || r >= rr || (nm && m == mm) || (nf && f == ff) {
        return Err(failure(
            "short-HF exact compiled module/function/record capacity",
        ));
    }
    Ok(())
}
pub(crate) fn require(
    policy: CubinSetPolicy,
    image: &VerifiedCubin,
    m: &CubinKernelManifest,
) -> Result<(), ServerError> {
    require_complete(policy)?;
    if image.derived_from().is_none() && image.captured_from().is_none() {
        return Err(failure(
            "short-HF original compiled executable issuer absent",
        ));
    }
    // Same original full source packet/image/function replaces its streamed
    // record without changing the admitted module/function/record ceilings.
    if super::head_policy::resident::matches_policy(policy, image, m).is_ok() {
        return Ok(());
    }
    let image = image.sha256();
    let abi = super::attention_policy::digest(m);
    match policy {
        CubinSetPolicy::Text08Cohort4 => require_cohort_rule(false, image, abi),
        CubinSetPolicy::Text2BCachedCohort4 => require_cohort_rule(true, image, abi),
        _ => require_rule(cohort(policy)?.rules, image, abi),
    }
}
#[cfg(test)]
fn require_cohort4_rule(image: [u8; 32], abi: [u8; 32]) -> Result<(), ServerError> {
    require_cohort_rule(false, image, abi)
}
fn require_cohort_rule(two_b: bool, image: [u8; 32], abi: [u8; 32]) -> Result<(), ServerError> {
    let (base, additions) = if two_b {
        (&records::QWEN2B, cohort2b::RULES)
    } else {
        (&records::QWEN08, cohort4::RULES)
    };
    if base
        .rules
        .iter()
        .chain(additions)
        .filter(|rule| rule.image == image && rule.abi == abi)
        .count()
        != 1
    {
        return Err(failure("B4 exact separate full-image/full-ABI catalogue"));
    }
    Ok(())
}
fn require_rule(
    rules: &[records::Rule],
    image: [u8; 32],
    abi: [u8; 32],
) -> Result<(), ServerError> {
    if rules
        .iter()
        .filter(|r| r.image == image && r.abi == abi)
        .count()
        != 1
    {
        return Err(failure(
            "short-HF exact whole image and complete immutable ABI differs",
        ));
    }
    Ok(())
}
pub(crate) fn require_n1_copy(
    image: &VerifiedCubin,
    m: &CubinKernelManifest,
) -> Result<(), ServerError> {
    let abi = super::attention_policy::digest(m);
    let count = [&records::QWEN08, &records::QWEN2B]
        .into_iter()
        .flat_map(|c| c.rules)
        .filter(|r| r.copy && r.image == image.sha256() && r.abi == abi)
        .count();
    // The same exact copy ABI can be shared by the two genuine family catalogues.
    if count == 0 || count > 2 || image.captured_from().is_none() || image.derived_from().is_some()
    {
        return Err(failure("short-HF exact compiled cached FA copy ABI absent"));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn b4_opt_in_preserves_original_n1_boundary_and_exact_union_ceilings() {
        let old = CubinSetPolicy::Text08;
        let b4 = CubinSetPolicy::Text08Cohort4;
        require_complete(old).unwrap();
        require_complete(b4).unwrap();
        assert_eq!(cohort(old).unwrap().limits, (17, 27, 164));
        assert_eq!(cohort(b4).unwrap().limits, (17, 33, 195));
        for rule in cohort4::RULES {
            assert!(require_rule(records::QWEN08.rules, rule.image, rule.abi).is_err());
            require_cohort4_rule(rule.image, rule.abi).unwrap();
            let mut image = rule.image;
            image[0] ^= 1;
            assert!(require_cohort4_rule(image, rule.abi).is_err());
            let mut abi = rule.abi;
            abi[31] ^= 1;
            assert!(require_cohort4_rule(rule.image, abi).is_err());
        }
        assert!(limits(old, 17, 27, 163, false, true).is_err());
        assert!(limits(b4, 17, 32, 194, false, true).is_ok());
        assert!(limits(b4, 17, 33, 194, false, true).is_err());
        assert!(limits(b4, 17, 33, 195, false, false).is_err());
    }
    #[test]
    fn cached2b_keeps_own_n1_rules_and_actual_separate_union_ceilings() {
        let old = CubinSetPolicy::Text2B;
        let cached = CubinSetPolicy::Text2BCachedCohort4;
        require_complete(old).unwrap();
        require_complete(cached).unwrap();
        assert_eq!(cohort(old).unwrap().limits, (18, 28, 154));
        assert_eq!(cohort(cached).unwrap().limits, (19, 32, 173));
        assert_eq!(cohort(cached).unwrap().calls, 47);
        for rule in records::QWEN2B.rules {
            require_cohort_rule(true, rule.image, rule.abi).unwrap();
        }
        for rule in cohort2b::RULES {
            assert!(require_rule(records::QWEN2B.rules, rule.image, rule.abi).is_err());
            require_cohort_rule(true, rule.image, rule.abi).unwrap();
            let mut image = rule.image;
            image[0] ^= 1;
            assert!(require_cohort_rule(true, image, rule.abi).is_err());
        }
        let extra_image =
            crate::owned_command::bf16_gemm::PinnedCapturedBf16Cubin::Cohort2BAmpere64SlicedSm80
                .capture()
                .image_sha256;
        let extra = cohort2b::RULES
            .iter()
            .filter(|rule| rule.image == extra_image)
            .collect::<Vec<_>>();
        assert_eq!(extra.len(), 1);
        assert!(require_cohort_rule(false, extra[0].image, extra[0].abi).is_err());
        assert!(limits(old, 18, 28, 153, false, true).is_err());
        assert!(limits(cached, 19, 31, 172, false, true).is_ok());
        assert!(limits(cached, 19, 32, 172, false, true).is_err());
        assert!(limits(cached, 19, 32, 173, false, false).is_err());
    }
    #[test]
    fn full_image_full_abi_and_uniqueness_are_independent() {
        let r = records::Rule {
            image: [1; 32],
            abi: [2; 32],
            copy: false,
        };
        assert!(require_rule(&[r], [1; 32], [2; 32]).is_ok());
        assert!(require_rule(&[r], [3; 32], [2; 32]).is_err());
        assert!(require_rule(&[r], [1; 32], [3; 32]).is_err());
        assert!(require_rule(&[r, r], [1; 32], [2; 32]).is_err());
        assert!(require_rule(&[], [1; 32], [2; 32]).is_err());
    }
}
