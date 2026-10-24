//! Actual B4 call/site packet identities, separate from every logical row.
use super::super::cubin_manifests::ProjectionImage;
use super::CohortInputGeometry;
use crate::resident_inference::native_fragment::short_hf::ShortHfCallInput;
use cubecl::cuda::CubinKernelManifest;
use std::sync::Arc;
use zerogravity_core::{
    Error, Result,
    accelerator::program::{ShortHfCall, VllmBf16Mode},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Role {
    W,
    X,
    Y,
    Workspace,
    Packed,
    Q,
    K,
    Gate,
    QGamma,
    KGamma,
    Rotary,
    Positions,
    Cache,
    Slots,
    KScale,
    VScale,
    O,
    Lse,
    CuQ,
    DummyCuK,
    SeqUsed,
    Table,
    PrivateO,
    PrivateQ,
    PublicLse,
}
pub(super) struct Definition {
    pub(super) image: Option<ProjectionImage>,
    pub(super) manifest: CubinKernelManifest,
}
pub(super) struct Packet {
    pub(super) definition: usize,
    pub(super) roles: Vec<Role>,
    pub(super) clears: Vec<(Role, usize, usize)>,
}
/// One shared family shape for stage storage and original native record choice.
/// Context and feedback are runtime row data, never a common logical frontier.
pub(super) fn geometry(calls: &[&ShortHfCall]) -> Result<CohortInputGeometry> {
    if calls.len() != 4 {
        return Err(Error::Unsupported("B4 actual request population".into()));
    }
    let first = calls[0];
    let initial = first.span().mode() == VllmBf16Mode::Prefill;
    let logical_rows = first.span().len();
    if !super::CohortPhases::for_profile(first.model_profile()).contains(initial) {
        return Err(Error::Unsupported(
            "2B cohort requires genuine N1 prefill before cached B4".into(),
        ));
    }
    let mut max_base = 0;
    for call in calls {
        call.validate()?;
        let span = call.span();
        if call.model_profile() != first.model_profile()
            || span.workload().prompt_rows() != 47
            || span.source() != first.span().source()
            || span.mode() != first.span().mode()
            || span.len() != logical_rows
            || !matches!((span.len(), span.base()), (47, 0) | (1, 47..=53))
        {
            return Err(Error::Unsupported(
                "B4 original observed source/profile/span".into(),
            ));
        }
        max_base = max_base.max(span.base());
    }
    Ok(CohortInputGeometry {
        profile: first.model_profile(),
        initial,
        logical_rows,
        physical_rows: logical_rows * calls.len(),
        // Source FA uses the maximum key length; RMS remains row-local. This
        // selector never changes any row's span, workload or teacher issuer.
        native_call: if initial { 0 } else { max_base - 46 },
    })
}

pub(super) fn call(inputs: &[Arc<ShortHfCallInput>]) -> Result<usize> {
    Ok(super::input_geometry(inputs)?.native_call)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::{
        accelerator::SourceIdentity,
        accelerator::program::{FullModelProfile, ShortHfSpan, ShortHfWorkload},
    };

    fn cached(source: SourceIdentity, base: usize, id: u32) -> ShortHfCall {
        ShortHfCall::new(
            FullModelProfile::Qwen08,
            ShortHfSpan::cached(source, ShortHfWorkload::Prompt47, base, id).unwrap(),
        )
        .unwrap()
    }
    fn inspect(calls: &[ShortHfCall; 4]) -> CohortInputGeometry {
        geometry(&calls.each_ref()).unwrap()
    }

    #[test]
    fn cached2b_catalogue_cannot_mint_initial_or_wrong_source_geometry() {
        use zerogravity_core::accelerator::program::FullModelProfile;
        let definitions = super::super::generated_2b::definitions();
        assert_eq!(definitions.len(), 105);
        assert_eq!(super::super::generated_2b::DEFINITION_COUNT, 105);
        assert_eq!(
            super::super::CohortNativeLibrary::host_bytes(FullModelProfile::Qwen2B),
            13_542_960
        );
        for node in 0..255 {
            assert!(super::super::generated_2b::site(0, node).is_err());
        }
        for call in 1..8 {
            let mut sites = 0;
            for node in 0..255 {
                if let Ok(packets) = super::super::generated_2b::site(call, node) {
                    sites += 1;
                    assert!(!packets.is_empty());
                    for packet in packets {
                        let manifest = &definitions[packet.definition].manifest;
                        assert_eq!(packet.roles.len(), manifest.bindings.len());
                        assert!(manifest.bindings.len() <= 32);
                        for (role, offset, bytes) in packet.clears {
                            assert_eq!((role, offset, bytes), (Role::Workspace, 0, 128));
                        }
                    }
                }
            }
            assert_eq!(sites, 169);
        }
    }
    #[test]
    fn mixed_cached_geometry_preserves_actual_time_and_teacher_issuers() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let calls = [
            cached(source, 47, 11),
            cached(source, 48, 353),
            cached(source, 49, 17),
            cached(source, 50, 42),
        ];
        let originals = calls.clone();
        let shape = inspect(&calls);
        assert_eq!(
            shape,
            CohortInputGeometry {
                profile: FullModelProfile::Qwen08,
                initial: false,
                logical_rows: 1,
                physical_rows: 4,
                native_call: 4,
            }
        );
        for (actual, original) in calls.iter().zip(&originals) {
            assert_eq!(actual, original);
            assert_eq!(actual.span(), original.span());
            assert_eq!(
                actual.request().teachers().short_hf_span(),
                Some(original.span())
            );
            assert_eq!(actual.request().teachers().ids(), original.span().ids());
        }
        let permuted = [
            calls[3].clone(),
            calls[1].clone(),
            calls[0].clone(),
            calls[2].clone(),
        ];
        assert_eq!(inspect(&permuted), shape);
    }

    #[test]
    fn every_cached_frontier_combination_has_four_row_local_rms_inputs() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let issued: [ShortHfCall; 7] =
            std::array::from_fn(|index| cached(source, 47 + index, 11 + index as u32));
        for a in 0..7 {
            for b in 0..7 {
                for c in 0..7 {
                    for d in 0..7 {
                        let shape =
                            geometry(&[&issued[a], &issued[b], &issued[c], &issued[d]]).unwrap();
                        assert!(!shape.initial);
                        assert_eq!((shape.logical_rows, shape.physical_rows), (1, 4));
                        assert_eq!(
                            shape.native_call,
                            [a, b, c, d].into_iter().max().unwrap() + 1
                        );
                    }
                }
            }
        }
        let prefill = ShortHfCall::new(
            FullModelProfile::Qwen08,
            ShortHfSpan::prefill(source, ShortHfWorkload::Prompt47, &[11; 47]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            geometry(&[&prefill; 4]).unwrap(),
            CohortInputGeometry {
                profile: FullModelProfile::Qwen08,
                initial: true,
                logical_rows: 47,
                physical_rows: 188,
                native_call: 0,
            }
        );
    }

    #[test]
    fn two_b_cached_phase_keeps_own_source_profile_and_never_grants_initial_b4() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let calls: [ShortHfCall; 4] = std::array::from_fn(|row| {
            ShortHfCall::new(
                FullModelProfile::Qwen2B,
                ShortHfSpan::cached(source, ShortHfWorkload::Prompt47, 47 + row, 11 + row as u32)
                    .unwrap(),
            )
            .unwrap()
        });
        let originals = calls.clone();
        let shape = inspect(&calls);
        assert_eq!(shape.profile, FullModelProfile::Qwen2B);
        assert_eq!(
            (
                shape.initial,
                shape.logical_rows,
                shape.physical_rows,
                shape.native_call
            ),
            (false, 1, 4, 4)
        );
        assert_eq!(calls, originals);
        let initial = ShortHfCall::new(
            FullModelProfile::Qwen2B,
            ShortHfSpan::prefill(source, ShortHfWorkload::Prompt47, &[11; 47]).unwrap(),
        )
        .unwrap();
        assert!(geometry(&[&initial; 4]).is_err());
        let foreign_profile = cached(source, 47, 11);
        assert!(geometry(&[&calls[0], &calls[1], &calls[2], &foreign_profile]).is_err());
        assert_eq!(
            super::super::CohortPhases::for_profile(FullModelProfile::Qwen2B)
                .iter()
                .collect::<Vec<_>>(),
            [false]
        );
        assert_eq!(
            super::super::CohortPhases::for_profile(FullModelProfile::Qwen08)
                .iter()
                .collect::<Vec<_>>(),
            [true, false]
        );
    }

    #[test]
    fn incompatible_source_profile_phase_workload_or_population_stays_closed() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let good = cached(source, 47, 11);
        let foreign = cached(SourceIdentity::for_new_allocation().unwrap(), 50, 42);
        let profile = ShortHfCall::new(FullModelProfile::Qwen2B, good.span().clone()).unwrap();
        let workload = ShortHfCall::new(
            FullModelProfile::Qwen08,
            ShortHfSpan::cached(source, ShortHfWorkload::Prompt57, 57, 11).unwrap(),
        )
        .unwrap();
        let prefill = ShortHfCall::new(
            FullModelProfile::Qwen08,
            ShortHfSpan::prefill(source, ShortHfWorkload::Prompt47, &[11; 47]).unwrap(),
        )
        .unwrap();
        for incompatible in [&foreign, &profile, &workload, &prefill] {
            assert!(geometry(&[&good, &good, &good, incompatible]).is_err());
        }
        for width in [0, 1, 3, 5, 8] {
            assert!(geometry(&vec![&good; width]).is_err());
        }
    }
}
