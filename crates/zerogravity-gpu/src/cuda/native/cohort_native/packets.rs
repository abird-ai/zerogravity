//! Actual B4 call/site packet identities, separate from every logical row.
use super::super::cubin_manifests::ProjectionImage;
use super::super::text_native::TextKernelCapture;
use super::CohortInputGeometry;
use crate::resident_inference::native_fragment::text::TextCallInput;
use cubecl::cuda::CubinKernelManifest;
use std::sync::Arc;
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::program::SourceCall;
use zerogravity_core::accelerator::program::SourceCallMode;

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
/// One shared captured shape for stage storage and original native record choice.
/// Context and feedback are runtime row data, never a common logical frontier.
pub(super) fn geometry(
    calls: &[(&SourceCall, usize, TextKernelCapture)],
) -> Result<CohortInputGeometry> {
    if calls.len() != 4 {
        return Err(Error::Unsupported("B4 actual request population".into()));
    }
    let (first, _, capture) = calls[0];
    let initial = first.mode() == SourceCallMode::Prefill;
    let logical_rows = first.rows().len();
    if !super::CohortPhases::for_capture(capture).contains(initial) {
        return Err(Error::Unsupported(
            "cached-only native capture requires genuine N1 prefill before cached B4".into(),
        ));
    }
    let mut max_base = 0;
    for &(call, prompt_tokens, actual_capture) in calls {
        let rows = call.rows();
        if actual_capture != capture
            || prompt_tokens != 47
            || rows.source() != first.rows().source()
            || rows.capacity() != 512
            || rows.vocabulary() != 248320
            || call.mode() != first.mode()
            || rows.len() != logical_rows
            || !matches!((rows.len(), rows.base()), (47, 0) | (1, 47..=53))
        {
            return Err(Error::Unsupported(
                "B4 original observed source/capture/rows".into(),
            ));
        }
        max_base = max_base.max(rows.base());
    }
    Ok(CohortInputGeometry {
        capture,
        initial,
        logical_rows,
        physical_rows: logical_rows * calls.len(),
        // Maximum key length selects the FA packet; each original row issuer stays intact.
        native_call: if initial { 0 } else { max_base - 46 },
    })
}

pub(super) fn call(inputs: &[Arc<TextCallInput>]) -> Result<usize> {
    Ok(super::input_geometry(inputs)?.native_call)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_core::accelerator::program::{SourceRows, TextCallGeometry};

    fn capture(wide: bool) -> TextKernelCapture {
        TextKernelCapture::bind(if wide {
            "0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0"
        } else {
            "39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d"
        })
        .unwrap()
    }
    fn issued(
        source: SourceIdentity,
        base: usize,
        ids: &[u32],
        mode: SourceCallMode,
    ) -> SourceCall {
        SourceCall::new(
            SourceRows::new(
                TextCallGeometry::new(source, 248320, 512).unwrap(),
                base,
                Arc::from(ids),
            )
            .unwrap(),
            mode,
        )
        .unwrap()
    }
    fn cached(source: SourceIdentity, base: usize, id: u32) -> SourceCall {
        issued(source, base, &[id], SourceCallMode::Decode)
    }
    fn inspect_capture(calls: &[SourceCall; 4], capture: TextKernelCapture) -> CohortInputGeometry {
        geometry(&calls.each_ref().map(|c| (c, 47, capture))).unwrap()
    }
    fn inspect(calls: &[SourceCall; 4]) -> CohortInputGeometry {
        inspect_capture(calls, capture(false))
    }

    #[test]
    fn cached2b_catalogue_cannot_mint_initial_or_wrong_source_geometry() {
        let definitions = super::super::generated_2b::definitions();
        assert_eq!(definitions.len(), 105);
        assert_eq!(super::super::generated_2b::DEFINITION_COUNT, 105);
        assert_eq!(
            super::super::CohortNativeLibrary::host_bytes(capture(true)),
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
                capture: capture(false),
                initial: false,
                logical_rows: 1,
                physical_rows: 4,
                native_call: 4,
            }
        );
        for (actual, original) in calls.iter().zip(&originals) {
            assert_eq!(actual, original);
            assert_eq!(actual.rows(), original.rows());
            assert_eq!(actual.rows().ids(), original.rows().ids());
            assert_eq!(actual.rows().source(), original.rows().source());
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
        let cached_calls: [SourceCall; 7] =
            std::array::from_fn(|index| cached(source, 47 + index, 11 + index as u32));
        for a in 0..7 {
            for b in 0..7 {
                for c in 0..7 {
                    for d in 0..7 {
                        let shape = geometry(
                            &[
                                &cached_calls[a],
                                &cached_calls[b],
                                &cached_calls[c],
                                &cached_calls[d],
                            ]
                            .map(|c| (c, 47, capture(false))),
                        )
                        .unwrap();
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
        let prefill = issued(source, 0, &[11; 47], SourceCallMode::Prefill);
        assert_eq!(
            geometry(&[(&prefill, 47, capture(false)); 4]).unwrap(),
            CohortInputGeometry {
                capture: capture(false),
                initial: true,
                logical_rows: 47,
                physical_rows: 188,
                native_call: 0,
            }
        );
    }

    #[test]
    fn two_b_cached_phase_keeps_own_capture_and_never_grants_initial_b4() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let calls: [SourceCall; 4] =
            std::array::from_fn(|row| cached(source, 47 + row, 11 + row as u32));
        let originals = calls.clone();
        let shape = inspect_capture(&calls, capture(true));
        assert_eq!(shape.capture, capture(true));
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
        let initial = issued(source, 0, &[11; 47], SourceCallMode::Prefill);
        assert!(geometry(&[(&initial, 47, capture(true)); 4]).is_err());
        assert!(
            geometry(&[
                (&calls[0], 47, capture(true)),
                (&calls[1], 47, capture(true)),
                (&calls[2], 47, capture(true)),
                (&calls[3], 47, capture(false))
            ])
            .is_err()
        );
        assert_eq!(
            super::super::CohortPhases::for_capture(capture(true))
                .iter()
                .collect::<Vec<_>>(),
            [false]
        );
        assert_eq!(
            super::super::CohortPhases::for_capture(capture(false))
                .iter()
                .collect::<Vec<_>>(),
            [true, false]
        );
    }

    #[test]
    fn incompatible_source_capture_phase_prompt_or_population_stays_closed() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let good = cached(source, 47, 11);
        let foreign = cached(SourceIdentity::for_new_allocation().unwrap(), 50, 42);
        let workload = cached(source, 57, 11);
        let prefill = issued(source, 0, &[11; 47], SourceCallMode::Prefill);
        for incompatible in [&foreign, &workload, &prefill] {
            assert!(
                geometry(&[&good, &good, &good, incompatible].map(|c| (c, 47, capture(false))))
                    .is_err()
            );
        }
        assert!(
            geometry(&[
                (&good, 47, capture(false)),
                (&good, 47, capture(false)),
                (&good, 47, capture(false)),
                (&good, 47, capture(true))
            ])
            .is_err()
        );
        assert!(geometry(&[(&good, 57, capture(false)); 4]).is_err());
        for width in [0, 1, 3, 5, 8] {
            assert!(geometry(&vec![(&good, 47, capture(false)); width]).is_err());
        }
    }
}
